// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
namespace {
struct TemplateArity {
  int minArgs = 0;
  int maxArgs = 0;
  bool hasParameterPack = false;
};

auto isPackParameter(TemplateParameterAST* parameter) -> bool {
  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
    return typeParameter->isPack;
  }

  if (auto nonTypeParameter =
          ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    return nonTypeParameter->declaration &&
           nonTypeParameter->declaration->isPack;
  }

  if (auto templateTypeParameter =
          ast_cast<TemplateTypeParameterAST>(parameter)) {
    return templateTypeParameter->isPack;
  }

  if (auto constraintParameter =
          ast_cast<ConstraintTypeParameterAST>(parameter)) {
    return static_cast<bool>(constraintParameter->ellipsisLoc);
  }

  return false;
}

auto hasDefaultTemplateArgument(TemplateParameterAST* parameter) -> bool {
  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
    return typeParameter->typeId && typeParameter->typeId->type;
  }

  if (auto nonTypeParameter =
          ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    return nonTypeParameter->declaration &&
           nonTypeParameter->declaration->equalLoc &&
           nonTypeParameter->declaration->expression;
  }

  if (auto templateTypeParameter =
          ast_cast<TemplateTypeParameterAST>(parameter)) {
    return templateTypeParameter->idExpression;
  }

  if (auto constraintParameter =
          ast_cast<ConstraintTypeParameterAST>(parameter)) {
    return constraintParameter->typeId && constraintParameter->typeId->type;
  }

  return false;
}

auto computeTemplateArity(TemplateDeclarationAST* templateDecl)
    -> TemplateArity {
  TemplateArity arity;
  if (!templateDecl) return arity;

  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    ++arity.maxArgs;

    if (isPackParameter(parameter)) {
      arity.hasParameterPack = true;
      continue;
    }

    if (!hasDefaultTemplateArgument(parameter)) {
      ++arity.minArgs;
    }
  }

  return arity;
}

auto templateArgumentCount(List<TemplateArgumentAST*>* templateArgumentList)
    -> int {
  int count = 0;
  for (auto argument : ListView{templateArgumentList}) {
    (void)argument;
    ++count;
  }
  return count;
}

auto isTemplateArityMatch(TemplateDeclarationAST* templateDecl,
                          List<TemplateArgumentAST*>* templateArgumentList)
    -> bool {
  if (!templateDecl) return true;

  auto arity = computeTemplateArity(templateDecl);
  auto argc = templateArgumentCount(templateArgumentList);

  if (argc < arity.minArgs) return false;
  if (!arity.hasParameterPack && argc > arity.maxArgs) return false;

  return true;
}
}  // namespace

struct [[nodiscard]] Binder::ResolveUnqualifiedId {
  Binder& binder;
  NestedNameSpecifierAST* nestedNameSpecifier;
  UnqualifiedIdAST* unqualifiedId;
  bool checkTemplates;
  Symbol* resolvedType = nullptr;

  auto control() const -> Control* { return binder.control(); }
  auto inTemplate() const -> bool { return binder.inTemplate(); }

  auto shouldKeepTemplateIdAsDependent(SimpleTemplateIdAST* templateId) const
      -> bool;
  auto resolveClassTemplateId(SimpleTemplateIdAST* templateId,
                              ClassSymbol* classSymbol) -> Symbol*;
  auto resolveTypeAliasTemplateId(SimpleTemplateIdAST* templateId,
                                  TypeAliasSymbol* typeAliasSymbol) -> Symbol*;
  auto resolveBuiltinTemplate(SimpleTemplateIdAST* templateId,
                              BuiltinTemplateKind kind) -> Symbol*;
  auto resolveBuiltinMakeIntegerSeq(SimpleTemplateIdAST* templateId) -> Symbol*;
  auto resolveBuiltinTypePackElement(SimpleTemplateIdAST* templateId)
      -> Symbol*;
  auto resolveBuiltinCommonType(SimpleTemplateIdAST* templateId) -> Symbol*;

  auto operator()(SimpleTemplateIdAST* templateId) -> Symbol*;

  auto operator()(NameIdAST* nameId) -> Symbol*;

  auto operator()(UnqualifiedIdAST*) -> Symbol* {
    binder.error(
        unqualifiedId->firstSourceLocation(),
        "unable to resolve unqualified-id: not a NameId or SimpleTemplateId");
    return nullptr;
  }
};

auto Binder::resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                     UnqualifiedIdAST* unqualifiedId, bool checkTemplates,
                     Symbol* resolvedType) -> Symbol* {
  return visit(ResolveUnqualifiedId{*this, nestedNameSpecifier, unqualifiedId,
                                    checkTemplates, resolvedType},
               unqualifiedId);
}

auto Binder::ResolveUnqualifiedId::shouldKeepTemplateIdAsDependent(
    SimpleTemplateIdAST* templateId) const -> bool {
  if (!inTemplate()) return false;
  if (symbol_cast<TemplateTypeParameterSymbol>(templateId->symbol)) {
    return true;
  }
  return hasDependentTemplateArguments(binder.unit_, templateId);
}

auto Binder::ResolveUnqualifiedId::resolveClassTemplateId(
    SimpleTemplateIdAST* templateId, ClassSymbol* classSymbol) -> Symbol* {
  if (!isTemplateArityMatch(classSymbol->templateDeclaration(),
                            templateId->templateArgumentList)) {
    return nullptr;
  }

  if (!classSymbol->templateDeclaration()) return nullptr;

  auto subst =
      Substitution::make(binder.unit_, classSymbol->templateDeclaration(),
                         templateId->templateArgumentList);

  if (!subst) return nullptr;

  auto templateArgs = std::move(*subst).templateArguments();

  if (auto cached = classSymbol->findSpecialization(templateArgs)) {
    return cached;
  }

  auto parentScope = classSymbol->enclosingNonTemplateParametersScope();
  auto spec = control()->newClassSymbol(parentScope, {});
  spec->setName(classSymbol->name());
  spec->setType(control()->getClassType(spec));
  classSymbol->addSpecialization(std::move(templateArgs), spec);

  for (auto& s : classSymbol->mutableSpecializations()) {
    if (s.symbol == spec) {
      s.pendingArgumentList = templateId->templateArgumentList;
      s.pendingInstantiationLoc = templateId->identifierLoc;
      s.isPendingInstantiation = true;
      break;
    }
  }

  return spec;
}

auto Binder::ResolveUnqualifiedId::resolveTypeAliasTemplateId(
    SimpleTemplateIdAST* templateId, TypeAliasSymbol* typeAliasSymbol)
    -> Symbol* {
  if (!isTemplateArityMatch(typeAliasSymbol->templateDeclaration(),
                            templateId->templateArgumentList)) {
    return nullptr;
  }

  return ASTRewriter::instantiate(binder.unit_,
                                  templateId->templateArgumentList,
                                  typeAliasSymbol, templateId->identifierLoc);
}

auto Binder::ResolveUnqualifiedId::resolveBuiltinMakeIntegerSeq(
    SimpleTemplateIdAST* templateId) -> Symbol* {
  auto ar = binder.unit_->arena();

  std::vector<TemplateArgumentAST*> args;
  for (auto arg : ListView{templateId->templateArgumentList}) {
    args.push_back(arg);
  }
  if (args.size() != 3) return nullptr;

  auto seqArg = ast_cast<TypeTemplateArgumentAST>(args[0]);
  if (!seqArg || !seqArg->typeId) return nullptr;
  auto seqType = type_cast<ClassType>(seqArg->typeId->type);
  if (!seqType) return nullptr;
  auto seqClass = seqType->symbol();
  if (!seqClass || !seqClass->templateDeclaration()) return nullptr;

  auto typeArg = ast_cast<TypeTemplateArgumentAST>(args[1]);
  if (!typeArg || !typeArg->typeId || !typeArg->typeId->type) return nullptr;
  auto elementType = typeArg->typeId->type;

  auto countArg = ast_cast<ExpressionTemplateArgumentAST>(args[2]);
  if (!countArg || !countArg->expression) return nullptr;

  auto interp = ASTInterpreter{binder.unit_};
  auto value = interp.evaluate(countArg->expression);
  if (!value.has_value()) return nullptr;
  auto intVal = interp.toInt(*value);
  if (!intVal.has_value()) return nullptr;
  auto N = *intVal;
  if (N < 0) return nullptr;

  auto expanded = SimpleTemplateIdAST::create(ar);
  expanded->identifier = templateId->identifier;
  expanded->symbol = seqClass;

  List<TemplateArgumentAST*>** it = &expanded->templateArgumentList;

  auto expandedTypeArg = TypeTemplateArgumentAST::create(ar);
  expandedTypeArg->typeId = typeArg->typeId;
  *it = make_list_node(ar, static_cast<TemplateArgumentAST*>(expandedTypeArg));
  it = &(*it)->next;

  for (std::intmax_t i = 0; i < N; ++i) {
    std::string spelling = std::format("{}", i);
    auto literal = control()->integerLiteral(spelling);
    auto intExpr = IntLiteralExpressionAST::create(
        ar, literal, ValueCategory::kPrValue, elementType);
    auto exprArg = ExpressionTemplateArgumentAST::create(ar, intExpr);
    *it = make_list_node(ar, static_cast<TemplateArgumentAST*>(exprArg));
    it = &(*it)->next;
  }

  return resolveClassTemplateId(expanded, seqClass);
}

auto Binder::ResolveUnqualifiedId::resolveBuiltinTypePackElement(
    SimpleTemplateIdAST* templateId) -> Symbol* {
  auto ar = binder.unit_->arena();

  std::vector<TemplateArgumentAST*> args;
  for (auto arg : ListView{templateId->templateArgumentList}) {
    args.push_back(arg);
  }
  if (args.size() < 2) return nullptr;

  auto indexArg = ast_cast<ExpressionTemplateArgumentAST>(args[0]);
  if (!indexArg || !indexArg->expression) return nullptr;

  auto interp = ASTInterpreter{binder.unit_};
  auto value = interp.evaluate(indexArg->expression);
  if (!value.has_value()) return nullptr;
  auto intVal = interp.toInt(*value);
  if (!intVal.has_value()) return nullptr;
  auto N = *intVal;

  auto packSize = static_cast<std::intmax_t>(args.size() - 1);
  if (N < 0 || N >= packSize) return nullptr;

  auto typeArg = ast_cast<TypeTemplateArgumentAST>(args[1 + N]);
  if (!typeArg || !typeArg->typeId) return nullptr;

  auto alias = control()->newTypeAliasSymbol(nullptr, {});
  alias->setName(templateId->identifier);
  alias->setType(typeArg->typeId->type);
  return alias;
}

auto Binder::ResolveUnqualifiedId::resolveBuiltinCommonType(
    SimpleTemplateIdAST* templateId) -> Symbol* {
  auto traits = binder.traits;

  std::vector<TemplateArgumentAST*> args;
  for (auto arg : ListView{templateId->templateArgumentList})
    args.push_back(arg);

  if (args.size() < 3) return nullptr;

  auto identityArg = ast_cast<TypeTemplateArgumentAST>(args[1]);
  auto emptyArg = ast_cast<TypeTemplateArgumentAST>(args[2]);
  if (!identityArg || !identityArg->typeId) return nullptr;
  if (!emptyArg || !emptyArg->typeId) return nullptr;

  auto identityType = type_cast<ClassType>(identityArg->typeId->type);
  if (!identityType) return nullptr;
  auto identityClass = symbol_cast<ClassSymbol>(identityType->symbol());
  if (!identityClass || !identityClass->templateDeclaration()) return nullptr;

  std::vector<const Type*> operands;
  for (std::size_t i = 3; i < args.size(); ++i) {
    auto typeArg = ast_cast<TypeTemplateArgumentAST>(args[i]);
    if (!typeArg || !typeArg->typeId || !typeArg->typeId->type) return nullptr;
    operands.push_back(typeArg->typeId->type);
  }

  auto resolveToEmpty = [&]() -> Symbol* {
    if (auto emptyClass = type_cast<ClassType>(emptyArg->typeId->type))
      return emptyClass->symbol();
    return nullptr;
  };

  if (operands.empty()) return resolveToEmpty();

  const Type* result = traits.remove_cvref(operands[0]);
  for (std::size_t i = 1; i < operands.size(); ++i) {
    auto next = traits.remove_cvref(operands[i]);
    if (traits.is_same(result, next)) continue;
    auto combined =
        StandardConversion{binder.unit_}.commonArithmeticType(result, next);
    if (!combined) return resolveToEmpty();
    result = combined;
  }

  auto ar = binder.unit_->arena();
  auto identityId = SimpleTemplateIdAST::create(ar);
  identityId->identifier = identityClass->name()
                               ? name_cast<Identifier>(identityClass->name())
                               : nullptr;
  identityId->symbol = identityClass;

  auto typeId = TypeIdAST::create(ar);
  typeId->type = result;
  auto typeArg = TypeTemplateArgumentAST::create(ar);
  typeArg->typeId = typeId;
  identityId->templateArgumentList =
      make_list_node<TemplateArgumentAST>(ar, typeArg);

  return resolveClassTemplateId(identityId, identityClass);
}

auto Binder::ResolveUnqualifiedId::resolveBuiltinTemplate(
    SimpleTemplateIdAST* templateId, BuiltinTemplateKind kind) -> Symbol* {
  switch (kind) {
    case BuiltinTemplateKind::T___MAKE_INTEGER_SEQ:
      return resolveBuiltinMakeIntegerSeq(templateId);
    case BuiltinTemplateKind::T___TYPE_PACK_ELEMENT:
      return resolveBuiltinTypePackElement(templateId);
    case BuiltinTemplateKind::T___BUILTIN_COMMON_TYPE:
      return resolveBuiltinCommonType(templateId);
    default:
      return nullptr;
  }
}

auto Binder::ResolveUnqualifiedId::operator()(SimpleTemplateIdAST* templateId)
    -> Symbol* {
  if (!checkTemplates) return templateId->symbol;

  if (!templateId->symbol && templateId->identifier) {
    auto builtinKind = templateId->identifier->builtinTemplate();
    if (builtinKind != BuiltinTemplateKind::T_NONE) {
      if (inTemplate() &&
          hasDependentTemplateArguments(binder.unit_, templateId)) {
        auto placeholder = control()->newTypeAliasSymbol(nullptr, {});
        placeholder->setName(templateId->identifier);
        return placeholder;
      }
      return resolveBuiltinTemplate(templateId, builtinKind);
    }
  }

  if (shouldKeepTemplateIdAsDependent(templateId)) return templateId->symbol;

  auto resolvedSymbol = templateId->symbol;
  if (auto injected = symbol_cast<InjectedClassNameSymbol>(resolvedSymbol)) {
    resolvedSymbol = injected->classSymbol();
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(resolvedSymbol)) {
    return resolveClassTemplateId(templateId, classSymbol);
  }

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(resolvedSymbol)) {
    return resolveTypeAliasTemplateId(templateId, typeAliasSymbol);
  }

  return templateId->symbol;
}

auto Binder::ResolveUnqualifiedId::operator()(NameIdAST* nameId) -> Symbol* {
  Symbol* symbol = nullptr;
  if (nestedNameSpecifier && nestedNameSpecifier->symbol)
    symbol =
        qualifiedLookupType(nestedNameSpecifier->symbol, nameId->identifier);
  else {
    symbol = resolvedType;
  }

  if (!is_type(symbol)) return nullptr;

  if (auto injected = symbol_cast<InjectedClassNameSymbol>(symbol)) {
    if (auto cls = injected->classSymbol()) return cls;
  }

  return symbol;
}
}  // namespace cxx
