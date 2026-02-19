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

#include <cxx/binder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {

namespace {

auto isDependentTypeParameterSymbol(Symbol* symbol) -> bool {
  return symbol_cast<TypeParameterSymbol>(symbol) ||
         symbol_cast<TemplateTypeParameterSymbol>(symbol);
}

auto lookupDependentTypeParameterInScopeChain(Control* control,
                                              ScopeSymbol* scope,
                                              const Identifier* identifier)
    -> Symbol* {
  if (!control || !scope || !identifier) return nullptr;

  for (auto current = scope; current; current = current->parent()) {
    auto candidate = lookupType(current, nullptr, identifier);
    if (isDependentTypeParameterSymbol(candidate)) return candidate;
  }

  return nullptr;
}

auto isDependentNestedNameSpecifier(Control* control, ScopeSymbol* scope,
                                    NestedNameSpecifierAST* ast) -> bool {
  if (!scope || !ast) return false;
  if (ast->symbol) return false;

  auto simple = ast_cast<SimpleNestedNameSpecifierAST>(ast);
  if (!simple || !simple->identifier) return false;

  if (isDependentNestedNameSpecifier(control, scope,
                                     simple->nestedNameSpecifier)) {
    return true;
  }

  if (lookupDependentTypeParameterInScopeChain(control, scope,
                                               simple->identifier)) {
    return true;
  }

  auto symbol =
      lookupType(scope, simple->nestedNameSpecifier, simple->identifier);
  return isDependentTypeParameterSymbol(symbol);
}

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

  auto control() const -> Control* { return binder.control(); }
  auto inTemplate() const -> bool { return binder.inTemplate_; }

  auto shouldKeepTemplateIdAsDependent(SimpleTemplateIdAST* templateId) const
      -> bool;
  auto hasDependentTemplateArguments(SimpleTemplateIdAST* templateId) const
      -> bool;
  auto isDependentTypeArgument(TypeTemplateArgumentAST* typeArg) const -> bool;
  auto isDependentExpressionArgument(
      ExpressionTemplateArgumentAST* expressionArg) const -> bool;
  auto resolveClassTemplateId(SimpleTemplateIdAST* templateId,
                              ClassSymbol* classSymbol) -> Symbol*;
  auto resolveTypeAliasTemplateId(SimpleTemplateIdAST* templateId,
                                  TypeAliasSymbol* typeAliasSymbol) -> Symbol*;
  auto resolveBuiltinTemplate(SimpleTemplateIdAST* templateId,
                              BuiltinTemplateKind kind) -> Symbol*;
  auto resolveBuiltinMakeIntegerSeq(SimpleTemplateIdAST* templateId) -> Symbol*;
  auto resolveBuiltinTypePackElement(SimpleTemplateIdAST* templateId)
      -> Symbol*;

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
                     UnqualifiedIdAST* unqualifiedId, bool checkTemplates)
    -> Symbol* {
  return visit(ResolveUnqualifiedId{*this, nestedNameSpecifier, unqualifiedId,
                                    checkTemplates},
               unqualifiedId);
}

auto Binder::ResolveUnqualifiedId::isDependentTypeArgument(
    TypeTemplateArgumentAST* typeArg) const -> bool {
  if (!typeArg || !typeArg->typeId) return false;
  auto unit = binder.unit_;
  if (isDependent(unit, typeArg->typeId->type)) return true;

  for (auto spec : ListView{typeArg->typeId->typeSpecifierList}) {
    // Handle decltype(dependent_expr): the expression may reference
    // template parameters making the decltype dependent even when
    // ast->type is null (unresolved).
    if (isDependent(unit, spec)) return true;

    auto named = ast_cast<NamedTypeSpecifierAST>(spec);
    if (!named) continue;

    if (isDependentTypeParameterSymbol(named->symbol)) return true;
    // A type alias with null/dependent type is unresolved (dependent on
    // template parameters that couldn't be substituted yet).
    if (auto alias = symbol_cast<TypeAliasSymbol>(named->symbol)) {
      if (!alias->type()) return true;
      if (isDependent(unit, alias->type())) return true;
    }
    if (isDependentNestedNameSpecifier(binder.control(), binder.scope(),
                                       named->nestedNameSpecifier)) {
      return true;
    }

    // Check if the unqualified-id is a template-id with dependent arguments,
    // e.g. __iter_diff_t<_InIter> where _InIter is a template type parameter.
    if (auto innerTemplateId =
            ast_cast<SimpleTemplateIdAST>(named->unqualifiedId)) {
      if (hasDependentTemplateArguments(innerTemplateId)) return true;
    }
  }

  return false;
}

auto Binder::ResolveUnqualifiedId::isDependentExpressionArgument(
    ExpressionTemplateArgumentAST* expressionArg) const -> bool {
  if (!expressionArg) return false;
  return isDependent(binder.unit_, expressionArg->expression);
}

auto Binder::ResolveUnqualifiedId::hasDependentTemplateArguments(
    SimpleTemplateIdAST* templateId) const -> bool {
  for (auto arg : ListView{templateId->templateArgumentList}) {
    if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
      if (isDependentTypeArgument(typeArg)) return true;
      continue;
    }

    auto expressionArg = ast_cast<ExpressionTemplateArgumentAST>(arg);
    if (isDependentExpressionArgument(expressionArg)) return true;
  }

  return false;
}

auto Binder::ResolveUnqualifiedId::shouldKeepTemplateIdAsDependent(
    SimpleTemplateIdAST* templateId) const -> bool {
  if (!inTemplate()) return false;
  if (symbol_cast<TemplateTypeParameterSymbol>(templateId->symbol)) {
    return true;
  }
  return hasDependentTemplateArguments(templateId);
}

auto Binder::ResolveUnqualifiedId::resolveClassTemplateId(
    SimpleTemplateIdAST* templateId, ClassSymbol* classSymbol) -> Symbol* {
  if (!isTemplateArityMatch(classSymbol->templateDeclaration(),
                            templateId->templateArgumentList)) {
    return nullptr;
  }

  auto instance = ASTRewriter::instantiate(
      binder.unit_, templateId->templateArgumentList, classSymbol);

  if (instance) return instance;
  if (!classSymbol->templateDeclaration()) return instance;

  auto templateArgs =
      Substitution(binder.unit_, classSymbol->templateDeclaration(),
                   templateId->templateArgumentList)
          .templateArguments();

  if (templateArgs.empty()) return instance;

  if (auto cached = classSymbol->findSpecialization(templateArgs)) {
    return cached;
  }

  auto parentScope = classSymbol->enclosingNonTemplateParametersScope();
  auto spec = control()->newClassSymbol(parentScope, {});
  spec->setName(classSymbol->name());
  classSymbol->addSpecialization(std::move(templateArgs), spec);
  return spec;
}

auto Binder::ResolveUnqualifiedId::resolveTypeAliasTemplateId(
    SimpleTemplateIdAST* templateId, TypeAliasSymbol* typeAliasSymbol)
    -> Symbol* {
  if (!isTemplateArityMatch(typeAliasSymbol->templateDeclaration(),
                            templateId->templateArgumentList)) {
    return nullptr;
  }

  return ASTRewriter::instantiate(
      binder.unit_, templateId->templateArgumentList, typeAliasSymbol);
}

auto Binder::ResolveUnqualifiedId::resolveBuiltinMakeIntegerSeq(
    SimpleTemplateIdAST* templateId) -> Symbol* {
  // __make_integer_seq<Seq, T, N> expands to Seq<T, 0, 1, ..., N-1>
  auto ar = binder.unit_->arena();

  // Collect template arguments: expect exactly 3
  std::vector<TemplateArgumentAST*> args;
  for (auto arg : ListView{templateId->templateArgumentList}) {
    args.push_back(arg);
  }
  if (args.size() != 3) return nullptr;

  // First arg: Seq — a class template (TypeTemplateArgument with ClassType)
  auto seqArg = ast_cast<TypeTemplateArgumentAST>(args[0]);
  if (!seqArg || !seqArg->typeId) return nullptr;
  auto seqType = type_cast<ClassType>(seqArg->typeId->type);
  if (!seqType) return nullptr;
  auto seqClass = seqType->symbol();
  if (!seqClass || !seqClass->templateDeclaration()) return nullptr;

  // Second arg: T — the element type
  auto typeArg = ast_cast<TypeTemplateArgumentAST>(args[1]);
  if (!typeArg || !typeArg->typeId || !typeArg->typeId->type) return nullptr;
  auto elementType = typeArg->typeId->type;

  // Third arg: N — an integer constant expression
  auto countArg = ast_cast<ExpressionTemplateArgumentAST>(args[2]);
  if (!countArg || !countArg->expression) return nullptr;

  auto interp = ASTInterpreter{binder.unit_};
  auto value = interp.evaluate(countArg->expression);
  if (!value.has_value()) return nullptr;
  auto intVal = interp.toInt(*value);
  if (!intVal.has_value()) return nullptr;
  auto N = *intVal;
  if (N < 0) return nullptr;

  // Build: Seq<T, 0, 1, ..., N-1>
  auto expanded = SimpleTemplateIdAST::create(ar);
  expanded->identifier = templateId->identifier;
  expanded->symbol = seqClass;

  List<TemplateArgumentAST*>** it = &expanded->templateArgumentList;

  // First expanded arg: the element type T
  auto expandedTypeArg = TypeTemplateArgumentAST::create(ar);
  expandedTypeArg->typeId = typeArg->typeId;
  *it = make_list_node(ar, static_cast<TemplateArgumentAST*>(expandedTypeArg));
  it = &(*it)->next;

  // Remaining args: integer literals 0, 1, ..., N-1
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
  // __type_pack_element<N, T0, T1, ...> expands to T_N
  auto ar = binder.unit_->arena();

  // Collect template arguments: expect at least 2
  std::vector<TemplateArgumentAST*> args;
  for (auto arg : ListView{templateId->templateArgumentList}) {
    args.push_back(arg);
  }
  if (args.size() < 2) return nullptr;

  // First arg: N — an integer constant expression
  auto indexArg = ast_cast<ExpressionTemplateArgumentAST>(args[0]);
  if (!indexArg || !indexArg->expression) return nullptr;

  auto interp = ASTInterpreter{binder.unit_};
  auto value = interp.evaluate(indexArg->expression);
  if (!value.has_value()) return nullptr;
  auto intVal = interp.toInt(*value);
  if (!intVal.has_value()) return nullptr;
  auto N = *intVal;

  // Validate: N must be in range [0, sizeof...(Types))
  auto packSize = static_cast<std::intmax_t>(args.size() - 1);
  if (N < 0 || N >= packSize) return nullptr;

  // Extract the N-th type argument (1-based in arguments list)
  auto typeArg = ast_cast<TypeTemplateArgumentAST>(args[1 + N]);
  if (!typeArg || !typeArg->typeId) return nullptr;

  // Return a type alias pointing to the selected type
  auto alias = control()->newTypeAliasSymbol(nullptr, {});
  alias->setName(templateId->identifier);
  alias->setType(typeArg->typeId->type);
  return alias;
}

auto Binder::ResolveUnqualifiedId::resolveBuiltinTemplate(
    SimpleTemplateIdAST* templateId, BuiltinTemplateKind kind) -> Symbol* {
  switch (kind) {
    case BuiltinTemplateKind::T___MAKE_INTEGER_SEQ:
      return resolveBuiltinMakeIntegerSeq(templateId);
    case BuiltinTemplateKind::T___TYPE_PACK_ELEMENT:
      return resolveBuiltinTypePackElement(templateId);
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
      if (inTemplate() && hasDependentTemplateArguments(templateId)) {
        auto placeholder = control()->newTypeAliasSymbol(nullptr, {});
        placeholder->setName(templateId->identifier);
        return placeholder;
      }
      return resolveBuiltinTemplate(templateId, builtinKind);
    }
  }

  if (shouldKeepTemplateIdAsDependent(templateId)) return templateId->symbol;

  if (auto classSymbol = symbol_cast<ClassSymbol>(templateId->symbol)) {
    return resolveClassTemplateId(templateId, classSymbol);
  }

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(templateId->symbol)) {
    return resolveTypeAliasTemplateId(templateId, typeAliasSymbol);
  }

  return templateId->symbol;
}

auto Binder::ResolveUnqualifiedId::operator()(NameIdAST* nameId) -> Symbol* {
  auto symbol =
      lookupType(binder.scope(), nestedNameSpecifier, nameId->identifier);

  if (!is_type(symbol)) return nullptr;

  return symbol;
}

}  // namespace cxx