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
#include <cxx/binder.h>
#include <cxx/dependent_types.h>
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <format>
#include <set>
#include <span>

namespace cxx {
namespace {
[[nodiscard]] auto enclosing_class_or_namespace(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;
  auto parent = symbol->parent();
  if (!parent || !parent->isClassOrNamespace()) return nullptr;
  return parent;
}

[[nodiscard]] auto is_unmangled_main(FunctionSymbol* function) -> bool {
  auto id = name_cast<Identifier>(function->name());
  return id && id->name() == "main" && is_global_namespace(function->parent());
}

[[nodiscard]] auto has_global_qualifier(NestedNameSpecifierAST* nns) -> bool {
  while (nns) {
    if (ast_cast<GlobalNestedNameSpecifierAST>(nns)) return true;
    if (auto simple = ast_cast<SimpleNestedNameSpecifierAST>(nns)) {
      nns = simple->nestedNameSpecifier;
      continue;
    }
    if (auto templ = ast_cast<TemplateNestedNameSpecifierAST>(nns)) {
      nns = templ->nestedNameSpecifier;
      continue;
    }
    return false;
  }
  return false;
}

[[nodiscard]] auto encodes_return_type(FunctionSymbol* function) -> bool {
  if (!function->isSpecialization()) return false;
  if (function->templateArguments().empty()) return false;
  if (function->isConstructor()) return false;
  if (name_cast<DestructorId>(function->name())) return false;
  if (name_cast<ConversionFunctionId>(function->name())) return false;
  return true;
}

[[nodiscard]] auto mangling_parent(Symbol* symbol) -> Symbol* {
  auto parent = enclosing_class_or_namespace(symbol);

  if (auto function = symbol_cast<FunctionSymbol>(symbol);
      function && function->isFriend()) {
    while (parent && parent->isClass()) {
      parent = enclosing_class_or_namespace(parent);
    }
  }

  return parent;
}

[[nodiscard]] auto isParameterPackExpansion(const Type* type) -> bool {
  while (type) {
    if (auto param = type_cast<TypeParameterType>(type)) {
      return param->isParameterPack();
    }
    if (auto param = type_cast<TemplateTypeParameterType>(type)) {
      return param->isParameterPack();
    }
    if (auto ref = type_cast<LvalueReferenceType>(type)) {
      type = ref->elementType();
    } else if (auto ref = type_cast<RvalueReferenceType>(type)) {
      type = ref->elementType();
    } else if (auto ptr = type_cast<PointerType>(type)) {
      type = ptr->elementType();
    } else if (auto qual = type_cast<QualType>(type)) {
      type = qual->elementType();
    } else {
      return false;
    }
  }
  return false;
}

[[nodiscard]] auto unary_builtin_name(UnaryBuiltinTypeKind kind)
    -> std::string_view {
  switch (kind) {
#define PROCESS_UNARY_BUILTIN(id, name) \
  case UnaryBuiltinTypeKind::T_##id:    \
    return name;
    FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(PROCESS_UNARY_BUILTIN)
#undef PROCESS_UNARY_BUILTIN
    default:
      return {};
  }
}

[[nodiscard]] auto has_complete_signature(const FunctionType* type) -> bool {
  if (!type) return false;
  if (!type->returnType()) return false;
  for (auto param : type->parameterTypes()) {
    if (!param) return false;
  }
  return true;
}

[[nodiscard]] auto signature_type(FunctionSymbol* function)
    -> const FunctionType* {
  if (auto inherited = function->inheritedConstructorOrigin()) {
    if (auto primary = inherited->primaryTemplateSymbol()) {
      if (auto primaryType = type_cast<FunctionType>(primary->type());
          primaryType && has_complete_signature(primaryType)) {
        return primaryType;
      }
    }
    if (auto inheritedType = type_cast<FunctionType>(inherited->type());
        inheritedType && has_complete_signature(inheritedType)) {
      return inheritedType;
    }
  }
  if (function->isSpecialization()) {
    if (auto primary = function->primaryTemplateSymbol()) {
      if (auto primaryType = type_cast<FunctionType>(primary->type());
          primaryType && has_complete_signature(primaryType)) {
        return primaryType;
      }
    }
  }
  return type_cast<FunctionType>(function->type());
}

[[nodiscard]] auto template_name(Symbol* symbol) -> Symbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    if (classSymbol->isSpecialization())
      return classSymbol->primaryTemplateSymbol();
    if (classSymbol->templateParameters()) return classSymbol;
  } else if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    if (functionSymbol->isSpecialization())
      return functionSymbol->primaryTemplateSymbol();
  } else if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
    if (variableSymbol->isSpecialization())
      return variableSymbol->primaryTemplateSymbol();
  }
  return nullptr;
}

[[nodiscard]] auto dependent_prefix_type_param(NestedNameSpecifierAST* nns)
    -> Symbol* {
  auto simple = ast_cast<SimpleNestedNameSpecifierAST>(nns);
  if (!simple) return nullptr;
  if (simple->nestedNameSpecifier) return nullptr;
  if (symbol_cast<TypeParameterSymbol>(simple->symbol) ||
      symbol_cast<TemplateTypeParameterSymbol>(simple->symbol)) {
    return simple->symbol;
  }
  return nullptr;
}

[[nodiscard]] auto isDeclaredExtern(VariableSymbol* variable) -> bool {
  if (variable->isExtern()) return true;
  auto canonical = variable->canonical();
  if (canonical->isExtern()) return true;
  return std::ranges::any_of(
      canonical->redeclarations(),
      [](VariableSymbol* redeclaration) { return redeclaration->isExtern(); });
}

[[nodiscard]] auto isInUnnamedNamespace(Symbol* symbol) -> bool {
  for (auto scope : symbol->enclosingSymbols()) {
    auto ns = symbol_cast<NamespaceSymbol>(scope);
    if (!ns || is_global_namespace(ns)) continue;
    if (!ns->name()) return true;
  }
  return false;
}

[[nodiscard]] auto hasInternalLinkage(Symbol* symbol) -> bool {
  auto parent = symbol->parent();
  if (!parent || !parent->isNamespace()) return false;

  if (isInUnnamedNamespace(symbol)) return false;

  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    return function->isStatic();
  }

  auto variable = symbol_cast<VariableSymbol>(symbol);
  if (!variable) return false;

  if (variable->isStatic()) return true;

  if (variable->isInline() || isDeclaredExtern(variable)) return false;
  if (variable->templateParameters() || variable->isSpecialization())
    return false;

  auto qualType = type_cast<QualType>(variable->type());
  return qualType && qualType->isConst() && !qualType->isVolatile();
}

[[nodiscard]] auto is_std_namespace(Symbol* symbol) -> bool {
  if (!symbol_cast<NamespaceSymbol>(symbol)) return false;

  auto parent = enclosing_class_or_namespace(symbol);
  if (!parent) return false;

  if (!is_global_namespace(parent)) return false;

  auto id = name_cast<Identifier>(symbol->name());
  if (!id) return false;

  if (id->name() != "std") return false;

  return true;
}
}  // namespace

struct ExternalNameEncoder::EncodeType {
  ExternalNameEncoder& encoder;

  auto operator()(const VoidType* type) -> bool {
    encoder.out("v");
    return false;
  }

  auto operator()(const NullptrType* type) -> bool {
    encoder.out("Dn");
    return false;
  }

  auto operator()(const DecltypeAutoType* type) -> bool {
    encoder.out("Dc");
    return false;
  }

  auto operator()(const AutoType* type) -> bool {
    encoder.out("Da");
    return false;
  }

  auto operator()(const BoolType* type) -> bool {
    encoder.out("b");
    return false;
  }

  auto operator()(const SignedCharType* type) -> bool {
    encoder.out("a");
    return false;
  }

  auto operator()(const ShortIntType* type) -> bool {
    encoder.out("s");
    return false;
  }

  auto operator()(const IntType* type) -> bool {
    encoder.out("i");
    return false;
  }

  auto operator()(const LongIntType* type) -> bool {
    encoder.out("l");
    return false;
  }

  auto operator()(const LongLongIntType* type) -> bool {
    encoder.out("x");
    return false;
  }

  auto operator()(const Int128Type* type) -> bool {
    encoder.out("n");
    return false;
  }

  auto operator()(const UnsignedCharType* type) -> bool {
    encoder.out("h");
    return false;
  }

  auto operator()(const UnsignedShortIntType* type) -> bool {
    encoder.out("t");
    return false;
  }

  auto operator()(const UnsignedIntType* type) -> bool {
    encoder.out("j");
    return false;
  }

  auto operator()(const UnsignedLongIntType* type) -> bool {
    encoder.out("m");
    return false;
  }

  auto operator()(const UnsignedLongLongIntType* type) -> bool {
    encoder.out("y");
    return false;
  }

  auto operator()(const UnsignedInt128Type* type) -> bool {
    encoder.out("o");
    return false;
  }

  auto operator()(const CharType* type) -> bool {
    encoder.out("c");
    return false;
  }

  auto operator()(const Char8Type* type) -> bool {
    encoder.out("Du");
    return false;
  }

  auto operator()(const Char16Type* type) -> bool {
    encoder.out("Ds");
    return false;
  }

  auto operator()(const Char32Type* type) -> bool {
    encoder.out("Di");
    return false;
  }

  auto operator()(const WideCharType* type) -> bool {
    encoder.out("w");
    return false;
  }

  auto operator()(const FloatType* type) -> bool {
    encoder.out("f");
    return false;
  }

  auto operator()(const DoubleType* type) -> bool {
    encoder.out("d");
    return false;
  }

  auto operator()(const LongDoubleType* type) -> bool {
    encoder.out("e");
    return false;
  }

  auto operator()(const Float16Type* type) -> bool {
    encoder.out("DF16_");
    return false;
  }

  auto operator()(const QualType* type) -> bool {
    if (type->isVolatile()) encoder.out("V");
    if (type->isConst()) encoder.out("K");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const BoundedArrayType* type) -> bool {
    encoder.out(std::format("A{}_", type->size()));
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const UnboundedArrayType* type) -> bool {
    encoder.out("A_");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const PointerType* type) -> bool {
    encoder.out("P");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const LvalueReferenceType* type) -> bool {
    encoder.out("R");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const RvalueReferenceType* type) -> bool {
    encoder.out("O");
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const FunctionType* type) -> bool {
    if (is_volatile(type->cvQualifiers())) encoder.out("V");
    if (is_const(type->cvQualifiers())) encoder.out("K");

    if (type->isNoexcept()) encoder.out("Do");

    encoder.out("F");

    encoder.encodeBareFunctionType(type, /*includeReturnType=*/true);

    if (type->refQualifier() == RefQualifier::kLvalue)
      encoder.out("R");
    else if (type->refQualifier() == RefQualifier::kRvalue)
      encoder.out("O");

    encoder.out("E");
    return true;
  }

  auto operator()(const ClassType* type) -> bool {
    if (!type->symbol()->name()) {
      cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
      return false;
    }

    encoder.encodeName(type->symbol());
    return true;
  }

  auto operator()(const EnumType* type) -> bool {
    if (!type->symbol()->name()) {
      cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
      return false;
    }
    encoder.encodeName(type->symbol());
    return true;
  }

  auto operator()(const ScopedEnumType* type) -> bool {
    if (!type->symbol()->name()) {
      cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
      return false;
    }
    encoder.encodeName(type->symbol());
    return true;
  }

  auto operator()(const MemberObjectPointerType* type) -> bool {
    encoder.out("M");
    encoder.encodeType(type->classType());
    encoder.encodeType(type->elementType());
    return true;
  }

  auto operator()(const MemberFunctionPointerType* type) -> bool {
    encoder.out("M");
    encoder.encodeType(type->classType());
    encoder.encodeType(type->functionType());
    return true;
  }

  auto operator()(const NamespaceType* type) -> bool { return false; }

  auto operator()(const TypeParameterType* type) -> bool {
    encoder.encodeTemplateParamValue(type->index());
    return true;
  }

  auto operator()(const TemplateTypeParameterType* type) -> bool {
    encoder.encodeTemplateParamValue(type->index());
    return true;
  }

  auto operator()(const UnresolvedNameType* type) -> bool {
    if (encoder.encodeDependentName(type->nestedNameSpecifier(),
                                    type->unqualifiedId())) {
      return true;
    }
    cxx_runtime_error(std::format("cannot mangle unresolved name type '{}'",
                                  to_string(type)));
  }

  auto operator()(const UnresolvedBoundedArrayType* type) -> bool {
    encoder.out("A0_");
    return true;
  }

  auto operator()(const UnresolvedUnderlyingType* type) -> bool {
    cxx_runtime_error(std::format(
        "cannot mangle unresolved underlying type '{}'", to_string(type)));
  }

  auto operator()(const UnresolvedBuiltinType* type) -> bool {
    auto name = unary_builtin_name(type->builtinKind());
    auto typeId = type->typeId();
    if (name.empty() || !typeId || !typeId->type) {
      cxx_runtime_error("cannot mangle unresolved builtin type");
    }
    encoder.out(std::format("u{}{}I", name.size(), name));
    encoder.encodeType(typeId->type);
    encoder.out("E");
    return true;
  }

  auto operator()(const OverloadSetType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const BuiltinVaListType* type) -> bool {
    encoder.out("Pc");
    return true;
  }

  auto operator()(const BuiltinMetaInfoType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return true;
  }

  auto operator()(const BitIntType* type) -> bool {
    encoder.out(std::format("DB{}_", type->numBits()));
    return false;
  }

  auto operator()(const UnsignedBitIntType* type) -> bool {
    encoder.out(std::format("DU{}_", type->numBits()));
    return false;
  }

  auto operator()(const UnresolvedBitIntType* type) -> bool {
    cxx_runtime_error(std::format("cannot mangle unresolved bit-int type '{}'",
                                  to_string(type)));
  }
};

struct ExternalNameEncoder::EncodeUnqualifiedName {
  ExternalNameEncoder& encoder;
  Symbol* symbol = nullptr;

  void encodeTemplateArguments(Symbol* symbol) {
    if (!symbol) return;
    if (symbol == encoder.templateNameOnly_) return;

    std::span<const TemplateArgument> args;
    Symbol* templateName = nullptr;

    if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
      args = classSymbol->templateArguments();
      if (classSymbol->isSpecialization()) {
        templateName = classSymbol->primaryTemplateSymbol();
      } else if (classSymbol->templateParameters()) {
        encodeTemplateParameters(classSymbol);
        return;
      }
    } else if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
      args = functionSymbol->templateArguments();
      if (functionSymbol->isSpecialization())
        templateName = functionSymbol->primaryTemplateSymbol();
    } else if (auto variableSymbol = symbol_cast<VariableSymbol>(symbol)) {
      args = variableSymbol->templateArguments();
      if (variableSymbol->isSpecialization())
        templateName = variableSymbol->primaryTemplateSymbol();
    }

    if (args.empty()) return;

    if (templateName) encoder.enterSubstitution(templateName);

    encoder.out("I");

    std::vector<TemplateParameterAST*> parameters;
    if (auto declaration =
            template_declaration_of(templateName ? templateName : symbol)) {
      for (auto parameter : ListView{declaration->templateParameterList}) {
        parameters.push_back(parameter);
      }
    }

    const bool isOverloadableTemplate = symbol_cast<FunctionSymbol>(symbol);

    for (std::size_t index = 0; index < args.size(); ++index) {
      const auto& arg = args[index];
      if (isOverloadableTemplate && index < parameters.size()) {
        if (auto parameter =
                ast_cast<NonTypeTemplateParameterAST>(parameters[index])) {
          auto declaration = parameter->declaration;
          auto declaredType = declaration ? declaration->type : nullptr;
          if (declaredType && encoder.unit_ &&
              isDependent(encoder.unit_, declaredType)) {
            if (declaration->isPack) encoder.out("Tp");
            encoder.out("Tn");
            encoder.encodeType(declaredType);
          }
        }
      }

      if (auto sym = std::get_if<Symbol*>(&arg)) {
        encodeTemplateArgumentSymbol(*sym);
      } else if (auto type = std::get_if<const Type*>(&arg)) {
        if (!*type) continue;
        encoder.encodeType(*type);
      } else if (auto val = std::get_if<ConstValue>(&arg)) {
        encoder.out(std::format("Li{}E", std::get<std::intmax_t>(*val)));
      } else if (auto exprArg = std::get_if<ExpressionAST*>(&arg)) {
        encodeDependentExpressionArgument(*exprArg);
      }
    }

    encoder.out("E");
  }

  void encodeAbiTagsAndTemplateArguments(Symbol* symbol) {
    if (!symbol) return;
    encoder.encodeAbiTags(symbol);
    encodeTemplateArguments(symbol);
  }

  void encodeTemplateArgumentSymbol(Symbol* sym) {
    if (!sym) return;

    if (encoder.encodeTemplateTemplateArgument(sym)) return;

    if (auto pack = symbol_cast<ParameterPackSymbol>(sym)) {
      encoder.out("J");
      for (auto element : pack->elements()) {
        encodeTemplateArgumentSymbol(element);
      }
      encoder.out("E");
      return;
    }

    auto type = sym->type();

    if (auto var = symbol_cast<VariableSymbol>(sym)) {
      if (var->constValue().has_value() && type) {
        encoder.encodeConstValue(type, var->constValue().value());
        return;
      }
      if (!var->constValue().has_value() && var->initializer()) {
        encodeDependentExpressionArgument(var->initializer());
        return;
      }
    }

    if (!type) return;
    encoder.encodeType(type);
  }

  void encodeDependentExpressionArgument(ExpressionAST* expression) {
    encoder.out("X");
    if (expression && encoder.encodeExpression(expression)) {
      encoder.out("E");
      return;
    }
    if (expression && encoder.unit_) {
      encoder.unit_->error(
          expression->firstSourceLocation(),
          std::format(
              "cannot mangle dependent template argument expression "
              "while encoding '{}'",
              encoder.encodingSymbol_
                  ? to_string(encoder.encodingSymbol_->type(),
                              to_string(encoder.encodingSymbol_->name()))
                  : std::string{}));
    }
    cxx_runtime_error("cannot mangle dependent template argument expression");
  }

  void encodeTemplateParameters(ClassSymbol* classSymbol) {
    encoder.enterSubstitution(classSymbol);

    encoder.out("I");

    for (auto member : classSymbol->templateParameters()->members()) {
      if (auto nonTypeParameter = symbol_cast<NonTypeParameterSymbol>(member)) {
        if (nonTypeParameter->isParameterPack()) {
          encoder.out("JXsp");
          encoder.encodeTemplateParamValue(nonTypeParameter->index());
          encoder.out("EE");
          continue;
        }
        encoder.out("X");
        encoder.encodeTemplateParamValue(nonTypeParameter->index());
        encoder.out("E");
        continue;
      }

      auto parameterType = member->type();
      if (!parameterType) continue;

      if (isParameterPackExpansion(parameterType)) {
        encoder.out("JDp");
        encoder.encodeType(parameterType);
        encoder.out("E");
        continue;
      }

      encoder.encodeType(parameterType);
    }

    encoder.out("E");
  }

  void operator()(const Identifier* id) {
    if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
      if (function->isConstructor()) {
        if (auto inherited = function->inheritedConstructorOrigin()) {
          if (encoder.structorVariant_ == StructorVariant::Base)
            out("CI2");
          else
            out("CI1");
          auto base = enclosing_class_or_namespace(inherited);
          if (!base || !base->type()) {
            cxx_runtime_error("cannot mangle inherited constructor");
          }
          encoder.encodeType(base->type());
          encodeTemplateArguments(inherited);
          return;
        }
        if (encoder.structorVariant_ == StructorVariant::Base)
          out("C2");
        else
          out("C1");
        encodeAbiTagsAndTemplateArguments(symbol);
        return;
      }
    }

    if (encoder.encodeTemplateNameSubstitution(symbol)) return;

    if (hasInternalLinkage(symbol)) out("L");

    out(std::format("{}{}", id->name().length(), id->name()));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const OperatorId* name) {
    auto is_unary = [&] {
      auto function = symbol_cast<FunctionSymbol>(symbol);
      if (!function) {
        cxx_runtime_error(
            std::format("cannot encode operator '{}' for non-function symbol",
                        to_string(name)));
      }

      auto functionType = type_cast<FunctionType>(function->type());
      if (!functionType) {
        cxx_runtime_error(
            std::format("cannot encode operator '{}' for non-function type",
                        to_string(name)));
      }

      bool unary = false;
      switch (name->op()) {
        case TokenKind::T_PLUS:
        case TokenKind::T_MINUS:
        case TokenKind::T_AMP:
        case TokenKind::T_STAR: {
          auto argc = functionType->parameterTypes().size();
          if (argc == 0)
            unary = true;
          else if (argc == 1 &&
                   (!function->parent()->isClass() || function->isFriend()))
            unary = true;
          break;
        }

        default:
          break;
      }

      return unary;
    };

    const auto unary = is_unary();

    out(encoder.encodeOperatorName(name->op(), unary));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const DestructorId* name) {
    switch (encoder.structorVariant_) {
      case StructorVariant::Complete:
        out("D1");
        break;
      case StructorVariant::Base:
        out("D2");
        break;
      case StructorVariant::Deleting:
        out("D0");
        break;
    }
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const LiteralOperatorId* name) {
    out("ll");
    encoder.out(std::format("{}{}", name->name().length(), name->name()));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const ConversionFunctionId* name) {
    out("cv");
    encoder.encodeType(name->type());
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void operator()(const TemplateId* name) {
    auto baseId = name_cast<Identifier>(name->name());
    if (!baseId) {
      cxx_runtime_error(
          std::format("cannot encode template-id '{}'", to_string(name)));
    }
    if (encoder.encodeTemplateNameSubstitution(symbol)) return;

    out(std::format("{}{}", baseId->name().length(), baseId->name()));
    encodeAbiTagsAndTemplateArguments(symbol);
  }

  void out(std::string_view str) { encoder.out(str); }
};

ExternalNameEncoder::ExternalNameEncoder(TranslationUnit* unit) : unit_(unit) {}

auto ExternalNameEncoder::encode(Symbol* symbol, std::string_view suffix)
    -> std::string {
  encodingSymbol_ = symbol;
  std::string result;
  if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    if (!hasExplicitStructorVariant_) {
      if (auto principal = functionSymbol->structorPrincipal()) {
        structorVariant_ = principal->deletingDtorVariant() == functionSymbol
                               ? StructorVariant::Deleting
                               : StructorVariant::Complete;
        functionSymbol = principal;
      } else if (functionSymbol->completeObjectVariant()) {
        structorVariant_ = StructorVariant::Base;
      }
    }
    result = encodeFunction(functionSymbol);
  } else {
    result = encodeData(symbol);
  }
  result.append(suffix);
  return result;
}

auto ExternalNameEncoder::encode(const Type* type) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  encodeType(type);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeVTable(ClassSymbol* classSymbol)
    -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  out("_ZTV");
  encodeName(classSymbol);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeVTT(ClassSymbol* classSymbol) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  out("_ZTT");
  encodeName(classSymbol);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeGuardVariable(Symbol* symbol) -> std::string {
  encodingSymbol_ = symbol;
  std::string externalName;
  std::swap(externalName, out_);

  out("_ZGV");
  encodeName(symbol);

  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeTypeInfo(const Type* type) -> std::string {
  return std::format("_ZTI{}", encode(type));
}

auto ExternalNameEncoder::encodeTypeInfoName(const Type* type) -> std::string {
  return std::format("_ZTS{}", encode(type));
}

auto ExternalNameEncoder::encodeData(Symbol* symbol) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);
  if (is_global_namespace(enclosing_class_or_namespace(symbol)) &&
      !hasInternalLinkage(symbol) && mangledAbiTags(symbol).empty()) {
    auto id = name_cast<Identifier>(symbol->name());
    out(id->name());
  } else {
    out("_Z");
    encodeName(symbol);
  }
  std::swap(externalName, out_);
  return externalName;
}

auto ExternalNameEncoder::encodeFunction(FunctionSymbol* function)
    -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  const auto id = name_cast<Identifier>(function->name());

  if (id &&
      (function->hasCLinkage() ||
       (id->name() == "main" && is_global_namespace(function->parent())))) {
    out(id->name());
  } else {
    out("_Z");
    encodeName(function);
    encodeBareFunctionType(signature_type(function),
                           encodes_return_type(function));
  }

  std::swap(externalName, out_);

  return externalName;
}

void ExternalNameEncoder::encodeName(Symbol* symbol) {
  if (encodeLocalName(symbol)) return;
  if (encodeNestedName(symbol)) return;
  if (encodeUnscopedName(symbol)) return;

  cxx_runtime_error(std::format("cannot encode name for symbol \'{}\'",
                                to_string(symbol->type(), symbol->name())));
}

auto ExternalNameEncoder::encodeTemplateTemplateArgument(Symbol* symbol)
    -> bool {
  auto templateName = template_name_symbol(symbol);
  if (!templateName || symbol_cast<TemplateTypeParameterSymbol>(templateName)) {
    return false;
  }
  encodeTemplateName(templateName);
  return true;
}

void ExternalNameEncoder::encodeTemplateName(Symbol* symbol) {
  auto saved = std::exchange(templateNameOnly_, symbol);
  encodeName(symbol);
  templateNameOnly_ = saved;
}

void ExternalNameEncoder::encodeClosureSourceName(ClassSymbol* classSymbol) {
  auto name = std::format("$_{}", classSymbol->closureDiscriminator());
  out(std::format("{}{}", name.length(), name));
}

auto ExternalNameEncoder::encodeLocalName(Symbol* symbol) -> bool {
  auto function = symbol->enclosingFunction();
  if (!function) return false;

  out("Z");
  encodeName(function);
  if (!is_unmangled_main(function)) {
    encodeBareFunctionType(signature_type(function),
                           encodes_return_type(function));
  }
  out("E");

  if (auto memberFunction = symbol_cast<FunctionSymbol>(symbol)) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(memberFunction->parent());
        classSymbol && classSymbol->isClosureType()) {
      out("N");
      encodeObjectParameterQualifiers(memberFunction);
      encodeClosureSourceName(classSymbol);
      encodeUnqualifiedName(memberFunction);
      out("E");
      return true;
    }
  }

  encodeUnqualifiedName(symbol);
  return true;
}

void ExternalNameEncoder::encodeObjectParameterQualifiers(
    FunctionSymbol* function) {
  if (function->hasExplicitObjectParameter()) {
    out("H");
    return;
  }

  auto functionType = type_cast<FunctionType>(function->type());
  if (!functionType) return;

  const auto cv = functionType->cvQualifiers();
  if (is_const(cv)) out("K");
  if (is_volatile(cv)) out("V");

  if (functionType->refQualifier() == RefQualifier::kLvalue)
    out("R");
  else if (functionType->refQualifier() == RefQualifier::kRvalue)
    out("O");
}

auto ExternalNameEncoder::encodeNestedName(Symbol* symbol) -> bool {
  auto parent = mangling_parent(symbol);
  if (!parent) return false;
  if (is_global_namespace(parent)) return false;
  if (is_std_namespace(parent)) return false;

  if (templateNameOnly_ == symbol) {
    if (auto templateName = template_name(symbol);
        templateName && encodeSubstitution(templateName)) {
      return true;
    }
  }

  out("N");

  if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    encodeObjectParameterQualifiers(functionSymbol);
  }

  if (encodeTemplateNameSubstitution(symbol)) {
    out("E");
    return true;
  }

  encodePrefix(parent);
  encodeUnqualifiedName(symbol);
  out("E");
  return true;
}

auto ExternalNameEncoder::encodeUnscopedName(Symbol* symbol) -> bool {
  if (is_std_namespace(mangling_parent(symbol))) {
    out("St");
  }

  encodeUnqualifiedName(symbol);
  return true;
}

void ExternalNameEncoder::encodePrefix(Symbol* symbol) {
  if (is_std_namespace(symbol)) {
    out("St");
    return;
  }

  if (encodeSubstitution(symbol->type())) return;

  if (auto parent = enclosing_class_or_namespace(symbol);
      parent && !is_global_namespace(parent)) {
    encodePrefix(parent);
  }

  encodeUnqualifiedName(symbol);
  enterSubstitution(symbol->type());
}

void ExternalNameEncoder::encodeTemplatePrefix(Symbol* symbol) {}

void ExternalNameEncoder::encodeUnqualifiedName(Symbol* symbol) {
  if (auto ns = symbol_cast<NamespaceSymbol>(symbol); ns && !ns->name()) {
    auto index = ns->anonNamespaceIndex().value();
    std::string name = std::format("_GLOBAL__N_{}", index + 1);
    out(std::format("{}{}", name.length(), name));
    return;
  }

  visit(EncodeUnqualifiedName{*this, symbol}, symbol->name());
}

void ExternalNameEncoder::encodeBareFunctionType(
    const FunctionType* functionType, bool includeReturnType) {
  if (includeReturnType) {
    encodeType(functionType->returnType());
  }

  for (auto param : functionType->parameterTypes()) {
    if (isParameterPackExpansion(param)) out("Dp");
    encodeType(param);
  }

  if (functionType->isVariadic()) {
    out("z");
  } else if (functionType->parameterTypes().empty()) {
    out("v");
  }
}

void ExternalNameEncoder::encodeType(const Type* type) {
  if (encodeSubstitution(type)) return;
  if (!visit(EncodeType{*this}, type)) return;
  enterSubstitution(type);
}

auto ExternalNameEncoder::encodeDependentName(NestedNameSpecifierAST* nns,
                                              UnqualifiedIdAST* id) -> bool {
  if (auto nameId = ast_cast<NameIdAST>(id)) {
    if (!nameId->identifier) return false;

    out("N");
    if (!encodeDependentQualifier(nns)) return false;
    const auto name = nameId->identifier->name();
    out(std::format("{}{}E", name.length(), name));
    return true;
  }

  if (auto templateId = ast_cast<SimpleTemplateIdAST>(id)) {
    if (!templateId->identifier) return false;

    out("N");
    if (!encodeDependentQualifier(nns)) return false;
    const auto name = templateId->identifier->name();
    out(std::format("{}{}", name.length(), name));
    if (!encodeTemplateArgumentList(templateId->templateArgumentList))
      return false;
    out("E");
    return true;
  }

  return false;
}

auto ExternalNameEncoder::encodeDependentQualifier(NestedNameSpecifierAST* nns)
    -> bool {
  if (!nns) return false;

  if (ast_cast<GlobalNestedNameSpecifierAST>(nns)) return true;

  if (auto simple = ast_cast<SimpleNestedNameSpecifierAST>(nns)) {
    if (simple->nestedNameSpecifier) {
      if (!encodeDependentQualifier(simple->nestedNameSpecifier)) return false;
    } else if (auto param = dependent_prefix_type_param(simple);
               param && type_cast<TypeParameterType>(param->type())) {
      encodeType(param->type());
      return true;
    }
    if (!simple->identifier) return false;
    const auto name = simple->identifier->name();
    out(std::format("{}{}", name.length(), name));
    return true;
  }

  if (auto tmplNns = ast_cast<TemplateNestedNameSpecifierAST>(nns)) {
    auto qualifierSymbol =
        tmplNns->templateId ? tmplNns->templateId->symbol : tmplNns->symbol;
    Symbol* templateName = nullptr;
    if (auto classSymbol = symbol_cast<ClassSymbol>(qualifierSymbol)) {
      if (classSymbol->isSpecialization())
        templateName = classSymbol->primaryTemplateSymbol();
      else if (classSymbol->templateParameters())
        templateName = classSymbol;
    } else if (auto alias = symbol_cast<TypeAliasSymbol>(qualifierSymbol)) {
      if (alias->isSpecialization())
        templateName = alias->primaryTemplateSymbol();
      else if (alias->templateParameters())
        templateName = alias;
    }
    if (templateName && tmplNns->templateId) {
      if (encodeTemplatePrefixSubstitution(
              templateName, tmplNns->templateId->templateArgumentList)) {
        return true;
      }
      if (tmplNns->nestedNameSpecifier) {
        if (!encodeDependentQualifier(tmplNns->nestedNameSpecifier)) {
          return false;
        }
      } else if (!encodeSubstitution(templateName)) {
        if (auto parent = enclosing_class_or_namespace(templateName);
            parent && !is_global_namespace(parent)) {
          encodePrefix(parent);
        }
        auto savedTemplateName = std::exchange(templateNameOnly_, templateName);
        encodeUnqualifiedName(templateName);
        templateNameOnly_ = savedTemplateName;
        enterSubstitution(templateName);
      }
      if (!encodeTemplateArgumentList(
              tmplNns->templateId->templateArgumentList)) {
        return false;
      }
      enterTemplatePrefixSubstitution(
          templateName, tmplNns->templateId->templateArgumentList);
      return true;
    }
    if (tmplNns->templateId && tmplNns->templateId->identifier) {
      if (tmplNns->nestedNameSpecifier &&
          !encodeDependentQualifier(tmplNns->nestedNameSpecifier)) {
        return false;
      }
      const auto name = tmplNns->templateId->identifier->name();
      out(std::format("{}{}", name.length(), name));
      return encodeTemplateArgumentList(
          tmplNns->templateId->templateArgumentList);
    }
  }

  return false;
}

auto ExternalNameEncoder::encodeUnresolvedQualifier(NestedNameSpecifierAST* nns)
    -> bool {
  if (ast_cast<GlobalNestedNameSpecifierAST>(nns)) return true;

  if (auto simple = ast_cast<SimpleNestedNameSpecifierAST>(nns)) {
    if (simple->nestedNameSpecifier &&
        !encodeUnresolvedQualifier(simple->nestedNameSpecifier)) {
      return false;
    }
    if (!simple->identifier) return false;
    const auto name = simple->identifier->name();
    out(std::format("{}{}", name.size(), name));
    return true;
  }

  auto templ = ast_cast<TemplateNestedNameSpecifierAST>(nns);
  if (!templ || !templ->templateId || !templ->templateId->identifier) {
    return false;
  }
  if (templ->nestedNameSpecifier &&
      !encodeUnresolvedQualifier(templ->nestedNameSpecifier)) {
    return false;
  }
  const auto name = templ->templateId->identifier->name();
  out(std::format("{}{}", name.size(), name));
  return encodeTemplateArgumentList(templ->templateId->templateArgumentList);
}

auto ExternalNameEncoder::encodeOperatorName(TokenKind op, bool isUnary)
    -> std::string_view {
  switch (op) {
    case TokenKind::T_NEW:
      return "nw";
    case TokenKind::T_NEW_ARRAY:
      return "na";
    case TokenKind::T_DELETE:
      return "dl";
    case TokenKind::T_DELETE_ARRAY:
      return "da";
    case TokenKind::T_CO_AWAIT:
      return "aw";
    case TokenKind::T_PLUS:
      return isUnary ? "ps" : "pl";
    case TokenKind::T_MINUS:
      return isUnary ? "ng" : "mi";
    case TokenKind::T_AMP:
      return isUnary ? "ad" : "an";
    case TokenKind::T_STAR:
      return isUnary ? "de" : "ml";
    case TokenKind::T_TILDE:
      return "co";
    case TokenKind::T_SLASH:
      return "dv";
    case TokenKind::T_PERCENT:
      return "rm";
    case TokenKind::T_BAR:
      return "or";
    case TokenKind::T_CARET:
      return "eo";
    case TokenKind::T_EQUAL:
      return "aS";
    case TokenKind::T_PLUS_EQUAL:
      return "pL";
    case TokenKind::T_MINUS_EQUAL:
      return "mI";
    case TokenKind::T_STAR_EQUAL:
      return "mL";
    case TokenKind::T_SLASH_EQUAL:
      return "dV";
    case TokenKind::T_PERCENT_EQUAL:
      return "rM";
    case TokenKind::T_AMP_EQUAL:
      return "aN";
    case TokenKind::T_BAR_EQUAL:
      return "oR";
    case TokenKind::T_CARET_EQUAL:
      return "eO";
    case TokenKind::T_LESS_LESS:
      return "ls";
    case TokenKind::T_GREATER_GREATER:
      return "rs";
    case TokenKind::T_LESS_LESS_EQUAL:
      return "lS";
    case TokenKind::T_GREATER_GREATER_EQUAL:
      return "rS";
    case TokenKind::T_EQUAL_EQUAL:
      return "eq";
    case TokenKind::T_EXCLAIM_EQUAL:
      return "ne";
    case TokenKind::T_LESS:
      return "lt";
    case TokenKind::T_GREATER:
      return "gt";
    case TokenKind::T_LESS_EQUAL:
      return "le";
    case TokenKind::T_GREATER_EQUAL:
      return "ge";
    case TokenKind::T_LESS_EQUAL_GREATER:
      return "ss";
    case TokenKind::T_EXCLAIM:
      return "nt";
    case TokenKind::T_AMP_AMP:
      return "aa";
    case TokenKind::T_BAR_BAR:
      return "oo";
    case TokenKind::T_PLUS_PLUS:
      return "pp";
    case TokenKind::T_MINUS_MINUS:
      return "mm";
    case TokenKind::T_COMMA:
      return "cm";
    case TokenKind::T_MINUS_GREATER_STAR:
      return "pm";
    case TokenKind::T_MINUS_GREATER:
      return "pt";
    case TokenKind::T_LPAREN:
      return "cl";
    case TokenKind::T_LBRACKET:
      return "ix";
    case TokenKind::T_QUESTION:
      return "qu";
    default:
      cxx_runtime_error(
          std::format("cannot encode operator '{}'", Token::spell(op)));
  }
}

void ExternalNameEncoder::encodeTemplateParamValue(int index) {
  if (index == 0) {
    out("T_");
  } else {
    out(std::format("T{}_", index - 1));
  }
}

struct ExternalNameEncoder::EncodeExpression {
  ExternalNameEncoder& encoder;

  [[nodiscard]] auto encode(ExpressionAST* expr) const -> bool {
    return encoder.encodeExpression(expr);
  }

  [[nodiscard]] auto operator()(NestedExpressionAST* ast) const -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast) const -> bool {
    encoder.out("sp");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast) const -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(ConstExpressionAST* ast) const -> bool {
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast) const -> bool {
    if (!ast->type) return false;
    encoder.encodeConstValue(ast->type,
                             ConstValue{std::intmax_t(ast->isTrue ? 1 : 0)});
    return true;
  }

  [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast) const -> bool {
    if (!ast->literal || !ast->type) return false;
    encoder.encodeConstValue(
        ast->type,
        ConstValue{static_cast<std::intmax_t>(ast->literal->integerValue())});
    return true;
  }

  [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast) const -> bool {
    auto parameter = template_parameter_info(ast->symbol);
    if (!parameter) return false;
    encoder.out("sZ");
    encoder.encodeTemplateParamValue(parameter->index);
    return true;
  }

  [[nodiscard]] auto operator()(TypeTraitExpressionAST* ast) const -> bool {
    const auto name = encoder.unit_->tokenText(ast->typeTraitLoc);
    encoder.out(std::format("u{}{}", name.length(), name));
    for (auto typeId : ListView{ast->typeIdList}) {
      if (!typeId || !typeId->type) return false;
      encoder.encodeType(typeId->type);
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast) const -> bool {
    if (!ast->typeId || !ast->typeId->type) return false;
    encoder.out("st");
    encoder.encodeType(ast->typeId->type);
    return true;
  }

  [[nodiscard]] auto operator()(SizeofExpressionAST* ast) const -> bool {
    encoder.out("sz");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast) const -> bool {
    if (!ast->typeId || !ast->typeId->type) return false;
    encoder.out("at");
    encoder.encodeType(ast->typeId->type);
    return true;
  }

  [[nodiscard]] auto operator()(AlignofExpressionAST* ast) const -> bool {
    encoder.out("az");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(NoexceptExpressionAST* ast) const -> bool {
    encoder.out("nx");
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(UnaryExpressionAST* ast) const -> bool {
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/true));
    return encode(ast->expression);
  }

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) const -> bool {
    encoder.out(encoder.encodeOperatorName(ast->op, /*isUnary=*/false));
    if (!encode(ast->leftExpression)) return false;
    return encode(ast->rightExpression);
  }

  [[nodiscard]] auto operator()(CallExpressionAST* ast) const -> bool {
    encoder.out("cl");
    if (!encode(ast->baseExpression)) return false;
    for (auto argument : ListView{ast->expressionList}) {
      if (!encode(argument)) return false;
    }
    encoder.out("E");
    return true;
  }

  [[nodiscard]] auto operator()(IdExpressionAST* ast) const -> bool {
    if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol)) {
      encoder.encodeTemplateParamValue(param->index());
      return true;
    }

    if (auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
        templateId && templateId->identifier) {
      if (!encodeUnresolvedPrefix(ast)) return false;
      const auto name = templateId->identifier->name();
      encoder.out(std::format("{}{}", name.length(), name));
      return encoder.encodeTemplateArgumentList(
          templateId->templateArgumentList);
    }

    auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId);
    if (!nameId || !nameId->identifier || !ast->nestedNameSpecifier) {
      return false;
    }

    if (!encodeUnresolvedPrefix(ast)) return false;
    const auto name = nameId->identifier->name();
    encoder.out(std::format("{}{}", name.size(), name));
    return true;
  }

  [[nodiscard]] auto operator()(ExpressionAST*) const -> bool { return false; }

 private:
  [[nodiscard]] auto encodeUnresolvedPrefix(IdExpressionAST* ast) const
      -> bool {
    if (!ast->nestedNameSpecifier) return true;
    if (has_global_qualifier(ast->nestedNameSpecifier)) encoder.out("gs");
    encoder.out("sr");
    if (!encoder.encodeUnresolvedQualifier(ast->nestedNameSpecifier)) {
      return false;
    }
    encoder.out("E");
    return true;
  }
};

auto ExternalNameEncoder::encodeExpression(ExpressionAST* expr) -> bool {
  if (!expr) return false;

  const auto outMark = out_.size();
  const auto substsSnapshot = substs_;

  if (visit(EncodeExpression{*this}, expr)) return true;

  out_.resize(outMark);
  substs_ = substsSnapshot;
  return false;
}

auto ExternalNameEncoder::encodeTemplateArgumentList(
    List<TemplateArgumentAST*>* arguments) -> bool {
  out("I");
  for (auto argument : ListView{arguments}) {
    if (auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument)) {
      if (!typeArgument->typeId || !typeArgument->typeId->type) return false;
      encodeType(typeArgument->typeId->type);
      continue;
    }
    if (auto expressionArgument =
            ast_cast<ExpressionTemplateArgumentAST>(argument)) {
      const bool isPackExpansion =
          ast_cast<PackExpansionExpressionAST>(expressionArgument->expression);
      if (isPackExpansion) out("J");
      out("X");
      if (!encodeExpression(expressionArgument->expression)) return false;
      out("E");
      if (isPackExpansion) out("E");
      continue;
    }
    return false;
  }
  out("E");
  return true;
}

void ExternalNameEncoder::encodeConstValue(const Type* type,
                                           const ConstValue& value) {
  out("L");
  encodeType(type);
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::intmax_t>) {
          if (v < 0) {
            out(std::format("n{}", -v));
          } else {
            out(std::format("{}", v));
          }
        } else if constexpr (std::is_same_v<T, bool>) {
          out(v ? "1" : "0");
        } else if constexpr (std::is_same_v<T, double>) {
          if (type_cast<FloatType>(type)) {
            const auto bits =
                std::bit_cast<std::uint32_t>(static_cast<float>(v));
            out(std::format("{:08x}", bits));
          } else {
            const auto bits = std::bit_cast<std::uint64_t>(v);
            out(std::format("{:016x}", bits));
          }
        }
      },
      value);
  out("E");
}

namespace {

struct CollectAbiTags {
  std::set<const Identifier*>& tags;
  std::set<const Type*> visited;

  void collect(const Type* type) {
    if (!type) return;
    if (!visited.insert(type).second) return;
    visit(*this, type);
  }

  void collect(std::span<const TemplateArgument> args) {
    for (const auto& arg : args) {
      if (auto sym = std::get_if<Symbol*>(&arg)) collect((*sym)->type());
    }
  }

  void operator()(const QualType* type) { collect(type->elementType()); }
  void operator()(const PointerType* type) { collect(type->elementType()); }

  void operator()(const LvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const RvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const BoundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const UnboundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const ClassType* type) {
    auto classSymbol = type->symbol();
    if (!classSymbol) return;
    addTags(classSymbol);
    collect(classSymbol->templateArguments());
  }

  void operator()(const EnumType* type) { addTags(type->symbol()); }
  void operator()(const ScopedEnumType* type) { addTags(type->symbol()); }

  void operator()(const Type*) {}

  void addTags(Symbol* symbol) {
    if (!symbol) return;
    for (auto tag : symbol->abiTags()) tags.insert(tag);
  }
};

}  // namespace

auto ExternalNameEncoder::mangledAbiTags(Symbol* symbol)
    -> std::vector<const Identifier*> {
  if (!symbol) return {};

  std::set<const Identifier*> declaredTags;
  auto addDeclaredTags = [&](Symbol* declaration) {
    for (auto tag : declaration->abiTags()) declaredTags.insert(tag);
  };

  addDeclaredTags(symbol);
  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    addDeclaredTags(function->canonical());
    for (auto redeclaration : function->canonical()->redeclarations()) {
      addDeclaredTags(redeclaration);
    }
  }

  std::vector<const Identifier*> tags{declaredTags.begin(), declaredTags.end()};
  std::ranges::sort(tags, {}, [](const Identifier* id) { return id->name(); });

  std::set<const Identifier*> mangled{tags.begin(), tags.end()};
  CollectAbiTags mangledCollector{mangled};
  const Type* unmangledType = nullptr;

  if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
    if (function->isConstructor() || function->isDestructor()) return tags;

    auto functionType = type_cast<FunctionType>(function->type());
    if (!functionType) return tags;

    unmangledType = functionType->returnType();

    for (auto parameterType : functionType->parameterTypes()) {
      mangledCollector.collect(parameterType);
    }
    mangledCollector.collect(function->templateArguments());
    if (encodes_return_type(function)) mangledCollector.collect(unmangledType);
  } else if (auto variable = symbol_cast<VariableSymbol>(symbol)) {
    unmangledType = variable->type();
    mangledCollector.collect(variable->templateArguments());
  } else if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    unmangledType = field->type();
  } else {
    return tags;
  }

  for (auto enclosing : symbol->enclosingSymbols()) {
    for (auto tag : enclosing->abiTags()) mangled.insert(tag);
  }

  std::set<const Identifier*> unmangled;
  CollectAbiTags{unmangled}.collect(unmangledType);

  for (auto tag : unmangled) {
    if (mangled.contains(tag)) continue;
    tags.push_back(tag);
  }

  std::ranges::sort(tags, {}, [](const Identifier* id) { return id->name(); });
  tags.erase(std::ranges::unique(tags).begin(), tags.end());

  return tags;
}

void ExternalNameEncoder::encodeAbiTags(Symbol* symbol) {
  for (auto tag : mangledAbiTags(symbol)) {
    out(std::format("B{}{}", tag->name().length(), tag->name()));
  }
}

auto ExternalNameEncoder::encodeTemplateNameSubstitution(Symbol* symbol)
    -> bool {
  auto templateName = template_name(symbol);
  if (!templateName) return false;
  if (!encodeSubstitution(templateName)) return false;
  EncodeUnqualifiedName{*this, symbol}.encodeAbiTagsAndTemplateArguments(
      symbol);
  return true;
}

auto ExternalNameEncoder::encodeSubstitution(const Type* type) -> bool {
  auto sameType = [&](const Substitution& substitution) {
    auto candidate = std::get_if<const Type*>(&substitution);
    if (!candidate) return false;
    if (*candidate == type) return true;
    return unit_ && TypeTraits{unit_}.is_same(*candidate, type);
  };
  auto it = std::ranges::find_if(substs_, sameType);
  if (it == substs_.end()) return false;
  const auto index = static_cast<int>(std::distance(substs_.begin(), it));

  if (index == 0) {
    out("S_");
    return true;
  }

  out(std::format("S{}_", encodeSeqId(index - 1)));
  return true;
}

auto ExternalNameEncoder::encodeSubstitution(Symbol* symbol) -> bool {
  auto matches = [&](const Substitution& substitution) {
    auto candidate = std::get_if<Symbol*>(&substitution);
    return candidate && *candidate == symbol;
  };
  auto it = std::ranges::find_if(substs_, matches);
  if (it == substs_.end()) return false;
  const auto index = static_cast<int>(std::distance(substs_.begin(), it));

  if (index == 0) {
    out("S_");
    return true;
  }

  out(std::format("S{}_", encodeSeqId(index - 1)));
  return true;
}

auto ExternalNameEncoder::encodeTemplatePrefixSubstitution(
    Symbol* templateSymbol, List<TemplateArgumentAST*>* arguments) -> bool {
  auto sameTemplateId = [&](const Substitution& substitution) {
    auto candidate = std::get_if<TemplatePrefixSubstitution>(&substitution);
    if (!candidate || candidate->templateSymbol != templateSymbol) return false;
    return unit_ && areTemplateArgumentListsSyntacticallyEquivalent(
                        unit_, candidate->arguments, arguments);
  };
  auto it = std::ranges::find_if(substs_, sameTemplateId);
  if (it == substs_.end()) return false;
  const auto index = static_cast<int>(std::distance(substs_.begin(), it));

  if (index == 0) {
    out("S_");
    return true;
  }

  out(std::format("S{}_", encodeSeqId(index - 1)));
  return true;
}

auto ExternalNameEncoder::encodeSeqId(int id) -> std::string {
  static constexpr char digits[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string result;
  do {
    result.insert(result.begin(), digits[id % 36]);
    id /= 36;
  } while (id != 0);
  return result;
}

void ExternalNameEncoder::enterSubstitution(const Type* type) {
  auto sameType = [&](const Substitution& substitution) {
    auto candidate = std::get_if<const Type*>(&substitution);
    if (!candidate) return false;
    if (*candidate == type) return true;
    return unit_ && TypeTraits{unit_}.is_same(*candidate, type);
  };
  if (std::ranges::any_of(substs_, sameType)) return;
  substs_.emplace_back(type);
}

void ExternalNameEncoder::enterSubstitution(Symbol* symbol) {
  auto matches = [&](const Substitution& substitution) {
    auto candidate = std::get_if<Symbol*>(&substitution);
    return candidate && *candidate == symbol;
  };
  if (std::ranges::any_of(substs_, matches)) return;
  substs_.emplace_back(symbol);
}

void ExternalNameEncoder::enterTemplatePrefixSubstitution(
    Symbol* templateSymbol, List<TemplateArgumentAST*>* arguments) {
  auto sameTemplateId = [&](const Substitution& substitution) {
    auto candidate = std::get_if<TemplatePrefixSubstitution>(&substitution);
    if (!candidate || candidate->templateSymbol != templateSymbol) return false;
    return unit_ && areTemplateArgumentListsSyntacticallyEquivalent(
                        unit_, candidate->arguments, arguments);
  };
  if (std::ranges::any_of(substs_, sameTemplateId)) return;
  substs_.push_back(TemplatePrefixSubstitution{templateSymbol, arguments});
}

}  // namespace cxx
