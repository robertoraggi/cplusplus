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
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <bit>
#include <cstdint>
#include <format>

namespace cxx {
namespace {
[[nodiscard]] auto enclosing_class_or_namespace(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;
  auto parent = symbol->enclosingNonTemplateParametersScope();
  if (!parent || !parent->isClassOrNamespace()) return nullptr;
  return parent;
}

[[nodiscard]] auto is_unmangled_main(FunctionSymbol* function) -> bool {
  auto id = name_cast<Identifier>(function->name());
  return id && id->name() == "main" && is_global_namespace(function->parent());
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
    if (type->index() == 0) {
      encoder.out("T_");
      return true;
    }

    encoder.out(std::format("T{}_", type->index() - 1));
    return true;
  }

  auto operator()(const TemplateTypeParameterType* type) -> bool {
    if (type->index() == 0) {
      encoder.out("T_");
      return true;
    }

    encoder.out(std::format("T{}_", type->index() - 1));
    return true;
  }

  auto operator()(const UnresolvedNameType* type) -> bool {
    if (encoder.encodeDependentName(type->nestedNameSpecifier(),
                                    type->unqualifiedId())) {
      return true;
    }
    encoder.out("u8__dep_ty");
    return true;
  }

  auto operator()(const UnresolvedBoundedArrayType* type) -> bool {
    encoder.out("A0_");
    return true;
  }

  auto operator()(const UnresolvedUnderlyingType* type) -> bool {
    encoder.out("u8__dep_ut");
    return true;
  }

  auto operator()(const UnresolvedBuiltinType* type) -> bool {
    encoder.out("u8__dep_bt");
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
    encoder.out("u8__dep_bi");
    return true;
  }
};

struct ExternalNameEncoder::EncodeUnqualifiedName {
  ExternalNameEncoder& encoder;
  Symbol* symbol = nullptr;

  void encodeTemplateArguments(Symbol* symbol) {
    if (!symbol) return;

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

    for (const auto& arg : args) {
      if (auto sym = std::get_if<Symbol*>(&arg)) {
        auto type = (*sym)->type();

        if (auto var = symbol_cast<VariableSymbol>(*sym)) {
          if (var->constValue().has_value() && type) {
            encoder.encodeConstValue(type, var->constValue().value());
            continue;
          }
          if (!var->constValue().has_value() && var->initializer()) {
            const auto mark = encoder.out_.size();
            encoder.out("X");
            if (encoder.encodeExpression(var->initializer())) {
              encoder.out("E");
              continue;
            }
            encoder.out_.resize(mark);
            encoder.out("u10__dep_expr");
            continue;
          }
        }

        if (!type) continue;
        if (type_cast<TypeParameterType>(type)) continue;
        if (type_cast<TemplateTypeParameterType>(type)) continue;

        encoder.encodeType(type);
      } else if (auto type = std::get_if<const Type*>(&arg)) {
        if (*type) encoder.encodeType(*type);
      } else if (auto val = std::get_if<ConstValue>(&arg)) {
        encoder.out(std::format("Li{}E", std::get<std::intmax_t>(*val)));
      } else if (auto exprArg = std::get_if<ExpressionAST*>(&arg)) {
        const auto mark = encoder.out_.size();
        encoder.out("X");
        if (*exprArg && encoder.encodeExpression(*exprArg)) {
          encoder.out("E");
        } else {
          encoder.out_.resize(mark);
          encoder.out("u10__dep_expr");
        }
      }
    }

    encoder.out("E");
  }

  void encodeTemplateParameters(ClassSymbol* classSymbol) {
    encoder.enterSubstitution(classSymbol);

    encoder.out("I");

    for (auto member : classSymbol->templateParameters()->members()) {
      if (auto typeParameter = symbol_cast<TypeParameterSymbol>(member)) {
        encoder.encodeType(typeParameter->type());
      } else if (auto nonTypeParameter =
                     symbol_cast<NonTypeParameterSymbol>(member)) {
        encoder.out("X");
        encoder.encodeTemplateParamValue(nonTypeParameter->index());
        encoder.out("E");
      }
    }

    encoder.out("E");
  }

  void operator()(const Identifier* id) {
    if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
      if (function->isConstructor()) {
        out(encoder.structorVariant_ == StructorVariant::Base ? "C2" : "C1");
        encodeTemplateArguments(symbol);
        return;
      }
    }

    if (encoder.encodeTemplateNameSubstitution(symbol)) return;

    out(std::format("{}{}", id->name().length(), id->name()));
    encodeTemplateArguments(symbol);
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
  }

  void operator()(const LiteralOperatorId* name) {
    out("ll");
    encoder.out(std::format("{}{}", name->name().length(), name->name()));
  }

  void operator()(const ConversionFunctionId* name) {
    out("cv");
    encoder.encodeType(name->type());
  }

  void operator()(const TemplateId* name) {
    auto baseId = name_cast<Identifier>(name->name());
    if (!baseId) {
      cxx_runtime_error(
          std::format("cannot encode template-id '{}'", to_string(name)));
    }
    if (encoder.encodeTemplateNameSubstitution(symbol)) return;

    out(std::format("{}{}", baseId->name().length(), baseId->name()));
    encodeTemplateArguments(symbol);
  }

  void out(std::string_view str) { encoder.out(str); }
};

ExternalNameEncoder::ExternalNameEncoder() {}

auto ExternalNameEncoder::encode(Symbol* symbol, std::string_view suffix)
    -> std::string {
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

auto ExternalNameEncoder::encodeData(Symbol* symbol) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);
  if (is_global_namespace(enclosing_class_or_namespace(symbol))) {
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
      if (auto functionType = type_cast<FunctionType>(memberFunction->type())) {
        const auto cv = functionType->cvQualifiers();
        if (is_const(cv)) out("K");
        if (is_volatile(cv)) out("V");
        if (functionType->refQualifier() == RefQualifier::kLvalue) {
          out("R");
        } else if (functionType->refQualifier() == RefQualifier::kRvalue) {
          out("O");
        }
      }
      encodeClosureSourceName(classSymbol);
      encodeUnqualifiedName(memberFunction);
      out("E");
      return true;
    }
  }

  encodeUnqualifiedName(symbol);
  return true;
}

auto ExternalNameEncoder::encodeNestedName(Symbol* symbol) -> bool {
  auto parent = mangling_parent(symbol);
  if (!parent) return false;
  if (is_global_namespace(parent)) return false;
  if (is_std_namespace(parent)) return false;

  out("N");

  if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    auto functionType = type_cast<FunctionType>(functionSymbol->type());
    if (is_const(functionType->cvQualifiers())) out("K");
    if (is_volatile(functionType->cvQualifiers())) out("V");

    if (functionType->refQualifier() == RefQualifier::kLvalue)
      out("R");
    else if (functionType->refQualifier() == RefQualifier::kRvalue)
      out("O");
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
  auto nameId = ast_cast<NameIdAST>(id);
  if (!nameId || !nameId->identifier) return false;

  const auto encodeName = [&] {
    const auto name = nameId->identifier->name();
    out(std::format("{}{}", name.length(), name));
  };

  if (auto param = dependent_prefix_type_param(nns);
      param && type_cast<TypeParameterType>(param->type())) {
    out("N");
    encodeType(param->type());
    encodeName();
    out("E");
    return true;
  }

  if (auto tmplNns = ast_cast<TemplateNestedNameSpecifierAST>(nns)) {
    auto classSymbol = symbol_cast<ClassSymbol>(
        tmplNns->templateId ? tmplNns->templateId->symbol : tmplNns->symbol);
    if (classSymbol && classSymbol->templateParameters()) {
      out("N");
      encodePrefix(classSymbol);
      encodeName();
      out("E");
      return true;
    }
  }

  return false;
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

auto ExternalNameEncoder::encodeExpression(ExpressionAST* expr) -> bool {
  if (!expr) return false;

  const auto outMark = out_.size();
  const auto substsSnapshot = substs_;
  const auto substCountSnapshot = substCount_;

  const auto rollback = [&] {
    out_.resize(outMark);
    substs_ = substsSnapshot;
    substCount_ = substCountSnapshot;
    return false;
  };

  if (auto nested = ast_cast<NestedExpressionAST>(expr)) {
    return encodeExpression(nested->expression);
  }

  if (auto boolLit = ast_cast<BoolLiteralExpressionAST>(expr)) {
    if (!boolLit->type) return rollback();
    encodeConstValue(boolLit->type,
                     ConstValue{std::intmax_t(boolLit->isTrue ? 1 : 0)});
    return true;
  }

  if (auto intLit = ast_cast<IntLiteralExpressionAST>(expr)) {
    if (!intLit->literal || !intLit->type) return rollback();
    encodeConstValue(intLit->type, ConstValue{static_cast<std::intmax_t>(
                                       intLit->literal->integerValue())});
    return true;
  }

  if (auto unary = ast_cast<UnaryExpressionAST>(expr)) {
    out(encodeOperatorName(unary->op, /*isUnary=*/true));
    if (!encodeExpression(unary->expression)) return rollback();
    return true;
  }

  if (auto binary = ast_cast<BinaryExpressionAST>(expr)) {
    out(encodeOperatorName(binary->op, /*isUnary=*/false));
    if (!encodeExpression(binary->leftExpression)) return rollback();
    if (!encodeExpression(binary->rightExpression)) return rollback();
    return true;
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(expr)) {
    if (auto param = symbol_cast<NonTypeParameterSymbol>(idExpr->symbol)) {
      encodeTemplateParamValue(param->index());
      return true;
    }

    if (auto templateId = ast_cast<SimpleTemplateIdAST>(idExpr->unqualifiedId);
        templateId && templateId->identifier && !idExpr->nestedNameSpecifier) {
      const auto name = templateId->identifier->name();
      out(std::format("{}{}", name.length(), name));
      out("I");
      for (auto arg : ListView{templateId->templateArgumentList}) {
        if (auto typeArg = ast_cast<TypeTemplateArgumentAST>(arg)) {
          if (!typeArg->typeId || !typeArg->typeId->type) return rollback();
          encodeType(typeArg->typeId->type);
        } else if (auto exprArg =
                       ast_cast<ExpressionTemplateArgumentAST>(arg)) {
          if (!encodeExpression(exprArg->expression)) return rollback();
        } else {
          return rollback();
        }
      }
      out("E");
      return true;
    }
  }

  return rollback();
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

auto ExternalNameEncoder::encodeTemplateNameSubstitution(Symbol* symbol)
    -> bool {
  auto templateName = template_name(symbol);
  if (!templateName) return false;
  if (!encodeSubstitution(templateName)) return false;
  EncodeUnqualifiedName{*this, symbol}.encodeTemplateArguments(symbol);
  return true;
}

auto ExternalNameEncoder::encodeSubstitution(const void* key) -> bool {
  auto it = substs_.find(key);
  if (it == substs_.end()) return false;

  const auto index = it->second;

  if (index == 0) {
    out("S_");
    return true;
  }

  out(std::format("S{}_", index - 1));
  return true;
}

void ExternalNameEncoder::enterSubstitution(const void* key) {
  if (substs_.contains(key)) return;

  const auto index = substCount_;
  ++substCount_;

  substs_.emplace(key, index);
}
}  // namespace cxx
