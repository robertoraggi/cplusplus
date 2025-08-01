// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/external_name_encoder.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

namespace {

[[nodiscard]] auto is_global_namespace(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (!symbol->isNamespace()) return false;
  if (symbol->enclosingSymbol()) return false;
  return true;
}

[[nodiscard]] auto enclosing_class_or_namespace(Symbol* symbol) -> Symbol* {
  if (!symbol) return nullptr;
  auto parent = symbol->enclosingSymbol();
  if (!parent) return nullptr;
  if (!parent->isClassOrNamespace()) return nullptr;
  return parent;
}

[[nodiscard]] auto is_std_namespace(Symbol* symbol) -> bool {
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
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
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

    if (type->isNoexcept()) {
      // todo: computed noexcept
      encoder.out("N");
    }

    // todo: "Y" prefix for the bare function type encodes extern "C"
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
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const MemberFunctionPointerType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const NamespaceType* type) -> bool { return false; }

  auto operator()(const TypeParameterType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const TemplateTypeParameterType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const UnresolvedNameType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const UnresolvedBoundedArrayType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const UnresolvedUnderlyingType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const OverloadSetType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }

  auto operator()(const BuiltinVaListType* type) -> bool {
    cxx_runtime_error(std::format("todo encode type '{}'", to_string(type)));
    return false;
  }
};

struct ExternalNameEncoder::EncodeUnqualifiedName {
  ExternalNameEncoder& encoder;
  Symbol* symbol = nullptr;

  void operator()(const Identifier* id) {
    if (auto function = symbol_cast<FunctionSymbol>(symbol)) {
      if (function->isConstructor()) {
        out("C2");
        return;
      }
    }

    out(std::format("{}{}", id->name().length(), id->name()));
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
          else if (argc == 1 && !function->enclosingSymbol()->isClass())
            unary = true;
          break;
        }

        default:
          break;
      }  // switch

      return unary;
    };

    const auto unary = is_unary();

    switch (name->op()) {
      case TokenKind::T_NEW:
        out("nw");
        break;
      case TokenKind::T_NEW_ARRAY:
        out("na");
        break;
      case TokenKind::T_DELETE:
        out("dl");
        break;
      case TokenKind::T_DELETE_ARRAY:
        out("da");
        break;
      case TokenKind::T_CO_AWAIT:
        out("aw");
        break;
      case TokenKind::T_PLUS:
        out(unary ? "ps" : "pl");
        break;
      case TokenKind::T_MINUS:
        out(unary ? "ng" : "mi");
        break;
      case TokenKind::T_AMP:
        out(unary ? "ad" : "an");
        break;
      case TokenKind::T_STAR:
        out(unary ? "de" : "ml");
        break;
      case TokenKind::T_TILDE:
        out("co");
        break;
      case TokenKind::T_SLASH:
        out("dv");
        break;
      case TokenKind::T_PERCENT:
        out("rm");
        break;
      case TokenKind::T_BAR:
        out("or");
        break;
      case TokenKind::T_CARET:
        out("eo");
        break;
      case TokenKind::T_EQUAL:
        out("aS");
        break;
      case TokenKind::T_PLUS_EQUAL:
        out("pL");
        break;
      case TokenKind::T_MINUS_EQUAL:
        out("mI");
        break;
      case TokenKind::T_STAR_EQUAL:
        out("mL");
        break;
      case TokenKind::T_SLASH_EQUAL:
        out("dV");
        break;
      case TokenKind::T_PERCENT_EQUAL:
        out("rM");
        break;
      case TokenKind::T_AMP_EQUAL:
        out("aN");
        break;
      case TokenKind::T_BAR_EQUAL:
        out("oR");
        break;
      case TokenKind::T_CARET_EQUAL:
        out("eO");
        break;
      case TokenKind::T_LESS_LESS:
        out("ls");
        break;
      case TokenKind::T_GREATER_GREATER:
        out("rs");
        break;
      case TokenKind::T_LESS_LESS_EQUAL:
        out("lS");
        break;
      case TokenKind::T_GREATER_GREATER_EQUAL:
        out("rS");
        break;
      case TokenKind::T_EQUAL_EQUAL:
        out("eq");
        break;
      case TokenKind::T_EXCLAIM_EQUAL:
        out("ne");
        break;
      case TokenKind::T_LESS:
        out("lt");
        break;
      case TokenKind::T_GREATER:
        out("gt");
        break;
      case TokenKind::T_LESS_EQUAL:
        out("le");
        break;
      case TokenKind::T_GREATER_EQUAL:
        out("ge");
        break;
      case TokenKind::T_LESS_EQUAL_GREATER:
        out("ss");
        break;
      case TokenKind::T_EXCLAIM:
        out("nt");
        break;
      case TokenKind::T_AMP_AMP:
        out("aa");
        break;
      case TokenKind::T_BAR_BAR:
        out("oo");
        break;
      case TokenKind::T_PLUS_PLUS:
        out("pp");
        break;
      case TokenKind::T_MINUS_MINUS:
        out("mm");
        break;
      case TokenKind::T_COMMA:
        out("cm");
        break;
      case TokenKind::T_MINUS_GREATER_STAR:
        out("pm");
        break;
      case TokenKind::T_MINUS_GREATER:
        out("pt");
        break;
      case TokenKind::T_LPAREN:
        out("cl");
        break;
      case TokenKind::T_LBRACKET:
        out("ix");
        break;
      case TokenKind::T_QUESTION:
        out("qu");
        break;
      default:
        cxx_runtime_error(
            std::format("cannot encode operator '{}'", to_string(name)));
    }  // switch
  }

  void operator()(const DestructorId* name) { out("D2"); }

  void operator()(const LiteralOperatorId* name) {
    out("ll");
    encoder.out(std::format("{}{}", name->name().length(), name->name()));
  }

  void operator()(const ConversionFunctionId* name) {
    out("cv");
    encoder.encodeType(name->type());
  }

  void operator()(const TemplateId* name) {
    cxx_runtime_error("template names not supported yet");
  }

  void out(std::string_view str) { encoder.out(str); }
};

ExternalNameEncoder::ExternalNameEncoder() {}

auto ExternalNameEncoder::encode(Symbol* symbol) -> std::string {
  if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
    return encodeFunction(functionSymbol);
  }

  return encodeData(symbol);
}

auto ExternalNameEncoder::encode(const Type* type) -> std::string {
  std::string externalName;
  std::swap(externalName, out_);

  encodeType(type);

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

  if (id && (function->hasCLinkage() ||
             (id->name() == "main" &&
              is_global_namespace(function->enclosingSymbol())))) {
    out(id->name());
  } else {
    out("_Z");
    encodeName(function);
    auto functionType = type_cast<FunctionType>(function->type());
    encodeBareFunctionType(functionType);
  }

  std::swap(externalName, out_);

  return externalName;
}

void ExternalNameEncoder::encodeName(Symbol* symbol) {
  if (encodeNestedName(symbol)) return;
  if (encodeUnscopedName(symbol)) return;

  cxx_runtime_error(std::format("cannot encode name for symbol '{}'",
                                to_string(symbol->type(), symbol->name())));
}

auto ExternalNameEncoder::encodeNestedName(Symbol* symbol) -> bool {
  auto parent = enclosing_class_or_namespace(symbol);
  if (!parent) return false;
  if (is_global_namespace(parent)) return false;
  if (is_std_namespace(parent)) return false;

  out("N");
  // todo: encode cv qualifiers
  // todo: encode ref qualifier
  encodePrefix(parent);
  encodeUnqualifiedName(symbol);
  out("E");
  return true;
}

auto ExternalNameEncoder::encodeUnscopedName(Symbol* symbol) -> bool {
  if (is_std_namespace(enclosing_class_or_namespace(symbol))) {
    out("St");
  }

  encodeUnqualifiedName(symbol);
  return true;
}

void ExternalNameEncoder::encodePrefix(Symbol* symbol) {
  if (encodeSubstitution(symbol->type())) return;

  if (auto parent = enclosing_class_or_namespace(symbol);
      parent && !is_global_namespace(parent)) {
    encodePrefix(parent);
  }

  enterSubstitution(symbol->type());
  encodeUnqualifiedName(symbol);
}

void ExternalNameEncoder::encodeTemplatePrefix(Symbol* symbol) {}

void ExternalNameEncoder::encodeUnqualifiedName(Symbol* symbol) {
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

  if (functionType->parameterTypes().empty()) {
    out("v");
  }

  if (functionType->isVariadic()) {
    out("z");
  }
}

void ExternalNameEncoder::encodeType(const Type* type) {
  if (encodeSubstitution(type)) return;
  if (!visit(EncodeType{*this}, type)) return;
  enterSubstitution(type);
}

auto ExternalNameEncoder::encodeSubstitution(const Type* type) -> bool {
  auto it = substs_.find(type);
  if (it == substs_.end()) return false;

  const auto index = it->second;

  if (index == 0) {
    out("S_");
    return true;
  }

  out(std::format("S{}_", index - 1));
  return true;
}

void ExternalNameEncoder::enterSubstitution(const Type* type) {
  const auto index = substCount_;
  ++substCount_;

  substs_.emplace(type, index);
}

}  // namespace cxx