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

// cxx
#include <cxx/ast.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

namespace {

class TypePrinter {
 public:
  TypePrinter() {
    specifiers_.clear();
    ptrOps_.clear();
    declarator_.clear();
    addFormals_ = true;
  }

  ~TypePrinter() {
    specifiers_.clear();
    ptrOps_.clear();
    declarator_.clear();
  }

  auto operator()(const Type* type, const std::string& id) -> std::string {
    specifiers_.clear();
    ptrOps_.clear();
    declarator_.clear();
    declarator_.append(id);

    accept(type);

    std::string buffer;

    buffer.append(specifiers_);
    buffer.append(ptrOps_);
    if (!declarator_.empty()) {
      buffer.append(" ");
      buffer.append(declarator_);
    }

    return buffer;
  }

  void accept(const Type* type) {
    if (type) visit(*this, type);
  }

  void operator()(const NullptrType* type) {
    specifiers_.append("decltype(nullptr)");
  }

  void operator()(const DecltypeAutoType* type) {
    specifiers_.append("decltype(auto)");
  }

  void operator()(const AutoType* type) { specifiers_.append("auto"); }

  void operator()(const BuiltinVaListType* type) {
    specifiers_.append("__builtin_va_list");
  }

  void operator()(const VoidType* type) { specifiers_.append("void"); }

  void operator()(const BoolType* type) { specifiers_.append("bool"); }

  void operator()(const CharType* type) { specifiers_.append("char"); }

  void operator()(const SignedCharType* type) {
    specifiers_.append("signed char");
  }

  void operator()(const UnsignedCharType* type) {
    specifiers_.append("unsigned char");
  }

  void operator()(const Char8Type* type) { specifiers_.append("char8_t"); }

  void operator()(const Char16Type* type) { specifiers_.append("char16_t"); }

  void operator()(const Char32Type* type) { specifiers_.append("char32_t"); }

  void operator()(const WideCharType* type) { specifiers_.append("wchar_t"); }

  void operator()(const ShortIntType* type) { specifiers_.append("short"); }

  void operator()(const UnsignedShortIntType* type) {
    specifiers_.append("unsigned short");
  }

  void operator()(const IntType* type) { specifiers_.append("int"); }

  void operator()(const UnsignedIntType* type) {
    specifiers_.append("unsigned int");
  }

  void operator()(const LongIntType* type) { specifiers_.append("long"); }

  void operator()(const UnsignedLongIntType* type) {
    specifiers_.append("unsigned long");
  }

  void operator()(const LongLongIntType* type) {
    specifiers_.append("long long");
  }

  void operator()(const UnsignedLongLongIntType* type) {
    specifiers_.append("unsigned long long");
  }

  void operator()(const FloatType* type) { specifiers_.append("float"); }

  void operator()(const DoubleType* type) { specifiers_.append("double"); }

  void operator()(const LongDoubleType* type) {
    specifiers_.append("long double");
  }

  void operator()(const QualType* type) {
    if (auto ptrTy = type_cast<PointerType>(type->elementType())) {
      accept(ptrTy->elementType());

      std::string op = "*";

      if (type->isConst()) {
        op += " const";
      }

      if (type->isVolatile()) {
        op += " volatile";
      }

      ptrOps_ = op + ptrOps_;

      return;
    }

    if (type->isConst()) {
      specifiers_.append("const ");
    }

    if (type->isVolatile()) {
      specifiers_.append("volatile ");
    }

    accept(type->elementType());
  }

  void operator()(const PointerType* type) {
    ptrOps_ = "*" + ptrOps_;
    accept(type->elementType());
  }

  void operator()(const LvalueReferenceType* type) {
    ptrOps_ = "&" + ptrOps_;
    accept(type->elementType());
  }

  void operator()(const RvalueReferenceType* type) {
    ptrOps_ = "&&" + ptrOps_;
    accept(type->elementType());
  }

  void operator()(const BoundedArrayType* type) {
    auto buf = "[" + std::to_string(type->size()) + "]";

    if (ptrOps_.empty()) {
      declarator_.append(buf);
    } else {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      declarator_.append(buf);
      ptrOps_.clear();
    }

    accept(type->elementType());
  }

  void operator()(const UnboundedArrayType* type) {
    std::string buf = "[]";

    if (ptrOps_.empty()) {
      declarator_.append(buf);
    } else {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      declarator_.append(buf);
      ptrOps_.clear();
    }

    accept(type->elementType());
  }

  void operator()(const OverloadSetType* type) {
    specifiers_.append("$overload-set");
  }

  void operator()(const FunctionType* type) {
    std::string signature;

    signature.append("(");

    const auto& params = type->parameterTypes();

    for (std::size_t i = 0; i < params.size(); ++i) {
      const auto& param = params[i];
      signature.append(to_string(param));

      if (i != params.size() - 1) {
        signature.append(", ");
      }
    }

    if (type->isVariadic()) {
      signature.append("...");
    }

    signature.append(")");

    switch (type->cvQualifiers()) {
      case CvQualifiers::kConst:
        signature.append(" const");
        break;
      case CvQualifiers::kVolatile:
        signature.append(" volatile");
        break;
      case CvQualifiers::kConstVolatile:
        signature.append(" const volatile");
        break;
      default:
        break;
    }  // switch

    switch (type->refQualifier()) {
      case RefQualifier::kLvalue:
        signature.append(" &");
        break;
      case RefQualifier::kRvalue:
        signature.append(" &&");
        break;
      default:
        break;
    }  // switch

    if (type->isNoexcept()) {
      signature.append(" noexcept");
    }

    if (!ptrOps_.empty()) {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      ptrOps_.clear();
    }

    declarator_.append(signature);

    accept(type->returnType());
  }

  void operator()(const ClassType* type) {
    std::string out;
    if (auto parent = type->symbol()->enclosingSymbol()) {
      accept(parent->type());
      out += "::";
    }
    out += to_string(type->symbol()->name());
    if (type->symbol()->isSpecialization()) {
      out += '<';
      std::string_view sep = "";
      for (auto arg : type->symbol()->templateArguments()) {
        auto type = std::get_if<const Type*>(&arg);
        if (!type) continue;
        out += std::format("{}{}", sep, to_string(*type));
        sep = ", ";
      }
      out += '>';
    }

    specifiers_.append(out);
  }

  void operator()(const NamespaceType* type) {
    specifiers_.append(to_string(type->symbol()->name()));
  }

  void operator()(const MemberObjectPointerType* type) {}

  void operator()(const MemberFunctionPointerType* type) {}

  void operator()(const EnumType* type) {
    specifiers_.append(to_string(type->symbol()->name()));
  }

  void operator()(const ScopedEnumType* type) {
    specifiers_.append(to_string(type->symbol()->name()));
  }

  void operator()(const TypeParameterType* type) {
    specifiers_.append(to_string(type->symbol()->name()));
  }

  void operator()(const TemplateTypeParameterType* type) {
    specifiers_.append(to_string(type->symbol()->name()));
  }

  void operator()(const UnresolvedNameType* type) {
    auto unit = type->translationUnit();
    SourceLocation first;
    if (type->nestedNameSpecifier()) {
      first = firstSourceLocation(type->nestedNameSpecifier());
    } else {
      first = firstSourceLocation(type->unqualifiedId());
    }
    auto last = lastSourceLocation(type->unqualifiedId());
    for (auto loc = first; loc != last; loc = loc.next()) {
      const auto& tk = unit->tokenAt(loc);
      if (loc != first && (tk.leadingSpace() || tk.startOfLine()))
        specifiers_ += ' ';
      specifiers_ += tk.spell();
    }
  }

  auto textOf(TranslationUnit* unit, SourceLocationRange range) const
      -> std::string {
    std::string buf;
    auto [first, last] = range;
    for (auto loc = first; loc != last; loc = loc.next()) {
      const auto& tk = unit->tokenAt(loc);
      if (loc != first && (tk.leadingSpace() || tk.startOfLine())) buf += ' ';
      buf += tk.spell();
    }
    return buf;
  }

  void operator()(const UnresolvedBoundedArrayType* type) {
    std::string buf;
    buf += '[';
    buf += textOf(type->translationUnit(), type->size()->sourceLocationRange());
    buf += ']';

    if (ptrOps_.empty()) {
      declarator_.append(buf);
    } else {
      std::string decl;
      std::swap(decl, declarator_);
      declarator_.append("(");
      declarator_.append(ptrOps_);
      declarator_.append(decl);
      declarator_.append(")");
      declarator_.append(buf);
      ptrOps_.clear();
    }

    accept(type->elementType());
  }

  void operator()(const UnresolvedUnderlyingType* type) {
    specifiers_ += "__underlying_type(";
    specifiers_ +=
        textOf(type->translationUnit(), type->typeId()->sourceLocationRange());
    specifiers_ += ")";
  }

 private:
  std::string specifiers_;
  std::string ptrOps_;
  std::string declarator_;
  bool addFormals_ = false;
};

}  // namespace

auto to_string(const Type* type, const std::string& id) -> std::string {
  if (!type) return {};
  return TypePrinter{}(type, id);
}

auto to_string(const Type* type, const Name* name) -> std::string {
  return TypePrinter{}(type, to_string(name));
}

}  // namespace cxx