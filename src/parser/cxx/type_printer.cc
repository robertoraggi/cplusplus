// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>

namespace cxx {

auto to_string(const Type* type, const std::string& id) -> std::string {
  TypePrinter printer;
  return printer.to_string(type, id);
}

TypePrinter::TypePrinter() {
  specifiers_.clear();
  ptrOps_.clear();
  declarator_.clear();
  addFormals_ = true;
}

TypePrinter::~TypePrinter() {
  specifiers_.clear();
  ptrOps_.clear();
  declarator_.clear();
}

auto TypePrinter::to_string(const Type* type, const std::string& id)
    -> std::string {
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

void TypePrinter::accept(const Type* type) {
  if (type) visit(*this, type);
}

void TypePrinter::operator()(const NullptrType* type) {
  specifiers_.append("decltype(nullptr)");
}

void TypePrinter::operator()(const DecltypeAutoType* type) {
  specifiers_.append("decltype(auto)");
}

void TypePrinter::operator()(const AutoType* type) {
  specifiers_.append("auto");
}

void TypePrinter::operator()(const VoidType* type) {
  specifiers_.append("void");
}

void TypePrinter::operator()(const BoolType* type) {
  specifiers_.append("bool");
}

void TypePrinter::operator()(const CharType* type) {
  specifiers_.append("char");
}

void TypePrinter::operator()(const SignedCharType* type) {
  specifiers_.append("signed char");
}

void TypePrinter::operator()(const UnsignedCharType* type) {
  specifiers_.append("unsigned char");
}

void TypePrinter::operator()(const Char8Type* type) {
  specifiers_.append("char8_t");
}

void TypePrinter::operator()(const Char16Type* type) {
  specifiers_.append("char16_t");
}

void TypePrinter::operator()(const Char32Type* type) {
  specifiers_.append("char32_t");
}

void TypePrinter::operator()(const WideCharType* type) {
  specifiers_.append("wchar_t");
}

void TypePrinter::operator()(const ShortIntType* type) {
  specifiers_.append("short");
}

void TypePrinter::operator()(const UnsignedShortIntType* type) {
  specifiers_.append("unsigned short");
}

void TypePrinter::operator()(const IntType* type) { specifiers_.append("int"); }

void TypePrinter::operator()(const UnsignedIntType* type) {
  specifiers_.append("unsigned int");
}

void TypePrinter::operator()(const LongIntType* type) {
  specifiers_.append("long");
}

void TypePrinter::operator()(const UnsignedLongIntType* type) {
  specifiers_.append("unsigned long");
}

void TypePrinter::operator()(const LongLongIntType* type) {
  specifiers_.append("long long");
}

void TypePrinter::operator()(const UnsignedLongLongIntType* type) {
  specifiers_.append("unsigned long long");
}

void TypePrinter::operator()(const FloatType* type) {
  specifiers_.append("float");
}

void TypePrinter::operator()(const DoubleType* type) {
  specifiers_.append("double");
}

void TypePrinter::operator()(const LongDoubleType* type) {
  specifiers_.append("long double");
}

void TypePrinter::operator()(const ClassDescriptionType* type) {
  specifiers_.append("class");
}

void TypePrinter::operator()(const QualType* type) {
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

void TypePrinter::operator()(const PointerType* type) {
  ptrOps_ = "*" + ptrOps_;
  accept(type->elementType());
}

void TypePrinter::operator()(const LvalueReferenceType* type) {
  ptrOps_ = "&" + ptrOps_;
  accept(type->elementType());
}

void TypePrinter::operator()(const RvalueReferenceType* type) {
  ptrOps_ = "&&" + ptrOps_;
  accept(type->elementType());
}

void TypePrinter::operator()(const BoundedArrayType* type) {
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

void TypePrinter::operator()(const UnboundedArrayType* type) {
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

void TypePrinter::operator()(const FunctionType* type) {
  std::string signature;

  signature.append("(");

  TypePrinter pp;

  const auto& params = type->parameterTypes();

  for (std::size_t i = 0; i < params.size(); ++i) {
    const auto& param = params[i];
    signature.append(pp.to_string(param));

    if (i != params.size() - 1) {
      signature.append(", ");
    }
  }

  if (type->isVariadic()) {
    signature.append("...");
  }

  signature.append(")");

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

void TypePrinter::operator()(const ClassType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol()->name());
  specifiers_.append(className->name());
}

void TypePrinter::operator()(const UnionType* type) {
  const Identifier* unionName = name_cast<Identifier>(type->symbol()->name());
  specifiers_.append(unionName->name());
}

void TypePrinter::operator()(const NamespaceType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol()->name());
  specifiers_.append(className->name());
}

void TypePrinter::operator()(const MemberObjectPointerType* type) {}

void TypePrinter::operator()(const MemberFunctionPointerType* type) {}

void TypePrinter::operator()(const EnumType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol()->name());
  specifiers_.append(className->name());
}

void TypePrinter::operator()(const ScopedEnumType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol()->name());
  specifiers_.append(className->name());
}

}  // namespace cxx