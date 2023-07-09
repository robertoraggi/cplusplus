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
  buffer.clear();

  buffer.append(specifiers_);
  buffer.append(ptrOps_);
  if (!declarator_.empty()) {
    buffer.append(" ");
    buffer.append(declarator_);
  }

  return buffer;
}

auto TypePrinter::to_string(const Type* type) -> std::string {
  return to_string(type, nullptr);
}

void TypePrinter::accept(const Type* type) {
  if (type) type->accept(this);
}

void TypePrinter::visit(const InvalidType* type) {}

void TypePrinter::visit(const NullptrType* type) {
  specifiers_.append("decltype(nullptr)");
}

void TypePrinter::visit(const DependentType* type) {
  specifiers_.append("dependent-type");
}

void TypePrinter::visit(const AutoType* type) { specifiers_.append("auto"); }

void TypePrinter::visit(const VoidType* type) { specifiers_.append("void"); }

void TypePrinter::visit(const BoolType* type) { specifiers_.append("bool"); }

void TypePrinter::visit(const CharType* type) { specifiers_.append("char"); }

void TypePrinter::visit(const SignedCharType* type) {
  specifiers_.append("signed char");
}

void TypePrinter::visit(const UnsignedCharType* type) {
  specifiers_.append("unsigned char");
}

void TypePrinter::visit(const ShortType* type) { specifiers_.append("short"); }

void TypePrinter::visit(const UnsignedShortType* type) {
  specifiers_.append("unsigned short");
}

void TypePrinter::visit(const IntType* type) { specifiers_.append("int"); }

void TypePrinter::visit(const UnsignedIntType* type) {
  specifiers_.append("unsigned int");
}

void TypePrinter::visit(const LongType* type) { specifiers_.append("long"); }

void TypePrinter::visit(const UnsignedLongType* type) {
  specifiers_.append("unsigned long");
}

void TypePrinter::visit(const FloatType* type) { specifiers_.append("float"); }

void TypePrinter::visit(const DoubleType* type) {
  specifiers_.append("double");
}

void TypePrinter::visit(const QualType* type) {
  if (type->isConst) {
    specifiers_.append("const ");
  }
  if (type->isVolatile) {
    specifiers_.append("volatile ");
  }
  accept(type->elementType);
}

void TypePrinter::visit(const PointerType* type) {
  ptrOps_.append("*");
  accept(type->elementType);
}

void TypePrinter::visit(const LValueReferenceType* type) {
  ptrOps_.append("&");
  accept(type->elementType);
}

void TypePrinter::visit(const RValueReferenceType* type) {
  ptrOps_.append("&&");
  accept(type->elementType);
}

void TypePrinter::visit(const ArrayType* type) {
  auto buf = "[" + std::to_string(type->dim) + "]";

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

  accept(type->elementType);
}

void TypePrinter::visit(const FunctionType* type) {
  std::string signature;
  signature.clear();

  signature.append("(");

  TypePrinter pp;

  for (std::size_t i = 0; i < type->parameters.size(); ++i) {
    const auto& param = type->parameters[i];
    const Identifier* paramName = name_cast<Identifier>(param.name());

    std::string paramId;

    if (addFormals_ && paramName) {
      paramId = paramName->name();
    }

    auto paramText = pp.to_string(param.type(), paramId);

    signature.append(paramText);

    if (i != type->parameters.size() - 1) {
      signature.append(", ");
    }
  }

  if (type->isVariadic) {
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

  accept(type->returnType);
}

void TypePrinter::visit(const ClassType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol->name());
  specifiers_.append(className->name());
}

void TypePrinter::visit(const NamespaceType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol->name());
  specifiers_.append(className->name());
}

void TypePrinter::visit(const MemberPointerType* type) {}

void TypePrinter::visit(const ConceptType* type) {}

void TypePrinter::visit(const EnumType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol->name());
  specifiers_.append(className->name());
}

void TypePrinter::visit(const GenericType* type) {}

void TypePrinter::visit(const PackType* type) {}

void TypePrinter::visit(const ScopedEnumType* type) {
  const Identifier* className = name_cast<Identifier>(type->symbol->name());
  specifiers_.append(className->name());
}

}  // namespace cxx