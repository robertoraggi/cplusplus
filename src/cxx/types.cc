// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/type_visitor.h>
#include <cxx/types.h>

namespace cxx {

Type::~Type() {}

const UndefinedType* UndefinedType::get() {
  static UndefinedType type;
  return &type;
}

const ErrorType* ErrorType::get() {
  static ErrorType type;
  return &type;
}

const AutoType* AutoType::get() {
  static AutoType type;
  return &type;
}

const DecltypeAutoType* DecltypeAutoType::get() {
  static DecltypeAutoType type;
  return &type;
}

const VoidType* VoidType::get() {
  static VoidType type;
  return &type;
}

const NullptrType* NullptrType::get() {
  static NullptrType type;
  return &type;
}

const BooleanType* BooleanType::get() {
  static BooleanType type;
  return &type;
}

void UndefinedType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void ErrorType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void AutoType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void DecltypeAutoType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void VoidType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void NullptrType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void BooleanType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void CharacterType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void IntegerType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void FloatingPointType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void EnumType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void ScopedEnumType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void PointerType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void PointerToMemberType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void ReferenceType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void RValueReferenceType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void ArrayType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void UnboundArrayType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void FunctionType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void MemberFunctionType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void NamespaceType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void ClassType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void TemplateType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void TemplateArgumentType::accept(TypeVisitor* visitor) const {
  visitor->visit(this);
}

void ConceptType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

}  // namespace cxx