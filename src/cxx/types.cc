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

const ErrorType* Type::asErrorType() const {
  return dynamic_cast<const ErrorType*>(this);
}

const UnresolvedType* Type::asUnresolvedType() const {
  return dynamic_cast<const UnresolvedType*>(this);
}

const VoidType* Type::asVoidType() const {
  return dynamic_cast<const VoidType*>(this);
}

const NullptrType* Type::asNullptrType() const {
  return dynamic_cast<const NullptrType*>(this);
}

const BooleanType* Type::asBooleanType() const {
  return dynamic_cast<const BooleanType*>(this);
}

const CharacterType* Type::asCharacterType() const {
  return dynamic_cast<const CharacterType*>(this);
}

const IntegerType* Type::asIntegerType() const {
  return dynamic_cast<const IntegerType*>(this);
}

const FloatingPointType* Type::asFloatingPointType() const {
  return dynamic_cast<const FloatingPointType*>(this);
}

const EnumType* Type::asEnumType() const {
  return dynamic_cast<const EnumType*>(this);
}

const ScopedEnumType* Type::asScopedEnumType() const {
  return dynamic_cast<const ScopedEnumType*>(this);
}

const PointerType* Type::asPointerType() const {
  return dynamic_cast<const PointerType*>(this);
}

const PointerToMemberType* Type::asPointerToMemberType() const {
  return dynamic_cast<const PointerToMemberType*>(this);
}

const ReferenceType* Type::asReferenceType() const {
  return dynamic_cast<const ReferenceType*>(this);
}

const RValueReferenceType* Type::asRValueReferenceType() const {
  return dynamic_cast<const RValueReferenceType*>(this);
}

const ArrayType* Type::asArrayType() const {
  return dynamic_cast<const ArrayType*>(this);
}

const UnboundArrayType* Type::asUnboundArrayType() const {
  return dynamic_cast<const UnboundArrayType*>(this);
}

const FunctionType* Type::asFunctionType() const {
  return dynamic_cast<const FunctionType*>(this);
}

const MemberFunctionType* Type::asMemberFunctionType() const {
  return dynamic_cast<const MemberFunctionType*>(this);
}

const NamespaceType* Type::asNamespaceType() const {
  return dynamic_cast<const NamespaceType*>(this);
}

const ClassType* Type::asClassType() const {
  return dynamic_cast<const ClassType*>(this);
}

const TemplateType* Type::asTemplateType() const {
  return dynamic_cast<const TemplateType*>(this);
}

const TemplateArgumentType* Type::asTemplateArgumentType() const {
  return dynamic_cast<const TemplateArgumentType*>(this);
}

const ConceptType* Type::asConceptType() const {
  return dynamic_cast<const ConceptType*>(this);
}

const UndefinedType* UndefinedType::get() {
  static UndefinedType type;
  return &type;
}

const ErrorType* ErrorType::get() {
  static ErrorType type;
  return &type;
}

void UndefinedType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void ErrorType::accept(TypeVisitor* visitor) const { visitor->visit(this); }

void UnresolvedType::accept(TypeVisitor* visitor) const {
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