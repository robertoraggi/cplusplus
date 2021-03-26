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

#pragma once

#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <memory>
#include <vector>

namespace cxx {

class TypeEnvironment {
 public:
  TypeEnvironment();
  ~TypeEnvironment();

  const UndefinedType* undefinedType();

  const ErrorType* errorType();

  const UnresolvedType* unresolvedType();

  const VoidType* voidType();

  const NullptrType* nullptrType();

  const BooleanType* booleanType();

  const CharacterType* characterType(CharacterKind kind);

  const IntegerType* integerType(IntegerKind kind, bool isUnsigned);

  const FloatingPointType* floatingPointType(FloatingPointKind kind);

  const EnumType* enumType(EnumSymbol* symbol);

  const ScopedEnumType* scopedEnumType(ScopedEnumSymbol* symbol);

  const PointerType* pointerType(const QualifiedType& elementType,
                                 Qualifiers qualifiers);

  const PointerToMemberType* pointerToMemberType(
      const ClassType* classType, const QualifiedType& elementType,
      Qualifiers qualifiers);

  const ReferenceType* referenceType(const QualifiedType& elementType);

  const RValueReferenceType* rvalueReferenceType(
      const QualifiedType& elementType);

  const ArrayType* arrayType(const QualifiedType& elementType,
                             std::size_t dimension);

  const UnboundArrayType* unboundArrayType(const QualifiedType& elementType);

  const FunctionType* functionType(const QualifiedType& returnType,
                                   std::vector<QualifiedType> argumentTypes,
                                   bool isVariadic);

  const MemberFunctionType* memberFunctionType(
      const ClassType* classType, const QualifiedType& returnType,
      std::vector<QualifiedType> argumentTypes, bool isVariadic);

  const NamespaceType* namespaceType(NamespaceSymbol* symbol);

  const ClassType* classType(ClassSymbol* symbol);

  const TemplateType* templateType(TemplateSymbol* symbol);

  const TemplateArgumentType* templateArgumentType(
      TemplateArgumentSymbol* symbol);

  const ConceptType* conceptType(ConceptSymbol* symbol);

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx