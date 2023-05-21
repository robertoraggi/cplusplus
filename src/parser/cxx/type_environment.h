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

  auto undefinedType() -> const UndefinedType*;

  auto errorType() -> const ErrorType*;

  auto autoType() -> const AutoType*;

  auto decltypeAuto() -> const DecltypeAutoType*;

  auto voidType() -> const VoidType*;

  auto nullptrType() -> const NullptrType*;

  auto booleanType() -> const BooleanType*;

  auto characterType(CharacterKind kind) -> const CharacterType*;

  auto integerType(IntegerKind kind, bool isUnsigned) -> const IntegerType*;

  auto floatingPointType(FloatingPointKind kind) -> const FloatingPointType*;

  auto enumType(EnumSymbol* symbol) -> const EnumType*;

  auto scopedEnumType(ScopedEnumSymbol* symbol) -> const ScopedEnumType*;

  auto pointerType(const QualifiedType& elementType, Qualifiers qualifiers)
      -> const PointerType*;

  auto pointerToMemberType(const ClassType* classType,
                           const QualifiedType& elementType,
                           Qualifiers qualifiers) -> const PointerToMemberType*;

  auto referenceType(const QualifiedType& elementType) -> const ReferenceType*;

  auto rvalueReferenceType(const QualifiedType& elementType)
      -> const RValueReferenceType*;

  auto arrayType(const QualifiedType& elementType, std::size_t dimension)
      -> const ArrayType*;

  auto unboundArrayType(const QualifiedType& elementType)
      -> const UnboundArrayType*;

  auto functionType(const QualifiedType& returnType,
                    std::vector<QualifiedType> argumentTypes, bool isVariadic)
      -> const FunctionType*;

  auto memberFunctionType(const ClassType* classType,
                          const QualifiedType& returnType,
                          std::vector<QualifiedType> argumentTypes,
                          bool isVariadic) -> const MemberFunctionType*;

  auto namespaceType(NamespaceSymbol* symbol) -> const NamespaceType*;

  auto classType(ClassSymbol* symbol) -> const ClassType*;

  auto templateType(TemplateParameterList* symbol) -> const TemplateType*;

  auto templateArgumentType(TemplateTypeParameterSymbol* symbol)
      -> const TemplateArgumentType*;

  auto conceptType(ConceptSymbol* symbol) -> const ConceptType*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx