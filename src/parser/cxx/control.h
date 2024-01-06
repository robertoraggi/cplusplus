// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/types_fwd.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cxx {

class MemoryLayout;

class Control {
 public:
  Control();
  ~Control();

  [[nodiscard]] auto memoryLayout() const -> MemoryLayout*;
  void setMemoryLayout(MemoryLayout* memoryLayout);

  auto integerLiteral(std::string_view spelling) -> const IntegerLiteral*;
  auto floatLiteral(std::string_view spelling) -> const FloatLiteral*;
  auto stringLiteral(std::string_view spelling) -> const StringLiteral*;
  auto charLiteral(std::string_view spelling) -> const CharLiteral*;
  auto wideStringLiteral(std::string_view spelling) -> const WideStringLiteral*;
  auto utf8StringLiteral(std::string_view spelling) -> const Utf8StringLiteral*;
  auto utf16StringLiteral(std::string_view spelling)
      -> const Utf16StringLiteral*;
  auto utf32StringLiteral(std::string_view spelling)
      -> const Utf32StringLiteral*;
  auto commentLiteral(std::string_view spelling) -> const CommentLiteral*;

  auto newAnonymousId(std::string_view base) -> const Identifier*;
  auto getIdentifier(std::string_view name) -> const Identifier*;
  auto getOperatorId(TokenKind op) -> const OperatorId*;
  auto getDestructorId(const Name*) -> const DestructorId*;
  auto getLiteralOperatorId(std::string_view name) -> const LiteralOperatorId*;
  auto getConversionFunctionId(const Type* type) -> const ConversionFunctionId*;
  auto getTemplateId(const Name* name, std::vector<TemplateArgument> arguments)
      -> const TemplateId*;

  auto getSizeType() -> const Type*;

  auto getVoidType() -> const VoidType*;
  auto getNullptrType() -> const NullptrType*;
  auto getDecltypeAutoType() -> const DecltypeAutoType*;
  auto getAutoType() -> const AutoType*;
  auto getBoolType() -> const BoolType*;
  auto getSignedCharType() -> const SignedCharType*;
  auto getShortIntType() -> const ShortIntType*;
  auto getIntType() -> const IntType*;
  auto getLongIntType() -> const LongIntType*;
  auto getLongLongIntType() -> const LongLongIntType*;
  auto getUnsignedCharType() -> const UnsignedCharType*;
  auto getUnsignedShortIntType() -> const UnsignedShortIntType*;
  auto getUnsignedIntType() -> const UnsignedIntType*;
  auto getUnsignedLongIntType() -> const UnsignedLongIntType*;
  auto getUnsignedLongLongIntType() -> const UnsignedLongLongIntType*;
  auto getCharType() -> const CharType*;
  auto getChar8Type() -> const Char8Type*;
  auto getChar16Type() -> const Char16Type*;
  auto getChar32Type() -> const Char32Type*;
  auto getWideCharType() -> const WideCharType*;
  auto getFloatType() -> const FloatType*;
  auto getDoubleType() -> const DoubleType*;
  auto getLongDoubleType() -> const LongDoubleType*;
  auto getQualType(const Type* elementType, CvQualifiers cvQualifiers)
      -> const QualType*;
  auto getConstType(const Type* elementType) -> const QualType*;
  auto getVolatileType(const Type* elementType) -> const QualType*;
  auto getConstVolatileType(const Type* elementType) -> const QualType*;
  auto getBoundedArrayType(const Type* elementType, std::size_t size)
      -> const BoundedArrayType*;
  auto getUnboundedArrayType(const Type* elementType)
      -> const UnboundedArrayType*;
  auto getPointerType(const Type* elementType) -> const PointerType*;
  auto getLvalueReferenceType(const Type* elementType)
      -> const LvalueReferenceType*;
  auto getRvalueReferenceType(const Type* elementType)
      -> const RvalueReferenceType*;
  auto getOverloadSetType(OverloadSetSymbol* symbol) -> const OverloadSetType*;
  auto getFunctionType(const Type* returnType,
                       std::vector<const Type*> parameterTypes,
                       bool isVariadic = false,
                       CvQualifiers cvQualifiers = CvQualifiers::kNone,
                       RefQualifier refQualifier = RefQualifier::kNone,
                       bool isNoexcept = false) -> const FunctionType*;
  auto getMemberObjectPointerType(const ClassType* classType,
                                  const Type* elementType)
      -> const MemberObjectPointerType*;
  auto getMemberFunctionPointerType(const ClassType* classType,
                                    const FunctionType* functionType)
      -> const MemberFunctionPointerType*;
  auto getTypeParameterType(TypeParameterSymbol* symbol)
      -> const TypeParameterType*;
  auto getUnresolvedNameType(TranslationUnit* unit,
                             NestedNameSpecifierAST* nestedNameSpecifier,
                             UnqualifiedIdAST* unqualifiedId)
      -> const UnresolvedNameType*;
  auto getUnresolvedBoundedArrayType(TranslationUnit* unit,
                                     const Type* elementType,
                                     ExpressionAST* sizeExpression)
      -> const UnresolvedBoundedArrayType*;
  auto getUnresolvedUnderlyingType(TranslationUnit* unit, TypeIdAST* typeId)
      -> const UnresolvedUnderlyingType*;

  auto getClassType(ClassSymbol* symbol) -> const ClassType*;
  auto getNamespaceType(NamespaceSymbol* symbol) -> const NamespaceType*;
  auto getEnumType(EnumSymbol* symbol) -> const EnumType*;
  auto getScopedEnumType(ScopedEnumSymbol* symbol) -> const ScopedEnumType*;

  auto newNamespaceSymbol(Scope* enclosingScope) -> NamespaceSymbol*;
  auto newConceptSymbol(Scope* enclosingScope) -> ConceptSymbol*;
  auto newBaseClassSymbol(Scope* enclosingScope) -> BaseClassSymbol*;
  auto newClassSymbol(Scope* enclosingScope) -> ClassSymbol*;
  auto newEnumSymbol(Scope* enclosingScope) -> EnumSymbol*;
  auto newScopedEnumSymbol(Scope* enclosingScope) -> ScopedEnumSymbol*;
  auto newOverloadSetSymbol(Scope* enclosingScope) -> OverloadSetSymbol*;
  auto newFunctionSymbol(Scope* enclosingScope) -> FunctionSymbol*;
  auto newLambdaSymbol(Scope* enclosingScope) -> LambdaSymbol*;
  auto newFunctionParametersSymbol(Scope* enclosingScope)
      -> FunctionParametersSymbol*;
  auto newTemplateParametersSymbol(Scope* enclosingScope)
      -> TemplateParametersSymbol*;
  auto newBlockSymbol(Scope* enclosingScope) -> BlockSymbol*;
  auto newTypeAliasSymbol(Scope* enclosingScope) -> TypeAliasSymbol*;
  auto newVariableSymbol(Scope* enclosingScope) -> VariableSymbol*;
  auto newFieldSymbol(Scope* enclosingScope) -> FieldSymbol*;
  auto newParameterSymbol(Scope* enclosingScope) -> ParameterSymbol*;
  auto newTypeParameterSymbol(Scope* enclosingScope) -> TypeParameterSymbol*;
  auto newNonTypeParameterSymbol(Scope* enclosingScope)
      -> NonTypeParameterSymbol*;
  auto newTemplateTypeParameterSymbol(Scope* enclosingScope)
      -> TemplateTypeParameterSymbol*;
  auto newConstraintTypeParameterSymbol(Scope* enclosingScope)
      -> ConstraintTypeParameterSymbol*;
  auto newEnumeratorSymbol(Scope* enclosingScope) -> EnumeratorSymbol*;

  // primary type categories
  auto is_void(const Type* type) -> bool;
  auto is_null_pointer(const Type* type) -> bool;
  auto is_integral(const Type* type) -> bool;
  auto is_floating_point(const Type* type) -> bool;
  auto is_array(const Type* type) -> bool;
  auto is_enum(const Type* type) -> bool;
  auto is_union(const Type* type) -> bool;
  auto is_class(const Type* type) -> bool;
  auto is_function(const Type* type) -> bool;
  auto is_pointer(const Type* type) -> bool;
  auto is_lvalue_reference(const Type* type) -> bool;
  auto is_rvalue_reference(const Type* type) -> bool;
  auto is_member_object_pointer(const Type* type) -> bool;
  auto is_member_function_pointer(const Type* type) -> bool;
  auto is_complete(const Type* type) -> bool;

  // composite type categories
  auto is_integer(const Type* type) -> bool;
  auto is_integral_or_unscoped_enum(const Type* type) -> bool;
  auto is_arithmetic_or_unscoped_enum(const Type* type) -> bool;
  auto is_fundamental(const Type* type) -> bool;
  auto is_arithmetic(const Type* type) -> bool;
  auto is_scalar(const Type* type) -> bool;
  auto is_object(const Type* type) -> bool;
  auto is_compound(const Type* type) -> bool;
  auto is_reference(const Type* type) -> bool;
  auto is_member_pointer(const Type* type) -> bool;
  auto is_class_or_union(const Type* type) -> bool;

  // type properties
  auto is_const(const Type* type) -> bool;
  auto is_volatile(const Type* type) -> bool;
  auto is_signed(const Type* type) -> bool;
  auto is_unsigned(const Type* type) -> bool;
  auto is_bounded_array(const Type* type) -> bool;
  auto is_unbounded_array(const Type* type) -> bool;
  auto is_scoped_enum(const Type* type) -> bool;

  // references
  auto remove_reference(const Type* type) -> const Type*;
  auto add_lvalue_reference(const Type* type) -> const Type*;
  auto add_rvalue_reference(const Type* type) -> const Type*;

  // arrays
  auto remove_extent(const Type* type) -> const Type*;

  // cv qualifiers
  auto remove_cv(const Type* type) -> const Type*;
  auto remove_cvref(const Type* type) -> const Type*;
  auto add_const_ref(const Type* type) -> const Type*;
  auto add_const(const Type* type) -> const Type*;
  auto add_volatile(const Type* type) -> const Type*;
  auto get_cv_qualifiers(const Type* type) -> CvQualifiers;

  // pointers
  auto remove_pointer(const Type* type) -> const Type*;
  auto add_pointer(const Type* type) -> const Type*;

  // functions
  auto remove_noexcept(const Type* type) -> const Type*;

  // classes
  auto is_base_of(const Type* base, const Type* derived) -> bool;

  // type relationships
  auto is_same(const Type* a, const Type* b) -> bool;
  auto decay(const Type* type) -> const Type*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx