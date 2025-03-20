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

#pragma once

#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
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

  [[nodiscard]] auto integerLiteral(std::string_view spelling)
      -> const IntegerLiteral*;
  [[nodiscard]] auto floatLiteral(std::string_view spelling)
      -> const FloatLiteral*;
  [[nodiscard]] auto stringLiteral(std::string_view spelling)
      -> const StringLiteral*;
  [[nodiscard]] auto charLiteral(std::string_view spelling)
      -> const CharLiteral*;
  [[nodiscard]] auto wideStringLiteral(std::string_view spelling)
      -> const WideStringLiteral*;
  [[nodiscard]] auto utf8StringLiteral(std::string_view spelling)
      -> const Utf8StringLiteral*;
  [[nodiscard]] auto utf16StringLiteral(std::string_view spelling)
      -> const Utf16StringLiteral*;
  [[nodiscard]] auto utf32StringLiteral(std::string_view spelling)
      -> const Utf32StringLiteral*;
  [[nodiscard]] auto commentLiteral(std::string_view spelling)
      -> const CommentLiteral*;

  [[nodiscard]] auto newAnonymousId(std::string_view base) -> const Identifier*;
  [[nodiscard]] auto getIdentifier(std::string_view name) -> const Identifier*;
  [[nodiscard]] auto getOperatorId(TokenKind op) -> const OperatorId*;
  [[nodiscard]] auto getDestructorId(const Name*) -> const DestructorId*;
  [[nodiscard]] auto getLiteralOperatorId(std::string_view name)
      -> const LiteralOperatorId*;
  [[nodiscard]] auto getConversionFunctionId(const Type* type)
      -> const ConversionFunctionId*;
  [[nodiscard]] auto getTemplateId(const Name* name,
                                   std::vector<TemplateArgument> arguments)
      -> const TemplateId*;

  [[nodiscard]] auto getSizeType() -> const Type*;

  [[nodiscard]] auto getBuiltinVaListType() -> const BuiltinVaListType*;
  [[nodiscard]] auto getVoidType() -> const VoidType*;
  [[nodiscard]] auto getNullptrType() -> const NullptrType*;
  [[nodiscard]] auto getDecltypeAutoType() -> const DecltypeAutoType*;
  [[nodiscard]] auto getAutoType() -> const AutoType*;
  [[nodiscard]] auto getBoolType() -> const BoolType*;
  [[nodiscard]] auto getSignedCharType() -> const SignedCharType*;
  [[nodiscard]] auto getShortIntType() -> const ShortIntType*;
  [[nodiscard]] auto getIntType() -> const IntType*;
  [[nodiscard]] auto getLongIntType() -> const LongIntType*;
  [[nodiscard]] auto getLongLongIntType() -> const LongLongIntType*;
  [[nodiscard]] auto getUnsignedCharType() -> const UnsignedCharType*;
  [[nodiscard]] auto getUnsignedShortIntType() -> const UnsignedShortIntType*;
  [[nodiscard]] auto getUnsignedIntType() -> const UnsignedIntType*;
  [[nodiscard]] auto getUnsignedLongIntType() -> const UnsignedLongIntType*;
  [[nodiscard]] auto getUnsignedLongLongIntType()
      -> const UnsignedLongLongIntType*;
  [[nodiscard]] auto getCharType() -> const CharType*;
  [[nodiscard]] auto getChar8Type() -> const Char8Type*;
  [[nodiscard]] auto getChar16Type() -> const Char16Type*;
  [[nodiscard]] auto getChar32Type() -> const Char32Type*;
  [[nodiscard]] auto getWideCharType() -> const WideCharType*;
  [[nodiscard]] auto getFloatType() -> const FloatType*;
  [[nodiscard]] auto getDoubleType() -> const DoubleType*;
  [[nodiscard]] auto getLongDoubleType() -> const LongDoubleType*;
  [[nodiscard]] auto getQualType(const Type* elementType,
                                 CvQualifiers cvQualifiers) -> const QualType*;
  [[nodiscard]] auto getConstType(const Type* elementType) -> const QualType*;
  [[nodiscard]] auto getVolatileType(const Type* elementType)
      -> const QualType*;
  [[nodiscard]] auto getConstVolatileType(const Type* elementType)
      -> const QualType*;
  [[nodiscard]] auto getBoundedArrayType(const Type* elementType,
                                         std::size_t size)
      -> const BoundedArrayType*;
  [[nodiscard]] auto getUnboundedArrayType(const Type* elementType)
      -> const UnboundedArrayType*;
  [[nodiscard]] auto getPointerType(const Type* elementType)
      -> const PointerType*;
  [[nodiscard]] auto getLvalueReferenceType(const Type* elementType)
      -> const LvalueReferenceType*;
  [[nodiscard]] auto getRvalueReferenceType(const Type* elementType)
      -> const RvalueReferenceType*;
  [[nodiscard]] auto getOverloadSetType(OverloadSetSymbol* symbol)
      -> const OverloadSetType*;
  [[nodiscard]] auto getFunctionType(
      const Type* returnType, std::vector<const Type*> parameterTypes,
      bool isVariadic = false, CvQualifiers cvQualifiers = CvQualifiers::kNone,
      RefQualifier refQualifier = RefQualifier::kNone, bool isNoexcept = false)
      -> const FunctionType*;
  [[nodiscard]] auto getMemberObjectPointerType(const ClassType* classType,
                                                const Type* elementType)
      -> const MemberObjectPointerType*;
  [[nodiscard]] auto getMemberFunctionPointerType(
      const ClassType* classType, const FunctionType* functionType)
      -> const MemberFunctionPointerType*;
  [[nodiscard]] auto getTypeParameterType(TypeParameterSymbol* symbol)
      -> const TypeParameterType*;
  [[nodiscard]] auto getTemplateTypeParameterType(
      TemplateTypeParameterSymbol* symbol) -> const TemplateTypeParameterType*;
  [[nodiscard]] auto getUnresolvedNameType(
      TranslationUnit* unit, NestedNameSpecifierAST* nestedNameSpecifier,
      UnqualifiedIdAST* unqualifiedId) -> const UnresolvedNameType*;
  [[nodiscard]] auto getUnresolvedBoundedArrayType(
      TranslationUnit* unit, const Type* elementType,
      ExpressionAST* sizeExpression) -> const UnresolvedBoundedArrayType*;
  [[nodiscard]] auto getUnresolvedUnderlyingType(TranslationUnit* unit,
                                                 TypeIdAST* typeId)
      -> const UnresolvedUnderlyingType*;

  [[nodiscard]] auto getClassType(ClassSymbol* symbol) -> const ClassType*;
  [[nodiscard]] auto getNamespaceType(NamespaceSymbol* symbol)
      -> const NamespaceType*;
  [[nodiscard]] auto getEnumType(EnumSymbol* symbol) -> const EnumType*;
  [[nodiscard]] auto getScopedEnumType(ScopedEnumSymbol* symbol)
      -> const ScopedEnumType*;

  [[nodiscard]] auto newNamespaceSymbol(Scope* enclosingScope,
                                        SourceLocation sourceLocation)
      -> NamespaceSymbol*;
  [[nodiscard]] auto newConceptSymbol(Scope* enclosingScope,
                                      SourceLocation sourceLocation)
      -> ConceptSymbol*;
  [[nodiscard]] auto newBaseClassSymbol(Scope* enclosingScope,
                                        SourceLocation sourceLocation)
      -> BaseClassSymbol*;
  [[nodiscard]] auto newClassSymbol(Scope* enclosingScope,
                                    SourceLocation sourceLocation)
      -> ClassSymbol*;
  [[nodiscard]] auto newEnumSymbol(Scope* enclosingScope,
                                   SourceLocation sourceLocation)
      -> EnumSymbol*;
  [[nodiscard]] auto newScopedEnumSymbol(Scope* enclosingScope,
                                         SourceLocation sourceLocation)
      -> ScopedEnumSymbol*;
  [[nodiscard]] auto newOverloadSetSymbol(Scope* enclosingScope,
                                          SourceLocation sourceLocation)
      -> OverloadSetSymbol*;
  [[nodiscard]] auto newFunctionSymbol(Scope* enclosingScope,
                                       SourceLocation sourceLocation)
      -> FunctionSymbol*;
  [[nodiscard]] auto newLambdaSymbol(Scope* enclosingScope,
                                     SourceLocation sourceLocation)
      -> LambdaSymbol*;
  [[nodiscard]] auto newFunctionParametersSymbol(Scope* enclosingScope,
                                                 SourceLocation sourceLocation)
      -> FunctionParametersSymbol*;
  [[nodiscard]] auto newTemplateParametersSymbol(Scope* enclosingScope,
                                                 SourceLocation sourceLocation)
      -> TemplateParametersSymbol*;
  [[nodiscard]] auto newBlockSymbol(Scope* enclosingScope,
                                    SourceLocation sourceLocation)
      -> BlockSymbol*;
  [[nodiscard]] auto newTypeAliasSymbol(Scope* enclosingScope,
                                        SourceLocation sourceLocation)
      -> TypeAliasSymbol*;
  [[nodiscard]] auto newVariableSymbol(Scope* enclosingScope,
                                       SourceLocation sourceLocation)
      -> VariableSymbol*;
  [[nodiscard]] auto newFieldSymbol(Scope* enclosingScope,
                                    SourceLocation sourceLocation)
      -> FieldSymbol*;
  [[nodiscard]] auto newParameterSymbol(Scope* enclosingScope,
                                        SourceLocation sourceLocation)
      -> ParameterSymbol*;
  [[nodiscard]] auto newParameterPackSymbol(Scope* enclosingScope,
                                            SourceLocation sourceLocation)
      -> ParameterPackSymbol*;
  [[nodiscard]] auto newTypeParameterSymbol(Scope* enclosingScope,
                                            SourceLocation sourceLocation)
      -> TypeParameterSymbol*;
  [[nodiscard]] auto newNonTypeParameterSymbol(Scope* enclosingScope,
                                               SourceLocation sourceLocation)
      -> NonTypeParameterSymbol*;
  [[nodiscard]] auto newTemplateTypeParameterSymbol(
      Scope* enclosingScope, SourceLocation sourceLocation)
      -> TemplateTypeParameterSymbol*;
  [[nodiscard]] auto newConstraintTypeParameterSymbol(
      Scope* enclosingScope, SourceLocation sourceLocation)
      -> ConstraintTypeParameterSymbol*;
  [[nodiscard]] auto newEnumeratorSymbol(Scope* enclosingScope,
                                         SourceLocation sourceLocation)
      -> EnumeratorSymbol*;
  [[nodiscard]] auto newUsingDeclarationSymbol(Scope* enclosingScope,
                                               SourceLocation sourceLocation)
      -> UsingDeclarationSymbol*;

  [[nodiscard]] auto instantiate(TranslationUnit* unit, Symbol* primaryTemplate,
                                 const std::vector<TemplateArgument>& arguments)
      -> Symbol*;

  // primary type categories
  [[nodiscard]] auto is_void(const Type* type) -> bool;
  [[nodiscard]] auto is_null_pointer(const Type* type) -> bool;
  [[nodiscard]] auto is_integral(const Type* type) -> bool;
  [[nodiscard]] auto is_floating_point(const Type* type) -> bool;
  [[nodiscard]] auto is_array(const Type* type) -> bool;
  [[nodiscard]] auto is_enum(const Type* type) -> bool;
  [[nodiscard]] auto is_union(const Type* type) -> bool;
  [[nodiscard]] auto is_class(const Type* type) -> bool;
  [[nodiscard]] auto is_function(const Type* type) -> bool;
  [[nodiscard]] auto is_pointer(const Type* type) -> bool;
  [[nodiscard]] auto is_lvalue_reference(const Type* type) -> bool;
  [[nodiscard]] auto is_rvalue_reference(const Type* type) -> bool;
  [[nodiscard]] auto is_member_object_pointer(const Type* type) -> bool;
  [[nodiscard]] auto is_member_function_pointer(const Type* type) -> bool;
  [[nodiscard]] auto is_complete(const Type* type) -> bool;

  // composite type categories
  [[nodiscard]] auto is_integer(const Type* type) -> bool;
  [[nodiscard]] auto is_integral_or_unscoped_enum(const Type* type) -> bool;
  [[nodiscard]] auto is_arithmetic_or_unscoped_enum(const Type* type) -> bool;
  [[nodiscard]] auto is_fundamental(const Type* type) -> bool;
  [[nodiscard]] auto is_arithmetic(const Type* type) -> bool;
  [[nodiscard]] auto is_scalar(const Type* type) -> bool;
  [[nodiscard]] auto is_object(const Type* type) -> bool;
  [[nodiscard]] auto is_compound(const Type* type) -> bool;
  [[nodiscard]] auto is_reference(const Type* type) -> bool;
  [[nodiscard]] auto is_member_pointer(const Type* type) -> bool;
  [[nodiscard]] auto is_class_or_union(const Type* type) -> bool;

  // type properties
  [[nodiscard]] auto is_const(const Type* type) -> bool;
  [[nodiscard]] auto is_volatile(const Type* type) -> bool;
  [[nodiscard]] auto is_signed(const Type* type) -> bool;
  [[nodiscard]] auto is_unsigned(const Type* type) -> bool;
  [[nodiscard]] auto is_bounded_array(const Type* type) -> bool;
  [[nodiscard]] auto is_unbounded_array(const Type* type) -> bool;
  [[nodiscard]] auto is_scoped_enum(const Type* type) -> bool;

  // references
  [[nodiscard]] auto remove_reference(const Type* type) -> const Type*;
  [[nodiscard]] auto add_lvalue_reference(const Type* type) -> const Type*;
  [[nodiscard]] auto add_rvalue_reference(const Type* type) -> const Type*;

  // arrays
  [[nodiscard]] auto remove_extent(const Type* type) -> const Type*;

  // cv qualifiers
  [[nodiscard]] auto remove_cv(const Type* type) -> const Type*;
  [[nodiscard]] auto remove_cvref(const Type* type) -> const Type*;
  [[nodiscard]] auto add_const_ref(const Type* type) -> const Type*;
  [[nodiscard]] auto add_const(const Type* type) -> const Type*;
  [[nodiscard]] auto add_volatile(const Type* type) -> const Type*;
  [[nodiscard]] auto get_cv_qualifiers(const Type* type) -> CvQualifiers;

  // pointers
  [[nodiscard]] auto remove_pointer(const Type* type) -> const Type*;
  [[nodiscard]] auto add_pointer(const Type* type) -> const Type*;

  // functions
  [[nodiscard]] auto remove_noexcept(const Type* type) -> const Type*;

  // classes
  [[nodiscard]] auto is_base_of(const Type* base, const Type* derived) -> bool;

  // type relationships
  [[nodiscard]] auto is_same(const Type* a, const Type* b) -> bool;
  [[nodiscard]] auto decay(const Type* type) -> const Type*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx