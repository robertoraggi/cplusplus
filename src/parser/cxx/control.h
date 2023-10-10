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

#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
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

  auto memoryLayout() const -> MemoryLayout*;

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

  auto makeAnonymousId(std::string_view base) -> const Identifier*;
  auto getIdentifier(std::string_view name) -> const Identifier*;
  auto getOperatorId(std::string_view name) -> const OperatorId*;
  auto getDestructorId(std::string_view name) -> const DestructorId*;

  auto makeTypeParameter(const Name* name) -> TemplateParameter*;
  auto makeTypeParameterPack(const Name* name) -> TemplateParameter*;
  auto makeNonTypeParameter(const Type* type, const Name* name)
      -> TemplateParameter*;

  auto makeParameterSymbol(const Name* name, const Type* type, int index)
      -> ParameterSymbol*;
  auto makeClassSymbol(const Name* name) -> ClassSymbol*;
  auto makeEnumeratorSymbol(const Name* name, const Type* type, long val)
      -> EnumeratorSymbol*;
  auto makeFunctionSymbol(const Name* name, const Type* type)
      -> FunctionSymbol*;
  auto makeGlobalSymbol(const Name* name, const Type* type) -> GlobalSymbol*;
  auto makeInjectedClassNameSymbol(const Name* name, const Type* type)
      -> InjectedClassNameSymbol*;
  auto makeLocalSymbol(const Name* name, const Type* type) -> LocalSymbol*;
  auto makeMemberSymbol(const Name* name, const Type* type, int offset)
      -> MemberSymbol*;
  auto makeNamespaceSymbol(const Name* name) -> NamespaceSymbol*;
  auto makeNamespaceAliasSymbol(const Name* name, Symbol* ns)
      -> NamespaceAliasSymbol*;
  auto makeNonTypeTemplateParameterSymbol(const Name* name, const Type* type,
                                          int index)
      -> NonTypeTemplateParameterSymbol*;
  auto makeScopedEnumSymbol(const Name* name, const Type* type)
      -> ScopedEnumSymbol*;
  auto makeTemplateParameterPackSymbol(const Name* name, int index)
      -> TemplateParameterPackSymbol*;
  auto makeTemplateParameterSymbol(const Name* name, int index)
      -> TemplateParameterSymbol*;
  auto makeConceptSymbol(const Name* name) -> ConceptSymbol*;
  auto makeTypeAliasSymbol(const Name* name, const Type* type)
      -> TypeAliasSymbol*;
  auto makeValueSymbol(const Name* name, const Type* type, long val)
      -> ValueSymbol*;

  auto getInvalidType() -> const InvalidType*;
  auto getNullptrType() -> const NullptrType*;
  auto getDecltypeAutoType() -> const DecltypeAutoType*;
  auto getAutoType() -> const AutoType*;
  auto getVoidType() -> const VoidType*;
  auto getBoolType() -> const BoolType*;
  auto getCharType() -> const CharType*;
  auto getSignedCharType() -> const SignedCharType*;
  auto getUnsignedCharType() -> const UnsignedCharType*;
  auto getChar8Type() -> const Char8Type*;
  auto getChar16Type() -> const Char16Type*;
  auto getChar32Type() -> const Char32Type*;
  auto getWideCharType() -> const WideCharType*;
  auto getShortType() -> const ShortType*;
  auto getUnsignedShortType() -> const UnsignedShortType*;
  auto getIntType() -> const IntType*;
  auto getUnsignedIntType() -> const UnsignedIntType*;
  auto getLongType() -> const LongType*;
  auto getUnsignedLongType() -> const UnsignedLongType*;
  auto getFloatType() -> const FloatType*;
  auto getDoubleType() -> const DoubleType*;
  auto getQualType(const Type* elementType, bool isConst, bool isVolatile)
      -> const QualType*;
  auto getPointerType(const Type* elementType) -> const PointerType*;
  auto getLValueReferenceType(const Type* elementType)
      -> const LValueReferenceType*;
  auto getRValueReferenceType(const Type* elementType)
      -> const RValueReferenceType*;
  auto getArrayType(const Type* elementType, int dimension) -> const ArrayType*;
  auto getFunctionType(const Type* returnType,
                       std::vector<Parameter> parameters,
                       bool isVariadic = false) -> const FunctionType*;
  auto getClassType(ClassSymbol* classSymbol) -> const ClassType*;
  auto getNamespaceType(NamespaceSymbol* namespaceSymbol)
      -> const NamespaceType*;
  auto getMemberPointerType(const Type* classType, const Type* memberType)
      -> const MemberPointerType*;
  auto getConceptType(Symbol* symbol) -> const ConceptType*;
  auto getEnumType(Symbol* symbol) -> const EnumType*;
  auto getGenericType(Symbol* symbol) -> const GenericType*;
  auto getPackType(Symbol* symbol) -> const PackType*;
  auto getScopedEnumType(ScopedEnumSymbol* symbol, const Type* elementType)
      -> const ScopedEnumType*;
  auto getConstType(const Type* type) -> const QualType*;
  auto getVolatileType(const Type* type) -> const QualType*;
  auto getConstVolatileType(const Type* type) -> const QualType*;
  auto getDependentType(DependentSymbol* symbol) -> const DependentType*;

 private:
  struct Private;
  std::unique_ptr<Private> d;
};

}  // namespace cxx