// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast_fwd.h>
#include <cxx/const_int.h>
#include <cxx/names_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cxx {
class ExternalNameEncoder {
 public:
  enum class StructorVariant { Complete, Base, Deleting };

  explicit ExternalNameEncoder(TranslationUnit* unit);

  void setStructorVariant(StructorVariant variant) {
    structorVariant_ = variant;
    hasExplicitStructorVariant_ = true;
  }

  [[nodiscard]] auto encode(Symbol* symbol, std::string_view suffix = "")
      -> std::string;
  [[nodiscard]] auto encode(const Type* type) -> std::string;

  [[nodiscard]] auto mangledAbiTags(Symbol* symbol)
      -> std::vector<const Identifier*>;
  [[nodiscard]] auto encodeVTable(ClassSymbol* classSymbol) -> std::string;
  [[nodiscard]] auto encodeConstructionVTable(ClassSymbol* completeClass,
                                              std::int64_t offset,
                                              ClassSymbol* baseClass)
      -> std::string;
  [[nodiscard]] auto encodeVTT(ClassSymbol* classSymbol) -> std::string;
  [[nodiscard]] auto encodeThunk(
      FunctionSymbol* target, const VTableLayout::CallOffset& thisAdjustment,
      const VTableLayout::CallOffset& returnAdjustment) -> std::string;
  [[nodiscard]] auto encodeGuardVariable(Symbol* symbol) -> std::string;
  [[nodiscard]] auto encodeTypeInfo(const Type* type) -> std::string;
  [[nodiscard]] auto encodeTypeInfoName(const Type* type) -> std::string;
  [[nodiscard]] auto encodeLambdaSignature(FunctionSymbol* callOperator)
      -> std::string;

 private:
  [[nodiscard]] auto encodeFunction(FunctionSymbol* function) -> std::string;
  [[nodiscard]] auto isUnmangledData(Symbol* symbol) -> bool;

  [[nodiscard]] auto encodeData(Symbol* symbol) -> std::string;

  void encodePrefix(Symbol* symbol);
  void encodeTemplatePrefix(Symbol* symbol);
  void encodeUnqualifiedName(Symbol* symbol);

  void encodeName(Symbol* symbol);
  void encodeTemplateName(Symbol* symbol);
  [[nodiscard]] auto encodeTemplateTemplateArgument(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeLocalName(Symbol* symbol) -> bool;
  void encodeClosureTypeName(ClassSymbol* classSymbol);
  void encodeUnnamedTypeName(Symbol* symbol);
  [[nodiscard]] auto unnamedTypeIndex(Symbol* symbol) const -> int;
  void encodeObjectParameterQualifiers(FunctionSymbol* function);
  void encodeCvQualifiers(CvQualifiers cvQualifiers);
  [[nodiscard]] auto encodeNestedName(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeStdTypeAbbreviation(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeStdTemplateAbbreviation(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeUnscopedName(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeOperatorName(TokenKind op, bool isUnary)
      -> std::string_view;

  void encodeType(const Type* type);
  [[nodiscard]] auto encodeDependentName(NestedNameSpecifierAST* nns,
                                         UnqualifiedIdAST* id) -> bool;
  [[nodiscard]] auto encodeUnresolvedQualifierLevels(
      NestedNameSpecifierAST* nns, NestedNameSpecifierAST* root) -> bool;
  void encodeSourceName(const Identifier* identifier);
  [[nodiscard]] auto encodeExpression(ExpressionAST* expr) -> bool;
  void encodeRequiredExpression(ExpressionAST* expr, std::string_view what);
  [[noreturn]] void reportUnencodable(SourceLocation location,
                                      std::string_view what);
  [[nodiscard]] auto encodeWrittenTemplateArguments(
      Symbol* templateName, List<TemplateArgumentAST*>* arguments) -> bool;
  [[nodiscard]] auto encodeWrittenTemplateId(
      Symbol* templateName, List<TemplateArgumentAST*>* arguments,
      bool isPrefix) -> bool;
  void encodeTemplateParamValue(int depth, int index);
  void encodeTemplateParameterDeclaration(TemplateParameterAST* parameter,
                                          TemplateParameterAST* written);
  void encodeTypeConstraint(TypeConstraintAST* typeConstraint);
  void encodeRequiresClause(RequiresClauseAST* requiresClause,
                            FunctionSymbol* function = nullptr);
  [[nodiscard]] auto encodeRequirement(RequirementAST* requirement) -> bool;
  [[nodiscard]] auto encodeFunctionParameter(ParameterSymbol* parameter)
      -> bool;
  [[nodiscard]] auto encodeUnresolvedName(NestedNameSpecifierAST* nns,
                                          UnqualifiedIdAST* id) -> bool;
  [[nodiscard]] auto encodeBaseUnresolvedName(UnqualifiedIdAST* id) -> bool;
  [[nodiscard]] auto encodeUnresolvedType(NestedNameSpecifierAST* nns) -> bool;
  void encodeFunctionSignature(FunctionSymbol* function);
  void encodeLambdaSig(FunctionSymbol* callOperator);
  void encodeConstValue(const Type* type, const ConstValue& value);
  void encodeFloatingValue(const Type* type, double value);
  void encodeTemplateArgumentValue(const Type* type, const ConstValue& value);
  void encodeValueExpression(const Type* type, const ConstValue& value);
  void encodeAddressValue(const Type* type, const ConstAddress& address);
  void encodeObjectValue(const Type* type, const ConstObject& object);
  void encodeArrayValue(const Type* type, const InitializerList& elements);
  void encodeExternalName(Symbol* symbol);
  [[nodiscard]] auto isZeroValue(const ConstValue& value) const -> bool;
  [[nodiscard]] auto isExpressionPrimary(const Type* type,
                                         const ConstValue& value) const -> bool;

  [[nodiscard]] auto normalizeConstInt(const Type* type,
                                       const ConstInt& value) const -> ConstInt;
  void encodeBareFunctionType(const FunctionType* functionType,
                              bool includeReturnType = false);

  void encodeAbiTags(Symbol* symbol);

  [[nodiscard]] auto encodeTemplateNameSubstitution(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeSubstitution(const Type* type) -> bool;
  [[nodiscard]] auto encodeSubstitution(Symbol* symbol) -> bool;
  [[nodiscard]] auto encodeSubstitutionAt(std::size_t index) -> bool;
  [[nodiscard]] auto encodeTemplatePrefixSubstitution(
      Symbol* templateSymbol, List<TemplateArgumentAST*>* arguments) -> bool;
  [[nodiscard]] static auto encodeSeqId(int id) -> std::string;
  void enterSubstitution(const Type* type);
  void enterSubstitution(Symbol* symbol);
  void enterTemplatePrefixSubstitution(Symbol* templateSymbol,
                                       List<TemplateArgumentAST*>* arguments);

  void out(std::string_view str) { out_.append(str); }

  struct EncodeType;
  struct EncodeUnqualifiedName;
  struct EncodeExpression;
  struct EncodeTemplateParameterDeclaration;
  struct EncodeUnresolvedQualifierLevel;
  struct EncodeWrittenTemplateArgument;
  struct EncodeSimpleId;
  struct EncodeDependentQualifier;
  struct EncodeBaseUnresolvedName;
  struct EncodeRequirement;

 private:
  struct TemplatePrefixSubstitution {
    Symbol* templateSymbol = nullptr;
    List<TemplateArgumentAST*>* arguments = nullptr;
  };

  struct TemplateParameterSubstitution {
    int depth = 0;
    int index = 0;

    [[nodiscard]] auto operator==(const TemplateParameterSubstitution&) const
        -> bool = default;
  };

  [[nodiscard]] auto templateParameterSubstitution(const Type* type) const
      -> std::optional<TemplateParameterSubstitution>;
  [[nodiscard]] auto encodeSubstitution(
      const TemplateParameterSubstitution& parameter) -> bool;
  void enterSubstitution(const TemplateParameterSubstitution& parameter);

  using Substitution =
      std::variant<const Type*, Symbol*, TemplatePrefixSubstitution,
                   TemplateParameterSubstitution>;
  std::vector<Substitution> substs_;
  std::string out_;
  StructorVariant structorVariant_ = StructorVariant::Complete;
  Symbol* templateNameOnly_ = nullptr;
  TypeConstraintAST* placeholderConstraint_ = nullptr;
  bool encodesTemplateParameterDepth_ = false;
  std::vector<FunctionParametersSymbol*> parameterScopes_;
  TranslationUnit* unit_ = nullptr;
  Symbol* encodingSymbol_ = nullptr;
  bool hasExplicitStructorVariant_ = false;
};
}  // namespace cxx
