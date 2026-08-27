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
#include <cxx/const_value.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/type_traits.h>
#include <cxx/types_fwd.h>

#include <optional>
#include <span>
#include <vector>

namespace cxx {
class Arena;
class Control;
class TranslationUnit;

struct TemplateParameterInfo {
  enum class Kind {
    kUnknown,
    kType,
    kNonType,
    kTemplate,
    kConstraint,
  };

  const TypeParameterType* typeParameterType = nullptr;
  TemplateParameterAST* parameterAST = nullptr;
  int depth = 0;
  int index = 0;
  bool isPack = false;
  bool hasDefault = false;
  Kind kind = Kind::kUnknown;
};

class TemplateArgumentDeduction {
 public:
  explicit TemplateArgumentDeduction(TranslationUnit* unit);

  [[nodiscard]] auto deduce(FunctionSymbol* func, List<ExpressionAST*>* args,
                            List<TemplateArgumentAST*>* explicitTemplateArgs)
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto deduceForGuide(TemplateDeclarationAST* templateDecl,
                                    const FunctionType* functionType,
                                    ParameterDeclarationClauseAST* parameters,
                                    List<ExpressionAST*>* args)
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto deduceFromTargetType(
      FunctionSymbol* func, const FunctionType* targetType,
      List<TemplateArgumentAST*>* explicitTemplateArgs = nullptr)
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto deduceFromConversionTarget(FunctionSymbol* func,
                                                const Type* targetType)
      -> std::optional<List<TemplateArgumentAST*>*>;

 private:
  struct DeducibleParameterVisitor;

  void collectTemplateParameters(TemplateDeclarationAST* templateDecl);

  [[nodiscard]] auto substituteExplicitTemplateArguments(
      List<TemplateArgumentAST*>* explicitTemplateArgs) -> bool;

  [[nodiscard]] auto isExplicitArgumentCompatible(
      const TemplateParameterInfo& info, TemplateArgumentAST* arg) -> bool;

  [[nodiscard]] auto isForwardingReference(const Type* paramType) const -> bool;

  [[nodiscard]] auto deduceTypeFromType(const Type* P, const Type* A) -> bool;

  [[nodiscard]] auto deduceTemplateId(
      SimpleTemplateIdAST* pattern, std::span<const TemplateArgument> arguments,
      std::span<TemplateArgumentAST* const> substitutions = {},
      std::span<const TemplateArgument> patternArguments = {}) -> bool;

  [[nodiscard]] auto completedTemplateArguments(const Type* type)
      -> std::span<const TemplateArgument>;

  [[nodiscard]] auto matchCompletedArgument(
      const TemplateArgument& patternArgument, const TemplateArgument& argument)
      -> bool;

  [[nodiscard]] auto deduceCurrentInstantiation(const Type* patternType,
                                                const Type* argumentType)
      -> bool;

  [[nodiscard]] auto adjustedCallArgumentType(const Type* P, const Type* A,
                                              ExpressionAST* argExpr) const
      -> const Type*;

  [[nodiscard]] auto deduceFromCall(const FunctionType* functionType,
                                    List<ExpressionAST*>* args) -> bool;

  [[nodiscard]] auto deduceFromInitializerList(const Type* P,
                                               BracedInitListAST* list) -> bool;

  [[nodiscard]] auto checkDeducedArguments() -> bool;

  [[nodiscard]] auto buildTemplateArgumentList()
      -> std::optional<List<TemplateArgumentAST*>*>;

  [[nodiscard]] auto collectDeducedSoFar(
      List<TemplateArgumentAST*>* argumentsSoFar)
      -> std::optional<std::vector<TemplateArgument>>;

  [[nodiscard]] auto substituteDefaultTypeId(
      TypeIdAST* typeId, const std::vector<TemplateArgument>& arguments)
      -> TypeIdAST*;

  [[nodiscard]] auto substituteDefaultExpression(
      ExpressionAST* expression, const std::vector<TemplateArgument>& arguments)
      -> ExpressionAST*;

  [[nodiscard]] auto defaultTemplateArgument(
      TemplateParameterAST* parameter,
      const std::vector<TemplateArgument>& argumentsSoFar)
      -> TemplateArgumentAST*;

  [[nodiscard]] auto makeTemplateNameArgument(Symbol* templateSymbol)
      -> TemplateArgumentAST*;

  static auto getParameterClause(DeclarationAST* decl)
      -> ParameterDeclarationClauseAST*;

  [[nodiscard]] auto makePackArgument(int parameterIndex)
      -> TemplateArgumentAST*;

  [[nodiscard]] auto makeTypePackElement(const Type* elementType) -> Symbol*;

  [[nodiscard]] auto deducedTypeArgument(int parameterIndex) const
      -> const Type*;

  [[nodiscard]] auto nonTypeParameterType(int parameterIndex) const
      -> const Type*;

  [[nodiscard]] auto makeValuePackElement(const ConstValue& value,
                                          const Type* elementType) -> Symbol*;

  [[nodiscard]] auto makeExplicitPackElement(TemplateArgumentAST* explicitArg,
                                             int parameterIndex) -> Symbol*;

  [[nodiscard]] auto recordDeducedValue(int index, const ConstValue& value,
                                        bool isPack) -> bool;

  void beginParameterDeduction();

  [[nodiscard]] auto deduceFromClassTemplateParam(
      ParameterDeclarationAST* paramDecl, const Type* argType, const Type* P)
      -> bool;

  [[nodiscard]] auto mentionsDeducibleParameter(const Type* type) const -> bool;

  [[nodiscard]] auto classMentionsDeducibleParameter(ClassSymbol* symbol) const
      -> bool;

  [[nodiscard]] auto deducedClassCandidates(ClassSymbol* argClass,
                                            ClassSymbol* paramClass) const
      -> std::vector<ClassSymbol*>;

  [[nodiscard]] auto recordDeducedTemplate(int index, Symbol* templateSymbol)
      -> bool;

  struct DeductionState {
    std::vector<const Type*> types;
    std::vector<Symbol*> templates;
    std::vector<std::optional<std::uint64_t>> values;
    std::vector<std::vector<const Type*>> packs;
    std::vector<std::size_t> packElementCursor;
    std::vector<std::vector<std::uint64_t>> valuePacks;
  };

  [[nodiscard]] auto saveDeductionState() const -> DeductionState;

  void restoreDeductionState(const DeductionState& state);

  [[nodiscard]] auto deduceArrayBound(const Type* P, const Type* A) -> bool;

  [[nodiscard]] auto nonTypeParameterIndex(ExpressionAST* expr) const -> int;

  [[nodiscard]] auto parameterSlot(int depth, int index) const -> int;

  [[nodiscard]] auto parameterSlot(const TypeParameterType* type) const -> int;

  TranslationUnit* unit_;
  TypeTraits traits;
  Control* control_;
  Arena* arena_;

  std::vector<TemplateParameterInfo> templateParams_;
  std::vector<TemplateArgumentAST*> explicitParamArg_;
  std::vector<std::vector<TemplateArgumentAST*>> explicitPackArgs_;
  std::vector<const Type*> deducedTypes_;
  std::vector<Symbol*> deducedTemplates_;
  std::vector<std::optional<std::uint64_t>> deducedValues_;
  std::vector<std::vector<const Type*>> deducedPacks_;
  std::vector<std::size_t> packElementCursor_;
  std::vector<std::vector<std::uint64_t>> deducedValuePacks_;
  List<ParameterDeclarationAST*>* parameterDeclarations_ = nullptr;
  TemplateDeclarationAST* templateDecl_ = nullptr;
};
}  // namespace cxx
