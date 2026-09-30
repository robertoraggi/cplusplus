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
#include <cxx/symbols_fwd.h>

#include <optional>
#include <vector>

namespace cxx {
class TranslationUnit;

struct TemplateArity {
  int minArgs = 0;
  int maxArgs = 0;
  int packCount = 0;
  bool hasParameterPack = false;

  [[nodiscard]] static auto of(TemplateDeclarationAST* templateDecl)
      -> TemplateArity;

  [[nodiscard]] static auto matches(
      TemplateDeclarationAST* templateDecl,
      List<TemplateArgumentAST*>* templateArgumentList,
      bool isFunctionTemplate = false) -> bool;
};

class TemplateArguments {
 public:
  explicit TemplateArguments(TranslationUnit* unit) : unit_(unit) {}

  [[nodiscard]] static auto count(
      List<TemplateArgumentAST*>* templateArgumentList) -> int;

  [[nodiscard]] static auto last(
      List<TemplateArgumentAST*>* templateArgumentList) -> TemplateArgumentAST*;

  [[nodiscard]] static auto isPackExpansion(TemplateArgumentAST* argument)
      -> bool;

  [[nodiscard]] static auto hasPackExpansion(
      List<TemplateArgumentAST*>* templateArgumentList) -> bool;

  [[nodiscard]] auto complete(Symbol* templateSymbol,
                              List<TemplateArgumentAST*>* writtenArguments)
      const -> List<TemplateArgumentAST*>*;

  [[nodiscard]] auto defaultArgument(
      TemplateDeclarationAST* templateDecl, TemplateParameterAST* parameter,
      const std::vector<TemplateArgument>& argumentsSoFar) const
      -> TemplateArgumentAST*;

  [[nodiscard]] auto templateName(Symbol* templateSymbol) const
      -> TemplateArgumentAST*;

  [[nodiscard]] auto integerLiteralExpression(const ConstInt& value,
                                              const Type* type) const
      -> ExpressionAST*;

 private:
  TranslationUnit* unit_;
};

[[nodiscard]] auto isPackParameter(TemplateParameterAST* parameter) -> bool;

[[nodiscard]] auto hasWrittenDefaultTemplateArgument(
    TemplateParameterAST* parameter) -> bool;

void recordDefaultTemplateArgument(TemplateParameterAST* parameter,
                                   TemplateParameterAST* pattern);

[[nodiscard]] auto hasDefaultTemplateArgument(TemplateParameterAST* parameter)
    -> bool;

[[nodiscard]] auto denotesTemplateName(TemplateArgumentAST* argument) -> bool;

[[nodiscard]] auto matchesTemplateParameterKind(TemplateParameterAST* parameter,
                                                TemplateArgumentAST* argument)
    -> bool;

[[nodiscard]] auto isPackExpansion(TypeIdAST* typeId) -> bool;

[[nodiscard]] auto injected_template_argument_list(
    TranslationUnit* unit, List<TemplateParameterAST*>* parameters)
    -> List<TemplateArgumentAST*>*;

[[nodiscard]] auto class_template_arguments(TranslationUnit* unit,
                                            ClassSymbol* classSymbol)
    -> std::vector<TemplateArgument>;

class Substitution {
 public:
  Substitution() = delete;
  Substitution(const Substitution&) = delete;
  Substitution(Substitution&&) = default;
  auto operator=(Substitution&&) -> Substitution& = default;

  Substitution(TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
               List<TemplateArgumentAST*>* templateArgumentList,
               bool argsComplete = false, bool fillDefaults = true);

  [[nodiscard]] static auto make(
      TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
      List<TemplateArgumentAST*>* templateArgumentList,
      bool argsComplete = false) -> std::optional<Substitution>;

  [[nodiscard]] static auto makePartial(
      TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
      List<TemplateArgumentAST*>* templateArgumentList)
      -> std::optional<Substitution>;

  [[nodiscard]] static auto writtenTemplateArguments(
      TranslationUnit* unit, List<TemplateArgumentAST*>* templateArgumentList)
      -> std::optional<std::vector<TemplateArgument>>;

  auto templateArguments() const& -> const std::vector<TemplateArgument>& {
    return templateArguments_;
  }

  auto templateArguments() && -> std::vector<TemplateArgument> {
    return std::move(templateArguments_);
  }

  [[nodiscard]] auto hadError() const -> bool { return hadError_; }

 private:
  Substitution(TranslationUnit* unit,
               List<TemplateArgumentAST*>* templateArgumentList);

  void doMake();

  [[nodiscard]] auto collectWrittenArguments() -> bool;

  [[nodiscard]] auto argumentFor(TemplateParameterAST* parameter,
                                 int index) const -> Symbol*;

  [[nodiscard]] auto checkArgumentKind(TemplateParameterAST* parameter,
                                       int index) -> bool;

  [[nodiscard]] auto injectedClassNameAsType(ClassSymbol* classTemplate) const
      -> Symbol*;

  [[nodiscard]] auto normalizeNonTypeArgument(
      NonTypeTemplateParameterAST* parameter, Symbol* argument) -> Symbol*;

  void convertNonTypeArgument(VariableSymbol* argument, const Type* targetType);

  void bindReferenceArgument(VariableSymbol* argument, const Type* targetType);

  [[nodiscard]] auto valueDependsOnParameterType(
      ExpressionAST* expression) const -> bool;

  [[nodiscard]] auto lacksConvertedValue(VariableSymbol* argument) const
      -> bool;

  [[nodiscard]] auto isConstexprRepresentable(const ConstValue& value) const
      -> bool;

  [[nodiscard]] auto substitutedNonTypeParameterType(
      NonTypeTemplateParameterAST* parameter) -> std::optional<const Type*>;

  [[nodiscard]] auto getDefaultTemplateArgument(TemplateParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  void maybeReportInvalidConstantExpression(SourceLocation loc);
  void maybeReportDefaultArgumentSubstitutionFailure(SourceLocation loc);
  [[nodiscard]] auto hasDependentArguments() const -> bool;
  void maybeReportMalformedTemplateArgument(SourceLocation loc);
  void maybeReportMissingTemplateArgument(SourceLocation loc);

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  struct MakeDefaultTemplateArgument;
  struct CollectRawTemplateArgument;

 private:
  TranslationUnit* unit_ = nullptr;
  TemplateDeclarationAST* templateDecl_ = nullptr;
  List<TemplateArgumentAST*>* templateArgumentList_ = nullptr;
  std::vector<TemplateArgument> templateArguments_;
  std::vector<Symbol*> collectedArguments_;
  std::vector<TemplateArgumentAST*> collectedNodes_;
  std::vector<bool> collectedIsPackExpansion_;
  bool hadError_ = false;
  bool argsComplete_ = false;
  bool fillDefaults_ = true;
};
}  // namespace cxx
