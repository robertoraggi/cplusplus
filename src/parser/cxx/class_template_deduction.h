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
#include <cxx/overload_resolution.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace cxx {
class Arena;
class Control;
class TranslationUnit;

class ClassTemplateArgumentDeduction {
 public:
  explicit ClassTemplateArgumentDeduction(TranslationUnit* unit);

  struct Initializer {
    std::vector<ExpressionAST*> arguments;
    bool isListInitialization = false;
    bool isCopyInitialization = false;
  };

  [[nodiscard]] static auto placeholderClassTemplate(
      SpecifierAST* typeSpecifier, ScopeSymbol* scope) -> ClassSymbol*;

  [[nodiscard]] static auto alreadyDeducedSpecialization(
      ClassSymbol* primaryTemplate, const Type* recordedType) -> const Type*;

  [[nodiscard]] auto deduce(ClassSymbol* primaryTemplate,
                            const Initializer& initializer,
                            SourceLocation location, ScopeSymbol* scope)
      -> ClassSymbol*;

  [[nodiscard]] auto selectedExplicitOnly() const -> bool {
    return explicitOnly_;
  }

 private:
  struct Guide {
    FunctionSymbol* function = nullptr;
    TemplateDeclarationAST* templateDeclaration = nullptr;
    ParameterDeclarationClauseAST* parameters = nullptr;
    SimpleTemplateIdAST* returnTemplateId = nullptr;
    int classParameterCount = 0;
    int constructorIndex = -1;
    bool isExplicit = false;
    bool isAggregate = false;
    DeductionCandidateInfo info;
  };

  void collectGuides(ClassSymbol* primaryTemplate, const Initializer& init,
                     ScopeSymbol* scope);

  void addConstructorGuide(ClassSymbol* primaryTemplate,
                           FunctionSymbol* constructor, int constructorIndex,
                           ScopeSymbol* scope);

  void addDefaultConstructorGuide(ClassSymbol* primaryTemplate,
                                  ScopeSymbol* scope);

  void addCopyDeductionCandidate(ClassSymbol* primaryTemplate,
                                 ScopeSymbol* scope);

  void addAggregateDeductionCandidate(ClassSymbol* primaryTemplate,
                                      const Initializer& init,
                                      ScopeSymbol* scope);

  void addWrittenGuide(ClassSymbol* primaryTemplate,
                       DeductionGuideSymbol* guide);

  [[nodiscard]] auto classTemplateParameters(ClassSymbol* primaryTemplate) const
      -> List<TemplateParameterAST*>*;

  [[nodiscard]] auto classTemplateParameterCount(
      ClassSymbol* primaryTemplate) const -> int;

  [[nodiscard]] auto makeGuideTemplateDeclaration(
      ClassSymbol* primaryTemplate,
      TemplateDeclarationAST* ownTemplateDeclaration,
      ParameterDeclarationClauseAST* parameters) -> TemplateDeclarationAST*;

  [[nodiscard]] auto makeInjectedTemplateId(ClassSymbol* primaryTemplate)
      -> SimpleTemplateIdAST*;

  [[nodiscard]] auto makeParameterDeclaration(const Type* type,
                                              SpecifierAST* writtenSpecifier)
      -> ParameterDeclarationAST*;

  [[nodiscard]] auto makeParameterClause(
      const std::vector<ParameterDeclarationAST*>& parameters, bool isVariadic)
      -> ParameterDeclarationClauseAST*;

  [[nodiscard]] auto aggregateElementTypes(ClassSymbol* classSymbol,
                                           std::size_t argumentCount)
      -> std::vector<const Type*>;

  [[nodiscard]] auto guideParameterTypes(
      ClassSymbol* primaryTemplate, const Guide& guide,
      List<TemplateArgumentAST*>* deducedArgs, const Initializer& initializer,
      SourceLocation location, ScopeSymbol* scope)
      -> std::optional<std::vector<const Type*>>;

  [[nodiscard]] auto requiredParameterCount(const Guide& guide,
                                            std::size_t parameterCount) const
      -> std::size_t;

  [[nodiscard]] auto argumentList(const Initializer& init)
      -> List<ExpressionAST*>*;

  [[nodiscard]] auto specializationFor(ClassSymbol* primaryTemplate,
                                       const Guide& guide,
                                       List<TemplateArgumentAST*>* deducedArgs,
                                       SourceLocation location,
                                       ScopeSymbol* scope) -> ClassSymbol*;

  TranslationUnit* unit_;
  Control* control_;
  Arena* arena_;
  std::vector<Guide> guides_;
  bool explicitOnly_ = false;
};
}  // namespace cxx
