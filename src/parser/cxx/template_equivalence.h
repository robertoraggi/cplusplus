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
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstddef>
#include <vector>

namespace cxx {

class TranslationUnit;

class TemplateEquivalence {
 public:
  struct ParameterCorrespondence {
    int lhsDepth = -1;
    int rhsDepth = -1;
    int count = 0;

    [[nodiscard]] auto applies() const -> bool { return lhsDepth != rhsDepth; }
  };

  explicit TemplateEquivalence(TranslationUnit* unit) : unit_(unit) {}

  TemplateEquivalence(TranslationUnit* unit,
                      ParameterCorrespondence correspondence)
      : unit_(unit), correspondence_(correspondence) {}

  [[nodiscard]] auto same(const ExceptionSpecification& a,
                          const ExceptionSpecification& b) const -> bool;
  [[nodiscard]] auto same(const Type* a, const Type* b) const -> bool;
  [[nodiscard]] auto same(ExpressionAST* a, ExpressionAST* b) const -> bool;
  [[nodiscard]] auto same(TypeIdAST* a, TypeIdAST* b) const -> bool;
  [[nodiscard]] auto same(UnqualifiedIdAST* a, UnqualifiedIdAST* b) const
      -> bool;
  [[nodiscard]] auto same(NestedNameSpecifierAST* a,
                          NestedNameSpecifierAST* b) const -> bool;
  [[nodiscard]] auto same(RequiresClauseAST* a, RequiresClauseAST* b) const
      -> bool;
  [[nodiscard]] auto same(TemplateDeclarationAST* a,
                          TemplateDeclarationAST* b) const -> bool;
  [[nodiscard]] auto same(List<TemplateArgumentAST*>* a,
                          List<TemplateArgumentAST*>* b) const -> bool;
  [[nodiscard]] auto same(List<TemplateParameterAST*>* a,
                          List<TemplateParameterAST*>* b) const -> bool;

  [[nodiscard]] auto sameWritten(List<TemplateArgumentAST*>* a,
                                 List<TemplateArgumentAST*>* b) const -> bool;

  [[nodiscard]] auto corresponds(const Type* lhs, const Type* rhs,
                                 ParameterCorrespondence correspondence) const
      -> bool;

  [[nodiscard]] auto sameForOrdering(const Type* a, const Type* b,
                                     TemplateDeclarationAST* aTemplate,
                                     TemplateDeclarationAST* bTemplate) const
      -> bool;
  [[nodiscard]] auto sameForOrdering(List<TemplateParameterAST*>* a,
                                     List<TemplateParameterAST*>* b) const
      -> bool;

  [[nodiscard]] auto ownFunctionTemplateHead(
      ClassSymbol* enclosingClass, TemplateDeclarationAST* templateHead) const
      -> TemplateDeclarationAST*;

 private:
  [[nodiscard]] auto same(NamedTypeSpecifierAST* a,
                          NamedTypeSpecifierAST* b) const -> bool;
  [[nodiscard]] auto same(TypenameSpecifierAST* a,
                          TypenameSpecifierAST* b) const -> bool;
  [[nodiscard]] auto same(NonTypeTemplateParameterAST* a,
                          NonTypeTemplateParameterAST* b) const -> bool;

  [[nodiscard]] auto sameWritten(NamedTypeSpecifierAST* a,
                                 NamedTypeSpecifierAST* b) const -> bool;

  [[nodiscard]] auto sameQualifiedName(NestedNameSpecifierAST* aQualifier,
                                       UnqualifiedIdAST* aName,
                                       NestedNameSpecifierAST* bQualifier,
                                       UnqualifiedIdAST* bName) const -> bool;

  [[nodiscard]] auto corresponds(const TemplateArgument& lhs,
                                 const TemplateArgument& rhs,
                                 ParameterCorrespondence correspondence) const
      -> bool;
  [[nodiscard]] auto corresponds(const std::vector<TemplateArgument>& lhs,
                                 const std::vector<TemplateArgument>& rhs,
                                 ParameterCorrespondence correspondence) const
      -> bool;

  enum class ArgumentMatch { kByType, kByWrittenTypeId };

  [[nodiscard]] auto walkArguments(List<TemplateArgumentAST*>* a,
                                   List<TemplateArgumentAST*>* b,
                                   ArgumentMatch match) const -> bool;

  TranslationUnit* unit_ = nullptr;
  ParameterCorrespondence correspondence_;
};

}  // namespace cxx
