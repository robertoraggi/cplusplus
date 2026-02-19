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
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>

// std
#include <optional>
#include <vector>

namespace cxx {

class TranslationUnit;

class Substitution {
 public:
  Substitution() = delete;
  Substitution(const Substitution&) = delete;

  Substitution(TranslationUnit* unit, TemplateDeclarationAST* templateDecl,
               List<TemplateArgumentAST*>* templateArgumentList);

  auto templateArguments() const -> const std::vector<TemplateArgument>& {
    return templateArguments_;
  }

 private:
  void make();

  [[nodiscard]] auto isDependentTypeArgument(TypeTemplateArgumentAST* typeArg)
      -> bool;

  [[nodiscard]] auto isPackParameter(TemplateParameterAST* parameter) -> bool;

  [[nodiscard]] auto isDependentExpressionArgument(
      ExpressionTemplateArgumentAST* expressionArg) -> bool;

  [[nodiscard]] auto hasDefaultTemplateArgument(TemplateParameterAST* parameter)
      -> bool;

  [[nodiscard]] auto normalizeNonTypeArgument(
      NonTypeTemplateParameterAST* parameter, Symbol* argument) -> Symbol*;

  [[nodiscard]] auto getDefaultTemplateArgument(TemplateParameterAST* parameter)
      -> std::optional<TemplateArgument>;

  void maybeReportInvalidConstantExpression(SourceLocation loc);
  void maybeReportMalformedTemplateArgument(SourceLocation loc);
  void maybeReportMissingTemplateArgument(SourceLocation loc);

  void error(SourceLocation loc, std::string message);
  void warning(SourceLocation loc, std::string message);

  struct MakeDefaultTemplateArgument;
  struct CollectRawTemplateArgument;
  struct HasDefaultTemplateArgument;
  struct IsPackParameter;

 private:
  TranslationUnit* unit_ = nullptr;
  TemplateDeclarationAST* templateDecl_ = nullptr;
  List<TemplateArgumentAST*>* templateArgumentList_ = nullptr;
  std::vector<TemplateArgument> templateArguments_;
};

}  // namespace cxx