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

#include "check_expression_types.h"

#include <cxx/ast.h>
#include <cxx/ast_visitor.h>

#include <format>

namespace cxx {

namespace {

class CheckExpressionTypes final : private ASTVisitor {
 public:
  [[nodiscard]] auto operator()(TranslationUnit* unit) {
    std::size_t missingTypes = 0;
    std::swap(unit_, unit);
    std::swap(missingTypes_, missingTypes);

    accept(unit_->ast());

    std::swap(unit_, unit);
    std::swap(missingTypes_, missingTypes);

    return missingTypes == 0;
  }

 private:
  using ASTVisitor::visit;

  auto preVisit(AST* ast) -> bool override {
    if (ast_cast<TemplateDeclarationAST>(ast)) {
      // skip template declarations, as they are not instantiated yet
      return false;
    }

    if (auto expression = ast_cast<ExpressionAST>(ast)) {
      if (!expression->type) {
        const auto loc = expression->firstSourceLocation();

        unit_->warning(loc, std::format("untyped expression of kind '{}'",
                                        to_string(expression->kind())));

        ++missingTypes_;
        return false;
      }
    }

    return true;  // visit children
  }

 private:
  TranslationUnit* unit_ = nullptr;
  std::size_t missingTypes_ = 0;
};

}  // namespace

auto checkExpressionTypes(TranslationUnit& unit) -> bool {
  CheckExpressionTypes checkExpressionTypes;
  return checkExpressionTypes(&unit);
}

}  // namespace cxx