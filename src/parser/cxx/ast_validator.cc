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

#include <cxx/ast.h>
#include <cxx/ast_printer.h>
#include <cxx/ast_validator.h>
#include <cxx/ast_visitor.h>
#include <cxx/dependent_types.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <cstdlib>
#include <format>
#include <iostream>

namespace cxx {
namespace {

class CompletedInstantiationValidator final : private ASTVisitor {
 public:
  explicit CompletedInstantiationValidator(TranslationUnit* unit)
      : unit_(unit) {}

  [[nodiscard]] auto operator()(FunctionDefinitionAST* ast) -> ExpressionAST* {
    accept(ast->functionBody);
    return unresolvedExpression_;
  }

 private:
  using ASTVisitor::visit;

  auto preVisit(AST* ast) -> bool override {
    if (unresolvedExpression_) return false;
    return ast_cast<TemplateDeclarationAST>(ast) == nullptr;
  }

  void postVisit(AST* ast) override {
    if (unresolvedExpression_) return;

    auto expression = ast_cast<ExpressionAST>(ast);
    if (!expression) return;

    if (!expression->type) return;

    auto unresolved = containsPlaceholderType(expression->type);
    if (!unresolved) unresolved = isDependent(unit_, expression->type);
    if (unresolved) unresolvedExpression_ = expression;
  }

  void visit(LambdaExpressionAST* ast) override {
    if (!ast->templateParameterList) {
      ASTVisitor::visit(ast);
      return;
    }

    for (auto capture : ListView{ast->captureList}) accept(capture);
  }

 private:
  TranslationUnit* unit_ = nullptr;
  ExpressionAST* unresolvedExpression_ = nullptr;
};

}  // namespace

void validateCompletedInstantiation(TranslationUnit* unit,
                                    FunctionSymbol* function,
                                    FunctionDefinitionAST* ast) {
  if (!unit || !function || !ast) return;
  if (!unit->config().validateAst) return;
  if (isEnclosedInDependentTemplate(unit, function, true)) return;

  auto expression = CompletedInstantiationValidator{unit}(ast);
  if (!expression) return;

  auto functionName = std::string{"<unnamed>"};
  if (function->name()) functionName = to_string(function->name());

  auto typeName = std::string{"<missing>"};
  if (expression->type) typeName = to_string(expression->type);

  std::cerr << std::format(
      "cxx: AST validation failed after completing instantiation of '{}': "
      "{} has unresolved type '{}'.\n",
      functionName, to_string(expression->kind()), typeName);
  ASTPrinter{unit, std::cerr}(ast);
  std::cerr.flush();
  std::abort();
}

}  // namespace cxx
