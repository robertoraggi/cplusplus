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

#include <cxx/expression_side_effects.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_visitor.h>
#include <cxx/types.h>

namespace cxx {

namespace {

struct SideEffectFinder final : ASTVisitor {
  bool found = false;

  void visit(CallExpressionAST* ast) override { found = true; }
  void visit(AssignmentExpressionAST* ast) override { found = true; }
  void visit(CompoundAssignmentExpressionAST* ast) override { found = true; }
  void visit(PostIncrExpressionAST* ast) override { found = true; }
  void visit(NewExpressionAST* ast) override { found = true; }
  void visit(DeleteExpressionAST* ast) override { found = true; }
  void visit(AwaitExpressionAST* ast) override { found = true; }
  void visit(YieldExpressionAST* ast) override { found = true; }
  void visit(ThrowExpressionAST* ast) override { found = true; }
  void visit(VaArgExpressionAST* ast) override { found = true; }
  void visit(NestedStatementExpressionAST* ast) override { found = true; }

  void visit(UnaryExpressionAST* ast) override {
    if (ast->symbol || ast->op == TokenKind::T_PLUS_PLUS ||
        ast->op == TokenKind::T_MINUS_MINUS) {
      found = true;
      return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(BinaryExpressionAST* ast) override {
    if (ast->symbol) {
      found = true;
      return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(SubscriptExpressionAST* ast) override {
    if (ast->symbol) {
      found = true;
      return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(TypeConstructionAST* ast) override {
    if (ast->constructorSymbol) {
      found = true;
      return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(BracedTypeConstructionAST* ast) override {
    if (ast->constructorSymbol) {
      found = true;
      return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(ImplicitCastExpressionAST* ast) override {
    if (ast->conversionFunction || readsVolatileObject(ast)) {
      found = true;
      return;
    }
    ASTVisitor::visit(ast);
  }

  [[nodiscard]] static auto readsVolatileObject(ImplicitCastExpressionAST* ast)
      -> bool {
    if (ast->castKind != ImplicitCastKind::kLValueToRValueConversion)
      return false;
    if (!ast->expression || !ast->expression->type) return false;
    const auto qualifiers = cv_qualifiers(ast->expression->type);
    return (qualifiers & CvQualifiers::kVolatile) != CvQualifiers::kNone;
  }
};

}  // namespace

auto expression_has_side_effects(ExpressionAST* ast) -> bool {
  SideEffectFinder finder;
  finder.accept(ast);
  return finder.found;
}

}  // namespace cxx
