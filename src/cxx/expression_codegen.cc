// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/codegen.h>
#include <cxx/control.h>
#include <cxx/expression_codegen.h>
#include <cxx/literals.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_environment.h>
#include <cxx/types.h>

#include <stdexcept>

namespace cxx {

ExpressionCodegen::ExpressionCodegen(Codegen* cg) : cg(cg) {}

ir::Expr* ExpressionCodegen::gen(ExpressionAST* ast) {
  ir::Expr* expr = nullptr;
  std::swap(expr_, expr);
  ast->accept(this);
  std::swap(expr_, expr);
  return expr;
}

ir::BinaryOp ExpressionCodegen::convertBinaryOp(TokenKind tk) const {
  switch (tk) {
    case TokenKind::T_STAR:
      return ir::BinaryOp::kStar;
    case TokenKind::T_SLASH:
      return ir::BinaryOp::kSlash;
    case TokenKind::T_PERCENT:
      return ir::BinaryOp::kPercent;
    case TokenKind::T_PLUS:
      return ir::BinaryOp::kPlus;
    case TokenKind::T_MINUS:
      return ir::BinaryOp::kMinus;
    case TokenKind::T_GREATER_GREATER:
      return ir::BinaryOp::kGreaterGreater;
    case TokenKind::T_LESS_LESS:
      return ir::BinaryOp::kLessLess;
    case TokenKind::T_GREATER:
      return ir::BinaryOp::kGreater;
    case TokenKind::T_LESS:
      return ir::BinaryOp::kLess;
    case TokenKind::T_GREATER_EQUAL:
      return ir::BinaryOp::kGreaterEqual;
    case TokenKind::T_LESS_EQUAL:
      return ir::BinaryOp::kLessEqual;
    case TokenKind::T_EQUAL_EQUAL:
      return ir::BinaryOp::kEqualEqual;
    case TokenKind::T_EXCLAIM_EQUAL:
      return ir::BinaryOp::kExclaimEqual;
    case TokenKind::T_AMP:
      return ir::BinaryOp::kAmp;
    case TokenKind::T_CARET:
      return ir::BinaryOp::kCaret;
    case TokenKind::T_BAR:
      return ir::BinaryOp::kBar;
    default:
      throw std::runtime_error("invalid binary op");
  }
}

void ExpressionCodegen::visit(ThisExpressionAST* ast) {
  throw std::runtime_error("visit(ThisExpressionAST): not implemented");
}

void ExpressionCodegen::visit(CharLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(CharLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(BoolLiteralExpressionAST* ast) {
  auto value = cg->unit()->tokenKind(ast->literalLoc) == TokenKind::T_TRUE;
  expr_ = cg->createIntegerLiteral(value);
}

void ExpressionCodegen::visit(IntLiteralExpressionAST* ast) {
  auto value = cg->unit()->tokenText(ast->literalLoc);
  std::int64_t literal = std::int64_t(std::stol(value));
  expr_ = cg->createIntegerLiteral(literal);
}

void ExpressionCodegen::visit(FloatLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(FloatLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(NullptrLiteralExpressionAST* ast) {
  throw std::runtime_error(
      "visit(NullptrLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(StringLiteralExpressionAST* ast) {
  throw std::runtime_error(
      "visit(StringLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(UserDefinedStringLiteralExpressionAST* ast) {
  throw std::runtime_error(
      "visit(UserDefinedStringLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(IdExpressionAST* ast) {
  if (ast->symbol->enclosingBlock() == ast->symbol->enclosingScope()->owner()) {
    expr_ = cg->createTemp(cg->getLocal(ast->symbol));
    return;
  }

  expr_ = cg->createId(ast->symbol);
}

void ExpressionCodegen::visit(NestedExpressionAST* ast) {
  ast->expression->accept(this);
}

void ExpressionCodegen::visit(RightFoldExpressionAST* ast) {
  throw std::runtime_error("visit(RightFoldExpressionAST): not implemented");
}

void ExpressionCodegen::visit(LeftFoldExpressionAST* ast) {
  throw std::runtime_error("visit(LeftFoldExpressionAST): not implemented");
}

void ExpressionCodegen::visit(FoldExpressionAST* ast) {
  throw std::runtime_error("visit(FoldExpressionAST): not implemented");
}

void ExpressionCodegen::visit(LambdaExpressionAST* ast) {
  throw std::runtime_error("visit(LambdaExpressionAST): not implemented");
}

void ExpressionCodegen::visit(SizeofExpressionAST* ast) {
  throw std::runtime_error("visit(SizeofExpressionAST): not implemented");
}

void ExpressionCodegen::visit(SizeofTypeExpressionAST* ast) {
  throw std::runtime_error("visit(SizeofTypeExpressionAST): not implemented");
}

void ExpressionCodegen::visit(SizeofPackExpressionAST* ast) {
  throw std::runtime_error("visit(SizeofPackExpressionAST): not implemented");
}

void ExpressionCodegen::visit(TypeidExpressionAST* ast) {
  throw std::runtime_error("visit(TypeidExpressionAST): not implemented");
}

void ExpressionCodegen::visit(TypeidOfTypeExpressionAST* ast) {
  throw std::runtime_error("visit(TypeidOfTypeExpressionAST): not implemented");
}

void ExpressionCodegen::visit(AlignofExpressionAST* ast) {
  throw std::runtime_error("visit(AlignofExpressionAST): not implemented");
}

void ExpressionCodegen::visit(UnaryExpressionAST* ast) {
  throw std::runtime_error("visit(UnaryExpressionAST): not implemented");
}

void ExpressionCodegen::visit(BinaryExpressionAST* ast) {
  if (ast->op == TokenKind::T_COMMA) {
    cg->statement(ast->leftExpression);
    expr_ = gen(ast->rightExpression);
    return;
  }

  auto left = cg->expression(ast->leftExpression);
  auto right = cg->expression(ast->rightExpression);
  auto op = convertBinaryOp(ast->op);

  auto e = cg->createBinary(op, left, right);
  auto control = cg->unit()->control();
  auto types = control->types();

  QualifiedType ty{types->integerType(IntegerKind::kInt, false)};

  auto local = cg->function()->addLocal(ty);

  cg->emitMove(cg->createTemp(local), e);

  expr_ = cg->createTemp(local);
}

void ExpressionCodegen::visit(AssignmentExpressionAST* ast) {
  if (ast->op == TokenKind::T_EQUAL) {
    auto left = gen(ast->leftExpression);
    auto right = gen(ast->rightExpression);
    cg->emitMove(left, right);
    expr_ = left;
    return;
  }

  throw std::runtime_error("visit(AssignmentExpressionAST): not implemented");
}

void ExpressionCodegen::visit(BracedTypeConstructionAST* ast) {
  throw std::runtime_error("visit(BracedTypeConstructionAST): not implemented");
}

void ExpressionCodegen::visit(TypeConstructionAST* ast) {
  throw std::runtime_error("visit(TypeConstructionAST): not implemented");
}

void ExpressionCodegen::visit(CallExpressionAST* ast) {
  throw std::runtime_error("visit(CallExpressionAST): not implemented");
}

void ExpressionCodegen::visit(SubscriptExpressionAST* ast) {
  auto base = gen(ast->baseExpression);
  auto index = gen(ast->indexExpression);
  expr_ = cg->createSubscript(base, index);
}

void ExpressionCodegen::visit(MemberExpressionAST* ast) {
  throw std::runtime_error("visit(MemberExpressionAST): not implemented");
}

void ExpressionCodegen::visit(ConditionalExpressionAST* ast) {
  auto iftrue = cg->createBlock();
  auto iffalse = cg->createBlock();
  auto endif = cg->createBlock();
  cg->condition(ast->condition, iftrue, iffalse);
  auto local = cg->function()->addLocal(ast->type);
  cg->place(iftrue);
  cg->emitMove(cg->createTemp(local), cg->expression(ast->iftrueExpression));
  cg->emitJump(endif);
  cg->place(iffalse);
  cg->emitMove(cg->createTemp(local), cg->expression(ast->iffalseExpression));
  cg->place(endif);
  expr_ = cg->createTemp(local);
}

void ExpressionCodegen::visit(CastExpressionAST* ast) {
  throw std::runtime_error("visit(CastExpressionAST): not implemented");
}

void ExpressionCodegen::visit(CppCastExpressionAST* ast) {
  throw std::runtime_error("visit(CppCastExpressionAST): not implemented");
}

void ExpressionCodegen::visit(NewExpressionAST* ast) {
  throw std::runtime_error("visit(NewExpressionAST): not implemented");
}

void ExpressionCodegen::visit(DeleteExpressionAST* ast) {
  throw std::runtime_error("visit(DeleteExpressionAST): not implemented");
}

void ExpressionCodegen::visit(ThrowExpressionAST* ast) {
  throw std::runtime_error("visit(ThrowExpressionAST): not implemented");
}

void ExpressionCodegen::visit(NoexceptExpressionAST* ast) {
  throw std::runtime_error("visit(NoexceptExpressionAST): not implemented");
}

}  // namespace cxx
