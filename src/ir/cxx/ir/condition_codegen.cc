// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/ir/codegen.h>
#include <cxx/ir/condition_codegen.h>
#include <cxx/literals.h>
#include <cxx/translation_unit.h>

#include <stdexcept>

namespace cxx::ir {

ConditionCodegen::ConditionCodegen(Codegen* cg) : ExpressionCodegen(cg) {}

void ConditionCodegen::gen(ExpressionAST* ast, ir::Block* iftrue,
                           ir::Block* iffalse) {
  std::swap(iftrue_, iftrue);
  std::swap(iffalse_, iffalse);
  if (ast && ast->constValue) {
    cg->emitJump(const_value_cast<bool>(*ast->constValue) ? iftrue_ : iffalse_);
  } else if (auto expr = gen(ast)) {
    std::uint64_t value = 0;
    cg->emitCondJump(cg->createBinary(ir::BinaryOp::kExclaimEqual, expr,
                                      cg->createIntegerLiteral(value)),
                     iftrue_, iffalse_);
  }
  std::swap(iftrue_, iftrue);
  std::swap(iffalse_, iffalse);
}

void ConditionCodegen::visit(IntLiteralExpressionAST* ast) {
  auto value = ast->literal->integerValue();
  cg->emitJump(value ? iftrue_ : iffalse_);
}

void ConditionCodegen::visit(BinaryExpressionAST* ast) {
  if (ast->op == TokenKind::T_AMP_AMP) {
    auto block = cg->createBlock();
    cg->condition(ast->leftExpression, block, iffalse_);
    cg->place(block);
    cg->condition(ast->rightExpression, iftrue_, iffalse_);
    return;
  }

  if (ast->op == TokenKind::T_BAR_BAR) {
    auto block = cg->createBlock();
    cg->condition(ast->leftExpression, iftrue_, block);
    cg->place(block);
    cg->condition(ast->rightExpression, iftrue_, iffalse_);
    return;
  }

  const auto op = convertBinaryOp(ast->op);

  if (isRelOp(op)) {
    auto left = cg->reduce(ast->leftExpression);
    auto right = cg->reduce(ast->rightExpression);
    auto cond = cg->createBinary(convertBinaryOp(ast->op), left, right);
    cg->emitCondJump(cond, iftrue_, iffalse_);
    return;
  }

  ExpressionCodegen::visit(ast);
}

auto ConditionCodegen::isRelOp(ir::BinaryOp op) const -> bool {
  switch (op) {
    case ir::BinaryOp::kGreaterGreater:
    case ir::BinaryOp::kLessLess:
    case ir::BinaryOp::kGreater:
    case ir::BinaryOp::kLess:
    case ir::BinaryOp::kGreaterEqual:
    case ir::BinaryOp::kLessEqual:
    case ir::BinaryOp::kEqualEqual:
    case ir::BinaryOp::kExclaimEqual:
      return true;
    default:
      return false;
  }  // switch
}

}  // namespace cxx::ir
