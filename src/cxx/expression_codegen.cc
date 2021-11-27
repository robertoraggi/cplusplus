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
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_environment.h>
#include <cxx/type_visitor.h>
#include <cxx/types.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

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

ir::Expr* ExpressionCodegen::reduce(ExpressionAST* ast) {
  auto expr = gen(ast);
  if (dynamic_cast<ir::Temp*>(expr) ||
      dynamic_cast<ir::IntegerLiteral*>(expr) ||
      dynamic_cast<ir::FloatLiteral*>(expr) ||
      dynamic_cast<ir::CharLiteral*>(expr))
    return expr;
  auto local = cg->function()->addLocal(ast->type);
  cg->emitMove(cg->createTemp(local), expr);
  expr = cg->createTemp(local);
  return expr;
}

ir::UnaryOp ExpressionCodegen::convertUnaryOp(TokenKind tk) const {
  switch (tk) {
    case TokenKind::T_STAR:
      return ir::UnaryOp::kStar;
    case TokenKind::T_AMP:
      return ir::UnaryOp::kAmp;
    case TokenKind::T_PLUS:
      return ir::UnaryOp::kPlus;
    case TokenKind::T_MINUS:
      return ir::UnaryOp::kMinus;
    case TokenKind::T_EXCLAIM:
      return ir::UnaryOp::kExclaim;
    case TokenKind::T_TILDE:
      return ir::UnaryOp::kTilde;
    default:
      throw std::runtime_error(
          fmt::format("invalid unary op '{}'", Token::spell(tk)));
  }  // switch
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
      throw std::runtime_error(
          fmt::format("invalid binary op '{}'", Token::spell(tk)));
  }
}

ir::BinaryOp ExpressionCodegen::convertAssignmentOp(TokenKind tk) const {
  switch (tk) {
    case TokenKind::T_STAR_EQUAL:
      return ir::BinaryOp::kStar;
    case TokenKind::T_SLASH_EQUAL:
      return ir::BinaryOp::kSlash;
    case TokenKind::T_PERCENT_EQUAL:
      return ir::BinaryOp::kPercent;
    case TokenKind::T_PLUS_EQUAL:
      return ir::BinaryOp::kPlus;
    case TokenKind::T_MINUS_EQUAL:
      return ir::BinaryOp::kMinus;
    case TokenKind::T_GREATER_GREATER_EQUAL:
      return ir::BinaryOp::kGreaterGreater;
    case TokenKind::T_LESS_LESS_EQUAL:
      return ir::BinaryOp::kLessLess;
    case TokenKind::T_AMP_EQUAL:
      return ir::BinaryOp::kAmp;
    case TokenKind::T_CARET_EQUAL:
      return ir::BinaryOp::kCaret;
    case TokenKind::T_BAR_EQUAL:
      return ir::BinaryOp::kBar;
    default:
      throw std::runtime_error(
          fmt::format("invalid binary op '{}'", Token::spell(tk)));
  }
}

void ExpressionCodegen::visit(ThisExpressionAST* ast) {
  expr_ = cg->createThis(ast->type);
}

void ExpressionCodegen::visit(CharLiteralExpressionAST* ast) {
  expr_ = cg->createCharLiteral(ast->literal);
}

void ExpressionCodegen::visit(BoolLiteralExpressionAST* ast) {
  const auto value = ast->literal == TokenKind::T_TRUE;
  expr_ = cg->createIntegerLiteral(value);
}

void ExpressionCodegen::visit(IntLiteralExpressionAST* ast) {
  const auto& value = ast->literal->value();
  if (value.starts_with("0x") || value.starts_with("0X")) {
    std::int64_t literal =
        std::int64_t(std::stol(value.substr(2), nullptr, 16));
    expr_ = cg->createIntegerLiteral(literal);
  } else if (value.starts_with("0b") || value.starts_with("0B")) {
    std::int64_t literal = std::int64_t(std::stol(value.substr(2), nullptr, 2));
    expr_ = cg->createIntegerLiteral(literal);
  } else if (value.starts_with("0")) {
    std::int64_t literal = std::int64_t(std::stol(value, nullptr, 8));
    expr_ = cg->createIntegerLiteral(literal);
  } else {
    std::int64_t literal = std::int64_t(std::stol(value));
    expr_ = cg->createIntegerLiteral(literal);
  }
}

void ExpressionCodegen::visit(FloatLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(FloatLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(NullptrLiteralExpressionAST* ast) {
  std::int64_t v = 0;
  expr_ = cg->createIntegerLiteral(v);
}

void ExpressionCodegen::visit(StringLiteralExpressionAST* ast) {
  expr_ = cg->createStringLiteral(ast->literal);
}

void ExpressionCodegen::visit(UserDefinedStringLiteralExpressionAST* ast) {
  throw std::runtime_error(
      "visit(UserDefinedStringLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(IdExpressionAST* ast) {
  if (!ast->symbol) {
    expr_ = cg->createExternalId(fmt::format("{}", *ast->name->name));
    return;
  }

  if (ast->symbol->enclosingBlock() == ast->symbol->enclosingScope()->owner()) {
    expr_ = cg->createTemp(cg->getLocal(ast->symbol));
    return;
  }

  if (auto field = dynamic_cast<FieldSymbol*>(ast->symbol)) {
    expr_ = cg->createAccess(
        cg->createUnary(ir::UnaryOp::kStar, cg->createThis(QualifiedType())),
        ast->symbol);
    return;
  }

  if (ast->symbol->enclosingScope()->owner() == ast->symbol->enclosingClass()) {
    expr_ = cg->createAccess(
        cg->createUnary(ir::UnaryOp::kStar, cg->createThis(QualifiedType())),
        ast->symbol);
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
  if (!ast->constValue) return;

  auto value = const_value_cast<std::uint64_t>(*ast->constValue);

  expr_ = cg->createIntegerLiteral(value);
}

void ExpressionCodegen::visit(SizeofTypeExpressionAST* ast) {
  if (!ast->constValue) return;

  auto value = const_value_cast<std::uint64_t>(*ast->constValue);

  expr_ = cg->createIntegerLiteral(value);
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
  if (!ast->constValue) return;

  auto value = const_value_cast<std::uint64_t>(*ast->constValue);

  expr_ = cg->createIntegerLiteral(value);
}

void ExpressionCodegen::visit(UnaryTypeTraitsExpressionAST* ast) {
  if (!ast->constValue) return;
  expr_ = cg->createIntegerLiteral(std::get<std::uint64_t>(*ast->constValue));
}

void ExpressionCodegen::visit(BinaryTypeTraitsExpressionAST* ast) {
  if (!ast->constValue) return;
  expr_ = cg->createIntegerLiteral(std::get<std::uint64_t>(*ast->constValue));
}

void ExpressionCodegen::visit(UnaryExpressionAST* ast) {
  switch (ast->op) {
    case TokenKind::T_PLUS_PLUS:
    case TokenKind::T_MINUS_MINUS: {
      auto expr = cg->expression(ast->expression);

      auto local = cg->function()->addLocal(ast->expression->type);

      const int i = 1;

      const auto op = ast->op == TokenKind::T_PLUS_PLUS ? ir::BinaryOp::kPlus
                                                        : ir::BinaryOp::kMinus;

      cg->emitMove(cg->createTemp(local),
                   cg->createBinary(op, expr, cg->createIntegerLiteral(i)));

      cg->emitMove(expr, cg->createTemp(local));

      expr_ = expr;

      break;
    }

    case TokenKind::T_AMP:
    case TokenKind::T_STAR: {
      auto expr = cg->expression(ast->expression);
      auto op = convertUnaryOp(ast->op);
      expr_ = cg->createUnary(op, expr);
      break;
    }

    default: {
      auto expr = cg->reduce(ast->expression);
      auto op = convertUnaryOp(ast->op);
      expr_ = cg->createUnary(op, expr);
    }
  }  // switch
}

void ExpressionCodegen::visit(BinaryExpressionAST* ast) {
  if (ast->op == TokenKind::T_AMP_AMP || ast->op == TokenKind::T_BAR_BAR) {
    auto control = cg->unit()->control();
    auto types = control->types();
    QualifiedType intTy{types->integerType(IntegerKind::kInt, false)};
    auto local = cg->function()->addLocal(intTy);
    auto iftrue = cg->createBlock();
    auto iffalse = cg->createBlock();
    auto endif = cg->createBlock();
    cg->condition(ast, iftrue, iffalse);
    cg->place(iftrue);
    cg->emitMove(cg->createTemp(local),
                 cg->createIntegerLiteral(std::int32_t(1)));
    cg->emitJump(endif);
    cg->place(iffalse);
    cg->emitMove(cg->createTemp(local),
                 cg->createIntegerLiteral(std::int32_t(0)));
    cg->place(endif);
    expr_ = cg->createTemp(local);
    return;
  }

  if (ast->op == TokenKind::T_COMMA) {
    cg->statement(ast->leftExpression);
    expr_ = gen(ast->rightExpression);
    return;
  }

  auto left = cg->reduce(ast->leftExpression);
  auto right = cg->reduce(ast->rightExpression);
  auto op = convertBinaryOp(ast->op);
  expr_ = cg->createBinary(op, left, right);
}

void ExpressionCodegen::visit(AssignmentExpressionAST* ast) {
  auto left = gen(ast->leftExpression);
  auto right = reduce(ast->rightExpression);
  if (ast->leftExpression->type != ast->rightExpression->type)
    right = cg->createCast(ast->leftExpression->type, right);
  if (ast->op == TokenKind::T_EQUAL)
    cg->emitMove(left, right);
  else
    cg->emitMove(left,
                 cg->createBinary(convertAssignmentOp(ast->op), left, right));
  expr_ = left;
}

void ExpressionCodegen::visit(BracedTypeConstructionAST* ast) {
  throw std::runtime_error("visit(BracedTypeConstructionAST): not implemented");
}

void ExpressionCodegen::visit(TypeConstructionAST* ast) {
  throw std::runtime_error("visit(TypeConstructionAST): not implemented");
}

void ExpressionCodegen::visit(CallExpressionAST* ast) {
  auto base = gen(ast->baseExpression);
  std::vector<ir::Expr*> args;
  for (auto it = ast->expressionList; it; it = it->next) {
    args.push_back(reduce(it->value));
  }
  expr_ = cg->createCall(base, std::move(args));
}

void ExpressionCodegen::visit(SubscriptExpressionAST* ast) {
  auto base = gen(ast->baseExpression);
  auto index = reduce(ast->indexExpression);
  expr_ = cg->createSubscript(base, index);
}

void ExpressionCodegen::visit(MemberExpressionAST* ast) {
  auto base = gen(ast->baseExpression);
  if (ast->accessOp == TokenKind::T_MINUS_GREATER)
    base = cg->createUnary(ir::UnaryOp::kStar, base);
  expr_ = cg->createAccess(base, ast->symbol);
}

void ExpressionCodegen::visit(PostIncrExpressionAST* ast) {
  auto expr = cg->expression(ast->baseExpression);
  auto local = cg->function()->addLocal(ast->baseExpression->type);
  cg->emitMove(cg->createTemp(local), expr);
  const auto op = ast->op == TokenKind::T_PLUS_PLUS ? ir::BinaryOp::kPlus
                                                    : ir::BinaryOp::kMinus;
  const int i = 1;
  cg->emitMove(expr, cg->createBinary(op, expr, cg->createIntegerLiteral(i)));
  expr_ = cg->createTemp(local);
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
  auto expr = cg->expression(ast->expression);
  expr_ = cg->createCast(ast->typeId->type, expr);
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
