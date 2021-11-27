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

#pragma once

#include <cxx/default_ast_visitor.h>
#include <cxx/ir_fwd.h>

namespace cxx {

class Codegen;

class ExpressionCodegen : protected DefaultASTVisitor {
 public:
  explicit ExpressionCodegen(Codegen* cg);

  ir::Expr* gen(ExpressionAST* ast);
  ir::Expr* reduce(ExpressionAST* ast);

 protected:
  using DefaultASTVisitor::visit;

  ir::UnaryOp convertUnaryOp(TokenKind tk) const;
  ir::BinaryOp convertBinaryOp(TokenKind tk) const;
  ir::BinaryOp convertAssignmentOp(TokenKind tk) const;

  void visit(ThisExpressionAST* ast) override;
  void visit(CharLiteralExpressionAST* ast) override;
  void visit(BoolLiteralExpressionAST* ast) override;
  void visit(IntLiteralExpressionAST* ast) override;
  void visit(FloatLiteralExpressionAST* ast) override;
  void visit(NullptrLiteralExpressionAST* ast) override;
  void visit(StringLiteralExpressionAST* ast) override;
  void visit(UserDefinedStringLiteralExpressionAST* ast) override;
  void visit(IdExpressionAST* ast) override;
  void visit(NestedExpressionAST* ast) override;
  void visit(RightFoldExpressionAST* ast) override;
  void visit(LeftFoldExpressionAST* ast) override;
  void visit(FoldExpressionAST* ast) override;
  void visit(LambdaExpressionAST* ast) override;
  void visit(SizeofExpressionAST* ast) override;
  void visit(SizeofTypeExpressionAST* ast) override;
  void visit(SizeofPackExpressionAST* ast) override;
  void visit(TypeidExpressionAST* ast) override;
  void visit(TypeidOfTypeExpressionAST* ast) override;
  void visit(AlignofExpressionAST* ast) override;
  void visit(TypeTraitsExpressionAST* ast) override;
  void visit(UnaryExpressionAST* ast) override;
  void visit(BinaryExpressionAST* ast) override;
  void visit(AssignmentExpressionAST* ast) override;
  void visit(BracedTypeConstructionAST* ast) override;
  void visit(TypeConstructionAST* ast) override;
  void visit(CallExpressionAST* ast) override;
  void visit(SubscriptExpressionAST* ast) override;
  void visit(MemberExpressionAST* ast) override;
  void visit(PostIncrExpressionAST* ast) override;
  void visit(ConditionalExpressionAST* ast) override;
  void visit(CastExpressionAST* ast) override;
  void visit(CppCastExpressionAST* ast) override;
  void visit(NewExpressionAST* ast) override;
  void visit(DeleteExpressionAST* ast) override;
  void visit(ThrowExpressionAST* ast) override;
  void visit(NoexceptExpressionAST* ast) override;

 protected:
  Codegen* cg = nullptr;
  ir::Expr* expr_ = nullptr;
};

}  // namespace cxx
