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
#include <cxx/expression_codegen.h>

#include <stdexcept>

namespace cxx {

ExpressionCodegen::ExpressionCodegen(Codegen* codegen) {}

ir::Expr* ExpressionCodegen::gen(ExpressionAST* ast) {
  ir::Expr* expr = nullptr;
  std::swap(expr_, expr);
  ast->accept(this);
  std::swap(expr_, expr);
  return expr;
}

void ExpressionCodegen::visit(ThisExpressionAST* ast) {
  throw std::runtime_error("visit(ThisExpressionAST): not implemented");
}

void ExpressionCodegen::visit(CharLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(CharLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(BoolLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(BoolLiteralExpressionAST): not implemented");
}

void ExpressionCodegen::visit(IntLiteralExpressionAST* ast) {
  throw std::runtime_error("visit(IntLiteralExpressionAST): not implemented");
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
  throw std::runtime_error("visit(IdExpressionAST): not implemented");
}

void ExpressionCodegen::visit(NestedExpressionAST* ast) {
  throw std::runtime_error("visit(NestedExpressionAST): not implemented");
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
  throw std::runtime_error("visit(BinaryExpressionAST): not implemented");
}

void ExpressionCodegen::visit(AssignmentExpressionAST* ast) {
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
  throw std::runtime_error("visit(SubscriptExpressionAST): not implemented");
}

void ExpressionCodegen::visit(MemberExpressionAST* ast) {
  throw std::runtime_error("visit(MemberExpressionAST): not implemented");
}

void ExpressionCodegen::visit(ConditionalExpressionAST* ast) {
  throw std::runtime_error("visit(ConditionalExpressionAST): not implemented");
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
