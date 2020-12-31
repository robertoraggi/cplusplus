// Copyright (c) 2020 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/statement_ast_visitor.h>

namespace cxx {

AST::~AST() = default;

void LabeledStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void CaseStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void DefaultStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ExpressionStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void CompoundStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void IfStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void SwitchStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void WhileStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void DoStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ForRangeStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ForStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void BreakStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ContinueStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void ReturnStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void GotoStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void CoroutineReturnStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

void DeclarationStatementAST::visit(StatementASTVisitor* visitor) {
  visitor->visit(this);
}

}  // namespace cxx