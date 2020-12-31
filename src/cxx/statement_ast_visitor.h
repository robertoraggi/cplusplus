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

#pragma once

#include <cxx/ast_fwd.h>

namespace cxx {

struct StatementASTVisitor {
  StatementASTVisitor();
  virtual ~StatementASTVisitor();

  virtual void visit(LabeledStatementAST*) = 0;
  virtual void visit(CaseStatementAST*) = 0;
  virtual void visit(DefaultStatementAST*) = 0;
  virtual void visit(ExpressionStatementAST*) = 0;
  virtual void visit(CompoundStatementAST*) = 0;
  virtual void visit(IfStatementAST*) = 0;
  virtual void visit(SwitchStatementAST*) = 0;
  virtual void visit(WhileStatementAST*) = 0;
  virtual void visit(DoStatementAST*) = 0;
  virtual void visit(ForRangeStatementAST*) = 0;
  virtual void visit(ForStatementAST*) = 0;
  virtual void visit(BreakStatementAST*) = 0;
  virtual void visit(ContinueStatementAST*) = 0;
  virtual void visit(ReturnStatementAST*) = 0;
  virtual void visit(GotoStatementAST*) = 0;
  virtual void visit(CoroutineReturnStatementAST*) = 0;
  virtual void visit(DeclarationStatementAST*) = 0;
};

}  // namespace cxx
