// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

class StatementCodegen final : public DefaultASTVisitor {
 public:
  explicit StatementCodegen(Codegen* cg);

  void gen(StatementAST* ast);
  void gen(ExpressionAST* ast);

 private:
  ir::IRBuilder& ir();

  ir::Expr* accept(InitializerAST* ast);

  using DefaultASTVisitor::visit;

  void visit(LabeledStatementAST* ast) override;
  void visit(CaseStatementAST* ast) override;
  void visit(DefaultStatementAST* ast) override;
  void visit(ExpressionStatementAST* ast) override;
  void visit(CompoundStatementAST* ast) override;
  void visit(IfStatementAST* ast) override;
  void visit(SwitchStatementAST* ast) override;
  void visit(WhileStatementAST* ast) override;
  void visit(DoStatementAST* ast) override;
  void visit(ForRangeStatementAST* ast) override;
  void visit(ForStatementAST* ast) override;
  void visit(BreakStatementAST* ast) override;
  void visit(ContinueStatementAST* ast) override;
  void visit(ReturnStatementAST* ast) override;
  void visit(GotoStatementAST* ast) override;
  void visit(CoroutineReturnStatementAST* ast) override;
  void visit(DeclarationStatementAST* ast) override;
  void visit(TryBlockStatementAST* ast) override;

  void visit(SimpleDeclarationAST* ast) override;

  // initializers
  void visit(EqualInitializerAST* ast) override;
  void visit(BracedInitListAST* ast) override;

 private:
  Codegen* cg;
  ir::Expr* initializer_ = nullptr;
};

}  // namespace cxx
