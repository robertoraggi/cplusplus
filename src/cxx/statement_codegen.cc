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
#include <cxx/statement_codegen.h>

#include <stdexcept>

namespace cxx {

StatementCodegen::StatementCodegen(Codegen* cg) : cg(cg) {}

void StatementCodegen::gen(StatementAST* ast) {
  if (ast) ast->accept(this);
}

// StatementAST
void StatementCodegen::visit(LabeledStatementAST* ast) {
  throw std::runtime_error("visit(LabeledStatementAST): not implemented");
}

void StatementCodegen::visit(CaseStatementAST* ast) {
  throw std::runtime_error("visit(CaseStatementAST): not implemented");
}

void StatementCodegen::visit(DefaultStatementAST* ast) {
  throw std::runtime_error("visit(DefaultStatementAST): not implemented");
}

void StatementCodegen::visit(ExpressionStatementAST* ast) {
  if (ast->expression) {
    auto expr = cg->expression(ast->expression);
  }
}

void StatementCodegen::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next) {
    cg->statement(it->value);
  }
}

void StatementCodegen::visit(IfStatementAST* ast) {
  if (ast->elseStatement) {
    auto iftrue = cg->createBlock();
    auto iffalse = cg->createBlock();
    auto endif = cg->createBlock();
    cg->condition(ast->condition, iftrue, iffalse);
    cg->place(iftrue);
    cg->statement(ast->statement);
    cg->emitJump(endif);
    cg->place(iffalse);
    cg->statement(ast->elseStatement);
    cg->place(endif);
    return;
  }

  auto iftrue = cg->createBlock();
  auto endif = cg->createBlock();
  cg->condition(ast->condition, iftrue, endif);
  cg->place(iftrue);
  cg->statement(ast->statement);
  cg->place(endif);
}

void StatementCodegen::visit(SwitchStatementAST* ast) {
  throw std::runtime_error("visit(SwitchStatementAST): not implemented");
}

void StatementCodegen::visit(WhileStatementAST* ast) {
  auto topLoop = cg->createBlock();
  auto bodyLoop = cg->createBlock();
  auto endLoop = cg->createBlock();

  auto previousBreakBlock = cg->changeBreakBlock(endLoop);
  auto previousContinueblock = cg->changeContinueBlock(topLoop);

  cg->place(topLoop);
  cg->condition(ast->condition, bodyLoop, endLoop);
  cg->place(bodyLoop);
  cg->statement(ast->statement);
  cg->emitJump(topLoop);
  cg->place(endLoop);

  cg->changeBreakBlock(previousBreakBlock);
  cg->changeContinueBlock(previousContinueblock);
}

void StatementCodegen::visit(DoStatementAST* ast) {
  auto topLoop = cg->createBlock();
  auto bodyLoop = cg->createBlock();
  auto continueLoop = cg->createBlock();
  auto endLoop = cg->createBlock();

  auto previousBreakBlock = cg->changeBreakBlock(endLoop);
  auto previousContinueblock = cg->changeContinueBlock(continueLoop);

  cg->place(topLoop);
  cg->statement(ast->statement);
  cg->place(continueLoop);
  cg->condition(ast->expression, topLoop, endLoop);
  cg->place(endLoop);

  cg->changeBreakBlock(previousBreakBlock);
  cg->changeContinueBlock(previousContinueblock);
}

void StatementCodegen::visit(ForRangeStatementAST* ast) {
  throw std::runtime_error("visit(ForRangeStatementAST): not implemented");
}

void StatementCodegen::visit(ForStatementAST* ast) {
  auto topLoop = cg->createBlock();
  auto bodyLoop = cg->createBlock();
  auto continueLoop = cg->createBlock();
  auto endLoop = cg->createBlock();

  auto previousBreakBlock = cg->changeBreakBlock(endLoop);
  auto previousContinueBlock = cg->changeContinueBlock(continueLoop);

  cg->statement(ast->initializer);
  cg->place(topLoop);
  cg->condition(ast->condition, bodyLoop, endLoop);
  cg->place(bodyLoop);
  cg->statement(ast->statement);
  cg->place(continueLoop);
  cg->expression(ast->expression);
  cg->emitJump(topLoop);
  cg->place(endLoop);

  cg->changeBreakBlock(previousBreakBlock);
  cg->changeContinueBlock(previousContinueBlock);
}

void StatementCodegen::visit(BreakStatementAST* ast) {
  cg->emitJump(cg->breakBlock());
}

void StatementCodegen::visit(ContinueStatementAST* ast) {
  cg->emitJump(cg->continueBlock());
}

void StatementCodegen::visit(ReturnStatementAST* ast) {
  if (ast->expression) {
    auto value = cg->expression(ast->expression);
    cg->emitMove(cg->createTemp(cg->result()), value);
  }

  cg->emitJump(cg->exitBlock());
}

void StatementCodegen::visit(GotoStatementAST* ast) {
  throw std::runtime_error("visit(GotoStatementAST): not implemented");
}

void StatementCodegen::visit(CoroutineReturnStatementAST* ast) {
  throw std::runtime_error(
      "visit(CoroutineReturnStatementAST): not implemented");
}

void StatementCodegen::visit(DeclarationStatementAST* ast) {
  if (auto simpleDecl = dynamic_cast<SimpleDeclarationAST*>(ast->declaration)) {
    simpleDecl->accept(this);
  }
}

void StatementCodegen::visit(TryBlockStatementAST* ast) {
  throw std::runtime_error("visit(TryBlockStatementAST): not implemented");
}

void StatementCodegen::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->initDeclaratorList; it; it = it->next) {
    auto initDeclarator = it->value;
    auto initializer = accept(initDeclarator->initializer);
  }
}

ir::Expr* StatementCodegen::accept(InitializerAST* ast) {
  if (ast) ast->accept(this);
  return nullptr;
}

void StatementCodegen::visit(EqualInitializerAST* ast) {
  throw std::runtime_error("visit(EqualInitializerAST): not implemented");
}

void StatementCodegen::visit(BracedInitListAST* ast) {
  throw std::runtime_error("visit(BracedInitListAST): not implemented");
}

}  // namespace cxx
