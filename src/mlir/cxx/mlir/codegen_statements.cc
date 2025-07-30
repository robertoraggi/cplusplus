// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>

// mlir
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

namespace cxx {

struct Codegen::StatementVisitor {
  Codegen& gen;

  [[nodiscard]] auto control() const -> Control* { return gen.control(); }

  void operator()(LabeledStatementAST* ast);
  void operator()(CaseStatementAST* ast);
  void operator()(DefaultStatementAST* ast);
  void operator()(ExpressionStatementAST* ast);
  void operator()(CompoundStatementAST* ast);
  void operator()(IfStatementAST* ast);
  void operator()(ConstevalIfStatementAST* ast);
  void operator()(SwitchStatementAST* ast);
  void operator()(WhileStatementAST* ast);
  void operator()(DoStatementAST* ast);
  void operator()(ForRangeStatementAST* ast);
  void operator()(ForStatementAST* ast);
  void operator()(BreakStatementAST* ast);
  void operator()(ContinueStatementAST* ast);
  void operator()(ReturnStatementAST* ast);
  void operator()(CoroutineReturnStatementAST* ast);
  void operator()(GotoStatementAST* ast);
  void operator()(DeclarationStatementAST* ast);
  void operator()(TryBlockStatementAST* ast);
};

struct Codegen::ExceptionDeclarationVisitor {
  Codegen& gen;

  auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;

  auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
};

void Codegen::statement(StatementAST* ast) {
  if (!ast) return;

  if (currentBlockMightHaveTerminator()) return;

  visit(StatementVisitor{*this}, ast);
}

auto Codegen::exceptionDeclaration(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationResult {
  if (ast) return visit(ExceptionDeclarationVisitor{*this}, ast);
  return {};
}

auto Codegen::handler(HandlerAST* ast) -> HandlerResult {
  if (!ast) return {};

  auto exceptionDeclarationResult =
      exceptionDeclaration(ast->exceptionDeclaration);

  statement(ast->statement);

  return {};
}

void Codegen::StatementVisitor::operator()(LabeledStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(CaseStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(DefaultStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ExpressionStatementAST* ast) {
  auto expressionResult = gen.expression(ast->expression);
}

void Codegen::StatementVisitor::operator()(CompoundStatementAST* ast) {
  for (auto node : ListView{ast->statementList}) {
    gen.statement(node);
  }
}

void Codegen::StatementVisitor::operator()(IfStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto conditionResult = gen.expression(ast->condition);
  gen.statement(ast->statement);
  gen.statement(ast->elseStatement);
#endif
}

void Codegen::StatementVisitor::operator()(ConstevalIfStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);
  gen.statement(ast->elseStatement);
#endif
}

void Codegen::StatementVisitor::operator()(SwitchStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto conditionResult = gen.expression(ast->condition);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(WhileStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  auto conditionResult = gen.expression(ast->condition);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(DoStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(ForRangeStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto rangeDeclarationResult = gen(ast->rangeDeclaration);
  auto rangeInitializerResult = gen.expression(ast->rangeInitializer);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(ForStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->initializer);
  auto conditionResult = gen.expression(ast->condition);
  auto expressionResult = gen.expression(ast->expression);
  gen.statement(ast->statement);
#endif
}

void Codegen::StatementVisitor::operator()(BreakStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ContinueStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ReturnStatementAST* ast) {
  // (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

  auto value = gen.expression(ast->expression);

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif

  auto loc = gen.getLocation(ast->firstSourceLocation());

  mlir::SmallVector<mlir::Value> results;

  if (gen.exitValue_) {
    gen.builder_.create<mlir::cxx::StoreOp>(loc, value.value,
                                            gen.exitValue_.getResult());

    results.push_back(gen.exitValue_);
  }

  gen.builder_.create<mlir::cf::BranchOp>(loc, results, gen.exitBlock_);
}

void Codegen::StatementVisitor::operator()(CoroutineReturnStatementAST* ast) {
  auto op = gen.emitTodoStmt(ast->firstSourceLocation(),
                             "CoroutineReturnStatementAST");

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(GotoStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(DeclarationStatementAST* ast) {
  auto declarationResult = gen.declaration(ast->declaration);
}

void Codegen::StatementVisitor::operator()(TryBlockStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);

  for (auto node : ListView{ast->handlerList}) {
    auto value = gen(node);
  }
#endif
}

auto Codegen::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  return {};
}

auto Codegen::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = gen.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = gen.specifier(node);
  }

  auto declaratorResult = gen.declarator(ast->declarator);

  return {};
}

}  // namespace cxx