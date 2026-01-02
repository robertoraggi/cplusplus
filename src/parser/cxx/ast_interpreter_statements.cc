// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct ASTInterpreter::StatementVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(LabeledStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(CaseStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(DefaultStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ExpressionStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(CompoundStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(IfStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ConstevalIfStatementAST* ast)
      -> StatementResult;

  [[nodiscard]] auto operator()(SwitchStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(WhileStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(DoStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ForRangeStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ForStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(BreakStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ContinueStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(ReturnStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(CoroutineReturnStatementAST* ast)
      -> StatementResult;

  [[nodiscard]] auto operator()(GotoStatementAST* ast) -> StatementResult;

  [[nodiscard]] auto operator()(DeclarationStatementAST* ast)
      -> StatementResult;

  [[nodiscard]] auto operator()(TryBlockStatementAST* ast) -> StatementResult;
};

struct ASTInterpreter::ExceptionDeclarationVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(EllipsisExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;

  [[nodiscard]] auto operator()(TypeExceptionDeclarationAST* ast)
      -> ExceptionDeclarationResult;
};

auto ASTInterpreter::statement(StatementAST* ast) -> StatementResult {
  if (ast) return visit(StatementVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::handler(HandlerAST* ast) -> HandlerResult {
  if (!ast) return {};

  auto exceptionDeclarationResult =
      exceptionDeclaration(ast->exceptionDeclaration);
  auto statementResult = statement(ast->statement);

  return {};
}

auto ASTInterpreter::exceptionDeclaration(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationResult {
  if (ast) return visit(ExceptionDeclarationVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementResult {
  for (auto node : ListView{ast->statementList}) {
    auto value = interp.statement(node);
  }

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementResult {
  auto initializerResult = interp.statement(ast->initializer);
  auto conditionResult = interp.expression(ast->condition);
  auto statementResult = interp.statement(ast->statement);
  auto elseStatementResult = interp.statement(ast->elseStatement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ConstevalIfStatementAST* ast)
    -> StatementResult {
  auto statementResult = interp.statement(ast->statement);
  auto elseStatementResult = interp.statement(ast->elseStatement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementResult {
  auto initializerResult = interp.statement(ast->initializer);
  auto conditionResult = interp.expression(ast->condition);
  auto statementResult = interp.statement(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(WhileStatementAST* ast)
    -> StatementResult {
  auto conditionResult = interp.expression(ast->condition);
  auto statementResult = interp.statement(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementResult {
  auto statementResult = interp.statement(ast->statement);
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementResult {
  auto initializerResult = interp.statement(ast->initializer);
  auto rangeDeclarationResult = interp.declaration(ast->rangeDeclaration);
  auto rangeInitializerResult = interp.expression(ast->rangeInitializer);
  auto statementResult = interp.statement(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementResult {
  auto initializerResult = interp.statement(ast->initializer);
  auto conditionResult = interp.expression(ast->condition);
  auto expressionResult = interp.expression(ast->expression);
  auto statementResult = interp.statement(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(BreakStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(
    CoroutineReturnStatementAST* ast) -> StatementResult {
  auto expressionResult = interp.expression(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementResult {
  auto declarationResult = interp.declaration(ast->declaration);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementResult {
  auto statementResult = interp.statement(ast->statement);

  for (auto node : ListView{ast->handlerList}) {
    auto value = interp.handler(node);
  }

  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = interp.specifier(node);
  }

  auto declaratorResult = interp.declarator(ast->declarator);

  return {};
}

}  // namespace cxx
