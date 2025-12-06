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
#include <cxx/memory_layout.h>
#include <cxx/names.h>

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

  // TODO: move to the op visitors
  // if (currentBlockMightHaveTerminator()) return;

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
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto targetBlock = gen.newBlock();

  gen.branch(loc, targetBlock);
  gen.builder_.setInsertionPointToEnd(targetBlock);

  mlir::cxx::LabelOp::create(gen.builder_, loc,
                             mlir::StringRef{ast->identifier->name()});
}

void Codegen::StatementVisitor::operator()(CaseStatementAST* ast) {
  auto block = gen.newBlock();

  gen.branch(gen.getLocation(ast->firstSourceLocation()), block);
  gen.builder_.setInsertionPointToEnd(block);

  gen.switch_.caseValues.push_back(ast->caseValue);
  gen.switch_.caseDestinations.push_back(block);
}

void Codegen::StatementVisitor::operator()(DefaultStatementAST* ast) {
  auto block = gen.newBlock();
  gen.branch(gen.getLocation(ast->firstSourceLocation()), block);
  gen.builder_.setInsertionPointToEnd(block);

  gen.switch_.defaultDestination = block;
}

void Codegen::StatementVisitor::operator()(ExpressionStatementAST* ast) {
  (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);
}

void Codegen::StatementVisitor::operator()(CompoundStatementAST* ast) {
  for (auto node : ListView{ast->statementList}) {
    gen.statement(node);
  }
}

void Codegen::StatementVisitor::operator()(IfStatementAST* ast) {
  auto trueBlock = gen.newBlock();
  auto falseBlock = gen.newBlock();
  auto mergeBlock = gen.newBlock();

  gen.statement(ast->initializer);
  gen.condition(ast->condition, trueBlock, falseBlock);

  gen.builder_.setInsertionPointToEnd(trueBlock);
  gen.statement(ast->statement);
  gen.branch(gen.getLocation(ast->statement->lastSourceLocation()), mergeBlock);
  gen.builder_.setInsertionPointToEnd(falseBlock);
  gen.statement(ast->elseStatement);
  gen.branch(gen.getLocation(ast->elseStatement
                                 ? ast->elseStatement->lastSourceLocation()
                                 : ast->elseLoc),
             mergeBlock);
  gen.builder_.setInsertionPointToEnd(mergeBlock);
}

void Codegen::StatementVisitor::operator()(ConstevalIfStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));

#if false
  gen.statement(ast->statement);
  gen.statement(ast->elseStatement);
#endif
}

void Codegen::StatementVisitor::operator()(SwitchStatementAST* ast) {
  gen.statement(ast->initializer);

  Switch previousSwitch;
  std::swap(gen.switch_, previousSwitch);

  auto beginSwitchBlock = gen.newBlock();
  auto bodySwitchBlock = gen.newBlock();
  auto endSwitchBlock = gen.newBlock();

  gen.branch(gen.getLocation(ast->firstSourceLocation()), beginSwitchBlock);

  gen.builder_.setInsertionPointToEnd(bodySwitchBlock);

  Loop previousLoop{gen.loop_.continueBlock, endSwitchBlock};
  std::swap(gen.loop_, previousLoop);

  gen.statement(ast->statement);
  gen.branch(gen.getLocation(ast->lastSourceLocation()), endSwitchBlock);

  gen.builder_.setInsertionPointToEnd(beginSwitchBlock);

  auto conditionResult = gen.expression(ast->condition);

  mlir::cxx::SwitchOp::create(
      gen.builder_, gen.getLocation(ast->firstSourceLocation()),
      conditionResult.value, gen.switch_.defaultDestination, {},
      gen.switch_.caseValues, gen.switch_.caseDestinations,
      mlir::SmallVector<mlir::ValueRange>(gen.switch_.caseValues.size()));

  std::swap(gen.switch_, previousSwitch);
  std::swap(gen.loop_, previousLoop);

  gen.builder_.setInsertionPointToEnd(endSwitchBlock);

  bodySwitchBlock->erase();
}

void Codegen::StatementVisitor::operator()(WhileStatementAST* ast) {
  auto beginLoopBlock = gen.newBlock();
  auto bodyLoopBlock = gen.newBlock();
  auto endLoopBlock = gen.newBlock();

  Loop loop{beginLoopBlock, endLoopBlock};

  std::swap(gen.loop_, loop);

  gen.branch(gen.getLocation(ast->condition->firstSourceLocation()),
             beginLoopBlock);

  gen.builder_.setInsertionPointToEnd(beginLoopBlock);
  gen.condition(ast->condition, bodyLoopBlock, endLoopBlock);

  gen.builder_.setInsertionPointToEnd(bodyLoopBlock);
  gen.statement(ast->statement);

  gen.branch(gen.getLocation(ast->statement->lastSourceLocation()),
             beginLoopBlock);
  gen.builder_.setInsertionPointToEnd(endLoopBlock);

  std::swap(gen.loop_, loop);
}

void Codegen::StatementVisitor::operator()(DoStatementAST* ast) {
  auto loopBlock = gen.newBlock();
  auto conditionBlock = gen.newBlock();
  auto endLoopBlock = gen.newBlock();

  Loop loop{conditionBlock, endLoopBlock};
  std::swap(gen.loop_, loop);

  gen.branch(gen.getLocation(ast->statement->firstSourceLocation()), loopBlock);

  gen.builder_.setInsertionPointToEnd(loopBlock);
  gen.statement(ast->statement);

  gen.branch(gen.getLocation(ast->statement->lastSourceLocation()),
             conditionBlock);

  gen.builder_.setInsertionPointToEnd(conditionBlock);
  gen.condition(ast->expression, loopBlock, endLoopBlock);

  gen.builder_.setInsertionPointToEnd(endLoopBlock);

  std::swap(gen.loop_, loop);
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
  gen.statement(ast->initializer);

  auto beginLoopBlock = gen.newBlock();
  auto loopBodyBlock = gen.newBlock();
  auto stepLoopBlock = gen.newBlock();
  auto endLoopBlock = gen.newBlock();

  Loop loop{stepLoopBlock, endLoopBlock};
  std::swap(gen.loop_, loop);

  gen.branch(gen.getLocation(ast->firstSourceLocation()), beginLoopBlock);
  gen.builder_.setInsertionPointToEnd(beginLoopBlock);

  if (ast->condition) {
    gen.condition(ast->condition, loopBodyBlock, endLoopBlock);
  } else {
    gen.branch(gen.getLocation(ast->semicolonLoc), loopBodyBlock);
  }

  gen.builder_.setInsertionPointToEnd(loopBodyBlock);
  gen.statement(ast->statement);

  gen.branch(gen.getLocation(ast->statement->lastSourceLocation()),
             stepLoopBlock);

  gen.builder_.setInsertionPointToEnd(stepLoopBlock);

  (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);

  gen.branch(
      gen.getLocation(ast->expression ? ast->expression->lastSourceLocation()
                                      : ast->rparenLoc),
      beginLoopBlock);

  gen.builder_.setInsertionPointToEnd(endLoopBlock);

  std::swap(gen.loop_, loop);
}

void Codegen::StatementVisitor::operator()(BreakStatementAST* ast) {
  if (auto target = gen.loop_.breakBlock) {
    auto loc = gen.getLocation(ast->firstSourceLocation());
    mlir::cf::BranchOp::create(gen.builder_, loc, target, {});
    return;
  }

  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ContinueStatementAST* ast) {
  if (auto target = gen.loop_.continueBlock) {
    mlir::cf::BranchOp::create(
        gen.builder_, gen.getLocation(ast->firstSourceLocation()), target);
    return;
  }

  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ReturnStatementAST* ast) {
  auto value = gen.expression(ast->expression);

  auto loc = gen.getLocation(ast->firstSourceLocation());

  if (gen.exitValue_) {
    mlir::cxx::StoreOp::create(gen.builder_, loc, value.value,
                               gen.exitValue_.getResult());
  }

  mlir::cf::BranchOp::create(gen.builder_, loc, gen.exitBlock_);
}

void Codegen::StatementVisitor::operator()(CoroutineReturnStatementAST* ast) {
  auto op = gen.emitTodoStmt(ast->firstSourceLocation(),
                             "CoroutineReturnStatementAST");

#if false
  auto expressionResult = gen.expression(ast->expression);
#endif
}

void Codegen::StatementVisitor::operator()(GotoStatementAST* ast) {
  if (ast->isIndirect) {
    (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
    return;
  }

  mlir::cxx::GotoOp::create(gen.builder_,
                            gen.getLocation(ast->firstSourceLocation()),
                            mlir::ValueRange{}, ast->identifier->name());

  auto nextBlock = gen.newBlock();
  gen.branch(gen.getLocation(ast->firstSourceLocation()), nextBlock);

  gen.builder_.setInsertionPointToEnd(nextBlock);
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
