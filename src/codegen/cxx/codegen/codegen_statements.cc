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

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/codegen/codegen.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

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
  void storeReturnValue(ReturnStatementAST* ast);
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

  if (currentBlockMightHaveTerminator()) {
    auto deadBlock = newBlock();
    emitter_.setInsertionBlock(deadBlock);
  }

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
  auto loc = ast->firstSourceLocation();

  auto targetBlock = gen.newBlock();

  gen.branch(loc, targetBlock);
  gen.emitter_.setInsertionBlock(targetBlock);

  gen.defineCleanupLabel(loc, ast->identifier->name());

  gen.statement(ast->statement);
}

void Codegen::StatementVisitor::operator()(CaseStatementAST* ast) {
  auto block = gen.newBlock();

  gen.branch(ast->firstSourceLocation(), block);
  gen.emitter_.setInsertionBlock(block);

  if (ast->upperExpression) {
    ASTInterpreter interp{gen.unit_};
    if (auto value = interp.evaluate(ast->upperExpression)) {
      auto lowerValue = interp.toArithmeticType(
          ConstValue{ConstInt{ast->caseValue}}, gen.switch_.conditionType);
      auto upperValue =
          interp.toArithmeticType(*value, gen.switch_.conditionType);
      if (lowerValue && upperValue) {
        auto lower = std::get_if<ConstInt>(&*lowerValue);
        auto upper = std::get_if<ConstInt>(&*upperValue);
        if (lower && upper && *lower <= *upper)
          gen.switch_.ranges.push_back({*lower, *upper, block});
      }
    }
  } else {
    gen.switch_.caseValues.push_back(ast->caseValue);
    gen.switch_.caseDestinations.push_back(block);
  }
  gen.statement(ast->statement);
}

void Codegen::StatementVisitor::operator()(DefaultStatementAST* ast) {
  auto block = gen.newBlock();
  gen.branch(ast->firstSourceLocation(), block);
  gen.emitter_.setInsertionBlock(block);

  gen.switch_.defaultDestination = block;
  gen.statement(ast->statement);
}

void Codegen::StatementVisitor::operator()(ExpressionStatementAST* ast) {
  auto fullExpression = FullExpression{gen, lastTokenLocation(ast)};
  (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);
}

void Codegen::StatementVisitor::operator()(CompoundStatementAST* ast) {
  gen.pushCleanup();
  for (auto node : ListView{ast->statementList}) {
    gen.statement(node);
  }
  gen.popCleanup(ast->rbraceLoc);
}

void Codegen::StatementVisitor::operator()(IfStatementAST* ast) {
  auto trueBlock = gen.newBlock();
  auto falseBlock = gen.newBlock();
  auto mergeBlock = gen.newBlock();

  gen.pushCleanup();
  gen.statement(ast->initializer);
  gen.conditionWithCleanups(ast->condition, trueBlock, falseBlock);

  gen.emitter_.setInsertionBlock(trueBlock);
  gen.statement(ast->statement);
  gen.branch(
      ast->statement ? lastTokenLocation(ast->statement) : ast->rparenLoc,
      mergeBlock);
  gen.emitter_.setInsertionBlock(falseBlock);
  gen.statement(ast->elseStatement);
  gen.branch(
      ast->elseStatement ? lastTokenLocation(ast->elseStatement) : ast->elseLoc,
      mergeBlock);
  gen.emitter_.setInsertionBlock(mergeBlock);
  gen.popCleanup(lastTokenLocation(ast));
}

void Codegen::StatementVisitor::operator()(ConstevalIfStatementAST* ast) {
  if (!ast->isNot) {
    if (ast->elseStatement) gen.statement(ast->elseStatement);
  } else {
    if (ast->statement) gen.statement(ast->statement);
  }
}

void Codegen::StatementVisitor::operator()(SwitchStatementAST* ast) {
  gen.pushCleanup();
  gen.statement(ast->initializer);

  Switch previousSwitch;
  std::swap(gen.switch_, previousSwitch);

  auto beginSwitchBlock = gen.newBlock();
  auto bodySwitchBlock = gen.newBlock();
  auto endSwitchBlock = gen.newBlock();

  gen.branch(ast->firstSourceLocation(), beginSwitchBlock);

  gen.emitter_.setInsertionBlock(beginSwitchBlock);

  auto conditionResult = [&] {
    auto fullExpression = FullExpression{gen, ast->rparenLoc};
    return gen.expression(ast->condition);
  }();

  auto dispatchBlock = gen.emitter_.insertionBlock();

  auto elementTy = gen.emitter_.typeOf(conditionResult.value);
  if (gen.emitter_.typeKind(elementTy) != ir::TypeKind::Integer) {
    gen.unit_->error(ast->condition->firstSourceLocation(),
                     "switch condition is not of integral type");
    gen.emitter_.setInsertionBlock(endSwitchBlock);
    gen.popCleanup(lastTokenLocation(ast));
    std::swap(gen.switch_, previousSwitch);
    gen.emitter_.eraseBlock(bodySwitchBlock);
    return;
  }

  gen.emitter_.setInsertionBlock(bodySwitchBlock);

  Loop previousLoop;
  previousLoop.continueBlock = gen.loop_.continueBlock;
  previousLoop.continueCleanupDepth = gen.loop_.continueCleanupDepth;
  previousLoop.breakBlock = endSwitchBlock;
  previousLoop.breakCleanupDepth = gen.cleanupStack_.size();
  std::swap(gen.loop_, previousLoop);

  gen.switch_.conditionType = ast->condition->type;
  gen.statement(ast->statement);
  gen.branch(lastTokenLocation(ast), endSwitchBlock);

  gen.emitter_.setInsertionBlock(dispatchBlock);

  if (!gen.switch_.defaultDestination) {
    gen.switch_.defaultDestination = endSwitchBlock;
  }

  for (const auto& range : gen.switch_.ranges) {
    auto loc = ast->firstSourceLocation();
    auto lower = gen.emitter_.constantInt(loc, elementTy, range.lower);
    auto upper = gen.emitter_.constantInt(loc, elementTy, range.upper);
    auto offset = gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt,
                                        conditionResult.value, lower);
    auto width = gen.emitter_.binaryOp(loc, ir::BinaryOp::SubInt, upper, lower);
    auto matches = gen.emitter_.compareInt(
        loc, ir::IntPredicate::UnsignedLessEqual, offset, width);
    auto next = gen.newBlock();
    gen.emitter_.condBranch(loc, matches, range.destination, next);
    gen.emitter_.setInsertionBlock(next);
  }

  gen.emitter_.switchBranch(ast->firstSourceLocation(), conditionResult.value,
                            gen.switch_.defaultDestination,
                            gen.switch_.caseValues,
                            gen.switch_.caseDestinations);

  std::swap(gen.switch_, previousSwitch);
  std::swap(gen.loop_, previousLoop);

  gen.emitter_.setInsertionBlock(endSwitchBlock);
  gen.popCleanup(lastTokenLocation(ast));

  gen.emitter_.eraseBlock(bodySwitchBlock);
}

void Codegen::StatementVisitor::operator()(WhileStatementAST* ast) {
  auto beginLoopBlock = gen.newBlock();
  auto bodyLoopBlock = gen.newBlock();
  auto conditionFalseBlock = gen.newBlock();
  auto endLoopBlock = gen.newBlock();

  const auto iterationDepth = gen.cleanupStack_.size();

  Loop loop;
  loop.continueBlock = beginLoopBlock;
  loop.breakBlock = endLoopBlock;
  loop.continueCleanupDepth = iterationDepth;
  loop.breakCleanupDepth = iterationDepth;

  std::swap(gen.loop_, loop);

  gen.branch(ast->condition->firstSourceLocation(), beginLoopBlock);

  gen.emitter_.setInsertionBlock(beginLoopBlock);
  gen.pushCleanup();
  gen.conditionWithCleanups(ast->condition, bodyLoopBlock, conditionFalseBlock);

  gen.emitter_.setInsertionBlock(bodyLoopBlock);
  gen.statement(ast->statement);

  gen.emitBranchWithCleanups(lastTokenLocation(ast->statement), beginLoopBlock,
                             iterationDepth);

  gen.emitter_.setInsertionBlock(conditionFalseBlock);
  gen.emitBranchWithCleanups(lastTokenLocation(ast), endLoopBlock,
                             iterationDepth);

  gen.popCleanup(lastTokenLocation(ast));
  gen.emitter_.setInsertionBlock(endLoopBlock);

  std::swap(gen.loop_, loop);
}

void Codegen::StatementVisitor::operator()(DoStatementAST* ast) {
  auto loopBlock = gen.newBlock();
  auto conditionBlock = gen.newBlock();
  auto endLoopBlock = gen.newBlock();

  Loop loop;
  loop.continueBlock = conditionBlock;
  loop.breakBlock = endLoopBlock;
  loop.continueCleanupDepth = gen.cleanupStack_.size();
  loop.breakCleanupDepth = gen.cleanupStack_.size();
  std::swap(gen.loop_, loop);

  gen.branch(ast->statement->firstSourceLocation(), loopBlock);

  gen.emitter_.setInsertionBlock(loopBlock);
  gen.statement(ast->statement);

  gen.branch(lastTokenLocation(ast->statement), conditionBlock);

  gen.emitter_.setInsertionBlock(conditionBlock);
  gen.conditionWithCleanups(ast->expression, loopBlock, endLoopBlock);

  gen.emitter_.setInsertionBlock(endLoopBlock);

  std::swap(gen.loop_, loop);
}

void Codegen::StatementVisitor::operator()(ForRangeStatementAST* ast) {
  gen.statement(ast->range.statement);
}

void Codegen::StatementVisitor::operator()(ForStatementAST* ast) {
  gen.pushCleanup();
  gen.statement(ast->initializer);

  auto beginLoopBlock = gen.newBlock();
  auto loopBodyBlock = gen.newBlock();
  auto stepLoopBlock = gen.newBlock();
  auto conditionFalseBlock = gen.newBlock();
  auto endLoopBlock = gen.newBlock();

  const auto iterationDepth = gen.cleanupStack_.size();

  gen.pushCleanup();

  Loop loop;
  loop.continueBlock = stepLoopBlock;
  loop.breakBlock = endLoopBlock;
  loop.continueCleanupDepth = gen.cleanupStack_.size();
  loop.breakCleanupDepth = iterationDepth;
  std::swap(gen.loop_, loop);

  gen.branch(ast->firstSourceLocation(), beginLoopBlock);
  gen.emitter_.setInsertionBlock(beginLoopBlock);

  if (ast->condition) {
    gen.conditionWithCleanups(ast->condition, loopBodyBlock,
                              conditionFalseBlock);
  } else {
    gen.branch(ast->semicolonLoc, loopBodyBlock);
  }

  gen.emitter_.setInsertionBlock(loopBodyBlock);
  gen.statement(ast->statement);

  gen.branch(lastTokenLocation(ast->statement), stepLoopBlock);

  gen.emitter_.setInsertionBlock(stepLoopBlock);

  {
    auto fullExpression =
        FullExpression{gen, ast->expression ? lastTokenLocation(ast->expression)
                                            : ast->rparenLoc};
    (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);
  }

  gen.emitBranchWithCleanups(
      ast->expression ? lastTokenLocation(ast->expression) : ast->rparenLoc,
      beginLoopBlock, iterationDepth);

  gen.emitter_.setInsertionBlock(conditionFalseBlock);
  gen.emitBranchWithCleanups(lastTokenLocation(ast), endLoopBlock,
                             iterationDepth);

  gen.popCleanup(lastTokenLocation(ast));
  gen.emitter_.setInsertionBlock(endLoopBlock);
  gen.popCleanup(lastTokenLocation(ast));

  std::swap(gen.loop_, loop);
}

void Codegen::StatementVisitor::operator()(BreakStatementAST* ast) {
  if (auto target = gen.loop_.breakBlock) {
    gen.emitBranchWithCleanups(ast->firstSourceLocation(), target,
                               gen.loop_.breakCleanupDepth);
    return;
  }

  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::operator()(ContinueStatementAST* ast) {
  if (auto target = gen.loop_.continueBlock) {
    gen.emitBranchWithCleanups(ast->firstSourceLocation(), target,
                               gen.loop_.continueCleanupDepth);
    return;
  }

  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
}

void Codegen::StatementVisitor::storeReturnValue(ReturnStatementAST* ast) {
  if (!gen.exitValue_) {
    (void)gen.expression(ast->expression);
    return;
  }

  (void)gen.emitPrvalueInto(gen.exitValue_, gen.returnType_, ast->expression,
                            ast->firstSourceLocation());
  if (gen.traits.has_trivial_destructor(gen.returnType_)) return;
  if (auto destructor = gen.objectDestructor(gen.returnType_))
    gen.addCleanup(gen.exitValue_, gen.completeObjectDtor(destructor), 1, true);
}

void Codegen::StatementVisitor::operator()(ReturnStatementAST* ast) {
  auto returnScope = CleanupScopeGuard{gen, lastTokenLocation(ast)};
  {
    auto fullExpression = FullExpression{gen, lastTokenLocation(ast)};
    storeReturnValue(ast);
  }

  gen.emitBranchWithCleanups(ast->firstSourceLocation(), gen.exitBlock_, 0);
}

void Codegen::StatementVisitor::operator()(CoroutineReturnStatementAST* ast) {
  auto op = gen.emitTodoStmt(ast->firstSourceLocation(),
                             "CoroutineReturnStatementAST");
}

void Codegen::StatementVisitor::operator()(GotoStatementAST* ast) {
  if (ast->isIndirect) {
    auto loc = ast->firstSourceLocation();
    auto ptrResult = gen.expression(ast->expression);
    auto ptr = ptrResult.value;
    if (ast->expression &&
        ast->expression->valueCategory == ValueCategory::kLValue) {
      auto loadedType = gen.convertType(ast->expression->type);
      ptr = gen.emitter_.load(loc, loadedType, ptr,
                              gen.getAlignment(ast->expression->type));
    }
    gen.emitter_.indirectGoto(loc, ptr);
    auto nextBlock = gen.newBlock();
    gen.emitter_.setInsertionBlock(nextBlock);
    return;
  }

  gen.emitGotoLabel(ast->firstSourceLocation(), ast->identifier->name());

  auto nextBlock = gen.newBlock();
  gen.branch(ast->firstSourceLocation(), nextBlock);

  gen.emitter_.setInsertionBlock(nextBlock);
}

void Codegen::StatementVisitor::operator()(DeclarationStatementAST* ast) {
  auto fullExpression = FullExpression{gen, lastTokenLocation(ast)};
  auto declarationResult = gen.declaration(ast->declaration);
}

void Codegen::StatementVisitor::operator()(TryBlockStatementAST* ast) {
  if (gen.unit_->config().exceptionsEnabled) {
    gen.emitTry([&] { gen.statement(ast->statement); }, ast->handlerList,
                ast->firstSourceLocation());
    return;
  }
  gen.statement(ast->statement);
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
