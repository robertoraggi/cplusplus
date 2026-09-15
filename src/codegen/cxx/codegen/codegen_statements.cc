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
  void emitRangeElementBindings(ForRangeStatementAST* ast);
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

  gen.emitter_.defineLabel(loc, ast->identifier->name(),
                           static_cast<std::int64_t>(gen.cleanupStack_.size()));

  gen.statement(ast->statement);
}

void Codegen::StatementVisitor::operator()(CaseStatementAST* ast) {
  auto block = gen.newBlock();

  gen.branch(ast->firstSourceLocation(), block);
  gen.emitter_.setInsertionBlock(block);

  gen.switch_.caseValues.push_back(ast->caseValue);
  gen.switch_.caseDestinations.push_back(block);
}

void Codegen::StatementVisitor::operator()(DefaultStatementAST* ast) {
  auto block = gen.newBlock();
  gen.branch(ast->firstSourceLocation(), block);
  gen.emitter_.setInsertionBlock(block);

  gen.switch_.defaultDestination = block;
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

  gen.statement(ast->statement);
  gen.branch(lastTokenLocation(ast), endSwitchBlock);

  gen.emitter_.setInsertionBlock(dispatchBlock);

  if (!gen.switch_.defaultDestination) {
    gen.switch_.defaultDestination = endSwitchBlock;
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

namespace {

auto rangeDeclarationVariable(DeclarationAST* rangeDeclaration)
    -> VariableSymbol* {
  if (auto simpleDecl = ast_cast<SimpleDeclarationAST>(rangeDeclaration)) {
    auto initDecl = simpleDecl->initDeclaratorList
                        ? simpleDecl->initDeclaratorList->value
                        : nullptr;
    return initDecl ? symbol_cast<VariableSymbol>(initDecl->symbol) : nullptr;
  }

  if (auto structuredBinding =
          ast_cast<StructuredBindingDeclarationAST>(rangeDeclaration)) {
    if (auto hidden = structuredBinding->hiddenVariable)
      return symbol_cast<VariableSymbol>(hidden->symbol);
  }

  return nullptr;
}

}  // namespace

void Codegen::StatementVisitor::emitRangeElementBindings(
    ForRangeStatementAST* ast) {
  auto structuredBinding =
      ast_cast<StructuredBindingDeclarationAST>(ast->rangeDeclaration);
  if (!structuredBinding) return;

  for (auto node : ListView{structuredBinding->bindingDeclaratorList}) {
    auto var = symbol_cast<VariableSymbol>(node->symbol);
    if (!var) continue;
    gen.emitLocalVariableInit(var, node->initializer);
  }
}

void Codegen::StatementVisitor::operator()(ForRangeStatementAST* ast) {
  auto loc = ast->firstSourceLocation();

  auto forRangeScope = CleanupScopeGuard{gen, lastTokenLocation(ast)};

  gen.statement(ast->initializer);

  if (ast->rangeVariable && ast->beginVariable && ast->endVariable &&
      ast->beginInitializer && ast->endInitializer && ast->condition &&
      ast->increment && ast->element) {
    gen.emitLocalVariableInit(ast->rangeVariable, ast->rangeInitializer);
    gen.emitLocalVariableInit(ast->beginVariable, ast->beginInitializer);
    gen.emitLocalVariableInit(ast->endVariable, ast->endInitializer);

    auto conditionBlock = gen.newBlock();
    auto bodyBlock = gen.newBlock();
    auto stepBlock = gen.newBlock();
    auto exitBlock = gen.newBlock();

    const auto iterationDepth = gen.cleanupStack_.size();
    gen.pushCleanup();

    Loop loop;
    loop.continueBlock = stepBlock;
    loop.breakBlock = exitBlock;
    loop.continueCleanupDepth = iterationDepth;
    loop.breakCleanupDepth = iterationDepth;
    std::swap(gen.loop_, loop);

    gen.branch(loc, conditionBlock);
    gen.emitter_.setInsertionBlock(conditionBlock);
    gen.conditionWithCleanups(ast->condition, bodyBlock, exitBlock);

    gen.emitter_.setInsertionBlock(bodyBlock);
    if (auto loopVar = rangeDeclarationVariable(ast->rangeDeclaration))
      gen.emitLocalVariableInit(loopVar, ast->element);
    emitRangeElementBindings(ast);
    gen.statement(ast->statement);
    auto bodyEndLoc = ast->rparenLoc;
    if (ast->statement) bodyEndLoc = lastTokenLocation(ast->statement);
    gen.emitBranchWithCleanups(bodyEndLoc, stepBlock, iterationDepth);
    gen.popCleanup(bodyEndLoc);

    gen.emitter_.setInsertionBlock(stepBlock);
    (void)gen.expression(ast->increment);
    gen.branch(loc, conditionBlock);

    gen.emitter_.setInsertionBlock(exitBlock);
    std::swap(gen.loop_, loop);
    return;
  }

  auto rangeResult = gen.expression(ast->rangeInitializer);

  auto rangeType = ast->rangeInitializer->type;
  if (!rangeType) {
    (void)gen.emitTodoStmt(ast->firstSourceLocation(), "for-range: no type");
    return;
  }
  rangeType = gen.traits.remove_cvref(rangeType);

  ir::ValueRef beginVal, endVal;
  bool isPointerIterator = false;
  FunctionSymbol* derefFunc = nullptr;
  FunctionSymbol* incrFunc = nullptr;
  FunctionSymbol* neqFunc = nullptr;

  if (auto arrayType = type_cast<BoundedArrayType>(rangeType)) {
    isPointerIterator = true;

    auto elementIrType = gen.convertType(arrayType->elementType());
    auto ptrType = gen.emitter_.pointerType(elementIrType);

    auto intTy = gen.emitter_.integerType(64);

    auto zeroOp = gen.emitter_.constantInt(loc, intTy, 0);
    beginVal = gen.emitter_.pointerAdd(loc, ptrType, rangeResult.value, zeroOp);

    auto sizeOp = gen.emitter_.constantInt(loc, intTy, arrayType->size());
    endVal = gen.emitter_.pointerAdd(loc, ptrType, beginVal, sizeOp);
  } else if (type_cast<ClassType>(rangeType)) {
    auto beginFunc = ast->beginFunction;
    auto endFunc = ast->endFunction;

    if (!beginFunc || !endFunc) {
      (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                             "for-range: no begin/end");
      return;
    }

    if (ast->usesMemberBeginEnd) {
      beginVal = gen.emitCall(ast->colonLoc, beginFunc, rangeResult, {}).value;
      endVal = gen.emitCall(ast->colonLoc, endFunc, rangeResult, {}).value;
    } else {
      beginVal =
          gen.emitCall(ast->colonLoc, beginFunc, {}, {rangeResult}).value;
      endVal = gen.emitCall(ast->colonLoc, endFunc, {}, {rangeResult}).value;
    }

    if (!beginVal || !endVal) {
      (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                             "for-range: begin/end call failed");
      return;
    }

    isPointerIterator = ast->isPointerIterator;
    derefFunc = ast->derefFunction;
    incrFunc = ast->incrementFunction;
    neqFunc = ast->notEqualFunction;

    if (!isPointerIterator && (!derefFunc || !incrFunc || !neqFunc)) {
      (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                             "for-range: missing iterator ops");
      return;
    }
  } else {
    (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                           "for-range: unsupported range type");
    return;
  }

  auto condBlock = gen.newBlock();
  auto bodyBlock = gen.newBlock();
  auto stepBlock = gen.newBlock();
  auto exitBlock = gen.newBlock();

  const auto iterationDepth = gen.cleanupStack_.size();
  gen.pushCleanup();

  auto iterType = gen.emitter_.typeOf(beginVal);
  auto iterPtrType = gen.emitter_.pointerType(iterType);
  auto iterAlloca = gen.emitter_.allocate(loc, iterPtrType, 8);
  gen.emitter_.store(loc, beginVal, iterAlloca, 8);

  auto endPtrType = gen.emitter_.pointerType(gen.emitter_.typeOf(endVal));
  auto endAlloca = gen.emitter_.allocate(loc, endPtrType, 8);
  gen.emitter_.store(loc, endVal, endAlloca, 8);

  Loop loop;
  loop.continueBlock = stepBlock;
  loop.breakBlock = exitBlock;
  loop.continueCleanupDepth = iterationDepth;
  loop.breakCleanupDepth = iterationDepth;
  std::swap(gen.loop_, loop);

  gen.branch(loc, condBlock);

  gen.emitter_.setInsertionBlock(condBlock);

  auto iterLoad = gen.emitter_.load(loc, iterType, iterAlloca, 8);
  auto endLoad =
      gen.emitter_.load(loc, gen.emitter_.typeOf(endVal), endAlloca, 8);

  ir::ValueRef condVal;
  if (isPointerIterator || !neqFunc) {
    auto intPtrType = gen.emitter_.integerType(64);
    auto leftInt = gen.emitter_.pointerToInt(loc, intPtrType, iterLoad);
    auto rightInt = gen.emitter_.pointerToInt(loc, intPtrType, endLoad);
    condVal = gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual, leftInt,
                                      rightInt);
  } else {
    auto neqParent = neqFunc->parent();
    bool isMemberNeq = neqParent && neqParent->kind() == SymbolKind::kClass;

    ir::ValueRef firstArg = iterAlloca;
    ir::ValueRef secondArg = endAlloca;
    if (ast->notEqualReversed) std::swap(firstArg, secondArg);

    ExpressionResult neqResult;
    if (isMemberNeq) {
      neqResult =
          gen.emitCall(ast->colonLoc, neqFunc, {firstArg}, {{secondArg}});
    } else {
      neqResult =
          gen.emitCall(ast->colonLoc, neqFunc, {}, {{firstArg}, {secondArg}});
    }
    condVal = neqResult.value;

    if (condVal && ast->notEqualRewritten) {
      auto boolType = gen.emitter_.typeOf(condVal);
      auto trueConst = gen.emitter_.constantInt(loc, boolType, 1);
      condVal =
          gen.emitter_.binaryOp(loc, ir::BinaryOp::XorInt, condVal, trueConst);
    }
  }

  if (!condVal) {
    auto intPtrType = gen.emitter_.integerType(64);
    auto leftInt = gen.emitter_.pointerToInt(loc, intPtrType, iterLoad);
    auto rightInt = gen.emitter_.pointerToInt(loc, intPtrType, endLoad);
    condVal = gen.emitter_.compareInt(loc, ir::IntPredicate::NotEqual, leftInt,
                                      rightInt);
  }

  gen.emitter_.condBranch(loc, condVal, bodyBlock, exitBlock);

  gen.emitter_.setInsertionBlock(bodyBlock);

  auto loopVar = rangeDeclarationVariable(ast->rangeDeclaration);

  if (loopVar) {
    auto local = gen.findOrCreateLocal(loopVar);
    if (local) {
      auto iterInBody = gen.emitter_.load(loc, iterType, iterAlloca, 8);

      if (isPointerIterator) {
        if (gen.traits.is_reference(loopVar->type())) {
          gen.emitter_.store(loc, iterInBody, local.value(),
                             gen.getAlignment(loopVar->type()));
        } else {
          auto elemType =
              gen.convertType(gen.traits.remove_cvref(loopVar->type()));
          auto elem = gen.emitter_.load(
              loc, elemType, iterInBody,
              gen.getAlignment(gen.traits.remove_cvref(loopVar->type())));
          gen.emitter_.store(loc, elem, local.value(),
                             gen.getAlignment(loopVar->type()));
        }
      } else if (derefFunc) {
        auto derefResult =
            gen.emitCall(ast->colonLoc, derefFunc, {iterAlloca}, {});
        if (derefResult.value) {
          if (gen.traits.is_reference(loopVar->type())) {
            gen.emitter_.store(loc, derefResult.value, local.value(),
                               gen.getAlignment(loopVar->type()));
          } else {
            auto elemType =
                gen.convertType(gen.traits.remove_cvref(loopVar->type()));
            auto elem = gen.emitter_.load(
                loc, elemType, derefResult.value,
                gen.getAlignment(gen.traits.remove_cvref(loopVar->type())));
            gen.emitter_.store(loc, elem, local.value(),
                               gen.getAlignment(loopVar->type()));
          }
        }
      }
    }
  }

  emitRangeElementBindings(ast);

  gen.statement(ast->statement);
  auto bodyEndLoc = ast->rparenLoc;
  if (ast->statement) bodyEndLoc = lastTokenLocation(ast->statement);
  gen.emitBranchWithCleanups(bodyEndLoc, stepBlock, iterationDepth);
  gen.popCleanup(bodyEndLoc);

  gen.emitter_.setInsertionBlock(stepBlock);

  if (isPointerIterator) {
    auto iterInStep = gen.emitter_.load(loc, iterType, iterAlloca, 8);
    auto intTy = gen.emitter_.integerType(32);
    auto oneOp = gen.emitter_.constantInt(loc, intTy, 1);
    auto nextIter = gen.emitter_.pointerAdd(loc, iterType, iterInStep, oneOp);
    gen.emitter_.store(loc, nextIter, iterAlloca, 8);
  } else if (incrFunc) {
    (void)gen.emitCall(ast->colonLoc, incrFunc, {iterAlloca}, {});
  }

  gen.branch(loc, condBlock);

  gen.emitter_.setInsertionBlock(exitBlock);

  std::swap(gen.loop_, loop);
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
}

void Codegen::StatementVisitor::operator()(ReturnStatementAST* ast) {
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

  auto cleanupSnapshot = gen.collectCleanupSnapshot();

  gen.emitter_.gotoLabel(ast->firstSourceLocation(), ast->identifier->name(),
                         cleanupSnapshot);

  auto nextBlock = gen.newBlock();
  gen.branch(ast->firstSourceLocation(), nextBlock);

  gen.emitter_.setInsertionBlock(nextBlock);
}

void Codegen::StatementVisitor::operator()(DeclarationStatementAST* ast) {
  auto fullExpression = FullExpression{gen, lastTokenLocation(ast)};
  auto declarationResult = gen.declaration(ast->declaration);
}

void Codegen::StatementVisitor::operator()(TryBlockStatementAST* ast) {
  (void)gen.emitTodoStmt(ast->firstSourceLocation(), to_string(ast->kind()));
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
