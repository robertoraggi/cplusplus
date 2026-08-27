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
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/mlir/codegen.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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
    builder_.setInsertionPointToEnd(deadBlock);
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
  auto loc = gen.getLocation(ast->firstSourceLocation());

  auto targetBlock = gen.newBlock();

  gen.branch(loc, targetBlock);
  gen.builder_.setInsertionPointToEnd(targetBlock);

  mlir::cxx::LabelOp::create(
      gen.builder_, loc, mlir::StringRef{ast->identifier->name()},
      static_cast<std::int64_t>(gen.cleanupStack_.size()));

  gen.statement(ast->statement);
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
  auto fullExpression = FullExpression{gen, ast->lastSourceLocation()};
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

  gen.builder_.setInsertionPointToEnd(trueBlock);
  gen.statement(ast->statement);
  gen.branch(
      gen.getLocation(ast->statement ? ast->statement->lastSourceLocation()
                                     : ast->rparenLoc),
      mergeBlock);
  gen.builder_.setInsertionPointToEnd(falseBlock);
  gen.statement(ast->elseStatement);
  gen.branch(gen.getLocation(ast->elseStatement
                                 ? ast->elseStatement->lastSourceLocation()
                                 : ast->elseLoc),
             mergeBlock);
  gen.builder_.setInsertionPointToEnd(mergeBlock);
  gen.popCleanup(ast->lastSourceLocation());
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

  gen.branch(gen.getLocation(ast->firstSourceLocation()), beginSwitchBlock);

  gen.builder_.setInsertionPointToEnd(beginSwitchBlock);

  auto conditionResult = [&] {
    auto fullExpression = FullExpression{gen, ast->rparenLoc};
    return gen.expression(ast->condition);
  }();

  auto dispatchBlock = gen.builder_.getInsertionBlock();

  auto elementTy =
      mlir::TypeSwitch<mlir::Type, mlir::IntegerType>(
          conditionResult.value.getType())
          .Case<mlir::IntegerType>(
              [&](mlir::IntegerType ty) -> mlir::IntegerType { return ty; })
          .Default([](mlir::Type ty) -> mlir::IntegerType { return {}; });

  if (!elementTy) {
    gen.unit_->error(ast->condition->firstSourceLocation(),
                     "switch condition is not of integral type");
    gen.builder_.setInsertionPointToEnd(endSwitchBlock);
    gen.popCleanup(ast->lastSourceLocation());
    std::swap(gen.switch_, previousSwitch);
    bodySwitchBlock->erase();
    return;
  }

  gen.builder_.setInsertionPointToEnd(bodySwitchBlock);

  Loop previousLoop;
  previousLoop.continueBlock = gen.loop_.continueBlock;
  previousLoop.continueCleanupDepth = gen.loop_.continueCleanupDepth;
  previousLoop.breakBlock = endSwitchBlock;
  previousLoop.breakCleanupDepth = gen.cleanupStack_.size();
  std::swap(gen.loop_, previousLoop);

  gen.statement(ast->statement);
  gen.branch(gen.getLocation(ast->lastSourceLocation()), endSwitchBlock);

  gen.builder_.setInsertionPointToEnd(dispatchBlock);

  if (!gen.switch_.defaultDestination) {
    gen.switch_.defaultDestination = endSwitchBlock;
  }

  auto shapeType = mlir::VectorType::get(
      static_cast<std::int64_t>(gen.switch_.caseValues.size()),
      gen.builder_.getIntegerType(64));

  auto caseValuesAttr = mlir::cast<mlir::DenseIntElementsAttr>(
      mlir::DenseIntElementsAttr::get(shapeType, gen.switch_.caseValues)
          .mapValues(elementTy, [&](mlir::APInt v) {
            return mlir::APInt(elementTy.getIntOrFloatBitWidth(),
                               v.getZExtValue(), false, true);
          }));

  auto flag = conditionResult.value;

  std::vector<mlir::ValueRange> caseOperands(
      gen.switch_.caseDestinations.size(), mlir::ValueRange{});

  mlir::cf::SwitchOp::create(gen.builder_,
                             gen.getLocation(ast->firstSourceLocation()), flag,
                             gen.switch_.defaultDestination, {}, caseValuesAttr,
                             gen.switch_.caseDestinations, caseOperands);

  std::swap(gen.switch_, previousSwitch);
  std::swap(gen.loop_, previousLoop);

  gen.builder_.setInsertionPointToEnd(endSwitchBlock);
  gen.popCleanup(ast->lastSourceLocation());

  bodySwitchBlock->erase();
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

  gen.branch(gen.getLocation(ast->condition->firstSourceLocation()),
             beginLoopBlock);

  gen.builder_.setInsertionPointToEnd(beginLoopBlock);
  gen.pushCleanup();
  gen.conditionWithCleanups(ast->condition, bodyLoopBlock, conditionFalseBlock);

  gen.builder_.setInsertionPointToEnd(bodyLoopBlock);
  gen.statement(ast->statement);

  gen.emitBranchWithCleanups(ast->statement->lastSourceLocation(),
                             beginLoopBlock, iterationDepth);

  gen.builder_.setInsertionPointToEnd(conditionFalseBlock);
  gen.emitBranchWithCleanups(ast->lastSourceLocation(), endLoopBlock,
                             iterationDepth);

  gen.popCleanup(ast->lastSourceLocation());
  gen.builder_.setInsertionPointToEnd(endLoopBlock);

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

  gen.branch(gen.getLocation(ast->statement->firstSourceLocation()), loopBlock);

  gen.builder_.setInsertionPointToEnd(loopBlock);
  gen.statement(ast->statement);

  gen.branch(gen.getLocation(ast->statement->lastSourceLocation()),
             conditionBlock);

  gen.builder_.setInsertionPointToEnd(conditionBlock);
  gen.conditionWithCleanups(ast->expression, loopBlock, endLoopBlock);

  gen.builder_.setInsertionPointToEnd(endLoopBlock);

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
  auto loc = gen.getLocation(ast->firstSourceLocation());

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
    gen.builder_.setInsertionPointToEnd(conditionBlock);
    gen.conditionWithCleanups(ast->condition, bodyBlock, exitBlock);

    gen.builder_.setInsertionPointToEnd(bodyBlock);
    if (auto loopVar = rangeDeclarationVariable(ast->rangeDeclaration))
      gen.emitLocalVariableInit(loopVar, ast->element);
    emitRangeElementBindings(ast);
    gen.statement(ast->statement);
    auto bodyEndLoc = ast->rparenLoc;
    if (ast->statement) bodyEndLoc = ast->statement->lastSourceLocation();
    gen.emitBranchWithCleanups(bodyEndLoc, stepBlock, iterationDepth);
    gen.popCleanup(bodyEndLoc);

    gen.builder_.setInsertionPointToEnd(stepBlock);
    (void)gen.expression(ast->increment);
    gen.branch(loc, conditionBlock);

    gen.builder_.setInsertionPointToEnd(exitBlock);
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

  mlir::Value beginVal, endVal;
  bool isPointerIterator = false;
  FunctionSymbol* derefFunc = nullptr;
  FunctionSymbol* incrFunc = nullptr;
  FunctionSymbol* neqFunc = nullptr;

  if (auto arrayType = type_cast<BoundedArrayType>(rangeType)) {
    isPointerIterator = true;

    auto elementMlirType = gen.convertType(arrayType->elementType());
    auto ptrType = mlir::cxx::PointerType::get(gen.context_, elementMlirType);

    auto intTy = gen.builder_.getIntegerType(64);

    auto zeroOp = mlir::arith::ConstantOp::create(
        gen.builder_, loc, intTy, gen.builder_.getIntegerAttr(intTy, 0));
    beginVal = mlir::cxx::PtrAddOp::create(gen.builder_, loc, ptrType,
                                           rangeResult.value, zeroOp);

    auto sizeOp = mlir::arith::ConstantOp::create(
        gen.builder_, loc, intTy,
        gen.builder_.getIntegerAttr(intTy, arrayType->size()));
    endVal = mlir::cxx::PtrAddOp::create(gen.builder_, loc, ptrType, beginVal,
                                         sizeOp);
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

  auto iterType = beginVal.getType();
  auto iterPtrType = mlir::cxx::PointerType::get(gen.context_, iterType);
  auto iterAlloca =
      mlir::cxx::AllocaOp::create(gen.builder_, loc, iterPtrType, 8);
  mlir::cxx::StoreOp::create(gen.builder_, loc, beginVal, iterAlloca, 8);

  auto endPtrType = mlir::cxx::PointerType::get(gen.context_, endVal.getType());
  auto endAlloca =
      mlir::cxx::AllocaOp::create(gen.builder_, loc, endPtrType, 8);
  mlir::cxx::StoreOp::create(gen.builder_, loc, endVal, endAlloca, 8);

  Loop loop;
  loop.continueBlock = stepBlock;
  loop.breakBlock = exitBlock;
  loop.continueCleanupDepth = iterationDepth;
  loop.breakCleanupDepth = iterationDepth;
  std::swap(gen.loop_, loop);

  gen.branch(loc, condBlock);

  gen.builder_.setInsertionPointToEnd(condBlock);

  auto iterLoad =
      mlir::cxx::LoadOp::create(gen.builder_, loc, iterType, iterAlloca, 8);
  auto endLoad = mlir::cxx::LoadOp::create(gen.builder_, loc, endVal.getType(),
                                           endAlloca, 8);

  mlir::Value condVal;
  if (isPointerIterator || !neqFunc) {
    auto intPtrType = gen.builder_.getIntegerType(64);
    auto leftInt =
        mlir::cxx::PtrToIntOp::create(gen.builder_, loc, intPtrType, iterLoad);
    auto rightInt =
        mlir::cxx::PtrToIntOp::create(gen.builder_, loc, intPtrType, endLoad);
    condVal = mlir::arith::CmpIOp::create(
        gen.builder_, loc, mlir::arith::CmpIPredicate::ne, leftInt, rightInt);
  } else {
    auto neqParent = neqFunc->parent();
    bool isMemberNeq = neqParent && neqParent->kind() == SymbolKind::kClass;

    mlir::Value firstArg = iterAlloca;
    mlir::Value secondArg = endAlloca;
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
      auto boolType = condVal.getType();
      auto trueConst = mlir::arith::ConstantOp::create(
          gen.builder_, loc, boolType,
          gen.builder_.getIntegerAttr(boolType, 1));
      condVal =
          mlir::arith::XOrIOp::create(gen.builder_, loc, condVal, trueConst);
    }
  }

  if (!condVal) {
    auto intPtrType = gen.builder_.getIntegerType(64);
    auto leftInt =
        mlir::cxx::PtrToIntOp::create(gen.builder_, loc, intPtrType, iterLoad);
    auto rightInt =
        mlir::cxx::PtrToIntOp::create(gen.builder_, loc, intPtrType, endLoad);
    condVal = mlir::arith::CmpIOp::create(
        gen.builder_, loc, mlir::arith::CmpIPredicate::ne, leftInt, rightInt);
  }

  mlir::cf::CondBranchOp::create(gen.builder_, loc, condVal, bodyBlock, {},
                                 exitBlock, {});

  gen.builder_.setInsertionPointToEnd(bodyBlock);

  auto loopVar = rangeDeclarationVariable(ast->rangeDeclaration);

  if (loopVar) {
    auto local = gen.findOrCreateLocal(loopVar);
    if (local) {
      auto iterInBody =
          mlir::cxx::LoadOp::create(gen.builder_, loc, iterType, iterAlloca, 8);

      if (isPointerIterator) {
        if (gen.traits.is_reference(loopVar->type())) {
          mlir::cxx::StoreOp::create(gen.builder_, loc, iterInBody,
                                     local.value(),
                                     gen.getAlignment(loopVar->type()));
        } else {
          auto elemType =
              gen.convertType(gen.traits.remove_cvref(loopVar->type()));
          auto elem = mlir::cxx::LoadOp::create(
              gen.builder_, loc, elemType, iterInBody,
              gen.getAlignment(gen.traits.remove_cvref(loopVar->type())));
          mlir::cxx::StoreOp::create(gen.builder_, loc, elem, local.value(),
                                     gen.getAlignment(loopVar->type()));
        }
      } else if (derefFunc) {
        auto derefResult =
            gen.emitCall(ast->colonLoc, derefFunc, {iterAlloca}, {});
        if (derefResult.value) {
          if (gen.traits.is_reference(loopVar->type())) {
            mlir::cxx::StoreOp::create(gen.builder_, loc, derefResult.value,
                                       local.value(),
                                       gen.getAlignment(loopVar->type()));
          } else {
            auto elemType =
                gen.convertType(gen.traits.remove_cvref(loopVar->type()));
            auto elem = mlir::cxx::LoadOp::create(
                gen.builder_, loc, elemType, derefResult.value,
                gen.getAlignment(gen.traits.remove_cvref(loopVar->type())));
            mlir::cxx::StoreOp::create(gen.builder_, loc, elem, local.value(),
                                       gen.getAlignment(loopVar->type()));
          }
        }
      }
    }
  }

  emitRangeElementBindings(ast);

  gen.statement(ast->statement);
  auto bodyEndLoc = ast->rparenLoc;
  if (ast->statement) bodyEndLoc = ast->statement->lastSourceLocation();
  gen.emitBranchWithCleanups(bodyEndLoc, stepBlock, iterationDepth);
  gen.popCleanup(bodyEndLoc);

  gen.builder_.setInsertionPointToEnd(stepBlock);

  if (isPointerIterator) {
    auto iterInStep =
        mlir::cxx::LoadOp::create(gen.builder_, loc, iterType, iterAlloca, 8);
    auto intTy = gen.builder_.getIntegerType(32);
    auto oneOp = mlir::arith::ConstantOp::create(
        gen.builder_, loc, intTy, gen.builder_.getIntegerAttr(intTy, 1));
    auto nextIter = mlir::cxx::PtrAddOp::create(gen.builder_, loc, iterType,
                                                iterInStep, oneOp);
    mlir::cxx::StoreOp::create(gen.builder_, loc, nextIter, iterAlloca, 8);
  } else if (incrFunc) {
    (void)gen.emitCall(ast->colonLoc, incrFunc, {iterAlloca}, {});
  }

  gen.branch(loc, condBlock);

  gen.builder_.setInsertionPointToEnd(exitBlock);

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

  gen.branch(gen.getLocation(ast->firstSourceLocation()), beginLoopBlock);
  gen.builder_.setInsertionPointToEnd(beginLoopBlock);

  if (ast->condition) {
    gen.conditionWithCleanups(ast->condition, loopBodyBlock,
                              conditionFalseBlock);
  } else {
    gen.branch(gen.getLocation(ast->semicolonLoc), loopBodyBlock);
  }

  gen.builder_.setInsertionPointToEnd(loopBodyBlock);
  gen.statement(ast->statement);

  gen.branch(gen.getLocation(ast->statement->lastSourceLocation()),
             stepLoopBlock);

  gen.builder_.setInsertionPointToEnd(stepLoopBlock);

  {
    auto fullExpression = FullExpression{
        gen, ast->expression ? ast->expression->lastSourceLocation()
                             : ast->rparenLoc};
    (void)gen.expression(ast->expression, ExpressionFormat::kSideEffect);
  }

  gen.emitBranchWithCleanups(
      ast->expression ? ast->expression->lastSourceLocation() : ast->rparenLoc,
      beginLoopBlock, iterationDepth);

  gen.builder_.setInsertionPointToEnd(conditionFalseBlock);
  gen.emitBranchWithCleanups(ast->lastSourceLocation(), endLoopBlock,
                             iterationDepth);

  gen.popCleanup(ast->lastSourceLocation());
  gen.builder_.setInsertionPointToEnd(endLoopBlock);
  gen.popCleanup(ast->lastSourceLocation());

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

  (void)gen.emitPrvalueInto(gen.exitValue_.getResult(), gen.returnType_,
                            ast->expression, ast->firstSourceLocation());
}

void Codegen::StatementVisitor::operator()(ReturnStatementAST* ast) {
  {
    auto fullExpression = FullExpression{gen, ast->lastSourceLocation()};
    storeReturnValue(ast);
  }

  gen.emitBranchWithCleanups(ast->firstSourceLocation(), gen.exitBlock_, 0);
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
    auto loc = gen.getLocation(ast->firstSourceLocation());
    auto ptrResult = gen.expression(ast->expression);
    auto ptr = ptrResult.value;
    if (ast->expression &&
        ast->expression->valueCategory == ValueCategory::kLValue) {
      auto loadedType = gen.convertType(ast->expression->type);
      ptr = mlir::cxx::LoadOp::create(gen.builder_, loc, loadedType, ptr,
                                      gen.getAlignment(ast->expression->type));
    }
    mlir::cxx::IndirectGotoOp::create(gen.builder_, loc, ptr,
                                      mlir::BlockRange{});
    auto nextBlock = gen.newBlock();
    gen.builder_.setInsertionPointToEnd(nextBlock);
    return;
  }

  auto cleanupSnapshot = gen.collectCleanupSnapshot();

  mlir::cxx::GotoOp::create(
      gen.builder_, gen.getLocation(ast->firstSourceLocation()),
      cleanupSnapshot.addresses, cleanupSnapshot.activeFlags,
      mlir::ArrayAttr::get(gen.context_, cleanupSnapshot.destructors),
      mlir::ArrayAttr::get(gen.context_, cleanupSnapshot.depths),
      gen.builder_.getDenseI32ArrayAttr(cleanupSnapshot.activeFlagIndices),
      ast->identifier->name());

  auto nextBlock = gen.newBlock();
  gen.branch(gen.getLocation(ast->firstSourceLocation()), nextBlock);

  gen.builder_.setInsertionPointToEnd(nextBlock);
}

void Codegen::StatementVisitor::operator()(DeclarationStatementAST* ast) {
  auto fullExpression = FullExpression{gen, ast->lastSourceLocation()};
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
