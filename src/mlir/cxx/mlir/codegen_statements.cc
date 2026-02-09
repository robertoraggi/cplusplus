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

#include <cxx/mlir/codegen.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

// mlir
#include <llvm/ADT/TypeSwitch.h>
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

  if (!gen.switch_.defaultDestination) {
    gen.switch_.defaultDestination = endSwitchBlock;
  }

  auto elementTy =
      mlir::TypeSwitch<mlir::Type, mlir::IntegerType>(
          conditionResult.value.getType())
          .Case<mlir::cxx::IntegerType>(
              [&](mlir::cxx::IntegerType ty) -> mlir::IntegerType {
                return gen.builder_.getIntegerType(ty.getWidth());
              })
          .Default([](mlir::Type ty) -> mlir::IntegerType { return {}; });

  auto shapeType = mlir::VectorType::get(
      static_cast<std::int64_t>(gen.switch_.caseValues.size()),
      gen.builder_.getIntegerType(64));

  auto caseValuesAttr = mlir::cast<mlir::DenseIntElementsAttr>(
      mlir::DenseIntElementsAttr::get(shapeType, gen.switch_.caseValues)
          .mapValues(elementTy, [&](mlir::APInt v) {
            return mlir::APInt(elementTy.getIntOrFloatBitWidth(),
                               v.getZExtValue(), false, true);
          }));

  auto flag = mlir::cxx::ToFlagOp::create(
      gen.builder_, gen.getLocation(ast->firstSourceLocation()), elementTy,
      conditionResult.value);

  std::vector<mlir::ValueRange> caseOperands(
      gen.switch_.caseDestinations.size(), mlir::ValueRange{});

  mlir::cf::SwitchOp::create(
      gen.builder_, gen.getLocation(ast->firstSourceLocation()),
      flag, gen.switch_.defaultDestination, {}, caseValuesAttr,
      gen.switch_.caseDestinations, caseOperands);

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
  auto loc = gen.getLocation(ast->firstSourceLocation());

  gen.statement(ast->initializer);

  auto rangeResult = gen.expression(ast->rangeInitializer);

  auto rangeType = ast->rangeInitializer->type;
  if (!rangeType) {
    (void)gen.emitTodoStmt(ast->firstSourceLocation(), "for-range: no type");
    return;
  }
  rangeType = control()->remove_cvref(rangeType);

  mlir::Value beginVal, endVal;
  bool isPointerIterator = false;
  const Type* iteratorElementType = nullptr;
  FunctionSymbol* derefFunc = nullptr;
  FunctionSymbol* incrFunc = nullptr;
  FunctionSymbol* neqFunc = nullptr;

  if (auto arrayType = type_cast<BoundedArrayType>(rangeType)) {
    isPointerIterator = true;
    iteratorElementType = arrayType->elementType();

    auto elementMlirType = gen.convertType(arrayType->elementType());
    auto ptrType =
        gen.builder_.getType<mlir::cxx::PointerType>(elementMlirType);

    beginVal = rangeResult.value;

    auto intTy =
        mlir::cxx::IntegerType::get(gen.builder_.getContext(), 64, true);
    auto sizeOp = mlir::cxx::IntConstantOp::create(gen.builder_, loc, intTy,
                                                   arrayType->size());
    endVal = mlir::cxx::PtrAddOp::create(gen.builder_, loc, ptrType, beginVal,
                                         sizeOp);
  } else if (auto classType = type_cast<ClassType>(rangeType)) {
    auto classSymbol = classType->symbol();
    if (classSymbol && classSymbol->definition())
      classSymbol = classSymbol->definition();

    if (!classSymbol) {
      (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                             "for-range: incomplete class");
      return;
    }

    auto beginName = control()->getIdentifier("begin");
    auto endName = control()->getIdentifier("end");

    // Look for member begin()/end()
    FunctionSymbol* beginFunc = nullptr;
    FunctionSymbol* endFunc = nullptr;

    for (auto sym : classSymbol->find(beginName)) {
      if (auto func = symbol_cast<FunctionSymbol>(sym)) {
        beginFunc = func;
        break;
      }
    }

    for (auto sym : classSymbol->find(endName)) {
      if (auto func = symbol_cast<FunctionSymbol>(sym)) {
        endFunc = func;
        break;
      }
    }

    if (!beginFunc || !endFunc) {
      ScopeSymbol* lookupScope = ast->symbol
                                     ? static_cast<ScopeSymbol*>(ast->symbol)
                                     : static_cast<ScopeSymbol*>(classSymbol);

      std::vector<const Type*> argTypes = {rangeType};

      auto beginCandidates =
          Lookup{lookupScope}.argumentDependentLookup(beginName, argTypes);
      auto endCandidates =
          Lookup{lookupScope}.argumentDependentLookup(endName, argTypes);

      if (!beginCandidates.empty()) beginFunc = beginCandidates.front();
      if (!endCandidates.empty()) endFunc = endCandidates.front();
    }

    if (!beginFunc || !endFunc) {
      (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                             "for-range: no begin/end");
      return;
    }

    bool isMember = beginFunc->parent() == classSymbol;

    if (isMember) {
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

    auto beginFuncType = type_cast<FunctionType>(beginFunc->type());
    if (!beginFuncType) {
      (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                             "for-range: bad begin type");
      return;
    }

    auto iterType = control()->remove_cvref(beginFuncType->returnType());

    if (control()->is_pointer(iterType)) {
      isPointerIterator = true;
      iteratorElementType = control()->get_element_type(iterType);
    } else if (auto iterClassType = type_cast<ClassType>(iterType)) {
      auto iterClass = iterClassType->symbol();
      if (iterClass && iterClass->definition())
        iterClass = iterClass->definition();

      if (iterClass) {
        auto starOp = control()->getOperatorId(TokenKind::T_STAR);
        auto plusPlusOp = control()->getOperatorId(TokenKind::T_PLUS_PLUS);
        auto neqOp = control()->getOperatorId(TokenKind::T_EXCLAIM_EQUAL);

        for (auto sym : iterClass->find(starOp)) {
          if (auto func = symbol_cast<FunctionSymbol>(sym)) {
            derefFunc = func;
            break;
          }
        }

        for (auto sym : iterClass->find(plusPlusOp)) {
          if (auto func = symbol_cast<FunctionSymbol>(sym)) {
            auto ft = type_cast<FunctionType>(func->type());
            if (ft && ft->parameterTypes().empty()) {
              incrFunc = func;
              break;
            }
          }
        }

        for (auto sym : iterClass->find(neqOp)) {
          if (auto func = symbol_cast<FunctionSymbol>(sym)) {
            neqFunc = func;
            break;
          }
        }

        if (!neqFunc) {
          ScopeSymbol* lookupScope =
              ast->symbol ? static_cast<ScopeSymbol*>(ast->symbol)
                          : static_cast<ScopeSymbol*>(iterClass);
          std::vector<const Type*> cmpArgTypes = {iterType, iterType};
          auto neqCandidates =
              Lookup{lookupScope}.argumentDependentLookup(neqOp, cmpArgTypes);
          if (!neqCandidates.empty()) neqFunc = neqCandidates.front();
        }
      }

      if (!derefFunc || !incrFunc || !neqFunc) {
        (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                               "for-range: missing iterator ops");
        return;
      }
    } else {
      // Unknown iterator type
      isPointerIterator = true;
      iteratorElementType = nullptr;
    }
  } else {
    (void)gen.emitTodoStmt(ast->firstSourceLocation(),
                           "for-range: unsupported range type");
    return;
  }

  // Create loop blocks
  auto condBlock = gen.newBlock();
  auto bodyBlock = gen.newBlock();
  auto stepBlock = gen.newBlock();
  auto exitBlock = gen.newBlock();

  // Alloca for the iterator
  auto iterType = beginVal.getType();
  auto iterPtrType = gen.builder_.getType<mlir::cxx::PointerType>(iterType);
  auto iterAlloca =
      mlir::cxx::AllocaOp::create(gen.builder_, loc, iterPtrType, 8);
  mlir::cxx::StoreOp::create(gen.builder_, loc, beginVal, iterAlloca, 8);

  // Alloca for end
  auto endPtrType =
      gen.builder_.getType<mlir::cxx::PointerType>(endVal.getType());
  auto endAlloca =
      mlir::cxx::AllocaOp::create(gen.builder_, loc, endPtrType, 8);
  mlir::cxx::StoreOp::create(gen.builder_, loc, endVal, endAlloca, 8);

  Loop loop{stepBlock, exitBlock};
  std::swap(gen.loop_, loop);

  gen.branch(loc, condBlock);

  gen.builder_.setInsertionPointToEnd(condBlock);

  auto iterLoad =
      mlir::cxx::LoadOp::create(gen.builder_, loc, iterType, iterAlloca, 8);
  auto endLoad = mlir::cxx::LoadOp::create(gen.builder_, loc, endVal.getType(),
                                           endAlloca, 8);

  mlir::Value condVal;
  if (isPointerIterator || !neqFunc) {
    auto boolType = gen.convertType(control()->getBoolType());
    condVal = mlir::cxx::NotEqualOp::create(gen.builder_, loc, boolType,
                                            iterLoad, endLoad);
  } else {
    auto neqParent = neqFunc->parent();
    bool isMemberNeq = neqParent && neqParent->kind() == SymbolKind::kClass;

    ExpressionResult neqResult;
    if (isMemberNeq) {
      neqResult = gen.emitCall(ast->colonLoc, neqFunc, {iterLoad}, {{endLoad}});
    } else {
      neqResult =
          gen.emitCall(ast->colonLoc, neqFunc, {}, {{iterLoad}, {endLoad}});
    }
    condVal = neqResult.value;
  }

  if (!condVal) {
    auto boolType = gen.convertType(control()->getBoolType());
    condVal = mlir::cxx::NotEqualOp::create(gen.builder_, loc, boolType,
                                            iterLoad, endLoad);
  }

  mlir::cf::CondBranchOp::create(gen.builder_, loc, condVal, bodyBlock, {},
                                 exitBlock, {});

  gen.builder_.setInsertionPointToEnd(bodyBlock);

  // Get the loop variable from the block scope
  VariableSymbol* loopVar = nullptr;
  if (ast->symbol) {
    for (auto member : ast->symbol->members()) {
      if (auto var = symbol_cast<VariableSymbol>(member)) {
        loopVar = var;
        break;
      }
    }
  }

  if (loopVar) {
    auto local = gen.findOrCreateLocal(loopVar);
    if (local) {
      auto iterInBody =
          mlir::cxx::LoadOp::create(gen.builder_, loc, iterType, iterAlloca, 8);

      if (isPointerIterator) {
        if (control()->is_reference(loopVar->type())) {
          mlir::cxx::StoreOp::create(gen.builder_, loc, iterInBody,
                                     local.value(),
                                     gen.getAlignment(loopVar->type()));
        } else {
          auto elemType =
              gen.convertType(control()->remove_cvref(loopVar->type()));
          auto elem = mlir::cxx::LoadOp::create(
              gen.builder_, loc, elemType, iterInBody,
              gen.getAlignment(control()->remove_cvref(loopVar->type())));
          mlir::cxx::StoreOp::create(gen.builder_, loc, elem, local.value(),
                                     gen.getAlignment(loopVar->type()));
        }
      } else if (derefFunc) {
        auto derefResult =
            gen.emitCall(ast->colonLoc, derefFunc, {iterInBody}, {});
        if (derefResult.value) {
          mlir::cxx::StoreOp::create(gen.builder_, loc, derefResult.value,
                                     local.value(),
                                     gen.getAlignment(loopVar->type()));
        }
      }
    }
  }

  gen.statement(ast->statement);
  gen.branch(
      gen.getLocation(ast->statement ? ast->statement->lastSourceLocation()
                                     : ast->rparenLoc),
      stepBlock);

  gen.builder_.setInsertionPointToEnd(stepBlock);

  auto iterInStep =
      mlir::cxx::LoadOp::create(gen.builder_, loc, iterType, iterAlloca, 8);

  if (isPointerIterator) {
    auto intTy =
        mlir::cxx::IntegerType::get(gen.builder_.getContext(), 32, true);
    auto oneOp = mlir::cxx::IntConstantOp::create(gen.builder_, loc, intTy, 1);
    auto nextIter = mlir::cxx::PtrAddOp::create(gen.builder_, loc, iterType,
                                                iterInStep, oneOp);
    mlir::cxx::StoreOp::create(gen.builder_, loc, nextIter, iterAlloca, 8);
  } else if (incrFunc) {
    auto incrResult = gen.emitCall(ast->colonLoc, incrFunc, {iterInStep}, {});
    if (incrResult.value) {
      mlir::cxx::StoreOp::create(gen.builder_, loc, incrResult.value,
                                 iterAlloca, 8);
    }
  }

  gen.branch(loc, condBlock);

  gen.builder_.setInsertionPointToEnd(exitBlock);

  std::swap(gen.loop_, loop);
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
                               gen.exitValue_.getResult(),
                               gen.getAlignment(gen.returnType_));
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
