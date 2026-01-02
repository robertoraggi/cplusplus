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
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

// mlir
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <format>

namespace cxx {

Codegen::Codegen(mlir::MLIRContext& context, TranslationUnit* unit)
    : builder_(&context), unit_(unit) {}

Codegen::~Codegen() {}

auto Codegen::control() const -> Control* { return unit_->control(); }

auto Codegen::currentBlockMightHaveTerminator() -> bool {
  auto block = builder_.getInsertionBlock();
  if (!block) {
    cxx_runtime_error("current block is null");
  }
  return block->mightHaveTerminator();
}

auto Codegen::newBlock() -> mlir::Block* {
  auto region = builder_.getBlock()->getParent();
  auto newBlock = new mlir::Block();
  region->getBlocks().push_back(newBlock);
  return newBlock;
}

auto Codegen::newUniqueSymbolName(std::string_view prefix) -> std::string {
  auto& uniqueName = uniqueSymbolNames_[prefix];
  if (uniqueName == 0) {
    uniqueName = 1;
    return std::format("{}{}", prefix, uniqueName);
  }
  return std::format("{}{}", prefix, ++uniqueName);
}

void Codegen::branch(mlir::Location loc, mlir::Block* block,
                     mlir::ValueRange operands) {
  if (currentBlockMightHaveTerminator()) return;
  mlir::cf::BranchOp::create(builder_, loc, block, operands);
}

auto Codegen::findOrCreateLocal(Symbol* symbol) -> std::optional<mlir::Value> {
  if (auto local = locals_.find(symbol); local != locals_.end()) {
    return local->second;
  }

  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var) return std::nullopt;

  if (var->isStatic()) return std::nullopt;
  if (!var->parent()->isBlock()) return std::nullopt;

  auto type = convertType(var->type());
  auto ptrType = builder_.getType<mlir::cxx::PointerType>(type);

  auto loc = getLocation(var->location());
  auto allocaOp = mlir::cxx::AllocaOp::create(builder_, loc, ptrType);

  locals_.emplace(var, allocaOp);

  return allocaOp;
}

auto Codegen::newTemp(const Type* type, SourceLocation loc)
    -> mlir::cxx::AllocaOp {
  auto ptrType = builder_.getType<mlir::cxx::PointerType>(convertType(type));
  return mlir::cxx::AllocaOp::create(builder_, getLocation(loc), ptrType);
}

auto Codegen::findOrCreateFunction(FunctionSymbol* functionSymbol)
    -> mlir::cxx::FuncOp {
  if (auto it = funcOps_.find(functionSymbol); it != funcOps_.end()) {
    return it->second;
  }

  const auto functionType = type_cast<FunctionType>(functionSymbol->type());
  const auto returnType = functionType->returnType();
  const auto needsExitValue = !control()->is_void(returnType);

  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;

  if (!functionSymbol->isStatic() && functionSymbol->parent()->isClass()) {
    // if it is a non static member function, we need to add the `this` pointer

    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());

    inputTypes.push_back(builder_.getType<mlir::cxx::PointerType>(
        convertType(classSymbol->type())));
  }

  for (auto paramTy : functionType->parameterTypes()) {
    inputTypes.push_back(convertType(paramTy));
  }

  if (needsExitValue) {
    resultTypes.push_back(convertType(returnType));
  }

  auto funcType =
      mlir::cxx::FunctionType::get(builder_.getContext(), inputTypes,
                                   resultTypes, functionType->isVariadic());

  std::string name;

  if (functionSymbol->hasCLinkage()) {
    name = to_string(functionSymbol->name());
  } else {
    ExternalNameEncoder encoder;
    name = encoder.encode(functionSymbol);
  }

  const auto loc = getLocation(functionSymbol->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::InlineKind inlineKind = mlir::cxx::InlineKind::NoInline;

  if (functionSymbol->isInline()) {
    inlineKind = mlir::cxx::InlineKind::InlineHint;
  }

  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(builder_.getContext(), inlineKind);

  mlir::cxx::LinkageKind linkageKind = mlir::cxx::LinkageKind::External;

  if (functionSymbol->isStatic()) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  }

  auto linkageAttr =
      mlir::cxx::LinkageKindAttr::get(builder_.getContext(), linkageKind);

  auto func = mlir::cxx::FuncOp::create(builder_, loc, name, funcType,
                                        linkageAttr, inlineAttr,
                                        mlir::ArrayAttr{}, mlir::ArrayAttr{});

  funcOps_.insert_or_assign(functionSymbol, func);

  return func;
}

auto Codegen::findOrCreateGlobal(VariableSymbol* variableSymbol)
    -> mlir::cxx::GlobalOp {
  if (auto it = globalOps_.find(variableSymbol); it != globalOps_.end()) {
    return it->second;
  }

  auto varType = convertType(variableSymbol->type());

  const auto loc = getLocation(variableSymbol->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::InlineKind inlineKind = mlir::cxx::InlineKind::NoInline;

  mlir::cxx::LinkageKind linkageKind = mlir::cxx::LinkageKind::External;

  if (variableSymbol->isStatic()) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  }

  auto linkageAttr =
      mlir::cxx::LinkageKindAttr::get(builder_.getContext(), linkageKind);

  std::string name;

  name = to_string(variableSymbol->name());

  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(varType);

  mlir::Attribute initializer;

  auto value = variableSymbol->constValue();

  if (value.has_value()) {
    auto interp = ASTInterpreter{unit_};

    if (control()->is_integral_or_unscoped_enum(variableSymbol->type())) {
      auto constValue = interp.toInt(*value);
      initializer = builder_.getI64IntegerAttr(constValue.value_or(0));
    } else if (control()->is_floating_point(variableSymbol->type())) {
      auto ty = control()->remove_cv(variableSymbol->type());
      if (type_cast<FloatType>(ty)) {
        auto constValue = interp.toFloat(*value);
        initializer = builder_.getF32FloatAttr(constValue.value_or(0));
      } else if (type_cast<DoubleType>(ty)) {
        auto constValue = interp.toDouble(*value);
        initializer = builder_.getF64FloatAttr(constValue.value_or(0));
      }
    }
  }

  if (!initializer) {
    // default initialize to zero
    initializer = builder_.getZeroAttr(varType);
  }

  auto var = mlir::cxx::GlobalOp::create(builder_, loc, varType, false, name,
                                         initializer);

  globalOps_.insert_or_assign(variableSymbol, var);

  return var;
}

auto Codegen::getLocation(SourceLocation location) -> mlir::Location {
  auto [filename, line, column] = unit_->tokenStartPosition(location);

  auto loc =
      mlir::FileLineColLoc::get(builder_.getContext(), filename, line, column);

  return loc;
}

auto Codegen::emitTodoStmt(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoStmtOp {
  const auto loc = getLocation(location);
  auto op = mlir::cxx::TodoStmtOp::create(builder_, loc, message);
  return op;
}

auto Codegen::emitTodoExpr(SourceLocation location, std::string_view message)
    -> mlir::cxx::TodoExprOp {
  const auto loc = getLocation(location);
  auto op = mlir::cxx::TodoExprOp::create(builder_, loc, message);
  return op;
}

}  // namespace cxx
