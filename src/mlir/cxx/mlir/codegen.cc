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
#include <cxx/ast_interpreter.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

// mlir
#include <llvm/BinaryFormat/Dwarf.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>

#include <filesystem>
#include <format>

namespace cxx {

Codegen::Codegen(mlir::MLIRContext& context, TranslationUnit* unit)
    : builder_(&context), unit_(unit) {}

Codegen::~Codegen() {}

auto Codegen::control() const -> Control* { return unit_->control(); }

auto Codegen::getAlignment(const Type* type) -> uint64_t {
  return control()->memoryLayout()->alignmentOf(type).value_or(1);
}

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

auto Codegen::getFloatAttr(const std::optional<ConstValue>& value,
                           const Type* type) -> std::optional<mlir::FloatAttr> {
  if (value.has_value()) {
    auto ty = control()->remove_cvref(type);

    auto interp = ASTInterpreter{unit_};

    switch (ty->kind()) {
      case TypeKind::kFloat:
        return interp.toFloat(*value).transform(
            [&](float value) { return builder_.getF32FloatAttr(value); });

      case TypeKind::kDouble:
        return interp.toDouble(*value).transform(
            [&](double value) { return builder_.getF64FloatAttr(value); });

      case TypeKind::kLongDouble:
        return interp.toDouble(*value).transform(
            [&](double value) { return builder_.getF64FloatAttr(value); });

      default:
        break;
    }  // switch
  }

  return {};
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
  auto allocaOp = mlir::cxx::AllocaOp::create(builder_, loc, ptrType,
                                              getAlignment(var->type()));

  attachDebugInfo(allocaOp, var);

  locals_.emplace(var, allocaOp);

  return allocaOp;
}

void Codegen::attachDebugInfo(mlir::cxx::AllocaOp allocaOp, Symbol* symbol,
                              std::string_view name, unsigned arg) {
  if (!function_) return;

  auto funcLoc = mlir::dyn_cast<mlir::FusedLoc>(function_.getLoc());
  if (!funcLoc) return;

  auto metadata = funcLoc.getMetadata();
  auto subprogram =
      mlir::dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>(metadata);
  if (!subprogram) return;

  auto ctx = builder_.getContext();
  auto nameAttr = mlir::StringAttr::get(
      ctx, name.empty() ? to_string(symbol->name()) : name);
  auto file =
      getFileAttr(unit_->tokenStartPosition(symbol->location()).fileName);
  unsigned line = unit_->tokenStartPosition(symbol->location()).line;
  auto typeAttr = convertDebugType(symbol->type());

  auto localVar = mlir::LLVM::DILocalVariableAttr::get(
      ctx, subprogram, nameAttr, file, line, arg, 0, typeAttr,
      mlir::LLVM::DIFlags::Zero);

  allocaOp->setAttr("cxx.di_local", localVar);
}

auto Codegen::newTemp(const Type* type, SourceLocation loc)
    -> mlir::cxx::AllocaOp {
  auto ptrType = builder_.getType<mlir::cxx::PointerType>(convertType(type));
  return mlir::cxx::AllocaOp::create(builder_, getLocation(loc), ptrType,
                                     getAlignment(type));
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

auto Codegen::findOrCreateGlobal(Symbol* symbol)
    -> std::optional<mlir::cxx::GlobalOp> {
  auto variableSymbol = symbol_cast<VariableSymbol>(symbol);
  if (!variableSymbol) return {};

  if (auto it = globalOps_.find(variableSymbol); it != globalOps_.end()) {
    return it->second;
  }

  if (!variableSymbol->isStatic() && !variableSymbol->parent()->isNamespace()) {
    return {};
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

  if (variableSymbol->isStatic() ||
      !is_global_namespace(variableSymbol->parent())) {
    ExternalNameEncoder encoder;
    name = encoder.encode(symbol);
  } else {
    name = to_string(symbol->name());
  }

  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(varType);

  mlir::Attribute initializer;

  auto value = variableSymbol->constValue();

  if (value.has_value()) {
    auto interp = ASTInterpreter{unit_};

    if (control()->is_integral_or_unscoped_enum(variableSymbol->type())) {
      auto constValue = interp.toInt(*value);
      initializer = builder_.getI64IntegerAttr(constValue.value_or(0));
    } else if (auto attr = getFloatAttr(value, variableSymbol->type())) {
      initializer = attr.value();
    } else if (control()->is_array(variableSymbol->type())) {
      if (auto constArrayPtr =
              std::get_if<std::shared_ptr<InitializerList>>(&*value)) {
        auto constArray = *constArrayPtr;
        std::vector<mlir::Attribute> elements;

        // todo: fill elements
        for (const auto& element : constArray->elements) {
          // convert each element to mlir::Attribute and push to elements
        }
        initializer = builder_.getArrayAttr(elements);
      }
    } else if (control()->is_class(variableSymbol->type())) {
      if (auto constArrayPtr =
              std::get_if<std::shared_ptr<InitializerList>>(&*value)) {
        auto constArray = *constArrayPtr;
        std::vector<mlir::Attribute> elements;

        // todo: fill elements
        for (const auto& element : constArray->elements) {
          // convert each element to mlir::Attribute and push to elements
        }

        initializer = builder_.getArrayAttr(elements);
      }
    }
  }

  if (!variableSymbol->initializer() && !variableSymbol->isExtern()) {
    // default initialize to zero
    initializer = builder_.getZeroAttr(varType);
  }

  auto var = mlir::cxx::GlobalOp::create(builder_, loc, varType, false, name,
                                         initializer);

  globalOps_.insert_or_assign(variableSymbol, var);

  return var;
}

auto Codegen::getCompileUnitAttr(std::string_view filename)
    -> mlir::LLVM::DICompileUnitAttr {
  if (auto it = compileUnitAttrs_.find(filename);
      it != compileUnitAttrs_.end()) {
    return it->second;
  }

  auto ctx = builder_.getContext();

  auto distinct = mlir::DistinctAttr::create(builder_.getUnitAttr());

  auto sourceLanguage = unit_->language() == LanguageKind::kCXX
                            ? llvm::dwarf::DW_LANG_C_plus_plus_20
                            : llvm::dwarf::DW_LANG_C;

  auto fileAttr = getFileAttr(filename);
  auto producer = mlir::StringAttr::get(ctx, "cxx");
  auto isOptimized = false;
  auto emissionKind = mlir::LLVM::DIEmissionKind::Full;

  mlir::LLVM::DINameTableKind nameTableKind =
      mlir::LLVM::DINameTableKind::Default;

  // for apple triple
  nameTableKind = mlir::LLVM::DINameTableKind::Apple;

  auto compileUnit = mlir::LLVM::DICompileUnitAttr::get(
      distinct, sourceLanguage, fileAttr, producer, isOptimized, emissionKind,
      nameTableKind);

  compileUnitAttrs_.insert_or_assign(filename, compileUnit);

  return compileUnit;
}

auto Codegen::getFileAttr(const std::string& filename)
    -> mlir::LLVM::DIFileAttr {
  if (auto it = fileAttrs_.find(filename); it != fileAttrs_.end()) {
    return it->second;
  }

  auto filePath = absolute(std::filesystem::path{filename});

  auto attr = mlir::LLVM::DIFileAttr::get(builder_.getContext(),
                                          filePath.filename().string(),
                                          filePath.parent_path().string());

  fileAttrs_.insert_or_assign(filename, attr);

  return attr;
}

auto Codegen::getFileAttr(std::string_view filename) -> mlir::LLVM::DIFileAttr {
  return getFileAttr(std::string{filename});
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
