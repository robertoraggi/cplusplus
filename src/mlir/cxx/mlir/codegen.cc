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
#include <cxx/type_traits.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

// mlir
#include <cxx/decl.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>

#include <filesystem>
#include <format>

namespace cxx {

static auto isInAnonymousNamespace(Symbol* symbol) -> bool {
  for (auto* scope = symbol->parent(); scope; scope = scope->parent()) {
    if (auto* ns = symbol_cast<NamespaceSymbol>(scope)) {
      if (ns->anonNamespaceIndex().has_value()) return true;
    }
  }
  return false;
}

static auto isMemberOfClassTemplateSpecialization(Symbol* symbol) -> bool {
  for (auto* scope = symbol->parent(); scope; scope = scope->parent()) {
    if (auto* cls = symbol_cast<ClassSymbol>(scope)) {
      if (cls->isSpecialization()) return true;
    }
  }
  return false;
}

static auto targetNeedsAppleNameTable(mlir::ModuleOp module) -> bool {
  auto tripleAttr = module->getAttrOfType<mlir::StringAttr>("cxx.triple");
  if (!tripleAttr) return false;
  llvm::Triple triple(tripleAttr.getValue());
  return triple.isAppleMachO();
}

Codegen::Codegen(mlir::MLIRContext& context, TranslationUnit* unit,
                 bool debugInfo)
    : context_(&context),
      builder_(&context),
      unit_(unit),
      debugInfo_(debugInfo) {}

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
    auto ty = unit_->typeTraits().remove_cvref(type);

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

auto Codegen::constValueToAttr(const ConstValue& value, const Type* type)
    -> std::optional<mlir::Attribute> {
  auto interp = ASTInterpreter{unit_};

  if (unit_->typeTraits().is_integral_or_unscoped_enum(type)) {
    auto constValue = interp.toInt(value);
    return builder_.getI64IntegerAttr(constValue.value_or(0));
  }

  if (auto attr = getFloatAttr(value, type)) {
    return *attr;
  }

  if (unit_->typeTraits().is_pointer(type)) {
    if (std::get_if<std::shared_ptr<ConstLabelAddress>>(&value))
      return std::nullopt;
    if (auto intVal = std::get_if<std::intmax_t>(&value)) {
      if (*intVal == 0) return builder_.getUnitAttr();
    }
    if (auto strLit = std::get_if<const StringLiteral*>(&value)) {
      (*strLit)->initialize((*strLit)->encoding());
      std::string str((*strLit)->stringValue());
      str.push_back('\0');
      return builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));
    }
    return std::nullopt;
  }

  if (unit_->typeTraits().is_array(type) ||
      unit_->typeTraits().is_class(type)) {
    if (auto constArrayPtr =
            std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      auto constArray = *constArrayPtr;
      std::vector<mlir::Attribute> elements;
      for (const auto& [elemValue, elemType] : constArray->elements) {
        if (auto attr = constValueToAttr(elemValue, elemType)) {
          elements.push_back(*attr);
        } else {
          return std::nullopt;
        }
      }
      return builder_.getArrayAttr(elements);
    }
  }

  return std::nullopt;
}

auto Codegen::emitConstInitValue(mlir::OpBuilder& builder, mlir::Location loc,
                                 const Type* type, const ConstValue& value)
    -> mlir::Value {
  auto interp = ASTInterpreter{unit_};

  if (unit_->typeTraits().is_integral_or_unscoped_enum(type)) {
    auto mlirType = convertType(type);
    auto constValue = interp.toInt(value);
    return mlir::arith::ConstantOp::create(
        builder, loc, mlirType,
        builder.getIntegerAttr(mlirType, constValue.value_or(0)));
  }

  if (unit_->typeTraits().is_floating_point(type)) {
    auto mlirType = convertType(type);
    auto floatType = mlir::cast<mlir::FloatType>(mlirType);
    auto constValue = interp.toDouble(value);
    return mlir::arith::ConstantOp::create(
        builder, loc, floatType,
        mlir::FloatAttr::get(floatType, constValue.value_or(0.0)));
  }

  if (unit_->typeTraits().is_pointer(type)) {
    auto ptrType = convertType(type);
    auto mlirPtrType = mlir::cast<mlir::cxx::PointerType>(ptrType);

    // Check if the value is a ConstAddress, address-of variable or function.
    if (auto addrPtr = std::get_if<std::shared_ptr<ConstAddress>>(&value)) {
      auto* symbol = (*addrPtr)->symbol();
      auto offset = (*addrPtr)->offset();
      if (symbol_cast<VariableSymbol>(symbol)) {
        if (auto glo = findOrCreateGlobal(symbol)) {
          mlir::Value result = mlir::cxx::AddressOfOp::create(
              builder, loc, mlirPtrType, glo->getSymName());
          if (offset != 0) {
            auto offsetVal = mlir::arith::ConstantOp::create(
                builder, loc, builder.getI64Type(),
                builder.getI64IntegerAttr(offset));
            result = mlir::cxx::PtrAddOp::create(builder, loc, mlirPtrType,
                                                 result, offsetVal);
          }
          return result;
        }
      } else if (auto funcSym = symbol_cast<FunctionSymbol>(symbol)) {
        auto funcOp = findOrCreateFunction(funcSym);
        return mlir::cxx::AddressOfOp::create(builder, loc, mlirPtrType,
                                              funcOp.getSymName());
      }
    }

    // Handle label address (&&label) pointer constant.
    if (auto labelAddrPtr =
            std::get_if<std::shared_ptr<ConstLabelAddress>>(&value)) {
      auto funcNameAttr =
          function_ ? mlir::StringAttr::get(context_, function_.getSymName())
                    : mlir::StringAttr{};
      return mlir::cxx::LabelAddressOp::create(builder, loc, mlirPtrType,
                                               (*labelAddrPtr)->name(),
                                               mlir::IntegerAttr{}, funcNameAttr);
    }

    // Handle string literal used as a pointer value.
    if (auto strLitPtr = std::get_if<const StringLiteral*>(&value)) {
      auto stringLiteral = *strLitPtr;
      stringLiteral->initialize(stringLiteral->encoding());
      std::string str(stringLiteral->stringValue());
      str.push_back('\0');

      auto i8Type = mlir::IntegerType::get(context_, 8);
      auto arrayType = mlir::cxx::ArrayType::get(context_, i8Type, str.size());
      auto strAttr =
          builder.getStringAttr(llvm::StringRef(str.data(), str.size()));
      auto strName = builder.getStringAttr(newUniqueSymbolName(".str"));

      {
        auto guard = mlir::OpBuilder::InsertionGuard(builder);
        builder.setInsertionPointToStart(module_.getBody());
        auto linkage = mlir::cxx::LinkageKindAttr::get(
            context_, mlir::cxx::LinkageKind::Internal);
        mlir::cxx::GlobalOp::create(builder, loc, mlir::TypeRange(), arrayType,
                                    true, strName.getValue(), strAttr, linkage);
      }

      return mlir::cxx::AddressOfOp::create(builder, loc, mlirPtrType, strName);
    }

    return mlir::cxx::NullPtrConstantOp::create(builder, loc, mlirPtrType);
  }

  if (unit_->typeTraits().is_class_or_union(type)) {
    auto classType = type_cast<ClassType>(unit_->typeTraits().remove_cv(type));
    auto mlirType = convertType(type);

    if (classType && classType->isUnion()) {
      // Union initialization: check if the first member value is zero
      if (auto initListPtr =
              std::get_if<std::shared_ptr<InitializerList>>(&value)) {
        auto& initList = *initListPtr;
        if (!initList->elements.empty()) {
          auto& [elemValue, elemType] = initList->elements[0];

          // Check if the value is zero
          bool isZero = false;
          if (auto intVal = std::get_if<std::intmax_t>(&elemValue)) {
            isZero = (*intVal == 0);
          } else if (auto floatVal = std::get_if<float>(&elemValue)) {
            isZero = (*floatVal == 0.0f);
          } else if (auto doubleVal = std::get_if<double>(&elemValue)) {
            isZero = (*doubleVal == 0.0);
          }

          if (isZero) {
            return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
          }

          auto elemVal = emitConstInitValue(builder, loc, elemType, elemValue);

          auto unionClassType = mlir::dyn_cast<mlir::cxx::ClassType>(mlirType);
          if (unionClassType && !unionClassType.getBody().empty() &&
              elemVal.getType() == unionClassType.getBody()[0]) {
            auto undef = mlir::cxx::UndefOp::create(builder, loc, mlirType);
            return mlir::cxx::InsertValueOp::create(
                builder, loc, mlirType, undef, elemVal,
                static_cast<int64_t>(0));
          }

          return mlir::cxx::BitcastOp::create(builder, loc, mlirType, elemVal);
        }
      }
      return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
    }

    // Struct: undef + insertvalue per field
    if (auto initListPtr =
            std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      auto& initList = *initListPtr;
      mlir::Value result = mlir::cxx::UndefOp::create(builder, loc, mlirType);
      auto fieldTypes =
          mlir::dyn_cast<mlir::cxx::ClassType>(mlirType).getBody();
      for (size_t i = 0; i < initList->elements.size(); ++i) {
        auto& [elemValue, elemType] = initList->elements[i];
        auto elemVal = emitConstInitValue(builder, loc, elemType, elemValue);
        if (i < fieldTypes.size() && elemVal.getType() != fieldTypes[i]) {
          auto srcArr = mlir::dyn_cast<mlir::cxx::ArrayType>(elemVal.getType());
          auto dstArr = mlir::dyn_cast<mlir::cxx::ArrayType>(fieldTypes[i]);
          if (srcArr && dstArr &&
              srcArr.getElementType() == dstArr.getElementType() &&
              srcArr.getSize() < dstArr.getSize()) {
            elemVal =
                mlir::cxx::ReshapeOp::create(builder, loc, dstArr, elemVal);
          }
        }
        result = mlir::cxx::InsertValueOp::create(
            builder, loc, mlirType, result, elemVal, static_cast<int64_t>(i));
      }
      return result;
    }

    return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
  }

  if (unit_->typeTraits().is_array(type)) {
    auto mlirType = convertType(type);
    auto cxxArrType = mlir::dyn_cast<mlir::cxx::ArrayType>(mlirType);
    if (auto initListPtr =
            std::get_if<std::shared_ptr<InitializerList>>(&value)) {
      auto& initList = *initListPtr;
      mlir::Value result = mlir::cxx::UndefOp::create(builder, loc, mlirType);
      for (size_t i = 0; i < initList->elements.size(); ++i) {
        auto& [elemValue, elemType] = initList->elements[i];
        auto elemVal = emitConstInitValue(builder, loc, elemType, elemValue);
        if (cxxArrType) {
          auto dstElemType = cxxArrType.getElementType();
          if (elemVal.getType() != dstElemType) {
            auto srcArr =
                mlir::dyn_cast<mlir::cxx::ArrayType>(elemVal.getType());
            auto dstArr = mlir::dyn_cast<mlir::cxx::ArrayType>(dstElemType);
            if (srcArr && dstArr &&
                srcArr.getElementType() == dstArr.getElementType() &&
                srcArr.getSize() < dstArr.getSize()) {
              elemVal =
                  mlir::cxx::ReshapeOp::create(builder, loc, dstArr, elemVal);
            }
          }
        }
        result = mlir::cxx::InsertValueOp::create(
            builder, loc, mlirType, result, elemVal, static_cast<int64_t>(i));
      }
      return result;
    }
    return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
  }

  // Fallback: zero
  auto mlirType = convertType(type);
  return mlir::cxx::ZeroOp::create(builder, loc, mlirType);
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

  auto loc = getLocation(var->location());

  if (auto* vlaType = type_cast<UnresolvedBoundedArrayType>(var->type())) {
    auto countResult = expression(vlaType->size());
    if (!countResult.value) return std::nullopt;

    auto countVal = countResult.value;
    if (mlir::isa<mlir::cxx::PointerType>(countVal.getType())) {
      auto valueType = convertType(vlaType->size()->type);
      countVal = mlir::cxx::LoadOp::create(builder_, loc, valueType, countVal,
                                           getAlignment(vlaType->size()->type));
    }

    mlir::Value totalElements = countVal;
    const Type* elemType = vlaType->elementType();

    while (auto* inner = type_cast<UnresolvedBoundedArrayType>(elemType)) {
      auto innerResult = expression(inner->size());
      if (!innerResult.value) return std::nullopt;
      auto innerVal = innerResult.value;
      if (mlir::isa<mlir::cxx::PointerType>(innerVal.getType())) {
        auto valueType = convertType(inner->size()->type);
        innerVal = mlir::cxx::LoadOp::create(builder_, loc, valueType, innerVal,
                                             getAlignment(inner->size()->type));
      }
      if (innerVal.getType() != totalElements.getType())
        innerVal = mlir::arith::ExtSIOp::create(
            builder_, loc, totalElements.getType(), innerVal);
      totalElements = mlir::arith::MulIOp::create(
          builder_, loc, totalElements.getType(), totalElements, innerVal);
      elemType = inner->elementType();
    }

    auto elementType = convertType(elemType);
    auto ptrType = mlir::cxx::PointerType::get(context_, elementType);
    auto alignment = getAlignment(elemType);

    auto leafSizeBytes = static_cast<int64_t>(
        control()->memoryLayout()->sizeOf(elemType).value_or(1));
    mlir::Value totalBytes = totalElements;
    if (leafSizeBytes > 1) {
      auto sizeConst = mlir::arith::ConstantOp::create(
          builder_, loc, totalElements.getType(),
          builder_.getIntegerAttr(totalElements.getType(), leafSizeBytes));
      totalBytes = mlir::arith::MulIOp::create(
          builder_, loc, totalElements.getType(), totalElements, sizeConst);
    }

    auto allocaOp = mlir::cxx::DynAllocaOp::create(builder_, loc, ptrType,
                                                   totalBytes, alignment);
    locals_.emplace(var, allocaOp);
    return allocaOp;
  }

  auto type = convertType(var->type());
  auto ptrType = mlir::cxx::PointerType::get(context_, type);

  auto allocaOp = mlir::cxx::AllocaOp::create(builder_, loc, ptrType,
                                              getAlignment(var->type()));

  attachDebugInfo(allocaOp, var);

  locals_.emplace(var, allocaOp);

  return allocaOp;
}

auto Codegen::getOrCreateDIScope(Symbol* symbol) -> mlir::LLVM::DIScopeAttr {
  if (!symbol) return {};

  if (auto it = diScopes_.find(symbol); it != diScopes_.end())
    return it->second;

  if (symbol_cast<FunctionParametersSymbol>(symbol))
    return getOrCreateDIScope(symbol->parent());

  if (auto* block = symbol_cast<BlockSymbol>(symbol)) {
    if (symbol_cast<FunctionParametersSymbol>(block->parent()) ||
        symbol_cast<FunctionSymbol>(block->parent()))
      return getOrCreateDIScope(block->parent());

    auto parentScope = getOrCreateDIScope(block->parent());
    if (!parentScope) return {};
    auto [filename, line, column] =
        unit_->tokenStartPosition(block->location());
    auto fileAttr = getFileAttr(filename);
    auto lexicalBlock = mlir::LLVM::DILexicalBlockAttr::get(
        context_, parentScope, fileAttr, line, column);
    diScopes_[symbol] = lexicalBlock;
    return lexicalBlock;
  }

  if (auto* func = symbol_cast<FunctionSymbol>(symbol)) {
    if (auto it = funcOps_.find(func); it != funcOps_.end()) {
      if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(it->second.getLoc())) {
        if (auto sp = mlir::dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>(
                fusedLoc.getMetadata())) {
          diScopes_[symbol] = sp;
          return sp;
        }
      }
    }
  }

  auto fileAttr =
      getFileAttr(unit_->tokenStartPosition(symbol->location()).fileName);
  return fileAttr;
}

void Codegen::attachDebugInfo(mlir::cxx::AllocaOp allocaOp, Symbol* symbol,
                              std::string_view name, unsigned arg) {
  if (!debugInfo_) return;
  if (!function_) return;

  auto scope = getOrCreateDIScope(symbol->parent());
  if (!scope) return;

  auto ctx = context_;
  auto nameAttr = mlir::StringAttr::get(
      ctx, name.empty() ? to_string(symbol->name()) : name);
  auto file =
      getFileAttr(unit_->tokenStartPosition(symbol->location()).fileName);
  unsigned line = unit_->tokenStartPosition(symbol->location()).line;
  auto typeAttr = convertDebugType(symbol->type());

  auto localVar = mlir::LLVM::DILocalVariableAttr::get(
      ctx, scope, nameAttr, file, line, arg, 0, typeAttr,
      mlir::LLVM::DIFlags::Zero);

  allocaOp->setAttr("cxx.di_local", localVar);
}

void Codegen::attachDebugInfo(mlir::cxx::AllocaOp allocaOp, const Type* type,
                              std::string_view name, unsigned arg,
                              mlir::LLVM::DIFlags flags) {
  if (!debugInfo_) return;
  if (!function_) return;

  auto scope = getOrCreateDIScope(currentFunctionSymbol_);
  if (!scope) return;

  auto ctx = context_;
  auto nameAttr = mlir::StringAttr::get(ctx, name);
  auto typeAttr = convertDebugType(type);

  mlir::LLVM::DIFileAttr file;
  unsigned line = 0;
  if (auto sp = mlir::dyn_cast<mlir::LLVM::DISubprogramAttr>(scope)) {
    file = sp.getFile();
    line = sp.getLine();
  }

  auto localVar = mlir::LLVM::DILocalVariableAttr::get(
      ctx, scope, nameAttr, file, line, arg, 0, typeAttr, flags);

  allocaOp->setAttr("cxx.di_local", localVar);
}

auto Codegen::buildSubroutineTypeAttr(FunctionSymbol* functionSymbol)
    -> mlir::LLVM::DISubroutineTypeAttr {
  auto functionType = type_cast<FunctionType>(functionSymbol->type());

  mlir::SmallVector<mlir::LLVM::DITypeAttr> signatureType;
  signatureType.push_back(convertDebugType(functionType->returnType()));

  if (auto classType = type_cast<ClassType>(functionSymbol->parent()->type());
      classType && !functionSymbol->isStatic()) {
    signatureType.push_back(
        convertDebugType(unit_->typeTraits().add_pointer(classType)));
  }

  for (auto paramType : functionType->parameterTypes()) {
    signatureType.push_back(convertDebugType(paramType));
  }

  return mlir::LLVM::DISubroutineTypeAttr::get(context_, signatureType);
}

void Codegen::buildSubprogramAttr(FunctionSymbol* functionSymbol,
                                  FunctionDefinitionAST* ast,
                                  mlir::cxx::FuncOp func, mlir::Location loc) {
  auto ctx = context_;

  mlir::DistinctAttr id = mlir::DistinctAttr::create(builder_.getUnitAttr());

  mlir::LLVM::DIScopeAttr scope;

  if (!functionSymbol->isStatic() && functionSymbol->parent()->isClass()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    if (classSymbol) {
      scope = mlir::dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(
          convertDebugType(classSymbol->type()));
    }
  }

  mlir::StringAttr name = mlir::StringAttr::get(ctx, func.getSymName());
  mlir::StringAttr linkageName = name;

  auto declaratorId = getDeclaratorId(ast->declarator);

  mlir::LLVM::DIFileAttr fileAttr;
  unsigned line = 0;
  unsigned scopeLine = 0;
  std::string_view fileName;

  if (declaratorId && declaratorId->firstSourceLocation()) {
    auto funcLoc =
        unit_->tokenStartPosition(declaratorId->firstSourceLocation());
    fileAttr = getFileAttr(funcLoc.fileName);
    line = funcLoc.line;
    fileName = funcLoc.fileName;
  }

  if (ast->functionBody) {
    auto bodyLoc = ast->functionBody->firstSourceLocation();
    if (bodyLoc) {
      scopeLine = unit_->tokenStartPosition(bodyLoc).line;
    }
  }

  if (!fileAttr) {
    auto classLoc = functionSymbol->location();
    if (classLoc) {
      auto pos = unit_->tokenStartPosition(classLoc);
      fileAttr = getFileAttr(pos.fileName);
      line = pos.line;
      scopeLine = pos.line;
      fileName = pos.fileName;
    } else {
      fileAttr = getFileAttr(std::string_view{""});
      fileName = "";
    }
  }

  if (!scope) scope = fileAttr;

  auto subprogramFlags = mlir::LLVM::DISubprogramFlags::Definition;

  auto type = buildSubroutineTypeAttr(functionSymbol);

  mlir::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;
  mlir::SmallVector<mlir::LLVM::DINodeAttr> annotations;

  auto compileUnitAttr = getCompileUnitAttr(fileName);

  auto subprogram = mlir::LLVM::DISubprogramAttr::get(
      ctx, id, compileUnitAttr, scope, name, linkageName,
      compileUnitAttr.getFile(), line, scopeLine, subprogramFlags, type,
      retainedNodes, annotations);

  func->setLoc(mlir::FusedLoc::get({loc}, subprogram, ctx));

  diScopes_[functionSymbol] = subprogram;
}

auto Codegen::newTemp(const Type* type, SourceLocation loc)
    -> mlir::cxx::AllocaOp {
  auto ptrType = mlir::cxx::PointerType::get(context_, convertType(type));
  return mlir::cxx::AllocaOp::create(builder_, getLocation(loc), ptrType,
                                     getAlignment(type));
}

void Codegen::pushCleanup() { cleanupStack_.emplace_back(); }

void Codegen::popCleanup(SourceLocation loc) {
  auto& scope = cleanupStack_.back();
  if (scope.entries.empty() || currentBlockMightHaveTerminator()) {
    cleanupStack_.pop_back();
    return;
  }
  auto mergeBlock = newBlock();
  emitBranchWithCleanups(loc, mergeBlock, cleanupStack_.size() - 1);
  builder_.setInsertionPointToEnd(mergeBlock);
  cleanupStack_.pop_back();
}

void Codegen::emitBranchWithCleanups(SourceLocation loc, mlir::Block* target,
                                     std::size_t targetDepth) {
  if (currentBlockMightHaveTerminator()) return;

  auto mlirLoc = getLocation(loc);

  llvm::SmallVector<mlir::Value> addresses;
  llvm::SmallVector<mlir::Attribute> destructors;

  for (auto i = cleanupStack_.size(); i > targetDepth; --i) {
    auto& scope = cleanupStack_[i - 1];
    for (auto jt = scope.entries.rbegin(); jt != scope.entries.rend(); ++jt) {
      addresses.push_back(jt->address);
      auto funcOp = findOrCreateFunction(jt->destructor);
      destructors.push_back(
          mlir::FlatSymbolRefAttr::get(funcOp.getSymNameAttr()));
    }
  }

  if (addresses.empty()) {
    mlir::cf::BranchOp::create(builder_, mlirLoc, target);
    return;
  }

  auto destructorsAttr = mlir::ArrayAttr::get(context_, destructors);
  mlir::cxx::CleanupBranchOp::create(builder_, mlirLoc, addresses,
                                     destructorsAttr, target);
}

void Codegen::addCleanup(mlir::Value address, FunctionSymbol* dtor) {
  if (cleanupStack_.empty()) return;
  cleanupStack_.back().entries.push_back({address, dtor});
}

auto Codegen::findOrCreateFunction(FunctionSymbol* functionSymbol)
    -> mlir::cxx::FuncOp {
  auto canonicalSymbol = functionSymbol->canonical();
  auto emittedSymbol = functionSymbol;

  if (!functionSymbol->isSpecialization()) {
    emittedSymbol = canonicalSymbol;
  }

  if (auto it = funcOps_.find(emittedSymbol); it != funcOps_.end()) {
    return it->second;
  }

  const auto functionType = type_cast<FunctionType>(emittedSymbol->type());
  if (!functionType) {
    return {};
  }

  const auto returnType = functionType->returnType();
  const auto needsExitValue = !unit_->typeTraits().is_void(returnType);

  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;

  if (!emittedSymbol->isStatic() && emittedSymbol->parent()->isClass()) {
    auto classSymbol = symbol_cast<ClassSymbol>(emittedSymbol->parent());

    inputTypes.push_back(mlir::cxx::PointerType::get(
        context_, convertType(classSymbol->type())));
  }

  for (auto paramTy : functionType->parameterTypes()) {
    inputTypes.push_back(convertType(paramTy));
  }

  if (needsExitValue) {
    resultTypes.push_back(convertType(returnType));
  }

  auto funcType = mlir::cxx::FunctionType::get(
      context_, inputTypes, resultTypes, functionType->isVariadic());

  std::string name;

  if (emittedSymbol->hasCLinkage()) {
    name = to_string(emittedSymbol->name());
  } else {
    ExternalNameEncoder encoder;
    name = encoder.encode(emittedSymbol);
  }

  const auto loc = getLocation(functionSymbol->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::InlineKind inlineKind = mlir::cxx::InlineKind::NoInline;

  if (emittedSymbol->isInline()) {
    inlineKind = mlir::cxx::InlineKind::InlineHint;
  }

  auto inlineAttr = mlir::cxx::InlineKindAttr::get(context_, inlineKind);

  mlir::cxx::LinkageKind linkageKind = mlir::cxx::LinkageKind::External;

  if (emittedSymbol->isStatic() && !emittedSymbol->parent()->isClass()) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (isInAnonymousNamespace(emittedSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (emittedSymbol->isInline()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (emittedSymbol->isSpecialization()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (isMemberOfClassTemplateSpecialization(emittedSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (emittedSymbol->isDefaulted()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  }

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkageKind);

  auto func = mlir::cxx::FuncOp::create(builder_, loc, name, funcType,
                                        linkageAttr, inlineAttr,
                                        mlir::ArrayAttr{}, mlir::ArrayAttr{});

  funcOps_.insert_or_assign(emittedSymbol, func);

  enqueueFunctionBody(emittedSymbol);

  return func;
}

void Codegen::enqueueFunctionBody(FunctionSymbol* symbol) {
  auto target = symbol->canonical();
  if (auto def = target->definition()) target = def;
  if (!target->declaration()) return;
  if (!enqueuedFunctions_.insert(target).second) return;
  pendingFunctions_.push_back(target);
}

void Codegen::processPendingFunctions() {
  while (!pendingFunctions_.empty()) {
    auto* sym = pendingFunctions_.back();
    pendingFunctions_.pop_back();

    auto target = sym;
    if (auto def = sym->definition()) target = def;

    if (auto funcDecl = target->declaration()) {
      declaration(funcDecl);
    }

    if (sym->parent() && sym->parent()->isClass()) {
      auto classSymbol = symbol_cast<ClassSymbol>(sym->parent());
      if (classSymbol) {
        generateVTable(classSymbol);
      }
    }
  }
}

auto Codegen::findOrCreateGlobal(Symbol* symbol)
    -> std::optional<mlir::cxx::GlobalOp> {
  auto variableSymbol = symbol_cast<VariableSymbol>(symbol);
  if (!variableSymbol) return {};

  auto canonicalVar = variableSymbol->canonical();

  if (auto it = globalOps_.find(canonicalVar); it != globalOps_.end()) {
    return it->second;
  }

  if (!variableSymbol->isStatic() && !variableSymbol->parent()->isNamespace()) {
    return {};
  }

  auto defVar =
      canonicalVar->definition() ? canonicalVar->definition() : canonicalVar;

  auto varType = convertType(defVar->type());

  const auto loc = getLocation(variableSymbol->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::InlineKind inlineKind = mlir::cxx::InlineKind::NoInline;

  mlir::cxx::LinkageKind linkageKind = mlir::cxx::LinkageKind::External;

  if (variableSymbol->isStatic() && !variableSymbol->parent()->isClass()) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (isInAnonymousNamespace(variableSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (variableSymbol->isInline()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (variableSymbol->isSpecialization()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (isMemberOfClassTemplateSpecialization(variableSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (variableSymbol->isStatic() &&
             variableSymbol->parent()->isClass()) {
    linkageKind = mlir::cxx::LinkageKind::External;
  }

  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkageKind);

  std::string name;

  if (variableSymbol->isStatic() ||
      !is_global_namespace(variableSymbol->parent())) {
    std::string suffix;
    if (variableSymbol->isStatic()) {
      if (auto function = symbol->enclosingFunction()) {
        auto& count = staticLocalCounts_[symbol->name()];
        if (count > 0) {
          suffix = std::format("_{}", count - 1);
        }
        ++count;
      }
    }

    ExternalNameEncoder encoder;
    name = encoder.encode(symbol, suffix);
  } else {
    name = to_string(symbol->name());
  }

  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(varType);

  mlir::Attribute initializer;
  bool needsRegionInit = false;

  auto value = defVar->constValue();

  if (value.has_value()) {
    auto interp = ASTInterpreter{unit_};

    if (unit_->typeTraits().is_integral_or_unscoped_enum(defVar->type())) {
      auto constValue = interp.toInt(*value);
      initializer = builder_.getI64IntegerAttr(constValue.value_or(0));
    } else if (auto attr = getFloatAttr(value, defVar->type())) {
      initializer = attr.value();
    } else if (unit_->typeTraits().is_array(defVar->type())) {
      if (auto constArrayPtr =
              std::get_if<std::shared_ptr<InitializerList>>(&*value)) {
        auto constArray = *constArrayPtr;
        std::vector<mlir::Attribute> elements;
        bool allConverted = true;

        for (const auto& [elemValue, elemType] : constArray->elements) {
          if (auto attr = constValueToAttr(elemValue, elemType)) {
            elements.push_back(*attr);
          } else {
            allConverted = false;
            break;
          }
        }

        if (allConverted) {
          initializer = builder_.getArrayAttr(elements);
        } else {
          needsRegionInit = true;
        }
      } else if (auto constStringPtr =
                     std::get_if<const StringLiteral*>(&*value)) {
        auto stringLiteral = *constStringPtr;
        stringLiteral->initialize(stringLiteral->encoding());
        std::string str(stringLiteral->stringValue());

        // Append null terminator.
        switch (stringLiteral->encoding()) {
          case StringLiteralEncoding::kUtf16:
            str.push_back('\0');
            str.push_back('\0');
            break;
          case StringLiteralEncoding::kUtf32:
          case StringLiteralEncoding::kWide:
            str.push_back('\0');
            str.push_back('\0');
            str.push_back('\0');
            str.push_back('\0');
            break;
          default:
            str.push_back('\0');
            break;
        }

        initializer =
            builder_.getStringAttr(llvm::StringRef(str.data(), str.size()));
      }
    } else if (unit_->typeTraits().is_class(defVar->type())) {
      needsRegionInit = true;
    } else if (unit_->typeTraits().is_pointer(defVar->type())) {
      if (auto attr = constValueToAttr(*value, defVar->type())) {
        initializer = *attr;
      } else {
        needsRegionInit = true;
      }
    }
  }

  auto isExternalOnly = variableSymbol->isExtern();
  if (isExternalOnly) {
    if (auto canon = variableSymbol->canonical()) {
      if (canon->definition() || !canon->isExtern()) isExternalOnly = false;
    }
  }

  if (!initializer && !isExternalOnly) {
    initializer = mlir::LLVM::ZeroAttr::get(context_);
  }

  bool isConstant = variableSymbol->isConstexpr() ||
                    unit_->typeTraits().is_const(defVar->type());

  auto var = mlir::cxx::GlobalOp::create(
      builder_, loc, mlir::TypeRange(), varType, isConstant,
      llvm::StringRef(name), initializer, linkageAttr);

  globalOps_.insert_or_assign(canonicalVar, var);

  if (needsRegionInit && value.has_value()) {
    auto& region = var.getInitializer();
    auto* block = new mlir::Block();
    region.push_back(block);
    mlir::OpBuilder initBuilder(block, block->begin());
    auto result = emitConstInitValue(initBuilder, loc, defVar->type(), *value);
    mlir::cxx::ReturnOp::create(initBuilder, loc, result);
  }

  return var;
}

auto Codegen::getCompileUnitAttr(std::string_view filename)
    -> mlir::LLVM::DICompileUnitAttr {
  if (auto it = compileUnitAttrs_.find(filename);
      it != compileUnitAttrs_.end()) {
    return it->second;
  }

  auto ctx = context_;

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

  if (targetNeedsAppleNameTable(module_)) {
    nameTableKind = mlir::LLVM::DINameTableKind::Apple;
  }

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

  auto attr = mlir::LLVM::DIFileAttr::get(
      context_, filePath.filename().string(), filePath.parent_path().string());

  fileAttrs_.insert_or_assign(filename, attr);

  return attr;
}

auto Codegen::getFileAttr(std::string_view filename) -> mlir::LLVM::DIFileAttr {
  return getFileAttr(std::string{filename});
}

auto Codegen::getLocation(SourceLocation location) -> mlir::Location {
  auto [filename, line, column] = unit_->tokenStartPosition(location);

  auto loc = mlir::FileLineColLoc::get(context_, filename, line, column);

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

auto Codegen::computeVtableSlots(ClassSymbol* classSymbol)
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> vtableSlots;

  auto baseClasses = classSymbol->baseClasses();
  if (!baseClasses.empty()) {
    auto baseClassSym = symbol_cast<ClassSymbol>(baseClasses[0]->symbol());
    if (baseClassSym && baseClassSym->layout() &&
        baseClassSym->layout()->hasVtable()) {
      vtableSlots = computeVtableSlots(baseClassSym);
    }
  }

  for (auto member : views::members(classSymbol)) {
    auto processFunc = [&](FunctionSymbol* func) {
      if (!func->isVirtual()) return;
      bool foundOverride = false;
      for (size_t i = 0; i < vtableSlots.size(); ++i) {
        auto isOverride = vtableSlots[i]->name() == func->name() ||
                          (name_cast<DestructorId>(vtableSlots[i]->name()) &&
                           name_cast<DestructorId>(func->name()));
        if (isOverride) {
          vtableSlots[i] = func;
          foundOverride = true;
          break;
        }
      }
      if (!foundOverride) {
        vtableSlots.push_back(func);
      }
    };

    if (auto func = symbol_cast<FunctionSymbol>(member)) {
      processFunc(func);
    } else if (auto ovl = symbol_cast<OverloadSetSymbol>(member)) {
      for (auto* f : ovl->functions()) processFunc(f);
    }
  }

  return vtableSlots;
}

void Codegen::emitCtorVtableInit(FunctionSymbol* functionSymbol,
                                 mlir::Location loc) {
  if (!functionSymbol->isConstructor() || !thisValue_) return;

  auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
  if (!classSymbol) return;

  auto layout = classSymbol->layout();
  if (!layout || !layout->hasVtable()) return;

  ExternalNameEncoder encoder;
  auto vtableName = encoder.encodeVTable(classSymbol);

  auto vtableSlots = computeVtableSlots(classSymbol);
  size_t vtableSize = 2 + vtableSlots.size();

  auto i8Type = builder_.getI8Type();
  auto i8PtrType = mlir::cxx::PointerType::get(context_, i8Type);
  auto vtableArrayType =
      mlir::cxx::ArrayType::get(context_, i8PtrType, vtableSize);
  auto vtablePtrType = mlir::cxx::PointerType::get(context_, vtableArrayType);

  auto vtableAddr = mlir::cxx::AddressOfOp::create(
      builder_, loc, vtablePtrType,
      mlir::FlatSymbolRefAttr::get(context_, vtableName));

  auto intTy = convertType(control()->getIntType());
  auto twoOp = mlir::arith::ConstantOp::create(
      builder_, loc, intTy, builder_.getIntegerAttr(intTy, 2));

  auto vtableDataPtr =
      mlir::cxx::PtrAddOp::create(builder_, loc, i8PtrType, vtableAddr, twoOp);

  auto thisType = convertType(classSymbol->type());
  auto ptrType = mlir::cxx::PointerType::get(context_, thisType);

  auto thisPtr = mlir::cxx::LoadOp::create(
      builder_, loc, ptrType, thisValue_,
      getAlignment(unit_->typeTraits().add_pointer(classSymbol->type())));

  mlir::Value vptrFieldPtr;
  if (layout->hasDirectVtable()) {
    vptrFieldPtr = mlir::cxx::MemberOp::create(
        builder_, loc, mlir::cxx::PointerType::get(context_, i8PtrType),
        thisPtr, layout->vtableIndex());
  } else {
    mlir::Value current = thisPtr;
    auto* currentClass = classSymbol;
    auto* currentLayout = layout;

    while (currentLayout && !currentLayout->hasDirectVtable()) {
      auto baseIdx = currentLayout->vtableIndex();
      ClassSymbol* baseSym = nullptr;
      for (auto base : currentClass->baseClasses()) {
        auto bs = symbol_cast<ClassSymbol>(base->symbol());
        if (!bs) continue;
        auto bi = currentLayout->getBaseInfo(bs);
        if (bi && bi->index == baseIdx) {
          baseSym = bs;
          break;
        }
      }
      if (!baseSym) break;

      auto basePtrType =
          mlir::cxx::PointerType::get(context_, convertType(baseSym->type()));
      current = mlir::cxx::MemberOp::create(builder_, loc, basePtrType, current,
                                            baseIdx);
      currentClass = baseSym;
      currentLayout = baseSym->layout();
    }

    auto vtableIdx = currentLayout ? currentLayout->vtableIndex() : 0;
    vptrFieldPtr = mlir::cxx::MemberOp::create(
        builder_, loc, mlir::cxx::PointerType::get(context_, i8PtrType),
        current, vtableIdx);
  }

  mlir::cxx::StoreOp::create(builder_, loc, vtableDataPtr, vptrFieldPtr, 8);
}

void Codegen::generateVTable(ClassSymbol* classSymbol) {
  auto layout = classSymbol->layout();
  if (!layout || !layout->hasVtable()) {
    return;
  }

  if (!emittedVTables_.insert(classSymbol).second) return;

  auto vtableSlots = computeVtableSlots(classSymbol);

  ExternalNameEncoder encoder;
  auto vtableName = encoder.encodeVTable(classSymbol);

  auto loc = getLocation(classSymbol->location());

  mlir::SmallVector<mlir::Attribute> vtableEntries;

  auto i64Type = builder_.getIntegerType(64);
  auto nullEntry = builder_.getIntegerAttr(i64Type, 0);
  vtableEntries.push_back(nullEntry);

  vtableEntries.push_back(nullEntry);

  for (auto func : vtableSlots) {
    auto funcOp = findOrCreateFunction(func);
    auto funcSymRef =
        mlir::FlatSymbolRefAttr::get(context_, funcOp.getSymName());
    vtableEntries.push_back(funcSymRef);
  }

  auto entriesAttr = builder_.getArrayAttr(vtableEntries);

  auto linkage = mlir::cxx::LinkageKind::LinkOnceODR;
  auto linkageAttr = mlir::cxx::LinkageKindAttr::get(context_, linkage);

  auto savedInsertionPoint = builder_.saveInsertionPoint();

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::VTableOp::create(builder_, loc, mlir::TypeRange(),
                              llvm::StringRef(vtableName), entriesAttr,
                              linkageAttr);

  builder_.restoreInsertionPoint(savedInsertionPoint);
}

}  // namespace cxx
