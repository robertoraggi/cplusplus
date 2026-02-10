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
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

// mlir
#include <llvm/BinaryFormat/Dwarf.h>
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

Codegen::Codegen(mlir::MLIRContext& context, TranslationUnit* unit,
                 bool debugInfo)
    : builder_(&context), unit_(unit), debugInfo_(debugInfo) {}

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

auto Codegen::getOrCreateDIScope(Symbol* symbol) -> mlir::LLVM::DIScopeAttr {
  if (!symbol) return {};

  if (auto it = diScopes_.find(symbol); it != diScopes_.end())
    return it->second;

  if (symbol_cast<FunctionParametersSymbol>(symbol))
    return getOrCreateDIScope(symbol->parent());

  if (auto* block = symbol_cast<BlockSymbol>(symbol)) {
    auto parentScope = getOrCreateDIScope(block->parent());
    if (!parentScope) return {};
    auto [filename, line, column] =
        unit_->tokenStartPosition(block->location());
    auto fileAttr = getFileAttr(filename);
    auto lexicalBlock = mlir::LLVM::DILexicalBlockAttr::get(
        builder_.getContext(), parentScope, fileAttr, line, column);
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

  auto ctx = builder_.getContext();
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

  auto ctx = builder_.getContext();
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

auto Codegen::newTemp(const Type* type, SourceLocation loc)
    -> mlir::cxx::AllocaOp {
  auto ptrType = builder_.getType<mlir::cxx::PointerType>(convertType(type));
  return mlir::cxx::AllocaOp::create(builder_, getLocation(loc), ptrType,
                                     getAlignment(type));
}

auto Codegen::findOrCreateFunction(FunctionSymbol* functionSymbol)
    -> mlir::cxx::FuncOp {
  auto canonicalSymbol = functionSymbol->canonical();

  if (auto it = funcOps_.find(canonicalSymbol); it != funcOps_.end()) {
    return it->second;
  }

  const auto functionType = type_cast<FunctionType>(canonicalSymbol->type());
  const auto returnType = functionType->returnType();
  const auto needsExitValue = !control()->is_void(returnType);

  std::vector<mlir::Type> inputTypes;
  std::vector<mlir::Type> resultTypes;

  if (!canonicalSymbol->isStatic() && canonicalSymbol->parent()->isClass()) {
    auto classSymbol = symbol_cast<ClassSymbol>(canonicalSymbol->parent());

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

  if (canonicalSymbol->hasCLinkage()) {
    name = to_string(canonicalSymbol->name());
  } else {
    ExternalNameEncoder encoder;
    name = encoder.encode(canonicalSymbol);
  }

  const auto loc = getLocation(functionSymbol->location());

  auto guard = mlir::OpBuilder::InsertionGuard(builder_);

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::InlineKind inlineKind = mlir::cxx::InlineKind::NoInline;

  if (canonicalSymbol->isInline()) {
    inlineKind = mlir::cxx::InlineKind::InlineHint;
  }

  auto inlineAttr =
      mlir::cxx::InlineKindAttr::get(builder_.getContext(), inlineKind);

  mlir::cxx::LinkageKind linkageKind = mlir::cxx::LinkageKind::External;

  if (canonicalSymbol->isStatic() && !canonicalSymbol->parent()->isClass()) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (isInAnonymousNamespace(canonicalSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::Internal;
  } else if (canonicalSymbol->isInline()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (canonicalSymbol->isSpecialization()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (isMemberOfClassTemplateSpecialization(canonicalSymbol)) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  } else if (canonicalSymbol->isDefaulted()) {
    linkageKind = mlir::cxx::LinkageKind::LinkOnceODR;
  }

  auto linkageAttr =
      mlir::cxx::LinkageKindAttr::get(builder_.getContext(), linkageKind);

  auto func = mlir::cxx::FuncOp::create(builder_, loc, name, funcType,
                                        linkageAttr, inlineAttr,
                                        mlir::ArrayAttr{}, mlir::ArrayAttr{});

  funcOps_.insert_or_assign(canonicalSymbol, func);

  enqueueFunctionBody(canonicalSymbol);

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

  auto varType = convertType(variableSymbol->type());

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

  auto linkageAttr =
      mlir::cxx::LinkageKindAttr::get(builder_.getContext(), linkageKind);

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

  auto isExternalOnly = variableSymbol->isExtern();
  if (isExternalOnly) {
    if (auto canon = variableSymbol->canonical()) {
      if (canon->definition()) isExternalOnly = false;
    }
  }

  if (!initializer && !isExternalOnly) {
    if (control()->is_integral_or_unscoped_enum(variableSymbol->type())) {
      initializer = builder_.getI64IntegerAttr(0);
    } else if (control()->is_floating_point(variableSymbol->type())) {
      initializer = builder_.getF64FloatAttr(0.0);
    } else if (control()->is_array(variableSymbol->type())) {
      auto arrayType = type_cast<BoundedArrayType>(variableSymbol->type());
      if (arrayType) {
        size_t numElements = arrayType->size();
        std::vector<mlir::Attribute> zeroElements;

        mlir::Attribute zeroElement;
        if (control()->is_integral_or_unscoped_enum(arrayType->elementType())) {
          zeroElement = builder_.getI64IntegerAttr(0);
        } else if (control()->is_floating_point(arrayType->elementType())) {
          zeroElement = builder_.getF64FloatAttr(0.0);
        } else {
          auto elementVarType = convertType(arrayType->elementType());
          zeroElement = builder_.getZeroAttr(elementVarType);
        }

        if (zeroElement) {
          for (size_t i = 0; i < numElements; ++i) {
            zeroElements.push_back(zeroElement);
          }
          initializer = builder_.getArrayAttr(zeroElements);
        }
      }
    } else if (control()->is_pointer(variableSymbol->type())) {
      initializer = builder_.getUnitAttr();
    } else {
      initializer = builder_.getZeroAttr(varType);
    }
  }

  bool isConstant = variableSymbol->isConstexpr() ||
                    control()->is_const(variableSymbol->type());

  auto var = mlir::cxx::GlobalOp::create(
      builder_, loc, mlir::TypeRange(), varType, isConstant,
      llvm::StringRef(name), initializer, linkageAttr);

  globalOps_.insert_or_assign(canonicalVar, var);

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
    if (auto func = symbol_cast<FunctionSymbol>(member)) {
      if (func->isVirtual()) {
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
      }
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
  auto i8PtrType = builder_.getType<mlir::cxx::PointerType>(i8Type);
  auto vtableArrayType =
      builder_.getType<mlir::cxx::ArrayType>(i8PtrType, vtableSize);
  auto vtablePtrType =
      builder_.getType<mlir::cxx::PointerType>(vtableArrayType);

  auto vtableAddr = mlir::cxx::AddressOfOp::create(
      builder_, loc, vtablePtrType,
      mlir::FlatSymbolRefAttr::get(builder_.getContext(), vtableName));

  auto intTy = convertType(control()->getIntType());
  auto twoOp = mlir::arith::ConstantOp::create(
      builder_, loc, intTy, builder_.getIntegerAttr(intTy, 2));

  auto vtableDataPtr =
      mlir::cxx::PtrAddOp::create(builder_, loc, i8PtrType, vtableAddr, twoOp);

  auto thisType = convertType(classSymbol->type());
  auto ptrType = builder_.getType<mlir::cxx::PointerType>(thisType);

  auto thisPtr = mlir::cxx::LoadOp::create(
      builder_, loc, ptrType, thisValue_,
      getAlignment(control()->add_pointer(classSymbol->type())));

  mlir::Value vptrFieldPtr;
  if (layout->hasDirectVtable()) {
    vptrFieldPtr = mlir::cxx::MemberOp::create(
        builder_, loc, builder_.getType<mlir::cxx::PointerType>(i8PtrType),
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

      auto basePtrType = builder_.getType<mlir::cxx::PointerType>(
          convertType(baseSym->type()));
      current = mlir::cxx::MemberOp::create(builder_, loc, basePtrType, current,
                                            baseIdx);
      currentClass = baseSym;
      currentLayout = baseSym->layout();
    }

    auto vtableIdx = currentLayout ? currentLayout->vtableIndex() : 0;
    vptrFieldPtr = mlir::cxx::MemberOp::create(
        builder_, loc, builder_.getType<mlir::cxx::PointerType>(i8PtrType),
        current, vtableIdx);
  }

  mlir::cxx::StoreOp::create(builder_, loc, vtableDataPtr, vptrFieldPtr, 8);
}

void Codegen::generateVTable(ClassSymbol* classSymbol) {
  auto layout = classSymbol->layout();
  if (!layout || !layout->hasVtable()) {
    return;
  }

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
    auto funcSymRef = mlir::FlatSymbolRefAttr::get(builder_.getContext(),
                                                   funcOp.getSymName());
    vtableEntries.push_back(funcSymRef);
  }

  auto entriesAttr = builder_.getArrayAttr(vtableEntries);

  auto linkage = mlir::cxx::LinkageKind::LinkOnceODR;
  auto linkageAttr =
      mlir::cxx::LinkageKindAttr::get(builder_.getContext(), linkage);

  auto savedInsertionPoint = builder_.saveInsertionPoint();

  builder_.setInsertionPointToStart(module_.getBody());

  mlir::cxx::VTableOp::create(builder_, loc, mlir::TypeRange(),
                              llvm::StringRef(vtableName), entriesAttr,
                              linkageAttr);

  builder_.restoreInsertionPoint(savedInsertionPoint);
}

}  // namespace cxx
