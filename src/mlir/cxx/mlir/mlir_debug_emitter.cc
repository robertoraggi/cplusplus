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

#include <cxx/control.h>
#include <cxx/mlir/mlir_debug_emitter.h>
#include <cxx/mlir/mlir_emitter.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/TargetParser/Triple.h>

#include <filesystem>
#include <format>

namespace cxx::ir {
static auto targetNeedsAppleNameTable(mlir::ModuleOp module) -> bool {
  auto tripleAttr = module->getAttrOfType<mlir::StringAttr>("cxx.triple");
  if (!tripleAttr) return false;
  llvm::Triple triple(tripleAttr.getValue());
  return triple.isAppleMachO();
}

static auto debugFilePath(const std::string& filename,
                          const std::string& compilationDirectory)
    -> std::pair<std::string, std::string> {
  const auto filePath = std::filesystem::path{filename};

  if (!filePath.is_absolute()) return {filename, compilationDirectory};

  const auto currentPath = std::filesystem::path{compilationDirectory};

  auto fileIt = filePath.begin();
  const auto fileEnd = filePath.end();
  auto dirIt = currentPath.begin();
  const auto dirEnd = currentPath.end();

  std::filesystem::path commonPrefix;
  for (; dirIt != dirEnd && fileIt != fileEnd && *dirIt == *fileIt;
       ++dirIt, ++fileIt) {
    commonPrefix /= *dirIt;
  }

  if (commonPrefix == commonPrefix.root_path()) return {filename, {}};

  std::filesystem::path relativePath;
  for (; fileIt != fileEnd; ++fileIt) relativePath /= *fileIt;

  return {relativePath.string(), commonPrefix.string()};
}

MlirDebugEmitter::MlirDebugEmitter(MlirEmitter& emitter, TranslationUnit* unit)
    : emitter_(emitter),
      context_(emitter.context()),
      builder_(emitter.builder()),
      unit_(unit),
      traits(unit) {}
auto MlirDebugEmitter::control() const -> Control* { return unit_->control(); }

auto MlirDebugEmitter::getOrCreateDIScope(Symbol* symbol)
    -> mlir::LLVM::DIScopeAttr {
  if (!symbol) return {};

  if (auto it = diScopes_.find(symbol); it != diScopes_.end())
    return it->second;

  if (symbol_cast<FunctionParametersSymbol>(symbol))
    return getOrCreateDIScope(symbol->parent());

  if (auto block = symbol_cast<BlockSymbol>(symbol)) {
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

  if (auto func = symbol_cast<FunctionSymbol>(symbol)) {
    if (auto it = funcOps_.find(func); it != funcOps_.end()) {
      if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(
              emitter_.function(it->second).getLoc())) {
        if (auto sp = mlir::dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>(
                fusedLoc.getMetadata())) {
          diScopes_[symbol] = sp;
          return sp;
        }
      }
    }
  }

  return getFileAttrAt(symbol->location());
}

static auto subprogramOf(mlir::LLVM::DIScopeAttr scope)
    -> mlir::LLVM::DISubprogramAttr {
  while (scope) {
    if (auto sp = mlir::dyn_cast<mlir::LLVM::DISubprogramAttr>(scope))
      return sp;
    auto block = mlir::dyn_cast<mlir::LLVM::DILexicalBlockAttr>(scope);
    if (!block) break;
    scope = block.getScope();
  }
  return {};
}

static auto enclosingSubprogram(mlir::Operation* op)
    -> mlir::LLVM::DISubprogramAttr {
  for (; op; op = op->getParentOp()) {
    auto fused = mlir::dyn_cast<mlir::FusedLoc>(op->getLoc());
    if (!fused) continue;
    if (auto sp = mlir::dyn_cast_or_null<mlir::LLVM::DISubprogramAttr>(
            fused.getMetadata()))
      return sp;
  }
  return {};
}

static auto scopeForOperation(mlir::Operation* op,
                              mlir::LLVM::DIScopeAttr declared)
    -> mlir::LLVM::DIScopeAttr {
  auto enclosing = enclosingSubprogram(op);
  if (enclosing && subprogramOf(declared) != enclosing) return enclosing;
  return declared;
}

void MlirDebugEmitter::localVariable(ir::ValueRef address, Symbol* symbol,
                                     std::string_view name, unsigned arg) {
  auto definingOp = emitter_.value(address).getDefiningOp();
  if (!definingOp) return;

  auto scope =
      scopeForOperation(definingOp, getOrCreateDIScope(symbol->parent()));
  if (!scope) return;

  auto ctx = context_;
  auto nameAttr = mlir::StringAttr::get(
      ctx, name.empty() ? to_string(symbol->name()) : name);
  auto file = getFileAttrAt(symbol->location());
  unsigned line = unit_->tokenStartPosition(symbol->location()).line;
  auto typeAttr = convertDebugType(symbol->type());
  if (!typeAttr)
    cxx_runtime_error(
        std::format("cannot describe the type '{}' of local variable '{}'",
                    to_string(symbol->type()), nameAttr.getValue().str()));

  auto localVar = mlir::LLVM::DILocalVariableAttr::get(
      ctx, scope, nameAttr, file, line, arg, 0, typeAttr,
      mlir::LLVM::DIFlags::Zero);

  definingOp->setAttr("cxx.di_local", localVar);
}

void MlirDebugEmitter::objectParameter(ir::ValueRef address, const Type* type,
                                       FunctionSymbol* currentFunctionSymbol_,
                                       std::string_view name, unsigned arg) {
  auto definingOp = emitter_.value(address).getDefiningOp();
  if (!definingOp) return;

  auto scope =
      scopeForOperation(definingOp, getOrCreateDIScope(currentFunctionSymbol_));
  if (!scope) return;

  auto ctx = context_;
  auto nameAttr = mlir::StringAttr::get(ctx, name);
  auto typeAttr = convertDebugType(type);
  if (!typeAttr)
    cxx_runtime_error(std::format(
        "cannot describe the type '{}' of the object parameter of '{}'",
        to_string(type), to_string(currentFunctionSymbol_->name())));

  mlir::LLVM::DIFileAttr file;
  unsigned line = 0;
  if (auto sp = mlir::dyn_cast<mlir::LLVM::DISubprogramAttr>(scope)) {
    file = sp.getFile();
    line = sp.getLine();
  }

  auto localVar = mlir::LLVM::DILocalVariableAttr::get(
      ctx, scope, nameAttr, file, line, arg, 0, typeAttr,
      mlir::LLVM::DIFlags::Artificial | mlir::LLVM::DIFlags::ObjectPointer);

  definingOp->setAttr("cxx.di_local", localVar);
}

auto MlirDebugEmitter::buildSubroutineTypeAttr(FunctionSymbol* functionSymbol)
    -> mlir::LLVM::DISubroutineTypeAttr {
  auto functionType = type_cast<FunctionType>(functionSymbol->type());

  mlir::SmallVector<mlir::LLVM::DITypeAttr> signatureType;
  signatureType.push_back(convertDebugType(functionType->returnType()));

  if (functionSymbol->isImplicitObjectMemberFunction()) {
    auto classType = type_cast<ClassType>(functionSymbol->parent()->type());
    signatureType.push_back(convertDebugType(traits.add_pointer(classType)));
  }

  for (auto paramType : functionType->parameterTypes()) {
    signatureType.push_back(convertDebugType(paramType));
  }

  return mlir::LLVM::DISubroutineTypeAttr::get(context_, signatureType);
}

void MlirDebugEmitter::defineFunction(FunctionSymbol* functionSymbol,
                                      ir::FunctionRef func, SourceLocation loc,
                                      SourceLocation declaratorLoc,
                                      SourceLocation bodyLoc) {
  auto ctx = context_;

  mlir::DistinctAttr id = mlir::DistinctAttr::create(builder_.getUnitAttr());

  mlir::LLVM::DIScopeAttr scope;

  if (functionSymbol->isImplicitObjectMemberFunction()) {
    auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
    scope = mlir::dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(
        convertDebugType(classSymbol->type()));
  }

  auto symbolName = emitter_.functionName(func);
  auto sourceName = to_string(functionSymbol->name());

  mlir::StringAttr name = mlir::StringAttr::get(
      ctx, sourceName.empty() ? std::string{symbolName} : sourceName);
  mlir::StringAttr linkageName;
  if (std::string_view{name.getValue()} != symbolName)
    linkageName = mlir::StringAttr::get(ctx, symbolName);

  funcOps_[functionSymbol] = func;

  auto compileUnitAttr = getCompileUnitAttr();

  mlir::LLVM::DIFileAttr fileAttr;
  unsigned line = 0;
  unsigned scopeLine = 0;

  if (declaratorLoc) {
    auto funcLoc = unit_->tokenStartPosition(declaratorLoc);
    fileAttr = getFileAttr(funcLoc.fileName);
    line = funcLoc.line;
  }

  {
    if (bodyLoc) {
      scopeLine = unit_->tokenStartPosition(bodyLoc).line;
    }
  }

  if (!fileAttr) {
    auto symbolLoc = functionSymbol->location();
    fileAttr = getFileAttrAt(symbolLoc);
    if (symbolLoc) {
      line = unit_->tokenStartPosition(symbolLoc).line;
      scopeLine = line;
    }
  }

  if (!scope) scope = fileAttr;

  auto subprogramFlags = mlir::LLVM::DISubprogramFlags::Definition;

  if (emitter_.functionLinkage(func) == Linkage::Internal)
    subprogramFlags =
        subprogramFlags | mlir::LLVM::DISubprogramFlags::LocalToUnit;

  auto type = buildSubroutineTypeAttr(functionSymbol);

#if LLVM_VERSION_MAJOR < 23
  mlir::SmallVector<mlir::LLVM::DINodeAttr> retainedNodes;
#else
  mlir::SmallVector<mlir::Attribute> retainedNodes;
#endif
  mlir::SmallVector<mlir::LLVM::DINodeAttr> annotations;

  auto subprogram = mlir::LLVM::DISubprogramAttr::get(
      ctx, id, compileUnitAttr, scope, name, linkageName, fileAttr, line,
      scopeLine, subprogramFlags, type, retainedNodes, annotations);

  emitter_.function(func)->setLoc(
      mlir::FusedLoc::get({emitter_.getLocation(loc)}, subprogram, ctx));

  diScopes_[functionSymbol] = subprogram;
}

auto MlirDebugEmitter::getCompileUnitAttr() -> mlir::LLVM::DICompileUnitAttr {
  if (compileUnitAttr_) return compileUnitAttr_;

  auto ctx = context_;

  auto distinct = mlir::DistinctAttr::create(builder_.getUnitAttr());

  auto sourceLanguage = unit_->language() == LanguageKind::kCXX
                            ? llvm::dwarf::DW_LANG_C_plus_plus_20
                            : llvm::dwarf::DW_LANG_C;

  auto fileAttr = getOrCreateFileAttr(unit_->fileName());
  auto producer = mlir::StringAttr::get(ctx, "cxx");
  auto isOptimized = false;
  auto emissionKind = mlir::LLVM::DIEmissionKind::Full;

  mlir::LLVM::DINameTableKind nameTableKind =
      mlir::LLVM::DINameTableKind::Default;

  if (targetNeedsAppleNameTable(emitter_.module())) {
    nameTableKind = mlir::LLVM::DINameTableKind::Apple;
  }

  auto compileUnit = mlir::LLVM::DICompileUnitAttr::get(
      distinct, sourceLanguage, fileAttr, producer, isOptimized, emissionKind,
#if LLVM_VERSION_MAJOR > 22
      /*isDebugInfoForProfiling*/ false,
#endif
      nameTableKind);

  compileUnitAttr_ = compileUnit;

  return compileUnit;
}

auto MlirDebugEmitter::compilationDirectory() -> const std::string& {
  if (!compilationDirectory_.has_value()) {
    auto attr = emitter_.module()->getAttrOfType<mlir::StringAttr>(
        "cxx.debug-compilation-dir");
    compilationDirectory_ = attr ? attr.getValue().str() : std::string{};
  }
  return compilationDirectory_.value();
}

auto MlirDebugEmitter::getOrCreateFileAttr(const std::string& filename)
    -> mlir::LLVM::DIFileAttr {
  if (auto it = fileAttrs_.find(filename); it != fileAttrs_.end()) {
    return it->second;
  }

  auto [file, directory] = debugFilePath(filename, compilationDirectory());
  auto attr = mlir::LLVM::DIFileAttr::get(context_, file, directory);

  fileAttrs_.insert_or_assign(filename, attr);

  return attr;
}

auto MlirDebugEmitter::getFileAttr(const std::string& filename)
    -> mlir::LLVM::DIFileAttr {
  if (filename.empty()) return getCompileUnitAttr().getFile();

  return getOrCreateFileAttr(filename);
}

auto MlirDebugEmitter::getFileAttr(std::string_view filename)
    -> mlir::LLVM::DIFileAttr {
  return getFileAttr(std::string{filename});
}

auto MlirDebugEmitter::getFileAttrAt(SourceLocation location)
    -> mlir::LLVM::DIFileAttr {
  if (!location) return getCompileUnitAttr().getFile();

  return getFileAttr(unit_->tokenStartPosition(location).fileName);
}

}  // namespace cxx::ir
