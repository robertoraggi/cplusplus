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
#include <cxx/ast_visitor.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/views/symbols.h>

// mlir
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/RegionUtils.h>

#include <format>

namespace cxx {

namespace {
struct ForEachExternalDefinition final : ASTVisitor {
  std::function<void(FunctionDefinitionAST*)> functionCallback;

  void visit(TemplateDeclarationAST*) override {
    // Skip template declarations, we only want to visit function definitions.
  }

  void visit(FunctionDefinitionAST* ast) override {
    if (ast->symbol && ast->symbol->templateDeclaration()) return;
    if (functionCallback) functionCallback(ast);

    ASTVisitor::visit(ast);
  }
};

}  // namespace

struct Codegen::UnitVisitor {
  Codegen& gen;

  auto operator()(TranslationUnitAST* ast) -> UnitResult;
  auto operator()(ModuleUnitAST* ast) -> UnitResult;

  void visitGlobals(ScopeSymbol* scope) {
    auto ns = symbol_cast<NamespaceSymbol>(scope);
    if (!ns) return;

    for (auto member : views::members(ns)) {
      if (auto var = symbol_cast<VariableSymbol>(member)) {
        if (var->templateParameters()) continue;
        gen.findOrCreateGlobal(var);
        continue;
      }

      if (auto nestedNs = symbol_cast<NamespaceSymbol>(member)) {
        visitGlobals(nestedNs);
      }
    }
  }
};

void Codegen::resolveLabels() {
  auto funcOp = function_;

  mlir::IRRewriter rewriter(funcOp.getContext());

  llvm::DenseMap<llvm::StringRef, mlir::Block*> labels;

  for (auto& block : funcOp.getBody()) {
    for (auto& op : block) {
      if (auto labelOp = mlir::dyn_cast<mlir::cxx::LabelOp>(&op))
        labels[labelOp.getName()] = labelOp->getBlock();
    }
  }

  llvm::SmallVector<mlir::cxx::GotoOp> gotoOps;
  llvm::SmallVector<mlir::cxx::LabelOp> labelOps;
  llvm::SmallVector<mlir::cxx::CleanupBranchOp> cleanupBranchOps;
  llvm::SmallVector<mlir::cxx::LabelAddressOp> labelAddressOps;
  llvm::SmallVector<mlir::cxx::IndirectGotoOp> indirectGotoOps;

  for (auto& block : funcOp.getBody()) {
    for (auto& op : block) {
      if (auto gotoOp = mlir::dyn_cast<mlir::cxx::GotoOp>(&op))
        gotoOps.push_back(gotoOp);
      else if (auto labelOp = mlir::dyn_cast<mlir::cxx::LabelOp>(&op))
        labelOps.push_back(labelOp);
      else if (auto cbOp = mlir::dyn_cast<mlir::cxx::CleanupBranchOp>(&op))
        cleanupBranchOps.push_back(cbOp);
      else if (auto laOp = mlir::dyn_cast<mlir::cxx::LabelAddressOp>(&op))
        labelAddressOps.push_back(laOp);
      else if (auto igOp = mlir::dyn_cast<mlir::cxx::IndirectGotoOp>(&op))
        indirectGotoOps.push_back(igOp);
    }
  }

  // Also collect LabelAddressOps from global initializer regions that reference
  // labels defined in this function (stored via func_name attribute).
  if (auto module = funcOp->getParentOfType<mlir::ModuleOp>()) {
    for (auto& moduleOp : *module.getBody()) {
      if (auto globalOp = mlir::dyn_cast<mlir::cxx::GlobalOp>(&moduleOp)) {
        globalOp.walk([&](mlir::cxx::LabelAddressOp laOp) {
          if (auto fnAttr = laOp.getFuncNameAttr())
            if (fnAttr.getValue() == funcOp.getSymName())
              labelAddressOps.push_back(laOp);
        });
      }
    }
  }

  for (auto gotoOp : gotoOps) {
    auto targetBlock = labels.lookup(gotoOp.getLabel());
    if (!targetBlock) continue;

    rewriter.setInsertionPoint(gotoOp);

    if (auto nextOp = ++gotoOp->getIterator();
        mlir::isa<mlir::cf::BranchOp>(&*nextOp)) {
      rewriter.eraseOp(&*nextOp);
    }

    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(gotoOp, targetBlock);
  }

  if (!labelAddressOps.empty()) {
    auto* ctx = funcOp.getContext();
    llvm::DenseMap<llvm::StringRef, unsigned> labelToTagId;
    llvm::SmallVector<mlir::Block*> labelTargets;
    unsigned nextTagId = 0;

    for (auto labelAddrOp : labelAddressOps) {
      auto name = labelAddrOp.getLabelName();
      unsigned tagId;
      auto it = labelToTagId.find(name);
      if (it == labelToTagId.end()) {
        tagId = nextTagId++;
        labelToTagId[name] = tagId;
        auto* targetBlock = labels.lookup(name);
        if (targetBlock) {
          auto tagAttr = mlir::LLVM::BlockTagAttr::get(ctx, tagId);
          rewriter.setInsertionPointToStart(targetBlock);
          mlir::LLVM::BlockTagOp::create(rewriter, labelAddrOp.getLoc(),
                                         tagAttr);
          labelTargets.push_back(targetBlock);
        }
      } else {
        tagId = it->second;
      }
      rewriter.modifyOpInPlace(labelAddrOp, [&] {
        labelAddrOp.setTagIdAttr(
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), tagId));
      });
    }

    for (auto igOp : indirectGotoOps) {
      rewriter.setInsertionPoint(igOp);
      rewriter.replaceOpWithNewOp<mlir::cxx::IndirectGotoOp>(
          igOp, igOp.getTarget(), mlir::BlockRange{labelTargets});
    }
  }

  for (auto labelOp : labelOps) {
    rewriter.eraseOp(labelOp);
  }

  for (auto cbOp : cleanupBranchOps) {
    rewriter.setInsertionPoint(cbOp);
    auto addresses = cbOp.getAddresses();
    auto destructors = cbOp.getDestructors();
    for (unsigned i = 0; i < addresses.size(); ++i) {
      auto dtorRef = mlir::cast<mlir::FlatSymbolRefAttr>(destructors[i]);
      mlir::cxx::CallOp::create(rewriter, cbOp.getLoc(), mlir::Type{}, dtorRef,
                                mlir::ValueRange{addresses[i]},
                                mlir::TypeAttr{});
    }
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(cbOp, cbOp.getDest());
  }

  for (auto& region : funcOp->getRegions()) {
    eraseUnreachableBlocks(rewriter, region);
  }
}

auto Codegen::operator()(UnitAST* ast) -> UnitResult {
  if (!ast) return {};
  return visit(UnitVisitor{*this}, ast);
}

auto Codegen::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitResult {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();

  auto ctx = gen.context_;

  auto loc = mlir::FileLineColLoc::get(ctx, gen.unit_->fileName(), 0, 0);

  auto module =
      mlir::ModuleOp::create(gen.builder_, loc, gen.unit_->fileName());

  gen.builder_.setInsertionPointToStart(module.getBody());

  auto memoryLayout = gen.control()->memoryLayout();

  auto triple = llvm::Triple{gen.control()->memoryLayout()->triple()};

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    cxx_runtime_error(std::format("failed to find target for triple '{}': {}",
                                  triple.getTriple(), error));
  }

  llvm::TargetOptions opt;

  auto RM = std::optional<llvm::Reloc::Model>();

  auto targetMachine =
      target->createTargetMachine(triple, "generic", "", opt, RM);

  auto llvmDataLayout = targetMachine->createDataLayout();

  auto dataLayout =
      mlir::translateDataLayout(llvmDataLayout, module->getContext());

  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dataLayout);

  module->setAttr("cxx.triple", mlir::StringAttr::get(module->getContext(),
                                                      memoryLayout->triple()));

  module->setAttr(
      "cxx.data-layout",
      mlir::StringAttr::get(module->getContext(),
                            llvmDataLayout.getStringRepresentation()));

  std::swap(gen.module_, module);

  auto compileUnit = gen.getCompileUnitAttr(gen.unit_->fileName());

  visitGlobals(gen.unit_->globalScope());

  auto mainFileId = gen.unit_->preprocessor()->mainSourceFileId();

  ForEachExternalDefinition forEachExternalDefinition;

  forEachExternalDefinition.functionCallback =
      [&](FunctionDefinitionAST* function) {
        auto loc = function->firstSourceLocation();
        if (loc && gen.unit_->tokenAt(loc).fileId() == mainFileId) {
          (void)gen.declaration(function);
        }
      };

  for (auto node : ListView{ast->declarationList}) {
    forEachExternalDefinition.accept(node);
  }

  gen.processPendingFunctions();

  std::swap(gen.module_, module);

  UnitResult result{module};
  return result;
}

auto Codegen::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto loc = gen.builder_.getUnknownLoc();
  auto name = gen.unit_->fileName();
  auto module = mlir::ModuleOp::create(gen.builder_, loc, name);
  gen.builder_.setInsertionPointToStart(module.getBody());

  std::swap(gen.module_, module);

  auto globalModuleFragmentResult =
      gen.globalModuleFragment(ast->globalModuleFragment);

  auto moduleDeclarationResult = gen.moduleDeclaration(ast->moduleDeclaration);

  for (auto node : ListView{ast->declarationList}) {
    auto value = gen.declaration(node);
  }

  auto privateModuleFragmentResult =
      gen.privateModuleFragment(ast->privateModuleFragment);

  std::swap(gen.module_, module);

  UnitResult result{module};
  return result;
}

auto Codegen::globalModuleFragment(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto Codegen::privateModuleFragment(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentResult {
  if (!ast) return {};

  for (auto node : ListView{ast->declarationList}) {
    auto value = declaration(node);
  }

  return {};
}

auto Codegen::moduleDeclaration(ModuleDeclarationAST* ast)
    -> ModuleDeclarationResult {
  if (!ast) return {};

  auto moduleNameResult = moduleName(ast->moduleName);

  auto modulePartitionResult = modulePartition(ast->modulePartition);

  for (auto node : ListView{ast->attributeList}) {
    auto value = attributeSpecifier(node);
  }

  return {};
}

auto Codegen::moduleName(ModuleNameAST* ast) -> ModuleNameResult {
  if (!ast) return {};

  auto moduleQualifierResult = moduleQualifier(ast->moduleQualifier);

  return {};
}

auto Codegen::moduleQualifier(ModuleQualifierAST* ast)
    -> ModuleQualifierResult {
  if (!ast) return {};

  auto moduleQualifierResult = moduleQualifier(ast->moduleQualifier);

  return {};
}

auto Codegen::modulePartition(ModulePartitionAST* ast)
    -> ModulePartitionResult {
  if (!ast) return {};

  auto moduleNameResult = moduleName(ast->moduleName);

  return {};
}

auto Codegen::importName(ImportNameAST* ast) -> ImportNameResult {
  if (!ast) return {};

  auto modulePartitionResult = modulePartition(ast->modulePartition);

  auto moduleNameResult = moduleName(ast->moduleName);

  return {};
}

}  // namespace cxx
