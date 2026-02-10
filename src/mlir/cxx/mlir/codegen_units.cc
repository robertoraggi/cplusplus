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
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/RegionUtils.h>

#include <filesystem>
#include <format>

namespace cxx {

namespace {
struct ForEachExternalDefinition final : ASTVisitor {
  std::function<void(FunctionDefinitionAST*)> functionCallback;

  void visit(TemplateDeclarationAST*) override {
    // Skip template declarations, we only want to visit function definitions.
  }

  void visit(FunctionDefinitionAST* ast) override {
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
        gen.findOrCreateGlobal(var);
        continue;
      }

      if (auto nestedNs = symbol_cast<NamespaceSymbol>(member)) {
        visitGlobals(nestedNs);
      }
    }
  }
};

auto Codegen::operator()(UnitAST* ast) -> UnitResult {
  if (!ast) return {};
  auto result = visit(UnitVisitor{*this}, ast);

  mlir::IRRewriter rewriter(result.module->getContext());

  result.module.walk([&](mlir::cxx::FuncOp funcOp) {
    llvm::DenseMap<llvm::StringRef, mlir::Block*> labels;

    for (auto& block : funcOp.getBody()) {
      for (auto& op : block) {
        if (auto labelOp = mlir::dyn_cast<mlir::cxx::LabelOp>(&op)) {
          labels[labelOp.getName()] = labelOp->getBlock();
        }
      }
    }

    llvm::SmallVector<mlir::cxx::GotoOp> gotoOps;
    llvm::SmallVector<mlir::cxx::LabelOp> labelOps;

    for (auto& block : funcOp.getBody()) {
      for (auto& op : block) {
        if (auto gotoOp = mlir::dyn_cast<mlir::cxx::GotoOp>(&op))
          gotoOps.push_back(gotoOp);
        else if (auto labelOp = mlir::dyn_cast<mlir::cxx::LabelOp>(&op))
          labelOps.push_back(labelOp);
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

    for (auto labelOp : labelOps) {
      rewriter.eraseOp(labelOp);
    }

    for (auto& region : funcOp->getRegions()) {
      eraseUnreachableBlocks(rewriter, region);
    }
  });

  return result;
}

auto Codegen::UnitVisitor::operator()(TranslationUnitAST* ast) -> UnitResult {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();

  auto ctx = gen.builder_.getContext();

  auto compileUnit = gen.getCompileUnitAttr(gen.unit_->fileName());

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
