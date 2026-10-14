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

#include <cxx/mlir/cxx_dialect_conversions.h>
#include <llvm/IR/Module.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

namespace cxx {
void registerCxxLLVMTranslation(mlir::MLIRContext& context);
}

namespace {
void canonicalizeLLVM(mlir::ModuleOp module) {
  auto* context = module.getContext();
  auto* dialect = context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlir::RewritePatternSet patterns(context);
  dialect->getCanonicalizationPatterns(patterns);
  for (auto operation : context->getRegisteredOperations()) {
    if (&operation.getDialect() == dialect)
      operation.getCanonicalizationPatterns(patterns, context);
  }
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Normal);
  (void)mlir::applyPatternsGreedily(module, std::move(patterns), config);
}
}

auto cxx::runMLIRPassPipeline(mlir::ModuleOp module,
                              llvm::StringRef passPipeline)
    -> mlir::LogicalResult {
  if (passPipeline.empty()) return mlir::success();

  registerCxxToLLVMPass();
  registerConvertCxxToPtrPass();
  registerConvertCxxToMemRefPass();
  registerLiftCxxControlFlowToSCFPass();
  mlir::registerPass([] { return mlir::createSCFToControlFlowPass(); });
  mlir::registerCanonicalizerPass();
  mlir::registerMem2RegPass();
  mlir::memref::registerExpandStridedMetadataPass();
  mlir::registerPass([] { return mlir::createLowerAffinePass(); });

  mlir::PassManager pm(module->getName());
  if (failed(mlir::parsePassPipeline(passPipeline, pm))) return mlir::failure();
  return pm.run(module);
}

auto cxx::lowerToMLIR(mlir::ModuleOp module, llvm::StringRef passPipeline)
    -> mlir::LogicalResult {
  if (failed(runMLIRPassPipeline(module, passPipeline))) return mlir::failure();

  mlir::PassManager pm(module->getName());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(cxx::createLowerToLLVMPass());

  if (failed(pm.run(module))) {
    return mlir::failure();
  }

  canonicalizeLLVM(module);
  return mlir::success();
}

#ifndef CXX_WITH_LLVM_OPTIMIZER

void cxx::optimizeLLVMIR(llvm::Module&, llvm::TargetMachine*,
                         llvm::OptimizationLevel) {}

#else

#include <llvm/Passes/PassBuilder.h>

void cxx::optimizeLLVMIR(llvm::Module& module,
                         llvm::TargetMachine* targetMachine,
                         llvm::OptimizationLevel level) {
  if (level == llvm::OptimizationLevel::O0) return;

  llvm::LoopAnalysisManager loopAnalysisManager;
  llvm::FunctionAnalysisManager functionAnalysisManager;
  llvm::CGSCCAnalysisManager cgsccAnalysisManager;
  llvm::ModuleAnalysisManager moduleAnalysisManager;

  llvm::PassBuilder passBuilder{targetMachine};
  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cgsccAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cgsccAnalysisManager, moduleAnalysisManager);

  auto modulePassManager = passBuilder.buildPerModuleDefaultPipeline(level);
  modulePassManager.run(module, moduleAnalysisManager);
}

#endif

auto cxx::exportToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext& context)
    -> std::unique_ptr<llvm::Module> {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  registerCxxLLVMTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto llvmModule = mlir::translateModuleToLLVMIR(module, context);
  module->getContext()->loadDialect<mlir::LLVM::LLVMDialect>();

  if (llvmModule) {
    llvmModule->addModuleFlag(llvm::Module::Max, "Dwarf Version", 5);
  }

  return llvmModule;
}
