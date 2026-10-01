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
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

auto cxx::lowerToMLIR(mlir::ModuleOp module) -> mlir::LogicalResult {
  mlir::PassManager pm(module->getName());

#if false
  module->getContext()->disableMultithreading();
  pm.enableIRPrinting();
#endif

  pm.addPass(cxx::createLowerToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());

#if false
  pm.addPass(mlir::createCSEPass());
#endif

  if (failed(pm.run(module))) {
    return mlir::failure();
  }

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
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto llvmModule = mlir::translateModuleToLLVMIR(module, context);
  module->getContext()->loadDialect<mlir::LLVM::LLVMDialect>();

  if (llvmModule) {
    llvmModule->addModuleFlag(llvm::Module::Max, "Dwarf Version", 5);
  }

  return llvmModule;
}
