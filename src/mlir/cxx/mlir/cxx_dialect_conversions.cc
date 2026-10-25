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
#include <cxx/time_trace.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/Passes.h>

namespace cxx {
void registerCxxLLVMTranslation(mlir::MLIRContext& context);
}

void cxx::initializeCxxModuleDataLayout(mlir::ModuleOp module) {
  auto triple = module->getAttrOfType<mlir::StringAttr>("cxx.triple");
  if (!triple || triple.getValue().empty()) return;
  auto layout = [&]() -> llvm::DataLayout {
    if (auto attr = module->getAttrOfType<mlir::StringAttr>("cxx.data-layout"))
      return llvm::DataLayout(attr.getValue());
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::Triple targetTriple{triple.getValue()};
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target)
      llvm::report_fatal_error(
          llvm::Twine("failed to find target for triple '") +
          triple.getValue() + "': " + error);
    llvm::TargetOptions options;
    auto machine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(targetTriple, "generic", "", options,
                                    std::optional<llvm::Reloc::Model>()));
    return machine->createDataLayout();
  }();
  auto* context = module.getContext();
  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                  mlir::translateDataLayout(layout, context));
  module->setAttr(
      "cxx.data-layout",
      mlir::StringAttr::get(context, layout.getStringRepresentation()));
}

auto cxx::runMLIRPassPipeline(mlir::ModuleOp module,
                              llvm::StringRef passPipeline, bool verifyIR,
                              llvm::raw_ostream* errors)
    -> mlir::LogicalResult {
  if (passPipeline.empty()) return mlir::success();

  registerCxxToLLVMPass();
  registerConvertCxxToPtrPass();
  registerConvertCxxToMemRefPass();
  registerConvertCxxToFuncPass();
  registerConvertCxxToMathPass();
  registerLiftCxxControlFlowToSCFPass();
#ifdef CXX_WITH_SPIRV
  registerConvertCxxKernelToGpuPass();
  registerLegalizeCxxDeviceCallsPass();
  registerConvertCxxDeviceToSPIRVPass();
#endif
  mlir::registerPass([] { return mlir::createSCFToControlFlowPass(); });
  mlir::registerCanonicalizerPass();
  mlir::registerMem2RegPass();
  mlir::registerSROAPass();
  mlir::memref::registerExpandStridedMetadataPass();
  mlir::registerPass([] { return mlir::createLowerAffinePass(); });

  mlir::PassManager pm(module->getName());
  pm.enableVerifier(verifyIR);
  if (failed(mlir::parsePassPipeline(passPipeline, pm,
                                     errors ? *errors : llvm::errs())))
    return mlir::failure();
  return pm.run(module);
}

auto cxx::lowerToMLIR(mlir::ModuleOp module, llvm::StringRef passPipeline,
                      MLIRLoweringOptions options) -> mlir::LogicalResult {
  if (failed(runMLIRPassPipeline(module, passPipeline, options.verifyIR)))
    return mlir::failure();

  mlir::PassManager pm(module->getName());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(cxx::createLowerToLLVMPass());
  pm.enableVerifier(options.verifyIR);

  {
    TimeTrace::Scope trace(options.timeTrace, "MLIR conversion");
    if (failed(pm.run(module))) return mlir::failure();
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

auto cxx::exportToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext& context,
                         bool verifyIR) -> std::unique_ptr<llvm::Module> {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  registerCxxLLVMTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto llvmModule = mlir::translateModuleToLLVMIR(
      module, context, "LLVMDialectModule", /*disableVerification=*/!verifyIR);
  module->getContext()->loadDialect<mlir::LLVM::LLVMDialect>();

  if (llvmModule) {
    llvmModule->addModuleFlag(llvm::Module::Max, "Dwarf Version", 5);
  }

  return llvmModule;
}
