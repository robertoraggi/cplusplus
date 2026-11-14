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

#include <cxx/mlir/backend.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/cxx_dialect_conversions.h>
#include <cxx/time_trace.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>

#ifdef CXX_WITH_SPIRV
#include <cxx/mlir/device_kernel.h>
#endif

#include <format>
#include <memory>
#include <optional>

namespace cxx {

namespace {

using BackendResult = std::expected<BackendArtifact, std::string>;

struct HostTarget {
  std::string triple;
  std::optional<llvm::Reloc::Model> relocationModel;
};

[[nodiscard]] auto isKernelOutput(BackendOutput output) -> bool {
  return output == BackendOutput::kSPIRV || output == BackendOutput::kWGSL;
}

[[nodiscard]] auto compilesForDevice(mlir::ModuleOp module,
                                     BackendOutput output) -> bool {
  if (isKernelOutput(output)) return true;
  return module->hasAttr("cxx.device_profile");
}

[[nodiscard]] auto printModule(mlir::ModuleOp module, bool debugInfo)
    -> BackendArtifact {
  BackendArtifact artifact;
  llvm::raw_string_ostream out(artifact.code);
  mlir::OpPrintingFlags flags;
  if (debugInfo) flags.enableDebugInfo(true);
  module->print(out, flags);
  return artifact;
}

[[nodiscard]] auto codeGenOptLevel(int level) -> llvm::CodeGenOptLevel {
  switch (level) {
    case 1:
      return llvm::CodeGenOptLevel::Less;
    case 2:
      return llvm::CodeGenOptLevel::Default;
    case 3:
      return llvm::CodeGenOptLevel::Aggressive;
    default:
      return llvm::CodeGenOptLevel::None;
  }
}

[[nodiscard]] auto pipelineOptLevel(int level) -> llvm::OptimizationLevel {
  switch (level) {
    case 1:
      return llvm::OptimizationLevel::O1;
    case 2:
      return llvm::OptimizationLevel::O2;
    case 3:
      return llvm::OptimizationLevel::O3;
    default:
      return llvm::OptimizationLevel::O0;
  }
}

[[nodiscard]] auto relocationModelOf(mlir::ModuleOp module)
    -> std::optional<llvm::Reloc::Model> {
  auto model = module->getAttrOfType<mlir::StringAttr>("cxx.relocation-model");
  if (!model) return std::nullopt;
  if (model.getValue() == "pic") return llvm::Reloc::PIC_;
  return llvm::Reloc::Static;
}

[[nodiscard]] auto hostTargetOf(mlir::ModuleOp module) -> HostTarget {
  HostTarget target;
  if (auto triple = module->getAttrOfType<mlir::StringAttr>("cxx.triple"))
    target.triple = triple.getValue().str();
  target.relocationModel = relocationModelOf(module);
  return target;
}

[[nodiscard]] auto createTargetMachine(const HostTarget& host,
                                       int optimizationLevel)
    -> std::expected<std::unique_ptr<llvm::TargetMachine>, std::string> {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  llvm::Triple triple{host.triple};
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target)
    return std::unexpected(std::format("cannot find target for triple '{}': {}",
                                       host.triple, error));

  llvm::TargetOptions options;
  options.UseInitArray = true;
  auto machine =
      std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
          triple, "generic", "", options, host.relocationModel, std::nullopt,
          codeGenOptLevel(optimizationLevel)));
  if (!machine)
    return std::unexpected(
        std::format("cannot create target machine for '{}'", host.triple));
  return machine;
}

[[nodiscard]] auto printLLVMModule(const llvm::Module& module)
    -> BackendArtifact {
  BackendArtifact artifact;
  llvm::raw_string_ostream out(artifact.code);
  module.print(out, nullptr);
  return artifact;
}

[[nodiscard]] auto emitMachineCode(llvm::Module& module,
                                   llvm::TargetMachine& machine,
                                   BackendOutput output) -> BackendResult {
  llvm::SmallString<0> buffer;
  llvm::raw_svector_ostream out(buffer);
  llvm::legacy::PassManager passes;
  const auto fileType = output == BackendOutput::kAssembly
                            ? llvm::CodeGenFileType::AssemblyFile
                            : llvm::CodeGenFileType::ObjectFile;
  if (machine.addPassesToEmitFile(passes, out, nullptr, fileType))
    return std::unexpected("target machine cannot emit assembly");
  passes.run(module);
  if (module.getContext().getDiagHandlerPtr()->HasErrors)
    return std::unexpected("failed to generate code");
  return BackendArtifact{.code = std::string(buffer.str())};
}

[[nodiscard]] auto compileHostOutput(mlir::ModuleOp module,
                                     const BackendOptions& options,
                                     llvm::raw_ostream&) -> BackendResult {
  const auto host = hostTargetOf(module);

  {
    TimeTrace::Scope trace(options.timeTrace, "Lower IR");
    if (failed(lowerToMLIR(
            module, {},
            {.verifyIR = options.verifyIR, .timeTrace = options.timeTrace})))
      return std::unexpected("failed to lower C++ AST to MLIR");
  }

  if (options.output == BackendOutput::kMLIR) {
    TimeTrace::Scope trace(options.timeTrace, "Print MLIR");
    return printModule(module, options.debugInfo);
  }

  auto machine = createTargetMachine(host, options.optimizationLevel);
  if (!machine) return std::unexpected(std::move(machine.error()));

  llvm::LLVMContext context;
  auto llvmModule = exportToLLVMIR(module, context, options.verifyIR);
  if (!llvmModule)
    return std::unexpected("failed to lower MLIR module to LLVM IR");
  if (auto name = module.getName()) llvmModule->setSourceFileName(*name);

  optimizeLLVMIR(*llvmModule, machine->get(),
                 pipelineOptLevel(options.optimizationLevel));

  if (options.output == BackendOutput::kLLVMIR)
    return printLLVMModule(*llvmModule);

  return emitMachineCode(*llvmModule, **machine, options.output);
}

#ifdef CXX_WITH_SPIRV

[[nodiscard]] auto kernelArtifact(mlir::FailureOr<KernelArtifact> compiled)
    -> BackendResult {
  if (failed(compiled))
    return std::unexpected("failed to compile the device kernel");
  return BackendArtifact{
      .code = std::string(compiled->bytes.begin(), compiled->bytes.end()),
      .manifest = std::move(compiled->manifest)};
}

[[nodiscard]] auto printDeviceStage(mlir::ModuleOp module,
                                    llvm::StringRef kernel, KernelStage stage)
    -> BackendResult {
  BackendArtifact artifact;
  llvm::raw_string_ostream out(artifact.code);
  if (failed(printDeviceModule(module, kernel, stage, out)))
    return std::unexpected("failed to lower the device module");
  return artifact;
}

[[nodiscard]] auto compileDeviceOutput(mlir::ModuleOp module,
                                       const BackendOptions& options,
                                       llvm::raw_ostream& diagnostics)
    -> BackendResult {
  DeviceDiagnostics scope(module.getContext(), diagnostics,
                          options.diagnosticSink);
  const llvm::StringRef kernel{options.kernel};
  switch (options.output) {
    case BackendOutput::kWGSL:
      return kernelArtifact(compileWGSLModule(module, kernel));
    case BackendOutput::kSPIRV:
    case BackendOutput::kObject:
      return kernelArtifact(compileSPIRVModule(module, kernel));
    case BackendOutput::kMLIR:
      return printDeviceStage(module, kernel, KernelStage::kGPU);
    case BackendOutput::kAssembly:
      return printDeviceStage(module, kernel, KernelStage::kSPIRV);
    case BackendOutput::kCxxIR:
      return printModule(module, options.debugInfo);
    case BackendOutput::kLLVMIR:
      return std::unexpected(
          "LLVM IR needs a host target; the device profile emits cxxir, "
          "mlir, asm, obj, wgsl and spirv");
  }
  std::unreachable();
}

#else

[[nodiscard]] auto compileDeviceOutput(mlir::ModuleOp, const BackendOptions&,
                                       llvm::raw_ostream&) -> BackendResult {
  return std::unexpected("this build has no device kernel support");
}

#endif

}  // namespace

auto parseCxxIR(mlir::MLIRContext& context, std::string_view source,
                bool verifyIR) -> mlir::OwningOpRef<mlir::ModuleOp> {
  context.loadDialect<mlir::cxx::CxxDialect>();
  mlir::ParserConfig config(&context, verifyIR);
  return mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef{source},
                                                 config);
}

auto compileModule(mlir::ModuleOp module, const BackendOptions& options,
                   llvm::raw_ostream& diagnostics)
    -> std::expected<BackendArtifact, std::string> {
  initializeCxxModuleDataLayout(module);

  {
    TimeTrace::Scope trace(options.timeTrace, "IR pipeline");
    if (failed(runMLIRPassPipeline(module,
                                   llvm::StringRef{options.passPipeline},
                                   options.verifyIR, &diagnostics)))
      return std::unexpected("failed to run MLIR pass pipeline");
  }

  if (options.output == BackendOutput::kCxxIR) {
    TimeTrace::Scope trace(options.timeTrace, "Print CXX IR");
    return printModule(module, options.debugInfo);
  }

  if (compilesForDevice(module, options.output))
    return compileDeviceOutput(module, options, diagnostics);

  return compileHostOutput(module, options, diagnostics);
}

}  // namespace cxx
