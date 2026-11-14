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

#include <cxx/mlir/device_kernel.h>
#include <cxx/mlir/kernel_manifest.h>
#include <cxx/mlir/passes.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <utility>

namespace cxx {

namespace {

[[nodiscard]] auto severityName(mlir::DiagnosticSeverity severity)
    -> llvm::StringRef {
  switch (severity) {
    case mlir::DiagnosticSeverity::Error:
      return "error";
    case mlir::DiagnosticSeverity::Warning:
      return "warning";
    case mlir::DiagnosticSeverity::Note:
      return "note";
    case mlir::DiagnosticSeverity::Remark:
      return "remark";
  }
  std::unreachable();
}

}  // namespace

DeviceDiagnostics::DeviceDiagnostics(mlir::MLIRContext* context,
                                     llvm::raw_ostream& out,
                                     MLIRDiagnosticSink sink)
    : context_(context),
      out_(out),
      sink_(std::move(sink)),
      printedOperations_(context->shouldPrintOpOnDiagnostic()),
      handler_(context, [this](mlir::Diagnostic& diagnostic) {
        print(diagnostic);
        for (auto& note : diagnostic.getNotes()) print(note);
        return mlir::success();
      }) {
  context_->printOpOnDiagnostic(false);
}

DeviceDiagnostics::~DeviceDiagnostics() {
  context_->printOpOnDiagnostic(printedOperations_);
}

void DeviceDiagnostics::print(mlir::Diagnostic& diagnostic) {
  if (sink_ && sink_(diagnostic)) return;
  if (auto location =
          diagnostic.getLocation()->findInstanceOf<mlir::FileLineColLoc>()) {
    out_ << location.getFilename().getValue() << ':' << location.getLine()
         << ':' << location.getColumn() << ": ";
  }
  out_ << severityName(diagnostic.getSeverity()) << ": " << diagnostic.str()
       << '\n';
}

auto lowerDeviceModuleToGPU(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> {
  mlir::OwningOpRef<mlir::ModuleOp> working =
      mlir::cast<mlir::ModuleOp>(source->clone());
  mlir::PassManager lowering(source.getContext());
  lowering.addPass(createConvertCxxToMathPass());
  lowering.addPass(createConvertCxxKernelToGpuPass(kernel));
  lowering.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(lowering.run(*working))) return mlir::failure();
  return working;
}

auto printDeviceModule(mlir::ModuleOp module, llvm::StringRef kernel,
                       KernelStage stage, llvm::raw_ostream& out)
    -> mlir::LogicalResult {
  if (stage == KernelStage::kGPU) {
    auto lowered = lowerDeviceModuleToGPU(module, kernel);
    if (mlir::failed(lowered)) return mlir::failure();
    for (auto gpuModule : lowered->get().getOps<mlir::gpu::GPUModuleOp>()) {
      gpuModule.print(out);
      out << '\n';
    }
    return mlir::success();
  }
  auto lowered = lowerDeviceModuleToSPIRV(module, kernel);
  if (mlir::failed(lowered)) return mlir::failure();
  lowered->module->print(out);
  out << '\n';
  return mlir::success();
}

}  // namespace cxx
