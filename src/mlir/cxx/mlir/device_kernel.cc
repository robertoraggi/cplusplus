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

namespace cxx {

auto lowerKernelToGPU(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> {
  if (mlir::failed(kernelEntryOf(source, kernel))) return mlir::failure();
  mlir::OwningOpRef<mlir::ModuleOp> working =
      mlir::cast<mlir::ModuleOp>(source->clone());
  mlir::PassManager lowering(source.getContext());
  lowering.addPass(createConvertCxxToMathPass());
  lowering.addPass(createConvertCxxKernelToGpuPass(kernel));
  lowering.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(lowering.run(*working))) return mlir::failure();
  return working;
}

namespace {

[[nodiscard]] auto printKernel(mlir::ModuleOp module, llvm::StringRef kernel,
                               KernelStage stage, llvm::raw_ostream& out)
    -> mlir::LogicalResult {
  if (stage == KernelStage::kGPU) {
    auto lowered = lowerKernelToGPU(module, kernel);
    if (mlir::failed(lowered)) return mlir::failure();
    for (auto gpuModule : lowered->get().getOps<mlir::gpu::GPUModuleOp>()) {
      gpuModule.print(out);
      out << '\n';
    }
    return mlir::success();
  }
  auto lowered = lowerKernelToSPIRV(module, kernel);
  if (mlir::failed(lowered)) return mlir::failure();
  lowered->get()->print(out);
  out << '\n';
  return mlir::success();
}

}  // namespace

auto printKernels(mlir::ModuleOp module, llvm::ArrayRef<std::string> kernels,
                  KernelStage stage, llvm::raw_ostream& out)
    -> mlir::LogicalResult {
  auto result = mlir::success();
  for (const auto& kernel : kernels) {
    if (kernels.size() > 1) out << "// kernel " << kernel << '\n';
    if (mlir::failed(printKernel(module, kernel, stage, out)))
      result = mlir::failure();
  }
  return result;
}

}  // namespace cxx
