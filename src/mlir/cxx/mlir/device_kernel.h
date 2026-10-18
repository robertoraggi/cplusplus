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

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstdint>
#include <string>

namespace cxx {

struct KernelArtifact {
  llvm::SmallVector<std::uint8_t> bytes;
  std::string manifest;
};

[[nodiscard]] auto kernelEntryPoints(mlir::ModuleOp module)
    -> llvm::SmallVector<std::string>;

enum class KernelStage {
  kGPU,
  kSPIRV,
};

[[nodiscard]] auto lowerKernelToGPU(mlir::ModuleOp module,
                                    llvm::StringRef kernel)
    -> mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>;

[[nodiscard]] auto lowerKernelToSPIRV(mlir::ModuleOp module,
                                      llvm::StringRef kernel)
    -> mlir::FailureOr<mlir::OwningOpRef<mlir::Operation*>>;

[[nodiscard]] auto printKernels(mlir::ModuleOp module,
                                llvm::ArrayRef<std::string> kernels,
                                KernelStage stage, llvm::raw_ostream& out)
    -> mlir::LogicalResult;

[[nodiscard]] auto compileSPIRVKernel(mlir::ModuleOp module,
                                      llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact>;

[[nodiscard]] auto compileWGSLKernel(mlir::ModuleOp module,
                                     llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact>;

}  // namespace cxx
