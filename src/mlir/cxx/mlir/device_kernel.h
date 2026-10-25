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

#include <cxx/mlir/kernel_contract.h>
#include <cxx/spirv_environment.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstdint>
#include <string>

namespace cxx {

class TranslationUnit;

class DeviceDiagnostics {
 public:
  DeviceDiagnostics(mlir::MLIRContext* context, llvm::raw_ostream& out,
                    const TranslationUnit* unit = nullptr);
  ~DeviceDiagnostics();

  DeviceDiagnostics(const DeviceDiagnostics&) = delete;
  auto operator=(const DeviceDiagnostics&) -> DeviceDiagnostics& = delete;

 private:
  void print(mlir::Diagnostic& diagnostic);
  [[nodiscard]] auto reportToUnit(mlir::Diagnostic& diagnostic) -> bool;

  mlir::MLIRContext* context_;
  llvm::raw_ostream& out_;
  const TranslationUnit* unit_;
  bool printedOperations_;
  mlir::ScopedDiagnosticHandler handler_;
};

struct KernelArtifact {
  llvm::SmallVector<std::uint8_t> bytes;
  std::string manifest;
};

enum class KernelStage {
  kGPU,
  kSPIRV,
};

[[nodiscard]] auto lowerDeviceModuleToGPU(mlir::ModuleOp module,
                                          llvm::StringRef kernel)
    -> mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>;

struct LoweredSPIRVModule {
  mlir::OwningOpRef<mlir::Operation*> module;
  mlir::cxx::KernelModuleContract contract;
  SPIRVEnvironment environment;
};

[[nodiscard]] auto lowerDeviceModuleToSPIRV(mlir::ModuleOp module,
                                            llvm::StringRef kernel)
    -> mlir::FailureOr<LoweredSPIRVModule>;

[[nodiscard]] auto printDeviceModule(mlir::ModuleOp module,
                                     llvm::StringRef kernel, KernelStage stage,
                                     llvm::raw_ostream& out)
    -> mlir::LogicalResult;

[[nodiscard]] auto compileSPIRVModule(mlir::ModuleOp module,
                                      llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact>;

[[nodiscard]] auto compileWGSLModule(mlir::ModuleOp module,
                                     llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact>;

}  // namespace cxx
