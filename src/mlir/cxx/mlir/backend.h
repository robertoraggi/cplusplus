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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>

#include <expected>
#include <functional>
#include <string>
#include <string_view>

namespace llvm {
class raw_ostream;
}

namespace cxx {

class TimeTrace;

using MLIRDiagnosticSink = std::function<bool(mlir::Diagnostic&)>;

enum class BackendOutput {
  kCxxIR,
  kMLIR,
  kLLVMIR,
  kAssembly,
  kObject,
  kSPIRV,
  kWGSL,
};

struct BackendOptions {
  BackendOutput output = BackendOutput::kObject;
  int optimizationLevel = 0;
  bool debugInfo = false;
  bool verifyIR = true;
  std::string_view passPipeline;
  std::string_view kernel;
  TimeTrace* timeTrace = nullptr;
  MLIRDiagnosticSink diagnosticSink;
};

struct BackendArtifact {
  std::string code;
  std::string manifest;
};

[[nodiscard]] auto parseCxxIR(mlir::MLIRContext& context,
                              std::string_view source, bool verifyIR)
    -> mlir::OwningOpRef<mlir::ModuleOp>;

[[nodiscard]] auto compileModule(mlir::ModuleOp module,
                                 const BackendOptions& options,
                                 llvm::raw_ostream& diagnostics)
    -> std::expected<BackendArtifact, std::string>;

}  // namespace cxx
