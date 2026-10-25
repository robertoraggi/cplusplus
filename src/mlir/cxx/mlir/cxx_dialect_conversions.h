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

#include <cxx/mlir/passes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <mlir/IR/BuiltinOps.h>

namespace llvm {
class TargetMachine;
class raw_ostream;
}

namespace cxx {
class TimeTrace;

// Preserve explicit layouts and otherwise obtain the target's LLVM layout.
// Both emitter paths need the same DLTI information before lowering.
void initializeCxxModuleDataLayout(mlir::ModuleOp module);

struct MLIRLoweringOptions {
  bool verifyIR = true;
  TimeTrace* timeTrace = nullptr;
};

[[nodiscard]] auto runMLIRPassPipeline(mlir::ModuleOp module,
                                       llvm::StringRef passPipeline,
                                       bool verifyIR = true,
                                       llvm::raw_ostream* errors = nullptr)
    -> mlir::LogicalResult;

[[nodiscard]] auto lowerToMLIR(mlir::ModuleOp module,
                               llvm::StringRef passPipeline = {},
                               MLIRLoweringOptions options = {})
    -> mlir::LogicalResult;

[[nodiscard]] auto exportToLLVMIR(mlir::ModuleOp module,
                                  llvm::LLVMContext& context,
                                  bool verifyIR = true)
    -> std::unique_ptr<llvm::Module>;

void optimizeLLVMIR(llvm::Module& module, llvm::TargetMachine* targetMachine,
                    llvm::OptimizationLevel level);

}  // namespace cxx
