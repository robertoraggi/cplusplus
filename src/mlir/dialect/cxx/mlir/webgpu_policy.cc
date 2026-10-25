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

#include <cxx/mlir/atomic_ops.h>
#include <cxx/mlir/webgpu_policy.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace mlir::cxx {
namespace {

[[nodiscard]] auto isWebGPUBarrier(ControlBarrierOp barrier) -> bool {
  if (barrier.getOrdering() != MemoryOrder::AcquireRelease) return false;
  if (barrier.getExecutionScope() != MemoryScope::Workgroup) return false;
  return deviceMemoryScope(barrier.getMemoryScope()) == MemoryScope::Workgroup;
}

[[nodiscard]] auto isNaNTest(arith::CmpFPredicate predicate) -> bool {
  return predicate == arith::CmpFPredicate::ORD ||
         predicate == arith::CmpFPredicate::UNO;
}

[[nodiscard]] auto isNonFinite(Attribute value) -> bool {
  if (auto real = dyn_cast<FloatAttr>(value))
    return !real.getValue().isFinite();
  auto elements = dyn_cast<DenseFPElementsAttr>(value);
  if (!elements) return false;
  return llvm::any_of(elements.getValues<APFloat>(),
                      [](const APFloat& real) { return !real.isFinite(); });
}

[[nodiscard]] auto verifyWebGPUOperation(Operation* op) -> LogicalResult {
  return llvm::TypeSwitch<Operation*, LogicalResult>(op)
      .Case([](AtomicFenceOp fence) -> LogicalResult {
        if (isSignalFence(fence)) return success();
        return fence.emitError("WebGPU cannot represent a C++ thread fence");
      })
      .Case([](ControlBarrierOp barrier) -> LogicalResult {
        if (isWebGPUBarrier(barrier)) return success();
        return barrier.emitError(
            "WebGPU barriers require workgroup scopes and acquire/release "
            "semantics");
      })
      .Case([](arith::CmpFOp compare) -> LogicalResult {
        if (!isNaNTest(compare.getPredicate())) return success();
        return compare.emitError(
            "WebGPU assumes finite floating point and cannot test for NaN");
      })
      .Case([](arith::ConstantOp constant) -> LogicalResult {
        if (!isNonFinite(constant.getValue())) return success();
        return constant.emitError(
            "WebGPU cannot represent a non-finite floating-point constant");
      })
      .Default([](Operation* op) -> LogicalResult {
        if (llvm::none_of(atomicOrders(op), isOrdering)) return success();
        return op->emitError("WebGPU atomics require relaxed memory order");
      });
}

}  // namespace

auto verifyWebGPUOperations(Operation* root) -> LogicalResult {
  auto result = root->walk([](Operation* op) {
    if (succeeded(verifyWebGPUOperation(op))) return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

}  // namespace mlir::cxx
