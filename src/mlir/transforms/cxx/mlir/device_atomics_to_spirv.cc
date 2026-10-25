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
#include <cxx/mlir/device_atomics_to_spirv.h>
#include <llvm/Config/llvm-config.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace {

using spirv::MemorySemantics;

[[nodiscard]] auto scopeToSPIRV(cxx::MemoryScope scope) -> spirv::Scope {
  switch (cxx::deviceMemoryScope(scope)) {
    case cxx::MemoryScope::Workgroup:
      return spirv::Scope::Workgroup;
    case cxx::MemoryScope::Subgroup:
      return spirv::Scope::Subgroup;
    case cxx::MemoryScope::Invocation:
      return spirv::Scope::Invocation;
    default:
      return spirv::Scope::Device;
  }
}

[[nodiscard]] auto orderToSPIRV(cxx::MemoryOrder order) -> MemorySemantics {
  switch (order) {
    case cxx::MemoryOrder::Acquire:
      return MemorySemantics::Acquire;
    case cxx::MemoryOrder::Release:
      return MemorySemantics::Release;
    case cxx::MemoryOrder::AcquireRelease:
      return MemorySemantics::AcquireRelease;
    case cxx::MemoryOrder::SequentiallyConsistent:
      return MemorySemantics::SequentiallyConsistent;
    default:
      return MemorySemantics::None;
  }
}

[[nodiscard]] auto regionsToSPIRV(cxx::MemoryRegions regions)
    -> MemorySemantics {
  auto result = MemorySemantics::None;
  if (cxx::bitEnumContainsAll(regions, cxx::MemoryRegions::Storage))
    result = result | MemorySemantics::UniformMemory;
  if (cxx::bitEnumContainsAll(regions, cxx::MemoryRegions::Workgroup))
    result = result | MemorySemantics::WorkgroupMemory;
  return result;
}

[[nodiscard]] auto semanticsToSPIRV(cxx::MemoryOrder order,
                                    cxx::MemoryRegions regions)
    -> MemorySemantics {
  if (order == cxx::MemoryOrder::Relaxed) return MemorySemantics::None;
  return orderToSPIRV(order) | regionsToSPIRV(regions);
}

[[nodiscard]] auto atomicSemantics(cxx::MemoryOrder order) -> MemorySemantics {
  return semanticsToSPIRV(
      order, cxx::MemoryRegions::Storage | cxx::MemoryRegions::Workgroup);
}

[[nodiscard]] auto isAtomicStorage(Value pointer) -> bool {
  auto type = dyn_cast<spirv::PointerType>(pointer.getType());
  if (!type || !type.getPointeeType().isInteger(32)) return false;
  switch (type.getStorageClass()) {
    case spirv::StorageClass::StorageBuffer:
    case spirv::StorageClass::Workgroup:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto equalOrder(cxx::AtomicCompareExchangeOp op)
    -> cxx::MemoryOrder {
  auto success = op.getSuccessOrder();
  if (op.getFailureOrder() != cxx::MemoryOrder::Acquire) return success;
  switch (success) {
    case cxx::MemoryOrder::Relaxed:
      return cxx::MemoryOrder::Acquire;
    case cxx::MemoryOrder::Release:
      return cxx::MemoryOrder::AcquireRelease;
    default:
      return success;
  }
}

[[nodiscard]] auto checkAtomic(Operation* op, Value pointer,
                               ConversionPatternRewriter& rewriter)
    -> LogicalResult {
  if (isAtomicStorage(pointer)) return success();
  return rewriter.notifyMatchFailure(
      op, "atomic pointer requires 32-bit storage-buffer or workgroup memory");
}

class AtomicLoadToSPIRV final : public OpConversionPattern<cxx::AtomicLoadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AtomicLoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (failed(checkAtomic(op, adaptor.getAddr(), rewriter))) return failure();
#if LLVM_VERSION_MAJOR >= 23
    rewriter.replaceOpWithNewOp<spirv::AtomicLoadOp>(
        op, op.getType(), adaptor.getAddr(), scopeToSPIRV(op.getScope()),
        atomicSemantics(op.getOrdering()));
    return success();
#else
    return op.emitError("SPIR-V atomic load requires MLIR 23 or later");
#endif
  }
};

class AtomicStoreToSPIRV final
    : public OpConversionPattern<cxx::AtomicStoreOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AtomicStoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (failed(checkAtomic(op, adaptor.getAddr(), rewriter))) return failure();
#if LLVM_VERSION_MAJOR >= 23
    rewriter.replaceOpWithNewOp<spirv::AtomicStoreOp>(
        op, adaptor.getAddr(), scopeToSPIRV(op.getScope()),
        atomicSemantics(op.getOrdering()), adaptor.getValue());
    return success();
#else
    return op.emitError("SPIR-V atomic store requires MLIR 23 or later");
#endif
  }
};

class AtomicRMWToSPIRV final : public OpConversionPattern<cxx::AtomicRMWOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AtomicRMWOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (failed(checkAtomic(op, adaptor.getAddr(), rewriter))) return failure();
    auto pointer = adaptor.getAddr();
    auto scope = scopeToSPIRV(op.getScope());
    auto semantics = atomicSemantics(op.getOrdering());
    auto value = adaptor.getValue();
    switch (op.getKind()) {
      case cxx::AtomicRMWKind::Exchange:
        rewriter.replaceOpWithNewOp<spirv::AtomicExchangeOp>(op, pointer, scope,
                                                             semantics, value);
        return success();
      case cxx::AtomicRMWKind::Add:
        rewriter.replaceOpWithNewOp<spirv::AtomicIAddOp>(op, pointer, scope,
                                                         semantics, value);
        return success();
      case cxx::AtomicRMWKind::Sub:
        rewriter.replaceOpWithNewOp<spirv::AtomicISubOp>(op, pointer, scope,
                                                         semantics, value);
        return success();
      case cxx::AtomicRMWKind::And:
        rewriter.replaceOpWithNewOp<spirv::AtomicAndOp>(op, pointer, scope,
                                                        semantics, value);
        return success();
      case cxx::AtomicRMWKind::Or:
        rewriter.replaceOpWithNewOp<spirv::AtomicOrOp>(op, pointer, scope,
                                                       semantics, value);
        return success();
      case cxx::AtomicRMWKind::Xor:
        rewriter.replaceOpWithNewOp<spirv::AtomicXorOp>(op, pointer, scope,
                                                        semantics, value);
        return success();
      case cxx::AtomicRMWKind::Nand:
        return rewriter.notifyMatchFailure(
            op, "NAND is legalized as a compare-exchange loop");
    }
    return failure();
  }
};

class CompareExchangeToSPIRV final
    : public OpConversionPattern<cxx::AtomicCompareExchangeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AtomicCompareExchangeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (failed(checkAtomic(op, adaptor.getAddr(), rewriter))) return failure();
    auto loc = op.getLoc();
    auto old = spirv::AtomicCompareExchangeOp::create(
        rewriter, loc, op.getOld().getType(), adaptor.getAddr(),
        scopeToSPIRV(op.getScope()), atomicSemantics(equalOrder(op)),
        atomicSemantics(op.getFailureOrder()), adaptor.getDesired(),
        adaptor.getExpected());
    auto exchanged =
        spirv::IEqualOp::create(rewriter, loc, old, adaptor.getExpected());
    rewriter.replaceOp(op, {old, exchanged});
    return success();
  }
};

class FenceToSPIRV final : public OpConversionPattern<cxx::AtomicFenceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::AtomicFenceOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (op.getScope() == cxx::MemoryScope::Invocation) {
      rewriter.eraseOp(op);
      return success();
    }
    rewriter.replaceOpWithNewOp<spirv::MemoryBarrierOp>(
        op, scopeToSPIRV(op.getScope()),
        semanticsToSPIRV(op.getOrdering(), op.getMemoryRegions()));
    return success();
  }
};

class ControlBarrierToSPIRV final
    : public OpConversionPattern<cxx::ControlBarrierOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ControlBarrierOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    rewriter.replaceOpWithNewOp<spirv::ControlBarrierOp>(
        op, scopeToSPIRV(op.getExecutionScope()),
        scopeToSPIRV(op.getMemoryScope()),
        semanticsToSPIRV(op.getOrdering(), op.getMemoryRegions()));
    return success();
  }
};

}  // namespace
}  // namespace mlir

void cxx::populateDeviceAtomicsToSPIRVPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<mlir::AtomicLoadToSPIRV, mlir::AtomicStoreToSPIRV,
               mlir::AtomicRMWToSPIRV, mlir::CompareExchangeToSPIRV,
               mlir::FenceToSPIRV, mlir::ControlBarrierToSPIRV>(
      patterns.getContext());
}
