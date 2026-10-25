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

#include <cxx/device_profile_limits.h>
#include <cxx/mlir/atomic_ops.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir::cxx {
namespace {

using Effects =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

[[nodiscard]] auto isLoadMemoryOrder(MemoryOrder order) -> bool {
  switch (order) {
    case MemoryOrder::Release:
    case MemoryOrder::AcquireRelease:
      return false;
    default:
      return true;
  }
}

[[nodiscard]] auto isStoreMemoryOrder(MemoryOrder order) -> bool {
  switch (order) {
    case MemoryOrder::Acquire:
    case MemoryOrder::AcquireRelease:
      return false;
    default:
      return true;
  }
}

[[nodiscard]] auto verifyAtomicObject(Operation* op, Value address, Type value)
    -> LogicalResult {
  if (cast<PointerType>(address.getType()).getElementType() != value)
    return op->emitOpError("atomic value type differs from its object");
  return success();
}

void addAtomicEffects(OpOperand& address, MemoryOrder order, bool isVolatile,
                      Effects& effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &address);
  effects.emplace_back(MemoryEffects::Write::get(), &address);
  if (order == MemoryOrder::Relaxed && !isVolatile) return;
  effects.emplace_back(MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get());
}

[[nodiscard]] auto memoryScopeOf(Operation* op) -> std::optional<MemoryScope> {
  return llvm::TypeSwitch<Operation*, std::optional<MemoryScope>>(op)
      .Case<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp,
            AtomicFenceOp>([](auto sync) { return sync.getScope(); })
      .Case([](ControlBarrierOp barrier) { return barrier.getMemoryScope(); })
      .Default([](Operation*) { return std::nullopt; });
}

[[nodiscard]] auto isDeviceMemoryScope(MemoryScope scope) -> bool {
  switch (deviceMemoryScope(scope)) {
    case MemoryScope::Device:
    case MemoryScope::Workgroup:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto hasRegions(MemoryRegions regions) -> bool {
  return regions != MemoryRegions::None;
}

[[nodiscard]] auto verifyDeviceMemoryScope(Operation* op) -> LogicalResult {
  if (isSignalFence(op)) return success();
  auto scope = memoryScopeOf(op);
  if (!scope || isDeviceMemoryScope(*scope)) return success();
  return op->emitError(
      "the device profile supports only device and workgroup memory scopes");
}

}  // namespace

auto atomicAddress(Operation* op) -> Value {
  return llvm::TypeSwitch<Operation*, Value>(op)
      .Case<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(
          [](auto atomic) { return atomic.getAddr(); })
      .Default([](Operation*) { return Value(); });
}

auto isAtomicWrite(Operation* op) -> bool {
  return isa<AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp>(op);
}

auto isSynchronization(Operation* op) -> bool {
  return isa<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp, AtomicCompareExchangeOp,
             AtomicFenceOp, ControlBarrierOp>(op);
}

auto atomicOrders(Operation* op) -> llvm::SmallVector<MemoryOrder, 2> {
  return llvm::TypeSwitch<Operation*, llvm::SmallVector<MemoryOrder, 2>>(op)
      .Case<AtomicLoadOp, AtomicStoreOp, AtomicRMWOp, AtomicFenceOp,
            ControlBarrierOp>([](auto sync) {
        return llvm::SmallVector<MemoryOrder, 2>{sync.getOrdering()};
      })
      .Case([](AtomicCompareExchangeOp exchange) {
        return llvm::SmallVector<MemoryOrder, 2>{exchange.getSuccessOrder(),
                                                 exchange.getFailureOrder()};
      })
      .Default([](Operation*) { return llvm::SmallVector<MemoryOrder, 2>{}; });
}

auto isOrdering(MemoryOrder order) -> bool {
  return order != MemoryOrder::Relaxed;
}

auto isSignalFence(Operation* op) -> bool {
  auto fence = dyn_cast<AtomicFenceOp>(op);
  return fence && fence.getScope() == MemoryScope::Invocation;
}

auto deviceMemoryScope(MemoryScope scope) -> MemoryScope {
  if (scope == MemoryScope::System) return MemoryScope::Device;
  return scope;
}

auto verifyDeviceSynchronization(Operation* op) -> LogicalResult {
  if (!isSynchronization(op)) return success();
  if (llvm::is_contained(atomicOrders(op), MemoryOrder::SequentiallyConsistent))
    return op->emitError(
        "the device profile cannot represent sequentially consistent atomics "
        "or fences");
  if (auto address = atomicAddress(op);
      address &&
      !cast<PointerType>(address.getType()).getElementType().isInteger(32))
    return op->emitError("device atomics require a 32-bit integer object");
  if (auto barrier = dyn_cast<ControlBarrierOp>(op);
      barrier && barrier.getExecutionScope() != MemoryScope::Workgroup)
    return op->emitError(
        "the device profile requires workgroup barrier execution scope");
  return verifyDeviceMemoryScope(op);
}

auto WorkgroupAllocOp::verify() -> LogicalResult {
  auto pointer = cast<PointerType>(getType());
  auto space = dyn_cast_or_null<LLVM::LLVMAddrSpaceAttrInterface>(
      pointer.getMemorySpace());
  if (!space || space.getAddressSpace() != ::cxx::kThreadgroupAddressSpace)
    return emitOpError("requires threadgroup address space 3");
  return success();
}

auto AtomicLoadOp::verify() -> LogicalResult {
  if (!isLoadMemoryOrder(getOrdering()))
    return emitOpError("invalid atomic load memory order");
  return verifyAtomicObject(*this, getAddr(), getType());
}

void AtomicLoadOp::getEffects(Effects& effects) {
  addAtomicEffects(getAddrMutable(), getOrdering(), getVolatile_(), effects);
}

auto AtomicStoreOp::verify() -> LogicalResult {
  if (!isStoreMemoryOrder(getOrdering()))
    return emitOpError("invalid atomic store memory order");
  return verifyAtomicObject(*this, getAddr(), getValue().getType());
}

void AtomicStoreOp::getEffects(Effects& effects) {
  addAtomicEffects(getAddrMutable(), getOrdering(), getVolatile_(), effects);
}

auto AtomicRMWOp::verify() -> LogicalResult {
  return verifyAtomicObject(*this, getAddr(), getType());
}

void AtomicRMWOp::getEffects(Effects& effects) {
  addAtomicEffects(getAddrMutable(), getOrdering(), getVolatile_(), effects);
}

auto AtomicCompareExchangeOp::verify() -> LogicalResult {
  if (!isLoadMemoryOrder(getFailureOrder()))
    return emitOpError("invalid compare-exchange failure memory order");
  return verifyAtomicObject(*this, getAddr(), getExpected().getType());
}

void AtomicCompareExchangeOp::getEffects(Effects& effects) {
  addAtomicEffects(getAddrMutable(), getSuccessOrder(), getVolatile_(),
                   effects);
}

auto AtomicFenceOp::verify() -> LogicalResult {
  if (!isOrdering(getOrdering()))
    return emitOpError("a relaxed fence has no effects");
  if (!hasRegions(getMemoryRegions()))
    return emitOpError("requires storage and/or workgroup memory regions");
  return success();
}

auto ControlBarrierOp::verify() -> LogicalResult {
  if (isOrdering(getOrdering()) == hasRegions(getMemoryRegions()))
    return success();
  return emitOpError(
      "an ordered barrier requires memory regions and a relaxed barrier "
      "requires none");
}

}  // namespace mlir::cxx
