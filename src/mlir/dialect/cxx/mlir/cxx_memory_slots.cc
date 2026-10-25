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

#include <cxx/mlir/cxx_dialect.h>
#include <mlir/IR/Builders.h>

namespace mlir::cxx {

[[nodiscard]] auto ClassType::getSubelementIndexMap() const
    -> std::optional<DenseMap<Attribute, Type>> {
  if (getBody().empty()) return std::nullopt;

  DenseMap<Attribute, Type> result;
  auto indexType = IntegerType::get(getContext(), 32);
  for (auto [index, field] : llvm::enumerate(getBody()))
    result.try_emplace(IntegerAttr::get(indexType, index), field);
  return result;
}

[[nodiscard]] auto ClassType::getTypeAtIndex(Attribute index) const -> Type {
  auto integer = dyn_cast<IntegerAttr>(index);
  if (!integer || !integer.getType().isSignlessInteger(32)) return {};
  auto position = integer.getValue().getZExtValue();
  if (position >= getBody().size()) return {};
  return getBody()[position];
}

[[nodiscard]] auto AllocaOp::getDestructurableSlots()
    -> SmallVector<DestructurableMemorySlot> {
  if (!getOperation()->getDiscardableAttrDictionary().empty()) return {};
  auto element = cast<PointerType>(getType()).getElementType();
  auto record = dyn_cast<ClassType>(element);
  if (!record) return {};
  auto fields = record.getSubelementIndexMap();
  if (!fields) return {};
  return {{{getResult(), element}, *fields}};
}

auto AllocaOp::destructure(
    const DestructurableMemorySlot& slot,
    const SmallPtrSetImpl<Attribute>& usedIndices, OpBuilder& builder,
    SmallVectorImpl<DestructurableAllocationOpInterface>& newAllocators)
    -> DenseMap<Attribute, MemorySlot> {
  auto record = cast<ClassType>(slot.elemType);
  auto space = cast<PointerType>(getType()).getAddressSpace();
  DenseMap<Attribute, MemorySlot> result;
  for (auto [position, field] : llvm::enumerate(record.getBody())) {
    auto index = builder.getI32IntegerAttr(position);
    if (!usedIndices.contains(index)) continue;

    auto allocation = AllocaOp::create(
        builder, getLoc(), PointerType::get(getContext(), field, space),
        int64_t(0));
    newAllocators.push_back(allocation);
    result.try_emplace(index, MemorySlot{allocation.getResult(), field});
  }
  return result;
}

auto AllocaOp::handleDestructuringComplete(const DestructurableMemorySlot&,
                                           OpBuilder&)
    -> std::optional<DestructurableAllocationOpInterface> {
  erase();
  return std::nullopt;
}

[[nodiscard]] auto MemberOp::canRewire(
    const DestructurableMemorySlot& slot,
    SmallPtrSetImpl<Attribute>& usedIndices,
    SmallVectorImpl<MemorySlot>& mustBeSafelyUsed, const DataLayout&) -> bool {
  if (getBase() != slot.ptr) return false;
  auto index =
      IntegerAttr::get(IntegerType::get(getContext(), 32), getMemberIndex());
  auto found = slot.subelementTypes.find(index);
  if (found == slot.subelementTypes.end()) return false;
  if (cast<PointerType>(getType()).getElementType() != found->second)
    return false;

  usedIndices.insert(index);
  mustBeSafelyUsed.push_back({getResult(), found->second});
  return true;
}

auto MemberOp::rewire(const DestructurableMemorySlot&,
                      DenseMap<Attribute, MemorySlot>& subslots,
                      OpBuilder& builder, const DataLayout&) -> DeletionKind {
  auto index = builder.getI32IntegerAttr(getMemberIndex());
  getResult().replaceAllUsesWith(subslots.find(index)->second.ptr);
  return DeletionKind::Delete;
}

[[nodiscard]] auto MemberOp::ensureOnlySafeAccesses(
    const MemorySlot& slot, SmallVectorImpl<MemorySlot>& mustBeSafelyUsed,
    const DataLayout&) -> LogicalResult {
  if (getBase() != slot.ptr) return failure();
  if (cast<PointerType>(getBase().getType()).getElementType() != slot.elemType)
    return failure();

  auto element = cast<PointerType>(getType()).getElementType();
  mustBeSafelyUsed.push_back({getResult(), element});
  return success();
}

namespace {

[[nodiscard]] auto canDestructureAccess(Operation* operation, Value address,
                                        Type value,
                                        const DestructurableMemorySlot& slot,
                                        SmallPtrSetImpl<Attribute>& usedIndices)
    -> bool {
  if (!operation->getDiscardableAttrDictionary().empty()) return false;
  if (!isa<ClassType>(slot.elemType)) return false;
  if (address != slot.ptr || value != slot.elemType) return false;
  for (const auto& field : slot.subelementTypes)
    usedIndices.insert(field.first);
  return true;
}

}  // namespace

[[nodiscard]] auto LoadOp::canRewire(const DestructurableMemorySlot& slot,
                                     SmallPtrSetImpl<Attribute>& usedIndices,
                                     SmallVectorImpl<MemorySlot>&,
                                     const DataLayout&) -> bool {
  if (getVolatile_()) return false;
  return canDestructureAccess(*this, getAddr(), getType(), slot, usedIndices);
}

auto LoadOp::rewire(const DestructurableMemorySlot& slot,
                    DenseMap<Attribute, MemorySlot>& subslots,
                    OpBuilder& builder, const DataLayout&) -> DeletionKind {
  Value value = UndefOp::create(builder, getLoc(), slot.elemType);
  auto record = cast<ClassType>(slot.elemType);
  for (auto [position, field] : llvm::enumerate(record.getBody())) {
    auto& subslot = subslots.find(builder.getI32IntegerAttr(position))->second;
    auto element =
        LoadOp::create(builder, getLoc(), field, subslot.ptr, int64_t(0));
    value = InsertValueOp::create(builder, getLoc(), slot.elemType, value,
                                  element, int64_t(position));
  }
  getResult().replaceAllUsesWith(value);
  return DeletionKind::Delete;
}

[[nodiscard]] auto StoreOp::canRewire(const DestructurableMemorySlot& slot,
                                      SmallPtrSetImpl<Attribute>& usedIndices,
                                      SmallVectorImpl<MemorySlot>&,
                                      const DataLayout&) -> bool {
  if (getVolatile_()) return false;
  if (getValue() == slot.ptr) return false;
  return canDestructureAccess(*this, getAddr(), getValue().getType(), slot,
                              usedIndices);
}

auto StoreOp::rewire(const DestructurableMemorySlot& slot,
                     DenseMap<Attribute, MemorySlot>& subslots,
                     OpBuilder& builder, const DataLayout&) -> DeletionKind {
  auto record = cast<ClassType>(slot.elemType);
  for (auto [position, field] : llvm::enumerate(record.getBody())) {
    auto& subslot = subslots.find(builder.getI32IntegerAttr(position))->second;
    auto element = ExtractValueOp::create(builder, getLoc(), field, getValue(),
                                          int64_t(position));
    StoreOp::create(builder, getLoc(), element, subslot.ptr, int64_t(0));
  }
  return DeletionKind::Delete;
}

[[nodiscard]] auto LoadOp::ensureOnlySafeAccesses(const MemorySlot& slot,
                                                  SmallVectorImpl<MemorySlot>&,
                                                  const DataLayout&)
    -> LogicalResult {
  if (!getOperation()->getDiscardableAttrDictionary().empty()) return failure();
  if (getVolatile_()) return failure();
  return success(getAddr() == slot.ptr && getType() == slot.elemType);
}

[[nodiscard]] auto StoreOp::ensureOnlySafeAccesses(const MemorySlot& slot,
                                                   SmallVectorImpl<MemorySlot>&,
                                                   const DataLayout&)
    -> LogicalResult {
  if (!getOperation()->getDiscardableAttrDictionary().empty()) return failure();
  if (getVolatile_()) return failure();
  return success(getAddr() == slot.ptr && getValue() != slot.ptr &&
                 getValue().getType() == slot.elemType);
}

}  // namespace mlir::cxx
