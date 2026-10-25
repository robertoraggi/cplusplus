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
#include <cxx/mlir/passes.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Target/LLVMIR/TypeToLLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <limits>
#include <optional>

namespace mlir {
namespace {

[[nodiscard]] auto isSupportedAlignment(int64_t alignment) -> bool {
  if (alignment < 0) return false;
  return alignment == 0 || llvm::isPowerOf2_64(alignment);
}

[[nodiscard]] auto isSupportedLocalAllocation(cxx::AllocaOp op) -> bool {
  auto scope = op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
  if (!isa_and_nonnull<FunctionOpInterface>(scope)) return false;
  if (!op->getDiscardableAttrDictionary().empty()) return false;
  if (!isSupportedAlignment(op.getAlignment())) return false;
  auto pointer = cast<cxx::PointerType>(op.getType());
  if (pointer.getAddressSpace()) return false;
  return true;
}

[[nodiscard]] auto areStorageValueTypesCompatible(Type value, Type storage)
    -> bool {
  if (value == storage) return true;
  auto valuePointer = dyn_cast<cxx::PointerType>(value);
  auto storagePointer = dyn_cast<cxx::PointerType>(storage);
  if (!valuePointer || !storagePointer) return false;
  return cxx::haveSamePointerMemorySpace(valuePointer, storagePointer);
}

[[nodiscard]] auto isSupportedElementAccess(Operation* user, Value address,
                                            Type element) -> bool {
  if (!user->getDiscardableAttrDictionary().empty()) return false;
  if (cxx::isVolatileAccess(user)) return false;
  if (auto load = dyn_cast<cxx::LoadOp>(user)) {
    return load.getAddr() == address &&
           areStorageValueTypesCompatible(load.getType(), element) &&
           isSupportedAlignment(load.getAlignment());
  }
  if (auto store = dyn_cast<cxx::StoreOp>(user)) {
    return store.getAddr() == address &&
           areStorageValueTypesCompatible(store.getValue().getType(),
                                          element) &&
           isSupportedAlignment(store.getAlignment());
  }
  return false;
}

[[nodiscard]] auto isSupportedPointerBoundary(OpOperand& use) -> bool {
  auto* user = use.getOwner();
  if (!user->getDiscardableAttrDictionary().empty()) return false;
  if (isa<cxx::CallOp, cxx::InvokeOp, cxx::PtrToIntOp, cxx::BitfieldLoadOp,
          cxx::BitfieldStoreOp>(user))
    return true;
  if (auto branch = dyn_cast<BranchOpInterface>(user))
    return branch.getSuccessorBlockArgument(use.getOperandNumber()).has_value();
  if (isa<cxx::StoreOp, memref::StoreOp>(user))
    return use.getOperandNumber() == 0;
  if (auto copy = dyn_cast<cxx::MemCpyOp>(user)) return copy.getSize() >= 0;
  if (auto cast = dyn_cast<cxx::BitcastOp>(user)) {
    auto result = dyn_cast<cxx::PointerType>(cast.getType());
    return result && !result.getAddressSpace();
  }
  return false;
}

[[nodiscard]] auto isSupportedArrayElement(Type element) -> bool {
  if (auto integer = dyn_cast<IntegerType>(element))
    return integer.isSignless();
  return isa<Float16Type, BFloat16Type, Float32Type, Float64Type, Float80Type,
             Float128Type, cxx::PointerType>(element);
}

[[nodiscard]] auto isSupportedScalarElement(Type element) -> bool {
  if (isa<IntegerType, FloatType, cxx::PointerType>(element)) return true;
  auto vector = dyn_cast<VectorType>(element);
  if (!vector || vector.getRank() != 1 || vector.isScalable()) return false;
  auto lane = vector.getElementType();
  if (auto integer = dyn_cast<IntegerType>(lane))
    return integer.isSignless() && integer.getWidth() != 1;
  return isa<FloatType>(lane);
}

[[nodiscard]] auto getScalarStorageType(Type element) -> Type {
  if (element.isInteger(1)) return IntegerType::get(element.getContext(), 8);
  return element;
}

struct ScalarStorageAnalysis {
  Type storageElement;
  SmallVector<Operation*> accesses;
  SmallVector<cxx::MemSetZeroOp> zeros;
  SmallVector<OpOperand*> boundaries;
};

[[nodiscard]] auto analyzeScalarStorage(cxx::AllocaOp op,
                                        const DataLayout& layout)
    -> FailureOr<ScalarStorageAnalysis> {
  if (!isSupportedLocalAllocation(op)) return failure();
  auto pointer = cast<cxx::PointerType>(op.getType());
  auto element = pointer.getElementType();
  if (!isSupportedScalarElement(element)) return failure();
  ScalarStorageAnalysis result{getScalarStorageType(element), {}, {}, {}};
  for (auto& use : op.getResult().getUses()) {
    auto* user = use.getOwner();
    if (isSupportedPointerBoundary(use)) {
      result.boundaries.push_back(&use);
      continue;
    }
    if (isSupportedElementAccess(user, op.getResult(), element)) {
      result.accesses.push_back(user);
      continue;
    }
    if (auto zero = dyn_cast<cxx::MemSetZeroOp>(user)) {
      if (!zero->getDiscardableAttrDictionary().empty()) return failure();
      auto bytes = layout.getTypeSize(result.storageElement).getFixedValue();
      if (zero.getSize() < 0 || uint64_t(zero.getSize()) != bytes)
        return failure();
      if (isSupportedArrayElement(element) && !isa<cxx::PointerType>(element))
        result.zeros.push_back(zero);
      else
        result.boundaries.push_back(&use);
      continue;
    }
    return failure();
  }
  return result;
}

void convertElementAccess(Operation* user, Value allocation, ValueRange indices,
                          PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(user);
  if (auto load = dyn_cast<cxx::LoadOp>(user)) {
    Value value =
        memref::LoadOp::create(rewriter, load.getLoc(), allocation, indices,
                               false, llvm::MaybeAlign(load.getAlignment()));
    if (load.getType().isInteger(1))
      value = arith::TruncIOp::create(rewriter, load.getLoc(), load.getType(),
                                      value);
    else if (load.getType() != value.getType())
      value = cxx::BitcastOp::create(rewriter, load.getLoc(), load.getType(),
                                     value);
    rewriter.replaceOp(load, value);
    return;
  }
  auto store = cast<cxx::StoreOp>(user);
  Value value = store.getValue();
  auto element = cast<MemRefType>(allocation.getType()).getElementType();
  if (value.getType().isInteger(1)) {
    value = arith::ExtUIOp::create(rewriter, store.getLoc(), element, value);
  } else if (value.getType() != element) {
    value = cxx::BitcastOp::create(rewriter, store.getLoc(), element, value);
  }
  memref::StoreOp::create(rewriter, store.getLoc(), value, allocation, indices,
                          false, llvm::MaybeAlign(store.getAlignment()));
  rewriter.eraseOp(store);
}

struct ArrayStorageAnalysis {
  cxx::ArrayType array;
  SmallVector<int64_t> shape;
  Type storageElement;
  unsigned indexBits;
  uint64_t elementBytes;
  llvm::SetVector<Operation*> derived;
  llvm::SetVector<Operation*> accesses;
  SmallVector<OpOperand*> boundaries;
};

[[nodiscard]] auto isArrayAddressDerivation(Operation* op) -> bool {
  return isa<cxx::ArrayToPointerOp, cxx::SubscriptOp, cxx::PtrAddOp>(op);
}

[[nodiscard]] auto isSupportedArrayIndex(Value index) -> bool {
  if (isa<IndexType>(index.getType())) return true;
  auto integer = dyn_cast<IntegerType>(index.getType());
  return integer && integer.isSignless();
}

[[nodiscard]] auto getArrayElementDepth(cxx::ArrayType array, Type element)
    -> std::optional<unsigned> {
  Type current = array;
  unsigned depth = 0;
  while (current != element) {
    auto nested = dyn_cast<cxx::ArrayType>(current);
    if (!nested) return std::nullopt;
    current = nested.getElementType();
    ++depth;
  }
  return depth;
}

[[nodiscard]] auto getArrayElementCount(ArrayRef<int64_t> shape, unsigned depth)
    -> uint64_t {
  uint64_t count = 1;
  for (auto size : shape.drop_front(depth)) count *= size;
  return count;
}

[[nodiscard]] auto analyzeArrayStorage(Value address, const DataLayout& layout,
                                       bool expandZeroInit)
    -> FailureOr<ArrayStorageAnalysis> {
  auto pointer = cast<cxx::PointerType>(address.getType());
  auto array = dyn_cast<cxx::ArrayType>(pointer.getElementType());
  if (!array || !array.getSize()) return failure();
  SmallVector<int64_t> shape;
  Type element = array;
  while (auto nested = dyn_cast<cxx::ArrayType>(element)) {
    auto size = nested.getSize();
    if (!size || size > uint64_t(std::numeric_limits<int64_t>::max()))
      return failure();
    shape.push_back(int64_t(size));
    element = nested.getElementType();
  }
  if (!isSupportedArrayElement(element)) return failure();
  Type storage = getScalarStorageType(element);
  auto bytes = layout.getTypeSize(storage).getFixedValue();
  if (llvm::alignTo(bytes, layout.getTypeABIAlignment(storage)) != bytes)
    return failure();
  auto bits = layout.getTypeIndexBitwidth(
      LLVM::LLVMPointerType::get(address.getContext()));
  if (!bits || (*bits != 32 && *bits != 64)) return failure();
  auto maxBytes = llvm::APInt::getSignedMaxValue(*bits).getZExtValue();
  auto maxElements = maxBytes / bytes;
  uint64_t count = 1;
  for (auto size : shape) {
    if (uint64_t(size) > maxElements / count) return failure();
    count *= size;
  }
  ArrayStorageAnalysis analysis{array, shape, storage, unsigned(*bits),
                                bytes, {},    {},      {}};
  ForwardSliceOptions options;
  options.filter = isArrayAddressDerivation;
  getForwardSlice(address, &analysis.derived, options);
  SmallVector<Value> pointers{address};
  for (auto* derived : analysis.derived) {
    if (!derived->getDiscardableAttrDictionary().empty()) return failure();
    auto result = cast<cxx::PointerType>(derived->getResult(0).getType());
    auto depth = getArrayElementDepth(array, result.getElementType());
    if (!depth || !*depth) return failure();
    if (result.getAddressSpace()) return failure();
    if (auto subscript = dyn_cast<cxx::SubscriptOp>(derived)) {
      auto base = cast<cxx::PointerType>(subscript.getBase().getType());
      auto parent = dyn_cast<cxx::ArrayType>(base.getElementType());
      if (!parent || parent.getElementType() != result.getElementType())
        return failure();
      if (!isSupportedArrayIndex(subscript.getIndex())) return failure();
    }
    if (auto decay = dyn_cast<cxx::ArrayToPointerOp>(derived)) {
      auto base = cast<cxx::PointerType>(decay.getValue().getType());
      auto parent = dyn_cast<cxx::ArrayType>(base.getElementType());
      if (!parent || parent.getElementType() != result.getElementType())
        return failure();
    }
    if (auto add = dyn_cast<cxx::PtrAddOp>(derived)) {
      if (!isSupportedArrayIndex(add.getOffset())) return failure();
      auto base = cast<cxx::PointerType>(add.getBase().getType());
      auto parent = getArrayElementDepth(array, base.getElementType());
      if (!parent || (*depth != *parent && *depth != *parent + 1))
        return failure();
    }
    pointers.push_back(derived->getResult(0));
  }
  for (auto address : pointers) {
    for (auto& use : address.getUses()) {
      auto* user = use.getOwner();
      if (analysis.derived.contains(user)) continue;
      if (isSupportedPointerBoundary(use)) {
        analysis.boundaries.push_back(&use);
        continue;
      }
      if (auto zero = dyn_cast<cxx::MemSetZeroOp>(user)) {
        if (!zero->getDiscardableAttrDictionary().empty()) return failure();
        auto pointer = cast<cxx::PointerType>(address.getType());
        auto depth = *getArrayElementDepth(array, pointer.getElementType());
        auto byteSize = getArrayElementCount(shape, depth) * bytes;
        if (zero.getSize() < 0 || uint64_t(zero.getSize()) != byteSize)
          return failure();
        if (expandZeroInit && !isa<cxx::PointerType>(storage))
          analysis.accesses.insert(user);
        else
          analysis.boundaries.push_back(&use);
        continue;
      }
      auto pointer = cast<cxx::PointerType>(address.getType());
      if (pointer.getElementType() != element) return failure();
      if (!isSupportedElementAccess(user, address, element)) return failure();
      analysis.accesses.insert(user);
    }
  }
  return analysis;
}

[[nodiscard]] auto convertArrayIndex(Value value, unsigned bits,
                                     PatternRewriter& rewriter) -> Value {
  if (isa<IndexType>(value.getType())) return value;
  auto type = cast<IntegerType>(value.getType());
  auto word = rewriter.getIntegerType(bits);
  if (type.getWidth() < bits)
    value = arith::ExtSIOp::create(rewriter, value.getLoc(), word, value);
  if (type.getWidth() > bits)
    value = arith::TruncIOp::create(rewriter, value.getLoc(), word, value);
  return arith::IndexCastOp::create(rewriter, value.getLoc(),
                                    rewriter.getIndexType(), value);
}

void convertArrayZero(cxx::MemSetZeroOp op, Value allocation, ValueRange prefix,
                      PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();
  auto type = cast<MemRefType>(allocation.getType());
  auto lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  SmallVector<Value> lowerBounds, upperBounds, steps;
  for (auto size : type.getShape().drop_front(prefix.size())) {
    lowerBounds.push_back(lower);
    upperBounds.push_back(arith::ConstantIndexOp::create(rewriter, loc, size));
    steps.push_back(step);
  }
  auto element = type.getElementType();
  auto zero = arith::ConstantOp::create(rewriter, loc, element,
                                        rewriter.getZeroAttr(element));
  scf::buildLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder& builder, Location location, ValueRange indices) {
        SmallVector<Value> coordinates(prefix);
        llvm::append_range(coordinates, indices);
        memref::StoreOp::create(builder, location, zero, allocation,
                                coordinates, false, llvm::MaybeAlign(1));
      });
  rewriter.eraseOp(op);
}

[[nodiscard]] auto getArrayAddressBase(Operation* op) -> Value {
  if (auto add = dyn_cast<cxx::PtrAddOp>(op)) return add.getBase();
  if (auto subscript = dyn_cast<cxx::SubscriptOp>(op))
    return subscript.getBase();
  return cast<cxx::ArrayToPointerOp>(op).getValue();
}

[[nodiscard]] auto getArrayCoordinates(Value address,
                                       ArrayRef<IRMapping> mappings)
    -> SmallVector<Value> {
  SmallVector<Value> coordinates;
  for (const auto& mapping : mappings)
    coordinates.push_back(mapping.lookup(address));
  return coordinates;
}

[[nodiscard]] auto getArrayByteOffset(Value address,
                                      const ArrayStorageAnalysis& analysis,
                                      ArrayRef<IRMapping> mappings,
                                      PatternRewriter& rewriter) -> Value {
  Value offset = arith::ConstantIndexOp::create(rewriter, address.getLoc(), 0);
  for (unsigned dimension = 0; dimension < mappings.size(); ++dimension) {
    auto index = mappings[dimension].lookup(address);
    if (matchPattern(index, m_Zero())) continue;
    auto bytes = analysis.elementBytes *
                 getArrayElementCount(analysis.shape, dimension + 1);
    auto stride =
        arith::ConstantIndexOp::create(rewriter, address.getLoc(), bytes);
    auto term =
        arith::MulIOp::create(rewriter, address.getLoc(), index, stride);
    offset = arith::AddIOp::create(rewriter, address.getLoc(), offset, term);
  }
  return offset;
}

[[nodiscard]] auto bridgeMemRefAddress(Value storage, Value address,
                                       PatternRewriter& rewriter,
                                       Value offset = {}) -> Value {
  auto type = cast<MemRefType>(storage.getType());
  auto space = cast<ptr::MemorySpaceAttrInterface>(type.getMemorySpace());
  auto pointerType = ptr::PtrType::get(space);
  Value pointer =
      ptr::ToPtrOp::create(rewriter, address.getLoc(), pointerType, storage);
  if (offset && !matchPattern(offset, m_Zero()))
    pointer =
        ptr::PtrAddOp::create(rewriter, address.getLoc(), pointerType, pointer,
                              offset, ptr::PtrAddFlags::inbounds);
  return ptr::FromPtrOp::create(rewriter, address.getLoc(), address.getType(),
                                pointer, Value());
}

class ScalarStorageToMemRef : public OpRewritePattern<cxx::AllocaOp> {
 public:
  ScalarStorageToMemRef(MLIRContext* context, const DataLayoutAnalysis& layouts)
      : OpRewritePattern(context), layouts_(layouts) {}

  auto matchAndRewrite(cxx::AllocaOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto analysis = analyzeScalarStorage(op, layouts_.getAtOrAbove(op));
    if (failed(analysis)) return failure();
    Attribute space;
    if (!analysis->boundaries.empty())
      space = ptr::GenericSpaceAttr::get(op.getContext());
    auto type = MemRefType::get({}, analysis->storageElement,
                                MemRefLayoutAttrInterface(), space);
    auto alignment = op.getAlignment()
                         ? rewriter.getI64IntegerAttr(op.getAlignment())
                         : IntegerAttr();
    auto allocation =
        memref::AllocaOp::create(rewriter, op.getLoc(), type, alignment);
    llvm::SmallDenseMap<Operation*, Value, 4> bridges;
    for (auto* use : analysis->boundaries) {
      auto* user = use->getOwner();
      rewriter.setInsertionPoint(user);
      auto found = bridges.find(user);
      if (found == bridges.end())
        found =
            bridges
                .try_emplace(user, bridgeMemRefAddress(
                                       allocation, op.getResult(), rewriter))
                .first;
      rewriter.modifyOpInPlace(user, [&] { use->set(found->second); });
    }
    for (auto zero : analysis->zeros)
      convertArrayZero(zero, allocation, {}, rewriter);
    for (auto* access : analysis->accesses)
      convertElementAccess(access, allocation, {}, rewriter);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  const DataLayoutAnalysis& layouts_;
};

auto prepareArrayStorage(Value root, Value allocation,
                         const ArrayStorageAnalysis& analysis,
                         PatternRewriter& rewriter) -> SmallVector<IRMapping> {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(allocation.getDefiningOp());
  Value zero = arith::ConstantIndexOp::create(rewriter, root.getLoc(), 0);
  SmallVector<IRMapping> indices(analysis.shape.size());
  for (auto& mapping : indices) mapping.map(root, zero);
  for (auto* derived : analysis.derived) {
    rewriter.setInsertionPoint(derived);
    auto base = getArrayAddressBase(derived);
    auto coordinates = getArrayCoordinates(base, indices);
    auto pointer = cast<cxx::PointerType>(derived->getResult(0).getType());
    auto depth =
        *getArrayElementDepth(analysis.array, pointer.getElementType());
    Value offset;
    if (auto add = dyn_cast<cxx::PtrAddOp>(derived)) offset = add.getOffset();
    if (auto subscript = dyn_cast<cxx::SubscriptOp>(derived))
      offset = subscript.getIndex();
    if (offset) {
      offset = convertArrayIndex(offset, analysis.indexBits, rewriter);
      coordinates[depth - 1] = arith::AddIOp::create(
          rewriter, derived->getLoc(), coordinates[depth - 1], offset);
    }
    for (auto [mapping, coordinate] : llvm::zip(indices, coordinates))
      mapping.map(derived->getResult(0), coordinate);
  }
  for (auto* use : analysis.boundaries) {
    auto address = use->get();
    rewriter.setInsertionPoint(use->getOwner());
    auto offset = getArrayByteOffset(address, analysis, indices, rewriter);
    auto bridge = bridgeMemRefAddress(allocation, address, rewriter, offset);
    rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(bridge); });
  }
  return indices;
}

void convertArrayAccesses(Value allocation,
                          const ArrayStorageAnalysis& analysis,
                          ArrayRef<IRMapping> indices,
                          PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (auto* access : analysis.accesses) {
    if (auto zeroOp = dyn_cast<cxx::MemSetZeroOp>(access)) {
      auto pointer = cast<cxx::PointerType>(zeroOp.getAddr().getType());
      auto depth =
          *getArrayElementDepth(analysis.array, pointer.getElementType());
      auto coordinates = getArrayCoordinates(zeroOp.getAddr(), indices);
      convertArrayZero(zeroOp, allocation,
                       ArrayRef<Value>(coordinates).take_front(depth),
                       rewriter);
      continue;
    }
    convertElementAccess(
        access, allocation,
        getArrayCoordinates(cxx::getLoadStoreAddress(access), indices),
        rewriter);
  }
  for (auto* derived : llvm::reverse(analysis.derived))
    rewriter.eraseOp(derived);
}

class ArrayStorageToMemRef : public OpRewritePattern<cxx::AllocaOp> {
 public:
  ArrayStorageToMemRef(MLIRContext* context, const DataLayoutAnalysis& layouts,
                       bool expandZeroInit)
      : OpRewritePattern(context),
        layouts_(layouts),
        expandZeroInit_(expandZeroInit) {}

  auto matchAndRewrite(cxx::AllocaOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!isSupportedLocalAllocation(op)) return failure();
    auto analysis = analyzeArrayStorage(
        op.getResult(), layouts_.getAtOrAbove(op), expandZeroInit_);
    if (failed(analysis)) return failure();
    Attribute space;
    if (!analysis->boundaries.empty())
      space = ptr::GenericSpaceAttr::get(op.getContext());
    auto type = MemRefType::get(analysis->shape, analysis->storageElement,
                                MemRefLayoutAttrInterface(), space);
    auto alignment = op.getAlignment()
                         ? rewriter.getI64IntegerAttr(op.getAlignment())
                         : IntegerAttr();
    auto allocation =
        memref::AllocaOp::create(rewriter, op.getLoc(), type, alignment);
    auto indices =
        prepareArrayStorage(op.getResult(), allocation, *analysis, rewriter);
    convertArrayAccesses(allocation, *analysis, indices, rewriter);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  const DataLayoutAnalysis& layouts_;
  bool expandZeroInit_;
};

struct RecordStorageLayout {
  uint64_t bytes;
  uint64_t alignment;
};

[[nodiscard]] auto getRecordStorageLimit(MLIRContext* context,
                                         const DataLayout& layout)
    -> std::optional<uint64_t> {
  auto bits = layout.getTypeIndexBitwidth(LLVM::LLVMPointerType::get(context));
  if (!bits || (*bits != 32 && *bits != 64)) return std::nullopt;
  return llvm::APInt::getSignedMaxValue(*bits).getZExtValue();
}

class RecordStorageTypes {
 public:
  RecordStorageTypes(const llvm::DataLayout& llvmLayout,
                     LLVM::TypeToLLVMIRTranslator& translator)
      : llvmLayout_(llvmLayout), translator_(translator) {}

  [[nodiscard]] auto convert(Type field, const DataLayout& layout) -> Type {
    auto key = std::make_pair(field, &layout);
    if (auto found = types_.find(key); found != types_.end())
      return found->second;
    if (!active_.insert(field).second) return {};
    Type storage;
    if (auto pointer = dyn_cast<cxx::PointerType>(field)) {
      auto space = pointer.getMemorySpace();
      if (isa<ptr::GenericSpaceAttr>(space))
        storage = LLVM::LLVMPointerType::get(field.getContext());
      else if (auto llvmSpace =
                   dyn_cast<LLVM::LLVMAddrSpaceAttrInterface>(space))
        storage = LLVM::LLVMPointerType::get(field.getContext(),
                                             llvmSpace.getAddressSpace());
    } else if (isSupportedArrayElement(field)) {
      storage = getScalarStorageType(field);
    } else if (auto array = dyn_cast<cxx::ArrayType>(field)) {
      storage = convertArray(array, layout);
    } else if (auto record = dyn_cast<cxx::ClassType>(field)) {
      storage = convertRecord(record, layout);
    }
    if (storage) {
      auto* native = translator_.translateType(storage);
      auto bytes = layout.getTypeSize(storage);
      auto alignment = layout.getTypeABIAlignment(storage);
      if (bytes.isScalable() || bytes != llvmLayout_.getTypeAllocSize(native) ||
          alignment != llvmLayout_.getABITypeAlign(native).value())
        storage = {};
    }
    active_.erase(field);
    types_.try_emplace(key, storage);
    return storage;
  }

 private:
  [[nodiscard]] auto convertArray(cxx::ArrayType array,
                                  const DataLayout& layout) -> Type {
    if (array.getSize() > uint64_t(std::numeric_limits<int64_t>::max()))
      return {};
    auto element = convert(array.getElementType(), layout);
    if (!element) return {};
    auto limit = getRecordStorageLimit(array.getContext(), layout);
    if (!limit) return {};
    auto bytes = layout.getTypeSize(element).getFixedValue();
    if (array.getSize() && bytes > *limit / array.getSize()) return {};
    return LLVM::LLVMArrayType::get(element, array.getSize());
  }

  [[nodiscard]] auto convertRecord(cxx::ClassType record,
                                   const DataLayout& layout) -> Type {
    if (record.getBody().empty()) return {};
    auto limit = getRecordStorageLimit(record.getContext(), layout);
    if (!limit) return {};
    auto remaining = *limit;
    SmallVector<Type> fields;
    for (auto field : record.getBody()) {
      auto storage = convert(field, layout);
      if (!storage) return {};
      auto bytes = layout.getTypeSize(storage).getFixedValue();
      auto alignment = layout.getTypeABIAlignment(storage);
      if (bytes > remaining) return {};
      remaining -= bytes;
      if (alignment - 1 > remaining) return {};
      remaining -= alignment - 1;
      fields.push_back(storage);
    }
    return LLVM::LLVMStructType::getLiteral(record.getContext(), fields,
                                            record.isPacked());
  }

  const llvm::DataLayout& llvmLayout_;
  LLVM::TypeToLLVMIRTranslator& translator_;
  DenseMap<std::pair<Type, const DataLayout*>, Type> types_;
  DenseSet<Type> active_;
};

[[nodiscard]] auto getRecordStorageLayout(cxx::ClassType record,
                                          const DataLayout& layout,
                                          RecordStorageTypes& types)
    -> FailureOr<RecordStorageLayout> {
  auto storage = types.convert(record, layout);
  if (!storage) return failure();
  auto bytes = layout.getTypeSize(storage).getFixedValue();
  if (bytes > *getRecordStorageLimit(record.getContext(), layout))
    return failure();
  return RecordStorageLayout{bytes, layout.getTypeABIAlignment(storage)};
}

struct RecordStorageAnalysis {
  llvm::SetVector<Operation*> members;
  llvm::SetVector<Operation*> accesses;
  SmallVector<OpOperand*> boundaries;
  DenseMap<Value, uint64_t> offsets;
  DenseMap<Value, ArrayStorageAnalysis> arrays;
  DenseMap<Value, uint64_t> byteFields;

  [[nodiscard]] auto usesRawPointers() const -> bool {
    if (!boundaries.empty()) return true;
    return llvm::any_of(arrays, [](const auto& entry) {
      return !entry.second.boundaries.empty();
    });
  }
};

[[nodiscard]] auto analyzeRecordStorage(
    cxx::AllocaOp op, const RecordStorageLayout& storage,
    const DataLayout& layout, const llvm::DataLayout& llvmLayout,
    LLVM::TypeToLLVMIRTranslator& translator, RecordStorageTypes& types,
    bool expandZeroInit) -> FailureOr<RecordStorageAnalysis> {
  RecordStorageAnalysis result;
  result.offsets[op.getResult()] = 0;
  ForwardSliceOptions options;
  options.filter = [](Operation* operation) {
    return isa<cxx::MemberOp>(operation);
  };
  getForwardSlice(op.getResult(), &result.members, options);
  SmallVector<Value> addresses{op.getResult()};
  for (auto* operation : result.members) {
    auto member = cast<cxx::MemberOp>(operation);
    if (!member->getDiscardableAttrDictionary().empty()) return failure();
    auto base = cast<cxx::PointerType>(member.getBase().getType());
    auto parent = dyn_cast<cxx::ClassType>(base.getElementType());
    auto offset = result.offsets.find(member.getBase());
    if (!parent || offset == result.offsets.end()) return failure();
    auto index = uint64_t(member.getMemberIndex());
    if (index >= parent.getBody().size()) return failure();
    auto element = parent.getBody()[index];
    auto pointer = cast<cxx::PointerType>(member.getType());
    if (pointer.getAddressSpace()) return failure();
    if (pointer.getElementType() != element) {
      if (!isa<IntegerType>(pointer.getElementType())) return failure();
      if (member.getResult().use_empty()) return failure();
      if (!llvm::all_of(member.getResult().getUsers(), [](Operation* user) {
            return isa<cxx::BitfieldLoadOp, cxx::BitfieldStoreOp>(user);
          }))
        return failure();
      auto bytes = layout.getTypeSize(types.convert(element, layout));
      result.byteFields.try_emplace(member.getResult(), bytes.getFixedValue());
    }
    auto physical = types.convert(parent, layout);
    if (!physical) return failure();
    auto* native = cast<llvm::StructType>(translator.translateType(physical));
    auto fieldOffset =
        llvmLayout.getStructLayout(native)->getElementOffset(index);
    if (fieldOffset > storage.bytes - offset->second) return failure();
    result.offsets[member.getResult()] = offset->second + fieldOffset;
    if (isa<cxx::ArrayType>(element)) {
      auto array =
          analyzeArrayStorage(member.getResult(), layout, expandZeroInit);
      if (failed(array)) return failure();
      result.arrays.try_emplace(member.getResult(), std::move(*array));
    }
    addresses.push_back(member.getResult());
  }
  for (auto address : addresses) {
    if (result.arrays.contains(address)) continue;
    auto element = cast<cxx::PointerType>(address.getType()).getElementType();
    for (auto& use : address.getUses()) {
      auto* user = use.getOwner();
      if (result.members.contains(user)) continue;
      if (isSupportedPointerBoundary(use)) {
        result.boundaries.push_back(&use);
        continue;
      }
      if (auto zero = dyn_cast<cxx::MemSetZeroOp>(user)) {
        if (!zero->getDiscardableAttrDictionary().empty()) return failure();
        auto bytes =
            layout.getTypeSize(types.convert(element, layout)).getFixedValue();
        if (zero.getSize() < 0 || uint64_t(zero.getSize()) != bytes)
          return failure();
        if (expandZeroInit && !isa<cxx::PointerType>(element))
          result.accesses.insert(user);
        else
          result.boundaries.push_back(&use);
        continue;
      }
      if (!isSupportedElementAccess(user, address, element)) return failure();
      if (isa<cxx::ClassType>(element))
        result.boundaries.push_back(&use);
      else
        result.accesses.insert(user);
    }
  }
  return result;
}

class RecordStorageToMemRef : public OpRewritePattern<cxx::AllocaOp> {
 public:
  RecordStorageToMemRef(MLIRContext* context, const DataLayoutAnalysis& layouts,
                        const llvm::DataLayout& llvmLayout, bool expandZeroInit)
      : OpRewritePattern(context),
        layouts_(layouts),
        llvmLayout_(llvmLayout),
        translator_(llvmContext_),
        types_(llvmLayout_, translator_),
        expandZeroInit_(expandZeroInit) {}

  auto matchAndRewrite(cxx::AllocaOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!isSupportedLocalAllocation(op)) return failure();
    auto record = dyn_cast<cxx::ClassType>(
        cast<cxx::PointerType>(op.getType()).getElementType());
    if (!record) return failure();
    const auto& layout = layouts_.getAtOrAbove(op);
    auto key = std::make_pair(record, &layout);
    auto found = records_.find(key);
    if (found == records_.end())
      found =
          records_
              .try_emplace(key, getRecordStorageLayout(record, layout, types_))
              .first;
    const auto& storage = found->second;
    if (failed(storage)) return failure();
    auto analysis = analyzeRecordStorage(op, *storage, layout, llvmLayout_,
                                         translator_, types_, expandZeroInit_);
    if (failed(analysis)) return failure();
    Attribute space;
    if (analysis->usesRawPointers())
      space = ptr::GenericSpaceAttr::get(op.getContext());
    auto byteType =
        MemRefType::get({int64_t(storage->bytes)}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface(), space);
    auto alignment =
        op.getAlignment() ? uint64_t(op.getAlignment()) : storage->alignment;
    auto allocation = memref::AllocaOp::create(
        rewriter, op.getLoc(), byteType, rewriter.getI64IntegerAttr(alignment));
    IRMapping views;
    views.map(op.getResult(), allocation);
    DenseMap<std::pair<uint64_t, Type>, Value> fields;
    for (auto* operation : analysis->members) {
      auto member = cast<cxx::MemberOp>(operation);
      auto element = cast<cxx::PointerType>(member.getType()).getElementType();
      auto offset = analysis->offsets.lookup(member.getResult());
      SmallVector<int64_t> shape;
      Type storageElement;
      if (auto bytes = analysis->byteFields.find(member.getResult());
          bytes != analysis->byteFields.end()) {
        shape.push_back(bytes->second);
        storageElement = rewriter.getI8Type();
      } else if (auto array = analysis->arrays.find(member.getResult());
                 array != analysis->arrays.end()) {
        shape = array->second.shape;
        storageElement = array->second.storageElement;
      } else if (isa<cxx::ClassType>(element)) {
        shape.push_back(layout.getTypeSize(types_.convert(element, layout))
                            .getFixedValue());
        storageElement = rewriter.getI8Type();
      } else {
        storageElement = getScalarStorageType(element);
      }
      auto type = MemRefType::get(shape, storageElement,
                                  MemRefLayoutAttrInterface(), space);
      auto key = std::make_pair(offset, Type(type));
      auto found = fields.find(key);
      if (found == fields.end()) {
        auto shift =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), offset);
        Value field = memref::ViewOp::create(rewriter, op.getLoc(), type,
                                             allocation, shift, ValueRange{});
        found = fields.try_emplace(key, field).first;
      }
      views.map(member.getResult(), found->second);
    }
    DenseMap<Value, SmallVector<IRMapping>> arrayIndices;
    for (auto* operation : analysis->members) {
      auto address = operation->getResult(0);
      auto array = analysis->arrays.find(address);
      if (array != analysis->arrays.end())
        arrayIndices.try_emplace(
            address, prepareArrayStorage(address, views.lookup(address),
                                         array->second, rewriter));
    }
    for (auto* use : analysis->boundaries) {
      auto address = use->get();
      rewriter.setInsertionPoint(use->getOwner());
      auto bridge =
          bridgeMemRefAddress(views.lookup(address), address, rewriter);
      rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(bridge); });
    }
    for (auto* operation : analysis->members) {
      auto address = operation->getResult(0);
      auto array = analysis->arrays.find(address);
      if (array != analysis->arrays.end())
        convertArrayAccesses(views.lookup(address), array->second,
                             arrayIndices.find(address)->second, rewriter);
    }
    for (auto* access : analysis->accesses) {
      if (auto zero = dyn_cast<cxx::MemSetZeroOp>(access)) {
        convertArrayZero(zero, views.lookup(zero.getAddr()), {}, rewriter);
        continue;
      }
      convertElementAccess(
          access, views.lookup(cxx::getLoadStoreAddress(access)), {}, rewriter);
    }
    for (auto* member : llvm::reverse(analysis->members))
      rewriter.eraseOp(member);
    rewriter.eraseOp(op);
    return success();
  }

 private:
  const DataLayoutAnalysis& layouts_;
  llvm::DataLayout llvmLayout_;
  llvm::LLVMContext llvmContext_;
  mutable LLVM::TypeToLLVMIRTranslator translator_;
  mutable RecordStorageTypes types_;
  bool expandZeroInit_;
  mutable DenseMap<std::pair<cxx::ClassType, const DataLayout*>,
                   FailureOr<RecordStorageLayout>>
      records_;
};

[[nodiscard]] auto getBridgedMemRef(Value address) -> Value {
  auto from = address.getDefiningOp<ptr::FromPtrOp>();
  if (!from || from.getMetadata()) return {};
  auto to = from.getPtr().getDefiningOp<ptr::ToPtrOp>();
  if (!to || !isa<MemRefType>(to.getPtr().getType())) return {};
  return to.getPtr();
}

[[nodiscard]] auto getByteBuffer(Value address) -> Value {
  auto buffer = getBridgedMemRef(address);
  if (!buffer) return {};
  auto type = cast<MemRefType>(buffer.getType());
  if (type.getRank() != 1 || !type.hasStaticShape()) return {};
  if (!type.getElementType().isInteger(8)) return {};
  if (!type.getLayout().isIdentity()) return {};
  return buffer;
}

class ByteCopyToMemRef : public OpRewritePattern<cxx::MemCpyOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::MemCpyOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!op->getDiscardableAttrDictionary().empty()) return failure();
    auto source = getByteBuffer(op.getSrc());
    auto target = getByteBuffer(op.getDest());
    if (!source || !target || op.getSize() < 0) return failure();
    auto sourceType = cast<MemRefType>(source.getType());
    auto targetType = cast<MemRefType>(target.getType());
    if (sourceType.getMemorySpace() != targetType.getMemorySpace())
      return failure();
    auto count = op.getSize();
    if (count > sourceType.getDimSize(0) || count > targetType.getDimSize(0))
      return failure();
    auto src = memref::SubViewOp::create(
        rewriter, op.getLoc(), source, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{count}, ArrayRef<int64_t>{1});
    auto dst = memref::SubViewOp::create(
        rewriter, op.getLoc(), target, ArrayRef<int64_t>{0},
        ArrayRef<int64_t>{count}, ArrayRef<int64_t>{1});
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, src, dst);
    return success();
  }
};

class ByteCastToMemRef : public OpRewritePattern<cxx::BitcastOp> {
 public:
  ByteCastToMemRef(MLIRContext* context, const DataLayoutAnalysis& layouts)
      : OpRewritePattern(context), layouts_(layouts) {}

  auto matchAndRewrite(cxx::BitcastOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!op->getDiscardableAttrDictionary().empty()) return failure();
    auto buffer = getByteBuffer(op.getValue());
    auto pointer = dyn_cast<cxx::PointerType>(op.getType());
    if (!buffer || !pointer || pointer.getAddressSpace()) return failure();
    auto element = pointer.getElementType();
    if (!isSupportedArrayElement(element)) return failure();
    Type storage = getScalarStorageType(element);
    auto bytes = layouts_.getAtOrAbove(op).getTypeSize(storage).getFixedValue();
    auto bufferType = cast<MemRefType>(buffer.getType());
    if (bytes > uint64_t(bufferType.getDimSize(0))) return failure();
    auto viewType = MemRefType::get({}, storage, MemRefLayoutAttrInterface(),
                                    bufferType.getMemorySpace());
    auto zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto view = memref::ViewOp::create(rewriter, op.getLoc(), viewType, buffer,
                                       zero, ValueRange{});
    auto space = cast<ptr::MemorySpaceAttrInterface>(viewType.getMemorySpace());
    auto raw = ptr::ToPtrOp::create(rewriter, op.getLoc(),
                                    ptr::PtrType::get(space), view);
    rewriter.replaceOpWithNewOp<ptr::FromPtrOp>(op, op.getType(), raw, Value());
    return success();
  }

 private:
  const DataLayoutAnalysis& layouts_;
};

class BridgedLoadToMemRef : public OpRewritePattern<cxx::LoadOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::LoadOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto buffer = getBridgedMemRef(op.getAddr());
    if (!buffer) return failure();
    auto type = cast<MemRefType>(buffer.getType());
    if (type.getRank() != 0 || !type.getLayout().isIdentity()) return failure();
    if (!isSupportedElementAccess(op, op.getAddr(), op.getType()))
      return failure();
    auto storage = getScalarStorageType(op.getType());
    if (!areStorageValueTypesCompatible(storage, type.getElementType()))
      return failure();
    convertElementAccess(op, buffer, {}, rewriter);
    return success();
  }
};

class BridgedStoreToMemRef : public OpRewritePattern<cxx::StoreOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::StoreOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto buffer = getBridgedMemRef(op.getAddr());
    if (!buffer) return failure();
    auto type = cast<MemRefType>(buffer.getType());
    if (type.getRank() != 0 || !type.getLayout().isIdentity()) return failure();
    auto element = op.getValue().getType();
    if (!isSupportedElementAccess(op, op.getAddr(), element)) return failure();
    auto storage = getScalarStorageType(element);
    if (!areStorageValueTypesCompatible(storage, type.getElementType()))
      return failure();
    convertElementAccess(op, buffer, {}, rewriter);
    return success();
  }
};

constexpr uint64_t kScalarLeafLimit = 64;

[[nodiscard]] auto scalarElementCount(Type type) -> uint64_t {
  if (auto array = dyn_cast<cxx::ArrayType>(type))
    return array.getSize() * scalarElementCount(array.getElementType());
  if (auto record = dyn_cast<cxx::ClassType>(type)) {
    uint64_t count = 0;
    for (auto field : record.getBody()) count += scalarElementCount(field);
    return count;
  }
  return 1;
}

[[nodiscard]] auto isScalarizable(Type type) -> bool {
  return scalarElementCount(type) <= kScalarLeafLimit;
}

[[nodiscard]] auto pointeeOf(Value address) -> Type {
  return cast<cxx::PointerType>(address.getType()).getElementType();
}

[[nodiscard]] auto pointerTo(Type element, Value like) -> cxx::PointerType {
  auto pointer = cast<cxx::PointerType>(like.getType());
  return cxx::PointerType::get(element.getContext(), element,
                               pointer.getAddressSpace());
}

[[nodiscard]] auto isTypedStorageElement(Type type) -> bool {
  if (isa<cxx::PointerType, FloatType>(type)) return true;
  if (auto integer = dyn_cast<IntegerType>(type)) return integer.isSignless();
  auto vector = dyn_cast<VectorType>(type);
  if (!vector || vector.getRank() != 1 || vector.isScalable()) return false;
  return isa<FloatType, IntegerType>(vector.getElementType());
}

[[nodiscard]] auto zeroValue(OpBuilder& builder, Location loc, Type type)
    -> Value {
  if (isa<cxx::PointerType>(type))
    return cxx::ZeroOp::create(builder, loc, type);
  return arith::ConstantOp::create(builder, loc, type,
                                   cast<TypedAttr>(builder.getZeroAttr(type)));
}

[[nodiscard]] auto subobject(OpBuilder& builder, Location loc, Value address,
                             uint64_t index) -> Value {
  if (auto record = dyn_cast<cxx::ClassType>(pointeeOf(address))) {
    auto field = record.getBody()[index];
    return cxx::MemberOp::create(builder, loc, pointerTo(field, address),
                                 address, int32_t(index));
  }
  auto element = cast<cxx::ArrayType>(pointeeOf(address)).getElementType();
  auto position = arith::ConstantIndexOp::create(builder, loc, index);
  return cxx::SubscriptOp::create(builder, loc, pointerTo(element, address),
                                  address, position);
}

void forEachElement(OpBuilder& builder, Location loc, cxx::ArrayType array,
                    function_ref<void(OpBuilder&, Value)> body) {
  const auto count = array.getSize();
  if (isScalarizable(array)) {
    for (uint64_t index = 0; index < count; ++index)
      body(builder, arith::ConstantIndexOp::create(builder, loc, index));
    return;
  }
  auto lower = arith::ConstantIndexOp::create(builder, loc, 0);
  auto upper = arith::ConstantIndexOp::create(builder, loc, count);
  auto step = arith::ConstantIndexOp::create(builder, loc, 1);
  constexpr bool countsElements = true;
  scf::ForOp::create(
      builder, loc, lower, upper, step, ValueRange{},
      [&](OpBuilder& nested, Location, Value index, ValueRange) {
        body(nested, index);
        scf::YieldOp::create(nested, loc);
      },
      countsElements);
}

void expandObject(OpBuilder& builder, Location loc, Value destination,
                  Value source) {
  auto object = pointeeOf(destination);
  if (auto record = dyn_cast<cxx::ClassType>(object)) {
    for (auto [index, field] : llvm::enumerate(record.getBody())) {
      auto member = [&](Value base) -> Value {
        return cxx::MemberOp::create(builder, loc, pointerTo(field, base), base,
                                     int32_t(index));
      };
      expandObject(builder, loc, member(destination),
                   source ? member(source) : Value());
    }
    return;
  }
  if (auto array = dyn_cast<cxx::ArrayType>(object)) {
    auto element = array.getElementType();
    forEachElement(builder, loc, array, [&](OpBuilder& nested, Value index) {
      auto at = [&](Value base) -> Value {
        return cxx::SubscriptOp::create(nested, loc, pointerTo(element, base),
                                        base, index);
      };
      expandObject(nested, loc, at(destination), source ? at(source) : Value());
    });
    return;
  }
  auto value =
      source
          ? Value(cxx::LoadOp::create(builder, loc, object, source, int64_t(0)))
          : zeroValue(builder, loc, object);
  cxx::StoreOp::create(builder, loc, value, destination, int64_t(0));
}

[[nodiscard]] auto canExpandObject(Type type) -> bool {
  if (auto record = dyn_cast<cxx::ClassType>(type))
    return !record.getBody().empty() &&
           llvm::all_of(record.getBody(), canExpandObject);
  if (auto array = dyn_cast<cxx::ArrayType>(type))
    return canExpandObject(array.getElementType());
  return isTypedStorageElement(type);
}

class TypedZeroInit : public OpRewritePattern<cxx::ZeroInitOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::ZeroInitOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!canExpandObject(pointeeOf(op.getAddr()))) return failure();
    expandObject(rewriter, op.getLoc(), op.getAddr(), Value());
    rewriter.eraseOp(op);
    return success();
  }
};

class TypedCopy : public OpRewritePattern<cxx::CopyOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::CopyOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!canExpandObject(pointeeOf(op.getDest()))) return failure();
    expandObject(rewriter, op.getLoc(), op.getDest(), op.getSrc());
    rewriter.eraseOp(op);
    return success();
  }
};

[[nodiscard]] auto writesMemory(Operation* op) -> bool {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  return !effects || effects.hasEffect<MemoryEffects::Write>();
}

[[nodiscard]] auto hasNoWritesBetween(Operation* first, Operation* last)
    -> bool {
  if (first->getBlock() != last->getBlock()) return false;
  for (auto* op = first->getNextNode(); op && op != last;
       op = op->getNextNode())
    if (writesMemory(op)) return false;
  return true;
}

void loadElements(OpBuilder& builder, Location loc, Value address,
                  SmallVectorImpl<Value>& values) {
  auto object = pointeeOf(address);
  if (auto record = dyn_cast<cxx::ClassType>(object)) {
    for (uint64_t index = 0; index < record.getBody().size(); ++index)
      loadElements(builder, loc, subobject(builder, loc, address, index),
                   values);
    return;
  }
  if (auto array = dyn_cast<cxx::ArrayType>(object)) {
    for (uint64_t index = 0; index < array.getSize(); ++index)
      loadElements(builder, loc, subobject(builder, loc, address, index),
                   values);
    return;
  }
  values.push_back(
      cxx::LoadOp::create(builder, loc, object, address, int64_t(0)));
}

void storeElements(OpBuilder& builder, Location loc, Value address,
                   ArrayRef<Value>& values) {
  auto object = pointeeOf(address);
  if (auto record = dyn_cast<cxx::ClassType>(object)) {
    for (uint64_t index = 0; index < record.getBody().size(); ++index)
      storeElements(builder, loc, subobject(builder, loc, address, index),
                    values);
    return;
  }
  if (auto array = dyn_cast<cxx::ArrayType>(object)) {
    for (uint64_t index = 0; index < array.getSize(); ++index)
      storeElements(builder, loc, subobject(builder, loc, address, index),
                    values);
    return;
  }
  cxx::StoreOp::create(builder, loc, values.front(), address, int64_t(0));
  values = values.drop_front();
}

class AggregateValueCopy : public OpRewritePattern<cxx::StoreOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::StoreOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto object = op.getValue().getType();
    if (!cxx::isAggregateType(object) || pointeeOf(op.getAddr()) != object)
      return failure();
    if (!canExpandObject(object)) return failure();
    if (op.getVolatile_()) return failure();
    auto value = op.getValue().getDefiningOp();
    if (auto load = dyn_cast_or_null<cxx::LoadOp>(value)) {
      if (pointeeOf(load.getAddr()) != object) return failure();
      if (load.getVolatile_()) return failure();
      if (hasNoWritesBetween(load, op)) {
        expandObject(rewriter, op.getLoc(), op.getAddr(), load.getAddr());
      } else {
        if (!isScalarizable(object)) return failure();
        SmallVector<Value> elements;
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointAfter(load);
          loadElements(rewriter, load.getLoc(), load.getAddr(), elements);
        }
        ArrayRef<Value> remaining(elements);
        storeElements(rewriter, op.getLoc(), op.getAddr(), remaining);
      }
    } else if (isa_and_nonnull<cxx::ZeroOp>(value)) {
      expandObject(rewriter, op.getLoc(), op.getAddr(), Value());
    } else if (!isa_and_nonnull<cxx::UndefOp>(value)) {
      return failure();
    }
    rewriter.eraseOp(op);
    if (value && value->use_empty()) rewriter.eraseOp(value);
    return success();
  }
};

class AggregateValueExtract : public OpRewritePattern<cxx::ExtractValueOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::ExtractValueOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto load = op.getContainer().getDefiningOp<cxx::LoadOp>();
    if (!load || pointeeOf(load.getAddr()) != load.getType()) return failure();
    if (!cxx::isAggregateType(load.getType())) return failure();
    if (load.getVolatile_()) return failure();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(load);
    auto element = subobject(rewriter, op.getLoc(), load.getAddr(),
                             uint64_t(op.getPosition()));
    if (pointeeOf(element) != op.getType()) return failure();
    rewriter.replaceOpWithNewOp<cxx::LoadOp>(op, op.getType(), element,
                                             int64_t(0));
    if (load->use_empty()) rewriter.eraseOp(load);
    return success();
  }
};

[[nodiscard]] auto parentSuccessor(Operation* op) -> RegionSuccessor {
#if LLVM_VERSION_MAJOR < 23
  return RegionSuccessor(op, op->getResults());
#else
  return RegionSuccessor(op);
#endif
}

[[nodiscard]] auto forwardedOperand(MutableOperandRange operands,
                                    unsigned inputCount, unsigned position)
    -> OpOperand* {
  auto leading = inputCount - operands.size();
  if (position < leading || position - leading >= operands.size())
    return nullptr;
  return &operands[position - leading];
}

[[nodiscard]] auto regionBranchOperands(RegionBranchOpInterface branch,
                                        RegionSuccessor successor,
                                        unsigned inputCount, unsigned position,
                                        SmallVectorImpl<OpOperand*>& operands)
    -> bool {
  SmallVector<RegionBranchPoint> predecessors;
  branch.getPredecessors(successor, predecessors);
  for (auto point : predecessors) {
    OpOperand* operand = nullptr;
    if (point.isParent()) {
      auto entry = branch.getEntrySuccessorOperands(successor);
      MutableOperandRange range(branch, entry.getBeginOperandIndex(),
                                entry.size());
      operand = forwardedOperand(range, inputCount, position);
    } else {
      auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
          static_cast<Operation*>(point.getTerminatorPredecessorOrNull()));
      if (!terminator) return false;
      operand =
          forwardedOperand(terminator.getMutableSuccessorOperands(successor),
                           inputCount, position);
    }
    if (!operand) return false;
    operands.push_back(operand);
  }
  return !operands.empty();
}

[[nodiscard]] auto incomingOperands(Value value,
                                    SmallVectorImpl<OpOperand*>& operands)
    -> bool {
  if (auto result = dyn_cast<OpResult>(value)) {
    auto branch = dyn_cast<RegionBranchOpInterface>(result.getOwner());
    if (!branch) return false;
    return regionBranchOperands(branch, parentSuccessor(branch),
                                branch->getNumResults(),
                                result.getResultNumber(), operands);
  }
  auto argument = cast<BlockArgument>(value);
  auto* block = argument.getOwner();
  if (block->isEntryBlock()) {
    auto branch =
        dyn_cast_or_null<RegionBranchOpInterface>(block->getParentOp());
    if (!branch) return false;
    return regionBranchOperands(branch, RegionSuccessor(block->getParent()),
                                block->getNumArguments(),
                                argument.getArgNumber(), operands);
  }
  for (auto it = block->pred_begin(); it != block->pred_end(); ++it) {
    auto branch = dyn_cast<BranchOpInterface>((*it)->getTerminator());
    if (!branch) return false;
    auto successor = branch.getSuccessorOperands(it.getSuccessorIndex());
    auto produced = successor.getProducedOperandCount();
    if (argument.getArgNumber() < produced) return false;
    operands.push_back(
        &successor.getMutableForwardedOperands()[argument.getArgNumber() -
                                                 produced]);
  }
  return !operands.empty();
}

[[nodiscard]] auto isSpeculatableAddress(Value address,
                                         DenseSet<Value>& visiting) -> bool;

[[nodiscard]] auto isSpeculatableIncoming(Value carried,
                                          DenseSet<Value>& visiting) -> bool {
  SmallVector<OpOperand*> operands;
  if (!incomingOperands(carried, operands)) return false;
  if (!visiting.insert(carried).second) return true;
  for (auto* operand : operands)
    if (!isSpeculatableAddress(operand->get(), visiting)) return false;
  return true;
}

[[nodiscard]] auto isSpeculatableAddress(Value address,
                                         DenseSet<Value>& visiting) -> bool {
  while (auto* op = address.getDefiningOp()) {
    if (isa<cxx::AllocaOp, cxx::AddressOfOp>(op)) return true;
    if (auto member = dyn_cast<cxx::MemberOp>(op)) {
      address = member.getBase();
      continue;
    }
    if (auto decay = dyn_cast<cxx::ArrayToPointerOp>(op)) {
      address = decay.getValue();
      continue;
    }
    if (auto subscript = dyn_cast<cxx::SubscriptOp>(op)) {
      auto array = dyn_cast<cxx::ArrayType>(pointeeOf(subscript.getBase()));
      auto index = getConstantIntValue(subscript.getIndex());
      if (!array || !index || *index < 0 || uint64_t(*index) >= array.getSize())
        return false;
      address = subscript.getBase();
      continue;
    }
    if (auto select = dyn_cast<arith::SelectOp>(op))
      return isSpeculatableAddress(select.getTrueValue(), visiting) &&
             isSpeculatableAddress(select.getFalseValue(), visiting);
    if (isa<RegionBranchOpInterface>(op))
      return isSpeculatableIncoming(address, visiting);
    return false;
  }
  return isSpeculatableIncoming(address, visiting);
}

[[nodiscard]] auto isSpeculatableAddress(Value address) -> bool {
  DenseSet<Value> visiting;
  return isSpeculatableAddress(address, visiting);
}

class AddressThroughSelect : public RewritePattern {
 public:
  explicit AddressThroughSelect(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  auto matchAndRewrite(Operation* op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!isa<cxx::MemberOp, cxx::SubscriptOp, cxx::ArrayToPointerOp,
             cxx::PtrAddOp>(op))
      return failure();
    auto select = op->getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!select) return failure();
    auto derive = [&](Value base) -> Value {
      IRMapping mapping;
      mapping.map(op->getOperand(0), base);
      return rewriter.clone(*op, mapping)->getResult(0);
    };
    auto lhs = derive(select.getTrueValue());
    auto rhs = derive(select.getFalseValue());
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, select.getCondition(), lhs,
                                                 rhs);
    return success();
  }
};

class LoadThroughSelect : public OpRewritePattern<cxx::LoadOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::LoadOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (op.getVolatile_()) return failure();
    auto select = op.getAddr().getDefiningOp<arith::SelectOp>();
    if (!select) return failure();
    auto load = [&](OpBuilder& builder, Value address) -> Value {
      return cxx::LoadOp::create(builder, op.getLoc(), op.getType(), address,
                                 op.getAlignment());
    };
    if (isSpeculatableAddress(select.getTrueValue()) &&
        isSpeculatableAddress(select.getFalseValue())) {
      auto lhs = load(rewriter, select.getTrueValue());
      auto rhs = load(rewriter, select.getFalseValue());
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, select.getCondition(),
                                                   lhs, rhs);
      return success();
    }
    auto branch = scf::IfOp::create(
        rewriter, op.getLoc(), select.getCondition(),
        [&](OpBuilder& builder, Location loc) {
          scf::YieldOp::create(builder, loc,
                               load(builder, select.getTrueValue()));
        },
        [&](OpBuilder& builder, Location loc) {
          scf::YieldOp::create(builder, loc,
                               load(builder, select.getFalseValue()));
        });
    rewriter.replaceOp(op, branch.getResults());
    return success();
  }
};

class StoreThroughSelect : public OpRewritePattern<cxx::StoreOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::StoreOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (op.getVolatile_()) return failure();
    auto select = op.getAddr().getDefiningOp<arith::SelectOp>();
    if (!select) return failure();
    auto value = op.getValue();
    auto loc = op.getLoc();
    auto condition = select.getCondition();
    auto lhs = select.getTrueValue();
    auto rhs = select.getFalseValue();
    if (isSpeculatableAddress(lhs) && isSpeculatableAddress(rhs)) {
      auto keep = [&](Value address, bool selected) {
        Value old = cxx::LoadOp::create(rewriter, loc, value.getType(), address,
                                        op.getAlignment());
        Value updated =
            selected
                ? arith::SelectOp::create(rewriter, loc, condition, value, old)
                : arith::SelectOp::create(rewriter, loc, condition, old, value);
        cxx::StoreOp::create(rewriter, loc, updated, address,
                             op.getAlignment());
      };
      keep(lhs, true);
      keep(rhs, false);
      rewriter.eraseOp(op);
      return success();
    }
    scf::IfOp::create(
        rewriter, loc, condition,
        [&](OpBuilder& builder, Location location) {
          cxx::StoreOp::create(builder, location, value, lhs,
                               op.getAlignment());
          scf::YieldOp::create(builder, location);
        },
        [&](OpBuilder& builder, Location location) {
          cxx::StoreOp::create(builder, location, value, rhs,
                               op.getAlignment());
          scf::YieldOp::create(builder, location);
        });
    rewriter.eraseOp(op);
    return success();
  }
};

struct ElementOffset {
  Value base;
  SmallVector<Value> terms;
};

[[nodiscard]] auto isArrayElementPointer(Value base, Value element) -> bool {
  auto array = dyn_cast<cxx::ArrayType>(pointeeOf(base));
  return array && array.getElementType() == pointeeOf(element);
}

[[nodiscard]] auto elementOffsetOf(Value address) -> ElementOffset {
  ElementOffset result{address, {}};
  while (auto* op = result.base.getDefiningOp()) {
    if (auto add = dyn_cast<cxx::PtrAddOp>(op)) {
      if (pointeeOf(add.getBase()) == pointeeOf(address)) {
        result.terms.push_back(add.getOffset());
        result.base = add.getBase();
        continue;
      }
      if (!isArrayElementPointer(add.getBase(), address)) break;
      result.terms.push_back(add.getOffset());
      result.base = add.getBase();
    } else if (auto subscript = dyn_cast<cxx::SubscriptOp>(op)) {
      if (!isArrayElementPointer(subscript.getBase(), address)) break;
      result.terms.push_back(subscript.getIndex());
      result.base = subscript.getBase();
    } else if (auto decay = dyn_cast<cxx::ArrayToPointerOp>(op)) {
      if (!isArrayElementPointer(decay.getValue(), address)) break;
      result.base = decay.getValue();
    }
    break;
  }
  return result;
}

[[nodiscard]] auto sumTerms(OpBuilder& builder, Location loc,
                            ArrayRef<Value> terms, Type type) -> Value {
  Value sum = arith::ConstantOp::create(builder, loc, type,
                                        builder.getIntegerAttr(type, 0));
  for (auto term : terms) {
    if (isa<IndexType>(type) || isa<IndexType>(term.getType())) {
      term = arith::IndexCastOp::create(builder, loc, type, term);
    } else {
      auto width = cast<IntegerType>(term.getType()).getWidth();
      auto target = cast<IntegerType>(type).getWidth();
      if (width < target)
        term = arith::ExtSIOp::create(builder, loc, type, term);
      else if (width > target)
        term = arith::TruncIOp::create(builder, loc, type, term);
    }
    sum = arith::AddIOp::create(builder, loc, sum, term);
  }
  return sum;
}

class PointerComparisonToIndex : public OpRewritePattern<arith::CmpIOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(arith::CmpIOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto lhs = op.getLhs().getDefiningOp<cxx::PtrToIntOp>();
    auto rhs = op.getRhs().getDefiningOp<cxx::PtrToIntOp>();
    if (!lhs || !rhs) return failure();
    auto left = elementOffsetOf(lhs.getValue());
    auto right = elementOffsetOf(rhs.getValue());
    if (left.base != right.base) return failure();
    auto type = rewriter.getIndexType();
    auto a = sumTerms(rewriter, op.getLoc(), left.terms, type);
    auto b = sumTerms(rewriter, op.getLoc(), right.terms, type);
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, signedPredicate(op.getPredicate()), a, b);
    return success();
  }

 private:
  [[nodiscard]] static auto signedPredicate(arith::CmpIPredicate predicate)
      -> arith::CmpIPredicate {
    switch (predicate) {
      case arith::CmpIPredicate::ult:
        return arith::CmpIPredicate::slt;
      case arith::CmpIPredicate::ule:
        return arith::CmpIPredicate::sle;
      case arith::CmpIPredicate::ugt:
        return arith::CmpIPredicate::sgt;
      case arith::CmpIPredicate::uge:
        return arith::CmpIPredicate::sge;
      default:
        return predicate;
    }
  }
};

class PointerDifferenceToIndex : public OpRewritePattern<cxx::PtrDiffOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::PtrDiffOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto left = elementOffsetOf(op.getLhs());
    auto right = elementOffsetOf(op.getRhs());
    if (left.base != right.base) return failure();
    auto type = op.getType();
    auto a = sumTerms(rewriter, op.getLoc(), left.terms, type);
    auto b = sumTerms(rewriter, op.getLoc(), right.terms, type);
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, a, b);
    return success();
  }
};

[[nodiscard]] auto insertionAnchor(Value value) -> Operation* {
  if (auto result = dyn_cast<OpResult>(value)) return result.getOwner();
  auto* block = cast<BlockArgument>(value).getOwner();
  if (block->isEntryBlock()) return block->getParentOp();
  return &block->front();
}

void collectOffsetRoots(Value value, SmallVectorImpl<Value>& roots) {
  auto offset = elementOffsetOf(value);
  if (offset.base != value) {
    if (isArrayElementPointer(offset.base, value))
      roots.push_back(offset.base);
    else
      collectOffsetRoots(offset.base, roots);
    return;
  }
  if (auto select = value.getDefiningOp<arith::SelectOp>();
      select && isa<cxx::PointerType>(select.getType())) {
    collectOffsetRoots(select.getTrueValue(), roots);
    collectOffsetRoots(select.getFalseValue(), roots);
    return;
  }
  if (value.getDefiningOp<cxx::UndefOp>()) return;
  roots.push_back(value);
}

[[nodiscard]] auto materializeOffset(OpBuilder& builder, Location loc,
                                     Value value, Value base,
                                     const DenseSet<Value>& members,
                                     Type offsetType) -> Value {
  if (members.contains(value)) return value;
  if (value == base) return sumTerms(builder, loc, {}, offsetType);
  auto address = elementOffsetOf(value);
  if (address.base != value) {
    auto offset = materializeOffset(builder, loc, address.base, base, members,
                                    offsetType);
    address.terms.push_back(offset);
    return sumTerms(builder, loc, address.terms, offsetType);
  }
  if (auto select = value.getDefiningOp<arith::SelectOp>();
      select && isa<cxx::PointerType>(select.getType())) {
    auto lhs = materializeOffset(builder, loc, select.getTrueValue(), base,
                                 members, offsetType);
    auto rhs = materializeOffset(builder, loc, select.getFalseValue(), base,
                                 members, offsetType);
    return arith::SelectOp::create(builder, loc, select.getCondition(), lhs,
                                   rhs);
  }
  return sumTerms(builder, loc, {}, offsetType);
}

struct CarriedPointers {
  SmallVector<Value> members;
  llvm::SetVector<OpOperand*> edges;
  Value base;
};

[[nodiscard]] auto carriedPointerCandidates(FunctionOpInterface function)
    -> SmallVector<Value> {
  SmallVector<Value> candidates;
  function.walk([&](Operation* op) {
    if (isa<RegionBranchOpInterface>(op))
      for (auto result : op->getResults())
        if (isa<cxx::PointerType>(result.getType()))
          candidates.push_back(result);
    for (auto& region : op->getRegions())
      for (auto& block : region)
        for (auto argument : block.getArguments())
          if (isa<cxx::PointerType>(argument.getType()) &&
              !(block.isEntryBlock() && op == function.getOperation()))
            candidates.push_back(argument);
  });
  return candidates;
}

[[nodiscard]] auto groupCarriedPointers(ArrayRef<Value> candidates)
    -> SmallVector<CarriedPointers> {
  DenseMap<Value, SmallVector<OpOperand*>> incoming;
  DenseSet<Value> unsupported;
  llvm::EquivalenceClasses<Value> classes;
  for (auto candidate : candidates) {
    classes.insert(candidate);
    if (!incomingOperands(candidate, incoming[candidate]))
      unsupported.insert(candidate);
  }
  for (auto& [candidate, edges] : incoming) {
    for (auto* edge : edges) {
      SmallVector<Value> roots;
      collectOffsetRoots(edge->get(), roots);
      for (auto root : roots)
        if (classes.contains(root)) classes.unionSets(candidate, root);
    }
  }

  SmallVector<CarriedPointers> groups;
  for (auto it = classes.begin(); it != classes.end(); ++it) {
    if (!(*it)->isLeader()) continue;
    CarriedPointers group;
    for (auto member = classes.member_begin(**it);
         member != classes.member_end(); ++member)
      group.members.push_back(*member);
    DenseSet<Value> members(group.members.begin(), group.members.end());
    bool valid = true;
    for (auto member : group.members) {
      if (unsupported.contains(member)) valid = false;
      for (auto* edge : incoming.lookup(member)) {
        if (!group.edges.insert(edge)) continue;
        SmallVector<Value> roots;
        collectOffsetRoots(edge->get(), roots);
        for (auto root : roots) {
          if (members.contains(root)) continue;
          if (pointeeOf(root) != pointeeOf(member) &&
              !isArrayElementPointer(root, member))
            valid = false;
          if (group.base && group.base != root) valid = false;
          group.base = root;
        }
      }
    }
    if (valid && group.base) groups.push_back(std::move(group));
  }
  return groups;
}

[[nodiscard]] auto dominatesCarriedPointers(const CarriedPointers& carried,
                                            DominanceInfo& dominance) -> bool {
  for (auto member : carried.members) {
    auto* anchor = insertionAnchor(member);
    if (auto* defining = carried.base.getDefiningOp()) {
      if (!dominance.properlyDominates(defining, anchor)) return false;
      continue;
    }
    auto* owner = cast<BlockArgument>(carried.base).getOwner();
    if (!dominance.dominates(owner, anchor->getBlock())) return false;
  }
  return true;
}

void rewriteCarriedPointers(const CarriedPointers& carried) {
  auto context = carried.base.getContext();
  auto offsetType = IntegerType::get(context, 32);
  DenseSet<Value> members(carried.members.begin(), carried.members.end());
  SmallVector<Type> pointerTypes;
  for (auto member : carried.members) {
    pointerTypes.push_back(member.getType());
    member.setType(offsetType);
  }
  OpBuilder builder(context);
  for (auto [index, member] : llvm::enumerate(carried.members)) {
    if (auto argument = dyn_cast<BlockArgument>(member))
      builder.setInsertionPointToStart(argument.getOwner());
    else
      builder.setInsertionPointAfter(cast<OpResult>(member).getOwner());
    Value value = member;
    auto rebuilt = cxx::PtrAddOp::create(
        builder, value.getLoc(), pointerTypes[index], carried.base, value);
    value.replaceAllUsesExcept(rebuilt.getResult(), rebuilt);
  }
  for (auto* edge : carried.edges) {
    builder.setInsertionPoint(edge->getOwner());
    edge->set(materializeOffset(builder, edge->getOwner()->getLoc(),
                                edge->get(), carried.base, members,
                                offsetType));
  }
}

void rewritePointerInductions(FunctionOpInterface function) {
  if (function.isExternal()) return;
  auto groups = groupCarriedPointers(carriedPointerCandidates(function));
  DominanceInfo dominance(function);
  for (const auto& group : groups)
    if (dominatesCarriedPointers(group, dominance))
      rewriteCarriedPointers(group);
}

[[nodiscard]] auto loadsBeforeWrites(Value carried,
                                     SmallVectorImpl<cxx::LoadOp>& loads)
    -> bool {
  auto argument = dyn_cast<BlockArgument>(carried);
  auto* block = argument ? argument.getOwner()
                         : cast<OpResult>(carried).getOwner()->getBlock();
  auto* start = argument ? &block->front()
                         : cast<OpResult>(carried).getOwner()->getNextNode();
  for (auto* user : carried.getUsers()) {
    auto load = dyn_cast<cxx::LoadOp>(user);
    if (!load || load->getBlock() != block) return false;
    if (load.getVolatile_()) return false;
    if (load.getType() != pointeeOf(carried)) return false;
    loads.push_back(load);
  }
  if (loads.empty()) return false;
  for (auto load : loads)
    for (auto* op = start; op && op != load; op = op->getNextNode())
      if (writesMemory(op)) return false;
  return true;
}

[[nodiscard]] auto speculatePointerPhi(Value carried) -> bool {
  SmallVector<cxx::LoadOp> loads;
  if (!loadsBeforeWrites(carried, loads)) return false;
  if (!isSpeculatableAddress(carried)) return false;
  SmallVector<OpOperand*> operands;
  if (!incomingOperands(carried, operands)) return false;
  auto alignment = loads.front().getAlignment();
  for (auto load : loads) alignment = std::min(alignment, load.getAlignment());
  auto valueType = pointeeOf(carried);
  OpBuilder builder(carried.getContext());
  for (auto* operand : operands) {
    builder.setInsertionPoint(operand->getOwner());
    operand->set(cxx::LoadOp::create(builder, loads.front().getLoc(), valueType,
                                     operand->get(), alignment));
  }
  carried.setType(valueType);
  for (auto load : loads) {
    load.getResult().replaceAllUsesWith(carried);
    load->erase();
  }
  return true;
}

void speculatePointerPhis(FunctionOpInterface function) {
  if (function.isExternal()) return;
  for (auto changed = true; changed;) {
    changed = false;
    for (auto carried : carriedPointerCandidates(function))
      if (isa<cxx::PointerType>(carried.getType()))
        changed |= speculatePointerPhi(carried);
  }
}

[[nodiscard]] auto aggregateElementCount(Type aggregate) -> uint64_t {
  if (auto record = dyn_cast<cxx::ClassType>(aggregate))
    return record.getBody().size();
  return cast<cxx::ArrayType>(aggregate).getSize();
}

[[nodiscard]] auto aggregateElementType(Type aggregate, uint64_t index)
    -> Type {
  if (auto record = dyn_cast<cxx::ClassType>(aggregate))
    return record.getBody()[index];
  return cast<cxx::ArrayType>(aggregate).getElementType();
}

[[nodiscard]] auto canStoreConstant(Type object, Attribute value) -> bool {
  if (isa<LLVM::ZeroAttr>(value)) return canExpandObject(object);
  if (cxx::isAggregateType(object)) {
    auto elements = dyn_cast<ArrayAttr>(value);
    if (!elements || elements.size() != aggregateElementCount(object))
      return false;
    for (auto [index, element] : llvm::enumerate(elements))
      if (!canStoreConstant(aggregateElementType(object, index), element))
        return false;
    return true;
  }
  auto typed = dyn_cast<TypedAttr>(value);
  return typed && typed.getType() == object && isTypedStorageElement(object) &&
         !isa<cxx::PointerType>(object);
}

void storeConstant(OpBuilder& builder, Location loc, Value address,
                   Attribute value) {
  if (isa<LLVM::ZeroAttr>(value)) {
    expandObject(builder, loc, address, Value());
    return;
  }
  if (auto elements = dyn_cast<ArrayAttr>(value)) {
    for (auto [index, element] : llvm::enumerate(elements))
      storeConstant(builder, loc, subobject(builder, loc, address, index),
                    element);
    return;
  }
  auto object = pointeeOf(address);
  auto constant =
      arith::ConstantOp::create(builder, loc, object, cast<TypedAttr>(value));
  cxx::StoreOp::create(builder, loc, constant, address, int64_t(0));
}

void materializeConstantGlobals(FunctionOpInterface function,
                                SymbolTableCollection& symbols) {
  if (function.isExternal()) return;
  DenseMap<Attribute, Value> copies;
  SmallVector<cxx::AddressOfOp> uses;
  function.walk([&](cxx::AddressOfOp op) { uses.push_back(op); });
  auto& entry = function.getFunctionBody().front();
  for (auto op : uses) {
    auto global =
        symbols.lookupNearestSymbolFrom<cxx::GlobalOp>(op, op.getSymNameAttr());
    if (!global || !global.getConstant() || !global.getValue() ||
        !global.getInitializer().empty())
      continue;
    auto type = dyn_cast<cxx::PointerType>(op.getType());
    if (!type || type.getElementType() != global.getGlobalType()) continue;
    if (!canStoreConstant(global.getGlobalType(), *global.getValue())) continue;
    auto found = copies.find(op.getSymNameAttr());
    if (found == copies.end()) {
      OpBuilder builder(&entry, entry.begin());
      auto local = cxx::AllocaOp::create(
          builder, op.getLoc(),
          cxx::PointerType::get(op.getContext(), global.getGlobalType()),
          int64_t(0));
      storeConstant(builder, op.getLoc(), local, *global.getValue());
      found = copies.try_emplace(op.getSymNameAttr(), local).first;
    }
    op.getResult().replaceAllUsesWith(found->second);
    op->erase();
  }
}

struct StorageLeaf {
  SmallVector<int64_t> shape;
  Type element;
};

[[nodiscard]] auto collectStorageLeaves(Type type,
                                        SmallVectorImpl<int64_t>& shape,
                                        SmallVectorImpl<StorageLeaf>& leaves)
    -> bool {
  if (isTypedStorageElement(type)) {
    if (isa<cxx::PointerType>(type) && !shape.empty()) return false;
    leaves.push_back({SmallVector<int64_t>(shape.begin(), shape.end()), type});
    return leaves.size() <= kScalarLeafLimit;
  }
  if (auto array = dyn_cast<cxx::ArrayType>(type)) {
    auto size = array.getSize();
    if (!size || size > uint64_t(std::numeric_limits<int32_t>::max()))
      return false;
    shape.push_back(int64_t(size));
    auto collected =
        collectStorageLeaves(array.getElementType(), shape, leaves);
    shape.pop_back();
    return collected;
  }
  if (auto record = dyn_cast<cxx::ClassType>(type)) {
    if (record.getBody().empty()) return false;
    for (auto field : record.getBody())
      if (!collectStorageLeaves(field, shape, leaves)) return false;
    return true;
  }
  return false;
}

[[nodiscard]] auto countStorageLeaves(Type type) -> unsigned {
  if (auto array = dyn_cast<cxx::ArrayType>(type))
    return countStorageLeaves(array.getElementType());
  if (auto record = dyn_cast<cxx::ClassType>(type)) {
    unsigned count = 0;
    for (auto field : record.getBody()) count += countStorageLeaves(field);
    return count;
  }
  return 1;
}

[[nodiscard]] auto canReinterpret(Type from, Type to) -> bool {
  if (from == to) return true;
  if (isa<cxx::PointerType>(from) || isa<cxx::PointerType>(to))
    return areStorageValueTypesCompatible(from, to);
  auto fromVector = dyn_cast<VectorType>(from);
  auto toVector = dyn_cast<VectorType>(to);
  if (bool(fromVector) != bool(toVector)) return false;
  if (fromVector && fromVector.getShape() != toVector.getShape()) return false;
  auto lane = [](Type type) {
    if (auto vector = dyn_cast<VectorType>(type))
      return vector.getElementType();
    return type;
  };
  auto fromLane = lane(from);
  auto toLane = lane(to);
  return fromLane.isIntOrFloat() && toLane.isIntOrFloat() &&
         fromLane.getIntOrFloatBitWidth() == toLane.getIntOrFloatBitWidth();
}

struct TypedAddress {
  unsigned leaf = 0;
  Type object;
  SmallVector<SmallVector<Value>> indices;
  bool element = false;
};

struct TypedAccess {
  Operation* op;
  unsigned leaf;
  unsigned zeros;
};

class TypedStorageAnalysis {
 public:
  explicit TypedStorageAnalysis(ArrayRef<StorageLeaf> leaves)
      : leaves_(leaves) {}

  [[nodiscard]] auto analyze(Value root) -> LogicalResult {
    addresses_[root] = TypedAddress{0, pointeeOf(root), {}, false};
    SmallVector<Value> worklist{root};
    while (!worklist.empty()) {
      auto address = worklist.pop_back_val();
      auto current = addresses_.find(address)->second;
      for (auto& use : address.getUses()) {
        auto* user = use.getOwner();
        if (failed(visit(user, use, current, worklist))) return failure();
      }
    }
    return success();
  }

  [[nodiscard]] auto address(Value value) const -> const TypedAddress& {
    return addresses_.find(value)->second;
  }

  [[nodiscard]] auto derivations() const -> ArrayRef<Operation*> {
    return derivations_;
  }

  [[nodiscard]] auto accesses() const -> ArrayRef<TypedAccess> {
    return accesses_;
  }

 private:
  [[nodiscard]] auto visit(Operation* user, OpOperand& use,
                           const TypedAddress& current,
                           SmallVectorImpl<Value>& worklist) -> LogicalResult {
    if (cxx::isVolatileAccess(user)) return failure();
    if (auto load = dyn_cast<cxx::LoadOp>(user))
      return access(user, current, load.getType());
    if (auto store = dyn_cast<cxx::StoreOp>(user)) {
      if (use.getOperandNumber() != 1) return failure();
      return access(user, current, store.getValue().getType());
    }
    auto derived = derive(user, current);
    if (failed(derived)) return failure();
    derivations_.push_back(user);
    addresses_[user->getResult(0)] = std::move(*derived);
    worklist.push_back(user->getResult(0));
    return success();
  }

  [[nodiscard]] auto derive(Operation* user, const TypedAddress& base)
      -> FailureOr<TypedAddress> {
    auto result = base;
    if (auto member = dyn_cast<cxx::MemberOp>(user)) {
      auto record = dyn_cast<cxx::ClassType>(base.object);
      if (!record || pointeeOf(member.getBase()) != base.object)
        return failure();
      auto index = unsigned(member.getMemberIndex());
      if (index >= record.getBody().size()) return failure();
      for (auto field : record.getBody().take_front(index))
        result.leaf += countStorageLeaves(field);
      result.object = record.getBody()[index];
      result.element = false;
      return result;
    }
    if (auto subscript = dyn_cast<cxx::SubscriptOp>(user)) {
      auto array = dyn_cast<cxx::ArrayType>(base.object);
      if (!array || pointeeOf(subscript.getBase()) != base.object ||
          !isSupportedArrayIndex(subscript.getIndex()))
        return failure();
      result.object = array.getElementType();
      result.indices.push_back({subscript.getIndex()});
      result.element = true;
      return result;
    }
    if (auto decay = dyn_cast<cxx::ArrayToPointerOp>(user)) {
      auto array = dyn_cast<cxx::ArrayType>(base.object);
      if (!array || pointeeOf(decay.getValue()) != base.object)
        return failure();
      result.object = array.getElementType();
      result.indices.push_back({});
      result.element = true;
      return result;
    }
    if (auto add = dyn_cast<cxx::PtrAddOp>(user)) {
      if (!isSupportedArrayIndex(add.getOffset())) return failure();
      auto target = pointeeOf(add.getResult());
      if (base.element && canReinterpret(base.object, target)) {
        result.indices.back().push_back(add.getOffset());
        return result;
      }
      auto array = dyn_cast<cxx::ArrayType>(base.object);
      if (array && pointeeOf(add.getBase()) == base.object &&
          canReinterpret(array.getElementType(), target)) {
        result.object = array.getElementType();
        result.indices.push_back({add.getOffset()});
        result.element = true;
        return result;
      }
      if (!matchPattern(add.getOffset(), m_Zero())) return failure();
      return result;
    }
    if (isa<cxx::BitcastOp, cxx::AddressSpaceCastOp>(user)) {
      if (!isa<cxx::PointerType>(user->getResult(0).getType()))
        return failure();
      return result;
    }
    return failure();
  }

  [[nodiscard]] auto access(Operation* user, const TypedAddress& address,
                            Type type) -> LogicalResult {
    auto object = address.object;
    unsigned zeros = 0;
    while (cxx::isAggregateType(object) && !canReinterpret(object, type)) {
      if (auto array = dyn_cast<cxx::ArrayType>(object)) {
        object = array.getElementType();
        ++zeros;
        continue;
      }
      auto record = cast<cxx::ClassType>(object);
      if (record.getBody().empty()) return failure();
      object = record.getBody().front();
    }
    if (!isTypedStorageElement(object) || !canReinterpret(object, type))
      return failure();
    if (address.indices.size() + zeros != leaves_[address.leaf].shape.size())
      return failure();
    accesses_.push_back({user, address.leaf, zeros});
    return success();
  }

  ArrayRef<StorageLeaf> leaves_;
  DenseMap<Value, TypedAddress> addresses_;
  SmallVector<Operation*> derivations_;
  SmallVector<TypedAccess> accesses_;
};

[[nodiscard]] auto toIndex(OpBuilder& builder, Location loc,
                           ArrayRef<Value> terms) -> Value {
  Value sum;
  for (auto term : terms) {
    if (!isa<IndexType>(term.getType()))
      term = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                        term);
    sum = sum ? Value(arith::AddIOp::create(builder, loc, sum, term)) : term;
  }
  return sum ? sum : arith::ConstantIndexOp::create(builder, loc, 0);
}

[[nodiscard]] auto reinterpret(OpBuilder& builder, Location loc, Value value,
                               Type type) -> Value {
  if (value.getType() == type) return value;
  if (isa<cxx::PointerType>(type))
    return cxx::BitcastOp::create(builder, loc, type, value);
  return arith::BitcastOp::create(builder, loc, type, value);
}

class TypedStorageToMemRef : public OpRewritePattern<cxx::AllocaOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(cxx::AllocaOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!isSupportedLocalAllocation(op)) return failure();
    auto object = pointeeOf(op.getResult());
    if (isa<cxx::PointerType>(object)) return failure();
    SmallVector<int64_t> shape;
    SmallVector<StorageLeaf> leaves;
    if (!collectStorageLeaves(object, shape, leaves)) return failure();
    TypedStorageAnalysis analysis(leaves);
    if (failed(analysis.analyze(op.getResult()))) return failure();
    auto function = op->getParentOfType<FunctionOpInterface>();
    SmallVector<Value> storage;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&function.getFunctionBody().front());
      for (const auto& leaf : leaves) {
        if (isa<cxx::PointerType>(leaf.element)) {
          storage.push_back(cxx::AllocaOp::create(
              rewriter, op.getLoc(),
              cxx::PointerType::get(op.getContext(), leaf.element),
              int64_t(0)));
          continue;
        }
        storage.push_back(memref::AllocaOp::create(
            rewriter, op.getLoc(), MemRefType::get(leaf.shape, leaf.element)));
      }
    }
    for (const auto& access : analysis.accesses()) {
      auto* user = access.op;
      rewriter.setInsertionPoint(user);
      auto loc = user->getLoc();
      const auto& address = analysis.address(cxx::getLoadStoreAddress(user));
      SmallVector<Value> indices;
      for (const auto& terms : address.indices)
        indices.push_back(toIndex(rewriter, loc, terms));
      for (unsigned zero = 0; zero < access.zeros; ++zero)
        indices.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
      auto leafStorage = storage[access.leaf];
      auto element = leaves[access.leaf].element;
      if (auto load = dyn_cast<cxx::LoadOp>(user)) {
        Value value = isa<cxx::PointerType>(element)
                          ? Value(cxx::LoadOp::create(rewriter, loc, element,
                                                      leafStorage, int64_t(0)))
                          : Value(memref::LoadOp::create(rewriter, loc,
                                                         leafStorage, indices));
        rewriter.replaceOp(load,
                           reinterpret(rewriter, loc, value, load.getType()));
        continue;
      }
      auto store = cast<cxx::StoreOp>(user);
      auto value = reinterpret(rewriter, loc, store.getValue(), element);
      if (isa<cxx::PointerType>(element))
        cxx::StoreOp::create(rewriter, loc, value, leafStorage, int64_t(0));
      else
        memref::StoreOp::create(rewriter, loc, value, leafStorage, indices);
      rewriter.eraseOp(store);
    }
    for (auto* derivation : llvm::reverse(analysis.derivations()))
      rewriter.eraseOp(derivation);
    rewriter.eraseOp(op);
    return success();
  }
};

[[nodiscard]] auto hasObjectLayout(Type type) -> bool {
  if (auto array = dyn_cast<cxx::ArrayType>(type))
    return hasObjectLayout(array.getElementType());
  if (auto record = dyn_cast<cxx::ClassType>(type))
    return llvm::all_of(record.getBody(), hasObjectLayout);
  return isa<IntegerType, FloatType, VectorType, cxx::PointerType>(type);
}

[[nodiscard]] auto objectByteSize(Value address, const DataLayout& layout)
    -> std::optional<int64_t> {
  auto object = cast<cxx::PointerType>(address.getType()).getElementType();
  if (!hasObjectLayout(object)) return std::nullopt;
  return int64_t(layout.getTypeSize(object).getFixedValue());
}

class ObjectZeroToBytes : public OpRewritePattern<cxx::ZeroInitOp> {
 public:
  ObjectZeroToBytes(MLIRContext* context, const DataLayoutAnalysis& layouts)
      : OpRewritePattern(context), layouts_(layouts) {}

  auto matchAndRewrite(cxx::ZeroInitOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto bytes = objectByteSize(op.getAddr(), layouts_.getAtOrAbove(op));
    if (!bytes) return failure();
    rewriter.replaceOpWithNewOp<cxx::MemSetZeroOp>(op, op.getAddr(), *bytes);
    return success();
  }

 private:
  const DataLayoutAnalysis& layouts_;
};

class ObjectCopyToBytes : public OpRewritePattern<cxx::CopyOp> {
 public:
  ObjectCopyToBytes(MLIRContext* context, const DataLayoutAnalysis& layouts)
      : OpRewritePattern(context), layouts_(layouts) {}

  auto matchAndRewrite(cxx::CopyOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto bytes = objectByteSize(op.getDest(), layouts_.getAtOrAbove(op));
    if (!bytes) return failure();
    rewriter.replaceOpWithNewOp<cxx::MemCpyOp>(op, op.getDest(), op.getSrc(),
                                               *bytes);
    return success();
  }

 private:
  const DataLayoutAnalysis& layouts_;
};

[[nodiscard]] auto stripPointerCasts(Value address) -> Value {
  while (auto cast = address.getDefiningOp<cxx::BitcastOp>()) {
    if (!isa<cxx::PointerType>(cast.getValue().getType())) break;
    address = cast.getValue();
  }
  return address;
}

class ConstantSizeMemcpy : public OpRewritePattern<cxx::BuiltinCallOp> {
 public:
  ConstantSizeMemcpy(MLIRContext* context, const DataLayoutAnalysis& layouts)
      : OpRewritePattern(context), layouts_(layouts) {}

  auto matchAndRewrite(cxx::BuiltinCallOp op, PatternRewriter& rewriter) const
      -> LogicalResult override {
    if (op.getBuiltinName() != "__builtin_memcpy" || op.getInputs().size() != 3)
      return failure();
    auto size = getConstantIntValue(op.getInputs()[2]);
    if (!size) return failure();
    auto destination = stripPointerCasts(op.getInputs()[0]);
    auto source = stripPointerCasts(op.getInputs()[1]);
    const auto& layout = layouts_.getAtOrAbove(op);
    auto destinationBytes = objectByteSize(destination, layout);
    auto sourceBytes = objectByteSize(source, layout);
    if (!destinationBytes || destinationBytes != size || sourceBytes != size)
      return failure();
    auto target = pointeeOf(destination);
    auto origin = pointeeOf(source);
    if (target == origin && canExpandObject(target)) {
      expandObject(rewriter, op.getLoc(), destination, source);
    } else if (isTypedStorageElement(target) && isTypedStorageElement(origin) &&
               canReinterpret(origin, target)) {
      Value value = cxx::LoadOp::create(rewriter, op.getLoc(), origin, source,
                                        int64_t(0));
      value = reinterpret(rewriter, op.getLoc(), value, target);
      cxx::StoreOp::create(rewriter, op.getLoc(), value, destination,
                           int64_t(0));
    } else {
      return failure();
    }
    if (op.getResult())
      rewriter.replaceOp(op, op.getInputs()[0]);
    else
      rewriter.eraseOp(op);
    return success();
  }

 private:
  const DataLayoutAnalysis& layouts_;
};

class ConvertCxxToMemRefPass
    : public PassWrapper<ConvertCxxToMemRefPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxToMemRefPass)

  ConvertCxxToMemRefPass() = default;
  ConvertCxxToMemRefPass(const ConvertCxxToMemRefPass& other)
      : PassWrapper(other) {}

  Option<bool> typedStorage{
      *this, "typed-storage",
      llvm::cl::desc("Store each scalar, vector and pointer of a local object "
                     "in its own typed leaf instead of a byte buffer, and "
                     "convert only objects whose every access is typed"),
      llvm::cl::init(false)};

  Option<bool> expandZeroInit{
      *this, "expand-zero-init",
      llvm::cl::desc("Expand zero initialization to SCF loops instead of "
                     "preserving byte zero operations"),
      llvm::cl::init(false)};

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "convert-cxx-to-memref";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Convert eligible scalar, fixed-array and record Cxx storage to "
           "MemRef";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cxx::CxxDialect, arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, ptr::PtrDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    if (typedStorage) {
      runTypedStorage();
      return;
    }
    RewritePatternSet patterns(&getContext());
    const auto& layouts = getAnalysis<DataLayoutAnalysis>();
    patterns.add<ScalarStorageToMemRef>(&getContext(), layouts);
    patterns.add<ArrayStorageToMemRef>(&getContext(), layouts,
                                       bool(expandZeroInit));
    auto layoutAttr =
        getOperation()->getAttrOfType<StringAttr>("cxx.data-layout");
    auto llvmLayout =
        llvm::DataLayout::parse(layoutAttr ? layoutAttr.getValue() : "");
    if (llvmLayout)
      patterns.add<RecordStorageToMemRef>(&getContext(), layouts, *llvmLayout,
                                          bool(expandZeroInit));
    else
      llvm::consumeError(llvmLayout.takeError());
    patterns.add<ByteCastToMemRef, ObjectZeroToBytes, ObjectCopyToBytes>(
        &getContext(), layouts);
    patterns.add<ByteCopyToMemRef, BridgedLoadToMemRef, BridgedStoreToMemRef>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

 private:
  void runTypedStorage() {
    auto module = getOperation();
    SymbolTableCollection symbols;
    module.walk([&](FunctionOpInterface function) {
      materializeConstantGlobals(function, symbols);
      rewritePointerInductions(function);
      speculatePointerPhis(function);
    });
    RewritePatternSet normalization(&getContext());
    normalization.add<TypedZeroInit, TypedCopy, AggregateValueCopy,
                      AggregateValueExtract, AddressThroughSelect,
                      LoadThroughSelect, StoreThroughSelect,
                      PointerComparisonToIndex, PointerDifferenceToIndex>(
        &getContext());
    normalization.add<ConstantSizeMemcpy>(&getContext(),
                                          getAnalysis<DataLayoutAnalysis>());
    if (failed(applyPatternsGreedily(module, std::move(normalization))))
      return signalPassFailure();
    RewritePatternSet storage(&getContext());
    storage.add<TypedStorageToMemRef>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(storage))))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createConvertCxxToMemRefPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::ConvertCxxToMemRefPass>();
}

void cxx::registerConvertCxxToMemRefPass() {
  mlir::registerPass(createConvertCxxToMemRefPass);
}
