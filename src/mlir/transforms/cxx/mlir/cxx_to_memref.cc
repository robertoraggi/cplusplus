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
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
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
  auto function = dyn_cast<cxx::FuncOp>(op->getParentOp());
  if (!function || op->getBlock() != &function.getBody().front()) return false;
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
  if (!isa<IntegerType, FloatType, cxx::PointerType>(element)) return failure();
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
    Value address;
    if (auto load = dyn_cast<cxx::LoadOp>(access)) address = load.getAddr();
    if (auto store = dyn_cast<cxx::StoreOp>(access)) address = store.getAddr();
    convertElementAccess(access, allocation,
                         getArrayCoordinates(address, indices), rewriter);
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
      Value address;
      if (auto load = dyn_cast<cxx::LoadOp>(access)) address = load.getAddr();
      if (auto store = dyn_cast<cxx::StoreOp>(access))
        address = store.getAddr();
      convertElementAccess(access, views.lookup(address), {}, rewriter);
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

class ConvertCxxToMemRefPass
    : public PassWrapper<ConvertCxxToMemRefPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxToMemRefPass)

  ConvertCxxToMemRefPass() = default;
  ConvertCxxToMemRefPass(const ConvertCxxToMemRefPass& other)
      : PassWrapper(other) {}

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
    patterns.add<ByteCastToMemRef>(&getContext(), layouts);
    patterns.add<ByteCopyToMemRef, BridgedLoadToMemRef, BridgedStoreToMemRef>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
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
