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
#include <cxx/mlir/buffer_layout.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/device_atomics_to_spirv.h>
#include <cxx/mlir/kernel_contract.h>
#include <cxx/mlir/passes.h>
#include <cxx/mlir/spirv_buffer_layout.h>
#include <cxx/mlir/storage_address.h>
#include <llvm/ADT/ScopeExit.h>
#include <mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Conversion/UBToSPIRV/UBToSPIRV.h>
#include <mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Iterators.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/PassRegistry.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace mlir {
namespace {

[[nodiscard]] auto storageClassOf(Attribute space)
    -> std::optional<spirv::StorageClass> {
  if (auto storage = dyn_cast_or_null<spirv::StorageClassAttr>(space))
    return storage.getValue();
  if (isa_and_nonnull<ptr::GenericSpaceAttr>(space))
    return spirv::StorageClass::Function;
  auto gpuSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(space);
  if (!gpuSpace) return spirv::mapMemorySpaceToVulkanStorageClass(space);
  switch (gpuSpace.getValue()) {
    case gpu::AddressSpace::Private:
      return spirv::StorageClass::Function;
    case gpu::AddressSpace::Workgroup:
      return spirv::StorageClass::Workgroup;
    case gpu::AddressSpace::Global:
      return spirv::StorageClass::StorageBuffer;
    default:
      return std::nullopt;
  }
}

void mapStorageClasses(ModuleOp module) {
  spirv::MemorySpaceToStorageClassConverter converter(storageClassOf);
  module.walk([&](gpu::GPUModuleOp gpuModule) {
    spirv::convertMemRefTypesAndAttrs(gpuModule, converter);
  });
}

class DeviceSPIRVTypeConverter final : public SPIRVTypeConverter {
 public:
  explicit DeviceSPIRVTypeConverter(spirv::TargetEnvAttr environment)
      : SPIRVTypeConverter(environment, options()) {
    addConversion([this](cxx::ClassType type) { return record(type); });
    addConversion([this](TupleType type) { return record(type); });
    addConversion([this](cxx::ArrayType type) { return array(type); });
    addConversion([this](MemRefType type) { return storage(type); });
    addConversion([this](cxx::PointerType type) -> Type {
      if (type.getAddressSpace()) return {};
      auto element = convertType(type.getElementType());
      if (!element) return {};
      return spirv::PointerType::get(element, spirv::StorageClass::Function);
    });
  }

 private:
  [[nodiscard]] static auto options() -> SPIRVConversionOptions {
    SPIRVConversionOptions result;
    result.use64bitIndex = false;
    return result;
  }

  [[nodiscard]] auto array(cxx::ArrayType type) -> Type {
    if (!type.getSize() ||
        type.getSize() > std::numeric_limits<std::uint32_t>::max())
      return {};
    auto element = convertType(type.getElementType());
    if (!element || isa<spirv::PointerType>(element)) return {};
    return spirv::ArrayType::get(element, type.getSize());
  }

  [[nodiscard]] auto record(Type type) -> Type {
    auto body = cxx::getRecordElementTypes(type);
    if (body.empty() || !active_.insert(type).second) return {};
    llvm::scope_exit leave([&] { active_.erase(type); });
    SmallVector<Type> values;
    for (auto position : cxx::valueElementPositions(type))
      values.push_back(body[position]);
    SmallVector<Type> fields;
    if (failed(convertTypes(values, fields))) return {};
    if (llvm::any_of(fields,
                     [](Type field) { return isa<spirv::PointerType>(field); }))
      return {};
    return spirv::StructType::get(fields);
  }

  [[nodiscard]] static auto isExplicitLayoutStorageClass(
      spirv::StorageClass storage) -> bool {
    switch (storage) {
      case spirv::StorageClass::StorageBuffer:
      case spirv::StorageClass::Uniform:
        return true;
      default:
        return false;
    }
  }

  [[nodiscard]] static auto isAggregateStorageClass(spirv::StorageClass storage)
      -> bool {
    switch (storage) {
      case spirv::StorageClass::Function:
      case spirv::StorageClass::Workgroup:
        return true;
      default:
        return false;
    }
  }

  [[nodiscard]] static auto resourceBlock(MemRefType type,
                                          spirv::StorageClass storage) -> Type {
    auto element = type.getElementType();
    auto layout = cxx::spirvBufferLayoutType(element);
    auto stride = cxx::bufferLayoutSize(element);
    Type member = spirv::RuntimeArrayType::get(layout, stride);
    if (type.getRank() == 0) member = spirv::ArrayType::get(layout, 1, stride);
    return spirv::PointerType::get(spirv::StructType::get({member}, {0}),
                                   storage);
  }

  [[nodiscard]] auto storage(MemRefType type) -> std::optional<Type> {
    if (!cxx::isAggregateType(type.getElementType())) return std::nullopt;
    auto space =
        dyn_cast_or_null<spirv::StorageClassAttr>(type.getMemorySpace());
    if (space && isExplicitLayoutStorageClass(space.getValue()) &&
        cxx::isBufferLayoutType(type.getElementType()) && type.getRank() <= 1 &&
        type.getLayout().isIdentity())
      return resourceBlock(type, space.getValue());
    if (!type.hasStaticShape() || !type.getLayout().isIdentity()) return Type{};
    if (!space || !isAggregateStorageClass(space.getValue())) return Type{};

    std::uint64_t count = 1;
    for (auto dimension : type.getShape()) {
      if (dimension <= 0) return Type{};
      if (std::uint64_t(dimension) >
          std::numeric_limits<std::uint32_t>::max() / count)
        return Type{};
      count *= dimension;
    }
    auto element = convertType(type.getElementType());
    if (!element) return Type{};
    auto array = spirv::ArrayType::get(element, count);
    auto block = spirv::StructType::get({array});
    return spirv::PointerType::get(block, space.getValue());
  }

  DenseSet<Type> active_;
};

[[nodiscard]] auto workgroupVariableName(spirv::ModuleOp module)
    -> std::string {
  auto index = llvm::range_size(module.getOps<spirv::GlobalVariableOp>());
  auto name = "__workgroup_mem__" + std::to_string(index);
  while (SymbolTable::lookupSymbolIn(module, name))
    name = "__workgroup_mem__" + std::to_string(++index);
  return name;
}

class AggregateAllocaToSPIRV final
    : public OpConversionPattern<memref::AllocaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(memref::AllocaOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!cxx::isAggregateType(op.getType().getElementType())) return failure();
    auto type = getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op, "unsupported aggregate storage");
    rewriter.replaceOpWithNewOp<spirv::VariableOp>(
        op, type, spirv::StorageClass::Function, nullptr);
    return success();
  }
};

class WorkgroupAggregateToSPIRV final
    : public OpConversionPattern<memref::AllocOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(memref::AllocOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    if (!cxx::isAggregateType(op.getType().getElementType())) return failure();
    auto type = dyn_cast_or_null<spirv::PointerType>(
        getTypeConverter()->convertType(op.getType()));
    if (!type || type.getStorageClass() != spirv::StorageClass::Workgroup)
      return rewriter.notifyMatchFailure(
          op, "unsupported workgroup aggregate storage");
    auto module = op->getParentOfType<spirv::ModuleOp>();
    if (!module) return failure();
    auto global = createGlobal(rewriter, module, type, op.getLoc());
    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(op, global);
    return success();
  }

 private:
  [[nodiscard]] static auto createGlobal(ConversionPatternRewriter& rewriter,
                                         spirv::ModuleOp module,
                                         spirv::PointerType type, Location loc)
      -> spirv::GlobalVariableOp {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module.getRegion().front());
    return spirv::GlobalVariableOp::create(
        rewriter, loc, type, workgroupVariableName(module), nullptr);
  }
};

class UndefToSPIRV final : public OpConversionPattern<cxx::UndefOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::UndefOp op, OpAdaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto type = getTypeConverter()->convertType(op.getType());
    if (!type) return failure();
    rewriter.replaceOpWithNewOp<spirv::UndefOp>(op, type);
    return success();
  }
};

using StorageAddresses = DenseMap<Value, cxx::StorageAddress>;

class StoragePathIndices {
 public:
  StoragePathIndices(MemRefType memory, Location loc,
                     ConversionPatternRewriter& rewriter)
      : object_(memory.getElementType()),
        dimensions_(memory.getRank()),
        loc_(loc),
        rewriter_(rewriter),
        zero_(constant(0)) {
    indices_.push_back(zero_);
    if (!dimensions_) indices_.push_back(zero_);
  }

  [[nodiscard]] auto append(const cxx::StorageIndex& index) -> LogicalResult {
    if (auto member = std::get_if<cxx::StorageMember>(&index))
      return appendMember(member->position);
    auto element = elementIndex(std::get<cxx::StorageElement>(index));
    if (failed(element)) return failure();
    indices_.push_back(*element);
    if (dimensions_) {
      --dimensions_;
      return success();
    }
    auto array = dyn_cast_or_null<cxx::ArrayType>(object_);
    if (!array) return failure();
    object_ = array.getElementType();
    return success();
  }

  [[nodiscard]] auto indices() const -> ArrayRef<Value> { return indices_; }

 private:
  [[nodiscard]] auto constant(std::int32_t value) -> Value {
    return spirv::ConstantOp::create(rewriter_, loc_, rewriter_.getI32Type(),
                                     rewriter_.getI32IntegerAttr(value));
  }

  [[nodiscard]] auto appendMember(std::int32_t position) -> LogicalResult {
    if (dimensions_ || position < 0) return failure();
    auto field = cxx::getAggregateElementType(object_, position);
    if (!field) return failure();
    auto index = cxx::valueElementIndex(object_, position);
    if (!index) return failure();
    indices_.push_back(constant(static_cast<std::int32_t>(*index)));
    object_ = field;
    return success();
  }

  [[nodiscard]] auto elementIndex(const cxx::StorageElement& element)
      -> FailureOr<Value> {
    Value total;
    for (auto term : element.terms) {
      auto converted = rewriter_.getRemappedValue(term);
      if (!converted || !converted.getType().isInteger(32)) return failure();
      total = total ? spirv::IAddOp::create(rewriter_, loc_, total, converted)
                    : converted;
    }
    return total ? total : zero_;
  }

  Type object_;
  std::int64_t dimensions_;
  Location loc_;
  ConversionPatternRewriter& rewriter_;
  Value zero_;
  SmallVector<Value> indices_;
};

[[nodiscard]] auto addressToSPIRV(Value pointer,
                                  const StorageAddresses& addresses,
                                  ConversionPatternRewriter& rewriter)
    -> FailureOr<Value> {
  auto found = addresses.find(pointer);
  if (found == addresses.end()) return failure();
  const auto& address = found->second;
  auto root = rewriter.getRemappedValue(address.root);
  if (!root || !isa<spirv::PointerType>(root.getType())) return failure();
  auto loc = pointer.getLoc();
  StoragePathIndices indices(cast<MemRefType>(address.root.getType()), loc,
                             rewriter);
  for (const auto& index : address.path)
    if (failed(indices.append(index))) return failure();
  return spirv::AccessChainOp::create(rewriter, loc, root, indices.indices())
      .getResult();
}

template <typename Op>
class StorageAddressToSPIRV final : public OpConversionPattern<Op> {
 public:
  StorageAddressToSPIRV(MLIRContext* context, const StorageAddresses& addresses)
      : OpConversionPattern<Op>(context), addresses_(addresses) {}

  auto matchAndRewrite(Op op, typename Op::Adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto pointer = addressToSPIRV(op.getResult(), addresses_, rewriter);
    if (failed(pointer)) return failure();
    rewriter.replaceOp(op, *pointer);
    return success();
  }

 private:
  const StorageAddresses& addresses_;
};

class StorageRootToSPIRV final : public OpConversionPattern<ptr::ToPtrOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(ptr::ToPtrOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto memory = dyn_cast<MemRefType>(op.getPtr().getType());
    if (!memory || memory.getRank() > 1 || !memory.getLayout().isIdentity())
      return failure();
    if (!isa<spirv::PointerType>(adaptor.getPtr().getType())) return failure();
    auto zero =
        spirv::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(op, adaptor.getPtr(),
                                                      ValueRange{zero, zero});
    return success();
  }
};

[[nodiscard]] auto storageTypeOf(Operation* op, Value address) -> Type {
  if (!op->getDiscardableAttrDictionary().empty()) return {};
  auto pointer = dyn_cast<spirv::PointerType>(address.getType());
  if (!pointer) return {};
  return pointer.getPointeeType();
}

class StorageLoadToSPIRV final : public OpConversionPattern<cxx::LoadOp> {
 public:
  StorageLoadToSPIRV(const TypeConverter& types, MLIRContext* context)
      : OpConversionPattern(context), types_(types) {}

  auto matchAndRewrite(cxx::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto storage = storageTypeOf(op, adaptor.getAddr());
    if (!storage) return failure();
    auto result = types_.convertType(op.getType());
    if (!result || !cxx::canBitcastStorageValue(storage, result))
      return failure();
    Value loaded =
        spirv::LoadOp::create(rewriter, op.getLoc(), adaptor.getAddr());
    if (loaded.getType() != result)
      loaded = spirv::BitcastOp::create(rewriter, op.getLoc(), result, loaded);
    rewriter.replaceOp(op, loaded);
    return success();
  }

 private:
  const TypeConverter& types_;
};

class StorageStoreToSPIRV final : public OpConversionPattern<cxx::StoreOp> {
 public:
  explicit StorageStoreToSPIRV(MLIRContext* context)
      : OpConversionPattern(context) {}

  auto matchAndRewrite(cxx::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto storage = storageTypeOf(op, adaptor.getAddr());
    if (!storage) return failure();
    Value value = adaptor.getValue();
    if (!cxx::canBitcastStorageValue(value.getType(), storage))
      return failure();
    if (value.getType() != storage)
      value = spirv::BitcastOp::create(rewriter, op.getLoc(), storage, value);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(op, adaptor.getAddr(), value);
    return success();
  }
};

class ScalarSelectToSPIRV final : public OpConversionPattern<arith::SelectOp> {
 public:
  ScalarSelectToSPIRV(const TypeConverter& types, MLIRContext* context)
      : OpConversionPattern(types, context, 2) {}

  auto matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto vector = dyn_cast<VectorType>(adaptor.getTrueValue().getType());
    if (!vector || isa<VectorType>(adaptor.getCondition().getType()))
      return failure();
    auto environment = spirv::lookupTargetEnvOrDefault(op);
    if (environment.getVersion() >= spirv::Version::V_1_4) return failure();
    SmallVector<Value> lanes(vector.getNumElements(), adaptor.getCondition());
    auto mask = VectorType::get(vector.getShape(), rewriter.getI1Type());
    auto condition =
        spirv::CompositeConstructOp::create(rewriter, op.getLoc(), mask, lanes);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(
        op, condition, adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

[[nodiscard]] auto compositeIndex(Operation* op, Type container,
                                  std::int64_t position,
                                  ConversionPatternRewriter& rewriter)
    -> FailureOr<std::int32_t> {
  if (position < 0 || position > std::numeric_limits<std::int32_t>::max())
    return rewriter.notifyMatchFailure(op,
                                       "aggregate index is not representable");
  auto index = cxx::valueElementIndex(container, position);
  if (!index)
    return rewriter.notifyMatchFailure(op, "aggregate index has no value");
  return std::int32_t(*index);
}

class InsertValueToSPIRV final
    : public OpConversionPattern<cxx::InsertValueOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::InsertValueOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto index = compositeIndex(op, op.getContainer().getType(),
                                op.getPosition(), rewriter);
    if (failed(index)) return failure();
    rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(
        op, adaptor.getValue(), adaptor.getContainer(),
        ArrayRef<std::int32_t>{*index});
    return success();
  }
};

class ExtractValueToSPIRV final
    : public OpConversionPattern<cxx::ExtractValueOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  auto matchAndRewrite(cxx::ExtractValueOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult override {
    auto index = compositeIndex(op, op.getContainer().getType(),
                                op.getPosition(), rewriter);
    if (failed(index)) return failure();
    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        op, adaptor.getContainer(), ArrayRef<std::int32_t>{*index});
    return success();
  }
};

void eraseDeadOperations(Operation* root) {
  root->walk<WalkOrder::PostOrder, ReverseIterator>([](Operation* op) {
    if (isOpTriviallyDead(op)) op->erase();
  });
}

[[nodiscard]] auto checkFunctionStorage(func::FuncOp function)
    -> LogicalResult {
  for (auto result : function.getResultTypes())
    if (isa<cxx::PointerType, MemRefType>(result))
      return function.emitError(
          "device reference results require a storage-root descriptor");
  for (auto [index, input] : llvm::enumerate(function.getArgumentTypes())) {
    if (isa<cxx::PointerType>(input))
      return function.emitError(
          "device reference parameters require storage-root legalization");
    auto memory = dyn_cast<MemRefType>(input);
    if (!memory) continue;
    auto space =
        dyn_cast_or_null<spirv::StorageClassAttr>(memory.getMemorySpace());
    if (!space)
      return function.emitError("device parameter has no storage class");
    if (space.getValue() != spirv::StorageClass::Function) {
      if (space.getValue() == spirv::StorageClass::Workgroup) continue;
      if (!cxx::deviceResourceBinding(function.getArgument(index)))
        return function.emitError(
            "device resource parameters require binding specialization");
      continue;
    }
    if (!memory.hasStaticShape() || !memory.getLayout().isIdentity())
      return function.emitError(
          "device local parameters require whole storage roots");
  }
  return success();
}

class ConvertCxxDeviceToSPIRVPass final
    : public PassWrapper<ConvertCxxDeviceToSPIRVPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxDeviceToSPIRVPass)

  [[nodiscard]] auto getArgument() const -> StringRef override {
    return "convert-cxx-device-to-spirv";
  }

  [[nodiscard]] auto getDescription() const -> StringRef override {
    return "Convert device functions and typed storage to SPIR-V";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cxx::CxxDialect, arith::ArithDialect, func::FuncDialect,
                    gpu::GPUDialect, math::MathDialect, memref::MemRefDialect,
                    ptr::PtrDialect, scf::SCFDialect, spirv::SPIRVDialect,
                    ub::UBDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    mapStorageClasses(module);
    auto environment = spirv::lookupTargetEnvOrDefault(module);
    DeviceSPIRVTypeConverter types(environment);

    for (auto device :
         llvm::make_early_inc_range(module.getOps<gpu::GPUModuleOp>())) {
      auto signatures = device.walk([](func::FuncOp function) {
        return succeeded(checkFunctionStorage(function))
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      });
      if (signatures.wasInterrupted()) return signalPassFailure();
      auto synchronization = device.walk([](Operation* op) {
        return succeeded(cxx::verifyDeviceSynchronization(op))
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      });
      if (synchronization.wasInterrupted()) return signalPassFailure();
      auto calls = device.walk([](func::CallOp call) {
        if (succeeded(cxx::verifyDeviceCallOperands(call)))
          return WalkResult::advance();
        return WalkResult::interrupt();
      });
      if (calls.wasInterrupted()) return signalPassFailure();
      StorageAddresses addresses;
      device.walk([&](Operation* operation) {
        for (auto result : operation->getResults()) {
          if (!isa<cxx::PointerType>(result.getType())) continue;
          if (auto address = cxx::resolveStorageAddress(result);
              succeeded(address))
            addresses.try_emplace(result, std::move(*address));
        }
      });
      RewritePatternSet patterns(&getContext());
      ::cxx::populateDeviceAtomicsToSPIRVPatterns(patterns);
      ScfToSPIRVContext controlFlow;
      populateGPUToSPIRVPatterns(types, patterns);
      populateBuiltinFuncToSPIRVPatterns(types, patterns);
      populateFuncToSPIRVPatterns(types, patterns);
      populateMemRefToSPIRVPatterns(types, patterns);
      populateSCFToSPIRVPatterns(types, controlFlow, patterns);
      populateMathToSPIRVPatterns(types, patterns);
      ub::populateUBToSPIRVConversionPatterns(types, patterns);
      arith::populateArithToSPIRVPatterns(types, patterns);
      populateVectorToSPIRVPatterns(types, patterns);
      patterns.add<AggregateAllocaToSPIRV, WorkgroupAggregateToSPIRV,
                   UndefToSPIRV, InsertValueToSPIRV, ExtractValueToSPIRV,
                   ScalarSelectToSPIRV, StorageRootToSPIRV, StorageLoadToSPIRV>(
          types, &getContext());
      patterns.add<StorageStoreToSPIRV>(&getContext());
      patterns.add<StorageAddressToSPIRV<cxx::StorageRefOp>,
                   StorageAddressToSPIRV<ptr::FromPtrOp>,
                   StorageAddressToSPIRV<cxx::MemberOp>,
                   StorageAddressToSPIRV<cxx::SubscriptOp>,
                   StorageAddressToSPIRV<cxx::ArrayToPointerOp>,
                   StorageAddressToSPIRV<cxx::PtrAddOp>,
                   StorageAddressToSPIRV<cxx::BitcastOp>>(&getContext(),
                                                          addresses);
      auto target = SPIRVConversionTarget::get(environment);
      if (failed(applyFullConversion(device, *target, std::move(patterns))))
        return signalPassFailure();
    }
    eraseDeadOperations(module);

    module.walk([](spirv::FuncOp function) {
      if (!function->hasAttr("no_inline")) return;
      function.setFunctionControl(spirv::FunctionControl::DontInline);
      function->removeAttr("no_inline");
    });
  }
};

}  // namespace
}  // namespace mlir

auto cxx::createConvertCxxDeviceToSPIRVPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::ConvertCxxDeviceToSPIRVPass>();
}

void cxx::registerConvertCxxDeviceToSPIRVPass() {
  mlir::registerPass(createConvertCxxDeviceToSPIRVPass);
}
