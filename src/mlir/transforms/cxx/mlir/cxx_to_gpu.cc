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
#include <cxx/mlir/bitfield_accesses.h>
#include <cxx/mlir/buffer_layout.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/device_barriers.h>
#include <cxx/mlir/kernel_contract.h>
#include <cxx/mlir/passes.h>
#include <cxx/mlir/storage_address.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Demangle/Demangle.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

namespace mlir {
namespace {

struct ResourceAddress {
  Value buffer;
  Value offset;
  bool readOnly;
};

[[nodiscard]] auto isResourcePointer(cxx::PointerType pointer) -> bool {
  if (!pointer) return false;
  auto space =
      dyn_cast<LLVM::LLVMAddrSpaceAttrInterface>(pointer.getMemorySpace());
  if (!space) return false;
  return space.getAddressSpace() == ::cxx::kDeviceAddressSpace ||
         space.getAddressSpace() == ::cxx::kConstantAddressSpace;
}

[[nodiscard]] auto memrefTypeOf(MLIRContext* context,
                                const cxx::KernelBuffer& buffer) -> MemRefType {
  if (buffer.kind == cxx::KernelBufferKind::Uniform)
    return MemRefType::get(
        {}, buffer.elementType, MemRefLayoutAttrInterface(),
        spirv::StorageClassAttr::get(context, spirv::StorageClass::Uniform));
  return MemRefType::get({ShapedType::kDynamic}, buffer.elementType);
}

[[nodiscard]] auto isUniformResource(Value buffer) -> bool {
  return cast<MemRefType>(buffer.getType()).getRank() == 0;
}

[[nodiscard]] auto resourceIndices(const ResourceAddress& address)
    -> SmallVector<Value, 1> {
  if (isUniformResource(address.buffer)) return {};
  return {address.offset};
}

[[nodiscard]] auto createInvocationId(OpBuilder& builder, Location loc,
                                      cxx::KernelInvocationIdKind kind,
                                      gpu::Dimension dimension) -> Value {
  Value id;
  switch (kind) {
    case cxx::KernelInvocationIdKind::Global:
      id = gpu::GlobalIdOp::create(builder, loc, dimension);
      break;
    case cxx::KernelInvocationIdKind::Local:
      id = gpu::ThreadIdOp::create(builder, loc, dimension);
      break;
    case cxx::KernelInvocationIdKind::Workgroup:
      id = gpu::BlockIdOp::create(builder, loc, dimension);
      break;
  }
  return arith::IndexCastUIOp::create(builder, loc, builder.getI32Type(), id);
}

[[nodiscard]] auto materializeInvocationId(OpBuilder& builder, Location loc,
                                           cxx::KernelInvocationIdKind kind,
                                           Type type) -> Value {
  constexpr gpu::Dimension dimensions[] = {gpu::Dimension::x, gpu::Dimension::y,
                                           gpu::Dimension::z};
  auto vector = dyn_cast<VectorType>(type);
  if (!vector) return createInvocationId(builder, loc, kind, dimensions[0]);
  SmallVector<Value> components;
  for (int64_t index = 0; index < vector.getNumElements(); ++index)
    components.push_back(
        createInvocationId(builder, loc, kind, dimensions[index]));
  return vector::FromElementsOp::create(builder, loc, vector, components);
}

class DeviceCallGraph {
  struct FunctionCalls {
    cxx::FuncOp function;
    SmallVector<CallOpInterface> calls;
    unsigned next = 0;
  };

 public:
  [[nodiscard]] auto collect(cxx::FuncOp entry) -> LogicalResult {
    if (visited_.contains(entry)) return success();
    SmallVector<FunctionCalls> pending;
    append(entry, pending);
    while (!pending.empty()) {
      auto& current = pending.back();
      if (current.next == current.calls.size()) {
        auto function = current.function;
        active_.erase(function);
        visited_.insert(function);
        functions_.push_back(function);
        pending.pop_back();
        continue;
      }
      auto call = current.calls[current.next++];
      auto callee = target(call);
      if (failed(callee)) return failure();
      if (!visited_.contains(*callee)) append(*callee, pending);
    }
    return success();
  }

  [[nodiscard]] auto functions() const -> ArrayRef<cxx::FuncOp> {
    return functions_;
  }

 private:
  [[nodiscard]] static auto displayName(cxx::FuncOp function) -> std::string {
    return llvm::demangle(function.getSymName().str());
  }

  void append(cxx::FuncOp function, SmallVectorImpl<FunctionCalls>& pending) {
    FunctionCalls frame{function, {}, 0};
    function.walk([&](CallOpInterface call) { frame.calls.push_back(call); });
    pending.push_back(std::move(frame));
    active_.insert(function);
  }

  [[nodiscard]] auto target(CallOpInterface call) -> FailureOr<cxx::FuncOp> {
    auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
    if (!symbol)
      return call->emitError("device code cannot call through a pointer");
    auto callee =
        dyn_cast_or_null<cxx::FuncOp>(call.resolveCallableInTable(&symbols_));
    if (!callee)
      return call->emitError("device code calls an unknown function ")
             << symbol;
    if (callee.getBody().empty())
      return call->emitError("device function '")
             << displayName(callee) << "' has no definition";
    if (cxx::isKernelEntry(callee))
      return call->emitError(
          "device code cannot call a kernel entry point directly");
    if (callee.getFunctionType().getVariadic())
      return call->emitError("device function '")
             << displayName(callee) << "' is variadic";
    if (active_.contains(callee))
      return call->emitError("device function '")
             << displayName(callee) << "' is called recursively";
    return callee;
  }

  SymbolTableCollection symbols_;
  DenseSet<Operation*> active_;
  DenseSet<Operation*> visited_;
  SmallVector<cxx::FuncOp> functions_;
};

class ResourceLowering {
 public:
  ResourceLowering(gpu::GPUFuncOp function,
                   ArrayRef<cxx::KernelBuffer> resources)
      : function_(function), builder_(function.getContext()) {
    builder_.setInsertionPointToStart(&function.getBody().front());
    auto zero = arith::ConstantIndexOp::create(builder_, function.getLoc(), 0);
    for (auto [argument, resource] :
         llvm::zip(function.getArguments(), resources))
      addresses_[argument] = {argument, zero, resource.readOnly};
  }

  [[nodiscard]] auto run() -> LogicalResult {
    auto result = function_.walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (op == function_.getOperation()) return WalkResult::advance();
      builder_.setInsertionPoint(op);
      if (failed(lower(op))) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
    for (auto* pointer : llvm::reverse(obsolete_)) {
      if (!pointer->use_empty())
        return pointer->emitError(
            "device resource pointer escapes indexed access");
      pointer->erase();
    }
    return success();
  }

 private:
  [[nodiscard]] auto address(Value pointer, Operation* op)
      -> FailureOr<ResourceAddress> {
    if (auto found = addresses_.find(pointer); found != addresses_.end())
      return found->second;
    auto address = cxx::resolveStorageAddress(pointer);
    if (succeeded(address)) {
      auto root = addresses_.find(address->root);
      if (root != addresses_.end()) {
        auto result = root->second;
        if (isUniformResource(result.buffer)) {
          if (address->path.empty()) return result;
        } else if (address->path.size() == 1) {
          if (auto element =
                  std::get_if<cxx::StorageElement>(&address->path.front())) {
            result.offset = cxx::storageIndex(builder_, op->getLoc(), *element);
            return result;
          }
        }
      }
    }
    op->emitError(
        "device code cannot determine which kernel buffer this access "
        "designates");
    return failure();
  }

  [[nodiscard]] auto aggregateResource(Value pointer)
      -> std::optional<ResourceAddress> {
    auto address = cxx::resolveStorageAddress(pointer);
    if (failed(address)) return std::nullopt;
    auto root = addresses_.find(address->root);
    if (root == addresses_.end() ||
        !cxx::isAggregateType(
            cast<MemRefType>(address->root.getType()).getElementType()))
      return std::nullopt;
    return root->second;
  }

  [[nodiscard]] auto lower(Operation* op) -> LogicalResult {
    if (auto addition = dyn_cast<cxx::PtrAddOp>(op)) {
      auto base = cxx::resolveStorageAddress(addition.getBase());
      if (succeeded(base) && cxx::deviceResourceBinding(base->root) &&
          isUniformResource(base->root) &&
          !matchPattern(addition.getOffset(), m_Zero()))
        return op->emitError(
            "device pointer arithmetic leaves its uniform resource");
    }
    if (auto load = dyn_cast<cxx::LoadOp>(op)) {
      if (aggregateResource(load.getAddr())) return success();
      if (!addresses_.contains(load.getAddr()) &&
          !isResourcePointer(
              dyn_cast<cxx::PointerType>(load.getAddr().getType())))
        return success();
      auto root = address(load.getAddr(), op);
      if (failed(root)) return failure();
      auto value = memref::LoadOp::create(builder_, op->getLoc(), root->buffer,
                                          resourceIndices(*root));
      if (!cxx::canBitcastStorageValue(value.getType(), load.getType()))
        return op->emitError("incompatible device resource load type");
      Value result = value;
      if (value.getType() != load.getType())
        result = arith::BitcastOp::create(builder_, op->getLoc(),
                                          load.getType(), value);
      load.getResult().replaceAllUsesWith(result);
      obsolete_.push_back(op);
      return success();
    }
    if (auto store = dyn_cast<cxx::StoreOp>(op)) {
      if (auto resource = aggregateResource(store.getAddr())) {
        if (resource->readOnly)
          return op->emitError("cannot write a read-only device resource");
        return success();
      }
      if (!addresses_.contains(store.getAddr()) &&
          !isResourcePointer(
              dyn_cast<cxx::PointerType>(store.getAddr().getType())))
        return success();
      auto root = address(store.getAddr(), op);
      if (failed(root)) return failure();
      if (root->readOnly)
        return op->emitError("cannot write a read-only device resource");
      Value value = store.getValue();
      auto element = cast<MemRefType>(root->buffer.getType()).getElementType();
      if (!cxx::canBitcastStorageValue(value.getType(), element))
        return op->emitError("incompatible device resource store type");
      if (value.getType() != element)
        value =
            arith::BitcastOp::create(builder_, op->getLoc(), element, value);
      memref::StoreOp::create(builder_, op->getLoc(), value, root->buffer,
                              resourceIndices(*root));
      obsolete_.push_back(op);
      return success();
    }
    return success();
  }

  gpu::GPUFuncOp function_;
  OpBuilder builder_;
  DenseMap<Value, ResourceAddress> addresses_;
  SmallVector<Operation*> obsolete_;
};

[[nodiscard]] auto selectValue(OpBuilder& builder, Location loc,
                               Value condition, Value whenTrue, Value whenFalse)
    -> Value {
  auto type = whenTrue.getType();
  if (!cxx::isAggregateType(type))
    return arith::SelectOp::create(builder, loc, condition, whenTrue,
                                   whenFalse);
  Value result = cxx::UndefOp::create(builder, loc, type);
  for (std::int64_t index = 0;; ++index) {
    auto element = cxx::getAggregateElementType(type, index);
    if (!element) break;
    auto chosen = selectValue(
        builder, loc, condition,
        cxx::ExtractValueOp::create(builder, loc, element, whenTrue, index),
        cxx::ExtractValueOp::create(builder, loc, element, whenFalse, index));
    result =
        cxx::InsertValueOp::create(builder, loc, type, result, chosen, index);
  }
  return result;
}

void scalarizeAggregateSelects(gpu::GPUModuleOp module) {
  SmallVector<arith::SelectOp> selects;
  module.walk([&](arith::SelectOp select) {
    if (cxx::isAggregateType(select.getType())) selects.push_back(select);
  });
  for (auto select : selects) {
    OpBuilder builder(select);
    select.replaceAllUsesWith(
        selectValue(builder, select.getLoc(), select.getCondition(),
                    select.getTrueValue(), select.getFalseValue()));
    select.erase();
  }
}

[[nodiscard]] auto accessesNonValueElement(Operation* op) -> bool {
  if (auto extract = dyn_cast<cxx::ExtractValueOp>(op))
    return !cxx::isValueElement(extract.getContainer().getType(),
                                extract.getPosition());
  if (auto insert = dyn_cast<cxx::InsertValueOp>(op))
    return !cxx::isValueElement(insert.getType(), insert.getPosition());
  auto address = cxx::getLoadStoreAddress(op);
  if (!address) return false;
  auto resolved = cxx::resolveStorageAddress(address);
  if (failed(resolved)) return false;
  return cxx::designatesNonValueElement(*resolved);
}

void dropNonValueAccesses(gpu::GPUModuleOp module) {
  SmallVector<Operation*> accesses;
  module.walk([&](Operation* op) {
    if (accessesNonValueElement(op)) accesses.push_back(op);
  });
  for (auto* op : accesses) {
    OpBuilder builder(op);
    if (auto insert = dyn_cast<cxx::InsertValueOp>(op))
      insert.getResult().replaceAllUsesWith(insert.getContainer());
    else if (op->getNumResults() == 1)
      op->getResult(0).replaceAllUsesWith(cxx::UndefOp::create(
          builder, op->getLoc(), op->getResult(0).getType()));
    op->erase();
  }
}

[[nodiscard]] auto isDeviceType(Type type) -> bool {
  enum class Use { Address, Value };
  SmallVector<std::pair<Type, Use>> pending{{type, Use::Address}};
  DenseSet<std::pair<Type, Use>> visited;
  while (!pending.empty()) {
    auto [current, address] = pending.pop_back_val();
    if (!visited.insert({current, address}).second) continue;
    if (current.isF32() || current.isIndex()) continue;
    if (auto integer = dyn_cast<IntegerType>(current)) {
      if (integer.isSignless() && integer.getWidth() <= 32) continue;
      return false;
    }
    if (auto vector = dyn_cast<VectorType>(current)) {
      if (vector.getRank() != 1 || vector.getNumElements() < 2 ||
          vector.getNumElements() > 4)
        return false;
      pending.push_back({vector.getElementType(), Use::Value});
      continue;
    }
    if (auto memory = dyn_cast<MemRefType>(current)) {
      if (!memory.getLayout().isIdentity()) return false;
      pending.push_back({memory.getElementType(), Use::Value});
      continue;
    }
    if (auto pointer = dyn_cast<cxx::PointerType>(current)) {
      if (address != Use::Address) return false;
      pending.push_back({pointer.getElementType(), Use::Value});
      continue;
    }
    if (isa<ptr::PtrType>(current)) {
      if (address != Use::Address) return false;
      continue;
    }
    if (auto array = dyn_cast<cxx::ArrayType>(current)) {
      if (!array.getSize()) return false;
      pending.push_back({array.getElementType(), Use::Value});
      continue;
    }
    if (isa<cxx::ClassType, TupleType>(current)) {
      auto body = cxx::getRecordElementTypes(current);
      for (auto position : cxx::valueElementPositions(current))
        pending.push_back({body[position], Use::Value});
      continue;
    }
    return false;
  }
  return true;
}

[[nodiscard]] auto hasPointerMembers(Type type) -> bool {
  SmallVector<Type> pending{type};
  SmallVector<Type> visited;
  while (!pending.empty()) {
    auto current = pending.pop_back_val();
    if (llvm::is_contained(visited, current)) continue;
    visited.push_back(current);
    if (isa<cxx::PointerType, ptr::PtrType>(current)) return true;
    if (auto array = dyn_cast<cxx::ArrayType>(current))
      pending.push_back(array.getElementType());
    for (auto field : cxx::getRecordElementTypes(current))
      pending.push_back(field);
  }
  return false;
}

[[nodiscard]] auto rejectLocalObject(cxx::AllocaOp allocation)
    -> LogicalResult {
  auto object = cast<cxx::PointerType>(allocation.getType()).getElementType();
  if (isa<cxx::PointerType>(object))
    return allocation.emitError(
        "device code cannot refer to a pointer variable through a reference "
        "or pointer; pass or capture the pointer by value");
  if (hasPointerMembers(object))
    return allocation.emitError(
        "device code requires each pointer or reference member of a local "
        "object to be bound once before use, without copying the object");
  return allocation.emitError("device code cannot represent this local object");
}

[[nodiscard]] auto rejectVolatileAccesses(ModuleOp module) -> LogicalResult {
  auto found = module.walk([](Operation* operation) {
    if (!cxx::isVolatileAccess(operation)) return WalkResult::advance();
    operation->emitError(
        "device code cannot access an object through a volatile glvalue");
    return WalkResult::interrupt();
  });
  return failure(found.wasInterrupted());
}

[[nodiscard]] auto rejectType(Operation* operation, Type type,
                              llvm::StringRef role) -> LogicalResult {
  if (hasPointerMembers(type) && !isa<cxx::PointerType>(type))
    return operation->emitError(
        "device code cannot copy or return an object with pointer or "
        "reference members");
  return operation->emitError("unsupported device ")
         << role << " type " << type;
}

[[nodiscard]] auto isWorkgroupStorage(MemRefType memory) -> bool {
  auto space = dyn_cast_or_null<gpu::AddressSpaceAttr>(memory.getMemorySpace());
  return space && space.getValue() == gpu::AddressSpace::Workgroup;
}

[[nodiscard]] auto isUniformStorage(MemRefType memory) -> bool {
  auto space =
      dyn_cast_or_null<spirv::StorageClassAttr>(memory.getMemorySpace());
  return space && space.getValue() == spirv::StorageClass::Uniform;
}

[[nodiscard]] auto atomicStorageRoot(Operation* operation, Value pointer)
    -> FailureOr<Value> {
  auto address = cxx::resolveStorageAddress(pointer);
  if (failed(address))
    return operation->emitError("device atomic object has no storage root");
  auto memory = cast<MemRefType>(address->root.getType());
  if (isUniformStorage(memory))
    return operation->emitError("device atomics cannot access uniform storage");
  if (cxx::deviceResourceBinding(address->root)) return address->root;
  if (isWorkgroupStorage(memory)) return address->root;
  return operation->emitError(
      "device atomics require storage-buffer or threadgroup storage");
}

[[nodiscard]] auto verifyDeviceOperation(Operation* operation)
    -> LogicalResult {
  if (isa<func::FuncOp, gpu::GPUFuncOp, gpu::GPUModuleOp>(operation))
    return success();
  if (auto allocation = dyn_cast<cxx::AllocaOp>(operation))
    return rejectLocalObject(allocation);
  if (isa<func::CallOp>(operation) &&
      failed(cxx::verifyDeviceCallOperands(operation)))
    return failure();
  auto supported =
      isa<arith::ArithDialect, math::MathDialect>(operation->getDialect()) ||
      isa<func::CallOp, func::ReturnOp, gpu::ReturnOp, gpu::GlobalIdOp,
          gpu::ThreadIdOp, gpu::BlockIdOp, ub::PoisonOp, scf::IfOp,
          scf::WhileOp, scf::ForOp, scf::IndexSwitchOp, scf::ConditionOp,
          scf::YieldOp, vector::BroadcastOp, vector::ExtractOp,
          vector::InsertOp, vector::FromElementsOp, vector::ShuffleOp,
          memref::AllocaOp, memref::AllocOp, memref::LoadOp, memref::StoreOp,
          ptr::ToPtrOp, ptr::FromPtrOp, cxx::StorageRefOp, cxx::MemberOp,
          cxx::SubscriptOp, cxx::ArrayToPointerOp, cxx::PtrAddOp, cxx::LoadOp,
          cxx::StoreOp, cxx::BitcastOp, cxx::UndefOp, cxx::InsertValueOp,
          cxx::ExtractValueOp>(operation);
  supported |= cxx::isSynchronization(operation);
  if (!supported)
    return operation->emitError("unsupported operation '")
           << operation->getName() << "' in device code";
  if (failed(cxx::verifyDeviceSynchronization(operation))) return failure();
  if (isa<scf::SCFDialect>(operation->getDialect()) &&
      llvm::any_of(operation->getOperandTypes(),
                   llvm::IsaPred<cxx::PointerType>))
    return operation->emitError(
        "device code cannot determine which object a pointer or reference "
        "carried through this control flow designates");
  for (auto type : operation->getResultTypes())
    if (!isDeviceType(type)) return rejectType(operation, type, "result");
  for (auto type : operation->getOperandTypes())
    if (!isDeviceType(type)) return rejectType(operation, type, "operand");
  for (auto& region : operation->getRegions())
    for (auto& block : region)
      for (auto argument : block.getArguments())
        if (!isDeviceType(argument.getType()))
          return operation->emitError(
              "unsupported device region argument type");

  Value stored;
  if (auto pointer = cxx::atomicAddress(operation)) {
    auto root = atomicStorageRoot(operation, pointer);
    if (failed(root)) return failure();
    if (cxx::isAtomicWrite(operation)) stored = *root;
  }
  if (auto store = dyn_cast<memref::StoreOp>(operation))
    stored = store.getMemRef();
  if (auto store = dyn_cast<cxx::StoreOp>(operation)) {
    auto address = cxx::resolveStorageAddress(store.getAddr());
    if (succeeded(address)) stored = address->root;
  }
  if (stored && cxx::isReadOnlyDeviceResource(stored))
    return operation->emitError("cannot write a read-only device resource");
  return success();
}

class ConvertCxxKernelToGpuPass final
    : public PassWrapper<ConvertCxxKernelToGpuPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCxxKernelToGpuPass)

  ConvertCxxKernelToGpuPass() = default;
  ConvertCxxKernelToGpuPass(const ConvertCxxKernelToGpuPass& other)
      : PassWrapper(other) {}
  explicit ConvertCxxKernelToGpuPass(StringRef name) { kernel = name.str(); }

  [[nodiscard]] auto getArgument() const -> StringRef final {
    return "convert-cxx-kernel-to-gpu";
  }

  [[nodiscard]] auto getDescription() const -> StringRef final {
    return "Lower a checked Vulkan entry and its declared resources to "
           "GPU/MemRef";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                spirv::SPIRVDialect, ub::UBDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    if (failed(convert())) signalPassFailure();
  }

 private:
  [[nodiscard]] auto convert() -> LogicalResult {
    auto module = getOperation();
    if (!cxx::deviceProfileLimitsOf(module))
      return module.emitError(
          "kernel conversion requires a checked device profile");
    auto contract = cxx::kernelModuleContractOf(module, kernel.getValue());
    if (failed(contract)) return failure();
    DeviceCallGraph callGraph;
    for (const auto& entry : contract->entries)
      if (failed(
              callGraph.collect(module.lookupSymbol<cxx::FuncOp>(entry.name))))
        return failure();
    OwningOpRef<ModuleOp> temporary = ModuleOp::create(module.getLoc());
    (*temporary)->setAttrs(module->getAttrs());
    for (auto function : callGraph.functions()) {
      auto clone = cast<cxx::FuncOp>(function->clone());
      auto selected = llvm::any_of(contract->entries, [&](const auto& entry) {
        return entry.name == function.getSymName();
      });
      SymbolTable::setSymbolVisibility(
          clone, selected ? SymbolTable::Visibility::Public
                          : SymbolTable::Visibility::Private);
      temporary->getBody()->push_back(clone);
    }
    cloneReferencedGlobals(module, *temporary);
    temporary->walk(
        [](Operation* op) { op->removeDiscardableAttr("cxx.di_local"); });
    if (failed(rejectVolatileAccesses(*temporary))) return failure();
    if (failed(::cxx::expandBitfieldAccesses(*temporary,
                                             ::cxx::BitfieldMemory::kTyped)))
      return failure();
    if (failed(lowerDeviceStorage(*temporary))) return failure();
    temporary->walk([](memref::AllocaOp allocation) {
      auto type = allocation.getType();
      if (type.getMemorySpace()) return;
      allocation.getResult().setType(MemRefType::get(
          type.getShape(), type.getElementType(), type.getLayout(),
          ptr::GenericSpaceAttr::get(allocation.getContext())));
    });
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(temporary->getBody());
    auto gpuModule =
        gpu::GPUModuleOp::create(builder, module.getLoc(), "kernels");
    for (const auto& entry : contract->entries)
      if (failed(lowerEntry(*temporary, gpuModule, entry))) return failure();

    for (auto helper :
         llvm::make_early_inc_range(temporary->getOps<func::FuncOp>()))
      helper->moveBefore(gpuModule.getBody(), gpuModule.getBody()->end());
    PassManager references(module.getContext());
    references.addPass(::cxx::createLegalizeCxxDeviceCallsPass());
    references.addPass(createSymbolDCEPass());
    references.addPass(createCanonicalizerPass());
    references.addPass(createRemoveDeadValuesPass());
    references.addPass(createSROA());
    references.addPass(createMem2Reg());
    references.addPass(createCanonicalizerPass());
    references.addPass(::cxx::createLiftCxxControlFlowToSCFPass());
    references.addPass(createCanonicalizerPass());
    references.addPass(::cxx::createLegalizeCxxDeviceCallsPass());
    references.addPass(createCanonicalizerPass());
    references.addPass(createSROA());
    references.addPass(createMem2Reg());
    references.addPass(createCanonicalizerPass());
    if (failed(references.run(*temporary))) return failure();
    auto objects = gpuModule.walk([](cxx::AllocaOp allocation) {
      (void)rejectLocalObject(allocation);
      return WalkResult::interrupt();
    });
    if (objects.wasInterrupted()) return failure();
    for (const auto& entry : contract->entries) {
      auto function = gpuModule.lookupSymbol<gpu::GPUFuncOp>(entry.name);
      if (failed(ResourceLowering(function, entry.contract.buffers).run()))
        return failure();
    }
    scalarizeAggregateSelects(gpuModule);
    dropNonValueAccesses(gpuModule);

    if (failed(::cxx::checkDeviceBarrierUniformity(gpuModule)))
      return failure();

    auto checked = gpuModule.walk([](Operation* operation) {
      return succeeded(verifyDeviceOperation(operation))
                 ? WalkResult::advance()
                 : WalkResult::interrupt();
    });
    if (checked.wasInterrupted()) return failure();
    (*temporary)
        ->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  builder.getUnitAttr());
    if (failed(verify(*temporary))) return failure();
    module.getBodyRegion().takeBody(temporary->getBodyRegion());
    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    builder.getUnitAttr());
    return success();
  }

  [[nodiscard]] static auto lowerEntry(ModuleOp module,
                                       gpu::GPUModuleOp gpuModule,
                                       const cxx::KernelEntryContract& entry)
      -> LogicalResult {
    auto context = module.getContext();
    const auto& resources = entry.contract.buffers;
    auto device = module.lookupSymbol<func::FuncOp>(entry.name);
    if (!device)
      return module.emitError("selected kernel was not lowered to func");
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(gpuModule.getBody());
    SmallVector<Type> arguments;
    for (const auto& resource : resources)
      arguments.push_back(memrefTypeOf(context, resource));
    auto function =
        gpu::GPUFuncOp::create(builder, device.getLoc(), entry.name,
                               builder.getFunctionType(arguments, {}));
    function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                      builder.getUnitAttr());
    function->setAttr(
        spirv::getEntryPointABIAttrName(),
        spirv::getEntryPointABIAttr(context, entry.contract.workgroup));
    IRMapping mapping;
    builder.setInsertionPointToStart(&function.getBody().front());
    for (auto [resource, newArgument] :
         llvm::zip(resources, function.getArguments())) {
      cxx::StorageAddress address{newArgument, {}};
      if (resource.kind == cxx::KernelBufferKind::Storage)
        address.path.push_back(cxx::StorageElement{});
      auto original = device.getArgument(resource.parameter);
      auto pointer = cxx::storagePointer(builder, original.getLoc(), address,
                                         original.getType());
      if (failed(pointer))
        return function.emitError(
            "cannot represent the device buffer reference");
      mapping.map(original, *pointer);
    }
    for (const auto& id : entry.contract.invocationIds) {
      auto argument = device.getArgument(id.parameter);
      mapping.map(argument,
                  materializeInvocationId(builder, argument.getLoc(), id.kind,
                                          argument.getType()));
    }
    for (auto [index, resource] : llvm::enumerate(resources)) {
      function.setArgAttr(
          index, spirv::getInterfaceVarABIAttrName(),
          spirv::getInterfaceVarABIAttr(::cxx::kBufferDescriptorSet,
                                        resource.binding, {}, context));
      function.setArgAttr(index, cxx::kResourceIndexAttrName,
                          builder.getI32IntegerAttr(entry.resources[index]));
      function.setArgAttr(index, cxx::kResourceReadOnlyAttrName,
                          builder.getBoolAttr(resource.readOnly));
      function.setArgAttr(index, cxx::kResourceUnsignedAttrName,
                          builder.getBoolAttr(resource.unsignedElements));
    }
    auto& entryBlock = function.getBody().front();
    device.getBody().cloneInto(&function.getBody(), mapping);
    auto* cloned = mapping.lookup(&device.getBody().front());
    entryBlock.getOperations().splice(entryBlock.end(),
                                      cloned->getOperations());
    cloned->erase();
    auto returns = function.walk([&](func::ReturnOp returned) {
      if (returned.getNumOperands()) {
        returned.emitError("device entry must return void");
        return WalkResult::interrupt();
      }
      OpBuilder replacement(returned);
      gpu::ReturnOp::create(replacement, returned.getLoc());
      returned.erase();
      return WalkResult::advance();
    });
    device.erase();
    return failure(returns.wasInterrupted());
  }

  static void cloneReferencedGlobals(ModuleOp source, ModuleOp target) {
    SymbolTable symbols(source);
    llvm::SetVector<Operation*> globals;
    target.walk([&](cxx::AddressOfOp op) {
      if (auto global = symbols.lookup<cxx::GlobalOp>(op.getSymName()))
        globals.insert(global);
    });
    for (auto* global : globals) {
      auto clone = global->clone();
      SymbolTable::setSymbolVisibility(clone, SymbolTable::Visibility::Private);
      target.getBody()->push_back(clone);
    }
  }

  [[nodiscard]] auto lowerDeviceStorage(ModuleOp module) -> LogicalResult {
    auto context = module.getContext();
    PassManager calls(context);
    calls.addPass(::cxx::createConvertCxxWorkgroupStoragePass());
    calls.addPass(::cxx::createConvertCxxToFuncPass());
    calls.addPass(createMem2Reg());
    calls.addPass(createCanonicalizerPass());
    calls.addPass(::cxx::createLegalizeCxxDeviceAtomicsPass());
    calls.addPass(::cxx::createLegalizeCxxDeviceCallsPass());
    calls.addPass(createCanonicalizerPass());
    calls.addPass(createSROA());
    calls.addPass(createMem2Reg());
    calls.addPass(createCanonicalizerPass());
    calls.addPass(createSymbolDCEPass());
    if (failed(calls.run(module))) return failure();
    for (;;) {
      OperationFingerPrint before(module);
      PassManager storage(context);
      storage.addPass(createMem2Reg());
      storage.addPass(createCanonicalizerPass());
      storage.addPass(createCSEPass());
      auto typed = ::cxx::createConvertCxxToMemRefPass();
      if (failed(typed->initializeOptions(
              "typed-storage=true",
              [&](const llvm::Twine& message) -> LogicalResult {
                return module.emitError(message);
              })))
        return failure();
      storage.addPass(std::move(typed));
      storage.addPass(createCanonicalizerPass());
      if (failed(storage.run(module))) return failure();
      if (OperationFingerPrint(module) == before) break;
    }
    PassManager cleanup(context);
    cleanup.addPass(createSROA());
    cleanup.addPass(createMem2Reg());
    cleanup.addPass(createCanonicalizerPass());
    return cleanup.run(module);
  }

  Option<std::string> kernel{*this, "kernel",
                             llvm::cl::desc("Selected checked entry name"),
                             llvm::cl::init("")};
};

}  // namespace
}  // namespace mlir

auto cxx::createConvertCxxKernelToGpuPass(llvm::StringRef kernel)
    -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<mlir::ConvertCxxKernelToGpuPass>(kernel);
}

void cxx::registerConvertCxxKernelToGpuPass() {
  mlir::registerPass([] { return createConvertCxxKernelToGpuPass(""); });
}
