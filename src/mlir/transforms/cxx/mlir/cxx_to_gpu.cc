#include <cxx/device_profile_limits.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/kernel_contract.h>
#include <cxx/mlir/passes.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Demangle/Demangle.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
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

[[nodiscard]] auto isSupportedElementType(Type type) -> bool {
  if (type.isF32() || type.isIndex()) return true;
  auto integer = dyn_cast<IntegerType>(type);
  return integer && (integer.getWidth() == 32 || integer.getWidth() == 1);
}

[[nodiscard]] auto isSupportedScalarType(Type type) -> bool {
  if (auto vector = dyn_cast<VectorType>(type))
    return vector.getRank() == 1 && !vector.isScalable() &&
           vector.getNumElements() >= 2 && vector.getNumElements() <= 4 &&
           !vector.getElementType().isIndex() &&
           isSupportedElementType(vector.getElementType());
  return isSupportedElementType(type);
}

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
 public:
  [[nodiscard]] auto collect(cxx::FuncOp entry) -> LogicalResult {
    entry_ = entry;
    return visit(entry);
  }

  [[nodiscard]] auto callees() const -> ArrayRef<cxx::FuncOp> {
    return callees_;
  }

 private:
  [[nodiscard]] static auto displayName(cxx::FuncOp function) -> std::string {
    return llvm::demangle(function.getSymName().str());
  }

  [[nodiscard]] auto visit(cxx::FuncOp function) -> LogicalResult {
    active_.insert(function);
    auto result = function.walk([&](CallOpInterface call) {
      return succeeded(visitCall(call)) ? WalkResult::advance()
                                        : WalkResult::interrupt();
    });
    active_.erase(function);
    if (result.wasInterrupted()) return failure();
    visited_.insert(function);
    if (function != entry_) callees_.push_back(function);
    return success();
  }

  [[nodiscard]] auto visitCall(CallOpInterface call) -> LogicalResult {
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
    if (callee.getFunctionType().getVariadic())
      return call->emitError("device function '")
             << displayName(callee) << "' is variadic";
    if (active_.contains(callee))
      return call->emitError("device function '")
             << displayName(callee) << "' is called recursively";
    if (visited_.contains(callee)) return success();
    return visit(callee);
  }

  cxx::FuncOp entry_;
  SymbolTableCollection symbols_;
  DenseSet<Operation*> active_;
  DenseSet<Operation*> visited_;
  SmallVector<cxx::FuncOp> callees_;
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
    auto it = addresses_.find(pointer);
    if (it != addresses_.end()) return it->second;
    op->emitError("device access has no declared resource root");
    return failure();
  }

  [[nodiscard]] auto index(Value value, Location loc) -> Value {
    if (value.getType().isIndex()) return value;
    return arith::IndexCastOp::create(builder_, loc, builder_.getIndexType(),
                                      value);
  }

  [[nodiscard]] auto lower(Operation* op) -> LogicalResult {
    if (auto addition = dyn_cast<cxx::PtrAddOp>(op)) {
      auto base = address(addition.getBase(), op);
      if (failed(base)) return failure();
      if (isUniformResource(base->buffer))
        return op->emitError(
            "device pointer arithmetic leaves its uniform resource");
      auto pointer = cast<cxx::PointerType>(addition.getType());
      auto memory = cast<MemRefType>(base->buffer.getType());
      if (!isResourcePointer(pointer) ||
          pointer.getElementType() != memory.getElementType())
        return op->emitError(
            "device pointer addition changes its resource element type");
      auto offset = index(addition.getOffset(), op->getLoc());
      auto total =
          arith::AddIOp::create(builder_, op->getLoc(), base->offset, offset);
      addresses_[addition.getResult()] = {base->buffer, total, base->readOnly};
      obsolete_.push_back(op);
      return success();
    }
    if (auto load = dyn_cast<cxx::LoadOp>(op)) {
      auto root = address(load.getAddr(), op);
      if (failed(root)) return failure();
      auto value = memref::LoadOp::create(builder_, op->getLoc(), root->buffer,
                                          resourceIndices(*root));
      if (value.getType() != load.getType())
        return op->emitError("device resource load changes its element type");
      load.getResult().replaceAllUsesWith(value.getResult());
      obsolete_.push_back(op);
      return success();
    }
    if (auto store = dyn_cast<cxx::StoreOp>(op)) {
      auto root = address(store.getAddr(), op);
      if (failed(root)) return failure();
      if (root->readOnly)
        return op->emitError("cannot write a read-only device resource");
      memref::StoreOp::create(builder_, op->getLoc(), store.getValue(),
                              root->buffer, resourceIndices(*root));
      obsolete_.push_back(op);
      return success();
    }
    if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
      auto type = allocation.getType();
      if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
          type.getMemorySpace() ||
          !isSupportedScalarType(type.getElementType()))
        return op->emitError("unsupported device local storage type ") << type;
      allocation.getResult().setType(MemRefType::get(
          type.getShape(), type.getElementType(), MemRefLayoutAttrInterface(),
          gpu::AddressSpaceAttr::get(op->getContext(),
                                     gpu::AddressSpace::Private)));
      private_.insert(allocation.getResult());
      return success();
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (!private_.contains(load.getMemRef()))
        return op->emitError("device load has no declared storage root");
      return success();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (!private_.contains(store.getMemRef()))
        return op->emitError("device store has no declared storage root");
      return success();
    }
    if (auto cast = dyn_cast<cxx::BitcastOp>(op)) {
      auto source = cast.getValue().getType();
      auto target = cast.getType();
      if (!isSupportedScalarType(source) || !isSupportedScalarType(target) ||
          !arith::BitcastOp::areCastCompatible(TypeRange{source},
                                               TypeRange{target}))
        return op->emitError("unsupported device reinterpretation from ")
               << source << " to " << target;
      auto bitcast = arith::BitcastOp::create(builder_, op->getLoc(), target,
                                              cast.getValue());
      cast.getResult().replaceAllUsesWith(bitcast.getResult());
      obsolete_.push_back(op);
      return success();
    }
    if (auto undef = dyn_cast<cxx::UndefOp>(op)) {
      if (!isSupportedScalarType(undef.getType()))
        return op->emitError("unsupported device result type ")
               << undef.getType();
      auto poison =
          ub::PoisonOp::create(builder_, op->getLoc(), undef.getType());
      undef.getResult().replaceAllUsesWith(poison.getResult());
      obsolete_.push_back(op);
      return success();
    }
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      if (ret.getNumOperands())
        return op->emitError("device entry must return void");
      gpu::ReturnOp::create(builder_, op->getLoc());
      obsolete_.push_back(op);
      return success();
    }
    auto supported =
        isa<arith::ArithDialect, math::MathDialect>(op->getDialect()) ||
        isa<ub::PoisonOp, scf::IfOp, scf::WhileOp, scf::ForOp,
            scf::IndexSwitchOp, scf::ConditionOp, scf::YieldOp,
            vector::BroadcastOp, vector::ExtractOp, vector::InsertOp,
            vector::FromElementsOp, vector::ShuffleOp, gpu::GlobalIdOp,
            gpu::ThreadIdOp, gpu::BlockIdOp>(op);
    if (!supported)
      return op->emitError("unsupported operation '")
             << op->getName() << "' in the selected device entry "
             << function_.getName();
    for (Type type : op->getOperandTypes())
      if (!isSupportedScalarType(type))
        return op->emitError("unsupported device operand type ") << type;
    for (Type type : op->getResultTypes())
      if (!isSupportedScalarType(type))
        return op->emitError("unsupported device result type ") << type;
    for (auto& region : op->getRegions())
      for (auto& block : region)
        for (auto argument : block.getArguments())
          if (!isSupportedScalarType(argument.getType()))
            return op->emitError("unsupported device region argument type");
    return success();
  }

  gpu::GPUFuncOp function_;
  OpBuilder builder_;
  DenseMap<Value, ResourceAddress> addresses_;
  DenseSet<Value> private_;
  SmallVector<Operation*> obsolete_;
};

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
    func::registerInlinerExtension(registry);
  }

  void runOnOperation() override {
    if (failed(convert())) signalPassFailure();
  }

 private:
  [[nodiscard]] auto convert() -> LogicalResult {
    auto module = getOperation();
    auto profile = module->getAttrOfType<StringAttr>("cxx.device_profile");
    if (!profile || profile.getValue() !=
                        "vulkan1.1-f32/as1-device-as2-constant/index32/v2")
      return module.emitError(
          "kernel conversion requires the checked Vulkan device profile");
    auto entry = module.lookupSymbol<cxx::FuncOp>(kernel.getValue());
    if (!entry)
      return module.emitError("selected kernel was not found: ")
             << kernel.getValue();
    auto contract = cxx::kernelContractOf(entry);
    if (failed(contract)) return failure();
    const auto& resources = contract->buffers;
    DeviceCallGraph callGraph;
    if (failed(callGraph.collect(entry))) return failure();
    OwningOpRef<ModuleOp> temporary = ModuleOp::create(module.getLoc());
    (*temporary)->setAttrs(module->getAttrs());
    temporary->getBody()->push_back(entry->clone());
    for (auto callee : callGraph.callees()) {
      auto clone = cast<cxx::FuncOp>(callee->clone());
      clone.removeInlineKindAttr();
      SymbolTable::setSymbolVisibility(clone, SymbolTable::Visibility::Private);
      temporary->getBody()->push_back(clone);
    }
    cloneReferencedGlobals(module, *temporary);
    temporary->walk(
        [](Operation* op) { op->removeDiscardableAttr("cxx.di_local"); });
    if (failed(lowerDeviceStorage(*temporary))) return failure();
    auto device = temporary->lookupSymbol<func::FuncOp>(kernel.getValue());
    if (!device)
      return temporary->emitError("selected kernel was not lowered to func");
    if (!llvm::hasSingleElement(device.getBody()))
      return device.emitError("device entry retains unstructured control flow");
    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(device);
    auto gpuModule =
        gpu::GPUModuleOp::create(builder, device.getLoc(), "kernels");
    builder.setInsertionPointToStart(gpuModule.getBody());
    SmallVector<Type> arguments;
    for (const auto& resource : resources)
      arguments.push_back(memrefTypeOf(module.getContext(), resource));
    auto function =
        gpu::GPUFuncOp::create(builder, device.getLoc(), kernel.getValue(),
                               builder.getFunctionType(arguments, {}));
    function->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                      builder.getUnitAttr());
    function->setAttr(
        spirv::getEntryPointABIAttrName(),
        spirv::getEntryPointABIAttr(module.getContext(), contract->workgroup));
    IRMapping mapping;
    for (auto [resource, newArgument] :
         llvm::zip(resources, function.getArguments()))
      mapping.map(device.getArgument(resource.parameter), newArgument);
    builder.setInsertionPointToStart(&function.getBody().front());
    for (const auto& id : contract->invocationIds) {
      auto argument = device.getArgument(id.parameter);
      mapping.map(argument,
                  materializeInvocationId(builder, argument.getLoc(), id.kind,
                                          argument.getType()));
    }
    for (auto [index, resource] : llvm::enumerate(resources))
      function.setArgAttr(index, spirv::getInterfaceVarABIAttrName(),
                          spirv::getInterfaceVarABIAttr(
                              ::cxx::kBufferDescriptorSet, resource.binding, {},
                              module.getContext()));
    builder.setInsertionPointToEnd(&function.getBody().front());
    for (auto& op : device.getBody().front()) builder.clone(op, mapping);
    if (failed(ResourceLowering(function, resources).run())) return failure();
    device.erase();
    (*temporary)
        ->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  builder.getUnitAttr());
    if (failed(verify(*temporary))) return failure();
    module.getBodyRegion().takeBody(temporary->getBodyRegion());
    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    builder.getUnitAttr());
    return success();
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

  [[nodiscard]] static auto countLocalObjects(ModuleOp module) -> unsigned {
    unsigned count = 0;
    module.walk([&](Operation* op) {
      if (isa<cxx::AllocaOp, cxx::AddressOfOp>(op)) ++count;
    });
    return count;
  }

  [[nodiscard]] auto lowerDeviceStorage(ModuleOp module) -> LogicalResult {
    auto context = module.getContext();
    PassManager calls(context);
    calls.addPass(::cxx::createConvertCxxToFuncPass());
    calls.addPass(createInlinerPass());
    calls.addPass(createSymbolDCEPass());
    if (failed(calls.run(module))) return failure();
    for (auto remaining = countLocalObjects(module);;) {
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
      auto current = countLocalObjects(module);
      if (current == 0 || current >= remaining) break;
      remaining = current;
    }
    PassManager cleanup(context);
    cleanup.addPass(createSROA());
    cleanup.addPass(createMem2Reg());
    cleanup.addPass(createCanonicalizerPass());
    cleanup.addPass(::cxx::createLiftCxxControlFlowToSCFPass());
    cleanup.addPass(createCanonicalizerPass());
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
