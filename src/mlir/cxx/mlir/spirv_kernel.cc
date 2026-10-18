#include <cxx/device_profile_limits.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/device_kernel.h>
#include <cxx/mlir/kernel_manifest.h>
#include <cxx/mlir/passes.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/SHA256.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h>
#include <mlir/Conversion/UBToSPIRV/UBToSPIRV.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/SPIRV/Deserialization.h>
#include <mlir/Target/SPIRV/Serialization.h>

#include <optional>
#include <string>

namespace cxx {
namespace {

using namespace mlir;

[[nodiscard]] auto checkEntry(spirv::ModuleOp module, StringRef kernel,
                              ArrayRef<std::int32_t> workgroup)
    -> LogicalResult {
  if (module.getAddressingModel() != spirv::AddressingModel::Logical ||
      module.getMemoryModel() != spirv::MemoryModel::GLSL450)
    return module.emitError("unexpected SPIR-V device execution model");
  auto entries = module.getOps<spirv::EntryPointOp>();
  if (!llvm::hasSingleElement(entries))
    return module.emitError("expected one SPIR-V entry point");
  auto entry = *entries.begin();
  if (entry.getExecutionModel() != spirv::ExecutionModel::GLCompute ||
      entry.getFnAttr().getValue() != kernel)
    return entry.emitError(
        "SPIR-V entry point differs from its source contract");
  auto modes = module.getOps<spirv::ExecutionModeOp>();
  if (!llvm::hasSingleElement(modes))
    return module.emitError("expected one SPIR-V local size execution mode");
  auto mode = *modes.begin();
  if (mode.getExecutionMode() != spirv::ExecutionMode::LocalSize ||
      mode.getFnAttr().getValue() != kernel || mode.getValues().size() != 3)
    return mode.emitError("unexpected SPIR-V execution mode");
  for (auto [value, size] : llvm::zip(mode.getValues(), workgroup))
    if (cast<IntegerAttr>(value).getInt() != size)
      return mode.emitError(
          "SPIR-V local size differs from its source contract");
  auto vce = module.getVceTripleAttr();
  if (!vce || vce.getVersion() > spirv::Version::V_1_3)
    return module.emitError("SPIR-V version exceeds the Vulkan 1.1 profile");
  for (auto capability : vce.getCapabilities())
    if (capability != spirv::Capability::Shader &&
        capability != spirv::Capability::Matrix)
      return module.emitError("SPIR-V capability exceeds the device profile");
  for (auto extension : vce.getExtensions())
    if (extension != spirv::Extension::SPV_KHR_storage_buffer_storage_class)
      return module.emitError("SPIR-V extension exceeds the device profile");
  return success();
}

[[nodiscard]] auto variableFor(spirv::ModuleOp module,
                               const mlir::cxx::KernelBuffer& buffer)
    -> FailureOr<spirv::GlobalVariableOp> {
  spirv::GlobalVariableOp match;
  for (auto variable : module.getOps<spirv::GlobalVariableOp>()) {
    if (variable.getDescriptorSet() != ::cxx::kBufferDescriptorSet ||
        variable.getBinding() != buffer.binding)
      continue;
    if (match) return failure();
    match = variable;
  }
  if (!match) return failure();
  return match;
}

[[nodiscard]] auto resourceStorageClass(const mlir::cxx::KernelBuffer& buffer)
    -> spirv::StorageClass {
  if (buffer.kind == mlir::cxx::KernelBufferKind::Uniform)
    return spirv::StorageClass::Uniform;
  return spirv::StorageClass::StorageBuffer;
}

[[nodiscard]] auto isWritableBlock(spirv::StructType block) -> bool {
  SmallVector<spirv::StructType::MemberDecorationInfo> decorations;
  block.getMemberDecorations(0, decorations);
  return llvm::none_of(decorations, [](const auto& decoration) {
    return decoration.decoration == spirv::Decoration::NonWritable;
  });
}

[[nodiscard]] auto checkBlockMember(Type member,
                                    const mlir::cxx::KernelBuffer& buffer)
    -> bool {
  if (buffer.kind == mlir::cxx::KernelBufferKind::Uniform)
    return member == buffer.elementType;
  auto array = dyn_cast<spirv::RuntimeArrayType>(member);
  return array && array.getArrayStride() == 4 &&
         array.getElementType() == buffer.elementType;
}

[[nodiscard]] auto checkBuffer(spirv::GlobalVariableOp variable,
                               const mlir::cxx::KernelBuffer& buffer,
                               bool serialized) -> LogicalResult {
  auto pointer = dyn_cast<spirv::PointerType>(variable.getType());
  if (!pointer || pointer.getStorageClass() != resourceStorageClass(buffer))
    return variable.emitError(
        "resource storage class differs from its source contract");
  auto block = dyn_cast<spirv::StructType>(pointer.getPointeeType());
  if (!block || block.getNumElements() != 1 || !block.hasOffset() ||
      block.getMemberOffset(0) != 0)
    return variable.emitError(
        "resource block layout differs from its source contract");
  if (serialized && !block.hasDecoration(spirv::Decoration::Block))
    return variable.emitError("serialized resource lacks Block decoration");
  if (!checkBlockMember(block.getElementType(0), buffer))
    return variable.emitError(
        "resource member layout differs from its source contract");
  if (buffer.kind == mlir::cxx::KernelBufferKind::Uniform) return success();
  if (variable->hasAttr("non_writable") != buffer.readOnly ||
      isWritableBlock(block) == buffer.readOnly)
    return variable.emitError(
        "resource access decoration differs from its source contract");
  return success();
}

[[nodiscard]] auto checkBuffers(spirv::ModuleOp module,
                                ArrayRef<mlir::cxx::KernelBuffer> buffers,
                                bool serialized) -> LogicalResult {
  unsigned resources = 0;
  for (auto variable : module.getOps<spirv::GlobalVariableOp>()) {
    switch (cast<spirv::PointerType>(variable.getType()).getStorageClass()) {
      case spirv::StorageClass::StorageBuffer:
      case spirv::StorageClass::Uniform:
        ++resources;
        break;
      case spirv::StorageClass::Input:
        break;
      default:
        return variable.emitError(
            "unexpected resource storage class in the device profile");
    }
  }
  if (resources != buffers.size())
    return module.emitError(
        "SPIR-V resource count differs from its source contract");
  for (const auto& buffer : buffers) {
    auto variable = variableFor(module, buffer);
    if (failed(variable))
      return module.emitError(
          "SPIR-V descriptor binding differs from its source contract");
    if (failed(checkBuffer(*variable, buffer, serialized))) return failure();
  }
  return success();
}

[[nodiscard]] auto uniformValueAccess(spirv::AccessChainOp access)
    -> FailureOr<spirv::AccessChainOp> {
  auto indices = access.getIndices();
  if (indices.size() != 2 || !matchPattern(indices[1], m_Zero()))
    return access.emitError("uniform resource access differs from its block");
  OpBuilder builder(access);
  auto replacement = spirv::AccessChainOp::create(
      builder, access.getLoc(), access.getBasePtr(), ValueRange{indices[0]});
  access.getResult().replaceAllUsesWith(replacement.getResult());
  access.erase();
  return replacement;
}

[[nodiscard]] auto addressesOf(spirv::ModuleOp module,
                               spirv::GlobalVariableOp variable)
    -> SmallVector<spirv::AddressOfOp> {
  SmallVector<spirv::AddressOfOp> addresses;
  module.walk([&](spirv::AddressOfOp address) {
    if (address.getVariable() == variable.getSymName())
      addresses.push_back(address);
  });
  return addresses;
}

[[nodiscard]] auto rebaseUniformAccesses(spirv::AddressOfOp address)
    -> LogicalResult {
  for (auto* user : llvm::make_early_inc_range(address->getUsers())) {
    auto access = dyn_cast<spirv::AccessChainOp>(user);
    if (!access)
      return user->emitError("uniform resource escapes its value access");
    if (failed(uniformValueAccess(access))) return failure();
  }
  return success();
}

[[nodiscard]] auto blockValueType(spirv::GlobalVariableOp variable,
                                  const mlir::cxx::KernelBuffer& buffer)
    -> FailureOr<Type> {
  auto pointer = cast<spirv::PointerType>(variable.getType());
  auto block = dyn_cast<spirv::StructType>(pointer.getPointeeType());
  if (!block || block.getNumElements() != 1)
    return variable.emitError(
        "resource block layout differs from its source contract");
  auto member = block.getElementType(0);
  if (buffer.kind != mlir::cxx::KernelBufferKind::Uniform) return member;
  auto array = dyn_cast<spirv::ArrayType>(member);
  if (!array || array.getNumElements() != 1)
    return variable.emitError(
        "uniform resource layout differs from its source contract");
  return array.getElementType();
}

[[nodiscard]] auto isReadOnlyStorage(const mlir::cxx::KernelBuffer& buffer)
    -> bool {
  return buffer.readOnly && buffer.kind == mlir::cxx::KernelBufferKind::Storage;
}

[[nodiscard]] auto blockDecorations(MLIRContext* context,
                                    const mlir::cxx::KernelBuffer& buffer)
    -> SmallVector<spirv::StructType::MemberDecorationInfo> {
  SmallVector<spirv::StructType::MemberDecorationInfo> decorations;
  if (isReadOnlyStorage(buffer))
    decorations.emplace_back(0, spirv::Decoration::NonWritable,
                             UnitAttr::get(context));
  return decorations;
}

[[nodiscard]] auto layoutResourceBlocks(
    spirv::ModuleOp module, ArrayRef<mlir::cxx::KernelBuffer> buffers)
    -> LogicalResult {
  auto* context = module.getContext();
  for (const auto& buffer : buffers) {
    auto variable = variableFor(module, buffer);
    if (failed(variable))
      return module.emitError(
          "SPIR-V descriptor binding differs from its source contract");
    auto member = blockValueType(*variable, buffer);
    if (failed(member)) return failure();
    auto block = spirv::StructType::get({*member}, {0},
                                        blockDecorations(context, buffer));
    auto type = spirv::PointerType::get(block, resourceStorageClass(buffer));
    for (auto address : addressesOf(module, *variable)) {
      address.getPointer().setType(type);
      if (buffer.kind == mlir::cxx::KernelBufferKind::Uniform &&
          failed(rebaseUniformAccesses(address)))
        return failure();
    }
    variable->setType(type);
    if (isReadOnlyStorage(buffer))
      (*variable)->setAttr("non_writable", UnitAttr::get(context));
  }
  return success();
}

#if LLVM_VERSION_MAJOR < 23
[[nodiscard]] auto constructNeedingNewerMLIR(Operation* op)
    -> std::optional<std::string> {
  if (isa<scf::IndexSwitchOp>(op)) return "a multiway switch";
  if (isa<math::TruncOp, math::Atan2Op, math::Exp2Op>(op))
    return "'" + op->getName().stripDialect().str() + "'";
  return std::nullopt;
}
#endif

[[nodiscard]] auto checkLowerableOperations(ModuleOp module) -> LogicalResult {
#if LLVM_VERSION_MAJOR < 23
  auto result = module.walk([](Operation* op) {
    auto construct = constructNeedingNewerMLIR(op);
    if (!construct) return WalkResult::advance();
    op->emitError(*construct +
                  " needs MLIR 23 to lower to SPIR-V; WGSL output supports it");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
#else
  return success();
#endif
}

[[nodiscard]] auto spirvTarget(spirv::ModuleOp module) -> llvm::json::Object {
  llvm::json::Array capabilities, extensions;
  auto vce = module.getVceTripleAttr();
  for (auto capability : vce.getCapabilities())
    capabilities.push_back(spirv::stringifyCapability(capability));
  for (auto extension : vce.getExtensions())
    extensions.push_back(spirv::stringifyExtension(extension));
  return llvm::json::Object{
      {"targetEnvironment", "vulkan1.1"},
      {"addressingModel", "Logical"},
      {"memoryModel", "GLSL450"},
      {"spirv", llvm::json::Object{
                    {"version", spirv::stringifyVersion(vce.getVersion())},
                    {"capabilities", std::move(capabilities)},
                    {"extensions", std::move(extensions)}}}};
}

[[nodiscard]] auto storageClassOf(Attribute space)
    -> std::optional<spirv::StorageClass> {
  if (auto storage = dyn_cast_or_null<spirv::StorageClassAttr>(space))
    return storage.getValue();
  auto gpuSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(space);
  if (!gpuSpace) return spirv::mapMemorySpaceToVulkanStorageClass(space);
  switch (gpuSpace.getValue()) {
    case gpu::AddressSpace::Private:
      return spirv::StorageClass::Function;
    case gpu::AddressSpace::Workgroup:
      return spirv::StorageClass::Workgroup;
    case gpu::AddressSpace::Global:
      return spirv::StorageClass::StorageBuffer;
  }
  return std::nullopt;
}

void mapStorageClasses(ModuleOp module) {
  spirv::MemorySpaceToStorageClassConverter converter(storageClassOf);
  module.walk([&](gpu::GPUModuleOp gpuModule) {
    spirv::convertMemRefTypesAndAttrs(gpuModule, converter);
  });
}

void vectorizeSelectConditions(ModuleOp module) {
  module.walk([](arith::SelectOp select) {
    auto vector = dyn_cast<VectorType>(select.getType());
    if (!vector || isa<VectorType>(select.getCondition().getType())) return;
    OpBuilder builder(select);
    auto condition = vector::BroadcastOp::create(
        builder, select.getLoc(),
        VectorType::get(vector.getShape(), builder.getI1Type()),
        select.getCondition());
    select.getConditionMutable().assign(condition);
  });
}

}  // namespace

auto lowerKernelToSPIRV(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<mlir::OwningOpRef<mlir::Operation*>> {
  using namespace mlir;
  auto entry = kernelEntryOf(source, kernel);
  if (failed(entry)) return failure();
  const auto& buffers = entry->contract.buffers;
  const auto& workgroup = entry->contract.workgroup;
  auto working = lowerKernelToGPU(source, kernel);
  if (failed(working) || failed(checkLowerableOperations(working->get())))
    return failure();
  auto* context = source.getContext();
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_3, {spirv::Capability::Shader}, {}, context);
  working->get()->setAttr(
      spirv::getTargetEnvAttrName(),
      spirv::TargetEnvAttr::get(triple,
                                spirv::getDefaultResourceLimits(context)));
  mapStorageClasses(working->get());
  vectorizeSelectConditions(working->get());
  PassManager lowering(context);
  lowering.addPass(memref::createFoldMemRefAliasOpsPass());
  lowering.addPass(createUBToSPIRVConversionPass());
  lowering.addPass(createConvertMathToSPIRVPass());
  auto gpuConversion = createConvertGPUToSPIRVPass(/*mapMemorySpace=*/false);
  if (failed(gpuConversion->initializeOptions(
          "use-64bit-index=false",
          [&](const llvm::Twine& message) -> LogicalResult {
            return source.emitError(message);
          })))
    return failure();
  lowering.addPass(std::move(gpuConversion));
  lowering.addNestedPass<spirv::ModuleOp>(
      spirv::createSPIRVLowerABIAttributesPass());
  lowering.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
  if (failed(lowering.run(working->get()))) return failure();
  auto modules = working->get().getOps<spirv::ModuleOp>();
  if (!llvm::hasSingleElement(modules)) {
    source.emitError("expected one converted SPIR-V module");
    return failure();
  }
  auto module = *modules.begin();
  if (failed(checkEntry(module, kernel, workgroup)) ||
      failed(layoutResourceBlocks(module, buffers)) ||
      failed(checkBuffers(module, buffers, false)) || failed(verify(module)))
    return failure();
  return OwningOpRef<Operation*>(module->clone());
}

auto compileSPIRVKernel(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact> {
  using namespace mlir;
  auto entry = kernelEntryOf(source, kernel);
  if (failed(entry)) return failure();
  const auto& buffers = entry->contract.buffers;
  const auto& workgroup = entry->contract.workgroup;
  auto lowered = lowerKernelToSPIRV(source, kernel);
  if (failed(lowered)) return failure();
  auto module = cast<spirv::ModuleOp>(lowered->get());
  auto* context = source.getContext();
  SmallVector<uint32_t> words;
  if (failed(spirv::serialize(module, words))) return failure();
  auto decoded = spirv::deserialize(words, context);
  if (!decoded || failed(verify(*decoded)) ||
      failed(checkEntry(*decoded, kernel, workgroup)) ||
      failed(checkBuffers(*decoded, buffers, true)))
    return failure();
  KernelArtifact artifact;
  artifact.bytes.resize(words.size() * 4);
  for (auto [index, word] : llvm::enumerate(words))
    llvm::support::endian::write32le(artifact.bytes.data() + index * 4, word);
  artifact.manifest =
      kernelManifest(*entry, "spirv", artifact.bytes, spirvTarget(*decoded));
  return artifact;
}

}  // namespace cxx
