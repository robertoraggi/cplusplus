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
#include <cxx/mlir/buffer_layout.h>
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/device_kernel.h>
#include <cxx/mlir/kernel_contract.h>
#include <cxx/mlir/kernel_manifest.h>
#include <cxx/mlir/passes.h>
#include <cxx/mlir/spirv_buffer_layout.h>
#include <cxx/mlir/webgpu_policy.h>
#include <cxx/spirv_environment.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/SHA256.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/SPIRV/Deserialization.h>
#include <mlir/Target/SPIRV/Serialization.h>

#include <optional>
#include <string>

namespace cxx {
namespace {

using namespace mlir;

[[nodiscard]] auto spirvVersionOf(SPIRVEnvironment environment)
    -> spirv::Version {
  switch (environment) {
    case SPIRVEnvironment::kVulkan1_1:
    case SPIRVEnvironment::kWebGPU:
      return spirv::Version::V_1_3;
    case SPIRVEnvironment::kVulkan1_2:
      return spirv::Version::V_1_5;
    case SPIRVEnvironment::kVulkan1_3:
    case SPIRVEnvironment::kVulkan1_4:
      return spirv::Version::V_1_6;
  }
  return spirv::Version::V_1_3;
}

[[nodiscard]] auto checkEntry(spirv::ModuleOp module,
                              const mlir::cxx::KernelModuleContract& contract,
                              SPIRVEnvironment environment) -> LogicalResult {
  if (module.getAddressingModel() != spirv::AddressingModel::Logical ||
      module.getMemoryModel() != spirv::MemoryModel::GLSL450)
    return module.emitError("unexpected SPIR-V device execution model");
  auto entries = module.getOps<spirv::EntryPointOp>();
  auto modes = module.getOps<spirv::ExecutionModeOp>();
  if (llvm::range_size(entries) != contract.entries.size())
    return module.emitError(
        "SPIR-V entry points differ from the source contract");
  if (llvm::range_size(modes) != contract.entries.size())
    return module.emitError(
        "SPIR-V execution modes differ from the source contract");
  for (const auto& expected : contract.entries) {
    unsigned entryCount = 0, modeCount = 0;
    for (auto entry : entries) {
      if (entry.getFnAttr().getValue() != expected.name) continue;
      if (entry.getExecutionModel() != spirv::ExecutionModel::GLCompute)
        return entry.emitError("unexpected SPIR-V entry execution model");
      ++entryCount;
    }
    for (auto mode : modes) {
      if (mode.getFnAttr().getValue() != expected.name) continue;
      if (mode.getExecutionMode() != spirv::ExecutionMode::LocalSize ||
          mode.getValues().size() != 3)
        return mode.emitError("unexpected SPIR-V execution mode");
      for (auto [value, size] :
           llvm::zip(mode.getValues(), expected.contract.workgroup))
        if (cast<IntegerAttr>(value).getInt() != size)
          return mode.emitError(
              "SPIR-V local size differs from its source contract");
      ++modeCount;
    }
    if (entryCount != 1 || modeCount != 1)
      return module.emitError(
                 "SPIR-V entry interface differs from its source contract: ")
             << expected.name;
  }
  auto vce = module.getVceTripleAttr();
  if (!vce || vce.getVersion() != spirvVersionOf(environment))
    return module.emitError("SPIR-V version differs from the '")
           << to_string(environment) << "' environment";
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
                               const mlir::cxx::KernelResource& buffer)
    -> FailureOr<spirv::GlobalVariableOp> {
  auto variable =
      module.lookupSymbol<spirv::GlobalVariableOp>(buffer.name.getValue());
  if (!variable || variable.getDescriptorSet() != ::cxx::kBufferDescriptorSet ||
      variable.getBinding() != buffer.binding)
    return failure();
  return variable;
}

[[nodiscard]] auto resourceStorageClass(const mlir::cxx::KernelResource& buffer)
    -> spirv::StorageClass {
  if (buffer.kind == mlir::cxx::KernelBufferKind::Uniform)
    return spirv::StorageClass::Uniform;
  return spirv::StorageClass::StorageBuffer;
}

class SPIRVResourceABI {
  struct Argument {
    BlockArgument value;
    unsigned resource;
  };
  struct Function {
    spirv::FuncOp operation;
    SmallVector<Argument> arguments;
    llvm::BitVector erased;
  };

 public:
  SPIRVResourceABI(spirv::ModuleOp module,
                   mlir::cxx::KernelModuleContract& contract)
      : module_(module),
        contract_(contract),
        types_(contract.resources.size()) {}

  [[nodiscard]] auto lower()
      -> FailureOr<SmallVector<spirv::GlobalVariableOp>> {
    for (auto function : module_.getOps<spirv::FuncOp>())
      if (failed(collect(function))) return failure();
    if (llvm::any_of(types_, [](Type type) { return !type; }))
      return module_.emitError("device module has an undeclared resource");

    SymbolTable symbols(module_);
    OpBuilder builder(module_.getBody(), module_.getBody()->begin());
    SmallVector<spirv::GlobalVariableOp> variables;
    for (auto [resource, type] : llvm::zip(contract_.resources, types_)) {
      auto variable = spirv::GlobalVariableOp::create(
          builder, module_.getLoc(), type, resource.name.getValue(),
          ::cxx::kBufferDescriptorSet, resource.binding);
      symbols.insert(variable);
      resource.name = variable.getSymNameAttr();
      variables.push_back(variable);
    }
    for (auto& function : functions_) {
      builder.setInsertionPointToStart(&function.operation.getBody().front());
      for (auto argument : function.arguments) {
        auto address = spirv::AddressOfOp::create(
            builder, function.operation.getLoc(), variables[argument.resource]);
        argument.value.replaceAllUsesWith(address);
      }
      module_.walk([&](spirv::FunctionCallOp call) {
        if (call.getCallee() == function.operation.getSymName())
          call->eraseOperands(function.erased);
      });
      if (failed(function.operation.eraseArguments(function.erased)))
        return function.operation.emitError(
            "cannot lower device resource arguments");
    }
    return variables;
  }

 private:
  [[nodiscard]] auto collect(spirv::FuncOp function) -> LogicalResult {
    if (function.isExternal())
      return function.emitError("device functions require definitions");
    Function plan{function, {}, llvm::BitVector(function.getNumArguments())};
    for (auto argument : function.getArguments()) {
      auto attribute = mlir::cxx::deviceResourceBinding(argument);
      if (!attribute) continue;
      auto binding = dyn_cast<spirv::InterfaceVarABIAttr>(attribute);
      if (!binding)
        return function.emitError("invalid device resource binding");
      auto index = mlir::cxx::deviceResourceIndex(argument);
      if (!index || index.getInt() < 0 ||
          std::uint64_t(index.getInt()) >= types_.size())
        return function.emitError("device resource has no module identity");
      auto resource = unsigned(index.getInt());
      const auto& expected = contract_.resources[resource];
      if (binding.getDescriptorSet() != ::cxx::kBufferDescriptorSet ||
          binding.getBinding() != expected.binding)
        return function.emitError(
            "device resource binding differs from its source contract");
      auto& type = types_[resource];
      if (type && type != argument.getType())
        return function.emitError("shared device resource types differ");
      type = argument.getType();
      plan.arguments.push_back({argument, resource});
      plan.erased.set(argument.getArgNumber());
    }
    if (!plan.arguments.empty()) functions_.push_back(std::move(plan));
    return success();
  }

  spirv::ModuleOp module_;
  mlir::cxx::KernelModuleContract& contract_;
  SmallVector<Type> types_;
  SmallVector<Function> functions_;
};

[[nodiscard]] auto isWritableBlock(spirv::StructType block) -> bool {
  SmallVector<spirv::StructType::MemberDecorationInfo> decorations;
  block.getMemberDecorations(0, decorations);
  return llvm::none_of(decorations, [](const auto& decoration) {
    return decoration.decoration == spirv::Decoration::NonWritable;
  });
}

[[nodiscard]] auto isRecordBlock(const mlir::cxx::KernelResource& buffer)
    -> bool {
  return buffer.kind == mlir::cxx::KernelBufferKind::Uniform &&
         mlir::cxx::isBufferRecordType(buffer.elementType);
}

[[nodiscard]] auto checkBlockMember(Type member,
                                    const mlir::cxx::KernelResource& buffer)
    -> bool {
  if (buffer.kind == mlir::cxx::KernelBufferKind::Uniform)
    return mlir::cxx::hasSPIRVBufferLayout(member, buffer.elementType);
  auto array = dyn_cast<spirv::RuntimeArrayType>(member);
  auto stride = mlir::cxx::kernelBufferElementStride(buffer);
  return array && array.getArrayStride() == stride &&
         mlir::cxx::hasSPIRVBufferLayout(array.getElementType(),
                                         buffer.elementType);
}

[[nodiscard]] auto hasBlockLayout(spirv::StructType block,
                                  const mlir::cxx::KernelResource& buffer)
    -> bool {
  if (isRecordBlock(buffer))
    return mlir::cxx::hasSPIRVBufferLayout(block, buffer.elementType);
  return block.getNumElements() == 1 && block.hasOffset() &&
         block.getMemberOffset(0) == 0 &&
         checkBlockMember(block.getElementType(0), buffer);
}

[[nodiscard]] auto checkBuffer(spirv::GlobalVariableOp variable,
                               const mlir::cxx::KernelResource& buffer,
                               bool serialized) -> LogicalResult {
  auto pointer = dyn_cast<spirv::PointerType>(variable.getType());
  if (!pointer || pointer.getStorageClass() != resourceStorageClass(buffer))
    return variable.emitError(
        "resource storage class differs from its source contract");
  auto block = dyn_cast<spirv::StructType>(pointer.getPointeeType());
  if (!block)
    return variable.emitError(
        "resource block layout differs from its source contract");
  if (serialized && !block.hasDecoration(spirv::Decoration::Block))
    return variable.emitError("serialized resource lacks Block decoration");
  if (!hasBlockLayout(block, buffer))
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
                                ArrayRef<mlir::cxx::KernelResource> buffers,
                                bool serialized) -> LogicalResult {
  unsigned resources = 0;
  for (auto variable : module.getOps<spirv::GlobalVariableOp>()) {
    switch (cast<spirv::PointerType>(variable.getType()).getStorageClass()) {
      case spirv::StorageClass::StorageBuffer:
      case spirv::StorageClass::Uniform:
        ++resources;
        break;
      case spirv::StorageClass::Input:
      case spirv::StorageClass::Workgroup:
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

[[nodiscard]] auto uniformValueAccess(spirv::AccessChainOp access,
                                      bool valueIsBlock) -> LogicalResult {
  auto indices = access.getIndices();
  if (indices.size() < 2 || !matchPattern(indices[1], m_Zero()))
    return access.emitError("uniform resource access differs from its block");
  SmallVector<Value> rebased;
  if (!valueIsBlock) rebased.push_back(indices[0]);
  llvm::append_range(rebased, llvm::drop_begin(indices, 2));
  Value replacement = access.getBasePtr();
  if (!rebased.empty()) {
    OpBuilder builder(access);
    replacement = spirv::AccessChainOp::create(builder, access.getLoc(),
                                               replacement, rebased);
  }
  access.getResult().replaceAllUsesWith(replacement);
  access.erase();
  return success();
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

[[nodiscard]] auto rebaseUniformAccesses(
    spirv::AddressOfOp address, const mlir::cxx::KernelResource& buffer)
    -> LogicalResult {
  for (auto* user : llvm::make_early_inc_range(address->getUsers())) {
    auto access = dyn_cast<spirv::AccessChainOp>(user);
    if (!access)
      return user->emitError("uniform resource escapes its value access");
    if (failed(uniformValueAccess(access, isRecordBlock(buffer))))
      return failure();
  }
  return success();
}

[[nodiscard]] auto blockValueType(spirv::GlobalVariableOp variable,
                                  const mlir::cxx::KernelResource& buffer)
    -> FailureOr<Type> {
  auto pointer = cast<spirv::PointerType>(variable.getType());
  auto block = dyn_cast<spirv::StructType>(pointer.getPointeeType());
  if (!block || block.getNumElements() != 1)
    return variable.emitError(
        "resource block layout differs from its source contract");
  auto member = block.getElementType(0);
  if (buffer.kind == mlir::cxx::KernelBufferKind::Storage) {
    auto array = dyn_cast<spirv::RuntimeArrayType>(member);
    if (!array)
      return variable.emitError(
          "storage resource layout differs from its source contract");
    return spirv::RuntimeArrayType::get(
        array.getElementType(), mlir::cxx::kernelBufferElementStride(buffer));
  }
  auto array = dyn_cast<spirv::ArrayType>(member);
  if (!array || array.getNumElements() != 1)
    return variable.emitError(
        "uniform resource layout differs from its source contract");
  return array.getElementType();
}

[[nodiscard]] auto isReadOnlyStorage(const mlir::cxx::KernelResource& buffer)
    -> bool {
  return buffer.readOnly && buffer.kind == mlir::cxx::KernelBufferKind::Storage;
}

[[nodiscard]] auto blockDecorations(MLIRContext* context,
                                    const mlir::cxx::KernelResource& buffer)
    -> SmallVector<spirv::StructType::MemberDecorationInfo> {
  SmallVector<spirv::StructType::MemberDecorationInfo> decorations;
  if (isReadOnlyStorage(buffer))
    decorations.emplace_back(0, spirv::Decoration::NonWritable,
                             UnitAttr::get(context));
  return decorations;
}

[[nodiscard]] auto blockOf(MLIRContext* context, Type member,
                           const mlir::cxx::KernelResource& buffer) -> Type {
  if (isRecordBlock(buffer)) return member;
  return spirv::StructType::get({member}, {0},
                                blockDecorations(context, buffer));
}

[[nodiscard]] auto layoutResourceBlocks(
    spirv::ModuleOp module, ArrayRef<mlir::cxx::KernelResource> buffers,
    ArrayRef<spirv::GlobalVariableOp> resources) -> LogicalResult {
  auto* context = module.getContext();
  for (auto [buffer, source] : llvm::zip(buffers, resources)) {
    auto variable = source;
    auto member = blockValueType(variable, buffer);
    if (failed(member)) return failure();
    auto type = spirv::PointerType::get(blockOf(context, *member, buffer),
                                        resourceStorageClass(buffer));
    for (auto address : addressesOf(module, variable)) {
      address.getPointer().setType(type);
      if (buffer.kind == mlir::cxx::KernelBufferKind::Uniform &&
          failed(rebaseUniformAccesses(address, buffer)))
        return failure();
    }
    variable.setType(type);
    if (isReadOnlyStorage(buffer))
      variable->setAttr("non_writable", UnitAttr::get(context));
  }
  return success();
}

[[nodiscard]] auto ordersMemory(Attribute attribute) -> bool {
  auto semantics = dyn_cast<spirv::MemorySemanticsAttr>(attribute);
  if (!semantics) return false;
  return spirv::bitEnumContainsAny(
      semantics.getValue(), spirv::MemorySemantics::Acquire |
                                spirv::MemorySemantics::Release |
                                spirv::MemorySemantics::AcquireRelease |
                                spirv::MemorySemantics::SequentiallyConsistent);
}

[[nodiscard]] auto hasOrderingSemantics(Operation* op) -> bool {
  return llvm::any_of(op->getAttrs(), [](NamedAttribute attribute) {
    return ordersMemory(attribute.getValue());
  });
}

[[nodiscard]] auto synchronizesMemory(spirv::ModuleOp module) -> bool {
  auto found = module.walk([](Operation* op) {
    return hasOrderingSemantics(op) ? WalkResult::interrupt()
                                    : WalkResult::advance();
  });
  return found.wasInterrupted();
}

[[nodiscard]] auto isWritableVariable(spirv::GlobalVariableOp variable)
    -> bool {
  auto pointer = cast<spirv::PointerType>(variable.getType());
  switch (pointer.getStorageClass()) {
    case spirv::StorageClass::StorageBuffer:
      return isWritableBlock(cast<spirv::StructType>(pointer.getPointeeType()));
    case spirv::StorageClass::Workgroup:
      return true;
    default:
      return false;
  }
}

void decorateCoherentStorage(spirv::ModuleOp module) {
  if (!synchronizesMemory(module)) return;
  auto unit = UnitAttr::get(module.getContext());
  for (auto variable : module.getOps<spirv::GlobalVariableOp>()) {
    auto pointer = cast<spirv::PointerType>(variable.getType());
    if (pointer.getStorageClass() != spirv::StorageClass::StorageBuffer)
      continue;
    if (isWritableVariable(variable)) variable->setAttr("coherent", unit);
  }
}

[[nodiscard]] auto targetEnvironmentOf(ModuleOp module)
    -> FailureOr<SPIRVEnvironment> {
  auto triple = module->getAttrOfType<StringAttr>("cxx.triple");
  if (!triple) return module.emitError("device module has no target triple");
  auto environment = spirvEnvironmentOf(Triple{triple.getValue()});
  if (!environment)
    return module.emitError("target '")
           << triple.getValue() << "' names no SPIR-V environment";
  return *environment;
}

[[nodiscard]] auto clientAPIOf(SPIRVEnvironment environment)
    -> spirv::ClientAPI {
  if (environment == SPIRVEnvironment::kWebGPU) return spirv::ClientAPI::WebGPU;
  return spirv::ClientAPI::Vulkan;
}

[[nodiscard]] auto bindingsMayAlias(spirv::ClientAPI clientAPI) -> bool {
  return clientAPI != spirv::ClientAPI::WebGPU;
}

[[nodiscard]] auto verifyClientOperations(ModuleOp module,
                                          spirv::ClientAPI clientAPI)
    -> LogicalResult {
  if (clientAPI != spirv::ClientAPI::WebGPU) return success();
  return mlir::cxx::verifyWebGPUOperations(module);
}

void declareVersion(spirv::ModuleOp module, spirv::Version version) {
  auto vce = module.getVceTripleAttr();
  module.setVceTripleAttr(spirv::VerCapExtAttr::get(
      version, llvm::to_vector(vce.getCapabilities()),
      llvm::to_vector(vce.getExtensions()), module.getContext()));
}

[[nodiscard]] auto isInterfaceStorage(spirv::StorageClass storageClass,
                                      spirv::Version version) -> bool {
  if (version >= spirv::Version::V_1_4) return true;
  return storageClass == spirv::StorageClass::Input ||
         storageClass == spirv::StorageClass::Output;
}

class EntryPointInterface {
 public:
  EntryPointInterface(spirv::ModuleOp module, spirv::Version version)
      : module_(module), version_(version) {}

  [[nodiscard]] auto of(spirv::EntryPointOp entry) -> ArrayAttr {
    llvm::SetVector<Attribute> variables;
    llvm::DenseSet<Operation*> visited;
    SmallVector<spirv::FuncOp> pending{
        module_.lookupSymbol<spirv::FuncOp>(entry.getFnAttr())};
    while (!pending.empty()) {
      auto function = pending.pop_back_val();
      if (!function || !visited.insert(function.getOperation()).second)
        continue;
      function.walk([&](spirv::AddressOfOp address) {
        if (isInterfaceVariable(address))
          variables.insert(address.getVariableAttr());
      });
      function.walk([&](spirv::FunctionCallOp call) {
        pending.push_back(
            module_.lookupSymbol<spirv::FuncOp>(call.getCalleeAttr()));
      });
    }
    return ArrayAttr::get(module_.getContext(), variables.getArrayRef());
  }

 private:
  [[nodiscard]] auto isInterfaceVariable(spirv::AddressOfOp address) -> bool {
    auto variable = module_.lookupSymbol<spirv::GlobalVariableOp>(
        address.getVariableAttr());
    auto pointer = cast<spirv::PointerType>(variable.getType());
    return isInterfaceStorage(pointer.getStorageClass(), version_);
  }

  spirv::ModuleOp module_;
  spirv::Version version_;
};

void completeEntryPointInterfaces(spirv::ModuleOp module) {
  EntryPointInterface interfaces(module,
                                 module.getVceTripleAttr().getVersion());
  for (auto entry : module.getOps<spirv::EntryPointOp>())
    entry.setInterfaceAttr(interfaces.of(entry));
}

[[nodiscard]] auto isBindingStorage(spirv::StorageClass storageClass) -> bool {
  switch (storageClass) {
    case spirv::StorageClass::StorageBuffer:
    case spirv::StorageClass::Uniform:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto storageClassOf(Type pointer) -> spirv::StorageClass {
  return cast<spirv::PointerType>(pointer).getStorageClass();
}

[[nodiscard]] auto mayShareStorage(Type first, Type second) -> bool {
  auto firstClass = storageClassOf(first);
  auto secondClass = storageClassOf(second);
  if (firstClass == secondClass) return true;
  return isBindingStorage(firstClass) && isBindingStorage(secondClass);
}

class MemoryAliasing {
  using Declarations = SmallVector<Operation*, 2>;

  struct PointerArgument {
    unsigned index;
    Type type;
    Declarations roots;
  };

 public:
  MemoryAliasing(spirv::ModuleOp module, spirv::ClientAPI clientAPI)
      : module_(module), clientAPI_(clientAPI) {
    module_.walk([&](spirv::FunctionCallOp call) {
      calls_[module_.lookupSymbol(call.getCalleeAttr())].push_back(call);
    });
  }

  void decorate() {
    if (bindingsMayAlias(clientAPI_)) decorateBindings();
    for (auto function : module_.getOps<spirv::FuncOp>()) {
      for (auto call : calls_.lookup(function))
        decorateArguments(call, function);
      decorateParameters(function);
    }
  }

 private:
  [[nodiscard]] static auto isBinding(spirv::GlobalVariableOp variable)
      -> bool {
    return isBindingStorage(storageClassOf(variable.getType()));
  }

  void decorateBindings() {
    SmallVector<spirv::GlobalVariableOp> bindings;
    for (auto variable : module_.getOps<spirv::GlobalVariableOp>())
      if (isBinding(variable)) bindings.push_back(variable);
    if (bindings.size() < 2) return;
    if (llvm::none_of(bindings, isWritableVariable)) return;
    for (auto variable : bindings) markAliased(variable);
  }

  void decorateArguments(spirv::FunctionCallOp call, spirv::FuncOp callee) {
    SmallVector<PointerArgument> pointers;
    for (auto [index, operand] : llvm::enumerate(call.getArguments()))
      if (isa<spirv::PointerType>(operand.getType()))
        pointers.push_back(
            {unsigned(index), operand.getType(), rootsOf(operand)});
    for (auto [position, first] : llvm::enumerate(pointers)) {
      for (const auto& second : llvm::drop_begin(pointers, position + 1)) {
        if (!mayAlias(first, second)) continue;
        markAliased(callee, first.index);
        markAliased(callee, second.index);
      }
    }
  }

  void decorateParameters(spirv::FuncOp function) {
    auto globals = globalsOf(function);
    for (auto parameter : function.getArguments()) {
      if (!isa<spirv::PointerType>(parameter.getType())) continue;
      auto actuals = actualsOf(parameter);
      for (auto global : globals) {
        if (!mayAccess(parameter, global) || !overlaps(actuals, {global}))
          continue;
        markAliased(function, parameter.getArgNumber());
        if (auto variable = dyn_cast_or_null<spirv::GlobalVariableOp>(global))
          markAliased(variable);
      }
    }
  }

  [[nodiscard]] auto mayAlias(const PointerArgument& first,
                              const PointerArgument& second) const -> bool {
    if (!mayShareStorage(first.type, second.type)) return false;
    return overlaps(first.roots, second.roots);
  }

  [[nodiscard]] static auto mayAccess(BlockArgument parameter,
                                      Operation* global) -> bool {
    auto variable = dyn_cast_or_null<spirv::GlobalVariableOp>(global);
    if (!variable) return true;
    return mayShareStorage(parameter.getType(), variable.getType());
  }

  [[nodiscard]] auto mayOverlap(Operation* first, Operation* second) const
      -> bool {
    if (!first || !second) return true;
    if (first == second) return true;
    if (!bindingsMayAlias(clientAPI_)) return false;
    auto firstGlobal = dyn_cast<spirv::GlobalVariableOp>(first);
    auto secondGlobal = dyn_cast<spirv::GlobalVariableOp>(second);
    if (!firstGlobal || !secondGlobal) return false;
    return isBinding(firstGlobal) && isBinding(secondGlobal);
  }

  [[nodiscard]] auto overlaps(ArrayRef<Operation*> first,
                              ArrayRef<Operation*> second) const -> bool {
    return llvm::any_of(first, [&](Operation* declaration) {
      return llvm::any_of(second, [&](Operation* other) {
        return mayOverlap(declaration, other);
      });
    });
  }

  [[nodiscard]] auto rootsOf(Value pointer) -> Declarations {
    while (auto access = pointer.getDefiningOp<spirv::AccessChainOp>())
      pointer = access.getBasePtr();
    if (auto parameter = dyn_cast<BlockArgument>(pointer))
      return actualsOf(parameter);
    if (auto variable = pointer.getDefiningOp<spirv::VariableOp>())
      return {variable};
    if (auto address = pointer.getDefiningOp<spirv::AddressOfOp>())
      return {module_.lookupSymbol(address.getVariableAttr())};
    return {nullptr};
  }

  [[nodiscard]] auto actualsOf(BlockArgument parameter) -> Declarations {
    if (auto found = actuals_.find(parameter); found != actuals_.end())
      return found->second;
    Declarations actuals;
    auto* block = parameter.getOwner();
    auto calls = calls_.lookup(block->getParentOp());
    if (!block->isEntryBlock() || calls.empty()) actuals.push_back(nullptr);
    for (auto call : calls)
      llvm::append_range(
          actuals, rootsOf(call.getArguments()[parameter.getArgNumber()]));
    actuals_[parameter] = actuals;
    return actuals;
  }

  [[nodiscard]] auto globalsOf(spirv::FuncOp function) -> Declarations {
    if (auto found = globals_.find(function); found != globals_.end())
      return found->second;
    llvm::SetVector<Operation*> globals;
    function.walk([&](spirv::AddressOfOp address) {
      globals.insert(module_.lookupSymbol(address.getVariableAttr()));
    });
    function.walk([&](spirv::FunctionCallOp call) {
      auto callee = module_.lookupSymbol<spirv::FuncOp>(call.getCalleeAttr());
      if (!callee) {
        globals.insert(nullptr);
        return;
      }
      auto accessed = globalsOf(callee);
      globals.insert(accessed.begin(), accessed.end());
    });
    Declarations accessed(globals.begin(), globals.end());
    globals_[function] = accessed;
    return accessed;
  }

  void markAliased(spirv::GlobalVariableOp variable) {
    variable->setAttr("aliased", UnitAttr::get(module_.getContext()));
  }

  void markAliased(spirv::FuncOp function, unsigned index) {
    function.setArgAttr(index, spirv::DecorationAttr::name,
                        spirv::DecorationAttr::get(module_.getContext(),
                                                   spirv::Decoration::Aliased));
  }

  spirv::ModuleOp module_;
  spirv::ClientAPI clientAPI_;
  llvm::DenseMap<Operation*, SmallVector<spirv::FunctionCallOp>> calls_;
  llvm::DenseMap<Value, Declarations> actuals_;
  llvm::DenseMap<Operation*, Declarations> globals_;
};

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

[[nodiscard]] auto standardUniformLayoutFeature(spirv::ClientAPI clientAPI)
    -> llvm::StringRef {
  if (clientAPI == spirv::ClientAPI::WebGPU)
    return mlir::cxx::kWebGPUStandardUniformLayout;
  return "uniformBufferStandardLayout";
}

[[nodiscard]] auto requiredFeatures(
    const mlir::cxx::KernelModuleContract& contract,
    SPIRVEnvironment environment) -> llvm::json::Array {
  llvm::json::Array features;
  if (mlir::cxx::declaresAggregateUniforms(contract))
    features.push_back(standardUniformLayoutFeature(clientAPIOf(environment)));
  return features;
}

[[nodiscard]] auto spirvTarget(spirv::ModuleOp module,
                               const mlir::cxx::KernelModuleContract& contract,
                               SPIRVEnvironment environment)
    -> llvm::json::Object {
  llvm::json::Array capabilities, extensions;
  auto vce = module.getVceTripleAttr();
  for (auto capability : vce.getCapabilities())
    capabilities.push_back(spirv::stringifyCapability(capability));
  for (auto extension : vce.getExtensions())
    extensions.push_back(spirv::stringifyExtension(extension));
  llvm::json::Object target{
      {"targetEnvironment", llvm::StringRef{to_string(environment)}},
      {"addressingModel",
       spirv::stringifyAddressingModel(module.getAddressingModel())},
      {"memoryModel", spirv::stringifyMemoryModel(module.getMemoryModel())},
      {"spirv", llvm::json::Object{
                    {"version", spirv::stringifyVersion(vce.getVersion())},
                    {"capabilities", std::move(capabilities)},
                    {"extensions", std::move(extensions)}}}};
  auto features = requiredFeatures(contract, environment);
  if (!features.empty()) target["requiredFeatures"] = std::move(features);
  return target;
}

}  // namespace

auto lowerDeviceModuleToSPIRV(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<LoweredSPIRVModule> {
  using namespace mlir;
  auto environment = targetEnvironmentOf(source);
  if (failed(environment)) return failure();
  auto contract = mlir::cxx::kernelModuleContractOf(source, kernel);
  if (failed(contract)) return failure();
  const auto version = spirvVersionOf(*environment);
  const auto clientAPI = clientAPIOf(*environment);
  auto working = lowerDeviceModuleToGPU(source, kernel);
  if (failed(working) || failed(checkLowerableOperations(working->get())) ||
      failed(verifyClientOperations(working->get(), clientAPI)))
    return failure();
  auto* context = source.getContext();
  auto triple = spirv::VerCapExtAttr::get(version, {spirv::Capability::Shader},
                                          {}, context);
  working->get()->setAttr(
      spirv::getTargetEnvAttrName(),
      spirv::TargetEnvAttr::get(
          triple, spirv::getDefaultResourceLimits(context), clientAPI));
  PassManager lowering(context);
  lowering.addPass(memref::createFoldMemRefAliasOpsPass());
  lowering.addPass(createConvertCxxDeviceToSPIRVPass());
  if (failed(lowering.run(working->get()))) return failure();
  auto modules = working->get().getOps<spirv::ModuleOp>();
  if (!llvm::hasSingleElement(modules)) {
    source.emitError("expected one converted SPIR-V module");
    return failure();
  }
  auto module = *modules.begin();
  auto resources = SPIRVResourceABI(module, *contract).lower();
  if (failed(resources) ||
      failed(layoutResourceBlocks(module, contract->resources, *resources)))
    return failure();

  decorateCoherentStorage(module);
  MemoryAliasing(module, clientAPI).decorate();

  PassManager abi(context);
  abi.addNestedPass<spirv::ModuleOp>(
      spirv::createSPIRVLowerABIAttributesPass());
  abi.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
  if (failed(abi.run(working->get()))) return failure();
  declareVersion(module, version);
  completeEntryPointInterfaces(module);
  if (failed(checkEntry(module, *contract, *environment)) ||
      failed(checkBuffers(module, contract->resources, false)) ||
      failed(verify(module)))
    return failure();
  return LoweredSPIRVModule{OwningOpRef<Operation*>(module->clone()),
                            std::move(*contract), *environment};
}

auto compileSPIRVModule(mlir::ModuleOp source, llvm::StringRef kernel)
    -> mlir::FailureOr<KernelArtifact> {
  using namespace mlir;
  auto lowered = lowerDeviceModuleToSPIRV(source, kernel);
  if (failed(lowered)) return failure();
  const auto& contract = lowered->contract;
  auto module = cast<spirv::ModuleOp>(lowered->module.get());
  auto* context = source.getContext();
  SmallVector<uint32_t> words;
  if (failed(spirv::serialize(module, words))) return failure();
  auto decoded = spirv::deserialize(words, context);
  if (!decoded || failed(verify(*decoded)) ||
      failed(checkEntry(*decoded, contract, lowered->environment)) ||
      failed(checkBuffers(*decoded, contract.resources, true)))
    return failure();
  KernelArtifact artifact;
  artifact.bytes.resize(words.size() * 4);
  for (auto [index, word] : llvm::enumerate(words))
    llvm::support::endian::write32le(artifact.bytes.data() + index * 4, word);
  artifact.manifest =
      kernelManifest(contract, "spirv", artifact.bytes,
                     spirvTarget(*decoded, contract, lowered->environment));
  return artifact;
}

}  // namespace cxx
