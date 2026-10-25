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
#include <cxx/mlir/kernel_contract.h>
#include <llvm/ADT/StringSet.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeUtilities.h>

#include <optional>

namespace mlir::cxx {
namespace {

[[nodiscard]] auto integerField(DictionaryAttr dict, StringRef name)
    -> std::optional<std::uint32_t> {
  auto value = dict.getAs<IntegerAttr>(name);
  if (!value || !value.getType().isSignlessInteger(32) || value.getInt() < 0 ||
      value.getInt() > 0x7fffffff)
    return std::nullopt;
  return static_cast<std::uint32_t>(value.getInt());
}

[[nodiscard]] auto addressSpaceOf(PointerType pointer)
    -> std::optional<std::uint32_t> {
  if (!pointer) return std::nullopt;
  auto space =
      dyn_cast<LLVM::LLVMAddrSpaceAttrInterface>(pointer.getMemorySpace());
  if (!space) return std::nullopt;
  return space.getAddressSpace();
}

[[nodiscard]] auto bufferKindOf(StringAttr name)
    -> std::optional<KernelBufferKind> {
  if (!name) return std::nullopt;
  for (auto kind : {KernelBufferKind::Storage, KernelBufferKind::Uniform})
    if (name.getValue() == stringifyKernelBufferKind(kind)) return kind;
  return std::nullopt;
}

[[nodiscard]] auto isBufferContract(KernelBufferKind kind, PointerType pointer,
                                    bool readOnly) -> bool {
  auto space = addressSpaceOf(pointer);
  if (!space) return false;
  if (*space == ::cxx::kConstantAddressSpace && !readOnly) return false;
  if (!isBufferLayoutType(pointer.getElementType())) return false;
  if (kind == KernelBufferKind::Uniform)
    return *space == ::cxx::kConstantAddressSpace;
  return *space == ::cxx::kDeviceAddressSpace ||
         *space == ::cxx::kConstantAddressSpace;
}

[[nodiscard]] auto invocationIdComponents(Type type)
    -> std::optional<std::uint32_t> {
  if (type.isSignlessInteger(32)) return 1;
  auto vector = dyn_cast<VectorType>(type);
  if (!vector || vector.getRank() != 1 ||
      !vector.getElementType().isSignlessInteger(32))
    return std::nullopt;
  auto lanes = vector.getNumElements();
  if (lanes != 2 && lanes != 3) return std::nullopt;
  return static_cast<std::uint32_t>(lanes);
}

[[nodiscard]] auto invocationIdKindOf(StringAttr name)
    -> std::optional<KernelInvocationIdKind> {
  if (!name) return std::nullopt;
  for (auto kind :
       {KernelInvocationIdKind::Global, KernelInvocationIdKind::Local,
        KernelInvocationIdKind::Workgroup})
    if (name.getValue() == stringifyKernelInvocationIdKind(kind)) return kind;
  return std::nullopt;
}

[[nodiscard]] auto workgroupOf(FuncOp entry)
    -> FailureOr<std::array<std::int32_t, 3>> {
  auto workgroup =
      entry->getAttrOfType<DenseI32ArrayAttr>("cxx.kernel_workgroup_size");
  if (!workgroup || workgroup.size() != 3 || entry.isExternal() ||
      entry.getFunctionType().getVariadic() ||
      !entry.getFunctionType().getResults().empty())
    return entry.emitError("selected function is not a defined checked kernel");
  if (llvm::any_of(workgroup.asArrayRef(), [](auto size) { return size <= 0; }))
    return entry.emitError("invalid device local size");
  auto limits = deviceProfileLimitsOf(entry->getParentOfType<ModuleOp>());
  if (!limits) return entry.emitError("selected kernel has no device profile");
  if (!limits->admitsWorkgroup(workgroup[0], workgroup[1], workgroup[2]))
    return entry.emitError("device local size exceeds the profile limits");
  return std::array{workgroup[0], workgroup[1], workgroup[2]};
}

[[nodiscard]] auto bufferOf(DictionaryAttr dict, TypeRange inputs)
    -> std::optional<KernelBuffer> {
  if (!dict) return std::nullopt;
  auto parameter = integerField(dict, "parameter");
  auto binding = integerField(dict, "binding");
  auto kind = bufferKindOf(dict.getAs<StringAttr>("kind"));
  auto readOnly = dict.getAs<BoolAttr>("read_only");
  auto name = dict.getAs<StringAttr>("name");
  auto unsignedElements = dict.getAs<BoolAttr>("unsigned");
  if (!parameter || *parameter >= inputs.size()) return std::nullopt;
  if (!binding || *binding > ::cxx::kMaxBufferBinding) return std::nullopt;
  if (!kind || !name) return std::nullopt;
  if (!readOnly || !unsignedElements) return std::nullopt;
  auto pointer = dyn_cast<PointerType>(inputs[*parameter]);
  if (!isBufferContract(*kind, pointer, readOnly.getValue()))
    return std::nullopt;
  return KernelBuffer{{name, *binding, *kind, readOnly.getValue(),
                       unsignedElements.getValue(), pointer.getElementType()},
                      *parameter};
}

[[nodiscard]] auto buffersOf(FuncOp entry)
    -> FailureOr<llvm::SmallVector<KernelBuffer>> {
  auto metadata = entry->getAttrOfType<ArrayAttr>("cxx.kernel_buffers");
  if (!metadata)
    return entry.emitError("device entry lacks checked buffer metadata");
  auto inputs = entry.getFunctionType().getInputs();
  llvm::SmallVector<KernelBuffer> buffers;
  for (auto attribute : metadata) {
    auto buffer = bufferOf(dyn_cast<DictionaryAttr>(attribute), inputs);
    if (!buffer) return entry.emitError("invalid device buffer contract");
    for (const auto& other : buffers)
      if (other.binding == buffer->binding)
        return entry.emitError("duplicate device buffer index");
    buffers.push_back(*buffer);
  }
  return buffers;
}

[[nodiscard]] auto invocationIdsOf(FuncOp entry)
    -> FailureOr<llvm::SmallVector<KernelInvocationId>> {
  auto metadata = entry->getAttrOfType<ArrayAttr>("cxx.kernel_invocation_ids");
  if (!metadata)
    return entry.emitError("device entry lacks checked invocation id metadata");
  auto inputs = entry.getFunctionType().getInputs();
  llvm::SmallVector<KernelInvocationId> ids;
  for (auto attribute : metadata) {
    auto dict = dyn_cast<DictionaryAttr>(attribute);
    auto parameter = dict ? integerField(dict, "parameter") : std::nullopt;
    auto kind = dict ? invocationIdKindOf(dict.getAs<StringAttr>("kind"))
                     : std::nullopt;
    auto components = parameter && *parameter < inputs.size()
                          ? invocationIdComponents(inputs[*parameter])
                          : std::nullopt;
    if (!kind || !components)
      return entry.emitError("invalid device invocation id contract");
    ids.push_back({*parameter, *kind, *components});
  }
  return ids;
}

[[nodiscard]] auto checkParameterCoverage(FuncOp entry,
                                          const KernelContract& contract)
    -> LogicalResult {
  llvm::SmallVector<bool> covered(entry.getFunctionType().getInputs().size());
  auto cover = [&](std::uint32_t parameter) -> LogicalResult {
    if (covered[parameter])
      return entry.emitError("device parameter ")
             << parameter << " has more than one binding";
    covered[parameter] = true;
    return success();
  };
  for (const auto& buffer : contract.buffers)
    if (failed(cover(buffer.parameter))) return failure();
  for (const auto& id : contract.invocationIds)
    if (failed(cover(id.parameter))) return failure();
  for (auto [index, isCovered] : llvm::enumerate(covered))
    if (!isCovered)
      return entry.emitError("device parameter ")
             << index << " has no checked binding";
  return success();
}

}  // namespace

namespace {
[[nodiscard]] auto resourceArgumentAttribute(Value root, llvm::StringRef name)
    -> Attribute {
  auto argument = dyn_cast<BlockArgument>(root);
  if (!argument) return {};
  auto function =
      dyn_cast<FunctionOpInterface>(argument.getOwner()->getParentOp());
  if (!function || function.getFunctionBody().empty()) return {};
  if (argument.getOwner() != &function.getFunctionBody().front()) return {};
  return function.getArgAttr(argument.getArgNumber(), name);
}
}  // namespace

auto deviceResourceBinding(Value root) -> Attribute {
  auto binding = resourceArgumentAttribute(root, kResourceBindingAttrName);
  return binding ? binding
                 : resourceArgumentAttribute(root, "spirv.interface_var_abi");
}

auto deviceResourceIndex(Value root) -> IntegerAttr {
  return dyn_cast_or_null<IntegerAttr>(
      resourceArgumentAttribute(root, kResourceIndexAttrName));
}

auto isReadOnlyDeviceResource(Value root) -> bool {
  auto value = dyn_cast_or_null<BoolAttr>(
      resourceArgumentAttribute(root, kResourceReadOnlyAttrName));
  return value && value.getValue();
}

auto hasUnsignedDeviceElements(Value root) -> bool {
  auto value = dyn_cast_or_null<BoolAttr>(
      resourceArgumentAttribute(root, kResourceUnsignedAttrName));
  return value && value.getValue();
}

auto deviceProfileLimitsOf(ModuleOp module)
    -> std::optional<::cxx::DeviceProfileLimits> {
  if (!module) return std::nullopt;
  auto profile = module->getAttrOfType<StringAttr>("cxx.device_profile");
  if (!profile) return std::nullopt;
  return ::cxx::deviceProfileLimits(profile.getValue());
}

auto kernelContractOf(FuncOp entry) -> FailureOr<KernelContract> {
  auto workgroup = workgroupOf(entry);
  if (failed(workgroup)) return failure();
  auto buffers = buffersOf(entry);
  if (failed(buffers)) return failure();
  auto ids = invocationIdsOf(entry);
  if (failed(ids)) return failure();
  KernelContract contract{*workgroup, std::move(*buffers), std::move(*ids)};
  if (failed(checkParameterCoverage(entry, contract))) return failure();
  return contract;
}

namespace {

[[nodiscard]] auto hasSameResourceInterface(const KernelResource& left,
                                            const KernelResource& right)
    -> bool {
  if (left.binding != right.binding || left.kind != right.kind) return false;
  if (left.readOnly != right.readOnly) return false;
  if (left.unsignedElements != right.unsignedElements) return false;
  return left.elementType == right.elementType;
}

}  // namespace

auto isKernelEntry(FuncOp function) -> bool {
  return function->hasAttr("cxx.kernel_workgroup_size");
}

auto kernelModuleContractOf(ModuleOp module, StringRef selected)
    -> FailureOr<KernelModuleContract> {
  SmallVector<FuncOp> functions;
  if (selected.empty()) {
    for (auto function : module.getOps<FuncOp>())
      if (!function.isExternal() && isKernelEntry(function))
        functions.push_back(function);
  } else {
    auto function = module.lookupSymbol<FuncOp>(selected);
    if (!function)
      return module.emitError("selected kernel was not found: ") << selected;
    functions.push_back(function);
  }
  if (functions.empty())
    return module.emitError("device output requires a [[kernel]] entry point");

  llvm::StringSet<> names;
  for (auto& operation : *module.getBody())
    if (auto name = operation.getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      names.insert(name.getValue());
  KernelModuleContract result;
  for (auto function : functions) {
    auto contract = kernelContractOf(function);
    if (failed(contract)) return failure();
    result.entries.push_back(
        {function.getSymName().str(), std::move(*contract), {}});
  }
  for (auto& entry : result.entries) {
    for (const auto& buffer : entry.contract.buffers) {
      auto found = llvm::find_if(result.resources, [&](const auto& other) {
        return hasSameResourceInterface(buffer, other);
      });
      entry.resources.push_back(unsigned(found - result.resources.begin()));
      if (found != result.resources.end()) continue;
      KernelResource resource = buffer;
      llvm::SmallString<64> name(
          buffer.name.getValue().empty() ? "buffer" : buffer.name.getValue());
      if (names.contains(name)) {
        unsigned counter = 0;
        name = SymbolTable::generateSymbolName<64>(
            name,
            [&](StringRef candidate) { return names.contains(candidate); },
            counter);
      }
      names.insert(name);
      resource.name = StringAttr::get(module.getContext(), name);
      result.resources.push_back(resource);
    }
  }
  return result;
}

auto stringifyKernelInvocationIdKind(KernelInvocationIdKind kind)
    -> llvm::StringRef {
  switch (kind) {
    case KernelInvocationIdKind::Global:
      return "global";
    case KernelInvocationIdKind::Local:
      return "local";
    case KernelInvocationIdKind::Workgroup:
      return "workgroup";
  }
  llvm_unreachable("unknown kernel invocation id kind");
}

auto stringifyKernelBufferKind(KernelBufferKind kind) -> llvm::StringRef {
  switch (kind) {
    case KernelBufferKind::Storage:
      return "storage";
    case KernelBufferKind::Uniform:
      return "uniform";
  }
  llvm_unreachable("unknown kernel buffer kind");
}

auto kernelBufferScalarTypeName(const KernelResource& buffer)
    -> llvm::StringRef {
  if (isa<ClassType>(buffer.elementType)) return "struct";
  if (isa<ArrayType>(buffer.elementType)) return "array";
  if (getElementTypeOrSelf(buffer.elementType).isF32()) return "f32";
  return buffer.unsignedElements ? "u32" : "i32";
}

auto kernelBufferComponents(const KernelResource& buffer) -> std::uint32_t {
  auto vector = dyn_cast<VectorType>(buffer.elementType);
  return vector ? static_cast<std::uint32_t>(vector.getNumElements()) : 1;
}

auto kernelBufferElementStride(const KernelResource& buffer) -> std::uint32_t {
  return bufferLayoutSize(buffer.elementType);
}

auto declaresAggregateUniforms(const KernelModuleContract& module) -> bool {
  return llvm::any_of(module.resources, [](const KernelResource& resource) {
    return resource.kind == KernelBufferKind::Uniform &&
           !isBufferValueType(resource.elementType);
  });
}

}  // namespace mlir::cxx
