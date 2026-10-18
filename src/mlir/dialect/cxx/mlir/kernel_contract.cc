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
#include <cxx/mlir/kernel_contract.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
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

[[nodiscard]] auto isBufferElementType(Type type) -> bool {
  return type.isF32() || type.isSignlessInteger(32);
}

[[nodiscard]] auto isUniformValueType(Type type) -> bool {
  if (isBufferElementType(type)) return true;
  auto vector = dyn_cast<VectorType>(type);
  return vector && vector.getRank() == 1 && !vector.isScalable() &&
         vector.getNumElements() >= 2 && vector.getNumElements() <= 4 &&
         isBufferElementType(vector.getElementType());
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
  if (kind == KernelBufferKind::Uniform)
    return *space == ::cxx::kConstantAddressSpace &&
           isUniformValueType(pointer.getElementType());
  return (*space == ::cxx::kDeviceAddressSpace ||
          *space == ::cxx::kConstantAddressSpace) &&
         isBufferElementType(pointer.getElementType());
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
  const auto& limits = ::cxx::kVulkan11F32Limits;
  const std::uint64_t maxSizes[] = {limits.maxWorkgroupSizeX,
                                    limits.maxWorkgroupSizeY,
                                    limits.maxWorkgroupSizeZ};
  std::uint64_t product = 1;
  for (auto [size, maxSize] : llvm::zip(workgroup.asArrayRef(), maxSizes)) {
    if (size <= 0) return entry.emitError("invalid device local size");
    if (std::uint64_t(size) > maxSize)
      return entry.emitError("device local size exceeds the profile limits");
    product *= std::uint64_t(size);
  }
  if (product > limits.maxWorkgroupInvocations)
    return entry.emitError("device local size exceeds the profile limits");
  return std::array{workgroup[0], workgroup[1], workgroup[2]};
}

[[nodiscard]] auto buffersOf(FuncOp entry)
    -> FailureOr<llvm::SmallVector<KernelBuffer>> {
  auto metadata = entry->getAttrOfType<ArrayAttr>("cxx.kernel_buffers");
  if (!metadata)
    return entry.emitError("device entry lacks checked buffer metadata");
  auto inputs = entry.getFunctionType().getInputs();
  llvm::SmallVector<KernelBuffer> buffers;
  for (auto attribute : metadata) {
    auto dict = dyn_cast<DictionaryAttr>(attribute);
    if (!dict) return entry.emitError("invalid device buffer contract");
    auto parameter = integerField(dict, "parameter");
    auto binding = integerField(dict, "binding");
    auto kind = bufferKindOf(dict.getAs<StringAttr>("kind"));
    auto readOnly = dict.getAs<BoolAttr>("read_only");
    auto name = dict.getAs<StringAttr>("name");
    auto unsignedElements = dict.getAs<BoolAttr>("unsigned");
    auto pointer = parameter && *parameter < inputs.size()
                       ? dyn_cast<PointerType>(inputs[*parameter])
                       : PointerType{};
    if (!binding || *binding > ::cxx::kMaxBufferBinding || !kind || !readOnly ||
        !name || !unsignedElements ||
        !isBufferContract(*kind, pointer, readOnly.getValue()))
      return entry.emitError("invalid device buffer contract");
    for (const auto& other : buffers)
      if (other.binding == *binding)
        return entry.emitError("duplicate device buffer index");
    buffers.push_back({*parameter, name, *binding, *kind, readOnly.getValue(),
                       unsignedElements.getValue(), pointer.getElementType()});
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
  return "global";
}

auto stringifyKernelBufferKind(KernelBufferKind kind) -> llvm::StringRef {
  switch (kind) {
    case KernelBufferKind::Storage:
      return "storage";
    case KernelBufferKind::Uniform:
      return "uniform";
  }
  return "storage";
}

auto kernelBufferScalarTypeName(const KernelBuffer& buffer) -> llvm::StringRef {
  if (getElementTypeOrSelf(buffer.elementType).isF32()) return "f32";
  return buffer.unsignedElements ? "u32" : "i32";
}

auto kernelBufferComponents(const KernelBuffer& buffer) -> std::uint32_t {
  auto vector = dyn_cast<VectorType>(buffer.elementType);
  return vector ? static_cast<std::uint32_t>(vector.getNumElements()) : 1;
}

}  // namespace mlir::cxx
