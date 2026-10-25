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
#include <cxx/mlir/kernel_manifest.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/SHA256.h>

namespace cxx {

namespace {

class ReflectionTypes {
 public:
  explicit ReflectionTypes(llvm::StringRef format) : format_(format) {}

  [[nodiscard]] auto member(mlir::Type type, std::string name,
                            std::uint32_t offset, bool unsignedElements)
      -> llvm::json::Object {
    llvm::json::Object member{{"name", std::move(name)}, {"offset", offset}};
    llvm::json::Array dimensions, literal;
    std::optional<std::uint32_t> stride;
    while (auto array = mlir::dyn_cast<mlir::cxx::ArrayType>(type)) {
      if (!stride) stride = mlir::cxx::bufferLayoutSize(array.getElementType());
      dimensions.insert(dimensions.begin(), array.getSize());
      literal.push_back(true);
      type = array.getElementType();
    }
    member["type"] = typeName(type, unsignedElements);
    if (stride) {
      member["array"] = std::move(dimensions);
      member["array_size_is_literal"] = std::move(literal);
      member["array_stride"] = *stride;
    }
    return member;
  }

  [[nodiscard]] auto members(mlir::Type record) -> llvm::json::Array {
    llvm::json::Array members;
    for (auto [index, field] :
         llvm::enumerate(mlir::cxx::bufferLayoutMembers(record)))
      members.push_back(member(field.type, "_m" + std::to_string(index),
                               field.offset, false));
    return members;
  }

  void define(std::string id, llvm::StringRef name, llvm::json::Array members) {
    types_[id] = llvm::json::Object{{"name", name.str()},
                                    {"members", std::move(members)}};
  }

  [[nodiscard]] auto take() -> llvm::json::Object { return std::move(types_); }

 private:
  [[nodiscard]] auto typeName(mlir::Type type, bool unsignedElements)
      -> std::string {
    if (auto record = mlir::dyn_cast<mlir::cxx::ClassType>(type))
      return recordId(record);
    auto scalar =
        scalarName(mlir::getElementTypeOrSelf(type), unsignedElements);
    auto vector = mlir::dyn_cast<mlir::VectorType>(type);
    if (!vector) return scalar.str();
    return vectorPrefix(scalar).str() + std::to_string(vector.getNumElements());
  }

  [[nodiscard]] auto scalarName(mlir::Type scalar, bool unsignedElements) const
      -> llvm::StringRef {
    if (scalar.isF32()) return "float";
    if (unsignedElements || format_ == "spirv") return "uint";
    return "int";
  }

  [[nodiscard]] static auto vectorPrefix(llvm::StringRef scalar)
      -> llvm::StringRef {
    if (scalar == "float") return "vec";
    if (scalar == "uint") return "uvec";
    return "ivec";
  }

  [[nodiscard]] auto recordId(mlir::cxx::ClassType record) -> std::string {
    auto found = records_.find(record);
    if (found != records_.end()) return found->second;
    auto id = "_struct_" + std::to_string(records_.size());
    records_[record] = id;
    define(id, record.getName(), members(record));
    return id;
  }

  llvm::StringRef format_;
  llvm::json::Object types_;
  llvm::DenseMap<mlir::Type, std::string> records_;
};

}  // namespace

auto kernelManifest(const mlir::cxx::KernelModuleContract& module,
                    llvm::StringRef format,
                    llvm::ArrayRef<std::uint8_t> artifact,
                    llvm::json::Object target) -> std::string {
  ReflectionTypes types(format);
  llvm::json::Array storage, uniforms;
  for (auto [index, buffer] : llvm::enumerate(module.resources)) {
    auto isUniform = buffer.kind == mlir::cxx::KernelBufferKind::Uniform;
    auto typeId = "_buffer_" + std::to_string(index);
    if (isUniform && mlir::cxx::isBufferRecordType(buffer.elementType)) {
      types.define(typeId, buffer.name.getValue(),
                   types.members(buffer.elementType));
    } else {
      auto member =
          types.member(buffer.elementType, "_m0", 0, buffer.unsignedElements);
      if (!isUniform) {
        auto dimensions = member.getArray("array");
        llvm::json::Array runtime{0}, literal{true};
        if (dimensions)
          for (auto& dimension : *dimensions) {
            runtime.insert(runtime.end() - 1, dimension);
            literal.push_back(true);
          }
        member["array"] = std::move(runtime);
        member["array_size_is_literal"] = std::move(literal);
        member["array_stride"] = mlir::cxx::kernelBufferElementStride(buffer);
      }
      types.define(typeId, buffer.name.getValue(),
                   llvm::json::Array{std::move(member)});
    }
    llvm::json::Object resource{
        {"type", typeId},
        {"name", buffer.name.str()},
        {"block_size",
         isUniform ? mlir::cxx::bufferLayoutExtent(buffer.elementType) : 0},
        {"set", ::cxx::kBufferDescriptorSet},
        {"binding", buffer.binding}};
    if (isUniform)
      uniforms.push_back(std::move(resource));
    else {
      if (buffer.readOnly) resource["readonly"] = true;
      storage.push_back(std::move(resource));
    }
  }
  llvm::json::Array entries;
  llvm::json::Object sourceEntries;
  for (const auto& entry : module.entries) {
    llvm::json::Array localSize, invocationIds, parameters;
    for (auto size : entry.contract.workgroup) localSize.push_back(size);
    entries.push_back(
        llvm::json::Object{{"name", entry.name},
                           {"mode", "comp"},
                           {"workgroup_size", std::move(localSize)},
                           {"workgroup_size_is_spec_constant_id",
                            llvm::json::Array{false, false, false}}});
    for (const auto& id : entry.contract.invocationIds)
      invocationIds.push_back(llvm::json::Object{
          {"parameter", id.parameter},
          {"kind", mlir::cxx::stringifyKernelInvocationIdKind(id.kind)},
          {"components", id.components}});
    for (auto [buffer, resource] :
         llvm::zip(entry.contract.buffers, entry.resources)) {
      const auto& source = module.resources[resource];
      llvm::json::Object parameter{
          {"parameter", buffer.parameter},
          {"resource", source.name.str()},
          {"elementType", mlir::cxx::kernelBufferScalarTypeName(source)}};
      if (!buffer.name.getValue().empty())
        parameter["name"] = buffer.name.str();
      parameters.push_back(std::move(parameter));
    }
    sourceEntries[entry.name] =
        llvm::json::Object{{"parameters", std::move(parameters)},
                           {"invocationIds", std::move(invocationIds)}};
  }
  llvm::json::Object metadata{
      {"format", format},
      {"profile", "vulkan1.1-f32"},
      {"abiVersion", 1},
      {"indexBits", 32},
      {"entries", std::move(sourceEntries)},
      {"sha256", llvm::toHex(llvm::SHA256::hash(artifact), true)}};
  for (auto& [key, value] : target) metadata[key] = std::move(value);
  llvm::json::Object manifest{{"entryPoints", std::move(entries)},
                              {"types", types.take()},
                              {"cxx", std::move(metadata)}};
  if (!storage.empty()) manifest["ssbos"] = std::move(storage);
  if (!uniforms.empty()) manifest["ubos"] = std::move(uniforms);
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
}

}  // namespace cxx
