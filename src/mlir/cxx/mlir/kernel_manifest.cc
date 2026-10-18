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
#include <cxx/mlir/cxx_dialect.h>
#include <cxx/mlir/device_kernel.h>
#include <cxx/mlir/kernel_manifest.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/SHA256.h>

namespace cxx {
namespace {

[[nodiscard]] auto uniformSize(std::uint32_t components) -> std::uint32_t {
  return components == 3 ? 16 : 4 * components;
}

}  // namespace

auto kernelEntryPoints(mlir::ModuleOp module)
    -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> entries;
  for (auto function : module.getOps<mlir::cxx::FuncOp>()) {
    if (function.isExternal()) continue;
    if (!function->hasAttr("cxx.kernel_workgroup_size")) continue;
    entries.push_back(function.getSymName().str());
  }
  return entries;
}

auto kernelEntryOf(mlir::ModuleOp module, llvm::StringRef kernel)
    -> mlir::FailureOr<KernelEntry> {
  auto entry = module.lookupSymbol<mlir::cxx::FuncOp>(kernel);
  if (!entry) {
    module.emitError("selected kernel was not found: ") << kernel;
    return mlir::failure();
  }
  auto contract = mlir::cxx::kernelContractOf(entry);
  if (mlir::failed(contract)) return mlir::failure();
  return KernelEntry{kernel.str(), std::move(*contract)};
}

auto kernelManifest(const KernelEntry& entry, llvm::StringRef format,
                    llvm::ArrayRef<std::uint8_t> artifact,
                    llvm::json::Object target) -> std::string {
  const auto& contract = entry.contract;
  llvm::json::Array resources;
  for (const auto& buffer : contract.buffers) {
    auto isUniform = buffer.kind == mlir::cxx::KernelBufferKind::Uniform;
    auto components = mlir::cxx::kernelBufferComponents(buffer);
    llvm::json::Object resource{
        {"parameter", buffer.parameter},
        {"name", buffer.name.str()},
        {"set", ::cxx::kBufferDescriptorSet},
        {"binding", buffer.binding},
        {"kind", mlir::cxx::stringifyKernelBufferKind(buffer.kind)},
        {"elementType", mlir::cxx::kernelBufferScalarTypeName(buffer)},
        {"components", components},
        {"storageClass", isUniform ? "Uniform" : "StorageBuffer"},
        {"memberOffset", 0},
        {"access", buffer.readOnly ? "read" : "read_write"}};
    if (isUniform)
      resource["size"] = uniformSize(components);
    else
      resource["arrayStride"] = 4;
    resources.push_back(std::move(resource));
  }
  llvm::json::Array invocationIds;
  for (const auto& id : contract.invocationIds)
    invocationIds.push_back(llvm::json::Object{
        {"parameter", id.parameter},
        {"kind", mlir::cxx::stringifyKernelInvocationIdKind(id.kind)},
        {"components", id.components}});
  llvm::json::Array localSize;
  for (auto size : contract.workgroup) localSize.push_back(size);
  llvm::json::Object manifest{
      {"schemaVersion", 3},
      {"producer", "cxx"},
      {"format", format},
      {"entryPoint", entry.name},
      {"profile", "vulkan1.1-f32"},
      {"abiVersion", 1},
      {"indexBits", 32},
      {"localSize", std::move(localSize)},
      {"resources", std::move(resources)},
      {"invocationIds", std::move(invocationIds)},
      {"sha256", llvm::toHex(llvm::SHA256::hash(artifact), true)}};
  for (auto& [key, value] : target) manifest[key] = std::move(value);
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
}

}  // namespace cxx
