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

#pragma once

#include <cxx/device_profile_limits.h>
#include <cxx/mlir/cxx_dialect.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::cxx {

inline constexpr llvm::StringLiteral kResourceBindingAttrName = "cxx.resource";
inline constexpr llvm::StringLiteral kResourceReadOnlyAttrName =
    "cxx.resource_read_only";
inline constexpr llvm::StringLiteral kResourceUnsignedAttrName =
    "cxx.resource_unsigned";
inline constexpr llvm::StringLiteral kResourceIndexAttrName =
    "cxx.resource_index";

[[nodiscard]] auto deviceResourceBinding(Value root) -> Attribute;
[[nodiscard]] auto deviceResourceIndex(Value root) -> IntegerAttr;
[[nodiscard]] auto isReadOnlyDeviceResource(Value root) -> bool;
[[nodiscard]] auto hasUnsignedDeviceElements(Value root) -> bool;

enum class KernelBufferKind { Storage, Uniform };

struct KernelResource {
  StringAttr name;
  std::uint32_t binding = 0;
  KernelBufferKind kind = KernelBufferKind::Storage;
  bool readOnly = false;
  bool unsignedElements = false;
  Type elementType;
};

struct KernelBuffer : KernelResource {
  std::uint32_t parameter = 0;
};

enum class KernelInvocationIdKind { Global, Local, Workgroup };

struct KernelInvocationId {
  std::uint32_t parameter = 0;
  KernelInvocationIdKind kind = KernelInvocationIdKind::Global;
  std::uint32_t components = 1;
};

struct KernelContract {
  std::array<std::int32_t, 3> workgroup{};
  llvm::SmallVector<KernelBuffer> buffers;
  llvm::SmallVector<KernelInvocationId> invocationIds;
};

struct KernelEntryContract {
  std::string name;
  KernelContract contract;
  llvm::SmallVector<unsigned> resources;
};

struct KernelModuleContract {
  llvm::SmallVector<KernelEntryContract> entries;
  llvm::SmallVector<KernelResource> resources;
};

[[nodiscard]] auto deviceProfileLimitsOf(ModuleOp module)
    -> std::optional<::cxx::DeviceProfileLimits>;

[[nodiscard]] auto isKernelEntry(FuncOp function) -> bool;

[[nodiscard]] auto kernelContractOf(FuncOp entry) -> FailureOr<KernelContract>;

[[nodiscard]] auto kernelModuleContractOf(ModuleOp module, StringRef selected)
    -> FailureOr<KernelModuleContract>;

[[nodiscard]] auto stringifyKernelInvocationIdKind(KernelInvocationIdKind kind)
    -> llvm::StringRef;

[[nodiscard]] auto stringifyKernelBufferKind(KernelBufferKind kind)
    -> llvm::StringRef;

[[nodiscard]] auto kernelBufferScalarTypeName(const KernelResource& buffer)
    -> llvm::StringRef;

[[nodiscard]] auto kernelBufferComponents(const KernelResource& buffer)
    -> std::uint32_t;

[[nodiscard]] auto kernelBufferElementStride(const KernelResource& buffer)
    -> std::uint32_t;

inline constexpr llvm::StringLiteral kWebGPUStandardUniformLayout =
    "uniform_buffer_standard_layout";

[[nodiscard]] auto declaresAggregateUniforms(const KernelModuleContract& module)
    -> bool;

}  // namespace mlir::cxx
