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

#include <cxx/mlir/cxx_dialect.h>
#include <llvm/ADT/SmallVector.h>

#include <array>
#include <cstdint>

namespace mlir::cxx {

enum class KernelBufferKind { Storage, Uniform };

struct KernelBuffer {
  std::uint32_t parameter = 0;
  StringAttr name;
  std::uint32_t binding = 0;
  KernelBufferKind kind = KernelBufferKind::Storage;
  bool readOnly = false;
  bool unsignedElements = false;
  Type elementType;
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

[[nodiscard]] auto kernelContractOf(FuncOp entry) -> FailureOr<KernelContract>;

[[nodiscard]] auto stringifyKernelInvocationIdKind(KernelInvocationIdKind kind)
    -> llvm::StringRef;

[[nodiscard]] auto stringifyKernelBufferKind(KernelBufferKind kind)
    -> llvm::StringRef;

[[nodiscard]] auto kernelBufferScalarTypeName(const KernelBuffer& buffer)
    -> llvm::StringRef;

[[nodiscard]] auto kernelBufferComponents(const KernelBuffer& buffer)
    -> std::uint32_t;

}  // namespace mlir::cxx
