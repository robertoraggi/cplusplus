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
#include <cxx/mlir/storage_address.h>
#include <llvm/ADT/SmallVector.h>

#include <cstdint>
#include <optional>

namespace mlir::cxx {

struct BufferLayoutMember {
  Type type;
  std::uint32_t position = 0;
  std::uint32_t offset = 0;
  std::uint32_t span = 0;
};

[[nodiscard]] auto valueElementPositions(Type aggregate)
    -> llvm::SmallVector<std::uint32_t>;

[[nodiscard]] auto isValueElement(Type aggregate, std::int64_t position)
    -> bool;

[[nodiscard]] auto valueElementIndex(Type aggregate, std::int64_t position)
    -> std::optional<std::uint32_t>;

[[nodiscard]] auto isBufferValueType(Type type) -> bool;

[[nodiscard]] auto isBufferLayoutType(Type type) -> bool;

[[nodiscard]] auto isBufferRecordType(Type type) -> bool;

[[nodiscard]] auto bufferLayoutSize(Type type) -> std::uint32_t;

[[nodiscard]] auto bufferLayoutExtent(Type type) -> std::uint32_t;

[[nodiscard]] auto bufferLayoutMembers(Type record)
    -> llvm::SmallVector<BufferLayoutMember>;

[[nodiscard]] auto designatesNonValueElement(const StorageAddress& address)
    -> bool;

}  // namespace mlir::cxx
