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

#include <cstdint>
#include <optional>

namespace cxx {

enum class MemoryOrder {
  kRelaxed,
  kAcquire,
  kRelease,
  kAcquireRelease,
  kSequentiallyConsistent,
};

enum class MemoryScope {
  kSystem,
  kDevice,
  kWorkgroup,
  kSubgroup,
  kInvocation,
};

[[nodiscard]] constexpr auto memoryOrderOf(std::intmax_t value)
    -> std::optional<MemoryOrder> {
  switch (value) {
    case 0:
      return MemoryOrder::kRelaxed;
    case 1:
    case 2:
      return MemoryOrder::kAcquire;
    case 3:
      return MemoryOrder::kRelease;
    case 4:
      return MemoryOrder::kAcquireRelease;
    case 5:
      return MemoryOrder::kSequentiallyConsistent;
    default:
      return std::nullopt;
  }
}

[[nodiscard]] constexpr auto memoryScopeOf(std::intmax_t value)
    -> std::optional<MemoryScope> {
  switch (value) {
    case 0:
      return MemoryScope::kSystem;
    case 1:
      return MemoryScope::kDevice;
    case 2:
      return MemoryScope::kWorkgroup;
    case 3:
      return MemoryScope::kSubgroup;
    case 4:
      return MemoryScope::kInvocation;
    default:
      return std::nullopt;
  }
}

}  // namespace cxx
