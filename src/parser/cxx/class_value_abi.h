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

#include <cxx/cxx_fwd.h>
#include <cxx/types_fwd.h>

#include <cstdint>
#include <vector>

namespace cxx {

enum class ClassValueAbiContext { Argument, Return };

struct ClassValueAbiSlot {
  const Type* type = nullptr;
  std::uint64_t offset = 0;
};

struct ClassValueAbi {
  enum class Kind { Direct, Empty, Coerce, Indirect };

  Kind kind = Kind::Direct;

  bool passedInMemory = false;
  std::uint64_t indirectAlignment = 0;

  std::vector<ClassValueAbiSlot> slots;
  std::uint64_t coerceSize = 0;
  std::uint64_t coerceAlignment = 0;
};

[[nodiscard]] auto usesClassValueAbi(const Type* type) -> bool;

[[nodiscard]] auto classifyClassValueAbi(TranslationUnit* unit,
                                         const Type* type,
                                         ClassValueAbiContext context)
    -> ClassValueAbi;

[[nodiscard]] auto isClassValueDestroyedInCallee(const Type* type) -> bool;

}  // namespace cxx
