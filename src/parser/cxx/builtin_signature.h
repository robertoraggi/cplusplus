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

#include <cxx/token_fwd.h>
#include <cxx/types_fwd.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cxx {

class Control;

enum class BuiltinFlags : std::uint8_t {
  kNone = 0,
  kConstexpr = 1,
  kConsteval = 2,
  kNoexcept = 4,
  kNoReturn = 8,
};

[[nodiscard]] constexpr auto operator|(BuiltinFlags a, BuiltinFlags b)
    -> BuiltinFlags {
  return BuiltinFlags(std::to_underlying(a) | std::to_underlying(b));
}

[[nodiscard]] constexpr auto operator&(BuiltinFlags a, BuiltinFlags b)
    -> BuiltinFlags {
  return BuiltinFlags(std::to_underlying(a) & std::to_underlying(b));
}

[[nodiscard]] constexpr auto contains(BuiltinFlags flags, BuiltinFlags flag)
    -> bool {
  return (flags & flag) != BuiltinFlags::kNone;
}

struct BuiltinSignature {
  std::size_t offset = 0;
  std::size_t count = 0;
  BuiltinFlags flags = BuiltinFlags::kNone;
};

[[nodiscard]] auto builtinSignatureOf(BuiltinFunctionKind kind)
    -> BuiltinSignature;

enum class BuiltinOverloadStatus : std::uint8_t {
  kOk,
  kUnavailable,
  kInvalid,
};

struct BuiltinOverload {
  const FunctionType* type = nullptr;
  BuiltinOverloadStatus status = BuiltinOverloadStatus::kInvalid;
};

[[nodiscard]] auto decodeBuiltinSignature(Control* control,
                                          BuiltinFunctionKind kind,
                                          std::size_t index) -> BuiltinOverload;

}  // namespace cxx
