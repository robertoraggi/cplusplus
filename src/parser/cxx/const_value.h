// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/literals_fwd.h>

#include <cstdint>
#include <variant>

namespace cxx {

using ConstValue =
    std::variant<bool, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t,
                 const StringLiteral*, float, double, long double>;

template <typename T>
struct ArithmeticConversion {
  auto operator()(const StringLiteral* value) const -> ConstValue {
    return ConstValue(value);
  }

  auto operator()(auto value) const -> ConstValue {
    return ConstValue(static_cast<T>(value));
  }
};

template <typename T>
struct ArithmeticCast {
  auto operator()(const StringLiteral*) const -> T {
    cxx_runtime_error("invalid artihmetic cast");
    return T{};
  }

  auto operator()(auto value) const -> T { return static_cast<T>(value); }
};

}  // namespace cxx