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
#include <functional>
#include <string_view>
#include <tuple>

namespace cxx {

class SourceLocation;

using SourceLocationRange = std::tuple<SourceLocation, SourceLocation>;

class SourceLocation {
  unsigned int index_ = 0;

 public:
  SourceLocation() = default;

  SourceLocation(const SourceLocation&) = default;

  auto operator=(const SourceLocation&) -> SourceLocation& = default;

  explicit SourceLocation(unsigned int index) : index_(index) {}

  explicit operator bool() const { return index_ != 0; }

  explicit operator unsigned int() const { return index_; }

  auto operator<=>(const SourceLocation&) const = default;

  [[nodiscard]] auto index() const -> unsigned int { return index_; }

  [[nodiscard]] auto next() const -> SourceLocation {
    return SourceLocation(index_ + 1);
  }

  [[nodiscard]] auto previous() const -> SourceLocation {
    if (!index_) return *this;
    return SourceLocation(index_ - 1);
  }

  auto operator==(const SourceLocation& other) const -> bool {
    return index_ == other.index_;
  }

  auto operator!=(const SourceLocation& other) const -> bool {
    return index_ != other.index_;
  }

  auto operator<(const SourceLocation& other) const -> bool {
    return index_ < other.index_;
  }
};

class SourcePosition {
 public:
  std::string_view fileName;
  std::uint32_t line = 0;
  std::uint32_t column = 0;
};

}  // namespace cxx

template <>
struct std::hash<cxx::SourceLocation> {
  auto operator()(cxx::SourceLocation loc) const -> size_t {
    return loc.index();
  }
};