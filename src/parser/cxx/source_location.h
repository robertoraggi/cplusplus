// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <functional>
#include <tuple>

namespace cxx {

class SourceLocation;

using SourceLocationRange = std::tuple<SourceLocation, SourceLocation>;

class SourceLocation {
  unsigned int index_ = 0;

 public:
  SourceLocation() = default;

  SourceLocation(const SourceLocation&) = default;

  SourceLocation& operator=(const SourceLocation&) = default;

  explicit SourceLocation(unsigned int index) : index_(index) {}

  explicit operator bool() const { return index_ != 0; }

  explicit operator unsigned int() const { return index_; }

  unsigned int index() const { return index_; }

  SourceLocation next() const { return SourceLocation(index_ + 1); }

  SourceLocation previous() const {
    if (!index_) return *this;
    return SourceLocation(index_ - 1);
  }

  bool operator==(const SourceLocation& other) const {
    return index_ == other.index_;
  }

  bool operator!=(const SourceLocation& other) const {
    return index_ != other.index_;
  }

  bool operator<(const SourceLocation& other) const {
    return index_ < other.index_;
  }
};

}  // namespace cxx

template <>
struct std::hash<cxx::SourceLocation> {
  size_t operator()(cxx::SourceLocation loc) const { return loc.index(); }
};