// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/types_fwd.h>

namespace cxx {

class MemoryLayout {
 public:
  explicit MemoryLayout(int bits);
  ~MemoryLayout();

  auto bits() const -> int;
  auto sizeOfPointer() const -> int;
  auto sizeOfLong() const -> int;

  auto sizeOf(const Type* type) const -> int;
  auto alignmentOf(const Type* type) const -> int;

#define DECLARE_METHOD(type) int sizeOf(const type##Type* type) const;
  CXX_FOR_EACH_TYPE_KIND(DECLARE_METHOD)

#undef DECLARE_METHOD

#define DECLARE_METHOD(type) int alignmentOf(const type##Type* type) const;
  CXX_FOR_EACH_TYPE_KIND(DECLARE_METHOD)

 private:
  int bits_;
  int sizeOfPointer_;
  int sizeOfLong_;
};

#undef DECLARE_METHOD

}  // namespace cxx