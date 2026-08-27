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

#include <cxx/types_fwd.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace cxx {
class MemoryLayout {
 public:
  explicit MemoryLayout(std::size_t bits);
  ~MemoryLayout();

  [[nodiscard]] auto bits() const -> std::size_t;
  [[nodiscard]] auto sizeOfSizeType() const -> std::size_t;
  [[nodiscard]] auto sizeOfPointer() const -> std::size_t;
  [[nodiscard]] auto sizeOfLong() const -> std::size_t;
  [[nodiscard]] auto sizeOfLongLong() const -> std::size_t;
  [[nodiscard]] auto sizeOfLongDouble() const -> std::size_t;
  [[nodiscard]] auto longDoubleMantissaDigits() const -> std::size_t;
  [[nodiscard]] auto sizeOfWideChar() const -> std::size_t;
  [[nodiscard]] auto isWideCharSigned() const -> bool;

  void setSizeOfPointer(std::size_t sizeOfPointer);
  void setSizeOfLong(std::size_t sizeOfLong);
  void setSizeOfLongLong(std::size_t sizeOfLongLong);
  void setSizeOfLongDouble(std::size_t sizeOfLongDouble,
                           std::size_t mantissaDigits);
  void setWideCharUnderlyingType(std::size_t size, bool isSigned);

  [[nodiscard]] auto sizeOf(const Type* type) const
      -> std::optional<std::size_t>;

  [[nodiscard]] auto alignmentOf(const Type* type) const
      -> std::optional<std::size_t>;

  [[nodiscard]] auto triple() const -> const std::string&;
  [[nodiscard]] auto arch() const -> std::string_view;
  [[nodiscard]] auto usesArmMemberPointerAbi() const -> bool;

  [[nodiscard]] auto nullMemberObjectPointer() const -> std::int64_t;
  [[nodiscard]] auto usesSingleScalarClassAbi() const -> bool;
  void setTriple(std::string triple);

 private:
  std::size_t bits_ = 0;
  std::size_t sizeOfPointer_ = 0;
  std::size_t sizeOfLong_ = 0;
  std::size_t sizeOfLongLong_ = 0;
  std::size_t sizeOfLongDouble_ = 0;
  std::size_t longDoubleMantissaDigits_ = 0;
  std::size_t sizeOfWideChar_ = 4;
  bool wideCharIsSigned_ = true;
  std::string triple_;
};

#undef DECLARE_METHOD
}  // namespace cxx
