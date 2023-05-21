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

#include <cxx/types_fwd.h>

#include <iosfwd>

namespace cxx {

class QualifiedType {
 public:
  explicit QualifiedType(const Type* type = nullptr,
                         Qualifiers qualifiers = Qualifiers::kNone) noexcept;

  explicit operator bool() const noexcept;

  auto operator->() const noexcept -> const Type* { return type_; }

  [[nodiscard]] auto type() const -> const Type* { return type_; }
  void setType(const Type* type);

  [[nodiscard]] auto qualifiers() const -> Qualifiers { return qualifiers_; }
  void setQualifiers(Qualifiers qualifiers) { qualifiers_ = qualifiers; }

  auto unqualified() -> QualifiedType { return QualifiedType(type()); }

  void mergeWith(const QualifiedType& other) {
    qualifiers_ |= other.qualifiers();
    if (other) setType(other.type());
  }

  [[nodiscard]] auto isConst() const -> bool {
    return (qualifiers_ & Qualifiers::kConst) != Qualifiers::kNone;
  }

  [[nodiscard]] auto isVolatile() const -> bool {
    return (qualifiers_ & Qualifiers::kVolatile) != Qualifiers::kNone;
  }

  [[nodiscard]] auto isRestrict() const -> bool {
    return (qualifiers_ & Qualifiers::kRestrict) != Qualifiers::kNone;
  }

  auto operator==(const QualifiedType& other) const -> bool {
    return type_ == other.type_ && qualifiers_ == other.qualifiers_;
  }

  auto operator!=(const QualifiedType& other) const -> bool {
    return !operator==(other);
  }

 private:
  const Type* type_;
  Qualifiers qualifiers_;
};

auto operator<<(std::ostream& out, const QualifiedType& type) -> std::ostream&;

}  // namespace cxx
