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

  const Type* operator->() const noexcept { return type_; }

  const Type* type() const { return type_; }
  void setType(const Type* type);

  Qualifiers qualifiers() const { return qualifiers_; }
  void setQualifiers(Qualifiers qualifiers) { qualifiers_ = qualifiers; }

  QualifiedType unqualified() { return QualifiedType(type()); }

  void mergeWith(const QualifiedType& other) {
    qualifiers_ |= other.qualifiers();
    if (other) setType(other.type());
  }

  bool isConst() const {
    return (qualifiers_ & Qualifiers::kConst) != Qualifiers::kNone;
  }

  bool isVolatile() const {
    return (qualifiers_ & Qualifiers::kVolatile) != Qualifiers::kNone;
  }

  bool isRestrict() const {
    return (qualifiers_ & Qualifiers::kRestrict) != Qualifiers::kNone;
  }

  bool operator==(const QualifiedType& other) const {
    return type_ == other.type_ && qualifiers_ == other.qualifiers_;
  }

  bool operator!=(const QualifiedType& other) const {
    return !operator==(other);
  }

 private:
  const Type* type_;
  Qualifiers qualifiers_;
};

std::ostream& operator<<(std::ostream& out, const QualifiedType& type);

}  // namespace cxx
