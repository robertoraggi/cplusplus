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

#include <cxx/qualified_type.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>

namespace cxx {

namespace {
TypePrinter printType;
}

std::ostream& operator<<(std::ostream& out, const QualifiedType& type) {
  printType(out, type);
  return out;
}

QualifiedType::QualifiedType(const Type* type, Qualifiers qualifiers) noexcept
    : type_(type), qualifiers_(qualifiers) {
  if (!type_) type_ = UndefinedType::get();
}

QualifiedType::operator bool() const noexcept {
  return type_ != UndefinedType::get();
}

void QualifiedType::setType(const Type* type) {
  type_ = type;
  if (!type_) type_ = UndefinedType::get();
}

}  // namespace cxx
