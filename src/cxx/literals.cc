// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/literals.h>

#include <charconv>

namespace cxx {

Literal::~Literal() {}

IntegerLiteral::IntegerLiteral(std::string text) : Literal(std::move(text)) {
  if (value().find('\'') == std::string_view::npos) {
    integerValue_ = interpretText(value());
    return;
  }

  std::string s;
  s.reserve(value().size());

  for (auto ch : value()) {
    if (ch != '\'') s += ch;
  }

  integerValue_ = interpretText(s);
}

std::uint64_t IntegerLiteral::interpretText(std::string_view text) {
  while (text.ends_with('l') || text.ends_with('L')  //
         || text.ends_with('u') || text.ends_with('U')) {
    text = text.substr(0, text.length() - 1);
  }

  int base = 10;

  if (text.starts_with('0')) {
    text = text.substr(1);

    if (text.starts_with('x') || text.starts_with('X')) {
      text = text.substr(1);
      base = 16;
    } else if (text.starts_with('b') || text.starts_with('B')) {
      text = text.substr(1);
      base = 2;
    } else {
      base = 8;
    }
  }

  std::uint64_t value = 0;
  auto result = std::from_chars(begin(text), end(text), value, base);
  return value;
}

FloatLiteral::FloatLiteral(std::string text) : Literal(std::move(text)) {
  std::string_view str(value());
  std::from_chars(begin(str), end(str), floatValue_);
}

}  // namespace cxx
