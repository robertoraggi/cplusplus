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

#include <cxx/cxx_fwd.h>

#include <cstdlib>
#include <string>

namespace cxx {

class Literal {
 public:
  explicit Literal(std::string value) : value_(std::move(value)) {}
  virtual ~Literal();

  const std::string& value() const { return value_; }

 private:
  std::string value_;
};

class IntegerLiteral final : public Literal {
 public:
  explicit IntegerLiteral(std::string text) : Literal(std::move(text)) {
    if (value().find('\'') == std::string_view::npos)
      integerValue_ = interpretText(value());
    else {
      std::string s;
      s.reserve(value().size());
      for (auto ch : value())
        if (ch != '\'') s += ch;
      integerValue_ = interpretText(s);
    }
  }

  std::uint64_t integerValue() const { return integerValue_; }

 private:
  static std::uint64_t interpretText(const std::string& s) {
    int base = 10;
    const char* str = s.c_str();
    if (s.starts_with("0x") || s.starts_with("0X")) {
      base = 16;
      str += 2;
    } else if (s.starts_with("0b") || s.starts_with("0B")) {
      base = 2;
      str += 2;
    } else if (s.starts_with('0')) {
      base = 8;
      ++str;
    }
    return (std::uint64_t)std::strtoull(str, nullptr, base);
  }

  std::uint64_t integerValue_ = 0;
};

class FloatLiteral final : public Literal {
 public:
  explicit FloatLiteral(std::string text) : Literal(std::move(text)) {
    floatValue_ = (double)std::strtold(value().c_str(), nullptr);
  }

  double floatValue() const { return floatValue_; }

 private:
  double floatValue_ = 0;
};

class StringLiteral final : public Literal {
 public:
  using Literal::Literal;
};

class CharLiteral final : public Literal {
 public:
  using Literal::Literal;
};

class CommentLiteral final : public Literal {
 public:
  using Literal::Literal;
};

}  // namespace cxx
