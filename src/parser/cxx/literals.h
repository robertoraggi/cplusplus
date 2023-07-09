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

#include <cxx/cxx_fwd.h>

#include <string>
#include <string_view>

namespace cxx {

class Literal {
 public:
  explicit Literal(std::string value) : value_(std::move(value)) {}
  virtual ~Literal();

  [[nodiscard]] auto value() const -> const std::string& { return value_; }

  auto hashCode() const -> std::size_t;

 private:
  std::string value_;
};

class IntegerLiteral final : public Literal {
 public:
  explicit IntegerLiteral(std::string text);

  [[nodiscard]] auto integerValue() const -> std::uint64_t {
    return integerValue_;
  }

  static auto interpretText(std::string_view text) -> std::uint64_t;

 private:
  std::uint64_t integerValue_ = 0;
};

class FloatLiteral final : public Literal {
 public:
  explicit FloatLiteral(std::string text);

  [[nodiscard]] auto floatValue() const -> double { return floatValue_; }

 private:
  double floatValue_ = 0;
};

class StringLiteral final : public Literal {
 public:
  using Literal::Literal;
};

class WideStringLiteral final : public Literal {
 public:
  using Literal::Literal;
};

class Utf8StringLiteral final : public Literal {
 public:
  using Literal::Literal;
};

class Utf16StringLiteral final : public Literal {
 public:
  using Literal::Literal;
};

class Utf32StringLiteral final : public Literal {
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
