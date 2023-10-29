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

class DiagnosticsClient;

class Literal {
 public:
  explicit Literal(std::string value) : value_(std::move(value)) {}
  virtual ~Literal();

  [[nodiscard]] auto value() const -> const std::string& { return value_; }

  [[nodiscard]] auto hashCode() const -> std::size_t;

 private:
  std::string value_;
};

class IntegerLiteral final : public Literal {
 public:
  enum struct Radix {
    kDecimal,
    kHexadecimal,
    kOctal,
    kBinary,
  };

  struct Components {
    std::uint64_t value;
    std::string_view integerPart;
    std::string_view userSuffix;
    Radix radix = Radix::kDecimal;
    bool isUnsigned = false;
    bool isLongLong = false;
    bool isLong = false;
    bool hasSizeSuffix = false;

    [[nodiscard]] static auto from(std::string_view text,
                                   DiagnosticsClient* diagnostics = nullptr)
        -> Components;
  };

  explicit IntegerLiteral(std::string text);

  [[nodiscard]] auto integerValue() const -> std::uint64_t {
    return components_.value;
  }

  [[nodiscard]] auto components() const { return components_; }

  void initialize() const;

 private:
  std::uint64_t integerValue_ = 0;
  mutable Components components_;
};

class FloatLiteral final : public Literal {
 public:
  struct Components {
    enum class FloatingPointSuffix {
      kNone,
      kF,
      kL,
      kF16,
      kF32,
      kF64,
      kF128,
      kBF16,
    };

    double value = 0;
    FloatingPointSuffix suffix = FloatingPointSuffix::kNone;
    bool isDouble = false;
    bool isFloat = false;
    bool isLongDouble = false;

    [[nodiscard]] static auto from(std::string_view text,
                                   DiagnosticsClient* diagnostics = nullptr)
        -> Components;
  };

  explicit FloatLiteral(std::string text);

  [[nodiscard]] auto floatValue() const -> double { return components_.value; }

  [[nodiscard]] auto components() const { return components_; }

  void initialize() const;

 private:
  mutable Components components_;
};

class StringLiteral final : public Literal {
 public:
  using Literal::Literal;

  struct Components {
    std::string value;

    [[nodiscard]] static auto from(std::string_view text,
                                   DiagnosticsClient* diagnostics = nullptr)
        -> Components;
  };

  [[nodiscard]] auto stringValue() const -> std::string_view {
    return components_.value;
  };

  [[nodiscard]] auto components() const { return components_; }

  void initialize() const;

 private:
  mutable Components components_;
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

  struct Components {
    int value = 0;
    std::string_view prefix;

    [[nodiscard]] static auto from(std::string_view text,
                                   DiagnosticsClient* diagnostics = nullptr)
        -> Components;
  };

  [[nodiscard]] auto charValue() const -> int { return components_.value; }

  [[nodiscard]] auto components() const { return components_; }

  void initialize() const;

 private:
  mutable Components components_;
};

class CommentLiteral final : public Literal {
 public:
  using Literal::Literal;
};

}  // namespace cxx
