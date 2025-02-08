// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

// cxx
#include <cxx/diagnostics_client.h>

#include <charconv>
#include <format>

namespace cxx {

namespace {

template <typename String>
struct StringLiteralParser {
  std::string_view text;
  DiagnosticsClient *diagnostics;
  String value;
  int pos = 0;

  explicit StringLiteralParser(std::string_view text,
                               DiagnosticsClient *diagnostics)
      : text(text), diagnostics(diagnostics) {}

  [[nodiscard]] auto LA(int n = 0) -> int {
    auto p = pos + n;
    if (p < 0 || p >= text.size()) return 0;
    return text[pos + n];
  }

  void consume(int n = 1) { pos += n; }

  [[nodiscard]] auto isOctDigit(int ch) -> bool {
    return ch >= '0' && ch <= '7';
  }

  [[nodiscard]] auto isHexDigit(int ch) -> bool {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
           (ch >= 'A' && ch <= 'F');
  }

  [[nodiscard]] auto hexDigitValue(int hexDigit) -> int {
    if (hexDigit >= '0' && hexDigit <= '9') {
      return hexDigit - '0';
    } else if (hexDigit >= 'a' && hexDigit <= 'f') {
      return hexDigit - 'a' + 10;
    } else if (hexDigit >= 'A' && hexDigit <= 'F') {
      return hexDigit - 'A' + 10;
    } else {
      cxx_runtime_error("invalid hex digit");
      return 0;
    }
  }

  [[nodiscard]] auto octDigitValue(int octDigit) -> int {
    return octDigit - '0';
  }

  void parseSimpleHexadecimalEscapeSequence() {
    int x = 0;
    while (isHexDigit(LA())) {
      x = x * 16 + hexDigitValue(LA());
      consume();
    }
    value += static_cast<char>(x);
  }

  [[nodiscard]] auto parseHexadecimalEscapeSequence() -> bool {
    if (LA() != 'x') return false;
    consume();
    parseSimpleHexadecimalEscapeSequence();
    return true;
  }

  [[nodiscard]] auto parseOctalEscapeSequence() -> bool {
    if (!isOctDigit(LA())) return false;
    int x = octDigitValue(LA());
    consume();
    if (isOctDigit(LA())) {
      x = x * 8 + octDigitValue(LA());
      consume();
      if (isOctDigit(LA())) {
        x = x * 8 + octDigitValue(LA());
        consume();
      }
    }
    value += static_cast<char>(x);
    return true;
  }

  [[nodiscard]] auto parseNumericEscapeSequence() -> bool {
    return parseHexadecimalEscapeSequence() || parseOctalEscapeSequence();
  }

  [[nodiscard]] auto parseSimpleEscapeSequence() -> bool {
    switch (LA()) {
      case '\'':
        value += '\'';
        consume();
        return true;
      case '"':
        value += '"';
        consume();
        return true;
      case '?':
        value += '?';
        consume();
        return true;
      case '\\':
        value += '\\';
        consume();
        return true;
      case 'a':
        value += '\a';
        consume();
        return true;
      case 'b':
        value += '\b';
        consume();
        return true;
      case 'f':
        value += '\f';
        consume();
        return true;
      case 'n':
        value += '\n';
        consume();
        return true;
      case 'r':
        value += '\r';
        consume();
        return true;
      case 't':
        value += '\t';
        consume();
        return true;
      case 'v':
        value += '\v';
        consume();
        return true;
      default:
        return false;
    }  // switch
  };

  [[nodiscard]] auto parseConditionalEscapeSequence() -> bool { return true; };

  [[nodiscard]] auto parseEscapeSequence() -> bool {
    const auto ch = LA();
    if (ch != '\\') return false;
    consume();
    return parseNumericEscapeSequence() || parseSimpleEscapeSequence() ||
           parseConditionalEscapeSequence();
  };

  void parseStringLiteral() {
    // TODO: handle encoding prefix

    while (auto ch = LA()) {
      if (ch == '"') break;
      consume();
    }

    if (LA() == '"') {
      consume();

      while (const auto ch = LA()) {
        if (ch == '"') break;

        if (parseEscapeSequence()) continue;

        value += ch;
        consume();
      }
    }
  }

  void parseCharLiteral() {
    // TODO: handle encoding prefix
    while (auto ch = LA()) {
      if (ch == '\'') break;
      consume();
    }

    if (LA() == '\'') {
      consume();

      while (const auto ch = LA()) {
        if (ch == '\'') break;

        if (parseEscapeSequence()) continue;

        value += ch;
        consume();
      }
    }
  }
};

using FloatingPointSuffix = FloatLiteral::Components::FloatingPointSuffix;

auto to_string(FloatingPointSuffix suffix) -> std::string_view {
  switch (suffix) {
    case FloatingPointSuffix::kNone:
      return "";
    case FloatingPointSuffix::kF:
      return "f";
    case FloatingPointSuffix::kL:
      return "l";
    case FloatingPointSuffix::kF16:
      return "f16";
    case FloatingPointSuffix::kF32:
      return "f32";
    case FloatingPointSuffix::kF64:
      return "f64";
    case FloatingPointSuffix::kBF16:
      return "bf16";
    case FloatingPointSuffix::kF128:
      return "f128";
    default:
      cxx_runtime_error("invalid floating point suffix");
  }  // switch
}

auto classifyFloatingPointSuffix(std::string_view s) -> FloatingPointSuffix {
  auto classifyFloatingPointSuffix1 =
      [](std::string_view s) -> FloatingPointSuffix {
    if (s[0] == 'F') {
      return FloatingPointSuffix::kF;
    } else if (s[0] == 'L') {
      return FloatingPointSuffix::kL;
    } else if (s[0] == 'f') {
      return FloatingPointSuffix::kF;
    } else if (s[0] == 'l') {
      return FloatingPointSuffix::kL;
    }
    return FloatingPointSuffix::kNone;
  };

  auto classifyFloatingPointSuffix3 =
      [](std::string_view s) -> FloatingPointSuffix {
    if (s[0] == 'F') {
      if (s[1] == '1') {
        if (s[2] == '6') {
          return FloatingPointSuffix::kF16;
        }
      } else if (s[1] == '3') {
        if (s[2] == '2') {
          return FloatingPointSuffix::kF32;
        }
      } else if (s[1] == '6') {
        if (s[2] == '4') {
          return FloatingPointSuffix::kF64;
        }
      }
    } else if (s[0] == 'f') {
      if (s[1] == '1') {
        if (s[2] == '6') {
          return FloatingPointSuffix::kF16;
        }
      } else if (s[1] == '3') {
        if (s[2] == '2') {
          return FloatingPointSuffix::kF32;
        }
      } else if (s[1] == '6') {
        if (s[2] == '4') {
          return FloatingPointSuffix::kF64;
        }
      }
    }
    return FloatingPointSuffix::kNone;
  };

  auto classifyFloatingPointSuffix4 =
      [](std::string_view s) -> FloatingPointSuffix {
    if (s[0] == 'B') {
      if (s[1] == 'F') {
        if (s[2] == '1') {
          if (s[3] == '6') {
            return FloatingPointSuffix::kBF16;
          }
        }
      }
    } else if (s[0] == 'F') {
      if (s[1] == '1') {
        if (s[2] == '2') {
          if (s[3] == '8') {
            return FloatingPointSuffix::kF128;
          }
        }
      }
    } else if (s[0] == 'b') {
      if (s[1] == 'f') {
        if (s[2] == '1') {
          if (s[3] == '6') {
            return FloatingPointSuffix::kBF16;
          }
        }
      }
    } else if (s[0] == 'f') {
      if (s[1] == '1') {
        if (s[2] == '2') {
          if (s[3] == '8') {
            return FloatingPointSuffix::kF128;
          }
        }
      }
    }
    return FloatingPointSuffix::kNone;
  };

  switch (s.length()) {
    case 1:
      return classifyFloatingPointSuffix1(s);
    case 3:
      return classifyFloatingPointSuffix3(s);
    case 4:
      return classifyFloatingPointSuffix4(s);
    default:
      return FloatingPointSuffix::kNone;
  }  // switch
}

}  // namespace

Literal::~Literal() = default;

auto Literal::hashCode() const -> std::size_t {
  return std::hash<std::string>{}(value_);
}

IntegerLiteral::IntegerLiteral(std::string text) : Literal(std::move(text)) {}

void IntegerLiteral::initialize() const {
  components_ = Components::from(value(), nullptr);
}

auto IntegerLiteral::Components::from(std::string_view text,
                                      DiagnosticsClient *diagnostics)
    -> Components {
  std::string integerPart;
  integerPart.reserve(text.size());

  Radix radix = Radix::kDecimal;

  bool hasUnsignedSuffix = false;
  bool hasLongLongSuffix = false;
  bool hasLongSuffix = false;
  bool hasSizeSuffix = false;

  std::size_t pos = 0;
  auto LA = [&](int n = 0) -> int {
    auto p = pos + n;
    if (p < 0 || p >= text.size()) return 0;
    return text[pos + n];
  };
  auto consume = [&](int n = 1) { pos += n; };

  auto parseUnsignedSuffix = [&] {
    if (const auto ch = LA(); ch == 'u' || ch == 'U') {
      consume(1);
      hasUnsignedSuffix = true;
      return true;
    }
    return false;
  };

  auto parseLongOrLongLongSuffix = [&] {
    if (const auto ch = LA(); ch == 'l' || ch == 'L') {
      consume(1);
      if (const auto ch = LA(); ch == 'l' || ch == 'L') {
        consume(1);
        hasLongLongSuffix = true;
      } else {
        hasLongSuffix = true;
      }
      return true;
    }
    return false;
  };

  auto parseSizeSuffix = [&] {
    if (const auto ch = LA(); ch == 'z' || ch == 'Z') {
      consume(1);
      hasSizeSuffix = true;
      return true;
    }
    return false;
  };

  auto parseOptionalIntegerSuffix = [&] {
    if (parseUnsignedSuffix()) {
      if (!parseLongOrLongLongSuffix()) parseSizeSuffix();
    } else if (parseLongOrLongLongSuffix()) {
      parseUnsignedSuffix();
    } else if (parseSizeSuffix()) {
      parseUnsignedSuffix();
    }
  };

  auto isDigit = [&](int ch) { return ch >= '0' && ch <= '9'; };
  auto isBinDigit = [&](int ch) { return ch == '0' || ch == '1'; };
  auto isOctDigit = [&](int ch) { return ch >= '0' && ch <= '7'; };

  auto isHexDigit = [&](int ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
           (ch >= 'A' && ch <= 'F');
  };

  auto parseBinLiteral = [&] {
    const auto start = pos;

    const auto ch1 = LA();
    const auto ch2 = LA(1);
    const auto ch3 = LA(2);

    if (ch1 != '0' || (ch2 != 'b' && ch2 != 'B') || !isBinDigit(ch3))
      return false;

    consume(2);

    integerPart.clear();
    while (const auto ch = LA()) {
      if (isBinDigit(ch)) {
        integerPart += ch;
        consume();
      } else if (ch == '\'' && isBinDigit(LA(1))) {
        integerPart += LA(1);
        consume(2);
      } else {
        break;
      }
    }

    if (isDigit(LA())) {
      pos = start;
      return false;
    }

    radix = Radix::kBinary;

    return true;
  };

  auto parseHexLiteral = [&] {
    const auto start = pos;
    const auto ch2 = LA(1);

    if (LA() != '0' || (ch2 != 'x' && ch2 != 'X') || !isHexDigit(LA(2)))
      return false;

    consume(2);

    integerPart.clear();
    while (const auto ch = LA()) {
      if (isHexDigit(ch)) {
        integerPart += ch;
        consume();
      } else if (ch == '\'' && isHexDigit(LA(1))) {
        integerPart += LA(1);
        consume(2);
      } else {
        break;
      }
    }

    if (isDigit(LA())) {
      pos = start;
      return false;
    }

    radix = Radix::kHexadecimal;

    return true;
  };

  auto parseOctLiteral = [&] {
    const auto start = pos;

    if (LA() != '0' || !isOctDigit(LA(1))) return false;
    consume();

    integerPart.clear();
    while (const auto ch = LA()) {
      if (isOctDigit(ch)) {
        integerPart += ch;
        consume();
      } else if (ch == '\'' && isOctDigit(LA(1))) {
        integerPart += LA(1);
        consume(2);
      } else {
        break;
      }
    }

    if (isDigit(LA())) {
      pos = start;
      return false;
    }

    radix = Radix::kOctal;

    return true;
  };

  auto parseDecLiteral = [&] {
    integerPart.clear();
    while (const auto ch = LA()) {
      if (isDigit(ch)) {
        integerPart += ch;
        consume();
      } else if (ch == '\'' && isDigit(LA(1))) {
        integerPart += LA(1);
        consume(2);
      } else {
        break;
      }
    }

    radix = Radix::kDecimal;
  };

  if (!parseHexLiteral() && !parseOctLiteral() && !parseBinLiteral()) {
    parseDecLiteral();
  }

  parseOptionalIntegerSuffix();

  int base = 10;
  if (radix == Radix::kHexadecimal)
    base = 16;
  else if (radix == Radix::kOctal)
    base = 8;
  else if (radix == Radix::kBinary)
    base = 2;

  std::uint64_t value = 0;
  auto firstChar = integerPart.data();
  auto lastChar = firstChar + integerPart.length();
  auto result = std::from_chars(firstChar, lastChar, value, base);
  (void)result;

  return Components{
      .value = value,
      .integerPart = text.substr(0, pos),
      .userSuffix = text.substr(pos),
      .radix = radix,
      .isUnsigned = hasUnsignedSuffix,
      .isLongLong = hasLongLongSuffix,
      .isLong = hasLongSuffix,
      .hasSizeSuffix = hasSizeSuffix,
  };
}

FloatLiteral::FloatLiteral(std::string text) : Literal(std::move(text)) {}

void FloatLiteral::initialize() const {
  components_ = Components::from(value());
}

auto FloatLiteral::Components::from(std::string_view text,
                                    DiagnosticsClient *diagnostics)
    -> Components {
  std::size_t pos = 0;

  auto LA = [&](int n = 0) -> int {
    auto p = pos + n;
    if (p < 0 || p >= text.size()) return 0;
    return text[pos + n];
  };

  auto isDigit = [&](int ch) { return ch >= '0' && ch <= '9'; };

  auto isHexDigit = [&](int ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
           (ch >= 'A' && ch <= 'F');
  };

  auto consume = [&](int n = 1) { pos += n; };

  Components components;

  std::string literalText;
  literalText.reserve(text.size());

  auto parseDigitSequence = [&] {
    const auto start = pos;
    while (true) {
      if (isDigit(LA())) {
        literalText += LA();
        consume();
      } else if (LA() == '\'' && isDigit(LA(1))) {
        literalText += LA(1);
        consume(2);
      } else {
        break;
      }
    }
    return pos != start;
  };

  auto parseExponentPart = [&]() -> std::string_view {
    const auto start = pos;
    const auto literalLength = literalText.length();
    if (LA() != 'e' && LA() != 'E') return {};
    literalText += LA();
    consume();
    if (LA() == '+' || LA() == '-') {
      literalText += LA();
      consume();
    }
    if (parseDigitSequence()) return text.substr(start, pos - start);
    pos = start;
    literalText.resize(literalLength);
    return {};
  };

  auto parseBinaryExponentPart = [&]() -> std::string_view {
    const auto start = pos;
    const auto literalLength = literalText.length();
    if (LA() != 'p' && LA() != 'P') return {};
    literalText += LA();
    consume();
    if (LA() == '+' || LA() == '-') {
      literalText += LA();
      consume();
    }
    if (parseDigitSequence()) return text.substr(start, pos - start);
    pos = start;
    literalText.resize(literalLength);
    return {};
  };

  auto parseOptionalFloatingPointSuffix = [&]() -> std::string_view {
    const auto suffix = text.substr(pos);
    components.suffix = classifyFloatingPointSuffix(suffix);
    if (components.suffix == FloatingPointSuffix::kNone) return {};
    return suffix;
  };

  auto parseDecimalFloatPointLiteral = [&] {
    if (LA() == '.') {
      literalText += LA();
      consume();
      parseDigitSequence();
    } else {
      parseDigitSequence();
      if (LA() == '.') {
        literalText += LA();
        consume();
        parseDigitSequence();
      }
    }

    parseExponentPart();
    parseOptionalFloatingPointSuffix();
  };

  auto parseHexadecimalPrefix = [&] {
    if (LA() != '0' && (LA(1) != 'x' || LA(1) != 'X')) return false;
    literalText += LA();
    literalText += LA(1);
    consume(2);
    return true;
  };

  auto parseHexadecimalDigitSequence = [&] {
    const auto start = pos;
    while (const auto ch = LA()) {
      if (isHexDigit(ch)) {
        literalText += ch;
        consume();
      } else if (ch == '\'' && isHexDigit(LA(1))) {
        literalText += LA(1);
        consume(2);
      } else {
        break;
      }
    }
    return pos != start;
  };

  auto parseHexadecimalFloatingPointLiteral = [&] {
    if (!parseHexadecimalPrefix()) return false;

    if (LA() == '.' && isHexDigit(LA(1))) {
      literalText += LA();
      consume();
      parseHexadecimalDigitSequence();
    } else {
      (void)parseHexadecimalDigitSequence();

      if (LA() == '.' && isHexDigit(LA(1))) {
        literalText += LA();
        consume();
        parseHexadecimalDigitSequence();
      }
    }

    parseBinaryExponentPart();
    parseOptionalFloatingPointSuffix();

    return true;
  };

  if (!parseHexadecimalFloatingPointLiteral()) {
    parseDecimalFloatPointLiteral();
  }

  const auto firstChar = literalText.data();
  components.value = strtod(firstChar, nullptr);

  components.isFloat = components.suffix == FloatingPointSuffix::kF;
  components.isLongDouble = components.suffix == FloatingPointSuffix::kL;
  components.isDouble = !components.isFloat && !components.isLongDouble;

  return components;
}

auto StringLiteral::Components::from(std::string_view text,
                                     DiagnosticsClient *diagnostics)
    -> Components {
  StringLiteralParser<std::string> parser(text, diagnostics);

  parser.parseStringLiteral();

  Components components;
  components.value = std::move(parser.value);

  return components;
}

void StringLiteral::initialize() const {
  components_ = Components::from(value());
}

auto CharLiteral::Components::from(std::string_view text,
                                   DiagnosticsClient *diagnostics)
    -> Components {
  StringLiteralParser<std::string> parser(text, diagnostics);

  parser.parseCharLiteral();

  Components components;
  components.prefix = text.substr(0, text.find_first_of('\''));
  components.value = !parser.value.empty() ? parser.value[0] : 0;

  return components;
}

void CharLiteral::initialize() const {
  components_ = Components::from(value());
}

}  // namespace cxx
