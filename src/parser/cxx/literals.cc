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

#include <cxx/literals.h>

// cxx
#include <cxx/diagnostics_client.h>

#include <charconv>
#include <cstdint>
#include <format>

namespace cxx {

namespace {

void encodeUtf8(std::uint32_t cp, std::string& out) {
  if (cp <= 0x7F) {
    out += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0x10FFFF) {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

void encodeUtf16LE(std::uint32_t cp, std::string& out) {
  if (cp <= 0xFFFF) {
    auto v = static_cast<std::uint16_t>(cp);
    out += static_cast<char>(v & 0xFF);
    out += static_cast<char>((v >> 8) & 0xFF);
  } else if (cp <= 0x10FFFF) {
    cp -= 0x10000;
    auto hi = static_cast<std::uint16_t>(0xD800 | (cp >> 10));
    auto lo = static_cast<std::uint16_t>(0xDC00 | (cp & 0x3FF));
    out += static_cast<char>(hi & 0xFF);
    out += static_cast<char>((hi >> 8) & 0xFF);
    out += static_cast<char>(lo & 0xFF);
    out += static_cast<char>((lo >> 8) & 0xFF);
  }
}

void encodeUtf32LE(std::uint32_t cp, std::string& out) {
  out += static_cast<char>(cp & 0xFF);
  out += static_cast<char>((cp >> 8) & 0xFF);
  out += static_cast<char>((cp >> 16) & 0xFF);
  out += static_cast<char>((cp >> 24) & 0xFF);
}

template <typename String>
struct StringLiteralParser {
  std::string_view text;
  DiagnosticsClient* diagnostics;
  String value;
  int pos = 0;
  StringLiteralEncoding encoding = StringLiteralEncoding::kNone;

  explicit StringLiteralParser(
      std::string_view text, DiagnosticsClient* diagnostics,
      StringLiteralEncoding encoding = StringLiteralEncoding::kNone)
      : text(text), diagnostics(diagnostics), encoding(encoding) {}

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
    emitCodeUnit(x);
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
    emitCodeUnit(x);
    return true;
  }

  [[nodiscard]] auto parseUniversalCharacterName4() -> bool {
    if (LA() != 'u') return false;
    if (!isHexDigit(LA(1)) || !isHexDigit(LA(2)) || !isHexDigit(LA(3)) ||
        !isHexDigit(LA(4)))
      return false;
    consume();
    std::uint32_t cp = 0;
    for (int i = 0; i < 4; ++i) {
      cp = cp * 16 + hexDigitValue(LA());
      consume();
    }
    emitCodePoint(cp);
    return true;
  }

  [[nodiscard]] auto parseUniversalCharacterName8() -> bool {
    if (LA() != 'U') return false;
    for (int i = 1; i <= 8; ++i) {
      if (!isHexDigit(LA(i))) return false;
    }
    consume();
    std::uint32_t cp = 0;
    for (int i = 0; i < 8; ++i) {
      cp = cp * 16 + hexDigitValue(LA());
      consume();
    }
    emitCodePoint(cp);
    return true;
  }

  [[nodiscard]] auto parseUniversalCharacterName() -> bool {
    return parseUniversalCharacterName4() || parseUniversalCharacterName8();
  }

  void emitCodeUnit(int x) {
    switch (encoding) {
      case StringLiteralEncoding::kNone:
      case StringLiteralEncoding::kUtf8:
        value += static_cast<char>(x);
        break;
      case StringLiteralEncoding::kUtf16: {
        auto v = static_cast<std::uint16_t>(x);
        value += static_cast<char>(v & 0xFF);
        value += static_cast<char>((v >> 8) & 0xFF);
        break;
      }
      case StringLiteralEncoding::kUtf32:
      case StringLiteralEncoding::kWide: {
        auto v = static_cast<std::uint32_t>(x);
        value += static_cast<char>(v & 0xFF);
        value += static_cast<char>((v >> 8) & 0xFF);
        value += static_cast<char>((v >> 16) & 0xFF);
        value += static_cast<char>((v >> 24) & 0xFF);
        break;
      }
    }
  }

  void emitCodePoint(std::uint32_t cp) {
    switch (encoding) {
      case StringLiteralEncoding::kNone:
      case StringLiteralEncoding::kUtf8:
        encodeUtf8(cp, value);
        break;
      case StringLiteralEncoding::kUtf16:
        encodeUtf16LE(cp, value);
        break;
      case StringLiteralEncoding::kUtf32:
      case StringLiteralEncoding::kWide:
        encodeUtf32LE(cp, value);
        break;
    }
  }

  void emitChar(char ch) {
    switch (encoding) {
      case StringLiteralEncoding::kNone:
      case StringLiteralEncoding::kUtf8:
        value += ch;
        break;
      case StringLiteralEncoding::kUtf16: {
        auto v = static_cast<std::uint16_t>(static_cast<unsigned char>(ch));
        value += static_cast<char>(v & 0xFF);
        value += static_cast<char>((v >> 8) & 0xFF);
        break;
      }
      case StringLiteralEncoding::kUtf32:
      case StringLiteralEncoding::kWide: {
        auto v = static_cast<std::uint32_t>(static_cast<unsigned char>(ch));
        value += static_cast<char>(v & 0xFF);
        value += static_cast<char>((v >> 8) & 0xFF);
        value += static_cast<char>((v >> 16) & 0xFF);
        value += static_cast<char>((v >> 24) & 0xFF);
        break;
      }
    }
  }

  [[nodiscard]] auto parseNumericEscapeSequence() -> bool {
    return parseHexadecimalEscapeSequence() || parseOctalEscapeSequence();
  }

  [[nodiscard]] auto parseSimpleEscapeSequence() -> bool {
    switch (LA()) {
      case '\'':
        emitChar('\'');
        consume();
        return true;
      case '"':
        emitChar('"');
        consume();
        return true;
      case '?':
        emitChar('?');
        consume();
        return true;
      case '\\':
        emitChar('\\');
        consume();
        return true;
      case 'a':
        emitChar('\a');
        consume();
        return true;
      case 'b':
        emitChar('\b');
        consume();
        return true;
      case 'f':
        emitChar('\f');
        consume();
        return true;
      case 'n':
        emitChar('\n');
        consume();
        return true;
      case 'r':
        emitChar('\r');
        consume();
        return true;
      case 't':
        emitChar('\t');
        consume();
        return true;
      case 'v':
        emitChar('\v');
        consume();
        return true;
      case '0':
        if (!isOctDigit(LA(1))) {
          emitCodeUnit(0);
          consume();
          return true;
        }
        return false;
      default:
        return false;
    }  // switch
  };

  [[nodiscard]] auto parseConditionalEscapeSequence() -> bool {
    if (LA()) {
      emitChar(LA());
      consume();
      return true;
    }
    return false;
  };

  [[nodiscard]] auto parseEscapeSequence() -> bool {
    const auto ch = LA();
    if (ch != '\\') return false;
    consume();
    return parseNumericEscapeSequence() || parseUniversalCharacterName() ||
           parseSimpleEscapeSequence() || parseConditionalEscapeSequence();
  };

  void parseStringLiteral() {
    bool isRaw = false;

    while (auto ch = LA()) {
      if (ch == '"') break;
      if (ch == 'R') {
        isRaw = true;
        consume();
        continue;
      }
      consume();
    }

    if (LA() != '"') return;
    consume();

    if (isRaw) {
      std::string delimiter;
      while (auto ch = LA()) {
        if (ch == '(') {
          consume();
          break;
        }
        delimiter += ch;
        consume();
      }

      while (LA()) {
        if (LA() == ')') {
          bool matched = true;
          for (std::size_t i = 0; i < delimiter.size(); ++i) {
            if (LA(1 + static_cast<int>(i)) != delimiter[i]) {
              matched = false;
              break;
            }
          }
          if (matched && LA(1 + static_cast<int>(delimiter.size())) == '"') {
            break;
          }
        }
        emitChar(LA());
        consume();
      }
    } else {
      while (const auto ch = LA()) {
        if (ch == '"') break;

        if (parseEscapeSequence()) continue;

        emitChar(ch);
        consume();
      }
    }
  }

  void parseCharLiteral() {
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
                                      DiagnosticsClient* diagnostics)
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
                                    DiagnosticsClient* diagnostics)
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
                                     StringLiteralEncoding encoding,
                                     DiagnosticsClient* diagnostics)
    -> Components {
  StringLiteralParser<std::string> parser(text, diagnostics, encoding);

  parser.parseStringLiteral();

  Components components;
  components.value = std::move(parser.value);
  components.encoding = encoding;

  auto quotePos = text.find('"');
  if (quotePos != std::string_view::npos && quotePos > 0 &&
      text[quotePos - 1] == 'R') {
    components.isRaw = true;
  }

  return components;
}

void StringLiteral::initialize(StringLiteralEncoding encoding) const {
  components_ = Components::from(value(), encoding);
}

auto StringLiteral::charCount() const -> std::size_t {
  switch (components_.encoding) {
    case StringLiteralEncoding::kNone:
    case StringLiteralEncoding::kUtf8:
      return components_.value.size();
    case StringLiteralEncoding::kUtf16:
      return components_.value.size() / 2;
    case StringLiteralEncoding::kUtf32:
    case StringLiteralEncoding::kWide:
      return components_.value.size() / 4;
  }
  return components_.value.size();
}

auto CharLiteral::Components::from(std::string_view text,
                                   DiagnosticsClient* diagnostics)
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
