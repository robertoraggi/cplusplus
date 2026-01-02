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

#include <cxx/lexer.h>
#include <cxx/private/c_keywords-priv.h>
#include <cxx/private/keywords-priv.h>
#include <utf8/unchecked.h>

#include <cctype>
#include <unordered_map>
#include <vector>

namespace cxx {

inline namespace {

enum class EncodingPrefix { kNone, kWide, kUtf8, kUtf16, kUtf32 };

const std::unordered_map<std::string_view, std::tuple<EncodingPrefix, bool>>
    kStringLiteralPrefixes{
        {"R", {EncodingPrefix::kNone, true}},
        {"L", {EncodingPrefix::kWide, false}},
        {"u8", {EncodingPrefix::kUtf8, false}},
        {"u", {EncodingPrefix::kUtf16, false}},
        {"U", {EncodingPrefix::kUtf32, false}},
        {"LR", {EncodingPrefix::kWide, true}},
        {"u8R", {EncodingPrefix::kUtf8, true}},
        {"uR", {EncodingPrefix::kUtf16, true}},
        {"UR", {EncodingPrefix::kUtf32, true}},
    };

const std::unordered_map<std::string_view, EncodingPrefix>
    kCharacterLiteralPrefixes{
        {"L", EncodingPrefix::kWide},
        {"u8", EncodingPrefix::kUtf8},
        {"u", EncodingPrefix::kUtf16},
        {"U", EncodingPrefix::kUtf32},
    };

inline auto is_idcont(int ch) -> bool {
  return ch == '_' || std::isalnum(static_cast<unsigned char>(ch));
}

template <typename It>
inline auto skipSlash(It it, It end) -> It {
  while (it < end && *it == '\\') {
    if (it + 1 < end && it[1] == '\n') {
      it += 2;
    } else if (it + 2 < end && it[1] == '\r' && it[2] == '\n') {
      it += 3;
    } else {
      break;
    }
  }
  return it;
}

template <typename It>
inline auto peekNext(It it, It end) -> std::uint32_t {
  it = skipSlash(it, end);
  return it < end ? utf8::unchecked::peek_next(it) : 0;
}

template <typename It>
inline void readNext(It& it, It end) {
  it = skipSlash(it, end);
  if (it < end) utf8::unchecked::next(it);
}

template <typename It>
inline void advance(It& it, int n, It end) {
  if (n > 0) {
    while (it < end && n--) readNext(it, end);
  } else if (n < 0) {
    while (n++) utf8::unchecked::prior(it);
  }
}

template <typename It>
inline auto skipBOM(It& it, It end) -> bool {
  if (it < end && *it == '\xEF') {
    if (it + 1 < end && it[1] == '\xBB') {
      if (it + 2 < end && it[2] == '\xBF') {
        it += 3;
        return true;
      }
    }
  }
  return false;
}
}  // namespace

Lexer::Lexer(std::string_view source, LanguageKind lang)
    : source_(source), pos_(cbegin(source_)), end_(cend(source_)), lang_(lang) {
  hasBOM_ = skipBOM(pos_, end_);
  currentChar_ = pos_ < end_ ? peekNext(pos_, end_) : 0;
}

Lexer::Lexer(std::string buffer, LanguageKind lang)
    : buffer_(std::move(buffer)),
      source_(buffer_),
      pos_(cbegin(source_)),
      end_(cend(source_)),
      lang_(lang) {
  hasBOM_ = skipBOM(pos_, end_);
  currentChar_ = pos_ < end_ ? peekNext(pos_, end_) : 0;
}

void Lexer::consume() {
  readNext(pos_, end_);
  currentChar_ = pos_ < end_ ? peekNext(pos_, end_) : 0;
}

void Lexer::consume(int n) {
  advance(pos_, n, end_);
  currentChar_ = pos_ < end_ ? peekNext(pos_, end_) : 0;
}

auto Lexer::LA(int n) const -> std::uint32_t {
  auto it = pos_;
  advance(it, n, n >= 0 ? end_ : source_.begin());
  return it < end_ ? peekNext(it, end_) : 0;
}

auto Lexer::readToken() -> TokenKind {
  const auto hasMoreChars = skipSpaces();

  tokenIsClean_ = true;
  tokenPos_ = int(pos_ - cbegin(source_));
  text_.clear();

  if (!hasMoreChars) return TokenKind::T_EOF_SYMBOL;

  const auto ch = LA();

  if (std::isdigit(ch)) {
    bool integer_literal = true;
    while (pos_ != end_) {
      const auto ch = LA();
      if (pos_ + 1 < end_ &&
          (ch == 'e' || ch == 'E' || ch == 'p' || ch == 'P') &&
          (LA(1) == '+' || LA(1) == '-')) {
        consume(2);
        integer_literal = false;
      } else if (pos_ + 1 < end_ && (ch == 'e' || ch == 'E') &&
                 std::isdigit(LA(1))) {
        consume(1);
        integer_literal = false;
      } else if (pos_ + 1 < end_ && ch == '\'' && is_idcont(LA(1))) {
        consume();
      } else if (is_idcont(ch)) {
        consume();
      } else if (ch == '.') {
        consume();
        integer_literal = false;
      } else {
        break;
      }
    }
    return integer_literal ? TokenKind::T_INTEGER_LITERAL
                           : TokenKind::T_FLOATING_POINT_LITERAL;
  }

  EncodingPrefix encodingPrefix = EncodingPrefix::kNone;

  bool isRawStringLiteral = false;

  if (std::isalpha(ch) || ch == '_') {
    do {
      text_ += static_cast<char>(LA());
      consume();
    } while (pos_ != end_ && is_idcont(LA()));

    bool isStringOrCharacterLiteral = false;

    if (pos_ != end_ && LA() == '"') {
      auto it = kStringLiteralPrefixes.find(text_);
      if (it != kStringLiteralPrefixes.end()) {
        auto [enc, raw] = it->second;
        encodingPrefix = enc;
        isRawStringLiteral = raw;
        isStringOrCharacterLiteral = true;
      }
    } else if (pos_ != end_ && LA() == '\'') {
      auto it = kCharacterLiteralPrefixes.find(text_);
      if (it != kCharacterLiteralPrefixes.end()) {
        encodingPrefix = it->second;
        isStringOrCharacterLiteral = true;
      }
    }

    if (!isStringOrCharacterLiteral) {
      tokenIsClean_ = text_.length() == tokenLength();

      if (preprocessing_) return TokenKind::T_IDENTIFIER;

      return classifyKeyword(text_, lang_);
    }
  }

  if (LA() == '"') {
    consume();

    if (isRawStringLiteral) {
      std::vector<std::uint32_t> delimiter;
      delimiter.reserve(8);

      for (; pos_ != end_; consume()) {
        const auto ch = LA();
        if (ch == '(' || ch == '"' || ch == '\\' || ch == '\n') break;
        delimiter.push_back(ch);
      }

      auto lookat_delimiter = [&]() -> bool {
        if (LA() != ')') return false;
        if (LA(int(delimiter.size() + 1)) != '"') return false;
        for (std::size_t i = 0; i < delimiter.size(); ++i) {
          if (LA(int(i + 1)) != delimiter[i]) return false;
        }
        return true;
      };

      while (pos_ != end_) {
        if (lookat_delimiter()) {
          consume(int(delimiter.size() + 2));
          break;
        }
        consume();
      }
    } else {
      while (pos_ != end_) {
        if (LA() == '"') {
          consume();
          break;
        } else if (pos_ + 1 < end_ && LA() == '\\') {
          consume(2);
        } else {
          consume();
        }
      }
    }

    if (lang_ == LanguageKind::kCXX) {
      bool ud = false;
      if (std::isalpha(LA()) || LA() == '_') {
        ud = true;
        do {
          consume();
        } while (pos_ != end_ && is_idcont(LA()));
      }

      if (ud) return TokenKind::T_USER_DEFINED_STRING_LITERAL;
    }

    switch (encodingPrefix) {
      case EncodingPrefix::kWide:
        return TokenKind::T_WIDE_STRING_LITERAL;
      case EncodingPrefix::kUtf8:
        return TokenKind::T_UTF8_STRING_LITERAL;
      case EncodingPrefix::kUtf16:
        return TokenKind::T_UTF16_STRING_LITERAL;
      case EncodingPrefix::kUtf32:
        return TokenKind::T_UTF32_STRING_LITERAL;
        // case TokenKind::kNone:
      default:
        return TokenKind::T_STRING_LITERAL;
    }  // switch
  }

  if (LA() == '\'') {
    consume();
    while (pos_ != end_ && LA() != '\'' && LA() != '\n') {
      if (pos_ + 1 < end_ && LA() == '\\') {
        consume(2);
      } else {
        consume();
      }
    }
    if (LA() == '\'') {
      consume();
    }

    return TokenKind::T_CHARACTER_LITERAL;
  }

  if (pos_ + 1 < end_ && LA() == '.' && std::isdigit(LA(1))) {
    consume();
    while (pos_ != end_) {
      const auto ch = LA();
      if (pos_ + 1 < end_ &&
          (ch == 'e' || ch == 'E' || ch == 'p' || ch == 'P') &&
          (LA(1) == '+' || LA(1) == '-')) {
        consume(2);
      } else if (pos_ + 1 < end_ && ch == '\'' && is_idcont(LA(1))) {
        consume();
      } else if (is_idcont(ch)) {
        consume();
      } else {
        break;
      }
    }
    return TokenKind::T_FLOATING_POINT_LITERAL;
  }

  consume();

  switch (ch) {
    case '#':
      if (pos_ != end_ && LA() == '#') {
        consume();
        return TokenKind::T_HASH_HASH;
      }
      return TokenKind::T_HASH;

    case '=':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_EQUAL_EQUAL;
      }
      return TokenKind::T_EQUAL;

    case ',':
      return TokenKind::T_COMMA;

    case '~':
      return TokenKind::T_TILDE;

    case '{':
      return TokenKind::T_LBRACE;

    case '}':
      return TokenKind::T_RBRACE;

    case '[':
      return TokenKind::T_LBRACKET;

    case ']':
      return TokenKind::T_RBRACKET;

    case '(':
      return TokenKind::T_LPAREN;

    case ')':
      return TokenKind::T_RPAREN;

    case ';':
      return TokenKind::T_SEMICOLON;

    case ':':
      if (pos_ != end_ && LA() == ':') {
        consume();
        return TokenKind::T_COLON_COLON;
      }
      return TokenKind::T_COLON;

    case '.':
      if (pos_ + 1 < end_ && LA() == '.' && LA(1) == '.') {
        consume(2);
        return TokenKind::T_DOT_DOT_DOT;
      } else if (pos_ != end_ && LA() == '*') {
        consume();
        return TokenKind::T_DOT_STAR;
      }
      return TokenKind::T_DOT;

    case '?':
      return TokenKind::T_QUESTION;

    case '*':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_STAR_EQUAL;
      }
      return TokenKind::T_STAR;

    case '%':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_PERCENT_EQUAL;
      }
      return TokenKind::T_PERCENT;

    case '^':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_CARET_EQUAL;
      } else if (pos_ != end_ && LA() == '^') {
        consume();
        return TokenKind::T_CARET_CARET;
      }
      return TokenKind::T_CARET;

    case '&':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_AMP_EQUAL;
      } else if (pos_ != end_ && LA() == '&') {
        consume();
        return TokenKind::T_AMP_AMP;
      }
      return TokenKind::T_AMP;

    case '|':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_BAR_EQUAL;
      } else if (pos_ != end_ && LA() == '|') {
        consume();
        return TokenKind::T_BAR_BAR;
      }
      return TokenKind::T_BAR;

    case '!':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_EXCLAIM_EQUAL;
      }
      return TokenKind::T_EXCLAIM;

    case '+':
      if (pos_ != end_ && LA() == '+') {
        consume();
        return TokenKind::T_PLUS_PLUS;
      } else if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_PLUS_EQUAL;
      }
      return TokenKind::T_PLUS;

    case '-':
      if (pos_ != end_ && LA() == '-') {
        consume();
        return TokenKind::T_MINUS_MINUS;
      } else if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_MINUS_EQUAL;
      } else if (pos_ != end_ && LA() == '>') {
        consume();
        if (pos_ != end_ && LA() == '*') {
          consume();
          return TokenKind::T_MINUS_GREATER_STAR;
        }
        return TokenKind::T_MINUS_GREATER;
      }
      return TokenKind::T_MINUS;

    case '<':
      if (pos_ != end_ && LA() == '=') {
        consume();
        if (pos_ != end_ && LA() == '>') {
          consume();
          return TokenKind::T_LESS_EQUAL_GREATER;
        }
        return TokenKind::T_LESS_EQUAL;
      } else if (pos_ != end_ && LA() == '<') {
        consume();
        if (pos_ != end_ && LA() == '=') {
          consume();
          return TokenKind::T_LESS_LESS_EQUAL;
        }
        return TokenKind::T_LESS_LESS;
      }
      return TokenKind::T_LESS;

    case '>':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_GREATER_EQUAL;
      } else if (pos_ != end_ && LA() == '>') {
        if (LA(1) == '=') {
          consume(2);
          return TokenKind::T_GREATER_GREATER_EQUAL;
        } else if (preprocessing_) {
          consume();
          return TokenKind::T_GREATER_GREATER;
        }

        tokenValue_.tokenKindValue = TokenKind::T_GREATER_GREATER;
      }
      return TokenKind::T_GREATER;

    case '/': {
      if (keepComments_ && LA() == '/') {
        consume();
        for (; pos_ != end_; consume()) {
          if (pos_ != end_ && LA() == '\n') {
            break;
          }
        }
        return TokenKind::T_COMMENT;
      }
      if (keepComments_ && LA() == '*') {
        consume();
        while (pos_ != end_) {
          if (pos_ + 1 < end_ && LA() == '*' && LA(1) == '/') {
            consume(2);
            break;
          }
          consume();
        }
        leadingSpace_ = tokenLeadingSpace_;
        startOfLine_ = tokenStartOfLine_;
        return TokenKind::T_COMMENT;
      }
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_SLASH_EQUAL;
      }
      return TokenKind::T_SLASH;
    }
  }

  return TokenKind::T_ERROR;
}  // namespace cxx

auto Lexer::skipSpaces() -> bool {
  tokenLeadingSpace_ = leadingSpace_;
  tokenStartOfLine_ = startOfLine_;

  while (pos_ != end_) {
    const auto ch = LA();

    if (std::isspace(ch)) {
      if (ch == '\n') {
        tokenStartOfLine_ = true;
        tokenLeadingSpace_ = false;
      } else {
        tokenLeadingSpace_ = true;
      }
      consume();
    } else if (!keepComments_ && pos_ + 1 < end_ && ch == '/' && LA(1) == '/') {
      consume(2);
      for (; pos_ != end_; consume()) {
        if (pos_ != end_ && LA() == '\n') {
          break;
        }
      }
    } else if (!keepComments_ && pos_ + 1 < end_ && ch == '/' && LA(1) == '*') {
      consume(2);
      while (pos_ != end_) {
        if (pos_ + 1 < end_ && LA() == '*' && LA(1) == '/') {
          consume(2);
          break;
        }
        consume();
      }
      // unexpected eof
    } else {
      break;
    }
  }

  leadingSpace_ = false;
  startOfLine_ = false;

  return pos_ != end_;
}

auto Lexer::classifyKeyword(const std::string_view& text, LanguageKind lang)
    -> TokenKind {
  if (lang == LanguageKind::kCXX) {
    return classify(text.data(), static_cast<int>(text.size()));
  }
  return classifyC(text.data(), static_cast<int>(text.size()));
}

void Lexer::clearBuffer() { buffer_.clear(); }

}  // namespace cxx
