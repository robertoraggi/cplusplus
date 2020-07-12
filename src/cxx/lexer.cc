// Copyright (c) 2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "lexer.h"

#include <fmt/format.h>

#include <cctype>
#include <unordered_map>

#include "keywords-priv.h"

namespace cxx {

inline namespace {

enum struct EncodingPrefix { kNone, kWide, kUtf8, kUtf16, kUtf32 };

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

}  // namespace

inline bool is_space(int ch) { return std::isspace((unsigned char)ch); }
inline bool is_digit(int ch) { return std::isdigit((unsigned char)ch); }
inline bool is_alpha(int ch) { return std::isalpha((unsigned char)ch); }
inline bool is_alnum(int ch) { return std::isalnum((unsigned char)ch); }

inline bool is_idcont(int ch) {
  return ch == '_' || std::isalnum((unsigned char)ch);
}

Lexer::Lexer(const std::string_view& text)
    : text_(text), pos_(0), end_(int(text.size())) {}

TokenKind Lexer::readToken() {
  const auto hasMoreChars = skipSpaces();

  tokenPos_ = pos_;

  if (!hasMoreChars) return TokenKind::T_EOF_SYMBOL;

  const char ch = text_[pos_];

  if (is_digit(ch)) {
    bool integer_literal = true;
    while (pos_ < end_) {
      const auto ch = text_[pos_];
      if (pos_ + 1 < end_ &&
          (ch == 'e' || ch == 'E' || ch == 'p' || ch == 'P') &&
          (text_[pos_ + 1] == '+' || text_[pos_ + 1] == '-')) {
        pos_ += 2;
        integer_literal = false;
      } else if (pos_ + 1 < end_ && ch == '\'' && is_idcont(text_[pos_ + 1])) {
        ++pos_;
      } else if (is_idcont(ch)) {
        ++pos_;
      } else if (ch == '.') {
        ++pos_;
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

  if (is_alpha(ch) || ch == '_') {
    do {
      ++pos_;
    } while (pos_ < end_ && is_idcont(text_[pos_]));

    const auto id = text_.substr(tokenPos_, pos_ - tokenPos_);

    bool isStringOrCharacterLiteral = false;

    if (pos_ < end_ && text_[pos_] == '"') {
      auto it = kStringLiteralPrefixes.find(id);
      if (it != kStringLiteralPrefixes.end()) {
        auto [enc, raw] = it->second;
        encodingPrefix = enc;
        isRawStringLiteral = raw;
        isStringOrCharacterLiteral = true;
      }
    } else if (pos_ < end_ && text_[pos_] == '\'') {
      auto it = kCharacterLiteralPrefixes.find(id);
      if (it != kCharacterLiteralPrefixes.end()) {
        encodingPrefix = it->second;
        isStringOrCharacterLiteral = true;
      }
    }

    if (!isStringOrCharacterLiteral)
      return (TokenKind)classify(id.data(), int(id.size()));
  }

  if (text_[pos_] == '"') {
    ++pos_;

    std::string_view delimiter;
    const auto startDelimiter = pos_;
    int endDelimiter = pos_;

    if (isRawStringLiteral) {
      for (; pos_ < end_; ++pos_) {
        const auto ch = text_[pos_];
        if (ch == '(' || ch == '"' || ch == '\\' || ch == '\n') break;
      }
      endDelimiter = pos_;
      delimiter = text_.substr(startDelimiter, endDelimiter - startDelimiter);
    }

    while (pos_ < end_) {
      if (text_[pos_] == '"') {
        ++pos_;

        if (!isRawStringLiteral) break;

        const auto N = endDelimiter - startDelimiter;

        if (text_[pos_ - 2 - N] == ')') {
          bool didMatch = true;
          for (int i = 0; i < N; ++i) {
            if (text_[startDelimiter + i] != text_[pos_ - 1 - N + i]) {
              didMatch = false;
              break;
            }
          }
          if (didMatch) break;
        }
      } else if (pos_ + 1 < end_ && text_[pos_] == '\\') {
        pos_ += 2;
      } else {
        ++pos_;
      }
    }
    if (is_alpha(text_[pos_]) || text_[pos_] == '_') {
      do {
        ++pos_;
      } while (pos_ < end_ && is_idcont(text_[pos_]));
    }
    return TokenKind::T_STRING_LITERAL;
  }

  if (text_[pos_] == '\'') {
    ++pos_;
    while (pos_ < end_ && text_[pos_] != '\'') {
      if (pos_ + 1 < end_ && text_[pos_] == '\\') {
        pos_ += 2;
      } else {
        ++pos_;
      }
    }
    if (text_[pos_] == '\'') {
      ++pos_;
    }

    return TokenKind::T_CHARACTER_LITERAL;
  }

  if (pos_ + 1 < end_ && text_[pos_] == '.' && is_digit(text_[pos_ + 1])) {
    ++pos_;
    while (pos_ < end_) {
      const auto ch = text_[pos_];
      if (pos_ + 1 < end_ &&
          (ch == 'e' || ch == 'E' || ch == 'p' || ch == 'P') &&
          (text_[pos_ + 1] == '+' || text_[pos_ + 1] == '-')) {
        pos_ += 2;
      } else if (pos_ + 1 < end_ && ch == '\'' && is_idcont(text_[pos_ + 1])) {
        ++pos_;
      } else if (is_idcont(ch)) {
        ++pos_;
      } else {
        break;
      }
    }
    return TokenKind::T_FLOATING_POINT_LITERAL;
  }

  ++pos_;

  switch (ch) {
    case '=':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
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
      if (pos_ < end_ && text_[pos_] == ':') {
        ++pos_;
        return TokenKind::T_COLON_COLON;
      }
      return TokenKind::T_COLON;

    case '.':
      if (pos_ + 1 < end_ && text_[pos_] == '.' && text_[pos_ + 1] == '.') {
        pos_ += 2;
        return TokenKind::T_DOT_DOT_DOT;
      } else if (pos_ < end_ && text_[pos_] == '*') {
        ++pos_;
        return TokenKind::T_DOT_STAR;
      }
      return TokenKind::T_DOT;

    case '?':
      return TokenKind::T_QUESTION;

    case '*':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_STAR_EQUAL;
      }
      return TokenKind::T_STAR;

    case '%':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_PERCENT_EQUAL;
      }
      return TokenKind::T_PERCENT;

    case '^':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_CARET_EQUAL;
      }
      return TokenKind::T_CARET;

    case '&':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_AMP_EQUAL;
      } else if (pos_ < end_ && text_[pos_] == '&') {
        ++pos_;
        return TokenKind::T_AMP_AMP;
      }
      return TokenKind::T_AMP;

    case '|':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_BAR_EQUAL;
      } else if (pos_ < end_ && text_[pos_] == '|') {
        ++pos_;
        return TokenKind::T_BAR_BAR;
      }
      return TokenKind::T_BAR;

    case '!':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_EXCLAIM_EQUAL;
      }
      return TokenKind::T_EXCLAIM;

    case '+':
      if (pos_ < end_ && text_[pos_] == '+') {
        ++pos_;
        return TokenKind::T_PLUS_PLUS;
      } else if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_PLUS_EQUAL;
      }
      return TokenKind::T_PLUS;

    case '-':
      if (pos_ < end_ && text_[pos_] == '-') {
        ++pos_;
        return TokenKind::T_MINUS_MINUS;
      } else if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_MINUS_EQUAL;
      } else if (pos_ < end_ && text_[pos_] == '>') {
        ++pos_;
        if (pos_ < end_ && text_[pos_] == '*') {
          ++pos_;
          return TokenKind::T_MINUS_GREATER_STAR;
        } else {
          return TokenKind::T_MINUS_GREATER;
        }
      }
      return TokenKind::T_MINUS;

    case '<':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        if (pos_ < end_ && text_[pos_] == '>') {
          ++pos_;
          return TokenKind::T_LESS_EQUAL_GREATER;
        }
        return TokenKind::T_LESS_EQUAL;
      } else if (pos_ < end_ && text_[pos_] == '<') {
        ++pos_;
        if (pos_ < end_ && text_[pos_] == '=') {
          ++pos_;
          return TokenKind::T_LESS_LESS_EQUAL;
        }
        return TokenKind::T_LESS_LESS;
      }
      return TokenKind::T_LESS;

    case '>':
      if (preprocessing_) {
        if (pos_ < end_ && text_[pos_] == '=') {
          ++pos_;
          return TokenKind::T_GREATER_EQUAL;
        } else if (pos_ < end_ && text_[pos_] == '>') {
          ++pos_;
          if (pos_ < end_ && text_[pos_] == '=') {
            ++pos_;
            return TokenKind::T_GREATER_GREATER_EQUAL;
          }
          return TokenKind::T_GREATER_GREATER;
        }
      }
      return TokenKind::T_GREATER;

    case '/':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_SLASH_EQUAL;
      }
      return TokenKind::T_SLASH;
  }

  return TokenKind::T_ERROR;
}  // namespace cxx

bool Lexer::skipSpaces() {
  tokenLeadingSpace_ = leadingSpace_;
  tokenStartOfLine_ = startOfLine_;

  while (pos_ < end_) {
    const auto ch = text_[pos_];

    if (is_space(ch)) {
      if (ch == '\n') {
        tokenStartOfLine_ = true;
        tokenLeadingSpace_ = false;
      } else {
        tokenLeadingSpace_ = true;
      }
      ++pos_;
    } else if (pos_ + 1 < end_ && ch == '/' && text_[pos_ + 1] == '/') {
      pos_ += 2;
      for (; pos_ < end_; ++pos_) {
        if (pos_ < end_ && text_[pos_] == '\n') {
          break;
        }
      }
    } else if (pos_ + 1 < end_ && ch == '/' && text_[pos_ + 1] == '*') {
      while (pos_ < end_) {
        if (pos_ + 1 < end_ && text_[pos_] == '*' && text_[pos_ + 1] == '/') {
          pos_ += 2;
          break;
        } else {
          ++pos_;
        }
      }
      // unexpected eof
    } else {
      break;
    }
  }

  leadingSpace_ = false;
  startOfLine_ = false;

  return pos_ < end_;
}

}  // namespace cxx
