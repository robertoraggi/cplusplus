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

#include <cxx/lexer.h>
#include <fmt/format.h>
#include <utf8.h>

#include <cassert>
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

inline bool is_idcont(int ch) {
  return ch == '_' || std::isalnum((unsigned char)ch);
}

Lexer::Lexer(const std::string_view& text)
    : text_(text), pos_(cbegin(text_)), end_(cend(text_)) {
  currentChar_ = pos_ < end_ ? utf8::peek_next(pos_, end_) : 0;
}

void Lexer::consume() {
  utf8::next(pos_, end_);
  currentChar_ = pos_ < end_ ? utf8::peek_next(pos_, end_) : 0;
}

void Lexer::consume(int n) {
  utf8::advance(pos_, n, end_);
  currentChar_ = pos_ < end_ ? utf8::peek_next(pos_, end_) : 0;
}

uint32_t Lexer::LA() const { return currentChar_; }

uint32_t Lexer::LA(int n) const {
  auto it = pos_;
  utf8::advance(it, n, end_);
  return it < end_ ? utf8::peek_next(it, end_) : 0;
}

TokenKind Lexer::readToken() {
  const auto hasMoreChars = skipSpaces();

  tokenPos_ = pos_ - cbegin(text_);

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
      consume();
    } while (pos_ != end_ && is_idcont(LA()));

    const auto n = (pos_ - cbegin(text_)) - tokenPos_;
    const auto id = text_.substr(tokenPos_, n);

    bool isStringOrCharacterLiteral = false;

    if (pos_ != end_ && LA() == '"') {
      auto it = kStringLiteralPrefixes.find(id);
      if (it != kStringLiteralPrefixes.end()) {
        auto [enc, raw] = it->second;
        encodingPrefix = enc;
        isRawStringLiteral = raw;
        isStringOrCharacterLiteral = true;
      }
    } else if (pos_ != end_ && LA() == '\'') {
      auto it = kCharacterLiteralPrefixes.find(id);
      if (it != kCharacterLiteralPrefixes.end()) {
        encodingPrefix = it->second;
        isStringOrCharacterLiteral = true;
      }
    }

    if (!isStringOrCharacterLiteral) return classify(id.data(), int(id.size()));
  }

  if (LA() == '"') {
    consume();

    std::string_view delimiter;
    const auto startDelimiter = pos_;
    auto endDelimiter = pos_;

    if (isRawStringLiteral) {
      for (; pos_ != end_; consume()) {
        const auto ch = LA();
        if (ch == '(' || ch == '"' || ch == '\\' || ch == '\n') break;
      }
      endDelimiter = pos_;
      delimiter = text_.substr(startDelimiter - cbegin(text_),
                               endDelimiter - startDelimiter);
    }

    while (pos_ != end_) {
      if (LA() == '"') {
        consume();

        if (!isRawStringLiteral) break;

        const auto S = startDelimiter - pos_;
        const auto N = endDelimiter - startDelimiter;

        if (LA(N - 2) == ')') {
          bool didMatch = true;
          for (int i = 0; i < N; ++i) {
            if (LA(S + i) != LA(N - i - 1)) {
              didMatch = false;
              break;
            }
          }
          if (didMatch) break;
        }
      } else if (pos_ + 1 < end_ && LA() == '\\') {
        consume(2);
      } else {
        consume();
      }
    }
    bool ud = false;
    if (std::isalpha(LA()) || LA() == '_') {
      ud = true;
      do {
        consume();
      } while (pos_ != end_ && is_idcont(LA()));
    }
    return !ud ? TokenKind::T_STRING_LITERAL
               : TokenKind::T_USER_DEFINED_STRING_LITERAL;
  }

  if (LA() == '\'') {
    consume();
    while (pos_ != end_ && LA() != '\'') {
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
        } else {
          return TokenKind::T_MINUS_GREATER;
        }
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
      if (preprocessing_) {
        if (pos_ != end_ && LA() == '=') {
          consume();
          return TokenKind::T_GREATER_EQUAL;
        } else if (pos_ != end_ && LA() == '>') {
          consume();
          if (pos_ != end_ && LA() == '=') {
            consume();
            return TokenKind::T_GREATER_GREATER_EQUAL;
          }
          return TokenKind::T_GREATER_GREATER;
        }
      }
      return TokenKind::T_GREATER;

    case '/':
      if (pos_ != end_ && LA() == '=') {
        consume();
        return TokenKind::T_SLASH_EQUAL;
      }
      return TokenKind::T_SLASH;
  }

  return TokenKind::T_ERROR;
}  // namespace cxx

bool Lexer::skipSpaces() {
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
    } else if (pos_ + 1 < end_ && ch == '/' && LA(1) == '/') {
      consume(2);
      for (; pos_ != end_; consume()) {
        if (pos_ != end_ && LA() == '\n') {
          break;
        }
      }
    } else if (pos_ + 1 < end_ && ch == '/' && LA(1) == '*') {
      while (pos_ != end_) {
        if (pos_ + 1 < end_ && LA() == '*' && LA(1) == '/') {
          consume(2);
          break;
        } else {
          consume();
        }
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

}  // namespace cxx
