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

#include "keywords-priv.h"

namespace cxx {

inline bool is_space(int ch) { return std::isspace((unsigned char)ch); }
inline bool is_digit(int ch) { return std::isdigit((unsigned char)ch); }
inline bool is_alpha(int ch) { return std::isalpha((unsigned char)ch); }
inline bool is_alnum(int ch) { return std::isalnum((unsigned char)ch); }

inline bool is_idcont(int ch) {
  return ch == '_' || std::isalnum((unsigned char)ch);
}

Lexer::Lexer(const std::string_view& text)
    : text_(text), pos_(0), end_(text.size()) {}

TokenKind Lexer::readToken() {
  const auto hasMoreChars = skipSpaces();

  tokenPos_ = pos_;

  if (!hasMoreChars) return TokenKind::T_EOF_SYMBOL;

  const char ch = text_[pos_];

  if (is_digit(ch)) {
    do {
      ++pos_;
    } while (pos_ < end_ && is_digit(text_[pos_]));
    return TokenKind::T_INTEGER_LITERAL;
  }

  if (is_alpha(ch) || ch == '_') {
    do {
      ++pos_;
    } while (pos_ < end_ && is_idcont(text_[pos_]));
    const auto id = text_.substr(tokenPos_, pos_ - tokenPos_);
    return (TokenKind)classify(id.data(), id.size());
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
#if 0
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
#endif
      return TokenKind::T_GREATER;

    case '/':
      if (pos_ < end_ && text_[pos_] == '=') {
        ++pos_;
        return TokenKind::T_SLASH_EQUAL;
      }
      return TokenKind::T_SLASH;
  }

  return TokenKind::T_ERROR;
}

bool Lexer::skipSpaces() {
  tokenLeadingSpace_ = leadingSpace_;
  tokenStartOfLine_ = startOfLine_;

  while (pos_ < end_) {
    const auto ch = text_[pos_];

    if (std::isspace(ch)) {
      if (ch == '\n') {
        tokenStartOfLine_ = true;
        tokenLeadingSpace_ = false;
      } else if (!tokenStartOfLine_) {
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
