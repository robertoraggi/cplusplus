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

bool Lexer::next() {
  if (!skipSpaces()) return false;

  std::string_view flags = leadingSpace_
                               ? " [leading-space]"
                               : startOfLine_ ? " [start-of-newline]" : "";

  fmt::print("char '{}'{}\n", text_[pos_], flags);

  ++pos_;

  return true;
}

bool Lexer::skipSpaces() {
  leadingSpace_ = false;
  startOfLine_ = false;

  while (pos_ < end_) {
    const auto ch = text_[pos_];

    if (std::isspace(ch)) {
      if (ch == '\n') {
        startOfLine_ = true;
        leadingSpace_ = false;
      } else {
        leadingSpace_ = true;
      }
      ++pos_;
    } else if (pos_ + 1 < end_ && ch == '/' && text_[pos_ + 1] == '/') {
      pos_ += 2;
      for (; pos_ < end_; ++pos_) {
        if (text_[pos_] == '\n') {
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
  return pos_ < end_;
}

}  // namespace cxx
