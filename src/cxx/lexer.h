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

#pragma once

#include <cxx/token.h>

#include <string_view>

namespace cxx {

class Lexer {
  std::string_view text_;
  int pos_ = 0;
  int end_ = 0;
  bool leadingSpace_ = false;
  bool startOfLine_ = true;

  TokenKind tokenKind_ = TokenKind::T_EOF_SYMBOL;
  bool tokenLeadingSpace_ = false;
  bool tokenStartOfLine_ = true;
  int tokenPos_ = 0;

  bool preprocessing_ = false;

 public:
  Lexer(const std::string_view& text);

  bool preprocessing() const { return preprocessing_; }
  void setPreprocessing(bool preprocessing) { preprocessing_ = preprocessing; }

  TokenKind operator()() { return next(); }

  TokenKind next() {
    tokenKind_ = readToken();
    return tokenKind_;
  }

  TokenKind tokenKind() const { return tokenKind_; }

  bool tokenLeadingSpace() const { return tokenLeadingSpace_; }

  bool tokenStartOfLine() const { return tokenStartOfLine_; }

  int tokenPos() const { return tokenPos_; }

  int tokenLength() const { return pos_ - tokenPos_; }

  std::string_view tokenText() const {
    return text_.substr(tokenPos_, tokenLength());
  }

  struct State {
    int pos_ = 0;
    bool leadingSpace_ = false;
    bool startOfLine_ = true;
  };

  State save() {
    State state;
    state.leadingSpace_ = leadingSpace_;
    state.startOfLine_ = startOfLine_;
    state.pos_ = pos_;
    return state;
  }

  void restore(const State& state) {
    leadingSpace_ = state.leadingSpace_;
    startOfLine_ = state.startOfLine_;
    pos_ = state.pos_;
  }

 private:
  TokenKind readToken();

 private:
  bool skipSpaces();
};

}  // namespace cxx
