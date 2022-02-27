// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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
  const std::string_view source_;
  std::string_view::const_iterator pos_;
  std::string_view::const_iterator end_;
  std::string text_;
  bool leadingSpace_ = false;
  bool startOfLine_ = true;
  bool keepComments_ = false;

  TokenKind tokenKind_ = TokenKind::T_EOF_SYMBOL;
  TokenValue tokenValue_{};
  bool tokenLeadingSpace_ = false;
  bool tokenStartOfLine_ = true;
  bool tokenIsClean_ = true;
  int tokenPos_ = 0;
  uint32_t currentChar_ = 0;

  bool preprocessing_ = false;

  void consume();
  void consume(int n);

  inline uint32_t LA() const { return currentChar_; }

  uint32_t LA(int n) const;

 public:
  explicit Lexer(const std::string_view& source);

  bool preprocessing() const { return preprocessing_; }
  void setPreprocessing(bool preprocessing) { preprocessing_ = preprocessing; }

  bool keepComments() const { return keepComments_; }
  void setKeepComments(bool keepComments) { keepComments_ = keepComments; }

  TokenKind operator()() { return next(); }

  TokenKind next() {
    tokenKind_ = readToken();
    return tokenKind_;
  }

  TokenKind tokenKind() const { return tokenKind_; }

  bool tokenLeadingSpace() const { return tokenLeadingSpace_; }

  bool tokenStartOfLine() const { return tokenStartOfLine_; }

  int tokenPos() const { return tokenPos_; }

  uint32_t tokenLength() const { return (pos_ - cbegin(source_)) - tokenPos_; }

  bool tokenIsClean() const { return tokenIsClean_; }

  std::string_view tokenText() const {
    if (tokenIsClean_) return source_.substr(tokenPos_, tokenLength());
    return text_;
  }

  TokenValue tokenValue() const { return tokenValue_; }

  struct State {
    std::string_view::const_iterator pos_;
    uint32_t currentChar_ = 0;
    bool leadingSpace_ = false;
    bool startOfLine_ = true;
  };

  State save() { return {pos_, currentChar_, leadingSpace_, startOfLine_}; }

  void restore(const State& state) {
    pos_ = state.pos_;
    currentChar_ = state.currentChar_;
    leadingSpace_ = state.leadingSpace_;
    startOfLine_ = state.startOfLine_;
  }

  std::string& text() { return text_; }
  const std::string& text() const { return text_; }

  static TokenKind classifyKeyword(const std::string_view& text);

 private:
  TokenKind readToken();

 private:
  bool skipSpaces();
};

}  // namespace cxx
