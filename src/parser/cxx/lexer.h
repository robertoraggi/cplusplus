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

  [[nodiscard]] inline auto LA() const -> uint32_t { return currentChar_; }

  [[nodiscard]] auto LA(int n) const -> uint32_t;

 public:
  explicit Lexer(const std::string_view& source);

  [[nodiscard]] auto preprocessing() const -> bool { return preprocessing_; }
  void setPreprocessing(bool preprocessing) { preprocessing_ = preprocessing; }

  [[nodiscard]] auto keepComments() const -> bool { return keepComments_; }
  void setKeepComments(bool keepComments) { keepComments_ = keepComments; }

  auto operator()() -> TokenKind { return next(); }

  auto next() -> TokenKind {
    tokenKind_ = readToken();
    return tokenKind_;
  }

  [[nodiscard]] auto tokenKind() const -> TokenKind { return tokenKind_; }

  [[nodiscard]] auto tokenLeadingSpace() const -> bool {
    return tokenLeadingSpace_;
  }

  [[nodiscard]] auto tokenStartOfLine() const -> bool {
    return tokenStartOfLine_;
  }

  [[nodiscard]] auto tokenPos() const -> int { return tokenPos_; }

  [[nodiscard]] auto tokenLength() const -> uint32_t {
    return (pos_ - cbegin(source_)) - tokenPos_;
  }

  [[nodiscard]] auto tokenIsClean() const -> bool { return tokenIsClean_; }

  [[nodiscard]] auto tokenText() const -> std::string_view {
    if (tokenIsClean_) return source_.substr(tokenPos_, tokenLength());
    return text_;
  }

  [[nodiscard]] auto tokenValue() const -> TokenValue { return tokenValue_; }

  struct State {
    std::string_view::const_iterator pos_;
    uint32_t currentChar_ = 0;
    bool leadingSpace_ = false;
    bool startOfLine_ = true;
  };

  auto save() -> State {
    return {pos_, currentChar_, leadingSpace_, startOfLine_};
  }

  void restore(const State& state) {
    pos_ = state.pos_;
    currentChar_ = state.currentChar_;
    leadingSpace_ = state.leadingSpace_;
    startOfLine_ = state.startOfLine_;
  }

  auto text() -> std::string& { return text_; }
  [[nodiscard]] auto text() const -> const std::string& { return text_; }

  static auto classifyKeyword(const std::string_view& text) -> TokenKind;

 private:
  auto readToken() -> TokenKind;

 private:
  auto skipSpaces() -> bool;
};

}  // namespace cxx
