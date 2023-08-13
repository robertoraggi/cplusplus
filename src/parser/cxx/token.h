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

#include <cxx/cxx_fwd.h>
#include <cxx/literals_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/token_fwd.h>

#include <cstdint>
#include <string>

namespace cxx {

union TokenValue {
  const void* ptrValue;
  std::string* stringValue;
  const Identifier* idValue;
  const Literal* literalValue;
  TokenKind tokenKindValue;
  int intValue;
};

class Token {
 public:
  Token(const Token&) = default;
  auto operator=(const Token&) -> Token& = default;

  Token(Token&&) = default;
  auto operator=(Token&&) -> Token& = default;

  Token() = default;

  explicit Token(TokenKind kind, unsigned offset = 0, unsigned length = 0,
                 TokenValue value = {});

  [[nodiscard]] inline auto kind() const -> TokenKind;
  inline void setKind(TokenKind kind);

  [[nodiscard]] inline auto value() const -> TokenValue;
  inline void setValue(TokenValue value);

  [[nodiscard]] inline auto offset() const -> unsigned;
  [[nodiscard]] inline auto length() const -> unsigned;

  [[nodiscard]] inline auto fileId() const -> unsigned;
  inline void setFileId(unsigned fileId);

  [[nodiscard]] inline auto startOfLine() const -> bool;
  inline void setStartOfLine(bool startOfLine);

  [[nodiscard]] inline auto leadingSpace() const -> bool;
  inline void setLeadingSpace(bool leadingSpace);

  explicit operator bool() const;
  explicit operator TokenKind() const;

  [[nodiscard]] auto is(TokenKind k) const -> bool;
  [[nodiscard]] auto isNot(TokenKind k) const -> bool;

  [[nodiscard]] auto isOneOf(auto... tokens) const -> bool {
    return (... || is(tokens));
  }

  [[nodiscard]] auto spell() const -> const std::string&;
  [[nodiscard]] auto name() const -> const std::string&;

  static auto spell(TokenKind kind) -> const std::string&;
  static auto name(TokenKind kind) -> const std::string&;

 private:
  TokenKind kind_ : 8;
  uint32_t startOfLine_ : 1;
  uint32_t leadingSpace_ : 1;
  uint32_t fileId_ : 10;
  uint32_t length_ : 16;
  uint32_t offset_ : 28;
  TokenValue value_;
};

inline Token::Token(TokenKind kind, unsigned offset, unsigned length,
                    TokenValue value)
    : kind_(kind),
      startOfLine_(0),
      leadingSpace_(0),
      fileId_(0),
      length_(length),
      offset_(offset),
      value_(value) {}

inline auto Token::kind() const -> TokenKind { return kind_; }

inline void Token::setKind(TokenKind kind) { kind_ = kind; }

inline auto Token::value() const -> TokenValue { return value_; }

inline void Token::setValue(TokenValue value) { value_ = value; }

inline auto Token::offset() const -> unsigned { return offset_; }

inline auto Token::length() const -> unsigned { return length_; }

inline auto Token::fileId() const -> unsigned { return fileId_; }

inline void Token::setFileId(unsigned fileId) { fileId_ = fileId; }

inline auto Token::startOfLine() const -> bool { return startOfLine_; }

inline void Token::setStartOfLine(bool startOfLine) {
  startOfLine_ = startOfLine;
}

inline auto Token::leadingSpace() const -> bool { return leadingSpace_; }

inline void Token::setLeadingSpace(bool leadingSpace) {
  leadingSpace_ = leadingSpace;
}

inline Token::operator bool() const {
  return kind() != TokenKind::T_EOF_SYMBOL;
}

inline Token::operator TokenKind() const { return kind(); }

inline auto Token::is(TokenKind k) const -> bool { return kind() == k; }

inline auto Token::isNot(TokenKind k) const -> bool { return kind() != k; }

}  // namespace cxx
