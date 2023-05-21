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

#include <cxx/token.h>

// cxx
#include <cxx/literals.h>
#include <cxx/names.h>

namespace cxx {

namespace {

std::string token_spell[] = {
#define TOKEN_SPELL(_, s) s,
    FOR_EACH_TOKEN(TOKEN_SPELL)};
#undef TOKEN_SPELL

std::string token_name[] = {
#define TOKEN_SPELL(s, _) #s,
    FOR_EACH_TOKEN(TOKEN_SPELL)};
#undef TOKEN_SPELL
}  // namespace

auto Token::spell(TokenKind kind) -> const std::string& {
  return token_spell[static_cast<int>(kind)];
}

auto Token::spell() const -> const std::string& {
  switch (kind()) {
    case TokenKind::T_IDENTIFIER:
      return value_.idValue ? value_.idValue->name() : spell(kind());

    case TokenKind::T_USER_DEFINED_STRING_LITERAL:
    case TokenKind::T_STRING_LITERAL:
    case TokenKind::T_WIDE_STRING_LITERAL:
    case TokenKind::T_UTF8_STRING_LITERAL:
    case TokenKind::T_UTF16_STRING_LITERAL:
    case TokenKind::T_UTF32_STRING_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_INTEGER_LITERAL:
    case TokenKind::T_COMMENT:
      return value_.literalValue ? value_.literalValue->value() : spell(kind());

    default:
      return spell(kind());
  }  // switch
}

auto Token::name(TokenKind kind) -> const std::string& {
  return token_name[static_cast<int>(kind)];
}

auto Token::name() const -> const std::string& { return name(kind()); }

}  // namespace cxx
