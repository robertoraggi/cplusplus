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

const std::string& Token::spell(TokenKind kind) {
  return token_spell[(int)kind];
}

const std::string& Token::spell() const {
  switch (kind_) {
    case TokenKind::T_IDENTIFIER:
      return value_.idValue ? value_.idValue->name() : spell(kind_);

    case TokenKind::T_STRING_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_INTEGER_LITERAL:
      return value_.literalValue ? value_.literalValue->value() : spell(kind_);

    default:
      return spell(kind_);
  }  // switch
}

const std::string& Token::name(TokenKind kind) { return token_name[(int)kind]; }

const std::string& Token::name() const { return name(kind_); }

}  // namespace cxx
