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

namespace cxx {

const char* token_spell[] = {
#define TOKEN_SPELL(_, s) s,
    FOR_EACH_TOKEN(TOKEN_SPELL)};
#undef TOKEN_SPELL

const char* token_name[] = {
#define TOKEN_SPELL(s, _) #s,
    FOR_EACH_TOKEN(TOKEN_SPELL)};
#undef TOKEN_SPELL

std::string_view Token::spell(TokenKind kind) { return token_spell[(int)kind]; }

std::string_view Token::spell() const { return spell(kind_); }

std::string_view Token::name(TokenKind kind) { return token_name[(int)kind]; }

std::string_view Token::name() const { return name(kind_); }

}  // namespace cxx
