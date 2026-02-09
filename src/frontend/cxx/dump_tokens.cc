// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "dump_tokens.h"

#include <cxx/lexer.h>
#include <cxx/preprocessor.h>

#include <format>
#include <iostream>

namespace cxx {

DumpTokens::DumpTokens(const CLI& cli) : cli(cli) {}

void DumpTokens::operator()(TranslationUnit& unit, std::ostream& output) {
  auto lang = LanguageKind::kCXX;

  if (auto x = cli.getSingle("x")) {
    if (x == "c") lang = LanguageKind::kC;
  } else if (unit.fileName().ends_with(".c")) {
    lang = LanguageKind::kC;
  }

  std::string flags;

  const auto builtinsFileId = unit.preprocessor()->builtinsFileId();

  for (SourceLocation loc(1);; loc = loc.next()) {
    const auto& tk = unit.tokenAt(loc);

    // skip tokens from the builtins prelude
    if (builtinsFileId && tk.fileId() == builtinsFileId) {
      if (tk.is(TokenKind::T_EOF_SYMBOL)) break;
      continue;
    }

    flags.clear();

    if (tk.startOfLine()) {
      flags += " [start-of-line]";
    }

    if (tk.leadingSpace()) {
      flags += " [leading-space]";
    }

    auto kind = tk.kind();
    if (kind == TokenKind::T_IDENTIFIER) {
      kind = Lexer::classifyKeyword(tk.spell(), lang);
    }

    output << std::format("{} '{}'{}", Token::name(kind), tk.spell(), flags);

    auto pos = unit.tokenStartPosition(loc);

    output << std::format(" at {}:{}:{}\n", pos.fileName, pos.line, pos.column);

    if (tk.is(TokenKind::T_EOF_SYMBOL)) break;
  }
}

}  // namespace cxx