// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

import * as fs from "fs";
import * as tokens from "./tokens.js";

export function gen_keywords_kwgen({ output }: { output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const isContextKeyword = (kw: string) => {
    return ["final", "override", "import", "module"].includes(kw);
  };

  tokens.KEYWORDS.filter((kw) => !isContextKeyword(kw)).forEach((tk) =>
    emit(tk),
  );

  emit();
  tokens.TOKEN_ALIASES.forEach(([tk]) => emit(tk));

  const out = `%no-enums
%token-prefix=cxx::TokenKind::T_
%token-type=cxx::TokenKind
%toupper

%%
${code.join("\n")}
`;

  fs.writeFileSync(output, out);
}
