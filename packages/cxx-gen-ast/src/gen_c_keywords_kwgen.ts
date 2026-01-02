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

import * as tokens from "./tokens.ts";
import kwgen from "./kwgen.ts";
import { cpy_header } from "./cpy_header.ts";

export function gen_c_keywords_kwgen({ output }: { output: string }) {
  const keywords: string[] = [];

  keywords.push(...tokens.C_KEYWORDS);

  Object.entries(tokens.C_TOKEN_ALIASES).forEach(([tk]) => {
    keywords.push(tk);
  });

  kwgen({
    copyright: cpy_header,
    output,
    keywords,
    tokenPrefix: "cxx::TokenKind::T_",
    tokenType: "cxx::TokenKind",
    toUpper: true,
    noEnums: true,
    defaultToken: "cxx::TokenKind::T_IDENTIFIER",
    classifier: "classifyC",
  });
}
