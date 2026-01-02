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

import { cpy_header } from "./cpy_header.ts";
import * as fs from "fs";
import * as tokens from "./tokens.ts";

export function gen_tokenkind_ts({ output }: { output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  emit("export enum TokenKind {");
  tokens.BASE_TOKENS.forEach((tk) => emit(`  ${tk},`));
  tokens.OPERATORS.forEach(([tk]) => emit(`  ${tk},`));
  tokens.C_AND_CXX_KEYWORDS.forEach((tk) => emit(`  ${tk.toUpperCase()},`));
  emit("}");

  const out = `// Generated file by: gen_tokenkind_ts.ts
${cpy_header}
${code.join("\n")}
`;

  fs.writeFileSync(output, out);
}
