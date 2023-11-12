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

import { cpy_header } from "./cpy_header.js";
import * as fs from "fs";
import * as tokens from "./tokens.js";

export function gen_token_fwd_h({ output }: { output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const baseTokenId = (s: string) => `<${s.toLowerCase()}>`;

  emit("#define FOR_EACH_BASE_TOKEN(V) \\");
  tokens.BASE_TOKENS.forEach((tk) =>
    emit(`  V(${tk}, "${baseTokenId(tk)}") \\`)
  );

  emit();
  emit("#define FOR_EACH_OPERATOR(V) \\");
  tokens.OPERATORS.forEach(([tk, spelling]) =>
    emit(`  V(${tk}, "${spelling}") \\`)
  );

  emit();
  emit("#define FOR_EACH_KEYWORD(V) \\");
  tokens.KEYWORDS.forEach((tk) => emit(`  V(${tk.toUpperCase()}, "${tk}") \\`));

  emit();
  emit("#define FOR_EACH_BUILTIN(V) \\");
  tokens.BUILTINS.forEach((tk) => emit(`  V(${tk.toUpperCase()}, "${tk}") \\`));

  emit();
  emit("#define FOR_EACH_TOKEN_ALIAS(V) \\");
  tokens.TOKEN_ALIASES.forEach(([tk, other]) =>
    emit(`  V(${tk.toUpperCase()}, ${other}) \\`)
  );

  const out = `${cpy_header}
#pragma once

#include <cxx/cxx_fwd.h>
#include <cstdint>

namespace cxx {

union TokenValue;
class Token;

${code.join("\n")}

#define FOR_EACH_TOKEN(V) \\
  FOR_EACH_BASE_TOKEN(V) \\
  FOR_EACH_OPERATOR(V) \\
  FOR_EACH_KEYWORD(V)

// clang-format off
#define TOKEN_ENUM(tk, _) T_##tk,
#define TOKEN_ALIAS_ENUM(tk, other) T_##tk = T_##other,
enum struct TokenKind : std::uint8_t {
  FOR_EACH_TOKEN(TOKEN_ENUM)
  FOR_EACH_TOKEN_ALIAS(TOKEN_ALIAS_ENUM)
};

enum struct BuiltinKind {
  T_IDENTIFIER,
  FOR_EACH_BUILTIN(TOKEN_ENUM)
};

#undef TOKEN_ENUM
#undef TOKEN_ALIAS_ENUM
// clang-format on

}  // namespace cxx
`;

  fs.writeFileSync(output, out);
}
