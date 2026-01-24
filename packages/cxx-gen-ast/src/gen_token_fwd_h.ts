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

export function gen_token_fwd_h({ output }: { output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const baseTokenId = (s: string) => `<${s.toLowerCase()}>`;

  emit("#define FOR_EACH_BASE_TOKEN(V) \\");
  tokens.BASE_TOKENS.forEach((tk) =>
    emit(`  V(${tk}, "${baseTokenId(tk)}") \\`),
  );

  emit();
  emit("#define FOR_EACH_OPERATOR(V) \\");
  tokens.OPERATORS.forEach(([tk, spelling]) =>
    emit(`  V(${tk}, "${spelling}") \\`),
  );

  emit();
  emit("#define FOR_EACH_KEYWORD(V) \\");
  tokens.C_AND_CXX_KEYWORDS.forEach((tk) =>
    emit(`  V(${tk.toUpperCase()}, "${tk}") \\`),
  );

  emit();
  emit("#define FOR_EACH_BUILTIN_TYPE_TRAIT(V) \\");
  tokens.BUILTIN_TYPE_TRAITS.forEach((tk) =>
    emit(`  V(${tk.toUpperCase()}, "${tk}") \\`),
  );

  emit();
  emit("#define FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(V) \\");
  tokens.UNARY_BUILTIN_TYPE_SPECIFIERS.forEach((tk) =>
    emit(`  V(${tk.toUpperCase()}, "${tk}") \\`),
  );

  emit();
  emit("#define FOR_EACH_BINARY_BUILTIN_TYPE_TRAIT(V) \\");
  tokens.BINARY_BUILTIN_TYPE_SPECIFIERS.forEach((tk) =>
    emit(`  V(${tk.toUpperCase()}, "${tk}") \\`),
  );

  emit();
  emit("#define FOR_EACH_TOKEN_ALIAS(V) \\");
  Object.entries(tokens.C_AND_CXX_TOKEN_ALIASES).forEach(([tk, other]) =>
    emit(`  V(${tk.toUpperCase()}, ${other}) \\`),
  );

  const firstKeyword = tokens.C_AND_CXX_KEYWORDS[0];

  const lastKeyword =
    tokens.C_AND_CXX_KEYWORDS[tokens.C_AND_CXX_KEYWORDS.length - 1];

  const out = `// Generated file by: gen_token_fwd_h.ts
${cpy_header}
#pragma once

#include <cxx/cxx_fwd.h>
#include <optional>
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
enum class TokenKind : std::uint8_t {
  FOR_EACH_TOKEN(TOKEN_ENUM)
  FOR_EACH_TOKEN_ALIAS(TOKEN_ALIAS_ENUM)
};

enum class BuiltinTypeTraitKind {
  T_NONE,
  FOR_EACH_BUILTIN_TYPE_TRAIT(TOKEN_ENUM)
};

enum class UnaryBuiltinTypeKind {
  T_NONE,
  FOR_EACH_UNARY_BUILTIN_TYPE_TRAIT(TOKEN_ENUM)
};

enum class BinaryBuiltinTypeKind {
  T_NONE,
  FOR_EACH_BINARY_BUILTIN_TYPE_TRAIT(TOKEN_ENUM)
};

#undef TOKEN_ENUM
#undef TOKEN_ALIAS_ENUM
// clang-format on

[[nodiscard]] inline auto is_keyword(TokenKind tk) -> bool {
  return tk >= TokenKind::T_${firstKeyword.toUpperCase()} &&
         tk <= TokenKind::T_${lastKeyword.toUpperCase()};
}

[[nodiscard]] inline auto get_underlying_binary_op(TokenKind op) -> TokenKind {
  switch (op) {
    case TokenKind::T_STAR_EQUAL:
      return TokenKind::T_STAR;
    case TokenKind::T_SLASH_EQUAL:
      return TokenKind::T_SLASH;
    case TokenKind::T_PERCENT_EQUAL:
      return TokenKind::T_PERCENT;
    case TokenKind::T_PLUS_EQUAL:
      return TokenKind::T_PLUS;
    case TokenKind::T_MINUS_EQUAL:
      return TokenKind::T_MINUS;
    case TokenKind::T_LESS_LESS_EQUAL:
      return TokenKind::T_LESS_LESS;
    case TokenKind::T_AMP_EQUAL:
      return TokenKind::T_AMP;
    case TokenKind::T_CARET_EQUAL:
      return TokenKind::T_CARET;
    case TokenKind::T_BAR_EQUAL:
      return TokenKind::T_BAR;
    case TokenKind::T_GREATER_GREATER_EQUAL:
      return TokenKind::T_GREATER_GREATER;
    default:
      return TokenKind::T_EOF_SYMBOL;
  }  // switch
}

}  // namespace cxx
`;

  fs.writeFileSync(output, out);
}
