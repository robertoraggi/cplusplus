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

import { cpy_header } from "./cpy_header.ts";
import { groupNodesByBaseType } from "./groupNodesByBaseType.ts";
import type { AST } from "./parseAST.ts";
import * as fs from "fs";

export function gen_syntax_cc({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    if (!Array.isArray(nodes)) throw new Error("not an array");
    emit();
    const variantName = base.replace(/AST$/, "");
    emit(`auto from(${base}* ast) -> std::optional<${variantName}> {`);
    emit(`  if (!ast) return std::nullopt;`);
    emit(`  switch (ast->kind()) {`);
    nodes.forEach(({ name }) => {
      emit(
        `  case ${name}::Kind: return ${variantName}(static_cast<${name}*>(ast));`,
      );
    });
    emit(`    default: cxx_runtime_error("unexpected ${variantName}");`);
    emit(`  } // switch`);
    emit(`}`);
  });

  const out = `// Generated file by: gen_syntax_cc.ts
${cpy_header}

#include <cxx/syntax.h>
#include <cxx/ast.h>

namespace cxx::syntax {

${code.join("\n")}

} // namespace cxx::syntax
`;

  fs.writeFileSync(output, out);
}
