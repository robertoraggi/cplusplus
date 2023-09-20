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
import { AST } from "./parseAST.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import { format } from "prettier";
import * as fs from "fs";

export async function gen_ast_visitor_ts({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_bases = groupNodesByBaseType(ast);

  const baseClassNames = Array.from(by_bases.keys()).filter((b) => b !== "AST");
  baseClassNames.sort();

  emit();

  const nodeName = (name: string) => name.slice(0, -3);

  emit(`export abstract class ASTVisitor<Context, Result> {`);
  emit(`    constructor() { }`);

  by_bases.forEach((nodes, base) => {
    emit();
    emit(`    // ${base}`);
    nodes.forEach(({ name }) => {
      emit(
        `    abstract visit${nodeName(
          name
        )}(node: ast.${name}, context: Context): Result;`
      );
    });
  });

  emit(`}`);
  emit();

  const out = `${cpy_header}
import * as ast from "./AST";

${code.join("\n")}
`;

  fs.writeFileSync(output, await format(out, { parser: "typescript" }));
}
