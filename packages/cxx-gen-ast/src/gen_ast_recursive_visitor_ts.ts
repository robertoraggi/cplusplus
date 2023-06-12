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
import * as fs from "fs";

export function gen_ast_recursive_visitor_ts({
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
  baseClassNames.push("AttributeAST");
  baseClassNames.sort();

  emit();

  const getterName = (name: string) =>
    `get${name[0].toUpperCase() + name.substr(1)}`;
  const nodeName = (name: string) => name.slice(0, -3);

  emit(
    `export class RecursiveASTVisitor<Context> extends ASTVisitor<Context, void> {`
  );
  emit(`    constructor() {`);
  emit(`        super();`);
  emit(`    }`);
  emit();
  emit(`    accept(node: ast.AST | undefined, context: Context) {`);
  emit(`        node?.accept(this, context);`);
  emit(`    }`);

  by_bases.forEach((nodes) => {
    nodes.forEach(({ name, members }) => {
      emit();
      emit(
        `    visit${nodeName(
          name
        )}(node: ast.${name}, context: Context): void {`
      );
      members.forEach((m) => {
        switch (m.kind) {
          case "node":
            emit(`        this.accept(node.${getterName(m.name)}(), context);`);
            break;
          case "node-list":
            emit(
              `        for (const element of node.${getterName(m.name)}()) {`
            );
            emit(`            this.accept(element, context);`);
            emit(`        }`);
            break;
        } // switch
      });
      emit(`    }`);
    });
  });

  emit(`}`);
  emit();

  const out = `${cpy_header}
import * as ast from "./AST.js";
import { ASTVisitor } from "./ASTVisitor.js";
${code.join("\n")}
`;

  fs.writeFileSync(output, out);
}
