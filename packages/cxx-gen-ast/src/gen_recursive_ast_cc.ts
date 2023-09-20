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

import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import { AST, getASTNodes } from "./parseAST.js";
import { cpy_header } from "./cpy_header.js";
import * as fs from "fs";

export function gen_recursive_ast_cc({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const funcName = (base: string) =>
    "accept" + base[0].toUpperCase() + base.slice(1, -3);

  const types = new Set<string>();

  ast.nodes.forEach((node) => {
    node.members.forEach((m) => {
      if (m.kind === "node" || m.kind === "node-list") types.add(m.type);
    });
  });

  Array.from(types.values()).forEach((base) => {
    emit(
      `void RecursiveASTVisitor::${funcName(
        base
      )}(${base}* ast){ accept(ast); }`
    );
    emit();
  });

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, members }) => {
      members = getASTNodes(members);

      emit();
      emit(`void RecursiveASTVisitor::visit(${name}* ast) {`);
      members.forEach((m) => {
        if (m.kind === "node") {
          emit(`    ${funcName(m.type)}(ast->${m.name});`);
        } else if (m.kind === "node-list") {
          emit(`    for (auto it = ast->${m.name}; it; it = it->next) {`);
          emit(`      ${funcName(m.type)}(it->value);`);
          emit(`}`);
        }
      });
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include <cxx/ast.h>
#include <cxx/recursive_ast_visitor.h>

namespace cxx {

void RecursiveASTVisitor::accept(AST* ast) {
    if (!ast) return;
    if (preVisit(ast)) ast->accept(this);
    postVisit(ast);
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
