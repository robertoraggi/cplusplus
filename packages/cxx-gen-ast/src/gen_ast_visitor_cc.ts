// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

import { groupNodesByBaseType } from "./groupNodesByBaseType.ts";
import { type AST, getASTNodes } from "./parseAST.ts";
import { cpy_header } from "./cpy_header.ts";
import * as fs from "fs";

export function gen_ast_visitor_cc({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const types = new Set<string>();

  ast.nodes.forEach((node) => {
    node.members.forEach((m) => {
      if (m.kind === "node" || m.kind === "node-list") types.add(m.type);
    });
  });

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, members }) => {
      members = getASTNodes(members);

      emit();
      emit(`void ASTVisitor::visit(${name}* ast) {`);
      members.forEach((m) => {
        if (m.kind === "node") {
          emit(`accept(ast->${m.name});`);
        } else if (m.kind === "node-list") {
          emit(`for (auto node : ListView{ast->${m.name}}) {`);
          emit(`accept(node);`);
          emit(`}`);
        }
      });
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include <cxx/ast_visitor.h>

// cxx
#include <cxx/ast.h>

namespace cxx {

auto ASTVisitor::preVisit(AST*) -> bool {
  return true;
}

void ASTVisitor::postVisit(AST*) {}

void ASTVisitor::accept(AST* ast) {
    if (!ast) return;
    if (preVisit(ast)) ast->accept(this);
    postVisit(ast);
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
