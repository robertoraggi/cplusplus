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

import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import { AST } from "./parseAST.js";
import { cpy_header } from "./cpy_header.js";
import * as fs from "fs";

export function gen_ast_cloner_cc({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, base, members }) => {
      emit();
      emit(`void ASTCloner::visit(${name}* ast) {`);
      emit(`  auto copy = new (arena_) ${name}();`);
      emit(`  copy_ = copy;`);
      emit();
      ast.baseMembers.get(base)?.forEach((m) => {
        emit();
        emit(`    copy->${m.name} = ast->${m.name};`);
      });
      members.forEach((m) => {
        if (m.kind === "token") {
          emit();
          emit(`    copy->${m.name} = ast->${m.name};`);
        } else if (m.kind === "node") {
          emit();
          emit(`    copy->${m.name} = accept(ast->${m.name});`);
        } else if (m.kind === "node-list") {
          emit();
          emit(`  if (auto it = ast->${m.name}) {`);
          emit(`    auto out = &copy->${m.name};`);
          emit();
          emit(`    for (; it; it = it->next) {`);
          emit(`      *out = new (arena_) List(accept(it->value));`);
          emit(`      out = &(*out)->next;`);
          emit(`    }`);
          emit(`  }`);
        } else if (m.kind === "token-list") {
          emit();
          emit(`  if (auto it = ast->${m.name}) {`);
          emit(`    auto out = &copy->${m.name};`);
          emit();
          emit(`    for (; it; it = it->next) {`);
          emit(`      *out = new (arena_) List(it->value);`);
          emit(`      out = &(*out)->next;`);
          emit(`    }`);
          emit(`  }`);
        } else if (m.kind === "attribute") {
          emit();
          emit(`    copy->${m.name} = ast->${m.name};`);
        }
      });
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include <cxx/ast_cloner.h>

namespace cxx {

auto ASTCloner::clone(AST* ast, Arena* arena) -> AST* {
    if (!ast) return nullptr;
    std::swap(arena_, arena);
    auto copy = accept(ast);
    std::swap(arena_, arena);
    return copy;
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
