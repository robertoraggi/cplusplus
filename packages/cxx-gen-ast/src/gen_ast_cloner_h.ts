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

export function gen_ast_cloner_h({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  emit(`class ASTCloner : public ASTVisitor {`);
  emit(`public:`);
  emit(`virtual auto clone(AST* ast, Arena* arena) -> AST*;`);
  emit();

  by_base.forEach((nodes) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    emit();
    nodes.forEach(({ name }) => {
      emit(`  void visit(${name}* ast) override;`);
    });
  });

  emit();
  emit(`protected:`);
  emit(`  template <typename T> auto accept(T ast) -> T {`);
  emit(`    if (!ast) return nullptr;`);
  emit(`    AST* copy = nullptr;`);
  emit(`    std::swap(copy_, copy);`);
  emit(`    ast->accept(this);`);
  emit(`    std::swap(copy_, copy);`);
  emit(`    return static_cast<T>(copy);`);
  emit(`  }`);
  emit();
  emit(`  Arena* arena_ = nullptr;`);
  emit(`  AST* copy_ = nullptr;`);
  emit(`};`);

  const out = `${cpy_header}
#pragma once

#include <cxx/ast_visitor.h>
#include <cxx/ast.h>
#include <cxx/arena.h>

namespace cxx {

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
