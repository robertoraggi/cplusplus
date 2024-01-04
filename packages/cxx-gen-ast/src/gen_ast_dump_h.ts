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

export function gen_ast_dump_h({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  emit(`class ASTPrinter : ASTVisitor {`);
  emit(`public:`);
  emit(`  explicit ASTPrinter(TranslationUnit* unit, std::ostream& out);`);
  emit();
  emit(`  void operator()(AST* ast);`);

  emit(`private:`);
  emit(`  void accept(AST* ast, std::string_view field = {});`);
  emit(`  void accept(const Identifier* id, std::string_view field = {});`);
  by_base.forEach((nodes) => {
    emit();
    nodes.forEach(({ name }) => {
      emit(`  void visit(${name}* ast) override;`);
    });
  });
  emit(`private:`);
  emit(`  TranslationUnit* unit_;`);
  emit(`  std::ostream& out_;`);
  emit(`  int indent_ = -1;`);
  emit(`};`);

  const out = `${cpy_header}
#pragma once

#include <cxx/ast_visitor.h>
#include <cxx/names_fwd.h>
#include <cxx/literals_fwd.h>
#include <cxx/types_fwd.h>
#include <iosfwd>
#include <string_view>

namespace cxx {

class TranslationUnit;

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
