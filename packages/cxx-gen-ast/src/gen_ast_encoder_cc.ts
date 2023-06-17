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
import { AST } from "./parseAST.js";
import { cpy_header } from "./cpy_header.js";
import * as fs from "fs";

export function gen_ast_encoder_cc({
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
    nodes.forEach(({ name, members }) => {
      emit();
      emit(`void ASTEncoder::visit(${name}* ast) {`);
      emit();
      members.forEach((m) => {
        emit();
        if (m.kind === "node") {
          // todo
        } else if (m.kind === "node-list") {
          // todo
        } else if (m.kind === "attribute" && m.type.endsWith("Literal")) {
          // todo
        } else if (m.kind === "attribute" && m.type === "Identifier") {
          // todo
        } else if (m.kind === "attribute" && m.type === "TokenKind") {
          // todo
        }
      });
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include "ast_encoder.h"

#include <cxx/ast.h>
#include <cxx/translation_unit.h>
#include <cxx/names.h>
#include <cxx/literals.h>

#include <algorithm>

namespace cxx {

void ASTEncoder::accept(AST* ast) {
  if (!ast) return;
  ast->accept(this);
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
