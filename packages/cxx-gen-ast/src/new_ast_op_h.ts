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

import { cpy_header } from "./cpy_header.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import { AST } from "./parseAST.js";
import * as fs from "fs";

export function new_ast_op_h({
  ast,
  opName,
  output,
}: {
  ast: AST;
  opName: string;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  // chop the AST suffix for the given name
  const chopAST = (name: string) => {
    if (name.endsWith("AST")) return name.slice(0, -3);
    return name;
  };

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    emit(`  struct ${chopAST(base)}Result {};`);
  });
  emit();
  by_base.get("AST")?.forEach(({ name }) => {
    emit(`  struct ${chopAST(name)}Result {};`);
  });

  emit();
  emit(`  // run on the base nodes`);
  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    const resultTy = `${chopAST(base)}Result`;
    emit(`  [[nodiscard]] auto operator()(${base}* ast) -> ${resultTy};`);
  });
  emit();
  emit(`  // run on the misc nodes`);
  by_base.get("AST")?.forEach(({ name }) => {
    const resultTy = `${chopAST(name)}Result`;
    emit(`  auto operator()(${name}* ast) -> ${resultTy};`);
  });

  emit();
  emit(`private:`);
  by_base.forEach((nodes, base) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    if (base === "AST") return;
    const className = chopAST(base);
    const resultTy = `${chopAST(base)}Result`;
    emit();
    emit(`  struct ${className}Visitor {`);
    emit(`    ${opName}& accept;`);
    nodes.forEach(({ name }) => {
      emit();
      emit(`    [[nodiscard]] auto operator()(${name}* ast) -> ${resultTy};`);
    });
    emit(`  };`);
  });

  const out = `${cpy_header}

#pragma once

#include <cxx/ast_fwd.h>

namespace cxx {

class TranslationUnit;
class Control;

class ${opName} {
public:
    explicit ${opName}(TranslationUnit* unit);
    ~${opName}();

    [[nodiscard]] auto translationUnit() const -> TranslationUnit* { return unit_; }

    [[nodiscard]] auto control() const -> Control*;

${code.join("\n")}

    TranslationUnit* unit_ = nullptr;
};

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
