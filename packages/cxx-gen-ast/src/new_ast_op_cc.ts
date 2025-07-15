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

export function new_ast_op_cc({
  ast,
  opName,
  opHeader,
  output,
}: {
  ast: AST;
  opName: string;
  opHeader: string;
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

  by_base.forEach((_nodes, base) => {
    if (base === "AST") return;
    emit();
    emit(`  struct ${opName}::${chopAST(base)}Result {};`);
  });

  by_base.get("AST")?.forEach(({ name }) => {
    emit();
    emit(`  struct ${opName}::${chopAST(name)}Result {};`);
  });

  by_base.forEach((nodes, base) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    if (base === "AST") return;
    const className = chopAST(base);
    const resultTy = `${chopAST(base)}Result`;
    emit();
    emit(`  struct ${opName}::${className}Visitor {`);
    emit(`    ${opName}& accept;`);
    nodes.forEach(({ name }) => {
      emit();
      emit(`    [[nodiscard]] auto operator()(${name}* ast) -> ${resultTy};`);
    });
    emit(`  };`);
  });

  by_base.forEach((_nodes, base) => {
    if (base === "AST") return;
    const resultTy = `${chopAST(base)}Result`;
    emit();
    emit(`auto ${opName}::operator()(${base}* ast) -> ${resultTy} {`);
    emit(`  if (ast)`);
    emit(`    return visit(${chopAST(base)}Visitor{*this}, ast);`);
    emit(`  return {};`);
    emit(`}`);
  });
  by_base.get("AST")?.forEach(({ name, members }) => {
    emit();
    const resultTy = `${chopAST(name)}Result`;
    emit(`auto ${opName}::operator()(${name}* ast) -> ${resultTy} {`);
    emit(`  if (!ast) return {};`);
    emit();
    members.forEach((m) => {
      switch (m.kind) {
        case "node": {
          emit(`  auto ${m.name}Result = operator()(ast->${m.name});`);
          break;
        }
        case "node-list": {
          emit();
          emit(`  for (auto node : ListView{ast->${m.name}}) {`);
          emit(`    auto value = operator()(node);`);
          emit(`  }`);
          emit();
          break;
        }
      }
    });
    emit();
    emit(`  return {};`);
    emit(`}`);
  });

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    if (!Array.isArray(nodes)) throw new Error("not an array");
    const className = chopAST(base);
    nodes.forEach(({ name, members }) => {
      const resultTy = `${chopAST(base)}Result`;
      emit();
      emit(
        `auto ${opName}::${className}Visitor::operator()(${name}* ast) -> ${resultTy} {`,
      );
      members.forEach((m) => {
        switch (m.kind) {
          case "node": {
            emit(`  auto ${m.name}Result = accept(ast->${m.name});`);
            break;
          }
          case "node-list": {
            emit();
            emit(`  for (auto node : ListView{ast->${m.name}}) {`);
            emit(`    auto value = accept(node);`);
            emit(`  }`);
            emit();
            break;
          }
        }
      });
      emit();
      emit(`  return {};`);
      emit(`}`);
    });
  });

  const out = `// Generated file by: new_ast_op_cc.ts
${cpy_header}

#include <cxx/${opHeader}>

// cxx
#include <cxx/ast.h>
#include <cxx/translation_unit.h>
#include <cxx/control.h>

namespace cxx {

${code.join("\n")}

${opName}::${opName}(TranslationUnit* unit) : unit_(unit) {}

${opName}::~${opName}() {}

auto ${opName}::control() const -> Control* {
    return unit_->control();
}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
