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
import { AST, Member } from "./parseAST.js";
import * as fs from "fs";

export function new_ast_rewriter_cc({
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

  const parent = new Map<string, string>();
  by_base.forEach((nodes, base) => {
    nodes.forEach(({ name }) => {
      parent.set(name, base);
    });
  });

  const isBase = (name: string) =>
    by_base.has(name) || parent.get(name) === "AST";

  // chop the AST suffix for the given name
  const chopAST = (name: string) => {
    if (name.endsWith("AST")) return name.slice(0, -3);
    return name;
  };

  const emitRewriterBody = (members: Member[], visitor: string = "rewrite") => {
    members.forEach((m) => {
      switch (m.kind) {
        case "node": {
          if (isBase(m.type)) {
            emit(`  copy->${m.name} = ${visitor}(ast->${m.name});`);
          } else {
            emit(
              `  copy->${m.name} = ast_cast<${m.type}>(${visitor}(ast->${m.name}));`,
            );
          }
          break;
        }
        case "node-list": {
          emit();
          emit(`  if (auto it = ast->${m.name}) {`);
          emit(`    auto out = &copy->${m.name};`);
          emit(`    for (auto it = ast->${m.name}; it; it = it->next) {`);
          emit(`        auto value = ${visitor}(it->value);`);
          if (isBase(m.type)) {
            emit(`*out = new (arena()) List(value);`);
          } else {
            emit(`*out = new (arena()) List(ast_cast<${m.type}>(value));`);
          }
          emit(`        out = &(*out)->next;`);
          emit(`    }`);
          emit(`  }`);
          emit();
          break;
        }
        case "attribute": {
          emit(`    copy->${m.name} = ast->${m.name};`);
          break;
        }
        case "token": {
          emit(`    copy->${m.name} = ast->${m.name};`);
          break;
        }
      }
    });
  };

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    emit();
    emit(`auto ${opName}::operator()(${base}* ast) -> ${base}* {`);
    emit(`  if (ast)`);
    emit(`    return visit(${chopAST(base)}Visitor{*this}, ast);`);
    emit(`  return {};`);
    emit(`}`);
  });
  by_base.get("AST")?.forEach(({ name, members }) => {
    emit();
    emit(`auto ${opName}::operator()(${name}* ast) -> ${name}* {`);
    emit(`  if (!ast) return {};`);
    emit();
    emit(`  auto copy = new (arena()) ${name}{};`);
    emit();
    emitRewriterBody(members, "operator()");
    emit();
    emit(`  return copy;`);
    emit(`}`);
  });

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    if (!Array.isArray(nodes)) throw new Error("not an array");
    const className = chopAST(base);
    nodes.forEach(({ name, members }) => {
      emit();
      emit(
        `auto ${opName}::${className}Visitor::operator()(${name}* ast) -> ${base}* {`,
      );
      emit(`  auto copy = new (arena()) ${name}{};`);
      emit();
      ast.baseMembers.get(base)?.forEach((m) => {
        emit(`  copy->${m.name} = ast->${m.name};`);
      });
      emitRewriterBody(members, "rewrite");
      emit();
      emit(`  return copy;`);
      emit(`}`);
    });
  });

  const out = `${cpy_header}

#include <cxx/${opHeader}>

// cxx
#include <cxx/ast.h>
#include <cxx/translation_unit.h>
#include <cxx/control.h>

namespace cxx {

${opName}::${opName}(TranslationUnit* unit) : unit_(unit) {}

${opName}::~${opName}() {}

auto ${opName}::control() const -> Control* {
    return unit_->control();
}

auto ${opName}::arena() const -> Arena* {
    return unit_->arena();
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
