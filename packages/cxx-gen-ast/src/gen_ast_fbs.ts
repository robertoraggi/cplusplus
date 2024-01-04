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
import { AST } from "./parseAST.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import * as fs from "fs";

export function gen_ast_fbs({ ast, output }: { ast: AST; output: string }) {
  const withTokens = true;
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_bases = groupNodesByBaseType(ast);

  const className = (name: string) =>
    name != "AST" ? name.slice(0, -3) : name;

  const baseNodeNames = Array.from(by_bases.keys()).sort();

  const toSnakeName = (name: string) =>
    name.replace(/([A-Z])/g, "_$1").toLocaleLowerCase();

  for (const base of baseNodeNames) {
    const baseClassName = className(base);
    emit(`union ${baseClassName} {`);
    by_bases.get(base)?.forEach((derived) => {
      emit(`  ${className(derived.name)},`);
    });
    emit(`}`);
    emit();
  }

  for (const base of baseNodeNames) {
    by_bases.get(base)?.forEach(({ name, base, members }) => {
      emit(`table ${className(name)} /* ${base} */ {`);

      members.forEach((m) => {
        const fieldName = toSnakeName(m.name);
        switch (m.kind) {
          case "node":
            emit(`  ${fieldName}: ${className(m.type)};`);
            break;
          case "node-list":
            emit(`  ${fieldName}: [${className(m.type)}];`);
            break;
          case "token":
            break;
          case "token-list":
            break;
          case "attribute": {
            if (
              [
                "Identifier",
                "CharLiteral",
                "StringLiteral",
                "IntegerLiteral",
                "FloatLiteral",
              ].includes(m.type)
            ) {
              emit(`  ${fieldName}: string;`);
            } else if (m.type === "TokenKind") {
              emit(`  ${fieldName}: uint32;`);
            } else {
              // emit(`  // skip: ${fieldName}: ${m.type};`);
            }
            break;
          }
        }
      });

      if (withTokens) {
        members.forEach((m) => {
          const fieldName = toSnakeName(m.name);
          switch (m.kind) {
            case "node":
              break;
            case "node-list":
              break;
            case "token":
              emit(`  ${fieldName}: SourceLocation;`);
              break;
            case "token-list":
              emit(`  ${fieldName}: [SourceLocation];`);
              break;
            case "attribute": {
              break;
            }
          }
        });
      }

      emit(`}`);
      emit();
    });
  }

  const out = `${cpy_header}
namespace cxx.io;

table SourceLine {
  file_name: string;
  line: uint32;
}

table SourceLocation {
  source_line: SourceLine;
  column: uint32;
}

${code.join("\n")}

table SerializedUnit {
  version: uint32;
  unit: Unit;
  file_name: string;
}

root_type SerializedUnit;
file_identifier "AST0";
file_extension "ast";
`;

  fs.writeFileSync(output, out);
}
