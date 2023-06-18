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

import { cpy_header } from "./cpy_header.js";
import { AST } from "./parseAST.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import * as fs from "fs";

export function gen_ast_fbs({ ast, output }: { ast: AST; output: string }) {
  const withTokens = true;
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_bases = groupNodesByBaseType(ast);
  by_bases.set("AttributeAST", []);

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
              emit(`  ${fieldName}: Token;`);
              break;
            case "token-list":
              emit(`  ${fieldName}: [Token];`);
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

union TokenValue {
  Identifier,
  StringConstant,
  FloatConstant,
  IntConstant,
}

table Identifier {
  name: string;
}

table FloatConstant {
  value: float64;
}

table IntConstant {
  value: int64;
}

table StringConstant {
  value: string;
}

table Token {
  kind: uint16;
  flags: uint16;
  offset: uint32;
  length: uint32;
  value: TokenValue;
}

${code.join("\n")}

table SerializedUnit {
  version: uint32;
  unit: Unit;
  file_name: string;
  identifiers: [string];
  integer_literals: [string];
  float_literals: [string];
  char_literals: [string];
  string_literals: [string];
  comment_literals: [string];
  wide_string_literals: [string];
  utf8_string_literals: [string];
  utf16_string_literals: [string];
  utf32_string_literals: [string];
}

root_type SerializedUnit;
file_identifier "AST0";
file_extension "ast";
`;

  fs.writeFileSync(output, out);
}
