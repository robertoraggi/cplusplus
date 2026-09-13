// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

import type { AST, Attribute, Member } from "./parseAST.ts";
import {
  ModelIndex,
  type ModelField,
  type ModelType,
  unqualified,
  typeArguments,
} from "./parseModel.ts";

function shortName(name: string): string {
  return name.replaceAll("::cxx::", "").replaceAll("::std::", "std::");
}

function attribute(field: ModelField): Attribute {
  let type = field.type;
  let ptrOps = "";
  let cv = "";
  while (type.kind === "pointer") {
    ptrOps += "*";
    type = type.element;
  }
  if (type.kind === "qual") {
    if (type.isVolatile) throw new Error(`volatile AST field ${field.name}`);
    if (type.isConst) cv = "const";
    type = type.element;
  }
  let name = field.declaredType;
  if (!name)
    throw new Error(
      `missing declared type for ${field.name}; refresh the semantic snapshot`,
    );
  if (field.initializer && field.initializerText === undefined)
    throw new Error(`missing initializer text for ${field.name}`);
  {
    if (cv) name = name.slice(cv.length).trim();
    for (const op of ptrOps) {
      if (!name.endsWith(op))
        throw new Error(`invalid declared type for ${field.name}`);
      name = name.slice(0, -1).trim();
    }
  }
  return {
    kind: "attribute",
    name: field.name,
    cv,
    ptrOps,
    type: name,
    initializer: field.initializerText ?? "",
  };
}

export function astFromModel(index: ModelIndex): AST {
  const ast: AST = { nodes: [], bases: [], baseMembers: new Map() };
  const classes = index.model.classes.filter(
    (c) => index.fileOf(c) === "ast.h",
  );
  const names = new Set(
    classes.filter((c) => c.name.endsWith("AST")).map((c) => c.name),
  );
  function member(field: ModelField): Member {
    const type = unqualified(field.type);
    if (type.kind === "class" && type.name === "::cxx::SourceLocation")
      return { kind: "token", name: field.name };
    if (type.kind === "pointer") {
      const element = unqualified(type.element);
      if (element.kind === "class") {
        if (names.has(element.name))
          return {
            kind: "node",
            name: field.name,
            type: shortName(element.name),
          };
        if (element.name === "::cxx::List") {
          const item = typeArguments(element)[0]!;
          if (item.kind === "pointer") {
            const target = unqualified(item.element);
            if (target.kind === "class" && names.has(target.name))
              return {
                kind: "node-list",
                name: field.name,
                type: shortName(target.name),
              };
          }
          if (item.kind === "class" && item.name === "::cxx::SourceLocation")
            return {
              kind: "token-list",
              name: field.name,
              type: "SourceLocation",
            };
          throw new Error(`unsupported AST list ${field.name}`);
        }
      }
    }
    return attribute(field);
  }
  for (const entry of classes) {
    if (entry.name === "::cxx::AST") continue;
    const base = entry.bases[0];
    if (!base || !names.has(base.type)) continue;
    if (!entry.isFinal) {
      ast.bases.push(entry.unqualifiedName);
      ast.baseMembers.set(entry.unqualifiedName, entry.fields.map(attribute));
      continue;
    }
    ast.nodes.push({
      name: entry.unqualifiedName,
      base: shortName(base.type),
      members: entry.fields.map(member),
    });
  }
  if (!ast.nodes.length)
    throw new Error("semantic model contains no AST nodes");
  return ast;
}
