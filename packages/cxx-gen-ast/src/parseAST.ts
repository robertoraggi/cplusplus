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

export type Member = Token | TokenList | Node | NodeList | Attribute;

export interface AST {
  nodes: Class[];
  bases: string[];
  baseMembers: Map<string, Attribute[]>;
}

export interface Class {
  name: string;
  base: string;
  members: Member[];
}

export interface Token {
  kind: "token";
  name: string;
}

export interface Node {
  kind: "node";
  name: string;
  type: string;
}

export interface TokenList {
  kind: "token-list";
  name: string;
  type: string;
}

export interface NodeList {
  kind: "node-list";
  name: string;
  type: string;
}

export interface Attribute {
  kind: "attribute";
  cv: string;
  type: string;
  ptrOps: string;
  name: string;
  initializer: string;
}

interface ParseArgs {
  fn: string;
  source: string;
}

export function parseAST({ source }: ParseArgs): AST {
  const baseRx = /^\s*class (.+AST) : public AST {/;
  const classRx = /^\s*class (.+AST) final : public (.*AST) {/;
  const tokRx = /^\s+SourceLocation (\w+Loc);$/;
  const astRx = /^\s+(\w*AST)\* (\w+) = nullptr;$/;
  const listRx = /^\s+List<(\w*AST)\*>\* (\w+) = nullptr;$/;
  const tokListRx = /^\s+List<(SourceLocation)>\* (\w+) = nullptr;$/;
  const attrRx =
    /^\s+(?:(const)\s+)?([\w:<>]+)\s*([*]*)\s*(\w+)(?:\s+=\s+(.*))?;$/;

  const nodes: Class[] = [];
  const bases: string[] = [];
  const baseMembers = new Map<string, Attribute[]>();

  const lines = source.split("\n");

  let i = 0;
  while (i < lines.length) {
    const line = lines[i++];
    const classMatch = classRx.exec(line);
    const baseMatch = baseRx.exec(line);
    if (baseMatch) {
      bases.push(baseMatch[1]);
      const members: Attribute[] = [];
      baseMembers.set(baseMatch[1], members);
      while (i < lines.length) {
        if (lines[i].match(/^};$/)) break;
        const attrMatch = attrRx.exec(lines[i]);
        if (attrMatch) {
          //console.log(`  list ${listMatch[2]}: ${listMatch[1]}`)
          members.push({
            kind: "attribute",
            cv: attrMatch[1],
            type: attrMatch[2],
            ptrOps: attrMatch[3],
            name: attrMatch[4],
            initializer: attrMatch[5],
          });
        }
        ++i;
      }
      //console.log(`base ${baseMatch[1]} with attributes`, members);
      continue;
    }
    if (!classMatch) continue;
    const className = classMatch[1];
    //console.log(`class '${match[1]}'`);
    const members: Member[] = [];
    const node = { name: className, base: classMatch[2], members };
    nodes.push(node);
    while (i < lines.length) {
      if (/^};/.exec(lines[i])) break;
      const tokMatch = tokRx.exec(lines[i]);
      const astMatch = astRx.exec(lines[i]);
      const listMatch = listRx.exec(lines[i]);
      const tokListMatch = tokListRx.exec(lines[i]);
      const attrMatch = attrRx.exec(lines[i]);
      if (tokMatch) {
        //console.log(`  tok ${tokMatch[1]}`)
        members.push({ kind: "token", name: tokMatch[1] });
      } else if (astMatch) {
        //console.log(`  child ${astMatch[2]}: ${astMatch[1]}`)
        members.push({ kind: "node", name: astMatch[2], type: astMatch[1] });
      } else if (listMatch) {
        //console.log(`  list ${listMatch[2]}: ${listMatch[1]}`)
        members.push({
          kind: "node-list",
          name: listMatch[2],
          type: listMatch[1],
        });
      } else if (tokListMatch) {
        //console.log(`  list ${listMatch[2]}: ${listMatch[1]}`)
        members.push({
          kind: "token-list",
          name: tokListMatch[2],
          type: tokListMatch[1],
        });
      } else if (attrMatch) {
        //console.log(`  list ${listMatch[2]}: ${listMatch[1]}`)
        members.push({
          kind: "attribute",
          cv: attrMatch[1],
          type: attrMatch[2],
          ptrOps: attrMatch[3],
          name: attrMatch[4],
          initializer: attrMatch[5],
        });
      }
      ++i;
    }
  }

  return { nodes, bases, baseMembers };
}

export function getASTNodes(members: Member[]): Member[] {
  return members.filter((m) => m.kind === "node" || m.kind == "node-list");
}
