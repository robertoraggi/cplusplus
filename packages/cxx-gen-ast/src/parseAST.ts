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

export { astFromModel as parseAST } from "./astFromModel.ts";

export function getASTNodes(members: Member[]): Member[] {
  return members.filter((m) => m.kind === "node" || m.kind === "node-list");
}
