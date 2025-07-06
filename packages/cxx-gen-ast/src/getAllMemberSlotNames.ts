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

import { groupNodesByBaseType } from "./groupNodesByBaseType.ts";
import type { AST, Member } from "./parseAST.ts";

export const MemberSlotClassification = [
  "bool-attr",
  "int-attr",
  "token-kind-attr",
  "identifier-attr",
  "literal-attr",
  "token",
  "node",
  "node-list",
  "token-list",
] as const;

export type MemberSlotClassification =
  (typeof MemberSlotClassification)[number];

export function classifyMemberSlot(
  m: Member,
): MemberSlotClassification | undefined {
  if (m.kind === "attribute" && m.type === "bool") {
    return "bool-attr";
  } else if (m.kind === "attribute" && m.type === "int") {
    return "int-attr";
  } else if (m.kind === "attribute" && m.type === "TokenKind") {
    return "token-kind-attr";
  } else if (m.kind === "attribute" && m.type === "Identifier") {
    return "identifier-attr";
  } else if (m.kind === "attribute" && m.type.endsWith("Literal")) {
    return "literal-attr";
  } else if (m.kind === "token") {
    return "token";
  } else if (m.kind === "node") {
    return "node";
  } else if (m.kind === "node-list") {
    return "node-list";
  } else if (m.kind === "token-list") {
    return "token-list";
  } else {
    return undefined;
  }
}
export function getAllMemberSlotNames({ ast }: { ast: AST }): string[] {
  const allMemberSlotNameSet = new Set<string>();

  groupNodesByBaseType(ast).forEach((nodes) => {
    nodes.forEach(({ members }) => {
      members
        .filter((m) => classifyMemberSlot(m) !== undefined)
        .forEach(({ name }) => allMemberSlotNameSet.add(name));
    });
  });

  return Array.from(allMemberSlotNameSet).sort();
}
