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
import { AST, Member } from "./parseAST.js";

export enum MemberSlotClassification {
  BoolAttribute,
  TokenKindAttribute,
  IdentifierAttribute,
  LiteralAttribute,
  Token,
  Node,
  NodeList,
  TokenList,
}
export function classifyMemberSlot(
  m: Member
): MemberSlotClassification | undefined {
  if (m.kind === "attribute" && m.type === "bool") {
    return MemberSlotClassification.BoolAttribute;
  } else if (m.kind === "attribute" && m.type === "TokenKind") {
    return MemberSlotClassification.TokenKindAttribute;
  } else if (m.kind === "attribute" && m.type === "Identifier") {
    return MemberSlotClassification.IdentifierAttribute;
  } else if (m.kind === "attribute" && m.type.endsWith("Literal")) {
    return MemberSlotClassification.LiteralAttribute;
  } else if (m.kind === "token") {
    return MemberSlotClassification.Token;
  } else if (m.kind === "node") {
    return MemberSlotClassification.Node;
  } else if (m.kind === "node-list") {
    return MemberSlotClassification.NodeList;
  } else if (m.kind === "token-list") {
    return MemberSlotClassification.TokenList;
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
