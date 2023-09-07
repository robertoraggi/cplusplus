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

import { FixedSizeList } from "react-window";
import AutoSizer from "react-virtualized-auto-sizer";

import { AST, ASTKind, Parser, TokenKind } from "cxx-frontend";

function hasOp(node: any): node is AST & { getOp(): TokenKind } {
  return typeof node.getOp === "function";
}

function hasIdentifier(node: any): node is AST & { getIdentifier(): string } {
  return typeof node.getIdentifier === "function";
}

function hasNamespaceName(
  node: any
): node is AST & { getNamespaceName(): string } {
  return typeof node.getNamespaceName === "function";
}

function hasLiteral(node: any): node is AST & { getLiteral(): string } {
  return typeof node.getLiteral === "function";
}

export function SyntaxTree({ parser }: { parser: Parser | null }) {
  const nodes: string[] = [];

  const ast = parser?.getAST();

  ast?.walk().preVisit((node, level) => {
    if (!(node instanceof AST)) return;

    const indent = " ".repeat(level * 4);
    const kind = ASTKind[node.getKind()];

    let extra = "";
    if (hasNamespaceName(node)) extra += ` (${node.getNamespaceName()})`;
    if (hasIdentifier(node)) extra += ` (${node.getIdentifier()})`;
    if (hasLiteral(node)) extra += ` (${node.getLiteral()})`;
    if (hasOp(node)) extra += ` (${TokenKind[node.getOp()]})`;

    nodes.push(`${indent}- ${kind}${extra}`);
  });

  function Item({ index, style }: { index: number; style: any }) {
    return <div style={{ ...style, whiteSpace: "pre" }}>{nodes[index]}</div>;
  }

  return (
    <div style={{ flex: 1 }}>
      <AutoSizer>
        {({ height, width }) => (
          <FixedSizeList
            height={height}
            width={width}
            itemCount={nodes.length}
            itemSize={20}
          >
            {Item}
          </FixedSizeList>
        )}
      </AutoSizer>
    </div>
  );
}
