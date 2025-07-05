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

/* eslint-disable @typescript-eslint/no-explicit-any */

import { FixedSizeList } from "react-window";
import { AST, ASTKind, ASTSlot, Parser, TokenKind } from "cxx-frontend";
import {
  type CSSProperties,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import AutoSizer from "react-virtualized-auto-sizer";
import clsx from "clsx";

interface SyntaxTreeProps {
  parser: Parser | null;
  cursorPosition?: { line: number; column: number };
  onNodeSelected?: (node: AST) => void;
}

interface SyntaxTreeNode {
  description: string;
  handle: number;
  level: number;
  slot?: ASTSlot;
}

export function SyntaxTree({
  parser,
  cursorPosition,
  onNodeSelected,
}: SyntaxTreeProps) {
  const listRef = useRef<FixedSizeList>(null);
  const [selectedNodeHandle, setSelectedNodeHandle] = useState(0);
  const [nodes, setNodes] = useState<SyntaxTreeNode[]>([]);

  const onNodeSelectedRef = useRef(onNodeSelected);

  useLayoutEffect(() => {
    onNodeSelectedRef.current = onNodeSelected;
  }, [onNodeSelected]);

  useEffect(() => {
    const update = async () => {
      const nodes: SyntaxTreeNode[] = [];

      const visitor = parser?.getAST()?.walk().preVisit();

      if (visitor) {
        let count = 0;

        for (const { node, slot, depth: level } of visitor) {
          if (!(node instanceof AST)) continue;

          ++count;

          if (count % 1000 === 0) {
            await new Promise((resolve) => setTimeout(resolve, 0));
          }

          const kind = ASTKind[node.getKind()];

          let extra = "";
          if (hasIdentifier(node)) extra += ` (${node.getIdentifier()})`;
          if (hasLiteral(node)) extra += ` (${node.getLiteral()})`;
          if (hasOp(node)) extra += ` (${TokenKind[node.getOp()]})`;
          if (hasAccessOp(node)) extra += ` (${TokenKind[node.getAccessOp()]})`;
          if (hasSpecifier(node))
            extra += ` (${TokenKind[node.getSpecifier()]})`;
          if (hasAccessSpecifier(node))
            extra += ` (${TokenKind[node.getAccessSpecifier()]})`;

          const member = slot !== undefined ? `${ASTSlot[slot]}: ` : "";

          const description = `${member}${kind}${extra}`;
          const handle = node.getHandle();

          nodes.push({ description, slot, handle, level });
        }
      }

      setNodes(nodes);
    };

    update();
  }, [parser]);

  useEffect(() => {
    const ast = parser?.getAST();

    if (ast && cursorPosition) {
      const { line, column } = cursorPosition;
      const node = ast ? findNodeAt(ast, line, column + 1) : null;
      const selectedNodeHandle = node?.getHandle() ?? 0;
      setSelectedNodeHandle(selectedNodeHandle);

      const index = nodes.findIndex(
        (node) => node.handle === selectedNodeHandle,
      );

      if (index != -1) {
        listRef.current?.scrollToItem(index, "smart");
      }
    }
  }, [parser, nodes, cursorPosition]);

  function Item({ index, style }: { index: number; style: CSSProperties }) {
    const { description, level, handle } = nodes[index];
    const indent = " ".repeat(level * 4);
    const isSelected = selectedNodeHandle === handle;

    return (
      <div className="whitespace-pre" style={style}>
        {indent}-{" "}
        <a
          className={clsx("cursor-default p-0.5 font-[monospace] text-xs", {
            "bg-primary text-primary-foreground dark:bg-[#264F78] dark:text-[#D4D4D4]":
              isSelected,
          })}
        >
          {description}
        </a>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-background text-foreground dark:bg-[#1E1E1E] dark:text-[#D4D4D4]">
      <AutoSizer>
        {({ height, width }) => (
          <FixedSizeList
            ref={listRef}
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

function isWithin(node: AST, line: number, column: number): boolean {
  const start = node.getStartLocation();
  if (!start) return false;

  const end = node.getEndLocation();
  if (!end) return false;

  const { startLine, startColumn } = start;
  const { endLine, endColumn } = end;

  if (line < startLine) return false;
  if (line > endLine) return false;
  if (line === startLine && column < startColumn) return false;
  if (line === endLine && column > endColumn) return false;

  return true;
}

function findNodeAt(root: AST, line: number, column: number): AST | null {
  if (!isWithin(root, line, column)) {
    return null;
  }

  const cursor = root.walk();

  cursor.gotoFirstChild();

  do {
    const childNode = cursor.node;

    if (childNode instanceof AST) {
      const result = findNodeAt(childNode, line, column);

      if (result) {
        return result;
      }
    }
  } while (cursor.gotoNextSibling());

  return root;
}

function hasOp(node: any): node is AST & { getOp(): TokenKind } {
  return typeof node.getOp === "function" && node.getOp();
}

function hasAccessOp(node: any): node is AST & { getAccessOp(): TokenKind } {
  return typeof node.getAccessOp === "function" && node.getAccessOp();
}

function hasAccessSpecifier(
  node: any,
): node is AST & { getAccessSpecifier(): TokenKind } {
  return (
    typeof node.getAccessSpecifier === "function" && node.getAccessSpecifier()
  );
}

function hasSpecifier(node: any): node is AST & { getSpecifier(): TokenKind } {
  return typeof node.getSpecifier === "function" && node.getSpecifier();
}

function hasIdentifier(node: any): node is AST & { getIdentifier(): string } {
  return typeof node.getIdentifier === "function" && node.getIdentifier();
}

function hasLiteral(node: any): node is AST & { getLiteral(): string } {
  return typeof node.getLiteral === "function" && node.getLiteral();
}
