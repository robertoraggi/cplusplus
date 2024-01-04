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

import wasmBinaryUrl from "cxx-frontend/dist/wasm/cxx-js.wasm?url";
import { Parser, AST, ASTKind, TokenKind } from "cxx-frontend";

const source = `
auto main() -> int {
  auto incr = [](auto x) { return x + 1; };
  return incr(-1);
}
`;

interface Attributes {
  getLiteral(): string;
  getIdentifier(): string;
  getOp(): TokenKind;
}

async function main() {
  const response = await fetch(wasmBinaryUrl);
  const wasm = await response.arrayBuffer();
  await Parser.init({ wasm });

  const parser = new Parser({
    path: "source.cc",
    source,
  });

  parser.parse();

  const rows: string[] = [];

  const ast = parser.getAST();

  ast?.walk().preVisit(({ node, depth }) => {
    if (!(node instanceof AST)) {
      return;
    }

    const nodeWithAttributes: AST & Partial<Attributes> = node;
    const indent = "  ".repeat(depth);
    const nodeKind = ASTKind[node.getKind()];

    let description = `${indent}${nodeKind}`;

    const id = nodeWithAttributes?.getIdentifier?.();
    if (id !== undefined) {
      description += ` (${id})`;
    }

    const literal = nodeWithAttributes?.getLiteral?.();
    if (literal !== undefined) {
      description += ` (${literal})`;
    }

    const op = nodeWithAttributes?.getOp?.();
    if (op !== undefined) {
      description += ` (${TokenKind[op]})`;
    }

    rows.push(description);
  });

  parser.dispose();

  const app = document.querySelector<HTMLDivElement>("#app")!;

  const sourceOutput = document.createElement("pre");
  sourceOutput.style.borderStyle = "solid";
  sourceOutput.innerText = source;
  app.appendChild(sourceOutput);

  const astOutput = document.createElement("pre");
  astOutput.style.borderStyle = "solid";
  astOutput.innerText = rows.join("\n");
  app.appendChild(astOutput);
}

main();
