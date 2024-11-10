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

const { Parser, AST, ASTKind, ASTSlot } = require("cxx-frontend");
const { readFileSync } = require("fs");

const source = `
#include <cstdio>

template <typename T>
concept CanAdd = requires(T n) {
  n + n;
};

auto twice(CanAdd auto n) {
  return n + n;
}

const char* str = "hello";

int main() {
  return twice(2);
}
`;

async function main() {
  const wasmBinaryFile = require.resolve("cxx-frontend/dist/wasm/cxx-js.wasm");
  const wasm = readFileSync(wasmBinaryFile);
  await Parser.init({ wasm });

  const parser = new Parser({
    source,
    path: "source.cc",
  });

  await parser.parse();

  const diagnostics = parser.getDiagnostics();

  if (diagnostics.length > 0) {
    console.log("diagnostics", diagnostics);
  }

  const ast = parser.getAST();

  for (const { node, slot, depth } of ast?.walk().preVisit() ?? []) {
    if ((!node) instanceof AST) continue;
    const ind = " ".repeat(depth * 2);
    const kind = ASTKind[node.getKind()];
    const member = slot !== undefined ? `${ASTSlot[slot]}: ` : "";
    console.log(`${ind}- ${member}${kind}`);
  }

  parser.dispose();
}

main().catch(console.error);
