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

import { Parser, AST, ASTKind, ASTSlot } from "cxx-frontend";
import { readFile } from "fs/promises";
import { fileURLToPath } from "node:url";

const source = `
auto twice(auto x) { return x + x; }

int main() {
  return twice(21) - 42;
}
`;

async function main() {
  const wasmBinaryUrl = import.meta.resolve("cxx-frontend/wasm");
  const wasmBinaryFile = fileURLToPath(wasmBinaryUrl);
  const wasmBinary = await readFile(wasmBinaryFile);

  // initialize the parser
  await Parser.init({ wasm: wasmBinary });

  const parser = new Parser({ source, path: "source.cc" });

  const ast = await parser.parse();

  const diagnostics = parser.getDiagnostics();

  if (diagnostics.length > 0) {
    console.log("diagnostics", diagnostics);
  }

  console.log("==== AST ====");

  for (const { node, slot, depth } of ast?.walk().preVisit() ?? []) {
    if (!(node instanceof AST)) continue;
    const ind = " ".repeat(depth * 2);
    const kind = ASTKind[node.getKind()];
    const member = slot !== undefined ? `${ASTSlot[slot]}: ` : "";
    console.log(`${ind}- ${member}${kind}`);
  }

  const ir = await parser.emitIR();

  console.log("==== IR ====");
  console.log(ir);

  parser.dispose();
}

await main();
