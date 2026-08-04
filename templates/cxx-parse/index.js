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

import { loadCxx, Parser, AST, ASTKind, ASTSlot } from "cxx-frontend";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const sourcePath = fileURLToPath(new URL("source.cc", import.meta.url));

async function main() {
  const { positionals } = parseArgs({
    allowPositionals: true,
  });

  const wasmBinaryUrl = import.meta.resolve("cxx-frontend/wasm");
  const wasmBinaryFile = fileURLToPath(wasmBinaryUrl);
  const wasmBinary = await readFile(wasmBinaryFile);

  await loadCxx({ wasm: wasmBinary });

  for (const path of positionals) {
    const source = await readFile(path, "utf8");

    await using parser = await Parser.parse({
      source,
      path,
      exists: (fn) => existsSync(fn),
      readFile: async (fn) => await readFile(fn, "utf8"),
    });

    if (parser.diagnostics.length > 0) {
      console.log("diagnostics", parser.diagnostics);
    }
  }
}

await main();
