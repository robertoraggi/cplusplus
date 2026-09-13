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

import * as fs from "node:fs";
import * as path from "node:path";
import { snapshotModel, serializeModel } from "./modelSnapshot.ts";

export const modelInputPath = (root: string) =>
  path.join(root, "src/parser/cxx/private/model_inputs.h");

export const modelSnapshotPath = (root: string) =>
  path.join(root, "packages/cxx-gen-ast/semantic-model.json");

const frontendWasmPath = (root: string) =>
  path.join(root, "packages/cxx-frontend/dist/wasm/cxx-js.wasm");

const appdirPath = (root: string) => path.join(root, "build.em/src/js");

const sysrootPath = (root: string) =>
  process.env.CXX_SYSROOT ?? path.join(root, "build.em/src/lib/wasi-sysroot");

export function missingFrontend(root: string): string | undefined {
  for (const required of [
    frontendWasmPath(root),
    appdirPath(root),
    sysrootPath(root),
  ])
    if (!fs.existsSync(required)) return required;
}

export async function dumpSemanticModel(root: string): Promise<string> {
  const missing = missingFrontend(root);
  if (missing)
    throw new Error(
      `${missing} is missing; build it with npm run build:cxx-frontend`,
    );

  const { loadCxx, Parser } = await import("cxx-frontend");
  const { dumpModel } = await import("./modelDumper.ts");

  const input = modelInputPath(root);
  await loadCxx({ wasm: fs.readFileSync(frontendWasmPath(root)) });

  const parser = await Parser.parse({
    appdir: appdirPath(root),
    sysroot: sysrootPath(root),
    path: input,
    source: fs.readFileSync(input, "utf8"),
    includePaths: [path.join(root, "src/parser"), path.join(root, "src/codegen")],
    exists: fs.existsSync,
    readFile: async (file: string) => {
      try {
        return await fs.promises.readFile(file, "utf8");
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
        throw error;
      }
    },
  });

  try {
    const errors = parser.diagnostics.filter(
      (d) => d.severity === "error" || d.severity === "fatal",
    );
    if (errors.length) throw new Error(JSON.stringify(errors, null, 2));
    return serializeModel(snapshotModel(dumpModel(parser), root));
  } finally {
    parser.dispose();
  }
}

export async function refreshSemanticModel(root: string): Promise<boolean> {
  const text = await dumpSemanticModel(root);
  const output = modelSnapshotPath(root);
  if (fs.existsSync(output) && fs.readFileSync(output, "utf8") === text)
    return false;
  fs.writeFileSync(output, text);
  return true;
}
