#!/usr/bin/env zx

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

import { $, fs, which } from "zx";
import { fileURLToPath } from "node:url";
import os from "node:os";
import path from "node:path";

$.verbose = true;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const workspacePath = path.join(__dirname, "../");

const wasm32WasiIncludeDir = path.join(
  workspacePath,
  "build.em/src/lib/wasi-sysroot/include/wasm32-wasip1",
);

const cxxIncludeDir = path.join(workspacePath, "build.em/src/lib/cxx/include");

const wasm32WasiLibraryDir = path.join(
  workspacePath,
  "build.em/src/lib/wasi-sysroot/lib/wasm32-wasip1",
);

const libraryFiles = [
  "crt1.o",
  "libc.a",
  "libc++.a",
  "libc++abi.a",
  "libclang_rt.builtins-wasm32.a",
];

const outputZip = path.join(
  workspacePath,
  "packages/cxx-playground/public/sysroot.zip",
);

async function findStrip() {
  const strip = await which("emstrip", { nothrow: true });
  if (!strip) {
    console.warn("emstrip not found, the libraries keep their debug sections");
  }
  return strip;
}

async function main() {
  for (const dir of [
    wasm32WasiIncludeDir,
    cxxIncludeDir,
    wasm32WasiLibraryDir,
  ]) {
    if (!(await fs.pathExists(dir))) {
      throw new Error(
        `${dir} does not exist, run "npm run build:emscripten" first`,
      );
    }
  }

  const strip = await findStrip();
  const stageDir = await fs.mkdtemp(
    path.join(os.tmpdir(), "cxx-playground-sysroot-"),
  );
  const stageLibraryDir = path.join(stageDir, "lib/wasm32-wasip1");

  try {
    await fs.ensureDir(stageLibraryDir);
    for (const file of libraryFiles) {
      const staged = path.join(stageLibraryDir, file);
      await fs.copy(path.join(wasm32WasiLibraryDir, file), staged);
      if (strip) await $`${strip} --strip-debug ${staged}`;
    }
    await fs.copy(
      wasm32WasiIncludeDir,
      path.join(stageDir, "include/wasm32-wasip1"),
    );
    await fs.copy(cxxIncludeDir, path.join(stageDir, "lib/cxx/include"));

    await fs.remove(outputZip);
    await fs.ensureDir(path.dirname(outputZip));

    $.cwd = stageDir;
    await $`zip -r -X -q ${outputZip} include lib`;
  } finally {
    await fs.remove(stageDir);
  }
}

await main();
