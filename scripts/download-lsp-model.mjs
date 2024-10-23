#!/usr/bin/env zx

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

import { writeFile } from "node:fs/promises";
import * as path from "node:path";

$.verbose = true;

// Current working directory computed related to the script location
const workspaceDir = path.join(__dirname, "../");

async function main() {
  const url =
    "https://raw.githubusercontent.com/microsoft/vscode-languageserver-node/refs/heads/main/protocol/metaModel.json";

  const response = await fetch(url);

  const json = await response.json();

  const model = JSON.stringify(json, null, 2);

  const outputPath = path.join(
    workspaceDir,
    "packages/cxx-gen-lsp/metaModel.json"
  );

  echo`Writing model to ${outputPath}`;

  await writeFile(outputPath, model);
}

main().catch((e) => {
  if (e instanceof ProcessOutput) {
    if (e.stdout) console.log(e.stdout);
    if (e.stderr) console.error(e.stderr);
    process.exit(e.exitCode);
  } else {
    console.error(e);
    process.exit(1);
  }
});
