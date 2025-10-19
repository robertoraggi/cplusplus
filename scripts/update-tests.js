#!/usr/bin/env zx

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

import "zx/globals";

const workspacePath = path.join(__dirname, "../");
const unitTestsPath = path.join(workspacePath, "tests/unit_tests/ast");
const cxx = path.join(workspacePath, "build/src/frontend/cxx");

const unitTestFilePaths = await glob(`${unitTestsPath}/**/*.{cc,c}`);

async function readLines(path) {
  const unitTestContent = await fs.readFile(path, "utf-8");
  const lines = unitTestContent.split("\n");
  return lines;
}

async function strip(unitTestPath) {
  const lines = await readLines(unitTestPath);
  const index = lines.lastIndexOf("// clang-format off");
  if (index == -1) return;
  const unitTestUpdatedContent = lines.slice(0, index).join("\n");
  return fs.writeFile(unitTestPath, unitTestUpdatedContent + "\n");
}

async function updateTest(unitTestPath) {
  const source = await fs.readFile(unitTestPath, "utf-8");

  const sourceLines = source.split("\n");

  const index = sourceLines.findIndex((line) => line.startsWith("// RUN:"));

  const fcheck = sourceLines[index]?.includes("-fcheck") ? "-fcheck" : "";

  const lang = unitTestPath.endsWith(".c") ? "-xc" : "";

  const ast =
    await $`${cxx} -verify -ast-dump ${unitTestPath} ${fcheck} ${lang}`.quiet();

  const lines = ast.stdout.split("\n");

  const out = lines.filter(Boolean).flatMap((line, index) => {
    if (index == 0) {
      return ["// clang-format off", `//      CHECK:${line}`];
    }
    return `// CHECK-NEXT:${line}`;
  });

  await fs.appendFile(unitTestPath, out.join("\n") + "\n");
}

for (const unitTestPath of unitTestFilePaths) {
  await strip(unitTestPath);
  await updateTest(unitTestPath);
}
