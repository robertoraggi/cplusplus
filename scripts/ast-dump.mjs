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

import "zx/globals";

// Current working directory computed related to the script location
const workspaceDir = path.join(__dirname, "../");

// Default path to the build directory
const buildDir = argv["build-dir"] ?? path.join(workspaceDir, "build");

// Paths to cxx and cmake from the command line
const cxx = argv.cxx ?? path.join(buildDir, "src/frontend/cxx");
const cmake = argv.cmake ?? (await which("cmake"));

// Path to the AST dump output directory
const astOutputDir = path.join(buildDir, "ast");

// Path to the source file to parse
const sourceFile =
  argv._[1] ?? path.join(workspaceDir, "tests/manual/source.cc");

// Path to the AST dump output file
const astFile = path.join(astOutputDir, path.basename(sourceFile) + ".ast");

// Path to the AST dump baseline file
const baselineAstFile = path.join(
  astOutputDir,
  path.basename(sourceFile) + ".ast.base",
);

async function main() {
  // Create the ast output directory if it doesn't exist
  await spinner("Setting up...", () =>
    fs.mkdir(astOutputDir, { recursive: true }),
  );

  // Build the project
  const p = await spinner("Building", () => $`${cmake} --build ${buildDir}`);
  if (p.exitCode !== 0) {
    throw new Error("Build failed");
  }

  // Run the AST dump
  await spinner(
    "Create AST",
    () => $`${cxx} -ast-dump ${sourceFile} > ${astFile}`,
  );

  const hasBaseline = await fs.exists(baselineAstFile);

  if (!hasBaseline) {
    // If there is no baseline, create one
    await fs.copyFile(astFile, baselineAstFile);
  } else {
    // Otherwise, compare the AST dump with the baseline
    await $`diff ${astFile} ${baselineAstFile}`;
  }
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
