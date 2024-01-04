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

const workspacePath = path.join(__dirname, "../");

async function main() {
  const gcc = argv.cc ?? "g++";
  const std = argv.std ?? "c++20";

  const predefinedMacros = String(
    await $`${gcc} -E -dM -x c++ -std=${std} - < /dev/null`.quiet()
  );

  const out = [];
  const emit = (s) => out.push(s);

  const toolchain = "GCCLinuxToolchain";

  emit(`void ${toolchain}::addPredefinedMacros() {`);

  predefinedMacros.split("\n").forEach((line) => {
    const m = /^#define (\w+(?:\([^)]+\))?) (.*)/.exec(line);
    if (!m) return;
    const macro = m[1];
    const value = m[2].replace(/"/g, '\\"');
    emit(`  defineMacro("${macro}", "${value}");`);
  });

  emit(`}`);

  const text = out.join("\n") + "\n";

  if (argv.output) {
    await fs.writeFile(argv.output, text);
  } else {
    console.log(text);
  }
}

main().catch((e) => {
  console.error("failed", e);
  process.exit(1);
});
