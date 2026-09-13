#!/usr/bin/env node

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

import * as child_process from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import * as process from "node:process";

import { loadModel } from "./parseModel.ts";
import { PlanBuilder } from "./semanticPlan.ts";
import { gen_semantic_codec } from "./gen_semantic_codec.ts";
import { gen_semantic_model_h } from "./gen_semantic_model_h.ts";

const outdir = process.cwd();

const index = loadModel(fs.readFileSync(path.join(outdir, "packages/cxx-gen-ast/semantic-model.json"), "utf8"));
const plan = new PlanBuilder(index).build();

if (plan.diagnostics.length > 0) {
  const unique = [...new Set(plan.diagnostics)];
  console.error(
    `the persistence model is incomplete (${unique.length} unresolved):`,
  );
  for (const diagnostic of unique) console.error(`  - ${diagnostic}`);
  console.error(
    "\nclassify the field in packages/cxx-gen-ast/src/semanticClassification.ts",
  );
  process.exit(1);
}

gen_semantic_model_h({
  plan,
  output: path.join(outdir, "src/parser/cxx/private/semantic_model.h"),
});

gen_semantic_codec({
  plan,
  headerOutput: path.join(outdir, "src/parser/cxx/private/semantic_codec.h"),
  sourceOutput: path.join(outdir, "src/parser/cxx/semantic_codec.cc"),
});

const counts: Record<string, number> = {};
for (const field of plan.report)
  counts[field.cls] = (counts[field.cls] ?? 0) + 1;

console.log(
  `semantic model: ${plan.names.length} names, ${plan.types.length} types, ` +
    `${plan.symbols.length} symbols, ${plan.nodes.length} AST nodes, ` +
    `${plan.structs.length} structs`,
);
console.log(
  `classification: ${Object.entries(counts)
    .sort()
    .map(([key, value]) => `${key}=${value}`)
    .join(" ")}`,
);

child_process.spawnSync(
  "clang-format",
  [
    "-i",
    "src/parser/cxx/private/semantic_model.h",
    "src/parser/cxx/private/semantic_codec.h",
    "src/parser/cxx/semantic_codec.cc",
  ],
  { cwd: outdir, stdio: "inherit" },
);
