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

import { gen_ast_cc } from "./gen_ast_cc.ts";
import { gen_ast_printer_cc } from "./gen_ast_printer_cc.ts";
import { gen_ast_printer_h } from "./gen_ast_printer_h.ts";
import { gen_ast_fwd_h } from "./gen_ast_fwd_h.ts";
import { gen_ast_h } from "./gen_ast_h.ts";
import { gen_ast_kind_h } from "./gen_ast_kind_h.ts";
import { gen_ast_visitor_h } from "./gen_ast_visitor_h.ts";
import { gen_ast_visitor_cc } from "./gen_ast_visitor_cc.ts";
import { gen_reflection } from "./gen_reflection.ts";
import { gen_traverse_ts } from "./gen_traverse_ts.ts";
import { loadModel } from "./parseModel.ts";
import {
  missingFrontend,
  modelSnapshotPath,
  refreshSemanticModel,
} from "./modelRefresh.ts";
import { parseAST } from "./parseAST.ts";
import { gen_token_fwd_h } from "./gen_token_fwd_h.ts";
import { gen_tokenkind_ts } from "./gen_tokenkind_ts.ts";
import { gen_keywords_kwgen } from "./gen_keywords_kwgen.ts";
import { gen_c_keywords_kwgen } from "./gen_c_keywords_kwgen.ts";
import { gen_pp_keywords_kwgen } from "./gen_pp_keywords_kwgen.ts";
import { gen_builtin_function_kwgen } from "./gen_builtin_function_kwgen.ts";
import { gen_ast_pretty_printer_h } from "./gen_ast_pretty_printer_h.ts";
import { gen_ast_pretty_printer_cc } from "./gen_ast_pretty_printer_cc.ts";
import { gen_builtins_h } from "./gen_builtins_h.ts";
import { gen_builtins_signatures_h } from "./gen_builtins_signatures_h.ts";
import { gen_builtins_interp_h } from "./gen_builtins_interp_h.ts";
import { gen_builtins_typechecker_h } from "./gen_builtins_typechecker_h.ts";
import { gen_builtins_codegen_h } from "./gen_builtins_codegen_h.ts";
import { gen_emitter_delegate_h } from "./gen_emitter_delegate_h.ts";
import { gen_emitter_ts } from "./gen_emitter_ts.ts";

import * as fs from "node:fs";
import * as path from "node:path";
import * as process from "node:process";
import * as child_process from "child_process";

const outdir = process.cwd();

const snapshot = modelSnapshotPath(outdir);
const missing = process.argv.includes("--no-refresh")
  ? "--no-refresh"
  : missingFrontend(outdir);

if (!missing) {
  if (await refreshSemanticModel(outdir))
    console.error("cxx-gen-ast: semantic-model.json refreshed");
} else if (fs.existsSync(snapshot)) {
  console.error(
    `cxx-gen-ast: ${missing === "--no-refresh" ? "--no-refresh was given" : `${path.relative(outdir, missing)} is missing`}, keeping the existing semantic-model.json`,
  );
} else {
  throw new Error(
    `${path.relative(outdir, snapshot)} does not exist and it cannot be generated ` +
      `(${missing === "--no-refresh" ? "--no-refresh was given" : `${path.relative(outdir, missing)} is missing`}); ` +
      "run npm run build:cxx-frontend first",
  );
}

const model = loadModel(fs.readFileSync(snapshot, "utf8"));
const ast = parseAST(model);
gen_reflection(model, outdir);
gen_traverse_ts({ ast, index: model, root: outdir });

gen_ast_fwd_h({ ast, output: path.join(outdir, "src/parser/cxx/ast_fwd.h") });
gen_ast_h({ ast, output: path.join(outdir, "src/parser/cxx/ast.h") });
gen_ast_cc({ ast, output: path.join(outdir, "src/parser/cxx/ast.cc") });
gen_ast_visitor_h({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast_visitor.h"),
});
gen_ast_visitor_cc({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast_visitor.cc"),
});
gen_ast_printer_h({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast_printer.h"),
});
gen_ast_printer_cc({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast_printer.cc"),
});
gen_ast_pretty_printer_h({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast_pretty_printer.h"),
});
gen_ast_pretty_printer_cc({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast_pretty_printer.cc"),
});
gen_ast_kind_h({ ast, output: path.join(outdir, "src/parser/cxx/ast_kind.h") });

gen_token_fwd_h({
  output: path.join(outdir, "src/parser/cxx/token_fwd.h"),
});

gen_builtins_h({
  output: path.join(outdir, "src/parser/cxx/private/builtins-priv.h"),
});

gen_builtins_signatures_h({
  output: path.join(
    outdir,
    "src/parser/cxx/private/builtins_signatures-priv.h",
  ),
});

gen_builtins_interp_h({
  output: path.join(
    outdir,
    "src/parser/cxx/private/builtins_interpreter-priv.h",
  ),
});

gen_builtins_typechecker_h({
  output: path.join(
    outdir,
    "src/parser/cxx/private/builtins_typechecker-priv.h",
  ),
});

gen_builtins_codegen_h({
  output: path.join(outdir, "src/codegen/cxx/codegen/builtins_codegen-priv.h"),
});

// js integration

gen_emitter_delegate_h({
  model,
  output: path.join(outdir, "src/js/cxx/emitter_delegate-priv.h"),
});
gen_emitter_ts({
  model,
  output: path.join(outdir, "packages/cxx-frontend/src/Emitter.ts"),
});
gen_tokenkind_ts({
  output: path.join(outdir, "packages/cxx-frontend/src/TokenKind.ts"),
});
gen_keywords_kwgen({
  output: path.join(outdir, "src/parser/cxx/private/keywords-priv.h"),
});
gen_c_keywords_kwgen({
  output: path.join(outdir, "src/parser/cxx/private/c_keywords-priv.h"),
});
gen_pp_keywords_kwgen({
  output: path.join(outdir, "src/parser/cxx/private/pp_directives-priv.h"),
});
gen_builtin_function_kwgen({
  output: path.join(
    outdir,
    "src/parser/cxx/private/builtin_function_keywords-priv.h",
  ),
});

child_process.execSync("clang-format -i *.h *.cc", {
  cwd: path.join(outdir, "src/parser/cxx"),
});

child_process.execSync("clang-format -i *.h", {
  cwd: path.join(outdir, "src/parser/cxx/private"),
});

child_process.execSync("clang-format -i *.h *.cc", {
  cwd: path.join(outdir, "src/frontend/cxx"),
});

child_process.execSync("clang-format -i emitter_delegate-priv.h", {
  cwd: path.join(outdir, "src/js/cxx"),
});

child_process.execFileSync(
  path.join(outdir, "node_modules/.bin/prettier"),
  ["--write", path.join(outdir, "packages/cxx-frontend/src/Emitter.ts")],
  { stdio: "ignore" },
);
