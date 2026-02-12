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
import { gen_ast_recursive_visitor_ts } from "./gen_ast_recursive_visitor_ts.ts";
import { gen_ast_slot_cc } from "./gen_ast_slot_cc.ts";
import { gen_ast_slot_h } from "./get_ast_slot_h.ts";
import { gen_ast_ts } from "./gen_ast_ts.ts";
import { gen_ast_visitor_h } from "./gen_ast_visitor_h.ts";
import { gen_ast_visitor_cc } from "./gen_ast_visitor_cc.ts";
import { gen_ast_visitor_ts } from "./gen_ast_visitor_ts.ts";
import { parseAST } from "./parseAST.ts";
import { gen_ast_kind_ts } from "./gen_ast_kind_ts.ts";
import { gen_ast_fbs } from "./gen_ast_fbs.ts";
import { gen_ast_encoder_h } from "./gen_ast_encoder_h.ts";
import { gen_ast_encoder_cc } from "./gen_ast_encoder_cc.ts";
import { gen_ast_decoder_h } from "./gen_ast_decoder_h.ts";
import { gen_ast_decoder_cc } from "./gen_ast_decoder_cc.ts";
import { gen_ast_slot_ts } from "./gen_ast_slot_ts.ts";
import { gen_token_fwd_h } from "./gen_token_fwd_h.ts";
import { gen_tokenkind_ts } from "./gen_tokenkind_ts.ts";
import { gen_keywords_kwgen } from "./gen_keywords_kwgen.ts";
import { gen_c_keywords_kwgen } from "./gen_c_keywords_kwgen.ts";
import { gen_pp_keywords_kwgen } from "./gen_pp_keywords_kwgen.ts";
import { gen_ast_pretty_printer_h } from "./gen_ast_pretty_printer_h.ts";
import { gen_ast_pretty_printer_cc } from "./gen_ast_pretty_printer_cc.ts";
import { gen_builtins_h } from "./gen_builtins_h.ts";
import { gen_builtins_interp_h } from "./gen_builtins_interp_h.ts";

import * as fs from "fs";
import * as path from "path";
import * as process from "process";
import * as child_process from "child_process";

const outdir = process.cwd();

const fn = path.join(outdir, "src/parser/cxx/ast.h");

if (!fs.existsSync(fn)) throw new Error("File 'ast.h' not found");

const source = fs.readFileSync(fn).toString();
const ast = parseAST({ fn, source });

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
gen_ast_slot_h({ ast, output: path.join(outdir, "src/parser/cxx/ast_slot.h") });
gen_ast_slot_cc({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast_slot.cc"),
});

gen_ast_fbs({
  ast,
  output: path.join(outdir, "src/parser/cxx/ast.fbs"),
});

gen_ast_encoder_h({
  ast,
  output: path.join(outdir, "src/parser/cxx/private/ast_encoder.h"),
});

gen_ast_encoder_cc({
  ast,
  output: path.join(outdir, "src/parser/cxx/flatbuffers/ast_encoder.cc"),
});

gen_ast_decoder_h({
  ast,
  output: path.join(outdir, "src/parser/cxx/private/ast_decoder.h"),
});

gen_ast_decoder_cc({
  ast,
  output: path.join(outdir, "src/parser/cxx/flatbuffers/ast_decoder.cc"),
});

gen_token_fwd_h({
  output: path.join(outdir, "src/parser/cxx/token_fwd.h"),
});

gen_builtins_h({
  output: path.join(outdir, "src/parser/cxx/private/builtins-priv.h"),
});

gen_builtins_interp_h({
  output: path.join(
    outdir,
    "src/parser/cxx/private/builtins_interpreter-priv.h",
  ),
});

// js integration

gen_ast_ts({
  ast,
  output: path.join(outdir, "packages/cxx-frontend/src/AST.ts"),
});
gen_ast_visitor_ts({
  ast,
  output: path.join(outdir, "packages/cxx-frontend/src/ASTVisitor.ts"),
});
gen_ast_recursive_visitor_ts({
  ast,
  output: path.join(outdir, "packages/cxx-frontend/src/RecursiveASTVisitor.ts"),
});
gen_ast_kind_ts({
  ast,
  output: path.join(outdir, "packages/cxx-frontend/src/ASTKind.ts"),
});
gen_ast_slot_ts({
  ast,
  output: path.join(outdir, "packages/cxx-frontend/src/ASTSlot.ts"),
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

child_process.execSync("clang-format -i *.h *.cc", {
  cwd: path.join(outdir, "src/parser/cxx"),
});

child_process.execSync("clang-format -i *.cc", {
  cwd: path.join(outdir, "src/parser/cxx/flatbuffers"),
});

child_process.execSync("clang-format -i *.h", {
  cwd: path.join(outdir, "src/parser/cxx/private"),
});

child_process.execSync("clang-format -i *.h *.cc", {
  cwd: path.join(outdir, "src/frontend/cxx"),
});
