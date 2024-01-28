#!/usr/bin/env node

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

import * as fs from "fs";
import * as path from "path";
import * as process from "process";
import * as child_process from "child_process";
import { parseAST } from "./parseAST.js";
import { new_ast_op_h } from "./new_ast_op_h.js";
import { new_ast_op_cc } from "./new_ast_op_cc.js";

import { hideBin } from "yargs/helpers";
import yargs from "yargs";

const toSnakeName = (name: string) => {
  const r = name
    .replace("AST", "Ast")
    .replace(/([A-Z]+)/g, "_$1")
    .toLocaleLowerCase();
  return r.startsWith("_") ? r.slice(1) : r;
};

interface MainArgs {
  name: string;
}

function main(args: MainArgs) {
  const opName = args.name;
  const baseFileName = toSnakeName(opName);

  const outdir = process.cwd();

  const fn = path.join(outdir, "src/parser/cxx/ast.h");

  if (!fs.existsSync(fn)) throw new Error("File 'ast.h' not found");

  const source = fs.readFileSync(fn).toString();
  const ast = parseAST({ fn, source });

  new_ast_op_h({
    ast,
    opName,
    output: path.join(outdir, `src/parser/cxx/${baseFileName}.h`),
  });

  new_ast_op_cc({
    ast,
    opName,
    opHeader: `${baseFileName}.h`,
    output: path.join(outdir, `src/parser/cxx/${baseFileName}.cc`),
  });

  child_process.execSync(
    `clang-format -i ${baseFileName}.h ${baseFileName}.cc`,
    {
      cwd: path.join(outdir, "src/parser/cxx"),
    }
  );
}

const _ = yargs(hideBin(process.argv))
  .option("name", {
    alias: "n",
    type: "string",
    default: "ASTInterpreter",
    description: "Name of the new operation",
  })
  .command("$0", "generate a new op", () => {}, main).argv;
