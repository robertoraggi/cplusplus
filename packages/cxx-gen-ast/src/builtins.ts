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
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function isInterpreterHook(
  eval_: string | BuiltinEval | undefined,
): eval_ is string {
  return typeof eval_ === "string";
}

export interface BuiltinEval {
  fn: string;
  args: string[];
  ret: string;
  cxx23?: boolean;
}

export interface BuiltinDef {
  name: string;
  prototype?: string | string[];
  constexpr: boolean;
  consteval?: boolean;
  noexcept?: boolean;
  noreturn?: boolean;
  libcall?: boolean;
  eval?: string | BuiltinEval;
  typeCheck?: string;
  codegen?: string;
}

const builtinsPath = path.join(__dirname, "builtins.json");

export const BUILTINS: BuiltinDef[] = JSON.parse(
  fs.readFileSync(builtinsPath, "utf-8"),
);

export const BUILTIN_FUNCTION_DEFS: BuiltinDef[] = BUILTINS.filter(
  (b) => b.prototype,
);

export const BUILTIN_NAMES: string[] = BUILTIN_FUNCTION_DEFS.map(
  (b) => b.name,
).sort();
