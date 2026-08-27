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

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import type { MetaModel } from "./MetaModel.ts";
import { orderTypeAliases } from "./gen_fwd_h.ts";

test("orders aliases from the latest meta model", async () => {
  const source = await readFile(new URL("../metaModel.json", import.meta.url), "utf8");
  const model = JSON.parse(source) as MetaModel;
  const names = orderTypeAliases(model).map((typeAlias) => typeAlias.name);

  const dependency = names.indexOf("DocumentDiagnosticReport");
  const dependent = names.indexOf("DocumentDiagnosticReportProgress");

  assert.notEqual(dependency, -1);
  assert.notEqual(dependent, -1);
  assert.ok(dependency < dependent);
});

test("reports aliases whose dependencies cannot be resolved", () => {
  const model: MetaModel = {
    metaData: {},
    enumerations: [],
    notifications: [],
    requests: [],
    structures: [],
    typeAliases: [
      {
        name: "MissingAlias",
        type: { kind: "reference", name: "MissingDependency" },
      },
    ],
  };

  assert.throws(
    () => orderTypeAliases(model),
    /Cannot resolve type aliases: MissingAlias: MissingDependency/,
  );
});
