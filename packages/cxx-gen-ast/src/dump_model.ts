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
import { fileURLToPath } from "node:url";
import { dumpSemanticModel, modelSnapshotPath } from "./modelRefresh.ts";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const args = new Set(process.argv.slice(2));
for (const arg of args)
  if (!["--write", "--check"].includes(arg))
    throw new Error(`unknown option ${arg}`);

const text = await dumpSemanticModel(root);
const output = modelSnapshotPath(root);

if (args.has("--check")) {
  if (fs.readFileSync(output, "utf8") !== text)
    throw new Error(
      "semantic-model.json is stale; run npm run cxx-dump-model -- --write",
    );
} else if (args.has("--write")) fs.writeFileSync(output, text);
else process.stdout.write(text);
