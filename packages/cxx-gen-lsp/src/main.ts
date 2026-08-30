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

import * as process from "node:process";
import * as child_process from "node:child_process";
import { parseArgs } from "node:util";
import { readFile, mkdir } from "node:fs/promises";
import type { MetaModel } from "./MetaModel.ts";
import { gen_enums_h } from "./gen_enums_h.ts";
import { gen_enums_cc } from "./gen_enums_cc.ts";
import { gen_fwd_h } from "./gen_fwd_h.ts";
import { gen_types_h } from "./gen_types_h.ts";
import { gen_types_cc } from "./gen_types_cc.ts";
import { gen_requests_h } from "./gen_requests_h.ts";
import { gen_requests_cc } from "./gen_requests_cc.ts";

async function main() {
  try {
    const args = parseArgs({
      args: process.argv.slice(2),
      allowPositionals: true,
      options: {
        output: {
          type: "string",
          short: "o",
          description: "Output directory",
        },
        extensions: {
          type: "string",
          short: "e",
          description: "Path to a partial metaModel with the cxx extensions",
        },
      },
    });

    const { positionals } = args;
    const { output: outputDirectory, extensions } = args.values;

    const input = positionals[0];

    if (!input) {
      throw new Error("Path to the LSP metaModel.json is required");
    }

    if (!outputDirectory) {
      throw new Error("Output directory is required");
    }

    const model = await readModel(input, extensions);

    await mkdir(outputDirectory, { recursive: true });

    gen_fwd_h({ outputDirectory, model });
    gen_enums_h({ outputDirectory, model });
    gen_enums_cc({ outputDirectory, model });
    gen_types_h({ outputDirectory, model });
    gen_types_cc({ outputDirectory, model });
    gen_requests_h({ outputDirectory, model });
    gen_requests_cc({ outputDirectory, model });

    console.log(
      child_process
        .execSync("clang-format --verbose -i *.h *.cc", {
          cwd: outputDirectory,
        })
        .toString(),
    );
  } catch (error) {
    if (error instanceof Error) {
      console.error(`${error.message}\n`);
    }

    console.error(
      "usage: cxx-gen-lsp --output output-directory [--extensions path to extensions.json] <path to metaModel.json>",
    );

    process.exit(1);
  }
}

async function readModel(input: string, extensions: string | undefined): Promise<MetaModel> {
  const model = JSON.parse(await readFile(input, "utf-8")) as MetaModel;

  if (!extensions) return model;

  const extension = JSON.parse(await readFile(extensions, "utf-8")) as Partial<MetaModel>;

  return {
    ...model,
    enumerations: [...model.enumerations, ...(extension.enumerations ?? [])],
    notifications: [...model.notifications, ...(extension.notifications ?? [])],
    requests: [...model.requests, ...(extension.requests ?? [])],
    structures: [...model.structures, ...(extension.structures ?? [])],
    typeAliases: [...model.typeAliases, ...(extension.typeAliases ?? [])],
  };
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
