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
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { z } from "zod";
import kwgen from "./kwgen.ts";

const OptionsSchema = z.object({
  noEnums: z
    .boolean()
    .default(true)
    .describe("Add enumerations for the tokens"),
  tokenPrefix: z.string().default("T_").describe("Prefix for the token names"),
  tokenType: z
    .string()
    .default("TokenKind")
    .describe("Type of the tokens, e.g., TokenKind"),
  toUpper: z
    .boolean()
    .default(true)
    .describe("Convert token names to uppercase"),
  defaultToken: z
    .string()
    .default("T_IDENTIFIER")
    .describe("Default token to return when no keyword matches"),
  output: z
    .string()
    .nonempty()
    .describe("Output file path or a function to handle the generated code"),
  keywords: z
    .array(z.string())
    .nonempty()
    .describe("List of keywords to classify"),
  classifier: z
    .string()
    .default("classify")
    .describe("Function name prefix for the keyword classification"),
  copyright: z.string().optional().describe("Optional copyright notice"),
});

async function main() {
  const { positionals } = parseArgs({
    allowPositionals: true,
  });

  positionals.forEach((inputFile) => {
    const source = readFileSync(inputFile, "utf8");
    try {
      const options = OptionsSchema.parse(JSON.parse(source));
      kwgen(options);
    } catch (error) {
      if (error instanceof z.ZodError) {
        console.error("error: Invalid options in", inputFile);
        console.error(error.errors);
      } else {
        console.error("Error processing", inputFile, ":", error);
      }
      process.exit(1);
    }
  });

  if (positionals.length === 0) {
    console.error("Usage: kwgen <input-file.json> [<input-file2.json> ...]");
    console.error(
      "Each input file should contain a JSON object with the keyword options.",
    );
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
