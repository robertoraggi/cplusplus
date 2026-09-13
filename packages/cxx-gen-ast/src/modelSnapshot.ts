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
import type { Model, Span } from "./parseModel.ts";

export function normalizeModel(model: Model, root: string): Model {
  return JSON.parse(
    JSON.stringify(model, (key, value) => {
      if (key === "file" && typeof value === "string" && path.isAbsolute(value))
        return path.relative(root, value).split(path.sep).join("/");
      return value;
    }),
  ) as Model;
}

export function snapshotModel(model: Model, root: string): Model {
  const sources = new Map<string, string[]>();
  function source(span: Span): string {
    let lines = sources.get(span.file);
    if (!lines) {
      lines = fs
        .readFileSync(path.resolve(root, span.file), "utf8")
        .split("\n");
      sources.set(span.file, lines);
    }
    const parts = lines.slice(span.startLine - 1, span.endLine);
    parts[parts.length - 1] = parts.at(-1)!.slice(0, span.endColumn - 1);
    parts[0] = parts[0]!.slice(span.startColumn - 1);
    return parts.join("\n");
  }
  const result = structuredClone(model);
  for (const entry of result.classes) {
    if (!entry.location?.file.endsWith("/ast.h")) continue;
    for (const field of entry.fields) {
      if (field.declaration && field.location) {
        field.declaredType = source({
          ...field.declaration,
          endLine: field.location.line,
          endColumn: field.location.column,
        }).trim();
      }
      if (field.initializer) {
        let text = source(field.initializer).trim();
        if (text.startsWith("=")) text = text.slice(1).trim();
        if (text.endsWith(";")) text = text.slice(0, -1).trim();
        field.initializerText = text;
      }
    }
  }
  return normalizeModel(result, root);
}

export function serializeModel(model: Model): string {
  return (
    JSON.stringify(
      model,
      (_key, value) => {
        if (!value || typeof value !== "object" || Array.isArray(value))
          return value;
        return Object.fromEntries(
          Object.keys(value)
            .sort()
            .map((key) => [key, value[key]]),
        );
      },
      2,
    ) + "\n"
  );
}
