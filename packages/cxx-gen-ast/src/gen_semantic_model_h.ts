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
import { cpy_header } from "./cpy_header.ts";
import type { CodecPlan } from "./semanticPlan.ts";

const persistenceNames: Record<string, string> = {
  P: "kPersisted",
  R: "kRebuilt",
  C: "kCompatibility",
  E: "kBoundary",
  D: "kDerived",
  F: "kFlushed",
};

export function gen_semantic_model_h({
  plan,
  output,
}: {
  plan: CodecPlan;
  output: string;
}) {
  const seen = new Set<string>();
  const rows: string[] = [];

  for (const field of plan.report) {
    const key = `${field.owner}::${field.name}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const note = (field.why ?? "").replace(/"/g, '\\"');
    rows.push(
      `    {"${field.owner}", "${field.name}", FieldPersistence::${persistenceNames[field.cls]}, "${note}"},`,
    );
  }

  const counts: Record<string, number> = {};
  for (const key of seen) void key;
  for (const field of plan.report)
    counts[field.cls] = (counts[field.cls] ?? 0) + 1;

  const out = `// Generated file by: gen_semantic_model_h.ts
${cpy_header}
#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace cxx {

enum class FieldPersistence : char {
  kPersisted = 'P',
  kRebuilt = 'R',
  kCompatibility = 'C',
  kBoundary = 'E',
  kDerived = 'D',
  kFlushed = 'F',
};

struct FieldDescriptor {
  std::string_view owner;
  std::string_view name;
  FieldPersistence persistence;
  std::string_view note;
};

inline constexpr FieldDescriptor kSemanticFieldModelStorage[] = {
${rows.join("\n")}
};

[[nodiscard]] inline auto semanticFieldModel() -> std::span<const FieldDescriptor> {
  return kSemanticFieldModelStorage;
}

[[nodiscard]] inline auto persistenceOf(std::string_view owner,
                                        std::string_view name)
    -> FieldPersistence {
  for (const auto& field : kSemanticFieldModelStorage) {
    if (field.owner == owner && field.name == name) return field.persistence;
  }
  return FieldPersistence::kDerived;
}

}  // namespace cxx
`;

  fs.writeFileSync(output, out);
}
