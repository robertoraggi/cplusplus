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

import { writeFileSync } from "node:fs";

interface Options {
  keywords: string[];
  output: string | ((code: string) => void);
  copyright?: string;
  noEnums: boolean;
  tokenPrefix: string;
  tokenType: string;
  toUpper: boolean;
  defaultToken: string;
  classifier: string;
  tokenValues?: Record<string, number>;
}

export default function kwgen(options: Options) {
  const {
    copyright,
    keywords,
    output,
    classifier,
    defaultToken,
    noEnums,
    tokenPrefix,
    tokenType,
  } = options;

  let out: string[] = [];
  const emit = (s: string = "") => out.push(s);

  if (copyright !== undefined) {
    emit("// Generated file by: kwgen.ts");
    emit(copyright);
  }

  emit();
  emit("#pragma once");
  emit();

  const keywordsByLength = Map.groupBy(keywords, (s) => s.length);

  const lengths = Array.from(keywordsByLength.keys());
  lengths.sort((a, b) => a - b);

  const getTokenName = (s: string) => {
    const x = options.toUpper ? s.toUpperCase() : s;
    let name = `${tokenPrefix}${x}`;
    const scopeIndex = name.lastIndexOf("::");
    if (scopeIndex !== -1) {
      name = `${name.slice(scopeIndex + 2)}`;
    }
    return name;
  };

  if (!noEnums) {
    emit(`enum class ${tokenType} {`);
    emit(`  ${getTokenName(defaultToken)},`);
    keywords.forEach((kw) => {
      const name = getTokenName(kw);
      emit(`  ${name},`);
    });
    emit(`};`);
    emit("");
  }

  const valueParameter = options.tokenValues ? ", int& value" : "";
  const valueArgument = options.tokenValues ? ", value" : "";

  for (const length of lengths) {
    const items = keywordsByLength.get(length)!;

    emit(
      `static inline auto ${classifier}${length}(const char* s${valueParameter}) -> ${tokenType} {`,
    );
    gen({ items, n: 0, emit, options });
    emit(`  return ${defaultToken};`);
    emit(`}`);
    emit("");
  }

  emit(
    `static auto ${classifier}(const char* s, int n${valueParameter}) -> ${tokenType} {`,
  );
  if (options.tokenValues) emit("  value = 0;");
  emit(`  switch (n) {`);

  for (const length of lengths) {
    emit(`  case ${length}:`);
    emit(`    return ${classifier}${length}(s${valueArgument});`);
  }

  emit(`  default:`);
  emit(`    return ${defaultToken};`);
  emit(`  } // switch`);
  emit(`}`);

  const code = out.join("\n");

  if (typeof output === "function") {
    output(code);
  } else {
    writeFileSync(output, code, "utf8");
  }
}

function gen({
  items,
  n,
  emit,
  options,
}: {
  items: string[];
  n: number;
  emit: (s: string) => void;
  options: Options;
}) {
  const { tokenPrefix, toUpper } = options;

  const groups = Map.groupBy(items, (s) => s[n] ?? "");

  const ind = "  ".repeat(n + 1);

  let sep = "";

  for (const [ch, next] of groups) {
    if (!ch) continue;

    // if (next.length == 1) {
    //   const item = next[0];
    //   const cond = Array.from(item.slice(n)).map(
    //     (c, i) => `s[${n + i}] == '${c}'`,
    //   );
    //   emit(`${ind}if (${cond.join(" && ")}) {`);
    //   const x = toUpper ? item.toUpperCase() : item;
    //   emit(`${ind}  return ${tokenPrefix}${x};`);
    //   emit(`${ind}}`);
    //   continue;
    // }

    emit(`${ind}${sep}if (s[${n}] == '${ch}') {`);
    gen({ items: next, n: n + 1, emit, options });
    emit(`${ind}}`);
    sep = " else ";
  }
  if (groups.has("")) {
    const item = groups.get("")![0]!;
    const x = toUpper ? item.toUpperCase() : item;
    const value = options.tokenValues?.[item];
    if (value !== undefined) emit(`${ind}  value = ${value};`);
    emit(`${ind}  return ${tokenPrefix}${x};`);
  }
}
