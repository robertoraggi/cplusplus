import { writeFileSync } from "node:fs";

interface Options {
  keywords: string[];
  output: string | ((code: string) => void);
  noEnums: boolean;
  tokenPrefix: string;
  tokenType: string;
  toUpper: boolean;
  defaultToken: string;
  classifier: string;
}

export default function kwgen(options: Options) {
  const { keywords, output, tokenType, defaultToken, classifier } = options;

  let out: string[] = [];
  const emit = (s: string) => out.push(s);

  const keywordsByLength = Map.groupBy(keywords, (s) => s.length);

  const lengths = Array.from(keywordsByLength.keys());
  lengths.sort((a, b) => a - b);

  for (const length of lengths) {
    const items = keywordsByLength.get(length)!;

    emit(
      `static inline auto ${classifier}${length}(const char* s) -> ${tokenType} {`,
    );
    gen({ items, n: 0, emit, options });
    emit(`  return ${defaultToken};`);
    emit(`}`);
    emit("");
  }

  emit(`static auto ${classifier}(const char* s, int n) -> ${tokenType} {`);
  emit(`  switch (n) {`);

  for (const length of lengths) {
    emit(`  case ${length}:`);
    emit(`    return ${classifier}${length}(s);`);
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

  for (const [ch, next] of groups) {
    if (!ch) continue;

    if (next.length == 1) {
      const item = next[0];
      const cond = Array.from(item.slice(n)).map(
        (c, i) => `s[${n + i}] == '${c}'`,
      );
      emit(`${ind}if (${cond.join(" && ")}) {`);
      const x = toUpper ? item.toUpperCase() : item;
      emit(`${ind}  return ${tokenPrefix}${x};`);
      emit(`${ind}}`);
      continue;
    }

    emit(`${ind}if (s[${n}] == '${ch}') {`);
    gen({ items: next, n: n + 1, emit, options });
    emit(`${ind}}`);
  }
  if (groups.has("")) {
    const item = groups.get("")![0];
    const x = toUpper ? item.toUpperCase() : item;
    emit(`${ind}  return ${tokenPrefix}${x};`);
  }
}
