// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

import { getEnumeratorName, type MetaModel } from "./MetaModel.ts";

import path from "node:path";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.ts";

export function gen_enums_cc({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  let out = "";

  const emit = (s: string = "") => {
    out += `${s}\n`;
  };

  emit(copyrightHeader);
  emit();
  emit(`#include <cxx/lsp/enums.h>`);
  emit(`#include <unordered_map>`);
  emit(`#include <format>`);
  emit();

  emit(`namespace cxx::lsp {`);

  model.enumerations.forEach((enumeration) => {
    emit();
    emit(`auto to_string(${enumeration.name} value) -> std::string {`);
    emit(`  switch (value) {`);

    enumeration.values.forEach((enumerator) => {
      const enumeratorName = getEnumeratorName(enumerator);

      const text = enumeration.type.name === "string" ? enumerator.value : enumerator.name;

      emit(`    case ${enumeration.name}::${enumeratorName}:`);
      emit(`      return "${text}";`);
    });

    emit(`  }`);
    emit();
    emit(`  lsp_runtime_error("invalid enumerator value");`);
    emit(`}`);
  });

  const stringEnums = model.enumerations.filter((enumeration) => enumeration.type.name === "string");

  emit();
  emit(`namespace string_enums {`);
  stringEnums.forEach((enumeration) => {
    emit();
    emit(`auto parse${enumeration.name}(std::string_view name) -> std::optional<${enumeration.name}> {`);
    emit(`  static std::unordered_map<std::string_view, ${enumeration.name}> map{`);
    enumeration.values.forEach((enumerator) => {
      const enumeratorName = getEnumeratorName(enumerator);
      const text = enumeration.type.name === "string" ? enumerator.value : enumerator.name;
      emit(`    {"${text}", ${enumeration.name}::${enumeratorName}},`);
    });
    emit(`  };`);
    emit(`  const auto it = map.find(name);`);
    emit(`  if (it != map.end()) `);
    emit(`    return it->second;`);
    emit(`  return std::nullopt;`);
    emit(`}`);
  });
  emit();
  emit(`} // namespace string_enums`);
  emit();

  emit(`} // namespace cxx::lsp`);

  const outputFile = path.join(outputDirectory, "enums.cc");
  writeFileSync(outputFile, out);
}
