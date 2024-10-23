// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

import { getEnumeratorName, MetaModel } from "./MetaModel.js";

import path from "node:path";
import { writeFile } from "node:fs/promises";
import { copyrightHeader } from "./copyrightHeader.js";

export async function gen_enums_cc({
  model,
  outputDirectory,
}: {
  model: MetaModel;
  outputDirectory: string;
}) {
  let out = "";

  const emit = (s: string = "") => {
    out += `${s}\n`;
  };

  emit(`#include <cxx/lsp/enums.h>`);
  emit();
  emit(`namespace cxx::lsp {`);

  model.enumerations.forEach((enumeration) => {
    emit();
    emit(`auto to_string(${enumeration.name} value) -> std::string {`);
    emit(`  switch (value) {`);

    enumeration.values.forEach((enumerator) => {
      const enumeratorName = getEnumeratorName(enumerator);

      const text =
        enumeration.type.name === "string" ? enumerator.value : enumerator.name;

      emit(`    case ${enumeration.name}::${enumeratorName}:`);
      emit(`      return "${text}";`);
    });

    emit(`  }`);
    emit(`}`);
  });

  emit();
  emit(`} // namespace cxx::lsp`);

  const outputFile = path.join(outputDirectory, "enums.cc");
  await writeFile(outputFile, out);
}
