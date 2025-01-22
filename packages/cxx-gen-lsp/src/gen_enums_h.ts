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

import { getEnumBaseType, getEnumeratorInitializer, getEnumeratorName, MetaModel } from "./MetaModel.js";

import path from "node:path";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.js";

export function gen_enums_h({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  let out = "";

  const emit = (s: string = "") => {
    out += `${s}\n`;
  };

  emit(copyrightHeader);
  emit();
  emit(`#pragma once`);
  emit();
  emit(`#include <string>`);
  emit(`#include <string_view>`);
  emit(`#include <optional>`);
  emit(`#include <cxx/lsp/fwd.h>`);
  emit();
  emit(`namespace cxx::lsp {`);

  model.enumerations.forEach((enumeration) => {
    emit();
    const enumBaseType = getEnumBaseType(enumeration);
    emit(`enum class ${enumeration.name}${enumBaseType} {`);
    enumeration.values.forEach((value) => {
      const enumeratorName = getEnumeratorName(value);
      const enumeratorInitializer = getEnumeratorInitializer(enumeration, value);
      emit(`  ${enumeratorName}${enumeratorInitializer},`);
    });
    emit(`};`);
  });
  emit();
  model.enumerations.forEach((enumeration) => {
    emit(`auto to_string(${enumeration.name} value) -> std::string;`);
  });
  emit();

  const stringEnums = model.enumerations.filter((enumeration) => enumeration.type.name === "string");
  emit(`namespace string_enums {`);
  emit();
  stringEnums.forEach((enumeration) => {
    emit(`[[nodiscard]] auto parse${enumeration.name}(std::string_view name) -> std::optional<${enumeration.name}>;`);
  });
  emit(`} // namespace string_enums`);
  emit();

  emit(`} // namespace cxx::lsp`);

  const outputFile = path.join(outputDirectory, "enums.h");
  writeFileSync(outputFile, out);
}
