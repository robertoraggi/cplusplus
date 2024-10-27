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

import * as path from "node:path";
import { getStructureProperties, isStringLike, MetaModel, Property, toCppType, Type } from "./MetaModel.js";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.js";

export function gen_types_h({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  let out = "";

  const emit = (s: string = "") => {
    out += `${s}\n`;
  };

  const getReturnType = (property: Property) => {
    let propertyType = toCppType(property.type);
    if (property.optional) {
      propertyType = `std::optional<${propertyType}>`;
    }
    return propertyType;
  };

  emit(copyrightHeader);
  emit();
  emit(`#pragma once`);
  emit();
  emit(`#include <cxx/lsp/fwd.h>`);
  emit();
  emit(`namespace cxx::lsp {`);
  emit();

  const typeAliasByName = new Map(model.typeAliases.map((t) => [t.name, t]));

  const isOrType = (type: Type) => {
    if (type.kind === "or") {
      return true;
    } else if (type.kind === "reference" && typeAliasByName.has(type.name)) {
      return isOrType(typeAliasByName.get(type.name)!.type);
    }
    return false;
  };

  model.structures.forEach((structure) => {
    const typeName = structure.name;
    emit();
    emit(`class ${typeName} final : public LSPObject {`);
    emit(`  public:`);
    emit(`    using LSPObject::LSPObject;`);
    emit();
    emit(`  explicit operator bool() const;`);

    const properties = getStructureProperties(model, structure);

    properties.forEach((property) => {
      const propertyName = property.name;
      const returnType = getReturnType(property);
      emit();
      emit(`    [[nodiscard ]]auto ${propertyName}() const -> ${returnType};`);
      if (property.optional || property.type.kind === "or") {
        emit();
        emit(`template <typename T>`);
        emit(`[[nodiscard]] auto ${propertyName}() -> T {`);
        emit(`  auto& value = (*repr_)["${propertyName}"];`);
        emit(`  return T(value);`);
        emit(`}`);
      }
    });

    emit();
    properties.forEach((property) => {
      const propertyName = property.name;
      const argumentType = getReturnType(property);
      emit();
      emit(`    auto ${propertyName}(${argumentType} ${propertyName}) -> ${typeName}&;`);

      if (property.type.kind === "array" && isStringLike(property.type.element, typeAliasByName)) {
        emit();
        emit(`    auto ${propertyName}(std::vector<std::string> ${propertyName}) -> ${typeName}& {`);
        emit(`      auto& value = (*repr_)["${propertyName}"];`);
        emit(`      value = std::move(${propertyName});`);
        emit(`      return *this;`);
        emit(`    }`);
      }
    });

    emit(`};`);
  });

  emit(`}`);

  const outputFile = path.join(outputDirectory, "types.h");
  writeFileSync(outputFile, out);
}
