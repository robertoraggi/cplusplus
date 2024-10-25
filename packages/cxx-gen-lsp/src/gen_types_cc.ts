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
import {
  Enumeration,
  getStructureProperties,
  MetaModel,
  Property,
  Structure,
  toCppType,
  Type,
  TypeAlias,
} from "./MetaModel.js";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.js";

class TypeGenerator {
  readonly structByName: Map<string, Structure>;
  readonly enumByName: Map<string, Enumeration>;
  readonly typeAliasByName: Map<string, TypeAlias>;
  readonly model: MetaModel;
  readonly outputDirectory: string;
  out: string = "";

  constructor({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
    this.model = model;
    this.outputDirectory = outputDirectory;
    this.structByName = new Map(model.structures.map((s) => [s.name, s]));
    this.enumByName = new Map(model.enumerations.map((e) => [e.name, e]));
    this.typeAliasByName = new Map(model.typeAliases.map((t) => [t.name, t]));
  }

  genTypes() {
    this.begin();

    this.model.structures.forEach((structure) => {
      const properties = getStructureProperties(this.model, structure);

      this.generateValidator({ structure, properties });
      this.generateGetters({ structure, properties });
      this.generateSetters({ structure, properties });
    });

    this.end();
  }

  emit(s: string = "") {
    this.out += `${s}\n`;
  }

  getPropertyType({ type, optional }: { type: Type; optional?: boolean }): string {
    let propertyType = toCppType(type);

    if (optional) {
      propertyType = `std::optional<${propertyType}>`;
    }

    return propertyType;
  }

  isStringLike(type: Type): boolean {
    switch (type.kind) {
      case "base":
        return type.name === "string";

      case "reference": {
        if (this.typeAliasByName.has(type.name)) {
          return this.isStringLike(this.typeAliasByName.get(type.name)!.type);
        }
        return false;
      }

      case "stringLiteral":
        return true;

      default:
        return false;
    } // switch
  }

  begin() {
    this.emit(copyrightHeader);
    this.emit();
    this.emit(`#include <cxx/lsp/types.h>`);
    this.emit();
    this.emit(`namespace cxx::lsp {`);
    this.emit();
  }

  end() {
    this.emit(`} // namespace cxx::lsp`);

    const outputFile = path.join(this.outputDirectory, "types.cc");
    writeFileSync(outputFile, this.out);
  }

  generateGetters({ structure, properties }: { structure: Structure; properties: Property[] }) {
    const typeName = structure.name;

    properties.forEach((property) => {
      this.beginPropertyGetter({ structure, property });
      this.generatePropertyGetter({ structure, property });
      this.endPropertyGetter();
    });
  }

  generateValidator({ structure, properties }: { structure: Structure; properties: Property[] }) {
    this.emit();
    this.emit(`${structure.name}::operator bool() const {`);
    this.emit(`if (!repr_.is_object() || repr_.is_null()) return false;`);

    const requiredProperties = properties.filter((p) => !p.optional);

    requiredProperties.forEach(({ name, type }) => {
      this.emit(`if (!repr_.contains("${name}")) return false;`);

      if (type.kind === "stringLiteral") {
        this.emit(`if (repr_["${name}"] != "${type.value}")`);
        this.emit(`  return false;`);
      }
    });

    this.emit(`return true;`);
    this.emit(`}`);
  }

  beginPropertyGetter({ structure, property }: { structure: Structure; property: Property }) {
    const returnType = this.getPropertyType(property);

    this.emit();
    this.emit(`auto ${structure.name}::${property.name}() const -> ${returnType} {`);

    if (property.optional) {
      this.emit(`if (!repr_.contains("${property.name}")) return std::nullopt;`);
      this.emit();
    }

    this.emit(`const auto& value = repr_["${property.name}"];`);
    this.emit();
  }

  endPropertyGetter() {
    this.emit(`}`);
  }

  generatePropertyGetter({ structure, property }: { structure: Structure; property: Property }): void {
    const propertyType = toCppType(property.type);

    switch (property.type.kind) {
      case "base":
        if (!this.generatePropertyGetterBase({ structure, property })) break;
        return;

      case "reference":
        if (!this.generatePropertyGetterReference({ structure, property })) break;
        return;

      case "tuple": {
        throw new Error(`Unexpected tuple type`);
      }

      case "or":
        this.emit(`${propertyType} result;`);
        this.emit();
        this.emit(`details::try_emplace(result, value);`);
        this.emit();
        this.emit(`return result;`);
        return;

      case "and": {
        throw new Error(`Unexpected and type`);
      }

      case "array":
        this.emit(`assert(value.is_array());`);
        this.emit(`return ${propertyType}(value);`);
        return;

      case "map":
        this.emit(`assert(value.is_object());`);
        this.emit(`return ${propertyType}(value);`);
        return;

      case "stringLiteral":
        this.emit(`assert(value.is_string());`);
        this.emit(`return value.get<std::string>();`);
        return;

      case "literal": {
        throw new Error(`Unexpected literal type`);
      }

      default:
        throw new Error(`Unknown type kind: ${JSON.stringify(property.type)}`);
    } // switch

    this.emit();
    this.emit(`lsp_runtime_error("${structure.name}::${property.name}: not implement yet");`);
  }

  generatePropertyGetterBase({ structure, property }: { structure: Structure; property: Property }): boolean {
    if (property.type.kind !== "base") return false;

    switch (property.type.name) {
      case "null":
        throw new Error(`Unexpected null type`);

      case "string":
        this.emit(`assert(value.is_string());`);
        this.emit(`return value.get<std::string>();`);
        return true;

      case "integer":
        this.emit(`assert(value.is_number_integer());`);
        this.emit(`return value.get<int>();`);
        return true;

      case "uinteger":
        this.emit(`assert(value.is_number_integer());`);
        this.emit(`return value.get<long>();`);
        return true;

      case "decimal":
        this.emit(`assert(value.is_number());`);
        this.emit(`return value.get<double>();`);
        return true;

      case "boolean":
        this.emit(`assert(value.is_boolean());`);
        this.emit(`return value.get<bool>();`);
        return true;

      case "DocumentUri":
        this.emit(`assert(value.is_string());`);
        this.emit(`return value.get<std::string>();`);
        return true;

      case "URI":
        this.emit(`assert(value.is_string());`);
        this.emit(`return value.get<std::string>();`);
        return true;

      default:
        throw new Error(`Unknown base type: ${JSON.stringify(property.type)}`);
    } // switch type.name
  }

  generatePropertyGetterReference({ structure, property }: { structure: Structure; property: Property }): boolean {
    if (property.type.kind !== "reference") return false;

    const propertyType = toCppType(property.type);

    if (this.structByName.has(property.type.name)) {
      this.emit(`return ${propertyType}(value);`);
      return true;
    }

    if (this.enumByName.has(property.type.name)) {
      const enumeration = this.enumByName.get(property.type.name)!;
      if (enumeration.type.name !== "string") {
        this.emit(`return ${propertyType}(value);`);
        return true;
      }

      // todo: string-like enumeration
      console.log(`string-like enumeration: ${property.type.name}`);
      return false;
    }

    if (this.typeAliasByName.has(property.type.name)) {
      const typeAlias = this.typeAliasByName.get(property.type.name)!;

      if (property.type.name === "LSPObject") {
        this.emit(`assert(value.is_object());`);
        this.emit(`return ${propertyType}(value);`);
        return true;
      }

      if (property.type.name === "LSPAny") {
        this.emit(`assert(value.is_object());`);
        this.emit(`return ${propertyType}(value);`);
        return true;
      }

      switch (typeAlias.type.kind) {
        case "base":
          return this.generatePropertyGetterBase({ structure, property: { ...property, type: typeAlias.type } });

        case "or":
          this.emit(`${propertyType} result;`);
          this.emit();
          this.emit(`details::try_emplace(result, value);`);
          this.emit();
          this.emit(`return result;`);
          return true;

        default:
          throw new Error(`Unexpected reference to type alias`);
      } // switch
    } else {
      throw new Error(`Unexpected reference type`);
    }
  }

  generateSetters({ structure, properties }: { structure: Structure; properties: Property[] }) {
    const typeName = structure.name;

    properties.forEach(({ name, type, optional }) => {
      const argumentType = this.getPropertyType({ type, optional });
      this.emit();
      this.emit(`auto ${typeName}::${name}(${argumentType} ${name})`);
      this.emit(`-> ${typeName}& { return *this; }`);
    });
  }
}

export function gen_types_cc({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  const generator = new TypeGenerator({ model, outputDirectory });
  generator.genTypes();
}
