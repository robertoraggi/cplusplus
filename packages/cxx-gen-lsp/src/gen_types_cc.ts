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

import * as path from "node:path";
import type { Enumeration, MetaModel, Property, Structure, Type, TypeAlias } from "./MetaModel.ts";
import { getStructureProperties, toCppType } from "./MetaModel.ts";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.ts";

export class TypeGenerator {
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

  begin() {
    this.emit(copyrightHeader);
    this.emit();
    this.emit(`#include <cxx/lsp/types.h>`);
    this.emit(`#include <cxx/lsp/enums.h>`);
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
    properties.forEach((property) => {
      this.beginPropertyGetter({ structure, property });
      this.generatePropertyGetter({ structure, property });
      this.endPropertyGetter();
    });
  }

  generateValidator({ structure, properties }: { structure: Structure; properties: Property[] }) {
    this.emit();
    this.emit(`${structure.name}::operator bool() const {`);
    this.emit(`if (!repr_->is_object() || repr_->is_null()) return false;`);

    const requiredProperties = properties.filter((p) => !p.optional);

    requiredProperties.forEach(({ name, type }) => {
      this.emit(`if (!repr_->contains("${name}")) return false;`);

      if (type.kind === "stringLiteral") {
        this.emit(`if ((*repr_)["${name}"] != "${type.value}")`);
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
      this.emit(`if (!repr_->contains("${property.name}")) return std::nullopt;`);
      this.emit();
    }

    this.emit(`auto& value = (*repr_)["${property.name}"];`);
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
        this.emit(`if (value.is_null()) value = json::array();`);
        this.emit(`return ${propertyType}(value);`);
        return;

      case "map":
        this.emit(`if (value.is_null()) value = json::object();`);
        this.emit(`return ${propertyType}(value);`);
        return;

      case "stringLiteral":
        this.emit(`if (value.is_null()) value = "${property.type.value}";`);
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

    this.emit(`lsp_runtime_error("${structure.name}::${property.name}: not implemented yet");`);
  }

  generatePropertyGetterBase({ property }: { structure: Structure; property: Property }): boolean {
    if (property.type.kind !== "base") return false;

    switch (property.type.name) {
      case "null":
        this.emit(`assert(value.is_null());`);
        this.emit(`return nullptr;`);
        return true;

      case "string":
        this.emit(`if (value.is_null()) value = "";`);
        this.emit(`assert(value.is_string());`);
        this.emit(`return value.get<std::string>();`);
        return true;

      case "integer":
        this.emit(`if (value.is_null()) value = 0;`);
        this.emit(`assert(value.is_number_integer());`);
        this.emit(`return value.get<int>();`);
        return true;

      case "uinteger":
        this.emit(`if (value.is_null()) value = 0;`);
        this.emit(`assert(value.is_number_integer());`);
        this.emit(`return value.get<long>();`);
        return true;

      case "decimal":
        this.emit(`if (value.is_null()) value = 0.0;`);
        this.emit(`assert(value.is_number());`);
        this.emit(`return value.get<double>();`);
        return true;

      case "boolean":
        this.emit(`if (value.is_null()) value = false;`);
        this.emit(`assert(value.is_boolean());`);
        this.emit(`return value.get<bool>();`);
        return true;

      case "DocumentUri":
        this.emit(`if (value.is_null()) value = "";`);
        this.emit(`assert(value.is_string());`);
        this.emit(`return value.get<std::string>();`);
        return true;

      case "URI":
        this.emit(`if (value.is_null()) value = "";`);
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
      this.emit(`const auto enumValue = string_enums::parse${property.type.name}(value.get<std::string>());`);
      this.emit();
      this.emit(`if (enumValue.has_value()) return *enumValue;`);
      this.emit();
      this.emit(`lsp_runtime_error("invalid value for ${property.type.name} enumeration");`);
      return true;
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

    properties.forEach((property) => {
      const argumentType = this.getPropertyType(property);
      this.emit();
      this.emit(`auto ${typeName}::${property.name}(${argumentType} ${property.name})`);
      this.emit(`-> ${typeName}& {`);

      if (property.optional) {
        this.emit(`if (!${property.name}.has_value()) {`);
        this.emit(`repr_->erase("${property.name}");`);
        this.emit(`return *this;`);
        this.emit(`}`);
      }

      this.generatePropertySetter({ property, structure });

      this.emit(`return *this;`);
      this.emit(`}`);
    });
  }

  private generatePropertySetter({ property, structure }: { property: Property; structure: Structure }): void {
    const typeName = structure.name;
    const value = property.optional ? `${property.name}.value()` : property.name;

    switch (property.type.kind) {
      case "base":
        this.emit(`(*repr_)["${property.name}"] = std::move(${value});`);
        return;

      case "reference":
        if (!this.generatePropertySetterReference({ structure, property, value })) break;
        return;

      case "array":
        this.emit(`(*repr_)["${property.name}"] = std::move(${value});`);
        return;

      case "or":
        if (!this.generatePropertySetterOr({ structure, property, value })) break;
        return;

      default:
        break;
    } // switch

    this.emit(`lsp_runtime_error("${typeName}::${property.name}: not implemented yet");`);
  }

  generatePropertySetterReference({
    property,
    value,
  }: {
    structure: Structure;
    property: Property;
    value: string;
  }): boolean {
    if (property.type.kind !== "reference") return false;

    if (this.enumByName.has(property.type.name)) {
      const enumeration = this.enumByName.get(property.type.name)!;

      if (enumeration.type.name !== "string") {
        const enumBaseType = toCppType(enumeration.type);
        this.emit(`(*repr_)["${property.name}"] = static_cast<${enumBaseType}>(${value});`);
        return true;
      }

      // string-like enumeration
      this.emit(`(*repr_)["${property.name}"] = to_string(${value});`);
      return true;
    }

    if (this.structByName.has(property.type.name)) {
      this.emit(`(*repr_)["${property.name}"] = ${value};`);
      return true;
    }

    return false;
  }

  generatePropertySetterOr({
    structure,
    property,
    value,
  }: {
    structure: Structure;
    property: Property;
    value: string;
  }): boolean {
    if (property.type.kind !== "or") return false;

    this.emit();
    this.emit(`struct {`);
    this.emit(`json* repr_;`);

    property.type.items.forEach((item) => {
      const itemType = this.getPropertyType({ type: item });
      this.emit();
      this.emit(`void operator()(${itemType} ${property.name}) {`);
      this.generatePropertySetter({ property: { name: property.name, type: item, optional: false }, structure });
      this.emit(`}`);
    });

    this.emit(`} v{repr_};`);
    this.emit();
    this.emit(`std::visit(v, ${value});`);
    this.emit();

    return true;
  }
}

export function gen_types_cc({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  const generator = new TypeGenerator({ model, outputDirectory });
  generator.genTypes();
}
