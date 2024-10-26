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
import { Enumeration, MetaModel, Structure, toCppType, Type, TypeAlias } from "./MetaModel.js";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.js";

class RequestGenerator {
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

    this.model.requests.forEach((request) => {
      const { typeName } = request;
      this.emit();
      this.emit(`auto ${typeName}::method() const -> std::string {`);
      this.emit(`  return repr_->at("method");`);
      this.emit(`}`);
      this.emit();
      this.emit(`auto ${typeName}::method(std::string method) -> ${typeName}& {`);
      this.emit(`  (*repr_)["method"] = std::move(method);`);
      this.emit(`  return *this;`);
      this.emit(`}`);
      this.emit();
      this.emit(`auto ${typeName}::id() const -> std::variant<long, std::string> {`);
      this.emit(`  const auto& id = repr_->at("id");`);
      this.emit(`  if (id.is_string()) return id.get<std::string>();`);
      this.emit(`  return id.get<long>();`);
      this.emit(`}`);
      this.emit();
      this.emit(`auto ${typeName}::id(long id) -> ${typeName}& {`);
      this.emit(`  (*repr_)["id"] = id;`);
      this.emit(`  return *this;`);
      this.emit(`}`);
      this.emit();
      this.emit(`auto ${typeName}::id(std::string id) -> ${typeName}& {`);
      this.emit(`  (*repr_)["id"] = std::move(id);`);
      this.emit(`  return *this;`);
      this.emit(`}`);

      if (request.params) {
        const paramsTypeName = toCppType(request.params);

        this.emit();
        this.emit(`auto ${typeName}::params() const -> ${paramsTypeName} {`);
        this.emit(`  if (!repr_->contains("params")) repr_->emplace("params", json::object());`);
        this.emit(`  return ${paramsTypeName}(repr_->at("params"));`);
        this.emit(`}`);
        this.emit();
        this.emit(`auto ${typeName}::params(${paramsTypeName} params) -> ${typeName}& {`);
        this.emit(`  (*repr_)["params"] = std::move(params);`);
        this.emit(`  return *this;`);
        this.emit(`}`);
      }

      if (request.result) {
        const resultTypeName = typeName.replace(/Request$/, "Response");
        this.emit();
        this.emit(`auto ${resultTypeName}::id() const -> std::variant<long, std::string> {`);
        this.emit(`  const auto& id = repr_->at("id");`);
        this.emit(`  if (id.is_string()) return id.get<std::string>();`);
        this.emit(`  return id.get<long>();`);
        this.emit(`}`);
        this.emit();
        this.emit(`auto ${resultTypeName}::id(long id) -> ${resultTypeName}& {`);
        this.emit(`  (*repr_)["id"] = id;`);
        this.emit(`  return *this;`);
        this.emit(`}`);
        this.emit();
        this.emit(`auto ${resultTypeName}::id(std::string id) -> ${resultTypeName}& {`);
        this.emit(`  (*repr_)["id"] = std::move(id);`);
        this.emit(`  return *this;`);
        this.emit(`}`);
      }
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
    this.emit(`#include <cxx/lsp/requests.h>`);
    this.emit(`#include <cxx/lsp/types.h>`);
    this.emit(`#include <cxx/lsp/enums.h>`);
    this.emit();
    this.emit(`namespace cxx::lsp {`);
    this.emit();
  }

  end() {
    this.emit(`} // namespace cxx::lsp`);

    const outputFile = path.join(this.outputDirectory, "requests.cc");
    writeFileSync(outputFile, this.out);
  }
}

export function gen_requests_cc({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  const generator = new RequestGenerator({ model, outputDirectory });
  generator.genTypes();
}
