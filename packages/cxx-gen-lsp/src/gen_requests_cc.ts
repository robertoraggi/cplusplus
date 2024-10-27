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
import { Enumeration, isRequest, MetaModel, Structure, toCppType, Type, TypeAlias } from "./MetaModel.js";
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

    const requestsAndNotifications = [...this.model.requests, ...this.model.notifications];

    requestsAndNotifications.forEach((request) => {
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

      if (isRequest(request) && request.result) {
        const responseTypeName = typeName.replace(/Request$/, "Response");
        const resultTypeName = toCppType(request.result);

        this.emit();
        this.emit(`auto ${responseTypeName}::id() const -> std::variant<long, std::string> {`);
        this.emit(`  const auto& id = repr_->at("id");`);
        this.emit(`  if (id.is_string()) return id.get<std::string>();`);
        this.emit(`  return id.get<long>();`);
        this.emit(`}`);
        this.emit();
        this.emit(`auto ${responseTypeName}::id(long id) -> ${responseTypeName}& {`);
        this.emit(`  (*repr_)["id"] = id;`);
        this.emit(`  return *this;`);
        this.emit(`}`);
        this.emit();
        this.emit(`auto ${responseTypeName}::id(std::string id) -> ${responseTypeName}& {`);
        this.emit(`  (*repr_)["id"] = std::move(id);`);
        this.emit(`  return *this;`);
        this.emit(`}`);
        this.emit();
        this.emit(`auto ${responseTypeName}::result() const -> ${resultTypeName} {`);
        switch (request.result.kind) {
          case "base": {
            if (request.result.name === "null") {
              this.emit(`  return nullptr;`);
            } else {
              this.emit(`  return repr_->at("result").get<${toCppType(request.result)}>(); // base`);
            }
            break;
          }

          case "reference": {
            if (this.structByName.has(request.result.name)) {
              this.emit(`  if (!repr_->contains("result")) (*repr_)["result"] = nullptr;`);
              this.emit(`  return ${resultTypeName}(repr_->at("result")); // reference`);
            } else {
              this.emit(`  lsp_runtime_error("${responseTypeName}::result() - not implemented yet");`);
            }
            break;
          }

          default: {
            this.emit(`  lsp_runtime_error("${responseTypeName}::result() - not implemented yet");`);
          }
        } // swtch
        this.emit(`}`);
        this.emit();
        this.emit(`auto ${responseTypeName}::result(${resultTypeName} result) -> ${responseTypeName}& {`);
        switch (request.result.kind) {
          case "base":
            this.emit(`  (*repr_)["result"] = std::move(result); // base`);
            break;
          default:
            this.emit(`  lsp_runtime_error("${responseTypeName}::result() - not implemented yet");`);
        } // switch
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
