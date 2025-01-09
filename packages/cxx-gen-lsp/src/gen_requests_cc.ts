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
import { isRequest, MetaModel, Structure, toCppType, Type } from "./MetaModel.js";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.js";
import { TypeGenerator } from "./gen_types_cc.js";

class RequestGenerator extends TypeGenerator {
  constructor({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
    super({ model, outputDirectory });
  }

  genTypes() {
    this.begin();

    this.emit();
    this.emit(`auto LSPRequest::id() const -> std::optional<std::variant<long, std::string>> {`);
    this.emit(`  if (!repr_->contains("id")) return std::nullopt;`);
    this.emit(`  const auto& id = repr_->at("id");`);
    this.emit(`  if (id.is_string()) return id.get<std::string>();`);
    this.emit(`  return id.get<long>();`);
    this.emit(`}`);
    this.emit();
    this.emit(`auto LSPRequest::id(std::optional<std::variant<long, std::string>> id)`);
    this.emit(`    -> LSPRequest& {`);
    this.emit(`  if (!id.has_value()) {`);
    this.emit(`    repr_->erase("id");`);
    this.emit(`    return *this;`);
    this.emit(`  }`);
    this.emit(`  if (std::holds_alternative<long>(*id)) {`);
    this.emit(`    (*repr_)["id"] = std::get<long>(*id);`);
    this.emit(`  } else {`);
    this.emit(`    (*repr_)["id"] = std::get<std::string>(*id);`);
    this.emit(`  }`);
    this.emit(`  return *this;`);
    this.emit(`}`);
    this.emit();
    this.emit(`auto LSPRequest::method() const -> std::string {`);
    this.emit(`  return repr_->at("method");`);
    this.emit(`}`);
    this.emit();
    this.emit(`auto LSPResponse::id() const -> std::optional<std::variant<long, std::string>> {`);
    this.emit(`  if (!repr_->contains("id")) return std::nullopt;`);
    this.emit(`  const auto& id = repr_->at("id");`);
    this.emit(`  if (id.is_string()) return id.get<std::string>();`);
    this.emit(`  return id.get<long>();`);
    this.emit(`}`);
    this.emit();
    this.emit(`auto LSPResponse::id(std::optional<std::variant<long, std::string>> id)`);
    this.emit(`    -> LSPResponse& {`);
    this.emit(`  if (!id.has_value()) {`);
    this.emit(`    repr_->erase("id");`);
    this.emit(`    return *this;`);
    this.emit(`  }`);
    this.emit(`  if (std::holds_alternative<long>(*id)) {`);
    this.emit(`    (*repr_)["id"] = std::get<long>(*id);`);
    this.emit(`  } else {`);
    this.emit(`    (*repr_)["id"] = std::get<std::string>(*id);`);
    this.emit(`  }`);
    this.emit(`  return *this;`);
    this.emit(`}`);

    const requestsAndNotifications = [...this.model.requests, ...this.model.notifications];

    requestsAndNotifications.forEach((request) => {
      const { typeName } = request;
      this.emit();
      this.emit(`auto ${typeName}::method(std::string method) -> ${typeName}& {`);
      this.emit(`  (*repr_)["method"] = std::move(method);`);
      this.emit(`  return *this;`);
      this.emit(`}`);
      this.emit();
      this.emit(`auto ${typeName}::id(std::variant<long, std::string> id) -> ${typeName}& {`);
      this.emit(`  return static_cast<${typeName}&>(LSPRequest::id(std::move(id)));`);
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

        this.emit();
        this.emit(`auto ${responseTypeName}::id(std::variant<long, std::string> id) -> ${responseTypeName}& {`);
        this.emit(`  return static_cast<${responseTypeName}&>(LSPResponse::id(std::move(id)));`);
        this.emit(`}`);

        // synthetic structure with result property

        const structure: Structure = {
          name: responseTypeName,
          properties: [{ name: "result", type: request.result, optional: false }],
        };

        this.generateGetters({ structure, properties: structure.properties });
      }
    });

    this.end();
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
