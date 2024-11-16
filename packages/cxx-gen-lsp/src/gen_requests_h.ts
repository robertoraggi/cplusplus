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
import { MetaModel, toCppType, Request, Notification, isRequest, Property, Structure } from "./MetaModel.js";
import { writeFileSync } from "node:fs";
import { copyrightHeader } from "./copyrightHeader.js";

const beginHeaderFragment = `
#pragma once

#include <cxx/lsp/fwd.h>

namespace cxx::lsp {

`;

export function gen_requests_h({ model, outputDirectory }: { model: MetaModel; outputDirectory: string }) {
  let out = "";

  const emit = (s: string = "") => {
    out += `${s}\n`;
  };

  emit(copyrightHeader);
  emit(beginHeaderFragment);

  const requestsAndNotifications: Array<Request | Notification> = [...model.requests, ...model.notifications];

  emit();
  emit(`#define FOR_EACH_LSP_REQUEST_TYPE(V) \\`);
  model.requests.forEach((request, index) => {
    const nameWithoutSuffix = request.typeName.replace(/Request$/, "");
    const sep = index + 1 < model.requests.length ? " \\" : "";
    emit(`  V(${nameWithoutSuffix}, "${request.method}")${sep}`);
  });

  emit();
  emit(`#define FOR_EACH_LSP_NOTIFICATION_TYPE(V) \\`);
  model.notifications.forEach((notification, index) => {
    const nameWithoutSuffix = notification.typeName.replace(/Notification$/, "");
    const sep = index + 1 < model.notifications.length ? " \\" : "";
    emit(`  V(${nameWithoutSuffix}, "${notification.method}")${sep}`);
  });

  requestsAndNotifications.forEach((request) => {
    const typeName = request.typeName;
    emit();
    emit(`class ${typeName} final : public LSPRequest {`);
    emit(`public:`);
    emit(`  using LSPRequest::LSPRequest;`);
    emit(`  using LSPRequest::id;`);
    emit(`  using LSPRequest::method;`);
    emit();
    emit(`  auto method(std::string method) -> ${typeName}&;`);
    emit(`  auto id(std::variant<long, std::string> id) -> ${typeName}&;`);

    if (request.params) {
      const paramsType = toCppType(request.params);
      emit();
      emit(`  [[nodiscard]] auto params() const -> ${paramsType};`);
      emit(`  auto params(${paramsType} result) -> ${typeName}&;`);
    }

    emit(`};`);

    // generate the respose type if the request has a result
    if (isRequest(request) && request.result) {
      const responseTypeName = typeName.replace(/Request$/, "Response");
      const resultType = toCppType(request.result);

      emit();

      emit(`class ${responseTypeName} final : public LSPResponse {`);
      emit(`public:`);
      emit(`  using LSPResponse::LSPResponse;`);
      emit(`  using LSPResponse::id;`);
      emit(`  using Result = ${resultType};`);
      emit();
      emit(`  auto id(std::variant<long, std::string> id) -> ${responseTypeName}&;`);
      emit();

      const structure: Structure = {
        name: responseTypeName,
        properties: [{ name: "result", type: request.result, optional: false }],
      };

      structure.properties.forEach((property) => {
        const propertyName = property.name;
        const returnType = toCppType(property.type);
        emit();
        emit(`    [[nodiscard ]]auto ${propertyName}() const -> ${returnType};`);
        if (property.type.kind === "or") {
          emit();
          emit(`template <typename T>`);
          emit(`[[nodiscard]] auto ${propertyName}() -> T {`);
          emit(`  auto& value = (*repr_)["${propertyName}"];`);
          emit(`  return T(value);`);
          emit(`}`);
        }
      });

      emit(`};`);
    }
  });

  emit();
  emit(`template <typename Visitor>`);
  emit(`auto visit(Visitor&& visitor, LSPRequest request) -> void {`);
  emit(`#define PROCESS_REQUEST_TYPE(NAME, METHOD) \\`);
  emit(`  if (request.method() == METHOD) \\`);
  emit(`    return visitor(static_cast<const NAME##Request&>(request));`);
  emit();
  emit(`#define PROCESS_NOTIFICATION_TYPE(NAME, METHOD) \\`);
  emit(`  if (request.method() == METHOD) \\`);
  emit(`    return visitor(static_cast<const NAME##Notification&>(request));`);
  emit();
  emit(`FOR_EACH_LSP_REQUEST_TYPE(PROCESS_REQUEST_TYPE)`);
  emit(`FOR_EACH_LSP_NOTIFICATION_TYPE(PROCESS_NOTIFICATION_TYPE)`);
  emit();
  emit(`#undef PROCESS_REQUEST_TYPE`);
  emit(`#undef PROCESS_NOTIFICATION_TYPE`);
  emit(`}`);

  emit();
  emit(`}`);

  const outputFile = path.join(outputDirectory, "requests.h");
  writeFileSync(outputFile, out);
}
