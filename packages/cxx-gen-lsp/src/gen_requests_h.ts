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
import { MetaModel, toCppType, Request, Notification, isRequest } from "./MetaModel.js";
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

  requestsAndNotifications.forEach((request) => {
    const typeName = request.typeName;
    emit();
    emit(`class ${typeName} final : public LSPRequest {`);
    emit(`public:`);
    emit(`  using LSPRequest::LSPRequest;`);
    emit();
    emit(`  [[nodiscard]] auto method() const -> std::string;`);
    emit(`  auto method(std::string method) -> ${typeName}&;`);
    emit();
    emit(`  [[nodiscard]] auto id() const -> std::variant<long, std::string>;`);
    emit(`  auto id(long id) -> ${typeName}&;`);
    emit(`  auto id(std::string id) -> ${typeName}&;`);

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
      emit();
      emit(`  [[nodiscard]] auto id() const -> std::variant<long, std::string>;`);
      emit(`  auto id(long id) -> ${responseTypeName}&;`);
      emit(`  auto id(std::string id) -> ${responseTypeName}&;`);
      emit();
      emit(`  [[nodiscard]] auto result() const -> ${resultType};`);
      emit();
      emit(`  auto result(${resultType} result) -> ${responseTypeName}&;`);
      emit(`};`);
    }
  });

  emit();
  emit(`}`);

  const outputFile = path.join(outputDirectory, "requests.h");
  writeFileSync(outputFile, out);
}
