// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import { AST } from "./parseAST.js";
import { cpy_header } from "./cpy_header.js";
import * as fs from "fs";

const json_vec = "std::vector<nlohmann::json>";

export function gen_ast_dump_cc({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const astName = (name: string) => name.slice(0, -3);

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, members }) => {
      emit();
      emit(`void ASTPrinter::visit(${name}* ast) {`);
      emit(`  json_ = nlohmann::json::array();`);
      emit();
      emit(`  json_.push_back("ast:${astName(name)}");`);
      emit();
      members.forEach((m) => {
        emit();
        if (m.kind === "node") {
          emit(`    if(ast->${m.name}) {`);
          emit(
            `      if (auto childNode = accept(ast->${m.name}); !childNode.is_null()) {`
          );
          emit(
            `        json_.push_back(${json_vec}{"attr:${m.name}", std::move(childNode)});`
          );
          emit(`      }`);
          emit(`    }`);
        } else if (m.kind === "node-list") {
          emit(`    if(ast->${m.name}) {`);
          emit(`      auto elements = nlohmann::json::array();`);
          emit(`      elements.push_back("array");`);
          emit(`      for (auto it = ast->${m.name}; it; it = it->next) {`);
          emit(
            `        if (auto childNode = accept(it->value); !childNode.is_null()) {`
          );
          emit(`          elements.push_back(std::move(childNode));`);
          emit(`        }`);
          emit(`      }`);
          emit(`      if (elements.size() > 1) {`);
          emit(
            `          json_.push_back(${json_vec}{"attr:${m.name}", elements});`
          );
          emit(`      }`);
          emit(`    }`);
        } else if (m.kind === "attribute" && m.type.endsWith("Literal")) {
          const tok = (s: string) => `${json_vec}{"literal", ${s}}`;
          const val = tok(`ast->${m.name}->value()`);
          emit(
            `    if (ast->${m.name}) { json_.push_back(${json_vec}{"attr:${m.name}", ${val}}); }`
          );
        } else if (m.kind === "attribute" && m.type === "Identifier") {
          const tok = (s: string) => `${json_vec}{"identifier", ${s}}`;
          const val = tok(`ast->${m.name}->name()`);
          emit(
            `    if (ast->${m.name}) { json_.push_back(${json_vec}{"attr:${m.name}", ${val}}); }`
          );
        } else if (m.kind === "attribute" && m.type === "TokenKind") {
          const tok = (s: string) => `${json_vec}{"token", ${s}}`;
          const val = tok(`Token::spell(ast->${m.name})`);
          emit(`    if (ast->${m.name} != TokenKind::T_EOF_SYMBOL) {`);
          emit(
            `        json_.push_back(${json_vec}{"attr:${m.name}", ${val}});`
          );
          emit(`    }`);
        }
      });
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include "ast_printer.h"

#include <cxx/ast.h>
#include <cxx/translation_unit.h>
#include <cxx/names.h>
#include <cxx/literals.h>

#include <algorithm>

namespace cxx {

  auto ASTPrinter::operator()(AST* ast, bool printLocations) -> nlohmann::json {
      std::vector<std::string_view> fileNames;
      std::swap(fileNames_, fileNames);
      std::swap(printLocations_, printLocations);
      auto result = accept(ast);
      std::swap(printLocations_, printLocations);
      std::swap(fileNames_, fileNames);
      result.push_back(std::vector<nlohmann::json>{"$files", std::move(fileNames)});
      return result;
  }

  auto ASTPrinter::accept(AST* ast) -> nlohmann::json {
    nlohmann::json json;

  if (ast) {
    std::swap(json_, json);
    ast->accept(this);
    std::swap(json_, json);

    if (!json.is_null() && printLocations_) {
      auto [startLoc, endLoc] = ast->sourceLocationRange();
      if (startLoc && endLoc) {
        unsigned startLine = 0, startColumn = 0;
        unsigned endLine = 0, endColumn = 0;
        std::string_view fileName, endFileName;

        unit_->getTokenStartPosition(startLoc, &startLine, &startColumn, &fileName);
        unit_->getTokenEndPosition(endLoc.previous(), &endLine, &endColumn, &endFileName);

        if (fileName == endFileName && !fileName.empty()) {
          auto it = std::find(begin(fileNames_), end(fileNames_), fileName);
          auto fileId = std::distance(begin(fileNames_), it);
          if (it == fileNames_.end()) fileNames_.push_back(fileName);
          json.push_back(std::vector<nlohmann::json>{"$range", fileId, startLine, startColumn, endLine, endColumn});
        }
      }
    }
  }

  return json;
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
