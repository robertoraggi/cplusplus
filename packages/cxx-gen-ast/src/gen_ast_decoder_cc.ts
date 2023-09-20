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

export function gen_ast_decoder_cc({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const makeClassName = (name: string) =>
    name != "AST" ? name.slice(0, -3) : name;

  const toSnakeName = (name: string) =>
    name.replace(/([A-Z])/g, "_$1").toLocaleLowerCase();

  by_base.forEach((nodes, base) => {
    const baseClassName = makeClassName(base);

    if (base === "AST") return;
    const className = makeClassName(base);
    emit();
    emit(
      `  auto ASTDecoder::decode${className}(const void* ptr, io::${className} type) -> ${base}* {`
    );
    emit(`  switch (type) {`);
    nodes.forEach(({ name }) => {
      const className = makeClassName(name);
      emit(`  case io::${baseClassName}_${className}:`);
      emit(
        `    return decode${className}(reinterpret_cast<const io::${className}*>(ptr));`
      );
    });
    emit(`  default:`);
    emit(`    return nullptr;`);
    emit(`  } // switch`);
    emit(`}`);
  });

  by_base.forEach((nodes) => {
    if (nodes.length === 0) return;
    nodes.forEach(({ name, members }) => {
      const className = makeClassName(name);
      emit();
      emit(
        `  auto ASTDecoder::decode${className}(const io::${className}* node) -> ${name}* {`
      );
      emit(`  if (!node) return nullptr;`);
      emit();
      emit(`  auto ast = new (pool_) ${name}();`);
      members.forEach((m) => {
        const snakeName = toSnakeName(m.name);
        if (m.kind === "node" && by_base.has(m.type)) {
          const baseClassName = makeClassName(m.type);
          emit(`  ast->${m.name} = decode${baseClassName}(`);
          emit(`    node->${snakeName}(), node->${snakeName}_type());`);
        } else if (m.kind === "node" && !by_base.has(m.type)) {
          const className = makeClassName(m.type);
          emit(`  ast->${m.name} = decode${className}(node->${snakeName}());`);
        } else if (m.kind === "node-list" && by_base.has(m.type)) {
          const className = makeClassName(m.type);
          emit(`  if (node->${snakeName}()) {`);
          emit(`    auto* inserter = &ast->${m.name};`);
          emit(`    for (std::size_t i = 0; i < node->${snakeName}()->size();`);
          emit(`         ++i) {`);
          emit(`    *inserter = new (pool_) List(decode${className}(`);
          emit(`      node->${snakeName}()->Get(i),`);
          emit(`      io::${className}(node->${snakeName}_type()->Get(i))));`);
          emit(`    inserter = &(*inserter)->next;`);
          emit(`  }`);
          emit(`}`);
        } else if (m.kind === "node-list" && !by_base.has(m.type)) {
          const className = makeClassName(m.type);
          emit(`  if (node->${snakeName}()) {`);
          emit(`    auto* inserter = &ast->${m.name};`);
          emit(`    for (std::size_t i = 0; i < node->${snakeName}()->size();`);
          emit(`         ++i) {`);
          emit(`    *inserter = new (pool_) List(decode${className}(`);
          emit(`      node->${snakeName}()->Get(i)));`);
          emit(`    inserter = &(*inserter)->next;`);
          emit(`  }`);
          emit(`}`);
        } else if (m.kind == "attribute" && m.type === "Identifier") {
          emit(`  if (node->${snakeName}()) {`);
          emit(`    ast->${m.name} = unit_->control()->getIdentifier(`);
          emit(`      node->${snakeName}()->str());`);
          emit(`  }`);
        } else if (m.kind == "attribute" && m.type === "CharLiteral") {
          emit(`  if (node->${snakeName}()) {`);
          emit(`    ast->${m.name} = unit_->control()->charLiteral(`);
          emit(`      node->${snakeName}()->str());`);
          emit(`  }`);
        } else if (m.kind == "attribute" && m.type === "StringLiteral") {
          emit(`  if (node->${snakeName}()) {`);
          emit(`    ast->${m.name} = unit_->control()->stringLiteral(`);
          emit(`      node->${snakeName}()->str());`);
          emit(`  }`);
        } else if (m.kind == "attribute" && m.type === "IntegerLiteral") {
          emit(`  if (node->${snakeName}()) {`);
          emit(`    ast->${m.name} = unit_->control()->integerLiteral(`);
          emit(`      node->${snakeName}()->str());`);
          emit(`  }`);
        } else if (m.kind == "attribute" && m.type === "FloatLiteral") {
          emit(`  if (node->${snakeName}()) {`);
          emit(`    ast->${m.name} = unit_->control()->floatLiteral(`);
          emit(`      node->${snakeName}()->str());`);
          emit(`  }`);
        } else if (m.kind == "attribute" && m.type === "TokenKind") {
          emit(`  ast->${m.name} = static_cast<TokenKind>(`);
          emit(`    node->${snakeName}());`);
        }
      });
      emit(`  return ast;`);
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include <cxx/private/ast_decoder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/translation_unit.h>
#include <cxx/control.h>

#include <algorithm>

namespace cxx {

ASTDecoder::ASTDecoder(TranslationUnit* unit)
  : unit_(unit), pool_(unit->arena()) {}

auto ASTDecoder::operator()(std::span<const std::uint8_t> bytes) -> bool {
  auto serializedUnit = io::GetSerializedUnit(bytes.data());

  if (auto file_name = serializedUnit->file_name()) {
    unit_->setSource(std::string(), file_name->str());
  }

  auto ast = decodeUnit(serializedUnit->unit(), serializedUnit->unit_type());
  unit_->setAST(ast);

  return true;
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
