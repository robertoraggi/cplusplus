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

export function gen_ast_encoder_h({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);
  by_base.set("AttributeAST", []);

  const makeClassName = (name: string) =>
    name != "AST" ? name.slice(0, -3) : name;

  emit(`class ASTEncoder : ASTVisitor {`);
  emit(`  template <typename T>`);
  emit(`  using Table = std::unordered_map<const T*,`);
  emit(`    flatbuffers::Offset<flatbuffers::String>>;`);
  emit();
  emit(`  TranslationUnit* unit_ = nullptr;`);
  emit(`  Table<Identifier> identifiers_;`);
  emit(`  Table<CharLiteral> charLiterals_;`);
  emit(`  Table<StringLiteral> stringLiterals_;`);
  emit(`  Table<IntegerLiteral> integerLiterals_;`);
  emit(`  Table<FloatLiteral> floatLiterals_;`);
  emit(`  flatbuffers::FlatBufferBuilder fbb_;`);
  emit(`  flatbuffers::Offset<> offset_;`);
  emit(`  std::uint32_t type_ = 0;`);
  emit();
  emit(`public:`);
  emit(`  explicit ASTEncoder() {}`);
  emit();
  emit(`  auto operator()(TranslationUnit* unit)`);
  emit(`    -> std::span<const std::uint8_t>;`);

  emit(`private:`);
  emit(`  auto accept(AST* ast) -> flatbuffers::Offset<>;`);
  by_base.forEach((_nodes, base) => {
    if (base === "AST") return;
    const className = makeClassName(base);
    emit();
    emit(`  auto accept${className}(${base}* ast)`);
    emit(`    -> std::tuple<flatbuffers::Offset<>, std::uint32_t>;`);
  });

  by_base.forEach((nodes) => {
    emit();
    nodes.forEach(({ name }) => {
      emit(`  void visit(${name}* ast) override;`);
    });
  });

  emit(`};`);

  const out = `${cpy_header}
#pragma once

#include <cxx/ast_visitor.h>
#include <cxx/names_fwd.h>
#include <cxx/literals_fwd.h>

#include <flatbuffers/flatbuffer_builder.h>
#include <tuple>
#include <span>
#include <unordered_map>

namespace cxx {

class TranslationUnit;

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
