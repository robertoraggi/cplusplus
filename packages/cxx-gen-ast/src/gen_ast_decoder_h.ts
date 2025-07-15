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

import { groupNodesByBaseType } from "./groupNodesByBaseType.ts";
import type { AST } from "./parseAST.ts";
import { cpy_header } from "./cpy_header.ts";
import * as fs from "fs";

export function gen_ast_decoder_h({
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

  emit(`class ASTDecoder {`);
  emit(`public:`);
  emit(`  explicit ASTDecoder(TranslationUnit* unit);`);
  emit();
  emit(`  auto operator()(`);
  emit(`    std::span<const std::uint8_t> data) -> bool;`);
  emit();
  emit(`private:`);
  by_base.forEach((_nodes, base) => {
    if (base === "AST") return;
    const className = makeClassName(base);
    emit(
      `  auto decode${className}(const void* ptr, io::${className} type) -> ${base}*;`,
    );
  });
  emit();
  by_base.forEach((nodes) => {
    if (nodes.length === 0) return;
    nodes.forEach(({ name }) => {
      const className = makeClassName(name);
      emit(
        `  auto decode${className}(const io::${className}* node) -> ${name}*;`,
      );
    });
    emit();
  });
  emit();
  emit(`private:`);
  emit(`  TranslationUnit* unit_ = nullptr;`);
  emit(`  Arena* pool_ = nullptr;`);
  emit(`};`);

  const out = `// Generated file by: gen_ast_decoder_h.ts
${cpy_header}
#pragma once

#include <cxx/ast_fwd.h>
#include <cxx-ast-flatbuffers/ast_generated.h>
#include <span>

namespace cxx {

class TranslationUnit;
class Control;
class Arena;

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
