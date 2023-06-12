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

export function gen_ast_slot_cc({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, members }) => {
      emit();
      emit(`void ASTSlot::visit(${name}* ast) {`);
      emit(`  switch (slot_) {`);
      let slotCount = 0;
      members.forEach((m) => {
        if (m.kind === "token") {
          emit(`  case ${slotCount}:`);
          emit(`    value_ = ast->${m.name}.index();`);
          emit(`    slotKind_ = ASTSlotKind::kToken;`);
          emit(`    break;`);
          ++slotCount;
        } else if (m.kind === "node") {
          emit(`  case ${slotCount}:`);
          emit(`    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`);
          emit(`    slotKind_ = ASTSlotKind::kNode;`);
          emit(`    break;`);
          ++slotCount;
        } else if (m.kind === "node-list") {
          emit(`  case ${slotCount}:`);
          emit(`    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`);
          emit(`    slotKind_ = ASTSlotKind::kNodeList;`);
          emit(`    break;`);
          ++slotCount;
        } else if (m.kind === "token-list") {
          emit(`  case ${slotCount}:`);
          emit(`    cxx_runtime_error("not implemented yet");`);
          emit(`    slotKind_ = ASTSlotKind::kTokenList;`);
          emit(`    break;`);
          ++slotCount;
        }
      });
      emit(`  } // switch`);
      emit();
      emit(`  slotCount_ = ${slotCount};`);
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include <cxx/ast_slot.h>
#include <cxx/ast.h>
#include <algorithm>
#include <stdexcept>

namespace cxx {

auto ASTSlot::operator()(AST* ast, int slot) -> std::tuple<std::intptr_t, ASTSlotKind, int> {
    std::intptr_t value = 0;
    ASTSlotKind slotKind = ASTSlotKind::kInvalid;
    int slotCount = 0;
    if (ast) {
      std::swap(slot_, slot);
      std::swap(value_, value);
      std::swap(slotKind_, slotKind);
      std::swap(slotCount_, slotCount);
      ast->accept(this);
      std::swap(slotCount_, slotCount);
      std::swap(slotKind_, slotKind);
      std::swap(value_, value);
      std::swap(slot_, slot);
    }
    return std::tuple(value, slotKind, slotCount);
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
