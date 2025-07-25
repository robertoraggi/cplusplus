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

export function gen_ast_slot_h({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  by_base.forEach((nodes) => {
    emit();
    nodes.forEach(({ name }) => {
      emit(`  void visit(${name}* ast) override;`);
    });
  });

  const out = `// Generated file by: gen_ast_slot_h.ts
${cpy_header}
#pragma once

#include <cxx/ast_visitor.h>
#include <cstdint>
#include <tuple>

namespace cxx {

enum ASTSlotKind {
  kInvalid,
  kToken,
  kNode,
  kTokenList,
  kNodeList,
  kIdentifierAttribute,
  kLiteralAttribute,
  kBoolAttribute,
  kIntAttribute,
};

enum class SlotNameIndex : int {};

class ASTSlot final : ASTVisitor {
public:
  struct SlotInfo {
    std::intptr_t handle;
    ASTSlotKind kind;
    SlotNameIndex nameIndex;
    int slotCount;
  };

  auto operator()(AST* ast, int slot) -> SlotInfo;

private:
${code.join("\n")}

private:
  std::intptr_t value_ = 0;
  int slot_ = 0;
  ASTSlotKind slotKind_ = ASTSlotKind::kInvalid;
  SlotNameIndex slotNameIndex_{};
  int slotCount_ = 0;
};

std::string_view to_string(SlotNameIndex index);

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
