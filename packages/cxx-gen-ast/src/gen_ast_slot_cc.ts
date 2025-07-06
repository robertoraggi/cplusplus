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
import {
  getAllMemberSlotNames,
  classifyMemberSlot,
} from "./getAllMemberSlotNames.ts";
import * as fs from "fs";

export function gen_ast_slot_cc({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const allMemberSlotNames = getAllMemberSlotNames({ ast });

  emit(`namespace {`);
  emit(`std::string_view kMemberSlotNames[] = {`);
  allMemberSlotNames.forEach((name, _nameIndex) => {
    emit(`  "${name}",`);
  });
  emit(`};`);
  emit(`} // namespace`);

  emit(`auto to_string(SlotNameIndex index) -> std::string_view {`);
  emit(`  return kMemberSlotNames[int(index)];`);
  emit(`}`);

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, members }) => {
      const memberSlots = members.filter(
        (m) => classifyMemberSlot(m) !== undefined,
      );

      emit();
      emit(`void ASTSlot::visit(${name}* ast) {`);
      if (memberSlots.length > 0) {
        emit(`  switch (slot_) {`);
        memberSlots.forEach((m, slotCount) => {
          const classification = classifyMemberSlot(m);
          const slotNameIndex = allMemberSlotNames.indexOf(m.name);

          switch (classification) {
            case "bool-attr":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = std::intptr_t(ast->${m.name} != 0);`);
              emit(`    slotKind_ = ASTSlotKind::kBoolAttribute;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "int-attr":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = ast->${m.name};`);
              emit(`    slotKind_ = ASTSlotKind::kIntAttribute;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "token-kind-attr":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = std::intptr_t(ast->${m.name});`);
              emit(`    slotKind_ = ASTSlotKind::kIntAttribute;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "identifier-attr":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`,
              );
              emit(`    slotKind_ = ASTSlotKind::kIdentifierAttribute;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "literal-attr":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`,
              );
              emit(`    slotKind_ = ASTSlotKind::kLiteralAttribute;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "token":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = ast->${m.name}.index();`);
              emit(`    slotKind_ = ASTSlotKind::kToken;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "node":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`,
              );
              emit(`    slotKind_ = ASTSlotKind::kNode;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "node-list":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`,
              );
              emit(`    slotKind_ = ASTSlotKind::kNodeList;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            case "token-list":
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = 0; // not implemented yet`);
              emit(`    slotKind_ = ASTSlotKind::kTokenList;`);
              emit(`    slotNameIndex_ = SlotNameIndex{${slotNameIndex}};`);
              emit(`    break;`);
              break;
            default:
              break;
          } // switch
        });
        emit(`  } // switch`);
        emit();
        emit(`  slotCount_ = ${memberSlots.length};`);
      }
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include <cxx/ast_slot.h>
#include <cxx/ast.h>
#include <algorithm>
#include <stdexcept>

namespace cxx {

auto ASTSlot::operator()(AST* ast, int slot) -> SlotInfo {
    std::intptr_t value = 0;
    ASTSlotKind slotKind = ASTSlotKind::kInvalid;
    SlotNameIndex slotNameIndex{};
    int slotCount = 0;
    if (ast) {
      std::swap(slot_, slot);
      std::swap(value_, value);
      std::swap(slotKind_, slotKind);
      std::swap(slotNameIndex_, slotNameIndex);
      std::swap(slotCount_, slotCount);
      ast->accept(this);
      std::swap(slot_, slot);
      std::swap(value_, value);
      std::swap(slotKind_, slotKind);
      std::swap(slotNameIndex_, slotNameIndex);
      std::swap(slotCount_, slotCount);
    }
    return {value, slotKind, SlotNameIndex{slotNameIndex}, slotCount};
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
