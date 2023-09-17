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
import { AST, Member } from "./parseAST.js";
import { cpy_header } from "./cpy_header.js";
import * as fs from "fs";

enum SlotClassification {
  BoolAttribute,
  TokenKindAttribute,
  IdentifierAttribute,
  LiteralAttribute,
  Token,
  Node,
  NodeList,
  TokenList,
}

function classifyMember(m: Member): SlotClassification | undefined {
  if (m.kind === "attribute" && m.type === "bool") {
    return SlotClassification.BoolAttribute;
  } else if (m.kind === "attribute" && m.type === "TokenKind") {
    return SlotClassification.TokenKindAttribute;
  } else if (m.kind === "attribute" && m.type === "Identifier") {
    return SlotClassification.IdentifierAttribute;
  } else if (m.kind === "attribute" && m.type.endsWith("Literal")) {
    return SlotClassification.LiteralAttribute;
  } else if (m.kind === "token") {
    return SlotClassification.Token;
  } else if (m.kind === "node") {
    return SlotClassification.Node;
  } else if (m.kind === "node-list") {
    return SlotClassification.NodeList;
  } else if (m.kind === "token-list") {
    return SlotClassification.TokenList;
  } else {
    return undefined;
  }
}

function getAllSlotMemberNames({ ast }: { ast: AST }): string[] {
  const allSlotMemberNameSet = new Set<string>();

  groupNodesByBaseType(ast).forEach((nodes) => {
    nodes.forEach(({ members }) => {
      members
        .filter((m) => classifyMember(m) !== undefined)
        .forEach(({ name }) => allSlotMemberNameSet.add(name));
    });
  });

  return Array.from(allSlotMemberNameSet).sort();
}

export function gen_ast_slot_cc({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const allSlotMemberNames = getAllSlotMemberNames({ ast });

  emit(`namespace {`);
  emit(`std::string_view kSlotMemberNames[] = {`);
  allSlotMemberNames.forEach((name, nameIndex) => {
    emit(`  "${name}",`);
  });
  emit(`};`);
  emit(`} // namespace`);

  emit(`std::string_view to_string(SlotNameIndex index) {`);
  emit(`  return kSlotMemberNames[int(index)];`);
  emit(`}`);

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, members }) => {
      const memberSlots = members.filter(
        (m) => classifyMember(m) !== undefined
      );

      emit();
      emit(`void ASTSlot::visit(${name}* ast) {`);
      if (memberSlots.length > 0) {
        emit(`  switch (slot_) {`);
        memberSlots.forEach((m, slotCount) => {
          const classification = classifyMember(m);

          switch (classification) {
            case SlotClassification.BoolAttribute:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = intptr_t(ast->${m.name} != 0);`);
              emit(`    slotKind_ = ASTSlotKind::kBoolAttribute;`);
              emit(`    break;`);
              break;
            case SlotClassification.TokenKindAttribute:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = intptr_t(ast->${m.name});`);
              emit(`    slotKind_ = ASTSlotKind::kIntAttribute;`);
              emit(`    break;`);
              break;
            case SlotClassification.IdentifierAttribute:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`
              );
              emit(`    slotKind_ = ASTSlotKind::kIdentifierAttribute;`);
              emit(`    break;`);
              break;
            case SlotClassification.LiteralAttribute:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`
              );
              emit(`    slotKind_ = ASTSlotKind::kLiteralAttribute;`);
              emit(`    break;`);
              break;
            case SlotClassification.Token:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = ast->${m.name}.index();`);
              emit(`    slotKind_ = ASTSlotKind::kToken;`);
              emit(`    break;`);
              break;
            case SlotClassification.Node:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`
              );
              emit(`    slotKind_ = ASTSlotKind::kNode;`);
              emit(`    break;`);
              break;
            case SlotClassification.NodeList:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(
                `    value_ = reinterpret_cast<std::intptr_t>(ast->${m.name});`
              );
              emit(`    slotKind_ = ASTSlotKind::kNodeList;`);
              emit(`    break;`);
              break;
            case SlotClassification.TokenList:
              emit(`  case ${slotCount}: // ${m.name}`);
              emit(`    value_ = 0; // not implemented yet`);
              emit(`    slotKind_ = ASTSlotKind::kTokenList;`);
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
