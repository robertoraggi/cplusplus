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

export function gen_ast_dump_cc({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const toKebapName = (name: string) =>
    name.replace(/([A-Z]+)/g, "-$1").toLocaleLowerCase();

  const astName = (name: string) => toKebapName(name.slice(0, -3)).slice(1);

  const dumpMember = (member: Member) => {
    const fieldName = toKebapName(member.name);

    if (member.kind === "node-list") {
      emit(`  if (ast->${member.name}) {`);
      emit(`    ++indent_;`);
      emit(`    out_ << cxx::format("{:{}}", "", indent_ * 2);`);
      emit(`    out_ << cxx::format("{}\\n", "${fieldName}");`);
      emit(`    for (auto it = ast->${member.name}; it; it = it->next) {`);
      emit(`      accept(it->value);`);
      emit(`    }`);
      emit(`    --indent_;`);
      emit(`  }`);
    } else if (member.kind === "node") {
      emit(`  accept(ast->${member.name}, "${fieldName}");`);
    } else if (member.kind == "attribute" && member.type === "Identifier") {
      emit(`  accept(ast->${member.name}, "${fieldName}");`);
    } else if (member.kind == "attribute" && member.type === "bool") {
      emit(`  if (ast->${member.name}) {`);
      emit(`    ++indent_;`);
      emit(`    out_ << cxx::format("{:{}}", "", indent_ * 2);`);
      emit(
        `    out_ << cxx::format("${fieldName}: {}\\n", ast->${member.name});`
      );
      emit(`    --indent_;`);
      emit(`  }`);
    } else if (member.kind == "attribute" && member.type === "int") {
      emit(`  ++indent_;`);
      emit(`  out_ << cxx::format("{:{}}", "", indent_ * 2);`);
      emit(
        `  out_ << cxx::format("${fieldName}: {}\\n", ast->${member.name});`
      );
      emit(`  --indent_;`);
    } else if (member.kind == "attribute" && member.type.endsWith("Literal")) {
      emit(`  if (ast->${member.name}) {`);
      emit(`    ++indent_;`);
      emit(`    out_ << cxx::format("{:{}}", "", indent_ * 2);`);
      emit(
        `    out_ << cxx::format("${fieldName}: {}\\n", ast->${member.name}->value());`
      );
      emit(`    --indent_;`);
      emit(`  }`);
    } else if (
      member.kind == "attribute" &&
      member.type.endsWith("TokenKind")
    ) {
      emit(`  if (ast->${member.name} != TokenKind::T_EOF_SYMBOL) {`);
      emit(`    ++indent_;`);
      emit(`    out_ << cxx::format("{:{}}", "", indent_ * 2);`);
      emit(
        `    out_ << cxx::format("${fieldName}: {}\\n", Token::spell(ast->${member.name}));`
      );
      emit(`    --indent_;`);
      emit(`  }`);
    } else if (
      member.kind == "attribute" &&
      member.type == "ImplicitCastKind"
    ) {
      emit(`  ++indent_;`);
      emit(`  out_ << cxx::format("{:{}}", "", indent_ * 2);`);
      emit(
        `  out_ << cxx::format("${fieldName}: {}\\n", to_string(ast->${member.name}));`
      );
      emit(`  --indent_;`);
    }
  };

  by_base.forEach((nodes) => {
    nodes.forEach(({ name, base, members }) => {
      emit();
      emit(`void ASTPrinter::visit(${name}* ast) {`);
      emit(`  out_ << cxx::format("{}\\n", "${astName(name)}");`);

      const baseMembers = ast.baseMembers.get(base);

      baseMembers
        ?.filter((m) => m.kind === "attribute")
        ?.forEach((member) => dumpMember(member));

      members
        .filter((m) => m.kind === "attribute")
        .forEach((member) => dumpMember(member));

      baseMembers
        ?.filter((m) => m.kind !== "attribute")
        ?.forEach((member) => dumpMember(member));

      members
        .filter((m) => m.kind !== "attribute")
        .forEach((member) => dumpMember(member));
      emit(`}`);
    });
  });

  const out = `${cpy_header}
#include "ast_printer.h"

#include <cxx/ast.h>
#include <cxx/translation_unit.h>
#include <cxx/names.h>
#include <cxx/literals.h>
#include <cxx/private/format.h>

#include <algorithm>
#include <iostream>

namespace cxx {

ASTPrinter::ASTPrinter(TranslationUnit* unit, std::ostream& out)
  : unit_(unit)
  , out_(out) {}

void ASTPrinter::operator()(AST* ast) {
  accept(ast);
}

void ASTPrinter::accept(AST* ast, std::string_view field) {
  if (!ast) return;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  if (!field.empty()) {
    out_ << cxx::format("{}: ", field);
  }
  ast->accept(this);
  --indent_;
}

void ASTPrinter::accept(const Identifier* id, std::string_view field) {
  if (!id) return;
  ++indent_;
  out_ << cxx::format("{:{}}", "", indent_ * 2);
  if (!field.empty()) out_ << cxx::format("{}: ", field);
  out_ << cxx::format("{}\\n", id->value());
  --indent_;
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
