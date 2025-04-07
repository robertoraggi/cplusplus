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

import { cpy_header } from "./cpy_header.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import { AST, Member } from "./parseAST.js";
import * as fs from "fs";

export function gen_ast_pretty_printer_cc({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  // chop the AST suffix for the given name
  const chopAST = (name: string) => {
    if (name.endsWith("AST")) return name.slice(0, -3);
    return name;
  };

  const isCommaSeparated = (m: Member, name: string) => {
    return (
      (m.name === "attributeList" && name === "CxxAttributeAST") ||
      [
        "baseSpecifierList",
        "bindingList",
        "captureList",
        "clobberList",
        "enumeratorList",
        "expressionList",
        "gotoLabelList",
        "initDeclaratorList",
        "initializerList",
        "inputOperandList",
        "memInitializerList",
        "outputOperandList",
        "parameterDeclarationList",
        "templateArgumentList",
        "templateParameterList",
        "typeIdList",
        "usingDeclaratorList",
      ].includes(m.name)
    );
  };

  by_base.forEach((nodes, base) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    if (base === "AST") return;
    const className = chopAST(base);
    emit();
    emit(`struct ASTPrettyPrinter::${className}Visitor {`);
    emit(`ASTPrettyPrinter& accept;`);
    emit(`[[nodiscard]] auto translationUnit() const -> TranslationUnit* {`);
    emit(`return accept.unit_;`);
    emit(`}`);
    emit(`void space() { accept.space(); }`);
    emit(`void nospace() { accept.nospace(); }`);
    emit(`void keepSpace() { accept.keepSpace(); }`);
    emit(`void newline() { accept.newline(); }`);
    emit(`void nonewline() { accept.nonewline(); }`);
    emit(`void indent() { accept.indent(); }`);
    emit(`void unindent() { accept.unindent(); }`);

    nodes.forEach(({ name }) => {
      emit();
      emit(`void operator()(${name}* ast);`);
    });
    emit(`};`);
  });

  const preToken = (m: Member, name: string) => {
    // pre token
    switch (m.name) {
      case "attributeLoc":
        if (["GccAttributeAST"].includes(name)) {
          emit(`newline();`);
        }
        break;

      case "opLoc":
        emit(`space();`);
        emit(`keepSpace();`);
        break;

      case "closeLoc":
      case "colonLoc":
      case "dotLoc":
      case "accessLoc":
      case "lbracketLoc":
      case "lbracket2Loc":
      case "lparen2Loc":
      case "lparenLoc":
      case "openLoc":
      case "rbracketLoc":
      case "rbracket2Loc":
      case "rparen2Loc":
      case "rparenLoc":
      case "greaterLoc":
      case "scopeLoc":
      case "commaLoc":
        emit(`nospace();`);
        break;

      case "semicolonLoc":
        emit(`nospace();`);
        emit(`nonewline();`);
        break;

      case "lbraceLoc":
      case "equalLoc":
        emit(`space();`);
        break;

      case "rbraceLoc":
        emit(`unindent();`);
        if (!["BracedInitListAST"].includes(name)) {
          emit(`newline();`);
        }
        break;

      default:
        break;
    } // switch
  };

  const postToken = (m: Member, name: string) => {
    switch (m.name) {
      case "lbraceLoc":
        emit(`indent();`);
        if (!["BracedInitListAST"].includes(name)) {
          emit(`newline();`);
        }
        break;

      case "equalLoc":
        emit(`keepSpace();`);
        break;

      case "opLoc":
        emit(`space();`);
        emit(`keepSpace();`);
        break;

      case "closeLoc":
      case "lbracketLoc":
      case "lbracket2Loc":
      case "dotLoc":
      case "lessLoc":
      case "lparenLoc":
      case "openLoc":
      case "scopeLoc":
      case "accessLoc":
        emit(`nospace();`);
        break;

      case "rbraceLoc":
        if (!["BracedInitListAST"].includes(name)) {
          emit(`newline();`);
        }
        break;

      case "semicolonLoc":
        if (!["ForStatementAST"].includes(name)) {
          emit(`newline();`);
        }
        break;

      case "greaterLoc":
        if (name === "TemplateDeclarationAST") emit(`newline();`);
        break;

      case "minusGreaterLoc":
        emit(`space();`);
        break;

      case "colonLoc":
        if (
          [
            "CaseStatementAST",
            "AccessDeclarationAST",
            "LabeledStatementAST",
          ].includes(name)
        ) {
          emit(`newline();`);
        }
        break;

      default:
        break;
    } // switch
  };

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    emit();
    emit(`void ASTPrettyPrinter::operator()(${base}* ast) {`);
    emit(`if (!ast) return;`);
    emit(`visit(${chopAST(base)}Visitor{*this}, ast);`);
    emit(`}`);
  });

  by_base.get("AST")?.forEach(({ name, members }) => {
    emit();
    emit(`void ASTPrettyPrinter::operator()(${name}* ast) {`);
    emit(`  if (!ast) return;`);
    emit();

    members.forEach((m) => {
      switch (m.kind) {
        case "node": {
          emit(`operator()(ast->${m.name});`);
          break;
        }

        case "token": {
          emit(`if (ast->${m.name}) {`);
          preToken(m, name);
          if (m.name === "opLoc") {
            emit(`write("{}", Token::spell(ast->op));`);
          } else {
            emit(`writeToken(ast->${m.name});`);
          }
          postToken(m, name);
          emit(`}`);
          break;
        }

        case "node-list": {
          emit();
          emit(`for (auto it = ast->${m.name}; it; it = it->next) {`);
          emit(`operator()(it->value);`);
          if (isCommaSeparated(m, name)) {
            emit(`if (it->next) { nospace(); write(","); }`);
          }
          emit(`}`);
          emit();
          break;
        }
      }
    });
    emit(`}`);
  });

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    if (!Array.isArray(nodes)) throw new Error("not an array");
    const className = chopAST(base);
    nodes.forEach(({ name, members }) => {
      emit();
      emit(
        `void ASTPrettyPrinter::${className}Visitor::operator()(${name}* ast) {`
      );

      members.forEach((m) => {
        switch (m.kind) {
          case "node": {
            if (
              m.name === "initalizer" &&
              [
                "IfStatementAST",
                "ForRangeStatementAST",
                "ForStatementAST",
              ].includes(name)
            ) {
              emit(`nonewline();`);
            }

            emit(`accept(ast->${m.name});`);
            break;
          }

          case "token": {
            emit(`if (ast->${m.name}) {`);

            preToken(m, name);

            if (m.name === "lparen2Loc" && name === "GccAttributeAST") {
              emit(`
nospace();
for (auto loc = ast->lparen2Loc; loc; loc = loc.next()) {
  if (loc == ast->rparenLoc) break;
  accept.writeToken(loc);
}`);
            } else if (m.name === "opLoc" && name === "OperatorFunctionIdAST") {
              emit(`
if (ast->op == TokenKind::T_NEW_ARRAY) {
  accept.write("new");
} else if (ast->op == TokenKind::T_DELETE_ARRAY) {
  accept.write("delete");
} else if (ast->op != TokenKind::T_LPAREN && ast->op != TokenKind::T_LBRACKET) {
  accept.write("{}", Token::spell(ast->op));
}
`);
            } else if (m.name === "opLoc") {
              emit(`accept.write("{}", Token::spell(ast->op));`);
            } else {
              emit(`accept.writeToken(ast->${m.name});`);

              if (m.name == "captureDefaultLoc") {
                emit(`if (ast->captureList) accept.write(",");`);
              }
            }

            // post token
            postToken(m, name);

            emit(`}`);

            break;
          }

          case "node-list": {
            switch (m.name) {
              case "outputOperandList":
              case "inputOperandList":
              case "clobberList":
              case "gotoLabelList":
                emit(`if (ast->${m.name}) accept.write(":");`);
                break;
              default:
                break;
            } // switch

            emit();
            emit(`for (auto it = ast->${m.name}; it; it = it->next) {`);
            emit(`accept(it->value);`);
            if (isCommaSeparated(m, name)) {
              if (["enumeratorList"].includes(m.name)) {
                emit(
                  `if (it->next) { nospace(); accept.write(","); newline(); }`
                );
              } else {
                emit(`if (it->next) { nospace(); accept.write(","); }`);
              }
            }
            emit(`}`);
            emit();
            break;
          }
        }
      });
      emit(`}`);
    });
  });

  const out = `${cpy_header}

#include <cxx/ast_pretty_printer.h>

// cxx
#include <cxx/ast.h>
#include <cxx/translation_unit.h>
#include <cxx/control.h>

namespace cxx {

${code.join("\n")}

ASTPrettyPrinter::ASTPrettyPrinter(TranslationUnit* unit, std::ostream& out)
: unit_(unit), output_(out) {}

ASTPrettyPrinter::~ASTPrettyPrinter() {}

auto ASTPrettyPrinter::control() const -> Control* {
    return unit_->control();
}

void ASTPrettyPrinter::space() {
  if (newline_) return;
  space_ = true;
}

void ASTPrettyPrinter::nospace() {
  if (keepSpace_) return;
  space_ = false;
}

void ASTPrettyPrinter::keepSpace() {
  keepSpace_ = true;
}

void ASTPrettyPrinter::newline() {
  space_ = false;
  keepSpace_ = false;
  newline_ = true;
}

void ASTPrettyPrinter::nonewline() {
  newline_ = false;
}

void ASTPrettyPrinter::indent() {
  ++depth_;
}

void ASTPrettyPrinter::unindent() {
  --depth_;
}

void ASTPrettyPrinter::writeToken(SourceLocation loc) {
  if (!loc) return;
  const auto& tk = unit_->tokenAt(loc);
  write("{}", tk.spell());
  if (!space_) cxx_runtime_error("no space");
}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
