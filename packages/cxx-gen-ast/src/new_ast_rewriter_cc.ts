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

export function new_ast_rewriter_cc({
  ast,
  opName,
  opHeader,
  output,
}: {
  ast: AST;
  opName: string;
  opHeader: string;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const parent = new Map<string, string>();
  by_base.forEach((nodes, base) => {
    nodes.forEach(({ name }) => {
      parent.set(name, base);
    });
  });

  const isBase = (name: string) =>
    by_base.has(name) || parent.get(name) === "AST";

  // chop the AST suffix for the given name
  const chopAST = (name: string) => {
    if (name.endsWith("AST")) return name.slice(0, -3);
    return name;
  };

  by_base.forEach((nodes, base) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    if (base === "AST") return;
    const className = chopAST(base);
    emit();
    emit(`  struct ASTRewriter::${className}Visitor {`);
    emit(`    ${opName}& rewrite;`);
    emit(
      `    [[nodiscard]] auto translationUnit() const -> TranslationUnit* { return rewrite.unit_; }`
    );
    emit();
    emit(
      `    [[nodiscard]] auto control() const -> Control* { return rewrite.control(); }`
    );
    emit(
      `    [[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }`
    );
    nodes.forEach(({ name }) => {
      emit();
      emit(`    [[nodiscard]] auto operator()(${name}* ast) -> ${base}*;`);
    });
    emit(`  };`);
  });

  const emitRewriterBody = (members: Member[], visitor: string = "rewrite") => {
    const blockSymbol = members.find(
      (m) => m.kind === "attribute" && m.type === "BlockSymbol"
    );

    if (blockSymbol) {
      emit(`auto _ = Binder::ScopeGuard(&rewrite.binder_);

  if (ast->${blockSymbol.name}) {
    copy->${blockSymbol.name} = control()->newBlockSymbol(rewrite.binder_.scope(),
                                             ast->${blockSymbol.name}->location());

    rewrite.binder_.setScope(copy->${blockSymbol.name});
  }
`);
    }

    members.forEach((m) => {
      if (m === blockSymbol) return;

      switch (m.kind) {
        case "node": {
          if (isBase(m.type)) {
            emit(`  copy->${m.name} = ${visitor}(ast->${m.name});`);
          } else {
            emit(
              `  copy->${m.name} = ast_cast<${m.type}>(${visitor}(ast->${m.name}));`
            );
          }
          break;
        }
        case "node-list": {
          emit();

          // check the base type has a context that must be passed
          switch (m.type) {
            case "SpecifierAST":
              emit(`  DeclSpecs ${m.name}Ctx{translationUnit()};`);
              break;

            default:
              break;
          } // switch

          emit(`  if (auto it = ast->${m.name}) {`);
          emit(`    auto out = &copy->${m.name};`);

          emit(`    for (auto node : ListView{ast->${m.name}}) {`);

          switch (m.type) {
            case "InitDeclaratorAST":
              emit(
                `        auto value = ${visitor}(node, declSpecifierListCtx);`
              );
              break;

            default:
              emit(`        auto value = ${visitor}(node);`);
              break;
          } // switch

          if (isBase(m.type)) {
            emit(`*out = make_list_node(arena(), value);`);
          } else {
            emit(`*out = make_list_node(arena(), ast_cast<${m.type}>(value));`);
          }
          emit(`        out = &(*out)->next;`);

          // update the context if needed
          switch (m.type) {
            case "SpecifierAST":
              emit(`${m.name}Ctx.accept(value);`);
              break;

            default:
              break;
          } // switch

          emit(`    }`);
          emit(`  }`);
          emit();
          break;
        }
        case "attribute": {
          emit(`    copy->${m.name} = ast->${m.name};`);
          break;
        }
        case "token": {
          emit(`    copy->${m.name} = ast->${m.name};`);
          break;
        }
      }
    });
  };

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    emit();

    switch (base) {
      case "ExpressionAST":
        emit(`auto ${opName}::operator()(${base}* ast) -> ${base}* {`);
        emit(`  if (!ast) return {};`);
        emit(`  auto expr = visit(${chopAST(base)}Visitor{*this}, ast);`);
        emit(`  if (expr) typeChecker_->check(expr);`);
        emit(`  return expr;`);
        emit(`}`);
        break;

      case "SpecifierAST":
        emit(`auto ${opName}::operator()(${base}* ast) -> ${base}* {`);
        emit(`  if (!ast) return {};`);
        emit(`  auto specifier = visit(${chopAST(base)}Visitor{*this}, ast);`);
        emit(`  return specifier;`);
        emit(`}`);
        break;

      default:
        emit(`auto ${opName}::operator()(${base}* ast) -> ${base}* {`);
        emit(`  if (!ast) return {};`);
        emit(`  return visit(${chopAST(base)}Visitor{*this}, ast);`);
        emit(`}`);
        break;
    } // switch
  });

  by_base.get("AST")?.forEach(({ name, members }) => {
    emit();
    switch (name) {
      case "InitDeclaratorAST":
        emit(
          `auto ${opName}::operator()(${name}* ast, const DeclSpecs& declSpecs) -> ${name}* {`
        );
        break;
      default:
        emit(`auto ${opName}::operator()(${name}* ast) -> ${name}* {`);
        break;
    } // switch

    emit(`  if (!ast) return {};`);
    emit();
    emit(`  auto copy = make_node<${name}>(arena());`);
    emit();
    emitRewriterBody(members, "operator()");
    emit();
    emit(`  return copy;`);
    emit(`}`);
  });

  by_base.forEach((nodes, base) => {
    if (base === "AST") return;
    if (!Array.isArray(nodes)) throw new Error("not an array");
    const className = chopAST(base);
    nodes.forEach(({ name, members }) => {
      emit();
      emit(
        `auto ${opName}::${className}Visitor::operator()(${name}* ast) -> ${base}* {`
      );
      emit(`  auto copy = make_node<${name}>(arena());`);
      emit();
      ast.baseMembers.get(base)?.forEach((m) => {
        emit(`  copy->${m.name} = ast->${m.name};`);
      });
      emitRewriterBody(members, "rewrite");
      emit();
      emit(`  return copy;`);
      emit(`}`);
    });
  });

  const out = `${cpy_header}

#include <cxx/${opHeader}>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/decl_specs.h>
#include <cxx/decl.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/binder.h>

namespace cxx {

${opName}::${opName}(TypeChecker* typeChcker,
  const std::vector<TemplateArgument>& templateArguments)
: typeChecker_(typeChcker)
, unit_(typeChcker->translationUnit())
, templateArguments_(templateArguments)
, binder_(typeChcker->translationUnit()) {}

${opName}::~${opName}() {}

auto ${opName}::control() const -> Control* {
    return unit_->control();
}

auto ${opName}::arena() const -> Arena* {
    return unit_->arena();
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
