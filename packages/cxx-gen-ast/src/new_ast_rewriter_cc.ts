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
    emit(`    ASTRewriter& rewrite;`);
    emit(
      `[[nodiscard]] auto translationUnit() const -> TranslationUnit* { return rewrite.unit_; }`
    );
    emit();
    emit(
      `[[nodiscard]] auto control() const -> Control* { return rewrite.control(); }`
    );
    emit(
      `[[nodiscard]] auto arena() const -> Arena* { return rewrite.arena(); }`
    );
    emit(
      `[[nodiscard]] auto rewriter() const -> ASTRewriter* { return &rewrite; }`
    );
    emit(
      `[[nodiscard]] auto binder() const -> Binder* { return &rewrite.binder_; }`
    );
    nodes.forEach(({ name }) => {
      emit();
      emit(`    [[nodiscard]] auto operator()(${name}* ast) -> ${base}*;`);
    });
    emit(`  };`);
  });

  const emitRewriterBody = ({
    name,
    members,
    visitor = "rewrite",
  }: {
    name: string;
    members: Member[];
    visitor?: string;
  }) => {
    const blockSymbol = members.find(
      (m) => m.kind === "attribute" && m.type === "BlockSymbol"
    );

    if (blockSymbol) {
      emit(`auto _ = Binder::ScopeGuard(binder());

  if (ast->${blockSymbol.name}) {
    copy->${blockSymbol.name} = binder()->enterBlock(ast->${blockSymbol.name}->location());
  }
`);
    }

    let typeAttr: Member | undefined;

    members.forEach((m) => {
      if (m === blockSymbol) return;
      if (m === typeAttr) return;

      switch (m.kind) {
        case "node": {
          if (isBase(m.type)) {
            emit(`copy->${m.name} = ${visitor}(ast->${m.name});`);

            switch (m.type) {
              case "DeclaratorAST":
                const specsAttr =
                  members.find((m) => m.name == "typeSpecifierList")?.name ??
                  members.find((m) => m.name == "declSpecifierList")?.name;
                if (specsAttr) {
                  emit();
                  emit(
                    `auto ${m.name}Decl = Decl{${specsAttr}Ctx, copy->${m.name}};`
                  );
                  emit(
                    `auto ${m.name}Type = getDeclaratorType(translationUnit(), copy->${m.name}, ${specsAttr}Ctx.type());`
                  );

                  typeAttr = members.find(
                    (m) => m.kind === "attribute" && m.name === "type"
                  );

                  if (typeAttr) {
                    emit(`copy->${typeAttr.name} = ${m.name}Type;`);
                  }
                }
                break;
              default:
                break;
            } // switch
          } else {
            emit(
              `copy->${m.name} = ast_cast<${m.type}>(${visitor}(ast->${m.name}));`
            );
          }
          break;
        }
        case "node-list": {
          emit();

          // check the base type has a context that must be passed
          switch (m.type) {
            case "SpecifierAST":
              emit(`  auto ${m.name}Ctx = DeclSpecs{rewriter()};`);
              break;

            default:
              break;
          } // switch

          emit(
            `for (auto ${m.name} = &copy->${m.name}; auto node : ListView{ast->${m.name}}) {`
          );

          switch (m.type) {
            case "InitDeclaratorAST":
              emit(`auto value = ${visitor}(node, declSpecifierListCtx);`);
              break;

            default:
              emit(`auto value = ${visitor}(node);`);
              break;
          } // switch

          if (isBase(m.type)) {
            emit(`*${m.name} = make_list_node(arena(), value);`);
          } else {
            emit(
              `*${m.name} = make_list_node(arena(), ast_cast<${m.type}>(value));`
            );
          }
          emit(`${m.name} = &(*${m.name})->next;`);

          // update the context if needed
          switch (m.type) {
            case "SpecifierAST":
              emit(`${m.name}Ctx.accept(value);`);
              break;

            default:
              break;
          } // switch

          emit(`    }`);

          // update the context if needed
          switch (m.type) {
            case "SpecifierAST":
              emit(`${m.name}Ctx.finish();`);
              break;

            default:
              break;
          } // switch

          emit();
          break;
        }
        case "attribute": {
          emit(`    copy->${m.name} = ast->${m.name};`);

          if (m.name == "symbol" && name == "NamedTypeSpecifierAST") {
            emit(`
  if (auto typeParameter = symbol_cast<TypeParameterSymbol>(copy->symbol)) {
    const auto& args = rewrite.templateArguments_;
    if (typeParameter && typeParameter->depth() == 0 &&
        typeParameter->index() < args.size()) {
      auto index = typeParameter->index();

      if (auto sym = std::get_if<Symbol*>(&args[index])) {
        copy->symbol = *sym;
      }
    }
  }
`);
          }

          if (m.name == "symbol" && name == "IdExpressionAST") {
            emit(`
  if (auto param = symbol_cast<NonTypeParameterSymbol>(copy->symbol);
      param && param->depth() == 0 &&
      param->index() < rewrite.templateArguments_.size()) {
    auto symbolPtr =
        std::get_if<Symbol*>(&rewrite.templateArguments_[param->index()]);

    if (!symbolPtr) {
      cxx_runtime_error("expected initializer for non-type template parameter");
    }

    copy->symbol = *symbolPtr;
    copy->type = copy->symbol->type();
  }`);
          }
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
        emit(`auto ASTRewriter::operator()(${base}* ast) -> ${base}* {`);
        emit(`  if (!ast) return {};`);
        emit(`  auto expr = visit(${chopAST(base)}Visitor{*this}, ast);`);
        emit(`  if (expr) typeChecker_->check(expr);`);
        emit(`  return expr;`);
        emit(`}`);
        break;

      case "SpecifierAST":
        emit(`auto ASTRewriter::operator()(${base}* ast) -> ${base}* {`);
        emit(`  if (!ast) return {};`);
        emit(`  auto specifier = visit(${chopAST(base)}Visitor{*this}, ast);`);
        emit(`  return specifier;`);
        emit(`}`);
        break;

      default:
        emit(`auto ASTRewriter::operator()(${base}* ast) -> ${base}* {`);
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
          `auto ASTRewriter::operator()(${name}* ast, const DeclSpecs& declSpecs) -> ${name}* {`
        );
        break;
      default:
        emit(`auto ASTRewriter::operator()(${name}* ast) -> ${name}* {`);
        break;
    } // switch

    emit(`  if (!ast) return {};`);
    emit();
    emit(`  auto copy = make_node<${name}>(arena());`);
    emit();
    emitRewriterBody({ name, members, visitor: "operator()" });
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
        `auto ASTRewriter::${className}Visitor::operator()(${name}* ast) -> ${base}* {`
      );

      if (name === "IdExpressionAST") {
        emit(`
if (auto param = symbol_cast<NonTypeParameterSymbol>(ast->symbol);
    param && param->depth() == 0 &&
    param->index() < rewrite.templateArguments_.size()) {
  auto symbolPtr =
      std::get_if<Symbol*>(&rewrite.templateArguments_[param->index()]);

  if (!symbolPtr) {
    cxx_runtime_error("expected initializer for non-type template parameter");
  }

  auto parameterPack = symbol_cast<ParameterPackSymbol>(*symbolPtr);

  if (parameterPack && parameterPack == rewrite.parameterPack_ &&
      rewrite.elementIndex_.has_value()) {
    auto idx = rewrite.elementIndex_.value();
    auto element = parameterPack->elements()[idx];
    if (auto var = symbol_cast<VariableSymbol>(element)) {
      return rewrite(var->initializer());
    }
  }
}
`);
      }
      if (name === "LeftFoldExpressionAST") {
        emit(`
  if (auto parameterPack = rewrite.getParameterPack(ast->expression)) {
    auto savedParameterPack = rewrite.parameterPack_;
    std::swap(rewrite.parameterPack_, parameterPack);

    std::vector<ExpressionAST*> instantiations;
    ExpressionAST* current = nullptr;

    int n = 0;
    for (auto element : rewrite.parameterPack_->elements()) {
      std::optional<int> index{n};
      std::swap(rewrite.elementIndex_, index);

      auto expression = rewrite(ast->expression);
      if (!current) {
        current = expression;
      } else {
        auto binop = make_node<BinaryExpressionAST>(arena());
        binop->valueCategory = current->valueCategory;
        binop->type = current->type;
        binop->leftExpression = current;
        binop->op = ast->op;
        binop->opLoc = ast->opLoc;
        binop->rightExpression = expression;
        current = binop;
      }

      std::swap(rewrite.elementIndex_, index);
      ++n;
    }

    std::swap(rewrite.parameterPack_, parameterPack);

    return current;
  }
`);
      }

      emit(`  auto copy = make_node<${name}>(arena());`);
      emit();
      ast.baseMembers.get(base)?.forEach((m) => {
        emit(`  copy->${m.name} = ast->${m.name};`);
      });
      emitRewriterBody({ name, members, visitor: "rewrite" });
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
#include <cxx/ast_cursor.h>

#include <format>

namespace cxx {

ASTRewriter::ASTRewriter(TypeChecker* typeChcker,
  const std::vector<TemplateArgument>& templateArguments)
: typeChecker_(typeChcker)
, unit_(typeChcker->translationUnit())
, templateArguments_(templateArguments)
, binder_(typeChcker->translationUnit()) {}

ASTRewriter::~ASTRewriter() {}

auto ASTRewriter::control() const -> Control* {
  return unit_->control();
}

auto ASTRewriter::arena() const -> Arena* {
  return unit_->arena();
}

auto ASTRewriter::restrictedToDeclarations() const -> bool {
  return restrictedToDeclarations_;
}

void ASTRewriter::setRestrictedToDeclarations(bool restrictedToDeclarations) {
  restrictedToDeclarations_ = restrictedToDeclarations;
}

auto ASTRewriter::getParameterPack(ExpressionAST* ast) -> ParameterPackSymbol* {
  for (auto cursor = ASTCursor{ast, {}}; cursor; ++cursor) {
    const auto& current = *cursor;
    if (!std::holds_alternative<AST*>(current.node)) continue;

    auto id = ast_cast<IdExpressionAST>(std::get<AST*>(current.node));
    if (!id) continue;

    auto param = symbol_cast<NonTypeParameterSymbol>(id->symbol);
    if (!param) continue;

    if (param->depth() != 0) continue;

    auto arg = templateArguments_[param->index()];
    auto argSymbol = std::get<Symbol*>(arg);

    auto parameterPack = symbol_cast<ParameterPackSymbol>(argSymbol);
    if (parameterPack) return parameterPack;
  }

  return nullptr;
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
