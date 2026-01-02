// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

import { cpy_header } from "./cpy_header.ts";
import { groupNodesByBaseType } from "./groupNodesByBaseType.ts";
import type { AST } from "./parseAST.ts";
import * as fs from "fs";

export function new_ast_rewriter_h({
  ast,
  output,
}: {
  ast: AST;
  opName: string;
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

  emit();
  emit(`  // run on the base nodes`);
  by_base.forEach((_nodes, base) => {
    if (base === "AST") return;
    emit(`  [[nodiscard]] auto operator()(${base}* ast) -> ${base}*;`);
  });
  emit();
  emit(`  // run on the misc nodes`);
  by_base.get("AST")?.forEach(({ name }) => {
    switch (name) {
      case "InitDeclaratorAST":
        emit(
          `  [[nodiscard]] auto operator()(${name}* ast, const DeclSpecs& declSpecs) -> ${name}*;`,
        );
        break;
      default:
        emit(`  [[nodiscard]] auto operator()(${name}* ast) -> ${name}*;`);
        break;
    } // switch
  });

  emit();
  emit(`private:`);
  by_base.forEach((nodes, base) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    if (base === "AST") return;
    const className = chopAST(base);
    emit(`  struct ${className}Visitor;`);
  });

  const out = `// Generated file by: new_ast_rewriter_h.ts
${cpy_header}

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/binder.h>

#include <vector>

namespace cxx {

class TranslationUnit;
class TypeChecker;
class Control;
class Arena;

class ASTRewriter {
public:
  explicit ASTRewriter(TypeChecker* typeChecker, const std::vector<TemplateArgument>& templateArguments);
  ~ASTRewriter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* { return unit_; }

  [[nodiscard]] const std::vector<TemplateArgument>& templateArguments() const {
    return templateArguments_;
  }

  [[nodiscard]] auto typeChecker() const -> TypeChecker* {
    return typeChecker_;
  }

  [[nodiscard]] auto binder() -> Binder& { return binder_; }

  [[nodiscard]] auto control() const -> Control*;
  [[nodiscard]] auto arena() const -> Arena*;

  [[nodiscard]] auto restrictedToDeclarations() const -> bool;
  void setRestrictedToDeclarations(bool restrictedToDeclarations);

${code.join("\n")}

private:
  [[nodiscard]] auto rewriter() -> ASTRewriter* { return this; }

  [[nodiscard]] auto getParameterPack(ExpressionAST* ast) -> ParameterPackSymbol*;

  TypeChecker* typeChecker_ = nullptr;
  const std::vector<TemplateArgument>& templateArguments_;
  ParameterPackSymbol* parameterPack_ = nullptr;
  std::optional<int> elementIndex_;
  TranslationUnit* unit_ = nullptr;
  Binder binder_;
  bool restrictedToDeclarations_ = false;
};

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
