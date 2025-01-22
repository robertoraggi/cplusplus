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
import { AST } from "./parseAST.js";
import * as fs from "fs";

export function gen_ast_fwd_h({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_base = groupNodesByBaseType(ast);

  const baseClassNames = Array.from(by_base.keys()).filter((b) => b !== "AST");
  baseClassNames.sort();

  for (const base of baseClassNames) {
    emit(`class ${base};`);
  }

  emit();

  by_base.forEach((nodes, base) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    emit();
    emit(`  // ${base}`);
    nodes.forEach(({ name }) => {
      emit(`  class ${name};`);
    });
  });

  const out = `${cpy_header}

#pragma once

#include <string_view>

namespace cxx {

template <typename T>
class List;

class AST;

enum class ValueCategory  {
  kNone,
  kLValue,
  kXValue,
  kPrValue,
};

enum class ImplicitCastKind {
  kIdentity,
  kLValueToRValueConversion,
  kArrayToPointerConversion,
  kFunctionToPointerConversion,
  kIntegralPromotion,
  kFloatingPointPromotion,
  kIntegralConversion,
  kFloatingPointConversion,
  kFloatingIntegralConversion,
  kPointerConversion,
  kPointerToMemberConversion,
  kBooleanConversion,
  kFunctionPointerConversion,
  kQualificationConversion,
  kTemporaryMaterializationConversion,
  kUserDefinedConversion,
};

${code.join("\n")}

auto to_string(ValueCategory valueCategory) -> std::string_view;
auto to_string(ImplicitCastKind implicitCastKind) -> std::string_view;

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
