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

export function gen_ast_pretty_printer_h({
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

  emit();
  emit(`// run on the base nodes`);
  by_base.forEach((_nodes, base) => {
    if (base === "AST") return;
    emit(`void operator()(${base}* ast);`);
  });
  emit();
  emit(`// run on the misc nodes`);
  by_base.get("AST")?.forEach(({ name }) => {
    emit(`void operator()(${name}* ast);`);
  });
  emit();
  emit(`private:`);
  emit(`// visitors`);
  by_base.forEach((nodes, base) => {
    if (!Array.isArray(nodes)) throw new Error("not an array");
    if (base === "AST") return;
    const className = chopAST(base);
    emit(`  struct ${className}Visitor;`);
  });

  const out = `// Generated file by: gen_ast_pretty_printer_h.ts
${cpy_header}

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/source_location.h>

#include <format>
#include <ostream>
#include <iterator>

namespace cxx {

class TranslationUnit;
class Control;

class ASTPrettyPrinter {
public:
  explicit ASTPrettyPrinter(TranslationUnit* unit, std::ostream& out);
  ~ASTPrettyPrinter();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* { return unit_; }

  [[nodiscard]] auto control() const -> Control*;

${code.join("\n")}

  template <typename... Args>
  void write(std::format_string<Args...> fmt, Args&&... args) {
    if (newline_) {
      std::format_to(output_, "\\n");
      if (depth_ > 0) std::format_to(output_, "{:{}}", "", depth_ * 2);
    } else if (space_) {
      std::format_to(output_, " ");
    }
    newline_ = false;
    space_ = false;
    keepSpace_ = false;
    std::format_to(output_, fmt, std::forward<Args>(args)...);
    space();
  }

  void writeToken(SourceLocation loc);

  void space();
  void nospace();
  void keepSpace();
  void newline();
  void nonewline();
  void indent();
  void unindent();

private:
  TranslationUnit* unit_ = nullptr;
  std::ostream_iterator<char> output_;
  int depth_ = 0;
  bool space_ = false;
  bool keepSpace_ = false;
  bool newline_ = false;
};

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
