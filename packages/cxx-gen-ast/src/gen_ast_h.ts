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

import { cpy_header } from "./cpy_header.js";
import { AST } from "./parseAST.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import * as fs from "fs";

export function gen_ast_h({ ast, output }: { ast: AST; output: string }) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_bases = groupNodesByBaseType(ast);

  const baseClassNames = Array.from(by_bases.keys()).filter((b) => b !== "AST");
  baseClassNames.sort();

  for (const base of baseClassNames) {
    emit(`class ${base} : public AST {`);
    emit(`public:`);
    emit(`  using AST::AST;`);
    ast.baseMembers.get(base)?.forEach((m) => {
      let s = "  ";
      if (m.cv) s = `${s}${m.cv} `;
      s = `${s}${m.type}${m.ptrOps} ${m.name}`;
      if (m.initializer) s = `${s} = ${m.initializer}`;
      s = `${s};`;
      emit(s);
    });
    emit(`};`);
    emit();
  }

  const enumName = (name: string) => name.slice(0, -3);

  by_bases.forEach((nodes) => {
    nodes.forEach(({ name, base, members }) => {
      emit(`class ${name} final : public ${base} {`);
      emit(`public:`);
      emit(`  ${name}(): ${base}(ASTKind::${enumName(name)}) {}`);
      emit();

      members.forEach((m) => {
        switch (m.kind) {
          case "node":
            emit(`  ${m.type}* ${m.name} = nullptr;`);
            break;
          case "node-list":
            emit(`  List<${m.type}*>* ${m.name} = nullptr;`);
            break;
          case "token":
            emit(`  SourceLocation ${m.name};`);
            break;
          case "token-list":
            emit(`  List<SourceLocation>* ${m.name} = nullptr;`);
            break;
          case "attribute": {
            let s = "  ";
            if (m.cv) s = `${s}${m.cv} `;
            s = `${s}${m.type}${m.ptrOps} ${m.name}`;
            if (m.initializer) s = `${s} = ${m.initializer}`;
            s = `${s};`;
            emit(s);
            break;
          }
        }
      });

      if (members.length > 0) {
        emit();
      }

      emit(
        `  void accept(ASTVisitor* visitor) override { visitor->visit(this); }`
      );
      emit();
      emit(`  auto firstSourceLocation() -> SourceLocation override;`);
      emit(`  auto lastSourceLocation() -> SourceLocation override;`);
      emit(`};`);
      emit();
    });
  });

  const out = `${cpy_header}
#pragma once

#include <cxx/arena.h>
#include <cxx/ast_fwd.h>
#include <cxx/ast_visitor.h>
#include <cxx/source_location.h>
#include <cxx/qualified_type.h>
#include <cxx/token.h>
#include <cxx/ast_kind.h>
#include <cxx/types_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/const_value.h>
#include <optional>

namespace cxx {

template <typename T>
class List final : public Managed {
public:
    T value;
    List* next;

    explicit List(const T& value, List* next = nullptr)
        : value(value), next(next) {}
};

class AST : public Managed {
    ASTKind kind_;
    bool checked_ = false;

public:
    explicit AST(ASTKind kind): kind_(kind) {}

    virtual ~AST();

    [[nodiscard]] auto kind() const -> ASTKind { return kind_; }

    [[nodiscard]] auto checked() const -> bool { return checked_; }
    void setChecked(bool checked) { checked_ = checked; }

    virtual void accept(ASTVisitor* visitor) = 0;

    virtual auto firstSourceLocation() -> SourceLocation = 0;
    virtual auto lastSourceLocation() -> SourceLocation = 0;

    auto sourceLocationRange() -> SourceLocationRange {
        return SourceLocationRange(firstSourceLocation(), lastSourceLocation());
    }
};

inline auto firstSourceLocation(SourceLocation loc) -> SourceLocation { return loc; }

template <typename T>
inline auto firstSourceLocation(T* node) -> SourceLocation {
    return node ? node->firstSourceLocation() : SourceLocation();
}

template <typename T>
inline auto firstSourceLocation(List<T>* nodes) -> SourceLocation {
    for (auto it = nodes; it; it = it->next) {
        if (auto loc = firstSourceLocation(it->value)) return loc;
    }
    return {};
}

inline auto lastSourceLocation(SourceLocation loc) -> SourceLocation {
    return loc ? loc.next() : SourceLocation();
}

template <typename T>
inline auto lastSourceLocation(T* node) -> SourceLocation {
    return node ? node->lastSourceLocation() : SourceLocation();
}

template <typename T>
inline auto lastSourceLocation(List<T>* nodes) -> SourceLocation {
    if (!nodes) return {};
    if (auto loc = lastSourceLocation(nodes->next)) return loc;
    if (auto loc = lastSourceLocation(nodes->value)) return loc;
    return {};
}

${code.join("\n")}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
