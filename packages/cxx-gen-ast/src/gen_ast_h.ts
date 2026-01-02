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
import type { AST } from "./parseAST.ts";
import { groupNodesByBaseType } from "./groupNodesByBaseType.ts";
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
      emit(`  static constexpr ASTKind Kind = ASTKind::${enumName(name)};`);
      emit();
      emit(`  ${name}(): ${base}(Kind) {}`);
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
        `  void accept(ASTVisitor* visitor) override { visitor->visit(this); }`,
      );
      emit();
      emit(`  auto firstSourceLocation() -> SourceLocation override;`);
      emit(`  auto lastSourceLocation() -> SourceLocation override;`);
      emit(`};`);
      emit();
    });
  });

  emit(`
template <typename T>
[[nodiscard]] auto ast_cast(AST* ast) -> T* {
  return ast && ast->kind() == T::Kind ? static_cast<T*>(ast) : nullptr;
}
`);

  by_bases.forEach((nodes, base) => {
    if (base === "AST") return;
    if (!Array.isArray(nodes)) throw new Error("not an array");
    emit();
    const variantName = base.replace(/AST$/, "");
    emit(`template <typename Visitor>`);
    emit(`auto visit(Visitor&& visitor, ${base}* ast) {`);
    emit(`  switch (ast->kind()) {`);
    nodes.forEach(({ name }) => {
      emit(
        `  case ${name}::Kind: return std::invoke(std::forward<Visitor>(visitor), static_cast<${name}*>(ast));`,
      );
    });
    emit(`    default: cxx_runtime_error("unexpected ${variantName}");`);
    emit(`  } // switch`);
    emit(`}`);

    emit();
    emit(`template <>`);
    emit(`[[nodiscard]] inline auto ast_cast<${base}>(AST* ast) -> ${base}* {`);
    emit(`  if (!ast) return nullptr;`);
    emit(`  switch (ast->kind()) {`);
    nodes.forEach(({ name }) => {
      emit(`  case ${name}::Kind: `);
    });
    emit(`    return static_cast<${base}*>(ast);`);
    emit(`    default: return nullptr;`);
    emit(`  } // switch`);
    emit(`}`);
  });

  const out = `// Generated file by: gen_ast_h.ts
${cpy_header}
#pragma once

#include <cxx/arena.h>
#include <cxx/ast_fwd.h>
#include <cxx/ast_visitor.h>
#include <cxx/source_location.h>
#include <cxx/token.h>
#include <cxx/ast_kind.h>
#include <cxx/const_value.h>
#include <cxx/symbols_fwd.h>
#include <optional>
#include <ranges>

namespace cxx {

template <typename T>
class List final : public Managed {
public:
  T value;
  List* next;

  explicit List(const T& value, List* next = nullptr)
      : value(value), next(next) {}
};

template <typename T>
class ListIterator {
 public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;

  ListIterator() = default;
  explicit ListIterator(List<T>* list) : list_(list) {}

  auto operator<=>(const ListIterator&) const = default;

  auto operator*() const -> const T& { return list_->value; }

  auto operator++() -> ListIterator& {
    list_ = list_->next;
    return *this;
  }

  auto operator++(int) -> ListIterator {
    auto it = *this;
    ++*this;
    return it;
  }

 private:
  List<T>* list_{};
};

template <typename T>
class ListView : std::ranges::view_interface<ListView<T>> {
 public:
  explicit ListView(List<T>* list) : list_(list) {}

  auto begin() const { return ListIterator<T>(list_); }
  auto end() const { return ListIterator<T>(); }

 private:
  List<T>* list_;
};

template <typename T>
ListView(List<T>*) -> ListView<T>;

class AST : public Managed {
public:
  explicit AST(ASTKind kind): kind_(kind) {}

  virtual ~AST();

  [[nodiscard]] auto kind() const -> ASTKind { return kind_; }

  virtual void accept(ASTVisitor* visitor) = 0;

  virtual auto firstSourceLocation() -> SourceLocation = 0;
  virtual auto lastSourceLocation() -> SourceLocation = 0;

  [[nodiscard]] auto sourceLocationRange() -> SourceLocationRange {
      return SourceLocationRange(firstSourceLocation(), lastSourceLocation());
  }

private:
    ASTKind kind_;
};

template <typename T>
auto make_node(Arena* arena) -> T* {
  auto node = new (arena) T();
  return node;
}

template <typename T>
auto make_list_node(Arena* arena, T* element = nullptr) -> List<T*>* {
  auto list = new (arena) List<T*>(element);
  return list;
}

[[nodiscard]] inline auto firstSourceLocation(SourceLocation loc) -> SourceLocation { return loc; }

template <typename T>
[[nodiscard]] inline auto firstSourceLocation(T* node) -> SourceLocation {
    return node ? node->firstSourceLocation() : SourceLocation();
}

template <typename T>
[[nodiscard]] inline auto firstSourceLocation(List<T>* nodes) -> SourceLocation {
    for (auto node : ListView{nodes}) {
        if (auto loc = firstSourceLocation(node)) return loc;
    }
    return {};
}

[[nodiscard]] inline auto lastSourceLocation(SourceLocation loc) -> SourceLocation {
    return loc ? loc.next() : SourceLocation();
}

template <typename T>
[[nodiscard]] inline auto lastSourceLocation(T* node) -> SourceLocation {
    return node ? node->lastSourceLocation() : SourceLocation();
}

template <typename T>
[[nodiscard]] inline auto lastSourceLocation(List<T>* nodes) -> SourceLocation {
    if (!nodes) return {};
    if (auto loc = lastSourceLocation(nodes->next)) return loc;
    if (auto loc = lastSourceLocation(nodes->value)) return loc;
    return {};
}

${code.join("\n")}

[[nodiscard]] inline auto is_prvalue(ExpressionAST* expr) -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kPrValue;
}

[[nodiscard]] inline auto is_lvalue(ExpressionAST* expr) -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kLValue;
}

[[nodiscard]] inline auto is_xvalue(ExpressionAST* expr) -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kXValue;
}

[[nodiscard]] inline auto is_glvalue(ExpressionAST* expr) -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kLValue ||
         expr->valueCategory == ValueCategory::kXValue;
}

} // namespace cxx
`;

  fs.writeFileSync(output, out);
}
