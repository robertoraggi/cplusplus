// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#pragma once

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <ranges>
#include <vector>

namespace cxx {

class Scope {
 public:
  class MemberIterator {
   public:
    using value_type = Symbol*;
    using difference_type = std::ptrdiff_t;

    MemberIterator() = default;
    explicit MemberIterator(Symbol* symbol) : symbol_(symbol) {}

    auto operator<=>(const MemberIterator&) const = default;

    auto operator*() const -> Symbol* { return symbol_; }
    auto operator++() -> MemberIterator&;
    auto operator++(int) -> MemberIterator;

   private:
    Symbol* symbol_ = nullptr;
  };

  explicit Scope(Scope* parent = nullptr);
  ~Scope();

  [[nodiscard]] auto isEnumScope() const -> bool;
  [[nodiscard]] auto isTemplateParametersScope() const -> bool;

  [[nodiscard]] auto parent() const -> Scope* { return parent_; }
  void setParent(Scope* parent) { parent_ = parent; }

  [[nodiscard]] auto enclosingNonTemplateParametersScope() const -> Scope*;

  [[nodiscard]] auto owner() const -> ScopedSymbol* { return owner_; }
  void setOwner(ScopedSymbol* owner) { owner_ = owner; }

  [[nodiscard]] auto symbols() const -> const std::vector<Symbol*>& {
    return symbols_;
  }

  [[nodiscard]] auto get(const Name* name) const {
    auto [first, last] = getHelper(name);
    return std::ranges::subrange(first, last);
  }

  void addSymbol(Symbol* symbol);
  void replaceSymbol(Symbol* symbol, Symbol* newSymbol);

  [[nodiscard]] auto usingDirectives() const -> const std::vector<Scope*>&;

  void addUsingDirective(Scope* scope);

 private:
  [[nodiscard]] auto getHelper(const Name* name) const
      -> std::pair<MemberIterator, MemberIterator>;

  void rehash();

 private:
  Scope* parent_ = nullptr;
  ScopedSymbol* owner_ = nullptr;
  std::vector<Symbol*> symbols_;
  std::vector<Symbol*> buckets_;
  std::vector<Scope*> usingDirectives_;
};

}  // namespace cxx
