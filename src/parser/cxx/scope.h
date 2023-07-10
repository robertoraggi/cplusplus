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

#pragma once

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <vector>

namespace cxx {

class Scope {
 public:
  using MemberIterator = std::vector<Symbol*>::const_iterator;

  Scope(const Scope& other) = delete;
  auto operator=(const Scope& other) -> Scope& = delete;

  explicit Scope(Scope* parent = nullptr);
  ~Scope();

  [[nodiscard]] auto empty() const -> bool;
  [[nodiscard]] auto begin() const -> MemberIterator;
  [[nodiscard]] auto end() const -> MemberIterator;

  [[nodiscard]] auto usings() const -> const std::vector<Scope*>&;
  [[nodiscard]] auto owner() const -> Symbol*;
  [[nodiscard]] auto parent() const -> Scope*;
  [[nodiscard]] auto isTemplateScope() const -> bool;

  [[nodiscard]] auto currentClassOrNamespaceScope() -> Scope*;
  [[nodiscard]] auto currentNonTemplateScope() -> Scope*;
  [[nodiscard]] auto enclosingClassOrNamespaceScope() -> Scope*;

  [[nodiscard]] auto currentNamespaceScope() -> Scope*;
  [[nodiscard]] auto enclosingNamespaceScope() -> Scope*;

  [[nodiscard]] auto get(const Name* name) const -> Symbol*;
  [[nodiscard]] auto getClass(const Name* name) const -> ClassSymbol*;

  void add(Symbol* symbol);
  void addUsing(Scope* scope);

 private:
  void rehash();
  void addHelper(Symbol* symbol);

 private:
  std::vector<Symbol*> symbols_;
  std::vector<Symbol*> buckets_;
  std::vector<Scope*> usings_;
  Symbol* owner_ = nullptr;
  Scope* parent_ = nullptr;
  bool isTemplateScope_ = false;
};

}  // namespace cxx
