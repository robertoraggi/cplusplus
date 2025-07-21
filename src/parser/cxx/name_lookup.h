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

#pragma once

#include <cxx/ast_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>

#include <functional>
#include <unordered_set>

namespace cxx {

class Lookup {
 public:
  explicit Lookup(Scope* scope);

  [[nodiscard]] auto operator()(
      const Name* name, const std::function<bool(Symbol*)>& accept = {}) const
      -> Symbol*;

  [[nodiscard]] auto operator()(
      NestedNameSpecifierAST* nestedNameSpecifier, const Name* name,
      const std::function<bool(Symbol*)>& accept = {}) const -> Symbol*;

  [[nodiscard]] auto lookup(
      NestedNameSpecifierAST* nestedNameSpecifier, const Name* name,
      const std::function<bool(Symbol*)>& accept = {}) const -> Symbol*;

  [[nodiscard]] auto qualifiedLookup(
      Scope* scope, const Name* name,
      const std::function<bool(Symbol*)>& accept = {}) const -> Symbol*;

  [[nodiscard]] auto lookupNamespace(
      NestedNameSpecifierAST* nestedNameSpecifier, const Identifier* id) const
      -> NamespaceSymbol*;

  [[nodiscard]] auto lookupType(NestedNameSpecifierAST* nestedNameSpecifier,
                                const Identifier* id) const -> Symbol*;

 private:
  [[nodiscard]] auto unqualifiedLookup(
      const Name* name, const std::function<bool(Symbol*)>& accept) const
      -> Symbol*;

  [[nodiscard]] auto qualifiedLookup(
      Symbol* scopedSymbol, const Name* name,
      const std::function<bool(Symbol*)>& accept) const -> Symbol*;

  [[nodiscard]] auto lookupHelper(
      Scope* scope, const Name* name, std::unordered_set<Scope*>& cache,
      const std::function<bool(Symbol*)>& accept) const -> Symbol*;

  [[nodiscard]] auto lookupNamespaceHelper(
      Scope* scope, const Identifier* id, std::unordered_set<Scope*>& set) const
      -> NamespaceSymbol*;

  [[nodiscard]] auto lookupTypeHelper(Scope* scope, const Identifier* id,
                                      std::unordered_set<Scope*>& set) const
      -> Symbol*;

 private:
  Scope* scope_ = nullptr;
};

}  // namespace cxx