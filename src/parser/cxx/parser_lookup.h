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

#pragma once

#include <cxx/name_lookup.h>
#include <cxx/scope.h>

namespace cxx {

template <typename Predicate>
  requires std::predicate<Predicate, Symbol*>
[[nodiscard]] auto unqualifiedLookup(Scope* lexicalScope, const Name* name,
                                     Predicate accept) -> Symbol* {
  if (!name) return nullptr;
  std::vector<ScopeSymbol*> visited;
  for (auto sc = lexicalScope; sc; sc = sc->parent) {
    if (!sc->symbol) continue;
    if (auto s = detail::searchScope(sc->symbol, name, visited, accept))
      return s;
  }
  return nullptr;
}

[[nodiscard]] inline auto unqualifiedLookup(Scope* lexicalScope,
                                            const Name* name) -> Symbol* {
  return unqualifiedLookup(lexicalScope, name, [](Symbol*) { return true; });
}

[[nodiscard]] auto unqualifiedLookupType(Scope* lexicalScope,
                                         const Identifier* id) -> Symbol*;

[[nodiscard]] auto unqualifiedLookupNamespace(Scope* lexicalScope,
                                              const Identifier* id)
    -> NamespaceSymbol*;

}  // namespace cxx
