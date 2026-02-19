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

#include <cxx/ast.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <algorithm>
#include <functional>
#include <span>
#include <vector>

namespace cxx {

namespace detail {

// searchScope: search a single scope (and its anonymous members, base classes,
// and using-directives) for a symbol matching `name` and `accept`.
// `visited` prevents infinite recursion through diamonds / cycles.
template <typename Predicate>
[[nodiscard]] auto searchScope(ScopeSymbol* scope, const Name* name,
                               std::vector<ScopeSymbol*>& visited,
                               Predicate accept) -> Symbol* {
  if (std::ranges::contains(visited, scope)) return nullptr;
  visited.push_back(scope);

  // 1. Direct members
  for (auto symbol : scope->find(name)) {
    if (symbol->isHidden()) continue;

    if (auto u = symbol_cast<UsingDeclarationSymbol>(symbol);
        u && u->target()) {
      if (std::invoke(accept, u->target())) return u->target();
    }

    if (std::invoke(accept, symbol)) return symbol;
  }

  // 2. Class scope: anonymous nested classes + base classes
  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
    for (auto member : classSymbol->find(/*unnamed=*/nullptr)) {
      auto nestedClass = symbol_cast<ClassSymbol>(member);
      if (!nestedClass) continue;
      if (auto s = searchScope(nestedClass, name, visited, accept)) return s;
    }

    for (const auto& base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (auto s = searchScope(baseClass, name, visited, accept)) return s;
    }
  }

  // 3. Using-directives
  for (auto u : scope->usingDirectives()) {
    if (auto s = searchScope(u, name, visited, accept)) return s;
  }

  return nullptr;
}

// resolveScope: resolve a Symbol* (possibly a TypeAlias or UsingDeclaration)
// to the underlying ScopeSymbol* for qualified lookup.
template <typename Predicate>
[[nodiscard]] auto resolveAndSearch(Symbol* scopeSymbol, const Name* name,
                                    Predicate accept) -> Symbol* {
  if (!scopeSymbol) return nullptr;

  if (auto alias = symbol_cast<TypeAliasSymbol>(scopeSymbol)) {
    if (auto ct = type_cast<ClassType>(alias->type()))
      return resolveAndSearch(ct->symbol(), name, accept);
    if (auto et = type_cast<EnumType>(alias->type()))
      return resolveAndSearch(et->symbol(), name, accept);
    if (auto st = type_cast<ScopedEnumType>(alias->type()))
      return resolveAndSearch(st->symbol(), name, accept);
  }

  switch (scopeSymbol->kind()) {
    case SymbolKind::kNamespace:
    case SymbolKind::kClass:
    case SymbolKind::kEnum:
    case SymbolKind::kScopedEnum: {
      std::vector<ScopeSymbol*> visited;
      return searchScope(scopeSymbol->asScopeSymbol(), name, visited, accept);
    }
    default:
      return nullptr;
  }
}

}  // namespace detail

template <typename Predicate>
  requires std::predicate<Predicate, Symbol*>
[[nodiscard]] auto unqualifiedLookup(ScopeSymbol* startScope, const Name* name,
                                     Predicate accept) -> Symbol* {
  if (!name) return nullptr;
  std::vector<ScopeSymbol*> visited;
  for (auto scope = startScope; scope; scope = scope->parent()) {
    if (auto s = detail::searchScope(scope, name, visited, accept)) return s;
  }
  return nullptr;
}

[[nodiscard]] inline auto unqualifiedLookup(ScopeSymbol* startScope,
                                            const Name* name) -> Symbol* {
  return unqualifiedLookup(startScope, name, [](Symbol*) { return true; });
}

template <typename Predicate>
  requires std::predicate<Predicate, Symbol*>
[[nodiscard]] auto qualifiedLookup(ScopeSymbol* scope, const Name* name,
                                   Predicate accept) -> Symbol* {
  if (!scope || !name) return nullptr;
  std::vector<ScopeSymbol*> visited;
  return detail::searchScope(scope, name, visited, accept);
}

[[nodiscard]] inline auto qualifiedLookup(ScopeSymbol* scope, const Name* name)
    -> Symbol* {
  return qualifiedLookup(scope, name, [](Symbol*) { return true; });
}

// Qualified lookup that resolves through TypeAlias / UsingDeclaration symbols.
template <typename Predicate>
  requires std::predicate<Predicate, Symbol*>
[[nodiscard]] auto qualifiedLookup(Symbol* scopeOrAlias, const Name* name,
                                   Predicate accept) -> Symbol* {
  if (!name) return nullptr;
  return detail::resolveAndSearch(scopeOrAlias, name, accept);
}

[[nodiscard]] inline auto qualifiedLookup(Symbol* scopeOrAlias,
                                          const Name* name) -> Symbol* {
  return qualifiedLookup(scopeOrAlias, name, [](Symbol*) { return true; });
}

template <typename Predicate>
  requires std::predicate<Predicate, Symbol*>
[[nodiscard]] auto lookupName(ScopeSymbol* startScope,
                              NestedNameSpecifierAST* nns, const Name* name,
                              Predicate accept) -> Symbol* {
  if (!name) return nullptr;
  if (!nns) return unqualifiedLookup(startScope, name, accept);
  if (!nns->symbol) return nullptr;
  return detail::resolveAndSearch(nns->symbol, name, accept);
}

[[nodiscard]] inline auto lookupName(ScopeSymbol* startScope,
                                     NestedNameSpecifierAST* nns,
                                     const Name* name) -> Symbol* {
  return lookupName(startScope, nns, name, [](Symbol*) { return true; });
}

[[nodiscard]] auto lookupType(ScopeSymbol* startScope,
                              NestedNameSpecifierAST* nns, const Identifier* id)
    -> Symbol*;

[[nodiscard]] auto lookupNamespace(ScopeSymbol* startScope,
                                   NestedNameSpecifierAST* nns,
                                   const Identifier* id) -> NamespaceSymbol*;

[[nodiscard]] auto argumentDependentLookup(
    const Name* name, std::span<const Type* const> argumentTypes)
    -> std::vector<FunctionSymbol*>;

}  // namespace cxx