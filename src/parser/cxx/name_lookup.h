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
#include <cxx/ast_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace cxx {

class Lookup {
 public:
  explicit Lookup(ScopeSymbol* scope);

  template <typename Predicate>
    requires std::predicate<Predicate, Symbol*>
  [[nodiscard]] auto operator()(const Name* name, Predicate accept) const
      -> Symbol* {
    return lookup(nullptr, name, accept);
  }

  [[nodiscard]] auto operator()(const Name* name) const -> Symbol* {
    return operator()(name, [](Symbol*) { return true; });
  }

  template <typename Predicate>
    requires std::predicate<Predicate, Symbol*>
  [[nodiscard]] auto operator()(NestedNameSpecifierAST* nestedNameSpecifier,
                                const Name* name, Predicate accept) const
      -> Symbol* {
    return lookup(nestedNameSpecifier, name, accept);
  }

  [[nodiscard]] auto operator()(NestedNameSpecifierAST* nestedNameSpecifier,
                                const Name* name) const -> Symbol* {
    return operator()(nestedNameSpecifier, name, [](Symbol*) { return true; });
  }

  template <typename Predicate>
    requires std::predicate<Predicate, Symbol*>
  [[nodiscard]] auto lookup(NestedNameSpecifierAST* nestedNameSpecifier,
                            const Name* name, Predicate accept) const
      -> Symbol* {
    if (!name) return nullptr;
    if (!nestedNameSpecifier) return unqualifiedLookup(name, accept);
    if (!nestedNameSpecifier->symbol) return nullptr;
    return qualifiedLookup(nestedNameSpecifier->symbol, name, accept);
  }

  [[nodiscard]] auto lookup(NestedNameSpecifierAST* nestedNameSpecifier,
                            const Name* name) const -> Symbol* {
    return lookup(nestedNameSpecifier, name, [](Symbol*) { return true; });
  }

  template <typename Predicate>
    requires std::predicate<Predicate, Symbol*>
  [[nodiscard]] auto qualifiedLookup(ScopeSymbol* scope, const Name* name,
                                     Predicate accept) const -> Symbol* {
    std::vector<ScopeSymbol*> cache;
    return lookupHelper(scope, name, cache, accept);
  }

  [[nodiscard]] auto qualifiedLookup(ScopeSymbol* scope, const Name* name) const
      -> Symbol* {
    return qualifiedLookup(scope, name, [](Symbol*) { return true; });
  }

  [[nodiscard]] auto lookupNamespace(
      NestedNameSpecifierAST* nestedNameSpecifier, const Identifier* id) const
      -> NamespaceSymbol*;

  [[nodiscard]] auto lookupType(NestedNameSpecifierAST* nestedNameSpecifier,
                                const Identifier* id) const -> Symbol*;

  [[nodiscard]] auto argumentDependentLookup(
      const Name* name, std::span<const Type* const> argumentTypes) const
      -> std::vector<FunctionSymbol*>;

 private:
  template <typename Predicate>
  [[nodiscard]] auto unqualifiedLookup(const Name* name, Predicate accept) const
      -> Symbol* {
    std::vector<ScopeSymbol*> cache;
    for (auto current = scope_; current; current = current->parent()) {
      if (auto symbol = lookupHelper(current, name, cache, accept)) {
        return symbol;
      }
    }
    return nullptr;
  }

  template <typename Predicate>
  [[nodiscard]] auto qualifiedLookup(Symbol* scopeSymbol, const Name* name,
                                     Predicate accept) const -> Symbol* {
    if (!scopeSymbol) return nullptr;

    if (auto alias = symbol_cast<TypeAliasSymbol>(scopeSymbol)) {
      if (auto classType = type_cast<ClassType>(alias->type())) {
        return qualifiedLookup(classType->symbol(), name, accept);
      }
      if (auto enumType = type_cast<EnumType>(alias->type())) {
        return qualifiedLookup(enumType->symbol(), name, accept);
      }
      if (auto scopedEnumType = type_cast<ScopedEnumType>(alias->type())) {
        return qualifiedLookup(scopedEnumType->symbol(), name, accept);
      }
    }

    switch (scopeSymbol->kind()) {
      case SymbolKind::kNamespace:
      case SymbolKind::kClass:
      case SymbolKind::kEnum:
      case SymbolKind::kScopedEnum:
        return qualifiedLookup(scopeSymbol->asScopeSymbol(), name, accept);
      default:
        return nullptr;
    }  // switch
  }

  template <typename Predicate>
  [[nodiscard]] auto lookupHelper(ScopeSymbol* scope, const Name* name,
                                  std::vector<ScopeSymbol*>& cache,
                                  Predicate accept) const -> Symbol* {
    if (std::ranges::contains(cache, scope)) {
      return nullptr;
    }

    cache.push_back(scope);

    for (auto symbol : scope->find(name)) {
      if (symbol->isHidden()) continue;

      if (auto u = symbol_cast<UsingDeclarationSymbol>(symbol);
          u && u->target()) {
        if (std::invoke(accept, u->target())) {
          return u->target();
        }
      }

      if (std::invoke(accept, symbol)) {
        return symbol;
      }
    }

    if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
      // iterate over the anonymous symbols
      for (auto member : classSymbol->find(/*unnamed=*/nullptr)) {
        auto nestedClass = symbol_cast<ClassSymbol>(member);
        if (!nestedClass) continue;

        auto symbol = lookupHelper(nestedClass, name, cache, accept);
        if (symbol) {
          // found a match in an anonymous nested class
          return symbol;
        }
      }

      for (const auto& base : classSymbol->baseClasses()) {
        auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
        if (!baseClass) continue;
        if (auto symbol = lookupHelper(baseClass, name, cache, accept)) {
          return symbol;
        }
      }
    }

    for (auto u : scope->usingDirectives()) {
      if (auto symbol = lookupHelper(u, name, cache, accept)) {
        return symbol;
      }
    }

    return nullptr;
  }

  [[nodiscard]] auto lookupNamespaceHelper(ScopeSymbol* scope,
                                           const Identifier* id,
                                           std::vector<ScopeSymbol*>& set) const
      -> NamespaceSymbol*;

  [[nodiscard]] auto lookupTypeHelper(ScopeSymbol* scope, const Identifier* id,
                                      std::vector<ScopeSymbol*>& set) const
      -> Symbol*;

 private:
  ScopeSymbol* scope_ = nullptr;
};

}  // namespace cxx