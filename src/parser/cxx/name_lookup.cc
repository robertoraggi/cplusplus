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

#include <cxx/name_lookup.h>

// cxx
#include <cxx/ast.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

namespace cxx {

Lookup::Lookup(Scope* scope) : scope_(scope) {}

auto Lookup::unqualifiedLookup(const Name* name) const -> Symbol* {
  std::unordered_set<Scope*> cache;
  for (auto current = scope_; current; current = current->parent()) {
    if (auto symbol = lookupHelper(current, name, cache)) {
      return symbol;
    }
  }
  return nullptr;
}

auto Lookup::qualifiedLookup(Scope* scope, const Name* name) const -> Symbol* {
  std::unordered_set<Scope*> cache;
  return lookupHelper(scope, name, cache);
}

auto Lookup::qualifiedLookup(Symbol* scopedSymbol, const Name* name) const
    -> Symbol* {
  if (!scopedSymbol) return nullptr;
  switch (scopedSymbol->kind()) {
    case SymbolKind::kNamespace:
      return qualifiedLookup(
          symbol_cast<NamespaceSymbol>(scopedSymbol)->scope(), name);

    case SymbolKind::kClass:
      return qualifiedLookup(symbol_cast<ClassSymbol>(scopedSymbol)->scope(),
                             name);

    case SymbolKind::kEnum:
      return qualifiedLookup(symbol_cast<EnumSymbol>(scopedSymbol)->scope(),
                             name);

    case SymbolKind::kScopedEnum:
      return qualifiedLookup(
          symbol_cast<ScopedEnumSymbol>(scopedSymbol)->scope(), name);

    case SymbolKind::kTypeAlias: {
      auto alias = symbol_cast<TypeAliasSymbol>(scopedSymbol);

      if (auto classType = type_cast<ClassType>(alias->type())) {
        auto classSymbol = classType->symbol();
        return qualifiedLookup(classSymbol->scope(), name);
      }

      if (auto enumType = type_cast<EnumType>(alias->type())) {
        auto enumSymbol = enumType->symbol();
        return qualifiedLookup(enumSymbol->scope(), name);
      }

      if (auto scopedEnumType = type_cast<ScopedEnumType>(alias->type())) {
        auto scopedEnumSymbol = scopedEnumType->symbol();
        return qualifiedLookup(scopedEnumSymbol->scope(), name);
      }

      return nullptr;
    }

    default:
      return nullptr;
  }  // switch
}

auto Lookup::lookup(NestedNameSpecifierAST* nestedNameSpecifier,
                    const Name* name) const -> Symbol* {
  if (!name) return nullptr;
  if (!nestedNameSpecifier) return unqualifiedLookup(name);
  if (!nestedNameSpecifier->symbol) return nullptr;
  return qualifiedLookup(nestedNameSpecifier->symbol, name);
}

auto Lookup::lookupHelper(Scope* scope, const Name* name,
                          std::unordered_set<Scope*>& cache) const -> Symbol* {
  if (cache.contains(scope)) {
    return nullptr;
  }

  cache.insert(scope);

  for (auto symbol : scope->find(name)) {
    if (auto u = symbol_cast<UsingDeclarationSymbol>(symbol);
        u && u->target()) {
      return u->target();
    }

    return symbol;
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(scope->owner())) {
    for (const auto& base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (auto symbol = lookupHelper(baseClass->scope(), name, cache)) {
        return symbol;
      }
    }
  }

  for (auto u : scope->usingDirectives()) {
    if (auto symbol = lookupHelper(u, name, cache)) {
      return symbol;
    }
  }

  return nullptr;
}

auto Lookup::lookupNamespace(NestedNameSpecifierAST* nestedNameSpecifier,
                             const Identifier* id) const -> NamespaceSymbol* {
  std::unordered_set<Scope*> set;

  if (!nestedNameSpecifier) {
    // unqualified lookup, start with the current scope and go up.
    for (auto scope = scope_; scope; scope = scope->parent()) {
      if (auto ns = lookupNamespaceHelper(scope, id, set)) {
        return ns;
      }
    }

    return nullptr;
  }

  auto base = symbol_cast<NamespaceSymbol>(nestedNameSpecifier->symbol);

  if (!base) return nullptr;

  return lookupNamespaceHelper(base->scope(), id, set);
}

auto Lookup::lookupNamespaceHelper(Scope* scope, const Identifier* id,
                                   std::unordered_set<Scope*>& set) const
    -> NamespaceSymbol* {
  if (!set.insert(scope).second) {
    return nullptr;
  }

  for (auto candidate : scope->find(id) | views::namespaces) {
    return candidate;
  }

  for (auto u : scope->usingDirectives()) {
    if (auto ns = lookupNamespaceHelper(u, id, set)) {
      return ns;
    }
  }

  return nullptr;
}

auto Lookup::lookupType(NestedNameSpecifierAST* nestedNameSpecifier,
                        const Identifier* id) const -> Symbol* {
  std::unordered_set<Scope*> set;

  if (!nestedNameSpecifier) {
    // unqualified lookup, start with the current scope and go up.
    for (auto scope = scope_; scope; scope = scope->parent()) {
      if (auto ns = lookupTypeHelper(scope, id, set)) {
        return ns;
      }
    }

    return nullptr;
  }

  if (!nestedNameSpecifier->symbol) return nullptr;

  switch (nestedNameSpecifier->symbol->kind()) {
    case SymbolKind::kNamespace:
    case SymbolKind::kClass:
    case SymbolKind::kEnum:
    case SymbolKind::kScopedEnum: {
      auto scopedSymbol =
          static_cast<ScopedSymbol*>(nestedNameSpecifier->symbol);
      return lookupTypeHelper(scopedSymbol->scope(), id, set);
    }

    case SymbolKind::kTypeAlias: {
      auto alias = symbol_cast<TypeAliasSymbol>(nestedNameSpecifier->symbol);
      auto classType = type_cast<ClassType>(alias->type());
      if (classType) {
        auto classSymbol = classType->symbol();
        return lookupTypeHelper(classSymbol->scope(), id, set);
      }
      return nullptr;
    }

    case SymbolKind::kUsingDeclaration: {
      auto usingDeclaration =
          symbol_cast<UsingDeclarationSymbol>(nestedNameSpecifier->symbol);

      if (!usingDeclaration->target()) return nullptr;

      if (auto classSymbol =
              symbol_cast<ClassSymbol>(usingDeclaration->target())) {
        return lookupTypeHelper(classSymbol->scope(), id, set);
      }

      if (auto enumSymbol =
              symbol_cast<EnumSymbol>(usingDeclaration->target())) {
        return lookupTypeHelper(enumSymbol->scope(), id, set);
      }

      if (auto scopedEnumSymbol =
              symbol_cast<ScopedEnumSymbol>(usingDeclaration->target())) {
        return lookupTypeHelper(scopedEnumSymbol->scope(), id, set);
      }

      return nullptr;
    }

    default:
      return nullptr;
  }  // swotch
}

auto Lookup::lookupTypeHelper(Scope* scope, const Identifier* id,
                              std::unordered_set<Scope*>& set) const
    -> Symbol* {
  if (!set.insert(scope).second) {
    return nullptr;
  }

  for (auto candidate : scope->find(id)) {
    if (auto u = symbol_cast<UsingDeclarationSymbol>(candidate);
        u && u->target()) {
      candidate = u->target();
    }

    if (is_type(candidate) || candidate->isNamespace()) {
      return candidate;
    }
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(scope->owner())) {
    for (const auto& base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (auto ns = lookupTypeHelper(baseClass->scope(), id, set)) {
        return ns;
      }
    }
  }

  for (auto u : scope->usingDirectives()) {
    if (auto ns = lookupTypeHelper(u, id, set)) {
      return ns;
    }
  }

  return nullptr;
}

}  // namespace cxx