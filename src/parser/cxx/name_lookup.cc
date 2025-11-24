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
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {

Lookup::Lookup(ScopeSymbol* scope) : scope_(scope) {}

auto Lookup::lookupNamespace(NestedNameSpecifierAST* nestedNameSpecifier,
                             const Identifier* id) const -> NamespaceSymbol* {
  std::vector<ScopeSymbol*> set;

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

  return lookupNamespaceHelper(base, id, set);
}

auto Lookup::lookupNamespaceHelper(ScopeSymbol* scope, const Identifier* id,
                                   std::vector<ScopeSymbol*>& set) const
    -> NamespaceSymbol* {
  if (std::ranges::contains(set, scope)) {
    return nullptr;
  }

  set.push_back(scope);

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
  std::vector<ScopeSymbol*> set;

  if (!nestedNameSpecifier) {
    // unqualified lookup, start with the current scope and go up.
    for (auto scope = scope_; scope; scope = scope->parent()) {
      if (auto symbol = lookupTypeHelper(scope, id, set)) {
        return symbol;
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
      auto scopeSymbol = static_cast<ScopeSymbol*>(nestedNameSpecifier->symbol);
      return lookupTypeHelper(scopeSymbol, id, set);
    }

    case SymbolKind::kTypeAlias: {
      auto alias = symbol_cast<TypeAliasSymbol>(nestedNameSpecifier->symbol);
      auto classType = type_cast<ClassType>(alias->type());
      if (classType) {
        auto classSymbol = classType->symbol();
        return lookupTypeHelper(classSymbol, id, set);
      }
      return nullptr;
    }

    case SymbolKind::kUsingDeclaration: {
      auto usingDeclaration =
          symbol_cast<UsingDeclarationSymbol>(nestedNameSpecifier->symbol);

      if (!usingDeclaration->target()) return nullptr;

      if (auto classSymbol =
              symbol_cast<ClassSymbol>(usingDeclaration->target())) {
        return lookupTypeHelper(classSymbol, id, set);
      }

      if (auto enumSymbol =
              symbol_cast<EnumSymbol>(usingDeclaration->target())) {
        return lookupTypeHelper(enumSymbol, id, set);
      }

      if (auto scopedEnumSymbol =
              symbol_cast<ScopedEnumSymbol>(usingDeclaration->target())) {
        return lookupTypeHelper(scopedEnumSymbol, id, set);
      }

      return nullptr;
    }

    default:
      return nullptr;
  }  // swotch
}

auto Lookup::lookupTypeHelper(ScopeSymbol* scope, const Identifier* id,
                              std::vector<ScopeSymbol*>& set) const -> Symbol* {
  if (std::ranges::contains(set, scope)) {
    return nullptr;
  }

  set.push_back(scope);

  for (auto candidate : scope->find(id)) {
    if (auto u = symbol_cast<UsingDeclarationSymbol>(candidate);
        u && u->target()) {
      candidate = u->target();
    }

    if (is_type(candidate) || candidate->isNamespace()) {
      return candidate;
    }
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
    for (const auto& base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (auto ns = lookupTypeHelper(baseClass, id, set)) {
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