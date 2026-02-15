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

#include <cxx/name_lookup.h>

// cxx
#include <cxx/ast.h>
#include <cxx/const_value.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {

namespace {

struct AssociatedNamespaceCollector {
  std::vector<NamespaceSymbol*>& namespaces;
  std::vector<ClassSymbol*>& classes;
  std::vector<const Type*>& visited;

  void collect(const Type* type) {
    if (!type) return;
    if (std::ranges::contains(visited, type)) return;
    visited.push_back(type);
    visit(*this, type);
  }

  void addNamespace(NamespaceSymbol* ns) {
    if (ns && !std::ranges::contains(namespaces, ns)) namespaces.push_back(ns);
  }

  void addClass(ClassSymbol* cls) {
    if (cls && !std::ranges::contains(classes, cls)) classes.push_back(cls);
  }

  void collect(const Symbol* symbol) {
    if (!symbol) return;
    collect(symbol->type());
    addNamespace(symbol->enclosingNamespace());
  }

  void collect(const ConstValue& value) {
    if (auto object = std::get_if<std::shared_ptr<ConstObject>>(&value)) {
      if (*object) collect((*object)->type());
    }
  }

  void collect(const TemplateArgument& arg) {
    if (auto* argType = std::get_if<const Type*>(&arg)) {
      collect(*argType);
      return;
    }

    if (auto* argSymbol = std::get_if<Symbol*>(&arg)) {
      collect(*argSymbol);
      return;
    }

    if (auto* argValue = std::get_if<ConstValue>(&arg)) {
      collect(*argValue);
      return;
    }

    if (auto* argExpr = std::get_if<ExpressionAST*>(&arg)) {
      if (*argExpr) collect((*argExpr)->type);
      return;
    }
  }

  void operator()(const QualType* type) { collect(type->elementType()); }

  void operator()(const PointerType* type) { collect(type->elementType()); }
  void operator()(const LvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const RvalueReferenceType* type) {
    collect(type->elementType());
  }

  void operator()(const BoundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const UnboundedArrayType* type) {
    collect(type->elementType());
  }

  void operator()(const ClassType* type) {
    auto classSymbol = type->symbol();
    if (!classSymbol) return;
    if (auto def = classSymbol->definition()) classSymbol = def;
    if (std::ranges::contains(classes, classSymbol)) return;
    addClass(classSymbol);

    addNamespace(classSymbol->enclosingNamespace());

    for (const auto& base : classSymbol->baseClasses()) {
      if (auto baseClass = symbol_cast<ClassSymbol>(base->symbol())) {
        if (auto baseType = type_cast<ClassType>(baseClass->type())) {
          collect(baseType);
        }
      }
    }

    for (const auto& arg : classSymbol->templateArguments()) collect(arg);
  }

  void operator()(const EnumType* type) {
    if (auto sym = type->symbol()) addNamespace(sym->enclosingNamespace());
  }

  void operator()(const ScopedEnumType* type) {
    if (auto sym = type->symbol()) addNamespace(sym->enclosingNamespace());
  }

  void operator()(const FunctionType* type) {
    collect(type->returnType());
    for (auto paramType : type->parameterTypes()) collect(paramType);
  }

  void operator()(const MemberObjectPointerType* type) {
    collect(type->classType());
    collect(type->elementType());
  }

  void operator()(const MemberFunctionPointerType* type) {
    collect(type->classType());
    collect(type->functionType());
  }

  void operator()(const Type*) {}
};

}  // namespace

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

  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) {
    if (classSymbol->name() == id) {
      return classSymbol;
    }
  }

  for (auto candidate : scope->find(id)) {
    // Skip hidden friend declarations — they are only visible via ADL.
    if (candidate->isHidden()) continue;

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

auto Lookup::argumentDependentLookup(
    const Name* name, std::span<const Type* const> argumentTypes) const
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> result;

  if (!name) return result;

  std::vector<NamespaceSymbol*> namespaces;
  std::vector<ClassSymbol*> classes;
  std::vector<const Type*> visited;

  AssociatedNamespaceCollector collector{namespaces, classes, visited};

  for (auto argType : argumentTypes) {
    collector.collect(argType);
  }

  // Search for functions with the given name in each associated namespace
  auto addCandidate = [&](FunctionSymbol* func) {
    auto canonical = func->canonical();
    if (!std::ranges::contains(result, canonical)) {
      result.push_back(canonical);
    }
  };

  for (auto ns : namespaces) {
    for (auto symbol : ns->find(name)) {
      if (auto func = symbol_cast<FunctionSymbol>(symbol)) {
        addCandidate(func);
      } else if (auto ovl = symbol_cast<OverloadSetSymbol>(symbol)) {
        for (auto f : ovl->functions()) addCandidate(f);
      }
    }
  }

  return result;
}

}  // namespace cxx