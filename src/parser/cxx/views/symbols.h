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

#include <cxx/names_fwd.h>
#include <cxx/symbols.h>
#include <cxx/symbols_fwd.h>
#include <cxx/token_fwd.h>
#include <cxx/types.h>
#include <cxx/types_fwd.h>
#include <cxx/views/symbol_chain.h>

namespace cxx {

class SymbolChainView;

namespace views {

constexpr auto class_or_namespaces =
    std::views::filter(&Symbol::isClassOrNamespace) |
    std::views::transform(
        [](Symbol* s) { return static_cast<ScopeSymbol*>(s); });

constexpr auto enum_or_scoped_enums =
    std::views::filter(&Symbol::isEnumOrScopedEnum) |
    std::views::transform(
        [](Symbol* s) { return static_cast<ScopeSymbol*>(s); });

constexpr const auto namespaces =
    std::views::filter(&Symbol::isNamespace) |
    std::views::transform(symbol_cast<NamespaceSymbol>);

constexpr auto concepts = std::views::filter(&Symbol::isConcept) |
                          std::views::transform(symbol_cast<ConceptSymbol>);

constexpr auto classes = std::views::filter(&Symbol::isClass) |
                         std::views::transform(symbol_cast<ClassSymbol>);

constexpr auto enums = std::views::filter(&Symbol::isEnum) |
                       std::views::transform(symbol_cast<EnumSymbol>);

constexpr auto scoped_enums =
    std::views::filter(&Symbol::isScopedEnum) |
    std::views::transform(symbol_cast<ScopedEnumSymbol>);

constexpr auto functions = std::views::filter(&Symbol::isFunction) |
                           std::views::transform(symbol_cast<FunctionSymbol>);

constexpr auto variables = std::views::filter(&Symbol::isVariable) |
                           std::views::transform(symbol_cast<VariableSymbol>);

inline auto members(ScopeSymbol* symbol) {
  return std::views::all(symbol->members());
}

inline auto find(ScopeSymbol* scope, const Name* name) {
  return scope ? scope->find(name) : SymbolChainView{nullptr};
}

constexpr auto named_symbol = std::views::filter(&Symbol::name);

// Field views
constexpr auto fields = std::views::filter(&Symbol::isField) |
                        std::views::transform(symbol_cast<FieldSymbol>);

constexpr auto non_static_fields =
    fields | std::views::filter([](FieldSymbol* f) { return !f->isStatic(); });

constexpr auto static_fields =
    fields | std::views::filter(&FieldSymbol::isStatic);

// Member function views
constexpr auto member_functions =
    std::views::filter([](Symbol* s) {
      if (auto func = symbol_cast<FunctionSymbol>(s)) {
        return func->parent() && func->parent()->isClass();
      }
      return false;
    }) |
    std::views::transform(symbol_cast<FunctionSymbol>);

constexpr auto non_static_member_functions =
    member_functions |
    std::views::filter([](FunctionSymbol* f) { return !f->isStatic(); });

constexpr auto static_member_functions =
    member_functions | std::views::filter(&FunctionSymbol::isStatic);

constexpr auto virtual_functions =
    member_functions | std::views::filter(&FunctionSymbol::isVirtual);

constexpr auto constructors =
    member_functions | std::views::filter(&FunctionSymbol::isConstructor);

constexpr auto converting_constructors =
    constructors | std::views::filter([](FunctionSymbol* f) {
      if (f->isExplicit()) return false;
      auto funcType = type_cast<FunctionType>(f->type());
      if (!funcType) return false;
      return !funcType->parameterTypes().empty();
    });

inline auto overloads(Symbol* symbol)
    -> std::variant<std::ranges::ref_view<const std::vector<FunctionSymbol*>>,
                    std::ranges::single_view<FunctionSymbol*>,
                    std::ranges::empty_view<FunctionSymbol*>> {
  if (auto overloadSet = symbol_cast<OverloadSetSymbol>(symbol)) {
    return std::views::all(overloadSet->functions());
  }
  if (auto func = symbol_cast<FunctionSymbol>(symbol)) {
    return std::views::single(func);
  }
  return std::views::empty<FunctionSymbol*>;
}

}  // namespace views

}  // namespace cxx
