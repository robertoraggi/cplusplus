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

}  // namespace views

}  // namespace cxx
