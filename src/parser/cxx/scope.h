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

#include <cxx/names_fwd.h>
#include <cxx/symbols.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>
#include <cxx/views/symbol_chain.h>

#include <vector>

namespace cxx {

class SymbolChainView;

class Scope {
 public:
  explicit Scope(Scope* parent);
  ~Scope();

  [[nodiscard]] auto isEnumScope() const -> bool;
  [[nodiscard]] auto isTemplateParametersScope() const -> bool;
  [[nodiscard]] auto enclosingNonTemplateParametersScope() const -> Scope*;

  [[nodiscard]] auto parent() const -> Scope* { return parent_; }
  [[nodiscard]] auto owner() const -> ScopedSymbol* { return owner_; }

  [[nodiscard]] auto symbols() const { return std::views::all(symbols_); }

  [[nodiscard]] auto usingDirectives() const {
    return std::views::all(usingDirectives_);
  }

  [[nodiscard]] auto find(const Name* name) const -> SymbolChainView;

  void setParent(Scope* parent) { parent_ = parent; }
  void setOwner(ScopedSymbol* owner) { owner_ = owner; }

  void addSymbol(Symbol* symbol);
  void addUsingDirective(Scope* scope);

  void replaceSymbol(Symbol* symbol, Symbol* newSymbol);

 private:
  void rehash();

 private:
  Scope* parent_ = nullptr;
  ScopedSymbol* owner_ = nullptr;
  std::vector<Symbol*> symbols_;
  std::vector<Symbol*> buckets_;
  std::vector<Scope*> usingDirectives_;
};

namespace views {

constexpr auto class_or_namespaces =
    std::views::filter(&Symbol::isClassOrNamespace) |
    std::views::transform(
        [](Symbol* s) { return static_cast<ScopedSymbol*>(s); });

constexpr auto enum_or_scoped_enums =
    std::views::filter(&Symbol::isEnumOrScopedEnum) |
    std::views::transform(
        [](Symbol* s) { return static_cast<ScopedSymbol*>(s); });

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

inline auto members(Scope* scope) { return std::views::all(scope->symbols()); }

inline auto members(ScopedSymbol* symbol) {
  return std::views::all(symbol->scope()->symbols());
}

inline auto find(Scope* scope, const Name* name) {
  return scope ? scope->find(name) : SymbolChainView{nullptr};
}

constexpr auto named_symbol = std::views::filter(&Symbol::name);

}  // namespace views

}  // namespace cxx
