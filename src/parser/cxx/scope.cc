// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/scope.h>

// cxx
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <algorithm>
#include <cassert>
#include <functional>

namespace cxx {

namespace {

struct AddTemplateSymbol {
  TemplateParametersSymbol* templateParameters;

  auto addSymbol(auto symbol) {
    auto templateScope = templateParameters->scope();
    auto parentScope = templateScope->enclosingNonTemplateParametersScope();
    symbol->setTemplateParameters(templateParameters);
    parentScope->addSymbol(symbol);
  }

  auto operator()(TypeAliasSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(ConceptSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(ClassSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(FunctionSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(VariableSymbol* symbol) -> bool {
    addSymbol(symbol);
    return true;
  }

  auto operator()(auto) -> bool { return false; }
};

}  // namespace

Scope::Scope(Scope* parent) : parent_(parent) {}

Scope::~Scope() {}

auto Scope::isEnumScope() const -> bool {
  return owner_ && (owner_->isEnum() || owner_->isScopedEnum());
}

auto Scope::isTemplateParametersScope() const -> bool {
  return owner_ && owner_->isTemplateParameters();
}

auto Scope::enclosingNonTemplateParametersScope() const -> Scope* {
  auto scope = parent_;

  while (scope && scope->isTemplateParametersScope()) {
    scope = scope->parent();
  }

  return scope;
}

void Scope::addSymbol(Symbol* symbol) {
  if (symbol->isTemplateParameters()) return;

  if (auto templateParameters = symbol_cast<TemplateParametersSymbol>(owner_)) {
    if (visit(AddTemplateSymbol{templateParameters}, symbol)) return;
  }

  symbol->setEnclosingScope(this);
  symbols_.push_back(symbol);

  if (3 * symbols_.size() >= 2 * buckets_.size()) {
    rehash();
  } else {
    const auto h = std::hash<const void*>{}(symbol->name()) % buckets_.size();
    symbol->link_ = buckets_[h];
    buckets_[h] = symbol;
  }
}

void Scope::rehash() {
  const auto newSize = std::max(std::size_t(8), buckets_.size() * 2);

  buckets_ = std::vector<Symbol*>(newSize);

  std::hash<const void*> hash_value;

  for (auto symbol : symbols_) {
    auto index = hash_value(symbol->name()) % newSize;
    symbol->link_ = buckets_[index];
    buckets_[index] = symbol;
  }
}

void Scope::replaceSymbol(Symbol* symbol, Symbol* newSymbol) {
  if (symbol == newSymbol) return;

  assert(symbol->name() == newSymbol->name());

  auto it = std::find(symbols_.begin(), symbols_.end(), symbol);
  assert(it != symbols_.end());

  if (it == symbols_.end()) return;

  *it = newSymbol;

  newSymbol->link_ = symbol->link_;

  const auto h = std::hash<const void*>{}(newSymbol->name()) % buckets_.size();
  if (buckets_[h] == symbol) {
    buckets_[h] = newSymbol;
  } else {
    for (auto p = buckets_[h]; p; p = p->link_) {
      if (p->link_ == symbol) {
        p->link_ = newSymbol;
        break;
      }
    }
  }
}

void Scope::addUsingDirective(Scope* scope) {
  usingDirectives_.push_back(scope);
}

auto Scope::find(const Name* name) const -> SymbolChainView {
  if (!symbols_.empty()) {
    const auto h = std::hash<const void*>{}(name) % buckets_.size();
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      if (symbol->name() == name) {
        return SymbolChainView{symbol};
      }
    }
  }
  return SymbolChainView{nullptr};
}

}  // namespace cxx
