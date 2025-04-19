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

#include <cxx/scope.h>

// cxx
#include <cxx/names.h>
#include <cxx/symbols.h>

#include <algorithm>
#include <cassert>

namespace cxx {

Scope::Scope(Scope* parent) : parent_(parent) {}

Scope::~Scope() {}

void Scope::reset() {
  for (auto symbol : symbols_) {
    symbol->link_ = nullptr;
    symbol->setEnclosingScope(nullptr);
  }
  symbols_.clear();
  buckets_.clear();
  usingDirectives_.clear();
}

auto Scope::isTransparent() const -> bool {
  if (!owner_) return true;
  if (owner_->isTemplateParameters()) return true;
  if (owner_->isFunctionParameters()) return true;
  return false;
}

auto Scope::isNamespaceScope() const -> bool {
  return owner_ && owner_->isNamespace();
}

auto Scope::isClassScope() const -> bool { return owner_ && owner_->isClass(); }

auto Scope::isClassOrNamespaceScope() const -> bool {
  return isClassScope() || isNamespaceScope();
}

auto Scope::isFunctionScope() const -> bool {
  return owner_ && owner_->isFunction();
}

auto Scope::isBlockScope() const -> bool { return owner_ && owner_->isBlock(); }

auto Scope::isEnumScope() const -> bool {
  return owner_ && (owner_->isEnum() || owner_->isScopedEnum());
}

auto Scope::isTemplateParametersScope() const -> bool {
  return owner_ && owner_->isTemplateParameters();
}

auto Scope::enclosingNamespaceScope() const -> Scope* {
  for (auto scope = parent_; scope; scope = scope->parent()) {
    if (scope->isNamespaceScope()) {
      return scope;
    }
  }
  return nullptr;
}

auto Scope::enclosingNonTemplateParametersScope() const -> Scope* {
  auto scope = parent_;

  while (scope && scope->isTemplateParametersScope()) {
    scope = scope->parent();
  }

  return scope;
}

void Scope::addSymbol(Symbol* symbol) {
  if (symbol->isTemplateParameters()) {
    cxx_runtime_error("trying to add a template parameters symbol to a scope");
    return;
  }

  if (isTemplateParametersScope()) {
    if (!(symbol->isTypeParameter() || symbol->isTemplateTypeParameter() ||
          symbol->isNonTypeParameter() ||
          symbol->isConstraintTypeParameter())) {
      cxx_runtime_error("invalid symbol in template parameters scope");
    }
  }

  symbol->setEnclosingScope(this);
  symbols_.push_back(symbol);

  if (name_cast<ConversionFunctionId>(symbol->name())) {
    if (auto functionSymbol = symbol_cast<FunctionSymbol>(symbol)) {
      if (auto classSymbol = symbol_cast<ClassSymbol>(owner_)) {
        classSymbol->addConversionFunction(functionSymbol);
      }
    }
  }

  if (3 * symbols_.size() >= 2 * buckets_.size()) {
    rehash();
  } else {
    auto h = symbol->name() ? symbol->name()->hashValue() : 0;
    h = h % buckets_.size();
    symbol->link_ = buckets_[h];
    buckets_[h] = symbol;
  }
}

void Scope::rehash() {
  const auto newSize = std::max(std::size_t(8), buckets_.size() * 2);

  buckets_ = std::vector<Symbol*>(newSize);

  for (auto symbol : symbols_) {
    auto h = symbol->name() ? symbol->name()->hashValue() : 0;
    auto index = h % newSize;
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

  auto h = newSymbol->name() ? newSymbol->name()->hashValue() : 0;
  h = h % buckets_.size();

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

  symbol->link_ = nullptr;
}

void Scope::addUsingDirective(Scope* scope) {
  usingDirectives_.push_back(scope);
}

auto Scope::find(const Name* name) const -> SymbolChainView {
  if (!symbols_.empty()) {
    auto h = name ? name->hashValue() : 0;
    h = h % buckets_.size();
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      if (symbol->name() == name) {
        return SymbolChainView{symbol};
      }
    }
  }
  return SymbolChainView{nullptr};
}

auto Scope::find(TokenKind op) const -> SymbolChainView {
  if (!symbols_.empty()) {
    const auto h = OperatorId::hash(op) % buckets_.size();
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      auto id = name_cast<OperatorId>(symbol->name());
      if (id && id->op() == op) return SymbolChainView{symbol};
    }
  }
  return SymbolChainView{nullptr};
}

auto Scope::find(const std::string_view& name) const -> SymbolChainView {
  if (!symbols_.empty()) {
    const auto h = Identifier::hash(name) % buckets_.size();
    for (auto symbol = buckets_[h]; symbol; symbol = symbol->link_) {
      auto id = name_cast<Identifier>(symbol->name());
      if (id && id->name() == name) return SymbolChainView{symbol};
    }
  }
  return SymbolChainView{nullptr};
}

}  // namespace cxx
