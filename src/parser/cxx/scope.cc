// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>

#include <cassert>

namespace cxx {

Scope::Scope() = default;

Scope::~Scope() = default;

auto Scope::owner() const -> Symbol* { return owner_; }

auto Scope::empty() const -> bool { return symbols_.empty(); }

auto Scope::begin() const -> Scope::MemberIterator { return symbols_.begin(); }

auto Scope::end() const -> Scope::MemberIterator { return symbols_.end(); }

auto Scope::currentClassOrNamespaceScope() -> Scope* {
  if (owner_ && owner_->isClassOrNamespace()) {
    return this;
  }
  return enclosingClassOrNamespaceScope();
}

auto Scope::currentNonTemplateScope() -> Scope* {
  if (!isTemplateScope_) {
    return this;
  }
  return parent_->currentClassOrNamespaceScope();
}

auto Scope::enclosingClassOrNamespaceScope() -> Scope* {
  Scope* scope = parent_;
  for (; scope; scope = scope->parent_) {
    if (scope->owner_ && scope->owner_->isClassOrNamespace()) {
      return scope;
    }
  }
  return nullptr;
}

auto Scope::currentNamespaceScope() -> Scope* {
  if (owner_ && owner_->is(SymbolKind::kNamespace)) {
    return this;
  }
  return enclosingNamespaceScope();
}

auto Scope::enclosingNamespaceScope() -> Scope* {
  Scope* scope = parent_;
  for (; scope; scope = scope->parent_) {
    if (scope->owner_ && scope->owner_->is(SymbolKind::kNamespace)) {
      return scope;
    }
  }
  return nullptr;
}

void Scope::add(Symbol* symbol) {
  const auto isTemplateParameter =
      symbol->kind() == SymbolKind::kTemplateParameter ||
      symbol->kind() == SymbolKind::kTemplateParameterPack ||
      symbol->kind() == SymbolKind::kNonTypeTemplateParameter;

  if (isTemplateScope_ && !isTemplateParameter) {
    assert(!"cannot add symbol to the this scope");
  }

  assert(!symbol->enclosingScope());

  symbol->setEnclosingScope(this);

  addHelper(symbol);
}

void Scope::addUsing(Scope* scope) {
  auto it = std::find(usings_.begin(), usings_.end(), scope);

  if (it != usings_.end()) {
    return;
  }

  usings_.push_back(scope);
}

auto Scope::get(const Name* name) const -> Symbol* {
  for (auto it = rbegin(symbols_); it != rend(symbols_); ++it) {
    Symbol* symbol = *it;
    if (symbol->name() == name) {
      return symbol;
    }
  }
  return nullptr;
}

auto Scope::getClass(const Name* name) const -> ClassSymbol* {
  ClassSymbol* classSymbol = nullptr;

  for (Symbol* symbol = get(name); symbol; symbol = symbol->next) {
    if (symbol->name() != name) {
      continue;
    }

    classSymbol = symbol_cast<ClassSymbol>(symbol);

    if (classSymbol) {
      break;
    }
  }

  return classSymbol;
}

void Scope::rehash() {
  const auto bucketCount =
      std::max(buckets_.size() * 2, static_cast<std::size_t>(8));
  buckets_ = std::vector<Symbol*>(bucketCount, nullptr);
  std::hash<const Name*> hashValue;
  for (auto member : symbols_) {
    const auto h = hashValue(member->name()) % bucketCount;
    member->next = buckets_[h];
    buckets_[h] = member;
  }
}

void Scope::addHelper(Symbol* symbol) {
  symbols_.push_back(symbol);
  if (3 * symbols_.size() > 2 * buckets_.size()) {
    rehash();
  } else {
    std::hash<const Name*> hashValue;

    const auto h = hashValue(symbol->name()) % buckets_.size();
    symbol->next = buckets_[h];
    buckets_[h] = symbol;
  }
}

}  // namespace cxx
