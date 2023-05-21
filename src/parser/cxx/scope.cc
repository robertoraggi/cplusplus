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

#include <algorithm>

namespace cxx {

namespace {

std::hash<const Name*> hashName;

inline auto is_set(LookupOptions options, LookupOptions flags) -> bool {
  return (options & flags) == flags;
}

}  // namespace

Scope::Scope() = default;

Scope::~Scope() = default;

auto Scope::enclosingScope() const -> Scope* {
  return owner_ ? owner_->enclosingScope() : nullptr;
}

auto Scope::skipTemplateScope() const -> Scope* {
  auto scope = const_cast<Scope*>(this);
  while (scope && scope->isTemplateScope()) scope = scope->enclosingScope();
  return scope;
}

auto Scope::owner() const -> Symbol* { return owner_; }

void Scope::setOwner(Symbol* owner) { owner_ = owner; }

auto Scope::isTemplateScope() const -> bool {
  return dynamic_cast<TemplateParameterList*>(owner_);
}

void Scope::add(Symbol* symbol) {
  if (dynamic_cast<TemplateParameterList*>(symbol) ||
      dynamic_cast<TemplateTypeParameterSymbol*>(symbol)) {
    if (isTemplateScope()) addHelper(symbol);
  } else if (auto templateParameterList =
                 dynamic_cast<TemplateParameterList*>(owner_)) {
    symbol->setTemplateParameterList(templateParameterList);

    auto declarationScope = skipTemplateScope();
    declarationScope->addHelper(symbol);
  } else {
    addHelper(symbol);
  }
}

void Scope::addHelper(Symbol* symbol) {
  members_.push_back(symbol);
  if (3 * members_.size() > 2 * buckets_.size()) {
    rehash();
  } else {
    const auto h = hashName(symbol->name()) % buckets_.size();
    symbol->setNext(buckets_[h]);
    buckets_[h] = symbol;
  }
}

void Scope::rehash() {
  const auto bucketCount =
      std::max(buckets_.size() * 2, static_cast<std::size_t>(8));
  buckets_ = std::vector<Symbol*>(bucketCount, nullptr);
  for (auto member : members_) {
    const auto h = hashName(member->name()) % bucketCount;
    member->setNext(buckets_[h]);
    buckets_[h] = member;
  }
}

auto Scope::find(const Name* name, LookupOptions lookupOptions) const
    -> Symbol* {
  if (name && members_.size()) {
    const auto h = hashName(name) % buckets_.size();

    for (auto symbol = buckets_[h]; symbol; symbol = symbol->next()) {
      if (symbol->name() == name && match(symbol, lookupOptions)) return symbol;
    }
  }

  return nullptr;
}

auto Scope::lookup(const Name* name, LookupOptions lookupOptions) const
    -> Symbol* {
  std::vector<const Scope*> processed;
  return lookup(name, lookupOptions, processed);
}

auto Scope::unqualifiedLookup(const Name* name,
                              LookupOptions lookupOptions) const -> Symbol* {
  std::vector<const Scope*> processed;

  auto scope = this;

  while (scope) {
    if (auto symbol = scope->lookup(name, lookupOptions, processed)) {
      return symbol;
    }

    scope = scope->enclosingScope();
  }

  return nullptr;
}

auto Scope::lookup(const Name* name, LookupOptions lookupOptions,
                   std::vector<const Scope*>& processed) const -> Symbol* {
  if (std::find(processed.begin(), processed.end(), this) == processed.end()) {
    processed.push_back(this);

    if (auto symbol = find(name, lookupOptions)) return symbol;

    if (auto ns = dynamic_cast<NamespaceSymbol*>(owner())) {
      for (const auto& u : ns->usingNamespaces()) {
        if (auto symbol = u->scope()->lookup(name, lookupOptions, processed)) {
          return symbol;
        }
      }
    } else if (auto classSymbol = dynamic_cast<ClassSymbol*>(owner())) {
      for (const auto& base : classSymbol->baseClasses()) {
        if (auto symbol =
                base->scope()->lookup(name, lookupOptions, processed)) {
          return symbol;
        }
      }
    }
  }

  return nullptr;
}

auto Scope::match(Symbol* symbol, LookupOptions options) const -> bool {
  if (options == LookupOptions::kDefault) return true;

  if (is_set(options, LookupOptions::kNamespace) &&
      dynamic_cast<NamespaceSymbol*>(symbol) != nullptr) {
    return true;
  }

  if (is_set(options, LookupOptions::kType) && symbol->isTypeSymbol()) {
    return true;
  }

  if (is_set(options, LookupOptions::kTemplate) &&
      dynamic_cast<TemplateParameterList*>(symbol) != nullptr) {
    return true;
  }

  return false;
}

}  // namespace cxx
