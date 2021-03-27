// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/symbols.h>

namespace cxx {

namespace {

inline bool is_set(LookupOptions options, LookupOptions flags) {
  return (options & flags) == flags;
}

}  // namespace

Scope::Scope() {}

Scope::~Scope() {}

Scope* Scope::enclosingScope() const {
  return owner_ ? owner_->scope() : nullptr;
}

Symbol* Scope::owner() const { return owner_; }

void Scope::setOwner(Symbol* owner) { owner_ = owner; }

void Scope::add(Symbol* symbol) { members_.push_back(symbol); }

LookupResult Scope::find(const Name* name, LookupOptions lookupOptions) const {
  LookupResult result;
  find(name, lookupOptions, result);
  return result;
}

LookupResult Scope::lookup(const Name* name,
                           LookupOptions lookupOptions) const {
  LookupResult result;
  std::vector<const Scope*> processed;
  processed.reserve(8);
  lookup(name, lookupOptions, processed, result);
  return result;
}

void Scope::find(const Name* name, LookupOptions lookupOptions,
                 LookupResult& result) const {
  for (auto it = rbegin(); it != rend(); ++it) {
    auto symbol = *it;

    if (symbol->name() == name && match(symbol, lookupOptions))
      result.push_back(symbol);
  }
}

void Scope::lookup(const Name* name, LookupOptions lookupOptions,
                   std::vector<const Scope*>& processed,
                   LookupResult& result) const {
  if (std::find(processed.begin(), processed.end(), this) != processed.end())
    return;

  processed.push_back(this);

  find(name, lookupOptions, result);

  if (auto ns = dynamic_cast<NamespaceSymbol*>(owner())) {
    for (const auto& u : ns->usingNamespaces())
      u->scope()->lookup(name, lookupOptions, processed, result);
  } else if (auto classSymbol = dynamic_cast<ClassSymbol*>(owner())) {
    for (const auto& base : classSymbol->baseClasses())
      base->scope()->lookup(name, lookupOptions, processed, result);
  }
}

bool Scope::match(Symbol* symbol, LookupOptions options) const {
  if (options == LookupOptions::kDefault) return true;

  if (is_set(options, LookupOptions::kNamespace) &&
      dynamic_cast<NamespaceSymbol*>(symbol) != nullptr)
    return true;

  if (is_set(options, LookupOptions::kType) && symbol->isTypeSymbol())
    return true;

  return false;
}

}  // namespace cxx
