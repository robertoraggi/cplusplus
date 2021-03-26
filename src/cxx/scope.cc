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

Scope::~Scope() {}

Scope* Scope::enclosingScope() const {
  return owner_ ? owner_->scope() : nullptr;
}

Symbol* Scope::owner() const { return owner_; }

void Scope::setOwner(Symbol* owner) { owner_ = owner; }

void Scope::add(Symbol* symbol) { members_.push_back(symbol); }

LookupResult Scope::find(const Name* name, LookupOptions options) const {
  LookupResult result;
  for (auto it = rbegin(); it != rend(); ++it) {
    auto symbol = *it;
    if (symbol->name() == name) result.push_back(symbol);
  }
  return result;
}

}  // namespace cxx
