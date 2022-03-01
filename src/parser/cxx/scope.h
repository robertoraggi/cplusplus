// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/symbols_fwd.h>

#include <variant>
#include <vector>

namespace cxx {

class Scope final {
 public:
  Scope(const Scope& other) = delete;
  Scope& operator=(const Scope& other) = delete;

  Scope();
  ~Scope();

  Scope* enclosingScope() const;
  Scope* skipTemplateScope() const;

  bool isTemplateScope() const;

  Symbol* owner() const;
  void setOwner(Symbol* owner);

  void add(Symbol* symbol);

  Symbol* find(const Name* name,
               LookupOptions lookupOptions = LookupOptions::kDefault) const;

  Symbol* lookup(const Name* name,
                 LookupOptions lookupOptions = LookupOptions::kDefault) const;

  Symbol* unqualifiedLookup(
      const Name* name,
      LookupOptions lookupOptions = LookupOptions::kDefault) const;

  using iterator = std::vector<Symbol*>::const_iterator;

  bool empty() const { return members_.empty(); }

  auto begin() const { return members_.begin(); }
  auto end() const { return members_.end(); }

  auto rbegin() const { return members_.rbegin(); }
  auto rend() const { return members_.rend(); }

 private:
  void rehash();

  void addHelper(Symbol* symbol);

  Symbol* lookup(const Name* name, LookupOptions lookupOptions,
                 std::vector<const Scope*>& processed) const;

  bool match(Symbol* symbol, LookupOptions options) const;

 private:
  Symbol* owner_ = nullptr;
  std::vector<Symbol*> members_;
  std::vector<Symbol*> buckets_;
};

}  // namespace cxx
