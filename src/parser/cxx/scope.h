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
  auto operator=(const Scope& other) -> Scope& = delete;

  Scope();
  ~Scope();

  [[nodiscard]] auto enclosingScope() const -> Scope*;
  [[nodiscard]] auto skipTemplateScope() const -> Scope*;

  [[nodiscard]] auto isTemplateScope() const -> bool;

  [[nodiscard]] auto owner() const -> Symbol*;
  void setOwner(Symbol* owner);

  void add(Symbol* symbol);

  auto find(const Name* name,
            LookupOptions lookupOptions = LookupOptions::kDefault) const
      -> Symbol*;

  auto lookup(const Name* name,
              LookupOptions lookupOptions = LookupOptions::kDefault) const
      -> Symbol*;

  auto unqualifiedLookup(const Name* name, LookupOptions lookupOptions =
                                               LookupOptions::kDefault) const
      -> Symbol*;

  using iterator = std::vector<Symbol*>::const_iterator;

  [[nodiscard]] auto empty() const -> bool { return members_.empty(); }

  [[nodiscard]] auto begin() const { return members_.begin(); }
  [[nodiscard]] auto end() const { return members_.end(); }

  [[nodiscard]] auto rbegin() const { return members_.rbegin(); }
  [[nodiscard]] auto rend() const { return members_.rend(); }

 private:
  void rehash();

  void addHelper(Symbol* symbol);

  auto lookup(const Name* name, LookupOptions lookupOptions,
              std::vector<const Scope*>& processed) const -> Symbol*;

  auto match(Symbol* symbol, LookupOptions options) const -> bool;

 private:
  Symbol* owner_ = nullptr;
  std::vector<Symbol*> members_;
  std::vector<Symbol*> buckets_;
};

}  // namespace cxx
