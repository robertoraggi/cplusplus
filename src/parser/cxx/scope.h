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

#pragma once

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <map>
#include <ranges>
#include <vector>

namespace cxx {

class Scope {
 public:
  explicit Scope(Scope* parent = nullptr);
  ~Scope();

  [[nodiscard]] auto isEnumScope() const -> bool;
  [[nodiscard]] auto isTemplateParametersScope() const -> bool;

  [[nodiscard]] auto parent() const -> Scope* { return parent_; }
  void setParent(Scope* parent) { parent_ = parent; }

  [[nodiscard]] auto enclosingNonTemplateParametersScope() const -> Scope*;

  [[nodiscard]] auto owner() const -> Symbol* { return owner_; }
  void setOwner(Symbol* owner) { owner_ = owner; }

  [[nodiscard]] auto symbols() const { return symbols_ | std::views::values; }

  [[nodiscard]] auto get(const Name* name) const {
    auto [first, last] = symbols_.equal_range(name);
    return std::ranges::subrange(first, last) | std::views::values;
  }

  void addSymbol(Symbol* symbol);
  void removeSymbol(Symbol* symbol);

 private:
  Scope* parent_ = nullptr;
  Symbol* owner_ = nullptr;
  std::multimap<const Name*, Symbol*> symbols_;
};

}  // namespace cxx
