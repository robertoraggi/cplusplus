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

#pragma once

#include <cxx/symbols_fwd.h>

#include <ranges>

namespace cxx {

class SymbolChainView : public std::ranges::view_interface<SymbolChainView> {
 public:
  explicit SymbolChainView(Symbol* symbol) : symbol_{symbol} {}

  auto begin() const { return Generator{symbol_}; }
  auto end() const { return std::default_sentinel; }

 private:
  class Generator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = Symbol*;

    explicit Generator(Symbol* symbol) : symbol_(symbol) {}

    auto operator*() const -> Symbol* { return symbol_; }

    auto operator++() -> Generator&;

    auto operator++(int) -> Generator {
      auto it = *this;
      ++*this;
      return it;
    }

    auto operator==(const std::default_sentinel_t&) const -> bool {
      return symbol_ == nullptr;
    }

   private:
    Symbol* symbol_ = nullptr;
  };

 private:
  Symbol* symbol_;
};

}  // namespace cxx