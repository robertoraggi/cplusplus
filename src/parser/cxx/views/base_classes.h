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

#pragma once

#include <cxx/symbols.h>
#include <memory.h>

#include <ranges>
#include <stack>

namespace cxx {

class BaseClassesView {
 public:
  explicit BaseClassesView(ClassSymbol* classSymbol) : root_(classSymbol) {}
  ~BaseClassesView() = default;

  auto begin() const { return Generator{root_}; }
  auto end() const { return std::default_sentinel; }

 private:
  class Generator {
   public:
    using value_type = BaseClassSymbol*;
    using difference_type = std::ptrdiff_t;

    explicit Generator(ClassSymbol* classSymbol) {
      if (classSymbol) {
        for (auto base : classSymbol->baseClasses() | std::views::reverse) {
          state_->stack.push(base);
        }
      }
    }

    auto operator*() const -> BaseClassSymbol* {
      if (state_->stack.empty()) return nullptr;
      return state_->stack.top();
    }

    auto operator++() -> Generator& {
      auto base = state_->stack.top();
      state_->stack.pop();

      if (auto classSymbol = symbol_cast<ClassSymbol>(base->symbol())) {
        if (!state_->visited.insert(classSymbol).second) {
          return *this;
        }

        for (auto base : classSymbol->baseClasses() | std::views::reverse) {
          state_->stack.push(base);
        }
      }

      return *this;
    }

    auto operator++(int) -> Generator {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    auto operator==(const std::default_sentinel_t&) const -> bool {
      return state_->stack.empty();
    }

   private:
    struct State {
      std::unordered_set<ClassSymbol*> visited;
      std::stack<BaseClassSymbol*> stack;
    };

    std::shared_ptr<State> state_ = std::make_shared<State>();
  };

 private:
  ClassSymbol* root_;
};

namespace views {

inline auto base_classes(ClassSymbol* classSymbol) -> BaseClassesView {
  return BaseClassesView{classSymbol};
}

}  // namespace views

}  // namespace cxx
