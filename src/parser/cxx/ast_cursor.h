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

#include <cxx/ast_fwd.h>

#include <deque>
#include <string_view>
#include <variant>

namespace cxx {

class ASTCursor {
 public:
  struct Node {
    std::variant<AST*, List<AST*>*> node;
    std::string_view name;
  };

  ASTCursor() = default;
  ~ASTCursor();

  ASTCursor(const ASTCursor&) = default;
  ASTCursor& operator=(const ASTCursor&) = default;

  ASTCursor(ASTCursor&&) = default;
  ASTCursor& operator=(ASTCursor&&) = default;

  ASTCursor(AST* root, std::string_view name);

  explicit operator bool() const { return !empty(); }

  [[nodiscard]] auto empty() const -> bool { return stack_.empty(); }

  [[nodiscard]] auto operator*() const -> const Node& { return stack_.back(); }

  void step();

  auto operator++() -> ASTCursor& {
    step();
    return *this;
  }

 private:
  std::deque<Node> stack_;
};

}  // namespace cxx