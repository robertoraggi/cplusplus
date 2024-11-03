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

#include <cxx/ast.h>
#include <cxx/ast_cursor.h>
#include <cxx/ast_slot.h>

namespace cxx {

ASTCursor::ASTCursor(AST* root, std::string_view name) {
  stack_.push_back(Node{root, name});
}

ASTCursor::~ASTCursor() {}

void ASTCursor::step() {
  ASTSlot slotInfo;

  while (!stack_.empty()) {
    auto [node, kind] = stack_.back();
    stack_.pop_back();

    if (auto ast = std::get_if<AST*>(&node)) {
      if (!*ast) continue;

      const auto slotCount = slotInfo(*ast, 0).slotCount;

      std::vector<ASTSlot::SlotInfo> slots;
      slots.reserve(slotCount);

      for (int i = 0; i < slotCount; ++i) {
        auto childInfo = slotInfo(*ast, i);
        slots.push_back(childInfo);
      }

      for (const auto& slot : slots | std::views::reverse) {
        auto name = to_string(slot.nameIndex);
        if (slot.kind == ASTSlotKind::kNode) {
          auto node = reinterpret_cast<AST*>(slot.handle);
          stack_.push_back(Node{node, name});
        } else if (slot.kind == ASTSlotKind::kNodeList) {
          auto list = reinterpret_cast<List<AST*>*>(slot.handle);
          stack_.push_back(Node{list, name});
        }
      }

      break;
    } else if (auto list = std::get_if<List<AST*>*>(&node)) {
      if (!*list) continue;

      std::vector<AST*> children;

      for (auto ast : ListView{*list}) {
        if (ast) children.push_back(ast);
      }

      for (const auto& ast : children | std::views::reverse) {
        stack_.push_back(Node{ast, kind});
      }

      break;
    }
  }
}

}  // namespace cxx