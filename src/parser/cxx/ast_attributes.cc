// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/ast_attributes.h>

namespace cxx {

auto cloneAttribute(Arena* arena, const InitializerInfo& info)
    -> InitializerInfo {
  if (!info.arguments) return info;
  List<ExpressionAST*>* arguments = nullptr;
  auto tail = &arguments;
  for (auto argument : ListView{*info.arguments}) {
    *tail = make_list_node<ExpressionAST>(
        arena, argument ? argument->clone(arena) : nullptr);
    tail = &(*tail)->next;
  }
  return {.arguments = arguments,
          .isParenthesizedAggregate = info.isParenthesizedAggregate};
}

auto cloneAttribute(Arena* arena, const ObjectInitializationInfo& info)
    -> ObjectInitializationInfo {
  if (!info.expression) return info;
  return {.expression = info.expression->clone(arena)};
}

auto loweredSplice(ExpressionAST* expression) -> ExpressionAST* {
  if (auto splice = ast_cast<SpliceExpressionAST>(expression))
    return splice->splice.expression;
  if (auto splice = ast_cast<SpliceMemberExpressionAST>(expression))
    return splice->splice.expression;
  return expression;
}

auto cloneAttribute(Arena* arena, const SpliceExpressionInfo& info)
    -> SpliceExpressionInfo {
  if (!info.expression) return info;
  return {.expression = info.expression->clone(arena)};
}

}  // namespace cxx
