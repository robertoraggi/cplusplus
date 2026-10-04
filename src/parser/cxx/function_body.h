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

#pragma once

#include <cxx/ast.h>

namespace cxx {

[[nodiscard]] inline auto body_statement(FunctionBodyAST* body)
    -> CompoundStatementAST* {
  if (auto compound = ast_cast<CompoundStatementFunctionBodyAST>(body))
    return compound->statement;
  if (auto tryBody = ast_cast<TryStatementFunctionBodyAST>(body))
    return tryBody->statement;
  return nullptr;
}

[[nodiscard]] inline auto body_handlers(FunctionBodyAST* body)
    -> List<HandlerAST*>* {
  if (auto tryBody = ast_cast<TryStatementFunctionBodyAST>(body))
    return tryBody->handlerList;
  return nullptr;
}

[[nodiscard]] inline auto ctor_initializers(FunctionBodyAST* body)
    -> List<MemInitializerAST*>* {
  if (auto compound = ast_cast<CompoundStatementFunctionBodyAST>(body))
    return compound->memInitializerList;
  if (auto tryBody = ast_cast<TryStatementFunctionBodyAST>(body))
    return tryBody->memInitializerList;
  return nullptr;
}

inline void set_ctor_initializers(FunctionBodyAST* body,
                                  List<MemInitializerAST*>* initializers) {
  if (auto compound = ast_cast<CompoundStatementFunctionBodyAST>(body))
    compound->memInitializerList = initializers;
  else if (auto tryBody = ast_cast<TryStatementFunctionBodyAST>(body))
    tryBody->memInitializerList = initializers;
}

}  // namespace cxx
