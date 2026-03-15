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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/translation_unit.h>

namespace cxx {

auto ASTInterpreter::evaluateBuiltinLine(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->tokenStartPosition(ast->firstSourceLocation());
  return ConstValue{static_cast<std::intmax_t>(pos.line)};
}

auto ASTInterpreter::evaluateBuiltinFile(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->tokenStartPosition(ast->firstSourceLocation());
  auto* lit = control()->stringLiteral(pos.fileName);
  return ConstValue{lit};
}

auto ASTInterpreter::evaluateBuiltinFunction(CallExpressionAST* /*ast*/)
    -> std::optional<ConstValue> {
  auto* lit = control()->stringLiteral(currentFunctionName_);
  return ConstValue{lit};
}

}  // namespace cxx
