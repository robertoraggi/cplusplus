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

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <span>

namespace cxx {

class ASTInterpreter;

class Reflection {
 public:
  explicit Reflection(TranslationUnit* unit) : unit_(unit) {}

  void check(ExpressionAST* expression) const;
  [[nodiscard]] static auto isScopeQuery(WellKnownName builtin) -> bool;
  [[nodiscard]] static auto scope(ScopeSymbol* lexicalScope) -> ScopeSymbol*;
  [[nodiscard]] auto builtin(FunctionSymbol* function) const -> WellKnownName;
  [[nodiscard]] auto call(WellKnownName builtin,
                          std::span<const ConstValue> arguments,
                          ScopeSymbol* evaluationScope,
                          ASTInterpreter* interpreter = nullptr,
                          std::shared_ptr<ConstObject> object = {},
                          FunctionSymbol* function = nullptr) const
      -> std::optional<ConstValue>;
  void throwFailure(FunctionSymbol* function, ScopeSymbol* evaluationScope,
                    ASTInterpreter* interpreter) const;
  [[nodiscard]] auto parameters(const ConstValue& argument) const
      -> std::optional<std::vector<Meta>>;
  [[nodiscard]] auto evaluate(ExpressionAST* expression) const
      -> std::optional<ConstValue>;
  [[nodiscard]] static auto compare(TokenKind op, const ConstValue& left,
                                    const ConstValue& right)
      -> std::optional<ConstValue>;
  [[nodiscard]] static auto hash(const Meta& reflection) -> std::size_t;
  [[nodiscard]] static auto type(const Meta& reflection) -> const Type*;
  [[nodiscard]] static auto equal(const Meta& left, const Meta& right) -> bool;

 private:
  TranslationUnit* unit_;
};

}  // namespace cxx
