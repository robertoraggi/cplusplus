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

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>

#include <optional>

namespace cxx {

class Parser;
class Control;

class ConstExpressionEvaluator {
  Parser& parser;

 public:
  explicit ConstExpressionEvaluator(Parser& parser) : parser(parser) {}

  auto evaluate(ExpressionAST* ast) -> std::optional<ConstValue>;

  auto control() const -> Control*;

  auto operator()(CharLiteralExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(BoolLiteralExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(IntLiteralExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(FloatLiteralExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(NullptrLiteralExpressionAST* ast)
      -> std::optional<ConstValue>;
  auto operator()(StringLiteralExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(UserDefinedStringLiteralExpressionAST* ast)
      -> std::optional<ConstValue>;
  auto operator()(ThisExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(NestedExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(IdExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(LambdaExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(FoldExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(RightFoldExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(LeftFoldExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(RequiresExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(VaArgExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(SubscriptExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(CallExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(TypeConstructionAST* ast) -> std::optional<ConstValue>;
  auto operator()(BracedTypeConstructionAST* ast) -> std::optional<ConstValue>;
  auto operator()(MemberExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(PostIncrExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(SpliceExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(GlobalScopeReflectExpressionAST* ast)
      -> std::optional<ConstValue>;
  auto operator()(NamespaceReflectExpressionAST* ast)
      -> std::optional<ConstValue>;
  auto operator()(TypeIdReflectExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(ReflectExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(CppCastExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(BuiltinBitCastExpressionAST* ast)
      -> std::optional<ConstValue>;
  auto operator()(TypeidExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(TypeidOfTypeExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(UnaryExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(AwaitExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(SizeofExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(SizeofTypeExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(SizeofPackExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(AlignofTypeExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(AlignofExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(NoexceptExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(NewExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(DeleteExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(CastExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(ImplicitCastExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(BinaryExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(ConditionalExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(YieldExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(ThrowExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(AssignmentExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(PackExpansionExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(DesignatedInitializerClauseAST* ast)
      -> std::optional<ConstValue>;
  auto operator()(TypeTraitsExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(ConditionExpressionAST* ast) -> std::optional<ConstValue>;
  auto operator()(EqualInitializerAST* ast) -> std::optional<ConstValue>;
  auto operator()(BracedInitListAST* ast) -> std::optional<ConstValue>;
  auto operator()(ParenInitializerAST* ast) -> std::optional<ConstValue>;
};

}  // namespace cxx