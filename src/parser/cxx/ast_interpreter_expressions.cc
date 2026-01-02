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
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

struct ASTInterpreter::ExpressionVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto unit() -> TranslationUnit* { return interp.unit_; }

  [[nodiscard]] auto control() -> Control* { return interp.control(); }

  [[nodiscard]] auto memoryLayout() -> MemoryLayout* {
    return control()->memoryLayout();
  }

  [[nodiscard]] auto evaluate(ExpressionAST* ast) -> ExpressionResult {
    return interp.expression(ast);
  }

  [[nodiscard]] auto toBool(const ConstValue& value) -> bool {
    return interp.toBool(value).value_or(false);
  }

  [[nodiscard]] auto toInt(const ConstValue& value) -> std::intmax_t {
    return interp.toInt(value).value_or(0);
  }

  [[nodiscard]] auto toInt32(const ConstValue& value) -> std::int32_t {
    return static_cast<std::int32_t>(toInt(value));
  }

  [[nodiscard]] auto toInt64(const ConstValue& value) -> std::int64_t {
    return static_cast<std::int64_t>(toInt(value));
  }

  [[nodiscard]] auto toUInt(const ConstValue& value) -> std::uintmax_t {
    return interp.toUInt(value).value_or(0);
  }

  [[nodiscard]] auto toUInt32(const ConstValue& value) -> std::uint32_t {
    return static_cast<std::uint32_t>(toUInt(value));
  }

  [[nodiscard]] auto toUInt64(const ConstValue& value) -> std::uint64_t {
    return static_cast<std::uint64_t>(toUInt(value));
  }

  [[nodiscard]] auto toFloat(const ConstValue& value) -> float {
    return interp.toFloat(value).value_or(0.0f);
  }

  [[nodiscard]] auto toDouble(const ConstValue& value) -> double {
    return interp.toDouble(value).value_or(0.0);
  }

  [[nodiscard]] auto toValue(std::uintmax_t value) -> ConstValue {
    return ConstValue(std::bit_cast<std::intmax_t>(value));
  }

  auto star_op(BinaryExpressionAST* ast, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type)) {
      return toDouble(*left) * toDouble(*right);
    }

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toValue(toUInt32(*left) * toUInt32(*right));
      return toValue(toUInt64(*left) * toUInt64(*right));
    }

    if (sz <= 4) return toValue(toInt32(*left) * toInt32(*right));
    return toValue(toInt64(*left) * toInt64(*right));
  }

  auto slash_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type)) {
      auto l = toDouble(*left);
      auto r = toDouble(*right);
      if (r == 0.0) return std::nullopt;
      return l / r;
    }

    if (control()->is_unsigned(type)) {
      if (sz <= 4) {
        auto l = toUInt32(*left);
        auto r = toUInt32(*right);
        if (r == 0) return std::nullopt;
        return toValue(l / r);
      }

      auto l = toUInt64(*left);
      auto r = toUInt64(*right);
      if (r == 0) return std::nullopt;
      return toValue(l / r);
    }

    if (sz <= 4) {
      auto l = toInt32(*left);
      auto r = toInt32(*right);
      if (r == 0) return std::nullopt;
      return toValue(l / r);
    }

    auto l = toInt64(*left);
    auto r = toInt64(*right);
    if (r == 0) return std::nullopt;
    return toValue(l / r);
  }

  auto percent_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                  const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) {
        auto l = toUInt32(*left);
        auto r = toUInt32(*right);
        if (r == 0) return std::nullopt;
        return toValue(l % r);
      }

      auto l = toUInt64(*left);
      auto r = toUInt64(*right);
      if (r == 0) return std::nullopt;
      return toValue(l % r);
    }

    if (sz <= 4) {
      auto l = toInt32(*left);
      auto r = toInt32(*right);
      if (r == 0) return std::nullopt;
      return toValue(l % r);
    }

    auto l = toInt64(*left);
    auto r = toInt64(*right);
    if (r == 0) return std::nullopt;
    return toValue(l % r);
  }

  auto plus_op(BinaryExpressionAST* ast, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type)) {
      return toDouble(*left) + toDouble(*right);
    }

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toValue(toUInt32(*left) + toUInt32(*right));
      return toValue(toUInt64(*left) + toUInt64(*right));
    }

    if (sz <= 4) return toValue(toInt32(*left) + toInt32(*right));
    return toValue(toInt64(*left) + toInt64(*right));
  }

  auto minus_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type)) {
      return toDouble(*left) - toDouble(*right);
    }

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toValue(toUInt32(*left) - toUInt32(*right));
      return toValue(toUInt64(*left) - toUInt64(*right));
    }

    if (sz <= 4) return toValue(toInt32(*left) - toInt32(*right));
    return toValue(toInt64(*left) - toInt64(*right));
  }

  auto less_less_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                    const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toValue(toUInt32(*left) << toUInt32(*right));
      return toValue(toUInt64(*left) << toUInt64(*right));
    }

    if (sz <= 4) return toValue(toInt32(*left) << toInt32(*right));
    return toValue(toInt64(*left) << toInt64(*right));
  }

  auto greater_greater_op(BinaryExpressionAST* ast,
                          const ExpressionResult& left,
                          const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toValue(toUInt32(*left) >> toUInt32(*right));
      return toValue(toUInt64(*left) >> toUInt64(*right));
    }

    if (sz <= 4) return toValue(toInt32(*left) >> toInt32(*right));
    return toValue(toInt64(*left) >> toInt64(*right));
  }

  auto less_equal_greater_op(BinaryExpressionAST* ast,
                             const ExpressionResult& left,
                             const ExpressionResult& right)
      -> ExpressionResult {
    auto convert = [](std::partial_ordering cmp) -> int {
      if (cmp < 0) return -1;
      if (cmp > 0) return 1;
      return 0;
    };

    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type))
      return convert(toDouble(*left) <=> toDouble(*right));

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return convert(toUInt32(*left) <=> toUInt32(*right));
      return convert(toUInt64(*left) <=> toUInt64(*right));
    }

    if (sz <= 4) return convert(toInt32(*left) <=> toInt32(*right));
    return convert(toInt64(*left) <=> toInt64(*right));
  }

  auto less_equal_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                     const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type))
      return toDouble(*left) <= toDouble(*right);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toUInt(*left) <= toUInt(*right);
      return toUInt64(*left) <= toUInt64(*right);
    }

    if (sz <= 4) return toInt(*left) <= toInt(*right);
    return toInt64(*left) <= toInt64(*right);
  }

  auto greater_equal_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                        const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type))
      return toDouble(*left) >= toDouble(*right);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toUInt(*left) >= toUInt(*right);
      return toUInt64(*left) >= toUInt64(*right);
    }

    if (sz <= 4) return toInt(*left) >= toInt(*right);
    return toInt64(*left) >= toInt64(*right);
  }

  auto less_op(BinaryExpressionAST* ast, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type))
      return toDouble(*left) < toDouble(*right);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toUInt(*left) < toUInt(*right);
      return toUInt64(*left) < toUInt64(*right);
    }

    if (sz <= 4) return toInt(*left) < toInt(*right);
    return toInt64(*left) < toInt64(*right);
  }

  auto greater_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                  const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type))
      return toDouble(*left) > toDouble(*right);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toUInt(*left) > toUInt(*right);
      return toUInt64(*left) > toUInt64(*right);
    }

    if (sz <= 4) return toInt(*left) > toInt(*right);
    return toInt64(*left) > toInt64(*right);
  }

  auto equal_equal_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                      const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type))
      return toDouble(*left) == toDouble(*right);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toUInt(*left) == toUInt(*right);
      return toUInt64(*left) == toUInt64(*right);
    }

    if (sz <= 4) return toInt(*left) == toInt(*right);
    return toInt64(*left) == toInt64(*right);
  }

  auto exclaim_equal_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                        const ExpressionResult& right) -> ExpressionResult {
    const auto type = ast->leftExpression->type;
    const auto sz = memoryLayout()->sizeOf(type);

    if (control()->is_floating_point(type))
      return toDouble(*left) != toDouble(*right);

    if (control()->is_unsigned(type)) {
      if (sz <= 4) return toUInt(*left) != toUInt(*right);
      return toUInt64(*left) != toUInt64(*right);
    }

    if (sz <= 4) return toInt(*left) != toInt(*right);
    return toInt64(*left) != toInt64(*right);
  }

  auto amp_op(BinaryExpressionAST* ast, const ExpressionResult& left,
              const ExpressionResult& right) -> ExpressionResult {
    return toInt(*left) & toInt(*right);
  }

  auto caret_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    return toInt(*left) ^ toInt(*right);
  }

  auto bar_op(BinaryExpressionAST* ast, const ExpressionResult& left,
              const ExpressionResult& right) -> ExpressionResult {
    return toInt(*left) | toInt(*right);
  }

  auto amp_amp_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                  const ExpressionResult& right) -> ExpressionResult {
    return toBool(*left) && toBool(*right);
  }

  auto bar_bar_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                  const ExpressionResult& right) -> ExpressionResult {
    return toBool(*left) || toBool(*right);
  }

  auto comma_op(BinaryExpressionAST* ast, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    // Comma operator returns the right operand
    return right;
  }

  [[nodiscard]] auto operator()(CharLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(BoolLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(IntLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(FloatLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(NullptrLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(StringLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(UserDefinedStringLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(ObjectLiteralExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(ThisExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(GenericSelectionExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(NestedStatementExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(NestedExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(IdExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(LambdaExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(FoldExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(RightFoldExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(LeftFoldExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(RequiresExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(VaArgExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(SubscriptExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(CallExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(TypeConstructionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(BracedTypeConstructionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(SpliceMemberExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(MemberExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(PostIncrExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(CppCastExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(BuiltinBitCastExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(BuiltinOffsetofExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(TypeidExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(TypeidOfTypeExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(SpliceExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(GlobalScopeReflectExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(NamespaceReflectExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(TypeIdReflectExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(ReflectExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(LabelAddressExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(UnaryExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(AwaitExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(SizeofExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(SizeofTypeExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(SizeofPackExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(AlignofTypeExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(AlignofExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(NoexceptExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(NewExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(DeleteExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(CastExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(ImplicitCastExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(ConditionalExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(YieldExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(ThrowExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(AssignmentExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(TargetExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(RightExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(CompoundAssignmentExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(PackExpansionExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(DesignatedInitializerClauseAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(TypeTraitExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(ConditionExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(EqualInitializerAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(BracedInitListAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(ParenInitializerAST* ast) -> ExpressionResult;
};

struct ASTInterpreter::NewInitializerVisitor {
  ASTInterpreter& interp;

  [[nodiscard]] auto operator()(NewParenInitializerAST* ast)
      -> NewInitializerResult;

  [[nodiscard]] auto operator()(NewBracedInitializerAST* ast)
      -> NewInitializerResult;
};

auto ASTInterpreter::expression(ExpressionAST* ast) -> ExpressionResult {
  if (ast) return visit(ExpressionVisitor{*this}, ast);
  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::newPlacement(NewPlacementAST* ast) -> NewPlacementResult {
  if (!ast) return {};

  for (auto node : ListView{ast->expressionList}) {
    auto value = expression(node);
  }

  return {};
}

auto ASTInterpreter::newInitializer(NewInitializerAST* ast)
    -> NewInitializerResult {
  if (ast) return visit(NewInitializerVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    CharLiteralExpressionAST* ast) -> ExpressionResult {
  return ConstValue(ast->literal->charValue());
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BoolLiteralExpressionAST* ast) -> ExpressionResult {
  return ConstValue(ast->isTrue);
}

auto ASTInterpreter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  const auto value = static_cast<std::uintmax_t>(ast->literal->integerValue());
  return ExpressionResult{std::bit_cast<std::intmax_t>(value)};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    FloatLiteralExpressionAST* ast) -> ExpressionResult {
  return ConstValue(ast->literal->floatValue());
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) -> ExpressionResult {
  return ConstValue{std::intmax_t(0)};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    StringLiteralExpressionAST* ast) -> ExpressionResult {
  return ConstValue(ast->literal);
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionResult {
  return ConstValue(ast->literal);
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ObjectLiteralExpressionAST* ast) -> ExpressionResult {
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    GenericSelectionExpressionAST* ast) -> ExpressionResult {
  if (ast->matchedAssocIndex == -1) return std::nullopt;

  GenericAssociationAST* assoc = nullptr;
  int index = 0;
  for (auto assocNode : ListView{ast->genericAssociationList}) {
    if (index == ast->matchedAssocIndex) {
      assoc = assocNode;
      break;
    }
    ++index;
  }

  if (auto def = ast_cast<DefaultGenericAssociationAST>(assoc)) {
    return interp.expression(def->expression);
  }

  if (auto entry = ast_cast<TypeGenericAssociationAST>(assoc)) {
    return interp.expression(entry->expression);
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
  if (ast->expression) {
    return evaluate(ast->expression);
  }
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NestedStatementExpressionAST* ast) -> ExpressionResult {
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionResult {
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
    return enumerator->value();
  }

  if (auto var = symbol_cast<VariableSymbol>(ast->symbol);
      var && var->isConstexpr()) {
    return var->constValue();
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->captureList}) {
    auto value = interp.lambdaCapture(node);
  }

  for (auto node : ListView{ast->templateParameterList}) {
    auto value = interp.templateParameter(node);
  }

  auto templateRequiresClauseResult =
      interp.requiresClause(ast->templateRequiresClause);

  auto parameterDeclarationClauseResult =
      interp.parameterDeclarationClause(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->gnuAtributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->lambdaSpecifierList}) {
    auto value = interp.lambdaSpecifier(node);
  }

  auto exceptionSpecifierResult =
      interp.exceptionSpecifier(ast->exceptionSpecifier);

  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  auto trailingReturnTypeResult =
      interp.trailingReturnType(ast->trailingReturnType);
  auto requiresClauseResult = interp.requiresClause(ast->requiresClause);
  auto statementResult = interp.statement(ast->statement);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = interp.expression(ast->leftExpression);
  auto rightExpressionResult = interp.expression(ast->rightExpression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionResult {
  auto parameterDeclarationClauseResult =
      interp.parameterDeclarationClause(ast->parameterDeclarationClause);

  for (auto node : ListView{ast->requirementList}) {
    auto value = interp.requirement(node);
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);
  auto typeIdResult = interp.typeId(ast->typeId);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);
  auto indexExpressionResult = interp.expression(ast->indexExpression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);

  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.expression(node);
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);

  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.expression(node);
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BracedTypeConstructionAST* ast) -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);
  auto bracedInitListResult = interp.expression(ast->bracedInitList);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    SpliceMemberExpressionAST* ast) -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);
  auto splicerResult = interp.splicer(ast->splicer);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);
  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = interp.typeId(ast->typeId);
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = interp.typeId(ast->typeId);
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinOffsetofExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  if (ast->symbol) return ast->symbol->offset();

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    TypeidOfTypeExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionResult {
  auto splicerResult = interp.splicer(ast->splicer);
  if (!splicerResult.has_value()) return std::nullopt;

  auto metaPtr = std::get_if<std::shared_ptr<Meta>>(&splicerResult.value());
  if (!metaPtr) return std::nullopt;

  auto meta = *metaPtr;

  auto constExprPtr = std::get_if<Meta::ConstExpr>(&meta->value);
  if (!constExprPtr) return std::nullopt;

  return constExprPtr->value;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionResult {
  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) -> ExpressionResult {
  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    TypeIdReflectExpressionAST* ast) -> ExpressionResult {
  if (!ast->typeId) return std::nullopt;
  if (!ast->typeId->type) return std::nullopt;

  auto meta = std::make_shared<Meta>(ast->typeId->type);

  return ConstValue{meta};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  if (expressionResult.has_value()) {
    auto meta = std::make_shared<Meta>(Meta::ConstExpr{
        .expression = ast->expression, .value = expressionResult.value()});
    return meta;
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    LabelAddressExpressionAST* ast) -> ExpressionResult {
  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  switch (ast->op) {
    case TokenKind::T_MINUS: {
      if (expressionResult.has_value() &&
          control()->is_integral_or_unscoped_enum(ast->expression->type)) {
        const auto sz = memoryLayout()->sizeOf(ast->expression->type);

        if (sz <= 4) {
          if (control()->is_unsigned(ast->expression->type)) {
            return toValue(-toUInt32(expressionResult.value()));
          }

          return ExpressionResult(-toInt32(expressionResult.value()));
        }

        if (control()->is_unsigned(ast->expression->type)) {
          return toValue(-toUInt64(expressionResult.value()));
        }

        return ExpressionResult(-toInt64(expressionResult.value()));
      }
      break;
    }

    default:
      break;
  }  // switch

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->expression || !ast->expression->type) return std::nullopt;
  auto size = memoryLayout()->sizeOf(ast->expression->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->typeId || !ast->typeId->type) return std::nullopt;
  auto size = memoryLayout()->sizeOf(ast->typeId->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionResult {
  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    AlignofTypeExpressionAST* ast) -> ExpressionResult {
  if (!ast->typeId || !ast->typeId->type) return std::nullopt;
  auto size = memoryLayout()->alignmentOf(ast->typeId->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  if (!ast->expression || !ast->expression->type) return std::nullopt;
  auto size = memoryLayout()->alignmentOf(ast->expression->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionResult {
  auto newPlacementResult = interp.newPlacement(ast->newPlacement);

  for (auto node : ListView{ast->typeSpecifierList}) {
    auto value = interp.specifier(node);
  }

  auto declaratorResult = interp.declarator(ast->declarator);
  auto newInitalizerResult = interp.newInitializer(ast->newInitalizer);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = interp.typeId(ast->typeId);
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  if (!ast->type) return std::nullopt;

  auto value = evaluate(ast->expression);
  if (!value.has_value()) return std::nullopt;

  switch (ast->type->kind()) {
    case TypeKind::kBool: {
      auto result = interp.toBool(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    case TypeKind::kFloat: {
      auto result = interp.toFloat(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    case TypeKind::kDouble: {
      auto result = interp.toDouble(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    case TypeKind::kLongDouble: {
      auto result = interp.toLongDouble(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    default:
      if (control()->is_integral_or_unscoped_enum(ast->type)) {
        if (control()->is_unsigned(ast->type)) {
          auto result = interp.toUInt(*value);
          if (!result.has_value()) return std::nullopt;
          return ConstValue{std::bit_cast<std::intmax_t>(result.value())};
        }

        auto result = interp.toInt(*value);
        if (!result.has_value()) return std::nullopt;
        return result.value();
      }

      return value;
  }  // switch

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->type) return std::nullopt;

  auto left = evaluate(ast->leftExpression);
  if (!left.has_value()) return std::nullopt;

  auto right = evaluate(ast->rightExpression);
  if (!right.has_value()) return std::nullopt;

  switch (ast->op) {
    case TokenKind::T_DOT_STAR:
      break;

    case TokenKind::T_MINUS_GREATER_STAR:
      break;

    case TokenKind::T_STAR:
      return star_op(ast, left, right);

    case TokenKind::T_SLASH:
      return slash_op(ast, left, right);

    case TokenKind::T_PERCENT:
      return percent_op(ast, left, right);

    case TokenKind::T_PLUS:
      return plus_op(ast, left, right);

    case TokenKind::T_MINUS:
      return minus_op(ast, left, right);

    case TokenKind::T_LESS_LESS:
      return less_less_op(ast, left, right);

    case TokenKind::T_GREATER_GREATER:
      return greater_greater_op(ast, left, right);

    case TokenKind::T_LESS_EQUAL_GREATER:
      return less_equal_greater_op(ast, left, right);

    case TokenKind::T_LESS_EQUAL:
      return less_equal_op(ast, left, right);

    case TokenKind::T_GREATER_EQUAL:
      return greater_equal_op(ast, left, right);

    case TokenKind::T_LESS:
      return less_op(ast, left, right);

    case TokenKind::T_GREATER:
      return greater_op(ast, left, right);

    case TokenKind::T_EQUAL_EQUAL:
      return equal_equal_op(ast, left, right);

    case TokenKind::T_EXCLAIM_EQUAL:
      return exclaim_equal_op(ast, left, right);

    case TokenKind::T_AMP:
      return amp_op(ast, left, right);

    case TokenKind::T_CARET:
      return caret_op(ast, left, right);

    case TokenKind::T_BAR:
      return bar_op(ast, left, right);

    case TokenKind::T_AMP_AMP:
      return amp_amp_op(ast, left, right);

    case TokenKind::T_BAR_BAR:
      return bar_bar_op(ast, left, right);

    case TokenKind::T_COMMA:
      return comma_op(ast, left, right);

    default:
      unit()->warning(ast->opLoc, "invalid binary expression");
      break;
  }  // switch

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ConditionalExpressionAST* ast) -> ExpressionResult {
  auto conditionResult = interp.expression(ast->condition);

  if (!conditionResult.has_value()) return std::nullopt;

  if (interp.toBool(conditionResult.value())) {
    auto result = interp.expression(ast->iftrueExpression);
    return result;
  }

  auto result = interp.expression(ast->iffalseExpression);

  return result;
}

auto ASTInterpreter::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = interp.expression(ast->leftExpression);
  auto rightExpressionResult = interp.expression(ast->rightExpression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TargetExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(RightExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    CompoundAssignmentExpressionAST* ast) -> ExpressionResult {
  auto leftExpressionResult = interp.expression(ast->targetExpression);
  auto rightExpressionResult = interp.expression(ast->rightExpression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    PackExpansionExpressionAST* ast) -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) -> ExpressionResult {
  auto initializerResult = interp.expression(ast->initializer);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionResult {
#if false
  for (auto node : ListView{ast->typeIdList}) {
    auto value = interp(node);
  }
#endif

  const Type* firstType = nullptr;
  const Type* secondType = nullptr;

  if (ast->typeIdList && ast->typeIdList->value) {
    firstType = ast->typeIdList->value->type;

    if (auto next = ast->typeIdList->next; next && next->value) {
      secondType = next->value->type;
    }
  }

  if (firstType) {
    switch (ast->typeTrait) {
      case BuiltinTypeTraitKind::T___IS_VOID:
        return control()->is_void(firstType);

      case BuiltinTypeTraitKind::T___IS_NULL_POINTER:
        return control()->is_null_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_INTEGRAL:
        return control()->is_integral(firstType);

      case BuiltinTypeTraitKind::T___IS_FLOATING_POINT:
        return control()->is_floating_point(firstType);

      case BuiltinTypeTraitKind::T___IS_ARRAY:
        return control()->is_array(firstType);

      case BuiltinTypeTraitKind::T___IS_ENUM:
        return control()->is_enum(firstType);

      case BuiltinTypeTraitKind::T___IS_SCOPED_ENUM:
        return control()->is_scoped_enum(firstType);

      case BuiltinTypeTraitKind::T___IS_UNION:
        return control()->is_union(firstType);

      case BuiltinTypeTraitKind::T___IS_CLASS:
        return control()->is_class(firstType);

      case BuiltinTypeTraitKind::T___IS_FUNCTION:
        return control()->is_function(firstType);

      case BuiltinTypeTraitKind::T___IS_POINTER:
        return control()->is_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_MEMBER_OBJECT_POINTER:
        return control()->is_member_object_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_MEMBER_FUNCTION_POINTER:
        return control()->is_member_function_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_LVALUE_REFERENCE:
        return control()->is_lvalue_reference(firstType);

      case BuiltinTypeTraitKind::T___IS_RVALUE_REFERENCE:
        return control()->is_rvalue_reference(firstType);

      case BuiltinTypeTraitKind::T___IS_FUNDAMENTAL:
        return control()->is_fundamental(firstType);

      case BuiltinTypeTraitKind::T___IS_ARITHMETIC:
        return control()->is_arithmetic(firstType);

      case BuiltinTypeTraitKind::T___IS_SCALAR:
        return control()->is_scalar(firstType);

      case BuiltinTypeTraitKind::T___IS_OBJECT:
        return control()->is_object(firstType);

      case BuiltinTypeTraitKind::T___IS_COMPOUND:
        return control()->is_compound(firstType);

      case BuiltinTypeTraitKind::T___IS_REFERENCE:
        return control()->is_reference(firstType);

      case BuiltinTypeTraitKind::T___IS_MEMBER_POINTER:
        return control()->is_member_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_BOUNDED_ARRAY:
        return control()->is_bounded_array(firstType);

      case BuiltinTypeTraitKind::T___IS_UNBOUNDED_ARRAY:
        return control()->is_unbounded_array(firstType);

      case BuiltinTypeTraitKind::T___IS_CONST:
        return control()->is_const(firstType);

      case BuiltinTypeTraitKind::T___IS_VOLATILE:
        return control()->is_volatile(firstType);

      case BuiltinTypeTraitKind::T___IS_SIGNED:
        return control()->is_signed(firstType);

      case BuiltinTypeTraitKind::T___IS_UNSIGNED:
        return control()->is_unsigned(firstType);

      case BuiltinTypeTraitKind::T___IS_SAME:
      case BuiltinTypeTraitKind::T___IS_SAME_AS: {
        if (!secondType) break;
        return control()->is_same(firstType, secondType);
      }

      case BuiltinTypeTraitKind::T___IS_BASE_OF: {
        if (!secondType) break;
        return control()->is_base_of(firstType, secondType);
      }

      case BuiltinTypeTraitKind::T___HAS_UNIQUE_OBJECT_REPRESENTATIONS: {
        break;
      }

      case BuiltinTypeTraitKind::T___HAS_VIRTUAL_DESTRUCTOR: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_ABSTRACT: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_AGGREGATE: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_ASSIGNABLE: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_EMPTY: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_FINAL: {
        if (auto classType =
                type_cast<ClassType>(control()->remove_cv(firstType))) {
          return classType->symbol()->isFinal();
        }
        break;
      }

      case BuiltinTypeTraitKind::T___IS_LAYOUT_COMPATIBLE: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_LITERAL_TYPE: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_POD: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_POLYMORPHIC: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_STANDARD_LAYOUT: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_SWAPPABLE_WITH: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_TRIVIAL: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_CONSTRUCTIBLE: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_ASSIGNABLE: {
        break;
      }

      case BuiltinTypeTraitKind::T_NONE: {
        // not a builtin
        break;
      }

    }  // switch
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->attributeList}) {
    auto value = interp.attributeSpecifier(node);
  }

  for (auto node : ListView{ast->declSpecifierList}) {
    auto value = interp.specifier(node);
  }

  auto declaratorResult = interp.declarator(ast->declarator);
  auto initializerResult = interp.expression(ast->initializer);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return expressionResult;
}

auto ASTInterpreter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  auto values = std::vector<std::tuple<ConstValue, const Type*>>();

  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.evaluate(node);
    if (!value) return std::nullopt;
    values.emplace_back(*value, node->type);
  }

  return std::make_shared<InitializerList>(std::move(values));
}

auto ASTInterpreter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.expression(node);
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::NewInitializerVisitor::operator()(
    NewParenInitializerAST* ast) -> NewInitializerResult {
  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.expression(node);
  }

  return {};
}

auto ASTInterpreter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) -> NewInitializerResult {
  auto bracedInitListResult = interp.expression(ast->bracedInitList);

  return {};
}

}  // namespace cxx
