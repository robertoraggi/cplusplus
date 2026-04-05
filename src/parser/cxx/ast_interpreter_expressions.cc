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
#include <cxx/type_traits.h>

// cxx
#include <cxx/ast.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

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

    if (unit()->typeTraits().is_floating_point(type)) {
      return toDouble(*left) * toDouble(*right);
    }

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type)) {
      auto l = toDouble(*left);
      auto r = toDouble(*right);
      if (r == 0.0) return std::nullopt;
      return l / r;
    }

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type)) {
      return toDouble(*left) + toDouble(*right);
    }

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type)) {
      return toDouble(*left) - toDouble(*right);
    }

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type))
      return convert(toDouble(*left) <=> toDouble(*right));

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type))
      return toDouble(*left) <= toDouble(*right);

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type))
      return toDouble(*left) >= toDouble(*right);

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type))
      return toDouble(*left) < toDouble(*right);

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type))
      return toDouble(*left) > toDouble(*right);

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type))
      return toDouble(*left) == toDouble(*right);

    if (unit()->typeTraits().is_unsigned(type)) {
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

    if (unit()->typeTraits().is_floating_point(type))
      return toDouble(*left) != toDouble(*right);

    if (unit()->typeTraits().is_unsigned(type)) {
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

  [[nodiscard]] auto operator()(PackIndexExpressionAST* ast)
      -> ExpressionResult;

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
  if (ast->bracedInitList) return interp.expression(ast->bracedInitList);
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(PackIndexExpressionAST* ast)
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

  if (auto var = symbol_cast<VariableSymbol>(ast->symbol);
      var && !var->isConstexpr() && var->constValue().has_value() &&
      unit()->typeTraits().is_const(var->type()) &&
      unit()->typeTraits().is_integral_or_unscoped_enum(
          unit()->typeTraits().remove_cvref(var->type()))) {
    return var->constValue();
  }

  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && field->isStatic() && field->initializer()) {
    return interp.expression(field->initializer());
  }

  if (ast->symbol) {
    auto local = interp.lookupLocal(ast->symbol);
    if (local.has_value()) return local;
  }

  if (auto param = symbol_cast<ParameterSymbol>(ast->symbol)) {
    auto local = interp.lookupLocal(param);
    if (local.has_value()) return local;
  }

  if (auto field = symbol_cast<FieldSymbol>(ast->symbol)) {
    if (interp.thisObject()) {
      auto* fieldVal = interp.thisObject()->getField(field);
      if (fieldVal) return *fieldVal;
    }
  }

  if (auto func = symbol_cast<FunctionSymbol>(ast->symbol)) {
    return std::make_shared<ConstAddress>(func);
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
  std::vector<ConstValue> args;
  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.evaluate(node);
    if (!value) {
      return ExpressionResult{std::nullopt};
    }
    args.push_back(std::move(*value));
  }

  if (auto* idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (auto* nameId = ast_cast<NameIdAST>(idExpr->unqualifiedId)) {
      if (nameId->identifier) {
        auto builtinKind = nameId->identifier->builtinFunction();
        if (builtinKind != BuiltinFunctionKind::T_NONE) {
          return interp.evaluateBuiltinCall(builtinKind, std::move(args), ast);
        }
      }
    }
  }

  FunctionSymbol* func = nullptr;

  if (auto* idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    func = symbol_cast<FunctionSymbol>(idExpr->symbol);
    if (!func) {
      if (auto* overloads = symbol_cast<OverloadSetSymbol>(idExpr->symbol)) {
        for (auto* f : overloads->functions()) {
          if (f->isConstexpr()) {
            func = f;
            break;
          }
        }
      }
    }
  } else if (auto* memberExpr =
                 ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    func = symbol_cast<FunctionSymbol>(memberExpr->symbol);
    if (!func) {
      if (auto* overloads =
              symbol_cast<OverloadSetSymbol>(memberExpr->symbol)) {
        for (auto* f : overloads->functions()) {
          if (f->isConstexpr()) {
            func = f;
            break;
          }
        }
      }
    }
    if (func && func->isConstexpr()) {
      auto baseVal = interp.evaluate(memberExpr->baseExpression);
      if (baseVal.has_value()) {
        if (auto* initList =
                std::get_if<std::shared_ptr<InitializerList>>(&*baseVal)) {
          if (auto* nameId = ast_cast<NameIdAST>(memberExpr->unqualifiedId)) {
            if (nameId->identifier && nameId->identifier->value() == "size") {
              return ConstValue(std::intmax_t((*initList)->elements.size()));
            }
          }
        }

        if (auto* objPtr =
                std::get_if<std::shared_ptr<ConstObject>>(&*baseVal)) {
          auto savedThis = interp.thisObject();
          interp.setThisObject(*objPtr);
          auto result = interp.evaluateCall(func, std::move(args));
          interp.setThisObject(savedThis);
          return result;
        }
      }
    }
    return ExpressionResult{std::nullopt};
  }

  if (func && func->isConstexpr()) {
    return interp.evaluateCall(func, std::move(args));
  }

  if (auto* idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    if (auto* classSym = symbol_cast<ClassSymbol>(idExpr->symbol)) {
      auto* classType = classSym->type();
      for (auto* ctor : classSym->constructors()) {
        if (ctor->isConstexpr()) {
          return interp.evaluateConstructor(ctor, classType, std::move(args));
        }
        // A defaulted constructor with no arguments is implicitly constexpr.
        if (ctor->isDefaulted() && args.empty()) {
          auto obj = std::make_shared<ConstObject>(classType);
          return ConstValue{std::move(obj)};
        }
      }
    }
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);

  std::vector<ConstValue> args;
  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.evaluate(node);
    if (!value) return ExpressionResult{std::nullopt};
    args.push_back(std::move(*value));
  }

  if (ast->type) {
    if (auto* classType = type_cast<ClassType>(ast->type)) {
      auto* classSym = classType->symbol();
      if (classSym) {
        for (auto* ctor : classSym->constructors()) {
          if (ctor->isConstexpr()) {
            return interp.evaluateConstructor(ctor, ast->type, std::move(args));
          }
        }
      }
    }
    if (args.size() == 1) {
      return std::move(args[0]);
    }
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BracedTypeConstructionAST* ast) -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);
  auto bracedInitListResult = interp.expression(ast->bracedInitList);

  if (ast->type && bracedInitListResult.has_value()) {
    if (auto* classType = type_cast<ClassType>(ast->type)) {
      auto* classSym = classType->symbol();
      if (classSym) {
        if (auto* initList = std::get_if<std::shared_ptr<InitializerList>>(
                &*bracedInitListResult)) {
          auto obj = std::make_shared<ConstObject>(ast->type);
          const auto& members = classSym->members();
          std::size_t fieldIdx = 0;
          for (const auto& [val, ty] : (*initList)->elements) {
            while (fieldIdx < members.size() &&
                   !symbol_cast<FieldSymbol>(members[fieldIdx]))
              ++fieldIdx;
            if (fieldIdx < members.size()) {
              obj->addField(members[fieldIdx], val);
              ++fieldIdx;
            }
          }
          return ConstValue{std::move(obj)};
        }
        for (auto* ctor : classSym->constructors()) {
          if (ctor->isConstexpr()) {
            std::vector<ConstValue> args;
            if (auto* initList = std::get_if<std::shared_ptr<InitializerList>>(
                    &*bracedInitListResult)) {
              for (auto& [v, t] : (*initList)->elements) {
                args.push_back(v);
              }
            } else {
              args.push_back(*bracedInitListResult);
            }
            return interp.evaluateConstructor(ctor, ast->type, std::move(args));
          }
        }
      }
    }
  }

  return bracedInitListResult;
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

  if (baseExpressionResult.has_value()) {
    if (auto* objPtr =
            std::get_if<std::shared_ptr<ConstObject>>(&*baseExpressionResult)) {
      if (ast->symbol) {
        auto* fieldVal = (*objPtr)->getField(ast->symbol);
        if (fieldVal) return *fieldVal;
      }
    }
  }

  if (interp.thisObject() && ast->symbol) {
    auto* fieldVal = interp.thisObject()->getField(ast->symbol);
    if (fieldVal) return *fieldVal;
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  // Static member access: e.g. is_copy_assignable<int>::value
  if (ast->symbol) {
    if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
        field && field->isStatic() && field->initializer()) {
      return interp.expression(field->initializer());
    }
    if (auto var = symbol_cast<VariableSymbol>(ast->symbol);
        var && var->isConstexpr()) {
      if (auto cv = var->constValue()) return cv;
      if (var->initializer()) return interp.expression(var->initializer());
    }
    if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
      return enumerator->value();
    }
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  return expressionResult;
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

  if (!ast->symbol) return std::nullopt;

  auto classType = type_cast<ClassType>(ast->typeId->type);
  if (!classType) return std::nullopt;

  auto classSymbol = classType->symbol();
  auto layout = classSymbol->layout();
  if (!layout) return std::nullopt;

  auto fieldInfo = layout->getFieldInfo(ast->symbol);
  if (!fieldInfo) return std::nullopt;

  return static_cast<int>(fieldInfo->offset);
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
  return ConstValue{
      std::make_shared<ConstLabelAddress>(ast->identifier->name())};
}

auto ASTInterpreter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  switch (ast->op) {
    case TokenKind::T_MINUS: {
      if (expressionResult.has_value() &&
          unit()->typeTraits().is_integral_or_unscoped_enum(
              ast->expression->type)) {
        const auto sz = memoryLayout()->sizeOf(ast->expression->type);

        if (sz <= 4) {
          if (unit()->typeTraits().is_unsigned(ast->expression->type)) {
            return toValue(-toUInt32(expressionResult.value()));
          }

          return ExpressionResult(-toInt32(expressionResult.value()));
        }

        if (unit()->typeTraits().is_unsigned(ast->expression->type)) {
          return toValue(-toUInt64(expressionResult.value()));
        }

        return ExpressionResult(-toInt64(expressionResult.value()));
      }
      break;
    }

    case TokenKind::T_EXCLAIM: {
      if (expressionResult.has_value()) {
        return ExpressionResult(
            static_cast<std::intmax_t>(!toBool(expressionResult.value())));
      }
      break;
    }

    case TokenKind::T_TILDE: {
      if (expressionResult.has_value() &&
          unit()->typeTraits().is_integral_or_unscoped_enum(
              ast->expression->type)) {
        const auto sz = memoryLayout()->sizeOf(ast->expression->type);

        if (sz <= 4) {
          if (unit()->typeTraits().is_unsigned(ast->expression->type)) {
            return toValue(~toUInt32(expressionResult.value()));
          }

          return ExpressionResult(
              static_cast<std::intmax_t>(~toInt32(expressionResult.value())));
        }

        if (unit()->typeTraits().is_unsigned(ast->expression->type)) {
          return toValue(~toUInt64(expressionResult.value()));
        }

        return ExpressionResult(~toInt64(expressionResult.value()));
      }
      break;
    }

    case TokenKind::T_PLUS: {
      if (expressionResult.has_value() &&
          unit()->typeTraits().is_integral_or_unscoped_enum(
              ast->expression->type)) {
        return expressionResult;
      }
      break;
    }

    case TokenKind::T_AMP: {
      if (auto* idExpr = ast_cast<IdExpressionAST>(ast->expression)) {
        if (idExpr->symbol) {
          return std::make_shared<ConstAddress>(idExpr->symbol);
        }
      }

      if (auto* objLit =
              ast_cast<ObjectLiteralExpressionAST>(ast->expression)) {
        if (objLit->symbol) {
          return std::make_shared<ConstAddress>(objLit->symbol);
        }
      }

      auto* subExpr = ast_cast<SubscriptExpressionAST>(ast->expression);
      if (!subExpr) break;

      auto* idExpr = ast_cast<IdExpressionAST>(subExpr->baseExpression);
      if (!idExpr || !idExpr->symbol) break;

      auto indexVal = interp.evaluate(subExpr->indexExpression);
      if (!indexVal) break;

      if (auto idx = interp.toInt(*indexVal))
        return std::make_shared<ConstAddress>(idExpr->symbol, *idx);

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
  if (auto ct = type_cast<ClassType>(
          unit()->typeTraits().remove_cv(ast->expression->type)))
    unit()->typeTraits().requireCompleteClass(ct->symbol());
  auto size = memoryLayout()->sizeOf(ast->expression->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->typeId || !ast->typeId->type) return std::nullopt;
  if (auto ct = type_cast<ClassType>(
          unit()->typeTraits().remove_cv(ast->typeId->type)))
    unit()->typeTraits().requireCompleteClass(ct->symbol());
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
  if (auto ct = type_cast<ClassType>(
          unit()->typeTraits().remove_cv(ast->typeId->type)))
    unit()->typeTraits().requireCompleteClass(ct->symbol());
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
  if (ast->value.has_value())
    return ExpressionResult(static_cast<std::intmax_t>(*ast->value ? 1 : 0));
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
  auto expressionResult = interp.expression(ast->expression);

  return expressionResult;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  if (!ast->type) return std::nullopt;

  if (ast->castKind == ImplicitCastKind::kArrayToPointerConversion) {
    auto innerExpr = ast->expression;
    // Unwrap EqualInitializerAST if present.
    if (auto eq = ast_cast<EqualInitializerAST>(innerExpr))
      innerExpr = eq->expression;
    if (auto id = ast_cast<IdExpressionAST>(innerExpr)) {
      if (auto var = symbol_cast<VariableSymbol>(id->symbol)) {
        bool isGlobal =
            var->parent() &&
            (var->parent()->isNamespace() || var->parent()->isClass() ||
             (var->isStatic() && var->parent()->isBlock()));
        if (isGlobal && unit()->typeTraits().is_array(var->type()))
          return std::make_shared<ConstAddress>(var);
      }
    }
    if (auto objLit = ast_cast<ObjectLiteralExpressionAST>(innerExpr)) {
      if (objLit->symbol) return std::make_shared<ConstAddress>(objLit->symbol);
    }
  }

  auto value = evaluate(ast->expression);
  if (!value.has_value()) return std::nullopt;

  switch (ast->type->kind()) {
    case TypeKind::kBool: {
      auto result = interp.toBool(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    case TypeKind::kFloat: {
      if (ast->expression && ast->expression->type &&
          unit()->typeTraits().is_unsigned(ast->expression->type)) {
        auto result = interp.toUInt(*value);
        if (!result.has_value()) return std::nullopt;
        return static_cast<float>(result.value());
      }
      auto result = interp.toFloat(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    case TypeKind::kDouble: {
      if (ast->expression && ast->expression->type &&
          unit()->typeTraits().is_unsigned(ast->expression->type)) {
        auto result = interp.toUInt(*value);
        if (!result.has_value()) return std::nullopt;
        return static_cast<double>(result.value());
      }
      auto result = interp.toDouble(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    case TypeKind::kLongDouble: {
      if (ast->expression && ast->expression->type &&
          unit()->typeTraits().is_unsigned(ast->expression->type)) {
        auto result = interp.toUInt(*value);
        if (!result.has_value()) return std::nullopt;
        return static_cast<long double>(result.value());
      }
      auto result = interp.toLongDouble(*value);
      if (!result.has_value()) return std::nullopt;
      return result.value();
    }

    default:
      if (unit()->typeTraits().is_integral_or_unscoped_enum(ast->type)) {
        if (unit()->typeTraits().is_unsigned(ast->type)) {
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
    if (auto classType =
            type_cast<ClassType>(unit()->typeTraits().remove_cv(firstType))) {
      unit()->typeTraits().requireCompleteClass(classType->symbol());
    }
    if (secondType) {
      if (auto classType = type_cast<ClassType>(
              unit()->typeTraits().remove_cv(secondType))) {
        unit()->typeTraits().requireCompleteClass(classType->symbol());
      }
    }

    switch (ast->typeTrait) {
      case BuiltinTypeTraitKind::T___IS_VOID:
        return unit()->typeTraits().is_void(firstType);

      case BuiltinTypeTraitKind::T___IS_NULL_POINTER:
        return unit()->typeTraits().is_null_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_INTEGRAL:
        return unit()->typeTraits().is_integral(firstType);

      case BuiltinTypeTraitKind::T___IS_FLOATING_POINT:
        return unit()->typeTraits().is_floating_point(firstType);

      case BuiltinTypeTraitKind::T___IS_ARRAY:
        return unit()->typeTraits().is_array(firstType);

      case BuiltinTypeTraitKind::T___IS_ENUM:
        return unit()->typeTraits().is_enum(firstType);

      case BuiltinTypeTraitKind::T___IS_SCOPED_ENUM:
        return unit()->typeTraits().is_scoped_enum(firstType);

      case BuiltinTypeTraitKind::T___IS_UNION:
        return unit()->typeTraits().is_union(firstType);

      case BuiltinTypeTraitKind::T___IS_CLASS:
        return unit()->typeTraits().is_class(firstType) &&
               !unit()->typeTraits().is_union(firstType);

      case BuiltinTypeTraitKind::T___IS_FUNCTION:
        return unit()->typeTraits().is_function(firstType);

      case BuiltinTypeTraitKind::T___IS_POINTER:
        return unit()->typeTraits().is_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_MEMBER_OBJECT_POINTER:
        return unit()->typeTraits().is_member_object_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_MEMBER_FUNCTION_POINTER:
        return unit()->typeTraits().is_member_function_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_LVALUE_REFERENCE:
        return unit()->typeTraits().is_lvalue_reference(firstType);

      case BuiltinTypeTraitKind::T___IS_RVALUE_REFERENCE:
        return unit()->typeTraits().is_rvalue_reference(firstType);

      case BuiltinTypeTraitKind::T___IS_FUNDAMENTAL:
        return unit()->typeTraits().is_fundamental(firstType);

      case BuiltinTypeTraitKind::T___IS_ARITHMETIC:
        return unit()->typeTraits().is_arithmetic(firstType);

      case BuiltinTypeTraitKind::T___IS_SCALAR:
        return unit()->typeTraits().is_scalar(firstType);

      case BuiltinTypeTraitKind::T___IS_OBJECT:
        return unit()->typeTraits().is_object(firstType);

      case BuiltinTypeTraitKind::T___IS_COMPOUND:
        return unit()->typeTraits().is_compound(firstType);

      case BuiltinTypeTraitKind::T___IS_REFERENCE:
        return unit()->typeTraits().is_reference(firstType);

      case BuiltinTypeTraitKind::T___IS_MEMBER_POINTER:
        return unit()->typeTraits().is_member_pointer(firstType);

      case BuiltinTypeTraitKind::T___IS_BOUNDED_ARRAY:
        return unit()->typeTraits().is_bounded_array(firstType);

      case BuiltinTypeTraitKind::T___IS_UNBOUNDED_ARRAY:
        return unit()->typeTraits().is_unbounded_array(firstType);

      case BuiltinTypeTraitKind::T___IS_CONST:
        return unit()->typeTraits().is_const(firstType);

      case BuiltinTypeTraitKind::T___IS_VOLATILE:
        return unit()->typeTraits().is_volatile(firstType);

      case BuiltinTypeTraitKind::T___IS_SIGNED:
        return unit()->typeTraits().is_signed(firstType);

      case BuiltinTypeTraitKind::T___IS_UNSIGNED:
        return unit()->typeTraits().is_unsigned(firstType);

      case BuiltinTypeTraitKind::T___IS_SAME:
      case BuiltinTypeTraitKind::T___IS_SAME_AS: {
        if (!secondType) break;
        return unit()->typeTraits().is_same(firstType, secondType);
      }

      case BuiltinTypeTraitKind::T___IS_BASE_OF: {
        if (!secondType) break;
        return unit()->typeTraits().is_base_of(firstType, secondType);
      }

      case BuiltinTypeTraitKind::T___HAS_UNIQUE_OBJECT_REPRESENTATIONS: {
        break;
      }

      case BuiltinTypeTraitKind::T___HAS_VIRTUAL_DESTRUCTOR:
        return unit()->typeTraits().has_virtual_destructor(firstType);

      case BuiltinTypeTraitKind::T___IS_ABSTRACT:
        return unit()->typeTraits().is_abstract(firstType);

      case BuiltinTypeTraitKind::T___IS_AGGREGATE:
        return unit()->typeTraits().is_aggregate(firstType);

      case BuiltinTypeTraitKind::T___IS_ASSIGNABLE: {
        if (!secondType) break;
        return unit()->typeTraits().is_assignable(firstType, secondType);
      }

      case BuiltinTypeTraitKind::T___IS_NOTHROW_ASSIGNABLE: {
        if (!secondType) break;
        return unit()->typeTraits().is_nothrow_assignable(firstType,
                                                          secondType);
      }

      case BuiltinTypeTraitKind::T___IS_CONVERTIBLE:
      case BuiltinTypeTraitKind::T___IS_CONVERTIBLE_TO: {
        if (!secondType) break;
        return unit()->typeTraits().is_convertible(firstType, secondType);
      }

      case BuiltinTypeTraitKind::T___IS_DESTRUCTIBLE:
        return unit()->typeTraits().is_destructible(firstType);

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_DESTRUCTIBLE:
        return unit()->typeTraits().is_trivially_destructible(firstType);

      case BuiltinTypeTraitKind::T___IS_EMPTY:
        return unit()->typeTraits().is_empty(firstType);

      case BuiltinTypeTraitKind::T___IS_FINAL:
        return unit()->typeTraits().is_final(firstType);

      case BuiltinTypeTraitKind::T___IS_LAYOUT_COMPATIBLE: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_LITERAL_TYPE:
        return unit()->typeTraits().is_literal_type(firstType);

      case BuiltinTypeTraitKind::T___IS_POD:
        return unit()->typeTraits().is_pod(firstType);

      case BuiltinTypeTraitKind::T___IS_POLYMORPHIC:
        return unit()->typeTraits().is_polymorphic(firstType);

      case BuiltinTypeTraitKind::T___IS_STANDARD_LAYOUT:
        return unit()->typeTraits().is_standard_layout(firstType);

      case BuiltinTypeTraitKind::T___IS_SWAPPABLE_WITH: {
        break;
      }

      case BuiltinTypeTraitKind::T___IS_TRIVIAL:
        return unit()->typeTraits().is_trivial(firstType);

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_CONSTRUCTIBLE:
        return unit()->typeTraits().is_trivially_constructible(firstType);

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_ASSIGNABLE:
        return unit()->typeTraits().is_trivially_assignable(firstType,
                                                            secondType);

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_COPYABLE:
        return unit()->typeTraits().is_trivially_copyable(firstType);

      case BuiltinTypeTraitKind::T___IS_CONSTRUCTIBLE:
      case BuiltinTypeTraitKind::T___IS_NOTHROW_CONSTRUCTIBLE: {
        std::vector<const Type*> argTypes;
        if (auto next = ast->typeIdList ? ast->typeIdList->next : nullptr) {
          for (auto node : ListView{next}) {
            if (node->type) argTypes.push_back(node->type);
          }
        }
        if (ast->typeTrait ==
            BuiltinTypeTraitKind::T___IS_NOTHROW_CONSTRUCTIBLE)
          return unit()->typeTraits().is_nothrow_constructible(firstType,
                                                               argTypes);
        return unit()->typeTraits().is_constructible(firstType, argTypes);
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

namespace {

auto makeZeroConstValue(TranslationUnit* unit, const Type* type)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;
  if (unit->typeTraits().is_integral_or_unscoped_enum(type))
    return std::intmax_t{0};
  if (unit->typeTraits().is_floating_point(type)) return double{0.0};
  if (unit->typeTraits().is_pointer(type)) return std::intmax_t{0};  // null
  if (auto arr = type_cast<BoundedArrayType>(type)) {
    auto list = std::make_shared<InitializerList>();
    list->elements.reserve(arr->size());
    for (size_t i = 0; i < arr->size(); ++i) {
      auto elemZero = makeZeroConstValue(unit, arr->elementType());
      if (!elemZero) return std::nullopt;
      list->elements.emplace_back(*elemZero, arr->elementType());
    }
    return ConstValue{list};
  }

  if (unit->typeTraits().is_class(type))
    return ConstValue{std::make_shared<InitializerList>()};
  return std::nullopt;
}

auto setDesignatedValue(ASTInterpreter& interp,
                        const std::shared_ptr<InitializerList>& list,
                        List<DesignatorAST*>* designatorList,
                        const ConstValue& value, const Type* valueType)
    -> bool {
  if (!designatorList || !list) return false;

  auto* subscript = ast_cast<SubscriptDesignatorAST>(designatorList->value);
  if (!subscript) return false;

  auto idxVal = interp.evaluate(subscript->expression);
  if (!idxVal) return false;
  auto idx = interp.toUInt(*idxVal);
  if (!idx || *idx >= list->elements.size()) return false;

  auto& [elemVal, elemType] = list->elements[*idx];

  if (!designatorList->next) {
    elemVal = value;
    elemType = valueType;
    return true;
  }

  auto* nestedPtr = std::get_if<std::shared_ptr<InitializerList>>(&elemVal);
  if (!nestedPtr || !*nestedPtr) return false;
  return setDesignatedValue(interp, *nestedPtr, designatorList->next, value,
                            valueType);
}

}  // namespace

auto ASTInterpreter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  bool hasDesignated = false;
  for (auto node : ListView{ast->expressionList}) {
    if (ast_cast<DesignatedInitializerClauseAST>(node)) {
      hasDesignated = true;
      break;
    }
  }

  if (hasDesignated) {
    auto arrayType = type_cast<BoundedArrayType>(ast->type);
    if (!arrayType) {
      auto classType = type_cast<ClassType>(ast->type);
      if (!classType) return std::nullopt;
      auto* classSymbol = classType->symbol();
      if (!classSymbol) return std::nullopt;
      auto* layout = classSymbol->layout();
      if (!layout) return std::nullopt;

      struct SlotInfo {
        size_t index;
        const Type* type;
        uint32_t bitOffset = 0;
        uint32_t bitWidth = 0;
      };

      std::unordered_map<FieldSymbol*, SlotInfo> fieldSlotMap;
      for (auto* member : classSymbol->members()) {
        if (auto* field = symbol_cast<FieldSymbol>(member)) {
          if (field->isStatic()) continue;
          if (auto info = layout->getFieldInfo(field))
            fieldSlotMap[field] = {info->index, field->type(), info->bitOffset,
                                   info->bitWidth};
        }
      }

      size_t maxSlot = 0;
      bool anyDot = false;
      for (auto node : ListView{ast->expressionList}) {
        auto* desig = ast_cast<DesignatedInitializerClauseAST>(node);
        if (!desig || !desig->designatorList) continue;
        auto* dot = ast_cast<DotDesignatorAST>(desig->designatorList->value);
        if (!dot || !dot->symbol) continue;
        auto it = fieldSlotMap.find(dot->symbol);
        if (it == fieldSlotMap.end()) continue;
        maxSlot = std::max(maxSlot, it->second.index);
        anyDot = true;
      }
      if (!anyDot) return std::nullopt;

      size_t slotCount = maxSlot + 1;

      std::vector<std::optional<std::pair<ConstValue, const Type*>>> slots(
          slotCount);
      for (auto& [field, info] : fieldSlotMap) {
        if (info.index >= slotCount) continue;
        if (slots[info.index]) continue;
        ConstValue zero = std::intmax_t{0};
        const Type* slotType = info.type;
        if (info.bitWidth == 0) {
          if (auto z = makeZeroConstValue(unit(), info.type)) zero = *z;
        }
        slots[info.index] = {{zero, slotType}};
      }

      std::unordered_map<size_t, std::intmax_t> bitSlotAccum;

      for (auto node : ListView{ast->expressionList}) {
        auto* desig = ast_cast<DesignatedInitializerClauseAST>(node);
        if (!desig) return std::nullopt;
        if (!desig->designatorList) continue;
        auto* dot = ast_cast<DotDesignatorAST>(desig->designatorList->value);
        if (!dot || !dot->symbol) continue;
        auto it = fieldSlotMap.find(dot->symbol);
        if (it == fieldSlotMap.end()) continue;
        size_t idx = it->second.index;
        if (idx >= slotCount) continue;

        ExpressionAST* initExpr = nullptr;
        if (auto eq = ast_cast<EqualInitializerAST>(desig->initializer))
          initExpr = eq->expression;
        else
          initExpr = desig->initializer;

        if (!initExpr) continue;
        auto val = interp.evaluate(initExpr);
        if (!val) continue;

        if (it->second.bitWidth > 0) {
          auto intVal = interp.toInt(*val).value_or(0);
          auto mask = (std::intmax_t{1} << it->second.bitWidth) - 1;
          bitSlotAccum[idx] |= (intVal & mask) << it->second.bitOffset;
        } else {
          const Type* initType = desig->type ? desig->type : it->second.type;
          slots[idx] = {{*val, initType}};
        }
      }

      for (auto& [idx, packed] : bitSlotAccum) {
        if (idx < slotCount) {
          const Type* slotType = slots[idx] ? slots[idx]->second : nullptr;
          slots[idx] = {{std::intmax_t{packed}, slotType}};
        }
      }

      auto topList = std::make_shared<InitializerList>();
      topList->elements.reserve(slotCount);
      for (size_t i = 0; i < slotCount; ++i) {
        if (!slots[i]) return std::nullopt;
        topList->elements.emplace_back(slots[i]->first, slots[i]->second);
      }
      return ConstValue{topList};
    }

    const Type* elementType = arrayType->elementType();
    size_t size = arrayType->size();

    // char array initialized by a string literal in braces.
    bool isCharElem = type_cast<CharType>(elementType) ||
                      type_cast<SignedCharType>(elementType) ||
                      type_cast<UnsignedCharType>(elementType);
    if (isCharElem && ast->expressionList && !ast->expressionList->next) {
      if (auto strLit = ast_cast<StringLiteralExpressionAST>(
              ast->expressionList->value)) {
        return ConstValue(strLit->literal);
      }
    }

    auto topList = std::make_shared<InitializerList>();
    topList->elements.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      auto slotZero = makeZeroConstValue(unit(), elementType);
      if (!slotZero) return std::nullopt;
      topList->elements.emplace_back(*slotZero, elementType);
    }

    size_t currentIndex = 0;
    for (auto node : ListView{ast->expressionList}) {
      if (auto desig = ast_cast<DesignatedInitializerClauseAST>(node)) {
        if (desig->designatorList) {
          if (auto sub = ast_cast<SubscriptDesignatorAST>(
                  desig->designatorList->value)) {
            if (auto idxVal = interp.evaluate(sub->expression)) {
              if (auto idx = interp.toUInt(*idxVal)) currentIndex = *idx;
            }
          }
        }

        ExpressionAST* initExpr = nullptr;
        if (auto eq = ast_cast<EqualInitializerAST>(desig->initializer)) {
          initExpr = eq->expression;
        } else {
          initExpr = desig->initializer;
        }

        if (initExpr && currentIndex < size) {
          if (auto val = interp.evaluate(initExpr)) {
            const Type* initType =
                desig->type ? desig->type
                            : (initExpr->type ? initExpr->type : elementType);
            setDesignatedValue(interp, topList, desig->designatorList, *val,
                               initType);
          }
        }
      } else {
        if (currentIndex < size) {
          if (auto val = interp.evaluate(node)) {
            const Type* nodeType = node->type ? node->type : elementType;
            topList->elements[currentIndex] = {*val, nodeType};
          }
        }
      }
      ++currentIndex;
    }

    return ConstValue{topList};
  }

  auto arrayType = type_cast<BoundedArrayType>(ast->type);
  const Type* elementType = arrayType ? arrayType->elementType() : nullptr;

  // char array initialized by a string literal in braces.
  if (arrayType && elementType) {
    bool isCharElem = type_cast<CharType>(elementType) ||
                      type_cast<SignedCharType>(elementType) ||
                      type_cast<UnsignedCharType>(elementType);
    if (isCharElem && ast->expressionList && !ast->expressionList->next) {
      if (auto strLit = ast_cast<StringLiteralExpressionAST>(
              ast->expressionList->value)) {
        return ConstValue(strLit->literal);
      }
    }
  }

  auto values = std::vector<std::tuple<ConstValue, const Type*>>();
  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.evaluate(node);
    if (!value) return std::nullopt;
    const Type* nodeType = node->type ? node->type : elementType;
    values.emplace_back(*value, nodeType);
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

#include "private/builtins_interpreter-priv.h"
