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
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/const_value.h>
#include <cxx/control.h>
#include <cxx/dependent_types.h>
#include <cxx/lambda_captures.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
namespace {
[[nodiscard]] auto resolvedFunction(Symbol* symbol) -> FunctionSymbol* {
  for (auto func : views::each_function(symbol)) {
    if (func->isConstexpr()) return func;
  }
  return designatedFunction(symbol);
}
}  // namespace

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

  auto star_op(const Type* type, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
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

  auto slash_op(const Type* type, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
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

  auto percent_op(const Type* type, const ExpressionResult& left,
                  const ExpressionResult& right) -> ExpressionResult {
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

  auto plus_op(const Type* type, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
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

  auto minus_op(const Type* type, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
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

  auto less_less_op(const Type* type, const ExpressionResult& left,
                    const ExpressionResult& right) -> ExpressionResult {
    const auto sz = memoryLayout()->sizeOf(type);

    if (unit()->typeTraits().is_unsigned(type)) {
      if (sz <= 4) return toValue(toUInt32(*left) << toUInt32(*right));
      return toValue(toUInt64(*left) << toUInt64(*right));
    }

    if (sz <= 4) return toValue(toInt32(*left) << toInt32(*right));
    return toValue(toInt64(*left) << toInt64(*right));
  }

  auto greater_greater_op(const Type* type, const ExpressionResult& left,
                          const ExpressionResult& right) -> ExpressionResult {
    const auto sz = memoryLayout()->sizeOf(type);

    if (unit()->typeTraits().is_unsigned(type)) {
      if (sz <= 4) return toValue(toUInt32(*left) >> toUInt32(*right));
      return toValue(toUInt64(*left) >> toUInt64(*right));
    }

    if (sz <= 4) return toValue(toInt32(*left) >> toInt32(*right));
    return toValue(toInt64(*left) >> toInt64(*right));
  }

  auto less_equal_greater_op(const Type* type, const ExpressionResult& left,
                             const ExpressionResult& right)
      -> ExpressionResult {
    auto convert = [](std::partial_ordering cmp) -> int {
      if (cmp < 0) return -1;
      if (cmp > 0) return 1;
      return 0;
    };

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

  auto less_equal_op(const Type* type, const ExpressionResult& left,
                     const ExpressionResult& right) -> ExpressionResult {
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

  auto greater_equal_op(const Type* type, const ExpressionResult& left,
                        const ExpressionResult& right) -> ExpressionResult {
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

  auto less_op(const Type* type, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
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

  auto greater_op(const Type* type, const ExpressionResult& left,
                  const ExpressionResult& right) -> ExpressionResult {
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

  auto equal_equal_op(const Type* type, const ExpressionResult& left,
                      const ExpressionResult& right) -> ExpressionResult {
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

  auto exclaim_equal_op(const Type* type, const ExpressionResult& left,
                        const ExpressionResult& right) -> ExpressionResult {
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

  auto amp_op(const Type* type, const ExpressionResult& left,
              const ExpressionResult& right) -> ExpressionResult {
    return toInt(*left) & toInt(*right);
  }

  auto caret_op(const Type* type, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    return toInt(*left) ^ toInt(*right);
  }

  auto bar_op(const Type* type, const ExpressionResult& left,
              const ExpressionResult& right) -> ExpressionResult {
    return toInt(*left) | toInt(*right);
  }

  [[nodiscard]] auto applyBinaryOp(TokenKind op, const Type* type,
                                   const ExpressionResult& left,
                                   const ExpressionResult& right)
      -> ExpressionResult;

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

  [[nodiscard]] auto constructorArgumentExpressions(
      BracedTypeConstructionAST* ast) -> std::vector<ExpressionAST*>;

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

  [[nodiscard]] auto operator()(ConstExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto evaluateConstructorConversion(
      ImplicitCastExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto evaluateConversionFunctionCall(
      ImplicitCastExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto evaluateMemberObjectPointerConversion(
      ImplicitCastExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(ThreeWayComparisonExpressionAST* ast)
      -> ExpressionResult;

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
  if (!ast) return ExpressionResult{std::nullopt};
  if (aborted_) return ExpressionResult{std::nullopt};
  return visit(ExpressionVisitor{*this}, ast);
}

auto ASTInterpreter::evaluateStaticField(FieldSymbol* field)
    -> ExpressionResult {
  if (std::ranges::contains(fieldsUnderEvaluation_, field))
    return ExpressionResult{std::nullopt};

  ASTRewriter::completePendingFieldInitializer(unit_, field);

  fieldsUnderEvaluation_.push_back(field);
  auto result = expression(field->initializer());
  fieldsUnderEvaluation_.pop_back();

  return result;
}

auto ASTInterpreter::lvalue(ExpressionAST* ast) -> ConstValue* {
  if (!ast) return nullptr;

  while (ast) {
    if (auto nested = ast_cast<NestedExpressionAST>(ast)) {
      ast = nested->expression;
      continue;
    }
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(ast)) {
      if (!is_glvalue(cast)) break;
      ast = cast->expression;
      continue;
    }
    if (auto constant = ast_cast<ConstExpressionAST>(ast)) {
      if (!is_glvalue(constant)) break;
      ast = constant->expression;
      continue;
    }
    break;
  }

  if (auto id = ast_cast<IdExpressionAST>(ast)) {
    auto sym = id->symbol;
    if (!sym) return nullptr;

    if (auto field = symbol_cast<FieldSymbol>(sym)) {
      if (!field->isStatic() && thisObject_ &&
          traits.is_member_of_object_type(thisObject_->type(), field)) {
        return thisObject_->getFieldMutable(field);
      }
      if (field->isStatic()) {
        if (auto slot = lookupLocalSlot(sym)) return slot;
        if (field->initializer()) {
          auto value = expression(field->initializer());
          if (!value) return nullptr;
          setLocal(sym, std::move(*value));
          return lookupLocalSlot(sym);
        }
      }
      return nullptr;
    }

    if (symbol_cast<VariableSymbol>(sym) || symbol_cast<ParameterSymbol>(sym)) {
      if (auto slot = lookupLocalSlot(sym)) return slot;
      if (auto var = symbol_cast<VariableSymbol>(sym)) {
        if (!var->parent() || !var->parent()->isBlock() || var->isStatic())
          return nullptr;
        if (var->constValue().has_value()) {
          setLocal(sym, *var->constValue());
          return lookupLocalSlot(sym);
        }
      }
      setLocal(sym, ConstValue{std::intmax_t{0}});
      return lookupLocalSlot(sym);
    }
    return nullptr;
  }

  if (auto member = ast_cast<MemberExpressionAST>(ast)) {
    if (!member->symbol) return nullptr;
    auto field = symbol_cast<FieldSymbol>(member->symbol);
    if (!field || field->isStatic()) return nullptr;

    if (auto base = lvalue(member->baseExpression)) {
      if (auto obj = std::get_if<std::shared_ptr<ConstObject>>(base))
        return (*obj)->getFieldMutable(field);
    }
    return nullptr;
  }

  if (auto sub = ast_cast<SubscriptExpressionAST>(ast)) {
    if (auto op = symbol_cast<FunctionSymbol>(sub->symbol);
        op && op->isConstexpr()) {
      auto baseVal = expression(sub->baseExpression);
      if (!baseVal.has_value()) return nullptr;
      auto objPtr = std::get_if<std::shared_ptr<ConstObject>>(&*baseVal);
      if (!objPtr) return nullptr;
      auto idxVal = expression(sub->indexExpression);
      if (!idxVal.has_value()) return nullptr;
      auto savedThis = thisObject_;
      thisObject_ = *objPtr;
      auto slot = evaluateCallLValue(op, {*idxVal});
      thisObject_ = savedThis;
      return slot;
    }

    auto baseVal = expression(sub->baseExpression);
    if (!baseVal.has_value()) return nullptr;

    if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&*baseVal)) {
      if (!*list) return nullptr;
      auto idxVal = expression(sub->indexExpression);
      if (!idxVal.has_value()) return nullptr;
      auto idx = toUInt(*idxVal);
      if (!idx.has_value() || *idx >= (*list)->elements.size()) return nullptr;
      return &std::get<0>((*list)->elements[*idx]);
    }

    if (auto addr = std::get_if<std::shared_ptr<ConstAddress>>(&*baseVal)) {
      if (!*addr) return nullptr;
      auto idxVal = expression(sub->indexExpression);
      if (!idxVal.has_value()) return nullptr;
      auto idx = toInt(*idxVal);
      if (!idx.has_value()) return nullptr;
      return addressSlot(**addr, *idx);
    }

    return nullptr;
  }

  if (auto unary = ast_cast<UnaryExpressionAST>(ast)) {
    if (unary->op != TokenKind::T_STAR) return nullptr;
    auto ptrVal = expression(unary->expression);
    if (!ptrVal.has_value()) return nullptr;
    auto addr = std::get_if<std::shared_ptr<ConstAddress>>(&*ptrVal);
    if (!addr || !*addr) return nullptr;
    return addressSlot(**addr, 0);
  }

  if (auto cond = ast_cast<ConditionalExpressionAST>(ast)) {
    auto condVal = expression(cond->condition);
    if (!condVal.has_value()) return nullptr;
    auto b = toBool(*condVal);
    if (!b.has_value()) return nullptr;
    return lvalue(*b ? cond->iftrueExpression : cond->iffalseExpression);
  }

  if (auto call = ast_cast<CallExpressionAST>(ast)) {
    if (auto idExpr = ast_cast<IdExpressionAST>(call->baseExpression)) {
      auto func = resolvedFunction(idExpr->symbol);
      if (!func || !func->isConstexpr()) return nullptr;
      return evaluateCallLValueFromExprs(func, call->expressionList);
    }

    if (auto memberExpr = ast_cast<MemberExpressionAST>(call->baseExpression)) {
      auto func = resolvedFunction(memberExpr->symbol);
      if (!func || !func->isConstexpr()) return nullptr;
      auto baseVal = expression(memberExpr->baseExpression);
      if (!baseVal.has_value()) return nullptr;
      auto objPtr = std::get_if<std::shared_ptr<ConstObject>>(&*baseVal);
      if (!objPtr) return nullptr;
      auto savedThis = thisObject_;
      thisObject_ = *objPtr;
      auto slot = evaluateCallLValueFromExprs(func, call->expressionList);
      thisObject_ = savedThis;
      return slot;
    }

    return nullptr;
  }

  return nullptr;
}

auto ASTInterpreter::loadAddress(const ConstAddress& address,
                                 std::intmax_t extraIndex)
    -> std::optional<ConstValue> {
  const auto index = address.offset() + extraIndex;
  if (index < 0) return std::nullopt;

  if (auto str = address.stringLiteral()) {
    const auto value = str->stringValue();
    if (static_cast<std::size_t>(index) > value.size()) return std::nullopt;
    auto ch =
        static_cast<std::size_t>(index) < value.size() ? value[index] : '\0';
    return ConstValue{
        static_cast<std::intmax_t>(static_cast<unsigned char>(ch))};
  }

  auto sym = address.symbol();
  if (!sym) return std::nullopt;

  if (symbol_cast<FunctionSymbol>(sym))
    return std::make_shared<ConstAddress>(sym);

  std::optional<ConstValue> storage;
  if (address.owner()) {
    if (auto fv = address.owner()->getField(sym)) storage = *fv;
  } else if (auto slot = lookupLocalSlot(sym)) {
    storage = *slot;
  } else if (auto var = symbol_cast<VariableSymbol>(sym)) {
    if (auto cv = var->constValue())
      storage = cv;
    else if (var->initializer())
      storage = expression(var->initializer());
  }
  if (!storage.has_value()) return std::nullopt;

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&*storage)) {
    if (!*list || static_cast<std::size_t>(index) >= (*list)->elements.size())
      return std::nullopt;
    return std::get<0>((*list)->elements[index]);
  }

  if (index == 0) return storage;
  return std::nullopt;
}

auto ASTInterpreter::addressSlot(const ConstAddress& address,
                                 std::intmax_t extraIndex) -> ConstValue* {
  const auto index = address.offset() + extraIndex;
  if (index < 0) return nullptr;

  if (address.stringLiteral()) return nullptr;

  auto sym = address.symbol();
  if (!sym) return nullptr;

  auto slot = address.owner() ? address.owner()->getFieldMutable(sym)
                              : lookupLocalSlot(sym);
  if (!slot) return nullptr;

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(slot)) {
    if (!*list || static_cast<std::size_t>(index) >= (*list)->elements.size())
      return nullptr;
    return &std::get<0>((*list)->elements[index]);
  }

  if (index == 0) return slot;
  return nullptr;
}

auto ASTInterpreter::fieldOwner(ExpressionAST* ast)
    -> std::shared_ptr<ConstObject> {
  if (auto id = ast_cast<IdExpressionAST>(ast)) {
    if (symbol_cast<FieldSymbol>(id->symbol)) return thisObject_;
    return nullptr;
  }
  if (auto member = ast_cast<MemberExpressionAST>(ast)) {
    if (!symbol_cast<FieldSymbol>(member->symbol)) return nullptr;
    auto baseVal = expression(member->baseExpression);
    if (!baseVal.has_value()) return nullptr;
    if (auto obj = std::get_if<std::shared_ptr<ConstObject>>(&*baseVal))
      return *obj;
  }
  return nullptr;
}

auto ASTInterpreter::addressOfLvalue(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  while (ast) {
    if (auto nested = ast_cast<NestedExpressionAST>(ast)) {
      ast = nested->expression;
      continue;
    }
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(ast)) {
      if (!is_glvalue(cast)) break;
      ast = cast->expression;
      continue;
    }
    if (auto constant = ast_cast<ConstExpressionAST>(ast)) {
      if (!is_glvalue(constant)) break;
      ast = constant->expression;
      continue;
    }
    break;
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast)) {
    if (!idExpr->symbol) return std::nullopt;
    if (symbol_cast<FieldSymbol>(idExpr->symbol)) {
      if (auto owner = fieldOwner(ast))
        return std::make_shared<ConstAddress>(owner, idExpr->symbol);
      return std::nullopt;
    }
    return std::make_shared<ConstAddress>(idExpr->symbol);
  }

  if (auto member = ast_cast<MemberExpressionAST>(ast)) {
    if (!symbol_cast<FieldSymbol>(member->symbol)) return std::nullopt;
    if (auto owner = fieldOwner(ast))
      return std::make_shared<ConstAddress>(owner, member->symbol);
    return std::nullopt;
  }

  if (auto objLit = ast_cast<ObjectLiteralExpressionAST>(ast)) {
    if (!objLit->symbol) return std::nullopt;
    return std::make_shared<ConstAddress>(objLit->symbol);
  }

  if (auto subExpr = ast_cast<SubscriptExpressionAST>(ast)) {
    auto idExpr = ast_cast<IdExpressionAST>(subExpr->baseExpression);
    if (!idExpr || !idExpr->symbol) return std::nullopt;

    auto indexVal = evaluate(subExpr->indexExpression);
    if (!indexVal) return std::nullopt;

    auto index = toInt(*indexVal);
    if (!index) return std::nullopt;

    return std::make_shared<ConstAddress>(idExpr->symbol, *index);
  }

  if (auto call = ast_cast<CallExpressionAST>(ast)) {
    if (auto id = ast_cast<IdExpressionAST>(call->baseExpression)) {
      auto function = resolvedFunction(id->symbol);
      return evaluateCallAddressFromExprs(function, call->expressionList);
    }

    if (auto member = ast_cast<MemberExpressionAST>(call->baseExpression)) {
      auto function = resolvedFunction(member->symbol);
      if (!function) return std::nullopt;
      if (!function->isConstexpr()) return std::nullopt;

      auto baseValue = expression(member->baseExpression);
      if (!baseValue.has_value()) return std::nullopt;
      auto object = std::get_if<std::shared_ptr<ConstObject>>(&*baseValue);
      if (!object) return std::nullopt;

      auto savedThis = thisObject_;
      thisObject_ = *object;
      auto result =
          evaluateCallAddressFromExprs(function, call->expressionList);
      thisObject_ = savedThis;
      return result;
    }
  }

  return std::nullopt;
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
  if (!interp.thisObject()) return std::nullopt;
  return ConstValue{interp.thisObject()};
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

  if (auto conceptSymbol = symbol_cast<ConceptSymbol>(ast->symbol)) {
    auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
    if (!templateId) return std::nullopt;
    auto satisfied = ASTRewriter::evaluateConcept(
        unit(), conceptSymbol, templateId->templateArgumentList);
    if (!satisfied.has_value()) return std::nullopt;
    return ConstValue{*satisfied};
  }

  if (auto var = symbol_cast<VariableSymbol>(ast->symbol);
      var && var->isConstexpr()) {
    if (unit()->typeTraits().is_reference(var->type())) {
      auto value = var->constValue();
      if (!value) return std::nullopt;
      auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value);
      if (!address || !*address) return std::nullopt;
      return interp.loadAddress(**address, 0);
    }
    return var->constValue();
  }

  if (auto var = symbol_cast<VariableSymbol>(ast->symbol);
      var && !var->isConstexpr() && var->constValue().has_value() &&
      unit()->typeTraits().is_const(var->type()) &&
      (unit()->typeTraits().is_integral_or_enum(
           unit()->typeTraits().remove_cvref(var->type())) ||
       isDependent(unit(), var->type()))) {
    return var->constValue();
  }

  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && field->isStatic() && field->initializer()) {
    return interp.evaluateStaticField(field);
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
      auto fieldVal = interp.thisObject()->getField(field);
      if (fieldVal) return *fieldVal;
    }
  }

  if (auto func = resolvedFunction(ast->symbol)) {
    return std::make_shared<ConstAddress>(func);
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  auto classType = type_cast<ClassType>(ast->type);
  if (!classType || !classType->symbol()) return ExpressionResult{std::nullopt};

  auto closure = std::make_shared<ConstObject>(classType);

  auto captureFields =
      views::members(classType->symbol()) | views::non_static_fields;
  auto fieldIt = captureFields.begin();
  const auto fieldEnd = captureFields.end();

  for (auto captureNode : ListView{ast->captureList}) {
    if (fieldIt == fieldEnd) return ExpressionResult{std::nullopt};
    if (is_pack_capture(captureNode)) return ExpressionResult{std::nullopt};

    auto captureField = *fieldIt;
    ++fieldIt;

    auto initializer = capture_initializer(captureNode);
    if (!initializer) return ExpressionResult{std::nullopt};

    auto value = interp.traits.is_reference(captureField->type())
                     ? interp.addressOfLvalue(initializer)
                     : interp.evaluate(initializer);
    if (!value) return ExpressionResult{std::nullopt};

    closure->addField(captureField, std::move(*value));
  }

  if (fieldIt != fieldEnd) return ExpressionResult{std::nullopt};

  return ExpressionResult{ConstValue{std::move(closure)}};
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
  auto scope = ast->parameterDeclarationClause
                   ? ast->parameterDeclarationClause->functionParametersSymbol
                   : nullptr;

  for (auto node : ListView{ast->requirementList}) {
    auto satisfied = interp.isRequirementSatisfied(node, scope);
    if (!satisfied.has_value()) return ExpressionResult{std::nullopt};
    if (!*satisfied) return ConstValue{false};
  }

  return ConstValue{true};
}

auto ASTInterpreter::isReturnTypeRequirementSatisfied(
    TypeConstraintAST* typeConstraint, ExpressionAST* expression)
    -> std::optional<bool> {
  if (!typeConstraint->symbol) return std::nullopt;

  auto arena = unit_->arena();

  auto deducedTypeId = TypeIdAST::create(arena);
  deducedTypeId->type = unit_->typeTraits().decltype_of(expression);
  if (!deducedTypeId->type) return std::nullopt;

  auto deducedArgument = TypeTemplateArgumentAST::create(arena);
  deducedArgument->typeId = deducedTypeId;

  List<TemplateArgumentAST*>* templateArgumentList = nullptr;
  auto out = &templateArgumentList;
  *out = make_list_node<TemplateArgumentAST>(arena, deducedArgument);
  out = &(*out)->next;

  for (auto argument : ListView{typeConstraint->templateArgumentList}) {
    *out = make_list_node(arena, argument);
    out = &(*out)->next;
  }

  return ASTRewriter::evaluateConcept(unit_, typeConstraint->symbol,
                                      templateArgumentList);
}

auto ASTInterpreter::isRequirementSatisfied(RequirementAST* ast,
                                            ScopeSymbol* scope)
    -> std::optional<bool> {
  if (!ast) return true;

  auto isValidExpression =
      [&](ExpressionAST* expression) -> std::optional<bool> {
    if (!expression) return std::nullopt;

    {
      SilentDiagnosticsScope silent{unit_};
      auto typeChecker = TypeChecker{unit_};
      typeChecker.setScope(scope);
      typeChecker.setReportErrors(true);
      typeChecker.check(expression);
      if (silent.hadError()) return false;
    }

    if (!expression->type) return false;
    if (isDependent(unit_, expression->type)) return std::nullopt;
    return true;
  };

  if (auto simple = ast_cast<SimpleRequirementAST>(ast))
    return isValidExpression(simple->expression);

  if (auto compound = ast_cast<CompoundRequirementAST>(ast)) {
    auto valid = isValidExpression(compound->expression);
    if (!valid.has_value() || !*valid) return valid;

    if (compound->noexceptLoc &&
        TypeChecker::isPotentiallyThrowing(compound->expression))
      return false;

    if (!compound->typeConstraint) return true;

    return isReturnTypeRequirementSatisfied(compound->typeConstraint,
                                            compound->expression);
  }

  if (auto typeRequirement = ast_cast<TypeRequirementAST>(ast)) {
    SilentDiagnosticsClient silent;
    auto saved = unit_->changeDiagnosticsClient(&silent);
    auto resolved = Binder{unit_}.resolve(typeRequirement->nestedNameSpecifier,
                                          typeRequirement->unqualifiedId,
                                          /*checkTemplates=*/true);
    (void)unit_->changeDiagnosticsClient(saved);
    if (silent.hadError()) return false;
    if (!resolved) return false;
    if (resolved->type() && isDependent(unit_, resolved->type()))
      return std::nullopt;
    return true;
  }

  if (auto nested = ast_cast<NestedRequirementAST>(ast)) {
    auto valid = isValidExpression(nested->expression);
    if (!valid.has_value() || !*valid) return valid;
    auto value = expression(nested->expression);
    if (!value.has_value()) return std::nullopt;
    return toBool(*value);
  }

  return true;
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

  if (!baseExpressionResult.has_value() || !indexExpressionResult.has_value())
    return std::nullopt;

  if (auto op = symbol_cast<FunctionSymbol>(ast->symbol);
      op && op->isConstexpr()) {
    if (auto objPtr =
            std::get_if<std::shared_ptr<ConstObject>>(&*baseExpressionResult)) {
      auto savedThis = interp.thisObject();
      interp.setThisObject(*objPtr);
      auto result = interp.evaluateCall(op, {*indexExpressionResult});
      interp.setThisObject(savedThis);
      return result;
    }
  }

  auto idx = interp.toUInt(*indexExpressionResult);
  if (!idx.has_value()) return std::nullopt;

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(
          &*baseExpressionResult)) {
    if (!*list || *idx >= (*list)->elements.size()) return std::nullopt;
    return std::get<0>((*list)->elements[*idx]);
  }

  if (auto str = std::get_if<const StringLiteral*>(&*baseExpressionResult)) {
    const auto value = (*str)->stringValue();
    if (*idx > value.size()) return std::nullopt;
    auto ch = *idx < value.size() ? value[*idx] : '\0';
    return ConstValue{
        static_cast<std::intmax_t>(static_cast<unsigned char>(ch))};
  }

  if (auto addr =
          std::get_if<std::shared_ptr<ConstAddress>>(&*baseExpressionResult)) {
    return interp.loadAddress(**addr, static_cast<std::intmax_t>(*idx));
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  if (auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression)) {
    auto builtinKind = resolveBuiltinFunctionKind(interp.unit_, idExpr);
    if (builtinKind != BuiltinFunctionKind::T_NONE) {
      std::vector<ConstValue> args;
      for (auto node : ListView{ast->expressionList}) {
        auto value = interp.evaluate(node);
        if (!value) return ExpressionResult{std::nullopt};
        args.push_back(std::move(*value));
      }
      return interp.evaluateBuiltinCall(builtinKind, std::move(args), ast);
    }

    auto func = resolvedFunction(idExpr->symbol);
    if (func && func->isConstexpr()) {
      return interp.evaluateCallExprs(func, ast->expressionList);
    }

    if (auto classSym = symbol_cast<ClassSymbol>(idExpr->symbol)) {
      auto classType = classSym->type();

      std::vector<ConstValue> args;
      for (auto node : ListView{ast->expressionList}) {
        auto value = interp.evaluate(node);
        if (!value) return ExpressionResult{std::nullopt};
        args.push_back(std::move(*value));
      }

      if (ast->constructorSymbol && (ast->constructorSymbol->isConstexpr() ||
                                     ast->constructorSymbol->isDefaulted())) {
        return interp.evaluateConstructor(ast->constructorSymbol, classType,
                                          std::move(args));
      }
      if (args.empty()) {
        for (auto ctor : classSym->constructors()) {
          if (ctor->isDefaulted()) {
            return interp.evaluateConstructor(ctor, classType, {});
          }
        }
      }
      return ExpressionResult{std::nullopt};
    }
  } else if (auto memberExpr =
                 ast_cast<MemberExpressionAST>(ast->baseExpression)) {
    auto func = resolvedFunction(memberExpr->symbol);
    if (func && func->isConstexpr()) {
      auto baseVal = interp.evaluate(memberExpr->baseExpression);
      if (baseVal.has_value()) {
        if (auto initList =
                std::get_if<std::shared_ptr<InitializerList>>(&*baseVal)) {
          if (auto nameId = ast_cast<NameIdAST>(memberExpr->unqualifiedId)) {
            if (nameId->identifier && nameId->identifier->value() == "size") {
              return ConstValue(std::intmax_t((*initList)->elements.size()));
            }
          }
        }

        if (auto objPtr =
                std::get_if<std::shared_ptr<ConstObject>>(&*baseVal)) {
          auto savedThis = interp.thisObject();
          interp.setThisObject(*objPtr);
          auto result = interp.evaluateCallExprs(func, ast->expressionList);
          interp.setThisObject(savedThis);
          return result;
        }
      }
    }
    return ExpressionResult{std::nullopt};
  }

  if (auto val = interp.evaluate(ast->baseExpression)) {
    if (auto addr = std::get_if<std::shared_ptr<ConstAddress>>(&*val)) {
      if (*addr) {
        if (auto fnSym = symbol_cast<FunctionSymbol>((*addr)->symbol());
            fnSym && fnSym->isConstexpr()) {
          return interp.evaluateCallExprs(fnSym, ast->expressionList);
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
    if (auto classType = type_cast<ClassType>(ast->type)) {
      auto classSym = classType->symbol();
      if (classSym) {
        if (ast->constructorSymbol && (ast->constructorSymbol->isConstexpr() ||
                                       ast->constructorSymbol->isDefaulted())) {
          return interp.evaluateConstructor(ast->constructorSymbol, ast->type,
                                            std::move(args));
        }
        for (auto ctor : classSym->constructors()) {
          if (ctor->isConstexpr()) {
            return interp.evaluateConstructor(ctor, ast->type, std::move(args));
          }
        }
        if (args.empty() && !classSym->hasUserDeclaredConstructors()) {
          return ConstValue{interp.valueInitializeClass(ast->type, classSym)};
        }
      }
    }
    if (args.size() == 1) {
      return std::move(args[0]);
    }
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::constructorArgumentExpressions(
    BracedTypeConstructionAST* ast) -> std::vector<ExpressionAST*> {
  if (!ast->bracedInitList) return {};

  if (unit()->typeTraits().initializer_list_element_type(
          ast->bracedInitList->type))
    return {ast->bracedInitList};

  std::vector<ExpressionAST*> arguments;
  for (auto argument : ListView{ast->bracedInitList->expressionList})
    arguments.push_back(argument);
  return arguments;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BracedTypeConstructionAST* ast) -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);

  auto classType = type_cast<ClassType>(ast->type);
  auto classSymbol = classType ? classType->symbol() : nullptr;

  if (classSymbol && ast->constructorSymbol) {
    const auto evaluableConstructor = ast->constructorSymbol->isConstexpr() ||
                                      ast->constructorSymbol->isDefaulted();
    if (!evaluableConstructor) return ExpressionResult{std::nullopt};

    std::vector<ConstValue> arguments;
    for (auto argument : constructorArgumentExpressions(ast)) {
      auto value = interp.expression(argument);
      if (!value) return ExpressionResult{std::nullopt};
      arguments.push_back(*value);
    }

    return interp.evaluateConstructor(ast->constructorSymbol, ast->type,
                                      std::move(arguments));
  }

  auto bracedInitListResult = interp.expression(ast->bracedInitList);

  if (classSymbol && bracedInitListResult.has_value() &&
      classSymbol->baseClasses().empty()) {
    if (auto initList = std::get_if<std::shared_ptr<InitializerList>>(
            &*bracedInitListResult)) {
      auto obj = std::make_shared<ConstObject>(ast->type);
      std::size_t elementIndex = 0;
      for (auto field :
           views::members(classSymbol) | views::non_static_fields) {
        if (elementIndex >= (*initList)->elements.size()) break;
        obj->addField(field, std::get<0>((*initList)->elements[elementIndex]));
        ++elementIndex;
      }
      return ConstValue{std::move(obj)};
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
    if (auto objPtr =
            std::get_if<std::shared_ptr<ConstObject>>(&*baseExpressionResult)) {
      if (ast->symbol) {
        auto fieldVal = (*objPtr)->getField(ast->symbol);
        if (fieldVal) return *fieldVal;
      }
    }
  }

  if (interp.thisObject() && ast->symbol) {
    auto fieldVal = interp.thisObject()->getField(ast->symbol);
    if (fieldVal) return *fieldVal;
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

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
  const bool inc = ast->op == TokenKind::T_PLUS_PLUS;
  const auto type = ast->baseExpression ? ast->baseExpression->type : nullptr;
  if (!type) return std::nullopt;

  auto slot = interp.lvalue(ast->baseExpression);
  if (!slot) return std::nullopt;

  auto oldValue = *slot;
  auto newValue =
      applyBinaryOp(inc ? TokenKind::T_PLUS : TokenKind::T_MINUS, type,
                    oldValue, ExpressionResult{std::intmax_t{1}});
  if (!newValue.has_value()) return std::nullopt;

  *slot = *newValue;
  return oldValue;
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
  unit()->typeTraits().requireCompleteClass(classSymbol);
  classSymbol = classSymbol->resolvedDefinition();
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
    case TokenKind::T_PLUS_PLUS:
    case TokenKind::T_MINUS_MINUS: {
      const auto type = ast->expression ? ast->expression->type : nullptr;
      if (!type) return std::nullopt;
      auto slot = interp.lvalue(ast->expression);
      if (!slot) return std::nullopt;
      auto newValue = applyBinaryOp(
          ast->op == TokenKind::T_PLUS_PLUS ? TokenKind::T_PLUS
                                            : TokenKind::T_MINUS,
          type, ExpressionResult{*slot}, ExpressionResult{std::intmax_t{1}});
      if (!newValue.has_value()) return std::nullopt;
      *slot = *newValue;
      return *slot;
    }

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

    case TokenKind::T_STAR: {
      if (!expressionResult.has_value()) break;
      if (auto obj =
              std::get_if<std::shared_ptr<ConstObject>>(&*expressionResult)) {
        return *obj;
      }
      if (auto addr =
              std::get_if<std::shared_ptr<ConstAddress>>(&*expressionResult)) {
        return interp.loadAddress(**addr, 0);
      }
      if (auto str = std::get_if<const StringLiteral*>(&*expressionResult)) {
        const auto value = (*str)->stringValue();
        auto ch = value.empty() ? '\0' : value[0];
        return ConstValue{
            static_cast<std::intmax_t>(static_cast<unsigned char>(ch))};
      }
      if (auto list = std::get_if<std::shared_ptr<InitializerList>>(
              &*expressionResult)) {
        if (*list && !(*list)->elements.empty())
          return std::get<0>((*list)->elements[0]);
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
      auto innerExpr = ast->expression;
      while (auto nested = ast_cast<NestedExpressionAST>(innerExpr))
        innerExpr = nested->expression;

      if (type_cast<MemberObjectPointerType>(ast->type)) {
        auto idExpr = ast_cast<IdExpressionAST>(innerExpr);
        if (!idExpr) break;
        auto field = symbol_cast<FieldSymbol>(idExpr->symbol);
        if (!field) break;
        auto offset = field->offsetInClass();
        if (!offset) break;
        return static_cast<std::intmax_t>(*offset);
      }

      if (auto address = interp.addressOfLvalue(innerExpr))
        return ExpressionResult{std::move(address)};

      break;
    }

    default:
      break;
  }

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

auto ASTInterpreter::ExpressionVisitor::operator()(ConstExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->constValue) return std::nullopt;
  return *ast->constValue;
}

auto ASTInterpreter::ExpressionVisitor::evaluateConstructorConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto constructor = ast->conversionFunction;

  std::vector<ConstValue> args;
  auto paren = ast_cast<ParenInitializerAST>(ast->expression);
  if (!paren) return std::nullopt;

  for (auto node : ListView{paren->expressionList}) {
    auto value = interp.evaluate(node);
    if (!value) return std::nullopt;
    args.push_back(std::move(*value));
  }

  if (constructor->isConstexpr() || constructor->isDefaulted()) {
    return interp.evaluateConstructor(constructor, ast->type, std::move(args));
  }
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::evaluateConversionFunctionCall(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto conversionFunction = ast->conversionFunction;
  if (!conversionFunction->isConstexpr()) return std::nullopt;

  auto objectValue = evaluate(ast->expression);
  if (!objectValue.has_value()) return std::nullopt;

  auto object = std::get_if<std::shared_ptr<ConstObject>>(&*objectValue);
  if (!object || !*object) return std::nullopt;

  return interp.evaluateCall(conversionFunction, {}, *object);
}

auto ASTInterpreter::ExpressionVisitor::evaluateMemberObjectPointerConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto targetType = type_cast<MemberObjectPointerType>(ast->type);
  if (!targetType) return std::nullopt;

  const auto nullValue = static_cast<std::intmax_t>(
      control()->memoryLayout()->nullMemberObjectPointer());

  auto sourceType =
      ast->expression
          ? type_cast<MemberObjectPointerType>(ast->expression->type)
          : nullptr;

  if (!sourceType) return nullValue;

  auto value = evaluate(ast->expression);
  if (!value.has_value()) return std::nullopt;

  auto offset = interp.toInt(*value);
  if (!offset.has_value()) return std::nullopt;
  if (*offset == nullValue) return nullValue;

  auto adjustment = memberPointerBaseAdjustment(sourceType, targetType);
  if (!adjustment.has_value()) return std::nullopt;

  return *offset + *adjustment;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  if (!ast->type) return std::nullopt;

  if (ast->castKind == ImplicitCastKind::kUserDefinedConversion &&
      ast->conversionFunction) {
    if (ast->conversionFunction->isConstructor())
      return evaluateConstructorConversion(ast);
    return evaluateConversionFunctionCall(ast);
  }

  if (ast->castKind == ImplicitCastKind::kPointerToMemberConversion) {
    return evaluateMemberObjectPointerConversion(ast);
  }

  if (ast->castKind == ImplicitCastKind::kArrayToPointerConversion) {
    auto innerExpr = ast->expression;
    if (auto eq = ast_cast<EqualInitializerAST>(innerExpr))
      innerExpr = eq->expression;
    if (auto id = ast_cast<IdExpressionAST>(innerExpr)) {
      if (auto var = symbol_cast<VariableSymbol>(id->symbol)) {
        if (unit()->typeTraits().is_array(var->type()))
          return std::make_shared<ConstAddress>(var);
      } else if (auto field = symbol_cast<FieldSymbol>(id->symbol)) {
        if (unit()->typeTraits().is_array(field->type())) {
          if (auto owner = interp.fieldOwner(innerExpr))
            return std::make_shared<ConstAddress>(owner, field);
        }
      }
    }
    if (auto member = ast_cast<MemberExpressionAST>(innerExpr)) {
      if (auto field = symbol_cast<FieldSymbol>(member->symbol)) {
        if (unit()->typeTraits().is_array(field->type())) {
          if (auto owner = interp.fieldOwner(innerExpr))
            return std::make_shared<ConstAddress>(owner, field);
        }
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
      if (unit()->typeTraits().is_integral_or_enum(ast->type)) {
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
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::applyBinaryOp(
    TokenKind op, const Type* type, const ExpressionResult& left,
    const ExpressionResult& right) -> ExpressionResult {
  auto asAddress =
      [](const ConstValue& v) -> std::optional<std::shared_ptr<ConstAddress>> {
    if (auto a = std::get_if<std::shared_ptr<ConstAddress>>(&v)) return *a;
    if (auto s = std::get_if<const StringLiteral*>(&v))
      return std::make_shared<ConstAddress>(*s, std::intmax_t{0});
    return std::nullopt;
  };
  auto offsetOf = [](const ConstAddress& a) { return a.offset(); };

  auto leftAddr = left.has_value() ? asAddress(*left) : std::nullopt;
  auto rightAddr = right.has_value() ? asAddress(*right) : std::nullopt;

  if (op == TokenKind::T_PLUS || op == TokenKind::T_MINUS) {
    auto rebased = [](const ConstAddress& a,
                      std::intmax_t off) -> std::shared_ptr<ConstAddress> {
      if (a.stringLiteral())
        return std::make_shared<ConstAddress>(a.stringLiteral(), off);
      if (a.owner())
        return std::make_shared<ConstAddress>(a.owner(), a.symbol(), off);
      return std::make_shared<ConstAddress>(a.symbol(), off);
    };

    if (leftAddr && rightAddr && op == TokenKind::T_MINUS) {
      return ConstValue{offsetOf(**leftAddr) - offsetOf(**rightAddr)};
    }
    if (leftAddr && right.has_value()) {
      auto n = toInt(*right);
      auto delta = op == TokenKind::T_PLUS ? n : -n;
      return ConstValue{rebased(**leftAddr, offsetOf(**leftAddr) + delta)};
    }
    if (rightAddr && op == TokenKind::T_PLUS && left.has_value()) {
      auto n = toInt(*left);
      return ConstValue{rebased(**rightAddr, offsetOf(**rightAddr) + n)};
    }
  }

  if (leftAddr && rightAddr) {
    const auto lo = offsetOf(**leftAddr);
    const auto ro = offsetOf(**rightAddr);
    const bool sameTarget =
        (*leftAddr)->symbol() == (*rightAddr)->symbol() &&
        (*leftAddr)->owner() == (*rightAddr)->owner() &&
        (*leftAddr)->stringLiteral() == (*rightAddr)->stringLiteral();

    switch (op) {
      case TokenKind::T_EQUAL_EQUAL:
        return ConstValue{std::intmax_t{sameTarget && lo == ro ? 1 : 0}};
      case TokenKind::T_EXCLAIM_EQUAL:
        return ConstValue{std::intmax_t{sameTarget && lo == ro ? 0 : 1}};
      case TokenKind::T_LESS:
        if (!sameTarget) return std::nullopt;
        return ConstValue{std::intmax_t{lo < ro ? 1 : 0}};
      case TokenKind::T_GREATER:
        if (!sameTarget) return std::nullopt;
        return ConstValue{std::intmax_t{lo > ro ? 1 : 0}};
      case TokenKind::T_LESS_EQUAL:
        if (!sameTarget) return std::nullopt;
        return ConstValue{std::intmax_t{lo <= ro ? 1 : 0}};
      case TokenKind::T_GREATER_EQUAL:
        if (!sameTarget) return std::nullopt;
        return ConstValue{std::intmax_t{lo >= ro ? 1 : 0}};
      default:
        break;
    }
  }

  if (bool(leftAddr) != bool(rightAddr)) {
    const ExpressionResult& other = leftAddr ? right : left;
    const auto* otherInt =
        other.has_value() ? std::get_if<std::intmax_t>(&*other) : nullptr;
    if (otherInt && *otherInt == 0) {
      switch (op) {
        case TokenKind::T_EQUAL_EQUAL:
          return ConstValue{std::intmax_t{0}};
        case TokenKind::T_EXCLAIM_EQUAL:
          return ConstValue{std::intmax_t{1}};
        case TokenKind::T_LESS:
        case TokenKind::T_GREATER:
        case TokenKind::T_LESS_EQUAL:
        case TokenKind::T_GREATER_EQUAL:
          return std::nullopt;
        default:
          break;
      }
    }
  }

  switch (op) {
    case TokenKind::T_STAR:
      return star_op(type, left, right);

    case TokenKind::T_SLASH:
      return slash_op(type, left, right);

    case TokenKind::T_PERCENT:
      return percent_op(type, left, right);

    case TokenKind::T_PLUS:
      return plus_op(type, left, right);

    case TokenKind::T_MINUS:
      return minus_op(type, left, right);

    case TokenKind::T_LESS_LESS:
      return less_less_op(type, left, right);

    case TokenKind::T_GREATER_GREATER:
      return greater_greater_op(type, left, right);

    case TokenKind::T_LESS_EQUAL_GREATER:
      return less_equal_greater_op(type, left, right);

    case TokenKind::T_LESS_EQUAL:
      return less_equal_op(type, left, right);

    case TokenKind::T_GREATER_EQUAL:
      return greater_equal_op(type, left, right);

    case TokenKind::T_LESS:
      return less_op(type, left, right);

    case TokenKind::T_GREATER:
      return greater_op(type, left, right);

    case TokenKind::T_EQUAL_EQUAL:
      return equal_equal_op(type, left, right);

    case TokenKind::T_EXCLAIM_EQUAL:
      return exclaim_equal_op(type, left, right);

    case TokenKind::T_AMP:
      return amp_op(type, left, right);

    case TokenKind::T_CARET:
      return caret_op(type, left, right);

    case TokenKind::T_BAR:
      return bar_op(type, left, right);

    default:
      break;
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->type) return std::nullopt;

  switch (ast->op) {
    case TokenKind::T_AMP_AMP: {
      auto left = evaluate(ast->leftExpression);
      if (!left.has_value()) return std::nullopt;
      if (!toBool(*left)) return ExpressionResult{std::intmax_t{0}};
      auto right = evaluate(ast->rightExpression);
      if (!right.has_value()) return std::nullopt;
      return ExpressionResult{std::intmax_t{toBool(*right) ? 1 : 0}};
    }

    case TokenKind::T_BAR_BAR: {
      auto left = evaluate(ast->leftExpression);
      if (!left.has_value()) return std::nullopt;
      if (toBool(*left)) return ExpressionResult{std::intmax_t{1}};
      auto right = evaluate(ast->rightExpression);
      if (!right.has_value()) return std::nullopt;
      return ExpressionResult{std::intmax_t{toBool(*right) ? 1 : 0}};
    }

    case TokenKind::T_COMMA: {
      (void)evaluate(ast->leftExpression);
      return evaluate(ast->rightExpression);
    }

    case TokenKind::T_DOT_STAR:
    case TokenKind::T_MINUS_GREATER_STAR:
      return std::nullopt;

    default:
      break;
  }

  auto left = evaluate(ast->leftExpression);
  if (!left.has_value()) return std::nullopt;

  auto right = evaluate(ast->rightExpression);
  if (!right.has_value()) return std::nullopt;

  if (ast->symbol) {
    if (ast->symbol->isImplicitObjectMemberFunction()) {
      auto object = std::get_if<std::shared_ptr<ConstObject>>(&*left);
      if (!object || !*object) return std::nullopt;
      return interp.evaluateCall(ast->symbol, {std::move(*right)}, *object);
    }

    return interp.evaluateCall(ast->symbol,
                               {std::move(*left), std::move(*right)});
  }

  auto result = applyBinaryOp(ast->op, ast->leftExpression->type, left, right);
  if (!result.has_value())
    unit()->warning(ast->opLoc, "invalid binary expression");
  return result;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ThreeWayComparisonExpressionAST* ast) -> ExpressionResult {
  if (!ast->comparison) return std::nullopt;
  if (ast->comparison->symbol) return evaluate(ast->comparison);

  auto left = evaluate(ast->comparison->leftExpression);
  if (!left) return std::nullopt;
  auto right = evaluate(ast->comparison->rightExpression);
  if (!right) return std::nullopt;

  Symbol* result = nullptr;
  auto operandType = ast->comparison->leftExpression->type;
  if (unit()->typeTraits().is_floating_point(operandType)) {
    auto leftValue = interp.toDouble(*left);
    auto rightValue = interp.toDouble(*right);
    if (!leftValue || !rightValue) return std::nullopt;

    if (*leftValue < *rightValue) {
      result = ast->lessResult;
    } else if (*leftValue > *rightValue) {
      result = ast->greaterResult;
    } else if (*leftValue == *rightValue) {
      result = ast->equalResult;
    } else {
      result = ast->unorderedResult;
    }
  } else {
    auto ordering = applyBinaryOp(TokenKind::T_LESS_EQUAL_GREATER, operandType,
                                  left, right);
    if (!ordering) return std::nullopt;
    auto value = toInt(*ordering);
    if (value < 0) {
      result = ast->lessResult;
    } else if (value > 0) {
      result = ast->greaterResult;
    } else {
      result = ast->equalResult;
    }
  }

  if (auto field = symbol_cast<FieldSymbol>(result))
    return interp.evaluateStaticField(field);
  if (auto variable = symbol_cast<VariableSymbol>(result))
    return variable->constValue();
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ConditionalExpressionAST* ast) -> ExpressionResult {
  auto conditionResult = interp.expression(ast->condition);

  if (!conditionResult.has_value()) return std::nullopt;

  if (toBool(conditionResult.value())) {
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
  auto rightExpressionResult = interp.expression(ast->rightExpression);
  if (!rightExpressionResult.has_value()) return std::nullopt;

  auto slot = interp.lvalue(ast->leftExpression);
  if (!slot) return std::nullopt;

  *slot = interp.cloneValue(*rightExpressionResult);
  return *slot;
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
  TokenKind binOp = TokenKind::T_EOF_SYMBOL;
  switch (ast->op) {
    case TokenKind::T_PLUS_EQUAL:
      binOp = TokenKind::T_PLUS;
      break;
    case TokenKind::T_MINUS_EQUAL:
      binOp = TokenKind::T_MINUS;
      break;
    case TokenKind::T_STAR_EQUAL:
      binOp = TokenKind::T_STAR;
      break;
    case TokenKind::T_SLASH_EQUAL:
      binOp = TokenKind::T_SLASH;
      break;
    case TokenKind::T_PERCENT_EQUAL:
      binOp = TokenKind::T_PERCENT;
      break;
    case TokenKind::T_AMP_EQUAL:
      binOp = TokenKind::T_AMP;
      break;
    case TokenKind::T_BAR_EQUAL:
      binOp = TokenKind::T_BAR;
      break;
    case TokenKind::T_CARET_EQUAL:
      binOp = TokenKind::T_CARET;
      break;
    case TokenKind::T_LESS_LESS_EQUAL:
      binOp = TokenKind::T_LESS_LESS;
      break;
    case TokenKind::T_GREATER_GREATER_EQUAL:
      binOp = TokenKind::T_GREATER_GREATER;
      break;
    default:
      return std::nullopt;
  }

  const auto type =
      ast->targetExpression ? ast->targetExpression->type : ast->type;
  if (!type) return std::nullopt;

  auto rightExpressionResult = interp.expression(ast->rightExpression);
  if (!rightExpressionResult.has_value()) return std::nullopt;

  auto slot = interp.lvalue(ast->targetExpression);
  if (!slot) return std::nullopt;

  auto result = applyBinaryOp(binOp, type, ExpressionResult{*slot},
                              rightExpressionResult);
  if (!result.has_value()) return std::nullopt;

  *slot = *result;
  return *slot;
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

      case BuiltinTypeTraitKind::T___BUILTIN_TYPES_COMPATIBLE_P: {
        if (!secondType) break;
        return unit()->typeTraits().is_compatible(firstType, secondType);
      }

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

      case BuiltinTypeTraitKind::T___IS_NOTHROW_DESTRUCTIBLE:
        return unit()->typeTraits().is_nothrow_destructible(firstType);

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

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_CONSTRUCTIBLE: {
        std::vector<const Type*> argTypes;
        auto next = ast->typeIdList;
        if (next) next = next->next;
        if (next) {
          for (auto node : ListView{next}) {
            if (node->type) argTypes.push_back(node->type);
          }
        }
        return unit()->typeTraits().is_trivially_constructible(firstType,
                                                               argTypes);
      }

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_ASSIGNABLE:
        return unit()->typeTraits().is_trivially_assignable(firstType,
                                                            secondType);

      case BuiltinTypeTraitKind::T___IS_TRIVIALLY_COPYABLE:
        return unit()->typeTraits().is_trivially_copyable(firstType);

      case BuiltinTypeTraitKind::T___IS_CONSTRUCTIBLE:
      case BuiltinTypeTraitKind::T___IS_NOTHROW_CONSTRUCTIBLE: {
        std::vector<const Type*> argTypes;
        auto next = ast->typeIdList;
        if (next) next = next->next;
        if (next) {
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
        break;
      }
    }
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
  if (unit->typeTraits().is_integral_or_enum(type)) return std::intmax_t{0};
  if (unit->typeTraits().is_floating_point(type)) return double{0.0};
  if (unit->typeTraits().is_pointer(type)) return std::intmax_t{0};
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

auto makeZeroClassInitList(TranslationUnit* unit, ClassSymbol* classSymbol)
    -> std::shared_ptr<InitializerList> {
  auto list = std::make_shared<InitializerList>();
  auto layout = classSymbol->layout();
  if (!layout) return list;

  for (auto member : classSymbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;
    auto info = layout->getFieldInfo(field);
    if (!info) continue;
    while (list->elements.size() <= info->index)
      list->elements.emplace_back(std::intmax_t{0}, nullptr);
    ConstValue zero = std::intmax_t{0};
    if (auto z = makeZeroConstValue(unit, field->type())) zero = *z;
    list->elements[info->index] = {zero, field->type()};
  }
  return list;
}

struct AnonMemberPath {
  FieldSymbol* anonField;
  ClassSymbol* anonClass;
};

auto findAnonymousMemberPath(ClassSymbol* classSymbol, FieldSymbol* target)
    -> std::optional<std::vector<AnonMemberPath>> {
  for (auto member : classSymbol->members()) {
    auto nested = symbol_cast<ClassSymbol>(member);
    if (!nested || nested->name()) continue;

    FieldSymbol* anonField = nullptr;
    for (auto m : classSymbol->members()) {
      auto f = symbol_cast<FieldSymbol>(m);
      if (!f) continue;
      if (auto ct = type_cast<ClassType>(f->type())) {
        if (ct->symbol() == nested) {
          anonField = f;
          break;
        }
      }
    }
    if (!anonField) continue;

    for (auto nm : nested->members()) {
      if (nm == target) {
        return std::vector<AnonMemberPath>{{anonField, nested}};
      }
    }

    auto sub = findAnonymousMemberPath(nested, target);
    if (sub) {
      sub->insert(sub->begin(), {anonField, nested});
      return sub;
    }
  }
  return std::nullopt;
}

auto setDesignatedValue(ASTInterpreter& interp,
                        const std::shared_ptr<InitializerList>& list,
                        List<DesignatorAST*>* designatorList,
                        const ConstValue& value, const Type* valueType)
    -> bool {
  if (!designatorList || !list) return false;

  auto subscript = ast_cast<SubscriptDesignatorAST>(designatorList->value);
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

  auto nestedPtr = std::get_if<std::shared_ptr<InitializerList>>(&elemVal);
  if (!nestedPtr || !*nestedPtr) return false;
  return setDesignatedValue(interp, *nestedPtr, designatorList->next, value,
                            valueType);
}
}  // namespace

auto ASTInterpreter::valueInitializeClass(const Type* type, ClassSymbol* symbol)
    -> std::shared_ptr<ConstObject> {
  auto obj = std::make_shared<ConstObject>(type);
  for (auto member : symbol->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;
    ConstValue zero = std::intmax_t{0};
    if (auto z = makeZeroConstValue(unit_, field->type())) zero = *z;
    obj->addField(field, std::move(zero));
  }
  applyNsdmis(obj);
  return obj;
}

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
      auto classSymbol = classType->symbol();
      if (!classSymbol) return std::nullopt;
      auto layout = classSymbol->layout();
      if (!layout) return std::nullopt;

      struct SlotInfo {
        size_t index;
        const Type* type;
        uint32_t bitOffset = 0;
        uint32_t bitWidth = 0;
      };

      std::unordered_map<FieldSymbol*, SlotInfo> fieldSlotMap;
      for (auto member : classSymbol->members()) {
        if (auto field = symbol_cast<FieldSymbol>(member)) {
          if (field->isStatic()) continue;
          if (auto info = layout->getFieldInfo(field))
            fieldSlotMap[field] = {info->index, field->type(), info->bitOffset,
                                   info->bitWidth};
        }
      }

      size_t maxSlot = 0;
      bool anyDot = false;
      for (auto node : ListView{ast->expressionList}) {
        auto desig = ast_cast<DesignatedInitializerClauseAST>(node);
        if (!desig || !desig->designatorList) continue;
        auto dot = ast_cast<DotDesignatorAST>(desig->designatorList->value);
        if (!dot || !dot->symbol) continue;
        auto it = fieldSlotMap.find(dot->symbol);
        if (it == fieldSlotMap.end()) {
          auto path = findAnonymousMemberPath(classSymbol, dot->symbol);
          if (path && !path->empty()) {
            auto topIt = fieldSlotMap.find((*path)[0].anonField);
            if (topIt != fieldSlotMap.end()) {
              maxSlot = std::max(maxSlot, topIt->second.index);
              anyDot = true;
            }
          }
          continue;
        }
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
        auto desig = ast_cast<DesignatedInitializerClauseAST>(node);
        if (!desig) return std::nullopt;
        if (!desig->designatorList) continue;
        auto dot = ast_cast<DotDesignatorAST>(desig->designatorList->value);
        if (!dot || !dot->symbol) continue;
        auto it = fieldSlotMap.find(dot->symbol);
        if (it == fieldSlotMap.end()) {
          auto path = findAnonymousMemberPath(classSymbol, dot->symbol);
          if (!path || path->empty()) continue;
          auto topIt = fieldSlotMap.find((*path)[0].anonField);
          if (topIt == fieldSlotMap.end()) continue;
          size_t topIdx = topIt->second.index;
          if (topIdx >= slotCount) continue;

          ExpressionAST* initExpr = nullptr;
          if (auto eq = ast_cast<EqualInitializerAST>(desig->initializer))
            initExpr = eq->expression;
          else
            initExpr = desig->initializer;
          if (!initExpr) continue;
          auto val = interp.evaluate(initExpr);
          if (!val) continue;

          auto& topSlot = slots[topIdx];
          if (!topSlot ||
              !std::holds_alternative<std::shared_ptr<InitializerList>>(
                  topSlot->first) ||
              !std::get<std::shared_ptr<InitializerList>>(topSlot->first) ||
              std::get<std::shared_ptr<InitializerList>>(topSlot->first)
                  ->elements.empty()) {
            auto list = makeZeroClassInitList(unit(), (*path)[0].anonClass);
            topSlot = {{ConstValue{list}, topIt->second.type}};
          }

          auto curList =
              &std::get<std::shared_ptr<InitializerList>>(topSlot->first);

          for (size_t pi = 1; pi < path->size(); ++pi) {
            auto prevClass = (*path)[pi - 1].anonClass;
            auto prevLayout = prevClass->layout();
            if (!prevLayout) break;
            auto subInfo = prevLayout->getFieldInfo((*path)[pi].anonField);
            if (!subInfo) break;
            size_t subIdx = subInfo->index;
            while ((*curList)->elements.size() <= subIdx)
              (*curList)->elements.emplace_back(std::intmax_t{0}, nullptr);
            auto& subVal = std::get<0>((*curList)->elements[subIdx]);
            auto subPtr =
                std::get_if<std::shared_ptr<InitializerList>>(&subVal);
            if (!subPtr || !*subPtr || (*subPtr)->elements.empty()) {
              auto newList =
                  makeZeroClassInitList(unit(), (*path)[pi].anonClass);
              (*curList)->elements[subIdx] = {ConstValue{newList},
                                              (*path)[pi].anonField->type()};
              subPtr = &std::get<std::shared_ptr<InitializerList>>(
                  std::get<0>((*curList)->elements[subIdx]));
            }
            curList = subPtr;
          }

          auto lastClass = path->back().anonClass;
          auto lastLayout = lastClass->layout();
          if (!lastLayout) continue;
          auto fieldInfo = lastLayout->getFieldInfo(dot->symbol);
          if (!fieldInfo) continue;
          size_t fieldIdx = fieldInfo->index;
          while ((*curList)->elements.size() <= fieldIdx)
            (*curList)->elements.emplace_back(std::intmax_t{0}, nullptr);

          auto curField = dot->symbol;
          auto designators = desig->designatorList->next;
          while (designators) {
            auto nextDot = ast_cast<DotDesignatorAST>(designators->value);
            if (!nextDot || !nextDot->symbol) break;
            auto fct = type_cast<ClassType>(curField->type());
            if (!fct || !fct->symbol()) break;
            auto fc = fct->symbol();
            auto fl = fc->layout();
            if (!fl) break;
            auto& fv = std::get<0>((*curList)->elements[fieldIdx]);
            auto fp = std::get_if<std::shared_ptr<InitializerList>>(&fv);
            if (!fp || !*fp || (*fp)->elements.empty()) {
              auto newList = makeZeroClassInitList(unit(), fc);
              (*curList)->elements[fieldIdx] = {ConstValue{newList},
                                                curField->type()};
              fp = &std::get<std::shared_ptr<InitializerList>>(
                  std::get<0>((*curList)->elements[fieldIdx]));
            }
            auto nextInfo = fl->getFieldInfo(nextDot->symbol);
            if (!nextInfo) break;
            curList = fp;
            fieldIdx = nextInfo->index;
            while ((*curList)->elements.size() <= fieldIdx)
              (*curList)->elements.emplace_back(std::intmax_t{0}, nullptr);
            curField = nextDot->symbol;
            designators = designators->next;
          }

          const Type* initType = desig->type ? desig->type : curField->type();
          (*curList)->elements[fieldIdx] = {*val, initType};
          continue;
        }
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

  if (arrayType && elementType) {
    const size_t size = arrayType->size();
    auto topList = std::make_shared<InitializerList>();
    topList->elements.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      auto slotZero = makeZeroConstValue(unit(), elementType);
      if (!slotZero) return std::nullopt;
      topList->elements.emplace_back(*slotZero, elementType);
    }

    size_t idx = 0;
    for (auto node : ListView{ast->expressionList}) {
      if (idx >= size) break;
      auto value = interp.evaluate(node);
      if (!value) return std::nullopt;
      const Type* nodeType = node->type ? node->type : elementType;
      topList->elements[idx] = {*value, nodeType};
      ++idx;
    }
    return ConstValue{topList};
  }

  if (!arrayType) {
    if (auto classType = type_cast<ClassType>(ast->type)) {
      if (auto classSymbol = classType->symbol()) {
        if (auto layout = classSymbol->layout();
            layout && !classSymbol->hasUserDeclaredConstructors()) {
          std::vector<FieldSymbol*> fields;
          for (auto member : classSymbol->members()) {
            if (auto field = symbol_cast<FieldSymbol>(member)) {
              if (!field->isStatic()) fields.push_back(field);
            }
          }

          bool hasBitfield = false;
          for (auto field : fields) {
            if (auto info = layout->getFieldInfo(field))
              if (info->bitWidth > 0) hasBitfield = true;
          }

          if (!hasBitfield && !classSymbol->isUnion()) {
            auto obj = interp.valueInitializeClass(ast->type, classSymbol);
            size_t fieldIdx = 0;
            for (auto node : ListView{ast->expressionList}) {
              if (fieldIdx >= fields.size()) break;
              auto field = fields[fieldIdx++];
              auto val = interp.evaluate(node);
              if (!val) continue;
              obj->setField(field, *val);
            }
            return ConstValue{std::move(obj)};
          }

          size_t maxSlot = 0;
          for (auto field : fields) {
            if (auto info = layout->getFieldInfo(field))
              maxSlot = std::max(maxSlot, static_cast<size_t>(info->index));
          }

          auto topList = std::make_shared<InitializerList>();
          topList->elements.resize(maxSlot + 1, {std::intmax_t{0}, nullptr});

          for (auto field : fields) {
            if (auto info = layout->getFieldInfo(field)) {
              if (!std::get<1>(topList->elements[info->index])) {
                ConstValue zero = std::intmax_t{0};
                if (info->bitWidth == 0) {
                  if (auto z = makeZeroConstValue(unit(), field->type()))
                    zero = *z;
                }
                topList->elements[info->index] = {zero, field->type()};
              }
            }
          }

          std::unordered_map<size_t, std::intmax_t> bitSlotAccum;
          size_t fieldIdx = 0;
          for (auto node : ListView{ast->expressionList}) {
            if (fieldIdx >= fields.size()) break;
            auto field = fields[fieldIdx++];
            auto info = layout->getFieldInfo(field);
            if (!info) continue;
            auto val = interp.evaluate(node);
            if (!val) continue;
            if (info->bitWidth > 0) {
              std::intmax_t bitVal = 0;
              if (auto iv = std::get_if<std::intmax_t>(&*val)) bitVal = *iv;
              std::intmax_t mask = (std::intmax_t(1) << info->bitWidth) - 1;
              bitSlotAccum[info->index] =
                  (bitSlotAccum[info->index] & ~(mask << info->bitOffset)) |
                  ((bitVal & mask) << info->bitOffset);
              topList->elements[info->index] = {bitSlotAccum[info->index],
                                                field->type()};
            } else {
              const Type* nodeType = node->type ? node->type : field->type();
              topList->elements[info->index] = {*val, nodeType};
            }
          }

          return ConstValue{topList};
        }
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
