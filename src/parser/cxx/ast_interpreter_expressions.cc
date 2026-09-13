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
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/initialization.h>
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

auto ASTInterpreter::zeroInitialize(const Type* type)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;
  if (traits.is_integral_or_enum(type)) return std::intmax_t{0};
  if (traits.is_floating_point(type)) return double{0.0};
  if (traits.is_pointer(type)) return std::intmax_t{0};
  if (auto complexType = unqualified_cast<ComplexType>(type)) {
    auto zero = zeroInitialize(complexType->elementType());
    if (!zero) return std::nullopt;
    return ConstValue{std::make_shared<ConstComplex>(*zero, *zero)};
  }
  if (auto arrayType = type_cast<BoundedArrayType>(type)) {
    auto list = std::make_shared<InitializerList>();
    list->elements.reserve(arrayType->size());
    for (size_t i = 0; i < arrayType->size(); ++i) {
      auto elementZero = zeroInitialize(arrayType->elementType());
      if (!elementZero) return std::nullopt;
      list->elements.emplace_back(*elementZero, arrayType->elementType());
    }
    return ConstValue{list};
  }

  auto classType = unqualified_cast<ClassType>(type);
  auto classSymbol = classType ? classType->symbol() : nullptr;
  if (!classSymbol) return std::nullopt;

  auto object = std::make_shared<ConstObject>(classType);
  for (auto element : traits.aggregate_elements(classSymbol)) {
    auto elementZero = zeroInitialize(traits.aggregate_element_type(element));
    if (!elementZero) return std::nullopt;
    object->addMember(element, std::move(*elementZero));
    if (classSymbol->isUnion()) break;
  }
  return ConstValue{object};
}

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

  [[nodiscard]] auto toLongDouble(const ConstValue& value) -> long double {
    return interp.toLongDouble(value).value_or(0.0L);
  }

  [[nodiscard]] auto toValue(std::uintmax_t value) -> ConstValue {
    return ConstValue(std::bit_cast<std::intmax_t>(value));
  }

  [[nodiscard]] auto floatingArithmetic(const Type* type,
                                        const ExpressionResult& left,
                                        const ExpressionResult& right,
                                        auto&& op) -> ExpressionResult {
    switch (type->kind()) {
      case TypeKind::kFloat:
        return ConstValue{op(toFloat(*left), toFloat(*right))};
      case TypeKind::kLongDouble:
        return ConstValue{op(toLongDouble(*left), toLongDouble(*right))};
      default:
        return ConstValue{op(toDouble(*left), toDouble(*right))};
    }
  }

  auto unary_plus_op(const Type* type, const ExpressionResult& operand)
      -> ExpressionResult {
    if (!operand.has_value()) return std::nullopt;

    if (unit()->typeTraits().is_complex(type)) return operand;

    if (unit()->typeTraits().is_integral_or_unscoped_enum(type)) return operand;

    switch (type->kind()) {
      case TypeKind::kFloat:
        return ConstValue{toFloat(*operand)};
      case TypeKind::kDouble:
        return ConstValue{toDouble(*operand)};
      case TypeKind::kLongDouble:
        return ConstValue{toLongDouble(*operand)};
      default:
        return std::nullopt;
    }
  }

  auto unary_minus_op(const Type* type, const ExpressionResult& operand)
      -> ExpressionResult {
    if (!operand.has_value()) return std::nullopt;

    if (auto complexType = unqualified_cast<ComplexType>(type)) {
      auto elementType = complexType->elementType();
      auto parts = complexParts(*operand, elementType);
      if (!parts) return std::nullopt;
      return makeComplex(unary_minus_op(elementType, parts->first),
                         unary_minus_op(elementType, parts->second));
    }

    if (unit()->typeTraits().is_integral_or_unscoped_enum(type)) {
      const auto sz = memoryLayout()->sizeOf(type);

      if (unit()->typeTraits().is_unsigned(type)) {
        if (sz <= 4) return toValue(-toUInt32(*operand));
        return toValue(-toUInt64(*operand));
      }

      if (sz <= 4) return ConstValue{-toInt32(*operand)};
      return ConstValue{-toInt64(*operand)};
    }

    switch (type->kind()) {
      case TypeKind::kFloat:
        return ConstValue{-toFloat(*operand)};
      case TypeKind::kDouble:
        return ConstValue{-toDouble(*operand)};
      case TypeKind::kLongDouble:
        return ConstValue{-toLongDouble(*operand)};
      default:
        return std::nullopt;
    }
  }

  auto star_op(const Type* type, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
    const auto sz = memoryLayout()->sizeOf(type);

    if (unit()->typeTraits().is_floating_point(type)) {
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a * b; });
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
      if (toDouble(*right) == 0.0) return std::nullopt;
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a / b; });
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

  auto complexParts(const ConstValue& value, const Type* elementType)
      -> std::optional<std::pair<ConstValue, ConstValue>> {
    if (auto complexValue =
            std::get_if<std::shared_ptr<ConstComplex>>(&value)) {
      if (!*complexValue) return std::nullopt;
      return std::pair{(*complexValue)->real(), (*complexValue)->imag()};
    }

    auto zero = interp.zeroInitialize(elementType);
    if (!zero) return std::nullopt;
    return std::pair{value, *zero};
  }

  auto makeComplex(const ExpressionResult& real, const ExpressionResult& imag)
      -> ExpressionResult {
    if (!real.has_value() || !imag.has_value()) return std::nullopt;
    return ConstValue{std::make_shared<ConstComplex>(*real, *imag)};
  }

  auto complex_op(TokenKind op, const ComplexType* complexType,
                  const ExpressionResult& left, const ExpressionResult& right)
      -> ExpressionResult {
    auto elementType = complexType->elementType();

    auto lhs = complexParts(*left, elementType);
    auto rhs = complexParts(*right, elementType);
    if (!lhs || !rhs) return std::nullopt;

    const auto& a = lhs->first;
    const auto& b = lhs->second;
    const auto& c = rhs->first;
    const auto& d = rhs->second;

    auto mul = [&](const ConstValue& x, const ConstValue& y) {
      return star_op(elementType, ExpressionResult{x}, ExpressionResult{y});
    };

    auto add = [&](const ExpressionResult& x, const ExpressionResult& y) {
      if (!x.has_value() || !y.has_value()) return ExpressionResult{};
      return plus_op(elementType, x, y);
    };

    auto sub = [&](const ExpressionResult& x, const ExpressionResult& y) {
      if (!x.has_value() || !y.has_value()) return ExpressionResult{};
      return minus_op(elementType, x, y);
    };

    auto div = [&](const ExpressionResult& x, const ExpressionResult& y) {
      if (!x.has_value() || !y.has_value()) return ExpressionResult{};
      return slash_op(elementType, x, y);
    };

    switch (op) {
      case TokenKind::T_PLUS:
        return makeComplex(add(a, c), add(b, d));

      case TokenKind::T_MINUS:
        return makeComplex(sub(a, c), sub(b, d));

      case TokenKind::T_STAR:
        return makeComplex(sub(mul(a, c), mul(b, d)),
                           add(mul(a, d), mul(b, c)));

      case TokenKind::T_SLASH: {
        auto denominator = add(mul(c, c), mul(d, d));
        return makeComplex(div(add(mul(a, c), mul(b, d)), denominator),
                           div(sub(mul(b, c), mul(a, d)), denominator));
      }

      case TokenKind::T_EQUAL_EQUAL:
      case TokenKind::T_EXCLAIM_EQUAL: {
        auto realEqual = equal_equal_op(elementType, ExpressionResult{a},
                                        ExpressionResult{c});
        auto imagEqual = equal_equal_op(elementType, ExpressionResult{b},
                                        ExpressionResult{d});
        if (!realEqual.has_value() || !imagEqual.has_value())
          return std::nullopt;
        const auto equal = toBool(*realEqual) && toBool(*imagEqual);
        const auto result = op == TokenKind::T_EQUAL_EQUAL ? equal : !equal;
        return ConstValue{std::intmax_t{result ? 1 : 0}};
      }

      default:
        return std::nullopt;
    }
  }

  auto plus_op(const Type* type, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
    const auto sz = memoryLayout()->sizeOf(type);

    if (unit()->typeTraits().is_floating_point(type)) {
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a + b; });
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
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a - b; });
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

  [[nodiscard]] auto operator()(DefaultInitializerExpressionAST* ast)
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

  [[nodiscard]] auto evaluateOperatorCall(
      FunctionSymbol* function, ExpressionResult operand,
      std::optional<ExpressionResult> extraArgument = std::nullopt)
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

  [[nodiscard]] auto evaluateMemberFunctionPointerConversion(
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

  [[nodiscard]] auto complexValue(BracedInitListAST* ast,
                                  const ComplexType* complexType)
      -> ExpressionResult;

  [[nodiscard]] auto aggregateObject(BracedInitListAST* ast,
                                     ClassSymbol* classSymbol)
      -> ExpressionResult;

  [[nodiscard]] auto arrayValue(BracedInitListAST* ast,
                                const BoundedArrayType* type)
      -> ExpressionResult;

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
  if (auto definition = symbol_cast<VariableSymbol>(field->definition())) {
    if (isDeclaredConstant(definition) && definition->constValue())
      return cloneValue(*definition->constValue());
  }
  if (field->constValue()) return cloneValue(*field->constValue());

  fieldsUnderEvaluation_.push_back(field);
  auto result = expression(field->initializer());
  fieldsUnderEvaluation_.pop_back();

  return result;
}

auto ASTInterpreter::lvalue(ExpressionAST* ast) -> ConstValue* {
  if (!ast) return nullptr;

  while (ast) {
    if (auto initializer = ast_cast<DefaultInitializerExpressionAST>(ast)) {
      InitializerContextGuard guard{*this, initializer->context};
      return lvalue(initializer->expression);
    }
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
      if (!constant->constValue) return nullptr;
      auto address =
          std::get_if<std::shared_ptr<ConstAddress>>(constant->constValue);
      if (address && *address) return addressSlot(**address, 0);
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
        return subobjectSlot(thisObject_, field);
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
      return nullptr;
    }
    return nullptr;
  }

  if (auto member = ast_cast<MemberExpressionAST>(ast)) {
    if (!member->symbol) return nullptr;
    auto field = symbol_cast<FieldSymbol>(member->symbol);
    if (!field || field->isStatic()) return nullptr;

    if (auto object = memberObject(member)) return subobjectSlot(object, field);
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

  if (auto call = ast_cast<CallExpressionAST>(ast))
    return evaluateCallExpression(call, CallResultKind::kLValue).lvalue;

  return nullptr;
}

auto ASTInterpreter::loadAddress(const ConstAddress& address,
                                 std::intmax_t extraIndex,
                                 const Type* objectType)
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
    if (auto fv = address.owner()->subobject(sym)) storage = *fv;
  } else if (auto slot = lookupLocalSlot(sym)) {
    storage = *slot;
  } else if (auto var = symbol_cast<VariableSymbol>(sym)) {
    if (auto cv = var->constValue())
      storage = cv;
    else if (var->initializer())
      storage = expression(var->initializer());
  }
  if (!storage.has_value()) return std::nullopt;
  if (index == 0 && objectType) {
    if (traits.is_same(traits.remove_cvref(sym->type()),
                       traits.remove_cvref(objectType)))
      return storage;
  }

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&*storage)) {
    if (!*list || static_cast<std::size_t>(index) >= (*list)->elements.size())
      return std::nullopt;
    return std::get<0>((*list)->elements[index]);
  }

  if (index == 0) return storage;
  return std::nullopt;
}

auto ASTInterpreter::addressSlot(const ConstAddress& address,
                                 std::intmax_t extraIndex,
                                 const Type* objectType) -> ConstValue* {
  const auto index = address.offset() + extraIndex;
  if (index < 0) return nullptr;

  if (address.stringLiteral()) return nullptr;

  auto sym = address.symbol();
  if (!sym) return nullptr;

  auto slot = address.owner() ? subobjectSlot(address.owner(), sym)
                              : lookupLocalSlot(sym);
  if (!slot) return nullptr;
  if (index == 0 && objectType) {
    if (traits.is_same(traits.remove_cvref(sym->type()),
                       traits.remove_cvref(objectType)))
      return slot;
  }

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(slot)) {
    if (!*list || static_cast<std::size_t>(index) >= (*list)->elements.size())
      return nullptr;
    return &std::get<0>((*list)->elements[index]);
  }

  if (index == 0) return slot;
  return nullptr;
}

auto ASTInterpreter::memberObject(MemberExpressionAST* ast)
    -> std::shared_ptr<ConstObject> {
  auto value = expression(ast->baseExpression);
  if (!value) {
    auto base = Initializer{ast->baseExpression}.clause();
    if (auto id = ast_cast<IdExpressionAST>(base);
        id && symbol_cast<VariableSymbol>(id->symbol) &&
        traits.is_empty(traits.remove_cvref(id->type)))
      return std::make_shared<ConstObject>(traits.remove_cvref(id->type));
    return {};
  }
  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value)) {
      if (!*address) return {};
      value = loadAddress(**address, 0);
    }
  }
  if (!value) return {};
  auto object = std::get_if<std::shared_ptr<ConstObject>>(&*value);
  return object ? *object : nullptr;
}

auto ASTInterpreter::fieldOwner(ExpressionAST* ast)
    -> std::shared_ptr<ConstObject> {
  if (auto id = ast_cast<IdExpressionAST>(ast)) {
    if (symbol_cast<FieldSymbol>(id->symbol)) return thisObject_;
    return nullptr;
  }
  if (auto member = ast_cast<MemberExpressionAST>(ast)) {
    if (!symbol_cast<FieldSymbol>(member->symbol)) return nullptr;
    return memberObject(member);
  }
  return nullptr;
}

auto ASTInterpreter::typeInfoAddress(const Type* type)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;
  return std::make_shared<ConstAddress>(
      traits.remove_cv(traits.remove_reference(type)));
}

auto ASTInterpreter::addressOfLvalue(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  while (ast) {
    if (auto initializer = ast_cast<DefaultInitializerExpressionAST>(ast)) {
      InitializerContextGuard guard{*this, initializer->context};
      return addressOfLvalue(initializer->expression);
    }
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
      if (!constant->constValue) return std::nullopt;
      if (std::holds_alternative<std::shared_ptr<ConstAddress>>(
              *constant->constValue))
        return *constant->constValue;
      ast = constant->expression;
      continue;
    }
    break;
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast)) {
    if (!idExpr->symbol) return std::nullopt;
    if (traits.is_reference(idExpr->symbol->type())) {
      for (auto frame = frames_.rbegin(); frame != frames_.rend(); ++frame) {
        auto address = frame->referenceAddresses.find(idExpr->symbol);
        if (address != frame->referenceAddresses.end()) return address->second;
      }
      if (auto variable = symbol_cast<VariableSymbol>(idExpr->symbol)) {
        if (variable->constValue()) return variable->constValue();
      }
      return std::nullopt;
    }
    if (symbol_cast<FieldSymbol>(idExpr->symbol)) {
      if (auto owner = fieldOwner(ast))
        return std::make_shared<ConstAddress>(owner, idExpr->symbol);
      return std::nullopt;
    }
    if (auto function = resolvedFunction(idExpr->symbol))
      return std::make_shared<ConstAddress>(function);
    return std::make_shared<ConstAddress>(idExpr->symbol);
  }

  if (auto member = ast_cast<MemberExpressionAST>(ast)) {
    if (!symbol_cast<FieldSymbol>(member->symbol)) return std::nullopt;
    if (auto owner = fieldOwner(ast))
      return std::make_shared<ConstAddress>(owner, member->symbol);
    return std::nullopt;
  }

  if (auto typeidOfType = ast_cast<TypeidOfTypeExpressionAST>(ast)) {
    return typeInfoAddress(typeidOfType->typeId->type);
  }

  if (auto typeidExpression = ast_cast<TypeidExpressionAST>(ast)) {
    auto operand = typeidExpression->expression;
    if (!operand) return std::nullopt;
    if (operand->valueCategory != ValueCategory::kPrValue &&
        traits.is_polymorphic(operand->type))
      return std::nullopt;
    return typeInfoAddress(operand->type);
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

  if (auto call = ast_cast<CallExpressionAST>(ast))
    return evaluateCallExpression(call, CallResultKind::kAddress).value;

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
  if (ast->literalOperatorCall) return evaluate(ast->literalOperatorCall);
  return ConstValue(ast->literal->charValue());
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BoolLiteralExpressionAST* ast) -> ExpressionResult {
  return ConstValue(ast->isTrue);
}

auto ASTInterpreter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  if (ast->literalOperatorCall) return evaluate(ast->literalOperatorCall);
  const auto value = static_cast<std::uintmax_t>(ast->literal->integerValue());
  return ExpressionResult{std::bit_cast<std::intmax_t>(value)};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    FloatLiteralExpressionAST* ast) -> ExpressionResult {
  if (ast->literalOperatorCall) return evaluate(ast->literalOperatorCall);
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
  if (ast->literalOperatorCall) return evaluate(ast->literalOperatorCall);
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
  auto selected = getGenericSelectionExpression(ast);
  if (!selected) return std::nullopt;
  return interp.expression(selected);
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    DefaultInitializerExpressionAST* ast) -> ExpressionResult {
  InitializerContextGuard guard{interp, ast->context};
  return interp.expression(ast->expression);
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
      field && field->isStatic()) {
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
      auto fieldVal = interp.thisObject()->subobject(field);
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

    closure->addMember(captureField, std::move(*value));
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

  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};

  auto isValidExpression =
      [&](ExpressionAST* expression) -> std::optional<bool> {
    if (!expression) return std::nullopt;

    {
      SilentDiagnosticsScope silent{unit_};
      auto typeChecker = TypeChecker{unit_};
      typeChecker.setScope(scope);
      typeChecker.setReportErrors(true);
      typeChecker.check(&expression);
      if (silent.hadError()) return false;
    }

    if (expression->type && containsPlaceholderType(expression->type)) {
      if (auto [base, callee] = calledFunction(expression); callee) {
        ASTRewriter::completeDeducedReturnType(unit_, callee);
        base->type = callee->type();

        SilentDiagnosticsScope silent{unit_};
        auto typeChecker = TypeChecker{unit_};
        typeChecker.setScope(scope);
        typeChecker.setReportErrors(true);
        typeChecker.check(&expression);
        if (silent.hadError()) return false;
      }
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
    Symbol* resolved = nullptr;
    bool hadError = false;

    {
      SilentDiagnosticsScope silent{unit_};
      resolved = Binder{unit_}.resolve(typeRequirement->nestedNameSpecifier,
                                       typeRequirement->unqualifiedId,
                                       /*checkTemplates=*/true);
      hadError = silent.hadError();
    }

    if (hadError) return false;
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

auto ASTInterpreter::evaluateCallExpression(CallExpressionAST* ast,
                                            CallResultKind kind) -> CallResult {
  auto base = ast->baseExpression;
  while (auto nested = ast_cast<NestedExpressionAST>(base))
    base = nested->expression;
  FunctionSymbol* function = nullptr;
  std::shared_ptr<ConstObject> object;
  if (auto id = ast_cast<IdExpressionAST>(base)) {
    auto builtin = resolveBuiltinFunctionKind(id);
    if (builtin != BuiltinFunctionKind::T_NONE) {
      std::vector<ConstValue> arguments;
      if (!builtinEvaluatesItsOwnArguments(builtin)) {
        for (auto argument : ListView{ast->expressionList}) {
          auto value = expression(argument);
          if (!value) return {};
          arguments.push_back(std::move(*value));
        }
      }
      return {evaluateBuiltinCall(builtin, std::move(arguments), ast), nullptr};
    }
    function = resolvedFunction(id->symbol);
  } else if (auto member = ast_cast<MemberExpressionAST>(base)) {
    function = resolvedFunction(member->symbol);
    if (!function) return {};
    if (function->isImplicitObjectMemberFunction()) {
      object = memberObject(member);
      if (!object) return {};
    } else if (!expression(member->baseExpression))
      return {};
  }
  if (ast->constructorSymbol) {
    std::vector<ExpressionAST*> arguments;
    for (auto argument : ListView{ast->expressionList})
      arguments.push_back(argument);
    return {evaluateConstructorFromExprs(ast->constructorSymbol, ast->type,
                                         arguments),
            nullptr};
  }
  if (!function) {
    auto value = expression(base);
    if (!value) return {};
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value);
    if (address && *address)
      function = symbol_cast<FunctionSymbol>((*address)->symbol());
  }
  if (!function || !function->isConstexpr()) return {};
  Frame frame;
  std::vector<ExpressionAST*> arguments;
  for (auto argument : ListView{ast->expressionList})
    arguments.push_back(argument);
  if (!bindParametersFromExprs(frame, function, arguments)) return {};
  return executeFunction(function, std::move(frame), kind, std::move(object));
}

auto ASTInterpreter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  return interp.evaluateCallExpression(ast, CallResultKind::kValue).value;
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);

  if (!ast->type) return std::nullopt;
  if (auto classType = unqualified_cast<ClassType>(ast->type)) {
    auto classSymbol = classType->symbol();
    if (!classSymbol) return std::nullopt;
    if (!ast->expressionList &&
        unit()->typeTraits().requires_zero_initialization(
            ast->type, ast->constructorSymbol))
      return ConstValue{interp.valueInitializeClass(ast->type, classSymbol)};
    std::vector<ExpressionAST*> arguments;
    for (auto argument : ListView{ast->expressionList})
      arguments.push_back(argument);
    return interp.evaluateConstructorFromExprs(ast->constructorSymbol,
                                               ast->type, arguments);
  }
  if (!ast->expressionList) return interp.zeroInitialize(ast->type);
  if (!ast->expressionList->next)
    return interp.expression(ast->expressionList->value);

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
    if (!ast->constructorSymbol->isConstexpr())
      return ExpressionResult{std::nullopt};

    return interp.evaluateConstructorFromExprs(
        ast->constructorSymbol, ast->type, constructorArgumentExpressions(ast));
  }

  return interp.expression(ast->bracedInitList);
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    SpliceMemberExpressionAST* ast) -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);
  auto splicerResult = interp.splicer(ast->splicer);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  if (auto object = interp.memberObject(ast)) {
    if (ast->symbol) {
      if (auto value = object->subobject(ast->symbol)) return *value;
    }
  }

  if (interp.thisObject() && ast->symbol) {
    auto fieldVal = interp.thisObject()->subobject(ast->symbol);
    if (fieldVal) return *fieldVal;
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  if (ast->symbol) {
    if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
        field && field->isStatic()) {
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

  if (ast->symbol) {
    return evaluateOperatorCall(ast->symbol,
                                interp.expression(ast->baseExpression),
                                ExpressionResult{std::intmax_t{0}});
  }

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

  if (auto idExpression = ast_cast<IdExpressionAST>(ast->expression);
      idExpression && idExpression->symbol) {
    return ConstValue{std::make_shared<Meta>(
        static_cast<const Symbol*>(idExpression->symbol))};
  }

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    LabelAddressExpressionAST* ast) -> ExpressionResult {
  return ConstValue{
      std::make_shared<ConstLabelAddress>(ast->identifier->name())};
}

auto ASTInterpreter::ExpressionVisitor::evaluateOperatorCall(
    FunctionSymbol* function, ExpressionResult operand,
    std::optional<ExpressionResult> extraArgument) -> ExpressionResult {
  if (!operand.has_value()) return std::nullopt;

  std::vector<ConstValue> arguments;
  if (extraArgument) {
    if (!extraArgument->has_value()) return std::nullopt;
    arguments.push_back(std::move(**extraArgument));
  }

  if (function->isImplicitObjectMemberFunction()) {
    auto object = std::get_if<std::shared_ptr<ConstObject>>(&*operand);
    if (!object || !*object) return std::nullopt;
    return interp.evaluateCall(function, std::move(arguments), *object);
  }

  arguments.insert(arguments.begin(), std::move(*operand));
  return interp.evaluateCall(function, std::move(arguments));
}

auto ASTInterpreter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression);

  if (ast->symbol) return evaluateOperatorCall(ast->symbol, expressionResult);

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
      if (auto result = unary_minus_op(ast->expression->type, expressionResult))
        return result;
      break;
    }

    case TokenKind::T___REAL__:
    case TokenKind::T___IMAG__: {
      if (!expressionResult.has_value()) break;

      const auto isReal = ast->op == TokenKind::T___REAL__;

      if (auto complexType =
              unqualified_cast<ComplexType>(ast->expression->type)) {
        auto parts =
            complexParts(*expressionResult, complexType->elementType());
        if (!parts) break;
        return isReal ? parts->first : parts->second;
      }

      if (isReal) return expressionResult;
      return interp.zeroInitialize(ast->type);
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
      if (auto complexType =
              unqualified_cast<ComplexType>(ast->expression->type)) {
        if (!expressionResult.has_value()) break;
        auto elementType = complexType->elementType();
        auto parts = complexParts(*expressionResult, elementType);
        if (!parts) break;
        return makeComplex(ExpressionResult{parts->first},
                           unary_minus_op(elementType, parts->second));
      }

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
      if (auto result = unary_plus_op(ast->expression->type, expressionResult))
        return result;
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

      if (auto pointerType = type_cast<MemberFunctionPointerType>(ast->type)) {
        auto idExpr = ast_cast<IdExpressionAST>(innerExpr);
        if (!idExpr) break;
        auto function = symbol_cast<FunctionSymbol>(idExpr->symbol);
        if (!function) break;
        auto declaringClass = symbol_cast<ClassSymbol>(function->parent());
        if (!declaringClass) break;
        auto adjustment = classSubobjectOffset(pointerType->classType(),
                                               declaringClass->type());
        if (!adjustment) break;
        return ExpressionResult{
            std::make_shared<ConstAddress>(function, *adjustment)};
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
  if (auto ct = unqualified_cast<ClassType>(ast->expression->type))
    unit()->typeTraits().requireCompleteClass(ct->symbol());
  auto size = memoryLayout()->sizeOf(ast->expression->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->typeId || !ast->typeId->type) return std::nullopt;
  if (auto ct = unqualified_cast<ClassType>(ast->typeId->type))
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
  if (auto ct = unqualified_cast<ClassType>(ast->typeId->type))
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

  auto paren = ast_cast<ParenInitializerAST>(ast->expression);
  if (!paren) return std::nullopt;
  std::vector<ExpressionAST*> arguments;
  for (auto argument : ListView{paren->expressionList})
    arguments.push_back(argument);
  return interp.evaluateConstructorFromExprs(constructor, ast->type, arguments);
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

auto ASTInterpreter::ExpressionVisitor::evaluateMemberFunctionPointerConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto targetType = type_cast<MemberFunctionPointerType>(ast->type);
  if (!targetType) return std::nullopt;

  auto nullPointer = [] {
    return ExpressionResult{
        std::make_shared<ConstAddress>(static_cast<Symbol*>(nullptr))};
  };

  auto sourceType =
      ast->expression
          ? type_cast<MemberFunctionPointerType>(ast->expression->type)
          : nullptr;

  if (!sourceType) return nullPointer();

  auto value = evaluate(ast->expression);
  if (!value.has_value()) return std::nullopt;

  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value);
  if (!address || !*address) return std::nullopt;
  if (!(*address)->symbol()) return nullPointer();

  auto adjustment = memberPointerBaseAdjustment(sourceType, targetType);
  if (!adjustment.has_value()) return std::nullopt;

  return ExpressionResult{std::make_shared<ConstAddress>(
      (*address)->symbol(), (*address)->offset() + *adjustment)};
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
    if (type_cast<MemberFunctionPointerType>(ast->type))
      return evaluateMemberFunctionPointerConversion(ast);
    return evaluateMemberObjectPointerConversion(ast);
  }

  if (ast->castKind == ImplicitCastKind::kArrayToPointerConversion) {
    auto innerExpr = Initializer{ast->expression}.clause();
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

    case TypeKind::kComplex:
      return interp.toArithmeticType(*value, ast->type);

    default:
      if (unit()->typeTraits().is_integral_or_enum(ast->type)) {
        return interp.toIntegralType(*value, ast->type);
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
    const bool sameTarget = (*leftAddr)->sameTarget(**rightAddr);

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

  if (auto complexType =
          unqualified_cast<ComplexType>(unit()->typeTraits().remove_cv(type))) {
    return complex_op(op, complexType, left, right);
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

  if (ast->symbol)
    return evaluateOperatorCall(ast->symbol,
                                interp.expression(ast->leftExpression),
                                rightExpressionResult);

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
  if (ast->symbol) {
    auto right = interp.expression(ast->rightExpression);
    return evaluateOperatorCall(
        ast->symbol, interp.expression(ast->targetExpression), right);
  }
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

namespace {
[[nodiscard]] auto typeTraitRequiresCompleteType(BuiltinTypeTraitKind trait)
    -> bool {
  switch (trait) {
    case BuiltinTypeTraitKind::T___IS_VOID:
    case BuiltinTypeTraitKind::T___IS_NULL_POINTER:
    case BuiltinTypeTraitKind::T___IS_INTEGRAL:
    case BuiltinTypeTraitKind::T___IS_FLOATING_POINT:
    case BuiltinTypeTraitKind::T___IS_ARRAY:
    case BuiltinTypeTraitKind::T___IS_ENUM:
    case BuiltinTypeTraitKind::T___IS_SCOPED_ENUM:
    case BuiltinTypeTraitKind::T___IS_UNION:
    case BuiltinTypeTraitKind::T___IS_CLASS:
    case BuiltinTypeTraitKind::T___IS_FUNCTION:
    case BuiltinTypeTraitKind::T___IS_POINTER:
    case BuiltinTypeTraitKind::T___IS_MEMBER_OBJECT_POINTER:
    case BuiltinTypeTraitKind::T___IS_MEMBER_FUNCTION_POINTER:
    case BuiltinTypeTraitKind::T___IS_LVALUE_REFERENCE:
    case BuiltinTypeTraitKind::T___IS_RVALUE_REFERENCE:
    case BuiltinTypeTraitKind::T___IS_FUNDAMENTAL:
    case BuiltinTypeTraitKind::T___IS_ARITHMETIC:
    case BuiltinTypeTraitKind::T___IS_SCALAR:
    case BuiltinTypeTraitKind::T___IS_OBJECT:
    case BuiltinTypeTraitKind::T___IS_COMPOUND:
    case BuiltinTypeTraitKind::T___IS_REFERENCE:
    case BuiltinTypeTraitKind::T___IS_MEMBER_POINTER:
    case BuiltinTypeTraitKind::T___IS_BOUNDED_ARRAY:
    case BuiltinTypeTraitKind::T___IS_UNBOUNDED_ARRAY:
    case BuiltinTypeTraitKind::T___IS_CONST:
    case BuiltinTypeTraitKind::T___IS_VOLATILE:
    case BuiltinTypeTraitKind::T___IS_SIGNED:
    case BuiltinTypeTraitKind::T___IS_UNSIGNED:
    case BuiltinTypeTraitKind::T___IS_SAME:
    case BuiltinTypeTraitKind::T___IS_SAME_AS:
    case BuiltinTypeTraitKind::T___BUILTIN_TYPES_COMPATIBLE_P:
      return false;
    default:
      return true;
  }
}
}  // namespace

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
    if (typeTraitRequiresCompleteType(ast->typeTrait)) {
      if (auto classType = unqualified_cast<ClassType>(firstType)) {
        unit()->typeTraits().requireCompleteClass(classType->symbol());
      }
      if (secondType) {
        if (auto classType = unqualified_cast<ClassType>(secondType)) {
          unit()->typeTraits().requireCompleteClass(classType->symbol());
        }
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
        return unit()->typeTraits().is_floating(firstType);

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

      case BuiltinTypeTraitKind::T___REFERENCE_CONSTRUCTS_FROM_TEMPORARY: {
        if (!secondType) break;
        return unit()->typeTraits().reference_constructs_from_temporary(
            firstType, secondType);
      }

      case BuiltinTypeTraitKind::T___REFERENCE_CONVERTS_FROM_TEMPORARY: {
        if (!secondType) break;
        return unit()->typeTraits().reference_converts_from_temporary(
            firstType, secondType);
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
[[nodiscard]] auto initializerExpression(ExpressionAST* ast) -> ExpressionAST* {
  if (auto equalInitializer = ast_cast<EqualInitializerAST>(ast))
    return equalInitializer->expression;
  return ast;
}

[[nodiscard]] auto designatedSlot(ASTInterpreter& interp, ConstValue* slot,
                                  List<DesignatorAST*>* designatorList)
    -> ConstValue* {
  for (auto designator : ListView{designatorList}) {
    if (!slot) return nullptr;

    if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
      if (!dot->symbol) return nullptr;
      auto object = std::get_if<std::shared_ptr<ConstObject>>(slot);
      if (!object || !*object) return nullptr;
      slot = interp.subobjectSlot(*object, dot->symbol);
      continue;
    }

    auto subscript = ast_cast<SubscriptDesignatorAST>(designator);
    if (!subscript) return nullptr;
    auto list = std::get_if<std::shared_ptr<InitializerList>>(slot);
    if (!list || !*list) return nullptr;
    auto indexValue = interp.evaluate(subscript->expression);
    if (!indexValue) return nullptr;
    auto index = interp.toUInt(*indexValue);
    if (!index || *index >= (*list)->elements.size()) return nullptr;
    slot = &std::get<0>((*list)->elements[*index]);
  }
  return slot;
}
}  // namespace

auto ASTInterpreter::valueInitializeClass(const Type* type, ClassSymbol* symbol)
    -> std::shared_ptr<ConstObject> {
  auto object = std::make_shared<ConstObject>(type);
  for (auto element : traits.aggregate_elements(symbol)) {
    ConstValue zero = std::intmax_t{0};
    if (auto elementZero =
            zeroInitialize(traits.aggregate_element_type(element)))
      zero = *elementZero;
    object->addMember(element, std::move(zero));
    if (symbol->isUnion()) break;
  }
  applyNsdmis(object);
  return object;
}

auto ASTInterpreter::ExpressionVisitor::aggregateObject(
    BracedInitListAST* ast, ClassSymbol* classSymbol) -> ExpressionResult {
  auto elements = unit()->typeTraits().aggregate_elements(classSymbol);
  auto object = std::make_shared<ConstObject>(ast->type);

  std::size_t elementIndex = 0;
  for (auto node : ListView{ast->expressionList}) {
    auto clause = node;
    Symbol* element = nullptr;
    List<DesignatorAST*>* subobjectDesignators = nullptr;

    if (auto designated = ast_cast<DesignatedInitializerClauseAST>(node)) {
      clause = initializerExpression(designated->initializer);
      auto dot = ast_cast<DotDesignatorAST>(designated->designatorList->value);
      if (!dot) return std::nullopt;
      auto designatedElement = std::ranges::find(elements, dot->symbol);
      if (designatedElement == elements.end()) return std::nullopt;
      element = *designatedElement;
      elementIndex = std::distance(elements.begin(), designatedElement);
      subobjectDesignators = designated->designatorList->next;
    }

    if (!element) {
      if (elementIndex >= elements.size()) break;
      element = elements[elementIndex];
    }
    ++elementIndex;

    auto value = interp.evaluate(clause);
    if (!value) return std::nullopt;

    if (!subobjectDesignators) {
      object->setMember(element, std::move(*value));
    } else {
      if (!object->subobject(element)) {
        auto zero = interp.zeroInitialize(
            unit()->typeTraits().aggregate_element_type(element));
        if (!zero) return std::nullopt;
        object->addMember(element, std::move(*zero));
      }
      auto slot = designatedSlot(interp, interp.subobjectSlot(object, element),
                                 subobjectDesignators);
      if (!slot) return std::nullopt;
      *slot = std::move(*value);
    }

    if (classSymbol->isUnion()) break;
  }

  if (!classSymbol->isUnion()) {
    for (; elementIndex < elements.size(); ++elementIndex) {
      auto element = elements[elementIndex];
      auto zero = interp.zeroInitialize(
          unit()->typeTraits().aggregate_element_type(element));
      if (!zero) return std::nullopt;
      object->setMember(element, std::move(*zero));
    }
  } else if (object->members().empty() && !elements.empty()) {
    auto zero = interp.zeroInitialize(
        unit()->typeTraits().aggregate_element_type(elements.front()));
    if (!zero) return std::nullopt;
    object->addMember(elements.front(), std::move(*zero));
  }

  return ConstValue{std::move(object)};
}

auto ASTInterpreter::ExpressionVisitor::arrayValue(BracedInitListAST* ast,
                                                   const BoundedArrayType* type)
    -> ExpressionResult {
  auto elementType = type->elementType();

  if (unit()->typeTraits().is_narrow_char_type(elementType) &&
      ast->expressionList && !ast->expressionList->next) {
    if (auto literal =
            ast_cast<StringLiteralExpressionAST>(ast->expressionList->value)) {
      return ConstValue(literal->literal);
    }
  }

  auto list = std::make_shared<InitializerList>();
  list->elements.reserve(type->size());
  for (std::size_t i = 0; i < type->size(); ++i) {
    auto elementZero = interp.zeroInitialize(elementType);
    if (!elementZero) return std::nullopt;
    list->elements.emplace_back(*elementZero, elementType);
  }

  std::size_t elementIndex = 0;
  for (auto node : ListView{ast->expressionList}) {
    auto clause = node;
    List<DesignatorAST*>* subobjectDesignators = nullptr;

    if (auto designated = ast_cast<DesignatedInitializerClauseAST>(node)) {
      clause = initializerExpression(designated->initializer);
      auto designatorList = designated->designatorList;
      if (auto subscript =
              ast_cast<SubscriptDesignatorAST>(designatorList->value)) {
        auto indexValue = interp.evaluate(subscript->expression);
        if (!indexValue) return std::nullopt;
        auto index = interp.toUInt(*indexValue);
        if (!index) return std::nullopt;
        elementIndex = *index;
        subobjectDesignators = designatorList->next;
      }
    }

    if (elementIndex < type->size() && clause) {
      auto value = interp.evaluate(clause);
      if (!value) return std::nullopt;
      auto slot =
          designatedSlot(interp, &std::get<0>(list->elements[elementIndex]),
                         subobjectDesignators);
      if (!slot) return std::nullopt;
      *slot = std::move(*value);
    }
    ++elementIndex;
  }

  return ConstValue{std::move(list)};
}

auto ASTInterpreter::ExpressionVisitor::complexValue(
    BracedInitListAST* ast, const ComplexType* complexType)
    -> ExpressionResult {
  auto elementType = complexType->elementType();

  auto zero = interp.zeroInitialize(elementType);
  if (!zero) return std::nullopt;

  ConstValue parts[2] = {*zero, *zero};
  std::size_t index = 0;

  for (auto node : ListView{ast->expressionList}) {
    if (index >= 2) break;
    auto value = interp.evaluate(node);
    if (!value) return std::nullopt;
    auto converted = interp.toArithmeticType(*value, elementType);
    if (!converted) return std::nullopt;
    parts[index++] = *converted;
  }

  return ConstValue{std::make_shared<ConstComplex>(parts[0], parts[1])};
}

auto ASTInterpreter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  auto traits = unit()->typeTraits();

  if (auto arrayType = type_cast<BoundedArrayType>(ast->type))
    return arrayValue(ast, arrayType);

  if (auto complexType = unqualified_cast<ComplexType>(ast->type))
    return complexValue(ast, complexType);

  if (auto classType = type_cast<ClassType>(ast->type)) {
    if (traits.is_aggregate(ast->type))
      return aggregateObject(ast, classType->symbol());
    if (!traits.initializer_list_element_type(ast->type)) return std::nullopt;
  } else if (!traits.is_class(ast->type)) {
    if (!ast->expressionList) return interp.zeroInitialize(ast->type);
    if (!ast->expressionList->next)
      return interp.evaluate(ast->expressionList->value);
  }

  auto values = std::vector<std::tuple<ConstValue, const Type*>>();
  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.evaluate(node);
    if (!value) return std::nullopt;
    values.emplace_back(*value, node->type);
  }
  auto elements = std::make_shared<InitializerList>(std::move(values));
  auto elementType = traits.initializer_list_element_type(ast->type);
  if (!elementType) return elements;
  auto classType = unqualified_cast<ClassType>(ast->type);
  if (!classType) return std::nullopt;
  std::vector<FieldSymbol*> fields;
  for (auto field :
       views::members(classType->symbol()) | views::non_static_fields)
    fields.push_back(field);
  if (fields.size() != 2 || !traits.is_pointer(fields[0]->type()))
    return std::nullopt;
  auto backing =
      control()->newVariableSymbol(nullptr, ast->firstSourceLocation());
  backing->setName(control()->newAnonymousId("initializer_list"));
  backing->setType(control()->getBoundedArrayType(traits.add_const(elementType),
                                                  elements->elements.size()));
  backing->setConstexpr(true);
  backing->setConstValue(ConstValue{elements});
  auto object = std::make_shared<ConstObject>(ast->type);
  object->addMember(fields[0], std::make_shared<ConstAddress>(backing));
  auto size = static_cast<std::intmax_t>(elements->elements.size());
  if (traits.is_pointer(fields[1]->type()))
    object->addMember(fields[1], std::make_shared<ConstAddress>(backing, size));
  else if (traits.is_integral(fields[1]->type()))
    object->addMember(fields[1], size);
  else
    return std::nullopt;
  return object;
}

auto ASTInterpreter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  std::optional<ConstValue> result;
  std::size_t count = 0;

  for (auto node : ListView{ast->expressionList}) {
    auto value = interp.expression(node);
    if (count == 0) result = std::move(value);
    ++count;
  }

  if (count != 1) return ExpressionResult{std::nullopt};

  return ExpressionResult{std::move(result)};
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
