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
#include <cxx/vector_components.h>
#include <cxx/views/symbols.h>

#include <format>
#include <functional>
#include <type_traits>
#include <utility>

namespace cxx {
namespace {
[[nodiscard]] auto structuredBindingAlias(ExpressionAST* expression)
    -> VariableSymbol* {
  auto id = ast_cast<IdExpressionAST>(expression);
  if (!id) return nullptr;
  auto binding = symbol_cast<VariableSymbol>(id->symbol);
  if (!binding || !binding->isStructuredBindingAlias()) return nullptr;
  return binding;
}

[[nodiscard]] auto isNonNullInteger(const ConstValue& value) -> bool {
  auto integer = std::get_if<ConstInt>(&value);
  return integer && !integer->isZero();
}

[[nodiscard]] auto stringLiteralElement(const StringLiteral* literal,
                                        std::intmax_t index)
    -> std::optional<ConstValue> {
  if (index < 0) return std::nullopt;
  const auto position = static_cast<std::uintmax_t>(index);
  const auto count = literal->charCount();
  if (position > count) return std::nullopt;
  if (position == count) return ConstValue{std::intmax_t{0}};
  return ConstValue{static_cast<std::intmax_t>(literal->charAt(position))};
}

[[nodiscard]] auto pointerAddress(const ConstValue& value)
    -> std::shared_ptr<ConstAddress> {
  if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value))
    return *address;
  if (auto literal = std::get_if<const StringLiteral*>(&value);
      literal && *literal)
    return std::make_shared<ConstAddress>(*literal);
  return nullptr;
}

[[nodiscard]] auto declarationNamedBy(ExpressionAST* expression) -> Symbol* {
  while (auto nested = ast_cast<NestedExpressionAST>(expression))
    expression = nested->expression;
  if (auto id = ast_cast<IdExpressionAST>(expression)) return id->symbol;
  if (auto member = ast_cast<MemberExpressionAST>(expression))
    return member->symbol;
  return nullptr;
}

[[nodiscard]] auto declaredAlignment(Symbol* declaration,
                                     std::size_t typeAlignment) -> std::size_t {
  if (auto field = symbol_cast<FieldSymbol>(declaration))
    return static_cast<std::size_t>(field->effectiveAlignment());
  if (auto variable = symbol_cast<VariableSymbol>(declaration))
    return std::max(typeAlignment,
                    static_cast<std::size_t>(variable->explicitAlignment()));
  return typeAlignment;
}

}  // namespace

auto ASTInterpreter::sameObjectOperand(ExpressionAST* ast) -> ExpressionAST* {
  if (auto nested = ast_cast<NestedExpressionAST>(ast))
    return nested->expression;
  if (!is_glvalue(ast)) return nullptr;
  if (auto cast = ast_cast<ImplicitCastExpressionAST>(ast)) {
    if (cast->castKind == ImplicitCastKind::kUserDefinedConversion)
      return nullptr;
    return cast->expression;
  }
  if (auto cast = ast_cast<CastExpressionAST>(ast)) return cast->expression;
  auto cast = ast_cast<CppCastExpressionAST>(ast);
  if (!cast || cast->castOp == TokenKind::T_DYNAMIC_CAST) return nullptr;
  return cast->expression;
}

auto ASTInterpreter::zeroInitialize(const Type* type)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;
  type = traits.remove_atomic(type);
  if (traits.is_integral_or_enum(type)) return std::intmax_t{0};
  if (traits.is_floating_point(type)) return double{0.0};
  if (traits.is_pointer(type) || traits.is_null_pointer(type))
    return std::intmax_t{0};
  if (traits.is_member_pointer(type))
    return ConstValue{
        std::make_shared<ConstAddress>(static_cast<Symbol*>(nullptr))};
  if (auto complexType = unqualified_cast<ComplexType>(type)) {
    auto zero = zeroInitialize(complexType->elementType());
    if (!zero) return std::nullopt;
    return ConstValue{std::make_shared<ConstComplex>(*zero, *zero)};
  }
  if (auto vectorType = unqualified_cast<VectorType>(type)) {
    auto lane = zeroInitialize(vectorType->elementType());
    if (!lane) return std::nullopt;
    auto list = std::make_shared<InitializerList>();
    for (std::size_t index = 0; index < vectorType->elementCount(); ++index)
      list->append(*lane, vectorType->elementType());
    return ConstValue{list};
  }
  if (auto arrayType = type_cast<BoundedArrayType>(type)) {
    auto elementZero = zeroInitialize(arrayType->elementType());
    if (!elementZero) return std::nullopt;
    return ConstValue{std::make_shared<InitializerList>(
        std::vector<InitializerList::Element>{}, arrayType->size(),
        InitializerList::Element{std::move(*elementZero),
                                 arrayType->elementType()})};
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
  std::shared_ptr<ConstAddress> destination;

  [[nodiscard]] auto unit() -> TranslationUnit* { return interp.unit_; }

  [[nodiscard]] auto control() -> Control* { return interp.control(); }

  [[nodiscard]] auto memoryLayout() -> MemoryLayout* {
    return control()->memoryLayout();
  }

  [[nodiscard]] auto evaluate(ExpressionAST* ast) -> ExpressionResult {
    return interp.expression(ast);
  }

  [[nodiscard]] auto constantVariable(Symbol* symbol) -> VariableSymbol*;

  [[nodiscard]] auto toBool(const ConstValue& value) -> std::optional<bool> {
    return interp.toBool(value);
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

  [[nodiscard]] auto toConstInt(const Type* type, const ConstValue& value)
      -> std::optional<ConstInt> {
    auto converted = interp.toIntegralType(value, type);
    if (!converted) return std::nullopt;

    auto result = std::get_if<ConstInt>(&*converted);
    if (!result) return std::nullopt;

    return *result;
  }

  [[nodiscard]] auto integerOperands(const Type* type,
                                     const ExpressionResult& left,
                                     const ExpressionResult& right)
      -> std::optional<std::pair<ConstInt, ConstInt>> {
    auto l = toConstInt(type, *left);
    if (!l) return std::nullopt;

    auto r = toConstInt(type, *right);
    if (!r) return std::nullopt;

    return std::pair{*l, *r};
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
      case TypeKind::kDouble:
        return ConstValue{op(toDouble(*left), toDouble(*right))};
      case TypeKind::kHalf:
      case TypeKind::kFloat16:
        return interp.toArithmeticType(
            ConstValue{op(toFloat(*left), toFloat(*right))}, type);
      default:
        return std::nullopt;
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
      auto value = toConstInt(type, *operand);
      if (!value) return std::nullopt;

      auto negated = value->checkedNegate();
      if (!negated) return std::nullopt;
      return ConstValue{*negated};
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
    if (unit()->typeTraits().is_floating_point(type)) {
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a * b; });
    }

    if (!unit()->typeTraits().is_integral_or_unscoped_enum(type)) {
      return toValue(toUInt64(*left) * toUInt64(*right));
    }

    auto operands = integerOperands(type, left, right);
    if (!operands) return std::nullopt;

    auto product = checkedMul(operands->first, operands->second);
    if (!product) return std::nullopt;
    return ConstValue{*product};
  }

  auto slash_op(const Type* type, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    if (unit()->typeTraits().is_floating_point(type)) {
      if (toDouble(*right) == 0.0) return std::nullopt;
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a / b; });
    }

    auto operands = integerOperands(type, left, right);
    if (!operands) return std::nullopt;

    auto result = operands->first / operands->second;
    if (!result) return std::nullopt;

    return ConstValue{*result};
  }

  auto percent_op(const Type* type, const ExpressionResult& left,
                  const ExpressionResult& right) -> ExpressionResult {
    auto operands = integerOperands(type, left, right);
    if (!operands) return std::nullopt;

    auto result = operands->first % operands->second;
    if (!result) return std::nullopt;

    return ConstValue{*result};
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
        auto realEqual =
            scalarComparison(TokenKind::T_EQUAL_EQUAL, elementType,
                             ExpressionResult{a}, ExpressionResult{c});
        auto imagEqual =
            scalarComparison(TokenKind::T_EQUAL_EQUAL, elementType,
                             ExpressionResult{b}, ExpressionResult{d});
        if (!realEqual.has_value() || !imagEqual.has_value())
          return std::nullopt;
        auto realIsEqual = toBool(*realEqual);
        auto imagIsEqual = toBool(*imagEqual);
        if (!realIsEqual || !imagIsEqual) return std::nullopt;
        const auto equal = *realIsEqual && *imagIsEqual;
        const auto result = op == TokenKind::T_EQUAL_EQUAL ? equal : !equal;
        return ConstValue{std::intmax_t{result ? 1 : 0}};
      }

      default:
        return std::nullopt;
    }
  }

  auto plus_op(const Type* type, const ExpressionResult& left,
               const ExpressionResult& right) -> ExpressionResult {
    if (unit()->typeTraits().is_floating_point(type)) {
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a + b; });
    }

    if (!unit()->typeTraits().is_integral_or_unscoped_enum(type)) {
      return toValue(toUInt64(*left) + toUInt64(*right));
    }

    auto operands = integerOperands(type, left, right);
    if (!operands) return std::nullopt;

    auto sum = checkedAdd(operands->first, operands->second);
    if (!sum) return std::nullopt;
    return ConstValue{*sum};
  }

  auto minus_op(const Type* type, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    if (unit()->typeTraits().is_floating_point(type)) {
      return floatingArithmetic(type, left, right,
                                [](auto a, auto b) { return a - b; });
    }

    if (!unit()->typeTraits().is_integral_or_unscoped_enum(type)) {
      return toValue(toUInt64(*left) - toUInt64(*right));
    }

    auto operands = integerOperands(type, left, right);
    if (!operands) return std::nullopt;

    auto difference = checkedSub(operands->first, operands->second);
    if (!difference) return std::nullopt;
    return ConstValue{*difference};
  }

  [[nodiscard]] auto shiftOperands(const Type* type,
                                   const ExpressionResult& left,
                                   const ExpressionResult& right)
      -> std::optional<std::pair<ConstInt, ConstInt>> {
    auto shifted = toConstInt(type, *left);
    if (!shifted) return std::nullopt;

    auto count = std::get_if<ConstInt>(&*right);
    if (!count) return std::nullopt;

    if (count->isNegative()) return std::nullopt;

    if (count->toUIntMax() >= static_cast<std::uintmax_t>(shifted->width())) {
      return std::nullopt;
    }

    return std::pair{*shifted, *count};
  }

  auto less_less_op(const Type* type, const ExpressionResult& left,
                    const ExpressionResult& right) -> ExpressionResult {
    auto operands = shiftOperands(type, left, right);
    if (!operands) return std::nullopt;

    return ConstValue{operands->first << operands->second};
  }

  auto greater_greater_op(const Type* type, const ExpressionResult& left,
                          const ExpressionResult& right) -> ExpressionResult {
    auto operands = shiftOperands(type, left, right);
    if (!operands) return std::nullopt;

    return ConstValue{operands->first >> operands->second};
  }

  [[nodiscard]] auto comparisonValue(TokenKind op, const auto& left,
                                     const auto& right) -> ExpressionResult {
    switch (op) {
      case TokenKind::T_EQUAL_EQUAL:
        return ConstValue{left == right};
      case TokenKind::T_EXCLAIM_EQUAL:
        return ConstValue{left != right};
      case TokenKind::T_LESS:
        return ConstValue{left < right};
      case TokenKind::T_GREATER:
        return ConstValue{left > right};
      case TokenKind::T_LESS_EQUAL:
        return ConstValue{left <= right};
      case TokenKind::T_GREATER_EQUAL:
        return ConstValue{left >= right};
      case TokenKind::T_LESS_EQUAL_GREATER: {
        auto order = left <=> right;
        if (order < 0) return ConstValue{std::intmax_t{-1}};
        if (order > 0) return ConstValue{std::intmax_t{1}};
        if (order == 0) return ConstValue{std::intmax_t{0}};
        return std::nullopt;
      }
      default:
        return std::nullopt;
    }
  }

  [[nodiscard]] auto scalarComparison(TokenKind op, const Type* type,
                                      const ExpressionResult& left,
                                      const ExpressionResult& right)
      -> ExpressionResult {
    if (unit()->typeTraits().is_floating_point(type)) {
      auto a = interp.toLongDouble(*left);
      auto b = interp.toLongDouble(*right);
      if (!a || !b) return std::nullopt;
      return comparisonValue(op, *a, *b);
    }

    if (auto operands = integerOperands(type, left, right))
      return comparisonValue(op, operands->first, operands->second);

    auto a = interp.toUInt(*left);
    auto b = interp.toUInt(*right);
    if (!a || !b) return std::nullopt;
    return comparisonValue(op, *a, *b);
  }

  [[nodiscard]] auto bitwiseOperation(const Type* type,
                                      const ExpressionResult& left,
                                      const ExpressionResult& right,
                                      auto operation) -> ExpressionResult {
    if (auto operands = integerOperands(type, left, right))
      return ConstValue{operation(operands->first, operands->second)};
    return operation(toInt(*left), toInt(*right));
  }

  auto amp_op(const Type* type, const ExpressionResult& left,
              const ExpressionResult& right) -> ExpressionResult {
    return bitwiseOperation(type, left, right, std::bit_and<>{});
  }

  auto caret_op(const Type* type, const ExpressionResult& left,
                const ExpressionResult& right) -> ExpressionResult {
    return bitwiseOperation(type, left, right, std::bit_xor<>{});
  }

  auto bar_op(const Type* type, const ExpressionResult& left,
              const ExpressionResult& right) -> ExpressionResult {
    return bitwiseOperation(type, left, right, std::bit_or<>{});
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

  [[nodiscard]] auto operator()(BracedTypeConstructionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(SpliceMemberExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(MemberExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(PostIncrExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(CppCastExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(BuiltinBitCastExpressionAST* ast)
      -> ExpressionResult;

  [[nodiscard]] auto operator()(BuiltinConvertVectorExpressionAST* ast)
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
      FunctionSymbol* function, ExpressionAST* operand,
      ExpressionAST* extraArgument = nullptr, bool rightToLeft = false)
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

  [[nodiscard]] auto evaluateMemberPointerConversion(
      ImplicitCastExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto operator()(BinaryExpressionAST* ast) -> ExpressionResult;

  [[nodiscard]] auto vectorLanes(const ConstValue& value,
                                 const VectorType* vector)
      -> std::optional<std::vector<ConstValue>>;

  [[nodiscard]] auto convertLane(const ConstValue& value, const Type* type)
      -> ExpressionResult;

  [[nodiscard]] auto makeVector(std::vector<ConstValue> lanes,
                                const VectorType* vector) -> ExpressionResult;

  [[nodiscard]] auto evaluateVectorCast(ImplicitCastExpressionAST* ast,
                                        const ConstValue& value)
      -> ExpressionResult;

  [[nodiscard]] auto applyVectorBinaryOp(TokenKind op, const Type* resultType,
                                         const Type* operandType,
                                         const ConstValue& left,
                                         const ConstValue& right)
      -> ExpressionResult;

  [[nodiscard]] auto applyVectorUnaryOp(TokenKind op, const Type* operandType,
                                        const Type* resultType,
                                        const ConstValue& value)
      -> ExpressionResult;

  [[nodiscard]] auto laneMask(std::optional<bool> truth) -> ExpressionResult;

  [[nodiscard]] auto logicalLane(TokenKind op, const ConstValue& left,
                                 const ConstValue& right) -> ExpressionResult;

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

  [[nodiscard]] auto vectorValue(BracedInitListAST* ast,
                                 const VectorType* vector) -> ExpressionResult;

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

auto ASTInterpreter::expression(ExpressionAST* ast,
                                std::shared_ptr<ConstAddress> destination)
    -> ExpressionResult {
  if (!ast) return ExpressionResult{std::nullopt};
  if (aborted_) return ExpressionResult{std::nullopt};
  if (isDependent(unit_, ast->type)) return ExpressionResult{std::nullopt};
  return visit(ExpressionVisitor{*this, std::move(destination)}, ast);
}

auto ASTInterpreter::evaluateStaticField(FieldSymbol* field)
    -> ExpressionResult {
  if (std::ranges::contains(fieldsUnderEvaluation_, field))
    return ExpressionResult{std::nullopt};

  if (traits.is_reference(field->type())) {
    auto address = staticFieldAddress(field);
    if (!address) return std::nullopt;
    auto referent = std::get_if<std::shared_ptr<ConstAddress>>(&*address);
    if (!referent || !*referent) return std::nullopt;
    return loadAddress(**referent, 0);
  }

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

auto ASTInterpreter::staticFieldSlot(FieldSymbol* field) -> ConstValue* {
  if (auto slot = lookupLocalSlot(field)) return slot;
  auto value = evaluateStaticField(field);
  if (!value) return nullptr;
  setLocal(field, std::move(*value));
  return lookupLocalSlot(field);
}

auto ASTInterpreter::staticFieldAddress(FieldSymbol* field)
    -> std::optional<ConstValue> {
  if (field->isThreadLocal()) return std::nullopt;
  if (!traits.is_reference(field->type()))
    return std::make_shared<ConstAddress>(field);
  if (auto definition = symbol_cast<VariableSymbol>(field->definition());
      definition && definition->constValue()) {
    return definition->constValue();
  }
  if (field->constValue()) return field->constValue();
  return extendedReferenceBinding(field->type(), field->initializer());
}

auto ASTInterpreter::lvalue(ExpressionAST* ast) -> ConstValue* {
  if (!ast) return nullptr;

  while (ast) {
    if (auto initializer = ast_cast<DefaultInitializerExpressionAST>(ast)) {
      InitializerContextGuard guard{*this, initializer->context};
      return lvalue(initializer->expression);
    }
    if (auto operand = sameObjectOperand(ast)) {
      ast = operand;
      continue;
    }
    if (auto alias = structuredBindingAlias(ast)) {
      ast = alias->initializer();
      continue;
    }
    if (auto constant = ast_cast<ConstExpressionAST>(ast)) {
      if (!frames_.empty() && constant->expression) {
        ast = constant->expression;
        continue;
      }
      if (!is_glvalue(constant)) break;
      if (!constant->constValue) return nullptr;
      auto address =
          std::get_if<std::shared_ptr<ConstAddress>>(constant->constValue);
      if (address && *address) return addressSlot(**address, 0, ast->type);
      ast = constant->expression;
      continue;
    }
    break;
  }

  if (auto cast = ast_cast<ImplicitCastExpressionAST>(ast)) {
    if (cast->castKind == ImplicitCastKind::kUserDefinedConversion)
      return evaluateConversionFunctionCall(cast, CallResultKind::kLValue)
          .lvalue;
  }

  if (auto condition = ast_cast<ConditionExpressionAST>(ast)) {
    auto variable = initializeDecisionVariable(condition);
    if (!variable) return nullptr;
    return lookupLocalSlot(variable);
  }

  if (auto id = ast_cast<IdExpressionAST>(ast)) {
    auto sym = id->symbol;
    if (!sym) return nullptr;

    if (auto field = symbol_cast<FieldSymbol>(sym)) {
      if (!field->isStatic()) {
        if (auto object = fieldOwner(ast)) return memberSlot(object, field);
        return nullptr;
      }
      if (field->isStatic()) return staticFieldSlot(field);
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
    if (auto lane = componentLane(member)) {
      auto lanes = componentVector(member);
      if (!lanes || *lane >= lanes->size()) return nullptr;
      return &std::get<0>(lanes->mutableElement(*lane));
    }
    if (!member->symbol) return nullptr;
    auto field = symbol_cast<FieldSymbol>(member->symbol);
    if (!field || field->isStatic()) return nullptr;

    if (auto object = memberObject(member)) return memberSlot(object, field);
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
      ThisScope thisScope{*this, *objPtr};
      return evaluateCallLValue(op, {*idxVal});
    }

    auto baseVal = expression(sub->baseExpression);
    if (!baseVal.has_value()) return nullptr;

    if (auto list = std::get_if<std::shared_ptr<InitializerList>>(&*baseVal)) {
      if (!*list) return nullptr;
      auto idxVal = expression(sub->indexExpression);
      if (!idxVal.has_value()) return nullptr;
      auto idx = toUInt(*idxVal);
      if (!idx.has_value() || *idx >= (*list)->size()) return nullptr;
      return &std::get<0>((*list)->mutableElement(*idx));
    }

    if (auto addr = std::get_if<std::shared_ptr<ConstAddress>>(&*baseVal)) {
      if (!*addr) return nullptr;
      auto idxVal = expression(sub->indexExpression);
      if (!idxVal.has_value()) return nullptr;
      auto idx = toOffset(*idxVal);
      if (!idx.has_value()) return nullptr;
      return addressSlot(**addr, *idx, ast->type);
    }

    return nullptr;
  }

  if (auto unary = ast_cast<UnaryExpressionAST>(ast)) {
    if (unary->symbol)
      return evaluateUnaryOperatorCall(unary, CallResultKind::kLValue).lvalue;
    if (unary->op == TokenKind::T___REAL__ ||
        unary->op == TokenKind::T___IMAG__) {
      auto slot = lvalue(unary->expression);
      if (!slot) return nullptr;
      if (!traits.is_complex(unary->expression->type))
        return unary->op == TokenKind::T___REAL__ ? slot : nullptr;
      auto value = std::get_if<std::shared_ptr<ConstComplex>>(slot);
      if (!value || !*value) return nullptr;
      if (unary->op == TokenKind::T___REAL__) return &(*value)->real();
      return &(*value)->imag();
    }
    if (unary->op != TokenKind::T_STAR) return nullptr;
    auto ptrVal = expression(unary->expression);
    if (!ptrVal.has_value()) return nullptr;
    auto addr = std::get_if<std::shared_ptr<ConstAddress>>(&*ptrVal);
    if (!addr || !*addr) return nullptr;
    return addressSlot(**addr, 0, ast->type);
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

template <typename Value>
auto ASTInterpreter::addressedElement(Value* storage,
                                      const ConstAddress& address) -> Value* {
  if (!storage || !validPosition(address)) return nullptr;
  if (static_cast<std::uintmax_t>(address.offset()) == *arrayExtent(address))
    return nullptr;
  for (const auto& subobject : address.subobjectPath()) {
    if (auto symbol = std::get_if<Symbol*>(&subobject)) {
      auto object = std::get_if<std::shared_ptr<ConstObject>>(storage);
      if (!object || !*object || (*object)->lifetimeEnded()) return nullptr;
      if constexpr (std::is_const_v<Value>)
        storage = (*object)->subobject(*symbol);
      else
        storage = (*object)->mutableSubobject(*symbol);
      if (!storage) return nullptr;
      continue;
    }
    auto index = std::get<std::size_t>(subobject);
    auto array = std::get_if<std::shared_ptr<InitializerList>>(storage);
    if (!array || !*array || index >= (*array)->size()) return nullptr;
    if constexpr (std::is_const_v<Value>)
      storage = &std::get<0>((*array)->distinctElement(index));
    else
      storage = &std::get<0>((*array)->mutableElement(index));
  }
  if (!address.arrayType()) return storage;
  auto array = std::get_if<std::shared_ptr<InitializerList>>(storage);
  if (!array || !*array) return nullptr;
  auto index = static_cast<std::uintmax_t>(address.offset());
  if (index >= (*array)->size()) return nullptr;
  if constexpr (std::is_const_v<Value>)
    return &std::get<0>((*array)->distinctElement(index));
  else
    return &std::get<0>((*array)->mutableElement(index));
}

auto ASTInterpreter::loadAddress(const ConstAddress& address,
                                 std::intmax_t extraIndex,
                                 const Type* objectType)
    -> std::optional<ConstValue> {
  if (symbol_cast<FunctionSymbol>(address.symbol()) && extraIndex == 0)
    return std::make_shared<ConstAddress>(address);
  auto located = elementAddress(address, extraIndex, objectType);
  if (!located) return std::nullopt;
  if (!validPosition(*located)) return std::nullopt;
  if (static_cast<std::uintmax_t>(located->offset()) == *arrayExtent(*located))
    return std::nullopt;
  if (auto str = located->stringLiteral()) {
    if (!located->arrayType()) return ConstValue{str};
    return stringLiteralElement(str, located->offset());
  }
  if (located->denotesWholeOwner()) return ConstValue{located->owner()};
  auto sym = located->root() ? located->root() : located->symbol();
  if (symbol_cast<FunctionSymbol>(sym))
    return std::make_shared<ConstAddress>(sym);
  if (auto storage = addressStorage(*located)) {
    auto slot = addressedElement(&std::as_const(*storage), *located);
    return slot ? std::optional<ConstValue>{*slot} : std::nullopt;
  }
  if (located->storage() || located->owner()) return std::nullopt;

  std::optional<ConstValue> storage;
  if (auto var = symbol_cast<VariableSymbol>(sym)) {
    var = var->resolvedDefinition();
    if (!isPotentiallyUsableInConstantExpressions(var)) return std::nullopt;
    if (auto cv = var->constValue())
      storage = cv;
    else if (var->initializer())
      storage = expression(var->initializer());
  } else if (auto field = symbol_cast<FieldSymbol>(sym);
             field && field->isStatic()) {
    storage = evaluateStaticField(field);
  }
  if (!storage) return std::nullopt;
  auto slot = addressedElement(&std::as_const(*storage), *located);
  return slot ? std::optional<ConstValue>{*slot} : std::nullopt;
}

auto ASTInterpreter::pointeeCharacter(const ConstValue& pointer,
                                      std::intmax_t index)
    -> std::optional<std::intmax_t> {
  std::optional<ConstValue> element;
  if (auto str = std::get_if<const StringLiteral*>(&pointer); str && *str)
    element = stringLiteralElement(*str, index);
  else if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&pointer);
           address && *address) {
    auto range = memoryRange(**address, nullptr);
    if (!range || index < 0) return std::nullopt;
    if (static_cast<std::uintmax_t>(index) >= range->count) return std::nullopt;
    if (!traits.is_narrow_char_type(range->elementType) &&
        !traits.is_same(range->elementType, control()->getChar8Type()))
      return std::nullopt;
    element = loadAddress(**address, index, range->elementType);
  }
  if (!element) return std::nullopt;
  return toInt(*element);
}

auto ASTInterpreter::nullTerminatedString(const ConstValue& pointer,
                                          std::size_t limit)
    -> std::optional<std::string> {
  std::string result;
  while (result.size() < limit) {
    auto unit =
        pointeeCharacter(pointer, static_cast<std::intmax_t>(result.size()));
    if (!unit) return std::nullopt;
    if (*unit == 0) break;
    result.push_back(static_cast<char>(*unit));
  }
  return result;
}

auto ASTInterpreter::addressStorage(const ConstAddress& address)
    -> ConstValue* {
  if (address.owner()) return subobjectSlot(address.owner(), address.symbol());
  if (address.storage()) return &address.storage()->value();
  for (auto it = initializations_.rbegin(); it != initializations_.rend(); ++it)
    if (it->first == address.root()) return it->second;
  auto field = symbol_cast<FieldSymbol>(address.symbol());
  if (field && field->isStatic()) return staticFieldSlot(field);
  return nullptr;
}

auto ASTInterpreter::addressSlot(const ConstAddress& address,
                                 std::intmax_t extraIndex,
                                 const Type* objectType) -> ConstValue* {
  auto located = elementAddress(address, extraIndex, objectType);
  if (!located || located->stringLiteral()) return nullptr;
  return addressedElement(addressStorage(*located), *located);
}

auto ASTInterpreter::dispatchVirtualCall(
    FunctionSymbol* function, const std::shared_ptr<ConstObject>& object)
    -> FunctionSymbol* {
  if (!object) return nullptr;
  if (object->isConstexprUnknown()) return nullptr;

  auto dynamicObject = object;
  for (auto phase = constructionPhases_.rbegin();
       phase != constructionPhases_.rend(); ++phase) {
    if (phase->completeObject != object) continue;
    dynamicObject = phase->dynamicObject;
    break;
  }
  auto classType = unqualified_cast<ClassType>(dynamicObject->type());
  auto classSymbol = classType ? classType->symbol() : nullptr;
  if (!classSymbol) return nullptr;

  auto vtableLayout = classSymbol->resolvedDefinition()->vtableLayout();
  if (!vtableLayout) return nullptr;

  return vtableLayout->finalOverrider(function);
}

auto ASTInterpreter::constexprUnknownObject(const Type* objectType)
    -> std::shared_ptr<ConstObject> {
  auto type = traits.remove_cvref(objectType);
  if (!traits.is_object(type)) return {};
  return ConstObject::makeConstexprUnknown(type);
}

auto ASTInterpreter::constexprUnknownObject(const ConstAddress& address,
                                            const Type* objectType)
    -> std::shared_ptr<ConstObject> {
  if (address.lifetimeEnded()) return {};
  if (address.storage()) return {};
  if (address.denotesWholeOwner()) return address.owner();

  auto symbol = address.symbol();
  if (!symbol) return {};

  if (auto owner = address.owner()) {
    if (auto value = owner->subobject(symbol))
      return asConstexprUnknownObject(*value);
  } else if (auto slot = lookupLocalSlot(symbol)) {
    return asConstexprUnknownObject(*slot);
  }

  return constexprUnknownObject(objectType);
}

auto ASTInterpreter::constexprUnknownObject(ExpressionAST* ast)
    -> std::shared_ptr<ConstObject> {
  if (!ast || !is_glvalue(ast)) return {};

  if (auto designator = addressOfLvalue(ast)) {
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*designator);
    if (!address || !*address) return {};
    return constexprUnknownObject(**address, ast->type);
  }

  auto idExpr = ast_cast<IdExpressionAST>(ast);
  if (!idExpr) return {};

  if (auto variable = symbol_cast<VariableSymbol>(idExpr->symbol);
      variable && variable->isThreadLocal())
    return {};
  if (auto field = symbol_cast<FieldSymbol>(idExpr->symbol);
      field && field->isThreadLocal())
    return {};

  if (!introduces_variable(idExpr->symbol)) return {};

  if (auto slot = lookupLocalSlot(idExpr->symbol))
    return asConstexprUnknownObject(*slot);

  return constexprUnknownObject(ast->type);
}

auto ASTInterpreter::constexprUnknownAddress(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto object = constexprUnknownObject(ast);
  if (!object) return std::nullopt;
  return std::make_shared<ConstAddress>(object, nullptr);
}

auto ASTInterpreter::designatedCallResult(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto result = evaluateCallExpression(ast, CallResultKind::kObject);
  if (!result.value) return std::nullopt;
  if (!result.isAddress) return result.value;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*result.value);
  if (!address || !*address) return std::nullopt;
  if (auto loaded = loadAddress(**address, 0)) return loaded;
  auto object = constexprUnknownObject(**address, ast->type);
  if (!object) return std::nullopt;
  return ConstValue{std::move(object)};
}

auto ASTInterpreter::designatedValue(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto designator = ast;
  while (auto operand = sameObjectOperand(designator)) designator = operand;
  if (auto call = ast_cast<CallExpressionAST>(designator);
      call && is_glvalue(call))
    return designatedCallResult(call);
  if (auto value = expression(ast)) return value;
  auto object = constexprUnknownObject(ast);
  if (!object) return std::nullopt;
  return ConstValue{std::move(object)};
}

auto ASTInterpreter::discardedValue(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!designatedValue(ast)) return std::nullopt;
  return ConstValue{ConstInt{std::intmax_t{0}}};
}

auto ASTInterpreter::designatedObject(ExpressionAST* ast)
    -> std::shared_ptr<ConstObject> {
  auto value = designatedValue(ast);
  if (!value) return {};
  auto object = std::get_if<std::shared_ptr<ConstObject>>(&*value);
  return object ? *object : nullptr;
}

auto ASTInterpreter::componentLane(MemberExpressionAST* ast)
    -> std::optional<std::size_t> {
  auto access = vectorComponentAccess(ast);
  if (!access || access->components.lanes.size() != 1) return std::nullopt;
  return access->components.lanes.front();
}

auto ASTInterpreter::componentVector(MemberExpressionAST* ast)
    -> std::shared_ptr<InitializerList> {
  auto base = expression(ast->baseExpression);
  if (!base) return nullptr;
  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*base);
    if (!address || !*address) return nullptr;
    base = loadAddress(**address, 0,
                       traits.remove_pointer(ast->baseExpression->type));
    if (!base) return nullptr;
  }
  auto lanes = std::get_if<std::shared_ptr<InitializerList>>(&*base);
  return lanes ? *lanes : nullptr;
}

auto ASTInterpreter::memberObject(MemberExpressionAST* ast)
    -> std::shared_ptr<ConstObject> {
  auto base = memberBase(ast);
  return addressedClassSubobject(base.object, base.address).object;
}

auto ASTInterpreter::addressedClassSubobject(
    const std::shared_ptr<ConstObject>& object,
    const std::shared_ptr<ConstAddress>& address, const Type* enclosingType)
    -> ClassSubobject {
  if (!object || !address || !address->elementType()) return {object, 0};
  auto completeType = unqualified_cast<ClassType>(object->type());
  auto completeClass = completeType ? completeType->definition() : nullptr;
  auto completeLayout = completeClass ? completeClass->layout() : nullptr;
  struct Subobject {
    std::shared_ptr<ConstObject> object;
    std::intmax_t offset;
    ClassSubobject enclosing;
    std::vector<Symbol*> path;
  };
  auto work = std::vector<Subobject>{{object, 0, {}}};

  while (!work.empty()) {
    auto current = std::move(work.back());
    work.pop_back();
    if (enclosingType &&
        traits.is_similar(current.object->type(), enclosingType))
      current.enclosing = {current.object, current.offset, current.path};
    if (current.offset == address->classDisplacement() &&
        traits.is_similar(current.object->type(), address->elementType())) {
      if (enclosingType) return current.enclosing;
      return {current.object, current.offset, current.path};
    }

    auto classType = unqualified_cast<ClassType>(current.object->type());
    auto classSymbol = classType ? classType->definition() : nullptr;
    auto layout = classSymbol ? classSymbol->layout() : nullptr;
    if (!layout) continue;
    for (auto base : classSymbol->baseClasses()) {
      auto value = current.object->subobject(base);
      auto nested =
          value ? std::get_if<std::shared_ptr<ConstObject>>(value) : nullptr;
      if (!nested || !*nested) continue;
      auto baseClass = resolved_base_class(base);
      auto ownerLayout = base->isVirtual() ? completeLayout : layout;
      if (!ownerLayout) continue;
      auto info = ownerLayout->getBaseInfo(baseClass, base->isVirtual());
      if (!info || !std::in_range<std::intmax_t>(info->offset)) continue;
      auto offset =
          checkedAdd(ConstInt{base->isVirtual() ? 0 : current.offset},
                     ConstInt{static_cast<std::intmax_t>(info->offset)});
      if (offset) {
        auto path = current.path;
        path.push_back(base);
        work.push_back(
            {*nested, offset->toIntMax(), current.enclosing, std::move(path)});
      }
    }
  }
  return {};
}

auto ASTInterpreter::pointeeObject(ConstValue value, const Type* pointeeType)
    -> std::shared_ptr<ConstObject> {
  if (auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value)) {
    if (!*address) return {};
    auto pointee = loadAddress(**address, 0);
    if (!pointee) return constexprUnknownObject(**address, pointeeType);
    value = std::move(*pointee);
  }

  auto object = std::get_if<std::shared_ptr<ConstObject>>(&value);
  return object ? *object : nullptr;
}

auto ASTInterpreter::designatedThis() -> DesignatedObject {
  if (!thisObject_) return {};
  auto classType = unqualified_cast<ClassType>(thisObject_->type());
  auto closure = classType ? classType->definition() : nullptr;
  if (!closure || !closure->isClosureType()) return {thisObject_, thisAddress_};
  auto capturedThis = closure->capturedThisField();
  if (!capturedThis) return {};
  auto value = thisObject_->subobject(capturedThis);
  if (!value) return {};
  return {pointeeObject(*value, traits.remove_pointer(capturedThis->type())),
          pointerAddress(*value)};
}

auto ASTInterpreter::implicitObjectFor(Symbol* member) -> DesignatedObject {
  if (!thisObject_) return {};
  if (traits.is_member_of_object_type(thisObject_->type(), member))
    return {thisObject_, thisAddress_};
  return designatedThis();
}

auto ASTInterpreter::fieldOwner(ExpressionAST* ast)
    -> std::shared_ptr<ConstObject> {
  if (auto id = ast_cast<IdExpressionAST>(ast)) {
    auto field = symbol_cast<FieldSymbol>(id->symbol);
    if (!field || field->isStatic()) return nullptr;
    auto object = implicitObjectFor(field);
    return addressedClassSubobject(object.object, object.address).object;
  }
  if (auto member = ast_cast<MemberExpressionAST>(ast)) {
    auto field = symbol_cast<FieldSymbol>(member->symbol);
    if (!field || field->isStatic()) return nullptr;
    return memberObject(member);
  }
  return nullptr;
}

auto ASTInterpreter::initialValue(const Type* type, ExpressionAST* initializer,
                                  std::shared_ptr<ConstAddress> destination)
    -> std::optional<ConstValue> {
  if (traits.is_reference(type)) return referenceBinding(initializer);
  return evaluateInitializer(type, initializer, std::move(destination));
}

auto ASTInterpreter::initializationValue(Symbol* variable,
                                         FunctionSymbol* constructor,
                                         ExpressionAST* initializer)
    -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  if (traits.is_reference(variable->type())) {
    auto value =
        extendedReferenceBinding(variable->type(), initializer, variable);
    if (!value || !isConstantInitialization(traits, variable, *value))
      return std::nullopt;
    return value;
  }
  ConstValue storage = IndeterminateValue{};
  initializations_.emplace_back(variable->canonical(), &storage);
  auto destination = typedAddress(ConstAddress{variable});
  auto value = initializationValue(variable->type(), constructor, initializer,
                                   std::move(destination));
  initializations_.pop_back();
  if (!value || !isConstantInitialization(traits, variable, *value))
    return std::nullopt;
  return value;
}

auto ASTInterpreter::constructionObject(
    const Type* type, const std::shared_ptr<ConstAddress>& destination,
    std::shared_ptr<ConstObject> object) -> std::shared_ptr<ConstObject> {
  ConstValue* slot = nullptr;
  if (destination) {
    slot = addressSlot(*destination, 0, type);
    if (!slot) return nullptr;
    if (!object) {
      auto existing = std::get_if<std::shared_ptr<ConstObject>>(slot);
      if (existing && *existing && traits.is_similar((*existing)->type(), type))
        object = *existing;
    }
  }
  if (!object) object = std::make_shared<ConstObject>(type);
  if (slot) *slot = object;
  return object;
}

auto ASTInterpreter::arrayElementDestination(
    const std::shared_ptr<ConstAddress>& destination, const Type* type,
    std::size_t index) -> std::shared_ptr<ConstAddress> {
  if (!destination || !std::in_range<std::intmax_t>(index)) return nullptr;
  auto array = arrayAddress(*destination, type);
  if (!array) return nullptr;
  return elementAddress(*array, static_cast<std::intmax_t>(index),
                        traits.get_element_type(type));
}

auto ASTInterpreter::subobjectDestination(
    const std::shared_ptr<ConstAddress>& destination, Symbol* subobject)
    -> std::shared_ptr<ConstAddress> {
  if (!destination) return nullptr;
  auto result = typedAddress(*destination);
  std::optional<std::intmax_t> displacement;
  if (auto field = symbol_cast<FieldSymbol>(subobject))
    displacement = fieldDisplacement(result->elementType(), field);
  else if (auto base = symbol_cast<BaseClassSymbol>(subobject)) {
    auto baseClass = resolved_base_class(base);
    if (!baseClass) return nullptr;
    displacement =
        classSubobjectOffset(result->elementType(), baseClass->type());
  }
  if (!displacement) return nullptr;

  auto path = result->subobjectPath();
  if (result->arrayType()) path.emplace_back(std::size_t(result->offset()));
  auto parent = loadAddress(*destination, 0);
  auto parentObject =
      parent ? std::get_if<std::shared_ptr<ConstObject>>(&*parent) : nullptr;
  if (!parentObject || !*parentObject) return nullptr;
  auto selected = addressedClassSubobject(*parentObject, destination);
  if (!selected.object) return nullptr;
  auto memberPath = selected.object->subobjectPath(subobject);
  if (!memberPath) return nullptr;
  for (auto member : selected.path) path.emplace_back(member);
  for (auto member : *memberPath) path.emplace_back(member);
  result = subobjectAddress(*result, *displacement);
  if (!result) return nullptr;
  result->setSubobjectPath(std::move(path));
  result->setOwner(nullptr);
  result->setArrayType(nullptr);
  result->setOffset(0);
  result->setClassDisplacement(0);
  result->setSymbol(subobject);
  result->setElementType(traits.aggregate_element_type(subobject));
  return result;
}

auto ASTInterpreter::initializationValue(
    const Type* type, FunctionSymbol* constructor, ExpressionAST* initializer,
    std::shared_ptr<ConstAddress> destination) -> std::optional<ConstValue> {
  EvaluationScope evaluationScope{*this};
  if (traits.is_reference(type)) return referenceBinding(initializer);
  if (!initializer)
    return defaultConstruct(type, std::move(destination), constructor);
  if (!constructor)
    return evaluateInitializer(type, initializer, std::move(destination));
  if (ast_cast<ConstExpressionAST>(Initializer{initializer}.clause()))
    return evaluateInitializer(type, initializer, std::move(destination));
  const auto clause = Initializer{initializer}.clause();
  auto paren = ast_cast<ParenInitializerAST>(clause);
  const auto valueInitialized =
      ast_cast<BracedInitListAST>(clause) || (paren && paren->lparenLoc);
  return classInitializationValue(type, constructor,
                                  Initializer{initializer}.arguments(),
                                  valueInitialized, std::move(destination));
}

auto ASTInterpreter::classInitializationValue(
    const Type* type, FunctionSymbol* constructor,
    const std::vector<ExpressionAST*>& arguments, bool valueInitialized,
    std::shared_ptr<ConstAddress> destination) -> std::optional<ConstValue> {
  if (auto array = unqualified_cast<BoundedArrayType>(type)) {
    auto elements = constructionArray(array, destination);
    if (!elements) return std::nullopt;
    for (std::size_t index = 0; index < array->size(); ++index) {
      auto address = arrayElementDestination(destination, type, index);
      auto value =
          classInitializationValue(array->elementType(), constructor, arguments,
                                   valueInitialized, std::move(address));
      if (!value) return std::nullopt;
      elements->mutableElement(index) = {std::move(*value),
                                         array->elementType()};
    }
    return ConstValue{std::move(elements)};
  }

  auto needsZeroInitialization = arguments.empty() && valueInitialized;
  if (!traits.requires_zero_initialization(type, constructor))
    needsZeroInitialization = false;
  if (!needsZeroInitialization)
    return evaluateConstructorFromExprs(constructor, type, arguments,
                                        std::move(destination));

  auto zero = zeroInitialize(type);
  if (!zero) return std::nullopt;
  if (traits.is_trivially_constructible(type, {})) return zero;
  auto object = std::get_if<std::shared_ptr<ConstObject>>(&*zero);
  if (!object || !*object) return std::nullopt;
  return evaluateConstructor(constructor, type, {}, *object,
                             std::move(destination));
}

auto ASTInterpreter::evaluateInitializer(
    const Type* type, ExpressionAST* initializer,
    std::shared_ptr<ConstAddress> destination) -> std::optional<ConstValue> {
  auto arrayType = type_cast<BoundedArrayType>(traits.remove_cv(type));
  auto literal = arrayType ? initializingStringLiteral(initializer) : nullptr;
  if (literal) return stringLiteralArray(arrayType, literal);

  auto value = expression(initializer, std::move(destination));
  auto clause = Initializer{initializer}.clause();
  if (value && traits.is_class(type) && is_glvalue(clause))
    return cloneValue(*value);
  return value;
}

auto ASTInterpreter::initializingStringLiteral(ExpressionAST* initializer)
    -> const StringLiteral* {
  auto arguments = Initializer{initializer}.arguments();
  if (arguments.size() != 1) return nullptr;
  auto expression = arguments.front();
  while (auto nested = ast_cast<NestedExpressionAST>(expression))
    expression = nested->expression;
  auto literal = ast_cast<StringLiteralExpressionAST>(expression);
  return literal ? literal->literal : nullptr;
}

auto ASTInterpreter::stringLiteralArray(const BoundedArrayType* type,
                                        const StringLiteral* literal)
    -> ConstValue {
  const auto initialized = std::min(type->size(), literal->charCount());
  std::vector<InitializerList::Element> elements;
  elements.reserve(initialized);
  for (std::size_t index = 0; index < initialized; ++index) {
    const std::intmax_t unit = literal->charAt(index);
    elements.emplace_back(ConstValue{unit}, type->elementType());
  }
  return ConstValue{std::make_shared<InitializerList>(
      std::move(elements), type->size(),
      InitializerList::Element{ConstValue{std::intmax_t{0}},
                               type->elementType()})};
}

auto ASTInterpreter::referenceBinding(ExpressionAST* initializer)
    -> std::optional<ConstValue> {
  if (auto defaultInitializer =
          ast_cast<DefaultInitializerExpressionAST>(initializer)) {
    InitializerContextGuard guard{*this, defaultInitializer->context};
    return referenceBinding(defaultInitializer->expression);
  }

  auto bound = referenceInitializer(initializer);
  if (!bound) return std::nullopt;

  EvaluationScope evaluationScope{*this};
  auto address = addressOfLvalue(bound);
  if (!address) return constexprUnknownAddress(bound);
  if (!std::holds_alternative<std::shared_ptr<ConstAddress>>(*address))
    return std::nullopt;
  return address;
}

auto ASTInterpreter::extendedReferenceBinding(const Type* referenceType,
                                              ExpressionAST* initializer,
                                              Symbol* extendingVariable)
    -> std::optional<ConstValue> {
  if (auto defaultInitializer =
          ast_cast<DefaultInitializerExpressionAST>(initializer)) {
    InitializerContextGuard guard{*this, defaultInitializer->context};
    return extendedReferenceBinding(
        referenceType, defaultInitializer->expression, extendingVariable);
  }

  auto temporary = materializedTemporary(referenceInitializer(initializer));
  if (!temporary) return referenceBinding(initializer);

  EvaluationScope evaluationScope{*this};
  auto referencedType = traits.remove_reference(referenceType);
  auto temporaryType = traits.is_same(traits.remove_cv(referencedType),
                                      traits.remove_cvref(temporary->type))
                           ? referencedType
                           : traits.remove_reference(temporary->type);
  auto variable = control()->newVariableSymbol(
      extendingVariable ? extendingVariable->parent() : nullptr,
      temporary->firstSourceLocation());
  variable->setName(control()->newAnonymousId("temporary"));
  variable->setType(temporaryType);
  variable->setLifetimeExtendingDeclaration(extendingVariable);
  variable->setConstexpr(traits.is_const(temporaryType));
  variable->setStatic(!extendingVariable ||
                      has_static_storage_duration(extendingVariable));
  if (has_thread_storage_duration(extendingVariable))
    variable->setThreadLocalKind(ThreadLocalKind::Cxx);
  auto value = initializationValue(variable, nullptr, temporary);
  if (!value) return std::nullopt;
  variable->setConstValue(std::move(*value));
  return std::make_shared<ConstAddress>(variable);
}

auto ASTInterpreter::isPotentiallyUsableInConstantExpressions(
    VariableSymbol* var) -> bool {
  auto definition = var->resolvedDefinition();
  if (!definition) definition = var;
  if (isUsableInConstantExpressions(traits, definition, currentFunction_))
    return true;
  return traits.is_const(definition->type()) &&
         isDependent(unit_, definition->type());
}

auto ASTInterpreter::referenceInitializer(ExpressionAST* initializer)
    -> ExpressionAST* {
  auto clause = Initializer{initializer}.clause();
  if (auto list = ast_cast<BracedInitListAST>(clause)) {
    if (auto element = listInitializedElement(traits, list)) return element;
    return list;
  }
  if (!ast_cast<ParenInitializerAST>(clause)) return clause;
  auto arguments = Initializer{clause}.arguments();
  if (arguments.size() != 1) return nullptr;
  return arguments.front();
}

auto ASTInterpreter::staticTemporary(const Type* type, ConstValue value,
                                     SourceLocation location,
                                     std::string_view name)
    -> std::shared_ptr<ConstAddress> {
  auto temporary = control()->newVariableSymbol(nullptr, location);
  temporary->setName(control()->newAnonymousId(name));
  temporary->setType(type);
  temporary->setConstexpr(traits.is_const(type));
  temporary->setStatic(true);
  temporary->setConstValue(std::move(value));
  return std::make_shared<ConstAddress>(temporary);
}

auto ASTInterpreter::memberValue(const std::shared_ptr<ConstObject>& object,
                                 const Symbol* member)
    -> std::optional<ConstValue> {
  if (object->lifetimeEnded()) return std::nullopt;
  auto value = object->subobject(member);
  if (!value) return std::nullopt;
  if (!traits.is_reference(member->type())) return *value;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(value);
  if (!address || !*address) return std::nullopt;
  return loadAddress(**address, 0, traits.remove_reference(member->type()));
}

auto ASTInterpreter::memberSlot(const std::shared_ptr<ConstObject>& object,
                                const Symbol* member) -> ConstValue* {
  if (object->lifetimeEnded()) return nullptr;
  auto slot = subobjectSlot(object, member);
  if (!slot || !traits.is_reference(member->type())) return slot;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(slot);
  if (!address || !*address) return nullptr;
  return addressSlot(**address, 0, traits.remove_reference(member->type()));
}

auto ASTInterpreter::memberAddress(MemberExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto field = symbol_cast<FieldSymbol>(ast->symbol);
  if (!field) return std::nullopt;
  if (field->isStatic()) return staticFieldAddress(field);

  auto base = memberBase(ast);
  base.object = addressedClassSubobject(base.object, base.address).object;
  auto baseType = traits.remove_cvref(ast->baseExpression->type);
  if (ast->accessOp == TokenKind::T_MINUS_GREATER)
    baseType = traits.remove_cv(traits.remove_pointer(baseType));
  auto displacement = fieldDisplacement(baseType, field);

  if (base.object && !base.object->isConstexprUnknown()) {
    auto result = memberAddress(base.object, field);
    if (result && base.address && displacement &&
        !traits.is_reference(field->type())) {
      if (!locateSubobject(*result, *base.address, *displacement))
        return std::nullopt;
    }
    return result;
  }

  if (traits.is_reference(field->type())) return std::nullopt;
  if (!base.address || !displacement) return std::nullopt;
  auto result = subobjectAddress(*base.address, *displacement);
  if (!result) return std::nullopt;
  result->setSymbol(field);
  result->setOwner(nullptr);
  result->setOffset(0);
  return result;
}

auto ASTInterpreter::designatedObjectAndAddress(ExpressionAST* ast,
                                                bool throughPointer)
    -> DesignatedObject {
  auto stripped = ast;
  while (auto nested = ast_cast<NestedExpressionAST>(stripped))
    stripped = nested->expression;

  if (throughPointer && ast_cast<ThisExpressionAST>(stripped)) {
    auto result = designatedThis();
    if (result.object && !result.address)
      result.address = std::make_shared<ConstAddress>(result.object, nullptr);
    return result;
  }

  if (throughPointer) {
    auto value = expression(ast);
    if (!value) return {};
    DesignatedObject result;
    result.address = pointerAddress(*value);
    result.object =
        pointeeObject(std::move(*value), traits.remove_pointer(ast->type));
    return result;
  }

  if (auto call = ast_cast<CallExpressionAST>(stripped);
      call && is_glvalue(call)) {
    auto result = evaluateCallExpression(call, CallResultKind::kObject);
    if (!result.value) return {};
    if (!result.isAddress) {
      auto object = std::get_if<std::shared_ptr<ConstObject>>(&*result.value);
      return {object ? *object : nullptr, nullptr};
    }
    auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*result.value);
    if (!address || !*address) return {};
    return {pointeeObject(*result.value, traits.remove_cvref(ast->type)),
            *address};
  }

  if (is_glvalue(ast)) {
    auto value = addressOfLvalue(ast);
    auto address =
        value ? std::get_if<std::shared_ptr<ConstAddress>>(&*value) : nullptr;
    if (address && *address)
      return {pointeeObject(*value, traits.remove_cvref(ast->type)), *address};
  }

  if (is_prvalue(ast) && traits.is_class(ast->type)) {
    auto address = materialize(ast);
    if (!address) return {};
    return {pointeeObject(ConstValue{address}, ast->type), address};
  }

  return {designatedObject(ast), nullptr};
}

auto ASTInterpreter::memberBase(MemberExpressionAST* ast) -> DesignatedObject {
  return designatedObjectAndAddress(
      ast->baseExpression, ast->accessOp == TokenKind::T_MINUS_GREATER);
}

auto ASTInterpreter::locateSubobject(ConstValue& value,
                                     const ConstAddress& base,
                                     std::intmax_t displacement) -> bool {
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value);
  if (!address || !*address) return false;
  if (base.root() || base.storage()) {
    auto located = subobjectDestination(std::make_shared<ConstAddress>(base),
                                        (*address)->symbol());
    if (!located) return false;
    value = std::move(located);
    return true;
  }
  auto located = std::make_shared<ConstAddress>(**address);
  located->setRoot(base.root());
  located->setStorage(base.storage());
  auto offset =
      checkedAdd(ConstInt{base.displacement()}, ConstInt{displacement});
  if (!offset) return false;
  located->setDisplacement(offset->toIntMax());
  value = located;
  return true;
}

auto ASTInterpreter::subscriptAddress(SubscriptExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (auto subscriptOperator = symbol_cast<FunctionSymbol>(ast->symbol)) {
    auto designated = designatedObjectAndAddress(ast->baseExpression, false);
    if (!designated.object) return std::nullopt;
    Frame frame;
    std::array<ExpressionAST*, 1> arguments{ast->indexExpression};
    if (!bindParametersFromExprs(frame, subscriptOperator, arguments))
      return std::nullopt;
    return executeFunction(subscriptOperator, std::move(frame),
                           CallResultKind::kAddress, designated.object,
                           /*constructor=*/false, designated.address)
        .value;
  }

  auto indexValue = evaluate(ast->indexExpression);
  if (!indexValue) return std::nullopt;
  auto index = toOffset(*indexValue);
  if (!index) return std::nullopt;

  auto baseType = traits.remove_cvref(ast->baseExpression->type);
  const auto subscriptsArray = traits.is_array(baseType);
  if (!subscriptsArray && !traits.is_pointer(baseType)) return std::nullopt;

  auto base = subscriptsArray ? addressOfLvalue(ast->baseExpression)
                              : evaluate(ast->baseExpression);
  if (!base) return std::nullopt;
  auto address = pointerAddress(*base);
  if (!address) return std::nullopt;

  if (subscriptsArray) address = arrayAddress(*address, baseType);
  if (!address) return std::nullopt;
  auto result =
      elementAddress(*address, *index, traits.get_element_type(baseType));
  if (!result) return std::nullopt;
  return result;
}

auto ASTInterpreter::convertsBetweenClasses(ImplicitCastExpressionAST* cast)
    -> bool {
  return cast->castKind == ImplicitCastKind::kDerivedToBaseConversion ||
         cast->castKind == ImplicitCastKind::kBaseToDerivedConversion;
}

auto ASTInterpreter::classConversionAddress(ImplicitCastExpressionAST* cast,
                                            const ConstValue& value)
    -> std::shared_ptr<ConstAddress> {
  if (!convertsBetweenClasses(cast)) return nullptr;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&value);
  if (!address || !*address) return nullptr;

  auto sourceType = traits.remove_cvref(cast->expression->type);
  auto targetType = traits.remove_cvref(cast->type);
  if (traits.is_pointer(sourceType)) {
    sourceType = traits.remove_cv(traits.remove_pointer(sourceType));
    targetType = traits.remove_cv(traits.remove_pointer(targetType));
  }

  const auto toBase =
      cast->castKind == ImplicitCastKind::kDerivedToBaseConversion;
  auto adjustment = toBase ? classSubobjectOffset(sourceType, targetType)
                           : classSubobjectOffset(targetType, sourceType);
  if (!adjustment) return nullptr;
  return adjustedClassAddress(**address, targetType,
                              toBase ? *adjustment : -*adjustment);
}

auto ASTInterpreter::callObjectAddress(
    FunctionSymbol* function, const std::shared_ptr<ConstObject>& object,
    const ConstAddress& address) -> std::shared_ptr<ConstAddress> {
  auto source = typedAddress(address);
  auto owner = function->enclosingClass();
  if (!owner || !function->isImplicitObjectMemberFunction()) return source;
  auto targetType = owner->type();
  if (traits.is_similar(source->elementType(), targetType)) return source;

  auto enclosing = addressedClassSubobject(object, source, targetType);
  std::optional<std::intmax_t> displacement;
  if (enclosing.object) {
    auto delta = checkedSub(ConstInt{enclosing.displacement},
                            ConstInt{source->classDisplacement()});
    if (delta) displacement = delta->toIntMax();
  } else {
    displacement = classSubobjectOffset(source->elementType(), targetType);
  }
  if (!displacement) return nullptr;
  return adjustedClassAddress(*source, targetType, *displacement);
}

auto ASTInterpreter::adjustedClassAddress(const ConstAddress& address,
                                          const Type* targetType,
                                          std::intmax_t displacement)
    -> std::shared_ptr<ConstAddress> {
  auto result = typedAddress(address);
  if (!validPosition(*result)) return nullptr;
  if (static_cast<std::uintmax_t>(result->offset()) == *arrayExtent(*result))
    return nullptr;
  if (result->arrayType()) {
    auto path = result->subobjectPath();
    path.emplace_back(std::size_t(result->offset()));
    result->setSubobjectPath(std::move(path));
  }
  result->setArrayType(nullptr);
  result->setElementType(targetType);
  result->setOffset(0);
  auto classOffset =
      checkedAdd(ConstInt{result->classDisplacement()}, ConstInt{displacement});
  if (!classOffset) return nullptr;
  result = subobjectAddress(*result, displacement);
  if (result) result->setClassDisplacement(classOffset->toIntMax());
  if (!result || result->subobjectPath().empty()) return result;

  auto storageType =
      result->storage() ? result->storage()->type() : result->root()->type();
  auto path = result->subobjectPath();
  for (std::size_t index = 0; index + 1 < path.size(); ++index) {
    if (auto member = std::get_if<Symbol*>(&path[index])) {
      storageType = traits.aggregate_element_type(*member);
      continue;
    }
    auto array = unqualified_cast<BoundedArrayType>(storageType);
    if (!array) return nullptr;
    storageType = array->elementType();
  }
  auto array = unqualified_cast<BoundedArrayType>(storageType);
  auto index = std::get_if<std::size_t>(&path.back());
  if (!array || !index || !traits.is_similar(array->elementType(), targetType))
    return result;
  result->setArrayType(storageType);
  result->setOffset(*index);
  path.pop_back();
  result->setSubobjectPath(std::move(path));
  return result;
}

auto ASTInterpreter::completeClassAddress(const ConstAddress& address)
    -> std::shared_ptr<ConstAddress> {
  auto result = typedAddress(address);
  auto path = result->subobjectPath();
  if (path.empty()) return result;
  auto last = std::get_if<Symbol*>(&path.back());
  if (!last || !symbol_cast<BaseClassSymbol>(*last)) return result;
  while (!path.empty()) {
    auto symbol = std::get_if<Symbol*>(&path.back());
    auto base = symbol ? symbol_cast<BaseClassSymbol>(*symbol) : nullptr;
    if (!base) break;
    auto enclosing = base->enclosingClass();
    auto baseClass = resolved_base_class(base);
    if (!enclosing || !baseClass) return nullptr;
    auto offset = classSubobjectOffset(enclosing->type(), baseClass->type());
    if (!offset) return nullptr;
    auto adjustment =
        checkedAdd(ConstInt{result->classDisplacement()}, ConstInt{*offset});
    if (!adjustment) return nullptr;
    result->setClassDisplacement(adjustment->toIntMax());
    path.pop_back();
  }
  result->setSubobjectPath(std::move(path));
  result->setSymbol(result->root());
  for (const auto& subobject : result->subobjectPath())
    if (auto symbol = std::get_if<Symbol*>(&subobject))
      result->setSymbol(*symbol);
  return result;
}

ASTInterpreter::ConstructionScope::ConstructionScope(
    ASTInterpreter& interpreter, const std::shared_ptr<ConstObject>& object,
    const std::shared_ptr<ConstAddress>& address)
    : interpreter_(interpreter) {
  auto completeObject = object;
  if (address) {
    auto completeAddress = interpreter.completeClassAddress(*address);
    if (completeAddress)
      completeObject = interpreter.pointeeObject(ConstValue{completeAddress},
                                                 object->type());
  }
  interpreter_.constructionPhases_.push_back(
      {std::move(completeObject), object});
}

ASTInterpreter::ConstructionScope::~ConstructionScope() {
  interpreter_.constructionPhases_.pop_back();
}

auto ASTInterpreter::fieldDisplacement(const Type* objectType,
                                       FieldSymbol* field)
    -> std::optional<std::intmax_t> {
  auto offset = fieldOffsetIn(objectType, field);
  if (!offset || !std::in_range<std::intmax_t>(*offset)) return std::nullopt;
  return static_cast<std::intmax_t>(*offset);
}

auto ASTInterpreter::typedAddress(const ConstAddress& address)
    -> std::shared_ptr<ConstAddress> {
  auto result = std::make_shared<ConstAddress>(address);
  if (result->elementType()) return result;
  if (auto literal = address.stringLiteral()) {
    const Type* type = control()->getCharType();
    switch (literal->encoding()) {
      case StringLiteralEncoding::kNone:
        break;
      case StringLiteralEncoding::kWide:
        type = control()->getWideCharType();
        break;
      case StringLiteralEncoding::kUtf8:
        type = control()->getChar8Type();
        break;
      case StringLiteralEncoding::kUtf16:
        type = control()->getChar16Type();
        break;
      case StringLiteralEncoding::kUtf32:
        type = control()->getChar32Type();
        break;
    }
    result->setElementType(type);
    result->setArrayType(
        control()->getBoundedArrayType(type, literal->charCount() + 1));
  } else if (address.symbol()) {
    result->setElementType(traits.remove_reference(address.symbol()->type()));
  } else if (address.storage()) {
    result->setElementType(address.storage()->type());
  } else if (address.owner()) {
    result->setElementType(address.owner()->type());
  }
  return result;
}

auto ASTInterpreter::arrayExtent(const ConstAddress& address) const
    -> std::optional<std::uintmax_t> {
  if (!address.elementType()) return std::nullopt;
  if (!address.arrayType()) return 1;
  auto array = unqualified_cast<BoundedArrayType>(address.arrayType());
  if (!array) return std::nullopt;
  return array->size();
}

auto ASTInterpreter::validPosition(const ConstAddress& address) const -> bool {
  if (address.lifetimeEnded() || address.offset() < 0) return false;
  auto extent = arrayExtent(address);
  if (!extent) return false;
  return static_cast<std::uintmax_t>(address.offset()) <= *extent;
}

auto ASTInterpreter::arrayAddress(const ConstAddress& base,
                                  const Type* arrayType)
    -> std::shared_ptr<ConstAddress> {
  auto result = typedAddress(base);
  if (!validPosition(*result)) return nullptr;
  if (static_cast<std::uintmax_t>(result->offset()) == *arrayExtent(*result))
    return nullptr;
  auto array = unqualified_cast<BoundedArrayType>(result->elementType());
  if (!array ||
      !traits.is_similar(traits.remove_address_space(result->elementType()),
                         traits.remove_address_space(arrayType)))
    return nullptr;
  auto path = result->subobjectPath();
  if (result->arrayType()) path.emplace_back(std::size_t(result->offset()));
  result->setSubobjectPath(std::move(path));
  result->setArrayType(result->elementType());
  result->setElementType(array->elementType());
  result->setOffset(0);
  return result;
}

auto ASTInterpreter::compareAddresses(TokenKind op, const Type* type,
                                      const ConstAddress& left,
                                      const ConstAddress& right)
    -> std::optional<ConstValue> {
  if (left.lifetimeEnded() || right.lifetimeEnded()) return std::nullopt;
  auto lhs = typedAddress(left);
  auto rhs = typedAddress(right);
  if (op == TokenKind::T_EQUAL_EQUAL || op == TokenKind::T_EXCLAIM_EQUAL) {
    bool equal = lhs->sameLocation(*rhs);
    if (traits.is_member_pointer(type)) {
      if (lhs->isNull() || rhs->isNull())
        equal = lhs->isNull() && rhs->isNull();
      else if (type_cast<MemberObjectPointerType>(type))
        equal = lhs->memberAdjustment() == rhs->memberAdjustment();
      else
        equal = lhs->symbol() == rhs->symbol() &&
                lhs->memberAdjustment() == rhs->memberAdjustment();
    }
    return ConstValue{op == TokenKind::T_EQUAL_EQUAL ? equal : !equal};
  }

  auto lo = lhs->offset();
  auto ro = rhs->offset();
  if (!lhs->sameArray(*rhs)) {
    if (!lhs->sameContainingObject(*rhs)) return std::nullopt;
    auto first = symbol_cast<FieldSymbol>(lhs->symbol());
    auto second = symbol_cast<FieldSymbol>(rhs->symbol());
    if (!first || !second) return std::nullopt;
    if (first->enclosingClass() != second->enclosingClass())
      return std::nullopt;
    lo = lhs->displacement();
    ro = rhs->displacement();
  }
  switch (op) {
    case TokenKind::T_LESS:
      return ConstValue{lo < ro};
    case TokenKind::T_GREATER:
      return ConstValue{lo > ro};
    case TokenKind::T_LESS_EQUAL:
      return ConstValue{lo <= ro};
    case TokenKind::T_GREATER_EQUAL:
      return ConstValue{lo >= ro};
    default:
      return std::nullopt;
  }
}

auto ASTInterpreter::elementAddress(const ConstAddress& base,
                                    std::intmax_t index,
                                    const Type* elementType)
    -> std::shared_ptr<ConstAddress> {
  if (base.isNull()) {
    if (index != 0) return nullptr;
    return std::make_shared<ConstAddress>(base);
  }
  auto result = typedAddress(base);
  if (!validPosition(*result)) return nullptr;
  elementType = traits.remove_reference(elementType);
  if (!elementType) elementType = result->elementType();
  if (!traits.is_similar(traits.remove_address_space(elementType),
                         traits.remove_address_space(result->elementType())))
    return nullptr;
  auto offset = checkedAdd(ConstInt{result->offset()}, ConstInt{index});
  if (!offset) return nullptr;
  result->setOffset(offset->toIntMax());
  if (!validPosition(*result)) return nullptr;

  auto size = control()->memoryLayout()->sizeOf(elementType);
  if (!size || !std::in_range<std::intmax_t>(*size)) return nullptr;
  auto delta =
      checkedMul(ConstInt{index}, ConstInt{static_cast<std::intmax_t>(*size)});
  if (!delta) return nullptr;
  auto displacement = checkedAdd(ConstInt{base.displacement()}, *delta);
  if (!displacement) return nullptr;
  result->setDisplacement(displacement->toIntMax());
  return result;
}

auto ASTInterpreter::subobjectAddress(const ConstAddress& base,
                                      std::intmax_t displacement)
    -> std::shared_ptr<ConstAddress> {
  auto offset =
      checkedAdd(ConstInt{base.displacement()}, ConstInt{displacement});
  if (!offset) return nullptr;
  auto result = std::make_shared<ConstAddress>(base);
  result->setDisplacement(offset->toIntMax());
  return result;
}

auto ASTInterpreter::memberAddress(const std::shared_ptr<ConstObject>& object,
                                   Symbol* member)
    -> std::optional<ConstValue> {
  if (!traits.is_reference(member->type())) {
    auto address = std::make_shared<ConstAddress>(object, member);
    if (auto field = symbol_cast<FieldSymbol>(member)) {
      auto displacement = fieldDisplacement(object->type(), field);
      if (!displacement) return std::nullopt;
      address->setDisplacement(*displacement);
    }
    return address;
  }
  if (object->lifetimeEnded()) return std::nullopt;
  auto value = object->subobject(member);
  if (!value) return std::nullopt;
  if (!std::holds_alternative<std::shared_ptr<ConstAddress>>(*value))
    return std::nullopt;
  return *value;
}

auto ASTInterpreter::typeInfoAddress(const Type* type)
    -> std::optional<ConstValue> {
  if (!type) return std::nullopt;
  return std::make_shared<ConstAddress>(
      traits.remove_cv(traits.remove_reference(type)));
}

auto ASTInterpreter::variableAddress(Symbol* variable)
    -> std::optional<ConstValue> {
  if (auto declared = symbol_cast<VariableSymbol>(variable);
      declared && declared->isThreadLocal())
    return std::nullopt;
  if (!traits.is_reference(variable->type())) {
    auto address = std::make_shared<ConstAddress>(variable);
    address->setStorage(lookupLocalStorage(variable));
    return address;
  }
  for (auto frame = frames_.rbegin(); frame != frames_.rend(); ++frame) {
    auto address = frame->referenceAddresses.find(variable);
    if (address != frame->referenceAddresses.end()) return address->second;
  }
  if (auto declared = symbol_cast<VariableSymbol>(variable)) {
    if (declared->constValue()) return declared->constValue();
  }
  return std::nullopt;
}

auto ASTInterpreter::addressOfLvalue(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  while (ast) {
    if (auto initializer = ast_cast<DefaultInitializerExpressionAST>(ast)) {
      InitializerContextGuard guard{*this, initializer->context};
      return addressOfLvalue(initializer->expression);
    }
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(ast);
        cast && cast->castKind ==
                    ImplicitCastKind::kTemporaryMaterializationConversion) {
      auto address = materialize(cast->expression);
      if (!address) return std::nullopt;
      return address;
    }
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(ast);
        cast && is_glvalue(cast) && convertsBetweenClasses(cast)) {
      auto operand = addressOfLvalue(cast->expression);
      if (!operand) return std::nullopt;
      auto converted = classConversionAddress(cast, *operand);
      if (!converted) return std::nullopt;
      return ConstValue{converted};
    }
    if (auto operand = sameObjectOperand(ast)) {
      ast = operand;
      continue;
    }
    if (auto alias = structuredBindingAlias(ast)) {
      ast = alias->initializer();
      continue;
    }
    if (auto constant = ast_cast<ConstExpressionAST>(ast)) {
      if (!frames_.empty() && constant->expression) {
        ast = constant->expression;
        continue;
      }
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

  if (auto cast = ast_cast<ImplicitCastExpressionAST>(ast)) {
    if (cast->castKind == ImplicitCastKind::kUserDefinedConversion)
      return evaluateConversionFunctionCall(cast, CallResultKind::kAddress)
          .value;
  }

  if (auto idExpr = ast_cast<IdExpressionAST>(ast)) {
    if (!idExpr->symbol) return std::nullopt;
    if (auto field = symbol_cast<FieldSymbol>(idExpr->symbol)) {
      if (field->isStatic()) return staticFieldAddress(field);
      auto owner = fieldOwner(ast);
      if (!owner) return std::nullopt;
      auto result = memberAddress(owner, field);
      auto thisAddress = implicitObjectFor(field).address;
      if (thisAddress) thisAddress = typedAddress(*thisAddress);
      auto objectType =
          thisAddress ? thisAddress->elementType() : owner->type();
      auto displacement = fieldDisplacement(objectType, field);
      if (result && thisAddress && displacement &&
          !traits.is_reference(field->type())) {
        if (!locateSubobject(*result, *thisAddress, *displacement))
          return std::nullopt;
      }
      return result;
    }
    if (auto function = designatedFunction(idExpr->symbol))
      return std::make_shared<ConstAddress>(function);
    return variableAddress(idExpr->symbol);
  }

  if (auto condition = ast_cast<ConditionExpressionAST>(ast)) {
    auto variable = initializeDecisionVariable(condition);
    if (!variable) return std::nullopt;
    return variableAddress(variable);
  }

  if (auto literal = ast_cast<StringLiteralExpressionAST>(ast)) {
    auto address = std::make_shared<ConstAddress>(literal->literal);
    address->setElementType(literal->type);
    return address;
  }

  if (auto member = ast_cast<MemberExpressionAST>(ast))
    return memberAddress(member);

  if (auto conditional = ast_cast<ConditionalExpressionAST>(ast)) {
    auto condition = expression(conditional->condition);
    auto truth = condition ? toBool(*condition) : std::nullopt;
    if (!truth) return std::nullopt;
    return addressOfLvalue(*truth ? conditional->iftrueExpression
                                  : conditional->iffalseExpression);
  }

  if (auto binary = ast_cast<BinaryExpressionAST>(ast);
      binary && binary->op == TokenKind::T_COMMA) {
    if (!discardedValue(binary->leftExpression)) return std::nullopt;
    return addressOfLvalue(binary->rightExpression);
  }

  if (auto unary = ast_cast<UnaryExpressionAST>(ast)) {
    if (unary->symbol)
      return evaluateUnaryOperatorCall(unary, CallResultKind::kAddress).value;
    if (unary->op != TokenKind::T_STAR) return std::nullopt;
    auto designated = designatedObjectAndAddress(unary->expression,
                                                 /*throughPointer=*/true);
    if (!designated.address) return std::nullopt;
    return ConstValue{designated.address};
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

  if (auto subExpr = ast_cast<SubscriptExpressionAST>(ast))
    return subscriptAddress(subExpr);

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

  auto representation = unit()->typeTraits().integral_representation(ast->type);

  if (representation && ConstInt::isRepresentableWidth(representation->bits)) {
    auto literal = ConstInt::make(
        static_cast<ConstInt::Wide>(static_cast<ConstInt::UWide>(value)),
        representation->bits, representation->isSigned);
    if (literal) return ConstValue{*literal};
  }

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
  if (ast->bracedInitList)
    return interp.expression(ast->bracedInitList, destination);
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  auto object = interp.designatedObjectAndAddress(ast, true);
  if (!object.address) return std::nullopt;
  auto address = interp.completeClassAddress(*object.address);
  if (!address) return std::nullopt;
  return ConstValue{std::move(address)};
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
  return interp.expression(ast->expression, destination);
}

auto ASTInterpreter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
  if (ast->expression) {
    return interp.expression(ast->expression, destination);
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

  if (auto alias = structuredBindingAlias(ast))
    return interp.expression(alias->initializer());

  if (auto conceptSymbol = symbol_cast<ConceptSymbol>(ast->symbol)) {
    auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
    if (!templateId) return std::nullopt;
    auto satisfied = ASTRewriter::evaluateConcept(
        unit(), conceptSymbol, templateId->templateArgumentList);
    if (!satisfied.has_value()) return std::nullopt;
    return ConstValue{*satisfied};
  }

  if (ast->symbol) {
    auto local = interp.lookupLocal(ast->symbol);
    if (local.has_value()) return local;
  }

  if (auto var = constantVariable(ast->symbol)) {
    if (!unit()->typeTraits().is_reference(var->type()))
      return var->constValue();
    auto address =
        std::get_if<std::shared_ptr<ConstAddress>>(&*var->constValue());
    if (!address || !*address) return std::nullopt;
    return interp.loadAddress(**address, 0, ast->type);
  }

  if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
      field && field->isStatic()) {
    return interp.evaluateStaticField(field);
  }

  if (auto field = symbol_cast<FieldSymbol>(ast->symbol)) {
    if (auto object = interp.fieldOwner(ast)) {
      if (auto fieldVal = interp.memberValue(object, field)) return fieldVal;
    }
  }

  if (auto func = designatedFunction(ast->symbol)) {
    return std::make_shared<ConstAddress>(func);
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::constantVariable(Symbol* symbol)
    -> VariableSymbol* {
  auto declared = symbol_cast<VariableSymbol>(symbol);
  if (!declared) return nullptr;
  auto var = declared->resolvedDefinition();
  if (!var) var = declared;
  if (!var->constValue().has_value()) return nullptr;
  if (!interp.isPotentiallyUsableInConstantExpressions(var)) return nullptr;
  return var;
}

auto ASTInterpreter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  auto classType = type_cast<ClassType>(ast->type);
  if (!classType || !classType->symbol()) return ExpressionResult{std::nullopt};

  auto closure = std::make_shared<ConstObject>(classType);

  for (auto captureNode : ListView{ast->captureList}) {
    if (is_pack_capture(captureNode)) return ExpressionResult{std::nullopt};

    auto captureField = capture_field(captureNode);
    auto initializer = capture_initializer(captureNode);
    if (!captureField || !initializer) return ExpressionResult{std::nullopt};

    auto value = interp.initializationValue(
        captureField->type(), captureField->constructor(), initializer);
    if (!value) return ExpressionResult{std::nullopt};

    closure->addMember(captureField, std::move(*value));
  }

  for (auto field :
       views::members(classType->symbol()) | views::non_static_fields) {
    if (!closure->subobject(field)) return ExpressionResult{std::nullopt};
  }

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
    auto typeId = typeRequirement->typeId;
    if (!typeId || !typeId->type) return false;
    if (isDependent(unit_, typeId->type)) return std::nullopt;
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
  if (auto op = symbol_cast<FunctionSymbol>(ast->symbol);
      op && op->isConstexpr()) {
    return evaluateOperatorCall(op, ast->baseExpression, ast->indexExpression);
  }

  auto baseExpressionResult = interp.expression(ast->baseExpression);
  auto indexExpressionResult = interp.expression(ast->indexExpression);

  if (!baseExpressionResult.has_value() || !indexExpressionResult.has_value())
    return std::nullopt;

  auto idx = interp.toOffset(*indexExpressionResult);
  if (!idx.has_value()) return std::nullopt;

  if (auto list = std::get_if<std::shared_ptr<InitializerList>>(
          &*baseExpressionResult)) {
    if (!*list || *idx < 0) return std::nullopt;
    if (static_cast<std::uintmax_t>(*idx) >= (*list)->size())
      return std::nullopt;
    return std::get<0>((*list)->distinctElement(*idx));
  }

  if (auto str = std::get_if<const StringLiteral*>(&*baseExpressionResult))
    return stringLiteralElement(*str, *idx);

  if (auto addr =
          std::get_if<std::shared_ptr<ConstAddress>>(&*baseExpressionResult)) {
    return interp.loadAddress(**addr, *idx, ast->type);
  }

  return std::nullopt;
}

auto ASTInterpreter::evaluateCallExpression(
    CallExpressionAST* ast, CallResultKind kind,
    std::shared_ptr<ConstAddress> destination) -> CallResult {
  auto base = ast->baseExpression;
  while (auto nested = ast_cast<NestedExpressionAST>(base))
    base = nested->expression;
  FunctionSymbol* function = nullptr;
  std::shared_ptr<ConstObject> object;
  std::shared_ptr<ConstAddress> objectAddress;
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
    function = designatedFunction(id->symbol);
    if (function && function->isImplicitObjectMemberFunction()) {
      auto designated = implicitObjectFor(function);
      object = std::move(designated.object);
      objectAddress = std::move(designated.address);
    }
  } else if (auto member = ast_cast<MemberExpressionAST>(base)) {
    function = designatedFunction(member->symbol);
    if (!function) return {};
    if (function->isStatic()) {
      if (!discardedValue(member->baseExpression)) return {};
    } else {
      auto designated = memberBase(member);
      if (!designated.object) return {};
      if (function->isImplicitObjectMemberFunction()) {
        object = std::move(designated.object);
        objectAddress = std::move(designated.address);
      }
    }
  }
  if (function && function->isVirtual() && ast->isVirtualDispatch) {
    function = dispatchVirtualCall(function, object);
    if (!function) return {};
  }
  if (ast->constructorSymbol) {
    std::vector<ExpressionAST*> arguments;
    for (auto argument : ListView{ast->expressionList})
      arguments.push_back(argument);
    return {evaluateConstructorFromExprs(ast->constructorSymbol, ast->type,
                                         arguments, std::move(destination)),
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
  return executeFunction(function, std::move(frame), kind, std::move(object),
                         /*constructor=*/false, std::move(objectAddress),
                         std::move(destination));
}

auto ASTInterpreter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  if (!is_glvalue(ast))
    return interp
        .evaluateCallExpression(ast, CallResultKind::kValue, destination)
        .value;
  auto result = interp.evaluateCallExpression(ast, CallResultKind::kAddress);
  if (!result.value) return std::nullopt;
  auto address = pointerAddress(*result.value);
  if (!address) return std::nullopt;
  return interp.loadAddress(*address, 0, ast->type);
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);

  if (!ast->type) return std::nullopt;
  if (auto classType = unqualified_cast<ClassType>(ast->type)) {
    auto classSymbol = classType->symbol();
    if (!classSymbol) return std::nullopt;
    std::vector<ExpressionAST*> arguments;
    for (auto argument : ListView{ast->expressionList})
      arguments.push_back(argument);
    return interp.classInitializationValue(ast->type, ast->constructorSymbol,
                                           arguments, true, destination);
  }
  if (!ast->expressionList) return interp.zeroInitialize(ast->type);
  if (!ast->expressionList->next)
    return interp.expression(ast->expressionList->value);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BracedTypeConstructionAST* ast) -> ExpressionResult {
  auto typeSpecifierResult = interp.specifier(ast->typeSpecifier);

  auto classType = unqualified_cast<ClassType>(ast->type);
  auto classSymbol = classType ? classType->symbol() : nullptr;

  if (classSymbol && ast->constructorSymbol) {
    auto initializer = ast->initialization.expression;
    if (!initializer) initializer = ast->bracedInitList;
    return interp.initializationValue(ast->type, ast->constructorSymbol,
                                      initializer, destination);
  }

  return interp.expression(ast->bracedInitList, destination);
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    SpliceMemberExpressionAST* ast) -> ExpressionResult {
  auto baseExpressionResult = interp.expression(ast->baseExpression);
  auto splicerResult = interp.splicer(ast->splicer);

  return ExpressionResult{std::nullopt};
}

auto ASTInterpreter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  if (vectorComponentAccess(ast)) {
    auto lane = interp.componentLane(ast);
    auto lanes = lane ? interp.componentVector(ast) : nullptr;
    if (!lanes || *lane >= lanes->size()) return std::nullopt;
    return std::get<0>(lanes->element(*lane));
  }

  auto object = interp.memberObject(ast);
  if (object && ast->symbol) {
    if (auto value = interp.memberValue(object, ast->symbol)) return value;
  }

  if (interp.thisObject() && ast->symbol) {
    if (auto fieldVal = interp.memberValue(interp.thisObject(), ast->symbol))
      return fieldVal;
  }

  auto nestedNameSpecifierResult =
      interp.nestedNameSpecifier(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = interp.unqualifiedId(ast->unqualifiedId);

  if (object && ast->symbol) {
    if (auto field = symbol_cast<FieldSymbol>(ast->symbol);
        field && field->isStatic()) {
      return interp.evaluateStaticField(field);
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
    auto zero = ConstExpressionAST::create(
        unit()->arena(), nullptr,
        unit()->arena()->make<ConstValue>(std::intmax_t{0}),
        ValueCategory::kPrValue, control()->getIntType());
    return evaluateOperatorCall(ast->symbol, ast->baseExpression, zero);
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
  if (interp.traits.is_void(ast->type))
    return interp.discardedValue(ast->expression);
  return interp.expression(ast->expression);
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) -> ExpressionResult {
  auto value = interp.expression(ast->expression);
  if (!value) return std::nullopt;
  return interp.bitCast(*value, ast->expression->type, ast->type);
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinConvertVectorExpressionAST*) -> ExpressionResult {
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinOffsetofExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = interp.typeId(ast->typeId);

  auto offset = interp.offsetOfDesignated(ast);
  if (!offset) return std::nullopt;

  return ConstValue{static_cast<std::intmax_t>(*offset)};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionResult {
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
    FunctionSymbol* function, ExpressionAST* operand,
    ExpressionAST* extraArgument, bool rightToLeft) -> ExpressionResult {
  if (!function || !function->isConstexpr()) return std::nullopt;

  std::vector<ExpressionAST*> arguments;
  if (!function->isImplicitObjectMemberFunction()) arguments.push_back(operand);
  if (extraArgument) arguments.push_back(extraArgument);

  Frame frame;
  if (rightToLeft &&
      !interp.bindParametersFromExprs(frame, function, arguments, true))
    return std::nullopt;

  std::shared_ptr<ConstObject> object;
  std::shared_ptr<ConstAddress> objectAddress;
  if (function->isImplicitObjectMemberFunction()) {
    auto designated = interp.designatedObjectAndAddress(operand, false);
    if (!designated.object) return std::nullopt;
    object = std::move(designated.object);
    objectAddress = std::move(designated.address);
  }

  if (!rightToLeft &&
      !interp.bindParametersFromExprs(frame, function, arguments))
    return std::nullopt;
  return interp
      .executeFunction(function, std::move(frame), CallResultKind::kValue,
                       std::move(object), false, std::move(objectAddress))
      .value;
}

auto ASTInterpreter::evaluateUnaryOperatorCall(UnaryExpressionAST* ast,
                                               CallResultKind kind)
    -> CallResult {
  auto function = symbol_cast<FunctionSymbol>(ast->symbol);
  if (!function || !function->isConstexpr()) return {};
  Frame frame;
  if (function->isImplicitObjectMemberFunction()) {
    auto object = designatedObjectAndAddress(ast->expression, false);
    if (!object.object) return {};
    return executeFunction(function, std::move(frame), kind,
                           std::move(object.object), false,
                           std::move(object.address));
  }
  std::array<ExpressionAST*, 1> arguments{ast->expression};
  if (!bindParametersFromExprs(frame, function, arguments)) return {};
  return executeFunction(function, std::move(frame), kind);
}

namespace {

[[nodiscard]] auto isLaneWiseUnaryOperator(TokenKind op) -> bool {
  return op == TokenKind::T_PLUS || op == TokenKind::T_MINUS ||
         op == TokenKind::T_TILDE || op == TokenKind::T_EXCLAIM;
}

}  // namespace

auto ASTInterpreter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  if (ast->symbol)
    return interp.evaluateUnaryOperatorCall(ast, CallResultKind::kValue).value;
  auto expressionResult = interp.expression(ast->expression);

  if (ast->expression && unqualified_cast<VectorType>(ast->expression->type) &&
      isLaneWiseUnaryOperator(ast->op)) {
    if (!expressionResult) return std::nullopt;
    return applyVectorUnaryOp(ast->op, ast->expression->type, ast->type,
                              *expressionResult);
  }

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
      if (!expressionResult.has_value()) break;
      auto truth = toBool(*expressionResult);
      if (!truth) return std::nullopt;
      return ExpressionResult(static_cast<std::intmax_t>(!*truth));
    }

    case TokenKind::T_STAR: {
      if (!expressionResult.has_value()) break;
      if (auto obj =
              std::get_if<std::shared_ptr<ConstObject>>(&*expressionResult)) {
        return *obj;
      }
      if (auto addr =
              std::get_if<std::shared_ptr<ConstAddress>>(&*expressionResult)) {
        return interp.loadAddress(**addr, 0, ast->type);
      }
      if (auto str = std::get_if<const StringLiteral*>(&*expressionResult))
        return stringLiteralElement(*str, 0);
      if (auto list = std::get_if<std::shared_ptr<InitializerList>>(
              &*expressionResult)) {
        if (*list && (*list)->size() != 0)
          return std::get<0>((*list)->distinctElement(0));
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
        auto operand =
            toConstInt(ast->expression->type, expressionResult.value());
        if (!operand) return std::nullopt;

        return ConstValue{~*operand};
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
        return ExpressionResult{ConstAddress::memberPointer(
            field, static_cast<std::intmax_t>(*offset))};
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
            ConstAddress::memberPointer(function, *adjustment)};
      }

      if (auto address = interp.addressOfLvalue(innerExpr))
        return ExpressionResult{std::move(address)};

      if (auto address = interp.constexprUnknownAddress(innerExpr))
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
  auto size = unit()->typeTraits().size_of(ast->expression->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->typeId || !ast->typeId->type) return std::nullopt;
  auto size = unit()->typeTraits().size_of(ast->typeId->type);
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
  auto size = unit()->typeTraits().alignment_of(ast->typeId->type);
  if (!size.has_value()) return std::nullopt;
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(*size)));
}

auto ASTInterpreter::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  if (!ast->expression || !ast->expression->type) return std::nullopt;
  auto typeAlignment = unit()->typeTraits().alignment_of(ast->expression->type);
  if (!typeAlignment.has_value()) return std::nullopt;
  auto alignment =
      declaredAlignment(declarationNamedBy(ast->expression), *typeAlignment);
  return ExpressionResult(
      std::bit_cast<std::intmax_t>(static_cast<std::uintmax_t>(alignment)));
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
  if (interp.traits.is_void(ast->type))
    return interp.discardedValue(ast->expression);
  return interp.expression(ast->expression);
}

auto ASTInterpreter::ExpressionVisitor::operator()(ConstExpressionAST* ast)
    -> ExpressionResult {
  if ((!interp.frames_.empty() || destination) && ast->expression)
    return interp.expression(ast->expression, destination);
  if (!ast->constValue) return std::nullopt;
  return *ast->constValue;
}

auto ASTInterpreter::ExpressionVisitor::evaluateConstructorConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto constructor = ast->conversionFunction;

  auto paren = ast_cast<ParenInitializerAST>(ast->expression);
  if (!paren) return std::nullopt;
  std::vector<ExpressionAST*> arguments;
  for (auto argument : ListView{Initializer{paren}.expressionList()})
    arguments.push_back(argument);
  return interp.evaluateConstructorFromExprs(constructor, ast->type, arguments,
                                             destination);
}

auto ASTInterpreter::evaluateConversionFunctionCall(
    ImplicitCastExpressionAST* ast, CallResultKind kind) -> CallResult {
  auto function = ast->conversionFunction;
  if (!function || !function->isConstexpr()) return {};
  auto object = designatedObject(ast->expression);
  if (!object) return {};
  if (ast->isVirtualDispatch && function->isVirtual()) {
    function = dispatchVirtualCall(function, object);
    if (!function) return {};
  }
  Frame frame;
  return executeFunction(function, std::move(frame), kind, std::move(object));
}

auto ASTInterpreter::ExpressionVisitor::evaluateMemberPointerConversion(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto target = decomposeMemberPointer(ast->type);
  if (!target) return std::nullopt;
  auto source = decomposeMemberPointer(ast->expression->type);
  if (!source) return ConstAddress::memberPointer(nullptr, 0);

  auto value = evaluate(ast->expression);
  if (!value) return std::nullopt;
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*value);
  if (!address || !*address) return std::nullopt;
  if (!(*address)->symbol()) return ConstAddress::memberPointer(nullptr, 0);

  auto adjustment = classSubobjectOffset(target.classType, source.classType);
  if (!adjustment) return std::nullopt;
  auto offset = checkedAdd(ConstInt{(*address)->memberAdjustment()},
                           ConstInt{*adjustment});
  if (!offset) return std::nullopt;
  return ConstAddress::memberPointer((*address)->symbol(), offset->toIntMax());
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  if (!ast->type) return std::nullopt;

  if (ast->castKind == ImplicitCastKind::kLValueToRValueConversion &&
      unit()->typeTraits().is_volatile(ast->expression->type))
    return std::nullopt;

  if (ast->castKind == ImplicitCastKind::kAddressSpaceConversion &&
      unit()->language() == LanguageKind::kCXX && !is_glvalue(ast))
    return std::nullopt;

  if (ast->castKind == ImplicitCastKind::kUserDefinedConversion &&
      ast->conversionFunction) {
    if (ast->conversionFunction->isConstructor())
      return evaluateConstructorConversion(ast);
    return interp.evaluateConversionFunctionCall(ast, CallResultKind::kValue)
        .value;
  }

  if (ast->castKind == ImplicitCastKind::kPointerToMemberConversion) {
    return evaluateMemberPointerConversion(ast);
  }

  if (ast->castKind == ImplicitCastKind::kArrayToPointerConversion) {
    auto innerExpr = Initializer{ast->expression}.clause();
    if (auto value = interp.addressOfLvalue(innerExpr)) {
      auto address = pointerAddress(*value);
      if (!address) return std::nullopt;
      auto result = interp.arrayAddress(*address, innerExpr->type);
      if (!result) return std::nullopt;
      return ConstValue{result};
    }
  }

  auto value = interp.expression(
      ast->expression, is_prvalue(ast) && interp.traits.is_class(ast->type)
                           ? destination
                           : nullptr);
  if (!value.has_value()) return std::nullopt;

  if (ast->castKind == ImplicitCastKind::kNonAtomicToAtomic) return value;

  if (auto converted = interp.classConversionAddress(ast, *value))
    return ConstValue{converted};

  if (unqualified_cast<VectorType>(ast->type))
    return evaluateVectorCast(ast, *value);

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
      if (unit()->typeTraits().is_floating_point(ast->type)) {
        return interp.toArithmeticType(*value, ast->type);
      }

      if (unit()->typeTraits().is_integral_or_enum(ast->type)) {
        return interp.toIntegralType(*value, ast->type);
      }

      if (unit()->language() == LanguageKind::kCXX &&
          unit()->typeTraits().is_pointer(ast->type) &&
          isNonNullInteger(*value))
        return std::nullopt;

      return value;
  }

  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::applyBinaryOp(
    TokenKind op, const Type* type, const ExpressionResult& left,
    const ExpressionResult& right) -> ExpressionResult {
  if (unqualified_cast<VectorType>(type)) {
    if (!left || !right) return std::nullopt;
    return applyVectorBinaryOp(op, type, type, *left, *right);
  }

  auto asAddress =
      [](const ConstValue& v) -> std::optional<std::shared_ptr<ConstAddress>> {
    if (auto a = std::get_if<std::shared_ptr<ConstAddress>>(&v)) return *a;
    if (auto s = std::get_if<const StringLiteral*>(&v))
      return std::make_shared<ConstAddress>(*s, std::intmax_t{0});
    return std::nullopt;
  };

  auto leftAddr = left.has_value() ? asAddress(*left) : std::nullopt;
  auto rightAddr = right.has_value() ? asAddress(*right) : std::nullopt;

  if (op == TokenKind::T_PLUS || op == TokenKind::T_MINUS) {
    if (!leftAddr && !rightAddr && interp.traits.is_pointer(type)) {
      if (!left || !right) return std::nullopt;
      auto pointer = interp.toOffset(*left);
      auto index = interp.toOffset(*right);
      if (!pointer || !index || *pointer != 0 || *index != 0)
        return std::nullopt;
      return ConstValue{ConstInt{std::intmax_t{0}}};
    }
    if (leftAddr && rightAddr && op == TokenKind::T_MINUS) {
      auto lhs = interp.typedAddress(**leftAddr);
      auto rhs = interp.typedAddress(**rightAddr);
      if (!interp.validPosition(*lhs) || !interp.validPosition(*rhs))
        return std::nullopt;
      if (!lhs->sameArray(*rhs)) return std::nullopt;
      const auto width = static_cast<int>(memoryLayout()->sizeOfPointer()) * 8;
      auto value = checkedSub(ConstInt{(*leftAddr)->offset()},
                              ConstInt{(*rightAddr)->offset()});
      if (!value) return std::nullopt;
      auto difference =
          ConstInt::make(value->toWide(), width, /*isSigned=*/true);
      if (!difference) return std::nullopt;
      return ConstValue{*difference};
    }
    auto elementType = interp.traits.get_element_type(type);
    if (leftAddr && right.has_value()) {
      auto n = interp.toOffset(*right);
      if (!n) return std::nullopt;
      auto delta = *n;
      if (op == TokenKind::T_MINUS) {
        auto negated = ConstInt{delta}.checkedNegate();
        if (!negated) return std::nullopt;
        delta = negated->toIntMax();
      }
      auto result = interp.elementAddress(**leftAddr, delta, elementType);
      if (!result) return std::nullopt;
      return ConstValue{result};
    }
    if (rightAddr && op == TokenKind::T_PLUS && left.has_value()) {
      auto n = interp.toOffset(*left);
      if (!n) return std::nullopt;
      auto result = interp.elementAddress(**rightAddr, *n, elementType);
      if (!result) return std::nullopt;
      return ConstValue{result};
    }
  }

  if (leftAddr && rightAddr)
    return interp.compareAddresses(op, type, **leftAddr, **rightAddr);

  if (bool(leftAddr) != bool(rightAddr)) {
    const ExpressionResult& other = leftAddr ? right : left;
    const auto* otherInt =
        other.has_value() ? std::get_if<ConstInt>(&*other) : nullptr;
    if (otherInt && otherInt->isZero()) {
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
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_GREATER:
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
      return scalarComparison(op, type, left, right);

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
    case TokenKind::T_AMP_AMP:
    case TokenKind::T_BAR_BAR: {
      if (ast->symbol || unqualified_cast<VectorType>(ast->type)) break;
      const auto shortCircuitValue = ast->op == TokenKind::T_BAR_BAR;
      auto left = evaluate(ast->leftExpression);
      if (!left.has_value()) return std::nullopt;
      auto leftTruth = toBool(*left);
      if (!leftTruth) return std::nullopt;
      if (*leftTruth == shortCircuitValue)
        return ExpressionResult{std::intmax_t{shortCircuitValue}};
      auto right = evaluate(ast->rightExpression);
      if (!right.has_value()) return std::nullopt;
      auto rightTruth = toBool(*right);
      if (!rightTruth) return std::nullopt;
      return ExpressionResult{std::intmax_t{*rightTruth}};
    }

    case TokenKind::T_COMMA: {
      if (!interp.discardedValue(ast->leftExpression)) return std::nullopt;
      return interp.expression(ast->rightExpression, destination);
    }

    case TokenKind::T_DOT_STAR:
    case TokenKind::T_MINUS_GREATER_STAR:
      return std::nullopt;

    default:
      break;
  }

  if (ast->symbol) {
    return evaluateOperatorCall(ast->symbol, ast->leftExpression,
                                ast->rightExpression);
  }

  auto left = evaluate(ast->leftExpression);
  if (!left.has_value()) return std::nullopt;

  auto right = evaluate(ast->rightExpression);
  if (!right.has_value()) return std::nullopt;

  if (unqualified_cast<VectorType>(ast->leftExpression->type))
    return applyVectorBinaryOp(ast->op, ast->type, ast->leftExpression->type,
                               *left, *right);

  auto operandType = interp.traits.is_pointer(ast->type)
                         ? ast->type
                         : ast->leftExpression->type;
  return applyBinaryOp(ast->op, operandType, left, right);
}

auto ASTInterpreter::ExpressionVisitor::vectorLanes(const ConstValue& value,
                                                    const VectorType* vector)
    -> std::optional<std::vector<ConstValue>> {
  auto list = std::get_if<std::shared_ptr<InitializerList>>(&value);
  if (!list || !*list || (*list)->size() > vector->elementCount())
    return std::nullopt;
  std::vector<ConstValue> lanes;
  lanes.reserve(vector->elementCount());
  for (std::size_t index = 0; index < (*list)->size(); ++index) {
    const auto& element = std::get<0>((*list)->element(index));
    auto lane = convertLane(element, vector->elementType());
    if (!lane) return std::nullopt;
    lanes.push_back(std::move(*lane));
  }
  while (lanes.size() < vector->elementCount()) {
    auto zero = interp.zeroInitialize(vector->elementType());
    if (!zero) return std::nullopt;
    lanes.push_back(std::move(*zero));
  }
  return lanes;
}

auto ASTInterpreter::ExpressionVisitor::convertLane(const ConstValue& value,
                                                    const Type* type)
    -> ExpressionResult {
  if (unit()->typeTraits().is_floating_point(type))
    return interp.toArithmeticType(value, type);
  if (unit()->typeTraits().is_integral_or_enum(type))
    return interp.toIntegralType(value, type);
  return std::nullopt;
}

auto ASTInterpreter::ExpressionVisitor::makeVector(
    std::vector<ConstValue> lanes, const VectorType* vector)
    -> ExpressionResult {
  auto list = std::make_shared<InitializerList>();
  for (auto& lane : lanes) list->append(std::move(lane), vector->elementType());
  return ConstValue{std::move(list)};
}

auto ASTInterpreter::ExpressionVisitor::evaluateVectorCast(
    ImplicitCastExpressionAST* ast, const ConstValue& value)
    -> ExpressionResult {
  auto target = unqualified_cast<VectorType>(ast->type);
  if (ast->castKind == ImplicitCastKind::kVectorSplat) {
    auto lane = convertLane(value, target->elementType());
    if (!lane) return std::nullopt;
    return makeVector(std::vector<ConstValue>(target->elementCount(), *lane),
                      target);
  }
  auto source = unqualified_cast<VectorType>(ast->expression->type);
  if (source && ast->castKind == ImplicitCastKind::kBooleanConversion) {
    auto lanes = vectorLanes(value, source);
    if (!lanes) return std::nullopt;
    for (auto& lane : *lanes) {
      auto converted = convertLane(lane, target->elementType());
      if (!converted) return std::nullopt;
      lane = std::move(*converted);
    }
    return makeVector(std::move(*lanes), target);
  }
  if (!source || source->elementCount() != target->elementCount() ||
      !unit()->typeTraits().is_same(
          unit()->typeTraits().remove_cv(source->elementType()),
          unit()->typeTraits().remove_cv(target->elementType())))
    return std::nullopt;
  auto lanes = vectorLanes(value, source);
  if (!lanes) return std::nullopt;
  return makeVector(std::move(*lanes), target);
}

auto ASTInterpreter::ExpressionVisitor::laneMask(std::optional<bool> truth)
    -> ExpressionResult {
  if (!truth) return std::nullopt;
  return ConstValue{std::intmax_t{*truth ? -1 : 0}};
}

auto ASTInterpreter::ExpressionVisitor::logicalLane(TokenKind op,
                                                    const ConstValue& left,
                                                    const ConstValue& right)
    -> ExpressionResult {
  auto leftTruth = toBool(left);
  auto rightTruth = toBool(right);
  if (!leftTruth || !rightTruth) return std::nullopt;
  const auto truth = op == TokenKind::T_AMP_AMP ? *leftTruth && *rightTruth
                                                : *leftTruth || *rightTruth;
  return ConstValue{std::intmax_t{truth}};
}

auto ASTInterpreter::ExpressionVisitor::applyVectorBinaryOp(
    TokenKind op, const Type* resultType, const Type* operandType,
    const ConstValue& left, const ConstValue& right) -> ExpressionResult {
  auto operand = unqualified_cast<VectorType>(operandType);
  auto result = unqualified_cast<VectorType>(resultType);
  if (!operand || !result || result->elementCount() != operand->elementCount())
    return std::nullopt;
  auto leftLanes = vectorLanes(left, operand);
  auto rightLanes = vectorLanes(right, operand);
  if (!leftLanes || !rightLanes) return std::nullopt;
  const auto isComparison =
      op == TokenKind::T_EQUAL_EQUAL || op == TokenKind::T_EXCLAIM_EQUAL ||
      op == TokenKind::T_LESS || op == TokenKind::T_LESS_EQUAL ||
      op == TokenKind::T_GREATER || op == TokenKind::T_GREATER_EQUAL;
  const auto isLogical =
      op == TokenKind::T_AMP_AMP || op == TokenKind::T_BAR_BAR;
  std::vector<ConstValue> lanes;
  lanes.reserve(operand->elementCount());
  for (std::size_t index = 0; index < operand->elementCount(); ++index) {
    const auto& leftLane = (*leftLanes)[index];
    const auto& rightLane = (*rightLanes)[index];
    auto lane = isLogical ? logicalLane(op, leftLane, rightLane)
                          : applyBinaryOp(op, operand->elementType(),
                                          ExpressionResult{leftLane},
                                          ExpressionResult{rightLane});
    if (lane && (isComparison || isLogical)) lane = laneMask(toBool(*lane));
    if (!lane) return std::nullopt;
    auto converted = convertLane(*lane, result->elementType());
    if (!converted) return std::nullopt;
    lanes.push_back(std::move(*converted));
  }
  return makeVector(std::move(lanes), result);
}

auto ASTInterpreter::ExpressionVisitor::applyVectorUnaryOp(
    TokenKind op, const Type* operandType, const Type* resultType,
    const ConstValue& value) -> ExpressionResult {
  auto vector = unqualified_cast<VectorType>(operandType);
  auto resultVector = unqualified_cast<VectorType>(resultType);
  auto lanes =
      vector && resultVector ? vectorLanes(value, vector) : std::nullopt;
  if (!lanes) return std::nullopt;
  for (auto& lane : *lanes) {
    ExpressionResult result;
    switch (op) {
      case TokenKind::T_PLUS:
        result = lane;
        break;
      case TokenKind::T_MINUS:
        if (auto integer = std::get_if<ConstInt>(&lane))
          result = ConstValue{-*integer};
        else
          result =
              unary_minus_op(vector->elementType(), ExpressionResult{lane});
        break;
      case TokenKind::T_TILDE:
        result = applyBinaryOp(TokenKind::T_CARET, vector->elementType(),
                               ExpressionResult{lane},
                               ExpressionResult{std::intmax_t{-1}});
        break;
      case TokenKind::T_EXCLAIM: {
        auto truth = toBool(lane);
        result = laneMask(truth ? std::optional{!*truth} : std::nullopt);
        break;
      }
      default:
        return std::nullopt;
    }
    if (!result) return std::nullopt;
    auto converted = convertLane(*result, resultVector->elementType());
    if (!converted) return std::nullopt;
    lane = std::move(*converted);
  }
  return makeVector(std::move(*lanes), resultVector);
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

  auto truth = toBool(*conditionResult);
  if (!truth) return std::nullopt;

  return interp.expression(
      *truth ? ast->iftrueExpression : ast->iffalseExpression, destination);
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
  if (ast->symbol)
    return evaluateOperatorCall(ast->symbol, ast->leftExpression,
                                ast->rightExpression, true);

  auto rightExpressionResult = interp.expression(ast->rightExpression);
  if (!rightExpressionResult.has_value()) return std::nullopt;

  auto slot = interp.lvalue(ast->leftExpression);
  if (!slot) return std::nullopt;

  *slot = cloneValue(*rightExpressionResult);
  return *slot;
}

auto ASTInterpreter::ExpressionVisitor::operator()(TargetExpressionAST* ast)
    -> ExpressionResult {
  return interp.assignmentValue_;
}

auto ASTInterpreter::ExpressionVisitor::operator()(RightExpressionAST* ast)
    -> ExpressionResult {
  return interp.assignmentValue_;
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    CompoundAssignmentExpressionAST* ast) -> ExpressionResult {
  if (ast->symbol) {
    return evaluateOperatorCall(ast->symbol, ast->targetExpression,
                                ast->rightExpression, true);
  }
  auto binOp = get_underlying_binary_op(ast->op);

  const auto type =
      ast->targetExpression ? ast->targetExpression->type : ast->type;
  if (!type) return std::nullopt;

  auto rightExpressionResult = interp.expression(ast->rightExpression);
  if (!rightExpressionResult.has_value()) return std::nullopt;

  auto slot = interp.lvalue(ast->targetExpression);
  if (!slot) return std::nullopt;

  ExpressionResult targetValue{*slot};
  std::swap(interp.assignmentValue_, targetValue);
  auto left = evaluate(ast->leftExpression);
  std::swap(interp.assignmentValue_, targetValue);
  if (!left) return std::nullopt;

  auto result = applyBinaryOp(binOp, ast->leftExpression->type, left,
                              rightExpressionResult);
  if (!result.has_value()) return std::nullopt;

  std::swap(interp.assignmentValue_, result);
  auto adjusted = evaluate(ast->adjustExpression);
  std::swap(interp.assignmentValue_, result);
  if (!adjusted) return std::nullopt;

  *slot = std::move(*adjusted);
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

      case BuiltinTypeTraitKind::T___HAS_TRIVIAL_CONSTRUCTOR:
        return unit()->typeTraits().has_trivial_default_constructor(firstType);

      case BuiltinTypeTraitKind::T___HAS_TRIVIAL_DESTRUCTOR:
        return unit()->typeTraits().has_trivial_destructor(firstType);

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
  auto variable = interp.initializeDecisionVariable(ast);
  if (!variable) return std::nullopt;
  return interp.lookupLocal(variable);
}

auto ASTInterpreter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionResult {
  auto expressionResult = interp.expression(ast->expression, destination);

  return expressionResult;
}

namespace {
[[nodiscard]] auto initializerExpression(ExpressionAST* ast) -> ExpressionAST* {
  if (auto equalInitializer = ast_cast<EqualInitializerAST>(ast))
    return equalInitializer->expression;
  return ast;
}

[[nodiscard]] auto designatedSlots(ASTInterpreter& interp,
                                   TranslationUnit* unit, ConstValue* slot,
                                   List<DesignatorAST*>* designatorList)
    -> std::vector<ConstValue*> {
  if (!slot) return {};
  if (!designatorList) return {slot};
  auto designator = designatorList->value;
  if (auto dot = ast_cast<DotDesignatorAST>(designator)) {
    if (!dot->symbol) return {};
    auto object = std::get_if<std::shared_ptr<ConstObject>>(slot);
    if (!object || !*object) return {};
    return designatedSlots(interp, unit,
                           interp.subobjectSlot(*object, dot->symbol),
                           designatorList->next);
  }
  auto subscript = ast_cast<SubscriptDesignatorAST>(designator);
  auto list = std::get_if<std::shared_ptr<InitializerList>>(slot);
  if (!list || !*list) return {};
  auto range = designatedArrayRange(unit, subscript);
  if (!range || *range->end > (*list)->size()) return {};
  std::vector<ConstValue*> slots;
  for (auto index = range->begin; index < *range->end; ++index) {
    auto nested = designatedSlots(interp, unit,
                                  &std::get<0>((*list)->mutableElement(index)),
                                  designatorList->next);
    slots.insert(slots.end(), nested.begin(), nested.end());
  }
  return slots;
}
}  // namespace

auto ASTInterpreter::ExpressionVisitor::aggregateObject(
    BracedInitListAST* ast, ClassSymbol* classSymbol) -> ExpressionResult {
  auto elements = unit()->typeTraits().aggregate_elements(classSymbol);
  auto object = interp.constructionObject(ast->type, destination);
  if (!object) return std::nullopt;

  std::size_t elementIndex = 0;
  for (auto node : ListView{Initializer{ast}.expressionList()}) {
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

    std::optional<ThisScope> defaultMemberScope;
    if (ast_cast<DefaultInitializerExpressionAST>(clause))
      defaultMemberScope.emplace(interp, object, destination);

    if (!object->subobject(element))
      object->setMember(element, IndeterminateValue{});
    auto address = interp.subobjectDestination(destination, element);

    auto value = subobjectDesignators
                     ? interp.evaluate(clause)
                     : interp.initialValue(
                           unit()->typeTraits().aggregate_element_type(element),
                           clause, std::move(address));
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
      auto slots =
          designatedSlots(interp, unit(), interp.subobjectSlot(object, element),
                          subobjectDesignators);
      if (slots.empty()) return std::nullopt;
      for (auto slot : slots) *slot = cloneValue(*value);
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

auto ASTInterpreter::ExpressionVisitor::vectorValue(BracedInitListAST* ast,
                                                    const VectorType* vector)
    -> ExpressionResult {
  auto list = std::make_shared<InitializerList>();
  for (auto node : ListView{Initializer{ast}.expressionList()}) {
    auto value = interp.evaluate(node);
    if (!value) return std::nullopt;
    list->append(std::move(*value), node->type);
  }
  auto lanes = vectorLanes(ConstValue{std::move(list)}, vector);
  if (!lanes) return std::nullopt;
  return makeVector(std::move(*lanes), vector);
}

auto ASTInterpreter::ExpressionVisitor::arrayValue(BracedInitListAST* ast,
                                                   const BoundedArrayType* type)
    -> ExpressionResult {
  auto elementType = type->elementType();

  if (auto literal = initializingStringLiteral(ast))
    return interp.stringLiteralArray(type, literal);

  auto zero = interp.zeroInitialize(type);
  if (!zero) return std::nullopt;
  auto list = std::get<std::shared_ptr<InitializerList>>(*zero);
  if (destination) {
    auto slot = interp.addressSlot(*destination, 0, type);
    if (!slot) return std::nullopt;
    *slot = list;
  }

  std::size_t elementIndex = 0;
  for (auto node : ListView{Initializer{ast}.expressionList()}) {
    auto clause = node;
    std::vector<ConstValue*> slots;
    if (auto designated = ast_cast<DesignatedInitializerClauseAST>(node)) {
      clause = initializerExpression(designated->initializer);
      auto subscript =
          ast_cast<SubscriptDesignatorAST>(designated->designatorList->value);
      auto range = designatedArrayRange(unit(), subscript);
      if (!range) return std::nullopt;
      elementIndex = *range->end;
      ConstValue aggregate = list;
      slots = designatedSlots(interp, unit(), &aggregate,
                              designated->designatorList);
    } else if (elementIndex < type->size()) {
      slots.push_back(&std::get<0>(list->mutableElement(elementIndex++)));
    }
    if (slots.empty() || !clause) return std::nullopt;
    auto address =
        destination && !ast_cast<DesignatedInitializerClauseAST>(node)
            ? interp.arrayElementDestination(destination, type,
                                             elementIndex - 1)
            : nullptr;
    auto value = interp.expression(clause, address);
    if (!value) return std::nullopt;
    for (auto slot : slots) *slot = address ? *value : cloneValue(*value);
  }

  if (auto element = ast->implicitElement) {
    for (auto range :
         implicitlyInitializedElements(unit(), ast, type->size())) {
      for (auto index = range.begin; index < *range.end; ++index) {
        auto address = interp.arrayElementDestination(destination, type, index);
        auto value = interp.initialValue(elementType, element->initializer(),
                                         std::move(address));
        if (!value) return std::nullopt;
        std::get<0>(list->mutableElement(index)) = std::move(*value);
      }
    }
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

  for (auto node : ListView{Initializer{ast}.expressionList()}) {
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

  if (auto vectorType = unqualified_cast<VectorType>(ast->type))
    return vectorValue(ast, vectorType);

  if (auto classType = unqualified_cast<ClassType>(ast->type)) {
    if (traits.is_aggregate(ast->type))
      return aggregateObject(ast, classType->symbol());
    if (!traits.initializer_list_element_type(ast->type)) return std::nullopt;
  } else if (!traits.is_class(ast->type)) {
    if (!Initializer{ast}.expressionList())
      return interp.zeroInitialize(ast->type);
    if (!Initializer{ast}.expressionList()->next)
      return interp.evaluate(Initializer{ast}.expressionList()->value);
  }

  auto values = std::vector<std::tuple<ConstValue, const Type*>>();
  for (auto node : ListView{Initializer{ast}.expressionList()}) {
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
  auto size = static_cast<std::intmax_t>(elements->size());
  auto first = interp.staticTemporary(
      control()->getBoundedArrayType(traits.add_const(elementType),
                                     elements->size()),
      ConstValue{elements}, ast->firstSourceLocation(), "initializer_list");
  first = interp.arrayAddress(*first, first->symbol()->type());
  if (!first) return std::nullopt;
  auto object = std::make_shared<ConstObject>(ast->type);
  object->addMember(fields[0], first);
  if (traits.is_pointer(fields[1]->type())) {
    auto last = interp.elementAddress(*first, size, elementType);
    if (!last) return std::nullopt;
    object->addMember(fields[1], last);
  } else if (traits.is_integral(fields[1]->type()))
    object->addMember(fields[1], size);
  else
    return std::nullopt;
  return object;
}

auto ASTInterpreter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  std::optional<ConstValue> result;
  std::size_t count = 0;

  for (auto node : ListView{Initializer{ast}.expressionList()}) {
    auto value = interp.expression(node, destination);
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
