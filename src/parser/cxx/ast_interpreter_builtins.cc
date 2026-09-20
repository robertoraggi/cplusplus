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
#include <cxx/builtin_bit_operations.h>
#include <cxx/const_int.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

// cstd
#include <bit>
#include <cmath>
#include <limits>

namespace cxx {

namespace {

[[nodiscard]] auto argumentAt(CallExpressionAST* ast, int index)
    -> ExpressionAST* {
  int current = 0;
  for (auto argument : ListView{ast->expressionList}) {
    if (current++ == index) return argument;
  }
  return nullptr;
}

[[nodiscard]] auto strippedPointerType(ExpressionAST* ast)
    -> const PointerType* {
  while (auto cast = ast_cast<ImplicitCastExpressionAST>(ast))
    ast = cast->expression;
  if (!ast) return nullptr;
  return type_cast<PointerType>(ast->type);
}

}  // namespace

auto ASTInterpreter::evaluateBuiltinArithmeticOverflow(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto leftExpression = argumentAt(ast, 0);
  auto rightExpression = argumentAt(ast, 1);
  auto outputExpression = argumentAt(ast, 2);
  if (!leftExpression || !rightExpression || !outputExpression)
    return std::nullopt;

  auto leftValue = evaluate(leftExpression);
  auto rightValue = evaluate(rightExpression);
  auto outputValue = evaluate(outputExpression);
  if (!leftValue || !rightValue || !outputValue) return std::nullopt;

  auto left = toInt(*leftValue);
  auto right = toInt(*rightValue);
  auto address = std::get_if<std::shared_ptr<ConstAddress>>(&*outputValue);
  if (!left || !right || !address || !*address) return std::nullopt;

  auto slot = addressSlot(**address, 0);
  auto pointer = type_cast<PointerType>(outputExpression->type);
  if (!slot || !pointer) return std::nullopt;

  auto outputType = traits.remove_cv(pointer->elementType());
  auto size = control()->memoryLayout()->sizeOf(outputType);
  if (!size || *size > sizeof(std::intmax_t)) return std::nullopt;

  const auto isBoolOutput =
      traits.is_same(outputType, control()->getBoolType());
  const auto width = isBoolOutput ? 1 : static_cast<int>(*size * 8);
  const auto isSignedOutput = traits.is_signed(outputType);

  auto widen = [&](std::intmax_t value, const Type* type) -> ConstInt::Wide {
    if (traits.is_signed(type)) return static_cast<ConstInt::Wide>(value);
    return static_cast<ConstInt::Wide>(
        static_cast<ConstInt::UWide>(static_cast<std::uintmax_t>(value)));
  };

  const auto lhs = widen(*left, leftExpression->type);
  const auto rhs = widen(*right, rightExpression->type);

  ConstInt::Wide value = 0;
  switch (resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression))) {
    case BuiltinFunctionKind::T___BUILTIN_ADD_OVERFLOW:
      value = lhs + rhs;
      break;
    case BuiltinFunctionKind::T___BUILTIN_SUB_OVERFLOW:
      value = lhs - rhs;
      break;
    case BuiltinFunctionKind::T___BUILTIN_MUL_OVERFLOW:
      value = lhs * rhs;
      break;
    default:
      return std::nullopt;
  }

  auto fits = [&]() -> bool {
    if (isSignedOutput) {
      const auto limit = ConstInt::Wide{1} << (width - 1);
      return value >= -limit && value < limit;
    }
    if (value < 0) return false;
    const auto limit = width >= ConstInt::maxWidth
                           ? ~ConstInt::UWide{0}
                           : (ConstInt::UWide{1} << width) - 1;
    return static_cast<ConstInt::UWide>(value) <= limit;
  };

  const auto overflow = !fits();

  auto result = ConstInt::make(value, width, isSignedOutput);
  if (!result) return std::nullopt;

  *slot = result->toIntMax();
  return ConstValue{overflow};
}

auto ASTInterpreter::evaluateBuiltinFloatComparison(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto leftExpression = argumentAt(ast, 0);
  auto rightExpression = argumentAt(ast, 1);
  if (!leftExpression || !rightExpression) return std::nullopt;

  auto leftValue = evaluate(leftExpression);
  auto rightValue = evaluate(rightExpression);
  if (!leftValue || !rightValue) return std::nullopt;

  auto left = toFloat(*leftValue);
  auto right = toFloat(*rightValue);
  if (!left || !right) return std::nullopt;

  const auto unordered = std::isnan(*left) || std::isnan(*right);

  switch (resolveBuiltinFunctionKind(
      ast_cast<IdExpressionAST>(ast->baseExpression))) {
    case BuiltinFunctionKind::T___BUILTIN_ISGREATER:
      return ConstValue{!unordered && *left > *right};
    case BuiltinFunctionKind::T___BUILTIN_ISGREATEREQUAL:
      return ConstValue{!unordered && *left >= *right};
    case BuiltinFunctionKind::T___BUILTIN_ISLESS:
      return ConstValue{!unordered && *left < *right};
    case BuiltinFunctionKind::T___BUILTIN_ISLESSEQUAL:
      return ConstValue{!unordered && *left <= *right};
    case BuiltinFunctionKind::T___BUILTIN_ISLESSGREATER:
      return ConstValue{!unordered && *left != *right};
    case BuiltinFunctionKind::T___BUILTIN_ISUNORDERED:
      return ConstValue{unordered};
    default:
      return std::nullopt;
  }
}

auto ASTInterpreter::evaluateBuiltinAddressof(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateAddress(argumentAt(ast, 0));
}

auto ASTInterpreter::sourceLocation(CallExpressionAST* ast) const
    -> SourceLocation {
  return defaultInitializerContext_.location
             ? defaultInitializerContext_.location
             : ast->firstSourceLocation();
}

auto ASTInterpreter::sourceFunction(CallExpressionAST* ast) const
    -> FunctionSymbol* {
  if (!defaultInitializerContext_.location) return currentFunction_;
  auto scope = defaultInitializerContext_.scope;
  if (auto function = symbol_cast<FunctionSymbol>(scope)) return function;
  return scope ? scope->enclosingFunction() : nullptr;
}

auto ASTInterpreter::evaluateBuiltinColumn(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  return ConstValue{static_cast<std::intmax_t>(pos.column)};
}

auto ASTInterpreter::evaluateBuiltinSourceLocation(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pointer = type_cast<PointerType>(ast->type);
  auto type =
      pointer ? unqualified_cast<ClassType>(pointer->elementType()) : nullptr;
  if (!type) return std::nullopt;
  auto object = std::make_shared<ConstObject>(type);
  auto function = sourceFunction(ast);
  auto functionName =
      function ? to_string(function->type(), to_string(function->name()))
               : std::string{};
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  for (auto member : type->symbol()->members()) {
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic()) continue;
    auto name = to_string(field->name());
    if (name == "_M_file_name")
      object->addMember(field, control()->stringLiteralFromValue(pos.fileName));
    else if (name == "_M_function_name")
      object->addMember(field, control()->stringLiteralFromValue(functionName));
    else if (name == "_M_line")
      object->addMember(field, static_cast<std::intmax_t>(pos.line));
    else if (name == "_M_column")
      object->addMember(field, static_cast<std::intmax_t>(pos.column));
    else
      return std::nullopt;
  }
  auto variable =
      control()->newVariableSymbol(unit_->globalScope(), sourceLocation(ast));
  variable->setName(control()->newAnonymousId("source_location"));
  variable->setType(traits.add_const(type));
  variable->setConstexpr(true);
  variable->setStatic(true);
  variable->setConstValue(ConstValue{std::move(object)});
  return ConstValue{std::make_shared<ConstAddress>(variable)};
}

auto ASTInterpreter::evaluateBuiltinLine(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  return ConstValue{static_cast<std::intmax_t>(pos.line)};
}

auto ASTInterpreter::evaluateBuiltinFile(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  auto pos = unit_->presumedTokenStartPosition(sourceLocation(ast));
  auto lit = control()->stringLiteralFromValue(pos.fileName);
  return ConstValue{lit};
}

auto ASTInterpreter::evaluateBuiltinFunction(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto function = sourceFunction(ast);
  auto lit = control()->stringLiteralFromValue(
      function ? to_string(function->name()) : std::string{});
  return ConstValue{lit};
}

auto ASTInterpreter::evaluateBuiltinComplex(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto complexType = unqualified_cast<ComplexType>(ast->type);
  if (!complexType) return std::nullopt;

  std::vector<ConstValue> parts;
  for (auto argument : ListView{ast->expressionList}) {
    auto value = expression(argument);
    if (!value) return std::nullopt;
    auto converted = toArithmeticType(*value, complexType->elementType());
    if (!converted) return std::nullopt;
    parts.push_back(*converted);
  }

  if (parts.size() != 2) return std::nullopt;

  return ConstValue{std::make_shared<ConstComplex>(parts[0], parts[1])};
}

auto ASTInterpreter::evaluateBuiltinHugeVal(CallExpressionAST* /*ast*/)
    -> std::optional<ConstValue> {
  return ConstValue{std::numeric_limits<double>::infinity()};
}

auto ASTInterpreter::evaluateBuiltinHugeValf(CallExpressionAST* /*ast*/)
    -> std::optional<ConstValue> {
  return ConstValue{std::numeric_limits<float>::infinity()};
}

auto ASTInterpreter::evaluateBuiltinHugeVall(CallExpressionAST* /*ast*/)
    -> std::optional<ConstValue> {
  return ConstValue{std::numeric_limits<long double>::infinity()};
}

auto ASTInterpreter::evaluateBuiltinNanPayload(CallExpressionAST* ast)
    -> std::optional<std::string> {
  auto argument = argumentAt(ast, 0);
  if (!argument) return std::nullopt;

  auto value = evaluate(argument);
  if (!value) return std::nullopt;

  auto literal = std::get_if<const StringLiteral*>(&*value);
  if (!literal || !*literal) return std::nullopt;

  return std::string((*literal)->stringValue());
}

auto ASTInterpreter::evaluateBuiltinNan(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto payload = evaluateBuiltinNanPayload(ast);
  if (!payload) return std::nullopt;
  return ConstValue{std::nan(payload->c_str())};
}

auto ASTInterpreter::evaluateBuiltinNanf(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto payload = evaluateBuiltinNanPayload(ast);
  if (!payload) return std::nullopt;
  return ConstValue{static_cast<double>(std::nanf(payload->c_str()))};
}

auto ASTInterpreter::evaluateBuiltinNanl(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  auto payload = evaluateBuiltinNanPayload(ast);
  if (!payload) return std::nullopt;
  return ConstValue{static_cast<double>(std::nanl(payload->c_str()))};
}

auto ASTInterpreter::evaluateBuiltinLockFree(CallExpressionAST* ast,
                                             bool alwaysLockFree,
                                             bool ignoresPointerOperand)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;

  auto sizeArgument = argumentAt(ast, 0);
  if (!sizeArgument) return std::nullopt;

  auto sizeValue = evaluate(sizeArgument);
  if (!sizeValue) return std::nullopt;

  auto size = toUInt(*sizeValue);
  if (!size) return std::nullopt;

  auto isLockFree = [&]() -> bool {
    if (*size == 0 || !std::has_single_bit(*size)) return false;

    const auto inlineWidth =
        control()->memoryLayout()->maxAtomicInlineWidth() / 8;
    if (*size > inlineWidth) return false;

    if (ignoresPointerOperand || *size == 1) return true;

    auto pointerArgument = argumentAt(ast, 1);
    if (!pointerArgument) return false;

    if (auto pointerValue = evaluate(pointerArgument)) {
      if (auto address = toUInt(*pointerValue)) {
        if (*address % *size == 0) return true;
      }
    }

    auto pointerType = strippedPointerType(pointerArgument);
    if (!pointerType) return false;

    auto pointeeType = pointerType->elementType();
    if (!unit_->typeTraits().is_complete(pointeeType)) return false;

    auto alignment = control()->memoryLayout()->alignmentOf(pointeeType);
    return alignment && *alignment >= *size;
  };

  if (isLockFree()) return ConstValue{std::intmax_t(1)};

  if (!alwaysLockFree) return std::nullopt;

  return ConstValue{std::intmax_t(0)};
}

auto ASTInterpreter::evaluateBuiltinBitCount(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;

  auto idExpr = ast_cast<IdExpressionAST>(ast->baseExpression);
  auto operation = bitCountOperation(resolveBuiltinFunctionKind(idExpr));
  if (!operation) return std::nullopt;

  auto argument = argumentAt(ast, 0);
  if (!argument) return std::nullopt;

  auto value = evaluate(argument);
  if (!value) return std::nullopt;

  auto representation = traits.integral_representation(argument->type);
  if (!representation) return std::nullopt;
  if (!ConstInt::isRepresentableWidth(representation->bits))
    return std::nullopt;

  auto stored = std::get_if<ConstInt>(&*value);
  if (!stored) return std::nullopt;

  auto operand = ConstInt::make(stored->toWideValue(), representation->bits,
                                /*isSigned=*/false);
  if (!operand) return std::nullopt;

  const auto width = representation->bits;

  if (operand->isZero()) {
    if (auto fallback = argumentAt(ast, 1)) {
      auto fallbackValue = evaluate(fallback);
      if (!fallbackValue) return std::nullopt;
      auto result = toInt(*fallbackValue);
      if (!result) return std::nullopt;
      return ConstValue{*result};
    }
  }

  switch (*operation) {
    case BitCountOperation::kCountLeadingZeros:
      if (operand->isZero()) return std::nullopt;
      return ConstValue{
          static_cast<std::intmax_t>(operand->countLeadingZeros())};

    case BitCountOperation::kCountTrailingZeros:
      if (operand->isZero()) return std::nullopt;
      return ConstValue{
          static_cast<std::intmax_t>(operand->countTrailingZeros())};

    case BitCountOperation::kPopulationCount:
      return ConstValue{static_cast<std::intmax_t>(operand->popcount())};

    case BitCountOperation::kParity:
      return ConstValue{static_cast<std::intmax_t>(operand->popcount() & 1)};

    case BitCountOperation::kFindFirstSet:
      if (operand->isZero()) return ConstValue{std::intmax_t(0)};
      return ConstValue{
          static_cast<std::intmax_t>(operand->countTrailingZeros() + 1)};

    case BitCountOperation::kCountLeadingRedundantSignBits: {
      auto signBit = ConstInt::make(1, width, /*isSigned=*/false);
      if (!signBit) return std::nullopt;
      auto shift = ConstInt::make(width - 1, width, /*isSigned=*/false);
      if (!shift) return std::nullopt;

      auto magnitude = *operand;
      if (!((*operand >> *shift) & *signBit).isZero()) {
        auto allOnes = ConstInt::make(-1, width, /*isSigned=*/false);
        if (!allOnes) return std::nullopt;
        magnitude = *operand ^ *allOnes;
      }

      return ConstValue{
          static_cast<std::intmax_t>(magnitude.countLeadingZeros() - 1)};
    }
  }

  return std::nullopt;
}

auto ASTInterpreter::evaluateBuiltinAtomicAlwaysLockFree(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateBuiltinLockFree(ast, /*alwaysLockFree=*/true,
                                 /*ignoresPointerOperand=*/false);
}

auto ASTInterpreter::evaluateBuiltinAtomicIsLockFree(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateBuiltinLockFree(ast, /*alwaysLockFree=*/false,
                                 /*ignoresPointerOperand=*/false);
}

auto ASTInterpreter::evaluateBuiltinC11AtomicIsLockFree(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return evaluateBuiltinLockFree(ast, /*alwaysLockFree=*/false,
                                 /*ignoresPointerOperand=*/true);
}

}  // namespace cxx
