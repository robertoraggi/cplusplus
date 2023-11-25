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

#include <cxx/const_expression_evaluator.h>

// cxx
#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

namespace cxx {

auto ConstExpressionEvaluator::evaluate(ExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (!ast) return std::nullopt;
  return visit(*this, ast);
}

auto ConstExpressionEvaluator::control() const -> Control* {
  return parser.control();
}

auto ConstExpressionEvaluator::operator()(CharLiteralExpressionAST* ast)
    -> std::optional<ConstValue> {
  return ConstValue(ast->literal->charValue());
}

auto ConstExpressionEvaluator::operator()(BoolLiteralExpressionAST* ast)
    -> std::optional<ConstValue> {
  return ConstValue(ast->isTrue);
}

auto ConstExpressionEvaluator::operator()(IntLiteralExpressionAST* ast)
    -> std::optional<ConstValue> {
  return ConstValue(ast->literal->integerValue());
}

auto ConstExpressionEvaluator::operator()(FloatLiteralExpressionAST* ast)
    -> std::optional<ConstValue> {
  return ConstValue(ast->literal->floatValue());
}

auto ConstExpressionEvaluator::operator()(NullptrLiteralExpressionAST* ast)
    -> std::optional<ConstValue> {
  return ConstValue(std::uint64_t(0));
}

auto ConstExpressionEvaluator::operator()(StringLiteralExpressionAST* ast)
    -> std::optional<ConstValue> {
  return ConstValue(ast->literal);
}

auto ConstExpressionEvaluator::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> std::optional<ConstValue> {
  return ConstValue(ast->literal);
}

auto ConstExpressionEvaluator::operator()(ThisExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(NestedExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (ast->expression) {
    return evaluate(ast->expression);
  }
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(IdExpressionAST* ast)
    -> std::optional<ConstValue> {
  if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
    return enumerator->value();
  }

  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(LambdaExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(FoldExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(RightFoldExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(LeftFoldExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(RequiresExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(SubscriptExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(CallExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(TypeConstructionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(BracedTypeConstructionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(MemberExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(PostIncrExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(CppCastExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(BuiltinBitCastExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(TypeidExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(TypeidOfTypeExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(UnaryExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(AwaitExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(SizeofExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(SizeofTypeExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(SizeofPackExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(AlignofTypeExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(AlignofExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(NoexceptExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(NewExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(DeleteExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(CastExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(ImplicitCastExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(BinaryExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(ConditionalExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(YieldExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(ThrowExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(AssignmentExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(PackExpansionExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(DesignatedInitializerClauseAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(TypeTraitsExpressionAST* ast)
    -> std::optional<ConstValue> {
  const Type* firstType = nullptr;
  const Type* secondType = nullptr;

  if (ast->typeIdList && ast->typeIdList->value) {
    firstType = ast->typeIdList->value->type;

    if (auto next = ast->typeIdList->next; next && next->value) {
      secondType = next->value->type;
    }
  }

  if (firstType) {
    switch (ast->typeTraits) {
      case BuiltinKind::T___IS_VOID: {
        return control()->is_void(firstType);
        break;
      }

      case BuiltinKind::T___IS_NULL_POINTER: {
        return control()->is_null_pointer(firstType);
        break;
      }

      case BuiltinKind::T___IS_INTEGRAL: {
        return control()->is_integral(firstType);
        break;
      }

      case BuiltinKind::T___IS_FLOATING_POINT: {
        return control()->is_floating_point(firstType);
        break;
      }

      case BuiltinKind::T___IS_ARRAY: {
        return control()->is_array(firstType);
        break;
      }

      case BuiltinKind::T___IS_ENUM: {
        return control()->is_enum(firstType);
        break;
      }

      case BuiltinKind::T___IS_SCOPED_ENUM: {
        return control()->is_scoped_enum(firstType);
        break;
      }

      case BuiltinKind::T___IS_UNION: {
        return control()->is_union(firstType);
        break;
      }

      case BuiltinKind::T___IS_CLASS: {
        return control()->is_class(firstType);
        break;
      }

      case BuiltinKind::T___IS_FUNCTION: {
        return control()->is_function(firstType);
        break;
      }

      case BuiltinKind::T___IS_POINTER: {
        return control()->is_pointer(firstType);
        break;
      }

      case BuiltinKind::T___IS_MEMBER_OBJECT_POINTER: {
        return control()->is_member_object_pointer(firstType);
        break;
      }

      case BuiltinKind::T___IS_MEMBER_FUNCTION_POINTER: {
        return control()->is_member_function_pointer(firstType);
        break;
      }

      case BuiltinKind::T___IS_LVALUE_REFERENCE: {
        return control()->is_lvalue_reference(firstType);
        break;
      }

      case BuiltinKind::T___IS_RVALUE_REFERENCE: {
        return control()->is_rvalue_reference(firstType);
        break;
      }

      case BuiltinKind::T___IS_FUNDAMENTAL: {
        return control()->is_fundamental(firstType);
        break;
      }

      case BuiltinKind::T___IS_ARITHMETIC: {
        return control()->is_arithmetic(firstType);
        break;
      }

      case BuiltinKind::T___IS_SCALAR: {
        return control()->is_scalar(firstType);
      }

      case BuiltinKind::T___IS_OBJECT: {
        return control()->is_object(firstType);
        break;
      }

      case BuiltinKind::T___IS_COMPOUND: {
        return control()->is_compound(firstType);
        break;
      }

      case BuiltinKind::T___IS_REFERENCE: {
        return control()->is_reference(firstType);
        break;
      }

      case BuiltinKind::T___IS_MEMBER_POINTER: {
        return control()->is_member_pointer(firstType);
        break;
      }

      case BuiltinKind::T___IS_BOUNDED_ARRAY: {
        return control()->is_bounded_array(firstType);
        break;
      }

      case BuiltinKind::T___IS_UNBOUNDED_ARRAY: {
        return control()->is_unbounded_array(firstType);
        break;
      }

      case BuiltinKind::T___IS_CONST: {
        return control()->is_const(firstType);
        break;
      }

      case BuiltinKind::T___IS_VOLATILE: {
        return control()->is_volatile(firstType);
        break;
      }

      case BuiltinKind::T___IS_SIGNED: {
        return control()->is_signed(firstType);
        break;
      }

      case BuiltinKind::T___IS_UNSIGNED: {
        return control()->is_unsigned(firstType);
        break;
      }

      case BuiltinKind::T___IS_SAME:
      case BuiltinKind::T___IS_SAME_AS: {
        if (!secondType) break;
        return control()->is_same(firstType, secondType);
        break;
      }

      default:
        break;
    }  // switch
  }

  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(ConditionExpressionAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(EqualInitializerAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(BracedInitListAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

auto ConstExpressionEvaluator::operator()(ParenInitializerAST* ast)
    -> std::optional<ConstValue> {
  return std::nullopt;
}

}  // namespace cxx