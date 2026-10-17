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

#include <cxx/builtin_arithmetic_overflow.h>

namespace cxx {

auto arithmeticOverflowOperation(BuiltinFunctionKind kind)
    -> std::optional<ArithmeticOverflowOperation> {
  switch (kind) {
    case BuiltinFunctionKind::T___BUILTIN_ADD_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SADD_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SADDL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SADDLL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_UADD_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_UADDL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_UADDLL_OVERFLOW:
      return ArithmeticOverflowOperation::kAdd;

    case BuiltinFunctionKind::T___BUILTIN_SUB_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SSUB_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SSUBL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SSUBLL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_USUB_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_USUBL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_USUBLL_OVERFLOW:
      return ArithmeticOverflowOperation::kSub;

    case BuiltinFunctionKind::T___BUILTIN_MUL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SMUL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SMULL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_SMULLL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_UMUL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_UMULL_OVERFLOW:
    case BuiltinFunctionKind::T___BUILTIN_UMULLL_OVERFLOW:
      return ArithmeticOverflowOperation::kMul;

    default:
      return std::nullopt;
  }
}

auto carryArithmeticOperation(BuiltinFunctionKind kind)
    -> std::optional<ArithmeticOverflowOperation> {
  switch (kind) {
    case BuiltinFunctionKind::T___BUILTIN_ADDCB:
    case BuiltinFunctionKind::T___BUILTIN_ADDCS:
    case BuiltinFunctionKind::T___BUILTIN_ADDC:
    case BuiltinFunctionKind::T___BUILTIN_ADDCL:
    case BuiltinFunctionKind::T___BUILTIN_ADDCLL:
      return ArithmeticOverflowOperation::kAdd;

    case BuiltinFunctionKind::T___BUILTIN_SUBCB:
    case BuiltinFunctionKind::T___BUILTIN_SUBCS:
    case BuiltinFunctionKind::T___BUILTIN_SUBC:
    case BuiltinFunctionKind::T___BUILTIN_SUBCL:
    case BuiltinFunctionKind::T___BUILTIN_SUBCLL:
      return ArithmeticOverflowOperation::kSub;

    default:
      return std::nullopt;
  }
}

auto applyArithmeticOverflowOperation(ArithmeticOverflowOperation operation,
                                      ConstInt::Wide lhs, ConstInt::Wide rhs)
    -> ConstInt::Wide {
  switch (operation) {
    case ArithmeticOverflowOperation::kAdd:
      return lhs + rhs;
    case ArithmeticOverflowOperation::kSub:
      return lhs - rhs;
    case ArithmeticOverflowOperation::kMul:
      return lhs * rhs;
  }
  return 0;
}

auto fitsIntegerWidth(ConstInt::Wide value, int width, bool isSigned) -> bool {
  if (isSigned) {
    const auto limit = ConstInt::Wide{1} << (width - 1);
    return value >= -limit && value < limit;
  }
  if (value < 0) return false;
  const auto limit = width >= ConstInt::maxWidth
                         ? ~ConstInt::UWide{0}
                         : (ConstInt::UWide{1} << width) - 1;
  return static_cast<ConstInt::UWide>(value) <= limit;
}

}  // namespace cxx
