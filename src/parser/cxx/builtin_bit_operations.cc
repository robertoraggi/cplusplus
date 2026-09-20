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

#include <cxx/builtin_bit_operations.h>

namespace cxx {

auto bitCountOperation(BuiltinFunctionKind kind)
    -> std::optional<BitCountOperation> {
  switch (kind) {
    case BuiltinFunctionKind::T___BUILTIN_CLZ:
    case BuiltinFunctionKind::T___BUILTIN_CLZS:
    case BuiltinFunctionKind::T___BUILTIN_CLZL:
    case BuiltinFunctionKind::T___BUILTIN_CLZLL:
    case BuiltinFunctionKind::T___BUILTIN_CLZG:
      return BitCountOperation::kCountLeadingZeros;

    case BuiltinFunctionKind::T___BUILTIN_CTZ:
    case BuiltinFunctionKind::T___BUILTIN_CTZS:
    case BuiltinFunctionKind::T___BUILTIN_CTZL:
    case BuiltinFunctionKind::T___BUILTIN_CTZLL:
    case BuiltinFunctionKind::T___BUILTIN_CTZG:
      return BitCountOperation::kCountTrailingZeros;

    case BuiltinFunctionKind::T___BUILTIN_POPCOUNT:
    case BuiltinFunctionKind::T___BUILTIN_POPCOUNTL:
    case BuiltinFunctionKind::T___BUILTIN_POPCOUNTLL:
    case BuiltinFunctionKind::T___BUILTIN_POPCOUNTG:
      return BitCountOperation::kPopulationCount;

    case BuiltinFunctionKind::T___BUILTIN_PARITY:
    case BuiltinFunctionKind::T___BUILTIN_PARITYL:
    case BuiltinFunctionKind::T___BUILTIN_PARITYLL:
      return BitCountOperation::kParity;

    case BuiltinFunctionKind::T___BUILTIN_FFS:
    case BuiltinFunctionKind::T___BUILTIN_FFSL:
    case BuiltinFunctionKind::T___BUILTIN_FFSLL:
      return BitCountOperation::kFindFirstSet;

    case BuiltinFunctionKind::T___BUILTIN_CLRSB:
    case BuiltinFunctionKind::T___BUILTIN_CLRSBL:
    case BuiltinFunctionKind::T___BUILTIN_CLRSBLL:
      return BitCountOperation::kCountLeadingRedundantSignBits;

    default:
      return std::nullopt;
  }
}

}  // namespace cxx
