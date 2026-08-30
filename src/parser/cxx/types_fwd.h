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

#pragma once

#include <cxx/cxx_fwd.h>

#include <string>
#include <utility>

namespace cxx {
class Name;

#define CXX_FOR_EACH_TYPE_KIND(V) \
  V(Void)                         \
  V(Nullptr)                      \
  V(DecltypeAuto)                 \
  V(Auto)                         \
  V(Bool)                         \
  V(SignedChar)                   \
  V(ShortInt)                     \
  V(Int)                          \
  V(LongInt)                      \
  V(LongLongInt)                  \
  V(Int128)                       \
  V(UnsignedChar)                 \
  V(UnsignedShortInt)             \
  V(UnsignedInt)                  \
  V(UnsignedLongInt)              \
  V(UnsignedLongLongInt)          \
  V(UnsignedInt128)               \
  V(Char)                         \
  V(Char8)                        \
  V(Char16)                       \
  V(Char32)                       \
  V(WideChar)                     \
  V(Float)                        \
  V(Double)                       \
  V(LongDouble)                   \
  V(Float16)                      \
  V(Qual)                         \
  V(BoundedArray)                 \
  V(UnboundedArray)               \
  V(Pointer)                      \
  V(LvalueReference)              \
  V(RvalueReference)              \
  V(Function)                     \
  V(Class)                        \
  V(Enum)                         \
  V(ScopedEnum)                   \
  V(MemberObjectPointer)          \
  V(MemberFunctionPointer)        \
  V(Namespace)                    \
  V(TypeParameter)                \
  V(TemplateTypeParameter)        \
  V(UnresolvedName)               \
  V(UnresolvedBoundedArray)       \
  V(UnresolvedUnderlying)         \
  V(UnresolvedBuiltin)            \
  V(OverloadSet)                  \
  V(BuiltinVaList)                \
  V(BuiltinMetaInfo)              \
  V(BitInt)                       \
  V(UnsignedBitInt)               \
  V(UnresolvedBitInt)

class Type;

#define PROCESS_TYPE(K) class K##Type;
CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
#undef PROCESS_TYPE

#define PROCESS_TYPE(K) k##K,
enum class TypeKind { CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE) };
#undef PROCESS_TYPE

enum class CvQualifiers {
  kNone = 0,
  kConst = 1,
  kVolatile = 2,
  kConstVolatile = kConst | kVolatile,
};

[[nodiscard]] constexpr auto operator|(CvQualifiers a, CvQualifiers b)
    -> CvQualifiers {
  return CvQualifiers(std::to_underlying(a) | std::to_underlying(b));
}

[[nodiscard]] constexpr auto operator&(CvQualifiers a, CvQualifiers b)
    -> CvQualifiers {
  return CvQualifiers(std::to_underlying(a) & std::to_underlying(b));
}

[[nodiscard]] constexpr auto operator~(CvQualifiers a) -> CvQualifiers {
  return CvQualifiers(~std::to_underlying(a) &
                      std::to_underlying(CvQualifiers::kConstVolatile));
}

constexpr auto operator|=(CvQualifiers& a, CvQualifiers b) -> CvQualifiers& {
  a = a | b;
  return a;
}

constexpr auto operator&=(CvQualifiers& a, CvQualifiers b) -> CvQualifiers& {
  a = a & b;
  return a;
}

[[nodiscard]] constexpr auto has_const(CvQualifiers cv) -> bool {
  return (cv & CvQualifiers::kConst) != CvQualifiers::kNone;
}

[[nodiscard]] constexpr auto has_volatile(CvQualifiers cv) -> bool {
  return (cv & CvQualifiers::kVolatile) != CvQualifiers::kNone;
}

[[nodiscard]] constexpr auto is_at_least_as_cv_qualified(CvQualifiers cv,
                                                         CvQualifiers other)
    -> bool {
  return (other & ~cv) == CvQualifiers::kNone;
}

[[nodiscard]] constexpr auto is_more_cv_qualified(CvQualifiers cv,
                                                  CvQualifiers other) -> bool {
  return cv != other && is_at_least_as_cv_qualified(cv, other);
}

enum class RefQualifier {
  kNone,
  kLvalue,
  kRvalue,
};

struct TypeParamInfo {
  int index = 0;
  int depth = 0;
  bool isPack = false;
};

struct TypePrintOptions {
  bool omitFunctionReturnType = false;
};

auto to_string(const Type* type, const std::string& id = "",
               TypePrintOptions options = {}) -> std::string;
auto to_string(const Type* type, const Name* name,
               TypePrintOptions options = {}) -> std::string;
}  // namespace cxx
