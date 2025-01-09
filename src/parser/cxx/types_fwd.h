// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

namespace cxx {

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
  V(UnsignedChar)                 \
  V(UnsignedShortInt)             \
  V(UnsignedInt)                  \
  V(UnsignedLongInt)              \
  V(UnsignedLongLongInt)          \
  V(Char)                         \
  V(Char8)                        \
  V(Char16)                       \
  V(Char32)                       \
  V(WideChar)                     \
  V(Float)                        \
  V(Double)                       \
  V(LongDouble)                   \
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
  V(OverloadSet)                  \
  V(BuiltinVaList)

class Type;

#define PROCESS_TYPE(K) class K##Type;
CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
#undef PROCESS_TYPE

#define PROCESS_TYPE(K) k##K,
enum class TypeKind { CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE) };
#undef PROCESS_TYPE

enum class CvQualifiers {
  kNone,
  kConst,
  kVolatile,
  kConstVolatile,
};

enum class RefQualifier {
  kNone,
  kLvalue,
  kRvalue,
};

}  // namespace cxx
