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

#pragma once

#include <string_view>

#define CXX_FOR_EACH_TYPE_KIND(V) \
  V(Invalid)                      \
  V(Nullptr)                      \
  V(Dependent)                    \
  V(DecltypeAuto)                 \
  V(Auto)                         \
  V(Void)                         \
  V(Bool)                         \
  V(Char)                         \
  V(SignedChar)                   \
  V(UnsignedChar)                 \
  V(Short)                        \
  V(UnsignedShort)                \
  V(Int)                          \
  V(UnsignedInt)                  \
  V(Long)                         \
  V(UnsignedLong)                 \
  V(Float)                        \
  V(Double)                       \
  V(Qual)                         \
  V(Pointer)                      \
  V(LValueReference)              \
  V(RValueReference)              \
  V(Array)                        \
  V(Function)                     \
  V(Class)                        \
  V(Namespace)                    \
  V(MemberPointer)                \
  V(Concept)                      \
  V(Enum)                         \
  V(Generic)                      \
  V(Pack)                         \
  V(ScopedEnum)

namespace cxx {

#define DECLARE_TYPE_KIND(kind) k##kind,

enum class TypeKind { CXX_FOR_EACH_TYPE_KIND(DECLARE_TYPE_KIND) };

#undef DECLARE_TYPE_KIND

enum class TemplateArgumentKind {
  kInvalid,
  kType,
  kLiteral,
};

class Type;
class TypeVisitor;
class ReferenceType;
class Parameter;

#define PROCESS_TYPE(ty) class ty##Type;
CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
#undef PROCESS_TYPE

auto equal_to(const Type* type, const Type* other) -> bool;

auto to_string(TypeKind kind) -> std::string_view;

}  // namespace cxx
