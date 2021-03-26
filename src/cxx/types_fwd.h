// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cstdint>

namespace cxx {

class QualifiedType;

class TypeEnvironment;
class TypeVisitor;
class Type;

class UndefinedType;
class UnresolvedType;
class VoidType;
class NullptrType;
class BooleanType;
class CharacterType;
class IntegerType;
class FloatingPointType;
class EnumType;
class ScopedEnumType;
class PointerType;
class PointerToMemberType;
class ReferenceType;
class RValueReferenceType;
class ArrayType;
class UnboundArrayType;
class FunctionType;
class MemberFunctionType;
class NamespaceType;
class ClassType;
class TemplateType;
class TemplateArgumentType;
class ConceptType;

enum class CharacterKind {
  kChar8T,
  kChar16T,
  kChar32T,
  kWCharT,
};

enum class IntegerKind {
  kChar,
  kShort,
  kInt,
  kInt64,
  kInt128,
  kLong,
  kLongLong,
};

enum class FloatingPointKind {
  kFloat,
  kFloat128,
  kDouble,
  kLongDouble,
};

enum class Qualifiers : std::uint8_t {
  kNone = 0,
  kConst = 1,
  kVolatile = 2,
  kRestrict = 4,
};

constexpr Qualifiers operator&(Qualifiers lhs, Qualifiers rhs) noexcept {
  return static_cast<Qualifiers>(static_cast<std::uint8_t>(lhs) &
                                 static_cast<std::uint8_t>(rhs));
}

constexpr Qualifiers operator|(Qualifiers lhs, Qualifiers rhs) noexcept {
  return static_cast<Qualifiers>(static_cast<std::uint8_t>(lhs) |
                                 static_cast<std::uint8_t>(rhs));
}

constexpr Qualifiers operator^(Qualifiers lhs, Qualifiers rhs) noexcept {
  return static_cast<Qualifiers>(static_cast<std::uint8_t>(lhs) ^
                                 static_cast<std::uint8_t>(rhs));
}

constexpr Qualifiers operator~(Qualifiers lhs) noexcept {
  return static_cast<Qualifiers>(~static_cast<std::uint8_t>(lhs));
}

inline Qualifiers& operator&=(Qualifiers& lhs, Qualifiers rhs) noexcept {
  return lhs = lhs & rhs;
}

inline Qualifiers& operator|=(Qualifiers& lhs, Qualifiers rhs) noexcept {
  return lhs = lhs | rhs;
}

inline Qualifiers& operator^=(Qualifiers& lhs, Qualifiers rhs) noexcept {
  return lhs = lhs ^ rhs;
}

}  // namespace cxx
