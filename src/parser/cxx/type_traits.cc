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

#include <cxx/access_control.h>
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/standard_conversion.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <array>
#include <cmath>
#include <optional>
#include <unordered_set>

namespace cxx {
namespace {
struct QualificationComponent {
  enum class Kind {
    kPointer,
    kMemberPointer,
    kBoundedArray,
    kUnboundedArray,
  };

  Kind kind = Kind::kPointer;
  const Type* classType = nullptr;
  std::size_t size = 0;

  [[nodiscard]] auto isArray() const -> bool {
    return kind == Kind::kBoundedArray || kind == Kind::kUnboundedArray;
  }
};

struct QualificationStep {
  std::optional<QualificationComponent> component;
  const Type* next = nullptr;
};

struct DecomposeQualificationComponent {
  auto operator()(const PointerType* type) const -> QualificationStep {
    return {QualificationComponent{QualificationComponent::Kind::kPointer},
            type->elementType()};
  }

  auto operator()(const MemberObjectPointerType* type) const
      -> QualificationStep {
    return {QualificationComponent{QualificationComponent::Kind::kMemberPointer,
                                   type->classType()},
            type->elementType()};
  }

  auto operator()(const MemberFunctionPointerType* type) const
      -> QualificationStep {
    return {QualificationComponent{QualificationComponent::Kind::kMemberPointer,
                                   type->classType()},
            type->functionType()};
  }

  auto operator()(const BoundedArrayType* type) const -> QualificationStep {
    return {QualificationComponent{QualificationComponent::Kind::kBoundedArray,
                                   nullptr, type->size()},
            type->elementType()};
  }

  auto operator()(const UnboundedArrayType* type) const -> QualificationStep {
    return {
        QualificationComponent{QualificationComponent::Kind::kUnboundedArray},
        type->elementType()};
  }

  auto operator()(const Type*) const -> QualificationStep { return {}; }
};

[[nodiscard]] auto decomposeQualificationComponent(const Type* type)
    -> QualificationStep {
  if (!type) return {};
  return visit(DecomposeQualificationComponent{}, unqualified_type(type));
}

struct QualificationDecomposition {
  std::vector<CvQualifiers> cv;
  std::vector<QualificationComponent> components;
  const Type* terminal = nullptr;
};

struct QualificationDecompositionPair {
  QualificationDecomposition left;
  QualificationDecomposition right;
};

[[nodiscard]] auto componentsMatch(const TypeTraits& traits,
                                   const QualificationComponent& lhs,
                                   const QualificationComponent& rhs) -> bool {
  if (lhs.isArray() && rhs.isArray()) {
    if (lhs.kind != rhs.kind) return true;
    return lhs.size == rhs.size;
  }
  if (lhs.kind != rhs.kind) return false;
  return traits.is_same(lhs.classType, rhs.classType);
}

struct IsVoid {
  auto operator()(const VoidType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsNullPointer {
  auto operator()(const NullptrType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsIntegral {
  auto operator()(const BoolType*) const -> bool { return true; }
  auto operator()(const SignedCharType*) const -> bool { return true; }
  auto operator()(const ShortIntType*) const -> bool { return true; }
  auto operator()(const IntType*) const -> bool { return true; }
  auto operator()(const LongIntType*) const -> bool { return true; }
  auto operator()(const LongLongIntType*) const -> bool { return true; }
  auto operator()(const UnsignedCharType*) const -> bool { return true; }
  auto operator()(const UnsignedShortIntType*) const -> bool { return true; }
  auto operator()(const UnsignedIntType*) const -> bool { return true; }
  auto operator()(const UnsignedLongIntType*) const -> bool { return true; }
  auto operator()(const UnsignedLongLongIntType*) const -> bool { return true; }
  auto operator()(const CharType*) const -> bool { return true; }
  auto operator()(const Char8Type*) const -> bool { return true; }
  auto operator()(const Char16Type*) const -> bool { return true; }
  auto operator()(const Char32Type*) const -> bool { return true; }
  auto operator()(const WideCharType*) const -> bool { return true; }
  auto operator()(const Int128Type*) const -> bool { return true; }
  auto operator()(const UnsignedInt128Type*) const -> bool { return true; }
  auto operator()(const BitIntType*) const -> bool { return true; }
  auto operator()(const UnsignedBitIntType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsFloatingPoint {
  auto operator()(const FloatType*) const -> bool { return true; }
  auto operator()(const DoubleType*) const -> bool { return true; }
  auto operator()(const LongDoubleType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsSigned {
  bool wideCharIsSigned = true;

  auto operator()(const WideCharType*) const -> bool {
    return wideCharIsSigned;
  }

  auto operator()(const SignedCharType*) const -> bool { return true; }
  auto operator()(const ShortIntType*) const -> bool { return true; }
  auto operator()(const IntType*) const -> bool { return true; }
  auto operator()(const LongIntType*) const -> bool { return true; }
  auto operator()(const LongLongIntType*) const -> bool { return true; }
  auto operator()(const Int128Type*) const -> bool { return true; }
  auto operator()(const CharType*) const -> bool { return true; }
  auto operator()(const FloatType*) const -> bool { return true; }
  auto operator()(const DoubleType*) const -> bool { return true; }
  auto operator()(const LongDoubleType*) const -> bool { return true; }
  auto operator()(const Float16Type*) const -> bool { return true; }
  auto operator()(const BitIntType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsUnsigned {
  bool wideCharIsSigned = true;

  auto operator()(const WideCharType*) const -> bool {
    return !wideCharIsSigned;
  }

  auto operator()(const BoolType*) const -> bool { return true; }
  auto operator()(const UnsignedCharType*) const -> bool { return true; }
  auto operator()(const UnsignedShortIntType*) const -> bool { return true; }
  auto operator()(const UnsignedIntType*) const -> bool { return true; }
  auto operator()(const UnsignedLongIntType*) const -> bool { return true; }
  auto operator()(const UnsignedLongLongIntType*) const -> bool { return true; }
  auto operator()(const Char8Type*) const -> bool { return true; }
  auto operator()(const Char16Type*) const -> bool { return true; }
  auto operator()(const Char32Type*) const -> bool { return true; }
  auto operator()(const UnsignedInt128Type*) const -> bool { return true; }
  auto operator()(const UnsignedBitIntType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsArray {
  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const UnboundedArrayType*) const -> bool { return true; }
  auto operator()(const BoundedArrayType*) const -> bool { return true; }
  auto operator()(const UnresolvedBoundedArrayType*) const -> bool {
    return true;
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsEnum {
  auto operator()(const EnumType*) const -> bool { return true; }
  auto operator()(const ScopedEnumType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct UnderlyingType {
  auto operator()(const EnumType* type) const -> const Type* {
    auto underlyingType = type->underlyingType();
    return underlyingType ? underlyingType : type;
  }

  auto operator()(const ScopedEnumType* type) const -> const Type* {
    auto underlyingType = type->underlyingType();
    return underlyingType ? underlyingType : type;
  }

  auto operator()(const Type* type) const -> const Type* { return type; }
};

struct IsScopedEnum {
  auto operator()(const ScopedEnumType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsClass {
  auto operator()(const ClassType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsUnion {
  auto operator()(const ClassType* classType) const -> bool {
    return classType->isUnion();
  }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsFunction {
  auto operator()(const FunctionType*) const -> bool { return true; }
  auto operator()(const Type*) const -> bool { return false; }
};

struct IsPointer {
  auto operator()(const PointerType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsMemberObjectPointer {
  auto operator()(const MemberObjectPointerType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsMemberFunctionPointer {
  auto operator()(const MemberFunctionPointerType*) const -> bool {
    return true;
  }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsBoundedArray {
  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const BoundedArrayType*) const -> bool { return true; }
  auto operator()(const UnresolvedBoundedArrayType*) const -> bool {
    return true;
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsUnboundedArray {
  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const UnboundedArrayType*) const -> bool { return true; }
  auto operator()(const Type*) const -> bool { return false; }
};

struct IsConst {
  auto operator()(const QualType* type) const -> bool {
    return type->isConst();
  }

  auto operator()(const BoundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const UnboundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsVolatile {
  auto operator()(const QualType* type) const -> bool {
    return type->isVolatile();
  }

  auto operator()(const BoundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const UnboundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsLvalueReference {
  auto operator()(const LvalueReferenceType*) const -> bool { return true; }
  auto operator()(const Type*) const -> bool { return false; }
};

struct IsRvalueReference {
  auto operator()(const RvalueReferenceType*) const -> bool { return true; }
  auto operator()(const Type*) const -> bool { return false; }
};

struct IsReference {
  auto operator()(const LvalueReferenceType*) const -> bool { return true; }
  auto operator()(const RvalueReferenceType*) const -> bool { return true; }
  auto operator()(const Type*) const -> bool { return false; }
};

struct IsComplete {
  auto operator()(const VoidType*) const -> bool { return false; }

  auto operator()(const ClassType* type) const -> bool {
    return type->isComplete();
  }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return true; }
};

struct RemoveReference {
  auto operator()(const LvalueReferenceType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const RvalueReferenceType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(auto type) const -> const Type* { return type; }
};

struct AddLvalueReference {
  const TypeTraits& typeTraits;

  [[nodiscard]] auto control() const -> Control* {
    return typeTraits.control();
  }

  auto operator()(const VoidType* type) const -> const Type* { return type; }

  auto operator()(const QualType* type) const -> const Type* {
    if (typeTraits.is_void(type->elementType())) return type;
    return control()->getLvalueReferenceType(type);
  }

  auto operator()(const LvalueReferenceType* type) const -> const Type* {
    return type;
  }

  auto operator()(const RvalueReferenceType* type) const -> const Type* {
    return control()->getLvalueReferenceType(type->elementType());
  }

  auto operator()(const FunctionType* type) const -> const Type* {
    if (type->cvQualifiers() != CvQualifiers::kNone) return type;
    if (type->refQualifier() != RefQualifier::kNone) return type;
    return control()->getLvalueReferenceType(type);
  }

  auto operator()(auto type) const -> const Type* {
    return control()->getLvalueReferenceType(type);
  }
};

struct AddRvalueReference {
  const TypeTraits& typeTraits;

  [[nodiscard]] auto control() const -> Control* {
    return typeTraits.control();
  }

  auto operator()(const VoidType* type) const -> const Type* { return type; }

  auto operator()(const QualType* type) const -> const Type* {
    if (typeTraits.is_void(type->elementType())) return type;
    return control()->getRvalueReferenceType(type);
  }

  auto operator()(const LvalueReferenceType* type) const -> const Type* {
    return type;
  }

  auto operator()(const RvalueReferenceType* type) const -> const Type* {
    return type;
  }

  auto operator()(const FunctionType* type) const -> const Type* {
    if (type->cvQualifiers() != CvQualifiers::kNone) return type;
    if (type->refQualifier() != RefQualifier::kNone) return type;
    return control()->getRvalueReferenceType(type);
  }

  auto operator()(auto type) const -> const Type* {
    return control()->getRvalueReferenceType(type);
  }
};

struct AddCvQualifiers {
  const TypeTraits& typeTraits;
  CvQualifiers qualifiers;

  [[nodiscard]] auto control() const -> Control* {
    return typeTraits.control();
  }

  auto operator()(const BoundedArrayType* type) const -> const Type* {
    auto elementType = visit(*this, type->elementType());
    return control()->getBoundedArrayType(elementType, type->size());
  }

  auto operator()(const UnboundedArrayType* type) const -> const Type* {
    auto elementType = visit(*this, type->elementType());
    return control()->getUnboundedArrayType(elementType);
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const -> const Type* {
    auto elementType = visit(*this, type->elementType());
    return control()->getUnresolvedBoundedArrayType(type->translationUnit(),
                                                    elementType, type->size());
  }

  auto operator()(const FunctionType* type) const -> const Type* {
    return type;
  }

  auto operator()(const LvalueReferenceType* type) const -> const Type* {
    return type;
  }

  auto operator()(const RvalueReferenceType* type) const -> const Type* {
    return type;
  }

  auto operator()(auto type) const -> const Type* {
    return control()->getQualType(type, qualifiers);
  }
};

struct RemoveExtent {
  auto operator()(const BoundedArrayType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const UnboundedArrayType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(auto type) const -> const Type* { return type; }
};

struct GetElementType {
  auto operator()(const BoundedArrayType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const UnboundedArrayType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const PointerType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const LvalueReferenceType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const RvalueReferenceType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const QualType* type) const -> const Type* {
    return visit(*this, type->elementType());
  }

  auto operator()(auto) const -> const Type* { return nullptr; }
};

struct RemoveQualifiers {
  const TypeTraits& typeTraits;
  CvQualifiers qualifiers;

  [[nodiscard]] auto control() const -> Control* {
    return typeTraits.control();
  }

  auto operator()(const QualType* type) const -> const Type* {
    auto remaining = type->cvQualifiers() & ~qualifiers;
    if (remaining == type->cvQualifiers()) return type;
    if (remaining == CvQualifiers::kNone) return type->elementType();
    return control()->getQualType(type->elementType(), remaining);
  }

  auto operator()(const BoundedArrayType* type) const -> const Type* {
    auto elementType = visit(*this, type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getBoundedArrayType(elementType, type->size());
  }

  auto operator()(const UnboundedArrayType* type) const -> const Type* {
    auto elementType = visit(*this, type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getUnboundedArrayType(elementType);
  }

  auto operator()(const UnresolvedBoundedArrayType* type) const -> const Type* {
    auto elementType = visit(*this, type->elementType());
    if (elementType == type->elementType()) return type;
    return control()->getUnresolvedBoundedArrayType(type->translationUnit(),
                                                    elementType, type->size());
  }

  auto operator()(auto type) const -> const Type* { return type; }
};

struct AddPointer {
  const TypeTraits& typeTraits;

  [[nodiscard]] auto control() const -> Control* {
    return typeTraits.control();
  }

  auto operator()(const LvalueReferenceType* type) const -> const Type* {
    return control()->getPointerType(type->elementType());
  }

  auto operator()(const RvalueReferenceType* type) const -> const Type* {
    return control()->getPointerType(type->elementType());
  }

  auto operator()(const FunctionType* type) const -> const Type* {
    if (type->refQualifier() != RefQualifier::kNone) return type;
    if (type->cvQualifiers() != CvQualifiers::kNone) return type;
    return control()->getPointerType(type);
  }

  auto operator()(auto type) const -> const Type* {
    return control()->getPointerType(type);
  }
};

struct IsSameVisitor {
  const TypeTraits& typeTraits;

  auto operator()(const BuiltinVaListType*, const BuiltinVaListType*) const
      -> bool {
    return true;
  }

  auto operator()(const VoidType*, const VoidType*) const -> bool {
    return true;
  }

  auto operator()(const NullptrType*, const NullptrType*) const -> bool {
    return true;
  }

  auto operator()(const DecltypeAutoType*, const DecltypeAutoType*) const
      -> bool {
    return true;
  }

  auto operator()(const AutoType*, const AutoType*) const -> bool {
    return true;
  }

  auto operator()(const BoolType*, const BoolType*) const -> bool {
    return true;
  }

  auto operator()(const SignedCharType*, const SignedCharType*) const -> bool {
    return true;
  }

  auto operator()(const ShortIntType*, const ShortIntType*) const -> bool {
    return true;
  }

  auto operator()(const IntType*, const IntType*) const -> bool { return true; }

  auto operator()(const LongIntType*, const LongIntType*) const -> bool {
    return true;
  }

  auto operator()(const LongLongIntType*, const LongLongIntType*) const
      -> bool {
    return true;
  }

  auto operator()(const Int128Type*, const Int128Type*) const -> bool {
    return true;
  }

  auto operator()(const UnsignedCharType*, const UnsignedCharType*) const
      -> bool {
    return true;
  }

  auto operator()(const UnsignedShortIntType*,
                  const UnsignedShortIntType*) const -> bool {
    return true;
  }

  auto operator()(const UnsignedIntType*, const UnsignedIntType*) const
      -> bool {
    return true;
  }

  auto operator()(const UnsignedLongIntType*, const UnsignedLongIntType*) const
      -> bool {
    return true;
  }

  auto operator()(const UnsignedLongLongIntType*,
                  const UnsignedLongLongIntType*) const -> bool {
    return true;
  }

  auto operator()(const UnsignedInt128Type*, const UnsignedInt128Type*) const
      -> bool {
    return true;
  }

  auto operator()(const CharType*, const CharType*) const -> bool {
    return true;
  }

  auto operator()(const Char8Type*, const Char8Type*) const -> bool {
    return true;
  }

  auto operator()(const Char16Type*, const Char16Type*) const -> bool {
    return true;
  }

  auto operator()(const Char32Type*, const Char32Type*) const -> bool {
    return true;
  }

  auto operator()(const WideCharType*, const WideCharType*) const -> bool {
    return true;
  }

  auto operator()(const FloatType*, const FloatType*) const -> bool {
    return true;
  }

  auto operator()(const DoubleType*, const DoubleType*) const -> bool {
    return true;
  }

  auto operator()(const LongDoubleType*, const LongDoubleType*) const -> bool {
    return true;
  }

  auto operator()(const Float16Type*, const Float16Type*) const -> bool {
    return true;
  }

  auto operator()(const QualType* type, const QualType* otherType) const
      -> bool {
    if (type->cvQualifiers() != otherType->cvQualifiers()) return false;
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const BoundedArrayType* type,
                  const BoundedArrayType* otherType) const -> bool {
    if (type->size() != otherType->size()) return false;
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const UnboundedArrayType* type,
                  const UnboundedArrayType* otherType) const -> bool {
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const PointerType* type, const PointerType* otherType) const
      -> bool {
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const LvalueReferenceType* type,
                  const LvalueReferenceType* otherType) const -> bool {
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const RvalueReferenceType* type,
                  const RvalueReferenceType* otherType) const -> bool {
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const FunctionType* type, const FunctionType* otherType) const
      -> bool {
    if (type->isVariadic() != otherType->isVariadic()) return false;
    if (type->refQualifier() != otherType->refQualifier()) return false;
    if (type->cvQualifiers() != otherType->cvQualifiers()) return false;
    if (type->isNoexcept() != otherType->isNoexcept()) return false;
    if (type->parameterTypes().size() != otherType->parameterTypes().size())
      return false;
    if (!typeTraits.is_same(type->returnType(), otherType->returnType()))
      return false;
    for (std::size_t i = 0; i < type->parameterTypes().size(); ++i) {
      if (!typeTraits.is_same(type->parameterTypes()[i],
                              otherType->parameterTypes()[i]))
        return false;
    }
    return true;
  }

  auto operator()(const ClassType* type, const ClassType* otherType) const
      -> bool {
    return type->symbol() == otherType->symbol();
  }

  auto operator()(const EnumType* type, const EnumType* otherType) const
      -> bool {
    return type->symbol() == otherType->symbol();
  }

  auto operator()(const ScopedEnumType* type,
                  const ScopedEnumType* otherType) const -> bool {
    return type->symbol() == otherType->symbol();
  }

  auto operator()(const MemberObjectPointerType* type,
                  const MemberObjectPointerType* otherType) const -> bool {
    if (!typeTraits.is_same(type->classType(), otherType->classType()))
      return false;
    if (!typeTraits.is_same(type->elementType(), otherType->elementType()))
      return false;
    return true;
  }

  auto operator()(const MemberFunctionPointerType* type,
                  const MemberFunctionPointerType* otherType) const -> bool {
    if (!typeTraits.is_same(type->classType(), otherType->classType()))
      return false;
    if (!typeTraits.is_same(type->functionType(), otherType->functionType()))
      return false;
    return true;
  }

  auto operator()(const NamespaceType* type,
                  const NamespaceType* otherType) const -> bool {
    return type->symbol() == otherType->symbol();
  }

  auto operator()(const TypeParameterType* type,
                  const TypeParameterType* otherType) const -> bool {
    return type->index() == otherType->index() &&
           type->depth() == otherType->depth() &&
           type->isParameterPack() == otherType->isParameterPack();
  }

  auto operator()(const TemplateTypeParameterType* type,
                  const TemplateTypeParameterType* otherType) const -> bool {
    if (type->index() != otherType->index()) return false;
    if (type->depth() != otherType->depth()) return false;
    if (type->isParameterPack() != otherType->isParameterPack()) return false;
    if (type->templateParameters().size() !=
        otherType->templateParameters().size())
      return false;
    for (std::size_t i = 0; i < type->templateParameters().size(); ++i) {
      if (!typeTraits.is_same(type->templateParameters()[i],
                              otherType->templateParameters()[i]))
        return false;
    }
    return true;
  }

  auto operator()(const UnresolvedNameType* type,
                  const UnresolvedNameType* otherType) const -> bool {
    return type == otherType;
  }

  auto operator()(const UnresolvedBoundedArrayType* type,
                  const UnresolvedBoundedArrayType* otherType) const -> bool {
    return type == otherType;
  }

  auto operator()(const UnresolvedUnderlyingType* type,
                  const UnresolvedUnderlyingType* otherType) const -> bool {
    return type == otherType;
  }

  auto operator()(const UnresolvedBuiltinType* type,
                  const UnresolvedBuiltinType* otherType) const -> bool {
    return type == otherType;
  }

  auto operator()(const OverloadSetType* type,
                  const OverloadSetType* otherType) const -> bool {
    return type->symbol() == otherType->symbol();
  }

  auto operator()(const BuiltinMetaInfoType*, const BuiltinMetaInfoType*) const
      -> bool {
    return true;
  }

  auto operator()(const BitIntType* type, const BitIntType* otherType) const
      -> bool {
    return type->numBits() == otherType->numBits();
  }

  auto operator()(const UnsignedBitIntType* type,
                  const UnsignedBitIntType* otherType) const -> bool {
    return type->numBits() == otherType->numBits();
  }

  auto operator()(const UnresolvedBitIntType* type,
                  const UnresolvedBitIntType* otherType) const -> bool {
    return type == otherType;
  }
};

auto isUserProvided(FunctionSymbol* fn) -> bool {
  return fn && !fn->isDefaulted() && !fn->isDeleted();
}

auto is_trivially_copyable_class(TypeTraits& traits, ClassSymbol* cls) -> bool {
  if (!cls || !cls->isComplete()) return false;

  auto dtor = cls->destructor();
  if (dtor && dtor->isDeleted()) return false;
  if (isUserProvided(dtor)) return false;
  if (dtor && dtor->isVirtual()) return false;

  if (isUserProvided(cls->copyConstructor())) return false;
  if (isUserProvided(cls->moveConstructor())) return false;
  if (isUserProvided(cls->copyAssignmentOperator())) return false;
  if (isUserProvided(cls->moveAssignmentOperator())) return false;

  if (cls->isPolymorphic()) return false;
  if (cls->hasVirtualBaseClasses()) return false;

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (!is_trivially_copyable_class(traits, baseClass)) return false;
  }

  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldType = traits.remove_all_extents(traits.remove_cv(field->type()));
    if (auto ct = type_cast<ClassType>(fieldType)) {
      if (!is_trivially_copyable_class(traits, ct->symbol())) return false;
    }
  }

  return true;
}

auto is_trivially_destructible_class(TypeTraits& traits, ClassSymbol* cls)
    -> bool {
  if (!cls || !cls->isComplete()) return false;

  auto dtor = cls->destructor();
  if (dtor && dtor->isDeleted()) return false;
  if (isUserProvided(dtor)) return false;
  if (dtor && dtor->isVirtual()) return false;

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (!is_trivially_destructible_class(traits, baseClass)) return false;
  }

  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldType = traits.remove_all_extents(field->type());
    if (!traits.has_trivial_destructor(fieldType)) return false;
  }

  return true;
}

enum class TrivialConstructorKind { kDefault, kCopy, kMove };

auto constructorKind(ClassSymbol* cls, FunctionSymbol* constructor)
    -> std::optional<TrivialConstructorKind> {
  if (constructor == cls->defaultConstructor())
    return TrivialConstructorKind::kDefault;
  if (constructor == cls->copyConstructor())
    return TrivialConstructorKind::kCopy;
  if (constructor == cls->moveConstructor())
    return TrivialConstructorKind::kMove;
  return std::nullopt;
}

auto constructorFor(ClassSymbol* cls, TrivialConstructorKind kind)
    -> FunctionSymbol* {
  if (kind == TrivialConstructorKind::kDefault)
    return cls->defaultConstructor();
  if (kind == TrivialConstructorKind::kCopy) return cls->copyConstructor();
  auto move = cls->moveConstructor();
  if (move) return move;
  return cls->copyConstructor();
}

auto has_trivial_constructor(TypeTraits& traits, ClassSymbol* cls,
                             TrivialConstructorKind kind) -> bool {
  if (!cls || !cls->isComplete()) return false;
  auto constructor = constructorFor(cls, kind);
  if (!constructor || constructor->isDeleted()) return false;
  if (isUserProvided(constructor)) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  if (cls->isUnion()) return true;

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    if (!has_trivial_constructor(traits, baseClass, kind)) return false;
  }

  for (auto field : cls->members() | views::non_static_fields) {
    if (kind == TrivialConstructorKind::kDefault && field->initializer())
      return false;
    auto fieldType = traits.remove_all_extents(traits.remove_cv(field->type()));
    auto classType = type_cast<ClassType>(fieldType);
    if (!classType) continue;
    auto fieldClass = classType->definition();
    if (!has_trivial_constructor(traits, fieldClass, kind)) return false;
  }

  return true;
}

enum class TrivialAssignmentKind { kCopy, kMove };

auto assignmentFor(ClassSymbol* cls, TrivialAssignmentKind kind)
    -> FunctionSymbol* {
  if (kind == TrivialAssignmentKind::kCopy)
    return cls->copyAssignmentOperator();
  auto move = cls->moveAssignmentOperator();
  if (move) return move;
  return cls->copyAssignmentOperator();
}

auto has_trivial_assignment(TypeTraits& traits, ClassSymbol* cls,
                            TrivialAssignmentKind kind) -> bool {
  if (!cls || !cls->isComplete()) return false;
  auto assignment = assignmentFor(cls, kind);
  if (!assignment || assignment->isDeleted()) return false;
  if (isUserProvided(assignment)) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  if (cls->isUnion()) return true;

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    if (!has_trivial_assignment(traits, baseClass, kind)) return false;
  }

  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldType = traits.remove_all_extents(traits.remove_cv(field->type()));
    auto classType = type_cast<ClassType>(fieldType);
    if (!classType) continue;
    auto fieldClass = classType->definition();
    if (!has_trivial_assignment(traits, fieldClass, kind)) return false;
  }

  return true;
}

auto has_non_static_data_members(ClassSymbol* cls) -> bool {
  if (!cls) return false;
  for (auto member : cls->members() | views::non_static_fields) {
    (void)member;
    return true;
  }
  return false;
}

auto has_data_members_in_hierarchy(ClassSymbol* cls) -> bool {
  if (has_non_static_data_members(cls)) return true;
  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    if (has_data_members_in_hierarchy(baseClass)) return true;
  }
  return false;
}

auto has_unique_base_subobject_types(ClassSymbol* cls,
                                     std::unordered_set<ClassSymbol*>& seen)
    -> bool {
  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    if (!seen.insert(baseClass).second) return false;
    if (!has_unique_base_subobject_types(baseClass, seen)) return false;
  }
  return true;
}

void collect_standard_layout_member_types(
    TypeTraits& traits, const Type* type,
    std::unordered_set<ClassSymbol*>& memberTypes,
    std::unordered_set<ClassSymbol*>& expanded);

void add_standard_layout_member_type(
    TypeTraits& traits, const Type* type,
    std::unordered_set<ClassSymbol*>& memberTypes,
    std::unordered_set<ClassSymbol*>& expanded) {
  auto unqual = traits.remove_cv(type);
  if (traits.is_array(unqual)) {
    add_standard_layout_member_type(traits, traits.remove_extent(unqual),
                                    memberTypes, expanded);
    return;
  }

  auto classType = type_cast<ClassType>(unqual);
  if (!classType) return;
  auto cls = classType->definition();
  if (!cls) return;
  cls = cls->resolvedDefinition();
  memberTypes.insert(cls);
  if (!expanded.insert(cls).second) return;
  collect_standard_layout_member_types(traits, cls->type(), memberTypes,
                                       expanded);
}

void collect_standard_layout_member_types(
    TypeTraits& traits, const Type* type,
    std::unordered_set<ClassSymbol*>& memberTypes,
    std::unordered_set<ClassSymbol*>& expanded) {
  auto unqual = traits.remove_cv(type);
  if (traits.is_array(unqual)) {
    add_standard_layout_member_type(traits, traits.remove_extent(unqual),
                                    memberTypes, expanded);
    return;
  }

  auto classType = type_cast<ClassType>(unqual);
  if (!classType) return;
  auto cls = classType->definition();
  if (!cls) return;
  cls = cls->resolvedDefinition();

  auto first = true;
  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldType = traits.remove_cv(field->type());
    auto zeroSize = field->isNoUniqueAddress();
    if (zeroSize) zeroSize = traits.is_empty(fieldType);
    if (cls->isUnion() || first || zeroSize) {
      add_standard_layout_member_type(traits, fieldType, memberTypes, expanded);
    }
    first = false;
  }
}
}  // namespace

TypeTraits::TypeTraits(TranslationUnit* unit) : unit_(unit) {}

auto TypeTraits::control() const -> Control* { return unit_->control(); }

auto TypeTraits::is_void(const Type* type) const -> bool {
  return type && visit(IsVoid{}, type);
}

auto TypeTraits::is_null_pointer(const Type* type) const -> bool {
  return type && visit(IsNullPointer{}, type);
}

auto TypeTraits::is_integral(const Type* type) const -> bool {
  return type && visit(IsIntegral{}, type);
}

auto TypeTraits::is_floating_point(const Type* type) const -> bool {
  return type && visit(IsFloatingPoint{}, type);
}

auto TypeTraits::is_array(const Type* type) const -> bool {
  return type && visit(IsArray{}, type);
}

auto TypeTraits::is_enum(const Type* type) const -> bool {
  return type && visit(IsEnum{}, type);
}

auto TypeTraits::is_union(const Type* type) const -> bool {
  return type && visit(IsUnion{}, type);
}

auto TypeTraits::is_class(const Type* type) const -> bool {
  return type && visit(IsClass{}, type);
}

auto TypeTraits::is_function(const Type* type) const -> bool {
  return type && visit(IsFunction{}, type);
}

auto TypeTraits::is_pointer(const Type* type) const -> bool {
  return type && visit(IsPointer{}, type);
}

auto TypeTraits::is_lvalue_reference(const Type* type) const -> bool {
  return type && visit(IsLvalueReference{}, type);
}

auto TypeTraits::is_rvalue_reference(const Type* type) const -> bool {
  return type && visit(IsRvalueReference{}, type);
}

auto TypeTraits::is_member_object_pointer(const Type* type) const -> bool {
  return type && visit(IsMemberObjectPointer{}, type);
}

auto TypeTraits::is_member_function_pointer(const Type* type) const -> bool {
  return type && visit(IsMemberFunctionPointer{}, type);
}

auto TypeTraits::is_complete(const Type* type) const -> bool {
  return type && visit(IsComplete{}, type);
}

auto TypeTraits::is_integer(const Type* type) const -> bool {
  return is_integral(type);
}

auto TypeTraits::is_integral_or_unscoped_enum(const Type* type) const -> bool {
  return is_integral(type) || (is_enum(type) && !is_scoped_enum(type));
}

auto TypeTraits::is_integral_or_enum(const Type* type) const -> bool {
  return is_integral(type) || is_enum(type);
}

auto TypeTraits::is_fundamental(const Type* type) const -> bool {
  return is_arithmetic(type) || is_void(type) || is_null_pointer(type);
}

auto TypeTraits::is_arithmetic(const Type* type) const -> bool {
  return is_integral(type) || is_floating_point(type);
}

auto TypeTraits::is_scalar(const Type* type) const -> bool {
  return is_arithmetic(type) || is_enum(type) || is_pointer(type) ||
         is_member_pointer(type) || is_null_pointer(type);
}

auto TypeTraits::is_object(const Type* type) const -> bool {
  return is_scalar(type) || is_array(type) || is_union(type) || is_class(type);
}

auto TypeTraits::is_compound(const Type* type) const -> bool {
  return !is_fundamental(type);
}

auto TypeTraits::is_reference(const Type* type) const -> bool {
  return type && visit(IsReference{}, type);
}

auto TypeTraits::is_member_pointer(const Type* type) const -> bool {
  return is_member_object_pointer(type) || is_member_function_pointer(type);
}

auto TypeTraits::is_const(const Type* type) const -> bool {
  return type && visit(IsConst{}, type);
}

auto TypeTraits::is_volatile(const Type* type) const -> bool {
  return type && visit(IsVolatile{}, type);
}

auto TypeTraits::is_signed(const Type* type) const -> bool {
  return type &&
         visit(IsSigned{control()->memoryLayout()->isWideCharSigned()}, type);
}

auto TypeTraits::is_unsigned(const Type* type) const -> bool {
  return type &&
         visit(IsUnsigned{control()->memoryLayout()->isWideCharSigned()}, type);
}

auto TypeTraits::is_bounded_array(const Type* type) const -> bool {
  return type && visit(IsBoundedArray{}, type);
}

auto TypeTraits::is_unbounded_array(const Type* type) const -> bool {
  return type && visit(IsUnboundedArray{}, type);
}

auto TypeTraits::is_scoped_enum(const Type* type) const -> bool {
  return type && visit(IsScopedEnum{}, type);
}

auto TypeTraits::is_member_of_object_type(const Type* objectType,
                                          Symbol* member) const -> bool {
  if (!member || !objectType) return false;

  auto memberClass = symbol_cast<ClassSymbol>(member->parent());
  if (!memberClass) return false;

  auto objectClass = type_cast<ClassType>(remove_cv(objectType));
  if (!objectClass || !objectClass->symbol()) return false;

  if (memberClass->resolvedDefinition() ==
      objectClass->symbol()->resolvedDefinition()) {
    return true;
  }

  return is_base_of(memberClass->type(), objectType);
}

auto TypeTraits::underlying_type(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(UnderlyingType{}, remove_cv(type));
}

auto TypeTraits::remove_reference(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(RemoveReference{}, type);
}

auto TypeTraits::add_lvalue_reference(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(AddLvalueReference{*this}, type);
}

auto TypeTraits::add_rvalue_reference(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(AddRvalueReference{*this}, type);
}

auto TypeTraits::decltype_of(ExpressionAST* expr) const -> const Type* {
  if (!expr) return nullptr;

  auto namedSymbol = [&]() -> Symbol* {
    if (auto id = ast_cast<IdExpressionAST>(expr)) return id->symbol;
    if (auto member = ast_cast<MemberExpressionAST>(expr))
      return member->symbol;
    return nullptr;
  }();

  if (symbol_cast<OverloadSetSymbol>(namedSymbol)) return expr->type;
  if (namedSymbol) return namedSymbol->type();

  if (!expr->type) return nullptr;

  if (unit_->language() != LanguageKind::kCXX) return expr->type;

  if (is_lvalue(expr)) return add_lvalue_reference(expr->type);
  if (is_xvalue(expr)) return add_rvalue_reference(expr->type);
  return expr->type;
}

auto TypeTraits::remove_extent(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(RemoveExtent{}, type);
}

auto TypeTraits::get_element_type(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(GetElementType{}, type);
}

auto TypeTraits::remove_cv(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(RemoveQualifiers{*this, CvQualifiers::kConstVolatile}, type);
}

auto TypeTraits::remove_cvref(const Type* type) const -> const Type* {
  if (!type) return type;
  return remove_cv(remove_reference(type));
}

auto TypeTraits::add_const_ref(const Type* type) const -> const Type* {
  if (!type) return type;
  return add_lvalue_reference(add_const(type));
}

auto TypeTraits::add_const(const Type* type) const -> const Type* {
  return add_cv(type, CvQualifiers::kConst);
}

auto TypeTraits::add_volatile(const Type* type) const -> const Type* {
  return add_cv(type, CvQualifiers::kVolatile);
}

auto TypeTraits::remove_pointer(const Type* type) const -> const Type* {
  if (auto ptrTy = type_cast<PointerType>(remove_cv(type)))
    return ptrTy->elementType();
  return type;
}

auto TypeTraits::add_pointer(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(AddPointer{*this}, type);
}

auto TypeTraits::is_same(const Type* a, const Type* b) const -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (a->kind() != b->kind()) return false;
#define PROCESS_TYPE(K)                                         \
  case TypeKind::k##K:                                          \
    return IsSameVisitor{*this}(static_cast<const K##Type*>(a), \
                                static_cast<const K##Type*>(b));
  switch (a->kind()) {
    CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
    default:
      return false;
  }
#undef PROCESS_TYPE
}

auto TypeTraits::is_compatible(const Type* a, const Type* b) const -> bool {
  return is_same(remove_cv(a), remove_cv(b));
}

auto TypeTraits::decay(const Type* type) const -> const Type* {
  if (!type) return type;
  auto noref = remove_reference(type);
  if (is_array(noref)) return add_pointer(remove_extent(noref));
  if (is_function(noref)) return add_pointer(noref);
  return remove_cvref(noref);
}

auto TypeTraits::integer_type_of_size(std::size_t size, bool isUnsigned) const
    -> const Type* {
  const Type* candidates[] = {
      isUnsigned ? static_cast<const Type*>(control()->getUnsignedCharType())
                 : static_cast<const Type*>(control()->getSignedCharType()),
      isUnsigned
          ? static_cast<const Type*>(control()->getUnsignedShortIntType())
          : static_cast<const Type*>(control()->getShortIntType()),
      isUnsigned ? static_cast<const Type*>(control()->getUnsignedIntType())
                 : static_cast<const Type*>(control()->getIntType()),
      isUnsigned ? static_cast<const Type*>(control()->getUnsignedLongIntType())
                 : static_cast<const Type*>(control()->getLongIntType()),
      isUnsigned
          ? static_cast<const Type*>(control()->getUnsignedLongLongIntType())
          : static_cast<const Type*>(control()->getLongLongIntType()),
      isUnsigned ? static_cast<const Type*>(control()->getUnsignedInt128Type())
                 : static_cast<const Type*>(control()->getInt128Type()),
  };

  for (auto candidate : candidates) {
    if (control()->memoryLayout()->sizeOf(candidate) == size) return candidate;
  }

  return nullptr;
}

auto TypeTraits::corresponding_integer_type(const Type* type,
                                            bool isUnsigned) const
    -> const Type* {
  switch (type->kind()) {
    case TypeKind::kSignedChar:
    case TypeKind::kUnsignedChar:
    case TypeKind::kChar:
      return isUnsigned
                 ? static_cast<const Type*>(control()->getUnsignedCharType())
                 : static_cast<const Type*>(control()->getSignedCharType());

    case TypeKind::kShortInt:
    case TypeKind::kUnsignedShortInt:
      return isUnsigned
                 ? static_cast<const Type*>(
                       control()->getUnsignedShortIntType())
                 : static_cast<const Type*>(control()->getShortIntType());

    case TypeKind::kInt:
    case TypeKind::kUnsignedInt:
      return isUnsigned
                 ? static_cast<const Type*>(control()->getUnsignedIntType())
                 : static_cast<const Type*>(control()->getIntType());

    case TypeKind::kLongInt:
    case TypeKind::kUnsignedLongInt:
      return isUnsigned
                 ? static_cast<const Type*>(control()->getUnsignedLongIntType())
                 : static_cast<const Type*>(control()->getLongIntType());

    case TypeKind::kLongLongInt:
    case TypeKind::kUnsignedLongLongInt:
      return isUnsigned
                 ? static_cast<const Type*>(
                       control()->getUnsignedLongLongIntType())
                 : static_cast<const Type*>(control()->getLongLongIntType());

    case TypeKind::kInt128:
    case TypeKind::kUnsignedInt128:
      return isUnsigned
                 ? static_cast<const Type*>(control()->getUnsignedInt128Type())
                 : static_cast<const Type*>(control()->getInt128Type());

    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar:
    case TypeKind::kEnum:
    case TypeKind::kScopedEnum: {
      auto size = control()->memoryLayout()->sizeOf(type);
      if (!size) return nullptr;
      return integer_type_of_size(*size, isUnsigned);
    }

    default:
      return nullptr;
  }
}

auto TypeTraits::make_signed(const Type* type) const -> const Type* {
  return apply_sign(type, /*isUnsigned=*/false);
}

auto TypeTraits::make_unsigned(const Type* type) const -> const Type* {
  return apply_sign(type, /*isUnsigned=*/true);
}

auto TypeTraits::apply_sign(const Type* type, bool isUnsigned) const
    -> const Type* {
  if (!type) return type;

  auto result = corresponding_integer_type(remove_cv(type), isUnsigned);
  if (!result) return type;

  return add_cv(result, cv_qualifiers(type));
}

auto TypeTraits::is_class_or_union(const Type* type) const -> bool {
  return is_class(type) || is_union(type);
}

auto TypeTraits::is_arithmetic_or_unscoped_enum(const Type* type) const
    -> bool {
  return is_arithmetic(type) || (is_enum(type) && !is_scoped_enum(type));
}

auto TypeTraits::is_narrow_char_type(const Type* type) const -> bool {
  struct V {
    auto operator()(const CharType*) const -> bool { return true; }
    auto operator()(const SignedCharType*) const -> bool { return true; }
    auto operator()(const UnsignedCharType*) const -> bool { return true; }
    auto operator()(const QualType* t) const -> bool {
      return visit(*this, t->elementType());
    }
    auto operator()(const Type*) const -> bool { return false; }
  };
  return type && visit(V{}, type);
}

auto TypeTraits::is_char_type(const Type* type) const -> bool {
  struct V {
    auto operator()(const CharType*) const -> bool { return true; }
    auto operator()(const SignedCharType*) const -> bool { return true; }
    auto operator()(const UnsignedCharType*) const -> bool { return true; }
    auto operator()(const Char8Type*) const -> bool { return true; }
    auto operator()(const Char16Type*) const -> bool { return true; }
    auto operator()(const Char32Type*) const -> bool { return true; }
    auto operator()(const WideCharType*) const -> bool { return true; }
    auto operator()(const QualType* t) const -> bool {
      return visit(*this, t->elementType());
    }
    auto operator()(const Type*) const -> bool { return false; }
  };
  return type && visit(V{}, type);
}

auto TypeTraits::is_narrowing_conversion(const Type* from, const Type* to) const
    -> bool {
  if (!from || !to) return false;

  from = remove_cvref(from);
  to = remove_cvref(to);

  if (is_same(from, to)) return false;

  if (is_floating_point(from) && is_integral(to)) return true;

  if (is_floating_point(from) && is_floating_point(to)) {
    auto fromSize = control()->memoryLayout()->sizeOf(from);
    auto toSize = control()->memoryLayout()->sizeOf(to);
    if (fromSize && toSize && *fromSize > *toSize) return true;
  }

  if (is_integral_or_unscoped_enum(from) && is_floating_point(to)) return true;

  if (is_integral_or_unscoped_enum(from) && is_integral(to)) {
    auto fromSize = control()->memoryLayout()->sizeOf(from);
    auto toSize = control()->memoryLayout()->sizeOf(to);
    if (fromSize && toSize) {
      if (*fromSize > *toSize) return true;
      if (*fromSize == *toSize && is_signed(from) != is_signed(to)) return true;
    }
  }

  return false;
}

namespace {

auto listElementSource(ExpressionAST* expr) -> ExpressionAST* {
  while (expr) {
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(expr)) {
      expr = cast->expression;
      continue;
    }
    if (auto constant = ast_cast<ConstExpressionAST>(expr)) {
      expr = constant->expression;
      continue;
    }
    if (auto nested = ast_cast<NestedExpressionAST>(expr)) {
      expr = nested->expression;
      continue;
    }
    if (ast_cast<EqualInitializerAST>(expr) ||
        ast_cast<ParenInitializerAST>(expr)) {
      auto inner = Initializer{expr}.singleExpression();
      if (!inner) return expr;
      expr = inner;
      continue;
    }
    return expr;
  }
  return expr;
}

}  // namespace

auto TypeTraits::is_narrowing_list_element(ExpressionAST* expr,
                                           const Type* targetType) const
    -> bool {
  if (!expr) return false;

  auto source = listElementSource(expr);
  if (!source) return false;

  auto sourceType = source->type ? source->type : expr->type;
  if (!is_narrowing_conversion(sourceType, targetType)) return false;

  targetType = remove_cv(targetType);

  auto fitsInteger = [&](std::intmax_t value) {
    if (is_integral(targetType)) {
      if (value >= 0) {
        return integer_constant_fits_in_type(static_cast<std::uint64_t>(value),
                                             targetType);
      }
      if (!is_signed(targetType)) return false;
      auto targetSize = control()->memoryLayout()->sizeOf(targetType);
      if (!targetSize) return false;
      const auto magnitude = std::uint64_t{1} << (*targetSize * 8 - 1);
      return static_cast<std::uint64_t>(-(value + 1)) < magnitude;
    }

    auto exact = static_cast<long double>(value);
    if (type_cast<FloatType>(targetType)) {
      auto converted = static_cast<float>(value);
      return std::isfinite(converted) &&
             static_cast<long double>(converted) == exact;
    }
    if (type_cast<DoubleType>(targetType)) {
      auto converted = static_cast<double>(value);
      return std::isfinite(converted) &&
             static_cast<long double>(converted) == exact;
    }
    if (type_cast<LongDoubleType>(targetType)) {
      auto converted = static_cast<long double>(value);
      return std::isfinite(static_cast<double>(converted)) &&
             converted == exact;
    }
    return false;
  };

  auto fitsFloating = [&](double value) {
    if (!is_floating_point(targetType)) return false;
    const bool convertedIsFinite =
        type_cast<FloatType>(targetType)
            ? std::isfinite(static_cast<float>(value))
            : std::isfinite(static_cast<long double>(value));
    if (!std::isfinite(value)) return !convertedIsFinite;
    return convertedIsFinite;
  };

  if (auto intLiteral = ast_cast<IntLiteralExpressionAST>(source)) {
    if (!intLiteral->literal) return true;
    return !fitsInteger(
        static_cast<std::intmax_t>(intLiteral->literal->integerValue()));
  }

  if (auto floatLiteral = ast_cast<FloatLiteralExpressionAST>(source)) {
    if (!floatLiteral->literal) return true;
    return !fitsFloating(floatLiteral->literal->floatValue());
  }

  auto value = ASTInterpreter{unit_}.evaluate(source);
  if (!value) return true;

  if (auto intValue = std::get_if<std::intmax_t>(&*value))
    return !fitsInteger(*intValue);
  if (auto floatValue = std::get_if<float>(&*value))
    return !fitsFloating(*floatValue);
  if (auto doubleValue = std::get_if<double>(&*value))
    return !fitsFloating(*doubleValue);
  if (auto longDoubleValue = std::get_if<long double>(&*value))
    return !fitsFloating(static_cast<double>(*longDoubleValue));

  return true;
}

auto TypeTraits::integer_constant_fits_in_type(std::uint64_t value,
                                               const Type* targetType) const
    -> bool {
  if (!is_integral(targetType)) return false;

  auto targetSize = control()->memoryLayout()->sizeOf(targetType);
  if (!targetSize) return false;

  if (is_signed(targetType)) {
    auto maxVal = (std::uint64_t{1} << (*targetSize * 8 - 1)) - 1;
    return value <= maxVal;
  }

  if (*targetSize >= 8) return true;
  auto maxVal = (std::uint64_t{1} << (*targetSize * 8)) - 1;
  return value <= maxVal;
}

auto TypeTraits::initializer_list_element_type(const Type* targetType)
    -> const Type* {
  if (!targetType) return nullptr;

  auto unrefTarget = remove_reference(targetType);
  auto unqualTarget = remove_cv(unrefTarget);
  auto classType = type_cast<ClassType>(unqualTarget);
  if (!classType || !classType->symbol()) return nullptr;

  auto classSymbol = classType->symbol();
  auto className = name_cast<Identifier>(classSymbol->name());
  if (!className || className->name() != "initializer_list") return nullptr;

  auto isWithinStdNamespace = [](Symbol* symbol) {
    auto parent = symbol->parent();
    while (parent) {
      if (auto ns = symbol_cast<NamespaceSymbol>(parent)) {
        if (auto id = name_cast<Identifier>(ns->name())) {
          if (id->name() == "std" || id->name() == "__1" ||
              id->name() == "__cxx11")
            return true;
        }
      }
      parent = parent->parent();
    }
    return false;
  };
  if (!isWithinStdNamespace(classSymbol)) return nullptr;
  if (!classSymbol->isSpecialization()) return nullptr;

  auto args = classSymbol->templateArguments();
  if (args.size() != 1) return nullptr;

  requireCompleteClass(classSymbol);

  if (auto typeArg = std::get_if<const Type*>(&args[0])) return *typeArg;
  if (auto symbolArg = std::get_if<Symbol*>(&args[0])) {
    auto sym = *symbolArg;
    if (!sym) return nullptr;
    return sym->type();
  }

  return nullptr;
}

auto TypeTraits::requireCompleteClass(ClassSymbol* classSymbol) -> bool {
  if (!classSymbol) return false;
  if (classSymbol->isComplete()) return true;
  if (!unit_) return false;
  if (!unit_->config().checkTypes) return false;
  return ASTRewriter::ensureCompleteClass(unit_, classSymbol);
}

auto TypeTraits::remove_all_extents(const Type* type) const -> const Type* {
  while (is_array(type)) {
    type = remove_extent(type);
  }
  return type;
}

auto TypeTraits::remove_const(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(RemoveQualifiers{*this, CvQualifiers::kConst}, type);
}

auto TypeTraits::remove_volatile(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(RemoveQualifiers{*this, CvQualifiers::kVolatile}, type);
}

auto TypeTraits::add_cv(const Type* type, CvQualifiers cv) const
    -> const Type* {
  if (!type) return type;
  if (cv == CvQualifiers::kNone) return type;
  return visit(AddCvQualifiers{*this, cv}, type);
}

auto TypeTraits::remove_noexcept(const Type* type) const -> const Type* {
  const auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return type;
  return control()->getFunctionType(
      functionType->returnType(), functionType->parameterTypes(),
      functionType->isVariadic(), functionType->cvQualifiers(),
      functionType->refQualifier(), false);
}

auto TypeTraits::replace_placeholder_types(const Type* type,
                                           const Type* replacement) const
    -> const Type* {
  if (type_cast<AutoType>(type) || type_cast<DecltypeAutoType>(type))
    return replacement;

  if (auto qualType = type_cast<QualType>(type)) {
    auto elementType =
        replace_placeholder_types(qualType->elementType(), replacement);
    return control()->getQualType(elementType, qualType->cvQualifiers());
  }

  if (auto arrayType = type_cast<BoundedArrayType>(type)) {
    auto elementType =
        replace_placeholder_types(arrayType->elementType(), replacement);
    return control()->getBoundedArrayType(elementType, arrayType->size());
  }

  if (auto arrayType = type_cast<UnboundedArrayType>(type)) {
    auto elementType =
        replace_placeholder_types(arrayType->elementType(), replacement);
    return control()->getUnboundedArrayType(elementType);
  }

  if (auto arrayType = type_cast<UnresolvedBoundedArrayType>(type)) {
    auto elementType =
        replace_placeholder_types(arrayType->elementType(), replacement);
    return control()->getUnresolvedBoundedArrayType(
        arrayType->translationUnit(), elementType, arrayType->size());
  }

  if (auto pointerType = type_cast<PointerType>(type)) {
    auto elementType =
        replace_placeholder_types(pointerType->elementType(), replacement);
    return control()->getPointerType(elementType);
  }

  if (auto referenceType = type_cast<LvalueReferenceType>(type)) {
    auto elementType =
        replace_placeholder_types(referenceType->elementType(), replacement);
    return add_lvalue_reference(elementType);
  }

  if (auto referenceType = type_cast<RvalueReferenceType>(type)) {
    auto elementType =
        replace_placeholder_types(referenceType->elementType(), replacement);
    return add_rvalue_reference(elementType);
  }

  if (auto functionType = type_cast<FunctionType>(type)) {
    auto returnType =
        replace_placeholder_types(functionType->returnType(), replacement);
    auto parameterTypes = std::vector<const Type*>{};
    parameterTypes.reserve(functionType->parameterTypes().size());
    for (auto parameterType : functionType->parameterTypes()) {
      parameterTypes.push_back(
          replace_placeholder_types(parameterType, replacement));
    }
    return control()->getFunctionType(
        returnType, std::move(parameterTypes), functionType->isVariadic(),
        functionType->cvQualifiers(), functionType->refQualifier(),
        functionType->isNoexcept());
  }

  if (auto pointerType = type_cast<MemberObjectPointerType>(type)) {
    auto classType =
        replace_placeholder_types(pointerType->classType(), replacement);
    auto elementType =
        replace_placeholder_types(pointerType->elementType(), replacement);
    return control()->getMemberObjectPointerType(classType, elementType);
  }

  if (auto pointerType = type_cast<MemberFunctionPointerType>(type)) {
    auto classType =
        replace_placeholder_types(pointerType->classType(), replacement);
    auto functionType =
        replace_placeholder_types(pointerType->functionType(), replacement);
    return control()->getMemberFunctionPointerType(
        classType, type_cast<FunctionType>(functionType));
  }

  return type;
}

auto TypeTraits::is_base_of(const Type* base, const Type* derived) const
    -> bool {
  auto baseClassType = type_cast<ClassType>(remove_cv(base));
  if (!baseClassType) return false;
  auto derivedClassType = type_cast<ClassType>(remove_cv(derived));
  if (!derivedClassType) return false;
  if (derivedClassType->symbol() == baseClassType->symbol()) return true;
  return derivedClassType->symbol()->hasBaseClass(baseClassType->symbol());
}

namespace {
[[nodiscard]] auto decomposeForQualification(const TypeTraits& traits,
                                             const Type* lhs, const Type* rhs)
    -> std::optional<QualificationDecompositionPair> {
  QualificationDecompositionPair result;

  for (;;) {
    result.left.cv.push_back(cv_qualifiers(lhs));
    result.right.cv.push_back(cv_qualifiers(rhs));

    auto leftStep = decomposeQualificationComponent(lhs);
    auto rightStep = decomposeQualificationComponent(rhs);

    if (!leftStep.component || !rightStep.component) break;
    if (!componentsMatch(traits, *leftStep.component, *rightStep.component))
      break;

    result.left.components.push_back(*leftStep.component);
    result.right.components.push_back(*rightStep.component);

    lhs = leftStep.next;
    rhs = rightStep.next;
  }

  result.left.terminal = unqualified_type(lhs);
  result.right.terminal = unqualified_type(rhs);

  if (!traits.is_same(result.left.terminal, result.right.terminal))
    return std::nullopt;

  return result;
}
}  // namespace

auto TypeTraits::is_known_complete_object(ExpressionAST* expression) const
    -> bool {
  while (expression) {
    if (auto nested = ast_cast<NestedExpressionAST>(expression)) {
      expression = nested->expression;
      continue;
    }
    if (auto cast = ast_cast<ImplicitCastExpressionAST>(expression);
        cast && cast->castKind == ImplicitCastKind::kDerivedToBaseConversion) {
      expression = cast->expression;
      continue;
    }
    break;
  }

  Symbol* symbol = nullptr;
  if (auto id = ast_cast<IdExpressionAST>(expression)) {
    symbol = id->symbol;
  } else if (auto member = ast_cast<MemberExpressionAST>(expression)) {
    symbol = member->symbol;
  }

  if (!symbol) return false;

  if (!symbol_cast<VariableSymbol>(symbol) &&
      !symbol_cast<ParameterSymbol>(symbol) &&
      !symbol_cast<FieldSymbol>(symbol))
    return false;

  return !is_reference(symbol->type());
}

auto TypeTraits::is_virtual_member_dispatch(
    FunctionSymbol* function, ExpressionAST* objectExpression) const -> bool {
  if (!function || !function->isVirtual()) return false;
  if (!function->isImplicitObjectMemberFunction()) return false;
  if (!objectExpression || !is_glvalue(objectExpression)) return false;
  return !is_known_complete_object(objectExpression);
}

auto TypeTraits::adjusted_cv_type(const Type* type) const -> const Type* {
  auto qualType = type_cast<QualType>(type);
  if (!qualType) return type;

  if (is_class(type) || is_array(type)) return type;

  return qualType->elementType();
}

auto TypeTraits::is_similar(const Type* lhs, const Type* rhs) const -> bool {
  return decomposeForQualification(*this, lhs, rhs).has_value();
}

auto TypeTraits::qualification_combined_type(const Type* lhs,
                                             const Type* rhs) const
    -> const Type* {
  auto decomposition = decomposeForQualification(*this, lhs, rhs);
  if (!decomposition) return nullptr;

  const auto& left = decomposition->left;
  const auto& right = decomposition->right;
  const auto depth = left.cv.size();

  std::vector<CvQualifiers> cv(depth);
  for (std::size_t i = 0; i < depth; ++i) cv[i] = left.cv[i] | right.cv[i];

  std::vector<QualificationComponent> components;
  components.reserve(left.components.size());
  for (std::size_t i = 0; i < left.components.size(); ++i) {
    auto component = left.components[i];
    if (right.components[i].kind ==
        QualificationComponent::Kind::kUnboundedArray) {
      component = right.components[i];
    }
    components.push_back(component);
  }

  auto isChangedAt = [&](std::size_t i) {
    if (cv[i] != left.cv[i] || cv[i] != right.cv[i]) return true;
    if (i >= components.size()) return false;
    return components[i].kind != left.components[i].kind ||
           components[i].kind != right.components[i].kind;
  };

  for (auto changed = true; changed;) {
    changed = false;

    for (std::size_t i = 0; i < depth; ++i) {
      if (!isChangedAt(i)) continue;
      for (std::size_t k = 1; k < i; ++k) {
        if (has_const(cv[k])) continue;
        cv[k] = cv[k] | CvQualifiers::kConst;
        changed = true;
      }
    }

    for (std::size_t i = 0; i < components.size(); ++i) {
      if (!components[i].isArray()) continue;
      auto shared = cv[i] | cv[i + 1];
      if (shared == cv[i] && shared == cv[i + 1]) continue;
      cv[i] = shared;
      cv[i + 1] = shared;
      changed = true;
    }
  }

  auto result = add_cv(left.terminal, cv[depth - 1]);

  for (auto i = components.size(); i-- > 0;) {
    const auto& component = components[i];
    switch (component.kind) {
      case QualificationComponent::Kind::kPointer:
        result = control()->getPointerType(result);
        break;
      case QualificationComponent::Kind::kMemberPointer:
        if (auto functionType = type_cast<FunctionType>(result)) {
          result = control()->getMemberFunctionPointerType(component.classType,
                                                           functionType);
        } else {
          result = control()->getMemberObjectPointerType(component.classType,
                                                         result);
        }
        break;
      case QualificationComponent::Kind::kBoundedArray:
        result = control()->getBoundedArrayType(result, component.size);
        break;
      case QualificationComponent::Kind::kUnboundedArray:
        result = control()->getUnboundedArrayType(result);
        break;
    }
    result = add_cv(result, cv[i]);
  }

  return result;
}

auto TypeTraits::is_qualification_convertible(const Type* from,
                                              const Type* to) const -> bool {
  return is_same(qualification_combined_type(from, to), to);
}

auto TypeTraits::is_reference_related(const Type* lhs, const Type* rhs) const
    -> bool {
  if (is_similar(remove_cv(lhs), remove_cv(rhs))) return true;
  return is_base_of(lhs, rhs);
}

auto TypeTraits::representsAllValuesOf(const Type* target,
                                       const Type* source) const -> bool {
  auto memoryLayout = control()->memoryLayout();
  auto targetSize = memoryLayout->sizeOf(target).value_or(0);
  auto sourceSize = memoryLayout->sizeOf(source).value_or(0);
  if (!targetSize || !sourceSize) return false;

  if (is_unsigned(source)) {
    if (is_unsigned(target)) return targetSize >= sourceSize;
    return targetSize > sourceSize;
  }

  if (is_unsigned(target)) return false;
  return targetSize >= sourceSize;
}

auto TypeTraits::integralPromotionCandidates() const
    -> std::array<const Type*, 6> {
  return {
      control()->getIntType(),         control()->getUnsignedIntType(),
      control()->getLongIntType(),     control()->getUnsignedLongIntType(),
      control()->getLongLongIntType(), control()->getUnsignedLongLongIntType()};
}

auto TypeTraits::promoted_integer_type(const Type* type) const -> const Type* {
  if (!type) return control()->getIntType();

  auto source = remove_cv(type);

  if (auto enumType = type_cast<EnumType>(source)) {
    auto [underlyingType, promotedType] = promoted_enumeration_types(enumType);
    return promotedType ? promotedType : underlyingType;
  }

  switch (source->kind()) {
    case TypeKind::kBool:
      return control()->getIntType();

    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar:
      for (auto candidate : integralPromotionCandidates()) {
        if (representsAllValuesOf(candidate, source)) return candidate;
      }
      return source;

    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kUnsignedChar:
    case TypeKind::kShortInt:
    case TypeKind::kUnsignedShortInt:
      if (representsAllValuesOf(control()->getIntType(), source))
        return control()->getIntType();
      return control()->getUnsignedIntType();

    default:
      return source;
  }
}

auto TypeTraits::promoted_enumeration_types(const EnumType* enumType) const
    -> std::pair<const Type*, const Type*> {
  auto enumSymbol = enumType->symbol();

  if (enumSymbol && enumSymbol->hasFixedUnderlyingType()) {
    auto underlyingType = enumType->underlyingType();
    if (!underlyingType) return {control()->getIntType(), nullptr};
    auto promotedType = promoted_integer_type(underlyingType);
    if (is_same(promotedType, remove_cv(underlyingType)))
      return {underlyingType, nullptr};
    return {underlyingType, promotedType};
  }

  auto minValue = std::intmax_t{0};
  auto maxValue = std::intmax_t{0};

  if (enumSymbol) {
    for (auto member : enumSymbol->members()) {
      auto enumerator = symbol_cast<EnumeratorSymbol>(member);
      if (!enumerator) continue;
      const auto& value = enumerator->value();
      if (!value) continue;
      auto intValue = std::get_if<std::intmax_t>(&*value);
      if (!intValue) continue;
      minValue = std::min(minValue, *intValue);
      maxValue = std::max(maxValue, *intValue);
    }
  }

  auto memoryLayout = control()->memoryLayout();

  auto representsEnumerationValues = [&](const Type* type) {
    auto bits = memoryLayout->sizeOf(type).value_or(0) * 8;
    if (bits == 0) return false;
    if (bits >= 64) return is_signed(type) || minValue >= 0;
    if (is_signed(type)) {
      const auto limit = std::intmax_t{1} << (bits - 1);
      return minValue >= -limit && maxValue < limit;
    }
    return minValue >= 0 && maxValue < (std::intmax_t{1} << bits);
  };

  for (auto candidate : integralPromotionCandidates()) {
    if (representsEnumerationValues(candidate)) return {candidate, nullptr};
  }

  auto underlyingType = enumType->underlyingType();
  return {underlyingType ? underlyingType : control()->getIntType(), nullptr};
}

auto TypeTraits::is_integral_promotion(const Type* from, const Type* to) const
    -> bool {
  if (!from || !to) return false;

  auto source = remove_cv(from);
  auto target = remove_cv(to);

  if (auto enumType = type_cast<EnumType>(source)) {
    auto [underlyingType, promotedType] = promoted_enumeration_types(enumType);
    if (underlyingType && is_same(remove_cv(underlyingType), target))
      return true;
    return promotedType && is_same(remove_cv(promotedType), target);
  }

  if (!is_integral(source)) return false;

  auto promotedType = promoted_integer_type(source);
  if (is_same(promotedType, source)) return false;
  return is_same(promotedType, target);
}

auto TypeTraits::is_floating_point_promotion(const Type* from,
                                             const Type* to) const -> bool {
  if (!from || !to) return false;
  return remove_cv(from)->kind() == TypeKind::kFloat &&
         remove_cv(to)->kind() == TypeKind::kDouble;
}

auto TypeTraits::is_reference_compatible(const Type* target,
                                         const Type* source) const -> bool {
  if (!target || !source) return false;

  if (is_qualification_convertible(control()->getPointerType(source),
                                   control()->getPointerType(target)))
    return true;

  auto targetUnqualified = remove_cv(target);
  auto sourceUnqualified = remove_cv(source);

  if (is_function(targetUnqualified) && is_function(sourceUnqualified)) {
    return is_same(remove_noexcept(sourceUnqualified), targetUnqualified);
  }

  if (!is_base_of(targetUnqualified, sourceUnqualified)) return false;

  return is_at_least_as_cv_qualified(cv_qualifiers(target),
                                     cv_qualifiers(source));
}

auto TypeTraits::is_virtual_base_of(const Type* base, const Type* derived) const
    -> bool {
  auto baseClassType = type_cast<ClassType>(remove_cv(base));
  if (!baseClassType) return false;
  auto derivedClassType = type_cast<ClassType>(remove_cv(derived));
  if (!derivedClassType) return false;
  if (derivedClassType->symbol() == baseClassType->symbol()) return false;
  return derivedClassType->symbol()->hasVirtualBasePath(
      baseClassType->symbol());
}

auto TypeTraits::is_corresponding_overrider(
    const FunctionSymbol* overrider, const FunctionSymbol* overridden) const
    -> bool {
  if (!overrider || !overridden) return false;

  if (overrider->isDestructor() || overridden->isDestructor())
    return overrider->isDestructor() && overridden->isDestructor();

  if (overrider->name() != overridden->name()) return false;

  auto overriderType = type_cast<FunctionType>(overrider->type());
  auto overriddenType = type_cast<FunctionType>(overridden->type());
  if (!overriderType || !overriddenType) return false;

  if (overriderType->cvQualifiers() != overriddenType->cvQualifiers())
    return false;

  if (overriderType->refQualifier() != overriddenType->refQualifier())
    return false;

  if (overriderType->isVariadic() != overriddenType->isVariadic()) return false;

  const auto& overriderParams = overriderType->parameterTypes();
  const auto& overriddenParams = overriddenType->parameterTypes();
  if (overriderParams.size() != overriddenParams.size()) return false;

  for (std::size_t i = 0; i < overriderParams.size(); ++i) {
    if (!is_same(overriderParams[i], overriddenParams[i])) return false;
  }

  return true;
}

auto TypeTraits::is_covariant_return_type(const Type* overriddenReturnType,
                                          const Type* overriderReturnType) const
    -> bool {
  if (!overriddenReturnType || !overriderReturnType) return false;
  if (is_same(overriddenReturnType, overriderReturnType)) return true;

  if (cv_qualifiers(overriddenReturnType) != cv_qualifiers(overriderReturnType))
    return false;

  auto overridden = remove_cv(overriddenReturnType);
  auto overrider = remove_cv(overriderReturnType);
  if (overridden->kind() != overrider->kind()) return false;

  auto classOf = [this](const Type* type) -> const Type* {
    const Type* element = nullptr;
    if (auto pointerType = type_cast<PointerType>(type))
      element = pointerType->elementType();
    else if (auto referenceType = type_cast<LvalueReferenceType>(type))
      element = referenceType->elementType();
    else if (auto referenceType = type_cast<RvalueReferenceType>(type))
      element = referenceType->elementType();
    if (!element || !is_class(remove_cv(element))) return nullptr;
    return element;
  };

  auto overriddenClass = classOf(overridden);
  auto overriderClass = classOf(overrider);
  if (!overriddenClass || !overriderClass) return false;

  const auto overriddenCv = cv_qualifiers(overriddenClass);
  const auto overriderCv = cv_qualifiers(overriderClass);
  if ((static_cast<int>(overriderCv) & ~static_cast<int>(overriddenCv)) != 0)
    return false;

  return is_base_of(overriddenClass, overriderClass);
}

auto TypeTraits::can_initialize(const Type* to, const Type* from,
                                bool directInitialization) const -> bool {
  if (!from || !to) return false;

  const auto fromIsVoid = is_void(from);
  const auto toIsVoid = is_void(to);
  if (fromIsVoid || toIsVoid) return fromIsVoid && toIsVoid;

  auto valueCategory = ValueCategory::kXValue;
  if (is_lvalue_reference(from)) valueCategory = ValueCategory::kLValue;

  auto declvalFrom = ThisExpressionAST::create(unit_->arena(), valueCategory,
                                               remove_reference(from));

  StandardConversion conversions{unit_};
  auto initializationKind = InitializationKind::kCopyInitialization;
  if (directInitialization)
    initializationKind = InitializationKind::kDirectInitialization;
  auto sequence = conversions.computeConversionSequence(declvalFrom, to,
                                                        initializationKind);
  if (!sequence) return false;
  if (!is_accessible_from_unrelated_context(sequence.udc.function))
    return false;

  if (is_reference(to)) return true;
  auto classType = unqualified_cast<ClassType>(to);
  if (!classType) return true;
  auto classSymbol = classType->definition();
  if (!classSymbol) return false;
  auto destructor = classSymbol->destructor();
  if (destructor && destructor->isDeleted()) return false;
  return is_accessible_from_unrelated_context(destructor);
}

auto TypeTraits::is_accessible_from_unrelated_context(
    FunctionSymbol* function) const -> bool {
  if (!function) return true;
  auto declaringClass = declaringClassOf(function);
  if (!declaringClass) return true;
  AccessContext accessContext{unit_, unit_->globalScope()};
  return accessContext.isAccessible(function, declaringClass, nullptr);
}

auto TypeTraits::is_nothrow_function(FunctionSymbol* function) const -> bool {
  if (!function) return true;
  ASTRewriter::completePendingExceptionSpecification(unit_, function);
  auto functionType = type_cast<FunctionType>(function->type());
  return functionType && functionType->isNoexcept();
}

auto TypeTraits::is_nothrow_initialization(const Type* to, const Type* from,
                                           bool directInitialization) const
    -> bool {
  if (!from || !to) return false;
  auto valueCategory = ValueCategory::kXValue;
  if (is_lvalue_reference(from)) valueCategory = ValueCategory::kLValue;
  auto expression = ThisExpressionAST::create(unit_->arena(), valueCategory,
                                              remove_reference(from));
  auto initializationKind = InitializationKind::kCopyInitialization;
  if (directInitialization)
    initializationKind = InitializationKind::kDirectInitialization;
  auto sequence = StandardConversion{unit_}.computeConversionSequence(
      expression, to, initializationKind);
  if (!sequence) return false;
  return is_nothrow_function(sequence.udc.function);
}

auto TypeTraits::is_trivial_initialization(const Type* to, const Type* from,
                                           bool directInitialization) const
    -> bool {
  if (!from || !to) return false;
  auto valueCategory = ValueCategory::kXValue;
  if (is_lvalue_reference(from)) valueCategory = ValueCategory::kLValue;
  auto expression = ThisExpressionAST::create(unit_->arena(), valueCategory,
                                              remove_reference(from));
  auto initializationKind = InitializationKind::kCopyInitialization;
  if (directInitialization)
    initializationKind = InitializationKind::kDirectInitialization;
  auto sequence = StandardConversion{unit_}.computeConversionSequence(
      expression, to, initializationKind);
  if (!sequence) return false;
  return sequence.udc.function == nullptr;
}

auto TypeTraits::is_convertible(const Type* from, const Type* to) const
    -> bool {
  return can_initialize(to, from, false);
}

auto TypeTraits::reference_binds_to_temporary(const Type* to, const Type* from,
                                              bool directInitialization) const
    -> bool {
  if (!to || !from) return false;
  if (!is_reference(to)) return false;

  auto valueCategory = ValueCategory::kPrValue;
  if (is_lvalue_reference(from)) valueCategory = ValueCategory::kLValue;
  if (is_rvalue_reference(from)) valueCategory = ValueCategory::kXValue;
  if (is_function(from)) valueCategory = ValueCategory::kLValue;

  auto source = ThisExpressionAST::create(unit_->arena(), valueCategory,
                                          remove_reference(from));

  auto initializationKind = InitializationKind::kCopyInitialization;
  if (directInitialization)
    initializationKind = InitializationKind::kDirectInitialization;

  auto sequence = StandardConversion{unit_}.computeConversionSequence(
      source, to, initializationKind);
  if (!sequence) return false;
  if (!is_accessible_from_unrelated_context(sequence.udc.function))
    return false;
  return sequence.binding.bindsToTemporary();
}

auto TypeTraits::reference_constructs_from_temporary(const Type* to,
                                                     const Type* from) const
    -> bool {
  return reference_binds_to_temporary(to, from, /*directInitialization=*/true);
}

auto TypeTraits::reference_converts_from_temporary(const Type* to,
                                                   const Type* from) const
    -> bool {
  return reference_binds_to_temporary(to, from, /*directInitialization=*/false);
}

auto TypeTraits::is_pod(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (is_void(unqual)) return false;
  if (is_class(unqual) || is_union(unqual))
    return is_trivial(unqual) && is_standard_layout(unqual);
  if (is_array(unqual)) return is_pod(remove_all_extents(unqual));
  return false;
}

auto TypeTraits::is_trivial(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    if (!has_trivial_constructor(*this, cls, TrivialConstructorKind::kDefault))
      return false;
    if (!is_trivially_copyable_class(*this, cls)) return false;
    return true;
  }
  if (is_array(unqual)) {
    return is_trivial(remove_all_extents(unqual));
  }
  return false;
}

auto TypeTraits::is_standard_layout(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    if (cls->hasVirtualFunctions()) return false;
    if (cls->hasVirtualBaseClasses()) return false;

    std::optional<AccessSpecifier> memberAccess;
    for (auto field : cls->members() | views::non_static_fields) {
      if (is_reference(field->type())) return false;
      if (!is_standard_layout(remove_all_extents(field->type()))) return false;
      if (!memberAccess) memberAccess = field->accessSpecifier();
      if (*memberAccess != field->accessSpecifier()) return false;
    }

    auto dataBearingSubobjects = 0;
    if (has_non_static_data_members(cls)) dataBearingSubobjects = 1;
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      baseClass = baseClass->resolvedDefinition();
      if (!is_standard_layout(baseClass->type())) return false;
      if (has_data_members_in_hierarchy(baseClass)) ++dataBearingSubobjects;
      if (dataBearingSubobjects > 1) return false;
    }

    std::unordered_set<ClassSymbol*> baseTypes;
    if (!has_unique_base_subobject_types(cls, baseTypes)) return false;

    std::unordered_set<ClassSymbol*> memberTypes;
    std::unordered_set<ClassSymbol*> expandedMemberTypes;
    collect_standard_layout_member_types(*this, unqual, memberTypes,
                                         expandedMemberTypes);
    for (auto baseType : baseTypes) {
      if (memberTypes.contains(baseType)) return false;
    }
    return true;
  }
  if (is_array(unqual)) return is_standard_layout(remove_all_extents(unqual));
  return false;
}

auto TypeTraits::is_literal_type(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_void(unqual)) return true;
  if (is_scalar(unqual)) return true;
  if (is_reference(unqual)) return true;
  if (is_array(unqual)) return is_literal_type(remove_all_extents(unqual));
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    auto destructor = cls->destructor();
    auto hasConstexprDestructor = !destructor;
    if (destructor && destructor->isDefaulted()) hasConstexprDestructor = true;
    if (destructor && destructor->isConstexpr()) hasConstexprDestructor = true;
    if (!hasConstexprDestructor) return false;

    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (!is_literal_type(baseClass->type())) return false;
    }

    auto hasVariant = false;
    auto hasLiteralVariant = false;
    for (auto field : cls->members() | views::non_static_fields) {
      hasVariant = true;
      if (is_volatile(field->type())) {
        if (!cls->isUnion()) return false;
        continue;
      }
      auto fieldIsLiteral = is_literal_type(field->type());
      if (!cls->isUnion() && !fieldIsLiteral) return false;
      if (cls->isUnion() && fieldIsLiteral) hasLiteralVariant = true;
    }
    if (cls->isUnion() && hasVariant && !hasLiteralVariant) return false;

    if (cls->isClosureType() || is_aggregate(unqual)) return true;

    for (auto constructor : cls->constructors()) {
      if (constructor->isDeleted()) continue;
      if (constructor == cls->copyConstructor()) continue;
      if (constructor == cls->moveConstructor()) continue;
      if (constructor->isConstexpr()) return true;
    }

    auto defaultConstructor = cls->defaultConstructor();
    if (!defaultConstructor || defaultConstructor->isDeleted()) return false;
    if (!defaultConstructor->isDefaulted()) return false;

    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      baseClass = baseClass->resolvedDefinition();
      auto baseConstructor = baseClass->defaultConstructor();
      if (!baseConstructor || baseConstructor->isDeleted()) return false;
      if (!baseConstructor->isConstexpr()) return false;
    }

    for (auto field : cls->members() | views::non_static_fields) {
      if (field->initializer()) continue;
      auto fieldType = remove_all_extents(remove_cv(field->type()));
      auto fieldClassType = type_cast<ClassType>(fieldType);
      if (!fieldClassType) continue;
      auto fieldConstructor =
          fieldClassType->definition()->defaultConstructor();
      if (!fieldConstructor || fieldConstructor->isDeleted()) return false;
      if (!fieldConstructor->isConstexpr()) return false;
    }
    return true;
  }
  return false;
}

namespace {
struct AggregateElementType {
  const TypeTraits& traits;

  auto operator()(FieldSymbol* symbol) const -> const Type* {
    return traits.remove_cv(symbol->type());
  }

  auto operator()(BaseClassSymbol* symbol) const -> const Type* {
    auto baseClass = symbol_cast<ClassSymbol>(symbol->symbol());
    if (!baseClass) return nullptr;
    return baseClass->type();
  }

  auto operator()(auto) const -> const Type* { return nullptr; }
};
}  // namespace

auto TypeTraits::aggregate_element_type(Symbol* element) const -> const Type* {
  if (!element) return nullptr;
  return visit(AggregateElementType{*this}, element);
}

auto TypeTraits::aggregate_elements(ClassSymbol* classSymbol) const
    -> std::vector<Symbol*> {
  std::vector<Symbol*> elements;
  for (auto base : classSymbol->baseClasses()) elements.push_back(base);
  for (auto field : views::members(classSymbol) | views::non_static_fields)
    elements.push_back(field);
  return elements;
}

auto TypeTraits::is_aggregate(const Type* type) -> bool {
  if (is_array(type)) return true;
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;
  if (cls->hasInheritedConstructors()) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  if (cls->hasUserDeclaredConstructors()) return false;
  for (auto field : cls->members() | views::non_static_fields) {
    if (field->accessSpecifier() != AccessSpecifier::kPublic) return false;
  }
  for (auto base : cls->baseClasses()) {
    if (base->isVirtual()) return false;
    if (base->accessSpecifier() != AccessSpecifier::kPublic) return false;
  }
  return true;
}

auto TypeTraits::is_zero_size_subobject(FieldSymbol* field) -> bool {
  if (field->isBitField()) {
    if (field->name()) return false;
    auto& width = field->bitFieldWidth();
    if (!width.has_value()) return false;
    auto bits = std::get_if<std::intmax_t>(&*width);
    return bits && *bits == 0;
  }

  if (!field->isNoUniqueAddress()) return false;

  return is_empty(field->type());
}

auto TypeTraits::is_empty(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;
  for (auto f : cls->members() | views::non_static_fields) {
    if (!is_zero_size_subobject(f)) return false;
  }
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (!is_empty(baseClass->type())) return false;
  }
  return true;
}

auto TypeTraits::is_polymorphic(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;
  return cls->isPolymorphic();
}

auto TypeTraits::is_final(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->definition();
  if (!cls) return false;
  return cls->isFinal();
}

auto TypeTraits::selectConstructor(ClassSymbol* classSymbol,
                                   std::span<const Type* const> argTypes)
    -> FunctionSymbol* {
  std::vector<ExpressionAST*> args;
  args.reserve(argTypes.size());

  for (auto argType : argTypes) {
    if (!argType) return nullptr;
    auto valueCategory = ValueCategory::kXValue;
    if (is_lvalue_reference(argType)) valueCategory = ValueCategory::kLValue;
    args.push_back(ThisExpressionAST::create(unit_->arena(), valueCategory,
                                             remove_reference(argType)));
  }

  auto result = OverloadResolution{unit_}.resolveConstructor(classSymbol, args);
  if (result.ambiguous || !result.best) return nullptr;
  return result.best->symbol;
}

auto TypeTraits::is_constructible(const Type* type,
                                  std::span<const Type* const> argTypes)
    -> bool {
  if (!type) return false;
  auto unqual = remove_cv(type);

  if (is_reference(unqual)) {
    if (argTypes.size() != 1) return false;
    return can_initialize(unqual, argTypes[0], true);
  }

  if (is_scalar(unqual)) {
    if (argTypes.empty()) return true;
    if (argTypes.size() == 1) return can_initialize(unqual, argTypes[0], true);
    return false;
  }

  if (is_array(unqual)) {
    if (!argTypes.empty()) return false;
    return is_constructible(remove_all_extents(unqual), argTypes);
  }

  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;

    auto selected = selectConstructor(cls, argTypes);
    if (!selected || selected->isDeleted()) return false;
    if (cls->isAbstract()) return false;
    if (!is_accessible_from_unrelated_context(selected)) return false;
    auto destructor = cls->destructor();
    if (destructor && destructor->isDeleted()) return false;
    return is_accessible_from_unrelated_context(destructor);
  }

  if (is_void(unqual)) return false;

  return false;
}

auto TypeTraits::is_nothrow_constructible(const Type* type,
                                          std::span<const Type* const> argTypes)
    -> bool {
  if (!type) return false;
  auto unqual = remove_cv(type);

  if (!is_constructible(type, argTypes)) return false;

  if (is_reference(unqual) || is_scalar(unqual)) {
    if (argTypes.empty()) return true;
    return is_nothrow_initialization(unqual, argTypes.front(), true);
  }

  if (is_array(unqual))
    return is_nothrow_constructible(remove_all_extents(unqual), argTypes);

  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;

    std::vector<ExpressionAST*> args;
    args.reserve(argTypes.size());
    for (auto argType : argTypes) {
      auto category = ValueCategory::kXValue;
      if (is_lvalue_reference(argType)) category = ValueCategory::kLValue;
      args.push_back(ThisExpressionAST::create(unit_->arena(), category,
                                               remove_reference(argType)));
    }
    auto result = OverloadResolution{unit_}.resolveConstructor(cls, args);
    if (result.ambiguous || !result.best) return false;
    auto selected = result.best->symbol;
    if (!selected || selected->isDeleted()) return false;
    if (!is_nothrow_function(selected)) return false;
    if (!is_nothrow_function(cls->destructor())) return false;
    for (const auto& conversion : result.best->conversions) {
      if (!is_nothrow_function(conversion.udc.function)) return false;
    }
    return true;
  }

  return false;
}

auto TypeTraits::is_trivially_constructible(
    const Type* type, std::span<const Type* const> argTypes) -> bool {
  if (!is_constructible(type, argTypes)) return false;
  auto unqual = remove_cv(type);
  if (is_reference(unqual) || is_scalar(unqual)) {
    if (argTypes.empty()) return true;
    return is_trivial_initialization(unqual, argTypes.front(), true);
  }
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    if (!has_trivial_destructor(unqual)) return false;
    auto constructor = selectConstructor(cls, argTypes);
    if (!constructor || isUserProvided(constructor)) return false;
    auto kind = constructorKind(cls, constructor);
    if (!kind) return false;
    return has_trivial_constructor(*this, cls, *kind);
  }
  if (is_array(unqual)) {
    return is_trivially_constructible(remove_all_extents(unqual), {});
  }
  return false;
}

auto TypeTraits::selectAssignmentOperator(const Type* to, const Type* from)
    -> FunctionSymbol* {
  if (!to || !from) return nullptr;

  auto makeOperand = [&](const Type* type) {
    auto valueCategory = ValueCategory::kXValue;
    if (is_lvalue_reference(type)) valueCategory = ValueCategory::kLValue;
    return ThisExpressionAST::create(unit_->arena(), valueCategory,
                                     remove_reference(type));
  };

  auto lhs = makeOperand(to);
  auto rhs = makeOperand(from);

  OverloadResolution resolution{unit_};
  auto selected = resolution.lookupOperator(lhs->type, TokenKind::T_EQUAL,
                                            rhs->type, lhs, rhs);

  if (resolution.wasLastLookupAmbiguous()) return nullptr;
  if (selected && selected->isDeleted()) return nullptr;
  if (!is_accessible_from_unrelated_context(selected)) return nullptr;
  return selected;
}

auto TypeTraits::is_assignable(const Type* to, const Type* from) -> bool {
  if (!to || !from) return false;

  auto targetType = remove_reference(to);
  auto target = remove_cv(targetType);

  if (auto classType = type_cast<ClassType>(target)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    return selectAssignmentOperator(to, from) != nullptr;
  }

  if (!is_lvalue_reference(to)) return false;
  if (is_const(targetType)) return false;
  if (!is_scalar(target)) return false;

  return is_convertible(remove_cvref(from), target);
}

auto TypeTraits::is_nothrow_assignable(const Type* to, const Type* from)
    -> bool {
  if (!is_assignable(to, from)) return false;

  if (!type_cast<ClassType>(remove_cvref(to))) {
    return is_nothrow_initialization(remove_reference(to), from, false);
  }

  auto selected = selectAssignmentOperator(to, from);
  if (!selected) return false;

  if (!is_nothrow_function(selected)) return false;
  auto assignmentType = type_cast<FunctionType>(selected->type());
  if (!assignmentType || assignmentType->parameterTypes().empty()) return true;
  return is_nothrow_initialization(assignmentType->parameterTypes().front(),
                                   from, false);
}

auto TypeTraits::is_trivially_assignable(const Type* to, const Type* from)
    -> bool {
  if (!is_assignable(to, from)) return false;
  auto unqual = remove_cvref(to);
  if (is_scalar(unqual))
    return is_trivial_initialization(remove_reference(to), from, false);
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    auto selected = selectAssignmentOperator(to, from);
    if (!selected) return false;
    if (selected == cls->copyAssignmentOperator())
      return has_trivial_assignment(*this, cls, TrivialAssignmentKind::kCopy);
    if (selected == cls->moveAssignmentOperator())
      return has_trivial_assignment(*this, cls, TrivialAssignmentKind::kMove);
    return false;
  }
  return false;
}

auto TypeTraits::is_trivially_copyable(const Type* type) -> bool {
  auto ty = remove_cv(remove_all_extents(type));
  if (is_scalar(ty)) return true;
  if (auto classType = type_cast<ClassType>(ty)) {
    return is_trivially_copyable_class(*this, classType->definition());
  }
  return false;
}

auto TypeTraits::is_abstract(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cvref(type));
  if (!classType) return false;
  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;
  return cls->isAbstract();
}

auto TypeTraits::is_destructible(const Type* type) -> bool {
  if (!type) return false;

  auto unqual = remove_cv(type);

  if (is_reference(unqual)) return true;
  if (is_void(unqual)) return false;
  if (is_function(unqual)) return false;
  if (is_unbounded_array(unqual)) return false;

  if (is_bounded_array(unqual))
    return is_destructible(remove_all_extents(unqual));

  if (is_scalar(unqual)) return true;

  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;

    auto dtor = cls->destructor();
    if (dtor && dtor->isDeleted()) return false;
    return is_accessible_from_unrelated_context(dtor);
  }

  if (is_enum(unqual)) return true;

  return false;
}

auto TypeTraits::is_nothrow_destructible(const Type* type) -> bool {
  if (!is_destructible(type)) return false;

  auto unqual = remove_cv(type);

  if (is_bounded_array(unqual))
    return is_nothrow_destructible(remove_all_extents(unqual));

  auto classType = type_cast<ClassType>(unqual);
  if (!classType) return true;

  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;

  auto destructor = cls->destructor();
  if (!destructor) return true;

  auto destructorType = type_cast<FunctionType>(destructor->type());
  return !destructorType || destructorType->isNoexcept();
}

auto TypeTraits::has_trivial_destructor(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_reference(unqual)) return true;
  if (is_void(unqual)) return false;
  if (is_function(unqual)) return false;
  if (is_unbounded_array(unqual)) return false;
  if (is_bounded_array(unqual))
    return is_trivially_destructible(remove_all_extents(unqual));
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    return is_trivially_destructible_class(*this, cls);
  }
  if (is_enum(unqual)) return true;
  return false;
}

auto TypeTraits::is_trivially_destructible(const Type* type) -> bool {
  if (!is_destructible(type)) return false;
  return has_trivial_destructor(type);
}

auto TypeTraits::has_virtual_destructor(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cvref(type));
  if (!classType) return false;
  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;
  return cls->hasVirtualDestructor();
}
}  // namespace cxx
