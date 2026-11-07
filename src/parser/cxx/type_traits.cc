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
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/external_name_encoder.h>
#include <cxx/initialization.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/standard_conversion.h>
#include <cxx/symbols.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
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
  std::vector<std::optional<std::uint32_t>> addressSpaces;
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
  auto operator()(const HalfType*) const -> bool { return true; }
  auto operator()(const Float16Type*) const -> bool { return true; }
  auto operator()(const Float128Type*) const -> bool { return true; }
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

  auto operator()(const ComplexType* type) const -> bool {
    return visit(IsFloatingPoint{}, type->elementType());
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
  auto operator()(const HalfType*) const -> bool { return true; }
  auto operator()(const Float16Type*) const -> bool { return true; }
  auto operator()(const Float128Type*) const -> bool { return true; }
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

struct IsVector {
  auto operator()(const VectorType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(auto) const -> bool { return false; }
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
  [[nodiscard]] auto operator()(const UnboundedArrayType*) const -> bool {
    return false;
  }
  [[nodiscard]] auto operator()(const BoundedArrayType* type) const -> bool {
    return visit(*this, type->elementType());
  }
  [[nodiscard]] auto operator()(const EnumType* type) const -> bool {
    auto symbol = type->symbol();
    return symbol->hasFixedUnderlyingType() || symbol->isDefined();
  }
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

struct AddTypeQualifiers {
  const TypeTraits& typeTraits;
  CvQualifiers qualifiers;
  std::optional<std::uint32_t> addressSpace;

  [[nodiscard]] auto control() const -> Control* {
    return typeTraits.control();
  }

  [[nodiscard]] auto operator()(const BoundedArrayType* type) const
      -> const Type* {
    auto elementType = visit(*this, type->elementType());
    return control()->getBoundedArrayType(elementType, type->size());
  }

  [[nodiscard]] auto operator()(const UnboundedArrayType* type) const
      -> const Type* {
    auto elementType = visit(*this, type->elementType());
    return control()->getUnboundedArrayType(elementType);
  }

  [[nodiscard]] auto operator()(const UnresolvedBoundedArrayType* type) const
      -> const Type* {
    auto elementType = visit(*this, type->elementType());
    return control()->getUnresolvedBoundedArrayType(type->translationUnit(),
                                                    elementType, type->size());
  }

  [[nodiscard]] auto operator()(const FunctionType* type) const -> const Type* {
    return type;
  }

  [[nodiscard]] auto operator()(const LvalueReferenceType* type) const
      -> const Type* {
    return type;
  }

  [[nodiscard]] auto operator()(const RvalueReferenceType* type) const
      -> const Type* {
    return type;
  }

  [[nodiscard]] auto operator()(auto type) const -> const Type* {
    return control()->getQualType(type, qualifiers, addressSpace);
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

  auto operator()(const VectorType* type) const -> const Type* {
    return type->elementType();
  }

  auto operator()(const ComplexType* type) const -> const Type* {
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
  bool removeAddressSpace = false;

  [[nodiscard]] auto control() const -> Control* {
    return typeTraits.control();
  }

  auto operator()(const QualType* type) const -> const Type* {
    auto remaining = type->cvQualifiers() & ~qualifiers;
    auto space = removeAddressSpace ? std::nullopt : type->addressSpace();
    if (remaining == type->cvQualifiers() && space == type->addressSpace())
      return type;
    if (remaining == CvQualifiers::kNone && !space) return type->elementType();
    return control()->getQualType(type->elementType(), remaining, space);
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

  auto operator()(const HalfType*, const HalfType*) const -> bool {
    return true;
  }

  auto operator()(const Float16Type*, const Float16Type*) const -> bool {
    return true;
  }

  auto operator()(const Float128Type*, const Float128Type*) const -> bool {
    return true;
  }

  auto operator()(const QualType* type, const QualType* otherType) const
      -> bool {
    if (type->cvQualifiers() != otherType->cvQualifiers()) return false;
    if (type->addressSpace() != otherType->addressSpace()) return false;
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

  auto operator()(const PackExpansionType* type,
                  const PackExpansionType* otherType) const -> bool {
    return typeTraits.is_same(type->pattern(), otherType->pattern());
  }

  auto operator()(const DecltypeType* type, const DecltypeType* otherType) const
      -> bool {
    return TemplateEquivalence{typeTraits.unit()}.same(type->expression(),
                                                       otherType->expression());
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
    if (!TemplateEquivalence{typeTraits.unit()}.same(
            type->exceptionSpecification(),
            otherType->exceptionSpecification()))
      return false;
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

  auto operator()(
      const TemplateTypeParameterSpecializationType* type,
      const TemplateTypeParameterSpecializationType* otherType) const -> bool {
    return type == otherType;
  }

  auto operator()(const UnresolvedNameType* type,
                  const UnresolvedNameType* otherType) const -> bool {
    return type == otherType ||
           TemplateEquivalence{typeTraits.unit()}.same(type, otherType);
  }

  auto operator()(const UnresolvedBoundedArrayType* type,
                  const UnresolvedBoundedArrayType* otherType) const -> bool {
    if (!typeTraits.is_same(type->elementType(), otherType->elementType()))
      return false;
    return TemplateEquivalence{typeTraits.unit()}.same(type->size(),
                                                       otherType->size());
  }

  auto operator()(const UnresolvedPackIndexType* type,
                  const UnresolvedPackIndexType* otherType) const -> bool {
    if (type->pack() != otherType->pack()) return false;
    return TemplateEquivalence{typeTraits.unit()}.same(type->index(),
                                                       otherType->index());
  }

  auto operator()(const UnresolvedUnderlyingType* type,
                  const UnresolvedUnderlyingType* otherType) const -> bool {
    return TemplateEquivalence{typeTraits.unit()}.same(type->typeId(),
                                                       otherType->typeId());
  }

  auto operator()(const UnresolvedBuiltinType* type,
                  const UnresolvedBuiltinType* otherType) const -> bool {
    if (type->builtinKind() != otherType->builtinKind()) return false;
    return TemplateEquivalence{typeTraits.unit()}.same(type->typeId(),
                                                       otherType->typeId());
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

  auto operator()(const SveType* type, const SveType* otherType) const -> bool {
    return type->sveKind() == otherType->sveKind();
  }

  auto operator()(const UnsignedBitIntType* type,
                  const UnsignedBitIntType* otherType) const -> bool {
    return type->numBits() == otherType->numBits();
  }

  auto operator()(const UnresolvedBitIntType* type,
                  const UnresolvedBitIntType* otherType) const -> bool {
    if (type->isUnsigned() != otherType->isUnsigned()) return false;
    return TemplateEquivalence{typeTraits.unit()}.same(
        type->sizeExpression(), otherType->sizeExpression());
  }

  auto operator()(const VectorType* type, const VectorType* otherType) const
      -> bool {
    return type->elementCount() == otherType->elementCount() &&
           type->vectorKind() == otherType->vectorKind() &&
           typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const UnresolvedVectorType* type,
                  const UnresolvedVectorType* otherType) const -> bool {
    if (type->vectorKind() != otherType->vectorKind()) return false;
    if (type->sizeKind() != otherType->sizeKind()) return false;
    if (!typeTraits.is_same(type->elementType(), otherType->elementType()))
      return false;
    return TemplateEquivalence{typeTraits.unit()}.same(
        type->sizeExpression(), otherType->sizeExpression());
  }

  auto operator()(const ComplexType* type, const ComplexType* otherType) const
      -> bool {
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }

  auto operator()(const AtomicType* type, const AtomicType* otherType) const
      -> bool {
    return typeTraits.is_same(type->elementType(), otherType->elementType());
  }
};

enum class TrivialCopyKind { kCopyable, kForCalls, kLayout };

[[nodiscard]] auto mostConstrainedSpecialMembers(
    TranslationUnit* unit, std::vector<FunctionSymbol*> members)
    -> std::vector<FunctionSymbol*> {
  std::erase_if(members, [unit](FunctionSymbol* member) {
    return ASTRewriter::evaluateAssociatedConstraints(unit, member) == false;
  });

  std::vector<FunctionSymbol*> eligible;
  for (auto member : members) {
    const auto hasMoreConstrained =
        std::ranges::any_of(members, [unit, member](FunctionSymbol* other) {
          auto cls = symbol_cast<ClassSymbol>(member->parent());
          if (cls && cls->isDefaultConstructor(member) &&
              cls->isDefaultConstructor(other))
            return ASTRewriter::isMoreConstrained(unit, other, member);
          return ASTRewriter::isMorePartialOrderingConstrained(unit, other,
                                                               member);
        });
    if (!hasMoreConstrained) eligible.push_back(member);
  }
  return eligible;
}

[[nodiscard]] auto eligibleSpecialMembers(TranslationUnit* unit,
                                          std::vector<FunctionSymbol*> members)
    -> std::vector<FunctionSymbol*> {
  auto eligible = mostConstrainedSpecialMembers(unit, std::move(members));
  std::erase_if(eligible, [unit](FunctionSymbol* member) {
    return unit->typeTraits().is_deleted_function(member);
  });
  return eligible;
}

[[nodiscard]] auto specialMembersOfKind(
    ClassSymbol* cls, std::vector<FunctionSymbol*> candidates,
    bool (ClassSymbol::*isKind)(FunctionSymbol*) const)
    -> std::vector<FunctionSymbol*> {
  std::erase_if(candidates, [cls, isKind](FunctionSymbol* candidate) {
    return !(cls->*isKind)(candidate);
  });
  return candidates;
}

[[nodiscard]] auto declaredAssignmentOperators(ClassSymbol* cls)
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> operators;
  for (auto function : cls->find(TokenKind::T_EQUAL) | views::member_functions)
    operators.push_back(function);
  return operators;
}

[[nodiscard]] auto hasUserProvidedEligibleMember(
    TranslationUnit* unit, std::vector<FunctionSymbol*> members) -> bool {
  return std::ranges::any_of(eligibleSpecialMembers(unit, std::move(members)),
                             [](FunctionSymbol* function) {
                               return TypeTraits::is_user_provided(function);
                             });
}

auto has_trivial_copy_members(TypeTraits& traits, ClassSymbol* cls,
                              TrivialCopyKind kind) -> bool {
  if (!cls || !cls->isComplete()) return false;

  auto unit = traits.unit();
  auto destructors = cls->prospectiveDestructors();
  auto eligibleDestructors = kind == TrivialCopyKind::kLayout
                                 ? destructors
                                 : eligibleSpecialMembers(unit, destructors);
  if (!destructors.empty() && eligibleDestructors.empty()) return false;
  for (auto dtor : eligibleDestructors) {
    if (traits.is_user_provided(dtor)) return false;
    if (dtor->isVirtual()) return false;
  }

  auto declaredConstructors = cls->declaredConstructors();
  if (hasUserProvidedEligibleMember(
          unit, specialMembersOfKind(cls, declaredConstructors,
                                     &ClassSymbol::isCopyConstructor)))
    return false;
  if (hasUserProvidedEligibleMember(
          unit, specialMembersOfKind(cls, declaredConstructors,
                                     &ClassSymbol::isMoveConstructor)))
    return false;

  if (kind != TrivialCopyKind::kForCalls) {
    auto assignments = declaredAssignmentOperators(cls);
    if (hasUserProvidedEligibleMember(
            unit, specialMembersOfKind(cls, assignments,
                                       &ClassSymbol::isCopyAssignmentOperator)))
      return false;
    if (hasUserProvidedEligibleMember(
            unit, specialMembersOfKind(cls, assignments,
                                       &ClassSymbol::isMoveAssignmentOperator)))
      return false;
  }

  if (cls->isPolymorphic()) return false;
  if (cls->hasVirtualBaseClasses()) return false;

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (!has_trivial_copy_members(traits, baseClass, kind)) return false;
  }

  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldType = traits.remove_all_extents(traits.remove_cv(field->type()));
    if (auto ct = type_cast<ClassType>(fieldType)) {
      if (!has_trivial_copy_members(traits, ct->symbol(), kind)) return false;
    }
  }

  return true;
}

auto has_non_deleted_copy_or_move_constructor(TranslationUnit* unit,
                                              ClassSymbol* cls) -> bool {
  auto isCopyOrMove = [cls](FunctionSymbol* constructor) {
    return cls->isCopyConstructor(constructor) ||
           cls->isMoveConstructor(constructor);
  };
  auto copyOrMove =
      cls->declaredConstructors() | std::views::filter(isCopyOrMove);
  if (std::ranges::empty(copyOrMove)) return true;
  return std::ranges::any_of(copyOrMove, [unit](FunctionSymbol* constructor) {
    return !unit->typeTraits().is_deleted_function(constructor);
  });
}

auto is_trivially_copyable_class(TypeTraits& traits, ClassSymbol* cls) -> bool {
  return has_trivial_copy_members(traits, cls, TrivialCopyKind::kCopyable);
}

[[nodiscard]] auto is_trivially_destructible_class(TypeTraits& traits,
                                                   ClassSymbol* cls) -> bool {
  if (!cls || !cls->isComplete()) return false;

  auto dtor = cls->destructor();
  if (traits.is_user_provided(dtor)) return false;
  if (dtor && dtor->isVirtual()) return false;
  if (cls->isUnion()) return true;

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

template <typename Predicate>
[[nodiscard]] auto allSubobjectClasses(TypeTraits& traits, ClassSymbol* cls,
                                       Predicate predicate) -> bool {
  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (!predicate(baseClass->resolvedDefinition())) return false;
  }
  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldType = traits.remove_all_extents(traits.remove_cv(field->type()));
    auto classType = type_cast<ClassType>(fieldType);
    if (!classType) continue;
    if (!predicate(classType->definition())) return false;
  }
  return true;
}

enum class TrivialConstructorKind { kDefault, kCopy, kMove };

[[nodiscard]] auto constructorKind(ClassSymbol* cls,
                                   FunctionSymbol* constructor)
    -> std::optional<TrivialConstructorKind> {
  if (cls->isDefaultConstructor(constructor))
    return TrivialConstructorKind::kDefault;
  if (cls->isCopyConstructor(constructor)) return TrivialConstructorKind::kCopy;
  if (cls->isMoveConstructor(constructor)) return TrivialConstructorKind::kMove;
  return std::nullopt;
}

auto declaredMember(ClassSymbol* cls, TrivialConstructorKind kind)
    -> FunctionSymbol* {
  if (kind == TrivialConstructorKind::kDefault)
    return cls->defaultConstructor();
  if (kind == TrivialConstructorKind::kCopy) return cls->copyConstructor();
  if (auto move = cls->moveConstructor()) return move;
  return cls->copyConstructor();
}

template <typename Kind>
[[nodiscard]] auto memberFor(TranslationUnit* unit, ClassSymbol* cls, Kind kind)
    -> FunctionSymbol*;

[[nodiscard]] auto has_trivial_subobject_constructor(
    TypeTraits& traits, const Type* type, TrivialConstructorKind kind,
    CvQualifiers sourceCv, bool allowDeleted) -> bool;

[[nodiscard]] auto has_trivial_constructor(
    TypeTraits& traits, ClassSymbol* cls, TrivialConstructorKind kind,
    bool allowDeleted = false, FunctionSymbol* constructor = nullptr) -> bool {
  if (!cls || !cls->isComplete()) return false;
  if (!constructor) constructor = memberFor(traits.unit(), cls, kind);
  if (!constructor ||
      (!allowDeleted &&
       traits.unit()->typeTraits().is_deleted_function(constructor)))
    return false;
  if (traits.is_user_provided(constructor)) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  if (kind == TrivialConstructorKind::kDefault &&
      std::ranges::any_of(cls->members() | views::non_static_fields,
                          &FieldSymbol::hasInitializer))
    return false;
  if (cls->isUnion()) return true;
  auto sourceCv = CvQualifiers::kNone;
  if (kind != TrivialConstructorKind::kDefault) {
    auto functionType = type_cast<FunctionType>(constructor->type());
    sourceCv = cv_qualifiers(
        traits.remove_reference(functionType->parameterTypes().front()));
  }
  for (auto base : cls->baseClasses()) {
    auto baseClass = resolved_base_class(base);
    if (!baseClass) return false;
    if (!has_trivial_subobject_constructor(traits, baseClass->type(), kind,
                                           sourceCv, allowDeleted))
      return false;
  }
  for (auto field : cls->members() | views::non_static_fields) {
    auto fieldCv = sourceCv;
    if (field->isMutable()) fieldCv &= ~CvQualifiers::kConst;
    if (!has_trivial_subobject_constructor(traits, field->type(), kind, fieldCv,
                                           allowDeleted))
      return false;
  }
  return true;
}

[[nodiscard]] auto has_trivial_subobject_constructor(
    TypeTraits& traits, const Type* type, TrivialConstructorKind kind,
    CvQualifiers sourceCv, bool allowDeleted) -> bool {
  type = traits.remove_all_extents(type);
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return true;
  auto cls = classType->definition();
  FunctionSymbol* selected = nullptr;
  if (kind == TrivialConstructorKind::kDefault) {
    selected = traits.select_constructor(cls, {});
  } else {
    auto source = traits.add_cv(type, sourceCv);
    auto argument = kind == TrivialConstructorKind::kCopy
                        ? traits.add_lvalue_reference(source)
                        : traits.add_rvalue_reference(source);
    selected = traits.select_constructor(cls, std::array{argument});
  }
  if (!selected) return false;
  auto selectedKind = constructorKind(cls, selected);
  if (!selectedKind) return false;
  return has_trivial_constructor(traits, cls, *selectedKind, allowDeleted,
                                 selected);
}

enum class TrivialAssignmentKind { kCopy, kMove };

auto assignmentKind(ClassSymbol* cls, FunctionSymbol* assignment)
    -> std::optional<TrivialAssignmentKind> {
  if (assignment == cls->copyAssignmentOperator())
    return TrivialAssignmentKind::kCopy;
  if (assignment == cls->moveAssignmentOperator())
    return TrivialAssignmentKind::kMove;
  return std::nullopt;
}

auto declaredMember(ClassSymbol* cls, TrivialAssignmentKind kind)
    -> FunctionSymbol* {
  if (kind == TrivialAssignmentKind::kCopy)
    return cls->copyAssignmentOperator();
  if (auto move = cls->moveAssignmentOperator()) return move;
  return cls->copyAssignmentOperator();
}

template <typename Kind>
auto memberFor(TranslationUnit* unit, ClassSymbol* cls, Kind kind)
    -> FunctionSymbol* {
  auto member = declaredMember(cls, kind);
  if (!unit->typeTraits().is_excluded_defaulted_move_member(member))
    return member;
  return declaredMember(cls, Kind::kCopy);
}

auto has_trivial_assignment(TypeTraits& traits, ClassSymbol* cls,
                            TrivialAssignmentKind kind) -> bool {
  if (!cls || !cls->isComplete()) return false;
  auto assignment = memberFor(traits.unit(), cls, kind);
  if (!assignment ||
      traits.unit()->typeTraits().is_deleted_function(assignment))
    return false;
  if (traits.is_user_provided(assignment)) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  if (cls->isUnion()) return true;
  return allSubobjectClasses(traits, cls, [&](ClassSymbol* subobject) {
    return has_trivial_assignment(traits, subobject, kind);
  });
}

[[nodiscard]] auto is_abi_pod(TypeTraits& traits, const Type* type,
                              AbiPodDefinition definition) -> bool;

[[nodiscard]] auto is_cxx11_abi_pod_class(TypeTraits& traits, ClassSymbol* cls)
    -> bool {
  if (!has_trivial_constructor(traits, cls, TrivialConstructorKind::kDefault,
                               true))
    return false;
  if (!has_trivial_copy_members(traits, cls, TrivialCopyKind::kLayout))
    return false;
  if (!traits.standard_layout(cls->type(),
                              TypeTraits::StandardLayoutRule::kLayout))
    return false;
  return std::ranges::all_of(
      cls->members() | views::non_static_fields, [&](FieldSymbol* field) {
        return is_abi_pod(traits, field->type(), AbiPodDefinition::kCxx11);
      });
}

[[nodiscard]] auto declaresSpecialMember(FunctionSymbol* function,
                                         AbiPodDefinition definition) -> bool {
  if (definition == AbiPodDefinition::kCxx03UserDeclared)
    return !function->isImplicitlyDeclared();
  return TypeTraits::is_user_provided(function);
}

[[nodiscard]] auto declaresPodAssignment(ClassSymbol* cls,
                                         FunctionSymbol* assignment,
                                         AbiPodDefinition definition) -> bool {
  if (!declaresSpecialMember(assignment, definition)) return false;
  if (cls->isCopyAssignmentOperator(assignment)) return true;
  if (definition != AbiPodDefinition::kCxx03UserDeclared) return false;
  return cls->isMoveAssignmentOperator(assignment);
}

[[nodiscard]] auto is_cxx03_abi_pod_field(TypeTraits& traits,
                                          FieldSymbol* field,
                                          AbiPodDefinition definition) -> bool {
  if (field->isBitField() && !field->name()) return true;
  if (field->hasInitializer()) return false;
  if (field->accessSpecifier() != AccessSpecifier::kPublic) return false;
  return is_abi_pod(traits, field->type(), definition);
}

[[nodiscard]] auto is_cxx03_abi_pod_class(TypeTraits& traits, ClassSymbol* cls,
                                          AbiPodDefinition definition) -> bool {
  if (!cls->baseClasses().empty()) return false;
  if (cls->hasVirtualFunctions()) return false;
  auto declares = [definition](FunctionSymbol* function) {
    return declaresSpecialMember(function, definition);
  };
  if (std::ranges::any_of(cls->declaredConstructors(), declares)) return false;
  if (std::ranges::any_of(cls->prospectiveDestructors(), declares))
    return false;
  for (auto assignment : declaredAssignmentOperators(cls)) {
    if (declaresPodAssignment(cls, assignment, definition)) return false;
  }
  return std::ranges::all_of(
      cls->members() | views::non_static_fields, [&](FieldSymbol* field) {
        return is_cxx03_abi_pod_field(traits, field, definition);
      });
}

auto is_abi_pod(TypeTraits& traits, const Type* type,
                AbiPodDefinition definition) -> bool {
  auto unqual = traits.remove_cv(traits.remove_all_extents(type));
  if (traits.is_scalar_or_vector(unqual)) return true;
  auto classType = type_cast<ClassType>(unqual);
  if (!classType) return false;
  auto cls = classType->definition();
  if (!cls || !cls->isComplete()) return false;
  if (definition == AbiPodDefinition::kCxx11)
    return is_cxx11_abi_pod_class(traits, cls);
  return is_cxx03_abi_pod_class(traits, cls, definition);
}

template <typename Kind>
[[nodiscard]] auto copiesAddressSpaceGenerically(TypeTraits& traits,
                                                 ClassSymbol* cls, Kind kind)
    -> bool {
  auto member = cls ? declaredMember(cls, kind) : nullptr;
  if (!member || !member->isImplicitlyDeclared()) return false;
  return allSubobjectClasses(traits, cls, [&](ClassSymbol* subobject) {
    return copiesAddressSpaceGenerically(traits, subobject, kind);
  });
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

auto has_base_of_first_member_type(TypeTraits& traits, ClassSymbol* cls)
    -> bool {
  FieldSymbol* firstField = nullptr;
  for (auto field : cls->members() | views::non_static_fields) {
    firstField = field;
    break;
  }

  if (!firstField) return false;

  auto fieldClass = type_cast<ClassType>(traits.remove_cv(firstField->type()));
  if (!fieldClass || !fieldClass->symbol()) return false;

  auto fieldSymbol = fieldClass->symbol()->resolvedDefinition();

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (baseClass->resolvedDefinition() == fieldSymbol) return true;
  }

  return false;
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
    if (field->isBitField() && !field->name()) continue;
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

auto TypeTraits::is_user_provided(FunctionSymbol* fn) -> bool {
  if (!fn) return false;
  fn = fn->canonical();
  return !fn->isImplicitlyDeclared() && !fn->isDefaulted() && !fn->isDeleted();
}

TypeTraits::TypeTraits(TranslationUnit* unit) : unit_(unit) {}

auto TypeTraits::control() const -> Control* { return unit_->control(); }

auto TypeTraits::is_void(const Type* type) const -> bool {
  return unqualified_cast<VoidType>(type) != nullptr;
}

auto TypeTraits::is_bool(const Type* type) const -> bool {
  return unqualified_cast<BoolType>(type) != nullptr;
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
  return is_class_or_union(type) && !is_union(type);
}

auto TypeTraits::is_function(const Type* type) const -> bool {
  return type && visit(IsFunction{}, type);
}

auto TypeTraits::is_pointer(const Type* type) const -> bool {
  return type && visit(IsPointer{}, type);
}

auto TypeTraits::is_vector(const Type* type) const -> bool {
  return type && visit(IsVector{}, type);
}

auto TypeTraits::is_device_value_type(const Type* type) const -> bool {
  auto value = unqualified_type(type);
  if (auto vector = type_cast<VectorType>(value)) {
    if (vector->elementCount() < 2 || vector->elementCount() > 4) return false;
    value = unqualified_type(vector->elementType());
  }
  return value == control()->getFloatType() ||
         value == control()->getIntType() ||
         value == control()->getUnsignedIntType();
}

auto TypeTraits::is_vector_element_type(const Type* type, VectorKind kind) const
    -> bool {
  if (is_floating_point(type)) return true;
  if (!is_integral(type)) return false;
  if (type->kind() == TypeKind::kBool) return kind == VectorKind::kExt;
  return true;
}

auto TypeTraits::is_scalar_or_vector(const Type* type) const -> bool {
  return is_scalar(type) || is_vector(type);
}

auto TypeTraits::vector_width_in_bytes(const Type* type) const -> std::size_t {
  auto vectorType = unqualified_cast<VectorType>(type);
  if (!vectorType) return 0;
  auto elementSize =
      control()->memoryLayout()->sizeOf(vectorType->elementType());
  if (!elementSize) return 0;
  return vectorType->elementCount() * *elementSize;
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

auto TypeTraits::select_destructor(ClassSymbol* cls) const -> FunctionSymbol* {
  auto destructors =
      mostConstrainedSpecialMembers(unit_, cls->prospectiveDestructors());
  if (destructors.size() != 1) return nullptr;
  return destructors.front();
}

auto TypeTraits::is_complete(const Type* type) const -> bool {
  return type && visit(IsComplete{}, type);
}

auto TypeTraits::is_integral_or_unscoped_enum(const Type* type) const -> bool {
  return is_integral(type) || (is_enum(type) && !is_scoped_enum(type));
}

auto TypeTraits::is_integral_or_enum(const Type* type) const -> bool {
  return is_integral(type) || is_enum(type);
}

auto TypeTraits::is_fundamental(const Type* type) const -> bool {
  return is_arithmetic(type) || is_void(type) || is_null_pointer(type) ||
         unqualified_cast<BuiltinMetaInfoType>(type);
}

auto TypeTraits::is_arithmetic(const Type* type) const -> bool {
  return is_integral(type) || is_floating_point(type) || is_complex(type);
}

auto TypeTraits::arithmetic_types() const -> std::vector<const Type*> {
#define PROCESS_TYPE(K) control()->get##K##Type(),
  return {CXX_FOR_EACH_ARITHMETIC_TYPE_KIND(PROCESS_TYPE)};
#undef PROCESS_TYPE
}

auto TypeTraits::is_floating(const Type* type) const -> bool {
  return is_floating_point(type) ||
         (is_complex(type) && is_floating_point(complex_element_type(type)));
}

auto TypeTraits::is_scalar(const Type* type) const -> bool {
  return is_arithmetic(type) || is_enum(type) || is_pointer(type) ||
         is_member_pointer(type) || is_null_pointer(type) ||
         unqualified_cast<BuiltinMetaInfoType>(type);
}

auto TypeTraits::is_object(const Type* type) const -> bool {
  if (!type) return false;
  return !is_function(type) && !is_reference(type) && !is_void(type);
}

auto TypeTraits::is_compound(const Type* type) const -> bool {
  return is_array(type) || is_function(type) || is_pointer(type) ||
         is_reference(type) || is_class_or_union(type) || is_enum(type) ||
         is_member_pointer(type);
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

auto TypeTraits::is_atomic(const Type* type) const -> bool {
  return type_cast<AtomicType>(unqualified_type(type)) != nullptr;
}

auto TypeTraits::is_complex(const Type* type) const -> bool {
  return type_cast<ComplexType>(remove_cv(type)) != nullptr;
}

auto TypeTraits::complex_element_type(const Type* type) const -> const Type* {
  if (auto complexType = type_cast<ComplexType>(remove_cv(type)))
    return complexType->elementType();
  return type;
}

auto TypeTraits::is_signed(const Type* type) const -> bool {
  return type &&
         visit(IsSigned{control()->memoryLayout()->isWideCharSigned()}, type);
}

auto TypeTraits::is_unsigned(const Type* type) const -> bool {
  return type &&
         visit(IsUnsigned{control()->memoryLayout()->isWideCharSigned()}, type);
}

auto TypeTraits::value_representation(const Type* type) const
    -> std::optional<IntegralRepresentation> {
  if (!type) return std::nullopt;

  auto unqualifiedType = remove_cv(type);

  if (is_bool(unqualifiedType)) return IntegralRepresentation{1, false};

  if (auto enumType = type_cast<EnumType>(unqualifiedType);
      enumType && enumType->symbol() &&
      !enumType->symbol()->hasFixedUnderlyingType())
    return integral_representation(promoted_integer_type(enumType));

  return integral_representation(unqualifiedType);
}

auto TypeTraits::integral_constant(const Type* type, ConstInt::Wide value) const
    -> std::optional<ConstInt> {
  auto representation = integral_representation(type);
  if (!representation) return std::nullopt;
  return ConstInt::make(value, representation->bits, representation->isSigned);
}

auto TypeTraits::converted_integral_constant(const Type* type,
                                             const ConstInt& value) const
    -> std::optional<ConstInt> {
  auto converted = integral_constant(type, value.toWideValue());
  if (!converted) return std::nullopt;
  if (*converted != value) return std::nullopt;
  return converted;
}

auto TypeTraits::converted_constant_value(const Type* type,
                                          const ConstValue& value) const
    -> std::optional<ConstValue> {
  auto number = std::get_if<ConstInt>(&value);
  if (!number) return value;
  if (!is_integral_or_enum(type)) return value;
  auto converted = converted_integral_constant(type, *number);
  if (!converted) return std::nullopt;
  return ConstValue{*converted};
}

auto TypeTraits::integral_representation(const Type* type) const
    -> std::optional<IntegralRepresentation> {
  if (!type) return std::nullopt;

  auto unqualifiedType = remove_cv(type);

  if (auto enumType = type_cast<EnumType>(unqualifiedType))
    return integral_representation(enumType->underlyingType());

  if (auto scopedEnumType = type_cast<ScopedEnumType>(unqualifiedType))
    return integral_representation(scopedEnumType->underlyingType());

  if (auto bitIntType = type_cast<BitIntType>(unqualifiedType))
    return IntegralRepresentation{bitIntType->numBits(), true};

  if (auto bitIntType = type_cast<UnsignedBitIntType>(unqualifiedType))
    return IntegralRepresentation{bitIntType->numBits(), false};

  if (!is_integral(unqualifiedType)) return std::nullopt;

  auto size = control()->memoryLayout()->sizeOf(unqualifiedType);
  if (!size) return std::nullopt;

  return IntegralRepresentation{static_cast<int>(*size) * 8,
                                is_signed(unqualifiedType)};
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
                                          Symbol* member) -> bool {
  if (!member || !objectType) return false;
  auto parent = member->parent();
  if (symbol_cast<EnumeratorSymbol>(member)) {
    if (auto enumeration = symbol_cast<EnumSymbol>(parent))
      parent = enumeration->parent();
  }
  auto memberClass = symbol_cast<ClassSymbol>(parent);
  auto objectClass = unqualified_cast<ClassType>(objectType);
  if (!objectClass || !objectClass->symbol()) return false;
  for (; memberClass;
       memberClass = symbol_cast<ClassSymbol>(memberClass->parent())) {
    if (memberClass->resolvedDefinition() ==
            objectClass->symbol()->resolvedDefinition() ||
        is_base_of(memberClass->type(), objectType))
      return true;
    auto field = symbol_cast<FieldSymbol>(member);
    if (!field || field->isStatic() || memberClass->name()) break;
  }
  return false;
}

auto TypeTraits::underlying_type(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(UnderlyingType{}, remove_cv(type));
}

auto TypeTraits::size_of(const Type* type) -> std::optional<std::size_t> {
  return control()->memoryLayout()->sizeOf(require_layout_type(type));
}

auto TypeTraits::alignment_of(const Type* type) -> std::optional<std::size_t> {
  return control()->memoryLayout()->alignmentOf(require_layout_type(type));
}

auto TypeTraits::require_layout_type(const Type* type) -> const Type* {
  if (!type) return nullptr;
  auto objectType = remove_reference(type);
  if (unit_ && isDependent(unit_, objectType)) return nullptr;
  auto elementType = remove_cv(remove_all_extents(objectType));
  if (auto elementClass = type_cast<ClassType>(elementType)) {
    if (!require_complete_class(elementClass->symbol())) return nullptr;
  }
  return objectType;
}

auto TypeTraits::overload_set_operand(const Type* type) const
    -> OverloadSetOperand {
  auto unreferenced = remove_reference(type);
  if (auto overloadSet = type_cast<OverloadSetType>(unreferenced))
    return {overloadSet, false};
  auto pointer = type_cast<PointerType>(unreferenced);
  if (!pointer) return {};
  auto overloadSet = type_cast<OverloadSetType>(pointer->elementType());
  if (!overloadSet) return {};
  return {overloadSet, true};
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

  if (auto lowered = loweredSplice(expr); lowered != expr) {
    if (!ast_cast<IdExpressionAST>(lowered) &&
        !ast_cast<MemberExpressionAST>(lowered))
      return expr->type;
    expr = lowered;
  }

  auto namedSymbol = [&]() -> Symbol* {
    if (auto id = ast_cast<IdExpressionAST>(expr)) return id->symbol;
    if (auto member = ast_cast<MemberExpressionAST>(expr))
      return member->symbol;
    return nullptr;
  }();

  if (symbol_cast<OverloadSetSymbol>(namedSymbol)) return expr->type;
  if (auto parameter = symbol_cast<NonTypeParameterSymbol>(namedSymbol))
    return constant_template_parameter_type(parameter);
  if (auto variable = symbol_cast<VariableSymbol>(namedSymbol);
      variable && variable->isTemplateParameterObject())
    return remove_cv(variable->type());
  if (namedSymbol) return namedSymbol->type();

  if (!expr->type) return nullptr;

  if (unit_->language() != LanguageKind::kCXX) return expr->type;

  if (is_lvalue(expr)) return add_lvalue_reference(expr->type);
  if (is_xvalue(expr)) return add_rvalue_reference(expr->type);
  return expr->type;
}

auto TypeTraits::constant_template_parameter_type(
    NonTypeParameterSymbol* parameter) const -> const Type* {
  auto type = parameter->objectType();
  if (type && containsPlaceholderType(type))
    return control()->getDependentType();
  return type;
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

auto TypeTraits::add_atomic(const Type* type) const -> const Type* {
  if (!type) return type;
  if (type_cast<AtomicType>(type)) return type;
  return control()->getAtomicType(type);
}

auto TypeTraits::remove_atomic(const Type* type) const -> const Type* {
  if (auto atomicType = type_cast<AtomicType>(unqualified_type(type)))
    return atomicType->elementType();
  return type;
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

auto TypeTraits::address_of_function(FunctionSymbol* function) const
    -> const Type* {
  if (!function->isImplicitObjectMemberFunction())
    return add_pointer(function->type());
  auto classType = type_cast<ClassType>(function->parent()->type());
  auto functionType = type_cast<FunctionType>(function->type());
  return control()->getMemberFunctionPointerType(classType, functionType);
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

auto TypeTraits::unwrap_reference(const Type* type) const -> const Type* {
  auto classType = type_cast<ClassType>(type);
  if (!classType || !is_std_type(type, WellKnownName::T_REFERENCE_WRAPPER))
    return type;
  auto symbol = classType->symbol();
  if (!symbol->isSpecialization()) return type;
  auto arguments = template_arguments_of(symbol);
  if (arguments.size() != 1) return type;
  auto referent = template_argument_as_type(arguments.front());
  if (!referent) return type;
  return add_lvalue_reference(referent);
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
  if (auto bitInt = type_cast<BitIntType>(type)) {
    if (isUnsigned) return control()->getUnsignedBitIntType(bitInt->numBits());
    return type;
  }
  if (auto bitInt = type_cast<UnsignedBitIntType>(type)) {
    if (!isUnsigned) return control()->getBitIntType(bitInt->numBits());
    return type;
  }

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

  auto result = corresponding_integer_type(unqualified_type(type), isUnsigned);
  if (!result) return type;

  return add_qualifiers(result, cv_qualifiers(type), address_space(type));
}

auto TypeTraits::is_default_constructible(const Type* type) -> bool {
  return is_constructible(type, {});
}

auto TypeTraits::is_copy_constructible(const Type* type) -> bool {
  return is_constructible(type, std::array{add_const_ref(type)});
}

auto TypeTraits::is_move_constructible(const Type* type) -> bool {
  return is_constructible(type, std::array{add_rvalue_reference(type)});
}

auto TypeTraits::is_trivially_default_constructible(const Type* type) -> bool {
  return is_trivially_constructible(type, {});
}

auto TypeTraits::is_trivially_copy_constructible(const Type* type) -> bool {
  return is_trivially_constructible(type, std::array{add_const_ref(type)});
}

auto TypeTraits::is_trivially_move_constructible(const Type* type) -> bool {
  return is_trivially_constructible(type,
                                    std::array{add_rvalue_reference(type)});
}

auto TypeTraits::is_nothrow_default_constructible(const Type* type) -> bool {
  return is_nothrow_constructible(type, {});
}

auto TypeTraits::is_nothrow_copy_constructible(const Type* type) -> bool {
  return is_nothrow_constructible(type, std::array{add_const_ref(type)});
}

auto TypeTraits::is_nothrow_move_constructible(const Type* type) -> bool {
  return is_nothrow_constructible(type, std::array{add_rvalue_reference(type)});
}

auto TypeTraits::is_copy_assignable(const Type* type) -> bool {
  return is_assignable(add_lvalue_reference(type), add_const_ref(type));
}

auto TypeTraits::is_move_assignable(const Type* type) -> bool {
  return is_assignable(add_lvalue_reference(type), add_rvalue_reference(type));
}

auto TypeTraits::is_trivially_copy_assignable(const Type* type) -> bool {
  return is_trivially_assignable(add_lvalue_reference(type),
                                 add_const_ref(type));
}

auto TypeTraits::is_trivially_move_assignable(const Type* type) -> bool {
  return is_trivially_assignable(add_lvalue_reference(type),
                                 add_rvalue_reference(type));
}

auto TypeTraits::is_nothrow_copy_assignable(const Type* type) -> bool {
  return is_nothrow_assignable(add_lvalue_reference(type), add_const_ref(type));
}

auto TypeTraits::is_nothrow_move_assignable(const Type* type) -> bool {
  return is_nothrow_assignable(add_lvalue_reference(type),
                               add_rvalue_reference(type));
}

auto TypeTraits::is_reflection(const Type* type) const -> bool {
  return unqualified_cast<BuiltinMetaInfoType>(type) != nullptr;
}

auto TypeTraits::rank(const Type* type) const -> std::intmax_t {
  std::intmax_t result = 0;
  for (; is_array(type); type = remove_extent(type)) ++result;
  return result;
}

auto TypeTraits::extent(const Type* type, std::uintmax_t dimension) const
    -> std::size_t {
  for (; dimension && is_array(type); --dimension) type = remove_extent(type);
  auto array = unqualified_cast<BoundedArrayType>(type);
  return !dimension && array ? array->size() : 0;
}

auto TypeTraits::unwrap_ref_decay(const Type* type) const -> const Type* {
  return unwrap_reference(decay(type));
}

auto TypeTraits::is_class_or_union(const Type* type) const -> bool {
  return type && visit(IsClass{}, type);
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

namespace {

[[nodiscard]] auto representsAllValues(
    const TypeTraits::IntegralRepresentation& target,
    const TypeTraits::IntegralRepresentation& source) -> bool {
  if (source.isSigned == target.isSigned) return source.bits <= target.bits;
  if (target.isSigned) return source.bits < target.bits;
  return false;
}

}  // namespace

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
    auto source = value_representation(from);
    auto target = value_representation(to);
    if (source && target) return !representsAllValues(*target, *source);
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
      if (constant->constValue) return expr;
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
  if (isDependent(unit_, source)) return false;

  targetType = remove_reference(targetType);

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

    auto format = control()->memoryLayout()->floatingPointFormat(targetType);
    return format && format->representsInteger(value);
  };

  auto fitsFloating = [&](double value) {
    auto format = control()->memoryLayout()->floatingPointFormat(targetType);
    return format && format->rangeContains(value);
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

  if (auto intValue = std::get_if<ConstInt>(&*value))
    return !fitsInteger(intValue->toIntMax());
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
  if (type_cast<BoolType>(remove_cv(targetType))) return value <= 1;

  auto representation = integral_representation(targetType);
  if (!representation) return false;

  const auto valueBits =
      representation->bits - (representation->isSigned ? 1 : 0);

  if (valueBits <= 0) return value == 0;
  if (valueBits >= 64) return true;
  return value < (std::uint64_t{1} << valueBits);
}

auto TypeTraits::is_std_namespace(Symbol* symbol) const -> bool {
  auto ns = symbol_cast<NamespaceSymbol>(symbol);
  if (!ns) return false;
  if (ns->isInline()) return is_std_namespace(ns->enclosingNamespace());
  if (!is_global_namespace(ns->parent())) return false;
  return well_known_name(ns->name()) == WellKnownName::T_STD;
}

auto TypeTraits::is_in_std_namespace(Symbol* symbol) const -> bool {
  return symbol && is_std_namespace(symbol->enclosingNamespace());
}

auto TypeTraits::standard_class_specialization(
    WellKnownName name, List<TemplateArgumentAST*>* arguments,
    SourceLocation location) -> ClassSymbol* {
  auto primary = lookupStandardLibraryType(unit_, name);
  if (!primary) return nullptr;
  auto instance = ASTRewriter::instantiate(unit_, arguments, primary, location);
  auto instanceClass = symbol_cast<ClassSymbol>(instance);
  if (!instanceClass) return nullptr;
  (void)require_complete_class(instanceClass);
  return symbol_cast<ClassSymbol>(instanceClass->resolvedDefinition());
}

auto TypeTraits::standard_ranges_member(std::string_view name) const
    -> Symbol* {
  auto control = unit_->control();
  auto standard = qualifiedLookupNamespace(
      unit_->globalScope(), control->getIdentifier(WellKnownName::T_STD));
  auto ranges = standard ? qualifiedLookupNamespace(
                               standard, control->getIdentifier("ranges"))
                         : nullptr;
  return ranges ? qualifiedLookup(ranges, control->getIdentifier(name))
                : nullptr;
}

auto TypeTraits::member_value(ClassSymbol* classSymbol,
                              std::string_view name) const
    -> std::optional<ConstValue> {
  auto arena = unit_->arena();
  auto identifier = control()->getIdentifier(name);
  auto symbol = qualifiedLookup(classSymbol, identifier);
  if (!symbol || !symbol->type()) return std::nullopt;
  if (auto field = symbol_cast<FieldSymbol>(symbol))
    ASTRewriter::requireFieldInitializer(unit_, field);
  auto expression = IdExpressionAST::create(arena);
  expression->unqualifiedId = NameIdAST::create(arena, identifier);
  expression->symbol = symbol;
  expression->type = symbol->type();
  expression->valueCategory = ValueCategory::kLValue;
  return ASTInterpreter{unit_}.evaluate(expression);
}

auto TypeTraits::type_order(const Type* first, const Type* second) const
    -> std::strong_ordering {
  if (first == second) return std::strong_ordering::equal;
  ExternalNameEncoder encoder{unit_};
  const auto order = encoder.encode(first) <=> encoder.encode(second);
  if (order != std::strong_ordering::equal) return order;
  auto firstEntity = entity_of(first);
  auto secondEntity = entity_of(second);
  if (firstEntity && secondEntity)
    return firstEntity->location().index() <=> secondEntity->location().index();
  return firstEntity ? std::strong_ordering::less
                     : std::strong_ordering::greater;
}

auto TypeTraits::member_type(ClassSymbol* classSymbol) const -> const Type* {
  auto symbol = qualifiedLookup(classSymbol, control()->getIdentifier("type"),
                                [](Symbol* s) { return is_type(s); });
  return symbol ? symbol->type() : nullptr;
}

auto TypeTraits::is_std_type(const Type* type, WellKnownName name) const
    -> bool {
  if (!type) return false;

  auto unqualified = remove_cv(type);

  Symbol* symbol = nullptr;
  if (auto classType = type_cast<ClassType>(unqualified))
    symbol = classType->symbol();
  else if (auto enumType = type_cast<ScopedEnumType>(unqualified))
    symbol = enumType->symbol();
  else if (auto enumType = type_cast<EnumType>(unqualified))
    symbol = enumType->symbol();

  if (!symbol || well_known_name(symbol->name()) != name) return false;

  return is_in_std_namespace(symbol);
}

auto TypeTraits::is_align_val_t(const Type* type) const -> bool {
  return is_std_type(type, WellKnownName::T_ALIGN_VAL_T);
}

auto TypeTraits::is_destroying_delete_t(const Type* type) const -> bool {
  return is_std_type(type, WellKnownName::T_DESTROYING_DELETE_T);
}

auto TypeTraits::initializer_list_element_type(const Type* targetType)
    -> const Type* {
  if (!targetType) return nullptr;

  auto unrefTarget = remove_reference(targetType);
  auto unqualTarget = remove_cv(unrefTarget);
  auto classType = type_cast<ClassType>(unqualTarget);
  if (!classType || !classType->symbol()) return nullptr;

  auto classSymbol = classType->symbol();
  if (well_known_name(classSymbol->name()) != WellKnownName::T_INITIALIZER_LIST)
    return nullptr;

  if (!is_in_std_namespace(classSymbol)) return nullptr;
  if (!classSymbol->isSpecialization()) return nullptr;

  auto args = classSymbol->templateArguments();
  if (args.size() != 1) return nullptr;

  require_complete_class(classSymbol);

  if (auto typeArg = std::get_if<const Type*>(&args[0])) return *typeArg;
  if (auto symbolArg = std::get_if<Symbol*>(&args[0])) {
    auto sym = *symbolArg;
    if (!sym) return nullptr;
    return sym->type();
  }

  return nullptr;
}

auto TypeTraits::require_complete_class(ClassSymbol* classSymbol) -> bool {
  if (!classSymbol) return false;
  if (classSymbol->isComplete()) return true;
  if (!unit_) return false;
  if (!unit_->config().checkTypes) return false;
  return ASTRewriter::ensureCompleteClass(unit_, classSymbol);
}

auto TypeTraits::require_complete_class(const Type* type) -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;
  return require_complete_class(classType->symbol());
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

auto TypeTraits::remove_address_space(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(RemoveQualifiers{*this, CvQualifiers::kNone, true}, type);
}

auto TypeTraits::add_cv(const Type* type, CvQualifiers cv) const
    -> const Type* {
  return add_qualifiers(type, cv, std::nullopt);
}

auto TypeTraits::add_qualifiers(const Type* type, CvQualifiers cv,
                                std::optional<std::uint32_t> addressSpace) const
    -> const Type* {
  if (!type) return type;
  if (cv == CvQualifiers::kNone && !addressSpace) return type;
  return visit(AddTypeQualifiers{*this, cv, addressSpace}, type);
}

auto TypeTraits::remove_noexcept(const Type* type) const -> const Type* {
  const auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return type;
  return control()->getFunctionType(
      functionType->returnType(), functionType->parameterTypes(),
      functionType->isVariadic(), functionType->cvQualifiers(),
      functionType->refQualifier(), false);
}

auto TypeTraits::remove_function_qualifiers(const FunctionType* type) const
    -> const FunctionType* {
  return control()->getFunctionType(type->returnType(), type->parameterTypes(),
                                    type->isVariadic(), {}, {},
                                    type->exceptionSpecification());
}

auto TypeTraits::replace_placeholder_types(const Type* type,
                                           const Type* replacement) const
    -> const Type* {
  if (!replacement) return nullptr;
  if (type_cast<AutoType>(type) || type_cast<DecltypeAutoType>(type))
    return replacement;

  if (auto qualType = type_cast<QualType>(type)) {
    auto elementType =
        replace_placeholder_types(qualType->elementType(), replacement);
    return add_qualifiers(elementType, qualType->cvQualifiers(),
                          qualType->addressSpace());
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
        functionType->exceptionSpecification());
  }

  if (auto pointerType = type_cast<MemberObjectPointerType>(type)) {
    auto classType =
        replace_placeholder_types(pointerType->classType(), replacement);
    auto elementType =
        replace_placeholder_types(pointerType->elementType(), replacement);
    return control()->getMemberPointerType(classType, elementType);
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

auto TypeTraits::is_base_of(const Type* base, const Type* derived) -> bool {
  auto baseClassType = unqualified_cast<ClassType>(base);
  if (!baseClassType) return false;
  auto derivedClassType = unqualified_cast<ClassType>(derived);
  if (!derivedClassType) return false;
  if (derivedClassType->symbol() == baseClassType->symbol()) return true;

  auto derivedClass = derivedClassType->definition();
  require_complete_class(derivedClass);
  if (!derivedClass) return false;

  return derivedClass->hasBaseClass(baseClassType->symbol());
}

auto TypeTraits::is_pointer_interconvertible_base_of(const Type* base,
                                                     const Type* derived)
    -> bool {
  if (!is_class(base)) return false;
  if (!is_class(derived)) return false;
  if (is_same(remove_cv(base), remove_cv(derived))) return true;
  auto derivedClass = unqualified_cast<ClassType>(derived)->definition();
  require_complete_class(derivedClass);
  if (!derivedClass || !derivedClass->isComplete()) return false;
  if (!is_standard_layout(derived)) return false;
  auto baseClass = unqualified_cast<ClassType>(base)->definition();
  return derivedClass->baseSubobjectInfo(baseClass).isUniqueSubobject();
}

namespace {
[[nodiscard]] auto decomposeForQualification(const TypeTraits& traits,
                                             const Type* lhs, const Type* rhs)
    -> std::optional<QualificationDecompositionPair> {
  QualificationDecompositionPair result;

  for (;;) {
    if (address_space(lhs) != address_space(rhs)) return std::nullopt;
    result.left.cv.push_back(cv_qualifiers(lhs));
    result.right.cv.push_back(cv_qualifiers(rhs));
    result.left.addressSpaces.push_back(address_space(lhs));

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

  result.left.terminal = traits.remove_cv(lhs);
  result.right.terminal = traits.remove_cv(rhs);

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

  if (is_class_or_union(type) || is_array(type)) return type;

  return qualType->elementType();
}

auto TypeTraits::adjusted_parameter_type(Control* control, const Type* type)
    -> const Type* {
  if (!type) return type;
  if (visit(IsArray{}, type))
    return control->getPointerType(visit(RemoveExtent{}, type));
  if (visit(IsFunction{}, type)) return control->getPointerType(type);
  return type;
}

auto TypeTraits::is_similar(const Type* lhs, const Type* rhs) const -> bool {
  if (lhs == rhs) return true;
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
        result = control()->getMemberPointerType(component.classType, result);
        break;
      case QualificationComponent::Kind::kBoundedArray:
        result = control()->getBoundedArrayType(result, component.size);
        break;
      case QualificationComponent::Kind::kUnboundedArray:
        result = control()->getUnboundedArrayType(result);
        break;
    }
    result = add_qualifiers(result, cv[i], left.addressSpaces[i]);
  }

  return result;
}

auto TypeTraits::is_qualification_convertible(const Type* from,
                                              const Type* to) const -> bool {
  return is_same(qualification_combined_type(from, to), to);
}

auto TypeTraits::is_variable_length_array(const Type* type) const -> bool {
  for (type = remove_cv(type); is_array(type);
       type = remove_cv(get_element_type(type))) {
    auto unresolved = type_cast<UnresolvedBoundedArrayType>(type);
    if (unresolved && !isDependent(unit_, unresolved->size())) return true;
  }
  return false;
}

auto TypeTraits::is_vla_compatible(const Type* left, const Type* right) const
    -> bool {
  for (;;) {
    left = remove_cv(left);
    right = remove_cv(right);
    if (is_same(left, right)) return true;
    auto leftArray = type_cast<UnresolvedBoundedArrayType>(left);
    auto rightArray = type_cast<UnresolvedBoundedArrayType>(right);
    if (!leftArray || !rightArray) return false;
    left = leftArray->elementType();
    right = rightArray->elementType();
  }
}

auto TypeTraits::is_reference_related(const Type* lhs, const Type* rhs)
    -> bool {
  if (is_similar(remove_cv(lhs), remove_cv(rhs))) return true;
  return is_base_of(lhs, rhs);
}

auto TypeTraits::integer_conversion_rank(const Type* type) const
    -> std::pair<int, int> {
  struct Rank {
    [[nodiscard]] auto operator()(const BoolType*) const -> int { return 1; }
    [[nodiscard]] auto operator()(const CharType*) const -> int { return 2; }
    [[nodiscard]] auto operator()(const SignedCharType*) const -> int {
      return 2;
    }
    [[nodiscard]] auto operator()(const UnsignedCharType*) const -> int {
      return 2;
    }
    [[nodiscard]] auto operator()(const ShortIntType*) const -> int {
      return 3;
    }
    [[nodiscard]] auto operator()(const UnsignedShortIntType*) const -> int {
      return 3;
    }
    [[nodiscard]] auto operator()(const IntType*) const -> int { return 4; }
    [[nodiscard]] auto operator()(const UnsignedIntType*) const -> int {
      return 4;
    }
    [[nodiscard]] auto operator()(const LongIntType*) const -> int { return 5; }
    [[nodiscard]] auto operator()(const UnsignedLongIntType*) const -> int {
      return 5;
    }
    [[nodiscard]] auto operator()(const LongLongIntType*) const -> int {
      return 6;
    }
    [[nodiscard]] auto operator()(const UnsignedLongLongIntType*) const -> int {
      return 6;
    }
    [[nodiscard]] auto operator()(const Int128Type*) const -> int { return 7; }
    [[nodiscard]] auto operator()(const UnsignedInt128Type*) const -> int {
      return 7;
    }
    [[nodiscard]] auto operator()(const Type*) const -> int { return 0; }
  };
  auto representation = integral_representation(type);
  if (!representation) return {};
  return {representation->bits, visit(Rank{}, remove_cv(type))};
}

namespace {

struct StandardFloatingPointRank {
  [[nodiscard]] auto operator()(const FloatType*) const -> int { return 1; }
  [[nodiscard]] auto operator()(const DoubleType*) const -> int { return 2; }
  [[nodiscard]] auto operator()(const LongDoubleType*) const -> int {
    return 3;
  }
  [[nodiscard]] auto operator()(const Type*) const -> int { return 0; }
};

[[nodiscard]] auto standardFloatingPointRank(const Type* type) -> int {
  return visit(StandardFloatingPointRank{}, type);
}

[[nodiscard]] auto compareValueSets(const FloatingPointFormat& a,
                                    const FloatingPointFormat& b)
    -> std::partial_ordering {
  const auto aInB = a.exponentBits <= b.exponentBits &&
                    a.significandDigits <= b.significandDigits;
  const auto bInA = b.exponentBits <= a.exponentBits &&
                    b.significandDigits <= a.significandDigits;
  if (aInB && bInA) return std::partial_ordering::equivalent;
  if (aInB) return std::partial_ordering::less;
  if (bInA) return std::partial_ordering::greater;
  return std::partial_ordering::unordered;
}

[[nodiscard]] auto floatingPointRankRepresentative(Control* control,
                                                   const Type* type)
    -> const Type* {
  if (standardFloatingPointRank(type)) return type;

  auto format = control->memoryLayout()->floatingPointFormat(type);
  if (!format) return type;

  const Type* standardTypes[] = {control->getFloatType(),
                                 control->getDoubleType(),
                                 control->getLongDoubleType()};

  const Type* sameValues = nullptr;
  int sameValuesCount = 0;
  for (auto standard : standardTypes) {
    auto standardFormat =
        control->memoryLayout()->floatingPointFormat(standard);
    if (!standardFormat) continue;
    if (compareValueSets(*format, *standardFormat) !=
        std::partial_ordering::equivalent)
      continue;
    sameValues = standard;
    ++sameValuesCount;
  }

  if (sameValuesCount == 1) return sameValues;
  if (sameValuesCount > 1) return control->getDoubleType();
  return type;
}

}  // namespace

auto TypeTraits::floating_point_conversion_order(const Type* a,
                                                 const Type* b) const
    -> std::partial_ordering {
  a = remove_cv(a);
  b = remove_cv(b);
  if (a == b) return std::partial_ordering::equivalent;

  a = floatingPointRankRepresentative(control(), a);
  b = floatingPointRankRepresentative(control(), b);

  const auto standardA = standardFloatingPointRank(a);
  const auto standardB = standardFloatingPointRank(b);
  if (standardA && standardB) return standardA <=> standardB;

  auto formatA = control()->memoryLayout()->floatingPointFormat(a);
  auto formatB = control()->memoryLayout()->floatingPointFormat(b);
  if (!formatA || !formatB) return std::partial_ordering::unordered;
  return compareValueSets(*formatA, *formatB);
}

auto TypeTraits::floating_point_conversion_subrank(const Type* type) const
    -> int {
  return standardFloatingPointRank(remove_cv(type)) ? 0 : 1;
}

auto TypeTraits::represents_all_values_of(const Type* target,
                                          const Type* source) const -> bool {
  auto targetRepresentation = integral_representation(target);
  auto sourceRepresentation = integral_representation(source);
  if (!targetRepresentation || !sourceRepresentation) return false;
  auto targetWidth = targetRepresentation->bits;
  auto sourceWidth = sourceRepresentation->bits;
  if (!sourceRepresentation->isSigned) {
    if (!targetRepresentation->isSigned) return targetWidth >= sourceWidth;
    return targetWidth > sourceWidth;
  }
  if (!targetRepresentation->isSigned) return false;
  return targetWidth >= sourceWidth;
}

auto TypeTraits::integral_promotion_candidates() const
    -> std::array<const Type*, 6> {
  return {
      control()->getIntType(),         control()->getUnsignedIntType(),
      control()->getLongIntType(),     control()->getUnsignedLongIntType(),
      control()->getLongLongIntType(), control()->getUnsignedLongLongIntType()};
}

auto TypeTraits::promoted_arithmetic_type(const Type* type) const
    -> const Type* {
  if (!type) return nullptr;
  type = remove_cv(type);
  if (type_cast<HalfType>(type)) return control()->getFloatType();
  return promoted_integer_type(type);
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
      for (auto candidate : integral_promotion_candidates()) {
        if (represents_all_values_of(candidate, source)) return candidate;
      }
      return source;

    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kUnsignedChar:
    case TypeKind::kShortInt:
    case TypeKind::kUnsignedShortInt:
      if (represents_all_values_of(control()->getIntType(), source))
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

  if (enumSymbol && enumSymbol->promotionType())
    return {enumSymbol->promotionType(), nullptr};

  auto representsEnumerationValues = [&](const Type* type) {
    if (!enumSymbol) return false;
    for (auto member : enumSymbol->members()) {
      auto enumerator = symbol_cast<EnumeratorSymbol>(member);
      if (!enumerator) continue;
      const auto& value = enumerator->value();
      auto integer = value ? std::get_if<ConstInt>(&*value) : nullptr;
      if (!integer || !converted_integral_constant(type, *integer))
        return false;
    }
    return true;
  };

  for (auto candidate : integral_promotion_candidates()) {
    if (representsEnumerationValues(candidate)) return {candidate, nullptr};
  }

  auto underlyingType = enumType->underlyingType();
  if (underlyingType && representsEnumerationValues(underlyingType))
    return {underlyingType, nullptr};
  return {nullptr, nullptr};
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
  if (remove_cv(from)->kind() == TypeKind::kHalf)
    return remove_cv(to)->kind() == TypeKind::kFloat;
  return remove_cv(from)->kind() == TypeKind::kFloat &&
         remove_cv(to)->kind() == TypeKind::kDouble;
}

auto TypeTraits::is_reference_compatible(const Type* target, const Type* source)
    -> bool {
  if (!target || !source) return false;
  if (address_space(target) != address_space(source)) return false;

  auto targetUnqualified = remove_cv(target);
  auto sourceUnqualified = remove_cv(source);

  if (targetUnqualified == sourceUnqualified)
    return is_at_least_as_cv_qualified(cv_qualifiers(target),
                                       cv_qualifiers(source));

  if (is_qualification_convertible(control()->getPointerType(source),
                                   control()->getPointerType(target)))
    return true;

  if (is_function(targetUnqualified) && is_function(sourceUnqualified)) {
    return is_same(remove_noexcept(sourceUnqualified), targetUnqualified);
  }

  if (!is_base_of(targetUnqualified, sourceUnqualified)) return false;

  return is_at_least_as_cv_qualified(cv_qualifiers(target),
                                     cv_qualifiers(source));
}

auto TypeTraits::is_virtual_base_of(const Type* base, const Type* derived) const
    -> bool {
  auto baseClassType = unqualified_cast<ClassType>(base);
  if (!baseClassType) return false;
  auto derivedClassType = unqualified_cast<ClassType>(derived);
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

  if (overriderType->isVariadic() != overriddenType->isVariadic()) return false;

  if (!has_corresponding_object_parameters(overrider, overridden)) return false;

  const auto& overriderParams = overriderType->parameterTypes();
  const auto& overriddenParams = overriddenType->parameterTypes();

  const std::size_t overriderObjectParams =
      overrider->hasExplicitObjectParameter() ? 1 : 0;
  const std::size_t overriddenObjectParams =
      overridden->hasExplicitObjectParameter() ? 1 : 0;

  if (overriderParams.size() - overriderObjectParams !=
      overriddenParams.size() - overriddenObjectParams)
    return false;

  for (std::size_t i = 0; i < overriderParams.size() - overriderObjectParams;
       ++i) {
    if (!is_same(overriderParams[i + overriderObjectParams],
                 overriddenParams[i + overriddenObjectParams]))
      return false;
  }

  return true;
}

auto TypeTraits::has_corresponding_object_parameters(
    const FunctionSymbol* overrider, const FunctionSymbol* overridden) const
    -> bool {
  auto overriderType = type_cast<FunctionType>(overrider->type());
  auto overriddenType = type_cast<FunctionType>(overridden->type());

  const auto overriderIsExplicit = overrider->hasExplicitObjectParameter();
  const auto overriddenIsExplicit = overridden->hasExplicitObjectParameter();

  if (!overriderIsExplicit && !overriddenIsExplicit) {
    if (overriderType->cvQualifiers() != overriddenType->cvQualifiers())
      return false;
    return overriderType->refQualifier() == overriddenType->refQualifier();
  }

  auto objectParameterCv = [&](const FunctionSymbol* function,
                               const FunctionType* type) {
    if (!function->hasExplicitObjectParameter()) return type->cvQualifiers();
    return cv_qualifiers(remove_reference(type->parameterTypes().front()));
  };

  auto namesOwnClass = [&](const FunctionSymbol* function,
                           const FunctionType* type) {
    if (!function->hasExplicitObjectParameter()) return true;
    auto declaringClass =
        declaringClassOf(const_cast<FunctionSymbol*>(function));
    if (!declaringClass) return false;
    auto objectType =
        remove_cv(remove_reference(type->parameterTypes().front()));
    return is_same(objectType, declaringClass->type());
  };

  if (!overriderIsExplicit &&
      overriderType->refQualifier() != RefQualifier::kNone)
    return false;

  if (!overriddenIsExplicit &&
      overriddenType->refQualifier() != RefQualifier::kNone)
    return false;

  if (objectParameterCv(overrider, overriderType) !=
      objectParameterCv(overridden, overriddenType))
    return false;

  return namesOwnClass(overrider, overriderType) &&
         namesOwnClass(overridden, overriddenType);
}

auto TypeTraits::is_covariant_return_type(const Type* overriddenReturnType,
                                          const Type* overriderReturnType,
                                          CovariantReturnClasses* classes)
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
    if (!element || !is_class_or_union(remove_cv(element))) return nullptr;
    return element;
  };

  auto overriddenClass = classOf(overridden);
  auto overriderClass = classOf(overrider);
  if (!overriddenClass || !overriderClass) return false;

  const auto overriddenCv = cv_qualifiers(overriddenClass);
  const auto overriderCv = cv_qualifiers(overriderClass);
  if ((static_cast<int>(overriderCv) & ~static_cast<int>(overriddenCv)) != 0)
    return false;

  if (!is_base_of(overriddenClass, overriderClass)) return false;

  if (classes) {
    auto symbolOf = [](const Type* type) -> ClassSymbol* {
      auto classType = unqualified_cast<ClassType>(type);
      return classType ? classType->symbol() : nullptr;
    };
    classes->overriddenClass = symbolOf(overriddenClass);
    classes->overriderClass = symbolOf(overriderClass);
  }

  return true;
}

auto TypeTraits::can_initialize(const Type* to, const Type* from,
                                bool directInitialization) const -> bool {
  if (!from || !to) return false;

  const auto fromIsVoid = is_void(from);
  const auto toIsVoid = is_void(to);
  if (fromIsVoid || toIsVoid) return fromIsVoid && toIsVoid;

  StandardConversion conversions{unit_};
  auto initializationKind = InitializationKind::kCopyInitialization;
  if (directInitialization)
    initializationKind = InitializationKind::kDirectInitialization;
  auto sequence = conversions.computeConversionSequence(declval(from), to,
                                                        initializationKind);
  if (!conversions.isAccessible(sequence)) return false;

  if (is_reference(to)) return true;
  auto classType = unqualified_cast<ClassType>(to);
  if (!classType) return true;
  auto classSymbol = classType->definition();
  if (!classSymbol) return false;
  auto destructor = classSymbol->destructor();
  if (destructor && unit_->typeTraits().is_deleted_function(destructor))
    return false;
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
  if (auto principal = function->structorPrincipal()) function = principal;
  ASTRewriter::completePendingExceptionSpecification(unit_, function);
  auto functionType = type_cast<FunctionType>(function->type());
  return functionType && functionType->isNoexcept();
}

auto TypeTraits::is_nothrow_initialization(const Type* to, const Type* from,
                                           bool directInitialization) const
    -> bool {
  if (!from || !to) return false;
  auto initializationKind = InitializationKind::kCopyInitialization;
  if (directInitialization)
    initializationKind = InitializationKind::kDirectInitialization;
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  StandardConversion conversions{unit_};
  auto expression = declval(from);
  auto sequence =
      conversions.computeConversionSequence(expression, to, initializationKind);
  if (!sequence) return false;
  conversions.applyConversionSequence(sequence, expression);
  return !TypeChecker::isPotentiallyThrowing(expression, unit_);
}

auto TypeTraits::is_trivial_initialization(const Type* to, const Type* from,
                                           bool directInitialization) const
    -> bool {
  if (!from || !to) return false;
  auto initializationKind = InitializationKind::kCopyInitialization;
  if (directInitialization)
    initializationKind = InitializationKind::kDirectInitialization;
  auto sequence = StandardConversion{unit_}.computeConversionSequence(
      declval(from), to, initializationKind);
  if (!sequence) return false;
  return sequence.udc.function == nullptr;
}

auto TypeTraits::is_convertible(const Type* from, const Type* to) const
    -> bool {
  return can_initialize(to, from, false);
}

auto TypeTraits::is_nothrow_convertible(const Type* from, const Type* to) const
    -> bool {
  if (!is_convertible(from, to)) return false;
  if (is_void(to)) return true;
  return is_nothrow_initialization(to, from, false);
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

auto TypeTraits::declval(const Type* type) const -> ExpressionAST* {
  auto referenced = remove_reference(type);
  if (is_void(referenced)) {
    return ThisExpressionAST::create(unit_->arena(), ValueCategory::kPrValue,
                                     remove_cv(referenced));
  }
  auto valueCategory = ValueCategory::kXValue;
  if (is_lvalue_reference(type) || is_function(referenced))
    valueCategory = ValueCategory::kLValue;
  return ThisExpressionAST::create(unit_->arena(), valueCategory, referenced);
}

auto TypeTraits::has_new_extended_alignment(const Type* type) const -> bool {
  if (!type) return false;
  auto memoryLayout = control()->memoryLayout();
  auto alignment = memoryLayout->alignmentOf(type);
  if (!alignment) return false;
  return *alignment > memoryLayout->defaultNewAlignment();
}

auto TypeTraits::conditional_operator_type(const Type* first,
                                           const Type* second) const
    -> const Type* {
  auto arena = unit_->arena();
  auto conditional = ConditionalExpressionAST::create(arena);
  conditional->condition = BoolLiteralExpressionAST::create(
      arena, false, ValueCategory::kPrValue, control()->getBoolType());
  conditional->iftrueExpression = declval(first);
  conditional->iffalseExpression = declval(second);

  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  SilentDiagnosticsScope diagnostics{unit_};
  TypeChecker check{unit_};
  check.setScope(unit_->globalScope());
  ExpressionAST* expression = conditional;
  check.check(&expression);
  diagnostics.finish();

  if (diagnostics.hadError()) return nullptr;
  return decltype_of(expression);
}

auto TypeTraits::is_pod(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar_or_vector(unqual)) return true;
  if (is_void(unqual)) return false;
  if (is_class_or_union(unqual))
    return is_trivial(unqual) && is_standard_layout(unqual);
  if (is_array(unqual)) return is_pod(remove_all_extents(unqual));
  return false;
}

auto TypeTraits::has_pod_object_sizes(const Type* type) -> bool {
  return is_abi_pod(*this, type, control()->memoryLayout()->abiPodDefinition());
}

auto TypeTraits::data_size(const Type* type) -> std::uint64_t {
  auto classType = unqualified_cast<ClassType>(type);
  auto cls = classType ? classType->definition() : nullptr;
  auto layout = cls ? cls->layout() : nullptr;
  if (!layout) return 0;
  if (has_pod_object_sizes(type)) return layout->size();
  return layout->dataSize();
}

auto TypeTraits::non_virtual_size(const Type* type) -> std::uint64_t {
  auto classType = unqualified_cast<ClassType>(type);
  auto cls = classType ? classType->definition() : nullptr;
  auto layout = cls ? cls->layout() : nullptr;
  if (!layout) return 0;
  if (has_pod_object_sizes(type)) return layout->size();
  return layout->nonVirtualSize();
}

auto TypeTraits::is_trivial(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar_or_vector(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
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
  return standard_layout(type, StandardLayoutRule::kLanguage);
}

auto TypeTraits::standard_layout(const Type* type, StandardLayoutRule rule)
    -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar_or_vector(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;
    if (cls->hasVirtualFunctions()) return false;
    if (cls->hasVirtualBaseClasses()) return false;

    std::optional<AccessSpecifier> memberAccess;
    for (auto field : cls->members() | views::non_static_fields) {
      if (field->isBitField() && !field->name()) continue;
      if (is_reference(field->type())) return false;
      if (!standard_layout(remove_all_extents(field->type()), rule))
        return false;
      if (!memberAccess) memberAccess = field->accessSpecifier();
      if (*memberAccess != field->accessSpecifier()) return false;
    }

    auto dataBearingSubobjects = 0;
    if (has_non_static_data_members(cls)) dataBearingSubobjects = 1;
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      baseClass = baseClass->resolvedDefinition();
      if (!standard_layout(baseClass->type(), rule)) return false;
      if (has_data_members_in_hierarchy(baseClass)) ++dataBearingSubobjects;
      if (dataBearingSubobjects > 1) return false;
    }

    if (rule == StandardLayoutRule::kLayout) {
      return !has_base_of_first_member_type(*this, cls);
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
  if (is_array(unqual))
    return standard_layout(remove_all_extents(unqual), rule);
  return false;
}

namespace {
[[nodiscard]] auto valueBitWidth(ConstInt::UWide value) -> int {
  auto width = 0;
  while (value) {
    ++width;
    value >>= 1;
  }
  return width;
}

[[nodiscard]] auto enumerationValues(TypeTraits& traits, const Type* type)
    -> std::optional<std::pair<int, bool>> {
  auto representation = traits.integral_representation(type);
  if (!representation) return std::nullopt;
  auto enumeration = type_cast<EnumType>(type);
  if (!enumeration || enumeration->symbol()->hasFixedUnderlyingType())
    return std::pair{representation->bits, representation->isSigned};
  auto positiveBits = 0;
  auto negativeBits = 0;
  for (auto member : enumeration->symbol()->members()) {
    auto enumerator = symbol_cast<EnumeratorSymbol>(member);
    if (!enumerator) continue;
    const auto& value = enumerator->value();
    auto integer = value ? std::get_if<ConstInt>(&*value) : nullptr;
    if (!integer) return std::nullopt;
    if (integer->isNegative()) {
      negativeBits =
          std::max(negativeBits, valueBitWidth(integer->magnitude() - 1) + 1);
    } else {
      positiveBits =
          std::max(positiveBits, valueBitWidth(integer->magnitude()));
    }
  }
  if (negativeBits)
    return std::pair{std::max(negativeBits, positiveBits + 1), true};
  return std::pair{std::max(1, positiveBits), false};
}

class LayoutCompatible {
  struct Result {
    const Type* first;
    const Type* second;
    bool compatible;
  };

  TypeTraits& traits_;
  std::vector<Result> results_;

  static void appendFields(ClassSymbol* cls,
                           std::vector<FieldSymbol*>& fields) {
    for (auto base : cls->baseClasses()) {
      if (auto baseClass = resolved_base_class(base))
        appendFields(baseClass, fields);
    }
    for (auto field : cls->members() | views::non_static_fields) {
      if (cls->isUnion() && field->isBitField() && !field->name()) continue;
      fields.push_back(field);
    }
  }

  [[nodiscard]] auto matchFields(FieldSymbol* first, FieldSymbol* second)
      -> bool {
    if (first->isNoUniqueAddress() || second->isNoUniqueAddress()) return false;
    if (first->effectiveAlignment() != second->effectiveAlignment())
      return false;
    if (first->isBitField() != second->isBitField()) return false;
    if (first->isBitField()) {
      auto firstWidth = first->bitFieldWidth();
      auto secondWidth = second->bitFieldWidth();
      if (!firstWidth || !secondWidth) return false;
      if (!equivalent_values(*firstWidth, *secondWidth)) return false;
    }
    return test(first->type(), second->type());
  }

  [[nodiscard]] auto matchClasses(ClassSymbol* first, ClassSymbol* second)
      -> bool {
    if (first->isUnion() != second->isUnion()) return false;
    if (!traits_.is_standard_layout(first->type())) return false;
    if (!traits_.is_standard_layout(second->type())) return false;
    std::vector<FieldSymbol*> firstFields;
    std::vector<FieldSymbol*> secondFields;
    appendFields(first, firstFields);
    appendFields(second, secondFields);
    if (firstFields.size() != secondFields.size()) return false;
    if (!first->isUnion()) {
      for (std::size_t i = 0; i < firstFields.size(); ++i) {
        if (!matchFields(firstFields[i], secondFields[i])) return false;
      }
      return true;
    }
    std::vector<bool> matched(secondFields.size());
    for (auto field : firstFields) {
      auto found = false;
      for (std::size_t i = 0; i < secondFields.size(); ++i) {
        if (matched[i]) continue;
        if (!test(field->type(), secondFields[i]->type())) continue;
        matched[i] = true;
        found = true;
        break;
      }
      if (!found) return false;
    }
    return true;
  }

  [[nodiscard]] auto compute(const Type* first, const Type* second) -> bool {
    if (traits_.is_enum(first) && traits_.is_enum(second)) {
      if (!traits_.is_same(traits_.underlying_type(first),
                           traits_.underlying_type(second)))
        return false;
      auto firstValues = enumerationValues(traits_, first);
      auto secondValues = enumerationValues(traits_, second);
      return firstValues && firstValues == secondValues;
    }
    auto firstClass = type_cast<ClassType>(first);
    auto secondClass = type_cast<ClassType>(second);
    if (!firstClass || !secondClass) return false;
    return matchClasses(firstClass->definition(), secondClass->definition());
  }

 public:
  explicit LayoutCompatible(TypeTraits& traits) : traits_(traits) {}

  [[nodiscard]] auto test(const Type* first, const Type* second) -> bool {
    first = traits_.remove_cv(first);
    second = traits_.remove_cv(second);
    if (traits_.is_same(first, second)) return true;
    for (const auto& result : results_) {
      if (result.first == first && result.second == second)
        return result.compatible;
    }
    auto result = compute(first, second);
    results_.push_back({first, second, result});
    return result;
  }
};

class LiteralTypes {
  TypeTraits& traits_;
  std::vector<std::pair<ClassSymbol*, bool>> results_;

  [[nodiscard]] auto testClass(ClassSymbol* cls) -> bool {
    auto destructor = cls->destructor();
    if (destructor && !destructor->isDefaulted() && !destructor->isConstexpr())
      return false;

    for (auto base : cls->baseClasses()) {
      auto baseClass = resolved_base_class(base);
      if (baseClass && !test(baseClass->type())) return false;
    }

    auto hasVariant = false;
    auto hasLiteralVariant = false;
    for (auto field : cls->members() | views::non_static_fields) {
      if (field->isBitField() && !field->name()) continue;
      hasVariant = true;
      if (traits_.is_volatile(field->type())) {
        if (!cls->isUnion()) return false;
        continue;
      }
      auto literal = test(field->type());
      if (!cls->isUnion() && !literal) return false;
      if (literal) hasLiteralVariant = true;
    }

    if (cls->isClosureType()) return true;
    if (traits_.is_aggregate(cls->type()))
      return !cls->isUnion() || !hasVariant || hasLiteralVariant;

    for (auto constructor : cls->constructors()) {
      if (cls->isCopyConstructor(constructor)) continue;
      if (cls->isMoveConstructor(constructor)) continue;
      if (constructor->isConstexpr()) return true;
    }
    return false;
  }

 public:
  explicit LiteralTypes(TypeTraits& traits) : traits_(traits) {}

  [[nodiscard]] auto test(const Type* type) -> bool {
    auto unqualified = traits_.remove_cv(type);
    if (traits_.is_void(unqualified)) return true;
    if (auto atomicType = type_cast<AtomicType>(unqualified))
      return test(atomicType->elementType());
    if (traits_.is_scalar_or_vector(unqualified)) return true;
    if (traits_.is_reference(unqualified)) return true;
    if (traits_.is_array(unqualified))
      return test(traits_.remove_all_extents(unqualified));
    auto classType = type_cast<ClassType>(unqualified);
    if (!classType) return false;
    auto cls = classType->definition();
    traits_.require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;
    for (auto [cached, result] : results_) {
      if (cached == cls) return result;
    }
    auto result = testClass(cls);
    results_.emplace_back(cls, result);
    return result;
  }
};

class StructuralTypes {
  TypeTraits& traits_;
  LiteralTypes literal_;
  std::vector<std::pair<ClassSymbol*, bool>> results_;

  [[nodiscard]] auto testClass(ClassSymbol* cls) -> bool {
    if (cls->isClosureType()) return !cls->hasLambdaCapture();
    if (!literal_.test(cls->type())) return false;
    for (auto base : cls->baseClasses()) {
      if (base->accessSpecifier() != AccessSpecifier::kPublic) return false;
      auto baseClass = resolved_base_class(base);
      if (!baseClass || !test(baseClass->type())) return false;
    }
    for (auto field : cls->members() | views::non_static_fields) {
      if (field->isBitField() && !field->name()) continue;
      if (field->accessSpecifier() != AccessSpecifier::kPublic) return false;
      if (field->isMutable()) return false;
      if (!test(traits_.remove_all_extents(field->type()))) return false;
    }
    return true;
  }

 public:
  explicit StructuralTypes(TypeTraits& traits)
      : traits_(traits), literal_(traits) {}

  [[nodiscard]] auto test(const Type* type) -> bool {
    if (traits_.is_scalar(type)) return true;
    if (traits_.is_lvalue_reference(type)) return true;
    auto classType = unqualified_cast<ClassType>(type);
    if (!classType) return false;
    auto cls = classType->definition();
    traits_.require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;
    for (auto [cached, result] : results_) {
      if (cached == cls) return result;
    }
    auto result = testClass(cls);
    results_.emplace_back(cls, result);
    return result;
  }
};
}  // namespace

auto TypeTraits::is_layout_compatible(const Type* first, const Type* second)
    -> bool {
  return LayoutCompatible{*this}.test(first, second);
}

auto TypeTraits::is_literal_type(const Type* type) -> bool {
  return LiteralTypes{*this}.test(type);
}

auto TypeTraits::has_constexpr_standard_exception() const -> bool {
  auto exception = symbol_cast<ClassSymbol>(
      lookupStandardLibraryType(unit_, WellKnownName::T_EXCEPTION));
  auto definition = exception ? exception->resolvedDefinition() : nullptr;
  auto destructor = definition ? definition->destructor() : nullptr;
  return destructor && destructor->isConstexpr();
}

auto TypeTraits::handler_matches(const Type* handlerType,
                                 const Type* exceptionType) const -> bool {
  auto target = remove_cv(remove_reference(handlerType));
  auto source = remove_cv(exceptionType);
  if (target == source) return true;
  if (is_class_or_union(target)) {
    auto base = entity_of(target);
    auto derived = entity_of(source);
    auto baseClass = symbol_cast<ClassSymbol>(base);
    auto derivedClass = symbol_cast<ClassSymbol>(derived);
    if (!baseClass || !derivedClass) return false;
    auto info = derivedClass->resolvedDefinition()->baseSubobjectInfo(
        baseClass->resolvedDefinition());
    return info.subobjectCount == 1 && info.publicSubobjectCount == 1;
  }
  if (auto reference = type_cast<LvalueReferenceType>(handlerType)) {
    if (!is_const(reference->elementType())) return false;
  }
  if (!is_pointer(target) && !is_member_pointer(target)) return false;
  if (is_null_pointer(source)) return true;
  if (is_pointer(target) != is_pointer(source)) return false;
  if (!is_pointer(target) && !is_similar(target, source)) return false;
  return is_convertible(source, target);
}

auto TypeTraits::is_std_meta_type(const Type* type, std::string_view name) const
    -> bool {
  auto entity = entity_of(type);
  if (!entity || entity->name() != control()->getIdentifier(name) ||
      template_declaration_of(entity) || primary_template_of(entity))
    return false;
  auto scope = symbol_cast<NamespaceSymbol>(entity->parent());
  return scope && well_known_name(scope->name()) == WellKnownName::T_META &&
         is_std_namespace(scope->parent());
}

auto TypeTraits::is_structural(const Type* type) -> bool {
  return StructuralTypes{*this}.test(type);
}

auto TypeTraits::converted_pack_index(ExpressionAST* index)
    -> std::optional<std::size_t> {
  StandardConversion conversions{unit_};
  auto converted =
      conversions.convertedConstantExpression(index, control()->getSizeType());
  if (!converted) return std::nullopt;
  ASTInterpreter interpreter{unit_};
  auto value = interpreter.evaluate(*converted);
  return value ? interpreter.toUInt(*value) : std::nullopt;
}

auto TypeTraits::synthesizes_from_spaceship(TokenKind op, const Type* left,
                                            const Type* right) -> bool {
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  SilentDiagnosticsScope diagnostics{unit_};
  auto comparison = BinaryExpressionAST::create(unit_->arena());
  comparison->op = op;
  auto operand = [this](const Type* type) -> ExpressionAST* {
    if (is_reference(type)) return declval(type);
    return ThisExpressionAST::create(unit_->arena(), ValueCategory::kPrValue,
                                     type);
  };
  comparison->leftExpression = operand(left);
  comparison->rightExpression = operand(right);
  ExpressionAST* expression = comparison;
  TypeChecker checker{unit_};
  checker.setScope(unit_->globalScope());
  checker.setReportErrors(true);
  checker.check(&expression);
  diagnostics.finish();
  if (diagnostics.hadError() || expression != comparison) return false;
  auto isThreeWay = [](ExpressionAST* operand) {
    auto call = ast_cast<BinaryExpressionAST>(operand);
    return call && call->symbol && call->op == TokenKind::T_LESS_EQUAL_GREATER;
  };
  return isThreeWay(comparison->leftExpression) ||
         isThreeWay(comparison->rightExpression);
}

auto TypeTraits::is_implicit_lifetime(const Type* type) -> bool {
  if (is_scalar(type) || is_array(type)) return true;
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls || !cls->isComplete()) return false;
  auto destructor = cls->destructor();
  if (is_aggregate(type) && !is_user_provided(destructor)) return true;
  if (!destructor || unit_->typeTraits().is_deleted_function(destructor))
    return false;
  if (!has_trivial_destructor(type)) return false;
  auto constructors = cls->constructors();
  std::erase_if(constructors, [cls](FunctionSymbol* constructor) {
    if (constructor->isTemplatePattern()) return true;
    return !constructorKind(cls, constructor);
  });
  for (auto constructor :
       eligibleSpecialMembers(unit_, std::move(constructors))) {
    if (is_trivial_special_member(constructor)) return true;
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
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->isBitField() && !field->name()) continue;
    elements.push_back(field);
  }
  return elements;
}

auto TypeTraits::is_aggregate(const Type* type) -> bool {
  if (is_array(type)) return true;
  if (is_vector(type)) return true;
  if (is_complex(type)) return true;
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls || !cls->isComplete()) return false;
  if (cls->hasInheritedConstructors()) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  if (cls->hasUserDeclaredConstructors()) return false;
  for (auto field : cls->members() | views::non_static_fields) {
    if (field->isBitField() && !field->name()) continue;
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
    auto bits = std::get_if<ConstInt>(&*width);
    return bits && bits->isZero();
  }

  if (!field->isNoUniqueAddress()) return false;

  auto classType = type_cast<ClassType>(remove_cv(field->type()));
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls) return false;
  auto layout = cls->resolvedDefinition()->layout();
  return layout && layout->isAbiEmpty();
}

auto TypeTraits::requires_zero_initialization(const Type* type,
                                              FunctionSymbol* constructor)
    -> bool {
  if (!is_class_or_union(type)) return false;
  if (is_empty(type)) return false;
  if (!constructor) return true;
  return constructor->isDefaulted() ||
         unit_->typeTraits().is_deleted_function(constructor);
}

auto TypeTraits::is_empty(const Type* type) -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;
  if (classType->isUnion()) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
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
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls || !cls->isComplete()) return false;
  return cls->isPolymorphic();
}

auto TypeTraits::is_final(const Type* type) -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;
  auto cls = classType->definition();
  if (!cls) return false;
  return cls->isFinal();
}

auto TypeTraits::select_constructor(ClassSymbol* classSymbol,
                                    std::span<const Type* const> argTypes)
    -> FunctionSymbol* {
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  std::vector<ExpressionAST*> args;
  args.reserve(argTypes.size());

  for (auto argType : argTypes) {
    if (!argType) return nullptr;
    args.push_back(declval(argType));
  }

  auto result = OverloadResolution{unit_}.resolveConstructor(classSymbol, args);
  return result.selected();
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

  if (is_scalar_or_vector(unqual)) {
    if (argTypes.empty()) return true;
    if (argTypes.size() == 1) return can_initialize(unqual, argTypes[0], true);
    return false;
  }

  if (is_array(unqual)) {
    if (is_unbounded_array(unqual)) return false;
    if (!argTypes.empty()) return false;
    return is_constructible(remove_all_extents(unqual), argTypes);
  }

  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;

    auto selected = select_constructor(cls, argTypes);
    if (!selected || unit_->typeTraits().is_deleted_function(selected))
      return false;
    if (cls->isAbstract()) return false;
    if (!is_accessible_from_unrelated_context(selected)) return false;
    auto destructor = cls->destructor();
    if (destructor && unit_->typeTraits().is_deleted_function(destructor))
      return false;
    return is_accessible_from_unrelated_context(destructor);
  }

  if (is_void(unqual)) return false;

  return false;
}

auto TypeTraits::is_const_default_constructible(const Type* type) -> bool {
  auto element = remove_cv(remove_all_extents(type));
  if (type_cast<BuiltinMetaInfoType>(element)) return true;
  if (is_null_pointer(element)) return true;

  auto classType = type_cast<ClassType>(element);
  if (!classType) return false;

  auto classSymbol = classType->definition();
  require_complete_class(classSymbol);
  if (!classSymbol || !classSymbol->isComplete()) return false;

  if (default_initialization_invokes_user_provided_constructor(classSymbol))
    return true;

  return has_const_default_constructible_subobjects(classSymbol);
}

auto TypeTraits::default_initialization_invokes_user_provided_constructor(
    ClassSymbol* classSymbol) -> bool {
  return is_user_provided(select_constructor(classSymbol, {}));
}

auto TypeTraits::has_const_default_constructible_subobjects(
    ClassSymbol* classSymbol) -> bool {
  auto members = views::members(classSymbol) | views::non_static_fields |
                 std::views::filter([](FieldSymbol* field) {
                   return field->name() || !field->isBitField();
                 });

  if (classSymbol->isUnion()) {
    if (members.begin() == members.end()) return true;
    return std::ranges::count_if(members, &FieldSymbol::hasInitializer) == 1;
  }

  for (auto field : members) {
    if (field->hasInitializer()) continue;
    if (!is_const_default_constructible(field->type())) return false;
  }

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual() || !base->symbol()) continue;
    if (!is_const_default_constructible(base->symbol()->type())) return false;
  }

  auto layout = classSymbol->layout();
  if (!layout || classSymbol->isAbstract()) return true;

  return std::ranges::all_of(layout->virtualBases(), [this](ClassSymbol* base) {
    return is_const_default_constructible(base->type());
  });
}

auto TypeTraits::is_nothrow_constructible(const Type* type,
                                          std::span<const Type* const> argTypes)
    -> bool {
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  if (!type) return false;
  auto unqual = remove_cv(type);

  if (!is_constructible(type, argTypes)) return false;

  if (is_reference(unqual) || is_scalar_or_vector(unqual)) {
    if (argTypes.empty()) return true;
    return is_nothrow_initialization(unqual, argTypes.front(), true);
  }

  if (is_array(unqual))
    return is_nothrow_constructible(remove_all_extents(unqual), argTypes);

  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;

    std::vector<ExpressionAST*> args;
    args.reserve(argTypes.size());
    for (auto argType : argTypes) args.push_back(declval(argType));
    auto result = OverloadResolution{unit_}.resolveConstructor(cls, args);
    auto selected = result.selected();
    if (!selected || unit_->typeTraits().is_deleted_function(selected))
      return false;
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
  if (is_reference(unqual) || is_scalar_or_vector(unqual)) {
    if (argTypes.empty()) return true;
    return is_trivial_initialization(unqual, argTypes.front(), true);
  }
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;
    if (!has_trivial_destructor(unqual)) return false;
    auto constructor = select_constructor(cls, argTypes);
    if (!constructor || is_user_provided(constructor)) return false;
    auto kind = constructorKind(cls, constructor);
    if (!kind) return false;
    return has_trivial_constructor(*this, cls, *kind, false, constructor);
  }
  if (is_array(unqual)) {
    return is_trivially_constructible(remove_all_extents(unqual), {});
  }
  return false;
}

auto TypeTraits::select_assignment_operator(const Type* to, const Type* from)
    -> FunctionSymbol* {
  if (!to || !from) return nullptr;

  auto lhs = declval(to);
  auto rhs = declval(from);

  OverloadResolution resolution{unit_};
  auto selected = resolution.lookupOperator(
      nullptr, lhs->type, TokenKind::T_EQUAL, rhs->type, lhs, rhs);

  if (resolution.wasLastLookupAmbiguous()) return nullptr;
  if (selected && unit_->typeTraits().is_deleted_function(selected))
    return nullptr;
  if (!is_accessible_from_unrelated_context(selected)) return nullptr;
  return selected;
}

namespace {
struct SwapResult {
  bool valid = false;
  bool nothrow = false;
};

class SwappableTypes {
  struct Entry {
    const Type* first;
    const Type* second;
    SwapResult result;
  };

  TypeTraits& traits_;
  TranslationUnit* unit_;
  Control* control_;
  std::vector<Entry> results_;

  [[nodiscard]] auto standardPattern(FunctionSymbol* function) const -> bool {
    if (!traits_.is_std_namespace(function->parent())) return false;
    if (!function->isTemplatePattern()) return false;
    auto type = type_cast<FunctionType>(function->type());
    if (!type || type->parameterTypes().size() != 2) return false;
    auto first = type->parameterTypes()[0];
    if (!traits_.is_same(first, type->parameterTypes()[1])) return false;
    if (!traits_.is_lvalue_reference(first)) return false;
    auto element = traits_.remove_reference(first);
    if (type_cast<TypeParameterType>(element)) return true;
    auto array = type_cast<UnresolvedBoundedArrayType>(element);
    return array && type_cast<TypeParameterType>(array->elementType());
  }

  [[nodiscard]] auto standardFunctions() const -> std::vector<FunctionSymbol*> {
    std::vector<FunctionSymbol*> functions;
    auto stdNamespace = qualifiedLookupNamespace(
        unit_->globalScope(), control_->getIdentifier("std"));
    if (!stdNamespace) return functions;
    auto found = qualifiedLookupIncludingInlineNamespaces(
        control_, stdNamespace, control_->getIdentifier("swap"));
    for (auto function : views::each_function(found)) {
      if (standardPattern(function)) addOverloadCandidate(functions, function);
    }
    return functions;
  }

  [[nodiscard]] auto standardResult(const Type* first, const Type* second)
      -> SwapResult {
    if (!traits_.is_lvalue_reference(first)) return {};
    if (!traits_.is_same(first, second)) return {};
    auto type = traits_.remove_reference(first);
    if (auto array = type_cast<BoundedArrayType>(type)) {
      auto element = traits_.add_lvalue_reference(array->elementType());
      return test(element, element);
    }
    auto source = traits_.add_rvalue_reference(type);
    if (!traits_.is_constructible(type, std::array{source})) return {};
    if (!traits_.is_assignable(first, source)) return {};
    auto nothrow = traits_.is_nothrow_constructible(type, std::array{source}) &&
                   traits_.is_nothrow_assignable(first, source);
    return {true, nothrow};
  }

  [[nodiscard]] auto standardCandidate(
      const Type* first, bool nothrow, OverloadResolution& resolution,
      std::span<ExpressionAST* const> arguments) -> Candidate {
    auto arena = unit_->arena();
    auto scope = control_->newBlockSymbol(unit_->globalScope(), {});
    auto pattern = control_->newFunctionSymbol(scope, {});
    pattern->setName(control_->getIdentifier("swap"));
    auto parameters = control_->newTemplateParametersSymbol(pattern, {});
    auto typeParameter =
        control_->newTypeParameterSymbol(parameters, {}, 0, 0, false);
    auto typeAst = TypenameTypeParameterAST::create(arena);
    typeAst->symbol = typeParameter;
    parameters->addSymbol(typeParameter);
    List<TemplateParameterAST*>* parameterList =
        make_list_node(arena, static_cast<TemplateParameterAST*>(typeAst));
    const Type* parameterType = typeParameter->type();
    auto elementType = parameterType;
    if (traits_.is_bounded_array(traits_.remove_reference(first))) {
      auto bound = control_->newNonTypeParameterSymbol(parameters, {});
      bound->setIndex(1);
      bound->setObjectType(control_->getSizeType());
      parameters->addSymbol(bound);
      auto boundAst = NonTypeTemplateParameterAST::create(arena);
      boundAst->symbol = bound;
      boundAst->index = 1;
      boundAst->declaration = ParameterDeclarationAST::create(arena);
      boundAst->declaration->type = control_->getSizeType();
      parameterList->next =
          make_list_node(arena, static_cast<TemplateParameterAST*>(boundAst));
      auto expression = IdExpressionAST::create(arena);
      expression->symbol = bound;
      expression->type = control_->getSizeType();
      expression->valueCategory = ValueCategory::kPrValue;
      parameterType = control_->getUnresolvedBoundedArrayType(
          unit_, parameterType, expression);
    }
    parameterType = traits_.add_lvalue_reference(parameterType);
    pattern->setType(control_->getFunctionType(control_->getVoidType(),
                                               {parameterType, parameterType}));
    pattern->setTemplateParameters(parameters);
    pattern->setTemplateDeclaration(TemplateDeclarationAST::create(
        arena, parameterList,
        RequiresClauseAST::create(
            arena, standardConstraint(elementType,
                                      traits_.is_bounded_array(
                                          traits_.remove_reference(first)))),
        nullptr, parameters, 0));
    auto function = control_->newFunctionSymbol(scope, {});
    function->setName(pattern->name());
    function->setType(control_->getFunctionType(
        control_->getVoidType(), {first, first}, false, CvQualifiers::kNone,
        RefQualifier::kNone, nothrow));
    auto typeArgument = traits_.remove_reference(first);
    std::vector<TemplateArgument> deducedArguments{typeArgument};
    if (auto array = unqualified_cast<BoundedArrayType>(typeArgument)) {
      deducedArguments.front() = array->elementType();
      deducedArguments.emplace_back(ConstValue{ConstInt{array->size()}});
    }
    pattern->addSpecialization(unit_, std::move(deducedArguments), function);
    auto candidate = resolution.buildCallCandidate(
        function, type_cast<FunctionType>(function->type()), arguments);
    candidate->fromTemplate = true;
    return *candidate;
  }

  [[nodiscard]] auto traitConstraint(BuiltinTypeTraitKind kind,
                                     std::span<const Type* const> types)
      -> ExpressionAST* {
    auto arena = unit_->arena();
    auto expression = TypeTraitExpressionAST::create(arena);
    expression->typeTrait = kind;
    expression->type = control_->getBoolType();
    expression->valueCategory = ValueCategory::kPrValue;
    auto tail = &expression->typeIdList;
    for (auto type : types) {
      auto typeId = TypeIdAST::create(arena);
      typeId->type = type;
      *tail = make_list_node(arena, typeId);
      tail = &(*tail)->next;
    }
    return expression;
  }

  [[nodiscard]] auto standardConstraint(const Type* type, bool array)
      -> ExpressionAST* {
    auto reference = traits_.add_lvalue_reference(type);
    if (array)
      return traitConstraint(BuiltinTypeTraitKind::T___IS_SWAPPABLE_WITH,
                             std::array{reference, reference});
    auto source = traits_.add_rvalue_reference(type);
    auto expression = BinaryExpressionAST::create(unit_->arena());
    expression->op = TokenKind::T_AMP_AMP;
    expression->leftExpression = traitConstraint(
        BuiltinTypeTraitKind::T___IS_CONSTRUCTIBLE, std::array{type, source});
    expression->rightExpression = traitConstraint(
        BuiltinTypeTraitKind::T___IS_ASSIGNABLE, std::array{reference, source});
    expression->type = control_->getBoolType();
    expression->valueCategory = ValueCategory::kPrValue;
    return expression;
  }

  [[nodiscard]] auto callResult(Candidate& candidate,
                                std::span<ExpressionAST* const> arguments)
      -> SwapResult {
    auto function = candidate.symbol;
    if (unit_->typeTraits().is_deleted_function(function)) return {};
    StandardConversion conversions{unit_};
    conversions.setAccessingScope(unit_->globalScope());
    for (const auto& sequence : candidate.conversions) {
      if (!conversions.isAccessible(sequence)) return {};
    }
    auto arena = unit_->arena();
    ASTRewriter::completeDeducedReturnType(unit_, function);
    (void)traits_.is_nothrow_function(function);
    auto type = type_cast<FunctionType>(function->type());
    const auto& parameters = type->parameterTypes();
    for (std::size_t index = 0;
         index < arguments.size() && index < parameters.size(); ++index) {
      if (traits_.is_reference(parameters[index])) continue;
      auto source = traits_.decltype_of(arguments[index]);
      if (!traits_.is_convertible(source, parameters[index])) return {};
    }
    auto callee = ThisExpressionAST::create(arena, ValueCategory::kPrValue,
                                            control_->getPointerType(type));
    auto call = CallExpressionAST::create(arena);
    call->baseExpression = callee;
    auto tail = &call->expressionList;
    for (auto argument : arguments) {
      *tail = make_list_node(arena, argument);
      tail = &(*tail)->next;
    }
    TypeChecker check{unit_};
    check.setScope(unit_->globalScope());
    check.append_default_arguments(function, &call->expressionList, {});
    ExpressionAST* expression = call;
    check.check(&expression);
    if (!expression->type) return {};
    return {true, !TypeChecker::isPotentiallyThrowing(expression, unit_)};
  }

  [[nodiscard]] auto direction(const Type* first, const Type* second)
      -> SwapResult {
    if (traits_.is_void(traits_.remove_reference(first)) ||
        traits_.is_void(traits_.remove_reference(second)))
      return {};
    auto functions = standardFunctions();
    auto hasStandard = !functions.empty();
    for (auto function :
         argumentDependentLookup(unit_, control_->getIdentifier("swap"),
                                 std::array{first, second})) {
      addOverloadCandidate(functions, function);
    }
    auto arguments =
        std::array{traits_.declval(first), traits_.declval(second)};
    OverloadResolution resolution{unit_};
    resolution.setAccessingScope(unit_->globalScope());
    std::vector<Candidate> candidates;
    for (auto function : functions) {
      auto candidate = resolution.buildCallCandidate(function, arguments);
      if (candidate) candidates.push_back(std::move(*candidate));
    }
    if (!hasStandard) {
      auto standard = standardResult(first, second);
      if (standard.valid)
        candidates.push_back(
            standardCandidate(first, standard.nothrow, resolution, arguments));
    }
    auto result = resolution.selectBestViableFunction(candidates, true);
    if (!result.best || result.ambiguous) return {};
    return callResult(*result.best, arguments);
  }

 public:
  explicit SwappableTypes(TypeTraits& traits)
      : traits_(traits), unit_(traits.unit()), control_(traits.control()) {}

  [[nodiscard]] auto test(const Type* first, const Type* second) -> SwapResult {
    for (const auto& entry : results_) {
      if (entry.first == first && entry.second == second) return entry.result;
    }
    auto result = direction(first, second);
    if (result.valid && first != second) {
      auto reverse = direction(second, first);
      result.valid = reverse.valid;
      result.nothrow &= reverse.nothrow;
    }
    results_.push_back({first, second, result});
    return result;
  }
};
}  // namespace

auto TypeTraits::is_swappable_with(const Type* first, const Type* second)
    -> bool {
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  SilentDiagnosticsScope diagnostics{unit_};
  auto result = SwappableTypes{*this}.test(first, second);
  diagnostics.finish();
  return !diagnostics.hadError() && result.valid;
}

auto TypeTraits::is_nothrow_swappable_with(const Type* first,
                                           const Type* second) -> bool {
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{unit_, false};
  SilentDiagnosticsScope diagnostics{unit_};
  auto result = SwappableTypes{*this}.test(first, second);
  diagnostics.finish();
  return !diagnostics.hadError() && result.valid && result.nothrow;
}

auto TypeTraits::is_swappable(const Type* type) -> bool {
  auto reference = add_lvalue_reference(type);
  return is_lvalue_reference(reference) &&
         is_swappable_with(reference, reference);
}

auto TypeTraits::is_nothrow_swappable(const Type* type) -> bool {
  auto reference = add_lvalue_reference(type);
  return is_lvalue_reference(reference) &&
         is_nothrow_swappable_with(reference, reference);
}

auto TypeTraits::is_assignable(const Type* to, const Type* from) -> bool {
  if (!to || !from) return false;

  auto targetType = remove_reference(to);
  auto target = remove_cv(targetType);

  if (auto classType = type_cast<ClassType>(target)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;
    return select_assignment_operator(to, from) != nullptr;
  }

  if (!is_lvalue_reference(to)) return false;
  if (is_const(targetType)) return false;
  if (!is_scalar_or_vector(target)) return false;

  return is_convertible(remove_cvref(from), target);
}

auto TypeTraits::is_nothrow_assignable(const Type* to, const Type* from)
    -> bool {
  if (!is_assignable(to, from)) return false;

  if (!type_cast<ClassType>(remove_cvref(to))) {
    return is_nothrow_initialization(remove_reference(to), from, false);
  }

  auto selected = select_assignment_operator(to, from);
  if (!selected) return false;

  if (!is_nothrow_function(selected)) return false;
  auto assignmentType = type_cast<FunctionType>(selected->type());
  if (!assignmentType || assignmentType->parameterTypes().empty()) return true;
  return is_nothrow_initialization(assignmentType->parameterTypes().front(),
                                   from, false);
}

auto TypeTraits::is_address_space_generic_member(FunctionSymbol* function)
    -> bool {
  if (!function || !function->isImplicitlyDeclared()) return false;
  auto cls = symbol_cast<ClassSymbol>(function->parent());
  if (!cls) return false;
  cls = cls->resolvedDefinition();
  if (auto kind = constructorKind(cls, function);
      kind && *kind != TrivialConstructorKind::kDefault)
    return copiesAddressSpaceGenerically(*this, cls, *kind);
  if (auto kind = assignmentKind(cls, function))
    return copiesAddressSpaceGenerically(*this, cls, *kind);
  return false;
}

auto TypeTraits::is_trivial_special_member(FunctionSymbol* function) -> bool {
  if (!function) return false;
  if (auto principal = function->structorPrincipal()) function = principal;
  if (is_user_provided(function)) return false;
  auto cls = symbol_cast<ClassSymbol>(function->parent());
  if (!cls) return false;
  cls = cls->resolvedDefinition();
  if (function->isDestructor())
    return is_trivially_destructible_class(*this, cls);
  if (auto kind = constructorKind(cls, function))
    return has_trivial_constructor(*this, cls, *kind, false, function);
  if (auto kind = assignmentKind(cls, function))
    return has_trivial_assignment(*this, cls, *kind);
  return false;
}

auto TypeTraits::is_trivially_assignable(const Type* to, const Type* from)
    -> bool {
  if (!is_assignable(to, from)) return false;
  auto unqual = remove_cvref(to);
  if (is_scalar_or_vector(unqual))
    return is_trivial_initialization(remove_reference(to), from, false);
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;
    auto selected = select_assignment_operator(to, from);
    if (!selected) return false;
    auto kind = assignmentKind(cls, selected);
    if (!kind) return false;
    return has_trivial_assignment(*this, cls, *kind);
  }
  return false;
}

auto TypeTraits::is_trivially_copyable(const Type* type) -> bool {
  auto ty = remove_cv(remove_all_extents(type));
  if (is_scalar_or_vector(ty)) return true;
  if (auto classType = type_cast<ClassType>(ty)) {
    return is_trivially_copyable_class(*this, classType->definition());
  }
  return false;
}

namespace {
class UniqueObjectRepresentations {
  struct Bits {
    std::uint64_t first;
    std::uint64_t last;
  };

  TypeTraits& traits_;
  const MemoryLayout& memory_;

  struct Collect {
    UniqueObjectRepresentations& query;
    std::uint64_t offset;
    std::vector<Bits>& bits;

    template <typename T>
    [[nodiscard]] auto operator()(const T* type) const -> bool {
      if (auto representation = query.traits_.integral_representation(type)) {
        auto size = query.memory_.sizeOf(type);
        if (!size || representation->bits != std::uint64_t{*size} * 8)
          return false;
        return add(type);
      }
      if (query.traits_.is_pointer(type) ||
          query.traits_.is_member_pointer(type))
        return add(type);
      return false;
    }

    [[nodiscard]] auto operator()(const ClassType* type) const -> bool {
      return query.collectClass(type->definition(), offset, bits);
    }

    [[nodiscard]] auto operator()(const BoundedArrayType* type) const -> bool {
      return query.test(type->elementType()) && add(type);
    }

    [[nodiscard]] auto operator()(const VectorType* type) const -> bool {
      auto size = query.memory_.sizeOf(type);
      if (!size || *size != query.traits_.vector_width_in_bytes(type))
        return false;
      return query.test(type->elementType()) && add(type);
    }

    [[nodiscard]] auto operator()(const ComplexType* type) const -> bool {
      return query.test(type->elementType()) && add(type);
    }

    [[nodiscard]] auto add(const Type* type) const -> bool {
      auto size = query.memory_.sizeOf(type);
      if (!size) return false;
      bits.push_back({offset, offset + std::uint64_t{*size} * 8});
      return true;
    }
  };

  [[nodiscard]] static auto covers(std::vector<Bits>& bits, std::uint64_t size)
      -> bool {
    std::ranges::sort(bits, {}, &Bits::first);
    std::uint64_t end = 0;
    for (auto range : bits) {
      if (range.first > end) return false;
      end = std::max(end, range.last);
    }
    return end == size;
  }

  [[nodiscard]] auto collectField(FieldSymbol* field,
                                  const ClassLayout::MemberInfo& info,
                                  std::uint64_t offset, std::vector<Bits>& bits)
      -> bool {
    offset += info.offset * 8;
    if (!field->isBitField()) return collect(field->type(), offset, bits);
    if (!field->name()) return true;
    auto representation = traits_.integral_representation(field->type());
    if (!representation) return false;
    auto width = std::min<std::uint64_t>(info.bitWidth, representation->bits);
    bits.push_back({offset + info.bitOffset, offset + info.bitOffset + width});
    return true;
  }

  [[nodiscard]] auto collectClass(ClassSymbol* cls, std::uint64_t offset,
                                  std::vector<Bits>& bits) -> bool {
    if (!cls) return false;
    traits_.require_complete_class(cls);
    cls = cls->resolvedDefinition();
    auto layout = cls->layout();
    if (!layout || !traits_.is_trivially_copyable(cls->type())) return false;

    for (auto [base, info] : layout->sortedBaseInfos()) {
      if (!collectClass(base, offset + info.offset * 8, bits)) return false;
    }

    bool hasUnionMember = false;
    for (auto field : cls->members() | views::non_static_fields) {
      if (field->isBitField() && !field->name()) continue;
      auto info = layout->getFieldInfo(field);
      if (!info) return false;
      if (!cls->isUnion()) {
        if (!collectField(field, *info, offset, bits)) return false;
        continue;
      }
      std::vector<Bits> memberBits;
      if (!collectField(field, *info, 0, memberBits)) return false;
      if (!covers(memberBits, layout->size() * 8)) return false;
      hasUnionMember = true;
    }

    if (!cls->isUnion()) return true;
    if (!hasUnionMember) return false;
    bits.push_back({offset, offset + layout->size() * 8});
    return true;
  }

  [[nodiscard]] auto collect(const Type* type, std::uint64_t offset,
                             std::vector<Bits>& bits) -> bool {
    return type && visit(Collect{*this, offset, bits}, traits_.remove_cv(type));
  }

 public:
  explicit UniqueObjectRepresentations(TypeTraits& traits)
      : traits_(traits), memory_(*traits.control()->memoryLayout()) {}

  [[nodiscard]] auto test(const Type* type) -> bool {
    type = traits_.remove_cv(traits_.remove_all_extents(type));
    traits_.require_complete_class(type);
    if (!traits_.is_trivially_copyable(type)) return false;
    auto size = memory_.sizeOf(type);
    if (!size) return false;
    std::vector<Bits> bits;
    return collect(type, 0, bits) && covers(bits, std::uint64_t{*size} * 8);
  }
};
}  // namespace

auto TypeTraits::has_unique_object_representations(const Type* type) -> bool {
  if (!control()->memoryLayout()) return false;
  return UniqueObjectRepresentations{*this}.test(type);
}

auto TypeTraits::has_constexpr_unknown_representation(const Type* type)
    -> bool {
  if (is_reference(type)) return true;
  auto element = remove_all_extents(type);
  if (is_volatile(element)) return true;
  auto unqualified = remove_cv(element);
  if (is_union(unqualified)) return true;
  if (is_pointer(unqualified) || is_member_pointer(unqualified)) return true;
  if (type_cast<BuiltinMetaInfoType>(unqualified)) return true;

  auto classType = type_cast<ClassType>(unqualified);
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls || !cls->isComplete()) return true;

  return std::ranges::any_of(aggregate_elements(cls), [this](Symbol* element) {
    return has_constexpr_unknown_representation(
        aggregate_element_type(element));
  });
}

auto TypeTraits::is_non_trivial_for_calls(const Type* type) -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType) return false;

  auto cls = classType->definition();
  if (!cls || !cls->isComplete()) return false;

  if (cls->isTrivialAbi()) return false;

  if (!has_trivial_copy_members(*this, cls, TrivialCopyKind::kForCalls))
    return true;

  return !has_non_deleted_copy_or_move_constructor(unit_, cls);
}

auto TypeTraits::is_abstract(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cvref(type));
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
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

  if (is_scalar_or_vector(unqual) || is_atomic(unqual)) return true;

  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    if (!cls || !cls->isComplete()) return false;

    auto dtor = cls->destructor();
    if (dtor && unit_->typeTraits().is_deleted_function(dtor)) return false;
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
  require_complete_class(cls);
  if (!cls || !cls->isComplete()) return false;

  auto destructor = cls->destructor();
  if (!destructor) return true;

  return is_nothrow_function(destructor);
}

auto TypeTraits::has_trivial_default_constructor(const Type* type) -> bool {
  auto unqual = remove_cv(remove_all_extents(type));
  if (is_scalar_or_vector(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    return has_trivial_constructor(*this, cls, TrivialConstructorKind::kDefault,
                                   true);
  }
  return false;
}

auto TypeTraits::has_trivial_destructor(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_reference(unqual)) return true;
  if (is_void(unqual)) return false;
  if (is_function(unqual)) return false;
  if (is_array(unqual))
    return has_trivial_destructor(remove_all_extents(unqual));
  if (is_scalar_or_vector(unqual) || is_atomic(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    require_complete_class(cls);
    return is_trivially_destructible_class(*this, cls);
  }
  if (is_enum(unqual)) return true;
  return false;
}

auto TypeTraits::is_trivially_destructible(const Type* type) -> bool {
  if (!is_destructible(type)) return false;
  return has_trivial_destructor(type);
}

auto TypeTraits::has_mutable_subobject(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(remove_all_extents(type)));
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls) return false;

  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    if (has_mutable_subobject(baseClass->type())) return true;
  }

  for (auto field : cls->members() | views::non_static_fields) {
    if (field->isMutable()) return true;
    if (has_mutable_subobject(field->type())) return true;
  }

  return false;
}

auto TypeTraits::has_const_member(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(remove_all_extents(type)));
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls) return false;

  for (auto field : cls->members() | views::non_static_fields) {
    if (is_const(field->type())) return true;
    if (has_const_member(field->type())) return true;
  }

  return false;
}

auto TypeTraits::has_virtual_destructor(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cvref(type));
  if (!classType) return false;
  auto cls = classType->definition();
  require_complete_class(cls);
  if (!cls || !cls->isComplete()) return false;
  return cls->hasVirtualDestructor();
}
}  // namespace cxx
