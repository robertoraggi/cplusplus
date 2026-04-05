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

#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {

namespace {

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
  auto operator()(const BoolType*) const -> bool { return true; }
  auto operator()(const UnsignedCharType*) const -> bool { return true; }
  auto operator()(const UnsignedShortIntType*) const -> bool { return true; }
  auto operator()(const UnsignedIntType*) const -> bool { return true; }
  auto operator()(const UnsignedLongIntType*) const -> bool { return true; }
  auto operator()(const UnsignedLongLongIntType*) const -> bool { return true; }
  auto operator()(const Char8Type*) const -> bool { return true; }
  auto operator()(const Char16Type*) const -> bool { return true; }
  auto operator()(const Char32Type*) const -> bool { return true; }
  auto operator()(const WideCharType*) const -> bool { return true; }
  auto operator()(const UnsignedInt128Type*) const -> bool { return true; }
  auto operator()(const UnsignedBitIntType*) const -> bool { return true; }

  auto operator()(const QualType* type) const -> bool {
    return visit(*this, type->elementType());
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsArray {
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
  auto operator()(const BoundedArrayType*) const -> bool { return true; }
  auto operator()(const UnresolvedBoundedArrayType*) const -> bool {
    return true;
  }

  auto operator()(const Type*) const -> bool { return false; }
};

struct IsUnboundedArray {
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

struct AddConst {
  const TypeTraits& typeTraits;

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
    return control()->getConstType(type);
  }
};

struct AddVolatile {
  const TypeTraits& typeTraits;

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
    return control()->getVolatileType(type);
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

struct RemoveCv {
  auto operator()(const QualType* type) const -> const Type* {
    return type->elementType();
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
  return type && visit(IsSigned{}, type);
}

auto TypeTraits::is_unsigned(const Type* type) const -> bool {
  return type && visit(IsUnsigned{}, type);
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
  return visit(RemoveCv{}, type);
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
  if (!type) return type;
  return visit(AddConst{*this}, type);
}

auto TypeTraits::add_volatile(const Type* type) const -> const Type* {
  if (!type) return type;
  return visit(AddVolatile{*this}, type);
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

  from = remove_cv(from);
  to = remove_cv(to);

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
  if (auto qualType = type_cast<QualType>(type)) {
    if (qualType->isConst()) {
      if (qualType->isVolatile())
        return control()->getQualType(qualType->elementType(),
                                      CvQualifiers::kVolatile);
      return qualType->elementType();
    }
  }
  return type;
}

auto TypeTraits::remove_volatile(const Type* type) const -> const Type* {
  if (auto qualType = type_cast<QualType>(type)) {
    if (qualType->isVolatile()) {
      if (qualType->isConst())
        return control()->getQualType(qualType->elementType(),
                                      CvQualifiers::kConst);
      return qualType->elementType();
    }
  }
  return type;
}

auto TypeTraits::add_cv(const Type* type, CvQualifiers cv) const
    -> const Type* {
  if (cxx::is_const(cv)) type = add_const(type);
  if (cxx::is_volatile(cv)) type = add_volatile(type);
  return type;
}

auto TypeTraits::get_cv_qualifiers(const Type* type) const -> CvQualifiers {
  if (auto qualType = type_cast<QualType>(type))
    return qualType->cvQualifiers();
  return CvQualifiers::kNone;
}

auto TypeTraits::remove_noexcept(const Type* type) const -> const Type* {
  const auto functionType = type_cast<FunctionType>(type);
  if (!functionType) return type;
  return control()->getFunctionType(
      functionType->returnType(), functionType->parameterTypes(),
      functionType->isVariadic(), functionType->cvQualifiers(),
      functionType->refQualifier(), false);
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

auto TypeTraits::is_convertible(const Type* from, const Type* to) const
    -> bool {
  if (!from || !to) return false;

  auto fromUnqual = remove_cv(from);
  auto toUnqual = remove_cv(to);

  if (is_void(fromUnqual) && is_void(toUnqual)) return true;
  if (is_void(fromUnqual) || is_void(toUnqual)) return false;
  if (is_same(fromUnqual, toUnqual)) return true;
  if (is_arithmetic(fromUnqual) && is_arithmetic(toUnqual)) return true;
  if (is_null_pointer(fromUnqual) && is_pointer(toUnqual)) return true;
  if (is_enum(fromUnqual) && is_integral(toUnqual)) return true;

  if (is_pointer(fromUnqual) && is_pointer(toUnqual)) {
    auto fromPointee = remove_pointer(fromUnqual);
    auto toPointee = remove_pointer(toUnqual);
    if (is_void(remove_cv(toPointee))) return true;
    if (is_base_of(remove_cv(toPointee), remove_cv(fromPointee))) return true;
  }

  if (auto fromClass = type_cast<ClassType>(fromUnqual)) {
    if (auto toClass = type_cast<ClassType>(toUnqual)) {
      (void)fromClass;
      (void)toClass;
      if (is_base_of(toUnqual, fromUnqual)) return true;
    }
  }

  if (type_cast<BoolType>(toUnqual)) {
    if (is_arithmetic(fromUnqual) || is_pointer(fromUnqual) ||
        is_enum(fromUnqual) || is_null_pointer(fromUnqual))
      return true;
  }

  return false;
}

auto TypeTraits::is_pod(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (is_void(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    if (cls->hasUserDeclaredConstructors()) return false;
    if (cls->hasVirtualFunctions()) return false;
    if (cls->hasVirtualBaseClasses()) return false;
    return true;
  }
  if (is_array(unqual)) return true;
  return false;
}

auto TypeTraits::is_trivial(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    auto defCtor = cls->defaultConstructor();
    if (isUserProvided(defCtor)) return false;
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
    return true;
  }
  if (is_array(unqual)) return true;
  return false;
}

auto TypeTraits::is_literal_type(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_void(unqual)) return true;
  if (is_scalar(unqual)) return true;
  if (is_reference(unqual)) return true;
  if (is_array(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    auto dtor = cls->destructor();
    if (dtor && !dtor->isDefaulted() && !dtor->isDeleted()) return false;
    return true;
  }
  return false;
}

auto TypeTraits::is_aggregate(const Type* type) -> bool {
  if (is_array(type)) return true;
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;
  if (cls->hasUserDeclaredConstructors()) return false;
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
  return true;
}

auto TypeTraits::is_empty(const Type* type) -> bool {
  auto classType = type_cast<ClassType>(remove_cv(type));
  if (!classType) return false;
  auto cls = classType->definition();
  requireCompleteClass(cls);
  if (!cls || !cls->isComplete()) return false;
  for (auto f : cls->members() | views::non_static_fields) {
    (void)f;
    return false;
  }
  if (cls->hasVirtualFunctions()) return false;
  if (cls->hasVirtualBaseClasses()) return false;
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

auto TypeTraits::is_constructible(const Type* type,
                                  std::span<const Type* const> argTypes)
    -> bool {
  if (!type) return false;
  auto unqual = remove_cv(type);

  if (is_reference(unqual)) {
    if (argTypes.size() != 1) return false;
    return true;
  }

  if (is_scalar(unqual)) {
    if (argTypes.empty()) return true;
    if (argTypes.size() == 1) return true;
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

    if (argTypes.empty()) {
      auto defCtor = cls->defaultConstructor();
      if (!defCtor) return !cls->hasUserDeclaredConstructors();
      return !defCtor->isDeleted();
    }

    if (argTypes.size() == 1) {
      auto argUnqual = remove_cvref(argTypes[0]);
      if (is_same(argUnqual, unqual)) {
        if (is_rvalue_reference(argTypes[0]) ||
            (!is_reference(argTypes[0]) && !is_const(argTypes[0]))) {
          auto moveCtor = cls->moveConstructor();
          if (moveCtor && !moveCtor->isDeleted()) return true;
        }
        auto copyCtor = cls->copyConstructor();
        if (copyCtor && !copyCtor->isDeleted()) return true;
        if (!cls->hasUserDeclaredConstructors()) return true;
      }
    }

    for (auto ctor : cls->constructors()) {
      if (ctor->isDeleted()) continue;
      auto ctorType = type_cast<FunctionType>(ctor->type());
      if (!ctorType) continue;
      auto params = ctorType->parameterTypes();
      if (params.size() == argTypes.size()) return true;
      if (ctorType->isVariadic() && params.size() <= argTypes.size())
        return true;
    }

    return false;
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

  if (is_reference(unqual)) return true;
  if (is_scalar(unqual)) return true;

  if (is_array(unqual))
    return is_nothrow_constructible(remove_all_extents(unqual), argTypes);

  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;

    if (argTypes.empty()) {
      auto defCtor = cls->defaultConstructor();
      if (!defCtor) {
        if (!cls->hasUserDeclaredConstructors()) return true;
        return false;
      }
      if (defCtor->isDeleted()) return false;
      if (defCtor->isDefaulted() || !cls->hasUserDeclaredConstructors()) {
        for (auto base : cls->baseClasses()) {
          auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
          if (!baseClass) continue;
          std::span<const Type* const> empty;
          if (!is_nothrow_constructible(baseClass->type(), empty)) return false;
        }
        return true;
      }
      auto ctorType = type_cast<FunctionType>(defCtor->type());
      return ctorType && ctorType->isNoexcept();
    }

    if (argTypes.size() == 1) {
      auto argUnqual = remove_cvref(argTypes[0]);
      if (is_same(argUnqual, unqual)) {
        if (is_rvalue_reference(argTypes[0]) ||
            (!is_reference(argTypes[0]) && !is_const(argTypes[0]))) {
          auto moveCtor = cls->moveConstructor();
          if (moveCtor && !moveCtor->isDeleted()) {
            auto ctorType = type_cast<FunctionType>(moveCtor->type());
            return ctorType && ctorType->isNoexcept();
          }
        }
        auto copyCtor = cls->copyConstructor();
        if (copyCtor && !copyCtor->isDeleted()) {
          auto ctorType = type_cast<FunctionType>(copyCtor->type());
          return ctorType && ctorType->isNoexcept();
        }
        if (!cls->hasUserDeclaredConstructors()) return true;
      }
    }

    for (auto ctor : cls->constructors()) {
      if (ctor->isDeleted()) continue;
      auto ctorType = type_cast<FunctionType>(ctor->type());
      if (!ctorType) continue;
      auto params = ctorType->parameterTypes();
      if (params.size() == argTypes.size()) return ctorType->isNoexcept();
      if (ctorType->isVariadic() && params.size() <= argTypes.size())
        return ctorType->isNoexcept();
    }

    return false;
  }

  return false;
}

auto TypeTraits::is_trivially_constructible(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    auto defCtor = cls->defaultConstructor();
    if (isUserProvided(defCtor)) return false;
    if (cls->isPolymorphic()) return false;
    if (cls->hasVirtualBaseClasses()) return false;
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      if (!is_trivially_constructible(baseClass->type())) return false;
    }
    for (auto field : cls->members() | views::non_static_fields) {
      auto fieldType = remove_all_extents(remove_cv(field->type()));
      if (auto ct = type_cast<ClassType>(fieldType)) {
        if (!is_trivially_constructible(ct->symbol()->type())) return false;
      }
    }
    return true;
  }
  if (is_array(unqual)) {
    return is_trivially_constructible(remove_all_extents(unqual));
  }
  return false;
}

auto TypeTraits::is_assignable(const Type* to, const Type* from) -> bool {
  if (!to || !from) return false;

  if (is_lvalue_reference(to)) {
    auto targetType = remove_reference(to);
    if (is_const(targetType)) return false;

    auto target = remove_cv(targetType);
    auto source = remove_cvref(from);

    if (is_scalar(target)) return is_convertible(source, target);

    if (auto classType = type_cast<ClassType>(target)) {
      auto cls = classType->definition();
      requireCompleteClass(cls);
      if (!cls || !cls->isComplete()) return false;

      if (is_same(source, target)) {
        if (is_rvalue_reference(from)) {
          auto moveOp = cls->moveAssignmentOperator();
          if (moveOp && !moveOp->isDeleted()) return true;
        }
        auto copyOp = cls->copyAssignmentOperator();
        if (copyOp && !copyOp->isDeleted()) return true;
        if (!cls->hasUserDeclaredConstructors()) return true;
        return false;
      }

      return true;
    }

    return false;
  }

  if (is_rvalue_reference(to)) {
    auto targetType = remove_reference(to);
    if (is_const(targetType)) return false;
    auto target = remove_cv(targetType);

    if (auto classType = type_cast<ClassType>(target)) {
      auto cls = classType->definition();
      requireCompleteClass(cls);
      if (!cls || !cls->isComplete()) return false;
      auto copyOp = cls->copyAssignmentOperator();
      if (copyOp && !copyOp->isDeleted()) return true;
      auto moveOp = cls->moveAssignmentOperator();
      if (moveOp && !moveOp->isDeleted()) return true;
      if (!cls->hasUserDeclaredConstructors()) return true;
      return false;
    }

    return false;
  }

  if (auto classType = type_cast<ClassType>(remove_cv(to))) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    auto copyOp = cls->copyAssignmentOperator();
    if (copyOp && !copyOp->isDeleted()) return true;
    auto moveOp = cls->moveAssignmentOperator();
    if (moveOp && !moveOp->isDeleted()) return true;
    if (!cls->hasUserDeclaredConstructors()) return true;
    return false;
  }

  return false;
}

auto TypeTraits::is_nothrow_assignable(const Type* to, const Type* from)
    -> bool {
  if (!is_assignable(to, from)) return false;
  // For scalar types, assignment never throws.
  auto target = remove_cvref(to);
  if (is_scalar(target)) return true;
  // For class types, use trivial assignability as a conservative approximation.
  return is_trivially_assignable(to, from);
}

auto TypeTraits::is_trivially_assignable(const Type* from, const Type* to)
    -> bool {
  if (!to) return false;
  auto unqual = remove_cvref(from);
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    auto cls = classType->definition();
    requireCompleteClass(cls);
    if (!cls || !cls->isComplete()) return false;
    if (is_lvalue_reference(to)) {
      auto inner = remove_cv(remove_reference(to));
      if (is_same(inner, unqual)) {
        auto op = cls->copyAssignmentOperator();
        if (!op || isUserProvided(op)) return false;
        return is_trivially_copyable_class(*this, cls);
      }
    }
    if (is_rvalue_reference(to)) {
      auto inner = remove_reference(to);
      if (is_same(inner, unqual)) {
        auto op = cls->moveAssignmentOperator();
        if (!op || isUserProvided(op)) return false;
        return is_trivially_copyable_class(*this, cls);
      }
    }
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
    return true;
  }

  if (is_enum(unqual)) return true;

  return false;
}

auto TypeTraits::is_trivially_destructible(const Type* type) -> bool {
  auto unqual = remove_cv(type);
  if (is_reference(unqual)) return true;
  if (is_void(unqual)) return false;
  if (is_function(unqual)) return false;
  if (is_unbounded_array(unqual)) return false;
  if (is_bounded_array(unqual))
    return is_trivially_destructible(remove_all_extents(unqual));
  if (is_scalar(unqual)) return true;
  if (auto classType = type_cast<ClassType>(unqual)) {
    return is_trivial(classType);
  }
  if (is_enum(unqual)) return true;
  return false;
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
