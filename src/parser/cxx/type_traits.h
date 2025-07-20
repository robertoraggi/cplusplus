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

#include <cxx/control.h>
#include <cxx/parser.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

namespace cxx {

class TypeTraits {
  Control* control_;

 public:
  explicit TypeTraits(Control* control) : control_(control) {}

  auto control() const -> Control* { return control_; }

  // primary type categories

  auto is_void(const Type* type) const -> bool {
    return type && visit(is_void_, type);
  }

  auto is_null_pointer(const Type* type) const -> bool {
    return type && visit(is_null_pointer_, type);
  }

  auto is_integral(const Type* type) const -> bool {
    return type && visit(is_integral_, type);
  }

  auto is_floating_point(const Type* type) const -> bool {
    return type && visit(is_floating_point_, type);
  }

  auto is_array(const Type* type) const -> bool {
    return type && visit(is_array_, type);
  }

  auto is_enum(const Type* type) const -> bool {
    return type && visit(is_enum_, type);
  }

  auto is_union(const Type* type) const -> bool {
    return type && visit(is_union_, type);
  }

  auto is_struct(const Type* type) const -> bool {
    return type && visit(is_struct_, type);
  }

  auto is_class(const Type* type) const -> bool {
    return type && visit(is_class_, type);
  }

  auto is_function(const Type* type) const -> bool {
    return type && visit(is_function_, type);
  }

  auto is_pointer(const Type* type) const -> bool {
    return type && visit(is_pointer_, type);
  }

  auto is_lvalue_reference(const Type* type) const -> bool {
    return type && visit(is_lvalue_reference_, type);
  }

  auto is_rvalue_reference(const Type* type) const -> bool {
    return type && visit(is_rvalue_reference_, type);
  }

  auto is_member_object_pointer(const Type* type) const -> bool {
    return type && visit(is_member_object_pointer_, type);
  }

  auto is_member_function_pointer(const Type* type) const -> bool {
    return type && visit(is_member_function_pointer_, type);
  }

  auto is_complete(const Type* type) const -> bool {
    return type && visit(is_complete_, type);
  }

  // composite type categories

  auto is_integer(const Type* type) const -> bool { return is_integral(type); }

  auto is_integral_or_unscoped_enum(const Type* type) const -> bool {
    return is_integral(type) || (is_enum(type) && !is_scoped_enum(type));
  }

  auto is_fundamental(const Type* type) const -> bool {
    return is_arithmetic(type) || is_void(type) || is_null_pointer(type);
  }

  auto is_arithmetic(const Type* type) const -> bool {
    return is_integral(type) || is_floating_point(type);
  }

  auto is_scalar(const Type* type) const -> bool {
    return is_arithmetic(type) || is_enum(type) || is_pointer(type) ||
           is_member_pointer(type) || is_null_pointer(type);
  }

  auto is_object(const Type* type) const -> bool {
    return is_scalar(type) || is_array(type) || is_union(type) ||
           is_struct(type) || is_class(type);
  }

  auto is_compound(const Type* type) const -> bool {
    return !is_fundamental(type);
  }

  auto is_reference(const Type* type) const -> bool {
    return type && visit(is_reference_, type);
  }

  auto is_member_pointer(const Type* type) const -> bool {
    return is_member_object_pointer(type) || is_member_function_pointer(type);
  }

  // type properties

  auto is_const(const Type* type) const -> bool {
    return type && visit(is_const_, type);
  }

  auto is_volatile(const Type* type) const -> bool {
    return type && visit(is_volatile_, type);
  }

  auto is_signed(const Type* type) const -> bool {
    return type && visit(is_signed_, type);
  }

  auto is_unsigned(const Type* type) const -> bool {
    return type && visit(is_unsigned_, type);
  }

  auto is_bounded_array(const Type* type) const -> bool {
    return type && visit(is_bounded_array_, type);
  }

  auto is_unbounded_array(const Type* type) const -> bool {
    return type && visit(is_unbounded_array_, type);
  }

  auto is_scoped_enum(const Type* type) const -> bool {
    return type && visit(is_scoped_enum_, type);
  }

  // references
  auto remove_reference(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(remove_reference_, type);
  }

  auto add_lvalue_reference(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(add_lvalue_reference_, type);
  }

  auto add_rvalue_reference(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(add_rvalue_reference_, type);
  }

  // arrays
  auto remove_extent(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(remove_extent_, type);
  }

  auto get_element_type(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(get_element_type_, type);
  }

  // cv qualifiers

  auto remove_cv(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(remove_cv_, type);
  }

  auto remove_cvref(const Type* type) const -> const Type* {
    if (!type) return type;
    return remove_cv(remove_reference(type));
  }

  auto add_const_ref(const Type* type) const -> const Type* {
    if (!type) return type;
    return add_lvalue_reference(add_const(type));
  }

  auto add_const(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(add_const_, type);
  }

  auto add_volatile(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(add_volatile_, type);
  }

  // pointers
  auto remove_pointer(const Type* type) const -> const Type* {
    if (auto ptrTy = type_cast<PointerType>(remove_cv(type)))
      return ptrTy->elementType();
    return type;
  }

  auto add_pointer(const Type* type) const -> const Type* {
    if (!type) return type;
    return visit(add_pointer_, type);
  }

  // type relationships
  auto is_same(const Type* a, const Type* b) const -> bool {
    if (a == b) return true;
    if (!a || !b) return false;
    if (a->kind() != b->kind()) return false;
#define PROCESS_TYPE(K)                             \
  case TypeKind::k##K:                              \
    return is_same_(static_cast<const K##Type*>(a), \
                    static_cast<const K##Type*>(b));
    switch (a->kind()) {
      CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
      default:
        return false;
    }
#undef PROCESS_TYPE
  }

  auto decay(const Type* type) const -> const Type* {
    if (!type) return type;
    auto noref = remove_reference(type);
    if (is_array(noref)) return add_pointer(remove_extent(noref));
    if (is_function(noref)) return add_pointer(noref);
    return remove_cvref(noref);
  }

 private:
  struct {
    auto operator()(const VoidType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_void_;

  struct {
    auto operator()(const NullptrType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_null_pointer_;

  struct {
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

    auto operator()(const UnsignedLongLongIntType*) const -> bool {
      return true;
    }

    auto operator()(const CharType*) const -> bool { return true; }

    auto operator()(const Char8Type*) const -> bool { return true; }

    auto operator()(const Char16Type*) const -> bool { return true; }

    auto operator()(const Char32Type*) const -> bool { return true; }

    auto operator()(const WideCharType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }

  } is_integral_;

  struct {
    auto operator()(const FloatType*) const -> bool { return true; }

    auto operator()(const DoubleType*) const -> bool { return true; }

    auto operator()(const LongDoubleType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_floating_point_;

  struct {
    auto operator()(const SignedCharType*) const -> bool { return true; }

    auto operator()(const ShortIntType*) const -> bool { return true; }

    auto operator()(const IntType*) const -> bool { return true; }

    auto operator()(const LongIntType*) const -> bool { return true; }

    auto operator()(const LongLongIntType*) const -> bool { return true; }

    auto operator()(const CharType*) const -> bool { return true; }

    auto operator()(const FloatType*) const -> bool { return true; }

    auto operator()(const DoubleType*) const -> bool { return true; }

    auto operator()(const LongDoubleType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_signed_;

  struct {
    auto operator()(const BoolType*) const -> bool { return true; }

    auto operator()(const UnsignedCharType*) const -> bool { return true; }

    auto operator()(const UnsignedShortIntType*) const -> bool { return true; }

    auto operator()(const UnsignedIntType*) const -> bool { return true; }

    auto operator()(const UnsignedLongIntType*) const -> bool { return true; }

    auto operator()(const UnsignedLongLongIntType*) const -> bool {
      return true;
    }

    auto operator()(const Char8Type*) const -> bool { return true; }

    auto operator()(const Char16Type*) const -> bool { return true; }

    auto operator()(const Char32Type*) const -> bool { return true; }

    auto operator()(const WideCharType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_unsigned_;

  struct {
    auto operator()(const UnboundedArrayType*) const -> bool { return true; }

    auto operator()(const BoundedArrayType*) const -> bool { return true; }

    auto operator()(const UnresolvedBoundedArrayType*) const -> bool {
      return true;
    }

    auto operator()(auto) const -> bool { return false; }
  } is_array_;

  struct {
    auto operator()(const EnumType*) const -> bool { return true; }

    auto operator()(const ScopedEnumType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_enum_;

  struct {
    auto operator()(const ScopedEnumType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_scoped_enum_;

  struct {
    auto operator()(const ClassType*) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_class_;

  struct {
    auto operator()(const ClassType* classType) const -> bool {
      return classType->isUnion();
    }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_union_;

  struct {
    auto operator()(const ClassType* classType) const -> bool {
      return classType->isStruct();
    }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_struct_;

  struct {
    auto operator()(const FunctionType*) const -> bool { return true; }

    auto operator()(auto) const -> bool { return false; }
  } is_function_;

  struct {
    auto operator()(const PointerType* type) const -> bool { return true; }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_pointer_;

  struct {
    auto operator()(const MemberObjectPointerType*) const -> bool {
      return true;
    }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_member_object_pointer_;

  struct {
    auto operator()(const MemberFunctionPointerType*) const -> bool {
      return true;
    }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return false; }
  } is_member_function_pointer_;

  struct {
    auto operator()(const BoundedArrayType*) const -> bool { return true; }

    auto operator()(const UnresolvedBoundedArrayType*) const -> bool {
      return true;
    }

    auto operator()(auto) const -> bool { return false; }
  } is_bounded_array_;

  struct {
    auto operator()(const UnboundedArrayType*) const -> bool { return true; }

    auto operator()(auto) const -> bool { return false; }
  } is_unbounded_array_;

  struct {
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

    auto operator()(auto) const -> bool { return false; }
  } is_const_;

  struct {
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

    auto operator()(auto) const -> bool { return false; }
  } is_volatile_;

  struct {
    auto operator()(const LvalueReferenceType*) const -> bool { return true; }

    auto operator()(auto) const -> bool { return false; }
  } is_lvalue_reference_;

  struct {
    auto operator()(const RvalueReferenceType*) const -> bool { return true; }

    auto operator()(auto) const -> bool { return false; }
  } is_rvalue_reference_;

  struct {
    auto operator()(const LvalueReferenceType*) const -> bool { return true; }

    auto operator()(const RvalueReferenceType*) const -> bool { return true; }

    auto operator()(auto) const -> bool { return false; }
  } is_reference_;

  struct {
    auto operator()(const VoidType*) const -> bool { return false; }

    auto operator()(const ClassType* type) const -> bool {
      return type->isComplete();
    }

    auto operator()(const QualType* type) const -> bool {
      return visit(*this, type->elementType());
    }

    auto operator()(auto) const -> bool { return true; }
  } is_complete_;

  struct {
    auto operator()(const LvalueReferenceType* type) const -> const Type* {
      return type->elementType();
    }

    auto operator()(const RvalueReferenceType* type) const -> const Type* {
      return type->elementType();
    }

    auto operator()(auto type) const -> const Type* { return type; }

  } remove_reference_;

  struct {
    TypeTraits& traits;

    auto control() const -> Control* { return traits.control(); }

    auto operator()(const VoidType* type) const -> const Type* { return type; }

    auto operator()(const QualType* type) const -> const Type* {
      if (traits.is_void(type->elementType())) return type;
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

  } add_lvalue_reference_{*this};

  struct {
    TypeTraits& traits;

    auto control() const -> Control* { return traits.control(); }

    auto operator()(const VoidType* type) const -> const Type* { return type; }

    auto operator()(const QualType* type) const -> const Type* {
      if (traits.is_void(type->elementType())) return type;
      return control()->getRvalueReferenceType(type);
    }

    auto operator()(const RvalueReferenceType* type) const -> const Type* {
      return control()->getRvalueReferenceType(type->elementType());
    }

    auto operator()(const FunctionType* type) const -> const Type* {
      if (type->cvQualifiers() != CvQualifiers::kNone) return type;
      if (type->refQualifier() != RefQualifier::kNone) return type;
      return control()->getRvalueReferenceType(type);
    }

    auto operator()(auto type) const -> const Type* {
      return control()->getRvalueReferenceType(type);
    }

  } add_rvalue_reference_{*this};

  struct {
    TypeTraits& traits;

    auto control() const -> Control* { return traits.control(); }

    auto operator()(const BoundedArrayType* type) const -> const Type* {
      auto elementType = visit(*this, type->elementType());
      return control()->getBoundedArrayType(elementType, type->size());
    }

    auto operator()(const UnboundedArrayType* type) const -> const Type* {
      auto elementType = visit(*this, type->elementType());
      return control()->getUnboundedArrayType(elementType);
    }

    auto operator()(const UnresolvedBoundedArrayType* type) const
        -> const Type* {
      auto elementType = visit(*this, type->elementType());
      return control()->getUnresolvedBoundedArrayType(
          type->translationUnit(), elementType, type->size());
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

  } add_const_{*this};

  struct {
    TypeTraits& traits;

    auto control() const -> Control* { return traits.control(); }

    auto operator()(const BoundedArrayType* type) const -> const Type* {
      auto elementType = visit(*this, type->elementType());
      return control()->getBoundedArrayType(elementType, type->size());
    }

    auto operator()(const UnboundedArrayType* type) const -> const Type* {
      auto elementType = visit(*this, type->elementType());
      return control()->getUnboundedArrayType(elementType);
    }

    auto operator()(const UnresolvedBoundedArrayType* type) const
        -> const Type* {
      auto elementType = visit(*this, type->elementType());
      return control()->getUnresolvedBoundedArrayType(
          type->translationUnit(), elementType, type->size());
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

  } add_volatile_{*this};

  struct {
    auto operator()(const BoundedArrayType* type) const -> const Type* {
      return type->elementType();
    }

    auto operator()(const UnboundedArrayType* type) const -> const Type* {
      return type->elementType();
    }

    auto operator()(const UnresolvedBoundedArrayType* type) const
        -> const Type* {
      return type->elementType();
    }

    auto operator()(auto type) const -> const Type* { return type; }
  } remove_extent_;

  struct {
    TypeTraits& traits;

    auto operator()(const BoundedArrayType* type) const -> const Type* {
      return type->elementType();
    }

    auto operator()(const UnboundedArrayType* type) const -> const Type* {
      return type->elementType();
    }

    auto operator()(const UnresolvedBoundedArrayType* type) const
        -> const Type* {
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

    auto operator()(auto type) const -> const Type* { return nullptr; }
  } get_element_type_{*this};

  struct {
    auto operator()(const QualType* type) const -> const Type* {
      return type->elementType();
    }

    auto operator()(auto type) const -> const Type* { return type; }
  } remove_cv_;

  struct {
    TypeTraits& traits;

    auto control() const -> Control* { return traits.control(); }

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
  } add_pointer_{*this};

  struct {
    TypeTraits& traits;

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

    auto operator()(const SignedCharType*, const SignedCharType*) const
        -> bool {
      return true;
    }

    auto operator()(const ShortIntType*, const ShortIntType*) const -> bool {
      return true;
    }

    auto operator()(const IntType*, const IntType*) const -> bool {
      return true;
    }

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

    auto operator()(const UnsignedLongIntType*,
                    const UnsignedLongIntType*) const -> bool {
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

    auto operator()(const LongDoubleType*, const LongDoubleType*) const
        -> bool {
      return true;
    }

    auto operator()(const QualType* type, const QualType* otherType) const
        -> bool {
      if (type->cvQualifiers() != otherType->cvQualifiers()) return false;
      return traits.is_same(type->elementType(), otherType->elementType());
    }

    auto operator()(const BoundedArrayType* type,
                    const BoundedArrayType* otherType) const -> bool {
      if (type->size() != otherType->size()) return false;
      return traits.is_same(type->elementType(), otherType->elementType());
    }

    auto operator()(const UnboundedArrayType* type,
                    const UnboundedArrayType* otherType) const -> bool {
      return traits.is_same(type->elementType(), otherType->elementType());
    }

    auto operator()(const PointerType* type, const PointerType* otherType) const
        -> bool {
      return traits.is_same(type->elementType(), otherType->elementType());
    }

    auto operator()(const LvalueReferenceType* type,
                    const LvalueReferenceType* otherType) const -> bool {
      return traits.is_same(type->elementType(), otherType->elementType());
    }

    auto operator()(const RvalueReferenceType* type,
                    const RvalueReferenceType* otherType) const -> bool {
      return traits.is_same(type->elementType(), otherType->elementType());
    }

    auto operator()(const FunctionType* type,
                    const FunctionType* otherType) const -> bool {
      if (type->isVariadic() != otherType->isVariadic()) return false;
      if (type->refQualifier() != otherType->refQualifier()) return false;
      if (type->cvQualifiers() != otherType->cvQualifiers()) return false;
      if (type->isNoexcept() != otherType->isNoexcept()) return false;
      if (type->parameterTypes().size() != otherType->parameterTypes().size())
        return false;
      if (!traits.is_same(type->returnType(), otherType->returnType()))
        return false;
      for (std::size_t i = 0; i < type->parameterTypes().size(); ++i) {
        if (!traits.is_same(type->parameterTypes()[i],
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
      if (!traits.is_same(type->classType(), otherType->classType()))
        return false;
      if (!traits.is_same(type->elementType(), otherType->elementType()))
        return false;
      return true;
    }

    auto operator()(const MemberFunctionPointerType* type,
                    const MemberFunctionPointerType* otherType) const -> bool {
      if (!traits.is_same(type->classType(), otherType->classType()))
        return false;
      if (!traits.is_same(type->functionType(), otherType->functionType()))
        return false;
      return true;
    }

    auto operator()(const NamespaceType* type,
                    const NamespaceType* otherType) const -> bool {
      return type->symbol() == otherType->symbol();
    }

    auto operator()(const TypeParameterType* type,
                    const TypeParameterType* otherType) const -> bool {
      return type->symbol() == otherType->symbol();
    }

    auto operator()(const TemplateTypeParameterType* type,
                    const TemplateTypeParameterType* otherType) const -> bool {
      return type->symbol() == otherType->symbol();
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

  } is_same_{*this};
};

}  // namespace cxx
