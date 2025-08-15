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

#include <cxx/ast_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <tuple>
#include <vector>

namespace cxx {

class Type {
 public:
  explicit Type(TypeKind kind) : kind_(kind) {}
  virtual ~Type() = default;

  [[nodiscard]] auto kind() const -> TypeKind { return kind_; }

 private:
  TypeKind kind_;
};

class BuiltinVaListType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kBuiltinVaList;
  BuiltinVaListType() : Type(Kind) {}
};

class VoidType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kVoid;
  VoidType() : Type(Kind) {}
};

class NullptrType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kNullptr;
  NullptrType() : Type(Kind) {}
};

class DecltypeAutoType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kDecltypeAuto;
  DecltypeAutoType() : Type(Kind) {}
};

class AutoType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kAuto;
  AutoType() : Type(Kind) {}
};

class BoolType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kBool;
  BoolType() : Type(Kind) {}
};

class SignedCharType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kSignedChar;
  SignedCharType() : Type(Kind) {}
};

class ShortIntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kShortInt;
  ShortIntType() : Type(Kind) {}
};

class IntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kInt;
  IntType() : Type(Kind) {}
};

class LongIntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kLongInt;
  LongIntType() : Type(Kind) {}
};

class LongLongIntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kLongLongInt;
  LongLongIntType() : Type(Kind) {}
};

class Int128Type final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kInt128;
  Int128Type() : Type(Kind) {}
};

class UnsignedCharType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnsignedChar;
  UnsignedCharType() : Type(Kind) {}
};

class UnsignedShortIntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnsignedShortInt;
  UnsignedShortIntType() : Type(Kind) {}
};

class UnsignedIntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnsignedInt;
  UnsignedIntType() : Type(Kind) {}
};

class UnsignedLongIntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnsignedLongInt;
  UnsignedLongIntType() : Type(Kind) {}
};

class UnsignedLongLongIntType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnsignedLongLongInt;
  UnsignedLongLongIntType() : Type(Kind) {}
};

class UnsignedInt128Type final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnsignedInt128;
  UnsignedInt128Type() : Type(Kind) {}
};

class CharType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kChar;
  CharType() : Type(Kind) {}
};

class Char8Type final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kChar8;
  Char8Type() : Type(Kind) {}
};

class Char16Type final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kChar16;
  Char16Type() : Type(Kind) {}
};

class Char32Type final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kChar32;
  Char32Type() : Type(Kind) {}
};

class WideCharType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kWideChar;
  WideCharType() : Type(Kind) {}
};

class FloatType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kFloat;
  FloatType() : Type(Kind) {}
};

class DoubleType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kDouble;
  DoubleType() : Type(Kind) {}
};

class LongDoubleType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kLongDouble;
  LongDoubleType() : Type(Kind) {}
};

class QualType final : public Type,
                       public std::tuple<const Type*, CvQualifiers> {
 public:
  static constexpr TypeKind Kind = TypeKind::kQual;

  QualType(const Type* elementType, CvQualifiers cvQualifiers)
      : Type(Kind), tuple(elementType, cvQualifiers) {}

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto cvQualifiers() const -> CvQualifiers {
    return std::get<1>(*this);
  }

  [[nodiscard]] auto isConst() const -> bool {
    return cvQualifiers() == CvQualifiers::kConst ||
           cvQualifiers() == CvQualifiers::kConstVolatile;
  }

  [[nodiscard]] auto isVolatile() const -> bool {
    return cvQualifiers() == CvQualifiers::kVolatile ||
           cvQualifiers() == CvQualifiers::kConstVolatile;
  }
};

class BoundedArrayType final : public Type,
                               public std::tuple<const Type*, std::size_t> {
 public:
  static constexpr TypeKind Kind = TypeKind::kBoundedArray;

  BoundedArrayType(const Type* elementType, std::size_t size)
      : Type(Kind), tuple(elementType, size) {}

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<0>(*this);
  }
  [[nodiscard]] auto size() const -> std::size_t { return std::get<1>(*this); }
};

class UnboundedArrayType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnboundedArray;

  explicit UnboundedArrayType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<0>(*this);
  }
};

class PointerType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kPointer;

  explicit PointerType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<0>(*this);
  }
};

class LvalueReferenceType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kLvalueReference;

  explicit LvalueReferenceType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<0>(*this);
  }
};

class RvalueReferenceType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kRvalueReference;

  explicit RvalueReferenceType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<0>(*this);
  }
};

class OverloadSetType final : public Type,
                              public std::tuple<OverloadSetSymbol*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kOverloadSet;

  explicit OverloadSetType(OverloadSetSymbol* symbol)
      : Type(Kind), tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> OverloadSetSymbol* {
    return std::get<0>(*this);
  }
};

class FunctionType final
    : public Type,
      public std::tuple<const Type*, std::vector<const Type*>, bool,
                        CvQualifiers, RefQualifier, bool> {
 public:
  static constexpr TypeKind Kind = TypeKind::kFunction;

  FunctionType(const Type* returnType, std::vector<const Type*> parameterTypes,
               bool isVariadic, CvQualifiers cvQualifiers,
               RefQualifier refQualifier, bool isNoexcept)
      : Type(Kind),
        tuple(returnType, std::move(parameterTypes), isVariadic, cvQualifiers,
              refQualifier, isNoexcept) {}

  [[nodiscard]] auto returnType() const -> const Type* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto parameterTypes() const -> const std::vector<const Type*>& {
    return std::get<1>(*this);
  }

  [[nodiscard]] auto isVariadic() const -> bool { return std::get<2>(*this); }

  [[nodiscard]] auto cvQualifiers() const -> CvQualifiers {
    return std::get<3>(*this);
  }

  [[nodiscard]] auto refQualifier() const -> RefQualifier {
    return std::get<4>(*this);
  }

  [[nodiscard]] auto isNoexcept() const -> bool { return std::get<5>(*this); }
};

class ClassType final : public Type, public std::tuple<ClassSymbol*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kClass;

  explicit ClassType(ClassSymbol* symbol) : Type(Kind), tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> ClassSymbol* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto isComplete() const -> bool;
  [[nodiscard]] auto isUnion() const -> bool;
};

class EnumType final : public Type, public std::tuple<EnumSymbol*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kEnum;

  explicit EnumType(EnumSymbol* symbol) : Type(Kind), tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> EnumSymbol* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto underlyingType() const -> const Type*;
};

class ScopedEnumType final : public Type, public std::tuple<ScopedEnumSymbol*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kScopedEnum;

  explicit ScopedEnumType(ScopedEnumSymbol* symbol)
      : Type(Kind), tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> ScopedEnumSymbol* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto underlyingType() const -> const Type*;
};

class MemberObjectPointerType final
    : public Type,
      public std::tuple<const ClassType*, const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kMemberObjectPointer;

  MemberObjectPointerType(const ClassType* classType, const Type* elementType)
      : Type(Kind), tuple(classType, elementType) {}

  [[nodiscard]] auto classType() const -> const ClassType* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<1>(*this);
  }
};

class MemberFunctionPointerType final
    : public Type,
      public std::tuple<const ClassType*, const FunctionType*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kMemberFunctionPointer;

  MemberFunctionPointerType(const ClassType* classType,
                            const FunctionType* functionType)
      : Type(Kind), tuple(classType, functionType) {}

  [[nodiscard]] auto classType() const -> const ClassType* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto functionType() const -> const FunctionType* {
    return std::get<1>(*this);
  }
};

class NamespaceType final : public Type, public std::tuple<NamespaceSymbol*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kNamespace;

  explicit NamespaceType(NamespaceSymbol* symbol) : Type(Kind), tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> NamespaceSymbol* {
    return std::get<0>(*this);
  }
};

class TypeParameterType final : public Type, public std::tuple<int, int, bool> {
 public:
  static constexpr TypeKind Kind = TypeKind::kTypeParameter;

  explicit TypeParameterType(int index, int depth, bool isPack)
      : Type(Kind), tuple(index, depth, isPack) {}

  [[nodiscard]] auto index() const -> int { return std::get<0>(*this); }
  [[nodiscard]] auto depth() const -> int { return std::get<1>(*this); }
  [[nodiscard]] auto isParameterPack() const -> bool {
    return std::get<2>(*this);
  }
};

class TemplateTypeParameterType final
    : public Type,
      public std::tuple<int, int, bool, std::vector<const Type*>> {
 public:
  static constexpr TypeKind Kind = TypeKind::kTypeParameter;

  explicit TemplateTypeParameterType(
      int index, int depth, bool isPack,
      std::vector<const Type*> templateParameters)
      : Type(Kind),
        tuple(index, depth, isPack, std::move(templateParameters)) {}

  [[nodiscard]] auto index() const -> int { return std::get<0>(*this); }
  [[nodiscard]] auto depth() const -> int { return std::get<1>(*this); }
  [[nodiscard]] auto isParameterPack() const -> bool {
    return std::get<2>(*this);
  }
  [[nodiscard]] auto templateParameters() const
      -> const std::vector<const Type*>& {
    return std::get<3>(*this);
  }
};

class UnresolvedNameType final
    : public Type,
      public std::tuple<TranslationUnit*, NestedNameSpecifierAST*,
                        UnqualifiedIdAST*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnresolvedName;

  UnresolvedNameType(TranslationUnit* unit,
                     NestedNameSpecifierAST* nestedNameSpecifier,
                     UnqualifiedIdAST* unqualifiedId)
      : Type(Kind), tuple(unit, nestedNameSpecifier, unqualifiedId) {}

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto nestedNameSpecifier() const -> NestedNameSpecifierAST* {
    return std::get<1>(*this);
  }

  [[nodiscard]] auto unqualifiedId() const -> UnqualifiedIdAST* {
    return std::get<2>(*this);
  }
};

class UnresolvedBoundedArrayType final
    : public Type,
      public std::tuple<TranslationUnit*, const Type*, ExpressionAST*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnresolvedBoundedArray;

  UnresolvedBoundedArrayType(TranslationUnit* unit, const Type* elementType,
                             ExpressionAST* size)
      : Type(Kind), tuple(unit, elementType, size) {}

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto elementType() const -> const Type* {
    return std::get<1>(*this);
  }

  [[nodiscard]] auto size() const -> ExpressionAST* {
    return std::get<2>(*this);
  }
};

class UnresolvedUnderlyingType final
    : public Type,
      public std::tuple<TranslationUnit*, TypeIdAST*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnresolvedUnderlying;

  UnresolvedUnderlyingType(TranslationUnit* unit, TypeIdAST* typeId)
      : Type(Kind), tuple(unit, typeId) {}

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto typeId() const -> TypeIdAST* { return std::get<1>(*this); }
};

template <typename Visitor>
auto visit(Visitor&& visitor, const Type* type) {
#define PROCESS_TYPE(K) \
  case TypeKind::k##K:  \
    return std::forward<Visitor>(visitor)(static_cast<const K##Type*>(type));

  switch (type->kind()) {
    CXX_FOR_EACH_TYPE_KIND(PROCESS_TYPE)
    default:
      cxx_runtime_error("invalid type kind");
  }  // switch

#undef PROCESS_TYPE
}

template <typename T>
[[nodiscard]] auto type_cast(const Type* type) -> const T* {
  return type && type->kind() == T::Kind ? static_cast<const T*>(type)
                                         : nullptr;
}

}  // namespace cxx
