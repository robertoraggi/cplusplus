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

  auto kind() const -> TypeKind { return kind_; }

 private:
  TypeKind kind_;
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

  auto elementType() const -> const Type* { return std::get<0>(*this); }
  auto cvQualifiers() const -> CvQualifiers { return std::get<1>(*this); }
  auto isConst() const -> bool {
    return cvQualifiers() == CvQualifiers::kConst ||
           cvQualifiers() == CvQualifiers::kConstVolatile;
  }
  auto isVolatile() const -> bool {
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

  auto elementType() const -> const Type* { return std::get<0>(*this); }
  auto size() const -> std::size_t { return std::get<1>(*this); }
};

class UnboundedArrayType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnboundedArray;

  explicit UnboundedArrayType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  auto elementType() const -> const Type* { return std::get<0>(*this); }
};

class PointerType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kPointer;

  explicit PointerType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  auto elementType() const -> const Type* { return std::get<0>(*this); }
};

class LvalueReferenceType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kLvalueReference;

  explicit LvalueReferenceType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  auto elementType() const -> const Type* { return std::get<0>(*this); }
};

class RvalueReferenceType final : public Type, public std::tuple<const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kRvalueReference;

  explicit RvalueReferenceType(const Type* elementType)
      : Type(Kind), tuple(elementType) {}

  auto elementType() const -> const Type* { return std::get<0>(*this); }
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

  auto returnType() const -> const Type* { return std::get<0>(*this); }
  auto parameterTypes() const -> const std::vector<const Type*>& {
    return std::get<1>(*this);
  }
  auto isVariadic() const -> bool { return std::get<2>(*this); }
  auto cvQualifiers() const -> CvQualifiers { return std::get<3>(*this); }
  auto refQualifier() const -> RefQualifier { return std::get<4>(*this); }
  auto isNoexcept() const -> bool { return std::get<5>(*this); }
};

class ClassType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kClass;

  ClassType() : Type(Kind) {}

  auto symbol() const -> ClassSymbol* { return symbol_; }
  void setSymbol(ClassSymbol* symbol) const { symbol_ = symbol; }

 private:
  mutable ClassSymbol* symbol_ = nullptr;
};

class UnionType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnion;

  UnionType() : Type(Kind) {}

  auto symbol() const -> UnionSymbol* { return symbol_; }
  void setSymbol(UnionSymbol* symbol) const { symbol_ = symbol; }

 private:
  mutable UnionSymbol* symbol_ = nullptr;
};

class EnumType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kEnum;

  EnumType() : Type(Kind) {}

  auto symbol() const -> EnumSymbol* { return symbol_; }
  void setSymbol(EnumSymbol* symbol) const { symbol_ = symbol; }

 private:
  mutable EnumSymbol* symbol_ = nullptr;
};

class ScopedEnumType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kScopedEnum;

  ScopedEnumType() : Type(Kind) {}

  auto symbol() const -> ScopedEnumSymbol* { return symbol_; }
  void setSymbol(ScopedEnumSymbol* symbol) const { symbol_ = symbol; }

 private:
  mutable ScopedEnumSymbol* symbol_ = nullptr;
};

class MemberObjectPointerType final
    : public Type,
      public std::tuple<const ClassType*, const Type*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kMemberObjectPointer;

  MemberObjectPointerType(const ClassType* classType, const Type* elementType)
      : Type(Kind), tuple(classType, elementType) {}

  auto classType() const -> const ClassType* { return std::get<0>(*this); }
  auto elementType() const -> const Type* { return std::get<1>(*this); }
};

class MemberFunctionPointerType final
    : public Type,
      public std::tuple<const ClassType*, const FunctionType*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kMemberFunctionPointer;

  MemberFunctionPointerType(const ClassType* classType,
                            const FunctionType* functionType)
      : Type(Kind), tuple(classType, functionType) {}

  auto classType() const -> const ClassType* { return std::get<0>(*this); }
  auto functionType() const -> const FunctionType* {
    return std::get<1>(*this);
  }
};

class ClassDescriptionType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kClassDescription;

  ClassDescriptionType() : Type(Kind) {}
};

class NamespaceType final : public Type {
 public:
  static constexpr TypeKind Kind = TypeKind::kNamespace;

  NamespaceType() : Type(Kind) {}

  auto symbol() const -> NamespaceSymbol* { return symbol_; }
  void setSymbol(NamespaceSymbol* symbol) const { symbol_ = symbol; }

 private:
  mutable NamespaceSymbol* symbol_ = nullptr;
};

class UnresolvedNameType final
    : public Type,
      public std::tuple<TranslationUnit*, NamedTypeSpecifierAST*> {
 public:
  static constexpr TypeKind Kind = TypeKind::kUnresolvedName;

  UnresolvedNameType(TranslationUnit* unit,
                     NamedTypeSpecifierAST* namedTypeSpecifier)
      : Type(Kind), tuple(unit, namedTypeSpecifier) {}

  auto translationUnit() const -> TranslationUnit* {
    return std::get<0>(*this);
  }

  auto specifier() const -> NamedTypeSpecifierAST* {
    return std::get<1>(*this);
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

  auto translationUnit() const -> TranslationUnit* {
    return std::get<0>(*this);
  }

  auto elementType() const -> const Type* { return std::get<1>(*this); }

  auto size() const -> ExpressionAST* { return std::get<2>(*this); }
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
