// Copyright (c) 2022 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/qualified_type.h>
#include <cxx/symbols_fwd.h>

#include <tuple>
#include <vector>

namespace cxx {

class Type {
 public:
  virtual ~Type();

  virtual void accept(TypeVisitor* visitor) const = 0;

  bool isIntegral() const;
  bool isArithmetic() const;
  bool isScalar() const;
  bool isFundamental() const;
  bool isCompound() const;
  bool isObject() const;

  template <typename T>
  static const T* cast(const QualifiedType& qualType) {
    return dynamic_cast<const T*>(qualType.type());
  }

  template <typename T>
  static const T* cast(const Type* type) {
    return dynamic_cast<const T*>(type);
  }

  template <typename T>
  static bool is(const QualifiedType& qualType) {
    return cast<T>(qualType) != nullptr;
  }

  template <typename T>
  static bool is(const Type* type) {
    return cast<T>(type) != nullptr;
  }
};

class UndefinedType final : public Type {
  UndefinedType() = default;

 public:
  static const UndefinedType* get();

  void accept(TypeVisitor* visitor) const override;
};

class ErrorType final : public Type {
  ErrorType() = default;

 public:
  static const ErrorType* get();

  void accept(TypeVisitor* visitor) const override;
};

class AutoType final : public Type {
 public:
  static const AutoType* get();

  void accept(TypeVisitor* visitor) const override;
};

class DecltypeAutoType final : public Type {
 public:
  static const DecltypeAutoType* get();

  void accept(TypeVisitor* visitor) const override;
};

class VoidType final : public Type {
 public:
  static const VoidType* get();

  void accept(TypeVisitor* visitor) const override;
};

class NullptrType final : public Type {
 public:
  static const NullptrType* get();

  void accept(TypeVisitor* visitor) const override;
};

class BooleanType final : public Type {
 public:
  static const BooleanType* get();

  void accept(TypeVisitor* visitor) const override;
};

class CharacterType final : public Type, public std::tuple<CharacterKind> {
 public:
  explicit CharacterType(CharacterKind kind) noexcept : tuple(kind) {}

  CharacterKind kind() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class IntegerType final : public Type, public std::tuple<IntegerKind, bool> {
 public:
  IntegerType(IntegerKind kind, bool isUnsigned) noexcept
      : tuple(kind, isUnsigned) {}

  IntegerKind kind() const { return get<0>(*this); }

  bool isUnsigned() const { return get<1>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class FloatingPointType final : public Type,
                                public std::tuple<FloatingPointKind> {
 public:
  explicit FloatingPointType(FloatingPointKind kind) noexcept : tuple(kind) {}

  FloatingPointKind kind() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class EnumType final : public Type, public std::tuple<EnumSymbol*> {
 public:
  explicit EnumType(EnumSymbol* symbol) noexcept : tuple(symbol) {}

  EnumSymbol* symbol() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class ScopedEnumType final : public Type, public std::tuple<ScopedEnumSymbol*> {
 public:
  explicit ScopedEnumType(ScopedEnumSymbol* symbol) noexcept : tuple(symbol) {}

  ScopedEnumSymbol* symbol() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class PointerType final : public Type,
                          public std::tuple<QualifiedType, Qualifiers> {
 public:
  PointerType(const QualifiedType& elementType, Qualifiers qualifiers) noexcept
      : tuple(elementType, qualifiers) {}

  const QualifiedType& elementType() const { return get<0>(*this); }

  Qualifiers qualifiers() const { return get<1>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class PointerToMemberType final
    : public Type,
      public std::tuple<const ClassType*, QualifiedType, Qualifiers> {
 public:
  PointerToMemberType(const ClassType* classType,
                      const QualifiedType& elementType,
                      Qualifiers qualifiers) noexcept
      : tuple(classType, elementType, qualifiers) {}

  const ClassType* classType() const { return get<0>(*this); }

  const QualifiedType& elementType() const { return get<1>(*this); }

  Qualifiers qualifiers() const { return get<2>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class ReferenceType final : public Type, public std::tuple<QualifiedType> {
 public:
  explicit ReferenceType(const QualifiedType& elementType) noexcept
      : tuple(elementType) {}

  const QualifiedType& elementType() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class RValueReferenceType final : public Type,
                                  public std::tuple<QualifiedType> {
 public:
  explicit RValueReferenceType(const QualifiedType& elementType) noexcept
      : tuple(elementType) {}

  const QualifiedType& elementType() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class ArrayType final : public Type,
                        public std::tuple<QualifiedType, std::size_t> {
 public:
  ArrayType(const QualifiedType& elementType, std::size_t dimension) noexcept
      : tuple(elementType, dimension) {}

  const QualifiedType& elementType() const { return get<0>(*this); }

  std::size_t dimension() const { return get<1>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class UnboundArrayType final : public Type, public std::tuple<QualifiedType> {
 public:
  explicit UnboundArrayType(const QualifiedType& elementType) noexcept
      : tuple(elementType) {}

  const QualifiedType& elementType() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class FunctionType final
    : public Type,
      public std::tuple<QualifiedType, std::vector<QualifiedType>, bool> {
 public:
  explicit FunctionType(const QualifiedType& returnType,
                        std::vector<QualifiedType> argumentTypes,
                        bool isVariadic) noexcept
      : tuple(returnType, std::move(argumentTypes), isVariadic) {}

  const QualifiedType& returnType() const { return get<0>(*this); }

  const std::vector<QualifiedType>& argumentTypes() const {
    return get<1>(*this);
  }

  bool isVariadic() const { return get<2>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class MemberFunctionType final
    : public Type,
      public std::tuple<const ClassType*, QualifiedType,
                        std::vector<QualifiedType>, bool> {
 public:
  explicit MemberFunctionType(const ClassType* classType,
                              const QualifiedType& returnType,
                              std::vector<QualifiedType> argumentTypes,
                              bool isVariadic) noexcept
      : tuple(classType, returnType, std::move(argumentTypes), isVariadic) {}

  const ClassType* classType() const { return get<0>(*this); }

  const QualifiedType& returnType() const { return get<1>(*this); }

  const std::vector<QualifiedType>& argumentTypes() const {
    return get<2>(*this);
  }

  bool isVariadic() const { return get<3>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class NamespaceType final : public Type, public std::tuple<NamespaceSymbol*> {
 public:
  explicit NamespaceType(NamespaceSymbol* symbol) noexcept : tuple(symbol) {}

  NamespaceSymbol* symbol() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class ClassType final : public Type, public std::tuple<ClassSymbol*> {
 public:
  explicit ClassType(ClassSymbol* symbol) noexcept : tuple(symbol) {}

  ClassSymbol* symbol() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class TemplateType final : public Type, public std::tuple<TemplateSymbol*> {
 public:
  explicit TemplateType(TemplateSymbol* symbol) noexcept : tuple(symbol) {}

  TemplateSymbol* symbol() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class TemplateArgumentType final
    : public Type,
      public std::tuple<TemplateTypeParameterSymbol*> {
 public:
  explicit TemplateArgumentType(TemplateTypeParameterSymbol* symbol) noexcept
      : tuple(symbol) {}

  TemplateTypeParameterSymbol* symbol() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class ConceptType final : public Type, public std::tuple<ConceptSymbol*> {
 public:
  explicit ConceptType(ConceptSymbol* symbol) noexcept : tuple(symbol) {}

  ConceptSymbol* symbol() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

}  // namespace cxx