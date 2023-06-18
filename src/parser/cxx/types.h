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

#include <cxx/qualified_type.h>
#include <cxx/symbols_fwd.h>

#include <tuple>
#include <vector>

namespace cxx {

class Type {
 public:
  virtual ~Type();

  virtual void accept(TypeVisitor* visitor) const = 0;

  [[nodiscard]] auto isIntegral() const -> bool;
  [[nodiscard]] auto isArithmetic() const -> bool;
  [[nodiscard]] auto isScalar() const -> bool;
  [[nodiscard]] auto isFundamental() const -> bool;
  [[nodiscard]] auto isCompound() const -> bool;
  [[nodiscard]] auto isObject() const -> bool;

  template <typename T>
  static auto cast(const QualifiedType& qualType) -> const T* {
    return dynamic_cast<const T*>(qualType.type());
  }

  template <typename T>
  static auto cast(const Type* type) -> const T* {
    return dynamic_cast<const T*>(type);
  }

  template <typename T>
  static auto is(const QualifiedType& qualType) -> bool {
    return cast<T>(qualType) != nullptr;
  }

  template <typename T>
  static auto is(const Type* type) -> bool {
    return cast<T>(type) != nullptr;
  }
};

class UndefinedType final : public Type {
  UndefinedType() = default;

 public:
  static auto get() -> const UndefinedType*;

  void accept(TypeVisitor* visitor) const override;
};

class ErrorType final : public Type {
  ErrorType() = default;

 public:
  static auto get() -> const ErrorType*;

  void accept(TypeVisitor* visitor) const override;
};

class AutoType final : public Type {
 public:
  static auto get() -> const AutoType*;

  void accept(TypeVisitor* visitor) const override;
};

class DecltypeAutoType final : public Type {
 public:
  static auto get() -> const DecltypeAutoType*;

  void accept(TypeVisitor* visitor) const override;
};

class VoidType final : public Type {
 public:
  static auto get() -> const VoidType*;

  void accept(TypeVisitor* visitor) const override;
};

class NullptrType final : public Type {
 public:
  static auto get() -> const NullptrType*;

  void accept(TypeVisitor* visitor) const override;
};

class BooleanType final : public Type {
 public:
  static auto get() -> const BooleanType*;

  void accept(TypeVisitor* visitor) const override;
};

class CharacterType final : public Type, public std::tuple<CharacterKind> {
 public:
  explicit CharacterType(CharacterKind kind) noexcept : tuple(kind) {}

  [[nodiscard]] auto kind() const -> CharacterKind {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class IntegerType final : public Type, public std::tuple<IntegerKind, bool> {
 public:
  IntegerType(IntegerKind kind, bool isUnsigned) noexcept
      : tuple(kind, isUnsigned) {}

  [[nodiscard]] auto kind() const -> IntegerKind { return std::get<0>(*this); }

  [[nodiscard]] auto isUnsigned() const -> bool { return std::get<1>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class FloatingPointType final : public Type,
                                public std::tuple<FloatingPointKind> {
 public:
  explicit FloatingPointType(FloatingPointKind kind) noexcept : tuple(kind) {}

  [[nodiscard]] auto kind() const -> FloatingPointKind {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class EnumType final : public Type, public std::tuple<EnumSymbol*> {
 public:
  explicit EnumType(EnumSymbol* symbol) noexcept : tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> EnumSymbol* {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class ScopedEnumType final : public Type, public std::tuple<ScopedEnumSymbol*> {
 public:
  explicit ScopedEnumType(ScopedEnumSymbol* symbol) noexcept : tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> ScopedEnumSymbol* {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class PointerType final : public Type,
                          public std::tuple<QualifiedType, Qualifiers> {
 public:
  PointerType(const QualifiedType& elementType, Qualifiers qualifiers) noexcept
      : tuple(elementType, qualifiers) {}

  [[nodiscard]] auto elementType() const -> const QualifiedType& {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto qualifiers() const -> Qualifiers {
    return std::get<1>(*this);
  }

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

  [[nodiscard]] auto classType() const -> const ClassType* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto elementType() const -> const QualifiedType& {
    return std::get<1>(*this);
  }

  [[nodiscard]] auto qualifiers() const -> Qualifiers {
    return std::get<2>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class ReferenceType final : public Type, public std::tuple<QualifiedType> {
 public:
  explicit ReferenceType(const QualifiedType& elementType) noexcept
      : tuple(elementType) {}

  [[nodiscard]] auto elementType() const -> const QualifiedType& {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class RValueReferenceType final : public Type,
                                  public std::tuple<QualifiedType> {
 public:
  explicit RValueReferenceType(const QualifiedType& elementType) noexcept
      : tuple(elementType) {}

  [[nodiscard]] auto elementType() const -> const QualifiedType& {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class ArrayType final : public Type,
                        public std::tuple<QualifiedType, std::size_t> {
 public:
  ArrayType(const QualifiedType& elementType, std::size_t dimension) noexcept
      : tuple(elementType, dimension) {}

  [[nodiscard]] auto elementType() const -> const QualifiedType& {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto dimension() const -> std::size_t {
    return std::get<1>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class UnboundArrayType final : public Type, public std::tuple<QualifiedType> {
 public:
  explicit UnboundArrayType(const QualifiedType& elementType) noexcept
      : tuple(elementType) {}

  [[nodiscard]] auto elementType() const -> const QualifiedType& {
    return std::get<0>(*this);
  }

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

  [[nodiscard]] auto returnType() const -> const QualifiedType& {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto argumentTypes() const
      -> const std::vector<QualifiedType>& {
    return std::get<1>(*this);
  }

  [[nodiscard]] auto isVariadic() const -> bool { return std::get<2>(*this); }

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

  [[nodiscard]] auto classType() const -> const ClassType* {
    return std::get<0>(*this);
  }

  [[nodiscard]] auto returnType() const -> const QualifiedType& {
    return std::get<1>(*this);
  }

  [[nodiscard]] auto argumentTypes() const
      -> const std::vector<QualifiedType>& {
    return std::get<2>(*this);
  }

  [[nodiscard]] auto isVariadic() const -> bool { return std::get<3>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class NamespaceType final : public Type, public std::tuple<NamespaceSymbol*> {
 public:
  explicit NamespaceType(NamespaceSymbol* symbol) noexcept : tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> NamespaceSymbol* {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class ClassType final : public Type, public std::tuple<ClassSymbol*> {
 public:
  explicit ClassType(ClassSymbol* symbol) noexcept : tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> ClassSymbol* {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class TemplateType final : public Type,
                           public std::tuple<TemplateParameterList*> {
 public:
  explicit TemplateType(TemplateParameterList* symbol) noexcept
      : tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> TemplateParameterList* {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class TemplateArgumentType final
    : public Type,
      public std::tuple<TemplateTypeParameterSymbol*> {
 public:
  explicit TemplateArgumentType(TemplateTypeParameterSymbol* symbol) noexcept
      : tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> TemplateTypeParameterSymbol* {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

class ConceptType final : public Type, public std::tuple<ConceptSymbol*> {
 public:
  explicit ConceptType(ConceptSymbol* symbol) noexcept : tuple(symbol) {}

  [[nodiscard]] auto symbol() const -> ConceptSymbol* {
    return std::get<0>(*this);
  }

  void accept(TypeVisitor* visitor) const override;
};

}  // namespace cxx