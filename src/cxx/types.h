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

#include <cxx/types_fwd.h>

#include <tuple>
#include <vector>

namespace cxx {

class Type {
 public:
  virtual ~Type();

  virtual void accept(TypeVisitor* visitor) const = 0;
};

class FullySpecifiedType {
 public:
  explicit FullySpecifiedType(
      const Type* type = nullptr,
      Qualifiers qualifiers = Qualifiers::kNone) noexcept
      : type_(type), qualifiers_(qualifiers) {}

  explicit operator bool() const noexcept { return type_ != nullptr; }

  const Type* operator->() const noexcept { return type_; }

  const Type* type() const { return type_; }
  void setType(const Type* type) { type_ = type; }

  Qualifiers qualifiers() const { return qualifiers_; }
  void setQualifiers(Qualifiers qualifiers) { qualifiers_ = qualifiers; }

  bool isConst() const {
    return (qualifiers_ & Qualifiers::kConst) != Qualifiers::kNone;
  }

  bool isVolatile() const {
    return (qualifiers_ & Qualifiers::kVolatile) != Qualifiers::kNone;
  }

  bool isRestrict() const {
    return (qualifiers_ & Qualifiers::kRestrict) != Qualifiers::kNone;
  }

 private:
  const Type* type_;
  Qualifiers qualifiers_;
};
class UnresolvedType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class VoidType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class NullptrType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class BooleanType final : public Type {
 public:
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

class EnumType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class ScopedEnumType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class PointerType final : public Type,
                          public std::tuple<FullySpecifiedType, Qualifiers> {
 public:
  PointerType(const FullySpecifiedType& elementType,
              Qualifiers qualifiers) noexcept
      : tuple(elementType, qualifiers) {}

  const FullySpecifiedType& elementType() const { return get<0>(*this); }

  Qualifiers qualifiers() const { return get<1>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class PointerToMemberType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class ReferenceType final : public Type, public std::tuple<FullySpecifiedType> {
 public:
  explicit ReferenceType(const FullySpecifiedType& elementType) noexcept
      : tuple(elementType) {}

  const FullySpecifiedType& elementType() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class RValueReferenceType final : public Type,
                                  public std::tuple<FullySpecifiedType> {
 public:
  explicit RValueReferenceType(const FullySpecifiedType& elementType) noexcept
      : tuple(elementType) {}

  const FullySpecifiedType& elementType() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class ArrayType final : public Type,
                        public std::tuple<FullySpecifiedType, std::size_t> {
 public:
  ArrayType(const FullySpecifiedType& elementType,
            std::size_t dimension) noexcept
      : tuple(elementType, dimension) {}

  const FullySpecifiedType& elementType() const { return get<0>(*this); }

  std::size_t dimension() const { return get<1>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class UnboundArrayType final : public Type,
                               public std::tuple<FullySpecifiedType> {
 public:
  explicit UnboundArrayType(const FullySpecifiedType& elementType) noexcept
      : tuple(elementType) {}

  const FullySpecifiedType& elementType() const { return get<0>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class FunctionType final
    : public Type,
      public std::tuple<FullySpecifiedType, std::vector<FullySpecifiedType>,
                        bool> {
 public:
  explicit FunctionType(const FullySpecifiedType& returnType,
                        std::vector<FullySpecifiedType> argumentTypes,
                        bool isVariadic) noexcept
      : tuple(returnType, std::move(argumentTypes), isVariadic) {}

  const FullySpecifiedType& returnType() const { return get<0>(*this); }

  const std::vector<FullySpecifiedType>& argumentTypes() const {
    return get<1>(*this);
  }

  bool isVariadic() const { return get<2>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class MemberFunctionType final
    : public Type,
      public std::tuple<const ClassType*, FullySpecifiedType,
                        std::vector<FullySpecifiedType>, bool> {
 public:
  explicit MemberFunctionType(const ClassType* classType,
                              const FullySpecifiedType& returnType,
                              std::vector<FullySpecifiedType> argumentTypes,
                              bool isVariadic) noexcept
      : tuple(classType, returnType, std::move(argumentTypes), isVariadic) {}

  const ClassType* classType() const { return get<0>(*this); }

  const FullySpecifiedType& returnType() const { return get<1>(*this); }

  const std::vector<FullySpecifiedType>& argumentTypes() const {
    return get<2>(*this);
  }

  bool isVariadic() const { return get<3>(*this); }

  void accept(TypeVisitor* visitor) const override;
};

class NamespaceType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class ClassType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class TemplateType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

class TemplateArgumentType final : public Type {
 public:
  void accept(TypeVisitor* visitor) const override;
};

}  // namespace cxx