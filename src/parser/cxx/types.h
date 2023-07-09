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

#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>
#include <cxx/types_fwd.h>

#include <cstddef>

namespace cxx {

class Control;

class Parameter : public std::tuple<const Name*, const Type*> {
 public:
  using tuple::tuple;

  auto name() const -> const Name* { return std::get<0>(*this); }
  auto type() const -> const Type* { return std::get<1>(*this); }
};

class Type {
 public:
  explicit Type(TypeKind kind) : kind_(kind) {}
  virtual ~Type();

  virtual void accept(TypeVisitor* visitor) const = 0;

  auto kind() const -> TypeKind { return kind_; }

  auto is(TypeKind kind) const -> bool { return kind_ == kind; }
  auto isNot(TypeKind kind) const -> bool { return kind_ != kind; }

  auto equalTo(const Type* other) const -> bool;

 private:
  TypeKind kind_;
};

template <TypeKind K, typename Base = Type>
class TypeMaker : public Base {
 public:
  static constexpr TypeKind Kind = K;

  TypeMaker() : Base(K) {}
};

class ReferenceType : public Type {
 public:
  using Type::Type;

  const Type* elementType = nullptr;
};

class InvalidType final : public TypeMaker<TypeKind::kInvalid> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const InvalidType* other) const -> bool;
};

class DependentType final : public TypeMaker<TypeKind::kDependent> {
 public:
  DependentSymbol* symbol = nullptr;

  DependentType(Control* control, DependentSymbol* symbol);

  void accept(TypeVisitor* visitor) const override;

  auto equalTo0(const DependentType* other) const -> bool;
};

class NullptrType final : public TypeMaker<TypeKind::kNullptr> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const NullptrType* other) const -> bool;
};

class AutoType final : public TypeMaker<TypeKind::kAuto> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const AutoType* other) const -> bool;
};

class VoidType final : public TypeMaker<TypeKind::kVoid> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const VoidType* other) const -> bool;
};

class BoolType final : public TypeMaker<TypeKind::kBool> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const BoolType* other) const -> bool;
};

class CharType final : public TypeMaker<TypeKind::kChar> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const CharType* other) const -> bool;
};

class SignedCharType final : public TypeMaker<TypeKind::kSignedChar> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const SignedCharType* other) const -> bool;
};

class UnsignedCharType final : public TypeMaker<TypeKind::kUnsignedChar> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const UnsignedCharType* other) const -> bool;
};

class ShortType final : public TypeMaker<TypeKind::kShort> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const ShortType* other) const -> bool;
};

class UnsignedShortType final : public TypeMaker<TypeKind::kUnsignedShort> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const UnsignedShortType* other) const -> bool;
};

class IntType final : public TypeMaker<TypeKind::kInt> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const IntType* other) const -> bool;
};

class UnsignedIntType final : public TypeMaker<TypeKind::kUnsignedInt> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const UnsignedIntType* other) const -> bool;
};

class LongType final : public TypeMaker<TypeKind::kLong> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const LongType* other) const -> bool;
};

class UnsignedLongType final : public TypeMaker<TypeKind::kUnsignedLong> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const UnsignedLongType* other) const -> bool;
};

class FloatType final : public TypeMaker<TypeKind::kFloat> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const FloatType* other) const -> bool;
};

class DoubleType final : public TypeMaker<TypeKind::kDouble> {
 public:
  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const DoubleType* other) const -> bool;
};

class QualType final : public TypeMaker<TypeKind::kQual> {
 public:
  const Type* elementType = nullptr;
  bool isConst = false;
  bool isVolatile = false;

  explicit QualType(Control* control, const Type* elementType, bool isConst,
                    bool isVolatile);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const QualType* other) const -> bool;
};

class PointerType final : public TypeMaker<TypeKind::kPointer> {
 public:
  const Type* elementType = nullptr;

  PointerType(Control* control, const Type* elementType);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const PointerType* other) const -> bool;
};

class LValueReferenceType final
    : public TypeMaker<TypeKind::kLValueReference, ReferenceType> {
 public:
  LValueReferenceType(Control* control, const Type* elementType);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const LValueReferenceType* other) const -> bool;
};

class RValueReferenceType final
    : public TypeMaker<TypeKind::kRValueReference, ReferenceType> {
 public:
  RValueReferenceType(Control* control, const Type* elementType);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const RValueReferenceType* other) const -> bool;
};

class ArrayType final : public TypeMaker<TypeKind::kArray> {
 public:
  const Type* elementType = nullptr;
  int dim = 0;

  ArrayType(Control* control, const Type* elementType, int dim);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const ArrayType* other) const -> bool;
};

class FunctionType final : public TypeMaker<TypeKind::kFunction> {
 public:
  FunctionSymbol* symbol = nullptr;
  const Type* classType = nullptr;
  const Type* returnType = nullptr;
  std::vector<Parameter> parameters;
  bool isVariadic = false;

  FunctionType(Control* control, const Type* classType, const Type* returnType,
               std::vector<Parameter> parameters, bool isVariadic);

  auto makeTemplate(Control* control, FunctionSymbol* symbol) const
      -> const FunctionType*;

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const FunctionType* other) const -> bool;
};

class ConceptType final : public TypeMaker<TypeKind::kConcept> {
 public:
  Symbol* symbol = nullptr;

  ConceptType(Control* control, Symbol* symbol);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const ConceptType* other) const -> bool;
};

class ClassType final : public TypeMaker<TypeKind::kClass> {
 public:
  ClassSymbol* symbol = nullptr;

  ClassType(Control* control, ClassSymbol* symbol);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const ClassType* other) const -> bool;

  auto isDerivedFrom(const ClassType* classType) const -> bool;
  auto isBaseOf(const ClassType* classType) const -> bool;
};

class NamespaceType final : public TypeMaker<TypeKind::kNamespace> {
 public:
  NamespaceSymbol* symbol = nullptr;

  NamespaceType(Control* control, NamespaceSymbol* symbol);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const NamespaceType* other) const -> bool;
};

class MemberPointerType final : public TypeMaker<TypeKind::kMemberPointer> {
 public:
  const Type* elementType = nullptr;
  const Type* classType = nullptr;

  MemberPointerType(Control* control, const Type* class_type,
                    const Type* member_type);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const MemberPointerType* other) const -> bool;
};

class EnumType final : public TypeMaker<TypeKind::kEnum> {
 public:
  Symbol* symbol = nullptr;

  EnumType(Control* control, Symbol* symbol);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const EnumType* other) const -> bool;
};

class GenericType final : public TypeMaker<TypeKind::kGeneric> {
 public:
  Symbol* symbol = nullptr;

  GenericType(Control* control, Symbol* symbol);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const GenericType* other) const -> bool;
};

class PackType final : public TypeMaker<TypeKind::kPack> {
 public:
  Symbol* symbol = nullptr;

  PackType(Control* control, Symbol* symbol);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const PackType* other) const -> bool;
};

class ScopedEnumType final : public TypeMaker<TypeKind::kScopedEnum> {
 public:
  ScopedEnumSymbol* symbol = nullptr;
  const Type* elementType = nullptr;

  ScopedEnumType(Control* control, ScopedEnumSymbol* symbol,
                 const Type* elementType);

  void accept(TypeVisitor* visitor) const override;
  auto equalTo0(const ScopedEnumType* other) const -> bool;
};

template <typename T>
inline auto type_cast(const Type* type) -> const T* {
  if (type && type->kind() == T::Kind) return static_cast<const T*>(type);
  return nullptr;
}

template <>
inline auto type_cast<ReferenceType>(const Type* type) -> const ReferenceType* {
  if (auto result = type_cast<LValueReferenceType>(type)) return result;
  return type_cast<RValueReferenceType>(type);
}

auto is_same_type(const Type* type, const Type* other) -> bool;
auto is_same_parameters(const std::vector<Parameter>& params,
                        const std::vector<Parameter>& other) -> bool;

auto is_same_template_arguments(const std::vector<TemplateArgument>& list,
                                const std::vector<TemplateArgument>& other)
    -> bool;

auto promote_type(Control* control, const Type* type) -> const Type*;

auto type_kind(const Type* ty) -> TypeKind;
auto type_element_type(const Type* ty) -> const Type*;

auto type_extent(const Type* ty) -> int;

auto is_void_type(const Type* ty) -> bool;
auto is_bool_type(const Type* ty) -> bool;
auto is_qual_type(const Type* ty) -> bool;
auto is_pointer_type(const Type* ty) -> bool;
auto is_lvalue_reference_type(const Type* ty) -> bool;
auto is_rvalue_reference_type(const Type* ty) -> bool;
auto is_array_type(const Type* ty) -> bool;
auto is_function_type(const Type* ty) -> bool;
auto is_namespace_type(const Type* ty) -> bool;
auto is_class_type(const Type* ty) -> bool;
auto is_union_type(const Type* ty) -> bool;
auto is_nullptr_type(const Type* ty) -> bool;
auto is_integral_type(const Type* ty) -> bool;
auto is_floating_point_type(const Type* ty) -> bool;
auto is_member_object_pointer_type(const Type* ty) -> bool;
auto is_member_function_pointer_type(const Type* ty) -> bool;
auto is_enum_type(const Type* ty) -> bool;
auto is_unscoped_enum_type(const Type* type) -> bool;
auto is_scoped_enum_type(const Type* type) -> bool;

auto is_signed(const Type* type) -> bool;
auto is_unsigned(const Type* type) -> bool;

auto is_const(const Type* type) -> bool;
auto is_volatile(const Type* type) -> bool;

auto make_signed(Control* control, const Type* type) -> const Type*;
auto make_unsigned(Control* control, const Type* type) -> const Type*;

auto remove_cv(const Type* type) -> const Type*;
auto remove_ref(const Type* type) -> const Type*;
auto remove_cvref(const Type* type) -> const Type*;

auto is_integral_or_unscoped_enum_type(const Type* ty) -> bool;

auto is_member_pointer_type(const Type* type) -> bool;
auto is_reference_type(const Type* type) -> bool;
auto is_compound_type(const Type* type) -> bool;
auto is_object_type(const Type* type) -> bool;
auto is_scalar_type(const Type* type) -> bool;
auto is_arithmetic_type(const Type* type) -> bool;
auto is_arithmetic_or_unscoped_type(const Type* type) -> bool;
auto is_fundamental_type(const Type* type) -> bool;
auto is_literal_type(const Type* type) -> bool;

auto common_type(Control* control, const Type* type, const Type* other)
    -> const Type*;

auto function_type_parameter_count(const Type* ty) -> std::size_t;

auto type_to_string(const Type* type, char* out, std::size_t size)
    -> std::string;

}  // namespace cxx
