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

#pragma once

#include <cxx/types_fwd.h>

#include <span>

namespace cxx {

class ClassSymbol;
class Control;
class TranslationUnit;
class Type;

class TypeTraits {
  TranslationUnit* unit_;

 public:
  explicit TypeTraits(TranslationUnit* unit);

  [[nodiscard]] auto unit() const -> TranslationUnit* { return unit_; }
  [[nodiscard]] auto control() const -> Control*;

  auto requireCompleteClass(ClassSymbol* classSymbol) -> bool;

  // primary type categories
  [[nodiscard]] auto is_void(const Type* type) const -> bool;
  [[nodiscard]] auto is_null_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto is_integral(const Type* type) const -> bool;
  [[nodiscard]] auto is_floating_point(const Type* type) const -> bool;
  [[nodiscard]] auto is_array(const Type* type) const -> bool;
  [[nodiscard]] auto is_enum(const Type* type) const -> bool;
  [[nodiscard]] auto is_union(const Type* type) const -> bool;
  [[nodiscard]] auto is_class(const Type* type) const -> bool;
  [[nodiscard]] auto is_function(const Type* type) const -> bool;
  [[nodiscard]] auto is_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto is_lvalue_reference(const Type* type) const -> bool;
  [[nodiscard]] auto is_rvalue_reference(const Type* type) const -> bool;
  [[nodiscard]] auto is_member_object_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto is_member_function_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto is_complete(const Type* type) const -> bool;

  // composite type categories
  [[nodiscard]] auto is_integer(const Type* type) const -> bool;
  [[nodiscard]] auto is_integral_or_unscoped_enum(const Type* type) const
      -> bool;
  [[nodiscard]] auto is_fundamental(const Type* type) const -> bool;
  [[nodiscard]] auto is_arithmetic(const Type* type) const -> bool;
  [[nodiscard]] auto is_scalar(const Type* type) const -> bool;
  [[nodiscard]] auto is_object(const Type* type) const -> bool;
  [[nodiscard]] auto is_compound(const Type* type) const -> bool;
  [[nodiscard]] auto is_reference(const Type* type) const -> bool;
  [[nodiscard]] auto is_member_pointer(const Type* type) const -> bool;

  // type properties
  [[nodiscard]] auto is_const(const Type* type) const -> bool;
  [[nodiscard]] auto is_volatile(const Type* type) const -> bool;
  [[nodiscard]] auto is_signed(const Type* type) const -> bool;
  [[nodiscard]] auto is_unsigned(const Type* type) const -> bool;
  [[nodiscard]] auto is_bounded_array(const Type* type) const -> bool;
  [[nodiscard]] auto is_unbounded_array(const Type* type) const -> bool;
  [[nodiscard]] auto is_scoped_enum(const Type* type) const -> bool;

  // references
  [[nodiscard]] auto remove_reference(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_lvalue_reference(const Type* type) const
      -> const Type*;
  [[nodiscard]] auto add_rvalue_reference(const Type* type) const
      -> const Type*;

  // arrays
  [[nodiscard]] auto remove_extent(const Type* type) const -> const Type*;
  [[nodiscard]] auto get_element_type(const Type* type) const -> const Type*;

  // cv qualifiers
  [[nodiscard]] auto remove_cv(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_cvref(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_const_ref(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_const(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_volatile(const Type* type) const -> const Type*;

  // pointers
  [[nodiscard]] auto remove_pointer(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_pointer(const Type* type) const -> const Type*;

  // type relationships
  [[nodiscard]] auto is_same(const Type* a, const Type* b) const -> bool;
  [[nodiscard]] auto decay(const Type* type) const -> const Type*;

  // convenience
  [[nodiscard]] auto is_class_or_union(const Type* type) const -> bool;
  [[nodiscard]] auto is_arithmetic_or_unscoped_enum(const Type* type) const
      -> bool;

  [[nodiscard]] auto remove_all_extents(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_const(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_volatile(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_cv(const Type* type, CvQualifiers cv) const
      -> const Type*;
  [[nodiscard]] auto get_cv_qualifiers(const Type* type) const -> CvQualifiers;
  [[nodiscard]] auto remove_noexcept(const Type* type) const -> const Type*;
  [[nodiscard]] auto is_base_of(const Type* base, const Type* derived) const
      -> bool;
  [[nodiscard]] auto is_convertible(const Type* from, const Type* to) const
      -> bool;

  auto is_pod(const Type* type) -> bool;
  auto is_trivial(const Type* type) -> bool;
  auto is_standard_layout(const Type* type) -> bool;
  auto is_literal_type(const Type* type) -> bool;
  auto is_aggregate(const Type* type) -> bool;
  auto is_empty(const Type* type) -> bool;
  auto is_polymorphic(const Type* type) -> bool;
  auto is_final(const Type* type) -> bool;
  auto is_constructible(const Type* type, std::span<const Type* const> argTypes)
      -> bool;
  auto is_nothrow_constructible(const Type* type,
                                std::span<const Type* const> argTypes) -> bool;
  auto is_trivially_constructible(const Type* type) -> bool;
  auto is_assignable(const Type* to, const Type* from) -> bool;
  auto is_trivially_assignable(const Type* from, const Type* to) -> bool;
  auto is_trivially_copyable(const Type* type) -> bool;
  auto is_abstract(const Type* type) -> bool;
  auto is_destructible(const Type* type) -> bool;
  auto is_trivially_destructible(const Type* type) -> bool;
  auto has_virtual_destructor(const Type* type) -> bool;
};

}  // namespace cxx
