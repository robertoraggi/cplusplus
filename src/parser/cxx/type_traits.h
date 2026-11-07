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

#include <cxx/const_int.h>
#include <cxx/names_fwd.h>
#include <cxx/types_fwd.h>

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace cxx {
class ClassSymbol;
class Control;
class ExpressionAST;
class FieldSymbol;
class FunctionSymbol;
class OverloadSetType;
class ScopeSymbol;
class Symbol;
class TranslationUnit;
class Type;

struct OverloadSetOperand {
  const OverloadSetType* type = nullptr;
  bool takesAddress = false;
};

class TypeTraits {
  TranslationUnit* unit_;

 public:
  explicit TypeTraits(TranslationUnit* unit);

  [[nodiscard]] auto unit() const -> TranslationUnit* { return unit_; }
  [[nodiscard]] auto control() const -> Control*;

  auto require_complete_class(ClassSymbol* classSymbol) -> bool;
  auto require_complete_class(const Type* type) -> bool;

  [[nodiscard]] auto is_void(const Type* type) const -> bool;
  [[nodiscard]] auto is_bool(const Type* type) const -> bool;
  [[nodiscard]] auto is_null_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto is_integral(const Type* type) const -> bool;
  [[nodiscard]] auto is_floating_point(const Type* type) const -> bool;
  [[nodiscard]] auto is_array(const Type* type) const -> bool;
  [[nodiscard]] auto is_enum(const Type* type) const -> bool;
  [[nodiscard]] auto is_union(const Type* type) const -> bool;
  [[nodiscard]] auto is_class(const Type* type) const -> bool;
  [[nodiscard]] auto is_function(const Type* type) const -> bool;
  [[nodiscard]] auto is_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto overload_set_operand(const Type* type) const
      -> OverloadSetOperand;
  [[nodiscard]] auto is_vector(const Type* type) const -> bool;
  [[nodiscard]] auto is_device_value_type(const Type* type) const -> bool;

  /**
   * A type whose special member functions are all trivial and which has no
   * class semantics: a scalar or a vector.
   */
  [[nodiscard]] auto is_scalar_or_vector(const Type* type) const -> bool;

  /**
   * The number of bytes a vector's lanes occupy, ignoring the padding that
   * rounds its storage size up to a power of two.
   */
  [[nodiscard]] auto vector_width_in_bytes(const Type* type) const
      -> std::size_t;
  [[nodiscard]] auto is_vector_element_type(const Type* type,
                                            VectorKind kind) const -> bool;
  [[nodiscard]] auto is_lvalue_reference(const Type* type) const -> bool;
  [[nodiscard]] auto is_rvalue_reference(const Type* type) const -> bool;
  [[nodiscard]] auto is_member_object_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto is_member_function_pointer(const Type* type) const -> bool;
  [[nodiscard]] auto is_complete(const Type* type) const -> bool;
  [[nodiscard]] auto select_destructor(ClassSymbol* cls) const
      -> FunctionSymbol*;

  [[nodiscard]] auto is_integral_or_unscoped_enum(const Type* type) const
      -> bool;
  [[nodiscard]] auto is_integral_or_enum(const Type* type) const -> bool;
  [[nodiscard]] auto is_fundamental(const Type* type) const -> bool;
  [[nodiscard]] auto is_arithmetic(const Type* type) const -> bool;
  [[nodiscard]] auto arithmetic_types() const -> std::vector<const Type*>;
  [[nodiscard]] auto is_floating(const Type* type) const -> bool;
  [[nodiscard]] auto is_scalar(const Type* type) const -> bool;
  [[nodiscard]] auto is_object(const Type* type) const -> bool;
  [[nodiscard]] auto is_compound(const Type* type) const -> bool;
  [[nodiscard]] auto is_reference(const Type* type) const -> bool;
  [[nodiscard]] auto is_member_pointer(const Type* type) const -> bool;

  [[nodiscard]] auto is_const(const Type* type) const -> bool;
  [[nodiscard]] auto is_volatile(const Type* type) const -> bool;
  [[nodiscard]] auto is_atomic(const Type* type) const -> bool;
  [[nodiscard]] auto is_signed(const Type* type) const -> bool;
  [[nodiscard]] auto is_unsigned(const Type* type) const -> bool;
  [[nodiscard]] auto is_bounded_array(const Type* type) const -> bool;
  [[nodiscard]] auto is_unbounded_array(const Type* type) const -> bool;
  [[nodiscard]] auto is_scoped_enum(const Type* type) const -> bool;

  struct IntegralRepresentation {
    int bits = 0;
    bool isSigned = false;
  };

  [[nodiscard]] auto integral_representation(const Type* type) const
      -> std::optional<IntegralRepresentation>;

  [[nodiscard]] auto value_representation(const Type* type) const
      -> std::optional<IntegralRepresentation>;

  [[nodiscard]] auto integral_constant(const Type* type,
                                       ConstInt::Wide value) const
      -> std::optional<ConstInt>;

  [[nodiscard]] auto converted_integral_constant(const Type* type,
                                                 const ConstInt& value) const
      -> std::optional<ConstInt>;
  [[nodiscard]] auto converted_constant_value(const Type* type,
                                              const ConstValue& value) const
      -> std::optional<ConstValue>;

  [[nodiscard]] auto remove_reference(const Type* type) const -> const Type*;
  [[nodiscard]] auto size_of(const Type* type) -> std::optional<std::size_t>;
  [[nodiscard]] auto alignment_of(const Type* type)
      -> std::optional<std::size_t>;
  [[nodiscard]] auto add_lvalue_reference(const Type* type) const
      -> const Type*;
  [[nodiscard]] auto add_rvalue_reference(const Type* type) const
      -> const Type*;

  [[nodiscard]] auto decltype_of(ExpressionAST* expr) const -> const Type*;

  [[nodiscard]] auto constant_template_parameter_type(
      NonTypeParameterSymbol* parameter) const -> const Type*;

  [[nodiscard]] auto remove_extent(const Type* type) const -> const Type*;
  [[nodiscard]] auto get_element_type(const Type* type) const -> const Type*;

  [[nodiscard]] auto underlying_type(const Type* type) const -> const Type*;

  [[nodiscard]] auto remove_cv(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_cvref(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_const_ref(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_const(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_volatile(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_atomic(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_atomic(const Type* type) const -> const Type*;
  [[nodiscard]] auto is_complex(const Type* type) const -> bool;
  [[nodiscard]] auto complex_element_type(const Type* type) const
      -> const Type*;

  [[nodiscard]] auto remove_pointer(const Type* type) const -> const Type*;
  [[nodiscard]] auto add_pointer(const Type* type) const -> const Type*;

  [[nodiscard]] auto address_of_function(FunctionSymbol* function) const
      -> const Type*;

  [[nodiscard]] auto make_signed(const Type* type) const -> const Type*;
  [[nodiscard]] auto make_unsigned(const Type* type) const -> const Type*;

  [[nodiscard]] auto is_compatible(const Type* a, const Type* b) const -> bool;
  [[nodiscard]] auto is_same(const Type* a, const Type* b) const -> bool;
  [[nodiscard]] auto decay(const Type* type) const -> const Type*;
  [[nodiscard]] auto unwrap_reference(const Type* type) const -> const Type*;

  [[nodiscard]] auto is_class_or_union(const Type* type) const -> bool;
  [[nodiscard]] auto is_arithmetic_or_unscoped_enum(const Type* type) const
      -> bool;
  [[nodiscard]] auto is_narrow_char_type(const Type* type) const -> bool;
  [[nodiscard]] auto is_char_type(const Type* type) const -> bool;
  [[nodiscard]] auto is_narrowing_conversion(const Type* from,
                                             const Type* to) const -> bool;
  [[nodiscard]] auto converted_pack_index(ExpressionAST* index)
      -> std::optional<std::size_t>;
  [[nodiscard]] auto is_narrowing_list_element(ExpressionAST* expr,
                                               const Type* targetType) const
      -> bool;
  [[nodiscard]] auto integer_constant_fits_in_type(std::uint64_t value,
                                                   const Type* targetType) const
      -> bool;

  [[nodiscard]] auto is_std_namespace(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_in_std_namespace(Symbol* symbol) const -> bool;
  [[nodiscard]] auto standard_class_specialization(
      WellKnownName name, List<TemplateArgumentAST*>* arguments,
      SourceLocation location) -> ClassSymbol*;
  [[nodiscard]] auto standard_ranges_member(std::string_view name) const
      -> Symbol*;
  [[nodiscard]] auto member_value(ClassSymbol* classSymbol,
                                  std::string_view name = "value") const
      -> std::optional<ConstValue>;
  [[nodiscard]] auto type_order(const Type* first, const Type* second) const
      -> std::strong_ordering;
  [[nodiscard]] auto member_type(ClassSymbol* classSymbol) const -> const Type*;
  [[nodiscard]] auto is_std_type(const Type* type, WellKnownName name) const
      -> bool;
  [[nodiscard]] auto is_align_val_t(const Type* type) const -> bool;
  [[nodiscard]] auto is_destroying_delete_t(const Type* type) const -> bool;

  [[nodiscard]] auto initializer_list_element_type(const Type* targetType)
      -> const Type*;

  [[nodiscard]] auto remove_all_extents(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_const(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_volatile(const Type* type) const -> const Type*;
  [[nodiscard]] auto remove_address_space(const Type* type) const
      -> const Type*;
  [[nodiscard]] auto add_cv(const Type* type, CvQualifiers cv) const
      -> const Type*;
  [[nodiscard]] auto add_qualifiers(
      const Type* type, CvQualifiers cv,
      std::optional<std::uint32_t> addressSpace) const -> const Type*;
  [[nodiscard]] auto remove_noexcept(const Type* type) const -> const Type*;

  [[nodiscard]] auto remove_function_qualifiers(const FunctionType* type) const
      -> const FunctionType*;
  [[nodiscard]] auto replace_placeholder_types(const Type* type,
                                               const Type* replacement) const
      -> const Type*;
  [[nodiscard]] auto is_member_of_object_type(const Type* objectType,
                                              Symbol* member) -> bool;

  [[nodiscard]] auto is_base_of(const Type* base, const Type* derived) -> bool;
  [[nodiscard]] auto is_layout_compatible(const Type* first, const Type* second)
      -> bool;
  [[nodiscard]] auto is_pointer_interconvertible_base_of(const Type* base,
                                                         const Type* derived)
      -> bool;

  [[nodiscard]] auto is_known_complete_object(ExpressionAST* expression) const
      -> bool;

  [[nodiscard]] auto is_virtual_member_dispatch(
      FunctionSymbol* function, ExpressionAST* objectExpression) const -> bool;

  [[nodiscard]] auto adjusted_cv_type(const Type* type) const -> const Type*;

  [[nodiscard]] static auto adjusted_parameter_type(Control* control,
                                                    const Type* type)
      -> const Type*;

  [[nodiscard]] auto is_similar(const Type* lhs, const Type* rhs) const -> bool;

  [[nodiscard]] auto qualification_combined_type(const Type* lhs,
                                                 const Type* rhs) const
      -> const Type*;

  [[nodiscard]] auto is_qualification_convertible(const Type* from,
                                                  const Type* to) const -> bool;

  [[nodiscard]] auto is_variable_length_array(const Type* type) const -> bool;
  [[nodiscard]] auto is_vla_compatible(const Type* left,
                                       const Type* right) const -> bool;

  [[nodiscard]] auto is_reference_related(const Type* lhs, const Type* rhs)
      -> bool;

  [[nodiscard]] auto is_reference_compatible(const Type* target,
                                             const Type* source) -> bool;

  [[nodiscard]] auto promoted_integer_type(const Type* type) const
      -> const Type*;

  [[nodiscard]] auto promoted_arithmetic_type(const Type* type) const
      -> const Type*;

  [[nodiscard]] auto promoted_enumeration_types(const EnumType* enumType) const
      -> std::pair<const Type*, const Type*>;

  [[nodiscard]] auto integral_promotion_candidates() const
      -> std::array<const Type*, 6>;

  [[nodiscard]] auto is_integral_promotion(const Type* from,
                                           const Type* to) const -> bool;

  [[nodiscard]] auto is_floating_point_promotion(const Type* from,
                                                 const Type* to) const -> bool;

  [[nodiscard]] auto integer_conversion_rank(const Type* type) const
      -> std::pair<int, int>;

  [[nodiscard]] auto floating_point_conversion_order(const Type* a,
                                                     const Type* b) const
      -> std::partial_ordering;

  [[nodiscard]] auto floating_point_conversion_subrank(const Type* type) const
      -> int;

  [[nodiscard]] auto represents_all_values_of(const Type* target,
                                              const Type* source) const -> bool;

  [[nodiscard]] auto is_virtual_base_of(const Type* base,
                                        const Type* derived) const -> bool;

  [[nodiscard]] auto is_corresponding_overrider(
      const FunctionSymbol* overrider, const FunctionSymbol* overridden) const
      -> bool;

  [[nodiscard]] auto has_corresponding_object_parameters(
      const FunctionSymbol* overrider, const FunctionSymbol* overridden) const
      -> bool;

  struct CovariantReturnClasses {
    ClassSymbol* overriddenClass = nullptr;
    ClassSymbol* overriderClass = nullptr;
  };

  [[nodiscard]] auto is_covariant_return_type(
      const Type* overriddenReturnType, const Type* overriderReturnType,
      CovariantReturnClasses* classes = nullptr) -> bool;
  [[nodiscard]] auto is_convertible(const Type* from, const Type* to) const
      -> bool;
  [[nodiscard]] auto is_nothrow_convertible(const Type* from,
                                            const Type* to) const -> bool;
  [[nodiscard]] auto reference_constructs_from_temporary(const Type* to,
                                                         const Type* from) const
      -> bool;
  [[nodiscard]] auto reference_converts_from_temporary(const Type* to,
                                                       const Type* from) const
      -> bool;
  [[nodiscard]] auto declval(const Type* type) const -> ExpressionAST*;
  [[nodiscard]] auto has_new_extended_alignment(const Type* type) const -> bool;
  [[nodiscard]] auto conditional_operator_type(const Type* first,
                                               const Type* second) const
      -> const Type*;

  auto is_pod(const Type* type) -> bool;
  [[nodiscard]] auto has_pod_object_sizes(const Type* type) -> bool;
  [[nodiscard]] auto data_size(const Type* type) -> std::uint64_t;
  [[nodiscard]] auto non_virtual_size(const Type* type) -> std::uint64_t;
  auto is_trivial(const Type* type) -> bool;
  auto is_standard_layout(const Type* type) -> bool;

  enum class StandardLayoutRule { kLanguage, kLayout };
  [[nodiscard]] auto standard_layout(const Type* type, StandardLayoutRule rule)
      -> bool;
  [[nodiscard]] auto is_literal_type(const Type* type) -> bool;
  [[nodiscard]] auto is_structural(const Type* type) -> bool;
  [[nodiscard]] auto is_std_meta_type(const Type* type,
                                      std::string_view name) const -> bool;
  [[nodiscard]] auto has_constexpr_standard_exception() const -> bool;
  [[nodiscard]] auto handler_matches(const Type* handlerType,
                                     const Type* exceptionType) const -> bool;
  [[nodiscard]] auto is_implicit_lifetime(const Type* type) -> bool;
  [[nodiscard]] auto synthesizes_from_spaceship(TokenKind op, const Type* left,
                                                const Type* right) -> bool;
  auto is_aggregate(const Type* type) -> bool;
  [[nodiscard]] auto aggregate_elements(ClassSymbol* classSymbol) const
      -> std::vector<Symbol*>;
  [[nodiscard]] auto aggregate_element_type(Symbol* element) const
      -> const Type*;
  auto is_empty(const Type* type) -> bool;
  [[nodiscard]] auto requires_zero_initialization(const Type* type,
                                                  FunctionSymbol* constructor)
      -> bool;
  [[nodiscard]] auto is_zero_size_subobject(FieldSymbol* field) -> bool;
  auto is_polymorphic(const Type* type) -> bool;
  auto is_final(const Type* type) -> bool;
  auto select_constructor(ClassSymbol* classSymbol,
                          std::span<const Type* const> argTypes)
      -> FunctionSymbol*;
  auto is_constructible(const Type* type, std::span<const Type* const> argTypes)
      -> bool;
  [[nodiscard]] auto is_const_default_constructible(const Type* type) -> bool;
  auto is_nothrow_constructible(const Type* type,
                                std::span<const Type* const> argTypes) -> bool;
  auto is_trivially_constructible(const Type* type,
                                  std::span<const Type* const> argTypes = {})
      -> bool;
  auto select_assignment_operator(const Type* to, const Type* from)
      -> FunctionSymbol*;
  auto is_assignable(const Type* to, const Type* from) -> bool;
  [[nodiscard]] auto is_swappable_with(const Type* first, const Type* second)
      -> bool;
  [[nodiscard]] auto is_nothrow_swappable_with(const Type* first,
                                               const Type* second) -> bool;
  [[nodiscard]] auto is_swappable(const Type* type) -> bool;
  [[nodiscard]] auto is_nothrow_swappable(const Type* type) -> bool;
  auto is_nothrow_assignable(const Type* to, const Type* from) -> bool;
  auto is_trivially_assignable(const Type* to, const Type* from) -> bool;
  auto is_trivially_copyable(const Type* type) -> bool;
  [[nodiscard]] auto has_unique_object_representations(const Type* type)
      -> bool;
  auto has_constexpr_unknown_representation(const Type* type) -> bool;
  auto is_non_trivial_for_calls(const Type* type) -> bool;
  auto is_abstract(const Type* type) -> bool;
  auto is_destructible(const Type* type) -> bool;
  auto is_nothrow_destructible(const Type* type) -> bool;
  [[nodiscard]] auto is_nothrow_function(FunctionSymbol* function) const
      -> bool;
  [[nodiscard]] auto has_trivial_default_constructor(const Type* type) -> bool;
  [[nodiscard]] auto is_address_space_generic_member(FunctionSymbol* function)
      -> bool;
  [[nodiscard]] auto is_trivial_special_member(FunctionSymbol* function)
      -> bool;
  auto has_trivial_destructor(const Type* type) -> bool;
  auto is_trivially_destructible(const Type* type) -> bool;
  auto has_virtual_destructor(const Type* type) -> bool;
  [[nodiscard]] auto has_mutable_subobject(const Type* type) -> bool;

  [[nodiscard]] static auto entity_of(const Type* type) -> Symbol*;
  [[nodiscard]] auto is_namespace(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_namespace_alias(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_variable(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_structured_binding(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_function(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_enumerator(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_template(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_function_template(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_variable_template(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_class_template(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_alias_template(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_concept(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_type_alias(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_nonstatic_data_member(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_bit_field(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_mutable_member(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_default_member_initializer(Symbol* symbol) const
      -> bool;
  [[nodiscard]] auto is_virtual(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_pure_virtual(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_override(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_final(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_deleted(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_defaulted(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_user_provided(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_user_declared(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_explicit(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_static_storage_duration(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_thread_storage_duration(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_automatic_storage_duration(Symbol* symbol) const
      -> bool;
  [[nodiscard]] auto is_conversion_function(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_operator_function(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_literal_operator(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_constructor(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_destructor(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_default_constructor(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_copy_constructor(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_move_constructor(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_assignment(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_copy_assignment(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_move_assignment(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_special_member_function(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_conversion_function_template(Symbol* symbol) const
      -> bool;
  [[nodiscard]] auto is_operator_function_template(Symbol* symbol) const
      -> bool;
  [[nodiscard]] auto is_literal_operator_template(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_constructor_template(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_static_member(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_public(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_protected(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_private(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_namespace_member(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_base(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_template_arguments(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_parent(Symbol* symbol) const -> bool;
  enum class NameLinkage { kNone, kInternal, kModule, kExternal };
  [[nodiscard]] auto linkage_of(Symbol* symbol) const -> NameLinkage;
  [[nodiscard]] auto has_internal_linkage(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_module_linkage(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_external_linkage(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_c_language_linkage(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_linkage(Symbol* symbol) const -> bool;
  [[nodiscard]] auto has_identifier(Symbol* symbol) const -> bool;
  [[nodiscard]] auto identifier_of(Symbol* symbol) const -> std::string;
  [[nodiscard]] auto parent_of(Symbol* symbol) const -> Symbol*;
  [[nodiscard]] auto is_enumerable(const Type* type) -> bool;
  [[nodiscard]] auto enumerators_of(const Type* type) const
      -> std::optional<std::vector<Symbol*>>;
  [[nodiscard]] auto subobject_offset(Symbol* symbol) const
      -> std::optional<std::uint64_t>;
  [[nodiscard]] auto layout_type(Symbol* symbol) const -> const Type*;
  [[nodiscard]] auto is_accessible_member(Symbol* member, ScopeSymbol* scope,
                                          ClassSymbol* designating) const
      -> std::optional<bool>;
  [[nodiscard]] auto members_of(ScopeSymbol* scope,
                                ScopeSymbol* evaluationScope,
                                SourceLocation evaluationPoint) const
      -> std::vector<Symbol*>;
  [[nodiscard]] auto class_is_complete_at(ClassSymbol* cls,
                                          ScopeSymbol* evaluationScope,
                                          SourceLocation evaluationPoint) const
      -> bool;
  [[nodiscard]] auto data_members_of(ClassSymbol* cls, bool isStatic) const
      -> std::vector<Symbol*>;

  [[nodiscard]] auto is_default_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_copy_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_move_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_trivially_default_constructible(const Type* type)
      -> bool;
  [[nodiscard]] auto is_trivially_copy_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_trivially_move_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_nothrow_default_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_nothrow_copy_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_nothrow_move_constructible(const Type* type) -> bool;
  [[nodiscard]] auto is_copy_assignable(const Type* type) -> bool;
  [[nodiscard]] auto is_move_assignable(const Type* type) -> bool;
  [[nodiscard]] auto is_trivially_copy_assignable(const Type* type) -> bool;
  [[nodiscard]] auto is_trivially_move_assignable(const Type* type) -> bool;
  [[nodiscard]] auto is_nothrow_copy_assignable(const Type* type) -> bool;
  [[nodiscard]] auto is_nothrow_move_assignable(const Type* type) -> bool;
  [[nodiscard]] auto is_reflection(const Type* type) const -> bool;
  [[nodiscard]] auto rank(const Type* type) const -> std::intmax_t;
  [[nodiscard]] auto extent(const Type* type, std::uintmax_t dimension) const
      -> std::size_t;
  [[nodiscard]] auto unwrap_ref_decay(const Type* type) const -> const Type*;
  [[nodiscard]] static auto is_user_provided(FunctionSymbol* function) -> bool;
  [[nodiscard]] auto is_deleted_function(FunctionSymbol* function) const
      -> bool;
  [[nodiscard]] auto is_excluded_defaulted_move_member(
      FunctionSymbol* function) const -> bool;
  [[nodiscard]] auto is_potentially_constant(Symbol* symbol) const -> bool;
  [[nodiscard]] auto is_usable_in_constant_expressions(
      Symbol* symbol, ScopeSymbol* point = nullptr) const -> bool;
  [[nodiscard]] auto is_constexpr_representable(const ConstValue& value,
                                                const Type* type,
                                                ScopeSymbol* point) const
      -> bool;
  [[nodiscard]] auto is_constant_initialization(Symbol* variable,
                                                const ConstValue& value) const
      -> bool;
  [[nodiscard]] auto has_manifestly_constant_evaluated_initializer(
      Symbol* variable) const -> bool;
  [[nodiscard]] auto is_constant_template_argument(const ConstValue& value,
                                                   const Type* type) const
      -> bool;
  [[nodiscard]] auto has_const_member(const Type* type) -> bool;

 private:
  [[nodiscard]] auto require_layout_type(const Type* type) -> const Type*;

  [[nodiscard]] auto can_initialize(const Type* to, const Type* from,
                                    bool directInitialization) const -> bool;
  [[nodiscard]] auto reference_binds_to_temporary(
      const Type* to, const Type* from, bool directInitialization) const
      -> bool;
  [[nodiscard]] auto is_accessible_from_unrelated_context(
      FunctionSymbol* function) const -> bool;
  [[nodiscard]] auto default_initialization_invokes_user_provided_constructor(
      ClassSymbol* classSymbol) -> bool;
  [[nodiscard]] auto has_const_default_constructible_subobjects(
      ClassSymbol* classSymbol) -> bool;
  [[nodiscard]] auto is_nothrow_initialization(const Type* to, const Type* from,
                                               bool directInitialization) const
      -> bool;
  [[nodiscard]] auto is_trivial_initialization(const Type* to, const Type* from,
                                               bool directInitialization) const
      -> bool;
  [[nodiscard]] auto apply_sign(const Type* type, bool isUnsigned) const
      -> const Type*;
  [[nodiscard]] auto corresponding_integer_type(const Type* type,
                                                bool isUnsigned) const
      -> const Type*;
  [[nodiscard]] auto integer_type_of_size(std::size_t size,
                                          bool isUnsigned) const -> const Type*;
};

}  // namespace cxx
