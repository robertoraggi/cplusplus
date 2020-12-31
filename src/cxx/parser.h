// Copyright (c) 2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/source_location.h>
#include <cxx/translation_unit.h>

#include <forward_list>
#include <functional>
#include <unordered_map>
#include <variant>

namespace cxx {

class TranslationUnit;

class Parser {
 public:
  bool operator()(TranslationUnit* unit);
  bool parse(TranslationUnit* unit);

  enum struct Prec {
    kLogicalOr,
    kLogicalAnd,
    kInclusiveOr,
    kExclusiveOr,
    kAnd,
    kEquality,
    kRelational,
    kCompare,
    kShift,
    kAdditive,
    kMultiplicative,
    kPm,
  };

  static Prec prec(TokenKind tk);

  struct DeclSpecs;
  struct TemplArgContext;

  struct DeclaratorId {};
  struct NestedDeclarator {};
  struct PtrDeclarator {};
  struct FunctionDeclarator {};
  struct ArrayDeclarator {};

  using DeclaratorComponent =
      std::variant<DeclaratorId, NestedDeclarator, PtrDeclarator,
                   FunctionDeclarator, ArrayDeclarator>;

  using Declarator = std::vector<DeclaratorComponent>;

  bool isFunctionDeclarator(const Declarator& decl) const;

  template <typename... Args>
  bool parse_warn(const std::string_view& format, const Args&... args) {
    unit->report(cursor_, MessageKind::Warning, format, args...);
    return true;
  }

  template <typename... Args>
  bool parse_error(const std::string_view& format, const Args&... args) {
    if (lastErrorCursor_ == cursor_) return true;
    lastErrorCursor_ = cursor_;
    unit->report(cursor_, MessageKind::Error, format, args...);
    // throw std::runtime_error("error");
    return true;
  }

  bool parse_error();
  bool parse_warn();
  bool parse_id(const Identifier* id);
  bool parse_nospace();
  bool parse_greater_greater();
  bool parse_greater_greater_equal();
  bool parse_greater_equal();
  bool parse_header_name();
  bool parse_export_keyword();
  bool parse_import_keyword();
  bool parse_module_keyword();
  bool parse_final();
  bool parse_override();
  bool parse_typedef_name();
  bool parse_namespace_name();
  bool parse_namespace_alias();
  bool parse_class_name();
  bool parse_class_name(Name& name);
  bool parse_name_id(Name& name);
  bool parse_enum_name();
  bool parse_template_name();
  bool parse_template_name(Name& name);
  bool parse_literal();
  bool parse_translation_unit(UnitAST*& yyast);
  bool parse_module_head();
  bool parse_module_unit(UnitAST*& yyast);
  bool parse_top_level_declaration_seq(UnitAST*& yyast);
  bool parse_skip_top_level_declaration(bool& skipping);
  bool parse_declaration_seq();
  bool parse_skip_declaration(bool& skipping);
  bool parse_primary_expression(ExpressionAST*& yyast);
  bool parse_id_expression();
  bool parse_maybe_template_id();
  bool parse_unqualified_id();
  bool parse_qualified_id();
  bool parse_nested_name_specifier();
  bool parse_start_of_nested_name_specifier(Name& id);
  bool parse_lambda_expression(ExpressionAST*& yyast);
  bool parse_lambda_introducer();
  bool parse_lambda_declarator();
  bool parse_lambda_capture();
  bool parse_capture_default();
  bool parse_capture_list();
  bool parse_capture();
  bool parse_simple_capture();
  bool parse_init_capture();
  bool parse_fold_expression(ExpressionAST*& yyast);
  bool parse_fold_operator();
  bool parse_requires_expression(ExpressionAST*& yyast);
  bool parse_requirement_parameter_list();
  bool parse_requirement_body();
  bool parse_requirement_seq();
  bool parse_requirement();
  bool parse_simple_requirement();
  bool parse_type_requirement();
  bool parse_compound_requirement();
  bool parse_return_type_requirement();
  bool parse_nested_requirement();
  bool parse_postfix_expression(ExpressionAST*& yyast);
  bool parse_start_of_postfix_expression(ExpressionAST*& yyast);
  bool parse_member_expression(ExpressionAST*& yyast);
  bool parse_subscript_expression(ExpressionAST*& yyast);
  bool parse_call_expression(ExpressionAST*& yyast);
  bool parse_postincr_expression(ExpressionAST*& yyast);
  bool parse_cpp_cast_head();
  bool parse_cpp_cast_expression(ExpressionAST*& yyast);
  bool parse_cpp_type_cast_expression(ExpressionAST*& yyast);
  bool parse_typeid_expression(ExpressionAST*& yyast);
  bool parse_typename_expression(ExpressionAST*& yyast);
  bool parse_builtin_function_1();
  bool parse_builtin_function_2();
  bool parse_builtin_call_expression(ExpressionAST*& yyast);
  bool parse_expression_list();
  bool parse_unary_expression(ExpressionAST*& yyast);
  bool parse_unop_expression(ExpressionAST*& yyast);
  bool parse_complex_expression(ExpressionAST*& yyast);
  bool parse_sizeof_expression(ExpressionAST*& yyast);
  bool parse_alignof_expression(ExpressionAST*& yyast);
  bool parse_unary_operator();
  bool parse_await_expression(ExpressionAST*& yyast);
  bool parse_noexcept_expression(ExpressionAST*& yyast);
  bool parse_new_expression(ExpressionAST*& yyast);
  bool parse_new_placement();
  bool parse_new_type_id();
  bool parse_new_declarator();
  bool parse_noptr_new_declarator();
  bool parse_new_initializer();
  bool parse_delete_expression(ExpressionAST*& yyast);
  bool parse_cast_expression(ExpressionAST*& yyast);
  bool parse_cast_expression_helper(ExpressionAST*& yyast);
  bool parse_binary_operator(TokenKind& tk, bool templArg);
  bool parse_binary_expression(ExpressionAST*& yyast, bool templArg);
  bool parse_lookahead_binary_operator(TokenKind& tk, bool templArg);
  bool parse_binary_expression_helper(ExpressionAST*& yyast, Prec minPrec,
                                      bool templArg);
  bool parse_logical_or_expression(ExpressionAST*& yyast, bool templArg);
  bool parse_conditional_expression(ExpressionAST*& yyast, bool templArg);
  bool parse_yield_expression(ExpressionAST*& yyast);
  bool parse_throw_expression(ExpressionAST*& yyast);
  bool parse_assignment_expression(ExpressionAST*& yyast);
  bool parse_assignment_operator();
  bool parse_expression(ExpressionAST*& yyast);
  bool parse_constant_expression(ExpressionAST*& yyast);
  bool parse_template_argument_constant_expression(ExpressionAST*& yyast);
  bool parse_statement(StatementAST*& yyast);
  bool parse_init_statement(StatementAST*& yyast);
  bool parse_condition(ExpressionAST*& yyast);
  bool parse_labeled_statement(StatementAST*& yyast);
  bool parse_case_statement(StatementAST*& yyast);
  bool parse_default_statement(StatementAST*& yyast);
  bool parse_expression_statement(StatementAST*& yyast);
  bool parse_compound_statement(StatementAST*& yyast);
  bool parse_skip_statement(bool& skipping);
  bool parse_if_statement(StatementAST*& yyast);
  bool parse_switch_statement(StatementAST*& yyast);
  bool parse_while_statement(StatementAST*& yyast);
  bool parse_do_statement(StatementAST*& yyast);
  bool parse_for_range_statement(StatementAST*& yyast);
  bool parse_for_statement(StatementAST*& yyast);
  bool parse_for_range_declaration(DeclarationAST*& yyast);
  bool parse_for_range_initializer(ExpressionAST*& yyast);
  bool parse_break_statement(StatementAST*& yyast);
  bool parse_continue_statement(StatementAST*& yyast);
  bool parse_return_statement(StatementAST*& yyast);
  bool parse_goto_statement(StatementAST*& yyast);
  bool parse_coroutine_return_statement(StatementAST*& yyast);
  bool parse_declaration_statement(StatementAST*& yyast);
  bool parse_maybe_module();
  bool parse_declaration(DeclarationAST*& yyast);
  bool parse_block_declaration(DeclarationAST*& yyast, bool fundef);
  bool parse_alias_declaration(DeclarationAST*& yyast);
  bool parse_simple_declaration(DeclarationAST*& yyast, bool fundef);
  bool parse_function_definition_body();
  bool parse_static_assert_declaration(DeclarationAST*& yyast);
  bool parse_string_literal_seq();
  bool parse_empty_declaration(DeclarationAST*& yyast);
  bool parse_attribute_declaration(DeclarationAST*& yyast);
  bool parse_decl_specifier(DeclSpecs& specs);
  bool parse_decl_specifier_seq(DeclSpecs& specs);
  bool parse_decl_specifier_seq_no_typespecs(DeclSpecs& specs);
  bool parse_decl_specifier_seq_no_typespecs();
  bool parse_decl_specifier_seq();
  bool parse_storage_class_specifier();
  bool parse_function_specifier();
  bool parse_explicit_specifier();
  bool parse_type_specifier(DeclSpecs& specs);
  bool parse_type_specifier_seq();
  bool parse_defining_type_specifier(DeclSpecs& specs);
  bool parse_defining_type_specifier_seq(DeclSpecs& specs);
  bool parse_simple_type_specifier(DeclSpecs& specs);
  bool parse_named_type_specifier(DeclSpecs& specs);
  bool parse_named_type_specifier_helper(DeclSpecs& specs);
  bool parse_placeholder_type_specifier_helper(DeclSpecs& specs);
  bool parse_decltype_specifier_type_specifier(DeclSpecs& specs);
  bool parse_underlying_type_specifier(DeclSpecs& specs);
  bool parse_automic_type_specifier(DeclSpecs& specs);
  bool parse_atomic_type_specifier(DeclSpecs& specs);
  bool parse_primitive_type_specifier(DeclSpecs& specs);
  bool parse_type_name();
  bool parse_elaborated_type_specifier(DeclSpecs& specs);
  bool parse_elaborated_enum_specifier();
  bool parse_decltype_specifier();
  bool parse_placeholder_type_specifier();
  bool parse_init_declarator_list();
  bool parse_init_declarator();
  bool parse_declarator_initializer();
  bool parse_declarator();
  bool parse_declarator(Declarator& decl);
  bool parse_ptr_operator_seq();
  bool parse_core_declarator(Declarator& decl);
  bool parse_noptr_declarator(Declarator& decl);
  bool parse_parameters_and_qualifiers();
  bool parse_cv_qualifier_seq();
  bool parse_trailing_return_type();
  bool parse_ptr_operator();
  bool parse_cv_qualifier();
  bool parse_ref_qualifier();
  bool parse_declarator_id();
  bool parse_type_id();
  bool parse_defining_type_id();
  bool parse_abstract_declarator();
  bool parse_ptr_abstract_declarator();
  bool parse_noptr_abstract_declarator();
  bool parse_abstract_pack_declarator();
  bool parse_noptr_abstract_pack_declarator();
  bool parse_parameter_declaration_clause();
  bool parse_parameter_declaration_list();
  bool parse_parameter_declaration();
  bool parse_initializer();
  bool parse_brace_or_equal_initializer();
  bool parse_initializer_clause(ExpressionAST*& yyast);
  bool parse_braced_init_list();
  bool parse_initializer_list();
  bool parse_designated_initializer_clause();
  bool parse_designator();
  bool parse_expr_or_braced_init_list(ExpressionAST*& yyast);
  bool parse_virt_specifier_seq();
  bool parse_function_body();
  bool parse_enum_specifier();
  bool parse_enum_head();
  bool parse_enum_head_name();
  bool parse_opaque_enum_declaration(DeclarationAST*& yyast);
  bool parse_enum_key();
  bool parse_enum_base();
  bool parse_enumerator_list();
  bool parse_enumerator_definition();
  bool parse_enumerator();
  bool parse_using_enum_declaration(DeclarationAST*& yyast);
  bool parse_namespace_definition(DeclarationAST*& yyast);
  bool parse_namespace_body();
  bool parse_namespace_alias_definition(DeclarationAST*& yyast);
  bool parse_qualified_namespace_specifier();
  bool parse_using_directive(DeclarationAST*& yyast);
  bool parse_using_declaration(DeclarationAST*& yyast);
  bool parse_using_declarator_list();
  bool parse_using_declarator();
  bool parse_asm_declaration(DeclarationAST*& yyast);
  bool parse_linkage_specification(DeclarationAST*& yyast);
  bool parse_attribute_specifier_seq(List<AttributeAST*>*& yyast);
  bool parse_attribute_specifier();
  bool parse_asm_specifier();
  bool parse_gcc_attribute();
  bool parse_gcc_attribute_seq();
  bool parse_skip_balanced();
  bool parse_alignment_specifier();
  bool parse_attribute_using_prefix();
  bool parse_attribute_list();
  bool parse_attribute();
  bool parse_attribute_token();
  bool parse_attribute_scoped_token();
  bool parse_attribute_namespace();
  bool parse_attribute_argument_clause();
  bool parse_module_declaration();
  bool parse_module_name();
  bool parse_module_partition();
  bool parse_module_name_qualifier();
  bool parse_export_declaration(DeclarationAST*& yyast);
  bool parse_maybe_import();
  bool parse_module_import_declaration(DeclarationAST*& yyast);
  bool parse_import_name();
  bool parse_global_module_fragment();
  bool parse_private_module_fragment();
  bool parse_class_specifier();
  bool parse_leave_class_specifier(SourceLocation start);
  bool parse_reject_class_specifier(SourceLocation start);
  bool parse_class_body();
  bool parse_class_head(Name& name);
  bool parse_class_head_name(Name& name);
  bool parse_class_virt_specifier();
  bool parse_class_key();
  bool parse_member_specification(DeclarationAST*& yyast);
  bool parse_member_declaration(DeclarationAST*& yyast);
  bool parse_maybe_template_member();
  bool parse_member_declaration_helper(DeclarationAST*& yyast);
  bool parse_member_function_definition_body();
  bool parse_member_declarator_modifier();
  bool parse_member_declarator_list();
  bool parse_member_declarator();
  bool parse_virt_specifier();
  bool parse_pure_specifier();
  bool parse_conversion_function_id();
  bool parse_conversion_type_id();
  bool parse_conversion_declarator();
  bool parse_base_clause();
  bool parse_base_specifier_list();
  bool parse_base_specifier();
  bool parse_class_or_decltype();
  bool parse_access_specifier();
  bool parse_ctor_initializer();
  bool parse_mem_initializer_list();
  bool parse_mem_initializer();
  bool parse_mem_initializer_id();
  bool parse_operator_function_id();
  bool parse_op();
  bool parse_literal_operator_id();
  bool parse_template_declaration(DeclarationAST*& yyast);
  bool parse_template_head();
  bool parse_template_parameter_list();
  bool parse_requires_clause();
  bool parse_constraint_logical_or_expression(ExpressionAST*& yyast);
  bool parse_constraint_logical_and_expression(ExpressionAST*& yyast);
  bool parse_template_parameter();
  bool parse_type_parameter();
  bool parse_typename_type_parameter();
  bool parse_template_type_parameter();
  bool parse_constraint_type_parameter();
  bool parse_type_parameter_key();
  bool parse_type_constraint();
  bool parse_simple_template_id();
  bool parse_simple_template_id(Name& name);
  bool parse_template_id();
  bool parse_template_argument_list();
  bool parse_template_argument();
  bool parse_constraint_expression(ExpressionAST*& yyast);
  bool parse_deduction_guide(DeclarationAST*& yyast);
  bool parse_concept_definition();
  bool parse_concept_name();
  bool parse_typename_specifier();
  bool parse_explicit_instantiation(DeclarationAST*& yyast);
  bool parse_explicit_specialization(DeclarationAST*& yyast);
  bool parse_try_block(StatementAST*& yyast);
  bool parse_function_try_block();
  bool parse_handler();
  bool parse_handler_seq();
  bool parse_exception_declaration();
  bool parse_noexcept_specifier();
  bool parse_identifier_list();

 private:
  const Token& LA(int n = 0) const;

  bool match(TokenKind tk) {
    if (LA().isNot(tk)) return false;
    (void)consumeToken();
    return true;
  }

  bool expect(TokenKind tk) {
    if (match(tk)) return true;
    parse_error("expected '{}'", Token::spell(tk));
    return false;
  }

  bool match(TokenKind tk, SourceLocation& location) {
    if (LA().isNot(tk)) return false;
    const auto loc = consumeToken();
    location = loc;
    return true;
  }

  bool expect(TokenKind tk, SourceLocation& location) {
    if (match(tk, location)) return true;
    parse_error("expected '{}'", Token::spell(tk));
    return false;
  }

  SourceLocation consumeToken() { return SourceLocation(cursor_++); }

  SourceLocation currentLocation() const { return SourceLocation(cursor_); }

  void rewind(SourceLocation location) { cursor_ = location.index(); }

 private:
  TranslationUnit* unit = nullptr;
  Arena* pool = nullptr;
  Control* control = nullptr;
  bool skip_function_body = false;

  std::unordered_map<SourceLocation, std::tuple<SourceLocation, bool>>
      class_specifiers_;

  std::unordered_map<SourceLocation, std::tuple<SourceLocation, bool>>
      template_arguments_;

  std::unordered_map<SourceLocation, std::tuple<SourceLocation, bool>>
      nested_name_specifiers_;

  bool module_unit = false;
  const Identifier* module_id = nullptr;
  const Identifier* import_id = nullptr;
  const Identifier* final_id = nullptr;
  const Identifier* override_id = nullptr;
  int templArgDepth = 0;
  uint32_t lastErrorCursor_ = 0;
  uint32_t cursor_ = 0;
};

}  // namespace cxx
