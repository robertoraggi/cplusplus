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
class Semantics;

class Parser final {
 public:
  Parser(const Parser&) = delete;
  Parser& operator=(const Parser&) = delete;

  Parser(TranslationUnit* unit);
  ~Parser();

  bool checkTypes() const;
  void setCheckTypes(bool checkTypes);

  bool operator()(UnitAST*& ast);

  bool parse(UnitAST*& ast);

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
  struct ClassSpecifierContext;

  struct ExprContext {
    bool templParam = false;
    bool templArg = false;
  };

  bool parse_warn(std::string message) {
    unit->warning(SourceLocation(cursor_), std::move(message));
    return true;
  }

  bool parse_warn(SourceLocation loc, std::string message) {
    unit->warning(loc, std::move(message));
    return true;
  }

  bool parse_error(std::string message) {
    if (lastErrorCursor_ == cursor_) return true;
    lastErrorCursor_ = cursor_;
    unit->error(SourceLocation(cursor_), std::move(message));
    return true;
  }

  bool parse_error(SourceLocation loc, std::string message) {
    unit->error(loc, std::move(message));
    return true;
  }

  bool parse_id(const Identifier* id);
  bool parse_id(const Identifier* id, SourceLocation& loc);
  bool parse_nospace();
  bool parse_greater_greater();
  bool parse_greater_greater_equal();
  bool parse_greater_equal();
  bool parse_header_name(SourceLocation& loc);
  bool parse_export_keyword(SourceLocation& loc);
  bool parse_import_keyword(SourceLocation& loc);
  bool parse_module_keyword(SourceLocation& loc);
  bool parse_final();
  bool parse_override();
  bool parse_name_id(NameAST*& yyast);
  bool parse_literal(ExpressionAST*& yyast);
  bool parse_translation_unit(UnitAST*& yyast);
  bool parse_module_head();
  bool parse_module_unit(UnitAST*& yyast);
  bool parse_top_level_declaration_seq(UnitAST*& yyast);
  bool parse_skip_top_level_declaration(bool& skipping);
  bool parse_declaration_seq(List<DeclarationAST*>*& yyast);
  bool parse_skip_declaration(bool& skipping);
  bool parse_primary_expression(ExpressionAST*& yyast,
                                bool inRequiresClause = false);
  bool parse_id_expression(NameAST*& yyast, bool inRequiresClause = false);
  bool parse_maybe_template_id(NameAST*& yyast, bool inRequiresClause = false);
  bool parse_unqualified_id(NameAST*& yyast, bool inRequiresClause = false);
  bool parse_qualified_id(NameAST*& yyast, bool inRequiresClause = false);
  bool parse_nested_name_specifier(NestedNameSpecifierAST*& yyast);
  bool parse_start_of_nested_name_specifier(NameAST*& yyast,
                                            SourceLocation& scopeLoc);
  bool parse_lambda_expression(ExpressionAST*& yyast);
  bool parse_lambda_introducer(LambdaIntroducerAST*& yyast);
  bool parse_lambda_declarator(LambdaDeclaratorAST*& yyast);
  bool parse_lambda_capture(SourceLocation& captureDefaultLoc,
                            List<LambdaCaptureAST*>*& yyast);
  bool parse_capture_default(SourceLocation& opLoc);
  bool parse_capture_list(List<LambdaCaptureAST*>*& yyast);
  bool parse_capture(LambdaCaptureAST*& yyast);
  bool parse_simple_capture(LambdaCaptureAST*& yyast);
  bool parse_init_capture(LambdaCaptureAST*& yyast);
  bool parse_fold_expression(ExpressionAST*& yyast);
  bool parse_fold_operator(SourceLocation& loc, TokenKind& op);
  bool parse_requires_expression(ExpressionAST*& yyast);
  bool parse_requirement_parameter_list(
      SourceLocation& lparenLoc,
      ParameterDeclarationClauseAST*& parameterDeclarationClause,
      SourceLocation& rparenLoc);
  bool parse_requirement_body(RequirementBodyAST*& yyast);
  bool parse_requirement_seq(List<RequirementAST*>*& yyast);
  bool parse_requirement(RequirementAST*& yyast);
  bool parse_simple_requirement(RequirementAST*& yyast);
  bool parse_type_requirement(RequirementAST*& yyast);
  bool parse_compound_requirement(RequirementAST*& yyast);
  bool parse_return_type_requirement(SourceLocation& minusGreaterLoc,
                                     TypeConstraintAST*& typeConstraint);
  bool parse_nested_requirement(RequirementAST*& yyast);
  bool parse_postfix_expression(ExpressionAST*& yyast);
  bool parse_start_of_postfix_expression(ExpressionAST*& yyast);
  bool parse_member_expression(ExpressionAST*& yyast);
  bool parse_subscript_expression(ExpressionAST*& yyast);
  bool parse_call_expression(ExpressionAST*& yyast);
  bool parse_postincr_expression(ExpressionAST*& yyast);
  bool parse_cpp_cast_head(SourceLocation& loc);
  bool parse_cpp_cast_expression(ExpressionAST*& yyast);
  bool parse_cpp_type_cast_expression(ExpressionAST*& yyast);
  bool parse_typeid_expression(ExpressionAST*& yyast);
  bool parse_typename_expression(ExpressionAST*& yyast);
  bool parse_builtin_function_1();
  bool parse_builtin_function_2();
  bool parse_builtin_call_expression(ExpressionAST*& yyast);
  bool parse_expression_list(List<ExpressionAST*>*& yyast);
  bool parse_unary_expression(ExpressionAST*& yyast);
  bool parse_unop_expression(ExpressionAST*& yyast);
  bool parse_complex_expression(ExpressionAST*& yyast);
  bool parse_sizeof_expression(ExpressionAST*& yyast);
  bool parse_alignof_expression(ExpressionAST*& yyast);
  bool parse_unary_operator(SourceLocation& opLoc);
  bool parse_await_expression(ExpressionAST*& yyast);
  bool parse_noexcept_expression(ExpressionAST*& yyast);
  bool parse_new_expression(ExpressionAST*& yyast);
  bool parse_new_placement();
  bool parse_new_type_id(NewTypeIdAST*& yyast);
  bool parse_new_declarator();
  bool parse_noptr_new_declarator();
  bool parse_new_initializer(NewInitializerAST*& yyast);
  bool parse_delete_expression(ExpressionAST*& yyast);
  bool parse_cast_expression(ExpressionAST*& yyast);
  bool parse_cast_expression_helper(ExpressionAST*& yyast);
  bool parse_binary_operator(SourceLocation& loc, TokenKind& tk,
                             const ExprContext& exprContext);
  bool parse_binary_expression(ExpressionAST*& yyast,
                               const ExprContext& exprContext);
  bool parse_lookahead_binary_operator(SourceLocation& loc, TokenKind& tk,
                                       const ExprContext& exprContext);
  bool parse_binary_expression_helper(ExpressionAST*& yyast, Prec minPrec,
                                      const ExprContext& exprContext);
  bool parse_logical_or_expression(ExpressionAST*& yyast,
                                   const ExprContext& exprContext);
  bool parse_conditional_expression(ExpressionAST*& yyast,
                                    const ExprContext& exprContext);
  bool parse_yield_expression(ExpressionAST*& yyast);
  bool parse_throw_expression(ExpressionAST*& yyast);
  bool parse_assignment_expression(ExpressionAST*& yyast);
  bool parse_assignment_expression(ExpressionAST*& yyast,
                                   const ExprContext& exprContext);
  bool parse_assignment_operator(SourceLocation& loc, TokenKind& op);
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
  bool parse_compound_statement(CompoundStatementAST*& yyast,
                                bool skip = false);
  void finish_compound_statement(CompoundStatementAST* yyast);
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
  bool parse_notypespec_function_definition(
      DeclarationAST*& yyast, List<SpecifierAST*>* declSpecifierList,
      const DeclSpecs& specs);
  bool parse_static_assert_declaration(DeclarationAST*& yyast);
  bool parse_string_literal_seq(List<SourceLocation>*& yyast);
  bool parse_empty_declaration(DeclarationAST*& yyast);
  bool parse_attribute_declaration(DeclarationAST*& yyast);
  bool parse_decl_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_decl_specifier_seq(List<SpecifierAST*>*& yyast, DeclSpecs& specs);
  bool parse_decl_specifier_seq_no_typespecs(List<SpecifierAST*>*& yyast,
                                             DeclSpecs& specs);
  bool parse_decl_specifier_seq_no_typespecs(List<SpecifierAST*>*& yyast);
  bool parse_storage_class_specifier(SpecifierAST*& yyast);
  bool parse_function_specifier(SpecifierAST*& yyast);
  bool parse_explicit_specifier(SpecifierAST*& yyast);
  bool parse_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_type_specifier_seq(List<SpecifierAST*>*& yyast);
  bool parse_defining_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_defining_type_specifier_seq(List<SpecifierAST*>*& yyast,
                                         DeclSpecs& specs);
  bool parse_simple_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_named_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_named_type_specifier_helper(SpecifierAST*& yyast,
                                         DeclSpecs& specs);
  bool parse_placeholder_type_specifier_helper(SpecifierAST*& yyast,
                                               DeclSpecs& specs);
  bool parse_decltype_specifier_type_specifier(SpecifierAST*& yyast,
                                               DeclSpecs& specs);
  bool parse_underlying_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_atomic_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_primitive_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_type_name(NameAST*& yyast);
  bool parse_elaborated_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_elaborated_type_specifier_helper(
      ElaboratedTypeSpecifierAST*& yyast, DeclSpecs& specs);
  bool parse_elaborated_enum_specifier(ElaboratedTypeSpecifierAST*& yyast,
                                       DeclSpecs& specs);
  bool parse_decltype_specifier(SpecifierAST*& yyast);
  bool parse_placeholder_type_specifier(SpecifierAST*& yyast);
  bool parse_init_declarator(InitDeclaratorAST*& yyast, const DeclSpecs& specs);
  bool parse_declarator_initializer(RequiresClauseAST*& requiresClause,
                                    InitializerAST*& yyast);
  bool parse_declarator(DeclaratorAST*& yyastl);
  bool parse_ptr_operator_seq(List<PtrOperatorAST*>*& yyast);
  bool parse_core_declarator(CoreDeclaratorAST*& yyast);
  bool parse_noptr_declarator(DeclaratorAST*& yyast,
                              List<PtrOperatorAST*>* ptrOpLst);
  bool parse_parameters_and_qualifiers(ParametersAndQualifiersAST*& yyast);
  bool parse_cv_qualifier_seq(List<SpecifierAST*>*& yyast);
  bool parse_trailing_return_type(TrailingReturnTypeAST*& yyast);
  bool parse_ptr_operator(PtrOperatorAST*& yyast);
  bool parse_cv_qualifier(SpecifierAST*& yyast);
  bool parse_ref_qualifier(SourceLocation& refLoc);
  bool parse_declarator_id(IdDeclaratorAST*& yyast);
  bool parse_type_id(TypeIdAST*& yyast);
  bool parse_defining_type_id(TypeIdAST*& yyast);
  bool parse_abstract_declarator(DeclaratorAST*& yyast);
  bool parse_ptr_abstract_declarator(DeclaratorAST*& yyast);
  bool parse_noptr_abstract_declarator(DeclaratorAST*& yyast);
  bool parse_abstract_pack_declarator();
  bool parse_noptr_abstract_pack_declarator();
  bool parse_parameter_declaration_clause(
      ParameterDeclarationClauseAST*& yyast);
  bool parse_parameter_declaration_list(List<ParameterDeclarationAST*>*& yyast);
  bool parse_parameter_declaration(ParameterDeclarationAST*& yyast,
                                   bool templParam);
  bool parse_initializer(InitializerAST*& yyast);
  bool parse_brace_or_equal_initializer(InitializerAST*& yyast);
  bool parse_initializer_clause(ExpressionAST*& yyast, bool templParam = false);
  bool parse_braced_init_list(BracedInitListAST*& yyast);
  bool parse_initializer_list(List<ExpressionAST*>*& yyast);
  bool parse_designated_initializer_clause();
  bool parse_designator();
  bool parse_expr_or_braced_init_list(ExpressionAST*& yyast);
  bool parse_virt_specifier_seq();
  bool lookat_function_body();
  bool parse_function_body(FunctionBodyAST*& yyast);
  bool parse_enum_specifier(SpecifierAST*& yyast);
  bool parse_enum_head_name(NestedNameSpecifierAST*& nestedNameSpecifier,
                            NameAST*& name);
  bool parse_opaque_enum_declaration(DeclarationAST*& yyast);
  bool parse_enum_key(SourceLocation& enumLoc, SourceLocation& classLoc);
  bool parse_enum_base(EnumBaseAST*& yyast);
  bool parse_enumerator_list(List<EnumeratorAST*>*& yyast);
  bool parse_enumerator_definition(EnumeratorAST*& yast);
  bool parse_enumerator(EnumeratorAST*& yyast);
  bool parse_using_enum_declaration(DeclarationAST*& yyast);
  bool parse_namespace_definition(DeclarationAST*& yyast);
  bool parse_namespace_body(NamespaceDefinitionAST* yyast);
  bool parse_namespace_alias_definition(DeclarationAST*& yyast);
  bool parse_qualified_namespace_specifier(
      NestedNameSpecifierAST*& nestedNameSpecifier, NameAST*& name);
  bool parse_using_directive(DeclarationAST*& yyast);
  bool parse_using_declaration(DeclarationAST*& yyast);
  bool parse_using_declarator_list(List<UsingDeclaratorAST*>*& yyast);
  bool parse_using_declarator(UsingDeclaratorAST*& yyast);
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
  bool parse_module_declaration(ModuleDeclarationAST*& yyast);
  bool parse_module_name(ModuleNameAST*& yyast);
  bool parse_module_partition(ModulePartitionAST*& yyast);
  bool parse_export_declaration(DeclarationAST*& yyast);
  bool parse_maybe_import();
  bool parse_module_import_declaration(DeclarationAST*& yyast);
  bool parse_import_name(ImportNameAST*& yyast);
  bool parse_global_module_fragment(GlobalModuleFragmentAST*& yyast);
  bool parse_private_module_fragment(PrivateModuleFragmentAST*& yyast);
  bool parse_class_specifier(SpecifierAST*& yyast);
  bool parse_class_body(List<DeclarationAST*>*& yyast);
  bool parse_class_head(SourceLocation& classLoc,
                        List<AttributeAST*>*& attributeList, NameAST*& name,
                        BaseClauseAST*& baseClause);
  bool parse_class_head_name(NameAST*& yyast);
  bool parse_class_virt_specifier();
  bool parse_class_key(SourceLocation& classLoc);
  bool parse_member_specification(DeclarationAST*& yyast);
  bool parse_member_declaration(DeclarationAST*& yyast);
  bool parse_maybe_template_member();
  bool parse_member_declaration_helper(DeclarationAST*& yyast);
  bool parse_member_declarator_list(List<InitDeclaratorAST*>*& yyast,
                                    const DeclSpecs& specs);
  bool parse_member_declarator(InitDeclaratorAST*& yyast,
                               const DeclSpecs& specs);
  bool parse_virt_specifier();
  bool parse_pure_specifier();
  bool parse_conversion_function_id(NameAST*& yyast);
  bool parse_base_clause(BaseClauseAST*& yyast);
  bool parse_base_specifier_list(List<BaseSpecifierAST*>*& yyast);
  bool parse_base_specifier(BaseSpecifierAST*& yyast);
  bool parse_class_or_decltype(NameAST*& yyast);
  bool parse_access_specifier(SourceLocation& loc);
  bool parse_ctor_initializer(CtorInitializerAST*& yyast);
  bool parse_mem_initializer_list(List<MemInitializerAST*>*& yyast);
  bool parse_mem_initializer(MemInitializerAST*& yyast);
  bool parse_mem_initializer_id(NameAST*& yyast);
  bool parse_operator_function_id(NameAST*& yyast);
  bool parse_operator(TokenKind& op, SourceLocation& opLoc,
                      SourceLocation& openLoc, SourceLocation& closeLoc);
  bool parse_literal_operator_id(NameAST*& yyast);
  bool parse_template_declaration(DeclarationAST*& yyast);
  bool parse_template_head(SourceLocation& templateLoc, SourceLocation& lessLoc,
                           List<DeclarationAST*>*& templateParameterList,
                           SourceLocation& greaterLoc,
                           RequiresClauseAST*& requiresClause);
  bool parse_template_parameter_list(List<DeclarationAST*>*& yyast);
  bool parse_requires_clause(RequiresClauseAST*& yyast);
  bool parse_constraint_logical_or_expression(ExpressionAST*& yyast);
  bool parse_constraint_logical_and_expression(ExpressionAST*& yyast);
  bool parse_template_parameter(DeclarationAST*& yyast);
  bool parse_type_parameter(DeclarationAST*& yyast);
  bool parse_typename_type_parameter(DeclarationAST*& yyast);
  bool parse_template_type_parameter(DeclarationAST*& yyast);
  bool parse_constraint_type_parameter(DeclarationAST*& yyast);
  bool parse_type_parameter_key(SourceLocation& classKeyLoc);
  bool parse_type_constraint(TypeConstraintAST*& yyast,
                             bool parsingPlaceholderTypeSpec);
  bool parse_simple_template_id(NameAST*& yyast);
  bool parse_template_id(NameAST*& yyast);
  bool parse_template_argument_list(List<TemplateArgumentAST*>*& yyast);
  bool parse_template_argument(TemplateArgumentAST*& yyast);
  bool parse_constraint_expression(ExpressionAST*& yyast);
  bool parse_deduction_guide(DeclarationAST*& yyast);
  bool parse_concept_definition(DeclarationAST*& yyast);
  bool parse_concept_name(NameAST*& yyast);
  bool parse_typename_specifier(SpecifierAST*& yyast);
  bool parse_explicit_instantiation(DeclarationAST*& yyast);
  bool parse_explicit_specialization(DeclarationAST*& yyast);
  bool parse_try_block(StatementAST*& yyast);
  bool parse_function_try_block(FunctionBodyAST*& yyast);
  bool parse_handler(HandlerAST*& yyast);
  bool parse_handler_seq(List<HandlerAST*>*& yyast);
  bool parse_exception_declaration(ExceptionDeclarationAST*& yyast);
  bool parse_noexcept_specifier();
  bool parse_identifier_list();

 private:
  const Token& LA(int n = 0) const;

  bool match(TokenKind tk);
  bool match(TokenKind tk, SourceLocation& location);

  bool expect(TokenKind tk);
  bool expect(TokenKind tk, SourceLocation& location);

  SourceLocation consumeToken() { return SourceLocation(cursor_++); }

  SourceLocation currentLocation() const { return SourceLocation(cursor_); }

  void rewind(SourceLocation location) { cursor_ = location.index(); }

  void completePendingFunctionDefinitions();
  void completeFunctionDefinition(FunctionDefinitionAST* ast);

  void enterFunctionScope(FunctionSymbol* functionSymbol,
                          FunctionDeclaratorAST* functionDeclarator);

 private:
  TranslationUnit* unit = nullptr;
  Arena* pool = nullptr;
  Control* control = nullptr;
  SymbolFactory* symbols = nullptr;
  TypeEnvironment* types = nullptr;
  std::unique_ptr<Semantics> sem;
  bool skipFunctionBody_ = false;
  bool checkTypes_ = false;

  std::unordered_map<SourceLocation,
                     std::tuple<SourceLocation, ClassSpecifierAST*, bool>>
      class_specifiers_;

  std::unordered_map<
      SourceLocation,
      std::tuple<SourceLocation, ElaboratedTypeSpecifierAST*, bool>>
      elaborated_type_specifiers_;

  std::unordered_map<SourceLocation,
                     std::tuple<SourceLocation, TemplateArgumentAST*, bool>>
      template_arguments_;

  std::unordered_map<SourceLocation,
                     std::tuple<SourceLocation, NestedNameSpecifierAST*, bool>>
      nested_name_specifiers_;

  std::vector<FunctionDefinitionAST*> pendingFunctionDefinitions_;

  NamespaceSymbol* globalNamespace_ = nullptr;

  bool module_unit = false;
  const Identifier* module_id = nullptr;
  const Identifier* import_id = nullptr;
  const Identifier* final_id = nullptr;
  const Identifier* override_id = nullptr;
  int templArgDepth = 0;
  int classDepth = 0;
  uint32_t lastErrorCursor_ = 0;
  uint32_t cursor_ = 0;
};

}  // namespace cxx
