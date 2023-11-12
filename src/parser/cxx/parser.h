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
#include <cxx/control.h>
#include <cxx/parser_fwd.h>
#include <cxx/source_location.h>
#include <cxx/symbols_fwd.h>
#include <cxx/translation_unit.h>

#include <deque>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace cxx {

class Parser final {
 public:
  Parser(const Parser&) = delete;
  auto operator=(const Parser&) -> Parser& = delete;

  /**
   * Constructs a new parser.
   *
   * @param unit the translation unit to parse
   */
  explicit Parser(TranslationUnit* unit);
  ~Parser();

  [[nodiscard]] auto translationUnit() const -> TranslationUnit* {
    return unit;
  }

  [[nodiscard]] auto control() const -> Control* { return control_; }

  [[nodiscard]] auto config() const -> const ParserConfiguration& {
    return config_;
  }

  void setConfig(const ParserConfiguration& config) { config_ = config; }

  /**
   * Whether to enable fuzzy template resolution.
   */
  [[nodiscard]] auto fuzzyTemplateResolution() const -> bool;

  /**
   * Sets whether to enable fuzzy template resolution.
   *
   * When enabled, the parser will try to resolve template names
   *
   * @param fuzzyTemplateResolution whether to enable fuzzy template resolution
   */
  void setFuzzyTemplateResolution(bool fuzzyTemplateResolution);

  /**
   * Parse the given unit.
   */
  void parse(UnitAST*& ast);

  /**
   * Parse the given unit.
   */
  void operator()(UnitAST*& ast);

 private:
  struct DeclSpecs;
  struct Decl;
  struct TemplateHeadContext;
  struct ClassSpecifierContext;
  struct ScopeGuard;
  struct ExprContext;
  struct LookaheadParser;
  struct LoopParser;
  struct ClassHead;
  struct GetDeclaratorType;

  enum struct BindingContext {
    kNamespace,
    kClass,
    kTemplate,
    kBlock,
    kInitStatement,
    kCondition,
  };

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

  static auto prec(TokenKind tk) -> Prec;

  [[nodiscard]] auto context_allows_function_definition(
      BindingContext ctx) const -> bool {
    if (ctx == BindingContext::kBlock) return false;
    if (ctx == BindingContext::kInitStatement) return false;
    return true;
  }

  [[nodiscard]] auto context_allows_structured_bindings(
      BindingContext ctx) const -> bool {
    if (ctx == BindingContext::kBlock) return true;
    if (ctx == BindingContext::kInitStatement) return true;
    if (ctx == BindingContext::kCondition) return true;
    return false;
  }

  // diagnostics
  void parse_warn(std::string message);
  void parse_warn(SourceLocation loc, std::string message);
  void parse_error(std::string message);
  void parse_error(SourceLocation loc, std::string message);

  // generate diagnostic messages regardless of the current parsing state
  void warning(std::string message);
  void warning(SourceLocation loc, std::string message);
  void error(std::string message);
  void error(SourceLocation loc, std::string message);

  void parse_translation_unit(UnitAST*& yyast);

  [[nodiscard]] auto parse_id(const Identifier* id, SourceLocation& loc)
      -> bool;
  [[nodiscard]] auto parse_nospace() -> bool;
  [[nodiscard]] auto parse_greater_greater() -> bool;
  [[nodiscard]] auto parse_header_name(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_export_keyword(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_import_keyword(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_module_keyword(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_final(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_override(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_name_id(NameIdAST*& yyast) -> bool;
  [[nodiscard]] auto parse_literal(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_module_head() -> bool;
  [[nodiscard]] auto parse_module_unit(UnitAST*& yyast) -> bool;
  void parse_top_level_declaration_seq(UnitAST*& yyast);
  void parse_declaration_seq(List<DeclarationAST*>*& yyast);
  void parse_skip_declaration(bool& skipping);
  [[nodiscard]] auto parse_primary_expression(ExpressionAST*& yyast,
                                              bool inRequiresClause = false)
      -> bool;
  [[nodiscard]] auto parse_id_expression(IdExpressionAST*& yyast,
                                         bool inRequiresClause = false) -> bool;
  [[nodiscard]] auto parse_maybe_template_id(UnqualifiedIdAST*& yyast,
                                             bool isTemplateIntroduced,
                                             bool inRequiresClause) -> bool;
  [[nodiscard]] auto parse_unqualified_id(UnqualifiedIdAST*& yyast,
                                          bool isTemplateIntroduced,
                                          bool inRequiresClause) -> bool;
  void parse_optional_nested_name_specifier(NestedNameSpecifierAST*& yyast);
  [[nodiscard]] auto parse_nested_name_specifier(NestedNameSpecifierAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_lambda_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_lambda_specifier_seq(
      List<LambdaSpecifierAST*>*& yyast, LambdaSymbol* symbol) -> bool;
  [[nodiscard]] auto parse_lambda_capture(SourceLocation& captureDefaultLoc,
                                          List<LambdaCaptureAST*>*& yyast)
      -> bool;
  [[nodiscard]] auto parse_capture_default(SourceLocation& opLoc) -> bool;
  [[nodiscard]] auto parse_capture_list(List<LambdaCaptureAST*>*& yyast)
      -> bool;
  [[nodiscard]] auto parse_capture(LambdaCaptureAST*& yyast) -> bool;
  [[nodiscard]] auto parse_simple_capture(LambdaCaptureAST*& yyast) -> bool;
  [[nodiscard]] auto parse_init_capture(LambdaCaptureAST*& yyast) -> bool;
  [[nodiscard]] auto parse_this_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_nested_expession(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_fold_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_left_fold_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_fold_operator(SourceLocation& loc, TokenKind& op)
      -> bool;
  [[nodiscard]] auto parse_requires_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_requirement_parameter_list(
      SourceLocation& lparenLoc,
      ParameterDeclarationClauseAST*& parameterDeclarationClause,
      SourceLocation& rparenLoc) -> bool;
  void parse_requirement_seq(List<RequirementAST*>*& yyast);
  void parse_requirement(RequirementAST*& yyast);
  void parse_simple_requirement(RequirementAST*& yyast);
  [[nodiscard]] auto parse_type_requirement(RequirementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_compound_requirement(RequirementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_return_type_requirement(
      SourceLocation& minusGreaterLoc, TypeConstraintAST*& typeConstraint)
      -> bool;
  [[nodiscard]] auto parse_nested_requirement(RequirementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_postfix_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_start_of_postfix_expression(ExpressionAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_member_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_subscript_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_call_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_postincr_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_cpp_cast_head(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_cpp_cast_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_cpp_type_cast_expression(ExpressionAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_typeid_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_typename_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_type_traits_op(SourceLocation& loc,
                                          BuiltinKind& builtinKind) -> bool;
  [[nodiscard]] auto parse_builtin_call_expression(ExpressionAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_expression_list(List<ExpressionAST*>*& yyast)
      -> bool;
  [[nodiscard]] auto parse_unary_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_unop_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_complex_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_sizeof_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_alignof_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_unary_operator(SourceLocation& opLoc) -> bool;
  [[nodiscard]] auto parse_await_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_noexcept_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_new_expression(ExpressionAST*& yyast) -> bool;
  void parse_optional_new_placement(NewPlacementAST*& yyast);
  void parse_optional_new_initializer(NewInitializerAST*& yyast);
  [[nodiscard]] auto parse_delete_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_cast_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_cast_expression_helper(ExpressionAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_binary_operator(SourceLocation& loc, TokenKind& tk,
                                           const ExprContext& exprContext)
      -> bool;
  [[nodiscard]] auto parse_binary_expression(ExpressionAST*& yyast,
                                             const ExprContext& exprContext)
      -> bool;
  [[nodiscard]] auto parse_lookahead_binary_operator(
      SourceLocation& loc, TokenKind& tk, const ExprContext& exprContext)
      -> bool;
  [[nodiscard]] auto parse_binary_expression_helper(
      ExpressionAST*& yyast, Prec minPrec, const ExprContext& exprContext)
      -> bool;
  [[nodiscard]] auto parse_logical_or_expression(ExpressionAST*& yyast,
                                                 const ExprContext& exprContext)
      -> bool;
  [[nodiscard]] auto parse_conditional_expression(
      ExpressionAST*& yyast, const ExprContext& exprContext) -> bool;
  [[nodiscard]] auto parse_yield_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_throw_expression(ExpressionAST*& yyast) -> bool;
  void parse_assignment_expression(ExpressionAST*& yyast);
  void parse_assignment_expression(ExpressionAST*& yyast,
                                   const ExprContext& exprContext);
  [[nodiscard]] auto parse_maybe_assignment_expression(ExpressionAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_maybe_assignment_expression(
      ExpressionAST*& yyast, const ExprContext& exprContext) -> bool;
  [[nodiscard]] auto parse_assignment_operator(SourceLocation& loc,
                                               TokenKind& op) -> bool;
  void parse_expression(ExpressionAST*& yyast);
  [[nodiscard]] auto parse_maybe_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_constant_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_template_argument_constant_expression(
      ExpressionAST*& yyast) -> bool;
  void parse_statement(StatementAST*& yyast);
  [[nodiscard]] auto parse_maybe_statement(StatementAST*& yyast) -> bool;
  void parse_init_statement(StatementAST*& yyast);
  void parse_condition(ExpressionAST*& yyast);
  [[nodiscard]] auto parse_labeled_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_case_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_default_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_expression_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_maybe_compound_statement(StatementAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_compound_statement(CompoundStatementAST*& yyast,
                                              bool skip = false) -> bool;
  void finish_compound_statement(CompoundStatementAST* yyast);
  void parse_skip_statement(bool& skipping);
  [[nodiscard]] auto parse_if_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_switch_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_while_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_do_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_for_range_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_for_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_for_range_declaration(DeclarationAST*& yyast)
      -> bool;
  void parse_for_range_initializer(ExpressionAST*& yyast);
  [[nodiscard]] auto parse_break_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_continue_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_return_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_goto_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_coroutine_return_statement(StatementAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_declaration_statement(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_maybe_module() -> bool;
  [[nodiscard]] auto parse_template_declaration_body(
      DeclarationAST*& yyast,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool;
  [[nodiscard]] auto parse_declaration(DeclarationAST*& yyast,
                                       BindingContext ctx) -> bool;
  [[nodiscard]] auto parse_block_declaration(DeclarationAST*& yyast,
                                             BindingContext ctx) -> bool;
  [[nodiscard]] auto parse_alias_declaration(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_alias_declaration(
      DeclarationAST*& yyast,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool;
  [[nodiscard]] auto parse_simple_declaration(DeclarationAST*& yyast,
                                              BindingContext ctx) -> bool;
  [[nodiscard]] auto parse_simple_declaration(
      DeclarationAST*& yyast,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations,
      BindingContext ctx) -> bool;
  [[nodiscard]] auto parse_template_class_declaration(
      DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations,
      BindingContext ctx) -> bool;
  [[nodiscard]] auto parse_empty_or_attribute_declaration(
      DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes) -> auto;

  [[nodiscard]] auto parse_notypespec_function_definition(
      DeclarationAST*& yyast, List<AttributeSpecifierAST*>* atributes,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations,
      BindingContext ctx) -> bool;

  [[nodiscard]] auto parse_notypespec_function_definition(
      DeclarationAST*& yyast, List<SpecifierAST*>* declSpecifierList,
      const DeclSpecs& specs) -> bool;

  [[nodiscard]] auto parse_type_or_forward_declaration(
      DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
      List<SpecifierAST*>* declSpecifierList, const DeclSpecs& specs,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations,
      BindingContext ctx) -> bool;

  [[nodiscard]] auto parse_structured_binding(
      DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
      List<SpecifierAST*>* declSpecifierList, const DeclSpecs& specs,
      BindingContext ctx) -> bool;

  [[nodiscard]] auto parse_simple_declaration(
      DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
      List<SpecifierAST*>* declSpecifierList, const DeclSpecs& specs,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations,
      BindingContext ctx) -> bool;

  [[nodiscard]] auto parse_static_assert_declaration(DeclarationAST*& yyast)
      -> bool;
  auto match_string_literal(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_empty_declaration(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_attribute_declaration(DeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_decl_specifier(SpecifierAST*& yyast,
                                          DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_decl_specifier_seq(List<SpecifierAST*>*& yyast,
                                              DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_decl_specifier_seq_no_typespecs(
      List<SpecifierAST*>*& yyast, DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_decl_specifier_seq_no_typespecs(
      List<SpecifierAST*>*& yyast) -> bool;
  [[nodiscard]] auto parse_storage_class_specifier(SpecifierAST*& yyast,
                                                   DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_function_specifier(SpecifierAST*& yyast,
                                              DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_explicit_specifier(SpecifierAST*& yyast,
                                              DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_type_specifier(SpecifierAST*& yyast,
                                          DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_type_specifier_seq(List<SpecifierAST*>*& yyast,
                                              DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_defining_type_specifier(SpecifierAST*& yyast,
                                                   DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_defining_type_specifier_seq(
      List<SpecifierAST*>*& yyast, DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_simple_type_specifier(SpecifierAST*& yyast,
                                                 DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_named_type_specifier(SpecifierAST*& yyast,
                                                DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_decltype_specifier_type_specifier(
      SpecifierAST*& yyast, DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_underlying_type_specifier(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_atomic_type_specifier(SpecifierAST*& yyast,
                                                 DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_primitive_type_specifier(SpecifierAST*& yyast,
                                                    DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_size_type_specifier(SpecifierAST*& yyast,
                                               DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_sign_type_specifier(SpecifierAST*& yyast,
                                               DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_complex_type_specifier(SpecifierAST*& yyast,
                                                  DeclSpecs& specs) -> bool;

  [[nodiscard]] auto parse_type_name(UnqualifiedIdAST*& yyast) -> bool;
  [[nodiscard]] auto parse_elaborated_type_specifier(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_elaborated_type_specifier_helper(
      ElaboratedTypeSpecifierAST*& yyast, DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_elaborated_enum_specifier(
      ElaboratedTypeSpecifierAST*& yyast, DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_decltype_specifier(DecltypeSpecifierAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_placeholder_type_specifier(SpecifierAST*& yyast,
                                                      DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_init_declarator(InitDeclaratorAST*& yyast,
                                           const DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_init_declarator(InitDeclaratorAST*& yyast,
                                           DeclaratorAST* declarator,
                                           Decl& decl) -> bool;
  [[nodiscard]] auto parse_declarator_initializer(
      RequiresClauseAST*& requiresClause, ExpressionAST*& yyast) -> bool;
  void parse_optional_declarator_or_abstract_declarator(DeclaratorAST*& yyast,
                                                        Decl& decl);

  enum struct DeclaratorKind {
    kDeclarator,
    kAbstractDeclarator,
    kNewDeclarator,
    kDeclaratorOrAbstractDeclarator,
  };

  [[nodiscard]] auto parse_declarator(
      DeclaratorAST*& yyastl, Decl& decl,
      DeclaratorKind declaratorKind = DeclaratorKind::kDeclarator) -> bool;

  [[nodiscard]] auto parse_declarator_id(CoreDeclaratorAST*& yyast, Decl& decl,
                                         DeclaratorKind declaratorKind) -> bool;

  [[nodiscard]] auto parse_nested_declarator(CoreDeclaratorAST*& yyast,
                                             Decl& decl,
                                             DeclaratorKind declaratorKind)
      -> bool;
  void parse_optional_abstract_declarator(DeclaratorAST*& yyastl, Decl& decl);

  [[nodiscard]] auto parse_ptr_operator_seq(List<PtrOperatorAST*>*& yyast)
      -> bool;
  [[nodiscard]] auto parse_core_declarator(CoreDeclaratorAST*& yyast,
                                           Decl& decl,
                                           DeclaratorKind declaratorKind)
      -> bool;
  [[nodiscard]] auto parse_array_declarator(ArrayDeclaratorChunkAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_function_declarator(
      FunctionDeclaratorChunkAST*& yyast, bool acceptTrailingReturnType = true)
      -> bool;
  [[nodiscard]] auto parse_cv_qualifier_seq(List<SpecifierAST*>*& yyast,
                                            DeclSpecs& declSpecs) -> bool;
  [[nodiscard]] auto parse_trailing_return_type(TrailingReturnTypeAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_ptr_operator(PtrOperatorAST*& yyast) -> bool;
  [[nodiscard]] auto parse_cv_qualifier(SpecifierAST*& yyast,
                                        DeclSpecs& declSpecs) -> bool;
  [[nodiscard]] auto parse_ref_qualifier(SourceLocation& refLoc) -> bool;
  [[nodiscard]] auto parse_type_id(TypeIdAST*& yyast) -> bool;
  [[nodiscard]] auto parse_defining_type_id(
      TypeIdAST*& yyast,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool;
  [[nodiscard]] auto parse_parameter_declaration_clause(
      ParameterDeclarationClauseAST*& yyast) -> bool;
  [[nodiscard]] auto parse_parameter_declaration_list(
      List<ParameterDeclarationAST*>*& yyast,
      FunctionParametersSymbol*& functionParametersSymbol) -> bool;
  [[nodiscard]] auto parse_parameter_declaration(
      ParameterDeclarationAST*& yyast, bool templParam) -> bool;
  [[nodiscard]] auto parse_initializer(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_brace_or_equal_initializer(ExpressionAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_initializer_clause(ExpressionAST*& yyast,
                                              bool templParam = false) -> bool;
  [[nodiscard]] auto parse_braced_init_list(BracedInitListAST*& yyast) -> bool;
  [[nodiscard]] auto parse_initializer_list(List<ExpressionAST*>*& yyast)
      -> bool;
  [[nodiscard]] auto parse_designated_initializer_clause(
      DesignatedInitializerClauseAST*& yyast) -> bool;
  void parse_expr_or_braced_init_list(ExpressionAST*& yyast);
  void parse_virt_specifier_seq(FunctionDeclaratorChunkAST* functionDeclarator);
  auto lookat_function_body() -> bool;
  [[nodiscard]] auto parse_function_body(FunctionBodyAST*& yyast) -> bool;
  [[nodiscard]] auto parse_enum_specifier(SpecifierAST*& yyast,
                                          DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_enum_head_name(
      NestedNameSpecifierAST*& nestedNameSpecifier, NameIdAST*& name) -> bool;
  [[nodiscard]] auto parse_opaque_enum_declaration(DeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_enum_key(SourceLocation& enumLoc,
                                    SourceLocation& classLoc) -> bool;
  [[nodiscard]] auto parse_enum_base(SourceLocation& colonLoc,
                                     List<SpecifierAST*>*& typeSpecifierList,
                                     DeclSpecs& specs) -> bool;
  void parse_enumerator_list(List<EnumeratorAST*>*& yyast, const Type* type);
  void parse_enumerator_definition(EnumeratorAST*& yast, const Type* type);
  void parse_enumerator(EnumeratorAST*& yyast, const Type* type);
  [[nodiscard]] auto parse_using_enum_declaration(DeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_namespace_definition(DeclarationAST*& yyast) -> bool;
  void parse_namespace_body(NamespaceDefinitionAST* yyast);
  [[nodiscard]] auto parse_namespace_alias_definition(DeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_qualified_namespace_specifier(
      NestedNameSpecifierAST*& nestedNameSpecifier, NameIdAST*& name) -> bool;
  [[nodiscard]] auto parse_using_directive(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_using_declaration(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_using_declarator_list(
      List<UsingDeclaratorAST*>*& yyast) -> bool;
  [[nodiscard]] auto parse_using_declarator(UsingDeclaratorAST*& yyast) -> bool;
  [[nodiscard]] auto parse_asm_operand(AsmOperandAST*& yyast) -> bool;
  [[nodiscard]] auto parse_asm_declaration(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_linkage_specification(DeclarationAST*& yyast)
      -> bool;
  void parse_optional_attribute_specifier_seq(
      List<AttributeSpecifierAST*>*& yyast, bool allowAsmSpecifier = false);
  [[nodiscard]] auto parse_attribute_specifier_seq(
      List<AttributeSpecifierAST*>*& yyast, bool allowAsmSpecifier = false)
      -> bool;
  [[nodiscard]] auto parse_attribute_specifier(AttributeSpecifierAST*& yyast,
                                               bool allowAsmSpecifier) -> bool;
  auto lookat_cxx_attribute_specifier() -> bool;
  [[nodiscard]] auto parse_cxx_attribute_specifier(
      AttributeSpecifierAST*& yyast) -> bool;
  [[nodiscard]] auto parse_asm_specifier(AttributeSpecifierAST*& yyast) -> bool;
  [[nodiscard]] auto parse_gcc_attribute(AttributeSpecifierAST*& yyast) -> bool;
  [[nodiscard]] auto parse_skip_balanced() -> bool;
  [[nodiscard]] auto parse_alignment_specifier(AttributeSpecifierAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_attribute_using_prefix(
      AttributeUsingPrefixAST*& yyast) -> bool;
  [[nodiscard]] auto parse_attribute_list(List<AttributeAST*>*& yyast) -> bool;
  [[nodiscard]] auto parse_attribute(AttributeAST*& yyast) -> bool;
  [[nodiscard]] auto parse_attribute_token(AttributeTokenAST*& yyast) -> bool;
  [[nodiscard]] auto parse_attribute_scoped_token(AttributeTokenAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_attribute_namespace(
      SourceLocation& attributeNamespaceLoc) -> bool;
  [[nodiscard]] auto parse_attribute_argument_clause(
      AttributeArgumentClauseAST*& yyast) -> bool;
  [[nodiscard]] auto parse_module_declaration(ModuleDeclarationAST*& yyast)
      -> bool;
  void parse_module_name(ModuleNameAST*& yyast);
  [[nodiscard]] auto parse_module_partition(ModulePartitionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_export_declaration(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_maybe_import() -> bool;
  [[nodiscard]] auto parse_module_import_declaration(DeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_import_name(ImportNameAST*& yyast) -> bool;
  void parse_global_module_fragment(GlobalModuleFragmentAST*& yyast);
  void parse_private_module_fragment(PrivateModuleFragmentAST*& yyast);
  [[nodiscard]] auto parse_class_specifier(ClassSpecifierAST*& yyast,
                                           DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_class_specifier(
      ClassSpecifierAST*& yyast, DeclSpecs& specs,
      const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool;
  void parse_class_body(List<DeclarationAST*>*& yyast);
  [[nodiscard]] auto parse_class_head(ClassHead& classHead) -> bool;
  [[nodiscard]] auto parse_class_head_name(
      NestedNameSpecifierAST*& nestedNameSpecifier, UnqualifiedIdAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_class_virt_specifier(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_class_key(SourceLocation& classLoc) -> bool;
  [[nodiscard]] auto parse_member_specification(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_member_declaration(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_bitfield_declarator(InitDeclaratorAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_maybe_template_member() -> bool;
  [[nodiscard]] auto parse_member_declaration_helper(DeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_member_declarator_list(
      List<InitDeclaratorAST*>*& yyast, const DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_member_declarator(InitDeclaratorAST*& yyast,
                                             const DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_virt_specifier(
      FunctionDeclaratorChunkAST* functionDeclarator) -> bool;
  [[nodiscard]] auto parse_pure_specifier(SourceLocation& equalLoc,
                                          SourceLocation& zeroLoc) -> bool;
  [[nodiscard]] auto parse_conversion_function_id(
      ConversionFunctionIdAST*& yyast) -> bool;
  [[nodiscard]] auto parse_base_clause(
      SourceLocation& colonLoc, List<BaseSpecifierAST*>*& baseSpecifierList)
      -> bool;
  [[nodiscard]] auto parse_base_specifier_list(List<BaseSpecifierAST*>*& yyast)
      -> bool;
  void parse_base_specifier(BaseSpecifierAST*& yyast);
  [[nodiscard]] auto parse_class_or_decltype(
      NestedNameSpecifierAST*& nestedNameSpecifier, SourceLocation& templateLoc,
      UnqualifiedIdAST*& unqualifiedId) -> bool;
  [[nodiscard]] auto parse_access_specifier(SourceLocation& loc) -> bool;
  [[nodiscard]] auto parse_ctor_initializer(
      SourceLocation& colonLoc, List<MemInitializerAST*>*& memInitializerList)
      -> bool;
  void parse_mem_initializer_list(List<MemInitializerAST*>*& yyast);
  void parse_mem_initializer(MemInitializerAST*& yyast);
  [[nodiscard]] auto parse_mem_initializer_id(
      NestedNameSpecifierAST*& nestedNameSpecifier, UnqualifiedIdAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_operator_function_id(OperatorFunctionIdAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_operator(TokenKind& op, SourceLocation& opLoc,
                                    SourceLocation& openLoc,
                                    SourceLocation& closeLoc) -> bool;
  [[nodiscard]] auto parse_literal_operator_id(LiteralOperatorIdAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_template_declaration(TemplateDeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_template_declaration(
      TemplateDeclarationAST*& yyast,
      std::vector<TemplateDeclarationAST*>& chain) -> bool;
  void parse_template_parameter_list(List<TemplateParameterAST*>*& yyast);
  [[nodiscard]] auto parse_requires_clause(RequiresClauseAST*& yyast) -> bool;
  [[nodiscard]] auto parse_constraint_logical_or_expression(
      ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_constraint_logical_and_expression(
      ExpressionAST*& yyast) -> bool;
  void parse_template_parameter(TemplateParameterAST*& yyast);
  [[nodiscard]] auto parse_type_parameter(TemplateParameterAST*& yyast) -> bool;
  [[nodiscard]] auto parse_typename_type_parameter(TemplateParameterAST*& yyast)
      -> bool;
  void parse_template_type_parameter(TemplateParameterAST*& yyast);
  [[nodiscard]] auto parse_constraint_type_parameter(
      TemplateParameterAST*& yyast) -> bool;
  [[nodiscard]] auto parse_type_parameter_key(SourceLocation& classKeyLoc)
      -> bool;
  [[nodiscard]] auto parse_type_constraint(TypeConstraintAST*& yyast,
                                           bool parsingPlaceholderTypeSpec)
      -> bool;
  [[nodiscard]] auto parse_simple_template_or_name_id(UnqualifiedIdAST*& yyast,
                                                      bool isTemplateIntroduced)
      -> bool;
  [[nodiscard]] auto parse_simple_template_id(SimpleTemplateIdAST*& yyast,
                                              bool isTemplateIntroduced = false)
      -> bool;
  [[nodiscard]] auto parse_literal_operator_template_id(
      LiteralOperatorTemplateIdAST*& yyast) -> bool;
  [[nodiscard]] auto parse_function_operator_template_id(
      OperatorFunctionTemplateIdAST*& yyast) -> bool;
  [[nodiscard]] auto parse_template_id(UnqualifiedIdAST*& yyast,
                                       bool isTemplateIntroduced) -> bool;
  [[nodiscard]] auto parse_template_argument_list(
      List<TemplateArgumentAST*>*& yyast) -> bool;
  [[nodiscard]] auto parse_template_argument(TemplateArgumentAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_constraint_expression(ExpressionAST*& yyast) -> bool;
  [[nodiscard]] auto parse_deduction_guide(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_concept_definition(DeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_typename_specifier(SpecifierAST*& yyast,
                                              DeclSpecs& specs) -> bool;
  [[nodiscard]] auto parse_explicit_instantiation(DeclarationAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_try_block(StatementAST*& yyast) -> bool;
  [[nodiscard]] auto parse_function_try_block(FunctionBodyAST*& yyast) -> bool;
  [[nodiscard]] auto parse_handler(HandlerAST*& yyast) -> bool;
  [[nodiscard]] auto parse_handler_seq(List<HandlerAST*>*& yyast) -> bool;
  [[nodiscard]] auto parse_exception_declaration(
      ExceptionDeclarationAST*& yyast) -> bool;
  [[nodiscard]] auto parse_noexcept_specifier(ExceptionSpecifierAST*& yyast)
      -> bool;
  [[nodiscard]] auto parse_identifier_list(List<NameIdAST*>*& yyast) -> bool;

 private:
  [[nodiscard]] auto lookat(auto... tokens) {
    return lookatHelper(0, tokens...);
  }

  [[nodiscard]] auto lookatHelper(int) const { return true; }

  [[nodiscard]] auto lookatHelper(int n, TokenKind tk, auto... rest) const {
    return LA(n).is(tk) && lookatHelper(n + 1, rest...);
  }

  [[nodiscard]] auto LA(int n = 0) const -> const Token&;

  auto match(TokenKind tk, SourceLocation& location) -> bool;
  auto expect(TokenKind tk, SourceLocation& location) -> bool;

  auto consumeToken() -> SourceLocation { return SourceLocation(cursor_++); }

  [[nodiscard]] auto currentLocation() const -> SourceLocation {
    return SourceLocation(cursor_);
  }

  void rewind(SourceLocation location) { cursor_ = location.index(); }

  void completePendingFunctionDefinitions();
  void completeFunctionDefinition(FunctionDefinitionAST* ast);

  auto enterOrCreateNamespace(const Name* name, bool isInline)
      -> NamespaceSymbol*;

  void enterFunctionScope(FunctionDeclaratorChunkAST* functionDeclarator);

  void applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs);
  void applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs);

  void check_type_traits();

  // standard conversions
  auto lvalue_to_rvalue_conversion(ExpressionAST*& expr) -> bool;
  auto array_to_pointer_conversion(ExpressionAST*& expr) -> bool;
  auto function_to_pointer_conversion(ExpressionAST*& expr) -> bool;
  auto integral_promotion(ExpressionAST*& expr) -> bool;
  auto floating_point_promotion(ExpressionAST*& expr) -> bool;
  auto integral_conversion(ExpressionAST*& expr, const Type* destinationType)
      -> bool;
  auto floating_point_conversion(ExpressionAST*& expr,
                                 const Type* destinationType) -> bool;
  auto floating_integral_conversion(ExpressionAST*& expr,
                                    const Type* destinationType) -> bool;

  auto is_prvalue(ExpressionAST* expr) const -> bool;
  auto is_lvalue(ExpressionAST* expr) const -> bool;
  auto is_xvalue(ExpressionAST* expr) const -> bool;
  auto is_glvalue(ExpressionAST* expr) const -> bool;

  auto maybe_template_name(const Identifier* id) -> bool;

  void mark_maybe_template_name(const Identifier* id);
  void mark_maybe_template_name(UnqualifiedIdAST* name);
  void mark_maybe_template_name(DeclaratorAST* declarator);

 private:
  TranslationUnit* unit = nullptr;
  Arena* pool_ = nullptr;
  Control* control_ = nullptr;
  DiagnosticsClient* diagnosticClient_ = nullptr;
  Scope* globalScope_ = nullptr;
  Scope* scope_ = nullptr;
  ParserConfiguration config_{};
  bool skipFunctionBody_ = false;
  bool moduleUnit_ = false;
  const Identifier* moduleId_ = nullptr;
  const Identifier* importId_ = nullptr;
  const Identifier* finalId_ = nullptr;
  const Identifier* overrideId_ = nullptr;
  int templArgDepth_ = 0;
  int classDepth_ = 0;
  std::uint32_t lastErrorCursor_ = 0;
  std::uint32_t cursor_ = 0;
  int templateParameterDepth_ = -1;
  int templateParameterCount_ = 0;

  std::vector<FunctionDefinitionAST*> pendingFunctionDefinitions_;

  template <typename T>
  class CachedAST {
   public:
    struct Entry {
      SourceLocation endLoc;
      T* ast = nullptr;
      bool parsed = false;
      int hit = 0;
    };

    using Map = std::unordered_map<SourceLocation, Entry>;
    using const_iterator = typename Map::const_iterator;

    CachedAST(const CachedAST&) = delete;
    auto operator=(const CachedAST&) -> CachedAST& = delete;

    CachedAST() = default;

    auto empty() const -> bool { return this->map_.empty(); }
    auto size() const -> std::size_t { return this->map_.size(); }
    auto begin() const -> const_iterator { return this->map_.begin(); }
    auto end() const -> const_iterator { return this->map_.end(); }

    auto get(SourceLocation loc) -> std::optional<Entry> {
      if (auto it = this->map_.find(loc); it != this->map_.end()) {
        queue_.erase(std::find(queue_.begin(), queue_.end(), it));
        queue_.push_front(it);
        ++it->second.hit;
        return it->second;
      }
      return std::nullopt;
    }

    void set(SourceLocation startLoc, SourceLocation endLoc, T* ast,
             bool parsed) {
      const std::size_t kMaxSize = 100;

      if (this->map_.size() == kMaxSize) {
        auto it = queue_.back();
        queue_.pop_back();
        this->map_.erase(it);
      }

      auto it = this->map_.find(startLoc);

      if (it != this->map_.end()) {
        // ### todo: speed up
        queue_.erase(std::find(queue_.begin(), queue_.end(), it));
      } else {
        it = this->map_.insert({startLoc, {}}).first;
      }

      queue_.push_front(it);

      it->second.endLoc = endLoc;
      it->second.ast = ast;
      it->second.parsed = parsed;
      ++it->second.hit;
    }

   private:
    Map map_;
    std::deque<const_iterator> queue_;
  };

  CachedAST<ClassSpecifierAST> class_specifiers_;
  CachedAST<ElaboratedTypeSpecifierAST> elaborated_type_specifiers_;
  CachedAST<ExpressionAST> cast_expressions_;
  CachedAST<NestedNameSpecifierAST> nested_name_specifiers_;
  CachedAST<ParameterDeclarationClauseAST> parameter_declaration_clauses_;
  CachedAST<TemplateArgumentAST> template_arguments_;

  // TODO: remove
  std::unordered_set<const Identifier*> concept_names_;
  std::unordered_set<const Identifier*> template_names_;
};

}  // namespace cxx
