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
#include <cxx/names.h>
#include <cxx/translation-unit.h>

#include <forward_list>
#include <unordered_map>
#include <variant>

namespace cxx {

struct Parser {
  class DeclarativeRegion;

  struct NamespaceData {
    DeclarativeRegion* region = nullptr;
  };

  struct ClassData {
    DeclarativeRegion* region = nullptr;
    DeclarativeRegion* lexicalRegion = nullptr;
    bool isComplete = false;
  };

  class Symbol {
   public:
    Name name;
    Symbol* next = nullptr;
    std::size_t index = 0;
    std::variant<std::monostate, NamespaceData, ClassData> data;

    size_t hashCode() { return cxx::hashCode(name); }

    bool isNamespace() const {
      return std::holds_alternative<NamespaceData>(data);
    }

    bool isClass() const { return std::holds_alternative<ClassData>(data); }
  };

  class Scope {
    std::vector<Symbol*> symbols_;
    std::vector<Symbol*> buckets_;

   public:
    auto begin() const { return symbols_.begin(); }

    auto end() const { return symbols_.end(); }

    Symbol* find(const Name& name) const {
      if (symbols_.empty()) return nullptr;
      const auto h = hashCode(name) % buckets_.size();
      for (auto sym = buckets_[h]; sym; sym = sym->next) {
        if (sym->name == name) return sym;
      }
      return nullptr;
    }

    void add(Symbol* symbol) {
      symbol->index = symbols_.size();
      symbols_.push_back(symbol);
      if (symbols_.size() < (buckets_.size() * 0.6)) {
        const auto h = symbol->hashCode() % buckets_.size();
        symbol->next = buckets_[h];
        buckets_[h] = symbol;
      } else {
        rehash();
      }
    }

    void rehash() {
      buckets_ =
          std::vector<Symbol*>(buckets_.empty() ? 8 : buckets_.size() * 2);
      for (auto symbol : symbols_) {
        const auto h = symbol->hashCode() % buckets_.size();
        symbol->next = buckets_[h];
        buckets_[h] = symbol;
      }
    }
  };

  class DeclarativeRegion {
   public:
    DeclarativeRegion* enclosing;
    Scope scope;

    explicit DeclarativeRegion(DeclarativeRegion* enclosing)
        : enclosing(enclosing) {}

    void dump(std::ostream& out, int depth = 0) {
      std::string ind(depth * 2, ' ');
      for (auto sym : scope) {
        if (auto ns = std::get_if<NamespaceData>(&sym->data)) {
          fmt::print("{}- namespace {}\n", ind, toString(sym->name));
          ns->region->dump(out, depth + 1);
        } else if (auto udt = std::get_if<ClassData>(&sym->data)) {
          fmt::print("{}- class {}\n", ind, toString(sym->name));
          udt->region->dump(out, depth + 1);
        }
      }
    }

    struct Context {
      Context(const Context&) = delete;
      Context& operator=(const Context&) = delete;

      Parser* p;
      DeclarativeRegion* savedRegion_;

      explicit Context(Parser* p) : p(p), savedRegion_(p->currentRegion_) {}

      ~Context() { p->currentRegion_ = savedRegion_; }

      void enter(DeclarativeRegion* region = nullptr) {
        if (!region) region = p->newDeclarativeRegion(savedRegion_);

        p->currentRegion_ = region;
      }

      void leave() { p->currentRegion_ = savedRegion_; }
    };
  };

  DeclarativeRegion* newDeclarativeRegion(DeclarativeRegion* enclosing) {
    return &regions_.emplace_front(DeclarativeRegion(enclosing));
  }

  Symbol* newSymbol() { return &symbols_.emplace_front(); }

  DeclarativeRegion* currentRegion_ = nullptr;
  std::forward_list<DeclarativeRegion> regions_;
  std::forward_list<Symbol> symbols_;

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

  static Prec prec(TokenKind tk) {
    switch (tk) {
      default:
        std::runtime_error("expected a binary operator");

      case TokenKind::T_DOT_STAR:
      case TokenKind::T_MINUS_GREATER_STAR:
        return Prec::kPm;

      case TokenKind::T_STAR:
      case TokenKind::T_SLASH:
      case TokenKind::T_PERCENT:
        return Prec::kMultiplicative;

      case TokenKind::T_PLUS:
      case TokenKind::T_MINUS:
        return Prec::kAdditive;

      case TokenKind::T_LESS_LESS:
      case TokenKind::T_GREATER_GREATER:
        return Prec::kShift;

      case TokenKind::T_LESS_EQUAL_GREATER:
        return Prec::kCompare;

      case TokenKind::T_LESS_EQUAL:
      case TokenKind::T_GREATER_EQUAL:
      case TokenKind::T_LESS:
      case TokenKind::T_GREATER:
        return Prec::kRelational;

      case TokenKind::T_EQUAL_EQUAL:
      case TokenKind::T_EXCLAIM_EQUAL:
        return Prec::kEquality;

      case TokenKind::T_AMP:
        return Prec::kAnd;

      case TokenKind::T_CARET:
        return Prec::kExclusiveOr;

      case TokenKind::T_BAR:
        return Prec::kInclusiveOr;

      case TokenKind::T_AMP_AMP:
        return Prec::kLogicalAnd;

      case TokenKind::T_BAR_BAR:
        return Prec::kLogicalOr;
    }  // switch
  }

  struct DeclaratorId {};
  struct NestedDeclarator {};
  struct PtrDeclarator {};
  struct FunctionDeclarator {};
  struct ArrayDeclarator {};

  using DeclaratorComponent =
      std::variant<DeclaratorId, NestedDeclarator, PtrDeclarator,
                   FunctionDeclarator, ArrayDeclarator>;

  using Declarator = std::vector<DeclaratorComponent>;

  bool isFunctionDeclarator(const Declarator& decl) const {
    for (auto d : decl) {
      if (std::holds_alternative<NestedDeclarator>(d))
        continue;
      else if (std::holds_alternative<DeclaratorId>(d))
        continue;
      else if (std::holds_alternative<FunctionDeclarator>(d))
        return true;
      else
        return false;
    }
    return false;
  }

  struct DeclSpecs {
    bool has_simple_typespec = false;
    bool has_complex_typespec = false;
    bool has_named_typespec = false;
    bool has_placeholder_typespec = false;
    bool no_typespecs = false;
    bool no_class_or_enum_specs = false;

    bool accepts_simple_typespec() const {
      return !(has_complex_typespec || has_named_typespec ||
               has_placeholder_typespec);
    }

    bool has_typespec() const {
      return has_simple_typespec || has_complex_typespec ||
             has_named_typespec || has_placeholder_typespec;
    }
  };

  struct TemplArgContext {
    TemplArgContext(const TemplArgContext&) = delete;
    TemplArgContext& operator=(const TemplArgContext&) = delete;

    Parser* p;

    explicit TemplArgContext(Parser* p) : p(p) { ++p->templArgDepth; }
    ~TemplArgContext() { --p->templArgDepth; }
  };

  int templArgDepth = 0;
  uint32_t lastErrorCursor = 0;

  template <typename... Args>
  bool parse_warn(const std::string_view& format, const Args&... args) {
    unit->report(yycursor, MessageKind::Warning, format, args...);
    return true;
  }

  template <typename... Args>
  bool parse_error(const std::string_view& format, const Args&... args) {
    if (lastErrorCursor == yycursor) return true;
    lastErrorCursor = yycursor;
    unit->report(yycursor, MessageKind::Error, format, args...);
    // throw std::runtime_error("error");
    return true;
  }

  bool parse_decl_specifier_seq_no_typespecs() {
    DeclSpecs specs;
    return parse_decl_specifier_seq_no_typespecs(specs);
  }

  bool parse_decl_specifier_seq() {
    DeclSpecs specs;
    return parse_decl_specifier_seq(specs);
  }

  bool parse_declarator() {
    Declarator decl;
    return parse_declarator(decl);
  }

  bool yyparse(TranslationUnit* unit, const std::function<void()>& consume);

  const Token& LA(int n = 0) const { return unit->tokenAt(yycursor + n); }

  bool match(TokenKind tk) {
    if (yytoken() != tk) return false;
    yyconsume();
    return true;
  }

  bool expect(TokenKind tk) {
    if (match(tk)) return true;
    parse_error("expected '{}'", Token::spell(tk));
    return false;
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
  bool parse_enter(DeclarativeRegion::Context& context);
  bool parse_leave(DeclarativeRegion::Context& context);
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
  bool parse_nested_name_specifier(DeclarativeRegion*& region);
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
  bool parse_for_range_declaration();
  bool parse_for_range_initializer();
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
  bool parse_expr_or_braced_init_list();
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
  bool parse_enter_named_namespace_definition(
      DeclarativeRegion::Context& context, uint32_t name);
  bool parse_enter_unnamed_namespace_definition(
      DeclarativeRegion::Context& region);
  bool parse_namespace_body();
  bool parse_namespace_alias_definition(DeclarationAST*& yyast);
  bool parse_qualified_namespace_specifier();
  bool parse_using_directive(DeclarationAST*& yyast);
  bool parse_using_declaration(DeclarationAST*& yyast);
  bool parse_using_declarator_list();
  bool parse_using_declarator();
  bool parse_asm_declaration(DeclarationAST*& yyast);
  bool parse_linkage_specification(DeclarationAST*& yyast);
  bool parse_attribute_specifier_seq();
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
  bool parse_enter_class_specifier(DeclarativeRegion::Context& region,
                                   DeclarativeRegion* enclosingRegion,
                                   Name& className, Symbol*& classSymbol);
  bool parse_leave_class_specifier(Symbol* classSymbol, uint32_t start);
  bool parse_reject_class_specifier(uint32_t start);
  bool parse_class_body();
  bool parse_class_head(DeclarativeRegion*& region, Name& name);
  bool parse_class_head_name(DeclarativeRegion*& region, Name& name);
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
  TranslationUnit* unit = nullptr;
  Arena* pool = nullptr;
  Control* control = nullptr;
  bool skip_function_body = false;
  std::unordered_map<uint32_t, std::tuple<uint32_t, bool>> class_specifiers_;
  std::unordered_map<uint32_t, std::tuple<uint32_t, bool>> template_arguments_;
  std::unordered_map<uint32_t, std::tuple<uint32_t, bool, DeclarativeRegion*>>
      nested_name_specifiers_;
  bool module_unit = false;
  const Identifier* module_id = nullptr;
  const Identifier* import_id = nullptr;
  const Identifier* final_id = nullptr;
  const Identifier* override_id = nullptr;
  DeclarativeRegion* globalRegion = nullptr;

  bool yyinvalid = true;
  TokenKind yytok{};
  inline TokenKind yytoken() {
    if (yyinvalid) {
      yytok = yytoken(0);
      yyinvalid = false;
    }
    return yytok;
  }
  inline void yyconsume() {
    ++yycursor;
    yyinvalid = true;
  }
  inline void yyrewind(int i) {
    if (yycursor == i) return;
    yycursor = i;
    yyinvalid = true;
  }
  TokenKind yytoken(int index);

  int yydepth = -1;
  unsigned yycursor = 0;
};

}  // namespace cxx
