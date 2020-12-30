// Copyright (c) 2014-2020 Roberto Raggi <roberto.raggi@gmail.com>
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

#include "parser.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <forward_list>
#include <iostream>
#include <unordered_map>
#include <variant>

#include "arena.h"
#include "control.h"
#include "token.h"

namespace cxx {

bool Parser::isFunctionDeclarator(const Declarator& decl) const {
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

Parser::Prec Parser::prec(TokenKind tk) {
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

struct Parser::DeclSpecs {
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
    return has_simple_typespec || has_complex_typespec || has_named_typespec ||
           has_placeholder_typespec;
  }
};

struct Parser::TemplArgContext {
  TemplArgContext(const TemplArgContext&) = delete;
  TemplArgContext& operator=(const TemplArgContext&) = delete;

  Parser* p;

  explicit TemplArgContext(Parser* p) : p(p) { ++p->templArgDepth; }
  ~TemplArgContext() { --p->templArgDepth; }
};

TokenKind Parser::yytoken(int la) { return unit->tokenKind(yycursor + la); }

const Token& Parser::LA(int la) const { return unit->tokenAt(yycursor + la); }

bool yyparse(TranslationUnit* unit, const std::function<void()>& consume) {
  Parser p;
  return p.yyparse(unit, consume);
}

bool Parser::yyparse(TranslationUnit* u, const std::function<void()>& consume) {
  unit = u;
  control = unit->control();
  yycursor = 1;

  Arena arena;
  pool = &arena;

  module_id = control->getIdentifier("module");
  import_id = control->getIdentifier("import");
  final_id = control->getIdentifier("final");
  override_id = control->getIdentifier("override");

  UnitAST* ast = nullptr;
  auto parsed = parse_translation_unit(ast);

  if (consume) consume();

  return parsed;
}

bool Parser::parse_id(const Identifier* id) {
  if (!match(TokenKind::T_IDENTIFIER)) return false;
  return unit->identifier(yycursor - 1) == id;
}

bool Parser::parse_nospace() {
  const auto& tk = unit->tokenAt(yycursor);
  return !tk.leadingSpace() && !tk.startOfLine();
}

bool Parser::parse_greater_greater() {
  const auto saved = yycursor;
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_GREATER))
    return true;
  yyrewind(saved);
  return false;
}

bool Parser::parse_greater_greater_equal() {
  const auto saved = yycursor;
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_EQUAL))
    return true;
  yyrewind(saved);
  return false;
}

bool Parser::parse_greater_equal() {
  const auto saved = yycursor;
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_EQUAL))
    return true;
  yyrewind(saved);
  return false;
}

bool Parser::parse_header_name() {
  // ### TODO
  return false;
}

bool Parser::parse_export_keyword() {
  if (!module_unit) return false;
  if (!match(TokenKind::T_EXPORT)) return false;
  return true;
}

bool Parser::parse_import_keyword() {
  if (!module_unit) return false;
  if (match(TokenKind::T_IMPORT)) return true;
  if (!parse_id(import_id)) return false;
  unit->setTokenKind(yycursor - 1, TokenKind::T_IMPORT);
  return true;
}

bool Parser::parse_module_keyword() {
  if (!module_unit) return false;

  if (match(TokenKind::T_MODULE)) return true;

  if (!parse_id(module_id)) return false;

  unit->setTokenKind(yycursor - 1, TokenKind::T_MODULE);
  return true;
}

bool Parser::parse_final() { return parse_id(final_id); }

bool Parser::parse_override() { return parse_id(override_id); }

bool Parser::parse_typedef_name() {
  const auto start = yycursor;

  if (parse_simple_template_id()) return true;

  yyrewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_namespace_name() {
  const auto start = yycursor;

  if (parse_namespace_alias()) return true;

  yyrewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_namespace_alias() {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_class_name() {
  Name name;

  if (!parse_class_name(name)) return false;

  return true;
}

bool Parser::parse_class_name(Name& name) {
  const auto start = yycursor;

  if (parse_simple_template_id()) return true;

  yyrewind(start);

  return parse_name_id(name);
}

bool Parser::parse_name_id(Name& name) {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  name = unit->identifier(yycursor - 1);
  return true;
}

bool Parser::parse_enum_name() {
  if (!match(TokenKind::T_IDENTIFIER)) return false;
  return true;
}

bool Parser::parse_template_name() {
  Name name;

  if (!parse_template_name(name)) return false;

  return true;
}

bool Parser::parse_template_name(Name& name) {
  if (!parse_name_id(name)) return false;

  return true;
}

bool Parser::parse_literal() {
  switch (yytoken()) {
    case TokenKind::T_TRUE:
    case TokenKind::T_FALSE:
    case TokenKind::T_NULLPTR:
    case TokenKind::T_INTEGER_LITERAL:
    case TokenKind::T_CHARACTER_LITERAL:
    case TokenKind::T_FLOATING_POINT_LITERAL:
    case TokenKind::T_USER_DEFINED_LITERAL:
    case TokenKind::T_USER_DEFINED_STRING_LITERAL:
      yyconsume();
      return true;

    case TokenKind::T_STRING_LITERAL:
      while (match(TokenKind::T_STRING_LITERAL)) {
        //
      }
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_translation_unit(UnitAST*& yyast) {
  if (parse_module_unit(yyast)) return true;
  parse_top_level_declaration_seq(yyast);
  // globalRegion->dump(std::cout);
  return true;
}

bool Parser::parse_module_head() {
  const auto start = yycursor;
  const auto has_export = match(TokenKind::T_EXPORT);
  const auto is_module = parse_id(module_id);
  yyrewind(start);
  return is_module;
}

bool Parser::parse_module_unit(UnitAST*& yyast) {
  module_unit = true;

  if (!parse_module_head()) return false;

  parse_global_module_fragment();

  if (!parse_module_declaration()) return false;

  parse_declaration_seq();

  parse_private_module_fragment();

  expect(TokenKind::T_EOF_SYMBOL);

  return true;
}

bool Parser::parse_top_level_declaration_seq(UnitAST*& yyast) {
  module_unit = false;

  bool skipping = false;

  DeclarationAST* d1 = nullptr;

  while (yytoken() != TokenKind::T_EOF_SYMBOL) {
    auto saved = yycursor;
    DeclarationAST* declaration = nullptr;
    if (parse_declaration(declaration)) {
      skipping = false;
    } else {
      parse_skip_top_level_declaration(skipping);
      if (yycursor == saved) yyconsume();
    }
  }
  return true;
}

bool Parser::parse_skip_top_level_declaration(bool& skipping) {
  if (yytoken() == TokenKind::T_EOF_SYMBOL) return false;
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
  return true;
}

bool Parser::parse_declaration_seq() {
  bool skipping = false;
  while (yytoken() != TokenKind::T_EOF_SYMBOL) {
    if (yytoken() == TokenKind::T_RBRACE) break;
    if (parse_maybe_module()) break;
    auto saved = yycursor;
    DeclarationAST* decl = nullptr;
    if (parse_declaration(decl)) {
      skipping = false;
    } else {
      parse_skip_declaration(skipping);
      if (yycursor == saved) yyconsume();
    }
  }
  return true;
}

bool Parser::parse_skip_declaration(bool& skipping) {
  if (yytoken() == TokenKind::T_RBRACE) return false;
  if (yytoken() == TokenKind::T_MODULE) return false;
  if (module_unit && yytoken() == TokenKind::T_EXPORT) return false;
  if (yytoken() == TokenKind::T_IMPORT) return false;
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
  return true;
}

bool Parser::parse_primary_expression(ExpressionAST*& yyast) {
  if (match(TokenKind::T_THIS))
    return true;
  else if (parse_literal())
    return true;
  else if (yytoken() == TokenKind::T_LBRACKET)
    return parse_lambda_expression(yyast);
  else if (yytoken() == TokenKind::T_REQUIRES)
    return parse_requires_expression(yyast);
  else if (yytoken() == TokenKind::T_LPAREN) {
    const auto saved = yycursor;

    if (parse_fold_expression(yyast)) return true;

    yyrewind(saved);
    yyconsume();

    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) return false;

    if (!match(TokenKind::T_RPAREN)) return false;

    return true;
  }

  return parse_id_expression();
}

bool Parser::parse_id_expression() {
  const auto start = yycursor;
  if (parse_qualified_id()) return true;
  yyrewind(start);
  return parse_unqualified_id();
}

bool Parser::parse_maybe_template_id() {
  const auto start = yycursor;
  const auto blockErrors = unit->blockErrors(true);
  auto template_id = parse_template_id();
  const auto tk = yytoken();
  unit->blockErrors(blockErrors);
  if (!template_id) return false;
  switch (tk) {
    case TokenKind::T_COMMA:
    case TokenKind::T_GREATER:
    case TokenKind::T_LPAREN:
    case TokenKind::T_RPAREN:
    case TokenKind::T_STAR:
    case TokenKind::T_AMP:
    case TokenKind::T_AMP_AMP:
    case TokenKind::T_BAR_BAR:
    case TokenKind::T_COLON_COLON:
    case TokenKind::T_DOT_DOT_DOT:
    case TokenKind::T_QUESTION:
    case TokenKind::T_LBRACE:
    case TokenKind::T_RBRACE:
    case TokenKind::T_SEMICOLON:
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_unqualified_id() {
  const auto start = yycursor;

  if (parse_maybe_template_id()) return true;

  yyrewind(start);

  if (match(TokenKind::T_TILDE)) {
    if (parse_decltype_specifier()) return true;
    return parse_type_name();
  }

  if (yytoken() == TokenKind::T_OPERATOR) {
    if (parse_operator_function_id()) return true;
    yyrewind(start);
    if (parse_conversion_function_id()) return true;
    yyrewind(start);
    return parse_literal_operator_id();
  }

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_qualified_id() {
  if (!parse_nested_name_specifier()) return false;

  const auto has_template = match(TokenKind::T_TEMPLATE);

  if (parse_unqualified_id()) return true;

  if (has_template) {
    parse_error("expected a template name");
    return true;
  }

  return false;
}

bool Parser::parse_start_of_nested_name_specifier(Name& id) {
  if (match(TokenKind::T_COLON_COLON)) return true;

  if (parse_decltype_specifier() && match(TokenKind::T_COLON_COLON))
    return true;

  const auto start = yycursor;

  if (parse_name_id(id) && match(TokenKind::T_COLON_COLON)) return true;

  yyrewind(start);

  if (parse_simple_template_id(id) && match(TokenKind::T_COLON_COLON))
    return true;

  return false;
}

bool Parser::parse_nested_name_specifier() {
  const auto start = yycursor;

  auto it = nested_name_specifiers_.find(start);

  if (it != nested_name_specifiers_.end()) {
    auto [cursor, parsed] = it->second;
    yyrewind(cursor);
    return parsed;
  }

  struct Context {
    Parser* p;
    uint32_t start;
    bool parsed = false;
    Context(Parser* p) : p(p), start(p->yycursor) {}
    ~Context() {
      p->nested_name_specifiers_.emplace(start,
                                         std::make_tuple(p->yycursor, parsed));
    }
  };

  Context context(this);
  Name id;

  if (!parse_start_of_nested_name_specifier(id)) return false;

  while (true) {
    const auto saved = yycursor;
    if (parse_name_id(id) && match(TokenKind::T_COLON_COLON)) {
      //
    } else {
      yyrewind(saved);
      const auto has_template = match(TokenKind::T_TEMPLATE);
      if (parse_simple_template_id(id) && match(TokenKind::T_COLON_COLON)) {
        //
      } else {
        yyrewind(saved);
        break;
      }
    }
  }
  context.parsed = true;
  return true;
}

bool Parser::parse_lambda_expression(ExpressionAST*& yyast) {
  if (!parse_lambda_introducer()) return false;

  if (match(TokenKind::T_LESS)) {
    if (!parse_template_parameter_list())
      parse_error("expected a template paramter");

    expect(TokenKind::T_GREATER);

    parse_requires_clause();
  }

  if (yytoken() != TokenKind::T_LBRACE) {
    if (!parse_lambda_declarator()) parse_error("expected lambda declarator");
  }

  StatementAST* statement = nullptr;
  if (!parse_compound_statement(statement))
    parse_error("expected a compound statement");

  return true;
}

bool Parser::parse_lambda_introducer() {
  if (!match(TokenKind::T_LBRACKET)) return false;
  if (!match(TokenKind::T_RBRACKET)) {
    if (!parse_lambda_capture()) parse_error("expected a lambda capture");
    expect(TokenKind::T_RBRACKET);
  }
  return true;
}

bool Parser::parse_lambda_declarator() {
  if (!match(TokenKind::T_LPAREN)) return false;
  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_parameter_declaration_clause())
      parse_error("expected a parameter declaration clause");
    expect(TokenKind::T_RPAREN);
  }
  parse_decl_specifier_seq();
  parse_noexcept_specifier();
  parse_attribute_specifier_seq();
  parse_trailing_return_type();
  parse_requires_clause();
  return true;
}

bool Parser::parse_lambda_capture() {
  if (parse_capture_default()) {
    if (match(TokenKind::T_COMMA)) {
      if (!parse_capture_list()) parse_error("expected a capture");
    }
    return true;
  }

  return parse_capture_list();
}

bool Parser::parse_capture_default() {
  const auto start = yycursor;

  if (!match(TokenKind::T_AMP) && !match(TokenKind::T_EQUAL)) return false;

  if (yytoken() != TokenKind::T_COMMA && yytoken() != TokenKind::T_RBRACKET) {
    yyrewind(start);
    return false;
  }

  return true;
}

bool Parser::parse_capture_list() {
  if (!parse_capture()) return false;

  while (match(TokenKind::T_COMMA)) {
    if (!parse_capture()) parse_error("expected a capture");
  }

  return true;
}

bool Parser::parse_capture() {
  const auto start = yycursor;

  if (parse_init_capture()) return true;

  yyrewind(start);

  return parse_simple_capture();
}

bool Parser::parse_simple_capture() {
  if (match(TokenKind::T_THIS)) return true;

  if (match(TokenKind::T_IDENTIFIER)) {
    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);
    return true;
  }

  if (match(TokenKind::T_STAR)) {
    if (!match(TokenKind::T_THIS)) return false;
    return true;
  }

  if (!match(TokenKind::T_AMP)) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);
  return true;
}

bool Parser::parse_init_capture() {
  if (match(TokenKind::T_AMP)) {
    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);
    if (!match(TokenKind::T_IDENTIFIER)) return false;
    if (!parse_initializer()) return false;
    return true;
  }

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  if (!parse_initializer()) return false;

  return true;
}

bool Parser::parse_fold_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (match(TokenKind::T_DOT_DOT_DOT)) {
    if (!parse_fold_operator()) parse_error("expected fold operator");

    ExpressionAST* expression;

    if (!parse_cast_expression(expression))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);

    return true;
  }

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) return false;

  if (!parse_fold_operator()) return false;

  if (!match(TokenKind::T_DOT_DOT_DOT)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_fold_operator()) parse_error("expected a fold operator");

    ExpressionAST* rhs = nullptr;

    if (!parse_cast_expression(rhs)) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_fold_operator() {
  switch (yytoken()) {
    case TokenKind::T_GREATER:
      if (parse_greater_greater_equal()) return true;
      if (parse_greater_greater()) return true;
      if (parse_greater_equal()) return true;
      yyconsume();
      return true;

    case TokenKind::T_GREATER_GREATER_EQUAL:
    case TokenKind::T_GREATER_GREATER:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
    case TokenKind::T_STAR:
    case TokenKind::T_SLASH:
    case TokenKind::T_PERCENT:
    case TokenKind::T_CARET:
    case TokenKind::T_AMP:
    case TokenKind::T_BAR:
    case TokenKind::T_LESS_LESS:
    case TokenKind::T_PLUS_EQUAL:
    case TokenKind::T_MINUS_EQUAL:
    case TokenKind::T_STAR_EQUAL:
    case TokenKind::T_SLASH_EQUAL:
    case TokenKind::T_PERCENT_EQUAL:
    case TokenKind::T_CARET_EQUAL:
    case TokenKind::T_AMP_EQUAL:
    case TokenKind::T_BAR_EQUAL:
    case TokenKind::T_LESS_LESS_EQUAL:
    case TokenKind::T_EQUAL:
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_AMP_AMP:
    case TokenKind::T_BAR_BAR:
    case TokenKind::T_COMMA:
    case TokenKind::T_DOT_STAR:
    case TokenKind::T_MINUS_GREATER_STAR:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_requires_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_REQUIRES)) return false;

  if (yytoken() != TokenKind::T_LBRACE) {
    if (!parse_requirement_parameter_list())
      parse_error("expected a requirement parameter");
  }

  if (!parse_requirement_body()) parse_error("expected a requirement body");

  return true;
}

bool Parser::parse_requirement_parameter_list() {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_parameter_declaration_clause())
      parse_error("expected a parmater declaration");
    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_requirement_body() {
  if (!match(TokenKind::T_LBRACE)) return false;

  if (!parse_requirement_seq()) parse_error("expected a requirement");

  expect(TokenKind::T_RBRACE);

  return true;
}

bool Parser::parse_requirement_seq() {
  bool skipping = false;

  if (!parse_requirement()) return false;

  while (yytoken() != TokenKind::T_EOF_SYMBOL) {
    if (yytoken() == TokenKind::T_RBRACE) break;

    const auto before_requirement = yycursor;

    if (parse_requirement()) {
      skipping = false;
    } else {
      if (!skipping) parse_error("expected a requirement");
      skipping = true;
      if (yycursor == before_requirement) yyconsume();
    }
  }

  return true;
}

bool Parser::parse_requirement() {
  if (parse_nested_requirement()) return true;

  if (parse_compound_requirement()) return true;

  if (parse_type_requirement()) return true;

  return parse_simple_requirement();
}

bool Parser::parse_simple_requirement() {
  ExpressionAST* expression = nullptr;
  if (!parse_expression(expression)) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_type_requirement() {
  if (!match(TokenKind::T_TYPENAME)) return false;

  const auto after_typename = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(after_typename);

  if (!parse_type_name()) parse_error("expected a type name");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_compound_requirement() {
  if (!match(TokenKind::T_LBRACE)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) return false;

  if (!match(TokenKind::T_RBRACE)) return false;

  const auto has_noexcept = match(TokenKind::T_NOEXCEPT);

  if (!match(TokenKind::T_SEMICOLON)) {
    if (!parse_return_type_requirement())
      parse_error("expected return type requirement");

    expect(TokenKind::T_SEMICOLON);
  }

  return true;
}

bool Parser::parse_return_type_requirement() {
  if (!match(TokenKind::T_MINUS_GREATER)) return false;

  if (!parse_type_constraint()) parse_error("expected type constraint");

  return true;
}

bool Parser::parse_nested_requirement() {
  if (!match(TokenKind::T_REQUIRES)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_constraint_expression(expression))
    parse_error("expected an expression");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_postfix_expression(ExpressionAST*& yyast) {
  ExpressionAST* e1 = nullptr;

  if (!parse_start_of_postfix_expression(yyast)) return false;

  while (true) {
    const auto saved = yycursor;
    if (parse_member_expression(yyast))
      continue;
    else if (parse_subscript_expression(yyast))
      continue;
    else if (parse_call_expression(yyast))
      continue;
    else if (parse_postincr_expression(yyast))
      continue;
    else {
      yyrewind(saved);
      break;
    }
  }

  return true;
}

bool Parser::parse_start_of_postfix_expression(ExpressionAST*& yyast) {
  const auto start = yycursor;

  if (parse_cpp_cast_expression(yyast)) return true;

  if (parse_typeid_expression(yyast)) return true;

  if (parse_builtin_call_expression(yyast)) return true;

  if (parse_typename_expression(yyast)) return true;

  if (parse_cpp_type_cast_expression(yyast)) return true;

  yyrewind(start);
  return parse_primary_expression(yyast);
}

bool Parser::parse_member_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_DOT) && !match(TokenKind::T_MINUS_GREATER))
    return false;

  const auto has_template = match(TokenKind::T_TEMPLATE);

  if (!parse_id_expression()) parse_error("expected a member name");

  return true;
}

bool Parser::parse_subscript_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_LBRACKET)) return false;

  if (!parse_expr_or_braced_init_list()) parse_error("expected an expression");

  expect(TokenKind::T_RBRACKET);

  return true;
}

bool Parser::parse_call_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_expression_list()) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_postincr_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_MINUS_MINUS) && !match(TokenKind::T_PLUS_PLUS))
    return false;

  return true;
}

bool Parser::parse_cpp_cast_head() {
  switch (yytoken()) {
    case TokenKind::T_CONST_CAST:
    case TokenKind::T_DYNAMIC_CAST:
    case TokenKind::T_REINTERPRET_CAST:
    case TokenKind::T_STATIC_CAST:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_cpp_cast_expression(ExpressionAST*& yyast) {
  if (!parse_cpp_cast_head()) return false;

  expect(TokenKind::T_LESS);

  if (!parse_type_id()) parse_error("expected a type id");

  expect(TokenKind::T_GREATER);
  expect(TokenKind::T_LPAREN);

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_cpp_type_cast_expression(ExpressionAST*& yyast) {
  DeclSpecs specs;

  if (!parse_simple_type_specifier(specs)) return false;

  if (parse_braced_init_list()) return true;

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_expression_list()) return false;

    if (!match(TokenKind::T_RPAREN)) return false;
  }

  return true;
}

bool Parser::parse_typeid_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_TYPEID)) return false;

  expect(TokenKind::T_LPAREN);

  const auto saved = yycursor;

  if (parse_type_id() && match(TokenKind::T_RPAREN)) {
    //
  } else {
    yyrewind(saved);

    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_typename_expression(ExpressionAST*& yyast) {
  if (!parse_typename_specifier()) return false;

  if (parse_braced_init_list()) return true;

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_expression_list()) return false;

    if (!match(TokenKind::T_RPAREN)) return false;
  }

  return true;
}

bool Parser::parse_builtin_function_1() {
  switch (yytoken()) {
    case TokenKind::T___HAS_UNIQUE_OBJECT_REPRESENTATIONS:
    case TokenKind::T___HAS_VIRTUAL_DESTRUCTOR:
    case TokenKind::T___IS_ABSTRACT:
    case TokenKind::T___IS_AGGREGATE:
    case TokenKind::T___IS_CLASS:
    case TokenKind::T___IS_EMPTY:
    case TokenKind::T___IS_ENUM:
    case TokenKind::T___IS_FINAL:
    case TokenKind::T___IS_FUNCTION:
    case TokenKind::T___IS_POD:
    case TokenKind::T___IS_POLYMORPHIC:
    case TokenKind::T___IS_STANDARD_LAYOUT:
    case TokenKind::T___IS_TRIVIAL:
    case TokenKind::T___IS_TRIVIALLY_COPYABLE:
    case TokenKind::T___IS_TRIVIALLY_DESTRUCTIBLE:
    case TokenKind::T___IS_UNION:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_builtin_function_2() {
  switch (yytoken()) {
    case TokenKind::T___IS_BASE_OF:
    case TokenKind::T___IS_CONSTRUCTIBLE:
    case TokenKind::T___IS_CONVERTIBLE_TO:
    case TokenKind::T___IS_NOTHROW_ASSIGNABLE:
    case TokenKind::T___IS_NOTHROW_CONSTRUCTIBLE:
    case TokenKind::T___IS_SAME:
    case TokenKind::T___IS_TRIVIALLY_ASSIGNABLE:
    case TokenKind::T___IS_TRIVIALLY_CONSTRUCTIBLE:
    case TokenKind::T___REFERENCE_BINDS_TO_TEMPORARY:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_builtin_call_expression(ExpressionAST*& yyast) {
  if (parse_builtin_function_1()) {
    expect(TokenKind::T_LPAREN);
    if (!parse_type_id()) parse_error("expected a type id");
    expect(TokenKind::T_RPAREN);
    return true;
  }

  if (!parse_builtin_function_2()) return false;

  expect(TokenKind::T_LPAREN);
  if (!parse_type_id()) parse_error("expected a type id");
  expect(TokenKind::T_COMMA);
  if (!parse_type_id()) parse_error("expected a type id");
  expect(TokenKind::T_RPAREN);
  return true;
}

bool Parser::parse_expression_list() { return parse_initializer_list(); }

bool Parser::parse_unary_expression(ExpressionAST*& yyast) {
  if (parse_unop_expression(yyast)) return true;

  if (parse_complex_expression(yyast)) return true;

  if (parse_await_expression(yyast)) return true;

  if (parse_sizeof_expression(yyast)) return true;

  if (parse_alignof_expression(yyast)) return true;

  if (parse_noexcept_expression(yyast)) return true;

  if (parse_new_expression(yyast)) return true;

  if (parse_delete_expression(yyast)) return true;

  return parse_postfix_expression(yyast);
}

bool Parser::parse_unop_expression(ExpressionAST*& yyast) {
  if (!parse_unary_operator() && !match(TokenKind::T_PLUS_PLUS) &&
      !match(TokenKind::T_MINUS_MINUS))
    return false;

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) parse_error("expected an expression");

  return true;
}

bool Parser::parse_complex_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T___IMAG__) && !match(TokenKind::T___REAL__))
    return false;

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) parse_error("expected an expression");

  return true;
}

bool Parser::parse_sizeof_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_SIZEOF)) return false;

  if (match(TokenKind::T_DOT_DOT_DOT)) {
    expect(TokenKind::T_LPAREN);
    expect(TokenKind::T_IDENTIFIER);
    expect(TokenKind::T_RPAREN);
    return true;
  }

  const auto after_sizeof_op = yycursor;

  if (match(TokenKind::T_LPAREN)) {
    if (parse_type_id() && match(TokenKind::T_RPAREN)) {
      return true;
    }
    yyrewind(after_sizeof_op);
  }

  ExpressionAST* expression = nullptr;

  if (!parse_unary_expression(expression))
    parse_error("expected an expression");

  return true;
}

bool Parser::parse_alignof_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_ALIGNOF) && !match(TokenKind::T___ALIGNOF))
    return false;

  expect(TokenKind::T_LPAREN);

  if (!parse_type_id()) parse_error("expected a type id");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_unary_operator() {
  switch (yytoken()) {
    case TokenKind::T_STAR:
    case TokenKind::T_AMP:
    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
    case TokenKind::T_EXCLAIM:
    case TokenKind::T_TILDE:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_await_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_CO_AWAIT)) return false;

  expect(TokenKind::T_LPAREN);

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_noexcept_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_NOEXCEPT)) return false;

  expect(TokenKind::T_LPAREN);

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_new_expression(ExpressionAST*& yyast) {
  const auto start = yycursor;

  const auto has_scope_op = match(TokenKind::T_COLON_COLON);

  if (!match(TokenKind::T_NEW)) {
    yyrewind(start);
    return false;
  }

  const auto after_new_op = yycursor;

  if (!parse_new_placement()) yyrewind(after_new_op);

  const auto after_new_placement = yycursor;

  if (match(TokenKind::T_LPAREN) && parse_type_id() &&
      match(TokenKind::T_RPAREN)) {
    const auto saved = yycursor;
    if (!parse_new_initializer()) yyrewind(saved);
    return true;
  }

  yyrewind(after_new_placement);

  if (!parse_new_type_id()) return false;

  const auto saved = yycursor;

  if (!parse_new_initializer()) yyrewind(saved);

  return true;
}

bool Parser::parse_new_placement() {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (!parse_expression_list()) return false;

  if (!match(TokenKind::T_RPAREN)) return false;

  return true;
}

bool Parser::parse_new_type_id() {
  if (!parse_type_specifier_seq()) return false;

  const auto saved = yycursor;

  if (!parse_new_declarator()) yyrewind(saved);

  return true;
}

bool Parser::parse_new_declarator() {
  if (parse_ptr_operator()) {
    auto saved = yycursor;

    if (!parse_new_declarator()) yyrewind(saved);

    return true;
  }

  return parse_noptr_new_declarator();
}

bool Parser::parse_noptr_new_declarator() {
  if (!match(TokenKind::T_LBRACKET)) return false;

  if (!match(TokenKind::T_RBRACKET)) {
    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) parse_error("expected an expression");

    expect(TokenKind::T_RBRACKET);
  }

  parse_attribute_specifier_seq();

  while (match(TokenKind::T_LBRACKET)) {
    if (!match(TokenKind::T_RBRACKET)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(expression))
        parse_error("expected an expression");

      expect(TokenKind::T_RBRACKET);
    }

    parse_attribute_specifier_seq();
  }

  return true;
}

bool Parser::parse_new_initializer() {
  if (yytoken() == TokenKind::T_LBRACE) return parse_braced_init_list();

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_expression_list()) return false;

    if (!match(TokenKind::T_RPAREN)) return false;
  }

  return true;
}

bool Parser::parse_delete_expression(ExpressionAST*& yyast) {
  const auto start = yycursor;

  const auto has_scope_op = match(TokenKind::T_COLON_COLON);

  if (!match(TokenKind::T_DELETE)) {
    yyrewind(start);
    return false;
  }

  if (match(TokenKind::T_LBRACKET)) {
    expect(TokenKind::T_RBRACKET);
  }

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) parse_error("expected an expression");

  return true;
}

bool Parser::parse_cast_expression(ExpressionAST*& yyast) {
  const auto start = yycursor;

  if (parse_cast_expression_helper(yyast)) return true;

  yyrewind(start);

  return parse_unary_expression(yyast);
}

bool Parser::parse_cast_expression_helper(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (!parse_type_id()) return false;

  if (!match(TokenKind::T_RPAREN)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) return false;

  return true;
}

bool Parser::parse_binary_operator(TokenKind& tk, bool templArg) {
  const auto start = yycursor;

  tk = TokenKind::T_EOF_SYMBOL;

  switch (yytoken()) {
    case TokenKind::T_GREATER: {
      if (parse_greater_greater()) {
        if (templArg && templArgDepth >= 2) {
          yyrewind(start);
          return false;
        }

        tk = TokenKind::T_GREATER_GREATER;
        return true;
      }

      if (parse_greater_equal()) {
        tk = TokenKind::T_GREATER_EQUAL;
        return true;
      }

      if (templArg) {
        yyrewind(start);
        return false;
      }

      yyconsume();
      tk = TokenKind::T_GREATER;
      return true;
    }

    case TokenKind::T_STAR:
    case TokenKind::T_SLASH:
    case TokenKind::T_PLUS:
    case TokenKind::T_PERCENT:
    case TokenKind::T_MINUS_GREATER_STAR:
    case TokenKind::T_MINUS:
    case TokenKind::T_LESS_LESS:
    case TokenKind::T_LESS_EQUAL_GREATER:
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_EXCLAIM_EQUAL:
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_DOT_STAR:
    case TokenKind::T_CARET:
    case TokenKind::T_BAR_BAR:
    case TokenKind::T_BAR:
    case TokenKind::T_AMP_AMP:
    case TokenKind::T_AMP:
      tk = unit->tokenKind(yycursor);
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_binary_expression(ExpressionAST*& yyast, bool templArg) {
  if (!parse_cast_expression(yyast)) return false;

  const auto saved = yycursor;

  if (!parse_binary_expression_helper(yyast, Prec::kLogicalOr, templArg))
    yyrewind(saved);

  return true;
}

bool Parser::parse_lookahead_binary_operator(TokenKind& tk, bool templArg) {
  const auto saved = yycursor;

  const auto has_binop = parse_binary_operator(tk, templArg);

  yyrewind(saved);

  return has_binop;
}

bool Parser::parse_binary_expression_helper(ExpressionAST*& yyast, Prec minPrec,
                                            bool templArg) {
  bool parsed = false;

  TokenKind op = TokenKind::T_EOF_SYMBOL;

  while (parse_lookahead_binary_operator(op, templArg) && prec(op) >= minPrec) {
    const auto saved = yycursor;

    ExpressionAST* rhs = nullptr;

    parse_binary_operator(op, templArg);

    if (!parse_cast_expression(rhs)) {
      yyrewind(saved);
      break;
    }

    parsed = true;

    TokenKind nextOp = TokenKind::T_EOF_SYMBOL;

    while (parse_lookahead_binary_operator(nextOp, templArg) &&
           prec(nextOp) > prec(op)) {
      if (!parse_binary_expression_helper(rhs, prec(op), templArg)) {
        break;
      }
    }
  }

  return parsed;
}

bool Parser::parse_logical_or_expression(ExpressionAST*& yyast, bool templArg) {
  return parse_binary_expression(yyast, templArg);
}

bool Parser::parse_conditional_expression(ExpressionAST*& yyast,
                                          bool templArg) {
  if (!parse_logical_or_expression(yyast, templArg)) return false;

  if (match(TokenKind::T_QUESTION)) {
    ExpressionAST* iftrue_expression = nullptr;

    if (!parse_expression(iftrue_expression))
      parse_error("expected an expression");

    expect(TokenKind::T_COLON);

    ExpressionAST* iffalse_expression = nullptr;

    if (templArg) {
      if (!parse_conditional_expression(iffalse_expression, templArg)) {
        parse_error("expected an expression");
      }
    } else if (!parse_assignment_expression(iffalse_expression)) {
      parse_error("expected an expression");
    }
  }

  return true;
}

bool Parser::parse_yield_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_CO_YIELD)) return false;

  if (yytoken() == TokenKind::T_LBRACE) {
    if (!parse_braced_init_list()) parse_error("expected a braced initializer");
  } else {
    ExpressionAST* expression = nullptr;

    if (!parse_assignment_expression(expression))
      parse_error("expected an expression");
  }

  return true;
}

bool Parser::parse_throw_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_THROW)) return false;

  ExpressionAST* expression = nullptr;

  const auto saved = yycursor;

  if (!parse_assignment_expression(expression)) yyrewind(saved);

  return true;
}

bool Parser::parse_assignment_expression(ExpressionAST*& yyast) {
  if (parse_yield_expression(yyast)) return true;

  if (parse_throw_expression(yyast)) return true;

  if (!parse_conditional_expression(yyast, false)) return false;

  if (parse_assignment_operator()) {
    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression))
      parse_error("expected an expression");
  }

  return true;
}

bool Parser::parse_assignment_operator() {
  switch (yytoken()) {
    case TokenKind::T_EQUAL:
    case TokenKind::T_STAR_EQUAL:
    case TokenKind::T_SLASH_EQUAL:
    case TokenKind::T_PERCENT_EQUAL:
    case TokenKind::T_PLUS_EQUAL:
    case TokenKind::T_MINUS_EQUAL:
    case TokenKind::T_LESS_LESS_EQUAL:
    case TokenKind::T_AMP_EQUAL:
    case TokenKind::T_CARET_EQUAL:
    case TokenKind::T_BAR_EQUAL:
    case TokenKind::T_GREATER_GREATER_EQUAL:
      yyconsume();
      return true;

    case TokenKind::T_GREATER:
      return parse_greater_greater_equal();

    default:
      return false;
  }  // switch
}

bool Parser::parse_expression(ExpressionAST*& yyast) {
  if (!parse_assignment_expression(yyast)) return false;

  while (match(TokenKind::T_COMMA)) {
    ExpressionAST* expression = nullptr;

    if (!parse_assignment_expression(expression))
      parse_error("expected an expression");
  }

  return true;
}

bool Parser::parse_constant_expression(ExpressionAST*& yyast) {
  return parse_conditional_expression(yyast, false);
}

bool Parser::parse_template_argument_constant_expression(
    ExpressionAST*& yyast) {
  return parse_conditional_expression(yyast, true);
}

bool Parser::parse_statement(StatementAST*& yyast) {
  bool has_extension = false;

  if (match(TokenKind::T___EXTENSION__)) {
    has_extension = false;
  }

  bool has_attribute_specifiers = false;

  if (parse_attribute_specifier_seq()) {
    has_attribute_specifiers = true;
  }

  const auto start = yycursor;

  switch (yytoken()) {
    case TokenKind::T_CASE:
      return parse_case_statement(yyast);
    case TokenKind::T_DEFAULT:
      return parse_default_statement(yyast);
    case TokenKind::T_WHILE:
      return parse_while_statement(yyast);
    case TokenKind::T_DO:
      return parse_do_statement(yyast);
    case TokenKind::T_FOR:
      if (parse_for_range_statement(yyast)) return true;
      yyrewind(start);
      return parse_for_statement(yyast);
    case TokenKind::T_IF:
      return parse_if_statement(yyast);
    case TokenKind::T_SWITCH:
      return parse_switch_statement(yyast);
    case TokenKind::T_BREAK:
      return parse_break_statement(yyast);
    case TokenKind::T_CONTINUE:
      return parse_continue_statement(yyast);
    case TokenKind::T_RETURN:
      return parse_return_statement(yyast);
    case TokenKind::T_GOTO:
      return parse_goto_statement(yyast);
    case TokenKind::T_CO_RETURN:
      return parse_coroutine_return_statement(yyast);
    case TokenKind::T_TRY:
      return parse_try_block(yyast);
    case TokenKind::T_LBRACE:
      return parse_compound_statement(yyast);
    default:
      if (yytoken() == TokenKind::T_IDENTIFIER &&
          yytoken(1) == TokenKind::T_COLON) {
        return parse_labeled_statement(yyast);
      } else if (parse_declaration_statement(yyast)) {
        return true;
      }
      yyrewind(start);
      return parse_expression_statement(yyast);
  }  // switch
}

bool Parser::parse_init_statement(StatementAST*& yyast) {
  if (yytoken() == TokenKind::T_RPAREN) return false;

  auto saved = yycursor;

  DeclarationAST* declaration = nullptr;

  if (parse_simple_declaration(declaration, false)) return true;

  yyrewind(saved);

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_condition(ExpressionAST*& yyast) {
  const auto start = yycursor;

  parse_attribute_specifier_seq();

  if (parse_decl_specifier_seq()) {
    if (parse_declarator()) {
      if (parse_brace_or_equal_initializer()) return true;
    }
  }

  yyrewind(start);

  return parse_expression(yyast);
}

bool Parser::parse_labeled_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  expect(TokenKind::T_COLON);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_case_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_CASE)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_constant_expression(expression))
    parse_error("expected an expression");

  expect(TokenKind::T_COLON);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_default_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_DEFAULT)) return false;

  expect(TokenKind::T_COLON);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_expression_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_SEMICOLON)) {
    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) return false;

    expect(TokenKind::T_SEMICOLON);
  }

  return true;
}

bool Parser::parse_compound_statement(StatementAST*& yyast) {
  bool skipping = false;

  if (!match(TokenKind::T_LBRACE)) return false;

  while (auto tk = LA()) {
    if (yytoken() == TokenKind::T_RBRACE) break;

    StatementAST* s = nullptr;

    if (parse_statement(s)) {
      skipping = false;
    } else {
      parse_skip_statement(skipping);
    }
  }

  if (!expect(TokenKind::T_RBRACE)) return false;

  return true;
}

bool Parser::parse_skip_statement(bool& skipping) {
  if (yytoken() == TokenKind::T_EOF_SYMBOL) return false;
  if (yytoken() == TokenKind::T_RBRACE) return false;
  if (!skipping) parse_error("expected a statement");
  for (; yytoken() != TokenKind::T_EOF_SYMBOL; yyconsume()) {
    if (yytoken() == TokenKind::T_SEMICOLON) break;
    if (yytoken() == TokenKind::T_LBRACE) break;
    if (yytoken() == TokenKind::T_RBRACE) break;
  }
  skipping = true;
  return true;
}

bool Parser::parse_if_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_IF)) return false;

  const auto has_constexpr = match(TokenKind::T_CONSTEXPR);

  expect(TokenKind::T_LPAREN);

  auto saved = yycursor;

  StatementAST* initializer = nullptr;

  if (!parse_init_statement(initializer)) yyrewind(saved);

  ExpressionAST* condition = nullptr;

  if (!parse_condition(condition)) parse_error("expected a condition");

  expect(TokenKind::T_RPAREN);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  if (match(TokenKind::T_ELSE)) {
    StatementAST* else_statement = nullptr;

    if (!parse_statement(else_statement)) parse_error("expected a statement");
  }

  return true;
}

bool Parser::parse_switch_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_SWITCH)) return false;

  expect(TokenKind::T_LPAREN);

  StatementAST* initializer = nullptr;

  const auto saved = yycursor;

  if (!parse_init_statement(initializer)) yyrewind(saved);

  ExpressionAST* condition = nullptr;

  if (!parse_condition(condition)) parse_error("expected a condition");

  expect(TokenKind::T_RPAREN);

  StatementAST* statement = nullptr;

  parse_statement(statement);

  return true;
}

bool Parser::parse_while_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_WHILE)) return false;

  expect(TokenKind::T_LPAREN);

  ExpressionAST* condition = nullptr;

  if (!parse_condition(condition)) parse_error("expected a condition");

  expect(TokenKind::T_RPAREN);
  StatementAST* statement = nullptr;
  if (!parse_statement(statement)) parse_error("expected a statement");
  return true;
}

bool Parser::parse_do_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_DO)) return false;

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  expect(TokenKind::T_WHILE);

  expect(TokenKind::T_LPAREN);

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_for_range_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_FOR)) return false;

  if (!match(TokenKind::T_LPAREN)) return false;

  const auto saved = yycursor;

  StatementAST* initializer = nullptr;

  if (!parse_init_statement(initializer)) yyrewind(saved);

  if (!parse_for_range_declaration()) return false;

  if (!match(TokenKind::T_COLON)) return false;

  if (!parse_for_range_initializer())
    parse_error("expected for-range intializer");

  expect(TokenKind::T_RPAREN);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_for_statement(StatementAST*& yyast) {
  StatementAST* s1 = nullptr;
  StatementAST* s2 = nullptr;
  ExpressionAST* e1 = nullptr;
  ExpressionAST* e2 = nullptr;

  if (!match(TokenKind::T_FOR)) return false;

  expect(TokenKind::T_LPAREN);

  if (!parse_init_statement(s1)) parse_error("expected a statement");

  if (!match(TokenKind::T_SEMICOLON)) {
    if (!parse_condition(e1)) parse_error("expected a condition");

    expect(TokenKind::T_SEMICOLON);
  }

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_expression(e2)) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);
  }

  if (!parse_statement(s2)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_for_range_declaration() {
  parse_attribute_specifier_seq();

  if (!parse_decl_specifier_seq()) return false;

  auto tk = yytoken();

  if (tk == TokenKind::T_AMP || tk == TokenKind::T_AMP_AMP ||
      tk == TokenKind::T_LBRACKET) {
    parse_ref_qualifier();

    if (!match(TokenKind::T_LBRACKET)) return false;

    if (!parse_identifier_list()) parse_error("expected an identifier");

    expect(TokenKind::T_RBRACKET);
  } else {
    return parse_declarator();
  }

  return true;
}

bool Parser::parse_for_range_initializer() {
  return parse_expr_or_braced_init_list();
}

bool Parser::parse_break_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_BREAK)) return false;

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_continue_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_CONTINUE)) return false;

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_return_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_RETURN)) return false;

  if (!match(TokenKind::T_SEMICOLON)) {
    if (!parse_expr_or_braced_init_list())
      parse_error("expected an expression or ';'");

    expect(TokenKind::T_SEMICOLON);
  }

  return true;
}

bool Parser::parse_goto_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_GOTO)) return false;

  expect(TokenKind::T_IDENTIFIER);

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_coroutine_return_statement(StatementAST*& yyast) {
  if (!match(TokenKind::T_CO_RETURN)) return false;

  if (!match(TokenKind::T_SEMICOLON)) {
    if (!parse_expr_or_braced_init_list())
      parse_error("expected an expression");

    expect(TokenKind::T_SEMICOLON);
  }

  return true;
}

bool Parser::parse_declaration_statement(StatementAST*& yyast) {
  DeclarationAST* declaration = nullptr;

  if (!parse_block_declaration(declaration, false)) return false;

  return true;
}

bool Parser::parse_maybe_module() {
  if (!module_unit) return false;

  const auto start = yycursor;

  match(TokenKind::T_EXPORT);

  const auto is_module = parse_module_keyword();

  yyrewind(start);

  return is_module;
}

bool Parser::parse_declaration(DeclarationAST*& yyast) {
  if (yytoken() == TokenKind::T_RBRACE) return false;

  auto start = yycursor;

  if (yytoken() == TokenKind::T_SEMICOLON)
    return parse_empty_declaration(yyast);

  yyrewind(start);
  if (parse_explicit_instantiation(yyast)) return true;

  yyrewind(start);
  if (parse_explicit_specialization(yyast)) return true;

  yyrewind(start);
  if (parse_template_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_deduction_guide(yyast)) return true;

  yyrewind(start);
  if (parse_export_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_linkage_specification(yyast)) return true;

  yyrewind(start);
  if (parse_namespace_definition(yyast)) return true;

  yyrewind(start);
  if (parse_attribute_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_module_import_declaration(yyast)) return true;

  yyrewind(start);
  return parse_block_declaration(yyast, true);
}

bool Parser::parse_block_declaration(DeclarationAST*& yyast, bool fundef) {
  const auto start = yycursor;

  const auto tk = yytoken();

  if (parse_asm_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_namespace_alias_definition(yyast)) return true;

  yyrewind(start);
  if (parse_using_directive(yyast)) return true;

  yyrewind(start);
  if (parse_alias_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_using_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_using_enum_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_static_assert_declaration(yyast)) return true;

  yyrewind(start);
  if (parse_opaque_enum_declaration(yyast)) return true;

  yyrewind(start);
  return parse_simple_declaration(yyast, fundef);
}

bool Parser::parse_alias_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_USING)) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  parse_attribute_specifier_seq();

  if (!match(TokenKind::T_EQUAL)) return false;

  if (!parse_defining_type_id()) parse_error("expected a type id");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_simple_declaration(DeclarationAST*& yyast, bool fundef) {
  const bool has_extension = match(TokenKind::T___EXTENSION__);

  parse_attribute_specifier_seq();

  if (match(TokenKind::T_SEMICOLON)) return true;

  const auto after_attributes = yycursor;

  DeclSpecs specs;
  if (!parse_decl_specifier_seq_no_typespecs(specs)) yyrewind(after_attributes);

  auto after_decl_specs = yycursor;

  if (parse_declarator_id()) {
    parse_attribute_specifier_seq();

    if (parse_parameters_and_qualifiers()) {
      if (match(TokenKind::T_SEMICOLON)) return true;
      if (fundef && parse_function_definition_body()) return true;
    }
  }

  yyrewind(after_decl_specs);

  if (!parse_decl_specifier_seq(specs)) yyrewind(after_decl_specs);

  after_decl_specs = yycursor;

  if (match(TokenKind::T_SEMICOLON)) return specs.has_complex_typespec;

  if (!specs.has_typespec()) return false;

  const auto has_ref_qualifier = parse_ref_qualifier();

  if (match(TokenKind::T_LBRACKET)) {
    if (parse_identifier_list() && match(TokenKind::T_RBRACKET)) {
      if (parse_initializer() && match(TokenKind::T_SEMICOLON)) return true;
    }
  }

  yyrewind(after_decl_specs);

  Declarator decl;
  if (!parse_declarator(decl)) return false;

  const auto after_declarator = yycursor;

  if (match(TokenKind::T_SEMICOLON)) return true;

  if (fundef && isFunctionDeclarator(decl)) {
    if (parse_function_definition_body()) return true;

    yyrewind(after_declarator);
  }

  if (!parse_declarator_initializer()) yyrewind(after_declarator);

  while (match(TokenKind::T_COMMA)) {
    if (!parse_init_declarator()) return false;
  }

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_function_definition_body() {
  if (parse_requires_clause()) {
    //
  } else {
    parse_virt_specifier_seq();
  }

  return parse_function_body();
}

bool Parser::parse_static_assert_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_STATIC_ASSERT)) return false;

  expect(TokenKind::T_LPAREN);

  ExpressionAST* expression = nullptr;

  if (!parse_constant_expression(expression))
    parse_error("expected an expression");

  if (match(TokenKind::T_COMMA)) {
    if (!parse_string_literal_seq()) parse_error("expected a string literal");
  }

  expect(TokenKind::T_RPAREN);

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_string_literal_seq() {
  if (!match(TokenKind::T_STRING_LITERAL)) return false;

  while (match(TokenKind::T_STRING_LITERAL)) {
    //
  }
  return true;
}

bool Parser::parse_empty_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_attribute_declaration(DeclarationAST*& yyast) {
  if (!parse_attribute_specifier_seq()) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_decl_specifier(DeclSpecs& specs) {
  switch (yytoken()) {
    case TokenKind::T_FRIEND:
    case TokenKind::T_TYPEDEF:
    case TokenKind::T_CONSTEXPR:
    case TokenKind::T_CONSTEVAL:
    case TokenKind::T_CONSTINIT:
    case TokenKind::T_INLINE:
    case TokenKind::T___INLINE:
    case TokenKind::T___INLINE__:
      yyconsume();
      return true;

    default:
      if (parse_storage_class_specifier())
        return true;
      else if (parse_function_specifier())
        return true;
      else if (!specs.no_typespecs)
        return parse_defining_type_specifier(specs);
      return false;
  }  // switch
}

bool Parser::parse_decl_specifier_seq(DeclSpecs& specs) {
  specs.no_typespecs = false;

  if (!parse_decl_specifier(specs)) return false;

  parse_attribute_specifier_seq();

  while (parse_decl_specifier(specs)) {
    parse_attribute_specifier_seq();
  }

  return true;
}

bool Parser::parse_decl_specifier_seq_no_typespecs(DeclSpecs& specs) {
  specs.no_typespecs = true;

  if (!parse_decl_specifier(specs)) return false;

  parse_attribute_specifier_seq();

  while (parse_decl_specifier(specs)) {
    parse_attribute_specifier_seq();
  }

  return true;
}

bool Parser::parse_storage_class_specifier() {
  switch (yytoken()) {
    case TokenKind::T_STATIC:
    case TokenKind::T_THREAD_LOCAL:
    case TokenKind::T_EXTERN:
    case TokenKind::T_MUTABLE:
    case TokenKind::T___THREAD:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_function_specifier() {
  if (match(TokenKind::T_VIRTUAL)) return true;

  return parse_explicit_specifier();
}

bool Parser::parse_explicit_specifier() {
  if (!match(TokenKind::T_EXPLICIT)) return false;

  if (match(TokenKind::T_LPAREN)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constant_expression(expression))
      parse_error("expected a expression");

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_type_specifier(DeclSpecs& specs) {
  if (parse_simple_type_specifier(specs)) return true;

  if (parse_elaborated_type_specifier(specs)) return true;

  if (parse_cv_qualifier()) return true;

  if (parse_typename_specifier()) {
    specs.has_named_typespec = true;
    return true;
  }

  return false;
}

bool Parser::parse_type_specifier_seq() {
  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  if (!parse_type_specifier(specs)) return false;

  parse_attribute_specifier_seq();

  while (yytoken() != TokenKind::T_EOF_SYMBOL) {
    const auto before_type_specifier = yycursor;

    if (!parse_type_specifier(specs)) {
      yyrewind(before_type_specifier);
      break;
    }

    parse_attribute_specifier_seq();
  }

  return true;
}

bool Parser::parse_defining_type_specifier(DeclSpecs& specs) {
  if (!specs.no_class_or_enum_specs) {
    const auto start = yycursor;

    if (parse_enum_specifier()) {
      specs.has_complex_typespec = true;
      return true;
    }

    if (parse_class_specifier()) {
      specs.has_complex_typespec = true;
      return true;
    }
    yyrewind(start);
  }

  return parse_type_specifier(specs);
}

bool Parser::parse_defining_type_specifier_seq(DeclSpecs& specs) {
  if (!parse_defining_type_specifier(specs)) return false;

  parse_attribute_specifier_seq();

  while (yytoken() != TokenKind::T_EOF_SYMBOL) {
    const auto before_type_specifier = yycursor;

    if (!parse_defining_type_specifier(specs)) {
      yyrewind(before_type_specifier);
      break;
    }

    parse_attribute_specifier_seq();
  }

  return true;
}

bool Parser::parse_simple_type_specifier(DeclSpecs& specs) {
  const auto start = yycursor;

  if (parse_named_type_specifier(specs)) return true;

  yyrewind(start);

  if (parse_placeholder_type_specifier_helper(specs)) return true;

  yyrewind(start);

  if (parse_primitive_type_specifier(specs)) return true;

  if (parse_underlying_type_specifier(specs)) return true;

  if (parse_atomic_type_specifier(specs)) return true;

  return parse_decltype_specifier_type_specifier(specs);
}

bool Parser::parse_named_type_specifier(DeclSpecs& specs) {
  if (!parse_named_type_specifier_helper(specs)) return false;

  specs.has_named_typespec = true;

  return true;
}

bool Parser::parse_named_type_specifier_helper(DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  const auto start = yycursor;

  if (parse_nested_name_specifier()) {
    const auto after_nested_name_specifier = yycursor;

    if (match(TokenKind::T_TEMPLATE) && parse_simple_template_id()) {
      return true;
    }

    yyrewind(after_nested_name_specifier);

    if (parse_type_name()) {
      return true;
    }

    yyrewind(after_nested_name_specifier);

    if (parse_template_name()) {
      return true;
    }
  }

  yyrewind(start);

  if (parse_type_name()) {
    return true;
  }

  yyrewind(start);

  if (parse_template_name()) {
    return true;
  }

  return false;
}

bool Parser::parse_placeholder_type_specifier_helper(DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  if (!parse_placeholder_type_specifier()) return false;

  specs.has_placeholder_typespec = true;

  return true;
}

bool Parser::parse_decltype_specifier_type_specifier(DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  if (!parse_decltype_specifier()) return false;

  specs.has_placeholder_typespec = true;

  return true;
}

bool Parser::parse_underlying_type_specifier(DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  if (!match(TokenKind::T___UNDERLYING_TYPE)) return false;

  expect(TokenKind::T_LPAREN);

  if (!parse_type_id()) parse_error("expected type id");

  expect(TokenKind::T_RPAREN);

  specs.has_named_typespec = true;

  return true;
}

bool Parser::parse_automic_type_specifier(DeclSpecs& specs) {
  if (!specs.accepts_simple_typespec()) return false;

  if (!match(TokenKind::T__ATOMIC)) return false;

  expect(TokenKind::T_LPAREN);

  if (!parse_type_id()) parse_error("expected type id");

  expect(TokenKind::T_RPAREN);

  specs.has_simple_typespec = true;

  return true;
}

bool Parser::parse_atomic_type_specifier(DeclSpecs& specs) {
  if (!specs.accepts_simple_typespec()) return false;

  if (!match(TokenKind::T__ATOMIC)) return false;

  expect(TokenKind::T_LPAREN);

  if (!parse_type_id()) parse_error("expected type id");

  expect(TokenKind::T_RPAREN);

  specs.has_simple_typespec = true;

  return true;
}

bool Parser::parse_primitive_type_specifier(DeclSpecs& specs) {
  if (!specs.accepts_simple_typespec()) return false;

  switch (yytoken()) {
    case TokenKind::T_CHAR:
    case TokenKind::T_CHAR8_T:
    case TokenKind::T_CHAR16_T:
    case TokenKind::T_CHAR32_T:
    case TokenKind::T_WCHAR_T:
    case TokenKind::T_BOOL:
    case TokenKind::T_SHORT:
    case TokenKind::T_INT:
    case TokenKind::T_LONG:
    case TokenKind::T_SIGNED:
    case TokenKind::T_UNSIGNED:
    case TokenKind::T_FLOAT:
    case TokenKind::T_DOUBLE:
    case TokenKind::T_VOID:
    case TokenKind::T___INT64:
    case TokenKind::T___INT128:
    case TokenKind::T___FLOAT80:
    case TokenKind::T___FLOAT128:
    case TokenKind::T___COMPLEX__:
      yyconsume();
      specs.has_simple_typespec = true;
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_type_name() {
  const auto start = yycursor;

  if (parse_class_name()) return true;

  yyrewind(start);

  if (parse_enum_name()) return true;

  yyrewind(start);

  return parse_typedef_name();
}

bool Parser::parse_elaborated_type_specifier(DeclSpecs& specs) {
  // ### cleanup

  if (yytoken() == TokenKind::T_ENUM) return parse_elaborated_enum_specifier();

  if (!parse_class_key()) return false;

  parse_attribute_specifier_seq();

  const auto before_nested_name_specifier = yycursor;

  if (!parse_nested_name_specifier()) {
    yyrewind(before_nested_name_specifier);

    if (parse_simple_template_id()) {
      specs.has_complex_typespec = true;
      return true;
    }

    yyrewind(before_nested_name_specifier);

    if (!match(TokenKind::T_IDENTIFIER)) return false;

    specs.has_complex_typespec = true;
    return true;
  }

  const auto after_nested_name_specifier = yycursor;

  const bool has_template = match(TokenKind::T_TEMPLATE);

  if (parse_simple_template_id()) {
    specs.has_complex_typespec = true;
    return true;
  }

  if (has_template) {
    parse_error("expected a template-id");
    specs.has_complex_typespec = true;
    return true;
  }

  yyrewind(after_nested_name_specifier);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  specs.has_complex_typespec = true;

  return true;
}

bool Parser::parse_elaborated_enum_specifier() {
  if (!match(TokenKind::T_ENUM)) return false;

  const auto saved = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(saved);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_decl_specifier_seq_no_typespecs() {
  DeclSpecs specs;
  return parse_decl_specifier_seq_no_typespecs(specs);
}

bool Parser::parse_decl_specifier_seq() {
  DeclSpecs specs;
  return parse_decl_specifier_seq(specs);
}

bool Parser::parse_declarator() {
  Declarator decl;
  return parse_declarator(decl);
}

bool Parser::parse_decltype_specifier() {
  if (match(TokenKind::T_DECLTYPE) || match(TokenKind::T___DECLTYPE) ||
      match(TokenKind::T___DECLTYPE__)) {
    if (!match(TokenKind::T_LPAREN)) return false;

    if (yytoken() == TokenKind::T_AUTO)
      return false;  // placeholder type specifier

    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);

    return true;
  }

  if (match(TokenKind::T___TYPEOF) || match(TokenKind::T___TYPEOF__)) {
    expect(TokenKind::T_LPAREN);

    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);

    return true;
  }

  return false;
}

bool Parser::parse_placeholder_type_specifier() {
  parse_type_constraint();

  if (match(TokenKind::T_AUTO)) return true;

  if (match(TokenKind::T_DECLTYPE)) {
    if (!match(TokenKind::T_LPAREN)) return false;

    if (!match(TokenKind::T_AUTO)) return false;

    if (!match(TokenKind::T_RPAREN)) return false;

    return true;
  }

  return false;
}

bool Parser::parse_init_declarator_list() {
  if (!parse_init_declarator()) return false;

  while (match(TokenKind::T_COMMA)) {
    if (!parse_init_declarator()) parse_error("expected a declarator");
  }

  return true;
}

bool Parser::parse_init_declarator() {
  if (!parse_declarator()) return false;

  const auto saved = yycursor;

  if (!parse_declarator_initializer()) yyrewind(saved);

  return true;
}

bool Parser::parse_declarator_initializer() {
  if (parse_requires_clause()) return true;

  return parse_initializer();
}

bool Parser::parse_declarator(Declarator& decl) {
  if (parse_ptr_operator()) {
    if (!parse_declarator(decl)) return false;

    decl.push_back(PtrDeclarator());

    return true;
  }

  return parse_noptr_declarator(decl);
}

bool Parser::parse_ptr_operator_seq() {
  if (!parse_ptr_operator()) return false;

  while (parse_ptr_operator()) {
    //
  }

  return true;
}

bool Parser::parse_core_declarator(Declarator& decl) {
  if (parse_declarator_id()) {
    parse_attribute_specifier_seq();
    decl.push_back(DeclaratorId());
    return true;
  }

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!parse_declarator(decl)) return false;

  if (!match(TokenKind::T_RPAREN)) return false;

  decl.push_back(NestedDeclarator());

  return true;
}

bool Parser::parse_noptr_declarator(Declarator& decl) {
  if (!parse_core_declarator(decl)) return false;

  while (true) {
    const auto saved = yycursor;

    if (match(TokenKind::T_LBRACKET)) {
      if (!match(TokenKind::T_RBRACKET)) {
        ExpressionAST* expression = nullptr;

        if (!parse_constant_expression(expression)) {
          yyrewind(saved);
          break;
        }

        if (!match(TokenKind::T_RBRACKET)) {
          yyrewind(saved);
          break;
        }
      }

      parse_attribute_specifier_seq();
      decl.push_back(ArrayDeclarator());
    } else if (parse_parameters_and_qualifiers()) {
      parse_trailing_return_type();
      decl.push_back(FunctionDeclarator());
    } else {
      yyrewind(saved);
      break;
    }
  }

  return true;
}

bool Parser::parse_parameters_and_qualifiers() {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_parameter_declaration_clause()) return false;

    if (!match(TokenKind::T_RPAREN)) return false;
  }

  parse_cv_qualifier_seq();

  parse_ref_qualifier();

  parse_noexcept_specifier();

  parse_attribute_specifier_seq();

  return true;
}

bool Parser::parse_cv_qualifier_seq() {
  if (!parse_cv_qualifier()) return false;

  while (parse_cv_qualifier()) {
    //
  }

  return true;
}

bool Parser::parse_trailing_return_type() {
  if (!match(TokenKind::T_MINUS_GREATER)) return false;

  if (!parse_type_id()) parse_error("expected a type id");

  return true;
}

bool Parser::parse_ptr_operator() {
  if (match(TokenKind::T_STAR)) {
    parse_attribute_specifier_seq();
    parse_cv_qualifier_seq();
    return true;
  }

  if (match(TokenKind::T_AMP) || match(TokenKind::T_AMP_AMP)) {
    parse_attribute_specifier_seq();
    return true;
  }

  const auto saved = yycursor;

  if (parse_nested_name_specifier() && match(TokenKind::T_STAR)) {
    parse_attribute_specifier_seq();
    parse_cv_qualifier_seq();
    return true;
  }

  yyrewind(saved);

  return false;
}

bool Parser::parse_cv_qualifier() {
  switch (yytoken()) {
    case TokenKind::T_CONST:
    case TokenKind::T_VOLATILE:
    case TokenKind::T___RESTRICT:
    case TokenKind::T___RESTRICT__:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_ref_qualifier() {
  switch (yytoken()) {
    case TokenKind::T_AMP:
    case TokenKind::T_AMP_AMP:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_declarator_id() {
  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  if (!parse_id_expression()) return false;

  return true;
}

bool Parser::parse_type_id() {
  if (!parse_type_specifier_seq()) return false;

  const auto before_declarator = yycursor;

  if (!parse_abstract_declarator()) yyrewind(before_declarator);

  return true;
}

bool Parser::parse_defining_type_id() {
  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  if (!parse_defining_type_specifier_seq(specs)) return false;

  const auto before_declarator = yycursor;

  if (!parse_abstract_declarator()) yyrewind(before_declarator);

  return true;
}

bool Parser::parse_abstract_declarator() {
  if (parse_abstract_pack_declarator()) return true;

  if (parse_ptr_abstract_declarator()) return true;

  const auto saved = yycursor;

  if (parse_parameters_and_qualifiers() && parse_trailing_return_type())
    return true;

  yyrewind(saved);

  if (!parse_noptr_abstract_declarator()) return false;

  const auto after_noptr_declarator = yycursor;

  if (parse_parameters_and_qualifiers() && parse_trailing_return_type()) {
    //
  } else {
    yyrewind(after_noptr_declarator);
  }

  return true;
}

bool Parser::parse_ptr_abstract_declarator() {
  if (!parse_ptr_operator_seq()) return false;

  const auto saved = yycursor;

  if (!parse_noptr_abstract_declarator()) yyrewind(saved);

  return true;
}

bool Parser::parse_noptr_abstract_declarator() {
  const auto start = yycursor;

  auto has_nested_declarator = false;

  if (match(TokenKind::T_LPAREN) && parse_ptr_abstract_declarator() &&
      match(TokenKind::T_RPAREN)) {
    has_nested_declarator = true;
  } else {
    yyrewind(start);
  }

  const auto after_nested_declarator = yycursor;

  if (yytoken() == TokenKind::T_LPAREN) {
    if (!parse_parameters_and_qualifiers()) yyrewind(after_nested_declarator);
  }

  if (yytoken() == TokenKind::T_LBRACKET) {
    while (match(TokenKind::T_LBRACKET)) {
      if (!match(TokenKind::T_RBRACKET)) {
        ExpressionAST* expression = nullptr;

        if (!parse_constant_expression(expression))
          parse_error("expected an expression");

        expect(TokenKind::T_RBRACKET);
      }
    }
  }

  return true;
}

bool Parser::parse_abstract_pack_declarator() {
  parse_ptr_operator_seq();

  if (!parse_noptr_abstract_pack_declarator()) return false;

  return true;
}

bool Parser::parse_noptr_abstract_pack_declarator() {
  if (!match(TokenKind::T_DOT_DOT_DOT)) return false;

  if (parse_parameters_and_qualifiers()) return true;

  while (match(TokenKind::T_LBRACKET)) {
    if (!match(TokenKind::T_RBRACKET)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(expression))
        parse_error("expected a constant expression");

      expect(TokenKind::T_RBRACKET);

      parse_attribute_specifier_seq();
    }
  }

  return true;
}

bool Parser::parse_parameter_declaration_clause() {
  if (match(TokenKind::T_DOT_DOT_DOT)) return true;

  if (!parse_parameter_declaration_list()) return false;

  match(TokenKind::T_COMMA);

  match(TokenKind::T_DOT_DOT_DOT);

  return true;
}

bool Parser::parse_parameter_declaration_list() {
  if (!parse_parameter_declaration()) return false;

  while (match(TokenKind::T_COMMA)) {
    if (!parse_parameter_declaration()) {
      yyrewind(yycursor - 1);
      break;
    }
  }

  return true;
}

bool Parser::parse_parameter_declaration() {
  parse_attribute_specifier_seq();

  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  if (!parse_decl_specifier_seq(specs)) return false;

  const auto before_declarator = yycursor;

  if (!parse_declarator()) {
    yyrewind(before_declarator);

    if (!parse_abstract_declarator()) yyrewind(before_declarator);
  }

  if (match(TokenKind::T_EQUAL)) {
    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression))
      parse_error("expected an initializer");
  }

  return true;
}

bool Parser::parse_initializer() {
  if (match(TokenKind::T_LPAREN)) {
    if (yytoken() == TokenKind::T_RPAREN) return false;

    if (!parse_expression_list()) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);

    return true;
  }

  return parse_brace_or_equal_initializer();
}

bool Parser::parse_brace_or_equal_initializer() {
  if (yytoken() == TokenKind::T_LBRACE) return parse_braced_init_list();

  if (!match(TokenKind::T_EQUAL)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_initializer_clause(expression))
    parse_error("expected an intializer");

  return true;
}

bool Parser::parse_initializer_clause(ExpressionAST*& yyast) {
  if (yytoken() == TokenKind::T_LBRACE) return parse_braced_init_list();

  return parse_assignment_expression(yyast);
}

bool Parser::parse_braced_init_list() {
  if (!match(TokenKind::T_LBRACE)) return false;

  if (yytoken() == TokenKind::T_DOT) {
    if (!parse_designated_initializer_clause())
      parse_error("expected designated initializer clause");

    while (match(TokenKind::T_COMMA)) {
      if (yytoken() == TokenKind::T_RBRACE) break;

      if (!parse_designated_initializer_clause())
        parse_error("expected designated initializer clause");
    }
  } else if (match(TokenKind::T_COMMA)) {
    // nothing to do
  } else if (yytoken() != TokenKind::T_RBRACE) {
    if (!parse_initializer_list()) parse_error("expected initializer list");
  }

  expect(TokenKind::T_RBRACE);

  return true;
}

bool Parser::parse_initializer_list() {
  ExpressionAST* e = nullptr;

  if (!parse_initializer_clause(e)) return false;

  bool has_triple_dot = false;
  if (match(TokenKind::T_DOT_DOT_DOT)) {
    has_triple_dot = true;
  }

  while (match(TokenKind::T_COMMA)) {
    if (yytoken() == TokenKind::T_RBRACE) break;

    ExpressionAST* e = nullptr;
    if (!parse_initializer_clause(e))
      parse_error("expected initializer clause");

    bool has_triple_dot = false;
    if (match(TokenKind::T_DOT_DOT_DOT)) {
      has_triple_dot = true;
    }
  }

  return true;
}

bool Parser::parse_designated_initializer_clause() {
  if (!parse_designator()) return false;

  if (!parse_brace_or_equal_initializer())
    parse_error("expected an initializer");

  return true;
}

bool Parser::parse_designator() {
  if (!match(TokenKind::T_DOT)) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_expr_or_braced_init_list() {
  if (yytoken() == TokenKind::T_LBRACE) return parse_braced_init_list();

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) parse_error("expected an expression");

  return true;
}

bool Parser::parse_virt_specifier_seq() {
  if (!parse_virt_specifier()) return false;

  while (parse_virt_specifier()) {
    //
  }

  return true;
}

bool Parser::parse_function_body() {
  if (yytoken() == TokenKind::T_SEMICOLON) return false;

  if (parse_function_try_block()) return true;

  if (match(TokenKind::T_EQUAL)) {
    if (match(TokenKind::T_DEFAULT)) {
      expect(TokenKind::T_SEMICOLON);
      return true;
    }

    if (match(TokenKind::T_DELETE)) {
      expect(TokenKind::T_SEMICOLON);
      return true;
    }

    return false;
  }

  parse_ctor_initializer();

  if (yytoken() != TokenKind::T_LBRACE) return false;

  if (skip_function_body) {
    expect(TokenKind::T_LBRACE);

    int depth = 1;

    TokenKind tok;

    while ((tok = yytoken()) != TokenKind::T_EOF_SYMBOL) {
      if (tok == TokenKind::T_LBRACE) {
        ++depth;
      } else if (tok == TokenKind::T_RBRACE) {
        if (!--depth) {
          break;
        }
      }

      yyconsume();
    }

    expect(TokenKind::T_RBRACE);
  }

  StatementAST* statement = nullptr;

  if (!parse_compound_statement(statement))
    parse_error("expected a compound statement");

  return true;
}

bool Parser::parse_enum_specifier() {
  if (!parse_enum_head()) return false;

  if (!match(TokenKind::T_LBRACE)) return false;

  if (!match(TokenKind::T_RBRACE)) {
    parse_enumerator_list();

    match(TokenKind::T_COMMA);

    expect(TokenKind::T_RBRACE);
  }

  return true;
}

bool Parser::parse_enum_head() {
  if (!parse_enum_key()) return false;

  parse_attribute_specifier_seq();

  parse_enum_head_name();

  parse_enum_base();

  return true;
}

bool Parser::parse_enum_head_name() {
  const auto start = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_opaque_enum_declaration(DeclarationAST*& yyast) {
  if (!parse_enum_key()) return false;

  parse_attribute_specifier_seq();

  if (!parse_enum_head_name()) return false;

  parse_enum_base();

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_enum_key() {
  if (!match(TokenKind::T_ENUM)) return false;

  if (match(TokenKind::T_CLASS)) {
    //
  } else if (match(TokenKind::T_STRUCT)) {
    //
  }

  return true;
}

bool Parser::parse_enum_base() {
  if (!match(TokenKind::T_COLON)) return false;

  if (!parse_type_specifier_seq()) parse_error("expected a type specifier");

  return true;
}

bool Parser::parse_enumerator_list() {
  if (!parse_enumerator_definition()) return false;

  while (match(TokenKind::T_COMMA)) {
    if (yytoken() == TokenKind::T_RBRACE) {
      yyrewind(yycursor - 1);
      break;
    }

    if (!parse_enumerator_definition()) parse_error("expected an enumerator");
  }

  return true;
}

bool Parser::parse_enumerator_definition() {
  if (!parse_enumerator()) return false;

  if (match(TokenKind::T_EQUAL)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constant_expression(expression))
      parse_error("expected an expression");
  }

  return true;
}

bool Parser::parse_enumerator() {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  parse_attribute_specifier_seq();

  return true;
}

bool Parser::parse_using_enum_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_USING)) return false;

  if (!parse_elaborated_enum_specifier()) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_namespace_definition(DeclarationAST*& yyast) {
  const auto start = yycursor;

  const auto has_inline = match(TokenKind::T_INLINE);

  if (!match(TokenKind::T_NAMESPACE)) {
    yyrewind(start);
    return false;
  }

  parse_attribute_specifier_seq();

  enum NamespaceKind {
    kAnonymous,
    kNamed,
    kNested
  } kind = NamespaceKind::kAnonymous;

  uint32_t namespace_name_token = 0;

  if (yytoken() == TokenKind::T_IDENTIFIER &&
      yytoken(1) == TokenKind::T_COLON_COLON) {
    yyconsume();

    while (match(TokenKind::T_COLON_COLON)) {
      match(TokenKind::T_INLINE);
      expect(TokenKind::T_IDENTIFIER);
    }
    kind = NamespaceKind::kNested;
  } else if (yytoken() == TokenKind::T_IDENTIFIER) {
    kind = NamespaceKind::kNamed;
    namespace_name_token = yycursor;
    yyconsume();
  }

  parse_attribute_specifier_seq();

  expect(TokenKind::T_LBRACE);

  parse_namespace_body();

  expect(TokenKind::T_RBRACE);

  return true;
}

bool Parser::parse_namespace_body() {
  bool skipping = false;

  while (yytoken() != TokenKind::T_EOF_SYMBOL) {
    if (yytoken() == TokenKind::T_RBRACE) break;

    const auto saved = yycursor;

    DeclarationAST* decl = nullptr;

    if (parse_declaration(decl)) {
      skipping = false;
    } else {
      parse_skip_declaration(skipping);

      if (yycursor == saved) yyconsume();
    }
  }

  return true;
}

bool Parser::parse_namespace_alias_definition(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_NAMESPACE)) return false;

  expect(TokenKind::T_IDENTIFIER);

  expect(TokenKind::T_EQUAL);

  if (!parse_qualified_namespace_specifier())
    parse_error("expected a namespace name");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_qualified_namespace_specifier() {
  const auto saved = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(saved);

  if (!parse_namespace_name()) return false;

  return true;
}

bool Parser::parse_using_directive(DeclarationAST*& yyast) {
  parse_attribute_specifier_seq();

  if (!match(TokenKind::T_USING)) return false;

  if (!match(TokenKind::T_NAMESPACE)) return false;

  const auto saved = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(saved);

  if (!parse_namespace_name()) parse_error("expected a namespace name");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_using_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_USING)) return false;

  if (!parse_using_declarator_list())
    parse_error("expected a using declarator");

  match(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_using_declarator_list() {
  if (!parse_using_declarator()) return false;

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  while (match(TokenKind::T_COMMA)) {
    if (!parse_using_declarator()) parse_error("expected a using declarator");

    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);
  }

  return true;
}

bool Parser::parse_using_declarator() {
  const auto has_typename = match(TokenKind::T_TYPENAME);

  const auto saved = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(saved);

  if (!parse_unqualified_id()) return false;

  return true;
}

bool Parser::parse_asm_declaration(DeclarationAST*& yyast) {
  parse_attribute_specifier_seq();

  if (!match(TokenKind::T_ASM)) return false;

  expect(TokenKind::T_LPAREN);

  while (match(TokenKind::T_STRING_LITERAL)) {
  }

  expect(TokenKind::T_RPAREN);

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_linkage_specification(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_EXTERN)) return false;

  if (!match(TokenKind::T_STRING_LITERAL)) return false;

  if (match(TokenKind::T_LBRACE)) {
    if (!match(TokenKind::T_RBRACE)) {
      if (!parse_declaration_seq()) parse_error("expected a declaration");

      expect(TokenKind::T_RBRACE);
    }

    return true;
  }

  DeclarationAST* declaration = nullptr;

  if (!parse_declaration(declaration)) return false;

  return true;
}

bool Parser::parse_attribute_specifier_seq() {
  if (!parse_attribute_specifier()) return false;

  while (parse_attribute_specifier()) {
    //
  }

  return true;
}

bool Parser::parse_attribute_specifier() {
  if (yytoken() == TokenKind::T_LBRACKET &&
      yytoken(1) == TokenKind::T_LBRACKET) {
    yyconsume();
    yyconsume();
    parse_attribute_using_prefix();
    parse_attribute_list();
    expect(TokenKind::T_RBRACKET);
    expect(TokenKind::T_RBRACKET);
    return true;
  }

  if (parse_gcc_attribute()) return true;

  if (parse_alignment_specifier()) return true;

  if (parse_asm_specifier()) return true;

  return false;
}

bool Parser::parse_asm_specifier() {
  if (!match(TokenKind::T___ASM__) && !match(TokenKind::T___ASM)) return false;

  expect(TokenKind::T_LPAREN);

  if (!parse_string_literal_seq()) parse_error("expected a string literal");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_gcc_attribute() {
  if (!match(TokenKind::T___ATTRIBUTE) && !match(TokenKind::T___ATTRIBUTE__))
    return false;

  expect(TokenKind::T_LPAREN);

  parse_skip_balanced();

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_gcc_attribute_seq() {
  if (!parse_gcc_attribute()) return false;

  while (parse_gcc_attribute()) {
    //
  }

  return true;
}

bool Parser::parse_skip_balanced() {
  int count = 1;

  TokenKind tk;

  while ((tk = yytoken()) != TokenKind::T_EOF_SYMBOL) {
    if (tk == TokenKind::T_LPAREN) {
      ++count;
    } else if (tk == TokenKind::T_RPAREN) {
      if (!--count) return true;
    }

    yyconsume();
  }

  return false;
}

bool Parser::parse_alignment_specifier() {
  if (!match(TokenKind::T_ALIGNAS)) return false;

  expect(TokenKind::T_LPAREN);

  const auto after_lparen = yycursor;

  if (parse_type_id()) {
    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

    if (match(TokenKind::T_RPAREN)) {
      return true;
    }
  }

  yyrewind(after_lparen);

  ExpressionAST* expression = nullptr;

  if (!parse_constant_expression(expression))
    parse_error("expected an expression");

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_attribute_using_prefix() {
  if (!match(TokenKind::T_USING)) return false;

  if (!parse_attribute_namespace())
    parse_error("expected an attribute namespace");

  expect(TokenKind::T_COLON);

  return true;
}

bool Parser::parse_attribute_list() {
  parse_attribute();

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  while (match(TokenKind::T_COMMA)) {
    parse_attribute();

    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);
  }

  return true;
}

bool Parser::parse_attribute() {
  if (!parse_attribute_token()) return false;

  parse_attribute_argument_clause();

  return true;
}

bool Parser::parse_attribute_token() {
  const auto start = yycursor;

  if (parse_attribute_scoped_token()) return true;

  yyrewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_attribute_scoped_token() {
  if (!parse_attribute_namespace()) return false;

  if (!match(TokenKind::T_COLON_COLON)) return false;

  expect(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_attribute_namespace() {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_attribute_argument_clause() {
  if (!match(TokenKind::T_LPAREN)) return false;

  parse_skip_balanced();

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_module_declaration() {
  const auto has_export = parse_export_keyword();

  if (!parse_module_keyword()) return false;

  if (!parse_module_name()) parse_error("expected a module name");

  if (yytoken() == TokenKind::T_COLON) {
    parse_module_partition();
  }

  parse_attribute_specifier_seq();

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_module_name() {
  const auto start = yycursor;

  if (!parse_module_name_qualifier()) yyrewind(start);

  expect(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_module_partition() {
  if (!match(TokenKind::T_COLON)) return false;

  const auto saved = yycursor;

  if (!parse_module_name_qualifier()) yyrewind(saved);

  expect(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_module_name_qualifier() {
  if (yytoken() != TokenKind::T_IDENTIFIER) return false;

  if (yytoken(1) != TokenKind::T_DOT) return false;

  do {
    yyconsume();
    yyconsume();
  } while (yytoken() == TokenKind::T_IDENTIFIER &&
           yytoken(1) == TokenKind::T_DOT);

  return true;
}

bool Parser::parse_export_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_EXPORT)) return false;

  if (match(TokenKind::T_LBRACE)) {
    if (!match(TokenKind::T_RBRACE)) {
      if (!parse_declaration_seq()) parse_error("expected a declaration");

      expect(TokenKind::T_RBRACE);
    }

    return true;
  }

  if (parse_maybe_import()) {
    DeclarationAST* declaration = nullptr;

    if (!parse_module_import_declaration(declaration))
      parse_error("expected a module import declaration");

    return true;
  }

  DeclarationAST* declaration = nullptr;

  if (!parse_declaration(declaration)) parse_error("expected a declaration");

  return true;
}

bool Parser::parse_maybe_import() {
  if (!module_unit) return false;

  const auto start = yycursor;

  const auto import = parse_import_keyword();

  yyrewind(start);

  return import;
}

bool Parser::parse_module_import_declaration(DeclarationAST*& yyast) {
  if (!parse_import_keyword()) return false;

  if (!parse_import_name()) parse_error("expected a module");

  parse_attribute_specifier_seq();

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_import_name() {
  if (parse_header_name()) return true;

  if (parse_module_partition()) return true;

  return parse_module_name();
}

bool Parser::parse_global_module_fragment() {
  if (!parse_module_keyword()) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  parse_declaration_seq();

  return true;
}

bool Parser::parse_private_module_fragment() {
  if (!parse_module_keyword()) return false;

  if (!match(TokenKind::T_COLON)) return false;

  if (!match(TokenKind::T_PRIVATE)) return false;

  expect(TokenKind::T_SEMICOLON);

  parse_declaration_seq();

  return true;
}

bool Parser::parse_class_specifier() {
  const auto start = yycursor;

  auto it = class_specifiers_.find(start);

  if (it != class_specifiers_.end()) {
    auto [cursor, parsed] = it->second;
    yyrewind(cursor);
    return parsed;
  }

  Name className;

  if (!parse_class_head(className)) {
    parse_reject_class_specifier(start);
    return false;
  }

  if (!match(TokenKind::T_LBRACE)) {
    parse_reject_class_specifier(start);
    return false;
  }

  if (!match(TokenKind::T_RBRACE)) {
    if (!parse_class_body()) parse_error("expected class body");

    expect(TokenKind::T_RBRACE);
  }

  parse_leave_class_specifier(start);

  return true;
}

bool Parser::parse_leave_class_specifier(uint32_t start) {
  class_specifiers_.emplace(start, std::make_tuple(yycursor, true));
  return true;
}

bool Parser::parse_reject_class_specifier(uint32_t start) {
  class_specifiers_.emplace(start, std::make_tuple(yycursor, false));
  return true;
}

bool Parser::parse_class_body() {
  bool skipping = false;

  while (yytoken() != TokenKind::T_EOF_SYMBOL) {
    if (yytoken() == TokenKind::T_RBRACE) break;

    const auto saved = yycursor;

    DeclarationAST* declaration = nullptr;

    if (parse_member_specification(declaration)) {
      skipping = false;
    } else {
      if (!skipping) parse_error("expected a declaration");

      if (yycursor == saved) yyconsume();

      skipping = true;
    }
  }

  return true;
}

bool Parser::parse_class_head(Name& name) {
  if (!parse_class_key()) return false;

  parse_attribute_specifier_seq();

  if (parse_class_head_name(name)) {
    parse_class_virt_specifier();
  }

  parse_base_clause();

  return true;
}

bool Parser::parse_class_head_name(Name& name) {
  const auto start = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(start);

  if (!parse_class_name(name)) return false;

  return true;
}

bool Parser::parse_class_virt_specifier() {
  if (!parse_final()) return false;

  return true;
}

bool Parser::parse_class_key() {
  switch (yytoken()) {
    case TokenKind::T_CLASS:
    case TokenKind::T_STRUCT:
    case TokenKind::T_UNION:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_member_specification(DeclarationAST*& yyast) {
  return parse_member_declaration(yyast);
}

bool Parser::parse_member_declaration(DeclarationAST*& yyast) {
  const auto start = yycursor;

  if (parse_access_specifier()) {
    expect(TokenKind::T_COLON);
    return true;
  }

  if (parse_empty_declaration(yyast)) return true;

  if (yytoken() == TokenKind::T_USING) {
    if (parse_using_enum_declaration(yyast)) return true;

    yyrewind(start);

    if (parse_alias_declaration(yyast)) return true;

    yyrewind(start);

    if (parse_using_declaration(yyast)) return true;

    return false;
  }

  if (parse_static_assert_declaration(yyast)) return true;

  if (parse_maybe_template_member()) {
    if (parse_template_declaration(yyast)) return true;

    if (parse_explicit_specialization(yyast)) return true;

    if (parse_deduction_guide(yyast)) return true;

    return false;
  }

  if (parse_opaque_enum_declaration(yyast)) return true;

  yyrewind(start);

  return parse_member_declaration_helper(yyast);
}

bool Parser::parse_maybe_template_member() {
  const auto start = yycursor;

  match(TokenKind::T_EXPLICIT);

  const auto has_template = match(TokenKind::T_TEMPLATE);

  yyrewind(start);

  return has_template;
}

bool Parser::parse_member_declaration_helper(DeclarationAST*& yyast) {
  const auto has_extension = match(TokenKind::T___EXTENSION__);

  parse_attribute_specifier_seq();

  auto after_decl_specs = yycursor;

  DeclSpecs specs;

  if (!parse_decl_specifier_seq_no_typespecs(specs)) yyrewind(after_decl_specs);

  after_decl_specs = yycursor;

  if (parse_declarator_id()) {
    parse_attribute_specifier_seq();

    if (parse_parameters_and_qualifiers()) {
      const auto after_parameters = yycursor;

      if (parse_member_function_definition_body()) return true;

      yyrewind(after_parameters);

      if (parse_member_declarator_modifier() && match(TokenKind::T_SEMICOLON)) {
        return true;
      }
    }
  }

  yyrewind(after_decl_specs);

  if (!parse_decl_specifier_seq(specs)) yyrewind(after_decl_specs);

  after_decl_specs = yycursor;

  if (!specs.has_typespec()) return false;

  if (match(TokenKind::T_SEMICOLON)) return true;  // ### complex typespec

  Declarator decl;

  if (parse_declarator(decl) && isFunctionDeclarator(decl)) {
    if (parse_member_function_definition_body()) return true;
  }

  yyrewind(after_decl_specs);

  if (!parse_member_declarator_list()) parse_error("expected a declarator");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_member_function_definition_body() {
  const auto has_requires_clause = parse_requires_clause();

  bool has_virt_specifier_seq = false;

  if (!has_requires_clause) has_virt_specifier_seq = parse_virt_specifier_seq();

  if (!parse_function_body()) return false;

  return true;
}

bool Parser::parse_member_declarator_modifier() {
  if (parse_requires_clause()) return true;

  if (yytoken() == TokenKind::T_LBRACE || yytoken() == TokenKind::T_EQUAL)
    return parse_brace_or_equal_initializer();

  parse_virt_specifier_seq();

  parse_pure_specifier();

  return true;
}

bool Parser::parse_member_declarator_list() {
  if (!parse_member_declarator()) return false;

  while (match(TokenKind::T_COMMA)) {
    if (!parse_member_declarator()) parse_error("expected a declarator");
  }

  return true;
}

bool Parser::parse_member_declarator() {
  const auto start = yycursor;

  const auto has_identifier = match(TokenKind::T_IDENTIFIER);

  parse_attribute_specifier();

  if (match(TokenKind::T_COLON)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constant_expression(expression))
      parse_error("expected an expression");

    parse_brace_or_equal_initializer();

    return true;
  }

  yyrewind(start);

  if (!parse_declarator()) return false;

  if (!parse_member_declarator_modifier()) return false;

  return true;
}

bool Parser::parse_virt_specifier() {
  if (parse_final()) return true;

  if (parse_override()) return true;

  return false;
}

bool Parser::parse_pure_specifier() {
  if (!match(TokenKind::T_EQUAL)) return false;

  if (!match(TokenKind::T_INTEGER_LITERAL)) return false;

  const auto& number = unit->tokenText(yycursor - 1);

  if (number != "0") return false;

  return true;
}

bool Parser::parse_conversion_function_id() {
  if (!match(TokenKind::T_OPERATOR)) return false;

  if (!parse_conversion_type_id()) return false;

  return true;
}

bool Parser::parse_conversion_type_id() {
  if (!parse_type_specifier_seq()) return false;

  parse_conversion_declarator();

  return true;
}

bool Parser::parse_conversion_declarator() { return parse_ptr_operator_seq(); }

bool Parser::parse_base_clause() {
  if (!match(TokenKind::T_COLON)) return false;

  if (!parse_base_specifier_list())
    parse_error("expected a base class specifier");

  return true;
}

bool Parser::parse_base_specifier_list() {
  if (!parse_base_specifier()) return false;

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  while (match(TokenKind::T_COMMA)) {
    if (!parse_base_specifier()) parse_error("expected a base class specifier");

    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);
  }

  return true;
}

bool Parser::parse_base_specifier() {
  parse_attribute_specifier_seq();

  bool has_virtual = match(TokenKind::T_VIRTUAL);

  bool has_access_specifier = false;

  if (has_virtual)
    has_access_specifier = parse_access_specifier();
  else if (parse_access_specifier()) {
    has_virtual = match(TokenKind::T_VIRTUAL);
    has_access_specifier = true;
  }

  if (!parse_class_or_decltype()) return false;

  return true;
}

bool Parser::parse_class_or_decltype() {
  const auto start = yycursor;

  if (parse_nested_name_specifier()) {
    if (match(TokenKind::T_TEMPLATE)) {
      if (parse_simple_template_id()) return true;
    }

    if (parse_type_name()) return true;
  }

  yyrewind(start);

  if (parse_decltype_specifier()) return true;

  return parse_type_name();
}

bool Parser::parse_access_specifier() {
  switch (yytoken()) {
    case TokenKind::T_PRIVATE:
    case TokenKind::T_PROTECTED:
    case TokenKind::T_PUBLIC:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_ctor_initializer() {
  if (!match(TokenKind::T_COLON)) return false;

  if (!parse_mem_initializer_list())
    parse_error("expected a member intializer");

  return true;
}

bool Parser::parse_mem_initializer_list() {
  if (!parse_mem_initializer()) return false;

  match(TokenKind::T_DOT_DOT_DOT);

  while (match(TokenKind::T_COMMA)) {
    if (!parse_mem_initializer()) parse_error("expected a member initializer");

    match(TokenKind::T_DOT_DOT_DOT);
  }

  return true;
}

bool Parser::parse_mem_initializer() {
  if (!parse_mem_initializer_id()) parse_error("expected an member id");

  if (yytoken() == TokenKind::T_LBRACE) {
    if (!parse_braced_init_list()) parse_error("expected an initializer");
  } else {
    expect(TokenKind::T_LPAREN);

    if (!match(TokenKind::T_RPAREN)) {
      if (!parse_expression_list()) parse_error("expected an expression");

      expect(TokenKind::T_RPAREN);
    }
  }

  return true;
}

bool Parser::parse_mem_initializer_id() {
  const auto start = yycursor;

  if (parse_class_or_decltype()) return true;

  yyrewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_operator_function_id() {
  if (!match(TokenKind::T_OPERATOR)) return false;

  if (!parse_op()) return false;

  return true;
}

bool Parser::parse_op() {
  switch (yytoken()) {
    case TokenKind::T_LPAREN:
      yyconsume();
      expect(TokenKind::T_RPAREN);
      return true;

    case TokenKind::T_LBRACKET:
      yyconsume();
      expect(TokenKind::T_RBRACKET);
      return true;

    case TokenKind::T_GREATER: {
      if (parse_greater_greater_equal()) return true;
      if (parse_greater_greater()) return true;
      if (parse_greater_equal()) return true;
      yyconsume();
      return true;
    }

    case TokenKind::T_NEW:
      yyconsume();
      if (match(TokenKind::T_LBRACKET)) {
        expect(TokenKind::T_RBRACKET);
      }
      return true;

    case TokenKind::T_DELETE:
      yyconsume();
      if (match(TokenKind::T_LBRACKET)) {
        expect(TokenKind::T_RBRACKET);
      }
      return true;

    case TokenKind::T_CO_AWAIT:
    case TokenKind::T_MINUS_GREATER:
    case TokenKind::T_MINUS_GREATER_STAR:
    case TokenKind::T_TILDE:
    case TokenKind::T_EXCLAIM:
    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
    case TokenKind::T_STAR:
    case TokenKind::T_SLASH:
    case TokenKind::T_PERCENT:
    case TokenKind::T_CARET:
    case TokenKind::T_AMP:
    case TokenKind::T_BAR:
    case TokenKind::T_EQUAL:
    case TokenKind::T_PLUS_EQUAL:
    case TokenKind::T_MINUS_EQUAL:
    case TokenKind::T_STAR_EQUAL:
    case TokenKind::T_SLASH_EQUAL:
    case TokenKind::T_PERCENT_EQUAL:
    case TokenKind::T_CARET_EQUAL:
    case TokenKind::T_AMP_EQUAL:
    case TokenKind::T_BAR_EQUAL:
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_GREATER_GREATER_EQUAL:
    case TokenKind::T_GREATER_GREATER:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_LESS_EQUAL_GREATER:
    case TokenKind::T_AMP_AMP:
    case TokenKind::T_BAR_BAR:
    case TokenKind::T_LESS_LESS:
    case TokenKind::T_LESS_LESS_EQUAL:
    case TokenKind::T_PLUS_PLUS:
    case TokenKind::T_MINUS_MINUS:
    case TokenKind::T_COMMA:
      yyconsume();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_literal_operator_id() {
  if (!match(TokenKind::T_OPERATOR)) return false;

  if (match(TokenKind::T_USER_DEFINED_STRING_LITERAL)) return true;

  if (!parse_string_literal_seq()) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_template_declaration(DeclarationAST*& yyast) {
  if (!parse_template_head()) return false;

  if (yytoken() == TokenKind::T_CONCEPT) {
    parse_concept_definition();
  } else {
    DeclarationAST* declaration = nullptr;

    if (!parse_declaration(declaration)) parse_error("expected a declaration");
  }

  return true;
}

bool Parser::parse_template_head() {
  if (!match(TokenKind::T_TEMPLATE)) return false;

  if (!match(TokenKind::T_LESS)) return false;

  if (!match(TokenKind::T_GREATER)) {
    if (!parse_template_parameter_list())
      parse_error("expected a template parameter");

    expect(TokenKind::T_GREATER);
  }

  parse_requires_clause();

  return true;
}

bool Parser::parse_template_parameter_list() {
  if (!parse_template_parameter()) return false;

  while (match(TokenKind::T_COMMA)) {
    if (!parse_template_parameter())
      parse_error("expected a template parameter");
  }

  return true;
}

bool Parser::parse_requires_clause() {
  if (!match(TokenKind::T_REQUIRES)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_constraint_logical_or_expression(expression)) return false;

  return true;
}

bool Parser::parse_constraint_logical_or_expression(ExpressionAST*& yyast) {
  if (!parse_constraint_logical_and_expression(yyast)) return false;

  while (match(TokenKind::T_BAR_BAR)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constraint_logical_and_expression(expression))
      parse_error("expected a requirement expression");
  }

  return true;
}

bool Parser::parse_constraint_logical_and_expression(ExpressionAST*& yyast) {
  if (!parse_primary_expression(yyast)) return false;

  while (match(TokenKind::T_AMP_AMP)) {
    ExpressionAST* expression = nullptr;

    if (!parse_primary_expression(expression))
      parse_error("expected an expression");
  }

  return true;
}

bool Parser::parse_template_parameter() {
  const auto start = yycursor;

  if (parse_type_parameter() &&
      (yytoken() == TokenKind::T_COMMA || yytoken() == TokenKind::T_GREATER))
    return true;

  yyrewind(start);

  return parse_parameter_declaration();
}

bool Parser::parse_type_parameter() {
  if (parse_template_type_parameter()) return true;

  if (parse_typename_type_parameter()) return true;

  return parse_constraint_type_parameter();
}

bool Parser::parse_typename_type_parameter() {
  if (!parse_type_parameter_key()) return false;

  const auto saved = yycursor;

  if ((yytoken() == TokenKind::T_IDENTIFIER &&
       yytoken(1) == TokenKind::T_EQUAL) ||
      yytoken() == TokenKind::T_EQUAL) {
    const auto has_identifier = match(TokenKind::T_IDENTIFIER);

    expect(TokenKind::T_EQUAL);

    if (!parse_type_id()) parse_error("expected a type id");

    return true;
  }

  const auto has_tripled_dot = match(TokenKind::T_DOT_DOT_DOT);

  const auto has_identifier = match(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_template_type_parameter() {
  const auto start = yycursor;

  if (!parse_template_head()) {
    yyrewind(start);
    return false;
  }

  if (!parse_type_parameter_key()) parse_error("expected a type parameter");

  const auto saved = yycursor;

  if ((yytoken() == TokenKind::T_IDENTIFIER &&
       yytoken(1) == TokenKind::T_EQUAL) ||
      yytoken() == TokenKind::T_EQUAL) {
    const auto has_identifier = match(TokenKind::T_IDENTIFIER);

    expect(TokenKind::T_EQUAL);

    if (!parse_id_expression()) parse_error("expected an id-expression");

    return true;
  }

  const auto has_tripled_dot = match(TokenKind::T_DOT_DOT_DOT);

  const auto has_identifier = match(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_constraint_type_parameter() {
  if (!parse_type_constraint()) return false;

  const auto saved = yycursor;

  if ((yytoken() == TokenKind::T_IDENTIFIER &&
       yytoken(1) == TokenKind::T_EQUAL) ||
      yytoken() == TokenKind::T_EQUAL) {
    const auto has_identifier = match(TokenKind::T_IDENTIFIER);

    expect(TokenKind::T_EQUAL);

    if (!parse_type_id())
      return false;  // ### FIXME: parse_error("expected a type id");

    return true;
  }

  const auto has_tripled_dot = match(TokenKind::T_DOT_DOT_DOT);

  const auto has_identifier = match(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_type_parameter_key() {
  if (!match(TokenKind::T_CLASS) && !match(TokenKind::T_TYPENAME)) return false;

  return true;
}

bool Parser::parse_type_constraint() {
  const auto start = yycursor;

  if (!parse_nested_name_specifier()) yyrewind(start);

  if (!parse_concept_name()) {
    yyrewind(start);
    return false;
  }

  if (match(TokenKind::T_LESS)) {
    if (!parse_template_argument_list())
      parse_error("expected a template argument");

    expect(TokenKind::T_GREATER);
  }

  return true;
}

bool Parser::parse_simple_template_id() {
  Name name;
  return parse_simple_template_id(name);
}

bool Parser::parse_simple_template_id(Name& name) {
  if (!parse_template_name(name)) return false;

  if (!match(TokenKind::T_LESS)) return false;

  if (!match(TokenKind::T_GREATER)) {
    if (!parse_template_argument_list()) return false;

    if (!match(TokenKind::T_GREATER)) return false;
  }

  name = control->getTemplateId(name);

  return true;
}

bool Parser::parse_template_id() {
  if (yytoken() == TokenKind::T_OPERATOR) {
    const auto start = yycursor;

    if (!parse_literal_operator_id()) {
      yyrewind(start);

      if (!parse_operator_function_id()) return false;
    }

    if (!match(TokenKind::T_LESS)) return false;

    if (!match(TokenKind::T_GREATER)) {
      if (!parse_template_argument_list()) return false;

      if (!match(TokenKind::T_GREATER)) return false;
    }

    return true;
  }

  return parse_simple_template_id();
}

bool Parser::parse_template_argument_list() {
  TemplArgContext templArgContext(this);

  if (!parse_template_argument()) return false;

  match(TokenKind::T_DOT_DOT_DOT);

  while (match(TokenKind::T_COMMA)) {
    if (!parse_template_argument()) {
      // parse_error("expected a template argument"); // ### FIXME
    }

    match(TokenKind::T_DOT_DOT_DOT);
  }

  return true;
}

bool Parser::parse_template_argument() {
  const auto start = yycursor;

  auto it = template_arguments_.find(start);

  if (it != template_arguments_.end()) {
    yyrewind(get<0>(it->second));
    return get<1>(it->second);
  }

  auto check = [&]() -> bool {
    auto tk = yytoken();

    return tk == TokenKind::T_COMMA || tk == TokenKind::T_GREATER ||
           tk == TokenKind::T_DOT_DOT_DOT;
  };

  const auto saved = yycursor;

  if (parse_type_id() && check()) {
    template_arguments_.emplace(start, std::make_tuple(yycursor, true));
    return true;
  }

  yyrewind(saved);

  ExpressionAST* expression = nullptr;

  const auto parsed = parse_template_argument_constant_expression(expression);

  if (parsed && check()) {
    template_arguments_.emplace(start, std::make_tuple(yycursor, true));
    return true;
  }

  template_arguments_.emplace(start, std::make_tuple(yycursor, false));

  return false;
}

bool Parser::parse_constraint_expression(ExpressionAST*& yyast) {
  return parse_logical_or_expression(yyast, false);
}

bool Parser::parse_deduction_guide(DeclarationAST*& yyast) {
  const auto has_explicit_spec = parse_explicit_specifier();

  if (!parse_template_name()) return false;

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_parameter_declaration_clause())
      parse_error("expected a parameter declaration");

    expect(TokenKind::T_RPAREN);
  }

  if (!match(TokenKind::T_MINUS_GREATER)) return false;

  if (!parse_simple_template_id()) parse_error("expected a template id");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_concept_definition() {
  if (!match(TokenKind::T_CONCEPT)) return false;

  if (!parse_concept_name()) parse_error("expected a concept name");

  expect(TokenKind::T_EQUAL);

  ExpressionAST* expression = nullptr;

  if (!parse_constraint_expression(expression))
    parse_error("expected a constraint expression");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_concept_name() {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_typename_specifier() {
  if (!match(TokenKind::T_TYPENAME)) return false;

  if (!parse_nested_name_specifier()) return false;

  const auto after_nested_name_specifier = yycursor;

  const auto has_template = match(TokenKind::T_TEMPLATE);

  if (parse_simple_template_id()) return true;

  yyrewind(after_nested_name_specifier);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_explicit_instantiation(DeclarationAST*& yyast) {
  const auto has_extern = match(TokenKind::T_EXTERN);

  if (!match(TokenKind::T_TEMPLATE)) return false;

  if (yytoken() == TokenKind::T_LESS) return false;

  DeclarationAST* declaration = nullptr;

  if (!parse_declaration(declaration)) parse_error("expected a declaration");

  return true;
}

bool Parser::parse_explicit_specialization(DeclarationAST*& yyast) {
  if (!match(TokenKind(TokenKind::T_TEMPLATE))) return false;

  if (!match(TokenKind::T_LESS)) return false;

  if (!match(TokenKind::T_GREATER)) return false;

  DeclarationAST* declaration = nullptr;

  if (!parse_declaration(declaration)) parse_error("expected a declaration");

  return true;
}

bool Parser::parse_try_block(StatementAST*& yyast) {
  if (!match(TokenKind::T_TRY)) return false;

  StatementAST* statement = nullptr;

  if (!parse_compound_statement(statement))
    parse_error("expected a compound statement");

  if (!parse_handler_seq()) parse_error("expected an exception handler");

  return true;
}

bool Parser::parse_function_try_block() {
  if (!match(TokenKind::T_TRY)) return false;

  if (yytoken() != TokenKind::T_LBRACE) {
    if (!parse_ctor_initializer()) parse_error("expected a ctor initializer");
  }

  StatementAST* statement = nullptr;

  if (!parse_compound_statement(statement))
    parse_error("expected a compound statement");

  if (!parse_handler_seq()) parse_error("expected an exception handler");

  return true;
}

bool Parser::parse_handler() {
  StatementAST* s1 = nullptr;

  if (!match(TokenKind::T_CATCH)) return false;

  expect(TokenKind::T_LPAREN);

  if (!parse_exception_declaration())
    parse_error("expected an exception declaration");

  expect(TokenKind::T_RPAREN);

  StatementAST* statement = nullptr;

  if (!parse_compound_statement(statement))
    parse_error("expected a compound statement");

  return true;
}

bool Parser::parse_handler_seq() {
  if (yytoken() != TokenKind::T_CATCH) return false;

  while (yytoken() == TokenKind::T_CATCH) {
    parse_handler();
  }

  return true;
}

bool Parser::parse_exception_declaration() {
  if (match(TokenKind::T_DOT_DOT_DOT)) return true;

  parse_attribute_specifier_seq();

  if (!parse_type_specifier_seq()) parse_error("expected a type specifier");

  if (yytoken() == TokenKind::T_RPAREN) return true;

  const auto before_declarator = yycursor;

  if (!parse_declarator()) {
    yyrewind(before_declarator);

    if (!parse_abstract_declarator()) yyrewind(before_declarator);
  }

  return true;
}

bool Parser::parse_noexcept_specifier() {
  if (match(TokenKind::T_THROW)) {
    expect(TokenKind::T_LPAREN);
    expect(TokenKind::T_RPAREN);
    return true;
  }

  if (!match(TokenKind::T_NOEXCEPT)) return false;

  if (match(TokenKind::T_LPAREN)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constant_expression(expression))
      parse_error("expected a declaration");

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_identifier_list() {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  while (match(TokenKind::T_COMMA)) {
    expect(TokenKind::T_IDENTIFIER);
  }

  return true;
}

}  // namespace cxx
