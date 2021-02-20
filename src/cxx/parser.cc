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

#include <cxx/control.h>
#include <cxx/parser.h>
#include <cxx/token.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <forward_list>
#include <iostream>
#include <unordered_map>
#include <variant>

namespace cxx {

std::pair<FunctionDeclaratorAST*, bool> getFunctionDeclaratorHelper(
    DeclaratorAST* declarator) {
  if (!declarator) return std::make_pair(nullptr, false);

  if (auto n = dynamic_cast<NestedDeclaratorAST*>(declarator->coreDeclarator)) {
    auto [fundecl, done] = getFunctionDeclaratorHelper(n->declarator);

    if (done) return std::make_pair(fundecl, done);
  }

  std::vector<DeclaratorModifierAST*> modifiers;

  for (auto it = declarator->modifiers; it; it = it->next)
    modifiers.push_back(it->value);

  for (auto it = rbegin(modifiers); it != rend(modifiers); ++it) {
    auto modifier = *it;

    if (auto decl = dynamic_cast<FunctionDeclaratorAST*>(modifier))
      return std::make_pair(decl, true);

    return std::make_pair(nullptr, true);
  }

  return std::make_pair(nullptr, false);
}

FunctionDeclaratorAST* getFunctionDeclarator(DeclaratorAST* declarator) {
  return get<0>(getFunctionDeclaratorHelper(declarator));
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

const Token& Parser::LA(int n) const {
  return unit->tokenAt(SourceLocation(cursor_ + n));
}

bool Parser::operator()(TranslationUnit* unit, UnitAST*& ast) {
  return parse(unit, ast);
}

bool Parser::parse(TranslationUnit* u, UnitAST*& ast) {
  unit = u;
  control = unit->control();
  cursor_ = 1;

  pool = u->arena();

  module_id = control->getIdentifier("module");
  import_id = control->getIdentifier("import");
  final_id = control->getIdentifier("final");
  override_id = control->getIdentifier("override");

  auto parsed = parse_translation_unit(ast);

  return parsed;
}

bool Parser::parse_id(const Identifier* id) {
  SourceLocation location;
  if (!match(TokenKind::T_IDENTIFIER, location)) return false;
  return unit->identifier(location) == id;
}

bool Parser::parse_nospace() {
  const auto& tk = unit->tokenAt(currentLocation());
  return !tk.leadingSpace() && !tk.startOfLine();
}

bool Parser::parse_greater_greater() {
  const auto saved = currentLocation();
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_GREATER))
    return true;
  rewind(saved);
  return false;
}

bool Parser::parse_greater_greater_equal() {
  const auto saved = currentLocation();
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_EQUAL))
    return true;
  rewind(saved);
  return false;
}

bool Parser::parse_greater_equal() {
  const auto saved = currentLocation();
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_EQUAL))
    return true;
  rewind(saved);
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
  unit->setTokenKind(SourceLocation(cursor_ - 1), TokenKind::T_IMPORT);
  return true;
}

bool Parser::parse_module_keyword() {
  if (!module_unit) return false;

  if (match(TokenKind::T_MODULE)) return true;

  if (!parse_id(module_id)) return false;

  unit->setTokenKind(SourceLocation(cursor_ - 1), TokenKind::T_MODULE);
  return true;
}

bool Parser::parse_final() { return parse_id(final_id); }

bool Parser::parse_override() { return parse_id(override_id); }

bool Parser::parse_typedef_name(NameAST*& yyast) {
  const auto start = currentLocation();

  if (parse_simple_template_id(yyast)) return true;

  rewind(start);

  return parse_name_id(yyast);
}

bool Parser::parse_class_name(NameAST*& yyast) {
  const auto start = currentLocation();

  if (parse_simple_template_id(yyast)) return true;

  rewind(start);

  return parse_name_id(yyast);
}

bool Parser::parse_name_id(NameAST*& yyast) {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = new (pool) SimpleNameAST();
  yyast = ast;

  ast->identifierLoc = identifierLoc;

  return true;
}

bool Parser::parse_enum_name(NameAST*& yyast) {
  if (!match(TokenKind::T_IDENTIFIER)) return false;
  return true;
}

bool Parser::parse_template_name(NameAST*& yyast) {
  if (!parse_name_id(yyast)) return false;

  return true;
}

bool Parser::parse_literal(ExpressionAST*& yyast) {
  switch (TokenKind(LA())) {
    case TokenKind::T_CHARACTER_LITERAL: {
      auto ast = new (pool) CharLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();

      return true;
    }

    case TokenKind::T_TRUE:
    case TokenKind::T_FALSE: {
      auto ast = new (pool) BoolLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();

      return true;
    }

    case TokenKind::T_INTEGER_LITERAL: {
      auto ast = new (pool) IntLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();

      return true;
    }

    case TokenKind::T_FLOATING_POINT_LITERAL: {
      auto ast = new (pool) FloatLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();

      return true;
    }

    case TokenKind::T_NULLPTR: {
      auto ast = new (pool) NullptrLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();

      return true;
    }

    case TokenKind::T_USER_DEFINED_STRING_LITERAL: {
      auto ast = new (pool) UserDefinedStringLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();

      return true;
    }

    case TokenKind::T_STRING_LITERAL: {
      List<SourceLocation>* stringLiterals = nullptr;

      parse_string_literal_seq(stringLiterals);

      auto ast = new (pool) StringLiteralExpressionAST();
      yyast = ast;

      ast->stringLiteralList = stringLiterals;

      return true;
    }

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
  const auto start = currentLocation();
  const auto has_export = match(TokenKind::T_EXPORT);
  const auto is_module = parse_id(module_id);
  rewind(start);
  return is_module;
}

bool Parser::parse_module_unit(UnitAST*& yyast) {
  module_unit = true;

  if (!parse_module_head()) return false;

  parse_global_module_fragment();

  if (!parse_module_declaration()) return false;

  List<DeclarationAST*>* declarationList = nullptr;

  parse_declaration_seq(declarationList);

  parse_private_module_fragment();

  expect(TokenKind::T_EOF_SYMBOL);

  return true;
}

bool Parser::parse_top_level_declaration_seq(UnitAST*& yyast) {
  auto ast = new (pool) TranslationUnitAST();
  yyast = ast;

  module_unit = false;

  bool skipping = false;

  auto it = &ast->declarationList;

  while (LA()) {
    const auto saved = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_declaration(declaration)) {
      if (declaration) {
        *it = new (pool) List(declaration);
        it = &(*it)->next;
      }

      skipping = false;
    } else {
      parse_skip_top_level_declaration(skipping);

      if (currentLocation() == saved) consumeToken();
    }
  }

  return true;
}

bool Parser::parse_skip_top_level_declaration(bool& skipping) {
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
  return true;
}

bool Parser::parse_declaration_seq(List<DeclarationAST*>*& yyast) {
  bool skipping = false;

  auto it = &yyast;

  while (LA()) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    if (parse_maybe_module()) break;

    auto saved = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_declaration(declaration)) {
      if (declaration) {
        *it = new (pool) List(declaration);
        it = &(*it)->next;
      }
      skipping = false;
    } else {
      parse_skip_declaration(skipping);

      if (currentLocation() == saved) consumeToken();
    }
  }

  return true;
}

bool Parser::parse_skip_declaration(bool& skipping) {
  if (LA().is(TokenKind::T_RBRACE)) return false;
  if (LA().is(TokenKind::T_MODULE)) return false;
  if (module_unit && LA().is(TokenKind::T_EXPORT)) return false;
  if (LA().is(TokenKind::T_IMPORT)) return false;
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
  return true;
}

bool Parser::parse_primary_expression(ExpressionAST*& yyast) {
  SourceLocation thisLoc;

  if (match(TokenKind::T_THIS, thisLoc)) {
    auto ast = new (pool) ThisExpressionAST();
    yyast = ast;
    ast->thisLoc = thisLoc;
    return true;
  }

  if (parse_literal(yyast)) return true;

  if (LA().is(TokenKind::T_LBRACKET)) return parse_lambda_expression(yyast);

  if (LA().is(TokenKind::T_REQUIRES)) return parse_requires_expression(yyast);

  if (LA().is(TokenKind::T_LPAREN)) {
    const auto saved = currentLocation();

    if (parse_fold_expression(yyast)) return true;

    rewind(saved);

    SourceLocation lparenLoc = consumeToken();

    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) return false;

    SourceLocation rparenLoc;

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

    auto ast = new (pool) NestedExpressionAST();
    yyast = ast;

    ast->lparenLoc = lparenLoc;
    ast->expression = expression;
    ast->rparenLoc = rparenLoc;

    return true;
  }

  NameAST* name = nullptr;

  if (!parse_id_expression(name)) return false;

  auto ast = new (pool) IdExpressionAST();
  yyast = ast;

  ast->name = name;

  return true;
}

bool Parser::parse_id_expression(NameAST*& yyast) {
  const auto start = currentLocation();

  if (parse_qualified_id(yyast)) return true;

  rewind(start);

  yyast = nullptr;

  return parse_unqualified_id(yyast);
}

bool Parser::parse_maybe_template_id(NameAST*& yyast) {
  const auto start = currentLocation();

  const auto blockErrors = unit->blockErrors(true);

  auto template_id = parse_template_id(yyast);

  const auto& tk = LA();

  unit->blockErrors(blockErrors);

  if (!template_id) return false;
  switch (TokenKind(tk)) {
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
      yyast = nullptr;
      return false;
  }  // switch
}

bool Parser::parse_unqualified_id(NameAST*& yyast) {
  const auto start = currentLocation();

  if (parse_maybe_template_id(yyast)) return true;

  rewind(start);

  SourceLocation tildeLoc;

  if (match(TokenKind::T_TILDE, tildeLoc)) {
    SpecifierAST* decltypeSpecifier = nullptr;

    if (parse_decltype_specifier(decltypeSpecifier)) {
      auto decltypeName = new (pool) DecltypeNameAST();
      decltypeName->decltypeSpecifier = decltypeSpecifier;

      auto ast = new (pool) DestructorNameAST();
      yyast = ast;

      ast->name = decltypeName;

      return true;
    }

    NameAST* name = nullptr;

    if (!parse_type_name(name)) return false;

    auto ast = new (pool) DestructorNameAST();
    yyast = ast;

    ast->name = name;

    return true;
  }

  if (LA().is(TokenKind::T_OPERATOR)) {
    if (parse_operator_function_id(yyast)) return true;

    rewind(start);

    if (parse_conversion_function_id(yyast)) return true;

    rewind(start);

    return parse_literal_operator_id(yyast);
  }

  return parse_name_id(yyast);
}

bool Parser::parse_qualified_id(NameAST*& yyast) {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

  SourceLocation templateLoc;

  const auto has_template = match(TokenKind::T_TEMPLATE, templateLoc);

  NameAST* name = nullptr;

  const auto hasName = parse_unqualified_id(name);

  if (hasName || templateLoc) {
    auto ast = new (pool) QualifiedNameAST();
    yyast = ast;

    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->templateLoc = templateLoc;
    ast->name = name;

    if (!hasName) parse_error("expected a template name");

    return true;
  }

  return false;
}

bool Parser::parse_start_of_nested_name_specifier(NameAST*& yyast,
                                                  SourceLocation& scopeLoc) {
  yyast = nullptr;

  if (match(TokenKind::T_COLON_COLON, scopeLoc)) return true;

  SpecifierAST* decltypeSpecifier = nullptr;

  if (parse_decltype_specifier(decltypeSpecifier) &&
      match(TokenKind::T_COLON_COLON, scopeLoc))
    return true;

  const auto start = currentLocation();

  if (parse_name_id(yyast) && match(TokenKind::T_COLON_COLON, scopeLoc))
    return true;

  rewind(start);

  if (parse_simple_template_id(yyast) &&
      match(TokenKind::T_COLON_COLON, scopeLoc))
    return true;

  return false;
}

bool Parser::parse_nested_name_specifier(NestedNameSpecifierAST*& yyast) {
  const auto start = currentLocation();

  auto it = nested_name_specifiers_.find(start);

  if (it != nested_name_specifiers_.end()) {
    auto [cursor, ast, parsed] = it->second;
    rewind(cursor);
    yyast = ast;
    return parsed;
  }

  struct Context {
    Parser* p;
    SourceLocation start;
    NestedNameSpecifierAST* ast = nullptr;
    bool parsed = false;

    Context(Parser* p) : p(p), start(p->currentLocation()) {}

    ~Context() {
      p->nested_name_specifiers_.emplace(
          start, std::make_tuple(p->currentLocation(), ast, parsed));
    }
  };

  Context context(this);

  NameAST* name = nullptr;
  SourceLocation scopeLoc;

  if (!parse_start_of_nested_name_specifier(name, scopeLoc)) return false;

  auto ast = new (pool) NestedNameSpecifierAST();
  context.ast = ast;
  yyast = ast;

  auto nameIt = &ast->nameList;

  if (!name)
    ast->scopeLoc = scopeLoc;
  else {
    *nameIt = new (pool) List(name);
    nameIt = &(*nameIt)->next;
  }

  while (true) {
    const auto saved = currentLocation();

    NameAST* name = nullptr;

    if (parse_name_id(name) && match(TokenKind::T_COLON_COLON)) {
      *nameIt = new (pool) List(name);
      nameIt = &(*nameIt)->next;
    } else {
      rewind(saved);

      const auto has_template = match(TokenKind::T_TEMPLATE);

      if (parse_simple_template_id(name) && match(TokenKind::T_COLON_COLON)) {
        *nameIt = new (pool) List(name);
        nameIt = &(*nameIt)->next;
      } else {
        rewind(saved);
        break;
      }
    }
  }

  context.parsed = true;

  return true;
}

bool Parser::parse_lambda_expression(ExpressionAST*& yyast) {
  LambdaIntroducerAST* lambdaIntroducer = nullptr;

  if (!parse_lambda_introducer(lambdaIntroducer)) return false;

  auto ast = new (pool) LambdaExpressionAST();
  yyast = ast;

  ast->lambdaIntroducer = lambdaIntroducer;

  if (match(TokenKind::T_LESS, ast->lessLoc)) {
    if (!parse_template_parameter_list(ast->templateParameterList))
      parse_error("expected a template paramter");

    expect(TokenKind::T_GREATER, ast->greaterLoc);

    parse_requires_clause();
  }

  if (LA().isNot(TokenKind::T_LBRACE)) {
    if (!parse_lambda_declarator(ast->lambdaDeclarator))
      parse_error("expected lambda declarator");
  }

  if (!parse_compound_statement(ast->statement))
    parse_error("expected a compound statement");

  return true;
}

bool Parser::parse_lambda_introducer(LambdaIntroducerAST*& yyast) {
  SourceLocation lbracketLoc;

  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  SourceLocation rbracketLoc;

  if (!match(TokenKind::T_RBRACKET, rbracketLoc)) {
    if (!parse_lambda_capture()) parse_error("expected a lambda capture");

    expect(TokenKind::T_RBRACKET, rbracketLoc);
  }

  auto ast = new (pool) LambdaIntroducerAST();
  yyast = ast;

  ast->lbracketLoc = lbracketLoc;
  ast->rbracketLoc = rbracketLoc;

  return true;
}

bool Parser::parse_lambda_declarator(LambdaDeclaratorAST*& yyast) {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = new (pool) LambdaDeclaratorAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_parameter_declaration_clause(ast->parameterDeclarationClause))
      parse_error("expected a parameter declaration clause");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  DeclSpecs specs;

  parse_decl_specifier_seq(ast->declSpecifierList, specs);

  parse_noexcept_specifier();

  parse_attribute_specifier_seq(ast->attributeList);

  parse_trailing_return_type(ast->trailingReturnType);

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
  const auto start = currentLocation();

  if (!match(TokenKind::T_AMP) && !match(TokenKind::T_EQUAL)) return false;

  if (LA().isNot(TokenKind::T_COMMA) && LA().isNot(TokenKind::T_RBRACKET)) {
    rewind(start);
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
  const auto start = currentLocation();

  if (parse_init_capture()) return true;

  rewind(start);

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

    InitializerAST* initializer = nullptr;

    if (!parse_initializer(initializer)) return false;

    return true;
  }

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  InitializerAST* initializer = nullptr;

  if (!parse_initializer(initializer)) return false;

  return true;
}

bool Parser::parse_fold_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (match(TokenKind::T_DOT_DOT_DOT)) {
    SourceLocation opLoc;
    TokenKind op = TokenKind::T_EOF_SYMBOL;

    if (!parse_fold_operator(opLoc, op)) parse_error("expected fold operator");

    ExpressionAST* expression;

    if (!parse_cast_expression(expression))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);

    return true;
  }

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) return false;

  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  if (!parse_fold_operator(opLoc, op)) return false;

  if (!match(TokenKind::T_DOT_DOT_DOT)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    SourceLocation opLoc;
    TokenKind op = TokenKind::T_EOF_SYMBOL;

    if (!parse_fold_operator(opLoc, op))
      parse_error("expected a fold operator");

    ExpressionAST* rhs = nullptr;

    if (!parse_cast_expression(rhs)) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_fold_operator(SourceLocation& loc, TokenKind& op) {
  switch (TokenKind(LA())) {
    case TokenKind::T_GREATER: {
      loc = currentLocation();

      if (parse_greater_greater_equal()) {
        op = TokenKind::T_GREATER_GREATER_EQUAL;
        return true;
      }

      if (parse_greater_greater()) {
        op = TokenKind::T_GREATER_GREATER;
        return true;
      }

      if (parse_greater_equal()) {
        op = TokenKind::T_GREATER_EQUAL;
        return true;
      }

      op = TokenKind::T_GREATER;

      consumeToken();

      return true;
    }

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
    case TokenKind::T_MINUS_GREATER_STAR: {
      op = LA().kind();
      consumeToken();
      return true;
    }

    default:
      return false;
  }  // switch
}

bool Parser::parse_requires_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_REQUIRES)) return false;

  if (LA().isNot(TokenKind::T_LBRACE)) {
    if (!parse_requirement_parameter_list())
      parse_error("expected a requirement parameter");
  }

  if (!parse_requirement_body()) parse_error("expected a requirement body");

  return true;
}

bool Parser::parse_requirement_parameter_list() {
  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

    if (!parse_parameter_declaration_clause(parameterDeclarationClause))
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

  while (LA()) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    const auto before_requirement = currentLocation();

    if (parse_requirement()) {
      skipping = false;
    } else {
      if (!skipping) parse_error("expected a requirement");
      skipping = true;
      if (currentLocation() == before_requirement) consumeToken();
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

  const auto after_typename = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(after_typename);

  NameAST* name = nullptr;

  if (!parse_type_name(name)) parse_error("expected a type name");

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
  if (!parse_start_of_postfix_expression(yyast)) return false;

  while (true) {
    const auto saved = currentLocation();
    if (parse_member_expression(yyast))
      continue;
    else if (parse_subscript_expression(yyast))
      continue;
    else if (parse_call_expression(yyast))
      continue;
    else if (parse_postincr_expression(yyast))
      continue;
    else {
      rewind(saved);
      break;
    }
  }

  return true;
}

bool Parser::parse_start_of_postfix_expression(ExpressionAST*& yyast) {
  const auto start = currentLocation();

  if (parse_cpp_cast_expression(yyast)) return true;

  if (parse_typeid_expression(yyast)) return true;

  if (parse_builtin_call_expression(yyast)) return true;

  if (parse_typename_expression(yyast)) return true;

  if (parse_cpp_type_cast_expression(yyast)) return true;

  rewind(start);
  return parse_primary_expression(yyast);
}

bool Parser::parse_member_expression(ExpressionAST*& yyast) {
  SourceLocation accessLoc;

  if (!match(TokenKind::T_DOT, accessLoc) &&
      !match(TokenKind::T_MINUS_GREATER, accessLoc))
    return false;

  auto ast = new (pool) MemberExpressionAST();
  ast->baseExpression = yyast;
  ast->accessLoc = accessLoc;

  yyast = ast;

  match(TokenKind::T_TEMPLATE, ast->templateLoc);

  if (!parse_id_expression(ast->name)) parse_error("expected a member name");

  return true;
}

bool Parser::parse_subscript_expression(ExpressionAST*& yyast) {
  SourceLocation lbracketLoc;

  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  auto ast = new (pool) SubscriptExpressionAST();
  ast->baseExpression = yyast;
  ast->lbracketLoc = lbracketLoc;

  yyast = ast;

  if (!parse_expr_or_braced_init_list(ast->indexExpression))
    parse_error("expected an expression");

  expect(TokenKind::T_RBRACKET, ast->rbracketLoc);

  return true;
}

bool Parser::parse_call_expression(ExpressionAST*& yyast) {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = new (pool) CallExpressionAST();
  ast->baseExpression = yyast;
  ast->lparenLoc = lparenLoc;

  yyast = ast;

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression_list(ast->expressionList))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  return true;
}

bool Parser::parse_postincr_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_MINUS_MINUS) && !match(TokenKind::T_PLUS_PLUS))
    return false;

  return true;
}

bool Parser::parse_cpp_cast_head(SourceLocation& castLoc) {
  switch (TokenKind(LA())) {
    case TokenKind::T_CONST_CAST:
    case TokenKind::T_DYNAMIC_CAST:
    case TokenKind::T_REINTERPRET_CAST:
    case TokenKind::T_STATIC_CAST:
      castLoc = consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_cpp_cast_expression(ExpressionAST*& yyast) {
  SourceLocation castLoc;

  if (!parse_cpp_cast_head(castLoc)) return false;

  auto ast = new (pool) CppCastExpressionAST();
  yyast = ast;

  ast->castLoc = castLoc;

  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  expect(TokenKind::T_GREATER, ast->greaterLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_expression(ast->expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

bool Parser::parse_cpp_type_cast_expression(ExpressionAST*& yyast) {
  const auto start = currentLocation();

  SpecifierAST* typeSpecifier = nullptr;

  DeclSpecs specs;

  if (!parse_simple_type_specifier(typeSpecifier, specs)) return false;

  // ### prefer function calls to cpp-cast expressions for now.
  if (LA().is(TokenKind::T_LPAREN) &&
      dynamic_cast<NamedTypeSpecifierAST*>(typeSpecifier)) {
    rewind(start);
    return false;
  }

  BracedInitListAST* bracedInitList = nullptr;

  if (parse_braced_init_list(bracedInitList)) return true;

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    List<ExpressionAST*>* expressionList = nullptr;

    if (!parse_expression_list(expressionList)) return false;

    if (!match(TokenKind::T_RPAREN)) return false;
  }

  return true;
}

bool Parser::parse_typeid_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_TYPEID)) return false;

  expect(TokenKind::T_LPAREN);

  const auto saved = currentLocation();

  TypeIdAST* typeId = nullptr;

  if (parse_type_id(typeId) && match(TokenKind::T_RPAREN)) {
    //
  } else {
    rewind(saved);

    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) parse_error("expected an expression");

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

bool Parser::parse_typename_expression(ExpressionAST*& yyast) {
  SpecifierAST* typenameSpecifier = nullptr;

  if (!parse_typename_specifier(typenameSpecifier)) return false;

  BracedInitListAST* bracedInitList = nullptr;

  if (parse_braced_init_list(bracedInitList)) return true;

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    List<ExpressionAST*>* expressionList = nullptr;

    if (!parse_expression_list(expressionList)) return false;

    if (!match(TokenKind::T_RPAREN)) return false;
  }

  return true;
}

bool Parser::parse_builtin_function_1() {
  switch (TokenKind(LA())) {
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
      consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_builtin_function_2() {
  switch (TokenKind(LA())) {
    case TokenKind::T___IS_BASE_OF:
    case TokenKind::T___IS_CONSTRUCTIBLE:
    case TokenKind::T___IS_CONVERTIBLE_TO:
    case TokenKind::T___IS_NOTHROW_ASSIGNABLE:
    case TokenKind::T___IS_NOTHROW_CONSTRUCTIBLE:
    case TokenKind::T___IS_SAME:
    case TokenKind::T___IS_TRIVIALLY_ASSIGNABLE:
    case TokenKind::T___IS_TRIVIALLY_CONSTRUCTIBLE:
    case TokenKind::T___REFERENCE_BINDS_TO_TEMPORARY:
      consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_builtin_call_expression(ExpressionAST*& yyast) {
  if (parse_builtin_function_1()) {
    expect(TokenKind::T_LPAREN);

    TypeIdAST* typeId = nullptr;

    if (!parse_type_id(typeId)) parse_error("expected a type id");

    expect(TokenKind::T_RPAREN);

    return true;
  }

  if (!parse_builtin_function_2()) return false;

  expect(TokenKind::T_LPAREN);

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) parse_error("expected a type id");

  expect(TokenKind::T_COMMA);

  TypeIdAST* secondTypeId = nullptr;

  if (!parse_type_id(secondTypeId)) parse_error("expected a type id");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_expression_list(List<ExpressionAST*>*& yyast) {
  return parse_initializer_list(yyast);
}

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
  SourceLocation opLoc;

  if (!parse_unary_operator(opLoc)) return false;

  auto ast = new (pool) UnaryExpressionAST();
  yyast = ast;

  ast->opLoc = opLoc;

  if (!parse_cast_expression(ast->expression))
    parse_error("expected an expression");

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

  const auto after_sizeof_op = currentLocation();

  TypeIdAST* typeId = nullptr;

  if (match(TokenKind::T_LPAREN) && parse_type_id(typeId) &&
      match(TokenKind::T_RPAREN)) {
    return true;
  }

  rewind(after_sizeof_op);

  ExpressionAST* expression = nullptr;

  if (!parse_unary_expression(expression))
    parse_error("expected an expression");

  return true;
}

bool Parser::parse_alignof_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_ALIGNOF) && !match(TokenKind::T___ALIGNOF))
    return false;

  expect(TokenKind::T_LPAREN);

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) parse_error("expected a type id");

  expect(TokenKind::T_RPAREN);

  return true;
}

bool Parser::parse_unary_operator(SourceLocation& opLoc) {
  switch (TokenKind(LA())) {
    case TokenKind::T_STAR:
    case TokenKind::T_AMP:
    case TokenKind::T_PLUS:
    case TokenKind::T_MINUS:
    case TokenKind::T_EXCLAIM:
    case TokenKind::T_TILDE:
    case TokenKind::T_MINUS_MINUS:
    case TokenKind::T_PLUS_PLUS:
      opLoc = consumeToken();
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
  const auto start = currentLocation();

  SourceLocation scopeLoc;

  match(TokenKind::T_COLON_COLON);

  SourceLocation newLoc;

  if (!match(TokenKind::T_NEW, newLoc)) {
    rewind(start);
    return false;
  }

  auto ast = new (pool) NewExpressionAST();
  yyast = ast;

  ast->scopeLoc = scopeLoc;
  ast->newLoc = newLoc;

  const auto after_new_op = currentLocation();

  if (!parse_new_placement()) rewind(after_new_op);

  const auto after_new_placement = currentLocation();

  TypeIdAST* typeId = nullptr;

  if (match(TokenKind::T_LPAREN) && parse_type_id(typeId) &&
      match(TokenKind::T_RPAREN)) {
    const auto saved = currentLocation();

    if (!parse_new_initializer(ast->newInitalizer)) rewind(saved);

    return true;
  }

  rewind(after_new_placement);

  if (!parse_new_type_id(ast->typeId)) return false;

  const auto saved = currentLocation();

  if (!parse_new_initializer(ast->newInitalizer)) rewind(saved);

  return true;
}

bool Parser::parse_new_placement() {
  if (!match(TokenKind::T_LPAREN)) return false;

  List<ExpressionAST*>* expressionList = nullptr;

  if (!parse_expression_list(expressionList)) return false;

  if (!match(TokenKind::T_RPAREN)) return false;

  return true;
}

bool Parser::parse_new_type_id(NewTypeIdAST*& yyast) {
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList)) return false;

  auto ast = new (pool) NewTypeIdAST();
  yyast = ast;

  ast->typeSpecifierList = typeSpecifierList;

  const auto saved = currentLocation();

  if (!parse_new_declarator()) rewind(saved);

  return true;
}

bool Parser::parse_new_declarator() {
  PtrOperatorAST* ptrOp = nullptr;

  if (parse_ptr_operator(ptrOp)) {
    auto saved = currentLocation();

    if (!parse_new_declarator()) rewind(saved);

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

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  while (match(TokenKind::T_LBRACKET)) {
    if (!match(TokenKind::T_RBRACKET)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(expression))
        parse_error("expected an expression");

      expect(TokenKind::T_RBRACKET);
    }

    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);
  }

  return true;
}

bool Parser::parse_new_initializer(NewInitializerAST*& yyast) {
  BracedInitListAST* bracedInitList = nullptr;

  if (parse_braced_init_list(bracedInitList)) return true;

  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = new (pool) NewParenInitializerAST();
  yyast = ast;

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression_list(ast->expressionList)) return false;

    if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) return false;
  }

  return true;
}

bool Parser::parse_delete_expression(ExpressionAST*& yyast) {
  const auto start = currentLocation();

  const auto has_scope_op = match(TokenKind::T_COLON_COLON);

  if (!match(TokenKind::T_DELETE)) {
    rewind(start);
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
  const auto start = currentLocation();

  if (parse_cast_expression_helper(yyast)) return true;

  rewind(start);

  return parse_unary_expression(yyast);
}

bool Parser::parse_cast_expression_helper(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_LPAREN)) return false;

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) return false;

  if (!match(TokenKind::T_RPAREN)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) return false;

  return true;
}

bool Parser::parse_binary_operator(SourceLocation& loc, TokenKind& tk,
                                   bool templArg) {
  const auto start = currentLocation();

  loc = start;
  tk = TokenKind::T_EOF_SYMBOL;

  switch (TokenKind(LA())) {
    case TokenKind::T_GREATER: {
      if (parse_greater_greater()) {
        if (templArg && templArgDepth >= 2) {
          rewind(start);
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
        rewind(start);
        return false;
      }

      consumeToken();
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
      tk = LA().kind();
      consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_binary_expression(ExpressionAST*& yyast, bool templArg) {
  if (!parse_cast_expression(yyast)) return false;

  const auto saved = currentLocation();

  if (!parse_binary_expression_helper(yyast, Prec::kLogicalOr, templArg))
    rewind(saved);

  return true;
}

bool Parser::parse_lookahead_binary_operator(SourceLocation& loc, TokenKind& tk,
                                             bool templArg) {
  const auto saved = currentLocation();

  const auto has_binop = parse_binary_operator(loc, tk, templArg);

  rewind(saved);

  return has_binop;
}

bool Parser::parse_binary_expression_helper(ExpressionAST*& yyast, Prec minPrec,
                                            bool templArg) {
  bool parsed = false;

  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  while (parse_lookahead_binary_operator(opLoc, op, templArg) &&
         prec(op) >= minPrec) {
    const auto saved = currentLocation();

    ExpressionAST* rhs = nullptr;

    parse_binary_operator(opLoc, op, templArg);

    if (!parse_cast_expression(rhs)) {
      rewind(saved);
      break;
    }

    parsed = true;

    SourceLocation nextOpLoc;
    TokenKind nextOp = TokenKind::T_EOF_SYMBOL;

    while (parse_lookahead_binary_operator(nextOpLoc, nextOp, templArg) &&
           prec(nextOp) > prec(op)) {
      if (!parse_binary_expression_helper(rhs, prec(op), templArg)) {
        break;
      }
    }

    auto ast = new (pool) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = rhs;

    yyast = ast;
  }

  return parsed;
}

bool Parser::parse_logical_or_expression(ExpressionAST*& yyast, bool templArg) {
  return parse_binary_expression(yyast, templArg);
}

bool Parser::parse_conditional_expression(ExpressionAST*& yyast,
                                          bool templArg) {
  if (!parse_logical_or_expression(yyast, templArg)) return false;

  SourceLocation questionLoc;

  if (match(TokenKind::T_QUESTION, questionLoc)) {
    auto ast = new (pool) ConditionalExpressionAST();
    ast->condition = yyast;
    ast->questionLoc = questionLoc;

    yyast = ast;

    if (!parse_expression(ast->iftrueExpression))
      parse_error("expected an expression");

    expect(TokenKind::T_COLON, ast->colonLoc);

    if (templArg) {
      if (!parse_conditional_expression(ast->iffalseExpression, templArg)) {
        parse_error("expected an expression");
      }
    } else if (!parse_assignment_expression(ast->iffalseExpression)) {
      parse_error("expected an expression");
    }
  }

  return true;
}

bool Parser::parse_yield_expression(ExpressionAST*& yyast) {
  if (!match(TokenKind::T_CO_YIELD)) return false;

  if (LA().is(TokenKind::T_LBRACE)) {
    BracedInitListAST* bracedInitList = nullptr;

    if (!parse_braced_init_list(bracedInitList))
      parse_error("expected a braced initializer");
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

  const auto saved = currentLocation();

  if (!parse_assignment_expression(expression)) rewind(saved);

  return true;
}

bool Parser::parse_assignment_expression(ExpressionAST*& yyast) {
  if (parse_yield_expression(yyast)) return true;

  if (parse_throw_expression(yyast)) return true;

  if (!parse_conditional_expression(yyast, false)) return false;

  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  if (parse_assignment_operator(opLoc, op)) {
    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression))
      parse_error("expected an expression");

    auto ast = new (pool) AssignmentExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = expression;

    yyast = ast;
  }

  return true;
}

bool Parser::parse_assignment_operator(SourceLocation& loc, TokenKind& op) {
  switch (TokenKind(LA())) {
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
    case TokenKind::T_GREATER_GREATER_EQUAL: {
      op = LA().kind();
      loc = consumeToken();
      return true;
    }

    case TokenKind::T_GREATER: {
      loc = currentLocation();
      if (!parse_greater_greater_equal()) return false;
      op = TokenKind::T_GREATER_GREATER_EQUAL;
      return true;
    }

    default:
      return false;
  }  // switch
}

bool Parser::parse_expression(ExpressionAST*& yyast) {
  if (!parse_assignment_expression(yyast)) return false;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    ExpressionAST* expression = nullptr;

    if (!parse_assignment_expression(expression))
      parse_error("expected an expression");
    else {
      auto ast = new (pool) BinaryExpressionAST();
      ast->leftExpression = yyast;
      ast->opLoc = commaLoc;
      ast->rightExpression = expression;
      yyast = ast;
    }
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
  const bool has_extension = match(TokenKind::T___EXTENSION__);

  List<AttributeAST*>* attributes = nullptr;

  const bool has_attribute_specifiers =
      parse_attribute_specifier_seq(attributes);

  const auto start = currentLocation();

  switch (TokenKind(LA())) {
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
      rewind(start);
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
      if (LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_COLON)) {
        return parse_labeled_statement(yyast);
      }

      if (parse_declaration_statement(yyast)) return true;

      rewind(start);

      return parse_expression_statement(yyast);
  }  // switch
}

bool Parser::parse_init_statement(StatementAST*& yyast) {
  if (LA().is(TokenKind::T_RPAREN)) return false;

  auto saved = currentLocation();

  DeclarationAST* declaration = nullptr;

  if (parse_simple_declaration(declaration, false)) return true;

  rewind(saved);

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_condition(ExpressionAST*& yyast) {
  const auto start = currentLocation();

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  List<SpecifierAST*>* declSpecifierList = nullptr;

  DeclSpecs specs;

  if (parse_decl_specifier_seq(declSpecifierList, specs)) {
    DeclaratorAST* declarator = nullptr;

    if (parse_declarator(declarator)) {
      InitializerAST* initializer = nullptr;

      if (parse_brace_or_equal_initializer(initializer)) return true;
    }
  }

  rewind(start);

  return parse_expression(yyast);
}

bool Parser::parse_labeled_statement(StatementAST*& yyast) {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  auto ast = new (pool) LabeledStatementAST();
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->colonLoc = colonLoc;
  ast->statement = statement;

  return true;
}

bool Parser::parse_case_statement(StatementAST*& yyast) {
  SourceLocation caseLoc;

  if (!match(TokenKind::T_CASE, caseLoc)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_constant_expression(expression))
    parse_error("expected an expression");

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  auto ast = new (pool) CaseStatementAST();
  yyast = ast;

  ast->caseLoc = caseLoc;
  ast->expression = expression;
  ast->colonLoc = colonLoc;
  ast->statement = statement;

  return true;
}

bool Parser::parse_default_statement(StatementAST*& yyast) {
  SourceLocation defaultLoc;

  if (!match(TokenKind::T_DEFAULT)) return false;

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  auto ast = new (pool) DefaultStatementAST();
  yyast = ast;

  ast->defaultLoc = defaultLoc;
  ast->colonLoc = colonLoc;
  ast->statement = statement;

  return true;
}

bool Parser::parse_expression_statement(StatementAST*& yyast) {
  SourceLocation semicolonLoc;

  ExpressionAST* expression = nullptr;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    if (!parse_expression(expression)) return false;

    expect(TokenKind::T_SEMICOLON, semicolonLoc);
  }

  auto ast = new (pool) ExpressionStatementAST;
  yyast = ast;

  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

bool Parser::parse_compound_statement(StatementAST*& yyast) {
  bool skipping = false;

  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  auto ast = new (pool) CompoundStatementAST();
  yyast = ast;

  ast->lbraceLoc = lbraceLoc;

  auto it = &ast->statementList;

  while (const auto& tk = LA()) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    StatementAST* statement = nullptr;

    if (parse_statement(statement)) {
      *it = new (pool) List(statement);
      it = &(*it)->next;
      skipping = false;
    } else {
      parse_skip_statement(skipping);
    }
  }

  if (!expect(TokenKind::T_RBRACE, ast->rbraceLoc)) return false;

  return true;
}

bool Parser::parse_skip_statement(bool& skipping) {
  if (!LA()) return false;
  if (LA().is(TokenKind::T_RBRACE)) return false;
  if (!skipping) parse_error("expected a statement");
  for (; LA(); consumeToken()) {
    if (LA().is(TokenKind::T_SEMICOLON)) break;
    if (LA().is(TokenKind::T_LBRACE)) break;
    if (LA().is(TokenKind::T_RBRACE)) break;
  }
  skipping = true;
  return true;
}

bool Parser::parse_if_statement(StatementAST*& yyast) {
  SourceLocation ifLoc;

  if (!match(TokenKind::T_IF, ifLoc)) return false;

  auto ast = new (pool) IfStatementAST();
  yyast = ast;

  const auto has_constexpr = match(TokenKind::T_CONSTEXPR, ast->constexprLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  auto saved = currentLocation();

  if (!parse_init_statement(ast->initializer)) rewind(saved);

  if (!parse_condition(ast->condition)) parse_error("expected a condition");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  if (!parse_statement(ast->statement)) parse_error("expected a statement");

  if (!match(TokenKind::T_ELSE)) return true;

  if (!parse_statement(ast->elseStatement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_switch_statement(StatementAST*& yyast) {
  SourceLocation switchLoc;

  if (!match(TokenKind::T_SWITCH, switchLoc)) return false;

  auto ast = new (pool) SwitchStatementAST();
  yyast = ast;

  ast->switchLoc = switchLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  const auto saved = currentLocation();

  if (!parse_init_statement(ast->initializer)) rewind(saved);

  if (!parse_condition(ast->condition)) parse_error("expected a condition");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  parse_statement(ast->statement);

  return true;
}

bool Parser::parse_while_statement(StatementAST*& yyast) {
  SourceLocation whileLoc;

  if (!match(TokenKind::T_WHILE, whileLoc)) return false;

  auto ast = new (pool) WhileStatementAST();
  yyast = ast;

  ast->whileLoc = whileLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_condition(ast->condition)) parse_error("expected a condition");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  if (!parse_statement(ast->statement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_do_statement(StatementAST*& yyast) {
  SourceLocation doLoc;

  if (!match(TokenKind::T_DO, doLoc)) return false;

  auto ast = new (pool) DoStatementAST();
  yyast = ast;

  ast->doLoc = doLoc;

  if (!parse_statement(ast->statement)) parse_error("expected a statement");

  expect(TokenKind::T_WHILE, ast->whileLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_expression(ast->expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

bool Parser::parse_for_range_statement(StatementAST*& yyast) {
  SourceLocation forLoc;

  if (!match(TokenKind::T_FOR, forLoc)) return false;

  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  const auto saved = currentLocation();

  StatementAST* initializer = nullptr;

  if (!parse_init_statement(initializer)) rewind(saved);

  DeclarationAST* rangeDeclaration = nullptr;

  if (!parse_for_range_declaration(rangeDeclaration)) return false;

  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  auto ast = new (pool) ForRangeStatementAST();
  yyast = ast;

  ast->forLoc = forLoc;
  ast->lparenLoc = lparenLoc;
  ast->initializer = initializer;
  ast->colonLoc = colonLoc;

  if (!parse_for_range_initializer(ast->rangeInitializer))
    parse_error("expected for-range intializer");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  if (!parse_statement(ast->statement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_for_statement(StatementAST*& yyast) {
  StatementAST* s1 = nullptr;
  StatementAST* s2 = nullptr;
  ExpressionAST* e1 = nullptr;
  ExpressionAST* e2 = nullptr;

  SourceLocation forLoc;

  if (!match(TokenKind::T_FOR, forLoc)) return false;

  auto ast = new (pool) ForStatementAST();
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_init_statement(ast->initializer))
    parse_error("expected a statement");

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    if (!parse_condition(ast->condition)) parse_error("expected a condition");

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression(ast->expression))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  if (!parse_statement(ast->statement)) parse_error("expected a statement");

  return true;
}

bool Parser::parse_for_range_declaration(DeclarationAST*& yyast) {
  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  List<SpecifierAST*>* declSpecifierList = nullptr;

  DeclSpecs specs;

  if (!parse_decl_specifier_seq(declSpecifierList, specs)) return false;

  const auto& tk = LA();

  if (tk.is(TokenKind::T_LBRACKET) ||
      ((tk.is(TokenKind::T_AMP) || tk.is(TokenKind::T_AMP_AMP)) &&
       LA(1).is(TokenKind::T_LBRACKET))) {
    SourceLocation refLoc;

    parse_ref_qualifier(refLoc);

    if (!match(TokenKind::T_LBRACKET)) return false;

    if (!parse_identifier_list()) parse_error("expected an identifier");

    expect(TokenKind::T_RBRACKET);
  } else {
    DeclaratorAST* declarator = nullptr;

    if (!parse_declarator(declarator)) return false;
  }

  return true;
}

bool Parser::parse_for_range_initializer(ExpressionAST*& yyast) {
  return parse_expr_or_braced_init_list(yyast);
}

bool Parser::parse_break_statement(StatementAST*& yyast) {
  SourceLocation breakLoc;

  if (!match(TokenKind::T_BREAK, breakLoc)) return false;

  auto ast = new (pool) BreakStatementAST();
  yyast = ast;

  ast->breakLoc = breakLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_continue_statement(StatementAST*& yyast) {
  SourceLocation continueLoc;

  if (!match(TokenKind::T_CONTINUE, continueLoc)) return false;

  auto ast = new (pool) ContinueStatementAST();
  yyast = ast;

  ast->continueLoc = continueLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_return_statement(StatementAST*& yyast) {
  SourceLocation returnLoc;

  if (!match(TokenKind::T_RETURN, returnLoc)) return false;

  auto ast = new (pool) ReturnStatementAST();
  yyast = ast;

  ast->returnLoc = returnLoc;

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    if (!parse_expr_or_braced_init_list(ast->expression))
      parse_error("expected an expression or ';'");

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  return true;
}

bool Parser::parse_goto_statement(StatementAST*& yyast) {
  SourceLocation gotoLoc;

  if (!match(TokenKind::T_GOTO, gotoLoc)) return false;

  auto ast = new (pool) GotoStatementAST();
  yyast = ast;

  ast->gotoLoc = gotoLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_coroutine_return_statement(StatementAST*& yyast) {
  SourceLocation coreturnLoc;

  if (!match(TokenKind::T_CO_RETURN, coreturnLoc)) return false;

  auto ast = new (pool) CoroutineReturnStatementAST();
  yyast = ast;

  ast->coreturnLoc = coreturnLoc;

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    if (!parse_expr_or_braced_init_list(ast->expression))
      parse_error("expected an expression");

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  return true;
}

bool Parser::parse_declaration_statement(StatementAST*& yyast) {
  DeclarationAST* declaration = nullptr;

  if (!parse_block_declaration(declaration, false)) return false;

  auto ast = new (pool) DeclarationStatementAST();
  yyast = ast;

  ast->declaration = declaration;

  return true;
}

bool Parser::parse_maybe_module() {
  if (!module_unit) return false;

  const auto start = currentLocation();

  match(TokenKind::T_EXPORT);

  const auto is_module = parse_module_keyword();

  rewind(start);

  return is_module;
}

bool Parser::parse_declaration(DeclarationAST*& yyast) {
  if (LA().is(TokenKind::T_RBRACE)) return false;

  auto start = currentLocation();

  if (LA().is(TokenKind::T_SEMICOLON)) return parse_empty_declaration(yyast);

  rewind(start);
  if (parse_explicit_instantiation(yyast)) return true;

  rewind(start);
  if (parse_explicit_specialization(yyast)) return true;

  rewind(start);
  if (parse_template_declaration(yyast)) return true;

  rewind(start);
  if (parse_deduction_guide(yyast)) return true;

  rewind(start);
  if (parse_export_declaration(yyast)) return true;

  rewind(start);
  if (parse_linkage_specification(yyast)) return true;

  rewind(start);
  if (parse_namespace_definition(yyast)) return true;

  rewind(start);
  if (parse_attribute_declaration(yyast)) return true;

  rewind(start);
  if (parse_module_import_declaration(yyast)) return true;

  rewind(start);
  return parse_block_declaration(yyast, true);
}

bool Parser::parse_block_declaration(DeclarationAST*& yyast, bool fundef) {
  const auto start = currentLocation();

  const auto& tk = LA();

  if (parse_asm_declaration(yyast)) return true;

  rewind(start);
  if (parse_namespace_alias_definition(yyast)) return true;

  rewind(start);
  if (parse_using_directive(yyast)) return true;

  rewind(start);
  if (parse_alias_declaration(yyast)) return true;

  rewind(start);
  if (parse_using_declaration(yyast)) return true;

  rewind(start);
  if (parse_using_enum_declaration(yyast)) return true;

  rewind(start);
  if (parse_static_assert_declaration(yyast)) return true;

  rewind(start);
  if (parse_opaque_enum_declaration(yyast)) return true;

  rewind(start);
  return parse_simple_declaration(yyast, fundef);
}

bool Parser::parse_alias_declaration(DeclarationAST*& yyast) {
  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  SourceLocation equalLoc;

  if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

  TypeIdAST* typeId = nullptr;

  if (!parse_defining_type_id(typeId)) parse_error("expected a type id");

  SourceLocation semicolonLoc;

  expect(TokenKind::T_SEMICOLON, semicolonLoc);

  auto ast = new (pool) AliasDeclarationAST;
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->identifierLoc = identifierLoc;
  ast->attributeList = attributes;
  ast->equalLoc = equalLoc;
  ast->typeId = typeId;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

bool Parser::parse_simple_declaration(DeclarationAST*& yyast, bool fundef) {
  const bool has_extension = match(TokenKind::T___EXTENSION__);

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  SourceLocation semicolonLoc;

  if (match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    if (attributes) {
      auto ast = new (pool) AttributeDeclarationAST();
      yyast = ast;
      ast->attributeList = attributes;
      ast->semicolonLoc = semicolonLoc;
      return true;
    }

    auto ast = new (pool) EmptyDeclarationAST();
    yyast = ast;
    ast->semicolonLoc = semicolonLoc;
    return true;
  }

  const auto after_attributes = currentLocation();

  DeclSpecs specs;

  List<SpecifierAST*>* declSpecifierList = nullptr;

  if (!parse_decl_specifier_seq_no_typespecs(declSpecifierList, specs))
    rewind(after_attributes);

  auto after_decl_specs = currentLocation();

  IdDeclaratorAST* declaratorId = nullptr;

  if (parse_declarator_id(declaratorId)) {
    ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

    if (parse_parameters_and_qualifiers(parametersAndQualifiers)) {
      if (match(TokenKind::T_SEMICOLON)) return true;

      StatementAST* functionBody = nullptr;

      if (fundef && parse_function_definition_body(functionBody)) {
        auto declarator = new (pool) DeclaratorAST();
        declarator->coreDeclarator = declaratorId;

        auto functionDeclarator = new (pool) FunctionDeclaratorAST();
        functionDeclarator->parametersAndQualifiers = parametersAndQualifiers;

        declarator->modifiers =
            new (pool) List<DeclaratorModifierAST*>(functionDeclarator);

        auto ast = new (pool) FunctionDefinitionAST();
        yyast = ast;

        ast->declSpecifierList = declSpecifierList;
        ast->declarator = declarator;
        ast->functionBody = functionBody;

        return true;
      }
    }
  }

  rewind(after_decl_specs);

  auto lastDeclSpecifier = &declSpecifierList;

  while (*lastDeclSpecifier) {
    lastDeclSpecifier = &(*lastDeclSpecifier)->next;
  }

  if (!parse_decl_specifier_seq(*lastDeclSpecifier, specs))
    rewind(after_decl_specs);

  after_decl_specs = currentLocation();

  if (specs.has_complex_typespec &&
      match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    auto ast = new (pool) SimpleDeclarationAST();
    yyast = ast;

    ast->declSpecifierList = declSpecifierList;
    ast->semicolonLoc = semicolonLoc;
    return true;
  }

  if (!specs.has_typespec()) return false;

  SourceLocation refLoc;

  parse_ref_qualifier(refLoc);

  if (match(TokenKind::T_LBRACKET)) {
    if (parse_identifier_list() && match(TokenKind::T_RBRACKET)) {
      InitializerAST* initializer = nullptr;

      if (parse_initializer(initializer) && match(TokenKind::T_SEMICOLON))
        return true;
    }
  }

  rewind(after_decl_specs);

  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  const auto after_declarator = currentLocation();

  if (fundef && getFunctionDeclarator(declarator)) {
    StatementAST* functionBody = nullptr;

    if (parse_function_definition_body(functionBody)) {
      auto ast = new (pool) FunctionDefinitionAST();
      yyast = ast;

      ast->declSpecifierList = declSpecifierList;
      ast->declarator = declarator;
      ast->functionBody = functionBody;

      return true;
    }

    rewind(after_declarator);
  }

  InitializerAST* initializer = nullptr;

  if (!parse_declarator_initializer(initializer)) rewind(after_declarator);

  List<DeclaratorAST*>* declaratorList = nullptr;

  auto declIt = &declaratorList;

  *declIt = new (pool) List(declarator);
  declIt = &(*declIt)->next;

  while (match(TokenKind::T_COMMA)) {
    DeclaratorAST* declarator = nullptr;

    if (!parse_init_declarator(declarator)) return false;

    *declIt = new (pool) List(declarator);
    declIt = &(*declIt)->next;
  }

  if (!match(TokenKind::T_SEMICOLON)) return false;

  auto ast = new (pool) SimpleDeclarationAST();
  yyast = ast;

  ast->declSpecifierList = declSpecifierList;
  ast->declaratorList = declaratorList;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

bool Parser::parse_function_definition_body(StatementAST*& yyast) {
  if (parse_requires_clause()) {
    //
  } else {
    parse_virt_specifier_seq();
  }

  return parse_function_body(yyast);
}

bool Parser::parse_static_assert_declaration(DeclarationAST*& yyast) {
  SourceLocation staticAssertLoc;

  if (!match(TokenKind::T_STATIC_ASSERT, staticAssertLoc)) return false;

  auto ast = new (pool) StaticAssertDeclarationAST();
  yyast = ast;

  ast->staticAssertLoc = staticAssertLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_constant_expression(ast->expression))
    parse_error("expected an expression");

  if (match(TokenKind::T_COMMA, ast->commaLoc)) {
    if (!parse_string_literal_seq(ast->stringLiteralList))
      parse_error("expected a string literal");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_string_literal_seq(List<SourceLocation>*& yyast) {
  auto it = &yyast;

  SourceLocation loc;

  if (!match(TokenKind::T_STRING_LITERAL, loc)) return false;

  *it = new (pool) List(loc);
  it = &(*it)->next;

  while (match(TokenKind::T_STRING_LITERAL, loc)) {
    *it = new (pool) List(loc);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_empty_declaration(DeclarationAST*& yyast) {
  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool) EmptyDeclarationAST();
  yyast = ast;

  ast->semicolonLoc = semicolonLoc;

  return true;
}

bool Parser::parse_attribute_declaration(DeclarationAST*& yyast) {
  List<AttributeAST*>* attributes = nullptr;

  if (!parse_attribute_specifier_seq(attributes)) return false;

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool) AttributeDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

bool Parser::parse_decl_specifier(SpecifierAST*& yyast, DeclSpecs& specs) {
  switch (TokenKind(LA())) {
    case TokenKind::T_TYPEDEF: {
      auto ast = new (pool) TypedefSpecifierAST();
      yyast = ast;
      ast->typedefLoc = consumeToken();
      return true;
    }

    case TokenKind::T_FRIEND: {
      auto ast = new (pool) FriendSpecifierAST();
      yyast = ast;
      ast->friendLoc = consumeToken();
      return true;
    }

    case TokenKind::T_CONSTEXPR: {
      auto ast = new (pool) ConstexprSpecifierAST();
      yyast = ast;
      ast->constexprLoc = consumeToken();
      return true;
    }

    case TokenKind::T_CONSTEVAL: {
      auto ast = new (pool) ConstevalSpecifierAST();
      yyast = ast;
      ast->constevalLoc = consumeToken();
      return true;
    }

    case TokenKind::T_CONSTINIT: {
      auto ast = new (pool) ConstinitSpecifierAST();
      yyast = ast;
      ast->constinitLoc = consumeToken();
      return true;
    }

    case TokenKind::T_INLINE:
    case TokenKind::T___INLINE:
    case TokenKind::T___INLINE__: {
      auto ast = new (pool) InlineSpecifierAST();
      yyast = ast;
      ast->inlineLoc = consumeToken();
      return true;
    }

    default:
      if (parse_storage_class_specifier(yyast)) return true;

      if (parse_function_specifier(yyast)) return true;

      if (!specs.no_typespecs)
        return parse_defining_type_specifier(yyast, specs);

      return false;
  }  // switch
}

bool Parser::parse_decl_specifier_seq(List<SpecifierAST*>*& yyast,
                                      DeclSpecs& specs) {
  auto it = &yyast;

  specs.no_typespecs = false;

  SpecifierAST* specifier = nullptr;

  if (!parse_decl_specifier(specifier, specs)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    *it = new (pool) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

bool Parser::parse_decl_specifier_seq_no_typespecs(List<SpecifierAST*>*& yyast,
                                                   DeclSpecs& specs) {
  auto it = &yyast;

  specs.no_typespecs = true;

  SpecifierAST* specifier = nullptr;

  if (!parse_decl_specifier(specifier, specs)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    *it = new (pool) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

bool Parser::parse_storage_class_specifier(SpecifierAST*& yyast) {
  SourceLocation loc;

  if (match(TokenKind::T_STATIC)) {
    auto ast = new (pool) StaticSpecifierAST();
    yyast = ast;
    ast->staticLoc = loc;
    return true;
  } else if (match(TokenKind::T_THREAD_LOCAL)) {
    auto ast = new (pool) ThreadLocalSpecifierAST();
    yyast = ast;
    ast->threadLocalLoc = loc;
    return true;
  } else if (match(TokenKind::T_EXTERN)) {
    auto ast = new (pool) ExternSpecifierAST();
    yyast = ast;
    ast->externLoc = loc;
    return true;
  } else if (match(TokenKind::T_MUTABLE)) {
    auto ast = new (pool) MutableSpecifierAST();
    yyast = ast;
    ast->mutableLoc = loc;
    return true;
  } else if (match(TokenKind::T___THREAD)) {
    auto ast = new (pool) ThreadSpecifierAST();
    yyast = ast;
    ast->threadLoc = consumeToken();
    return true;
  }

  return false;
}

bool Parser::parse_function_specifier(SpecifierAST*& yyast) {
  SourceLocation virtualLoc;
  if (match(TokenKind::T_VIRTUAL)) {
    auto ast = new (pool) SimpleSpecifierAST();
    yyast = ast;
    ast->specifierLoc = virtualLoc;

    return true;
  }

  return parse_explicit_specifier(yyast);
}

bool Parser::parse_explicit_specifier(SpecifierAST*& yyast) {
  SourceLocation explicitLoc;

  if (!match(TokenKind::T_EXPLICIT, explicitLoc)) return false;

  auto ast = new (pool) ExplicitSpecifierAST();
  yyast = ast;

  ast->explicitLoc = explicitLoc;

  if (match(TokenKind::T_LPAREN, ast->lparenLoc)) {
    if (!parse_constant_expression(ast->expression))
      parse_error("expected a expression");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  return true;
}

bool Parser::parse_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs) {
  if (parse_simple_type_specifier(yyast, specs)) return true;

  if (parse_elaborated_type_specifier(yyast, specs)) return true;

  if (parse_cv_qualifier(yyast)) return true;

  if (parse_typename_specifier(yyast)) {
    specs.has_named_typespec = true;
    return true;
  }

  return false;
}

bool Parser::parse_type_specifier_seq(List<SpecifierAST*>*& yyast) {
  auto it = &yyast;

  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  SpecifierAST* typeSpecifier = nullptr;

  if (!parse_type_specifier(typeSpecifier, specs)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  *it = new (pool) List(typeSpecifier);
  it = &(*it)->next;

  typeSpecifier = nullptr;

  while (LA()) {
    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_type_specifier(typeSpecifier, specs)) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    *it = new (pool) List(typeSpecifier);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_defining_type_specifier(SpecifierAST*& yyast,
                                           DeclSpecs& specs) {
  if (!specs.no_class_or_enum_specs) {
    const auto start = currentLocation();

    if (parse_enum_specifier(yyast)) {
      specs.has_complex_typespec = true;
      return true;
    }

    if (parse_class_specifier(yyast)) {
      specs.has_complex_typespec = true;
      return true;
    }
    rewind(start);
  }

  return parse_type_specifier(yyast, specs);
}

bool Parser::parse_defining_type_specifier_seq(List<SpecifierAST*>*& yyast,
                                               DeclSpecs& specs) {
  SpecifierAST* typeSpecifier = nullptr;

  if (!parse_defining_type_specifier(typeSpecifier, specs)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  while (LA()) {
    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_defining_type_specifier(typeSpecifier, specs)) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);
  }

  return true;
}

bool Parser::parse_simple_type_specifier(SpecifierAST*& yyast,
                                         DeclSpecs& specs) {
  const auto start = currentLocation();

  if (parse_named_type_specifier(yyast, specs)) return true;

  rewind(start);

  if (parse_placeholder_type_specifier_helper(yyast, specs)) return true;

  rewind(start);

  if (parse_primitive_type_specifier(yyast, specs)) return true;

  if (parse_underlying_type_specifier(yyast, specs)) return true;

  if (parse_atomic_type_specifier(yyast, specs)) return true;

  return parse_decltype_specifier_type_specifier(yyast, specs);
}

bool Parser::parse_named_type_specifier(SpecifierAST*& yyast,
                                        DeclSpecs& specs) {
  if (!parse_named_type_specifier_helper(yyast, specs)) return false;

  specs.has_named_typespec = true;

  return true;
}

bool Parser::parse_named_type_specifier_helper(SpecifierAST*& yyast,
                                               DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (parse_nested_name_specifier(nestedNameSpecifier)) {
    const auto after_nested_name_specifier = currentLocation();

    SourceLocation templateLoc;
    NameAST* name = nullptr;

    if (match(TokenKind::T_TEMPLATE, templateLoc) &&
        parse_simple_template_id(name)) {
      auto qualifiedId = new (pool) QualifiedNameAST();
      qualifiedId->nestedNameSpecifier = nestedNameSpecifier;
      qualifiedId->templateLoc = templateLoc;
      qualifiedId->name = name;

      auto ast = new (pool) NamedTypeSpecifierAST();
      yyast = ast;

      ast->name = name;

      return true;
    }

    rewind(after_nested_name_specifier);

    if (parse_type_name(name)) {
      auto ast = new (pool) NamedTypeSpecifierAST();
      yyast = ast;

      ast->name = name;

      return true;
    }

    rewind(after_nested_name_specifier);

    if (parse_template_name(name)) {
      auto ast = new (pool) NamedTypeSpecifierAST();
      yyast = ast;

      ast->name = name;

      return true;
    }
  }

  rewind(start);

  NameAST* name = nullptr;

  if (parse_type_name(name)) {
    auto ast = new (pool) NamedTypeSpecifierAST();
    yyast = ast;

    ast->name = name;
    return true;
  }

  rewind(start);

  if (parse_template_name(name)) {
    auto ast = new (pool) NamedTypeSpecifierAST();
    yyast = ast;

    ast->name = name;

    return true;
  }

  return false;
}

bool Parser::parse_placeholder_type_specifier_helper(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  if (!parse_placeholder_type_specifier(yyast)) return false;

  specs.has_placeholder_typespec = true;

  return true;
}

bool Parser::parse_decltype_specifier_type_specifier(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  if (!parse_decltype_specifier(yyast)) return false;

  specs.has_placeholder_typespec = true;

  return true;
}

bool Parser::parse_underlying_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) {
  if (specs.has_typespec()) return false;

  if (!match(TokenKind::T___UNDERLYING_TYPE)) return false;

  expect(TokenKind::T_LPAREN);

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) parse_error("expected type id");

  expect(TokenKind::T_RPAREN);

  specs.has_named_typespec = true;

  return true;
}

bool Parser::parse_atomic_type_specifier(SpecifierAST*& yyast,
                                         DeclSpecs& specs) {
  if (!specs.accepts_simple_typespec()) return false;

  SourceLocation atomicLoc;

  if (!match(TokenKind::T__ATOMIC, atomicLoc)) return false;

  auto ast = new (pool) AtomicTypeSpecifierAST();
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected type id");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  specs.has_simple_typespec = true;

  return true;
}

bool Parser::parse_primitive_type_specifier(SpecifierAST*& yyast,
                                            DeclSpecs& specs) {
  if (!specs.accepts_simple_typespec()) return false;

  switch (TokenKind(LA())) {
    case TokenKind::T_CHAR:
    case TokenKind::T_CHAR8_T:
    case TokenKind::T_CHAR16_T:
    case TokenKind::T_CHAR32_T:
    case TokenKind::T_WCHAR_T:
    case TokenKind::T_BOOL:
    case TokenKind::T_SHORT:
    case TokenKind::T_INT:
    case TokenKind::T___INT64:
    case TokenKind::T___INT128:
    case TokenKind::T_LONG:
    case TokenKind::T_SIGNED:
    case TokenKind::T_UNSIGNED: {
      auto ast = new (pool) IntegralTypeSpecifierAST();
      yyast = ast;
      ast->specifierLoc = consumeToken();
      specs.has_simple_typespec = true;
      return true;
    }

    case TokenKind::T_FLOAT:
    case TokenKind::T_DOUBLE:
    case TokenKind::T___FLOAT80:
    case TokenKind::T___FLOAT128: {
      auto ast = new (pool) FloatingPointTypeSpecifierAST();
      yyast = ast;
      ast->specifierLoc = consumeToken();
      specs.has_simple_typespec = true;
      return true;
    }

    case TokenKind::T_VOID: {
      auto ast = new (pool) VoidTypeSpecifierAST();
      yyast = ast;
      ast->voidLoc = consumeToken();
      specs.has_simple_typespec = true;
      return true;
    }

    case TokenKind::T___COMPLEX__: {
      auto ast = new (pool) ComplexTypeSpecifierAST();
      yyast = ast;
      ast->complexLoc = consumeToken();
      specs.has_simple_typespec = true;
      return true;
    }

    default:
      return false;
  }  // switch
}

bool Parser::parse_type_name(NameAST*& yyast) {
  const auto start = currentLocation();

  if (parse_class_name(yyast)) return true;

  rewind(start);

  if (parse_enum_name(yyast)) return true;

  rewind(start);

  return parse_typedef_name(yyast);
}

bool Parser::parse_elaborated_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) {
  // ### cleanup

  if (LA().is(TokenKind::T_ENUM)) return parse_elaborated_enum_specifier(yyast);

  SourceLocation classLoc;

  if (!parse_class_key(classLoc)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  const auto before_nested_name_specifier = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) {
    rewind(before_nested_name_specifier);

    NameAST* name = nullptr;

    if (parse_simple_template_id(name)) {
      specs.has_complex_typespec = true;
      return true;
    }

    rewind(before_nested_name_specifier);

    if (!match(TokenKind::T_IDENTIFIER)) return false;

    specs.has_complex_typespec = true;
    return true;
  }

  const auto after_nested_name_specifier = currentLocation();

  const bool has_template = match(TokenKind::T_TEMPLATE);

  NameAST* name = nullptr;

  if (parse_simple_template_id(name)) {
    specs.has_complex_typespec = true;
    return true;
  }

  if (has_template) {
    parse_error("expected a template-id");
    specs.has_complex_typespec = true;
    return true;
  }

  rewind(after_nested_name_specifier);

  if (!parse_name_id(name)) return false;

  specs.has_complex_typespec = true;

  return true;
}

bool Parser::parse_elaborated_enum_specifier(SpecifierAST*& yyast) {
  if (!match(TokenKind::T_ENUM)) return false;

  const auto saved = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(saved);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_decl_specifier_seq_no_typespecs(
    List<SpecifierAST*>*& yyast) {
  DeclSpecs specs;
  return parse_decl_specifier_seq_no_typespecs(yyast, specs);
}

bool Parser::parse_decltype_specifier(SpecifierAST*& yyast) {
  SourceLocation decltypeLoc;

  if (match(TokenKind::T_DECLTYPE, decltypeLoc) ||
      match(TokenKind::T___DECLTYPE, decltypeLoc) ||
      match(TokenKind::T___DECLTYPE__, decltypeLoc)) {
    SourceLocation lparenLoc;

    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    if (LA().is(TokenKind::T_AUTO)) return false;  // placeholder type specifier

    auto ast = new (pool) DecltypeSpecifierAST();
    yyast = ast;

    ast->decltypeLoc = decltypeLoc;
    ast->lparenLoc = lparenLoc;

    if (!parse_expression(ast->expression))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  SourceLocation typeofLoc;

  if (match(TokenKind::T___TYPEOF, typeofLoc) ||
      match(TokenKind::T___TYPEOF__, typeofLoc)) {
    auto ast = new (pool) TypeofSpecifierAST();
    yyast = ast;

    ast->typeofLoc = typeofLoc;

    expect(TokenKind::T_LPAREN, ast->lparenLoc);

    if (!parse_expression(ast->expression))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  return false;
}

bool Parser::parse_placeholder_type_specifier(SpecifierAST*& yyast) {
  parse_type_constraint();

  SourceLocation autoLoc;

  if (match(TokenKind::T_AUTO, autoLoc)) {
    auto ast = new (pool) AutoTypeSpecifierAST();
    yyast = ast;

    ast->autoLoc = autoLoc;

    return true;
  }

  SourceLocation decltypeLoc;

  if (match(TokenKind::T_DECLTYPE, decltypeLoc)) {
    SourceLocation lparenLoc;

    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    SourceLocation autoLoc;

    if (!match(TokenKind::T_AUTO, autoLoc)) return false;

    auto ast = new (pool) DecltypeAutoSpecifierAST();
    yyast = ast;

    ast->decltypeLoc = decltypeLoc;
    ast->lparenLoc = lparenLoc;
    ast->autoLoc = autoLoc;

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  return false;
}

bool Parser::parse_init_declarator(DeclaratorAST*& yyast) {
  if (!parse_declarator(yyast)) return false;

  const auto saved = currentLocation();

  InitializerAST* initializer = nullptr;

  if (!parse_declarator_initializer(initializer)) rewind(saved);

  return true;
}

bool Parser::parse_declarator_initializer(InitializerAST*& yyast) {
  if (parse_requires_clause()) return true;

  return parse_initializer(yyast);
}

bool Parser::parse_declarator(DeclaratorAST*& yyast) {
  const auto start = currentLocation();

  List<PtrOperatorAST*>* ptrOpList = nullptr;

  if (!parse_ptr_operator_seq(ptrOpList)) {
    rewind(start);

    ptrOpList = nullptr;
  }

  if (!parse_noptr_declarator(yyast, ptrOpList)) return false;

  return true;
}

bool Parser::parse_ptr_operator_seq(List<PtrOperatorAST*>*& yyast) {
  auto it = &yyast;

  PtrOperatorAST* ptrOp = nullptr;

  if (!parse_ptr_operator(ptrOp)) return false;

  *it = new (pool) List(ptrOp);
  it = &(*it)->next;

  ptrOp = nullptr;

  while (parse_ptr_operator(ptrOp)) {
    *it = new (pool) List(ptrOp);
    it = &(*it)->next;
    ptrOp = nullptr;
  }

  return true;
}

bool Parser::parse_core_declarator(CoreDeclaratorAST*& yyast) {
  IdDeclaratorAST* declaratorId = nullptr;

  if (parse_declarator_id(declaratorId)) {
    yyast = declaratorId;
    return true;
  }

  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  SourceLocation rparenLoc;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

  auto ast = new (pool) NestedDeclaratorAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->declarator = declarator;
  ast->rparenLoc = lparenLoc;

  return true;
}

bool Parser::parse_noptr_declarator(DeclaratorAST*& yyast,
                                    List<PtrOperatorAST*>* ptrOpLst) {
  CoreDeclaratorAST* coreDeclarator = nullptr;

  if (!parse_core_declarator(coreDeclarator)) return false;

  yyast = new (pool) DeclaratorAST();

  yyast->ptrOpList = ptrOpLst;

  yyast->coreDeclarator = coreDeclarator;

  auto it = &yyast->modifiers;

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  while (true) {
    const auto saved = currentLocation();

    SourceLocation lbracketLoc;

    if (match(TokenKind::T_LBRACKET, lbracketLoc)) {
      SourceLocation rbracketLoc;
      ExpressionAST* expression = nullptr;

      if (!match(TokenKind::T_RBRACKET, rbracketLoc)) {
        if (!parse_constant_expression(expression)) {
          rewind(saved);
          break;
        }

        if (!match(TokenKind::T_RBRACKET, rbracketLoc)) {
          rewind(saved);
          break;
        }
      }

      List<AttributeAST*>* attributes = nullptr;

      parse_attribute_specifier_seq(attributes);

      auto modifier = new (pool) ArrayDeclaratorAST();
      modifier->lbracketLoc = lbracketLoc;
      modifier->expression = expression;
      modifier->rbracketLoc = rbracketLoc;
      modifier->attributeList = attributes;

      *it = new (pool) List<DeclaratorModifierAST*>(modifier);

      it = &(*it)->next;

    } else if (parse_parameters_and_qualifiers(parametersAndQualifiers)) {
      auto modifier = new (pool) FunctionDeclaratorAST();

      modifier->parametersAndQualifiers = parametersAndQualifiers;

      parse_trailing_return_type(modifier->trailingReturnType);

      *it = new (pool) List<DeclaratorModifierAST*>(modifier);

      it = &(*it)->next;

      parametersAndQualifiers = nullptr;
    } else {
      rewind(saved);
      break;
    }
  }

  return true;
}

bool Parser::parse_parameters_and_qualifiers(
    ParametersAndQualifiersAST*& yyast) {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  SourceLocation rparenLoc;

  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_parameter_declaration_clause(parameterDeclarationClause))
      return false;

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;
  }

  auto ast = new (pool) ParametersAndQualifiersAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->parameterDeclarationClause = parameterDeclarationClause;
  ast->rparenLoc = rparenLoc;

  parse_cv_qualifier_seq(ast->cvQualifierList);

  parse_ref_qualifier(ast->refLoc);

  parse_noexcept_specifier();

  parse_attribute_specifier_seq(ast->attributeList);

  return true;
}

bool Parser::parse_cv_qualifier_seq(List<SpecifierAST*>*& yyast) {
  auto it = &yyast;

  SpecifierAST* specifier = nullptr;

  if (!parse_cv_qualifier(specifier)) return false;

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_cv_qualifier(specifier)) {
    *it = new (pool) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

bool Parser::parse_trailing_return_type(TrailingReturnTypeAST*& yyast) {
  SourceLocation minusGreaterLoc;

  if (!match(TokenKind::T_MINUS_GREATER, minusGreaterLoc)) return false;

  auto ast = new (pool) TrailingReturnTypeAST();
  yyast = ast;

  ast->minusGreaterLoc = minusGreaterLoc;

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  return true;
}

bool Parser::parse_ptr_operator(PtrOperatorAST*& yyast) {
  SourceLocation starLoc;

  if (match(TokenKind::T_STAR, starLoc)) {
    auto ast = new (pool) PointerOperatorAST();
    yyast = ast;

    ast->starLoc = starLoc;

    parse_attribute_specifier_seq(ast->attributeList);

    parse_cv_qualifier_seq(ast->cvQualifierList);

    return true;
  }

  SourceLocation refLoc;

  if (parse_ref_qualifier(refLoc)) {
    auto ast = new (pool) ReferenceOperatorAST();
    yyast = ast;

    ast->refLoc = refLoc;

    parse_attribute_specifier_seq(ast->attributeList);

    return true;
  }

  const auto saved = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (parse_nested_name_specifier(nestedNameSpecifier) &&
      match(TokenKind::T_STAR, starLoc)) {
    auto ast = new (pool) PtrToMemberOperatorAST();
    yyast = ast;

    ast->nestedNameSpecifier = nestedNameSpecifier;

    ast->starLoc = starLoc;

    parse_attribute_specifier_seq(ast->attributeList);

    parse_cv_qualifier_seq(ast->cvQualifierList);

    return true;
  }

  rewind(saved);

  return false;
}

bool Parser::parse_cv_qualifier(SpecifierAST*& yyast) {
  switch (TokenKind(LA())) {
    case TokenKind::T_CONST:
    case TokenKind::T_VOLATILE:
    case TokenKind::T___RESTRICT:
    case TokenKind::T___RESTRICT__: {
      auto ast = new (pool) CvQualifierAST();
      yyast = ast;

      ast->qualifierLoc = consumeToken();

      return true;
    }

    default:
      return false;
  }  // switch
}

bool Parser::parse_ref_qualifier(SourceLocation& refLoc) {
  switch (TokenKind(LA())) {
    case TokenKind::T_AMP:
    case TokenKind::T_AMP_AMP:
      consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_declarator_id(IdDeclaratorAST*& yyast) {
  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  NameAST* name = nullptr;

  if (!parse_id_expression(name)) return false;

  yyast = new (pool) IdDeclaratorAST();
  yyast->ellipsisLoc = ellipsisLoc;
  yyast->name = name;

  parse_attribute_specifier_seq(yyast->attributeList);

  return true;
}

bool Parser::parse_type_id(TypeIdAST*& yyast) {
  List<SpecifierAST*>* specifierList = nullptr;

  if (!parse_type_specifier_seq(specifierList)) return false;

  yyast = new (pool) TypeIdAST();

  yyast->typeSpecifierList = specifierList;

  const auto before_declarator = currentLocation();

  if (!parse_abstract_declarator(yyast->declarator)) rewind(before_declarator);

  return true;
}

bool Parser::parse_defining_type_id(TypeIdAST*& yyast) {
  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_defining_type_specifier_seq(typeSpecifierList, specs))
    return false;

  const auto before_declarator = currentLocation();

  DeclaratorAST* declarator = nullptr;

  if (!parse_abstract_declarator(declarator)) rewind(before_declarator);

  return true;
}

bool Parser::parse_abstract_declarator(DeclaratorAST*& yyast) {
  if (parse_abstract_pack_declarator()) return true;

  if (parse_ptr_abstract_declarator(yyast)) return true;

  const auto saved = currentLocation();

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;
  TrailingReturnTypeAST* trailingReturnType = nullptr;

  if (parse_parameters_and_qualifiers(parametersAndQualifiers) &&
      parse_trailing_return_type(trailingReturnType)) {
    auto functionDeclarator = new (pool) FunctionDeclaratorAST();
    functionDeclarator->parametersAndQualifiers = parametersAndQualifiers;
    functionDeclarator->trailingReturnType = trailingReturnType;

    auto ast = new (pool) DeclaratorAST();
    yyast = ast;

    ast->modifiers =
        new (pool) List<DeclaratorModifierAST*>(functionDeclarator);

    return true;
  }

  rewind(saved);

  if (!parse_noptr_abstract_declarator(yyast)) return false;

  const auto after_noptr_declarator = currentLocation();

  auto ast = new (pool) DeclaratorAST();
  yyast = ast;

  parametersAndQualifiers = nullptr;
  trailingReturnType = nullptr;

  if (parse_parameters_and_qualifiers(parametersAndQualifiers) &&
      parse_trailing_return_type(trailingReturnType)) {
    auto functionDeclarator = new (pool) FunctionDeclaratorAST();
    functionDeclarator->parametersAndQualifiers = parametersAndQualifiers;
    functionDeclarator->trailingReturnType = trailingReturnType;

    ast->modifiers =
        new (pool) List<DeclaratorModifierAST*>(functionDeclarator);
  } else {
    rewind(after_noptr_declarator);
  }

  return true;
}

bool Parser::parse_ptr_abstract_declarator(DeclaratorAST*& yyast) {
  List<PtrOperatorAST*>* ptrOpList = nullptr;

  if (!parse_ptr_operator_seq(ptrOpList)) return false;

  auto ast = new (pool) DeclaratorAST();
  yyast = ast;

  ast->ptrOpList = ptrOpList;

  const auto saved = currentLocation();

  if (!parse_noptr_abstract_declarator(yyast)) rewind(saved);

  return true;
}

bool Parser::parse_noptr_abstract_declarator(DeclaratorAST*& yyast) {
  if (!yyast) yyast = new (pool) DeclaratorAST();

  const auto start = currentLocation();

  DeclaratorAST* declarator = nullptr;

  if (match(TokenKind::T_LPAREN) && parse_ptr_abstract_declarator(declarator) &&
      match(TokenKind::T_RPAREN)) {
    auto nestedDeclarator = new (pool) NestedDeclaratorAST();

    nestedDeclarator->declarator = declarator;

    yyast->coreDeclarator = nestedDeclarator;
  } else {
    rewind(start);
  }

  const auto after_nested_declarator = currentLocation();

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  auto it = &yyast->modifiers;

  if (LA().is(TokenKind::T_LPAREN)) {
    if (parse_parameters_and_qualifiers(parametersAndQualifiers)) {
      auto functionDeclarator = new (pool) FunctionDeclaratorAST();

      functionDeclarator->parametersAndQualifiers = parametersAndQualifiers;

      *it = new (pool) List<DeclaratorModifierAST*>(functionDeclarator);
      it = &(*it)->next;

    } else {
      rewind(after_nested_declarator);
    }
  }

  if (LA().is(TokenKind::T_LBRACKET)) {
    SourceLocation lbracketLoc;

    while (match(TokenKind::T_LBRACKET, lbracketLoc)) {
      SourceLocation rbracketLoc;

      auto arrayDeclarator = new (pool) ArrayDeclaratorAST();
      arrayDeclarator->lbracketLoc = lbracketLoc;

      *it = new (pool) List<DeclaratorModifierAST*>(arrayDeclarator);
      it = &(*it)->next;

      if (!match(TokenKind::T_RBRACKET, arrayDeclarator->rbracketLoc)) {
        if (!parse_constant_expression(arrayDeclarator->expression))
          parse_error("expected an expression");

        expect(TokenKind::T_RBRACKET, arrayDeclarator->rbracketLoc);
      }
    }
  }

  return true;
}

bool Parser::parse_abstract_pack_declarator() {
  auto start = currentLocation();

  List<PtrOperatorAST*>* ptrOpList = nullptr;

  parse_ptr_operator_seq(ptrOpList);

  if (!parse_noptr_abstract_pack_declarator()) {
    rewind(start);
    return false;
  }

  return true;
}

bool Parser::parse_noptr_abstract_pack_declarator() {
  if (!match(TokenKind::T_DOT_DOT_DOT)) return false;

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  if (parse_parameters_and_qualifiers(parametersAndQualifiers)) return true;

  while (match(TokenKind::T_LBRACKET)) {
    if (!match(TokenKind::T_RBRACKET)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(expression))
        parse_error("expected a constant expression");

      expect(TokenKind::T_RBRACKET);

      List<AttributeAST*>* attributes = nullptr;

      parse_attribute_specifier_seq(attributes);
    }
  }

  return true;
}

bool Parser::parse_parameter_declaration_clause(
    ParameterDeclarationClauseAST*& yyast) {
  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool) ParameterDeclarationClauseAST();
    yyast = ast;

    ast->ellipsisLoc = ellipsisLoc;

    return true;
  }

  auto ast = new (pool) ParameterDeclarationClauseAST();
  yyast = ast;

  if (!parse_parameter_declaration_list(ast->parameterDeclarationList))
    return false;

  match(TokenKind::T_COMMA, ast->commaLoc);

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  return true;
}

bool Parser::parse_parameter_declaration_list(
    List<ParameterDeclarationAST*>*& yyast) {
  auto it = &yyast;

  ParameterDeclarationAST* declaration = nullptr;

  if (!parse_parameter_declaration(declaration)) return false;

  *it = new (pool) List(declaration);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    ParameterDeclarationAST* declaration = nullptr;

    if (!parse_parameter_declaration(declaration)) {
      rewind(commaLoc);
      break;
    }

    *it = new (pool) List(declaration);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_parameter_declaration(ParameterDeclarationAST*& yyast) {
  auto ast = new (pool) ParameterDeclarationAST();
  yyast = ast;

  parse_attribute_specifier_seq(ast->attributeList);

  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  if (!parse_decl_specifier_seq(ast->typeSpecifierList, specs)) return false;

  const auto before_declarator = currentLocation();

  if (!parse_declarator(ast->declarator)) {
    rewind(before_declarator);

    if (!parse_abstract_declarator(ast->declarator)) rewind(before_declarator);
  }

  if (match(TokenKind::T_EQUAL, ast->equalLoc)) {
    if (!parse_initializer_clause(ast->expression))
      parse_error("expected an initializer");
  }

  return true;
}

bool Parser::parse_initializer(InitializerAST*& yyast) {
  SourceLocation lparenLoc;

  if (match(TokenKind::T_LPAREN, lparenLoc)) {
    if (LA().is(TokenKind::T_RPAREN)) return false;

    auto ast = new (pool) ParenInitializerAST();
    yyast = ast;

    ast->lparenLoc = lparenLoc;

    if (!parse_expression_list(ast->expressionList))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  return parse_brace_or_equal_initializer(yyast);
}

bool Parser::parse_brace_or_equal_initializer(InitializerAST*& yyast) {
  BracedInitListAST* bracedInitList = nullptr;

  if (LA().is(TokenKind::T_LBRACE))
    return parse_braced_init_list(bracedInitList);

  SourceLocation equalLoc;

  if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

  auto ast = new (pool) EqualInitializerAST();
  yyast = ast;

  ast->equalLoc = equalLoc;

  if (!parse_initializer_clause(ast->expression))
    parse_error("expected an intializer");

  return true;
}

bool Parser::parse_initializer_clause(ExpressionAST*& yyast) {
  BracedInitListAST* bracedInitList = nullptr;

  if (LA().is(TokenKind::T_LBRACE))
    return parse_braced_init_list(bracedInitList);

  return parse_assignment_expression(yyast);
}

bool Parser::parse_braced_init_list(BracedInitListAST*& yyast) {
  SourceLocation lbraceLoc;
  SourceLocation commaLoc;
  SourceLocation rbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  if (LA().is(TokenKind::T_DOT)) {
    if (!parse_designated_initializer_clause())
      parse_error("expected designated initializer clause");

    while (match(TokenKind::T_COMMA)) {
      if (LA().is(TokenKind::T_RBRACE)) break;

      if (!parse_designated_initializer_clause())
        parse_error("expected designated initializer clause");
    }

    expect(TokenKind::T_RBRACE, rbraceLoc);

    return true;
  }

  if (match(TokenKind::T_COMMA, commaLoc)) {
    expect(TokenKind::T_RBRACE, rbraceLoc);

    auto ast = new (pool) BracedInitListAST();
    yyast = ast;

    ast->lbraceLoc = lbraceLoc;
    ast->commaLoc = commaLoc;
    ast->rbraceLoc = rbraceLoc;

    return true;
  }

  List<ExpressionAST*>* expressionList = nullptr;

  if (!match(TokenKind::T_RBRACE, rbraceLoc)) {
    if (!parse_initializer_list(expressionList))
      parse_error("expected initializer list");

    expect(TokenKind::T_RBRACE, rbraceLoc);
  }

  auto ast = new (pool) BracedInitListAST();
  yyast = ast;

  ast->lbraceLoc = lbraceLoc;
  ast->expressionList = expressionList;
  ast->rbraceLoc = rbraceLoc;

  return true;
}

bool Parser::parse_initializer_list(List<ExpressionAST*>*& yyast) {
  auto it = &yyast;

  ExpressionAST* expression = nullptr;

  if (!parse_initializer_clause(expression)) return false;

  bool has_triple_dot = false;
  if (match(TokenKind::T_DOT_DOT_DOT)) {
    has_triple_dot = true;
  }

  *it = new (pool) List(expression);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression))
      parse_error("expected initializer clause");

    bool has_triple_dot = false;
    if (match(TokenKind::T_DOT_DOT_DOT)) {
      has_triple_dot = true;
    }

    *it = new (pool) List(expression);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_designated_initializer_clause() {
  if (!parse_designator()) return false;

  InitializerAST* initializer = nullptr;

  if (!parse_brace_or_equal_initializer(initializer))
    parse_error("expected an initializer");

  return true;
}

bool Parser::parse_designator() {
  if (!match(TokenKind::T_DOT)) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_expr_or_braced_init_list(ExpressionAST*& yyast) {
  if (LA().is(TokenKind::T_LBRACE)) {
    BracedInitListAST* bracedInitList = nullptr;

    return parse_braced_init_list(bracedInitList);
  }

  if (!parse_expression(yyast)) parse_error("expected an expression");

  return true;
}

bool Parser::parse_virt_specifier_seq() {
  if (!parse_virt_specifier()) return false;

  while (parse_virt_specifier()) {
    //
  }

  return true;
}

bool Parser::parse_function_body(StatementAST*& yyast) {
  if (LA().is(TokenKind::T_SEMICOLON)) return false;

  if (parse_function_try_block(yyast)) return true;

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

  if (LA().isNot(TokenKind::T_LBRACE)) return false;

  if (skip_function_body) {
    expect(TokenKind::T_LBRACE);

    int depth = 1;

    while (const auto& tok = LA()) {
      if (tok.is(TokenKind::T_LBRACE)) {
        ++depth;
      } else if (tok.is(TokenKind::T_RBRACE)) {
        if (!--depth) {
          break;
        }
      }

      consumeToken();
    }

    expect(TokenKind::T_RBRACE);
  } else {
    if (!parse_compound_statement(yyast))
      parse_error("expected a compound statement");
  }

  return true;
}

bool Parser::parse_enum_specifier(SpecifierAST*& yyast) {
  const auto start = currentLocation();

  SourceLocation enumLoc;
  SourceLocation classLoc;

  if (!parse_enum_key(enumLoc, classLoc)) {
    rewind(start);
    return false;
  }

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  parse_enum_head_name(nestedNameSpecifier, name);

  EnumBaseAST* enumBase = nullptr;

  parse_enum_base(enumBase);

  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  auto ast = new (pool) EnumSpecifierAST();
  yyast = ast;

  ast->enumLoc = enumLoc;
  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;
  ast->enumBase = enumBase;
  ast->lbraceLoc = lbraceLoc;

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    parse_enumerator_list(ast->enumeratorList);

    match(TokenKind::T_COMMA, ast->commaLoc);

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  return true;
}

bool Parser::parse_enum_head_name(NestedNameSpecifierAST*& nestedNameSpecifier,
                                  NameAST*& name) {
  const auto start = currentLocation();

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(start);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = new (pool) SimpleNameAST();
  id->identifierLoc = identifierLoc;

  name = id;

  return true;
}

bool Parser::parse_opaque_enum_declaration(DeclarationAST*& yyast) {
  SourceLocation enumLoc;
  SourceLocation classLoc;

  if (!parse_enum_key(enumLoc, classLoc)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  if (!parse_enum_head_name(nestedNameSpecifier, name)) return false;

  EnumBaseAST* enumBase = nullptr;

  parse_enum_base(enumBase);

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool) OpaqueEnumDeclarationAST();
  yyast = ast;

  ast->enumLoc = enumLoc;
  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;
  ast->enumBase = enumBase;
  ast->emicolonLoc = semicolonLoc;

  return true;
}

bool Parser::parse_enum_key(SourceLocation& enumLoc, SourceLocation& classLoc) {
  if (!match(TokenKind::T_ENUM, enumLoc)) return false;

  if (match(TokenKind::T_CLASS, classLoc)) {
    //
  } else if (match(TokenKind::T_STRUCT, classLoc)) {
    //
  }

  return true;
}

bool Parser::parse_enum_base(EnumBaseAST*& yyast) {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList))
    parse_error("expected a type specifier");

  auto ast = new (pool) EnumBaseAST();
  yyast = ast;

  ast->colonLoc = colonLoc;
  ast->typeSpecifierList = typeSpecifierList;

  return true;
}

bool Parser::parse_enumerator_list(List<EnumeratorAST*>*& yyast) {
  auto it = &yyast;

  EnumeratorAST* enumerator = nullptr;

  if (!parse_enumerator_definition(enumerator)) return false;

  *it = new (pool) List(enumerator);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (LA().is(TokenKind::T_RBRACE)) {
      rewind(commaLoc);
      break;
    }

    EnumeratorAST* enumerator = nullptr;

    if (!parse_enumerator_definition(enumerator))
      parse_error("expected an enumerator");

    *it = new (pool) List(enumerator);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_enumerator_definition(EnumeratorAST*& yyast) {
  if (!parse_enumerator(yyast)) return false;

  if (!match(TokenKind::T_EQUAL, yyast->equalLoc)) return true;

  if (!parse_constant_expression(yyast->expression))
    parse_error("expected an expression");

  return true;
}

bool Parser::parse_enumerator(EnumeratorAST*& yyast) {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto name = new (pool) SimpleNameAST();
  name->identifierLoc = identifierLoc;

  auto ast = new (pool) EnumeratorAST();
  yyast = ast;

  ast->name = name;

  parse_attribute_specifier_seq(ast->attributeList);

  return true;
}

bool Parser::parse_using_enum_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_USING)) return false;

  SpecifierAST* enumSpecifier = nullptr;

  if (!parse_elaborated_enum_specifier(enumSpecifier)) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_namespace_definition(DeclarationAST*& yyast) {
  const auto start = currentLocation();

  SourceLocation inlineLoc;

  const auto hasInline = match(TokenKind::T_INLINE, inlineLoc);

  SourceLocation namespaceLoc;

  if (!match(TokenKind::T_NAMESPACE, namespaceLoc)) {
    rewind(start);
    return false;
  }

  auto ast = new (pool) NamespaceDefinitionAST();
  yyast = ast;

  ast->inlineLoc = inlineLoc;
  ast->namespaceLoc = namespaceLoc;

  parse_attribute_specifier_seq(ast->attributeList);

  if (LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_COLON_COLON)) {
    consumeToken();

    while (match(TokenKind::T_COLON_COLON)) {
      match(TokenKind::T_INLINE);
      expect(TokenKind::T_IDENTIFIER);
    }
  } else {
    SourceLocation identifierLoc;

    if (match(TokenKind::T_IDENTIFIER, identifierLoc)) {
      auto name = new (pool) SimpleNameAST();
      name->identifierLoc = identifierLoc;

      ast->name = name;
    }
  }

  parse_attribute_specifier_seq(ast->extraAttributeList);

  expect(TokenKind::T_LBRACE, ast->lbraceLoc);

  parse_namespace_body(ast);

  expect(TokenKind::T_RBRACE, ast->rbraceLoc);

  return true;
}

bool Parser::parse_namespace_body(NamespaceDefinitionAST* yyast) {
  bool skipping = false;

  auto it = &yyast->declarationList;

  while (LA()) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    const auto beforeDeclaration = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_declaration(declaration)) {
      skipping = false;

      if (declaration) {
        *it = new (pool) List(declaration);
        it = &(*it)->next;
      }
    } else {
      parse_skip_declaration(skipping);

      if (currentLocation() == beforeDeclaration) consumeToken();
    }
  }

  return true;
}

bool Parser::parse_namespace_alias_definition(DeclarationAST*& yyast) {
  SourceLocation namespaceLoc;

  if (!match(TokenKind::T_NAMESPACE, namespaceLoc)) return false;

  auto ast = new (pool) NamespaceAliasDefinitionAST();
  yyast = ast;

  ast->namespaceLoc = namespaceLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  expect(TokenKind::T_EQUAL, ast->equalLoc);

  if (!parse_qualified_namespace_specifier(ast->nestedNameSpecifier, ast->name))
    parse_error("expected a namespace name");

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_qualified_namespace_specifier(
    NestedNameSpecifierAST*& nestedNameSpecifier, NameAST*& name) {
  const auto saved = currentLocation();

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(saved);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = new (pool) SimpleNameAST();
  id->identifierLoc = identifierLoc;

  name = id;

  return true;
}

bool Parser::parse_using_directive(DeclarationAST*& yyast) {
  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  SourceLocation namespaceLoc;

  if (!match(TokenKind::T_NAMESPACE, namespaceLoc)) return false;

  const auto saved = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(saved);

  if (!match(TokenKind::T_IDENTIFIER)) parse_error("expected a namespace name");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_using_declaration(DeclarationAST*& yyast) {
  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  auto ast = new (pool) UsingDeclarationAST();
  yyast = ast;

  ast->usingLoc = usingLoc;

  if (!parse_using_declarator_list(ast->usingDeclaratorList))
    parse_error("expected a using declarator");

  match(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_using_declarator_list(List<UsingDeclaratorAST*>*& yyast) {
  auto it = &yyast;

  UsingDeclaratorAST* declarator = nullptr;

  if (!parse_using_declarator(declarator)) return false;

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  *it = new (pool) List(declarator);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    UsingDeclaratorAST* declarator = nullptr;

    if (!parse_using_declarator(declarator))
      parse_error("expected a using declarator");

    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

    *it = new (pool) List(declarator);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_using_declarator(UsingDeclaratorAST*& yyast) {
  SourceLocation typenameLoc;

  match(TokenKind::T_TYPENAME, typenameLoc);

  const auto saved = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(saved);

  NameAST* name = nullptr;

  if (!parse_unqualified_id(name)) return false;

  yyast = new (pool) UsingDeclaratorAST();
  yyast->typenameLoc = typenameLoc;
  yyast->nestedNameSpecifier = nestedNameSpecifier;
  yyast->name = name;

  return true;
}

bool Parser::parse_asm_declaration(DeclarationAST*& yyast) {
  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  SourceLocation asmLoc;

  if (!match(TokenKind::T_ASM, asmLoc)) return false;

  auto ast = new (pool) AsmDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->asmLoc = asmLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_string_literal_seq(ast->stringLiteralList);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_linkage_specification(DeclarationAST*& yyast) {
  SourceLocation externLoc;

  if (!match(TokenKind::T_EXTERN, externLoc)) return false;

  SourceLocation stringLiteralLoc;

  if (!match(TokenKind::T_STRING_LITERAL, stringLiteralLoc)) return false;

  SourceLocation lbraceLoc;

  if (match(TokenKind::T_LBRACE, lbraceLoc)) {
    SourceLocation rbraceLoc;

    auto ast = new (pool) LinkageSpecificationAST();
    yyast = ast;

    ast->externLoc = externLoc;
    ast->stringliteralLoc = stringLiteralLoc;
    ast->lbraceLoc = lbraceLoc;

    if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
      if (!parse_declaration_seq(ast->declarationList))
        parse_error("expected a declaration");

      expect(TokenKind::T_RBRACE, ast->rbraceLoc);
    }

    return true;
  }

  DeclarationAST* declaration = nullptr;

  if (!parse_declaration(declaration)) return false;

  auto ast = new (pool) LinkageSpecificationAST();
  yyast = ast;

  ast->externLoc = externLoc;
  ast->stringliteralLoc = stringLiteralLoc;
  ast->declarationList = new (pool) List(declaration);

  return true;
}

bool Parser::parse_attribute_specifier_seq(List<AttributeAST*>*& yyast) {
  if (!parse_attribute_specifier()) return false;

  while (parse_attribute_specifier()) {
    //
  }

  return true;
}

bool Parser::parse_attribute_specifier() {
  if (LA().is(TokenKind::T_LBRACKET) && LA(1).is(TokenKind::T_LBRACKET)) {
    consumeToken();
    consumeToken();
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

  List<SourceLocation>* stringLiteralList = nullptr;

  if (!parse_string_literal_seq(stringLiteralList))
    parse_error("expected a string literal");

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

  while (const auto& tk = LA()) {
    if (tk.is(TokenKind::T_LPAREN)) {
      ++count;
    } else if (tk.is(TokenKind::T_RPAREN)) {
      if (!--count) return true;
    }

    consumeToken();
  }

  return false;
}

bool Parser::parse_alignment_specifier() {
  if (!match(TokenKind::T_ALIGNAS)) return false;

  expect(TokenKind::T_LPAREN);

  const auto after_lparen = currentLocation();

  TypeIdAST* typeId = nullptr;

  if (parse_type_id(typeId)) {
    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

    if (match(TokenKind::T_RPAREN)) {
      return true;
    }
  }

  rewind(after_lparen);

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
  const auto start = currentLocation();

  if (parse_attribute_scoped_token()) return true;

  rewind(start);

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

  if (LA().is(TokenKind::T_COLON)) {
    parse_module_partition();
  }

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_module_name() {
  const auto start = currentLocation();

  if (!parse_module_name_qualifier()) rewind(start);

  expect(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_module_partition() {
  if (!match(TokenKind::T_COLON)) return false;

  const auto saved = currentLocation();

  if (!parse_module_name_qualifier()) rewind(saved);

  expect(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_module_name_qualifier() {
  if (LA().isNot(TokenKind::T_IDENTIFIER)) return false;

  if (LA(1).isNot(TokenKind::T_DOT)) return false;

  do {
    consumeToken();
    consumeToken();
  } while (LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_DOT));

  return true;
}

bool Parser::parse_export_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_EXPORT)) return false;

  if (match(TokenKind::T_LBRACE)) {
    if (!match(TokenKind::T_RBRACE)) {
      List<DeclarationAST*>* declarationList = nullptr;

      if (!parse_declaration_seq(declarationList))
        parse_error("expected a declaration");

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

  const auto start = currentLocation();

  const auto import = parse_import_keyword();

  rewind(start);

  return import;
}

bool Parser::parse_module_import_declaration(DeclarationAST*& yyast) {
  if (!parse_import_keyword()) return false;

  if (!parse_import_name()) parse_error("expected a module");

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

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

  List<DeclarationAST*>* declarationList = nullptr;

  parse_declaration_seq(declarationList);

  return true;
}

bool Parser::parse_private_module_fragment() {
  if (!parse_module_keyword()) return false;

  if (!match(TokenKind::T_COLON)) return false;

  if (!match(TokenKind::T_PRIVATE)) return false;

  expect(TokenKind::T_SEMICOLON);

  List<DeclarationAST*>* declarationList = nullptr;

  parse_declaration_seq(declarationList);

  return true;
}

bool Parser::parse_class_specifier(SpecifierAST*& yyast) {
  const auto start = currentLocation();

  SourceLocation classKey;

  const auto maybeClassSpecifier = parse_class_key(classKey);

  rewind(start);

  if (!maybeClassSpecifier) return false;

  auto it = class_specifiers_.find(start);

  if (it != class_specifiers_.end()) {
    auto [cursor, parsed] = it->second;
    rewind(cursor);
    return parsed;
  }

  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
  NameAST* className = nullptr;
  BaseClauseAST* baseClause = nullptr;

  if (!parse_class_head(classLoc, attributeList, className, baseClause)) {
    parse_reject_class_specifier(start);
    return false;
  }

  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) {
    parse_reject_class_specifier(start);
    return false;
  }

  auto ast = new (pool) ClassSpecifierAST();
  yyast = ast;

  ast->classLoc = classLoc;
  ast->attributeList = attributeList;
  ast->name = className;
  ast->baseClause = baseClause;
  ast->lbraceLoc = lbraceLoc;

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    if (!parse_class_body(ast->declarationList))
      parse_error("expected class body");

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  parse_leave_class_specifier(start);

  return true;
}

bool Parser::parse_leave_class_specifier(SourceLocation start) {
  class_specifiers_.emplace(start, std::make_tuple(currentLocation(), true));
  return true;
}

bool Parser::parse_reject_class_specifier(SourceLocation start) {
  class_specifiers_.emplace(start, std::make_tuple(currentLocation(), false));
  return true;
}

bool Parser::parse_class_body(List<DeclarationAST*>*& yyast) {
  auto it = &yyast;

  bool skipping = false;

  while (LA()) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    const auto saved = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_member_specification(declaration)) {
      if (declaration) {
        *it = new (pool) List(declaration);
        it = &(*it)->next;
      }
      skipping = false;
    } else {
      if (!skipping) parse_error("expected a declaration");

      if (currentLocation() == saved) consumeToken();

      skipping = true;
    }
  }

  return true;
}

bool Parser::parse_class_head(SourceLocation& classLoc,
                              List<AttributeAST*>*& attributeList,
                              NameAST*& name, BaseClauseAST*& baseClause) {
  if (!parse_class_key(classLoc)) return false;

  parse_attribute_specifier_seq(attributeList);

  if (parse_class_head_name(name)) {
    parse_class_virt_specifier();
  }

  parse_base_clause(baseClause);

  return true;
}

bool Parser::parse_class_head_name(NameAST*& yyast) {
  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(start);

  NameAST* name = nullptr;

  if (!parse_class_name(name)) return false;

  if (!nestedNameSpecifier) yyast = name;

  return true;
}

bool Parser::parse_class_virt_specifier() {
  if (!parse_final()) return false;

  return true;
}

bool Parser::parse_class_key(SourceLocation& classLoc) {
  switch (TokenKind(LA())) {
    case TokenKind::T_CLASS:
    case TokenKind::T_STRUCT:
    case TokenKind::T_UNION:
      classLoc = consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_member_specification(DeclarationAST*& yyast) {
  return parse_member_declaration(yyast);
}

bool Parser::parse_member_declaration(DeclarationAST*& yyast) {
  const auto start = currentLocation();

  if (parse_access_specifier()) {
    expect(TokenKind::T_COLON);
    return true;
  }

  if (parse_empty_declaration(yyast)) return true;

  if (LA().is(TokenKind::T_USING)) {
    if (parse_using_enum_declaration(yyast)) return true;

    rewind(start);

    if (parse_alias_declaration(yyast)) return true;

    rewind(start);

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

  rewind(start);

  return parse_member_declaration_helper(yyast);
}

bool Parser::parse_maybe_template_member() {
  const auto start = currentLocation();

  match(TokenKind::T_EXPLICIT);

  const auto has_template = match(TokenKind::T_TEMPLATE);

  rewind(start);

  return has_template;
}

bool Parser::parse_member_declaration_helper(DeclarationAST*& yyast) {
  const auto has_extension = match(TokenKind::T___EXTENSION__);

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  auto after_decl_specs = currentLocation();

  DeclSpecs specs;

  List<SpecifierAST*>* declSpecifierList = nullptr;

  if (!parse_decl_specifier_seq_no_typespecs(declSpecifierList, specs))
    rewind(after_decl_specs);

  after_decl_specs = currentLocation();

  IdDeclaratorAST* declaratorId = nullptr;

  if (parse_declarator_id(declaratorId)) {
    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

    if (parse_parameters_and_qualifiers(parametersAndQualifiers)) {
      const auto after_parameters = currentLocation();

      StatementAST* functionBody = nullptr;

      if (parse_member_function_definition_body(functionBody)) {
        DeclaratorAST* declarator = new (pool) DeclaratorAST();
        declarator->coreDeclarator = declaratorId;

        auto functionDeclarator = new (pool) FunctionDeclaratorAST();

        functionDeclarator->parametersAndQualifiers = parametersAndQualifiers;

        declarator->modifiers =
            new (pool) List<DeclaratorModifierAST*>(functionDeclarator);

        auto ast = new (pool) FunctionDefinitionAST();
        yyast = ast;

        ast->declSpecifierList = declSpecifierList;
        ast->declarator = declarator;
        ast->functionBody = functionBody;

        return true;
      }

      rewind(after_parameters);

      if (parse_member_declarator_modifier() && match(TokenKind::T_SEMICOLON)) {
        return true;
      }
    }
  }

  rewind(after_decl_specs);

  auto lastDeclSpecifier = &declSpecifierList;

  while (*lastDeclSpecifier) {
    lastDeclSpecifier = &(*lastDeclSpecifier)->next;
  }

  if (!parse_decl_specifier_seq(*lastDeclSpecifier, specs))
    rewind(after_decl_specs);

  after_decl_specs = currentLocation();

  if (!specs.has_typespec()) return false;

  if (match(TokenKind::T_SEMICOLON)) return true;  // ### complex typespec

  DeclaratorAST* declarator = nullptr;

  if (parse_declarator(declarator) && getFunctionDeclarator(declarator)) {
    StatementAST* functionBody = nullptr;

    if (parse_member_function_definition_body(functionBody)) {
      auto ast = new (pool) FunctionDefinitionAST();
      yyast = ast;

      ast->declSpecifierList = declSpecifierList;
      ast->declarator = declarator;
      ast->functionBody = functionBody;

      return true;
    }
  }

  rewind(after_decl_specs);

  List<DeclaratorAST*>* declaratorList = nullptr;

  if (!parse_member_declarator_list(declaratorList))
    parse_error("expected a declarator");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_member_function_definition_body(StatementAST*& yyast) {
  const auto has_requires_clause = parse_requires_clause();

  bool has_virt_specifier_seq = false;

  if (!has_requires_clause) has_virt_specifier_seq = parse_virt_specifier_seq();

  if (!parse_function_body(yyast)) return false;

  return true;
}

bool Parser::parse_member_declarator_modifier() {
  if (parse_requires_clause()) return true;

  if (LA().is(TokenKind::T_LBRACE) || LA().is(TokenKind::T_EQUAL)) {
    InitializerAST* initializer = nullptr;

    return parse_brace_or_equal_initializer(initializer);
  }

  parse_virt_specifier_seq();

  parse_pure_specifier();

  return true;
}

bool Parser::parse_member_declarator_list(List<DeclaratorAST*>*& yyast) {
  auto it = &yyast;

  DeclaratorAST* declarator = nullptr;

  if (!parse_member_declarator(declarator)) return false;

  *it = new (pool) List(declarator);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    DeclaratorAST* declarator = nullptr;

    if (!parse_member_declarator(declarator))
      parse_error("expected a declarator");

    if (declarator) {
      *it = new (pool) List(declarator);
      it = &(*it)->next;
    }
  }

  return true;
}

bool Parser::parse_member_declarator(DeclaratorAST*& yyast) {
  const auto start = currentLocation();

  const auto has_identifier = match(TokenKind::T_IDENTIFIER);

  parse_attribute_specifier();

  if (match(TokenKind::T_COLON)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constant_expression(expression))
      parse_error("expected an expression");

    InitializerAST* initializer = nullptr;

    parse_brace_or_equal_initializer(initializer);

    return true;
  }

  rewind(start);

  if (!parse_declarator(yyast)) return false;

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

  SourceLocation literalLoc;

  if (!match(TokenKind::T_INTEGER_LITERAL, literalLoc)) return false;

  const auto& number = unit->tokenText(literalLoc);

  if (number != "0") return false;

  return true;
}

bool Parser::parse_conversion_function_id(NameAST*& yyast) {
  if (!match(TokenKind::T_OPERATOR)) return false;

  if (!parse_conversion_type_id()) return false;

  return true;
}

bool Parser::parse_conversion_type_id() {
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList)) return false;

  parse_conversion_declarator();

  return true;
}

bool Parser::parse_conversion_declarator() {
  List<PtrOperatorAST*>* ptrOpList = nullptr;

  return parse_ptr_operator_seq(ptrOpList);
}

bool Parser::parse_base_clause(BaseClauseAST*& yyast) {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  auto ast = new (pool) BaseClauseAST();
  yyast = ast;

  ast->colonLoc = colonLoc;

  if (!parse_base_specifier_list(ast->baseSpecifierList))
    parse_error("expected a base class specifier");

  return true;
}

bool Parser::parse_base_specifier_list(List<BaseSpecifierAST*>*& yyast) {
  auto it = &yyast;

  BaseSpecifierAST* baseSpecifier = nullptr;

  if (!parse_base_specifier(baseSpecifier)) return false;

  const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

  *it = new (pool) List(baseSpecifier);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    BaseSpecifierAST* baseSpecifier = nullptr;

    if (!parse_base_specifier(baseSpecifier))
      parse_error("expected a base class specifier");

    const auto has_triple_dot = match(TokenKind::T_DOT_DOT_DOT);

    *it = new (pool) List(baseSpecifier);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_base_specifier(BaseSpecifierAST*& yyast) {
  auto ast = new (pool) BaseSpecifierAST();
  yyast = ast;

  parse_attribute_specifier_seq(ast->attributeList);

  bool has_virtual = match(TokenKind::T_VIRTUAL);

  bool has_access_specifier = false;

  if (has_virtual)
    has_access_specifier = parse_access_specifier();
  else if (parse_access_specifier()) {
    has_virtual = match(TokenKind::T_VIRTUAL);
    has_access_specifier = true;
  }

  if (!parse_class_or_decltype(ast->name)) return false;

  return true;
}

bool Parser::parse_class_or_decltype(NameAST*& yyast) {
  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (parse_nested_name_specifier(nestedNameSpecifier)) {
    const auto saved = currentLocation();

    NameAST* name = nullptr;

    SourceLocation templateLoc;

    if (match(TokenKind::T_TEMPLATE, templateLoc) &&
        parse_simple_template_id(name)) {
      auto ast = new (pool) QualifiedNameAST();
      yyast = ast;

      ast->nestedNameSpecifier = nestedNameSpecifier;
      ast->templateLoc = templateLoc;
      ast->name = name;

      return true;
    }

    if (parse_type_name(name)) {
      auto ast = new (pool) QualifiedNameAST();
      yyast = ast;

      ast->nestedNameSpecifier = nestedNameSpecifier;
      ast->name = name;

      return true;
    }
  }

  rewind(start);

  SpecifierAST* decltypeSpecifier = nullptr;

  if (parse_decltype_specifier(decltypeSpecifier)) {
    auto ast = new (pool) DecltypeNameAST();
    yyast = ast;

    ast->decltypeSpecifier = decltypeSpecifier;

    return true;
  }

  return parse_type_name(yyast);
}

bool Parser::parse_access_specifier() {
  switch (TokenKind(LA())) {
    case TokenKind::T_PRIVATE:
    case TokenKind::T_PROTECTED:
    case TokenKind::T_PUBLIC:
      consumeToken();
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

  if (LA().is(TokenKind::T_LBRACE)) {
    BracedInitListAST* bracedInitList = nullptr;

    if (!parse_braced_init_list(bracedInitList))
      parse_error("expected an initializer");
  } else {
    expect(TokenKind::T_LPAREN);

    if (!match(TokenKind::T_RPAREN)) {
      List<ExpressionAST*>* expressionList = nullptr;

      if (!parse_expression_list(expressionList))
        parse_error("expected an expression");

      expect(TokenKind::T_RPAREN);
    }
  }

  return true;
}

bool Parser::parse_mem_initializer_id() {
  const auto start = currentLocation();

  NameAST* name = nullptr;

  if (parse_class_or_decltype(name)) return true;

  rewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_operator_function_id(NameAST*& yyast) {
  if (!match(TokenKind::T_OPERATOR)) return false;

  if (!parse_operator()) return false;

  return true;
}

bool Parser::parse_operator() {
  switch (TokenKind(LA())) {
    case TokenKind::T_LPAREN:
      consumeToken();
      expect(TokenKind::T_RPAREN);
      return true;

    case TokenKind::T_LBRACKET:
      consumeToken();
      expect(TokenKind::T_RBRACKET);
      return true;

    case TokenKind::T_GREATER: {
      if (parse_greater_greater_equal()) return true;
      if (parse_greater_greater()) return true;
      if (parse_greater_equal()) return true;
      consumeToken();
      return true;
    }

    case TokenKind::T_NEW:
      consumeToken();
      if (match(TokenKind::T_LBRACKET)) {
        expect(TokenKind::T_RBRACKET);
      }
      return true;

    case TokenKind::T_DELETE:
      consumeToken();
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
      consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

bool Parser::parse_literal_operator_id(NameAST*& yyast) {
  if (!match(TokenKind::T_OPERATOR)) return false;

  if (match(TokenKind::T_USER_DEFINED_STRING_LITERAL)) return true;

  List<SourceLocation>* stringLiteralList = nullptr;

  if (!parse_string_literal_seq(stringLiteralList)) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_template_declaration(DeclarationAST*& yyast) {
  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;

  if (!parse_template_head(templateLoc, lessLoc, templateParameterList,
                           greaterLoc))
    return false;

  DeclarationAST* declaration = nullptr;

  if (!parse_concept_definition(declaration)) {
    if (!parse_declaration(declaration)) parse_error("expected a declaration");
  }

  auto ast = new (pool) TemplateDeclarationAST();
  yyast = ast;

  ast->templateLoc = templateLoc;
  ast->lessLoc = lessLoc;
  ast->templateParameterList = templateParameterList;
  ast->greaterLoc = greaterLoc;
  ast->declaration = declaration;

  return true;
}

bool Parser::parse_template_head(SourceLocation& templateLoc,
                                 SourceLocation& lessLoc,
                                 List<DeclarationAST*>* templateParameterList,
                                 SourceLocation& greaterLoc) {
  if (!match(TokenKind::T_TEMPLATE, templateLoc)) return false;

  if (!match(TokenKind::T_LESS, lessLoc)) return false;

  if (!match(TokenKind::T_GREATER, greaterLoc)) {
    List<DeclarationAST*>* templateParameterList = nullptr;

    if (!parse_template_parameter_list(templateParameterList))
      parse_error("expected a template parameter");

    expect(TokenKind::T_GREATER, greaterLoc);
  }

  parse_requires_clause();

  return true;
}

bool Parser::parse_template_parameter_list(List<DeclarationAST*>*& yyast) {
  auto it = &yyast;

  DeclarationAST* declaration = nullptr;

  if (!parse_template_parameter(declaration)) return false;

  *it = new (pool) List(declaration);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    DeclarationAST* declaration = nullptr;

    if (!parse_template_parameter(declaration))
      parse_error("expected a template parameter");

    *it = new (pool) List(declaration);
    it = &(*it)->next;
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

bool Parser::parse_template_parameter(DeclarationAST*& yyast) {
  const auto start = currentLocation();

  if (parse_type_parameter() &&
      (LA().is(TokenKind::T_COMMA) || LA().is(TokenKind::T_GREATER)))
    return true;

  rewind(start);

  ParameterDeclarationAST* parameter = nullptr;

  if (parse_parameter_declaration(parameter)) {
    yyast = parameter;
    return true;
  }

  return false;
}

bool Parser::parse_type_parameter() {
  if (parse_template_type_parameter()) return true;

  if (parse_typename_type_parameter()) return true;

  return parse_constraint_type_parameter();
}

bool Parser::parse_typename_type_parameter() {
  if (!parse_type_parameter_key()) return false;

  const auto saved = currentLocation();

  if ((LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_EQUAL)) ||
      LA().is(TokenKind::T_EQUAL)) {
    const auto has_identifier = match(TokenKind::T_IDENTIFIER);

    expect(TokenKind::T_EQUAL);

    TypeIdAST* typeId = nullptr;

    if (!parse_type_id(typeId)) parse_error("expected a type id");

    return true;
  }

  const auto has_tripled_dot = match(TokenKind::T_DOT_DOT_DOT);

  const auto has_identifier = match(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_template_type_parameter() {
  const auto start = currentLocation();

  SourceLocation templateLoc;
  SourceLocation lessLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;
  SourceLocation greaterLoc;

  if (!parse_template_head(templateLoc, lessLoc, templateParameterList,
                           greaterLoc)) {
    rewind(start);
    return false;
  }

  if (!parse_type_parameter_key()) parse_error("expected a type parameter");

  const auto saved = currentLocation();

  if ((LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_EQUAL)) ||
      LA().is(TokenKind::T_EQUAL)) {
    const auto has_identifier = match(TokenKind::T_IDENTIFIER);

    expect(TokenKind::T_EQUAL);

    NameAST* name = nullptr;

    if (!parse_id_expression(name)) parse_error("expected an id-expression");

    return true;
  }

  const auto has_tripled_dot = match(TokenKind::T_DOT_DOT_DOT);

  const auto has_identifier = match(TokenKind::T_IDENTIFIER);

  return true;
}

bool Parser::parse_constraint_type_parameter() {
  if (!parse_type_constraint()) return false;

  const auto saved = currentLocation();

  if ((LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_EQUAL)) ||
      LA().is(TokenKind::T_EQUAL)) {
    const auto has_identifier = match(TokenKind::T_IDENTIFIER);

    expect(TokenKind::T_EQUAL);

    TypeIdAST* typeId = nullptr;

    if (!parse_type_id(typeId))
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
  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(start);

  NameAST* name = nullptr;

  if (!parse_concept_name(name)) {
    rewind(start);
    return false;
  }

  SourceLocation lessLoc;

  if (match(TokenKind::T_LESS, lessLoc)) {
    SourceLocation greaterLoc;

    List<TemplateArgumentAST*>* templateArgumentList = nullptr;

    if (!parse_template_argument_list(templateArgumentList))
      parse_error("expected a template argument");

    expect(TokenKind::T_GREATER, greaterLoc);

    auto templateId = new (pool) TemplateNameAST();
    templateId->name = name;
    templateId->lessLoc = lessLoc;
    templateId->templateArgumentList = templateArgumentList;
    templateId->greaterLoc = greaterLoc;

    name = templateId;
  }

  return true;
}

bool Parser::parse_simple_template_id(NameAST*& yyast) {
  NameAST* name = nullptr;

  if (!parse_template_name(name)) return false;

  SourceLocation lessLoc;

  if (!match(TokenKind::T_LESS, lessLoc)) return false;

  SourceLocation greaterLoc;

  List<TemplateArgumentAST*>* templateArgumentList = nullptr;

  if (!match(TokenKind::T_GREATER, greaterLoc)) {
    if (!parse_template_argument_list(templateArgumentList)) return false;

    if (!match(TokenKind::T_GREATER, greaterLoc)) return false;
  }

  auto ast = new (pool) TemplateNameAST();
  yyast = ast;

  ast->name = name;
  ast->lessLoc = lessLoc;
  ast->templateArgumentList = templateArgumentList;
  ast->greaterLoc = greaterLoc;

  return true;
}

bool Parser::parse_template_id(NameAST*& yyast) {
  if (LA().is(TokenKind::T_OPERATOR)) {
    const auto start = currentLocation();

    NameAST* name = nullptr;

    if (!parse_literal_operator_id(name)) {
      rewind(start);

      name = nullptr;

      if (!parse_operator_function_id(name)) return false;
    }

    SourceLocation lessLoc;

    if (!match(TokenKind::T_LESS, lessLoc)) return false;

    List<TemplateArgumentAST*>* templateArgumentList = nullptr;

    SourceLocation greaterLoc;

    if (!match(TokenKind::T_GREATER, greaterLoc)) {
      if (!parse_template_argument_list(templateArgumentList)) return false;

      if (!match(TokenKind::T_GREATER, greaterLoc)) return false;
    }

    auto ast = new (pool) TemplateNameAST();
    yyast = ast;

    ast->name = name;
    ast->lessLoc = lessLoc;
    ast->templateArgumentList = templateArgumentList;
    ast->greaterLoc = greaterLoc;

    return true;
  }

  return parse_simple_template_id(yyast);
}

bool Parser::parse_template_argument_list(List<TemplateArgumentAST*>*& yyast) {
  auto it = &yyast;

  TemplArgContext templArgContext(this);

  TemplateArgumentAST* templateArgument = nullptr;

  if (!parse_template_argument(templateArgument)) return false;

  match(TokenKind::T_DOT_DOT_DOT);

  *it = new (pool) List(templateArgument);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    TemplateArgumentAST* templateArgument = nullptr;

    if (!parse_template_argument(templateArgument)) {
      // parse_error("expected a template argument"); // ### FIXME
      return false;
    }

    match(TokenKind::T_DOT_DOT_DOT);

    *it = new (pool) List(templateArgument);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_template_argument(TemplateArgumentAST*& yyast) {
  const auto start = currentLocation();

  auto it = template_arguments_.find(start);

  if (it != template_arguments_.end()) {
    rewind(get<0>(it->second));
    return get<1>(it->second);
  }

  auto check = [&]() -> bool {
    const auto& tk = LA();

    return tk.is(TokenKind::T_COMMA) || tk.is(TokenKind::T_GREATER) ||
           tk.is(TokenKind::T_DOT_DOT_DOT);
  };

  const auto saved = currentLocation();

  TypeIdAST* typeId = nullptr;

  if (parse_type_id(typeId) && check()) {
    template_arguments_.emplace(start,
                                std::make_tuple(currentLocation(), true));
    return true;
  }

  rewind(saved);

  ExpressionAST* expression = nullptr;

  const auto parsed = parse_template_argument_constant_expression(expression);

  if (parsed && check()) {
    template_arguments_.emplace(start,
                                std::make_tuple(currentLocation(), true));
    return true;
  }

  template_arguments_.emplace(start, std::make_tuple(currentLocation(), false));

  return false;
}

bool Parser::parse_constraint_expression(ExpressionAST*& yyast) {
  return parse_logical_or_expression(yyast, false);
}

bool Parser::parse_deduction_guide(DeclarationAST*& yyast) {
  SpecifierAST* explicitSpecifier = nullptr;

  const auto has_explicit_spec = parse_explicit_specifier(explicitSpecifier);

  NameAST* name = nullptr;

  if (!parse_template_name(name)) return false;

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

    if (!parse_parameter_declaration_clause(parameterDeclarationClause))
      parse_error("expected a parameter declaration");

    expect(TokenKind::T_RPAREN);
  }

  if (!match(TokenKind::T_MINUS_GREATER)) return false;

  NameAST* templateId = nullptr;

  if (!parse_simple_template_id(templateId))
    parse_error("expected a template id");

  expect(TokenKind::T_SEMICOLON);

  return true;
}

bool Parser::parse_concept_definition(DeclarationAST*& yyast) {
  SourceLocation conceptLoc;

  if (!match(TokenKind::T_CONCEPT, conceptLoc)) return false;

  auto ast = new (pool) ConceptDefinitionAST();
  yyast = ast;

  ast->conceptLoc = conceptLoc;

  if (!parse_concept_name(ast->name)) parse_error("expected a concept name");

  expect(TokenKind::T_EQUAL, ast->equalLoc);

  if (!parse_constraint_expression(ast->expression))
    parse_error("expected a constraint expression");

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

bool Parser::parse_concept_name(NameAST*& yyast) {
  return parse_name_id(yyast);
}

bool Parser::parse_typename_specifier(SpecifierAST*& yyast) {
  if (!match(TokenKind::T_TYPENAME)) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

  const auto after_nested_name_specifier = currentLocation();

  const auto has_template = match(TokenKind::T_TEMPLATE);

  NameAST* name = nullptr;

  if (parse_simple_template_id(name)) return true;

  rewind(after_nested_name_specifier);

  if (!parse_name_id(name)) return false;

  return true;
}

bool Parser::parse_explicit_instantiation(DeclarationAST*& yyast) {
  const auto start = currentLocation();

  SourceLocation externLoc;

  match(TokenKind::T_EXTERN, externLoc);

  SourceLocation templateLoc;

  if (!match(TokenKind::T_TEMPLATE, templateLoc)) {
    rewind(start);
    return false;
  }

  if (LA().is(TokenKind::T_LESS)) {
    rewind(start);
    return false;
  }

  DeclarationAST* declaration = nullptr;

  if (!parse_declaration(declaration)) parse_error("expected a declaration");

  auto ast = new (pool) ExplicitInstantiationAST();
  yyast = ast;

  ast->externLoc = externLoc;
  ast->templateLoc = templateLoc;
  ast->declaration = declaration;

  return true;
}

bool Parser::parse_explicit_specialization(DeclarationAST*& yyast) {
  SourceLocation templateLoc;

  if (!match(TokenKind::T_TEMPLATE, templateLoc)) return false;

  SourceLocation lessLoc;

  if (!match(TokenKind::T_LESS, lessLoc)) return false;

  SourceLocation greaterLoc;

  if (!match(TokenKind::T_GREATER, greaterLoc)) return false;

  DeclarationAST* declaration = nullptr;

  if (!parse_declaration(declaration)) parse_error("expected a declaration");

  auto ast = new (pool) TemplateDeclarationAST();
  yyast = ast;

  ast->templateLoc = templateLoc;
  ast->lessLoc = lessLoc;
  ast->greaterLoc = greaterLoc;
  ast->declaration = declaration;

  return true;
}

bool Parser::parse_try_block(StatementAST*& yyast) {
  SourceLocation tryLoc;

  if (!match(TokenKind::T_TRY, tryLoc)) return false;

  auto ast = new (pool) TryBlockStatementAST();
  yyast = ast;

  ast->tryLoc = tryLoc;

  if (!parse_compound_statement(ast->statement))
    parse_error("expected a compound statement");

  if (!parse_handler_seq(ast->handlerList))
    parse_error("expected an exception handler");

  return true;
}

bool Parser::parse_function_try_block(StatementAST*& yyast) {
  if (!match(TokenKind::T_TRY)) return false;

  if (LA().isNot(TokenKind::T_LBRACE)) {
    if (!parse_ctor_initializer()) parse_error("expected a ctor initializer");
  }

  StatementAST* statement = nullptr;

  if (!parse_compound_statement(statement))
    parse_error("expected a compound statement");

  List<HandlerAST*>* handlerList = nullptr;

  if (!parse_handler_seq(handlerList))
    parse_error("expected an exception handler");

  return true;
}

bool Parser::parse_handler(HandlerAST*& yyast) {
  SourceLocation catchLoc;

  if (!match(TokenKind::T_CATCH, catchLoc)) return false;

  yyast = new (pool) HandlerAST();

  yyast->catchLoc = catchLoc;

  expect(TokenKind::T_LPAREN, yyast->lparenLoc);

  if (!parse_exception_declaration(yyast->exceptionDeclaration))
    parse_error("expected an exception declaration");

  expect(TokenKind::T_RPAREN, yyast->rparenLoc);

  if (!parse_compound_statement(yyast->statement))
    parse_error("expected a compound statement");

  return true;
}

bool Parser::parse_handler_seq(List<HandlerAST*>*& yyast) {
  if (LA().isNot(TokenKind::T_CATCH)) return false;

  auto it = &yyast;

  while (LA().is(TokenKind::T_CATCH)) {
    HandlerAST* handler = nullptr;
    parse_handler(handler);
    *it = new (pool) List(handler);
    it = &(*it)->next;
  }

  return true;
}

bool Parser::parse_exception_declaration(ExceptionDeclarationAST*& yyast) {
  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool) EllipsisExceptionDeclarationAST();
    yyast = ast;

    ast->ellipsisLoc = ellipsisLoc;
    return true;
  }

  auto ast = new (pool) TypeExceptionDeclarationAST();
  yyast = ast;

  parse_attribute_specifier_seq(ast->attributeList);

  if (!parse_type_specifier_seq(ast->typeSpecifierList))
    parse_error("expected a type specifier");

  if (LA().is(TokenKind::T_RPAREN)) return true;

  const auto before_declarator = currentLocation();

  if (!parse_declarator(ast->declarator)) {
    rewind(before_declarator);

    if (!parse_abstract_declarator(ast->declarator)) rewind(before_declarator);
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
