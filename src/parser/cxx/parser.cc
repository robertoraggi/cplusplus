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

#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/private/format.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/types.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <forward_list>
#include <iostream>
#include <unordered_map>
#include <variant>

namespace cxx {

static auto getClassKey(TokenKind kind) -> ClassKey {
  switch (kind) {
    case TokenKind::T_CLASS:
      return ClassKey::kClass;
    case TokenKind::T_STRUCT:
      return ClassKey::kStruct;
    case TokenKind::T_UNION:
      return ClassKey::kUnion;
    default:
      cxx_runtime_error("invalid class key");
  }  // switch
}

static auto getFunctionDeclaratorHelper(DeclaratorAST* declarator)
    -> std::pair<FunctionDeclaratorAST*, bool> {
  if (!declarator) return std::make_pair(nullptr, false);

  if (auto n = dynamic_cast<NestedDeclaratorAST*>(declarator->coreDeclarator)) {
    auto [fundecl, done] = getFunctionDeclaratorHelper(n->declarator);

    if (done) return std::make_pair(fundecl, done);
  }

  std::vector<DeclaratorModifierAST*> modifiers;

  for (auto it = declarator->modifiers; it; it = it->next) {
    modifiers.push_back(it->value);
  }

  for (auto it = rbegin(modifiers); it != rend(modifiers); ++it) {
    auto modifier = *it;

    if (auto decl = dynamic_cast<FunctionDeclaratorAST*>(modifier)) {
      return std::make_pair(decl, true);
    }

    return std::make_pair(nullptr, true);
  }

  return std::make_pair(nullptr, false);
}

static auto getFunctionDeclarator(DeclaratorAST* declarator)
    -> FunctionDeclaratorAST* {
  return get<0>(getFunctionDeclaratorHelper(declarator));
}

Parser::Parser(TranslationUnit* unit) : unit(unit) {
  control = unit->control();
  symbols = control->symbols();
  types = control->types();
  cursor_ = 1;

  pool = unit->arena();

  module_id = control->identifier("module");
  import_id = control->identifier("import");
  final_id = control->identifier("final");
  override_id = control->identifier("override");
}

Parser::~Parser() = default;

auto Parser::checkTypes() const -> bool { return checkTypes_; }

void Parser::setCheckTypes(bool checkTypes) { checkTypes_ = checkTypes; }

auto Parser::prec(TokenKind tk) -> Parser::Prec {
  switch (tk) {
    default:
      cxx_runtime_error("expected a binary operator");

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

  [[nodiscard]] auto accepts_simple_typespec() const -> bool {
    return !(has_complex_typespec || has_named_typespec ||
             has_placeholder_typespec);
  }

  [[nodiscard]] auto has_typespec() const -> bool {
    return has_simple_typespec || has_complex_typespec || has_named_typespec ||
           has_placeholder_typespec;
  }
};

struct Parser::TemplArgContext {
  TemplArgContext(const TemplArgContext&) = delete;
  auto operator=(const TemplArgContext&) -> TemplArgContext& = delete;

  Parser* p;

  explicit TemplArgContext(Parser* p) : p(p) { ++p->templArgDepth; }
  ~TemplArgContext() { --p->templArgDepth; }
};

struct Parser::ClassSpecifierContext {
  ClassSpecifierContext(const ClassSpecifierContext&) = delete;
  auto operator=(const ClassSpecifierContext&)
      -> ClassSpecifierContext& = delete;

  Parser* p;

  explicit ClassSpecifierContext(Parser* p) : p(p) { ++p->classDepth; }

  ~ClassSpecifierContext() {
    if (--p->classDepth == 0) p->completePendingFunctionDefinitions();
  }
};

auto Parser::LA(int n) const -> const Token& {
  return unit->tokenAt(SourceLocation(cursor_ + n));
}

auto Parser::match(TokenKind tk) -> bool {
  if (LA().isNot(tk)) return false;
  (void)consumeToken();
  return true;
}

auto Parser::match(TokenKind tk, SourceLocation& location) -> bool {
  if (LA().isNot(tk)) return false;
  const auto loc = consumeToken();
  location = loc;
  return true;
}

auto Parser::expect(TokenKind tk) -> bool {
  if (match(tk)) return true;
  parse_error(fmt::format("expected '{}'", Token::spell(tk)));
  return false;
}

auto Parser::expect(TokenKind tk, SourceLocation& location) -> bool {
  if (match(tk, location)) return true;
  parse_error(fmt::format("expected '{}'", Token::spell(tk)));
  return false;
}

auto Parser::operator()(UnitAST*& ast) -> bool {
  auto result = parse(ast);
  return result;
}

auto Parser::parse(UnitAST*& ast) -> bool {
  auto parsed = parse_translation_unit(ast);

  return parsed;
}

auto Parser::parse_id(const Identifier* id) -> bool {
  SourceLocation identifierLoc;
  return parse_id(id, identifierLoc);
}

auto Parser::parse_id(const Identifier* id, SourceLocation& loc) -> bool {
  SourceLocation location;
  if (!match(TokenKind::T_IDENTIFIER, location)) return false;
  if (unit->identifier(location) != id) return false;
  loc = location;
  return true;
}

auto Parser::parse_nospace() -> bool {
  const auto& tk = unit->tokenAt(currentLocation());
  return !tk.leadingSpace() && !tk.startOfLine();
}

auto Parser::parse_greater_greater() -> bool {
  const auto saved = currentLocation();
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_GREATER)) {
    return true;
  }
  rewind(saved);
  return false;
}

auto Parser::parse_greater_greater_equal() -> bool {
  const auto saved = currentLocation();
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_EQUAL)) {
    return true;
  }
  rewind(saved);
  return false;
}

auto Parser::parse_greater_equal() -> bool {
  const auto saved = currentLocation();
  if (match(TokenKind::T_GREATER) && parse_nospace() &&
      match(TokenKind::T_EQUAL)) {
    return true;
  }
  rewind(saved);
  return false;
}

auto Parser::parse_header_name(SourceLocation& loc) -> bool {
  if (match(TokenKind::T_STRING_LITERAL, loc)) return true;

  // ### TODO
  return false;
}

auto Parser::parse_export_keyword(SourceLocation& loc) -> bool {
  if (!module_unit) return false;
  return match(TokenKind::T_EXPORT, loc);
}

auto Parser::parse_import_keyword(SourceLocation& loc) -> bool {
  if (!module_unit) return false;
  if (match(TokenKind::T_IMPORT, loc)) return true;
  if (!parse_id(import_id, loc)) return false;
  unit->setTokenKind(loc, TokenKind::T_IMPORT);
  return true;
}

auto Parser::parse_module_keyword(SourceLocation& loc) -> bool {
  if (!module_unit) return false;

  if (match(TokenKind::T_MODULE, loc)) return true;

  if (!parse_id(module_id, loc)) return false;

  unit->setTokenKind(loc, TokenKind::T_MODULE);
  return true;
}

auto Parser::parse_final() -> bool { return parse_id(final_id); }

auto Parser::parse_override() -> bool { return parse_id(override_id); }

auto Parser::parse_type_name(NameAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (parse_simple_template_id(yyast)) return true;

  rewind(start);

  return parse_name_id(yyast);
}

auto Parser::parse_name_id(NameAST*& yyast) -> bool {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = new (pool) SimpleNameAST();
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_literal(ExpressionAST*& yyast) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_CHARACTER_LITERAL: {
      auto ast = new (pool) CharLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal =
          static_cast<const CharLiteral*>(unit->literal(ast->literalLoc));

      return true;
    }

    case TokenKind::T_TRUE:
    case TokenKind::T_FALSE: {
      auto ast = new (pool) BoolLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal = unit->tokenKind(ast->literalLoc);

      return true;
    }

    case TokenKind::T_INTEGER_LITERAL: {
      auto ast = new (pool) IntLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal =
          static_cast<const IntegerLiteral*>(unit->literal(ast->literalLoc));

      return true;
    }

    case TokenKind::T_FLOATING_POINT_LITERAL: {
      auto ast = new (pool) FloatLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal =
          static_cast<const FloatLiteral*>(unit->literal(ast->literalLoc));

      return true;
    }

    case TokenKind::T_NULLPTR: {
      auto ast = new (pool) NullptrLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal = unit->tokenKind(ast->literalLoc);

      return true;
    }

    case TokenKind::T_USER_DEFINED_STRING_LITERAL: {
      auto ast = new (pool) UserDefinedStringLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal =
          static_cast<const StringLiteral*>(unit->literal(ast->literalLoc));

      return true;
    }

    case TokenKind::T_WIDE_STRING_LITERAL:
    case TokenKind::T_UTF8_STRING_LITERAL:
    case TokenKind::T_UTF16_STRING_LITERAL:
    case TokenKind::T_UTF32_STRING_LITERAL:
    case TokenKind::T_STRING_LITERAL: {
      List<SourceLocation>* stringLiterals = nullptr;

      parse_string_literal_seq(stringLiterals);

      auto ast = new (pool) StringLiteralExpressionAST();
      yyast = ast;

      ast->stringLiteralList = stringLiterals;

      if (ast->stringLiteralList) {
        ast->literal = static_cast<const StringLiteral*>(
            unit->literal(ast->stringLiteralList->value));
      }

      return true;
    }

    default:
      return false;
  }  // switch
}

auto Parser::parse_translation_unit(UnitAST*& yyast) -> bool {
  if (parse_module_unit(yyast)) return true;
  parse_top_level_declaration_seq(yyast);
  // globalRegion->dump(std::cout);
  return true;
}

auto Parser::parse_module_head() -> bool {
  const auto start = currentLocation();
  match(TokenKind::T_EXPORT);
  const auto is_module = parse_id(module_id);
  rewind(start);
  return is_module;
}

auto Parser::parse_module_unit(UnitAST*& yyast) -> bool {
  module_unit = true;

  if (!parse_module_head()) return false;

  auto ast = new (pool) ModuleUnitAST();
  yyast = ast;

  parse_global_module_fragment(ast->globalModuleFragment);

  if (!parse_module_declaration(ast->moduleDeclaration)) {
    parse_error("expected a module declaration");
  }

  parse_declaration_seq(ast->declarationList);

  parse_private_module_fragment(ast->privateModuleFragment);

  expect(TokenKind::T_EOF_SYMBOL);

  return true;
}

auto Parser::parse_top_level_declaration_seq(UnitAST*& yyast) -> bool {
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

auto Parser::parse_skip_top_level_declaration(bool& skipping) -> bool {
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
  return true;
}

auto Parser::parse_declaration_seq(List<DeclarationAST*>*& yyast) -> bool {
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

auto Parser::parse_skip_declaration(bool& skipping) -> bool {
  if (LA().is(TokenKind::T_RBRACE)) return false;
  if (LA().is(TokenKind::T_MODULE)) return false;
  if (module_unit && LA().is(TokenKind::T_EXPORT)) return false;
  if (LA().is(TokenKind::T_IMPORT)) return false;
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
  return true;
}

auto Parser::parse_primary_expression(ExpressionAST*& yyast,
                                      bool inRequiresClause) -> bool {
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

  if (!parse_id_expression(name, inRequiresClause)) return false;

  auto ast = new (pool) IdExpressionAST();
  yyast = ast;

  ast->name = name;

  return true;
}

auto Parser::parse_id_expression(NameAST*& yyast, bool inRequiresClause)
    -> bool {
  const auto start = currentLocation();

  if (parse_qualified_id(yyast, inRequiresClause)) return true;

  rewind(start);

  yyast = nullptr;

  if (!parse_unqualified_id(yyast, inRequiresClause)) return false;

  return true;
}

auto Parser::parse_maybe_template_id(NameAST*& yyast, bool inRequiresClause)
    -> bool {
  const auto blockErrors = unit->blockErrors(true);

  auto template_id = parse_template_id(yyast);

  const auto& tk = LA();

  unit->blockErrors(blockErrors);

  if (!template_id) return false;

  if (inRequiresClause) return true;

  switch (TokenKind(tk)) {
    case TokenKind::T_COMMA:
    case TokenKind::T_LPAREN:
    case TokenKind::T_RPAREN:
    case TokenKind::T_COLON_COLON:
    case TokenKind::T_DOT_DOT_DOT:
    case TokenKind::T_QUESTION:
    case TokenKind::T_LBRACE:
    case TokenKind::T_RBRACE:
    case TokenKind::T_LBRACKET:
    case TokenKind::T_RBRACKET:
    case TokenKind::T_SEMICOLON:
    case TokenKind::T_EQUAL:
    case TokenKind::T_DOT:
    case TokenKind::T_MINUS_GREATER:
    case TokenKind::T_AMP_AMP:
    case TokenKind::T_BAR_BAR:
      return true;

    default: {
      SourceLocation loc;
      TokenKind tk = TokenKind::T_EOF_SYMBOL;
      ExprContext ctx;
      if (parse_lookahead_binary_operator(loc, tk, ctx)) {
        return true;
      }
      yyast = nullptr;
      return false;
    }
  }  // switch
}

auto Parser::parse_unqualified_id(NameAST*& yyast, bool inRequiresClause)
    -> bool {
  const auto start = currentLocation();

  if (parse_maybe_template_id(yyast, inRequiresClause)) return true;

  rewind(start);

  SourceLocation tildeLoc;

  if (match(TokenKind::T_TILDE, tildeLoc)) {
    SpecifierAST* decltypeSpecifier = nullptr;

    if (parse_decltype_specifier(decltypeSpecifier)) {
      auto decltypeName = new (pool) DecltypeNameAST();
      decltypeName->decltypeSpecifier = decltypeSpecifier;

      auto ast = new (pool) DestructorNameAST();
      yyast = ast;

      ast->id = decltypeName;

      return true;
    }

    NameAST* name = nullptr;

    if (!parse_type_name(name)) return false;

    auto ast = new (pool) DestructorNameAST();
    yyast = ast;

    ast->id = name;

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

auto Parser::parse_qualified_id(NameAST*& yyast, bool inRequiresClause)
    -> bool {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

  SourceLocation templateLoc;

  match(TokenKind::T_TEMPLATE, templateLoc);

  NameAST* name = nullptr;

  const auto hasName = parse_unqualified_id(name, inRequiresClause);

  if (hasName || templateLoc) {
    auto ast = new (pool) QualifiedNameAST();
    yyast = ast;

    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->templateLoc = templateLoc;
    ast->id = name;

    if (!hasName) parse_error("expected a template name");

    return true;
  }

  return false;
}

auto Parser::parse_start_of_nested_name_specifier(NameAST*& yyast,
                                                  SourceLocation& scopeLoc)
    -> bool {
  yyast = nullptr;

  if (match(TokenKind::T_COLON_COLON, scopeLoc)) return true;

  SpecifierAST* decltypeSpecifier = nullptr;

  if (parse_decltype_specifier(decltypeSpecifier) &&
      match(TokenKind::T_COLON_COLON, scopeLoc)) {
    return true;
  }

  const auto start = currentLocation();

  if (parse_name_id(yyast) && match(TokenKind::T_COLON_COLON, scopeLoc)) {
    return true;
  }

  rewind(start);

  if (parse_simple_template_id(yyast) &&
      match(TokenKind::T_COLON_COLON, scopeLoc)) {
    return true;
  }

  return false;
}

auto Parser::parse_nested_name_specifier(NestedNameSpecifierAST*& yyast)
    -> bool {
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

    explicit Context(Parser* p) : p(p), start(p->currentLocation()) {}

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

  if (!name) {
    ast->scopeLoc = scopeLoc;
  } else {
    *nameIt = new (pool) List(name);
    nameIt = &(*nameIt)->next;
  }

  while (true) {
    if (LA().is(TokenKind::T_IDENTIFIER) &&
        LA(1).is(TokenKind::T_COLON_COLON)) {
      NameAST* name = nullptr;
      parse_name_id(name);
      expect(TokenKind::T_COLON_COLON);

      *nameIt = new (pool) List(name);
      nameIt = &(*nameIt)->next;
    } else {
      const auto saved = currentLocation();

      match(TokenKind::T_TEMPLATE);

      NameAST* name = nullptr;

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

auto Parser::parse_lambda_expression(ExpressionAST*& yyast) -> bool {
  LambdaIntroducerAST* lambdaIntroducer = nullptr;

  if (!parse_lambda_introducer(lambdaIntroducer)) return false;

  auto ast = new (pool) LambdaExpressionAST();
  yyast = ast;

  ast->lambdaIntroducer = lambdaIntroducer;

  if (match(TokenKind::T_LESS, ast->lessLoc)) {
    if (!parse_template_parameter_list(ast->templateParameterList)) {
      parse_error("expected a template paramter");
    }

    expect(TokenKind::T_GREATER, ast->greaterLoc);

    parse_requires_clause(ast->requiresClause);
  }

  if (LA().isNot(TokenKind::T_LBRACE)) {
    if (!parse_lambda_declarator(ast->lambdaDeclarator)) {
      parse_error("expected lambda declarator");
    }
  }

  if (!parse_compound_statement(ast->statement)) {
    parse_error("expected a compound statement");
  }

  return true;
}

auto Parser::parse_lambda_introducer(LambdaIntroducerAST*& yyast) -> bool {
  SourceLocation lbracketLoc;

  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  SourceLocation rbracketLoc;
  SourceLocation defaultCaptureLoc;
  List<LambdaCaptureAST*>* captureList = nullptr;

  if (!match(TokenKind::T_RBRACKET, rbracketLoc)) {
    if (!parse_lambda_capture(defaultCaptureLoc, captureList)) {
      parse_error("expected a lambda capture");
    }

    expect(TokenKind::T_RBRACKET, rbracketLoc);
  }

  auto ast = new (pool) LambdaIntroducerAST();
  yyast = ast;

  ast->lbracketLoc = lbracketLoc;
  ast->captureDefaultLoc = defaultCaptureLoc;
  ast->captureList = captureList;
  ast->rbracketLoc = rbracketLoc;

  return true;
}

auto Parser::parse_lambda_declarator(LambdaDeclaratorAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = new (pool) LambdaDeclaratorAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_parameter_declaration_clause(ast->parameterDeclarationClause)) {
      parse_error("expected a parameter declaration clause");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  DeclSpecs specs;

  parse_decl_specifier_seq(ast->declSpecifierList, specs);

  parse_noexcept_specifier();

  parse_attribute_specifier_seq(ast->attributeList);

  parse_trailing_return_type(ast->trailingReturnType);

  parse_requires_clause(ast->requiresClause);

  return true;
}

auto Parser::parse_lambda_capture(SourceLocation& captureDefaultLoc,
                                  List<LambdaCaptureAST*>*& captureList)
    -> bool {
  if (parse_capture_default(captureDefaultLoc)) {
    if (match(TokenKind::T_COMMA)) {
      if (!parse_capture_list(captureList)) parse_error("expected a capture");
    }

    return true;
  }

  return parse_capture_list(captureList);
}

auto Parser::parse_capture_default(SourceLocation& opLoc) -> bool {
  const auto start = currentLocation();

  if (!match(TokenKind::T_AMP) && !match(TokenKind::T_EQUAL)) return false;

  if (LA().isNot(TokenKind::T_COMMA) && LA().isNot(TokenKind::T_RBRACKET)) {
    rewind(start);
    return false;
  }

  opLoc = start;

  return true;
}

auto Parser::parse_capture_list(List<LambdaCaptureAST*>*& yyast) -> bool {
  auto it = &yyast;

  LambdaCaptureAST* capture = nullptr;

  if (!parse_capture(capture)) return false;

  if (capture) {
    *it = new (pool) List(capture);
    it = &(*it)->next;
  }

  while (match(TokenKind::T_COMMA)) {
    LambdaCaptureAST* capture = nullptr;

    if (!parse_capture(capture)) parse_error("expected a capture");

    if (capture) {
      *it = new (pool) List(capture);
      it = &(*it)->next;
    }
  }

  return true;
}

auto Parser::parse_capture(LambdaCaptureAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (parse_init_capture(yyast)) return true;

  rewind(start);

  return parse_simple_capture(yyast);
}

auto Parser::parse_simple_capture(LambdaCaptureAST*& yyast) -> bool {
  SourceLocation thisLoc;

  if (match(TokenKind::T_THIS, thisLoc)) {
    auto ast = new (pool) ThisLambdaCaptureAST();
    yyast = ast;

    ast->thisLoc = thisLoc;

    return true;
  }

  SourceLocation identifierLoc;

  if (match(TokenKind::T_IDENTIFIER, identifierLoc)) {
    auto ast = new (pool) SimpleLambdaCaptureAST();
    yyast = ast;

    ast->identifierLoc = identifierLoc;
    ast->identifier = unit->identifier(ast->identifierLoc);
    match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

    return true;
  }

  if (LA().is(TokenKind::T_STAR) && LA(1).is(TokenKind::T_THIS)) {
    auto ast = new (pool) DerefThisLambdaCaptureAST();
    yyast = ast;

    ast->starLoc = consumeToken();
    ast->thisLoc = consumeToken();

    return true;
  }

  SourceLocation ampLoc;

  if (!match(TokenKind::T_AMP, ampLoc)) return false;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = new (pool) RefLambdaCaptureAST();
  yyast = ast;

  ast->ampLoc = ampLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  return true;
}

auto Parser::parse_init_capture(LambdaCaptureAST*& yyast) -> bool {
  SourceLocation ampLoc;

  if (match(TokenKind::T_AMP, ampLoc)) {
    SourceLocation ellipsisLoc;

    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    SourceLocation identifierLoc;

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    InitializerAST* initializer = nullptr;

    if (!parse_initializer(initializer)) return false;

    auto ast = new (pool) RefInitLambdaCaptureAST();
    yyast = ast;

    ast->ampLoc = ampLoc;
    ast->ellipsisLoc = ellipsisLoc;
    ast->identifierLoc = identifierLoc;
    ast->identifier = unit->identifier(ast->identifierLoc);
    ast->initializer = initializer;

    return true;
  }

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  InitializerAST* initializer = nullptr;

  if (!parse_initializer(initializer)) return false;

  auto ast = new (pool) InitLambdaCaptureAST();
  yyast = ast;

  ast->ellipsisLoc = ellipsisLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);
  ast->initializer = initializer;

  return true;
}

auto Parser::parse_fold_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool) LeftFoldExpressionAST();
    yyast = ast;

    ast->lparenLoc = lparenLoc;
    ast->ellipsisLoc = ellipsisLoc;

    if (!parse_fold_operator(ast->opLoc, ast->op)) {
      parse_error("expected fold operator");
    }

    if (!parse_cast_expression(ast->expression)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) return false;

  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  if (!parse_fold_operator(opLoc, op)) return false;

  if (!match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) return false;

  SourceLocation rparenLoc;

  if (match(TokenKind::T_RPAREN, rparenLoc)) {
    auto ast = new (pool) RightFoldExpressionAST();
    yyast = ast;

    ast->lparenLoc = lparenLoc;
    ast->expression = expression;
    ast->opLoc = opLoc;
    ast->op = op;
    ast->ellipsisLoc = ellipsisLoc;
    ast->rparenLoc = rparenLoc;

    return true;
  }

  auto ast = new (pool) FoldExpressionAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->leftExpression = expression;
  ast->opLoc = opLoc;
  ast->op = op;
  ast->ellipsisLoc = ellipsisLoc;

  if (!parse_fold_operator(ast->foldOpLoc, ast->foldOp)) {
    parse_error("expected a fold operator");
  }

  if (!parse_cast_expression(ast->rightExpression)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_fold_operator(SourceLocation& loc, TokenKind& op) -> bool {
  loc = currentLocation();

  switch (TokenKind(LA())) {
    case TokenKind::T_GREATER: {
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

auto Parser::parse_requires_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation requiresLoc;

  if (!match(TokenKind::T_REQUIRES, requiresLoc)) return false;

  auto ast = new (pool) RequiresExpressionAST();
  yyast = ast;

  ast->requiresLoc = requiresLoc;

  if (LA().isNot(TokenKind::T_LBRACE)) {
    if (!parse_requirement_parameter_list(
            ast->lparenLoc, ast->parameterDeclarationClause, ast->rparenLoc)) {
      parse_error("expected a requirement parameter");
    }
  }

  if (!parse_requirement_body(ast->requirementBody)) {
    parse_error("expected a requirement body");
  }

  return true;
}

auto Parser::parse_requirement_parameter_list(
    SourceLocation& lparenLoc,
    ParameterDeclarationClauseAST*& parameterDeclarationClause,
    SourceLocation& rparenLoc) -> bool {
  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_parameter_declaration_clause(parameterDeclarationClause)) {
      parse_error("expected a parmater declaration");
    }

    expect(TokenKind::T_RPAREN, rparenLoc);
  }

  return true;
}

auto Parser::parse_requirement_body(RequirementBodyAST*& yyast) -> bool {
  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  yyast = new (pool) RequirementBodyAST();
  yyast->lbraceLoc = lbraceLoc;

  if (!parse_requirement_seq(yyast->requirementList)) {
    parse_error("expected a requirement");
  }

  expect(TokenKind::T_RBRACE, yyast->rbraceLoc);

  return true;
}

auto Parser::parse_requirement_seq(List<RequirementAST*>*& yyast) -> bool {
  auto it = &yyast;

  bool skipping = false;

  RequirementAST* requirement = nullptr;

  if (!parse_requirement(requirement)) return false;

  *it = new (pool) List(requirement);
  it = &(*it)->next;

  while (LA()) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    const auto before_requirement = currentLocation();

    RequirementAST* requirement = nullptr;

    if (parse_requirement(requirement)) {
      skipping = false;

      *it = new (pool) List(requirement);
      it = &(*it)->next;
    } else {
      if (!skipping) parse_error("expected a requirement");
      skipping = true;
      if (currentLocation() == before_requirement) consumeToken();
    }
  }

  return true;
}

auto Parser::parse_requirement(RequirementAST*& yyast) -> bool {
  if (parse_nested_requirement(yyast)) return true;

  if (parse_compound_requirement(yyast)) return true;

  if (parse_type_requirement(yyast)) return true;

  return parse_simple_requirement(yyast);
}

auto Parser::parse_simple_requirement(RequirementAST*& yyast) -> bool {
  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) return false;

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool) SimpleRequirementAST();
  yyast = ast;

  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_type_requirement(RequirementAST*& yyast) -> bool {
  SourceLocation typenameLoc;

  if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

  auto ast = new (pool) TypeRequirementAST();
  yyast = ast;

  const auto after_typename = currentLocation();

  if (!parse_nested_name_specifier(ast->nestedNameSpecifier)) {
    rewind(after_typename);
  }

  if (!parse_type_name(ast->name)) parse_error("expected a type name");

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_compound_requirement(RequirementAST*& yyast) -> bool {
  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) return false;

  SourceLocation rbraceLoc;

  if (!match(TokenKind::T_RBRACE, rbraceLoc)) return false;

  auto ast = new (pool) CompoundRequirementAST();
  yyast = ast;

  ast->lbraceLoc = lbraceLoc;
  ast->expression = expression;
  ast->rbraceLoc = rbraceLoc;

  match(TokenKind::T_NOEXCEPT, ast->noexceptLoc);

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    if (!parse_return_type_requirement(ast->minusGreaterLoc,
                                       ast->typeConstraint)) {
      parse_error("expected return type requirement");
    }

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  return true;
}

auto Parser::parse_return_type_requirement(SourceLocation& minusGreaterLoc,
                                           TypeConstraintAST*& typeConstraint)
    -> bool {
  if (!match(TokenKind::T_MINUS_GREATER, minusGreaterLoc)) return false;

  if (!parse_type_constraint(typeConstraint, /*parsing placeholder=*/false)) {
    parse_error("expected type constraint");
  }

  return true;
}

auto Parser::parse_nested_requirement(RequirementAST*& yyast) -> bool {
  SourceLocation requiresLoc;

  if (!match(TokenKind::T_REQUIRES, requiresLoc)) return false;

  auto ast = new (pool) NestedRequirementAST();
  yyast = ast;

  ast->requiresLoc = requiresLoc;

  if (!parse_constraint_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_postfix_expression(ExpressionAST*& yyast) -> bool {
  if (!parse_start_of_postfix_expression(yyast)) return false;

  while (true) {
    const auto saved = currentLocation();
    if (parse_member_expression(yyast)) continue;
    if (parse_subscript_expression(yyast)) continue;
    if (parse_call_expression(yyast)) continue;
    if (parse_postincr_expression(yyast)) continue;
    rewind(saved);
    break;
  }

  return true;
}

auto Parser::parse_start_of_postfix_expression(ExpressionAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (parse_cpp_cast_expression(yyast)) return true;

  if (parse_typeid_expression(yyast)) return true;

  if (parse_builtin_call_expression(yyast)) return true;

  if (parse_typename_expression(yyast)) return true;

  if (parse_cpp_type_cast_expression(yyast)) return true;

  rewind(start);
  return parse_primary_expression(yyast);
}

auto Parser::parse_member_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation accessLoc;

  if (!match(TokenKind::T_DOT, accessLoc) &&
      !match(TokenKind::T_MINUS_GREATER, accessLoc)) {
    return false;
  }

  auto ast = new (pool) MemberExpressionAST();
  ast->baseExpression = yyast;
  ast->accessLoc = accessLoc;
  ast->accessOp = unit->tokenKind(accessLoc);

  yyast = ast;

  match(TokenKind::T_TEMPLATE, ast->templateLoc);

  if (!parse_id_expression(ast->name)) parse_error("expected a member name");

  return true;
}

auto Parser::parse_subscript_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation lbracketLoc;

  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  auto ast = new (pool) SubscriptExpressionAST();
  ast->baseExpression = yyast;
  ast->lbracketLoc = lbracketLoc;

  yyast = ast;

  if (!parse_expr_or_braced_init_list(ast->indexExpression)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_RBRACKET, ast->rbracketLoc);

  return true;
}

auto Parser::parse_call_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = new (pool) CallExpressionAST();
  ast->baseExpression = yyast;
  ast->lparenLoc = lparenLoc;

  yyast = ast;

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression_list(ast->expressionList)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  return true;
}

auto Parser::parse_postincr_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation opLoc;

  if (!match(TokenKind::T_MINUS_MINUS, opLoc) &&
      !match(TokenKind::T_PLUS_PLUS, opLoc)) {
    return false;
  }

  auto ast = new (pool) PostIncrExpressionAST();
  ast->baseExpression = yyast;
  ast->opLoc = opLoc;
  ast->op = unit->tokenKind(ast->opLoc);
  yyast = ast;

  return true;
}

auto Parser::parse_cpp_cast_head(SourceLocation& castLoc) -> bool {
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

auto Parser::parse_cpp_cast_expression(ExpressionAST*& yyast) -> bool {
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

auto Parser::parse_cpp_type_cast_expression(ExpressionAST*& yyast) -> bool {
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

  if (parse_braced_init_list(bracedInitList)) {
    auto ast = new (pool) BracedTypeConstructionAST();
    yyast = ast;

    ast->typeSpecifier = typeSpecifier;
    ast->bracedInitList = bracedInitList;

    return true;
  }

  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  SourceLocation rparenLoc;

  List<ExpressionAST*>* expressionList = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_expression_list(expressionList)) return false;

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;
  }

  auto ast = new (pool) TypeConstructionAST();
  yyast = ast;

  ast->typeSpecifier = typeSpecifier;
  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_typeid_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation typeidLoc;

  if (!match(TokenKind::T_TYPEID, typeidLoc)) return false;

  SourceLocation lparenLoc;

  expect(TokenKind::T_LPAREN, lparenLoc);

  const auto saved = currentLocation();

  TypeIdAST* typeId = nullptr;

  SourceLocation rparenLoc;

  if (parse_type_id(typeId) && match(TokenKind::T_RPAREN, rparenLoc)) {
    auto ast = new (pool) TypeidOfTypeExpressionAST();
    yyast = ast;

    ast->typeidLoc = typeidLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;

    return true;
  }

  rewind(saved);

  auto ast = new (pool) TypeidExpressionAST();
  yyast = ast;

  ast->typeidLoc = typeidLoc;
  ast->lparenLoc = lparenLoc;

  if (!parse_expression(ast->expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_typename_expression(ExpressionAST*& yyast) -> bool {
  SpecifierAST* typenameSpecifier = nullptr;

  if (!parse_typename_specifier(typenameSpecifier)) return false;

  BracedInitListAST* bracedInitList = nullptr;

  if (parse_braced_init_list(bracedInitList)) {
    auto ast = new (pool) BracedTypeConstructionAST();
    yyast = ast;

    ast->typeSpecifier = typenameSpecifier;
    ast->bracedInitList = bracedInitList;

    return true;
  }

  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  SourceLocation rparenLoc;

  List<ExpressionAST*>* expressionList = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_expression_list(expressionList)) return false;

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;
  }

  auto ast = new (pool) TypeConstructionAST();
  yyast = ast;

  ast->typeSpecifier = typenameSpecifier;
  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_type_traits_op(SourceLocation& loc) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T___HAS_UNIQUE_OBJECT_REPRESENTATIONS:
    case TokenKind::T___HAS_VIRTUAL_DESTRUCTOR:
    case TokenKind::T___IS_ABSTRACT:
    case TokenKind::T___IS_AGGREGATE:
    case TokenKind::T___IS_ARITHMETIC:
    case TokenKind::T___IS_ARRAY:
    case TokenKind::T___IS_ASSIGNABLE:
    case TokenKind::T___IS_BASE_OF:
    case TokenKind::T___IS_BOUNDED_ARRAY:
    case TokenKind::T___IS_CLASS:
    case TokenKind::T___IS_COMPOUND:
    case TokenKind::T___IS_CONST:
    case TokenKind::T___IS_CONSTRUCTIBLE:
    case TokenKind::T___IS_CONVERTIBLE:
    case TokenKind::T___IS_COPY_ASSIGNABLE:
    case TokenKind::T___IS_COPY_CONSTRUCTIBLE:
    case TokenKind::T___IS_DEFAULT_CONSTRUCTIBLE:
    case TokenKind::T___IS_DESTRUCTIBLE:
    case TokenKind::T___IS_EMPTY:
    case TokenKind::T___IS_ENUM:
    case TokenKind::T___IS_FINAL:
    case TokenKind::T___IS_FLOATING_POINT:
    case TokenKind::T___IS_FUNCTION:
    case TokenKind::T___IS_FUNDAMENTAL:
    case TokenKind::T___IS_INTEGRAL:
    case TokenKind::T___IS_INVOCABLE:
    case TokenKind::T___IS_INVOCABLE_R:
    case TokenKind::T___IS_LAYOUT_COMPATIBLE:
    case TokenKind::T___IS_LITERAL_TYPE:
    case TokenKind::T___IS_LVALUE_REFERENCE:
    case TokenKind::T___IS_MEMBER_FUNCTION_POINTER:
    case TokenKind::T___IS_MEMBER_OBJECT_POINTER:
    case TokenKind::T___IS_MEMBER_POINTER:
    case TokenKind::T___IS_MOVE_ASSIGNABLE:
    case TokenKind::T___IS_MOVE_CONSTRUCTIBLE:
    case TokenKind::T___IS_NOTHROW_ASSIGNABLE:
    case TokenKind::T___IS_NOTHROW_CONSTRUCTIBLE:
    case TokenKind::T___IS_NOTHROW_CONVERTIBLE:
    case TokenKind::T___IS_NOTHROW_COPY_ASSIGNABLE:
    case TokenKind::T___IS_NOTHROW_COPY_CONSTRUCTIBLE:
    case TokenKind::T___IS_NOTHROW_DEFAULT_CONSTRUCTIBLE:
    case TokenKind::T___IS_NOTHROW_DESTRUCTIBLE:
    case TokenKind::T___IS_NOTHROW_INVOCABLE:
    case TokenKind::T___IS_NOTHROW_INVOCABLE_R:
    case TokenKind::T___IS_NOTHROW_MOVE_ASSIGNABLE:
    case TokenKind::T___IS_NOTHROW_MOVE_CONSTRUCTIBLE:
    case TokenKind::T___IS_NOTHROW_SWAPPABLE:
    case TokenKind::T___IS_NOTHROW_SWAPPABLE_WITH:
    case TokenKind::T___IS_NULL_POINTER:
    case TokenKind::T___IS_OBJECT:
    case TokenKind::T___IS_POD:
    case TokenKind::T___IS_POINTER:
    case TokenKind::T___IS_POINTER_INTERCONVERTIBLE_BASE_OF:
    case TokenKind::T___IS_POLYMORPHIC:
    case TokenKind::T___IS_REFERENCE:
    case TokenKind::T___IS_RVALUE_REFERENCE:
    case TokenKind::T___IS_SAME:
    case TokenKind::T___IS_SCALAR:
    case TokenKind::T___IS_SCOPED_ENUM:
    case TokenKind::T___IS_SIGNED:
    case TokenKind::T___IS_STANDARD_LAYOUT:
    case TokenKind::T___IS_SWAPPABLE:
    case TokenKind::T___IS_SWAPPABLE_WITH:
    case TokenKind::T___IS_TRIVIAL:
    case TokenKind::T___IS_TRIVIALLY_ASSIGNABLE:
    case TokenKind::T___IS_TRIVIALLY_CONSTRUCTIBLE:
    case TokenKind::T___IS_TRIVIALLY_COPY_ASSIGNABLE:
    case TokenKind::T___IS_TRIVIALLY_COPY_CONSTRUCTIBLE:
    case TokenKind::T___IS_TRIVIALLY_COPYABLE:
    case TokenKind::T___IS_TRIVIALLY_DEFAULT_CONSTRUCTIBLE:
    case TokenKind::T___IS_TRIVIALLY_DESTRUCTIBLE:
    case TokenKind::T___IS_TRIVIALLY_MOVE_ASSIGNABLE:
    case TokenKind::T___IS_TRIVIALLY_MOVE_CONSTRUCTIBLE:
    case TokenKind::T___IS_UNBOUNDED_ARRAY:
    case TokenKind::T___IS_UNION:
    case TokenKind::T___IS_UNSIGNED:
    case TokenKind::T___IS_VOID:
    case TokenKind::T___IS_VOLATILE:
    case TokenKind::T___REFERENCE_BINDS_TO_TEMPORARY:
      loc = consumeToken();
      return true;
    default:
      return false;
  }  // switch
}

auto Parser::parse_builtin_call_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation typeTraitsLoc;

  if (!parse_type_traits_op(typeTraitsLoc)) return false;

  auto ast = new (pool) TypeTraitsExpressionAST();
  yyast = ast;

  ast->typeTraitsLoc = typeTraitsLoc;
  ast->typeTraits = unit->tokenKind(typeTraitsLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  auto it = &ast->typeIdList;

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) {
    parse_error("expected a type id");
  } else {
    *it = new (pool) List(typeId);
    it = &(*it)->next;
  }

  while (match(TokenKind::T_COMMA)) {
    TypeIdAST* typeId = nullptr;

    if (!parse_type_id(typeId)) {
      parse_error("expected a type id");
    } else {
      *it = new (pool) List(typeId);
      it = &(*it)->next;
    }
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_expression_list(List<ExpressionAST*>*& yyast) -> bool {
  return parse_initializer_list(yyast);
}

auto Parser::parse_unary_expression(ExpressionAST*& yyast) -> bool {
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

auto Parser::parse_unop_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation opLoc;

  if (!parse_unary_operator(opLoc)) return false;

  auto ast = new (pool) UnaryExpressionAST();
  yyast = ast;

  ast->opLoc = opLoc;
  ast->op = unit->tokenKind(opLoc);

  if (!parse_cast_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  return true;
}

auto Parser::parse_complex_expression(ExpressionAST*& yyast) -> bool {
  if (!match(TokenKind::T___IMAG__) && !match(TokenKind::T___REAL__)) {
    return false;
  }

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) parse_error("expected an expression");

  return true;
}

auto Parser::parse_sizeof_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation sizeofLoc;

  if (!match(TokenKind::T_SIZEOF, sizeofLoc)) return false;

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool) SizeofPackExpressionAST();
    yyast = ast;

    ast->sizeofLoc = sizeofLoc;
    ast->ellipsisLoc = ellipsisLoc;

    expect(TokenKind::T_LPAREN, ast->lparenLoc);

    expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
    ast->identifier = unit->identifier(ast->identifierLoc);

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  const auto after_sizeof_op = currentLocation();

  SourceLocation lparenLoc;
  TypeIdAST* typeId = nullptr;
  SourceLocation rparenLoc;

  if (match(TokenKind::T_LPAREN, lparenLoc) && parse_type_id(typeId) &&
      match(TokenKind::T_RPAREN, rparenLoc)) {
    auto ast = new (pool) SizeofTypeExpressionAST();
    yyast = ast;

    ast->sizeofLoc = sizeofLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;

    return true;
  }

  rewind(after_sizeof_op);

  auto ast = new (pool) SizeofExpressionAST();
  yyast = ast;

  ast->sizeofLoc = sizeofLoc;

  if (!parse_unary_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  return true;
}

auto Parser::parse_alignof_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation alignofLoc;

  if (!match(TokenKind::T_ALIGNOF, alignofLoc)) return false;

  auto ast = new (pool) AlignofExpressionAST();
  yyast = ast;

  ast->alignofLoc = alignofLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_unary_operator(SourceLocation& opLoc) -> bool {
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

auto Parser::parse_await_expression(ExpressionAST*& yyast) -> bool {
  if (!match(TokenKind::T_CO_AWAIT)) return false;

  expect(TokenKind::T_LPAREN);

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN);

  return true;
}

auto Parser::parse_noexcept_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation noexceptLoc;

  if (!match(TokenKind::T_NOEXCEPT, noexceptLoc)) return false;

  auto ast = new (pool) NoexceptExpressionAST();
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_expression(ast->expression)) parse_error("expected an expression");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_new_expression(ExpressionAST*& yyast) -> bool {
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

auto Parser::parse_new_placement() -> bool {
  if (!match(TokenKind::T_LPAREN)) return false;

  List<ExpressionAST*>* expressionList = nullptr;

  if (!parse_expression_list(expressionList)) return false;

  if (!match(TokenKind::T_RPAREN)) return false;

  return true;
}

auto Parser::parse_new_type_id(NewTypeIdAST*& yyast) -> bool {
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList)) return false;

  auto ast = new (pool) NewTypeIdAST();
  yyast = ast;

  ast->typeSpecifierList = typeSpecifierList;

  const auto saved = currentLocation();

  if (!parse_new_declarator()) rewind(saved);

  return true;
}

auto Parser::parse_new_declarator() -> bool {
  PtrOperatorAST* ptrOp = nullptr;

  if (parse_ptr_operator(ptrOp)) {
    auto saved = currentLocation();

    if (!parse_new_declarator()) rewind(saved);

    return true;
  }

  return parse_noptr_new_declarator();
}

auto Parser::parse_noptr_new_declarator() -> bool {
  if (!match(TokenKind::T_LBRACKET)) return false;

  if (!match(TokenKind::T_RBRACKET)) {
    ExpressionAST* expression = nullptr;

    if (!parse_expression(expression)) parse_error("expected an expression");

    expect(TokenKind::T_RBRACKET);
  }

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  while (match(TokenKind::T_LBRACKET)) {
    if (!match(TokenKind::T_RBRACKET)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(expression)) {
        parse_error("expected an expression");
      }

      expect(TokenKind::T_RBRACKET);
    }

    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);
  }

  return true;
}

auto Parser::parse_new_initializer(NewInitializerAST*& yyast) -> bool {
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

auto Parser::parse_delete_expression(ExpressionAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation scopeLoc;

  match(TokenKind::T_COLON_COLON, scopeLoc);

  SourceLocation deleteLoc;

  if (!match(TokenKind::T_DELETE, deleteLoc)) {
    rewind(start);
    return false;
  }

  auto ast = new (pool) DeleteExpressionAST();
  yyast = ast;

  ast->scopeLoc = scopeLoc;
  ast->deleteLoc = deleteLoc;

  if (match(TokenKind::T_LBRACKET, ast->lbracketLoc)) {
    expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  }

  if (!parse_cast_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  return true;
}

auto Parser::parse_cast_expression(ExpressionAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (parse_cast_expression_helper(yyast)) return true;

  rewind(start);

  return parse_unary_expression(yyast);
}

auto Parser::parse_cast_expression_helper(ExpressionAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) return false;

  SourceLocation rparenLoc;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression)) return false;

  auto ast = new (pool) CastExpressionAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->typeId = typeId;
  ast->rparenLoc = rparenLoc;
  ast->expression = expression;

  return true;
}

auto Parser::parse_binary_operator(SourceLocation& loc, TokenKind& tk,
                                   const ExprContext& exprContext) -> bool {
  const auto start = currentLocation();

  loc = start;
  tk = TokenKind::T_EOF_SYMBOL;

  switch (TokenKind(LA())) {
    case TokenKind::T_GREATER: {
      if (parse_greater_greater()) {
        if (exprContext.templArg && templArgDepth >= 2) {
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

      if (exprContext.templArg || exprContext.templParam) {
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

auto Parser::parse_binary_expression(ExpressionAST*& yyast,
                                     const ExprContext& exprContext) -> bool {
  if (!parse_cast_expression(yyast)) return false;

  const auto saved = currentLocation();

  if (!parse_binary_expression_helper(yyast, Prec::kLogicalOr, exprContext)) {
    rewind(saved);
  }

  return true;
}

auto Parser::parse_lookahead_binary_operator(SourceLocation& loc, TokenKind& tk,
                                             const ExprContext& exprContext)
    -> bool {
  const auto saved = currentLocation();

  const auto has_binop = parse_binary_operator(loc, tk, exprContext);

  rewind(saved);

  return has_binop;
}

auto Parser::parse_binary_expression_helper(ExpressionAST*& yyast, Prec minPrec,
                                            const ExprContext& exprContext)
    -> bool {
  bool parsed = false;

  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  while (parse_lookahead_binary_operator(opLoc, op, exprContext) &&
         prec(op) >= minPrec) {
    const auto saved = currentLocation();

    ExpressionAST* rhs = nullptr;

    parse_binary_operator(opLoc, op, exprContext);

    if (!parse_cast_expression(rhs)) {
      rewind(saved);
      break;
    }

    parsed = true;

    SourceLocation nextOpLoc;
    TokenKind nextOp = TokenKind::T_EOF_SYMBOL;

    while (parse_lookahead_binary_operator(nextOpLoc, nextOp, exprContext) &&
           prec(nextOp) > prec(op)) {
      if (!parse_binary_expression_helper(
              rhs, static_cast<Prec>(static_cast<int>(prec(op)) + 1),
              exprContext)) {
        break;
      }
    }

    auto ast = new (pool) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = rhs;
    ast->op = op;

    yyast = ast;
  }

  return parsed;
}

auto Parser::parse_logical_or_expression(ExpressionAST*& yyast,
                                         const ExprContext& exprContext)
    -> bool {
  return parse_binary_expression(yyast, exprContext);
}

auto Parser::parse_conditional_expression(ExpressionAST*& yyast,
                                          const ExprContext& exprContext)
    -> bool {
  if (!parse_logical_or_expression(yyast, exprContext)) return false;

  SourceLocation questionLoc;

  if (match(TokenKind::T_QUESTION, questionLoc)) {
    auto ast = new (pool) ConditionalExpressionAST();
    ast->condition = yyast;
    ast->questionLoc = questionLoc;

    yyast = ast;

    if (!parse_expression(ast->iftrueExpression)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_COLON, ast->colonLoc);

    if (exprContext.templArg || exprContext.templParam) {
      if (!parse_conditional_expression(ast->iffalseExpression, exprContext)) {
        parse_error("expected an expression");
      }
    } else if (!parse_assignment_expression(ast->iffalseExpression)) {
      parse_error("expected an expression");
    }
  }

  return true;
}

auto Parser::parse_yield_expression(ExpressionAST*& yyast) -> bool {
  if (!match(TokenKind::T_CO_YIELD)) return false;

  if (LA().is(TokenKind::T_LBRACE)) {
    BracedInitListAST* bracedInitList = nullptr;

    if (!parse_braced_init_list(bracedInitList)) {
      parse_error("expected a braced initializer");
    }
  } else {
    ExpressionAST* expression = nullptr;

    if (!parse_assignment_expression(expression)) {
      parse_error("expected an expression");
    }
  }

  return true;
}

auto Parser::parse_throw_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation throwLoc;

  if (!match(TokenKind::T_THROW, throwLoc)) return false;

  auto ast = new (pool) ThrowExpressionAST();
  yyast = ast;

  ast->throwLoc = throwLoc;

  const auto saved = currentLocation();

  if (!parse_assignment_expression(ast->expression)) rewind(saved);

  return true;
}

auto Parser::parse_assignment_expression(ExpressionAST*& yyast) -> bool {
  ExprContext context;
  return parse_assignment_expression(yyast, context);
}

auto Parser::parse_assignment_expression(ExpressionAST*& yyast,
                                         const ExprContext& exprContext)
    -> bool {
  if (parse_yield_expression(yyast)) return true;

  if (parse_throw_expression(yyast)) return true;

  if (!parse_conditional_expression(yyast, exprContext)) return false;

  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  if (parse_assignment_operator(opLoc, op)) {
    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression)) {
      parse_error("expected an expression");
    }

    auto ast = new (pool) AssignmentExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = expression;
    ast->op = op;

    yyast = ast;
  }

  return true;
}

auto Parser::parse_assignment_operator(SourceLocation& loc, TokenKind& op)
    -> bool {
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

auto Parser::parse_expression(ExpressionAST*& yyast) -> bool {
  if (!parse_assignment_expression(yyast)) return false;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    ExpressionAST* expression = nullptr;

    if (!parse_assignment_expression(expression)) {
      parse_error("expected an expression");
    } else {
      auto ast = new (pool) BinaryExpressionAST();
      ast->leftExpression = yyast;
      ast->opLoc = commaLoc;
      ast->op = TokenKind::T_COMMA;
      ast->rightExpression = expression;
      yyast = ast;
    }
  }

  return true;
}

auto Parser::parse_constant_expression(ExpressionAST*& yyast) -> bool {
  ExprContext exprContext;
  return parse_conditional_expression(yyast, exprContext);
}

auto Parser::parse_template_argument_constant_expression(ExpressionAST*& yyast)
    -> bool {
  ExprContext exprContext;
  exprContext.templArg = true;
  return parse_conditional_expression(yyast, exprContext);
}

auto Parser::parse_statement(StatementAST*& yyast) -> bool {
  match(TokenKind::T___EXTENSION__);

  List<AttributeSpecifierAST*>* attributes = nullptr;
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
    case TokenKind::T_LBRACE: {
      CompoundStatementAST* statement = nullptr;
      if (parse_compound_statement(statement)) {
        yyast = statement;
        return true;
      }
      return false;
    }
    default:
      if (LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_COLON)) {
        return parse_labeled_statement(yyast);
      }

      if (parse_declaration_statement(yyast)) return true;

      rewind(start);

      return parse_expression_statement(yyast);
  }  // switch
}

auto Parser::parse_init_statement(StatementAST*& yyast) -> bool {
  if (LA().is(TokenKind::T_RPAREN)) return false;

  auto saved = currentLocation();

  DeclarationAST* declaration = nullptr;

  if (parse_simple_declaration(declaration, false)) {
    auto ast = new (pool) DeclarationStatementAST();
    yyast = ast;

    ast->declaration = declaration;

    return true;
  }

  rewind(saved);

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) return false;

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool) ExpressionStatementAST();
  yyast = ast;

  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_condition(ExpressionAST*& yyast) -> bool {
  const auto start = currentLocation();

  List<AttributeSpecifierAST*>* attributes = nullptr;

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

  if (!parse_expression(yyast)) return false;

  return true;
}

auto Parser::parse_labeled_statement(StatementAST*& yyast) -> bool {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  StatementAST* statement = nullptr;

  if (!parse_statement(statement)) parse_error("expected a statement");

  auto ast = new (pool) LabeledStatementAST();
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);
  ast->colonLoc = colonLoc;
  ast->statement = statement;

  return true;
}

auto Parser::parse_case_statement(StatementAST*& yyast) -> bool {
  SourceLocation caseLoc;

  if (!match(TokenKind::T_CASE, caseLoc)) return false;

  ExpressionAST* expression = nullptr;

  if (!parse_constant_expression(expression)) {
    parse_error("expected an expression");
  }

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

auto Parser::parse_default_statement(StatementAST*& yyast) -> bool {
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

auto Parser::parse_expression_statement(StatementAST*& yyast) -> bool {
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

auto Parser::parse_compound_statement(CompoundStatementAST*& yyast, bool skip)
    -> bool {
  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  auto ast = new (pool) CompoundStatementAST();
  yyast = ast;

  ast->lbraceLoc = lbraceLoc;

  if (skip) {
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

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);

    return true;
  }

  finish_compound_statement(ast);

  if (!expect(TokenKind::T_RBRACE, ast->rbraceLoc)) return false;

  return true;
}

void Parser::finish_compound_statement(CompoundStatementAST* ast) {
  bool skipping = false;

  auto it = &ast->statementList;

  while (LA()) {
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
}

auto Parser::parse_skip_statement(bool& skipping) -> bool {
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

auto Parser::parse_if_statement(StatementAST*& yyast) -> bool {
  SourceLocation ifLoc;

  if (!match(TokenKind::T_IF, ifLoc)) return false;

  auto ast = new (pool) IfStatementAST();
  yyast = ast;

  ast->ifLoc = ifLoc;

  match(TokenKind::T_CONSTEXPR, ast->constexprLoc);

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

auto Parser::parse_switch_statement(StatementAST*& yyast) -> bool {
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

auto Parser::parse_while_statement(StatementAST*& yyast) -> bool {
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

auto Parser::parse_do_statement(StatementAST*& yyast) -> bool {
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

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_for_range_statement(StatementAST*& yyast) -> bool {
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

  if (!parse_for_range_initializer(ast->rangeInitializer)) {
    parse_error("expected for-range intializer");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  if (!parse_statement(ast->statement)) parse_error("expected a statement");

  return true;
}

auto Parser::parse_for_statement(StatementAST*& yyast) -> bool {
  SourceLocation forLoc;

  if (!match(TokenKind::T_FOR, forLoc)) return false;

  auto ast = new (pool) ForStatementAST();
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_init_statement(ast->initializer)) {
    parse_error("expected a statement");
  }

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    if (!parse_condition(ast->condition)) parse_error("expected a condition");

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression(ast->expression)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  if (!parse_statement(ast->statement)) parse_error("expected a statement");

  return true;
}

auto Parser::parse_for_range_declaration(DeclarationAST*& yyast) -> bool {
  List<AttributeSpecifierAST*>* attributeList = nullptr;

  parse_attribute_specifier_seq(attributeList);

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

    auto initDeclarator = new (pool) InitDeclaratorAST();
    initDeclarator->declarator = declarator;

    auto ast = new (pool) SimpleDeclarationAST();
    yyast = ast;

    ast->attributeList = attributeList;
    ast->declSpecifierList = declSpecifierList;
    ast->initDeclaratorList = new (pool) List(initDeclarator);
  }

  return true;
}

auto Parser::parse_for_range_initializer(ExpressionAST*& yyast) -> bool {
  return parse_expr_or_braced_init_list(yyast);
}

auto Parser::parse_break_statement(StatementAST*& yyast) -> bool {
  SourceLocation breakLoc;

  if (!match(TokenKind::T_BREAK, breakLoc)) return false;

  auto ast = new (pool) BreakStatementAST();
  yyast = ast;

  ast->breakLoc = breakLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_continue_statement(StatementAST*& yyast) -> bool {
  SourceLocation continueLoc;

  if (!match(TokenKind::T_CONTINUE, continueLoc)) return false;

  auto ast = new (pool) ContinueStatementAST();
  yyast = ast;

  ast->continueLoc = continueLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_return_statement(StatementAST*& yyast) -> bool {
  SourceLocation returnLoc;

  if (!match(TokenKind::T_RETURN, returnLoc)) return false;

  auto ast = new (pool) ReturnStatementAST();
  yyast = ast;

  ast->returnLoc = returnLoc;

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    if (!parse_expr_or_braced_init_list(ast->expression)) {
      parse_error("expected an expression or ';'");
    }

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  return true;
}

auto Parser::parse_goto_statement(StatementAST*& yyast) -> bool {
  SourceLocation gotoLoc;

  if (!match(TokenKind::T_GOTO, gotoLoc)) return false;

  auto ast = new (pool) GotoStatementAST();
  yyast = ast;

  ast->gotoLoc = gotoLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_coroutine_return_statement(StatementAST*& yyast) -> bool {
  SourceLocation coreturnLoc;

  if (!match(TokenKind::T_CO_RETURN, coreturnLoc)) return false;

  auto ast = new (pool) CoroutineReturnStatementAST();
  yyast = ast;

  ast->coreturnLoc = coreturnLoc;

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    if (!parse_expr_or_braced_init_list(ast->expression)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  return true;
}

auto Parser::parse_declaration_statement(StatementAST*& yyast) -> bool {
  DeclarationAST* declaration = nullptr;

  if (!parse_block_declaration(declaration, false)) return false;

  auto ast = new (pool) DeclarationStatementAST();
  yyast = ast;

  ast->declaration = declaration;

  return true;
}

auto Parser::parse_maybe_module() -> bool {
  if (!module_unit) return false;

  const auto start = currentLocation();

  match(TokenKind::T_EXPORT);

  SourceLocation moduleLoc;

  const auto is_module = parse_module_keyword(moduleLoc);

  rewind(start);

  return is_module;
}

auto Parser::parse_declaration(DeclarationAST*& yyast) -> bool {
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

auto Parser::parse_block_declaration(DeclarationAST*& yyast, bool fundef)
    -> bool {
  const auto start = currentLocation();

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

auto Parser::parse_alias_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

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
  ast->identifier = unit->identifier(ast->identifierLoc);
  ast->attributeList = attributes;
  ast->equalLoc = equalLoc;
  ast->typeId = typeId;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

void Parser::enterFunctionScope(FunctionSymbol* functionSymbol,
                                FunctionDeclaratorAST* functionDeclarator) {}

auto Parser::parse_simple_declaration(DeclarationAST*& yyast,
                                      bool acceptFunctionDefinition) -> bool {
  match(TokenKind::T___EXTENSION__);

  List<AttributeSpecifierAST*>* attributes = nullptr;

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

  if (!parse_decl_specifier_seq_no_typespecs(declSpecifierList, specs)) {
    rewind(after_attributes);
  }

  auto after_decl_specs = currentLocation();

  if (acceptFunctionDefinition &&
      parse_notypespec_function_definition(yyast, declSpecifierList, specs)) {
    return true;
  }

  rewind(after_decl_specs);

  auto lastDeclSpecifier = &declSpecifierList;

  while (*lastDeclSpecifier) {
    lastDeclSpecifier = &(*lastDeclSpecifier)->next;
  }

  if (!parse_decl_specifier_seq(*lastDeclSpecifier, specs)) {
    rewind(after_decl_specs);
  }

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

      if (parse_initializer(initializer) && match(TokenKind::T_SEMICOLON)) {
        return true;
      }
    }
  }

  rewind(after_decl_specs);

  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  const auto after_declarator = currentLocation();

  auto functionDeclarator = getFunctionDeclarator(declarator);

  RequiresClauseAST* requiresClause = nullptr;

  parse_requires_clause(requiresClause);

  if (acceptFunctionDefinition && functionDeclarator &&
      lookat_function_body()) {
    FunctionBodyAST* functionBody = nullptr;

    if (!parse_function_body(functionBody)) {
      parse_error("expected function body");
    }

    auto ast = new (pool) FunctionDefinitionAST();
    yyast = ast;

    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->declarator = declarator;
    ast->requiresClause = requiresClause;
    ast->functionBody = functionBody;

    if (classDepth) pendingFunctionDefinitions_.push_back(ast);

    return true;
  }

  rewind(after_declarator);

  auto* initDeclarator = new (pool) InitDeclaratorAST();

  initDeclarator->declarator = declarator;

  requiresClause = nullptr;

  if (!parse_declarator_initializer(initDeclarator->requiresClause,
                                    initDeclarator->initializer)) {
    rewind(after_declarator);
  }

  List<InitDeclaratorAST*>* initDeclaratorList = nullptr;

  auto declIt = &initDeclaratorList;

  *declIt = new (pool) List(initDeclarator);
  declIt = &(*declIt)->next;

  while (match(TokenKind::T_COMMA)) {
    InitDeclaratorAST* initDeclarator = nullptr;

    if (!parse_init_declarator(initDeclarator, specs)) return false;

    *declIt = new (pool) List(initDeclarator);
    declIt = &(*declIt)->next;
  }

  if (!match(TokenKind::T_SEMICOLON)) return false;

  auto ast = new (pool) SimpleDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->declSpecifierList = declSpecifierList;
  ast->initDeclaratorList = initDeclaratorList;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_notypespec_function_definition(
    DeclarationAST*& yyast, List<SpecifierAST*>* declSpecifierList,
    const DeclSpecs& specs) -> bool {
  IdDeclaratorAST* declaratorId = nullptr;

  if (!parse_declarator_id(declaratorId)) return false;

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  if (!parse_parameters_and_qualifiers(parametersAndQualifiers)) return false;

  auto declarator = new (pool) DeclaratorAST();
  declarator->coreDeclarator = declaratorId;

  auto functionDeclarator = new (pool) FunctionDeclaratorAST();
  functionDeclarator->parametersAndQualifiers = parametersAndQualifiers;

  declarator->modifiers =
      new (pool) List<DeclaratorModifierAST*>(functionDeclarator);

  RequiresClauseAST* requiresClause = nullptr;

  const auto has_requires_clause = parse_requires_clause(requiresClause);

  if (!has_requires_clause) parse_virt_specifier_seq();

  SourceLocation semicolonLoc;

  if (match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    auto initDeclarator = new (pool) InitDeclaratorAST();
    initDeclarator->declarator = declarator;

    auto ast = new (pool) SimpleDeclarationAST();
    yyast = ast;
    ast->declSpecifierList = declSpecifierList;
    ast->initDeclaratorList = new (pool) List(initDeclarator);
    ast->requiresClause = requiresClause;
    ast->semicolonLoc = semicolonLoc;

    return true;
  }

  if (!lookat_function_body()) return false;

  FunctionBodyAST* functionBody = nullptr;

  if (!parse_function_body(functionBody)) parse_error("expected function body");

  auto ast = new (pool) FunctionDefinitionAST();
  yyast = ast;

  ast->declSpecifierList = declSpecifierList;
  ast->declarator = declarator;
  ast->functionBody = functionBody;

  if (classDepth) pendingFunctionDefinitions_.push_back(ast);

  return true;
}

auto Parser::parse_static_assert_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation staticAssertLoc;

  if (!match(TokenKind::T_STATIC_ASSERT, staticAssertLoc)) return false;

  auto ast = new (pool) StaticAssertDeclarationAST();
  yyast = ast;

  ast->staticAssertLoc = staticAssertLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_constant_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  if (match(TokenKind::T_COMMA, ast->commaLoc)) {
    if (!parse_string_literal_seq(ast->stringLiteralList)) {
      parse_error("expected a string literal");
    }
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::match_string_literal(SourceLocation& loc) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_WIDE_STRING_LITERAL:
    case TokenKind::T_UTF8_STRING_LITERAL:
    case TokenKind::T_UTF16_STRING_LITERAL:
    case TokenKind::T_UTF32_STRING_LITERAL:
    case TokenKind::T_STRING_LITERAL:
      loc = consumeToken();
      return true;
    default:
      return false;
  }  // switch
}

auto Parser::parse_string_literal_seq(List<SourceLocation>*& yyast) -> bool {
  auto it = &yyast;

  SourceLocation loc;

  if (!match_string_literal(loc)) return false;

  *it = new (pool) List(loc);
  it = &(*it)->next;

  while (match_string_literal(loc)) {
    *it = new (pool) List(loc);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_empty_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool) EmptyDeclarationAST();
  yyast = ast;

  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_attribute_declaration(DeclarationAST*& yyast) -> bool {
  List<AttributeSpecifierAST*>* attributes = nullptr;

  if (!parse_attribute_specifier_seq(attributes)) return false;

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool) AttributeDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_decl_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
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

    case TokenKind::T_INLINE: {
      auto ast = new (pool) InlineSpecifierAST();
      yyast = ast;
      ast->inlineLoc = consumeToken();
      return true;
    }

    default:
      if (parse_storage_class_specifier(yyast)) return true;

      if (parse_function_specifier(yyast)) return true;

      if (!specs.no_typespecs) {
        return parse_defining_type_specifier(yyast, specs);
      }

      return false;
  }  // switch
}

auto Parser::parse_decl_specifier_seq(List<SpecifierAST*>*& yyast,
                                      DeclSpecs& specs) -> bool {
  auto it = &yyast;

  specs.no_typespecs = false;

  SpecifierAST* specifier = nullptr;

  if (!parse_decl_specifier(specifier, specs)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    *it = new (pool) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

auto Parser::parse_decl_specifier_seq_no_typespecs(List<SpecifierAST*>*& yyast,
                                                   DeclSpecs& specs) -> bool {
  auto it = &yyast;

  specs.no_typespecs = true;

  SpecifierAST* specifier = nullptr;

  if (!parse_decl_specifier(specifier, specs)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    *it = new (pool) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

auto Parser::parse_storage_class_specifier(SpecifierAST*& yyast) -> bool {
  SourceLocation loc;

  if (match(TokenKind::T_STATIC, loc)) {
    auto ast = new (pool) StaticSpecifierAST();
    yyast = ast;
    ast->staticLoc = loc;
    return true;
  }
  if (match(TokenKind::T_THREAD_LOCAL, loc)) {
    auto ast = new (pool) ThreadLocalSpecifierAST();
    yyast = ast;
    ast->threadLocalLoc = loc;
    return true;
  }
  if (match(TokenKind::T_EXTERN, loc)) {
    auto ast = new (pool) ExternSpecifierAST();
    yyast = ast;
    ast->externLoc = loc;
    return true;
  }
  if (match(TokenKind::T_MUTABLE, loc)) {
    auto ast = new (pool) MutableSpecifierAST();
    yyast = ast;
    ast->mutableLoc = loc;
    return true;
  }
  if (match(TokenKind::T___THREAD, loc)) {
    auto ast = new (pool) ThreadSpecifierAST();
    yyast = ast;
    ast->threadLoc = loc;
    return true;
  }

  return false;
}

auto Parser::parse_function_specifier(SpecifierAST*& yyast) -> bool {
  SourceLocation virtualLoc;

  if (match(TokenKind::T_VIRTUAL, virtualLoc)) {
    auto ast = new (pool) VirtualSpecifierAST();
    yyast = ast;
    ast->virtualLoc = virtualLoc;
    return true;
  }

  return parse_explicit_specifier(yyast);
}

auto Parser::parse_explicit_specifier(SpecifierAST*& yyast) -> bool {
  SourceLocation explicitLoc;

  if (!match(TokenKind::T_EXPLICIT, explicitLoc)) return false;

  auto ast = new (pool) ExplicitSpecifierAST();
  yyast = ast;

  ast->explicitLoc = explicitLoc;

  if (match(TokenKind::T_LPAREN, ast->lparenLoc)) {
    if (!parse_constant_expression(ast->expression)) {
      parse_error("expected a expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  return true;
}

auto Parser::parse_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (parse_simple_type_specifier(yyast, specs)) return true;

  if (parse_elaborated_type_specifier(yyast, specs)) return true;

  if (parse_cv_qualifier(yyast)) return true;

  if (parse_typename_specifier(yyast)) {
    specs.has_named_typespec = true;
    return true;
  }

  return false;
}

auto Parser::parse_type_specifier_seq(List<SpecifierAST*>*& yyast) -> bool {
  auto it = &yyast;

  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  SpecifierAST* typeSpecifier = nullptr;

  if (!parse_type_specifier(typeSpecifier, specs)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

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

    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    *it = new (pool) List(typeSpecifier);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_defining_type_specifier(SpecifierAST*& yyast,
                                           DeclSpecs& specs) -> bool {
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

auto Parser::parse_defining_type_specifier_seq(List<SpecifierAST*>*& yyast,
                                               DeclSpecs& specs) -> bool {
  auto it = &yyast;

  SpecifierAST* typeSpecifier = nullptr;

  if (!parse_defining_type_specifier(typeSpecifier, specs)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  *it = new (pool) List(typeSpecifier);
  it = &(*it)->next;

  while (LA()) {
    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_defining_type_specifier(typeSpecifier, specs)) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    *it = new (pool) List(typeSpecifier);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_simple_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  const auto start = currentLocation();

  if (parse_placeholder_type_specifier_helper(yyast, specs)) return true;

  rewind(start);

  if (parse_named_type_specifier(yyast, specs)) return true;

  rewind(start);

  if (parse_primitive_type_specifier(yyast, specs)) return true;

  if (parse_underlying_type_specifier(yyast, specs)) return true;

  if (parse_atomic_type_specifier(yyast, specs)) return true;

  return parse_decltype_specifier_type_specifier(yyast, specs);
}

auto Parser::parse_named_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (!parse_named_type_specifier_helper(yyast, specs)) return false;

  specs.has_named_typespec = true;

  return true;
}

auto Parser::parse_named_type_specifier_helper(SpecifierAST*& yyast,
                                               DeclSpecs& specs) -> bool {
  if (specs.has_typespec()) return false;

  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (parse_nested_name_specifier(nestedNameSpecifier)) {
    SourceLocation templateLoc;
    NameAST* name = nullptr;

    match(TokenKind::T_TEMPLATE, templateLoc);

    if (parse_type_name(name)) {
      auto qualifiedId = new (pool) QualifiedNameAST();
      qualifiedId->nestedNameSpecifier = nestedNameSpecifier;
      qualifiedId->templateLoc = templateLoc;
      qualifiedId->id = name;

      auto ast = new (pool) NamedTypeSpecifierAST();
      yyast = ast;

      ast->name = qualifiedId;

      return true;
    }
  }

  rewind(start);

  NameAST* name = nullptr;

  if (!parse_type_name(name)) return false;

  auto ast = new (pool) NamedTypeSpecifierAST();
  yyast = ast;

  ast->name = name;

  return true;
}

auto Parser::parse_placeholder_type_specifier_helper(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) -> bool {
  if (specs.has_typespec()) return false;

  if (!parse_placeholder_type_specifier(yyast)) return false;

  specs.has_placeholder_typespec = true;

  return true;
}

auto Parser::parse_decltype_specifier_type_specifier(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) -> bool {
  if (specs.has_typespec()) return false;

  if (!parse_decltype_specifier(yyast)) return false;

  specs.has_placeholder_typespec = true;

  return true;
}

auto Parser::parse_underlying_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  if (specs.has_typespec()) return false;

  if (!match(TokenKind::T___UNDERLYING_TYPE)) return false;

  expect(TokenKind::T_LPAREN);

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) parse_error("expected type id");

  expect(TokenKind::T_RPAREN);

  specs.has_named_typespec = true;

  return true;
}

auto Parser::parse_atomic_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
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

auto Parser::parse_primitive_type_specifier(SpecifierAST*& yyast,
                                            DeclSpecs& specs) -> bool {
  if (!specs.accepts_simple_typespec()) return false;

  switch (auto tk = LA(); tk.kind()) {
    case TokenKind::T___BUILTIN_VA_LIST: {
      auto ast = new (pool) VaListTypeSpecifierAST();
      yyast = ast;
      ast->specifierLoc = consumeToken();
      ast->specifier = unit->tokenKind(ast->specifierLoc);
      specs.has_simple_typespec = true;
      return true;
    };

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
      ast->specifier = unit->tokenKind(ast->specifierLoc);
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
      ast->specifier = unit->tokenKind(ast->specifierLoc);
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

    case TokenKind::T__COMPLEX:
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

auto Parser::parse_elaborated_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_ENUM:
    case TokenKind::T_CLASS:
    case TokenKind::T_STRUCT:
    case TokenKind::T_UNION:
      break;
    default:
      return false;
  }  // switch

  const auto start = currentLocation();

  auto it = elaborated_type_specifiers_.find(start);

  if (it != elaborated_type_specifiers_.end()) {
    auto [cursor, ast, parsed] = it->second;
    rewind(cursor);
    yyast = ast;
    if (parsed) specs.has_complex_typespec = true;
    return parsed;
  }

  ElaboratedTypeSpecifierAST* ast = nullptr;

  const auto parsed = parse_elaborated_type_specifier_helper(ast, specs);

  yyast = ast;

  elaborated_type_specifiers_.emplace(
      start, std::tuple(currentLocation(), ast, parsed));

  return parsed;
}

void Parser::check_type_traits() {
  SourceLocation typeTraitsLoc;

  if (!parse_type_traits_op(typeTraitsLoc)) return;

#if 0
  parse_warn(
      typeTraitsLoc,
      fmt::format("keyword '{}' will be made available as an identifier for "
                  "the remainder of the translation unit",
                  Token::spell(unit->tokenKind(typeTraitsLoc))));
#endif

  unit->replaceWithIdentifier(typeTraitsLoc);

  rewind(typeTraitsLoc);
}

auto Parser::parse_elaborated_type_specifier_helper(
    ElaboratedTypeSpecifierAST*& yyast, DeclSpecs& specs) -> bool {
  // ### cleanup

  if (LA().is(TokenKind::T_ENUM)) {
    return parse_elaborated_enum_specifier(yyast, specs);
  }

  SourceLocation classLoc;

  if (!parse_class_key(classLoc)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  const auto before_nested_name_specifier = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) {
    rewind(before_nested_name_specifier);

    NameAST* name = nullptr;

    if (parse_simple_template_id(name)) {
      specs.has_complex_typespec = true;

      auto ast = new (pool) ElaboratedTypeSpecifierAST();
      yyast = ast;

      ast->classLoc = classLoc;
      ast->attributeList = attributes;
      ast->name = name;

      return true;
    }

    rewind(before_nested_name_specifier);

    check_type_traits();

    SourceLocation identifierLoc;

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    auto id = new (pool) SimpleNameAST();
    name = id;

    id->identifierLoc = identifierLoc;
    id->identifier = unit->identifier(id->identifierLoc);

    specs.has_complex_typespec = true;

    auto ast = new (pool) ElaboratedTypeSpecifierAST();
    yyast = ast;

    ast->classLoc = classLoc;
    ast->attributeList = attributes;
    ast->name = name;

    return true;
  }

  const auto after_nested_name_specifier = currentLocation();

  const bool has_template = match(TokenKind::T_TEMPLATE);

  NameAST* name = nullptr;

  if (parse_simple_template_id(name)) {
    specs.has_complex_typespec = true;

    auto ast = new (pool) ElaboratedTypeSpecifierAST();
    yyast = ast;

    ast->classLoc = classLoc;
    ast->attributeList = attributes;
    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->name = name;

    return true;
  }

  if (has_template) {
    parse_error("expected a template-id");
    specs.has_complex_typespec = true;

    auto ast = new (pool) ElaboratedTypeSpecifierAST();
    yyast = ast;

    ast->classLoc = classLoc;
    ast->attributeList = attributes;
    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->name = nullptr;  // error

    return true;
  }

  rewind(after_nested_name_specifier);

  if (!parse_name_id(name)) return false;

  specs.has_complex_typespec = true;

  auto ast = new (pool) ElaboratedTypeSpecifierAST();
  yyast = ast;

  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;

  return true;
}

auto Parser::parse_elaborated_enum_specifier(ElaboratedTypeSpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  SourceLocation enumLoc;

  if (!match(TokenKind::T_ENUM, enumLoc)) return false;

  const auto saved = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(saved);

  NameAST* name = nullptr;

  if (!parse_name_id(name)) return false;

  specs.has_complex_typespec = true;

  auto ast = new (pool) ElaboratedTypeSpecifierAST();
  yyast = ast;

  ast->classLoc = enumLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;

  return true;
}

auto Parser::parse_decl_specifier_seq_no_typespecs(List<SpecifierAST*>*& yyast)
    -> bool {
  DeclSpecs specs;
  return parse_decl_specifier_seq_no_typespecs(yyast, specs);
}

auto Parser::parse_decltype_specifier(SpecifierAST*& yyast) -> bool {
  SourceLocation decltypeLoc;

  if (match(TokenKind::T_DECLTYPE, decltypeLoc)) {
    SourceLocation lparenLoc;

    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    if (LA().is(TokenKind::T_AUTO)) return false;  // placeholder type specifier

    auto ast = new (pool) DecltypeSpecifierAST();
    yyast = ast;

    ast->decltypeLoc = decltypeLoc;
    ast->lparenLoc = lparenLoc;

    if (!parse_expression(ast->expression)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  return false;
}

auto Parser::parse_placeholder_type_specifier(SpecifierAST*& yyast) -> bool {
  TypeConstraintAST* typeConstraint = nullptr;

  parse_type_constraint(typeConstraint, /*parsing placeholder=*/true);

  SourceLocation autoLoc;

  if (match(TokenKind::T_AUTO, autoLoc)) {
    auto ast = new (pool) AutoTypeSpecifierAST();
    yyast = ast;
    ast->autoLoc = autoLoc;

    if (typeConstraint) {
      auto ast = new (pool) PlaceholderTypeSpecifierAST();
      ast->typeConstraint = typeConstraint;
      ast->specifier = yyast;
      yyast = ast;
    }

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

    if (typeConstraint) {
      auto ast = new (pool) PlaceholderTypeSpecifierAST();
      ast->typeConstraint = typeConstraint;
      ast->specifier = yyast;
      yyast = ast;
    }

    return true;
  }

  return false;
}

auto Parser::parse_init_declarator(InitDeclaratorAST*& yyast,
                                   const DeclSpecs& specs) -> bool {
  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  const auto saved = currentLocation();

  RequiresClauseAST* requiresClause = nullptr;
  InitializerAST* initializer = nullptr;

  if (!parse_declarator_initializer(requiresClause, initializer)) rewind(saved);

  auto ast = new (pool) InitDeclaratorAST();
  yyast = ast;

  ast->declarator = declarator;
  ast->requiresClause = requiresClause;
  ast->initializer = initializer;

  return true;
}

auto Parser::parse_declarator_initializer(RequiresClauseAST*& requiresClause,
                                          InitializerAST*& yyast) -> bool {
  if (parse_requires_clause(requiresClause)) return true;

  return parse_initializer(yyast);
}

auto Parser::parse_declarator(DeclaratorAST*& yyast) -> bool {
  const auto start = currentLocation();

  List<PtrOperatorAST*>* ptrOpList = nullptr;

  if (!parse_ptr_operator_seq(ptrOpList)) {
    rewind(start);

    ptrOpList = nullptr;
  }

  if (!parse_noptr_declarator(yyast, ptrOpList)) return false;

  return true;
}

auto Parser::parse_ptr_operator_seq(List<PtrOperatorAST*>*& yyast) -> bool {
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

auto Parser::parse_core_declarator(CoreDeclaratorAST*& yyast) -> bool {
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

auto Parser::parse_noptr_declarator(DeclaratorAST*& yyast,
                                    List<PtrOperatorAST*>* ptrOpLst) -> bool {
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

      List<AttributeSpecifierAST*>* attributes = nullptr;

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

auto Parser::parse_parameters_and_qualifiers(ParametersAndQualifiersAST*& yyast)
    -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  SourceLocation rparenLoc;

  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_parameter_declaration_clause(parameterDeclarationClause)) {
      return false;
    }

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

auto Parser::parse_cv_qualifier_seq(List<SpecifierAST*>*& yyast) -> bool {
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

auto Parser::parse_trailing_return_type(TrailingReturnTypeAST*& yyast) -> bool {
  SourceLocation minusGreaterLoc;

  if (!match(TokenKind::T_MINUS_GREATER, minusGreaterLoc)) return false;

  auto ast = new (pool) TrailingReturnTypeAST();
  yyast = ast;

  ast->minusGreaterLoc = minusGreaterLoc;

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  return true;
}

auto Parser::parse_ptr_operator(PtrOperatorAST*& yyast) -> bool {
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
    ast->refOp = unit->tokenKind(refLoc);

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

auto Parser::parse_cv_qualifier(SpecifierAST*& yyast) -> bool {
  SourceLocation loc;

  if (match(TokenKind::T_CONST, loc)) {
    auto ast = new (pool) ConstQualifierAST();
    yyast = ast;
    ast->constLoc = loc;
    return true;
  }
  if (match(TokenKind::T_VOLATILE, loc)) {
    auto ast = new (pool) VolatileQualifierAST();
    yyast = ast;
    ast->volatileLoc = loc;
    return true;
  }
  if (match(TokenKind::T___RESTRICT__, loc)) {
    auto ast = new (pool) RestrictQualifierAST();
    yyast = ast;
    ast->restrictLoc = loc;
    return true;
  }
  return false;
}

auto Parser::parse_ref_qualifier(SourceLocation& refLoc) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_AMP:
    case TokenKind::T_AMP_AMP:
      refLoc = consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

auto Parser::parse_declarator_id(IdDeclaratorAST*& yyast) -> bool {
  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  check_type_traits();

  NameAST* name = nullptr;

  if (!parse_id_expression(name)) return false;

  yyast = new (pool) IdDeclaratorAST();
  yyast->ellipsisLoc = ellipsisLoc;
  yyast->name = name;

  parse_attribute_specifier_seq(yyast->attributeList);

  return true;
}

auto Parser::parse_type_id(TypeIdAST*& yyast) -> bool {
  List<SpecifierAST*>* specifierList = nullptr;

  if (!parse_type_specifier_seq(specifierList)) return false;

  yyast = new (pool) TypeIdAST();

  yyast->typeSpecifierList = specifierList;

  const auto before_declarator = currentLocation();

  if (!parse_abstract_declarator(yyast->declarator)) rewind(before_declarator);

  return true;
}

auto Parser::parse_defining_type_id(TypeIdAST*& yyast) -> bool {
  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_defining_type_specifier_seq(typeSpecifierList, specs)) {
    return false;
  }

  const auto before_declarator = currentLocation();

  DeclaratorAST* declarator = nullptr;

  if (!parse_abstract_declarator(declarator)) rewind(before_declarator);

  auto ast = new (pool) TypeIdAST();
  yyast = ast;

  ast->typeSpecifierList = typeSpecifierList;
  ast->declarator = declarator;

  return true;
}

auto Parser::parse_abstract_declarator(DeclaratorAST*& yyast) -> bool {
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

#if 0
  const auto after_noptr_declarator = currentLocation();

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
#endif

  return true;
}

auto Parser::parse_ptr_abstract_declarator(DeclaratorAST*& yyast) -> bool {
  List<PtrOperatorAST*>* ptrOpList = nullptr;

  if (!parse_ptr_operator_seq(ptrOpList)) return false;

  auto ast = new (pool) DeclaratorAST();
  yyast = ast;

  ast->ptrOpList = ptrOpList;

  const auto saved = currentLocation();

  if (!parse_noptr_abstract_declarator(yyast)) rewind(saved);

  return true;
}

auto Parser::parse_noptr_abstract_declarator(DeclaratorAST*& yyast) -> bool {
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
        if (!parse_constant_expression(arrayDeclarator->expression)) {
          parse_error("expected an expression");
        }

        expect(TokenKind::T_RBRACKET, arrayDeclarator->rbracketLoc);
      }
    }
  }

  return true;
}

auto Parser::parse_abstract_pack_declarator() -> bool {
  auto start = currentLocation();

  List<PtrOperatorAST*>* ptrOpList = nullptr;

  parse_ptr_operator_seq(ptrOpList);

  if (!parse_noptr_abstract_pack_declarator()) {
    rewind(start);
    return false;
  }

  return true;
}

auto Parser::parse_noptr_abstract_pack_declarator() -> bool {
  if (!match(TokenKind::T_DOT_DOT_DOT)) return false;

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  if (parse_parameters_and_qualifiers(parametersAndQualifiers)) return true;

  while (match(TokenKind::T_LBRACKET)) {
    if (!match(TokenKind::T_RBRACKET)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(expression)) {
        parse_error("expected a constant expression");
      }

      expect(TokenKind::T_RBRACKET);

      List<AttributeSpecifierAST*>* attributes = nullptr;

      parse_attribute_specifier_seq(attributes);
    }
  }

  return true;
}

auto Parser::parse_parameter_declaration_clause(
    ParameterDeclarationClauseAST*& yyast) -> bool {
  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool) ParameterDeclarationClauseAST();
    yyast = ast;

    ast->ellipsisLoc = ellipsisLoc;

    return true;
  }

  auto ast = new (pool) ParameterDeclarationClauseAST();
  yyast = ast;

  if (!parse_parameter_declaration_list(ast->parameterDeclarationList)) {
    return false;
  }

  match(TokenKind::T_COMMA, ast->commaLoc);

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  return true;
}

auto Parser::parse_parameter_declaration_list(
    List<ParameterDeclarationAST*>*& yyast) -> bool {
  auto it = &yyast;

  ParameterDeclarationAST* declaration = nullptr;

  if (!parse_parameter_declaration(declaration, /*templParam*/ false)) {
    return false;
  }

  *it = new (pool) List(declaration);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    ParameterDeclarationAST* declaration = nullptr;

    if (!parse_parameter_declaration(declaration, /*templParam*/ false)) {
      rewind(commaLoc);
      break;
    }

    *it = new (pool) List(declaration);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_parameter_declaration(ParameterDeclarationAST*& yyast,
                                         bool templParam) -> bool {
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
    if (!parse_initializer_clause(ast->expression, templParam)) {
      if (templParam) return false;

      parse_error("expected an initializer");
    }
  }

  return true;
}

auto Parser::parse_initializer(InitializerAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (match(TokenKind::T_LPAREN, lparenLoc)) {
    if (LA().is(TokenKind::T_RPAREN)) return false;

    auto ast = new (pool) ParenInitializerAST();
    yyast = ast;

    ast->lparenLoc = lparenLoc;

    if (!parse_expression_list(ast->expressionList)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  return parse_brace_or_equal_initializer(yyast);
}

auto Parser::parse_brace_or_equal_initializer(InitializerAST*& yyast) -> bool {
  BracedInitListAST* bracedInitList = nullptr;

  if (LA().is(TokenKind::T_LBRACE)) {
    if (!parse_braced_init_list(bracedInitList)) return false;
    yyast = bracedInitList;
    return true;
  }

  SourceLocation equalLoc;

  if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

  auto ast = new (pool) EqualInitializerAST();
  yyast = ast;

  ast->equalLoc = equalLoc;

  if (!parse_initializer_clause(ast->expression)) {
    parse_error("expected an intializer");
  }

  return true;
}

auto Parser::parse_initializer_clause(ExpressionAST*& yyast, bool templParam)
    -> bool {
  BracedInitListAST* bracedInitList = nullptr;

  if (LA().is(TokenKind::T_LBRACE)) {
    return parse_braced_init_list(bracedInitList);
  }

  ExprContext exprContext;
  exprContext.templParam = templParam;

  if (!parse_assignment_expression(yyast, exprContext)) return false;

  return true;
}

auto Parser::parse_braced_init_list(BracedInitListAST*& yyast) -> bool {
  SourceLocation lbraceLoc;
  SourceLocation commaLoc;
  SourceLocation rbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  if (LA().is(TokenKind::T_DOT)) {
    if (!parse_designated_initializer_clause()) {
      parse_error("expected designated initializer clause");
    }

    while (match(TokenKind::T_COMMA)) {
      if (LA().is(TokenKind::T_RBRACE)) break;

      if (!parse_designated_initializer_clause()) {
        parse_error("expected designated initializer clause");
      }
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
    if (!parse_initializer_list(expressionList)) {
      parse_error("expected initializer list");
    }

    expect(TokenKind::T_RBRACE, rbraceLoc);
  }

  auto ast = new (pool) BracedInitListAST();
  yyast = ast;

  ast->lbraceLoc = lbraceLoc;
  ast->expressionList = expressionList;
  ast->rbraceLoc = rbraceLoc;

  return true;
}

auto Parser::parse_initializer_list(List<ExpressionAST*>*& yyast) -> bool {
  auto it = &yyast;

  ExpressionAST* expression = nullptr;

  if (!parse_initializer_clause(expression)) return false;

  match(TokenKind::T_DOT_DOT_DOT);

  *it = new (pool) List(expression);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    if (LA().is(TokenKind::T_RBRACE)) break;

    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression)) {
      parse_error("expected initializer clause");
    }

    match(TokenKind::T_DOT_DOT_DOT);

    *it = new (pool) List(expression);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_designated_initializer_clause() -> bool {
  if (!parse_designator()) return false;

  InitializerAST* initializer = nullptr;

  if (!parse_brace_or_equal_initializer(initializer)) {
    parse_error("expected an initializer");
  }

  return true;
}

auto Parser::parse_designator() -> bool {
  if (!match(TokenKind::T_DOT)) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

auto Parser::parse_expr_or_braced_init_list(ExpressionAST*& yyast) -> bool {
  if (LA().is(TokenKind::T_LBRACE)) {
    BracedInitListAST* bracedInitList = nullptr;

    return parse_braced_init_list(bracedInitList);
  }

  if (!parse_expression(yyast)) parse_error("expected an expression");

  return true;
}

auto Parser::parse_virt_specifier_seq() -> bool {
  if (!parse_virt_specifier()) return false;

  while (parse_virt_specifier()) {
    //
  }

  return true;
}

auto Parser::lookat_function_body() -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_TRY:
      // function-try-block
      return true;
    case TokenKind::T_LBRACE:
      // compound statement
      return true;
    case TokenKind::T_COLON:
      // ctor-initializer
      return true;
    case TokenKind::T_EQUAL:
      // default/delete functions
      return LA(1).isNot(TokenKind::T_INTEGER_LITERAL);
    default:
      return false;
  }  // swtich
}

auto Parser::parse_function_body(FunctionBodyAST*& yyast) -> bool {
  if (LA().is(TokenKind::T_SEMICOLON)) return false;

  if (parse_function_try_block(yyast)) return true;

  SourceLocation equalLoc;

  if (match(TokenKind::T_EQUAL, equalLoc)) {
    SourceLocation defaultLoc;

    if (match(TokenKind::T_DEFAULT, defaultLoc)) {
      auto ast = new (pool) DefaultFunctionBodyAST();
      yyast = ast;

      ast->equalLoc = equalLoc;
      ast->defaultLoc = defaultLoc;

      expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

      return true;
    }

    SourceLocation deleteLoc;

    if (match(TokenKind::T_DELETE, deleteLoc)) {
      auto ast = new (pool) DeleteFunctionBodyAST();
      yyast = ast;

      ast->equalLoc = equalLoc;
      ast->deleteLoc = deleteLoc;

      expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

      return true;
    }

    return false;
  }

  CtorInitializerAST* ctorInitializer = nullptr;

  parse_ctor_initializer(ctorInitializer);

  if (LA().isNot(TokenKind::T_LBRACE)) return false;

  auto ast = new (pool) CompoundStatementFunctionBodyAST();
  yyast = ast;

  ast->ctorInitializer = ctorInitializer;

  const bool skip = skipFunctionBody_ || classDepth > 0;

  if (!parse_compound_statement(ast->statement, skip)) {
    parse_error("expected a compound statement");
  }

  return true;
}

auto Parser::parse_enum_specifier(SpecifierAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation enumLoc;
  SourceLocation classLoc;

  if (!parse_enum_key(enumLoc, classLoc)) {
    rewind(start);
    return false;
  }

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  parse_enum_head_name(nestedNameSpecifier, name);

  EnumBaseAST* enumBase = nullptr;

  parse_enum_base(enumBase);

  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  const Name* enumName = name ? name->name : nullptr;
  Scope* enumScope = nullptr;

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

auto Parser::parse_enum_head_name(NestedNameSpecifierAST*& nestedNameSpecifier,
                                  NameAST*& name) -> bool {
  const auto start = currentLocation();

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(start);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = new (pool) SimpleNameAST();
  id->identifierLoc = identifierLoc;
  id->identifier = unit->identifier(id->identifierLoc);

  name = id;

  return true;
}

auto Parser::parse_opaque_enum_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation enumLoc;
  SourceLocation classLoc;

  if (!parse_enum_key(enumLoc, classLoc)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

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

auto Parser::parse_enum_key(SourceLocation& enumLoc, SourceLocation& classLoc)
    -> bool {
  if (!match(TokenKind::T_ENUM, enumLoc)) return false;

  if (match(TokenKind::T_CLASS, classLoc)) {
    //
  } else if (match(TokenKind::T_STRUCT, classLoc)) {
    //
  }

  return true;
}

auto Parser::parse_enum_base(EnumBaseAST*& yyast) -> bool {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList)) {
    parse_error("expected a type specifier");
  }

  auto ast = new (pool) EnumBaseAST();
  yyast = ast;

  ast->colonLoc = colonLoc;
  ast->typeSpecifierList = typeSpecifierList;

  return true;
}

auto Parser::parse_enumerator_list(List<EnumeratorAST*>*& yyast) -> bool {
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

    if (!parse_enumerator_definition(enumerator)) {
      parse_error("expected an enumerator");
    }

    *it = new (pool) List(enumerator);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_enumerator_definition(EnumeratorAST*& yyast) -> bool {
  if (!parse_enumerator(yyast)) return false;

  if (!match(TokenKind::T_EQUAL, yyast->equalLoc)) return true;

  if (!parse_constant_expression(yyast->expression)) {
    parse_error("expected an expression");
  }

  return true;
}

auto Parser::parse_enumerator(EnumeratorAST*& yyast) -> bool {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto name = new (pool) SimpleNameAST();
  name->identifierLoc = identifierLoc;
  name->identifier = unit->identifier(name->identifierLoc);

  auto ast = new (pool) EnumeratorAST();
  yyast = ast;

  ast->name = name;

  parse_attribute_specifier_seq(ast->attributeList);

  return true;
}

auto Parser::parse_using_enum_declaration(DeclarationAST*& yyasts) -> bool {
  if (!match(TokenKind::T_USING)) return false;

  ElaboratedTypeSpecifierAST* enumSpecifier = nullptr;

  DeclSpecs specs;

  if (!parse_elaborated_enum_specifier(enumSpecifier, specs)) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

auto Parser::parse_namespace_definition(DeclarationAST*& yyast) -> bool {
  if (LA().is(TokenKind::T_NAMESPACE) && LA(1).is(TokenKind::T_IDENTIFIER) &&
      LA(2).is(TokenKind::T_EQUAL)) {
    // skip namespace alias definitons
    return false;
  }

  const auto start = currentLocation();

  SourceLocation inlineLoc;

  match(TokenKind::T_INLINE, inlineLoc);

  SourceLocation namespaceLoc;

  if (!match(TokenKind::T_NAMESPACE, namespaceLoc)) {
    rewind(start);
    return false;
  }

  NamespaceSymbol* namespaceSymbol = nullptr;

  auto ast = new (pool) NamespaceDefinitionAST();
  yyast = ast;

  ast->inlineLoc = inlineLoc;
  ast->namespaceLoc = namespaceLoc;

  parse_attribute_specifier_seq(ast->attributeList);

  const Name* namespaceName = nullptr;

  if (LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_COLON_COLON)) {
    SourceLocation identifierLoc = consumeToken();

    auto id = unit->identifier(identifierLoc);

    while (match(TokenKind::T_COLON_COLON)) {
      SourceLocation inlineLoc;
      match(TokenKind::T_INLINE, inlineLoc);

      SourceLocation identifierLoc;
      expect(TokenKind::T_IDENTIFIER, identifierLoc);

      auto id = unit->identifier(identifierLoc);
    }
  } else if (parse_name_id(ast->name)) {
    //
  }

  parse_attribute_specifier_seq(ast->extraAttributeList);

  expect(TokenKind::T_LBRACE, ast->lbraceLoc);

  parse_namespace_body(ast);

  expect(TokenKind::T_RBRACE, ast->rbraceLoc);

  return true;
}

auto Parser::parse_namespace_body(NamespaceDefinitionAST* yyast) -> bool {
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

auto Parser::parse_namespace_alias_definition(DeclarationAST*& yyast) -> bool {
  SourceLocation namespaceLoc;

  if (!match(TokenKind::T_NAMESPACE, namespaceLoc)) return false;

  auto ast = new (pool) NamespaceAliasDefinitionAST();
  yyast = ast;

  ast->namespaceLoc = namespaceLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  expect(TokenKind::T_EQUAL, ast->equalLoc);

  if (!parse_qualified_namespace_specifier(ast->nestedNameSpecifier,
                                           ast->name)) {
    parse_error("expected a namespace name");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_qualified_namespace_specifier(
    NestedNameSpecifierAST*& nestedNameSpecifier, NameAST*& name) -> bool {
  const auto saved = currentLocation();

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(saved);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = new (pool) SimpleNameAST();
  id->identifierLoc = identifierLoc;
  id->identifier = unit->identifier(id->identifierLoc);

  name = id;

  return true;
}

auto Parser::parse_using_directive(DeclarationAST*& yyast) -> bool {
  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  SourceLocation namespaceLoc;

  if (!match(TokenKind::T_NAMESPACE, namespaceLoc)) return false;

  auto ast = new (pool) UsingDirectiveAST;
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->namespaceLoc = namespaceLoc;

  const auto saved = currentLocation();

  if (!parse_nested_name_specifier(ast->nestedNameSpecifier)) rewind(saved);

  if (!parse_name_id(ast->name)) parse_error("expected a namespace name");

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_using_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  auto ast = new (pool) UsingDeclarationAST();
  yyast = ast;

  ast->usingLoc = usingLoc;

  if (!parse_using_declarator_list(ast->usingDeclaratorList)) {
    parse_error("expected a using declarator");
  }

  match(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_using_declarator_list(List<UsingDeclaratorAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  UsingDeclaratorAST* declarator = nullptr;

  if (!parse_using_declarator(declarator)) return false;

  match(TokenKind::T_DOT_DOT_DOT);

  *it = new (pool) List(declarator);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    UsingDeclaratorAST* declarator = nullptr;

    if (!parse_using_declarator(declarator)) {
      parse_error("expected a using declarator");
    }

    match(TokenKind::T_DOT_DOT_DOT);

    *it = new (pool) List(declarator);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_using_declarator(UsingDeclaratorAST*& yyast) -> bool {
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

auto Parser::parse_asm_declaration(DeclarationAST*& yyast) -> bool {
  List<AttributeSpecifierAST*>* attributes = nullptr;

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

auto Parser::parse_linkage_specification(DeclarationAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation externLoc;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  if (!match(TokenKind::T_EXTERN, externLoc)) {
    rewind(start);
    return false;
  }

  SourceLocation stringLiteralLoc;

  if (!match(TokenKind::T_STRING_LITERAL, stringLiteralLoc)) {
    rewind(start);
    return false;
  }

  SourceLocation lbraceLoc;

  if (match(TokenKind::T_LBRACE, lbraceLoc)) {
    SourceLocation rbraceLoc;

    auto ast = new (pool) LinkageSpecificationAST();
    yyast = ast;

    ast->externLoc = externLoc;
    ast->stringliteralLoc = stringLiteralLoc;
    ast->stringLiteral =
        static_cast<const StringLiteral*>(unit->literal(ast->stringliteralLoc));
    ast->lbraceLoc = lbraceLoc;

    if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
      if (!parse_declaration_seq(ast->declarationList)) {
        parse_error("expected a declaration");
      }

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
  ast->stringLiteral =
      static_cast<const StringLiteral*>(unit->literal(ast->stringliteralLoc));
  ast->declarationList = new (pool) List(declaration);

  return true;
}

auto Parser::parse_attribute_specifier_seq(List<AttributeSpecifierAST*>*& yyast)
    -> bool {
  auto it = &yyast;
  AttributeSpecifierAST* attribute = nullptr;

  if (!parse_attribute_specifier(attribute)) return false;

  *it = new (pool) List(attribute);
  it = &(*it)->next;

  attribute = nullptr;

  while (parse_attribute_specifier(attribute)) {
    *it = new (pool) List(attribute);
    it = &(*it)->next;
    attribute = nullptr;
  }

  return true;
}

auto Parser::parse_attribute_specifier(AttributeSpecifierAST*& yyast) -> bool {
  if (parse_cxx_attribute_specifier(yyast)) return true;

  if (parse_gcc_attribute(yyast)) return true;

  if (parse_alignment_specifier(yyast)) return true;

  if (parse_asm_specifier(yyast)) return true;

  return false;
}

auto Parser::lookat_cxx_attribute_specifier() -> bool {
  if (LA().isNot(TokenKind::T_LBRACKET)) return false;
  if (LA(1).isNot(TokenKind::T_LBRACKET)) return false;
  if (LA(1).leadingSpace() || LA(1).startOfLine()) return false;
  return true;
}

auto Parser::parse_cxx_attribute_specifier(AttributeSpecifierAST*& yyast)
    -> bool {
  if (!lookat_cxx_attribute_specifier()) return false;

  auto ast = new (pool) CxxAttributeAST();
  yyast = ast;
  ast->lbracketLoc = consumeToken();
  ast->lbracket2Loc = consumeToken();
  parse_attribute_using_prefix(ast->attributeUsingPrefix);
  parse_attribute_list(ast->attributeList);
  expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  expect(TokenKind::T_RBRACKET, ast->rbracket2Loc);
  return true;
}

auto Parser::parse_asm_specifier(AttributeSpecifierAST*& yyast) -> bool {
  SourceLocation asmLoc;

  if (!match(TokenKind::T_ASM, asmLoc)) return false;

  auto ast = new (pool) AsmAttributeAST();
  yyast = ast;

  ast->asmLoc = asmLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  List<SourceLocation>* stringLiteralList = nullptr;

  if (!parse_string_literal_seq(stringLiteralList)) {
    parse_error("expected a string literal");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_gcc_attribute(AttributeSpecifierAST*& yyast) -> bool {
  SourceLocation attributeLoc;

  if (!match(TokenKind::T___ATTRIBUTE__, attributeLoc)) return false;

  auto ast = new (pool) GCCAttributeAST();
  yyast = ast;

  ast->attributeLoc = attributeLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_skip_balanced();

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_skip_balanced() -> bool {
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

auto Parser::parse_alignment_specifier(AttributeSpecifierAST*& yyast) -> bool {
  SourceLocation alignasLoc;
  if (!match(TokenKind::T_ALIGNAS, alignasLoc)) return false;

  auto ast = new (pool) AlignasAttributeAST();
  yyast = ast;

  ast->alignasLoc = alignasLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  const auto after_lparen = currentLocation();

  TypeIdAST* typeId = nullptr;

  if (parse_type_id(typeId)) {
    match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

    if (match(TokenKind::T_RPAREN, ast->rparenLoc)) {
      return true;
    }
  }

  rewind(after_lparen);

  if (!parse_constant_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_attribute_using_prefix(AttributeUsingPrefixAST*& yyast)
    -> bool {
  SourceLocation usingLoc;
  if (!match(TokenKind::T_USING, usingLoc)) return false;

  SourceLocation attributeNamespaceLoc;

  if (!parse_attribute_namespace(attributeNamespaceLoc)) {
    parse_error("expected an attribute namespace");
  }

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  auto ast = new (pool) AttributeUsingPrefixAST();
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->attributeNamespaceLoc = attributeNamespaceLoc;
  ast->colonLoc = colonLoc;

  return true;
}

auto Parser::parse_attribute_list(List<AttributeAST*>*& yyast) -> bool {
  auto it = &yyast;

  AttributeAST* attribute = nullptr;
  parse_attribute(attribute);

  SourceLocation ellipsisLoc;
  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  if (attribute) {
    attribute->ellipsisLoc = ellipsisLoc;

    *it = new (pool) List(attribute);
    it = &(*it)->next;
  }

  while (match(TokenKind::T_COMMA)) {
    AttributeAST* attribute = nullptr;
    parse_attribute(attribute);

    SourceLocation ellipsisLoc;
    match(TokenKind::T_DOT_DOT_DOT);

    if (attribute) {
      attribute->ellipsisLoc = ellipsisLoc;

      *it = new (pool) List(attribute);
      it = &(*it)->next;
    }
  }

  return true;
}

auto Parser::parse_attribute(AttributeAST*& yyast) -> bool {
  AttributeTokenAST* attributeToken = nullptr;

  if (!parse_attribute_token(attributeToken)) return false;

  AttributeArgumentClauseAST* attributeArgumentClause = nullptr;

  parse_attribute_argument_clause(attributeArgumentClause);

  auto ast = new (pool) AttributeAST();
  yyast = ast;

  ast->attributeToken = attributeToken;
  ast->attributeArgumentClause = attributeArgumentClause;

  return true;
}

auto Parser::parse_attribute_token(AttributeTokenAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (parse_attribute_scoped_token(yyast)) return true;

  rewind(start);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = new (pool) SimpleAttributeTokenAST();
  yyast = ast;

  ast->identifierLoc = identifierLoc;

  return true;
}

auto Parser::parse_attribute_scoped_token(AttributeTokenAST*& yyast) -> bool {
  SourceLocation attributeNamespaceLoc;

  if (!parse_attribute_namespace(attributeNamespaceLoc)) return false;

  SourceLocation scopeLoc;

  if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;

  SourceLocation identifierLoc;

  expect(TokenKind::T_IDENTIFIER, identifierLoc);

  auto ast = new (pool) ScopedAttributeTokenAST();
  yyast = ast;

  ast->attributeNamespaceLoc = attributeNamespaceLoc;
  ast->scopeLoc = scopeLoc;
  ast->identifierLoc = identifierLoc;

  return true;
}

auto Parser::parse_attribute_namespace(SourceLocation& attributeNamespaceLoc)
    -> bool {
  if (!match(TokenKind::T_IDENTIFIER, attributeNamespaceLoc)) return false;

  return true;
}

auto Parser::parse_attribute_argument_clause(AttributeArgumentClauseAST*& yyast)
    -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  parse_skip_balanced();

  SourceLocation rparenLoc;

  expect(TokenKind::T_RPAREN, rparenLoc);

  auto ast = new (pool) AttributeArgumentClauseAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_module_declaration(ModuleDeclarationAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation exportLoc;

  parse_export_keyword(exportLoc);

  SourceLocation moduleLoc;

  if (!parse_module_keyword(moduleLoc)) {
    rewind(start);
    return false;
  }

  yyast = new (pool) ModuleDeclarationAST();

  yyast->exportLoc = exportLoc;
  yyast->moduleLoc = moduleLoc;

  if (!parse_module_name(yyast->moduleName)) {
    parse_error("expected a module name");
  }

  if (LA().is(TokenKind::T_COLON)) {
    parse_module_partition(yyast->modulePartition);
  }

  parse_attribute_specifier_seq(yyast->attributeList);

  expect(TokenKind::T_SEMICOLON, yyast->semicolonLoc);

  return true;
}

auto Parser::parse_module_name(ModuleNameAST*& yyast) -> bool {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  yyast = new (pool) ModuleNameAST();

  auto it = &yyast->identifierList;

  *it = new (pool) List(identifierLoc);
  it = &(*it)->next;

  while (match(TokenKind::T_DOT)) {
    SourceLocation identifierLoc;

    expect(TokenKind::T_IDENTIFIER, identifierLoc);

    if (!identifierLoc) break;

    *it = new (pool) List(consumeToken());
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_module_partition(ModulePartitionAST*& yyast) -> bool {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  yyast = new (pool) ModulePartitionAST();

  yyast->colonLoc = colonLoc;

  if (!parse_module_name(yyast->moduleName)) {
    parse_error("expected module name");
  }

  return true;
}

auto Parser::parse_export_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation exportLoc;

  if (!match(TokenKind::T_EXPORT, exportLoc)) return false;

  SourceLocation lbraceLoc;

  if (match(TokenKind::T_LBRACE, lbraceLoc)) {
    auto ast = new (pool) ExportCompoundDeclarationAST();
    yyast = ast;

    ast->exportLoc = exportLoc;
    ast->lbraceLoc = lbraceLoc;

    if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
      if (!parse_declaration_seq(ast->declarationList)) {
        parse_error("expected a declaration");
      }

      expect(TokenKind::T_RBRACE, ast->rbraceLoc);
    }

    return true;
  }

  auto ast = new (pool) ExportDeclarationAST();
  yyast = ast;

  ast->exportLoc = exportLoc;

  if (parse_maybe_import()) {
    if (!parse_module_import_declaration(ast->declaration)) {
      parse_error("expected a module import declaration");
    }

    return true;
  }

  if (!parse_declaration(ast->declaration)) {
    parse_error("expected a declaration");
  }

  return true;
}

auto Parser::parse_maybe_import() -> bool {
  if (!module_unit) return false;

  const auto start = currentLocation();

  SourceLocation importLoc;

  const auto import = parse_import_keyword(importLoc);

  rewind(start);

  return import;
}

auto Parser::parse_module_import_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation importLoc;

  if (!parse_import_keyword(importLoc)) return false;

  auto ast = new (pool) ModuleImportDeclarationAST();
  yyast = ast;

  ast->importLoc = importLoc;

  if (!parse_import_name(ast->importName)) {
    parse_error("expected a module name");
  }

  parse_attribute_specifier_seq(ast->attributeList);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_import_name(ImportNameAST*& yyast) -> bool {
  SourceLocation headerLoc;

  if (parse_header_name(headerLoc)) return true;

  yyast = new (pool) ImportNameAST();

  yyast->headerLoc = headerLoc;

  if (parse_module_partition(yyast->modulePartition)) return true;

  if (!parse_module_name(yyast->moduleName)) {
    parse_error("expected module name");
  }

  return true;
}

auto Parser::parse_global_module_fragment(GlobalModuleFragmentAST*& yyast)
    -> bool {
  const auto start = currentLocation();

  SourceLocation moduleLoc;

  if (!parse_module_keyword(moduleLoc)) {
    rewind(start);
    return false;
  }

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    rewind(start);
    return false;
  }

  yyast = new (pool) GlobalModuleFragmentAST();
  yyast->moduleLoc = moduleLoc;
  yyast->semicolonLoc = semicolonLoc;

  // ### must be from preprocessor inclusion
  parse_declaration_seq(yyast->declarationList);

  return true;
}

auto Parser::parse_private_module_fragment(PrivateModuleFragmentAST*& yyast)
    -> bool {
  const auto start = currentLocation();

  SourceLocation moduleLoc;

  if (!parse_module_keyword(moduleLoc)) return false;

  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) {
    rewind(start);
    return false;
  }

  SourceLocation privateLoc;

  if (!match(TokenKind::T_PRIVATE, privateLoc)) {
    rewind(start);
    return false;
  }

  yyast = new (pool) PrivateModuleFragmentAST();

  yyast->moduleLoc = moduleLoc;
  yyast->colonLoc = colonLoc;
  yyast->privateLoc = privateLoc;

  expect(TokenKind::T_SEMICOLON, yyast->semicolonLoc);

  parse_declaration_seq(yyast->declarationList);

  return true;
}

auto Parser::parse_class_specifier(SpecifierAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation classKey;

  const auto maybeClassSpecifier = parse_class_key(classKey);

  rewind(start);

  if (!maybeClassSpecifier) return false;

  auto it = class_specifiers_.find(start);

  if (it != class_specifiers_.end()) {
    auto [cursor, ast, parsed] = it->second;
    rewind(cursor);
    yyast = ast;
    return parsed;
  }

  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NameAST* className = nullptr;
  BaseClauseAST* baseClause = nullptr;

  if (!parse_class_head(classLoc, attributeList, className, baseClause)) {
    class_specifiers_.emplace(
        start,
        std::make_tuple(currentLocation(),
                        static_cast<ClassSpecifierAST*>(nullptr), false));
    return false;
  }

  ClassSpecifierContext classContext(this);

  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) {
    class_specifiers_.emplace(
        start,
        std::make_tuple(currentLocation(),
                        static_cast<ClassSpecifierAST*>(nullptr), false));
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
    if (!parse_class_body(ast->declarationList)) {
      parse_error("expected class body");
    }

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  class_specifiers_.emplace(start,
                            std::make_tuple(currentLocation(), ast, true));

  return true;
}

auto Parser::parse_class_body(List<DeclarationAST*>*& yyast) -> bool {
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

auto Parser::parse_class_head(SourceLocation& classLoc,
                              List<AttributeSpecifierAST*>*& attributeList,
                              NameAST*& name, BaseClauseAST*& baseClause)
    -> bool {
  if (!parse_class_key(classLoc)) return false;

  parse_attribute_specifier_seq(attributeList);

  if (parse_class_head_name(name)) {
    parse_class_virt_specifier();
  }

  parse_base_clause(baseClause);

  return true;
}

auto Parser::parse_class_head_name(NameAST*& yyast) -> bool {
  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(start);

  check_type_traits();

  NameAST* name = nullptr;

  if (!parse_type_name(name)) return false;

  if (!nestedNameSpecifier) yyast = name;

  return true;
}

auto Parser::parse_class_virt_specifier() -> bool {
  if (!parse_final()) return false;

  return true;
}

auto Parser::parse_class_key(SourceLocation& classLoc) -> bool {
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

auto Parser::parse_member_specification(DeclarationAST*& yyast) -> bool {
  return parse_member_declaration(yyast);
}

auto Parser::parse_member_declaration(DeclarationAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation accessLoc;

  if (parse_access_specifier(accessLoc)) {
    auto ast = new (pool) AccessDeclarationAST();
    yyast = ast;

    ast->accessLoc = accessLoc;
    expect(TokenKind::T_COLON, ast->colonLoc);

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

auto Parser::parse_maybe_template_member() -> bool {
  const auto start = currentLocation();

  match(TokenKind::T_EXPLICIT);

  const auto has_template = match(TokenKind::T_TEMPLATE);

  rewind(start);

  return has_template;
}

auto Parser::parse_member_declaration_helper(DeclarationAST*& yyast) -> bool {
  SourceLocation extensionLoc;

  match(TokenKind::T___EXTENSION__, extensionLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  auto after_decl_specs = currentLocation();

  DeclSpecs specs;

  List<SpecifierAST*>* declSpecifierList = nullptr;

  if (!parse_decl_specifier_seq_no_typespecs(declSpecifierList, specs)) {
    rewind(after_decl_specs);
  }

  after_decl_specs = currentLocation();

  if (parse_notypespec_function_definition(yyast, declSpecifierList, specs)) {
    return true;
  }

  rewind(after_decl_specs);

  auto lastDeclSpecifier = &declSpecifierList;

  while (*lastDeclSpecifier) {
    lastDeclSpecifier = &(*lastDeclSpecifier)->next;
  }

  if (!parse_decl_specifier_seq(*lastDeclSpecifier, specs)) {
    rewind(after_decl_specs);
  }

  after_decl_specs = currentLocation();

  if (!specs.has_typespec()) return false;

  SourceLocation semicolonLoc;

  if (match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    auto ast = new (pool) SimpleDeclarationAST();
    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->semicolonLoc = semicolonLoc;
    yyast = ast;
    return true;  // ### complex typespec
  }

  DeclaratorAST* declarator = nullptr;

  const auto hasDeclarator = parse_declarator(declarator);

  if (hasDeclarator && getFunctionDeclarator(declarator)) {
    RequiresClauseAST* requiresClause = nullptr;

    const auto has_requires_clause = parse_requires_clause(requiresClause);

    if (!has_requires_clause) parse_virt_specifier_seq();

    if (lookat_function_body()) {
      FunctionBodyAST* functionBody = nullptr;

      if (!parse_function_body(functionBody)) {
        parse_error("expected function body");
      }

      auto ast = new (pool) FunctionDefinitionAST();
      yyast = ast;

      ast->declSpecifierList = declSpecifierList;
      ast->declarator = declarator;
      ast->requiresClause = requiresClause;
      ast->functionBody = functionBody;

      if (classDepth) pendingFunctionDefinitions_.push_back(ast);

      return true;
    }
  }

  rewind(after_decl_specs);

  auto ast = new (pool) SimpleDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->declSpecifierList = declSpecifierList;

  if (!parse_member_declarator_list(ast->initDeclaratorList, specs)) {
    parse_error("expected a declarator");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_member_declarator_list(List<InitDeclaratorAST*>*& yyast,
                                          const DeclSpecs& specs) -> bool {
  auto it = &yyast;

  InitDeclaratorAST* initDeclarator = nullptr;

  if (!parse_member_declarator(initDeclarator, specs)) return false;

  if (initDeclarator) {
    *it = new (pool) List(initDeclarator);
    it = &(*it)->next;
  }

  while (match(TokenKind::T_COMMA)) {
    InitDeclaratorAST* initDeclarator = nullptr;

    if (!parse_member_declarator(initDeclarator, specs)) {
      parse_error("expected a declarator");
    }

    if (initDeclarator) {
      *it = new (pool) List(initDeclarator);
      it = &(*it)->next;
    }
  }

  return true;
}

auto Parser::parse_member_declarator(InitDeclaratorAST*& yyast,
                                     const DeclSpecs& specs) -> bool {
  const auto start = currentLocation();

  SourceLocation identifierLoc;

  match(TokenKind::T_IDENTIFIER, identifierLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_attribute_specifier_seq(attributes);

  if (match(TokenKind::T_COLON)) {
    // ### TODO bit field declarators

    auto name = new (pool) SimpleNameAST();
    name->identifierLoc = identifierLoc;
    name->identifier = unit->identifier(name->identifierLoc);

    auto coreDeclarator = new (pool) IdDeclaratorAST();
    coreDeclarator->name = name;

    auto declarator = new (pool) DeclaratorAST();
    declarator->coreDeclarator = coreDeclarator;

    ExpressionAST* expression = nullptr;

    if (!parse_constant_expression(expression)) {
      parse_error("expected an expression");
    }

    InitializerAST* initializer = nullptr;

    parse_brace_or_equal_initializer(initializer);

    auto ast = new (pool) InitDeclaratorAST();
    yyast = ast;

    ast->declarator = declarator;
    ast->initializer = initializer;

    return true;
  }

  rewind(start);

  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  auto ast = new (pool) InitDeclaratorAST();
  yyast = ast;

  ast->declarator = declarator;

  if (getFunctionDeclarator(declarator)) {
    RequiresClauseAST* requiresClause = nullptr;

    if (parse_requires_clause(requiresClause)) {
      ast->requiresClause = requiresClause;
    } else {
      parse_virt_specifier_seq();
      parse_pure_specifier();
    }

    return true;
  }

  parse_brace_or_equal_initializer(ast->initializer);

  return true;
}

auto Parser::parse_virt_specifier() -> bool {
  if (parse_final()) return true;

  if (parse_override()) return true;

  return false;
}

auto Parser::parse_pure_specifier() -> bool {
  if (!match(TokenKind::T_EQUAL)) return false;

  SourceLocation literalLoc;

  if (!match(TokenKind::T_INTEGER_LITERAL, literalLoc)) return false;

  const auto& number = unit->tokenText(literalLoc);

  if (number != "0") return false;

  return true;
}

auto Parser::parse_conversion_function_id(NameAST*& yyast) -> bool {
  SourceLocation operatorLoc;

  if (!match(TokenKind::T_OPERATOR, operatorLoc)) return false;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList)) return false;

  auto declarator = new (pool) DeclaratorAST();

  parse_ptr_operator_seq(declarator->ptrOpList);

  auto typeId = new (pool) TypeIdAST();
  typeId->typeSpecifierList = typeSpecifierList;
  typeId->declarator = declarator;

  auto ast = new (pool) ConversionNameAST();
  yyast = ast;

  ast->operatorLoc = operatorLoc;
  ast->typeId = typeId;

  return true;
}

auto Parser::parse_base_clause(BaseClauseAST*& yyast) -> bool {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  auto ast = new (pool) BaseClauseAST();
  yyast = ast;

  ast->colonLoc = colonLoc;

  if (!parse_base_specifier_list(ast->baseSpecifierList)) {
    parse_error("expected a base class specifier");
  }

  return true;
}

auto Parser::parse_base_specifier_list(List<BaseSpecifierAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  BaseSpecifierAST* baseSpecifier = nullptr;

  if (!parse_base_specifier(baseSpecifier)) return false;

  match(TokenKind::T_DOT_DOT_DOT);

  *it = new (pool) List(baseSpecifier);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    BaseSpecifierAST* baseSpecifier = nullptr;

    if (!parse_base_specifier(baseSpecifier)) {
      parse_error("expected a base class specifier");
    }

    match(TokenKind::T_DOT_DOT_DOT);

    *it = new (pool) List(baseSpecifier);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_base_specifier(BaseSpecifierAST*& yyast) -> bool {
  auto ast = new (pool) BaseSpecifierAST();
  yyast = ast;

  parse_attribute_specifier_seq(ast->attributeList);

  SourceLocation virtualLoc;
  SourceLocation accessLoc;

  if (match(TokenKind::T_VIRTUAL, virtualLoc)) {
    parse_access_specifier(accessLoc);
  } else if (parse_access_specifier(accessLoc)) {
    match(TokenKind::T_VIRTUAL, virtualLoc);
  }

  if (!parse_class_or_decltype(ast->name)) return false;

  return true;
}

auto Parser::parse_class_or_decltype(NameAST*& yyast) -> bool {
  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (parse_nested_name_specifier(nestedNameSpecifier)) {
    NameAST* name = nullptr;

    SourceLocation templateLoc;

    if (match(TokenKind::T_TEMPLATE, templateLoc) &&
        parse_simple_template_id(name)) {
      auto ast = new (pool) QualifiedNameAST();
      yyast = ast;

      ast->nestedNameSpecifier = nestedNameSpecifier;
      ast->templateLoc = templateLoc;
      ast->id = name;

      return true;
    }

    if (parse_type_name(name)) {
      auto ast = new (pool) QualifiedNameAST();
      yyast = ast;

      ast->nestedNameSpecifier = nestedNameSpecifier;
      ast->id = name;

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

auto Parser::parse_access_specifier(SourceLocation& loc) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_PRIVATE:
    case TokenKind::T_PROTECTED:
    case TokenKind::T_PUBLIC:
      loc = consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

auto Parser::parse_ctor_initializer(CtorInitializerAST*& yyast) -> bool {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  auto ast = new (pool) CtorInitializerAST();
  yyast = ast;

  ast->colonLoc = colonLoc;

  if (!parse_mem_initializer_list(ast->memInitializerList)) {
    parse_error("expected a member intializer");
  }

  return true;
}

auto Parser::parse_mem_initializer_list(List<MemInitializerAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  MemInitializerAST* mem_initializer = nullptr;

  if (!parse_mem_initializer(mem_initializer)) return false;

  *it = new (pool) List(mem_initializer);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    MemInitializerAST* mem_initializer = nullptr;

    if (!parse_mem_initializer(mem_initializer)) {
      parse_error("expected a member initializer");
    } else {
      *it = new (pool) List(mem_initializer);
      it = &(*it)->next;
    }
  }

  return true;
}

auto Parser::parse_mem_initializer(MemInitializerAST*& yyast) -> bool {
  NameAST* name = nullptr;

  if (!parse_mem_initializer_id(name)) parse_error("expected an member id");

  if (LA().is(TokenKind::T_LBRACE)) {
    auto ast = new (pool) BracedMemInitializerAST();
    yyast = ast;

    ast->name = name;

    if (!parse_braced_init_list(ast->bracedInitList)) {
      parse_error("expected an initializer");
    }

    match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

    return true;
  }

  auto ast = new (pool) ParenMemInitializerAST();
  yyast = ast;

  ast->name = name;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_expression_list(ast->expressionList)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  return true;
}

auto Parser::parse_mem_initializer_id(NameAST*& yyast) -> bool {
  const auto start = currentLocation();

  NameAST* name = nullptr;

  if (parse_class_or_decltype(name)) return true;

  rewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

auto Parser::parse_operator_function_id(NameAST*& yyast) -> bool {
  SourceLocation operatorLoc;

  if (!match(TokenKind::T_OPERATOR, operatorLoc)) return false;

  TokenKind op = TokenKind::T_EOF_SYMBOL;
  SourceLocation opLoc;
  SourceLocation openLoc;
  SourceLocation closeLoc;

  if (!parse_operator(op, opLoc, openLoc, closeLoc)) return false;

  auto ast = new (pool) OperatorNameAST();
  yyast = ast;

  ast->operatorLoc = operatorLoc;
  ast->opLoc = opLoc;
  ast->openLoc = openLoc;
  ast->closeLoc = closeLoc;
  ast->op = op;

  return true;
}

auto Parser::parse_operator(TokenKind& op, SourceLocation& opLoc,
                            SourceLocation& openLoc, SourceLocation& closeLoc)
    -> bool {
  op = TokenKind(LA());
  switch (op) {
    case TokenKind::T_LPAREN:
      openLoc = consumeToken();
      expect(TokenKind::T_RPAREN, closeLoc);
      return true;

    case TokenKind::T_LBRACKET:
      openLoc = consumeToken();
      expect(TokenKind::T_RBRACKET, closeLoc);
      return true;

    case TokenKind::T_GREATER: {
      opLoc = currentLocation();
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
      consumeToken();
      return true;
    }

    case TokenKind::T_NEW:
      opLoc = consumeToken();
      if (match(TokenKind::T_LBRACKET, openLoc)) {
        expect(TokenKind::T_RBRACKET, closeLoc);
        op = TokenKind::T_NEW_ARRAY;
      }
      return true;

    case TokenKind::T_DELETE:
      opLoc = consumeToken();
      if (match(TokenKind::T_LBRACKET, openLoc)) {
        expect(TokenKind::T_RBRACKET, closeLoc);
        op = TokenKind::T_DELETE_ARRAY;
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
      opLoc = consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

auto Parser::parse_literal_operator_id(NameAST*& yyast) -> bool {
  if (!match(TokenKind::T_OPERATOR)) return false;

  if (match(TokenKind::T_USER_DEFINED_STRING_LITERAL)) return true;

  List<SourceLocation>* stringLiteralList = nullptr;

  if (!parse_string_literal_seq(stringLiteralList)) return false;

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

auto Parser::parse_template_declaration(DeclarationAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation templateLoc;

  if (!match(TokenKind::T_TEMPLATE, templateLoc)) return false;

  SourceLocation lessLoc;

  if (!match(TokenKind::T_LESS, lessLoc)) {
    rewind(start);
    return false;
  }

  SourceLocation greaterLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;

  if (!match(TokenKind::T_GREATER, greaterLoc)) {
    if (!parse_template_parameter_list(templateParameterList)) {
      parse_error("expected a template parameter");
    }

    expect(TokenKind::T_GREATER, greaterLoc);
  }

  RequiresClauseAST* requiresClause = nullptr;
  parse_requires_clause(requiresClause);

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
  ast->requiresClause = requiresClause;
  ast->declaration = declaration;

  return true;
}

auto Parser::parse_template_parameter_list(List<DeclarationAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  DeclarationAST* declaration = nullptr;

  if (!parse_template_parameter(declaration)) return false;

  *it = new (pool) List(declaration);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    DeclarationAST* declaration = nullptr;

    if (!parse_template_parameter(declaration)) {
      parse_error("expected a template parameter");
    }

    *it = new (pool) List(declaration);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_requires_clause(RequiresClauseAST*& yyast) -> bool {
  SourceLocation requiresLoc;

  if (!match(TokenKind::T_REQUIRES, requiresLoc)) return false;

  yyast = new (pool) RequiresClauseAST();

  yyast->requiresLoc = requiresLoc;

  if (!parse_constraint_logical_or_expression(yyast->expression)) return false;

  return true;
}

auto Parser::parse_constraint_logical_or_expression(ExpressionAST*& yyast)
    -> bool {
  if (!parse_constraint_logical_and_expression(yyast)) return false;

  SourceLocation opLoc;

  while (match(TokenKind::T_BAR_BAR, opLoc)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constraint_logical_and_expression(expression)) {
      parse_error("expected a requirement expression");
    }

    auto ast = new (pool) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->op = TokenKind::T_BAR_BAR;
    ast->rightExpression = expression;
    yyast = ast;
  }

  return true;
}

auto Parser::parse_constraint_logical_and_expression(ExpressionAST*& yyast)
    -> bool {
  if (!parse_primary_expression(yyast, /*inRequiresClause*/ true)) return false;

  SourceLocation opLoc;

  while (match(TokenKind::T_AMP_AMP, opLoc)) {
    ExpressionAST* expression = nullptr;

    if (!parse_primary_expression(expression, /*inRequiresClause*/ true)) {
      parse_error("expected an expression");
    }

    auto ast = new (pool) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->op = TokenKind::T_AMP_AMP;
    ast->rightExpression = expression;
    yyast = ast;
  }

  return true;
}

auto Parser::parse_template_parameter(DeclarationAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (parse_type_parameter(yyast)) return true;

  rewind(start);

  ParameterDeclarationAST* parameter = nullptr;

  if (parse_parameter_declaration(parameter, /*templParam*/ true)) {
    yyast = parameter;
    return true;
  }

  rewind(start);

  return parse_constraint_type_parameter(yyast);
}

auto Parser::parse_type_parameter(DeclarationAST*& yyast) -> bool {
  if (parse_template_type_parameter(yyast)) return true;

  if (parse_typename_type_parameter(yyast)) return true;

  return false;
}

auto Parser::parse_typename_type_parameter(DeclarationAST*& yyast) -> bool {
  auto maybe_elaborated_type_spec = [this]() {
    if (!match(TokenKind::T_TYPENAME)) return false;
    if (!match(TokenKind::T_IDENTIFIER)) return false;
    if (match(TokenKind::T_COLON_COLON) || match(TokenKind::T_LESS)) {
      return true;
    }
    return false;
  };

  const auto start = currentLocation();
  const auto is_type_spec = maybe_elaborated_type_spec();
  rewind(start);

  if (is_type_spec) return false;

  SourceLocation classKeyLoc;

  if (!parse_type_parameter_key(classKeyLoc)) return false;

  auto ast = new (pool) TypenameTypeParameterAST();
  yyast = ast;

  ast->classKeyLoc = classKeyLoc;

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  if (!match(TokenKind::T_EQUAL, ast->equalLoc)) return true;

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  return true;
}

auto Parser::parse_template_type_parameter(DeclarationAST*& yyast) -> bool {
  const auto start = currentLocation();

  SourceLocation templateLoc;

  if (!match(TokenKind::T_TEMPLATE, templateLoc)) return false;

  SourceLocation lessLoc;

  if (!match(TokenKind::T_LESS, lessLoc)) {
    rewind(start);
    return false;
  }

  SourceLocation greaterLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;

  if (!match(TokenKind::T_GREATER, greaterLoc)) {
    if (!parse_template_parameter_list(templateParameterList)) {
      parse_error("expected a template parameter");
    }

    expect(TokenKind::T_GREATER, greaterLoc);
  }

  RequiresClauseAST* requiresClause = nullptr;
  parse_requires_clause(requiresClause);

  SourceLocation classsKeyLoc;

  if (!parse_type_parameter_key(classsKeyLoc)) {
    parse_error("expected a type parameter");
  }

  if ((LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_EQUAL)) ||
      LA().is(TokenKind::T_EQUAL)) {
    auto ast = new (pool) TemplateTypeParameterAST();
    yyast = ast;

    ast->templateLoc = templateLoc;
    ast->lessLoc = lessLoc;
    ast->templateParameterList = templateParameterList;
    ast->greaterLoc = greaterLoc;
    ast->requiresClause = requiresClause;
    ast->classKeyLoc = classsKeyLoc;

    match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

    ast->identifier = unit->identifier(ast->identifierLoc);

    expect(TokenKind::T_EQUAL, ast->equalLoc);

    if (!parse_id_expression(ast->name)) {
      parse_error("expected an id-expression");
    }

    return true;
  }

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    SourceLocation identifierLoc;

    match(TokenKind::T_IDENTIFIER, identifierLoc);

    auto ast = new (pool) TemplatePackTypeParameterAST();
    yyast = ast;

    ast->templateLoc = templateLoc;
    ast->lessLoc = lessLoc;
    ast->templateParameterList = templateParameterList;
    ast->greaterLoc = greaterLoc;
    ast->classKeyLoc = classsKeyLoc;
    ast->ellipsisLoc = ellipsisLoc;
    ast->identifierLoc = identifierLoc;
    ast->identifier = unit->identifier(ast->identifierLoc);
  }

  auto ast = new (pool) TemplateTypeParameterAST();
  yyast = ast;

  ast->templateLoc = templateLoc;
  ast->lessLoc = lessLoc;
  ast->templateParameterList = templateParameterList;
  ast->greaterLoc = greaterLoc;
  ast->classKeyLoc = classsKeyLoc;

  match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  return true;
}

auto Parser::parse_constraint_type_parameter(DeclarationAST*& yyast) -> bool {
  TypeConstraintAST* typeConstraint = nullptr;

  if (!parse_type_constraint(typeConstraint, /*parsing placeholder=*/false)) {
    return false;
  }

  if ((LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_EQUAL)) ||
      LA().is(TokenKind::T_EQUAL)) {
    match(TokenKind::T_IDENTIFIER);

    expect(TokenKind::T_EQUAL);

    TypeIdAST* typeId = nullptr;

    if (!parse_type_id(typeId)) {
      return false;  // ### FIXME: parse_error("expected a type id");
    }

    return true;
  }

  match(TokenKind::T_DOT_DOT_DOT);

  match(TokenKind::T_IDENTIFIER);

  return true;
}

auto Parser::parse_type_parameter_key(SourceLocation& classKeyLoc) -> bool {
  if (!match(TokenKind::T_CLASS, classKeyLoc) &&
      !match(TokenKind::T_TYPENAME, classKeyLoc)) {
    return false;
  }

  return true;
}

auto Parser::parse_type_constraint(TypeConstraintAST*& yyast,
                                   bool parsingPlaceholderTypeSpec) -> bool {
  const auto start = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(start);

  NameAST* name = nullptr;

  if (!parse_concept_name(name)) {
    rewind(start);
    return false;
  }

  auto ast = new (pool) TypeConstraintAST();
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;

  SourceLocation lessLoc;

  if (match(TokenKind::T_LESS, lessLoc)) {
    SourceLocation greaterLoc;

    List<TemplateArgumentAST*>* templateArgumentList = nullptr;

    if (!parse_template_argument_list(templateArgumentList)) {
      if (parsingPlaceholderTypeSpec) return false;
      parse_error("expected a template argument");
    }

    expect(TokenKind::T_GREATER, greaterLoc);

    auto templateId = new (pool) TemplateNameAST();
    templateId->id = name;
    templateId->lessLoc = lessLoc;
    templateId->templateArgumentList = templateArgumentList;
    templateId->greaterLoc = greaterLoc;

    ast->name = templateId;
  }

  return true;
}

auto Parser::parse_simple_template_id(NameAST*& yyast) -> bool {
  if (LA().isNot(TokenKind::T_IDENTIFIER) || LA(1).isNot(TokenKind::T_LESS)) {
    return false;
  }

  NameAST* name = nullptr;

  if (!parse_name_id(name)) return false;

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

  ast->id = name;
  ast->lessLoc = lessLoc;
  ast->templateArgumentList = templateArgumentList;
  ast->greaterLoc = greaterLoc;

  return true;
}

auto Parser::parse_template_id(NameAST*& yyast) -> bool {
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

    ast->id = name;
    ast->lessLoc = lessLoc;
    ast->templateArgumentList = templateArgumentList;
    ast->greaterLoc = greaterLoc;

    return true;
  }

  return parse_simple_template_id(yyast);
}

auto Parser::parse_template_argument_list(List<TemplateArgumentAST*>*& yyast)
    -> bool {
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

auto Parser::parse_template_argument(TemplateArgumentAST*& yyast) -> bool {
  const auto start = currentLocation();

  auto it = template_arguments_.find(start);

  if (it != template_arguments_.end()) {
    auto [loc, ast, parsed] = it->second;
    rewind(loc);
    yyast = ast;
    return parsed;
  }

  auto check = [&]() -> bool {
    const auto& tk = LA();

    return tk.is(TokenKind::T_COMMA) || tk.is(TokenKind::T_GREATER) ||
           tk.is(TokenKind::T_DOT_DOT_DOT);
  };

  const auto saved = currentLocation();

  TypeIdAST* typeId = nullptr;

  if (parse_type_id(typeId) && check()) {
    auto ast = new (pool) TypeTemplateArgumentAST();
    yyast = ast;

    ast->typeId = typeId;

    template_arguments_.emplace(
        start, std::make_tuple(currentLocation(), yyast, true));

    return true;
  }

  rewind(saved);

  ExpressionAST* expression = nullptr;

  const auto parsed = parse_template_argument_constant_expression(expression);

  if (parsed && check()) {
    auto ast = new (pool) ExpressionTemplateArgumentAST();
    yyast = ast;

    ast->expression = expression;

    template_arguments_.emplace(
        start, std::make_tuple(currentLocation(), yyast, true));

    return true;
  }

  template_arguments_.emplace(
      start, std::make_tuple(currentLocation(), nullptr, false));

  return false;
}

auto Parser::parse_constraint_expression(ExpressionAST*& yyast) -> bool {
  ExprContext exprContext;
  return parse_logical_or_expression(yyast, exprContext);
}

auto Parser::parse_deduction_guide(DeclarationAST*& yyast) -> bool {
  SpecifierAST* explicitSpecifier = nullptr;

  parse_explicit_specifier(explicitSpecifier);

  NameAST* name = nullptr;

  if (!parse_name_id(name)) return false;

  if (!match(TokenKind::T_LPAREN)) return false;

  if (!match(TokenKind::T_RPAREN)) {
    ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

    if (!parse_parameter_declaration_clause(parameterDeclarationClause)) {
      parse_error("expected a parameter declaration");
    }

    expect(TokenKind::T_RPAREN);
  }

  if (!match(TokenKind::T_MINUS_GREATER)) return false;

  NameAST* templateId = nullptr;

  if (!parse_simple_template_id(templateId)) {
    parse_error("expected a template id");
  }

  expect(TokenKind::T_SEMICOLON);

  return true;
}

auto Parser::parse_concept_definition(DeclarationAST*& yyast) -> bool {
  SourceLocation conceptLoc;

  if (!match(TokenKind::T_CONCEPT, conceptLoc)) return false;

  auto ast = new (pool) ConceptDefinitionAST();
  yyast = ast;

  ast->conceptLoc = conceptLoc;

  if (!parse_concept_name(ast->name)) parse_error("expected a concept name");

  expect(TokenKind::T_EQUAL, ast->equalLoc);

  if (!parse_constraint_expression(ast->expression)) {
    parse_error("expected a constraint expression");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_concept_name(NameAST*& yyast) -> bool {
  return parse_name_id(yyast);
}

auto Parser::parse_typename_specifier(SpecifierAST*& yyast) -> bool {
  SourceLocation typenameLoc;

  if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

  const auto after_nested_name_specifier = currentLocation();

  match(TokenKind::T_TEMPLATE);

  NameAST* name = nullptr;

  if (parse_simple_template_id(name)) {
    auto ast = new (pool) TypenameSpecifierAST();
    yyast = ast;

    ast->typenameLoc = typenameLoc;
    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->name = name;

    return true;
  }

  rewind(after_nested_name_specifier);

  if (!parse_name_id(name)) return false;

  auto ast = new (pool) TypenameSpecifierAST();
  yyast = ast;

  ast->typenameLoc = typenameLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;

  return true;
}

auto Parser::parse_explicit_instantiation(DeclarationAST*& yyast) -> bool {
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

auto Parser::parse_explicit_specialization(DeclarationAST*& yyast) -> bool {
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

auto Parser::parse_try_block(StatementAST*& yyast) -> bool {
  SourceLocation tryLoc;

  if (!match(TokenKind::T_TRY, tryLoc)) return false;

  auto ast = new (pool) TryBlockStatementAST();
  yyast = ast;

  ast->tryLoc = tryLoc;

  if (!parse_compound_statement(ast->statement)) {
    parse_error("expected a compound statement");
  }

  if (!parse_handler_seq(ast->handlerList)) {
    parse_error("expected an exception handler");
  }

  return true;
}

auto Parser::parse_function_try_block(FunctionBodyAST*& yyast) -> bool {
  SourceLocation tryLoc;

  if (!match(TokenKind::T_TRY, tryLoc)) return false;

  auto ast = new (pool) TryStatementFunctionBodyAST();
  yyast = ast;

  ast->tryLoc = tryLoc;

  if (LA().isNot(TokenKind::T_LBRACE)) {
    if (!parse_ctor_initializer(ast->ctorInitializer)) {
      parse_error("expected a ctor initializer");
    }
  }

  if (!parse_compound_statement(ast->statement)) {
    parse_error("expected a compound statement");
  }

  if (!parse_handler_seq(ast->handlerList)) {
    parse_error("expected an exception handler");
  }

  return true;
}

auto Parser::parse_handler(HandlerAST*& yyast) -> bool {
  SourceLocation catchLoc;

  if (!match(TokenKind::T_CATCH, catchLoc)) return false;

  yyast = new (pool) HandlerAST();

  yyast->catchLoc = catchLoc;

  expect(TokenKind::T_LPAREN, yyast->lparenLoc);

  if (!parse_exception_declaration(yyast->exceptionDeclaration)) {
    parse_error("expected an exception declaration");
  }

  expect(TokenKind::T_RPAREN, yyast->rparenLoc);

  if (!parse_compound_statement(yyast->statement)) {
    parse_error("expected a compound statement");
  }

  return true;
}

auto Parser::parse_handler_seq(List<HandlerAST*>*& yyast) -> bool {
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

auto Parser::parse_exception_declaration(ExceptionDeclarationAST*& yyast)
    -> bool {
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

  if (!parse_type_specifier_seq(ast->typeSpecifierList)) {
    parse_error("expected a type specifier");
  }

  if (LA().is(TokenKind::T_RPAREN)) return true;

  const auto before_declarator = currentLocation();

  if (!parse_declarator(ast->declarator)) {
    rewind(before_declarator);

    if (!parse_abstract_declarator(ast->declarator)) rewind(before_declarator);
  }

  return true;
}

auto Parser::parse_noexcept_specifier() -> bool {
  if (match(TokenKind::T_THROW)) {
    expect(TokenKind::T_LPAREN);
    expect(TokenKind::T_RPAREN);
    return true;
  }

  if (!match(TokenKind::T_NOEXCEPT)) return false;

  if (match(TokenKind::T_LPAREN)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constant_expression(expression)) {
      parse_error("expected a declaration");
    }

    expect(TokenKind::T_RPAREN);
  }

  return true;
}

auto Parser::parse_identifier_list() -> bool {
  if (!match(TokenKind::T_IDENTIFIER)) return false;

  while (match(TokenKind::T_COMMA)) {
    expect(TokenKind::T_IDENTIFIER);
  }

  return true;
}

void Parser::completePendingFunctionDefinitions() {
  if (pendingFunctionDefinitions_.empty()) return;

  std::vector<FunctionDefinitionAST*> functions;

  std::swap(pendingFunctionDefinitions_, functions);

  for (const auto& function : functions) {
    completeFunctionDefinition(function);
  }
}

void Parser::completeFunctionDefinition(FunctionDefinitionAST* ast) {
  if (!ast->functionBody) return;

  auto functionBody =
      dynamic_cast<CompoundStatementFunctionBodyAST*>(ast->functionBody);

  if (!functionBody) return;

  auto functionSymbol = ast->symbol;

  const auto saved = currentLocation();

  rewind(functionBody->statement->lbraceLoc.next());

  finish_compound_statement(functionBody->statement);

  rewind(saved);
}

}  // namespace cxx
