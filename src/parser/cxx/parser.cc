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
#include <cxx/token_fwd.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <forward_list>
#include <iostream>
#include <unordered_map>
#include <variant>

namespace cxx {

namespace {

class RecordingDiagnosticsClient : public DiagnosticsClient {
 public:
  void reset() { messages_.clear(); }

  void reportTo(DiagnosticsClient* client) {
    for (const auto& message : messages_) {
      client->report(message);
    }
  }

  auto messages() const -> const std::vector<Diagnostic>& { return messages_; }

  void report(const Diagnostic& message) override {
    messages_.push_back(message);
  }

 private:
  std::vector<Diagnostic> messages_;
};

auto getFunctionDeclaratorHelper(DeclaratorAST* declarator)
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

auto getFunctionDeclarator(DeclaratorAST* declarator)
    -> FunctionDeclaratorAST* {
  return get<0>(getFunctionDeclaratorHelper(declarator));
}

}  // namespace

Parser::Parser(TranslationUnit* unit) : unit(unit) {
  control = unit->control();
  cursor_ = 1;

  pool = unit->arena();

  moduleId_ = control->getIdentifier("module");
  importId_ = control->getIdentifier("import");
  finalId_ = control->getIdentifier("final");
  overrideId_ = control->getIdentifier("override");
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

struct Parser::LookaheadParser {
  Parser* p;
  SourceLocation loc;
  RecordingDiagnosticsClient client;
  DiagnosticsClient* previousClient = nullptr;
  bool committed = false;

  LookaheadParser(const LookaheadParser&) = delete;
  auto operator=(const LookaheadParser&) -> LookaheadParser& = delete;

  explicit LookaheadParser(Parser* p) : p(p), loc(p->currentLocation()) {
    previousClient = p->unit->changeDiagnosticsClient(&client);
  }

  ~LookaheadParser() {
    p->unit->changeDiagnosticsClient(previousClient);

    if (!committed) {
      p->rewind(loc);
    } else {
      client.reportTo(p->unit->diagnosticsClient());
    }
  }

  void commit() { committed = true; }
};

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

  explicit TemplArgContext(Parser* p) : p(p) { ++p->templArgDepth_; }
  ~TemplArgContext() { --p->templArgDepth_; }
};

struct Parser::ClassSpecifierContext {
  ClassSpecifierContext(const ClassSpecifierContext&) = delete;
  auto operator=(const ClassSpecifierContext&)
      -> ClassSpecifierContext& = delete;

  Parser* p;

  explicit ClassSpecifierContext(Parser* p) : p(p) { ++p->classDepth_; }

  ~ClassSpecifierContext() {
    if (--p->classDepth_ == 0) p->completePendingFunctionDefinitions();
  }
};

auto Parser::LA(int n) const -> const Token& {
  return unit->tokenAt(SourceLocation(cursor_ + n));
}

auto Parser::match(TokenKind tk, SourceLocation& location) -> bool {
  if (lookat(tk)) {
    location = consumeToken();
    return true;
  }

  location = {};
  return false;
}

auto Parser::expect(TokenKind tk, SourceLocation& location) -> bool {
  if (match(tk, location)) return true;
  parse_error(fmt::format("expected '{}'", Token::spell(tk)));
  return false;
}

void Parser::operator()(UnitAST*& ast) { parse(ast); }

void Parser::parse(UnitAST*& ast) { parse_translation_unit(ast); }

void Parser::parse_warn(std::string message) {
  unit->warning(SourceLocation(cursor_), std::move(message));
}

void Parser::parse_warn(SourceLocation loc, std::string message) {
  unit->warning(loc, std::move(message));
}

void Parser::parse_error(std::string message) {
  if (lastErrorCursor_ == cursor_) return;
  lastErrorCursor_ = cursor_;
  unit->error(SourceLocation(cursor_), std::move(message));
}

auto Parser::parse_error(SourceLocation loc, std::string message) {
  unit->error(loc, std::move(message));
}

auto Parser::parse_id(const Identifier* id, SourceLocation& loc) -> bool {
  loc = {};
  if (!lookat(TokenKind::T_IDENTIFIER)) return false;
  if (unit->identifier(currentLocation()) != id) return false;
  loc = consumeToken();
  return true;
}

auto Parser::parse_nospace() -> bool {
  const auto& tk = unit->tokenAt(currentLocation());
  return !tk.leadingSpace() && !tk.startOfLine();
}

auto Parser::parse_greater_greater() -> bool {
  const auto saved = currentLocation();

  SourceLocation greaterLoc;
  SourceLocation secondGreaterLoc;

  if (match(TokenKind::T_GREATER, greaterLoc) && parse_nospace() &&
      match(TokenKind::T_GREATER, secondGreaterLoc)) {
    return true;
  }

  rewind(saved);

  return false;
}

auto Parser::parse_greater_greater_equal() -> bool {
  const auto saved = currentLocation();

  SourceLocation greaterLoc;
  SourceLocation secondGreaterLoc;
  SourceLocation equalLoc;

  if (match(TokenKind::T_GREATER, greaterLoc) && parse_nospace() &&
      match(TokenKind::T_GREATER, secondGreaterLoc) && parse_nospace() &&
      match(TokenKind::T_EQUAL, equalLoc)) {
    return true;
  }

  rewind(saved);

  return false;
}

auto Parser::parse_greater_equal() -> bool {
  const auto saved = currentLocation();

  SourceLocation greaterLoc;
  SourceLocation equalLoc;

  if (match(TokenKind::T_GREATER, greaterLoc) && parse_nospace() &&
      match(TokenKind::T_EQUAL, equalLoc)) {
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
  if (!moduleUnit_) return false;
  return match(TokenKind::T_EXPORT, loc);
}

auto Parser::parse_import_keyword(SourceLocation& loc) -> bool {
  if (!moduleUnit_) return false;
  if (match(TokenKind::T_IMPORT, loc)) return true;
  if (!parse_id(importId_, loc)) return false;
  unit->setTokenKind(loc, TokenKind::T_IMPORT);
  return true;
}

auto Parser::parse_module_keyword(SourceLocation& loc) -> bool {
  if (!moduleUnit_) return false;

  if (match(TokenKind::T_MODULE, loc)) return true;

  if (!parse_id(moduleId_, loc)) return false;

  unit->setTokenKind(loc, TokenKind::T_MODULE);
  return true;
}

auto Parser::parse_final(SourceLocation& loc) -> bool {
  return parse_id(finalId_, loc);
}

auto Parser::parse_override(SourceLocation& loc) -> bool {
  return parse_id(overrideId_, loc);
}

auto Parser::parse_type_name(NameAST*& yyast) -> bool {
  auto lookat_simple_template_id = [&] {
    LookaheadParser lookahead{this};
    if (!parse_simple_template_id(yyast)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_simple_template_id()) return true;

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

      const auto isTrue = lookat(TokenKind::T_TRUE);

      ast->literalLoc = consumeToken();
      ast->isTrue = isTrue;

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
      auto literalLoc = consumeToken();

      auto ast = new (pool) StringLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = literalLoc;
      ast->literal = unit->literal(literalLoc);

      return true;
    }

    default:
      return false;
  }  // switch
}

void Parser::parse_translation_unit(UnitAST*& yyast) {
  if (parse_module_unit(yyast)) return;
  parse_top_level_declaration_seq(yyast);
}

auto Parser::parse_module_head() -> bool {
  const auto start = currentLocation();

  SourceLocation exportLoc;

  match(TokenKind::T_EXPORT, exportLoc);

  SourceLocation moduleLoc;

  const auto is_module = parse_id(moduleId_, moduleLoc);

  rewind(start);

  return is_module;
}

auto Parser::parse_module_unit(UnitAST*& yyast) -> bool {
  moduleUnit_ = true;

  if (!parse_module_head()) return false;

  auto ast = new (pool) ModuleUnitAST();
  yyast = ast;

  parse_global_module_fragment(ast->globalModuleFragment);

  if (!parse_module_declaration(ast->moduleDeclaration)) {
    parse_error("expected a module declaration");
  }

  parse_declaration_seq(ast->declarationList);

  parse_private_module_fragment(ast->privateModuleFragment);

  SourceLocation eofLoc;

  expect(TokenKind::T_EOF_SYMBOL, eofLoc);

  return true;
}

void Parser::parse_top_level_declaration_seq(UnitAST*& yyast) {
  auto ast = new (pool) TranslationUnitAST();
  yyast = ast;

  moduleUnit_ = false;

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
}

void Parser::parse_skip_top_level_declaration(bool& skipping) {
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
}

void Parser::parse_declaration_seq(List<DeclarationAST*>*& yyast) {
  bool skipping = false;

  auto it = &yyast;

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

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
}

void Parser::parse_skip_declaration(bool& skipping) {
  if (lookat(TokenKind::T_RBRACE)) return;
  if (lookat(TokenKind::T_MODULE)) return;
  if (moduleUnit_ && lookat(TokenKind::T_EXPORT)) return;
  if (lookat(TokenKind::T_IMPORT)) return;
  if (!skipping) parse_error("expected a declaration");
  skipping = true;
}

auto Parser::parse_primary_expression(ExpressionAST*& yyast,
                                      bool inRequiresClause) -> bool {
  NameAST* name = nullptr;

  if (parse_this_expression(yyast)) {
    return true;
  } else if (parse_literal(yyast)) {
    return true;
  } else if (parse_lambda_expression(yyast)) {
    return true;
  } else if (parse_requires_expression(yyast)) {
    return true;
  } else if (lookat(TokenKind::T_LPAREN, TokenKind::T_RPAREN)) {
    return false;
  } else if (parse_fold_expression(yyast)) {
    return true;
  } else if (parse_nested_expession(yyast)) {
    return true;
  } else if (parse_id_expression(name, inRequiresClause)) {
    auto ast = new (pool) IdExpressionAST();
    yyast = ast;

    ast->name = name;

    return true;
  } else {
    return false;
  }
}

auto Parser::parse_id_expression(NameAST*& yyast, bool inRequiresClause)
    -> bool {
  auto lookat_qualified_id = [&] {
    LookaheadParser lookahead{this};
    if (!parse_qualified_id(yyast, inRequiresClause)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_qualified_id()) return true;

  LookaheadParser lookahead{this};

  if (!parse_unqualified_id(yyast, inRequiresClause)) return false;

  lookahead.commit();

  return true;
}

auto Parser::parse_maybe_template_id(NameAST*& yyast, bool inRequiresClause)
    -> bool {
  LookaheadParser lookahead{this};

  auto template_id = parse_template_id(yyast);

  if (!template_id) return false;

  if (inRequiresClause) {
    lookahead.commit();
    return true;
  }

  switch (TokenKind(LA())) {
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
      lookahead.commit();
      return true;

    default: {
      SourceLocation loc;
      TokenKind tk = TokenKind::T_EOF_SYMBOL;
      ExprContext ctx;
      if (parse_lookahead_binary_operator(loc, tk, ctx)) {
        lookahead.commit();
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

  if (lookat(TokenKind::T_OPERATOR)) {
    if (parse_literal_operator_id(yyast)) return true;
    if (parse_operator_function_id(yyast)) return true;

    rewind(start);

    LookaheadParser lookahead{this};
    if (!parse_conversion_function_id(yyast)) return false;
    lookahead.commit();
    return true;
  }

  return parse_name_id(yyast);
}

auto Parser::parse_qualified_id(NameAST*& yyast, bool inRequiresClause)
    -> bool {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  bool hasTemplate = false;
  NameAST* name = nullptr;
  bool hasName = false;

  auto lookat_qualified_id = [&] {
    LookaheadParser lookahead{this};
    if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;
    hasTemplate = match(TokenKind::T_TEMPLATE, templateLoc);
    hasName = parse_unqualified_id(name, inRequiresClause);
    if (!hasName && !templateLoc) return false;
    lookahead.commit();
    return true;
  };

  if (!lookat_qualified_id()) return false;

  auto ast = new (pool) QualifiedNameAST();
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->id = name;

  if (!hasName) parse_error("expected a template name");

  return true;
}

auto Parser::parse_start_of_nested_name_specifier(NameAST*& yyast,
                                                  SourceLocation& scopeLoc)
    -> bool {
  yyast = nullptr;

  if (match(TokenKind::T_COLON_COLON, scopeLoc)) return true;

  auto lookat_decltype_specifier = [&] {
    LookaheadParser lookahead{this};
    SpecifierAST* decltypeSpecifier = nullptr;
    if (!parse_decltype_specifier(decltypeSpecifier)) return false;
    if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;
    lookahead.commit();
    auto ast = new (pool) DecltypeNameAST();
    yyast = ast;
    ast->decltypeSpecifier = decltypeSpecifier;
    return true;
  };

  auto lookat_simple_template_id = [&] {
    LookaheadParser lookahead{this};
    NameAST* templateId = nullptr;
    if (!parse_simple_template_id(templateId)) return false;
    if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;
    lookahead.commit();
    yyast = templateId;
    return true;
  };

  if (lookat_decltype_specifier()) return true;
  if (lookat_simple_template_id()) return true;

  LookaheadParser lookahead{this};
  NameAST* nameId = nullptr;
  if (!parse_name_id(nameId)) return false;
  if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;
  lookahead.commit();
  yyast = nameId;
  return true;
}

void Parser::parse_optional_nested_name_specifier(
    NestedNameSpecifierAST*& yyast) {
  LookaheadParser lookahead(this);
  if (!parse_nested_name_specifier(yyast)) return;
  lookahead.commit();
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
    if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON_COLON)) {
      NameAST* name = nullptr;
      (void)parse_name_id(name);

      SourceLocation scopeLoc;

      expect(TokenKind::T_COLON_COLON, scopeLoc);

      *nameIt = new (pool) List(name);
      nameIt = &(*nameIt)->next;
    } else {
      auto lookat_template_id = [&] {
        LookaheadParser lookahead{this};
        SourceLocation templateLoc;
        const auto hasTemplate = match(TokenKind::T_TEMPLATE, templateLoc);
        NameAST* name = nullptr;
        if (!parse_simple_template_id(name)) return false;
        if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;
        lookahead.commit();
        *nameIt = new (pool) List(name);
        nameIt = &(*nameIt)->next;
        return true;
      };

      if (!lookat_template_id()) break;
    }
  }

  context.parsed = true;

  return true;
}

auto Parser::parse_lambda_expression(ExpressionAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_LBRACKET)) return false;

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

    (void)parse_requires_clause(ast->requiresClause);
  }

  if (!lookat(TokenKind::T_LBRACE)) {
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

  (void)parse_decl_specifier_seq(ast->declSpecifierList, specs);

  (void)parse_noexcept_specifier(ast->exceptionSpecifier);

  parse_optional_attribute_specifier_seq(ast->attributeList);

  (void)parse_trailing_return_type(ast->trailingReturnType);

  (void)parse_requires_clause(ast->requiresClause);

  return true;
}

auto Parser::parse_lambda_capture(SourceLocation& captureDefaultLoc,
                                  List<LambdaCaptureAST*>*& captureList)
    -> bool {
  if (parse_capture_default(captureDefaultLoc)) {
    SourceLocation commaLoc;

    if (match(TokenKind::T_COMMA, commaLoc)) {
      if (!parse_capture_list(captureList)) parse_error("expected a capture");
    }

    return true;
  }

  return parse_capture_list(captureList);
}

auto Parser::parse_capture_default(SourceLocation& opLoc) -> bool {
  const auto start = currentLocation();

  SourceLocation ampLoc;
  SourceLocation equalLoc;

  if (!match(TokenKind::T_AMP, ampLoc) && !match(TokenKind::T_EQUAL, equalLoc))
    return false;

  if (!lookat(TokenKind::T_COMMA) && !lookat(TokenKind::T_RBRACKET)) {
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

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
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

  if (lookat(TokenKind::T_STAR, TokenKind::T_THIS)) {
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

    ExpressionAST* initializer = nullptr;

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

  ExpressionAST* initializer = nullptr;

  if (!parse_initializer(initializer)) return false;

  auto ast = new (pool) InitLambdaCaptureAST();
  yyast = ast;

  ast->ellipsisLoc = ellipsisLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);
  ast->initializer = initializer;

  return true;
}

auto Parser::parse_left_fold_expression(ExpressionAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_LPAREN, TokenKind::T_DOT_DOT_DOT)) return false;

  auto ast = new (pool) LeftFoldExpressionAST();
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  expect(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  if (!parse_fold_operator(ast->opLoc, ast->op)) {
    parse_error("expected fold operator");
  }

  if (!parse_cast_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_this_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation thisLoc;

  if (!match(TokenKind::T_THIS, thisLoc)) return false;

  auto ast = new (pool) ThisExpressionAST();
  yyast = ast;
  ast->thisLoc = thisLoc;

  return true;
}

auto Parser::parse_nested_expession(ExpressionAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = new (pool) NestedExpressionAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;

  if (!parse_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_fold_expression(ExpressionAST*& yyast) -> bool {
  if (parse_left_fold_expression(yyast)) return true;

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;
  SourceLocation ellipsisLoc;

  auto lookat_fold_expression = [&] {
    LookaheadParser lookahead{this};
    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;
    if (!parse_cast_expression(expression)) return false;
    if (!parse_fold_operator(opLoc, op)) return false;
    if (!match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) return false;
    lookahead.commit();
    return true;
  };

  if (!lookat_fold_expression()) return false;

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

  if (!lookat(TokenKind::T_LBRACE)) {
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
    if (lookat(TokenKind::T_RBRACE)) break;

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

  parse_optional_nested_name_specifier(ast->nestedNameSpecifier);

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

  parse_expr_or_braced_init_list(ast->indexExpression);

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
  if (LA().isOneOf(TokenKind::T_CONST_CAST, TokenKind::T_DYNAMIC_CAST,
                   TokenKind::T_REINTERPRET_CAST, TokenKind::T_STATIC_CAST)) {
    castLoc = consumeToken();
    return true;
  }
  return false;
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
  if (lookat(TokenKind::T_LPAREN) &&
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

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
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
    rewind(opLoc);
    return false;
  }

  return true;
}

auto Parser::parse_complex_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation opLoc;

  if (!match(TokenKind::T___IMAG__, opLoc) &&
      !match(TokenKind::T___REAL__, opLoc)) {
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
  SourceLocation awaitLoc;

  if (!match(TokenKind::T_CO_AWAIT, awaitLoc)) return false;

  auto ast = new (pool) AwaitExpressionAST();
  yyast = ast;

  ast->awaitLoc = awaitLoc;

  if (!parse_cast_expression(ast->expression))
    parse_error("expected an expression");

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

  match(TokenKind::T_COLON_COLON, scopeLoc);

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

  NewPlacementAST* newPlacement = nullptr;

  if (!parse_new_placement(ast->newPlacement)) rewind(after_new_op);

  const auto after_new_placement = currentLocation();

  TypeIdAST* typeId = nullptr;

  SourceLocation lparenLoc;
  SourceLocation rparenLoc;

  if (match(TokenKind::T_LPAREN, lparenLoc) && parse_type_id(typeId) &&
      match(TokenKind::T_RPAREN, rparenLoc)) {
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

auto Parser::parse_new_placement(NewPlacementAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  List<ExpressionAST*>* expressionList = nullptr;

  if (!parse_expression_list(expressionList)) return false;

  SourceLocation rparenLoc;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

  auto ast = new (pool) NewPlacementAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_new_type_id(NewTypeIdAST*& yyast) -> bool {
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList)) return false;

  auto ast = new (pool) NewTypeIdAST();
  yyast = ast;

  ast->typeSpecifierList = typeSpecifierList;

  const auto saved = currentLocation();

  if (!parse_new_declarator(ast->newDeclarator)) rewind(saved);

  return true;
}

auto Parser::parse_new_declarator(NewDeclaratorAST*& yyast) -> bool {
  List<PtrOperatorAST*>* ptrOpList = nullptr;

  const auto hasPtrList = parse_ptr_operator_seq(ptrOpList);

  List<ArrayDeclaratorAST*>* modifiers = nullptr;

  const auto hasDeclChunks = parse_noptr_new_declarator(modifiers);

  if (!hasPtrList && !hasDeclChunks) return false;

  auto ast = new (pool) NewDeclaratorAST();
  yyast = ast;

  ast->ptrOpList = ptrOpList;
  ast->modifiers = modifiers;

  return true;
}

auto Parser::parse_noptr_new_declarator(List<ArrayDeclaratorAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  SourceLocation lbracketLoc;

  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  auto declarator = new (pool) ArrayDeclaratorAST();

  declarator->lbracketLoc = lbracketLoc;

  if (!match(TokenKind::T_RBRACKET, declarator->rbracketLoc)) {
    if (!parse_expression(declarator->expression))
      parse_error("expected an expression");

    expect(TokenKind::T_RBRACKET, declarator->rbracketLoc);
  }

  parse_optional_attribute_specifier_seq(declarator->attributeList);

  *it = new (pool) List(declarator);
  it = &(*it)->next;

  while (match(TokenKind::T_LBRACKET, lbracketLoc)) {
    auto declarator = new (pool) ArrayDeclaratorAST();
    *it = new (pool) List(declarator);
    it = &(*it)->next;

    declarator->lbracketLoc = lbracketLoc;

    if (!match(TokenKind::T_RBRACKET, declarator->rbracketLoc)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(declarator->expression)) {
        parse_error("expected an expression");
      }

      expect(TokenKind::T_RBRACKET, declarator->rbracketLoc);
    }

    parse_optional_attribute_specifier_seq(declarator->attributeList);
  }

  return true;
}

auto Parser::parse_new_initializer(NewInitializerAST*& yyast) -> bool {
  BracedInitListAST* bracedInitList = nullptr;

  if (parse_braced_init_list(bracedInitList)) {
    auto ast = new (pool) NewBracedInitializerAST();
    yyast = ast;

    ast->bracedInitList = bracedInitList;

    return true;
  }

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
  if (!lookat(TokenKind::T_DELETE) &&
      !lookat(TokenKind::T_COLON_COLON, TokenKind::T_DELETE))
    return false;

  auto ast = new (pool) DeleteExpressionAST();
  yyast = ast;

  match(TokenKind::T_COLON_COLON, ast->scopeLoc);
  expect(TokenKind::T_DELETE, ast->deleteLoc);

  if (match(TokenKind::T_LBRACKET, ast->lbracketLoc)) {
    expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  }

  if (!parse_cast_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  return true;
}

auto Parser::parse_cast_expression(ExpressionAST*& yyast) -> bool {
  auto lookat_cast_expression = [&] {
    LookaheadParser lookahead{this};
    if (!parse_cast_expression_helper(yyast)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_cast_expression()) return true;

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

  if (!parse_cast_expression(expression)) {
    return false;
  }

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
        if (exprContext.templArg && templArgDepth_ >= 2) {
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

  LookaheadParser lookahead{this};

  if (parse_binary_expression_helper(yyast, Prec::kLogicalOr, exprContext)) {
    lookahead.commit();
  }

  return true;
}

auto Parser::parse_lookahead_binary_operator(SourceLocation& loc, TokenKind& tk,
                                             const ExprContext& exprContext)
    -> bool {
  LookaheadParser lookahead{this};

  return parse_binary_operator(loc, tk, exprContext);
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

    (void)parse_binary_operator(opLoc, op, exprContext);

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
  SourceLocation yieldLoc;

  if (!match(TokenKind::T_CO_YIELD, yieldLoc)) return false;

  auto ast = new (pool) YieldExpressionAST();
  yyast = ast;

  ast->yieldLoc = yieldLoc;
  parse_expr_or_braced_init_list(ast->expression);

  return true;
}

auto Parser::parse_throw_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation throwLoc;

  if (!match(TokenKind::T_THROW, throwLoc)) return false;

  auto ast = new (pool) ThrowExpressionAST();
  yyast = ast;

  ast->throwLoc = throwLoc;

  LookaheadParser lookahead{this};

  if (parse_assignment_expression(ast->expression)) {
    lookahead.commit();
  }

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

void Parser::parse_statement(StatementAST*& yyast) {
  if (!parse_statement_helper(yyast)) {
    parse_error("expected a statement");
  }
}

auto Parser::parse_statement_helper(StatementAST*& yyast) -> bool {
  SourceLocation extensionLoc;

  match(TokenKind::T___EXTENSION__, extensionLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

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
    case TokenKind::T_FOR: {
      auto lookat_for_range_statement = [&] {
        LookaheadParser lookahead{this};
        if (!parse_for_range_statement(yyast)) return false;
        lookahead.commit();
        return true;
      };
      if (lookat_for_range_statement()) return true;
      return parse_for_statement(yyast);
    }
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
      if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON)) {
        return parse_labeled_statement(yyast);
      }

      auto lookat_declaration_statement = [&] {
        LookaheadParser lookahead{this};
        if (!parse_declaration_statement(yyast)) return false;
        lookahead.commit();
        return true;
      };

      if (lookat_declaration_statement()) return true;

      return parse_expression_statement(yyast);
  }  // switch
}

void Parser::parse_init_statement(StatementAST*& yyast) {
  auto lookat_simple_declaration = [&] {
    LookaheadParser lookahead{this};
    DeclarationAST* declaration = nullptr;
    if (!parse_simple_declaration(declaration, true)) return false;
    lookahead.commit();

    auto ast = new (pool) DeclarationStatementAST();
    yyast = ast;
    ast->declaration = declaration;
    return true;
  };

  if (lookat_simple_declaration()) return;

  auto lookat_expression = [&] {
    LookaheadParser lookahead{this};

    ExpressionAST* expression = nullptr;
    if (!parse_expression(expression)) return false;

    SourceLocation semicolonLoc;
    if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

    lookahead.commit();

    auto ast = new (pool) ExpressionStatementAST();
    yyast = ast;
    ast->expression = expression;
    ast->semicolonLoc = semicolonLoc;
    return true;
  };

  if (lookat_expression()) return;
}

auto Parser::parse_condition(ExpressionAST*& yyast) -> bool {
  auto lookat_condition = [&] {
    LookaheadParser lookahead{this};

    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_optional_attribute_specifier_seq(attributes);

    List<SpecifierAST*>* declSpecifierList = nullptr;

    DeclSpecs specs;

    if (!parse_decl_specifier_seq(declSpecifierList, specs)) return false;

    DeclaratorAST* declarator = nullptr;

    if (!parse_declarator(declarator)) return false;

    ExpressionAST* initializer = nullptr;

    if (!parse_brace_or_equal_initializer(initializer)) return false;

    lookahead.commit();

    auto ast = new (pool) ConditionExpressionAST();
    yyast = ast;
    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->declarator = declarator;
    ast->initializer = initializer;

    return true;
  };

  if (lookat_condition()) return true;

  if (!parse_expression(yyast)) return false;

  return true;
}

auto Parser::parse_labeled_statement(StatementAST*& yyast) -> bool {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  StatementAST* statement = nullptr;

  parse_statement(statement);

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

  parse_statement(statement);

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

  if (!match(TokenKind::T_DEFAULT, defaultLoc)) return false;

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  StatementAST* statement = nullptr;

  parse_statement(statement);

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
    if (lookat(TokenKind::T_RBRACE)) break;

    StatementAST* statement = nullptr;

    if (parse_statement_helper(statement)) {
      *it = new (pool) List(statement);
      it = &(*it)->next;
      skipping = false;
    } else {
      parse_skip_statement(skipping);
    }
  }
}

void Parser::parse_skip_statement(bool& skipping) {
  if (!LA()) return;
  if (lookat(TokenKind::T_RBRACE)) return;
  if (!skipping) parse_error("expected a statement");
  for (; LA(); consumeToken()) {
    if (lookat(TokenKind::T_SEMICOLON)) break;
    if (lookat(TokenKind::T_LBRACE)) break;
    if (lookat(TokenKind::T_RBRACE)) break;
  }
  skipping = true;
}

auto Parser::parse_if_statement(StatementAST*& yyast) -> bool {
  SourceLocation ifLoc;

  if (!match(TokenKind::T_IF, ifLoc)) return false;

  auto ast = new (pool) IfStatementAST();
  yyast = ast;

  ast->ifLoc = ifLoc;

  match(TokenKind::T_CONSTEXPR, ast->constexprLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_init_statement(ast->initializer);

  if (!parse_condition(ast->condition)) parse_error("expected a condition");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  parse_statement(ast->statement);

  if (!match(TokenKind::T_ELSE, ast->elseLoc)) return true;

  parse_statement(ast->elseStatement);

  return true;
}

auto Parser::parse_switch_statement(StatementAST*& yyast) -> bool {
  SourceLocation switchLoc;

  if (!match(TokenKind::T_SWITCH, switchLoc)) return false;

  auto ast = new (pool) SwitchStatementAST();
  yyast = ast;

  ast->switchLoc = switchLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_init_statement(ast->initializer);

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

  parse_statement(ast->statement);

  return true;
}

auto Parser::parse_do_statement(StatementAST*& yyast) -> bool {
  SourceLocation doLoc;

  if (!match(TokenKind::T_DO, doLoc)) return false;

  auto ast = new (pool) DoStatementAST();
  yyast = ast;

  ast->doLoc = doLoc;

  parse_statement(ast->statement);

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

  StatementAST* initializer = nullptr;

  parse_init_statement(initializer);

  DeclarationAST* rangeDeclaration = nullptr;

  if (!parse_for_range_declaration(rangeDeclaration)) return false;

  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  auto ast = new (pool) ForRangeStatementAST();
  yyast = ast;

  ast->forLoc = forLoc;
  ast->rangeDeclaration = rangeDeclaration;
  ast->lparenLoc = lparenLoc;
  ast->initializer = initializer;
  ast->colonLoc = colonLoc;

  parse_for_range_initializer(ast->rangeInitializer);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  parse_statement(ast->statement);

  return true;
}

auto Parser::parse_for_statement(StatementAST*& yyast) -> bool {
  SourceLocation forLoc;

  if (!match(TokenKind::T_FOR, forLoc)) return false;

  auto ast = new (pool) ForStatementAST();
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_init_statement(ast->initializer);

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

  parse_statement(ast->statement);

  return true;
}

auto Parser::parse_for_range_declaration(DeclarationAST*& yyast) -> bool {
  List<AttributeSpecifierAST*>* attributeList = nullptr;

  parse_optional_attribute_specifier_seq(attributeList);

  List<SpecifierAST*>* declSpecifierList = nullptr;

  DeclSpecs specs;

  if (!parse_decl_specifier_seq(declSpecifierList, specs)) return false;

  const auto& tk = LA();

  if (tk.is(TokenKind::T_LBRACKET) ||
      ((tk.is(TokenKind::T_AMP) || tk.is(TokenKind::T_AMP_AMP)) &&
       LA(1).is(TokenKind::T_LBRACKET))) {
    SourceLocation refLoc;

    (void)parse_ref_qualifier(refLoc);

    SourceLocation lbracketLoc;

    if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

    List<NameAST*>* bindings = nullptr;

    if (!parse_identifier_list(bindings)) parse_error("expected an identifier");

    SourceLocation rbracketLoc;

    expect(TokenKind::T_RBRACKET, rbracketLoc);

    auto ast = new (pool) StructuredBindingDeclarationAST();
    yyast = ast;
    ast->attributeList = attributeList;
    ast->declSpecifierList = declSpecifierList;
    ast->refQualifierLoc = refLoc;
    ast->lbracketLoc = lbracketLoc;
    ast->bindingList = bindings;
    ast->rbracketLoc = rbracketLoc;
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

void Parser::parse_for_range_initializer(ExpressionAST*& yyast) {
  parse_expr_or_braced_init_list(yyast);
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
    parse_expr_or_braced_init_list(ast->expression);

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
    parse_expr_or_braced_init_list(ast->expression);

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
  if (!moduleUnit_) return false;

  const auto start = currentLocation();

  SourceLocation exportLoc;

  match(TokenKind::T_EXPORT, exportLoc);

  SourceLocation moduleLoc;

  const auto is_module = parse_module_keyword(moduleLoc);

  rewind(start);

  return is_module;
}

auto Parser::parse_declaration(DeclarationAST*& yyast) -> bool {
  if (lookat(TokenKind::T_RBRACE))
    return false;
  else if (lookat(TokenKind::T_SEMICOLON))
    return parse_empty_declaration(yyast);
  else if (parse_explicit_instantiation(yyast))
    return true;
  else if (parse_explicit_specialization(yyast))
    return true;
  else if (parse_template_declaration(yyast))
    return true;
  else if (parse_linkage_specification(yyast))
    return true;
  else if (parse_namespace_definition(yyast))
    return true;
  else if (parse_deduction_guide(yyast))
    return true;
  else if (parse_export_declaration(yyast))
    return true;
  else if (parse_module_import_declaration(yyast))
    return true;
  else if (parse_attribute_declaration(yyast))
    return true;
  else
    return parse_block_declaration(yyast, true);
}

auto Parser::parse_block_declaration(DeclarationAST*& yyast, bool fundef)
    -> bool {
  if (parse_asm_declaration(yyast))
    return true;
  else if (parse_namespace_alias_definition(yyast))
    return true;
  else if (parse_static_assert_declaration(yyast))
    return true;
  else if (parse_opaque_enum_declaration(yyast))
    return true;
  else if (parse_using_enum_declaration(yyast))
    return true;
  else if (parse_using_directive(yyast))
    return true;
  else if (parse_alias_declaration(yyast))
    return true;
  else if (parse_using_declaration(yyast))
    return true;
  else
    return parse_simple_declaration(yyast, fundef);
}

auto Parser::parse_alias_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation usingLoc;
  SourceLocation identifierLoc;
  List<AttributeSpecifierAST*>* attributes = nullptr;
  SourceLocation equalLoc;

  auto lookat_alias_declaration = [&] {
    LookaheadParser lookhead{this};

    if (!match(TokenKind::T_USING, usingLoc)) return false;

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    parse_optional_attribute_specifier_seq(attributes);

    if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

    lookhead.commit();

    return true;
  };

  if (!lookat_alias_declaration()) return false;

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

void Parser::enterFunctionScope(FunctionDeclaratorAST* functionDeclarator) {}

auto Parser::parse_simple_declaration(DeclarationAST*& yyast,
                                      bool acceptFunctionDefinition) -> bool {
  SourceLocation extensionLoc;

  match(TokenKind::T___EXTENSION__, extensionLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_optional_attribute_specifier_seq(attributes);

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

  (void)parse_ref_qualifier(refLoc);

  SourceLocation lbracketLoc;

  if (match(TokenKind::T_LBRACKET, lbracketLoc)) {
    List<NameAST*>* bindings = nullptr;
    SourceLocation rbracketLoc;

    if (parse_identifier_list(bindings) &&
        match(TokenKind::T_RBRACKET, rbracketLoc)) {
      ExpressionAST* initializer = nullptr;
      SourceLocation semicolonLoc;

      if (parse_initializer(initializer) &&
          match(TokenKind::T_SEMICOLON, semicolonLoc)) {
        auto ast = new (pool) StructuredBindingDeclarationAST();
        yyast = ast;

        ast->attributeList = attributes;
        ast->declSpecifierList = declSpecifierList;
        ast->refQualifierLoc = refLoc;
        ast->lbracketLoc = lbracketLoc;
        ast->bindingList = bindings;
        ast->rbracketLoc = rbracketLoc;
        ast->initializer = initializer;
        ast->semicolonLoc = semicolonLoc;

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

  (void)parse_requires_clause(requiresClause);

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

    if (classDepth_) pendingFunctionDefinitions_.push_back(ast);

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

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    InitDeclaratorAST* initDeclarator = nullptr;

    if (!parse_init_declarator(initDeclarator, specs)) return false;

    *declIt = new (pool) List(initDeclarator);
    declIt = &(*declIt)->next;
  }

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

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
  CoreDeclaratorAST* declaratorId = nullptr;

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

  if (!has_requires_clause) parse_virt_specifier_seq(functionDeclarator);

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

  if (classDepth_) pendingFunctionDefinitions_.push_back(ast);

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
    expect(TokenKind::T_STRING_LITERAL, ast->literalLoc);
    ast->literal = unit->literal(ast->literalLoc);
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
  SourceLocation semicolonLoc;

  auto lookat_attribute_declaration = [&] {
    LookaheadParser lookahead{this};
    if (!parse_attribute_specifier_seq(attributes)) return false;

    if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

    lookahead.commit();
    return true;
  };

  if (!lookat_attribute_declaration()) return false;

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

  parse_optional_attribute_specifier_seq(attributes);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_optional_attribute_specifier_seq(attributes);

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

  parse_optional_attribute_specifier_seq(attributes);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_optional_attribute_specifier_seq(attributes);

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

  parse_optional_attribute_specifier_seq(attributes);

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

    parse_optional_attribute_specifier_seq(attributes);

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

  parse_optional_attribute_specifier_seq(attributes);

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

    parse_optional_attribute_specifier_seq(attributes);

    *it = new (pool) List(typeSpecifier);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_simple_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (parse_primitive_type_specifier(yyast, specs)) return true;
  if (parse_underlying_type_specifier(yyast, specs)) return true;
  if (parse_atomic_type_specifier(yyast, specs)) return true;
  if (parse_placeholder_type_specifier_helper(yyast, specs)) return true;
  if (parse_decltype_specifier_type_specifier(yyast, specs)) return true;
  if (parse_named_type_specifier(yyast, specs)) return true;

  return false;
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

  SourceLocation underlyingTypeLoc;

  if (!match(TokenKind::T___UNDERLYING_TYPE, underlyingTypeLoc)) return false;

  specs.has_named_typespec = true;

  auto ast = new (pool) UnderlyingTypeSpecifierAST();
  yyast = ast;

  ast->underlyingTypeLoc = underlyingTypeLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected type id");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_atomic_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (!specs.accepts_simple_typespec()) return false;

  SourceLocation atomicLoc;

  if (!match(TokenKind::T__ATOMIC, atomicLoc)) return false;

  auto ast = new (pool) AtomicTypeSpecifierAST();
  yyast = ast;

  ast->atomicLoc = atomicLoc;

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
  if (!LA().isOneOf(TokenKind::T_ENUM, TokenKind::T_CLASS, TokenKind::T_STRUCT,
                    TokenKind::T_UNION))
    return false;

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

  if (lookat(TokenKind::T_ENUM)) {
    return parse_elaborated_enum_specifier(yyast, specs);
  }

  SourceLocation classLoc;

  if (!parse_class_key(classLoc)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_optional_attribute_specifier_seq(attributes);

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
      ast->classKey = unit->tokenKind(classLoc);

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
    ast->classKey = unit->tokenKind(classLoc);

    return true;
  }

  const auto after_nested_name_specifier = currentLocation();

  SourceLocation templateLoc;

  const bool has_template = match(TokenKind::T_TEMPLATE, templateLoc);

  NameAST* name = nullptr;

  if (parse_simple_template_id(name)) {
    specs.has_complex_typespec = true;

    auto ast = new (pool) ElaboratedTypeSpecifierAST();
    yyast = ast;

    ast->classLoc = classLoc;
    ast->attributeList = attributes;
    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->name = name;
    ast->classKey = unit->tokenKind(classLoc);

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
    ast->classKey = unit->tokenKind(classLoc);

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
  ast->classKey = unit->tokenKind(classLoc);

  return true;
}

auto Parser::parse_elaborated_enum_specifier(ElaboratedTypeSpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  SourceLocation enumLoc;

  if (!match(TokenKind::T_ENUM, enumLoc)) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  parse_optional_nested_name_specifier(nestedNameSpecifier);

  NameAST* name = nullptr;

  if (!parse_name_id(name)) return false;

  specs.has_complex_typespec = true;

  auto ast = new (pool) ElaboratedTypeSpecifierAST();
  yyast = ast;

  ast->classLoc = enumLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;
  ast->classKey = TokenKind::T_ENUM;

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

    if (lookat(TokenKind::T_AUTO)) return false;  // placeholder type specifier

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

  auto lookat_placeholder_type_specifier = [&] {
    LookaheadParser lookahead{this};

    (void)parse_type_constraint(typeConstraint, /*parsing placeholder=*/true);

    if (!lookat(TokenKind::T_AUTO) &&
        !lookat(TokenKind::T_DECLTYPE, TokenKind::T_LPAREN, TokenKind::T_AUTO))
      return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_placeholder_type_specifier()) return false;

  SourceLocation autoLoc;

  if (match(TokenKind::T_AUTO, autoLoc)) {
    auto ast = new (pool) AutoTypeSpecifierAST();
    yyast = ast;
    ast->autoLoc = autoLoc;
  } else {
    auto ast = new (pool) DecltypeAutoSpecifierAST();
    yyast = ast;

    expect(TokenKind::T_DECLTYPE, ast->decltypeLoc);
    expect(TokenKind::T_LPAREN, ast->lparenLoc);
    expect(TokenKind::T_AUTO, ast->autoLoc);
    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  if (typeConstraint) {
    auto ast = new (pool) PlaceholderTypeSpecifierAST();
    ast->typeConstraint = typeConstraint;
    ast->specifier = yyast;
    yyast = ast;
  }

  return true;
}

auto Parser::parse_init_declarator(InitDeclaratorAST*& yyast,
                                   const DeclSpecs& specs) -> bool {
  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  const auto saved = currentLocation();

  RequiresClauseAST* requiresClause = nullptr;
  ExpressionAST* initializer = nullptr;

  if (!parse_declarator_initializer(requiresClause, initializer)) rewind(saved);

  auto ast = new (pool) InitDeclaratorAST();
  yyast = ast;

  ast->declarator = declarator;
  ast->requiresClause = requiresClause;
  ast->initializer = initializer;

  return true;
}

auto Parser::parse_declarator_initializer(RequiresClauseAST*& requiresClause,
                                          ExpressionAST*& yyast) -> bool {
  if (parse_requires_clause(requiresClause)) return true;

  return parse_initializer(yyast);
}

void Parser::parse_optional_declarator_or_abstract_declarator(
    DeclaratorAST*& yyast) {
  auto lookat_declarator = [&] {
    LookaheadParser lookahead{this};
    if (!parse_declarator(yyast)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_declarator()) return;

  auto lookat_abstract_declarator = [&] {
    LookaheadParser lookahead{this};
    if (!parse_abstract_declarator(yyast)) return;
    lookahead.commit();
  };

  lookat_abstract_declarator();
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
  if (parse_declarator_id(yyast)) return true;

  LookaheadParser lookahead{this};

  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  SourceLocation rparenLoc;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

  lookahead.commit();

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

      parse_optional_attribute_specifier_seq(attributes);

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

      (void)parse_trailing_return_type(modifier->trailingReturnType);

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

  (void)parse_cv_qualifier_seq(ast->cvQualifierList);

  (void)parse_ref_qualifier(ast->refLoc);

  (void)parse_noexcept_specifier(ast->exceptionSpecifier);

  parse_optional_attribute_specifier_seq(ast->attributeList);

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

    parse_optional_attribute_specifier_seq(ast->attributeList);

    (void)parse_cv_qualifier_seq(ast->cvQualifierList);

    return true;
  }

  SourceLocation refLoc;

  if (parse_ref_qualifier(refLoc)) {
    auto ast = new (pool) ReferenceOperatorAST();
    yyast = ast;

    ast->refLoc = refLoc;
    ast->refOp = unit->tokenKind(refLoc);

    parse_optional_attribute_specifier_seq(ast->attributeList);

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

    parse_optional_attribute_specifier_seq(ast->attributeList);

    (void)parse_cv_qualifier_seq(ast->cvQualifierList);

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
  if (match(TokenKind::T_AMP, refLoc)) return true;
  if (match(TokenKind::T_AMP_AMP, refLoc)) return true;
  return false;
}

auto Parser::parse_declarator_id(CoreDeclaratorAST*& yyast) -> bool {
  SourceLocation ellipsisLoc;

  const auto isPack = match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  check_type_traits();

  NameAST* name = nullptr;

  if (!parse_id_expression(name)) return false;

  auto ast = new (pool) IdDeclaratorAST();
  yyast = ast;

  ast->name = name;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  if (isPack) {
    auto ast = new (pool) ParameterPackAST();
    ast->ellipsisLoc = ellipsisLoc;
    ast->coreDeclarator = yyast;
    yyast = ast;
  }

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
  if (parse_abstract_pack_declarator(yyast)) return true;

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

  SourceLocation lparenLoc;
  SourceLocation rparenLoc;

  if (match(TokenKind::T_LPAREN, lparenLoc) &&
      parse_ptr_abstract_declarator(declarator) &&
      match(TokenKind::T_RPAREN, rparenLoc)) {
    auto nestedDeclarator = new (pool) NestedDeclaratorAST();
    nestedDeclarator->lparenLoc = lparenLoc;
    nestedDeclarator->declarator = declarator;
    nestedDeclarator->rparenLoc = rparenLoc;

    yyast->coreDeclarator = nestedDeclarator;
  } else {
    rewind(start);
  }

  const auto after_nested_declarator = currentLocation();

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  auto it = &yyast->modifiers;

  if (lookat(TokenKind::T_LPAREN)) {
    if (parse_parameters_and_qualifiers(parametersAndQualifiers)) {
      auto functionDeclarator = new (pool) FunctionDeclaratorAST();

      functionDeclarator->parametersAndQualifiers = parametersAndQualifiers;

      *it = new (pool) List<DeclaratorModifierAST*>(functionDeclarator);
      it = &(*it)->next;
    } else {
      rewind(after_nested_declarator);
    }
  }

  if (lookat(TokenKind::T_LBRACKET)) {
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

auto Parser::parse_abstract_pack_declarator(DeclaratorAST*& yyast) -> bool {
  auto start = currentLocation();

  List<PtrOperatorAST*>* ptrOpList = nullptr;

  (void)parse_ptr_operator_seq(ptrOpList);

  if (!parse_noptr_abstract_pack_declarator(yyast, ptrOpList)) {
    rewind(start);
    return false;
  }

  return true;
}

auto Parser::parse_noptr_abstract_pack_declarator(
    DeclaratorAST*& yyast, List<PtrOperatorAST*>* ptrOpLst) -> bool {
  SourceLocation ellipsisLoc;

  if (!match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) return false;

  auto ast = new (pool) DeclaratorAST();
  yyast = ast;

  auto coreDeclarator = new (pool) ParameterPackAST();
  coreDeclarator->ellipsisLoc = ellipsisLoc;

  ast->coreDeclarator = coreDeclarator;
  ast->ptrOpList = ptrOpLst;

  auto it = &yyast->modifiers;

  ParametersAndQualifiersAST* parametersAndQualifiers = nullptr;

  if (parse_parameters_and_qualifiers(parametersAndQualifiers)) {
    auto functionDeclarator = new (pool) FunctionDeclaratorAST();
    *it = new (pool) List<DeclaratorModifierAST*>(functionDeclarator);
    return true;
  }

  SourceLocation lbracketLoc;

  while (match(TokenKind::T_LBRACKET, lbracketLoc)) {
    auto arrayDeclarator = new (pool) ArrayDeclaratorAST();

    *it = new (pool) List<DeclaratorModifierAST*>(arrayDeclarator);
    it = &(*it)->next;

    arrayDeclarator->lbracketLoc = lbracketLoc;

    if (!match(TokenKind::T_RBRACKET, arrayDeclarator->rbracketLoc)) {
      if (!parse_constant_expression(arrayDeclarator->expression)) {
        parse_error("expected a constant expression");
      }

      expect(TokenKind::T_RBRACKET, arrayDeclarator->rbracketLoc);

      parse_optional_attribute_specifier_seq(arrayDeclarator->attributeList);
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
    ast->isVariadic = true;

    return true;
  }

  auto ast = new (pool) ParameterDeclarationClauseAST();
  yyast = ast;

  if (!parse_parameter_declaration_list(ast->parameterDeclarationList)) {
    return false;
  }

  match(TokenKind::T_COMMA, ast->commaLoc);

  ast->isVariadic = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

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

  parse_optional_attribute_specifier_seq(ast->attributeList);

  DeclSpecs specs;

  specs.no_class_or_enum_specs = true;

  if (!parse_decl_specifier_seq(ast->typeSpecifierList, specs)) return false;

  parse_optional_declarator_or_abstract_declarator(ast->declarator);

  if (match(TokenKind::T_EQUAL, ast->equalLoc)) {
    if (!parse_initializer_clause(ast->expression, templParam)) {
      if (templParam) return false;

      parse_error("expected an initializer");
    }
  }

  return true;
}

auto Parser::parse_initializer(ExpressionAST*& yyast) -> bool {
  SourceLocation lparenLoc;

  if (match(TokenKind::T_LPAREN, lparenLoc)) {
    if (lookat(TokenKind::T_RPAREN)) return false;

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

auto Parser::parse_brace_or_equal_initializer(ExpressionAST*& yyast) -> bool {
  BracedInitListAST* bracedInitList = nullptr;

  if (lookat(TokenKind::T_LBRACE)) {
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
  if (parse_braced_init_list(bracedInitList)) {
    yyast = bracedInitList;
    return true;
  }

  ExprContext exprContext;
  exprContext.templParam = templParam;

  if (!parse_assignment_expression(yyast, exprContext)) return false;

  return true;
}

auto Parser::parse_braced_init_list(BracedInitListAST*& yyast) -> bool {
  SourceLocation lbraceLoc;
  SourceLocation rbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  if (lookat(TokenKind::T_DOT)) {
    auto ast = new (pool) BracedInitListAST();
    yyast = ast;

    ast->lbraceLoc = lbraceLoc;

    auto it = &ast->expressionList;

    DesignatedInitializerClauseAST* designatedInitializerClause = nullptr;

    if (!parse_designated_initializer_clause(designatedInitializerClause)) {
      parse_error("expected designated initializer clause");
    }

    if (designatedInitializerClause) {
      *it = new (pool) List<ExpressionAST*>(designatedInitializerClause);
      it = &(*it)->next;
    }

    SourceLocation commaLoc;

    while (match(TokenKind::T_COMMA, commaLoc)) {
      if (lookat(TokenKind::T_RBRACE)) break;

      DesignatedInitializerClauseAST* designatedInitializerClause = nullptr;

      if (!parse_designated_initializer_clause(designatedInitializerClause)) {
        parse_error("expected designated initializer clause");
      }

      if (designatedInitializerClause) {
        *it = new (pool) List<ExpressionAST*>(designatedInitializerClause);
        it = &(*it)->next;
      }
    }

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);

    return true;
  }

  SourceLocation commaLoc;

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

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto pack = new (pool) PackExpansionExpressionAST();
    pack->expression = expression;
    pack->ellipsisLoc = ellipsisLoc;
    expression = pack;
  }

  *it = new (pool) List(expression);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (lookat(TokenKind::T_RBRACE)) break;

    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression)) {
      parse_error("expected initializer clause");
    }

    SourceLocation ellipsisLoc;

    if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
      auto pack = new (pool) PackExpansionExpressionAST();
      pack->expression = expression;
      pack->ellipsisLoc = ellipsisLoc;
      expression = pack;
    }

    *it = new (pool) List(expression);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_designated_initializer_clause(
    DesignatedInitializerClauseAST*& yyast) -> bool {
  DesignatorAST* designator = nullptr;

  if (!parse_designator(designator)) return false;

  auto ast = new (pool) DesignatedInitializerClauseAST();
  yyast = ast;

  ast->designator = designator;

  if (!parse_brace_or_equal_initializer(ast->initializer)) {
    parse_error("expected an initializer");
  }

  return true;
}

auto Parser::parse_designator(DesignatorAST*& yyast) -> bool {
  SourceLocation dotLoc;

  if (!match(TokenKind::T_DOT, dotLoc)) return false;

  auto ast = new (pool) DesignatorAST();
  yyast = ast;

  ast->dotLoc = dotLoc;

  SourceLocation identifierLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

void Parser::parse_expr_or_braced_init_list(ExpressionAST*& yyast) {
  BracedInitListAST* bracedInitList = nullptr;

  if (parse_braced_init_list(bracedInitList)) {
    yyast = bracedInitList;
  } else if (!parse_expression(yyast)) {
    parse_error("expected an expression");
  }
}

void Parser::parse_virt_specifier_seq(
    FunctionDeclaratorAST* functionDeclarator) {
  while (parse_virt_specifier(functionDeclarator)) {
    //
  }
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
  if (lookat(TokenKind::T_SEMICOLON)) return false;

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

  (void)parse_ctor_initializer(ctorInitializer);

  if (!lookat(TokenKind::T_LBRACE)) return false;

  auto ast = new (pool) CompoundStatementFunctionBodyAST();
  yyast = ast;

  ast->ctorInitializer = ctorInitializer;

  const bool skip = skipFunctionBody_ || classDepth_ > 0;

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

  parse_optional_attribute_specifier_seq(attributes);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  (void)parse_enum_head_name(nestedNameSpecifier, name);

  EnumBaseAST* enumBase = nullptr;

  (void)parse_enum_base(enumBase);

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

auto Parser::parse_enum_head_name(NestedNameSpecifierAST*& nestedNameSpecifier,
                                  NameAST*& name) -> bool {
  parse_optional_nested_name_specifier(nestedNameSpecifier);

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
  List<AttributeSpecifierAST*>* attributes = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;
  EnumBaseAST* enumBase = nullptr;
  SourceLocation semicolonLoc;

  auto lookat_opaque_enum_declaration = [&] {
    LookaheadParser lookahead{this};
    parse_optional_attribute_specifier_seq(attributes);
    if (!parse_enum_key(enumLoc, classLoc)) return false;
    if (!parse_enum_head_name(nestedNameSpecifier, name)) return false;
    (void)parse_enum_base(enumBase);
    if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;
    lookahead.commit();
    return true;
  };

  if (!lookat_opaque_enum_declaration()) return false;

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

void Parser::parse_enumerator_list(List<EnumeratorAST*>*& yyast) {
  auto it = &yyast;

  EnumeratorAST* enumerator = nullptr;

  parse_enumerator_definition(enumerator);

  *it = new (pool) List(enumerator);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (lookat(TokenKind::T_RBRACE)) {
      rewind(commaLoc);
      break;
    }

    EnumeratorAST* enumerator = nullptr;

    parse_enumerator_definition(enumerator);

    *it = new (pool) List(enumerator);
    it = &(*it)->next;
  }
}

void Parser::parse_enumerator_definition(EnumeratorAST*& yyast) {
  parse_enumerator(yyast);

  if (match(TokenKind::T_EQUAL, yyast->equalLoc)) {
    if (!parse_constant_expression(yyast->expression)) {
      parse_error("expected an expression");
    }
  }
}

void Parser::parse_enumerator(EnumeratorAST*& yyast) {
  auto ast = new (pool) EnumeratorAST();
  yyast = ast;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  parse_optional_attribute_specifier_seq(ast->attributeList);
}

auto Parser::parse_using_enum_declaration(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_USING, TokenKind::T_ENUM)) return false;

  auto ast = new (pool) UsingEnumDeclarationAST();
  yyast = ast;

  expect(TokenKind::T_USING, ast->usingLoc);

  DeclSpecs specs;

  if (!parse_elaborated_enum_specifier(ast->enumTypeSpecifier, specs)) {
    parse_error("expected an elaborated enum specifier");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_namespace_definition(DeclarationAST*& yyast) -> bool {
  if (lookat(TokenKind::T_NAMESPACE, TokenKind::T_IDENTIFIER,
             TokenKind::T_EQUAL)) {
    // skip namespace alias definition
    return false;
  }

  if (!lookat(TokenKind::T_NAMESPACE) &&
      !lookat(TokenKind::T_INLINE, TokenKind::T_NAMESPACE)) {
    return false;
  }

  auto ast = new (pool) NamespaceDefinitionAST();
  yyast = ast;

  ast->isInline = match(TokenKind::T_INLINE, ast->inlineLoc);

  expect(TokenKind::T_NAMESPACE, ast->namespaceLoc);

  parse_optional_attribute_specifier_seq(ast->attributeList);

  if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON_COLON)) {
    auto it = &ast->nestedNamespaceSpecifierList;

    auto name = new (pool) NestedNamespaceSpecifierAST();

    expect(TokenKind::T_IDENTIFIER, name->identifierLoc);
    expect(TokenKind::T_COLON_COLON, name->scopeLoc);

    name->namespaceName = unit->identifier(name->identifierLoc);

    *it = new (pool) List(name);
    it = &(*it)->next;

    while (true) {
      const auto saved = currentLocation();

      SourceLocation inlineLoc;

      auto isInline = match(TokenKind::T_INLINE, inlineLoc);

      SourceLocation identifierLoc;

      if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) {
        rewind(saved);
        break;
      }

      SourceLocation scopeLoc;

      if (!match(TokenKind::T_COLON_COLON, scopeLoc)) {
        rewind(saved);
        break;
      }

      auto name = new (pool) NestedNamespaceSpecifierAST();
      name->inlineLoc = inlineLoc;
      name->identifierLoc = identifierLoc;
      name->scopeLoc = scopeLoc;
      name->namespaceName = unit->identifier(name->identifierLoc);
      name->isInline = isInline;

      *it = new (pool) List(name);
      it = &(*it)->next;
    }
  }

  if (ast->nestedNamespaceSpecifierList) {
    ast->isInline = match(TokenKind::T_INLINE, ast->inlineLoc);
    expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  } else {
    match(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  }

  ast->namespaceName = unit->identifier(ast->identifierLoc);

  parse_optional_attribute_specifier_seq(ast->extraAttributeList);

  expect(TokenKind::T_LBRACE, ast->lbraceLoc);

  parse_namespace_body(ast);

  expect(TokenKind::T_RBRACE, ast->rbraceLoc);

  return true;
}

void Parser::parse_namespace_body(NamespaceDefinitionAST* yyast) {
  bool skipping = false;

  auto it = &yyast->declarationList;

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

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
}

auto Parser::parse_namespace_alias_definition(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_NAMESPACE, TokenKind::T_IDENTIFIER,
              TokenKind::T_EQUAL)) {
    return false;
  }

  auto ast = new (pool) NamespaceAliasDefinitionAST();
  yyast = ast;

  expect(TokenKind::T_NAMESPACE, ast->namespaceLoc);
  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  expect(TokenKind::T_EQUAL, ast->equalLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  if (!parse_qualified_namespace_specifier(ast->nestedNameSpecifier,
                                           ast->name)) {
    parse_error("expected a namespace name");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_qualified_namespace_specifier(
    NestedNameSpecifierAST*& nestedNameSpecifier, NameAST*& name) -> bool {
  parse_optional_nested_name_specifier(nestedNameSpecifier);

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
  SourceLocation usingLoc;
  SourceLocation namespaceLoc;

  auto lookat_using_directive = [&] {
    LookaheadParser lookahead{this};

    parse_optional_attribute_specifier_seq(attributes);

    if (!match(TokenKind::T_USING, usingLoc)) return false;

    if (!match(TokenKind::T_NAMESPACE, namespaceLoc)) return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_using_directive()) return false;

  auto ast = new (pool) UsingDirectiveAST;
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->namespaceLoc = namespaceLoc;

  parse_optional_nested_name_specifier(ast->nestedNameSpecifier);

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

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  *it = new (pool) List(declarator);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    UsingDeclaratorAST* declarator = nullptr;

    if (!parse_using_declarator(declarator)) {
      parse_error("expected a using declarator");
    }

    SourceLocation ellipsisLoc;

    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    *it = new (pool) List(declarator);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_using_declarator(UsingDeclaratorAST*& yyast) -> bool {
  SourceLocation typenameLoc;

  match(TokenKind::T_TYPENAME, typenameLoc);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  parse_optional_nested_name_specifier(nestedNameSpecifier);

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
  SourceLocation asmLoc;

  auto lookat_asm_declaration = [&] {
    LookaheadParser lookahead{this};

    parse_optional_attribute_specifier_seq(attributes);

    if (!match(TokenKind::T_ASM, asmLoc)) return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_asm_declaration()) return false;

  auto ast = new (pool) AsmDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->asmLoc = asmLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  expect(TokenKind::T_STRING_LITERAL, ast->literalLoc);
  expect(TokenKind::T_RPAREN, ast->rparenLoc);
  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  ast->literal = unit->literal(ast->literalLoc);

  return true;
}

auto Parser::parse_linkage_specification(DeclarationAST*& yyast) -> bool {
  SourceLocation externLoc;
  List<AttributeSpecifierAST*>* attributes = nullptr;
  SourceLocation stringLiteralLoc;

  auto lookat_linkage_specification = [&] {
    LookaheadParser lookahead{this};

    if (!match(TokenKind::T_EXTERN, externLoc)) return false;

    parse_optional_attribute_specifier_seq(attributes);

    if (!match(TokenKind::T_STRING_LITERAL, stringLiteralLoc)) return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_linkage_specification()) return false;

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
      parse_declaration_seq(ast->declarationList);
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

void Parser::parse_optional_attribute_specifier_seq(
    List<AttributeSpecifierAST*>*& yyast) {
  if (!parse_attribute_specifier_seq(yyast)) {
    yyast = nullptr;
  }
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
  if (!lookat(TokenKind::T_LBRACKET)) return false;
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
  (void)parse_attribute_using_prefix(ast->attributeUsingPrefix);
  (void)parse_attribute_list(ast->attributeList);
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
  expect(TokenKind::T_STRING_LITERAL, ast->literalLoc);
  expect(TokenKind::T_RPAREN, ast->rparenLoc);
  ast->literal = unit->literal(ast->literalLoc);

  return true;
}

auto Parser::parse_gcc_attribute(AttributeSpecifierAST*& yyast) -> bool {
  SourceLocation attributeLoc;

  if (!match(TokenKind::T___ATTRIBUTE__, attributeLoc)) return false;

  auto ast = new (pool) GccAttributeAST();
  yyast = ast;

  ast->attributeLoc = attributeLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  (void)parse_skip_balanced();

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
  (void)parse_attribute(attribute);

  SourceLocation ellipsisLoc;
  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  if (attribute) {
    attribute->ellipsisLoc = ellipsisLoc;

    *it = new (pool) List(attribute);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    AttributeAST* attribute = nullptr;
    (void)parse_attribute(attribute);

    SourceLocation ellipsisLoc;
    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

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

  (void)parse_attribute_argument_clause(attributeArgumentClause);

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

  (void)parse_skip_balanced();

  SourceLocation rparenLoc;

  expect(TokenKind::T_RPAREN, rparenLoc);

  auto ast = new (pool) AttributeArgumentClauseAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_module_declaration(ModuleDeclarationAST*& yyast) -> bool {
  SourceLocation exportLoc;
  SourceLocation moduleLoc;

  auto lookat_module_declaration = [&] {
    LookaheadParser lookahead{this};

    (void)parse_export_keyword(exportLoc);

    if (!parse_module_keyword(moduleLoc)) return false;

    lookahead.commit();
    return true;
  };

  if (!lookat_module_declaration()) return false;

  yyast = new (pool) ModuleDeclarationAST();

  yyast->exportLoc = exportLoc;
  yyast->moduleLoc = moduleLoc;

  if (!parse_module_name(yyast->moduleName)) {
    parse_error("expected a module name");
  }

  (void)parse_module_partition(yyast->modulePartition);

  parse_optional_attribute_specifier_seq(yyast->attributeList);

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

  SourceLocation dotLoc;

  while (match(TokenKind::T_DOT, dotLoc)) {
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
      parse_declaration_seq(ast->declarationList);
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
  if (!moduleUnit_) return false;

  LookaheadParser lookahead{this};

  SourceLocation importLoc;

  return parse_import_keyword(importLoc);
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

  parse_optional_attribute_specifier_seq(ast->attributeList);

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

void Parser::parse_global_module_fragment(GlobalModuleFragmentAST*& yyast) {
  SourceLocation moduleLoc;
  SourceLocation semicolonLoc;

  auto lookat_global_module_fragment = [&] {
    LookaheadParser lookahead{this};

    if (!parse_module_keyword(moduleLoc)) return false;
    if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

    lookahead.commit();
    return true;
  };

  if (!lookat_global_module_fragment()) return;

  yyast = new (pool) GlobalModuleFragmentAST();
  yyast->moduleLoc = moduleLoc;
  yyast->semicolonLoc = semicolonLoc;

  // ### must be from preprocessor inclusion
  parse_declaration_seq(yyast->declarationList);
}

void Parser::parse_private_module_fragment(PrivateModuleFragmentAST*& yyast) {
  SourceLocation moduleLoc;
  SourceLocation colonLoc;
  SourceLocation privateLoc;

  auto lookat_private_module_fragment = [&] {
    LookaheadParser lookahead{this};
    if (!parse_module_keyword(moduleLoc)) return false;
    if (!match(TokenKind::T_COLON, colonLoc)) return false;
    if (!match(TokenKind::T_PRIVATE, privateLoc)) return false;
    lookahead.commit();
    return true;
  };

  if (!lookat_private_module_fragment()) return;

  yyast = new (pool) PrivateModuleFragmentAST();

  yyast->moduleLoc = moduleLoc;
  yyast->colonLoc = colonLoc;
  yyast->privateLoc = privateLoc;

  expect(TokenKind::T_SEMICOLON, yyast->semicolonLoc);

  parse_declaration_seq(yyast->declarationList);
}

auto Parser::parse_class_specifier(SpecifierAST*& yyast) -> bool {
  if (!LA().isOneOf(TokenKind::T_CLASS, TokenKind::T_STRUCT,
                    TokenKind::T_UNION))
    return false;

  const auto start = currentLocation();

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
  SourceLocation finalLoc;

  auto lookat_class_specifier = [&] {
    LookaheadParser lookahead{this};

    if (parse_class_head(classLoc, attributeList, className, finalLoc,
                         baseClause)) {
      if (baseClause || lookat(TokenKind::T_LBRACE)) {
        lookahead.commit();
        return true;
      }
    }

    class_specifiers_.emplace(
        start,
        std::make_tuple(currentLocation(),
                        static_cast<ClassSpecifierAST*>(nullptr), false));

    return false;
  };

  if (!lookat_class_specifier()) return false;

  SourceLocation lbraceLoc;
  expect(TokenKind::T_LBRACE, lbraceLoc);

  ClassSpecifierContext classContext(this);

  auto ast = new (pool) ClassSpecifierAST();
  yyast = ast;

  ast->classLoc = classLoc;
  ast->attributeList = attributeList;
  ast->name = className;
  ast->finalLoc = finalLoc;
  ast->baseClause = baseClause;
  ast->lbraceLoc = lbraceLoc;

  ast->classKey = unit->tokenKind(ast->classLoc);

  if (finalLoc) {
    ast->isFinal = true;
  }

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
    if (lookat(TokenKind::T_RBRACE)) break;

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
                              NameAST*& name, SourceLocation& finalLoc,
                              BaseClauseAST*& baseClause) -> bool {
  if (!parse_class_key(classLoc)) return false;

  parse_optional_attribute_specifier_seq(attributeList);

  if (parse_class_head_name(name)) {
    (void)parse_class_virt_specifier(finalLoc);
  }

  (void)parse_base_clause(baseClause);

  return true;
}

auto Parser::parse_class_head_name(NameAST*& yyast) -> bool {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  parse_optional_nested_name_specifier(nestedNameSpecifier);

  check_type_traits();

  NameAST* name = nullptr;

  if (!parse_type_name(name)) return false;

  if (!nestedNameSpecifier) yyast = name;

  return true;
}

auto Parser::parse_class_virt_specifier(SourceLocation& finalLoc) -> bool {
  if (!parse_final(finalLoc)) return false;

  return true;
}

auto Parser::parse_class_key(SourceLocation& classLoc) -> bool {
  if (LA().isOneOf(TokenKind::T_CLASS, TokenKind::T_STRUCT,
                   TokenKind::T_UNION)) {
    classLoc = consumeToken();
    return true;
  }

  return false;
}

auto Parser::parse_member_specification(DeclarationAST*& yyast) -> bool {
  return parse_member_declaration(yyast);
}

auto Parser::parse_member_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation accessLoc;

  if (parse_access_specifier(accessLoc)) {
    auto ast = new (pool) AccessDeclarationAST();
    yyast = ast;

    ast->accessLoc = accessLoc;
    expect(TokenKind::T_COLON, ast->colonLoc);

    ast->accessSpecifier = unit->tokenKind(ast->accessLoc);

    return true;
  } else if (parse_empty_declaration(yyast))
    return true;
  else if (parse_using_enum_declaration(yyast))
    return true;
  else if (parse_alias_declaration(yyast))
    return true;
  else if (parse_using_declaration(yyast))
    return true;
  else if (parse_static_assert_declaration(yyast))
    return true;
  else if (parse_deduction_guide(yyast))
    return true;
  else if (parse_opaque_enum_declaration(yyast))
    return true;
  else if (parse_template_declaration(yyast))
    return true;
  else if (parse_explicit_specialization(yyast))
    return true;
  else
    return parse_member_declaration_helper(yyast);
}

auto Parser::parse_maybe_template_member() -> bool {
  if (lookat(TokenKind::T_TEMPLATE) ||
      lookat(TokenKind::T_EXPLICIT, TokenKind::T_TEMPLATE))
    return true;

  return false;
}

auto Parser::parse_member_declaration_helper(DeclarationAST*& yyast) -> bool {
  SourceLocation extensionLoc;

  match(TokenKind::T___EXTENSION__, extensionLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_optional_attribute_specifier_seq(attributes);

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

  auto functionDeclarator = getFunctionDeclarator(declarator);

  if (hasDeclarator && functionDeclarator) {
    RequiresClauseAST* requiresClause = nullptr;

    const auto has_requires_clause = parse_requires_clause(requiresClause);

    if (!has_requires_clause) parse_virt_specifier_seq(functionDeclarator);

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

      if (classDepth_) pendingFunctionDefinitions_.push_back(ast);

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

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
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
  auto lookat_bitfield = [&] {
    LookaheadParser lookahead{this};

    SourceLocation identifierLoc;

    match(TokenKind::T_IDENTIFIER, identifierLoc);

    List<AttributeSpecifierAST*>* attributes = nullptr;
    parse_optional_attribute_specifier_seq(attributes);

    SourceLocation colonLoc;

    if (!match(TokenKind::T_COLON, colonLoc)) return false;

    ExpressionAST* sizeExpression = nullptr;

    if (!parse_constant_expression(sizeExpression)) {
      parse_error("expected an expression");
    }

    lookahead.commit();

    auto bitfieldDeclarator = new (pool) BitfieldDeclaratorAST();
    bitfieldDeclarator->identifierLoc = identifierLoc;
    bitfieldDeclarator->identifier = unit->identifier(identifierLoc);
    bitfieldDeclarator->colonLoc = colonLoc;
    bitfieldDeclarator->sizeExpression = sizeExpression;

    auto declarator = new (pool) DeclaratorAST();
    declarator->coreDeclarator = bitfieldDeclarator;

    ExpressionAST* initializer = nullptr;

    (void)parse_brace_or_equal_initializer(initializer);

    auto ast = new (pool) InitDeclaratorAST();
    yyast = ast;

    ast->declarator = declarator;
    ast->initializer = initializer;

    return true;
  };

  if (lookat_bitfield()) return true;

  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  auto ast = new (pool) InitDeclaratorAST();
  yyast = ast;

  ast->declarator = declarator;

  if (auto functionDeclarator = getFunctionDeclarator(declarator)) {
    RequiresClauseAST* requiresClause = nullptr;

    if (parse_requires_clause(requiresClause)) {
      ast->requiresClause = requiresClause;
    } else {
      parse_virt_specifier_seq(functionDeclarator);

      SourceLocation equalLoc;
      SourceLocation zeroLoc;

      const auto isPure = parse_pure_specifier(equalLoc, zeroLoc);

      functionDeclarator->isPure = isPure;
    }

    return true;
  }

  (void)parse_brace_or_equal_initializer(ast->initializer);

  return true;
}

auto Parser::parse_virt_specifier(FunctionDeclaratorAST* functionDeclarator)
    -> bool {
  SourceLocation loc;

  if (parse_final(loc)) {
    functionDeclarator->isFinal = true;
    return true;
  }

  if (parse_override(loc)) {
    functionDeclarator->isOverride = true;
    return true;
  }

  return false;
}

auto Parser::parse_pure_specifier(SourceLocation& equalLoc,
                                  SourceLocation& zeroLoc) -> bool {
  if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

  if (!match(TokenKind::T_INTEGER_LITERAL, zeroLoc)) return false;

  const auto& number = unit->tokenText(zeroLoc);

  if (number != "0") return false;

  return true;
}

auto Parser::parse_conversion_function_id(NameAST*& yyast) -> bool {
  SourceLocation operatorLoc;

  if (!match(TokenKind::T_OPERATOR, operatorLoc)) return false;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_type_specifier_seq(typeSpecifierList)) return false;

  auto declarator = new (pool) DeclaratorAST();

  (void)parse_ptr_operator_seq(declarator->ptrOpList);

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

  parse_base_specifier(baseSpecifier);

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  *it = new (pool) List(baseSpecifier);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    BaseSpecifierAST* baseSpecifier = nullptr;

    parse_base_specifier(baseSpecifier);

    SourceLocation ellipsisLoc;

    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    *it = new (pool) List(baseSpecifier);
    it = &(*it)->next;
  }

  return true;
}

void Parser::parse_base_specifier(BaseSpecifierAST*& yyast) {
  auto ast = new (pool) BaseSpecifierAST();
  yyast = ast;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  SourceLocation virtualLoc;
  SourceLocation accessLoc;

  if (match(TokenKind::T_VIRTUAL, virtualLoc)) {
    ast->isVirtual = true;
    (void)parse_access_specifier(accessLoc);
  } else if (parse_access_specifier(accessLoc)) {
    ast->isVirtual = match(TokenKind::T_VIRTUAL, virtualLoc);
  }

  if (accessLoc) {
    ast->accessSpecifier = unit->tokenKind(accessLoc);
  }

  if (!parse_class_or_decltype(ast->name)) {
    parse_error("expected a class name");
  }
}

auto Parser::parse_class_or_decltype(NameAST*& yyast) -> bool {
  auto lookat_qualified_name = [&] {
    LookaheadParser lookahead{this};

    NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

    if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

    SourceLocation templateLoc;

    const auto hasTemplate = match(TokenKind::T_TEMPLATE, templateLoc);

    NameAST* name = nullptr;

    if (hasTemplate) {
      if (!parse_simple_template_id(name)) {
        parse_error("expected a template id");
      }
    } else {
      if (!parse_type_name(name)) return false;
    }

    lookahead.commit();

    auto ast = new (pool) QualifiedNameAST();
    yyast = ast;

    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->templateLoc = templateLoc;
    ast->id = name;

    return true;
  };

  if (lookat_qualified_name()) return true;

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
  if (LA().isOneOf(TokenKind::T_PRIVATE, TokenKind::T_PROTECTED,
                   TokenKind::T_PUBLIC)) {
    loc = consumeToken();
    return true;
  }

  return false;
}

auto Parser::parse_ctor_initializer(CtorInitializerAST*& yyast) -> bool {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  auto ast = new (pool) CtorInitializerAST();
  yyast = ast;

  ast->colonLoc = colonLoc;

  parse_mem_initializer_list(ast->memInitializerList);

  return true;
}

void Parser::parse_mem_initializer_list(List<MemInitializerAST*>*& yyast) {
  auto it = &yyast;

  MemInitializerAST* mem_initializer = nullptr;

  parse_mem_initializer(mem_initializer);

  *it = new (pool) List(mem_initializer);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    MemInitializerAST* mem_initializer = nullptr;

    parse_mem_initializer(mem_initializer);
    *it = new (pool) List(mem_initializer);
    it = &(*it)->next;
  }
}

void Parser::parse_mem_initializer(MemInitializerAST*& yyast) {
  NameAST* name = nullptr;

  if (!parse_mem_initializer_id(name)) parse_error("expected an member id");

  if (lookat(TokenKind::T_LBRACE)) {
    auto ast = new (pool) BracedMemInitializerAST();
    yyast = ast;

    ast->name = name;

    if (!parse_braced_init_list(ast->bracedInitList)) {
      parse_error("expected an initializer");
    }

    match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

    return;
  }

  auto ast = new (pool) ParenMemInitializerAST();
  yyast = ast;

  ast->name = name;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression_list(ast->expressionList)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);
}

auto Parser::parse_mem_initializer_id(NameAST*& yyast) -> bool {
  auto lookat_class_or_decltype = [&] {
    LookaheadParser lookahead{this};
    if (!parse_class_or_decltype(yyast)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_class_or_decltype()) return true;

  SourceLocation identifierLoc;

  if (match(TokenKind::T_IDENTIFIER, identifierLoc)) {
    auto name = new (pool) SimpleNameAST();
    yyast = name;
    name->identifierLoc = identifierLoc;
    name->identifier = unit->identifier(identifierLoc);
    return true;
  }

  return false;
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
  SourceLocation operatorLoc;

  auto lookat_literal_operator_id = [&] {
    LookaheadParser lookahead{this};

    if (!match(TokenKind::T_OPERATOR, operatorLoc)) return false;

    if (!lookat(TokenKind::T_USER_DEFINED_STRING_LITERAL) &&
        !lookat(TokenKind::T_STRING_LITERAL, TokenKind::T_IDENTIFIER))
      return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_literal_operator_id()) return false;

  auto ast = new (pool) LiteralOperatorNameAST();
  yyast = ast;

  ast->operatorLoc = operatorLoc;

  if (match(TokenKind::T_STRING_LITERAL, ast->literalLoc)) {
    expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
    ast->literal = unit->literal(ast->literalLoc);
    ast->identifier = unit->identifier(ast->identifierLoc);
  } else {
    expect(TokenKind::T_USER_DEFINED_STRING_LITERAL, ast->literalLoc);
    ast->literal = unit->literal(ast->literalLoc);
  }

  return true;
}

auto Parser::parse_template_declaration(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_TEMPLATE, TokenKind::T_LESS)) return false;

  auto ast = new (pool) TemplateDeclarationAST();
  yyast = ast;

  expect(TokenKind::T_TEMPLATE, ast->templateLoc);

  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    if (!parse_template_parameter_list(ast->templateParameterList)) {
      parse_error("expected a template parameter");
    }

    expect(TokenKind::T_GREATER, ast->greaterLoc);
  }

  (void)parse_requires_clause(ast->requiresClause);

  if (!parse_concept_definition(ast->declaration)) {
    if (!parse_declaration(ast->declaration))
      parse_error("expected a declaration");
  }

  return true;
}

auto Parser::parse_template_parameter_list(List<DeclarationAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  DeclarationAST* declaration = nullptr;

  if (!parse_template_parameter(declaration)) return false;

  *it = new (pool) List(declaration);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
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

  if (!parse_constraint_logical_or_expression(yyast->expression)) {
    parse_error("expected a requirement expression");
  }

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
  auto lookat_type_parameter = [&] {
    LookaheadParser lookahead{this};

    if (!parse_type_parameter(yyast)) return false;

    lookahead.commit();

    return true;
  };

  auto lookat_parameter_declaration = [&] {
    LookaheadParser lookahead{this};

    ParameterDeclarationAST* parameter = nullptr;

    if (!parse_parameter_declaration(parameter, /*templParam*/ true))
      return false;

    yyast = parameter;

    lookahead.commit();

    return true;
  };

  if (lookat_type_parameter()) return true;
  if (lookat_parameter_declaration()) return true;

  return parse_constraint_type_parameter(yyast);
}

auto Parser::parse_type_parameter(DeclarationAST*& yyast) -> bool {
  if (parse_template_type_parameter(yyast)) return true;

  if (parse_typename_type_parameter(yyast)) return true;

  return false;
}

auto Parser::parse_typename_type_parameter(DeclarationAST*& yyast) -> bool {
  auto maybe_elaborated_type_spec = [this]() {
    SourceLocation typenameLoc;

    if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

    SourceLocation identifierLoc;

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    SourceLocation scopeLoc;
    SourceLocation lessLoc;

    if (match(TokenKind::T_COLON_COLON, scopeLoc) ||
        match(TokenKind::T_LESS, lessLoc)) {
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
  if (!lookat(TokenKind::T_TEMPLATE, TokenKind::T_LESS)) return false;

  SourceLocation templateLoc;

  expect(TokenKind::T_TEMPLATE, templateLoc);

  SourceLocation lessLoc;

  expect(TokenKind::T_LESS, lessLoc);

  SourceLocation greaterLoc;
  List<DeclarationAST*>* templateParameterList = nullptr;

  if (!match(TokenKind::T_GREATER, greaterLoc)) {
    if (!parse_template_parameter_list(templateParameterList)) {
      parse_error("expected a template parameter");
    }

    expect(TokenKind::T_GREATER, greaterLoc);
  }

  RequiresClauseAST* requiresClause = nullptr;
  (void)parse_requires_clause(requiresClause);

  SourceLocation classsKeyLoc;

  if (!parse_type_parameter_key(classsKeyLoc)) {
    parse_error("expected a type parameter");
  }

  if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_EQUAL) ||
      lookat(TokenKind::T_EQUAL)) {
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

  if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_EQUAL) ||
      lookat(TokenKind::T_EQUAL)) {
    SourceLocation identifierLoc;

    match(TokenKind::T_IDENTIFIER, identifierLoc);

    SourceLocation equalLoc;

    expect(TokenKind::T_EQUAL, equalLoc);

    TypeIdAST* typeId = nullptr;

    if (!parse_type_id(typeId)) {
      return false;  // ### FIXME: parse_error("expected a type id");
    }

    return true;
  }

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  SourceLocation identifierLoc;

  match(TokenKind::T_IDENTIFIER, identifierLoc);

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
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameAST* name = nullptr;

  auto lookat_type_constraint = [&] {
    LookaheadParser lookahead{this};

    parse_optional_nested_name_specifier(nestedNameSpecifier);

    if (!parse_concept_name(name)) return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_type_constraint()) return false;

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

auto Parser::parse_simple_template_or_name_id(NameAST*& yyast) -> bool {
  auto lookat_simple_template_id = [&] {
    LookaheadParser lookahead{this};
    if (!parse_simple_template_id(yyast)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_simple_template_id()) return true;

  return parse_name_id(yyast);
}

auto Parser::parse_simple_template_id(NameAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_IDENTIFIER) || LA(1).isNot(TokenKind::T_LESS)) {
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
  if (lookat(TokenKind::T_OPERATOR)) {
    NameAST* name = nullptr;

    auto lookat_operator = [&] {
      LookaheadParser lookahead{this};

      if (!parse_literal_operator_id(name) && !parse_operator_function_id(name))
        return false;

      if (!lookat(TokenKind::T_LESS)) return false;

      lookahead.commit();

      return true;
    };

    if (!lookat_operator()) return false;

    auto ast = new (pool) TemplateNameAST();
    yyast = ast;

    ast->id = name;

    expect(TokenKind::T_LESS, ast->lessLoc);

    if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
      if (!parse_template_argument_list(ast->templateArgumentList)) {
        parse_error("expected a template argument");
      }

      expect(TokenKind::T_GREATER, ast->greaterLoc);
    }

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

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  *it = new (pool) List(templateArgument);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    TemplateArgumentAST* templateArgument = nullptr;

    if (!parse_template_argument(templateArgument)) {
      // parse_error("expected a template argument"); // ### FIXME
      return false;
    }

    SourceLocation ellipsisLoc;

    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

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
    return LA().isOneOf(TokenKind::T_COMMA, TokenKind::T_GREATER,
                        TokenKind::T_DOT_DOT_DOT);
  };

  auto lookat_type_id = [&] {
    LookaheadParser lookahead{this};

    TypeIdAST* typeId = nullptr;

    if (!parse_type_id(typeId)) return false;

    if (!check()) return false;

    lookahead.commit();

    auto ast = new (pool) TypeTemplateArgumentAST();
    yyast = ast;

    ast->typeId = typeId;

    return true;
  };

  auto lookat_template_argument_constant_expression = [&] {
    LookaheadParser lookahead{this};

    ExpressionAST* expression = nullptr;

    if (!parse_template_argument_constant_expression(expression)) return false;

    if (!check()) return false;

    lookahead.commit();

    auto ast = new (pool) ExpressionTemplateArgumentAST();
    yyast = ast;

    ast->expression = expression;

    return true;
  };

  if (lookat_type_id() || lookat_template_argument_constant_expression()) {
    template_arguments_.emplace(
        start, std::make_tuple(currentLocation(), yyast, true));

    return true;
  }

  return false;
}

auto Parser::parse_constraint_expression(ExpressionAST*& yyast) -> bool {
  ExprContext exprContext;
  return parse_logical_or_expression(yyast, exprContext);
}

auto Parser::parse_deduction_guide(DeclarationAST*& yyast) -> bool {
  SpecifierAST* explicitSpecifier = nullptr;
  SourceLocation identifierLoc;
  SourceLocation lparenLoc;

  auto lookat_deduction_guide = [&] {
    LookaheadParser lookahead{this};

    (void)parse_explicit_specifier(explicitSpecifier);

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    const SourceLocation saved = currentLocation();

    if (!parse_skip_balanced()) return false;

    if (!lookat(TokenKind::T_RPAREN, TokenKind::T_MINUS_GREATER)) return false;

    rewind(saved);
    lookahead.commit();
    return true;
  };

  if (!lookat_deduction_guide()) return false;

  SourceLocation rparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_parameter_declaration_clause(parameterDeclarationClause)) {
      parse_error("expected a parameter declaration");
    }

    expect(TokenKind::T_RPAREN, rparenLoc);
  }

  SourceLocation arrowLoc;

  expect(TokenKind::T_MINUS_GREATER, arrowLoc);

  NameAST* templateId = nullptr;

  if (!parse_simple_template_id(templateId)) {
    parse_error("expected a template id");
  }

  SourceLocation semicolonLoc;

  expect(TokenKind::T_SEMICOLON, semicolonLoc);

  auto ast = new (pool) DeductionGuideAST();
  yyast = ast;
  ast->explicitSpecifier = explicitSpecifier;
  ast->identifierLoc = identifierLoc;
  ast->lparenLoc = lparenLoc;
  ast->parameterDeclarationClause = parameterDeclarationClause;
  ast->rparenLoc = rparenLoc;
  ast->arrowLoc = arrowLoc;
  ast->templateId = templateId;
  ast->semicolonLoc = semicolonLoc;
  ast->identifier = unit->identifier(identifierLoc);

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
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  NameAST* name = nullptr;

  auto lookat_typename_specifier = [&] {
    LookaheadParser lookahead{this};
    if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

    if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

    match(TokenKind::T_TEMPLATE, templateLoc);

    if (!parse_simple_template_or_name_id(name)) return false;

    lookahead.commit();
    return true;
  };

  if (!lookat_typename_specifier()) return false;

  auto ast = new (pool) TypenameSpecifierAST();
  yyast = ast;

  ast->typenameLoc = typenameLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;

  return true;
}

auto Parser::parse_explicit_instantiation(DeclarationAST*& yyast) -> bool {
  auto lookat_explicit_instantiation = [&] {
    LookaheadParser _{this};

    SourceLocation externLoc;
    match(TokenKind::T_EXTERN, externLoc);

    SourceLocation templateLoc;
    if (!match(TokenKind::T_TEMPLATE, templateLoc)) return false;

    if (lookat(TokenKind::T_LESS)) return false;

    return true;
  };

  if (!lookat_explicit_instantiation()) return false;

  auto ast = new (pool) ExplicitInstantiationAST();
  yyast = ast;

  match(TokenKind::T_EXTERN, ast->externLoc);
  expect(TokenKind::T_TEMPLATE, ast->templateLoc);

  if (!parse_declaration(ast->declaration))
    parse_error("expected a declaration");

  return true;
}

auto Parser::parse_explicit_specialization(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_TEMPLATE, TokenKind::T_LESS, TokenKind::T_GREATER))
    return false;

  auto ast = new (pool) TemplateDeclarationAST();
  yyast = ast;

  expect(TokenKind::T_TEMPLATE, ast->templateLoc);
  expect(TokenKind::T_LESS, ast->lessLoc);
  expect(TokenKind::T_GREATER, ast->greaterLoc);

  if (!parse_declaration(ast->declaration))
    parse_error("expected a declaration");

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

  if (!lookat(TokenKind::T_LBRACE)) {
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
  if (!lookat(TokenKind::T_CATCH)) return false;

  auto it = &yyast;

  HandlerAST* handler = nullptr;
  while (parse_handler(handler)) {
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

  parse_optional_attribute_specifier_seq(ast->attributeList);

  if (!parse_type_specifier_seq(ast->typeSpecifierList)) {
    parse_error("expected a type specifier");
  }

  if (lookat(TokenKind::T_RPAREN)) return true;

  parse_optional_declarator_or_abstract_declarator(ast->declarator);

  return true;
}

auto Parser::parse_noexcept_specifier(ExceptionSpecifierAST*& yyast) -> bool {
  SourceLocation throwLoc;

  if (match(TokenKind::T_THROW, throwLoc)) {
    auto ast = new (pool) ThrowExceptionSpecifierAST();
    yyast = ast;

    ast->throwLoc = throwLoc;
    expect(TokenKind::T_LPAREN, ast->lparenLoc);
    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  SourceLocation noexceptLoc;

  if (!match(TokenKind::T_NOEXCEPT, noexceptLoc)) return false;

  auto ast = new (pool) NoexceptSpecifierAST();
  yyast = ast;

  ast->noexceptLoc = noexceptLoc;

  if (match(TokenKind::T_LPAREN, ast->lparenLoc)) {
    if (!parse_constant_expression(ast->expression)) {
      parse_error("expected a declaration");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  return true;
}

auto Parser::parse_identifier_list(List<NameAST*>*& yyast) -> bool {
  auto it = &yyast;

  NameAST* id = nullptr;

  if (!parse_name_id(id)) return false;

  if (id) {
    *it = new (pool) List<NameAST*>(id);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    NameAST* id = nullptr;

    if (!parse_name_id(id)) {
      parse_error("expected an identifier");
    }

    if (id) {
      *it = new (pool) List<NameAST*>(id);
      it = &(*it)->next;
    }
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

  const auto saved = currentLocation();

  rewind(functionBody->statement->lbraceLoc.next());

  finish_compound_statement(functionBody->statement);

  rewind(saved);
}

}  // namespace cxx
