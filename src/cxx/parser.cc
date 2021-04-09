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
#include <cxx/names.h>
#include <cxx/parser.h>
#include <cxx/scope.h>
#include <cxx/semantics.h>
#include <cxx/symbol_factory.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/type_environment.h>
#include <cxx/types.h>

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

Parser::Parser(TranslationUnit* unit) : unit(unit) {
  control = unit->control();
  symbols = control->symbols();
  types = control->types();
  cursor_ = 1;

  pool = unit->arena();

  sem = std::make_unique<Semantics>(unit);

  module_id = control->identifier("module");
  import_id = control->identifier("import");
  final_id = control->identifier("final");
  override_id = control->identifier("override");
}

Parser::~Parser() {}

bool Parser::checkTypes() const { return checkTypes_; }

void Parser::setCheckTypes(bool checkTypes) { checkTypes_ = checkTypes; }

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
  Semantics::SpecifiersSem specifiers;

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

struct Parser::ClassSpecifierContext {
  ClassSpecifierContext(const ClassSpecifierContext&) = delete;
  ClassSpecifierContext& operator=(const ClassSpecifierContext&) = delete;

  Parser* p;

  explicit ClassSpecifierContext(Parser* p) : p(p) { ++p->classDepth; }

  ~ClassSpecifierContext() {
    if (--p->classDepth == 0) p->completePendingFunctionDefinitions();
  }
};

const Token& Parser::LA(int n) const {
  return unit->tokenAt(SourceLocation(cursor_ + n));
}

bool Parser::operator()(UnitAST*& ast) {
  auto result = parse(ast);
  return result;
}

bool Parser::parse(UnitAST*& ast) {
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

bool Parser::parse_type_name(NameAST*& yyast) {
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
  globalNamespace_ = symbols->newNamespaceSymbol(nullptr, nullptr);

  Semantics::ScopeContext scopeContext(sem.get(), globalNamespace_->scope());

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

  Semantics::NameSem nameSem;

  sem->name(ast->name, &nameSem);

  if (checkTypes_ && nameSem.name) {
    ast->symbol = sem->scope()->unqualifiedLookup(nameSem.name);

    if (!ast->symbol)
      parse_warn(ast->name->firstSourceLocation(), "undefined symbol '{}'",
                 *nameSem.name);
    else {
#if 0
      parse_warn(ast->name->firstSourceLocation(),
                 "'{}' resolved to '{}' with type '{}'", *nameSem.name,
                 typeid(*ast->symbol).name(), ast->symbol->type());
#endif
    }
  }

  return true;
}

bool Parser::parse_id_expression(NameAST*& yyast) {
  const auto start = currentLocation();

  if (parse_qualified_id(yyast)) return true;

  rewind(start);

  yyast = nullptr;

  if (!parse_unqualified_id(yyast)) return false;

  return true;
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
    ast->id = name;

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
    if (LA().is(TokenKind::T_IDENTIFIER) &&
        LA(1).is(TokenKind::T_COLON_COLON)) {
      NameAST* name = nullptr;
      parse_name_id(name);
      expect(TokenKind::T_COLON_COLON);

      *nameIt = new (pool) List(name);
      nameIt = &(*nameIt)->next;
    } else {
      const auto saved = currentLocation();

      const auto has_template = match(TokenKind::T_TEMPLATE);

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
  SourceLocation defaultCaptureLoc;
  List<LambdaCaptureAST*>* captureList = nullptr;

  if (!match(TokenKind::T_RBRACKET, rbracketLoc)) {
    if (!parse_lambda_capture(defaultCaptureLoc, captureList))
      parse_error("expected a lambda capture");

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

bool Parser::parse_lambda_capture(SourceLocation& captureDefaultLoc,
                                  List<LambdaCaptureAST*>*& captureList) {
  if (parse_capture_default(captureDefaultLoc)) {
    if (match(TokenKind::T_COMMA)) {
      if (!parse_capture_list(captureList)) parse_error("expected a capture");
    }

    return true;
  }

  return parse_capture_list(captureList);
}

bool Parser::parse_capture_default(SourceLocation& opLoc) {
  const auto start = currentLocation();

  if (!match(TokenKind::T_AMP) && !match(TokenKind::T_EQUAL)) return false;

  if (LA().isNot(TokenKind::T_COMMA) && LA().isNot(TokenKind::T_RBRACKET)) {
    rewind(start);
    return false;
  }

  opLoc = start;

  return true;
}

bool Parser::parse_capture_list(List<LambdaCaptureAST*>*& yyast) {
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

bool Parser::parse_capture(LambdaCaptureAST*& yyast) {
  const auto start = currentLocation();

  if (parse_init_capture(yyast)) return true;

  rewind(start);

  return parse_simple_capture(yyast);
}

bool Parser::parse_simple_capture(LambdaCaptureAST*& yyast) {
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

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  return true;
}

bool Parser::parse_init_capture(LambdaCaptureAST*& yyast) {
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
  ast->initializer = initializer;

  return true;
}

bool Parser::parse_fold_expression(ExpressionAST*& yyast) {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool) LeftFoldExpressionAST();
    yyast = ast;

    ast->lparenLoc = lparenLoc;
    ast->ellipsisLoc = ellipsisLoc;

    if (!parse_fold_operator(ast->opLoc, ast->op))
      parse_error("expected fold operator");

    if (!parse_cast_expression(ast->expression))
      parse_error("expected an expression");

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

  if (!parse_fold_operator(ast->foldOpLoc, ast->foldOp))
    parse_error("expected a fold operator");

  if (!parse_cast_expression(ast->rightExpression))
    parse_error("expected an expression");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

bool Parser::parse_fold_operator(SourceLocation& loc, TokenKind& op) {
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

  Semantics::ExpressionSem expr;

  sem->expression(expression, &expr);

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

  Semantics::ExpressionSem expr;

  sem->expression(expression, &expr);

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

  Semantics::ExpressionSem expr;

  sem->expression(ast->expression, &expr);

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

bool Parser::parse_typeid_expression(ExpressionAST*& yyast) {
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

  Semantics::ExpressionSem expr;

  sem->expression(ast->expression, &expr);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

bool Parser::parse_typename_expression(ExpressionAST*& yyast) {
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
  ast->op = unit->tokenKind(opLoc);

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

  if (!parse_unary_expression(ast->expression))
    parse_error("expected an expression");

  return true;
}

bool Parser::parse_alignof_expression(ExpressionAST*& yyast) {
  SourceLocation alignofLoc;

  if (!match(TokenKind::T_ALIGNOF, alignofLoc) &&
      !match(TokenKind::T___ALIGNOF, alignofLoc) &&
      !match(TokenKind::T___ALIGNOF__, alignofLoc))
    return false;

  auto ast = new (pool) AlignofExpressionAST();
  yyast = ast;

  ast->alignofLoc = alignofLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

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
  SourceLocation noexceptLoc;

  if (!match(TokenKind::T_NOEXCEPT, noexceptLoc)) return false;

  auto ast = new (pool) NoexceptExpressionAST();
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_expression(ast->expression)) parse_error("expected an expression");

  Semantics::ExpressionSem expr;

  sem->expression(ast->expression, &expr);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

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

    Semantics::ExpressionSem expr;

    sem->expression(expression, &expr);

    expect(TokenKind::T_RBRACKET);
  }

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  while (match(TokenKind::T_LBRACKET)) {
    if (!match(TokenKind::T_RBRACKET)) {
      ExpressionAST* expression = nullptr;

      if (!parse_constant_expression(expression))
        parse_error("expected an expression");

      Semantics::ExpressionSem expr;

      sem->expression(expression, &expr);

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

  SourceLocation scopeLoc;

  const auto has_scope_op = match(TokenKind::T_COLON_COLON, scopeLoc);

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

  if (!parse_cast_expression(ast->expression))
    parse_error("expected an expression");

  return true;
}

bool Parser::parse_cast_expression(ExpressionAST*& yyast) {
  const auto start = currentLocation();

  if (parse_cast_expression_helper(yyast)) return true;

  rewind(start);

  return parse_unary_expression(yyast);
}

bool Parser::parse_cast_expression_helper(ExpressionAST*& yyast) {
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

bool Parser::parse_binary_operator(SourceLocation& loc, TokenKind& tk,
                                   const ExprContext& exprContext) {
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

bool Parser::parse_binary_expression(ExpressionAST*& yyast,
                                     const ExprContext& exprContext) {
  if (!parse_cast_expression(yyast)) return false;

  const auto saved = currentLocation();

  if (!parse_binary_expression_helper(yyast, Prec::kLogicalOr, exprContext))
    rewind(saved);

  return true;
}

bool Parser::parse_lookahead_binary_operator(SourceLocation& loc, TokenKind& tk,
                                             const ExprContext& exprContext) {
  const auto saved = currentLocation();

  const auto has_binop = parse_binary_operator(loc, tk, exprContext);

  rewind(saved);

  return has_binop;
}

bool Parser::parse_binary_expression_helper(ExpressionAST*& yyast, Prec minPrec,
                                            const ExprContext& exprContext) {
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
      if (!parse_binary_expression_helper(rhs, prec(op), exprContext)) {
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

bool Parser::parse_logical_or_expression(ExpressionAST*& yyast,
                                         const ExprContext& exprContext) {
  return parse_binary_expression(yyast, exprContext);
}

bool Parser::parse_conditional_expression(ExpressionAST*& yyast,
                                          const ExprContext& exprContext) {
  if (!parse_logical_or_expression(yyast, exprContext)) return false;

  SourceLocation questionLoc;

  if (match(TokenKind::T_QUESTION, questionLoc)) {
    auto ast = new (pool) ConditionalExpressionAST();
    ast->condition = yyast;
    ast->questionLoc = questionLoc;

    yyast = ast;

    if (!parse_expression(ast->iftrueExpression))
      parse_error("expected an expression");

    Semantics::ExpressionSem expr;

    sem->expression(ast->iftrueExpression, &expr);

    expect(TokenKind::T_COLON, ast->colonLoc);

    if (exprContext.templArg) {
      if (!parse_conditional_expression(ast->iffalseExpression, exprContext))
        parse_error("expected an expression");
    } else if (!parse_assignment_expression(ast->iffalseExpression)) {
      parse_error("expected an expression");
    }

    Semantics::ExpressionSem iffalseExpr;

    sem->expression(ast->iffalseExpression, &iffalseExpr);
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

    Semantics::ExpressionSem expr;

    sem->expression(expression, &expr);
  }

  return true;
}

bool Parser::parse_throw_expression(ExpressionAST*& yyast) {
  SourceLocation throwLoc;

  if (!match(TokenKind::T_THROW, throwLoc)) return false;

  auto ast = new (pool) ThrowExpressionAST();
  yyast = ast;

  ast->throwLoc = throwLoc;

  const auto saved = currentLocation();

  if (!parse_assignment_expression(ast->expression)) rewind(saved);

  Semantics::ExpressionSem expr;

  sem->expression(ast->expression, &expr);

  return true;
}

bool Parser::parse_assignment_expression(ExpressionAST*& yyast) {
  ExprContext context;
  return parse_assignment_expression(yyast, context);
}

bool Parser::parse_assignment_expression(ExpressionAST*& yyast,
                                         const ExprContext& exprContext) {
  if (parse_yield_expression(yyast)) return true;

  if (parse_throw_expression(yyast)) return true;

  if (!parse_conditional_expression(yyast, exprContext)) return false;

  Semantics::ExpressionSem expr;

  sem->expression(yyast, &expr);

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
    ast->op = op;

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
  const auto start = currentLocation();

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
  ExprContext exprContext;
  return parse_conditional_expression(yyast, exprContext);
}

bool Parser::parse_template_argument_constant_expression(
    ExpressionAST*& yyast) {
  ExprContext exprContext;
  exprContext.templArg = true;
  return parse_conditional_expression(yyast, exprContext);
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

bool Parser::parse_init_statement(StatementAST*& yyast) {
  if (LA().is(TokenKind::T_RPAREN)) return false;

  auto saved = currentLocation();

  DeclarationAST* declaration = nullptr;

  if (parse_simple_declaration(declaration, false)) return true;

  rewind(saved);

  ExpressionAST* expression = nullptr;

  if (!parse_expression(expression)) return false;

  Semantics::ExpressionSem expr;

  sem->expression(expression, &expr);

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

  if (!parse_expression(yyast)) return false;

  Semantics::ExpressionSem expr;

  sem->expression(yyast, &expr);

  return true;
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

  Semantics::ExpressionSem expr;

  sem->expression(expression, &expr);

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

    Semantics::ExpressionSem expr;

    sem->expression(expression, &expr);

    expect(TokenKind::T_SEMICOLON, semicolonLoc);
  }

  auto ast = new (pool) ExpressionStatementAST;
  yyast = ast;

  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

bool Parser::parse_compound_statement(CompoundStatementAST*& yyast, bool skip) {
  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  auto blockSymbol = symbols->newBlockSymbol(sem->scope(), nullptr);
  sem->scope()->add(blockSymbol);

  Semantics::ScopeContext scopeContext(sem.get(), blockSymbol->scope());

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

  Semantics::ExpressionSem expr;

  sem->expression(ast->expression, &expr);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

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

    Semantics::ExpressionSem expr;

    sem->expression(ast->expression, &expr);

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

bool Parser::parse_simple_declaration(DeclarationAST*& yyast,
                                      bool acceptFunctionDefinition) {
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

  if (acceptFunctionDefinition &&
      parse_notypespec_function_definition(yyast, declSpecifierList, specs))
    return true;

  rewind(after_decl_specs);

  auto lastDeclSpecifier = &declSpecifierList;

  while (*lastDeclSpecifier) {
    lastDeclSpecifier = &(*lastDeclSpecifier)->next;
  }

  if (!parse_decl_specifier_seq(*lastDeclSpecifier, specs))
    rewind(after_decl_specs);

  after_decl_specs = currentLocation();

  for (auto it = declSpecifierList; it; it = it->next) {
    if (auto elaboratedTypeSpec =
            dynamic_cast<ElaboratedTypeSpecifierAST*>(it->value)) {
      if (!elaboratedTypeSpec->symbol) {
        auto classKey = unit->tokenKind(elaboratedTypeSpec->classLoc);

        Semantics::NameSem nameSem;
        sem->name(elaboratedTypeSpec->name, &nameSem);

        if (classKey == TokenKind::T_ENUM) {
          auto enumSymbol = symbols->newEnumSymbol(sem->scope(), nameSem.name);
          enumSymbol->setType(QualifiedType(types->enumType(enumSymbol)));
          sem->scope()->add(enumSymbol);
        } else {
          auto classSymbol =
              symbols->newClassSymbol(sem->scope(), nameSem.name);
          classSymbol->setType(QualifiedType(types->classType(classSymbol)));
          sem->scope()->add(classSymbol);
        }
      }
      break;
    }
  }

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

  Semantics::DeclaratorSem decl{specs.specifiers};

  sem->declarator(declarator, &decl);

  const auto after_declarator = currentLocation();

  auto functionDeclarator = getFunctionDeclarator(declarator);

  if (acceptFunctionDefinition && functionDeclarator &&
      lookat_function_body()) {
    FunctionSymbol* functionSymbol = nullptr;

    functionSymbol = symbols->newFunctionSymbol(sem->scope(), decl.name);
    functionSymbol->setType(decl.type);
    sem->scope()->add(functionSymbol);

    FunctionBodyAST* functionBody = nullptr;

    Semantics::ScopeContext scopeContext(sem.get(), functionSymbol->scope());

    if (auto params = functionDeclarator->parametersAndQualifiers) {
      if (auto paramDeclarations = params->parameterDeclarationClause) {
        for (auto it = paramDeclarations->parameterDeclarationList; it;
             it = it->next) {
          Semantics::SpecifiersSem specifiers;
          sem->specifiers(it->value->typeSpecifierList, &specifiers);
          Semantics::DeclaratorSem decl{specifiers};
          sem->declarator(it->value->declarator, &decl);
          auto param = symbols->newArgumentSymbol(sem->scope(), decl.name);
          param->setType(decl.type);
          sem->scope()->add(param);
        }
      }
    }

    if (!parse_function_body(functionBody))
      parse_error("expected function body");

    auto ast = new (pool) FunctionDefinitionAST();
    yyast = ast;

    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->declarator = declarator;
    ast->functionBody = functionBody;
    ast->symbol = functionSymbol;

    if (classDepth) pendingFunctionDefinitions_.push_back(ast);

    return true;
  }

  const bool isTypedef = decl.specifiers.isTypedef;

  InitDeclaratorAST* initDeclarator = new (pool) InitDeclaratorAST();

  initDeclarator->declarator = declarator;

  if (!parse_declarator_initializer(initDeclarator->initializer))
    rewind(after_declarator);

  List<InitDeclaratorAST*>* initDeclaratorList = nullptr;

  auto declIt = &initDeclaratorList;

  *declIt = new (pool) List(initDeclarator);
  declIt = &(*declIt)->next;

  if (isTypedef) {
    auto typedefSymbol = symbols->newTypedefSymbol(sem->scope(), decl.name);
    typedefSymbol->setType(decl.type);
    sem->scope()->add(typedefSymbol);
  } else if (functionDeclarator) {
    FunctionSymbol* functionSymbol = nullptr;
    functionSymbol = symbols->newFunctionSymbol(sem->scope(), decl.name);
    functionSymbol->setType(decl.type);
    sem->scope()->add(functionSymbol);
  } else {
    VariableSymbol* varSymbol = nullptr;
    varSymbol = symbols->newVariableSymbol(sem->scope(), decl.name);
    varSymbol->setType(decl.type);
    sem->scope()->add(varSymbol);
  }

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

bool Parser::parse_notypespec_function_definition(
    DeclarationAST*& yyast, List<SpecifierAST*>* declSpecifierList,
    const DeclSpecs& specs) {
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

  Semantics::DeclaratorSem decl{specs.specifiers};

  sem->declarator(declarator, &decl);

  const auto has_requires_clause = parse_requires_clause();

  bool has_virt_specifier_seq = false;

  if (!has_requires_clause) has_virt_specifier_seq = parse_virt_specifier_seq();

  SourceLocation semicolonLoc;

  if (match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    auto initDeclarator = new (pool) InitDeclaratorAST();
    initDeclarator->declarator = declarator;

    auto ast = new (pool) SimpleDeclarationAST();
    yyast = ast;
    ast->declSpecifierList = declSpecifierList;
    ast->initDeclaratorList = new (pool) List(initDeclarator);
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

bool Parser::parse_static_assert_declaration(DeclarationAST*& yyast) {
  SourceLocation staticAssertLoc;

  if (!match(TokenKind::T_STATIC_ASSERT, staticAssertLoc)) return false;

  auto ast = new (pool) StaticAssertDeclarationAST();
  yyast = ast;

  ast->staticAssertLoc = staticAssertLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_constant_expression(ast->expression))
    parse_error("expected an expression");

  Semantics::ExpressionSem expr;

  sem->expression(ast->expression, &expr);

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

  sem->specifiers(specifier, &specs.specifiers);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    sem->specifiers(specifier, &specs.specifiers);

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

  sem->specifiers(specifier, &specs.specifiers);

  *it = new (pool) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    sem->specifiers(specifier, &specs.specifiers);

    *it = new (pool) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

bool Parser::parse_storage_class_specifier(SpecifierAST*& yyast) {
  SourceLocation loc;

  if (match(TokenKind::T_STATIC, loc)) {
    auto ast = new (pool) StaticSpecifierAST();
    yyast = ast;
    ast->staticLoc = loc;
    return true;
  } else if (match(TokenKind::T_THREAD_LOCAL, loc)) {
    auto ast = new (pool) ThreadLocalSpecifierAST();
    yyast = ast;
    ast->threadLocalLoc = loc;
    return true;
  } else if (match(TokenKind::T_EXTERN, loc)) {
    auto ast = new (pool) ExternSpecifierAST();
    yyast = ast;
    ast->externLoc = loc;
    return true;
  } else if (match(TokenKind::T_MUTABLE, loc)) {
    auto ast = new (pool) MutableSpecifierAST();
    yyast = ast;
    ast->mutableLoc = loc;
    return true;
  } else if (match(TokenKind::T___THREAD, loc)) {
    auto ast = new (pool) ThreadSpecifierAST();
    yyast = ast;
    ast->threadLoc = loc;
    return true;
  }

  return false;
}

bool Parser::parse_function_specifier(SpecifierAST*& yyast) {
  SourceLocation virtualLoc;

  if (match(TokenKind::T_VIRTUAL, virtualLoc)) {
    auto ast = new (pool) VirtualSpecifierAST();
    yyast = ast;
    ast->virtualLoc = virtualLoc;
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

    Semantics::ExpressionSem expr;

    sem->expression(ast->expression, &expr);

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

  sem->specifiers(typeSpecifier, &specs.specifiers);

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

    sem->specifiers(typeSpecifier, &specs.specifiers);

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
  auto it = &yyast;

  SpecifierAST* typeSpecifier = nullptr;

  if (!parse_defining_type_specifier(typeSpecifier, specs)) return false;

  List<AttributeAST*>* attributes = nullptr;

  parse_attribute_specifier_seq(attributes);

  sem->specifiers(typeSpecifier, &specs.specifiers);

  *it = new (pool) List(typeSpecifier);
  it = &(*it)->next;

  while (LA()) {
    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_defining_type_specifier(typeSpecifier, specs)) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeAST*>* attributes = nullptr;

    parse_attribute_specifier_seq(attributes);

    sem->specifiers(typeSpecifier, &specs.specifiers);

    *it = new (pool) List(typeSpecifier);
    it = &(*it)->next;
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

    match(TokenKind::T_TEMPLATE, templateLoc);

    if (parse_type_name(name)) {
      auto qualifiedId = new (pool) QualifiedNameAST();
      qualifiedId->nestedNameSpecifier = nestedNameSpecifier;
      qualifiedId->templateLoc = templateLoc;
      qualifiedId->id = name;

      auto ast = new (pool) NamedTypeSpecifierAST();
      yyast = ast;

      ast->name = name;

      return true;
    }
  }

  rewind(start);

  NameAST* name = nullptr;

  if (!parse_type_name(name)) return false;

  Semantics::NameSem nameSem;

  sem->name(name, &nameSem);

  Symbol* typeSymbol = nullptr;

  if (checkTypes_) {
    typeSymbol =
        sem->scope()->unqualifiedLookup(nameSem.name, LookupOptions::kType);

    if (!typeSymbol) return false;
  }

  auto ast = new (pool) NamedTypeSpecifierAST();
  yyast = ast;

  ast->name = name;
  ast->symbol = typeSymbol;

  return true;
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
    case TokenKind::T___BUILTIN_VA_LIST: {
      auto ast = new (pool) VaListTypeSpecifierAST();
      yyast = ast;
      ast->specifierLoc = consumeToken();
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

bool Parser::parse_elaborated_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) {
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

  if (checkTypes_ && parsed && !ast->nestedNameSpecifier) {
    Semantics::NameSem nameSem;

    sem->name(ast->name, &nameSem);

    ast->symbol =
        sem->scope()->unqualifiedLookup(nameSem.name, LookupOptions::kType);
  }

  elaborated_type_specifiers_.emplace(
      start, std::tuple(currentLocation(), ast, parsed));

  return parsed;
}

bool Parser::parse_elaborated_type_specifier_helper(
    ElaboratedTypeSpecifierAST*& yyast, DeclSpecs& specs) {
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

      auto ast = new (pool) ElaboratedTypeSpecifierAST();
      yyast = ast;

      ast->classLoc = classLoc;
      ast->attributeList = attributes;
      ast->name = name;

      return true;
    }

    rewind(before_nested_name_specifier);

    SourceLocation identifierLoc;

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    auto id = new (pool) SimpleNameAST();
    name = id;

    id->identifierLoc = identifierLoc;

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

bool Parser::parse_elaborated_enum_specifier(
    ElaboratedTypeSpecifierAST*& yyast) {
  SourceLocation enumLoc;

  if (!match(TokenKind::T_ENUM, enumLoc)) return false;

  const auto saved = currentLocation();

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) rewind(saved);

  NameAST* name = nullptr;

  if (!parse_name_id(name)) return false;

  auto ast = new (pool) ElaboratedTypeSpecifierAST();
  yyast = ast;

  ast->classLoc = enumLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;

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

    Semantics::ExpressionSem expr;

    sem->expression(ast->expression, &expr);

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

    Semantics::ExpressionSem expr;

    sem->expression(ast->expression, &expr);

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

bool Parser::parse_init_declarator(InitDeclaratorAST*& yyast,
                                   const DeclSpecs& specs) {
  DeclaratorAST* declarator = nullptr;

  if (!parse_declarator(declarator)) return false;

  Semantics::DeclaratorSem decl{specs.specifiers};

  sem->declarator(declarator, &decl);

  const auto saved = currentLocation();

  InitializerAST* initializer = nullptr;

  if (!parse_declarator_initializer(initializer)) rewind(saved);

  auto ast = new (pool) InitDeclaratorAST();
  yyast = ast;

  ast->declarator = declarator;
  ast->initializer = initializer;

  if (decl.specifiers.isTypedef) {
    auto typedefSymbol = symbols->newTypedefSymbol(sem->scope(), decl.name);
    typedefSymbol->setType(decl.type);
    sem->scope()->add(typedefSymbol);
  } else if (auto functionDeclarator = getFunctionDeclarator(declarator)) {
    FunctionSymbol* functionSymbol = nullptr;
    functionSymbol = symbols->newFunctionSymbol(sem->scope(), decl.name);
    functionSymbol->setType(decl.type);
    sem->scope()->add(functionSymbol);
  } else {
    VariableSymbol* varSymbol = nullptr;
    varSymbol = symbols->newVariableSymbol(sem->scope(), decl.name);
    varSymbol->setType(decl.type);
    sem->scope()->add(varSymbol);
  }

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

        Semantics::ExpressionSem expr;

        sem->expression(expression, &expr);
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
  SourceLocation loc;

  if (match(TokenKind::T_CONST, loc)) {
    auto ast = new (pool) ConstQualifierAST();
    yyast = ast;
    ast->constLoc = loc;
    return true;
  } else if (match(TokenKind::T_VOLATILE, loc)) {
    auto ast = new (pool) VolatileQualifierAST();
    yyast = ast;
    ast->volatileLoc = loc;
    return true;
  } else if (match(TokenKind::T___RESTRICT, loc) ||
             match(TokenKind::T___RESTRICT__, loc)) {
    auto ast = new (pool) RestrictQualifierAST();
    yyast = ast;
    ast->restrictLoc = loc;
    return true;
  }
  return false;
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

  Semantics::DeclaratorSem decl{specs.specifiers};

  sem->declarator(declarator, &decl);

  auto ast = new (pool) TypeIdAST();
  yyast = ast;

  ast->typeSpecifierList = typeSpecifierList;
  ast->declarator = declarator;

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

        Semantics::ExpressionSem expr;

        sem->expression(arrayDeclarator->expression, &expr);

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

      Semantics::ExpressionSem expr;

      sem->expression(expression, &expr);

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

  if (!parse_parameter_declaration(declaration, /*templParam*/ false))
    return false;

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

bool Parser::parse_parameter_declaration(ParameterDeclarationAST*& yyast,
                                         bool templParam) {
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

  Semantics::DeclaratorSem decl{specs.specifiers};

  sem->declarator(ast->declarator, &decl);

  if (match(TokenKind::T_EQUAL, ast->equalLoc)) {
    if (!parse_initializer_clause(ast->expression, templParam))
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

bool Parser::parse_initializer_clause(ExpressionAST*& yyast, bool templParam) {
  BracedInitListAST* bracedInitList = nullptr;

  if (LA().is(TokenKind::T_LBRACE))
    return parse_braced_init_list(bracedInitList);

  ExprContext exprContext;
  exprContext.templParam = templParam;

  if (!parse_assignment_expression(yyast, exprContext)) return false;

  Semantics::ExpressionSem expr;

  sem->expression(yyast, &expr);

  return true;
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

  Semantics::ExpressionSem expr;

  sem->expression(yyast, &expr);

  return true;
}

bool Parser::parse_virt_specifier_seq() {
  if (!parse_virt_specifier()) return false;

  while (parse_virt_specifier()) {
    //
  }

  return true;
}

bool Parser::lookat_function_body() {
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

bool Parser::parse_function_body(FunctionBodyAST*& yyast) {
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

  if (!parse_compound_statement(ast->statement, skip))
    parse_error("expected a compound statement");

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

  const Name* enumName = name ? name->name : nullptr;
  Scope* enumScope = nullptr;

  if (classLoc) {
    auto* scopedEnumSymbol =
        symbols->newScopedEnumSymbol(sem->scope(), enumName);

    scopedEnumSymbol->setType(
        QualifiedType(types->scopedEnumType(scopedEnumSymbol)));

    sem->scope()->add(scopedEnumSymbol);

    enumScope = scopedEnumSymbol->scope();

    QualifiedType underlyingType;

    if (enumBase) {
      Semantics::SpecifiersSem specifiers;

      sem->specifiers(enumBase->typeSpecifierList, &specifiers);

      scopedEnumSymbol->setUnderlyingType(specifiers.type);
    }
  } else {
    auto* enumSymbol = symbols->newEnumSymbol(sem->scope(), enumName);

    enumSymbol->setType(QualifiedType(types->enumType(enumSymbol)));

    sem->scope()->add(enumSymbol);

    enumScope = enumSymbol->scope();
  }

  Semantics::ScopeContext scopeContext(sem.get(), enumScope);

  auto ast = new (pool) EnumSpecifierAST();
  yyast = ast;

  ast->enumLoc = enumLoc;
  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->name = name;
  ast->enumBase = enumBase;
  ast->lbraceLoc = lbraceLoc;
  ast->symbol = enumScope->owner();

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

  Semantics::NameSem nameSem;

  sem->name(name, &nameSem);

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

  Semantics::ExpressionSem expr;

  sem->expression(yyast->expression, &expr);

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

  Semantics::NameSem nameSem;

  sem->name(ast->name, &nameSem);

  auto symbol = symbols->newEnumeratorSymbol(sem->scope(), ast->name->name);
  symbol->setType(sem->scope()->owner()->type());
  sem->scope()->add(symbol);

  if (auto enumSymbol = dynamic_cast<EnumSymbol*>(sem->scope()->owner())) {
    auto enclosingNamespace = enumSymbol->enclosingNamespace();

    auto symbol = symbols->newEnumeratorSymbol(enclosingNamespace->scope(),
                                               ast->name->name);
    symbol->setType(enumSymbol->type());
    enclosingNamespace->scope()->add(symbol);
  }

  return true;
}

bool Parser::parse_using_enum_declaration(DeclarationAST*& yyast) {
  if (!match(TokenKind::T_USING)) return false;

  ElaboratedTypeSpecifierAST* enumSpecifier = nullptr;

  if (!parse_elaborated_enum_specifier(enumSpecifier)) return false;

  if (!match(TokenKind::T_SEMICOLON)) return false;

  return true;
}

bool Parser::parse_namespace_definition(DeclarationAST*& yyast) {
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

    namespaceSymbol = dynamic_cast<NamespaceSymbol*>(
        sem->scope()->find(id, LookupOptions::kNamespace));

    if (!namespaceSymbol) {
      namespaceSymbol = symbols->newNamespaceSymbol(sem->scope(), id);
      sem->scope()->add(namespaceSymbol);
    }

    while (match(TokenKind::T_COLON_COLON)) {
      SourceLocation inlineLoc;
      match(TokenKind::T_INLINE, inlineLoc);

      SourceLocation identifierLoc;
      expect(TokenKind::T_IDENTIFIER, identifierLoc);

      auto id = unit->identifier(identifierLoc);

      if (!id) continue;

      auto ns = dynamic_cast<NamespaceSymbol*>(
          namespaceSymbol->scope()->find(id, LookupOptions::kNamespace));

      if (!ns) {
        ns = symbols->newNamespaceSymbol(namespaceSymbol->scope(), id);
        if (inlineLoc) ns->setInline(true);

        namespaceSymbol->scope()->add(ns);
      }

      namespaceSymbol = ns;
    }
  } else if (parse_name_id(ast->name)) {
    Semantics::NameSem nameSem;

    sem->name(ast->name, &nameSem);

    namespaceName = nameSem.name;

    namespaceSymbol = dynamic_cast<NamespaceSymbol*>(
        sem->scope()->find(namespaceName, LookupOptions::kNamespace));
  }

  if (!namespaceSymbol) {
    namespaceSymbol = symbols->newNamespaceSymbol(sem->scope(), namespaceName);
    if (ast->inlineLoc) namespaceSymbol->setInline(true);

    sem->scope()->add(namespaceSymbol);
  }

  Semantics::ScopeContext scopeContext(sem.get(), namespaceSymbol->scope());

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
  const auto start = currentLocation();

  SourceLocation externLoc;

  List<AttributeAST*>* attributes = nullptr;

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

  Semantics::ExpressionSem expr;

  sem->expression(expression, &expr);

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
    auto [cursor, ast, parsed] = it->second;
    rewind(cursor);
    yyast = ast;
    return parsed;
  }

  SourceLocation classLoc;
  List<AttributeAST*>* attributeList = nullptr;
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

  Semantics::NameSem nameSem;

  sem->name(className, &nameSem);

  ClassSymbol* classSymbol = nullptr;

  for (auto s = sem->scope()->find(nameSem.name); s; s = s->next()) {
    if (s->name() != nameSem.name) continue;
    if (auto symbol = dynamic_cast<ClassSymbol*>(s)) {
      if (!symbol->isDefined()) classSymbol = symbol;
      break;
    }
  }

  if (!classSymbol) {
    classSymbol = symbols->newClassSymbol(sem->scope(), nameSem.name);
    classSymbol->setType(QualifiedType(types->classType(classSymbol)));
    sem->scope()->add(classSymbol);
  }

  classSymbol->setDefined(true);

  auto ast = new (pool) ClassSpecifierAST();
  yyast = ast;

  ast->classLoc = classLoc;
  ast->attributeList = attributeList;
  ast->name = className;
  ast->baseClause = baseClause;
  ast->lbraceLoc = lbraceLoc;
  ast->symbol = classSymbol;

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    if (!parse_class_body(ast->declarationList))
      parse_error("expected class body");

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  class_specifiers_.emplace(start,
                            std::make_tuple(currentLocation(), ast, true));

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

  if (!parse_type_name(name)) return false;

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

  if (parse_notypespec_function_definition(yyast, declSpecifierList, specs))
    return true;

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
    FunctionBodyAST* functionBody = nullptr;

    Semantics::DeclaratorSem decl{specs.specifiers};

    sem->declarator(declarator, &decl);

    const auto has_requires_clause = parse_requires_clause();

    bool has_virt_specifier_seq = false;

    if (!has_requires_clause)
      has_virt_specifier_seq = parse_virt_specifier_seq();

    if (lookat_function_body()) {
      if (!parse_function_body(functionBody))
        parse_error("expected function body");

      auto ast = new (pool) FunctionDefinitionAST();
      yyast = ast;

      ast->declSpecifierList = declSpecifierList;
      ast->declarator = declarator;
      ast->functionBody = functionBody;

      if (classDepth) pendingFunctionDefinitions_.push_back(ast);

      return true;
    }
  }

  rewind(after_decl_specs);

  List<DeclaratorAST*>* declaratorList = nullptr;

  if (!parse_member_declarator_list(declaratorList, specs))
    parse_error("expected a declarator");

  expect(TokenKind::T_SEMICOLON);

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

bool Parser::parse_member_declarator_list(List<DeclaratorAST*>*& yyast,
                                          const DeclSpecs& specs) {
  auto it = &yyast;

  DeclaratorAST* declarator = nullptr;

  if (!parse_member_declarator(declarator)) return false;

  Semantics::DeclaratorSem decl{specs.specifiers};

  sem->declarator(declarator, &decl);

  *it = new (pool) List(declarator);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    DeclaratorAST* declarator = nullptr;

    if (!parse_member_declarator(declarator))
      parse_error("expected a declarator");

    if (declarator) {
      Semantics::DeclaratorSem decl{specs.specifiers};

      sem->declarator(declarator, &decl);

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

    Semantics::ExpressionSem expr;

    sem->expression(expression, &expr);

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

  SourceLocation virtualLoc;
  SourceLocation accessLoc;

  if (match(TokenKind::T_VIRTUAL, virtualLoc))
    parse_access_specifier(accessLoc);
  else if (parse_access_specifier(accessLoc))
    match(TokenKind::T_VIRTUAL, virtualLoc);

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

bool Parser::parse_access_specifier(SourceLocation& loc) {
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

bool Parser::parse_ctor_initializer(CtorInitializerAST*& yyast) {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  auto ast = new (pool) CtorInitializerAST();
  yyast = ast;

  ast->colonLoc = colonLoc;

  if (!parse_mem_initializer_list(ast->memInitializerList))
    parse_error("expected a member intializer");

  return true;
}

bool Parser::parse_mem_initializer_list(List<MemInitializerAST*>*& yyast) {
  auto it = &yyast;

  MemInitializerAST* mem_initializer = nullptr;

  if (!parse_mem_initializer(mem_initializer)) return false;

  *it = new (pool) List(mem_initializer);
  it = &(*it)->next;

  while (match(TokenKind::T_COMMA)) {
    MemInitializerAST* mem_initializer = nullptr;

    if (!parse_mem_initializer(mem_initializer))
      parse_error("expected a member initializer");
    else {
      *it = new (pool) List(mem_initializer);
      it = &(*it)->next;
    }
  }

  return true;
}

bool Parser::parse_mem_initializer(MemInitializerAST*& yyast) {
  NameAST* name = nullptr;

  if (!parse_mem_initializer_id(name)) parse_error("expected an member id");

  if (LA().is(TokenKind::T_LBRACE)) {
    auto ast = new (pool) BracedMemInitializerAST();
    yyast = ast;

    ast->name = name;

    if (!parse_braced_init_list(ast->bracedInitList))
      parse_error("expected an initializer");

    match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

    return true;
  }

  auto ast = new (pool) ParenMemInitializerAST();
  yyast = ast;

  ast->name = name;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!match(TokenKind::T_RPAREN)) {
    if (!parse_expression_list(ast->expressionList))
      parse_error("expected an expression");

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  return true;
}

bool Parser::parse_mem_initializer_id(NameAST*& yyast) {
  const auto start = currentLocation();

  NameAST* name = nullptr;

  if (parse_class_or_decltype(name)) return true;

  rewind(start);

  if (!match(TokenKind::T_IDENTIFIER)) return false;

  return true;
}

bool Parser::parse_operator_function_id(NameAST*& yyast) {
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

bool Parser::parse_operator(TokenKind& op, SourceLocation& opLoc,
                            SourceLocation& openLoc, SourceLocation& closeLoc) {
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
      } else if (parse_greater_greater()) {
        op = TokenKind::T_GREATER_GREATER;
        return true;
      } else if (parse_greater_equal()) {
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

  if (parse_type_parameter(yyast) &&
      (LA().is(TokenKind::T_COMMA) || LA().is(TokenKind::T_GREATER)))
    return true;

  rewind(start);

  ParameterDeclarationAST* parameter = nullptr;

  if (parse_parameter_declaration(parameter, /*templParam*/ true)) {
    yyast = parameter;
    return true;
  }

  rewind(start);

  return parse_constraint_type_parameter(yyast);
}

bool Parser::parse_type_parameter(DeclarationAST*& yyast) {
  if (parse_template_type_parameter(yyast)) return true;

  if (parse_typename_type_parameter(yyast)) return true;

  return false;
}

bool Parser::parse_typename_type_parameter(DeclarationAST*& yyast) {
  SourceLocation classKeyLoc;

  if (!parse_type_parameter_key(classKeyLoc)) return false;

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool) TypenamePackTypeParameterAST();
    yyast = ast;
    ast->classKeyLoc = classKeyLoc;
    ast->ellipsisLoc = ellipsisLoc;
    match(TokenKind::T_IDENTIFIER, ast->identifierLoc);
    return true;
  }

  const auto saved = currentLocation();

  if ((LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_EQUAL)) ||
      LA().is(TokenKind::T_EQUAL)) {
    auto ast = new (pool) TypenameTypeParameterAST();
    yyast = ast;

    ast->classKeyLoc = classKeyLoc;

    match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

    expect(TokenKind::T_EQUAL, ast->equalLoc);

    if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

    return true;
  }

  auto ast = new (pool) TypenameTypeParameterAST();
  yyast = ast;

  ast->classKeyLoc = classKeyLoc;

  match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  return true;
}

bool Parser::parse_template_type_parameter(DeclarationAST*& yyast) {
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

  SourceLocation classsKeyLoc;

  if (!parse_type_parameter_key(classsKeyLoc))
    parse_error("expected a type parameter");

  const auto saved = currentLocation();

  if ((LA().is(TokenKind::T_IDENTIFIER) && LA(1).is(TokenKind::T_EQUAL)) ||
      LA().is(TokenKind::T_EQUAL)) {
    auto ast = new (pool) TemplateTypeParameterAST();
    yyast = ast;

    ast->templateLoc = templateLoc;
    ast->lessLoc = lessLoc;
    ast->templateParameterList = templateParameterList;
    ast->greaterLoc = greaterLoc;
    ast->classKeyLoc = classsKeyLoc;

    match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

    expect(TokenKind::T_EQUAL, ast->equalLoc);

    if (!parse_id_expression(ast->name))
      parse_error("expected an id-expression");

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

bool Parser::parse_constraint_type_parameter(DeclarationAST*& yyast) {
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

bool Parser::parse_type_parameter_key(SourceLocation& classKeyLoc) {
  if (!match(TokenKind::T_CLASS, classKeyLoc) &&
      !match(TokenKind::T_TYPENAME, classKeyLoc))
    return false;

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

  if (checkTypes_) {
    Semantics::NameSem nameSem;

    sem->name(name, &nameSem);

    auto conceptSymbol = dynamic_cast<ConceptSymbol*>(
        sem->scope()->unqualifiedLookup(nameSem.name, LookupOptions::kType));

    if (!conceptSymbol) {
      rewind(start);
      return false;
    }
  }

  SourceLocation lessLoc;

  if (match(TokenKind::T_LESS, lessLoc)) {
    SourceLocation greaterLoc;

    List<TemplateArgumentAST*>* templateArgumentList = nullptr;

    if (!parse_template_argument_list(templateArgumentList))
      parse_error("expected a template argument");

    expect(TokenKind::T_GREATER, greaterLoc);

    auto templateId = new (pool) TemplateNameAST();
    templateId->id = name;
    templateId->lessLoc = lessLoc;
    templateId->templateArgumentList = templateArgumentList;
    templateId->greaterLoc = greaterLoc;

    name = templateId;
  }

  return true;
}

bool Parser::parse_simple_template_id(NameAST*& yyast) {
  if (LA().isNot(TokenKind::T_IDENTIFIER) || LA(1).isNot(TokenKind::T_LESS))
    return false;

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

    ast->id = name;
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
    Semantics::ExpressionSem expr;

    sem->expression(expression, &expr);

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

bool Parser::parse_constraint_expression(ExpressionAST*& yyast) {
  ExprContext exprContext;
  return parse_logical_or_expression(yyast, exprContext);
}

bool Parser::parse_deduction_guide(DeclarationAST*& yyast) {
  SpecifierAST* explicitSpecifier = nullptr;

  const auto has_explicit_spec = parse_explicit_specifier(explicitSpecifier);

  NameAST* name = nullptr;

  if (!parse_name_id(name)) return false;

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

  Semantics::NameSem nameSem;

  sem->name(ast->name, &nameSem);

  ConceptSymbol* conceptSymbol = nullptr;

  conceptSymbol = symbols->newConceptSymbol(sem->scope(), nameSem.name);
  conceptSymbol->setType(QualifiedType(types->conceptType(conceptSymbol)));
  sem->scope()->add(conceptSymbol);

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
  SourceLocation typenameLoc;

  if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

  const auto after_nested_name_specifier = currentLocation();

  const auto has_template = match(TokenKind::T_TEMPLATE);

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

bool Parser::parse_function_try_block(FunctionBodyAST*& yyast) {
  SourceLocation tryLoc;

  if (!match(TokenKind::T_TRY, tryLoc)) return false;

  auto ast = new (pool) TryStatementFunctionBodyAST();
  yyast = ast;

  ast->tryLoc = tryLoc;

  if (LA().isNot(TokenKind::T_LBRACE)) {
    if (!parse_ctor_initializer(ast->ctorInitializer))
      parse_error("expected a ctor initializer");
  }

  if (!parse_compound_statement(ast->statement))
    parse_error("expected a compound statement");

  if (!parse_handler_seq(ast->handlerList))
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

    Semantics::ExpressionSem expr;

    sem->expression(expression, &expr);

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

  if (ast->functionBody->kind() == ASTKind::CompoundStatement) {
    auto functionBody =
        static_cast<CompoundStatementFunctionBodyAST*>(ast->functionBody);

    const auto saved = currentLocation();

    rewind(functionBody->statement->lbraceLoc.next());

    finish_compound_statement(functionBody->statement);

    rewind(saved);
  }
}

}  // namespace cxx
