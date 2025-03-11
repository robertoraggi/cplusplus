// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FRnewOM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <cxx/parser.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbol_instantiation.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbol_chain.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <ranges>
#include <unordered_set>

namespace cxx {

namespace {

inline constexpr struct {
  auto operator()(const StringLiteral*) const -> bool { return true; }
  auto operator()(auto value) const -> bool { return !!value; }
} to_bool;

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

}  // namespace

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
    (void)p->unit->changeDiagnosticsClient(previousClient);

    if (!committed) {
      p->rewind(loc);
    } else {
      client.reportTo(p->unit->diagnosticsClient());
    }
  }

  void commit() { committed = true; }
};

struct Parser::LoopParser {
  Parser* p;
  std::optional<SourceLocation> startLocation;
  bool recovering = false;

  LoopParser(const LoopParser&) = delete;
  auto operator=(const LoopParser&) -> LoopParser& = delete;

  explicit LoopParser(Parser* p) : p(p) {}

  ~LoopParser() {}

  void start() {
    auto loc = p->currentLocation();

    if (startLocation == loc) {
      if (!recovering) {
        recovering = true;
        p->parse_error("skip spurious token");
      }
      loc = p->consumeToken();
    } else {
      recovering = false;
    }

    startLocation = loc;
  }
};

struct Parser::ExprContext {
  bool templParam = false;
  bool templArg = false;
  bool isConstantEvaluated = false;
  bool inRequiresClause = false;
};

struct Parser::TemplateHeadContext {
  TemplateHeadContext(const TemplateHeadContext&) = delete;
  auto operator=(const TemplateHeadContext&) -> TemplateHeadContext& = delete;

  Parser* p;

  explicit TemplateHeadContext(Parser* p) : p(p) {
    ++p->templateParameterDepth_;
  }

  ~TemplateHeadContext() { --p->templateParameterDepth_; }
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

Parser::Parser(TranslationUnit* unit) : unit(unit), binder_(unit) {
  control_ = unit->control();
  diagnosticClient_ = unit->diagnosticsClient();
  cursor_ = 1;

  pool_ = unit->arena();

  moduleId_ = control_->getIdentifier("module");
  importId_ = control_->getIdentifier("import");
  finalId_ = control_->getIdentifier("final");
  overrideId_ = control_->getIdentifier("override");

  globalScope_ = unit->globalScope();
  setScope(globalScope_);

  // temporary workarounds to the gnu  until we have a proper
  // support for templates
  mark_maybe_template_name(control_->getIdentifier("__make_integer_seq"));
  mark_maybe_template_name(control_->getIdentifier("__remove_reference_t"));
  mark_maybe_template_name(control_->getIdentifier("__integer_pack"));

  template_names_.insert(control_->getIdentifier("_S_invoke"));
  template_names_.insert(control_->getIdentifier("__type_pack_element"));
}

Parser::~Parser() = default;

auto Parser::prec(TokenKind tk) -> Parser::Prec {
  switch (tk) {
    default:
      cxx_runtime_error(std::format("expected a binary operator, found {}",
                                    Token::spell(tk)));

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
  parse_error(std::format("expected '{}'", Token::spell(tk)));
  return false;
}

void Parser::operator()(UnitAST*& ast) { parse(ast); }

auto Parser::config() const -> const ParserConfiguration& { return config_; }

void Parser::setConfig(ParserConfiguration config) {
  config_ = std::move(config);
}

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

void Parser::parse_error(SourceLocation loc, std::string message) {
  unit->error(loc, std::move(message));
}

void Parser::warning(std::string message) {
  warning(currentLocation(), std::move(message));
}

void Parser::error(std::string message) { error(currentLocation(), message); }

void Parser::warning(SourceLocation loc, std::string message) {
  auto savedDiagnosticClient = unit->changeDiagnosticsClient(diagnosticClient_);
  unit->warning(loc, std::move(message));
  (void)unit->changeDiagnosticsClient(savedDiagnosticClient);
}

void Parser::error(SourceLocation loc, std::string message) {
  auto savedDiagnosticClient = unit->changeDiagnosticsClient(diagnosticClient_);
  unit->error(loc, std::move(message));
  (void)unit->changeDiagnosticsClient(savedDiagnosticClient);
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

auto Parser::parse_type_name(UnqualifiedIdAST*& yyast,
                             NestedNameSpecifierAST* nestedNameSpecifier,
                             bool isTemplateIntroduced) -> bool {
  auto lookat_simple_template_id = [&] {
    LookaheadParser lookahead{this};
    SimpleTemplateIdAST* templateId = nullptr;
    if (!parse_simple_template_id(templateId, nestedNameSpecifier,
                                  isTemplateIntroduced))
      return false;
    yyast = templateId;
    lookahead.commit();
    return true;
  };

  if (lookat_simple_template_id()) return true;

  NameIdAST* nameId = nullptr;
  if (!parse_name_id(nameId)) return false;

  yyast = nameId;
  return true;
}

auto Parser::parse_name_id(NameIdAST*& yyast) -> bool {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = make_node<NameIdAST>(pool_);
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_literal(ExpressionAST*& yyast) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_CHARACTER_LITERAL: {
      auto ast = make_node<CharLiteralExpressionAST>(pool_);
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal =
          static_cast<const CharLiteral*>(unit->literal(ast->literalLoc));

      auto prefix = ast->literal->components().prefix;

      if (prefix == "u8")
        ast->type = control_->getChar8Type();
      else if (prefix == "u")
        ast->type = control_->getChar16Type();
      else if (prefix == "U")
        ast->type = control_->getChar32Type();
      else if (prefix == "L")
        ast->type = control_->getWideCharType();
      else
        ast->type = control_->getCharType();

      return true;
    }

    case TokenKind::T_TRUE:
    case TokenKind::T_FALSE: {
      auto ast = make_node<BoolLiteralExpressionAST>(pool_);
      yyast = ast;

      const auto isTrue = lookat(TokenKind::T_TRUE);

      ast->literalLoc = consumeToken();
      ast->isTrue = isTrue;
      ast->type = control_->getBoolType();

      return true;
    }

    case TokenKind::T_INTEGER_LITERAL: {
      auto ast = make_node<IntLiteralExpressionAST>(pool_);
      yyast = ast;

      ast->literalLoc = consumeToken();

      ast->literal =
          static_cast<const IntegerLiteral*>(unit->literal(ast->literalLoc));

      const auto& components = ast->literal->components();

      if (components.isLongLong && components.isUnsigned)
        ast->type = control_->getUnsignedLongLongIntType();
      else if (components.isLongLong)
        ast->type = control_->getLongLongIntType();
      else if (components.isLong && components.isUnsigned)
        ast->type = control_->getUnsignedLongIntType();
      else if (components.isLong)
        ast->type = control_->getLongIntType();
      else if (components.isUnsigned)
        ast->type = control_->getUnsignedIntType();
      else
        ast->type = control_->getIntType();

      return true;
    }

    case TokenKind::T_FLOATING_POINT_LITERAL: {
      auto ast = make_node<FloatLiteralExpressionAST>(pool_);
      yyast = ast;

      ast->literalLoc = consumeToken();

      ast->literal =
          static_cast<const FloatLiteral*>(unit->literal(ast->literalLoc));

      const auto& components = ast->literal->components();

      if (components.isLongDouble)
        ast->type = control_->getLongDoubleType();
      else if (components.isDouble)
        ast->type = control_->getDoubleType();
      else if (components.isFloat)
        ast->type = control_->getFloatType();
      else
        ast->type = control_->getDoubleType();

      return true;
    }

    case TokenKind::T_NULLPTR: {
      auto ast = make_node<NullptrLiteralExpressionAST>(pool_);
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal = unit->tokenKind(ast->literalLoc);
      ast->type = control_->getNullptrType();

      return true;
    }

    case TokenKind::T_USER_DEFINED_STRING_LITERAL: {
      auto ast = make_node<UserDefinedStringLiteralExpressionAST>(pool_);
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

      auto ast = make_node<StringLiteralExpressionAST>(pool_);
      yyast = ast;

      ast->literalLoc = literalLoc;
      ast->literal =
          static_cast<const StringLiteral*>(unit->literal(literalLoc));

      if (unit->tokenKind(literalLoc) == TokenKind::T_STRING_LITERAL) {
        ast->type = control_->getBoundedArrayType(
            control_->add_const(control_->getCharType()),
            ast->literal->stringValue().size() + 1);

        ast->valueCategory = ValueCategory::kLValue;
      }

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

  auto ast = make_node<ModuleUnitAST>(pool_);
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
  auto ast = make_node<TranslationUnitAST>(pool_);
  yyast = ast;

  moduleUnit_ = false;

  auto it = &ast->declarationList;

  LoopParser loop(this);

  while (LA()) {
    if (shouldStopParsing()) break;

    loop.start();

    DeclarationAST* declaration = nullptr;

    if (!parse_declaration(declaration, BindingContext::kNamespace)) {
      parse_error("expected a declaration");
      continue;
    }

    if (declaration) {
      *it = make_list_node(pool_, declaration);
      it = &(*it)->next;
    }
  }
}

void Parser::parse_declaration_seq(List<DeclarationAST*>*& yyast) {
  auto it = &yyast;

  LoopParser loop(this);

  while (LA()) {
    if (shouldStopParsing()) break;

    if (lookat(TokenKind::T_RBRACE)) break;

    if (parse_maybe_module()) break;

    loop.start();

    DeclarationAST* declaration = nullptr;

    if (parse_declaration(declaration, BindingContext::kNamespace)) {
      if (declaration) {
        *it = make_list_node(pool_, declaration);
        it = &(*it)->next;
      }
    } else {
      parse_error("expected a declaration");
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

auto Parser::parse_completion(SourceLocation& loc) -> bool {
  // if already reported a completion, return false
  if (didAcceptCompletionToken_) return false;

  // if there is no completer, return false
  if (!config_.complete) return false;

  if (!match(TokenKind::T_CODE_COMPLETION, loc)) return false;

  didAcceptCompletionToken_ = true;

  return true;
}

auto Parser::parse_primary_expression(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  UnqualifiedIdAST* name = nullptr;

  if (parse_builtin_call_expression(yyast, ctx)) {
    return true;
  } else if (parse_builtin_offsetof_expression(yyast, ctx)) {
    return true;
  } else if (parse_this_expression(yyast)) {
    return true;
  } else if (parse_literal(yyast)) {
    return true;
  } else if (parse_lambda_expression(yyast)) {
    return true;
  } else if (parse_requires_expression(yyast)) {
    return true;
  } else if (lookat(TokenKind::T_LPAREN, TokenKind::T_RPAREN)) {
    return false;
  } else if (parse_fold_expression(yyast, ctx)) {
    return true;
  } else if (parse_nested_expession(yyast, ctx)) {
    return true;
  } else if (IdExpressionAST* idExpression = nullptr; parse_id_expression(
                 idExpression, ctx.inRequiresClause
                                   ? IdExpressionContext::kRequiresClause
                                   : IdExpressionContext::kExpression)) {
    yyast = idExpression;
    return true;
  } else if (parse_splicer_expression(yyast, ctx)) {
    return true;
  } else {
    return false;
  }
}

auto Parser::parse_splicer(SplicerAST*& yyast) -> bool {
  if (!config_.reflect) return false;

  if (!lookat(TokenKind::T_LBRACKET, TokenKind::T_COLON)) return false;

  auto ast = make_node<SplicerAST>(pool_);
  yyast = ast;
  ast->lbracketLoc = consumeToken();
  ast->colonLoc = consumeToken();
  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);
  std::optional<ConstValue> value;
  if (!parse_constant_expression(ast->expression, value)) {
    parse_error("expected a constant expression");
  }
  expect(TokenKind::T_COLON, ast->secondColonLoc);
  expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  return true;
}

auto Parser::parse_splicer_expression(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  if (!config_.reflect) return false;

  SplicerAST* splicer = nullptr;
  if (!parse_splicer(splicer)) return false;
  auto ast = make_node<SpliceExpressionAST>(pool_);
  yyast = ast;
  ast->splicer = splicer;
  return true;
}

auto Parser::parse_reflect_expression(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  if (!config_.reflect) return false;

  SourceLocation caretLoc;

  if (!match(TokenKind::T_CARET, caretLoc)) return false;

  auto lookat_namespace_name = [&] {
    LookaheadParser lookahead{this};
    SourceLocation identifierLoc;
    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;
    auto identifier = unit->identifier(identifierLoc);
    auto symbol = symbol_cast<NamespaceSymbol>(Lookup{scope()}(identifier));
    if (!symbol) return false;
    lookahead.commit();

    auto ast = make_node<NamespaceReflectExpressionAST>(pool_);
    yyast = ast;
    ast->caretLoc = caretLoc;
    ast->identifierLoc = identifierLoc;
    ast->identifier = identifier;
    ast->symbol = symbol;
    return true;
  };

  auto lookat_type_id = [&] {
    LookaheadParser lookahead{this};
    TypeIdAST* typeId = nullptr;
    if (!parse_type_id(typeId)) return false;
    lookahead.commit();

    auto ast = make_node<TypeIdReflectExpressionAST>(pool_);
    yyast = ast;
    ast->caretLoc = caretLoc;
    ast->typeId = typeId;
    return true;
  };

  auto lookat_expression = [&] {
    LookaheadParser lookahead{this};
    ExpressionAST* expression = nullptr;
    if (!parse_cast_expression(expression, ctx)) return false;
    lookahead.commit();

    auto ast = make_node<ReflectExpressionAST>(pool_);
    yyast = ast;
    ast->caretLoc = caretLoc;
    ast->expression = expression;
    return true;
  };

  if (SourceLocation scopeLoc; match(TokenKind::T_COLON_COLON, scopeLoc)) {
    auto ast = make_node<GlobalScopeReflectExpressionAST>(pool_);
    yyast = ast;
    ast->caretLoc = caretLoc;
    ast->scopeLoc = scopeLoc;
    return true;
  }

  if (lookat_namespace_name()) return true;
  if (lookat_type_id()) return true;
  if (lookat_expression()) return true;

  parse_error("expected a reflacted expression");

  return true;
}

auto Parser::parse_id_expression(IdExpressionAST*& yyast,
                                 IdExpressionContext ctx) -> bool {
  LookaheadParser lookahead{this};

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  const auto inRequiresClause = ctx == IdExpressionContext::kRequiresClause;

  UnqualifiedIdAST* unqualifiedId = nullptr;
  if (!parse_unqualified_id(unqualifiedId, nestedNameSpecifier,
                            isTemplateIntroduced, inRequiresClause))
    return false;

  lookahead.commit();

  auto ast = make_node<IdExpressionAST>(pool_);
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->unqualifiedId = unqualifiedId;
  ast->isTemplateIntroduced = isTemplateIntroduced;

  if (unqualifiedId) {
    auto name = convertName(unqualifiedId);
    const Name* componentName = name;
    if (auto templateId = name_cast<TemplateId>(name))
      componentName = templateId->name();
    ast->symbol = Lookup{scope()}(nestedNameSpecifier, componentName);
  }

  if (ctx == IdExpressionContext::kExpression) {
    if (ast->symbol) {
      if (auto conceptSymbol = symbol_cast<ConceptSymbol>(ast->symbol)) {
        ast->type = control_->getBoolType();
        ast->valueCategory = ValueCategory::kPrValue;
      } else {
        ast->type = control_->remove_reference(ast->symbol->type());

        if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
          ast->valueCategory = ValueCategory::kPrValue;
        } else {
          ast->valueCategory = ValueCategory::kLValue;
        }
      }
    }
  }

  return true;
}

auto Parser::parse_unqualified_id(UnqualifiedIdAST*& yyast,
                                  NestedNameSpecifierAST* nestedNameSpecifier,
                                  bool isTemplateIntroduced,
                                  bool inRequiresClause) -> bool {
  if (SourceLocation tildeLoc; match(TokenKind::T_TILDE, tildeLoc)) {
    if (DecltypeSpecifierAST* decltypeSpecifier = nullptr;
        parse_decltype_specifier(decltypeSpecifier)) {
      auto decltypeName = make_node<DecltypeIdAST>(pool_);
      decltypeName->decltypeSpecifier = decltypeSpecifier;

      auto ast = make_node<DestructorIdAST>(pool_);
      yyast = ast;
      ast->tildeLoc = tildeLoc;
      ast->id = decltypeName;

      return true;
    }

    UnqualifiedIdAST* name = nullptr;
    if (!parse_type_name(name, nestedNameSpecifier, isTemplateIntroduced))
      return false;

    auto ast = make_node<DestructorIdAST>(pool_);
    yyast = ast;
    ast->tildeLoc = tildeLoc;
    ast->id = name;

    return true;
  }

  auto lookat_template_id = [&] {
    LookaheadParser lookahead{this};
    if (!parse_template_id(yyast, nestedNameSpecifier, isTemplateIntroduced))
      return false;
    lookahead.commit();
    return true;
  };

  if (lookat_template_id()) return true;

  if (LiteralOperatorIdAST* literalOperatorName = nullptr;
      parse_literal_operator_id(literalOperatorName)) {
    yyast = literalOperatorName;
    return true;
  }

  if (ConversionFunctionIdAST* conversionFunctionName = nullptr;
      parse_conversion_function_id(conversionFunctionName)) {
    yyast = conversionFunctionName;
    return true;
  }

  if (OperatorFunctionIdAST* functionOperatorName = nullptr;
      parse_operator_function_id(functionOperatorName)) {
    yyast = functionOperatorName;
    return true;
  }

  NameIdAST* nameId = nullptr;
  if (!parse_name_id(nameId)) return false;

  yyast = nameId;
  return true;
}

void Parser::parse_optional_nested_name_specifier(
    NestedNameSpecifierAST*& yyast, NestedNameSpecifierContext ctx) {
  LookaheadParser lookahead(this);
  if (!parse_nested_name_specifier(yyast, ctx)) return;
  lookahead.commit();
}

auto Parser::parse_decltype_nested_name_specifier(
    NestedNameSpecifierAST*& yyast, NestedNameSpecifierContext ctx) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation decltypeLoc;
  if (!match(TokenKind::T_DECLTYPE, decltypeLoc)) return false;
  if (!lookat(TokenKind::T_LPAREN)) return false;
  if (!parse_skip_balanced()) return false;
  if (!lookat(TokenKind::T_COLON_COLON)) return false;

  rewind(decltypeLoc);

  DecltypeSpecifierAST* decltypeSpecifier = nullptr;
  if (!parse_decltype_specifier(decltypeSpecifier)) return false;

  SourceLocation scopeLoc;
  if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;

  lookahead.commit();

  auto ast = make_node<DecltypeNestedNameSpecifierAST>(pool_);
  yyast = ast;

  ast->decltypeSpecifier = decltypeSpecifier;
  ast->scopeLoc = scopeLoc;

  if (decltypeSpecifier) {
    if (auto classsType = type_cast<ClassType>(decltypeSpecifier->type)) {
      ast->symbol = classsType->symbol();
    } else if (auto enumType = type_cast<EnumType>(decltypeSpecifier->type)) {
      ast->symbol = enumType->symbol();
    } else if (auto scopedEnumType =
                   type_cast<ScopedEnumType>(decltypeSpecifier->type)) {
      ast->symbol = scopedEnumType->symbol();
    }
  }

  return true;
}

auto Parser::parse_type_nested_name_specifier(NestedNameSpecifierAST*& yyast,
                                              NestedNameSpecifierContext ctx)
    -> bool {
  if (!lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON_COLON)) return false;

  auto identifierLoc = consumeToken();
  auto identifier = unit->identifier(identifierLoc);
  auto scopeLoc = consumeToken();
  auto symbol = Lookup{scope()}.lookupType(yyast, identifier);

  auto ast = make_node<SimpleNestedNameSpecifierAST>(pool_);
  ast->nestedNameSpecifier = yyast;
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->identifier = identifier;
  ast->scopeLoc = scopeLoc;
  ast->symbol = symbol;

  return true;
}

namespace {
struct IsReferencingTemplateParameter {
  Parser& p;
  int depth = 0;
  int index = 0;

  auto operator()(TypeTemplateArgumentAST* ast) const -> bool {
    auto typeId = ast->typeId;
    if (!typeId) return false;

    if (checkTypeParam(typeId->type)) return true;

    return false;
  }

  [[nodiscard]] auto checkTypeParam(const Type* type) const -> bool {
    auto typeParam = type_cast<TypeParameterType>(type);
    if (!typeParam) return false;

    auto typeParamSymbol = typeParam->symbol();
    if (typeParamSymbol->depth() != depth) return false;
    if (typeParamSymbol->index() != index) return false;

    return true;
  }

  auto operator()(ExpressionTemplateArgumentAST* ast) const -> bool {
    // ### TODO
    return false;
  }
};
}  // namespace

auto Parser::parse_template_nested_name_specifier(
    NestedNameSpecifierAST*& yyast, NestedNameSpecifierContext ctx, int depth)
    -> bool {
  LookaheadParser lookahead{this};

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  SimpleTemplateIdAST* templateId = nullptr;
  if (!parse_simple_template_id(templateId, yyast, isTemplateIntroduced))
    return false;

  SourceLocation scopeLoc;
  if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;

  lookahead.commit();

  auto ast = make_node<TemplateNestedNameSpecifierAST>(pool_);
  ast->nestedNameSpecifier = yyast;
  yyast = ast;

  ast->templateLoc = templateLoc;
  ast->templateId = templateId;
  ast->scopeLoc = scopeLoc;
  ast->isTemplateIntroduced = isTemplateIntroduced;

  if (ctx == NestedNameSpecifierContext::kDeclarative) {
    bool isReferencingPrimaryTemplate = true;

    for (int index = 0; auto arg : ListView{templateId->templateArgumentList}) {
      if (!visit(IsReferencingTemplateParameter{*this, depth, index}, arg)) {
        isReferencingPrimaryTemplate = false;
        break;
      }
      ++index;
    }

    if (isReferencingPrimaryTemplate) {
      ast->symbol = templateId->primaryTemplateSymbol;
    }
  }

  if (!ast->symbol) {
    ast->symbol = instantiate(templateId);
  }

  return true;
}

auto Parser::parse_nested_name_specifier(NestedNameSpecifierAST*& yyast,
                                         NestedNameSpecifierContext ctx)
    -> bool {
  if (SourceLocation scopeLoc; match(TokenKind::T_COLON_COLON, scopeLoc)) {
    auto ast = make_node<GlobalNestedNameSpecifierAST>(pool_);
    yyast = ast;
    ast->scopeLoc = scopeLoc;
    ast->symbol = globalScope_->owner();
  } else if (parse_decltype_nested_name_specifier(yyast, ctx)) {
    //
  }

  int depth = 0;

  while (true) {
    if (parse_type_nested_name_specifier(yyast, ctx)) {
      continue;
    }

    if (parse_template_nested_name_specifier(yyast, ctx, depth)) {
      ++depth;
      continue;
    }

    break;
  }

  const auto parsed = yyast != nullptr;

  return parsed;
}

auto Parser::parse_lambda_expression(ExpressionAST*& yyast) -> bool {
  if (lookat(TokenKind::T_LBRACKET, TokenKind::T_LBRACKET)) return false;
  if (lookat(TokenKind::T_LBRACKET, TokenKind::T_COLON)) return false;
  if (!lookat(TokenKind::T_LBRACKET)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  TemplateHeadContext templateHeadContext{this};

  auto ast = make_node<LambdaExpressionAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_LBRACKET, ast->lbracketLoc);

  binder_.bind(ast);

  if (!match(TokenKind::T_RBRACKET, ast->rbracketLoc)) {
    if (!parse_lambda_capture(ast->captureDefaultLoc, ast->captureList)) {
      parse_error("expected a lambda capture");
    }

    expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  }

  if (ast->captureDefaultLoc)
    ast->captureDefault = unit->tokenKind(ast->captureDefaultLoc);

  Binder::ScopeGuard templateScopeGuard{&binder_};

  if (match(TokenKind::T_LESS, ast->lessLoc)) {
    parse_template_parameter_list(ast->templateParameterList);

    expect(TokenKind::T_GREATER, ast->greaterLoc);

    (void)parse_requires_clause(ast->templateRequiresClause);
  }

  if (match(TokenKind::T_LPAREN, ast->lparenLoc)) {
    if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
      if (!parse_parameter_declaration_clause(
              ast->parameterDeclarationClause)) {
        parse_error("expected a parameter declaration clause");
      }

      expect(TokenKind::T_RPAREN, ast->rparenLoc);
    }

    parse_optional_attribute_specifier_seq(ast->gnuAtributeList,
                                           AllowedAttributes::kGnuAttribute);

    (void)parse_lambda_specifier_seq(ast->lambdaSpecifierList, ast->symbol);

    (void)parse_noexcept_specifier(ast->exceptionSpecifier);

    (void)parse_trailing_return_type(ast->trailingReturnType);

    parse_optional_attribute_specifier_seq(ast->attributeList,
                                           AllowedAttributes::kAll);

    (void)parse_requires_clause(ast->requiresClause);
  }

  if (!lookat(TokenKind::T_LBRACE)) return false;

  binder_.complete(ast);

  if (!parse_compound_statement(ast->statement)) {
    parse_error("expected a compound statement");
  }

  return true;
}

auto Parser::parse_lambda_specifier_seq(List<LambdaSpecifierAST*>*& yyast,
                                        LambdaSymbol* symbol) -> bool {
  yyast = nullptr;

  auto it = &yyast;

  while (true) {
    if (lookat(TokenKind::T_CONSTEVAL))
      symbol->setConsteval(true);
    else if (lookat(TokenKind::T_CONSTEXPR))
      symbol->setConstexpr(true);
    else if (lookat(TokenKind::T_MUTABLE))
      symbol->setMutable(true);
    else if (lookat(TokenKind::T_STATIC))
      symbol->setStatic(true);
    else
      break;

    auto specifier = make_node<LambdaSpecifierAST>(pool_);
    specifier->specifierLoc = consumeToken();
    specifier->specifier = unit->tokenKind(specifier->specifierLoc);
    *it = make_list_node(pool_, specifier);
    it = &(*it)->next;
  }
  return yyast != nullptr;
}

auto Parser::parse_lambda_capture(SourceLocation& captureDefaultLoc,
                                  List<LambdaCaptureAST*>*& captureList)
    -> bool {
  if (parse_capture_default(captureDefaultLoc)) {
    if (SourceLocation commaLoc; match(TokenKind::T_COMMA, commaLoc)) {
      if (!parse_capture_list(captureList)) parse_error("expected a capture");
    }

    return true;
  }

  return parse_capture_list(captureList);
}

auto Parser::parse_capture_default(SourceLocation& opLoc) -> bool {
  if (!LA().isOneOf(TokenKind::T_AMP, TokenKind::T_EQUAL)) return false;
  if (!LA(1).isOneOf(TokenKind::T_COMMA, TokenKind::T_RBRACKET)) return false;

  opLoc = consumeToken();

  return true;
}

auto Parser::parse_capture_list(List<LambdaCaptureAST*>*& yyast) -> bool {
  auto it = &yyast;

  LambdaCaptureAST* capture = nullptr;

  if (!parse_capture(capture)) return false;

  if (capture) {
    *it = make_list_node(pool_, capture);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    LambdaCaptureAST* capture = nullptr;

    if (!parse_capture(capture)) parse_error("expected a capture");

    if (capture) {
      *it = make_list_node(pool_, capture);
      it = &(*it)->next;
    }
  }

  return true;
}

auto Parser::parse_capture(LambdaCaptureAST*& yyast) -> bool {
  if (parse_simple_capture(yyast)) return true;
  if (parse_init_capture(yyast)) return true;
  return false;
}

auto Parser::parse_simple_capture(LambdaCaptureAST*& yyast) -> bool {
  if (SourceLocation thisLoc; match(TokenKind::T_THIS, thisLoc)) {
    auto ast = make_node<ThisLambdaCaptureAST>(pool_);
    yyast = ast;

    ast->thisLoc = thisLoc;

    return true;
  } else if (lookat(TokenKind::T_STAR, TokenKind::T_THIS)) {
    auto ast = make_node<DerefThisLambdaCaptureAST>(pool_);
    yyast = ast;

    ast->starLoc = consumeToken();
    ast->thisLoc = consumeToken();

    return true;
  }

  auto lookat_simple_capture = [&] {
    LookaheadParser lookahead{this};

    SourceLocation identifierLoc;
    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    SourceLocation ellipsisLoc;
    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    if (!LA().isOneOf(TokenKind::T_COMMA, TokenKind::T_RBRACKET)) return false;

    auto ast = make_node<SimpleLambdaCaptureAST>(pool_);
    yyast = ast;

    ast->identifierLoc = identifierLoc;
    ast->identifier = unit->identifier(ast->identifierLoc);
    ast->ellipsisLoc = ellipsisLoc;

    lookahead.commit();

    return true;
  };

  if (lookat_simple_capture()) return true;

  LookaheadParser lookahead{this};

  SourceLocation ampLoc;
  if (!match(TokenKind::T_AMP, ampLoc)) return false;

  SourceLocation identifierLoc;
  expect(TokenKind::T_IDENTIFIER, identifierLoc);

  SourceLocation ellipsisLoc;
  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  if (!LA().isOneOf(TokenKind::T_COMMA, TokenKind::T_RBRACKET)) return false;

  lookahead.commit();

  auto ast = make_node<RefLambdaCaptureAST>(pool_);
  yyast = ast;

  ast->ampLoc = ampLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);
  ast->ellipsisLoc = ellipsisLoc;

  return true;
}

auto Parser::parse_init_capture(LambdaCaptureAST*& yyast) -> bool {
  auto lookat_init_capture = [&] {
    LookaheadParser lookahead{this};

    SourceLocation ampLoc;
    match(TokenKind::T_AMP, ampLoc);

    if (LA().isOneOf(TokenKind::T_DOT_DOT_DOT, TokenKind::T_IDENTIFIER))
      return true;

    return false;
  };

  if (!lookat_init_capture()) return false;

  if (SourceLocation ampLoc; match(TokenKind::T_AMP, ampLoc)) {
    SourceLocation ellipsisLoc;

    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    SourceLocation identifierLoc;

    expect(TokenKind::T_IDENTIFIER, identifierLoc);

    ExpressionAST* initializer = nullptr;

    if (!parse_initializer(initializer, ExprContext{})) {
      parse_error("expected an initializer");
    }

    auto ast = make_node<RefInitLambdaCaptureAST>(pool_);
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

  expect(TokenKind::T_IDENTIFIER, identifierLoc);

  ExpressionAST* initializer = nullptr;

  if (!parse_initializer(initializer, ExprContext{})) {
    parse_error("expected an initializer");
  }

  auto ast = make_node<InitLambdaCaptureAST>(pool_);
  yyast = ast;

  ast->ellipsisLoc = ellipsisLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);
  ast->initializer = initializer;

  return true;
}

auto Parser::parse_left_fold_expression(ExpressionAST*& yyast,
                                        const ExprContext& ctx) -> bool {
  if (!lookat(TokenKind::T_LPAREN, TokenKind::T_DOT_DOT_DOT)) return false;

  auto ast = make_node<LeftFoldExpressionAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  expect(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  if (!parse_fold_operator(ast->opLoc, ast->op)) {
    parse_error("expected fold operator");
  }

  if (!parse_cast_expression(ast->expression, ctx)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_this_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation thisLoc;

  if (!match(TokenKind::T_THIS, thisLoc)) return false;

  auto ast = make_node<ThisExpressionAST>(pool_);
  yyast = ast;
  ast->thisLoc = thisLoc;
  ast->valueCategory = ValueCategory::kPrValue;

  check(ast);

  return true;
}

auto Parser::parse_nested_expession(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  if (lookat(TokenKind::T_LBRACE)) {
    auto ast = make_node<NestedStatementExpressionAST>(pool_);
    yyast = ast;

    ast->lparenLoc = lparenLoc;

    if (!parse_compound_statement(ast->statement)) {
      parse_error("expected a compound statement");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
    return ast;
  }

  auto ast = make_node<NestedExpressionAST>(pool_);
  yyast = ast;

  ast->lparenLoc = lparenLoc;

  parse_expression(ast->expression, ctx);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  check(ast);

  return true;
}

auto Parser::parse_fold_expression(ExpressionAST*& yyast,
                                   const ExprContext& ctx) -> bool {
  if (parse_left_fold_expression(yyast, ctx)) return true;

  SourceLocation lparenLoc;
  ExpressionAST* expression = nullptr;
  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;
  SourceLocation ellipsisLoc;

  auto lookat_fold_expression = [&] {
    LookaheadParser lookahead{this};
    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;
    if (!parse_cast_expression(expression, ctx)) return false;
    if (!parse_fold_operator(opLoc, op)) return false;
    if (!match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) return false;
    lookahead.commit();
    return true;
  };

  if (!lookat_fold_expression()) return false;

  if (SourceLocation rparenLoc; match(TokenKind::T_RPAREN, rparenLoc)) {
    auto ast = make_node<RightFoldExpressionAST>(pool_);
    yyast = ast;

    ast->lparenLoc = lparenLoc;
    ast->expression = expression;
    ast->opLoc = opLoc;
    ast->op = op;
    ast->ellipsisLoc = ellipsisLoc;
    ast->rparenLoc = rparenLoc;

    return true;
  }

  auto ast = make_node<FoldExpressionAST>(pool_);
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->leftExpression = expression;
  ast->opLoc = opLoc;
  ast->op = op;
  ast->ellipsisLoc = ellipsisLoc;

  if (!parse_fold_operator(ast->foldOpLoc, ast->foldOp)) {
    parse_error("expected a fold operator");
  }

  if (!parse_cast_expression(ast->rightExpression, ctx)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_fold_operator(SourceLocation& loc, TokenKind& op) -> bool {
  loc = currentLocation();

  switch (TokenKind(LA())) {
    case TokenKind::T_GREATER: {
      if (parse_greater_greater()) {
        op = TokenKind::T_GREATER_GREATER;
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

  auto ast = make_node<RequiresExpressionAST>(pool_);
  yyast = ast;

  ast->requiresLoc = requiresLoc;

  if (!lookat(TokenKind::T_LBRACE)) {
    if (!parse_requirement_parameter_list(
            ast->lparenLoc, ast->parameterDeclarationClause, ast->rparenLoc)) {
      parse_error("expected a requirement parameter");
    }
  }

  expect(TokenKind::T_LBRACE, ast->lbraceLoc);
  parse_requirement_seq(ast->requirementList);
  expect(TokenKind::T_RBRACE, ast->rbraceLoc);

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

void Parser::parse_requirement_seq(List<RequirementAST*>*& yyast) {
  auto it = &yyast;

  bool skipping = false;

  RequirementAST* requirement = nullptr;

  parse_requirement(requirement);

  *it = make_list_node(pool_, requirement);
  it = &(*it)->next;

  LoopParser loop(this);

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    RequirementAST* requirement = nullptr;
    parse_requirement(requirement);

    *it = make_list_node(pool_, requirement);
    it = &(*it)->next;
  }
}

void Parser::parse_requirement(RequirementAST*& yyast) {
  if (parse_nested_requirement(yyast)) return;
  if (parse_compound_requirement(yyast)) return;
  if (parse_type_requirement(yyast)) return;
  parse_simple_requirement(yyast);
}

void Parser::parse_simple_requirement(RequirementAST*& yyast) {
  ExpressionAST* expression = nullptr;

  parse_expression(expression, ExprContext{});

  SourceLocation semicolonLoc;

  expect(TokenKind::T_SEMICOLON, semicolonLoc);

  auto ast = make_node<SimpleRequirementAST>(pool_);
  yyast = ast;

  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;
}

auto Parser::parse_type_requirement(RequirementAST*& yyast) -> bool {
  SourceLocation typenameLoc;

  if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

  auto ast = make_node<TypeRequirementAST>(pool_);
  yyast = ast;

  parse_optional_nested_name_specifier(
      ast->nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  if (!parse_type_name(ast->unqualifiedId, ast->nestedNameSpecifier,
                       isTemplateIntroduced)) {
    parse_error("expected a type name");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_compound_requirement(RequirementAST*& yyast) -> bool {
  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  ExpressionAST* expression = nullptr;

  parse_expression(expression, ExprContext{});

  SourceLocation rbraceLoc;

  expect(TokenKind::T_RBRACE, rbraceLoc);

  auto ast = make_node<CompoundRequirementAST>(pool_);
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

  auto ast = make_node<NestedRequirementAST>(pool_);
  yyast = ast;

  ast->requiresLoc = requiresLoc;

  if (!parse_constraint_expression(ast->expression)) {
    parse_error("expected an expression");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_postfix_expression(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  if (!parse_start_of_postfix_expression(yyast, ctx)) return false;

  while (true) {
    LookaheadParser lookahead{this};

    if (parse_member_expression(yyast)) {
      //
    } else if (parse_subscript_expression(yyast, ctx)) {
      //
    } else if (parse_call_expression(yyast, ctx)) {
      //
    } else if (parse_postincr_expression(yyast, ctx)) {
      //
    } else {
      break;
    }

    lookahead.commit();
  }

  return true;
}

auto Parser::parse_start_of_postfix_expression(ExpressionAST*& yyast,
                                               const ExprContext& ctx) -> bool {
  if (parse_va_arg_expression(yyast, ctx))
    return true;
  else if (parse_cpp_cast_expression(yyast, ctx))
    return true;
  else if (parse_typeid_expression(yyast, ctx))
    return true;
  else if (parse_typename_expression(yyast, ctx))
    return true;
  else if (parse_cpp_type_cast_expression(yyast, ctx))
    return true;
  else if (parse_builtin_bit_cast_expression(yyast, ctx))
    return true;
  else
    return parse_primary_expression(yyast, ctx);
}

auto Parser::parse_member_expression(ExpressionAST*& yyast) -> bool {
  SourceLocation accessLoc;

  if (!match(TokenKind::T_DOT, accessLoc) &&
      !match(TokenKind::T_MINUS_GREATER, accessLoc)) {
    return false;
  }

  auto lookat_splice_member = [&] {
    LookaheadParser lookahead{this};
    SourceLocation templateLoc;
    const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);
    SplicerAST* splicer = nullptr;
    if (!parse_splicer(splicer)) return false;
    lookahead.commit();

    auto ast = make_node<SpliceMemberExpressionAST>(pool_);
    ast->baseExpression = yyast;
    ast->accessLoc = accessLoc;
    ast->accessOp = unit->tokenKind(ast->accessLoc);
    ast->templateLoc = templateLoc;
    ast->isTemplateIntroduced = isTemplateIntroduced;
    ast->splicer = splicer;
    yyast = ast;

    return true;
  };

  if (lookat_splice_member()) return true;

  auto ast = make_node<MemberExpressionAST>(pool_);
  ast->baseExpression = yyast;
  ast->accessLoc = accessLoc;
  ast->accessOp = unit->tokenKind(accessLoc);

  parse_optional_nested_name_specifier(
      ast->nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

  ast->isTemplateIntroduced = match(TokenKind::T_TEMPLATE, ast->templateLoc);

  if (SourceLocation completionLoc; parse_completion(completionLoc)) {
    if (ast->baseExpression) {
      // test if the base expression has a type
      auto objectType = ast->baseExpression->type;

      // trigger the completion
      config_.complete(MemberCompletionContext{
          .objectType = objectType,
          .accessOp = ast->accessOp,
      });
    }
  }

  if (!parse_unqualified_id(ast->unqualifiedId, ast->nestedNameSpecifier,
                            ast->isTemplateIntroduced,
                            /*inRequiresClause*/ false))
    parse_error("expected an unqualified id");

  check_member_expression(ast);

  yyast = ast;

  return true;
}

void Parser::check_member_expression(MemberExpressionAST* ast) {
  if (check_pseudo_destructor_access(ast)) return;
  if (check_member_access(ast)) return;
}

auto Parser::check_member_access(MemberExpressionAST* ast) -> bool {
  const Type* objectType = ast->baseExpression->type;
  auto cv1 = strip_cv(objectType);

  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    auto pointerType = type_cast<PointerType>(objectType);
    if (!pointerType) return false;

    objectType = pointerType->elementType();
    cv1 = strip_cv(objectType);
  }

  auto classType = type_cast<ClassType>(objectType);
  if (!classType) return false;

  auto memberName = convertName(ast->unqualifiedId);

  auto classSymbol = classType->symbol();

  auto symbol =
      Lookup{scope()}.qualifiedLookup(classSymbol->scope(), memberName);

  ast->symbol = symbol;

  if (symbol) {
    ast->type = symbol->type();

    if (symbol->isEnumerator()) {
      ast->valueCategory = ValueCategory::kPrValue;
    } else {
      if (is_lvalue(ast->baseExpression)) {
        ast->valueCategory = ValueCategory::kLValue;
      } else {
        ast->valueCategory = ValueCategory::kXValue;
      }

      if (auto field = symbol_cast<FieldSymbol>(symbol);
          field && !field->isStatic()) {
        auto cv2 = strip_cv(ast->type);

        if (is_volatile(cv1) || is_volatile(cv2))
          ast->type = control_->add_volatile(ast->type);

        if (!field->isMutable() && (is_const(cv1) || is_const(cv2)))
          ast->type = control_->add_const(ast->type);
      }
    }
  }

  return true;
}

auto Parser::check_pseudo_destructor_access(MemberExpressionAST* ast) -> bool {
  auto objectType = ast->baseExpression->type;
  auto cv = strip_cv(objectType);

  if (ast->accessOp == TokenKind::T_MINUS_GREATER) {
    auto pointerType = type_cast<PointerType>(objectType);
    if (!pointerType) return false;
    objectType = pointerType->elementType();
    cv = strip_cv(objectType);
  }

  if (!control_->is_scalar(objectType)) {
    // return false if the object type is not a scalar type
    return false;
  }

  // from this point on we are going to assume that we want a pseudo destructor
  // to be called on a scalar type.

  auto dtor = ast_cast<DestructorIdAST>(ast->unqualifiedId);
  if (!dtor) return true;

  auto name = ast_cast<NameIdAST>(dtor->id);
  if (!name) return true;

  auto symbol =
      Lookup{scope()}.lookupType(ast->nestedNameSpecifier, name->identifier);
  if (!symbol) return true;

  if (!control_->is_same(symbol->type(), objectType)) {
    parse_error(ast->unqualifiedId->firstSourceLocation(),
                "the type of object expression does not match the type "
                "being destroyed");
    return true;
  }

  ast->symbol = symbol;
  ast->type = control_->getFunctionType(control_->getVoidType(), {});

  return true;
}

auto Parser::parse_subscript_expression(ExpressionAST*& yyast,
                                        const ExprContext& ctx) -> bool {
  SourceLocation lbracketLoc;

  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  auto ast = make_node<SubscriptExpressionAST>(pool_);
  ast->baseExpression = yyast;
  ast->lbracketLoc = lbracketLoc;

  yyast = ast;

  parse_expr_or_braced_init_list(ast->indexExpression, ctx);

  expect(TokenKind::T_RBRACKET, ast->rbracketLoc);

  return true;
}

auto Parser::parse_call_expression(ExpressionAST*& yyast,
                                   const ExprContext& ctx) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = make_node<CallExpressionAST>(pool_);
  ast->baseExpression = yyast;
  ast->lparenLoc = lparenLoc;

  yyast = ast;

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression_list(ast->expressionList, ctx)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  check(ast);

  return true;
}

auto Parser::parse_postincr_expression(ExpressionAST*& yyast,
                                       const ExprContext& ctx) -> bool {
  SourceLocation opLoc;

  if (!match(TokenKind::T_MINUS_MINUS, opLoc) &&
      !match(TokenKind::T_PLUS_PLUS, opLoc)) {
    return false;
  }

  auto ast = make_node<PostIncrExpressionAST>(pool_);
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

auto Parser::parse_cpp_cast_expression(ExpressionAST*& yyast,
                                       const ExprContext& ctx) -> bool {
  SourceLocation castLoc;

  if (!parse_cpp_cast_head(castLoc)) return false;

  auto ast = make_node<CppCastExpressionAST>(pool_);
  yyast = ast;

  ast->castLoc = castLoc;

  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  expect(TokenKind::T_GREATER, ast->greaterLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_expression(ast->expression, ctx);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  check(ast);

  return true;
}

auto Parser::parse_builtin_bit_cast_expression(ExpressionAST*& yyast,
                                               const ExprContext& ctx) -> bool {
  if (!lookat(TokenKind::T___BUILTIN_BIT_CAST)) return false;

  auto ast = make_node<BuiltinBitCastExpressionAST>(pool_);
  yyast = ast;

  ast->castLoc = consumeToken();
  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");
  expect(TokenKind::T_COMMA, ast->commaLoc);
  parse_expression(ast->expression, ctx);
  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_builtin_offsetof_expression(ExpressionAST*& yyast,
                                               const ExprContext& ctx) -> bool {
  if (!lookat(TokenKind::T___BUILTIN_OFFSETOF)) return false;

  auto ast = make_node<BuiltinOffsetofExpressionAST>(pool_);
  yyast = ast;

  ast->offsetofLoc = consumeToken();
  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");
  expect(TokenKind::T_COMMA, ast->commaLoc);
  parse_expression(ast->expression, ctx);
  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  auto classType = type_cast<ClassType>(ast->typeId->type);
  auto id = ast_cast<IdExpressionAST>(ast->expression);

  if (classType && id && !id->nestedNameSpecifier) {
    auto symbol = classType->symbol();
    auto name = convertName(id->unqualifiedId);
    auto member = Lookup{scope()}.qualifiedLookup(symbol->scope(), name);
    auto field = symbol_cast<FieldSymbol>(member);
    ast->symbol = field;
  }

  ast->type = control_->getSizeType();

  return true;
}

auto Parser::parse_cpp_type_cast_expression(ExpressionAST*& yyast,
                                            const ExprContext& ctx) -> bool {
  auto lookat_function_call = [&] {
    LookaheadParser lookahead{this};

    SpecifierAST* typeSpecifier = nullptr;
    DeclSpecs specs{unit};

    if (!parse_simple_type_specifier(typeSpecifier, specs)) return false;

    if (!lookat(TokenKind::T_LPAREN)) return false;

    // ### prefer function calls to cpp-cast expressions for now.
    if (ast_cast<NamedTypeSpecifierAST>(typeSpecifier)) return true;

    return false;
  };

  auto lookat_braced_type_construction = [&] {
    LookaheadParser lookahead{this};

    SpecifierAST* typeSpecifier = nullptr;
    DeclSpecs specs{unit};

    if (!parse_simple_type_specifier(typeSpecifier, specs)) return false;

    BracedInitListAST* bracedInitList = nullptr;

    if (!parse_braced_init_list(bracedInitList, ctx)) return false;

    lookahead.commit();

    auto ast = make_node<BracedTypeConstructionAST>(pool_);
    yyast = ast;

    ast->typeSpecifier = typeSpecifier;
    ast->bracedInitList = bracedInitList;

    return true;
  };

  if (lookat_function_call()) return false;
  if (lookat_braced_type_construction()) return true;

  LookaheadParser lookahead{this};

  SpecifierAST* typeSpecifier = nullptr;
  DeclSpecs specs{unit};

  if (!parse_simple_type_specifier(typeSpecifier, specs)) return false;

  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  SourceLocation rparenLoc;

  List<ExpressionAST*>* expressionList = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_expression_list(expressionList, ctx)) return false;

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;
  }

  lookahead.commit();

  auto ast = make_node<TypeConstructionAST>(pool_);
  yyast = ast;

  ast->typeSpecifier = typeSpecifier;
  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_typeid_expression(ExpressionAST*& yyast,
                                     const ExprContext& ctx) -> bool {
  SourceLocation typeidLoc;

  if (!match(TokenKind::T_TYPEID, typeidLoc)) return false;

  SourceLocation lparenLoc;
  expect(TokenKind::T_LPAREN, lparenLoc);

  auto lookat_typeid_of_type = [&] {
    LookaheadParser lookahead{this};

    TypeIdAST* typeId = nullptr;
    if (!parse_type_id(typeId)) return false;

    SourceLocation rparenLoc;
    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

    lookahead.commit();

    auto ast = make_node<TypeidOfTypeExpressionAST>(pool_);
    yyast = ast;

    ast->typeidLoc = typeidLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;

    return true;
  };

  if (lookat_typeid_of_type()) return true;

  auto ast = make_node<TypeidExpressionAST>(pool_);
  yyast = ast;

  ast->typeidLoc = typeidLoc;
  ast->lparenLoc = lparenLoc;

  parse_expression(ast->expression, ctx);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_typename_expression(ExpressionAST*& yyast,
                                       const ExprContext& ctx) -> bool {
  LookaheadParser lookahead{this};

  SpecifierAST* typenameSpecifier = nullptr;
  DeclSpecs specs{unit};
  if (!parse_typename_specifier(typenameSpecifier, specs)) return false;

  if (BracedInitListAST* bracedInitList = nullptr;
      parse_braced_init_list(bracedInitList, ctx)) {
    lookahead.commit();

    auto ast = make_node<BracedTypeConstructionAST>(pool_);
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
    if (!parse_expression_list(expressionList, ctx)) return false;

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;
  }

  lookahead.commit();

  auto ast = make_node<TypeConstructionAST>(pool_);
  yyast = ast;

  ast->typeSpecifier = typenameSpecifier;
  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_type_traits_op(SourceLocation& loc,
                                  BuiltinTypeTraitKind& builtinKind) -> bool {
  const auto builtin = LA().builtinTypeTrait();
  if (builtin == BuiltinTypeTraitKind::T_NONE) return false;
  builtinKind = builtin;
  loc = consumeToken();
  return true;
}

auto Parser::parse_va_arg_expression(ExpressionAST*& yyast,
                                     const ExprContext& ctx) -> bool {
  SourceLocation vaArgLoc;
  if (!match(TokenKind::T___BUILTIN_VA_ARG, vaArgLoc)) return false;
  auto ast = make_node<VaArgExpressionAST>(pool_);
  yyast = ast;
  ast->vaArgLoc = vaArgLoc;
  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  parse_assignment_expression(ast->expression, ExprContext{});
  expect(TokenKind::T_COMMA, ast->commaLoc);
  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");
  expect(TokenKind::T_RPAREN, ast->rparenLoc);
  if (ast->type) {
    ast->type = ast->typeId->type;
  }
  return true;
}

auto Parser::parse_builtin_call_expression(ExpressionAST*& yyast,
                                           const ExprContext& ctx) -> bool {
  SourceLocation typeTraitLoc;
  BuiltinTypeTraitKind builtinKind = BuiltinTypeTraitKind::T_NONE;
  if (!parse_type_traits_op(typeTraitLoc, builtinKind)) return false;

  auto ast = make_node<TypeTraitExpressionAST>(pool_);
  yyast = ast;

  ast->typeTraitLoc = typeTraitLoc;
  ast->typeTrait = builtinKind;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  auto it = &ast->typeIdList;

  if (TypeIdAST* typeId = nullptr; parse_type_id(typeId)) {
    *it = make_list_node(pool_, typeId);
    it = &(*it)->next;
  } else {
    parse_error("expected a type id");
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (TypeIdAST* typeId = nullptr; parse_type_id(typeId)) {
      *it = make_list_node(pool_, typeId);
      it = &(*it)->next;
    } else {
      parse_error("expected a type id");
    }
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  ast->type = control_->getBoolType();

  return true;
}

auto Parser::parse_expression_list(List<ExpressionAST*>*& yyast,
                                   const ExprContext& ctx) -> bool {
  return parse_initializer_list(yyast, ctx);
}

auto Parser::parse_unary_expression(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  if (parse_unop_expression(yyast, ctx)) return true;
  if (parse_complex_expression(yyast, ctx)) return true;
  if (parse_await_expression(yyast, ctx)) return true;
  if (parse_sizeof_expression(yyast, ctx)) return true;
  if (parse_alignof_expression(yyast, ctx)) return true;
  if (parse_noexcept_expression(yyast, ctx)) return true;
  if (parse_new_expression(yyast, ctx)) return true;
  if (parse_delete_expression(yyast, ctx)) return true;
  if (parse_reflect_expression(yyast, ctx)) return true;
  return parse_postfix_expression(yyast, ctx);
}

auto Parser::parse_unop_expression(ExpressionAST*& yyast,
                                   const ExprContext& ctx) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation opLoc;
  if (!parse_unary_operator(opLoc)) return false;

  ExpressionAST* expression = nullptr;
  if (!parse_cast_expression(expression, ctx)) return false;

  lookahead.commit();

  auto ast = make_node<UnaryExpressionAST>(pool_);
  yyast = ast;

  ast->opLoc = opLoc;
  ast->op = unit->tokenKind(opLoc);
  ast->expression = expression;

  check(ast);

  return true;
}

auto Parser::parse_complex_expression(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  SourceLocation opLoc;

  if (!match(TokenKind::T___IMAG__, opLoc) &&
      !match(TokenKind::T___REAL__, opLoc)) {
    return false;
  }

  ExpressionAST* expression = nullptr;

  if (!parse_cast_expression(expression, ctx))
    parse_error("expected an expression");

  auto ast = make_node<UnaryExpressionAST>(pool_);
  yyast = ast;

  ast->opLoc = opLoc;
  ast->op = unit->tokenKind(opLoc);
  ast->expression = expression;

  return true;
}

auto Parser::parse_sizeof_expression(ExpressionAST*& yyast,
                                     const ExprContext& ctx) -> bool {
  SourceLocation sizeofLoc;

  if (!match(TokenKind::T_SIZEOF, sizeofLoc)) return false;

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = make_node<SizeofPackExpressionAST>(pool_);
    yyast = ast;

    ast->sizeofLoc = sizeofLoc;
    ast->ellipsisLoc = ellipsisLoc;

    expect(TokenKind::T_LPAREN, ast->lparenLoc);

    expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
    ast->identifier = unit->identifier(ast->identifierLoc);

    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    ast->type = control_->getSizeType();

    return true;
  }

  auto lookat_sizeof_type_id = [&] {
    LookaheadParser lookahead{this};

    SourceLocation lparenLoc;
    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    TypeIdAST* typeId = nullptr;
    if (!parse_type_id(typeId)) return false;

    SourceLocation rparenLoc;
    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

    lookahead.commit();

    auto ast = make_node<SizeofTypeExpressionAST>(pool_);
    yyast = ast;

    ast->sizeofLoc = sizeofLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;
    ast->type = control_->getSizeType();

    return true;
  };

  if (lookat_sizeof_type_id()) return true;

  auto ast = make_node<SizeofExpressionAST>(pool_);
  yyast = ast;

  ast->sizeofLoc = sizeofLoc;

  if (!parse_unary_expression(ast->expression, ctx)) {
    parse_error("expected an expression");
  }

  check(ast);

  return true;
}

auto Parser::parse_alignof_expression(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  SourceLocation alignofLoc;

  if (!match(TokenKind::T_ALIGNOF, alignofLoc)) return false;

  auto lookat_alignof_type_id = [&] {
    LookaheadParser lookahead{this};

    SourceLocation lparenLoc;
    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    TypeIdAST* typeId = nullptr;
    if (!parse_type_id(typeId)) return false;

    SourceLocation rparenLoc;
    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

    lookahead.commit();

    auto ast = make_node<AlignofTypeExpressionAST>(pool_);
    yyast = ast;

    ast->alignofLoc = alignofLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;

    check(ast);

    return true;
  };

  if (lookat_alignof_type_id()) return true;

  auto ast = make_node<AlignofExpressionAST>(pool_);
  yyast = ast;

  ast->alignofLoc = alignofLoc;

  if (!parse_unary_expression(ast->expression, ctx)) {
    parse_error("expected an expression");
  }

  check(ast);

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

auto Parser::parse_await_expression(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation awaitLoc;

  if (!match(TokenKind::T_CO_AWAIT, awaitLoc)) return false;

  auto ast = make_node<AwaitExpressionAST>(pool_);
  yyast = ast;

  ast->awaitLoc = awaitLoc;

  if (!parse_cast_expression(ast->expression, ctx))
    parse_error("expected an expression");

  return true;
}

auto Parser::parse_noexcept_expression(ExpressionAST*& yyast,
                                       const ExprContext& ctx) -> bool {
  SourceLocation noexceptLoc;

  if (!match(TokenKind::T_NOEXCEPT, noexceptLoc)) return false;

  auto ast = make_node<NoexceptExpressionAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_expression(ast->expression, ctx);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_new_expression(ExpressionAST*& yyast, const ExprContext& ctx)
    -> bool {
  if (!lookat(TokenKind::T_NEW) &&
      !lookat(TokenKind::T_COLON_COLON, TokenKind::T_NEW))
    return false;

  auto ast = make_node<NewExpressionAST>(pool_);
  yyast = ast;

  match(TokenKind::T_COLON_COLON, ast->scopeLoc);
  expect(TokenKind::T_NEW, ast->newLoc);

  parse_optional_new_placement(ast->newPlacement, ctx);

  const auto after_new_placement = currentLocation();

  auto lookat_nested_type_id = [&] {
    LookaheadParser lookahead{this};

    SourceLocation lparenLoc;
    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    List<SpecifierAST*>* typeSpecifierList = nullptr;
    DeclSpecs specs{unit};
    if (!parse_type_specifier_seq(typeSpecifierList, specs)) return false;

    DeclaratorAST* declarator = nullptr;
    Decl decl{specs};
    parse_optional_abstract_declarator(declarator, decl);

    SourceLocation rparenLoc;
    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

    lookahead.commit();

    NewInitializerAST* newInitializer = nullptr;
    parse_optional_new_initializer(newInitializer, ctx);

    ast->lparenLoc = lparenLoc;
    ast->typeSpecifierList = typeSpecifierList;
    ast->rparenLoc = rparenLoc;
    ast->newInitalizer = newInitializer;

    return true;
  };

  if (lookat_nested_type_id()) return true;

  DeclSpecs specs{unit};
  if (!parse_type_specifier_seq(ast->typeSpecifierList, specs))
    parse_error("expected a type specifier");

  Decl decl{specs};

  (void)parse_declarator(ast->declarator, decl, DeclaratorKind::kNewDeclarator);

  parse_optional_new_initializer(ast->newInitalizer, ctx);

  return true;
}

void Parser::parse_optional_new_placement(NewPlacementAST*& yyast,
                                          const ExprContext& ctx) {
  LookaheadParser lookahead{this};

  SourceLocation lparenLoc;
  if (!match(TokenKind::T_LPAREN, lparenLoc)) return;

  List<ExpressionAST*>* expressionList = nullptr;
  if (!parse_expression_list(expressionList, ctx)) return;

  SourceLocation rparenLoc;
  if (!match(TokenKind::T_RPAREN, rparenLoc)) return;

  lookahead.commit();

  auto ast = make_node<NewPlacementAST>(pool_);
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;
}

void Parser::parse_optional_new_initializer(NewInitializerAST*& yyast,
                                            const ExprContext& ctx) {
  if (BracedInitListAST* bracedInitList = nullptr;
      parse_braced_init_list(bracedInitList, ctx)) {
    auto ast = make_node<NewBracedInitializerAST>(pool_);
    yyast = ast;

    ast->bracedInitList = bracedInitList;
    return;
  }

  LookaheadParser lookahead{this};

  SourceLocation lparenLoc;
  List<ExpressionAST*>* expressionList = nullptr;
  SourceLocation rparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return;
  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_expression_list(expressionList, ctx)) return;
    if (!match(TokenKind::T_RPAREN, rparenLoc)) return;
  }

  lookahead.commit();

  auto ast = make_node<NewParenInitializerAST>(pool_);
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;
}

auto Parser::parse_delete_expression(ExpressionAST*& yyast,
                                     const ExprContext& ctx) -> bool {
  if (!lookat(TokenKind::T_DELETE) &&
      !lookat(TokenKind::T_COLON_COLON, TokenKind::T_DELETE))
    return false;

  auto ast = make_node<DeleteExpressionAST>(pool_);
  yyast = ast;

  match(TokenKind::T_COLON_COLON, ast->scopeLoc);
  expect(TokenKind::T_DELETE, ast->deleteLoc);

  if (match(TokenKind::T_LBRACKET, ast->lbracketLoc)) {
    expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  }

  if (!parse_cast_expression(ast->expression, ctx)) {
    parse_error("expected an expression");
  }

  return true;
}

auto Parser::parse_cast_expression(ExpressionAST*& yyast,
                                   const ExprContext& ctx) -> bool {
  const auto start = currentLocation();

  auto lookat_cast_expression = [&] {
    LookaheadParser lookahead{this};
    if (!parse_cast_expression_helper(yyast, ctx)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_cast_expression()) return true;

  return parse_unary_expression(yyast, ctx);
}

auto Parser::parse_cast_expression_helper(ExpressionAST*& yyast,
                                          const ExprContext& ctx) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  TypeIdAST* typeId = nullptr;

  if (!parse_type_id(typeId)) return false;

  SourceLocation rparenLoc;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

  ExpressionAST* expression = nullptr;

  if (BracedInitListAST* bracedInitList = nullptr;
      parse_braced_init_list(bracedInitList, ctx)) {
    expression = bracedInitList;
  } else if (!parse_cast_expression(expression, ctx)) {
    return false;
  }

  auto ast = make_node<CastExpressionAST>(pool_);
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
        if (exprContext.templArg) {
          rewind(start);
          return false;
        }

        tk = TokenKind::T_GREATER_GREATER;
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

    case TokenKind::T_AMP_AMP:
    case TokenKind::T_AMP:
    case TokenKind::T_BAR_BAR:
    case TokenKind::T_BAR:
    case TokenKind::T_CARET:
    case TokenKind::T_DOT_STAR:
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_LESS_EQUAL_GREATER:
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_LESS_LESS:
    case TokenKind::T_LESS:
    case TokenKind::T_MINUS_GREATER_STAR:
    case TokenKind::T_MINUS:
    case TokenKind::T_PERCENT:
    case TokenKind::T_PLUS:
    case TokenKind::T_SLASH:
    case TokenKind::T_STAR:
      tk = LA().kind();
      consumeToken();
      return true;

    default:
      return false;
  }  // switch
}

auto Parser::parse_binary_expression(ExpressionAST*& yyast,
                                     const ExprContext& exprContext) -> bool {
  if (!parse_cast_expression(yyast, exprContext)) return false;

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

    if (!parse_cast_expression(rhs, exprContext)) {
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

    auto ast = make_node<BinaryExpressionAST>(pool_);
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = rhs;
    ast->op = op;

    check(ast);

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
    auto ast = make_node<ConditionalExpressionAST>(pool_);
    ast->condition = yyast;
    ast->questionLoc = questionLoc;

    yyast = ast;

    parse_expression(ast->iftrueExpression, exprContext);

    expect(TokenKind::T_COLON, ast->colonLoc);

    if (exprContext.templArg || exprContext.templParam) {
      if (!parse_conditional_expression(ast->iffalseExpression, exprContext)) {
        parse_error("expected an expression");
      }
    } else {
      parse_assignment_expression(ast->iffalseExpression, exprContext);
    }
  }

  return true;
}

auto Parser::parse_yield_expression(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation yieldLoc;

  if (!match(TokenKind::T_CO_YIELD, yieldLoc)) return false;

  auto ast = make_node<YieldExpressionAST>(pool_);
  yyast = ast;

  ast->yieldLoc = yieldLoc;
  parse_expr_or_braced_init_list(ast->expression, ctx);

  return true;
}

auto Parser::parse_throw_expression(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation throwLoc;

  if (!match(TokenKind::T_THROW, throwLoc)) return false;

  auto ast = make_node<ThrowExpressionAST>(pool_);
  yyast = ast;

  ast->throwLoc = throwLoc;

  LookaheadParser lookahead{this};

  if (parse_maybe_assignment_expression(ast->expression, ctx)) {
    lookahead.commit();
  }

  return true;
}

void Parser::parse_assignment_expression(ExpressionAST*& yyast,
                                         const ExprContext& exprContext) {
  if (!parse_maybe_assignment_expression(yyast, exprContext)) {
    parse_error("expected an expression");
  }
}

auto Parser::parse_maybe_assignment_expression(ExpressionAST*& yyast,
                                               const ExprContext& exprContext)
    -> bool {
  if (parse_yield_expression(yyast, exprContext)) return true;

  if (parse_throw_expression(yyast, exprContext)) return true;

  if (!parse_conditional_expression(yyast, exprContext)) return false;

  SourceLocation opLoc;
  TokenKind op = TokenKind::T_EOF_SYMBOL;

  if (parse_assignment_operator(opLoc, op)) {
    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression, exprContext)) {
      parse_error("expected an expression");
    }

    auto ast = make_node<AssignmentExpressionAST>(pool_);
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = expression;
    ast->op = op;

    check(ast);

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

    default:
      return false;
  }  // switch
}

void Parser::parse_expression(ExpressionAST*& yyast, const ExprContext& ctx) {
  if (!parse_maybe_expression(yyast, ctx)) {
    parse_error("expected an expression");
  }
}

auto Parser::parse_maybe_expression(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  if (!parse_maybe_assignment_expression(yyast, ExprContext{})) return false;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    ExpressionAST* expression = nullptr;

    parse_assignment_expression(expression, ctx);

    auto ast = make_node<BinaryExpressionAST>(pool_);
    ast->leftExpression = yyast;
    ast->opLoc = commaLoc;
    ast->op = TokenKind::T_COMMA;
    ast->rightExpression = expression;
    if (ast->rightExpression) {
      ast->type = ast->rightExpression->type;
    }

    check(ast);

    yyast = ast;
  }

  return true;
}

auto Parser::parse_constant_expression(ExpressionAST*& yyast,
                                       std::optional<ConstValue>& value)
    -> bool {
  ExprContext exprContext;
  exprContext.isConstantEvaluated = true;
  if (!parse_conditional_expression(yyast, exprContext)) return false;
  value = evaluate_constant_expression(yyast);
  return true;
}

auto Parser::parse_template_argument_constant_expression(ExpressionAST*& yyast)
    -> bool {
  ExprContext exprContext;
  exprContext.templArg = true;
  return parse_conditional_expression(yyast, exprContext);
}

void Parser::parse_statement(StatementAST*& yyast) {
  if (!parse_maybe_statement(yyast)) {
    parse_error("expected a statement");
  }
}

auto Parser::parse_maybe_statement(StatementAST*& yyast) -> bool {
  SourceLocation extensionLoc;

  match(TokenKind::T___EXTENSION__, extensionLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  if (!extensionLoc) {
    match(TokenKind::T___EXTENSION__, extensionLoc);
  }

  if (parse_case_statement(yyast)) return true;
  if (parse_default_statement(yyast)) return true;
  if (parse_while_statement(yyast)) return true;
  if (parse_do_statement(yyast)) return true;
  if (parse_for_statement(yyast)) return true;
  if (parse_if_statement(yyast)) return true;
  if (parse_switch_statement(yyast)) return true;
  if (parse_break_statement(yyast)) return true;
  if (parse_continue_statement(yyast)) return true;
  if (parse_return_statement(yyast)) return true;
  if (parse_goto_statement(yyast)) return true;
  if (parse_coroutine_return_statement(yyast)) return true;
  if (parse_try_block(yyast)) return true;
  if (parse_maybe_compound_statement(yyast)) return true;
  if (parse_labeled_statement(yyast)) return true;

  auto lookat_declaration_statement = [&] {
    LookaheadParser lookahead{this};
    if (!parse_declaration_statement(yyast)) return false;
    lookahead.commit();
    return true;
  };

  if (lookat_declaration_statement()) return true;

  return parse_expression_statement(yyast);
}

void Parser::parse_init_statement(StatementAST*& yyast) {
  auto lookat_simple_declaration = [&] {
    LookaheadParser lookahead{this};
    DeclarationAST* declaration = nullptr;

    if (!scope()->isBlockScope()) {
      cxx_runtime_error("not a block scope");
    }

    if (!scope()->empty()) {
      cxx_runtime_error("enclosing scope of init statement is not empty");
    }

    if (!parse_simple_declaration(declaration,
                                  BindingContext::kInitStatement)) {
      scope()->reset();
      return false;
    }

    lookahead.commit();

    auto ast = make_node<DeclarationStatementAST>(pool_);
    yyast = ast;
    ast->declaration = declaration;
    return true;
  };

  if (lookat_simple_declaration()) return;

  LookaheadParser lookahead{this};

  ExpressionAST* expression = nullptr;
  if (!parse_maybe_expression(expression, ExprContext{})) return;

  SourceLocation semicolonLoc;
  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return;

  lookahead.commit();

  auto ast = make_node<ExpressionStatementAST>(pool_);
  yyast = ast;
  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;
}

void Parser::parse_condition(ExpressionAST*& yyast, const ExprContext& ctx) {
  auto lookat_condition = [&] {
    LookaheadParser lookahead{this};

    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_optional_attribute_specifier_seq(attributes);

    List<SpecifierAST*>* declSpecifierList = nullptr;

    DeclSpecs specs{unit};

    if (!parse_decl_specifier_seq(declSpecifierList, specs, {})) return false;

    DeclaratorAST* declarator = nullptr;
    Decl decl{specs};
    if (!parse_declarator(declarator, decl)) return false;

    auto symbol = binder_.declareVariable(declarator, decl);

    ExpressionAST* initializer = nullptr;

    if (!parse_brace_or_equal_initializer(initializer)) return false;

    lookahead.commit();

    auto ast = make_node<ConditionExpressionAST>(pool_);
    yyast = ast;
    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->declarator = declarator;
    ast->initializer = initializer;
    ast->symbol = symbol;

    return true;
  };

  if (lookat_condition()) return;

  parse_expression(yyast, ctx);
}

auto Parser::parse_labeled_statement(StatementAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON)) return false;

  auto ast = make_node<LabeledStatementAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  expect(TokenKind::T_COLON, ast->colonLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_case_statement(StatementAST*& yyast) -> bool {
  SourceLocation caseLoc;

  if (!match(TokenKind::T_CASE, caseLoc)) return false;

  ExpressionAST* expression = nullptr;
  std::optional<ConstValue> value;

  if (!parse_constant_expression(expression, value)) {
    parse_error("expected an expression");
  }

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  auto ast = make_node<CaseStatementAST>(pool_);
  yyast = ast;

  ast->caseLoc = caseLoc;
  ast->expression = expression;
  ast->colonLoc = colonLoc;

  return true;
}

auto Parser::parse_default_statement(StatementAST*& yyast) -> bool {
  SourceLocation defaultLoc;

  if (!match(TokenKind::T_DEFAULT, defaultLoc)) return false;

  SourceLocation colonLoc;

  expect(TokenKind::T_COLON, colonLoc);

  auto ast = make_node<DefaultStatementAST>(pool_);
  yyast = ast;

  ast->defaultLoc = defaultLoc;
  ast->colonLoc = colonLoc;

  return true;
}

auto Parser::parse_expression_statement(StatementAST*& yyast) -> bool {
  SourceLocation semicolonLoc;

  ExpressionAST* expression = nullptr;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    if (!parse_maybe_expression(expression, ExprContext{})) return false;

    expect(TokenKind::T_SEMICOLON, semicolonLoc);
  }

  auto ast = make_node<ExpressionStatementAST>(pool_);
  yyast = ast;

  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_maybe_compound_statement(StatementAST*& yyast) -> bool {
  CompoundStatementAST* statement = nullptr;
  if (parse_compound_statement(statement)) {
    yyast = statement;
    return true;
  }
  return false;
}

auto Parser::parse_compound_statement(CompoundStatementAST*& yyast, bool skip)
    -> bool {
  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  auto blockSymbol = binder_.enterBlock(lbraceLoc);

  auto ast = make_node<CompoundStatementAST>(pool_);
  yyast = ast;

  ast->symbol = blockSymbol;
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
  auto _ = Binder::ScopeGuard{&binder_};

  setScope(ast->symbol);

  bool skipping = false;

  auto it = &ast->statementList;

  LoopParser loop{this};

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    StatementAST* statement = nullptr;

    if (parse_maybe_statement(statement)) {
      *it = make_list_node(pool_, statement);
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

  auto _ = Binder::ScopeGuard{&binder_};

  auto blockSymbol = binder_.enterBlock(ifLoc);

  if (LA().isOneOf(TokenKind::T_EXCLAIM, TokenKind::T_CONSTEVAL)) {
    auto ast = make_node<ConstevalIfStatementAST>(pool_);
    yyast = ast;
    ast->ifLoc = ifLoc;

    ast->isNot = match(TokenKind::T_EXCLAIM, ast->exclaimLoc);

    expect(TokenKind::T_CONSTEVAL, ast->constvalLoc);

    if (CompoundStatementAST* statement = nullptr;
        parse_compound_statement(statement)) {
      ast->statement = statement;
    } else {
      parse_error("expected compound statement");
    }

    if (match(TokenKind::T_ELSE, ast->elseLoc)) {
      parse_statement(ast->elseStatement);
    }

    return true;
  }

  auto ast = make_node<IfStatementAST>(pool_);
  yyast = ast;

  ast->ifLoc = ifLoc;

  match(TokenKind::T_CONSTEXPR, ast->constexprLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_init_statement(ast->initializer);

  parse_condition(ast->condition, ExprContext{});

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  parse_statement(ast->statement);

  if (!match(TokenKind::T_ELSE, ast->elseLoc)) return true;

  parse_statement(ast->elseStatement);

  return true;
}

auto Parser::parse_switch_statement(StatementAST*& yyast) -> bool {
  SourceLocation switchLoc;

  if (!match(TokenKind::T_SWITCH, switchLoc)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  auto blockSymbol = binder_.enterBlock(switchLoc);

  auto ast = make_node<SwitchStatementAST>(pool_);
  yyast = ast;

  ast->switchLoc = switchLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_init_statement(ast->initializer);

  parse_condition(ast->condition, ExprContext{});

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  parse_statement(ast->statement);

  return true;
}

auto Parser::parse_while_statement(StatementAST*& yyast) -> bool {
  SourceLocation whileLoc;

  if (!match(TokenKind::T_WHILE, whileLoc)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  auto blockSymbol = binder_.enterBlock(whileLoc);

  auto ast = make_node<WhileStatementAST>(pool_);
  yyast = ast;

  ast->whileLoc = whileLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_condition(ast->condition, ExprContext{});

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  parse_statement(ast->statement);

  return true;
}

auto Parser::parse_do_statement(StatementAST*& yyast) -> bool {
  SourceLocation doLoc;

  if (!match(TokenKind::T_DO, doLoc)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  auto ast = make_node<DoStatementAST>(pool_);
  yyast = ast;

  ast->doLoc = doLoc;

  parse_statement(ast->statement);

  expect(TokenKind::T_WHILE, ast->whileLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_expression(ast->expression, ExprContext{});

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_for_statement(StatementAST*& yyast) -> bool {
  SourceLocation forLoc;
  if (!match(TokenKind::T_FOR, forLoc)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  auto blockSymbol = binder_.enterBlock(forLoc);

  setScope(blockSymbol);

  SourceLocation lparenLoc;
  expect(TokenKind::T_LPAREN, lparenLoc);

  StatementAST* initializer = nullptr;
  parse_init_statement(initializer);

  DeclarationAST* rangeDeclaration = nullptr;
  SourceLocation colonLoc;

  auto lookat_for_range_declaration = [&] {
    LookaheadParser lookahead{this};

    if (!parse_for_range_declaration(rangeDeclaration)) return false;

    if (!match(TokenKind::T_COLON, colonLoc)) return false;

    lookahead.commit();

    return true;
  };

  if (lookat_for_range_declaration()) {
    auto ast = make_node<ForRangeStatementAST>(pool_);
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

  auto ast = make_node<ForStatementAST>(pool_);
  yyast = ast;

  ast->forLoc = forLoc;
  ast->lparenLoc = lparenLoc;
  ast->initializer = initializer;

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    parse_condition(ast->condition, ExprContext{});
    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    parse_expression(ast->expression, ExprContext{});
    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  parse_statement(ast->statement);

  return true;
}

auto Parser::parse_for_range_declaration(DeclarationAST*& yyast) -> bool {
  List<AttributeSpecifierAST*>* attributeList = nullptr;

  parse_optional_attribute_specifier_seq(attributeList);

  List<SpecifierAST*>* declSpecifierList = nullptr;

  DeclSpecs specs{unit};

  if (!parse_decl_specifier_seq(declSpecifierList, specs, {})) return false;

  if (parse_structured_binding(yyast, attributeList, declSpecifierList, specs,
                               BindingContext::kCondition)) {
    return true;
  }

  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  if (!parse_declarator(declarator, decl)) return false;

  auto initDeclarator = make_node<InitDeclaratorAST>(pool_);
  initDeclarator->declarator = declarator;

  auto ast = make_node<SimpleDeclarationAST>(pool_);
  yyast = ast;

  ast->attributeList = attributeList;
  ast->declSpecifierList = declSpecifierList;
  ast->initDeclaratorList = make_list_node(pool_, initDeclarator);

  return true;
}

void Parser::parse_for_range_initializer(ExpressionAST*& yyast) {
  parse_expr_or_braced_init_list(yyast, ExprContext{});
}

auto Parser::parse_break_statement(StatementAST*& yyast) -> bool {
  SourceLocation breakLoc;

  if (!match(TokenKind::T_BREAK, breakLoc)) return false;

  auto ast = make_node<BreakStatementAST>(pool_);
  yyast = ast;

  ast->breakLoc = breakLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_continue_statement(StatementAST*& yyast) -> bool {
  SourceLocation continueLoc;

  if (!match(TokenKind::T_CONTINUE, continueLoc)) return false;

  auto ast = make_node<ContinueStatementAST>(pool_);
  yyast = ast;

  ast->continueLoc = continueLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_return_statement(StatementAST*& yyast) -> bool {
  SourceLocation returnLoc;

  if (!match(TokenKind::T_RETURN, returnLoc)) return false;

  auto ast = make_node<ReturnStatementAST>(pool_);
  yyast = ast;

  ast->returnLoc = returnLoc;

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    parse_expr_or_braced_init_list(ast->expression, ExprContext{});

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  return true;
}

auto Parser::parse_goto_statement(StatementAST*& yyast) -> bool {
  SourceLocation gotoLoc;

  if (!match(TokenKind::T_GOTO, gotoLoc)) return false;

  auto ast = make_node<GotoStatementAST>(pool_);
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

  auto ast = make_node<CoroutineReturnStatementAST>(pool_);
  yyast = ast;

  ast->coreturnLoc = coreturnLoc;

  if (!match(TokenKind::T_SEMICOLON, ast->semicolonLoc)) {
    parse_expr_or_braced_init_list(ast->expression, ExprContext{});

    expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);
  }

  return true;
}

auto Parser::parse_declaration_statement(StatementAST*& yyast) -> bool {
  DeclarationAST* declaration = nullptr;

  if (!parse_block_declaration(declaration, BindingContext::kBlock))
    return false;

  auto ast = make_node<DeclarationStatementAST>(pool_);
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

auto Parser::parse_template_declaration_body(
    DeclarationAST*& yyast,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  if (parse_deduction_guide(yyast))
    return true;
  else if (parse_export_declaration(yyast))
    return true;
  else if (parse_opaque_enum_declaration(yyast))
    return true;
  else if (parse_alias_declaration(yyast, templateDeclarations))
    return true;
  else
    return parse_simple_declaration(yyast, templateDeclarations,
                                    BindingContext::kTemplate);
}

auto Parser::parse_declaration(DeclarationAST*& yyast, BindingContext ctx)
    -> bool {
  if (lookat(TokenKind::T_RBRACE)) {
    return false;
  } else if (lookat(TokenKind::T_SEMICOLON)) {
    return parse_empty_declaration(yyast);
  } else if (parse_explicit_instantiation(yyast)) {
    return true;
  } else if (TemplateDeclarationAST* templateDeclaration = nullptr;
             parse_template_declaration(templateDeclaration)) {
    yyast = templateDeclaration;
    return true;
  } else if (parse_linkage_specification(yyast)) {
    return true;
  } else if (parse_namespace_definition(yyast)) {
    return true;
  } else if (parse_deduction_guide(yyast)) {
    return true;
  } else if (parse_export_declaration(yyast)) {
    return true;
  } else if (parse_module_import_declaration(yyast)) {
    return true;
  } else if (parse_attribute_declaration(yyast)) {
    return true;
  } else {
    return parse_block_declaration(yyast, ctx);
  }
}

auto Parser::parse_block_declaration(DeclarationAST*& yyast, BindingContext ctx)
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
    return parse_simple_declaration(yyast, ctx);
}

auto Parser::parse_alias_declaration(DeclarationAST*& yyast) -> bool {
  std::vector<TemplateDeclarationAST*> templateDeclarations;
  return parse_alias_declaration(yyast, templateDeclarations);
}

auto Parser::parse_alias_declaration(
    DeclarationAST*& yyast,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  SourceLocation usingLoc;
  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;
  List<AttributeSpecifierAST*>* attributes = nullptr;
  SourceLocation equalLoc;

  auto lookat_alias_declaration = [&] {
    LookaheadParser lookhead{this};

    if (!match(TokenKind::T_USING, usingLoc)) return false;

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    identifier = unit->identifier(identifierLoc);

    parse_optional_attribute_specifier_seq(attributes);

    if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

    lookhead.commit();

    return true;
  };

  if (!lookat_alias_declaration()) return false;

  if (!templateDeclarations.empty()) {
    mark_maybe_template_name(unit->identifier(identifierLoc));
  }

  List<AttributeSpecifierAST*>* gnuAttributeList = nullptr;
  parse_optional_attribute_specifier_seq(gnuAttributeList,
                                         AllowedAttributes::kGnuAttribute);

  TypeIdAST* typeId = nullptr;

  if (!parse_defining_type_id(typeId, templateDeclarations))
    parse_error("expected a type id");

  SourceLocation semicolonLoc;

  expect(TokenKind::T_SEMICOLON, semicolonLoc);

  auto symbol = binder_.declareTypeAlias(identifierLoc, typeId);

  auto ast = make_node<AliasDeclarationAST>(pool_);
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = identifier;
  ast->attributeList = attributes;
  ast->equalLoc = equalLoc;
  ast->gnuAttributeList = gnuAttributeList;
  ast->typeId = typeId;
  ast->semicolonLoc = semicolonLoc;
  ast->symbol = symbol;

  return true;
}

auto Parser::enterOrCreateNamespace(const Identifier* identifier,
                                    SourceLocation identifierLoc, bool isInline)
    -> NamespaceSymbol* {
  auto parentScope = scope();
  auto parentNamespace = symbol_cast<NamespaceSymbol>(parentScope->owner());

  NamespaceSymbol* namespaceSymbol = nullptr;

  if (!identifier) {
    namespaceSymbol = parentNamespace->unnamedNamespace();
  } else {
    auto resolved = parentScope->find(identifier) | views::namespaces;
    if (std::ranges::distance(resolved) == 1) {
      namespaceSymbol =
          symbol_cast<NamespaceSymbol>(*std::ranges::begin(resolved));
    }
  }

  if (!namespaceSymbol) {
    namespaceSymbol = control_->newNamespaceSymbol(parentScope, identifierLoc);

    if (identifier) {
      namespaceSymbol->setName(identifier);
    } else {
      parentNamespace->setUnnamedNamespace(namespaceSymbol);
    }

    namespaceSymbol->setInline(isInline);

    parentScope->addSymbol(namespaceSymbol);

    if (isInline || !namespaceSymbol->name()) {
      parentNamespace->scope()->addUsingDirective(namespaceSymbol->scope());
    }
  }

  setScope(namespaceSymbol);

  return namespaceSymbol;
}

void Parser::enterFunctionScope(
    FunctionDeclaratorChunkAST* functionDeclarator) {}

auto Parser::parse_empty_or_attribute_declaration(
    DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
    BindingContext ctx) -> auto {
  LookaheadParser lookahead{this};

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  lookahead.commit();

  if (attributes) {
    auto ast = make_node<AttributeDeclarationAST>(pool_);
    yyast = ast;
    ast->attributeList = attributes;
    ast->semicolonLoc = semicolonLoc;
    return true;
  }

  auto ast = make_node<EmptyDeclarationAST>(pool_);
  yyast = ast;
  ast->semicolonLoc = semicolonLoc;
  return true;
}

auto Parser::parse_notypespec_function_definition(
    DeclarationAST*& yyast, List<AttributeSpecifierAST*>* atributes,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations,
    BindingContext ctx) -> bool {
  if (!context_allows_function_definition(ctx)) return false;

  LookaheadParser lookahead{this};

  DeclSpecs specs{unit};
  List<SpecifierAST*>* declSpecifierList = nullptr;

  auto parse_optional_decl_specifier_seq_no_typespecs = [&] {
    LookaheadParser lookahead{this};
    if (!parse_decl_specifier_seq_no_typespecs(declSpecifierList, specs)) {
      specs = DeclSpecs{unit};
      return;
    }
    lookahead.commit();
  };

  parse_optional_decl_specifier_seq_no_typespecs();

  if (!parse_notypespec_function_definition(yyast, declSpecifierList, specs))
    return false;

  lookahead.commit();

  return true;
}

auto Parser::parse_type_or_forward_declaration(
    DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
    List<SpecifierAST*>* declSpecifierList, const DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations,
    BindingContext ctx) -> bool {
  if (ctx == BindingContext::kInitStatement) return false;

  LookaheadParser lookahead{this};

  List<AttributeSpecifierAST*>* trailingAttributes = nullptr;
  (void)parse_attribute_specifier_seq(trailingAttributes);

  if (!specs.hasClassOrEnumSpecifier()) return false;

  SourceLocation semicolonLoc;
  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  lookahead.commit();

  if (!declSpecifierList) cxx_runtime_error("no specs");

  const auto is_template_declaration = !templateDeclarations.empty();

  if (is_template_declaration) {
    auto classSpec =
        ast_cast<ElaboratedTypeSpecifierAST>(declSpecifierList->value);
    if (classSpec && !classSpec->nestedNameSpecifier)
      mark_maybe_template_name(classSpec->unqualifiedId);
  }

  auto ast = make_node<SimpleDeclarationAST>(pool_);
  yyast = ast;

  ast->attributeList = attributes;
  ast->declSpecifierList = declSpecifierList;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_structured_binding(DeclarationAST*& yyast,
                                      List<AttributeSpecifierAST*>* attributes,
                                      List<SpecifierAST*>* declSpecifierList,
                                      const DeclSpecs& specs,
                                      BindingContext ctx) -> bool {
  LookaheadParser lookahead{this};

  if (!context_allows_structured_bindings(ctx)) {
    return false;
  }

  SourceLocation refLoc;
  (void)parse_ref_qualifier(refLoc);

  SourceLocation lbracketLoc;
  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  List<NameIdAST*>* bindings = nullptr;
  if (!parse_identifier_list(bindings)) return false;

  SourceLocation rbracketLoc;
  if (!match(TokenKind::T_RBRACKET, rbracketLoc)) return false;

  ExpressionAST* initializer = nullptr;
  SourceLocation semicolonLoc;

  if (ctx != BindingContext::kCondition) {
    if (!parse_initializer(initializer, ExprContext{})) return false;
    if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;
  }

  lookahead.commit();

  auto ast = make_node<StructuredBindingDeclarationAST>(pool_);
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

auto Parser::parse_simple_declaration(DeclarationAST*& yyast,
                                      BindingContext ctx) -> bool {
  std::vector<TemplateDeclarationAST*> templateDeclarations;
  return parse_simple_declaration(yyast, templateDeclarations, ctx);
}

auto Parser::parse_simple_declaration(
    DeclarationAST*& yyast,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations,
    BindingContext ctx) -> bool {
  SourceLocation extensionLoc;

  match(TokenKind::T___EXTENSION__, extensionLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  if (SourceLocation semicolonLoc;
      ctx != BindingContext::kTemplate && attributes &&
      match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    // Found an attribute declaration instead of a simple declaration.
    auto ast = make_node<AttributeDeclarationAST>(pool_);
    yyast = ast;

    ast->attributeList = attributes;
    ast->semicolonLoc = semicolonLoc;
    return true;
  }

  if (parse_empty_or_attribute_declaration(yyast, attributes, ctx)) return true;

  if (parse_notypespec_function_definition(yyast, attributes,
                                           templateDeclarations, ctx))
    return true;

  DeclSpecs specs{unit};
  List<SpecifierAST*>* declSpecifierList = nullptr;

  auto lookat_decl_specifiers = [&] {
    LookaheadParser lookahead{this};

    if (!parse_decl_specifier_seq(declSpecifierList, specs,
                                  templateDeclarations))
      return false;

    if (!specs.hasTypeSpecifier()) return false;
    lookahead.commit();
    return true;
  };

  if (!lookat_decl_specifiers()) return false;

  if (parse_type_or_forward_declaration(yyast, attributes, declSpecifierList,
                                        specs, templateDeclarations, ctx))
    return true;

  if (parse_structured_binding(yyast, attributes, declSpecifierList, specs,
                               ctx))
    return true;

  return parse_simple_declaration(yyast, attributes, declSpecifierList, specs,
                                  templateDeclarations, ctx);
}

auto Parser::parse_simple_declaration(
    DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
    List<SpecifierAST*>* declSpecifierList, const DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations,
    BindingContext ctx) -> bool {
  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  if (!parse_declarator(declarator, decl)) return false;

  auto lookat_function_definition = [&] {
    if (!context_allows_function_definition(ctx)) return false;

    LookaheadParser lookahead{this};

    auto functionDeclarator = getFunctionPrototype(declarator);
    if (!functionDeclarator) return false;

    RequiresClauseAST* requiresClause = nullptr;
    (void)parse_requires_clause(requiresClause);

    if (!lookat_function_body()) return false;

    auto _ = Binder::ScopeGuard{&binder_};

    auto functionType =
        getDeclaratorType(unit, declarator, decl.specs.getType());

    auto q = decl.getNestedNameSpecifier();

    if (auto scope = decl.getScope()) {
      setScope(scope);
    } else if (q && config_.checkTypes) {
      parse_error(q->firstSourceLocation(),
                  std::format("unresolved class or namespace"));
    }

    const Name* functionName = decl.getName();
    auto functionSymbol = getFunction(scope(), functionName, functionType);

    if (!functionSymbol) {
      if (q && config_.checkTypes) {
        parse_error(q->firstSourceLocation(),
                    std::format("class or namespace has no member named '{}'",
                                to_string(functionName)));
      }

      functionSymbol = binder_.declareFunction(declarator, decl);
    }

    if (auto params = functionDeclarator->parameterDeclarationClause) {
      auto functionScope = functionSymbol->scope();
      functionScope->addSymbol(params->functionParametersSymbol);
      setScope(params->functionParametersSymbol);
    } else {
      setScope(functionSymbol);
    }

    if (ctx == BindingContext::kTemplate) {
      mark_maybe_template_name(declarator);
    }

    FunctionBodyAST* functionBody = nullptr;
    if (!parse_function_body(functionBody))
      parse_error("expected function body");

    lookahead.commit();

    functionSymbol->setDefined(true);

    auto ast = make_node<FunctionDefinitionAST>(pool_);
    yyast = ast;

    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->declarator = declarator;
    ast->requiresClause = requiresClause;
    ast->functionBody = functionBody;
    ast->symbol = functionSymbol;

    if (classDepth_) pendingFunctionDefinitions_.push_back(ast);

    return true;
  };

  if (lookat_function_definition()) return true;

  List<InitDeclaratorAST*>* initDeclaratorList = nullptr;
  auto declIt = &initDeclaratorList;

  InitDeclaratorAST* initDeclarator = nullptr;
  if (!parse_init_declarator(initDeclarator, declarator, decl, ctx))
    return false;

  if (ctx == BindingContext::kTemplate) {
    auto declarator = initDeclarator->declarator;
    mark_maybe_template_name(declarator);
  }

  *declIt = make_list_node(pool_, initDeclarator);
  declIt = &(*declIt)->next;

  if (ctx != BindingContext::kTemplate) {
    SourceLocation commaLoc;

    while (match(TokenKind::T_COMMA, commaLoc)) {
      InitDeclaratorAST* initDeclarator = nullptr;
      if (!parse_init_declarator(initDeclarator, specs, ctx)) return false;

      *declIt = make_list_node(pool_, initDeclarator);
      declIt = &(*declIt)->next;
    }
  }

  SourceLocation semicolonLoc;
  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = make_node<SimpleDeclarationAST>(pool_);
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

  Decl decl{specs};
  if (!parse_declarator_id(declaratorId, decl, DeclaratorKind::kDeclarator))
    return false;

  auto _ = Binder::ScopeGuard{&binder_};

  auto nestedNameSpecifier = decl.getNestedNameSpecifier();

  if (auto scope = decl.getScope()) {
    setScope(scope);
  } else if (auto q = decl.getNestedNameSpecifier()) {
    if (config_.checkTypes) {
      parse_error(q->firstSourceLocation(),
                  std::format("unresolved class or namespace"));
    }
  }

  FunctionDeclaratorChunkAST* functionDeclarator = nullptr;
  if (!parse_function_declarator(functionDeclarator)) return false;

  auto declarator = make_node<DeclaratorAST>(pool_);
  declarator->coreDeclarator = declaratorId;

  declarator->declaratorChunkList =
      make_list_node<DeclaratorChunkAST>(pool_, functionDeclarator);

  RequiresClauseAST* requiresClause = nullptr;

  const auto has_requires_clause = parse_requires_clause(requiresClause);

  if (!has_requires_clause) parse_virt_specifier_seq(functionDeclarator);

  parse_optional_attribute_specifier_seq(functionDeclarator->attributeList);

  auto functionType = getDeclaratorType(unit, declarator, decl.specs.getType());

  SourceLocation equalLoc;
  SourceLocation zeroLoc;

  const auto isPure = parse_pure_specifier(equalLoc, zeroLoc);

  functionDeclarator->isPure = isPure;

  const auto isDeclaration = isPure || lookat(TokenKind::T_SEMICOLON);
  const auto isDefinition = lookat_function_body();

  if (!isDeclaration && !isDefinition) return false;

  FunctionSymbol* functionSymbol =
      getFunction(scope(), decl.getName(), functionType);

  if (!functionSymbol) {
    functionSymbol = binder_.declareFunction(declarator, decl);
  }

  SourceLocation semicolonLoc;

  if (isPure) {
    expect(TokenKind::T_SEMICOLON, semicolonLoc);
  }

  if (isDeclaration) {
    auto initDeclarator = make_node<InitDeclaratorAST>(pool_);
    initDeclarator->declarator = declarator;

    auto ast = make_node<SimpleDeclarationAST>(pool_);
    yyast = ast;
    ast->declSpecifierList = declSpecifierList;
    ast->initDeclaratorList = make_list_node(pool_, initDeclarator);
    ast->requiresClause = requiresClause;
    ast->semicolonLoc = semicolonLoc;

    return true;
  }

  // function definition

  functionSymbol->setDefined(true);

  if (auto params = functionDeclarator->parameterDeclarationClause) {
    auto functionScope = functionSymbol->scope();
    functionScope->addSymbol(params->functionParametersSymbol);
    setScope(params->functionParametersSymbol);
  } else {
    setScope(functionSymbol);
  }

  FunctionBodyAST* functionBody = nullptr;

  if (!parse_function_body(functionBody)) parse_error("expected function body");

  auto ast = make_node<FunctionDefinitionAST>(pool_);
  yyast = ast;

  ast->declSpecifierList = declSpecifierList;
  ast->declarator = declarator;
  ast->functionBody = functionBody;
  ast->symbol = functionSymbol;

  if (classDepth_) pendingFunctionDefinitions_.push_back(ast);

  return true;
}

auto Parser::parse_static_assert_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation staticAssertLoc;

  if (!match(TokenKind::T_STATIC_ASSERT, staticAssertLoc)) return false;

  auto ast = make_node<StaticAssertDeclarationAST>(pool_);
  yyast = ast;

  ast->staticAssertLoc = staticAssertLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  std::optional<ConstValue> constValue;

  if (!parse_constant_expression(ast->expression, constValue)) {
    parse_error("expected an expression");
  }

  if (match(TokenKind::T_COMMA, ast->commaLoc)) {
    expect(TokenKind::T_STRING_LITERAL, ast->literalLoc);
    ast->literal = unit->literal(ast->literalLoc);
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  if (!binder_.inTemplate()) {
    // not in a template context

    bool value = false;

    if (constValue.has_value()) {
      value = visit(to_bool, *constValue);
    }

    if (!value && config_.checkTypes) {
      SourceLocation loc = ast->firstSourceLocation();

      if (!ast->expression || !constValue.has_value()) {
        parse_error(loc,
                    "static assertion expression is not an integral constant "
                    "expression");
      } else {
        if (ast->literalLoc)
          loc = ast->literalLoc;
        else if (ast->expression)
          ast->expression->firstSourceLocation();

        std::string message =
            ast->literal ? ast->literal->value() : "static assert failed";

        unit->error(loc, std::move(message));
      }
    }
  }

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

  auto ast = make_node<EmptyDeclarationAST>(pool_);
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

  auto ast = make_node<AttributeDeclarationAST>(pool_);
  yyast = ast;

  ast->attributeList = attributes;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_decl_specifier(
    SpecifierAST*& yyast, DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_TYPEDEF: {
      auto ast = make_node<TypedefSpecifierAST>(pool_);
      yyast = ast;
      ast->typedefLoc = consumeToken();
      specs.isTypedef = true;
      return true;
    }

    case TokenKind::T_FRIEND: {
      auto ast = make_node<FriendSpecifierAST>(pool_);
      yyast = ast;
      ast->friendLoc = consumeToken();
      specs.isFriend = true;
      return true;
    }

    case TokenKind::T_CONSTEXPR: {
      auto ast = make_node<ConstexprSpecifierAST>(pool_);
      yyast = ast;
      ast->constexprLoc = consumeToken();
      specs.isConstexpr = true;
      return true;
    }

    case TokenKind::T_CONSTEVAL: {
      auto ast = make_node<ConstevalSpecifierAST>(pool_);
      yyast = ast;
      ast->constevalLoc = consumeToken();
      specs.isConsteval = true;
      return true;
    }

    case TokenKind::T_CONSTINIT: {
      auto ast = make_node<ConstinitSpecifierAST>(pool_);
      yyast = ast;
      ast->constinitLoc = consumeToken();
      specs.isConstinit = true;
      return true;
    }

    case TokenKind::T_INLINE: {
      auto ast = make_node<InlineSpecifierAST>(pool_);
      yyast = ast;
      ast->inlineLoc = consumeToken();
      specs.isInline = true;
      return true;
    }

    default:
      if (parse_storage_class_specifier(yyast, specs)) return true;

      if (parse_function_specifier(yyast, specs)) return true;

      if (!specs.no_typespecs) {
        return parse_defining_type_specifier(yyast, specs,
                                             templateDeclarations);
      }

      return false;
  }  // switch
}

auto Parser::parse_decl_specifier_seq(
    List<SpecifierAST*>*& yyast, DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  auto it = &yyast;

  specs.no_typespecs = false;

  SpecifierAST* specifier = nullptr;
  if (!parse_decl_specifier(specifier, specs, templateDeclarations))
    return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  *it = make_list_node(pool_, specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs, templateDeclarations)) {
    List<AttributeSpecifierAST*>* attributes = nullptr;
    parse_optional_attribute_specifier_seq(attributes);

    *it = make_list_node(pool_, specifier);
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

  if (!parse_decl_specifier(specifier, specs, {})) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_optional_attribute_specifier_seq(attributes);

  *it = make_list_node(pool_, specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs, {})) {
    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_optional_attribute_specifier_seq(attributes);

    *it = make_list_node(pool_, specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

auto Parser::parse_storage_class_specifier(SpecifierAST*& yyast,
                                           DeclSpecs& specs) -> bool {
  SourceLocation loc;

  if (match(TokenKind::T_STATIC, loc)) {
    auto ast = make_node<StaticSpecifierAST>(pool_);
    yyast = ast;
    ast->staticLoc = loc;
    specs.isStatic = true;
    return true;
  }
  if (match(TokenKind::T_THREAD_LOCAL, loc)) {
    auto ast = make_node<ThreadLocalSpecifierAST>(pool_);
    yyast = ast;
    ast->threadLocalLoc = loc;
    specs.isThreadLocal = true;
    return true;
  }
  if (match(TokenKind::T_EXTERN, loc)) {
    auto ast = make_node<ExternSpecifierAST>(pool_);
    yyast = ast;
    ast->externLoc = loc;
    specs.isExtern = true;
    return true;
  }
  if (match(TokenKind::T_MUTABLE, loc)) {
    auto ast = make_node<MutableSpecifierAST>(pool_);
    yyast = ast;
    ast->mutableLoc = loc;
    specs.isMutable = true;
    return true;
  }
  if (match(TokenKind::T___THREAD, loc)) {
    auto ast = make_node<ThreadSpecifierAST>(pool_);
    yyast = ast;
    ast->threadLoc = loc;
    specs.isThread = true;
    return true;
  }

  return false;
}

auto Parser::parse_function_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  SourceLocation virtualLoc;

  if (match(TokenKind::T_VIRTUAL, virtualLoc)) {
    auto ast = make_node<VirtualSpecifierAST>(pool_);
    yyast = ast;
    ast->virtualLoc = virtualLoc;
    specs.isVirtual = true;
    return true;
  }

  return parse_explicit_specifier(yyast, specs);
}

auto Parser::parse_explicit_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  SourceLocation explicitLoc;

  if (!match(TokenKind::T_EXPLICIT, explicitLoc)) return false;

  specs.isExplicit = true;

  auto ast = make_node<ExplicitSpecifierAST>(pool_);
  yyast = ast;

  ast->explicitLoc = explicitLoc;

  if (match(TokenKind::T_LPAREN, ast->lparenLoc)) {
    std::optional<ConstValue> value;

    if (!parse_constant_expression(ast->expression, value)) {
      parse_error("expected a expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  return true;
}

auto Parser::parse_type_specifier(
    SpecifierAST*& yyast, DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  if (parse_simple_type_specifier(yyast, specs)) {
    return true;
  } else if (parse_cv_qualifier(yyast, specs)) {
    return true;
  } else if (parse_elaborated_type_specifier(yyast, specs,
                                             templateDeclarations)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else if (parse_splicer_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else if (parse_typename_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else {
    return false;
  }
}

auto Parser::parse_type_specifier_seq(List<SpecifierAST*>*& yyast,
                                      DeclSpecs& specs) -> bool {
  auto it = &yyast;

  specs.no_class_or_enum_specs = true;

  SpecifierAST* typeSpecifier = nullptr;
  if (!parse_type_specifier(typeSpecifier, specs, {})) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  *it = make_list_node(pool_, typeSpecifier);
  it = &(*it)->next;

  typeSpecifier = nullptr;

  LoopParser loop{this};

  while (LA()) {
    loop.start();

    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_type_specifier(typeSpecifier, specs, {})) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeSpecifierAST*>* attributes = nullptr;
    parse_optional_attribute_specifier_seq(attributes);

    *it = make_list_node(pool_, typeSpecifier);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_defining_type_specifier(
    SpecifierAST*& yyast, DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  if (!specs.no_class_or_enum_specs && !specs.typeSpecifier) {
    LookaheadParser lookahead{this};

    if (parse_enum_specifier(yyast, specs)) {
      lookahead.commit();

      if (auto enumSpec = ast_cast<EnumSpecifierAST>(yyast)) {
        specs.accept(enumSpec);
      }

      return true;
    }

    if (ClassSpecifierAST* classSpecifier = nullptr;
        parse_class_specifier(classSpecifier, specs, templateDeclarations)) {
      lookahead.commit();

      specs.accept(classSpecifier);

      yyast = classSpecifier;

      return true;
    }
  }

  return parse_type_specifier(yyast, specs, templateDeclarations);
}

auto Parser::parse_defining_type_specifier_seq(List<SpecifierAST*>*& yyast,
                                               DeclSpecs& specs) -> bool {
  auto it = &yyast;

  SpecifierAST* typeSpecifier = nullptr;

  if (!parse_defining_type_specifier(typeSpecifier, specs, {})) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_optional_attribute_specifier_seq(attributes);

  *it = make_list_node(pool_, typeSpecifier);
  it = &(*it)->next;

  LoopParser loop{this};

  while (LA()) {
    loop.start();

    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_defining_type_specifier(typeSpecifier, specs, {})) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeSpecifierAST*>* attributes = nullptr;
    parse_optional_attribute_specifier_seq(attributes);

    *it = make_list_node(pool_, typeSpecifier);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_simple_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (parse_size_type_specifier(yyast, specs)) return true;
  if (parse_sign_type_specifier(yyast, specs)) return true;
  if (parse_complex_type_specifier(yyast, specs)) return true;

  if (specs.typeSpecifier) {
    return false;
  } else if (parse_primitive_type_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else if (parse_placeholder_type_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else if (parse_underlying_type_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else if (parse_atomic_type_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else if (parse_named_type_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  } else if (parse_decltype_specifier_type_specifier(yyast, specs)) {
    specs.setTypeSpecifier(yyast);
    return true;
  }

  return false;
}

auto Parser::parse_size_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (SourceLocation specifierLoc; match(TokenKind::T_LONG, specifierLoc)) {
    auto ast = make_node<SizeTypeSpecifierAST>(pool_);
    yyast = ast;
    ast->specifierLoc = specifierLoc;
    ast->specifier = unit->tokenKind(specifierLoc);

    if (specs.isLong)
      specs.isLongLong = true;
    else
      specs.isLong = true;

    return true;
  }

  if (SourceLocation specifierLoc; match(TokenKind::T_SHORT, specifierLoc)) {
    auto ast = make_node<SizeTypeSpecifierAST>(pool_);
    yyast = ast;
    ast->specifierLoc = specifierLoc;
    ast->specifier = unit->tokenKind(specifierLoc);

    specs.isShort = true;

    return true;
  }

  return false;
}

auto Parser::parse_sign_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (SourceLocation specifierLoc; match(TokenKind::T_UNSIGNED, specifierLoc)) {
    auto ast = make_node<SignTypeSpecifierAST>(pool_);
    yyast = ast;
    ast->specifierLoc = specifierLoc;
    ast->specifier = unit->tokenKind(specifierLoc);

    specs.isUnsigned = true;

    return true;
  }

  if (SourceLocation specifierLoc; match(TokenKind::T_SIGNED, specifierLoc)) {
    auto ast = make_node<SignTypeSpecifierAST>(pool_);
    yyast = ast;
    ast->specifierLoc = specifierLoc;
    ast->specifier = unit->tokenKind(specifierLoc);

    specs.isSigned = true;

    return true;
  }

  return false;
}

auto Parser::parse_complex_type_specifier(SpecifierAST*& yyast,
                                          DeclSpecs& specs) -> bool {
  if (!LA().isOneOf(TokenKind::T__COMPLEX, TokenKind::T___COMPLEX__))
    return false;
  auto ast = make_node<ComplexTypeSpecifierAST>(pool_);
  yyast = ast;
  ast->complexLoc = consumeToken();
  specs.isComplex = true;

  return true;
}

auto Parser::instantiate(SimpleTemplateIdAST* templateId) -> Symbol* {
  if (!config_.checkTypes) return nullptr;

  std::vector<TemplateArgument> args;
  for (auto it = templateId->templateArgumentList; it; it = it->next) {
    if (auto arg = ast_cast<TypeTemplateArgumentAST>(it->value)) {
      args.push_back(arg->typeId->type);
    } else {
      parse_error(it->value->firstSourceLocation(),
                  std::format("only type template arguments are supported"));
    }
  }

  auto needsInstantiation = [&]() -> bool {
    if (args.empty()) return true;
    for (std::size_t i = 0; i < args.size(); ++i) {
      auto typeArgument = std::get_if<const Type*>(&args[i]);
      if (!typeArgument) return true;
      auto ty = type_cast<TypeParameterType>(*typeArgument);
      if (!ty) return true;
      if (ty->symbol()->index() != i) return true;
    }
    return false;
  };

  if (!needsInstantiation()) return nullptr;

  auto symbol = control_->instantiate(unit, templateId->primaryTemplateSymbol,
                                      std::move(args));

  return symbol;
}

auto Parser::parse_named_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (specs.isUnsigned || specs.isSigned || specs.isShort || specs.isLong)
    return false;

  LookaheadParser lookahead{this};

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  if (!lookat(TokenKind::T_IDENTIFIER)) return false;

  UnqualifiedIdAST* unqualifiedId = nullptr;
  if (!parse_type_name(unqualifiedId, nestedNameSpecifier,
                       isTemplateIntroduced)) {
    return false;
  }

  Symbol* typeSymbol = nullptr;

  if (auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId)) {
    if (auto conceptSymbol =
            symbol_cast<ConceptSymbol>(templateId->primaryTemplateSymbol)) {
      if (!lookat(TokenKind::T_AUTO)) return false;
    }

    if (auto symbol = instantiate(templateId)) {
      specs.type = symbol->type();
    }
  } else {
    auto name = ast_cast<NameIdAST>(unqualifiedId);
    auto symbol =
        Lookup{scope()}.lookupType(nestedNameSpecifier, name->identifier);

    if (is_type(symbol)) {
      typeSymbol = symbol;
      specs.type = symbol->type();
    } else {
      if (config_.checkTypes) return false;
    }
  }

  lookahead.commit();

  if (!specs.type) {
    specs.type = control_->getUnresolvedNameType(unit, nestedNameSpecifier,
                                                 unqualifiedId);
  }

  auto ast = make_node<NamedTypeSpecifierAST>(pool_);
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->unqualifiedId = unqualifiedId;
  ast->isTemplateIntroduced = isTemplateIntroduced;
  ast->symbol = typeSymbol;

  return true;
}

auto Parser::parse_decltype_specifier_type_specifier(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) -> bool {
  DecltypeSpecifierAST* decltypeSpecifier = nullptr;
  if (!parse_decltype_specifier(decltypeSpecifier)) return false;

  specs.setTypeSpecifier(decltypeSpecifier);

  yyast = decltypeSpecifier;

  specs.accept(decltypeSpecifier);

  return true;
}

auto Parser::parse_underlying_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  SourceLocation underlyingTypeLoc;
  if (!match(TokenKind::T___UNDERLYING_TYPE, underlyingTypeLoc)) return false;

  auto ast = make_node<UnderlyingTypeSpecifierAST>(pool_);
  yyast = ast;

  ast->underlyingTypeLoc = underlyingTypeLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected type id");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  specs.accept(ast);

  return true;
}

auto Parser::parse_atomic_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  SourceLocation atomicLoc;
  if (!match(TokenKind::T__ATOMIC, atomicLoc)) return false;

  auto ast = make_node<AtomicTypeSpecifierAST>(pool_);
  yyast = ast;

  ast->atomicLoc = atomicLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected type id");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_primitive_type_specifier(SpecifierAST*& yyast,
                                            DeclSpecs& specs) -> bool {
  auto makeIntegralTypeSpecifier = [&] {
    auto ast = make_node<IntegralTypeSpecifierAST>(pool_);
    yyast = ast;
    ast->specifierLoc = consumeToken();
    ast->specifier = unit->tokenKind(ast->specifierLoc);
  };

  auto makeFloatingPointTypeSpecifier = [&] {
    auto ast = make_node<FloatingPointTypeSpecifierAST>(pool_);
    yyast = ast;
    ast->specifierLoc = consumeToken();
    ast->specifier = unit->tokenKind(ast->specifierLoc);
  };

  switch (auto tk = LA(); tk.kind()) {
    case TokenKind::T___BUILTIN_VA_LIST: {
      auto ast = make_node<VaListTypeSpecifierAST>(pool_);
      yyast = ast;
      ast->specifierLoc = consumeToken();
      ast->specifier = unit->tokenKind(ast->specifierLoc);
      specs.accept(ast);
      specs.type = control_->getBuiltinVaListType();
      return true;
    }

    case TokenKind::T_CHAR:
    case TokenKind::T_CHAR8_T:
    case TokenKind::T_CHAR16_T:
    case TokenKind::T_CHAR32_T:
    case TokenKind::T_WCHAR_T:
    case TokenKind::T_BOOL:
    case TokenKind::T_INT:
    case TokenKind::T___INT64:
    case TokenKind::T___INT128:
      makeIntegralTypeSpecifier();
      specs.accept(yyast);
      return true;

    case TokenKind::T_FLOAT:
    case TokenKind::T_DOUBLE:
    case TokenKind::T___FLOAT80:
    case TokenKind::T___FLOAT128:
      makeFloatingPointTypeSpecifier();
      specs.accept(yyast);
      return true;

    case TokenKind::T_VOID: {
      auto ast = make_node<VoidTypeSpecifierAST>(pool_);
      yyast = ast;
      ast->voidLoc = consumeToken();
      specs.accept(ast);
      return true;
    }

    default:
      return false;
  }  // switch
}

auto Parser::maybe_template_name(const Identifier* id) -> bool {
  if (!config_.fuzzyTemplateResolution) return true;
  if (template_names_.contains(id)) return true;
  if (concept_names_.contains(id)) return true;
  return false;
}

void Parser::mark_maybe_template_name(const Identifier* id) {
  if (!config_.fuzzyTemplateResolution) return;
  if (id) template_names_.insert(id);
}

void Parser::mark_maybe_template_name(UnqualifiedIdAST* name) {
  if (auto nameId = ast_cast<NameIdAST>(name)) {
    mark_maybe_template_name(nameId->identifier);
  }
}

void Parser::mark_maybe_template_name(DeclaratorAST* declarator) {
  if (!declarator) return;
  auto declaratorId = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator);
  if (!declaratorId) return;
  if (declaratorId->nestedNameSpecifier) return;
  mark_maybe_template_name(declaratorId->unqualifiedId);
}

void Parser::check_type_traits() {
  SourceLocation typeTraitLoc;
  BuiltinTypeTraitKind builtinKind = BuiltinTypeTraitKind::T_NONE;
  if (!parse_type_traits_op(typeTraitLoc, builtinKind)) return;

  // reset the builtin type traits to be available as identifiers
  if (auto id = unit->identifier(typeTraitLoc)) {
    id->setInfo(nullptr);
  }

#if false

  parse_warn(
      typeTraitLoc,
      std::format("keyword '{}' will be made available as an identifier for "
                  "the remainder of the translation unit",
                  Token::spell(builtinKind)));
#endif

  rewind(typeTraitLoc);
}

auto Parser::strip_parentheses(ExpressionAST* ast) -> ExpressionAST* {
  while (auto paren = ast_cast<NestedExpressionAST>(ast)) {
    ast = paren->expression;
  }
  return ast;
}

auto Parser::strip_cv(const Type*& type) -> CvQualifiers {
  if (auto qualType = type_cast<QualType>(type)) {
    auto cv = qualType->cvQualifiers();
    type = qualType->elementType();
    return cv;
  }
  return {};
}

auto Parser::is_const(CvQualifiers cv) const -> bool {
  return cv == CvQualifiers::kConst || cv == CvQualifiers::kConstVolatile;
}

auto Parser::is_volatile(CvQualifiers cv) const -> bool {
  return cv == CvQualifiers::kVolatile || cv == CvQualifiers::kConstVolatile;
}

auto Parser::is_prvalue(ExpressionAST* expr) const -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kPrValue;
}

auto Parser::is_lvalue(ExpressionAST* expr) const -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kLValue;
}

auto Parser::is_xvalue(ExpressionAST* expr) const -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kXValue;
}

auto Parser::is_glvalue(ExpressionAST* expr) const -> bool {
  if (!expr) return false;
  return expr->valueCategory == ValueCategory::kLValue ||
         expr->valueCategory == ValueCategory::kXValue;
}

auto Parser::is_template(Symbol* symbol) const -> bool {
  if (!symbol) return false;
  if (symbol->isTemplateTypeParameter()) return true;
  auto templateParameters = cxx::getTemplateParameters(symbol);
  return templateParameters != nullptr;
}

auto Parser::evaluate_constant_expression(ExpressionAST* expr)
    -> std::optional<ConstValue> {
  ASTInterpreter sem{unit};
  return sem.evaluate(expr);
}

auto Parser::parse_elaborated_enum_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  SourceLocation enumLoc;
  if (!match(TokenKind::T_ENUM, enumLoc)) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kDeclarative);

  NameIdAST* name = nullptr;
  if (!parse_name_id(name)) {
    parse_error("expected a name");
  }

  auto ast = make_node<ElaboratedTypeSpecifierAST>(pool_);
  yyast = ast;

  ast->classLoc = enumLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = name;
  ast->classKey = TokenKind::T_ENUM;

  return true;
}

auto Parser::parse_elaborated_type_specifier(
    SpecifierAST*& yyast, DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  if (specs.typeSpecifier) return false;

  if (parse_elaborated_enum_specifier(yyast, specs)) return true;

  SourceLocation classLoc;
  if (!parse_class_key(classLoc)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kDeclarative);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  if (!lookat(TokenKind::T_IDENTIFIER)) return false;

  const auto classKey = unit->tokenKind(classLoc);

  auto ast = make_node<ElaboratedTypeSpecifierAST>(pool_);
  yyast = ast;

  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->classKey = unit->tokenKind(classLoc);
  ast->isTemplateIntroduced = isTemplateIntroduced;

  const Identifier* className = nullptr;
  SimpleTemplateIdAST* templateId = nullptr;

  if (parse_simple_template_id(templateId)) {
    ast->unqualifiedId = templateId;
    className = templateId->identifier;
  } else {
    // if we reach here, we have a name-id
    NameIdAST* nameId = nullptr;
    (void)parse_name_id(nameId);

    ast->unqualifiedId = nameId;
    className = nameId->identifier;
  }

  binder_.bind(ast, specs);

  return true;
}

auto Parser::parse_decl_specifier_seq_no_typespecs(List<SpecifierAST*>*& yyast)
    -> bool {
  DeclSpecs specs{unit};
  return parse_decl_specifier_seq_no_typespecs(yyast, specs);
}

auto Parser::parse_decltype_specifier(DecltypeSpecifierAST*& yyast) -> bool {
  SourceLocation decltypeLoc;
  if (!match(TokenKind::T_DECLTYPE, decltypeLoc)) return false;

  SourceLocation lparenLoc;
  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  if (lookat(TokenKind::T_AUTO)) return false;  // placeholder type specifier

  auto ast = make_node<DecltypeSpecifierAST>(pool_);
  yyast = ast;

  ast->decltypeLoc = decltypeLoc;
  ast->lparenLoc = lparenLoc;

  parse_expression(ast->expression, ExprContext{});

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  if (auto id = ast_cast<IdExpressionAST>(ast->expression)) {
    if (id->symbol) ast->type = id->symbol->type();
  } else if (auto member = ast_cast<MemberExpressionAST>(ast->expression)) {
    if (member->symbol) ast->type = member->symbol->type();
  } else if (ast->expression && ast->expression->type) {
    if (is_lvalue(ast->expression)) {
      ast->type = control_->add_lvalue_reference(ast->expression->type);
    } else if (is_xvalue(ast->expression)) {
      ast->type = control_->add_rvalue_reference(ast->expression->type);
    } else {
      ast->type = ast->expression->type;
    }
  }

  return true;
}

auto Parser::parse_placeholder_type_specifier(SpecifierAST*& yyast,
                                              DeclSpecs& specs) -> bool {
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

  if (SourceLocation autoLoc; match(TokenKind::T_AUTO, autoLoc)) {
    auto ast = make_node<AutoTypeSpecifierAST>(pool_);
    yyast = ast;
    ast->autoLoc = autoLoc;

    specs.isAuto = true;
    specs.type = control_->getAutoType();
  } else {
    auto ast = make_node<DecltypeAutoSpecifierAST>(pool_);
    yyast = ast;

    expect(TokenKind::T_DECLTYPE, ast->decltypeLoc);
    expect(TokenKind::T_LPAREN, ast->lparenLoc);
    expect(TokenKind::T_AUTO, ast->autoLoc);
    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    specs.isDecltypeAuto = true;
    specs.type = control_->getDecltypeAutoType();
  }

  if (typeConstraint) {
    auto ast = make_node<PlaceholderTypeSpecifierAST>(pool_);

    ast->typeConstraint = typeConstraint;
    ast->specifier = yyast;

    yyast = ast;
  }

  return true;
}

auto Parser::parse_init_declarator(InitDeclaratorAST*& yyast,
                                   const DeclSpecs& specs, BindingContext ctx)
    -> bool {
  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  if (!parse_declarator(declarator, decl)) return false;

  return parse_init_declarator(yyast, declarator, decl, ctx);
}

auto Parser::parse_init_declarator(InitDeclaratorAST*& yyast,
                                   DeclaratorAST* declarator, Decl& decl,
                                   BindingContext ctx) -> bool {
  Symbol* symbol = nullptr;

  if (auto declId = decl.declaratorId; declId) {
    if (decl.specs.isTypedef) {
      auto typedefSymbol = binder_.declareTypedef(declarator, decl);
      symbol = typedefSymbol;
    } else if (getFunctionPrototype(declarator)) {
      auto functionSymbol = binder_.declareFunction(declarator, decl);
      symbol = functionSymbol;
    } else {
      auto variableSymbol = binder_.declareVariable(declarator, decl);
      symbol = variableSymbol;
    }
  }

  RequiresClauseAST* requiresClause = nullptr;
  ExpressionAST* initializer = nullptr;

  LookaheadParser lookahead{this};
  if (parse_declarator_initializer(requiresClause, initializer)) {
    lookahead.commit();
  }

  auto ast = make_node<InitDeclaratorAST>(pool_);
  yyast = ast;

  ast->declarator = declarator;
  ast->requiresClause = requiresClause;
  ast->initializer = initializer;
  ast->symbol = symbol;

  return true;
}

auto Parser::parse_declarator_initializer(RequiresClauseAST*& requiresClause,
                                          ExpressionAST*& yyast) -> bool {
  if (parse_requires_clause(requiresClause)) return true;

  return parse_initializer(yyast, ExprContext{});
}

void Parser::parse_optional_declarator_or_abstract_declarator(
    DeclaratorAST*& yyast, Decl& decl) {
  (void)parse_declarator(yyast, decl,
                         DeclaratorKind::kDeclaratorOrAbstractDeclarator);
}

auto Parser::parse_declarator(DeclaratorAST*& yyast, Decl& decl,
                              DeclaratorKind declaratorKind) -> bool {
  if (declaratorKind == DeclaratorKind::kDeclaratorOrAbstractDeclarator) {
    if (Decl tempDecl{decl};
        parse_declarator(yyast, tempDecl, DeclaratorKind::kDeclarator)) {
      decl = tempDecl;
      return true;
    }

    if (Decl tempDecl{decl}; parse_declarator(
            yyast, tempDecl, DeclaratorKind::kAbstractDeclarator)) {
      decl = tempDecl;
      return true;
    }

    return false;
  }

  LookaheadParser lookahead{this};

  List<PtrOperatorAST*>* ptrOpList = nullptr;
  (void)parse_ptr_operator_seq(ptrOpList);

  CoreDeclaratorAST* coreDeclarator = nullptr;
  if (!parse_core_declarator(coreDeclarator, decl, declaratorKind)) {
    return false;
  }

  auto _ = Binder::ScopeGuard{&binder_};

  auto q = decl.getNestedNameSpecifier();

  if (auto scope = decl.getScope()) {
    setScope(scope);
  } else if (q && config_.checkTypes) {
    parse_error(q->firstSourceLocation(),
                std::format("unresolved class or namespace"));
  }

  List<DeclaratorChunkAST*>* declaratorChunkList = nullptr;
  auto it = &declaratorChunkList;

  while (LA().isOneOf(TokenKind::T_LPAREN, TokenKind::T_LBRACKET)) {
    if (ArrayDeclaratorChunkAST* arrayDeclaratorChunk = nullptr;
        parse_array_declarator(arrayDeclaratorChunk)) {
      *it = make_list_node<DeclaratorChunkAST>(pool_, arrayDeclaratorChunk);
      it = &(*it)->next;
    } else if (FunctionDeclaratorChunkAST* functionDeclaratorChunk = nullptr;
               declaratorKind != DeclaratorKind::kNewDeclarator &&
               parse_function_declarator(functionDeclaratorChunk)) {
      *it = make_list_node<DeclaratorChunkAST>(pool_, functionDeclaratorChunk);
      it = &(*it)->next;
      if (declaratorKind == DeclaratorKind::kAbstractDeclarator &&
          functionDeclaratorChunk->trailingReturnType) {
        break;
      }
    } else {
      break;
    }
  }

  if (!ptrOpList && !coreDeclarator && !declaratorChunkList) {
    return false;
  }

  lookahead.commit();

  yyast = make_node<DeclaratorAST>(pool_);
  yyast->ptrOpList = ptrOpList;
  yyast->coreDeclarator = coreDeclarator;
  yyast->declaratorChunkList = declaratorChunkList;

  return true;
}

void Parser::parse_optional_abstract_declarator(DeclaratorAST*& yyast,
                                                Decl& decl) {
  LookaheadParser lookahead{this};
  if (!parse_declarator(yyast, decl, DeclaratorKind::kAbstractDeclarator))
    return;
  lookahead.commit();
}

auto Parser::parse_ptr_operator_seq(List<PtrOperatorAST*>*& yyast) -> bool {
  auto it = &yyast;

  PtrOperatorAST* ptrOp = nullptr;

  if (!parse_ptr_operator(ptrOp)) return false;

  *it = make_list_node(pool_, ptrOp);
  it = &(*it)->next;

  ptrOp = nullptr;

  while (parse_ptr_operator(ptrOp)) {
    *it = make_list_node(pool_, ptrOp);
    it = &(*it)->next;
    ptrOp = nullptr;
  }

  return true;
}

auto Parser::parse_core_declarator(CoreDeclaratorAST*& yyast, Decl& decl,
                                   DeclaratorKind declaratorKind) -> bool {
  if (parse_declarator_id(yyast, decl, declaratorKind)) return true;
  if (parse_nested_declarator(yyast, decl, declaratorKind)) return true;
  if (declaratorKind != DeclaratorKind::kDeclarator) return true;
  return false;
}

auto Parser::parse_array_declarator(ArrayDeclaratorChunkAST*& yyast) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation lbracketLoc;
  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  SourceLocation rbracketLoc;
  ExpressionAST* expression = nullptr;
  std::optional<ConstValue> value;

  if (!match(TokenKind::T_RBRACKET, rbracketLoc)) {
    if (!parse_constant_expression(expression, value)) return false;
    if (!match(TokenKind::T_RBRACKET, rbracketLoc)) return false;
  }

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  lookahead.commit();

  auto modifier = make_node<ArrayDeclaratorChunkAST>(pool_);
  yyast = modifier;

  modifier->lbracketLoc = lbracketLoc;
  modifier->expression = expression;
  modifier->rbracketLoc = rbracketLoc;
  modifier->attributeList = attributes;

  return true;
}

auto Parser::parse_function_declarator(FunctionDeclaratorChunkAST*& yyast,
                                       bool acceptTrailingReturnType) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation lparenLoc;
  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto _ = Binder::ScopeGuard{&binder_};

  SourceLocation rparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_parameter_declaration_clause(parameterDeclarationClause)) {
      return false;
    }

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;
  }

  lookahead.commit();

  auto ast = make_node<FunctionDeclaratorChunkAST>(pool_);
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->parameterDeclarationClause = parameterDeclarationClause;
  ast->rparenLoc = rparenLoc;

  DeclSpecs cvQualifiers{unit};

  (void)parse_cv_qualifier_seq(ast->cvQualifierList, cvQualifiers);

  (void)parse_ref_qualifier(ast->refLoc);

  (void)parse_noexcept_specifier(ast->exceptionSpecifier);

  if (acceptTrailingReturnType) {
    (void)parse_trailing_return_type(ast->trailingReturnType);
  }

  parse_optional_attribute_specifier_seq(ast->attributeList,
                                         AllowedAttributes::kAll);

  return true;
}

auto Parser::parse_cv_qualifier_seq(List<SpecifierAST*>*& yyast,
                                    DeclSpecs& declSpecs) -> bool {
  auto it = &yyast;

  SpecifierAST* specifier = nullptr;

  if (!parse_cv_qualifier(specifier, declSpecs)) return false;

  *it = make_list_node(pool_, specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_cv_qualifier(specifier, declSpecs)) {
    *it = make_list_node(pool_, specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

auto Parser::parse_trailing_return_type(TrailingReturnTypeAST*& yyast) -> bool {
  SourceLocation minusGreaterLoc;

  if (!match(TokenKind::T_MINUS_GREATER, minusGreaterLoc)) return false;

  auto ast = make_node<TrailingReturnTypeAST>(pool_);
  yyast = ast;

  ast->minusGreaterLoc = minusGreaterLoc;

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  return true;
}

auto Parser::parse_ptr_operator(PtrOperatorAST*& yyast) -> bool {
  if (SourceLocation starLoc; match(TokenKind::T_STAR, starLoc)) {
    auto ast = make_node<PointerOperatorAST>(pool_);
    yyast = ast;

    ast->starLoc = starLoc;

    parse_optional_attribute_specifier_seq(ast->attributeList);

    DeclSpecs cvQualifiers{unit};
    (void)parse_cv_qualifier_seq(ast->cvQualifierList, cvQualifiers);

    return true;
  } else if (SourceLocation refLoc; parse_ref_qualifier(refLoc)) {
    auto ast = make_node<ReferenceOperatorAST>(pool_);
    yyast = ast;

    ast->refLoc = refLoc;
    ast->refOp = unit->tokenKind(refLoc);

    parse_optional_attribute_specifier_seq(ast->attributeList);

    return true;
  }

  LookaheadParser lookahead{this};

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  if (!parse_nested_name_specifier(nestedNameSpecifier,
                                   NestedNameSpecifierContext::kNonDeclarative))
    return false;

  SourceLocation starLoc;
  if (!match(TokenKind::T_STAR, starLoc)) return false;

  lookahead.commit();

  auto ast = make_node<PtrToMemberOperatorAST>(pool_);
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->starLoc = starLoc;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  DeclSpecs cvQualifiers{unit};
  (void)parse_cv_qualifier_seq(ast->cvQualifierList, cvQualifiers);

  return true;
}

auto Parser::parse_cv_qualifier(SpecifierAST*& yyast, DeclSpecs& declSpecs)
    -> bool {
  SourceLocation loc;

  if (match(TokenKind::T_CONST, loc)) {
    auto ast = make_node<ConstQualifierAST>(pool_);
    yyast = ast;
    ast->constLoc = loc;
    declSpecs.isConst = true;
    return true;
  }
  if (match(TokenKind::T_VOLATILE, loc)) {
    auto ast = make_node<VolatileQualifierAST>(pool_);
    yyast = ast;
    ast->volatileLoc = loc;
    declSpecs.isVolatile = true;
    return true;
  }
  if (match(TokenKind::T___RESTRICT__, loc)) {
    auto ast = make_node<RestrictQualifierAST>(pool_);
    yyast = ast;
    ast->restrictLoc = loc;
    declSpecs.isRestrict = true;
    return true;
  }
  return false;
}

auto Parser::parse_ref_qualifier(SourceLocation& refLoc) -> bool {
  if (match(TokenKind::T_AMP, refLoc)) return true;
  if (match(TokenKind::T_AMP_AMP, refLoc)) return true;
  return false;
}

auto Parser::parse_declarator_id(CoreDeclaratorAST*& yyast, Decl& decl,
                                 DeclaratorKind declaratorKind) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation ellipsisLoc;

  const auto isPack = match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  if (isPack && (declaratorKind == DeclaratorKind::kAbstractDeclarator ||
                 declaratorKind == DeclaratorKind::kNewDeclarator)) {
    lookahead.commit();

    decl.isPack = isPack;

    auto ast = make_node<ParameterPackAST>(pool_);
    ast->ellipsisLoc = ellipsisLoc;
    yyast = ast;

    return true;
  }

  if (declaratorKind != DeclaratorKind::kDeclarator) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kDeclarative);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  check_type_traits();

  UnqualifiedIdAST* unqualifiedId = nullptr;
  if (!parse_unqualified_id(unqualifiedId, nestedNameSpecifier,
                            isTemplateIntroduced,
                            /*inRequiresClause*/ false))
    return false;

  lookahead.commit();

  auto ast = make_node<IdDeclaratorAST>(pool_);
  yyast = ast;

  decl.declaratorId = ast;
  decl.isPack = isPack;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->unqualifiedId = unqualifiedId;
  ast->isTemplateIntroduced = isTemplateIntroduced;

  parse_optional_attribute_specifier_seq(ast->attributeList,
                                         AllowedAttributes::kAll);

  if (isPack) {
    auto ast = make_node<ParameterPackAST>(pool_);
    ast->ellipsisLoc = ellipsisLoc;
    ast->coreDeclarator = yyast;
    yyast = ast;
  }

  return true;
}

auto Parser::parse_type_id(TypeIdAST*& yyast) -> bool {
  List<SpecifierAST*>* specifierList = nullptr;
  DeclSpecs specs{unit};
  if (!parse_type_specifier_seq(specifierList, specs)) return false;

  yyast = make_node<TypeIdAST>(pool_);
  yyast->typeSpecifierList = specifierList;

  Decl decl{specs};
  parse_optional_abstract_declarator(yyast->declarator, decl);

  yyast->type =
      getDeclaratorType(unit, yyast->declarator, decl.specs.getType());

  return true;
}

auto Parser::parse_defining_type_id(
    TypeIdAST*& yyast,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  DeclSpecs specs{unit};

  if (!templateDeclarations.empty()) specs.no_class_or_enum_specs = true;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_defining_type_specifier_seq(typeSpecifierList, specs)) {
    return false;
  }

  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  parse_optional_abstract_declarator(declarator, decl);

  auto ast = make_node<TypeIdAST>(pool_);
  yyast = ast;

  ast->typeSpecifierList = typeSpecifierList;
  ast->declarator = declarator;
  ast->type = getDeclaratorType(unit, ast->declarator, decl.specs.getType());

  return true;
}

auto Parser::parse_nested_declarator(CoreDeclaratorAST*& yyast, Decl& decl,
                                     DeclaratorKind declaratorKind) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation lparenLoc;
  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  DeclaratorAST* declarator = nullptr;
  if (!parse_declarator(declarator, decl, declaratorKind)) return false;

  SourceLocation rparenLoc;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

  lookahead.commit();

  auto ast = make_node<NestedDeclaratorAST>(pool_);
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->declarator = declarator;
  ast->rparenLoc = lparenLoc;

  return true;
}

auto Parser::parse_parameter_declaration_clause(
    ParameterDeclarationClauseAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (auto entry = parameter_declaration_clauses_.get(start)) {
    auto [cursor, ast, parsed, hit] = *entry;
    rewind(cursor);
    yyast = ast;
    return parsed;
  }

  auto _ = Binder::ScopeGuard{&binder_};

  auto ast = make_node<ParameterDeclarationClauseAST>(pool_);

  binder_.bind(ast);

  if (match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc)) {
    yyast = ast;

    ast->isVariadic = true;
  } else if (parse_parameter_declaration_list(ast)) {
    yyast = ast;

    match(TokenKind::T_COMMA, ast->commaLoc);
    ast->isVariadic = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);
  }

  const auto parsed = yyast != nullptr;

  parameter_declaration_clauses_.set(start, currentLocation(), yyast, parsed);

  return parsed;
}

auto Parser::parse_parameter_declaration_list(
    ParameterDeclarationClauseAST* ast) -> bool {
  auto it = &ast->parameterDeclarationList;

  auto _ = Binder::ScopeGuard{&binder_};

  setScope(ast->functionParametersSymbol);

  ParameterDeclarationAST* declaration = nullptr;

  if (!parse_parameter_declaration(declaration, /*templParam*/ false)) {
    return false;
  }

  *it = make_list_node(pool_, declaration);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    ParameterDeclarationAST* declaration = nullptr;

    if (!parse_parameter_declaration(declaration, /*templParam*/ false)) {
      rewind(commaLoc);
      break;
    }

    *it = make_list_node(pool_, declaration);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_parameter_declaration(ParameterDeclarationAST*& yyast,
                                         bool templParam) -> bool {
  auto ast = make_node<ParameterDeclarationAST>(pool_);
  yyast = ast;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  DeclSpecs specs{unit};

  specs.no_class_or_enum_specs = true;

  ast->isThisIntroduced = match(TokenKind::T_THIS, ast->thisLoc);

  if (!parse_decl_specifier_seq(ast->typeSpecifierList, specs, {}))
    return false;

  Decl decl{specs};
  parse_optional_declarator_or_abstract_declarator(ast->declarator, decl);

  ast->isPack = decl.isPack;

  binder_.bind(ast, decl, templParam);

  if (match(TokenKind::T_EQUAL, ast->equalLoc)) {
    ExprContext ctx;
    ctx.templParam = templParam;
    if (!parse_initializer_clause(ast->expression, ctx)) {
      if (templParam) return false;

      parse_error("expected an initializer");
    }
  }

  return true;
}

auto Parser::parse_initializer(ExpressionAST*& yyast, const ExprContext& ctx)
    -> bool {
  SourceLocation lparenLoc;

  if (match(TokenKind::T_LPAREN, lparenLoc)) {
    if (lookat(TokenKind::T_RPAREN)) return false;

    auto ast = make_node<ParenInitializerAST>(pool_);
    yyast = ast;

    ast->lparenLoc = lparenLoc;

    if (!parse_expression_list(ast->expressionList, ctx)) {
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
    if (!parse_braced_init_list(bracedInitList, ExprContext{})) return false;
    yyast = bracedInitList;
    return true;
  }

  SourceLocation equalLoc;

  if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

  auto ast = make_node<EqualInitializerAST>(pool_);
  yyast = ast;

  ast->equalLoc = equalLoc;

  if (!parse_initializer_clause(ast->expression, ExprContext{})) {
    parse_error("expected an intializer");
  }

  if (ast->expression) {
    ast->type = ast->expression->type;
    ast->valueCategory = ast->expression->valueCategory;
  }

  return true;
}

auto Parser::parse_initializer_clause(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  BracedInitListAST* bracedInitList = nullptr;
  if (parse_braced_init_list(bracedInitList, ctx)) {
    yyast = bracedInitList;
    return true;
  }

  parse_assignment_expression(yyast, ctx);
  return true;
}

auto Parser::parse_braced_init_list(BracedInitListAST*& ast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation lbraceLoc;
  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  if (!ast) {
    ast = make_node<BracedInitListAST>(pool_);
  }

  ast->lbraceLoc = lbraceLoc;

  if (lookat(TokenKind::T_DOT)) {
    auto it = &ast->expressionList;

    DesignatedInitializerClauseAST* designatedInitializerClause = nullptr;

    if (!parse_designated_initializer_clause(designatedInitializerClause)) {
      parse_error("expected designated initializer clause");
    }

    if (designatedInitializerClause) {
      *it = make_list_node<ExpressionAST>(pool_, designatedInitializerClause);
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
        *it = make_list_node<ExpressionAST>(pool_, designatedInitializerClause);
        it = &(*it)->next;
      }
    }

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);

    return true;
  }

  if (match(TokenKind::T_COMMA, ast->commaLoc)) {
    expect(TokenKind::T_RBRACE, ast->rbraceLoc);

    return true;
  }

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    if (!parse_initializer_list(ast->expressionList, ctx)) {
      parse_error("expected initializer list");
    }

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  return true;
}

auto Parser::parse_initializer_list(List<ExpressionAST*>*& yyast,
                                    const ExprContext& ctx) -> bool {
  auto it = &yyast;

  ExpressionAST* expression = nullptr;

  if (!parse_initializer_clause(expression, ctx)) return false;

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto pack = make_node<PackExpansionExpressionAST>(pool_);
    pack->expression = expression;
    pack->ellipsisLoc = ellipsisLoc;
    expression = pack;
  }

  *it = make_list_node(pool_, expression);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (lookat(TokenKind::T_RBRACE)) break;

    ExpressionAST* expression = nullptr;

    if (!parse_initializer_clause(expression, ctx)) {
      parse_error("expected initializer clause");
    }

    SourceLocation ellipsisLoc;

    if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
      auto pack = make_node<PackExpansionExpressionAST>(pool_);
      pack->expression = expression;
      pack->ellipsisLoc = ellipsisLoc;
      expression = pack;
    }

    *it = make_list_node(pool_, expression);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_designated_initializer_clause(
    DesignatedInitializerClauseAST*& yyast) -> bool {
  SourceLocation dotLoc;
  if (!match(TokenKind::T_DOT, dotLoc)) return false;

  auto ast = make_node<DesignatedInitializerClauseAST>(pool_);
  yyast = ast;

  ast->dotLoc = dotLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  if (!parse_brace_or_equal_initializer(ast->initializer)) {
    parse_error("expected an initializer");
  }

  return true;
}

void Parser::parse_expr_or_braced_init_list(ExpressionAST*& yyast,
                                            const ExprContext& ctx) {
  BracedInitListAST* bracedInitList = nullptr;

  if (parse_braced_init_list(bracedInitList, ctx)) {
    yyast = bracedInitList;
  } else {
    parse_expression(yyast, ctx);
  }
}

void Parser::parse_virt_specifier_seq(
    FunctionDeclaratorChunkAST* functionDeclarator) {
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
      auto ast = make_node<DefaultFunctionBodyAST>(pool_);
      yyast = ast;

      ast->equalLoc = equalLoc;
      ast->defaultLoc = defaultLoc;

      expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

      return true;
    }

    SourceLocation deleteLoc;

    if (match(TokenKind::T_DELETE, deleteLoc)) {
      auto ast = make_node<DeleteFunctionBodyAST>(pool_);
      yyast = ast;

      ast->equalLoc = equalLoc;
      ast->deleteLoc = deleteLoc;

      expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

      return true;
    }

    return false;
  }

  SourceLocation colonLoc;
  List<MemInitializerAST*>* memInitializerList = nullptr;

  (void)parse_ctor_initializer(colonLoc, memInitializerList);

  if (!lookat(TokenKind::T_LBRACE)) return false;

  auto ast = make_node<CompoundStatementFunctionBodyAST>(pool_);
  yyast = ast;

  ast->colonLoc = colonLoc;
  ast->memInitializerList = memInitializerList;

  const bool skip = skipFunctionBody_ || classDepth_ > 0;

  if (!parse_compound_statement(ast->statement, skip)) {
    parse_error("expected a compound statement");
  }

  return true;
}

auto Parser::parse_enum_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  LookaheadParser lookahead{this};

  SourceLocation enumLoc;
  SourceLocation classLoc;

  if (!parse_enum_key(enumLoc, classLoc)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;

  parse_optional_attribute_specifier_seq(attributes);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  NameIdAST* name = nullptr;

  (void)parse_enum_head_name(nestedNameSpecifier, name);

  SourceLocation colonLoc;
  List<SpecifierAST*>* typeSpecifierList = nullptr;

  DeclSpecs underlyingTypeSpecs{unit};
  (void)parse_enum_base(colonLoc, typeSpecifierList, underlyingTypeSpecs);

  SourceLocation lbraceLoc;
  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  lookahead.commit();

  auto _ = Binder::ScopeGuard{&binder_};

  auto ast = make_node<EnumSpecifierAST>(pool_);
  yyast = ast;

  ast->enumLoc = enumLoc;
  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = name;
  ast->colonLoc = colonLoc;
  ast->typeSpecifierList = typeSpecifierList;
  ast->lbraceLoc = lbraceLoc;

  binder_.bind(ast, underlyingTypeSpecs);

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    parse_enumerator_list(ast->enumeratorList, ast->symbol->type());

    match(TokenKind::T_COMMA, ast->commaLoc);

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  return true;
}

auto Parser::parse_enum_head_name(NestedNameSpecifierAST*& nestedNameSpecifier,
                                  NameIdAST*& name) -> bool {
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kDeclarative);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = make_node<NameIdAST>(pool_);
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
  NameIdAST* name = nullptr;
  SourceLocation colonLoc;
  DeclSpecs underlyingTypeSpecs{unit};
  List<SpecifierAST*>* typeSpecifierList = nullptr;
  SourceLocation semicolonLoc;

  auto lookat_opaque_enum_declaration = [&] {
    LookaheadParser lookahead{this};
    parse_optional_attribute_specifier_seq(attributes);
    if (!parse_enum_key(enumLoc, classLoc)) return false;
    if (!parse_enum_head_name(nestedNameSpecifier, name)) return false;
    (void)parse_enum_base(colonLoc, typeSpecifierList, underlyingTypeSpecs);
    if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;
    lookahead.commit();
    return true;
  };

  if (!lookat_opaque_enum_declaration()) return false;

  auto ast = make_node<OpaqueEnumDeclarationAST>(pool_);
  yyast = ast;

  ast->enumLoc = enumLoc;
  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = name;
  ast->colonLoc = colonLoc;
  ast->typeSpecifierList = typeSpecifierList;
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

auto Parser::parse_enum_base(SourceLocation& colonLoc,
                             List<SpecifierAST*>*& typeSpecifierList,
                             DeclSpecs& specs) -> bool {
  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  if (!parse_type_specifier_seq(typeSpecifierList, specs)) {
    parse_error("expected a type specifier");
  }

  return true;
}

void Parser::parse_enumerator_list(List<EnumeratorAST*>*& yyast,
                                   const Type* type) {
  auto it = &yyast;

  EnumeratorAST* enumerator = nullptr;
  parse_enumerator(enumerator, type);

  *it = make_list_node(pool_, enumerator);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (lookat(TokenKind::T_RBRACE)) {
      rewind(commaLoc);
      break;
    }

    EnumeratorAST* enumerator = nullptr;
    parse_enumerator(enumerator, type);

    *it = make_list_node(pool_, enumerator);
    it = &(*it)->next;
  }
}

void Parser::parse_enumerator(EnumeratorAST*& yyast, const Type* type) {
  auto ast = make_node<EnumeratorAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  parse_optional_attribute_specifier_seq(ast->attributeList);

  std::optional<ConstValue> value;

  if (match(TokenKind::T_EQUAL, ast->equalLoc)) {
    if (!parse_constant_expression(ast->expression, value)) {
      parse_error("expected an expression");
    }
  }

  binder_.bind(ast, type, std::move(value));
}

auto Parser::parse_using_enum_declaration(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_USING, TokenKind::T_ENUM)) return false;

  auto ast = make_node<UsingEnumDeclarationAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_USING, ast->usingLoc);

  DeclSpecs specs{unit};

  SpecifierAST* typeSpecifier = nullptr;
  if (!parse_elaborated_type_specifier(typeSpecifier, specs, {})) {
    parse_error("expected an elaborated enum specifier");
  }

  ast->enumTypeSpecifier = ast_cast<ElaboratedTypeSpecifierAST>(typeSpecifier);

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

  auto _ = Binder::ScopeGuard{&binder_};

  auto ast = make_node<NamespaceDefinitionAST>(pool_);
  yyast = ast;

  ast->isInline = match(TokenKind::T_INLINE, ast->inlineLoc);

  expect(TokenKind::T_NAMESPACE, ast->namespaceLoc);

  parse_optional_attribute_specifier_seq(ast->attributeList);

  if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON_COLON)) {
    auto it = &ast->nestedNamespaceSpecifierList;

    auto name = make_node<NestedNamespaceSpecifierAST>(pool_);

    expect(TokenKind::T_IDENTIFIER, name->identifierLoc);
    expect(TokenKind::T_COLON_COLON, name->scopeLoc);

    name->identifier = unit->identifier(name->identifierLoc);

    *it = make_list_node(pool_, name);
    it = &(*it)->next;

    auto namepaceSymbol = enterOrCreateNamespace(
        name->identifier, name->identifierLoc, /*isInline*/ false);

    LoopParser loop{this};

    while (true) {
      loop.start();

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

      auto namespaceName = unit->identifier(identifierLoc);

      auto namepaceSymbol =
          enterOrCreateNamespace(namespaceName, identifierLoc, isInline);

      auto name = make_node<NestedNamespaceSpecifierAST>(pool_);
      name->inlineLoc = inlineLoc;
      name->identifierLoc = identifierLoc;
      name->scopeLoc = scopeLoc;
      name->identifier = namespaceName;
      name->isInline = isInline;

      *it = make_list_node(pool_, name);
      it = &(*it)->next;
    }
  }

  if (ast->nestedNamespaceSpecifierList) {
    ast->isInline = match(TokenKind::T_INLINE, ast->inlineLoc);
    expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  } else {
    match(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  }

  ast->identifier = unit->identifier(ast->identifierLoc);

  auto location = ast->identifierLoc ? ast->identifierLoc : currentLocation();

  auto namespaceSymbol =
      enterOrCreateNamespace(ast->identifier, location, ast->isInline);

  parse_optional_attribute_specifier_seq(ast->extraAttributeList);

  expect(TokenKind::T_LBRACE, ast->lbraceLoc);

  parse_namespace_body(ast);

  expect(TokenKind::T_RBRACE, ast->rbraceLoc);

  return true;
}

void Parser::parse_namespace_body(NamespaceDefinitionAST* yyast) {
  auto it = &yyast->declarationList;

  LoopParser loop{this};

  while (LA()) {
    if (shouldStopParsing()) break;

    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    const auto beforeDeclaration = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_declaration(declaration, BindingContext::kNamespace)) {
      if (declaration) {
        *it = make_list_node(pool_, declaration);
        it = &(*it)->next;
      }
    } else {
      parse_error("expected a declaration");
    }
  }
}

auto Parser::parse_namespace_alias_definition(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_NAMESPACE, TokenKind::T_IDENTIFIER,
              TokenKind::T_EQUAL)) {
    return false;
  }

  auto ast = make_node<NamespaceAliasDefinitionAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_NAMESPACE, ast->namespaceLoc);
  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  expect(TokenKind::T_EQUAL, ast->equalLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  if (!parse_qualified_namespace_specifier(ast->nestedNameSpecifier,
                                           ast->unqualifiedId)) {
    parse_error("expected a namespace name");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_qualified_namespace_specifier(
    NestedNameSpecifierAST*& nestedNameSpecifier, NameIdAST*& name) -> bool {
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = make_node<NameIdAST>(pool_);
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

  auto ast = make_node<UsingDirectiveAST>(pool_);
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->namespaceLoc = namespaceLoc;

  parse_optional_nested_name_specifier(
      ast->nestedNameSpecifier, NestedNameSpecifierContext::kDeclarative);

  auto currentNamespace = scope()->owner();

  if (!parse_name_id(ast->unqualifiedId)) {
    parse_error("expected a namespace name");
  } else {
    auto id = ast->unqualifiedId->identifier;

    NamespaceSymbol* namespaceSymbol =
        Lookup{scope()}.lookupNamespace(ast->nestedNameSpecifier, id);

    if (namespaceSymbol) {
      scope()->addUsingDirective(namespaceSymbol->scope());
    } else {
      parse_error(ast->unqualifiedId->firstSourceLocation(),
                  std::format("'{}' is not a namespace name", id->name()));
    }
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_using_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  auto ast = make_node<UsingDeclarationAST>(pool_);
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

  *it = make_list_node(pool_, declarator);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (UsingDeclaratorAST* declarator = nullptr;
        parse_using_declarator(declarator)) {
      *it = make_list_node(pool_, declarator);
      it = &(*it)->next;
    } else {
      parse_error("expected a using declarator");
    }
  }

  return true;
}

auto Parser::parse_using_declarator(UsingDeclaratorAST*& yyast) -> bool {
  SourceLocation typenameLoc;

  match(TokenKind::T_TYPENAME, typenameLoc);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;

  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

  UnqualifiedIdAST* unqualifiedId = nullptr;

  if (!parse_unqualified_id(unqualifiedId, nestedNameSpecifier,
                            /*isTemplateIntroduced*/ false,
                            /*inRequiresClause*/ false))
    return false;

  auto name = convertName(unqualifiedId);
  auto target = Lookup{scope()}.lookup(nestedNameSpecifier, name);

  SourceLocation ellipsisLoc;
  auto isPack = match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  yyast = make_node<UsingDeclaratorAST>(pool_);
  yyast->typenameLoc = typenameLoc;
  yyast->nestedNameSpecifier = nestedNameSpecifier;
  yyast->unqualifiedId = unqualifiedId;
  yyast->ellipsisLoc = ellipsisLoc;
  yyast->isPack = isPack;

  binder_.bind(yyast, target);

  return true;
}

auto Parser::parse_asm_operand(AsmOperandAST*& yyast) -> bool {
  if (!LA().isOneOf(TokenKind::T_LBRACKET, TokenKind::T_STRING_LITERAL))
    return false;

  auto ast = make_node<AsmOperandAST>(pool_);
  yyast = ast;

  if (match(TokenKind::T_LBRACKET, ast->lbracketLoc)) {
    expect(TokenKind::T_IDENTIFIER, ast->symbolicNameLoc);
    ast->symbolicName = unit->identifier(ast->symbolicNameLoc);
    expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  }

  expect(TokenKind::T_STRING_LITERAL, ast->constraintLiteralLoc);

  ast->constraintLiteral = static_cast<const StringLiteral*>(
      unit->literal(ast->constraintLiteralLoc));

  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  parse_expression(ast->expression, ExprContext{});
  expect(TokenKind::T_RPAREN, ast->rparenLoc);

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

  auto ast = make_node<AsmDeclarationAST>(pool_);
  yyast = ast;

  ast->attributeList = attributes;
  ast->asmLoc = asmLoc;

  auto it = &ast->asmQualifierList;
  while (LA().isOneOf(TokenKind::T_INLINE, TokenKind::T_VOLATILE,
                      TokenKind::T_GOTO)) {
    auto qualifier = make_node<AsmQualifierAST>(pool_);
    qualifier->qualifierLoc = consumeToken();
    *it = make_list_node(pool_, qualifier);
    it = &(*it)->next;
  }

  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  expect(TokenKind::T_STRING_LITERAL, ast->literalLoc);

  if (SourceLocation colonLoc; match(TokenKind::T_COLON, colonLoc)) {
    if (AsmOperandAST* operand = nullptr; parse_asm_operand(operand)) {
      auto it = &ast->outputOperandList;
      *it = make_list_node(pool_, operand);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        if (AsmOperandAST* operand = nullptr; parse_asm_operand(operand)) {
          *it = make_list_node(pool_, operand);
          it = &(*it)->next;
        } else {
          parse_error("expected an asm operand");
        }
      }
    }
  }
  if (SourceLocation colonLoc; match(TokenKind::T_COLON, colonLoc)) {
    if (AsmOperandAST* operand = nullptr; parse_asm_operand(operand)) {
      auto it = &ast->inputOperandList;
      *it = make_list_node(pool_, operand);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        if (AsmOperandAST* operand = nullptr; parse_asm_operand(operand)) {
          *it = make_list_node(pool_, operand);
          it = &(*it)->next;
        } else {
          parse_error("expected an asm operand");
        }
      }
    }
  }

  if (SourceLocation colonLoc; match(TokenKind::T_COLON, colonLoc)) {
    if (SourceLocation literalLoc;
        match(TokenKind::T_STRING_LITERAL, literalLoc)) {
      auto it = &ast->clobberList;
      auto clobber = make_node<AsmClobberAST>(pool_);
      clobber->literalLoc = literalLoc;
      clobber->literal =
          static_cast<const StringLiteral*>(unit->literal(literalLoc));
      *it = make_list_node(pool_, clobber);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        SourceLocation literalLoc;
        expect(TokenKind::T_STRING_LITERAL, literalLoc);
        if (!literalLoc) continue;
        auto clobber = make_node<AsmClobberAST>(pool_);
        clobber->literalLoc = literalLoc;
        clobber->literal =
            static_cast<const StringLiteral*>(unit->literal(literalLoc));
        *it = make_list_node(pool_, clobber);
        it = &(*it)->next;
      }
    }
  }

  if (SourceLocation colonLoc; match(TokenKind::T_COLON, colonLoc)) {
    if (SourceLocation identifierLoc;
        match(TokenKind::T_IDENTIFIER, identifierLoc)) {
      auto it = &ast->gotoLabelList;
      auto label = make_node<AsmGotoLabelAST>(pool_);
      label->identifierLoc = identifierLoc;
      label->identifier = unit->identifier(label->identifierLoc);
      *it = make_list_node(pool_, label);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        SourceLocation identifierLoc;
        expect(TokenKind::T_IDENTIFIER, identifierLoc);
        if (!identifierLoc) continue;
        auto label = make_node<AsmGotoLabelAST>(pool_);
        label->identifierLoc = identifierLoc;
        label->identifier = unit->identifier(label->identifierLoc);
        *it = make_list_node(pool_, label);
        it = &(*it)->next;
      }
    }
  }

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

    auto ast = make_node<LinkageSpecificationAST>(pool_);
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

  if (!parse_declaration(declaration, BindingContext::kNamespace)) return false;

  auto ast = make_node<LinkageSpecificationAST>(pool_);
  yyast = ast;

  ast->externLoc = externLoc;
  ast->stringliteralLoc = stringLiteralLoc;
  ast->stringLiteral =
      static_cast<const StringLiteral*>(unit->literal(ast->stringliteralLoc));
  ast->declarationList = make_list_node(pool_, declaration);

  return true;
}

void Parser::parse_optional_attribute_specifier_seq(
    List<AttributeSpecifierAST*>*& yyast, AllowedAttributes allowedAttributes) {
  if (!parse_attribute_specifier_seq(yyast, allowedAttributes)) {
    yyast = nullptr;
  }
}

auto Parser::parse_attribute_specifier_seq(List<AttributeSpecifierAST*>*& yyast,
                                           AllowedAttributes allowedAttributes)
    -> bool {
  auto it = &yyast;
  AttributeSpecifierAST* attribute = nullptr;

  if (!parse_attribute_specifier(attribute, allowedAttributes)) return false;

  *it = make_list_node(pool_, attribute);
  it = &(*it)->next;

  attribute = nullptr;

  while (parse_attribute_specifier(attribute, allowedAttributes)) {
    *it = make_list_node(pool_, attribute);
    it = &(*it)->next;
    attribute = nullptr;
  }

  return true;
}

auto Parser::parse_attribute_specifier(AttributeSpecifierAST*& yyast,
                                       AllowedAttributes allowedAttributes)
    -> bool {
  auto is_allowed = [allowedAttributes](AllowedAttributes attr) {
    return static_cast<int>(attr) & static_cast<int>(allowedAttributes);
  };

  if (is_allowed(AllowedAttributes::kCxxAttribute) &&
      parse_cxx_attribute_specifier(yyast))
    return true;

  if (is_allowed(AllowedAttributes::kGnuAttribute) &&
      parse_gcc_attribute(yyast))
    return true;

  if (is_allowed(AllowedAttributes::kAlignasSpecifier) &&
      parse_alignment_specifier(yyast))
    return true;

  if (is_allowed(AllowedAttributes::kAsmSpecifier) &&
      parse_asm_specifier(yyast))
    return true;

  return false;
}

auto Parser::lookat_cxx_attribute_specifier() -> bool {
  if (!lookat(TokenKind::T_LBRACKET)) return false;
  if (LA(1).isNot(TokenKind::T_LBRACKET)) return false;
  return true;
}

auto Parser::parse_cxx_attribute_specifier(AttributeSpecifierAST*& yyast)
    -> bool {
  if (!lookat_cxx_attribute_specifier()) return false;

  auto ast = make_node<CxxAttributeAST>(pool_);
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

  auto ast = make_node<AsmAttributeAST>(pool_);
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

  auto ast = make_node<GccAttributeAST>(pool_);
  yyast = ast;

  ast->attributeLoc = attributeLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  (void)parse_skip_balanced();

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_skip_balanced() -> bool {
  if (lookat(TokenKind::T_EOF_SYMBOL)) {
    return false;
  }

  if (SourceLocation lbraceLoc; match(TokenKind::T_LBRACE, lbraceLoc)) {
    while (!lookat(TokenKind::T_EOF_SYMBOL)) {
      if (SourceLocation rbraceLoc; match(TokenKind::T_RBRACE, rbraceLoc)) {
        break;
      }
      if (!parse_skip_balanced()) return false;
    }
  } else if (SourceLocation lbracketLoc;
             match(TokenKind::T_LBRACKET, lbracketLoc)) {
    while (!lookat(TokenKind::T_EOF_SYMBOL)) {
      if (SourceLocation rbracketLoc;
          match(TokenKind::T_RBRACKET, rbracketLoc)) {
        break;
      }
      if (!parse_skip_balanced()) return false;
    }
  } else if (SourceLocation lparenLoc; match(TokenKind::T_LPAREN, lparenLoc)) {
    while (!lookat(TokenKind::T_EOF_SYMBOL)) {
      if (SourceLocation rparenLoc; match(TokenKind::T_RPAREN, rparenLoc)) {
        break;
      }
      if (!parse_skip_balanced()) return false;
    }
  } else {
    (void)consumeToken();
  }

  return true;
}

auto Parser::parse_alignment_specifier(AttributeSpecifierAST*& yyast) -> bool {
  SourceLocation alignasLoc;
  if (!match(TokenKind::T_ALIGNAS, alignasLoc)) return false;

  auto lookat_type_id = [&] {
    LookaheadParser lookahead{this};

    SourceLocation lparenLoc;
    if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

    TypeIdAST* typeId = nullptr;
    if (!parse_type_id(typeId)) return false;

    SourceLocation ellipsisLoc;
    const auto isPack = match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    SourceLocation rparenLoc;
    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;

    lookahead.commit();

    auto ast = make_node<AlignasTypeAttributeAST>(pool_);
    yyast = ast;

    ast->alignasLoc = alignasLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->ellipsisLoc = ellipsisLoc;
    ast->rparenLoc = rparenLoc;

    ast->isPack = isPack;

    return true;
  };

  if (lookat_type_id()) return true;

  auto ast = make_node<AlignasAttributeAST>(pool_);
  yyast = ast;

  ast->alignasLoc = alignasLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  std::optional<ConstValue> value;

  if (!parse_constant_expression(ast->expression, value)) {
    parse_error("expected an expression");
  }

  ast->isPack = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

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

  auto ast = make_node<AttributeUsingPrefixAST>(pool_);
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

    *it = make_list_node(pool_, attribute);
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

      *it = make_list_node(pool_, attribute);
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

  auto ast = make_node<AttributeAST>(pool_);
  yyast = ast;

  ast->attributeToken = attributeToken;
  ast->attributeArgumentClause = attributeArgumentClause;

  return true;
}

auto Parser::parse_attribute_token(AttributeTokenAST*& yyast) -> bool {
  if (parse_attribute_scoped_token(yyast)) return true;

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = make_node<SimpleAttributeTokenAST>(pool_);
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_attribute_scoped_token(AttributeTokenAST*& yyast) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation attributeNamespaceLoc;

  if (!parse_attribute_namespace(attributeNamespaceLoc)) return false;

  SourceLocation scopeLoc;

  if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;

  lookahead.commit();

  SourceLocation identifierLoc;

  expect(TokenKind::T_IDENTIFIER, identifierLoc);

  auto ast = make_node<ScopedAttributeTokenAST>(pool_);
  yyast = ast;

  ast->attributeNamespaceLoc = attributeNamespaceLoc;
  ast->scopeLoc = scopeLoc;
  ast->identifierLoc = identifierLoc;
  ast->attributeNamespace = unit->identifier(ast->attributeNamespaceLoc);
  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_attribute_namespace(SourceLocation& attributeNamespaceLoc)
    -> bool {
  if (!match(TokenKind::T_IDENTIFIER, attributeNamespaceLoc)) return false;

  return true;
}

auto Parser::parse_attribute_argument_clause(AttributeArgumentClauseAST*& yyast)
    -> bool {
  const SourceLocation lparenLoc = currentLocation();

  if (!lookat(TokenKind::T_LPAREN)) return false;

  SourceLocation rparenLoc;
  if (parse_skip_balanced()) {
    rparenLoc = currentLocation().previous();
  } else {
    expect(TokenKind::T_RPAREN, rparenLoc);
  }

  auto ast = make_node<AttributeArgumentClauseAST>(pool_);
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

  yyast = make_node<ModuleDeclarationAST>(pool_);

  yyast->exportLoc = exportLoc;
  yyast->moduleLoc = moduleLoc;
  parse_module_name(yyast->moduleName);

  (void)parse_module_partition(yyast->modulePartition);

  parse_optional_attribute_specifier_seq(yyast->attributeList);

  expect(TokenKind::T_SEMICOLON, yyast->semicolonLoc);

  return true;
}

void Parser::parse_module_name(ModuleNameAST*& yyast) {
  auto ast = make_node<ModuleNameAST>(pool_);
  yyast = ast;

  if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_DOT)) {
    ast->moduleQualifier = make_node<ModuleQualifierAST>(pool_);
    ast->moduleQualifier->identifierLoc = consumeToken();
    ast->moduleQualifier->identifier =
        unit->identifier(ast->moduleQualifier->identifierLoc);
    ast->moduleQualifier->dotLoc = consumeToken();

    while (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_DOT)) {
      auto baseModuleQualifier = ast->moduleQualifier;
      ast->moduleQualifier = make_node<ModuleQualifierAST>(pool_);
      ast->moduleQualifier->moduleQualifier = baseModuleQualifier;
      ast->moduleQualifier->identifierLoc = consumeToken();
      ast->moduleQualifier->identifier =
          unit->identifier(ast->moduleQualifier->identifierLoc);
      ast->moduleQualifier->dotLoc = consumeToken();
    }
  }

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  ast->identifier = unit->identifier(ast->identifierLoc);
}

auto Parser::parse_module_partition(ModulePartitionAST*& yyast) -> bool {
  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  yyast = make_node<ModulePartitionAST>(pool_);

  yyast->colonLoc = colonLoc;

  parse_module_name(yyast->moduleName);

  return true;
}

auto Parser::parse_export_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation exportLoc;

  if (!match(TokenKind::T_EXPORT, exportLoc)) return false;

  SourceLocation lbraceLoc;

  if (match(TokenKind::T_LBRACE, lbraceLoc)) {
    auto ast = make_node<ExportCompoundDeclarationAST>(pool_);
    yyast = ast;

    ast->exportLoc = exportLoc;
    ast->lbraceLoc = lbraceLoc;

    if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
      parse_declaration_seq(ast->declarationList);
      expect(TokenKind::T_RBRACE, ast->rbraceLoc);
    }

    return true;
  }

  auto ast = make_node<ExportDeclarationAST>(pool_);
  yyast = ast;

  ast->exportLoc = exportLoc;

  if (parse_maybe_import()) {
    if (!parse_module_import_declaration(ast->declaration)) {
      parse_error("expected a module import declaration");
    }

    return true;
  }

  if (!parse_declaration(ast->declaration, BindingContext::kNamespace)) {
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

  auto ast = make_node<ModuleImportDeclarationAST>(pool_);
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

  yyast = make_node<ImportNameAST>(pool_);

  yyast->headerLoc = headerLoc;

  if (parse_module_partition(yyast->modulePartition)) return true;

  parse_module_name(yyast->moduleName);

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

  yyast = make_node<GlobalModuleFragmentAST>(pool_);
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

  yyast = make_node<PrivateModuleFragmentAST>(pool_);

  yyast->moduleLoc = moduleLoc;
  yyast->colonLoc = colonLoc;
  yyast->privateLoc = privateLoc;

  expect(TokenKind::T_SEMICOLON, yyast->semicolonLoc);

  parse_declaration_seq(yyast->declarationList);
}

auto Parser::parse_class_specifier(ClassSpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  std::vector<TemplateDeclarationAST*> templateDeclarations;
  return parse_class_specifier(yyast, specs, templateDeclarations);
}

auto Parser::parse_class_specifier(
    ClassSpecifierAST*& yyast, DeclSpecs& specs,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  SourceLocation classLoc;
  if (!parse_class_key(classLoc)) return false;

  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  SimpleTemplateIdAST* templateId = nullptr;
  const Identifier* className = nullptr;
  ClassSymbol* symbol = nullptr;
  SourceLocation finalLoc;
  bool isUnion = false;
  bool isTemplateSpecialization = false;
  SourceLocation location = classLoc;

  auto lookat_class_head = [&] {
    LookaheadParser lookahead{this};

    isUnion = unit->tokenKind(classLoc) == TokenKind::T_UNION;

    parse_optional_attribute_specifier_seq(attributeList);

    parse_optional_nested_name_specifier(
        nestedNameSpecifier, NestedNameSpecifierContext::kDeclarative);

    if (lookat(TokenKind::T_IDENTIFIER)) {
      check_type_traits();

      if (parse_simple_template_id(templateId)) {
        unqualifiedId = templateId;
        className = templateId->identifier;
        isTemplateSpecialization = true;
        location = templateId->firstSourceLocation();
      } else {
        NameIdAST* nameId = nullptr;
        (void)parse_name_id(nameId);
        unqualifiedId = nameId;
        className = nameId->identifier;
        location = nameId->firstSourceLocation();
      }

      (void)parse_class_virt_specifier(finalLoc);
    }

    if (nestedNameSpecifier && !className) {
      parse_error("expected class name");
    }

    if (!LA().isOneOf(TokenKind::T_COLON, TokenKind::T_LBRACE)) {
      return false;
    }

    lookahead.commit();
    return true;
  };

  if (!lookat_class_head()) return false;

  auto ast = make_node<ClassSpecifierAST>(pool_);
  yyast = ast;

  ast->classLoc = classLoc;
  ast->attributeList = attributeList;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = unqualifiedId;
  ast->finalLoc = finalLoc;

  ast->classKey = unit->tokenKind(ast->classLoc);

  if (finalLoc) {
    ast->isFinal = true;
  }

  if (scope()->isTemplateParametersScope()) {
    mark_maybe_template_name(unqualifiedId);
  }

  auto _ = Binder::ScopeGuard{&binder_};

  binder_.bind(ast, specs);

  setScope(ast->symbol);

  (void)parse_base_clause(ast);

  expect(TokenKind::T_LBRACE, ast->lbraceLoc);

  ClassSpecifierContext classContext(this);

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    parse_class_body(ast->declarationList);
    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  binder_.complete(ast);

  return true;
}

void Parser::parse_class_body(List<DeclarationAST*>*& yyast) {
  auto it = &yyast;

  LoopParser loop{this};

  while (LA()) {
    if (shouldStopParsing()) break;

    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    const auto saved = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_member_specification(declaration)) {
      if (declaration) {
        *it = make_list_node(pool_, declaration);
        it = &(*it)->next;
      }
    } else {
      parse_error("expected a declaration");
    }
  }
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
    auto ast = make_node<AccessDeclarationAST>(pool_);
    yyast = ast;

    ast->accessLoc = accessLoc;
    expect(TokenKind::T_COLON, ast->colonLoc);

    ast->accessSpecifier = unit->tokenKind(ast->accessLoc);

    return true;
  } else if (parse_empty_declaration(yyast)) {
    return true;
  } else if (parse_using_enum_declaration(yyast)) {
    return true;
  } else if (parse_alias_declaration(yyast)) {
    return true;
  } else if (parse_using_declaration(yyast)) {
    return true;
  } else if (parse_static_assert_declaration(yyast)) {
    return true;
  } else if (parse_deduction_guide(yyast)) {
    return true;
  } else if (parse_opaque_enum_declaration(yyast)) {
    return true;
  } else if (TemplateDeclarationAST* templateDeclaration = nullptr;
             parse_template_declaration(templateDeclaration)) {
    yyast = templateDeclaration;
    return true;
  } else {
    return parse_member_declaration_helper(yyast);
  }
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

  List<SpecifierAST*>* declSpecifierList = nullptr;
  DeclSpecs specs{unit};
  (void)parse_decl_specifier_seq_no_typespecs(declSpecifierList, specs);

  auto lookat_notypespec_function_definition = [&] {
    LookaheadParser lookahead{this};
    if (!parse_notypespec_function_definition(yyast, declSpecifierList, specs))
      return false;
    lookahead.commit();
    return true;
  };

  if (lookat_notypespec_function_definition()) return true;

  auto lastDeclSpecifier = &declSpecifierList;
  while (*lastDeclSpecifier) {
    lastDeclSpecifier = &(*lastDeclSpecifier)->next;
  }

  (void)parse_decl_specifier_seq(*lastDeclSpecifier, specs, {});

  if (!specs.hasTypeSpecifier()) return false;

  if (SourceLocation semicolonLoc;
      match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    auto ast = make_node<SimpleDeclarationAST>(pool_);
    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->semicolonLoc = semicolonLoc;
    yyast = ast;
    return true;
  }

  Decl decl{specs};

  DeclaratorAST* declarator = nullptr;
  InitDeclaratorAST* initDeclarator = nullptr;
  if (!parse_bitfield_declarator(initDeclarator)) {
    (void)parse_declarator(declarator, decl);
  }

  auto lookat_function_definition = [&] {
    LookaheadParser lookahead{this};

    auto functionDeclarator = getFunctionPrototype(declarator);
    if (!functionDeclarator) return false;

    RequiresClauseAST* requiresClause = nullptr;
    if (!parse_requires_clause(requiresClause)) {
      parse_virt_specifier_seq(functionDeclarator);
    }

    parse_optional_attribute_specifier_seq(functionDeclarator->attributeList);

    if (!lookat_function_body()) return false;

    lookahead.commit();

    auto functionSymbol = binder_.declareFunction(declarator, decl);

    auto _ = Binder::ScopeGuard{&binder_};

    if (auto params = functionDeclarator->parameterDeclarationClause) {
      auto functionScope = functionSymbol->scope();
      functionScope->addSymbol(params->functionParametersSymbol);
      setScope(params->functionParametersSymbol);
    } else {
      setScope(functionSymbol);
    }

    FunctionBodyAST* functionBody = nullptr;
    if (!parse_function_body(functionBody)) {
      parse_error("expected function body");
    }

    auto ast = make_node<FunctionDefinitionAST>(pool_);
    yyast = ast;

    ast->declSpecifierList = declSpecifierList;
    ast->declarator = declarator;
    ast->requiresClause = requiresClause;
    ast->functionBody = functionBody;
    ast->symbol = functionSymbol;

    if (classDepth_) pendingFunctionDefinitions_.push_back(ast);

    return true;
  };

  if (lookat_function_definition()) return true;

  auto ast = make_node<SimpleDeclarationAST>(pool_);
  yyast = ast;

  ast->attributeList = attributes;
  ast->declSpecifierList = declSpecifierList;

  if (!initDeclarator) {
    if (!parse_member_declarator(initDeclarator, declarator, decl)) {
      parse_error("expected a member declarator");
    }
  }

  auto it = &ast->initDeclaratorList;

  if (initDeclarator) {
    *it = make_list_node(pool_, initDeclarator);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    InitDeclaratorAST* initDeclarator = nullptr;

    if (!parse_member_declarator(initDeclarator, specs)) {
      parse_error("expected a declarator");
    }

    if (initDeclarator) {
      *it = make_list_node(pool_, initDeclarator);
      it = &(*it)->next;
    }
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_bitfield_declarator(InitDeclaratorAST*& yyast) -> bool {
  LookaheadParser lookahead{this};

  SourceLocation identifierLoc;

  match(TokenKind::T_IDENTIFIER, identifierLoc);

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  SourceLocation colonLoc;

  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  ExpressionAST* sizeExpression = nullptr;

  std::optional<ConstValue> constValue;

  if (!parse_constant_expression(sizeExpression, constValue)) {
    parse_error("expected an expression");
  }

  lookahead.commit();

  auto nameId = make_node<NameIdAST>(pool_);
  nameId->identifierLoc = identifierLoc;
  nameId->identifier = unit->identifier(identifierLoc);

  auto bitfieldDeclarator = make_node<BitfieldDeclaratorAST>(pool_);
  bitfieldDeclarator->unqualifiedId = nameId;
  bitfieldDeclarator->colonLoc = colonLoc;
  bitfieldDeclarator->sizeExpression = sizeExpression;

  auto declarator = make_node<DeclaratorAST>(pool_);
  declarator->coreDeclarator = bitfieldDeclarator;

  ExpressionAST* initializer = nullptr;

  (void)parse_brace_or_equal_initializer(initializer);

  auto ast = make_node<InitDeclaratorAST>(pool_);
  yyast = ast;

  ast->declarator = declarator;
  ast->initializer = initializer;

  return true;
}

auto Parser::parse_member_declarator(InitDeclaratorAST*& yyast,
                                     const DeclSpecs& specs) -> bool {
  if (parse_bitfield_declarator(yyast)) {
    return true;
  }

  LookaheadParser lookahead{this};

  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  if (!parse_declarator(declarator, decl)) return false;

  lookahead.commit();

  return parse_member_declarator(yyast, declarator, decl);
}

auto Parser::parse_member_declarator(InitDeclaratorAST*& yyast,
                                     DeclaratorAST* declarator,
                                     const Decl& decl) -> bool {
  if (!declarator) {
    return false;
  }

  auto symbol = binder_.declareMemberSymbol(declarator, decl);

  auto ast = make_node<InitDeclaratorAST>(pool_);
  yyast = ast;

  ast->declarator = declarator;
  ast->symbol = symbol;

  if (auto functionDeclarator = getFunctionPrototype(declarator)) {
    RequiresClauseAST* requiresClause = nullptr;

    if (parse_requires_clause(requiresClause)) {
      ast->requiresClause = requiresClause;
    } else {
      parse_virt_specifier_seq(functionDeclarator);

      if (!functionDeclarator->attributeList) {
        parse_optional_attribute_specifier_seq(
            functionDeclarator->attributeList);
      }

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

auto Parser::parse_virt_specifier(
    FunctionDeclaratorChunkAST* functionDeclarator) -> bool {
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
  LookaheadParser lookahead{this};

  if (!match(TokenKind::T_EQUAL, equalLoc)) return false;

  if (!match(TokenKind::T_INTEGER_LITERAL, zeroLoc)) return false;

  const auto& number = unit->tokenText(zeroLoc);

  if (number != "0") return false;

  lookahead.commit();

  return true;
}

auto Parser::parse_conversion_function_id(ConversionFunctionIdAST*& yyast)
    -> bool {
  LookaheadParser lookahead{this};

  SourceLocation operatorLoc;

  if (!match(TokenKind::T_OPERATOR, operatorLoc)) return false;

  List<SpecifierAST*>* typeSpecifierList = nullptr;
  DeclSpecs specs{unit};
  if (!parse_type_specifier_seq(typeSpecifierList, specs)) return false;

  lookahead.commit();

  auto declarator = make_node<DeclaratorAST>(pool_);

  (void)parse_ptr_operator_seq(declarator->ptrOpList);

  auto typeId = make_node<TypeIdAST>(pool_);
  typeId->typeSpecifierList = typeSpecifierList;
  typeId->declarator = declarator;
  typeId->type = getDeclaratorType(unit, declarator, specs.getType());

  auto ast = make_node<ConversionFunctionIdAST>(pool_);
  yyast = ast;

  ast->operatorLoc = operatorLoc;
  ast->typeId = typeId;

  return true;
}

auto Parser::parse_base_clause(ClassSpecifierAST* ast) -> bool {
  if (!match(TokenKind::T_COLON, ast->colonLoc)) return false;

  if (!parse_base_specifier_list(ast)) {
    parse_error("expected a base class specifier");
  }

  return true;
}

auto Parser::parse_base_specifier_list(ClassSpecifierAST* ast) -> bool {
  auto it = &ast->baseSpecifierList;

  BaseSpecifierAST* baseSpecifier = nullptr;

  parse_base_specifier(baseSpecifier);

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  if (baseSpecifier && baseSpecifier->symbol) {
    ast->symbol->addBaseClass(baseSpecifier->symbol);
  }

  *it = make_list_node(pool_, baseSpecifier);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    BaseSpecifierAST* baseSpecifier = nullptr;

    parse_base_specifier(baseSpecifier);

    SourceLocation ellipsisLoc;

    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    if (baseSpecifier && baseSpecifier->symbol) {
      ast->symbol->addBaseClass(baseSpecifier->symbol);
    }

    *it = make_list_node(pool_, baseSpecifier);
    it = &(*it)->next;
  }

  return true;
}

void Parser::parse_base_specifier(BaseSpecifierAST*& yyast) {
  auto ast = make_node<BaseSpecifierAST>(pool_);
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

  if (!parse_class_or_decltype(ast->nestedNameSpecifier, ast->templateLoc,
                               ast->unqualifiedId)) {
    parse_error("expected a class name");
    return;
  }

  if (ast->templateLoc) {
    ast->isTemplateIntroduced = true;
  }

  binder_.bind(ast);
}

auto Parser::parse_class_or_decltype(
    NestedNameSpecifierAST*& yynestedNameSpecifier,
    SourceLocation& yytemplateLoc, UnqualifiedIdAST*& yyast) -> bool {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(
      nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

  if (!nestedNameSpecifier) {
    DecltypeSpecifierAST* decltypeSpecifier = nullptr;
    if (parse_decltype_specifier(decltypeSpecifier)) {
      DecltypeIdAST* decltypeName = make_node<DecltypeIdAST>(pool_);
      decltypeName->decltypeSpecifier = decltypeSpecifier;
      yynestedNameSpecifier = nullptr;
      yyast = decltypeName;
      return true;
    }
  }

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  UnqualifiedIdAST* unqualifiedName = nullptr;
  if (!parse_type_name(unqualifiedName, nestedNameSpecifier,
                       isTemplateIntroduced)) {
    parse_error("expected a class name");
  }

  yytemplateLoc = templateLoc;
  yynestedNameSpecifier = nestedNameSpecifier;
  yyast = unqualifiedName;
  yytemplateLoc = templateLoc;

  return true;
}

auto Parser::parse_access_specifier(SourceLocation& loc) -> bool {
  if (LA().isOneOf(TokenKind::T_PRIVATE, TokenKind::T_PROTECTED,
                   TokenKind::T_PUBLIC)) {
    loc = consumeToken();
    return true;
  }

  return false;
}

auto Parser::parse_ctor_initializer(
    SourceLocation& colonLoc, List<MemInitializerAST*>*& memInitializerList)
    -> bool {
  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  parse_mem_initializer_list(memInitializerList);

  return true;
}

void Parser::parse_mem_initializer_list(List<MemInitializerAST*>*& yyast) {
  auto it = &yyast;

  MemInitializerAST* mem_initializer = nullptr;

  parse_mem_initializer(mem_initializer);

  *it = make_list_node(pool_, mem_initializer);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    MemInitializerAST* mem_initializer = nullptr;

    parse_mem_initializer(mem_initializer);
    *it = make_list_node(pool_, mem_initializer);
    it = &(*it)->next;
  }
}

void Parser::parse_mem_initializer(MemInitializerAST*& yyast) {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* name = nullptr;

  parse_mem_initializer_id(nestedNameSpecifier, name);

  if (lookat(TokenKind::T_LBRACE)) {
    auto ast = make_node<BracedMemInitializerAST>(pool_);
    yyast = ast;

    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->unqualifiedId = name;

    if (classDepth_) {
      ast->bracedInitList = make_node<BracedInitListAST>(pool_);
      ast->bracedInitList->lbraceLoc = currentLocation();
      if (parse_skip_balanced()) {
        ast->bracedInitList->rbraceLoc = currentLocation().previous();
      }
    } else {
      if (!parse_braced_init_list(ast->bracedInitList, ExprContext{})) {
        parse_error("expected an initializer");
      }
    }

    match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

    return;
  }

  auto ast = make_node<ParenMemInitializerAST>(pool_);
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = name;

  if (classDepth_) {
    // postpone parsing of the intiializer
    if (lookat(TokenKind::T_LPAREN)) {
      ast->lparenLoc = currentLocation();

      if (parse_skip_balanced()) {
        ast->rparenLoc = currentLocation().previous();
      }
    } else {
      expect(TokenKind::T_LPAREN, ast->lparenLoc);
    }
  } else {
    expect(TokenKind::T_LPAREN, ast->lparenLoc);

    if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
      if (!parse_expression_list(ast->expressionList, ExprContext{})) {
        parse_error("expected an expression");
      }

      expect(TokenKind::T_RPAREN, ast->rparenLoc);
    }
  }

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);
}

void Parser::parse_mem_initializer_id(
    NestedNameSpecifierAST*& yynestedNameSpecifier, UnqualifiedIdAST*& yyast) {
  if (lookat(TokenKind::T_IDENTIFIER) &&
      LA(1).isOneOf(TokenKind::T_LPAREN, TokenKind::T_LBRACE)) {
    NameIdAST* nameId = nullptr;
    if (!parse_name_id(nameId)) {
      parse_error("expected a name");
    } else {
      yynestedNameSpecifier = nullptr;
      yyast = nameId;
    }
    return;
  }

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  if (!parse_class_or_decltype(nestedNameSpecifier, templateLoc, yyast)) {
    parse_error("expected a name");
  }
}

auto Parser::parse_operator_function_id(OperatorFunctionIdAST*& yyast) -> bool {
  SourceLocation operatorLoc;

  if (!match(TokenKind::T_OPERATOR, operatorLoc)) return false;

  TokenKind op = TokenKind::T_EOF_SYMBOL;
  SourceLocation opLoc;
  SourceLocation openLoc;
  SourceLocation closeLoc;

  if (!parse_operator(op, opLoc, openLoc, closeLoc)) return false;

  auto ast = make_node<OperatorFunctionIdAST>(pool_);
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
      if (parse_greater_greater()) {
        op = TokenKind::T_GREATER_GREATER;
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

auto Parser::parse_literal_operator_id(LiteralOperatorIdAST*& yyast) -> bool {
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

  auto ast = make_node<LiteralOperatorIdAST>(pool_);
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

auto Parser::parse_template_declaration(TemplateDeclarationAST*& yyast)
    -> bool {
  std::vector<TemplateDeclarationAST*> templateDeclarations;
  return parse_template_declaration(yyast, templateDeclarations);
}

auto Parser::parse_template_declaration(
    TemplateDeclarationAST*& yyast,
    std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  if (!lookat(TokenKind::T_TEMPLATE, TokenKind::T_LESS)) return false;

  auto _ = Binder::ScopeGuard{&binder_};
  TemplateHeadContext templateHeadContext{this};

  auto ast = make_node<TemplateDeclarationAST>(pool_);
  yyast = ast;

  auto templateParametersSymbol =
      control_->newTemplateParametersSymbol(scope(), {});
  ast->symbol = templateParametersSymbol;

  setScope(ast->symbol);

  templateDeclarations.push_back(ast);

  expect(TokenKind::T_TEMPLATE, ast->templateLoc);
  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    parse_template_parameter_list(ast->templateParameterList);
    expect(TokenKind::T_GREATER, ast->greaterLoc);
  }

  (void)parse_requires_clause(ast->requiresClause);

  if (lookat(TokenKind::T_TEMPLATE, TokenKind::T_LESS)) {
    TemplateDeclarationAST* templateDeclaration = nullptr;
    (void)parse_template_declaration(templateDeclaration, templateDeclarations);
    ast->declaration = templateDeclaration;
    return true;
  }

  if (parse_concept_definition(ast->declaration)) {
    return true;
  }

  if (!parse_template_declaration_body(ast->declaration, templateDeclarations))
    parse_error("expected a declaration");

  return true;
}

void Parser::parse_template_parameter_list(
    List<TemplateParameterAST*>*& yyast) {
  auto it = &yyast;

  int templateParameterCount = 0;
  std::swap(templateParameterCount_, templateParameterCount);

  TemplateParameterAST* parameter = nullptr;
  parse_template_parameter(parameter);

  if (parameter) {
    parameter->depth = templateParameterDepth_;
    parameter->index = templateParameterCount_++;

    *it = make_list_node(pool_, parameter);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    TemplateParameterAST* parameter = nullptr;
    parse_template_parameter(parameter);

    if (!parameter) continue;

    parameter->depth = templateParameterDepth_;
    parameter->index = templateParameterCount_++;

    *it = make_list_node(pool_, parameter);
    it = &(*it)->next;
  }

  std::swap(templateParameterCount_, templateParameterCount);
}

auto Parser::parse_requires_clause(RequiresClauseAST*& yyast) -> bool {
  SourceLocation requiresLoc;

  if (!match(TokenKind::T_REQUIRES, requiresLoc)) return false;

  yyast = make_node<RequiresClauseAST>(pool_);

  yyast->requiresLoc = requiresLoc;

  ExprContext ctx;
  ctx.inRequiresClause = true;

  if (!parse_constraint_logical_or_expression(yyast->expression, ctx)) {
    parse_error("expected a requirement expression");
  }

  return true;
}

auto Parser::parse_constraint_logical_or_expression(ExpressionAST*& yyast,
                                                    const ExprContext& ctx)
    -> bool {
  if (!parse_constraint_logical_and_expression(yyast, ctx)) return false;

  SourceLocation opLoc;

  while (match(TokenKind::T_BAR_BAR, opLoc)) {
    ExpressionAST* expression = nullptr;

    if (!parse_constraint_logical_and_expression(expression, ctx)) {
      parse_error("expected a requirement expression");
    }

    auto ast = make_node<BinaryExpressionAST>(pool_);
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->op = TokenKind::T_BAR_BAR;
    ast->rightExpression = expression;
    yyast = ast;

    check(ast);
  }

  return true;
}

auto Parser::parse_constraint_logical_and_expression(ExpressionAST*& yyast,
                                                     const ExprContext& ctx)
    -> bool {
  if (!parse_primary_expression(yyast, ctx)) return false;

  SourceLocation opLoc;

  while (match(TokenKind::T_AMP_AMP, opLoc)) {
    ExpressionAST* expression = nullptr;

    if (!parse_primary_expression(expression, ctx)) {
      parse_error("expected an expression");
    }

    auto ast = make_node<BinaryExpressionAST>(pool_);
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->op = TokenKind::T_AMP_AMP;
    ast->rightExpression = expression;
    yyast = ast;

    check(ast);
  }

  return true;
}

void Parser::parse_template_parameter(TemplateParameterAST*& yyast) {
  auto lookat_constraint_type_parameter = [&] {
    LookaheadParser lookahead{this};

    if (!parse_constraint_type_parameter(yyast)) return false;

    if (!LA().isOneOf(TokenKind::T_GREATER, TokenKind::T_COMMA)) return false;

    lookahead.commit();

    return true;
  };

  auto lookat_type_parameter = [&] {
    LookaheadParser lookahead{this};

    if (!parse_type_parameter(yyast)) return false;

    lookahead.commit();

    return true;
  };

  if (lookat_constraint_type_parameter()) return;
  if (lookat_type_parameter()) return;

  LookaheadParser lookahead{this};

  ParameterDeclarationAST* parameter = nullptr;

  if (!parse_parameter_declaration(parameter, /*templParam*/ true)) return;

  lookahead.commit();

  auto ast = make_node<NonTypeTemplateParameterAST>(pool_);
  yyast = ast;

  ast->declaration = parameter;

  binder_.bind(ast, templateParameterCount_, templateParameterDepth_);
}

auto Parser::parse_type_parameter(TemplateParameterAST*& yyast) -> bool {
  if (lookat(TokenKind::T_TEMPLATE, TokenKind::T_LESS)) {
    parse_template_type_parameter(yyast);
    return true;
  }

  if (parse_typename_type_parameter(yyast)) return true;

  return false;
}

auto Parser::parse_typename_type_parameter(TemplateParameterAST*& yyast)
    -> bool {
  auto maybe_elaborated_type_spec = [this]() {
    if (!lookat(TokenKind::T_TYPENAME, TokenKind::T_IDENTIFIER)) return false;

    if (!LA(2).isOneOf(TokenKind::T_COLON_COLON, TokenKind::T_LESS))
      return false;

    return true;
  };

  if (maybe_elaborated_type_spec()) return false;

  SourceLocation classKeyLoc;

  if (!parse_type_parameter_key(classKeyLoc)) return false;

  auto ast = make_node<TypenameTypeParameterAST>(pool_);
  yyast = ast;

  ast->classKeyLoc = classKeyLoc;

  const auto isPack = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  ast->isPack = isPack;

  match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  binder_.bind(ast, templateParameterCount_, templateParameterDepth_);

  if (match(TokenKind::T_EQUAL, ast->equalLoc)) {
    if (!parse_type_id(ast->typeId)) parse_error("expected a type id");
  }

  return true;
}

void Parser::parse_template_type_parameter(TemplateParameterAST*& yyast) {
  auto _ = Binder::ScopeGuard{&binder_};

  auto ast = make_node<TemplateTypeParameterAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_TEMPLATE, ast->templateLoc);
  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    TemplateHeadContext templateHeadContext{this};

    auto parameters =
        control_->newTemplateParametersSymbol(scope(), ast->templateLoc);

    setScope(parameters);

    parse_template_parameter_list(ast->templateParameterList);

    expect(TokenKind::T_GREATER, ast->greaterLoc);

    setScope(parameters->enclosingScope());
  }

  (void)parse_requires_clause(ast->requiresClause);

  if (!parse_type_parameter_key(ast->classKeyLoc)) {
    parse_error("expected a type parameter");
  }

  ast->isPack = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  if (match(TokenKind::T_IDENTIFIER, ast->identifierLoc)) {
    ast->identifier = unit->identifier(ast->identifierLoc);

    mark_maybe_template_name(ast->identifier);
  }

  binder_.bind(ast, templateParameterCount_, templateParameterDepth_);

  if (match(TokenKind::T_EQUAL, ast->equalLoc)) {
    if (!parse_id_expression(ast->idExpression,
                             IdExpressionContext::kTemplateParameter)) {
      parse_error("expected an id-expression");
    }
  }
}

auto Parser::parse_constraint_type_parameter(TemplateParameterAST*& yyast)
    -> bool {
  TypeConstraintAST* typeConstraint = nullptr;

  if (!parse_type_constraint(typeConstraint, /*parsing placeholder=*/false)) {
    return false;
  }

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  SourceLocation identifierLoc;
  match(TokenKind::T_IDENTIFIER, identifierLoc);

  SourceLocation equalLoc;
  TypeIdAST* typeId = nullptr;

  if (match(TokenKind::T_EQUAL, equalLoc)) {
    if (!parse_type_id(typeId)) {
      return false;  // ### FIXME: parse_error("expected a type id");
    }
  }

  auto ast = make_node<ConstraintTypeParameterAST>(pool_);
  yyast = ast;

  ast->typeConstraint = typeConstraint;
  ast->ellipsisLoc = ellipsisLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(identifierLoc);
  ast->equalLoc = equalLoc;
  ast->typeId = typeId;

  binder_.bind(ast, templateParameterCount_, templateParameterDepth_);

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
  SourceLocation identifierLoc;
  const Identifier* identifier = nullptr;

  auto lookat_type_constraint = [&] {
    LookaheadParser lookahead{this};

    parse_optional_nested_name_specifier(
        nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative);

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    identifier = unit->identifier(identifierLoc);

    if (!concept_names_.contains(identifier)) return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_type_constraint()) return false;

  auto ast = make_node<TypeConstraintAST>(pool_);
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->identifierLoc = identifierLoc;
  ast->identifier = identifier;

  if (match(TokenKind::T_LESS, ast->lessLoc)) {
    if (!parse_template_argument_list(ast->templateArgumentList)) {
      parse_error("expected a template argument");
    }

    expect(TokenKind::T_GREATER, ast->greaterLoc);
  }

  return true;
}

auto Parser::parse_simple_template_id(SimpleTemplateIdAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_IDENTIFIER, TokenKind::T_LESS)) return false;

  auto ast = make_node<SimpleTemplateIdAST>(pool_);
  yyast = ast;

  ast->identifierLoc = consumeToken();
  ast->lessLoc = consumeToken();
  ast->identifier = unit->identifier(ast->identifierLoc);

  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    if (!parse_template_argument_list(ast->templateArgumentList)) {
      parse_error("expected a template argument");
    }

    expect(TokenKind::T_GREATER, ast->greaterLoc);
  }

  return true;
}

auto Parser::parse_simple_template_id(
    SimpleTemplateIdAST*& yyast, NestedNameSpecifierAST* nestedNameSpecifier,
    bool isTemplateIntroduced) -> bool {
  LookaheadParser lookahead{this};

  SimpleTemplateIdAST* templateId = nullptr;
  if (!parse_simple_template_id(templateId)) return false;
  if (!templateId->greaterLoc) return false;

  auto candidate = Lookup{scope()}(nestedNameSpecifier, templateId->identifier);

  if (symbol_cast<NonTypeParameterSymbol>(candidate)) return false;

  Symbol* primaryTemplateSymbol = nullptr;

  if (is_template(candidate))
    primaryTemplateSymbol = candidate;
  else if (auto overloads = symbol_cast<OverloadSetSymbol>(candidate)) {
    for (auto overload : overloads->functions()) {
      if (is_template(overload)) {
        primaryTemplateSymbol = overload;
        break;
      }
    }
  }

  if (!primaryTemplateSymbol && !isTemplateIntroduced) {
    if (!maybe_template_name(templateId->identifier)) return false;
  }

  templateId->primaryTemplateSymbol = primaryTemplateSymbol;

  yyast = templateId;

  lookahead.commit();

  return true;
}

auto Parser::parse_literal_operator_template_id(
    LiteralOperatorTemplateIdAST*& yyast,
    NestedNameSpecifierAST* nestedNameSpecifier) -> bool {
  if (!lookat(TokenKind::T_OPERATOR)) return false;

  LookaheadParser lookahead{this};

  LiteralOperatorIdAST* literalOperatorName = nullptr;
  if (!parse_literal_operator_id(literalOperatorName)) return false;

  if (!lookat(TokenKind::T_LESS)) return false;

  lookahead.commit();

  auto ast = make_node<LiteralOperatorTemplateIdAST>(pool_);
  yyast = ast;

  ast->literalOperatorId = literalOperatorName;
  expect(TokenKind::T_LESS, ast->lessLoc);
  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    if (!parse_template_argument_list(ast->templateArgumentList))
      parse_error("expected a template argument");
    expect(TokenKind::T_GREATER, ast->greaterLoc);
  }

  return true;
}

auto Parser::parse_function_operator_template_id(
    OperatorFunctionTemplateIdAST*& yyast,
    NestedNameSpecifierAST* nestedNameSpecifier) -> bool {
  if (!lookat(TokenKind::T_OPERATOR)) return false;

  LookaheadParser lookahead{this};

  OperatorFunctionIdAST* operatorFunctionName = nullptr;
  if (!parse_operator_function_id(operatorFunctionName)) return false;

  if (!lookat(TokenKind::T_LESS)) return false;

  lookahead.commit();

  auto ast = make_node<OperatorFunctionTemplateIdAST>(pool_);
  yyast = ast;

  ast->operatorFunctionId = operatorFunctionName;
  expect(TokenKind::T_LESS, ast->lessLoc);
  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    if (!parse_template_argument_list(ast->templateArgumentList))
      parse_error("expected a template argument");
    expect(TokenKind::T_GREATER, ast->greaterLoc);
  }

  return true;
}

auto Parser::parse_template_id(UnqualifiedIdAST*& yyast,
                               NestedNameSpecifierAST* nestedNameSpecifier,
                               bool isTemplateIntroduced) -> bool {
  if (LiteralOperatorTemplateIdAST* templateName = nullptr;
      parse_literal_operator_template_id(templateName, nestedNameSpecifier)) {
    yyast = templateName;
    return true;
  }

  if (OperatorFunctionTemplateIdAST* templateName = nullptr;
      parse_function_operator_template_id(templateName, nestedNameSpecifier)) {
    yyast = templateName;
    return true;
  }

  SimpleTemplateIdAST* templateName = nullptr;
  if (!parse_simple_template_id(templateName, nestedNameSpecifier,
                                isTemplateIntroduced))
    return false;

  yyast = templateName;
  return true;
}

auto Parser::parse_template_argument_list(List<TemplateArgumentAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  TemplateArgumentAST* templateArgument = nullptr;

  if (!parse_template_argument(templateArgument)) return false;

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  *it = make_list_node(pool_, templateArgument);
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

    *it = make_list_node(pool_, templateArgument);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_template_argument(TemplateArgumentAST*& yyast) -> bool {
  const auto start = currentLocation();

  if (auto entry = template_arguments_.get(start)) {
    auto [loc, ast, parsed, hit] = *entry;
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

    auto ast = make_node<TypeTemplateArgumentAST>(pool_);
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

    auto ast = make_node<ExpressionTemplateArgumentAST>(pool_);
    yyast = ast;

    ast->expression = expression;

    return true;
  };

  if (lookat_type_id() || lookat_template_argument_constant_expression()) {
    template_arguments_.set(start, currentLocation(), yyast, true);

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

    DeclSpecs specs{unit};
    (void)parse_explicit_specifier(explicitSpecifier, specs);

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

  SimpleTemplateIdAST* templateId = nullptr;

  if (!parse_simple_template_id(templateId, /*nestedNameSpecifier=*/nullptr)) {
    parse_error("expected a template id");
  }

  SourceLocation semicolonLoc;

  expect(TokenKind::T_SEMICOLON, semicolonLoc);

  auto ast = make_node<DeductionGuideAST>(pool_);
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

  auto ast = make_node<ConceptDefinitionAST>(pool_);
  yyast = ast;

  ast->conceptLoc = conceptLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  ast->identifier = unit->identifier(ast->identifierLoc);

  binder_.bind(ast);

  if (ast->identifierLoc) {
    concept_names_.insert(ast->identifier);
  }

  expect(TokenKind::T_EQUAL, ast->equalLoc);

  if (!parse_constraint_expression(ast->expression)) {
    parse_error("expected a constraint expression");
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_splicer_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (!config_.reflect) return false;
  if (specs.typeSpecifier) return false;
  LookaheadParser lookahead{this};
  SourceLocation typenameLoc;
  match(TokenKind::T_TYPENAME, typenameLoc);
  SplicerAST* splicer = nullptr;
  if (!parse_splicer(splicer)) return false;
  lookahead.commit();
  auto ast = make_node<SplicerTypeSpecifierAST>(pool_);
  yyast = ast;
  ast->typenameLoc = typenameLoc;
  ast->splicer = splicer;
  specs.setTypeSpecifier(ast);
  return true;
}

auto Parser::parse_typename_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (specs.typeSpecifier) return false;

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  UnqualifiedIdAST* unqualifiedId = nullptr;
  bool isTemplateIntroduced = false;

  auto lookat_typename_specifier = [&] {
    LookaheadParser lookahead{this};
    if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

    if (!parse_nested_name_specifier(
            nestedNameSpecifier, NestedNameSpecifierContext::kNonDeclarative))
      return false;

    isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

    if (!lookat(TokenKind::T_IDENTIFIER)) return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_typename_specifier()) return false;

  SimpleTemplateIdAST* templateId = nullptr;
  if (parse_simple_template_id(templateId, nestedNameSpecifier,
                               isTemplateIntroduced)) {
    unqualifiedId = templateId;
  } else {
    NameIdAST* nameId = nullptr;
    (void)parse_name_id(nameId);
    unqualifiedId = nameId;
  }

  auto ast = make_node<TypenameSpecifierAST>(pool_);
  yyast = ast;

  ast->typenameLoc = typenameLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = unqualifiedId;

  specs.type =
      control_->getUnresolvedNameType(unit, nestedNameSpecifier, unqualifiedId);

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

  auto ast = make_node<ExplicitInstantiationAST>(pool_);
  yyast = ast;

  match(TokenKind::T_EXTERN, ast->externLoc);
  expect(TokenKind::T_TEMPLATE, ast->templateLoc);

  if (!parse_declaration(ast->declaration, BindingContext::kTemplate))
    parse_error("expected a declaration");

  return true;
}

auto Parser::parse_try_block(StatementAST*& yyast) -> bool {
  SourceLocation tryLoc;

  if (!match(TokenKind::T_TRY, tryLoc)) return false;

  auto ast = make_node<TryBlockStatementAST>(pool_);
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

  auto ast = make_node<TryStatementFunctionBodyAST>(pool_);
  yyast = ast;

  ast->tryLoc = tryLoc;

  if (!lookat(TokenKind::T_LBRACE)) {
    if (!parse_ctor_initializer(ast->colonLoc, ast->memInitializerList)) {
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

  yyast = make_node<HandlerAST>(pool_);

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
    *it = make_list_node(pool_, handler);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_exception_declaration(ExceptionDeclarationAST*& yyast)
    -> bool {
  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = make_node<EllipsisExceptionDeclarationAST>(pool_);
    yyast = ast;

    ast->ellipsisLoc = ellipsisLoc;
    return true;
  }

  auto ast = make_node<TypeExceptionDeclarationAST>(pool_);
  yyast = ast;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  DeclSpecs specs{unit};
  if (!parse_type_specifier_seq(ast->typeSpecifierList, specs)) {
    parse_error("expected a type specifier");
  }

  if (lookat(TokenKind::T_RPAREN)) return true;

  Decl decl{specs};
  parse_optional_declarator_or_abstract_declarator(ast->declarator, decl);

  return true;
}

auto Parser::parse_noexcept_specifier(ExceptionSpecifierAST*& yyast) -> bool {
  SourceLocation throwLoc;

  if (match(TokenKind::T_THROW, throwLoc)) {
    auto ast = make_node<ThrowExceptionSpecifierAST>(pool_);
    yyast = ast;

    ast->throwLoc = throwLoc;
    expect(TokenKind::T_LPAREN, ast->lparenLoc);
    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  SourceLocation noexceptLoc;

  if (!match(TokenKind::T_NOEXCEPT, noexceptLoc)) return false;

  auto ast = make_node<NoexceptSpecifierAST>(pool_);
  yyast = ast;

  ast->noexceptLoc = noexceptLoc;

  if (match(TokenKind::T_LPAREN, ast->lparenLoc)) {
    std::optional<ConstValue> constValue;

    if (!parse_constant_expression(ast->expression, constValue)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  return true;
}

auto Parser::parse_identifier_list(List<NameIdAST*>*& yyast) -> bool {
  auto it = &yyast;

  if (NameIdAST* id = nullptr; parse_name_id(id)) {
    *it = make_list_node(pool_, id);
    it = &(*it)->next;
  } else {
    return false;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (NameIdAST* id = nullptr; parse_name_id(id)) {
      *it = make_list_node(pool_, id);
      it = &(*it)->next;
    } else {
      parse_error("expected an identifier");
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

auto Parser::scope() const -> Scope* { return binder_.scope(); }

void Parser::setScope(Scope* scope) { binder_.setScope(scope); }

void Parser::setScope(ScopedSymbol* symbol) { setScope(symbol->scope()); }

void Parser::completeFunctionDefinition(FunctionDefinitionAST* ast) {
  if (!ast->functionBody) return;

  auto functionBody =
      ast_cast<CompoundStatementFunctionBodyAST>(ast->functionBody);

  if (!functionBody) return;

  auto _ = Binder::ScopeGuard{&binder_};

  setScope(ast->symbol);

  const auto saved = currentLocation();

  for (auto memInitializer : ListView{functionBody->memInitializerList}) {
    if (auto parenMemInitializer =
            ast_cast<ParenMemInitializerAST>(memInitializer)) {
      if (!parenMemInitializer->lparenLoc) {
        // found an invalid mem-initializer, the parser
        // already reported an error in this case, so
        // we just skip it
        continue;
      }

      // go after the lparen
      rewind(parenMemInitializer->lparenLoc.next());

      if (SourceLocation rparenLoc; !match(TokenKind::T_RPAREN, rparenLoc)) {
        if (!parse_expression_list(parenMemInitializer->expressionList,
                                   ExprContext{})) {
          parse_error("expected an expression");
        }

        expect(TokenKind::T_RPAREN, rparenLoc);
      }
    }

    if (auto bracedMemInitializer =
            ast_cast<BracedMemInitializerAST>(memInitializer)) {
      rewind(bracedMemInitializer->bracedInitList->lbraceLoc);
      if (!parse_braced_init_list(bracedMemInitializer->bracedInitList,
                                  ExprContext{})) {
        parse_error("expected a braced-init-list");
      }
    }
  }

  rewind(functionBody->statement->lbraceLoc.next());

  finish_compound_statement(functionBody->statement);

  rewind(saved);
}

void Parser::check(ExpressionAST* ast) {
  if (binder_.inTemplate()) return;
  TypeChecker check{unit};
  check.setScope(scope());
  check.setReportErrors(config_.checkTypes);
  check(ast);
}

auto Parser::convertName(UnqualifiedIdAST* id) -> const Name* {
  if (!id) return nullptr;
  return get_name(control_, id);
}

auto Parser::getFunction(Scope* scope, const Name* name, const Type* type)
    -> FunctionSymbol* {
  auto parentScope = scope;

  while (parentScope && parentScope->isTransparent()) {
    parentScope = parentScope->parent();
  }

  if (auto parentClass = symbol_cast<ClassSymbol>(parentScope->owner());
      parentClass && parentClass->name() == name) {
    for (auto ctor : parentClass->constructors()) {
      if (control_->is_same(ctor->type(), type)) {
        return ctor;
      }
    }
  }

  for (auto candidate : scope->find(name)) {
    if (auto function = symbol_cast<FunctionSymbol>(candidate)) {
      if (control_->is_same(function->type(), type)) {
        return function;
      }
    } else if (auto overloads = symbol_cast<OverloadSetSymbol>(candidate)) {
      for (auto function : overloads->functions()) {
        if (control_->is_same(function->type(), type)) {
          return function;
        }
      }
    }
  }

  return nullptr;
}

}  // namespace cxx
