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

#include <cxx/parser.h>

// cxx
#include <cxx/ast.h>
#include <cxx/const_expression_evaluator.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/name_printer.h>
#include <cxx/names.h>
#include <cxx/private/format.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>

#include <algorithm>
#include <cstring>
#include <forward_list>

namespace cxx {

namespace {

inline constexpr struct {
  auto operator()(const StringLiteral*) const -> bool { return true; }
  auto operator()(auto value) const -> bool { return !!value; }
} to_bool;

struct ConvertToName {
  Control* control_;

  explicit ConvertToName(Control* control) : control_(control) {}

  auto operator()(NameIdAST* ast) const -> const Name* {
    return ast->identifier;
  }

  auto operator()(DestructorIdAST* ast) const -> const Name* {
    return control_->getDestructorId(visit(*this, ast->id));
  }

  auto operator()(DecltypeIdAST* ast) const -> const Name* {
    cxx_runtime_error("DecltypeIdAST not implemented");
    return {};
  }

  auto operator()(OperatorFunctionIdAST* ast) const -> const Name* {
    return control_->getOperatorId(ast->op);
  }

  auto operator()(LiteralOperatorIdAST* ast) const -> const Name* {
    if (ast->identifier)
      return control_->getLiteralOperatorId(ast->identifier->name());

    auto value = ast->literal->value();
    auto suffix = value.substr(value.find_last_of('"') + 1);
    return control_->getLiteralOperatorId(suffix);
  }

  auto operator()(ConversionFunctionIdAST* ast) const -> const Name* {
    return control_->getConversionFunctionId(ast->typeId->type);
  }

  auto operator()(SimpleTemplateIdAST* ast) const -> const Name* {
    std::vector<TemplateArgument> arguments;
    return control_->getTemplateId(ast->identifier, std::move(arguments));
  }

  auto operator()(LiteralOperatorTemplateIdAST* ast) const -> const Name* {
    std::vector<TemplateArgument> arguments;
    return control_->getTemplateId(operator()(ast->literalOperatorId),
                                   std::move(arguments));
  }

  auto operator()(OperatorFunctionTemplateIdAST* ast) const -> const Name* {
    std::vector<TemplateArgument> arguments;
    return control_->getTemplateId(operator()(ast->operatorFunctionId),
                                   std::move(arguments));
  }
};

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

class FunctionPrototype {
  enum struct Kind { Direct, Ptr, Function, Array };

  FunctionDeclaratorChunkAST* prototype_ = nullptr;
  Kind kind_ = Kind::Direct;

 public:
  FunctionPrototype() = default;

  FunctionDeclaratorChunkAST* operator()(DeclaratorAST* declarator) {
    makeDirect();
    process(declarator);
    return prototype_;
  }

 private:
  void makeDirect() {
    prototype_ = nullptr;
    kind_ = Kind::Direct;
  }

  void makePtr() {
    prototype_ = nullptr;
    kind_ = Kind::Ptr;
  }

  void makeArray() {
    prototype_ = nullptr;
    kind_ = Kind::Array;
  }

  void makeFunction(FunctionDeclaratorChunkAST* prototype) {
    prototype_ = prototype;
    kind_ = Kind::Function;
  }

  void process(DeclaratorAST* declarator) {
    if (!declarator) return;
    if (declarator->ptrOpList) makePtr();
    if (declarator->declaratorChunkList) {
      auto prototype = ast_cast<FunctionDeclaratorChunkAST>(
          declarator->declaratorChunkList->value);
      if (prototype) makeFunction(prototype);
    }
    auto nested = ast_cast<NestedDeclaratorAST>(declarator->coreDeclarator);
    if (nested) process(nested->declarator);
  }
};

auto getFunctionPrototype(DeclaratorAST* declarator)
    -> FunctionDeclaratorChunkAST* {
  FunctionPrototype prototype;
  return prototype(declarator);
}

}  // namespace

struct Parser::GetDeclaratorType {
  Parser* p;
  const Type* type_ = nullptr;

  struct {
    auto operator()(float value) const -> std::optional<std::size_t> {
      return std::nullopt;
    }

    auto operator()(double value) const -> std::optional<std::size_t> {
      return std::nullopt;
    }

    auto operator()(const StringLiteral* value) const
        -> std::optional<std::size_t> {
      return std::nullopt;
    }

    template <typename T>
    auto operator()(T value) const -> std::optional<std::size_t> {
      return static_cast<std::size_t>(value);
    }
  } get_size_value;

  explicit GetDeclaratorType(Parser* p) : p(p) {}

  auto control() const -> Control* { return p->unit->control(); }

  auto operator()(DeclaratorAST* ast, const Type* type) -> const Type* {
    if (!ast) return type;

    std::swap(type_, type);

    std::invoke(*this, ast);

    std::swap(type_, type);

    return type;
  }

  void operator()(DeclaratorAST* ast) {
    for (auto it = ast->ptrOpList; it; it = it->next) visit(*this, it->value);

    auto nestedNameSpecifier = getNestedNameSpecifier(ast->coreDeclarator);

    std::invoke(*this, ast->declaratorChunkList);

    if (ast->coreDeclarator) visit(*this, ast->coreDeclarator);
  }

  auto getNestedNameSpecifier(CoreDeclaratorAST* ast) const
      -> NestedNameSpecifierAST* {
    struct {
      auto operator()(DeclaratorAST* ast) const -> NestedNameSpecifierAST* {
        if (ast->coreDeclarator) return visit(*this, ast->coreDeclarator);
        return nullptr;
      }

      auto operator()(BitfieldDeclaratorAST* ast) const
          -> NestedNameSpecifierAST* {
        return nullptr;
      }

      auto operator()(ParameterPackAST* ast) const -> NestedNameSpecifierAST* {
        return nullptr;
      }

      auto operator()(IdDeclaratorAST* ast) const -> NestedNameSpecifierAST* {
        return ast->nestedNameSpecifier;
      }

      auto operator()(NestedDeclaratorAST* ast) const
          -> NestedNameSpecifierAST* {
        if (!ast->declarator) return nullptr;
        return std::invoke(*this, ast->declarator);
      }
    } v;

    if (!ast) return nullptr;
    return visit(v, ast);
  }

  void operator()(List<DeclaratorChunkAST*>* chunks) {
    if (!chunks) return;
    std::invoke(*this, chunks->next);
    visit(*this, chunks->value);
  }

  void operator()(PointerOperatorAST* ast) {
    type_ = control()->getPointerType(type_);

    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (ast_cast<ConstQualifierAST>(it->value)) {
        type_ = control()->getConstType(type_);
      } else if (ast_cast<VolatileQualifierAST>(it->value)) {
        type_ = control()->getVolatileType(type_);
      }
    }
  }

  void operator()(ReferenceOperatorAST* ast) {
    if (ast->refOp == TokenKind::T_AMP_AMP) {
      type_ = control()->getRvalueReferenceType(type_);
    } else {
      type_ = control()->getLvalueReferenceType(type_);
    }
  }

  void operator()(PtrToMemberOperatorAST* ast) {}

  void operator()(BitfieldDeclaratorAST* ast) {}

  void operator()(ParameterPackAST* ast) {
    if (ast->coreDeclarator) visit(*this, ast->coreDeclarator);
  }

  void operator()(IdDeclaratorAST* ast) {}

  void operator()(NestedDeclaratorAST* ast) {
    std::invoke(*this, ast->declarator);
  }

  void operator()(FunctionDeclaratorChunkAST* ast) {
    auto returnType = type_;
    std::vector<const Type*> parameterTypes;
    bool isVariadic = false;
    CvQualifiers cvQualifiers = CvQualifiers::kNone;

    if (auto params = ast->parameterDeclarationClause) {
      for (auto it = params->parameterDeclarationList; it; it = it->next) {
        auto paramType = it->value->type;
        parameterTypes.push_back(paramType);
      }

      isVariadic = params->isVariadic;
    }

    RefQualifier refQualifier = RefQualifier::kNone;

    if (ast->refLoc) {
      if (p->unit->tokenKind(ast->refLoc) == TokenKind::T_AMP_AMP) {
        refQualifier = RefQualifier::kRvalue;
      } else {
        refQualifier = RefQualifier::kLvalue;
      }
    }

    bool isNoexcept = false;

    if (ast->exceptionSpecifier)
      isNoexcept = visit(*this, ast->exceptionSpecifier);

    if (ast->trailingReturnType) {
#if false
      if (!type_cast<AutoType>(returnType)) {
        p->parse_warn(ast->trailingReturnType->firstSourceLocation(),
                      cxx::format("function with trailing return type must "
                                  "be declared with 'auto', not '{}'",
                                  to_string(returnType)));
      }
#endif

      if (ast->trailingReturnType->typeId) {
        returnType = ast->trailingReturnType->typeId->type;
      }
    }

    type_ = control()->getFunctionType(returnType, std::move(parameterTypes),
                                       isVariadic, cvQualifiers, refQualifier,
                                       isNoexcept);
  }

  void operator()(ArrayDeclaratorChunkAST* ast) {
    if (!ast->expression) {
      type_ = control()->getUnboundedArrayType(type_);
      return;
    }

    auto constValue = p->evaluate_constant_expression(ast->expression);

    if (constValue) {
      if (auto size = std::visit(get_size_value, *constValue)) {
        type_ = control()->getBoundedArrayType(type_, *size);
        return;
      }
    }

    type_ = control()->getUnresolvedBoundedArrayType(p->unit, type_,
                                                     ast->expression);
  }

  auto operator()(ThrowExceptionSpecifierAST* ast) -> bool { return false; }

  auto operator()(NoexceptSpecifierAST* ast) -> bool {
    if (!ast->expression) return true;
    return false;
  }
};

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

struct Parser::ClassHead {
  explicit ClassHead(
      const std::vector<TemplateDeclarationAST*>& templateDeclarations)
      : templateDeclarations(templateDeclarations) {}

  const std::vector<TemplateDeclarationAST*>& templateDeclarations;
  SourceLocation classLoc;
  List<AttributeSpecifierAST*>* attributeList = nullptr;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* name = nullptr;
  SourceLocation finalLoc;
  SourceLocation colonLoc;
  List<BaseSpecifierAST*>* baseSpecifierList = nullptr;
};

struct Parser::DeclSpecs {
  Parser* parser = nullptr;

  explicit DeclSpecs(Parser* parser) : parser(parser) {}

  auto control() const -> Control* { return parser->control_; }

  auto getType() const -> const Type* {
    auto type = this->type;

    if (!type || type == control()->getIntType()) {
      if (isLongLong && isUnsigned)
        type = control()->getUnsignedLongLongIntType();
      else if (isLongLong)
        type = control()->getLongLongIntType();
      else if (isLong && isUnsigned)
        type = control()->getUnsignedLongIntType();
      else if (isLong)
        type = control()->getLongIntType();
      else if (isShort && isUnsigned)
        type = control()->getUnsignedShortIntType();
      else if (isShort)
        type = control()->getShortIntType();
      else if (isUnsigned)
        type = control()->getUnsignedIntType();
      else if (isSigned)
        type = control()->getIntType();
    }

    if (!type) return nullptr;

    if (type == control()->getDoubleType() && isLong)
      type = control()->getLongDoubleType();

    if (isSigned && type == control()->getCharType())
      type = control()->getSignedCharType();

    if (isUnsigned) {
      switch (type->kind()) {
        case TypeKind::kChar:
          type = control()->getUnsignedCharType();
          break;
        case TypeKind::kShortInt:
          type = control()->getUnsignedShortIntType();
          break;
        case TypeKind::kInt:
          type = control()->getUnsignedIntType();
          break;
        case TypeKind::kLongInt:
          type = control()->getUnsignedLongIntType();
          break;
        case TypeKind::kLongLongInt:
          type = control()->getUnsignedLongLongIntType();
          break;
        case TypeKind::kChar8:
          type = control()->getUnsignedCharType();
          break;
        case TypeKind::kChar16:
          type = control()->getUnsignedShortIntType();
          break;
        case TypeKind::kChar32:
          type = control()->getUnsignedIntType();
          break;
        case TypeKind::kWideChar:
          type = control()->getUnsignedIntType();
          break;
        default:
          break;
      }  // switch
    }

    if (isConst && isVolatile) {
      type = control()->getConstVolatileType(type);
    } else if (isConst) {
      type = control()->getConstType(type);
    } else if (isVolatile) {
      type = control()->getVolatileType(type);
    }

    return type;
  }

  [[nodiscard]] auto hasTypeSpecifier() const -> bool {
    if (typeSpecifier) return true;
    if (isShort || isLong) return true;
    if (isSigned || isUnsigned) return true;
    return false;
  }

  void setTypeSpecifier(SpecifierAST* specifier) { typeSpecifier = specifier; }

  [[nodiscard]] auto hasClassOrEnumSpecifier() const -> bool {
    if (!typeSpecifier) return false;
    switch (typeSpecifier->kind()) {
      case ASTKind::ClassSpecifier:
      case ASTKind::EnumSpecifier:
      case ASTKind::ElaboratedTypeSpecifier:
      case ASTKind::TypenameSpecifier:
        return true;
      default:
        return false;
    }  // switch
  }

  [[nodiscard]] auto hasPlaceholderTypeSpecifier() const -> bool {
    if (!typeSpecifier) return false;
    switch (typeSpecifier->kind()) {
      case ASTKind::AutoTypeSpecifier:
      case ASTKind::DecltypeAutoSpecifier:
      case ASTKind::PlaceholderTypeSpecifier:
      case ASTKind::DecltypeSpecifier:
        return true;
      default:
        return false;
    }  // switch
  }

  const Type* type = nullptr;
  SpecifierAST* typeSpecifier = nullptr;

  bool isTypedef = false;
  bool isFriend = false;
  bool isConstexpr = false;
  bool isConsteval = false;
  bool isConstinit = false;
  bool isInline = false;

  // cv qualifiers
  bool isConst = false;
  bool isVolatile = false;
  bool isRestrict = false;

  // storage class specifiers
  bool isStatic = false;
  bool isThreadLocal = false;
  bool isExtern = false;
  bool isMutable = false;
  bool isThread = false;

  // function specifiers
  bool isVirtual = false;
  bool isExplicit = false;

  // sign specifiers
  bool isSigned = false;
  bool isUnsigned = false;

  // sized specifiers
  bool isShort = false;
  bool isLong = false;
  bool isLongLong = false;

  bool isComplex = false;

  // placeholder type specifiers
  bool isAuto = false;
  bool isDecltypeAuto = false;

  bool no_typespecs = false;
  bool no_class_or_enum_specs = false;
};

struct Parser::Decl {
  DeclSpecs specs;
  IdDeclaratorAST* declaratorId = nullptr;
  bool isPack = false;

  explicit Decl(const DeclSpecs& specs) : specs{specs} {}

  auto getName() const -> const Name* {
    auto control = specs.control();
    if (!declaratorId) return nullptr;
    if (!declaratorId->unqualifiedId) return nullptr;
    return visit(ConvertToName{control}, declaratorId->unqualifiedId);
  }
};

struct Parser::ScopeGuard {
  Parser* p = nullptr;
  Scope* savedScope = nullptr;

  ScopeGuard(const ScopeGuard&) = delete;
  auto operator=(const ScopeGuard&) -> ScopeGuard& = delete;

  ScopeGuard() = default;

  explicit ScopeGuard(Parser* p, Scope* scope = nullptr)
      : p(p), savedScope(p->scope_) {
    if (scope) p->scope_ = scope;
  }

  ~ScopeGuard() { p->scope_ = savedScope; }
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

struct Parser::DeclareSymbol {
  Parser* p;
  Scope* scope_;

  explicit DeclareSymbol(Parser* p, Scope* scope) : p(p), scope_(scope) {}

  auto control() const -> Control* { return p->control_; }

  void operator()(Symbol* symbol) {
    if (auto f = symbol_cast<FunctionSymbol>(symbol)) {
      for (Symbol* candidate : scope_->get(symbol->name())) {
        if (auto currentFunction = symbol_cast<FunctionSymbol>(candidate)) {
          scope_->removeSymbol(currentFunction);

          auto ovl =
              control()->newOverloadSetSymbol(candidate->enclosingScope());
          ovl->setName(symbol->name());
          scope_->addSymbol(ovl);

          ovl->addFunction(currentFunction);
          ovl->addFunction(f);
          return;
        }

        if (auto ovl = symbol_cast<OverloadSetSymbol>(candidate)) {
          ovl->addFunction(f);
          return;
        }
      }
    }

    scope_->addSymbol(symbol);
  }
};

Parser::Parser(TranslationUnit* unit) : unit(unit) {
  control_ = unit->control();
  diagnosticClient_ = unit->diagnosticsClient();
  cursor_ = 1;

  pool_ = unit->arena();

  moduleId_ = control_->getIdentifier("module");
  importId_ = control_->getIdentifier("import");
  finalId_ = control_->getIdentifier("final");
  overrideId_ = control_->getIdentifier("override");

  globalScope_ = unit->globalScope();
  scope_ = globalScope_;

  mark_maybe_template_name(control_->getIdentifier("__make_integer_seq"));
  mark_maybe_template_name(control_->getIdentifier("__type_pack_element"));
}

Parser::~Parser() = default;

auto Parser::prec(TokenKind tk) -> Parser::Prec {
  switch (tk) {
    default:
      cxx_runtime_error(cxx::format("expected a binary operator, found {}",
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
  parse_error(cxx::format("expected '{}'", Token::spell(tk)));
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

auto Parser::parse_type_name(UnqualifiedIdAST*& yyast) -> bool {
  auto lookat_simple_template_id = [&] {
    LookaheadParser lookahead{this};
    SimpleTemplateIdAST* templateId = nullptr;
    if (!parse_simple_template_id(templateId)) return false;
    yyast = templateId;
    lookahead.commit();
    return true;
  };

  if (lookat_simple_template_id()) return true;

  if (NameIdAST* nameId = nullptr; parse_name_id(nameId)) {
    yyast = nameId;
    return true;
  }

  return false;
}

auto Parser::parse_name_id(NameIdAST*& yyast) -> bool {
  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = new (pool_) NameIdAST();
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(ast->identifierLoc);

  return true;
}

auto Parser::parse_literal(ExpressionAST*& yyast) -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_CHARACTER_LITERAL: {
      auto ast = new (pool_) CharLiteralExpressionAST();
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
      auto ast = new (pool_) BoolLiteralExpressionAST();
      yyast = ast;

      const auto isTrue = lookat(TokenKind::T_TRUE);

      ast->literalLoc = consumeToken();
      ast->isTrue = isTrue;
      ast->type = control_->getBoolType();

      return true;
    }

    case TokenKind::T_INTEGER_LITERAL: {
      auto ast = new (pool_) IntLiteralExpressionAST();
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
      auto ast = new (pool_) FloatLiteralExpressionAST();
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
      auto ast = new (pool_) NullptrLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = consumeToken();
      ast->literal = unit->tokenKind(ast->literalLoc);
      ast->type = control_->getNullptrType();

      return true;
    }

    case TokenKind::T_USER_DEFINED_STRING_LITERAL: {
      auto ast = new (pool_) UserDefinedStringLiteralExpressionAST();
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

      auto ast = new (pool_) StringLiteralExpressionAST();
      yyast = ast;

      ast->literalLoc = literalLoc;
      ast->literal =
          static_cast<const StringLiteral*>(unit->literal(literalLoc));

      if (unit->tokenKind(literalLoc) == TokenKind::T_STRING_LITERAL) {
        ast->type = control_->getBoundedArrayType(
            control_->getConstType(control_->getCharType()),
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

  auto ast = new (pool_) ModuleUnitAST();
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
  auto ast = new (pool_) TranslationUnitAST();
  yyast = ast;

  moduleUnit_ = false;

  auto it = &ast->declarationList;

  LoopParser loop(this);

  while (LA()) {
    loop.start();

    DeclarationAST* declaration = nullptr;

    if (!parse_declaration(declaration, BindingContext::kNamespace)) {
      parse_error("expected a declaration");
      continue;
    }

    if (declaration) {
      *it = new (pool_) List(declaration);
      it = &(*it)->next;
    }
  }
}

void Parser::parse_declaration_seq(List<DeclarationAST*>*& yyast) {
  auto it = &yyast;

  LoopParser loop(this);

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

    if (parse_maybe_module()) break;

    loop.start();

    DeclarationAST* declaration = nullptr;

    if (parse_declaration(declaration, BindingContext::kNamespace)) {
      if (declaration) {
        *it = new (pool_) List(declaration);
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

auto Parser::parse_primary_expression(ExpressionAST*& yyast,
                                      const ExprContext& ctx) -> bool {
  UnqualifiedIdAST* name = nullptr;

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
  } else {
    return false;
  }
}

auto Parser::parse_id_expression(IdExpressionAST*& yyast,
                                 IdExpressionContext ctx) -> bool {
  LookaheadParser lookahead{this};

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  const auto inRequiresClause = ctx == IdExpressionContext::kRequiresClause;

  UnqualifiedIdAST* unqualifiedId = nullptr;
  if (!parse_unqualified_id(unqualifiedId, isTemplateIntroduced,
                            inRequiresClause))
    return false;

  lookahead.commit();

  auto ast = new (pool_) IdExpressionAST();
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->unqualifiedId = unqualifiedId;
  ast->isTemplateIntroduced = isTemplateIntroduced;

  if (unqualifiedId) {
    auto name = convertName(unqualifiedId);
    if (!nestedNameSpecifier) {
      ast->symbol = unqualifiedLookup(name);
    } else {
      if (nestedNameSpecifier->symbol)
        ast->symbol = qualifiedLookup(nestedNameSpecifier->symbol, name);
    }
  }

  if (ctx == IdExpressionContext::kExpression) {
    if (ast->symbol) {
      ast->type = control_->remove_reference(ast->symbol->type());

      if (auto enumerator = symbol_cast<EnumeratorSymbol>(ast->symbol)) {
        ast->valueCategory = ValueCategory::kPrValue;
      } else {
        ast->valueCategory = ValueCategory::kLValue;
      }
    }
  }

  return true;
}

auto Parser::parse_maybe_template_id(UnqualifiedIdAST*& yyast,
                                     bool isTemplateIntroduced,
                                     bool inRequiresClause) -> bool {
  LookaheadParser lookahead{this};

  auto template_id = parse_template_id(yyast, isTemplateIntroduced);

  if (!template_id) return false;

  if (inRequiresClause) {
    lookahead.commit();
    return true;
  }

  switch (TokenKind(LA())) {
    case TokenKind::T_COLON:
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

auto Parser::parse_unqualified_id(UnqualifiedIdAST*& yyast,
                                  bool isTemplateIntroduced,
                                  bool inRequiresClause) -> bool {
  auto lookat_template_id = [&] {
    LookaheadParser lookahead{this};

    if (!parse_maybe_template_id(yyast, isTemplateIntroduced, inRequiresClause))
      return false;

    lookahead.commit();

    return true;
  };

  if (lookat_template_id()) return true;

  if (SourceLocation tildeLoc; match(TokenKind::T_TILDE, tildeLoc)) {
    if (DecltypeSpecifierAST* decltypeSpecifier = nullptr;
        parse_decltype_specifier(decltypeSpecifier)) {
      auto decltypeName = new (pool_) DecltypeIdAST();
      decltypeName->decltypeSpecifier = decltypeSpecifier;

      auto ast = new (pool_) DestructorIdAST();
      yyast = ast;

      ast->id = decltypeName;
      return true;
    } else if (UnqualifiedIdAST* name = nullptr; parse_type_name(name)) {
      auto ast = new (pool_) DestructorIdAST();
      yyast = ast;

      ast->id = name;
      return true;
    }
    return false;
  } else if (LiteralOperatorIdAST* literalOperatorName = nullptr;
             parse_literal_operator_id(literalOperatorName)) {
    yyast = literalOperatorName;
    return true;
  } else if (ConversionFunctionIdAST* conversionFunctionName = nullptr;
             parse_conversion_function_id(conversionFunctionName)) {
    yyast = conversionFunctionName;
    return true;
  } else if (OperatorFunctionIdAST* functionOperatorName = nullptr;
             parse_operator_function_id(functionOperatorName)) {
    yyast = functionOperatorName;
    return true;
  } else if (NameIdAST* nameId = nullptr; parse_name_id(nameId)) {
    yyast = nameId;
    return true;
  } else {
    return false;
  }
}

void Parser::parse_optional_nested_name_specifier(
    NestedNameSpecifierAST*& yyast) {
  LookaheadParser lookahead(this);
  if (!parse_nested_name_specifier(yyast)) return;
  lookahead.commit();
}

auto Parser::parse_nested_name_specifier(NestedNameSpecifierAST*& yyast)
    -> bool {
  auto lookat_global_nested_name_specifier = [&] {
    if (yyast) return false;

    SourceLocation scopeLoc;
    if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;

    auto ast = new (pool_) GlobalNestedNameSpecifierAST();
    yyast = ast;
    ast->scopeLoc = scopeLoc;
    ast->symbol = globalScope_->owner();

    return true;
  };

  auto lookat_decltype_nested_name_specifier = [&] {
    if (yyast) return false;

    LookaheadParser lookahead{this};

    DecltypeSpecifierAST* decltypeSpecifier = nullptr;
    if (!parse_decltype_specifier(decltypeSpecifier)) return false;

    SourceLocation scopeLoc;
    if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;

    lookahead.commit();

    auto ast = new (pool_) DecltypeNestedNameSpecifierAST();
    ast->nestedNameSpecifier = yyast;
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
  };

  auto lookat_simple_nested_name_specifier = [&] {
    if (!lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON_COLON))
      return false;

    auto identifierLoc = consumeToken();
    auto identifier = unit->identifier(identifierLoc);
    auto scopeLoc = consumeToken();
    Symbol* symbol = nullptr;

    if (!yyast) {
      symbol = unqualifiedLookup(identifier);
    } else {
      if (yyast->symbol) {
        symbol = qualifiedLookup(yyast->symbol, identifier);
      }
    }

    auto ast = new (pool_) SimpleNestedNameSpecifierAST();
    ast->nestedNameSpecifier = yyast;
    yyast = ast;

    ast->identifierLoc = identifierLoc;
    ast->identifier = identifier;
    ast->scopeLoc = scopeLoc;
    ast->symbol = symbol;

    return true;
  };

  auto lookat_template_nested_name_specifier = [&] {
    LookaheadParser lookahead{this};

    SourceLocation templateLoc;
    const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

    SimpleTemplateIdAST* templateName = nullptr;
    if (!parse_simple_template_id(templateName, isTemplateIntroduced))
      return false;

    SourceLocation scopeLoc;
    if (!match(TokenKind::T_COLON_COLON, scopeLoc)) return false;

    lookahead.commit();

    auto ast = new (pool_) TemplateNestedNameSpecifierAST();
    ast->nestedNameSpecifier = yyast;
    yyast = ast;

    ast->templateLoc = templateLoc;
    ast->templateId = templateName;
    ast->scopeLoc = scopeLoc;
    ast->isTemplateIntroduced = isTemplateIntroduced;

    return true;
  };

  const auto start = currentLocation();

  if (auto entry = nested_name_specifiers_.get(start)) {
    auto [cursor, ast, parsed, hit] = *entry;
    rewind(cursor);
    yyast = ast;
    return parsed;
  }

  yyast = nullptr;

  while (true) {
    if (lookat_global_nested_name_specifier())
      continue;
    else if (lookat_simple_nested_name_specifier())
      continue;
    else if (lookat_decltype_nested_name_specifier())
      continue;
    else if (lookat_template_nested_name_specifier())
      continue;
    else
      break;
  }

  const auto parsed = yyast != nullptr;

  nested_name_specifiers_.set(start, currentLocation(), yyast, parsed);

  return parsed;
}

auto Parser::parse_lambda_expression(ExpressionAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_LBRACKET)) return false;

  ScopeGuard scopeGuard{this};

  auto symbol = control_->newLambdaSymbol(scope_);

  scope_ = symbol->scope();

  TemplateHeadContext templateHeadContext{this};

  auto ast = new (pool_) LambdaExpressionAST();
  yyast = ast;

  expect(TokenKind::T_LBRACKET, ast->lbracketLoc);

  if (!match(TokenKind::T_RBRACKET, ast->rbracketLoc)) {
    if (!parse_lambda_capture(ast->captureDefaultLoc, ast->captureList)) {
      parse_error("expected a lambda capture");
    }

    expect(TokenKind::T_RBRACKET, ast->rbracketLoc);
  }

  if (ast->captureDefaultLoc)
    ast->captureDefault = unit->tokenKind(ast->captureDefaultLoc);

  ScopeGuard templateScopeGuard{this};

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

    (void)parse_lambda_specifier_seq(ast->lambdaSpecifierList, symbol);

    (void)parse_noexcept_specifier(ast->exceptionSpecifier);

    (void)parse_trailing_return_type(ast->trailingReturnType);

    parse_optional_attribute_specifier_seq(ast->attributeList,
                                           AllowedAttributes::kAll);

    (void)parse_requires_clause(ast->requiresClause);
  }

  if (auto params = ast->parameterDeclarationClause) {
    auto lambdaScope = symbol->scope();
    std::invoke(DeclareSymbol{this, lambdaScope},
                params->functionParametersSymbol);
    scope_ = params->functionParametersSymbol->scope();
  } else {
    scope_ = symbol->scope();
  }

  if (!lookat(TokenKind::T_LBRACE)) return false;

  std::invoke(DeclareSymbol{this, scope_}, symbol);

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

    auto specifier = new (pool_) LambdaSpecifierAST();
    specifier->specifierLoc = consumeToken();
    specifier->specifier = unit->tokenKind(specifier->specifierLoc);
    *it = new (pool_) List(specifier);
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
    *it = new (pool_) List(capture);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    LambdaCaptureAST* capture = nullptr;

    if (!parse_capture(capture)) parse_error("expected a capture");

    if (capture) {
      *it = new (pool_) List(capture);
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
    auto ast = new (pool_) ThisLambdaCaptureAST();
    yyast = ast;

    ast->thisLoc = thisLoc;

    return true;
  } else if (lookat(TokenKind::T_STAR, TokenKind::T_THIS)) {
    auto ast = new (pool_) DerefThisLambdaCaptureAST();
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

    auto ast = new (pool_) SimpleLambdaCaptureAST();
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

  auto ast = new (pool_) RefLambdaCaptureAST();
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

    auto ast = new (pool_) RefInitLambdaCaptureAST();
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

  auto ast = new (pool_) InitLambdaCaptureAST();
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

  auto ast = new (pool_) LeftFoldExpressionAST();
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

  auto ast = new (pool_) ThisExpressionAST();
  yyast = ast;
  ast->thisLoc = thisLoc;

  return true;
}

auto Parser::parse_nested_expession(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation lparenLoc;

  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  auto ast = new (pool_) NestedExpressionAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;

  parse_expression(ast->expression, ctx);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  if (ast->expression) {
    ast->type = ast->expression->type;
    ast->valueCategory = ast->expression->valueCategory;
  }

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
    auto ast = new (pool_) RightFoldExpressionAST();
    yyast = ast;

    ast->lparenLoc = lparenLoc;
    ast->expression = expression;
    ast->opLoc = opLoc;
    ast->op = op;
    ast->ellipsisLoc = ellipsisLoc;
    ast->rparenLoc = rparenLoc;

    return true;
  }

  auto ast = new (pool_) FoldExpressionAST();
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

  auto ast = new (pool_) RequiresExpressionAST();
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

  *it = new (pool_) List(requirement);
  it = &(*it)->next;

  LoopParser loop(this);

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    RequirementAST* requirement = nullptr;
    parse_requirement(requirement);

    *it = new (pool_) List(requirement);
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

  auto ast = new (pool_) SimpleRequirementAST();
  yyast = ast;

  ast->expression = expression;
  ast->semicolonLoc = semicolonLoc;
}

auto Parser::parse_type_requirement(RequirementAST*& yyast) -> bool {
  SourceLocation typenameLoc;

  if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

  auto ast = new (pool_) TypeRequirementAST();
  yyast = ast;

  parse_optional_nested_name_specifier(ast->nestedNameSpecifier);

  if (!parse_type_name(ast->unqualifiedId)) parse_error("expected a type name");

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_compound_requirement(RequirementAST*& yyast) -> bool {
  SourceLocation lbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  ScopeGuard scopeGuard{this};

  ExpressionAST* expression = nullptr;

  parse_expression(expression, ExprContext{});

  SourceLocation rbraceLoc;

  expect(TokenKind::T_RBRACE, rbraceLoc);

  auto ast = new (pool_) CompoundRequirementAST();
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

  auto ast = new (pool_) NestedRequirementAST();
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
  if (parse_builtin_call_expression(yyast, ctx))
    return true;
  else if (parse_va_arg_expression(yyast, ctx))
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

  auto ast = new (pool_) MemberExpressionAST();
  ast->baseExpression = yyast;
  ast->accessLoc = accessLoc;
  ast->accessOp = unit->tokenKind(accessLoc);

  parse_optional_nested_name_specifier(ast->nestedNameSpecifier);

  ast->isTemplateIntroduced = match(TokenKind::T_TEMPLATE, ast->templateLoc);

  if (!parse_unqualified_id(ast->unqualifiedId, ast->isTemplateIntroduced,
                            /*inRequiresClause*/ false))
    parse_error("expected an unqualified id");

  yyast = ast;

  return true;
}

auto Parser::parse_subscript_expression(ExpressionAST*& yyast,
                                        const ExprContext& ctx) -> bool {
  SourceLocation lbracketLoc;

  if (!match(TokenKind::T_LBRACKET, lbracketLoc)) return false;

  auto ast = new (pool_) SubscriptExpressionAST();
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

  auto ast = new (pool_) CallExpressionAST();
  ast->baseExpression = yyast;
  ast->lparenLoc = lparenLoc;

  yyast = ast;

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression_list(ast->expressionList, ctx)) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  std::vector<const Type*> argumentTypes;

  for (auto it = ast->expressionList; it; it = it->next) {
    const Type* argumentType = nullptr;
    if (it->value) argumentType = it->value->type;

    argumentTypes.push_back(argumentType);
  }

#if false
  if (auto ovlType = type_cast<OverloadSetType>(ast->baseExpression->type)) {
    parse_warn(lparenLoc, "overload set call");
  }

  if (auto functionType = type_cast<FunctionType>(ast->baseExpression->type)) {
    parse_warn(lparenLoc,
               cxx::format("call function {}", to_string(functionType)));
  }
#endif

  return true;
}

auto Parser::parse_postincr_expression(ExpressionAST*& yyast,
                                       const ExprContext& ctx) -> bool {
  SourceLocation opLoc;

  if (!match(TokenKind::T_MINUS_MINUS, opLoc) &&
      !match(TokenKind::T_PLUS_PLUS, opLoc)) {
    return false;
  }

  auto ast = new (pool_) PostIncrExpressionAST();
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

  auto ast = new (pool_) CppCastExpressionAST();
  yyast = ast;

  ast->castLoc = castLoc;

  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  expect(TokenKind::T_GREATER, ast->greaterLoc);

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_expression(ast->expression, ctx);

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_builtin_bit_cast_expression(ExpressionAST*& yyast,
                                               const ExprContext& ctx) -> bool {
  if (!lookat(BuiltinKind::T___BUILTIN_BIT_CAST)) return false;

  auto ast = new (pool_) BuiltinBitCastExpressionAST();
  yyast = ast;

  ast->castLoc = consumeToken();
  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");
  expect(TokenKind::T_COMMA, ast->commaLoc);
  parse_expression(ast->expression, ctx);
  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_cpp_type_cast_expression(ExpressionAST*& yyast,
                                            const ExprContext& ctx) -> bool {
  auto lookat_function_call = [&] {
    LookaheadParser lookahead{this};

    SpecifierAST* typeSpecifier = nullptr;
    DeclSpecs specs{this};

    if (!parse_simple_type_specifier(typeSpecifier, specs)) return false;

    if (!lookat(TokenKind::T_LPAREN)) return false;

    // ### prefer function calls to cpp-cast expressions for now.
    if (ast_cast<NamedTypeSpecifierAST>(typeSpecifier)) return true;

    return false;
  };

  auto lookat_braced_type_construction = [&] {
    LookaheadParser lookahead{this};

    SpecifierAST* typeSpecifier = nullptr;
    DeclSpecs specs{this};

    if (!parse_simple_type_specifier(typeSpecifier, specs)) return false;

    BracedInitListAST* bracedInitList = nullptr;

    if (!parse_braced_init_list(bracedInitList, ctx)) return false;

    lookahead.commit();

    auto ast = new (pool_) BracedTypeConstructionAST();
    yyast = ast;

    ast->typeSpecifier = typeSpecifier;
    ast->bracedInitList = bracedInitList;

    return true;
  };

  if (lookat_function_call()) return false;
  if (lookat_braced_type_construction()) return true;

  LookaheadParser lookahead{this};

  SpecifierAST* typeSpecifier = nullptr;
  DeclSpecs specs{this};

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

  auto ast = new (pool_) TypeConstructionAST();
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

    auto ast = new (pool_) TypeidOfTypeExpressionAST();
    yyast = ast;

    ast->typeidLoc = typeidLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;

    return true;
  };

  if (lookat_typeid_of_type()) return true;

  auto ast = new (pool_) TypeidExpressionAST();
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
  DeclSpecs specs{this};
  if (!parse_typename_specifier(typenameSpecifier, specs)) return false;

  if (BracedInitListAST* bracedInitList = nullptr;
      parse_braced_init_list(bracedInitList, ctx)) {
    lookahead.commit();

    auto ast = new (pool_) BracedTypeConstructionAST();
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

  auto ast = new (pool_) TypeConstructionAST();
  yyast = ast;

  ast->typeSpecifier = typenameSpecifier;
  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;

  return true;
}

auto Parser::parse_type_traits_op(SourceLocation& loc, BuiltinKind& builtinKind)
    -> bool {
  if (!LA().isBuiltinTypeTrait()) return false;
  builtinKind = static_cast<BuiltinKind>(LA().value().intValue);
  loc = consumeToken();
  return true;
}

auto Parser::parse_va_arg_expression(ExpressionAST*& yyast,
                                     const ExprContext& ctx) -> bool {
  SourceLocation vaArgLoc;
  if (!match(TokenKind::T___BUILTIN_VA_ARG, vaArgLoc)) return false;
  auto ast = new (pool_) VaArgExpressionAST();
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
  SourceLocation typeTraitsLoc;
  BuiltinKind builtinKind = BuiltinKind::T_IDENTIFIER;
  if (!parse_type_traits_op(typeTraitsLoc, builtinKind)) return false;

  auto ast = new (pool_) TypeTraitsExpressionAST();
  yyast = ast;

  ast->typeTraitsLoc = typeTraitsLoc;
  ast->typeTraits = builtinKind;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  auto it = &ast->typeIdList;

  if (TypeIdAST* typeId = nullptr; parse_type_id(typeId)) {
    *it = new (pool_) List(typeId);
    it = &(*it)->next;
  } else {
    parse_error("expected a type id");
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (TypeIdAST* typeId = nullptr; parse_type_id(typeId)) {
      *it = new (pool_) List(typeId);
      it = &(*it)->next;
    } else {
      parse_error("expected a type id");
    }
  }

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  return true;
}

auto Parser::parse_expression_list(List<ExpressionAST*>*& yyast,
                                   const ExprContext& ctx) -> bool {
  return parse_initializer_list(yyast, ctx);
}

auto Parser::parse_unary_expression(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  if (parse_unop_expression(yyast, ctx))
    return true;
  else if (parse_complex_expression(yyast, ctx))
    return true;
  else if (parse_await_expression(yyast, ctx))
    return true;
  else if (parse_sizeof_expression(yyast, ctx))
    return true;
  else if (parse_alignof_expression(yyast, ctx))
    return true;
  else if (parse_noexcept_expression(yyast, ctx))
    return true;
  else if (parse_new_expression(yyast, ctx))
    return true;
  else if (parse_delete_expression(yyast, ctx))
    return true;
  else
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

  auto ast = new (pool_) UnaryExpressionAST();
  yyast = ast;

  ast->opLoc = opLoc;
  ast->op = unit->tokenKind(opLoc);
  ast->expression = expression;

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

  auto ast = new (pool_) UnaryExpressionAST();
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
    auto ast = new (pool_) SizeofPackExpressionAST();
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

    auto ast = new (pool_) SizeofTypeExpressionAST();
    yyast = ast;

    ast->sizeofLoc = sizeofLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;
    ast->type = control_->getSizeType();

    return true;
  };

  if (lookat_sizeof_type_id()) return true;

  auto ast = new (pool_) SizeofExpressionAST();
  yyast = ast;

  ast->sizeofLoc = sizeofLoc;

  if (!parse_unary_expression(ast->expression, ctx)) {
    parse_error("expected an expression");
  }

  ast->type = control_->getSizeType();

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

    auto ast = new (pool_) AlignofTypeExpressionAST();
    yyast = ast;

    ast->alignofLoc = alignofLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;
    ast->type = control_->getSizeType();

    return true;
  };

  if (lookat_alignof_type_id()) return true;

  auto ast = new (pool_) AlignofExpressionAST();
  yyast = ast;

  ast->alignofLoc = alignofLoc;

  if (!parse_unary_expression(ast->expression, ctx)) {
    parse_error("expected an expression");
  }

  ast->type = control_->getSizeType();

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

  auto ast = new (pool_) AwaitExpressionAST();
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

  auto ast = new (pool_) NoexceptExpressionAST();
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

  auto ast = new (pool_) NewExpressionAST();
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
    DeclSpecs specs{this};
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

  DeclSpecs specs{this};
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

  auto ast = new (pool_) NewPlacementAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->expressionList = expressionList;
  ast->rparenLoc = rparenLoc;
}

void Parser::parse_optional_new_initializer(NewInitializerAST*& yyast,
                                            const ExprContext& ctx) {
  if (BracedInitListAST* bracedInitList = nullptr;
      parse_braced_init_list(bracedInitList, ctx)) {
    auto ast = new (pool_) NewBracedInitializerAST();
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

  auto ast = new (pool_) NewParenInitializerAST();
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

  auto ast = new (pool_) DeleteExpressionAST();
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

  if (auto it = cast_expressions_.get(start)) {
    auto [endLoc, ast, parsed, hit] = *it;
    rewind(endLoc);
    yyast = ast;
    return parsed;
  }

  auto lookat_cast_expression = [&] {
    LookaheadParser lookahead{this};
    if (!parse_cast_expression_helper(yyast, ctx)) return false;
    lookahead.commit();
    return true;
  };

  auto parsed = lookat_cast_expression();

  if (!parsed) {
    parsed = parse_unary_expression(yyast, ctx);
  }

  cast_expressions_.set(start, currentLocation(), yyast, parsed);

  return parsed;
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

  if (!parse_cast_expression(expression, ctx)) {
    return false;
  }

  auto ast = new (pool_) CastExpressionAST();
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

    auto ast = new (pool_) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = rhs;
    ast->op = op;

    if (ast->leftExpression && ast->rightExpression) {
      switch (ast->op) {
        case TokenKind::T_DOT_STAR:
          break;

        case TokenKind::T_MINUS_GREATER_STAR:
          break;

        case TokenKind::T_STAR:
        case TokenKind::T_SLASH:
        case TokenKind::T_PLUS:
        case TokenKind::T_MINUS:
          ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                                  ast->rightExpression);
          break;

        case TokenKind::T_PERCENT:
          if (!control_->is_integral_or_unscoped_enum(
                  ast->leftExpression->type))
            break;

          if (!control_->is_integral_or_unscoped_enum(
                  ast->rightExpression->type))
            break;

          ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                                  ast->rightExpression);

          break;

        case TokenKind::T_LESS_LESS:
        case TokenKind::T_GREATER_GREATER:
          if (!control_->is_integral_or_unscoped_enum(
                  ast->leftExpression->type))
            break;

          if (!control_->is_integral_or_unscoped_enum(
                  ast->rightExpression->type))
            break;

          (void)usual_arithmetic_conversion(ast->leftExpression,
                                            ast->rightExpression);

          ast->type = ast->leftExpression->type;
          break;

        case TokenKind::T_LESS_EQUAL_GREATER:
          (void)usual_arithmetic_conversion(ast->leftExpression,
                                            ast->rightExpression);
          ast->type = control_->getIntType();
          break;

        case TokenKind::T_LESS_EQUAL:
        case TokenKind::T_GREATER_EQUAL:
        case TokenKind::T_LESS:
        case TokenKind::T_GREATER:
        case TokenKind::T_EQUAL_EQUAL:
        case TokenKind::T_EXCLAIM_EQUAL:
          (void)usual_arithmetic_conversion(ast->leftExpression,
                                            ast->rightExpression);
          ast->type = control_->getBoolType();
          break;

        case TokenKind::T_AMP:
        case TokenKind::T_CARET:
        case TokenKind::T_BAR:
          if (!control_->is_integral_or_unscoped_enum(
                  ast->leftExpression->type))
            break;

          if (!control_->is_integral_or_unscoped_enum(
                  ast->rightExpression->type))
            break;

          ast->type = usual_arithmetic_conversion(ast->leftExpression,
                                                  ast->rightExpression);

          break;

        case TokenKind::T_AMP_AMP:
        case TokenKind::T_BAR_BAR:
          (void)implicit_conversion(ast->leftExpression,
                                    control_->getBoolType());

          (void)implicit_conversion(ast->rightExpression,
                                    control_->getBoolType());

          ast->type = control_->getBoolType();
          break;

        default:
          cxx_runtime_error("invalid operator");
      }  // switch
    }

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
    auto ast = new (pool_) ConditionalExpressionAST();
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

  auto ast = new (pool_) YieldExpressionAST();
  yyast = ast;

  ast->yieldLoc = yieldLoc;
  parse_expr_or_braced_init_list(ast->expression, ctx);

  return true;
}

auto Parser::parse_throw_expression(ExpressionAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation throwLoc;

  if (!match(TokenKind::T_THROW, throwLoc)) return false;

  auto ast = new (pool_) ThrowExpressionAST();
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

    auto ast = new (pool_) AssignmentExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->rightExpression = expression;
    ast->op = op;

    if (ast->leftExpression && ast->rightExpression) {
      ast->type = ast->leftExpression->type;

      auto sourceType = ast->rightExpression->type;

      (void)implicit_conversion(ast->rightExpression, ast->type);

#if false
      parse_warning(ast->opLoc,
              cxx::format("did convert {} to {}", to_string(sourceType),
                          to_string(ast->type)));
#endif
    }

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

    auto ast = new (pool_) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = commaLoc;
    ast->op = TokenKind::T_COMMA;
    ast->rightExpression = expression;
    if (ast->rightExpression) {
      ast->type = ast->rightExpression->type;
    }
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

  if (parse_case_statement(yyast)) {
    return true;
  } else if (parse_default_statement(yyast)) {
    return true;
  } else if (parse_while_statement(yyast)) {
    return true;
  } else if (parse_do_statement(yyast)) {
    return true;
  } else if (parse_for_statement(yyast)) {
    return true;
  } else if (parse_if_statement(yyast)) {
    return true;
  } else if (parse_switch_statement(yyast)) {
    return true;
  } else if (parse_break_statement(yyast)) {
    return true;
  } else if (parse_continue_statement(yyast)) {
    return true;
  } else if (parse_return_statement(yyast)) {
    return true;
  } else if (parse_goto_statement(yyast)) {
    return true;
  } else if (parse_coroutine_return_statement(yyast)) {
    return true;
  } else if (parse_try_block(yyast)) {
    return true;
  } else if (parse_maybe_compound_statement(yyast)) {
    return true;
  } else if (parse_labeled_statement(yyast)) {
    return true;
  } else {
    auto lookat_declaration_statement = [&] {
      LookaheadParser lookahead{this};
      if (!parse_declaration_statement(yyast)) return false;
      lookahead.commit();
      return true;
    };

    if (lookat_declaration_statement()) return true;

    return parse_expression_statement(yyast);
  }
}

void Parser::parse_init_statement(StatementAST*& yyast) {
  auto lookat_simple_declaration = [&] {
    LookaheadParser lookahead{this};
    DeclarationAST* declaration = nullptr;
    if (!parse_simple_declaration(declaration, BindingContext::kInitStatement))
      return false;
    lookahead.commit();

    auto ast = new (pool_) DeclarationStatementAST();
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

  auto ast = new (pool_) ExpressionStatementAST();
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

    DeclSpecs specs{this};

    if (!parse_decl_specifier_seq(declSpecifierList, specs)) return false;

    DeclaratorAST* declarator = nullptr;
    Decl decl{specs};
    if (!parse_declarator(declarator, decl)) return false;

    ExpressionAST* initializer = nullptr;

    if (!parse_brace_or_equal_initializer(initializer)) return false;

    lookahead.commit();

    auto ast = new (pool_) ConditionExpressionAST();
    yyast = ast;
    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->declarator = declarator;
    ast->initializer = initializer;

    return true;
  };

  if (lookat_condition()) return;

  parse_expression(yyast, ctx);
}

auto Parser::parse_labeled_statement(StatementAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON)) return false;

  auto ast = new (pool_) LabeledStatementAST();
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

  auto ast = new (pool_) CaseStatementAST();
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

  auto ast = new (pool_) DefaultStatementAST();
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

  auto ast = new (pool_) ExpressionStatementAST;
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

  ScopeGuard scopeGuard{this};

  auto blockSymbol = control_->newBlockSymbol(scope_);
  std::invoke(DeclareSymbol{this, scope_}, blockSymbol);
  scope_ = blockSymbol->scope();

  auto ast = new (pool_) CompoundStatementAST();
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
  ScopeGuard scopeGuard{this};

  scope_ = ast->symbol->scope();

  bool skipping = false;

  auto it = &ast->statementList;

  LoopParser loop{this};

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    StatementAST* statement = nullptr;

    if (parse_maybe_statement(statement)) {
      *it = new (pool_) List(statement);
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

  ScopeGuard scopeGuard{this};

  if (LA().isOneOf(TokenKind::T_EXCLAIM, TokenKind::T_CONSTEVAL)) {
    auto ast = new (pool_) ConstevalIfStatementAST();
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

  auto ast = new (pool_) IfStatementAST();
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

  ScopeGuard scopeGuard{this};

  auto ast = new (pool_) SwitchStatementAST();
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

  ScopeGuard scopeGuard{this};

  auto ast = new (pool_) WhileStatementAST();
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

  ScopeGuard scopeGuard{this};

  auto ast = new (pool_) DoStatementAST();
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

auto Parser::parse_for_range_statement(StatementAST*& yyast) -> bool {
  LookaheadParser lookahead{this};

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

  lookahead.commit();

  auto ast = new (pool_) ForRangeStatementAST();
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
  if (parse_for_range_statement(yyast)) return true;

  SourceLocation forLoc;

  if (!match(TokenKind::T_FOR, forLoc)) return false;

  auto ast = new (pool_) ForStatementAST();
  yyast = ast;

  ast->forLoc = forLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  parse_init_statement(ast->initializer);

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

  DeclSpecs specs{this};

  if (!parse_decl_specifier_seq(declSpecifierList, specs)) return false;

  if (parse_structured_binding(yyast, attributeList, declSpecifierList, specs,
                               BindingContext::kCondition)) {
    return true;
  }

  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  if (!parse_declarator(declarator, decl)) return false;

  auto initDeclarator = new (pool_) InitDeclaratorAST();
  initDeclarator->declarator = declarator;

  auto ast = new (pool_) SimpleDeclarationAST();
  yyast = ast;

  ast->attributeList = attributeList;
  ast->declSpecifierList = declSpecifierList;
  ast->initDeclaratorList = new (pool_) List(initDeclarator);

  return true;
}

void Parser::parse_for_range_initializer(ExpressionAST*& yyast) {
  parse_expr_or_braced_init_list(yyast, ExprContext{});
}

auto Parser::parse_break_statement(StatementAST*& yyast) -> bool {
  SourceLocation breakLoc;

  if (!match(TokenKind::T_BREAK, breakLoc)) return false;

  auto ast = new (pool_) BreakStatementAST();
  yyast = ast;

  ast->breakLoc = breakLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_continue_statement(StatementAST*& yyast) -> bool {
  SourceLocation continueLoc;

  if (!match(TokenKind::T_CONTINUE, continueLoc)) return false;

  auto ast = new (pool_) ContinueStatementAST();
  yyast = ast;

  ast->continueLoc = continueLoc;

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_return_statement(StatementAST*& yyast) -> bool {
  SourceLocation returnLoc;

  if (!match(TokenKind::T_RETURN, returnLoc)) return false;

  auto ast = new (pool_) ReturnStatementAST();
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

  auto ast = new (pool_) GotoStatementAST();
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

  auto ast = new (pool_) CoroutineReturnStatementAST();
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

  auto ast = new (pool_) DeclarationStatementAST();
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

  if (!templateDeclarations.empty()) {
    mark_maybe_template_name(unit->identifier(identifierLoc));
  }

  TypeIdAST* typeId = nullptr;

  if (!parse_defining_type_id(typeId, templateDeclarations))
    parse_error("expected a type id");

  SourceLocation semicolonLoc;

  expect(TokenKind::T_SEMICOLON, semicolonLoc);

  auto aliasName = unit->identifier(identifierLoc);

  auto aliasSymbol = control_->newTypeAliasSymbol(scope_);
  aliasSymbol->setName(aliasName);
  if (typeId) aliasSymbol->setType(typeId->type);
  std::invoke(DeclareSymbol{this, scope_}, aliasSymbol);

  auto ast = new (pool_) AliasDeclarationAST;
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = aliasName;
  ast->attributeList = attributes;
  ast->equalLoc = equalLoc;
  ast->typeId = typeId;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::enterOrCreateNamespace(const Name* name, bool isInline)
    -> NamespaceSymbol* {
  auto parentScope = scope_;
  auto parentNamespace = symbol_cast<NamespaceSymbol>(parentScope->owner());

  NamespaceSymbol* namespaceSymbol = nullptr;

  if (!name) {
    namespaceSymbol = parentNamespace->unnamedNamespace();
  } else {
    auto resolved =
        parentScope->get(name) | std::views::filter(&Symbol::isNamespace);
    if (std::ranges::distance(resolved) == 1) {
      namespaceSymbol = symbol_cast<NamespaceSymbol>(*begin(resolved));
    }
  }

  if (!namespaceSymbol) {
    namespaceSymbol = control_->newNamespaceSymbol(parentScope);

    if (name) {
      namespaceSymbol->setName(name);
    } else {
      parentNamespace->setUnnamedNamespace(namespaceSymbol);
    }

    namespaceSymbol->setInline(isInline);

    std::invoke(DeclareSymbol{this, parentScope}, namespaceSymbol);

    if (isInline || !namespaceSymbol->name()) {
      parentNamespace->scope()->addUsingDirective(namespaceSymbol->scope());
    }
  }

  scope_ = namespaceSymbol->scope();

  return namespaceSymbol;
}

void Parser::enterFunctionScope(
    FunctionDeclaratorChunkAST* functionDeclarator) {}

void Parser::applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setExtern(specs.isExtern);
  symbol->setFriend(specs.isFriend);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConsteval(specs.isConsteval);
  symbol->setInline(specs.isInline);
  symbol->setVirtual(specs.isVirtual);
  symbol->setExplicit(specs.isExplicit);
}

void Parser::applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setExtern(specs.isExtern);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

void Parser::applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

auto Parser::parse_template_class_declaration(
    DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations,
    BindingContext ctx) -> bool {
  LookaheadParser lookahead{this};

  if (ctx != BindingContext::kTemplate) return false;

  ClassSpecifierAST* classSpecifier = nullptr;
  DeclSpecs specs{this};
  if (!parse_class_specifier(classSpecifier, specs, templateDeclarations))
    return false;

  lookahead.commit();

  List<AttributeSpecifierAST*>* trailingAttributes = nullptr;
  (void)parse_attribute_specifier_seq(trailingAttributes);

  SourceLocation semicolonToken;
  expect(TokenKind::T_SEMICOLON, semicolonToken);

  auto ast = new (pool_) SimpleDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->declSpecifierList = new (pool_) List<SpecifierAST*>(classSpecifier);
  ast->semicolonLoc = semicolonToken;

  return true;
}

auto Parser::parse_empty_or_attribute_declaration(
    DeclarationAST*& yyast, List<AttributeSpecifierAST*>* attributes) -> auto {
  LookaheadParser lookahead{this};

  SourceLocation semicolonLoc;

  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  lookahead.commit();

  if (attributes) {
    auto ast = new (pool_) AttributeDeclarationAST();
    yyast = ast;
    ast->attributeList = attributes;
    ast->semicolonLoc = semicolonLoc;
    return true;
  }

  auto ast = new (pool_) EmptyDeclarationAST();
  yyast = ast;
  ast->semicolonLoc = semicolonLoc;
  return true;
}

auto Parser::parse_notypespec_function_definition(
    DeclarationAST*& yyast, List<AttributeSpecifierAST*>* atributes,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations,
    BindingContext ctx) -> bool {
  LookaheadParser lookahead{this};

  if (!context_allows_function_definition(ctx)) return false;

  DeclSpecs specs{this};
  List<SpecifierAST*>* declSpecifierList = nullptr;

  auto parse_optional_decl_specifier_seq_no_typespecs = [&] {
    LookaheadParser lookahead{this};
    if (!parse_decl_specifier_seq_no_typespecs(declSpecifierList, specs)) {
      specs = DeclSpecs{this};
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

  auto ast = new (pool_) SimpleDeclarationAST();
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

  auto ast = new (pool_) StructuredBindingDeclarationAST();
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
    auto ast = new (pool_) AttributeDeclarationAST();
    yyast = ast;

    ast->attributeList = attributes;
    ast->semicolonLoc = semicolonLoc;
    return true;
  }

  if (parse_template_class_declaration(yyast, attributes, templateDeclarations,
                                       ctx))
    return true;
  else if (parse_empty_or_attribute_declaration(yyast, attributes))
    return true;
  else if (parse_notypespec_function_definition(yyast, attributes,
                                                templateDeclarations, ctx))
    return true;

  DeclSpecs specs{this};
  List<SpecifierAST*>* declSpecifierList = nullptr;

  auto lookat_decl_specifiers = [&] {
    LookaheadParser lookahead{this};
    if (!parse_decl_specifier_seq(declSpecifierList, specs)) return false;
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

    ScopeGuard scopeGuard{this};

    auto functionType =
        GetDeclaratorType{this}(declarator, decl.specs.getType());

    const Name* functionName = decl.getName();
    FunctionSymbol* functionSymbol = nullptr;

    if (decl.declaratorId->nestedNameSpecifier) {
      auto enclosingSymbol = decl.declaratorId->nestedNameSpecifier->symbol;

      if (!enclosingSymbol) {
        if (config_.checkTypes) {
          parse_error(
              decl.declaratorId->nestedNameSpecifier->firstSourceLocation(),
              cxx::format("unresolved class or namespace"));
        }
      } else {
        if (auto classSymbol = symbol_cast<ClassSymbol>(enclosingSymbol)) {
          scope_ = classSymbol->scope();
        } else if (auto namespaceSymbol =
                       symbol_cast<NamespaceSymbol>(enclosingSymbol)) {
          scope_ = namespaceSymbol->scope();
        } else {
          if (config_.checkTypes) {
            parse_error("expected a class or namespace");
          }
        }

        for (auto s : scope_->get(functionName)) {
          if (auto previous = symbol_cast<FunctionSymbol>(s)) {
            if (control_->is_same(previous->type(), functionType)) {
              functionSymbol = previous;
              break;
            }
          } else if (auto ovl = symbol_cast<OverloadSetSymbol>(s)) {
            for (auto previous : ovl->functions()) {
              if (control_->is_same(previous->type(), functionType)) {
                functionSymbol = previous;
                break;
              }
            }
          }
        }

        if (!functionSymbol) {
          if (config_.checkTypes) {
            parse_error(decl.declaratorId->unqualifiedId->firstSourceLocation(),
                        cxx::format("'{}' has no member named '{}'",
                                    to_string(enclosingSymbol->name()),
                                    to_string(functionName)));
          }
        }
      }
    } else {
      for (auto s : scope_->get(functionName)) {
        if (auto previous = symbol_cast<FunctionSymbol>(s)) {
          if (control_->is_same(previous->type(), functionType)) {
            functionSymbol = previous;
            break;
          }
        } else if (auto ovl = symbol_cast<OverloadSetSymbol>(s)) {
          for (auto previous : ovl->functions()) {
            if (control_->is_same(previous->type(), functionType)) {
              functionSymbol = previous;
              break;
            }
          }
        }
      }
    }

    if (!functionSymbol) {
      functionSymbol = control_->newFunctionSymbol(scope_);
      applySpecifiers(functionSymbol, decl.specs);
      functionSymbol->setName(functionName);
      functionSymbol->setType(functionType);
      std::invoke(DeclareSymbol{this, scope_}, functionSymbol);
    }

    if (auto params = functionDeclarator->parameterDeclarationClause) {
      auto functionScope = functionSymbol->scope();
      std::invoke(DeclareSymbol{this, functionScope},
                  params->functionParametersSymbol);
      scope_ = params->functionParametersSymbol->scope();
    } else {
      scope_ = functionSymbol->scope();
    }

    if (ctx == BindingContext::kTemplate) {
      mark_maybe_template_name(declarator);
    }

    FunctionBodyAST* functionBody = nullptr;
    if (!parse_function_body(functionBody))
      parse_error("expected function body");

    lookahead.commit();

    auto ast = new (pool_) FunctionDefinitionAST();
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
  if (!parse_init_declarator(initDeclarator, declarator, decl)) return false;

  if (ctx == BindingContext::kTemplate) {
    auto declarator = initDeclarator->declarator;
    mark_maybe_template_name(declarator);
  }

  *declIt = new (pool_) List(initDeclarator);
  declIt = &(*declIt)->next;

  if (ctx != BindingContext::kTemplate) {
    SourceLocation commaLoc;

    while (match(TokenKind::T_COMMA, commaLoc)) {
      InitDeclaratorAST* initDeclarator = nullptr;
      if (!parse_init_declarator(initDeclarator, specs)) return false;

      *declIt = new (pool_) List(initDeclarator);
      declIt = &(*declIt)->next;
    }
  }

  SourceLocation semicolonLoc;
  if (!match(TokenKind::T_SEMICOLON, semicolonLoc)) return false;

  auto ast = new (pool_) SimpleDeclarationAST();
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

  ScopeGuard scopeGuard{this};

  if (decl.declaratorId) {
    auto nested = decl.declaratorId->nestedNameSpecifier;
    if (nested) {
      if (auto classSymbol = symbol_cast<ClassSymbol>(nested->symbol)) {
        scope_ = classSymbol->scope();
      } else if (auto namespaceSymbol =
                     symbol_cast<NamespaceSymbol>(nested->symbol)) {
        scope_ = namespaceSymbol->scope();
      } else {
        if (config_.checkTypes) {
          parse_error("expected a class or namespace");
        }
      }
    }
  }

  FunctionDeclaratorChunkAST* functionDeclarator = nullptr;
  if (!parse_function_declarator(functionDeclarator)) return false;

  auto declarator = new (pool_) DeclaratorAST();
  declarator->coreDeclarator = declaratorId;

  declarator->declaratorChunkList =
      new (pool_) List<DeclaratorChunkAST*>(functionDeclarator);

  RequiresClauseAST* requiresClause = nullptr;

  const auto has_requires_clause = parse_requires_clause(requiresClause);

  if (!has_requires_clause) parse_virt_specifier_seq(functionDeclarator);

  parse_optional_attribute_specifier_seq(functionDeclarator->attributeList);

  auto functionType = GetDeclaratorType{this}(declarator, decl.specs.getType());

  SourceLocation equalLoc;
  SourceLocation zeroLoc;

  const auto isPure = parse_pure_specifier(equalLoc, zeroLoc);

  functionDeclarator->isPure = isPure;

  const auto isDeclaration = isPure || lookat(TokenKind::T_SEMICOLON);
  const auto isDefinition = lookat_function_body();

  if (!isDeclaration && !isDefinition) return false;

  FunctionSymbol* functionSymbol = nullptr;
  functionSymbol = control_->newFunctionSymbol(scope_);
  applySpecifiers(functionSymbol, decl.specs);
  functionSymbol->setName(decl.getName());
  functionSymbol->setType(functionType);

  if (is_constructor(functionSymbol)) {
    // constructors don't have names
    if (auto enclosingClass = symbol_cast<ClassSymbol>(scope_->owner())) {
      if (!decl.declaratorId->nestedNameSpecifier) {
        enclosingClass->addConstructor(functionSymbol);
      }
    }
  } else {
    std::invoke(DeclareSymbol{this, scope_}, functionSymbol);
  }

  SourceLocation semicolonLoc;

  if (isPure) {
    expect(TokenKind::T_SEMICOLON, semicolonLoc);
  }

  if (isDeclaration) {
    auto initDeclarator = new (pool_) InitDeclaratorAST();
    initDeclarator->declarator = declarator;

    auto ast = new (pool_) SimpleDeclarationAST();
    yyast = ast;
    ast->declSpecifierList = declSpecifierList;
    ast->initDeclaratorList = new (pool_) List(initDeclarator);
    ast->requiresClause = requiresClause;
    ast->semicolonLoc = semicolonLoc;

    return true;
  }

  // function definition

  if (auto params = functionDeclarator->parameterDeclarationClause) {
    auto functionScope = functionSymbol->scope();
    std::invoke(DeclareSymbol{this, functionScope},
                params->functionParametersSymbol);
    scope_ = params->functionParametersSymbol->scope();
  } else {
    scope_ = functionSymbol->scope();
  }

  FunctionBodyAST* functionBody = nullptr;

  if (!parse_function_body(functionBody)) parse_error("expected function body");

  auto ast = new (pool_) FunctionDefinitionAST();
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

  auto ast = new (pool_) StaticAssertDeclarationAST();
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

  bool value = false;

  if (constValue.has_value()) {
    value = visit(to_bool, *constValue);
  }

  if (!value && config_.staticAssert) {
    SourceLocation loc = ast->firstSourceLocation();

    if (!ast->expression || !constValue.has_value()) {
      parse_error(
          loc,
          "static assertion expression is not an integral constant expression");
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

  auto ast = new (pool_) EmptyDeclarationAST();
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

  auto ast = new (pool_) AttributeDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->semicolonLoc = semicolonLoc;

  return true;
}

auto Parser::parse_decl_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  switch (TokenKind(LA())) {
    case TokenKind::T_TYPEDEF: {
      auto ast = new (pool_) TypedefSpecifierAST();
      yyast = ast;
      ast->typedefLoc = consumeToken();
      specs.isTypedef = true;
      return true;
    }

    case TokenKind::T_FRIEND: {
      auto ast = new (pool_) FriendSpecifierAST();
      yyast = ast;
      ast->friendLoc = consumeToken();
      specs.isFriend = true;
      return true;
    }

    case TokenKind::T_CONSTEXPR: {
      auto ast = new (pool_) ConstexprSpecifierAST();
      yyast = ast;
      ast->constexprLoc = consumeToken();
      specs.isConstexpr = true;
      return true;
    }

    case TokenKind::T_CONSTEVAL: {
      auto ast = new (pool_) ConstevalSpecifierAST();
      yyast = ast;
      ast->constevalLoc = consumeToken();
      specs.isConsteval = true;
      return true;
    }

    case TokenKind::T_CONSTINIT: {
      auto ast = new (pool_) ConstinitSpecifierAST();
      yyast = ast;
      ast->constinitLoc = consumeToken();
      specs.isConstinit = true;
      return true;
    }

    case TokenKind::T_INLINE: {
      auto ast = new (pool_) InlineSpecifierAST();
      yyast = ast;
      ast->inlineLoc = consumeToken();
      specs.isInline = true;
      return true;
    }

    default:
      if (parse_storage_class_specifier(yyast, specs)) return true;

      if (parse_function_specifier(yyast, specs)) return true;

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

  *it = new (pool_) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeSpecifierAST*>* attributes = nullptr;
    parse_optional_attribute_specifier_seq(attributes);

    *it = new (pool_) List(specifier);
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

  *it = new (pool_) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_decl_specifier(specifier, specs)) {
    List<AttributeSpecifierAST*>* attributes = nullptr;

    parse_optional_attribute_specifier_seq(attributes);

    *it = new (pool_) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

auto Parser::parse_storage_class_specifier(SpecifierAST*& yyast,
                                           DeclSpecs& specs) -> bool {
  SourceLocation loc;

  if (match(TokenKind::T_STATIC, loc)) {
    auto ast = new (pool_) StaticSpecifierAST();
    yyast = ast;
    ast->staticLoc = loc;
    specs.isStatic = true;
    return true;
  }
  if (match(TokenKind::T_THREAD_LOCAL, loc)) {
    auto ast = new (pool_) ThreadLocalSpecifierAST();
    yyast = ast;
    ast->threadLocalLoc = loc;
    specs.isThreadLocal = true;
    return true;
  }
  if (match(TokenKind::T_EXTERN, loc)) {
    auto ast = new (pool_) ExternSpecifierAST();
    yyast = ast;
    ast->externLoc = loc;
    specs.isExtern = true;
    return true;
  }
  if (match(TokenKind::T_MUTABLE, loc)) {
    auto ast = new (pool_) MutableSpecifierAST();
    yyast = ast;
    ast->mutableLoc = loc;
    specs.isMutable = true;
    return true;
  }
  if (match(TokenKind::T___THREAD, loc)) {
    auto ast = new (pool_) ThreadSpecifierAST();
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
    auto ast = new (pool_) VirtualSpecifierAST();
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

  auto ast = new (pool_) ExplicitSpecifierAST();
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

auto Parser::parse_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (parse_simple_type_specifier(yyast, specs)) {
    return true;
  } else if (parse_cv_qualifier(yyast, specs)) {
    return true;
  } else if (parse_elaborated_type_specifier(yyast, specs)) {
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
  if (!parse_type_specifier(typeSpecifier, specs)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  *it = new (pool_) List(typeSpecifier);
  it = &(*it)->next;

  typeSpecifier = nullptr;

  LoopParser loop{this};

  while (LA()) {
    loop.start();

    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_type_specifier(typeSpecifier, specs)) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeSpecifierAST*>* attributes = nullptr;
    parse_optional_attribute_specifier_seq(attributes);

    *it = new (pool_) List(typeSpecifier);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_defining_type_specifier(SpecifierAST*& yyast,
                                           DeclSpecs& specs) -> bool {
  if (!specs.no_class_or_enum_specs && !specs.typeSpecifier) {
    LookaheadParser lookahead{this};

    if (parse_enum_specifier(yyast, specs)) {
      lookahead.commit();

      specs.setTypeSpecifier(yyast);

      return true;
    }

    if (ClassSpecifierAST* classSpecifier = nullptr;
        parse_class_specifier(classSpecifier, specs)) {
      lookahead.commit();

      specs.setTypeSpecifier(classSpecifier);
      yyast = classSpecifier;

      return true;
    }
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

  *it = new (pool_) List(typeSpecifier);
  it = &(*it)->next;

  LoopParser loop{this};

  while (LA()) {
    loop.start();

    const auto before_type_specifier = currentLocation();

    typeSpecifier = nullptr;

    if (!parse_defining_type_specifier(typeSpecifier, specs)) {
      rewind(before_type_specifier);
      break;
    }

    List<AttributeSpecifierAST*>* attributes = nullptr;
    parse_optional_attribute_specifier_seq(attributes);

    *it = new (pool_) List(typeSpecifier);
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
    auto ast = new (pool_) SizeTypeSpecifierAST();
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
    auto ast = new (pool_) SizeTypeSpecifierAST();
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
    auto ast = new (pool_) SignTypeSpecifierAST();
    yyast = ast;
    ast->specifierLoc = specifierLoc;
    ast->specifier = unit->tokenKind(specifierLoc);

    specs.isUnsigned = true;

    return true;
  }

  if (SourceLocation specifierLoc; match(TokenKind::T_SIGNED, specifierLoc)) {
    auto ast = new (pool_) SignTypeSpecifierAST();
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
  auto ast = new (pool_) ComplexTypeSpecifierAST();
  yyast = ast;
  ast->complexLoc = consumeToken();
  specs.isComplex = true;

  return true;
}

auto Parser::parse_named_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (specs.isUnsigned || specs.isSigned || specs.isShort || specs.isLong)
    return false;

  LookaheadParser lookahead{this};

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  UnqualifiedIdAST* unqualifiedId = nullptr;

  if (isTemplateIntroduced) {
    if (SimpleTemplateIdAST* templateId = nullptr;
        parse_simple_template_id(templateId, isTemplateIntroduced)) {
      unqualifiedId = templateId;
    } else {
      parse_error("expected a template id");
    }
  } else if (!parse_type_name(unqualifiedId)) {
    return false;
  }

  if (config_.checkTypes) {
    if (!unqualifiedId) return false;
  }

  auto name = convertName(unqualifiedId);
  Symbol* symbol = nullptr;

  if (!nestedNameSpecifier) {
    symbol = unqualifiedLookup(name);
  } else {
    if (!nestedNameSpecifier->symbol && config_.checkTypes) {
      parse_error(nestedNameSpecifier->firstSourceLocation(),
                  cxx::format("expected class or namespace"));
    } else {
      symbol = qualifiedLookup(nestedNameSpecifier->symbol, name);
    }
  }

  auto is_type = [](const Symbol* symbol) -> bool {
    if (!symbol) return false;
    switch (symbol->kind()) {
      case SymbolKind::kTypeParameter:
      case SymbolKind::kTypeAlias:
      case SymbolKind::kClass:
      case SymbolKind::kEnum:
      case SymbolKind::kScopedEnum:
        return true;
      default:
        return false;
    }  // switch
  };

  if (config_.checkTypes) {
    if (!is_type(symbol)) return false;
  }

  if (symbol) specs.type = symbol->type();

  lookahead.commit();

  auto ast = new (pool_) NamedTypeSpecifierAST();
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->unqualifiedId = unqualifiedId;
  ast->isTemplateIntroduced = isTemplateIntroduced;

  if (!specs.type) {
    specs.type = control_->getUnresolvedNameType(unit, nestedNameSpecifier,
                                                 unqualifiedId);
  }

  return true;
}

auto Parser::parse_decltype_specifier_type_specifier(SpecifierAST*& yyast,
                                                     DeclSpecs& specs) -> bool {
  DecltypeSpecifierAST* decltypeSpecifier = nullptr;
  if (!parse_decltype_specifier(decltypeSpecifier)) return false;

  specs.setTypeSpecifier(decltypeSpecifier);

  yyast = decltypeSpecifier;

  specs.type = decltypeSpecifier->type;

  return true;
}

auto Parser::parse_underlying_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  SourceLocation underlyingTypeLoc;
  if (!match(TokenKind::T___UNDERLYING_TYPE, underlyingTypeLoc)) return false;

  auto ast = new (pool_) UnderlyingTypeSpecifierAST();
  yyast = ast;

  ast->underlyingTypeLoc = underlyingTypeLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!parse_type_id(ast->typeId)) parse_error("expected type id");

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  if (ast->typeId) {
    if (auto enumType = type_cast<EnumType>(ast->typeId->type)) {
      specs.type = enumType->underlyingType();
    } else if (auto scopedEnumType =
                   type_cast<ScopedEnumType>(ast->typeId->type)) {
      specs.type = scopedEnumType->underlyingType();
    } else {
      specs.type = control_->getUnresolvedUnderlyingType(unit, ast->typeId);
    }
  }

  return true;
}

auto Parser::parse_atomic_type_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  SourceLocation atomicLoc;
  if (!match(TokenKind::T__ATOMIC, atomicLoc)) return false;

  auto ast = new (pool_) AtomicTypeSpecifierAST();
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
    auto ast = new (pool_) IntegralTypeSpecifierAST();
    yyast = ast;
    ast->specifierLoc = consumeToken();
    ast->specifier = unit->tokenKind(ast->specifierLoc);
  };

  auto makeFloatingPointTypeSpecifier = [&] {
    auto ast = new (pool_) FloatingPointTypeSpecifierAST();
    yyast = ast;
    ast->specifierLoc = consumeToken();
    ast->specifier = unit->tokenKind(ast->specifierLoc);
  };

  switch (auto tk = LA(); tk.kind()) {
    case TokenKind::T___BUILTIN_VA_LIST: {
      auto ast = new (pool_) VaListTypeSpecifierAST();
      yyast = ast;
      ast->specifierLoc = consumeToken();
      ast->specifier = unit->tokenKind(ast->specifierLoc);

      return true;
    };

    case TokenKind::T_CHAR:
      makeIntegralTypeSpecifier();
      specs.type = control_->getCharType();
      return true;

    case TokenKind::T_CHAR8_T:
      makeIntegralTypeSpecifier();
      specs.type = control_->getChar8Type();
      return true;

    case TokenKind::T_CHAR16_T:
      makeIntegralTypeSpecifier();
      specs.type = control_->getChar16Type();
      return true;

    case TokenKind::T_CHAR32_T:
      makeIntegralTypeSpecifier();
      specs.type = control_->getChar32Type();
      return true;

    case TokenKind::T_WCHAR_T:
      makeIntegralTypeSpecifier();
      specs.type = control_->getWideCharType();
      return true;

    case TokenKind::T_BOOL:
      makeIntegralTypeSpecifier();
      specs.type = control_->getBoolType();
      return true;

    case TokenKind::T_INT:
      makeIntegralTypeSpecifier();
      specs.type = control_->getIntType();
      return true;

    case TokenKind::T___INT64:
      makeIntegralTypeSpecifier();
      return true;

    case TokenKind::T___INT128:
      makeIntegralTypeSpecifier();
      return true;

    case TokenKind::T_FLOAT:
      makeFloatingPointTypeSpecifier();
      specs.type = control_->getFloatType();
      return true;

    case TokenKind::T_DOUBLE:
      makeFloatingPointTypeSpecifier();
      specs.type = control_->getDoubleType();
      return true;

    case TokenKind::T___FLOAT80:
      makeFloatingPointTypeSpecifier();
      return true;

    case TokenKind::T___FLOAT128:
      makeFloatingPointTypeSpecifier();
      return true;

    case TokenKind::T_VOID: {
      auto ast = new (pool_) VoidTypeSpecifierAST();
      yyast = ast;
      ast->voidLoc = consumeToken();
      specs.type = control_->getVoidType();

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
  SourceLocation typeTraitsLoc;
  BuiltinKind builtinKind = BuiltinKind::T_IDENTIFIER;
  if (!parse_type_traits_op(typeTraitsLoc, builtinKind)) return;

#if false
  parse_warn(
      typeTraitsLoc,
      cxx::format("keyword '{}' will be made available as an identifier for "
                  "the remainder of the translation unit",
                  Token::spell(builtinKind)));
#endif

  unit->replaceWithIdentifier(typeTraitsLoc);

  rewind(typeTraitsLoc);
}

auto Parser::lvalue_to_rvalue_conversion(ExpressionAST*& expr) -> bool {
  if (!is_glvalue(expr)) return false;

  auto unref = control_->remove_cvref(expr->type);
  if (control_->is_function(unref)) return false;
  if (control_->is_array(unref)) return false;
  if (!control_->is_complete(unref)) return false;
  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kLValueToRValueConversion;
  cast->expression = expr;
  cast->type = control_->remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto Parser::array_to_pointer_conversion(ExpressionAST*& expr) -> bool {
  auto unref = control_->remove_reference(expr->type);
  if (!control_->is_array(unref)) return false;
  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kArrayToPointerConversion;
  cast->expression = expr;
  cast->type = control_->add_pointer(control_->remove_extent(unref));
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto Parser::function_to_pointer_conversion(ExpressionAST*& expr) -> bool {
  auto unref = control_->remove_reference(expr->type);
  if (!control_->is_function(unref)) return false;
  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kFunctionToPointerConversion;
  cast->expression = expr;
  cast->type = control_->add_pointer(unref);
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;
  return true;
}

auto Parser::integral_promotion(ExpressionAST*& expr) -> bool {
  if (!is_prvalue(expr)) return false;

  auto ty = control_->remove_cv(expr->type);

  if (!control_->is_integral(ty) && !control_->is_enum(ty)) return false;

  auto make_implicit_cast = [&](const Type* type) {
    auto cast = new (pool_) ImplicitCastExpressionAST();
    cast->castKind = ImplicitCastKind::kIntegralPromotion;
    cast->expression = expr;
    cast->type = type;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  // TODO: bit-fields

  switch (ty->kind()) {
    case TypeKind::kChar:
    case TypeKind::kSignedChar:
    case TypeKind::kUnsignedChar:
    case TypeKind::kShortInt:
    case TypeKind::kUnsignedShortInt: {
      make_implicit_cast(control_->getIntType());
      return true;
    }

    case TypeKind::kChar8:
    case TypeKind::kChar16:
    case TypeKind::kChar32:
    case TypeKind::kWideChar: {
      make_implicit_cast(control_->getIntType());
      return true;
    }

    case TypeKind::kBool: {
      make_implicit_cast(control_->getIntType());
      return true;
    }

    default:
      break;
  }  // switch

  if (auto enumType = type_cast<EnumType>(ty)) {
    auto type = enumType->underlyingType();

    if (!type) {
      // TODO: compute the from the value of the enumeration
      type = control_->getIntType();
    }

    make_implicit_cast(type);

    return true;
  }

  return false;
}

auto Parser::floating_point_promotion(ExpressionAST*& expr) -> bool {
  if (!is_prvalue(expr)) return false;

  auto ty = control_->remove_cv(expr->type);

  if (!control_->is_floating_point(ty)) return false;

  if (ty->kind() != TypeKind::kFloat) return false;

  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kFloatingPointPromotion;
  cast->expression = expr;
  cast->type = control_->getDoubleType();
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto Parser::integral_conversion(ExpressionAST*& expr,
                                 const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control_->is_integral_or_unscoped_enum(expr->type)) return false;
  if (!control_->is_integer(destinationType)) return false;

  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kIntegralConversion;
  cast->expression = expr;
  cast->type = destinationType;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto Parser::floating_point_conversion(ExpressionAST*& expr,
                                       const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control_->is_floating_point(expr->type)) return false;
  if (!control_->is_floating_point(destinationType)) return false;

  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kFloatingPointConversion;
  cast->expression = expr;
  cast->type = destinationType;
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto Parser::floating_integral_conversion(ExpressionAST*& expr,
                                          const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  auto make_integral_conversion = [&] {
    auto cast = new (pool_) ImplicitCastExpressionAST();
    cast->castKind = ImplicitCastKind::kFloatingIntegralConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  if (control_->is_integral_or_unscoped_enum(expr->type) &&
      control_->is_floating_point(destinationType)) {
    make_integral_conversion();
    return true;
  }

  if (!control_->is_floating_point(expr->type)) return false;
  if (!control_->is_integer(destinationType)) return false;

  make_integral_conversion();

  return true;
}

auto Parser::pointer_to_member_conversion(ExpressionAST*& expr,
                                          const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  if (!control_->is_member_pointer(destinationType)) return false;

  auto make_implicit_cast = [&] {
    auto cast = new (pool_) ImplicitCastExpressionAST();
    cast->castKind = ImplicitCastKind::kPointerToMemberConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  auto can_convert_null_pointer_constant = [&] {
    if (!is_null_pointer_constant(expr)) return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_member_object_pointer = [&] {
    auto memberObjectPointerType =
        type_cast<MemberObjectPointerType>(expr->type);

    if (!memberObjectPointerType) return false;

    auto destinationMemberObjectPointerType =
        type_cast<MemberObjectPointerType>(destinationType);

    if (!destinationMemberObjectPointerType) return false;

    if (control_->get_cv_qualifiers(memberObjectPointerType->elementType()) !=
        control_->get_cv_qualifiers(
            destinationMemberObjectPointerType->elementType()))
      return false;

    if (!control_->is_base_of(destinationMemberObjectPointerType->classType(),
                              memberObjectPointerType->classType()))
      return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_member_function_pointer = [&] {
    auto memberFunctionPointerType =
        type_cast<MemberFunctionPointerType>(expr->type);

    if (!memberFunctionPointerType) return false;

    auto destinationMemberFunctionPointerType =
        type_cast<MemberFunctionPointerType>(destinationType);

    if (!destinationMemberFunctionPointerType) return false;

    if (control_->get_cv_qualifiers(
            memberFunctionPointerType->functionType()) !=
        control_->get_cv_qualifiers(
            destinationMemberFunctionPointerType->functionType()))
      return false;

    if (!control_->is_base_of(destinationMemberFunctionPointerType->classType(),
                              memberFunctionPointerType->classType()))
      return false;

    make_implicit_cast();

    return true;
  };

  if (can_convert_null_pointer_constant()) return true;
  if (can_convert_member_object_pointer()) return true;

  return false;
}

auto Parser::pointer_conversion(ExpressionAST*& expr,
                                const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  auto make_implicit_cast = [&] {
    auto cast = new (pool_) ImplicitCastExpressionAST();
    cast->castKind = ImplicitCastKind::kPointerConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;
  };

  auto can_convert_null_pointer_literal = [&] {
    if (!is_null_pointer_constant(expr)) return false;

    if (!control_->is_pointer(destinationType) &&
        !control_->is_null_pointer(destinationType))
      return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_to_void_pointer = [&] {
    const auto pointerType = type_cast<PointerType>(expr->type);
    if (!pointerType) return false;

    const auto destinationPointerType = type_cast<PointerType>(destinationType);
    if (!destinationPointerType) return false;

    if (control_->get_cv_qualifiers(pointerType->elementType()) !=
        control_->get_cv_qualifiers(destinationPointerType->elementType()))
      return false;

    if (!control_->is_void(
            control_->remove_cv(destinationPointerType->elementType())))
      return false;

    make_implicit_cast();

    return true;
  };

  auto can_convert_to_base_class_pointer = [&] {
    const auto pointerType = type_cast<PointerType>(expr->type);
    if (!pointerType) return false;

    const auto destinationPointerType = type_cast<PointerType>(destinationType);
    if (!destinationPointerType) return false;

    if (control_->get_cv_qualifiers(pointerType->elementType()) !=
        control_->get_cv_qualifiers(destinationPointerType->elementType()))
      return false;

    if (!control_->is_base_of(
            control_->remove_cv(destinationPointerType->elementType()),
            control_->remove_cv(pointerType->elementType())))
      return false;

    make_implicit_cast();

    return true;
  };

  if (can_convert_null_pointer_literal()) return true;
  if (can_convert_to_void_pointer()) return true;
  if (can_convert_to_base_class_pointer()) return true;

  return false;
}

auto Parser::function_pointer_conversion(ExpressionAST*& expr,
                                         const Type* destinationType) -> bool {
  if (!is_prvalue(expr)) return false;

  auto can_remove_noexcept_from_function = [&] {
    const auto pointerType = type_cast<PointerType>(expr->type);
    if (!pointerType) return false;

    const auto functionType =
        type_cast<FunctionType>(pointerType->elementType());

    if (!functionType) return false;

    if (functionType->isNoexcept()) return false;

    const auto destinationPointerType = type_cast<PointerType>(destinationType);
    if (!destinationPointerType) return false;

    const auto destinationFunctionType =
        type_cast<FunctionType>(destinationPointerType->elementType());

    if (!destinationFunctionType) return false;

    if (!control_->is_same(control_->remove_noexcept(functionType),
                           destinationFunctionType))
      return false;

    auto cast = new (pool_) ImplicitCastExpressionAST();
    cast->castKind = ImplicitCastKind::kFunctionPointerConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;

    return true;
  };

  auto can_remove_noexcept_from_member_function__pointer = [&] {
    const auto memberFunctionPointer =
        type_cast<MemberFunctionPointerType>(expr->type);

    if (!memberFunctionPointer) return false;

    if (!memberFunctionPointer->functionType()->isNoexcept()) return false;

    const auto destinationMemberFunctionPointer =
        type_cast<MemberFunctionPointerType>(destinationType);

    if (!destinationMemberFunctionPointer) return false;

    if (destinationMemberFunctionPointer->functionType()->isNoexcept())
      return false;

    if (!control_->is_same(
            control_->remove_noexcept(memberFunctionPointer->functionType()),
            destinationMemberFunctionPointer->functionType()))
      return false;

    auto cast = new (pool_) ImplicitCastExpressionAST();
    cast->castKind = ImplicitCastKind::kFunctionPointerConversion;
    cast->expression = expr;
    cast->type = destinationType;
    cast->valueCategory = ValueCategory::kPrValue;
    expr = cast;

    return true;
  };

  if (can_remove_noexcept_from_function()) return true;
  if (can_remove_noexcept_from_member_function__pointer()) return true;

  return false;
}

auto Parser::boolean_conversion(ExpressionAST*& expr,
                                const Type* destinationType) -> bool {
  if (!type_cast<BoolType>(control_->remove_cv(destinationType))) return false;

  if (!is_prvalue(expr)) return false;

  if (!control_->is_arithmetic_or_unscoped_enum(expr->type) &&
      !control_->is_pointer(expr->type) &&
      !control_->is_member_pointer(expr->type))
    return false;

  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kBooleanConversion;
  cast->expression = expr;
  cast->type = control_->getBoolType();
  cast->valueCategory = ValueCategory::kPrValue;
  expr = cast;

  return true;
}

auto Parser::temporary_materialization_conversion(ExpressionAST*& expr)
    -> bool {
  if (!is_prvalue(expr)) return false;

  auto cast = new (pool_) ImplicitCastExpressionAST();
  cast->castKind = ImplicitCastKind::kTemporaryMaterializationConversion;
  cast->expression = expr;
  cast->type = control_->remove_reference(expr->type);
  cast->valueCategory = ValueCategory::kXValue;
  expr = cast;

  return true;
}

auto Parser::qualification_conversion(ExpressionAST*& expr,
                                      const Type* destinationType) -> bool {
  return false;
}

auto Parser::implicit_conversion(ExpressionAST*& expr,
                                 const Type* destinationType) -> bool {
  if (!expr || !expr->type) return false;
  if (!destinationType) return false;

  auto savedExpr = expr;
  auto didConvert = false;

  if (lvalue_to_rvalue_conversion(expr)) {
    didConvert = true;
  } else if (array_to_pointer_conversion(expr)) {
    didConvert = true;
  } else if (function_to_pointer_conversion(expr)) {
    didConvert = true;
  }

  if (integral_promotion(expr)) return true;
  if (floating_point_promotion(expr)) return true;
  if (integral_conversion(expr, destinationType)) return true;
  if (floating_point_conversion(expr, destinationType)) return true;
  if (floating_integral_conversion(expr, destinationType)) return true;
  if (pointer_conversion(expr, destinationType)) return true;
  if (pointer_to_member_conversion(expr, destinationType)) return true;
  if (boolean_conversion(expr, destinationType)) return true;
  if (function_pointer_conversion(expr, destinationType)) return true;
  if (qualification_conversion(expr, destinationType)) return true;

  if (didConvert) return true;

  expr = savedExpr;

  return false;
}

auto Parser::usual_arithmetic_conversion(ExpressionAST*& expr,
                                         ExpressionAST*& other) -> const Type* {
  if (!expr || !expr->type) return nullptr;
  if (!other || !other->type) return nullptr;

  ExpressionAST* savedExpr = expr;
  ExpressionAST* savedOther = other;

  auto unmodifiedExpressions = [&]() -> const Type* {
    expr = savedExpr;
    other = savedOther;
    return nullptr;
  };

  if (!control_->is_arithmetic_or_unscoped_enum(expr->type) &&
      !control_->is_arithmetic_or_unscoped_enum(other->type))
    return unmodifiedExpressions();

  (void)lvalue_to_rvalue_conversion(expr);
  (void)lvalue_to_rvalue_conversion(other);

  if (control_->is_scoped_enum(expr->type) ||
      control_->is_scoped_enum(other->type))
    return unmodifiedExpressions();

  if (control_->is_floating_point(expr->type) ||
      control_->is_floating_point(other->type)) {
    auto leftType = control_->remove_cv(expr->type);
    auto rightType = control_->remove_cv(other->type);

    if (control_->is_same(leftType, rightType)) return leftType;

    if (!control_->is_floating_point(leftType)) {
      if (floating_integral_conversion(expr, rightType)) return rightType;
      return unmodifiedExpressions();
    } else if (!control_->is_floating_point(rightType)) {
      if (floating_integral_conversion(other, leftType)) return leftType;
      return unmodifiedExpressions();
    } else if (leftType->kind() == TypeKind::kLongDouble ||
               rightType->kind() == TypeKind::kLongDouble) {
      (void)floating_point_conversion(expr, control_->getLongDoubleType());
      return control_->getLongDoubleType();
    } else if (leftType->kind() == TypeKind::kDouble ||
               rightType->kind() == TypeKind::kDouble) {
      (void)floating_point_conversion(expr, control_->getDoubleType());
      return control_->getDoubleType();
    }

    return unmodifiedExpressions();
  }

  (void)integral_promotion(expr);
  (void)integral_promotion(other);

  const auto leftType = control_->remove_cv(expr->type);
  const auto rightType = control_->remove_cv(other->type);

  if (control_->is_same(leftType, rightType)) return leftType;

  auto match_integral_type = [&](const Type* type) -> bool {
    if (leftType->kind() == type->kind() || rightType->kind() == type->kind()) {
      (void)integral_conversion(expr, type);
      (void)integral_conversion(other, type);
      return true;
    }
    return false;
  };

  if (control_->is_signed(leftType) && control_->is_signed(rightType)) {
    if (match_integral_type(control_->getLongLongIntType())) {
      return control_->getLongLongIntType();
    } else if (match_integral_type(control_->getLongIntType())) {
      return control_->getLongIntType();
    } else {
      (void)integral_conversion(expr, control_->getIntType());
      (void)integral_conversion(other, control_->getIntType());
      return control_->getIntType();
    }
  }

  if (control_->is_unsigned(leftType) && control_->is_unsigned(rightType)) {
    if (match_integral_type(control_->getUnsignedLongLongIntType())) {
      return control_->getUnsignedLongLongIntType();
    } else if (match_integral_type(control_->getUnsignedLongIntType())) {
      return control_->getUnsignedLongIntType();
    } else {
      (void)integral_conversion(expr, control_->getUnsignedIntType());
      return control_->getUnsignedIntType();
    }
  }

  if (match_integral_type(control_->getUnsignedLongLongIntType())) {
    return control_->getUnsignedLongLongIntType();
  } else if (match_integral_type(control_->getUnsignedLongIntType())) {
    return control_->getUnsignedLongIntType();
  } else if (match_integral_type(control_->getUnsignedIntType())) {
    return control_->getUnsignedIntType();
  } else if (match_integral_type(control_->getUnsignedShortIntType())) {
    return control_->getUnsignedShortIntType();
  } else if (match_integral_type(control_->getUnsignedCharType())) {
    return control_->getUnsignedCharType();
  } else if (match_integral_type(control_->getLongLongIntType())) {
    return control_->getLongLongIntType();
  } else if (match_integral_type(control_->getLongIntType())) {
    return control_->getLongIntType();
  }

  (void)integral_conversion(expr, control_->getIntType());
  (void)integral_conversion(other, control_->getIntType());
  return control_->getIntType();
}

auto Parser::is_null_pointer_constant(ExpressionAST* expr) const -> bool {
  if (control_->is_null_pointer(expr->type)) return true;
  if (auto integerLiteral = ast_cast<IntLiteralExpressionAST>(expr)) {
    return integerLiteral->literal->value() == "0";
  }
  return false;
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

auto Parser::is_constructor(Symbol* symbol) const -> bool {
  auto functionSymbol = symbol_cast<FunctionSymbol>(symbol);
  if (!functionSymbol) return false;
  if (!functionSymbol->enclosingScope()) return false;
  auto classSymbol =
      symbol_cast<ClassSymbol>(functionSymbol->enclosingScope()->owner());
  if (!classSymbol) return false;
  if (classSymbol->name() != functionSymbol->name()) return false;
  return true;
}

auto Parser::evaluate_constant_expression(ExpressionAST* expr)
    -> std::optional<ConstValue> {
  ConstExpressionEvaluator sem{*this};
  return sem.evaluate(expr);
}

auto Parser::parse_elaborated_type_specifier(SpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  if (specs.typeSpecifier) return false;

  if (!LA().isOneOf(TokenKind::T_ENUM, TokenKind::T_CLASS, TokenKind::T_STRUCT,
                    TokenKind::T_UNION))
    return false;

  const auto start = currentLocation();

  if (auto entry = elaborated_type_specifiers_.get(start)) {
    auto [cursor, ast, parsed, hit] = *entry;
    rewind(cursor);
    yyast = ast;

    return parsed;
  }

  ElaboratedTypeSpecifierAST* ast = nullptr;

  const auto parsed = parse_elaborated_type_specifier_helper(ast, specs);

  yyast = ast;

  elaborated_type_specifiers_.set(start, currentLocation(), ast, parsed);

  return parsed;
}

auto Parser::parse_elaborated_type_specifier_helper(
    ElaboratedTypeSpecifierAST*& yyast, DeclSpecs& specs) -> bool {
  if (parse_elaborated_enum_specifier(yyast, specs)) return true;

  SourceLocation classLoc;
  if (!parse_class_key(classLoc)) return false;

  List<AttributeSpecifierAST*>* attributes = nullptr;
  parse_optional_attribute_specifier_seq(attributes);

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  UnqualifiedIdAST* unqualifiedId = nullptr;
  if (!parse_simple_template_or_name_id(unqualifiedId, isTemplateIntroduced))
    parse_error("expected unqualified id");

  auto ast = new (pool_) ElaboratedTypeSpecifierAST();
  yyast = ast;

  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->templateLoc = templateLoc;
  ast->unqualifiedId = unqualifiedId;
  ast->classKey = unit->tokenKind(classLoc);
  ast->isTemplateIntroduced = isTemplateIntroduced;

  return true;
}

auto Parser::parse_elaborated_enum_specifier(ElaboratedTypeSpecifierAST*& yyast,
                                             DeclSpecs& specs) -> bool {
  SourceLocation enumLoc;
  if (!match(TokenKind::T_ENUM, enumLoc)) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  NameIdAST* name = nullptr;
  if (!parse_name_id(name)) {
    parse_error("expected a name");
  }

  auto ast = new (pool_) ElaboratedTypeSpecifierAST();
  yyast = ast;

  ast->classLoc = enumLoc;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = name;
  ast->classKey = TokenKind::T_ENUM;

  return true;
}

auto Parser::parse_decl_specifier_seq_no_typespecs(List<SpecifierAST*>*& yyast)
    -> bool {
  DeclSpecs specs{this};
  return parse_decl_specifier_seq_no_typespecs(yyast, specs);
}

auto Parser::parse_decltype_specifier(DecltypeSpecifierAST*& yyast) -> bool {
  SourceLocation decltypeLoc;
  if (!match(TokenKind::T_DECLTYPE, decltypeLoc)) return false;

  SourceLocation lparenLoc;
  if (!match(TokenKind::T_LPAREN, lparenLoc)) return false;

  if (lookat(TokenKind::T_AUTO)) return false;  // placeholder type specifier

  auto ast = new (pool_) DecltypeSpecifierAST();
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
    auto ast = new (pool_) AutoTypeSpecifierAST();
    yyast = ast;
    ast->autoLoc = autoLoc;

    specs.isAuto = true;
    specs.type = control_->getAutoType();
  } else {
    auto ast = new (pool_) DecltypeAutoSpecifierAST();
    yyast = ast;

    expect(TokenKind::T_DECLTYPE, ast->decltypeLoc);
    expect(TokenKind::T_LPAREN, ast->lparenLoc);
    expect(TokenKind::T_AUTO, ast->autoLoc);
    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    specs.isDecltypeAuto = true;
    specs.type = control_->getDecltypeAutoType();
  }

  if (typeConstraint) {
    auto ast = new (pool_) PlaceholderTypeSpecifierAST();

    ast->typeConstraint = typeConstraint;
    ast->specifier = yyast;

    yyast = ast;
  }

  return true;
}

auto Parser::parse_init_declarator(InitDeclaratorAST*& yyast,
                                   const DeclSpecs& specs) -> bool {
  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  if (!parse_declarator(declarator, decl)) return false;

  return parse_init_declarator(yyast, declarator, decl);
}

auto Parser::parse_init_declarator(InitDeclaratorAST*& yyast,
                                   DeclaratorAST* declarator, Decl& decl)
    -> bool {
  if (auto declId = decl.declaratorId; declId) {
    auto symbolType = GetDeclaratorType{this}(declarator, decl.specs.getType());
    const auto name = convertName(declId->unqualifiedId);
    if (name) {
      if (decl.specs.isTypedef) {
        auto symbol = control_->newTypeAliasSymbol(scope_);
        symbol->setName(name);
        symbol->setType(symbolType);
        std::invoke(DeclareSymbol{this, scope_}, symbol);
      } else if (getFunctionPrototype(declarator)) {
        auto functionSymbol = control_->newFunctionSymbol(scope_);
        applySpecifiers(functionSymbol, decl.specs);
        functionSymbol->setName(name);
        functionSymbol->setType(symbolType);
        std::invoke(DeclareSymbol{this, scope_}, functionSymbol);
      } else {
        auto symbol = control_->newVariableSymbol(scope_);
        applySpecifiers(symbol, decl.specs);
        symbol->setName(name);
        symbol->setType(symbolType);
        std::invoke(DeclareSymbol{this, scope_}, symbol);
      }
    }
  }
  RequiresClauseAST* requiresClause = nullptr;
  ExpressionAST* initializer = nullptr;

  LookaheadParser lookahead{this};
  if (parse_declarator_initializer(requiresClause, initializer)) {
    lookahead.commit();
  }

  auto ast = new (pool_) InitDeclaratorAST();
  yyast = ast;

  ast->declarator = declarator;
  ast->requiresClause = requiresClause;
  ast->initializer = initializer;

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

  List<DeclaratorChunkAST*>* declaratorChunkList = nullptr;
  auto it = &declaratorChunkList;

  while (LA().isOneOf(TokenKind::T_LPAREN, TokenKind::T_LBRACKET)) {
    if (ArrayDeclaratorChunkAST* arrayDeclaratorChunk = nullptr;
        parse_array_declarator(arrayDeclaratorChunk)) {
      *it = new (pool_) List<DeclaratorChunkAST*>(arrayDeclaratorChunk);
      it = &(*it)->next;
    } else if (FunctionDeclaratorChunkAST* functionDeclaratorChunk = nullptr;
               declaratorKind != DeclaratorKind::kNewDeclarator &&
               parse_function_declarator(functionDeclaratorChunk)) {
      *it = new (pool_) List<DeclaratorChunkAST*>(functionDeclaratorChunk);
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

  yyast = new (pool_) DeclaratorAST();
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

  *it = new (pool_) List(ptrOp);
  it = &(*it)->next;

  ptrOp = nullptr;

  while (parse_ptr_operator(ptrOp)) {
    *it = new (pool_) List(ptrOp);
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

  auto modifier = new (pool_) ArrayDeclaratorChunkAST();
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

  ScopeGuard scopeGuard{this};

  SourceLocation rparenLoc;
  ParameterDeclarationClauseAST* parameterDeclarationClause = nullptr;

  if (!match(TokenKind::T_RPAREN, rparenLoc)) {
    if (!parse_parameter_declaration_clause(parameterDeclarationClause)) {
      return false;
    }

    if (!match(TokenKind::T_RPAREN, rparenLoc)) return false;
  }

  lookahead.commit();

  auto ast = new (pool_) FunctionDeclaratorChunkAST();
  yyast = ast;

  ast->lparenLoc = lparenLoc;
  ast->parameterDeclarationClause = parameterDeclarationClause;
  ast->rparenLoc = rparenLoc;

  DeclSpecs cvQualifiers{this};

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

  *it = new (pool_) List(specifier);
  it = &(*it)->next;

  specifier = nullptr;

  while (parse_cv_qualifier(specifier, declSpecs)) {
    *it = new (pool_) List(specifier);
    it = &(*it)->next;

    specifier = nullptr;
  }

  return true;
}

auto Parser::parse_trailing_return_type(TrailingReturnTypeAST*& yyast) -> bool {
  SourceLocation minusGreaterLoc;

  if (!match(TokenKind::T_MINUS_GREATER, minusGreaterLoc)) return false;

  auto ast = new (pool_) TrailingReturnTypeAST();
  yyast = ast;

  ast->minusGreaterLoc = minusGreaterLoc;

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  return true;
}

auto Parser::parse_ptr_operator(PtrOperatorAST*& yyast) -> bool {
  if (SourceLocation starLoc; match(TokenKind::T_STAR, starLoc)) {
    auto ast = new (pool_) PointerOperatorAST();
    yyast = ast;

    ast->starLoc = starLoc;

    parse_optional_attribute_specifier_seq(ast->attributeList);

    DeclSpecs cvQualifiers{this};
    (void)parse_cv_qualifier_seq(ast->cvQualifierList, cvQualifiers);

    return true;
  } else if (SourceLocation refLoc; parse_ref_qualifier(refLoc)) {
    auto ast = new (pool_) ReferenceOperatorAST();
    yyast = ast;

    ast->refLoc = refLoc;
    ast->refOp = unit->tokenKind(refLoc);

    parse_optional_attribute_specifier_seq(ast->attributeList);

    return true;
  }

  LookaheadParser lookahead{this};

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

  SourceLocation starLoc;
  if (!match(TokenKind::T_STAR, starLoc)) return false;

  lookahead.commit();

  auto ast = new (pool_) PtrToMemberOperatorAST();
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->starLoc = starLoc;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  DeclSpecs cvQualifiers{this};
  (void)parse_cv_qualifier_seq(ast->cvQualifierList, cvQualifiers);

  return true;
}

auto Parser::parse_cv_qualifier(SpecifierAST*& yyast, DeclSpecs& declSpecs)
    -> bool {
  SourceLocation loc;

  if (match(TokenKind::T_CONST, loc)) {
    auto ast = new (pool_) ConstQualifierAST();
    yyast = ast;
    ast->constLoc = loc;
    declSpecs.isConst = true;
    return true;
  }
  if (match(TokenKind::T_VOLATILE, loc)) {
    auto ast = new (pool_) VolatileQualifierAST();
    yyast = ast;
    ast->volatileLoc = loc;
    declSpecs.isVolatile = true;
    return true;
  }
  if (match(TokenKind::T___RESTRICT__, loc)) {
    auto ast = new (pool_) RestrictQualifierAST();
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

    auto ast = new (pool_) ParameterPackAST();
    ast->ellipsisLoc = ellipsisLoc;
    yyast = ast;

    return true;
  }

  if (declaratorKind != DeclaratorKind::kDeclarator) return false;

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  check_type_traits();

  UnqualifiedIdAST* unqualifiedId = nullptr;
  if (!parse_unqualified_id(unqualifiedId, isTemplateIntroduced,
                            /*inRequiresClause*/ false))
    return false;

  lookahead.commit();

  auto ast = new (pool_) IdDeclaratorAST();
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
    auto ast = new (pool_) ParameterPackAST();
    ast->ellipsisLoc = ellipsisLoc;
    ast->coreDeclarator = yyast;
    yyast = ast;
  }

  return true;
}

auto Parser::parse_type_id(TypeIdAST*& yyast) -> bool {
  List<SpecifierAST*>* specifierList = nullptr;
  DeclSpecs specs{this};
  if (!parse_type_specifier_seq(specifierList, specs)) return false;

  yyast = new (pool_) TypeIdAST();
  yyast->typeSpecifierList = specifierList;

  Decl decl{specs};
  parse_optional_abstract_declarator(yyast->declarator, decl);

  yyast->type =
      GetDeclaratorType{this}(yyast->declarator, decl.specs.getType());

  return true;
}

auto Parser::parse_defining_type_id(
    TypeIdAST*& yyast,
    const std::vector<TemplateDeclarationAST*>& templateDeclarations) -> bool {
  DeclSpecs specs{this};

  if (!templateDeclarations.empty()) specs.no_class_or_enum_specs = true;

  List<SpecifierAST*>* typeSpecifierList = nullptr;

  if (!parse_defining_type_specifier_seq(typeSpecifierList, specs)) {
    return false;
  }

  DeclaratorAST* declarator = nullptr;
  Decl decl{specs};
  parse_optional_abstract_declarator(declarator, decl);

  auto ast = new (pool_) TypeIdAST();
  yyast = ast;

  ast->typeSpecifierList = typeSpecifierList;
  ast->declarator = declarator;
  ast->type = GetDeclaratorType{this}(ast->declarator, decl.specs.getType());

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

  auto ast = new (pool_) NestedDeclaratorAST();
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

  ScopeGuard scopeGuard{this};

  bool parsed = false;

  SourceLocation ellipsisLoc;
  FunctionParametersSymbol* functionParametersSymbol = nullptr;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    parsed = true;

    auto ast = new (pool_) ParameterDeclarationClauseAST();
    yyast = ast;

    ast->ellipsisLoc = ellipsisLoc;
    ast->isVariadic = true;
    ast->functionParametersSymbol =
        control_->newFunctionParametersSymbol(scope_);
  } else if (List<ParameterDeclarationAST*>* parameterDeclarationList = nullptr;
             parse_parameter_declaration_list(parameterDeclarationList,
                                              functionParametersSymbol)) {
    parsed = true;

    auto ast = new (pool_) ParameterDeclarationClauseAST();
    yyast = ast;
    ast->parameterDeclarationList = parameterDeclarationList;
    match(TokenKind::T_COMMA, ast->commaLoc);
    ast->isVariadic = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);
    ast->functionParametersSymbol = functionParametersSymbol;
  } else {
    parsed = false;
  }

  parameter_declaration_clauses_.set(start, currentLocation(), yyast, parsed);

  return parsed;
}

auto Parser::parse_parameter_declaration_list(
    List<ParameterDeclarationAST*>*& yyast,
    FunctionParametersSymbol*& functionParametersSymbol) -> bool {
  auto it = &yyast;

  ScopeGuard scopeGuard{this};

  functionParametersSymbol = control_->newFunctionParametersSymbol(scope_);

  scope_ = functionParametersSymbol->scope();

  ParameterDeclarationAST* declaration = nullptr;

  if (!parse_parameter_declaration(declaration, /*templParam*/ false)) {
    return false;
  }

  *it = new (pool_) List(declaration);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    ParameterDeclarationAST* declaration = nullptr;

    if (!parse_parameter_declaration(declaration, /*templParam*/ false)) {
      rewind(commaLoc);
      break;
    }

    *it = new (pool_) List(declaration);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_parameter_declaration(ParameterDeclarationAST*& yyast,
                                         bool templParam) -> bool {
  auto ast = new (pool_) ParameterDeclarationAST();
  yyast = ast;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  DeclSpecs specs{this};

  specs.no_class_or_enum_specs = true;

  ast->isThisIntroduced = match(TokenKind::T_THIS, ast->thisLoc);

  if (!parse_decl_specifier_seq(ast->typeSpecifierList, specs)) return false;

  Decl decl{specs};
  parse_optional_declarator_or_abstract_declarator(ast->declarator, decl);

  ast->type = GetDeclaratorType{this}(ast->declarator, decl.specs.getType());
  ast->isPack = decl.isPack;

  if (auto declId = decl.declaratorId; declId && declId->unqualifiedId) {
    auto paramName = convertName(declId->unqualifiedId);
    if (auto identifier = name_cast<Identifier>(paramName)) {
      ast->identifier = identifier;
    } else {
      parse_error(declId->unqualifiedId->firstSourceLocation(),
                  "expected an identifier");
    }
  }

  if (!templParam) {
    auto parameterSymbol = control_->newParameterSymbol(scope_);
    parameterSymbol->setName(ast->identifier);
    parameterSymbol->setType(ast->type);
    std::invoke(DeclareSymbol{this, scope_}, parameterSymbol);
  }

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

    auto ast = new (pool_) ParenInitializerAST();
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

  auto ast = new (pool_) EqualInitializerAST();
  yyast = ast;

  ast->equalLoc = equalLoc;

  if (!parse_initializer_clause(ast->expression, ExprContext{})) {
    parse_error("expected an intializer");
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

auto Parser::parse_braced_init_list(BracedInitListAST*& yyast,
                                    const ExprContext& ctx) -> bool {
  SourceLocation lbraceLoc;
  SourceLocation rbraceLoc;

  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  if (lookat(TokenKind::T_DOT)) {
    auto ast = new (pool_) BracedInitListAST();
    yyast = ast;

    ast->lbraceLoc = lbraceLoc;

    auto it = &ast->expressionList;

    DesignatedInitializerClauseAST* designatedInitializerClause = nullptr;

    if (!parse_designated_initializer_clause(designatedInitializerClause)) {
      parse_error("expected designated initializer clause");
    }

    if (designatedInitializerClause) {
      *it = new (pool_) List<ExpressionAST*>(designatedInitializerClause);
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
        *it = new (pool_) List<ExpressionAST*>(designatedInitializerClause);
        it = &(*it)->next;
      }
    }

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);

    return true;
  }

  SourceLocation commaLoc;

  if (match(TokenKind::T_COMMA, commaLoc)) {
    expect(TokenKind::T_RBRACE, rbraceLoc);

    auto ast = new (pool_) BracedInitListAST();
    yyast = ast;

    ast->lbraceLoc = lbraceLoc;
    ast->commaLoc = commaLoc;
    ast->rbraceLoc = rbraceLoc;

    return true;
  }

  List<ExpressionAST*>* expressionList = nullptr;

  if (!match(TokenKind::T_RBRACE, rbraceLoc)) {
    if (!parse_initializer_list(expressionList, ctx)) {
      parse_error("expected initializer list");
    }

    expect(TokenKind::T_RBRACE, rbraceLoc);
  }

  auto ast = new (pool_) BracedInitListAST();
  yyast = ast;

  ast->lbraceLoc = lbraceLoc;
  ast->expressionList = expressionList;
  ast->rbraceLoc = rbraceLoc;

  return true;
}

auto Parser::parse_initializer_list(List<ExpressionAST*>*& yyast,
                                    const ExprContext& ctx) -> bool {
  auto it = &yyast;

  ExpressionAST* expression = nullptr;

  if (!parse_initializer_clause(expression, ctx)) return false;

  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto pack = new (pool_) PackExpansionExpressionAST();
    pack->expression = expression;
    pack->ellipsisLoc = ellipsisLoc;
    expression = pack;
  }

  *it = new (pool_) List(expression);
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
      auto pack = new (pool_) PackExpansionExpressionAST();
      pack->expression = expression;
      pack->ellipsisLoc = ellipsisLoc;
      expression = pack;
    }

    *it = new (pool_) List(expression);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_designated_initializer_clause(
    DesignatedInitializerClauseAST*& yyast) -> bool {
  SourceLocation dotLoc;
  if (!match(TokenKind::T_DOT, dotLoc)) return false;

  auto ast = new (pool_) DesignatedInitializerClauseAST();
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
      auto ast = new (pool_) DefaultFunctionBodyAST();
      yyast = ast;

      ast->equalLoc = equalLoc;
      ast->defaultLoc = defaultLoc;

      expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

      return true;
    }

    SourceLocation deleteLoc;

    if (match(TokenKind::T_DELETE, deleteLoc)) {
      auto ast = new (pool_) DeleteFunctionBodyAST();
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

  auto ast = new (pool_) CompoundStatementFunctionBodyAST();
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

  DeclSpecs underlyingTypeSpecs{this};
  (void)parse_enum_base(colonLoc, typeSpecifierList, underlyingTypeSpecs);

  SourceLocation lbraceLoc;
  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  ScopeGuard scopeGuard{this};

  lookahead.commit();

  const auto underlyingType = underlyingTypeSpecs.getType();

  const Identifier* enumName = name ? name->identifier : nullptr;

  Symbol* symbol = nullptr;

  if (classLoc) {
    auto enumSymbol = control_->newScopedEnumSymbol(scope_);
    symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    std::invoke(DeclareSymbol{this, scope_}, enumSymbol);

    scope_ = enumSymbol->scope();
  } else {
    auto enumSymbol = control_->newEnumSymbol(scope_);
    symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    std::invoke(DeclareSymbol{this, scope_}, enumSymbol);

    scope_ = enumSymbol->scope();
  }

  auto ast = new (pool_) EnumSpecifierAST();
  yyast = ast;

  ast->enumLoc = enumLoc;
  ast->classLoc = classLoc;
  ast->attributeList = attributes;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = name;
  ast->colonLoc = colonLoc;
  ast->typeSpecifierList = typeSpecifierList;
  ast->lbraceLoc = lbraceLoc;

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    parse_enumerator_list(ast->enumeratorList, symbol->type());

    match(TokenKind::T_COMMA, ast->commaLoc);

    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  return true;
}

auto Parser::parse_enum_head_name(NestedNameSpecifierAST*& nestedNameSpecifier,
                                  NameIdAST*& name) -> bool {
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = new (pool_) NameIdAST();
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
  DeclSpecs underlyingTypeSpecs{this};
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

  auto ast = new (pool_) OpaqueEnumDeclarationAST();
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

  *it = new (pool_) List(enumerator);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (lookat(TokenKind::T_RBRACE)) {
      rewind(commaLoc);
      break;
    }

    EnumeratorAST* enumerator = nullptr;
    parse_enumerator(enumerator, type);

    *it = new (pool_) List(enumerator);
    it = &(*it)->next;
  }
}

void Parser::parse_enumerator(EnumeratorAST*& yyast, const Type* type) {
  auto ast = new (pool_) EnumeratorAST();
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

  auto enumeratorSymbol = control_->newEnumeratorSymbol(scope_);
  enumeratorSymbol->setName(ast->identifier);
  enumeratorSymbol->setType(type);
  enumeratorSymbol->setValue(value);

  std::invoke(DeclareSymbol{this, scope_}, enumeratorSymbol);

  if (auto enumSymbol = symbol_cast<EnumSymbol>(scope_->owner())) {
    auto enumeratorSymbol = control_->newEnumeratorSymbol(scope_);
    enumeratorSymbol->setName(ast->identifier);
    enumeratorSymbol->setType(type);
    enumeratorSymbol->setValue(value);

    auto parentScope = enumSymbol->enclosingScope();
    std::invoke(DeclareSymbol{this, parentScope}, enumeratorSymbol);
  }
}

auto Parser::parse_using_enum_declaration(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_USING, TokenKind::T_ENUM)) return false;

  auto ast = new (pool_) UsingEnumDeclarationAST();
  yyast = ast;

  expect(TokenKind::T_USING, ast->usingLoc);

  DeclSpecs specs{this};

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

  ScopeGuard scopeGuard{this};

  auto ast = new (pool_) NamespaceDefinitionAST();
  yyast = ast;

  ast->isInline = match(TokenKind::T_INLINE, ast->inlineLoc);

  expect(TokenKind::T_NAMESPACE, ast->namespaceLoc);

  parse_optional_attribute_specifier_seq(ast->attributeList);

  if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_COLON_COLON)) {
    auto it = &ast->nestedNamespaceSpecifierList;

    auto name = new (pool_) NestedNamespaceSpecifierAST();

    expect(TokenKind::T_IDENTIFIER, name->identifierLoc);
    expect(TokenKind::T_COLON_COLON, name->scopeLoc);

    name->identifier = unit->identifier(name->identifierLoc);

    *it = new (pool_) List(name);
    it = &(*it)->next;

    auto namepaceSymbol =
        enterOrCreateNamespace(name->identifier, /*isInline*/ false);

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

      auto namepaceSymbol = enterOrCreateNamespace(namespaceName, isInline);

      auto name = new (pool_) NestedNamespaceSpecifierAST();
      name->inlineLoc = inlineLoc;
      name->identifierLoc = identifierLoc;
      name->scopeLoc = scopeLoc;
      name->identifier = namespaceName;
      name->isInline = isInline;

      *it = new (pool_) List(name);
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

  auto namespaceSymbol = enterOrCreateNamespace(ast->identifier, ast->isInline);

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
    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    const auto beforeDeclaration = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_declaration(declaration, BindingContext::kNamespace)) {
      if (declaration) {
        *it = new (pool_) List(declaration);
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

  auto ast = new (pool_) NamespaceAliasDefinitionAST();
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
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto id = new (pool_) NameIdAST();
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

  auto ast = new (pool_) UsingDirectiveAST;
  yyast = ast;

  ast->usingLoc = usingLoc;
  ast->namespaceLoc = namespaceLoc;

  parse_optional_nested_name_specifier(ast->nestedNameSpecifier);

  auto currentNamespace = scope_->owner();

  if (!parse_name_id(ast->unqualifiedId)) {
    parse_error("expected a namespace name");
  } else {
    auto id = convertName(ast->unqualifiedId);

    Symbol* symbol = nullptr;

    if (!ast->nestedNameSpecifier)
      symbol = unqualifiedLookup(id);
    else {
      if (!ast->nestedNameSpecifier->symbol) {
        parse_error(ast->nestedNameSpecifier->firstSourceLocation(),
                    "expected a namespace name");
      } else {
        if (auto base = symbol_cast<NamespaceSymbol>(
                ast->nestedNameSpecifier->symbol)) {
          symbol = qualifiedLookup(base, id);
        } else {
          parse_error(ast->nestedNameSpecifier->firstSourceLocation(),
                      "expected a namespace name");
        }
      }
    }

    if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(symbol)) {
      scope_->addUsingDirective(namespaceSymbol->scope());
    } else {
      parse_error(ast->unqualifiedId->firstSourceLocation(),
                  "expected a namespace name");
    }
  }

  expect(TokenKind::T_SEMICOLON, ast->semicolonLoc);

  return true;
}

auto Parser::parse_using_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation usingLoc;

  if (!match(TokenKind::T_USING, usingLoc)) return false;

  auto ast = new (pool_) UsingDeclarationAST();
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

  declarator->isPack = match(TokenKind::T_DOT_DOT_DOT, declarator->ellipsisLoc);

  *it = new (pool_) List(declarator);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (UsingDeclaratorAST* declarator = nullptr;
        parse_using_declarator(declarator)) {
      declarator->isPack =
          match(TokenKind::T_DOT_DOT_DOT, declarator->ellipsisLoc);

      *it = new (pool_) List(declarator);
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

  parse_optional_nested_name_specifier(nestedNameSpecifier);

  UnqualifiedIdAST* unqualifiedId = nullptr;

  if (!parse_unqualified_id(unqualifiedId, /*isTemplateIntroduced*/ false,
                            /*inRequiresClause*/ false))
    return false;

  yyast = new (pool_) UsingDeclaratorAST();
  yyast->typenameLoc = typenameLoc;
  yyast->nestedNameSpecifier = nestedNameSpecifier;
  yyast->unqualifiedId = unqualifiedId;

  return true;
}

auto Parser::parse_asm_operand(AsmOperandAST*& yyast) -> bool {
  if (!LA().isOneOf(TokenKind::T_LBRACKET, TokenKind::T_STRING_LITERAL))
    return false;

  auto ast = new (pool_) AsmOperandAST();
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

  auto ast = new (pool_) AsmDeclarationAST();
  yyast = ast;

  ast->attributeList = attributes;
  ast->asmLoc = asmLoc;

  auto it = &ast->asmQualifierList;
  while (LA().isOneOf(TokenKind::T_INLINE, TokenKind::T_VOLATILE,
                      TokenKind::T_GOTO)) {
    auto qualifier = new (pool_) AsmQualifierAST();
    qualifier->qualifierLoc = consumeToken();
    *it = new (pool_) List(qualifier);
    it = &(*it)->next;
  }

  expect(TokenKind::T_LPAREN, ast->lparenLoc);
  expect(TokenKind::T_STRING_LITERAL, ast->literalLoc);

  if (SourceLocation colonLoc; match(TokenKind::T_COLON, colonLoc)) {
    if (AsmOperandAST* operand = nullptr; parse_asm_operand(operand)) {
      auto it = &ast->outputOperandList;
      *it = new (pool_) List(operand);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        if (AsmOperandAST* operand = nullptr; parse_asm_operand(operand)) {
          *it = new (pool_) List(operand);
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
      *it = new (pool_) List(operand);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        if (AsmOperandAST* operand = nullptr; parse_asm_operand(operand)) {
          *it = new (pool_) List(operand);
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
      auto clobber = new (pool_) AsmClobberAST();
      clobber->literalLoc = literalLoc;
      clobber->literal =
          static_cast<const StringLiteral*>(unit->literal(literalLoc));
      *it = new (pool_) List(clobber);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        SourceLocation literalLoc;
        expect(TokenKind::T_STRING_LITERAL, literalLoc);
        if (!literalLoc) continue;
        auto clobber = new (pool_) AsmClobberAST();
        clobber->literalLoc = literalLoc;
        clobber->literal =
            static_cast<const StringLiteral*>(unit->literal(literalLoc));
        *it = new (pool_) List(clobber);
        it = &(*it)->next;
      }
    }
  }

  if (SourceLocation colonLoc; match(TokenKind::T_COLON, colonLoc)) {
    if (SourceLocation identifierLoc;
        match(TokenKind::T_IDENTIFIER, identifierLoc)) {
      auto it = &ast->gotoLabelList;
      auto label = new (pool_) AsmGotoLabelAST();
      label->identifierLoc = identifierLoc;
      label->identifier = unit->identifier(label->identifierLoc);
      *it = new (pool_) List(label);
      it = &(*it)->next;
      SourceLocation commaLoc;
      while (match(TokenKind::T_COMMA, commaLoc)) {
        SourceLocation identifierLoc;
        expect(TokenKind::T_IDENTIFIER, identifierLoc);
        if (!identifierLoc) continue;
        auto label = new (pool_) AsmGotoLabelAST();
        label->identifierLoc = identifierLoc;
        label->identifier = unit->identifier(label->identifierLoc);
        *it = new (pool_) List(label);
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

    auto ast = new (pool_) LinkageSpecificationAST();
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

  auto ast = new (pool_) LinkageSpecificationAST();
  yyast = ast;

  ast->externLoc = externLoc;
  ast->stringliteralLoc = stringLiteralLoc;
  ast->stringLiteral =
      static_cast<const StringLiteral*>(unit->literal(ast->stringliteralLoc));
  ast->declarationList = new (pool_) List(declaration);

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

  *it = new (pool_) List(attribute);
  it = &(*it)->next;

  attribute = nullptr;

  while (parse_attribute_specifier(attribute, allowedAttributes)) {
    *it = new (pool_) List(attribute);
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
  if (LA(1).leadingSpace() || LA(1).startOfLine()) return false;
  return true;
}

auto Parser::parse_cxx_attribute_specifier(AttributeSpecifierAST*& yyast)
    -> bool {
  if (!lookat_cxx_attribute_specifier()) return false;

  auto ast = new (pool_) CxxAttributeAST();
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

  auto ast = new (pool_) AsmAttributeAST();
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

  auto ast = new (pool_) GccAttributeAST();
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

    auto ast = new (pool_) AlignasTypeAttributeAST();
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

  auto ast = new (pool_) AlignasAttributeAST();
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

  auto ast = new (pool_) AttributeUsingPrefixAST();
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

    *it = new (pool_) List(attribute);
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

      *it = new (pool_) List(attribute);
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

  auto ast = new (pool_) AttributeAST();
  yyast = ast;

  ast->attributeToken = attributeToken;
  ast->attributeArgumentClause = attributeArgumentClause;

  return true;
}

auto Parser::parse_attribute_token(AttributeTokenAST*& yyast) -> bool {
  if (parse_attribute_scoped_token(yyast)) return true;

  SourceLocation identifierLoc;

  if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

  auto ast = new (pool_) SimpleAttributeTokenAST();
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

  auto ast = new (pool_) ScopedAttributeTokenAST();
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

  auto ast = new (pool_) AttributeArgumentClauseAST();
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

  yyast = new (pool_) ModuleDeclarationAST();

  yyast->exportLoc = exportLoc;
  yyast->moduleLoc = moduleLoc;
  parse_module_name(yyast->moduleName);

  (void)parse_module_partition(yyast->modulePartition);

  parse_optional_attribute_specifier_seq(yyast->attributeList);

  expect(TokenKind::T_SEMICOLON, yyast->semicolonLoc);

  return true;
}

void Parser::parse_module_name(ModuleNameAST*& yyast) {
  auto ast = new (pool_) ModuleNameAST();
  yyast = ast;

  if (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_DOT)) {
    ast->moduleQualifier = new (pool_) ModuleQualifierAST();
    ast->moduleQualifier->identifierLoc = consumeToken();
    ast->moduleQualifier->identifier =
        unit->identifier(ast->moduleQualifier->identifierLoc);
    ast->moduleQualifier->dotLoc = consumeToken();

    while (lookat(TokenKind::T_IDENTIFIER, TokenKind::T_DOT)) {
      auto baseModuleQualifier = ast->moduleQualifier;
      ast->moduleQualifier = new (pool_) ModuleQualifierAST();
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

  yyast = new (pool_) ModulePartitionAST();

  yyast->colonLoc = colonLoc;

  parse_module_name(yyast->moduleName);

  return true;
}

auto Parser::parse_export_declaration(DeclarationAST*& yyast) -> bool {
  SourceLocation exportLoc;

  if (!match(TokenKind::T_EXPORT, exportLoc)) return false;

  SourceLocation lbraceLoc;

  if (match(TokenKind::T_LBRACE, lbraceLoc)) {
    auto ast = new (pool_) ExportCompoundDeclarationAST();
    yyast = ast;

    ast->exportLoc = exportLoc;
    ast->lbraceLoc = lbraceLoc;

    if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
      parse_declaration_seq(ast->declarationList);
      expect(TokenKind::T_RBRACE, ast->rbraceLoc);
    }

    return true;
  }

  auto ast = new (pool_) ExportDeclarationAST();
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

  auto ast = new (pool_) ModuleImportDeclarationAST();
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

  yyast = new (pool_) ImportNameAST();

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

  yyast = new (pool_) GlobalModuleFragmentAST();
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

  yyast = new (pool_) PrivateModuleFragmentAST();

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
  if (!LA().isOneOf(TokenKind::T_CLASS, TokenKind::T_STRUCT,
                    TokenKind::T_UNION))
    return false;

  const auto start = currentLocation();

  if (auto entry = class_specifiers_.get(start)) {
    auto [cursor, ast, parsed, hit] = *entry;
    rewind(cursor);
    yyast = ast;

    return parsed;
  }

  ScopeGuard scopeGuard{this};

  ClassHead classHead{templateDeclarations};

  auto lookat_class_specifier = [&] {
    LookaheadParser lookahead{this};

    if (parse_class_head(classHead)) {
      if (classHead.colonLoc || lookat(TokenKind::T_LBRACE)) {
        lookahead.commit();
        return true;
      }
    }

    class_specifiers_.set(start, currentLocation(),
                          static_cast<ClassSpecifierAST*>(nullptr), false);

    return false;
  };

  if (!lookat_class_specifier()) return false;

  SourceLocation lbraceLoc;
  expect(TokenKind::T_LBRACE, lbraceLoc);

  ClassSpecifierContext classContext(this);

  auto ast = new (pool_) ClassSpecifierAST();
  yyast = ast;

  ast->classLoc = classHead.classLoc;
  ast->attributeList = classHead.attributeList;
  ast->nestedNameSpecifier = classHead.nestedNameSpecifier;
  ast->unqualifiedId = classHead.name;
  ast->finalLoc = classHead.finalLoc;
  ast->colonLoc = classHead.colonLoc;
  ast->baseSpecifierList = classHead.baseSpecifierList;
  ast->lbraceLoc = lbraceLoc;

  ast->classKey = unit->tokenKind(ast->classLoc);

  if (classHead.finalLoc) {
    ast->isFinal = true;
  }

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    parse_class_body(ast->declarationList);
    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  class_specifiers_.set(start, currentLocation(), ast, true);

  return true;
}

void Parser::parse_class_body(List<DeclarationAST*>*& yyast) {
  auto it = &yyast;

  LoopParser loop{this};

  while (LA()) {
    if (lookat(TokenKind::T_RBRACE)) break;

    loop.start();

    const auto saved = currentLocation();

    DeclarationAST* declaration = nullptr;

    if (parse_member_specification(declaration)) {
      if (declaration) {
        *it = new (pool_) List(declaration);
        it = &(*it)->next;
      }
    } else {
      parse_error("expected a declaration");
    }
  }
}

auto Parser::parse_class_head(ClassHead& classHead) -> bool {
  if (!parse_class_key(classHead.classLoc)) return false;

  const auto isUnion =
      unit->tokenKind(classHead.classLoc) == TokenKind::T_UNION;

  parse_optional_attribute_specifier_seq(classHead.attributeList);

  auto is_class_declaration = false;

  if (parse_class_head_name(classHead.nestedNameSpecifier, classHead.name)) {
    if (parse_class_virt_specifier(classHead.finalLoc)) {
      is_class_declaration = true;
    }
  }

  if (LA().isOneOf(TokenKind::T_COLON, TokenKind::T_LBRACE)) {
    is_class_declaration = true;
  }

  auto is_template_declaration = !classHead.templateDeclarations.empty();

  if (is_class_declaration && is_template_declaration &&
      !classHead.nestedNameSpecifier) {
    mark_maybe_template_name(classHead.name);
  }

  if (classHead.nestedNameSpecifier) {
    auto enclosingSymbol = classHead.nestedNameSpecifier->symbol;
    if (!enclosingSymbol) {
      if (config_.checkTypes) {
        parse_error(classHead.nestedNameSpecifier->firstSourceLocation(),
                    "unresolved nested name specifier");
      }
    } else {
      Scope* enclosingScope = nullptr;
      if (auto enclosingClass = symbol_cast<ClassSymbol>(enclosingSymbol))
        enclosingScope = enclosingClass->scope();
      else if (auto enclosingNamespace =
                   symbol_cast<NamespaceSymbol>(enclosingSymbol))
        enclosingScope = enclosingNamespace->scope();

      scope_ = enclosingScope;
    }
  }

  const Identifier* id = nullptr;
  bool isTemplateSpecialization = false;
  if (const auto simpleName = ast_cast<NameIdAST>(classHead.name))
    id = simpleName->identifier;
  else if (const auto t = ast_cast<SimpleTemplateIdAST>(classHead.name)) {
    isTemplateSpecialization = true;
    id = t->identifier;
  }

  ClassSymbol* classSymbol = nullptr;

  if (id && !isTemplateSpecialization) {
    for (auto previous :
         scope_->get(id) | std::views::filter(&Symbol::isClass)) {
      if (auto previousClass = symbol_cast<ClassSymbol>(previous)) {
        if (previousClass->isComplete()) {
          parse_error(classHead.name->firstSourceLocation(),
                      "class name already declared");
        } else {
          classSymbol = previousClass;
        }
        break;
      }
    }
  }

  if (!classSymbol) {
    classSymbol = control_->newClassSymbol(scope_);
    classSymbol->setIsUnion(isUnion);

    classSymbol->setName(id);

    std::invoke(DeclareSymbol{this, scope_}, classSymbol);
  }

  scope_ = classSymbol->scope();

  (void)parse_base_clause(classSymbol, classHead.colonLoc,
                          classHead.baseSpecifierList);

  return true;
}

auto Parser::parse_class_head_name(NestedNameSpecifierAST*& nestedNameSpecifier,
                                   UnqualifiedIdAST*& yyast) -> bool {
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  check_type_traits();

  UnqualifiedIdAST* name = nullptr;

  if (!parse_type_name(name)) return false;

  yyast = name;

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
    auto ast = new (pool_) AccessDeclarationAST();
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
  DeclSpecs specs{this};
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

  (void)parse_decl_specifier_seq(*lastDeclSpecifier, specs);

  if (!specs.hasTypeSpecifier()) return false;

  if (SourceLocation semicolonLoc;
      match(TokenKind::T_SEMICOLON, semicolonLoc)) {
    auto ast = new (pool_) SimpleDeclarationAST();
    ast->attributeList = attributes;
    ast->declSpecifierList = declSpecifierList;
    ast->semicolonLoc = semicolonLoc;
    yyast = ast;
    return true;
  }

  auto lookat_function_definition = [&] {
    LookaheadParser lookahead{this};

    DeclaratorAST* declarator = nullptr;
    Decl decl{specs};
    (void)parse_declarator(declarator, decl);

    auto functionDeclarator = getFunctionPrototype(declarator);
    if (!functionDeclarator) return false;

    RequiresClauseAST* requiresClause = nullptr;
    if (!parse_requires_clause(requiresClause)) {
      parse_virt_specifier_seq(functionDeclarator);
    }

    parse_optional_attribute_specifier_seq(functionDeclarator->attributeList);

    if (!lookat_function_body()) return false;

    lookahead.commit();

    auto functionType =
        GetDeclaratorType{this}(declarator, decl.specs.getType());

    auto functionSymbol = control_->newFunctionSymbol(scope_);
    applySpecifiers(functionSymbol, decl.specs);
    functionSymbol->setName(decl.getName());
    functionSymbol->setType(functionType);
    std::invoke(DeclareSymbol{this, scope_}, functionSymbol);

    ScopeGuard scopeGuard{this};

    if (auto params = functionDeclarator->parameterDeclarationClause) {
      auto functionScope = functionSymbol->scope();
      std::invoke(DeclareSymbol{this, functionScope},
                  params->functionParametersSymbol);
      scope_ = params->functionParametersSymbol->scope();
    } else {
      scope_ = functionSymbol->scope();
    }

    FunctionBodyAST* functionBody = nullptr;
    if (!parse_function_body(functionBody)) {
      parse_error("expected function body");
    }

    auto ast = new (pool_) FunctionDefinitionAST();
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

  auto ast = new (pool_) SimpleDeclarationAST();
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
    *it = new (pool_) List(initDeclarator);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    InitDeclaratorAST* initDeclarator = nullptr;

    if (!parse_member_declarator(initDeclarator, specs)) {
      parse_error("expected a declarator");
    }

    if (initDeclarator) {
      *it = new (pool_) List(initDeclarator);
      it = &(*it)->next;
    }
  }

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

  auto nameId = new (pool_) NameIdAST();
  nameId->identifierLoc = identifierLoc;
  nameId->identifier = unit->identifier(identifierLoc);

  auto bitfieldDeclarator = new (pool_) BitfieldDeclaratorAST();
  bitfieldDeclarator->unqualifiedId = nameId;
  bitfieldDeclarator->colonLoc = colonLoc;
  bitfieldDeclarator->sizeExpression = sizeExpression;

  auto declarator = new (pool_) DeclaratorAST();
  declarator->coreDeclarator = bitfieldDeclarator;

  ExpressionAST* initializer = nullptr;

  (void)parse_brace_or_equal_initializer(initializer);

  auto ast = new (pool_) InitDeclaratorAST();
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

  auto symbolType = GetDeclaratorType{this}(declarator, decl.specs.getType());

  if (specs.isTypedef) {
    auto typedefSymbol = control_->newTypeAliasSymbol(scope_);
    typedefSymbol->setName(decl.getName());
    typedefSymbol->setType(symbolType);
    std::invoke(DeclareSymbol{this, scope_}, typedefSymbol);
  } else {
    if (auto functionDeclarator = getFunctionPrototype(declarator)) {
      auto functionSymbol = control_->newFunctionSymbol(scope_);
      applySpecifiers(functionSymbol, decl.specs);
      functionSymbol->setName(decl.getName());
      functionSymbol->setType(symbolType);
      std::invoke(DeclareSymbol{this, scope_}, functionSymbol);
    } else {
      auto fieldSymbol = control_->newFieldSymbol(scope_);
      applySpecifiers(fieldSymbol, decl.specs);
      fieldSymbol->setName(decl.getName());
      fieldSymbol->setType(symbolType);
      std::invoke(DeclareSymbol{this, scope_}, fieldSymbol);
    }
  }

  auto ast = new (pool_) InitDeclaratorAST();
  yyast = ast;

  ast->declarator = declarator;

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
  DeclSpecs specs{this};
  if (!parse_type_specifier_seq(typeSpecifierList, specs)) return false;

  lookahead.commit();

  auto declarator = new (pool_) DeclaratorAST();

  (void)parse_ptr_operator_seq(declarator->ptrOpList);

  auto typeId = new (pool_) TypeIdAST();
  typeId->typeSpecifierList = typeSpecifierList;
  typeId->declarator = declarator;
  typeId->type = GetDeclaratorType{this}(declarator, specs.getType());

  auto ast = new (pool_) ConversionFunctionIdAST();
  yyast = ast;

  ast->operatorLoc = operatorLoc;
  ast->typeId = typeId;

  return true;
}

auto Parser::parse_base_clause(ClassSymbol* classSymbol,
                               SourceLocation& colonLoc,
                               List<BaseSpecifierAST*>*& baseSpecifierList)
    -> bool {
  if (!match(TokenKind::T_COLON, colonLoc)) return false;

  if (!parse_base_specifier_list(classSymbol, baseSpecifierList)) {
    parse_error("expected a base class specifier");
  }

  return true;
}

auto Parser::parse_base_specifier_list(ClassSymbol* classSymbol,
                                       List<BaseSpecifierAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  BaseSpecifierAST* baseSpecifier = nullptr;

  parse_base_specifier(baseSpecifier);

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  if (baseSpecifier && baseSpecifier->symbol) {
    classSymbol->addBaseClass(baseSpecifier->symbol);
  }

  *it = new (pool_) List(baseSpecifier);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    BaseSpecifierAST* baseSpecifier = nullptr;

    parse_base_specifier(baseSpecifier);

    SourceLocation ellipsisLoc;

    match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

    if (baseSpecifier && baseSpecifier->symbol) {
      classSymbol->addBaseClass(baseSpecifier->symbol);
    }

    *it = new (pool_) List(baseSpecifier);
    it = &(*it)->next;
  }

  return true;
}

void Parser::parse_base_specifier(BaseSpecifierAST*& yyast) {
  auto ast = new (pool_) BaseSpecifierAST();
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
  } else {
    if (ast_cast<NameIdAST>(ast->unqualifiedId)) {
      auto name = convertName(ast->unqualifiedId);
      if (!ast->nestedNameSpecifier) {
        ast->symbol = unqualifiedLookup(name);
      } else {
        if (ast->nestedNameSpecifier->symbol) {
          ast->symbol = qualifiedLookup(ast->nestedNameSpecifier->symbol, name);
        }
      }
    }
  }

  if (ast->templateLoc) {
    ast->isTemplateIntroduced = true;
  }
}

auto Parser::parse_class_or_decltype(
    NestedNameSpecifierAST*& yynestedNameSpecifier,
    SourceLocation& yytemplateLoc, UnqualifiedIdAST*& yyast) -> bool {
  LookaheadParser lookahead{this};

  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  parse_optional_nested_name_specifier(nestedNameSpecifier);

  SourceLocation templateLoc;
  const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

  UnqualifiedIdAST* unqualifiedName = nullptr;

  if (isTemplateIntroduced) {
    if (SimpleTemplateIdAST* templateId = nullptr;
        parse_simple_template_id(templateId, isTemplateIntroduced)) {
      unqualifiedName = templateId;
    } else {
      parse_error("expected a template-id");
    }
  } else if (DecltypeSpecifierAST* decltypeSpecifier = nullptr;
             parse_decltype_specifier(decltypeSpecifier)) {
    DecltypeIdAST* decltypeName = new (pool_) DecltypeIdAST();
    decltypeName->decltypeSpecifier = decltypeSpecifier;
    unqualifiedName = decltypeName;
  } else if (UnqualifiedIdAST* name = nullptr; parse_type_name(name)) {
    unqualifiedName = name;
  } else {
    return false;
  }

  lookahead.commit();

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

  *it = new (pool_) List(mem_initializer);
  it = &(*it)->next;

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    MemInitializerAST* mem_initializer = nullptr;

    parse_mem_initializer(mem_initializer);
    *it = new (pool_) List(mem_initializer);
    it = &(*it)->next;
  }
}

void Parser::parse_mem_initializer(MemInitializerAST*& yyast) {
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  UnqualifiedIdAST* name = nullptr;

  if (!parse_mem_initializer_id(nestedNameSpecifier, name))
    parse_error("expected an member id");

  if (lookat(TokenKind::T_LBRACE)) {
    auto ast = new (pool_) BracedMemInitializerAST();
    yyast = ast;

    ast->nestedNameSpecifier = nestedNameSpecifier;
    ast->unqualifiedId = name;

    if (!parse_braced_init_list(ast->bracedInitList, ExprContext{})) {
      parse_error("expected an initializer");
    }

    match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

    return;
  }

  auto ast = new (pool_) ParenMemInitializerAST();
  yyast = ast;

  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = name;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  if (!match(TokenKind::T_RPAREN, ast->rparenLoc)) {
    if (!parse_expression_list(ast->expressionList, ExprContext{})) {
      parse_error("expected an expression");
    }

    expect(TokenKind::T_RPAREN, ast->rparenLoc);
  }

  match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);
}

auto Parser::parse_mem_initializer_id(
    NestedNameSpecifierAST*& yynestedNameSpecifier, UnqualifiedIdAST*& yyast)
    -> bool {
  auto lookat_class_or_decltype = [&] {
    LookaheadParser lookahead{this};
    NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
    SourceLocation templateLoc;
    if (!parse_class_or_decltype(nestedNameSpecifier, templateLoc, yyast))
      return false;
    lookahead.commit();
    yynestedNameSpecifier = nestedNameSpecifier;
    return true;
  };

  if (lookat_class_or_decltype()) return true;

  SourceLocation identifierLoc;

  if (match(TokenKind::T_IDENTIFIER, identifierLoc)) {
    auto name = new (pool_) NameIdAST();
    yyast = name;
    name->identifierLoc = identifierLoc;
    name->identifier = unit->identifier(identifierLoc);
    return true;
  }

  return false;
}

auto Parser::parse_operator_function_id(OperatorFunctionIdAST*& yyast) -> bool {
  SourceLocation operatorLoc;

  if (!match(TokenKind::T_OPERATOR, operatorLoc)) return false;

  TokenKind op = TokenKind::T_EOF_SYMBOL;
  SourceLocation opLoc;
  SourceLocation openLoc;
  SourceLocation closeLoc;

  if (!parse_operator(op, opLoc, openLoc, closeLoc)) return false;

  auto ast = new (pool_) OperatorFunctionIdAST();
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

  auto ast = new (pool_) LiteralOperatorIdAST();
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

  ScopeGuard scopeGuard{this};
  TemplateHeadContext templateHeadContext{this};

  auto ast = new (pool_) TemplateDeclarationAST();
  yyast = ast;

  auto templateParametersSymbol = control_->newTemplateParametersSymbol(scope_);
  ast->symbol = templateParametersSymbol;

  std::invoke(DeclareSymbol{this, scope_}, templateParametersSymbol);

  scope_ = ast->symbol->scope();

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

    *it = new (pool_) List(parameter);
    it = &(*it)->next;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    TemplateParameterAST* parameter = nullptr;
    parse_template_parameter(parameter);

    if (!parameter) continue;

    parameter->depth = templateParameterDepth_;
    parameter->index = templateParameterCount_++;

    *it = new (pool_) List(parameter);
    it = &(*it)->next;
  }

  std::swap(templateParameterCount_, templateParameterCount);
}

auto Parser::parse_requires_clause(RequiresClauseAST*& yyast) -> bool {
  SourceLocation requiresLoc;

  if (!match(TokenKind::T_REQUIRES, requiresLoc)) return false;

  yyast = new (pool_) RequiresClauseAST();

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

    auto ast = new (pool_) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->op = TokenKind::T_BAR_BAR;
    ast->rightExpression = expression;
    yyast = ast;
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

    auto ast = new (pool_) BinaryExpressionAST();
    ast->leftExpression = yyast;
    ast->opLoc = opLoc;
    ast->op = TokenKind::T_AMP_AMP;
    ast->rightExpression = expression;
    yyast = ast;
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

  auto symbol = control_->newNonTypeParameterSymbol(scope_);
  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);
  symbol->setName(parameter->identifier);
  symbol->setParameterPack(parameter->isPack);
  symbol->setObjectType(parameter->type);
  std::invoke(DeclareSymbol{this, scope_}, symbol);

  auto ast = new (pool_) NonTypeTemplateParameterAST();
  yyast = ast;

  ast->declaration = parameter;
  ast->symbol = symbol;

  lookahead.commit();
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

  auto ast = new (pool_) TypenameTypeParameterAST();
  yyast = ast;

  ast->classKeyLoc = classKeyLoc;

  const auto isPack = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  auto symbol = control_->newTypeParameterSymbol(scope_);
  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);
  symbol->setParameterPack(isPack);
  symbol->setName(ast->identifier);
  std::invoke(DeclareSymbol{this, scope_}, symbol);

  ast->symbol = symbol;

  if (!match(TokenKind::T_EQUAL, ast->equalLoc)) return true;

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  ast->isPack = isPack;

  return true;
}

void Parser::parse_template_type_parameter(TemplateParameterAST*& yyast) {
  ScopeGuard scopeGuard{this};

  auto ast = new (pool_) TemplateTypeParameterAST();
  yyast = ast;

  auto symbol = control_->newTemplateTypeParameterSymbol(scope_);
  ast->symbol = symbol;

  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);

  expect(TokenKind::T_TEMPLATE, ast->templateLoc);
  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    TemplateHeadContext templateHeadContext{this};

    auto parameters = control_->newTemplateParametersSymbol(scope_);

    scope_ = parameters->scope();

    parse_template_parameter_list(ast->templateParameterList);

    expect(TokenKind::T_GREATER, ast->greaterLoc);

    scope_ = parameters->enclosingScope();
  }

  (void)parse_requires_clause(ast->requiresClause);

  if (!parse_type_parameter_key(ast->classKeyLoc)) {
    parse_error("expected a type parameter");
  }

  ast->isPack = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  symbol->setParameterPack(ast->isPack);

  if (match(TokenKind::T_IDENTIFIER, ast->identifierLoc)) {
    ast->identifier = unit->identifier(ast->identifierLoc);
    symbol->setName(ast->identifier);

    mark_maybe_template_name(ast->identifier);
  }

  std::invoke(DeclareSymbol{this, scope_}, symbol);

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

  auto ast = new (pool_) ConstraintTypeParameterAST();
  yyast = ast;

  ast->typeConstraint = typeConstraint;
  ast->ellipsisLoc = ellipsisLoc;
  ast->identifierLoc = identifierLoc;
  ast->identifier = unit->identifier(identifierLoc);
  ast->equalLoc = equalLoc;
  ast->typeId = typeId;

  auto symbol = control_->newConstraintTypeParameterSymbol(scope_);
  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);
  symbol->setName(ast->identifier);
  std::invoke(DeclareSymbol{this, scope_}, symbol);

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

    parse_optional_nested_name_specifier(nestedNameSpecifier);

    if (!match(TokenKind::T_IDENTIFIER, identifierLoc)) return false;

    identifier = unit->identifier(identifierLoc);

    if (!concept_names_.contains(identifier)) return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_type_constraint()) return false;

  auto ast = new (pool_) TypeConstraintAST();
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

auto Parser::parse_simple_template_or_name_id(UnqualifiedIdAST*& yyast,
                                              bool isTemplateIntroduced)
    -> bool {
  auto lookat_simple_template_id = [&] {
    LookaheadParser lookahead{this};
    SimpleTemplateIdAST* templateName = nullptr;
    if (!parse_simple_template_id(templateName, isTemplateIntroduced))
      return false;
    lookahead.commit();
    yyast = templateName;
    return true;
  };

  if (lookat_simple_template_id()) return true;

  if (NameIdAST* nameId = nullptr; parse_name_id(nameId)) {
    yyast = nameId;
    return true;
  }

  return false;
}

auto Parser::parse_simple_template_id(SimpleTemplateIdAST*& yyast,
                                      bool isTemplateIntroduced) -> bool {
  LookaheadParser lookahead{this};

  if (!lookat(TokenKind::T_IDENTIFIER, TokenKind::T_LESS)) return false;

  SourceLocation identifierLoc = consumeToken();
  SourceLocation lessLoc = consumeToken();

  auto identifier = unit->identifier(identifierLoc);

  if (!isTemplateIntroduced && !maybe_template_name(identifier)) return false;

  SourceLocation greaterLoc;

  List<TemplateArgumentAST*>* templateArgumentList = nullptr;

  if (!match(TokenKind::T_GREATER, greaterLoc)) {
    if (!parse_template_argument_list(templateArgumentList)) return false;

    if (!match(TokenKind::T_GREATER, greaterLoc)) return false;
  }

  lookahead.commit();

  auto ast = new (pool_) SimpleTemplateIdAST();
  yyast = ast;

  ast->identifierLoc = identifierLoc;
  ast->identifier = identifier;
  ast->lessLoc = lessLoc;
  ast->templateArgumentList = templateArgumentList;
  ast->greaterLoc = greaterLoc;

  return true;
}

auto Parser::parse_literal_operator_template_id(
    LiteralOperatorTemplateIdAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_OPERATOR)) return false;

  LookaheadParser lookahead{this};

  LiteralOperatorIdAST* literalOperatorName = nullptr;
  if (!parse_literal_operator_id(literalOperatorName)) return false;

  if (!lookat(TokenKind::T_LESS)) return false;

  lookahead.commit();

  auto ast = new (pool_) LiteralOperatorTemplateIdAST();
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
    OperatorFunctionTemplateIdAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_OPERATOR)) return false;

  LookaheadParser lookahead{this};

  OperatorFunctionIdAST* operatorFunctionName = nullptr;
  if (!parse_operator_function_id(operatorFunctionName)) return false;

  if (!lookat(TokenKind::T_LESS)) return false;

  lookahead.commit();

  auto ast = new (pool_) OperatorFunctionTemplateIdAST();
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
                               bool isTemplateIntroduced) -> bool {
  if (LiteralOperatorTemplateIdAST* templateName = nullptr;
      parse_literal_operator_template_id(templateName)) {
    yyast = templateName;
    return true;
  } else if (OperatorFunctionTemplateIdAST* templateName = nullptr;
             parse_function_operator_template_id(templateName)) {
    yyast = templateName;
    return true;
  } else if (SimpleTemplateIdAST* templateName = nullptr;
             parse_simple_template_id(templateName, isTemplateIntroduced)) {
    yyast = templateName;
    return true;
  } else {
    return false;
  }
}

auto Parser::parse_template_argument_list(List<TemplateArgumentAST*>*& yyast)
    -> bool {
  auto it = &yyast;

  TemplateArgumentAST* templateArgument = nullptr;

  if (!parse_template_argument(templateArgument)) return false;

  SourceLocation ellipsisLoc;

  match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  *it = new (pool_) List(templateArgument);
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

    *it = new (pool_) List(templateArgument);
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

    auto ast = new (pool_) TypeTemplateArgumentAST();
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

    auto ast = new (pool_) ExpressionTemplateArgumentAST();
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

    DeclSpecs specs{this};
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

  if (!parse_simple_template_id(templateId)) {
    parse_error("expected a template id");
  }

  SourceLocation semicolonLoc;

  expect(TokenKind::T_SEMICOLON, semicolonLoc);

  auto ast = new (pool_) DeductionGuideAST();
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

  auto ast = new (pool_) ConceptDefinitionAST();
  yyast = ast;

  ast->conceptLoc = conceptLoc;

  expect(TokenKind::T_IDENTIFIER, ast->identifierLoc);
  ast->identifier = unit->identifier(ast->identifierLoc);

  auto symbol = control_->newConceptSymbol(scope_);
  symbol->setName(ast->identifier);
  std::invoke(DeclareSymbol{this, scope_}, symbol);

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

auto Parser::parse_typename_specifier(SpecifierAST*& yyast, DeclSpecs& specs)
    -> bool {
  if (specs.typeSpecifier) return false;

  SourceLocation typenameLoc;
  NestedNameSpecifierAST* nestedNameSpecifier = nullptr;
  SourceLocation templateLoc;
  UnqualifiedIdAST* unqualifiedId = nullptr;

  auto lookat_typename_specifier = [&] {
    LookaheadParser lookahead{this};
    if (!match(TokenKind::T_TYPENAME, typenameLoc)) return false;

    if (!parse_nested_name_specifier(nestedNameSpecifier)) return false;

    const auto isTemplateIntroduced = match(TokenKind::T_TEMPLATE, templateLoc);

    if (!parse_simple_template_or_name_id(unqualifiedId, isTemplateIntroduced))
      return false;

    lookahead.commit();

    return true;
  };

  if (!lookat_typename_specifier()) return false;

  auto ast = new (pool_) TypenameSpecifierAST();
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

  auto ast = new (pool_) ExplicitInstantiationAST();
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

  auto ast = new (pool_) TryBlockStatementAST();
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

  auto ast = new (pool_) TryStatementFunctionBodyAST();
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

  yyast = new (pool_) HandlerAST();

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
    *it = new (pool_) List(handler);
    it = &(*it)->next;
  }

  return true;
}

auto Parser::parse_exception_declaration(ExceptionDeclarationAST*& yyast)
    -> bool {
  SourceLocation ellipsisLoc;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    auto ast = new (pool_) EllipsisExceptionDeclarationAST();
    yyast = ast;

    ast->ellipsisLoc = ellipsisLoc;
    return true;
  }

  auto ast = new (pool_) TypeExceptionDeclarationAST();
  yyast = ast;

  parse_optional_attribute_specifier_seq(ast->attributeList);

  DeclSpecs specs{this};
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
    auto ast = new (pool_) ThrowExceptionSpecifierAST();
    yyast = ast;

    ast->throwLoc = throwLoc;
    expect(TokenKind::T_LPAREN, ast->lparenLoc);
    expect(TokenKind::T_RPAREN, ast->rparenLoc);

    return true;
  }

  SourceLocation noexceptLoc;

  if (!match(TokenKind::T_NOEXCEPT, noexceptLoc)) return false;

  auto ast = new (pool_) NoexceptSpecifierAST();
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
    *it = new (pool_) List<NameIdAST*>(id);
    it = &(*it)->next;
  } else {
    return false;
  }

  SourceLocation commaLoc;

  while (match(TokenKind::T_COMMA, commaLoc)) {
    if (NameIdAST* id = nullptr; parse_name_id(id)) {
      *it = new (pool_) List<NameIdAST*>(id);
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

void Parser::completeFunctionDefinition(FunctionDefinitionAST* ast) {
  if (!ast->functionBody) return;

  auto functionBody =
      ast_cast<CompoundStatementFunctionBodyAST>(ast->functionBody);

  if (!functionBody) return;

  ScopeGuard scopeGuard{this};

  scope_ = ast->symbol->scope();

  const auto saved = currentLocation();

  rewind(functionBody->statement->lbraceLoc.next());

  finish_compound_statement(functionBody->statement);

  rewind(saved);
}

auto Parser::convertName(UnqualifiedIdAST* id) -> const Name* {
  if (!id) return nullptr;
  return visit(ConvertToName{control_}, id);
}

auto Parser::unqualifiedLookup(const Name* name) -> Symbol* {
  std::unordered_set<Scope*> cache;
  for (auto current = scope_; current; current = current->parent()) {
    if (auto symbol = lookupHelper(current, name, cache)) {
      return symbol;
    }
  }
  return nullptr;
}

auto Parser::qualifiedLookup(Scope* scope, const Name* name) -> Symbol* {
  std::unordered_set<Scope*> cache;
  return lookupHelper(scope, name, cache);
}

auto Parser::qualifiedLookup(Symbol* scopedSymbol, const Name* name)
    -> Symbol* {
  if (!scopedSymbol) return nullptr;
  switch (scopedSymbol->kind()) {
    case SymbolKind::kNamespace:
      return qualifiedLookup(
          symbol_cast<NamespaceSymbol>(scopedSymbol)->scope(), name);
    case SymbolKind::kClass:
      return qualifiedLookup(symbol_cast<ClassSymbol>(scopedSymbol)->scope(),
                             name);
    case SymbolKind::kEnum:
      return qualifiedLookup(symbol_cast<EnumSymbol>(scopedSymbol)->scope(),
                             name);
    case SymbolKind::kScopedEnum:
      return qualifiedLookup(
          symbol_cast<ScopedEnumSymbol>(scopedSymbol)->scope(), name);
    default:
      return nullptr;
  }  // switch
}

auto Parser::lookup(Symbol* where, const Name* name) -> Symbol* {
  if (where) return qualifiedLookup(where, name);
  return unqualifiedLookup(name);
}

auto Parser::lookupHelper(Scope* scope, const Name* name,
                          std::unordered_set<Scope*>& cache) -> Symbol* {
  if (cache.contains(scope)) {
    return nullptr;
  }

  cache.insert(scope);

  for (auto symbol : scope->get(name)) {
    return symbol;
  }

  if (auto classSymbol = symbol_cast<ClassSymbol>(scope->owner())) {
    for (const auto& base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base);
      if (!baseClass) continue;
      if (auto symbol = lookupHelper(baseClass->scope(), name, cache)) {
        return symbol;
      }
    }
  }

  for (auto u : scope->usingDirectives()) {
    if (auto symbol = lookupHelper(u, name, cache)) {
      return symbol;
    }
  }

  return nullptr;
}

}  // namespace cxx
