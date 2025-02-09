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
#include <cxx/const_expression_evaluator.h>
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/name_lookup.h>
#include <cxx/name_printer.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbol_instantiation.h>
#include <cxx/symbols.h>
#include <cxx/token.h>
#include <cxx/type_printer.h>
#include <cxx/types.h>
#include <cxx/views/symbol_chain.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <ranges>
#include <unordered_set>

#include "cxx/cxx_fwd.h"
#include "cxx/parser_fwd.h"
#include "cxx/source_location.h"
#include "cxx/symbols_fwd.h"
#include "cxx/token_fwd.h"

namespace cxx {

namespace {

template <typename T>
auto make_node(Arena* arena) -> T* {
  auto node = new (arena) T();
  return node;
}

template <typename T>
auto make_list_node(Arena* arena, T* element = nullptr) -> List<T*>* {
  auto list = new (arena) List<T*>(element);
  return list;
}

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

    if (auto params = ast->parameterDeclarationClause) {
      for (auto it = params->parameterDeclarationList; it; it = it->next) {
        auto paramType = it->value->type;
        parameterTypes.push_back(paramType);
      }

      isVariadic = params->isVariadic;
    }

    CvQualifiers cvQualifiers = CvQualifiers::kNone;
    for (auto it = ast->cvQualifierList; it; it = it->next) {
      if (ast_cast<ConstQualifierAST>(it->value)) {
        if (cvQualifiers == CvQualifiers::kVolatile)
          cvQualifiers = CvQualifiers::kConstVolatile;
        else
          cvQualifiers = CvQualifiers::kConst;
      } else if (ast_cast<VolatileQualifierAST>(it->value)) {
        if (cvQualifiers == CvQualifiers::kConst)
          cvQualifiers = CvQualifiers::kConstVolatile;
        else
          cvQualifiers = CvQualifiers::kVolatile;
      }
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
                      std::format("function with trailing return type must "
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

  [[nodiscard]] auto location() const -> SourceLocation {
    if (declaratorId) return declaratorId->firstSourceLocation();
    return {};
  }

  [[nodiscard]] auto getName() const -> const Name* {
    auto control = specs.control();
    if (!declaratorId) return nullptr;
    if (!declaratorId->unqualifiedId) return nullptr;
    return visit(ConvertToName{control}, declaratorId->unqualifiedId);
  }

  [[nodiscard]] auto getNestedNameSpecifier() const -> NestedNameSpecifierAST* {
    if (!declaratorId) return nullptr;
    return declaratorId->nestedNameSpecifier;
  }

  [[nodiscard]] auto getScope() const -> Scope* {
    auto nestedNameSpecifier = getNestedNameSpecifier();
    if (!nestedNameSpecifier) return nullptr;

    auto symbol = nestedNameSpecifier->symbol;
    if (!symbol) return nullptr;

    if (auto alias = symbol_cast<TypeAliasSymbol>(symbol)) {
      if (auto classType = type_cast<ClassType>(alias->type())) {
        symbol = classType->symbol();
      }
    }

    if (auto classSymbol = symbol_cast<ClassSymbol>(symbol))
      return classSymbol->scope();

    if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(symbol))
      return namespaceSymbol->scope();

    return nullptr;
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
    if (scope) p->setScope(scope);
  }

  ~ScopeGuard() { p->setScope(savedScope); }
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
    auto symbol = symbol_cast<NamespaceSymbol>(Lookup{scope_}(identifier));
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
    ast->symbol = Lookup{scope_}(nestedNameSpecifier, name);
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
      ast->id = decltypeName;

      return true;
    }

    UnqualifiedIdAST* name = nullptr;
    if (!parse_type_name(name, nestedNameSpecifier, isTemplateIntroduced))
      return false;

    auto ast = make_node<DestructorIdAST>(pool_);
    yyast = ast;
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
  auto symbol = Lookup{scope_}.lookupType(yyast, identifier);

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

  auto _ = ScopeGuard{this};

  auto parentScope = declaringScope();
  auto symbol = control_->newLambdaSymbol(scope_, currentLocation());

  setScope(symbol);

  TemplateHeadContext templateHeadContext{this};

  auto ast = make_node<LambdaExpressionAST>(pool_);
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
    lambdaScope->addSymbol(params->functionParametersSymbol);
    setScope(params->functionParametersSymbol);
  } else {
    setScope(symbol);
  }

  if (!lookat(TokenKind::T_LBRACE)) return false;

  parentScope->addSymbol(symbol);

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

  for (auto current = scope_; current; current = current->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(current->owner())) {
      // maybe a this expression in a field initializer
      ast->type = control_->getPointerType(classSymbol->type());
      break;
    }

    if (auto functionSymbol = symbol_cast<FunctionSymbol>(current->owner())) {
      if (auto classSymbol =
              symbol_cast<ClassSymbol>(functionSymbol->enclosingSymbol())) {
        auto functionType = type_cast<FunctionType>(functionSymbol->type());
        const auto cv = functionType->cvQualifiers();
        if (cv != CvQualifiers::kNone) {
          auto elementType = control_->getQualType(classSymbol->type(), cv);
          ast->type = control_->getPointerType(elementType);
        } else {
          ast->type = control_->getPointerType(classSymbol->type());
        }
      }

      break;
    }
  }

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

  auto _ = ScopeGuard{this};

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

  const Type* objectType = nullptr;

  if (ast->baseExpression) {
    // test if the base expression has a type
    objectType = ast->baseExpression->type;
  }

  if (SourceLocation completionLoc;
      objectType && parse_completion(completionLoc)) {
    // trigger the completion
    config_.complete(MemberCompletionContext{
        .objectType = objectType,
        .accessOp = ast->accessOp,
    });
  }

  if (!parse_unqualified_id(ast->unqualifiedId, ast->nestedNameSpecifier,
                            ast->isTemplateIntroduced,
                            /*inRequiresClause*/ false))
    parse_error("expected an unqualified id");

  yyast = ast;

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
               std::format("call function {}", to_string(functionType)));
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
  DeclSpecs specs{this};
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
  else if (parse_reflect_expression(yyast, ctx))
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

  auto ast = make_node<UnaryExpressionAST>(pool_);
  yyast = ast;

  ast->opLoc = opLoc;
  ast->op = unit->tokenKind(opLoc);
  ast->expression = expression;

  switch (ast->op) {
    case TokenKind::T_STAR: {
      auto pointerType = type_cast<PointerType>(expression->type);
      if (pointerType) {
        ensure_prvalue(ast->expression);
        ast->type = pointerType->elementType();
        ast->valueCategory = ValueCategory::kLValue;
      }
      break;
    }

    case TokenKind::T_PLUS: {
      ExpressionAST* expr = ast->expression;
      ensure_prvalue(expr);
      auto ty = control_->remove_cvref(expr->type);
      if (control_->is_arithmetic_or_unscoped_enum(ty) ||
          control_->is_pointer(ty)) {
        if (control_->is_integral_or_unscoped_enum(ty)) {
          (void)integral_promotion(expr);
        }
        ast->expression = expr;
        ast->type = expr->type;
        ast->valueCategory = ValueCategory::kPrValue;
      }
      break;
    }

    case TokenKind::T_MINUS: {
      ExpressionAST* expr = ast->expression;
      ensure_prvalue(expr);
      auto ty = control_->remove_cvref(expr->type);
      if (control_->is_arithmetic_or_unscoped_enum(ty)) {
        if (control_->is_integral_or_unscoped_enum(ty)) {
          (void)integral_promotion(expr);
        }
        ast->expression = expr;
        ast->type = expr->type;
        ast->valueCategory = ValueCategory::kPrValue;
      }
      break;
    }

    default:
      break;
  }  // switch

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

    auto ast = make_node<AlignofTypeExpressionAST>(pool_);
    yyast = ast;

    ast->alignofLoc = alignofLoc;
    ast->lparenLoc = lparenLoc;
    ast->typeId = typeId;
    ast->rparenLoc = rparenLoc;
    ast->type = control_->getSizeType();

    return true;
  };

  if (lookat_alignof_type_id()) return true;

  auto ast = make_node<AlignofExpressionAST>(pool_);
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

    if (ast->leftExpression && ast->rightExpression) {
      ast->type = ast->leftExpression->type;

      auto sourceType = ast->rightExpression->type;

      (void)implicit_conversion(ast->rightExpression, ast->type);

#if false
      parse_warning(ast->opLoc,
              std::format("did convert {} to {}", to_string(sourceType),
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

    auto ast = make_node<BinaryExpressionAST>(pool_);
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

    if (!scope_->isBlockScope()) {
      cxx_runtime_error("not a block scope");
    }

    if (!scope_->empty()) {
      cxx_runtime_error("enclosing scope of init statement is not empty");
    }

    if (!parse_simple_declaration(declaration,
                                  BindingContext::kInitStatement)) {
      scope_->reset();
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

    DeclSpecs specs{this};

    if (!parse_decl_specifier_seq(declSpecifierList, specs, {})) return false;

    DeclaratorAST* declarator = nullptr;
    Decl decl{specs};
    if (!parse_declarator(declarator, decl)) return false;

    auto symbol = declareVariable(declarator, decl);

    ExpressionAST* initializer = nullptr;

    if (!parse_brace_or_equal_initializer(initializer)) return false;

    lookahead.commit();

    auto ast = make_node<ConditionExpressionAST>(pool_);
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

  auto _ = ScopeGuard{this};

  auto blockSymbol = control_->newBlockSymbol(scope_, lbraceLoc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol);

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
  auto _ = ScopeGuard{this};

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

  auto _ = ScopeGuard{this};

  auto blockSymbol = control_->newBlockSymbol(scope_, ifLoc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol);

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

  auto _ = ScopeGuard{this};

  auto ast = make_node<SwitchStatementAST>(pool_);
  yyast = ast;

  ast->switchLoc = switchLoc;

  expect(TokenKind::T_LPAREN, ast->lparenLoc);

  auto blockSymbol = control_->newBlockSymbol(scope_, ast->lparenLoc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol);

  parse_init_statement(ast->initializer);

  parse_condition(ast->condition, ExprContext{});

  expect(TokenKind::T_RPAREN, ast->rparenLoc);

  parse_statement(ast->statement);

  return true;
}

auto Parser::parse_while_statement(StatementAST*& yyast) -> bool {
  SourceLocation whileLoc;

  if (!match(TokenKind::T_WHILE, whileLoc)) return false;

  auto _ = ScopeGuard{this};

  auto blockSymbol = control_->newBlockSymbol(scope_, whileLoc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol);

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

  auto _ = ScopeGuard{this};

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

  auto _ = ScopeGuard{this};

  auto parentScope = scope_;
  auto blockSymbol = control_->newBlockSymbol(scope_, forLoc);
  parentScope->addSymbol(blockSymbol);

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

  DeclSpecs specs{this};

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

  auto symbol = declareTypeAlias(identifierLoc, typeId);

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
  auto parentScope = scope_;
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

  DeclSpecs specs{this};
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

    auto _ = ScopeGuard{this};

    auto functionType =
        GetDeclaratorType{this}(declarator, decl.specs.getType());

    auto q = decl.getNestedNameSpecifier();

    if (auto scope = decl.getScope()) {
      setScope(scope);
    } else if (q && config_.checkTypes) {
      parse_error(q->firstSourceLocation(),
                  std::format("unresolved class or namespace"));
    }

    const Name* functionName = decl.getName();
    auto functionSymbol = getFunction(scope_, functionName, functionType);

    if (!functionSymbol) {
      if (q && config_.checkTypes) {
        parse_error(q->firstSourceLocation(),
                    std::format("class or namespace has no member named '{}'",
                                to_string(functionName)));
      }

      functionSymbol = declareFunction(declarator, decl);
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

  auto _ = ScopeGuard{this};

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

  auto functionType = GetDeclaratorType{this}(declarator, decl.specs.getType());

  SourceLocation equalLoc;
  SourceLocation zeroLoc;

  const auto isPure = parse_pure_specifier(equalLoc, zeroLoc);

  functionDeclarator->isPure = isPure;

  const auto isDeclaration = isPure || lookat(TokenKind::T_SEMICOLON);
  const auto isDefinition = lookat_function_body();

  if (!isDeclaration && !isDefinition) return false;

  FunctionSymbol* functionSymbol =
      getFunction(scope_, decl.getName(), functionType);

  if (!functionSymbol) {
    functionSymbol = declareFunction(declarator, decl);
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

      specs.setTypeSpecifier(yyast);

      return true;
    }

    if (ClassSpecifierAST* classSpecifier = nullptr;
        parse_class_specifier(classSpecifier, specs, templateDeclarations)) {
      lookahead.commit();

      specs.setTypeSpecifier(classSpecifier);
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
  if (!config_.templates) return nullptr;

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

  if (auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId)) {
    if (auto symbol = instantiate(templateId)) {
      specs.type = symbol->type();
    }
  } else {
    auto name = ast_cast<NameIdAST>(unqualifiedId);
    auto symbol =
        Lookup{scope_}.lookupType(nestedNameSpecifier, name->identifier);

    if (is_type(symbol)) {
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

  auto ast = make_node<UnderlyingTypeSpecifierAST>(pool_);
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
      specs.type = control_->getBuiltinVaListType();
      return true;
    }

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
      auto ast = make_node<VoidTypeSpecifierAST>(pool_);
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

auto Parser::lvalue_to_rvalue_conversion(ExpressionAST*& expr) -> bool {
  if (!is_glvalue(expr)) return false;

  auto unref = control_->remove_cvref(expr->type);
  if (control_->is_function(unref)) return false;
  if (control_->is_array(unref)) return false;
  if (!control_->is_complete(unref)) return false;
  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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
  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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
  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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
    auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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
    auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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
    auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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
    auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

    auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

    auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

  auto cast = make_node<ImplicitCastExpressionAST>(pool_);
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

void Parser::ensure_prvalue(ExpressionAST*& expr) {
  if (lvalue_to_rvalue_conversion(expr)) {
    expr->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (array_to_pointer_conversion(expr)) {
    expr->valueCategory = ValueCategory::kPrValue;
    return;
  }

  if (function_to_pointer_conversion(expr)) {
    expr->valueCategory = ValueCategory::kPrValue;
    return;
  }
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

auto Parser::is_template(Symbol* symbol) const -> bool {
  if (!symbol) return false;
  if (symbol->isTemplateTypeParameter()) return true;
  auto templateParameters = cxx::getTemplateParameters(symbol);
  return templateParameters != nullptr;
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

  const auto location = ast->unqualifiedId->firstSourceLocation();

  const auto _ = ScopeGuard{this};

  if (ast->nestedNameSpecifier) {
    auto parent = ast->nestedNameSpecifier->symbol;

    if (parent && parent->isClassOrNamespace()) {
      setScope(static_cast<ScopedSymbol*>(parent));
    }
  }

  ClassSymbol* classSymbol = nullptr;

  if (scope_->isClassOrNamespaceScope()) {
    for (auto candidate : scope_->find(className) | views::classes) {
      classSymbol = candidate;
      break;
    }
  }

  if (!classSymbol) {
    const auto isUnion = classKey == TokenKind::T_UNION;
    classSymbol = control_->newClassSymbol(scope_, location);

    classSymbol->setIsUnion(isUnion);
    classSymbol->setName(className);
    classSymbol->setTemplateParameters(currentTemplateParameters());
    declaringScope()->addSymbol(classSymbol);
  }

  ast->symbol = classSymbol;

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
      auto typedefSymbol = declareTypedef(declarator, decl);
      symbol = typedefSymbol;
    } else if (getFunctionPrototype(declarator)) {
      auto functionSymbol = declareFunction(declarator, decl);
      symbol = functionSymbol;
    } else {
      auto variableSymbol = declareVariable(declarator, decl);
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

  auto _ = ScopeGuard{this};

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

  auto _ = ScopeGuard{this};

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

    DeclSpecs cvQualifiers{this};
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

  DeclSpecs cvQualifiers{this};
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
  DeclSpecs specs{this};
  if (!parse_type_specifier_seq(specifierList, specs)) return false;

  yyast = make_node<TypeIdAST>(pool_);
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

  auto ast = make_node<TypeIdAST>(pool_);
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

  auto _ = ScopeGuard{this};

  bool parsed = false;

  SourceLocation ellipsisLoc;
  FunctionParametersSymbol* functionParametersSymbol = nullptr;

  if (match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc)) {
    parsed = true;

    auto ast = make_node<ParameterDeclarationClauseAST>(pool_);
    yyast = ast;

    ast->ellipsisLoc = ellipsisLoc;
    ast->isVariadic = true;
    ast->functionParametersSymbol =
        control_->newFunctionParametersSymbol(scope_, {});
  } else if (List<ParameterDeclarationAST*>* parameterDeclarationList = nullptr;
             parse_parameter_declaration_list(parameterDeclarationList,
                                              functionParametersSymbol)) {
    parsed = true;

    auto ast = make_node<ParameterDeclarationClauseAST>(pool_);
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

  auto _ = ScopeGuard{this};

  functionParametersSymbol = control_->newFunctionParametersSymbol(scope_, {});

  setScope(functionParametersSymbol);

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

  DeclSpecs specs{this};

  specs.no_class_or_enum_specs = true;

  ast->isThisIntroduced = match(TokenKind::T_THIS, ast->thisLoc);

  if (!parse_decl_specifier_seq(ast->typeSpecifierList, specs, {}))
    return false;

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
    auto parameterSymbol =
        control_->newParameterSymbol(scope_, decl.location());
    parameterSymbol->setName(ast->identifier);
    parameterSymbol->setType(ast->type);
    scope_->addSymbol(parameterSymbol);
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

  DeclSpecs underlyingTypeSpecs{this};
  (void)parse_enum_base(colonLoc, typeSpecifierList, underlyingTypeSpecs);

  SourceLocation lbraceLoc;
  if (!match(TokenKind::T_LBRACE, lbraceLoc)) return false;

  auto _ = ScopeGuard{this};

  lookahead.commit();

  const auto underlyingType = underlyingTypeSpecs.getType();

  const Identifier* enumName = name ? name->identifier : nullptr;

  auto location = name ? name->firstSourceLocation() : lbraceLoc;

  Symbol* symbol = nullptr;

  if (classLoc) {
    auto enumSymbol = control_->newScopedEnumSymbol(scope_, location);
    symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope_->addSymbol(enumSymbol);

    setScope(enumSymbol);
  } else {
    auto enumSymbol = control_->newEnumSymbol(scope_, location);
    symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope_->addSymbol(enumSymbol);

    setScope(enumSymbol);
  }

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

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    parse_enumerator_list(ast->enumeratorList, symbol->type());

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

  auto enumeratorSymbol =
      control_->newEnumeratorSymbol(scope_, ast->identifierLoc);
  enumeratorSymbol->setName(ast->identifier);
  enumeratorSymbol->setType(type);
  enumeratorSymbol->setValue(value);

  scope_->addSymbol(enumeratorSymbol);

  if (auto enumSymbol = symbol_cast<EnumSymbol>(scope_->owner())) {
    auto enumeratorSymbol =
        control_->newEnumeratorSymbol(scope_, ast->identifierLoc);
    enumeratorSymbol->setName(ast->identifier);
    enumeratorSymbol->setType(type);
    enumeratorSymbol->setValue(value);

    auto parentScope = enumSymbol->enclosingScope();
    parentScope->addSymbol(enumeratorSymbol);
  }
}

auto Parser::parse_using_enum_declaration(DeclarationAST*& yyast) -> bool {
  if (!lookat(TokenKind::T_USING, TokenKind::T_ENUM)) return false;

  auto ast = make_node<UsingEnumDeclarationAST>(pool_);
  yyast = ast;

  expect(TokenKind::T_USING, ast->usingLoc);

  DeclSpecs specs{this};

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

  auto _ = ScopeGuard{this};

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

  auto currentNamespace = scope_->owner();

  if (!parse_name_id(ast->unqualifiedId)) {
    parse_error("expected a namespace name");
  } else {
    auto id = ast->unqualifiedId->identifier;

    NamespaceSymbol* namespaceSymbol =
        Lookup{scope_}.lookupNamespace(ast->nestedNameSpecifier, id);

    if (namespaceSymbol) {
      scope_->addUsingDirective(namespaceSymbol->scope());
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

  SourceLocation ellipsisLoc;
  auto isPack = match(TokenKind::T_DOT_DOT_DOT, ellipsisLoc);

  yyast = make_node<UsingDeclaratorAST>(pool_);
  yyast->typenameLoc = typenameLoc;
  yyast->nestedNameSpecifier = nestedNameSpecifier;
  yyast->unqualifiedId = unqualifiedId;
  yyast->ellipsisLoc = ellipsisLoc;
  yyast->isPack = isPack;

  auto target = Lookup{scope_}.lookup(nestedNameSpecifier, name);

  if (auto u = symbol_cast<UsingDeclarationSymbol>(target)) {
    target = u->target();
  }

  auto symbol = control_->newUsingDeclarationSymbol(
      scope_, unqualifiedId->firstSourceLocation());

  yyast->symbol = symbol;

  symbol->setName(name);
  symbol->setDeclarator(yyast);
  symbol->setTarget(target);

  scope_->addSymbol(symbol);

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
  SourceLocation colonLoc;
  List<BaseSpecifierAST*>* baseSpecifierList = nullptr;
  bool isUnion = false;
  bool isTemplateSpecialization = false;
  SourceLocation location = classLoc;
  ClassSymbol* classSymbol = nullptr;

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

  auto _ = ScopeGuard{this};

  if (scope_->isTemplateParametersScope()) {
    mark_maybe_template_name(unqualifiedId);
  }

  auto templateParameters = currentTemplateParameters();

  if (nestedNameSpecifier) {
    auto parent = nestedNameSpecifier->symbol;

    if (parent && parent->isClassOrNamespace()) {
      setScope(static_cast<ScopedSymbol*>(parent));
    }
  }

  ClassSymbol* primaryTemplate = nullptr;

  if (templateId && scope_->isTemplateParametersScope()) {
    for (auto candidate : declaringScope()->find(className) | views::classes) {
      primaryTemplate = candidate;
      break;
    }

    if (!primaryTemplate && config_.checkTypes) {
      parse_error(location,
                  std::format("specialization of undeclared template '{}'",
                              className->name()));
    }
  }

  if (className) {
    for (auto candidate : declaringScope()->find(className) | views::classes) {
      classSymbol = candidate;
      break;
    }
  }

  if (classSymbol && classSymbol->isComplete()) {
    classSymbol = nullptr;
  }

  if (!classSymbol) {
    classSymbol = control_->newClassSymbol(scope_, location);
    classSymbol->setIsUnion(isUnion);
    classSymbol->setName(className);
    classSymbol->setTemplateParameters(templateParameters);

    if (!primaryTemplate) {
      declaringScope()->addSymbol(classSymbol);
    } else {
      std::vector<TemplateArgument> arguments;
      // TODO: parse template arguments
      primaryTemplate->addSpecialization(arguments, classSymbol);
    }
  }

  if (finalLoc) {
    classSymbol->setFinal(true);
  }

  setScope(classSymbol);

  (void)parse_base_clause(classSymbol, colonLoc, baseSpecifierList);

  SourceLocation lbraceLoc;
  expect(TokenKind::T_LBRACE, lbraceLoc);

  ClassSpecifierContext classContext(this);

  auto ast = make_node<ClassSpecifierAST>(pool_);
  yyast = ast;

  ast->symbol = classSymbol;
  ast->classLoc = classLoc;
  ast->attributeList = attributeList;
  ast->nestedNameSpecifier = nestedNameSpecifier;
  ast->unqualifiedId = unqualifiedId;
  ast->finalLoc = finalLoc;
  ast->colonLoc = colonLoc;
  ast->baseSpecifierList = baseSpecifierList;
  ast->lbraceLoc = lbraceLoc;

  ast->classKey = unit->tokenKind(ast->classLoc);

  if (finalLoc) {
    ast->isFinal = true;
  }

  if (!match(TokenKind::T_RBRACE, ast->rbraceLoc)) {
    parse_class_body(ast->declarationList);
    expect(TokenKind::T_RBRACE, ast->rbraceLoc);
  }

  ast->symbol->setComplete(true);

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

    auto functionSymbol = declareFunction(declarator, decl);

    auto _ = ScopeGuard{this};

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

auto Parser::declareTypeAlias(SourceLocation identifierLoc, TypeIdAST* typeId)
    -> TypeAliasSymbol* {
  auto name = unit->identifier(identifierLoc);
  auto symbol = control_->newTypeAliasSymbol(scope_, identifierLoc);
  symbol->setName(name);
  if (typeId) symbol->setType(typeId->type);
  symbol->setTemplateParameters(currentTemplateParameters());
  declaringScope()->addSymbol(symbol);
  return symbol;
}

auto Parser::declareTypedef(DeclaratorAST* declarator, const Decl& decl)
    -> TypeAliasSymbol* {
  auto name = decl.getName();
  auto type = GetDeclaratorType{this}(declarator, decl.specs.getType());
  auto symbol = control_->newTypeAliasSymbol(scope_, decl.location());
  symbol->setName(name);
  symbol->setType(type);
  scope_->addSymbol(symbol);
  return symbol;
}

auto Parser::declareFunction(DeclaratorAST* declarator, const Decl& decl)
    -> FunctionSymbol* {
  auto name = decl.getName();
  auto type = GetDeclaratorType{this}(declarator, decl.specs.getType());

  auto parentScope = scope_;

  if (parentScope->isBlockScope()) {
    parentScope = parentScope->enclosingNamespaceScope();
  }

  auto functionSymbol = control_->newFunctionSymbol(scope_, decl.location());
  applySpecifiers(functionSymbol, decl.specs);
  functionSymbol->setName(name);
  functionSymbol->setType(type);
  functionSymbol->setTemplateParameters(currentTemplateParameters());

  if (is_constructor(functionSymbol)) {
    auto enclosingClass = symbol_cast<ClassSymbol>(scope_->owner());

    if (enclosingClass) {
      enclosingClass->addConstructor(functionSymbol);
    }

    return functionSymbol;
  }

  auto scope = declaringScope();

  OverloadSetSymbol* overloadSet = nullptr;

  for (Symbol* candidate : scope->find(functionSymbol->name())) {
    overloadSet = symbol_cast<OverloadSetSymbol>(candidate);
    if (overloadSet) break;

    if (auto previousFunction = symbol_cast<FunctionSymbol>(candidate)) {
      overloadSet = control_->newOverloadSetSymbol(scope, {});
      overloadSet->setName(functionSymbol->name());
      overloadSet->addFunction(previousFunction);
      scope->replaceSymbol(previousFunction, overloadSet);
      break;
    }
  }

  if (overloadSet) {
    overloadSet->addFunction(functionSymbol);
  } else {
    scope->addSymbol(functionSymbol);
  }

  return functionSymbol;
}

auto Parser::declareField(DeclaratorAST* declarator, const Decl& decl)
    -> FieldSymbol* {
  auto name = decl.getName();
  auto type = GetDeclaratorType{this}(declarator, decl.specs.getType());
  auto fieldSymbol = control_->newFieldSymbol(scope_, decl.location());
  applySpecifiers(fieldSymbol, decl.specs);
  fieldSymbol->setName(name);
  fieldSymbol->setType(type);
  scope_->addSymbol(fieldSymbol);
  return fieldSymbol;
}

auto Parser::declareVariable(DeclaratorAST* declarator, const Decl& decl)
    -> VariableSymbol* {
  auto name = decl.getName();
  auto symbol = control_->newVariableSymbol(scope_, decl.location());
  auto type = GetDeclaratorType{this}(declarator, decl.specs.getType());
  applySpecifiers(symbol, decl.specs);
  symbol->setName(name);
  symbol->setType(type);
  symbol->setTemplateParameters(currentTemplateParameters());
  declaringScope()->addSymbol(symbol);
  return symbol;
}

auto Parser::declareMemberSymbol(DeclaratorAST* declarator, const Decl& decl)
    -> Symbol* {
  if (decl.specs.isTypedef) return declareTypedef(declarator, decl);

  if (getFunctionPrototype(declarator))
    return declareFunction(declarator, decl);

  return declareField(declarator, decl);
}

auto Parser::parse_member_declarator(InitDeclaratorAST*& yyast,
                                     DeclaratorAST* declarator,
                                     const Decl& decl) -> bool {
  if (!declarator) {
    return false;
  }

  auto symbol = declareMemberSymbol(declarator, decl);

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
  DeclSpecs specs{this};
  if (!parse_type_specifier_seq(typeSpecifierList, specs)) return false;

  lookahead.commit();

  auto declarator = make_node<DeclaratorAST>(pool_);

  (void)parse_ptr_operator_seq(declarator->ptrOpList);

  auto typeId = make_node<TypeIdAST>(pool_);
  typeId->typeSpecifierList = typeSpecifierList;
  typeId->declarator = declarator;
  typeId->type = GetDeclaratorType{this}(declarator, specs.getType());

  auto ast = make_node<ConversionFunctionIdAST>(pool_);
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

  *it = make_list_node(pool_, baseSpecifier);
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
  } else {
    Symbol* symbol = nullptr;

    if (auto decltypeId = ast_cast<DecltypeIdAST>(ast->unqualifiedId)) {
      if (auto classType = type_cast<ClassType>(
              control_->remove_cv(decltypeId->decltypeSpecifier->type))) {
        symbol = classType->symbol();
      }
    }

    if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId)) {
      symbol = Lookup{scope_}(ast->nestedNameSpecifier, nameId->identifier);
    }

    if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
      if (auto classType =
              type_cast<ClassType>(control_->remove_cv(typeAlias->type()))) {
        symbol = classType->symbol();
      }
    }

    if (symbol) {
      auto location = ast->unqualifiedId->firstSourceLocation();
      auto baseClassSymbol = control_->newBaseClassSymbol(scope_, location);
      ast->symbol = baseClassSymbol;

      baseClassSymbol->setVirtual(ast->isVirtual);
      baseClassSymbol->setSymbol(symbol);

      if (symbol) {
        baseClassSymbol->setName(symbol->name());
      }

      switch (ast->accessSpecifier) {
        case TokenKind::T_PRIVATE:
          baseClassSymbol->setAccessSpecifier(AccessSpecifier::kPrivate);
          break;
        case TokenKind::T_PROTECTED:
          baseClassSymbol->setAccessSpecifier(AccessSpecifier::kProtected);
          break;
        case TokenKind::T_PUBLIC:
          baseClassSymbol->setAccessSpecifier(AccessSpecifier::kPublic);
          break;
        default:
          break;
      }  // switch
    }
  }

  if (ast->templateLoc) {
    ast->isTemplateIntroduced = true;
  }
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

  auto _ = ScopeGuard{this};
  TemplateHeadContext templateHeadContext{this};

  auto ast = make_node<TemplateDeclarationAST>(pool_);
  yyast = ast;

  auto templateParametersSymbol =
      control_->newTemplateParametersSymbol(scope_, {});
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

  auto symbol = control_->newNonTypeParameterSymbol(
      scope_, parameter->firstSourceLocation());
  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);
  symbol->setName(parameter->identifier);
  symbol->setParameterPack(parameter->isPack);
  symbol->setObjectType(parameter->type);
  scope_->addSymbol(symbol);

  auto ast = make_node<NonTypeTemplateParameterAST>(pool_);
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

  auto ast = make_node<TypenameTypeParameterAST>(pool_);
  yyast = ast;

  ast->classKeyLoc = classKeyLoc;

  const auto isPack = match(TokenKind::T_DOT_DOT_DOT, ast->ellipsisLoc);

  match(TokenKind::T_IDENTIFIER, ast->identifierLoc);

  ast->identifier = unit->identifier(ast->identifierLoc);

  auto location = ast->identifier ? ast->identifierLoc : classKeyLoc;

  auto symbol = control_->newTypeParameterSymbol(scope_, location);
  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);
  symbol->setParameterPack(isPack);
  symbol->setName(ast->identifier);
  scope_->addSymbol(symbol);

  ast->symbol = symbol;

  if (!match(TokenKind::T_EQUAL, ast->equalLoc)) return true;

  if (!parse_type_id(ast->typeId)) parse_error("expected a type id");

  ast->isPack = isPack;

  return true;
}

void Parser::parse_template_type_parameter(TemplateParameterAST*& yyast) {
  auto _ = ScopeGuard{this};

  auto ast = make_node<TemplateTypeParameterAST>(pool_);
  yyast = ast;

  auto symbol =
      control_->newTemplateTypeParameterSymbol(scope_, currentLocation());
  ast->symbol = symbol;

  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);

  expect(TokenKind::T_TEMPLATE, ast->templateLoc);
  expect(TokenKind::T_LESS, ast->lessLoc);

  if (!match(TokenKind::T_GREATER, ast->greaterLoc)) {
    TemplateHeadContext templateHeadContext{this};

    auto parameters =
        control_->newTemplateParametersSymbol(scope_, ast->templateLoc);

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

  symbol->setParameterPack(ast->isPack);

  if (match(TokenKind::T_IDENTIFIER, ast->identifierLoc)) {
    ast->identifier = unit->identifier(ast->identifierLoc);
    symbol->setName(ast->identifier);

    mark_maybe_template_name(ast->identifier);
  }

  scope_->addSymbol(symbol);

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

  auto symbol =
      control_->newConstraintTypeParameterSymbol(scope_, identifierLoc);
  symbol->setIndex(templateParameterCount_);
  symbol->setDepth(templateParameterDepth_);
  symbol->setName(ast->identifier);
  scope_->addSymbol(symbol);

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

  auto candidate = Lookup{scope_}(nestedNameSpecifier, templateId->identifier);

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

  auto templateParameters = currentTemplateParameters();

  auto symbol = control_->newConceptSymbol(scope_, ast->identifierLoc);
  symbol->setName(ast->identifier);
  symbol->setTemplateParameters(templateParameters);

  declaringScope()->addSymbol(symbol);

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

void Parser::setScope(Scope* scope) { scope_ = scope; }

void Parser::setScope(ScopedSymbol* symbol) { setScope(symbol->scope()); }

auto Parser::currentTemplateParameters() const -> TemplateParametersSymbol* {
  auto templateParameters =
      symbol_cast<TemplateParametersSymbol>(scope_->owner());

  return templateParameters;
}

auto Parser::declaringScope() const -> Scope* {
  if (!scope_->isTemplateParametersScope()) return scope_;
  return scope_->enclosingNonTemplateParametersScope();
}

void Parser::completeFunctionDefinition(FunctionDefinitionAST* ast) {
  if (!ast->functionBody) return;

  auto functionBody =
      ast_cast<CompoundStatementFunctionBodyAST>(ast->functionBody);

  if (!functionBody) return;

  auto _ = ScopeGuard{this};

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

auto Parser::convertName(UnqualifiedIdAST* id) -> const Name* {
  if (!id) return nullptr;
  return visit(ConvertToName{control_}, id);
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
