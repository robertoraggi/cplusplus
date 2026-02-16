// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/binder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {

Binder::Binder(TranslationUnit* unit) : unit_(unit) {
  languageLinkage_ = unit->language() == LanguageKind::kC ? LanguageKind::kC
                                                          : LanguageKind::kCXX;
}

auto Binder::translationUnit() const -> TranslationUnit* { return unit_; }

void Binder::setTranslationUnit(TranslationUnit* unit) { unit_ = unit; }

auto Binder::control() const -> Control* {
  return unit_ ? unit_->control() : nullptr;
}

auto Binder::is_parsing_c() const -> bool {
  return unit_->language() == LanguageKind::kC;
}

auto Binder::is_parsing_cxx() const -> bool {
  return unit_->language() == LanguageKind::kCXX;
}

auto Binder::reportErrors() const -> bool { return reportErrors_; }

void Binder::setReportErrors(bool reportErrors) {
  reportErrors_ = reportErrors;
}

void Binder::error(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  if (!unit_->config().checkTypes) return;
  unit_->error(loc, std::move(message));
}

void Binder::warning(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  if (!unit_->config().checkTypes) return;
  unit_->warning(loc, std::move(message));
}

auto Binder::inTemplate() const -> bool { return inTemplate_; }

auto Binder::currentTemplateParameters() const -> TemplateParametersSymbol* {
  auto templateParameters = symbol_cast<TemplateParametersSymbol>(scope());
  return templateParameters;
}

auto Binder::isInstantiating() const -> bool {
  return instantiatingSymbol_ != nullptr;
}

auto Binder::instantiatingSymbol() const -> Symbol* {
  return instantiatingSymbol_;
}

void Binder::setInstantiatingSymbol(Symbol* symbol) {
  instantiatingSymbol_ = symbol;
}

auto Binder::declaringScope() const -> ScopeSymbol* {
  if (!scope_) return nullptr;
  if (!scope_->isTemplateParameters()) return scope_;
  return scope_->enclosingNonTemplateParametersScope();
}

auto Binder::scope() const -> ScopeSymbol* { return scope_; }

void Binder::setScope(ScopeSymbol* scope) {
  scope_ = scope;
  inTemplate_ = false;

  for (auto current = scope_; current; current = current->parent()) {
    if (current->isTemplateParameters()) {
      inTemplate_ = true;
      break;
    }
  }
}

auto Binder::languageLinkage() const -> LanguageKind {
  return languageLinkage_;
}

void Binder::setLanguageLinkage(LanguageKind linkage) {
  languageLinkage_ = linkage;
}

auto Binder::changeLanguageLinkage(LanguageKind linkage) -> LanguageKind {
  std::swap(languageLinkage_, linkage);
  return linkage;
}

auto Binder::enterBlock(SourceLocation loc) -> BlockSymbol* {
  auto blockSymbol = control()->newBlockSymbol(scope_, loc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol);
  return blockSymbol;
}

void Binder::bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpecs) {
  const Type* underlyingType = control()->getIntType();

  if (underlyingTypeSpecs.hasTypeOrSizeSpecifier())
    underlyingType = underlyingTypeSpecs.type();

  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->lbraceLoc;

  auto enumName = get_name(control(), ast->unqualifiedId);

  if (ast->classLoc && is_parsing_cxx()) {
    auto enumSymbol = control()->newScopedEnumSymbol(scope(), location);
    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  } else {
    if (is_parsing_c() && ast->classLoc) {
      error(ast->classLoc, "scoped enums are not allowed in C");
    }

    auto enumSymbol = control()->newEnumSymbol(scope(), location);

    if (ast->typeSpecifierList) {
      enumSymbol->setHasFixedUnderlyingType(true);
    }

    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  }
}

void Binder::bind(ElaboratedTypeSpecifierAST* ast, DeclSpecs& declSpecs,
                  bool isDeclaration) {
  const auto _ = ScopeGuard{this};

  if (ast->nestedNameSpecifier) {
    auto parent = ast->nestedNameSpecifier->symbol;

    if (parent && parent->isClassOrNamespace()) {
      setScope(static_cast<ScopeSymbol*>(parent));
    }
  }

  auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);

  const Identifier* name = nullptr;
  if (templateId)
    name = templateId->identifier;
  else if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId))
    name = nameId->identifier;

  const auto location = ast->unqualifiedId->firstSourceLocation();

  if (ast->classKey == TokenKind::T_CLASS ||
      ast->classKey == TokenKind::T_STRUCT ||
      ast->classKey == TokenKind::T_UNION) {
    auto is_class = [](Symbol* symbol) {
      if (symbol->isClass()) return true;
      return false;
    };

    auto targetScope = [&]() -> ScopeSymbol* {
      if (!declSpecs.isFriend) return declaringScope();
      auto ds = declaringScope();
      if (ds->isNamespace()) return ds;
      if (auto ns = ds->enclosingNamespace()) return ns;
      return ds;
    }();

    auto candidate =
        declSpecs.isFriend
            ? Lookup{targetScope}.lookup(nullptr, name, is_class)
            : Lookup{scope()}.lookup(ast->nestedNameSpecifier, name, is_class);

    auto classSymbol = symbol_cast<ClassSymbol>(candidate);

    if (classSymbol && isDeclaration &&
        classSymbol->enclosingNonTemplateParametersScope() != targetScope) {
      // the class is declared in a different scope
      classSymbol = nullptr;
    }

    if (!classSymbol) {
      const auto isUnion = ast->classKey == TokenKind::T_UNION;
      classSymbol = control()->newClassSymbol(scope(), location);

      classSymbol->setIsUnion(isUnion);
      classSymbol->setName(name);
      classSymbol->setTemplateDeclaration(declSpecs.templateHead);
      targetScope->addSymbol(classSymbol);

      if (declSpecs.isFriend) {
        classSymbol->setFriend(true);
        classSymbol->setHidden(true);
      }

      classSymbol->setDeclaration(ast);
    }

    ast->symbol = classSymbol;
  }

  declSpecs.setTypeSpecifier(ast);

  if (ast->symbol) {
    declSpecs.setType(ast->symbol->type());
  }
}

void Binder::bind(ParameterDeclarationAST* ast, const Decl& decl,
                  bool inTemplateParameters) {
  ast->type = getDeclaratorType(unit_, ast->declarator, decl.specs.type());

  // decay the type of the parameters
  if (control()->is_array(ast->type))
    ast->type = control()->add_pointer(control()->remove_extent(ast->type));
  else if (control()->is_function(ast->type))
    ast->type = control()->add_pointer(ast->type);
  else if (control()->is_scalar(ast->type))
    ast->type = control()->remove_cv(ast->type);

  if (auto declId = decl.declaratorId; declId && declId->unqualifiedId) {
    auto paramName = get_name(control(), declId->unqualifiedId);
    if (auto identifier = name_cast<Identifier>(paramName)) {
      ast->identifier = identifier;
    } else {
      error(declId->unqualifiedId->firstSourceLocation(),
            "expected an identifier");
    }
  }

  if (!inTemplateParameters) {
    auto parameterSymbol =
        control()->newParameterSymbol(scope_, decl.location());
    parameterSymbol->setName(ast->identifier);
    parameterSymbol->setType(ast->type);
    parameterSymbol->setDefaultArgument(ast->expression);
    scope_->addSymbol(parameterSymbol);
  }
}

void Binder::bind(DecltypeSpecifierAST* ast) {
  if (auto id = ast_cast<IdExpressionAST>(ast->expression)) {
    if (id->symbol) ast->type = id->symbol->type();
  } else if (auto member = ast_cast<MemberExpressionAST>(ast->expression)) {
    if (member->symbol) ast->type = member->symbol->type();
  } else if (ast->expression && ast->expression->type) {
    if (is_lvalue(ast->expression)) {
      ast->type = control()->add_lvalue_reference(ast->expression->type);
    } else if (is_xvalue(ast->expression)) {
      ast->type = control()->add_rvalue_reference(ast->expression->type);
    } else {
      ast->type = ast->expression->type;
    }
  }
}

void Binder::bind(EnumeratorAST* ast, const Type* type,
                  std::optional<ConstValue> value) {
  if (is_parsing_cxx()) {
    auto symbol = control()->newEnumeratorSymbol(scope(), ast->identifierLoc);
    ast->symbol = symbol;

    symbol->setName(ast->identifier);
    symbol->setType(type);
    ast->symbol->setValue(value);
    scope()->addSymbol(symbol);

    if (auto enumSymbol = symbol_cast<EnumSymbol>(scope())) {
      auto parentScope = enumSymbol->parent();

      auto u =
          control()->newUsingDeclarationSymbol(parentScope, ast->identifierLoc);
      u->setName(ast->identifier);
      u->setTarget(symbol);
      parentScope->addSymbol(u);
    }

    return;
  }

  // in C mode

  if (auto enumSymbol = symbol_cast<EnumSymbol>(scope())) {
    auto parentScope = enumSymbol->parent();

    auto enumeratorSymbol =
        control()->newEnumeratorSymbol(parentScope, ast->identifierLoc);
    ast->symbol = enumeratorSymbol;

    enumeratorSymbol->setName(ast->identifier);
    enumeratorSymbol->setType(type);
    enumeratorSymbol->setValue(value);

    parentScope->addSymbol(enumeratorSymbol);
  }
}

auto Binder::declareTypeAlias(SourceLocation identifierLoc, TypeIdAST* typeId,
                              bool addSymbolToParentScope) -> TypeAliasSymbol* {
  auto symbol = control()->newTypeAliasSymbol(scope(), identifierLoc);

  auto name = unit_->identifier(identifierLoc);
  symbol->setName(name);

  if (typeId) symbol->setType(typeId->type);

  if (auto classType = type_cast<ClassType>(symbol->type())) {
    auto classSymbol = classType->symbol();
    if (!classSymbol->name()) {
      classSymbol->setName(symbol->name());
    }
  }

  if (auto enumType = type_cast<EnumType>(symbol->type())) {
    auto enumSymbol = enumType->symbol();
    if (!enumSymbol->name()) {
      enumSymbol->setName(symbol->name());
    }
  }

  if (auto scopedEnumType = type_cast<ScopedEnumType>(symbol->type())) {
    auto scopedEnumSymbol = scopedEnumType->symbol();
    if (!scopedEnumSymbol->name()) {
      scopedEnumSymbol->setName(symbol->name());
    }
  }

  if (addSymbolToParentScope) {
    auto scope = declaringScope();

    for (auto candidate : scope->find(name)) {
      if (auto existing = symbol_cast<TypeAliasSymbol>(candidate)) {
        auto canon = existing->canonical();
        canon->addRedeclaration(symbol);
        break;
      }
    }

    scope->addSymbol(symbol);
  }

  return symbol;
}

void Binder::bind(UsingDeclaratorAST* ast, Symbol* target) {
  if (auto u = symbol_cast<UsingDeclarationSymbol>(target)) {
    target = u->target();
  }

  const auto name = get_name(control(), ast->unqualifiedId);

  auto symbol = control()->newUsingDeclarationSymbol(
      scope(), ast->unqualifiedId->firstSourceLocation());

  ast->symbol = symbol;

  symbol->setName(name);
  symbol->setDeclarator(ast);
  symbol->setTarget(target);

  scope()->addSymbol(symbol);
}

void Binder::bind(BaseSpecifierAST* ast) {
  const auto checkTemplates = unit_->config().checkTypes;

  Symbol* symbol = nullptr;

  if (auto decltypeId = ast_cast<DecltypeIdAST>(ast->unqualifiedId)) {
    if (auto classType = type_cast<ClassType>(
            control()->remove_cv(decltypeId->decltypeSpecifier->type))) {
      symbol = classType->symbol();
    }
  } else {
    symbol =
        resolve(ast->nestedNameSpecifier, ast->unqualifiedId, checkTemplates);
  }

  // dealias
  if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType =
            type_cast<ClassType>(control()->remove_cv(typeAlias->type()))) {
      symbol = classType->symbol();
    }
  }

  if (!symbol || !symbol->isClass()) {
    if (symbol_cast<TypeParameterSymbol>(symbol)) {
      return;
    }
    if (!inTemplate()) {
      error(ast->unqualifiedId->firstSourceLocation(),
            "base class specifier must be a class");
    }
    return;
  }

  // Check if the base class is final
  if (auto baseClass = symbol_cast<ClassSymbol>(symbol)) {
    if (baseClass->isFinal()) {
      error(ast->unqualifiedId->firstSourceLocation(),
            std::format("cannot derive from 'final' class '{}'",
                        to_string(baseClass->name())));
    }
  }

  auto location = ast->unqualifiedId->firstSourceLocation();
  auto baseClassSymbol = control()->newBaseClassSymbol(scope(), location);
  ast->symbol = baseClassSymbol;

  baseClassSymbol->setVirtual(ast->isVirtual);
  baseClassSymbol->setSymbol(symbol);

  baseClassSymbol->setName(symbol->name());

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

void Binder::bind(NonTypeTemplateParameterAST* ast, int index, int depth) {
  auto symbol = control()->newNonTypeParameterSymbol(
      scope(), ast->declaration->firstSourceLocation());
  ast->symbol = symbol;

  symbol->setIndex(index);
  symbol->setDepth(depth);
  symbol->setName(ast->declaration->identifier);
  symbol->setParameterPack(ast->declaration->isPack);
  symbol->setObjectType(ast->declaration->type);
  scope()->addSymbol(symbol);
}

void Binder::bind(TypenameTypeParameterAST* ast, int index, int depth) {
  auto location = ast->identifier ? ast->identifierLoc : ast->classKeyLoc;

  auto symbol = control()->newTypeParameterSymbol(scope(), location, index,
                                                  depth, ast->isPack);
  ast->symbol = symbol;

  symbol->setName(ast->identifier);
  scope()->addSymbol(symbol);
}

void Binder::bind(ConstraintTypeParameterAST* ast, int index, int depth) {
  auto symbol =
      control()->newConstraintTypeParameterSymbol(scope(), ast->identifierLoc);
  symbol->setIndex(index);
  symbol->setDepth(depth);
  symbol->setName(ast->identifier);
  scope()->addSymbol(symbol);
}

void Binder::bind(TemplateTypeParameterAST* ast, int index, int depth) {
  std::vector<const Type*> parameters;

  for (auto param : ListView{ast->templateParameterList}) {
    if (param->symbol && param->symbol->type()) {
      parameters.push_back(param->symbol->type());
    }
  }

  auto symbol = control()->newTemplateTypeParameterSymbol(
      scope(), ast->templateLoc, index, depth, ast->isPack,
      std::move(parameters));

  symbol->setName(ast->identifier);

  ast->symbol = symbol;

  scope()->addSymbol(symbol);
}

void Binder::bind(ConceptDefinitionAST* ast) {
  auto templateParameters = currentTemplateParameters();

  auto symbol = control()->newConceptSymbol(scope(), ast->identifierLoc);
  symbol->setName(ast->identifier);

  declaringScope()->addSymbol(symbol);
}

void Binder::bind(LambdaExpressionAST* ast) {
  auto parentScope = declaringScope();
  auto symbol = control()->newLambdaSymbol(parentScope, ast->lbracketLoc);
  ast->symbol = symbol;

  setScope(symbol);
}

void Binder::complete(LambdaExpressionAST* ast) {
  if (auto params = ast->parameterDeclarationClause) {
    auto lambdaScope = ast->symbol;
    lambdaScope->addSymbol(params->functionParametersSymbol);
    setScope(params->functionParametersSymbol);
  } else {
    setScope(ast->symbol);
  }

  auto parentScope = ast->symbol->parent();
  parentScope->addSymbol(ast->symbol);

  const Type* returnType = control()->getAutoType();
  std::vector<const Type*> parameterTypes;
  bool isVariadic = false;

  if (auto params = ast->parameterDeclarationClause) {
    for (auto it = params->parameterDeclarationList; it; it = it->next) {
      auto paramType = it->value->type;

      if (control()->is_void(paramType)) {
        continue;
      }

      parameterTypes.push_back(paramType);
    }

    isVariadic = params->isVariadic;
  }

  bool isNoexcept = false;

  if (auto noexceptSpec =
          ast_cast<NoexceptSpecifierAST>(ast->exceptionSpecifier)) {
    if (!noexceptSpec->expression) {
      isNoexcept = true;
    } else {
      ASTInterpreter sem{unit_};
      auto value = sem.evaluate(noexceptSpec->expression);
      if (value.has_value()) {
        isNoexcept = sem.toBool(*value).value_or(false);
      }
    }
  }

  if (ast->trailingReturnType && ast->trailingReturnType->typeId) {
    returnType = ast->trailingReturnType->typeId->type;
  }

  auto funcType = control()->getFunctionType(
      returnType, std::move(parameterTypes), isVariadic, {}, {}, isNoexcept);
  ast->symbol->setType(funcType);

  if (is_parsing_cxx() && !inTemplate()) {
    auto closureName =
        control()->getIdentifier(std::format("__lambda_{}", lambdaCount_++));

    // Create the ClassSymbol for the closure type
    auto classSymbol = control()->newClassSymbol(parentScope, ast->lbracketLoc);
    classSymbol->setName(closureName);
    parentScope->addSymbol(classSymbol);

    // Create operator() FunctionSymbol
    auto operatorCallName = control()->getOperatorId(TokenKind::T_LPAREN);
    auto operatorFunc =
        control()->newFunctionSymbol(classSymbol, ast->lbracketLoc);
    operatorFunc->setName(operatorCallName);
    operatorFunc->setType(funcType);
    operatorFunc->setDefined(true);
    operatorFunc->setLanguageLinkage(LanguageKind::kCXX);
    classSymbol->addSymbol(operatorFunc);

    if (auto lambdaParams = ast->parameterDeclarationClause) {
      if (lambdaParams->functionParametersSymbol) {
        operatorFunc->addSymbol(lambdaParams->functionParametersSymbol);
      }
    }

    // Create implicit default constructor
    auto ctorSymbol =
        control()->newFunctionSymbol(classSymbol, ast->lbracketLoc);
    ctorSymbol->setName(closureName);
    ctorSymbol->setType(
        control()->getFunctionType(control()->getVoidType(), {}));
    ctorSymbol->setDefined(true);
    ctorSymbol->setDefaulted(true);
    ctorSymbol->setLanguageLinkage(LanguageKind::kCXX);
    classSymbol->addConstructor(ctorSymbol);

    if (ast->captureDefault == TokenKind::T_EOF_SYMBOL && !ast->captureList) {
      auto fptrType = control()->getPointerType(funcType);
      auto convFuncType = control()->getFunctionType(fptrType, {});
      auto convName = control()->getConversionFunctionId(fptrType);
      auto convFunc =
          control()->newFunctionSymbol(classSymbol, ast->lbracketLoc);
      convFunc->setName(convName);
      convFunc->setType(convFuncType);
      convFunc->setDefined(true);
      convFunc->setLanguageLinkage(LanguageKind::kCXX);
      classSymbol->addConversionFunction(convFunc);
    }

    classSymbol->setComplete(true);
    auto status = classSymbol->buildClassLayout(control());
    if (!status.has_value()) {
      error(ast->lbracketLoc, status.error());
    }

    ast->type = classSymbol->type();
    ast->valueCategory = ValueCategory::kPrValue;
  }
}

void Binder::completeLambdaBody(LambdaExpressionAST* ast) {
  auto classType = type_cast<ClassType>(ast->type);
  if (!classType) return;

  auto classSymbol = classType->symbol();
  auto ar = unit_->arena();

  // Find the operator() FunctionSymbol
  FunctionSymbol* operatorFunc = nullptr;
  for (auto member : classSymbol->members()) {
    if (auto func = symbol_cast<FunctionSymbol>(member)) {
      operatorFunc = func;
      break;
    }
  }
  if (!operatorFunc) return;

  ScopeSymbol* bodyScope = operatorFunc;
  for (auto member : operatorFunc->members()) {
    if (auto params = symbol_cast<FunctionParametersSymbol>(member)) {
      bodyScope = params;
      break;
    }
  }

  ASTRewriter rewriter{unit_, bodyScope, {}};
  auto reboundBody =
      ast_cast<CompoundStatementAST>(rewriter.statement(ast->statement));

  auto opId = OperatorFunctionIdAST::create(ar, TokenKind::T_LPAREN);

  auto idDecl = IdDeclaratorAST::create(ar);
  idDecl->unqualifiedId = opId;

  auto funcChunk = FunctionDeclaratorChunkAST::create(ar);
  if (ast->parameterDeclarationClause) {
    funcChunk->parameterDeclarationClause =
        ast->parameterDeclarationClause->clone(ar);
  }

  auto declarator = DeclaratorAST::create(
      ar, /*ptrOpList=*/nullptr, /*coreDeclarator=*/idDecl,
      /*declaratorChunkList=*/
      make_list_node<DeclaratorChunkAST>(ar, funcChunk));

  auto funcBody = CompoundStatementFunctionBodyAST::create(
      ar, /*memInitializerList=*/nullptr, reboundBody);

  auto funcDef = FunctionDefinitionAST::create(ar);
  funcDef->declarator = declarator;
  funcDef->functionBody = funcBody;
  funcDef->symbol = operatorFunc;
  operatorFunc->setDeclaration(funcDef);

  // Build FunctionDefinitionAST for the default constructor
  auto closureName = name_cast<Identifier>(classSymbol->name());
  for (auto ctor : classSymbol->constructors()) {
    if (ctor->declaration()) continue;  // already created

    auto ctorNameId = NameIdAST::create(ar, closureName);
    auto ctorIdDecl = IdDeclaratorAST::create(ar);
    ctorIdDecl->unqualifiedId = ctorNameId;
    auto ctorFuncChunk = FunctionDeclaratorChunkAST::create(ar);
    auto ctorDeclarator = DeclaratorAST::create(
        ar, /*ptrOpList=*/nullptr, /*coreDeclarator=*/ctorIdDecl,
        /*declaratorChunkList=*/
        make_list_node<DeclaratorChunkAST>(ar, ctorFuncChunk));
    auto ctorBody = DefaultFunctionBodyAST::create(ar);
    auto ctorDef = FunctionDefinitionAST::create(ar);
    ctorDef->declarator = ctorDeclarator;
    ctorDef->functionBody = ctorBody;
    ctorDef->symbol = ctor;
    ctor->setDeclaration(ctorDef);
  }
}

void Binder::bind(ParameterDeclarationClauseAST* ast) {
  ast->functionParametersSymbol =
      control()->newFunctionParametersSymbol(scope(), {});
}

void Binder::bind(UsingDirectiveAST* ast) {
  auto id = ast->unqualifiedId->identifier;

  auto namespaceSymbol =
      Lookup{scope()}.lookupNamespace(ast->nestedNameSpecifier, id);

  if (namespaceSymbol) {
    scope()->addUsingDirective(namespaceSymbol);
  } else {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("'{}' is not a namespace name", id->name()));
  }
}

void Binder::bind(TypeIdAST* ast, const Decl& decl) {
  ast->type = getDeclaratorType(unit_, ast->declarator, decl.specs.type());
}

auto Binder::declareTypedef(DeclaratorAST* declarator, const Decl& decl)
    -> TypeAliasSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());
  auto symbol = control()->newTypeAliasSymbol(scope(), decl.location());
  symbol->setName(name);
  symbol->setType(type);

  for (auto candidate : scope()->find(name)) {
    if (auto existing = symbol_cast<TypeAliasSymbol>(candidate)) {
      auto canon = existing->canonical();
      canon->addRedeclaration(symbol);
      break;
    }
  }

  scope()->addSymbol(symbol);

  if (auto classType = type_cast<ClassType>(symbol->type())) {
    auto classSymbol = classType->symbol();
    if (!classSymbol->name()) {
      classSymbol->setName(symbol->name());
    }
  }

  if (auto enumType = type_cast<EnumType>(symbol->type())) {
    auto enumSymbol = enumType->symbol();
    if (!enumSymbol->name()) {
      enumSymbol->setName(symbol->name());
    }
  }

  if (auto scopedEnumType = type_cast<ScopedEnumType>(symbol->type())) {
    auto scopedEnumSymbol = scopedEnumType->symbol();
    if (!scopedEnumSymbol->name()) {
      scopedEnumSymbol->setName(symbol->name());
    }
  }

  return symbol;
}

namespace {

auto collectDefaultArguments(DeclaratorAST* declarator)
    -> std::vector<Binder::DefaultArgumentInfo> {
  std::vector<Binder::DefaultArgumentInfo> result;

  if (!declarator) return result;

  auto functionDeclarator = getFunctionPrototype(declarator);
  if (!functionDeclarator) return result;

  auto params = functionDeclarator->parameterDeclarationClause;
  if (!params || !params->functionParametersSymbol) return result;

  for (auto member : params->functionParametersSymbol->members()) {
    auto param = symbol_cast<ParameterSymbol>(member);
    if (!param) {
      result.push_back({});
      continue;
    }

    result.push_back({.expression = param->defaultArgument(),
                      .location = param->location()});
  }

  return result;
}

void applyDefaultArguments(
    DeclaratorAST* declarator,
    const std::vector<Binder::DefaultArgumentInfo>& defaultArguments) {
  if (!declarator) return;

  auto functionDeclarator = getFunctionPrototype(declarator);
  if (!functionDeclarator) return;

  auto params = functionDeclarator->parameterDeclarationClause;
  if (!params || !params->functionParametersSymbol) return;

  size_t index = 0;
  for (auto member : params->functionParametersSymbol->members()) {
    auto param = symbol_cast<ParameterSymbol>(member);
    if (!param) {
      ++index;
      continue;
    }

    if (index >= defaultArguments.size()) {
      ++index;
      continue;
    }

    if (!param->defaultArgument()) {
      param->setDefaultArgument(defaultArguments[index].expression);
    }

    ++index;
  }
}

}  // namespace

void Binder::computeClassFlags(ClassSymbol* classSymbol) {
  // Compute isPolymorphic: class has virtual functions or a base is polymorphic
  bool polymorphic =
      views::any_function(classSymbol->members(),
                          [](FunctionSymbol* fn) { return fn->isVirtual(); });

  if (!polymorphic) {
    for (auto base : classSymbol->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (baseClass && baseClass->isPolymorphic()) {
        polymorphic = true;
        break;
      }
    }
  }
  classSymbol->setPolymorphic(polymorphic);

  bool abstract = views::any_function(
      classSymbol->members(),
      [](FunctionSymbol* fn) { return fn->isVirtual() && fn->isPure(); });

  if (!abstract) {
    auto overridesInClass = [&](FunctionSymbol* fn) -> bool {
      auto match = views::find_function(
          classSymbol->members(), [&](FunctionSymbol* member) {
            if (fn->isDestructor() && member->isDestructor()) return true;
            return fn->name() == member->name() &&
                   control()->is_same(fn->type(), member->type());
          });
      return match && !match->isPure();
    };

    for (auto base : classSymbol->baseClasses()) {
      if (abstract) break;
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass || !baseClass->isAbstract()) continue;

      auto unresolvedPure =
          views::find_function(baseClass->members(), [&](FunctionSymbol* fn) {
            return fn->isVirtual() && fn->isPure() && !overridesInClass(fn);
          });
      if (unresolvedPure) {
        abstract = true;
        break;
      }

      if (!abstract) {
        std::vector<ClassSymbol*> worklist;
        std::unordered_set<ClassSymbol*> visitedAncestors;
        for (auto bb : baseClass->baseClasses()) {
          auto bbc = symbol_cast<ClassSymbol>(bb->symbol());
          if (bbc && bbc->isAbstract() && visitedAncestors.insert(bbc).second)
            worklist.push_back(bbc);
        }

        auto overridesInBaseOrClass = [&](FunctionSymbol* fn) -> bool {
          auto match = views::find_function(
              baseClass->members(), [&](FunctionSymbol* m) {
                if (fn->isDestructor() && m->isDestructor()) return true;
                return fn->name() == m->name() &&
                       control()->is_same(fn->type(), m->type());
              });
          if (match && !match->isPure()) return true;
          return overridesInClass(fn);
        };

        while (!worklist.empty() && !abstract) {
          auto ancestor = worklist.back();
          worklist.pop_back();
          auto unresolvedAncestor = views::find_function(
              ancestor->members(), [&](FunctionSymbol* fn) {
                return fn->isVirtual() && fn->isPure() &&
                       !overridesInBaseOrClass(fn);
              });
          if (unresolvedAncestor) {
            abstract = true;
            break;
          }
          if (!abstract) {
            for (auto ab : ancestor->baseClasses()) {
              auto abc = symbol_cast<ClassSymbol>(ab->symbol());
              if (abc && abc->isAbstract() &&
                  visitedAncestors.insert(abc).second)
                worklist.push_back(abc);
            }
          }
        }
      }
    }
  }
  classSymbol->setAbstract(abstract);

  // Compute hasVirtualDestructor
  auto dtor = classSymbol->destructor();
  classSymbol->setHasVirtualDestructor(dtor && dtor->isVirtual());
}

void Binder::mergeDefaultArguments(FunctionSymbol* functionSymbol,
                                   DeclaratorAST* declarator) {
  if (!functionSymbol) return;

  auto collected = collectDefaultArguments(declarator);
  if (collected.empty()) return;

  auto canonical = functionSymbol->canonical();
  if (!canonical) canonical = functionSymbol;

  auto& known = defaultArguments_[canonical];
  if (known.size() < collected.size()) {
    known.resize(collected.size());
  }

  for (size_t index = 0; index < collected.size(); ++index) {
    const auto& incoming = collected[index];
    if (!incoming.expression) continue;

    auto& existing = known[index];
    if (existing.expression) {
      error(incoming.location, "redefinition of default argument");
      continue;
    }

    existing = incoming;
  }

  applyDefaultArguments(declarator, known);
}

struct [[nodiscard]] Binder::DeclareFunction {
  Binder& binder;
  DeclaratorAST* declarator = nullptr;
  const Decl& decl;
  // the symbol we're currently declaring, used for merging redeclarations
  FunctionDeclaratorChunkAST* functionDeclarator = nullptr;
  // the symbol we're currently declaring, used for merging redeclarations
  FunctionSymbol* functionSymbol = nullptr;
  // the shadowed function symbol
  FunctionSymbol* shadowedFunction = nullptr;

  auto control() const -> Control* { return binder.control(); }
  auto scope() const -> ScopeSymbol* { return binder.scope(); }
  auto lookup() const -> Lookup { return Lookup{scope()}; }

  auto isTemplateFunction() const -> bool {
    return scope()->isTemplateParameters();
  }

  auto isDestructor() const -> bool {
    return name_cast<DestructorId>(decl.getName()) != nullptr;
  }

  auto operator()() -> FunctionSymbol* {
    functionDeclarator = getFunctionPrototype(declarator);

    auto name = decl.getName();
    auto returnType = decl.getReturnType(scope());
    auto type = getDeclaratorType(binder.unit_, declarator, returnType);

    functionSymbol = control()->newFunctionSymbol(scope(), decl.location());
    functionSymbol->setName(name);
    functionSymbol->setType(type);

    checkDeclSpecifiers();
    checkExternalLinkageSpec();
    checkVirtualSpecifier();

    if (functionSymbol->isConstructor()) {
      checkConstructor();
      return functionSymbol;
    }

    checkRedeclaration();

    return functionSymbol;
  }

  void checkRedeclaration() {
    auto declaringScope = [&]() -> ScopeSymbol* {
      if (!functionSymbol->isFriend()) return binder.declaringScope();
      auto ds = binder.declaringScope();
      if (ds->isNamespace()) return ds;
      if (auto ns = ds->enclosingNamespace()) return ns;
      return ds;
    }();

    OverloadSetSymbol* overloadSet = nullptr;

    for (Symbol* candidate : declaringScope->find(functionSymbol->name())) {
      overloadSet = symbol_cast<OverloadSetSymbol>(candidate);
      if (overloadSet) break;

      if (auto otherFunction = symbol_cast<FunctionSymbol>(candidate)) {
        if (binder.is_parsing_c()) {
          auto canon = otherFunction->canonical();
          canon->addRedeclaration(functionSymbol);
          mergeRedeclaration();
          break;
        }

        overloadSet = control()->newOverloadSetSymbol(
            declaringScope, otherFunction->location());
        overloadSet->setName(otherFunction->name());
        overloadSet->addFunction(otherFunction);
        declaringScope->replaceSymbol(otherFunction, overloadSet);
        break;
      }
    }

    if (overloadSet) {
      bool isRedecl = false;

      for (auto existingFunction : overloadSet->functions()) {
        if (control()->is_same(existingFunction->type(),
                               functionSymbol->type())) {
          auto canon = existingFunction->canonical();
          canon->addRedeclaration(functionSymbol);
          mergeRedeclaration();
          isRedecl = true;
          break;
        }
      }

      if (!isRedecl) overloadSet->addFunction(functionSymbol);
    } else {
      if (functionSymbol->isFriend() && !declaringScope->isClass()) {
        functionSymbol->setHidden(true);
      }
      declaringScope->addSymbol(functionSymbol);
    }

    binder.mergeDefaultArguments(functionSymbol, declarator);
  }

  void checkConstructor() {
    auto enclosingClass = symbol_cast<ClassSymbol>(binder.scope());

    if (!enclosingClass) {
      cxx_runtime_error("constructor must be declared inside a class");
    }

    for (auto ctor : enclosingClass->constructors()) {
      if (control()->is_same(ctor->type(), functionSymbol->type())) {
        auto canon = ctor->canonical();
        canon->addRedeclaration(functionSymbol);
        mergeRedeclaration();
        break;
      }
    }

    binder.mergeDefaultArguments(functionSymbol, declarator);

    enclosingClass->addConstructor(functionSymbol);
  }

  void checkDeclSpecifiers() {
    binder.applySpecifiers(functionSymbol, decl.specs);
  }

  void checkExternalLinkageSpec() {
    if (binder.is_parsing_c()) {
      // in C mode, functions have C linkage
      functionSymbol->setLanguageLinkage(LanguageKind::kC);
      return;
    }

    if (scope()->isClass()) {
      // member functions always have C++ linkage
      functionSymbol->setLanguageLinkage(LanguageKind::kCXX);
      return;
    }

    // namespace-scope functions inherit the active language linkage,
    // which is kC inside extern "C" blocks and kCXX otherwise.
    functionSymbol->setLanguageLinkage(binder.languageLinkage_);
  }

  auto findOverriddenFunction(ClassSymbol* cls, FunctionSymbol* fn)
      -> FunctionSymbol* {
    std::unordered_set<ClassSymbol*> visited;
    return findOverriddenFunctionImpl(cls, fn, visited);
  }

  auto findOverriddenFunctionImpl(ClassSymbol* cls, FunctionSymbol* fn,
                                  std::unordered_set<ClassSymbol*>& visited)
      -> FunctionSymbol* {
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass || !visited.insert(baseClass).second) continue;

      for (auto member : baseClass->members() | views::virtual_functions) {
        if (fn->isDestructor() && member->isDestructor()) return member;

        // Non-destructors: match by name and signature
        if (fn->name() == member->name() &&
            control()->is_same(fn->type(), member->type())) {
          return member;
        }
      }

      if (auto result = findOverriddenFunctionImpl(baseClass, fn, visited))
        return result;
    }
    return nullptr;
  }

  void checkVirtualSpecifier() {
    // Propagate override/final from AST to symbol
    if (functionDeclarator->isOverride) functionSymbol->setOverride(true);
    if (functionDeclarator->isFinal) functionSymbol->setFinal(true);

    // Pure specifier (= 0) implies virtual
    if (functionDeclarator->isPure) {
      functionSymbol->setPure(true);
      functionSymbol->setVirtual(true);
    }

    auto enclosingClass = symbol_cast<ClassSymbol>(scope());

    // virtual/override/final/pure on non-member functions is invalid
    if (!enclosingClass) {
      if (functionSymbol->isVirtual()) {
        binder.error(functionSymbol->location(),
                     "'virtual' can only appear on non-static member "
                     "functions");
        functionSymbol->setVirtual(false);
      }
      if (functionSymbol->isOverride()) {
        binder.error(functionSymbol->location(),
                     "'override' can only appear on non-static member "
                     "functions");
      }
      if (functionSymbol->isFinal()) {
        binder.error(functionSymbol->location(),
                     "'final' can only appear on non-static member functions");
      }
      return;
    }

    // Constructors cannot be virtual
    if (functionSymbol->isConstructor()) return;

    // Look up the overridden virtual function in base classes
    auto overridden = findOverriddenFunction(enclosingClass, functionSymbol);

    if (overridden) {
      functionSymbol->setVirtual(true);

      // Check if the base function is final
      if (overridden->isFinal()) {
        binder.error(
            functionSymbol->location(),
            std::format("declaration of '{}' overrides a 'final' function",
                        to_string(functionSymbol->name())));
      }
    }

    if (functionSymbol->isOverride() && !overridden) {
      binder.error(functionSymbol->location(),
                   std::format("'{}' marked 'override' but does not override "
                               "any member function",
                               to_string(functionSymbol->name())));
    }

    if (functionSymbol->isFinal() && !functionSymbol->isVirtual()) {
      binder.error(functionSymbol->location(),
                   std::format("'{}' marked 'final' but is not virtual",
                               to_string(functionSymbol->name())));
    }
  }

  void mergeRedeclaration() {
    auto canonical = functionSymbol->canonical();
    if (!canonical || canonical == functionSymbol) return;

    if (!functionSymbol->isFriend() && canonical->isHidden()) {
      canonical->setHidden(false);
    }

    if (canonical->isStatic()) functionSymbol->setStatic(true);
    if (canonical->isExtern()) functionSymbol->setExtern(true);
    if (canonical->isFriend()) functionSymbol->setFriend(true);
    if (canonical->isConstexpr()) functionSymbol->setConstexpr(true);
    if (canonical->isConsteval()) functionSymbol->setConsteval(true);
    if (canonical->isInline()) functionSymbol->setInline(true);
    if (canonical->isVirtual()) functionSymbol->setVirtual(true);
    if (canonical->isExplicit()) functionSymbol->setExplicit(true);
    if (canonical->isOverride()) functionSymbol->setOverride(true);
    if (canonical->isFinal()) functionSymbol->setFinal(true);
    if (canonical->isPure()) functionSymbol->setPure(true);
    if (canonical->hasCLinkage())
      functionSymbol->setLanguageLinkage(LanguageKind::kC);

    if (functionSymbol->isInline()) canonical->setInline(true);
    if (functionSymbol->isConstexpr()) canonical->setConstexpr(true);
    if (functionSymbol->isConsteval()) canonical->setConsteval(true);
    if (functionSymbol->hasCLinkage())
      canonical->setLanguageLinkage(LanguageKind::kC);

    auto canonParams = canonical->functionParameters();
    auto redeclParams = functionSymbol->functionParameters();
    if (!canonParams || !redeclParams) return;

    auto canonIt = canonParams->members().begin();
    auto canonEnd = canonParams->members().end();
    auto redeclIt = redeclParams->members().begin();
    auto redeclEnd = redeclParams->members().end();

    for (; canonIt != canonEnd && redeclIt != redeclEnd;
         ++canonIt, ++redeclIt) {
      auto cp = symbol_cast<ParameterSymbol>(*canonIt);
      auto rp = symbol_cast<ParameterSymbol>(*redeclIt);
      if (!cp || !rp) continue;

      if (cp->defaultArgument() && rp->defaultArgument()) {
        binder.error(rp->location(), "redefinition of default argument");
        continue;
      }

      if (!cp->defaultArgument() && rp->defaultArgument()) {
        cp->setDefaultArgument(rp->defaultArgument());
        continue;
      }

      if (cp->defaultArgument() && !rp->defaultArgument()) {
        rp->setDefaultArgument(cp->defaultArgument());
      }
    }
  }
};

auto Binder::declareFunction(DeclaratorAST* declarator, const Decl& decl)
    -> FunctionSymbol* {
  return DeclareFunction{*this, declarator, decl}();
}

auto Binder::declareField(DeclaratorAST* declarator, const Decl& decl)
    -> FieldSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());
  auto fieldSymbol = control()->newFieldSymbol(scope(), decl.location());
  applySpecifiers(fieldSymbol, decl.specs);
  fieldSymbol->setName(name);
  fieldSymbol->setType(type);
  fieldSymbol->setMutable(decl.specs.isMutable);
  if (auto alignment = control()->memoryLayout()->alignmentOf(type)) {
    fieldSymbol->setAlignment(alignment.value());
  }

  if (decl.isBitField()) {
    fieldSymbol->setBitField(true);
  }

  scope()->addSymbol(fieldSymbol);
  return fieldSymbol;
}

auto Binder::declareVariable(DeclaratorAST* declarator, const Decl& decl,
                             bool addSymbolToParentScope) -> VariableSymbol* {
  auto name = decl.getName();
  auto symbol = control()->newVariableSymbol(scope(), decl.location());
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());
  applySpecifiers(symbol, decl.specs);
  symbol->setName(name);
  symbol->setType(type);
  if (addSymbolToParentScope) {
    auto scope = declaringScope();

    // Check for redeclaration of an existing variable with the same name
    for (auto candidate : scope->find(name)) {
      if (auto existing = symbol_cast<VariableSymbol>(candidate)) {
        if (!control()->is_same(existing->type(), symbol->type())) {
          error(
              symbol->location(),
              std::format("conflicting declaration of '{}'", to_string(name)));
          continue;
        }

        auto canon = existing->canonical();
        canon->addRedeclaration(symbol);
        break;
      }
    }

    scope->addSymbol(symbol);
  }
  return symbol;
}

auto Binder::declareMemberSymbol(DeclaratorAST* declarator, const Decl& decl)
    -> Symbol* {
  if (decl.specs.isTypedef) return declareTypedef(declarator, decl);

  if (getFunctionPrototype(declarator))
    return declareFunction(declarator, decl);

  return declareField(declarator, decl);
}

void Binder::applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setExtern(specs.isExtern);
  symbol->setFriend(specs.isFriend);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConsteval(specs.isConsteval);
  symbol->setInline(specs.isInline);
  symbol->setVirtual(specs.isVirtual);
  symbol->setExplicit(specs.isExplicit);
}

void Binder::applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setExtern(specs.isExtern);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

void Binder::applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs) {
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

auto Binder::resolveNestedNameSpecifier(Symbol* symbol) -> ScopeSymbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) return classSymbol;

  if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(symbol))
    return namespaceSymbol;

  if (auto enumSymbol = symbol_cast<EnumSymbol>(symbol)) return enumSymbol;

  if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(symbol))
    return scopedEnumSymbol;

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType = type_cast<ClassType>(typeAliasSymbol->type()))
      return classType->symbol();

    if (auto enumType = type_cast<EnumType>(typeAliasSymbol->type()))
      return enumType->symbol();

    if (auto scopedEnumType =
            type_cast<ScopedEnumType>(typeAliasSymbol->type()))
      return scopedEnumType->symbol();
  }

  return nullptr;
}

void Binder::bind(IdExpressionAST* ast) {
  if (!ast->unqualifiedId) {
    error(ast->firstSourceLocation(),
          "expected an unqualified identifier in id expression");
    return;
  }

  auto name = get_name(control(), ast->unqualifiedId);

  const Name* componentName = name;

  if (auto templateId = name_cast<TemplateId>(name)) {
    componentName = templateId->name();
  }

  if (ast->nestedNameSpecifier) {
    if (!ast->nestedNameSpecifier->symbol) {
      if (!inTemplate()) {
        error(ast->nestedNameSpecifier->firstSourceLocation(),
              "nested name specifier must be a class or namespace");
      }
      return;
    }
  }

  ast->symbol = Lookup{scope()}(ast->nestedNameSpecifier, componentName);

  if (unit_->config().checkTypes) {
    if (auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)) {
      // Try to find a template symbol to instantiate
      Symbol* templateSymbol = nullptr;

      if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
        templateSymbol = var;
      } else if (auto func = symbol_cast<FunctionSymbol>(ast->symbol)) {
        if (func->templateDeclaration()) templateSymbol = func;
      } else if (auto ovl = symbol_cast<OverloadSetSymbol>(ast->symbol)) {
        for (auto func : ovl->functions()) {
          if (!func->templateDeclaration()) continue;
          if (!templateSymbol) templateSymbol = func;
          auto instance = ASTRewriter::instantiate(
              unit_, templateId->templateArgumentList, func);
          if (instance) {
            ast->symbol = instance;
            templateSymbol = func;
            break;
          }
        }
        if (templateSymbol) return;
      }

      if (!templateSymbol) {
        if (!inTemplate()) {
          error(templateId->firstSourceLocation(),
                std::format("not a template"));
        }
      } else {
        auto instance = ASTRewriter::instantiate(
            unit_, templateId->templateArgumentList, templateSymbol);

        ast->symbol = instance;
      }
    }
  }
}

auto Binder::getFunction(ScopeSymbol* scope, const Name* name, const Type* type)
    -> FunctionSymbol* {
  auto parentScope = scope;

  while (parentScope && parentScope->isTransparent()) {
    parentScope = parentScope->parent();
  }

  if (auto parentClass = symbol_cast<ClassSymbol>(parentScope);
      parentClass && parentClass->name() == name) {
    for (auto ctor : parentClass->constructors()) {
      if (control()->is_same(ctor->type(), type)) {
        return ctor;
      }
    }
  }

  for (auto candidate : scope->find(name)) {
    if (auto function = symbol_cast<FunctionSymbol>(candidate)) {
      if (control()->is_same(function->type(), type)) {
        return function;
      }
    } else if (auto overloads = symbol_cast<OverloadSetSymbol>(candidate)) {
      for (auto function : overloads->functions()) {
        if (control()->is_same(function->type(), type)) {
          return function;
        }
      }
    }
  }

  return nullptr;
}

}  // namespace cxx
