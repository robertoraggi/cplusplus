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

Binder::Binder(TranslationUnit* unit) : unit_(unit) {}

auto Binder::translationUnit() const -> TranslationUnit* { return unit_; }

void Binder::setTranslationUnit(TranslationUnit* unit) { unit_ = unit; }

auto Binder::control() const -> Control* {
  return unit_ ? unit_->control() : nullptr;
}

auto Binder::is_parsing_c() const {
  return unit_->language() == LanguageKind::kC;
}

auto Binder::is_parsing_cxx() const {
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
    if (auto params = current->templateParameters()) {
      inTemplate_ = true;
      break;
    }
  }
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

    auto candidate =
        Lookup{scope()}.lookup(ast->nestedNameSpecifier, name, is_class);

    auto classSymbol = symbol_cast<ClassSymbol>(candidate);

    if (classSymbol && isDeclaration &&
        classSymbol->enclosingNonTemplateParametersScope() !=
            declaringScope()) {
      // the class is declared in a different scope
      classSymbol = nullptr;
    }

    if (!classSymbol) {
      const auto isUnion = ast->classKey == TokenKind::T_UNION;
      classSymbol = control()->newClassSymbol(scope(), location);

      classSymbol->setIsUnion(isUnion);
      classSymbol->setName(name);
      classSymbol->setTemplateDeclaration(declSpecs.templateHead);
      declaringScope()->addSymbol(classSymbol);

      classSymbol->setDeclaration(ast);
    }

    ast->symbol = classSymbol;
  }

  declSpecs.setTypeSpecifier(ast);

  if (ast->symbol) {
    declSpecs.setType(ast->symbol->type());
  }
}

void Binder::bind(ClassSpecifierAST* ast, DeclSpecs& declSpecs) {
  auto check_optional_nested_name_specifier = [&] {
    if (!ast->nestedNameSpecifier) return;

    auto parent = ast->nestedNameSpecifier->symbol;

    if (!parent || !parent->isClassOrNamespace()) {
      error(ast->nestedNameSpecifier->firstSourceLocation(),
            "nested name specifier must be a class or namespace");
      return;
    }

    setScope(static_cast<ScopeSymbol*>(parent));
  };

  auto check_template_specialization = [&] {
    auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);
    if (!templateId) return false;

    const auto location = templateId->identifierLoc;

    ClassSymbol* primaryTemplateSymbol = nullptr;

    for (auto candidate :
         declaringScope()->find(templateId->identifier) | views::classes) {
      primaryTemplateSymbol = candidate;
      break;
    }

    if (!primaryTemplateSymbol ||
        !primaryTemplateSymbol->templateParameters()) {
      error(location, std::format("specialization of undeclared template '{}'",
                                  templateId->identifier->name()));
      // return true;
    }

    std::vector<TemplateArgument> templateArguments;
    ClassSymbol* specialization = nullptr;

    if (primaryTemplateSymbol) {
      templateArguments = ASTRewriter::make_substitution(
          unit_, primaryTemplateSymbol->templateDeclaration(),
          templateId->templateArgumentList);

      specialization = primaryTemplateSymbol
                           ? symbol_cast<ClassSymbol>(
                                 primaryTemplateSymbol->findSpecialization(
                                     templateArguments))
                           : nullptr;

      if (specialization) {
        error(location, std::format("redefinition of specialization '{}'",
                                    templateId->identifier->name()));
        // return true;
      }
    }

    const auto isUnion = ast->classKey == TokenKind::T_UNION;

    auto classSymbol = control()->newClassSymbol(declaringScope(), location);
    ast->symbol = classSymbol;

    classSymbol->setIsUnion(isUnion);
    classSymbol->setName(templateId->identifier);
    ast->symbol->setDeclaration(ast);
    ast->symbol->setFinal(ast->isFinal);

    // if (declSpecs.templateHead) {
    //   warning(location, "setting template head");
    //   ast->symbol->setTemplateDeclaration(declSpecs.templateHead);
    // }

    declSpecs.setTypeSpecifier(ast);
    declSpecs.setType(ast->symbol->type());

    if (primaryTemplateSymbol) {
      primaryTemplateSymbol->addSpecialization(std::move(templateArguments),
                                               classSymbol);
    }

    return true;
  };

  check_optional_nested_name_specifier();

  if (check_template_specialization()) return;

  // get the component anme
  const Identifier* className = nullptr;
  if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId))
    className = nameId->identifier;

  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->classLoc;

  ClassSymbol* classSymbol = nullptr;

  if (className) {
    for (auto candidate : declaringScope()->find(className) | views::classes) {
      classSymbol = candidate;
      break;
    }
  }

  if (classSymbol && classSymbol->isComplete()) {
    // not a template-id, but a class with the same name already exists
    error(location,
          std::format("redefinition of class '{}'", to_string(className)));
    classSymbol = nullptr;
  }

  if (!classSymbol) {
    const auto isUnion = ast->classKey == TokenKind::T_UNION;
    classSymbol = control()->newClassSymbol(scope(), location);
    classSymbol->setIsUnion(isUnion);
    classSymbol->setName(className);

    declaringScope()->addSymbol(classSymbol);
  }

  ast->symbol = classSymbol;

  ast->symbol->setDeclaration(ast);

  if (declSpecs.templateHead) {
    ast->symbol->setTemplateDeclaration(declSpecs.templateHead);
  }

  ast->symbol->setFinal(ast->isFinal);

  declSpecs.setTypeSpecifier(ast);
  declSpecs.setType(ast->symbol->type());
}

void Binder::complete(ClassSpecifierAST* ast) {
  if (!inTemplate()) {
    auto status = ast->symbol->buildClassLayout(control());
    if (!status.has_value()) {
      error(ast->symbol->location(), status.error());
    }
  }

  ast->symbol->setComplete(true);
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
    declaringScope()->addSymbol(symbol);
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
    error(ast->unqualifiedId->firstSourceLocation(),
          "base class specifier must be a class");
    return;
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

  auto type = control()->getFunctionType(returnType, std::move(parameterTypes),
                                         isVariadic, {}, {}, isNoexcept);
  ast->symbol->setType(type);
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

auto Binder::declareFunction(DeclaratorAST* declarator, const Decl& decl)
    -> FunctionSymbol* {
  auto name = decl.getName();

  auto returnType = decl.getReturnType(scope());

  auto type = getDeclaratorType(unit_, declarator, returnType);

  auto parentScope = scope();

  if (parentScope->isBlock()) {
    parentScope = parentScope->enclosingNamespace();
  }

  auto functionSymbol = control()->newFunctionSymbol(scope(), decl.location());

  if (is_parsing_c()) {
    functionSymbol->setHasCxxLinkage(false);
  }

  applySpecifiers(functionSymbol, decl.specs);
  functionSymbol->setName(name);
  functionSymbol->setType(type);

  if (functionSymbol->isConstructor()) {
    auto enclosingClass = symbol_cast<ClassSymbol>(scope());

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
      overloadSet = control()->newOverloadSetSymbol(scope, {});
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
    declaringScope()->addSymbol(symbol);
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

auto Binder::resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                     UnqualifiedIdAST* unqualifiedId, bool checkTemplates)
    -> Symbol* {
  if (auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId)) {
    if (!checkTemplates) return templateId->symbol;

    if (auto classSymbol = symbol_cast<ClassSymbol>(templateId->symbol)) {
      // todo: delay
      auto instance = ASTRewriter::instantiate(
          unit_, templateId->templateArgumentList, classSymbol);

      return instance;
    }

    if (auto typeAliasSymbol =
            symbol_cast<TypeAliasSymbol>(templateId->symbol)) {
      auto instance = ASTRewriter::instantiate(
          unit_, templateId->templateArgumentList, typeAliasSymbol);

      return instance;
    }

    return templateId->symbol;
  }

  auto name = ast_cast<NameIdAST>(unqualifiedId);

  auto symbol =
      Lookup{scope()}.lookupType(nestedNameSpecifier, name->identifier);

  if (!is_type(symbol)) return nullptr;

  return symbol;
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
      error(ast->nestedNameSpecifier->firstSourceLocation(),
            "nested name specifier must be a class or namespace");
      return;
    }
  }

  ast->symbol = Lookup{scope()}(ast->nestedNameSpecifier, componentName);

  if (unit_->config().checkTypes) {
    if (auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)) {
      auto var = symbol_cast<VariableSymbol>(ast->symbol);

      if (!var) {
        error(templateId->firstSourceLocation(), std::format("not a template"));
      } else {
        auto instance = ASTRewriter::instantiate(
            unit_, templateId->templateArgumentList, var);

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
