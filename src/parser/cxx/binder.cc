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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cxx/binder.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <format>

namespace cxx {

Binder::Binder(TranslationUnit* unit) : unit_(unit) {}

auto Binder::translationUnit() const -> TranslationUnit* { return unit_; }

void Binder::setTranslationUnit(TranslationUnit* unit) { unit_ = unit; }

auto Binder::control() const -> Control* {
  return unit_ ? unit_->control() : nullptr;
}

auto Binder::reportErrors() const -> bool { return reportErrors_; }

void Binder::setReportErrors(bool reportErrors) {
  reportErrors_ = reportErrors;
}

void Binder::error(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->error(loc, std::move(message));
}

void Binder::warning(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  unit_->warning(loc, std::move(message));
}

auto Binder::inTemplate() const -> bool { return inTemplate_; }

auto Binder::currentTemplateParameters() const -> TemplateParametersSymbol* {
  auto templateParameters =
      symbol_cast<TemplateParametersSymbol>(scope()->owner());

  return templateParameters;
}

auto Binder::declaringScope() const -> Scope* {
  if (!scope_) return nullptr;
  if (!scope_->isTemplateParametersScope()) return scope_;
  return scope_->enclosingNonTemplateParametersScope();
}

auto Binder::scope() const -> Scope* { return scope_; }

void Binder::setScope(Scope* scope) {
  scope_ = scope;
  inTemplate_ = false;

  for (auto current = scope_; current; current = current->parent()) {
    if (current->isTemplateParametersScope()) {
      inTemplate_ = true;
      break;
    }
  }
}

void Binder::setScope(ScopedSymbol* symbol) { setScope(symbol->scope()); }

auto Binder::enterBlock(SourceLocation loc) -> BlockSymbol* {
  auto blockSymbol = control()->newBlockSymbol(scope_, loc);
  scope_->addSymbol(blockSymbol);
  setScope(blockSymbol->scope());
  return blockSymbol;
}

void Binder::bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpecs) {
  const auto underlyingType = underlyingTypeSpecs.getType();

  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->lbraceLoc;

  auto enumName = get_name(control(), ast->unqualifiedId);

  if (ast->classLoc) {
    auto enumSymbol = control()->newScopedEnumSymbol(scope(), location);
    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  } else {
    auto enumSymbol = control()->newEnumSymbol(scope(), location);
    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  }
}

void Binder::bind(ElaboratedTypeSpecifierAST* ast, DeclSpecs& declSpecs) {
  auto className = get_name(control(), ast->unqualifiedId);
  const auto location = ast->unqualifiedId->firstSourceLocation();

  const auto _ = ScopeGuard{this};

  if (ast->nestedNameSpecifier) {
    auto parent = ast->nestedNameSpecifier->symbol;

    if (parent && parent->isClassOrNamespace()) {
      setScope(static_cast<ScopedSymbol*>(parent));
    }
  }

  ClassSymbol* classSymbol = nullptr;

  if (scope()->isClassOrNamespaceScope()) {
    for (auto candidate : scope()->find(className) | views::classes) {
      classSymbol = candidate;
      break;
    }
  }

  if (!classSymbol) {
    const auto isUnion = ast->classKey == TokenKind::T_UNION;
    classSymbol = control()->newClassSymbol(scope(), location);

    classSymbol->setIsUnion(isUnion);
    classSymbol->setName(className);
    classSymbol->setTemplateParameters(currentTemplateParameters());
    declaringScope()->addSymbol(classSymbol);
  }

  ast->symbol = classSymbol;

  declSpecs.type = ast->symbol->type();
  declSpecs.setTypeSpecifier(ast);
}

void Binder::bind(ClassSpecifierAST* ast, DeclSpecs& declSpecs) {
  auto templateParameters = currentTemplateParameters();

  if (ast->nestedNameSpecifier) {
    auto parent = ast->nestedNameSpecifier->symbol;

    if (parent && parent->isClassOrNamespace()) {
      setScope(static_cast<ScopedSymbol*>(parent));
    }
  }

  auto className = get_name(control(), ast->unqualifiedId);
  auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId);

  auto location = ast->classLoc;
  if (templateId) {
    location = templateId->identifierLoc;
  } else if (ast->unqualifiedId) {
    location = ast->unqualifiedId->firstSourceLocation();
  }

  ClassSymbol* primaryTemplate = nullptr;

  if (templateId && scope()->isTemplateParametersScope()) {
    for (auto candidate : declaringScope()->find(className) | views::classes) {
      primaryTemplate = candidate;
      break;
    }

    if (!primaryTemplate) {
      error(location, std::format("specialization of undeclared template '{}'",
                                  templateId->identifier->name()));
    }
  }

  ClassSymbol* classSymbol = nullptr;

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
    const auto isUnion = ast->classKey == TokenKind::T_UNION;
    classSymbol = control()->newClassSymbol(scope(), location);
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

  classSymbol->setFinal(ast->isFinal);

  ast->symbol = classSymbol;

  declSpecs.setTypeSpecifier(ast);
  declSpecs.type = classSymbol->type();
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
  ast->type = getDeclaratorType(unit_, ast->declarator, decl.specs.getType());

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
  auto symbol = control()->newEnumeratorSymbol(scope(), ast->identifierLoc);
  ast->symbol = symbol;

  symbol->setName(ast->identifier);
  symbol->setType(type);
  ast->symbol->setValue(value);
  scope()->addSymbol(symbol);

  if (auto enumSymbol = symbol_cast<EnumSymbol>(scope()->owner())) {
    auto enumeratorSymbol =
        control()->newEnumeratorSymbol(scope(), ast->identifierLoc);
    enumeratorSymbol->setName(ast->identifier);
    enumeratorSymbol->setType(type);
    enumeratorSymbol->setValue(value);

    auto parentScope = enumSymbol->enclosingScope();
    parentScope->addSymbol(enumeratorSymbol);
  }
}

auto Binder::declareTypeAlias(SourceLocation identifierLoc, TypeIdAST* typeId)
    -> TypeAliasSymbol* {
  auto name = unit_->identifier(identifierLoc);
  auto symbol = control()->newTypeAliasSymbol(scope(), identifierLoc);
  symbol->setName(name);
  if (typeId) symbol->setType(typeId->type);
  symbol->setTemplateParameters(currentTemplateParameters());
  declaringScope()->addSymbol(symbol);
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
  Symbol* symbol = nullptr;

  if (auto decltypeId = ast_cast<DecltypeIdAST>(ast->unqualifiedId)) {
    if (auto classType = type_cast<ClassType>(
            control()->remove_cv(decltypeId->decltypeSpecifier->type))) {
      symbol = classType->symbol();
    }
  }

  if (auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId)) {
    symbol = Lookup{scope_}(ast->nestedNameSpecifier, nameId->identifier);
  }

  if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType =
            type_cast<ClassType>(control()->remove_cv(typeAlias->type()))) {
      symbol = classType->symbol();
    }
  }

  if (symbol) {
    auto location = ast->unqualifiedId->firstSourceLocation();
    auto baseClassSymbol = control()->newBaseClassSymbol(scope(), location);
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

  auto symbol = control()->newTypeParameterSymbol(scope(), location);
  ast->symbol = symbol;

  symbol->setIndex(index);
  symbol->setDepth(depth);
  symbol->setParameterPack(ast->isPack);
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
  auto symbol =
      control()->newTemplateTypeParameterSymbol(scope(), ast->templateLoc);

  ast->symbol = symbol;

  symbol->setIndex(index);
  symbol->setDepth(depth);
  symbol->setName(ast->identifier);
  symbol->setParameterPack(ast->isPack);
  scope()->addSymbol(symbol);
}

void Binder::bind(ConceptDefinitionAST* ast) {
  auto templateParameters = currentTemplateParameters();

  auto symbol = control()->newConceptSymbol(scope(), ast->identifierLoc);
  symbol->setName(ast->identifier);
  symbol->setTemplateParameters(templateParameters);

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
    auto lambdaScope = ast->symbol->scope();
    lambdaScope->addSymbol(params->functionParametersSymbol);
    setScope(params->functionParametersSymbol);
  } else {
    setScope(ast->symbol);
  }

  auto parentScope = ast->symbol->enclosingScope();
  parentScope->addSymbol(ast->symbol);
}

void Binder::bind(ParameterDeclarationClauseAST* ast) {
  ast->functionParametersSymbol =
      control()->newFunctionParametersSymbol(scope(), {});
}

void Binder::bind(UsingDirectiveAST* ast) {
  auto id = ast->unqualifiedId->identifier;

  NamespaceSymbol* namespaceSymbol =
      Lookup{scope()}.lookupNamespace(ast->nestedNameSpecifier, id);

  if (namespaceSymbol) {
    scope()->addUsingDirective(namespaceSymbol->scope());
  } else {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("'{}' is not a namespace name", id->name()));
  }
}

void Binder::bind(TypeIdAST* ast, const Decl& decl) {
  ast->type = getDeclaratorType(unit_, ast->declarator, decl.specs.getType());
}

auto Binder::declareTypedef(DeclaratorAST* declarator, const Decl& decl)
    -> TypeAliasSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());
  auto symbol = control()->newTypeAliasSymbol(scope(), decl.location());
  symbol->setName(name);
  symbol->setType(type);
  scope()->addSymbol(symbol);
  return symbol;
}

auto Binder::declareFunction(DeclaratorAST* declarator, const Decl& decl)
    -> FunctionSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());

  auto parentScope = scope();

  if (parentScope->isBlockScope()) {
    parentScope = parentScope->enclosingNamespaceScope();
  }

  auto functionSymbol = control()->newFunctionSymbol(scope(), decl.location());
  applySpecifiers(functionSymbol, decl.specs);
  functionSymbol->setName(name);
  functionSymbol->setType(type);
  functionSymbol->setTemplateParameters(currentTemplateParameters());

  if (isConstructor(functionSymbol)) {
    auto enclosingClass = symbol_cast<ClassSymbol>(scope()->owner());

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
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());
  auto fieldSymbol = control()->newFieldSymbol(scope(), decl.location());
  applySpecifiers(fieldSymbol, decl.specs);
  fieldSymbol->setName(name);
  fieldSymbol->setType(type);
  fieldSymbol->setMutable(decl.specs.isMutable);
  if (auto alignment = control()->memoryLayout()->alignmentOf(type)) {
    fieldSymbol->setAlignment(alignment.value());
  }
  scope()->addSymbol(fieldSymbol);
  return fieldSymbol;
}

auto Binder::declareVariable(DeclaratorAST* declarator, const Decl& decl)
    -> VariableSymbol* {
  auto name = decl.getName();
  auto symbol = control()->newVariableSymbol(scope(), decl.location());
  auto type = getDeclaratorType(unit_, declarator, decl.specs.getType());
  applySpecifiers(symbol, decl.specs);
  symbol->setName(name);
  symbol->setType(type);
  symbol->setTemplateParameters(currentTemplateParameters());
  declaringScope()->addSymbol(symbol);
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

auto Binder::isConstructor(Symbol* symbol) const -> bool {
  auto functionSymbol = symbol_cast<FunctionSymbol>(symbol);
  if (!functionSymbol) return false;
  if (!functionSymbol->enclosingScope()) return false;
  auto classSymbol =
      symbol_cast<ClassSymbol>(functionSymbol->enclosingScope()->owner());
  if (!classSymbol) return false;
  if (classSymbol->name() != functionSymbol->name()) return false;
  return true;
}

auto Binder::resolve(NestedNameSpecifierAST* nestedNameSpecifier,
                     UnqualifiedIdAST* unqualifiedId, bool canInstantiate)
    -> Symbol* {
  if (auto templateId = ast_cast<SimpleTemplateIdAST>(unqualifiedId)) {
    if (!canInstantiate) return nullptr;

    auto instance = instantiate(templateId);

    if (!is_type(instance)) return nullptr;

    return instance;
  }

  auto name = ast_cast<NameIdAST>(unqualifiedId);

  auto symbol =
      Lookup{scope()}.lookupType(nestedNameSpecifier, name->identifier);

  if (!is_type(symbol)) return nullptr;

  return symbol;
}

auto Binder::instantiate(SimpleTemplateIdAST* templateId) -> Symbol* {
  std::vector<TemplateArgument> args;
  for (auto it = templateId->templateArgumentList; it; it = it->next) {
    if (auto arg = ast_cast<TypeTemplateArgumentAST>(it->value)) {
      args.push_back(arg->typeId->type);
    } else {
      error(it->value->firstSourceLocation(),
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

  auto symbol = control()->instantiate(unit_, templateId->primaryTemplateSymbol,
                                       std::move(args));

  return symbol;
}

}  // namespace cxx
