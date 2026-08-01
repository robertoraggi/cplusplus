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

#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/ast_visitor.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
Binder::Binder(TranslationUnit* unit) : unit_(unit), traits(unit) {
  languageLinkage_ = unit_->language();
}

auto Binder::translationUnit() const -> TranslationUnit* { return unit_; }

auto Binder::control() const -> Control* { return unit_->control(); }

auto Binder::isC() const -> bool {
  return unit_->language() == LanguageKind::kC;
}

auto Binder::isCxx() const -> bool {
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

void Binder::note(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  if (!unit_->config().checkTypes) return;
  unit_->note(loc, std::move(message));
}

auto Binder::inTemplate() const -> bool {
  return inTemplate_ || explicitTemplateHeadDepth_ > 0;
}

void Binder::enterExplicitTemplateHead() { ++explicitTemplateHeadDepth_; }

void Binder::leaveExplicitTemplateHead() { --explicitTemplateHeadDepth_; }

void Binder::finishAutoReturnType(FunctionSymbol* functionSymbol) {
  if (!functionSymbol) return;
  auto funcType = type_cast<FunctionType>(functionSymbol->type());
  if (!funcType) return;
  if (!type_cast<AutoType>(funcType->returnType())) return;

  auto newFuncType = control()->getFunctionType(
      control()->getVoidType(),
      std::vector<const Type*>(funcType->parameterTypes().begin(),
                               funcType->parameterTypes().end()),
      funcType->isVariadic(), funcType->cvQualifiers(),
      funcType->refQualifier(), funcType->isNoexcept());
  functionSymbol->setType(newFuncType);
}

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

auto Binder::instantiationLoc() const -> SourceLocation {
  return instantiationLoc_;
}

void Binder::setInstantiationLoc(SourceLocation loc) {
  instantiationLoc_ = loc;
}

auto Binder::declaringScope() const -> ScopeSymbol* {
  if (!scope_) return nullptr;
  if (!scope_->isTemplateParameters()) return scope_;
  return scope_->enclosingNonTemplateParametersScope();
}

auto Binder::scopeForBlockDecl(ScopeSymbol* scope) const -> ScopeSymbol* {
  if (scope && scope->isBlock()) {
    if (auto ns = scope->enclosingNamespace()) return ns;
  }
  return scope;
}

void Binder::injectUsing(ScopeSymbol* scope, const Name* name, Symbol* target,
                         SourceLocation loc) {
  auto u = control()->newUsingDeclarationSymbol(scope, loc);
  u->setName(name);
  u->setTarget(target);
  scope->addSymbol(u);
}

auto Binder::scope() const -> ScopeSymbol* { return scope_; }

void Binder::setScope(ScopeSymbol* scope) {
  scope_ = scope;
  inTemplate_ = isEnclosedInTemplate(scope_);
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

  if (ast->classLoc && isCxx()) {
    auto enumSymbol =
        control()->newScopedEnumSymbol(declaringScope(), location);
    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  } else {
    if (isC() && ast->classLoc) {
      error(ast->classLoc, "scoped enums are not allowed in C");
    }

    auto enumSymbol = control()->newEnumSymbol(declaringScope(), location);

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
                  bool isDeclaration, Symbol* unqualifiedCandidate) {
  const auto _ = ScopeGuard{this};

  if (ast->nestedNameSpecifier) {
    auto parent = ast->nestedNameSpecifier->symbol;

    if (!parent || !parent->isClassOrNamespace()) {
      (void)reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier);
      return;
    }

    setScope(parent->asScopeSymbol());
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

    auto candidate = [&]() -> Symbol* {
      if (declSpecs.isFriend) {
        for (auto s = targetScope; s; s = s->parent()) {
          if (auto found = qualifiedLookup(s, name, is_class)) return found;
        }
        return nullptr;
      }
      if (ast->nestedNameSpecifier)
        return qualifiedLookup(ast->nestedNameSpecifier->symbol, name,
                               is_class);
      return unqualifiedCandidate;
    }();

    auto classSymbol = symbol_cast<ClassSymbol>(candidate);

    if (classSymbol && isDeclaration &&
        classSymbol->enclosingNonTemplateParametersScope() != targetScope) {
      classSymbol = nullptr;
    }

    if (!classSymbol) {
      const auto isUnion = ast->classKey == TokenKind::T_UNION;
      classSymbol = control()->newClassSymbol(targetScope, location);

      classSymbol->setIsUnion(isUnion);
      classSymbol->setName(name);
      classSymbol->setTemplateDeclaration(declSpecs.templateHead);
      if (declSpecs.templateHead)
        classSymbol->setTemplateParameters(declSpecs.templateHead->symbol);
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

  if (traits.is_array(ast->type))
    ast->type = traits.add_pointer(traits.remove_extent(ast->type));
  else if (traits.is_function(ast->type))
    ast->type = traits.add_pointer(ast->type);
  else if (traits.is_scalar(ast->type))
    ast->type = traits.remove_cv(ast->type);

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
    auto parameterLoc = decl.location();
    if (!parameterLoc) parameterLoc = ast->firstSourceLocation();

    auto parameterSymbol = control()->newParameterSymbol(scope_, parameterLoc);
    parameterSymbol->setName(ast->identifier);
    parameterSymbol->setType(ast->type);
    parameterSymbol->setDefaultArgument(ast->expression);
    scope_->addSymbol(parameterSymbol);
  }
}

void Binder::bind(DecltypeSpecifierAST* ast) {
  auto namedSymbol = [&]() -> Symbol* {
    if (auto id = ast_cast<IdExpressionAST>(ast->expression)) return id->symbol;
    if (auto member = ast_cast<MemberExpressionAST>(ast->expression))
      return member->symbol;
    return nullptr;
  }();

  if (symbol_cast<OverloadSetSymbol>(namedSymbol)) {
    ast->type = ast->expression->type;
  } else if (namedSymbol) {
    ast->type = namedSymbol->type();
  } else if (ast->expression && ast->expression->type) {
    if (isCxx() && is_lvalue(ast->expression)) {
      ast->type = traits.add_lvalue_reference(ast->expression->type);
    } else if (isCxx() && is_xvalue(ast->expression)) {
      ast->type = traits.add_rvalue_reference(ast->expression->type);
    } else {
      ast->type = ast->expression->type;
    }
  }
}

void Binder::bind(EnumeratorAST* ast, const Type* type,
                  std::optional<ConstValue> value) {
  if (isCxx()) {
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
  auto symbol = control()->newTypeAliasSymbol(declaringScope(), identifierLoc);

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
    bool hasConflict = false;

    auto should_report_conflict = [&](SourceLocation loc) {
      if (auto preprocessor = unit_->preprocessor()) {
        const auto& token = unit_->tokenAt(loc);
        if (token) return !preprocessor->isSystemHeader(token.fileId());
      }
      return true;
    };

    auto aliases_named_type_symbol = [&](Symbol* candidate) {
      if (isC() && (symbol_cast<ClassSymbol>(candidate) ||
                    symbol_cast<EnumSymbol>(candidate)))
        return true;

      if (auto classSymbol = symbol_cast<ClassSymbol>(candidate)) {
        if (auto classType = type_cast<ClassType>(symbol->type())) {
          return classType->symbol() == classSymbol;
        }
      }

      if (auto enumSymbol = symbol_cast<EnumSymbol>(candidate)) {
        if (auto enumType = type_cast<EnumType>(symbol->type())) {
          return enumType->symbol() == enumSymbol;
        }
      }

      if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(candidate)) {
        if (auto scopedEnumType = type_cast<ScopedEnumType>(symbol->type())) {
          return scopedEnumType->symbol() == scopedEnumSymbol;
        }
      }

      return false;
    };

    for (auto candidate : scope->find(name)) {
      if (auto existing = symbol_cast<TypeAliasSymbol>(candidate)) {
        if (existing->type() && symbol->type() &&
            !traits.is_same(existing->type(), symbol->type())) {
          if (should_report_conflict(identifierLoc)) {
            error(identifierLoc, std::format("conflicting declaration of '{}'",
                                             to_string(name)));
            hasConflict = true;
          }
          break;
        }

        auto canon = existing->canonical();
        canon->addRedeclaration(symbol);
        break;
      } else {
        if (aliases_named_type_symbol(candidate)) continue;

        if (should_report_conflict(identifierLoc)) {
          error(identifierLoc, std::format("conflicting declaration of '{}'",
                                           to_string(name)));
          hasConflict = true;
        }
        break;
      }
    }

    if (!hasConflict) {
      scope->addSymbol(symbol);
    }
  }

  return symbol;
}

void Binder::bind(UsingDeclaratorAST* ast, Symbol* target) {
  auto makeDependentTypeTarget = [&]() -> Symbol* {
    if (!ast->typenameLoc) return nullptr;
    auto alias = control()->newTypeAliasSymbol(
        scope(), ast->unqualifiedId->firstSourceLocation());
    alias->setName(get_name(control(), ast->unqualifiedId));
    alias->setType(control()->getUnresolvedNameType(
        unit_, ast->nestedNameSpecifier, ast->unqualifiedId));
    return alias;
  };

  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    if (reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier)) return;
  }

  const bool dependentQualifier =
      inTemplate() && isDependent(unit_, ast->nestedNameSpecifier);

  if (dependentQualifier) target = makeDependentTypeTarget();

  if (auto u = symbol_cast<UsingDeclarationSymbol>(target)) {
    target = u->target();
  }

  if (!target && !dependentQualifier) {
    if (!inTemplate()) {
      auto missingName = get_name(control(), ast->unqualifiedId);
      error(ast->unqualifiedId->firstSourceLocation(),
            std::format("using declaration refers to unresolved name '{}'",
                        to_string(missingName)));
      return;
    }
    target = makeDependentTypeTarget();
  }

  const auto name = get_name(control(), ast->unqualifiedId);

  auto symbol = control()->newUsingDeclarationSymbol(
      scope(), ast->unqualifiedId->firstSourceLocation());

  ast->symbol = symbol;

  symbol->setName(name);
  symbol->setDeclarator(ast);
  symbol->setTarget(target);

  const auto joinsAnOverloadSet =
      !symbol->introducedFunctions().empty() &&
      std::ranges::any_of(scope()->find(name), [](Symbol* candidate) {
        return symbol_cast<FunctionSymbol>(candidate) ||
               symbol_cast<OverloadSetSymbol>(candidate);
      });

  if (!joinsAnOverloadSet) {
    scope()->addSymbol(symbol);
    return;
  }

  overloadSetFor(scope(), name, symbol->location())
      ->addUsingDeclaration(symbol);
}

void Binder::bind(BaseSpecifierAST* ast, Symbol* resolvedType) {
  const auto checkTemplates = unit_->config().checkTypes;

  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    (void)reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier);
    return;
  }

  Symbol* symbol = nullptr;

  if (auto decltypeId = ast_cast<DecltypeIdAST>(ast->unqualifiedId)) {
    if (auto classType = type_cast<ClassType>(
            traits.remove_cv(decltypeId->decltypeSpecifier->type))) {
      symbol = classType->symbol();
    }
  } else {
    symbol = resolve(ast->nestedNameSpecifier, ast->unqualifiedId,
                     checkTemplates, resolvedType);
  }

  if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType =
            type_cast<ClassType>(traits.remove_cv(typeAlias->type()))) {
      symbol = classType->symbol();
    }
  }

  if (!symbol || !symbol->isClass()) {
    if (!symbol) {
      if (!inTemplate()) {
        auto baseName = get_name(control(), ast->unqualifiedId);
        error(ast->unqualifiedId->firstSourceLocation(),
              std::format("unknown base class '{}'", to_string(baseName)));
      }
      return;
    }

    if (auto typeParam = symbol_cast<TypeParameterSymbol>(symbol)) {
      auto location = ast->unqualifiedId->firstSourceLocation();
      auto baseClassSymbol = control()->newBaseClassSymbol(scope(), location);
      ast->symbol = baseClassSymbol;

      baseClassSymbol->setVirtual(ast->isVirtual);
      baseClassSymbol->setSymbol(typeParam);
      baseClassSymbol->setName(typeParam->name());

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
      }
      return;
    }
    if (!inTemplate()) {
      error(ast->unqualifiedId->firstSourceLocation(),
            "base class specifier must be a class");
    }
    return;
  }

  if (auto baseClass = symbol_cast<ClassSymbol>(symbol)) {
    traits.requireCompleteClass(baseClass);
  }

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

  auto symbol =
      control()->newConceptSymbol(declaringScope(), ast->identifierLoc);
  symbol->setName(ast->identifier);
  if (templateParameters) {
    symbol->setTemplateParameters(templateParameters);
  }
  ast->symbol = symbol;

  declaringScope()->addSymbol(symbol);
}

void Binder::bind(DeductionGuideAST* ast) {
  auto templateParameters = currentTemplateParameters();

  auto symbol =
      control()->newDeductionGuideSymbol(declaringScope(), ast->identifierLoc);
  symbol->setName(ast->identifier);
  if (templateParameters) {
    symbol->setTemplateParameters(templateParameters);
  }
  if (ast->explicitSpecifier) {
    symbol->setExplicit(true);
  }
  ast->symbol = symbol;

  std::vector<const Type*> parameterTypes;
  bool isVariadic = false;

  if (auto params = ast->parameterDeclarationClause) {
    for (auto it = params->parameterDeclarationList; it; it = it->next) {
      auto paramType = it->value ? it->value->type : nullptr;
      if (paramType && !type_cast<VoidType>(paramType))
        parameterTypes.push_back(paramType);
    }
    isVariadic = params->isVariadic;
  }

  auto primaryTemplate = ast->templateId
                             ? symbol_cast<ClassSymbol>(ast->templateId->symbol)
                             : nullptr;
  if (!primaryTemplate) return;

  ClassSymbol* deducedClassSymbol = primaryTemplate;

  if (auto templateDecl = primaryTemplate->templateDeclaration();
      templateDecl && ast->templateId->templateArgumentList) {
    auto templateArgs =
        Substitution(unit_, templateDecl, ast->templateId->templateArgumentList)
            .templateArguments();

    if (!templateArgs.empty()) {
      if (auto cached = primaryTemplate->findSpecialization(templateArgs)) {
        deducedClassSymbol = symbol_cast<ClassSymbol>(cached);
      } else {
        auto parentScope =
            primaryTemplate->enclosingNonTemplateParametersScope();
        auto spec =
            control()->newClassSymbol(parentScope, primaryTemplate->location());
        spec->setName(primaryTemplate->name());
        spec->setType(control()->getClassType(spec));
        primaryTemplate->addSpecialization(std::move(templateArgs), spec);
        for (auto& s : primaryTemplate->mutableSpecializations()) {
          if (s.symbol == spec) {
            s.pendingArgumentList = ast->templateId->templateArgumentList;
            s.pendingInstantiationLoc = ast->templateId->identifierLoc;
            s.isPendingInstantiation = true;
            break;
          }
        }
        deducedClassSymbol = spec;
      }
    }
  }

  const Type* returnType =
      deducedClassSymbol ? deducedClassSymbol->type() : nullptr;
  if (!returnType) return;

  auto funcType = control()->getFunctionType(
      returnType, std::move(parameterTypes), isVariadic, {}, {}, false);
  symbol->setType(funcType);

  primaryTemplate->addDeductionGuide(symbol);
}

auto Binder::lookupCaptureName(ScopeSymbol* scope, const Name* name)
    -> Symbol* {
  for (auto current = scope; current; current = current->parent()) {
    for (auto candidate : current->find(name)) return candidate;
  }
  return nullptr;
}

auto Binder::isCapturableLocalEntity(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (symbol_cast<ParameterSymbol>(symbol)) return true;
  if (symbol_cast<ParameterPackSymbol>(symbol)) return true;
  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var) return false;
  if (var->isStatic() || var->isExtern() || var->isThreadLocal()) return false;
  return var->enclosingFunction() != nullptr;
}

auto Binder::checkCapturedEntity(Symbol* symbol, const Identifier* identifier,
                                 SourceLocation loc) -> bool {
  if (isCapturableLocalEntity(symbol)) return true;

  if (symbol_cast<FieldSymbol>(symbol)) {
    error(loc, std::format("class member '{}' cannot appear in capture list "
                           "as it is not a variable",
                           identifier->name()));
  } else if (symbol_cast<VariableSymbol>(symbol)) {
    error(loc, std::format("'{}' cannot be captured because it does not have "
                           "automatic storage duration",
                           identifier->name()));
  } else {
    error(loc, std::format("'{}' in capture list does not name a variable",
                           identifier->name()));
  }

  return false;
}

auto Binder::enclosingThisType(ScopeSymbol* scope) -> const Type* {
  for (auto current = scope; current; current = current->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(current)) {
      if (classSymbol->isClosureType()) {
        if (auto capturedThisField = classSymbol->capturedThisField()) {
          return capturedThisField->type();
        }
        continue;
      }
      return control()->getPointerType(classSymbol->type());
    }

    if (auto functionSymbol = symbol_cast<FunctionSymbol>(current)) {
      auto classSymbol = symbol_cast<ClassSymbol>(functionSymbol->parent());
      if (!classSymbol) return nullptr;

      if (classSymbol->isClosureType()) {
        if (auto capturedThisField = classSymbol->capturedThisField()) {
          return capturedThisField->type();
        }
        continue;
      }

      auto functionType = type_cast<FunctionType>(functionSymbol->type());
      const auto cv =
          functionType ? functionType->cvQualifiers() : CvQualifiers::kNone;
      if (cv != CvQualifiers::kNone) {
        auto elementType = control()->getQualType(classSymbol->type(), cv);
        return control()->getPointerType(elementType);
      }
      return control()->getPointerType(classSymbol->type());
    }
  }
  return nullptr;
}

namespace {
[[nodiscard]] auto namesNonStaticMember(Symbol* symbol) -> bool {
  if (!symbol) return false;

  if (auto field = symbol_cast<FieldSymbol>(symbol)) return !field->isStatic();

  return std::ranges::any_of(
      views::each_function(symbol), [](FunctionSymbol* function) {
        return function->isImplicitObjectMemberFunction();
      });
}

[[nodiscard]] auto formsPointerToMember(UnaryExpressionAST* ast) -> bool {
  if (ast->op != TokenKind::T_AMP) return false;
  auto id = ast_cast<IdExpressionAST>(ast->expression);
  return id && id->nestedNameSpecifier && namesNonStaticMember(id->symbol);
}

struct ThisUseFinder : ASTVisitor {
  bool found = false;

  void visit(ThisExpressionAST*) override { found = true; }
  void visit(DerefThisLambdaCaptureAST*) override { found = true; }

  void visit(IdExpressionAST* ast) override {
    if (namesNonStaticMember(ast->symbol)) found = true;
    ASTVisitor::visit(ast);
  }

  void visit(UnaryExpressionAST* ast) override {
    if (formsPointerToMember(ast)) return;
    ASTVisitor::visit(ast);
  }

  void visit(LambdaExpressionAST* ast) override {
    for (auto capture : ListView{ast->captureList}) accept(capture);
  }
};

[[nodiscard]] auto isUsableInConstantExpressions(Symbol* symbol) -> bool {
  auto var = symbol_cast<VariableSymbol>(symbol);
  if (!var || !var->constValue().has_value()) return false;
  if (var->isConstexpr()) return true;
  auto qualType = type_cast<QualType>(var->type());
  return qualType && qualType->isConst();
}

struct OdrUsedLocalFinder : ASTVisitor {
  std::vector<IdExpressionAST*> uses;

  void visit(IdExpressionAST* ast) override {
    if (ast->nestedNameSpecifier) return;
    if (ast->symbol) uses.push_back(ast);
  }

  void visit(ImplicitCastExpressionAST* ast) override {
    if (ast->castKind == ImplicitCastKind::kLValueToRValueConversion) {
      auto id = ast_cast<IdExpressionAST>(ast->expression);
      if (id && isUsableInConstantExpressions(id->symbol)) return;
    }
    ASTVisitor::visit(ast);
  }

  void visit(SizeofExpressionAST*) override {}
  void visit(SizeofPackExpressionAST*) override {}
  void visit(AlignofExpressionAST*) override {}
  void visit(NoexceptExpressionAST*) override {}
  void visit(DecltypeSpecifierAST*) override {}
  void visit(RequiresExpressionAST*) override {}
};
}  // namespace

auto Binder::abiTags(List<AttributeSpecifierAST*>* attributes)
    -> std::vector<const Identifier*> {
  std::vector<const Identifier*> tags;

  auto namesAbiTag = [&](SourceLocation loc) {
    auto id = unit_->identifier(loc);
    return id && (id->name() == "abi_tag" || id->name() == "__abi_tag__");
  };

  bool foundAbiTag = false;

  auto collectTags = [&](SourceLocation begin, SourceLocation end) {
    bool inAbiTagArguments = false;

    for (auto loc = begin; loc && loc < end; loc = loc.next()) {
      if (foundAbiTag) return;

      const auto tokenKind = unit_->tokenKind(loc);

      if (tokenKind == TokenKind::T_IDENTIFIER) {
        inAbiTagArguments = namesAbiTag(loc);
        continue;
      }

      if (!inAbiTagArguments) continue;

      if (tokenKind == TokenKind::T_RPAREN) {
        foundAbiTag = true;
        return;
      }

      if (tokenKind != TokenKind::T_STRING_LITERAL) continue;

      auto literal = unit_->literal(loc);
      if (!literal) continue;

      auto components = StringLiteral::Components::from(
          literal->value(), StringLiteralEncoding::kNone);

      tags.push_back(control()->getIdentifier(components.value));
    }
  };

  for (auto attribute : ListView{attributes}) {
    if (foundAbiTag) break;

    if (auto gccAttribute = ast_cast<GccAttributeAST>(attribute)) {
      collectTags(gccAttribute->lparen2Loc, gccAttribute->rparenLoc);
    } else if (auto cxxAttribute = ast_cast<CxxAttributeAST>(attribute)) {
      collectTags(cxxAttribute->lbracketLoc, cxxAttribute->rbracketLoc);
    }
  }

  std::ranges::sort(tags, {}, [](const Identifier* id) { return id->name(); });
  tags.erase(std::ranges::unique(tags).begin(), tags.end());

  return tags;
}

void Binder::applyAbiTags(Symbol* symbol,
                          List<AttributeSpecifierAST*>* attributes) {
  if (!symbol || !attributes) return;

  auto tags = abiTags(attributes);
  if (tags.empty()) return;

  if (auto function = symbol_cast<FunctionSymbol>(symbol);
      function && function->canonical() != function) {
    auto canonicalTags = function->canonical()->abiTags();
    if (!canonicalTags.empty()) return;
  }

  symbol->setAbiTags(control()->getAbiTags(std::move(tags)));
}

void Binder::applyAbiTags(SimpleDeclarationAST* ast) {
  if (!ast || !ast->attributeList) return;

  auto interned = control()->getAbiTags(abiTags(ast->attributeList));
  if (!interned) return;

  for (auto initDeclarator : ListView{ast->initDeclaratorList}) {
    if (initDeclarator->symbol) initDeclarator->symbol->setAbiTags(interned);
  }
}

auto Binder::usesImplicitThis(StatementAST* stmt) -> bool {
  if (!stmt) return false;
  ThisUseFinder finder;
  finder.accept(stmt);
  return finder.found;
}

auto Binder::addImplicitThisCapture(ClassSymbol* classSymbol,
                                    const Type* thisType, SourceLocation loc)
    -> ThisLambdaCaptureAST* {
  auto ar = unit_->arena();

  auto field = control()->newFieldSymbol(classSymbol, loc);
  field->setName(control()->getIdentifier("__this"));
  field->setType(thisType);
  if (auto alignment = control()->memoryLayout()->alignmentOf(thisType)) {
    field->setAlignment(alignment.value());
  }
  classSymbol->addSymbol(field);
  classSymbol->setCapturedThisField(field);

  const auto& ctors = classSymbol->constructors();
  if (!ctors.empty()) {
    auto ctorSymbol = ctors.front();
    auto ctorType = type_cast<FunctionType>(ctorSymbol->type());
    auto paramTypes = ctorType->parameterTypes();
    paramTypes.push_back(thisType);
    ctorSymbol->setType(
        control()->getFunctionType(control()->getVoidType(), paramTypes));
  }

  auto thisExpr = ThisExpressionAST::create(ar);
  thisExpr->thisLoc = loc;
  thisExpr->type = thisType;
  thisExpr->valueCategory = ValueCategory::kPrValue;

  return ThisLambdaCaptureAST::create(ar, loc, thisExpr);
}

void Binder::addImplicitCaptures(LambdaExpressionAST* ast,
                                 ClassSymbol* classSymbol) {
  auto ar = unit_->arena();
  auto loc = ast->lbracketLoc;
  const auto hasCaptureDefault = ast->captureDefault != TokenKind::T_EOF_SYMBOL;
  const auto byCopy = ast->captureDefault == TokenKind::T_EQUAL;

  OdrUsedLocalFinder finder;
  finder.accept(ast->statement);

  auto isDeclaredInsideClosure = [&](Symbol* symbol) {
    for (auto scope = symbol->parent(); scope; scope = scope->parent()) {
      if (scope == classSymbol || scope == ast->symbol) return true;
    }
    return false;
  };

  std::unordered_set<const Identifier*> explicitlyCaptured;
  for (auto captureNode : ListView{ast->captureList}) {
    if (auto simple = ast_cast<SimpleLambdaCaptureAST>(captureNode)) {
      explicitlyCaptured.insert(simple->identifier);
    } else if (auto ref = ast_cast<RefLambdaCaptureAST>(captureNode)) {
      explicitlyCaptured.insert(ref->identifier);
    } else if (auto init = ast_cast<InitLambdaCaptureAST>(captureNode)) {
      explicitlyCaptured.insert(init->identifier);
    } else if (auto refInit = ast_cast<RefInitLambdaCaptureAST>(captureNode)) {
      explicitlyCaptured.insert(refInit->identifier);
    }
  }

  auto tail = &ast->captureList;
  while (*tail) tail = &(*tail)->next;

  std::unordered_map<Symbol*, FieldSymbol*> captured;
  std::unordered_set<Symbol*> reportedUncapturable;
  std::vector<const Type*> capturedTypes;

  for (auto use : finder.uses) {
    auto outerSymbol = use->symbol;

    if (auto known = captured.find(outerSymbol); known != captured.end()) {
      use->symbol = known->second;
      continue;
    }

    if (!isCapturableLocalEntity(outerSymbol)) continue;
    if (isDeclaredInsideClosure(outerSymbol)) continue;

    auto elementType = traits.remove_reference(outerSymbol->type());
    if (!elementType) continue;

    auto fieldType =
        byCopy ? elementType : control()->getLvalueReferenceType(elementType);

    auto identifier = name_cast<Identifier>(outerSymbol->name());
    if (!identifier) continue;
    if (explicitlyCaptured.contains(identifier)) continue;

    if (!hasCaptureDefault) {
      if (!reportedUncapturable.insert(outerSymbol).second) continue;
      error(use->firstSourceLocation(),
            std::format("variable '{}' cannot be implicitly captured in a "
                        "lambda with no capture-default specified",
                        identifier->name()));
      continue;
    }

    if (byCopy && type_cast<ClassType>(fieldType) &&
        !traits.is_trivially_copyable(fieldType)) {
      error(use->firstSourceLocation(),
            std::format("capturing '{}' by value is not yet supported for "
                        "non-trivially-copyable class types",
                        identifier->name()));
      continue;
    }

    auto idExpr = IdExpressionAST::create(ar);
    idExpr->unqualifiedId = NameIdAST::create(ar, identifier);
    idExpr->symbol = outerSymbol;
    idExpr->type = elementType;
    idExpr->valueCategory = ValueCategory::kLValue;

    ExpressionAST* initializer = idExpr;
    if (byCopy) (void)StandardConversion{unit_}.lvalueToRvalue(initializer);

    auto field = control()->newFieldSymbol(classSymbol, loc);
    field->setName(identifier);
    field->setType(fieldType);
    if (auto alignment = control()->memoryLayout()->alignmentOf(fieldType)) {
      field->setAlignment(alignment.value());
    }
    classSymbol->addSymbol(field);
    capturedTypes.push_back(fieldType);
    captured.emplace(outerSymbol, field);

    LambdaCaptureAST* capture = nullptr;
    if (byCopy) {
      auto simple = SimpleLambdaCaptureAST::create(ar);
      simple->identifierLoc = loc;
      simple->identifier = identifier;
      simple->initializer = initializer;
      capture = simple;
    } else {
      auto ref = RefLambdaCaptureAST::create(ar);
      ref->ampLoc = loc;
      ref->identifierLoc = loc;
      ref->identifier = identifier;
      ref->initializer = initializer;
      capture = ref;
    }

    *tail = make_list_node<LambdaCaptureAST>(ar, capture);
    tail = &(*tail)->next;

    use->symbol = field;
  }

  if (capturedTypes.empty()) return;

  const auto& ctors = classSymbol->constructors();
  if (!ctors.empty()) {
    auto ctorSymbol = ctors.front();
    auto ctorType = type_cast<FunctionType>(ctorSymbol->type());
    auto paramTypes = ctorType->parameterTypes();
    paramTypes.insert(paramTypes.end(), capturedTypes.begin(),
                      capturedTypes.end());
    ctorSymbol->setType(
        control()->getFunctionType(control()->getVoidType(), paramTypes));
  }

  auto status = buildRecordLayout(classSymbol);
  if (!status.has_value()) error(loc, status.error());
}

void Binder::bind(LambdaExpressionAST* ast) {
  auto parentScope = declaringScope();
  auto symbol = control()->newLambdaSymbol(parentScope, ast->lbracketLoc);
  ast->symbol = symbol;

  symbol->setInTemplate(inTemplate());

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

      if (traits.is_void(paramType)) {
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

  if (isCxx() && !isEnclosedInTemplate(ast->symbol->parent()) &&
      !ast->symbol->isInTemplate()) {
    auto closureName =
        control()->getIdentifier(std::format("__lambda_{}", lambdaCount_++));

    auto classSymbol = control()->newClassSymbol(parentScope, ast->lbracketLoc);
    classSymbol->setName(closureName);
    parentScope->addSymbol(classSymbol);
    classSymbol->setClosureDiscriminator(
        lambdaDiscriminators_[classSymbol->enclosingFunction()]++);

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

    if (ast->symbol->isTemplate()) {
      auto ar = unit_->arena();
      auto templateParamsSymbol = control()->newTemplateParametersSymbol(
          operatorFunc, ast->lbracketLoc);
      for (auto p : ListView{ast->templateParameterList}) {
        if (p && p->symbol) templateParamsSymbol->addSymbol(p->symbol);
      }
      int depth = ast->templateParameterList
                      ? ast->templateParameterList->value->depth
                      : 0;
      auto templateDecl = TemplateDeclarationAST::create(
          ar, ast->templateParameterList, ast->templateRequiresClause,
          /*declaration=*/nullptr, templateParamsSymbol, depth);
      operatorFunc->setTemplateParameters(templateParamsSymbol);
      operatorFunc->setTemplateDeclaration(templateDecl);
    }

    classSymbol->setIsClosureType(true);

    std::vector<const Type*> ctorParamTypes;

    for (auto captureNode : ListView{ast->captureList}) {
      auto captureLoc = captureNode->firstSourceLocation();
      auto ar = unit_->arena();

      auto addField = [&](const Identifier* fieldName,
                          const Type* fieldType) -> FieldSymbol* {
        auto field = control()->newFieldSymbol(classSymbol, captureLoc);
        field->setName(fieldName);
        field->setType(fieldType);
        if (auto alignment =
                control()->memoryLayout()->alignmentOf(fieldType)) {
          field->setAlignment(alignment.value());
        }
        classSymbol->addSymbol(field);
        ctorParamTypes.push_back(fieldType);
        return field;
      };

      if (auto simple = ast_cast<SimpleLambdaCaptureAST>(captureNode)) {
        auto outerSymbol = lookupCaptureName(parentScope, simple->identifier);
        if (!outerSymbol) {
          error(simple->identifierLoc,
                std::format("use of undeclared identifier '{}'",
                            simple->identifier->name()));
          continue;
        }
        if (!checkCapturedEntity(outerSymbol, simple->identifier,
                                 simple->identifierLoc)) {
          continue;
        }
        auto fieldType = traits.remove_reference(outerSymbol->type());
        if (type_cast<ClassType>(fieldType) &&
            !traits.is_trivially_copyable(fieldType)) {
          error(captureLoc,
                std::format("capturing '{}' by value is not yet supported for "
                            "non-trivially-copyable class types",
                            simple->identifier->name()));
          continue;
        }

        auto idExpr = IdExpressionAST::create(ar);
        idExpr->unqualifiedId = NameIdAST::create(ar, simple->identifier);
        idExpr->symbol = outerSymbol;
        idExpr->type = fieldType;
        idExpr->valueCategory = ValueCategory::kLValue;

        ExpressionAST* valueExpr = idExpr;
        (void)StandardConversion{unit_}.lvalueToRvalue(valueExpr);
        simple->initializer = valueExpr;

        addField(simple->identifier, fieldType);
      } else if (auto ref = ast_cast<RefLambdaCaptureAST>(captureNode)) {
        auto outerSymbol = lookupCaptureName(parentScope, ref->identifier);
        if (!outerSymbol) {
          error(ref->identifierLoc,
                std::format("use of undeclared identifier '{}'",
                            ref->identifier->name()));
          continue;
        }
        if (!checkCapturedEntity(outerSymbol, ref->identifier,
                                 ref->identifierLoc)) {
          continue;
        }
        auto elementType = traits.remove_reference(outerSymbol->type());
        auto fieldType = control()->getLvalueReferenceType(elementType);

        auto idExpr = IdExpressionAST::create(ar);
        idExpr->unqualifiedId = NameIdAST::create(ar, ref->identifier);
        idExpr->symbol = outerSymbol;
        idExpr->type = elementType;
        idExpr->valueCategory = ValueCategory::kLValue;
        ref->initializer = idExpr;

        addField(ref->identifier, fieldType);
      } else if (auto th = ast_cast<ThisLambdaCaptureAST>(captureNode)) {
        auto thisType = enclosingThisType(parentScope);
        if (!thisType) {
          error(captureLoc, "'this' cannot be captured in this context");
          continue;
        }

        auto thisExpr = ThisExpressionAST::create(ar);
        thisExpr->thisLoc = th->thisLoc;
        thisExpr->type = thisType;
        thisExpr->valueCategory = ValueCategory::kPrValue;
        th->initializer = thisExpr;

        auto field = addField(control()->getIdentifier("__this"), thisType);
        classSymbol->setCapturedThisField(field);
      } else if (auto deref =
                     ast_cast<DerefThisLambdaCaptureAST>(captureNode)) {
        error(captureLoc, "capture of '*this' is not yet supported");
      } else if (auto initCap = ast_cast<InitLambdaCaptureAST>(captureNode)) {
        if (!initCap->initializer || !initCap->initializer->type) continue;
        auto fieldType = traits.decay(initCap->initializer->type);
        if (type_cast<ClassType>(fieldType) &&
            !traits.is_trivially_copyable(fieldType)) {
          error(captureLoc,
                std::format("init-capturing '{}' by value is not yet "
                            "supported for non-trivially-copyable class types",
                            initCap->identifier->name()));
          continue;
        }
        addField(initCap->identifier, fieldType);
      } else if (auto refInitCap =
                     ast_cast<RefInitLambdaCaptureAST>(captureNode)) {
        if (!refInitCap->initializer || !refInitCap->initializer->type) {
          continue;
        }
        auto elementType =
            traits.remove_reference(refInitCap->initializer->type);
        auto fieldType = control()->getLvalueReferenceType(elementType);
        addField(refInitCap->identifier, fieldType);
      }
    }

    auto ctorSymbol =
        control()->newFunctionSymbol(classSymbol, ast->lbracketLoc);
    ctorSymbol->setName(closureName);
    ctorSymbol->setType(
        control()->getFunctionType(control()->getVoidType(), ctorParamTypes));
    ctorSymbol->setDefined(true);
    ctorSymbol->setDefaulted(true);
    ctorSymbol->setLanguageLinkage(LanguageKind::kCXX);
    classSymbol->addConstructor(ctorSymbol);

    ast->constructorSymbol = ctorSymbol;

    if (!ast->symbol->isTemplate() &&
        ast->captureDefault == TokenKind::T_EOF_SYMBOL && !ast->captureList) {
      auto fptrType = control()->getPointerType(funcType);
      auto convFuncType = control()->getFunctionType(fptrType, {});
      auto convName = control()->getConversionFunctionId(fptrType);
      auto convFunc =
          control()->newFunctionSymbol(classSymbol, ast->lbracketLoc);
      convFunc->setName(convName);
      convFunc->setType(convFuncType);
      convFunc->setDefined(true);
      convFunc->setLanguageLinkage(LanguageKind::kCXX);
      classSymbol->addSymbol(convFunc);
    }

    classSymbol->setComplete(true);
    auto status = buildRecordLayout(classSymbol);
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

  if (!classSymbol->capturedThisField() &&
      (ast->captureDefault == TokenKind::T_AMP ||
       ast->captureDefault == TokenKind::T_EQUAL) &&
      usesImplicitThis(ast->statement)) {
    if (auto thisType = enclosingThisType(ast->symbol->parent())) {
      auto capture =
          addImplicitThisCapture(classSymbol, thisType, ast->lbracketLoc);

      auto tail = &ast->captureList;
      while (*tail) tail = &(*tail)->next;
      *tail = make_list_node<LambdaCaptureAST>(ar, capture);

      auto status = buildRecordLayout(classSymbol);
      if (!status.has_value()) {
        error(ast->lbracketLoc, status.error());
      }
    }
  }

  addImplicitCaptures(ast, classSymbol);

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

  auto reboundBody = ast_cast<CompoundStatementAST>(
      ASTRewriter::paste(unit_, bodyScope, ast->statement));

  if (!ast->trailingReturnType) finishAutoReturnType(operatorFunc);

  auto opId = OperatorFunctionIdAST::create(ar, TokenKind::T_LPAREN);

  auto idDecl = IdDeclaratorAST::create(ar);
  idDecl->unqualifiedId = opId;

  auto funcChunk = FunctionDeclaratorChunkAST::create(ar);
  if (ast->parameterDeclarationClause) {
    funcChunk->parameterDeclarationClause =
        ast->parameterDeclarationClause->clone(ar);
  }
  if (ast->trailingReturnType) {
    funcChunk->trailingReturnType = ast->trailingReturnType->clone(ar);
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

  if (!ast->trailingReturnType) {
    auto autoSpec = AutoTypeSpecifierAST::create(ar);
    funcDef->declSpecifierList = make_list_node<SpecifierAST>(ar, autoSpec);
  }

  operatorFunc->setDeclaration(funcDef);

  if (auto templateDecl = operatorFunc->templateDeclaration())
    templateDecl->declaration = funcDef;

  auto closureName = name_cast<Identifier>(classSymbol->name());
  for (auto ctor : classSymbol->constructors()) {
    if (ctor->declaration()) continue;

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

void Binder::bind(UsingDirectiveAST* ast, NamespaceSymbol* resolvedNamespace) {
  auto id = ast->unqualifiedId->identifier;

  NamespaceSymbol* namespaceSymbol = nullptr;
  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol)
    namespaceSymbol =
        qualifiedLookupNamespace(ast->nestedNameSpecifier->symbol, id);
  else
    namespaceSymbol = resolvedNamespace;

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
  auto symbol =
      control()->newTypeAliasSymbol(declaringScope(), decl.location());
  symbol->setName(name);
  symbol->setType(type);

  bool hasConflict = false;

  auto should_report_conflict = [&](SourceLocation loc) {
    if (auto preprocessor = unit_->preprocessor()) {
      const auto& token = unit_->tokenAt(loc);
      if (token) return !preprocessor->isSystemHeader(token.fileId());
    }
    return true;
  };

  auto aliases_named_type_symbol = [&](Symbol* candidate) {
    if (isC() && (symbol_cast<ClassSymbol>(candidate) ||
                  symbol_cast<EnumSymbol>(candidate)))
      return true;

    if (auto classSymbol = symbol_cast<ClassSymbol>(candidate)) {
      if (auto classType = type_cast<ClassType>(symbol->type())) {
        return classType->symbol() == classSymbol;
      }
    }

    if (auto enumSymbol = symbol_cast<EnumSymbol>(candidate)) {
      if (auto enumType = type_cast<EnumType>(symbol->type())) {
        return enumType->symbol() == enumSymbol;
      }
    }

    if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(candidate)) {
      if (auto scopedEnumType = type_cast<ScopedEnumType>(symbol->type())) {
        return scopedEnumType->symbol() == scopedEnumSymbol;
      }
    }

    return false;
  };

  for (auto candidate : scope()->find(name)) {
    if (auto existing = symbol_cast<TypeAliasSymbol>(candidate)) {
      if (existing->type() && symbol->type() &&
          !traits.is_same(existing->type(), symbol->type())) {
        if (should_report_conflict(decl.location())) {
          error(decl.location(), std::format("conflicting declaration of '{}'",
                                             to_string(name)));
          hasConflict = true;
        }
        break;
      }

      auto canon = existing->canonical();
      canon->addRedeclaration(symbol);
      break;
    } else {
      if (aliases_named_type_symbol(candidate)) continue;

      if (should_report_conflict(decl.location())) {
        error(decl.location(),
              std::format("conflicting declaration of '{}'", to_string(name)));
        hasConflict = true;
      }
      break;
    }
  }

  if (!hasConflict) {
    scope()->addSymbol(symbol);
  }

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
auto arrayBoundToString(const Type* type) -> std::optional<std::string> {
  if (auto bounded = type_cast<BoundedArrayType>(type)) {
    return std::to_string(bounded->size());
  }
  return std::nullopt;
}

auto isEffectivelyUnboundedArray(TranslationUnit* unit, const Type* type)
    -> bool {
  if (!unit || !type) return false;
  if (unit->typeTraits().is_unbounded_array(type)) return true;

  auto unresolved = type_cast<UnresolvedBoundedArrayType>(type);
  if (!unresolved) return false;
  return !arrayBoundToString(type).has_value();
}

auto areRedeclarationTypesCompatible(TranslationUnit* unit,
                                     const Type* existingType,
                                     const Type* incomingType) -> bool {
  if (!unit || !existingType || !incomingType) return false;

  while (auto qual = type_cast<QualType>(existingType)) {
    existingType = qual->elementType();
  }
  while (auto qual = type_cast<QualType>(incomingType)) {
    incomingType = qual->elementType();
  }

  if (unit->typeTraits().is_same(existingType, incomingType)) return true;

  if (!unit->typeTraits().is_array(existingType) ||
      !unit->typeTraits().is_array(incomingType)) {
    return false;
  }

  auto existingElementType = unit->typeTraits().get_element_type(existingType);
  auto incomingElementType = unit->typeTraits().get_element_type(incomingType);
  if (!areRedeclarationTypesCompatible(unit, existingElementType,
                                       incomingElementType)) {
    return false;
  }

  if (isEffectivelyUnboundedArray(unit, existingType) ||
      isEffectivelyUnboundedArray(unit, incomingType)) {
    return true;
  }

  auto existingBound = arrayBoundToString(existingType);
  auto incomingBound = arrayBoundToString(incomingType);
  if (!existingBound || !incomingBound) return true;
  return *existingBound == *incomingBound;
}

auto preferredRedeclarationType(TranslationUnit* unit, const Type* existingType,
                                const Type* incomingType) -> const Type* {
  if (!unit || !existingType || !incomingType) return existingType;
  if (unit->typeTraits().is_same(existingType, incomingType))
    return existingType;

  if (isEffectivelyUnboundedArray(unit, existingType) &&
      unit->typeTraits().is_array(incomingType) &&
      !isEffectivelyUnboundedArray(unit, incomingType) &&
      areRedeclarationTypesCompatible(
          unit, unit->typeTraits().get_element_type(existingType),
          unit->typeTraits().get_element_type(incomingType))) {
    return incomingType;
  }

  auto existingBounded = type_cast<BoundedArrayType>(existingType);
  auto incomingUnbounded = isEffectivelyUnboundedArray(unit, incomingType);
  if (existingBounded && incomingUnbounded &&
      areRedeclarationTypesCompatible(
          unit, existingBounded->elementType(),
          unit->typeTraits().get_element_type(incomingType))) {
    return existingType;
  }

  return existingType;
}

auto areFunctionSignaturesEquivalentForRedeclaration(TranslationUnit* unit,
                                                     const Type* lhs,
                                                     const Type* rhs) -> bool {
  if (!unit || !lhs || !rhs) return false;
  if (unit->typeTraits().is_same(lhs, rhs)) return true;

  auto lhsFn = type_cast<FunctionType>(lhs);
  auto rhsFn = type_cast<FunctionType>(rhs);
  if (!lhsFn || !rhsFn) return false;

  const bool dependentReturnType = isDependent(unit, lhsFn->returnType()) ||
                                   isDependent(unit, rhsFn->returnType());
  if (!dependentReturnType &&
      !unit->typeTraits().is_same(lhsFn->returnType(), rhsFn->returnType()))
    return false;
  if (lhsFn->cvQualifiers() != rhsFn->cvQualifiers()) return false;
  if (lhsFn->refQualifier() != rhsFn->refQualifier()) return false;
  if (lhsFn->isVariadic() != rhsFn->isVariadic()) return false;

  const auto& lhsParams = lhsFn->parameterTypes();
  const auto& rhsParams = rhsFn->parameterTypes();
  if (lhsParams.size() != rhsParams.size()) return false;

  for (std::size_t i = 0; i < lhsParams.size(); ++i) {
    if (!areRedeclarationTypesCompatible(unit, lhsParams[i], rhsParams[i])) {
      return false;
    }
  }

  return true;
}

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
                   traits.is_same(fn->type(), member->type());
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
                       traits.is_same(fn->type(), m->type());
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

auto Binder::declareField(DeclaratorAST* declarator, const Decl& decl)
    -> FieldSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());

  if (name) {
    for (auto candidate : scope()->find(name)) {
      auto existingField = symbol_cast<FieldSymbol>(candidate);
      const bool collides = existingField ||
                            symbol_cast<FunctionSymbol>(candidate) ||
                            symbol_cast<OverloadSetSymbol>(candidate) ||
                            symbol_cast<EnumeratorSymbol>(candidate);
      if (!collides) continue;

      error(decl.location(),
            std::format("duplicate member '{}'", to_string(name)));

      if (existingField) return existingField;
      break;
    }
  }

  auto fieldSymbol = control()->newFieldSymbol(scope(), decl.location());
  applySpecifiers(fieldSymbol, decl.specs);
  fieldSymbol->setName(name);
  fieldSymbol->setType(type);
  fieldSymbol->setMutable(decl.specs.isMutable);
  fieldSymbol->setNoUniqueAddress(decl.specs.isNoUniqueAddress);
  if (auto alignment = control()->memoryLayout()->alignmentOf(type)) {
    fieldSymbol->setAlignment(alignment.value());
  }

  if (decl.isBitField()) {
    fieldSymbol->setBitField(true);

    if (!traits.is_integral(type) && !traits.is_enum(type) && !inTemplate() &&
        !isDependent(unit_, type)) {
      error(decl.location(), "bit-field has non-integral type");
    }

    if (decl.bitfieldDeclarator && decl.bitfieldDeclarator->sizeExpression) {
      ASTInterpreter interp{unit_};
      auto value = interp.evaluate(decl.bitfieldDeclarator->sizeExpression);

      if (value) {
        fieldSymbol->setBitFieldWidth(*value);
        if (auto width = std::get_if<std::intmax_t>(&*value)) {
          if (*width < 0) {
            error(decl.location(), "bit-field width is negative");
          } else if (*width == 0 && name) {
            error(decl.location(), "zero-width bit-field must be unnamed");
          } else if (!inTemplate()) {
            auto typeSize = control()->memoryLayout()->sizeOf(type);
            if (typeSize && *width > *typeSize * 8) {
              error(decl.location(),
                    "width of bit-field exceeds width of its type");
            }
          }
        } else {
          error(decl.location(), "bit-field width is not an integer");
        }
      } else if (!inTemplate() &&
                 !isDependent(unit_, decl.bitfieldDeclarator->sizeExpression)) {
        error(decl.location(), "bit-field width is not a constant expression");
      }
    }
  }

  scope()->addSymbol(fieldSymbol);
  return fieldSymbol;
}

void Binder::declareAnonymousField(ClassSpecifierAST* classSpecifier) {
  auto classSymbol = classSpecifier->symbol;
  if (!classSymbol) return;
  if (classSymbol->name()) return;

  auto fieldSymbol =
      control()->newFieldSymbol(scope(), classSymbol->location());
  fieldSymbol->setName(nullptr);
  fieldSymbol->setType(classSymbol->type());
  if (auto alignment =
          control()->memoryLayout()->alignmentOf(classSymbol->type())) {
    fieldSymbol->setAlignment(alignment.value());
  }
  scope()->addSymbol(fieldSymbol);
}

auto Binder::declareVariable(DeclaratorAST* declarator, const Decl& decl,
                             bool addSymbolToParentScope) -> VariableSymbol* {
  auto name = decl.getName();
  auto currentScope = declaringScope();
  auto targetScope =
      decl.specs.isExtern ? scopeForBlockDecl(currentScope) : currentScope;

  auto symbol = control()->newVariableSymbol(targetScope, decl.location());
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());
  applySpecifiers(symbol, decl.specs);
  symbol->setName(name);
  symbol->setType(type);

  bool isOutOfClassStaticMemberDef = false;
  if (auto declId = decl.declaratorId) {
    if (auto nns = declId->nestedNameSpecifier; nns && nns->symbol) {
      auto classSymbol = symbol_cast<ClassSymbol>(nns->symbol);
      if (!classSymbol) {
        if (auto classType =
                type_cast<ClassType>(traits.remove_cv(nns->symbol->type()))) {
          classSymbol = classType->symbol();
        }
      }
      if (classSymbol) {
        for (auto candidate : classSymbol->find(name)) {
          auto field = symbol_cast<FieldSymbol>(candidate);
          if (!field || !field->isStatic()) continue;
          field->setDefinition(symbol);
          symbol->setStatic(true);
          symbol->setParent(classSymbol);
          isOutOfClassStaticMemberDef = true;
          break;
        }
      }
    }
  }
  if (auto classType = type_cast<ClassType>(traits.remove_cv(type))) {
    traits.requireCompleteClass(classType->symbol());
  }
  if (addSymbolToParentScope) {
    for (auto candidate : targetScope->find(name)) {
      if (auto existing = symbol_cast<VariableSymbol>(candidate)) {
        if (isOutOfClassStaticMemberDef &&
            (existing->isStatic() ||
             symbol_cast<ClassSymbol>(existing->parent()))) {
          break;
        }
        if (!areRedeclarationTypesCompatible(unit_, existing->type(),
                                             symbol->type())) {
          error(
              symbol->location(),
              std::format("conflicting declaration of '{}'", to_string(name)));
          continue;
        }

        auto canon = existing->canonical();
        auto mergedType =
            preferredRedeclarationType(unit_, canon->type(), symbol->type());
        canon->setType(mergedType);
        symbol->setType(mergedType);
        canon->addRedeclaration(symbol);
        break;
      }
    }

    targetScope->addSymbol(symbol);

    if (targetScope != currentScope) {
      if (symbol->canonical() == symbol) symbol->setHidden(true);
      injectUsing(currentScope, name, symbol->canonical(), decl.location());
    }
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

auto Binder::reportUnresolvedNestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> bool {
  if (inTemplate() || isDependentNestedNameSpecifier(ast)) return false;

  error(ast->firstSourceLocation(),
        "nested name specifier must be a class or namespace");

  return true;
}

auto Binder::resolveNestedNameSpecifier(Symbol* symbol) -> ScopeSymbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    traits.requireCompleteClass(classSymbol);
    return classSymbol;
  }

  if (auto injected = symbol_cast<InjectedClassNameSymbol>(symbol)) {
    traits.requireCompleteClass(injected->classSymbol());
    return injected->classSymbol();
  }

  if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(symbol))
    return namespaceSymbol;

  if (auto enumSymbol = symbol_cast<EnumSymbol>(symbol)) return enumSymbol;

  if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(symbol))
    return scopedEnumSymbol;

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType = type_cast<ClassType>(typeAliasSymbol->type())) {
      traits.requireCompleteClass(classType->symbol());
      return classType->symbol();
    }

    if (auto enumType = type_cast<EnumType>(typeAliasSymbol->type()))
      return enumType->symbol();

    if (auto scopedEnumType =
            type_cast<ScopedEnumType>(typeAliasSymbol->type()))
      return scopedEnumType->symbol();
  }

  return nullptr;
}

namespace {
struct TemplateArity {
  int minArgs = 0;
  int maxArgs = 0;
  bool hasParameterPack = false;
};

auto isPackParameter(TemplateParameterAST* parameter) -> bool {
  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
    return typeParameter->isPack;
  }

  if (auto nonTypeParameter =
          ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    return nonTypeParameter->declaration &&
           nonTypeParameter->declaration->isPack;
  }

  if (auto templateTypeParameter =
          ast_cast<TemplateTypeParameterAST>(parameter)) {
    return templateTypeParameter->isPack;
  }

  if (auto constraintParameter =
          ast_cast<ConstraintTypeParameterAST>(parameter)) {
    return static_cast<bool>(constraintParameter->ellipsisLoc);
  }

  return false;
}

auto hasDefaultTemplateArgument(TemplateParameterAST* parameter) -> bool {
  if (auto typeParameter = ast_cast<TypenameTypeParameterAST>(parameter)) {
    return typeParameter->typeId && typeParameter->typeId->type;
  }

  if (auto nonTypeParameter =
          ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    return nonTypeParameter->declaration &&
           nonTypeParameter->declaration->equalLoc &&
           nonTypeParameter->declaration->expression;
  }

  if (auto templateTypeParameter =
          ast_cast<TemplateTypeParameterAST>(parameter)) {
    return templateTypeParameter->idExpression;
  }

  if (auto constraintParameter =
          ast_cast<ConstraintTypeParameterAST>(parameter)) {
    return constraintParameter->typeId && constraintParameter->typeId->type;
  }

  return false;
}

auto computeTemplateArity(TemplateDeclarationAST* templateDecl)
    -> TemplateArity {
  TemplateArity arity;
  if (!templateDecl) return arity;

  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    ++arity.maxArgs;

    if (isPackParameter(parameter)) {
      arity.hasParameterPack = true;
      continue;
    }

    if (!hasDefaultTemplateArgument(parameter)) {
      ++arity.minArgs;
    }
  }

  return arity;
}

auto templateArgumentCount(List<TemplateArgumentAST*>* templateArgumentList)
    -> int {
  int count = 0;
  for (auto argument : ListView{templateArgumentList}) {
    (void)argument;
    ++count;
  }
  return count;
}

auto isTemplateArityMatch(TemplateDeclarationAST* templateDecl,
                          List<TemplateArgumentAST*>* templateArgumentList,
                          bool isFunctionTemplate = false) -> bool {
  if (!templateDecl) return true;

  auto arity = computeTemplateArity(templateDecl);
  auto argc = templateArgumentCount(templateArgumentList);

  if (!isFunctionTemplate && argc < arity.minArgs) return false;
  if (!arity.hasParameterPack && argc > arity.maxArgs) return false;

  return true;
}

enum class TemplateParameterKind {
  kUnknown,
  kType,
  kNonType,
  kTemplate,
  kConstraint,
};

auto templateParameterKind(TemplateParameterAST* parameter)
    -> TemplateParameterKind {
  if (ast_cast<TypenameTypeParameterAST>(parameter)) {
    return TemplateParameterKind::kType;
  }

  if (ast_cast<NonTypeTemplateParameterAST>(parameter)) {
    return TemplateParameterKind::kNonType;
  }

  if (ast_cast<TemplateTypeParameterAST>(parameter)) {
    return TemplateParameterKind::kTemplate;
  }

  if (ast_cast<ConstraintTypeParameterAST>(parameter)) {
    return TemplateParameterKind::kConstraint;
  }

  return TemplateParameterKind::kUnknown;
}

auto isTemplateArgumentCompatibleWithParameter(TemplateArgumentAST* argument,
                                               TemplateParameterKind kind)
    -> bool {
  if (!argument) return false;

  switch (kind) {
    case TemplateParameterKind::kType:
    case TemplateParameterKind::kTemplate:
    case TemplateParameterKind::kConstraint: {
      auto typeArg = ast_cast<TypeTemplateArgumentAST>(argument);
      return typeArg && typeArg->typeId;
    }

    case TemplateParameterKind::kNonType: {
      auto exprArg = ast_cast<ExpressionTemplateArgumentAST>(argument);
      return exprArg && exprArg->expression;
    }

    case TemplateParameterKind::kUnknown:
      return false;
  }

  return false;
}

auto isTemplateArgumentKindMatch(
    TemplateDeclarationAST* templateDecl,
    List<TemplateArgumentAST*>* templateArgumentList) -> bool {
  if (!templateDecl) return true;

  std::vector<TemplateParameterAST*> parameters;
  for (auto parameter : ListView{templateDecl->templateParameterList}) {
    parameters.push_back(parameter);
  }

  std::vector<TemplateArgumentAST*> arguments;
  for (auto argument : ListView{templateArgumentList}) {
    arguments.push_back(argument);
  }

  int argumentIndex = 0;
  for (int parameterIndex = 0;
       parameterIndex < static_cast<int>(parameters.size()); ++parameterIndex) {
    if (argumentIndex >= static_cast<int>(arguments.size())) break;

    auto parameter = parameters[parameterIndex];
    auto kind = templateParameterKind(parameter);
    if (kind == TemplateParameterKind::kUnknown) return false;

    if (isPackParameter(parameter)) {
      while (argumentIndex < static_cast<int>(arguments.size())) {
        if (!isTemplateArgumentCompatibleWithParameter(arguments[argumentIndex],
                                                       kind)) {
          return false;
        }
        ++argumentIndex;
      }
      break;
    }

    if (!isTemplateArgumentCompatibleWithParameter(arguments[argumentIndex],
                                                   kind)) {
      return false;
    }

    ++argumentIndex;
  }

  return argumentIndex == static_cast<int>(arguments.size());
}
}  // namespace

auto Binder::overloadSetFor(ScopeSymbol* scope, const Name* name,
                            SourceLocation location) -> OverloadSetSymbol* {
  for (auto candidate : scope->find(name)) {
    if (auto overloadSet = symbol_cast<OverloadSetSymbol>(candidate))
      return overloadSet;

    auto function = symbol_cast<FunctionSymbol>(candidate);
    auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(candidate);
    if (usingDeclaration && usingDeclaration->introducedFunctions().empty())
      continue;
    if (!function && !usingDeclaration) continue;

    auto overloadSet = control()->newOverloadSetSymbol(scope, location);
    overloadSet->setName(name);
    if (function) overloadSet->addFunction(function);
    if (usingDeclaration) overloadSet->addUsingDeclaration(usingDeclaration);
    scope->replaceSymbol(candidate, overloadSet);
    return overloadSet;
  }

  auto overloadSet = control()->newOverloadSetSymbol(scope, location);
  overloadSet->setName(name);
  scope->addSymbol(overloadSet);
  return overloadSet;
}

void Binder::declareArgumentDependentCallee(IdExpressionAST* ast) {
  auto name = get_name(control(), ast->unqualifiedId);
  if (auto templateId = name_cast<TemplateId>(name)) name = templateId->name();
  if (!name_cast<Identifier>(name) && !name_cast<OperatorId>(name)) return;

  auto callee = control()->newOverloadSetSymbol(declaringScope(),
                                                ast->firstSourceLocation());
  callee->setName(name);
  ast->symbol = callee;
}

void Binder::bind(IdExpressionAST* ast, bool mayUseArgumentDependentLookup) {
  if (!ast->unqualifiedId) {
    error(ast->firstSourceLocation(),
          "expected an unqualified identifier in id expression");
    return;
  }

  if (!ast->symbol && !ast->nestedNameSpecifier &&
      mayUseArgumentDependentLookup) {
    declareArgumentDependentCallee(ast);
  }

  if (ast->nestedNameSpecifier) {
    if (!ast->nestedNameSpecifier->symbol) {
      (void)reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier);
      return;
    }

    auto name = get_name(control(), ast->unqualifiedId);

    const Name* componentName = name;

    if (auto templateId = name_cast<TemplateId>(name)) {
      componentName = templateId->name();
    }

    ast->symbol =
        qualifiedLookup(ast->nestedNameSpecifier->symbol, componentName);

    if (auto ns =
            symbol_cast<NamespaceSymbol>(ast->nestedNameSpecifier->symbol)) {
      ast->symbol = mergeInlineNamespaceOverloads(control(), ns, componentName,
                                                  ast->symbol);
    }
  }

  resolveIdExpression(ast, mayUseArgumentDependentLookup);
}

void Binder::qualifiedLookupIdExpression(IdExpressionAST* ast) {
  if (!ast->unqualifiedId) return;
  if (!ast->nestedNameSpecifier || !ast->nestedNameSpecifier->symbol) return;

  if (auto classSymbol =
          symbol_cast<ClassSymbol>(ast->nestedNameSpecifier->symbol)) {
    traits.requireCompleteClass(classSymbol);
  }

  auto name = get_name(control(), ast->unqualifiedId);
  const Name* componentName = name;
  if (auto templateId = name_cast<TemplateId>(name))
    componentName = templateId->name();

  ast->symbol =
      qualifiedLookup(ast->nestedNameSpecifier->symbol, componentName);

  if (auto ns =
          symbol_cast<NamespaceSymbol>(ast->nestedNameSpecifier->symbol)) {
    ast->symbol = mergeInlineNamespaceOverloads(control(), ns, componentName,
                                                ast->symbol);
  }

  resolveIdExpression(ast, /*isCallee=*/false);
}

void Binder::resolveIdExpression(IdExpressionAST* ast, bool isCallee) {
  if (isArgumentDependentCallee(ast->symbol)) return;

  if (unit_->config().checkTypes) {
    if (auto templateId = ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)) {
      auto templateIdName = get_name(control(), templateId);
      Symbol* templateSymbol = nullptr;
      bool instantiated = false;
      bool hasTemplateCandidate = false;
      bool hasDeferredFunctionTemplate = false;

      auto needsCallSiteDeduction =
          [&](TemplateDeclarationAST* templateDecl) -> bool {
        auto arity = computeTemplateArity(templateDecl);
        auto argc = templateArgumentCount(templateId->templateArgumentList);
        if (argc < arity.minArgs) return true;
        if (arity.hasParameterPack && argc > 0 && argc == arity.minArgs)
          return true;
        return false;
      };

      auto hasDependentArguments = [&]() -> bool {
        return hasDependentTemplateArguments(unit_, templateId);
      };

      if (symbol_cast<ConceptSymbol>(ast->symbol)) return;

      if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
        if (var->isSpecialization()) return;
        if (var->templateDeclaration() &&
            (!inTemplate() || !hasDependentArguments())) {
          templateSymbol = var;
        }
      } else if (auto func = symbol_cast<FunctionSymbol>(ast->symbol)) {
        if (func->isSpecialization() && !func->templateDeclaration()) return;
        if (func->templateDeclaration()) {
          hasTemplateCandidate = true;
          if (!inTemplate() &&
              isTemplateArityMatch(func->templateDeclaration(),
                                   templateId->templateArgumentList,
                                   /*isFunctionTemplate=*/true) &&
              isTemplateArgumentKindMatch(func->templateDeclaration(),
                                          templateId->templateArgumentList)) {
            if (needsCallSiteDeduction(func->templateDeclaration())) {
              hasDeferredFunctionTemplate = true;
            } else {
              templateSymbol = func;
            }
          }
        }
      } else if (auto ovl = symbol_cast<OverloadSetSymbol>(ast->symbol)) {
        const auto ovlFunctions = ovl->functions();

        int matchingTemplateCount = 0;
        for (auto func : ovlFunctions) {
          if (!func->templateDeclaration()) continue;
          if (isTemplateArityMatch(func->templateDeclaration(),
                                   templateId->templateArgumentList,
                                   /*isFunctionTemplate=*/true) &&
              isTemplateArgumentKindMatch(func->templateDeclaration(),
                                          templateId->templateArgumentList)) {
            ++matchingTemplateCount;
          }
        }
        if (matchingTemplateCount > 1) return;

        for (auto func : ovlFunctions) {
          if (!func->templateDeclaration()) continue;
          hasTemplateCandidate = true;
          if (!isTemplateArityMatch(func->templateDeclaration(),
                                    templateId->templateArgumentList,
                                    /*isFunctionTemplate=*/true) ||
              !isTemplateArgumentKindMatch(func->templateDeclaration(),
                                           templateId->templateArgumentList)) {
            continue;
          }
          if (needsCallSiteDeduction(func->templateDeclaration())) {
            hasDeferredFunctionTemplate = true;
            continue;
          }
          if (!templateSymbol) templateSymbol = func;
          if (inTemplate()) continue;
          auto instance = ASTRewriter::instantiate(
              unit_, templateId->templateArgumentList, func, {},
              /*sfinaeContext=*/true);
          if (instance) {
            ast->symbol = instance;
            templateSymbol = func;
            instantiated = true;
            break;
          }
        }
        if (instantiated) return;

        if (hasDeferredFunctionTemplate) return;

        if (templateSymbol && !inTemplate()) {
          if (isCallee) return;
          if (reportErrors_) {
            error(templateId->firstSourceLocation(),
                  std::format("invalid template-id '{}'",
                              to_string(templateIdName)));
          }
          return;
        }
      }

      if (hasDeferredFunctionTemplate) return;

      if (!templateSymbol) {
        if (!inTemplate()) {
          if (hasTemplateCandidate) {
            if (isCallee) return;
            error(templateId->firstSourceLocation(),
                  std::format("invalid template-id '{}'",
                              to_string(templateIdName)));
          } else {
            error(templateId->firstSourceLocation(),
                  std::format("not a template"));
          }
        }
      } else {
        if (inTemplate() && hasDependentArguments()) return;

        const bool isFuncTemplate =
            symbol_cast<FunctionSymbol>(templateSymbol) != nullptr;
        auto instance = ASTRewriter::instantiate(
            unit_, templateId->templateArgumentList, templateSymbol, {},
            /*sfinaeContext=*/isFuncTemplate);
        if (!instance) {
          if (!inTemplate()) {
            error(templateId->firstSourceLocation(),
                  std::format("invalid template-id '{}'",
                              to_string(templateIdName)));
          }
          return;
        }

        ast->symbol = instance;
      }
    }
  }
}

auto Binder::namesOwnTemplateParameters(SimpleTemplateIdAST* templateId,
                                        ClassSymbol* classSymbol) -> bool {
  auto templateParameters = classSymbol->templateParameters();
  if (!templateParameters) return false;

  auto parameters = templateParameters->members();
  auto parameter = parameters.begin();

  for (auto argument : ListView{templateId->templateArgumentList}) {
    if (parameter == parameters.end()) return false;
    auto typeArgument = ast_cast<TypeTemplateArgumentAST>(argument);
    if (!typeArgument || !typeArgument->typeId) return false;
    if (typeArgument->typeId->type != (*parameter)->type()) return false;
    ++parameter;
  }

  return parameter == parameters.end();
}

auto Binder::denotesCurrentInstantiation(NestedNameSpecifierAST* nns,
                                         ClassSymbol* currentInstantiation)
    -> bool {
  if (!nns || !currentInstantiation) return false;
  auto qualifier = symbol_cast<ClassSymbol>(nns->symbol);
  if (!qualifier) return false;

  auto enclosesCurrentInstantiation = false;
  for (auto cls = currentInstantiation; cls;
       cls = symbol_cast<ClassSymbol>(cls->parent())) {
    if (cls == qualifier) {
      enclosesCurrentInstantiation = true;
      break;
    }
  }
  if (!enclosesCurrentInstantiation) return false;

  auto templateNns = ast_cast<TemplateNestedNameSpecifierAST>(nns);
  if (!templateNns) return true;

  return namesOwnTemplateParameters(templateNns->templateId, qualifier);
}

auto Binder::currentInstantiationOf(ScopeSymbol* scope) -> ClassSymbol* {
  for (auto current = scope; current; current = current->parent()) {
    if (auto classSymbol = symbol_cast<ClassSymbol>(current))
      return classSymbol;
  }
  return nullptr;
}

auto Binder::resolveMemberOfCurrentInstantiation(
    const Type* type, ClassSymbol* currentInstantiation) -> const Type* {
  if (!type || !currentInstantiation) return type;

  auto resolve = [&](const Type* nested) {
    return resolveMemberOfCurrentInstantiation(nested, currentInstantiation);
  };

  if (auto unresolved = type_cast<UnresolvedNameType>(type)) {
    if (!denotesCurrentInstantiation(unresolved->nestedNameSpecifier(),
                                     currentInstantiation)) {
      return type;
    }
    auto nameId = ast_cast<NameIdAST>(unresolved->unqualifiedId());
    if (!nameId) return type;
    auto qualifier =
        symbol_cast<ClassSymbol>(unresolved->nestedNameSpecifier()->symbol);
    auto member = qualifiedLookup(qualifier, nameId->identifier,
                                  [](Symbol* s) { return is_type(s); });
    if (!member || !member->type()) return type;
    return member->type();
  }

  if (auto qual = type_cast<QualType>(type)) {
    auto elementType = resolve(qual->elementType());
    if (elementType == qual->elementType()) return type;
    if (qual->isConst()) elementType = control()->getConstType(elementType);
    if (qual->isVolatile()) {
      elementType = control()->getVolatileType(elementType);
    }
    return elementType;
  }

  if (auto ptr = type_cast<PointerType>(type)) {
    auto elementType = resolve(ptr->elementType());
    if (elementType == ptr->elementType()) return type;
    return control()->getPointerType(elementType);
  }

  if (auto ref = type_cast<LvalueReferenceType>(type)) {
    auto elementType = resolve(ref->elementType());
    if (elementType == ref->elementType()) return type;
    return control()->getLvalueReferenceType(elementType);
  }

  if (auto ref = type_cast<RvalueReferenceType>(type)) {
    auto elementType = resolve(ref->elementType());
    if (elementType == ref->elementType()) return type;
    return control()->getRvalueReferenceType(elementType);
  }

  if (auto array = type_cast<BoundedArrayType>(type)) {
    auto elementType = resolve(array->elementType());
    if (elementType == array->elementType()) return type;
    return control()->getBoundedArrayType(elementType, array->size());
  }

  if (auto array = type_cast<UnboundedArrayType>(type)) {
    auto elementType = resolve(array->elementType());
    if (elementType == array->elementType()) return type;
    return control()->getUnboundedArrayType(elementType);
  }

  if (auto function = type_cast<FunctionType>(type)) {
    auto returnType = resolve(function->returnType());
    auto changed = returnType != function->returnType();
    std::vector<const Type*> parameterTypes;
    parameterTypes.reserve(function->parameterTypes().size());
    for (auto param : function->parameterTypes()) {
      auto resolved = resolve(param);
      changed = changed || resolved != param;
      parameterTypes.push_back(resolved);
    }
    if (!changed) return type;
    return control()->getFunctionType(
        returnType, std::move(parameterTypes), function->isVariadic(),
        function->cvQualifiers(), function->refQualifier(),
        function->isNoexcept());
  }

  return type;
}

auto Binder::getFunction(ScopeSymbol* scope, const Name* name, const Type* type,
                         TemplateDeclarationAST* templateHead)
    -> FunctionSymbol* {
  auto parentScope = scope;

  while (parentScope && parentScope->isTransparent()) {
    parentScope = parentScope->parent();
  }

  auto matches = [&](FunctionSymbol* function) {
    if (!areFunctionSignaturesEquivalentForRedeclaration(
            unit_, function->type(), type)) {
      return false;
    }
    return areTemplateHeadsEquivalentForRedeclaration(
        unit_, function->templateDeclaration(), templateHead);
  };

  if (auto parentClass = symbol_cast<ClassSymbol>(parentScope);
      parentClass && parentClass->name() == name) {
    for (auto ctor : parentClass->constructors()) {
      if (matches(ctor)) return ctor;
    }
  }

  return views::find_function(scope->find(name), matches);
}
}  // namespace cxx
