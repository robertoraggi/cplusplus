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
#include <cxx/type_traits.h>

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
#include <cxx/preprocessor.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {

namespace {

auto isDependentTypeParameterSymbol(Symbol* symbol) -> bool {
  return symbol_cast<TypeParameterSymbol>(symbol) ||
         symbol_cast<TemplateTypeParameterSymbol>(symbol);
}

auto isDependentNestedNameSpecifier(NestedNameSpecifierAST* ast) -> bool {
  if (!ast) return false;
  if (ast->symbol && ast->symbol->asScopeSymbol()) return false;
  if (isDependentTypeParameterSymbol(ast->symbol)) return true;
  return false;
}

}  // namespace

Binder::Binder(TranslationUnit* unit) : unit_(unit) {
  languageLinkage_ = unit_->language();
}

auto Binder::translationUnit() const -> TranslationUnit* { return unit_; }

auto Binder::control() const -> Control* { return unit_->control(); }

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

void Binder::note(SourceLocation loc, std::string message) {
  if (!reportErrors_) return;
  if (!unit_->config().checkTypes) return;
  unit_->note(loc, std::move(message));
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
  inTemplate_ = false;

  for (auto current = scope_; current; current = current->parent()) {
    if (current->isTemplateParameters()) {
      inTemplate_ = true;
      break;
    }
    if (auto cls = symbol_cast<ClassSymbol>(current)) {
      if (cls->templateParameters()) {
        inTemplate_ = true;
        break;
      }
    } else if (auto func = symbol_cast<FunctionSymbol>(current)) {
      if (func->templateParameters()) {
        inTemplate_ = true;
        break;
      }
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
    auto enumSymbol =
        control()->newScopedEnumSymbol(declaringScope(), location);
    ast->symbol = enumSymbol;

    enumSymbol->setName(enumName);
    enumSymbol->setUnderlyingType(underlyingType);
    scope()->addSymbol(enumSymbol);

    setScope(enumSymbol);
  } else {
    if (is_parsing_c() && ast->classLoc) {
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
      const bool isDependentNested =
          isDependentNestedNameSpecifier(ast->nestedNameSpecifier);
      if (!inTemplate() && !isDependentNested) {
        error(ast->nestedNameSpecifier->firstSourceLocation(),
              "nested name specifier must be a class or namespace");
      }
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
      // the class is declared in a different scope
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

  // decay the type of the parameters
  if (unit_->typeTraits().is_array(ast->type))
    ast->type = unit_->typeTraits().add_pointer(
        unit_->typeTraits().remove_extent(ast->type));
  else if (unit_->typeTraits().is_function(ast->type))
    ast->type = unit_->typeTraits().add_pointer(ast->type);
  else if (unit_->typeTraits().is_scalar(ast->type))
    ast->type = unit_->typeTraits().remove_cv(ast->type);

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
      ast->type =
          unit_->typeTraits().add_lvalue_reference(ast->expression->type);
    } else if (is_xvalue(ast->expression)) {
      ast->type =
          unit_->typeTraits().add_rvalue_reference(ast->expression->type);
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
            !unit_->typeTraits().is_same(existing->type(), symbol->type())) {
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
  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    const bool isDependentNested =
        isDependentNestedNameSpecifier(ast->nestedNameSpecifier);
    if (!inTemplate() && !isDependentNested) {
      error(ast->nestedNameSpecifier->firstSourceLocation(),
            "nested name specifier must be a class or namespace");
    }
    return;
  }

  if (auto u = symbol_cast<UsingDeclarationSymbol>(target)) {
    target = u->target();
  }

  if (!target) {
    if (!inTemplate()) {
      auto missingName = get_name(control(), ast->unqualifiedId);
      error(ast->unqualifiedId->firstSourceLocation(),
            std::format("using declaration refers to unresolved name '{}'",
                        to_string(missingName)));
    }
    return;
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

void Binder::bind(BaseSpecifierAST* ast, Symbol* resolvedType) {
  const auto checkTemplates = unit_->config().checkTypes;

  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    const bool isDependentNested =
        isDependentNestedNameSpecifier(ast->nestedNameSpecifier);
    if (!inTemplate() && !isDependentNested) {
      error(ast->nestedNameSpecifier->firstSourceLocation(),
            "nested name specifier must be a class or namespace");
    }
    return;
  }

  Symbol* symbol = nullptr;

  if (auto decltypeId = ast_cast<DecltypeIdAST>(ast->unqualifiedId)) {
    if (auto classType = type_cast<ClassType>(unit_->typeTraits().remove_cv(
            decltypeId->decltypeSpecifier->type))) {
      symbol = classType->symbol();
    }
  } else {
    symbol = resolve(ast->nestedNameSpecifier, ast->unqualifiedId,
                     checkTemplates, resolvedType);
  }

  // dealias
  if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType = type_cast<ClassType>(
            unit_->typeTraits().remove_cv(typeAlias->type()))) {
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
    unit_->typeTraits().requireCompleteClass(baseClass);
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

  if (auto* params = ast->parameterDeclarationClause) {
    for (auto it = params->parameterDeclarationList; it; it = it->next) {
      auto paramType = it->value ? it->value->type : nullptr;
      if (paramType && !type_cast<VoidType>(paramType))
        parameterTypes.push_back(paramType);
    }
    isVariadic = params->isVariadic;
  }

  auto* primaryTemplate =
      ast->templateId ? symbol_cast<ClassSymbol>(ast->templateId->symbol)
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
        auto spec = control()->newClassSymbol(parentScope, {});
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

      if (unit_->typeTraits().is_void(paramType)) {
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

  auto reboundBody = ast_cast<CompoundStatementAST>(
      ASTRewriter::paste(unit_, bodyScope, ast->statement));

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
          !unit_->typeTraits().is_same(existing->type(), symbol->type())) {
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

  if (!unit->typeTraits().is_same(lhsFn->returnType(), rhsFn->returnType()))
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
                   unit_->typeTraits().is_same(fn->type(), member->type());
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
                       unit_->typeTraits().is_same(fn->type(), m->type());
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

auto Binder::declareField(DeclaratorAST* declarator, const Decl& decl)
    -> FieldSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());

  if (name) {
    for (auto candidate : scope()->find(name)) {
      if (auto existingField = symbol_cast<FieldSymbol>(candidate)) {
        error(decl.location(),
              std::format("duplicate member '{}'", to_string(name)));
        return existingField;
      }
    }
  }

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

    if (!unit_->typeTraits().is_integral(type) &&
        !unit_->typeTraits().is_enum(type) && !inTemplate()) {
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
      } else if (!inTemplate()) {
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
  if (classSymbol->name()) return;  // not anonymous

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
  if (auto classType =
          type_cast<ClassType>(unit_->typeTraits().remove_cv(type))) {
    unit_->typeTraits().requireCompleteClass(classType->symbol());
  }
  if (addSymbolToParentScope) {
    for (auto candidate : targetScope->find(name)) {
      if (auto existing = symbol_cast<VariableSymbol>(candidate)) {
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

auto Binder::resolveNestedNameSpecifier(Symbol* symbol) -> ScopeSymbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(symbol)) {
    unit_->typeTraits().requireCompleteClass(classSymbol);
    return classSymbol;
  }

  if (auto injected = symbol_cast<InjectedClassNameSymbol>(symbol)) {
    unit_->typeTraits().requireCompleteClass(injected->classSymbol());
    return injected->classSymbol();
  }

  if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(symbol))
    return namespaceSymbol;

  if (auto enumSymbol = symbol_cast<EnumSymbol>(symbol)) return enumSymbol;

  if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(symbol))
    return scopedEnumSymbol;

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType = type_cast<ClassType>(typeAliasSymbol->type())) {
      unit_->typeTraits().requireCompleteClass(classType->symbol());
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

void Binder::bind(IdExpressionAST* ast) {
  if (!ast->unqualifiedId) {
    error(ast->firstSourceLocation(),
          "expected an unqualified identifier in id expression");
    return;
  }

  if (ast->nestedNameSpecifier) {
    if (!ast->nestedNameSpecifier->symbol) {
      return;
    }

    auto name = get_name(control(), ast->unqualifiedId);

    const Name* componentName = name;

    if (auto templateId = name_cast<TemplateId>(name)) {
      componentName = templateId->name();
    }

    ast->symbol =
        qualifiedLookup(ast->nestedNameSpecifier->symbol, componentName);
  }

  // For unqualified ids, the parser has already resolved ast->symbol
  // via unqualified lookup before calling bind().

  resolveIdExpression(ast);
}

void Binder::qualifiedLookupIdExpression(IdExpressionAST* ast) {
  if (!ast->unqualifiedId) return;
  if (!ast->nestedNameSpecifier || !ast->nestedNameSpecifier->symbol) return;

  if (auto classSymbol =
          symbol_cast<ClassSymbol>(ast->nestedNameSpecifier->symbol)) {
    unit_->typeTraits().requireCompleteClass(classSymbol);
  }

  auto name = get_name(control(), ast->unqualifiedId);
  const Name* componentName = name;
  if (auto templateId = name_cast<TemplateId>(name))
    componentName = templateId->name();

  ast->symbol =
      qualifiedLookup(ast->nestedNameSpecifier->symbol, componentName);

  resolveIdExpression(ast);
}

void Binder::resolveIdExpression(IdExpressionAST* ast) {
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
        // If the template has a trailing pack and the explicit args only cover
        // the non-pack parameters, defer so function arguments can fill the
        // pack via call-site deduction.
        if (arity.hasParameterPack && argc > 0 && argc == arity.minArgs)
          return true;
        return false;
      };

      if (auto var = symbol_cast<VariableSymbol>(ast->symbol)) {
        if (!inTemplate()) {
          templateSymbol = var;
        }
      } else if (auto func = symbol_cast<FunctionSymbol>(ast->symbol)) {
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
            error(templateId->firstSourceLocation(),
                  std::format("invalid template-id '{}'",
                              to_string(templateIdName)));
          } else {
            error(templateId->firstSourceLocation(),
                  std::format("not a template"));
          }
        }
      } else {
        if (inTemplate()) return;

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

auto Binder::getFunction(ScopeSymbol* scope, const Name* name, const Type* type)
    -> FunctionSymbol* {
  auto parentScope = scope;

  while (parentScope && parentScope->isTransparent()) {
    parentScope = parentScope->parent();
  }

  if (auto parentClass = symbol_cast<ClassSymbol>(parentScope);
      parentClass && parentClass->name() == name) {
    for (auto ctor : parentClass->constructors()) {
      if (areFunctionSignaturesEquivalentForRedeclaration(unit_, ctor->type(),
                                                          type)) {
        return ctor;
      }
    }
  }

  for (auto candidate : scope->find(name)) {
    if (auto function = symbol_cast<FunctionSymbol>(candidate)) {
      if (areFunctionSignaturesEquivalentForRedeclaration(
              unit_, function->type(), type)) {
        return function;
      }
    } else if (auto overloads = symbol_cast<OverloadSetSymbol>(candidate)) {
      for (auto function : overloads->functions()) {
        if (areFunctionSignaturesEquivalentForRedeclaration(
                unit_, function->type(), type)) {
          return function;
        }
      }
    }
  }

  return nullptr;
}

}  // namespace cxx
