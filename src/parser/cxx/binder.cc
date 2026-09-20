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

#include <cxx/access_control.h>
#include <cxx/ast.h>
#include <cxx/ast_interpreter.h>
#include <cxx/ast_rewriter.h>
#include <cxx/ast_visitor.h>
#include <cxx/attributes.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/decl_specs.h>
#include <cxx/dependent_types.h>
#include <cxx/function_body_warnings.h>
#include <cxx/lambda_captures.h>
#include <cxx/literals.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/standard_conversion.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>

namespace cxx {
namespace {
auto redeclarationTypesEquivalent(TranslationUnit* unit,
                                  const Type* existingType,
                                  const Type* incomingType,
                                  bool ignoresArrayBound = true) -> bool;
}

auto Binder::closureNamingState() const -> ClosureNamingState {
  return {control()->closureNameCount(), lambdaDiscriminators_};
}

void Binder::setClosureNamingState(ClosureNamingState state) {
  control()->setClosureNameCount(state.lambdaCount);
  lambdaDiscriminators_ = std::move(state.lambdaDiscriminators);
}

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
  return inTemplate_ || explicitTemplateHeadDepth_ > 0 ||
         retainsEnclosingTemplateLevels_;
}

void Binder::enterExplicitTemplateHead() { ++explicitTemplateHeadDepth_; }

auto Binder::inDiscardedStatement() const -> bool {
  return discardedStatementDepth_ > 0;
}

void Binder::leaveExplicitTemplateHead() { --explicitTemplateHeadDepth_; }

void Binder::enterExplicitInstantiation(bool isDefinition) {
  ++explicitInstantiationDepth_;
  explicitInstantiationIsDefinition_ = isDefinition;
}

void Binder::leaveExplicitInstantiation() { --explicitInstantiationDepth_; }

auto Binder::inExplicitInstantiation() const -> bool {
  return explicitInstantiationDepth_ > 0;
}

auto Binder::inExplicitInstantiationDefinition() const -> bool {
  return inExplicitInstantiation() && explicitInstantiationIsDefinition_;
}

void Binder::setRetainsEnclosingTemplateLevels(bool value) {
  retainsEnclosingTemplateLevels_ = value;
}

namespace {
struct FindReturnedValue final : ASTVisitor {
  bool found = false;

  auto preVisit(AST*) -> bool override { return !found; }

  void visit(ReturnStatementAST* ast) override {
    if (ast->expression) found = true;
  }

  void visit(LambdaExpressionAST*) override {}
};
}  // namespace

auto Binder::returnsAValue(AST* declaration) -> bool {
  if (!declaration) return false;
  FindReturnedValue scan;
  scan.accept(declaration);
  return scan.found;
}

void Binder::finishAutoReturnType(FunctionSymbol* functionSymbol) {
  if (!functionSymbol) return;
  auto funcType = type_cast<FunctionType>(functionSymbol->type());
  if (!funcType) return;
  if (!isPlaceholderType(funcType->returnType())) return;
  if (returnsAValue(functionSymbol->declaration())) return;

  auto newFuncType = control()->getFunctionType(
      control()->getVoidType(),
      std::vector<const Type*>(funcType->parameterTypes().begin(),
                               funcType->parameterTypes().end()),
      funcType->isVariadic(), funcType->cvQualifiers(),
      funcType->refQualifier(), funcType->exceptionSpecification());
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
  return scope_->parent();
}

auto Binder::classBeingDefined() const -> ClassSymbol* {
  if (classBodyStack_.empty()) return nullptr;
  return classBodyStack_.back().classSymbol;
}

auto Binder::currentAccessSpecifier() const -> AccessSpecifier {
  if (classBodyStack_.empty()) return AccessSpecifier::kPublic;
  return classBodyStack_.back().accessSpecifier;
}

auto Binder::defaultAccessSpecifier() const -> AccessSpecifier {
  if (classBodyStack_.empty()) return AccessSpecifier::kPublic;
  return classBodyStack_.back().defaultAccessSpecifier;
}

void Binder::setCurrentAccessSpecifier(AccessSpecifier accessSpecifier) {
  if (classBodyStack_.empty()) return;
  classBodyStack_.back().accessSpecifier = accessSpecifier;
}

void Binder::applyAccessSpecifier(Symbol* symbol) const {
  if (!symbol) return;
  if (symbol_cast<ClassSymbol>(symbol->parent()) != classBeingDefined()) return;
  symbol->setAccessSpecifier(currentAccessSpecifier());
}

namespace {
template <typename S>
void applyTemplateHead(Binder& binder, S* symbol,
                       TemplateDeclarationAST* templateHead) {
  if (!symbol) return;

  if (templateHead) {
    binder.mergeTemplateParameterDefaults(
        symbol->canonical()->templateParameters(), templateHead->symbol);
  }

  symbol->setTemplateDeclaration(templateHead);

  if (!templateHead) return;

  symbol->setTemplateParameters(templateHead->symbol);
  binder.checkDefaultTemplateArgumentOnPack(templateHead->symbol);
}
}  // namespace

void Binder::setTemplateHead(FunctionSymbol* symbol,
                             TemplateDeclarationAST* templateHead) {
  applyTemplateHead(*this, symbol, templateHead);
}

void Binder::setTemplateHead(VariableSymbol* symbol,
                             TemplateDeclarationAST* templateHead) {
  applyTemplateHead(*this, symbol, templateHead);
}

void Binder::setTemplateHead(TypeAliasSymbol* symbol,
                             TemplateDeclarationAST* templateHead) {
  applyTemplateHead(*this, symbol, templateHead);
}

void Binder::checkTemplateParameterDefaultOrder(
    TemplateParametersSymbol* parameters) {
  if (!parameters) return;

  Symbol* defaulted = nullptr;

  for (auto parameter : parameters->members()) {
    if (default_template_argument(parameter)) {
      defaulted = parameter;
      continue;
    }

    if (!defaulted) continue;

    if (is_template_parameter_pack(parameter)) continue;

    error(parameter->location(),
          "template parameter missing a default argument");
    note(defaulted->location(),
         "previous default template argument defined here");
    return;
  }
}

void Binder::checkDefaultTemplateArgumentOnPack(
    TemplateParametersSymbol* parameters) {
  if (!parameters) return;

  for (auto parameter : parameters->members()) {
    if (!is_template_parameter_pack(parameter)) continue;
    if (!default_template_argument(parameter)) continue;

    error(parameter->location(),
          "a template parameter pack cannot have a default argument");
    return;
  }
}

void Binder::rejectDefaultTemplateArguments(
    TemplateDeclarationAST* templateHead, std::string message) {
  if (!templateHead) return;

  for (auto parameter : ListView{templateHead->templateParameterList}) {
    if (!hasWrittenDefaultTemplateArgument(parameter)) continue;
    error(parameter->firstSourceLocation(), std::move(message));
    return;
  }
}

void Binder::copyDefaultArguments(FunctionParametersSymbol* from,
                                  FunctionParametersSymbol* to) {
  if (!from || !to || from == to) return;

  auto sources = from->members() | views::parameters;
  auto targets = to->members() | views::parameters;

  auto sourceIt = sources.begin();
  auto targetIt = targets.begin();

  for (; sourceIt != sources.end() && targetIt != targets.end();
       ++sourceIt, ++targetIt) {
    auto source = *sourceIt;
    auto target = *targetIt;

    if (target->defaultArgument()) continue;
    if (!source->defaultArgument()) continue;

    setSpeculativeValue(
        target->defaultArgument(), source->defaultArgument(),
        [target](ExpressionAST* value) { target->setDefaultArgument(value); });
  }
}

void Binder::mergeTemplateParameterDefaults(
    TemplateParametersSymbol* accumulated, TemplateParametersSymbol* incoming) {
  if (!accumulated || !incoming) return;
  if (accumulated == incoming) return;

  const auto& previousParameters = accumulated->members();
  const auto& currentParameters = incoming->members();
  const auto count =
      std::min(previousParameters.size(), currentParameters.size());

  for (std::size_t index = 0; index < count; ++index) {
    auto previous = previousParameters[index];
    auto current = currentParameters[index];
    auto previousDefault = default_template_argument(previous);
    auto currentDefault = default_template_argument(current);

    if (previousDefault == currentDefault) continue;

    if (!currentDefault) {
      setSpeculativeValue(currentDefault, previousDefault,
                          [current](TemplateParameterAST* value) {
                            set_default_template_argument(current, value);
                          });
      continue;
    }

    if (!previousDefault) {
      setSpeculativeValue(previousDefault, currentDefault,
                          [previous](TemplateParameterAST* value) {
                            set_default_template_argument(previous, value);
                          });
      continue;
    }

    error(current->location(), "redefinition of default template argument");
    note(previous->location(), "previous definition is here");
  }
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
  if (target) u->setType(target->type());
  scope->addSymbol(u);
}

auto Binder::scope() const -> ScopeSymbol* { return scope_; }

void Binder::setScope(ScopeSymbol* scope) {
  scope_ = scope;
  inTemplate_ = isEnclosedInDependentTemplate(
      unit_, scope_, /*stopAtConcreteSpecialization=*/true);
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

auto Binder::declareEnum(const Name* name, SourceLocation location,
                         const Type* underlyingType, bool scoped,
                         bool fixedUnderlyingType, bool isDefinition,
                         bool isValidDeclaration) -> ScopeSymbol* {
  auto effectiveUnderlyingType = unit_->typeTraits().remove_cv(underlyingType);
  const auto invalidUnderlyingType =
      (scoped || fixedUnderlyingType) &&
      !unit_->typeTraits().is_integral(effectiveUnderlyingType) &&
      !isDependent(unit_, effectiveUnderlyingType);
  if (invalidUnderlyingType) {
    error(location, "enumeration underlying type must be integral");
    effectiveUnderlyingType = control()->getIntType();
  }

  auto createEnum = [&](bool addToScope) -> ScopeSymbol* {
    auto enclosingScope = declaringScope();
    if (!addToScope)
      enclosingScope = control()->newBlockSymbol(enclosingScope, location);
    if (scoped) {
      auto symbol = control()->newScopedEnumSymbol(enclosingScope, location);
      symbol->setName(name);
      symbol->setUnderlyingType(effectiveUnderlyingType);
      symbol->setDefined(isDefinition);
      if (addToScope) scope()->addSymbol(symbol);
      return symbol;
    }

    auto symbol = control()->newEnumSymbol(enclosingScope, location);
    symbol->setName(name);
    symbol->setUnderlyingType(effectiveUnderlyingType);
    symbol->setHasFixedUnderlyingType(fixedUnderlyingType);
    symbol->setDefined(isDefinition);
    if (addToScope) scope()->addSymbol(symbol);
    return symbol;
  };

  if (invalidUnderlyingType || !isValidDeclaration) return createEnum(false);

  if (!name) return createEnum(true);

  for (auto candidate : declaringScope()->find(name)) {
    auto existingEnum = symbol_cast<EnumSymbol>(candidate);
    auto existingScopedEnum = symbol_cast<ScopedEnumSymbol>(candidate);
    if (!existingEnum && !existingScopedEnum) {
      if (symbol_cast<ClassSymbol>(candidate) ||
          symbol_cast<TypeAliasSymbol>(candidate)) {
        error(location,
              std::format("conflicting declaration of '{}'", to_string(name)));
        return createEnum(false);
      }
      continue;
    }
    if (scoped != (existingScopedEnum != nullptr)) {
      error(location,
            std::format("enumeration '{}' redeclared with different scopedness",
                        to_string(name)));
      return createEnum(false);
    }

    const auto existingType = existingEnum
                                  ? existingEnum->underlyingType()
                                  : existingScopedEnum->underlyingType();
    const auto existingFixed =
        existingEnum ? existingEnum->hasFixedUnderlyingType() : true;
    const auto newFixed = scoped || fixedUnderlyingType;
    if (existingFixed != newFixed) {
      error(location,
            std::format(
                "enumeration '{}' redeclared with incompatible underlying type",
                to_string(name)));
      return createEnum(false);
    }
    if (existingFixed && !unit_->typeTraits().is_same(
                             unit_->typeTraits().remove_cv(existingType),
                             effectiveUnderlyingType)) {
      error(location,
            std::format(
                "enumeration '{}' redeclared with different underlying type",
                to_string(name)));
      return createEnum(false);
    }

    const auto defined = existingEnum ? existingEnum->isDefined()
                                      : existingScopedEnum->isDefined();
    if (isDefinition && defined) {
      error(location,
            std::format("redefinition of enumeration '{}'", to_string(name)));
      return createEnum(false);
    }
    if (isDefinition) {
      if (existingEnum)
        existingEnum->setDefined(true);
      else
        existingScopedEnum->setDefined(true);
    }
    if (existingEnum) return existingEnum;
    return existingScopedEnum;
  }

  return createEnum(true);
}

void Binder::bind(EnumSpecifierAST* ast, const DeclSpecs& underlyingTypeSpecs) {
  const auto underlyingType = underlyingTypeSpecs.hasTypeOrSizeSpecifier()
                                  ? underlyingTypeSpecs.type()
                                  : control()->getIntType();
  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->lbraceLoc;
  if (isC() && ast->classLoc)
    error(ast->classLoc, "scoped enums are not allowed in C");
  ast->symbol = declareEnum(get_name(control(), ast->unqualifiedId), location,
                            underlyingType, ast->classLoc && isCxx(),
                            ast->typeSpecifierList != nullptr, true);
  applyAccessSpecifier(ast->symbol);
  applyDeclarationAttributes(ast->symbol, ast->attributeList);
  setScope(ast->symbol->asScopeSymbol());
}

void Binder::bind(OpaqueEnumDeclarationAST* ast,
                  const DeclSpecs& underlyingTypeSpecs) {
  const auto underlyingType = underlyingTypeSpecs.hasTypeOrSizeSpecifier()
                                  ? underlyingTypeSpecs.type()
                                  : control()->getIntType();
  const auto location = ast->unqualifiedId
                            ? ast->unqualifiedId->firstSourceLocation()
                            : ast->enumLoc;
  if (isC() && ast->classLoc)
    error(ast->classLoc, "scoped enums are not allowed in C");
  const auto missingUnscopedBase =
      isCxx() && !ast->classLoc && !ast->typeSpecifierList;
  if (missingUnscopedBase)
    error(location,
          "opaque declaration of an unscoped enumeration requires an "
          "underlying type");
  if (ast->nestedNameSpecifier)
    error(location,
          "opaque enumeration declaration cannot have a qualified name");
  ast->symbol =
      declareEnum(get_name(control(), ast->unqualifiedId), location,
                  underlyingType, ast->classLoc && isCxx(), true, false,
                  !missingUnscopedBase && !ast->nestedNameSpecifier);
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
          for (auto candidate : s->find(name)) {
            auto hiddenClass = symbol_cast<ClassSymbol>(candidate);
            if (!hiddenClass) continue;
            if (hiddenClass->isFriend()) return hiddenClass;
          }
        }
        return nullptr;
      }
      if (ast->nestedNameSpecifier)
        return qualifiedLookup(ast->nestedNameSpecifier->symbol, name,
                               is_class);
      return unqualifiedCandidate;
    }();

    auto classSymbol = symbol_cast<ClassSymbol>(candidate);

    if (classSymbol && isDeclaration && classSymbol->parent() != targetScope) {
      classSymbol = nullptr;
    }

    auto adoptedClassSymbol = static_cast<ClassSymbol*>(nullptr);
    if (!classSymbol && !declSpecs.isFriend) {
      adoptedClassSymbol = adoptFriendDeclaredClass(targetScope, name);
      classSymbol = adoptedClassSymbol;
    }

    if (adoptedClassSymbol) {
      adoptedClassSymbol->setIsUnion(ast->classKey == TokenKind::T_UNION);
      adoptedClassSymbol->setTemplateDeclaration(declSpecs.templateHead);
      if (declSpecs.templateHead) {
        adoptedClassSymbol->setTemplateParameters(
            declSpecs.templateHead->symbol);
      }
      adoptedClassSymbol->setDeclaration(ast);
    }

    if (!classSymbol) {
      const auto isUnion = ast->classKey == TokenKind::T_UNION;
      classSymbol = control()->newClassSymbol(targetScope, location);

      applyAccessSpecifier(classSymbol);
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
    } else if (declSpecs.templateHead && isDeclaration) {
      if (classSymbol->templateParameters()) {
        mergeTemplateParameterDefaults(classSymbol->templateParameters(),
                                       declSpecs.templateHead->symbol);
      } else {
        classSymbol->setTemplateDeclaration(declSpecs.templateHead);
        classSymbol->setTemplateParameters(declSpecs.templateHead->symbol);
      }
    }

    if (declSpecs.templateHead && isDeclaration && declSpecs.isFriend) {
      rejectDefaultTemplateArguments(
          declSpecs.templateHead,
          "a default template argument cannot be specified on a friend "
          "template declaration");
    }

    if (declSpecs.templateHead && isDeclaration) {
      checkDefaultTemplateArgumentOnPack(classSymbol->templateParameters());
    }

    if (declSpecs.templateHead && isDeclaration && !declSpecs.isFriend) {
      checkTemplateParameterDefaultOrder(classSymbol->templateParameters());
    }

    ast->symbol = classSymbol;

    if (auto alignment = explicitAlignment(ast->attributeList, location)) {
      checkRedeclaredAlignment(classSymbol, *alignment, location);
    }

    if (declSpecs.isFriend && !templateId && classBeingDefined()) {
      classSymbol->canonical()->addBefriendingClass(classBeingDefined());
    }
  }

  declSpecs.setTypeSpecifier(ast);

  if (ast->symbol) {
    declSpecs.setType(ast->symbol->type());
  }
}

void Binder::enterSpeculativeDeclarations() { ++speculationDepth_; }

void Binder::leaveSpeculativeDeclarations() {
  if (--speculationDepth_ == 0) speculativeMutations_.clear();
}

void Binder::recordSpeculativeMutation(std::function<void()> undo) {
  if (!speculationDepth_) return;
  speculativeMutations_.push_back(std::move(undo));
}

void Binder::recordSpeculativeOverload(OverloadSetSymbol* overloadSet) {
  if (!speculationDepth_) return;
  auto functionCount = overloadSet->declaredFunctions().size();
  recordSpeculativeMutation([overloadSet, functionCount] {
    overloadSet->truncateFunctions(functionCount);
  });
}

void Binder::undoSpeculativeMutations(std::size_t count) {
  while (speculativeMutations_.size() > count) {
    auto undo = std::move(speculativeMutations_.back());
    speculativeMutations_.pop_back();
    undo();
  }
}

auto Binder::adoptFriendDeclaredClass(ScopeSymbol* targetScope,
                                      const Identifier* name) -> ClassSymbol* {
  if (!name) return nullptr;

  for (auto candidate : targetScope->find(name)) {
    auto classSymbol = symbol_cast<ClassSymbol>(candidate);
    if (!classSymbol) continue;
    if (!classSymbol->isFriend()) continue;
    if (classSymbol->parent() != targetScope) continue;

    classSymbol->setFriend(false);
    classSymbol->setHidden(false);
    return classSymbol;
  }

  return nullptr;
}

void Binder::disableAccessControlForUnsupportedFriend(
    NestedNameSpecifierAST* nestedNameSpecifier,
    ClassSymbol* befriendingClass) {
  if (!inTemplate() || !isDependent(unit_, nestedNameSpecifier)) return;
  if (!befriendingClass) return;

  befriendingClass->setAccessControlDisabled(true);
}

void Binder::checkExceptionDeclarationType(TypeExceptionDeclarationAST* ast,
                                           const Type* type) {
  auto loc = ast->firstSourceLocation();

  if (traits.is_rvalue_reference(type)) {
    error(loc, std::format("cannot catch an rvalue reference of type '{}'",
                           to_string(type)));
    return;
  }

  const auto isReference = traits.is_reference(type);
  auto declaredType = traits.remove_reference(type);

  if (!traits.is_complete(declaredType)) {
    error(loc, std::format("cannot catch an incomplete type '{}'",
                           to_string(declaredType)));
    return;
  }

  if (!isReference && traits.is_abstract(declaredType)) {
    error(loc, std::format("cannot catch an abstract class type '{}'",
                           to_string(declaredType)));
    return;
  }

  auto pointee = traits.remove_pointer(declaredType);
  if (pointee == declaredType) return;
  if (traits.is_void(traits.remove_cv(pointee))) return;

  if (!traits.is_complete(pointee)) {
    error(loc, std::format("cannot catch a pointer to the incomplete type '{}'",
                           to_string(pointee)));
  }
}

void Binder::checkTrailingRequiresClauseIsTemplated(
    FunctionSymbol* functionSymbol, TemplateDeclarationAST* templateHead) {
  if (!functionSymbol) return;

  auto requiresClause = functionSymbol->trailingRequiresClause();
  if (!requiresClause) return;
  if (templateHead) return;
  if (functionSymbol->templateDeclaration()) return;
  if (isInstantiating()) return;
  if (unit_->isInstantiatingTemplate()) return;

  for (auto s = functionSymbol->parent(); s; s = s->parent()) {
    if (s->isTemplateParameters()) return;
    if (auto classSymbol = symbol_cast<ClassSymbol>(s)) {
      if (classSymbol->templateDeclaration()) return;
      if (classSymbol->isSpecialization()) return;
    }
    if (auto function = symbol_cast<FunctionSymbol>(s)) {
      if (function->templateDeclaration()) return;
    }
  }

  error(requiresClause->firstSourceLocation(),
        "non-templated function cannot have a requires clause");
}

void Binder::bind(TypeExceptionDeclarationAST* ast, const Decl& decl) {
  if (explicitAlignment(ast->attributeList, ast->firstSourceLocation())) {
    error(ast->firstSourceLocation(),
          "'alignas' attribute cannot be applied to an exception declaration");
  }

  auto type = traits.adjusted_parameter_type(
      getDeclaratorType(unit_, ast->declarator, decl.specs.type()));

  auto declaredType = traits.remove_reference(type);

  if (auto classType = unqualified_cast<ClassType>(declaredType)) {
    traits.requireCompleteClass(classType->symbol());
  } else if (auto pointee = traits.remove_pointer(declaredType);
             pointee != declaredType) {
    if (auto classType = unqualified_cast<ClassType>(pointee))
      traits.requireCompleteClass(classType->symbol());
  }

  checkExceptionDeclarationType(ast, type);

  auto location = decl.location();
  if (!location) location = ast->firstSourceLocation();

  auto symbol = control()->newVariableSymbol(scope_, location);
  symbol->setName(decl.getName());
  symbol->setType(type);
  ast->symbol = symbol;

  if (symbol->name()) scope_->addSymbol(symbol);
}

void Binder::bind(ParameterDeclarationAST* ast, const Decl& decl,
                  bool inTemplateParameters) {
  auto parameterObjectType = traits.adjusted_parameter_type(
      getDeclaratorType(unit_, ast->declarator, decl.specs.type()));

  ast->type = unqualified_type(parameterObjectType);

  if (explicitAlignment(ast->attributeList, decl.location())) {
    error(decl.location(),
          "'alignas' attribute cannot be applied to a function parameter");
  }

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

    const auto isFirstParameter = scope_->members().empty();

    if (ast->isThisIntroduced && !isFirstParameter) {
      error(ast->thisLoc,
            "an explicit object parameter must be the first parameter");
    }

    if (ast->isThisIntroduced && decl.isPack) {
      error(ast->thisLoc,
            "an explicit object parameter cannot be a function parameter pack");
    }

    auto parameterSymbol = control()->newParameterSymbol(scope_, parameterLoc);
    parameterSymbol->setName(ast->identifier);
    parameterSymbol->setType(parameterObjectType);
    parameterSymbol->setDefaultArgument(ast->expression);
    parameterSymbol->setExplicitObject(ast->isThisIntroduced &&
                                       isFirstParameter);
    scope_->addSymbol(parameterSymbol);
    ast->symbol = parameterSymbol;
  }
}

void Binder::bind(DecltypeSpecifierAST* ast) {
  if (auto type = traits.decltype_of(ast->expression)) ast->type = type;
}

auto Binder::nextEnumeratorValue(TranslationUnit* unit,
                                 const Type* underlyingType,
                                 const std::optional<ConstValue>& previous)
    -> std::optional<ConstValue> {
  if (!previous.has_value()) return std::intmax_t{0};

  ASTInterpreter interp{unit};

  if (unit->typeTraits().is_unsigned(underlyingType)) {
    if (auto v = interp.toUInt(previous.value()))
      return std::bit_cast<std::intmax_t>(v.value() + 1);
    return std::nullopt;
  }

  if (auto v = interp.toInt(previous.value())) return v.value() + 1;
  return std::nullopt;
}

void Binder::bind(EnumeratorAST* ast, const Type* type,
                  std::optional<ConstValue> value) {
  if (isCxx()) {
    auto symbol = control()->newEnumeratorSymbol(scope(), ast->identifierLoc);
    ast->symbol = symbol;

    symbol->setName(ast->identifier);
    symbol->setType(type);
    ast->symbol->setValue(value);
    if (auto enclosingEnum = symbol_cast<EnumSymbol>(scope()))
      symbol->setAccessSpecifier(enclosingEnum->accessSpecifier());
    if (auto enclosingEnum = symbol_cast<ScopedEnumSymbol>(scope()))
      symbol->setAccessSpecifier(enclosingEnum->accessSpecifier());
    scope()->addSymbol(symbol);

    if (auto enumSymbol = symbol_cast<EnumSymbol>(scope())) {
      auto parentScope = enumSymbol->parent();

      auto u =
          control()->newUsingDeclarationSymbol(parentScope, ast->identifierLoc);
      u->setName(ast->identifier);
      u->setTarget(symbol);
      parentScope->addSymbol(u);
      applyAccessSpecifier(u);
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

void Binder::addTypeAliasToScope(TypeAliasSymbol* symbol) {
  auto scope = symbol->parent();
  auto name = symbol->name();
  auto aliasesNamedType = [&](Symbol* candidate) {
    if (isC()) {
      if (symbol_cast<ClassSymbol>(candidate)) return true;
      if (symbol_cast<EnumSymbol>(candidate)) return true;
    }
    if (auto type = type_cast<ClassType>(symbol->type())) {
      if (type->symbol() == candidate) return true;
    }
    if (auto type = type_cast<EnumType>(symbol->type())) {
      if (type->symbol() == candidate) return true;
    }
    if (auto type = type_cast<ScopedEnumType>(symbol->type())) {
      if (type->symbol() == candidate) return true;
    }
    return false;
  };

  for (auto declaration : scope->find(name)) {
    auto candidate = resolve_using_declaration(declaration);
    if (auto existing = symbol_cast<TypeAliasSymbol>(candidate)) {
      auto equivalent = TemplateEquivalence{unit_}.same(
          existing->templateDeclaration(), symbol->templateDeclaration());
      if (existing->type() && symbol->type()) {
        if (!redeclarationTypesEquivalent(unit_, existing->type(),
                                          symbol->type(), false)) {
          equivalent = false;
        }
      }
      if (equivalent) {
        addRedeclaration(existing->canonical(), symbol);
        break;
      }
    } else if (aliasesNamedType(candidate)) {
      continue;
    }
    error(symbol->location(),
          std::format("conflicting declaration of '{}'", to_string(name)));
    return;
  }
  scope->addSymbol(symbol);
}

auto Binder::declareTypeAlias(SourceLocation identifierLoc,
                              const Identifier* identifier, TypeIdAST* typeId,
                              bool addSymbolToParentScope,
                              TemplateDeclarationAST* templateHead)
    -> TypeAliasSymbol* {
  auto symbol = control()->newTypeAliasSymbol(declaringScope(), identifierLoc);
  applyAccessSpecifier(symbol);

  auto name = identifier;
  symbol->setName(name);

  if (typeId) symbol->setType(typeId->type);
  setTemplateHead(symbol, templateHead);
  checkTemplateParameterDefaultOrder(symbol->canonical()->templateParameters());

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

  if (addSymbolToParentScope) addTypeAliasToScope(symbol);

  return symbol;
}

namespace {

[[nodiscard]] auto joinsFunctionOverloadSet(Symbol* candidate) -> bool {
  if (symbol_cast<OverloadSetSymbol>(candidate)) return true;
  if (symbol_cast<FunctionSymbol>(candidate)) return true;
  auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(candidate);
  return usingDeclaration && !usingDeclaration->introducedFunctions().empty();
}

struct TerminalNestedNameSpecifierName {
  auto operator()(GlobalNestedNameSpecifierAST*) const -> const Identifier* {
    return nullptr;
  }

  auto operator()(SimpleNestedNameSpecifierAST* ast) const
      -> const Identifier* {
    return ast->identifier;
  }

  auto operator()(DecltypeNestedNameSpecifierAST*) const -> const Identifier* {
    return nullptr;
  }

  auto operator()(TemplateNestedNameSpecifierAST* ast) const
      -> const Identifier* {
    return ast->templateId ? ast->templateId->identifier : nullptr;
  }
};

}  // namespace

auto Binder::usingDeclaratorNamesConstructor(UsingDeclaratorAST* ast) -> bool {
  if (!ast->nestedNameSpecifier || ast->typenameLoc) return false;
  auto terminal =
      visit(TerminalNestedNameSpecifierName{}, ast->nestedNameSpecifier);
  if (!terminal) return false;
  auto nameId = ast_cast<NameIdAST>(ast->unqualifiedId);
  return nameId && nameId->identifier == terminal;
}

auto Binder::bindInheritedConstructors(UsingDeclaratorAST* ast) -> bool {
  auto derived = symbol_cast<ClassSymbol>(scope());
  if (!derived) return false;

  auto lookupContext =
      ast->nestedNameSpecifier ? ast->nestedNameSpecifier->symbol : nullptr;
  auto base = symbol_cast<ClassSymbol>(lookupContext);
  if (!base) return isDependent(unit_, ast->nestedNameSpecifier);
  if (isDependent(unit_, base->type())) return true;
  base = base->resolvedDefinition();

  const auto isDirectBase =
      std::ranges::any_of(derived->baseClasses(), [&](BaseClassSymbol* b) {
        auto candidate = symbol_cast<ClassSymbol>(b->symbol());
        return candidate && candidate->resolvedDefinition() == base;
      });

  if (!isDirectBase) {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("'{}' is not a direct base class of '{}'",
                      to_string(base->name()), to_string(derived->name())));
    return true;
  }

  auto symbol = control()->newUsingDeclarationSymbol(
      derived, ast->unqualifiedId->firstSourceLocation());
  ast->symbol = symbol;
  symbol->setName(derived->name());
  symbol->setDeclarator(ast);
  symbol->setTarget(base->constructorOverloadSet());

  derived->constructorOverloadSet()->addUsingDeclaration(symbol);
  return true;
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

  if (usingDeclaratorNamesConstructor(ast)) {
    if (bindInheritedConstructors(ast)) return;
  }

  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    if (reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier)) return;
  }

  const bool dependentQualifier =
      inTemplate() && isDependent(unit_, ast->nestedNameSpecifier);

  if (dependentQualifier) target = makeDependentTypeTarget();

  if (auto u = symbol_cast<UsingDeclarationSymbol>(target)) {
    target = resolve_using_declaration(target);
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

  applyAccessSpecifier(symbol);
  symbol->setName(name);
  symbol->setDeclarator(ast);
  symbol->setTarget(target);

  if (!dependentQualifier) checkUsingDeclaratorAccess(ast, symbol);

  const auto joinsAnOverloadSet =
      !symbol->introducedFunctions().empty() &&
      std::ranges::any_of(scope()->find(name), joinsFunctionOverloadSet);

  if (!joinsAnOverloadSet) {
    scope()->addSymbol(symbol);
    return;
  }

  overloadSetFor(scope(), name, symbol->location())
      ->addUsingDeclaration(symbol);
}

void Binder::checkUsingDeclaratorAccess(UsingDeclaratorAST* ast,
                                        UsingDeclarationSymbol* symbol) {
  if (usingDeclaratorNamesConstructor(ast)) return;
  if (!ast->nestedNameSpecifier) return;

  auto designatingClass =
      symbol_cast<ClassSymbol>(ast->nestedNameSpecifier->symbol);
  if (!designatingClass) return;

  const auto location = ast->unqualifiedId->firstSourceLocation();

  auto reportInaccessible = [&](Symbol* named) {
    if (!named) return;
    (void)checkMemberAccess(unit_, scope(), named, designatingClass, nullptr,
                            location);
  };

  auto introduced = symbol->introducedFunctions();
  if (introduced.empty()) {
    reportInaccessible(symbol->target());
    return;
  }

  for (auto function : introduced) reportInaccessible(function);
}

void Binder::bind(BaseSpecifierAST* ast, Symbol* resolvedType) {
  const auto checkTemplates = unit_->config().checkTypes;

  if (ast->nestedNameSpecifier && !ast->nestedNameSpecifier->symbol) {
    (void)reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier);
    return;
  }

  Symbol* symbol = nullptr;

  if (auto decltypeId = ast_cast<DecltypeIdAST>(ast->unqualifiedId)) {
    if (auto classType =
            unqualified_cast<ClassType>(decltypeId->decltypeSpecifier->type)) {
      symbol = classType->symbol();
    }
  } else {
    symbol = resolve(ast->nestedNameSpecifier, ast->unqualifiedId,
                     checkTemplates, resolvedType);
  }

  if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    if (auto classType = unqualified_cast<ClassType>(typeAlias->type())) {
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

    if (isDependent(unit_, symbol->type())) {
      auto location = ast->unqualifiedId->firstSourceLocation();
      auto baseClassSymbol = control()->newBaseClassSymbol(scope(), location);
      ast->symbol = baseClassSymbol;

      baseClassSymbol->setVirtual(ast->isVirtual);
      baseClassSymbol->setSymbol(symbol);
      baseClassSymbol->setName(symbol->name());

      baseClassSymbol->setAccessSpecifier(
          toAccessSpecifier(ast->accessSpecifier, defaultAccessSpecifier()));
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

  baseClassSymbol->setAccessSpecifier(
      toAccessSpecifier(ast->accessSpecifier, defaultAccessSpecifier()));
}

void Binder::bind(NonTypeTemplateParameterAST* ast, int index, int depth) {
  auto symbol = control()->newNonTypeParameterSymbol(
      scope(), ast->declaration->firstSourceLocation());
  ast->symbol = symbol;
  ast->index = index;
  ast->depth = depth;

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
  ast->index = index;
  ast->depth = depth;

  symbol->setName(ast->identifier);
  scope()->addSymbol(symbol);
}

void Binder::bind(ConstraintTypeParameterAST* ast, int index, int depth) {
  const auto isParameterPack = static_cast<bool>(ast->ellipsisLoc);
  auto symbol = control()->newConstraintTypeParameterSymbol(
      scope(), ast->identifierLoc, index, depth, isParameterPack);
  symbol->setName(ast->identifier);
  symbol->setTypeConstraint(ast->typeConstraint);
  ast->symbol = symbol;
  ast->index = index;
  ast->depth = depth;
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
  ast->index = index;
  ast->depth = depth;

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

void Binder::bind(DeductionGuideAST* ast,
                  TemplateDeclarationAST* templateHead) {
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
  symbol->setDeclaration(ast);
  if (templateHead) symbol->setTemplateDeclaration(templateHead);
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
    const bool dependent = std::ranges::any_of(
        ListView{ast->templateId->templateArgumentList},
        [&](TemplateArgumentAST* argument) {
          return isDependentTemplateArgument(unit_, argument);
        });
    auto templateArgs =
        Substitution(unit_, templateDecl, ast->templateId->templateArgumentList)
            .templateArguments();

    if (!templateArgs.empty()) {
      if (auto cached =
              primaryTemplate->findSpecialization(unit_, templateArgs)) {
        deducedClassSymbol = symbol_cast<ClassSymbol>(cached);
      } else {
        auto parentScope = primaryTemplate->parent();
        auto spec =
            control()->newClassSymbol(parentScope, primaryTemplate->location());
        spec->setName(primaryTemplate->name());
        spec->setType(control()->getClassType(spec));
        primaryTemplate->addSpecialization(unit_, std::move(templateArgs),
                                           spec);
        primaryTemplate->setPendingInstantiation(
            spec, ast->templateId->templateArgumentList,
            ast->templateId->identifierLoc, !dependent);
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
  auto enclosingClosureCaptureField = [](ScopeSymbol* enclosingScope,
                                         const Name* name) -> FieldSymbol* {
    auto lambda = symbol_cast<LambdaSymbol>(enclosingScope);
    if (!lambda || !lambda->closureType()) return nullptr;
    for (auto candidate : lambda->closureType()->find(name)) {
      if (auto field = symbol_cast<FieldSymbol>(candidate)) return field;
    }
    return nullptr;
  };

  for (auto current = scope; current; current = current->parent()) {
    if (auto field = enclosingClosureCaptureField(current, name)) return field;
    for (auto candidate : current->find(name)) return candidate;
  }
  return nullptr;
}

auto Binder::isCapturableLocalEntity(Symbol* symbol) -> bool {
  if (!symbol) return false;
  if (symbol_cast<ParameterSymbol>(symbol)) return true;
  if (symbol_cast<ParameterPackSymbol>(symbol)) return true;
  if (auto field = symbol_cast<FieldSymbol>(symbol)) {
    auto closure = symbol_cast<ClassSymbol>(field->parent());
    return closure && closure->isClosureType();
  }
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
    if (auto parameters = symbol_cast<FunctionParametersSymbol>(current);
        parameters && !symbol_cast<FunctionSymbol>(parameters->parent())) {
      if (auto cls = parameters->enclosingClass())
        return control()->getPointerType(
            control()->getQualType(cls->type(), parameters->cvQualifiers()));
    }
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

void Binder::applyFunctionDefinitionKind(FunctionSymbol* functionSymbol,
                                         FunctionBodyAST* functionBody) {
  if (!functionSymbol) return;

  if (ast_cast<DeleteFunctionBodyAST>(functionBody)) {
    functionSymbol->setDeleted(true);
    return;
  }

  if (!ast_cast<DefaultFunctionBodyAST>(functionBody)) return;

  functionSymbol->setDefaulted(true);

  const auto isFirstDeclaration = functionSymbol->canonical() == functionSymbol;

  if (isFirstDeclaration) functionSymbol->setConstexpr(true);
}

void Binder::applyDeclarationAttributes(
    Symbol* symbol, List<AttributeSpecifierAST*>* attributes) {
  if (!symbol || !attributes) return;
  applyAttributeMap(symbol, collectAttributes(unit_, attributes));
}

void Binder::inheritDeclarationAttributes(Symbol* symbol, Symbol* pattern) {
  if (!symbol || !pattern) return;
  auto attributes = pattern->attributes();
  if (!attributes) return;
  applyAttributeMap(symbol, *attributes);
}

void Binder::applyAttributeMap(Symbol* symbol, AttributeMap collected) {
  if (!symbol || collected.empty()) return;

  auto canonical = symbol->canonical();
  if (!canonical) canonical = symbol;

  auto merged = control()->getAttributes(
      mergeAttributes(collected, canonical->attributes()));

  symbol->setAttributes(merged);
  canonical->setAttributes(merged);

  if (auto tags = findAttribute(merged, "abi_tag");
      tags && !tags->arguments.empty() && canonical->abiTags().empty()) {
    symbol->setAbiTags(control()->getAbiTags(tags->arguments));
    canonical->setAbiTags(symbol->abiTagList());
  }

  const auto isNodiscard = findAttribute(merged, "nodiscard") ||
                           findAttribute(merged, "warn_unused_result");
  symbol->setNodiscard(isNodiscard);
  canonical->setNodiscard(isNodiscard);

  if (findAttribute(merged, "used")) {
    symbol->setUsed(true);
    canonical->setUsed(true);
  }

  if (findAttribute(merged, "exclude_from_explicit_instantiation")) {
    symbol->setExcludedFromExplicitInstantiation(true);
    canonical->setExcludedFromExplicitInstantiation(true);
  }

  if (findAttribute(merged, "trivial_abi")) {
    symbol->setTrivialAbi(true);
    canonical->setTrivialAbi(true);
  }

  auto function = symbol_cast<FunctionSymbol>(symbol);
  if (!function) return;

  if (findAttribute(merged, "noreturn")) {
    function->setNoReturn(true);
    function->canonical()->setNoReturn(true);
  }

  if (findAttribute(merged, "nothrow")) {
    function->setExceptionSpecifier(true);
    setFunctionNoexcept(control(), function, true);

    auto canonicalFunction = function->canonical();
    if (canonicalFunction != function) {
      canonicalFunction->setExceptionSpecifier(true);
      setFunctionNoexcept(control(), canonicalFunction, true);
    }
  }

  applyWasmFunctionAttributes(function, merged);
}

void Binder::applyWasmFunctionAttributes(FunctionSymbol* function,
                                         const AttributeMap* attributes) {
  if (!control()->memoryLayout()->isWebAssembly()) return;

  auto canonical = function->canonical();

  auto apply = [&](std::string_view name,
                   void (FunctionSymbol::*setter)(const Identifier*)) {
    auto argument = attributeArgument(attributes, name);
    if (!argument) return;
    (function->*setter)(argument);
    (canonical->*setter)(argument);
  };

  apply("import_module", &FunctionSymbol::setImportModule);
  apply("import_name", &FunctionSymbol::setImportName);
  apply("export_name", &FunctionSymbol::setExportName);

  if (!function->exportName()) return;

  function->setUsed(true);
  canonical->setUsed(true);
}

void Binder::applyDeclarationAttributes(SimpleDeclarationAST* ast) {
  if (!ast || !ast->attributeList) return;

  for (auto initDeclarator : ListView{ast->initDeclaratorList}) {
    applyDeclarationAttributes(initDeclarator->symbol, ast->attributeList);
  }
}

auto Binder::usesImplicitThis(StatementAST* stmt) -> bool {
  if (!stmt) return false;
  ThisUseFinder finder;
  finder.accept(stmt);
  return finder.found;
}

void Binder::initializeCapturedField(FieldSymbol* field, ScopeSymbol* scope,
                                     ExpressionAST*& initializer,
                                     InitializationKind kind) {
  TypeChecker check{unit_};
  check.setScope(scope);
  check.setReportErrors(reportErrors());
  check.check_member_initialization(field, initializer, kind,
                                    ArrayCopyPolicy::kElementwiseCopyAllowed);
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

  ExpressionAST* thisExpr =
      ThisExpressionAST::create(ar, loc, ValueCategory::kPrValue, thisType);
  initializeCapturedField(field, scope(), thisExpr,
                          InitializationKind::kDirectInitialization);

  return ThisLambdaCaptureAST::create(ar, loc, thisExpr, field);
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

    auto idExpr = IdExpressionAST::create(ar);
    idExpr->unqualifiedId = NameIdAST::create(ar, identifier);
    idExpr->symbol = outerSymbol;
    idExpr->type = elementType;
    idExpr->valueCategory = ValueCategory::kLValue;

    auto field = control()->newFieldSymbol(classSymbol, loc);
    field->setName(identifier);
    field->setType(fieldType);
    if (auto alignment = control()->memoryLayout()->alignmentOf(fieldType)) {
      field->setAlignment(alignment.value());
    }
    classSymbol->addSymbol(field);
    capturedTypes.push_back(fieldType);
    captured.emplace(outerSymbol, field);

    ExpressionAST* initializer = idExpr;
    initializeCapturedField(field, scope(), initializer,
                            InitializationKind::kDirectInitialization);

    LambdaCaptureAST* capture = nullptr;
    if (byCopy) {
      auto simple = SimpleLambdaCaptureAST::create(ar);
      simple->identifierLoc = loc;
      simple->identifier = identifier;
      simple->initializer = initializer;
      simple->symbol = field;
      capture = simple;
    } else {
      auto ref = RefLambdaCaptureAST::create(ar);
      ref->ampLoc = loc;
      ref->identifierLoc = loc;
      ref->identifier = identifier;
      ref->initializer = initializer;
      ref->symbol = field;
      capture = ref;
    }

    *tail = make_list_node<LambdaCaptureAST>(ar, capture);
    tail = &(*tail)->next;

    use->symbol = field;
  }

  if (capturedTypes.empty()) return;

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

auto Binder::initCapture(LambdaCaptureAST* captureNode)
    -> std::optional<InitCapture> {
  if (auto initCap = ast_cast<InitLambdaCaptureAST>(captureNode)) {
    InitCapture capture;
    capture.name = initCap->identifier;
    capture.isPack = static_cast<bool>(initCap->ellipsisLoc);
    if (initCap->initializer && initCap->initializer->type)
      capture.type = traits.decay(initCap->initializer->type);
    return capture;
  }

  if (auto refInitCap = ast_cast<RefInitLambdaCaptureAST>(captureNode)) {
    InitCapture capture;
    capture.name = refInitCap->identifier;
    capture.isPack = static_cast<bool>(refInitCap->ellipsisLoc);
    if (refInitCap->initializer && refInitCap->initializer->type)
      capture.type = control()->getLvalueReferenceType(
          traits.remove_reference(refInitCap->initializer->type));
    return capture;
  }

  return std::nullopt;
}

void Binder::declareInitCapturesInLambdaScope(LambdaExpressionAST* ast) {
  for (auto captureNode : ListView{ast->captureList}) {
    auto capture = initCapture(captureNode);
    if (!capture || !capture->name) continue;

    auto variable = control()->newVariableSymbol(
        ast->symbol, captureNode->firstSourceLocation());
    variable->setName(capture->name);
    variable->setType(capture->type ? capture->type : control()->getAutoType());
    ast->symbol->addSymbol(variable);
  }
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

  const bool isNoexcept =
      exceptionSpecifierIsNoexcept(unit_, ast->exceptionSpecifier);

  if (ast->trailingReturnType && ast->trailingReturnType->typeId) {
    returnType = ast->trailingReturnType->typeId->type;
  }

  auto funcType = control()->getFunctionType(
      returnType, std::move(parameterTypes), isVariadic, {}, {}, isNoexcept);
  ast->symbol->setType(funcType);

  const bool inDependentContext =
      isEnclosedInDependentTemplate(unit_, ast->symbol->parent(),
                                    /*stopAtConcreteSpecialization=*/true) ||
      ast->symbol->isInTemplate();

  if (isCxx()) declareInitCapturesInLambdaScope(ast);

  if (isCxx() && !inDependentContext) {
    auto closureName = control()->newClosureName();

    auto classSymbol = control()->newClassSymbol(parentScope, ast->lbracketLoc);
    classSymbol->setName(closureName);
    parentScope->addSymbol(classSymbol);
    classSymbol->setClosureDiscriminator(
        lambdaDiscriminators_[classSymbol->enclosingFunction()]++);

    auto operatorCallName = control()->getOperatorId(TokenKind::T_LPAREN);
    auto operatorFunc = declareClosureMemberFunction(
        classSymbol, operatorCallName, funcType, ast->lbracketLoc);
    operatorFunc->setTrailingRequiresClause(ast->requiresClause);

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
    ast->symbol->setClosureType(classSymbol);
    classSymbol->setHasLambdaCapture(ast->captureDefault !=
                                         TokenKind::T_EOF_SYMBOL ||
                                     ast->captureList != nullptr);

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

        auto idExpr = IdExpressionAST::create(ar);
        idExpr->unqualifiedId = NameIdAST::create(ar, simple->identifier);
        idExpr->symbol = outerSymbol;
        idExpr->type = fieldType;
        idExpr->valueCategory = ValueCategory::kLValue;

        simple->initializer = idExpr;
        simple->symbol = addField(simple->identifier, fieldType);
        initializeCapturedField(simple->symbol, parentScope,
                                simple->initializer,
                                InitializationKind::kDirectInitialization);
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
        ref->symbol = addField(ref->identifier, fieldType);
        initializeCapturedField(ref->symbol, parentScope, ref->initializer,
                                InitializationKind::kDirectInitialization);
      } else if (auto th = ast_cast<ThisLambdaCaptureAST>(captureNode)) {
        auto thisType = enclosingThisType(parentScope);
        if (!thisType) {
          error(captureLoc, "'this' cannot be captured in this context");
          continue;
        }

        th->initializer = ThisExpressionAST::create(
            ar, th->thisLoc, ValueCategory::kPrValue, thisType);

        th->symbol = addField(control()->getIdentifier("__this"), thisType);
        classSymbol->setCapturedThisField(th->symbol);
        initializeCapturedField(th->symbol, parentScope, th->initializer,
                                InitializationKind::kDirectInitialization);
      } else if (auto deref =
                     ast_cast<DerefThisLambdaCaptureAST>(captureNode)) {
        error(captureLoc, "capture of '*this' is not yet supported");
      } else if (auto capture = initCapture(captureNode)) {
        if (!capture->type) continue;
        auto field = addField(capture->name, capture->type);
        *capture_field_slot(captureNode) = field;
        if (auto initializer = capture_initializer_slot(captureNode)) {
          initializeCapturedField(
              field, parentScope, *initializer,
              Initializer{*initializer}.initializationKind());
        }
      }
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

auto Binder::declareClosureMemberFunction(ClassSymbol* classSymbol,
                                          const Name* name, const Type* type,
                                          SourceLocation loc)
    -> FunctionSymbol* {
  auto function = control()->newFunctionSymbol(classSymbol, loc);
  function->setName(name);
  function->setType(type);
  function->setDefined(true);
  function->setConstexpr(true);
  function->setInline(true);
  function->setLanguageLinkage(LanguageKind::kCXX);
  classSymbol->addSymbol(function);
  return function;
}

auto Binder::declareClosureInvoker(ClassSymbol* classSymbol,
                                   FunctionSymbol* operatorFunc,
                                   const FunctionType* operatorType,
                                   SourceLocation loc) -> FunctionSymbol* {
  auto pool = unit_->arena();

  auto invoker = declareClosureMemberFunction(
      classSymbol, control()->getIdentifier("__invoke"), operatorType, loc);
  invoker->setStatic(true);

  auto parametersSymbol = control()->newFunctionParametersSymbol(invoker, loc);
  invoker->addSymbol(parametersSymbol);

  List<ExpressionAST*>* arguments = nullptr;
  auto argumentTail = &arguments;
  int index = 0;
  for (auto parameterType : operatorType->parameterTypes()) {
    auto parameter = control()->newParameterSymbol(parametersSymbol, loc);
    parameter->setName(control()->getIdentifier(std::format("__p{}", index++)));
    parameter->setType(parameterType);
    parametersSymbol->addSymbol(parameter);

    auto reference = IdExpressionAST::create(pool);
    reference->unqualifiedId =
        NameIdAST::create(pool, name_cast<Identifier>(parameter->name()));
    reference->symbol = parameter;
    reference->type = parameterType;
    reference->valueCategory = ValueCategory::kLValue;

    ExpressionAST* argument = reference;
    (void)TypeChecker{unit_}.implicit_conversion(argument, parameterType);

    *argumentTail = make_list_node<ExpressionAST>(pool, argument);
    argumentTail = &(*argumentTail)->next;
  }

  auto closureObject = TypeConstructionAST::create(pool);
  closureObject->type = classSymbol->type();
  closureObject->valueCategory = ValueCategory::kPrValue;
  for (auto constructor : classSymbol->declaredConstructors()) {
    auto constructorType = type_cast<FunctionType>(constructor->type());
    if (constructorType && constructorType->parameterTypes().empty()) {
      closureObject->constructorSymbol = constructor;
      break;
    }
  }

  auto callee = MemberExpressionAST::create(pool);
  callee->baseExpression = closureObject;
  callee->accessOp = TokenKind::T_DOT;
  callee->unqualifiedId =
      OperatorFunctionIdAST::create(pool, TokenKind::T_LPAREN);
  callee->symbol = operatorFunc;
  callee->type = operatorType;
  callee->valueCategory = ValueCategory::kLValue;

  auto call = CallExpressionAST::create(pool);
  call->baseExpression = callee;
  call->expressionList = arguments;
  call->type = operatorType->returnType();
  call->valueCategory = ValueCategory::kPrValue;

  StatementAST* bodyStatement = nullptr;
  if (traits.is_void(operatorType->returnType())) {
    auto expressionStatement = ExpressionStatementAST::create(pool);
    expressionStatement->expression = call;
    bodyStatement = expressionStatement;
  } else {
    auto returnStatement = ReturnStatementAST::create(pool);
    returnStatement->expression = call;
    bodyStatement = returnStatement;
  }

  auto block = CompoundStatementAST::create(pool);
  block->statementList = make_list_node<StatementAST>(pool, bodyStatement);

  attachSynthesizedBody(
      invoker, NameIdAST::create(pool, control()->getIdentifier("__invoke")),
      CompoundStatementFunctionBodyAST::create(
          pool, /*memInitializerList=*/nullptr, block));

  return invoker;
}

auto Binder::materializeClosureFunctionPointerConversion(
    ClassSymbol* closureClass, const FunctionType* targetFunctionType)
    -> FunctionSymbol* {
  if (!closureClass || !targetFunctionType) return nullptr;

  closureClass = closureClass->resolvedDefinition();
  if (!closureClass->isClosureType()) return nullptr;
  if (closureClass->hasLambdaCapture()) return nullptr;

  auto pointerType = control()->getPointerType(targetFunctionType);
  auto conversionName = control()->getConversionFunctionId(pointerType);

  for (auto existing : closureClass->find(conversionName)) {
    if (auto function = symbol_cast<FunctionSymbol>(existing)) return function;
  }

  auto operatorCallName = control()->getOperatorId(TokenKind::T_LPAREN);

  auto pattern = views::find_function(
      closureClass->find(operatorCallName), [](FunctionSymbol* function) {
        return function->templateDeclaration() && !function->isSpecialization();
      });

  if (!pattern) return nullptr;

  TemplateArgumentDeduction deduction{unit_};
  auto deducedArguments =
      deduction.deduceFromTargetType(pattern, targetFunctionType);
  if (!deducedArguments.has_value()) return nullptr;

  auto instance = ASTRewriter::instantiateOverloadCandidate(
      unit_, *deducedArguments, pattern, closureClass->location(),
      /*argsComplete=*/true);
  if (!instance) return nullptr;

  ASTRewriter::completeDeducedReturnType(unit_, instance);

  auto instanceType = type_cast<FunctionType>(instance->type());
  if (!instanceType) return nullptr;

  auto invoker = declareClosureInvoker(closureClass, instance, instanceType,
                                       closureClass->location());

  declareClosureFunctionPointerConversion(closureClass, invoker, instanceType,
                                          closureClass->location());

  for (auto existing : closureClass->find(control()->getConversionFunctionId(
           control()->getPointerType(instanceType)))) {
    if (auto function = symbol_cast<FunctionSymbol>(existing)) return function;
  }

  return nullptr;
}

void Binder::declareClosureFunctionPointerConversion(
    ClassSymbol* classSymbol, FunctionSymbol* invoker,
    const FunctionType* operatorType, SourceLocation loc) {
  auto pool = unit_->arena();

  auto pointerType = control()->getPointerType(operatorType);

  auto convFunc = declareClosureMemberFunction(
      classSymbol, control()->getConversionFunctionId(pointerType),
      control()->getFunctionType(pointerType, {}, false, CvQualifiers::kConst),
      loc);

  convFunc->addSymbol(control()->newFunctionParametersSymbol(convFunc, loc));

  auto reference = IdExpressionAST::create(pool);
  reference->unqualifiedId =
      NameIdAST::create(pool, name_cast<Identifier>(invoker->name()));
  reference->symbol = invoker;
  reference->type = operatorType;
  reference->valueCategory = ValueCategory::kLValue;

  auto addressOf = UnaryExpressionAST::create(pool);
  addressOf->op = TokenKind::T_AMP;
  addressOf->expression = reference;
  addressOf->type = pointerType;
  addressOf->valueCategory = ValueCategory::kPrValue;

  auto returnStatement = ReturnStatementAST::create(pool);
  returnStatement->expression = addressOf;

  auto block = CompoundStatementAST::create(pool);
  block->statementList = make_list_node<StatementAST>(pool, returnStatement);

  auto conversionId = ConversionFunctionIdAST::create(pool);

  attachSynthesizedBody(convFunc, conversionId,
                        CompoundStatementFunctionBodyAST::create(
                            pool, /*memInitializerList=*/nullptr, block));
}

void Binder::attachSynthesizedBody(FunctionSymbol* function,
                                   UnqualifiedIdAST* id,
                                   FunctionBodyAST* body) {
  auto pool = unit_->arena();

  auto idDeclarator = IdDeclaratorAST::create(pool);
  idDeclarator->unqualifiedId = id;

  auto functionChunk = FunctionDeclaratorChunkAST::create(pool);

  auto declarator = DeclaratorAST::create(
      pool, /*ptrOpList=*/nullptr, idDeclarator,
      make_list_node<DeclaratorChunkAST>(pool, functionChunk));

  auto definition = FunctionDefinitionAST::create(pool);
  definition->declarator = declarator;
  definition->functionBody = body;
  definition->symbol = function;
  function->setDeclaration(definition);
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

  completeClosureType(classSymbol);

  ast->constructorSymbol = classSymbol->defaultConstructor();

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

  if (!inTemplate() && !ast->symbol->isTemplate())
    checkLambdaBodyWarnings(unit_, ast);

  if (auto opFuncType = type_cast<FunctionType>(operatorFunc->type());
      opFuncType && !ast->symbol->isTemplate() &&
      ast->captureDefault == TokenKind::T_EOF_SYMBOL && !ast->captureList) {
    auto invoker = declareClosureInvoker(classSymbol, operatorFunc, opFuncType,
                                         ast->lbracketLoc);
    declareClosureFunctionPointerConversion(classSymbol, invoker, opFuncType,
                                            ast->lbracketLoc);
  }

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
  for (auto ctor : classSymbol->declaredConstructors()) {
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

void Binder::bind(NamespaceAliasDefinitionAST* ast,
                  NamespaceSymbol* resolvedNamespace) {
  auto id = ast->unqualifiedId->identifier;

  NamespaceSymbol* namespaceSymbol = resolvedNamespace;
  if (ast->nestedNameSpecifier && ast->nestedNameSpecifier->symbol)
    namespaceSymbol =
        qualifiedLookupNamespace(ast->nestedNameSpecifier->symbol, id);

  if (!namespaceSymbol) {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("'{}' is not a namespace name", id->name()));
    return;
  }

  auto scope = declaringScope();

  for (auto candidate : scope->find(ast->identifier)) {
    auto previous = symbol_cast<NamespaceAliasSymbol>(candidate);
    auto candidateNamespace = resolve_namespace_alias(candidate);
    if (candidateNamespace == namespaceSymbol) {
      if (previous) {
        ast->symbol = previous;
        return;
      }
      continue;
    }
    error(ast->identifierLoc,
          std::format("redefinition of namespace alias '{}'",
                      ast->identifier->name()));
    return;
  }

  auto symbol = control()->newNamespaceAliasSymbol(scope, ast->identifierLoc);
  symbol->setName(ast->identifier);
  symbol->setNamespaceSymbol(namespaceSymbol);
  ast->symbol = symbol;
  scope->addSymbol(symbol);
}

void Binder::bind(UsingEnumDeclarationAST* ast) {
  if (!ast || !ast->enumTypeSpecifier) return;

  auto spec = ast->enumTypeSpecifier;

  if (spec->nestedNameSpecifier && !spec->nestedNameSpecifier->symbol) {
    if (reportUnresolvedNestedNameSpecifier(spec->nestedNameSpecifier)) return;
  }

  const auto checkTemplates = unit_->config().checkTypes;
  auto symbol = resolve(spec->nestedNameSpecifier, spec->unqualifiedId,
                        checkTemplates, spec->symbol);

  symbol = resolve_using_declaration(symbol);

  ScopeSymbol* enumScope = nullptr;
  if (auto enumSymbol = symbol_cast<EnumSymbol>(symbol)) {
    enumScope = enumSymbol;
  } else if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(symbol)) {
    enumScope = scopedEnumSymbol;
  } else if (auto typeAlias = symbol_cast<TypeAliasSymbol>(symbol)) {
    auto unqualType = traits.remove_cv(typeAlias->type());
    if (auto enumType = type_cast<EnumType>(unqualType)) {
      enumScope = enumType->symbol();
    } else if (auto scopedEnumType = type_cast<ScopedEnumType>(unqualType)) {
      enumScope = scopedEnumType->symbol();
    }
  }

  if (!enumScope) {
    if (!inTemplate()) {
      auto missingName = get_name(control(), spec->unqualifiedId);
      error(spec->unqualifiedId->firstSourceLocation(),
            std::format("'{}' does not name an enumeration",
                        to_string(missingName)));
    }
    return;
  }

  spec->symbol = enumScope;

  for (auto member : enumScope->members()) {
    if (auto enumerator = symbol_cast<EnumeratorSymbol>(member)) {
      injectUsing(scope(), enumerator->name(), enumerator, ast->usingLoc);
    }
  }
}

void Binder::bind(TypeIdAST* ast, const Decl& decl) {
  ast->type = getDeclaratorType(unit_, ast->declarator, decl.specs.type());
}

auto Binder::declareTypedef(DeclaratorAST* declarator, const Decl& decl)
    -> TypeAliasSymbol* {
  auto name = decl.getName();
  auto type = getDeclaratorType(unit_, declarator, decl.specs.type());
  auto targetScope = declaringScope();
  auto symbol = control()->newTypeAliasSymbol(targetScope, decl.location());
  applyAccessSpecifier(symbol);
  symbol->setName(name);
  symbol->setType(type);

  addTypeAliasToScope(symbol);

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

auto unqualifiedIdsStructurallyEquivalentForRedeclaration(TranslationUnit* unit,
                                                          UnqualifiedIdAST* a,
                                                          UnqualifiedIdAST* b)
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;

  if (auto aName = ast_cast<NameIdAST>(a)) {
    auto bName = ast_cast<NameIdAST>(b);
    return bName && aName->identifier == bName->identifier;
  }

  auto aTemplateId = ast_cast<SimpleTemplateIdAST>(a);
  auto bTemplateId = ast_cast<SimpleTemplateIdAST>(b);
  if (!aTemplateId || !bTemplateId) return false;
  if (aTemplateId->identifier != bTemplateId->identifier) return false;
  return TemplateEquivalence{unit}.same(aTemplateId->templateArgumentList,
                                        bTemplateId->templateArgumentList);
}

auto nestedNameSpecifiersStructurallyEquivalent(TranslationUnit* unit,
                                                NestedNameSpecifierAST* a,
                                                NestedNameSpecifierAST* b)
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (auto ta = ast_cast<TemplateNestedNameSpecifierAST>(a)) {
    auto tb = ast_cast<TemplateNestedNameSpecifierAST>(b);
    if (!tb || !ta->templateId || !tb->templateId) return false;
    if (ta->templateId->identifier != tb->templateId->identifier) return false;
    if (!TemplateEquivalence{unit}.same(ta->templateId->templateArgumentList,
                                        tb->templateId->templateArgumentList)) {
      return false;
    }
    return nestedNameSpecifiersStructurallyEquivalent(
        unit, ta->nestedNameSpecifier, tb->nestedNameSpecifier);
  }

  if (a->symbol && b->symbol) {
    if (a->symbol == b->symbol) return true;
    auto aInfo = template_parameter_info(a->symbol);
    auto bInfo = template_parameter_info(b->symbol);
    return aInfo && bInfo && aInfo->index == bInfo->index &&
           aInfo->depth == bInfo->depth && aInfo->isPack == bInfo->isPack;
  }

  if (ast_cast<GlobalNestedNameSpecifierAST>(a)) {
    return ast_cast<GlobalNestedNameSpecifierAST>(b) != nullptr;
  }

  if (auto sa = ast_cast<SimpleNestedNameSpecifierAST>(a)) {
    auto sb = ast_cast<SimpleNestedNameSpecifierAST>(b);
    if (!sb || sa->identifier != sb->identifier) return false;
    return nestedNameSpecifiersStructurallyEquivalent(
        unit, sa->nestedNameSpecifier, sb->nestedNameSpecifier);
  }

  if (auto da = ast_cast<DecltypeNestedNameSpecifierAST>(a)) {
    auto db = ast_cast<DecltypeNestedNameSpecifierAST>(b);
    if (!db || !da->decltypeSpecifier || !db->decltypeSpecifier) return false;
    auto aType = da->decltypeSpecifier->type;
    auto bType = db->decltypeSpecifier->type;
    if (!aType || !bType) return false;
    return unit->typeTraits().is_same(aType, bType);
  }

  return false;
}

auto unresolvedNameTypesStructurallyEquivalent(TranslationUnit* unit,
                                               const UnresolvedNameType* a,
                                               const UnresolvedNameType* b)
    -> bool {
  if (a == b) return true;
  if (!a || !b) return false;
  if (!unqualifiedIdsStructurallyEquivalentForRedeclaration(
          unit, a->unqualifiedId(), b->unqualifiedId())) {
    return false;
  }
  return nestedNameSpecifiersStructurallyEquivalent(
      unit, a->nestedNameSpecifier(), b->nestedNameSpecifier());
}

[[nodiscard]] auto indirectElementType(const Type* type) -> const Type* {
  if (auto pointer = type_cast<PointerType>(type))
    return pointer->elementType();
  if (auto reference = type_cast<LvalueReferenceType>(type))
    return reference->elementType();
  if (auto reference = type_cast<RvalueReferenceType>(type))
    return reference->elementType();
  return nullptr;
}

}  // namespace

namespace {

auto redeclarationTypesEquivalent(TranslationUnit* unit,
                                  const Type* existingType,
                                  const Type* incomingType,
                                  bool ignoresArrayBound) -> bool {
  if (!existingType || !incomingType) return false;

  if (unit->typeTraits().is_same(existingType, incomingType)) return true;

  auto existingQual = type_cast<QualType>(existingType);
  auto incomingQual = type_cast<QualType>(incomingType);
  if (existingQual || incomingQual) {
    if (!existingQual || !incomingQual) return false;
    if (existingQual->cvQualifiers() != incomingQual->cvQualifiers())
      return false;
    return redeclarationTypesEquivalent(unit, existingQual->elementType(),
                                        incomingQual->elementType(),
                                        ignoresArrayBound);
  }

  auto existingReferencedType = indirectElementType(existingType);
  auto incomingReferencedType = indirectElementType(incomingType);
  if (existingReferencedType || incomingReferencedType) {
    if (!existingReferencedType || !incomingReferencedType) return false;
    if (existingType->kind() != incomingType->kind()) return false;
    return redeclarationTypesEquivalent(unit, existingReferencedType,
                                        incomingReferencedType,
                                        /*ignoresArrayBound=*/false);
  }

  auto existingUnresolved = type_cast<UnresolvedNameType>(existingType);
  auto incomingUnresolved = type_cast<UnresolvedNameType>(incomingType);
  if (existingUnresolved || incomingUnresolved) {
    if (!existingUnresolved || !incomingUnresolved) return false;
    return unresolvedNameTypesStructurallyEquivalent(unit, existingUnresolved,
                                                     incomingUnresolved);
  }

  if (!unit->typeTraits().is_array(existingType)) return false;
  if (!unit->typeTraits().is_array(incomingType)) return false;

  auto existingElementType = unit->typeTraits().get_element_type(existingType);
  auto incomingElementType = unit->typeTraits().get_element_type(incomingType);
  if (!redeclarationTypesEquivalent(unit, existingElementType,
                                    incomingElementType, ignoresArrayBound)) {
    return false;
  }

  if (ignoresArrayBound) {
    if (isEffectivelyUnboundedArray(unit, existingType)) return true;
    if (isEffectivelyUnboundedArray(unit, incomingType)) return true;
  } else if (isEffectivelyUnboundedArray(unit, existingType) !=
             isEffectivelyUnboundedArray(unit, incomingType)) {
    return false;
  }

  auto existingBound = arrayBoundToString(existingType);
  auto incomingBound = arrayBoundToString(incomingType);
  if (!existingBound || !incomingBound) return true;
  return *existingBound == *incomingBound;
}

}  // namespace

auto areRedeclarationTypesCompatible(TranslationUnit* unit,
                                     const Type* existingType,
                                     const Type* incomingType) -> bool {
  if (!unit || !existingType || !incomingType) return false;

  return redeclarationTypesEquivalent(unit, unqualified_type(existingType),
                                      unqualified_type(incomingType));
}

auto areFunctionSignaturesEquivalentForRedeclaration(
    TranslationUnit* unit, const Type* lhs, const Type* rhs,
    TemplateDeclarationAST* lhsHead, TemplateDeclarationAST* rhsHead,
    bool isOutOfLineDeclaration) -> bool {
  if (!unit || !lhs || !rhs) return false;
  if (unit->typeTraits().is_same(lhs, rhs)) return true;

  auto lhsFn = type_cast<FunctionType>(lhs);
  auto rhsFn = type_cast<FunctionType>(rhs);
  if (!lhsFn || !rhsFn) return false;

  if (lhsHead && rhsHead && lhsHead->depth != rhsHead->depth) {
    int ownParameterCount = 0;
    for ([[maybe_unused]] auto parameter :
         ListView{rhsHead->templateParameterList})
      ++ownParameterCount;

    if (TemplateEquivalence{unit}.corresponds(
            lhs, rhs, {lhsHead->depth, rhsHead->depth, ownParameterCount}))
      return true;
  }

  const bool dependentReturnType =
      isOutOfLineDeclaration && (isDependent(unit, lhsFn->returnType()) ||
                                 isDependent(unit, rhsFn->returnType()));
  if (!dependentReturnType &&
      !areRedeclarationTypesCompatible(unit, lhsFn->returnType(),
                                       rhsFn->returnType()))
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

namespace {
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
    auto hasFinalOverriderIn = [&](ScopeSymbol* cls, FunctionSymbol* fn) {
      auto match =
          views::find_function(cls->members(), [&](FunctionSymbol* member) {
            return traits.is_corresponding_overrider(member, fn);
          });
      return match && !match->isPure();
    };

    auto overridesInClass = [&](FunctionSymbol* fn) -> bool {
      return hasFinalOverriderIn(classSymbol, fn);
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
          return hasFinalOverriderIn(baseClass, fn) || overridesInClass(fn);
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

void Binder::checkRedeclaredAlignment(ClassSymbol* classSymbol, int requested,
                                      SourceLocation loc) {
  const auto declared = classSymbol->explicitAlignment();

  if (declared && declared != requested) {
    error(loc, std::format("redeclaration has a different alignment "
                           "requirement ({} vs {})",
                           requested, declared));
    return;
  }

  if (!declared && classSymbol->isComplete()) {
    error(loc,
          "'alignas' must be specified on the definition if it is specified "
          "on any declaration");
    return;
  }

  classSymbol->setExplicitAlignment(requested);
}

auto Binder::hasDependentAlignment(
    List<AttributeSpecifierAST*>* attributeList) const -> bool {
  for (auto specifier : ListView{attributeList}) {
    if (auto alignas_ = ast_cast<AlignasAttributeAST>(specifier)) {
      if (isDependent(unit_, alignas_->expression)) return true;
      continue;
    }

    if (auto alignas_ = ast_cast<AlignasTypeAttributeAST>(specifier)) {
      if (!alignas_->typeId) continue;
      if (isDependent(unit_, alignas_->typeId->type)) return true;
    }
  }

  return false;
}

auto Binder::explicitAlignment(List<AttributeSpecifierAST*>* attributeList,
                               SourceLocation loc) -> std::optional<int> {
  std::optional<int> strictest;

  auto require = [&](std::optional<std::intmax_t> value, SourceLocation at) {
    if (!value) {
      error(at, "'aligned' attribute requires integer constant");
      return;
    }
    if (*value == 0) return;
    if (*value < 0 || (*value & (*value - 1)) != 0) {
      error(at, "requested alignment is not a power of 2");
      return;
    }
    auto alignment = static_cast<int>(*value);
    if (!strictest || *strictest < alignment) strictest = alignment;
  };

  for (auto specifier : ListView{attributeList}) {
    if (auto alignas_ = ast_cast<AlignasAttributeAST>(specifier)) {
      auto at = alignas_->alignasLoc ? alignas_->alignasLoc : loc;
      if (isDependent(unit_, alignas_->expression)) continue;
      ASTInterpreter interp{unit_};
      auto value = interp.evaluate(alignas_->expression);
      if (!value) {
        require(std::nullopt, at);
        continue;
      }
      require(interp.toInt(*value), at);
      continue;
    }

    if (auto alignas_ = ast_cast<AlignasTypeAttributeAST>(specifier)) {
      auto at = alignas_->alignasLoc ? alignas_->alignasLoc : loc;
      if (!alignas_->typeId) continue;
      auto type = alignas_->typeId->type;
      if (!type || isDependent(unit_, type)) continue;
      auto alignment = control()->memoryLayout()->alignmentOf(type);
      if (!alignment) {
        require(std::nullopt, at);
        continue;
      }
      require(static_cast<std::intmax_t>(*alignment), at);
    }
  }

  return strictest;
}

auto Binder::checkExplicitAlignment(int requested, const Type* type,
                                    SourceLocation loc) -> bool {
  auto natural = control()->memoryLayout()->alignmentOf(type);
  if (!natural) return true;
  if (static_cast<int>(*natural) <= requested) return true;

  error(loc, std::format("requested alignment is less than minimum alignment "
                         "of {} for type '{}'",
                         *natural, to_string(type)));
  return false;
}

void Binder::applyExplicitAlignment(FieldSymbol* field, const Decl& decl) {
  auto requested = explicitAlignment(decl.specs.attributeList, decl.location());
  if (!requested) return;

  if (field->isBitField()) {
    error(decl.location(),
          "'alignas' attribute cannot be applied to a bit-field");
    return;
  }

  if (!checkExplicitAlignment(*requested, field->type(), decl.location()))
    return;

  field->setAlignment(*requested);
}

void Binder::applyExplicitAlignment(VariableSymbol* variable,
                                    const Decl& decl) {
  auto requested = explicitAlignment(decl.specs.attributeList, decl.location());
  if (!requested) return;

  if (!checkExplicitAlignment(*requested, variable->type(), decl.location()))
    return;

  variable->setExplicitAlignment(*requested);
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
        if (auto bitWidth = std::get_if<ConstInt>(&*value)) {
          const auto width = bitWidth->toIntMax();
          if (width < 0) {
            error(decl.location(), "bit-field width is negative");
          } else if (width == 0 && name) {
            error(decl.location(), "zero-width bit-field must be unnamed");
          } else if (!inTemplate()) {
            auto typeSize = control()->memoryLayout()->sizeOf(type);
            if (typeSize && width > static_cast<std::intmax_t>(*typeSize) * 8) {
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

  applyExplicitAlignment(fieldSymbol, decl);

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
                             bool addSymbolToParentScope,
                             const Type* declaratorType) -> VariableSymbol* {
  auto name = decl.getName();
  auto currentScope = declaringScope();
  auto qualifiedScope = decl.getScope();
  auto qualifiedClass = symbol_cast<ClassSymbol>(qualifiedScope);
  auto qualifiedNamespace = symbol_cast<NamespaceSymbol>(qualifiedScope);

  ClassSymbol* outOfClassMemberClass = nullptr;
  FieldSymbol* outOfClassMemberField = nullptr;
  if (qualifiedClass) {
    for (auto candidate : qualifiedClass->find(name)) {
      auto field = symbol_cast<FieldSymbol>(candidate);
      if (!field || !field->isStatic()) continue;
      outOfClassMemberClass = qualifiedClass;
      outOfClassMemberField = field;
      break;
    }
  }

  const bool isOutOfClassStaticMemberDef = outOfClassMemberField != nullptr;
  const bool isOutOfNamespaceMemberDef = qualifiedNamespace != nullptr;

  auto targetScope = isOutOfClassStaticMemberDef
                         ? static_cast<ScopeSymbol*>(outOfClassMemberClass)
                     : isOutOfNamespaceMemberDef
                         ? static_cast<ScopeSymbol*>(qualifiedNamespace)
                     : decl.specs.isExtern ? scopeForBlockDecl(currentScope)
                                           : currentScope;

  auto symbol = control()->newVariableSymbol(targetScope, decl.location());
  auto type = declaratorType;
  if (!type) type = getDeclaratorType(unit_, declarator, decl.specs.type());
  applySpecifiers(symbol, decl.specs);
  symbol->setName(name);
  symbol->setType(type);

  if (isOutOfClassStaticMemberDef) {
    outOfClassMemberField->setDefinition(symbol);
    symbol->setStatic(true);
    symbol->setInitializer(outOfClassMemberField->initializer());
  }

  if (auto classType = unqualified_cast<ClassType>(type)) {
    traits.requireCompleteClass(classType->symbol());
  }

  applyExplicitAlignment(symbol, decl);

  if (!addSymbolToParentScope || isOutOfClassStaticMemberDef) return symbol;

  if (auto block = symbol_cast<BlockSymbol>(targetScope);
      block && block->isOutermostBlockScope()) {
    if (auto parentScope = block->parent()) {
      for (auto candidate : parentScope->find(name)) {
        if (!symbol_cast<VariableSymbol>(candidate) &&
            !symbol_cast<ParameterSymbol>(candidate))
          continue;
        error(symbol->location(),
              std::format("redefinition of '{}'", to_string(name)));
        note(candidate->location(), "previous definition is here");
        break;
      }
    }
  }

  for (auto candidate : targetScope->find(name)) {
    if (auto existing = symbol_cast<VariableSymbol>(candidate)) {
      if (targetScope->isBlock()) {
        error(symbol->location(),
              std::format("redefinition of '{}'", to_string(name)));
        note(existing->location(), "previous definition is here");
        break;
      }

      if (!areRedeclarationTypesCompatible(unit_, existing->type(),
                                           symbol->type())) {
        error(symbol->location(),
              std::format("conflicting declaration of '{}'", to_string(name)));
        continue;
      }

      auto canon = existing->canonical();
      auto mergedType =
          preferredRedeclarationType(unit_, canon->type(), symbol->type());
      setSpeculativeValue(
          canon->type(), mergedType,
          [canon](const Type* value) { canon->setType(value); });
      symbol->setType(mergedType);
      addRedeclaration(canon, symbol);
      break;
    }
  }

  targetScope->addSymbol(symbol);

  if (targetScope != currentScope && !isOutOfNamespaceMemberDef) {
    if (symbol->canonical() == symbol) symbol->setHidden(true);
    injectUsing(currentScope, name, symbol->canonical(), decl.location());
  }
  return symbol;
}

void Binder::declareVariableTemplate(VariableSymbol* symbol,
                                     IdDeclaratorAST* declaratorId,
                                     TemplateDeclarationAST* templateHead) {
  setTemplateHead(symbol, templateHead);
  if (!templateHead) return;

  checkTemplateParameterDefaultOrder(symbol->canonical()->templateParameters());

  if (!declaratorId) return;

  auto templateId = ast_cast<SimpleTemplateIdAST>(declaratorId->unqualifiedId);
  if (!templateId) return;

  for (auto candidate :
       declaringScope()->find(templateId->identifier) | views::variables) {
    if (candidate == symbol) continue;
    if (!candidate->templateDeclaration()) continue;

    auto templateArguments =
        Substitution(unit_, candidate->templateDeclaration(),
                     templateId->templateArgumentList)
            .templateArguments();

    candidate->addSpecialization(unit_, std::move(templateArguments), symbol);
    break;
  }
}

auto Binder::declareMemberSymbol(DeclaratorAST* declarator, const Decl& decl,
                                 bool addSymbolToParentScope) -> Symbol* {
  if (decl.specs.isTypedef) return declareTypedef(declarator, decl);

  if (getFunctionPrototype(declarator))
    return declareFunction(declarator, decl, addSymbolToParentScope);

  return declareField(declarator, decl);
}

void Binder::applySpecifiers(FunctionSymbol* symbol, const DeclSpecs& specs) {
  applyAccessSpecifier(symbol);
  symbol->setStatic(specs.isStatic);
  symbol->setExtern(specs.isExtern);
  symbol->setFriend(specs.isFriend);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConsteval(specs.isConsteval);
  auto isInline = specs.isInline;
  if (specs.isConstexpr) isInline = true;
  if (specs.isConsteval) isInline = true;
  symbol->setInline(isInline);
  symbol->setVirtual(specs.isVirtual);
  symbol->setExplicit(specs.isExplicit);
}

void Binder::applySpecifiers(VariableSymbol* symbol, const DeclSpecs& specs) {
  applyAccessSpecifier(symbol);
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setExtern(specs.isExtern);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

void Binder::applySpecifiers(FieldSymbol* symbol, const DeclSpecs& specs) {
  applyAccessSpecifier(symbol);
  symbol->setStatic(specs.isStatic);
  symbol->setThreadLocal(specs.isThreadLocal);
  symbol->setConstexpr(specs.isConstexpr);
  symbol->setConstinit(specs.isConstinit);
  symbol->setInline(specs.isInline);
}

auto Binder::reportUnresolvedNestedNameSpecifier(NestedNameSpecifierAST* ast)
    -> bool {
  if (inTemplate() || isDependentTypeParameterSymbol(ast->symbol)) return false;

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

  if (auto namespaceSymbol = resolve_namespace_alias(symbol))
    return namespaceSymbol;

  if (auto enumSymbol = symbol_cast<EnumSymbol>(symbol)) return enumSymbol;

  if (auto scopedEnumSymbol = symbol_cast<ScopedEnumSymbol>(symbol))
    return scopedEnumSymbol;

  if (auto typeAliasSymbol = symbol_cast<TypeAliasSymbol>(symbol)) {
    auto aliasedType = unqualified_type(typeAliasSymbol->type());

    if (auto classType = type_cast<ClassType>(aliasedType)) {
      traits.requireCompleteClass(classType->symbol());
      return classType->symbol();
    }

    if (auto enumType = type_cast<EnumType>(aliasedType))
      return enumType->symbol();

    if (auto scopedEnumType = type_cast<ScopedEnumType>(aliasedType))
      return scopedEnumType->symbol();
  }

  return nullptr;
}

namespace {
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

    if (!joinsFunctionOverloadSet(candidate)) continue;

    auto function = symbol_cast<FunctionSymbol>(candidate);
    auto usingDeclaration = symbol_cast<UsingDeclarationSymbol>(candidate);

    auto overloadSet = control()->newOverloadSetSymbol(scope, location);
    overloadSet->setName(name);
    if (function) overloadSet->addFunction(function);
    if (usingDeclaration) overloadSet->addUsingDeclaration(usingDeclaration);
    if (speculationDepth_) {
      recordSpeculativeMutation([scope, candidate, overloadSet] {
        scope->replaceSymbol(overloadSet, candidate);
      });
    }
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

void Binder::declareBuiltinFunctionCallee(IdExpressionAST* ast) {
  if (ast->nestedNameSpecifier &&
      ast->nestedNameSpecifier->symbol != unit_->globalScope())
    return;

  auto name = get_name(control(), ast->unqualifiedId);
  auto id = name_cast<Identifier>(name);
  if (!id) return;

  ast->symbol = resolveBuiltinFunctionSymbol(unit_, id, id->builtinFunction());
}

void Binder::bind(IdExpressionAST* ast, bool mayUseArgumentDependentLookup) {
  if (!ast->unqualifiedId) {
    error(ast->firstSourceLocation(),
          "expected an unqualified identifier in id expression");
    return;
  }

  if (ast->nestedNameSpecifier) {
    if (!ast->nestedNameSpecifier->symbol) {
      (void)reportUnresolvedNestedNameSpecifier(ast->nestedNameSpecifier);
      return;
    }

    if (isDeferredDependentLookupContext(
            unit_, ast->nestedNameSpecifier->symbol, scope()))
      return;

    auto name = get_name(control(), ast->unqualifiedId);

    const Name* componentName = name;

    if (auto templateId = name_cast<TemplateId>(name)) {
      componentName = templateId->name();
    }

    bool ambiguous = false;
    ast->symbol = qualifiedLookupIncludingInlineNamespaces(
        control(), ast->nestedNameSpecifier->symbol, componentName, &ambiguous);
    if (ambiguous) {
      error(ast->unqualifiedId->firstSourceLocation(),
            std::format("reference to '{}' is ambiguous",
                        to_string(componentName)));
      return;
    }
  }

  if (!ast->symbol && mayUseArgumentDependentLookup) {
    declareBuiltinFunctionCallee(ast);
  }

  if (!ast->symbol && !ast->nestedNameSpecifier &&
      mayUseArgumentDependentLookup) {
    declareArgumentDependentCallee(ast);
  }

  resolveIdExpression(ast, mayUseArgumentDependentLookup);
}

void Binder::qualifiedLookupIdExpression(IdExpressionAST* ast, bool isCallee) {
  if (!ast->unqualifiedId) return;
  if (!ast->nestedNameSpecifier || !ast->nestedNameSpecifier->symbol) return;

  if (isDeferredDependentLookupContext(unit_, ast->nestedNameSpecifier->symbol,
                                       scope()))
    return;

  if (auto classSymbol =
          symbol_cast<ClassSymbol>(ast->nestedNameSpecifier->symbol)) {
    traits.requireCompleteClass(classSymbol);
  }

  auto name = get_name(control(), ast->unqualifiedId);
  const Name* componentName = name;
  if (auto templateId = name_cast<TemplateId>(name))
    componentName = templateId->name();

  bool ambiguous = false;
  ast->symbol = qualifiedLookupIncludingInlineNamespaces(
      control(), ast->nestedNameSpecifier->symbol, componentName, &ambiguous);
  if (ambiguous) {
    error(ast->unqualifiedId->firstSourceLocation(),
          std::format("reference to '{}' is ambiguous",
                      to_string(componentName)));
    return;
  }

  resolveIdExpression(ast, isCallee);

  if (auto function = designatedFunction(ast->symbol)) {
    ast->symbol = function;
    ast->type = function->type();
  }
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
        if (!templateDecl) return false;
        auto arity = TemplateArity::of(templateDecl);
        auto argc = TemplateArguments::count(templateId->templateArgumentList);
        if (argc < arity.minArgs) return true;
        if (arity.packCount > 0 && (isCallee || argc <= arity.minArgs))
          return true;
        return arity.packCount > 1;
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
              TemplateArity::matches(func->templateDeclaration(),
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
          if (TemplateArity::matches(func->templateDeclaration(),
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
          if (!TemplateArity::matches(func->templateDeclaration(),
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
              /*sfinaeContext=*/true, /*argsComplete=*/false,
              /*declarationOnly=*/true);
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
            /*sfinaeContext=*/isFuncTemplate, /*argsComplete=*/false,
            /*declarationOnly=*/isFuncTemplate);
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

  return names_template_head_parameters(templateNns->templateId, qualifier);
}

auto Binder::currentInstantiationOf(ScopeSymbol* scope) -> ClassSymbol* {
  if (auto classSymbol = symbol_cast<ClassSymbol>(scope)) return classSymbol;
  return scope->enclosingClass();
}

auto Binder::resolveMemberOfCurrentInstantiation(
    NestedNameSpecifierAST* nestedNameSpecifier,
    UnqualifiedIdAST* unqualifiedId, ClassSymbol* currentInstantiation)
    -> Symbol* {
  if (!denotesCurrentInstantiation(nestedNameSpecifier, currentInstantiation))
    return nullptr;
  auto nameId = ast_cast<NameIdAST>(unqualifiedId);
  if (!nameId) return nullptr;
  auto qualifier = symbol_cast<ClassSymbol>(nestedNameSpecifier->symbol);
  return qualifiedLookup(qualifier, nameId->identifier,
                         [](Symbol* s) { return is_type(s); });
}

struct Binder::ResolveCurrentInstantiationMembers {
  Binder& binder;
  ClassSymbol* currentInstantiation = nullptr;

  [[nodiscard]] auto specifierList(List<SpecifierAST*>* list) -> bool {
    auto changed = false;
    for (auto specifier : ListView{list}) {
      changed |= visit(*this, specifier);
    }
    return changed;
  }

  [[nodiscard]] auto typeId(TypeIdAST* ast) -> bool {
    if (!ast) return false;
    if (!specifierList(ast->typeSpecifierList)) return false;
    DeclSpecs specs{binder.unit_};
    for (auto specifier : ListView{ast->typeSpecifierList}) {
      specs.accept(specifier);
    }
    specs.finish();
    ast->type = getDeclaratorType(binder.unit_, ast->declarator, specs.type());
    return true;
  }

  [[nodiscard]] auto expression(ExpressionAST* ast) -> bool {
    auto recheck = [&](TypeIdAST* operand) {
      if (!typeId(operand)) return false;
      auto typeChecker = TypeChecker{binder.unit_};
      typeChecker.setScope(binder.scope());
      typeChecker.setReportErrors(false);
      typeChecker.check(&ast);
      return true;
    };

    if (auto sizeofType = ast_cast<SizeofTypeExpressionAST>(ast))
      return recheck(sizeofType->typeId);

    if (auto alignofType = ast_cast<AlignofTypeExpressionAST>(ast))
      return recheck(alignofType->typeId);

    return false;
  }

  [[nodiscard]] auto templateId(SimpleTemplateIdAST* ast) -> bool {
    if (!ast) return false;
    auto changed = false;
    for (auto argument : ListView{ast->templateArgumentList}) {
      changed |= visit(*this, argument);
    }
    return changed;
  }

  auto operator()(TypeTemplateArgumentAST* ast) -> bool {
    return typeId(ast->typeId);
  }

  auto operator()(ExpressionTemplateArgumentAST* ast) -> bool {
    return expression(ast->expression);
  }

  auto operator()(TypenameSpecifierAST* ast) -> bool {
    if (ast->symbol) return false;
    if (templateId(ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId))) {
      return true;
    }
    auto member = binder.resolveMemberOfCurrentInstantiation(
        ast->nestedNameSpecifier, ast->unqualifiedId, currentInstantiation);
    if (!member) return false;
    ast->symbol = member;
    return true;
  }

  auto operator()(NamedTypeSpecifierAST* ast) -> bool {
    if (!templateId(ast_cast<SimpleTemplateIdAST>(ast->unqualifiedId)))
      return false;
    auto symbol = binder.resolve(ast->nestedNameSpecifier, ast->unqualifiedId,
                                 /*checkTemplates=*/true);
    if (!symbol) return false;
    ast->symbol = symbol;
    return true;
  }

  auto operator()(SpecifierAST*) -> bool { return false; }
};

auto Binder::resolveMembersOfCurrentInstantiation(
    List<SpecifierAST*>* specifierList, ClassSymbol* currentInstantiation)
    -> bool {
  if (!currentInstantiation) return false;
  return ResolveCurrentInstantiationMembers{*this, currentInstantiation}
      .specifierList(specifierList);
}

auto Binder::resolveMemberOfCurrentInstantiation(
    const Type* type, ClassSymbol* currentInstantiation) -> const Type* {
  if (!type || !currentInstantiation) return type;

  auto resolve = [&](const Type* nested) {
    return resolveMemberOfCurrentInstantiation(nested, currentInstantiation);
  };

  if (auto unresolved = type_cast<UnresolvedNameType>(type)) {
    auto member = resolveMemberOfCurrentInstantiation(
        unresolved->nestedNameSpecifier(), unresolved->unqualifiedId(),
        currentInstantiation);
    if (!member || !member->type()) return type;
    return member->type();
  }

  if (auto qual = type_cast<QualType>(type)) {
    auto elementType = resolve(qual->elementType());
    if (elementType == qual->elementType()) return type;
    return control()->getQualType(elementType, qual->cvQualifiers());
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
        function->exceptionSpecification());
  }

  return type;
}

auto isExplicitSpecializationHead(TemplateDeclarationAST* templateHead)
    -> bool {
  return templateHead && !templateHead->templateParameterList;
}

auto Binder::getSpecializedFunctionTemplate(
    ScopeSymbol* scope, const Name* name, TemplateDeclarationAST* templateHead)
    -> FunctionSymbol* {
  if (!scope) return nullptr;
  if (!isExplicitSpecializationHead(templateHead) && !inExplicitInstantiation())
    return nullptr;

  auto templateName = name;
  if (auto templateId = name_cast<TemplateId>(name))
    templateName = templateId->name();

  for (auto candidate : scope->find(templateName)) {
    for (auto function : views::each_function(candidate)) {
      auto templateDeclaration = function->templateDeclaration();
      if (!templateDeclaration || !templateDeclaration->templateParameterList)
        continue;
      return function;
    }
  }

  return nullptr;
}

auto Binder::getFunction(ScopeSymbol* scope, const Name* name, const Type* type,
                         TemplateDeclarationAST* templateHead,
                         RequiresClauseAST* trailingRequiresClause)
    -> FunctionSymbol* {
  auto parentScope = scope;

  while (parentScope && parentScope->isTransparent()) {
    parentScope = parentScope->parent();
  }

  auto matches = [&](FunctionSymbol* function) {
    if (!areFunctionSignaturesEquivalentForRedeclaration(
            unit_, function->type(), type, function->templateDeclaration(),
            templateHead,
            /*isOutOfLineDeclaration=*/true)) {
      return false;
    }
    if (!TemplateEquivalence{unit_}.same(function->trailingRequiresClause(),
                                         trailingRequiresClause))
      return false;
    return areFunctionTemplateHeadsEquivalentForRedeclaration(
        unit_, symbol_cast<ClassSymbol>(parentScope),
        function->templateDeclaration(), templateHead);
  };

  if (auto parentClass = symbol_cast<ClassSymbol>(parentScope);
      parentClass && parentClass->name() == name) {
    for (auto ctor : parentClass->constructors()) {
      if (matches(ctor)) return ctor;
    }
  }

  if (auto namespaceSymbol = symbol_cast<NamespaceSymbol>(parentScope)) {
    for (auto candidateScope : inlineNamespaceSet(namespaceSymbol)) {
      if (auto function =
              views::find_function(candidateScope->find(name), matches))
        return function;
    }
    return nullptr;
  }

  return views::find_function(scope->find(name), matches);
}

auto areFunctionTemplateHeadsEquivalentForRedeclaration(
    TranslationUnit* unit, ClassSymbol* enclosingClass,
    TemplateDeclarationAST* existingHead, TemplateDeclarationAST* newHead)
    -> bool {
  existingHead = TemplateEquivalence{unit}.ownFunctionTemplateHead(
      enclosingClass, existingHead);
  newHead = TemplateEquivalence{unit}.ownFunctionTemplateHead(enclosingClass,
                                                              newHead);
  return TemplateEquivalence{unit}.same(existingHead, newHead);
}
}  // namespace cxx
