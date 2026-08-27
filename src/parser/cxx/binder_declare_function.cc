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
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/substitution.h>
#include <cxx/symbols.h>
#include <cxx/template_argument_deduction.h>
#include <cxx/template_equivalence.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <format>

namespace cxx {
namespace {
auto areFunctionSignaturesEquivalentForRedeclaration(TranslationUnit* unit,
                                                     const Type* lhs,
                                                     const Type* rhs) -> bool {
  if (!unit || !lhs || !rhs) return false;
  if (unit->typeTraits().is_same(lhs, rhs)) return true;

  auto lhsFn = type_cast<FunctionType>(lhs);
  auto rhsFn = type_cast<FunctionType>(rhs);
  if (!lhsFn || !rhsFn) return false;

  if (!areRedeclarationTypesCompatible(unit, lhsFn->returnType(),
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

[[nodiscard]] auto declaresExplicitObjectParameter(
    FunctionDeclaratorChunkAST* prototype) -> bool {
  if (!prototype || !prototype->parameterDeclarationClause) return false;
  auto parameters =
      prototype->parameterDeclarationClause->parameterDeclarationList;
  if (!parameters) return false;
  auto parameter = ast_cast<ParameterDeclarationAST>(parameters->value);
  return parameter && parameter->isThisIntroduced;
}
}  // namespace

struct [[nodiscard]] Binder::DeclareFunction {
  Binder& binder;
  DeclaratorAST* declarator = nullptr;
  const Decl& decl;
  FunctionDeclaratorChunkAST* functionDeclarator = nullptr;
  FunctionSymbol* functionSymbol = nullptr;
  FunctionSymbol* shadowedFunction = nullptr;
  bool addSymbolToParentScope = true;

  DeclareFunction(Binder& binder, DeclaratorAST* declarator, const Decl& decl,
                  bool addSymbolToParentScope)
      : binder(binder),
        declarator(declarator),
        decl(decl),
        addSymbolToParentScope(addSymbolToParentScope) {}

  struct NamedTemplateSpecialization {
    FunctionSymbol* primary = nullptr;
    std::vector<TemplateArgument> arguments;
  };

  auto control() const -> Control* { return binder.control(); }
  auto scope() const -> ScopeSymbol* { return binder.scope(); }

  auto isTemplateFunction() const -> bool {
    return scope()->isTemplateParameters();
  }

  auto isDestructor() const -> bool {
    return name_cast<DestructorId>(decl.getName()) != nullptr;
  }

  auto declaringScopeForFunction() const -> ScopeSymbol*;
  auto namedTemplateSpecialization() const
      -> std::optional<NamedTemplateSpecialization>;
  void mergeAsCRedeclaration(FunctionSymbol* otherFunction);
  auto mergeWithMatchingOverload(OverloadSetSymbol* overloadSet) -> bool;
  void checkCRedeclaration(ScopeSymbol* declaringScope);
  [[nodiscard]] auto isLexicallyInsideClass() const -> bool;
  void reportDifferentKindOfSymbol(ScopeSymbol* declaringScope);
  void reportMemberRedeclaration(FunctionSymbol* previous);

  void applyVirtualFlagsFromDeclarator();
  auto enclosingClass() const -> ClassSymbol*;
  void checkVirtualSpecifierOutsideClass();
  void checkOverrideAndFinalSpecifiers(FunctionSymbol* overridden);
  void checkCovariantReturnType(FunctionSymbol* overridden);

  auto declare() -> FunctionSymbol*;

  void checkRedeclaration();
  void checkConstructor();

  void inheritAbiTags(FunctionSymbol* canonical);
  void checkDeclSpecifiers();
  void checkExternalLinkageSpec();

  void checkVirtualSpecifier();
  void checkExplicitObjectParameter();
  void checkDestructorParameters();
  [[nodiscard]] auto declaresClassMember() const -> bool;
  void mergeRedeclaration();
};

auto Binder::declareFunction(DeclaratorAST* declarator, const Decl& decl,
                             bool addSymbolToParentScope) -> FunctionSymbol* {
  return DeclareFunction{*this, declarator, decl, addSymbolToParentScope}
      .declare();
}

auto Binder::DeclareFunction::declare() -> FunctionSymbol* {
  functionDeclarator = getFunctionPrototype(declarator);

  auto name = decl.getName();
  auto returnType = decl.getReturnType(scope());
  auto type =
      type_cast<FunctionType>(binder.resolveMemberOfCurrentInstantiation(
          getDeclaratorType(binder.unit_, declarator, returnType),
          binder.currentInstantiationOf(binder.declaringScope())));

  auto originalScope = binder.declaringScope();
  auto targetScope = !decl.specs.isFriend
                         ? binder.scopeForBlockDecl(originalScope)
                         : originalScope;

  functionSymbol = control()->newFunctionSymbol(targetScope, decl.location());
  functionSymbol->setName(name);
  functionSymbol->setType(type);

  functionSymbol->setTrailingRequiresClause(decl.trailingRequiresClause);

  functionSymbol->setExplicitObjectParameter(
      declaresExplicitObjectParameter(functionDeclarator));

  if (functionDeclarator && functionDeclarator->exceptionSpecifier)
    functionSymbol->setExceptionSpecifier(true);

  binder.applyImplicitExceptionSpecification(functionSymbol);

  if (binder.isC() && binder.unit_->config().allowUnprototypedFunctions &&
      functionDeclarator && !functionDeclarator->parameterDeclarationClause) {
    functionSymbol->setNoPrototype(true);
  }

  checkDeclSpecifiers();
  checkExternalLinkageSpec();
  checkVirtualSpecifier();
  checkExplicitObjectParameter();
  checkDestructorParameters();

  if (functionSymbol->isConstructor()) {
    checkConstructor();
    return functionSymbol;
  }

  if (addSymbolToParentScope) checkRedeclaration();

  auto namedSpecialization = namedTemplateSpecialization();
  if (namedSpecialization) {
    namedSpecialization->primary->addSpecialization(
        binder.unit_, namedSpecialization->arguments, functionSymbol);
  }

  if (targetScope != originalScope) {
    if (functionSymbol->canonical() == functionSymbol)
      functionSymbol->setHidden(true);
    binder.injectUsing(originalScope, name, functionSymbol->canonical(),
                       functionSymbol->location());
  }

  if (decl.specs.isFriend) {
    ClassSymbol* befriendingClass = nullptr;
    for (auto symbol = static_cast<Symbol*>(originalScope); symbol;
         symbol = symbol->parent()) {
      befriendingClass = symbol_cast<ClassSymbol>(symbol);
      if (befriendingClass) break;
    }

    if (namedSpecialization) {
      namedSpecialization->primary->canonical()->addBefriendingClass(
          befriendingClass, std::move(namedSpecialization->arguments));
    } else {
      functionSymbol->canonical()->addBefriendingClass(befriendingClass);
    }
  }

  return functionSymbol;
}

auto Binder::DeclareFunction::namedTemplateSpecialization() const
    -> std::optional<NamedTemplateSpecialization> {
  auto declaratorId = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator);
  if (!declaratorId) return std::nullopt;
  auto templateId = ast_cast<SimpleTemplateIdAST>(declaratorId->unqualifiedId);
  if (!templateId || hasDependentTemplateArguments(binder.unit_, templateId))
    return std::nullopt;

  auto functionType = type_cast<FunctionType>(functionSymbol->type());
  if (!functionType || isDependent(binder.unit_, functionType))
    return std::nullopt;

  auto primary = symbol_cast<FunctionSymbol>(templateId->symbol);
  if (!primary || !primary->templateDeclaration()) return std::nullopt;

  TemplateArgumentDeduction deduction{binder.unit_};
  auto deducedArgs = deduction.deduceFromTargetType(
      primary, functionType, templateId->templateArgumentList);
  if (!deducedArgs.has_value()) return std::nullopt;

  auto substitution = Substitution::make(
      binder.unit_, primary->templateDeclaration(), *deducedArgs);
  if (!substitution) return std::nullopt;
  return NamedTemplateSpecialization{
      primary, std::move(*substitution).templateArguments()};
}

auto Binder::DeclareFunction::declaringScopeForFunction() const
    -> ScopeSymbol* {
  auto declaringScope = binder.declaringScope();

  if (!functionSymbol->isFriend()) {
    if (!declaringScope->isClassOrNamespace()) {
      if (auto ns = declaringScope->enclosingNamespace()) return ns;
    }
    return declaringScope;
  }

  if (declaringScope->isNamespace()) return declaringScope;

  auto enclosingNamespace = declaringScope->enclosingNamespace();
  if (enclosingNamespace) return enclosingNamespace;

  return declaringScope;
}

void Binder::DeclareFunction::mergeAsCRedeclaration(
    FunctionSymbol* otherFunction) {
  auto canonical = otherFunction->canonical();
  canonical->addRedeclaration(functionSymbol);
  if (canonical->hasNoPrototype() && !functionSymbol->hasNoPrototype()) {
    canonical->setType(functionSymbol->type());
    canonical->setNoPrototype(false);
  }
  mergeRedeclaration();
}

auto Binder::DeclareFunction::mergeWithMatchingOverload(
    OverloadSetSymbol* overloadSet) -> bool {
  for (auto existingFunction : overloadSet->declaredFunctions()) {
    if (existingFunction->isSpecialization()) continue;

    auto existingTemplateDecl = existingFunction->templateDeclaration();
    auto newTemplateHead = decl.specs.templateHead;
    auto headsEquivalent = areFunctionTemplateHeadsEquivalentForRedeclaration(
        binder.unit_, symbol_cast<ClassSymbol>(declaringScopeForFunction()),
        existingTemplateDecl, newTemplateHead);
    if (!headsEquivalent) {
      auto instantiatingFunction =
          symbol_cast<FunctionSymbol>(binder.instantiatingSymbol());
      headsEquivalent =
          instantiatingFunction &&
          instantiatingFunction->canonical() == existingFunction->canonical() &&
          (existingTemplateDecl != nullptr) != (newTemplateHead != nullptr);
    }
    if (!headsEquivalent) {
      continue;
    }

    bool sigEq = areFunctionSignaturesEquivalentForRedeclaration(
        binder.unit_, existingFunction->type(), functionSymbol->type());

    if (!sigEq && existingTemplateDecl && newTemplateHead &&
        existingTemplateDecl->depth != newTemplateHead->depth) {
      int ownParamCount = 0;
      for ([[maybe_unused]] auto p :
           ListView{newTemplateHead->templateParameterList}) {
        ++ownParamCount;
      }
      sigEq = typesEquivalentModuloOwnHeadDepth(
          binder.unit_, existingFunction->type(), functionSymbol->type(),
          existingTemplateDecl->depth, newTemplateHead->depth, ownParamCount);
    }

    if (!sigEq) continue;

    if (!trailingRequiresClausesEquivalent(
            binder.unit_, existingFunction->trailingRequiresClause(),
            functionSymbol->trailingRequiresClause()))
      continue;

    reportMemberRedeclaration(existingFunction);

    auto canonical = existingFunction->canonical();
    canonical->addRedeclaration(functionSymbol);
    mergeRedeclaration();
    return true;
  }

  return false;
}

void Binder::DeclareFunction::reportDifferentKindOfSymbol(
    ScopeSymbol* declaringScope) {
  if (!isLexicallyInsideClass()) return;

  for (auto candidate : declaringScope->find(functionSymbol->name())) {
    if (symbol_cast<FunctionSymbol>(candidate)) continue;
    if (symbol_cast<OverloadSetSymbol>(candidate)) continue;
    if (symbol_cast<UsingDeclarationSymbol>(candidate)) continue;
    if (symbol_cast<InjectedClassNameSymbol>(candidate)) continue;
    if (is_type(candidate)) continue;

    binder.error(functionSymbol->location(),
                 std::format("redefinition of '{}' as different kind of symbol",
                             to_string(functionSymbol->name())));
    binder.note(candidate->location(), "previous definition is here");
    return;
  }
}

auto Binder::DeclareFunction::isLexicallyInsideClass() const -> bool {
  if (functionSymbol->isFriend()) return false;

  auto enclosingClass = symbol_cast<ClassSymbol>(binder.declaringScope());
  if (!enclosingClass) return false;
  if (enclosingClass->isSpecialization()) return false;

  auto id = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator);
  return id && !id->nestedNameSpecifier;
}

void Binder::DeclareFunction::reportMemberRedeclaration(
    FunctionSymbol* previous) {
  if (!isLexicallyInsideClass()) return;

  const bool previousIsTemplate = previous->templateDeclaration() != nullptr;
  const bool redeclarationIsTemplate = decl.specs.templateHead != nullptr;
  if (previousIsTemplate != redeclarationIsTemplate) return;

  if (isDependent(binder.unit_, previous->type())) return;
  if (isDependent(binder.unit_, functionSymbol->type())) return;

  binder.error(functionSymbol->location(),
               functionSymbol->isConstructor()
                   ? "constructor cannot be redeclared"
                   : "class member cannot be redeclared");
  binder.note(previous->canonical()->location(),
              "previous declaration is here");
}

void Binder::DeclareFunction::checkRedeclaration() {
  if (auto id = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator)) {
    if (auto nns = id->nestedNameSpecifier) {
      if (!nns->symbol && !binder.declaringScope()->isClass()) return;
    }
  }

  auto declaringScope = declaringScopeForFunction();

  if (binder.isC()) {
    checkCRedeclaration(declaringScope);
    return;
  }

  if (functionSymbol->isFriend() && !declaringScope->isClass()) {
    functionSymbol->setHidden(true);
  }

  reportDifferentKindOfSymbol(declaringScope);

  auto overloadSet = binder.overloadSetFor(
      declaringScope, functionSymbol->name(), functionSymbol->location());

  if (!mergeWithMatchingOverload(overloadSet)) {
    overloadSet->addFunction(functionSymbol);
  }

  binder.mergeDefaultArguments(functionSymbol, declarator);
}

void Binder::DeclareFunction::checkCRedeclaration(ScopeSymbol* declaringScope) {
  for (Symbol* candidate : declaringScope->find(functionSymbol->name())) {
    auto otherFunction = symbol_cast<FunctionSymbol>(candidate);
    if (!otherFunction) continue;

    auto canonical = otherFunction->canonical();
    const bool canMerge =
        (binder.unit_->config().allowUnprototypedFunctions &&
         canonical->hasNoPrototype()) ||
        areFunctionSignaturesEquivalentForRedeclaration(
            binder.unit_, canonical->type(), functionSymbol->type());
    if (canMerge) {
      mergeAsCRedeclaration(otherFunction);
    } else {
      binder.error(functionSymbol->location(),
                   std::format("conflicting types for '{}'",
                               to_string(functionSymbol->name())));
      binder.note(canonical->location(),
                  std::format("previous declaration of '{}' is here",
                              to_string(canonical->name())));
    }
    return;
  }

  declaringScope->addSymbol(functionSymbol);

  binder.mergeDefaultArguments(functionSymbol, declarator);
}

void Binder::DeclareFunction::checkConstructor() {
  auto classScope = binder.scope();
  if (classScope && classScope->isTemplateParameters()) {
    classScope = classScope->parent();
  }
  auto enclosingClass = symbol_cast<ClassSymbol>(classScope);

  if (!enclosingClass) {
    cxx_runtime_error("constructor must be declared inside a class");
  }

  if (!mergeWithMatchingOverload(enclosingClass->constructorOverloadSet())) {
    enclosingClass->addConstructor(functionSymbol);
  }

  binder.mergeDefaultArguments(functionSymbol, declarator);
}

void Binder::DeclareFunction::checkDeclSpecifiers() {
  binder.applySpecifiers(functionSymbol, decl.specs);
}

void Binder::DeclareFunction::checkDestructorParameters() {
  if (!functionSymbol->isDestructor()) return;
  if (!functionDeclarator) return;

  auto parameterDeclarationClause =
      functionDeclarator->parameterDeclarationClause;
  if (!parameterDeclarationClause) return;
  if (!parameterDeclarationClause->parameterDeclarationList) return;

  binder.error(parameterDeclarationClause->parameterDeclarationList->value
                   ->firstSourceLocation(),
               "a destructor cannot have any parameters");
}

void Binder::DeclareFunction::checkExplicitObjectParameter() {
  if (!functionSymbol->hasExplicitObjectParameter()) return;

  auto explicitObjectLoc = functionDeclarator->parameterDeclarationClause
                               ->parameterDeclarationList->value->thisLoc;

  if (!declaresClassMember()) {
    binder.error(explicitObjectLoc,
                 "an explicit object parameter is only allowed in a member "
                 "function declaration");
    functionSymbol->setExplicitObjectParameter(false);
    return;
  }

  if (functionSymbol->isConstructor()) {
    binder.error(explicitObjectLoc,
                 "a constructor cannot have an explicit object parameter");
    functionSymbol->setExplicitObjectParameter(false);
    return;
  }

  if (functionSymbol->isDestructor()) {
    binder.error(explicitObjectLoc,
                 "a destructor cannot have an explicit object parameter");
    functionSymbol->setExplicitObjectParameter(false);
    return;
  }

  if (functionSymbol->isStatic()) {
    binder.error(explicitObjectLoc,
                 "a static member function cannot have an explicit object "
                 "parameter");
  }

  if (functionSymbol->isVirtual()) {
    binder.error(explicitObjectLoc,
                 "a virtual member function cannot have an explicit object "
                 "parameter");
  }

  if (functionDeclarator->cvQualifierList) {
    binder.error(explicitObjectLoc,
                 "a member function with an explicit object parameter cannot "
                 "be cv-qualified");
  }

  if (functionDeclarator->refLoc) {
    binder.error(explicitObjectLoc,
                 "a member function with an explicit object parameter cannot "
                 "have a ref-qualifier");
  }
}

void Binder::DeclareFunction::checkExternalLinkageSpec() {
  if (binder.isC()) {
    functionSymbol->setLanguageLinkage(LanguageKind::kC);
    return;
  }

  if (scope()->isClass()) {
    functionSymbol->setLanguageLinkage(LanguageKind::kCXX);
    return;
  }

  functionSymbol->setLanguageLinkage(binder.languageLinkage_);
}

void Binder::DeclareFunction::applyVirtualFlagsFromDeclarator() {
  if (!functionDeclarator) return;
  if (functionDeclarator->isOverride) functionSymbol->setOverride(true);
  if (functionDeclarator->isFinal) functionSymbol->setFinal(true);

  if (!functionDeclarator->isPure) return;

  functionSymbol->setPure(true);
  functionSymbol->setVirtual(true);
}

auto Binder::DeclareFunction::enclosingClass() const -> ClassSymbol* {
  return symbol_cast<ClassSymbol>(scope());
}

auto Binder::DeclareFunction::declaresClassMember() const -> bool {
  auto enclosing = scope();
  while (enclosing && enclosing->isTemplateParameters())
    enclosing = enclosing->parent();
  if (symbol_cast<ClassSymbol>(enclosing)) return true;

  auto declaratorId = ast_cast<IdDeclaratorAST>(declarator->coreDeclarator);
  return declaratorId && declaratorId->nestedNameSpecifier;
}

void Binder::DeclareFunction::checkVirtualSpecifierOutsideClass() {
  if (!functionSymbol->isVirtual() && !functionSymbol->isOverride() &&
      !functionSymbol->isFinal()) {
    return;
  }

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
}

void Binder::DeclareFunction::checkCovariantReturnType(
    FunctionSymbol* overridden) {
  auto overriderType = type_cast<FunctionType>(functionSymbol->type());
  auto overriddenType = type_cast<FunctionType>(overridden->type());
  if (!overriderType || !overriddenType) return;

  auto overriderReturnType = overriderType->returnType();
  auto overriddenReturnType = overriddenType->returnType();
  if (!overriderReturnType || !overriddenReturnType) return;

  if (isDependent(binder.unit_, overriderReturnType) ||
      isDependent(binder.unit_, overriddenReturnType)) {
    return;
  }

  if (binder.traits.is_covariant_return_type(overriddenReturnType,
                                             overriderReturnType)) {
    return;
  }

  binder.error(functionSymbol->location(),
               std::format("return type of virtual function '{}' is not "
                           "covariant with the return type of the function it "
                           "overrides",
                           to_string(functionSymbol->name())));
  binder.note(overridden->location(), "overridden virtual function is here");
}

void Binder::DeclareFunction::checkOverrideAndFinalSpecifiers(
    FunctionSymbol* overridden) {
  if (functionSymbol->isOverride() && !overridden) {
    binder.error(functionSymbol->location(),
                 std::format("'{}' marked 'override' but does not override "
                             "any member function",
                             to_string(functionSymbol->name())));
  }

  if (!functionSymbol->isFinal() || functionSymbol->isVirtual()) return;

  binder.error(functionSymbol->location(),
               std::format("'{}' marked 'final' but is not virtual",
                           to_string(functionSymbol->name())));
}

auto Binder::findOverriddenFunctions(ClassSymbol* cls, FunctionSymbol* fn)
    -> std::vector<FunctionSymbol*> {
  std::unordered_set<ClassSymbol*> visited;
  std::vector<FunctionSymbol*> overriddenFunctions;
  findOverriddenFunctionsImpl(cls, fn, visited, overriddenFunctions);
  return overriddenFunctions;
}

void Binder::findOverriddenFunctionsImpl(
    ClassSymbol* cls, FunctionSymbol* fn,
    std::unordered_set<ClassSymbol*>& visited,
    std::vector<FunctionSymbol*>& overriddenFunctions) {
  for (auto base : cls->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass || !visited.insert(baseClass).second) continue;
    baseClass = baseClass->resolvedDefinition();

    auto checkMember = [&](FunctionSymbol* member) {
      if (!member->isVirtual()) return;
      if (!traits.is_corresponding_overrider(fn, member)) return;
      if (!std::ranges::contains(overriddenFunctions, member))
        overriddenFunctions.push_back(member);
    };

    for (auto symbol : baseClass->members()) {
      if (auto func = symbol_cast<FunctionSymbol>(symbol)) {
        checkMember(func);
      } else if (auto ovl = symbol_cast<OverloadSetSymbol>(symbol)) {
        for (auto func : ovl->declaredFunctions()) checkMember(func);
      }
    }

    findOverriddenFunctionsImpl(baseClass, fn, visited, overriddenFunctions);
  }
}

void Binder::DeclareFunction::checkVirtualSpecifier() {
  applyVirtualFlagsFromDeclarator();

  auto cls = enclosingClass();
  if (!cls) {
    checkVirtualSpecifierOutsideClass();
    return;
  }

  if (functionSymbol->isConstructor()) return;

  auto overriddenFunctions =
      binder.findOverriddenFunctions(cls, functionSymbol);
  auto overridden =
      overriddenFunctions.empty() ? nullptr : overriddenFunctions.front();

  if (overridden) {
    for (auto function : overriddenFunctions)
      functionSymbol->addOverriddenFunction(function);
    functionSymbol->setVirtual(true);

    for (auto function : overriddenFunctions) {
      if (function->isFinal()) {
        binder.error(
            functionSymbol->location(),
            std::format("declaration of '{}' overrides a 'final' function",
                        to_string(functionSymbol->name())));
        binder.note(function->location(), "overridden final function is here");
      }
      checkCovariantReturnType(function);
    }
  }

  if (!overridden) {
    for (auto base : cls->baseClasses()) {
      auto baseSymbol = base->symbol();
      if (!baseSymbol) return;
      if (auto baseType = baseSymbol->type();
          baseType && isDependent(binder.unit_, baseType)) {
        return;
      }
    }
  }

  checkOverrideAndFinalSpecifiers(overridden);
}

void Binder::DeclareFunction::inheritAbiTags(FunctionSymbol* canonical) {
  functionSymbol->setAbiTags(canonical->abiTagList());
}

void Binder::DeclareFunction::mergeRedeclaration() {
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
  if (canonical->isExplicit()) {
    if (!binder.instantiatingSymbol()) functionSymbol->setExplicit(true);
  }
  if (canonical->isOverride()) functionSymbol->setOverride(true);
  if (canonical->isFinal()) functionSymbol->setFinal(true);
  if (canonical->isPure()) functionSymbol->setPure(true);
  if (canonical->hasCLinkage())
    functionSymbol->setLanguageLinkage(LanguageKind::kC);

  inheritAbiTags(canonical);

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

  for (; canonIt != canonEnd && redeclIt != redeclEnd; ++canonIt, ++redeclIt) {
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
}  // namespace cxx
