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
#include <cxx/ast_rewriter.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/views/symbols.h>

// std
#include <format>

namespace cxx {

namespace {

auto arrayBoundToString(const Type* type) -> std::optional<std::string> {
  if (auto bounded = type_cast<BoundedArrayType>(type)) {
    return std::to_string(bounded->size());
  }
  return std::nullopt;
}

auto isEffectivelyUnboundedArray(Control* control, const Type* type) -> bool {
  if (!control || !type) return false;
  if (control->is_unbounded_array(type)) return true;

  auto unresolved = type_cast<UnresolvedBoundedArrayType>(type);
  if (!unresolved) return false;
  return !arrayBoundToString(type).has_value();
}

auto areRedeclarationTypesCompatible(Control* control, const Type* lhs,
                                     const Type* rhs) -> bool {
  if (!control || !lhs || !rhs) return false;

  while (auto qual = type_cast<QualType>(lhs)) {
    lhs = qual->elementType();
  }
  while (auto qual = type_cast<QualType>(rhs)) {
    rhs = qual->elementType();
  }

  if (control->is_same(lhs, rhs)) return true;

  if (!control->is_array(lhs) || !control->is_array(rhs)) return false;

  auto lhsElement = control->get_element_type(lhs);
  auto rhsElement = control->get_element_type(rhs);
  if (!areRedeclarationTypesCompatible(control, lhsElement, rhsElement)) {
    return false;
  }

  if (isEffectivelyUnboundedArray(control, lhs) ||
      isEffectivelyUnboundedArray(control, rhs)) {
    return true;
  }

  auto lhsBound = arrayBoundToString(lhs);
  auto rhsBound = arrayBoundToString(rhs);
  if (!lhsBound || !rhsBound) return true;
  return *lhsBound == *rhsBound;
}

auto areFunctionSignaturesEquivalentForRedeclaration(Control* control,
                                                     const Type* lhs,
                                                     const Type* rhs) -> bool {
  if (!control || !lhs || !rhs) return false;
  if (control->is_same(lhs, rhs)) return true;

  auto lhsFn = type_cast<FunctionType>(lhs);
  auto rhsFn = type_cast<FunctionType>(rhs);
  if (!lhsFn || !rhsFn) return false;

  if (!control->is_same(lhsFn->returnType(), rhsFn->returnType())) return false;
  if (lhsFn->cvQualifiers() != rhsFn->cvQualifiers()) return false;
  if (lhsFn->refQualifier() != rhsFn->refQualifier()) return false;
  if (lhsFn->isVariadic() != rhsFn->isVariadic()) return false;

  const auto& lhsParams = lhsFn->parameterTypes();
  const auto& rhsParams = rhsFn->parameterTypes();
  if (lhsParams.size() != rhsParams.size()) return false;

  for (std::size_t i = 0; i < lhsParams.size(); ++i) {
    if (!areRedeclarationTypesCompatible(control, lhsParams[i], rhsParams[i])) {
      return false;
    }
  }

  return true;
}

}  // namespace

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

  auto isTemplateFunction() const -> bool {
    return scope()->isTemplateParameters();
  }

  auto isDestructor() const -> bool {
    return name_cast<DestructorId>(decl.getName()) != nullptr;
  }

  auto declaringScopeForFunction() const -> ScopeSymbol*;
  void mergeAsCRedeclaration(FunctionSymbol* otherFunction);
  auto createOverloadSet(ScopeSymbol* declaringScope,
                         FunctionSymbol* otherFunction) -> OverloadSetSymbol*;
  auto mergeWithMatchingOverload(OverloadSetSymbol* overloadSet) -> bool;

  void applyVirtualFlagsFromDeclarator();
  auto enclosingClass() const -> ClassSymbol*;
  void checkVirtualSpecifierOutsideClass();
  void checkOverrideAndFinalSpecifiers(FunctionSymbol* overridden);

  auto declare() -> FunctionSymbol*;

  void checkRedeclaration();
  void checkConstructor();
  void checkDeclSpecifiers();
  void checkExternalLinkageSpec();

  auto findOverriddenFunction(ClassSymbol* cls, FunctionSymbol* fn)
      -> FunctionSymbol*;

  auto findOverriddenFunctionImpl(ClassSymbol* cls, FunctionSymbol* fn,
                                  std::unordered_set<ClassSymbol*>& visited)
      -> FunctionSymbol*;
  void checkVirtualSpecifier();
  void mergeRedeclaration();
};

auto Binder::declareFunction(DeclaratorAST* declarator, const Decl& decl)
    -> FunctionSymbol* {
  return DeclareFunction{*this, declarator, decl}.declare();
}

auto Binder::DeclareFunction::declare() -> FunctionSymbol* {
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

auto Binder::DeclareFunction::declaringScopeForFunction() const
    -> ScopeSymbol* {
  if (!functionSymbol->isFriend()) return binder.declaringScope();

  auto declaringScope = binder.declaringScope();
  if (declaringScope->isNamespace()) return declaringScope;

  auto enclosingNamespace = declaringScope->enclosingNamespace();
  if (enclosingNamespace) return enclosingNamespace;

  return declaringScope;
}

void Binder::DeclareFunction::mergeAsCRedeclaration(
    FunctionSymbol* otherFunction) {
  auto canonical = otherFunction->canonical();
  canonical->addRedeclaration(functionSymbol);
  mergeRedeclaration();
}

auto Binder::DeclareFunction::createOverloadSet(ScopeSymbol* declaringScope,
                                                FunctionSymbol* otherFunction)
    -> OverloadSetSymbol* {
  auto overloadSet = control()->newOverloadSetSymbol(declaringScope,
                                                     otherFunction->location());
  overloadSet->setName(otherFunction->name());
  overloadSet->addFunction(otherFunction);
  declaringScope->replaceSymbol(otherFunction, overloadSet);
  return overloadSet;
}

auto Binder::DeclareFunction::mergeWithMatchingOverload(
    OverloadSetSymbol* overloadSet) -> bool {
  for (auto existingFunction : overloadSet->functions()) {
    if (!areFunctionSignaturesEquivalentForRedeclaration(
            control(), existingFunction->type(), functionSymbol->type())) {
      continue;
    }

    auto canonical = existingFunction->canonical();
    canonical->addRedeclaration(functionSymbol);
    mergeRedeclaration();
    return true;
  }

  return false;
}

void Binder::DeclareFunction::checkRedeclaration() {
  auto declaringScope = declaringScopeForFunction();

  OverloadSetSymbol* overloadSet = nullptr;

  for (Symbol* candidate : declaringScope->find(functionSymbol->name())) {
    overloadSet = symbol_cast<OverloadSetSymbol>(candidate);
    if (overloadSet) break;

    if (auto otherFunction = symbol_cast<FunctionSymbol>(candidate)) {
      if (binder.is_parsing_c()) {
        mergeAsCRedeclaration(otherFunction);
        break;
      }

      overloadSet = createOverloadSet(declaringScope, otherFunction);
      break;
    }
  }

  if (overloadSet) {
    if (!mergeWithMatchingOverload(overloadSet)) {
      overloadSet->addFunction(functionSymbol);
    }

    binder.mergeDefaultArguments(functionSymbol, declarator);
    return;
  }

  if (functionSymbol->isFriend() && !declaringScope->isClass()) {
    functionSymbol->setHidden(true);
  }

  declaringScope->addSymbol(functionSymbol);

  binder.mergeDefaultArguments(functionSymbol, declarator);
}

void Binder::DeclareFunction::checkConstructor() {
  // For constructor templates, binder.scope() is the TemplateParametersSymbol.
  // Look through to find the enclosing class.
  auto classScope = binder.scope();
  if (classScope && classScope->isTemplateParameters()) {
    classScope = classScope->enclosingNonTemplateParametersScope();
  }
  auto enclosingClass = symbol_cast<ClassSymbol>(classScope);

  if (!enclosingClass) {
    cxx_runtime_error("constructor must be declared inside a class");
  }

  for (auto ctor : enclosingClass->constructors()) {
    if (areFunctionSignaturesEquivalentForRedeclaration(
            control(), ctor->type(), functionSymbol->type())) {
      auto canon = ctor->canonical();
      canon->addRedeclaration(functionSymbol);
      mergeRedeclaration();
      break;
    }
  }

  binder.mergeDefaultArguments(functionSymbol, declarator);

  enclosingClass->addConstructor(functionSymbol);
}

void Binder::DeclareFunction::checkDeclSpecifiers() {
  binder.applySpecifiers(functionSymbol, decl.specs);
}

void Binder::DeclareFunction::checkExternalLinkageSpec() {
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

void Binder::DeclareFunction::applyVirtualFlagsFromDeclarator() {
  if (functionDeclarator->isOverride) functionSymbol->setOverride(true);
  if (functionDeclarator->isFinal) functionSymbol->setFinal(true);

  if (!functionDeclarator->isPure) return;

  functionSymbol->setPure(true);
  functionSymbol->setVirtual(true);
}

auto Binder::DeclareFunction::enclosingClass() const -> ClassSymbol* {
  return symbol_cast<ClassSymbol>(scope());
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

auto Binder::DeclareFunction::findOverriddenFunction(ClassSymbol* cls,
                                                     FunctionSymbol* fn)
    -> FunctionSymbol* {
  std::unordered_set<ClassSymbol*> visited;
  return findOverriddenFunctionImpl(cls, fn, visited);
}

auto Binder::DeclareFunction::findOverriddenFunctionImpl(
    ClassSymbol* cls, FunctionSymbol* fn,
    std::unordered_set<ClassSymbol*>& visited) -> FunctionSymbol* {
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

void Binder::DeclareFunction::checkVirtualSpecifier() {
  applyVirtualFlagsFromDeclarator();

  auto cls = enclosingClass();
  if (!cls) {
    checkVirtualSpecifierOutsideClass();
    return;
  }

  // Constructors cannot be virtual
  if (functionSymbol->isConstructor()) return;

  // Look up the overridden virtual function in base classes
  auto overridden = findOverriddenFunction(cls, functionSymbol);

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

  checkOverrideAndFinalSpecifiers(overridden);
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