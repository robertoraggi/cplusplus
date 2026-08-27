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
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>
#include <cxx/util.h>
#include <cxx/views/symbols.h>

#include <algorithm>
#include <format>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace cxx {
[[nodiscard]] static auto defersClassSemanticCompletion(
    TranslationUnit* unit, ClassSymbol* classSymbol) -> bool {
  if (!isEnclosedInDependentTemplate(unit, classSymbol,
                                     /*stopAtConcreteSpecialization=*/true))
    return false;
  if (!classSymbol->isSpecialization()) return true;
  auto templateParameters = classSymbol->templateParameters();
  return !templateParameters ||
         !templateParameters->isExplicitTemplateSpecialization();
}

struct [[nodiscard]] Binder::CompleteClass {
  Binder& binder;
  ClassSpecifierAST* ast;
  ClassSymbol* classSymbol;
  Arena* pool;

  CompleteClass(Binder& b, ClassSpecifierAST* a)
      : binder(b), ast(a), classSymbol(a->symbol), pool(b.unit_->arena()) {}

  CompleteClass(Binder& b, ClassSymbol* cls)
      : binder(b), ast(nullptr), classSymbol(cls), pool(b.unit_->arena()) {}

  [[nodiscard]] auto isCapturingClosure() const -> bool {
    return classSymbol->isClosureType() && classSymbol->hasLambdaCapture();
  }

  auto control() const -> Control* { return binder.control(); }

  void complete(bool deferExceptionSpecificationChecks);

  void markComplete();
  auto shouldSynthesizeSpecialMembers() const -> bool;
  void synthesizeSpecialMembers();

  auto buildRecordLayout() -> std::expected<bool, std::string>;

  auto newDefaultedFunction(const Name* name, const Type* type)
      -> FunctionSymbol*;
  void attachDeclaration(FunctionSymbol* symbol, UnqualifiedIdAST* id);
  auto makeCtorNameId() -> NameIdAST*;
  void addFunctionToClassScope(FunctionSymbol* symbol);
  auto hasUserDeclaredAssignmentOperator(bool moveForm) const -> bool;
  void addDefaultConstructor();
  [[nodiscard]] auto defaultConstructorIsDeleted() const -> bool;
  void addCopyConstructor();
  void addMoveConstructor();
  void addCopyAssignmentOperator();
  void addMoveAssignmentOperator();
  void addDestructor();
  void addInheritedConstructors();
  auto declareInheritedConstructor(FunctionSymbol* inherited)
      -> FunctionSymbol*;
  auto directInheritedConstructor(FunctionSymbol* inherited)
      -> std::pair<BaseClassSymbol*, FunctionSymbol*>;
  void synthesizeInheritedConstructorBody(FunctionSymbol* fn);

  void synthesizeStructorVariants();
  auto newStructorVariant(FunctionSymbol* principal) -> FunctionSymbol*;
  void attachVariantDefinition(FunctionSymbol* variant, UnqualifiedIdAST* id,
                               FunctionBodyAST* body);
  auto makeThisExpr() -> ExpressionAST*;
  auto makeParamRef(ParameterSymbol* param) -> ExpressionAST*;
  auto makeQualifier(ClassSymbol* cls) -> NestedNameSpecifierAST*;
  auto makeStructorCallStatement(FunctionSymbol* callee,
                                 ExpressionAST* objectPtr) -> StatementAST*;
  auto pickVBaseConstructor(ClassSymbol* vbase, bool isCopy, bool isMove)
      -> FunctionSymbol*;
  void synthesizeCompleteObjectCtor(FunctionSymbol* ctor);
  void synthesizeCompleteObjectDtor(FunctionSymbol* dtor);
  void synthesizeDeletingDtor(FunctionSymbol* dtor);

  void synthesizeMemberwiseBodies();
  void typeFieldInitializers();
  void checkOverriderExceptionSpecifications();
  [[nodiscard]] auto hasNonAssignableSubobject(bool moveForm) const -> bool;
  [[nodiscard]] auto hasNonCopyConstructibleSubobject(bool moveForm) const
      -> bool;
  auto ensureSourceParameter(FunctionSymbol* fn) -> ParameterSymbol*;
  auto makeSourceSubobjectRef(ExpressionAST* expr, const Type* type,
                              bool isMove) -> ExpressionAST*;
  void synthesizeCopyMoveCtorBody(FunctionSymbol* fn, bool isMove);
  void synthesizeCopyMoveAssignBody(FunctionSymbol* fn, bool isMove);
};

void Binder::complete(ClassSpecifierAST* ast,
                      bool deferExceptionSpecificationChecks) {
  CompleteClass{*this, ast}.complete(deferExceptionSpecificationChecks);
}

void Binder::completeClosureType(ClassSymbol* classSymbol) {
  CompleteClass{*this, classSymbol}.complete(false);
}

auto Binder::inheritedConstructorFor(ClassSymbol* classSymbol,
                                     FunctionSymbol* baseConstructor)
    -> FunctionSymbol* {
  if (!classSymbol || !baseConstructor) return nullptr;
  classSymbol = classSymbol->resolvedDefinition();

  CompleteClass completeClass{*this, classSymbol};
  auto symbol = completeClass.declareInheritedConstructor(baseConstructor);
  if (!symbol) return nullptr;

  completeClass.synthesizeInheritedConstructorBody(symbol);
  synthesizeCompleteObjectCtor(symbol);
  return symbol;
}

void Binder::synthesizeCompleteObjectCtor(FunctionSymbol* ctor) {
  if (!ctor->isConstructor()) return;
  if (ctor->isDeleted() || ctor->completeObjectVariant()) return;
  if (!type_cast<FunctionType>(ctor->type())) return;

  auto classSymbol = symbol_cast<ClassSymbol>(ctor->parent());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();

  auto layout = classSymbol->layout();
  if (!layout || layout->virtualBases().empty()) return;

  CompleteClass{*this, classSymbol}.synthesizeCompleteObjectCtor(ctor);
}

void Binder::synthesizeDefaultedMemberBody(FunctionSymbol* fn) {
  if (!fn || fn->isDeleted()) return;

  auto def = fn->declaration();
  if (!def || !ast_cast<DefaultFunctionBodyAST>(def->functionBody)) return;

  auto classSymbol = symbol_cast<ClassSymbol>(fn->parent());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  if (classSymbol->isUnion()) return;

  auto canon = fn->canonical();
  auto matches = [&](FunctionSymbol* member) {
    return member && member->canonical() == canon;
  };

  CompleteClass cc{*this, classSymbol};
  if (matches(classSymbol->copyConstructor()))
    cc.synthesizeCopyMoveCtorBody(fn, /*isMove=*/false);
  else if (matches(classSymbol->moveConstructor()))
    cc.synthesizeCopyMoveCtorBody(fn, /*isMove=*/true);
  else if (matches(classSymbol->copyAssignmentOperator()))
    cc.synthesizeCopyMoveAssignBody(fn, /*isMove=*/false);
  else if (matches(classSymbol->moveAssignmentOperator()))
    cc.synthesizeCopyMoveAssignBody(fn, /*isMove=*/true);
}

void Binder::CompleteClass::markComplete() { classSymbol->setComplete(true); }

auto Binder::CompleteClass::shouldSynthesizeSpecialMembers() const -> bool {
  if (!binder.isCxx()) return false;
  if (!classSymbol->name()) return false;
  return true;
}

void Binder::CompleteClass::synthesizeSpecialMembers() {
  const bool userDeclaredCopyConstructor = classSymbol->copyConstructor();
  const bool userDeclaredMoveConstructor = classSymbol->moveConstructor();
  const bool userDeclaredCopyAssignment =
      hasUserDeclaredAssignmentOperator(/*moveForm=*/false);
  const bool userDeclaredMoveAssignment =
      hasUserDeclaredAssignmentOperator(/*moveForm=*/true);
  const bool userDeclaredDestructor = classSymbol->destructor();

  const bool suppressMoveMembers =
      userDeclaredCopyConstructor || userDeclaredMoveConstructor ||
      userDeclaredCopyAssignment || userDeclaredMoveAssignment ||
      userDeclaredDestructor;

  const bool deleteCopyMembers =
      userDeclaredMoveConstructor || userDeclaredMoveAssignment;

  if (isCapturingClosure()) {
    if (auto defaultConstructor = classSymbol->defaultConstructor()) {
      defaultConstructor->setDeleted(true);
    }
  } else {
    addDefaultConstructor();
  }

  addCopyConstructor();
  if (!userDeclaredCopyConstructor) {
    if (auto copyConstructor = classSymbol->copyConstructor()) {
      if (deleteCopyMembers || hasNonCopyConstructibleSubobject(false))
        copyConstructor->setDeleted(true);
    }
  }

  if (!suppressMoveMembers && !hasNonCopyConstructibleSubobject(true))
    addMoveConstructor();

  addCopyAssignmentOperator();
  if (!userDeclaredCopyAssignment) {
    if (auto copyAssignment = classSymbol->copyAssignmentOperator()) {
      if (isCapturingClosure()) {
        copyAssignment->setDeleted(true);
      } else if (deleteCopyMembers ||
                 hasNonAssignableSubobject(/*moveForm=*/false)) {
        copyAssignment->setDeleted(true);
      }
    }
  }

  if (!isCapturingClosure() && !suppressMoveMembers) {
    if (!hasNonAssignableSubobject(/*moveForm=*/true)) {
      addMoveAssignmentOperator();
    }
  }

  addDestructor();

  addInheritedConstructors();
}

void Binder::CompleteClass::addInheritedConstructors() {
  auto overloadSet = classSymbol->constructorOverloadSet();
  if (overloadSet->usingDeclarations().empty()) return;

  for (auto inherited : overloadSet->functions()) {
    if (inherited->templateDeclaration() && !inherited->isSpecialization())
      continue;
    (void)declareInheritedConstructor(inherited);
  }
}

auto Binder::CompleteClass::declareInheritedConstructor(
    FunctionSymbol* inherited) -> FunctionSymbol* {
  auto base = symbol_cast<ClassSymbol>(inherited->parent());
  if (!base || base->resolvedDefinition() == classSymbol) return nullptr;

  auto inheritedType = type_cast<FunctionType>(inherited->type());
  if (!inheritedType) return nullptr;

  auto canonical = inherited->canonical();
  for (auto existing : classSymbol->declaredConstructors()) {
    if (auto from = existing->inheritedConstructor();
        from && from->canonical() == canonical)
      return existing;
  }

  auto symbol = newDefaultedFunction(classSymbol->name(), inherited->type());
  symbol->setInheritedConstructor(inherited);
  symbol->setExplicit(inherited->isExplicit());
  symbol->setConstexpr(inherited->isConstexpr());

  auto params = control()->newFunctionParametersSymbol(symbol, {});
  symbol->addSymbol(params);

  std::vector<ParameterSymbol*> sourceParameters;
  if (auto sourceScope = inherited->functionParameters()) {
    for (auto source : views::members(sourceScope) | views::parameters)
      sourceParameters.push_back(source);
  }

  std::size_t position = 0;
  for (auto parameterType : inheritedType->parameterTypes()) {
    auto param = control()->newParameterSymbol(params, symbol->location());
    param->setType(parameterType);
    if (position < sourceParameters.size()) {
      param->setName(sourceParameters[position]->name());
      param->setDefaultArgument(sourceParameters[position]->defaultArgument());
    }
    params->addSymbol(param);
    ++position;
  }

  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
  return symbol;
}

auto Binder::CompleteClass::directInheritedConstructor(
    FunctionSymbol* inherited) -> std::pair<BaseClassSymbol*, FunctionSymbol*> {
  auto origin = inherited->inheritedConstructorOrigin();
  if (!origin) origin = inherited;
  auto canonical = origin->canonical();

  for (auto baseClass : classSymbol->baseClasses()) {
    auto base = symbol_cast<ClassSymbol>(baseClass->symbol());
    if (!base) continue;
    base = base->resolvedDefinition();

    for (auto constructor : base->declaredConstructors()) {
      auto candidateOrigin = constructor->inheritedConstructorOrigin();
      if (!candidateOrigin) candidateOrigin = constructor;
      if (candidateOrigin->canonical() == canonical)
        return {baseClass, constructor};
    }

    for (auto constructor : base->constructors()) {
      auto candidateOrigin = constructor->inheritedConstructorOrigin();
      if (!candidateOrigin) candidateOrigin = constructor;
      if (candidateOrigin->canonical() != canonical) continue;
      auto direct = binder.inheritedConstructorFor(base, constructor);
      if (direct) return {baseClass, direct};
    }
  }

  return {};
}

void Binder::CompleteClass::synthesizeInheritedConstructorBody(
    FunctionSymbol* fn) {
  auto def = fn->declaration();
  if (!def || !ast_cast<DefaultFunctionBodyAST>(def->functionBody)) return;

  auto inherited = fn->inheritedConstructor();
  auto [baseClass, constructor] = directInheritedConstructor(inherited);
  if (!baseClass || !constructor) return;
  fn->setInheritedConstructor(constructor);

  auto base = symbol_cast<ClassSymbol>(baseClass->symbol());
  if (!base) return;
  base = base->resolvedDefinition();

  auto init = ParenMemInitializerAST::create(pool);
  if (auto id = name_cast<Identifier>(base->name()))
    init->unqualifiedId = NameIdAST::create(pool, id);
  init->symbol = baseClass;
  init->constructor = constructor;

  List<ExpressionAST*>* args = nullptr;
  auto argsTail = &args;
  for (auto param :
       views::members(fn->functionParameters()) | views::parameters) {
    auto argExpr = makeParamRef(param);
    if (!binder.traits.is_reference(param->type())) {
      auto load = ImplicitCastExpressionAST::create(pool);
      load->castKind = ImplicitCastKind::kLValueToRValueConversion;
      load->expression = argExpr;
      load->type = binder.traits.remove_cv(argExpr->type);
      load->valueCategory = ValueCategory::kPrValue;
      argExpr = load;
    }
    *argsTail = make_list_node<ExpressionAST>(pool, argExpr);
    argsTail = &(*argsTail)->next;
  }
  init->expressionList = args;

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->memInitializerList = make_list_node<MemInitializerAST>(pool, init);
  body->statement = CompoundStatementAST::create(pool);
  def->functionBody = body;

  TypeChecker check{binder.unit_};
  check.setScope(fn);
  check.setReportErrors(false);
  check.check_mem_initializers(body);
}

auto Binder::CompleteClass::buildRecordLayout()
    -> std::expected<bool, std::string> {
  return binder.buildRecordLayout(classSymbol);
}

void Binder::CompleteClass::complete(bool deferExceptionSpecificationChecks) {
  classSymbol->setHasUserDeclaredConstructors(
      !classSymbol->declaredConstructors().empty());

  if (defersClassSemanticCompletion(binder.unit_, classSymbol)) {
    markComplete();
    return;
  }

  if (shouldSynthesizeSpecialMembers()) synthesizeSpecialMembers();

  auto status = buildRecordLayout();
  if (!status.has_value())
    binder.error(classSymbol->location(), status.error());

  binder.computeClassFlags(classSymbol);

  typeFieldInitializers();

  binder.refreshImplicitExceptionSpecifications(classSymbol);

  if (!deferExceptionSpecificationChecks)
    checkOverriderExceptionSpecifications();

  if (shouldSynthesizeSpecialMembers()) {
    synthesizeStructorVariants();
    synthesizeMemberwiseBodies();
  }

  markComplete();
}

void Binder::applyImplicitExceptionSpecification(FunctionSymbol* fn) {
  if (!fn || fn->hasExceptionSpecifier()) return;
  if (!fn->isDestructor() && !fn->isDefaulted()) return;

  auto funcType = type_cast<FunctionType>(fn->type());
  if (!funcType) return;

  auto classSymbol = symbol_cast<ClassSymbol>(fn->parent());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();

  const bool isMoveForm = fn == classSymbol->moveConstructor() ||
                          fn == classSymbol->moveAssignmentOperator();

  const bool isAssignment = fn == classSymbol->copyAssignmentOperator() ||
                            fn == classSymbol->moveAssignmentOperator();

  const bool isCopyOrMoveConstructor = fn == classSymbol->copyConstructor() ||
                                       fn == classSymbol->moveConstructor();

  const bool isDefaultConstructor = fn == classSymbol->defaultConstructor();

  auto operatorId = name_cast<OperatorId>(fn->name());
  const bool isDefaultedComparison =
      operatorId && (operatorId->op() == TokenKind::T_EQUAL_EQUAL ||
                     operatorId->op() == TokenKind::T_LESS_EQUAL_GREATER ||
                     operatorId->op() == TokenKind::T_EXCLAIM_EQUAL ||
                     operatorId->op() == TokenKind::T_LESS ||
                     operatorId->op() == TokenKind::T_LESS_EQUAL ||
                     operatorId->op() == TokenKind::T_GREATER ||
                     operatorId->op() == TokenKind::T_GREATER_EQUAL);

  auto inherited = fn->inheritedConstructor();
  auto inheritedBase =
      inherited ? symbol_cast<ClassSymbol>(inherited->parent()) : nullptr;
  if (inheritedBase) inheritedBase = inheritedBase->resolvedDefinition();

  auto sourceType = [&](const Type* subobjectType) -> const Type* {
    if (isMoveForm) return control()->getRvalueReferenceType(subobjectType);
    return control()->getLvalueReferenceType(
        control()->getConstType(subobjectType));
  };

  auto initializationIsPotentiallyThrowing = [&](const Type* type,
                                                 FieldSymbol* field) {
    if (isDefaultedComparison) {
      auto left = ThisExpressionAST::create(unit_->arena(),
                                            ValueCategory::kLValue, type);
      auto right = ThisExpressionAST::create(unit_->arena(),
                                             ValueCategory::kLValue, type);
      auto comparison = BinaryExpressionAST::create(unit_->arena());
      comparison->leftExpression = left;
      comparison->rightExpression = right;
      comparison->op = operatorId->op();
      comparison->opLoc = fn->location();

      TypeChecker check{unit_};
      check.setScope(classSymbol);
      check.setReportErrors(false);
      check.check(comparison);
      return !comparison->type ||
             TypeChecker::isPotentiallyThrowing(comparison);
    }

    if (fn->isDestructor()) {
      return traits.is_destructible(type) &&
             !traits.is_nothrow_destructible(type);
    }

    if (isDefaultConstructor && field && field->initializer())
      return TypeChecker::isPotentiallyThrowing(field->initializer());

    auto subobjectType = traits.remove_all_extents(type);
    if (traits.is_reference(subobjectType)) return false;

    auto classType = type_cast<ClassType>(traits.remove_cv(subobjectType));
    if (!classType) return false;

    if (inheritedBase && classType->symbol() &&
        classType->symbol()->resolvedDefinition() == inheritedBase) {
      auto inheritedType = type_cast<FunctionType>(inherited->type());
      return !inheritedType || !inheritedType->isNoexcept();
    }

    if (isAssignment) {
      return !traits.is_nothrow_assignable(
          control()->getLvalueReferenceType(subobjectType),
          sourceType(subobjectType));
    }

    if (isCopyOrMoveConstructor) {
      const Type* argumentTypes[] = {sourceType(subobjectType)};
      return !traits.is_nothrow_constructible(subobjectType, argumentTypes);
    }

    return !traits.is_nothrow_constructible(subobjectType, {});
  };

  auto isPotentiallyThrowing = [&] {
    for (auto base : classSymbol->baseClasses()) {
      if (initializationIsPotentiallyThrowing(base->symbol()->type(), nullptr))
        return true;
    }

    if (auto layout = classSymbol->layout()) {
      for (auto base : layout->virtualBases()) {
        if (initializationIsPotentiallyThrowing(base->type(), nullptr))
          return true;
      }
    }

    for (auto field : views::members(classSymbol) | views::non_static_fields) {
      if (initializationIsPotentiallyThrowing(field->type(), field))
        return true;
    }

    return false;
  };

  setFunctionNoexcept(control(), fn, !isPotentiallyThrowing());
}

void Binder::refreshImplicitExceptionSpecifications(ClassSymbol* classSymbol) {
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();

  for (auto constructor : classSymbol->declaredConstructors())
    applyImplicitExceptionSpecification(constructor);

  for (auto member : classSymbol->members()) {
    for (auto func : views::each_function(member))
      applyImplicitExceptionSpecification(func);
  }
}

void Binder::finalizeExceptionSpecifications(ClassSymbol* classSymbol) {
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  if (defersClassSemanticCompletion(unit_, classSymbol)) return;
  refreshImplicitExceptionSpecifications(classSymbol);
  CompleteClass{*this, classSymbol}.checkOverriderExceptionSpecifications();
}

void Binder::CompleteClass::checkOverriderExceptionSpecifications() {
  for (auto member : classSymbol->members()) {
    for (auto func : views::each_function(member)) {
      if (!func->isVirtual() || func->isDeleted()) continue;

      ASTRewriter::completePendingExceptionSpecification(binder.unit_, func);

      auto overriderType = type_cast<FunctionType>(func->type());
      if (!overriderType || overriderType->isNoexcept()) continue;

      for (auto overridden :
           binder.findOverriddenFunctions(classSymbol, func)) {
        ASTRewriter::completePendingExceptionSpecification(binder.unit_,
                                                           overridden);

        auto overriddenType = type_cast<FunctionType>(overridden->type());
        if (!overriddenType || !overriddenType->isNoexcept()) continue;

        binder.error(func->location(),
                     std::format("exception specification of overriding "
                                 "function '{}' is more lax than the function "
                                 "it overrides",
                                 to_string(func->name())));
        binder.note(overridden->location(),
                    "overridden virtual function is here");
      }
    }
  }
}

auto Binder::CompleteClass::newDefaultedFunction(const Name* name,
                                                 const Type* type)
    -> FunctionSymbol* {
  auto symbol =
      control()->newFunctionSymbol(classSymbol, classSymbol->location());
  symbol->setName(name);
  symbol->setType(type);
  symbol->setDefined(true);
  symbol->setDefaulted(true);
  symbol->setInline(true);
  symbol->setLanguageLinkage(LanguageKind::kCXX);
  return symbol;
}

void Binder::CompleteClass::attachDeclaration(FunctionSymbol* symbol,
                                              UnqualifiedIdAST* id) {
  auto idDecl = IdDeclaratorAST::create(pool);
  idDecl->unqualifiedId = id;

  auto funcChunk = FunctionDeclaratorChunkAST::create(pool);

  auto declarator = DeclaratorAST::create(
      pool, nullptr, idDecl,
      make_list_node<DeclaratorChunkAST>(pool, funcChunk));

  auto funcDef = FunctionDefinitionAST::create(pool);
  funcDef->declarator = declarator;
  funcDef->functionBody = DefaultFunctionBodyAST::create(pool);
  funcDef->symbol = symbol;
  symbol->setDeclaration(funcDef);
}

auto Binder::CompleteClass::makeCtorNameId() -> NameIdAST* {
  return NameIdAST::create(pool, name_cast<Identifier>(classSymbol->name()));
}

void Binder::CompleteClass::addFunctionToClassScope(FunctionSymbol* symbol) {
  binder.overloadSetFor(classSymbol, symbol->name(), symbol->location())
      ->addFunction(symbol);
}

auto Binder::CompleteClass::defaultConstructorIsDeleted() const -> bool {
  auto traits = binder.traits;

  auto subobjectIsNotDefaultConstructible = [&](const Type* type) {
    auto classType = type_cast<ClassType>(traits.remove_cv(type));
    if (!classType || !classType->symbol()) return false;
    auto subobject = classType->symbol()->resolvedDefinition();
    if (!subobject->isComplete()) return false;

    if (auto destructor = subobject->destructor();
        destructor && destructor->isDeleted())
      return true;

    if (subobject->constructors().empty()) return false;

    auto defaultConstructor = subobject->defaultConstructor();
    return !defaultConstructor || defaultConstructor->isDeleted();
  };

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass && subobjectIsNotDefaultConstructible(baseClass->type()))
      return true;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->initializer()) continue;

    auto type = field->type();
    if (traits.is_reference(type)) return true;

    auto element = traits.remove_all_extents(type);
    if (traits.is_const(element) && !traits.is_class(traits.remove_cv(element)))
      return true;

    if (subobjectIsNotDefaultConstructible(element)) return true;
  }

  return false;
}

void Binder::CompleteClass::addDefaultConstructor() {
  if (!classSymbol->declaredConstructors().empty()) return;

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {}));
  if (defaultConstructorIsDeleted()) symbol->setDeleted(true);
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addCopyConstructor() {
  if (classSymbol->copyConstructor()) return;

  auto constRefType = control()->getLvalueReferenceType(
      control()->getConstType(classSymbol->type()));

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {constRefType}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

void Binder::CompleteClass::addMoveConstructor() {
  if (classSymbol->moveConstructor()) return;

  auto rvalRefType = control()->getRvalueReferenceType(classSymbol->type());

  auto symbol = newDefaultedFunction(
      classSymbol->name(),
      control()->getFunctionType(control()->getVoidType(), {rvalRefType}));
  classSymbol->addConstructor(symbol);
  attachDeclaration(symbol, makeCtorNameId());
}

auto Binder::CompleteClass::hasUserDeclaredAssignmentOperator(
    bool moveForm) const -> bool {
  auto traits = binder.traits;
  return views::any_function(
      classSymbol->find(TokenKind::T_EQUAL), [&](FunctionSymbol* fn) {
        auto funcType = type_cast<FunctionType>(fn->type());
        if (!funcType) return false;
        auto& params = funcType->parameterTypes();
        if (params.size() != 1) return false;

        auto paramType = params[0];
        if (auto lref = type_cast<LvalueReferenceType>(paramType)) {
          if (moveForm) return false;
          paramType = lref->elementType();
        } else if (auto rref = type_cast<RvalueReferenceType>(paramType)) {
          if (!moveForm) return false;
          paramType = rref->elementType();
        } else if (moveForm) {
          return false;
        }

        auto classType = type_cast<ClassType>(traits.remove_cv(paramType));
        return classType && classType->symbol() == classSymbol;
      });
}

void Binder::CompleteClass::addCopyAssignmentOperator() {
  if (hasUserDeclaredAssignmentOperator(/*moveForm=*/false)) return;

  auto constRefType = control()->getLvalueReferenceType(
      control()->getConstType(classSymbol->type()));
  auto retType = control()->getLvalueReferenceType(classSymbol->type());

  auto symbol =
      newDefaultedFunction(control()->getOperatorId(TokenKind::T_EQUAL),
                           control()->getFunctionType(retType, {constRefType}));
  addFunctionToClassScope(symbol);
  attachDeclaration(symbol,
                    OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL));
}

void Binder::CompleteClass::addMoveAssignmentOperator() {
  if (hasUserDeclaredAssignmentOperator(/*moveForm=*/true)) return;

  auto rvalRefType = control()->getRvalueReferenceType(classSymbol->type());
  auto retType = control()->getLvalueReferenceType(classSymbol->type());

  auto symbol =
      newDefaultedFunction(control()->getOperatorId(TokenKind::T_EQUAL),
                           control()->getFunctionType(retType, {rvalRefType}));
  addFunctionToClassScope(symbol);
  attachDeclaration(symbol,
                    OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL));
}

auto Binder::CompleteClass::hasNonCopyConstructibleSubobject(
    bool moveForm) const -> bool {
  auto traits = binder.traits;

  auto subobjectIsNotCopyConstructible = [&](const Type* type) {
    auto classType = type_cast<ClassType>(traits.remove_cv(type));
    if (!classType || !classType->symbol()) return false;
    auto subobject = classType->symbol()->resolvedDefinition();
    if (!subobject->isComplete()) return false;

    if (auto destructor = subobject->destructor();
        destructor && destructor->isDeleted())
      return true;

    auto constructor = moveForm ? subobject->moveConstructor() : nullptr;
    if (!constructor) constructor = subobject->copyConstructor();
    return constructor && constructor->isDeleted();
  };

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass && subobjectIsNotCopyConstructible(baseClass->type()))
      return true;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    auto type = field->type();
    if (!moveForm && type_cast<RvalueReferenceType>(type)) return true;

    if (subobjectIsNotCopyConstructible(traits.remove_all_extents(type)))
      return true;
  }

  return false;
}

auto Binder::CompleteClass::hasNonAssignableSubobject(bool moveForm) const
    -> bool {
  auto traits = binder.traits;

  auto subobjectAssignmentIsDeleted = [&](const Type* type) {
    auto classType = type_cast<ClassType>(traits.remove_cv(type));
    if (!classType || !classType->symbol()) return false;
    auto subobject = classType->symbol()->resolvedDefinition();
    if (!subobject->isComplete()) return false;

    auto assignment = moveForm ? subobject->moveAssignmentOperator() : nullptr;
    if (!assignment) assignment = subobject->copyAssignmentOperator();
    return assignment && assignment->isDeleted();
  };

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass && subobjectAssignmentIsDeleted(baseClass->type()))
      return true;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    auto type = field->type();
    if (traits.is_reference(type)) return true;

    auto element = traits.remove_all_extents(type);
    if (traits.is_const(element) && !traits.is_class(traits.remove_cv(element)))
      return true;

    if (subobjectAssignmentIsDeleted(element)) return true;
  }

  return false;
}

void Binder::CompleteClass::addDestructor() {
  if (classSymbol->destructor()) return;

  auto symbol = newDefaultedFunction(
      control()->getDestructorId(classSymbol->name()),
      control()->getFunctionType(control()->getVoidType(), {}));

  auto overriddenFunctions =
      binder.findOverriddenFunctions(classSymbol, symbol);
  if (!overriddenFunctions.empty()) {
    symbol->setVirtual(true);
    for (auto overridden : overriddenFunctions)
      symbol->addOverriddenFunction(overridden);
  }

  classSymbol->addSymbol(symbol);

  auto dtorId = DestructorIdAST::create(pool);
  if (auto id = name_cast<Identifier>(classSymbol->name()))
    dtorId->id = NameIdAST::create(pool, id);
  attachDeclaration(symbol, dtorId);
}

void Binder::CompleteClass::synthesizeStructorVariants() {
  auto layout = classSymbol->layout();
  if (!layout) return;

  const bool hasVirtualBases = !layout->virtualBases().empty();

  if (hasVirtualBases) {
    for (auto ctor : classSymbol->declaredConstructors()) {
      if (ctor->completeObjectVariant()) continue;
      if (ctor->isDeleted()) continue;
      if (ctor->templateDeclaration()) continue;
      if (!type_cast<FunctionType>(ctor->type())) continue;
      synthesizeCompleteObjectCtor(ctor);
    }
  }

  if (auto dtor = classSymbol->destructor(); dtor && !dtor->isDeleted()) {
    if (hasVirtualBases && !dtor->completeObjectVariant())
      synthesizeCompleteObjectDtor(dtor);
    if (dtor->isVirtual() && !dtor->deletingDtorVariant())
      synthesizeDeletingDtor(dtor);
  }
}

auto Binder::CompleteClass::newStructorVariant(FunctionSymbol* principal)
    -> FunctionSymbol* {
  auto variant =
      control()->newFunctionSymbol(classSymbol, principal->location());
  variant->setName(principal->name());
  variant->setType(principal->type());
  variant->setDefined(true);
  variant->setLanguageLinkage(LanguageKind::kCXX);
  variant->setStructorPrincipal(principal);

  auto params = control()->newFunctionParametersSymbol(variant, {});
  variant->addSymbol(params);

  if (auto funcType = type_cast<FunctionType>(principal->type())) {
    int index = 0;
    for (auto paramType : funcType->parameterTypes()) {
      auto param = control()->newParameterSymbol(params, principal->location());
      param->setName(control()->getIdentifier(std::format("__p{}", index++)));
      param->setType(paramType);
      params->addSymbol(param);
    }
  }

  return variant;
}

void Binder::CompleteClass::attachVariantDefinition(FunctionSymbol* variant,
                                                    UnqualifiedIdAST* id,
                                                    FunctionBodyAST* body) {
  auto idDecl = IdDeclaratorAST::create(pool);
  idDecl->unqualifiedId = id;

  auto funcChunk = FunctionDeclaratorChunkAST::create(pool);

  auto declarator = DeclaratorAST::create(
      pool, nullptr, idDecl,
      make_list_node<DeclaratorChunkAST>(pool, funcChunk));

  auto funcDef = FunctionDefinitionAST::create(pool);
  funcDef->declarator = declarator;
  funcDef->functionBody = body;
  funcDef->symbol = variant;
  variant->setDeclaration(funcDef);
}

auto Binder::CompleteClass::makeThisExpr() -> ExpressionAST* {
  auto thisExpr = ThisExpressionAST::create(pool);
  thisExpr->type = control()->getPointerType(classSymbol->type());
  thisExpr->valueCategory = ValueCategory::kPrValue;
  return thisExpr;
}

auto Binder::CompleteClass::makeParamRef(ParameterSymbol* param)
    -> ExpressionAST* {
  auto idExpr = IdExpressionAST::create(pool);
  if (auto id = name_cast<Identifier>(param->name()))
    idExpr->unqualifiedId = NameIdAST::create(pool, id);
  idExpr->symbol = param;
  idExpr->type = binder.traits.remove_reference(param->type());
  idExpr->valueCategory = ValueCategory::kLValue;
  return idExpr;
}

auto Binder::CompleteClass::makeQualifier(ClassSymbol* cls)
    -> NestedNameSpecifierAST* {
  auto nns = SimpleNestedNameSpecifierAST::create(pool);
  nns->identifier = name_cast<Identifier>(cls->name());
  return nns;
}

auto Binder::CompleteClass::makeStructorCallStatement(FunctionSymbol* callee,
                                                      ExpressionAST* objectPtr)
    -> StatementAST* {
  auto calleeClass = symbol_cast<ClassSymbol>(callee->parent());

  auto member = MemberExpressionAST::create(pool);
  member->baseExpression = objectPtr;
  member->accessOp = TokenKind::T_MINUS_GREATER;
  member->nestedNameSpecifier =
      calleeClass ? makeQualifier(calleeClass) : nullptr;
  if (name_cast<DestructorId>(callee->name())) {
    auto dtorId = DestructorIdAST::create(pool);
    if (calleeClass) {
      if (auto id = name_cast<Identifier>(calleeClass->name()))
        dtorId->id = NameIdAST::create(pool, id);
    }
    member->unqualifiedId = dtorId;
  } else if (auto id = name_cast<Identifier>(callee->name())) {
    member->unqualifiedId = NameIdAST::create(pool, id);
  }
  member->symbol = callee;
  member->type = callee->type();
  member->valueCategory = ValueCategory::kPrValue;

  auto call = CallExpressionAST::create(pool);
  call->baseExpression = member;
  call->type = control()->getVoidType();
  call->valueCategory = ValueCategory::kPrValue;

  auto stmt = ExpressionStatementAST::create(pool);
  stmt->expression = call;
  return stmt;
}

auto Binder::CompleteClass::pickVBaseConstructor(ClassSymbol* vbase,
                                                 bool isCopy, bool isMove)
    -> FunctionSymbol* {
  if (isCopy) return vbase->copyConstructor();
  if (isMove) return vbase->moveConstructor();
  return vbase->defaultConstructor();
}

void Binder::CompleteClass::synthesizeCompleteObjectCtor(FunctionSymbol* ctor) {
  auto layout = classSymbol->layout();
  auto traits = binder.traits;

  bool isCopy = false;
  bool isMove = false;
  ParameterSymbol* sourceParam = nullptr;

  auto variant = newStructorVariant(ctor);

  auto range =
      views::members(variant->functionParameters()) | views::parameters;

  std::vector params(begin(range), end(range));

  if (params.size() == 1) {
    auto paramType = params[0]->type();
    if (auto ref = type_cast<LvalueReferenceType>(paramType)) {
      if (traits.remove_cv(ref->elementType()) == classSymbol->type()) {
        isCopy = true;
        sourceParam = params[0];
      }
    } else if (auto rref = type_cast<RvalueReferenceType>(paramType)) {
      if (traits.remove_cv(rref->elementType()) == classSymbol->type()) {
        isMove = true;
        sourceParam = params[0];
      }
    }
  }

  List<MemInitializerAST*>* memInits = nullptr;
  auto memInitsTail = &memInits;

  for (auto vbase : layout->virtualBases()) {
    auto vbaseCtor = pickVBaseConstructor(vbase, isCopy, isMove);

    auto init = ParenMemInitializerAST::create(pool);
    if (auto id = name_cast<Identifier>(vbase->name()))
      init->unqualifiedId = NameIdAST::create(pool, id);
    init->symbol = vbase;
    init->constructor = vbaseCtor;

    if (sourceParam) {
      auto cast = ImplicitCastExpressionAST::create(pool);
      cast->castKind = ImplicitCastKind::kDerivedToBaseConversion;
      cast->expression = makeParamRef(sourceParam);
      cast->type =
          isCopy ? control()->getConstType(vbase->type()) : vbase->type();
      cast->valueCategory = ValueCategory::kLValue;
      init->expressionList = make_list_node<ExpressionAST>(pool, cast);
    } else if (vbaseCtor) {
      TypeChecker check{binder.unit_};
      check.setScope(ctor);
      check.append_default_arguments(vbaseCtor, &init->expressionList);
    }

    *memInitsTail = make_list_node<MemInitializerAST>(pool, init);
    memInitsTail = &(*memInitsTail)->next;
  }

  auto delegate = ParenMemInitializerAST::create(pool);
  delegate->unqualifiedId = makeCtorNameId();
  delegate->symbol = classSymbol;
  delegate->constructor = ctor;

  List<ExpressionAST*>* args = nullptr;
  auto argsTail = &args;
  for (auto param : params) {
    auto argExpr = makeParamRef(param);
    if (!traits.is_reference(param->type())) {
      auto load = ImplicitCastExpressionAST::create(pool);
      load->castKind = ImplicitCastKind::kLValueToRValueConversion;
      load->expression = argExpr;
      load->type = traits.remove_cv(argExpr->type);
      load->valueCategory = ValueCategory::kPrValue;
      argExpr = load;
    }
    *argsTail = make_list_node<ExpressionAST>(pool, argExpr);
    argsTail = &(*argsTail)->next;
  }
  delegate->expressionList = args;

  *memInitsTail = make_list_node<MemInitializerAST>(pool, delegate);

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->memInitializerList = memInits;
  body->statement = CompoundStatementAST::create(pool);

  attachVariantDefinition(variant, makeCtorNameId(), body);
  ctor->setCompleteObjectVariant(variant);
}

void Binder::CompleteClass::synthesizeCompleteObjectDtor(FunctionSymbol* dtor) {
  auto layout = classSymbol->layout();

  auto variant = newStructorVariant(dtor);
  variant->setVirtual(dtor->isVirtual());

  List<StatementAST*>* stmts = nullptr;
  auto stmtsTail = &stmts;
  auto appendStatement = [&](StatementAST* stmt) {
    *stmtsTail = make_list_node<StatementAST>(pool, stmt);
    stmtsTail = &(*stmtsTail)->next;
  };

  appendStatement(makeStructorCallStatement(dtor, makeThisExpr()));

  const auto& vbases = layout->virtualBases();
  for (auto it = vbases.rbegin(); it != vbases.rend(); ++it) {
    auto vbase = *it;
    auto vbaseDtor = vbase->destructor();
    if (!vbaseDtor) continue;

    auto cast = ImplicitCastExpressionAST::create(pool);
    cast->castKind = ImplicitCastKind::kDerivedToBaseConversion;
    cast->expression = makeThisExpr();
    cast->type = control()->getPointerType(vbase->type());
    cast->valueCategory = ValueCategory::kPrValue;

    appendStatement(makeStructorCallStatement(vbaseDtor, cast));
  }

  auto compound = CompoundStatementAST::create(pool);
  compound->statementList = stmts;

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compound;

  auto dtorId = DestructorIdAST::create(pool);
  if (auto id = name_cast<Identifier>(classSymbol->name()))
    dtorId->id = NameIdAST::create(pool, id);

  attachVariantDefinition(variant, dtorId, body);
  dtor->setCompleteObjectVariant(variant);
}

void Binder::CompleteClass::synthesizeDeletingDtor(FunctionSymbol* dtor) {
  auto variant = newStructorVariant(dtor);
  variant->setVirtual(true);

  List<StatementAST*>* stmts = nullptr;
  auto stmtsTail = &stmts;
  auto appendStatement = [&](StatementAST* stmt) {
    *stmtsTail = make_list_node<StatementAST>(pool, stmt);
    stmtsTail = &(*stmtsTail)->next;
  };

  auto completeDtor = dtor->completeObjectVariant();
  if (!completeDtor) completeDtor = dtor;
  appendStatement(makeStructorCallStatement(completeDtor, makeThisExpr()));

  auto operatorDelete =
      resolveUsualOperatorDelete(binder.unit_, classSymbol, false);

  auto calleeExpr = IdExpressionAST::create(pool);
  calleeExpr->unqualifiedId =
      OperatorFunctionIdAST::create(pool, TokenKind::T_DELETE);
  calleeExpr->symbol = operatorDelete;
  calleeExpr->type = operatorDelete->type();
  calleeExpr->valueCategory = ValueCategory::kLValue;

  auto voidPtrType = control()->getPointerType(control()->getVoidType());
  auto thisAsVoidPtr = ImplicitCastExpressionAST::create(pool);
  thisAsVoidPtr->castKind = ImplicitCastKind::kPointerConversion;
  thisAsVoidPtr->expression = makeThisExpr();
  thisAsVoidPtr->type = voidPtrType;
  thisAsVoidPtr->valueCategory = ValueCategory::kPrValue;

  auto call = CallExpressionAST::create(pool);
  call->baseExpression = calleeExpr;
  call->expressionList = make_list_node<ExpressionAST>(pool, thisAsVoidPtr);
  call->type = control()->getVoidType();
  call->valueCategory = ValueCategory::kPrValue;

  auto stmt = ExpressionStatementAST::create(pool);
  stmt->expression = call;
  appendStatement(stmt);

  auto compound = CompoundStatementAST::create(pool);
  compound->statementList = stmts;

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compound;

  auto dtorId = DestructorIdAST::create(pool);
  if (auto id = name_cast<Identifier>(classSymbol->name()))
    dtorId->id = NameIdAST::create(pool, id);

  attachVariantDefinition(variant, dtorId, body);
  dtor->setDeletingDtorVariant(variant);
}

void Binder::CompleteClass::synthesizeMemberwiseBodies() {
  if (classSymbol->isUnion()) return;

  auto needsBody = [&](FunctionSymbol* fn) {
    if (!fn || fn->isDeleted()) return false;
    auto def = fn->declaration();
    return def && ast_cast<DefaultFunctionBodyAST>(def->functionBody);
  };

  for (auto fn : classSymbol->declaredConstructors()) {
    if (fn->inheritedConstructor() && needsBody(fn))
      synthesizeInheritedConstructorBody(fn);
  }

  if (auto fn = classSymbol->copyConstructor(); needsBody(fn))
    synthesizeCopyMoveCtorBody(fn, /*isMove=*/false);
  if (auto fn = classSymbol->moveConstructor(); needsBody(fn))
    synthesizeCopyMoveCtorBody(fn, /*isMove=*/true);
  if (auto fn = classSymbol->copyAssignmentOperator(); needsBody(fn))
    synthesizeCopyMoveAssignBody(fn, /*isMove=*/false);
  if (auto fn = classSymbol->moveAssignmentOperator(); needsBody(fn))
    synthesizeCopyMoveAssignBody(fn, /*isMove=*/true);
}

void Binder::CompleteClass::typeFieldInitializers() {
  for (auto field : views::members(classSymbol) | views::fields) {
    auto init = field->initializer();
    if (!init) continue;

    TypeChecker check{binder.unit_};
    check.setScope(classSymbol);
    check.setReportErrors(binder.reportErrors());
    check.check_field_initializer(field);
  }
}

auto Binder::CompleteClass::ensureSourceParameter(FunctionSymbol* fn)
    -> ParameterSymbol* {
  if (auto params = fn->functionParameters()) {
    for (auto member : views::members(params)) {
      if (auto param = symbol_cast<ParameterSymbol>(member)) return param;
    }
  }

  auto funcType = type_cast<FunctionType>(fn->type());
  if (!funcType || funcType->parameterTypes().size() != 1) return nullptr;

  auto params = control()->newFunctionParametersSymbol(fn, {});
  fn->addSymbol(params);

  auto param = control()->newParameterSymbol(params, fn->location());
  param->setType(funcType->parameterTypes()[0]);
  params->addSymbol(param);
  return param;
}

auto Binder::CompleteClass::makeSourceSubobjectRef(ExpressionAST* expr,
                                                   const Type* type,
                                                   bool isMove)
    -> ExpressionAST* {
  if (!isMove) return expr;
  auto cast = ImplicitCastExpressionAST::create(pool);
  cast->castKind = ImplicitCastKind::kIdentity;
  cast->expression = expr;
  cast->type = type;
  cast->valueCategory = ValueCategory::kXValue;
  return cast;
}

void Binder::CompleteClass::synthesizeCopyMoveCtorBody(FunctionSymbol* fn,
                                                       bool isMove) {
  auto def = fn->declaration();
  if (!def || !ast_cast<DefaultFunctionBodyAST>(def->functionBody)) return;

  auto param = ensureSourceParameter(fn);
  if (!param) return;

  auto traits = binder.traits;

  List<MemInitializerAST*>* memInits = nullptr;
  auto tail = &memInits;
  auto append = [&](MemInitializerAST* init) {
    *tail = make_list_node<MemInitializerAST>(pool, init);
    tail = &(*tail)->next;
  };

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseSym) continue;
    baseSym = baseSym->resolvedDefinition();

    auto init = ParenMemInitializerAST::create(pool);
    if (auto id = name_cast<Identifier>(baseSym->name()))
      init->unqualifiedId = NameIdAST::create(pool, id);
    init->symbol = base;

    auto cast = ImplicitCastExpressionAST::create(pool);
    cast->castKind = ImplicitCastKind::kDerivedToBaseConversion;
    cast->expression = makeParamRef(param);
    cast->type =
        isMove ? baseSym->type() : control()->getConstType(baseSym->type());
    cast->valueCategory =
        isMove ? ValueCategory::kXValue : ValueCategory::kLValue;
    init->expressionList = make_list_node<ExpressionAST>(pool, cast);
    append(init);
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!field->name() && field->isBitField()) continue;

    auto id = name_cast<Identifier>(field->name());

    auto init = ParenMemInitializerAST::create(pool);
    if (id) init->unqualifiedId = NameIdAST::create(pool, id);
    init->symbol = field;

    auto access = MemberExpressionAST::create(pool);
    access->baseExpression = makeParamRef(param);
    access->accessOp = TokenKind::T_DOT;
    if (id) access->unqualifiedId = NameIdAST::create(pool, id);
    access->symbol = field;
    access->type = traits.remove_reference(field->type());
    access->valueCategory = ValueCategory::kLValue;

    ExpressionAST* arg = access;
    const bool elementwiseCopy =
        !id || traits.is_array(traits.remove_cv(field->type()));
    if (elementwiseCopy) {
      auto load = ImplicitCastExpressionAST::create(pool);
      load->castKind = ImplicitCastKind::kLValueToRValueConversion;
      load->expression = access;
      load->type = traits.remove_cv(access->type);
      load->valueCategory = ValueCategory::kPrValue;
      arg = load;
    } else {
      arg = makeSourceSubobjectRef(access, access->type, isMove);
    }
    init->expressionList = make_list_node<ExpressionAST>(pool, arg);
    append(init);
  }

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->memInitializerList = memInits;
  body->statement = CompoundStatementAST::create(pool);
  def->functionBody = body;

  TypeChecker check{binder.unit_};
  check.setScope(fn);
  check.setReportErrors(false);
  check.check_mem_initializers(body);
}

void Binder::CompleteClass::synthesizeCopyMoveAssignBody(FunctionSymbol* fn,
                                                         bool isMove) {
  auto def = fn->declaration();
  if (!def || !ast_cast<DefaultFunctionBodyAST>(def->functionBody)) return;

  auto param = ensureSourceParameter(fn);
  if (!param) return;

  auto traits = binder.traits;

  TypeChecker check{binder.unit_};
  check.setScope(fn);

  List<StatementAST*>* stmts = nullptr;
  auto tail = &stmts;
  auto append = [&](StatementAST* stmt) {
    *tail = make_list_node<StatementAST>(pool, stmt);
    tail = &(*tail)->next;
  };

  auto appendAssignment = [&](ExpressionAST* lhs, ExpressionAST* rhs,
                              bool resolve) {
    auto assign = AssignmentExpressionAST::create(pool);
    assign->leftExpression = lhs;
    assign->op = TokenKind::T_EQUAL;
    assign->rightExpression = rhs;
    if (resolve) {
      check.check(assign);
    } else {
      assign->type = lhs->type;
      assign->valueCategory = ValueCategory::kLValue;
    }
    auto stmt = ExpressionStatementAST::create(pool);
    stmt->expression = assign;
    append(stmt);
  };

  for (auto base : classSymbol->baseClasses()) {
    auto baseSym = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseSym) continue;
    baseSym = baseSym->resolvedDefinition();

    auto thisCast = ImplicitCastExpressionAST::create(pool);
    thisCast->castKind = ImplicitCastKind::kDerivedToBaseConversion;
    thisCast->expression = makeThisExpr();
    thisCast->type = control()->getPointerType(baseSym->type());
    thisCast->valueCategory = ValueCategory::kPrValue;

    auto lhs = UnaryExpressionAST::create(pool);
    lhs->op = TokenKind::T_STAR;
    lhs->expression = thisCast;
    lhs->type = baseSym->type();
    lhs->valueCategory = ValueCategory::kLValue;

    auto rhs = ImplicitCastExpressionAST::create(pool);
    rhs->castKind = ImplicitCastKind::kDerivedToBaseConversion;
    rhs->expression = makeParamRef(param);
    rhs->type =
        isMove ? baseSym->type() : control()->getConstType(baseSym->type());
    rhs->valueCategory =
        isMove ? ValueCategory::kXValue : ValueCategory::kLValue;

    appendAssignment(lhs, rhs, /*resolve=*/true);
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!field->name() && field->isBitField()) continue;

    auto id = name_cast<Identifier>(field->name());
    auto fieldType = traits.remove_reference(field->type());

    auto lhs = MemberExpressionAST::create(pool);
    lhs->baseExpression = makeThisExpr();
    lhs->accessOp = TokenKind::T_MINUS_GREATER;
    if (id) lhs->unqualifiedId = NameIdAST::create(pool, id);
    lhs->symbol = field;
    lhs->type = fieldType;
    lhs->valueCategory = ValueCategory::kLValue;

    auto access = MemberExpressionAST::create(pool);
    access->baseExpression = makeParamRef(param);
    access->accessOp = TokenKind::T_DOT;
    if (id) access->unqualifiedId = NameIdAST::create(pool, id);
    access->symbol = field;
    access->type = isMove ? fieldType : control()->getConstType(fieldType);
    access->valueCategory = ValueCategory::kLValue;

    const bool bitwiseCopy =
        !id || traits.is_array(traits.remove_cv(fieldType));
    if (bitwiseCopy) {
      auto load = ImplicitCastExpressionAST::create(pool);
      load->castKind = ImplicitCastKind::kLValueToRValueConversion;
      load->expression = access;
      load->type = traits.remove_cv(access->type);
      load->valueCategory = ValueCategory::kPrValue;
      appendAssignment(lhs, load, /*resolve=*/false);
    } else {
      appendAssignment(lhs,
                       makeSourceSubobjectRef(access, access->type, isMove),
                       /*resolve=*/true);
    }
  }

  auto self = UnaryExpressionAST::create(pool);
  self->op = TokenKind::T_STAR;
  self->expression = makeThisExpr();
  self->type = classSymbol->type();
  self->valueCategory = ValueCategory::kLValue;

  auto returnStmt = ReturnStatementAST::create(pool);
  returnStmt->expression = self;
  append(returnStmt);

  auto compound = CompoundStatementAST::create(pool);
  compound->statementList = stmts;

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compound;
  def->functionBody = body;
}

struct [[nodiscard]] Binder::BuildRecordLayout {
  Binder& binder;
  ClassSymbol* classSymbol;
  const MemoryLayout* memoryLayout;
  std::unique_ptr<ClassLayout> layout;

  int calculatedSize = 0;
  int calculatedAlignment = 1;
  std::uint32_t currentIndex = 0;

  int nextBitPos = 0;
  int runStartByte = 0;
  std::uint32_t runIndex = 0;
  bool inBitfieldRun = false;
  std::vector<FieldSymbol*> runFields;
  std::vector<std::pair<ClassSymbol*, std::uint64_t>> placedClassSubobjects;

  int packValue = 0;

  BuildRecordLayout(Binder& b, ClassSymbol* cls)
      : binder(b),
        classSymbol(cls),
        memoryLayout(b.control()->memoryLayout()),
        layout(std::make_unique<ClassLayout>()) {
    if (auto loc = cls->location()) {
      const auto& tok = b.unit_->tokenAt(loc);
      packValue =
          b.unit_->preprocessor()->packValueAt(tok.fileId(), tok.offset());
    }
  }

  auto control() const -> Control* { return binder.control(); }

  auto operator()() -> std::expected<bool, std::string>;
  auto validate() -> std::expected<bool, std::string>;
  [[nodiscard]] auto computeAbiEmpty() const -> bool;
  [[nodiscard]] auto isNearlyEmptyClass(ClassSymbol* classSymbol) const -> bool;
  [[nodiscard]] auto selectPrimaryBase() const -> std::pair<ClassSymbol*, bool>;
  void layoutVtable();
  void layoutBases();
  void layoutVirtualBases();
  [[nodiscard]] auto classSubobjectOffset(ClassSymbol* classSymbol,
                                          bool tryZero, std::uint64_t alignment)
      -> std::uint64_t;
  [[nodiscard]] auto nonVirtualClassSubobjects(ClassSymbol* classSymbol)
      -> std::vector<std::pair<ClassSymbol*, std::uint64_t>>;
  void recordNonVirtualClassSubobjects(ClassSymbol* classSymbol,
                                       std::uint64_t offset);
  auto layoutFields() -> std::expected<bool, std::string>;
  auto layoutBitfield(FieldSymbol* field) -> std::expected<bool, std::string>;
  auto layoutRegularField(FieldSymbol* field)
      -> std::expected<bool, std::string>;
  void closeBitfieldRun();
  void propagateBaseFields();
  void propagateAnonymousFields(ClassSymbol* cls, std::uint64_t baseOffset);
  void finalize();
  void buildVTableLayout();
};

auto Binder::buildRecordLayout(ClassSymbol* classSymbol)
    -> std::expected<bool, std::string> {
  return BuildRecordLayout{*this, classSymbol}();
}

auto Binder::BuildRecordLayout::operator()()
    -> std::expected<bool, std::string> {
  if (auto status = validate(); !status) return status;

  layout->setAbiEmpty(computeAbiEmpty());
  layoutVtable();
  layoutBases();

  auto fieldsStatus = layoutFields();
  if (!fieldsStatus) return fieldsStatus;
  if (!fieldsStatus.value()) return false;

  layout->setNonVirtualSize(calculatedSize);
  layout->setNonVirtualAlignment(calculatedAlignment);

  layoutVirtualBases();

  propagateBaseFields();
  finalize();

  return true;
}

auto Binder::BuildRecordLayout::validate() -> std::expected<bool, std::string> {
  for (auto base : classSymbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) {
      return std::unexpected(
          std::format("base class '{}' not found", to_string(base->name())));
    }
    if (!baseClassSymbol->isComplete()) {
      binder.traits.requireCompleteClass(baseClassSymbol);
    }
    baseClassSymbol = baseClassSymbol->resolvedDefinition();
    if (!baseClassSymbol->isComplete()) {
      return std::unexpected(std::format("base class '{}' is incomplete",
                                         to_string(baseClassSymbol->name())));
    }
  }
  return true;
}

auto Binder::BuildRecordLayout::computeAbiEmpty() const -> bool {
  if (classSymbol->isUnion()) return false;
  if (views::any_function(classSymbol->members(),
                          [](FunctionSymbol* f) { return f->isVirtual(); }))
    return false;

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->isNoUniqueAddress()) {
      auto fieldClass =
          type_cast<ClassType>(binder.traits.remove_cv(field->type()));
      auto fieldSymbol = fieldClass ? fieldClass->symbol() : nullptr;
      auto fieldLayout =
          fieldSymbol ? fieldSymbol->resolvedDefinition()->layout() : nullptr;
      if (fieldLayout && fieldLayout->isAbiEmpty()) continue;
    }
    if (field->isBitField() && !field->name()) {
      auto width = field->bitFieldWidth();
      auto value = width ? std::get_if<std::intmax_t>(&*width) : nullptr;
      if (value && *value == 0) continue;
    }
    return false;
  }

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) return false;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) return false;
    auto baseLayout = baseClass->resolvedDefinition()->layout();
    if (!baseLayout || !baseLayout->isAbiEmpty()) return false;
  }
  return true;
}

void Binder::BuildRecordLayout::layoutVtable() {
  if (classSymbol->isUnion()) return;

  const auto hasVirtualFunction = views::any_function(
      classSymbol->members(), [](FunctionSymbol* f) { return f->isVirtual(); });
  const auto hasDynamicBase = std::ranges::any_of(
      classSymbol->baseClasses(), [](BaseClassSymbol* base) {
        auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
        if (baseClass) baseClass = baseClass->resolvedDefinition();
        return baseClass && baseClass->layout() &&
               baseClass->layout()->hasVtable();
      });
  const auto hasVirtualBase = std::ranges::any_of(
      classSymbol->baseClasses(),
      [](BaseClassSymbol* base) { return base->isVirtual(); });
  if (!hasVirtualFunction && !hasDynamicBase && !hasVirtualBase) return;

  layout->setHasVtable(true);
  auto [primaryBase, primaryIsVirtual] = selectPrimaryBase();
  if (primaryBase) {
    layout->setPrimaryBase(primaryBase, primaryIsVirtual);
    if (!primaryIsVirtual) return;

    ClassLayout::MemberInfo primaryInfo;
    primaryInfo.index = currentIndex++;
    layout->setBaseInfo(primaryBase, primaryInfo);
    layout->setHasDirectVtable(true);
    layout->setVtableIndex(primaryInfo.index);
    recordNonVirtualClassSubobjects(primaryBase, 0);
  } else {
    layout->setHasDirectVtable(true);
    layout->setVtableIndex(currentIndex++);
  }

  auto ptrSize = static_cast<int>(memoryLayout->sizeOfPointer());
  calculatedSize = ptrSize;
  calculatedAlignment = ptrSize;
  nextBitPos = calculatedSize * 8;
}

auto Binder::BuildRecordLayout::isNearlyEmptyClass(ClassSymbol* candidate) const
    -> bool {
  if (!candidate) return false;
  candidate = candidate->resolvedDefinition();
  auto candidateLayout = candidate->layout();
  if (!candidateLayout || !candidateLayout->hasVtable()) return false;

  for (auto field : views::members(candidate) | views::non_static_fields) {
    if (field->isNoUniqueAddress()) {
      auto fieldClass =
          type_cast<ClassType>(binder.traits.remove_cv(field->type()));
      auto fieldSymbol = fieldClass ? fieldClass->symbol() : nullptr;
      auto fieldLayout =
          fieldSymbol ? fieldSymbol->resolvedDefinition()->layout() : nullptr;
      if (fieldLayout && fieldLayout->isAbiEmpty()) continue;
    }
    if (field->isBitField() && !field->name()) {
      auto width = field->bitFieldWidth();
      auto value = width ? std::get_if<std::intmax_t>(&*width) : nullptr;
      if (value && *value == 0) continue;
    }
    return false;
  }

  int nearlyEmptyNonVirtualBases = 0;
  for (auto base : candidate->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) return false;
    baseClass = baseClass->resolvedDefinition();
    auto baseLayout = baseClass->layout();
    if (baseLayout && baseLayout->isAbiEmpty()) continue;
    if (!isNearlyEmptyClass(baseClass)) return false;
    if (++nearlyEmptyNonVirtualBases > 1) return false;
  }

  std::vector<std::pair<ClassSymbol*, std::uint64_t>> pendingBases{
      {candidate, 0}};
  while (!pendingBases.empty()) {
    auto [cls, offset] = pendingBases.back();
    pendingBases.pop_back();
    auto classLayout = cls->layout();
    if (!classLayout) continue;
    for (auto base : cls->baseClasses()) {
      if (base->isVirtual()) continue;
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      baseClass = baseClass->resolvedDefinition();
      auto info = classLayout->getBaseInfo(baseClass);
      if (!info) continue;
      const auto baseOffset = offset + info->offset;
      auto baseLayout = baseClass->layout();
      if (baseOffset != 0 && baseLayout && baseLayout->isAbiEmpty())
        return false;
      pendingBases.emplace_back(baseClass, baseOffset);
    }
  }
  return true;
}

auto Binder::BuildRecordLayout::selectPrimaryBase() const
    -> std::pair<ClassSymbol*, bool> {
  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass) baseClass = baseClass->resolvedDefinition();
    if (baseClass && baseClass->layout() && baseClass->layout()->hasVtable())
      return {baseClass, false};
  }

  std::unordered_set<ClassSymbol*> indirectPrimaryBases;
  std::vector<ClassSymbol*> pendingClasses{classSymbol};
  while (!pendingClasses.empty()) {
    auto cls = pendingClasses.back();
    pendingClasses.pop_back();
    for (auto base : cls->baseClasses()) {
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      baseClass = baseClass->resolvedDefinition();
      if (auto baseLayout = baseClass->layout();
          baseLayout && baseLayout->primaryBaseIsVirtual())
        indirectPrimaryBases.insert(baseLayout->primaryBase());
      pendingClasses.push_back(baseClass);
    }
  }

  std::vector<ClassSymbol*> candidates;
  std::unordered_set<ClassSymbol*> seenVirtualBases;
  struct InheritanceFrame {
    ClassSymbol* classSymbol;
    std::size_t nextBase = 0;
  };
  std::vector<InheritanceFrame> frames{{classSymbol}};
  while (!frames.empty()) {
    auto& frame = frames.back();
    auto& bases = frame.classSymbol->baseClasses();
    if (frame.nextBase == bases.size()) {
      frames.pop_back();
      continue;
    }
    auto base = bases[frame.nextBase++];
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    if (base->isVirtual()) {
      if (!seenVirtualBases.insert(baseClass).second) continue;
      if (isNearlyEmptyClass(baseClass)) candidates.push_back(baseClass);
    }
    frames.push_back({baseClass});
  }

  auto candidate = std::ranges::find_if(candidates, [&](ClassSymbol* base) {
    return !indirectPrimaryBases.contains(base);
  });
  if (candidate != candidates.end()) return {*candidate, true};
  if (!candidates.empty()) return {candidates.front(), true};
  return {nullptr, false};
}

auto Binder::BuildRecordLayout::classSubobjectOffset(ClassSymbol* target,
                                                     bool tryZero,
                                                     std::uint64_t alignment)
    -> std::uint64_t {
  auto subobjects = nonVirtualClassSubobjects(target);

  auto conflicts = [&](std::uint64_t offset) {
    return std::ranges::any_of(subobjects, [&](const auto& candidate) {
      return std::ranges::any_of(
          placedClassSubobjects, [&](const auto& placed) {
            return placed.first == candidate.first &&
                   placed.second == offset + candidate.second;
          });
    });
  };

  if (tryZero && !conflicts(0)) return 0;

  auto offset = align_to(calculatedSize, alignment);
  while (conflicts(offset)) offset += alignment;
  return offset;
}

auto Binder::BuildRecordLayout::nonVirtualClassSubobjects(ClassSymbol* target)
    -> std::vector<std::pair<ClassSymbol*, std::uint64_t>> {
  std::vector<std::pair<ClassSymbol*, std::uint64_t>> subobjects;
  std::vector<std::pair<ClassSymbol*, std::uint64_t>> pending{{target, 0}};
  while (!pending.empty()) {
    auto [cls, offset] = pending.back();
    pending.pop_back();
    cls = cls->resolvedDefinition();
    subobjects.emplace_back(cls, offset);

    auto classLayout = cls->layout();
    if (!classLayout) continue;

    for (auto base : cls->baseClasses()) {
      if (base->isVirtual()) continue;
      auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClass) continue;
      baseClass = baseClass->resolvedDefinition();
      auto baseInfo = classLayout->getBaseInfo(baseClass);
      if (!baseInfo) continue;
      pending.emplace_back(baseClass, offset + baseInfo->offset);
    }
  }
  return subobjects;
}

void Binder::BuildRecordLayout::recordNonVirtualClassSubobjects(
    ClassSymbol* target, std::uint64_t offset) {
  for (auto& [classSymbol, relativeOffset] : nonVirtualClassSubobjects(target))
    placedClassSubobjects.emplace_back(classSymbol, offset + relativeOffset);
}

void Binder::BuildRecordLayout::layoutBases() {
  if (classSymbol->isUnion()) return;

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;
    baseClassSymbol = baseClassSymbol->resolvedDefinition();

    auto baseLayout = baseClassSymbol->layout();

    const bool baseHasVirtualBases =
        baseLayout && !baseLayout->virtualBases().empty();
    int baseSizeInBytes = baseHasVirtualBases
                              ? static_cast<int>(baseLayout->nonVirtualSize())
                              : baseClassSymbol->sizeInBytes();
    int baseAlignment =
        baseHasVirtualBases
            ? static_cast<int>(baseLayout->nonVirtualAlignment())
            : static_cast<int>(baseClassSymbol->alignment());

    const bool isEmpty = baseLayout && baseLayout->isAbiEmpty();
    const auto baseOffset = classSubobjectOffset(baseClassSymbol, isEmpty,
                                                 std::max(baseAlignment, 1));

    ClassLayout::MemberInfo baseInfo;
    baseInfo.offset = baseOffset;
    baseInfo.index = currentIndex++;
    layout->setBaseInfo(baseClassSymbol, baseInfo);

    if (baseClassSymbol == layout->primaryBase() &&
        !layout->primaryBaseIsVirtual()) {
      layout->setVtableIndex(baseInfo.index);
    }

    if (!isEmpty)
      calculatedSize = std::max(calculatedSize,
                                static_cast<int>(baseOffset) + baseSizeInBytes);
    recordNonVirtualClassSubobjects(baseClassSymbol, baseOffset);
    calculatedAlignment = std::max(calculatedAlignment, baseAlignment);
  }

  nextBitPos = calculatedSize * 8;
}

void Binder::BuildRecordLayout::layoutVirtualBases() {
  if (classSymbol->isUnion()) return;

  std::vector<ClassSymbol*> orderedVirtualBases;
  std::unordered_set<ClassSymbol*> seenVirtualBases;

  struct InheritanceFrame {
    ClassSymbol* classSymbol;
    std::size_t nextBase = 0;
  };
  std::vector<InheritanceFrame> frames{{classSymbol}};
  while (!frames.empty()) {
    auto& frame = frames.back();
    auto& bases = frame.classSymbol->baseClasses();
    if (frame.nextBase == bases.size()) {
      frames.pop_back();
      continue;
    }
    auto base = bases[frame.nextBase++];
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    if (base->isVirtual()) {
      if (!seenVirtualBases.insert(baseClass).second) continue;
      orderedVirtualBases.push_back(baseClass);
    }
    frames.push_back({baseClass});
  }

  std::unordered_map<ClassSymbol*, ClassLayout::MemberInfo>
      indirectPrimaryPlacements;
  const auto recordPrimaryChain = [&](ClassSymbol* cls, std::uint64_t offset,
                                      std::uint32_t topIndex) {
    while (cls) {
      auto classLayout = cls->layout();
      if (!classLayout || !classLayout->primaryBase()) break;
      auto primary = classLayout->primaryBase();
      auto info = classLayout->getBaseInfo(primary);
      if (!info) break;
      const auto primaryOffset = offset + info->offset;
      if (classLayout->primaryBaseIsVirtual())
        indirectPrimaryPlacements.try_emplace(
            primary, ClassLayout::MemberInfo{primaryOffset, topIndex});
      cls = primary;
      offset = primaryOffset;
    }
  };
  struct PlacementWork {
    ClassSymbol* classSymbol;
    std::uint64_t offset;
    std::uint32_t topIndex;
  };
  const auto collectIndirectPrimaryPlacements = [&](ClassSymbol* root,
                                                    std::uint64_t rootOffset,
                                                    std::uint32_t rootIndex) {
    std::vector<PlacementWork> pending{{root, rootOffset, rootIndex}};
    while (!pending.empty()) {
      auto [cls, offset, topIndex] = pending.back();
      pending.pop_back();
      auto classLayout = cls->layout();
      if (!classLayout) continue;
      recordPrimaryChain(cls, offset, topIndex);
      auto& bases = cls->baseClasses();
      for (auto it = bases.rbegin(); it != bases.rend(); ++it) {
        auto base = *it;
        if (base->isVirtual()) continue;
        auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
        if (!baseClass) continue;
        baseClass = baseClass->resolvedDefinition();
        auto info = classLayout->getBaseInfo(baseClass);
        if (info)
          pending.push_back({baseClass, offset + info->offset, topIndex});
      }
    }
  };

  if (layout->primaryBaseIsVirtual()) {
    auto primary = layout->primaryBase();
    auto info = layout->getBaseInfo(primary);
    if (info)
      indirectPrimaryPlacements.try_emplace(
          primary, ClassLayout::MemberInfo{info->offset, info->index});
    if (info) recordPrimaryChain(primary, info->offset, info->index);
  }
  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();
    auto info = layout->getBaseInfo(baseClass);
    if (info)
      collectIndirectPrimaryPlacements(baseClass, info->offset, info->index);
  }

  for (auto baseClassSymbol : orderedVirtualBases) {
    if (auto placement = indirectPrimaryPlacements.find(baseClassSymbol);
        placement != indirectPrimaryPlacements.end()) {
      layout->setBaseInfo(baseClassSymbol, placement->second);
      layout->addVirtualBase(baseClassSymbol);
      recordNonVirtualClassSubobjects(baseClassSymbol,
                                      placement->second.offset);
      recordPrimaryChain(baseClassSymbol, placement->second.offset,
                         placement->second.index);
      continue;
    }

    auto vbaseLayout = baseClassSymbol->layout();

    const bool vbaseHasVirtualBases =
        vbaseLayout && !vbaseLayout->virtualBases().empty();
    int baseSizeInBytes = vbaseHasVirtualBases
                              ? static_cast<int>(vbaseLayout->nonVirtualSize())
                              : baseClassSymbol->sizeInBytes();
    int baseAlignment =
        vbaseHasVirtualBases
            ? static_cast<int>(vbaseLayout->nonVirtualAlignment())
            : static_cast<int>(baseClassSymbol->alignment());

    const bool isEmpty = vbaseLayout && vbaseLayout->isAbiEmpty();
    const auto baseOffset = classSubobjectOffset(baseClassSymbol, isEmpty,
                                                 std::max(baseAlignment, 1));

    ClassLayout::MemberInfo baseInfo;
    baseInfo.offset = baseOffset;
    baseInfo.index = currentIndex++;
    layout->setBaseInfo(baseClassSymbol, baseInfo);
    layout->addVirtualBase(baseClassSymbol);

    if (!isEmpty)
      calculatedSize = std::max(calculatedSize,
                                static_cast<int>(baseOffset) + baseSizeInBytes);
    recordNonVirtualClassSubobjects(baseClassSymbol, baseOffset);
    recordPrimaryChain(baseClassSymbol, baseOffset, baseInfo.index);
    calculatedAlignment = std::max(calculatedAlignment, baseAlignment);
  }

  nextBitPos = calculatedSize * 8;
}

void Binder::BuildRecordLayout::closeBitfieldRun() {
  if (!inBitfieldRun) return;

  calculatedSize = (nextBitPos + 7) / 8;

  auto allocUnitSizeBytes =
      static_cast<std::uint32_t>(calculatedSize - runStartByte);

  for (auto f : runFields) {
    if (auto info = layout->getFieldInfo(f)) {
      auto updated = *info;
      updated.allocUnitSizeBytes = allocUnitSizeBytes;
      layout->setFieldInfo(f, updated);
    }
  }

  runFields.clear();
  inBitfieldRun = false;
  currentIndex++;
}

auto Binder::BuildRecordLayout::layoutBitfield(FieldSymbol* field)
    -> std::expected<bool, std::string> {
  const bool isUnion = classSymbol->isUnion();

  int bitWidth = 0;
  if (auto& bfw = field->bitFieldWidth()) {
    if (auto iv = std::get_if<std::intmax_t>(&*bfw)) {
      bitWidth = static_cast<int>(*iv);
    }
  }

  auto fieldAlign = field->alignment();
  if (packValue > 0) fieldAlign = std::min(fieldAlign, packValue);
  auto fieldSizeBytes =
      static_cast<int>(memoryLayout->sizeOf(field->type()).value_or(0));
  auto fieldSizeBits = fieldSizeBytes * 8;

  if (bitWidth == 0) {
    if (inBitfieldRun) {
      closeBitfieldRun();
    }
    if (!isUnion && fieldSizeBits > 0) {
      auto alignBits = fieldAlign * 8;
      nextBitPos = align_to(nextBitPos, alignBits);
      calculatedSize = (nextBitPos + 7) / 8;
    }
    return true;
  }

  if (isUnion) {
    field->setLocalOffset(0);
    field->setBitFieldOffset(0);

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = 0;
    fieldInfo.index = 0;
    fieldInfo.bitOffset = 0;
    fieldInfo.bitWidth = bitWidth;
    fieldInfo.allocUnitSizeBytes = (bitWidth + 7) / 8;
    layout->setFieldInfo(field, fieldInfo);

    auto fieldSizeForUnion = std::max(fieldSizeBytes, (bitWidth + 7) / 8);
    calculatedSize = std::max(calculatedSize, fieldSizeForUnion);
    calculatedAlignment = std::max(calculatedAlignment, fieldAlign);
    return true;
  }

  if (fieldSizeBits > 0) {
    auto startUnit = nextBitPos / fieldSizeBits;
    auto endUnit = (nextBitPos + bitWidth - 1) / fieldSizeBits;
    if (startUnit != endUnit) {
      if (inBitfieldRun) {
        closeBitfieldRun();
      }
      nextBitPos = align_to(nextBitPos, fieldSizeBits);
      calculatedSize = nextBitPos / 8;
    }
  }

  if (!inBitfieldRun) {
    runStartByte = calculatedSize;
    nextBitPos = runStartByte * 8;
    runIndex = currentIndex;
    inBitfieldRun = true;
    runFields.clear();
  }

  auto bitOffsetInRun = nextBitPos - runStartByte * 8;

  field->setLocalOffset(runStartByte);
  field->setBitFieldOffset(bitOffsetInRun);

  ClassLayout::MemberInfo fieldInfo;
  fieldInfo.offset = runStartByte;
  fieldInfo.index = runIndex;
  fieldInfo.bitOffset = bitOffsetInRun;
  fieldInfo.bitWidth = bitWidth;
  layout->setFieldInfo(field, fieldInfo);

  runFields.push_back(field);
  nextBitPos += bitWidth;
  calculatedAlignment = std::max(calculatedAlignment, fieldAlign);

  return true;
}

auto Binder::BuildRecordLayout::layoutRegularField(FieldSymbol* field)
    -> std::expected<bool, std::string> {
  const bool isUnion = classSymbol->isUnion();

  closeBitfieldRun();

  std::optional<std::size_t> size;
  if (binder.traits.is_unbounded_array(field->type())) {
    size = 0;
  } else if (field->isNoUniqueAddress()) {
    auto fieldClass =
        type_cast<ClassType>(binder.traits.remove_cv(field->type()));
    auto fieldSymbol = fieldClass ? fieldClass->symbol() : nullptr;
    auto fieldLayout =
        fieldSymbol ? fieldSymbol->resolvedDefinition()->layout() : nullptr;
    size = fieldLayout && fieldLayout->isAbiEmpty()
               ? std::optional<std::size_t>{0}
               : memoryLayout->sizeOf(field->type());
  } else {
    size = memoryLayout->sizeOf(field->type());
  }

  if (!size.has_value()) {
    return std::unexpected(
        std::format("size of incomplete type '{}'",
                    to_string(field->type(), field->name())));
  }

  if (isUnion) {
    field->setLocalOffset(0);
    calculatedSize = std::max(calculatedSize, int(size.value()));

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = 0;
    fieldInfo.index = 0;
    layout->setFieldInfo(field, fieldInfo);
  } else {
    auto fieldAlign = field->alignment();
    if (packValue > 0) fieldAlign = std::min(fieldAlign, packValue);
    auto fieldOffset =
        static_cast<std::uint64_t>(align_to(calculatedSize, fieldAlign));
    auto fieldClass =
        type_cast<ClassType>(binder.traits.remove_cv(field->type()));
    if (fieldClass && fieldClass->symbol()) {
      const bool tryZero = field->isNoUniqueAddress() && size.value() == 0;
      fieldOffset = classSubobjectOffset(fieldClass->symbol(), tryZero,
                                         std::max(fieldAlign, 1));
    }
    field->setLocalOffset(static_cast<int>(fieldOffset));

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = fieldOffset;
    fieldInfo.index = currentIndex++;
    layout->setFieldInfo(field, fieldInfo);

    calculatedSize =
        std::max(calculatedSize, static_cast<int>(fieldOffset + size.value()));
    if (fieldClass && fieldClass->symbol())
      recordNonVirtualClassSubobjects(fieldClass->symbol(), fieldOffset);
  }

  nextBitPos = calculatedSize * 8;

  auto cappedAlign = packValue > 0 ? std::min(field->alignment(), packValue)
                                   : field->alignment();
  calculatedAlignment = std::max(calculatedAlignment, cappedAlign);
  return true;
}

auto Binder::BuildRecordLayout::layoutFields()
    -> std::expected<bool, std::string> {
  FieldSymbol* lastField = nullptr;

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    auto fieldElementType =
        binder.traits.remove_cv(binder.traits.remove_all_extents(
            binder.traits.remove_cv(field->type())));

    if (auto classType = type_cast<ClassType>(fieldElementType)) {
      binder.traits.requireCompleteClass(classType->symbol());
      if (!field->alignment()) {
        if (auto alignment =
                binder.control()->memoryLayout()->alignmentOf(field->type())) {
          field->setAlignment(alignment.value());
        }
      }
    }

    if (lastField && binder.traits.is_unbounded_array(lastField->type())) {
      return std::unexpected(
          std::format("size of incomplete type '{}'",
                      to_string(lastField->type(), lastField->name())));
    }

    if (!field->alignment()) {
      if (isDependent(binder.unit_, field->type())) return false;
      return std::unexpected(
          std::format("alignment of incomplete type '{}'",
                      to_string(field->type(), field->name())));
    }

    if (field->isBitField()) {
      if (auto status = layoutBitfield(field); !status) return status;
    } else {
      if (auto status = layoutRegularField(field); !status) return status;
    }

    lastField = field;
  }

  closeBitfieldRun();
  return true;
}

void Binder::BuildRecordLayout::propagateBaseFields() {
  for (auto base : classSymbol->baseClasses()) {
    auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClassSymbol) continue;
    baseClassSymbol = baseClassSymbol->resolvedDefinition();

    auto baseLayout = baseClassSymbol->layout();
    if (!baseLayout) continue;

    auto baseInfo = layout->getBaseInfo(baseClassSymbol);
    if (!baseInfo) continue;

    for (auto field :
         views::members(baseClassSymbol) | views::non_static_fields) {
      auto baseFieldInfo = baseLayout->getFieldInfo(field);
      if (baseFieldInfo) {
        ClassLayout::MemberInfo adjustedInfo;
        adjustedInfo.offset = baseInfo->offset + baseFieldInfo->offset;
        adjustedInfo.index = baseFieldInfo->index;
        adjustedInfo.bitOffset = baseFieldInfo->bitOffset;
        adjustedInfo.bitWidth = baseFieldInfo->bitWidth;
        adjustedInfo.allocUnitSizeBytes = baseFieldInfo->allocUnitSizeBytes;
        layout->setFieldInfo(field, adjustedInfo);
      }
    }
  }

  propagateAnonymousFields(classSymbol, 0);
}

void Binder::BuildRecordLayout::propagateAnonymousFields(
    ClassSymbol* cls, std::uint64_t baseOffset) {
  for (auto member : cls->members()) {
    auto nestedClass = symbol_cast<ClassSymbol>(member);
    if (!nestedClass) continue;
    if (nestedClass->name()) continue;
    if (!nestedClass->isComplete()) continue;

    auto nestedLayout = nestedClass->layout();
    if (!nestedLayout) continue;

    FieldSymbol* anonField = nullptr;
    for (auto m : cls->members()) {
      auto f = symbol_cast<FieldSymbol>(m);
      if (!f) continue;
      if (auto ct = type_cast<ClassType>(f->type())) {
        if (ct->symbol() == nestedClass) {
          anonField = f;
          break;
        }
      }
    }
    if (!anonField) continue;

    auto anonFieldInfo = layout->getFieldInfo(anonField);
    if (!anonFieldInfo) continue;

    std::uint64_t anonOffset = baseOffset + anonFieldInfo->offset;

    for (auto field : views::members(nestedClass) | views::non_static_fields) {
      if (auto nestedFieldInfo = nestedLayout->getFieldInfo(field)) {
        if (!field->name()) {
          auto fieldType = type_cast<ClassType>(field->type());
          if (fieldType && !fieldType->symbol()->name()) continue;
        }

        ClassLayout::MemberInfo adjustedInfo;
        adjustedInfo.offset = anonOffset + nestedFieldInfo->offset;
        adjustedInfo.index = nestedFieldInfo->index;
        adjustedInfo.bitOffset = nestedFieldInfo->bitOffset;
        adjustedInfo.bitWidth = nestedFieldInfo->bitWidth;
        adjustedInfo.allocUnitSizeBytes = nestedFieldInfo->allocUnitSizeBytes;
        layout->setFieldInfo(field, adjustedInfo);
      }
    }

    propagateAnonymousFields(nestedClass, anonOffset);
  }
}

void Binder::BuildRecordLayout::finalize() {
  calculatedSize = align_to(calculatedSize, calculatedAlignment);
  if (calculatedSize == 0) calculatedSize = 1;

  classSymbol->setAlignment(calculatedAlignment);
  classSymbol->setSizeInBytes(calculatedSize);

  layout->setSize(calculatedSize);
  layout->setAlignment(calculatedAlignment);

  classSymbol->setLayout(std::move(layout));

  buildVTableLayout();
}

void Binder::BuildRecordLayout::buildVTableLayout() {
  auto classLayout = classSymbol->layout();
  if (!classLayout || !classLayout->hasVtable()) return;

  auto vtable = std::make_unique<VTableLayout>();

  auto resolvedClass = [](Symbol* symbol) -> ClassSymbol* {
    auto classSym = symbol_cast<ClassSymbol>(symbol);
    return classSym ? classSym->resolvedDefinition() : nullptr;
  };

  auto primaryBaseOf = [](const ClassSymbol* cls) -> ClassSymbol* {
    auto classLayout = cls->layout();
    return classLayout ? classLayout->primaryBase() : nullptr;
  };

  auto primaryBase = primaryBaseOf(classSymbol);

  auto& slots = vtable->primary.slots;
  if (primaryBase && primaryBase->vtableLayout()) {
    slots = primaryBase->vtableLayout()->primary.slots;
  }
  const auto inheritedPrimarySlotCount = slots.size();

  auto processFunc = [&](FunctionSymbol* func) {
    if (!func->isVirtual()) return;
    if (!vtable->keyFunction && !func->isPure() && !func->isInline())
      vtable->keyFunction = func;
    const bool isDtor = func->isDestructor();
    bool foundOverride = false;
    for (std::size_t i = 0; i < slots.size(); ++i) {
      const bool isOverride =
          isDtor ? slots[i].kind != VTableLayout::SlotKind::kFunction
                 : func->overrides(slots[i].function);
      if (!isOverride) continue;
      slots[i].function = func;
      foundOverride = true;
      if (!isDtor) {
        func->setVtableSlotIndex(static_cast<int>(i));
        break;
      }
      if (slots[i].kind == VTableLayout::SlotKind::kCompleteDtor)
        func->setVtableSlotIndex(static_cast<int>(i));
    }
    if (!foundOverride) {
      func->setVtableSlotIndex(static_cast<int>(slots.size()));
      slots.push_back({.function = func,
                       .kind = isDtor ? VTableLayout::SlotKind::kCompleteDtor
                                      : VTableLayout::SlotKind::kFunction,
                       .introducingFunction = func});
      if (isDtor) {
        slots.push_back({.function = func,
                         .kind = VTableLayout::SlotKind::kDeletingDtor,
                         .introducingFunction = func});
      }
    }
  };

  for (auto member : classSymbol->members()) {
    for (auto func : views::each_function(member)) processFunc(func);
  }

  for (auto vbase : classLayout->virtualBases()) {
    auto info = classLayout->getBaseInfo(vbase);
    if (!info) continue;
    vtable->primary.vbaseOffsets.emplace_back(
        vbase, static_cast<std::int64_t>(info->offset));
  }
  std::ranges::reverse(vtable->primary.vbaseOffsets);

  struct Subobject {
    ClassSymbol* classSymbol;
    std::uint64_t offset;
  };

  std::vector<Subobject> subobjects{{classSymbol, 0}};
  {
    std::unordered_set<ClassSymbol*> visitedVirtualBases;
    for (std::size_t index = 0; index < subobjects.size(); ++index) {
      auto cls = subobjects[index].classSymbol;
      const auto offset = subobjects[index].offset;
      auto layout = cls->layout();
      if (!layout) continue;
      for (auto base : cls->baseClasses()) {
        auto baseSym = resolvedClass(base->symbol());
        if (!baseSym) continue;
        auto info = base->isVirtual() ? classLayout->getBaseInfo(baseSym)
                                      : layout->getBaseInfo(baseSym);
        if (!info) continue;
        if (base->isVirtual() && !visitedVirtualBases.insert(baseSym).second)
          continue;
        subobjects.push_back({baseSym, base->isVirtual()
                                           ? info->offset
                                           : offset + info->offset});
      }
    }
  }

  struct Overrider {
    FunctionSymbol* function;
    std::uint64_t classOffset;
  };

  std::unordered_set<FunctionSymbol*> reportedFinalOverriderAmbiguities;

  auto findFinalOverrider =
      [&](FunctionSymbol* baseFunc, const std::vector<Subobject>& derivedFirst,
          ClassSymbol* virtualTarget = nullptr,
          std::optional<std::uint64_t> virtualTargetOffset =
              std::nullopt) -> std::optional<Overrider> {
    std::vector<Overrider> candidates;

    for (auto& subobject : derivedFirst) {
      if (virtualTarget) {
        if (subobject.classSymbol == virtualTarget) {
          if (virtualTargetOffset && subobject.offset != *virtualTargetOffset)
            continue;
        } else if (!subobject.classSymbol->hasVirtualBasePath(virtualTarget)) {
          continue;
        }
      }
      for (auto member : subobject.classSymbol->members()) {
        for (auto func : views::each_function(member)) {
          if (!func->isVirtual()) continue;
          if (func != baseFunc && !func->overrides(baseFunc)) continue;
          candidates.push_back({func, subobject.offset});
        }
      }
    }

    std::vector<Overrider> finalOverriders;
    for (auto& candidate : candidates) {
      const auto isOverridden =
          std::ranges::any_of(candidates, [&](const Overrider& other) {
            return other.function != candidate.function &&
                   other.function->overrides(candidate.function);
          });
      if (!isOverridden) finalOverriders.push_back(candidate);
    }
    candidates = std::move(finalOverriders);

    if (candidates.size() == 1) return candidates.front();
    if (candidates.empty()) return std::nullopt;

    if (reportedFinalOverriderAmbiguities.insert(baseFunc).second) {
      binder.error(classSymbol->location(),
                   std::format("virtual function '{}' has more than one final "
                               "overrider in '{}'",
                               to_string(baseFunc->name()),
                               to_string(classSymbol->name())));
      for (auto& candidate : candidates)
        binder.note(candidate.function->location(), "final overrider is here");
    }

    return std::nullopt;
  };

  std::vector<Subobject> primarySubobjects{{classSymbol, 0}};
  std::uint64_t primaryOffset = 0;
  for (auto cls = classSymbol; cls;) {
    auto nestedLayout = cls->layout();
    if (!nestedLayout || !nestedLayout->primaryBase()) break;
    auto primary = nestedLayout->primaryBase();
    if (nestedLayout->primaryBaseIsVirtual()) {
      auto info = classLayout->getBaseInfo(primary);
      if (info) primaryOffset = info->offset;
    } else if (auto info = nestedLayout->getBaseInfo(primary)) {
      primaryOffset += info->offset;
    }
    primarySubobjects.push_back({primary, primaryOffset});
    cls = primary;
  }

  std::unordered_map<FunctionSymbol*, int> primaryVcallIndexOf;
  for (std::size_t index = 0; index < inheritedPrimarySlotCount; ++index) {
    auto& slot = slots[index];
    if (classLayout->primaryBaseIsVirtual()) {
      slot.usesVcallOffset = true;
      slot.vcallBase = classLayout->primaryBase();
    }
    if (slot.kind == VTableLayout::SlotKind::kDeletingDtor) continue;
    const auto& overriderSubobjects =
        slot.usesVcallOffset ? subobjects : primarySubobjects;
    auto virtualTarget = slot.usesVcallOffset ? slot.vcallBase : nullptr;
    std::optional<std::uint64_t> virtualTargetOffset;
    if (virtualTarget) {
      if (auto info = classLayout->getBaseInfo(virtualTarget))
        virtualTargetOffset = info->offset;
    }
    auto overrider =
        findFinalOverrider(slot.introducingFunction, overriderSubobjects,
                           virtualTarget, virtualTargetOffset);
    if (!overrider) continue;
    slot.function = overrider->function;
    if (!slot.usesVcallOffset) {
      slot.thisAdjustment = -static_cast<std::int64_t>(overrider->classOffset);
      continue;
    }
    if (!primaryVcallIndexOf.contains(overrider->function)) {
      primaryVcallIndexOf.emplace(
          overrider->function,
          static_cast<int>(vtable->primary.vcallOffsets.size()));
      vtable->primary.vcallOffsets.emplace_back(
          overrider->function,
          static_cast<std::int64_t>(overrider->classOffset));
    }
  }
  if (!vtable->primary.vcallOffsets.empty()) {
    std::ranges::reverse(vtable->primary.vcallOffsets);
    primaryVcallIndexOf.clear();
    for (std::size_t index = 0; index < vtable->primary.vcallOffsets.size();
         ++index)
      primaryVcallIndexOf.emplace(vtable->primary.vcallOffsets[index].first,
                                  static_cast<int>(index));
    for (std::size_t index = 0; index < inheritedPrimarySlotCount; ++index) {
      auto& slot = slots[index];
      auto found = primaryVcallIndexOf.find(slot.function);
      if (found != primaryVcallIndexOf.end() &&
          vtable->primary.vcallOffsets[found->second].second != 0)
        slot.vcallOffsetIndex = found->second;
    }
  }

  auto buildGroup = [&](ClassSymbol* baseSym, std::uint64_t offset,
                        bool isVirtualBase,
                        const std::vector<Subobject>& derivedFirst)
      -> std::optional<VTableLayout::Group> {
    if (!baseSym->layout() || !baseSym->layout()->hasVtable())
      return std::nullopt;

    VTableLayout::Group group;
    group.base = baseSym;
    group.offset = offset;

    if (auto baseLayout = baseSym->layout()) {
      for (auto vbase : baseLayout->virtualBases()) {
        auto vbaseInfo = classLayout->getBaseInfo(vbase);
        if (!vbaseInfo) continue;
        group.vbaseOffsets.emplace_back(
            vbase, static_cast<std::int64_t>(vbaseInfo->offset) -
                       static_cast<std::int64_t>(group.offset));
      }
      std::ranges::reverse(group.vbaseOffsets);
    }

    if (auto baseVtable = baseSym->vtableLayout()) {
      group.slots = baseVtable->primary.slots;
    }

    std::vector<bool> inheritedVcallSlots;
    inheritedVcallSlots.reserve(group.slots.size());
    for (auto& slot : group.slots) {
      inheritedVcallSlots.push_back(slot.usesVcallOffset);
      slot.thisAdjustment = 0;
      slot.vcallOffsetIndex = -1;
    }

    std::unordered_map<FunctionSymbol*, Overrider> overrideOf;
    for (std::size_t index = 0; index < group.slots.size(); ++index) {
      auto& slot = group.slots[index];
      if (slot.kind == VTableLayout::SlotKind::kDeletingDtor) continue;
      auto virtualTarget = inheritedVcallSlots[index]
                               ? slot.vcallBase
                               : (isVirtualBase ? baseSym : nullptr);
      std::optional<std::uint64_t> virtualTargetOffset;
      if (virtualTarget) {
        if (auto info = classLayout->getBaseInfo(virtualTarget))
          virtualTargetOffset = info->offset;
      }
      auto overrider =
          findFinalOverrider(slot.introducingFunction, derivedFirst,
                             virtualTarget, virtualTargetOffset);
      if (!overrider) continue;
      const auto changed = overrider->function != slot.function;
      overrideOf.emplace(slot.function, *overrider);
      if (inheritedVcallSlots[index] || (isVirtualBase && changed)) {
        slot.usesVcallOffset = true;
        if (!inheritedVcallSlots[index]) slot.vcallBase = baseSym;
        group.vcallOffsets.emplace_back(
            overrider->function,
            static_cast<std::int64_t>(overrider->classOffset) -
                static_cast<std::int64_t>(group.offset));
      }
    }
    if (isVirtualBase) std::ranges::reverse(group.vcallOffsets);

    std::unordered_map<FunctionSymbol*, int> vcallIndexOf;
    for (std::size_t i = 0; i < group.vcallOffsets.size(); ++i) {
      vcallIndexOf.emplace(group.vcallOffsets[i].first, static_cast<int>(i));
    }

    for (auto& slot : group.slots) {
      auto it = overrideOf.find(slot.function);
      if (it == overrideOf.end()) continue;
      auto overrider = it->second;

      slot.function = overrider.function;
      if (auto vcall = vcallIndexOf.find(overrider.function);
          vcall != vcallIndexOf.end() &&
          group.vcallOffsets[vcall->second].second != 0) {
        slot.vcallOffsetIndex = vcall->second;
      } else {
        slot.thisAdjustment = static_cast<std::int64_t>(group.offset) -
                              static_cast<std::int64_t>(overrider.classOffset);
      }
    }

    return group;
  };

  struct SecondaryWork {
    ClassSymbol* classSymbol;
    std::uint64_t offset;
    std::vector<Subobject> enclosing;
    bool emitGroup = false;
  };
  std::vector<SecondaryWork> pendingSecondary{
      {classSymbol, 0, {{classSymbol, 0}}}};
  while (!pendingSecondary.empty()) {
    auto work = std::move(pendingSecondary.back());
    pendingSecondary.pop_back();
    auto cls = work.classSymbol;
    auto offset = work.offset;
    auto& enclosing = work.enclosing;
    auto layout = cls->layout();
    if (!layout) continue;

    if (work.emitGroup && layout->hasVtable()) {
      if (auto group = buildGroup(cls, offset, false, enclosing))
        vtable->secondary.push_back(std::move(*group));
    }

    auto primary = primaryBaseOf(cls);

    auto& bases = cls->baseClasses();
    for (auto it = bases.rbegin(); it != bases.rend(); ++it) {
      auto base = *it;
      if (base->isVirtual()) continue;
      auto baseSym = resolvedClass(base->symbol());
      if (!baseSym) continue;
      auto info = layout->getBaseInfo(baseSym);
      if (!info) continue;

      const auto baseOffset = offset + info->offset;

      auto derivedFirst = enclosing;
      derivedFirst.push_back({baseSym, baseOffset});

      pendingSecondary.push_back(
          {baseSym, baseOffset, std::move(derivedFirst), baseSym != primary});
    }
  }

  std::unordered_set<ClassSymbol*> primaryVirtualBases;
  std::vector<ClassSymbol*> pendingPrimaryVirtualBases{classSymbol};
  while (!pendingPrimaryVirtualBases.empty()) {
    auto cls = pendingPrimaryVirtualBases.back();
    pendingPrimaryVirtualBases.pop_back();
    auto nestedLayout = cls->layout();
    if (nestedLayout && nestedLayout->primaryBaseIsVirtual())
      primaryVirtualBases.insert(nestedLayout->primaryBase());
    for (auto base : cls->baseClasses()) {
      auto baseClass = resolvedClass(base->symbol());
      if (baseClass) pendingPrimaryVirtualBases.push_back(baseClass);
    }
  }

  for (auto vbase : classLayout->virtualBases()) {
    if (primaryVirtualBases.contains(vbase)) continue;
    auto vbaseInfo = classLayout->getBaseInfo(vbase);
    if (!vbaseInfo) continue;

    if (auto group = buildGroup(vbase, vbaseInfo->offset, true, subobjects))
      vtable->secondary.push_back(std::move(*group));
  }

  classSymbol->setVTableLayout(std::move(vtable));
}
}  // namespace cxx
