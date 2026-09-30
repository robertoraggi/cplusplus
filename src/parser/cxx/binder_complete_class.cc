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
#include <cxx/ast_rewriter.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/decl.h>
#include <cxx/dependent_types.h>
#include <cxx/diagnostics_client.h>
#include <cxx/memory_layout.h>
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
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
#include <ranges>
#include <unordered_map>
#include <unordered_set>

namespace cxx {
[[nodiscard]] static auto hasUninstantiatedDefaultMemberInitializer(
    ClassSymbol* classSymbol) -> bool {
  return std::ranges::any_of(
      views::members(classSymbol) | views::non_static_fields,
      [](FieldSymbol* field) { return field->hasPendingInitializer(); });
}

[[nodiscard]] static auto inheritedConstructorSource(
    FunctionSymbol* constructor) -> FunctionSymbol* {
  auto origin = constructor->inheritedConstructorOrigin();
  return origin ? origin : constructor;
}

[[nodiscard]] static auto sameInheritedConstructorSource(FunctionSymbol* lhs,
                                                         FunctionSymbol* rhs)
    -> bool {
  return inheritedConstructorSource(lhs)->canonical() ==
         inheritedConstructorSource(rhs)->canonical();
}

[[nodiscard]] static auto declaresInheritedConstructorSource(
    ClassSymbol* base, FunctionSymbol* constructor) -> bool {
  auto source = inheritedConstructorSource(constructor);
  auto primary = source->primaryTemplateSymbol();

  for (auto candidate : base->constructors()) {
    auto candidateSource = inheritedConstructorSource(candidate);
    if (candidateSource->canonical() == source->canonical()) return true;
    if (primary && candidateSource->canonical() == primary->canonical())
      return true;
  }

  return false;
}

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

[[nodiscard]] static auto boundedArrayExtents(const Type* type)
    -> std::vector<std::size_t> {
  std::vector<std::size_t> extents;
  while (auto arrayType = unqualified_cast<BoundedArrayType>(type)) {
    extents.push_back(arrayType->size());
    type = arrayType->elementType();
  }
  return extents;
}

[[nodiscard]] static auto arrayElementCount(const Type* type) -> std::size_t {
  std::size_t count = 1;
  for (auto extent : boundedArrayExtents(type)) count *= extent;
  return count;
}

[[nodiscard]] static auto comparisonParameters(FunctionSymbol* fn)
    -> std::vector<ParameterSymbol*> {
  std::vector<ParameterSymbol*> parameters;
  if (auto parameterScope = fn->functionParameters()) {
    for (auto parameter : views::members(parameterScope) | views::parameters)
      parameters.push_back(parameter);
  }
  return parameters;
}

[[nodiscard]] static auto hasComparisonParameters(FunctionSymbol* fn) -> bool {
  const auto expected = fn->isImplicitObjectMemberFunction() ? 1u : 2u;
  return comparisonParameters(fn).size() == expected;
}

[[nodiscard]] static auto isFriendDeclaredIn(FunctionSymbol* function,
                                             ClassSymbol* classSymbol) -> bool {
  if (!function->isFriend()) return false;
  return std::ranges::contains(function->befriendingClasses(), classSymbol);
}

[[nodiscard]] static auto virtualBasesInInheritanceGraphOrder(
    ClassSymbol* classSymbol) -> std::vector<ClassSymbol*> {
  struct InheritanceFrame {
    ClassSymbol* classSymbol;
    std::size_t nextBase = 0;
  };

  std::vector<ClassSymbol*> virtualBases;
  std::vector<InheritanceFrame> frames{{classSymbol}};
  while (!frames.empty()) {
    auto& frame = frames.back();
    auto& bases = frame.classSymbol->baseClasses();
    if (frame.nextBase == bases.size()) {
      frames.pop_back();
      continue;
    }
    auto base = bases[frame.nextBase++];
    auto baseClass = resolved_base_class(base);
    if (!baseClass) continue;
    if (base->isVirtual()) {
      if (std::ranges::contains(virtualBases, baseClass)) continue;
      virtualBases.push_back(baseClass);
    }
    frames.push_back({baseClass});
  }
  return virtualBases;
}

[[nodiscard]] static auto indirectPrimaryBasesOf(ClassSymbol* classSymbol)
    -> std::vector<ClassSymbol*> {
  std::vector<ClassSymbol*> primaryBases;
  std::vector<ClassSymbol*> visited{classSymbol};

  for (std::size_t index = 0; index < visited.size(); ++index) {
    for (auto baseClass : visited[index]->baseClasses()) {
      auto base = resolved_base_class(baseClass);
      if (!base) continue;
      if (std::ranges::contains(visited, base)) continue;
      visited.push_back(base);

      auto baseLayout = base->layout();
      if (!baseLayout || !baseLayout->primaryBaseIsVirtual()) continue;
      if (std::ranges::contains(primaryBases, baseLayout->primaryBase()))
        continue;
      primaryBases.push_back(baseLayout->primaryBase());
    }
  }

  return primaryBases;
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

  void complete(DeferredMemberContexts deferred);

  void markComplete();
  auto shouldSynthesizeSpecialMembers() const -> bool;
  void synthesizeSpecialMembers();

  auto buildRecordLayout() -> std::expected<bool, std::string>;

  auto newDefaultedFunction(const Name* name, const Type* type)
      -> FunctionSymbol*;
  void attachDeclaration(FunctionSymbol* symbol, UnqualifiedIdAST* id);
  [[nodiscard]] auto makeDestructorId(ClassSymbol* cls) -> DestructorIdAST*;
  auto makeCtorNameId() -> NameIdAST*;
  void addFunctionToClassScope(FunctionSymbol* symbol);
  auto comparisonFunctions(TokenKind op) const -> std::vector<FunctionSymbol*>;
  void synthesizeImplicitEqualityOperators();
  auto declareImplicitEqualityOperator(FunctionSymbol* threeWay)
      -> FunctionSymbol*;
  void synthesizeDefaultedEqualityBodies();
  void synthesizeDefaultedEqualityBody(FunctionSymbol* fn);
  [[nodiscard]] auto hasReferenceOrVariantMembers() const -> bool;
  enum class ComparisonSide { kLeft, kRight };
  struct ComparisonElement {
    Symbol* subobject = nullptr;
    std::span<VariableSymbol* const> indices;
  };
  struct DefaultedThreeWayComparison;
  struct ElementLoops {
    std::vector<ForStatementAST*> loops;
    std::vector<VariableSymbol*> indices;
    ScopeSymbol* scope = nullptr;

    [[nodiscard]] auto enclose(StatementAST* statement) const -> StatementAST*;
  };
  [[nodiscard]] auto comparisonSubobjects(FunctionSymbol* fn)
      -> std::vector<Symbol*>;
  [[nodiscard]] auto elementLoops(TypeChecker& check, ScopeSymbol* scope,
                                  Symbol* subobject, SourceLocation location)
      -> ElementLoops;
  void appendElementLoop(TypeChecker& check, ElementLoops& nest,
                         std::size_t extent, SourceLocation location);
  [[nodiscard]] auto indexValues(std::span<VariableSymbol* const> indices)
      -> std::vector<ExpressionAST*>;
  [[nodiscard]] auto firstElementIndices(Symbol* subobject)
      -> std::vector<ExpressionAST*>;
  [[nodiscard]] auto sizeLiteral(std::size_t value) -> ExpressionAST*;
  [[nodiscard]] auto newBlock(ScopeSymbol* scope, SourceLocation location)
      -> BlockSymbol*;
  [[nodiscard]] auto compoundStatement(
      BlockSymbol* block, std::span<StatementAST* const> statements)
      -> CompoundStatementAST*;
  [[nodiscard]] auto comparisonOperand(FunctionSymbol* fn, ComparisonSide side,
                                       Symbol* subobject,
                                       std::span<ExpressionAST* const> indices)
      -> ExpressionAST*;
  [[nodiscard]] auto returnFalseIfUnequal(TypeChecker& check,
                                          FunctionSymbol* fn,
                                          const ComparisonElement& element)
      -> IfStatementAST*;
  [[nodiscard]] auto boolReturn(TypeChecker& check, bool value)
      -> ReturnStatementAST*;
  [[nodiscard]] auto comparisonObject(FunctionSymbol* fn, ComparisonSide side)
      -> ExpressionAST*;
  [[nodiscard]] auto implicitComparisonObject(FunctionSymbol* fn)
      -> ExpressionAST*;
  [[nodiscard]] auto baseSubobject(ExpressionAST* object, ClassSymbol* base)
      -> ExpressionAST*;
  [[nodiscard]] auto memberSubobject(ExpressionAST* object, FieldSymbol* field)
      -> ExpressionAST*;
  [[nodiscard]] auto arrayElement(ExpressionAST* array, ExpressionAST* index)
      -> ExpressionAST*;
  [[nodiscard]] auto comparisonCategoryType(WellKnownName name) const
      -> const Type*;
  [[nodiscard]] auto checkUsable(TypeChecker& check, ExpressionAST*& expression)
      -> bool;
  [[nodiscard]] auto threeWayCategoryRank(
      TypeChecker& check, FunctionSymbol* fn, Symbol* subobject,
      std::span<const Type* const> categories) -> std::optional<int>;
  void deduceDefaultedThreeWayReturnTypes();
  void synthesizeDefaultedThreeWayBody(FunctionSymbol* fn);
  void synthesizeDefaultedSecondaryComparisonBodies();
  void synthesizeDefaultedSecondaryComparisonBody(FunctionSymbol* fn,
                                                  TokenKind op);
  [[nodiscard]] auto secondaryComparisonReturn(FunctionSymbol* fn, TokenKind op,
                                               BlockSymbol* block)
      -> ReturnStatementAST*;
  auto hasUserDeclaredAssignmentOperator(bool moveForm) const -> bool;
  void addDefaultConstructor();
  [[nodiscard]] auto isSubobjectMemberUsable(FunctionSymbol* function,
                                             ClassSymbol* subobject) const
      -> bool;

  [[nodiscard]] auto defaultConstructorIsDeleted() const -> bool;
  [[nodiscard]] auto declaresNonConstDefaultConstructibleConstMember(
      FieldSymbol* field) const -> bool;
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
  auto makeThisExpr() -> ExpressionAST*;
  [[nodiscard]] auto makeSelfExpr() -> ExpressionAST*;
  [[nodiscard]] auto copiedMemberSource(FieldSymbol* field,
                                        ParameterSymbol* source, bool isMove)
      -> ExpressionAST*;
  [[nodiscard]] auto assignsArrayElementwise(FieldSymbol* field,
                                             bool isMove) const -> bool;
  [[nodiscard]] auto arrayMemberAssignment(TypeChecker& check,
                                           FunctionSymbol* fn,
                                           FieldSymbol* field,
                                           ParameterSymbol* source, bool isMove)
      -> StatementAST*;
  [[nodiscard]] auto baseAssignmentStatement(TypeChecker& check,
                                             ClassSymbol* base,
                                             ParameterSymbol* source,
                                             bool isMove) -> StatementAST*;
  auto makeParamRef(ParameterSymbol* param) -> ExpressionAST*;
  auto makeForwardedParamRef(ParameterSymbol* param) -> ExpressionAST*;
  auto makeQualifier(ClassSymbol* cls) -> NestedNameSpecifierAST*;
  [[nodiscard]] auto typeSpecifier(const Type* type) -> SpecifierAST*;
  [[nodiscard]] auto declarationStatement(VariableSymbol* variable)
      -> StatementAST*;
  [[nodiscard]] auto variableReference(VariableSymbol* variable)
      -> IdExpressionAST*;
  [[nodiscard]] auto variableValue(VariableSymbol* variable) -> ExpressionAST*;
  auto makeStructorCallStatement(FunctionSymbol* callee,
                                 ExpressionAST* objectPtr) -> StatementAST*;
  auto pickVBaseConstructor(ClassSymbol* vbase, bool isCopy, bool isMove)
      -> FunctionSymbol*;
  void synthesizeCompleteObjectCtor(FunctionSymbol* ctor);
  void synthesizeDelegatingCompleteObjectCtor(FunctionSymbol* ctor);
  void synthesizeCompleteObjectDtor(FunctionSymbol* dtor);
  void synthesizeDeletingDtor(FunctionSymbol* dtor);

  void synthesizeMemberwiseBodies();
  void typeFieldInitializers();
  void checkOverriderExceptionSpecifications();
  [[nodiscard]] auto hasNonAssignableSubobject(bool moveForm) const -> bool;
  [[nodiscard]] auto subobjectAssignmentIsUnusable(const Type* type,
                                                   bool moveForm) const -> bool;
  [[nodiscard]] auto hasNonCopyConstructibleSubobject(bool moveForm) const
      -> bool;
  auto ensureSourceParameter(FunctionSymbol* fn) -> ParameterSymbol*;
  auto makeSourceSubobjectRef(ExpressionAST* expr, const Type* type,
                              bool isMove) -> ExpressionAST*;
  void synthesizeDefaultConstructorBody(FunctionSymbol* fn);
  void synthesizeCopyMoveCtorBody(FunctionSymbol* fn, bool isMove);
  void synthesizeCopyMoveAssignBody(FunctionSymbol* fn, bool isMove);
};

void Binder::completeForMemberContexts(ClassSymbol* classSymbol) {
  if (!classSymbol) return;
  if (classSymbol->isComplete()) return;
  if (defersClassSemanticCompletion(unit_, classSymbol)) return;

  auto status = buildRecordLayout(classSymbol);
  if (!status.has_value()) return;

  classSymbol->setComplete(true);
}

void Binder::complete(ClassSpecifierAST* ast, DeferredMemberContexts deferred) {
  CompleteClass{*this, ast}.complete(deferred);
}

void Binder::completeFieldInitializers(ClassSymbol* classSymbol) {
  CompleteClass{*this, classSymbol}.typeFieldInitializers();
}

void Binder::completeClosureType(ClassSymbol* classSymbol) {
  CompleteClass{*this, classSymbol}.complete({});
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
  if (!ctor || !ctor->isConstructor()) return;
  if (ctor->isDeleted()) return;
  if (!type_cast<FunctionType>(ctor->type())) return;
  if (ctor->completeObjectVariant() && !ctor->delegatingConstructor()) return;

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
  if (!classSymbol && fn->isFriend()) {
    auto parameters = comparisonParameters(fn);
    if (parameters.empty()) return;
    auto classType =
        unqualified_cast<ClassType>(traits.remove_cvref(parameters[0]->type()));
    if (!classType) return;
    classSymbol = classType->symbol();
    if (!isFriendDeclaredIn(fn, classSymbol)) return;
  }
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  if (classSymbol->isUnion()) return;

  auto canon = fn->canonical();
  auto matches = [&](FunctionSymbol* member) {
    return member && member->canonical() == canon;
  };

  CompleteClass cc{*this, classSymbol};
  if (auto operatorId = name_cast<OperatorId>(fn->name())) {
    switch (operatorId->op()) {
      case TokenKind::T_EQUAL_EQUAL:
        cc.synthesizeDefaultedEqualityBody(fn);
        return;
      case TokenKind::T_LESS_EQUAL_GREATER:
        cc.deduceDefaultedThreeWayReturnTypes();
        cc.synthesizeDefaultedThreeWayBody(fn);
        return;
      case TokenKind::T_EXCLAIM_EQUAL:
      case TokenKind::T_LESS:
      case TokenKind::T_LESS_EQUAL:
      case TokenKind::T_GREATER:
      case TokenKind::T_GREATER_EQUAL:
        cc.synthesizeDefaultedSecondaryComparisonBody(fn, operatorId->op());
        return;
      default:
        break;
    }
  }
  if (matches(classSymbol->defaultConstructor()))
    cc.synthesizeDefaultConstructorBody(fn);
  else if (matches(classSymbol->copyConstructor()))
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
  return binder.isCxx();
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

  for (auto existing : classSymbol->declaredConstructors()) {
    if (!existing->inheritedConstructor()) continue;
    if (sameInheritedConstructorSource(existing, inherited)) return existing;
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
      binder.inheritDefaultArgument(param, sourceParameters[position]);
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
  auto source = inheritedConstructorSource(inherited);

  auto declaringClass = source->enclosingClass();
  if (declaringClass) declaringClass = declaringClass->resolvedDefinition();

  for (auto baseClass : classSymbol->baseClasses()) {
    auto base = resolved_base_class(baseClass);
    if (!base) continue;

    if (base == declaringClass) return {baseClass, source};

    for (auto constructor : base->declaredConstructors()) {
      if (sameInheritedConstructorSource(constructor, inherited))
        return {baseClass, constructor};
    }

    if (!declaresInheritedConstructorSource(base, inherited)) continue;

    if (auto direct = binder.inheritedConstructorFor(base, source))
      return {baseClass, direct};
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
    *argsTail =
        make_list_node<ExpressionAST>(pool, makeForwardedParamRef(param));
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

  TranslationUnit::PotentiallyEvaluatedScope unusedUntilOdrUsed{
      binder.unit_, fn->isDefinitionRequired()};
  check.check_mem_initializers(body);
}

auto Binder::CompleteClass::buildRecordLayout()
    -> std::expected<bool, std::string> {
  return binder.buildRecordLayout(classSymbol);
}

void Binder::CompleteClass::complete(DeferredMemberContexts deferred) {
  classSymbol->setHasUserDeclaredConstructors(
      !classSymbol->declaredConstructors().empty());

  if (defersClassSemanticCompletion(binder.unit_, classSymbol)) {
    markComplete();
    return;
  }

  if (shouldSynthesizeSpecialMembers()) synthesizeSpecialMembers();

  synthesizeImplicitEqualityOperators();

  auto status = buildRecordLayout();
  if (!status.has_value())
    binder.error(classSymbol->location(), status.error());

  binder.computeClassFlags(classSymbol);

  if (!deferred.fieldInitializers) typeFieldInitializers();

  binder.refreshImplicitExceptionSpecifications(classSymbol);

  if (!deferred.exceptionSpecifications)
    checkOverriderExceptionSpecifications();

  if (shouldSynthesizeSpecialMembers()) {
    synthesizeStructorVariants();
    synthesizeMemberwiseBodies();
  }

  synthesizeDefaultedEqualityBodies();
  deduceDefaultedThreeWayReturnTypes();
  for (auto function : comparisonFunctions(TokenKind::T_LESS_EQUAL_GREATER)) {
    if (function->isDefaulted() && !function->isDeleted())
      synthesizeDefaultedThreeWayBody(function);
  }
  synthesizeDefaultedSecondaryComparisonBodies();

  markComplete();
}

namespace {
[[nodiscard]] auto isComparisonOperator(TokenKind op) -> bool {
  switch (op) {
    case TokenKind::T_EQUAL_EQUAL:
    case TokenKind::T_EXCLAIM_EQUAL:
    case TokenKind::T_LESS:
    case TokenKind::T_LESS_EQUAL:
    case TokenKind::T_GREATER:
    case TokenKind::T_GREATER_EQUAL:
    case TokenKind::T_LESS_EQUAL_GREATER:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] auto isComparisonOperatorFunction(FunctionSymbol* fn) -> bool {
  auto operatorId = name_cast<OperatorId>(fn->name());
  return operatorId && isComparisonOperator(operatorId->op());
}
}  // namespace

struct Binder::ImplicitExceptionSpecification {
  Binder& binder;
  FunctionSymbol* fn;
  ClassSymbol* classSymbol;
  bool isMoveForm;
  bool isAssignment;
  bool isCopyOrMoveConstructor;
  bool isDefaultConstructor;
  ClassSymbol* inheritedBase = nullptr;

  ImplicitExceptionSpecification(Binder& b, FunctionSymbol* f, ClassSymbol* cls)
      : binder(b),
        fn(f),
        classSymbol(cls),
        isMoveForm(f == cls->moveConstructor() ||
                   f == cls->moveAssignmentOperator()),
        isAssignment(f == cls->copyAssignmentOperator() ||
                     f == cls->moveAssignmentOperator()),
        isCopyOrMoveConstructor(f == cls->copyConstructor() ||
                                f == cls->moveConstructor()),
        isDefaultConstructor(f == cls->defaultConstructor()) {
    if (auto inherited = f->inheritedConstructor()) {
      inheritedBase = symbol_cast<ClassSymbol>(inherited->parent());
      if (inheritedBase) inheritedBase = inheritedBase->resolvedDefinition();
    }
  }

  [[nodiscard]] auto control() const -> Control* { return binder.control(); }

  [[nodiscard]] auto isPotentiallyThrowing() const -> bool {
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
  }

  [[nodiscard]] auto sourceType(const Type* subobjectType) const
      -> const Type* {
    if (isMoveForm) return control()->getRvalueReferenceType(subobjectType);
    return control()->getLvalueReferenceType(
        binder.traits.add_cv(subobjectType, CvQualifiers::kConst));
  }

  [[nodiscard]] auto initializerIsPotentiallyThrowing(FieldSymbol* field) const
      -> bool {
    SilentDiagnosticsScope silent{binder.unit_};
    auto initializer = field->initializer();
    if (silent.hadError()) return true;
    return TypeChecker::isPotentiallyThrowing(initializer);
  }

  [[nodiscard]] auto isInheritedBase(const ClassType* classType) const -> bool {
    if (!inheritedBase || !classType->symbol()) return false;
    return classType->symbol()->resolvedDefinition() == inheritedBase;
  }

  [[nodiscard]] auto initializationIsPotentiallyThrowing(
      const Type* type, FieldSymbol* field) const -> bool {
    auto& traits = binder.traits;

    if (fn->isDestructor()) {
      return traits.is_destructible(type) &&
             !traits.is_nothrow_destructible(type);
    }

    if (isDefaultConstructor && field && field->hasInitializer())
      return initializerIsPotentiallyThrowing(field);

    auto subobjectType = traits.remove_all_extents(type);
    if (traits.is_reference(subobjectType)) return false;

    auto classType = unqualified_cast<ClassType>(subobjectType);
    if (!classType) return false;

    if (isInheritedBase(classType)) {
      auto inheritedType =
          type_cast<FunctionType>(fn->inheritedConstructor()->type());
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
  }
};

void Binder::applyImplicitExceptionSpecification(FunctionSymbol* fn) {
  if (!fn || fn->hasExceptionSpecifier()) return;
  if (!fn->isDestructor() && !fn->isDefaulted()) return;
  if (isComparisonOperatorFunction(fn)) return;

  if (!type_cast<FunctionType>(fn->type())) return;

  auto classSymbol = symbol_cast<ClassSymbol>(fn->parent());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();

  ImplicitExceptionSpecification specification{*this, fn, classSymbol};

  if (specification.isDefaultConstructor &&
      hasUninstantiatedDefaultMemberInitializer(classSymbol)) {
    fn->setDeferredImplicitExceptionSpecification(true);
    return;
  }

  setFunctionNoexcept(control(), fn, !specification.isPotentiallyThrowing());
}

void Binder::completeDeferredImplicitExceptionSpecification(
    FunctionSymbol* fn) {
  if (!fn->hasDeferredImplicitExceptionSpecification()) return;
  fn->setDeferredImplicitExceptionSpecification(false);

  auto classSymbol = symbol_cast<ClassSymbol>(fn->parent());
  for (auto field : views::members(classSymbol) | views::non_static_fields)
    ASTRewriter::requireFieldInitializer(unit_, field);

  applyImplicitExceptionSpecification(fn);
}

void Binder::refreshImplicitExceptionSpecifications(ClassSymbol* classSymbol) {
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();

  for (auto constructor : classSymbol->declaredConstructors())
    applyImplicitExceptionSpecification(constructor);

  for (auto member : classSymbol->members()) {
    for (auto func : views::declared_functions(member))
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
    for (auto func : views::declared_functions(member)) {
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
  symbol->setConstexpr(true);
  symbol->setInline(true);
  symbol->setLanguageLinkage(LanguageKind::kCXX);
  return symbol;
}

void Binder::CompleteClass::attachDeclaration(FunctionSymbol* symbol,
                                              UnqualifiedIdAST* id) {
  binder.attachSynthesizedBody(symbol, id,
                               DefaultFunctionBodyAST::create(pool));
}

auto Binder::CompleteClass::makeDestructorId(ClassSymbol* cls)
    -> DestructorIdAST* {
  auto destructorId = DestructorIdAST::create(pool);
  if (auto id = name_cast<Identifier>(cls->name()))
    destructorId->id = NameIdAST::create(pool, id);
  return destructorId;
}

auto Binder::CompleteClass::makeCtorNameId() -> NameIdAST* {
  return NameIdAST::create(pool, name_cast<Identifier>(classSymbol->name()));
}

void Binder::CompleteClass::addFunctionToClassScope(FunctionSymbol* symbol) {
  binder.overloadSetFor(classSymbol, symbol->name(), symbol->location())
      ->addFunction(symbol);
}

auto Binder::CompleteClass::comparisonFunctions(TokenKind op) const
    -> std::vector<FunctionSymbol*> {
  std::vector<FunctionSymbol*> functions;
  auto name = control()->getOperatorId(op);

  auto appendUnique = [&](FunctionSymbol* function) {
    if (!std::ranges::contains(functions, function))
      functions.push_back(function);
  };

  for (auto candidate : classSymbol->find(name)) {
    for (auto function : views::declared_functions(candidate))
      appendUnique(function);
  }

  auto enclosingNamespace = classSymbol->enclosingNamespace();
  if (!enclosingNamespace) return functions;

  for (auto candidate : enclosingNamespace->find(name)) {
    for (auto function : views::declared_functions(candidate)) {
      if (!isFriendDeclaredIn(function, classSymbol)) continue;
      appendUnique(function);
    }
  }

  return functions;
}

void Binder::CompleteClass::synthesizeImplicitEqualityOperators() {
  if (!ast) return;
  if (!comparisonFunctions(TokenKind::T_EQUAL_EQUAL).empty()) return;

  for (auto declaration : ListView{ast->declarationList}) {
    auto definition = ast_cast<FunctionDefinitionAST>(declaration);
    if (!definition || !definition->symbol) continue;

    auto function = definition->symbol;
    if (!function->isDefaulted()) continue;
    auto operatorId = name_cast<OperatorId>(function->name());
    if (!operatorId) continue;
    if (operatorId->op() != TokenKind::T_LESS_EQUAL_GREATER) continue;
    (void)declareImplicitEqualityOperator(function);
  }
}

auto Binder::CompleteClass::declareImplicitEqualityOperator(
    FunctionSymbol* threeWay) -> FunctionSymbol* {
  auto threeWayType = type_cast<FunctionType>(threeWay->type());
  if (!threeWayType) return nullptr;

  auto equalityType = control()->getFunctionType(
      control()->getBoolType(), threeWayType->parameterTypes(),
      threeWayType->isVariadic(), threeWayType->cvQualifiers(),
      threeWayType->refQualifier(), threeWayType->exceptionSpecification());

  auto equality = newDefaultedFunction(
      control()->getOperatorId(TokenKind::T_EQUAL_EQUAL), equalityType);
  equality->setLocation(threeWay->location());
  equality->setAccessSpecifier(threeWay->accessSpecifier());
  equality->setStatic(threeWay->isStatic());
  equality->setFriend(threeWay->isFriend());
  equality->setExplicitObjectParameter(threeWay->hasExplicitObjectParameter());
  equality->setConsteval(threeWay->isConsteval());
  equality->setVirtual(threeWay->isVirtual());
  equality->setExceptionSpecifier(threeWay->hasExceptionSpecifier());
  equality->setTrailingRequiresClause(threeWay->trailingRequiresClause());

  auto parameters = control()->newFunctionParametersSymbol(equality, {});
  equality->addSymbol(parameters);
  if (auto sourceParameters = threeWay->functionParameters()) {
    for (auto source : views::members(sourceParameters) | views::parameters) {
      auto parameter =
          control()->newParameterSymbol(parameters, source->location());
      parameter->setName(source->name());
      parameter->setType(source->type());
      parameters->addSymbol(parameter);
    }
  }

  attachDeclaration(
      equality, OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL_EQUAL));

  if (equality->isFriend()) {
    equality->setHidden(true);
    equality->addBefriendingClass(classSymbol);
    auto enclosingNamespace = classSymbol->enclosingNamespace();
    if (enclosingNamespace) {
      binder
          .overloadSetFor(enclosingNamespace, equality->name(),
                          equality->location())
          ->addFunction(equality);
    }
  } else {
    addFunctionToClassScope(equality);
  }

  return equality;
}

void Binder::CompleteClass::synthesizeDefaultedEqualityBodies() {
  for (auto function : comparisonFunctions(TokenKind::T_EQUAL_EQUAL)) {
    if (!function->isDefaulted()) continue;
    synthesizeDefaultedEqualityBody(function);
  }
}

auto Binder::CompleteClass::checkUsable(TypeChecker& check,
                                        ExpressionAST*& expression) -> bool {
  CapturingDiagnosticsScope diagnostics{binder.unit_};
  check.check(&expression);
  diagnostics.finish();
  if (!diagnostics.diagnostics().empty()) return false;
  return expression->type != nullptr;
}

auto Binder::CompleteClass::comparisonCategoryType(WellKnownName name) const
    -> const Type* {
  auto symbol = lookupStandardLibraryType(binder.unit_, name);
  if (!symbol) return nullptr;
  return binder.traits.remove_cv(symbol->type());
}

auto Binder::CompleteClass::threeWayCategoryRank(
    TypeChecker& check, FunctionSymbol* fn, Symbol* subobject,
    std::span<const Type* const> categories) -> std::optional<int> {
  auto comparison = BinaryExpressionAST::create(pool);
  comparison->leftExpression = comparisonOperand(
      fn, ComparisonSide::kLeft, subobject, firstElementIndices(subobject));
  comparison->rightExpression = comparisonOperand(
      fn, ComparisonSide::kRight, subobject, firstElementIndices(subobject));
  comparison->op = TokenKind::T_LESS_EQUAL_GREATER;
  comparison->opLoc = fn->location();
  ExpressionAST* expression = comparison;
  if (!checkUsable(check, expression)) return std::nullopt;
  auto resultType = binder.traits.remove_cv(expression->type);
  for (int index = 0; index != static_cast<int>(categories.size()); ++index) {
    if (!categories[index]) continue;
    if (resultType == categories[index]) return index;
  }
  return std::nullopt;
}

void Binder::CompleteClass::deduceDefaultedThreeWayReturnTypes() {
  const Type* categories[] = {
      comparisonCategoryType(WellKnownName::T_STRONG_ORDERING),
      comparisonCategoryType(WellKnownName::T_WEAK_ORDERING),
      comparisonCategoryType(WellKnownName::T_PARTIAL_ORDERING)};
  for (auto function : comparisonFunctions(TokenKind::T_LESS_EQUAL_GREATER)) {
    if (!function->isDefaulted()) continue;
    auto type = type_cast<FunctionType>(function->type());
    if (!type || !type_cast<AutoType>(type->returnType())) continue;
    TypeChecker check{binder.unit_};
    check.setScope(function);
    check.setReportErrors(true);
    TranslationUnit::PotentiallyEvaluatedScope unevaluated{binder.unit_, false};
    int rank = 0;
    for (auto subobject : comparisonSubobjects(function)) {
      auto categoryRank =
          threeWayCategoryRank(check, function, subobject, categories);
      if (!categoryRank) {
        function->setDeleted(true);
        break;
      }
      rank = std::max(rank, *categoryRank);
    }
    if (function->isDeleted()) continue;
    if (!categories[rank]) {
      binder.error(
          function->location(),
          "you need to include <compare> before using the '<=>' operator");
      continue;
    }
    auto deducedType = control()->getFunctionType(
        categories[rank], type->parameterTypes(), type->isVariadic(),
        type->cvQualifiers(), type->refQualifier(),
        type->exceptionSpecification());
    for (auto declaration : function->declarations()) {
      declaration->setType(deducedType);
      declaration->setDeducedReturnType(true);
    }
  }
}

struct Binder::CompleteClass::DefaultedThreeWayComparison {
  CompleteClass& owner;
  FunctionSymbol* fn;
  const Type* returnType;
  TypeChecker check;
  bool isStrong;
  bool isWeak;
  bool isPartial;
  bool potentiallyThrowing = false;

  DefaultedThreeWayComparison(CompleteClass& o, FunctionSymbol* f,
                              const Type* r)
      : owner(o),
        fn(f),
        returnType(r),
        check(o.binder.unit_),
        isStrong(r ==
                 o.comparisonCategoryType(WellKnownName::T_STRONG_ORDERING)),
        isWeak(r == o.comparisonCategoryType(WellKnownName::T_WEAK_ORDERING)),
        isPartial(r ==
                  o.comparisonCategoryType(WellKnownName::T_PARTIAL_ORDERING)) {
    check.setScope(fn);
    check.setReportErrors(true);
  }

  [[nodiscard]] auto pool() const -> Arena* { return owner.pool; }
  [[nodiscard]] auto control() const -> Control* { return owner.control(); }

  [[nodiscard]] auto usable(ExpressionAST*& expression) -> bool {
    if (!owner.checkUsable(check, expression)) return false;
    if (TypeChecker::isPotentiallyThrowing(expression))
      potentiallyThrowing = true;
    return true;
  }

  [[nodiscard]] auto returnTypeSpecifier() const -> SpecifierAST* {
    return owner.typeSpecifier(returnType);
  }

  [[nodiscard]] auto castToReturnType(ExpressionAST* expression) const
      -> ExpressionAST* {
    auto cast = CppCastExpressionAST::create(pool());
    cast->castOp = TokenKind::T_STATIC_CAST;
    cast->castLoc = fn->location();
    cast->typeId = TypeIdAST::create(pool());
    cast->typeId->type = returnType;
    cast->typeId->typeSpecifierList =
        make_list_node<SpecifierAST>(pool(), returnTypeSpecifier());
    cast->expression = expression;
    return cast;
  }

  [[nodiscard]] auto subobjectBinary(const ComparisonElement& element,
                                     ComparisonSide leftSide, TokenKind op,
                                     ComparisonSide rightSide) const
      -> BinaryExpressionAST* {
    auto expression = BinaryExpressionAST::create(pool());
    expression->leftExpression = owner.comparisonOperand(
        fn, leftSide, element.subobject, owner.indexValues(element.indices));
    expression->op = op;
    expression->opLoc = fn->location();
    expression->rightExpression = owner.comparisonOperand(
        fn, rightSide, element.subobject, owner.indexValues(element.indices));
    return expression;
  }

  [[nodiscard]] auto categoryValue(const Type* type, WellKnownName name) const
      -> ExpressionAST* {
    auto classType = unqualified_cast<ClassType>(type);
    if (!classType) return nullptr;
    auto identifier = control()->getIdentifier(name);
    auto symbol = qualifiedLookup(classType->definition(), identifier);
    if (!symbol) return nullptr;
    auto expression = IdExpressionAST::create(pool());
    expression->symbol = symbol;
    expression->unqualifiedId = NameIdAST::create(pool(), identifier);
    expression->type = symbol->type();
    expression->valueCategory = ValueCategory::kLValue;
    return expression;
  }

  [[nodiscard]] auto conditional(ExpressionAST* condition, ExpressionAST* left,
                                 ExpressionAST* right) -> ExpressionAST* {
    if (!condition || !left || !right) return nullptr;
    auto conditionalExpression = ConditionalExpressionAST::create(pool());
    conditionalExpression->condition = condition;
    conditionalExpression->iftrueExpression = left;
    conditionalExpression->iffalseExpression = right;
    ExpressionAST* expression = conditionalExpression;
    if (!usable(expression)) return nullptr;
    return expression;
  }

  [[nodiscard]] auto orderingFromEqualityAndLess(
      const ComparisonElement& element) -> ExpressionAST* {
    if (!isStrong && !isWeak && !isPartial) return nullptr;
    ExpressionAST* equal =
        subobjectBinary(element, ComparisonSide::kLeft,
                        TokenKind::T_EQUAL_EQUAL, ComparisonSide::kRight);
    ExpressionAST* less =
        subobjectBinary(element, ComparisonSide::kLeft, TokenKind::T_LESS,
                        ComparisonSide::kRight);
    if (!usable(equal) || !usable(less)) return nullptr;
    auto tail = categoryValue(returnType, WellKnownName::T_GREATER);
    if (isPartial) {
      ExpressionAST* greater =
          subobjectBinary(element, ComparisonSide::kRight, TokenKind::T_LESS,
                          ComparisonSide::kLeft);
      if (!usable(greater)) return nullptr;
      tail = conditional(greater, tail,
                         categoryValue(returnType, WellKnownName::T_UNORDERED));
    }
    tail = conditional(less, categoryValue(returnType, WellKnownName::T_LESS),
                       tail);
    auto equalName =
        isStrong ? WellKnownName::T_EQUAL : WellKnownName::T_EQUIVALENT;
    return conditional(equal, categoryValue(returnType, equalName), tail);
  }

  [[nodiscard]] auto elementComparison(const ComparisonElement& element)
      -> ExpressionAST* {
    TranslationUnit::PotentiallyEvaluatedScope unevaluated{owner.binder.unit_,
                                                           false};
    auto threeWay = ThreeWayComparisonExpressionAST::create(pool());
    threeWay->comparison = subobjectBinary(element, ComparisonSide::kLeft,
                                           TokenKind::T_LESS_EQUAL_GREATER,
                                           ComparisonSide::kRight);
    ExpressionAST* comparison = threeWay;
    if (usable(comparison)) {
      auto cast = castToReturnType(comparison);
      if (!usable(cast)) return nullptr;
      return cast;
    }
    if (threeWay->comparison->symbol) return nullptr;
    if (check.wasLastOperatorLookupAmbiguous()) return nullptr;
    return orderingFromEqualityAndLess(element);
  }

  [[nodiscard]] auto declareComparisonResult(BlockSymbol* block,
                                             ExpressionAST* comparison)
      -> VariableSymbol* {
    auto variable = control()->newVariableSymbol(block, fn->location());
    variable->setName(control()->getIdentifier(
        std::format("$comparison{}", block->members().size())));
    variable->setType(returnType);
    block->addSymbol(variable);
    check.check_variable_initializer(variable, comparison, fn->location());
    variable->setInitializer(comparison);
    return variable;
  }

  [[nodiscard]] auto checkedReturn(ExpressionAST* expression)
      -> ReturnStatementAST* {
    auto result = ReturnStatementAST::create(pool());
    result->expression = expression;
    check.check_return_statement(result);
    if (TypeChecker::isPotentiallyThrowing(result->expression))
      potentiallyThrowing = true;
    return result;
  }

  [[nodiscard]] auto returnIfNonZero(VariableSymbol* variable)
      -> StatementAST* {
    auto zero = IntLiteralExpressionAST::create(pool());
    zero->literal = control()->integerLiteral("0");
    zero->type = control()->getIntType();
    zero->valueCategory = ValueCategory::kPrValue;
    auto nonZero = BinaryExpressionAST::create(pool());
    nonZero->leftExpression = owner.variableReference(variable);
    nonZero->op = TokenKind::T_EXCLAIM_EQUAL;
    nonZero->opLoc = fn->location();
    nonZero->rightExpression = zero;
    ExpressionAST* condition = nonZero;
    if (!usable(condition) || !check.check_bool_condition(condition))
      return nullptr;
    auto statement = IfStatementAST::create(pool());
    statement->condition = condition;
    statement->statement = checkedReturn(owner.variableReference(variable));
    return statement;
  }

  [[nodiscard]] auto returnEqual() -> StatementAST* {
    auto strongOrdering =
        owner.comparisonCategoryType(WellKnownName::T_STRONG_ORDERING);
    auto equal = categoryValue(strongOrdering, WellKnownName::T_EQUAL);
    if (!equal) return nullptr;
    auto cast = castToReturnType(equal);
    if (!usable(cast)) return nullptr;
    return checkedReturn(cast);
  }

  [[nodiscard]] auto elementStatement(ScopeSymbol* scope,
                                      const ComparisonElement& element)
      -> StatementAST* {
    auto block = owner.newBlock(scope, fn->location());
    check.setScope(block);
    auto comparison = elementComparison(element);
    if (!comparison) return nullptr;
    auto variable = declareComparisonResult(block, comparison);
    auto returnIfUnequal = returnIfNonZero(variable);
    if (!returnIfUnequal) return nullptr;
    StatementAST* statements[] = {owner.declarationStatement(variable),
                                  returnIfUnequal};
    return owner.compoundStatement(block, statements);
  }

  [[nodiscard]] auto body() -> CompoundStatementAST* {
    auto subobjects = owner.comparisonSubobjects(fn);
    if (fn->isDeleted()) return nullptr;

    TranslationUnit::PotentiallyEvaluatedScope evaluated{owner.binder.unit_,
                                                         true};
    auto block = owner.newBlock(fn, fn->location());
    std::vector<StatementAST*> statements;
    for (auto subobject : subobjects) {
      auto loops = owner.elementLoops(check, block, subobject, fn->location());
      auto statement =
          elementStatement(loops.scope, {subobject, loops.indices});
      if (!statement) return nullptr;
      statements.push_back(loops.enclose(statement));
    }

    check.setScope(block);
    auto result = returnEqual();
    if (!result) return nullptr;
    statements.push_back(result);
    return owner.compoundStatement(block, statements);
  }
};

void Binder::CompleteClass::synthesizeDefaultedThreeWayBody(
    FunctionSymbol* fn) {
  auto definition = fn->declaration();
  if (!definition) return;
  if (!ast_cast<DefaultFunctionBodyAST>(definition->functionBody)) return;
  auto functionType = type_cast<FunctionType>(fn->type());
  if (!functionType) return;
  if (containsPlaceholderType(functionType->returnType())) return;

  DefaultedThreeWayComparison synthesis{*this, fn, functionType->returnType()};
  CapturingDiagnosticsScope diagnostics{binder.unit_};
  auto compound = synthesis.body();
  diagnostics.finish();
  if (!compound || !diagnostics.diagnostics().empty()) {
    fn->setDeleted(true);
    return;
  }

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compound;
  definition->functionBody = body;
  if (!fn->hasExceptionSpecifier())
    setFunctionNoexcept(control(), fn, !synthesis.potentiallyThrowing);
}

void Binder::CompleteClass::synthesizeDefaultedSecondaryComparisonBodies() {
  const TokenKind secondaryOperators[] = {
      TokenKind::T_EXCLAIM_EQUAL, TokenKind::T_LESS, TokenKind::T_GREATER,
      TokenKind::T_LESS_EQUAL, TokenKind::T_GREATER_EQUAL};
  for (auto op : secondaryOperators) {
    for (auto function : comparisonFunctions(op)) {
      if (function->isDefaulted())
        synthesizeDefaultedSecondaryComparisonBody(function, op);
    }
  }
}

void Binder::CompleteClass::synthesizeDefaultedSecondaryComparisonBody(
    FunctionSymbol* fn, TokenKind op) {
  auto definition = fn->declaration();
  if (!definition) return;
  if (!ast_cast<DefaultFunctionBodyAST>(definition->functionBody)) return;
  if (!type_cast<FunctionType>(fn->type())) return;
  if (!hasComparisonParameters(fn)) return;

  auto block = newBlock(fn, fn->location());
  auto result = secondaryComparisonReturn(fn, op, block);
  if (!result) {
    fn->setDeleted(true);
    return;
  }

  StatementAST* statements[] = {result};
  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compoundStatement(block, statements);
  definition->functionBody = body;
  if (!fn->hasExceptionSpecifier()) {
    setFunctionNoexcept(
        control(), fn, !TypeChecker::isPotentiallyThrowing(result->expression));
  }
}

auto Binder::CompleteClass::secondaryComparisonReturn(FunctionSymbol* fn,
                                                      TokenKind op,
                                                      BlockSymbol* block)
    -> ReturnStatementAST* {
  TypeChecker check{binder.unit_};
  check.setScope(block);
  check.setReportErrors(true);
  check.excludeOperatorCandidate(fn);
  CapturingDiagnosticsScope diagnostics{binder.unit_};

  auto left = comparisonObject(fn, ComparisonSide::kLeft);
  auto right = comparisonObject(fn, ComparisonSide::kRight);
  if (!check.lookupOperator(left->type, op, right->type, left, right))
    return nullptr;
  if (!check.wasLastOperatorRewritten()) return nullptr;

  auto comparison = BinaryExpressionAST::create(pool);
  comparison->leftExpression = left;
  comparison->op = op;
  comparison->opLoc = fn->location();
  comparison->rightExpression = right;

  auto result = ReturnStatementAST::create(pool);
  result->expression = comparison;
  check.check(&result->expression);
  if (!result->expression->type) return nullptr;
  check.check_return_statement(result);
  if (!diagnostics.diagnostics().empty()) return nullptr;
  return result;
}

auto Binder::CompleteClass::hasReferenceOrVariantMembers() const -> bool {
  if (has_variant_members(classSymbol)) return true;
  return std::ranges::any_of(
      views::members(classSymbol) | views::non_static_fields,
      [this](FieldSymbol* field) {
        return binder.traits.is_reference(field->type());
      });
}

auto Binder::CompleteClass::comparisonSubobjects(FunctionSymbol* fn)
    -> std::vector<Symbol*> {
  auto definition = fn->declaration();
  if (!definition) return {};
  if (!ast_cast<DefaultFunctionBodyAST>(definition->functionBody)) return {};

  if (hasReferenceOrVariantMembers()) {
    fn->setDeleted(true);
    return {};
  }

  if (!type_cast<FunctionType>(fn->type())) return {};
  if (!hasComparisonParameters(fn)) return {};

  std::vector<Symbol*> subobjects;
  for (auto baseClass : classSymbol->baseClasses()) {
    if (auto base = resolved_base_class(baseClass)) subobjects.push_back(base);
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (arrayElementCount(field->type()) == 0) continue;
    subobjects.push_back(field);
  }

  return subobjects;
}

auto Binder::CompleteClass::ElementLoops::enclose(StatementAST* statement) const
    -> StatementAST* {
  if (loops.empty()) return statement;
  loops.back()->statement = statement;
  return loops.front();
}

auto Binder::CompleteClass::elementLoops(TypeChecker& check, ScopeSymbol* scope,
                                         Symbol* subobject,
                                         SourceLocation location)
    -> ElementLoops {
  ElementLoops nest{.scope = scope};
  if (auto field = symbol_cast<FieldSymbol>(subobject)) {
    for (auto extent : boundedArrayExtents(field->type()))
      appendElementLoop(check, nest, extent, location);
  }
  return nest;
}

void Binder::CompleteClass::appendElementLoop(TypeChecker& check,
                                              ElementLoops& nest,
                                              std::size_t extent,
                                              SourceLocation location) {
  auto block = newBlock(nest.scope, location);
  check.setScope(block);

  auto index = control()->newVariableSymbol(block, location);
  index->setName(
      control()->getIdentifier(std::format("$index{}", nest.indices.size())));
  index->setType(control()->getSizeType());
  block->addSymbol(index);
  auto initializer = sizeLiteral(0);
  check.check_variable_initializer(index, initializer, location);
  index->setInitializer(initializer);

  auto condition = BinaryExpressionAST::create(pool);
  condition->leftExpression = variableReference(index);
  condition->op = TokenKind::T_LESS;
  condition->opLoc = location;
  condition->rightExpression = sizeLiteral(extent);

  auto increment = UnaryExpressionAST::create(pool);
  increment->op = TokenKind::T_PLUS_PLUS;
  increment->opLoc = location;
  increment->expression = variableReference(index);

  auto loop = ForStatementAST::create(pool);
  loop->symbol = block;
  loop->forLoc = location;
  loop->initializer = declarationStatement(index);
  loop->condition = condition;
  loop->expression = increment;
  check.check(&loop->condition);
  (void)check.check_bool_condition(loop->condition);
  check.check(&loop->expression);

  if (!nest.loops.empty()) nest.loops.back()->statement = loop;
  nest.loops.push_back(loop);
  nest.indices.push_back(index);
  nest.scope = block;
}

auto Binder::CompleteClass::indexValues(
    std::span<VariableSymbol* const> indices) -> std::vector<ExpressionAST*> {
  std::vector<ExpressionAST*> values;
  for (auto index : indices) values.push_back(variableValue(index));
  return values;
}

auto Binder::CompleteClass::firstElementIndices(Symbol* subobject)
    -> std::vector<ExpressionAST*> {
  auto field = symbol_cast<FieldSymbol>(subobject);
  if (!field) return {};

  const auto rank = boundedArrayExtents(field->type()).size();
  std::vector<ExpressionAST*> indices;
  for (std::size_t level = 0; level < rank; ++level)
    indices.push_back(sizeLiteral(0));
  return indices;
}

auto Binder::CompleteClass::sizeLiteral(std::size_t value) -> ExpressionAST* {
  auto literal = IntLiteralExpressionAST::create(pool);
  literal->literal = control()->integerLiteral(std::to_string(value));
  literal->type = control()->getSizeType();
  literal->valueCategory = ValueCategory::kPrValue;
  return literal;
}

auto Binder::CompleteClass::newBlock(ScopeSymbol* scope,
                                     SourceLocation location) -> BlockSymbol* {
  auto block = control()->newBlockSymbol(scope, location);
  scope->addSymbol(block);
  return block;
}

auto Binder::CompleteClass::compoundStatement(
    BlockSymbol* block, std::span<StatementAST* const> statements)
    -> CompoundStatementAST* {
  auto compound = CompoundStatementAST::create(pool);
  compound->symbol = block;
  auto tail = &compound->statementList;
  for (auto statement : statements) {
    *tail = make_list_node(pool, statement);
    tail = &(*tail)->next;
  }
  return compound;
}

auto Binder::CompleteClass::comparisonOperand(
    FunctionSymbol* fn, ComparisonSide side, Symbol* subobject,
    std::span<ExpressionAST* const> indices) -> ExpressionAST* {
  auto object = comparisonObject(fn, side);

  auto field = symbol_cast<FieldSymbol>(subobject);
  if (!field) return baseSubobject(object, symbol_cast<ClassSymbol>(subobject));

  ExpressionAST* expression = memberSubobject(object, field);
  for (auto index : indices) expression = arrayElement(expression, index);
  return expression;
}

auto Binder::CompleteClass::comparisonObject(FunctionSymbol* fn,
                                             ComparisonSide side)
    -> ExpressionAST* {
  auto parameters = comparisonParameters(fn);
  if (!fn->isImplicitObjectMemberFunction())
    return makeParamRef(parameters[side == ComparisonSide::kLeft ? 0 : 1]);
  if (side == ComparisonSide::kRight) return makeParamRef(parameters[0]);
  return implicitComparisonObject(fn);
}

auto Binder::CompleteClass::implicitComparisonObject(FunctionSymbol* fn)
    -> ExpressionAST* {
  auto functionType = type_cast<FunctionType>(fn->type());
  auto objectType =
      binder.traits.add_cv(classSymbol->type(), functionType->cvQualifiers());
  auto thisExpression = ThisExpressionAST::create(pool);
  thisExpression->type = control()->getPointerType(objectType);
  thisExpression->valueCategory = ValueCategory::kPrValue;

  auto object = UnaryExpressionAST::create(pool);
  object->op = TokenKind::T_STAR;
  object->expression = thisExpression;
  object->type = objectType;
  object->valueCategory = ValueCategory::kLValue;
  return object;
}

auto Binder::CompleteClass::baseSubobject(ExpressionAST* object,
                                          ClassSymbol* base) -> ExpressionAST* {
  auto conversion = ImplicitCastExpressionAST::create(pool);
  conversion->castKind = ImplicitCastKind::kDerivedToBaseConversion;
  conversion->expression = object;
  conversion->type =
      binder.traits.add_cv(base->type(), cv_qualifiers(object->type));
  conversion->valueCategory = ValueCategory::kLValue;
  return conversion;
}

auto Binder::CompleteClass::memberSubobject(ExpressionAST* object,
                                            FieldSymbol* field)
    -> ExpressionAST* {
  auto cv = cv_qualifiers(object->type);
  if (field->isMutable()) cv &= ~CvQualifiers::kConst;

  auto member = MemberExpressionAST::create(pool);
  member->baseExpression = object;
  member->accessOp = TokenKind::T_DOT;
  if (auto id = name_cast<Identifier>(field->name()))
    member->unqualifiedId = NameIdAST::create(pool, id);
  member->symbol = field;
  member->type = binder.traits.add_cv(field->type(), cv);
  if (binder.traits.is_reference(field->type()))
    member->type = binder.traits.remove_reference(field->type());
  member->valueCategory = ValueCategory::kLValue;
  return member;
}

auto Binder::CompleteClass::arrayElement(ExpressionAST* array,
                                         ExpressionAST* index)
    -> ExpressionAST* {
  auto arrayType = unqualified_cast<BoundedArrayType>(array->type);

  auto subscript = SubscriptExpressionAST::create(pool);
  subscript->baseExpression = array;
  subscript->indexExpression = index;
  subscript->type = binder.traits.add_cv(arrayType->elementType(),
                                         cv_qualifiers(array->type));
  subscript->valueCategory = ValueCategory::kLValue;
  return subscript;
}

auto Binder::CompleteClass::returnFalseIfUnequal(
    TypeChecker& check, FunctionSymbol* fn, const ComparisonElement& element)
    -> IfStatementAST* {
  auto comparison = BinaryExpressionAST::create(pool);
  comparison->leftExpression =
      comparisonOperand(fn, ComparisonSide::kLeft, element.subobject,
                        indexValues(element.indices));
  comparison->op = TokenKind::T_EQUAL_EQUAL;
  comparison->opLoc = fn->location();
  comparison->rightExpression =
      comparisonOperand(fn, ComparisonSide::kRight, element.subobject,
                        indexValues(element.indices));
  ExpressionAST* equal = comparison;
  check.check(&equal);
  if (!check.check_bool_condition(equal)) return nullptr;

  auto negation = UnaryExpressionAST::create(pool);
  negation->op = TokenKind::T_EXCLAIM;
  negation->opLoc = fn->location();
  negation->expression = equal;
  ExpressionAST* unequal = negation;
  check.check(&unequal);

  auto statement = IfStatementAST::create(pool);
  statement->condition = unequal;
  statement->statement = boolReturn(check, false);
  return statement;
}

auto Binder::CompleteClass::boolReturn(TypeChecker& check, bool value)
    -> ReturnStatementAST* {
  auto result = ReturnStatementAST::create(pool);
  result->expression = BoolLiteralExpressionAST::create(
      pool, {}, value, ValueCategory::kPrValue, control()->getBoolType());
  check.check_return_statement(result);
  return result;
}

void Binder::CompleteClass::synthesizeDefaultedEqualityBody(
    FunctionSymbol* fn) {
  auto definition = fn->declaration();
  if (!definition) return;
  if (!ast_cast<DefaultFunctionBodyAST>(definition->functionBody)) return;

  auto subobjects = comparisonSubobjects(fn);
  if (fn->isDeleted()) return;

  TypeChecker check{binder.unit_};
  check.setReportErrors(true);
  CapturingDiagnosticsScope diagnostics{binder.unit_};
  TranslationUnit::PotentiallyEvaluatedScope unevaluated{binder.unit_, false};

  auto block = newBlock(fn, fn->location());
  std::vector<StatementAST*> statements;
  bool potentiallyThrowing = false;
  for (auto subobject : subobjects) {
    auto loops = elementLoops(check, block, subobject, fn->location());
    check.setScope(loops.scope);
    auto statement =
        returnFalseIfUnequal(check, fn, {subobject, loops.indices});
    if (!statement) {
      fn->setDeleted(true);
      return;
    }
    if (TypeChecker::isPotentiallyThrowing(statement->condition))
      potentiallyThrowing = true;
    statements.push_back(loops.enclose(statement));
  }

  check.setScope(block);
  statements.push_back(boolReturn(check, true));

  diagnostics.finish();
  if (!diagnostics.diagnostics().empty()) {
    fn->setDeleted(true);
    return;
  }

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compoundStatement(block, statements);
  definition->functionBody = body;
  if (!fn->hasExceptionSpecifier())
    setFunctionNoexcept(control(), fn, !potentiallyThrowing);
}

auto Binder::CompleteClass::isSubobjectMemberUsable(
    FunctionSymbol* function, ClassSymbol* subobject) const -> bool {
  if (!function) return false;
  if (function->isDeleted()) return false;

  AccessContext accessContext{binder.unit_, classSymbol};
  return accessContext.isAccessible(function, subobject, nullptr);
}

auto Binder::CompleteClass::defaultConstructorIsDeleted() const -> bool {
  auto traits = binder.traits;

  auto subobjectIsNotDefaultConstructible = [&](const Type* type) {
    auto classType = unqualified_cast<ClassType>(type);
    if (!classType || !classType->symbol()) return false;
    auto subobject = classType->symbol()->resolvedDefinition();
    if (!subobject->isComplete()) return false;

    if (auto destructor = subobject->destructor();
        destructor && !isSubobjectMemberUsable(destructor, subobject))
      return true;

    if (subobject->constructors().empty()) return false;

    return !isSubobjectMemberUsable(subobject->defaultConstructor(), subobject);
  };

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (baseClass && subobjectIsNotDefaultConstructible(baseClass->type()))
      return true;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (field->hasInitializer()) continue;

    auto type = field->type();
    if (traits.is_reference(type)) return true;

    auto element = traits.remove_all_extents(type);
    if (declaresNonConstDefaultConstructibleConstMember(field)) return true;

    if (subobjectIsNotDefaultConstructible(element)) return true;
  }

  return false;
}

auto Binder::CompleteClass::declaresNonConstDefaultConstructibleConstMember(
    FieldSymbol* field) const -> bool {
  if (classSymbol->isUnion()) return false;
  if (!binder.traits.is_const(field->type())) return false;
  return !binder.traits.is_const_default_constructible(field->type());
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
      control()->getQualType(classSymbol->type(), CvQualifiers::kConst));

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

        auto classType = unqualified_cast<ClassType>(paramType);
        return classType && classType->symbol() == classSymbol;
      });
}

void Binder::CompleteClass::addCopyAssignmentOperator() {
  if (hasUserDeclaredAssignmentOperator(/*moveForm=*/false)) return;

  auto constRefType = control()->getLvalueReferenceType(
      control()->getQualType(classSymbol->type(), CvQualifiers::kConst));
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
    auto classType = unqualified_cast<ClassType>(type);
    if (!classType || !classType->symbol()) return false;
    auto subobject = classType->symbol()->resolvedDefinition();
    if (!subobject->isComplete()) return false;

    if (auto destructor = subobject->destructor();
        destructor && !isSubobjectMemberUsable(destructor, subobject))
      return true;

    auto constructor = moveForm ? subobject->moveConstructor() : nullptr;
    if (!constructor) constructor = subobject->copyConstructor();
    return constructor && !isSubobjectMemberUsable(constructor, subobject);
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
  auto& traits = binder.traits;

  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = resolved_base_class(base);
    if (baseClass && subobjectAssignmentIsUnusable(baseClass->type(), moveForm))
      return true;
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    auto type = field->type();
    if (traits.is_reference(type)) return true;

    auto element = traits.remove_all_extents(type);
    if (traits.is_const(element) && !traits.is_class(traits.remove_cv(element)))
      return true;

    if (subobjectAssignmentIsUnusable(element, moveForm)) return true;
  }

  return false;
}

auto Binder::CompleteClass::subobjectAssignmentIsUnusable(const Type* type,
                                                          bool moveForm) const
    -> bool {
  auto classType = unqualified_cast<ClassType>(type);
  if (!classType || !classType->symbol()) return false;
  auto subobject = classType->symbol()->resolvedDefinition();
  if (!subobject->isComplete()) return false;

  auto arena = binder.unit_->arena();
  auto sourceType = moveForm ? type : binder.traits.add_const(type);
  auto target = ThisExpressionAST::create(arena, ValueCategory::kLValue, type);
  auto source = ThisExpressionAST::create(
      arena, moveForm ? ValueCategory::kXValue : ValueCategory::kLValue,
      sourceType);

  OverloadResolution resolution{binder.unit_};
  auto assignment = resolution.lookupOperator(nullptr, type, TokenKind::T_EQUAL,
                                              sourceType, target, source);
  if (resolution.wasLastLookupAmbiguous()) return true;
  return !isSubobjectMemberUsable(assignment, subobject);
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

  auto dtorId = makeDestructorId(classSymbol);
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
  variant->setInline(principal->isInline());
  variant->setConstexpr(principal->isConstexpr());
  variant->setConsteval(principal->isConsteval());
  binder.inheritDeclarationAttributes(variant, principal);

  binder.declareSynthesizedParameters(
      variant, type_cast<FunctionType>(principal->type()),
      principal->location());

  return variant;
}

auto Binder::CompleteClass::makeThisExpr() -> ExpressionAST* {
  auto thisExpr = ThisExpressionAST::create(pool);
  thisExpr->type = control()->getPointerType(classSymbol->type());
  thisExpr->valueCategory = ValueCategory::kPrValue;
  return thisExpr;
}

auto Binder::CompleteClass::makeSelfExpr() -> ExpressionAST* {
  auto self = UnaryExpressionAST::create(pool);
  self->op = TokenKind::T_STAR;
  self->expression = makeThisExpr();
  self->type = classSymbol->type();
  self->valueCategory = ValueCategory::kLValue;
  return self;
}

auto Binder::CompleteClass::assignsArrayElementwise(FieldSymbol* field,
                                                    bool isMove) const -> bool {
  if (!field->name()) return false;
  auto& traits = binder.traits;
  auto elementType = traits.remove_all_extents(field->type());
  if (elementType == field->type()) return false;
  auto target = control()->getLvalueReferenceType(elementType);
  const Type* source = control()->getLvalueReferenceType(
      traits.add_cv(elementType, CvQualifiers::kConst));
  if (isMove) source = control()->getRvalueReferenceType(elementType);
  return !traits.is_trivially_assignable(target, source);
}

auto Binder::CompleteClass::arrayMemberAssignment(
    TypeChecker& check, FunctionSymbol* fn, FieldSymbol* field,
    ParameterSymbol* source, bool isMove) -> StatementAST* {
  auto loops = elementLoops(check, fn, field, field->location());
  check.setScope(loops.scope);

  auto target = memberSubobject(makeSelfExpr(), field);
  for (auto index : indexValues(loops.indices))
    target = arrayElement(target, index);

  auto element = memberSubobject(makeParamRef(source), field);
  for (auto index : indexValues(loops.indices))
    element = arrayElement(element, index);

  auto assignment = AssignmentExpressionAST::create(pool);
  assignment->leftExpression = target;
  assignment->op = TokenKind::T_EQUAL;
  assignment->rightExpression =
      makeSourceSubobjectRef(element, element->type, isMove);

  auto statement = ExpressionStatementAST::create(pool);
  statement->expression = assignment;
  check.check(&statement->expression);
  check.setScope(fn);
  return loops.enclose(statement);
}

auto Binder::CompleteClass::baseAssignmentStatement(TypeChecker& check,
                                                    ClassSymbol* base,
                                                    ParameterSymbol* source,
                                                    bool isMove)
    -> StatementAST* {
  auto callee = MemberExpressionAST::create(pool);
  callee->baseExpression = makeSelfExpr();
  callee->accessOp = TokenKind::T_DOT;
  callee->nestedNameSpecifier = makeQualifier(base);
  callee->unqualifiedId =
      OperatorFunctionIdAST::create(pool, TokenKind::T_EQUAL);

  auto argument = ImplicitCastExpressionAST::create(pool);
  argument->castKind = ImplicitCastKind::kDerivedToBaseConversion;
  argument->expression = makeParamRef(source);
  argument->type =
      isMove ? base->type()
             : control()->getQualType(base->type(), CvQualifiers::kConst);
  argument->valueCategory =
      isMove ? ValueCategory::kXValue : ValueCategory::kLValue;

  auto call = CallExpressionAST::create(pool);
  call->baseExpression = callee;
  call->expressionList = make_list_node<ExpressionAST>(pool, argument);
  check.check(&call->baseExpression);

  auto statement = ExpressionStatementAST::create(pool);
  statement->expression = call;
  check.check(&statement->expression);
  return statement;
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

auto Binder::CompleteClass::makeForwardedParamRef(ParameterSymbol* param)
    -> ExpressionAST* {
  auto argExpr = makeParamRef(param);
  auto paramType = binder.traits.remove_cv(param->type());
  if (binder.traits.is_reference(paramType)) return argExpr;
  if (binder.traits.is_class_or_union(paramType)) {
    argExpr->valueCategory = ValueCategory::kPrValue;
    return argExpr;
  }
  auto load = ImplicitCastExpressionAST::create(pool);
  load->castKind = ImplicitCastKind::kLValueToRValueConversion;
  load->expression = argExpr;
  load->type = binder.traits.remove_cv(argExpr->type);
  load->valueCategory = ValueCategory::kPrValue;
  return load;
}

auto Binder::CompleteClass::makeQualifier(ClassSymbol* cls)
    -> NestedNameSpecifierAST* {
  auto nns = SimpleNestedNameSpecifierAST::create(pool);
  nns->identifier = name_cast<Identifier>(cls->name());
  nns->symbol = cls;
  return nns;
}

auto Binder::CompleteClass::typeSpecifier(const Type* type) -> SpecifierAST* {
  auto alias = control()->newTypeAliasSymbol(nullptr, {});
  alias->setType(type);
  auto specifier = NamedTypeSpecifierAST::create(pool);
  specifier->symbol = alias;
  return specifier;
}

auto Binder::CompleteClass::declarationStatement(VariableSymbol* variable)
    -> StatementAST* {
  auto id = IdDeclaratorAST::create(pool);
  id->unqualifiedId =
      NameIdAST::create(pool, name_cast<Identifier>(variable->name()));
  auto declarator = DeclaratorAST::create(pool);
  declarator->coreDeclarator = id;
  auto init = InitDeclaratorAST::create(pool);
  init->declarator = declarator;
  init->symbol = variable;
  init->initializer = variable->initializer();
  auto declaration = SimpleDeclarationAST::create(pool);
  declaration->declSpecifierList =
      make_list_node<SpecifierAST>(pool, typeSpecifier(variable->type()));
  declaration->initDeclaratorList = make_list_node(pool, init);
  auto statement = DeclarationStatementAST::create(pool);
  statement->declaration = declaration;
  return statement;
}

auto Binder::CompleteClass::variableReference(VariableSymbol* variable)
    -> IdExpressionAST* {
  auto reference = IdExpressionAST::create(pool);
  reference->symbol = variable;
  reference->unqualifiedId =
      NameIdAST::create(pool, name_cast<Identifier>(variable->name()));
  reference->type = variable->type();
  reference->valueCategory = ValueCategory::kLValue;
  return reference;
}

auto Binder::CompleteClass::variableValue(VariableSymbol* variable)
    -> ExpressionAST* {
  auto value = ImplicitCastExpressionAST::create(pool);
  value->castKind = ImplicitCastKind::kLValueToRValueConversion;
  value->expression = variableReference(variable);
  value->type = variable->type();
  value->valueCategory = ValueCategory::kPrValue;
  return value;
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
    member->unqualifiedId = makeDestructorId(calleeClass);
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

void Binder::CompleteClass::synthesizeDelegatingCompleteObjectCtor(
    FunctionSymbol* ctor) {
  auto definition = ast_cast<FunctionDefinitionAST>(ctor->declaration());
  if (!definition || !definition->functionBody) return;

  if (auto variant = ctor->completeObjectVariant()) {
    auto variantDefinition =
        ast_cast<FunctionDefinitionAST>(variant->declaration());
    if (variantDefinition &&
        variantDefinition->functionBody == definition->functionBody)
      return;
  }

  auto target = ctor->delegatingConstructor();
  ASTRewriter::requireFunctionDefinition(binder.unit_, target);
  binder.synthesizeCompleteObjectCtor(target);
  ASTRewriter::requireFunctionDefinition(binder.unit_,
                                         target->completeObjectVariant());

  auto variant = newStructorVariant(ctor);
  binder.attachSynthesizedBody(variant, makeCtorNameId(),
                               definition->functionBody);
  ctor->setCompleteObjectVariant(variant);
}

void Binder::CompleteClass::synthesizeCompleteObjectCtor(FunctionSymbol* ctor) {
  if (ctor->delegatingConstructor()) {
    synthesizeDelegatingCompleteObjectCtor(ctor);
    return;
  }

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

  for (auto vbase : virtual_base_initialization_order(classSymbol)) {
    auto vbaseCtor = pickVBaseConstructor(vbase, isCopy, isMove);
    ASTRewriter::requireFunctionDefinition(binder.unit_, vbaseCtor);

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
          isCopy ? control()->getQualType(vbase->type(), CvQualifiers::kConst)
                 : vbase->type();
      cast->valueCategory = ValueCategory::kLValue;
      init->expressionList = make_list_node<ExpressionAST>(pool, cast);
    } else if (vbaseCtor) {
      TypeChecker check{binder.unit_};
      check.setScope(ctor);
      check.append_default_arguments(vbaseCtor, &init->expressionList,
                                     ctor->location());
    }

    *memInitsTail = make_list_node<MemInitializerAST>(pool, init);
    memInitsTail = &(*memInitsTail)->next;
  }

  auto delegate = ParenMemInitializerAST::create(pool);
  delegate->unqualifiedId = makeCtorNameId();
  delegate->symbol = classSymbol;
  delegate->constructor = ctor;
  ASTRewriter::requireFunctionDefinition(binder.unit_, ctor);

  List<ExpressionAST*>* args = nullptr;
  auto argsTail = &args;
  for (auto param : params) {
    *argsTail =
        make_list_node<ExpressionAST>(pool, makeForwardedParamRef(param));
    argsTail = &(*argsTail)->next;
  }
  delegate->expressionList = args;

  *memInitsTail = make_list_node<MemInitializerAST>(pool, delegate);

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->memInitializerList = memInits;
  body->statement = CompoundStatementAST::create(pool);

  binder.attachSynthesizedBody(variant, makeCtorNameId(), body);
  ctor->setCompleteObjectVariant(variant);
}

void Binder::CompleteClass::synthesizeCompleteObjectDtor(FunctionSymbol* dtor) {
  auto layout = classSymbol->layout();

  auto variant = newStructorVariant(dtor);
  variant->setVirtual(dtor->isVirtual());

  auto compound = CompoundStatementAST::create(pool);
  compound->symbol = control()->newBlockSymbol(variant, dtor->location());
  variant->addSymbol(compound->symbol);

  std::vector<StatementAST*> statements;
  std::vector<std::pair<FunctionSymbol*, VariableSymbol*>> virtualBases;
  for (auto vbase :
       virtual_base_initialization_order(classSymbol) | std::views::reverse) {
    auto vbaseDtor = vbase->destructor();
    if (!vbaseDtor) continue;

    auto address = ImplicitCastExpressionAST::create(pool);
    address->castKind = ImplicitCastKind::kDerivedToBaseConversion;
    address->expression = makeThisExpr();
    address->type = control()->getPointerType(vbase->type());
    address->valueCategory = ValueCategory::kPrValue;

    auto variable =
        control()->newVariableSymbol(compound->symbol, dtor->location());
    variable->setName(control()->getIdentifier(
        std::format("$vbase{}", compound->symbol->members().size())));
    variable->setType(address->type);
    variable->setInitializer(address);
    compound->symbol->addSymbol(variable);

    statements.push_back(declarationStatement(variable));
    virtualBases.emplace_back(vbaseDtor, variable);
  }

  statements.push_back(makeStructorCallStatement(dtor, makeThisExpr()));
  for (auto [vbaseDtor, variable] : virtualBases)
    statements.push_back(
        makeStructorCallStatement(vbaseDtor, variableValue(variable)));

  auto tail = &compound->statementList;
  for (auto statement : statements) {
    *tail = make_list_node(pool, statement);
    tail = &(*tail)->next;
  }

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compound;

  auto dtorId = makeDestructorId(classSymbol);

  binder.attachSynthesizedBody(variant, dtorId, body);
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

  auto operatorDelete = resolveUsualOperatorDelete(binder.unit_, classSymbol,
                                                   classSymbol->type(), false);
  if (!operatorDelete) {
    binder.error(dtor->location(), "no suitable deallocation function");
    return;
  }

  auto signature = deallocationSignatureOf(binder.unit_, operatorDelete);
  if (signature->isDestroying) {
    binder.error(dtor->location(),
                 "destroying operator delete is not supported");
    return;
  }

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

  auto argumentsTail = &call->expressionList->next;
  const auto& parameterTypes =
      type_cast<FunctionType>(operatorDelete->type())->parameterTypes();

  auto appendArgument = [&](std::size_t index, std::uint64_t value) {
    auto literal = IntLiteralExpressionAST::create(pool);
    literal->literal = control()->integerLiteral(std::to_string(value));
    literal->type = parameterTypes[index];
    literal->valueCategory = ValueCategory::kPrValue;
    *argumentsTail = make_list_node<ExpressionAST>(pool, literal);
    argumentsTail = &(*argumentsTail)->next;
  };

  auto memoryLayout = control()->memoryLayout();

  if (signature->hasSize)
    appendArgument(1, memoryLayout->sizeOf(classSymbol->type()).value_or(0));

  if (signature->hasAlignment)
    appendArgument(signature->hasSize ? 2 : 1,
                   memoryLayout->alignmentOf(classSymbol->type()).value_or(1));

  auto stmt = ExpressionStatementAST::create(pool);
  stmt->expression = call;
  appendStatement(stmt);

  auto compound = CompoundStatementAST::create(pool);
  compound->statementList = stmts;

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = compound;

  auto dtorId = makeDestructorId(classSymbol);

  binder.attachSynthesizedBody(variant, dtorId, body);
  dtor->setDeletingDtorVariant(variant);
}

void Binder::CompleteClass::synthesizeMemberwiseBodies() {
  if (classSymbol->isUnion()) return;

  auto needsBody = [&](FunctionSymbol* fn) {
    if (!fn || fn->isDeleted()) return false;
    auto def = fn->declaration();
    return def && ast_cast<DefaultFunctionBodyAST>(def->functionBody);
  };

  auto needsBodyNow = [&](FunctionSymbol* fn) {
    if (!needsBody(fn)) return false;
    (void)ensureSourceParameter(fn);
    return fn->isDefinitionRequired();
  };

  for (auto fn : classSymbol->declaredConstructors()) {
    if (fn->inheritedConstructor() && needsBody(fn))
      synthesizeInheritedConstructorBody(fn);
  }

  if (auto fn = classSymbol->copyConstructor(); needsBodyNow(fn))
    synthesizeCopyMoveCtorBody(fn, /*isMove=*/false);
  if (auto fn = classSymbol->moveConstructor(); needsBodyNow(fn))
    synthesizeCopyMoveCtorBody(fn, /*isMove=*/true);
  if (auto fn = classSymbol->copyAssignmentOperator(); needsBodyNow(fn))
    synthesizeCopyMoveAssignBody(fn, /*isMove=*/false);
  if (auto fn = classSymbol->moveAssignmentOperator(); needsBodyNow(fn))
    synthesizeCopyMoveAssignBody(fn, /*isMove=*/true);
}

void Binder::CompleteClass::typeFieldInitializers() {
  for (auto field : views::members(classSymbol) | views::fields) {
    if (field->hasPendingInitializer()) continue;

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

void Binder::CompleteClass::synthesizeDefaultConstructorBody(
    FunctionSymbol* fn) {
  auto def = fn->declaration();
  if (!def || !ast_cast<DefaultFunctionBodyAST>(def->functionBody)) return;

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->statement = CompoundStatementAST::create(pool);
  def->functionBody = body;

  TypeChecker check{binder.unit_};
  check.setScope(fn);
  check.setReportErrors(false);
  check.check_mem_initializers(body);
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
    auto baseSym = resolved_base_class(base);
    if (!baseSym) continue;

    auto init = ParenMemInitializerAST::create(pool);
    if (auto id = name_cast<Identifier>(baseSym->name()))
      init->unqualifiedId = NameIdAST::create(pool, id);
    init->symbol = base;

    auto cast = ImplicitCastExpressionAST::create(pool);
    cast->castKind = ImplicitCastKind::kDerivedToBaseConversion;
    cast->expression = makeParamRef(param);
    cast->type =
        isMove ? baseSym->type()
               : control()->getQualType(baseSym->type(), CvQualifiers::kConst);
    cast->valueCategory =
        isMove ? ValueCategory::kXValue : ValueCategory::kLValue;
    init->expressionList = make_list_node<ExpressionAST>(pool, cast);
    append(init);
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!field->name() && field->isBitField()) continue;

    auto init = ParenMemInitializerAST::create(pool);
    if (auto id = name_cast<Identifier>(field->name()))
      init->unqualifiedId = NameIdAST::create(pool, id);
    init->symbol = field;
    init->expressionList = make_list_node<ExpressionAST>(
        pool, copiedMemberSource(field, param, isMove));
    append(init);
  }

  auto body = CompoundStatementFunctionBodyAST::create(pool);
  body->memInitializerList = memInits;
  body->statement = CompoundStatementAST::create(pool);
  def->functionBody = body;

  TypeChecker check{binder.unit_};
  check.setScope(fn);
  check.setReportErrors(false);
  check.check_mem_initializers(body, ArrayCopyPolicy::kElementwiseCopyAllowed);
}

auto Binder::CompleteClass::copiedMemberSource(FieldSymbol* field,
                                               ParameterSymbol* source,
                                               bool isMove) -> ExpressionAST* {
  auto member = memberSubobject(makeParamRef(source), field);
  if (!field->name()) {
    auto representation = ImplicitCastExpressionAST::create(pool);
    representation->castKind = ImplicitCastKind::kLValueToRValueConversion;
    representation->expression = member;
    representation->type = binder.traits.remove_cv(member->type);
    representation->valueCategory = ValueCategory::kPrValue;
    return representation;
  }
  if (binder.traits.is_reference(field->type())) return member;
  return makeSourceSubobjectRef(member, member->type, isMove);
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
    auto stmt = ExpressionStatementAST::create(pool);
    stmt->expression = assign;
    if (resolve) {
      check.check(&stmt->expression);
    } else {
      assign->type = lhs->type;
      assign->valueCategory = ValueCategory::kLValue;
    }
    append(stmt);
  };

  for (auto base : classSymbol->baseClasses()) {
    auto baseSym = resolved_base_class(base);
    if (!baseSym) continue;
    append(baseAssignmentStatement(check, baseSym, param, isMove));
  }

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    if (!field->name() && field->isBitField()) continue;

    if (assignsArrayElementwise(field, isMove)) {
      append(arrayMemberAssignment(check, fn, field, param, isMove));
      continue;
    }

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
    access->type =
        isMove ? fieldType : traits.add_cv(fieldType, CvQualifiers::kConst);
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

  auto returnStmt = ReturnStatementAST::create(pool);
  returnStmt->expression = makeSelfExpr();
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
  std::uint64_t runningSizeof = 0;
  std::uint64_t emptyComponentEnd = 0;
  std::uint64_t emittedEnd = 0;
  std::uint32_t currentIndex = 0;

  int nextBitPos = 0;
  int runStartByte = 0;
  std::uint32_t runIndex = 0;
  bool inBitfieldRun = false;
  std::vector<FieldSymbol*> runFields;
  ClassSubobjectList placedClassSubobjects;
  std::uint64_t maxPlacedSubobjectOffset = 0;
  std::vector<std::pair<ClassSymbol*, ClassLayout::MemberInfo>>
      indirectPrimaryPlacements;

  int packValue = 0;

  BuildRecordLayout(Binder& b, ClassSymbol* cls)
      : binder(b),
        classSymbol(cls),
        memoryLayout(b.control()->memoryLayout()),
        layout(std::make_unique<ClassLayout>()) {
    packValue = cls->packAlignment();
  }

  auto control() const -> Control* { return binder.control(); }

  auto operator()() -> std::expected<bool, std::string>;
  auto validate() -> std::expected<bool, std::string>;
  void completeFieldTypes();
  [[nodiscard]] auto computeAbiEmpty() const -> bool;
  [[nodiscard]] auto hasOnlyZeroSizeDataMembers(ClassSymbol* candidate) const
      -> bool;
  [[nodiscard]] auto isNearlyEmptyClass(ClassSymbol* classSymbol) const -> bool;
  [[nodiscard]] auto selectPrimaryBase() const -> std::pair<ClassSymbol*, bool>;
  void padTo(std::uint64_t offset);
  void layoutVtable();
  void layoutBases();
  void layoutVirtualBases();
  void recordIndirectPrimaryPlacement(ClassSymbol* primary,
                                      ClassLayout::MemberInfo info);
  [[nodiscard]] auto indirectPrimaryPlacement(ClassSymbol* primary) const
      -> std::optional<ClassLayout::MemberInfo>;
  void recordPrimaryChain(ClassSymbol* cls, std::uint64_t offset,
                          std::uint32_t topIndex);
  void collectIndirectPrimaryPlacements(ClassSymbol* root,
                                        std::uint64_t rootOffset,
                                        std::uint32_t rootIndex);
  [[nodiscard]] auto baseNonVirtualSize(ClassSymbol* base) -> std::uint64_t;
  [[nodiscard]] auto allocateBaseSubobject(ClassSymbol* base, bool isVirtual)
      -> ClassLayout::MemberInfo;
  void growSizeof(std::uint64_t offset, std::uint64_t sizeInBytes);
  void recordEmptyComponent(std::uint64_t offset, std::uint64_t sizeInBytes);
  [[nodiscard]] static auto subobjectCovers(const ClassSubobject& subobject,
                                            std::uint64_t address) -> bool;

  [[nodiscard]] static auto subobjectLast(const ClassSubobject& subobject)
      -> std::uint64_t;

  [[nodiscard]] auto placedAt(ClassSymbol* symbol, std::uint64_t address) const
      -> bool;

  [[nodiscard]] auto conflictsAt(const ClassSubobject& candidate,
                                 std::size_t level, std::uint64_t address) const
      -> bool;

  [[nodiscard]] auto classSubobjectOffset(ClassSymbol* classSymbol,
                                          bool tryZero, std::uint64_t alignment)
      -> std::uint64_t;
  void recordNonVirtualClassSubobjects(ClassSymbol* classSymbol,
                                       std::uint64_t offset,
                                       ClassSubobjectExtent extent = {});
  auto layoutFields() -> std::expected<bool, std::string>;
  auto layoutBitfield(FieldSymbol* field) -> std::expected<bool, std::string>;
  void layoutZeroWidthBitfield(FieldSymbol* field);
  auto layoutRegularField(FieldSymbol* field)
      -> std::expected<bool, std::string>;
  void closeBitfieldRun();
  [[nodiscard]] auto isPackedClass() const -> bool;
  [[nodiscard]] auto packAlignment(int alignment) const -> int;
  [[nodiscard]] auto keepsBitFieldInAllocationUnit(FieldSymbol* field) const
      -> bool;
  [[nodiscard]] auto alignedAttributeOfClass() -> std::optional<int>;
  void propagateBaseFields();
  void propagateAnonymousFields(ClassSymbol* owner,
                                const ClassLayout* ownerLayout,
                                std::uint64_t ownerOffset);
  void copyFieldInfos(ClassSymbol* owner, const ClassLayout* ownerLayout,
                      std::uint64_t ownerOffset);
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

  completeFieldTypes();

  layout->setAbiEmpty(computeAbiEmpty());
  layoutVtable();
  layoutBases();

  auto fieldsStatus = layoutFields();
  if (!fieldsStatus) return fieldsStatus;
  if (!fieldsStatus.value()) return false;

  auto nonVirtualSize = static_cast<std::uint64_t>(calculatedSize);
  if (emptyComponentEnd > nonVirtualSize) nonVirtualSize = emptyComponentEnd;

  layout->setNonVirtualSize(nonVirtualSize);
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

void Binder::BuildRecordLayout::completeFieldTypes() {
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
    auto fieldElementType =
        binder.traits.remove_cv(binder.traits.remove_all_extents(
            binder.traits.remove_cv(field->type())));

    auto classType = type_cast<ClassType>(fieldElementType);
    if (!classType) continue;

    binder.traits.requireCompleteClass(classType->symbol());

    if (auto alignment =
            binder.control()->memoryLayout()->alignmentOf(field->type())) {
      field->setAlignment(alignment.value());
    }
  }
}

auto Binder::BuildRecordLayout::hasOnlyZeroSizeDataMembers(
    ClassSymbol* candidate) const -> bool {
  return std::ranges::all_of(
      views::members(candidate) | views::non_static_fields,
      [this](FieldSymbol* field) {
        return binder.traits.is_zero_size_subobject(field);
      });
}

auto Binder::BuildRecordLayout::computeAbiEmpty() const -> bool {
  if (views::any_function(classSymbol->members(),
                          [](FunctionSymbol* f) { return f->isVirtual(); }))
    return false;

  if (!hasOnlyZeroSizeDataMembers(classSymbol)) return false;

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
        auto baseClass = resolved_base_class(base);
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
    layout->setVirtualBaseInfo(primaryBase, primaryInfo);
    layout->setHasDirectVtable(true);
    layout->setVtableIndex(primaryInfo.index);
    recordNonVirtualClassSubobjects(primaryBase, 0);
  } else {
    layout->setHasDirectVtable(true);
    layout->setVtableIndex(currentIndex++);
  }

  auto ptrSize = static_cast<int>(memoryLayout->sizeOfPointer());
  calculatedSize = ptrSize;
  calculatedAlignment = packAlignment(isPackedClass() ? 1 : ptrSize);
  emittedEnd = static_cast<std::uint64_t>(ptrSize);
  nextBitPos = calculatedSize * 8;
}

auto Binder::BuildRecordLayout::isNearlyEmptyClass(ClassSymbol* candidate) const
    -> bool {
  if (!candidate) return false;
  candidate = candidate->resolvedDefinition();
  auto candidateLayout = candidate->layout();
  if (!candidateLayout || !candidateLayout->hasVtable()) return false;

  if (!hasOnlyZeroSizeDataMembers(candidate)) return false;

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
      auto baseClass = resolved_base_class(base);
      if (!baseClass) continue;
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
    auto baseClass = resolved_base_class(base);
    if (baseClass && baseClass->layout() && baseClass->layout()->hasVtable())
      return {baseClass, false};
  }

  auto indirectPrimaryBases = indirectPrimaryBasesOf(classSymbol);

  std::vector<ClassSymbol*> candidates;
  for (auto virtualBase : virtualBasesInInheritanceGraphOrder(classSymbol)) {
    if (isNearlyEmptyClass(virtualBase)) candidates.push_back(virtualBase);
  }

  auto candidate = std::ranges::find_if(candidates, [&](ClassSymbol* base) {
    return !std::ranges::contains(indirectPrimaryBases, base);
  });
  if (candidate != candidates.end()) return {*candidate, true};
  if (!candidates.empty()) return {candidates.front(), true};
  return {nullptr, false};
}

auto Binder::BuildRecordLayout::baseNonVirtualSize(ClassSymbol* base)
    -> std::uint64_t {
  if (!base->layout()) return base->sizeInBytes();
  return binder.traits.non_virtual_size(base->type());
}

auto Binder::fieldElementClass(FieldSymbol* field) -> ClassSymbol* {
  auto elementType = traits.remove_cv(traits.remove_all_extents(field->type()));
  auto classType = type_cast<ClassType>(elementType);
  if (!classType || !classType->symbol()) return nullptr;
  return classType->symbol()->resolvedDefinition();
}

auto Binder::fieldArrayExtent(FieldSymbol* field) -> ClassSubobjectExtent {
  ClassSubobjectExtent extent;
  if (!traits.is_array(field->type())) return extent;

  auto elementType = traits.remove_cv(traits.remove_all_extents(field->type()));
  auto memoryLayout = control()->memoryLayout();

  const auto elementSize = memoryLayout->sizeOf(elementType);
  const auto totalSize = memoryLayout->sizeOf(field->type());

  if (!elementSize.has_value() || !totalSize.has_value()) return extent;
  if (!elementSize.value()) return extent;

  extent.stride = elementSize.value();
  extent.count = totalSize.value() / elementSize.value();
  return extent;
}

void Binder::appendClassSubobjects(ClassSubobjectList& subobjects,
                                   ClassSymbol* classSymbol,
                                   std::uint64_t offset,
                                   ClassSubobjectExtent extent) {
  if (!extent.count) return;

  for (const auto& subobject : emptyClassSubobjects(classSymbol)) {
    auto& merged = subobjects.emplace_back(subobject);
    merged.offset += offset;
    if (extent.count > 1) merged.extents.insert(merged.extents.begin(), extent);
  }
}

auto Binder::emptyClassSubobjects(ClassSymbol* classSymbol)
    -> const ClassSubobjectList& {
  classSymbol = classSymbol->resolvedDefinition();

  const auto layout = classSymbol->layout();

  if (auto it = emptyClassSubobjects_.find(classSymbol);
      it != emptyClassSubobjects_.end() && it->second.layout == layout) {
    return it->second.subobjects;
  }

  ClassSubobjectList subobjects;

  if (layout && layout->isAbiEmpty()) {
    subobjects.emplace_back(classSymbol, 0,
                            std::vector<ClassSubobjectExtent>{});
  }

  if (layout) {
    for (auto base : classSymbol->baseClasses()) {
      if (base->isVirtual()) continue;
      auto baseClass = resolved_base_class(base);
      if (!baseClass) continue;
      if (auto baseInfo = layout->getBaseInfo(baseClass)) {
        appendClassSubobjects(subobjects, baseClass, baseInfo->offset,
                              ClassSubobjectExtent{});
      }
    }

    for (auto field : views::members(classSymbol) | views::non_static_fields) {
      auto fieldClass = fieldElementClass(field);
      if (!fieldClass || !fieldClass->layout()) continue;
      auto fieldInfo = layout->getFieldInfo(field);
      if (!fieldInfo) continue;

      appendClassSubobjects(subobjects, fieldClass, fieldInfo->offset,
                            fieldArrayExtent(field));
    }
  }

  auto& entry = emptyClassSubobjects_[classSymbol];
  entry.layout = layout;
  entry.subobjects = std::move(subobjects);
  return entry.subobjects;
}

auto Binder::BuildRecordLayout::subobjectCovers(const ClassSubobject& subobject,
                                                std::uint64_t address) -> bool {
  if (address < subobject.offset) return false;
  auto displacement = address - subobject.offset;
  for (const auto& extent : subobject.extents) {
    if (!extent.stride) continue;
    const auto index = displacement / extent.stride;
    if (index >= extent.count) return false;
    displacement -= index * extent.stride;
  }
  return displacement == 0;
}

auto Binder::BuildRecordLayout::subobjectLast(const ClassSubobject& subobject)
    -> std::uint64_t {
  auto last = subobject.offset;
  for (const auto& extent : subobject.extents)
    last += (extent.count - 1) * extent.stride;
  return last;
}

auto Binder::BuildRecordLayout::placedAt(ClassSymbol* symbol,
                                         std::uint64_t address) const -> bool {
  return std::ranges::any_of(
      placedClassSubobjects, [&](const ClassSubobject& placed) {
        return placed.symbol == symbol && subobjectCovers(placed, address);
      });
}

auto Binder::BuildRecordLayout::conflictsAt(const ClassSubobject& candidate,
                                            std::size_t level,
                                            std::uint64_t address) const
    -> bool {
  if (level == candidate.extents.size())
    return placedAt(candidate.symbol, address);

  const auto& extent = candidate.extents[level];
  for (std::uint64_t index = 0; index != extent.count; ++index) {
    const auto elementAddress = address + index * extent.stride;
    if (elementAddress > maxPlacedSubobjectOffset) break;
    if (conflictsAt(candidate, level + 1, elementAddress)) return true;
  }
  return false;
}

auto Binder::BuildRecordLayout::classSubobjectOffset(ClassSymbol* target,
                                                     bool tryZero,
                                                     std::uint64_t alignment)
    -> std::uint64_t {
  const auto& subobjects = binder.emptyClassSubobjects(target);

  auto conflicts = [&](std::uint64_t offset) {
    return std::ranges::any_of(
        subobjects, [&](const ClassSubobject& candidate) {
          return conflictsAt(candidate, 0, offset + candidate.offset);
        });
  };

  if (tryZero && !conflicts(0)) return 0;

  auto offset = align_to(calculatedSize, alignment);
  while (conflicts(offset)) offset += alignment;
  return offset;
}

void Binder::BuildRecordLayout::recordNonVirtualClassSubobjects(
    ClassSymbol* target, std::uint64_t offset, ClassSubobjectExtent extent) {
  const auto first = placedClassSubobjects.size();

  binder.appendClassSubobjects(placedClassSubobjects, target, offset, extent);

  for (auto index = first; index != placedClassSubobjects.size(); ++index) {
    maxPlacedSubobjectOffset = std::max(
        maxPlacedSubobjectOffset, subobjectLast(placedClassSubobjects[index]));
  }
}

void Binder::BuildRecordLayout::growSizeof(std::uint64_t offset,
                                           std::uint64_t sizeInBytes) {
  runningSizeof = std::max(runningSizeof, offset + sizeInBytes);
}

void Binder::BuildRecordLayout::recordEmptyComponent(
    std::uint64_t offset, std::uint64_t sizeInBytes) {
  growSizeof(offset, sizeInBytes);
  emptyComponentEnd = std::max(emptyComponentEnd, offset + sizeInBytes);
}

auto Binder::BuildRecordLayout::allocateBaseSubobject(ClassSymbol* base,
                                                      bool isVirtual)
    -> ClassLayout::MemberInfo {
  auto baseLayout = base->layout();

  auto baseAlignment = static_cast<int>(base->alignment());
  if (baseLayout && !baseLayout->virtualBases().empty()) {
    baseAlignment = static_cast<int>(baseLayout->nonVirtualAlignment());
  }
  baseAlignment = packAlignment(baseAlignment);

  const auto baseSizeInBytes = static_cast<int>(baseNonVirtualSize(base));
  const bool isEmpty = baseLayout && baseLayout->isAbiEmpty();

  const auto baseOffset =
      classSubobjectOffset(base, isEmpty, std::max(baseAlignment, 1));

  if (!isEmpty) padTo(baseOffset);

  ClassLayout::MemberInfo baseInfo;
  baseInfo.offset = baseOffset;
  baseInfo.index = currentIndex++;
  if (isVirtual)
    layout->setVirtualBaseInfo(base, baseInfo);
  else
    layout->setBaseInfo(base, baseInfo);

  if (isEmpty) {
    const auto emptySize = memoryLayout->sizeOf(base->type()).value_or(1);
    if (isVirtual) {
      growSizeof(baseOffset, emptySize);
    } else {
      recordEmptyComponent(baseOffset, emptySize);
    }
  } else {
    calculatedSize = std::max(calculatedSize,
                              static_cast<int>(baseOffset) + baseSizeInBytes);
    emittedEnd = std::max(emittedEnd, baseOffset + baseSizeInBytes);
  }

  recordNonVirtualClassSubobjects(base, baseOffset);
  calculatedAlignment = std::max(calculatedAlignment, baseAlignment);

  return baseInfo;
}

void Binder::BuildRecordLayout::layoutBases() {
  if (classSymbol->isUnion()) return;

  ClassSymbol* primaryBase = nullptr;
  if (!layout->primaryBaseIsVirtual()) primaryBase = layout->primaryBase();

  std::vector<ClassSymbol*> orderedBases;
  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClassSymbol = resolved_base_class(base);
    if (!baseClassSymbol) continue;
    if (baseClassSymbol == primaryBase) {
      orderedBases.insert(orderedBases.begin(), baseClassSymbol);
    } else {
      orderedBases.push_back(baseClassSymbol);
    }
  }

  for (auto baseClassSymbol : orderedBases) {
    const auto baseInfo = allocateBaseSubobject(baseClassSymbol, false);

    if (baseClassSymbol == primaryBase) layout->setVtableIndex(baseInfo.index);
  }

  nextBitPos = calculatedSize * 8;
}

void Binder::BuildRecordLayout::layoutVirtualBases() {
  if (classSymbol->isUnion()) return;

  if (layout->primaryBaseIsVirtual()) {
    auto primary = layout->primaryBase();
    if (auto info = layout->getVirtualBaseInfo(primary)) {
      recordIndirectPrimaryPlacement(primary, {info->offset, info->index});
      recordPrimaryChain(primary, info->offset, info->index);
    }
  }

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = resolved_base_class(base);
    if (!baseClass) continue;
    if (auto info = layout->getBaseInfo(baseClass))
      collectIndirectPrimaryPlacements(baseClass, info->offset, info->index);
  }

  for (auto virtualBase : virtualBasesInInheritanceGraphOrder(classSymbol)) {
    if (auto placement = indirectPrimaryPlacement(virtualBase)) {
      layout->setVirtualBaseInfo(virtualBase, *placement);
      layout->addVirtualBase(virtualBase);
      recordNonVirtualClassSubobjects(virtualBase, placement->offset);
      recordPrimaryChain(virtualBase, placement->offset, placement->index);
      continue;
    }

    const auto baseInfo = allocateBaseSubobject(virtualBase, true);

    layout->addVirtualBase(virtualBase);
    recordPrimaryChain(virtualBase, baseInfo.offset, baseInfo.index);
  }

  nextBitPos = calculatedSize * 8;
}

void Binder::BuildRecordLayout::recordIndirectPrimaryPlacement(
    ClassSymbol* primary, ClassLayout::MemberInfo info) {
  if (indirectPrimaryPlacement(primary)) return;
  indirectPrimaryPlacements.emplace_back(primary, info);
}

auto Binder::BuildRecordLayout::indirectPrimaryPlacement(
    ClassSymbol* primary) const -> std::optional<ClassLayout::MemberInfo> {
  for (const auto& [base, info] : indirectPrimaryPlacements) {
    if (base == primary) return info;
  }
  return std::nullopt;
}

void Binder::BuildRecordLayout::recordPrimaryChain(ClassSymbol* cls,
                                                   std::uint64_t offset,
                                                   std::uint32_t topIndex) {
  while (cls) {
    auto classLayout = cls->layout();
    if (!classLayout || !classLayout->primaryBase()) return;
    auto primary = classLayout->primaryBase();
    auto info =
        classLayout->getBaseInfo(primary, classLayout->primaryBaseIsVirtual());
    if (!info) return;
    const auto primaryOffset = offset + info->offset;
    if (classLayout->primaryBaseIsVirtual())
      recordIndirectPrimaryPlacement(primary, {primaryOffset, topIndex});
    cls = primary;
    offset = primaryOffset;
  }
}

void Binder::BuildRecordLayout::collectIndirectPrimaryPlacements(
    ClassSymbol* root, std::uint64_t rootOffset, std::uint32_t rootIndex) {
  struct PlacementWork {
    ClassSymbol* classSymbol;
    std::uint64_t offset;
  };

  std::vector<PlacementWork> pending{{root, rootOffset}};
  while (!pending.empty()) {
    auto [cls, offset] = pending.back();
    pending.pop_back();
    auto classLayout = cls->layout();
    if (!classLayout) continue;
    recordPrimaryChain(cls, offset, rootIndex);
    for (auto base : cls->baseClasses() | std::views::reverse) {
      if (base->isVirtual()) continue;
      auto baseClass = resolved_base_class(base);
      if (!baseClass) continue;
      if (auto info = classLayout->getBaseInfo(baseClass))
        pending.push_back({baseClass, offset + info->offset});
    }
  }
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
  emittedEnd = std::max(emittedEnd, static_cast<std::uint64_t>(calculatedSize));
  currentIndex++;
}

auto Binder::BuildRecordLayout::layoutBitfield(FieldSymbol* field)
    -> std::expected<bool, std::string> {
  const bool isUnion = classSymbol->isUnion();

  int bitWidth = 0;
  if (auto& bfw = field->bitFieldWidth()) {
    if (auto iv = std::get_if<ConstInt>(&*bfw)) {
      bitWidth = static_cast<int>(iv->toIntMax());
    }
  }

  if (bitWidth == 0) {
    layoutZeroWidthBitfield(field);
    return true;
  }

  auto fieldAlign = field->effectiveAlignment();
  auto fieldSizeBytes =
      static_cast<int>(memoryLayout->sizeOf(field->type()).value_or(0));
  auto fieldSizeBits = fieldSizeBytes * 8;

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

  if (fieldSizeBits > 0 && keepsBitFieldInAllocationUnit(field)) {
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
    padTo(static_cast<std::uint64_t>(runStartByte));
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

void Binder::BuildRecordLayout::layoutZeroWidthBitfield(FieldSymbol* field) {
  if (inBitfieldRun) closeBitfieldRun();

  if (memoryLayout->zeroWidthBitFieldAlignsAggregate())
    calculatedAlignment = std::max(calculatedAlignment, field->alignment());

  if (classSymbol->isUnion()) return;
  if (!memoryLayout->sizeOf(field->type()).value_or(0)) return;

  nextBitPos = align_to(nextBitPos, field->alignment() * 8);
  calculatedSize = (nextBitPos + 7) / 8;
}

auto Binder::BuildRecordLayout::layoutRegularField(FieldSymbol* field)
    -> std::expected<bool, std::string> {
  const bool isUnion = classSymbol->isUnion();

  closeBitfieldRun();

  const bool isEmptyDataMember = binder.traits.is_zero_size_subobject(field);

  std::optional<std::size_t> size;
  if (binder.traits.is_unbounded_array(field->type())) {
    size = 0;
  } else if (isEmptyDataMember) {
    size = 0;
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
    auto fieldAlign = field->effectiveAlignment();
    auto fieldOffset =
        static_cast<std::uint64_t>(align_to(calculatedSize, fieldAlign));
    auto elementClass = binder.fieldElementClass(field);
    if (elementClass) {
      fieldOffset = classSubobjectOffset(elementClass, isEmptyDataMember,
                                         std::max(fieldAlign, 1));
    }
    field->setLocalOffset(static_cast<int>(fieldOffset));

    if (!isEmptyDataMember) padTo(fieldOffset);

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = fieldOffset;
    fieldInfo.index = currentIndex++;
    layout->setFieldInfo(field, fieldInfo);

    if (!isEmptyDataMember) {
      auto allocatedExtent = static_cast<std::uint64_t>(size.value());
      auto emittedExtent = allocatedExtent;

      if (field->isNoUniqueAddress() &&
          unqualified_cast<ClassType>(field->type())) {
        emittedExtent = binder.traits.non_virtual_size(field->type());
        allocatedExtent =
            std::max(emittedExtent, binder.traits.data_size(field->type()));
      }

      calculatedSize = std::max(
          calculatedSize, static_cast<int>(fieldOffset + allocatedExtent));
      emittedEnd = std::max(emittedEnd, fieldOffset + emittedExtent);
    }

    if (field->isNoUniqueAddress()) {
      const auto memberSize = memoryLayout->sizeOf(field->type()).value_or(0);
      if (isEmptyDataMember) {
        recordEmptyComponent(fieldOffset, memberSize);
      } else {
        growSizeof(fieldOffset, memberSize);
      }
    }

    if (elementClass) {
      recordNonVirtualClassSubobjects(elementClass, fieldOffset,
                                      binder.fieldArrayExtent(field));
    }
  }

  nextBitPos = calculatedSize * 8;

  calculatedAlignment =
      std::max(calculatedAlignment, field->effectiveAlignment());
  return true;
}

auto Binder::BuildRecordLayout::isPackedClass() const -> bool {
  return findAttribute(classSymbol->attributes(), "packed") != nullptr;
}

auto Binder::BuildRecordLayout::packAlignment(int alignment) const -> int {
  if (packValue <= 0) return alignment;
  return std::min(alignment, packValue);
}

auto Binder::BuildRecordLayout::keepsBitFieldInAllocationUnit(
    FieldSymbol* field) const -> bool {
  if (packValue > 0) return false;
  return !field->isPacked();
}

auto Binder::BuildRecordLayout::alignedAttributeOfClass()
    -> std::optional<int> {
  auto specifier = ast_cast<ClassSpecifierAST>(classSymbol->declaration());
  if (!specifier) return std::nullopt;
  auto head = binder.alignedAttribute(specifier->attributeList);
  auto trailing = binder.alignedAttribute(specifier->trailingAttributeList);
  if (!head) return trailing;
  if (!trailing) return head;
  return std::max(*head, *trailing);
}

auto Binder::BuildRecordLayout::layoutFields()
    -> std::expected<bool, std::string> {
  FieldSymbol* lastField = nullptr;

  for (auto field : views::members(classSymbol) | views::non_static_fields) {
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
    auto baseClassSymbol = resolved_base_class(base);
    if (!baseClassSymbol) continue;

    auto baseLayout = baseClassSymbol->layout();
    if (!baseLayout) continue;

    auto baseInfo = layout->getBaseInfo(baseClassSymbol, base->isVirtual());
    if (!baseInfo) continue;

    copyFieldInfos(baseClassSymbol, baseLayout, baseInfo->offset);
  }

  propagateAnonymousFields(classSymbol, layout.get(), 0);
}

void Binder::BuildRecordLayout::propagateAnonymousFields(
    ClassSymbol* owner, const ClassLayout* ownerLayout,
    std::uint64_t ownerOffset) {
  for (auto field : views::members(owner) | views::non_static_fields) {
    auto anonymous = anonymous_member_class(field);
    if (!anonymous) continue;

    auto anonymousLayout = anonymous->layout();
    if (!anonymousLayout) continue;

    auto fieldInfo = ownerLayout->getFieldInfo(field);
    if (!fieldInfo) continue;

    const auto anonymousOffset = ownerOffset + fieldInfo->offset;
    copyFieldInfos(anonymous, anonymousLayout, anonymousOffset);
    propagateAnonymousFields(anonymous, anonymousLayout, anonymousOffset);
  }
}

void Binder::BuildRecordLayout::copyFieldInfos(ClassSymbol* owner,
                                               const ClassLayout* ownerLayout,
                                               std::uint64_t ownerOffset) {
  for (auto field : views::members(owner) | views::non_static_fields) {
    auto info = ownerLayout->getFieldInfo(field);
    if (!info) continue;
    info->offset += ownerOffset;
    layout->setFieldInfo(field, *info);
  }
}

void Binder::BuildRecordLayout::padTo(std::uint64_t offset) {
  if (offset <= emittedEnd) return;
  layout->addPadding(currentIndex++, emittedEnd, offset - emittedEnd);
  emittedEnd = offset;
}

void Binder::BuildRecordLayout::finalize() {
  if (auto requested = classSymbol->explicitAlignment()) {
    if (requested < calculatedAlignment) {
      binder.error(
          classSymbol->location(),
          std::format("requested alignment is less than minimum "
                      "alignment of {} for type '{}'",
                      calculatedAlignment, to_string(classSymbol->type())));
    } else {
      calculatedAlignment = requested;
    }
  }

  if (auto requested = alignedAttributeOfClass())
    calculatedAlignment = std::max(calculatedAlignment, *requested);

  const auto dataSize = static_cast<std::uint64_t>(calculatedSize);

  calculatedSize = std::max(calculatedSize, static_cast<int>(runningSizeof));

  if (calculatedSize == 0) calculatedSize = 1;

  calculatedSize = align_to(calculatedSize, calculatedAlignment);

  padTo(static_cast<std::uint64_t>(calculatedSize));

  classSymbol->setAlignment(calculatedAlignment);
  classSymbol->setSizeInBytes(calculatedSize);

  layout->setSize(calculatedSize);
  layout->setAlignment(calculatedAlignment);
  layout->setDataSize(dataSize);

  binder.emptyClassSubobjects_.erase(classSymbol->resolvedDefinition());
  classSymbol->setLayout(std::move(layout));

  buildVTableLayout();
}

void Binder::BuildRecordLayout::buildVTableLayout() {
  auto classLayout = classSymbol->layout();
  if (!classLayout || !classLayout->hasVtable()) return;
  binder.buildVTableLayout(classSymbol);
}
}  // namespace cxx
