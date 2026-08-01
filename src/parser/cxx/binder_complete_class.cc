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
#include <cxx/binder.h>
#include <cxx/control.h>
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
struct [[nodiscard]] Binder::CompleteClass {
  Binder& binder;
  ClassSpecifierAST* ast;
  ClassSymbol* classSymbol;
  Arena* pool;

  CompleteClass(Binder& b, ClassSpecifierAST* a)
      : binder(b), ast(a), classSymbol(a->symbol), pool(b.unit_->arena()) {}

  CompleteClass(Binder& b, ClassSymbol* cls)
      : binder(b), ast(nullptr), classSymbol(cls), pool(b.unit_->arena()) {}

  auto control() const -> Control* { return binder.control(); }

  void complete();

  void markComplete();
  auto shouldSynthesizeSpecialMembers() const -> bool;
  void synthesizeSpecialMembers();
  auto hasVirtualBaseDestructor() const -> bool;

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
  [[nodiscard]] auto hasNonAssignableSubobject(bool moveForm) const -> bool;
  auto ensureSourceParameter(FunctionSymbol* fn) -> ParameterSymbol*;
  auto makeSourceSubobjectRef(ExpressionAST* expr, const Type* type,
                              bool isMove) -> ExpressionAST*;
  void synthesizeCopyMoveCtorBody(FunctionSymbol* fn, bool isMove);
  void synthesizeCopyMoveAssignBody(FunctionSymbol* fn, bool isMove);
};

void Binder::complete(ClassSpecifierAST* ast) {
  CompleteClass{*this, ast}.complete();
}

void Binder::synthesizeCompleteObjectCtor(FunctionSymbol* ctor) {
  if (!ctor->isConstructor()) return;
  if (ctor->isDeleted() || ctor->completeObjectVariant()) return;
  if (!type_cast<FunctionType>(ctor->type())) return;

  auto classSymbol =
      symbol_cast<ClassSymbol>(ctor->enclosingNonTemplateParametersScope());
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

  auto classSymbol =
      symbol_cast<ClassSymbol>(fn->enclosingNonTemplateParametersScope());
  if (!classSymbol) return;
  classSymbol = classSymbol->resolvedDefinition();
  if (classSymbol->isUnion() || classSymbol->isClosureType()) return;

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

void Binder::CompleteClass::markComplete() { ast->symbol->setComplete(true); }

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

  addDefaultConstructor();

  addCopyConstructor();
  if (deleteCopyMembers && !userDeclaredCopyConstructor) {
    if (auto copyConstructor = classSymbol->copyConstructor())
      copyConstructor->setDeleted(true);
  }

  if (!suppressMoveMembers) addMoveConstructor();

  addCopyAssignmentOperator();
  if (!userDeclaredCopyAssignment) {
    if (auto copyAssignment = classSymbol->copyAssignmentOperator()) {
      if (deleteCopyMembers || hasNonAssignableSubobject(/*moveForm=*/false))
        copyAssignment->setDeleted(true);
    }
  }

  if (!suppressMoveMembers) {
    addMoveAssignmentOperator();
    if (auto moveAssignment = classSymbol->moveAssignmentOperator()) {
      if (hasNonAssignableSubobject(/*moveForm=*/true))
        moveAssignment->setDeleted(true);
    }
  }

  addDestructor();
}

auto Binder::CompleteClass::hasVirtualBaseDestructor() const -> bool {
  for (auto base : classSymbol->baseClasses()) {
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (!baseClass) continue;
    baseClass = baseClass->resolvedDefinition();

    auto dtor = baseClass->destructor();
    if (dtor && dtor->isVirtual()) return true;
  }

  return false;
}

auto Binder::CompleteClass::buildRecordLayout()
    -> std::expected<bool, std::string> {
  return binder.buildRecordLayout(classSymbol);
}

void Binder::CompleteClass::complete() {
  auto isFullExplicitSpecialization = [&]() {
    if (!classSymbol->isSpecialization()) return false;
    auto tp = classSymbol->templateParameters();
    return tp && tp->isExplicitTemplateSpecialization();
  };

  if (binder.inTemplate() && !isFullExplicitSpecialization()) {
    markComplete();
    return;
  }

  if (shouldSynthesizeSpecialMembers()) synthesizeSpecialMembers();

  auto status = buildRecordLayout();
  if (!status.has_value())
    binder.error(classSymbol->location(), status.error());

  binder.computeClassFlags(classSymbol);

  typeFieldInitializers();

  if (shouldSynthesizeSpecialMembers()) {
    synthesizeStructorVariants();
    synthesizeMemberwiseBodies();
  }

  markComplete();
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
  if (!classSymbol->constructors().empty()) return;

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

  if (hasVirtualBaseDestructor()) symbol->setVirtual(true);

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
    for (auto ctor : classSymbol->constructors()) {
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
  for (auto candidate : vbase->constructors()) {
    auto funcType = type_cast<FunctionType>(candidate->type());
    if (funcType && funcType->parameterTypes().empty()) return candidate;
  }
  return nullptr;
}

void Binder::CompleteClass::synthesizeCompleteObjectCtor(FunctionSymbol* ctor) {
  auto layout = classSymbol->layout();
  auto traits = binder.traits;

  bool isCopy = false;
  bool isMove = false;
  ParameterSymbol* sourceParam = nullptr;

  auto variant = newStructorVariant(ctor);

  std::vector<ParameterSymbol*> params;
  for (auto member : views::members(variant)) {
    auto paramsSym = symbol_cast<FunctionParametersSymbol>(member);
    if (!paramsSym) continue;
    for (auto param : views::members(paramsSym)) {
      if (auto p = symbol_cast<ParameterSymbol>(param)) params.push_back(p);
    }
  }

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
  if (classSymbol->isClosureType()) return;

  auto needsBody = [&](FunctionSymbol* fn) {
    if (!fn || fn->isDeleted()) return false;
    auto def = fn->declaration();
    return def && ast_cast<DefaultFunctionBodyAST>(def->functionBody);
  };

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
  for (auto field : views::members(classSymbol) | views::non_static_fields) {
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
    const bool bitwiseCopy =
        !id || traits.is_array(traits.remove_cv(field->type()));
    if (bitwiseCopy) {
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
  void layoutVtable();
  void layoutBases();
  void layoutVirtualBases();
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

void Binder::BuildRecordLayout::layoutVtable() {
  if (classSymbol->isUnion()) return;

  bool hasPolymorphicBase = false;
  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseClass = symbol_cast<ClassSymbol>(base->symbol());
    if (auto def = baseClass ? baseClass->definition() : nullptr)
      baseClass = def;
    if (baseClass && baseClass->layout() && baseClass->layout()->hasVtable()) {
      hasPolymorphicBase = true;
      break;
    }
  }

  bool needsOwnVptr = false;
  if (!hasPolymorphicBase) {
    needsOwnVptr =
        views::any_function(classSymbol->members(),
                            [](FunctionSymbol* f) { return f->isVirtual(); });

    if (!needsOwnVptr) {
      for (auto base : classSymbol->baseClasses()) {
        if (base->isVirtual()) {
          needsOwnVptr = true;
          break;
        }
      }
    }
  }

  if (needsOwnVptr) {
    layout->setHasVtable(true);
    layout->setHasDirectVtable(true);
    layout->setVtableIndex(currentIndex++);

    auto ptrSize = static_cast<int>(memoryLayout->sizeOfPointer());
    calculatedSize = ptrSize;
    calculatedAlignment = ptrSize;
    nextBitPos = calculatedSize * 8;
  } else if (hasPolymorphicBase) {
    layout->setHasVtable(true);
    layout->setVtableIndex(0);
  }
}

void Binder::BuildRecordLayout::layoutBases() {
  if (classSymbol->isUnion()) return;

  bool foundPolymorphicBase = false;
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

    if (baseAlignment > 0) {
      calculatedSize = align_to(calculatedSize, baseAlignment);
    }

    ClassLayout::MemberInfo baseInfo;
    baseInfo.offset = calculatedSize;
    baseInfo.index = currentIndex++;
    layout->setBaseInfo(baseClassSymbol, baseInfo);

    if (!foundPolymorphicBase && baseLayout && baseLayout->hasVtable()) {
      layout->setVtableIndex(baseInfo.index);
      foundPolymorphicBase = true;
    }

    if (!binder.traits.is_empty(baseClassSymbol->type())) {
      calculatedSize += baseSizeInBytes;
    }
    calculatedAlignment = std::max(calculatedAlignment, baseAlignment);
  }

  nextBitPos = calculatedSize * 8;
}

void Binder::BuildRecordLayout::layoutVirtualBases() {
  if (classSymbol->isUnion()) return;

  std::vector<ClassSymbol*> orderedVirtualBases;
  std::unordered_set<ClassSymbol*> seenVirtualBases;

  std::function<void(ClassSymbol*)> visitBases = [&](ClassSymbol* cls) {
    for (auto base : cls->baseClasses()) {
      auto baseClassSymbol = symbol_cast<ClassSymbol>(base->symbol());
      if (!baseClassSymbol) continue;
      baseClassSymbol = baseClassSymbol->resolvedDefinition();
      if (base->isVirtual() &&
          seenVirtualBases.insert(baseClassSymbol).second) {
        orderedVirtualBases.push_back(baseClassSymbol);
      }
      visitBases(baseClassSymbol);
    }
  };
  visitBases(classSymbol);

  for (auto baseClassSymbol : orderedVirtualBases) {
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

    if (baseAlignment > 0) {
      calculatedSize = align_to(calculatedSize, baseAlignment);
    }

    ClassLayout::MemberInfo baseInfo;
    baseInfo.offset = calculatedSize;
    baseInfo.index = currentIndex++;
    layout->setBaseInfo(baseClassSymbol, baseInfo);
    layout->addVirtualBase(baseClassSymbol);

    if (!binder.traits.is_empty(baseClassSymbol->type())) {
      calculatedSize += baseSizeInBytes;
    }
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
  } else if (field->isNoUniqueAddress() &&
             binder.traits.is_empty(field->type())) {
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
    auto fieldAlign = field->alignment();
    if (packValue > 0) fieldAlign = std::min(fieldAlign, packValue);
    calculatedSize = align_to(calculatedSize, fieldAlign);
    field->setLocalOffset(calculatedSize);

    ClassLayout::MemberInfo fieldInfo;
    fieldInfo.offset = calculatedSize;
    fieldInfo.index = currentIndex++;
    layout->setFieldInfo(field, fieldInfo);

    calculatedSize += size.value();
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
    if (auto classType =
            type_cast<ClassType>(binder.traits.remove_cv(field->type()))) {
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

  auto typeTraits = binder.traits;

  auto resolvedClass = [](Symbol* symbol) -> ClassSymbol* {
    auto classSym = symbol_cast<ClassSymbol>(symbol);
    return classSym ? classSym->resolvedDefinition() : nullptr;
  };

  ClassSymbol* primaryBase = nullptr;
  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseSym = resolvedClass(base->symbol());
    if (baseSym && baseSym->layout() && baseSym->layout()->hasVtable()) {
      primaryBase = baseSym;
      break;
    }
  }

  auto& slots = vtable->primary.slots;
  if (primaryBase && primaryBase->vtableLayout()) {
    slots = primaryBase->vtableLayout()->primary.slots;
  }

  auto processFunc = [&](FunctionSymbol* func) {
    if (!func->isVirtual()) return;
    const bool isDtor = func->isDestructor();
    bool foundOverride = false;
    for (std::size_t i = 0; i < slots.size(); ++i) {
      const bool isOverride =
          isDtor
              ? slots[i].kind != VTableLayout::SlotKind::kFunction
              : slots[i].function->name() == func->name() &&
                    typeTraits.is_same(slots[i].function->type(), func->type());
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
      slots.push_back({func, isDtor ? VTableLayout::SlotKind::kCompleteDtor
                                    : VTableLayout::SlotKind::kFunction});
      if (isDtor)
        slots.push_back({func, VTableLayout::SlotKind::kDeletingDtor});
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

  std::unordered_map<const Name*, std::vector<FunctionSymbol*>> virtualsByName;
  for (auto member : classSymbol->members()) {
    for (auto func : views::each_function(member)) {
      if (func->isVirtual()) virtualsByName[func->name()].push_back(func);
    }
  }

  auto buildGroup =
      [&](ClassSymbol* baseSym, std::uint64_t offset,
          bool isVirtualBase) -> std::optional<VTableLayout::Group> {
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

    auto findOverride = [&](FunctionSymbol* baseFunc) -> FunctionSymbol* {
      if (baseFunc->isDestructor()) {
        auto dtor = classSymbol->destructor();
        return (dtor && dtor->isVirtual()) ? dtor : nullptr;
      }
      auto it = virtualsByName.find(baseFunc->name());
      if (it == virtualsByName.end()) return nullptr;
      for (auto func : it->second) {
        if (typeTraits.is_same(func->type(), baseFunc->type())) return func;
      }
      return nullptr;
    };

    std::unordered_map<FunctionSymbol*, FunctionSymbol*> overrideOf;
    for (auto& slot : group.slots) {
      if (slot.kind == VTableLayout::SlotKind::kDeletingDtor) continue;
      auto overrider = findOverride(slot.function);
      if (!overrider) continue;
      overrideOf.emplace(slot.function, overrider);
      if (isVirtualBase) {
        group.vcallOffsets.emplace_back(
            overrider, -static_cast<std::int64_t>(group.offset));
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

      slot.function = overrider;
      if (isVirtualBase) {
        slot.vcallOffsetIndex = vcallIndexOf.at(overrider);
      } else {
        slot.thisAdjustment = group.offset;
      }
    }

    return group;
  };

  for (auto base : classSymbol->baseClasses()) {
    if (base->isVirtual()) continue;
    auto baseSym = resolvedClass(base->symbol());
    if (!baseSym || baseSym == primaryBase) continue;

    auto baseInfo = classLayout->getBaseInfo(baseSym);
    if (!baseInfo) continue;

    if (auto group = buildGroup(baseSym, baseInfo->offset,
                                /*isVirtualBase=*/false))
      vtable->secondary.push_back(std::move(*group));
  }

  for (auto vbase : classLayout->virtualBases()) {
    auto vbaseInfo = classLayout->getBaseInfo(vbase);
    if (!vbaseInfo) continue;

    if (auto group =
            buildGroup(vbase, vbaseInfo->offset, /*isVirtualBase=*/true))
      vtable->secondary.push_back(std::move(*group));
  }

  classSymbol->setVTableLayout(std::move(vtable));
}
}  // namespace cxx
