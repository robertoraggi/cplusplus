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
#include <cxx/name_lookup.h>
#include <cxx/names.h>
#include <cxx/overload_resolution.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

#include <vector>

namespace cxx {
namespace {
auto rangeDeclarationVariable(DeclarationAST* rangeDeclaration)
    -> VariableSymbol* {
  auto simpleDecl = ast_cast<SimpleDeclarationAST>(rangeDeclaration);
  if (!simpleDecl || !simpleDecl->initDeclaratorList) return nullptr;

  auto initDeclarator = simpleDecl->initDeclaratorList->value;
  if (!initDeclarator) return nullptr;

  return symbol_cast<VariableSymbol>(initDeclarator->symbol);
}

[[nodiscard]] auto rangeElementCopyPolicy(
    StructuredBindingDeclarationAST* structuredBinding) -> ArrayCopyPolicy {
  if (structuredBinding) return ArrayCopyPolicy::kElementwiseCopyAllowed;
  return ArrayCopyPolicy::kBracedInitializerOnly;
}

[[nodiscard]] auto usesMemberBeginEnd(Control* control,
                                      ClassSymbol* classSymbol) -> bool {
  if (!qualifiedLookup(classSymbol, control->getIdentifier("begin")))
    return false;
  return qualifiedLookup(classSymbol, control->getIdentifier("end")) != nullptr;
}

auto resolveRangeIteration(TranslationUnit* unit, ForRangeStatementAST* ast,
                           const Type* rangeType) -> const Type* {
  auto traits = unit->typeTraits();

  if (auto arrayType = type_cast<BoundedArrayType>(rangeType)) {
    ast->isPointerIterator = true;
    return arrayType->elementType();
  }

  auto classType = type_cast<ClassType>(rangeType);
  if (!classType) return nullptr;

  auto classSymbol = classType->symbol();
  if (classSymbol) classSymbol = classSymbol->resolvedDefinition();
  if (!classSymbol) return nullptr;

  (void)traits.requireCompleteClass(classSymbol);

  auto beginName = unit->control()->getIdentifier("begin");
  auto endName = unit->control()->getIdentifier("end");

  const bool memberBeginEnd = usesMemberBeginEnd(unit->control(), classSymbol);

  FunctionSymbol* beginFunc = nullptr;
  FunctionSymbol* endFunc = nullptr;

  if (memberBeginEnd) {
    OverloadResolution memberLookup(unit);
    auto beginMembers = memberLookup.findCandidates(classSymbol, beginName);
    auto endMembers = memberLookup.findCandidates(classSymbol, endName);
    if (!beginMembers.functions.empty())
      beginFunc = beginMembers.functions.front();
    if (!endMembers.functions.empty()) endFunc = endMembers.functions.front();
  } else {
    std::vector<const Type*> argTypes = {rangeType};

    auto beginCandidates = argumentDependentLookup(unit, beginName, argTypes);
    auto endCandidates = argumentDependentLookup(unit, endName, argTypes);

    if (!beginCandidates.empty()) beginFunc = beginCandidates.front();
    if (!endCandidates.empty()) endFunc = endCandidates.front();
  }

  if (!beginFunc || !endFunc) return nullptr;

  ast->beginFunction = beginFunc;
  ast->endFunction = endFunc;
  ast->usesMemberBeginEnd = memberBeginEnd;

  auto beginFuncType = type_cast<FunctionType>(beginFunc->type());
  if (!beginFuncType) return nullptr;

  auto iterType = traits.remove_cvref(beginFuncType->returnType());

  if (traits.is_pointer(iterType)) {
    ast->isPointerIterator = true;
    return traits.get_element_type(iterType);
  }

  auto iterClassType = type_cast<ClassType>(iterType);
  if (!iterClassType) {
    ast->isPointerIterator = true;
    return nullptr;
  }

  auto iterClass = iterClassType->symbol();
  if (iterClass) iterClass = iterClass->resolvedDefinition();
  if (!iterClass) return nullptr;

  (void)traits.requireCompleteClass(iterClass);

  auto definedIterType = iterClass->type();

  auto placeholder = ThisExpressionAST::create(
      unit->arena(), ValueCategory::kLValue, definedIterType);

  OverloadResolution resolution(unit);
  ast->derefFunction = resolution.lookupOperator(
      ast->symbol, definedIterType, TokenKind::T_STAR, nullptr, placeholder);
  ast->incrementFunction =
      resolution.lookupOperator(ast->symbol, definedIterType,
                                TokenKind::T_PLUS_PLUS, nullptr, placeholder);
  ast->notEqualFunction = resolution.lookupOperator(
      ast->symbol, definedIterType, TokenKind::T_EXCLAIM_EQUAL, definedIterType,
      placeholder, placeholder);
  ast->notEqualRewritten = resolution.wasLastOperatorRewritten();
  ast->notEqualReversed = resolution.wasLastOperatorReversed();

  if (!ast->derefFunction) return nullptr;

  auto derefFuncType = type_cast<FunctionType>(ast->derefFunction->type());
  if (!derefFuncType) return nullptr;

  auto returnType = derefFuncType->returnType();
  if (traits.is_lvalue_reference(returnType) ||
      traits.is_rvalue_reference(returnType)) {
    return traits.remove_reference(returnType);
  }
  return returnType;
}
}  // namespace

auto Binder::declareRangeStructuredBindingEntity(
    StructuredBindingDeclarationAST* ast, const DeclSpecs& specs)
    -> VariableSymbol* {
  const auto refOp = ast->refQualifierLoc
                         ? unit_->tokenKind(ast->refQualifierLoc)
                         : TokenKind::T_EOF_SYMBOL;

  auto entityDeclarator = declareStructuredBindingEntity(
      ast->lbracketLoc, structuredBindingEntityName(), specs, refOp,
      /*initializer=*/nullptr, /*addSymbolToParentScope=*/false);
  if (!entityDeclarator) return nullptr;

  ast->hiddenVariable = entityDeclarator;
  return symbol_cast<VariableSymbol>(entityDeclarator->symbol);
}

struct Binder::ClassRangeRewrite {
  Binder& binder;
  ForRangeStatementAST* ast;
  TypeChecker& check;
  bool memberBeginEnd = false;
  SourceLocation implicitLoc;

  ClassRangeRewrite(Binder& b, ForRangeStatementAST* a, TypeChecker& c,
                    const ClassType* classType)
      : binder(b), ast(a), check(c) {
    if (auto classSymbol = classType->symbol()) {
      memberBeginEnd =
          usesMemberBeginEnd(b.control(), classSymbol->resolvedDefinition());
    }
    implicitLoc = a->rangeInitializer->firstSourceLocation();
  }

  [[nodiscard]] auto arena() const -> Arena* { return binder.unit_->arena(); }
  [[nodiscard]] auto control() const -> Control* { return binder.control(); }

  [[nodiscard]] auto declareVariable(const Type* type) const
      -> VariableSymbol* {
    auto symbol = control()->newVariableSymbol(ast->symbol, ast->colonLoc);
    symbol->setType(type);
    ast->symbol->addSymbol(symbol);
    return symbol;
  }

  [[nodiscard]] auto reference(VariableSymbol* symbol) const
      -> IdExpressionAST* {
    auto id = IdExpressionAST::create(arena());
    id->symbol = symbol;
    id->type = binder.traits.remove_reference(symbol->type());
    id->valueCategory = ValueCategory::kLValue;
    return id;
  }

  [[nodiscard]] auto implicitName(const Identifier* name) const -> NameIdAST* {
    auto id = NameIdAST::create(arena(), name);
    id->identifierLoc = implicitLoc;
    return id;
  }

  [[nodiscard]] auto callee(const Identifier* name) const -> ExpressionAST* {
    if (memberBeginEnd) {
      auto member = MemberExpressionAST::create(arena());
      member->baseExpression = reference(ast->rangeVariable);
      member->unqualifiedId = implicitName(name);
      member->accessOp = TokenKind::T_DOT;
      member->accessLoc = implicitLoc;
      return member;
    }

    auto id = IdExpressionAST::create(arena());
    id->unqualifiedId = implicitName(name);
    binder.declareArgumentDependentCallee(id);
    return id;
  }

  [[nodiscard]] auto beginEndCall(const Identifier* name) const
      -> ExpressionAST* {
    auto call = CallExpressionAST::create(arena());
    call->baseExpression = callee(name);
    call->lparenLoc = implicitLoc;
    call->rparenLoc = implicitLoc;
    if (!memberBeginEnd) {
      call->expressionList =
          make_list_node<ExpressionAST>(arena(), reference(ast->rangeVariable));
    }
    check.check(&call->baseExpression);
    ExpressionAST* result = call;
    check.check(&result);
    return result;
  }

  [[nodiscard]] auto rangeReferenceType() const -> const Type* {
    auto rangeInitializer = ast->rangeInitializer;
    if (rangeInitializer->valueCategory == ValueCategory::kLValue)
      return control()->getLvalueReferenceType(rangeInitializer->type);
    return control()->getRvalueReferenceType(rangeInitializer->type);
  }

  [[nodiscard]] auto iteratorOperation(TokenKind op) const
      -> UnaryExpressionAST* {
    auto operation = UnaryExpressionAST::create(arena());
    operation->expression = reference(ast->beginVariable);
    operation->op = op;
    operation->opLoc = ast->colonLoc;
    return operation;
  }

  [[nodiscard]] auto operator()() -> const Type* {
    ast->rangeVariable = declareVariable(rangeReferenceType());

    ast->beginInitializer = beginEndCall(control()->getIdentifier("begin"));
    ast->endInitializer = beginEndCall(control()->getIdentifier("end"));
    if (!ast->beginInitializer->type || !ast->endInitializer->type)
      return nullptr;

    auto& traits = binder.traits;
    ast->beginVariable =
        declareVariable(traits.remove_cvref(ast->beginInitializer->type));
    ast->endVariable =
        declareVariable(traits.remove_cvref(ast->endInitializer->type));

    auto notEqual = BinaryExpressionAST::create(arena());
    notEqual->leftExpression = reference(ast->beginVariable);
    notEqual->rightExpression = reference(ast->endVariable);
    notEqual->op = TokenKind::T_EXCLAIM_EQUAL;
    notEqual->opLoc = ast->colonLoc;
    ExpressionAST* condition = notEqual;
    check.check(&condition);
    check.check_bool_condition(condition);
    ast->condition = condition;

    ast->increment = iteratorOperation(TokenKind::T_PLUS_PLUS);
    check.check(&ast->increment);

    ast->element = iteratorOperation(TokenKind::T_STAR);
    check.check(&ast->element);
    return ast->element->type;
  }
};

void Binder::finishForRangeDeclaration(ForRangeStatementAST* ast,
                                       const DeclSpecs& specs) {
  auto rangeInitializer = ast->rangeInitializer;
  auto var = rangeDeclarationVariable(ast->rangeDeclaration);
  auto structuredBinding =
      ast_cast<StructuredBindingDeclarationAST>(ast->rangeDeclaration);

  TypeChecker check{unit_};
  check.setScope(scope());
  check.setReportErrors(reportErrors());

  if (auto braced = ast_cast<BracedInitListAST>(rangeInitializer)) {
    if (!braced->type) {
      auto rangeType =
          control()->getRvalueReferenceType(control()->getAutoType());
      (void)check.deducePlaceholderType(rangeType, braced);
    }
    if (braced->type)
      check.initializeBracedArgument(rangeInitializer, braced->type);
  }

  const bool needsDeduction = var && containsPlaceholderType(var->type());

  if (!rangeInitializer || !rangeInitializer->type ||
      isDependent(unit_, rangeInitializer->type)) {
    if (!isEnclosedInDependentTemplate(unit_, scope(),
                                       /*stopAtConcreteSpecialization=*/true))
      return;

    if (needsDeduction) var->setType(control()->getDependentType());

    if (!structuredBinding) return;

    auto entity = declareRangeStructuredBindingEntity(structuredBinding, specs);
    if (!entity) return;

    entity->setType(control()->getDependentType());
    decomposeStructuredBinding(structuredBinding, entity);
    return;
  }

  auto rangeType = traits.remove_cvref(rangeInitializer->type);

  auto elementType = resolveRangeIteration(unit_, ast, rangeType);

  if (auto classType = type_cast<ClassType>(rangeType)) {
    ClassRangeRewrite rewrite{*this, ast, check, classType};
    if (auto rewrittenElementType = rewrite())
      elementType = rewrittenElementType;
  }

  auto elementExpression = ast->element;
  if (!elementExpression && elementType) {
    elementExpression = ThisExpressionAST::create(
        unit_->arena(), ValueCategory::kLValue, elementType);
  }

  if (elementType && needsDeduction) {
    auto deduced = check.deducePlaceholderType(var->type(), elementExpression);
    if (!deduced) return;

    var->setType(deduced);

    if (auto classType =
            type_cast<ClassType>(traits.remove_cvref(var->type()))) {
      (void)traits.requireCompleteClass(classType->symbol());
    }
  }

  if (structuredBinding && elementType) {
    auto entity = declareRangeStructuredBindingEntity(structuredBinding, specs);
    if (!entity) return;

    auto deduced = structuredBindingArrayCopyType(
        structuredBinding, entity->type(), elementExpression->type);
    if (!deduced)
      deduced = check.deducePlaceholderType(entity->type(), elementExpression);
    if (!deduced) return;
    entity->setType(deduced);

    if (auto classType =
            type_cast<ClassType>(traits.remove_cvref(entity->type()))) {
      (void)traits.requireCompleteClass(classType->symbol());
    }

    decomposeStructuredBinding(structuredBinding, entity);

    var = entity;
  }

  if (var && ast->element && var->type()) {
    ast->element = EqualInitializerAST::create(
        unit_->arena(), ast->colonLoc, ast->element,
        ast->element->valueCategory, ast->element->type);
    check.check_variable_initializer(var, ast->element, ast->colonLoc,
                                     rangeElementCopyPolicy(structuredBinding));
  }
}
}  // namespace cxx
