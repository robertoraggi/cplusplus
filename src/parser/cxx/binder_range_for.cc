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

  auto beginFunc = views::find_function(classSymbol->find(beginName),
                                        [](FunctionSymbol*) { return true; });
  auto endFunc = views::find_function(classSymbol->find(endName),
                                      [](FunctionSymbol*) { return true; });

  if (!beginFunc || !endFunc) {
    std::vector<const Type*> argTypes = {rangeType};

    auto beginCandidates = argumentDependentLookup(unit, beginName, argTypes);
    auto endCandidates = argumentDependentLookup(unit, endName, argTypes);

    if (!beginCandidates.empty()) beginFunc = beginCandidates.front();
    if (!endCandidates.empty()) endFunc = endCandidates.front();
  }

  if (!beginFunc || !endFunc) return nullptr;

  ast->beginFunction = beginFunc;
  ast->endFunction = endFunc;
  ast->usesMemberBeginEnd = beginFunc->parent() == classSymbol;

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
      definedIterType, TokenKind::T_STAR, nullptr, placeholder);
  ast->incrementFunction = resolution.lookupOperator(
      definedIterType, TokenKind::T_PLUS_PLUS, nullptr, placeholder);
  ast->notEqualFunction =
      resolution.lookupOperator(definedIterType, TokenKind::T_EXCLAIM_EQUAL,
                                definedIterType, placeholder, placeholder);
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

void Binder::finishForRangeDeclaration(ForRangeStatementAST* ast,
                                       const DeclSpecs& specs) {
  auto rangeInitializer = ast->rangeInitializer;
  auto var = rangeDeclarationVariable(ast->rangeDeclaration);
  auto structuredBinding =
      ast_cast<StructuredBindingDeclarationAST>(ast->rangeDeclaration);

  TypeChecker check{unit_};
  check.setScope(scope());

  const bool needsDeduction = var && containsPlaceholderType(var->type());

  if (!rangeInitializer || !rangeInitializer->type ||
      isDependent(unit_, rangeInitializer->type)) {
    if (needsDeduction && isEnclosedInTemplate(scope()))
      var->setType(control()->getTypeParameterType(0, 0, false));
    return;
  }

  auto rangeType = traits.remove_cvref(rangeInitializer->type);

  auto elementType = resolveRangeIteration(unit_, ast, rangeType);

  if (type_cast<ClassType>(rangeType)) {
    auto makeVariable = [&](const Type* type) {
      auto symbol = control()->newVariableSymbol(ast->symbol, ast->colonLoc);
      symbol->setType(type);
      ast->symbol->addSymbol(symbol);
      return symbol;
    };

    const Type* rangeReferenceType = nullptr;
    if (rangeInitializer->valueCategory == ValueCategory::kLValue)
      rangeReferenceType =
          control()->getLvalueReferenceType(rangeInitializer->type);
    else
      rangeReferenceType =
          control()->getRvalueReferenceType(rangeInitializer->type);
    ast->rangeVariable = makeVariable(rangeReferenceType);

    auto makeId = [&](VariableSymbol* symbol) {
      auto id = IdExpressionAST::create(unit_->arena());
      id->symbol = symbol;
      id->type = traits.remove_reference(symbol->type());
      id->valueCategory = ValueCategory::kLValue;
      return id;
    };

    auto classType = type_cast<ClassType>(rangeType);
    auto classSymbol = classType ? classType->symbol() : nullptr;
    if (classSymbol) classSymbol = classSymbol->resolvedDefinition();
    auto beginName = control()->getIdentifier("begin");
    auto endName = control()->getIdentifier("end");
    const bool memberCase = classSymbol &&
                            qualifiedLookup(classSymbol, beginName) &&
                            qualifiedLookup(classSymbol, endName);

    auto makeCall = [&](const Identifier* name) -> ExpressionAST* {
      ExpressionAST* callee = nullptr;
      if (memberCase) {
        auto member = MemberExpressionAST::create(unit_->arena());
        member->baseExpression = makeId(ast->rangeVariable);
        member->unqualifiedId = NameIdAST::create(unit_->arena(), name);
        member->accessOp = TokenKind::T_DOT;
        callee = member;
      } else {
        auto id = IdExpressionAST::create(unit_->arena());
        id->unqualifiedId = NameIdAST::create(unit_->arena(), name);
        declareArgumentDependentCallee(id);
        callee = id;
      }

      auto call = CallExpressionAST::create(unit_->arena());
      call->baseExpression = callee;
      if (!memberCase) {
        call->expressionList = make_list_node<ExpressionAST>(
            unit_->arena(), makeId(ast->rangeVariable));
      }
      check.check(callee);
      check.check(call);
      return call;
    };

    ast->beginInitializer = makeCall(beginName);
    ast->endInitializer = makeCall(endName);
    if (ast->beginInitializer->type && ast->endInitializer->type) {
      ast->beginVariable =
          makeVariable(traits.remove_cvref(ast->beginInitializer->type));
      ast->endVariable =
          makeVariable(traits.remove_cvref(ast->endInitializer->type));

      ExpressionAST* condition = BinaryExpressionAST::create(unit_->arena());
      auto binaryCondition = ast_cast<BinaryExpressionAST>(condition);
      binaryCondition->leftExpression = makeId(ast->beginVariable);
      binaryCondition->rightExpression = makeId(ast->endVariable);
      binaryCondition->op = TokenKind::T_EXCLAIM_EQUAL;
      binaryCondition->opLoc = ast->colonLoc;
      check.check(condition);
      check.check_bool_condition(condition);
      ast->condition = condition;

      auto increment = UnaryExpressionAST::create(unit_->arena());
      increment->expression = makeId(ast->beginVariable);
      increment->op = TokenKind::T_PLUS_PLUS;
      increment->opLoc = ast->colonLoc;
      check.check(increment);
      ast->increment = increment;

      auto dereference = UnaryExpressionAST::create(unit_->arena());
      dereference->expression = makeId(ast->beginVariable);
      dereference->op = TokenKind::T_STAR;
      dereference->opLoc = ast->colonLoc;
      check.check(dereference);
      ast->element = dereference;
      if (dereference->type) elementType = dereference->type;
    }
  }

  if (elementType && needsDeduction) {
    auto deduced = check.deduceAutoType(var->type(), elementType);
    if (!deduced) return;

    var->setType(deduced);

    if (auto classType =
            type_cast<ClassType>(traits.remove_cvref(var->type()))) {
      (void)traits.requireCompleteClass(classType->symbol());
    }
  }

  if (structuredBinding && elementType) {
    const auto refOp =
        structuredBinding->refQualifierLoc
            ? unit_->tokenKind(structuredBinding->refQualifierLoc)
            : TokenKind::T_EOF_SYMBOL;

    auto entityDeclarator = declareStructuredBindingEntity(
        structuredBinding->lbracketLoc, structuredBindingEntityName(), specs,
        refOp, /*initializer=*/nullptr, /*addSymbolToParentScope=*/false);
    if (!entityDeclarator) return;

    auto entity = symbol_cast<VariableSymbol>(entityDeclarator->symbol);
    if (!entity) return;

    auto deduced = check.deduceAutoType(entity->type(), elementType);
    if (!deduced) return;
    entity->setType(deduced);

    if (auto classType =
            type_cast<ClassType>(traits.remove_cvref(entity->type()))) {
      (void)traits.requireCompleteClass(classType->symbol());
    }

    structuredBinding->hiddenVariable = entityDeclarator;
    decomposeStructuredBinding(structuredBinding, entity);

    var = entity;
  }

  if (var && ast->element && var->type()) {
    (void)check.implicit_conversion(ast->element, var->type());
    ast->element = EqualInitializerAST::create(
        unit_->arena(), ast->colonLoc, ast->element,
        ast->element->valueCategory, ast->element->type);
  }
}
}  // namespace cxx
