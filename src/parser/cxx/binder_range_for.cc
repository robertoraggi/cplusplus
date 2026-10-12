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
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_checker.h>
#include <cxx/types.h>
#include <cxx/views/symbols.h>

namespace cxx {
namespace {
[[nodiscard]] auto rangeDeclarationVariable(DeclarationAST* rangeDeclaration)
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

  ast->binding.hiddenVariable = entityDeclarator;
  return symbol_cast<VariableSymbol>(entityDeclarator->symbol);
}

struct Binder::RangeForRewrite {
  Binder& binder;
  ForRangeStatementAST* ast;
  TypeChecker& check;
  ExpressionAST* rangeInitializer;
  VariableSymbol* rangeVariable = nullptr;
  VariableSymbol* beginVariable = nullptr;
  VariableSymbol* endVariable = nullptr;
  ForStatementAST* loop = nullptr;
  bool memberBeginEnd = false;
  SourceLocation implicitLoc;

  RangeForRewrite(Binder& b, ForRangeStatementAST* a, TypeChecker& c,
                  ExpressionAST* initializer)
      : binder(b),
        ast(a),
        check(c),
        rangeInitializer(initializer),
        implicitLoc(a->colonLoc) {
    auto rangeType = binder.traits.remove_cvref(initializer->type);
    if (auto classType = type_cast<ClassType>(rangeType)) {
      auto symbol = classType->symbol()->resolvedDefinition();
      (void)binder.traits.requireCompleteClass(symbol);
      memberBeginEnd = usesMemberBeginEnd(b.control(), symbol);
    }
  }

  [[nodiscard]] auto arena() const -> Arena* { return binder.unit_->arena(); }
  [[nodiscard]] auto control() const -> Control* { return binder.control(); }

  auto declareVariable(const Type* type, ExpressionAST* initializer)
      -> VariableSymbol* {
    auto symbol = control()->newVariableSymbol(ast->symbol, implicitLoc);
    symbol->setType(type);
    ast->symbol->addSymbol(symbol);
    ExpressionAST* equal = EqualInitializerAST::create(
        arena(), implicitLoc, initializer, initializer->valueCategory,
        initializer->type);
    check.check_variable_initializer(symbol, equal, implicitLoc);
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

  auto callee(const Identifier* name) -> ExpressionAST* {
    if (memberBeginEnd) {
      auto member = MemberExpressionAST::create(arena());
      member->baseExpression = reference(rangeVariable);
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

  auto beginEndCall(const Identifier* name) -> ExpressionAST* {
    auto call = CallExpressionAST::create(arena());
    call->baseExpression = callee(name);
    call->lparenLoc = implicitLoc;
    call->rparenLoc = implicitLoc;
    if (!memberBeginEnd) {
      call->expressionList =
          make_list_node<ExpressionAST>(arena(), reference(rangeVariable));
    }
    check.check(&call->baseExpression);
    ExpressionAST* result = call;
    check.check(&result);
    return result;
  }

  auto arrayEnd(const BoundedArrayType* arrayType) -> ExpressionAST* {
    auto size = IntLiteralExpressionAST::create(arena());
    size->literal =
        control()->integerLiteral(std::to_string(arrayType->size()));
    size->type = control()->getSizeType();
    size->valueCategory = ValueCategory::kPrValue;
    auto addition = BinaryExpressionAST::create(arena());
    addition->leftExpression = reference(rangeVariable);
    addition->rightExpression = size;
    addition->op = TokenKind::T_PLUS;
    addition->opLoc = implicitLoc;
    ExpressionAST* result = addition;
    check.check(&result);
    return result;
  }

  auto iteratorOperation(TokenKind op) -> ExpressionAST* {
    auto operation = UnaryExpressionAST::create(arena());
    operation->expression = reference(beginVariable);
    operation->op = op;
    operation->opLoc = implicitLoc;
    ExpressionAST* result = operation;
    check.check(&result);
    return result;
  }

  auto operator()() -> ExpressionAST* {
    auto rangeType = rangeInitializer->type;
    const Type* referenceType = control()->getRvalueReferenceType(rangeType);
    if (rangeInitializer->valueCategory == ValueCategory::kLValue)
      referenceType = control()->getLvalueReferenceType(rangeType);
    rangeVariable = declareVariable(referenceType, rangeInitializer);

    ExpressionAST* begin = nullptr;
    ExpressionAST* end = nullptr;
    if (auto arrayType = type_cast<BoundedArrayType>(rangeType)) {
      begin = reference(rangeVariable);
      if (!check.implicit_conversion(
              begin, control()->getPointerType(arrayType->elementType())))
        return nullptr;
      end = arrayEnd(arrayType);
    } else {
      begin = beginEndCall(control()->getIdentifier("begin"));
      end = beginEndCall(control()->getIdentifier("end"));
    }
    if (!begin->type || !end->type) return nullptr;
    beginVariable =
        declareVariable(binder.traits.remove_cvref(begin->type), begin);
    endVariable = declareVariable(binder.traits.remove_cvref(end->type), end);

    auto notEqual = BinaryExpressionAST::create(arena());
    notEqual->leftExpression = reference(beginVariable);
    notEqual->rightExpression = reference(endVariable);
    notEqual->op = TokenKind::T_EXCLAIM_EQUAL;
    notEqual->opLoc = implicitLoc;
    loop = ForStatementAST::create(arena());
    loop->forLoc = ast->forLoc;
    loop->lparenLoc = ast->lparenLoc;
    loop->semicolonLoc = implicitLoc;
    loop->condition = notEqual;
    check.check(&loop->condition);
    check.check_bool_condition(loop->condition);
    loop->expression = iteratorOperation(TokenKind::T_PLUS_PLUS);
    return iteratorOperation(TokenKind::T_STAR);
  }

  [[nodiscard]] auto declaration(DeclarationAST* declaration) const
      -> DeclarationStatementAST* {
    auto statement = DeclarationStatementAST::create(arena());
    statement->declaration = declaration;
    return statement;
  }

  [[nodiscard]] auto declaration(VariableSymbol* variable) const
      -> DeclarationStatementAST* {
    auto declarator = InitDeclaratorAST::create(arena());
    declarator->symbol = variable;
    auto simple = SimpleDeclarationAST::create(arena());
    simple->initDeclaratorList = make_list_node(arena(), declarator);
    simple->semicolonLoc = implicitLoc;
    return declaration(simple);
  }

  void finish() {
    auto body = CompoundStatementAST::create(arena());
    body->lbraceLoc = implicitLoc;
    body->statementList = make_list_node<StatementAST>(
        arena(), declaration(ast->rangeDeclaration));
    loop->statement = body;
    auto outer = CompoundStatementAST::create(arena());
    outer->lbraceLoc = ast->forLoc;
    auto tail = &outer->statementList;
    for (auto statement :
         {ast->initializer,
          static_cast<StatementAST*>(declaration(rangeVariable)),
          static_cast<StatementAST*>(declaration(beginVariable)),
          static_cast<StatementAST*>(declaration(endVariable)),
          static_cast<StatementAST*>(loop)}) {
      if (!statement) continue;
      *tail = make_list_node(arena(), statement);
      tail = &(*tail)->next;
    }
    ast->range.statement = outer;
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

  RangeForRewrite rewrite{*this, ast, check, rangeInitializer};
  auto elementExpression = rewrite();
  if (!elementExpression || !elementExpression->type) return;
  auto elementType = elementExpression->type;

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

  if (!var || !var->type()) return;
  ExpressionAST* initializer = EqualInitializerAST::create(
      unit_->arena(), ast->colonLoc, elementExpression,
      elementExpression->valueCategory, elementExpression->type);
  check.check_variable_initializer(var, initializer, ast->colonLoc,
                                   rangeElementCopyPolicy(structuredBinding));
  rewrite.finish();
}

void Binder::finishForRangeStatement(ForRangeStatementAST* ast) {
  auto outer = ast->range.statement;
  if (!outer || !ast->statement) return;
  outer->rbraceLoc = ast->statement->lastSourceLocation();
  for (auto statement : ListView{outer->statementList}) {
    auto loop = ast_cast<ForStatementAST>(statement);
    if (!loop) continue;
    loop->rparenLoc = ast->rparenLoc;
    auto body = ast_cast<CompoundStatementAST>(loop->statement);
    body->rbraceLoc = outer->rbraceLoc;
    body->statementList->next = make_list_node(unit_->arena(), ast->statement);
  }
}

}  // namespace cxx
