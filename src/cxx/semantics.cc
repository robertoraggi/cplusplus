// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
#include <cxx/control.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/scope.h>
#include <cxx/semantics.h>
#include <cxx/symbol_factory.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_environment.h>
#include <cxx/types.h>

namespace cxx {

Semantics::Semantics(TranslationUnit* unit)
    : unit_(unit),
      control_(unit_->control()),
      types_(control_->types()),
      symbols_(control_->symbols()) {}

Semantics::~Semantics() {}

void Semantics::unit(UnitAST* ast) { accept(ast); }

void Semantics::specifiers(List<SpecifierAST*>* ast,
                           SpecifiersSem* specifiers) {
  if (!ast) return;
  std::swap(specifiers_, specifiers);
  for (auto it = ast; it; it = it->next) accept(it->value);
  std::swap(specifiers_, specifiers);
}

void Semantics::specifiers(SpecifierAST* ast, SpecifiersSem* specifiers) {
  if (!ast) return;
  std::swap(specifiers_, specifiers);
  accept(ast);
  std::swap(specifiers_, specifiers);
}

void Semantics::declarator(DeclaratorAST* ast, DeclaratorSem* declarator) {
  std::swap(declarator_, declarator);
  accept(ast);
  std::swap(declarator_, declarator);
}

void Semantics::initDeclarator(InitDeclaratorAST* ast,
                               DeclaratorSem* declarator) {
  this->declarator(ast->declarator, declarator);
  initializer(ast->initializer);
}

void Semantics::name(NameAST* ast, NameSem* nameSem) {
  if (!ast) return;
  if (ast->name) {
    nameSem->name = ast->name;
    return;
  }
  std::swap(nameSem_, nameSem);
  accept(ast);
  std::swap(nameSem_, nameSem);
  ast->name = nameSem->name;
}

void Semantics::nestedNameSpecifier(
    NestedNameSpecifierAST* ast, NestedNameSpecifierSem* nestedNameSpecifier) {
  std::swap(nestedNameSpecifier_, nestedNameSpecifier);
  accept(ast);
  std::swap(nestedNameSpecifier_, nestedNameSpecifier);
}

void Semantics::exceptionDeclaration(ExceptionDeclarationAST* ast) {
  accept(ast);
}

void Semantics::compoundStatement(CompoundStatementAST* ast) { accept(ast); }

void Semantics::attribute(AttributeAST* ast) { accept(ast); }

void Semantics::expression(ExpressionAST* ast, ExpressionSem* expression) {
  if (!ast) return;
  if (ast->type) {
    expression->type = ast->type;
    return;
  }
  expression->type = QualifiedType(types_->errorType());
  std::swap(expression_, expression);
  accept(ast);
  std::swap(expression_, expression);
  ast->type = expression->type;
}

void Semantics::ptrOperator(PtrOperatorAST* ast) { accept(ast); }

void Semantics::coreDeclarator(CoreDeclaratorAST* ast) { accept(ast); }

void Semantics::declaratorModifier(DeclaratorModifierAST* ast) { accept(ast); }

void Semantics::declaratorModifiers(List<DeclaratorModifierAST*>* ast) {
  if (!ast) return;
  if (ast->next) declaratorModifiers(ast->next);
  declaratorModifier(ast->value);
}

void Semantics::initializer(InitializerAST* ast) { accept(ast); }

void Semantics::baseSpecifier(BaseSpecifierAST* ast) { accept(ast); }

void Semantics::parameterDeclaration(ParameterDeclarationAST* ast) {
  accept(ast);
}

void Semantics::parameterDeclarationClause(ParameterDeclarationClauseAST* ast) {
  accept(ast);
}

void Semantics::lambdaCapture(LambdaCaptureAST* ast) { accept(ast); }

void Semantics::trailingReturnType(TrailingReturnTypeAST* ast) { accept(ast); }

void Semantics::typeId(TypeIdAST* ast) {
  if (!ast || ast->type) return;
  SpecifiersSem specs;
  this->specifiers(ast->typeSpecifierList, &specs);
  DeclaratorSem decl{specs};
  this->declarator(ast->declarator, &decl);
  ast->type = decl.type;
}

void Semantics::memInitializer(MemInitializerAST* ast) { accept(ast); }

void Semantics::bracedInitList(BracedInitListAST* ast) { accept(ast); }

void Semantics::ctorInitializer(CtorInitializerAST* ast) { accept(ast); }

void Semantics::handler(HandlerAST* ast) { accept(ast); }

void Semantics::declaration(DeclarationAST* ast) { accept(ast); }

void Semantics::lambdaIntroducer(LambdaIntroducerAST* ast) { accept(ast); }

void Semantics::lambdaDeclarator(LambdaDeclaratorAST* ast) { accept(ast); }

void Semantics::newTypeId(NewTypeIdAST* ast) { accept(ast); }

void Semantics::newInitializer(NewInitializerAST* ast) { accept(ast); }

void Semantics::statement(StatementAST* ast) { accept(ast); }

void Semantics::functionBody(FunctionBodyAST* ast) { accept(ast); }

void Semantics::enumBase(EnumBaseAST* ast) { accept(ast); }

void Semantics::usingDeclarator(UsingDeclaratorAST* ast) { accept(ast); }

void Semantics::templateArgument(TemplateArgumentAST* ast) { accept(ast); }

void Semantics::enumerator(EnumeratorAST* ast) { accept(ast); }

void Semantics::baseClause(BaseClauseAST* ast) { accept(ast); }

void Semantics::parametersAndQualifiers(ParametersAndQualifiersAST* ast) {
  accept(ast);
}

void Semantics::accept(AST* ast) {
  if (ast) ast->accept(this);
}

void Semantics::visit(TypeIdAST* ast) {
  SpecifiersSem specifiers;
  this->specifiers(ast->typeSpecifierList, &specifiers);
  DeclaratorSem declarator{specifiers};
  this->declarator(ast->declarator, &declarator);
}

void Semantics::visit(NestedNameSpecifierAST* ast) {
  Scope* scope = scope_;
  bool unqualifiedLookup = true;
  for (auto it = ast->nameList; it; it = it->next) {
    NameSem name;
    this->name(it->value, &name);
    if (checkTypes_ && scope) {
      Symbol* sym = nullptr;
      if (unqualifiedLookup) {
        sym = scope->lookup(name.name, LookupOptions::kTypeOrNamespace);
        unqualifiedLookup = false;
      } else {
        sym = scope->find(name.name, LookupOptions::kTypeOrNamespace);
      }
      scope = sym ? sym->scope() : nullptr;
    }
  }
  if (scope) ast->symbol = scope->owner();
}

void Semantics::visit(UsingDeclaratorAST* ast) {
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->name, &name);
}

void Semantics::visit(HandlerAST* ast) {
  exceptionDeclaration(ast->exceptionDeclaration);
  compoundStatement(ast->statement);
}

void Semantics::visit(TypeTemplateArgumentAST* ast) { typeId(ast->typeId); }

void Semantics::visit(ExpressionTemplateArgumentAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(EnumBaseAST* ast) {
  SpecifiersSem specifiers;
  this->specifiers(ast->typeSpecifierList, &specifiers);
}

void Semantics::visit(EnumeratorAST* ast) {
  NameSem name;
  this->name(ast->name, &name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(DeclaratorAST* ast) {
  for (auto it = ast->ptrOpList; it; it = it->next) ptrOperator(it->value);
  declaratorModifiers(ast->modifiers);
  coreDeclarator(ast->coreDeclarator);
}

void Semantics::visit(InitDeclaratorAST* ast) {
  throw std::runtime_error("unreachable");
}

void Semantics::visit(BaseSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  NameSem name;
  this->name(ast->name, &name);
}

void Semantics::visit(BaseClauseAST* ast) {
  for (auto it = ast->baseSpecifierList; it; it = it->next)
    baseSpecifier(it->value);
}

void Semantics::visit(NewTypeIdAST* ast) {
  SpecifiersSem specifiers;
  this->specifiers(ast->typeSpecifierList, &specifiers);
}

void Semantics::visit(ParameterDeclarationClauseAST* ast) {
  for (auto it = ast->parameterDeclarationList; it; it = it->next)
    parameterDeclaration(it->value);
}

void Semantics::visit(ParametersAndQualifiersAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  SpecifiersSem specifiers;
  this->specifiers(ast->cvQualifierList, &specifiers);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(LambdaIntroducerAST* ast) {
  for (auto it = ast->captureList; it; it = it->next) lambdaCapture(it->value);
}

void Semantics::visit(LambdaDeclaratorAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  SpecifiersSem specifiers;
  this->specifiers(ast->declSpecifierList, &specifiers);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  trailingReturnType(ast->trailingReturnType);
}

void Semantics::visit(TrailingReturnTypeAST* ast) { typeId(ast->typeId); }

void Semantics::visit(CtorInitializerAST* ast) {
  for (auto it = ast->memInitializerList; it; it = it->next)
    memInitializer(it->value);
}

void Semantics::visit(ParenMemInitializerAST* ast) {
  NameSem name;
  this->name(ast->name, &name);
  for (auto it = ast->expressionList; it; it = it->next) {
    ExpressionSem expression;
    this->expression(it->value, &expression);
  }
}

void Semantics::visit(BracedMemInitializerAST* ast) {
  NameSem name;
  this->name(ast->name, &name);
  bracedInitList(ast->bracedInitList);
}

void Semantics::visit(ThisLambdaCaptureAST* ast) {}

void Semantics::visit(DerefThisLambdaCaptureAST* ast) {}

void Semantics::visit(SimpleLambdaCaptureAST* ast) {}

void Semantics::visit(RefLambdaCaptureAST* ast) {}

void Semantics::visit(RefInitLambdaCaptureAST* ast) {
  initializer(ast->initializer);
}

void Semantics::visit(InitLambdaCaptureAST* ast) {
  initializer(ast->initializer);
}

void Semantics::visit(EqualInitializerAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(BracedInitListAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) {
    ExpressionSem expression;
    this->expression(it->value, &expression);
  }
}

void Semantics::visit(ParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) {
    ExpressionSem expression;
    this->expression(it->value, &expression);
  }
}

void Semantics::visit(NewParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) {
    ExpressionSem expression;
    this->expression(it->value, &expression);
  }
}

void Semantics::visit(NewBracedInitializerAST* ast) {
  bracedInitList(ast->bracedInit);
}

void Semantics::visit(EllipsisExceptionDeclarationAST* ast) {}

void Semantics::visit(TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  SpecifiersSem specifiers;
  this->specifiers(ast->typeSpecifierList, &specifiers);
  DeclaratorSem declarator{specifiers};
  this->declarator(ast->declarator, &declarator);
}

void Semantics::visit(DefaultFunctionBodyAST* ast) {}

void Semantics::visit(CompoundStatementFunctionBodyAST* ast) {
  ctorInitializer(ast->ctorInitializer);
  compoundStatement(ast->statement);
}

void Semantics::visit(TryStatementFunctionBodyAST* ast) {
  ctorInitializer(ast->ctorInitializer);
  compoundStatement(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) handler(it->value);
}

void Semantics::visit(DeleteFunctionBodyAST* ast) {}

void Semantics::visit(TranslationUnitAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(ModuleUnitAST* ast) {}

void Semantics::visit(ThisExpressionAST* ast) {}

void Semantics::visit(CharLiteralExpressionAST* ast) {
  expression_->type =
      QualifiedType{types_->integerType(IntegerKind::kChar, false)};
}

void Semantics::visit(BoolLiteralExpressionAST* ast) {}

void Semantics::visit(IntLiteralExpressionAST* ast) {
  auto kind = IntegerKind::kInt;
  bool isUnsigned = false;
  QualifiedType intTy{types_->integerType(kind, isUnsigned)};
  expression_->type = intTy;
}

void Semantics::visit(FloatLiteralExpressionAST* ast) {}

void Semantics::visit(NullptrLiteralExpressionAST* ast) {}

void Semantics::visit(StringLiteralExpressionAST* ast) {
  QualifiedType charTy{types_->integerType(IntegerKind::kChar, false)};
  charTy.setQualifiers(Qualifiers::kConst);
  QualifiedType charPtrTy{types_->pointerType(charTy, Qualifiers::kNone)};
  expression_->type = charPtrTy;
}

void Semantics::visit(UserDefinedStringLiteralExpressionAST* ast) {}

void Semantics::visit(IdExpressionAST* ast) {
  NameSem name;
  this->name(ast->name, &name);

  if (ast->symbol) expression_->type = ast->symbol->type();
}

void Semantics::visit(NestedExpressionAST* ast) {
  ExpressionSem expr;
  expression(ast->expression, &expr);
  expression_->type = expr.type;
}

void Semantics::visit(RightFoldExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(LeftFoldExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(FoldExpressionAST* ast) {
  ExpressionSem leftExpression;
  this->expression(ast->leftExpression, &leftExpression);
  ExpressionSem rightExpression;
  this->expression(ast->rightExpression, &rightExpression);
}

void Semantics::visit(LambdaExpressionAST* ast) {
  lambdaIntroducer(ast->lambdaIntroducer);
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  lambdaDeclarator(ast->lambdaDeclarator);
  compoundStatement(ast->statement);
}

void Semantics::visit(SizeofExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(SizeofTypeExpressionAST* ast) {
  typeId(ast->typeId);

  expression_->type =
      QualifiedType{types_->integerType(IntegerKind::kLong, false)};
}

void Semantics::visit(SizeofPackExpressionAST* ast) {}

void Semantics::visit(TypeidExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(TypeidOfTypeExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(AlignofExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(UnaryExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);

  switch (ast->op) {
    case TokenKind::T_STAR: {
      if (auto ptrTy = expression.type->asPointerType())
        expression_->type = ptrTy->elementType();
      break;
    }
    case TokenKind::T_AMP: {
      expression_->type = QualifiedType(
          types_->pointerType(expression.type, Qualifiers::kNone));
      break;
    }
    default: {
      // TODO
      expression_->type = expression.type;
    }
  }  // switch
}

void Semantics::visit(BinaryExpressionAST* ast) {
  ExpressionSem leftExpression;
  this->expression(ast->leftExpression, &leftExpression);

  ExpressionSem rightExpression;
  this->expression(ast->rightExpression, &rightExpression);

  if (ast->op == TokenKind::T_PLUS) {
    if (leftExpression.type->asPointerType() &&
        rightExpression.type->asIntegerType()) {
      expression_->type = leftExpression.type;
      return;
    }

    if (rightExpression.type->asPointerType() &&
        leftExpression.type->asIntegerType()) {
      expression_->type = rightExpression.type;
      return;
    }
  }

  if (ast->op == TokenKind::T_MINUS && leftExpression.type->asPointerType()) {
    if (rightExpression.type->asIntegerType()) {
      expression_->type = leftExpression.type;
      return;
    }

    if (leftExpression.type == rightExpression.type) {
      expression_->type =
          QualifiedType{types_->integerType(IntegerKind::kLong, false)};
      return;
    }
  }

  auto leftIntTy = leftExpression.type->asIntegerType();

  auto rightIntTy = rightExpression.type->asIntegerType();

  if (leftIntTy && rightIntTy && !leftIntTy->isUnsigned() &&
      !rightIntTy->isUnsigned()) {
    const auto maxIntKind = static_cast<IntegerKind>(
        std::max(static_cast<int>(leftIntTy->kind()),
                 static_cast<int>(rightIntTy->kind())));

    if (maxIntKind <= IntegerKind::kInt) {
      expression_->type =
          QualifiedType{types_->integerType(IntegerKind::kInt, false)};
    } else if (maxIntKind <= IntegerKind::kLong) {
      expression_->type =
          QualifiedType{types_->integerType(IntegerKind::kLong, false)};
    } else {
      expression_->type =
          QualifiedType{types_->integerType(IntegerKind::kLongLong, false)};
    }
  }
}

void Semantics::visit(AssignmentExpressionAST* ast) {
  ExpressionSem leftExpression;
  this->expression(ast->leftExpression, &leftExpression);
  ExpressionSem rightExpression;
  this->expression(ast->rightExpression, &rightExpression);
  expression_->type = leftExpression.type;
}

void Semantics::visit(BracedTypeConstructionAST* ast) {
  SpecifiersSem specifiers;
  this->specifiers(ast->typeSpecifier, &specifiers);
  bracedInitList(ast->bracedInitList);
}

void Semantics::visit(TypeConstructionAST* ast) {
  SpecifiersSem specifiers;
  this->specifiers(ast->typeSpecifier, &specifiers);
  for (auto it = ast->expressionList; it; it = it->next) {
    ExpressionSem expression;
    this->expression(it->value, &expression);
  }
}

void Semantics::visit(CallExpressionAST* ast) {
  ExpressionSem baseExpression;
  this->expression(ast->baseExpression, &baseExpression);
  for (auto it = ast->expressionList; it; it = it->next) {
    ExpressionSem expression;
    this->expression(it->value, &expression);
  }
  if (auto funTy = baseExpression.type->asFunctionType())
    expression_->type = funTy->returnType();
}

void Semantics::visit(SubscriptExpressionAST* ast) {
  ExpressionSem baseExpression;
  this->expression(ast->baseExpression, &baseExpression);
  ExpressionSem indexExpression;
  this->expression(ast->indexExpression, &indexExpression);
  if (auto arrayTy = baseExpression.type->asArrayType())
    expression_->type = arrayTy->elementType();
  else if (auto vlaTy = baseExpression.type->asUnboundArrayType())
    expression_->type = vlaTy->elementType();
  else if (auto ptrTy = baseExpression.type->asPointerType())
    expression_->type = ptrTy->elementType();
}

void Semantics::visit(MemberExpressionAST* ast) {
  ExpressionSem baseExpression;
  this->expression(ast->baseExpression, &baseExpression);

  NameSem name;
  this->name(ast->name, &name);

  auto baseTy = baseExpression.type;

  if (auto ptrTy = baseTy->asPointerType();
      ptrTy && ast->accessOp == TokenKind::T_MINUS_GREATER) {
    baseTy = ptrTy->elementType();
  }

  if (auto classTy = baseTy->asClassType()) {
    auto memberName = name.name;

    auto classSymbol = classTy->symbol();

    auto member = classSymbol->scope()->find(name.name);

    if (member) {
      ast->symbol = member;
      expression_->type = ast->symbol->type();
    } else if (checkTypes_) {
      unit_->error(ast->name->firstSourceLocation(), "undefined member '{}'",
                   *name.name);
    }
  }
}

void Semantics::visit(PostIncrExpressionAST* ast) {
  ExpressionSem baseExpression;
  this->expression(ast->baseExpression, &baseExpression);
}

void Semantics::visit(ConditionalExpressionAST* ast) {
  ExpressionSem condition;
  this->expression(ast->condition, &condition);
  ExpressionSem iftrueExpression;
  this->expression(ast->iftrueExpression, &iftrueExpression);
  ExpressionSem iffalseExpression;
  this->expression(ast->iffalseExpression, &iffalseExpression);
}

void Semantics::visit(ImplicitCastExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(CastExpressionAST* ast) {
  typeId(ast->typeId);
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
  expression_->type = ast->typeId->type;
}

void Semantics::visit(CppCastExpressionAST* ast) {
  typeId(ast->typeId);
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(NewExpressionAST* ast) {
  newTypeId(ast->typeId);
  newInitializer(ast->newInitalizer);
}

void Semantics::visit(DeleteExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(ThrowExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(NoexceptExpressionAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(LabeledStatementAST* ast) { statement(ast->statement); }

void Semantics::visit(CaseStatementAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
  statement(ast->statement);
}

void Semantics::visit(DefaultStatementAST* ast) { statement(ast->statement); }

void Semantics::visit(ExpressionStatementAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next) statement(it->value);
}

void Semantics::visit(IfStatementAST* ast) {
  statement(ast->initializer);
  ExpressionSem condition;
  this->expression(ast->condition, &condition);
  statement(ast->statement);
  statement(ast->elseStatement);
}

void Semantics::visit(SwitchStatementAST* ast) {
  statement(ast->initializer);
  ExpressionSem condition;
  this->expression(ast->condition, &condition);
  statement(ast->statement);
}

void Semantics::visit(WhileStatementAST* ast) {
  ExpressionSem condition;
  this->expression(ast->condition, &condition);
  statement(ast->statement);
}

void Semantics::visit(DoStatementAST* ast) {
  statement(ast->statement);
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(ForRangeStatementAST* ast) {
  statement(ast->initializer);
  declaration(ast->rangeDeclaration);
  ExpressionSem rangeInitializer;
  this->expression(ast->rangeInitializer, &rangeInitializer);
  statement(ast->statement);
}

void Semantics::visit(ForStatementAST* ast) {
  statement(ast->initializer);
  ExpressionSem condition;
  this->expression(ast->condition, &condition);
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
  statement(ast->statement);
}

void Semantics::visit(BreakStatementAST* ast) {}

void Semantics::visit(ContinueStatementAST* ast) {}

void Semantics::visit(ReturnStatementAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(GotoStatementAST* ast) {}

void Semantics::visit(CoroutineReturnStatementAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(DeclarationStatementAST* ast) {
  declaration(ast->declaration);
}

void Semantics::visit(TryBlockStatementAST* ast) {
  compoundStatement(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) handler(it->value);
}

void Semantics::visit(AccessDeclarationAST* ast) {}

void Semantics::visit(FunctionDefinitionAST* ast) {
  SpecifiersSem specifiers;
  this->specifiers(ast->declSpecifierList, &specifiers);
  DeclaratorSem declarator{specifiers};
  this->declarator(ast->declarator, &declarator);
  functionBody(ast->functionBody);
}

void Semantics::visit(ConceptDefinitionAST* ast) {
  NameSem name;
  this->name(ast->name, &name);
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(ForRangeDeclarationAST* ast) {}

void Semantics::visit(AliasDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  typeId(ast->typeId);
}

void Semantics::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  SpecifiersSem specifiers;
  this->specifiers(ast->declSpecifierList, &specifiers);
  for (auto it = ast->initDeclaratorList; it; it = it->next) {
    DeclaratorSem declarator{specifiers};
    this->initDeclarator(it->value, &declarator);
  }
}

void Semantics::visit(StaticAssertDeclarationAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(EmptyDeclarationAST* ast) {}

void Semantics::visit(AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->name, &name);
  enumBase(ast->enumBase);
}

void Semantics::visit(UsingEnumDeclarationAST* ast) {}

void Semantics::visit(NamespaceDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->name, &name);
  for (auto it = ast->extraAttributeList; it; it = it->next)
    attribute(it->value);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(NamespaceAliasDefinitionAST* ast) {
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->name, &name);
}

void Semantics::visit(UsingDirectiveAST* ast) {}

void Semantics::visit(UsingDeclarationAST* ast) {
  for (auto it = ast->usingDeclaratorList; it; it = it->next)
    usingDeclarator(it->value);
}

void Semantics::visit(AsmDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(ExportDeclarationAST* ast) {}

void Semantics::visit(ModuleImportDeclarationAST* ast) {}

void Semantics::visit(TemplateDeclarationAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  declaration(ast->declaration);
}

void Semantics::visit(TypenameTypeParameterAST* ast) { typeId(ast->typeId); }

void Semantics::visit(TypenamePackTypeParameterAST* ast) {}

void Semantics::visit(TemplateTypeParameterAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  NameSem name;
  this->name(ast->name, &name);
}

void Semantics::visit(TemplatePackTypeParameterAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(DeductionGuideAST* ast) {}

void Semantics::visit(ExplicitInstantiationAST* ast) {
  declaration(ast->declaration);
}

void Semantics::visit(ParameterDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  SpecifiersSem specifiers;
  this->specifiers(ast->typeSpecifierList, &specifiers);
  DeclaratorSem declarator{specifiers};
  this->declarator(ast->declarator, &declarator);
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(LinkageSpecificationAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(SimpleNameAST* ast) { nameSem_->name = ast->identifier; }

void Semantics::visit(DestructorNameAST* ast) {
  NameSem name;
  this->name(ast->id, &name);
}

void Semantics::visit(DecltypeNameAST* ast) {
  SpecifiersSem specifiers;
  this->specifiers(ast->decltypeSpecifier, &specifiers);
}

void Semantics::visit(OperatorNameAST* ast) {
  nameSem_->name = control_->operatorNameId(ast->op);
}

void Semantics::visit(ConversionNameAST* ast) {
  // typeId(ast->typeId);
  if (!ast->typeId) return;
  SpecifiersSem specifiers;
  this->specifiers(ast->typeId->typeSpecifierList, &specifiers);
  DeclaratorSem decl{specifiers};
  this->declarator(ast->typeId->declarator, &decl);
  QualifiedType type = decl.type;
  nameSem_->name = control_->conversionNameId(type);
}

void Semantics::visit(TemplateNameAST* ast) {
  NameSem name;
  this->name(ast->id, &name);
  for (auto it = ast->templateArgumentList; it; it = it->next)
    templateArgument(it->value);
}

void Semantics::visit(QualifiedNameAST* ast) {
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->id, &name);
  nameSem_->name = name.name;
}

void Semantics::visit(TypedefSpecifierAST* ast) {
  specifiers_->isTypedef = true;
}

void Semantics::visit(FriendSpecifierAST* ast) { specifiers_->isFriend = true; }

void Semantics::visit(ConstevalSpecifierAST* ast) {}

void Semantics::visit(ConstinitSpecifierAST* ast) {}

void Semantics::visit(ConstexprSpecifierAST* ast) {
  specifiers_->isConstexpr = true;
}

void Semantics::visit(InlineSpecifierAST* ast) {}

void Semantics::visit(StaticSpecifierAST* ast) { specifiers_->isStatic = true; }

void Semantics::visit(ExternSpecifierAST* ast) { specifiers_->isExtern = true; }

void Semantics::visit(ThreadLocalSpecifierAST* ast) {}

void Semantics::visit(ThreadSpecifierAST* ast) {}

void Semantics::visit(MutableSpecifierAST* ast) {}

void Semantics::visit(VirtualSpecifierAST* ast) {}

void Semantics::visit(ExplicitSpecifierAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(AutoTypeSpecifierAST* ast) {}

void Semantics::visit(VoidTypeSpecifierAST* ast) {
  specifiers_->type.setType(types_->voidType());
}

void Semantics::visit(VaListTypeSpecifierAST* ast) {}

void Semantics::visit(IntegralTypeSpecifierAST* ast) {
  switch (ast->specifier) {
    case TokenKind::T_CHAR8_T:
      specifiers_->type.setType(types_->characterType(CharacterKind::kChar8T));
      break;

    case TokenKind::T_CHAR16_T:
      specifiers_->type.setType(types_->characterType(CharacterKind::kChar16T));
      break;

    case TokenKind::T_CHAR32_T:
      specifiers_->type.setType(types_->characterType(CharacterKind::kChar32T));
      break;

    case TokenKind::T_WCHAR_T:
      specifiers_->type.setType(types_->characterType(CharacterKind::kWCharT));
      break;

    case TokenKind::T_BOOL:
      specifiers_->type.setType(types_->booleanType());
      break;

    case TokenKind::T_CHAR:
      specifiers_->type.setType(
          types_->integerType(IntegerKind::kChar, specifiers_->isUnsigned));
      break;

    case TokenKind::T_SHORT:
      specifiers_->type.setType(
          types_->integerType(IntegerKind::kShort, specifiers_->isUnsigned));
      break;

    case TokenKind::T_INT: {
      auto kind = IntegerKind::kInt;

      if (auto intTy = specifiers_->type->asIntegerType()) {
        using U = std::underlying_type<IntegerKind>::type;

        if (static_cast<U>(intTy->kind()) > static_cast<U>(IntegerKind::kInt))
          kind = intTy->kind();
      }

      specifiers_->type.setType(
          types_->integerType(kind, specifiers_->isUnsigned));

      break;
    }

    case TokenKind::T___INT64:
      specifiers_->type.setType(
          types_->integerType(IntegerKind::kInt64, specifiers_->isUnsigned));
      break;

    case TokenKind::T___INT128:
      specifiers_->type.setType(
          types_->integerType(IntegerKind::kInt128, specifiers_->isUnsigned));
      break;

    case TokenKind::T_LONG: {
      auto ty = dynamic_cast<const IntegerType*>(specifiers_->type.type());

      if (ty && ty->kind() == IntegerKind::kLong) {
        specifiers_->type.setType(types_->integerType(IntegerKind::kLongLong,
                                                      specifiers_->isUnsigned));
      } else {
        specifiers_->type.setType(
            types_->integerType(IntegerKind::kLong, specifiers_->isUnsigned));
      }

      break;
    }

    case TokenKind::T_SIGNED:
      specifiers_->isUnsigned = false;

      if (!specifiers_->type)
        specifiers_->type.setType(
            types_->integerType(IntegerKind::kInt, specifiers_->isUnsigned));
      break;

    case TokenKind::T_UNSIGNED:
      specifiers_->isUnsigned = true;

      if (!specifiers_->type)
        specifiers_->type.setType(
            types_->integerType(IntegerKind::kInt, specifiers_->isUnsigned));
      break;

    default:
      throw std::runtime_error(fmt::format("invalid integral type: '{}'",
                                           Token::spell(ast->specifier)));
  }  // switch
}

void Semantics::visit(FloatingPointTypeSpecifierAST* ast) {
  switch (ast->specifier) {
    case TokenKind::T_FLOAT:
      specifiers_->type.setType(
          types_->floatingPointType(FloatingPointKind::kFloat));
      break;

    case TokenKind::T_DOUBLE: {
      auto ty = dynamic_cast<const IntegerType*>(specifiers_->type.type());

      if (ty && ty->kind() == IntegerKind::kLong) {
        specifiers_->type.setType(
            types_->floatingPointType(FloatingPointKind::kLongDouble));
      } else {
        specifiers_->type.setType(
            types_->floatingPointType(FloatingPointKind::kDouble));
      }
      break;
    }

    case TokenKind::T___FLOAT128:
      specifiers_->type.setType(
          types_->floatingPointType(FloatingPointKind::kFloat));
      break;

    default:
      throw std::runtime_error(fmt::format("invalid floating point type: '{}'",
                                           Token::spell(ast->specifier)));
  }  // switch
}

void Semantics::visit(ComplexTypeSpecifierAST* ast) {}

void Semantics::visit(NamedTypeSpecifierAST* ast) {
  NameSem name;
  this->name(ast->name, &name);
  if (ast->symbol) {
    specifiers_->type.mergeWith(ast->symbol->type());
  }
}

void Semantics::visit(AtomicTypeSpecifierAST* ast) { typeId(ast->typeId); }

void Semantics::visit(UnderlyingTypeSpecifierAST* ast) {}

void Semantics::visit(ElaboratedTypeSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->name, &name);
  if (auto classSymbol = dynamic_cast<ClassSymbol*>(ast->symbol)) {
    specifiers_->type.setType(types_->classType(classSymbol));
  } else if (auto enumSymbol = dynamic_cast<EnumSymbol*>(ast->symbol)) {
    specifiers_->type.setType(types_->enumType(enumSymbol));
  }
}

void Semantics::visit(DecltypeAutoSpecifierAST* ast) {}

void Semantics::visit(DecltypeSpecifierAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(TypeofSpecifierAST* ast) {
  ExpressionSem expression;
  this->expression(ast->expression, &expression);
}

void Semantics::visit(PlaceholderTypeSpecifierAST* ast) {}

void Semantics::visit(ConstQualifierAST* ast) {
  specifiers_->type.setQualifiers(specifiers_->type.qualifiers() |
                                  Qualifiers::kConst);
}

void Semantics::visit(VolatileQualifierAST* ast) {
  specifiers_->type.setQualifiers(specifiers_->type.qualifiers() |
                                  Qualifiers::kVolatile);
}

void Semantics::visit(RestrictQualifierAST* ast) {
  specifiers_->type.setQualifiers(specifiers_->type.qualifiers() |
                                  Qualifiers::kRestrict);
}

void Semantics::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->name, &name);
  enumBase(ast->enumBase);
  for (auto it = ast->enumeratorList; it; it = it->next) enumerator(it->value);
  if (auto enumSymbol = dynamic_cast<EnumSymbol*>(ast->symbol))
    specifiers_->type.setType(types_->enumType(enumSymbol));
  else if (auto scopedEnumSymbol = dynamic_cast<ScopedEnumSymbol*>(ast->symbol))
    specifiers_->type.setType(types_->scopedEnumType(scopedEnumSymbol));
}

void Semantics::visit(ClassSpecifierAST* ast) {
  specifiers_->type.setType(types_->classType(ast->symbol));
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  NameSem name;
  this->name(ast->name, &name);
  baseClause(ast->baseClause);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(TypenameSpecifierAST* ast) {
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  NameSem name;
  this->name(ast->name, &name);
}

void Semantics::visit(IdDeclaratorAST* ast) {
  NameSem name;
  this->name(ast->name, &name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  declarator_->name = name.name;
  if (auto q = dynamic_cast<QualifiedNameAST*>(ast->name)) {
    declarator_->typeOrNamespaceSymbol = q->nestedNameSpecifier->symbol;
  }
}

void Semantics::visit(NestedDeclaratorAST* ast) { accept(ast->declarator); }

void Semantics::visit(PointerOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);

  SpecifiersSem specifiers;
  this->specifiers(ast->cvQualifierList, &specifiers);
  auto qualifiers = specifiers.type.qualifiers();

  QualifiedType ptrTy(types_->pointerType(declarator_->type, qualifiers));

  declarator_->type = ptrTy;
}

void Semantics::visit(ReferenceOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);

  if (ast->refOp == TokenKind::T_AMP) {
    QualifiedType refTy(types_->referenceType(declarator_->type));
    declarator_->type = refTy;
  } else {
    QualifiedType refTy(types_->rvalueReferenceType(declarator_->type));
    declarator_->type = refTy;
  }
}

void Semantics::visit(PtrToMemberOperatorAST* ast) {
  NestedNameSpecifierSem nestedNameSpecifierSem;
  nestedNameSpecifier(ast->nestedNameSpecifier, &nestedNameSpecifierSem);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  SpecifiersSem specifiers;
  this->specifiers(ast->cvQualifierList, &specifiers);
}

void Semantics::visit(FunctionDeclaratorAST* ast) {
  std::vector<QualifiedType> argumentTypes;
  bool isVariadic = false;

  if (ast->parametersAndQualifiers &&
      ast->parametersAndQualifiers->parameterDeclarationClause) {
    auto params = ast->parametersAndQualifiers->parameterDeclarationClause;

    for (auto it = params->parameterDeclarationList; it; it = it->next) {
      auto param = it->value;
      SpecifiersSem specifiers;
      this->specifiers(param->typeSpecifierList, &specifiers);
      if (!it->next && specifiers.type->asVoidType()) break;
      DeclaratorSem declarator{specifiers};
      this->declarator(param->declarator, &declarator);
      ExpressionSem expression;
      this->expression(param->expression, &expression);
      argumentTypes.push_back(declarator.type);
    }

    isVariadic = bool(params->ellipsisLoc);
  }

  trailingReturnType(ast->trailingReturnType);

  QualifiedType returnTy(declarator_->type);

  QualifiedType funTy(
      types_->functionType(returnTy, std::move(argumentTypes), isVariadic));

  declarator_->type = funTy;
}

void Semantics::visit(ArrayDeclaratorAST* ast) {
  if (!ast->expression) {
    QualifiedType arrayType(types_->unboundArrayType(declarator_->type));
    declarator_->type = arrayType;
  } else {
    ExpressionSem expression;
    this->expression(ast->expression, &expression);

    if (auto cst = dynamic_cast<IntLiteralExpressionAST*>(ast->expression)) {
      auto dim = std::stoul(cst->literal->value());
      QualifiedType arrayType(types_->arrayType(declarator_->type, dim));
      declarator_->type = arrayType;
    } else {
      QualifiedType arrayType(types_->unboundArrayType(declarator_->type));
      declarator_->type = arrayType;
    }
  }
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

QualifiedType Semantics::commonType(ExpressionAST* ast, ExpressionAST* other) {
  return QualifiedType();
}

void Semantics::implicitConversion(ExpressionAST* ast,
                                   const QualifiedType& type) {
  standardConversion(ast, type);
}

void Semantics::standardConversion(ExpressionAST* ast,
                                   const QualifiedType& type) {
  // tier 1
  if (lvalueToRvalueConversion(ast, type) ||
      arrayToPointerConversion(ast, type) ||
      functionToPointerConversion(ast, type)) {
    // tier 1
  }

  if (numericPromotion(ast, type) || numericConversion(ast, type)) {
    // tier 2
  }

  functionPointerConversion(ast, type);

  qualificationConversion(ast, type);
}

bool Semantics::lvalueToRvalueConversion(ExpressionAST* ast,
                                         const QualifiedType& type) {
  return false;
}

bool Semantics::arrayToPointerConversion(ExpressionAST* ast,
                                         const QualifiedType& type) {
  return false;
}

bool Semantics::functionToPointerConversion(ExpressionAST* ast,
                                            const QualifiedType& type) {
  return false;
}

bool Semantics::numericPromotion(ExpressionAST* ast,
                                 const QualifiedType& type) {
  return false;
}

bool Semantics::numericConversion(ExpressionAST* ast,
                                  const QualifiedType& type) {
  return false;
}

void Semantics::functionPointerConversion(ExpressionAST* ast,
                                          const QualifiedType& type) {}

void Semantics::qualificationConversion(ExpressionAST* ast,
                                        const QualifiedType& type) {}

}  // namespace cxx
