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
#include <cxx/recursive_ast_visitor.h>

namespace cxx {

void RecursiveASTVisitor::accept(AST* ast) {
  if (!ast) return;
  if (preVisit(ast)) ast->accept(this);
  postVisit(ast);
}

void RecursiveASTVisitor::acceptSpecifier(SpecifierAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptDeclarator(DeclaratorAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptName(NameAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptNestedNameSpecifier(
    NestedNameSpecifierAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptExceptionDeclaration(
    ExceptionDeclarationAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptCompoundStatement(CompoundStatementAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptAttribute(AttributeAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptExpression(ExpressionAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptPtrOperator(PtrOperatorAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptCoreDeclarator(CoreDeclaratorAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptDeclaratorModifier(DeclaratorModifierAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptRequiresClause(RequiresClauseAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptInitializer(InitializerAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptBaseSpecifier(BaseSpecifierAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptParameterDeclaration(
    ParameterDeclarationAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptParameterDeclarationClause(
    ParameterDeclarationClauseAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptLambdaCapture(LambdaCaptureAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptTrailingReturnType(TrailingReturnTypeAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptTypeId(TypeIdAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptMemInitializer(MemInitializerAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptRequirement(RequirementAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptDeclaration(DeclarationAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptModuleName(ModuleNameAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptModulePartition(ModulePartitionAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptTypeConstraint(TypeConstraintAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptBracedInitList(BracedInitListAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptCtorInitializer(CtorInitializerAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptHandler(HandlerAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptGlobalModuleFragment(
    GlobalModuleFragmentAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptModuleDeclaration(ModuleDeclarationAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptPrivateModuleFragment(
    PrivateModuleFragmentAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptRequirementBody(RequirementBodyAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptLambdaIntroducer(LambdaIntroducerAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptLambdaDeclarator(LambdaDeclaratorAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptNewTypeId(NewTypeIdAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptNewInitializer(NewInitializerAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptStatement(StatementAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptFunctionBody(FunctionBodyAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptInitDeclarator(InitDeclaratorAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptEnumBase(EnumBaseAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptUsingDeclarator(UsingDeclaratorAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptImportName(ImportNameAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptTemplateArgument(TemplateArgumentAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::acceptEnumerator(EnumeratorAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptBaseClause(BaseClauseAST* ast) { accept(ast); }

void RecursiveASTVisitor::acceptParametersAndQualifiers(
    ParametersAndQualifiersAST* ast) {
  accept(ast);
}

void RecursiveASTVisitor::visit(TypeIdAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
  acceptDeclarator(ast->declarator);
}

void RecursiveASTVisitor::visit(NestedNameSpecifierAST* ast) {
  for (auto it = ast->nameList; it; it = it->next) acceptName(it->value);
}

void RecursiveASTVisitor::visit(UsingDeclaratorAST* ast) {
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(HandlerAST* ast) {
  acceptExceptionDeclaration(ast->exceptionDeclaration);
  acceptCompoundStatement(ast->statement);
}

void RecursiveASTVisitor::visit(EnumBaseAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
}

void RecursiveASTVisitor::visit(EnumeratorAST* ast) {
  acceptName(ast->name);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(DeclaratorAST* ast) {
  for (auto it = ast->ptrOpList; it; it = it->next)
    acceptPtrOperator(it->value);
  acceptCoreDeclarator(ast->coreDeclarator);
  for (auto it = ast->modifiers; it; it = it->next)
    acceptDeclaratorModifier(it->value);
}

void RecursiveASTVisitor::visit(InitDeclaratorAST* ast) {
  acceptDeclarator(ast->declarator);
  acceptRequiresClause(ast->requiresClause);
  acceptInitializer(ast->initializer);
}

void RecursiveASTVisitor::visit(BaseSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(BaseClauseAST* ast) {
  for (auto it = ast->baseSpecifierList; it; it = it->next)
    acceptBaseSpecifier(it->value);
}

void RecursiveASTVisitor::visit(NewTypeIdAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
}

void RecursiveASTVisitor::visit(RequiresClauseAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(ParameterDeclarationClauseAST* ast) {
  for (auto it = ast->parameterDeclarationList; it; it = it->next)
    acceptParameterDeclaration(it->value);
}

void RecursiveASTVisitor::visit(ParametersAndQualifiersAST* ast) {
  acceptParameterDeclarationClause(ast->parameterDeclarationClause);
  for (auto it = ast->cvQualifierList; it; it = it->next)
    acceptSpecifier(it->value);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

void RecursiveASTVisitor::visit(LambdaIntroducerAST* ast) {
  for (auto it = ast->captureList; it; it = it->next)
    acceptLambdaCapture(it->value);
}

void RecursiveASTVisitor::visit(LambdaDeclaratorAST* ast) {
  acceptParameterDeclarationClause(ast->parameterDeclarationClause);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptTrailingReturnType(ast->trailingReturnType);
  acceptRequiresClause(ast->requiresClause);
}

void RecursiveASTVisitor::visit(TrailingReturnTypeAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(CtorInitializerAST* ast) {
  for (auto it = ast->memInitializerList; it; it = it->next)
    acceptMemInitializer(it->value);
}

void RecursiveASTVisitor::visit(RequirementBodyAST* ast) {
  for (auto it = ast->requirementList; it; it = it->next)
    acceptRequirement(it->value);
}

void RecursiveASTVisitor::visit(TypeConstraintAST* ast) {
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(GlobalModuleFragmentAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(PrivateModuleFragmentAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(ModuleDeclarationAST* ast) {
  acceptModuleName(ast->moduleName);
  acceptModulePartition(ast->modulePartition);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

void RecursiveASTVisitor::visit(ModuleNameAST* ast) {}

void RecursiveASTVisitor::visit(ImportNameAST* ast) {
  acceptModulePartition(ast->modulePartition);
  acceptModuleName(ast->moduleName);
}

void RecursiveASTVisitor::visit(ModulePartitionAST* ast) {
  acceptModuleName(ast->moduleName);
}

void RecursiveASTVisitor::visit(SimpleRequirementAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(CompoundRequirementAST* ast) {
  acceptExpression(ast->expression);
  acceptTypeConstraint(ast->typeConstraint);
}

void RecursiveASTVisitor::visit(TypeRequirementAST* ast) {
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(NestedRequirementAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(TypeTemplateArgumentAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(ExpressionTemplateArgumentAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(ParenMemInitializerAST* ast) {
  acceptName(ast->name);
  for (auto it = ast->expressionList; it; it = it->next)
    acceptExpression(it->value);
}

void RecursiveASTVisitor::visit(BracedMemInitializerAST* ast) {
  acceptName(ast->name);
  acceptBracedInitList(ast->bracedInitList);
}

void RecursiveASTVisitor::visit(ThisLambdaCaptureAST* ast) {}

void RecursiveASTVisitor::visit(DerefThisLambdaCaptureAST* ast) {}

void RecursiveASTVisitor::visit(SimpleLambdaCaptureAST* ast) {}

void RecursiveASTVisitor::visit(RefLambdaCaptureAST* ast) {}

void RecursiveASTVisitor::visit(RefInitLambdaCaptureAST* ast) {
  acceptInitializer(ast->initializer);
}

void RecursiveASTVisitor::visit(InitLambdaCaptureAST* ast) {
  acceptInitializer(ast->initializer);
}

void RecursiveASTVisitor::visit(EqualInitializerAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(BracedInitListAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next)
    acceptExpression(it->value);
}

void RecursiveASTVisitor::visit(ParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next)
    acceptExpression(it->value);
}

void RecursiveASTVisitor::visit(NewParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next)
    acceptExpression(it->value);
}

void RecursiveASTVisitor::visit(NewBracedInitializerAST* ast) {
  acceptBracedInitList(ast->bracedInit);
}

void RecursiveASTVisitor::visit(EllipsisExceptionDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
  acceptDeclarator(ast->declarator);
}

void RecursiveASTVisitor::visit(DefaultFunctionBodyAST* ast) {}

void RecursiveASTVisitor::visit(CompoundStatementFunctionBodyAST* ast) {
  acceptCtorInitializer(ast->ctorInitializer);
  acceptCompoundStatement(ast->statement);
}

void RecursiveASTVisitor::visit(TryStatementFunctionBodyAST* ast) {
  acceptCtorInitializer(ast->ctorInitializer);
  acceptCompoundStatement(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) acceptHandler(it->value);
}

void RecursiveASTVisitor::visit(DeleteFunctionBodyAST* ast) {}

void RecursiveASTVisitor::visit(TranslationUnitAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(ModuleUnitAST* ast) {
  acceptGlobalModuleFragment(ast->globalModuleFragment);
  acceptModuleDeclaration(ast->moduleDeclaration);
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
  acceptPrivateModuleFragment(ast->privateModuleFragmentAST);
}

void RecursiveASTVisitor::visit(ThisExpressionAST* ast) {}

void RecursiveASTVisitor::visit(CharLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(BoolLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(IntLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(FloatLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(NullptrLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(StringLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(UserDefinedStringLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(IdExpressionAST* ast) { acceptName(ast->name); }

void RecursiveASTVisitor::visit(RequiresExpressionAST* ast) {
  acceptParameterDeclarationClause(ast->parameterDeclarationClause);
  acceptRequirementBody(ast->requirementBody);
}

void RecursiveASTVisitor::visit(NestedExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(RightFoldExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(LeftFoldExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(FoldExpressionAST* ast) {
  acceptExpression(ast->leftExpression);
  acceptExpression(ast->rightExpression);
}

void RecursiveASTVisitor::visit(LambdaExpressionAST* ast) {
  acceptLambdaIntroducer(ast->lambdaIntroducer);
  for (auto it = ast->templateParameterList; it; it = it->next)
    acceptDeclaration(it->value);
  acceptRequiresClause(ast->requiresClause);
  acceptLambdaDeclarator(ast->lambdaDeclarator);
  acceptCompoundStatement(ast->statement);
}

void RecursiveASTVisitor::visit(SizeofExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(SizeofTypeExpressionAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(SizeofPackExpressionAST* ast) {}

void RecursiveASTVisitor::visit(TypeidExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(TypeidOfTypeExpressionAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(AlignofExpressionAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(IsSameAsExpressionAST* ast) {
  acceptTypeId(ast->typeId);
  acceptTypeId(ast->otherTypeId);
}

void RecursiveASTVisitor::visit(UnaryExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(BinaryExpressionAST* ast) {
  acceptExpression(ast->leftExpression);
  acceptExpression(ast->rightExpression);
}

void RecursiveASTVisitor::visit(AssignmentExpressionAST* ast) {
  acceptExpression(ast->leftExpression);
  acceptExpression(ast->rightExpression);
}

void RecursiveASTVisitor::visit(BracedTypeConstructionAST* ast) {
  acceptSpecifier(ast->typeSpecifier);
  acceptBracedInitList(ast->bracedInitList);
}

void RecursiveASTVisitor::visit(TypeConstructionAST* ast) {
  acceptSpecifier(ast->typeSpecifier);
  for (auto it = ast->expressionList; it; it = it->next)
    acceptExpression(it->value);
}

void RecursiveASTVisitor::visit(CallExpressionAST* ast) {
  acceptExpression(ast->baseExpression);
  for (auto it = ast->expressionList; it; it = it->next)
    acceptExpression(it->value);
}

void RecursiveASTVisitor::visit(SubscriptExpressionAST* ast) {
  acceptExpression(ast->baseExpression);
  acceptExpression(ast->indexExpression);
}

void RecursiveASTVisitor::visit(MemberExpressionAST* ast) {
  acceptExpression(ast->baseExpression);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(PostIncrExpressionAST* ast) {
  acceptExpression(ast->baseExpression);
}

void RecursiveASTVisitor::visit(ConditionalExpressionAST* ast) {
  acceptExpression(ast->condition);
  acceptExpression(ast->iftrueExpression);
  acceptExpression(ast->iffalseExpression);
}

void RecursiveASTVisitor::visit(ImplicitCastExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(CastExpressionAST* ast) {
  acceptTypeId(ast->typeId);
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(CppCastExpressionAST* ast) {
  acceptTypeId(ast->typeId);
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(NewExpressionAST* ast) {
  acceptNewTypeId(ast->typeId);
  acceptNewInitializer(ast->newInitalizer);
}

void RecursiveASTVisitor::visit(DeleteExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(ThrowExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(NoexceptExpressionAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(LabeledStatementAST* ast) {
  acceptStatement(ast->statement);
}

void RecursiveASTVisitor::visit(CaseStatementAST* ast) {
  acceptExpression(ast->expression);
  acceptStatement(ast->statement);
}

void RecursiveASTVisitor::visit(DefaultStatementAST* ast) {
  acceptStatement(ast->statement);
}

void RecursiveASTVisitor::visit(ExpressionStatementAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next)
    acceptStatement(it->value);
}

void RecursiveASTVisitor::visit(IfStatementAST* ast) {
  acceptStatement(ast->initializer);
  acceptExpression(ast->condition);
  acceptStatement(ast->statement);
  acceptStatement(ast->elseStatement);
}

void RecursiveASTVisitor::visit(SwitchStatementAST* ast) {
  acceptStatement(ast->initializer);
  acceptExpression(ast->condition);
  acceptStatement(ast->statement);
}

void RecursiveASTVisitor::visit(WhileStatementAST* ast) {
  acceptExpression(ast->condition);
  acceptStatement(ast->statement);
}

void RecursiveASTVisitor::visit(DoStatementAST* ast) {
  acceptStatement(ast->statement);
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(ForRangeStatementAST* ast) {
  acceptStatement(ast->initializer);
  acceptDeclaration(ast->rangeDeclaration);
  acceptExpression(ast->rangeInitializer);
  acceptStatement(ast->statement);
}

void RecursiveASTVisitor::visit(ForStatementAST* ast) {
  acceptStatement(ast->initializer);
  acceptExpression(ast->condition);
  acceptExpression(ast->expression);
  acceptStatement(ast->statement);
}

void RecursiveASTVisitor::visit(BreakStatementAST* ast) {}

void RecursiveASTVisitor::visit(ContinueStatementAST* ast) {}

void RecursiveASTVisitor::visit(ReturnStatementAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(GotoStatementAST* ast) {}

void RecursiveASTVisitor::visit(CoroutineReturnStatementAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(DeclarationStatementAST* ast) {
  acceptDeclaration(ast->declaration);
}

void RecursiveASTVisitor::visit(TryBlockStatementAST* ast) {
  acceptCompoundStatement(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) acceptHandler(it->value);
}

void RecursiveASTVisitor::visit(AccessDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(FunctionDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
  acceptDeclarator(ast->declarator);
  acceptRequiresClause(ast->requiresClause);
  acceptFunctionBody(ast->functionBody);
}

void RecursiveASTVisitor::visit(ConceptDefinitionAST* ast) {
  acceptName(ast->name);
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(ForRangeDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(AliasDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
  for (auto it = ast->initDeclaratorList; it; it = it->next)
    acceptInitDeclarator(it->value);
  acceptRequiresClause(ast->requiresClause);
}

void RecursiveASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(EmptyDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

void RecursiveASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
  acceptEnumBase(ast->enumBase);
}

void RecursiveASTVisitor::visit(UsingEnumDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(NamespaceDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
  for (auto it = ast->extraAttributeList; it; it = it->next)
    acceptAttribute(it->value);
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(UsingDirectiveAST* ast) {}

void RecursiveASTVisitor::visit(UsingDeclarationAST* ast) {
  for (auto it = ast->usingDeclaratorList; it; it = it->next)
    acceptUsingDeclarator(it->value);
}

void RecursiveASTVisitor::visit(AsmDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

void RecursiveASTVisitor::visit(ExportDeclarationAST* ast) {
  acceptDeclaration(ast->declaration);
}

void RecursiveASTVisitor::visit(ExportCompoundDeclarationAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(ModuleImportDeclarationAST* ast) {
  acceptImportName(ast->importName);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

void RecursiveASTVisitor::visit(TemplateDeclarationAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    acceptDeclaration(it->value);
  acceptRequiresClause(ast->requiresClause);
  acceptDeclaration(ast->declaration);
}

void RecursiveASTVisitor::visit(TypenameTypeParameterAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(TypenamePackTypeParameterAST* ast) {}

void RecursiveASTVisitor::visit(TemplateTypeParameterAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    acceptDeclaration(it->value);
  acceptRequiresClause(ast->requiresClause);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(TemplatePackTypeParameterAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(DeductionGuideAST* ast) {}

void RecursiveASTVisitor::visit(ExplicitInstantiationAST* ast) {
  acceptDeclaration(ast->declaration);
}

void RecursiveASTVisitor::visit(ParameterDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    acceptSpecifier(it->value);
  acceptDeclarator(ast->declarator);
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(LinkageSpecificationAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(SimpleNameAST* ast) {}

void RecursiveASTVisitor::visit(DestructorNameAST* ast) { acceptName(ast->id); }

void RecursiveASTVisitor::visit(DecltypeNameAST* ast) {
  acceptSpecifier(ast->decltypeSpecifier);
}

void RecursiveASTVisitor::visit(OperatorNameAST* ast) {}

void RecursiveASTVisitor::visit(ConversionNameAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(TemplateNameAST* ast) {
  acceptName(ast->id);
  for (auto it = ast->templateArgumentList; it; it = it->next)
    acceptTemplateArgument(it->value);
}

void RecursiveASTVisitor::visit(QualifiedNameAST* ast) {
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->id);
}

void RecursiveASTVisitor::visit(TypedefSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(FriendSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ConstevalSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ConstinitSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ConstexprSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(InlineSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(StaticSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ExternSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ThreadLocalSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ThreadSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(MutableSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(VirtualSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ExplicitSpecifierAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(AutoTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(VoidTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(VaListTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(IntegralTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(FloatingPointTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ComplexTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(NamedTypeSpecifierAST* ast) {
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(AtomicTypeSpecifierAST* ast) {
  acceptTypeId(ast->typeId);
}

void RecursiveASTVisitor::visit(UnderlyingTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(DecltypeAutoSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(DecltypeSpecifierAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(TypeofSpecifierAST* ast) {
  acceptExpression(ast->expression);
}

void RecursiveASTVisitor::visit(PlaceholderTypeSpecifierAST* ast) {
  acceptTypeConstraint(ast->typeConstraint);
  acceptSpecifier(ast->specifier);
}

void RecursiveASTVisitor::visit(ConstQualifierAST* ast) {}

void RecursiveASTVisitor::visit(VolatileQualifierAST* ast) {}

void RecursiveASTVisitor::visit(RestrictQualifierAST* ast) {}

void RecursiveASTVisitor::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
  acceptEnumBase(ast->enumBase);
  for (auto it = ast->enumeratorList; it; it = it->next)
    acceptEnumerator(it->value);
}

void RecursiveASTVisitor::visit(ClassSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  acceptName(ast->name);
  acceptBaseClause(ast->baseClause);
  for (auto it = ast->declarationList; it; it = it->next)
    acceptDeclaration(it->value);
}

void RecursiveASTVisitor::visit(TypenameSpecifierAST* ast) {
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  acceptName(ast->name);
}

void RecursiveASTVisitor::visit(IdDeclaratorAST* ast) {
  acceptName(ast->name);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

void RecursiveASTVisitor::visit(NestedDeclaratorAST* ast) {
  acceptDeclarator(ast->declarator);
}

void RecursiveASTVisitor::visit(PointerOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next)
    acceptSpecifier(it->value);
}

void RecursiveASTVisitor::visit(ReferenceOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

void RecursiveASTVisitor::visit(PtrToMemberOperatorAST* ast) {
  acceptNestedNameSpecifier(ast->nestedNameSpecifier);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next)
    acceptSpecifier(it->value);
}

void RecursiveASTVisitor::visit(FunctionDeclaratorAST* ast) {
  acceptParametersAndQualifiers(ast->parametersAndQualifiers);
  acceptTrailingReturnType(ast->trailingReturnType);
}

void RecursiveASTVisitor::visit(ArrayDeclaratorAST* ast) {
  acceptExpression(ast->expression);
  for (auto it = ast->attributeList; it; it = it->next)
    acceptAttribute(it->value);
}

}  // namespace cxx
