
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

void RecursiveASTVisitor::visit(TypeIdAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
}

void RecursiveASTVisitor::visit(NestedNameSpecifierAST* ast) {
  for (auto it = ast->nameList; it; it = it->next) name(it->value);
}

void RecursiveASTVisitor::visit(UsingDeclaratorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void RecursiveASTVisitor::visit(HandlerAST* ast) {
  exceptionDeclaration(ast->exceptionDeclaration);
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(TemplateArgumentAST* ast) {}

void RecursiveASTVisitor::visit(EnumBaseAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
}

void RecursiveASTVisitor::visit(EnumeratorAST* ast) {
  name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(DeclaratorAST* ast) {
  for (auto it = ast->ptrOpList; it; it = it->next) ptrOperator(it->value);
  coreDeclarator(ast->coreDeclarator);
  for (auto it = ast->modifiers; it; it = it->next)
    declaratorModifier(it->value);
}

void RecursiveASTVisitor::visit(BaseSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  name(ast->name);
}

void RecursiveASTVisitor::visit(BaseClauseAST* ast) {
  for (auto it = ast->baseSpecifierList; it; it = it->next)
    baseSpecifier(it->value);
}

void RecursiveASTVisitor::visit(NewTypeIdAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
}

void RecursiveASTVisitor::visit(NewParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void RecursiveASTVisitor::visit(EllipsisExceptionDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
}

void RecursiveASTVisitor::visit(TranslationUnitAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void RecursiveASTVisitor::visit(ModuleUnitAST* ast) {}

void RecursiveASTVisitor::visit(ThisExpressionAST* ast) {}

void RecursiveASTVisitor::visit(CharLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(BoolLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(IntLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(FloatLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(NullptrLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(StringLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(UserDefinedStringLiteralExpressionAST* ast) {}

void RecursiveASTVisitor::visit(IdExpressionAST* ast) { name(ast->name); }

void RecursiveASTVisitor::visit(NestedExpressionAST* ast) {
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(BinaryExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void RecursiveASTVisitor::visit(AssignmentExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void RecursiveASTVisitor::visit(CallExpressionAST* ast) {
  expression(ast->baseExpression);
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void RecursiveASTVisitor::visit(SubscriptExpressionAST* ast) {
  expression(ast->baseExpression);
  expression(ast->indexExpression);
}

void RecursiveASTVisitor::visit(MemberExpressionAST* ast) {
  expression(ast->baseExpression);
  name(ast->name);
}

void RecursiveASTVisitor::visit(ConditionalExpressionAST* ast) {
  expression(ast->condition);
  expression(ast->iftrueExpression);
  expression(ast->iffalseExpression);
}

void RecursiveASTVisitor::visit(CppCastExpressionAST* ast) {
  typeId(ast->typeId);
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(NewExpressionAST* ast) {
  newTypeId(ast->typeId);
  newInitializer(ast->newInitalizer);
}

void RecursiveASTVisitor::visit(LabeledStatementAST* ast) {
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(CaseStatementAST* ast) {
  expression(ast->expression);
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(DefaultStatementAST* ast) {
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(ExpressionStatementAST* ast) {
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next) statement(it->value);
}

void RecursiveASTVisitor::visit(IfStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  statement(ast->statement);
  statement(ast->elseStatement);
}

void RecursiveASTVisitor::visit(SwitchStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(WhileStatementAST* ast) {
  expression(ast->condition);
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(DoStatementAST* ast) {
  statement(ast->statement);
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(ForRangeStatementAST* ast) {
  statement(ast->initializer);
  declaration(ast->rangeDeclaration);
  expression(ast->rangeInitializer);
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(ForStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  expression(ast->expression);
  statement(ast->statement);
}

void RecursiveASTVisitor::visit(BreakStatementAST* ast) {}

void RecursiveASTVisitor::visit(ContinueStatementAST* ast) {}

void RecursiveASTVisitor::visit(ReturnStatementAST* ast) {
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(GotoStatementAST* ast) {}

void RecursiveASTVisitor::visit(CoroutineReturnStatementAST* ast) {
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(DeclarationStatementAST* ast) {
  declaration(ast->declaration);
}

void RecursiveASTVisitor::visit(TryBlockStatementAST* ast) {
  statement(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) handler(it->value);
}

void RecursiveASTVisitor::visit(FunctionDefinitionAST* ast) {
  for (auto it = ast->declSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
  statement(ast->functionBody);
}

void RecursiveASTVisitor::visit(ConceptDefinitionAST* ast) {
  name(ast->name);
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(ForRangeDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(AliasDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  typeId(ast->typeId);
}

void RecursiveASTVisitor::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attributes; it; it = it->next) attribute(it->value);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    specifier(it->value);
  for (auto it = ast->declaratorList; it; it = it->next) declarator(it->value);
}

void RecursiveASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  expression(ast->expression);
}

void RecursiveASTVisitor::visit(EmptyDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void RecursiveASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  enumBase(ast->enumBase);
}

void RecursiveASTVisitor::visit(UsingEnumDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(NamespaceDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  for (auto it = ast->extraAttributeList; it; it = it->next)
    attribute(it->value);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void RecursiveASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void RecursiveASTVisitor::visit(UsingDirectiveAST* ast) {}

void RecursiveASTVisitor::visit(UsingDeclarationAST* ast) {
  for (auto it = ast->usingDeclaratorList; it; it = it->next)
    usingDeclarator(it->value);
}

void RecursiveASTVisitor::visit(AsmDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void RecursiveASTVisitor::visit(LinkageSpecificationAST* ast) {}

void RecursiveASTVisitor::visit(ExportDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(ModuleImportDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(TemplateDeclarationAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  declaration(ast->declaration);
}

void RecursiveASTVisitor::visit(DeductionGuideAST* ast) {}

void RecursiveASTVisitor::visit(ExplicitInstantiationAST* ast) {
  declaration(ast->declaration);
}

void RecursiveASTVisitor::visit(SimpleNameAST* ast) {}

void RecursiveASTVisitor::visit(DestructorNameAST* ast) { name(ast->name); }

void RecursiveASTVisitor::visit(DecltypeNameAST* ast) {
  specifier(ast->decltypeSpecifier);
}

void RecursiveASTVisitor::visit(OperatorNameAST* ast) {}

void RecursiveASTVisitor::visit(TemplateNameAST* ast) {
  name(ast->name);
  for (auto it = ast->templateArgumentList; it; it = it->next)
    templateArgument(it->value);
}

void RecursiveASTVisitor::visit(QualifiedNameAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void RecursiveASTVisitor::visit(SimpleSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ExplicitSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(NamedTypeSpecifierAST* ast) { name(ast->name); }

void RecursiveASTVisitor::visit(PlaceholderTypeSpecifierHelperAST* ast) {}

void RecursiveASTVisitor::visit(DecltypeSpecifierTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(UnderlyingTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(AtomicTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(DecltypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(PlaceholderTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(CvQualifierAST* ast) {}

void RecursiveASTVisitor::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  enumBase(ast->enumBase);
  for (auto it = ast->enumeratorList; it; it = it->next) enumerator(it->value);
}

void RecursiveASTVisitor::visit(ClassSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  name(ast->name);
  baseClause(ast->baseClause);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void RecursiveASTVisitor::visit(TypenameSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(IdDeclaratorAST* ast) {
  name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void RecursiveASTVisitor::visit(NestedDeclaratorAST* ast) {
  declarator(ast->declarator);
}

void RecursiveASTVisitor::visit(PointerOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
}

void RecursiveASTVisitor::visit(ReferenceOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void RecursiveASTVisitor::visit(PtrToMemberOperatorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
}

void RecursiveASTVisitor::visit(FunctionDeclaratorAST* ast) {}

void RecursiveASTVisitor::visit(ArrayDeclaratorAST* ast) {
  expression(ast->expression);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

}  // namespace cxx
