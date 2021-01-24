
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
  for (auto it = ast->typeSpecifierList; it; it = it->next) accept(it->value);
  accept(ast->declarator);
}

void RecursiveASTVisitor::visit(NestedNameSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(UsingDeclaratorAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(HandlerAST* ast) {
  accept(ast->exceptionDeclaration);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(TemplateArgumentAST* ast) {}

void RecursiveASTVisitor::visit(EnumBaseAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(EnumeratorAST* ast) {
  accept(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(DeclaratorAST* ast) {
  for (auto it = ast->ptrOpList; it; it = it->next) accept(it->value);
  accept(ast->coreDeclarator);
  for (auto it = ast->modifiers; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(EllipsisExceptionDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  for (auto it = ast->typeSpecifierList; it; it = it->next) accept(it->value);
  accept(ast->declarator);
}

void RecursiveASTVisitor::visit(TranslationUnitAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next) accept(it->value);
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

void RecursiveASTVisitor::visit(IdExpressionAST* ast) { accept(ast->name); }

void RecursiveASTVisitor::visit(NestedExpressionAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(BinaryExpressionAST* ast) {
  accept(ast->leftExpression);
  accept(ast->rightExpression);
}

void RecursiveASTVisitor::visit(AssignmentExpressionAST* ast) {
  accept(ast->leftExpression);
  accept(ast->rightExpression);
}

void RecursiveASTVisitor::visit(LabeledStatementAST* ast) {
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(CaseStatementAST* ast) {
  accept(ast->expression);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(DefaultStatementAST* ast) {
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(ExpressionStatementAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(IfStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->condition);
  accept(ast->statement);
  accept(ast->elseStatement);
}

void RecursiveASTVisitor::visit(SwitchStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->condition);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(WhileStatementAST* ast) {
  accept(ast->condition);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(DoStatementAST* ast) {
  accept(ast->statement);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(ForRangeStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->rangeDeclaration);
  accept(ast->rangeInitializer);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(ForStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->condition);
  accept(ast->expression);
  accept(ast->statement);
}

void RecursiveASTVisitor::visit(BreakStatementAST* ast) {}

void RecursiveASTVisitor::visit(ContinueStatementAST* ast) {}

void RecursiveASTVisitor::visit(ReturnStatementAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(GotoStatementAST* ast) {}

void RecursiveASTVisitor::visit(CoroutineReturnStatementAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(DeclarationStatementAST* ast) {
  accept(ast->declaration);
}

void RecursiveASTVisitor::visit(TryBlockStatementAST* ast) {
  accept(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(FunctionDefinitionAST* ast) {
  for (auto it = ast->declSpecifierList; it; it = it->next) accept(it->value);
  accept(ast->declarator);
  accept(ast->functionBody);
}

void RecursiveASTVisitor::visit(ConceptDefinitionAST* ast) {
  accept(ast->name);
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(ForRangeDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(AliasDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  accept(ast->typeId);
}

void RecursiveASTVisitor::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attributes; it; it = it->next) accept(it->value);
  for (auto it = ast->declSpecifierList; it; it = it->next) accept(it->value);
  for (auto it = ast->declaratorList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  accept(ast->expression);
}

void RecursiveASTVisitor::visit(EmptyDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  accept(ast->nestedNameSpecifier);
  accept(ast->name);
  accept(ast->enumBase);
}

void RecursiveASTVisitor::visit(UsingEnumDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(NamespaceDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  accept(ast->nestedNameSpecifier);
  accept(ast->name);
  for (auto it = ast->extraAttributeList; it; it = it->next) accept(it->value);
  for (auto it = ast->declarationList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->name);
}

void RecursiveASTVisitor::visit(UsingDirectiveAST* ast) {}

void RecursiveASTVisitor::visit(UsingDeclarationAST* ast) {
  for (auto it = ast->usingDeclaratorList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(AsmDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(LinkageSpecificationAST* ast) {}

void RecursiveASTVisitor::visit(ExportDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(ModuleImportDeclarationAST* ast) {}

void RecursiveASTVisitor::visit(TemplateDeclarationAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    accept(it->value);
  accept(ast->declaration);
}

void RecursiveASTVisitor::visit(DeductionGuideAST* ast) {}

void RecursiveASTVisitor::visit(ExplicitInstantiationAST* ast) {
  accept(ast->declaration);
}

void RecursiveASTVisitor::visit(SimpleNameAST* ast) {}

void RecursiveASTVisitor::visit(DestructorNameAST* ast) { accept(ast->name); }

void RecursiveASTVisitor::visit(DecltypeNameAST* ast) {
  accept(ast->decltypeSpecifier);
}

void RecursiveASTVisitor::visit(OperatorNameAST* ast) {}

void RecursiveASTVisitor::visit(TemplateNameAST* ast) {
  accept(ast->name);
  for (auto it = ast->templateArgumentList; it; it = it->next)
    accept(it->value);
}

void RecursiveASTVisitor::visit(SimpleSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ExplicitSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(NamedTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(PlaceholderTypeSpecifierHelperAST* ast) {}

void RecursiveASTVisitor::visit(DecltypeSpecifierTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(UnderlyingTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(AtomicTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(DecltypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(PlaceholderTypeSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(CvQualifierAST* ast) {}

void RecursiveASTVisitor::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  accept(ast->nestedNameSpecifier);
  accept(ast->name);
  accept(ast->enumBase);
  for (auto it = ast->enumeratorList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(ClassSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  accept(ast->name);
  for (auto it = ast->declarationList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(TypenameSpecifierAST* ast) {}

void RecursiveASTVisitor::visit(IdDeclaratorAST* ast) {
  accept(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(NestedDeclaratorAST* ast) {
  accept(ast->declarator);
}

void RecursiveASTVisitor::visit(PointerOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(ReferenceOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(PtrToMemberOperatorAST* ast) {
  accept(ast->nestedNameSpecifier);
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) accept(it->value);
}

void RecursiveASTVisitor::visit(FunctionDeclaratorAST* ast) {}

void RecursiveASTVisitor::visit(ArrayDeclaratorAST* ast) {
  accept(ast->expression);
  for (auto it = ast->attributeList; it; it = it->next) accept(it->value);
}

}  // namespace cxx
