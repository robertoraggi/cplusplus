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

void RecursiveASTVisitor::visit(TypeIdAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->typeSpecifierList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->declarator) ast->declarator->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(NestedNameSpecifierAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(UsingDeclaratorAST* ast) {
  if (preVisit(ast)) {
    if (ast->nestedNameSpecifier) ast->nestedNameSpecifier->accept(this);
    if (ast->name) ast->name->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(HandlerAST* ast) {
  if (preVisit(ast)) {
    if (ast->exceptionDeclaration) ast->exceptionDeclaration->accept(this);
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(TemplateArgumentAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(EnumBaseAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->typeSpecifierList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(EnumeratorAST* ast) {
  if (preVisit(ast)) {
    if (ast->name) ast->name->accept(this);
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DeclaratorAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->ptrOpList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->coreDeclarator) ast->coreDeclarator->accept(this);
    for (auto it = ast->modifiers; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(EllipsisExceptionDeclarationAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(TypeExceptionDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    for (auto it = ast->typeSpecifierList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->declarator) ast->declarator->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(TranslationUnitAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->declarationList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ModuleUnitAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(ThisExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(CharLiteralExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(BoolLiteralExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(IntLiteralExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(FloatLiteralExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(NullptrLiteralExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(StringLiteralExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(UserDefinedStringLiteralExpressionAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(IdExpressionAST* ast) {
  if (preVisit(ast)) {
    if (ast->name) ast->name->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(NestedExpressionAST* ast) {
  if (preVisit(ast)) {
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(BinaryExpressionAST* ast) {
  if (preVisit(ast)) {
    if (ast->leftExpression) ast->leftExpression->accept(this);
    if (ast->rightExpression) ast->rightExpression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(AssignmentExpressionAST* ast) {
  if (preVisit(ast)) {
    if (ast->leftExpression) ast->leftExpression->accept(this);
    if (ast->rightExpression) ast->rightExpression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(LabeledStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(CaseStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->expression) ast->expression->accept(this);
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DefaultStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ExpressionStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(CompoundStatementAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->statementList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(IfStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->initializer) ast->initializer->accept(this);
    if (ast->condition) ast->condition->accept(this);
    if (ast->statement) ast->statement->accept(this);
    if (ast->elseStatement) ast->elseStatement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(SwitchStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->initializer) ast->initializer->accept(this);
    if (ast->condition) ast->condition->accept(this);
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(WhileStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->condition) ast->condition->accept(this);
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DoStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->statement) ast->statement->accept(this);
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ForRangeStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->initializer) ast->initializer->accept(this);
    if (ast->rangeDeclaration) ast->rangeDeclaration->accept(this);
    if (ast->rangeInitializer) ast->rangeInitializer->accept(this);
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ForStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->initializer) ast->initializer->accept(this);
    if (ast->condition) ast->condition->accept(this);
    if (ast->expression) ast->expression->accept(this);
    if (ast->statement) ast->statement->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(BreakStatementAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ContinueStatementAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ReturnStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(GotoStatementAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(CoroutineReturnStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DeclarationStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->declaration) ast->declaration->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(TryBlockStatementAST* ast) {
  if (preVisit(ast)) {
    if (ast->statement) ast->statement->accept(this);
    for (auto it = ast->handlerList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(FunctionDefinitionAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->declSpecifierList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->declarator) ast->declarator->accept(this);
    if (ast->functionBody) ast->functionBody->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ConceptDefinitionAST* ast) {
  if (preVisit(ast)) {
    if (ast->name) ast->name->accept(this);
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ForRangeDeclarationAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(AliasDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->typeId) ast->typeId->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(SimpleDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributes; it; it = it->next)
      if (it->value) it->value->accept(this);
    for (auto it = ast->declSpecifierList; it; it = it->next)
      if (it->value) it->value->accept(this);
    for (auto it = ast->declaratorList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  if (preVisit(ast)) {
    if (ast->expression) ast->expression->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(EmptyDeclarationAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(AttributeDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->nestedNameSpecifier) ast->nestedNameSpecifier->accept(this);
    if (ast->name) ast->name->accept(this);
    if (ast->enumBase) ast->enumBase->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(UsingEnumDeclarationAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(NamespaceDefinitionAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->nestedNameSpecifier) ast->nestedNameSpecifier->accept(this);
    if (ast->name) ast->name->accept(this);
    for (auto it = ast->extraAttributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    for (auto it = ast->declarationList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  if (preVisit(ast)) {
    if (ast->nestedNameSpecifier) ast->nestedNameSpecifier->accept(this);
    if (ast->name) ast->name->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(UsingDirectiveAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(UsingDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->usingDeclaratorList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(AsmDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(LinkageSpecificationAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ExportDeclarationAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(ModuleImportDeclarationAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(TemplateDeclarationAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->templateParameterList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->declaration) ast->declaration->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DeductionGuideAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(ExplicitInstantiationAST* ast) {
  if (preVisit(ast)) {
    if (ast->declaration) ast->declaration->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(SimpleNameAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DestructorNameAST* ast) {
  if (preVisit(ast)) {
    if (ast->name) ast->name->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DecltypeNameAST* ast) {
  if (preVisit(ast)) {
    if (ast->decltypeSpecifier) ast->decltypeSpecifier->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(OperatorNameAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(TemplateNameAST* ast) {
  if (preVisit(ast)) {
    if (ast->name) ast->name->accept(this);
    for (auto it = ast->templateArgumentList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(SimpleSpecifierAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(ExplicitSpecifierAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(NamedTypeSpecifierAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(PlaceholderTypeSpecifierHelperAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DecltypeSpecifierTypeSpecifierAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(UnderlyingTypeSpecifierAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(AtomicTypeSpecifierAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(DecltypeSpecifierAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(PlaceholderTypeSpecifierAST* ast) {
  postVisit(ast);
}

void RecursiveASTVisitor::visit(CvQualifierAST* ast) {
  if (preVisit(ast)) {
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(EnumSpecifierAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->nestedNameSpecifier) ast->nestedNameSpecifier->accept(this);
    if (ast->name) ast->name->accept(this);
    if (ast->enumBase) ast->enumBase->accept(this);
    for (auto it = ast->enumeratorList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ClassSpecifierAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    if (ast->name) ast->name->accept(this);
    for (auto it = ast->declarationList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(TypenameSpecifierAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(IdDeclaratorAST* ast) {
  if (preVisit(ast)) {
    if (ast->name) ast->name->accept(this);
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(NestedDeclaratorAST* ast) {
  if (preVisit(ast)) {
    if (ast->declarator) ast->declarator->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(PointerOperatorAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    for (auto it = ast->cvQualifierList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(ReferenceOperatorAST* ast) {
  if (preVisit(ast)) {
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(PtrToMemberOperatorAST* ast) {
  if (preVisit(ast)) {
    if (ast->nestedNameSpecifier) ast->nestedNameSpecifier->accept(this);
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
    for (auto it = ast->cvQualifierList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

void RecursiveASTVisitor::visit(FunctionDeclaratorAST* ast) { postVisit(ast); }

void RecursiveASTVisitor::visit(ArrayDeclaratorAST* ast) {
  if (preVisit(ast)) {
    if (ast->expression) ast->expression->accept(this);
    for (auto it = ast->attributeList; it; it = it->next)
      if (it->value) it->value->accept(this);
  }
  postVisit(ast);
}

}  // namespace cxx