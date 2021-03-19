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
#include <cxx/names.h>
#include <cxx/semantics.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

namespace cxx {

void Semantics::operator()(TranslationUnit* unit) {
  std::swap(unit_, unit);
  accept(unit_->ast());
  std::swap(unit_, unit);
}

void Semantics::specifier(SpecifierAST* ast) { accept(ast); }

void Semantics::declarator(DeclaratorAST* ast) { accept(ast); }

const Name* Semantics::name(NameAST* ast) {
  const Name* name = nullptr;
  std::swap(name_, name);
  accept(ast);
  std::swap(name_, name);
  return name;
}

void Semantics::nestedNameSpecifier(NestedNameSpecifierAST* ast) {
  accept(ast);
}

void Semantics::exceptionDeclaration(ExceptionDeclarationAST* ast) {
  accept(ast);
}

void Semantics::compoundStatement(CompoundStatementAST* ast) { accept(ast); }

void Semantics::attribute(AttributeAST* ast) { accept(ast); }

void Semantics::expression(ExpressionAST* ast) { accept(ast); }

void Semantics::ptrOperator(PtrOperatorAST* ast) { accept(ast); }

void Semantics::coreDeclarator(CoreDeclaratorAST* ast) { accept(ast); }

void Semantics::declaratorModifier(DeclaratorModifierAST* ast) { accept(ast); }

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

void Semantics::typeId(TypeIdAST* ast) { accept(ast); }

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

void Semantics::initDeclarator(InitDeclaratorAST* ast) { accept(ast); }

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
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
}

void Semantics::visit(NestedNameSpecifierAST* ast) {
  for (auto it = ast->nameList; it; it = it->next) name(it->value);
}

void Semantics::visit(UsingDeclaratorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void Semantics::visit(HandlerAST* ast) {
  exceptionDeclaration(ast->exceptionDeclaration);
  compoundStatement(ast->statement);
}

void Semantics::visit(TemplateArgumentAST* ast) {}

void Semantics::visit(EnumBaseAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
}

void Semantics::visit(EnumeratorAST* ast) {
  name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  expression(ast->expression);
}

void Semantics::visit(DeclaratorAST* ast) {
  for (auto it = ast->ptrOpList; it; it = it->next) ptrOperator(it->value);
  coreDeclarator(ast->coreDeclarator);
  for (auto it = ast->modifiers; it; it = it->next)
    declaratorModifier(it->value);
}

void Semantics::visit(InitDeclaratorAST* ast) {
  declarator(ast->declarator);
  initializer(ast->initializer);
}

void Semantics::visit(BaseSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  name(ast->name);
}

void Semantics::visit(BaseClauseAST* ast) {
  for (auto it = ast->baseSpecifierList; it; it = it->next)
    baseSpecifier(it->value);
}

void Semantics::visit(NewTypeIdAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
}

void Semantics::visit(ParameterDeclarationClauseAST* ast) {
  for (auto it = ast->parameterDeclarationList; it; it = it->next)
    parameterDeclaration(it->value);
}

void Semantics::visit(ParametersAndQualifiersAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(LambdaIntroducerAST* ast) {
  for (auto it = ast->captureList; it; it = it->next) lambdaCapture(it->value);
}

void Semantics::visit(LambdaDeclaratorAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    specifier(it->value);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  trailingReturnType(ast->trailingReturnType);
}

void Semantics::visit(TrailingReturnTypeAST* ast) { typeId(ast->typeId); }

void Semantics::visit(CtorInitializerAST* ast) {
  for (auto it = ast->memInitializerList; it; it = it->next)
    memInitializer(it->value);
}

void Semantics::visit(ParenMemInitializerAST* ast) {
  name(ast->name);
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Semantics::visit(BracedMemInitializerAST* ast) {
  name(ast->name);
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

void Semantics::visit(EqualInitializerAST* ast) { expression(ast->expression); }

void Semantics::visit(BracedInitListAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Semantics::visit(ParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Semantics::visit(NewParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Semantics::visit(NewBracedInitializerAST* ast) {
  bracedInitList(ast->bracedInit);
}

void Semantics::visit(EllipsisExceptionDeclarationAST* ast) {}

void Semantics::visit(TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
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

void Semantics::visit(CharLiteralExpressionAST* ast) {}

void Semantics::visit(BoolLiteralExpressionAST* ast) {}

void Semantics::visit(IntLiteralExpressionAST* ast) {}

void Semantics::visit(FloatLiteralExpressionAST* ast) {}

void Semantics::visit(NullptrLiteralExpressionAST* ast) {}

void Semantics::visit(StringLiteralExpressionAST* ast) {}

void Semantics::visit(UserDefinedStringLiteralExpressionAST* ast) {}

void Semantics::visit(IdExpressionAST* ast) { name(ast->name); }

void Semantics::visit(NestedExpressionAST* ast) { expression(ast->expression); }

void Semantics::visit(RightFoldExpressionAST* ast) {
  expression(ast->expression);
}

void Semantics::visit(LeftFoldExpressionAST* ast) {
  expression(ast->expression);
}

void Semantics::visit(FoldExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void Semantics::visit(LambdaExpressionAST* ast) {
  lambdaIntroducer(ast->lambdaIntroducer);
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  lambdaDeclarator(ast->lambdaDeclarator);
  compoundStatement(ast->statement);
}

void Semantics::visit(SizeofExpressionAST* ast) { expression(ast->expression); }

void Semantics::visit(SizeofTypeExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(SizeofPackExpressionAST* ast) {}

void Semantics::visit(TypeidExpressionAST* ast) { expression(ast->expression); }

void Semantics::visit(TypeidOfTypeExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(AlignofExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(UnaryExpressionAST* ast) { expression(ast->expression); }

void Semantics::visit(BinaryExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void Semantics::visit(AssignmentExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void Semantics::visit(BracedTypeConstructionAST* ast) {
  specifier(ast->typeSpecifier);
  bracedInitList(ast->bracedInitList);
}

void Semantics::visit(TypeConstructionAST* ast) {
  specifier(ast->typeSpecifier);
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Semantics::visit(CallExpressionAST* ast) {
  expression(ast->baseExpression);
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Semantics::visit(SubscriptExpressionAST* ast) {
  expression(ast->baseExpression);
  expression(ast->indexExpression);
}

void Semantics::visit(MemberExpressionAST* ast) {
  expression(ast->baseExpression);
  name(ast->name);
}

void Semantics::visit(ConditionalExpressionAST* ast) {
  expression(ast->condition);
  expression(ast->iftrueExpression);
  expression(ast->iffalseExpression);
}

void Semantics::visit(CastExpressionAST* ast) {
  typeId(ast->typeId);
  expression(ast->expression);
}

void Semantics::visit(CppCastExpressionAST* ast) {
  typeId(ast->typeId);
  expression(ast->expression);
}

void Semantics::visit(NewExpressionAST* ast) {
  newTypeId(ast->typeId);
  newInitializer(ast->newInitalizer);
}

void Semantics::visit(DeleteExpressionAST* ast) { expression(ast->expression); }

void Semantics::visit(ThrowExpressionAST* ast) { expression(ast->expression); }

void Semantics::visit(NoexceptExpressionAST* ast) {
  expression(ast->expression);
}

void Semantics::visit(LabeledStatementAST* ast) { statement(ast->statement); }

void Semantics::visit(CaseStatementAST* ast) {
  expression(ast->expression);
  statement(ast->statement);
}

void Semantics::visit(DefaultStatementAST* ast) { statement(ast->statement); }

void Semantics::visit(ExpressionStatementAST* ast) {
  expression(ast->expression);
}

void Semantics::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next) statement(it->value);
}

void Semantics::visit(IfStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  statement(ast->statement);
  statement(ast->elseStatement);
}

void Semantics::visit(SwitchStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  statement(ast->statement);
}

void Semantics::visit(WhileStatementAST* ast) {
  expression(ast->condition);
  statement(ast->statement);
}

void Semantics::visit(DoStatementAST* ast) {
  statement(ast->statement);
  expression(ast->expression);
}

void Semantics::visit(ForRangeStatementAST* ast) {
  statement(ast->initializer);
  declaration(ast->rangeDeclaration);
  expression(ast->rangeInitializer);
  statement(ast->statement);
}

void Semantics::visit(ForStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  expression(ast->expression);
  statement(ast->statement);
}

void Semantics::visit(BreakStatementAST* ast) {}

void Semantics::visit(ContinueStatementAST* ast) {}

void Semantics::visit(ReturnStatementAST* ast) { expression(ast->expression); }

void Semantics::visit(GotoStatementAST* ast) {}

void Semantics::visit(CoroutineReturnStatementAST* ast) {
  expression(ast->expression);
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
  for (auto it = ast->declSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
  functionBody(ast->functionBody);
}

void Semantics::visit(ConceptDefinitionAST* ast) {
  name(ast->name);
  expression(ast->expression);
}

void Semantics::visit(ForRangeDeclarationAST* ast) {}

void Semantics::visit(AliasDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  typeId(ast->typeId);
}

void Semantics::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attributes; it; it = it->next) attribute(it->value);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    specifier(it->value);
  for (auto it = ast->initDeclaratorList; it; it = it->next)
    initDeclarator(it->value);
}

void Semantics::visit(StaticAssertDeclarationAST* ast) {
  expression(ast->expression);
}

void Semantics::visit(EmptyDeclarationAST* ast) {}

void Semantics::visit(AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  enumBase(ast->enumBase);
}

void Semantics::visit(UsingEnumDeclarationAST* ast) {}

void Semantics::visit(NamespaceDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  for (auto it = ast->extraAttributeList; it; it = it->next)
    attribute(it->value);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(NamespaceAliasDefinitionAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
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
  name(ast->name);
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
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
  expression(ast->expression);
}

void Semantics::visit(LinkageSpecificationAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(SimpleNameAST* ast) {
  auto id = unit_->identifier(ast->identifierLoc);
  name_ = id;
}

void Semantics::visit(DestructorNameAST* ast) { name(ast->id); }

void Semantics::visit(DecltypeNameAST* ast) {
  specifier(ast->decltypeSpecifier);
}

void Semantics::visit(OperatorNameAST* ast) {}

void Semantics::visit(TemplateNameAST* ast) {
  name(ast->id);
  for (auto it = ast->templateArgumentList; it; it = it->next)
    templateArgument(it->value);
}

void Semantics::visit(QualifiedNameAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->id);
}

void Semantics::visit(TypedefSpecifierAST* ast) {}

void Semantics::visit(FriendSpecifierAST* ast) {}

void Semantics::visit(ConstevalSpecifierAST* ast) {}

void Semantics::visit(ConstinitSpecifierAST* ast) {}

void Semantics::visit(ConstexprSpecifierAST* ast) {}

void Semantics::visit(InlineSpecifierAST* ast) {}

void Semantics::visit(StaticSpecifierAST* ast) {}

void Semantics::visit(ExternSpecifierAST* ast) {}

void Semantics::visit(ThreadLocalSpecifierAST* ast) {}

void Semantics::visit(ThreadSpecifierAST* ast) {}

void Semantics::visit(MutableSpecifierAST* ast) {}

void Semantics::visit(VirtualSpecifierAST* ast) {}

void Semantics::visit(ExplicitSpecifierAST* ast) {
  expression(ast->expression);
}

void Semantics::visit(AutoTypeSpecifierAST* ast) {}

void Semantics::visit(VoidTypeSpecifierAST* ast) {}

void Semantics::visit(VaListTypeSpecifierAST* ast) {}

void Semantics::visit(IntegralTypeSpecifierAST* ast) {}

void Semantics::visit(FloatingPointTypeSpecifierAST* ast) {}

void Semantics::visit(ComplexTypeSpecifierAST* ast) {}

void Semantics::visit(NamedTypeSpecifierAST* ast) { name(ast->name); }

void Semantics::visit(AtomicTypeSpecifierAST* ast) { typeId(ast->typeId); }

void Semantics::visit(UnderlyingTypeSpecifierAST* ast) {}

void Semantics::visit(ElaboratedTypeSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void Semantics::visit(DecltypeAutoSpecifierAST* ast) {}

void Semantics::visit(DecltypeSpecifierAST* ast) {
  expression(ast->expression);
}

void Semantics::visit(TypeofSpecifierAST* ast) { expression(ast->expression); }

void Semantics::visit(PlaceholderTypeSpecifierAST* ast) {}

void Semantics::visit(ConstQualifierAST* ast) {}

void Semantics::visit(VolatileQualifierAST* ast) {}

void Semantics::visit(RestrictQualifierAST* ast) {}

void Semantics::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  enumBase(ast->enumBase);
  for (auto it = ast->enumeratorList; it; it = it->next) enumerator(it->value);
}

void Semantics::visit(ClassSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  name(ast->name);
  baseClause(ast->baseClause);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(TypenameSpecifierAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void Semantics::visit(IdDeclaratorAST* ast) {
  name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(NestedDeclaratorAST* ast) { declarator(ast->declarator); }

void Semantics::visit(PointerOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
}

void Semantics::visit(ReferenceOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(PtrToMemberOperatorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
}

void Semantics::visit(FunctionDeclaratorAST* ast) {
  parametersAndQualifiers(ast->parametersAndQualifiers);
  trailingReturnType(ast->trailingReturnType);
}

void Semantics::visit(ArrayDeclaratorAST* ast) {
  expression(ast->expression);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

}  // namespace cxx
