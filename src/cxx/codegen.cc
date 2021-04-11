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
#include <cxx/codegen.h>
#include <cxx/ir.h>
#include <cxx/ir_factory.h>
#include <cxx/translation_unit.h>

namespace cxx {

Codegen::Codegen() {}

Codegen::~Codegen() {}

std::unique_ptr<ir::Module> Codegen::operator()(TranslationUnit* unit) {
  auto module = std::make_unique<ir::Module>();
  std::swap(module_, module);
  std::swap(unit_, unit);
  accept(unit_->ast());
  std::swap(unit_, unit);
  std::swap(module_, module);
  return module;
}

ir::IRFactory* Codegen::irFactory() { return module_->irFactory(); }

void Codegen::place(ir::Block* block) {
  if (block_ && !block_->hasTerminator()) emit(irFactory()->createJump(block));
  function_->addBlock(block);
  block_ = block;
}

void Codegen::emit(ir::Stmt* stmt) { block_->code().push_back(stmt); }

void Codegen::specifier(SpecifierAST* ast) { accept(ast); }

void Codegen::declarator(DeclaratorAST* ast) { accept(ast); }

void Codegen::name(NameAST* ast) { accept(ast); }

void Codegen::nestedNameSpecifier(NestedNameSpecifierAST* ast) { accept(ast); }

void Codegen::exceptionDeclaration(ExceptionDeclarationAST* ast) {
  accept(ast);
}

void Codegen::compoundStatement(CompoundStatementAST* ast) { accept(ast); }

void Codegen::attribute(AttributeAST* ast) { accept(ast); }

void Codegen::expression(ExpressionAST* ast) { accept(ast); }

void Codegen::ptrOperator(PtrOperatorAST* ast) { accept(ast); }

void Codegen::coreDeclarator(CoreDeclaratorAST* ast) { accept(ast); }

void Codegen::declaratorModifier(DeclaratorModifierAST* ast) { accept(ast); }

void Codegen::initializer(InitializerAST* ast) { accept(ast); }

void Codegen::baseSpecifier(BaseSpecifierAST* ast) { accept(ast); }

void Codegen::parameterDeclaration(ParameterDeclarationAST* ast) {
  accept(ast);
}

void Codegen::parameterDeclarationClause(ParameterDeclarationClauseAST* ast) {
  accept(ast);
}

void Codegen::lambdaCapture(LambdaCaptureAST* ast) { accept(ast); }

void Codegen::trailingReturnType(TrailingReturnTypeAST* ast) { accept(ast); }

void Codegen::typeId(TypeIdAST* ast) { accept(ast); }

void Codegen::memInitializer(MemInitializerAST* ast) { accept(ast); }

void Codegen::bracedInitList(BracedInitListAST* ast) { accept(ast); }

void Codegen::ctorInitializer(CtorInitializerAST* ast) { accept(ast); }

void Codegen::handler(HandlerAST* ast) { accept(ast); }

void Codegen::declaration(DeclarationAST* ast) { accept(ast); }

void Codegen::lambdaIntroducer(LambdaIntroducerAST* ast) { accept(ast); }

void Codegen::lambdaDeclarator(LambdaDeclaratorAST* ast) { accept(ast); }

void Codegen::newTypeId(NewTypeIdAST* ast) { accept(ast); }

void Codegen::newInitializer(NewInitializerAST* ast) { accept(ast); }

void Codegen::statement(StatementAST* ast) { accept(ast); }

void Codegen::functionBody(FunctionBodyAST* ast) { accept(ast); }

void Codegen::initDeclarator(InitDeclaratorAST* ast) { accept(ast); }

void Codegen::enumBase(EnumBaseAST* ast) { accept(ast); }

void Codegen::usingDeclarator(UsingDeclaratorAST* ast) { accept(ast); }

void Codegen::templateArgument(TemplateArgumentAST* ast) { accept(ast); }

void Codegen::enumerator(EnumeratorAST* ast) { accept(ast); }

void Codegen::baseClause(BaseClauseAST* ast) { accept(ast); }

void Codegen::parametersAndQualifiers(ParametersAndQualifiersAST* ast) {
  accept(ast);
}

void Codegen::accept(AST* ast) {
  if (ast) ast->accept(this);
}

void Codegen::visit(TypeIdAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
}

void Codegen::visit(NestedNameSpecifierAST* ast) {
  for (auto it = ast->nameList; it; it = it->next) name(it->value);
}

void Codegen::visit(UsingDeclaratorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void Codegen::visit(HandlerAST* ast) {
  exceptionDeclaration(ast->exceptionDeclaration);
  compoundStatement(ast->statement);
}

void Codegen::visit(TypeTemplateArgumentAST* ast) { typeId(ast->typeId); }

void Codegen::visit(ExpressionTemplateArgumentAST* ast) {
  expression(ast->expression);
}

void Codegen::visit(EnumBaseAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
}

void Codegen::visit(EnumeratorAST* ast) {
  name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  expression(ast->expression);
}

void Codegen::visit(DeclaratorAST* ast) {
  for (auto it = ast->ptrOpList; it; it = it->next) ptrOperator(it->value);
  coreDeclarator(ast->coreDeclarator);
  for (auto it = ast->modifiers; it; it = it->next)
    declaratorModifier(it->value);
}

void Codegen::visit(InitDeclaratorAST* ast) {
  declarator(ast->declarator);
  initializer(ast->initializer);
}

void Codegen::visit(BaseSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  name(ast->name);
}

void Codegen::visit(BaseClauseAST* ast) {
  for (auto it = ast->baseSpecifierList; it; it = it->next)
    baseSpecifier(it->value);
}

void Codegen::visit(NewTypeIdAST* ast) {
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
}

void Codegen::visit(ParameterDeclarationClauseAST* ast) {
  for (auto it = ast->parameterDeclarationList; it; it = it->next)
    parameterDeclaration(it->value);
}

void Codegen::visit(ParametersAndQualifiersAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Codegen::visit(LambdaIntroducerAST* ast) {
  for (auto it = ast->captureList; it; it = it->next) lambdaCapture(it->value);
}

void Codegen::visit(LambdaDeclaratorAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    specifier(it->value);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  trailingReturnType(ast->trailingReturnType);
}

void Codegen::visit(TrailingReturnTypeAST* ast) { typeId(ast->typeId); }

void Codegen::visit(CtorInitializerAST* ast) {
  for (auto it = ast->memInitializerList; it; it = it->next)
    memInitializer(it->value);
}

void Codegen::visit(ParenMemInitializerAST* ast) {
  name(ast->name);
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Codegen::visit(BracedMemInitializerAST* ast) {
  name(ast->name);
  bracedInitList(ast->bracedInitList);
}

void Codegen::visit(ThisLambdaCaptureAST* ast) {}

void Codegen::visit(DerefThisLambdaCaptureAST* ast) {}

void Codegen::visit(SimpleLambdaCaptureAST* ast) {}

void Codegen::visit(RefLambdaCaptureAST* ast) {}

void Codegen::visit(RefInitLambdaCaptureAST* ast) {
  initializer(ast->initializer);
}

void Codegen::visit(InitLambdaCaptureAST* ast) {
  initializer(ast->initializer);
}

void Codegen::visit(EqualInitializerAST* ast) { expression(ast->expression); }

void Codegen::visit(BracedInitListAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Codegen::visit(ParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Codegen::visit(NewParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Codegen::visit(NewBracedInitializerAST* ast) {
  bracedInitList(ast->bracedInit);
}

void Codegen::visit(EllipsisExceptionDeclarationAST* ast) {}

void Codegen::visit(TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
}

void Codegen::visit(DefaultFunctionBodyAST* ast) {}

void Codegen::visit(CompoundStatementFunctionBodyAST* ast) {
  ctorInitializer(ast->ctorInitializer);
  compoundStatement(ast->statement);
}

void Codegen::visit(TryStatementFunctionBodyAST* ast) {
  ctorInitializer(ast->ctorInitializer);
  compoundStatement(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) handler(it->value);
}

void Codegen::visit(DeleteFunctionBodyAST* ast) {}

void Codegen::visit(TranslationUnitAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Codegen::visit(ModuleUnitAST* ast) {}

void Codegen::visit(ThisExpressionAST* ast) {}

void Codegen::visit(CharLiteralExpressionAST* ast) {}

void Codegen::visit(BoolLiteralExpressionAST* ast) {}

void Codegen::visit(IntLiteralExpressionAST* ast) {}

void Codegen::visit(FloatLiteralExpressionAST* ast) {}

void Codegen::visit(NullptrLiteralExpressionAST* ast) {}

void Codegen::visit(StringLiteralExpressionAST* ast) {}

void Codegen::visit(UserDefinedStringLiteralExpressionAST* ast) {}

void Codegen::visit(IdExpressionAST* ast) { name(ast->name); }

void Codegen::visit(NestedExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(RightFoldExpressionAST* ast) {
  expression(ast->expression);
}

void Codegen::visit(LeftFoldExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(FoldExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void Codegen::visit(LambdaExpressionAST* ast) {
  lambdaIntroducer(ast->lambdaIntroducer);
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  lambdaDeclarator(ast->lambdaDeclarator);
  compoundStatement(ast->statement);
}

void Codegen::visit(SizeofExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(SizeofTypeExpressionAST* ast) { typeId(ast->typeId); }

void Codegen::visit(SizeofPackExpressionAST* ast) {}

void Codegen::visit(TypeidExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(TypeidOfTypeExpressionAST* ast) { typeId(ast->typeId); }

void Codegen::visit(AlignofExpressionAST* ast) { typeId(ast->typeId); }

void Codegen::visit(UnaryExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(BinaryExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void Codegen::visit(AssignmentExpressionAST* ast) {
  expression(ast->leftExpression);
  expression(ast->rightExpression);
}

void Codegen::visit(BracedTypeConstructionAST* ast) {
  specifier(ast->typeSpecifier);
  bracedInitList(ast->bracedInitList);
}

void Codegen::visit(TypeConstructionAST* ast) {
  specifier(ast->typeSpecifier);
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Codegen::visit(CallExpressionAST* ast) {
  expression(ast->baseExpression);
  for (auto it = ast->expressionList; it; it = it->next) expression(it->value);
}

void Codegen::visit(SubscriptExpressionAST* ast) {
  expression(ast->baseExpression);
  expression(ast->indexExpression);
}

void Codegen::visit(MemberExpressionAST* ast) {
  expression(ast->baseExpression);
  name(ast->name);
}

void Codegen::visit(ConditionalExpressionAST* ast) {
  expression(ast->condition);
  expression(ast->iftrueExpression);
  expression(ast->iffalseExpression);
}

void Codegen::visit(CastExpressionAST* ast) {
  typeId(ast->typeId);
  expression(ast->expression);
}

void Codegen::visit(CppCastExpressionAST* ast) {
  typeId(ast->typeId);
  expression(ast->expression);
}

void Codegen::visit(NewExpressionAST* ast) {
  newTypeId(ast->typeId);
  newInitializer(ast->newInitalizer);
}

void Codegen::visit(DeleteExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(ThrowExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(NoexceptExpressionAST* ast) { expression(ast->expression); }

void Codegen::visit(LabeledStatementAST* ast) { statement(ast->statement); }

void Codegen::visit(CaseStatementAST* ast) {
  expression(ast->expression);
  statement(ast->statement);
}

void Codegen::visit(DefaultStatementAST* ast) { statement(ast->statement); }

void Codegen::visit(ExpressionStatementAST* ast) {
  expression(ast->expression);
}

void Codegen::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next) statement(it->value);
}

void Codegen::visit(IfStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  statement(ast->statement);
  statement(ast->elseStatement);
}

void Codegen::visit(SwitchStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  statement(ast->statement);
}

void Codegen::visit(WhileStatementAST* ast) {
  expression(ast->condition);
  statement(ast->statement);
}

void Codegen::visit(DoStatementAST* ast) {
  statement(ast->statement);
  expression(ast->expression);
}

void Codegen::visit(ForRangeStatementAST* ast) {
  statement(ast->initializer);
  declaration(ast->rangeDeclaration);
  expression(ast->rangeInitializer);
  statement(ast->statement);
}

void Codegen::visit(ForStatementAST* ast) {
  statement(ast->initializer);
  expression(ast->condition);
  expression(ast->expression);
  statement(ast->statement);
}

void Codegen::visit(BreakStatementAST* ast) {}

void Codegen::visit(ContinueStatementAST* ast) {}

void Codegen::visit(ReturnStatementAST* ast) { expression(ast->expression); }

void Codegen::visit(GotoStatementAST* ast) {}

void Codegen::visit(CoroutineReturnStatementAST* ast) {
  expression(ast->expression);
}

void Codegen::visit(DeclarationStatementAST* ast) {
  declaration(ast->declaration);
}

void Codegen::visit(TryBlockStatementAST* ast) {
  compoundStatement(ast->statement);
  for (auto it = ast->handlerList; it; it = it->next) handler(it->value);
}

void Codegen::visit(AccessDeclarationAST* ast) {}

void Codegen::visit(FunctionDefinitionAST* ast) {
  ir::Function* function = irFactory()->createFunction(ast->symbol);

  module_->addFunction(function);

  ir::Block* entryBlock = irFactory()->createBlock(function);
  ir::Block* exitBlock = irFactory()->createBlock(function);
  ir::Block* block = nullptr;

  std::swap(function_, function);
  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);
  std::swap(block_, block);

  place(entryBlock_);

  functionBody(ast->functionBody);

  place(exitBlock_);
  emit(irFactory()->createRetVoid());

  std::swap(function_, function);
  std::swap(entryBlock_, entryBlock);
  std::swap(exitBlock_, exitBlock);
  std::swap(block_, block);
}

void Codegen::visit(ConceptDefinitionAST* ast) {
  name(ast->name);
  expression(ast->expression);
}

void Codegen::visit(ForRangeDeclarationAST* ast) {}

void Codegen::visit(AliasDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  typeId(ast->typeId);
}

void Codegen::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->declSpecifierList; it; it = it->next)
    specifier(it->value);
  for (auto it = ast->initDeclaratorList; it; it = it->next)
    initDeclarator(it->value);
}

void Codegen::visit(StaticAssertDeclarationAST* ast) {
  expression(ast->expression);
}

void Codegen::visit(EmptyDeclarationAST* ast) {}

void Codegen::visit(AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Codegen::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  enumBase(ast->enumBase);
}

void Codegen::visit(UsingEnumDeclarationAST* ast) {}

void Codegen::visit(NamespaceDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  for (auto it = ast->extraAttributeList; it; it = it->next)
    attribute(it->value);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Codegen::visit(NamespaceAliasDefinitionAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void Codegen::visit(UsingDirectiveAST* ast) {}

void Codegen::visit(UsingDeclarationAST* ast) {
  for (auto it = ast->usingDeclaratorList; it; it = it->next)
    usingDeclarator(it->value);
}

void Codegen::visit(AsmDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Codegen::visit(ExportDeclarationAST* ast) {}

void Codegen::visit(ModuleImportDeclarationAST* ast) {}

void Codegen::visit(TemplateDeclarationAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  declaration(ast->declaration);
}

void Codegen::visit(TypenameTypeParameterAST* ast) { typeId(ast->typeId); }

void Codegen::visit(TypenamePackTypeParameterAST* ast) {}

void Codegen::visit(TemplateTypeParameterAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  name(ast->name);
}

void Codegen::visit(TemplatePackTypeParameterAST* ast) {
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
}

void Codegen::visit(DeductionGuideAST* ast) {}

void Codegen::visit(ExplicitInstantiationAST* ast) {
  declaration(ast->declaration);
}

void Codegen::visit(ParameterDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->typeSpecifierList; it; it = it->next)
    specifier(it->value);
  declarator(ast->declarator);
  expression(ast->expression);
}

void Codegen::visit(LinkageSpecificationAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Codegen::visit(SimpleNameAST* ast) {}

void Codegen::visit(DestructorNameAST* ast) { name(ast->id); }

void Codegen::visit(DecltypeNameAST* ast) { specifier(ast->decltypeSpecifier); }

void Codegen::visit(OperatorNameAST* ast) {}

void Codegen::visit(ConversionNameAST* ast) { typeId(ast->typeId); }

void Codegen::visit(TemplateNameAST* ast) {
  name(ast->id);
  for (auto it = ast->templateArgumentList; it; it = it->next)
    templateArgument(it->value);
}

void Codegen::visit(QualifiedNameAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->id);
}

void Codegen::visit(TypedefSpecifierAST* ast) {}

void Codegen::visit(FriendSpecifierAST* ast) {}

void Codegen::visit(ConstevalSpecifierAST* ast) {}

void Codegen::visit(ConstinitSpecifierAST* ast) {}

void Codegen::visit(ConstexprSpecifierAST* ast) {}

void Codegen::visit(InlineSpecifierAST* ast) {}

void Codegen::visit(StaticSpecifierAST* ast) {}

void Codegen::visit(ExternSpecifierAST* ast) {}

void Codegen::visit(ThreadLocalSpecifierAST* ast) {}

void Codegen::visit(ThreadSpecifierAST* ast) {}

void Codegen::visit(MutableSpecifierAST* ast) {}

void Codegen::visit(VirtualSpecifierAST* ast) {}

void Codegen::visit(ExplicitSpecifierAST* ast) { expression(ast->expression); }

void Codegen::visit(AutoTypeSpecifierAST* ast) {}

void Codegen::visit(VoidTypeSpecifierAST* ast) {}

void Codegen::visit(VaListTypeSpecifierAST* ast) {}

void Codegen::visit(IntegralTypeSpecifierAST* ast) {}

void Codegen::visit(FloatingPointTypeSpecifierAST* ast) {}

void Codegen::visit(ComplexTypeSpecifierAST* ast) {}

void Codegen::visit(NamedTypeSpecifierAST* ast) { name(ast->name); }

void Codegen::visit(AtomicTypeSpecifierAST* ast) { typeId(ast->typeId); }

void Codegen::visit(UnderlyingTypeSpecifierAST* ast) {}

void Codegen::visit(ElaboratedTypeSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void Codegen::visit(DecltypeAutoSpecifierAST* ast) {}

void Codegen::visit(DecltypeSpecifierAST* ast) { expression(ast->expression); }

void Codegen::visit(TypeofSpecifierAST* ast) { expression(ast->expression); }

void Codegen::visit(PlaceholderTypeSpecifierAST* ast) {}

void Codegen::visit(ConstQualifierAST* ast) {}

void Codegen::visit(VolatileQualifierAST* ast) {}

void Codegen::visit(RestrictQualifierAST* ast) {}

void Codegen::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
  enumBase(ast->enumBase);
  for (auto it = ast->enumeratorList; it; it = it->next) enumerator(it->value);
}

void Codegen::visit(ClassSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  name(ast->name);
  baseClause(ast->baseClause);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Codegen::visit(TypenameSpecifierAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  name(ast->name);
}

void Codegen::visit(IdDeclaratorAST* ast) {
  name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Codegen::visit(NestedDeclaratorAST* ast) { declarator(ast->declarator); }

void Codegen::visit(PointerOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
}

void Codegen::visit(ReferenceOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Codegen::visit(PtrToMemberOperatorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  for (auto it = ast->cvQualifierList; it; it = it->next) specifier(it->value);
}

void Codegen::visit(FunctionDeclaratorAST* ast) {
  parametersAndQualifiers(ast->parametersAndQualifiers);
  trailingReturnType(ast->trailingReturnType);
}

void Codegen::visit(ArrayDeclaratorAST* ast) {
  expression(ast->expression);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

}  // namespace cxx
