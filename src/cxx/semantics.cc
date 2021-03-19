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
#include <cxx/names.h>
#include <cxx/semantics.h>
#include <cxx/translation_unit.h>
#include <cxx/type_environment.h>
#include <cxx/types.h>

namespace cxx {

Semantics::Semantics(TranslationUnit* unit)
    : unit_(unit), control_(unit_->control()), types_(control_->types()) {}

Semantics::~Semantics() {}

void Semantics::unit(UnitAST* ast) { accept(ast); }

Semantics::Specifiers Semantics::specifiers(List<SpecifierAST*>* ast) {
  Specifiers specifiers;
  std::swap(specifiers_, specifiers);
  for (auto it = ast; it; it = it->next) accept(it->value);
  std::swap(specifiers_, specifiers);
  return specifiers;
}

Semantics::Specifiers Semantics::specifiers(SpecifierAST* ast) {
  Specifiers specifiers;
  std::swap(specifiers_, specifiers);
  accept(ast);
  std::swap(specifiers_, specifiers);
  return specifiers;
}

Semantics::Declarator Semantics::declarator(DeclaratorAST* ast,
                                            const Specifiers& specifiers) {
  Declarator declarator{specifiers};
  std::swap(declarator_, declarator);
  accept(ast);
  std::swap(declarator_, declarator);
  return declarator;
}

const Name* Semantics::name(NameAST* ast) {
  if (!ast) return nullptr;
  if (ast->name) return ast->name;
  const Name* name = nullptr;
  std::swap(name_, name);
  accept(ast);
  std::swap(name_, name);
  ast->name = name;
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

Semantics::Expression Semantics::expression(ExpressionAST* ast) {
  Expression expression;
  if (ast) {
    if (ast->type) {
      expression.type = ast->type;
      return expression;
    }
    std::swap(expression_, expression);
    accept(ast);
    std::swap(expression_, expression);
  }
  return expression;
}

void Semantics::ptrOperator(PtrOperatorAST* ast) { accept(ast); }

void Semantics::coreDeclarator(CoreDeclaratorAST* ast) { accept(ast); }

void Semantics::declaratorModifier(DeclaratorModifierAST* ast) { accept(ast); }

void Semantics::declaratorModifiers(List<DeclaratorModifierAST*>* ast) {
  if (!ast) return;
  declaratorModifiers(ast->next);
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

Semantics::Declarator Semantics::initDeclarator(InitDeclaratorAST* ast,
                                                const Specifiers& specifiers) {
  Declarator declarator = this->declarator(ast->declarator, specifiers);
  initializer(ast->initializer);
  return declarator;
}

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
  auto specifiers = this->specifiers(ast->typeSpecifierList);
  auto declarator = this->declarator(ast->declarator, specifiers);
}

void Semantics::visit(NestedNameSpecifierAST* ast) {
  for (auto it = ast->nameList; it; it = it->next)
    auto name = this->name(it->value);
}

void Semantics::visit(UsingDeclaratorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->name);
}

void Semantics::visit(HandlerAST* ast) {
  exceptionDeclaration(ast->exceptionDeclaration);
  compoundStatement(ast->statement);
}

void Semantics::visit(TemplateArgumentAST* ast) {}

void Semantics::visit(EnumBaseAST* ast) {
  auto specifiers = this->specifiers(ast->typeSpecifierList);
}

void Semantics::visit(EnumeratorAST* ast) {
  auto name = this->name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  auto expression = this->expression(ast->expression);
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
  auto name = this->name(ast->name);
}

void Semantics::visit(BaseClauseAST* ast) {
  for (auto it = ast->baseSpecifierList; it; it = it->next)
    baseSpecifier(it->value);
}

void Semantics::visit(NewTypeIdAST* ast) {
  auto specifiers = this->specifiers(ast->typeSpecifierList);
}

void Semantics::visit(ParameterDeclarationClauseAST* ast) {
  for (auto it = ast->parameterDeclarationList; it; it = it->next)
    parameterDeclaration(it->value);
}

void Semantics::visit(ParametersAndQualifiersAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  auto specifiers = this->specifiers(ast->cvQualifierList);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(LambdaIntroducerAST* ast) {
  for (auto it = ast->captureList; it; it = it->next) lambdaCapture(it->value);
}

void Semantics::visit(LambdaDeclaratorAST* ast) {
  parameterDeclarationClause(ast->parameterDeclarationClause);
  auto specifiers = this->specifiers(ast->declSpecifierList);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  trailingReturnType(ast->trailingReturnType);
}

void Semantics::visit(TrailingReturnTypeAST* ast) { typeId(ast->typeId); }

void Semantics::visit(CtorInitializerAST* ast) {
  for (auto it = ast->memInitializerList; it; it = it->next)
    memInitializer(it->value);
}

void Semantics::visit(ParenMemInitializerAST* ast) {
  auto name = this->name(ast->name);
  for (auto it = ast->expressionList; it; it = it->next)
    auto expression = this->expression(it->value);
}

void Semantics::visit(BracedMemInitializerAST* ast) {
  auto name = this->name(ast->name);
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
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(BracedInitListAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next)
    auto expression = this->expression(it->value);
}

void Semantics::visit(ParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next)
    auto expression = this->expression(it->value);
}

void Semantics::visit(NewParenInitializerAST* ast) {
  for (auto it = ast->expressionList; it; it = it->next)
    auto expression = this->expression(it->value);
}

void Semantics::visit(NewBracedInitializerAST* ast) {
  bracedInitList(ast->bracedInit);
}

void Semantics::visit(EllipsisExceptionDeclarationAST* ast) {}

void Semantics::visit(TypeExceptionDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  auto specifiers = this->specifiers(ast->typeSpecifierList);
  auto declarator = this->declarator(ast->declarator, specifiers);
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

void Semantics::visit(IdExpressionAST* ast) {
  auto name = this->name(ast->name);
}

void Semantics::visit(NestedExpressionAST* ast) { accept(ast->expression); }

void Semantics::visit(RightFoldExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(LeftFoldExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(FoldExpressionAST* ast) {
  auto leftExpression = this->expression(ast->leftExpression);
  auto rightExpression = this->expression(ast->rightExpression);
}

void Semantics::visit(LambdaExpressionAST* ast) {
  lambdaIntroducer(ast->lambdaIntroducer);
  for (auto it = ast->templateParameterList; it; it = it->next)
    declaration(it->value);
  lambdaDeclarator(ast->lambdaDeclarator);
  compoundStatement(ast->statement);
}

void Semantics::visit(SizeofExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(SizeofTypeExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(SizeofPackExpressionAST* ast) {}

void Semantics::visit(TypeidExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(TypeidOfTypeExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(AlignofExpressionAST* ast) { typeId(ast->typeId); }

void Semantics::visit(UnaryExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(BinaryExpressionAST* ast) {
  auto leftExpression = this->expression(ast->leftExpression);
  auto rightExpression = this->expression(ast->rightExpression);
}

void Semantics::visit(AssignmentExpressionAST* ast) {
  auto leftExpression = this->expression(ast->leftExpression);
  auto rightExpression = this->expression(ast->rightExpression);
}

void Semantics::visit(BracedTypeConstructionAST* ast) {
  auto specifiers = this->specifiers(ast->typeSpecifier);
  bracedInitList(ast->bracedInitList);
}

void Semantics::visit(TypeConstructionAST* ast) {
  auto specifiers = this->specifiers(ast->typeSpecifier);
  for (auto it = ast->expressionList; it; it = it->next)
    auto expression = this->expression(it->value);
}

void Semantics::visit(CallExpressionAST* ast) {
  auto expression = this->expression(ast->baseExpression);
  for (auto it = ast->expressionList; it; it = it->next)
    auto expression = this->expression(it->value);
}

void Semantics::visit(SubscriptExpressionAST* ast) {
  auto baseExpression = this->expression(ast->baseExpression);
  auto indexExpression = this->expression(ast->indexExpression);
}

void Semantics::visit(MemberExpressionAST* ast) {
  auto expression = this->expression(ast->baseExpression);
  auto name = this->name(ast->name);
}

void Semantics::visit(ConditionalExpressionAST* ast) {
  auto condition = this->expression(ast->condition);
  auto iftrueExpression = this->expression(ast->iftrueExpression);
  auto iffalseExpression = this->expression(ast->iffalseExpression);
}

void Semantics::visit(CastExpressionAST* ast) {
  typeId(ast->typeId);
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(CppCastExpressionAST* ast) {
  typeId(ast->typeId);
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(NewExpressionAST* ast) {
  newTypeId(ast->typeId);
  newInitializer(ast->newInitalizer);
}

void Semantics::visit(DeleteExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(ThrowExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(NoexceptExpressionAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(LabeledStatementAST* ast) { statement(ast->statement); }

void Semantics::visit(CaseStatementAST* ast) {
  auto expression = this->expression(ast->expression);
  statement(ast->statement);
}

void Semantics::visit(DefaultStatementAST* ast) { statement(ast->statement); }

void Semantics::visit(ExpressionStatementAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(CompoundStatementAST* ast) {
  for (auto it = ast->statementList; it; it = it->next) statement(it->value);
}

void Semantics::visit(IfStatementAST* ast) {
  statement(ast->initializer);
  auto expression = this->expression(ast->condition);
  statement(ast->statement);
  statement(ast->elseStatement);
}

void Semantics::visit(SwitchStatementAST* ast) {
  statement(ast->initializer);
  auto expression = this->expression(ast->condition);
  statement(ast->statement);
}

void Semantics::visit(WhileStatementAST* ast) {
  auto expression = this->expression(ast->condition);
  statement(ast->statement);
}

void Semantics::visit(DoStatementAST* ast) {
  statement(ast->statement);
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(ForRangeStatementAST* ast) {
  statement(ast->initializer);
  declaration(ast->rangeDeclaration);
  auto expression = this->expression(ast->rangeInitializer);
  statement(ast->statement);
}

void Semantics::visit(ForStatementAST* ast) {
  statement(ast->initializer);
  auto condition = this->expression(ast->condition);
  auto expression = this->expression(ast->expression);
  statement(ast->statement);
}

void Semantics::visit(BreakStatementAST* ast) {}

void Semantics::visit(ContinueStatementAST* ast) {}

void Semantics::visit(ReturnStatementAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(GotoStatementAST* ast) {}

void Semantics::visit(CoroutineReturnStatementAST* ast) {
  auto expression = this->expression(ast->expression);
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
  auto specifiers = this->specifiers(ast->declSpecifierList);
  auto declarator = this->declarator(ast->declarator, specifiers);
  functionBody(ast->functionBody);
}

void Semantics::visit(ConceptDefinitionAST* ast) {
  auto name = this->name(ast->name);
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(ForRangeDeclarationAST* ast) {}

void Semantics::visit(AliasDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  typeId(ast->typeId);
}

void Semantics::visit(SimpleDeclarationAST* ast) {
  for (auto it = ast->attributes; it; it = it->next) attribute(it->value);
  auto specifiers = this->specifiers(ast->declSpecifierList);
  for (auto it = ast->initDeclaratorList; it; it = it->next) {
    auto declarator = this->initDeclarator(it->value, specifiers);
  }
}

void Semantics::visit(StaticAssertDeclarationAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(EmptyDeclarationAST* ast) {}

void Semantics::visit(AttributeDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

void Semantics::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->name);
  enumBase(ast->enumBase);
}

void Semantics::visit(UsingEnumDeclarationAST* ast) {}

void Semantics::visit(NamespaceDefinitionAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->name);
  for (auto it = ast->extraAttributeList; it; it = it->next)
    attribute(it->value);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(NamespaceAliasDefinitionAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->name);
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
  auto name = this->name(ast->name);
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
  auto specifiers = this->specifiers(ast->typeSpecifierList);
  auto declarator = this->declarator(ast->declarator, specifiers);
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(LinkageSpecificationAST* ast) {
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(SimpleNameAST* ast) {
  auto id = unit_->identifier(ast->identifierLoc);
  name_ = id;
}

void Semantics::visit(DestructorNameAST* ast) {
  auto name = this->name(ast->id);
}

void Semantics::visit(DecltypeNameAST* ast) {
  auto specifiers = this->specifiers(ast->decltypeSpecifier);
}

void Semantics::visit(OperatorNameAST* ast) {}

void Semantics::visit(TemplateNameAST* ast) {
  auto name = this->name(ast->id);
  for (auto it = ast->templateArgumentList; it; it = it->next)
    templateArgument(it->value);
}

void Semantics::visit(QualifiedNameAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->id);
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
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(AutoTypeSpecifierAST* ast) {}

void Semantics::visit(VoidTypeSpecifierAST* ast) {
  specifiers_.type.setType(types_->voidType());
}

void Semantics::visit(VaListTypeSpecifierAST* ast) {}

void Semantics::visit(IntegralTypeSpecifierAST* ast) {
  switch (unit_->tokenKind(ast->specifierLoc)) {
    case TokenKind::T_CHAR8_T:
      specifiers_.type.setType(types_->characterType(CharacterKind::kChar8T));
      break;

    case TokenKind::T_CHAR16_T:
      specifiers_.type.setType(types_->characterType(CharacterKind::kChar16T));
      break;

    case TokenKind::T_CHAR32_T:
      specifiers_.type.setType(types_->characterType(CharacterKind::kChar32T));
      break;

    case TokenKind::T_WCHAR_T:
      specifiers_.type.setType(types_->characterType(CharacterKind::kWCharT));
      break;

    case TokenKind::T_BOOL:
      specifiers_.type.setType(types_->booleanType());
      break;

    case TokenKind::T_CHAR:
      specifiers_.type.setType(
          types_->integerType(IntegerKind::kChar, specifiers_.isUnsigned));
      break;

    case TokenKind::T_SHORT:
      specifiers_.type.setType(
          types_->integerType(IntegerKind::kShort, specifiers_.isUnsigned));
      break;

    case TokenKind::T_INT:
      specifiers_.type.setType(
          types_->integerType(IntegerKind::kInt, specifiers_.isUnsigned));
      break;

    case TokenKind::T___INT64:
      specifiers_.type.setType(
          types_->integerType(IntegerKind::kInt64, specifiers_.isUnsigned));
      break;

    case TokenKind::T___INT128:
      specifiers_.type.setType(
          types_->integerType(IntegerKind::kInt128, specifiers_.isUnsigned));
      break;

    case TokenKind::T_LONG: {
      auto ty = dynamic_cast<const IntegerType*>(specifiers_.type.type());

      if (ty && ty->kind() == IntegerKind::kLong) {
        specifiers_.type.setType(types_->integerType(IntegerKind::kLongLong,
                                                     specifiers_.isUnsigned));
      } else {
        specifiers_.type.setType(
            types_->integerType(IntegerKind::kLong, specifiers_.isUnsigned));
      }

      break;
    }

    case TokenKind::T_SIGNED:
      specifiers_.isUnsigned = false;

      if (!specifiers_.type)
        specifiers_.type.setType(
            types_->integerType(IntegerKind::kInt, specifiers_.isUnsigned));
      break;

    case TokenKind::T_UNSIGNED:
      specifiers_.isUnsigned = true;

      if (!specifiers_.type)
        specifiers_.type.setType(
            types_->integerType(IntegerKind::kInt, specifiers_.isUnsigned));
      break;

    default:
      throw std::runtime_error(fmt::format(
          "invalid integral type: '{}'", unit_->tokenText(ast->specifierLoc)));
  }  // switch
}

void Semantics::visit(FloatingPointTypeSpecifierAST* ast) {
  switch (unit_->tokenKind(ast->specifierLoc)) {
    case TokenKind::T_FLOAT:
      specifiers_.type.setType(
          types_->floatingPointType(FloatingPointKind::kFloat));
      break;

    case TokenKind::T_DOUBLE: {
      auto ty = dynamic_cast<const IntegerType*>(specifiers_.type.type());

      if (ty && ty->kind() == IntegerKind::kLong) {
        specifiers_.type.setType(
            types_->floatingPointType(FloatingPointKind::kLongDouble));
      } else {
        specifiers_.type.setType(
            types_->floatingPointType(FloatingPointKind::kDouble));
      }
      break;
    }

    case TokenKind::T___FLOAT128:
      specifiers_.type.setType(
          types_->floatingPointType(FloatingPointKind::kFloat));
      break;

    default:
      throw std::runtime_error(
          fmt::format("invalid floating point type: '{}'",
                      unit_->tokenText(ast->specifierLoc)));
  }  // switch
}

void Semantics::visit(ComplexTypeSpecifierAST* ast) {}

void Semantics::visit(NamedTypeSpecifierAST* ast) {
  auto name = this->name(ast->name);
}

void Semantics::visit(AtomicTypeSpecifierAST* ast) { typeId(ast->typeId); }

void Semantics::visit(UnderlyingTypeSpecifierAST* ast) {}

void Semantics::visit(ElaboratedTypeSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->name);
}

void Semantics::visit(DecltypeAutoSpecifierAST* ast) {}

void Semantics::visit(DecltypeSpecifierAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(TypeofSpecifierAST* ast) {
  auto expression = this->expression(ast->expression);
}

void Semantics::visit(PlaceholderTypeSpecifierAST* ast) {}

void Semantics::visit(ConstQualifierAST* ast) {}

void Semantics::visit(VolatileQualifierAST* ast) {}

void Semantics::visit(RestrictQualifierAST* ast) {}

void Semantics::visit(EnumSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->name);
  enumBase(ast->enumBase);
  for (auto it = ast->enumeratorList; it; it = it->next) enumerator(it->value);
}

void Semantics::visit(ClassSpecifierAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  auto name = this->name(ast->name);
  baseClause(ast->baseClause);
  for (auto it = ast->declarationList; it; it = it->next)
    declaration(it->value);
}

void Semantics::visit(TypenameSpecifierAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  auto name = this->name(ast->name);
}

void Semantics::visit(IdDeclaratorAST* ast) {
  auto name = this->name(ast->name);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  declarator_.name = name;
}

void Semantics::visit(NestedDeclaratorAST* ast) { accept(ast->declarator); }

void Semantics::visit(PointerOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);

  auto qualifiers = this->specifiers(ast->cvQualifierList).type.qualifiers();

  FullySpecifiedType ptrTy(types_->pointerType(declarator_.type, qualifiers));

  declarator_.type = ptrTy;
}

void Semantics::visit(ReferenceOperatorAST* ast) {
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);

  FullySpecifiedType ptrTy(types_->referenceType(declarator_.type));

  declarator_.type = ptrTy;
}

void Semantics::visit(PtrToMemberOperatorAST* ast) {
  nestedNameSpecifier(ast->nestedNameSpecifier);
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
  auto specifiers = this->specifiers(ast->cvQualifierList);
}

void Semantics::visit(FunctionDeclaratorAST* ast) {
  std::vector<FullySpecifiedType> argumentTypes;
  bool isVariadic = false;

  if (ast->parametersAndQualifiers &&
      ast->parametersAndQualifiers->parameterDeclarationClause) {
    auto params = ast->parametersAndQualifiers->parameterDeclarationClause;

    for (auto it = params->parameterDeclarationList; it; it = it->next) {
      auto param = it->value;
      auto specifiers = this->specifiers(param->typeSpecifierList);
      auto declarator = this->declarator(param->declarator, specifiers);
      auto expression = this->expression(param->expression);
      argumentTypes.push_back(declarator.type);
    }

    isVariadic = bool(params->ellipsisLoc);
  }

  trailingReturnType(ast->trailingReturnType);

  FullySpecifiedType returnTy(declarator_.type);

  FullySpecifiedType funTy(
      types_->functionType(returnTy, std::move(argumentTypes), isVariadic));

  declarator_.type = funTy;
}

void Semantics::visit(ArrayDeclaratorAST* ast) {
  if (!ast->expression) {
    FullySpecifiedType arrayType(types_->unboundArrayType(declarator_.type));
    declarator_.type = arrayType;
  } else {
    auto expression = this->expression(ast->expression);
  }
  for (auto it = ast->attributeList; it; it = it->next) attribute(it->value);
}

}  // namespace cxx
