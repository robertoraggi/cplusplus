// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

#include <cxx/ast_interpreter.h>

// cxx
#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/translation_unit.h>

namespace cxx {

ASTInterpreter::ASTInterpreter(TranslationUnit* unit) : unit_(unit) {}

ASTInterpreter::~ASTInterpreter() {}

auto ASTInterpreter::control() const -> Control* { return unit_->control(); }

auto ASTInterpreter::operator()(UnitAST* ast) -> UnitResult {
  if (ast) return visit(UnitVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(DeclarationAST* ast) -> DeclarationResult {
  if (ast) return visit(DeclarationVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(StatementAST* ast) -> StatementResult {
  if (ast) return visit(StatementVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(ExpressionAST* ast) -> ExpressionResult {
  if (ast) return visit(ExpressionVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(TemplateParameterAST* ast)
    -> TemplateParameterResult {
  if (ast) return visit(TemplateParameterVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(SpecifierAST* ast) -> SpecifierResult {
  if (ast) return visit(SpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(PtrOperatorAST* ast) -> PtrOperatorResult {
  if (ast) return visit(PtrOperatorVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(CoreDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  if (ast) return visit(CoreDeclaratorVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(DeclaratorChunkAST* ast)
    -> DeclaratorChunkResult {
  if (ast) return visit(DeclaratorChunkVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(UnqualifiedIdAST* ast) -> UnqualifiedIdResult {
  if (ast) return visit(UnqualifiedIdVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(NestedNameSpecifierAST* ast)
    -> NestedNameSpecifierResult {
  if (ast) return visit(NestedNameSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(FunctionBodyAST* ast) -> FunctionBodyResult {
  if (ast) return visit(FunctionBodyVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(TemplateArgumentAST* ast)
    -> TemplateArgumentResult {
  if (ast) return visit(TemplateArgumentVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(ExceptionSpecifierAST* ast)
    -> ExceptionSpecifierResult {
  if (ast) return visit(ExceptionSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(RequirementAST* ast) -> RequirementResult {
  if (ast) return visit(RequirementVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(NewInitializerAST* ast)
    -> NewInitializerResult {
  if (ast) return visit(NewInitializerVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(MemInitializerAST* ast)
    -> MemInitializerResult {
  if (ast) return visit(MemInitializerVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(LambdaCaptureAST* ast) -> LambdaCaptureResult {
  if (ast) return visit(LambdaCaptureVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(ExceptionDeclarationAST* ast)
    -> ExceptionDeclarationResult {
  if (ast) return visit(ExceptionDeclarationVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(AttributeSpecifierAST* ast)
    -> AttributeSpecifierResult {
  if (ast) return visit(AttributeSpecifierVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(AttributeTokenAST* ast)
    -> AttributeTokenResult {
  if (ast) return visit(AttributeTokenVisitor{*this}, ast);
  return {};
}

auto ASTInterpreter::operator()(SplicerAST* ast) -> SplicerResult {
  if (!ast) return {};

  auto expressionResult = operator()(ast->expression);

  return {};
}

auto ASTInterpreter::operator()(GlobalModuleFragmentAST* ast)
    -> GlobalModuleFragmentResult {
  if (!ast) return {};

  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  return {};
}

auto ASTInterpreter::operator()(PrivateModuleFragmentAST* ast)
    -> PrivateModuleFragmentResult {
  if (!ast) return {};

  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  return {};
}

auto ASTInterpreter::operator()(ModuleDeclarationAST* ast)
    -> ModuleDeclarationResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);
  auto modulePartitionResult = operator()(ast->modulePartition);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  return {};
}

auto ASTInterpreter::operator()(ModuleNameAST* ast) -> ModuleNameResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto ASTInterpreter::operator()(ModuleQualifierAST* ast)
    -> ModuleQualifierResult {
  if (!ast) return {};

  auto moduleQualifierResult = operator()(ast->moduleQualifier);

  return {};
}

auto ASTInterpreter::operator()(ModulePartitionAST* ast)
    -> ModulePartitionResult {
  if (!ast) return {};

  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

auto ASTInterpreter::operator()(ImportNameAST* ast) -> ImportNameResult {
  if (!ast) return {};

  auto modulePartitionResult = operator()(ast->modulePartition);
  auto moduleNameResult = operator()(ast->moduleName);

  return {};
}

auto ASTInterpreter::operator()(InitDeclaratorAST* ast)
    -> InitDeclaratorResult {
  if (!ast) return {};

  auto declaratorResult = operator()(ast->declarator);
  auto requiresClauseResult = operator()(ast->requiresClause);
  auto initializerResult = operator()(ast->initializer);

  return {};
}

auto ASTInterpreter::operator()(DeclaratorAST* ast) -> DeclaratorResult {
  if (!ast) return {};

  for (auto it = ast->ptrOpList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  auto coreDeclaratorResult = operator()(ast->coreDeclarator);

  for (auto it = ast->declaratorChunkList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  return {};
}

auto ASTInterpreter::operator()(UsingDeclaratorAST* ast)
    -> UsingDeclaratorResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = operator()(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::operator()(EnumeratorAST* ast) -> EnumeratorResult {
  if (!ast) return {};

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  auto expressionResult = operator()(ast->expression);

  return {};
}

auto ASTInterpreter::operator()(TypeIdAST* ast) -> TypeIdResult {
  if (!ast) return {};

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  auto declaratorResult = operator()(ast->declarator);

  return {};
}

auto ASTInterpreter::operator()(HandlerAST* ast) -> HandlerResult {
  if (!ast) return {};

  auto exceptionDeclarationResult = operator()(ast->exceptionDeclaration);
  auto statementResult = operator()(ast->statement);

  return {};
}

auto ASTInterpreter::operator()(BaseSpecifierAST* ast) -> BaseSpecifierResult {
  if (!ast) return {};

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = operator()(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::operator()(RequiresClauseAST* ast)
    -> RequiresClauseResult {
  if (!ast) return {};

  auto expressionResult = operator()(ast->expression);

  return {};
}

auto ASTInterpreter::operator()(ParameterDeclarationClauseAST* ast)
    -> ParameterDeclarationClauseResult {
  if (!ast) return {};

  for (auto it = ast->parameterDeclarationList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  return {};
}

auto ASTInterpreter::operator()(TrailingReturnTypeAST* ast)
    -> TrailingReturnTypeResult {
  if (!ast) return {};

  auto typeIdResult = operator()(ast->typeId);

  return {};
}

auto ASTInterpreter::operator()(LambdaSpecifierAST* ast)
    -> LambdaSpecifierResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::operator()(TypeConstraintAST* ast)
    -> TypeConstraintResult {
  if (!ast) return {};

  auto nestedNameSpecifierResult = operator()(ast->nestedNameSpecifier);

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  return {};
}

auto ASTInterpreter::operator()(AttributeArgumentClauseAST* ast)
    -> AttributeArgumentClauseResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::operator()(AttributeAST* ast) -> AttributeResult {
  if (!ast) return {};

  auto attributeTokenResult = operator()(ast->attributeToken);
  auto attributeArgumentClauseResult = operator()(ast->attributeArgumentClause);

  return {};
}

auto ASTInterpreter::operator()(AttributeUsingPrefixAST* ast)
    -> AttributeUsingPrefixResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::operator()(NewPlacementAST* ast) -> NewPlacementResult {
  if (!ast) return {};

  for (auto it = ast->expressionList; it; it = it->next) {
    auto value = operator()(it->value);
  }

  return {};
}

auto ASTInterpreter::operator()(NestedNamespaceSpecifierAST* ast)
    -> NestedNamespaceSpecifierResult {
  if (!ast) return {};

  return {};
}

auto ASTInterpreter::UnitVisitor::operator()(TranslationUnitAST* ast)
    -> UnitResult {
  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::UnitVisitor::operator()(ModuleUnitAST* ast) -> UnitResult {
  auto globalModuleFragmentResult = accept(ast->globalModuleFragment);
  auto moduleDeclarationResult = accept(ast->moduleDeclaration);

  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto privateModuleFragmentResult = accept(ast->privateModuleFragment);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(SimpleDeclarationAST* ast)
    -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->initDeclaratorList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto requiresClauseResult = accept(ast->requiresClause);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmDeclarationAST* ast)
    -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->asmQualifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->outputOperandList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->inputOperandList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->clobberList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->gotoLabelList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    NamespaceAliasDefinitionAST* ast) -> DeclarationResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(UsingDeclarationAST* ast)
    -> DeclarationResult {
  for (auto it = ast->usingDeclaratorList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    UsingEnumDeclarationAST* ast) -> DeclarationResult {
  auto enumTypeSpecifierResult = accept(ast->enumTypeSpecifier);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(UsingDirectiveAST* ast)
    -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    StaticAssertDeclarationAST* ast) -> DeclarationResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AliasDeclarationAST* ast)
    -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    OpaqueEnumDeclarationAST* ast) -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(FunctionDefinitionAST* ast)
    -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto declaratorResult = accept(ast->declarator);
  auto requiresClauseResult = accept(ast->requiresClause);
  auto functionBodyResult = accept(ast->functionBody);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(TemplateDeclarationAST* ast)
    -> DeclarationResult {
  for (auto it = ast->templateParameterList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto requiresClauseResult = accept(ast->requiresClause);
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ConceptDefinitionAST* ast)
    -> DeclarationResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(DeductionGuideAST* ast)
    -> DeclarationResult {
  auto explicitSpecifierResult = accept(ast->explicitSpecifier);
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);
  auto templateIdResult = accept(ast->templateId);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ExplicitInstantiationAST* ast) -> DeclarationResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ExportDeclarationAST* ast)
    -> DeclarationResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ExportCompoundDeclarationAST* ast) -> DeclarationResult {
  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    LinkageSpecificationAST* ast) -> DeclarationResult {
  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(NamespaceDefinitionAST* ast)
    -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->nestedNamespaceSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->extraAttributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(EmptyDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    AttributeDeclarationAST* ast) -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ModuleImportDeclarationAST* ast) -> DeclarationResult {
  auto importNameResult = accept(ast->importName);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    ParameterDeclarationAST* ast) -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto declaratorResult = accept(ast->declarator);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AccessDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(ForRangeDeclarationAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(
    StructuredBindingDeclarationAST* ast) -> DeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->bindingList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmOperandAST* ast)
    -> DeclarationResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmQualifierAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmClobberAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::DeclarationVisitor::operator()(AsmGotoLabelAST* ast)
    -> DeclarationResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(LabeledStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(CaseStatementAST* ast)
    -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DefaultStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ExpressionStatementAST* ast)
    -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(CompoundStatementAST* ast)
    -> StatementResult {
  for (auto it = ast->statementList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(IfStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto conditionResult = accept(ast->condition);
  auto statementResult = accept(ast->statement);
  auto elseStatementResult = accept(ast->elseStatement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ConstevalIfStatementAST* ast)
    -> StatementResult {
  auto statementResult = accept(ast->statement);
  auto elseStatementResult = accept(ast->elseStatement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(SwitchStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto conditionResult = accept(ast->condition);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(WhileStatementAST* ast)
    -> StatementResult {
  auto conditionResult = accept(ast->condition);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DoStatementAST* ast)
    -> StatementResult {
  auto statementResult = accept(ast->statement);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForRangeStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto rangeDeclarationResult = accept(ast->rangeDeclaration);
  auto rangeInitializerResult = accept(ast->rangeInitializer);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ForStatementAST* ast)
    -> StatementResult {
  auto initializerResult = accept(ast->initializer);
  auto conditionResult = accept(ast->condition);
  auto expressionResult = accept(ast->expression);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(BreakStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ContinueStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(ReturnStatementAST* ast)
    -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(
    CoroutineReturnStatementAST* ast) -> StatementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(GotoStatementAST* ast)
    -> StatementResult {
  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(DeclarationStatementAST* ast)
    -> StatementResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::StatementVisitor::operator()(TryBlockStatementAST* ast)
    -> StatementResult {
  auto statementResult = accept(ast->statement);

  for (auto it = ast->handlerList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    GeneratedLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    CharLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BoolLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(IntLiteralExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    FloatLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NullptrLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    StringLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    UserDefinedStringLiteralExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThisExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(NestedExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(IdExpressionAST* ast)
    -> ExpressionResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(LambdaExpressionAST* ast)
    -> ExpressionResult {
  for (auto it = ast->captureList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->templateParameterList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto templateRequiresClauseResult = accept(ast->templateRequiresClause);
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);

  for (auto it = ast->gnuAtributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->lambdaSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto exceptionSpecifierResult = accept(ast->exceptionSpecifier);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto trailingReturnTypeResult = accept(ast->trailingReturnType);
  auto requiresClauseResult = accept(ast->requiresClause);
  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(FoldExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = accept(ast->leftExpression);
  auto rightExpressionResult = accept(ast->rightExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(RightFoldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(LeftFoldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(RequiresExpressionAST* ast)
    -> ExpressionResult {
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);

  for (auto it = ast->requirementList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(VaArgExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SubscriptExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);
  auto indexExpressionResult = accept(ast->indexExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CallExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);

  for (auto it = ast->expressionList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeConstructionAST* ast)
    -> ExpressionResult {
  auto typeSpecifierResult = accept(ast->typeSpecifier);

  for (auto it = ast->expressionList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BracedTypeConstructionAST* ast) -> ExpressionResult {
  auto typeSpecifierResult = accept(ast->typeSpecifier);
  auto bracedInitListResult = accept(ast->bracedInitList);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    SpliceMemberExpressionAST* ast) -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);
  auto splicerResult = accept(ast->splicer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(MemberExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(PostIncrExpressionAST* ast)
    -> ExpressionResult {
  auto baseExpressionResult = accept(ast->baseExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CppCastExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    BuiltinBitCastExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeidExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    TypeidOfTypeExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SpliceExpressionAST* ast)
    -> ExpressionResult {
  auto splicerResult = accept(ast->splicer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    GlobalScopeReflectExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    NamespaceReflectExpressionAST* ast) -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    TypeIdReflectExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ReflectExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(UnaryExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AwaitExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofTypeExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(SizeofPackExpressionAST* ast)
    -> ExpressionResult {
  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    AlignofTypeExpressionAST* ast) -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AlignofExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(NoexceptExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(NewExpressionAST* ast)
    -> ExpressionResult {
  auto newPlacementResult = accept(ast->newPlacement);

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto declaratorResult = accept(ast->declarator);
  auto newInitalizerResult = accept(ast->newInitalizer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(DeleteExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(CastExpressionAST* ast)
    -> ExpressionResult {
  auto typeIdResult = accept(ast->typeId);
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ImplicitCastExpressionAST* ast) -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(BinaryExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = accept(ast->leftExpression);
  auto rightExpressionResult = accept(ast->rightExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    ConditionalExpressionAST* ast) -> ExpressionResult {
  auto conditionResult = accept(ast->condition);
  auto iftrueExpressionResult = accept(ast->iftrueExpression);
  auto iffalseExpressionResult = accept(ast->iffalseExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(YieldExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ThrowExpressionAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(AssignmentExpressionAST* ast)
    -> ExpressionResult {
  auto leftExpressionResult = accept(ast->leftExpression);
  auto rightExpressionResult = accept(ast->rightExpression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    PackExpansionExpressionAST* ast) -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(
    DesignatedInitializerClauseAST* ast) -> ExpressionResult {
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(TypeTraitExpressionAST* ast)
    -> ExpressionResult {
  for (auto it = ast->typeIdList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ConditionExpressionAST* ast)
    -> ExpressionResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->declSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto declaratorResult = accept(ast->declarator);
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(EqualInitializerAST* ast)
    -> ExpressionResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(BracedInitListAST* ast)
    -> ExpressionResult {
  for (auto it = ast->expressionList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::ExpressionVisitor::operator()(ParenInitializerAST* ast)
    -> ExpressionResult {
  for (auto it = ast->expressionList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    TemplateTypeParameterAST* ast) -> TemplateParameterResult {
  for (auto it = ast->templateParameterList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto requiresClauseResult = accept(ast->requiresClause);
  auto idExpressionResult = accept(ast->idExpression);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    NonTypeTemplateParameterAST* ast) -> TemplateParameterResult {
  auto declarationResult = accept(ast->declaration);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    TypenameTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::TemplateParameterVisitor::operator()(
    ConstraintTypeParameterAST* ast) -> TemplateParameterResult {
  auto typeConstraintResult = accept(ast->typeConstraint);
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    GeneratedTypeSpecifierAST* ast) -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(TypedefSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(FriendSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstevalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstinitSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstexprSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(InlineSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(StaticSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ExternSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ThreadLocalSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ThreadSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(MutableSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VirtualSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ExplicitSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(AutoTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VoidTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SizeTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SignTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VaListTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(IntegralTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    FloatingPointTypeSpecifierAST* ast) -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ComplexTypeSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(NamedTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(AtomicTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    UnderlyingTypeSpecifierAST* ast) -> SpecifierResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    ElaboratedTypeSpecifierAST* ast) -> SpecifierResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(DecltypeAutoSpecifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(DecltypeSpecifierAST* ast)
    -> SpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(
    PlaceholderTypeSpecifierAST* ast) -> SpecifierResult {
  auto typeConstraintResult = accept(ast->typeConstraint);
  auto specifierResult = accept(ast->specifier);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ConstQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(VolatileQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(RestrictQualifierAST* ast)
    -> SpecifierResult {
  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(EnumSpecifierAST* ast)
    -> SpecifierResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->enumeratorList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(ClassSpecifierAST* ast)
    -> SpecifierResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto it = ast->baseSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->declarationList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(TypenameSpecifierAST* ast)
    -> SpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::SpecifierVisitor::operator()(SplicerTypeSpecifierAST* ast)
    -> SpecifierResult {
  auto splicerResult = accept(ast->splicer);

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(PointerOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(ReferenceOperatorAST* ast)
    -> PtrOperatorResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::PtrOperatorVisitor::operator()(PtrToMemberOperatorAST* ast)
    -> PtrOperatorResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(
    BitfieldDeclaratorAST* ast) -> CoreDeclaratorResult {
  auto unqualifiedIdResult = accept(ast->unqualifiedId);
  auto sizeExpressionResult = accept(ast->sizeExpression);

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(ParameterPackAST* ast)
    -> CoreDeclaratorResult {
  auto coreDeclaratorResult = accept(ast->coreDeclarator);

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(IdDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::CoreDeclaratorVisitor::operator()(NestedDeclaratorAST* ast)
    -> CoreDeclaratorResult {
  auto declaratorResult = accept(ast->declarator);

  return {};
}

auto ASTInterpreter::DeclaratorChunkVisitor::operator()(
    FunctionDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto parameterDeclarationClauseResult =
      accept(ast->parameterDeclarationClause);

  for (auto it = ast->cvQualifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto exceptionSpecifierResult = accept(ast->exceptionSpecifier);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto trailingReturnTypeResult = accept(ast->trailingReturnType);

  return {};
}

auto ASTInterpreter::DeclaratorChunkVisitor::operator()(
    ArrayDeclaratorChunkAST* ast) -> DeclaratorChunkResult {
  auto expressionResult = accept(ast->expression);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(NameIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(DestructorIdAST* ast)
    -> UnqualifiedIdResult {
  auto idResult = accept(ast->id);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(DecltypeIdAST* ast)
    -> UnqualifiedIdResult {
  auto decltypeSpecifierResult = accept(ast->decltypeSpecifier);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionIdAST* ast) -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(LiteralOperatorIdAST* ast)
    -> UnqualifiedIdResult {
  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    ConversionFunctionIdAST* ast) -> UnqualifiedIdResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(SimpleTemplateIdAST* ast)
    -> UnqualifiedIdResult {
  for (auto it = ast->templateArgumentList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    LiteralOperatorTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto literalOperatorIdResult = accept(ast->literalOperatorId);

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::UnqualifiedIdVisitor::operator()(
    OperatorFunctionTemplateIdAST* ast) -> UnqualifiedIdResult {
  auto operatorFunctionIdResult = accept(ast->operatorFunctionId);

  for (auto it = ast->templateArgumentList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    GlobalNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    SimpleNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    DecltypeNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto decltypeSpecifierResult = accept(ast->decltypeSpecifier);

  return {};
}

auto ASTInterpreter::NestedNameSpecifierVisitor::operator()(
    TemplateNestedNameSpecifierAST* ast) -> NestedNameSpecifierResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto templateIdResult = accept(ast->templateId);

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    DefaultFunctionBodyAST* ast) -> FunctionBodyResult {
  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    CompoundStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto it = ast->memInitializerList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto statementResult = accept(ast->statement);

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(
    TryStatementFunctionBodyAST* ast) -> FunctionBodyResult {
  for (auto it = ast->memInitializerList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto statementResult = accept(ast->statement);

  for (auto it = ast->handlerList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::FunctionBodyVisitor::operator()(DeleteFunctionBodyAST* ast)
    -> FunctionBodyResult {
  return {};
}

auto ASTInterpreter::TemplateArgumentVisitor::operator()(
    TypeTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::TemplateArgumentVisitor::operator()(
    ExpressionTemplateArgumentAST* ast) -> TemplateArgumentResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::ExceptionSpecifierVisitor::operator()(
    ThrowExceptionSpecifierAST* ast) -> ExceptionSpecifierResult {
  return {};
}

auto ASTInterpreter::ExceptionSpecifierVisitor::operator()(
    NoexceptSpecifierAST* ast) -> ExceptionSpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(SimpleRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(CompoundRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = accept(ast->expression);
  auto typeConstraintResult = accept(ast->typeConstraint);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(TypeRequirementAST* ast)
    -> RequirementResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  return {};
}

auto ASTInterpreter::RequirementVisitor::operator()(NestedRequirementAST* ast)
    -> RequirementResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::NewInitializerVisitor::operator()(
    NewParenInitializerAST* ast) -> NewInitializerResult {
  for (auto it = ast->expressionList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::NewInitializerVisitor::operator()(
    NewBracedInitializerAST* ast) -> NewInitializerResult {
  auto bracedInitListResult = accept(ast->bracedInitList);

  return {};
}

auto ASTInterpreter::MemInitializerVisitor::operator()(
    ParenMemInitializerAST* ast) -> MemInitializerResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);

  for (auto it = ast->expressionList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::MemInitializerVisitor::operator()(
    BracedMemInitializerAST* ast) -> MemInitializerResult {
  auto nestedNameSpecifierResult = accept(ast->nestedNameSpecifier);
  auto unqualifiedIdResult = accept(ast->unqualifiedId);
  auto bracedInitListResult = accept(ast->bracedInitList);

  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(ThisLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    DerefThisLambdaCaptureAST* ast) -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    SimpleLambdaCaptureAST* ast) -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(RefLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(
    RefInitLambdaCaptureAST* ast) -> LambdaCaptureResult {
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::LambdaCaptureVisitor::operator()(InitLambdaCaptureAST* ast)
    -> LambdaCaptureResult {
  auto initializerResult = accept(ast->initializer);

  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    EllipsisExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  return {};
}

auto ASTInterpreter::ExceptionDeclarationVisitor::operator()(
    TypeExceptionDeclarationAST* ast) -> ExceptionDeclarationResult {
  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  for (auto it = ast->typeSpecifierList; it; it = it->next) {
    auto value = accept(it->value);
  }

  auto declaratorResult = accept(ast->declarator);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(CxxAttributeAST* ast)
    -> AttributeSpecifierResult {
  auto attributeUsingPrefixResult = accept(ast->attributeUsingPrefix);

  for (auto it = ast->attributeList; it; it = it->next) {
    auto value = accept(it->value);
  }

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(GccAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(
    AlignasAttributeAST* ast) -> AttributeSpecifierResult {
  auto expressionResult = accept(ast->expression);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(
    AlignasTypeAttributeAST* ast) -> AttributeSpecifierResult {
  auto typeIdResult = accept(ast->typeId);

  return {};
}

auto ASTInterpreter::AttributeSpecifierVisitor::operator()(AsmAttributeAST* ast)
    -> AttributeSpecifierResult {
  return {};
}

auto ASTInterpreter::AttributeTokenVisitor::operator()(
    ScopedAttributeTokenAST* ast) -> AttributeTokenResult {
  return {};
}

auto ASTInterpreter::AttributeTokenVisitor::operator()(
    SimpleAttributeTokenAST* ast) -> AttributeTokenResult {
  return {};
}

}  // namespace cxx
