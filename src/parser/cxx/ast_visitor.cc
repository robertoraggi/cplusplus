// Generated file by: gen_ast_visitor_cc.ts
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

#include <cxx/ast_visitor.h>

// cxx
#include <cxx/ast.h>

namespace cxx {

auto ASTVisitor::preVisit(AST*) -> bool { return true; }

void ASTVisitor::postVisit(AST*) {}

void ASTVisitor::accept(AST* ast) {
  if (!ast) return;
  if (preVisit(ast)) ast->accept(this);
  postVisit(ast);
}

void ASTVisitor::visit(TranslationUnitAST* ast) {
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ModuleUnitAST* ast) {
  accept(ast->globalModuleFragment);
  accept(ast->moduleDeclaration);
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
  accept(ast->privateModuleFragment);
}

void ASTVisitor::visit(SimpleDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->declSpecifierList}) {
    accept(node);
  }
  for (auto node : ListView{ast->initDeclaratorList}) {
    accept(node);
  }
  accept(ast->requiresClause);
}

void ASTVisitor::visit(AsmDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->asmQualifierList}) {
    accept(node);
  }
  for (auto node : ListView{ast->outputOperandList}) {
    accept(node);
  }
  for (auto node : ListView{ast->inputOperandList}) {
    accept(node);
  }
  for (auto node : ListView{ast->clobberList}) {
    accept(node);
  }
  for (auto node : ListView{ast->gotoLabelList}) {
    accept(node);
  }
}

void ASTVisitor::visit(NamespaceAliasDefinitionAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(UsingDeclarationAST* ast) {
  for (auto node : ListView{ast->usingDeclaratorList}) {
    accept(node);
  }
}

void ASTVisitor::visit(UsingEnumDeclarationAST* ast) {
  accept(ast->enumTypeSpecifier);
}

void ASTVisitor::visit(UsingDirectiveAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(StaticAssertDeclarationAST* ast) {
  accept(ast->expression);
}

void ASTVisitor::visit(AliasDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->gnuAttributeList}) {
    accept(node);
  }
  accept(ast->typeId);
}

void ASTVisitor::visit(OpaqueEnumDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  for (auto node : ListView{ast->typeSpecifierList}) {
    accept(node);
  }
}

void ASTVisitor::visit(FunctionDefinitionAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->declSpecifierList}) {
    accept(node);
  }
  accept(ast->declarator);
  accept(ast->requiresClause);
  accept(ast->functionBody);
}

void ASTVisitor::visit(TemplateDeclarationAST* ast) {
  for (auto node : ListView{ast->templateParameterList}) {
    accept(node);
  }
  accept(ast->requiresClause);
  accept(ast->declaration);
}

void ASTVisitor::visit(ConceptDefinitionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(DeductionGuideAST* ast) {
  accept(ast->explicitSpecifier);
  accept(ast->parameterDeclarationClause);
  accept(ast->templateId);
}

void ASTVisitor::visit(ExplicitInstantiationAST* ast) {
  accept(ast->declaration);
}

void ASTVisitor::visit(ExportDeclarationAST* ast) { accept(ast->declaration); }

void ASTVisitor::visit(ExportCompoundDeclarationAST* ast) {
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(LinkageSpecificationAST* ast) {
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(NamespaceDefinitionAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->nestedNamespaceSpecifierList}) {
    accept(node);
  }
  for (auto node : ListView{ast->extraAttributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(EmptyDeclarationAST* ast) {}

void ASTVisitor::visit(AttributeDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ModuleImportDeclarationAST* ast) {
  accept(ast->importName);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ParameterDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->typeSpecifierList}) {
    accept(node);
  }
  accept(ast->declarator);
  accept(ast->expression);
}

void ASTVisitor::visit(AccessDeclarationAST* ast) {}

void ASTVisitor::visit(ForRangeDeclarationAST* ast) {}

void ASTVisitor::visit(StructuredBindingDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->declSpecifierList}) {
    accept(node);
  }
  for (auto node : ListView{ast->bindingList}) {
    accept(node);
  }
  accept(ast->initializer);
}

void ASTVisitor::visit(AsmOperandAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(AsmQualifierAST* ast) {}

void ASTVisitor::visit(AsmClobberAST* ast) {}

void ASTVisitor::visit(AsmGotoLabelAST* ast) {}

void ASTVisitor::visit(SplicerAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(GlobalModuleFragmentAST* ast) {
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(PrivateModuleFragmentAST* ast) {
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ModuleDeclarationAST* ast) {
  accept(ast->moduleName);
  accept(ast->modulePartition);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ModuleNameAST* ast) { accept(ast->moduleQualifier); }

void ASTVisitor::visit(ModuleQualifierAST* ast) {
  accept(ast->moduleQualifier);
}

void ASTVisitor::visit(ModulePartitionAST* ast) { accept(ast->moduleName); }

void ASTVisitor::visit(ImportNameAST* ast) {
  accept(ast->modulePartition);
  accept(ast->moduleName);
}

void ASTVisitor::visit(InitDeclaratorAST* ast) {
  accept(ast->declarator);
  accept(ast->requiresClause);
  accept(ast->initializer);
}

void ASTVisitor::visit(DeclaratorAST* ast) {
  for (auto node : ListView{ast->ptrOpList}) {
    accept(node);
  }
  accept(ast->coreDeclarator);
  for (auto node : ListView{ast->declaratorChunkList}) {
    accept(node);
  }
}

void ASTVisitor::visit(UsingDeclaratorAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(EnumeratorAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->expression);
}

void ASTVisitor::visit(TypeIdAST* ast) {
  for (auto node : ListView{ast->typeSpecifierList}) {
    accept(node);
  }
  accept(ast->declarator);
}

void ASTVisitor::visit(HandlerAST* ast) {
  accept(ast->exceptionDeclaration);
  accept(ast->statement);
}

void ASTVisitor::visit(BaseSpecifierAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(RequiresClauseAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(ParameterDeclarationClauseAST* ast) {
  for (auto node : ListView{ast->parameterDeclarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(TrailingReturnTypeAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(LambdaSpecifierAST* ast) {}

void ASTVisitor::visit(TypeConstraintAST* ast) {
  accept(ast->nestedNameSpecifier);
  for (auto node : ListView{ast->templateArgumentList}) {
    accept(node);
  }
}

void ASTVisitor::visit(AttributeArgumentClauseAST* ast) {}

void ASTVisitor::visit(AttributeAST* ast) {
  accept(ast->attributeToken);
  accept(ast->attributeArgumentClause);
}

void ASTVisitor::visit(AttributeUsingPrefixAST* ast) {}

void ASTVisitor::visit(NewPlacementAST* ast) {
  for (auto node : ListView{ast->expressionList}) {
    accept(node);
  }
}

void ASTVisitor::visit(NestedNamespaceSpecifierAST* ast) {}

void ASTVisitor::visit(LabeledStatementAST* ast) {}

void ASTVisitor::visit(CaseStatementAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(DefaultStatementAST* ast) {}

void ASTVisitor::visit(ExpressionStatementAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(CompoundStatementAST* ast) {
  for (auto node : ListView{ast->statementList}) {
    accept(node);
  }
}

void ASTVisitor::visit(IfStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->condition);
  accept(ast->statement);
  accept(ast->elseStatement);
}

void ASTVisitor::visit(ConstevalIfStatementAST* ast) {
  accept(ast->statement);
  accept(ast->elseStatement);
}

void ASTVisitor::visit(SwitchStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->condition);
  accept(ast->statement);
}

void ASTVisitor::visit(WhileStatementAST* ast) {
  accept(ast->condition);
  accept(ast->statement);
}

void ASTVisitor::visit(DoStatementAST* ast) {
  accept(ast->statement);
  accept(ast->expression);
}

void ASTVisitor::visit(ForRangeStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->rangeDeclaration);
  accept(ast->rangeInitializer);
  accept(ast->statement);
}

void ASTVisitor::visit(ForStatementAST* ast) {
  accept(ast->initializer);
  accept(ast->condition);
  accept(ast->expression);
  accept(ast->statement);
}

void ASTVisitor::visit(BreakStatementAST* ast) {}

void ASTVisitor::visit(ContinueStatementAST* ast) {}

void ASTVisitor::visit(ReturnStatementAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(CoroutineReturnStatementAST* ast) {
  accept(ast->expression);
}

void ASTVisitor::visit(GotoStatementAST* ast) {}

void ASTVisitor::visit(DeclarationStatementAST* ast) {
  accept(ast->declaration);
}

void ASTVisitor::visit(TryBlockStatementAST* ast) {
  accept(ast->statement);
  for (auto node : ListView{ast->handlerList}) {
    accept(node);
  }
}

void ASTVisitor::visit(CharLiteralExpressionAST* ast) {}

void ASTVisitor::visit(BoolLiteralExpressionAST* ast) {}

void ASTVisitor::visit(IntLiteralExpressionAST* ast) {}

void ASTVisitor::visit(FloatLiteralExpressionAST* ast) {}

void ASTVisitor::visit(NullptrLiteralExpressionAST* ast) {}

void ASTVisitor::visit(StringLiteralExpressionAST* ast) {}

void ASTVisitor::visit(UserDefinedStringLiteralExpressionAST* ast) {}

void ASTVisitor::visit(ObjectLiteralExpressionAST* ast) {
  accept(ast->typeId);
  accept(ast->bracedInitList);
}

void ASTVisitor::visit(ThisExpressionAST* ast) {}

void ASTVisitor::visit(GenericSelectionExpressionAST* ast) {
  accept(ast->expression);
  for (auto node : ListView{ast->genericAssociationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(NestedStatementExpressionAST* ast) {
  accept(ast->statement);
}

void ASTVisitor::visit(NestedExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(IdExpressionAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(LambdaExpressionAST* ast) {
  for (auto node : ListView{ast->captureList}) {
    accept(node);
  }
  for (auto node : ListView{ast->templateParameterList}) {
    accept(node);
  }
  accept(ast->templateRequiresClause);
  for (auto node : ListView{ast->expressionAttributeList}) {
    accept(node);
  }
  accept(ast->parameterDeclarationClause);
  for (auto node : ListView{ast->gnuAtributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->lambdaSpecifierList}) {
    accept(node);
  }
  accept(ast->exceptionSpecifier);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->trailingReturnType);
  accept(ast->requiresClause);
  accept(ast->statement);
}

void ASTVisitor::visit(FoldExpressionAST* ast) {
  accept(ast->leftExpression);
  accept(ast->rightExpression);
}

void ASTVisitor::visit(RightFoldExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(LeftFoldExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(RequiresExpressionAST* ast) {
  accept(ast->parameterDeclarationClause);
  for (auto node : ListView{ast->requirementList}) {
    accept(node);
  }
}

void ASTVisitor::visit(VaArgExpressionAST* ast) {
  accept(ast->expression);
  accept(ast->typeId);
}

void ASTVisitor::visit(SubscriptExpressionAST* ast) {
  accept(ast->baseExpression);
  accept(ast->indexExpression);
}

void ASTVisitor::visit(CallExpressionAST* ast) {
  accept(ast->baseExpression);
  for (auto node : ListView{ast->expressionList}) {
    accept(node);
  }
}

void ASTVisitor::visit(TypeConstructionAST* ast) {
  accept(ast->typeSpecifier);
  for (auto node : ListView{ast->expressionList}) {
    accept(node);
  }
}

void ASTVisitor::visit(BracedTypeConstructionAST* ast) {
  accept(ast->typeSpecifier);
  accept(ast->bracedInitList);
}

void ASTVisitor::visit(SpliceMemberExpressionAST* ast) {
  accept(ast->baseExpression);
  accept(ast->splicer);
}

void ASTVisitor::visit(MemberExpressionAST* ast) {
  accept(ast->baseExpression);
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(PostIncrExpressionAST* ast) {
  accept(ast->baseExpression);
}

void ASTVisitor::visit(CppCastExpressionAST* ast) {
  accept(ast->typeId);
  accept(ast->expression);
}

void ASTVisitor::visit(BuiltinBitCastExpressionAST* ast) {
  accept(ast->typeId);
  accept(ast->expression);
}

void ASTVisitor::visit(BuiltinOffsetofExpressionAST* ast) {
  accept(ast->typeId);
  for (auto node : ListView{ast->designatorList}) {
    accept(node);
  }
}

void ASTVisitor::visit(TypeidExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(TypeidOfTypeExpressionAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(SpliceExpressionAST* ast) { accept(ast->splicer); }

void ASTVisitor::visit(GlobalScopeReflectExpressionAST* ast) {}

void ASTVisitor::visit(NamespaceReflectExpressionAST* ast) {}

void ASTVisitor::visit(TypeIdReflectExpressionAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(ReflectExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(LabelAddressExpressionAST* ast) {}

void ASTVisitor::visit(UnaryExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(AwaitExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(SizeofExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(SizeofTypeExpressionAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(SizeofPackExpressionAST* ast) {}

void ASTVisitor::visit(AlignofTypeExpressionAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(AlignofExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(NoexceptExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(NewExpressionAST* ast) {
  accept(ast->newPlacement);
  for (auto node : ListView{ast->typeSpecifierList}) {
    accept(node);
  }
  accept(ast->declarator);
  accept(ast->newInitalizer);
}

void ASTVisitor::visit(DeleteExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(CastExpressionAST* ast) {
  accept(ast->typeId);
  accept(ast->expression);
}

void ASTVisitor::visit(ImplicitCastExpressionAST* ast) {
  accept(ast->expression);
}

void ASTVisitor::visit(BinaryExpressionAST* ast) {
  accept(ast->leftExpression);
  accept(ast->rightExpression);
}

void ASTVisitor::visit(ConditionalExpressionAST* ast) {
  accept(ast->condition);
  accept(ast->iftrueExpression);
  accept(ast->iffalseExpression);
}

void ASTVisitor::visit(YieldExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(ThrowExpressionAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(AssignmentExpressionAST* ast) {
  accept(ast->leftExpression);
  accept(ast->rightExpression);
}

void ASTVisitor::visit(LeftExpressionAST* ast) {}

void ASTVisitor::visit(CompoundAssignmentExpressionAST* ast) {
  accept(ast->targetExpression);
  accept(ast->leftExpression);
  accept(ast->rightExpression);
}

void ASTVisitor::visit(PackExpansionExpressionAST* ast) {
  accept(ast->expression);
}

void ASTVisitor::visit(DesignatedInitializerClauseAST* ast) {
  for (auto node : ListView{ast->designatorList}) {
    accept(node);
  }
  accept(ast->initializer);
}

void ASTVisitor::visit(TypeTraitExpressionAST* ast) {
  for (auto node : ListView{ast->typeIdList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ConditionExpressionAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->declSpecifierList}) {
    accept(node);
  }
  accept(ast->declarator);
  accept(ast->initializer);
}

void ASTVisitor::visit(EqualInitializerAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(BracedInitListAST* ast) {
  for (auto node : ListView{ast->expressionList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ParenInitializerAST* ast) {
  for (auto node : ListView{ast->expressionList}) {
    accept(node);
  }
}

void ASTVisitor::visit(DefaultGenericAssociationAST* ast) {
  accept(ast->expression);
}

void ASTVisitor::visit(TypeGenericAssociationAST* ast) {
  accept(ast->typeId);
  accept(ast->expression);
}

void ASTVisitor::visit(DotDesignatorAST* ast) {}

void ASTVisitor::visit(SubscriptDesignatorAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(TemplateTypeParameterAST* ast) {
  for (auto node : ListView{ast->templateParameterList}) {
    accept(node);
  }
  accept(ast->requiresClause);
  accept(ast->idExpression);
}

void ASTVisitor::visit(NonTypeTemplateParameterAST* ast) {
  accept(ast->declaration);
}

void ASTVisitor::visit(TypenameTypeParameterAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(ConstraintTypeParameterAST* ast) {
  accept(ast->typeConstraint);
  accept(ast->typeId);
}

void ASTVisitor::visit(TypedefSpecifierAST* ast) {}

void ASTVisitor::visit(FriendSpecifierAST* ast) {}

void ASTVisitor::visit(ConstevalSpecifierAST* ast) {}

void ASTVisitor::visit(ConstinitSpecifierAST* ast) {}

void ASTVisitor::visit(ConstexprSpecifierAST* ast) {}

void ASTVisitor::visit(InlineSpecifierAST* ast) {}

void ASTVisitor::visit(NoreturnSpecifierAST* ast) {}

void ASTVisitor::visit(StaticSpecifierAST* ast) {}

void ASTVisitor::visit(ExternSpecifierAST* ast) {}

void ASTVisitor::visit(RegisterSpecifierAST* ast) {}

void ASTVisitor::visit(ThreadLocalSpecifierAST* ast) {}

void ASTVisitor::visit(ThreadSpecifierAST* ast) {}

void ASTVisitor::visit(MutableSpecifierAST* ast) {}

void ASTVisitor::visit(VirtualSpecifierAST* ast) {}

void ASTVisitor::visit(ExplicitSpecifierAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(AutoTypeSpecifierAST* ast) {}

void ASTVisitor::visit(VoidTypeSpecifierAST* ast) {}

void ASTVisitor::visit(SizeTypeSpecifierAST* ast) {}

void ASTVisitor::visit(SignTypeSpecifierAST* ast) {}

void ASTVisitor::visit(BuiltinTypeSpecifierAST* ast) {}

void ASTVisitor::visit(UnaryBuiltinTypeSpecifierAST* ast) {
  accept(ast->typeId);
}

void ASTVisitor::visit(BinaryBuiltinTypeSpecifierAST* ast) {
  accept(ast->leftTypeId);
  accept(ast->rightTypeId);
}

void ASTVisitor::visit(IntegralTypeSpecifierAST* ast) {}

void ASTVisitor::visit(FloatingPointTypeSpecifierAST* ast) {}

void ASTVisitor::visit(ComplexTypeSpecifierAST* ast) {}

void ASTVisitor::visit(NamedTypeSpecifierAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(AtomicTypeSpecifierAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(UnderlyingTypeSpecifierAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(ElaboratedTypeSpecifierAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(DecltypeAutoSpecifierAST* ast) {}

void ASTVisitor::visit(DecltypeSpecifierAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(PlaceholderTypeSpecifierAST* ast) {
  accept(ast->typeConstraint);
  accept(ast->specifier);
}

void ASTVisitor::visit(ConstQualifierAST* ast) {}

void ASTVisitor::visit(VolatileQualifierAST* ast) {}

void ASTVisitor::visit(AtomicQualifierAST* ast) {}

void ASTVisitor::visit(RestrictQualifierAST* ast) {}

void ASTVisitor::visit(EnumSpecifierAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  for (auto node : ListView{ast->typeSpecifierList}) {
    accept(node);
  }
  for (auto node : ListView{ast->enumeratorList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ClassSpecifierAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  for (auto node : ListView{ast->baseSpecifierList}) {
    accept(node);
  }
  for (auto node : ListView{ast->declarationList}) {
    accept(node);
  }
}

void ASTVisitor::visit(TypenameSpecifierAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(SplicerTypeSpecifierAST* ast) { accept(ast->splicer); }

void ASTVisitor::visit(PointerOperatorAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->cvQualifierList}) {
    accept(node);
  }
}

void ASTVisitor::visit(ReferenceOperatorAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
}

void ASTVisitor::visit(PtrToMemberOperatorAST* ast) {
  accept(ast->nestedNameSpecifier);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->cvQualifierList}) {
    accept(node);
  }
}

void ASTVisitor::visit(BitfieldDeclaratorAST* ast) {
  accept(ast->unqualifiedId);
  accept(ast->sizeExpression);
}

void ASTVisitor::visit(ParameterPackAST* ast) { accept(ast->coreDeclarator); }

void ASTVisitor::visit(IdDeclaratorAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
}

void ASTVisitor::visit(NestedDeclaratorAST* ast) { accept(ast->declarator); }

void ASTVisitor::visit(FunctionDeclaratorChunkAST* ast) {
  accept(ast->parameterDeclarationClause);
  for (auto node : ListView{ast->cvQualifierList}) {
    accept(node);
  }
  accept(ast->exceptionSpecifier);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  accept(ast->trailingReturnType);
}

void ASTVisitor::visit(ArrayDeclaratorChunkAST* ast) {
  for (auto node : ListView{ast->typeQualifierList}) {
    accept(node);
  }
  accept(ast->expression);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
}

void ASTVisitor::visit(NameIdAST* ast) {}

void ASTVisitor::visit(DestructorIdAST* ast) { accept(ast->id); }

void ASTVisitor::visit(DecltypeIdAST* ast) { accept(ast->decltypeSpecifier); }

void ASTVisitor::visit(OperatorFunctionIdAST* ast) {}

void ASTVisitor::visit(LiteralOperatorIdAST* ast) {}

void ASTVisitor::visit(ConversionFunctionIdAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(SimpleTemplateIdAST* ast) {
  for (auto node : ListView{ast->templateArgumentList}) {
    accept(node);
  }
}

void ASTVisitor::visit(LiteralOperatorTemplateIdAST* ast) {
  accept(ast->literalOperatorId);
  for (auto node : ListView{ast->templateArgumentList}) {
    accept(node);
  }
}

void ASTVisitor::visit(OperatorFunctionTemplateIdAST* ast) {
  accept(ast->operatorFunctionId);
  for (auto node : ListView{ast->templateArgumentList}) {
    accept(node);
  }
}

void ASTVisitor::visit(GlobalNestedNameSpecifierAST* ast) {}

void ASTVisitor::visit(SimpleNestedNameSpecifierAST* ast) {
  accept(ast->nestedNameSpecifier);
}

void ASTVisitor::visit(DecltypeNestedNameSpecifierAST* ast) {
  accept(ast->decltypeSpecifier);
}

void ASTVisitor::visit(TemplateNestedNameSpecifierAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->templateId);
}

void ASTVisitor::visit(DefaultFunctionBodyAST* ast) {}

void ASTVisitor::visit(CompoundStatementFunctionBodyAST* ast) {
  for (auto node : ListView{ast->memInitializerList}) {
    accept(node);
  }
  accept(ast->statement);
}

void ASTVisitor::visit(TryStatementFunctionBodyAST* ast) {
  for (auto node : ListView{ast->memInitializerList}) {
    accept(node);
  }
  accept(ast->statement);
  for (auto node : ListView{ast->handlerList}) {
    accept(node);
  }
}

void ASTVisitor::visit(DeleteFunctionBodyAST* ast) {}

void ASTVisitor::visit(TypeTemplateArgumentAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(ExpressionTemplateArgumentAST* ast) {
  accept(ast->expression);
}

void ASTVisitor::visit(ThrowExceptionSpecifierAST* ast) {}

void ASTVisitor::visit(NoexceptSpecifierAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(SimpleRequirementAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(CompoundRequirementAST* ast) {
  accept(ast->expression);
  accept(ast->typeConstraint);
}

void ASTVisitor::visit(TypeRequirementAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
}

void ASTVisitor::visit(NestedRequirementAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(NewParenInitializerAST* ast) {
  for (auto node : ListView{ast->expressionList}) {
    accept(node);
  }
}

void ASTVisitor::visit(NewBracedInitializerAST* ast) {
  accept(ast->bracedInitList);
}

void ASTVisitor::visit(ParenMemInitializerAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  for (auto node : ListView{ast->expressionList}) {
    accept(node);
  }
}

void ASTVisitor::visit(BracedMemInitializerAST* ast) {
  accept(ast->nestedNameSpecifier);
  accept(ast->unqualifiedId);
  accept(ast->bracedInitList);
}

void ASTVisitor::visit(ThisLambdaCaptureAST* ast) {}

void ASTVisitor::visit(DerefThisLambdaCaptureAST* ast) {}

void ASTVisitor::visit(SimpleLambdaCaptureAST* ast) {}

void ASTVisitor::visit(RefLambdaCaptureAST* ast) {}

void ASTVisitor::visit(RefInitLambdaCaptureAST* ast) {
  accept(ast->initializer);
}

void ASTVisitor::visit(InitLambdaCaptureAST* ast) { accept(ast->initializer); }

void ASTVisitor::visit(EllipsisExceptionDeclarationAST* ast) {}

void ASTVisitor::visit(TypeExceptionDeclarationAST* ast) {
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
  for (auto node : ListView{ast->typeSpecifierList}) {
    accept(node);
  }
  accept(ast->declarator);
}

void ASTVisitor::visit(CxxAttributeAST* ast) {
  accept(ast->attributeUsingPrefix);
  for (auto node : ListView{ast->attributeList}) {
    accept(node);
  }
}

void ASTVisitor::visit(GccAttributeAST* ast) {}

void ASTVisitor::visit(AlignasAttributeAST* ast) { accept(ast->expression); }

void ASTVisitor::visit(AlignasTypeAttributeAST* ast) { accept(ast->typeId); }

void ASTVisitor::visit(AsmAttributeAST* ast) {}

void ASTVisitor::visit(ScopedAttributeTokenAST* ast) {}

void ASTVisitor::visit(SimpleAttributeTokenAST* ast) {}

}  // namespace cxx
