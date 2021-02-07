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

import * as ast from "./AST";


export abstract class ASTVisitor {
    constructor() { }

    abstract visitTypeId<T, C>(node: ast.TypeIdAST, context: C): T
    abstract visitNestedNameSpecifier<T, C>(node: ast.NestedNameSpecifierAST, context: C): T
    abstract visitUsingDeclarator<T, C>(node: ast.UsingDeclaratorAST, context: C): T
    abstract visitHandler<T, C>(node: ast.HandlerAST, context: C): T
    abstract visitTemplateArgument<T, C>(node: ast.TemplateArgumentAST, context: C): T
    abstract visitEnumBase<T, C>(node: ast.EnumBaseAST, context: C): T
    abstract visitEnumerator<T, C>(node: ast.EnumeratorAST, context: C): T
    abstract visitDeclarator<T, C>(node: ast.DeclaratorAST, context: C): T
    abstract visitBaseSpecifier<T, C>(node: ast.BaseSpecifierAST, context: C): T
    abstract visitBaseClause<T, C>(node: ast.BaseClauseAST, context: C): T
    abstract visitNewTypeId<T, C>(node: ast.NewTypeIdAST, context: C): T
    abstract visitParameterDeclarationClause<T, C>(node: ast.ParameterDeclarationClauseAST, context: C): T
    abstract visitParametersAndQualifiers<T, C>(node: ast.ParametersAndQualifiersAST, context: C): T
    abstract visitEqualInitializer<T, C>(node: ast.EqualInitializerAST, context: C): T
    abstract visitBracedInitList<T, C>(node: ast.BracedInitListAST, context: C): T
    abstract visitParenInitializer<T, C>(node: ast.ParenInitializerAST, context: C): T
    abstract visitNewParenInitializer<T, C>(node: ast.NewParenInitializerAST, context: C): T
    abstract visitNewBracedInitializer<T, C>(node: ast.NewBracedInitializerAST, context: C): T
    abstract visitEllipsisExceptionDeclaration<T, C>(node: ast.EllipsisExceptionDeclarationAST, context: C): T
    abstract visitTypeExceptionDeclaration<T, C>(node: ast.TypeExceptionDeclarationAST, context: C): T
    abstract visitTranslationUnit<T, C>(node: ast.TranslationUnitAST, context: C): T
    abstract visitModuleUnit<T, C>(node: ast.ModuleUnitAST, context: C): T
    abstract visitThisExpression<T, C>(node: ast.ThisExpressionAST, context: C): T
    abstract visitCharLiteralExpression<T, C>(node: ast.CharLiteralExpressionAST, context: C): T
    abstract visitBoolLiteralExpression<T, C>(node: ast.BoolLiteralExpressionAST, context: C): T
    abstract visitIntLiteralExpression<T, C>(node: ast.IntLiteralExpressionAST, context: C): T
    abstract visitFloatLiteralExpression<T, C>(node: ast.FloatLiteralExpressionAST, context: C): T
    abstract visitNullptrLiteralExpression<T, C>(node: ast.NullptrLiteralExpressionAST, context: C): T
    abstract visitStringLiteralExpression<T, C>(node: ast.StringLiteralExpressionAST, context: C): T
    abstract visitUserDefinedStringLiteralExpression<T, C>(node: ast.UserDefinedStringLiteralExpressionAST, context: C): T
    abstract visitIdExpression<T, C>(node: ast.IdExpressionAST, context: C): T
    abstract visitNestedExpression<T, C>(node: ast.NestedExpressionAST, context: C): T
    abstract visitBinaryExpression<T, C>(node: ast.BinaryExpressionAST, context: C): T
    abstract visitAssignmentExpression<T, C>(node: ast.AssignmentExpressionAST, context: C): T
    abstract visitCallExpression<T, C>(node: ast.CallExpressionAST, context: C): T
    abstract visitSubscriptExpression<T, C>(node: ast.SubscriptExpressionAST, context: C): T
    abstract visitMemberExpression<T, C>(node: ast.MemberExpressionAST, context: C): T
    abstract visitConditionalExpression<T, C>(node: ast.ConditionalExpressionAST, context: C): T
    abstract visitCppCastExpression<T, C>(node: ast.CppCastExpressionAST, context: C): T
    abstract visitNewExpression<T, C>(node: ast.NewExpressionAST, context: C): T
    abstract visitLabeledStatement<T, C>(node: ast.LabeledStatementAST, context: C): T
    abstract visitCaseStatement<T, C>(node: ast.CaseStatementAST, context: C): T
    abstract visitDefaultStatement<T, C>(node: ast.DefaultStatementAST, context: C): T
    abstract visitExpressionStatement<T, C>(node: ast.ExpressionStatementAST, context: C): T
    abstract visitCompoundStatement<T, C>(node: ast.CompoundStatementAST, context: C): T
    abstract visitIfStatement<T, C>(node: ast.IfStatementAST, context: C): T
    abstract visitSwitchStatement<T, C>(node: ast.SwitchStatementAST, context: C): T
    abstract visitWhileStatement<T, C>(node: ast.WhileStatementAST, context: C): T
    abstract visitDoStatement<T, C>(node: ast.DoStatementAST, context: C): T
    abstract visitForRangeStatement<T, C>(node: ast.ForRangeStatementAST, context: C): T
    abstract visitForStatement<T, C>(node: ast.ForStatementAST, context: C): T
    abstract visitBreakStatement<T, C>(node: ast.BreakStatementAST, context: C): T
    abstract visitContinueStatement<T, C>(node: ast.ContinueStatementAST, context: C): T
    abstract visitReturnStatement<T, C>(node: ast.ReturnStatementAST, context: C): T
    abstract visitGotoStatement<T, C>(node: ast.GotoStatementAST, context: C): T
    abstract visitCoroutineReturnStatement<T, C>(node: ast.CoroutineReturnStatementAST, context: C): T
    abstract visitDeclarationStatement<T, C>(node: ast.DeclarationStatementAST, context: C): T
    abstract visitTryBlockStatement<T, C>(node: ast.TryBlockStatementAST, context: C): T
    abstract visitFunctionDefinition<T, C>(node: ast.FunctionDefinitionAST, context: C): T
    abstract visitConceptDefinition<T, C>(node: ast.ConceptDefinitionAST, context: C): T
    abstract visitForRangeDeclaration<T, C>(node: ast.ForRangeDeclarationAST, context: C): T
    abstract visitAliasDeclaration<T, C>(node: ast.AliasDeclarationAST, context: C): T
    abstract visitSimpleDeclaration<T, C>(node: ast.SimpleDeclarationAST, context: C): T
    abstract visitStaticAssertDeclaration<T, C>(node: ast.StaticAssertDeclarationAST, context: C): T
    abstract visitEmptyDeclaration<T, C>(node: ast.EmptyDeclarationAST, context: C): T
    abstract visitAttributeDeclaration<T, C>(node: ast.AttributeDeclarationAST, context: C): T
    abstract visitOpaqueEnumDeclaration<T, C>(node: ast.OpaqueEnumDeclarationAST, context: C): T
    abstract visitUsingEnumDeclaration<T, C>(node: ast.UsingEnumDeclarationAST, context: C): T
    abstract visitNamespaceDefinition<T, C>(node: ast.NamespaceDefinitionAST, context: C): T
    abstract visitNamespaceAliasDefinition<T, C>(node: ast.NamespaceAliasDefinitionAST, context: C): T
    abstract visitUsingDirective<T, C>(node: ast.UsingDirectiveAST, context: C): T
    abstract visitUsingDeclaration<T, C>(node: ast.UsingDeclarationAST, context: C): T
    abstract visitAsmDeclaration<T, C>(node: ast.AsmDeclarationAST, context: C): T
    abstract visitExportDeclaration<T, C>(node: ast.ExportDeclarationAST, context: C): T
    abstract visitModuleImportDeclaration<T, C>(node: ast.ModuleImportDeclarationAST, context: C): T
    abstract visitTemplateDeclaration<T, C>(node: ast.TemplateDeclarationAST, context: C): T
    abstract visitDeductionGuide<T, C>(node: ast.DeductionGuideAST, context: C): T
    abstract visitExplicitInstantiation<T, C>(node: ast.ExplicitInstantiationAST, context: C): T
    abstract visitParameterDeclaration<T, C>(node: ast.ParameterDeclarationAST, context: C): T
    abstract visitLinkageSpecification<T, C>(node: ast.LinkageSpecificationAST, context: C): T
    abstract visitSimpleName<T, C>(node: ast.SimpleNameAST, context: C): T
    abstract visitDestructorName<T, C>(node: ast.DestructorNameAST, context: C): T
    abstract visitDecltypeName<T, C>(node: ast.DecltypeNameAST, context: C): T
    abstract visitOperatorName<T, C>(node: ast.OperatorNameAST, context: C): T
    abstract visitTemplateName<T, C>(node: ast.TemplateNameAST, context: C): T
    abstract visitQualifiedName<T, C>(node: ast.QualifiedNameAST, context: C): T
    abstract visitSimpleSpecifier<T, C>(node: ast.SimpleSpecifierAST, context: C): T
    abstract visitExplicitSpecifier<T, C>(node: ast.ExplicitSpecifierAST, context: C): T
    abstract visitNamedTypeSpecifier<T, C>(node: ast.NamedTypeSpecifierAST, context: C): T
    abstract visitPlaceholderTypeSpecifierHelper<T, C>(node: ast.PlaceholderTypeSpecifierHelperAST, context: C): T
    abstract visitDecltypeSpecifierTypeSpecifier<T, C>(node: ast.DecltypeSpecifierTypeSpecifierAST, context: C): T
    abstract visitUnderlyingTypeSpecifier<T, C>(node: ast.UnderlyingTypeSpecifierAST, context: C): T
    abstract visitAtomicTypeSpecifier<T, C>(node: ast.AtomicTypeSpecifierAST, context: C): T
    abstract visitElaboratedTypeSpecifier<T, C>(node: ast.ElaboratedTypeSpecifierAST, context: C): T
    abstract visitDecltypeSpecifier<T, C>(node: ast.DecltypeSpecifierAST, context: C): T
    abstract visitPlaceholderTypeSpecifier<T, C>(node: ast.PlaceholderTypeSpecifierAST, context: C): T
    abstract visitCvQualifier<T, C>(node: ast.CvQualifierAST, context: C): T
    abstract visitEnumSpecifier<T, C>(node: ast.EnumSpecifierAST, context: C): T
    abstract visitClassSpecifier<T, C>(node: ast.ClassSpecifierAST, context: C): T
    abstract visitTypenameSpecifier<T, C>(node: ast.TypenameSpecifierAST, context: C): T
    abstract visitIdDeclarator<T, C>(node: ast.IdDeclaratorAST, context: C): T
    abstract visitNestedDeclarator<T, C>(node: ast.NestedDeclaratorAST, context: C): T
    abstract visitPointerOperator<T, C>(node: ast.PointerOperatorAST, context: C): T
    abstract visitReferenceOperator<T, C>(node: ast.ReferenceOperatorAST, context: C): T
    abstract visitPtrToMemberOperator<T, C>(node: ast.PtrToMemberOperatorAST, context: C): T
    abstract visitFunctionDeclarator<T, C>(node: ast.FunctionDeclaratorAST, context: C): T
    abstract visitArrayDeclarator<T, C>(node: ast.ArrayDeclaratorAST, context: C): T
}

