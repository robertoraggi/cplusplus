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


export abstract class ASTVisitor<Context, Result> {
    constructor() { }

    // AST
    abstract visitTypeId(node: ast.TypeIdAST, context: Context): Result;
    abstract visitNestedNameSpecifier(node: ast.NestedNameSpecifierAST, context: Context): Result;
    abstract visitUsingDeclarator(node: ast.UsingDeclaratorAST, context: Context): Result;
    abstract visitHandler(node: ast.HandlerAST, context: Context): Result;
    abstract visitEnumBase(node: ast.EnumBaseAST, context: Context): Result;
    abstract visitEnumerator(node: ast.EnumeratorAST, context: Context): Result;
    abstract visitDeclarator(node: ast.DeclaratorAST, context: Context): Result;
    abstract visitInitDeclarator(node: ast.InitDeclaratorAST, context: Context): Result;
    abstract visitBaseSpecifier(node: ast.BaseSpecifierAST, context: Context): Result;
    abstract visitBaseClause(node: ast.BaseClauseAST, context: Context): Result;
    abstract visitNewTypeId(node: ast.NewTypeIdAST, context: Context): Result;
    abstract visitRequiresClause(node: ast.RequiresClauseAST, context: Context): Result;
    abstract visitParameterDeclarationClause(node: ast.ParameterDeclarationClauseAST, context: Context): Result;
    abstract visitParametersAndQualifiers(node: ast.ParametersAndQualifiersAST, context: Context): Result;
    abstract visitLambdaIntroducer(node: ast.LambdaIntroducerAST, context: Context): Result;
    abstract visitLambdaDeclarator(node: ast.LambdaDeclaratorAST, context: Context): Result;
    abstract visitTrailingReturnType(node: ast.TrailingReturnTypeAST, context: Context): Result;
    abstract visitCtorInitializer(node: ast.CtorInitializerAST, context: Context): Result;

    // TemplateArgumentAST
    abstract visitTypeTemplateArgument(node: ast.TypeTemplateArgumentAST, context: Context): Result;
    abstract visitExpressionTemplateArgument(node: ast.ExpressionTemplateArgumentAST, context: Context): Result;

    // MemInitializerAST
    abstract visitParenMemInitializer(node: ast.ParenMemInitializerAST, context: Context): Result;
    abstract visitBracedMemInitializer(node: ast.BracedMemInitializerAST, context: Context): Result;

    // LambdaCaptureAST
    abstract visitThisLambdaCapture(node: ast.ThisLambdaCaptureAST, context: Context): Result;
    abstract visitDerefThisLambdaCapture(node: ast.DerefThisLambdaCaptureAST, context: Context): Result;
    abstract visitSimpleLambdaCapture(node: ast.SimpleLambdaCaptureAST, context: Context): Result;
    abstract visitRefLambdaCapture(node: ast.RefLambdaCaptureAST, context: Context): Result;
    abstract visitRefInitLambdaCapture(node: ast.RefInitLambdaCaptureAST, context: Context): Result;
    abstract visitInitLambdaCapture(node: ast.InitLambdaCaptureAST, context: Context): Result;

    // InitializerAST
    abstract visitEqualInitializer(node: ast.EqualInitializerAST, context: Context): Result;
    abstract visitBracedInitList(node: ast.BracedInitListAST, context: Context): Result;
    abstract visitParenInitializer(node: ast.ParenInitializerAST, context: Context): Result;

    // NewInitializerAST
    abstract visitNewParenInitializer(node: ast.NewParenInitializerAST, context: Context): Result;
    abstract visitNewBracedInitializer(node: ast.NewBracedInitializerAST, context: Context): Result;

    // ExceptionDeclarationAST
    abstract visitEllipsisExceptionDeclaration(node: ast.EllipsisExceptionDeclarationAST, context: Context): Result;
    abstract visitTypeExceptionDeclaration(node: ast.TypeExceptionDeclarationAST, context: Context): Result;

    // FunctionBodyAST
    abstract visitDefaultFunctionBody(node: ast.DefaultFunctionBodyAST, context: Context): Result;
    abstract visitCompoundStatementFunctionBody(node: ast.CompoundStatementFunctionBodyAST, context: Context): Result;
    abstract visitTryStatementFunctionBody(node: ast.TryStatementFunctionBodyAST, context: Context): Result;
    abstract visitDeleteFunctionBody(node: ast.DeleteFunctionBodyAST, context: Context): Result;

    // UnitAST
    abstract visitTranslationUnit(node: ast.TranslationUnitAST, context: Context): Result;
    abstract visitModuleUnit(node: ast.ModuleUnitAST, context: Context): Result;

    // ExpressionAST
    abstract visitThisExpression(node: ast.ThisExpressionAST, context: Context): Result;
    abstract visitCharLiteralExpression(node: ast.CharLiteralExpressionAST, context: Context): Result;
    abstract visitBoolLiteralExpression(node: ast.BoolLiteralExpressionAST, context: Context): Result;
    abstract visitIntLiteralExpression(node: ast.IntLiteralExpressionAST, context: Context): Result;
    abstract visitFloatLiteralExpression(node: ast.FloatLiteralExpressionAST, context: Context): Result;
    abstract visitNullptrLiteralExpression(node: ast.NullptrLiteralExpressionAST, context: Context): Result;
    abstract visitStringLiteralExpression(node: ast.StringLiteralExpressionAST, context: Context): Result;
    abstract visitUserDefinedStringLiteralExpression(node: ast.UserDefinedStringLiteralExpressionAST, context: Context): Result;
    abstract visitIdExpression(node: ast.IdExpressionAST, context: Context): Result;
    abstract visitNestedExpression(node: ast.NestedExpressionAST, context: Context): Result;
    abstract visitRightFoldExpression(node: ast.RightFoldExpressionAST, context: Context): Result;
    abstract visitLeftFoldExpression(node: ast.LeftFoldExpressionAST, context: Context): Result;
    abstract visitFoldExpression(node: ast.FoldExpressionAST, context: Context): Result;
    abstract visitLambdaExpression(node: ast.LambdaExpressionAST, context: Context): Result;
    abstract visitSizeofExpression(node: ast.SizeofExpressionAST, context: Context): Result;
    abstract visitSizeofTypeExpression(node: ast.SizeofTypeExpressionAST, context: Context): Result;
    abstract visitSizeofPackExpression(node: ast.SizeofPackExpressionAST, context: Context): Result;
    abstract visitTypeidExpression(node: ast.TypeidExpressionAST, context: Context): Result;
    abstract visitTypeidOfTypeExpression(node: ast.TypeidOfTypeExpressionAST, context: Context): Result;
    abstract visitAlignofExpression(node: ast.AlignofExpressionAST, context: Context): Result;
    abstract visitUnaryExpression(node: ast.UnaryExpressionAST, context: Context): Result;
    abstract visitBinaryExpression(node: ast.BinaryExpressionAST, context: Context): Result;
    abstract visitAssignmentExpression(node: ast.AssignmentExpressionAST, context: Context): Result;
    abstract visitBracedTypeConstruction(node: ast.BracedTypeConstructionAST, context: Context): Result;
    abstract visitTypeConstruction(node: ast.TypeConstructionAST, context: Context): Result;
    abstract visitCallExpression(node: ast.CallExpressionAST, context: Context): Result;
    abstract visitSubscriptExpression(node: ast.SubscriptExpressionAST, context: Context): Result;
    abstract visitMemberExpression(node: ast.MemberExpressionAST, context: Context): Result;
    abstract visitPostIncrExpression(node: ast.PostIncrExpressionAST, context: Context): Result;
    abstract visitConditionalExpression(node: ast.ConditionalExpressionAST, context: Context): Result;
    abstract visitImplicitCastExpression(node: ast.ImplicitCastExpressionAST, context: Context): Result;
    abstract visitCastExpression(node: ast.CastExpressionAST, context: Context): Result;
    abstract visitCppCastExpression(node: ast.CppCastExpressionAST, context: Context): Result;
    abstract visitNewExpression(node: ast.NewExpressionAST, context: Context): Result;
    abstract visitDeleteExpression(node: ast.DeleteExpressionAST, context: Context): Result;
    abstract visitThrowExpression(node: ast.ThrowExpressionAST, context: Context): Result;
    abstract visitNoexceptExpression(node: ast.NoexceptExpressionAST, context: Context): Result;

    // StatementAST
    abstract visitLabeledStatement(node: ast.LabeledStatementAST, context: Context): Result;
    abstract visitCaseStatement(node: ast.CaseStatementAST, context: Context): Result;
    abstract visitDefaultStatement(node: ast.DefaultStatementAST, context: Context): Result;
    abstract visitExpressionStatement(node: ast.ExpressionStatementAST, context: Context): Result;
    abstract visitCompoundStatement(node: ast.CompoundStatementAST, context: Context): Result;
    abstract visitIfStatement(node: ast.IfStatementAST, context: Context): Result;
    abstract visitSwitchStatement(node: ast.SwitchStatementAST, context: Context): Result;
    abstract visitWhileStatement(node: ast.WhileStatementAST, context: Context): Result;
    abstract visitDoStatement(node: ast.DoStatementAST, context: Context): Result;
    abstract visitForRangeStatement(node: ast.ForRangeStatementAST, context: Context): Result;
    abstract visitForStatement(node: ast.ForStatementAST, context: Context): Result;
    abstract visitBreakStatement(node: ast.BreakStatementAST, context: Context): Result;
    abstract visitContinueStatement(node: ast.ContinueStatementAST, context: Context): Result;
    abstract visitReturnStatement(node: ast.ReturnStatementAST, context: Context): Result;
    abstract visitGotoStatement(node: ast.GotoStatementAST, context: Context): Result;
    abstract visitCoroutineReturnStatement(node: ast.CoroutineReturnStatementAST, context: Context): Result;
    abstract visitDeclarationStatement(node: ast.DeclarationStatementAST, context: Context): Result;
    abstract visitTryBlockStatement(node: ast.TryBlockStatementAST, context: Context): Result;

    // DeclarationAST
    abstract visitAccessDeclaration(node: ast.AccessDeclarationAST, context: Context): Result;
    abstract visitFunctionDefinition(node: ast.FunctionDefinitionAST, context: Context): Result;
    abstract visitConceptDefinition(node: ast.ConceptDefinitionAST, context: Context): Result;
    abstract visitForRangeDeclaration(node: ast.ForRangeDeclarationAST, context: Context): Result;
    abstract visitAliasDeclaration(node: ast.AliasDeclarationAST, context: Context): Result;
    abstract visitSimpleDeclaration(node: ast.SimpleDeclarationAST, context: Context): Result;
    abstract visitStaticAssertDeclaration(node: ast.StaticAssertDeclarationAST, context: Context): Result;
    abstract visitEmptyDeclaration(node: ast.EmptyDeclarationAST, context: Context): Result;
    abstract visitAttributeDeclaration(node: ast.AttributeDeclarationAST, context: Context): Result;
    abstract visitOpaqueEnumDeclaration(node: ast.OpaqueEnumDeclarationAST, context: Context): Result;
    abstract visitUsingEnumDeclaration(node: ast.UsingEnumDeclarationAST, context: Context): Result;
    abstract visitNamespaceDefinition(node: ast.NamespaceDefinitionAST, context: Context): Result;
    abstract visitNamespaceAliasDefinition(node: ast.NamespaceAliasDefinitionAST, context: Context): Result;
    abstract visitUsingDirective(node: ast.UsingDirectiveAST, context: Context): Result;
    abstract visitUsingDeclaration(node: ast.UsingDeclarationAST, context: Context): Result;
    abstract visitAsmDeclaration(node: ast.AsmDeclarationAST, context: Context): Result;
    abstract visitExportDeclaration(node: ast.ExportDeclarationAST, context: Context): Result;
    abstract visitModuleImportDeclaration(node: ast.ModuleImportDeclarationAST, context: Context): Result;
    abstract visitTemplateDeclaration(node: ast.TemplateDeclarationAST, context: Context): Result;
    abstract visitTypenameTypeParameter(node: ast.TypenameTypeParameterAST, context: Context): Result;
    abstract visitTypenamePackTypeParameter(node: ast.TypenamePackTypeParameterAST, context: Context): Result;
    abstract visitTemplateTypeParameter(node: ast.TemplateTypeParameterAST, context: Context): Result;
    abstract visitTemplatePackTypeParameter(node: ast.TemplatePackTypeParameterAST, context: Context): Result;
    abstract visitDeductionGuide(node: ast.DeductionGuideAST, context: Context): Result;
    abstract visitExplicitInstantiation(node: ast.ExplicitInstantiationAST, context: Context): Result;
    abstract visitParameterDeclaration(node: ast.ParameterDeclarationAST, context: Context): Result;
    abstract visitLinkageSpecification(node: ast.LinkageSpecificationAST, context: Context): Result;

    // NameAST
    abstract visitSimpleName(node: ast.SimpleNameAST, context: Context): Result;
    abstract visitDestructorName(node: ast.DestructorNameAST, context: Context): Result;
    abstract visitDecltypeName(node: ast.DecltypeNameAST, context: Context): Result;
    abstract visitOperatorName(node: ast.OperatorNameAST, context: Context): Result;
    abstract visitConversionName(node: ast.ConversionNameAST, context: Context): Result;
    abstract visitTemplateName(node: ast.TemplateNameAST, context: Context): Result;
    abstract visitQualifiedName(node: ast.QualifiedNameAST, context: Context): Result;

    // SpecifierAST
    abstract visitTypedefSpecifier(node: ast.TypedefSpecifierAST, context: Context): Result;
    abstract visitFriendSpecifier(node: ast.FriendSpecifierAST, context: Context): Result;
    abstract visitConstevalSpecifier(node: ast.ConstevalSpecifierAST, context: Context): Result;
    abstract visitConstinitSpecifier(node: ast.ConstinitSpecifierAST, context: Context): Result;
    abstract visitConstexprSpecifier(node: ast.ConstexprSpecifierAST, context: Context): Result;
    abstract visitInlineSpecifier(node: ast.InlineSpecifierAST, context: Context): Result;
    abstract visitStaticSpecifier(node: ast.StaticSpecifierAST, context: Context): Result;
    abstract visitExternSpecifier(node: ast.ExternSpecifierAST, context: Context): Result;
    abstract visitThreadLocalSpecifier(node: ast.ThreadLocalSpecifierAST, context: Context): Result;
    abstract visitThreadSpecifier(node: ast.ThreadSpecifierAST, context: Context): Result;
    abstract visitMutableSpecifier(node: ast.MutableSpecifierAST, context: Context): Result;
    abstract visitVirtualSpecifier(node: ast.VirtualSpecifierAST, context: Context): Result;
    abstract visitExplicitSpecifier(node: ast.ExplicitSpecifierAST, context: Context): Result;
    abstract visitAutoTypeSpecifier(node: ast.AutoTypeSpecifierAST, context: Context): Result;
    abstract visitVoidTypeSpecifier(node: ast.VoidTypeSpecifierAST, context: Context): Result;
    abstract visitVaListTypeSpecifier(node: ast.VaListTypeSpecifierAST, context: Context): Result;
    abstract visitIntegralTypeSpecifier(node: ast.IntegralTypeSpecifierAST, context: Context): Result;
    abstract visitFloatingPointTypeSpecifier(node: ast.FloatingPointTypeSpecifierAST, context: Context): Result;
    abstract visitComplexTypeSpecifier(node: ast.ComplexTypeSpecifierAST, context: Context): Result;
    abstract visitNamedTypeSpecifier(node: ast.NamedTypeSpecifierAST, context: Context): Result;
    abstract visitAtomicTypeSpecifier(node: ast.AtomicTypeSpecifierAST, context: Context): Result;
    abstract visitUnderlyingTypeSpecifier(node: ast.UnderlyingTypeSpecifierAST, context: Context): Result;
    abstract visitElaboratedTypeSpecifier(node: ast.ElaboratedTypeSpecifierAST, context: Context): Result;
    abstract visitDecltypeAutoSpecifier(node: ast.DecltypeAutoSpecifierAST, context: Context): Result;
    abstract visitDecltypeSpecifier(node: ast.DecltypeSpecifierAST, context: Context): Result;
    abstract visitTypeofSpecifier(node: ast.TypeofSpecifierAST, context: Context): Result;
    abstract visitPlaceholderTypeSpecifier(node: ast.PlaceholderTypeSpecifierAST, context: Context): Result;
    abstract visitConstQualifier(node: ast.ConstQualifierAST, context: Context): Result;
    abstract visitVolatileQualifier(node: ast.VolatileQualifierAST, context: Context): Result;
    abstract visitRestrictQualifier(node: ast.RestrictQualifierAST, context: Context): Result;
    abstract visitEnumSpecifier(node: ast.EnumSpecifierAST, context: Context): Result;
    abstract visitClassSpecifier(node: ast.ClassSpecifierAST, context: Context): Result;
    abstract visitTypenameSpecifier(node: ast.TypenameSpecifierAST, context: Context): Result;

    // CoreDeclaratorAST
    abstract visitIdDeclarator(node: ast.IdDeclaratorAST, context: Context): Result;
    abstract visitNestedDeclarator(node: ast.NestedDeclaratorAST, context: Context): Result;

    // PtrOperatorAST
    abstract visitPointerOperator(node: ast.PointerOperatorAST, context: Context): Result;
    abstract visitReferenceOperator(node: ast.ReferenceOperatorAST, context: Context): Result;
    abstract visitPtrToMemberOperator(node: ast.PtrToMemberOperatorAST, context: Context): Result;

    // DeclaratorModifierAST
    abstract visitFunctionDeclarator(node: ast.FunctionDeclaratorAST, context: Context): Result;
    abstract visitArrayDeclarator(node: ast.ArrayDeclaratorAST, context: Context): Result;
}

