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
import { ASTVisitor } from "./ASTVisitor";

export class RecursiveASTVisitor<Context> extends ASTVisitor<Context, void> {
    constructor() {
        super();
    }

    accept(node: ast.AST | undefined, context: Context) {
        node?.accept(this, context);
    }

    visitTypeId(node: ast.TypeIdAST, context: Context): void {
        for (const element of node.getTypeSpecifierList()) {
             this.accept(element, context);
        }
        this.accept(node.getDeclarator(), context);
    }

    visitNestedNameSpecifier(node: ast.NestedNameSpecifierAST, context: Context): void {
        for (const element of node.getNameList()) {
             this.accept(element, context);
        }
    }

    visitUsingDeclarator(node: ast.UsingDeclaratorAST, context: Context): void {
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitHandler(node: ast.HandlerAST, context: Context): void {
        this.accept(node.getExceptionDeclaration(), context);
        this.accept(node.getStatement(), context);
    }

    visitTemplateArgument(node: ast.TemplateArgumentAST, context: Context): void {
    }

    visitEnumBase(node: ast.EnumBaseAST, context: Context): void {
        for (const element of node.getTypeSpecifierList()) {
             this.accept(element, context);
        }
    }

    visitEnumerator(node: ast.EnumeratorAST, context: Context): void {
        this.accept(node.getName(), context);
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getExpression(), context);
    }

    visitDeclarator(node: ast.DeclaratorAST, context: Context): void {
        for (const element of node.getPtrOpList()) {
             this.accept(element, context);
        }
        this.accept(node.getCoreDeclarator(), context);
        for (const element of node.getModifiers()) {
             this.accept(element, context);
        }
    }

    visitBaseSpecifier(node: ast.BaseSpecifierAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getName(), context);
    }

    visitBaseClause(node: ast.BaseClauseAST, context: Context): void {
        for (const element of node.getBaseSpecifierList()) {
             this.accept(element, context);
        }
    }

    visitNewTypeId(node: ast.NewTypeIdAST, context: Context): void {
        for (const element of node.getTypeSpecifierList()) {
             this.accept(element, context);
        }
    }

    visitParameterDeclarationClause(node: ast.ParameterDeclarationClauseAST, context: Context): void {
        for (const element of node.getParameterDeclarationList()) {
             this.accept(element, context);
        }
    }

    visitParametersAndQualifiers(node: ast.ParametersAndQualifiersAST, context: Context): void {
        this.accept(node.getParameterDeclarationClause(), context);
        for (const element of node.getCvQualifierList()) {
             this.accept(element, context);
        }
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
    }

    visitLambdaIntroducer(node: ast.LambdaIntroducerAST, context: Context): void {
    }

    visitLambdaDeclarator(node: ast.LambdaDeclaratorAST, context: Context): void {
        this.accept(node.getParameterDeclarationClause(), context);
        for (const element of node.getDeclSpecifierList()) {
             this.accept(element, context);
        }
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getTrailingReturnType(), context);
    }

    visitTrailingReturnType(node: ast.TrailingReturnTypeAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitEqualInitializer(node: ast.EqualInitializerAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitBracedInitList(node: ast.BracedInitListAST, context: Context): void {
        for (const element of node.getExpressionList()) {
             this.accept(element, context);
        }
    }

    visitParenInitializer(node: ast.ParenInitializerAST, context: Context): void {
        for (const element of node.getExpressionList()) {
             this.accept(element, context);
        }
    }

    visitNewParenInitializer(node: ast.NewParenInitializerAST, context: Context): void {
        for (const element of node.getExpressionList()) {
             this.accept(element, context);
        }
    }

    visitNewBracedInitializer(node: ast.NewBracedInitializerAST, context: Context): void {
        this.accept(node.getBracedInit(), context);
    }

    visitEllipsisExceptionDeclaration(node: ast.EllipsisExceptionDeclarationAST, context: Context): void {
    }

    visitTypeExceptionDeclaration(node: ast.TypeExceptionDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        for (const element of node.getTypeSpecifierList()) {
             this.accept(element, context);
        }
        this.accept(node.getDeclarator(), context);
    }

    visitTranslationUnit(node: ast.TranslationUnitAST, context: Context): void {
        for (const element of node.getDeclarationList()) {
             this.accept(element, context);
        }
    }

    visitModuleUnit(node: ast.ModuleUnitAST, context: Context): void {
    }

    visitThisExpression(node: ast.ThisExpressionAST, context: Context): void {
    }

    visitCharLiteralExpression(node: ast.CharLiteralExpressionAST, context: Context): void {
    }

    visitBoolLiteralExpression(node: ast.BoolLiteralExpressionAST, context: Context): void {
    }

    visitIntLiteralExpression(node: ast.IntLiteralExpressionAST, context: Context): void {
    }

    visitFloatLiteralExpression(node: ast.FloatLiteralExpressionAST, context: Context): void {
    }

    visitNullptrLiteralExpression(node: ast.NullptrLiteralExpressionAST, context: Context): void {
    }

    visitStringLiteralExpression(node: ast.StringLiteralExpressionAST, context: Context): void {
    }

    visitUserDefinedStringLiteralExpression(node: ast.UserDefinedStringLiteralExpressionAST, context: Context): void {
    }

    visitIdExpression(node: ast.IdExpressionAST, context: Context): void {
        this.accept(node.getName(), context);
    }

    visitNestedExpression(node: ast.NestedExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitLambdaExpression(node: ast.LambdaExpressionAST, context: Context): void {
        this.accept(node.getLambdaIntroducer(), context);
        for (const element of node.getTemplateParameterList()) {
             this.accept(element, context);
        }
        this.accept(node.getLambdaDeclarator(), context);
        this.accept(node.getStatement(), context);
    }

    visitUnaryExpression(node: ast.UnaryExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitBinaryExpression(node: ast.BinaryExpressionAST, context: Context): void {
        this.accept(node.getLeftExpression(), context);
        this.accept(node.getRightExpression(), context);
    }

    visitAssignmentExpression(node: ast.AssignmentExpressionAST, context: Context): void {
        this.accept(node.getLeftExpression(), context);
        this.accept(node.getRightExpression(), context);
    }

    visitCallExpression(node: ast.CallExpressionAST, context: Context): void {
        this.accept(node.getBaseExpression(), context);
        for (const element of node.getExpressionList()) {
             this.accept(element, context);
        }
    }

    visitSubscriptExpression(node: ast.SubscriptExpressionAST, context: Context): void {
        this.accept(node.getBaseExpression(), context);
        this.accept(node.getIndexExpression(), context);
    }

    visitMemberExpression(node: ast.MemberExpressionAST, context: Context): void {
        this.accept(node.getBaseExpression(), context);
        this.accept(node.getName(), context);
    }

    visitConditionalExpression(node: ast.ConditionalExpressionAST, context: Context): void {
        this.accept(node.getCondition(), context);
        this.accept(node.getIftrueExpression(), context);
        this.accept(node.getIffalseExpression(), context);
    }

    visitCppCastExpression(node: ast.CppCastExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
        this.accept(node.getExpression(), context);
    }

    visitNewExpression(node: ast.NewExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
        this.accept(node.getNewInitalizer(), context);
    }

    visitLabeledStatement(node: ast.LabeledStatementAST, context: Context): void {
        this.accept(node.getStatement(), context);
    }

    visitCaseStatement(node: ast.CaseStatementAST, context: Context): void {
        this.accept(node.getExpression(), context);
        this.accept(node.getStatement(), context);
    }

    visitDefaultStatement(node: ast.DefaultStatementAST, context: Context): void {
        this.accept(node.getStatement(), context);
    }

    visitExpressionStatement(node: ast.ExpressionStatementAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitCompoundStatement(node: ast.CompoundStatementAST, context: Context): void {
        for (const element of node.getStatementList()) {
             this.accept(element, context);
        }
    }

    visitIfStatement(node: ast.IfStatementAST, context: Context): void {
        this.accept(node.getInitializer(), context);
        this.accept(node.getCondition(), context);
        this.accept(node.getStatement(), context);
        this.accept(node.getElseStatement(), context);
    }

    visitSwitchStatement(node: ast.SwitchStatementAST, context: Context): void {
        this.accept(node.getInitializer(), context);
        this.accept(node.getCondition(), context);
        this.accept(node.getStatement(), context);
    }

    visitWhileStatement(node: ast.WhileStatementAST, context: Context): void {
        this.accept(node.getCondition(), context);
        this.accept(node.getStatement(), context);
    }

    visitDoStatement(node: ast.DoStatementAST, context: Context): void {
        this.accept(node.getStatement(), context);
        this.accept(node.getExpression(), context);
    }

    visitForRangeStatement(node: ast.ForRangeStatementAST, context: Context): void {
        this.accept(node.getInitializer(), context);
        this.accept(node.getRangeDeclaration(), context);
        this.accept(node.getRangeInitializer(), context);
        this.accept(node.getStatement(), context);
    }

    visitForStatement(node: ast.ForStatementAST, context: Context): void {
        this.accept(node.getInitializer(), context);
        this.accept(node.getCondition(), context);
        this.accept(node.getExpression(), context);
        this.accept(node.getStatement(), context);
    }

    visitBreakStatement(node: ast.BreakStatementAST, context: Context): void {
    }

    visitContinueStatement(node: ast.ContinueStatementAST, context: Context): void {
    }

    visitReturnStatement(node: ast.ReturnStatementAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitGotoStatement(node: ast.GotoStatementAST, context: Context): void {
    }

    visitCoroutineReturnStatement(node: ast.CoroutineReturnStatementAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitDeclarationStatement(node: ast.DeclarationStatementAST, context: Context): void {
        this.accept(node.getDeclaration(), context);
    }

    visitTryBlockStatement(node: ast.TryBlockStatementAST, context: Context): void {
        this.accept(node.getStatement(), context);
        for (const element of node.getHandlerList()) {
             this.accept(element, context);
        }
    }

    visitFunctionDefinition(node: ast.FunctionDefinitionAST, context: Context): void {
        for (const element of node.getDeclSpecifierList()) {
             this.accept(element, context);
        }
        this.accept(node.getDeclarator(), context);
        this.accept(node.getFunctionBody(), context);
    }

    visitConceptDefinition(node: ast.ConceptDefinitionAST, context: Context): void {
        this.accept(node.getName(), context);
        this.accept(node.getExpression(), context);
    }

    visitForRangeDeclaration(node: ast.ForRangeDeclarationAST, context: Context): void {
    }

    visitAliasDeclaration(node: ast.AliasDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getTypeId(), context);
    }

    visitSimpleDeclaration(node: ast.SimpleDeclarationAST, context: Context): void {
        for (const element of node.getAttributes()) {
             this.accept(element, context);
        }
        for (const element of node.getDeclSpecifierList()) {
             this.accept(element, context);
        }
        for (const element of node.getDeclaratorList()) {
             this.accept(element, context);
        }
    }

    visitStaticAssertDeclaration(node: ast.StaticAssertDeclarationAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitEmptyDeclaration(node: ast.EmptyDeclarationAST, context: Context): void {
    }

    visitAttributeDeclaration(node: ast.AttributeDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
    }

    visitOpaqueEnumDeclaration(node: ast.OpaqueEnumDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
        this.accept(node.getEnumBase(), context);
    }

    visitUsingEnumDeclaration(node: ast.UsingEnumDeclarationAST, context: Context): void {
    }

    visitNamespaceDefinition(node: ast.NamespaceDefinitionAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
        for (const element of node.getExtraAttributeList()) {
             this.accept(element, context);
        }
        for (const element of node.getDeclarationList()) {
             this.accept(element, context);
        }
    }

    visitNamespaceAliasDefinition(node: ast.NamespaceAliasDefinitionAST, context: Context): void {
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitUsingDirective(node: ast.UsingDirectiveAST, context: Context): void {
    }

    visitUsingDeclaration(node: ast.UsingDeclarationAST, context: Context): void {
        for (const element of node.getUsingDeclaratorList()) {
             this.accept(element, context);
        }
    }

    visitAsmDeclaration(node: ast.AsmDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
    }

    visitExportDeclaration(node: ast.ExportDeclarationAST, context: Context): void {
    }

    visitModuleImportDeclaration(node: ast.ModuleImportDeclarationAST, context: Context): void {
    }

    visitTemplateDeclaration(node: ast.TemplateDeclarationAST, context: Context): void {
        for (const element of node.getTemplateParameterList()) {
             this.accept(element, context);
        }
        this.accept(node.getDeclaration(), context);
    }

    visitDeductionGuide(node: ast.DeductionGuideAST, context: Context): void {
    }

    visitExplicitInstantiation(node: ast.ExplicitInstantiationAST, context: Context): void {
        this.accept(node.getDeclaration(), context);
    }

    visitParameterDeclaration(node: ast.ParameterDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        for (const element of node.getTypeSpecifierList()) {
             this.accept(element, context);
        }
        this.accept(node.getDeclarator(), context);
        this.accept(node.getExpression(), context);
    }

    visitLinkageSpecification(node: ast.LinkageSpecificationAST, context: Context): void {
        for (const element of node.getDeclarationList()) {
             this.accept(element, context);
        }
    }

    visitSimpleName(node: ast.SimpleNameAST, context: Context): void {
    }

    visitDestructorName(node: ast.DestructorNameAST, context: Context): void {
        this.accept(node.getName(), context);
    }

    visitDecltypeName(node: ast.DecltypeNameAST, context: Context): void {
        this.accept(node.getDecltypeSpecifier(), context);
    }

    visitOperatorName(node: ast.OperatorNameAST, context: Context): void {
    }

    visitTemplateName(node: ast.TemplateNameAST, context: Context): void {
        this.accept(node.getName(), context);
        for (const element of node.getTemplateArgumentList()) {
             this.accept(element, context);
        }
    }

    visitQualifiedName(node: ast.QualifiedNameAST, context: Context): void {
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitTypedefSpecifier(node: ast.TypedefSpecifierAST, context: Context): void {
    }

    visitFriendSpecifier(node: ast.FriendSpecifierAST, context: Context): void {
    }

    visitConstevalSpecifier(node: ast.ConstevalSpecifierAST, context: Context): void {
    }

    visitConstinitSpecifier(node: ast.ConstinitSpecifierAST, context: Context): void {
    }

    visitConstexprSpecifier(node: ast.ConstexprSpecifierAST, context: Context): void {
    }

    visitInlineSpecifier(node: ast.InlineSpecifierAST, context: Context): void {
    }

    visitStaticSpecifier(node: ast.StaticSpecifierAST, context: Context): void {
    }

    visitExternSpecifier(node: ast.ExternSpecifierAST, context: Context): void {
    }

    visitThreadLocalSpecifier(node: ast.ThreadLocalSpecifierAST, context: Context): void {
    }

    visitThreadSpecifier(node: ast.ThreadSpecifierAST, context: Context): void {
    }

    visitMutableSpecifier(node: ast.MutableSpecifierAST, context: Context): void {
    }

    visitSimpleSpecifier(node: ast.SimpleSpecifierAST, context: Context): void {
    }

    visitExplicitSpecifier(node: ast.ExplicitSpecifierAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitAutoTypeSpecifier(node: ast.AutoTypeSpecifierAST, context: Context): void {
    }

    visitVoidTypeSpecifier(node: ast.VoidTypeSpecifierAST, context: Context): void {
    }

    visitIntegralTypeSpecifier(node: ast.IntegralTypeSpecifierAST, context: Context): void {
    }

    visitFloatingPointTypeSpecifier(node: ast.FloatingPointTypeSpecifierAST, context: Context): void {
    }

    visitComplexTypeSpecifier(node: ast.ComplexTypeSpecifierAST, context: Context): void {
    }

    visitNamedTypeSpecifier(node: ast.NamedTypeSpecifierAST, context: Context): void {
        this.accept(node.getName(), context);
    }

    visitAtomicTypeSpecifier(node: ast.AtomicTypeSpecifierAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitUnderlyingTypeSpecifier(node: ast.UnderlyingTypeSpecifierAST, context: Context): void {
    }

    visitElaboratedTypeSpecifier(node: ast.ElaboratedTypeSpecifierAST, context: Context): void {
    }

    visitDecltypeAutoSpecifier(node: ast.DecltypeAutoSpecifierAST, context: Context): void {
    }

    visitDecltypeSpecifier(node: ast.DecltypeSpecifierAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitTypeofSpecifier(node: ast.TypeofSpecifierAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitPlaceholderTypeSpecifier(node: ast.PlaceholderTypeSpecifierAST, context: Context): void {
    }

    visitCvQualifier(node: ast.CvQualifierAST, context: Context): void {
    }

    visitEnumSpecifier(node: ast.EnumSpecifierAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
        this.accept(node.getEnumBase(), context);
        for (const element of node.getEnumeratorList()) {
             this.accept(element, context);
        }
    }

    visitClassSpecifier(node: ast.ClassSpecifierAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        this.accept(node.getName(), context);
        this.accept(node.getBaseClause(), context);
        for (const element of node.getDeclarationList()) {
             this.accept(element, context);
        }
    }

    visitTypenameSpecifier(node: ast.TypenameSpecifierAST, context: Context): void {
    }

    visitIdDeclarator(node: ast.IdDeclaratorAST, context: Context): void {
        this.accept(node.getName(), context);
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
    }

    visitNestedDeclarator(node: ast.NestedDeclaratorAST, context: Context): void {
        this.accept(node.getDeclarator(), context);
    }

    visitPointerOperator(node: ast.PointerOperatorAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        for (const element of node.getCvQualifierList()) {
             this.accept(element, context);
        }
    }

    visitReferenceOperator(node: ast.ReferenceOperatorAST, context: Context): void {
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
    }

    visitPtrToMemberOperator(node: ast.PtrToMemberOperatorAST, context: Context): void {
        this.accept(node.getNestedNameSpecifier(), context);
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
        for (const element of node.getCvQualifierList()) {
             this.accept(element, context);
        }
    }

    visitFunctionDeclarator(node: ast.FunctionDeclaratorAST, context: Context): void {
        this.accept(node.getParametersAndQualifiers(), context);
        this.accept(node.getTrailingReturnType(), context);
    }

    visitArrayDeclarator(node: ast.ArrayDeclaratorAST, context: Context): void {
        this.accept(node.getExpression(), context);
        for (const element of node.getAttributeList()) {
             this.accept(element, context);
        }
    }
}

