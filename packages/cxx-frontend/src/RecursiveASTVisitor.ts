// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

import * as ast from "./AST.js";
import { ASTVisitor } from "./ASTVisitor.js";

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

    visitEnumBase(node: ast.EnumBaseAST, context: Context): void {
        for (const element of node.getTypeSpecifierList()) {
            this.accept(element, context);
        }
    }

    visitEnumerator(node: ast.EnumeratorAST, context: Context): void {
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

    visitInitDeclarator(node: ast.InitDeclaratorAST, context: Context): void {
        this.accept(node.getDeclarator(), context);
        this.accept(node.getRequiresClause(), context);
        this.accept(node.getInitializer(), context);
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

    visitRequiresClause(node: ast.RequiresClauseAST, context: Context): void {
        this.accept(node.getExpression(), context);
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
        for (const element of node.getCaptureList()) {
            this.accept(element, context);
        }
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
        this.accept(node.getRequiresClause(), context);
    }

    visitTrailingReturnType(node: ast.TrailingReturnTypeAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitCtorInitializer(node: ast.CtorInitializerAST, context: Context): void {
        for (const element of node.getMemInitializerList()) {
            this.accept(element, context);
        }
    }

    visitRequirementBody(node: ast.RequirementBodyAST, context: Context): void {
        for (const element of node.getRequirementList()) {
            this.accept(element, context);
        }
    }

    visitTypeConstraint(node: ast.TypeConstraintAST, context: Context): void {
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitGlobalModuleFragment(node: ast.GlobalModuleFragmentAST, context: Context): void {
        for (const element of node.getDeclarationList()) {
            this.accept(element, context);
        }
    }

    visitPrivateModuleFragment(node: ast.PrivateModuleFragmentAST, context: Context): void {
        for (const element of node.getDeclarationList()) {
            this.accept(element, context);
        }
    }

    visitModuleDeclaration(node: ast.ModuleDeclarationAST, context: Context): void {
        this.accept(node.getModuleName(), context);
        this.accept(node.getModulePartition(), context);
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
    }

    visitModuleName(node: ast.ModuleNameAST, context: Context): void {
    }

    visitImportName(node: ast.ImportNameAST, context: Context): void {
        this.accept(node.getModulePartition(), context);
        this.accept(node.getModuleName(), context);
    }

    visitModulePartition(node: ast.ModulePartitionAST, context: Context): void {
        this.accept(node.getModuleName(), context);
    }

    visitAttributeArgumentClause(node: ast.AttributeArgumentClauseAST, context: Context): void {
    }

    visitAttribute(node: ast.AttributeAST, context: Context): void {
        this.accept(node.getAttributeToken(), context);
        this.accept(node.getAttributeArgumentClause(), context);
    }

    visitAttributeUsingPrefix(node: ast.AttributeUsingPrefixAST, context: Context): void {
    }

    visitDesignator(node: ast.DesignatorAST, context: Context): void {
    }

    visitDesignatedInitializerClause(node: ast.DesignatedInitializerClauseAST, context: Context): void {
        this.accept(node.getDesignator(), context);
        this.accept(node.getInitializer(), context);
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

    visitRequiresExpression(node: ast.RequiresExpressionAST, context: Context): void {
        this.accept(node.getParameterDeclarationClause(), context);
        this.accept(node.getRequirementBody(), context);
    }

    visitNestedExpression(node: ast.NestedExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitRightFoldExpression(node: ast.RightFoldExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitLeftFoldExpression(node: ast.LeftFoldExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitFoldExpression(node: ast.FoldExpressionAST, context: Context): void {
        this.accept(node.getLeftExpression(), context);
        this.accept(node.getRightExpression(), context);
    }

    visitLambdaExpression(node: ast.LambdaExpressionAST, context: Context): void {
        this.accept(node.getLambdaIntroducer(), context);
        for (const element of node.getTemplateParameterList()) {
            this.accept(element, context);
        }
        this.accept(node.getRequiresClause(), context);
        this.accept(node.getLambdaDeclarator(), context);
        this.accept(node.getStatement(), context);
    }

    visitSizeofExpression(node: ast.SizeofExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitSizeofTypeExpression(node: ast.SizeofTypeExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitSizeofPackExpression(node: ast.SizeofPackExpressionAST, context: Context): void {
    }

    visitTypeidExpression(node: ast.TypeidExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitTypeidOfTypeExpression(node: ast.TypeidOfTypeExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitAlignofExpression(node: ast.AlignofExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitTypeTraitsExpression(node: ast.TypeTraitsExpressionAST, context: Context): void {
        for (const element of node.getTypeIdList()) {
            this.accept(element, context);
        }
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

    visitBracedTypeConstruction(node: ast.BracedTypeConstructionAST, context: Context): void {
        this.accept(node.getTypeSpecifier(), context);
        this.accept(node.getBracedInitList(), context);
    }

    visitTypeConstruction(node: ast.TypeConstructionAST, context: Context): void {
        this.accept(node.getTypeSpecifier(), context);
        for (const element of node.getExpressionList()) {
            this.accept(element, context);
        }
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

    visitPostIncrExpression(node: ast.PostIncrExpressionAST, context: Context): void {
        this.accept(node.getBaseExpression(), context);
    }

    visitConditionalExpression(node: ast.ConditionalExpressionAST, context: Context): void {
        this.accept(node.getCondition(), context);
        this.accept(node.getIftrueExpression(), context);
        this.accept(node.getIffalseExpression(), context);
    }

    visitImplicitCastExpression(node: ast.ImplicitCastExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitCastExpression(node: ast.CastExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
        this.accept(node.getExpression(), context);
    }

    visitCppCastExpression(node: ast.CppCastExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
        this.accept(node.getExpression(), context);
    }

    visitNewExpression(node: ast.NewExpressionAST, context: Context): void {
        this.accept(node.getTypeId(), context);
        this.accept(node.getNewInitalizer(), context);
    }

    visitDeleteExpression(node: ast.DeleteExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitThrowExpression(node: ast.ThrowExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitNoexceptExpression(node: ast.NoexceptExpressionAST, context: Context): void {
        this.accept(node.getExpression(), context);
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

    visitSimpleRequirement(node: ast.SimpleRequirementAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitCompoundRequirement(node: ast.CompoundRequirementAST, context: Context): void {
        this.accept(node.getExpression(), context);
        this.accept(node.getTypeConstraint(), context);
    }

    visitTypeRequirement(node: ast.TypeRequirementAST, context: Context): void {
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitNestedRequirement(node: ast.NestedRequirementAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitTypeTemplateArgument(node: ast.TypeTemplateArgumentAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitExpressionTemplateArgument(node: ast.ExpressionTemplateArgumentAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitParenMemInitializer(node: ast.ParenMemInitializerAST, context: Context): void {
        this.accept(node.getName(), context);
        for (const element of node.getExpressionList()) {
            this.accept(element, context);
        }
    }

    visitBracedMemInitializer(node: ast.BracedMemInitializerAST, context: Context): void {
        this.accept(node.getName(), context);
        this.accept(node.getBracedInitList(), context);
    }

    visitThisLambdaCapture(node: ast.ThisLambdaCaptureAST, context: Context): void {
    }

    visitDerefThisLambdaCapture(node: ast.DerefThisLambdaCaptureAST, context: Context): void {
    }

    visitSimpleLambdaCapture(node: ast.SimpleLambdaCaptureAST, context: Context): void {
    }

    visitRefLambdaCapture(node: ast.RefLambdaCaptureAST, context: Context): void {
    }

    visitRefInitLambdaCapture(node: ast.RefInitLambdaCaptureAST, context: Context): void {
        this.accept(node.getInitializer(), context);
    }

    visitInitLambdaCapture(node: ast.InitLambdaCaptureAST, context: Context): void {
        this.accept(node.getInitializer(), context);
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

    visitDefaultFunctionBody(node: ast.DefaultFunctionBodyAST, context: Context): void {
    }

    visitCompoundStatementFunctionBody(node: ast.CompoundStatementFunctionBodyAST, context: Context): void {
        this.accept(node.getCtorInitializer(), context);
        this.accept(node.getStatement(), context);
    }

    visitTryStatementFunctionBody(node: ast.TryStatementFunctionBodyAST, context: Context): void {
        this.accept(node.getCtorInitializer(), context);
        this.accept(node.getStatement(), context);
        for (const element of node.getHandlerList()) {
            this.accept(element, context);
        }
    }

    visitDeleteFunctionBody(node: ast.DeleteFunctionBodyAST, context: Context): void {
    }

    visitTranslationUnit(node: ast.TranslationUnitAST, context: Context): void {
        for (const element of node.getDeclarationList()) {
            this.accept(element, context);
        }
    }

    visitModuleUnit(node: ast.ModuleUnitAST, context: Context): void {
        this.accept(node.getGlobalModuleFragment(), context);
        this.accept(node.getModuleDeclaration(), context);
        for (const element of node.getDeclarationList()) {
            this.accept(element, context);
        }
        this.accept(node.getPrivateModuleFragment(), context);
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

    visitAccessDeclaration(node: ast.AccessDeclarationAST, context: Context): void {
    }

    visitFunctionDefinition(node: ast.FunctionDefinitionAST, context: Context): void {
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
        for (const element of node.getDeclSpecifierList()) {
            this.accept(element, context);
        }
        this.accept(node.getDeclarator(), context);
        this.accept(node.getRequiresClause(), context);
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
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
        for (const element of node.getDeclSpecifierList()) {
            this.accept(element, context);
        }
        for (const element of node.getInitDeclaratorList()) {
            this.accept(element, context);
        }
        this.accept(node.getRequiresClause(), context);
    }

    visitStructuredBindingDeclaration(node: ast.StructuredBindingDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
        for (const element of node.getDeclSpecifierList()) {
            this.accept(element, context);
        }
        for (const element of node.getBindingList()) {
            this.accept(element, context);
        }
        this.accept(node.getInitializer(), context);
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

    visitNestedNamespaceSpecifier(node: ast.NestedNamespaceSpecifierAST, context: Context): void {
    }

    visitNamespaceDefinition(node: ast.NamespaceDefinitionAST, context: Context): void {
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
        for (const element of node.getNestedNamespaceSpecifierList()) {
            this.accept(element, context);
        }
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
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitUsingDeclaration(node: ast.UsingDeclarationAST, context: Context): void {
        for (const element of node.getUsingDeclaratorList()) {
            this.accept(element, context);
        }
    }

    visitUsingEnumDeclaration(node: ast.UsingEnumDeclarationAST, context: Context): void {
        this.accept(node.getEnumTypeSpecifier(), context);
    }

    visitAsmDeclaration(node: ast.AsmDeclarationAST, context: Context): void {
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
    }

    visitExportDeclaration(node: ast.ExportDeclarationAST, context: Context): void {
        this.accept(node.getDeclaration(), context);
    }

    visitExportCompoundDeclaration(node: ast.ExportCompoundDeclarationAST, context: Context): void {
        for (const element of node.getDeclarationList()) {
            this.accept(element, context);
        }
    }

    visitModuleImportDeclaration(node: ast.ModuleImportDeclarationAST, context: Context): void {
        this.accept(node.getImportName(), context);
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
    }

    visitTemplateDeclaration(node: ast.TemplateDeclarationAST, context: Context): void {
        for (const element of node.getTemplateParameterList()) {
            this.accept(element, context);
        }
        this.accept(node.getRequiresClause(), context);
        this.accept(node.getDeclaration(), context);
    }

    visitTypenameTypeParameter(node: ast.TypenameTypeParameterAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitTemplateTypeParameter(node: ast.TemplateTypeParameterAST, context: Context): void {
        for (const element of node.getTemplateParameterList()) {
            this.accept(element, context);
        }
        this.accept(node.getRequiresClause(), context);
        this.accept(node.getName(), context);
    }

    visitTemplatePackTypeParameter(node: ast.TemplatePackTypeParameterAST, context: Context): void {
        for (const element of node.getTemplateParameterList()) {
            this.accept(element, context);
        }
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
        this.accept(node.getId(), context);
    }

    visitDecltypeName(node: ast.DecltypeNameAST, context: Context): void {
        this.accept(node.getDecltypeSpecifier(), context);
    }

    visitOperatorName(node: ast.OperatorNameAST, context: Context): void {
    }

    visitConversionName(node: ast.ConversionNameAST, context: Context): void {
        this.accept(node.getTypeId(), context);
    }

    visitTemplateName(node: ast.TemplateNameAST, context: Context): void {
        this.accept(node.getId(), context);
        for (const element of node.getTemplateArgumentList()) {
            this.accept(element, context);
        }
    }

    visitQualifiedName(node: ast.QualifiedNameAST, context: Context): void {
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getId(), context);
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

    visitVirtualSpecifier(node: ast.VirtualSpecifierAST, context: Context): void {
    }

    visitExplicitSpecifier(node: ast.ExplicitSpecifierAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitAutoTypeSpecifier(node: ast.AutoTypeSpecifierAST, context: Context): void {
    }

    visitVoidTypeSpecifier(node: ast.VoidTypeSpecifierAST, context: Context): void {
    }

    visitVaListTypeSpecifier(node: ast.VaListTypeSpecifierAST, context: Context): void {
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
        this.accept(node.getTypeId(), context);
    }

    visitElaboratedTypeSpecifier(node: ast.ElaboratedTypeSpecifierAST, context: Context): void {
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitDecltypeAutoSpecifier(node: ast.DecltypeAutoSpecifierAST, context: Context): void {
    }

    visitDecltypeSpecifier(node: ast.DecltypeSpecifierAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitPlaceholderTypeSpecifier(node: ast.PlaceholderTypeSpecifierAST, context: Context): void {
        this.accept(node.getTypeConstraint(), context);
        this.accept(node.getSpecifier(), context);
    }

    visitConstQualifier(node: ast.ConstQualifierAST, context: Context): void {
    }

    visitVolatileQualifier(node: ast.VolatileQualifierAST, context: Context): void {
    }

    visitRestrictQualifier(node: ast.RestrictQualifierAST, context: Context): void {
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
        this.accept(node.getNestedNameSpecifier(), context);
        this.accept(node.getName(), context);
    }

    visitBitfieldDeclarator(node: ast.BitfieldDeclaratorAST, context: Context): void {
        this.accept(node.getSizeExpression(), context);
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

    visitCxxAttribute(node: ast.CxxAttributeAST, context: Context): void {
        this.accept(node.getAttributeUsingPrefix(), context);
        for (const element of node.getAttributeList()) {
            this.accept(element, context);
        }
    }

    visitGCCAttribute(node: ast.GCCAttributeAST, context: Context): void {
    }

    visitAlignasAttribute(node: ast.AlignasAttributeAST, context: Context): void {
        this.accept(node.getExpression(), context);
    }

    visitAsmAttribute(node: ast.AsmAttributeAST, context: Context): void {
    }

    visitScopedAttributeToken(node: ast.ScopedAttributeTokenAST, context: Context): void {
    }

    visitSimpleAttributeToken(node: ast.SimpleAttributeTokenAST, context: Context): void {
    }
}

