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

import * as ast from "./AST";
import { ASTVisitor } from "./ASTVisitor";

/**
 * RecursiveASTVisitor.
 *
 * Base class for recursive AST visitors.
 */
export class RecursiveASTVisitor<Context> extends ASTVisitor<Context, void> {
  /**
   * Constructor.
   */
  constructor() {
    super();
  }

  /**
   * Visits a node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  accept(node: ast.AST | undefined, context: Context) {
    node?.accept(this, context);
  }

  /**
   * Visit a TranslationUnit node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTranslationUnit(node: ast.TranslationUnitAST, context: Context): void {
    for (const element of node.getDeclarationList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ModuleUnit node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitModuleUnit(node: ast.ModuleUnitAST, context: Context): void {
    this.accept(node.getGlobalModuleFragment(), context);
    this.accept(node.getModuleDeclaration(), context);
    for (const element of node.getDeclarationList()) {
      this.accept(element, context);
    }
    this.accept(node.getPrivateModuleFragment(), context);
  }

  /**
   * Visit a SimpleDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSimpleDeclaration(
    node: ast.SimpleDeclarationAST,
    context: Context,
  ): void {
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

  /**
   * Visit a AsmDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAsmDeclaration(node: ast.AsmDeclarationAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    for (const element of node.getAsmQualifierList()) {
      this.accept(element, context);
    }
    for (const element of node.getOutputOperandList()) {
      this.accept(element, context);
    }
    for (const element of node.getInputOperandList()) {
      this.accept(element, context);
    }
    for (const element of node.getClobberList()) {
      this.accept(element, context);
    }
    for (const element of node.getGotoLabelList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a NamespaceAliasDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNamespaceAliasDefinition(
    node: ast.NamespaceAliasDefinitionAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a UsingDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitUsingDeclaration(node: ast.UsingDeclarationAST, context: Context): void {
    for (const element of node.getUsingDeclaratorList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a UsingEnumDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitUsingEnumDeclaration(
    node: ast.UsingEnumDeclarationAST,
    context: Context,
  ): void {
    this.accept(node.getEnumTypeSpecifier(), context);
  }

  /**
   * Visit a UsingDirective node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitUsingDirective(node: ast.UsingDirectiveAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a StaticAssertDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitStaticAssertDeclaration(
    node: ast.StaticAssertDeclarationAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a AliasDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAliasDeclaration(node: ast.AliasDeclarationAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a OpaqueEnumDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitOpaqueEnumDeclaration(
    node: ast.OpaqueEnumDeclarationAST,
    context: Context,
  ): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
    for (const element of node.getTypeSpecifierList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a FunctionDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitFunctionDefinition(
    node: ast.FunctionDefinitionAST,
    context: Context,
  ): void {
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

  /**
   * Visit a TemplateDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTemplateDeclaration(
    node: ast.TemplateDeclarationAST,
    context: Context,
  ): void {
    for (const element of node.getTemplateParameterList()) {
      this.accept(element, context);
    }
    this.accept(node.getRequiresClause(), context);
    this.accept(node.getDeclaration(), context);
  }

  /**
   * Visit a ConceptDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConceptDefinition(
    node: ast.ConceptDefinitionAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a DeductionGuide node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDeductionGuide(node: ast.DeductionGuideAST, context: Context): void {
    this.accept(node.getExplicitSpecifier(), context);
    this.accept(node.getParameterDeclarationClause(), context);
    this.accept(node.getTemplateId(), context);
  }

  /**
   * Visit a ExplicitInstantiation node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitExplicitInstantiation(
    node: ast.ExplicitInstantiationAST,
    context: Context,
  ): void {
    this.accept(node.getDeclaration(), context);
  }

  /**
   * Visit a ExportDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitExportDeclaration(
    node: ast.ExportDeclarationAST,
    context: Context,
  ): void {
    this.accept(node.getDeclaration(), context);
  }

  /**
   * Visit a ExportCompoundDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitExportCompoundDeclaration(
    node: ast.ExportCompoundDeclarationAST,
    context: Context,
  ): void {
    for (const element of node.getDeclarationList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a LinkageSpecification node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitLinkageSpecification(
    node: ast.LinkageSpecificationAST,
    context: Context,
  ): void {
    for (const element of node.getDeclarationList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a NamespaceDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNamespaceDefinition(
    node: ast.NamespaceDefinitionAST,
    context: Context,
  ): void {
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

  /**
   * Visit a EmptyDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitEmptyDeclaration(
    node: ast.EmptyDeclarationAST,
    context: Context,
  ): void {}

  /**
   * Visit a AttributeDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAttributeDeclaration(
    node: ast.AttributeDeclarationAST,
    context: Context,
  ): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ModuleImportDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitModuleImportDeclaration(
    node: ast.ModuleImportDeclarationAST,
    context: Context,
  ): void {
    this.accept(node.getImportName(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ParameterDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitParameterDeclaration(
    node: ast.ParameterDeclarationAST,
    context: Context,
  ): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    for (const element of node.getTypeSpecifierList()) {
      this.accept(element, context);
    }
    this.accept(node.getDeclarator(), context);
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a AccessDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAccessDeclaration(
    node: ast.AccessDeclarationAST,
    context: Context,
  ): void {}

  /**
   * Visit a ForRangeDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitForRangeDeclaration(
    node: ast.ForRangeDeclarationAST,
    context: Context,
  ): void {}

  /**
   * Visit a StructuredBindingDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitStructuredBindingDeclaration(
    node: ast.StructuredBindingDeclarationAST,
    context: Context,
  ): void {
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

  /**
   * Visit a AsmOperand node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAsmOperand(node: ast.AsmOperandAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a AsmQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAsmQualifier(node: ast.AsmQualifierAST, context: Context): void {}

  /**
   * Visit a AsmClobber node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAsmClobber(node: ast.AsmClobberAST, context: Context): void {}

  /**
   * Visit a AsmGotoLabel node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAsmGotoLabel(node: ast.AsmGotoLabelAST, context: Context): void {}

  /**
   * Visit a LabeledStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitLabeledStatement(
    node: ast.LabeledStatementAST,
    context: Context,
  ): void {}

  /**
   * Visit a CaseStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCaseStatement(node: ast.CaseStatementAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a DefaultStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDefaultStatement(
    node: ast.DefaultStatementAST,
    context: Context,
  ): void {}

  /**
   * Visit a ExpressionStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitExpressionStatement(
    node: ast.ExpressionStatementAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a CompoundStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCompoundStatement(
    node: ast.CompoundStatementAST,
    context: Context,
  ): void {
    for (const element of node.getStatementList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a IfStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitIfStatement(node: ast.IfStatementAST, context: Context): void {
    this.accept(node.getInitializer(), context);
    this.accept(node.getCondition(), context);
    this.accept(node.getStatement(), context);
    this.accept(node.getElseStatement(), context);
  }

  /**
   * Visit a ConstevalIfStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConstevalIfStatement(
    node: ast.ConstevalIfStatementAST,
    context: Context,
  ): void {
    this.accept(node.getStatement(), context);
    this.accept(node.getElseStatement(), context);
  }

  /**
   * Visit a SwitchStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSwitchStatement(node: ast.SwitchStatementAST, context: Context): void {
    this.accept(node.getInitializer(), context);
    this.accept(node.getCondition(), context);
    this.accept(node.getStatement(), context);
  }

  /**
   * Visit a WhileStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitWhileStatement(node: ast.WhileStatementAST, context: Context): void {
    this.accept(node.getCondition(), context);
    this.accept(node.getStatement(), context);
  }

  /**
   * Visit a DoStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDoStatement(node: ast.DoStatementAST, context: Context): void {
    this.accept(node.getStatement(), context);
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a ForRangeStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitForRangeStatement(
    node: ast.ForRangeStatementAST,
    context: Context,
  ): void {
    this.accept(node.getInitializer(), context);
    this.accept(node.getRangeDeclaration(), context);
    this.accept(node.getRangeInitializer(), context);
    this.accept(node.getStatement(), context);
  }

  /**
   * Visit a ForStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitForStatement(node: ast.ForStatementAST, context: Context): void {
    this.accept(node.getInitializer(), context);
    this.accept(node.getCondition(), context);
    this.accept(node.getExpression(), context);
    this.accept(node.getStatement(), context);
  }

  /**
   * Visit a BreakStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBreakStatement(node: ast.BreakStatementAST, context: Context): void {}

  /**
   * Visit a ContinueStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitContinueStatement(
    node: ast.ContinueStatementAST,
    context: Context,
  ): void {}

  /**
   * Visit a ReturnStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitReturnStatement(node: ast.ReturnStatementAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a CoroutineReturnStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCoroutineReturnStatement(
    node: ast.CoroutineReturnStatementAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a GotoStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitGotoStatement(node: ast.GotoStatementAST, context: Context): void {}

  /**
   * Visit a DeclarationStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDeclarationStatement(
    node: ast.DeclarationStatementAST,
    context: Context,
  ): void {
    this.accept(node.getDeclaration(), context);
  }

  /**
   * Visit a TryBlockStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTryBlockStatement(
    node: ast.TryBlockStatementAST,
    context: Context,
  ): void {
    this.accept(node.getStatement(), context);
    for (const element of node.getHandlerList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a CharLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCharLiteralExpression(
    node: ast.CharLiteralExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a BoolLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBoolLiteralExpression(
    node: ast.BoolLiteralExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a IntLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitIntLiteralExpression(
    node: ast.IntLiteralExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a FloatLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitFloatLiteralExpression(
    node: ast.FloatLiteralExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a NullptrLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNullptrLiteralExpression(
    node: ast.NullptrLiteralExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a StringLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitStringLiteralExpression(
    node: ast.StringLiteralExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a UserDefinedStringLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitUserDefinedStringLiteralExpression(
    node: ast.UserDefinedStringLiteralExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a ThisExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitThisExpression(node: ast.ThisExpressionAST, context: Context): void {}

  /**
   * Visit a NestedExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNestedExpression(node: ast.NestedExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a IdExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitIdExpression(node: ast.IdExpressionAST, context: Context): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a LambdaExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitLambdaExpression(node: ast.LambdaExpressionAST, context: Context): void {
    for (const element of node.getCaptureList()) {
      this.accept(element, context);
    }
    for (const element of node.getTemplateParameterList()) {
      this.accept(element, context);
    }
    this.accept(node.getTemplateRequiresClause(), context);
    this.accept(node.getParameterDeclarationClause(), context);
    for (const element of node.getLambdaSpecifierList()) {
      this.accept(element, context);
    }
    this.accept(node.getExceptionSpecifier(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getTrailingReturnType(), context);
    this.accept(node.getRequiresClause(), context);
    this.accept(node.getStatement(), context);
  }

  /**
   * Visit a FoldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitFoldExpression(node: ast.FoldExpressionAST, context: Context): void {
    this.accept(node.getLeftExpression(), context);
    this.accept(node.getRightExpression(), context);
  }

  /**
   * Visit a RightFoldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitRightFoldExpression(
    node: ast.RightFoldExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a LeftFoldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitLeftFoldExpression(
    node: ast.LeftFoldExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a RequiresExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitRequiresExpression(
    node: ast.RequiresExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getParameterDeclarationClause(), context);
    for (const element of node.getRequirementList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a SubscriptExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSubscriptExpression(
    node: ast.SubscriptExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getBaseExpression(), context);
    this.accept(node.getIndexExpression(), context);
  }

  /**
   * Visit a CallExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCallExpression(node: ast.CallExpressionAST, context: Context): void {
    this.accept(node.getBaseExpression(), context);
    for (const element of node.getExpressionList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a TypeConstruction node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeConstruction(node: ast.TypeConstructionAST, context: Context): void {
    this.accept(node.getTypeSpecifier(), context);
    for (const element of node.getExpressionList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a BracedTypeConstruction node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBracedTypeConstruction(
    node: ast.BracedTypeConstructionAST,
    context: Context,
  ): void {
    this.accept(node.getTypeSpecifier(), context);
    this.accept(node.getBracedInitList(), context);
  }

  /**
   * Visit a MemberExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitMemberExpression(node: ast.MemberExpressionAST, context: Context): void {
    this.accept(node.getBaseExpression(), context);
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a PostIncrExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitPostIncrExpression(
    node: ast.PostIncrExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getBaseExpression(), context);
  }

  /**
   * Visit a CppCastExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCppCastExpression(
    node: ast.CppCastExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a BuiltinBitCastExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBuiltinBitCastExpression(
    node: ast.BuiltinBitCastExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a TypeidExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeidExpression(node: ast.TypeidExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a TypeidOfTypeExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeidOfTypeExpression(
    node: ast.TypeidOfTypeExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a UnaryExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitUnaryExpression(node: ast.UnaryExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a AwaitExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAwaitExpression(node: ast.AwaitExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a SizeofExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSizeofExpression(node: ast.SizeofExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a SizeofTypeExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSizeofTypeExpression(
    node: ast.SizeofTypeExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a SizeofPackExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSizeofPackExpression(
    node: ast.SizeofPackExpressionAST,
    context: Context,
  ): void {}

  /**
   * Visit a AlignofTypeExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAlignofTypeExpression(
    node: ast.AlignofTypeExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a AlignofExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAlignofExpression(
    node: ast.AlignofExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a NoexceptExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNoexceptExpression(
    node: ast.NoexceptExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a NewExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNewExpression(node: ast.NewExpressionAST, context: Context): void {
    this.accept(node.getNewPlacement(), context);
    for (const element of node.getTypeSpecifierList()) {
      this.accept(element, context);
    }
    this.accept(node.getDeclarator(), context);
    this.accept(node.getNewInitalizer(), context);
  }

  /**
   * Visit a DeleteExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDeleteExpression(node: ast.DeleteExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a CastExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCastExpression(node: ast.CastExpressionAST, context: Context): void {
    this.accept(node.getTypeId(), context);
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a ImplicitCastExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitImplicitCastExpression(
    node: ast.ImplicitCastExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a BinaryExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBinaryExpression(node: ast.BinaryExpressionAST, context: Context): void {
    this.accept(node.getLeftExpression(), context);
    this.accept(node.getRightExpression(), context);
  }

  /**
   * Visit a ConditionalExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConditionalExpression(
    node: ast.ConditionalExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getCondition(), context);
    this.accept(node.getIftrueExpression(), context);
    this.accept(node.getIffalseExpression(), context);
  }

  /**
   * Visit a YieldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitYieldExpression(node: ast.YieldExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a ThrowExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitThrowExpression(node: ast.ThrowExpressionAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a AssignmentExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAssignmentExpression(
    node: ast.AssignmentExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getLeftExpression(), context);
    this.accept(node.getRightExpression(), context);
  }

  /**
   * Visit a PackExpansionExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitPackExpansionExpression(
    node: ast.PackExpansionExpressionAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a DesignatedInitializerClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDesignatedInitializerClause(
    node: ast.DesignatedInitializerClauseAST,
    context: Context,
  ): void {
    this.accept(node.getInitializer(), context);
  }

  /**
   * Visit a TypeTraitsExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeTraitsExpression(
    node: ast.TypeTraitsExpressionAST,
    context: Context,
  ): void {
    for (const element of node.getTypeIdList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ConditionExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConditionExpression(
    node: ast.ConditionExpressionAST,
    context: Context,
  ): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    for (const element of node.getDeclSpecifierList()) {
      this.accept(element, context);
    }
    this.accept(node.getDeclarator(), context);
    this.accept(node.getInitializer(), context);
  }

  /**
   * Visit a EqualInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitEqualInitializer(node: ast.EqualInitializerAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a BracedInitList node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBracedInitList(node: ast.BracedInitListAST, context: Context): void {
    for (const element of node.getExpressionList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ParenInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitParenInitializer(node: ast.ParenInitializerAST, context: Context): void {
    for (const element of node.getExpressionList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a TemplateTypeParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTemplateTypeParameter(
    node: ast.TemplateTypeParameterAST,
    context: Context,
  ): void {
    for (const element of node.getTemplateParameterList()) {
      this.accept(element, context);
    }
    this.accept(node.getRequiresClause(), context);
    this.accept(node.getIdExpression(), context);
  }

  /**
   * Visit a NonTypeTemplateParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNonTypeTemplateParameter(
    node: ast.NonTypeTemplateParameterAST,
    context: Context,
  ): void {
    this.accept(node.getDeclaration(), context);
  }

  /**
   * Visit a TypenameTypeParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypenameTypeParameter(
    node: ast.TypenameTypeParameterAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a ConstraintTypeParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConstraintTypeParameter(
    node: ast.ConstraintTypeParameterAST,
    context: Context,
  ): void {
    this.accept(node.getTypeConstraint(), context);
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a TypedefSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypedefSpecifier(
    node: ast.TypedefSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a FriendSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitFriendSpecifier(node: ast.FriendSpecifierAST, context: Context): void {}

  /**
   * Visit a ConstevalSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConstevalSpecifier(
    node: ast.ConstevalSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a ConstinitSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConstinitSpecifier(
    node: ast.ConstinitSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a ConstexprSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConstexprSpecifier(
    node: ast.ConstexprSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a InlineSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitInlineSpecifier(node: ast.InlineSpecifierAST, context: Context): void {}

  /**
   * Visit a StaticSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitStaticSpecifier(node: ast.StaticSpecifierAST, context: Context): void {}

  /**
   * Visit a ExternSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitExternSpecifier(node: ast.ExternSpecifierAST, context: Context): void {}

  /**
   * Visit a ThreadLocalSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitThreadLocalSpecifier(
    node: ast.ThreadLocalSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a ThreadSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitThreadSpecifier(node: ast.ThreadSpecifierAST, context: Context): void {}

  /**
   * Visit a MutableSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitMutableSpecifier(
    node: ast.MutableSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a VirtualSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitVirtualSpecifier(
    node: ast.VirtualSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a ExplicitSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitExplicitSpecifier(
    node: ast.ExplicitSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a AutoTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAutoTypeSpecifier(
    node: ast.AutoTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a VoidTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitVoidTypeSpecifier(
    node: ast.VoidTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a SizeTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSizeTypeSpecifier(
    node: ast.SizeTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a SignTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSignTypeSpecifier(
    node: ast.SignTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a VaListTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitVaListTypeSpecifier(
    node: ast.VaListTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a IntegralTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitIntegralTypeSpecifier(
    node: ast.IntegralTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a FloatingPointTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitFloatingPointTypeSpecifier(
    node: ast.FloatingPointTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a ComplexTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitComplexTypeSpecifier(
    node: ast.ComplexTypeSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a NamedTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNamedTypeSpecifier(
    node: ast.NamedTypeSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a AtomicTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAtomicTypeSpecifier(
    node: ast.AtomicTypeSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a UnderlyingTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitUnderlyingTypeSpecifier(
    node: ast.UnderlyingTypeSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a ElaboratedTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitElaboratedTypeSpecifier(
    node: ast.ElaboratedTypeSpecifierAST,
    context: Context,
  ): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a DecltypeAutoSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDecltypeAutoSpecifier(
    node: ast.DecltypeAutoSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a DecltypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDecltypeSpecifier(
    node: ast.DecltypeSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a PlaceholderTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitPlaceholderTypeSpecifier(
    node: ast.PlaceholderTypeSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getTypeConstraint(), context);
    this.accept(node.getSpecifier(), context);
  }

  /**
   * Visit a ConstQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConstQualifier(node: ast.ConstQualifierAST, context: Context): void {}

  /**
   * Visit a VolatileQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitVolatileQualifier(
    node: ast.VolatileQualifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a RestrictQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitRestrictQualifier(
    node: ast.RestrictQualifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a EnumSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitEnumSpecifier(node: ast.EnumSpecifierAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
    for (const element of node.getTypeSpecifierList()) {
      this.accept(element, context);
    }
    for (const element of node.getEnumeratorList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ClassSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitClassSpecifier(node: ast.ClassSpecifierAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
    for (const element of node.getBaseSpecifierList()) {
      this.accept(element, context);
    }
    for (const element of node.getDeclarationList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a TypenameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypenameSpecifier(
    node: ast.TypenameSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a PointerOperator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitPointerOperator(node: ast.PointerOperatorAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    for (const element of node.getCvQualifierList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ReferenceOperator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitReferenceOperator(
    node: ast.ReferenceOperatorAST,
    context: Context,
  ): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a PtrToMemberOperator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitPtrToMemberOperator(
    node: ast.PtrToMemberOperatorAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    for (const element of node.getCvQualifierList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a BitfieldDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBitfieldDeclarator(
    node: ast.BitfieldDeclaratorAST,
    context: Context,
  ): void {
    this.accept(node.getUnqualifiedId(), context);
    this.accept(node.getSizeExpression(), context);
  }

  /**
   * Visit a ParameterPack node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitParameterPack(node: ast.ParameterPackAST, context: Context): void {
    this.accept(node.getCoreDeclarator(), context);
  }

  /**
   * Visit a IdDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitIdDeclarator(node: ast.IdDeclaratorAST, context: Context): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a NestedDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNestedDeclarator(node: ast.NestedDeclaratorAST, context: Context): void {
    this.accept(node.getDeclarator(), context);
  }

  /**
   * Visit a FunctionDeclaratorChunk node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitFunctionDeclaratorChunk(
    node: ast.FunctionDeclaratorChunkAST,
    context: Context,
  ): void {
    this.accept(node.getParameterDeclarationClause(), context);
    for (const element of node.getCvQualifierList()) {
      this.accept(element, context);
    }
    this.accept(node.getExceptionSpecifier(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getTrailingReturnType(), context);
  }

  /**
   * Visit a ArrayDeclaratorChunk node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitArrayDeclaratorChunk(
    node: ast.ArrayDeclaratorChunkAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a NameId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNameId(node: ast.NameIdAST, context: Context): void {}

  /**
   * Visit a DestructorId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDestructorId(node: ast.DestructorIdAST, context: Context): void {
    this.accept(node.getId(), context);
  }

  /**
   * Visit a DecltypeId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDecltypeId(node: ast.DecltypeIdAST, context: Context): void {
    this.accept(node.getDecltypeSpecifier(), context);
  }

  /**
   * Visit a OperatorFunctionId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitOperatorFunctionId(
    node: ast.OperatorFunctionIdAST,
    context: Context,
  ): void {}

  /**
   * Visit a LiteralOperatorId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitLiteralOperatorId(
    node: ast.LiteralOperatorIdAST,
    context: Context,
  ): void {}

  /**
   * Visit a ConversionFunctionId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitConversionFunctionId(
    node: ast.ConversionFunctionIdAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a SimpleTemplateId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSimpleTemplateId(node: ast.SimpleTemplateIdAST, context: Context): void {
    for (const element of node.getTemplateArgumentList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a LiteralOperatorTemplateId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitLiteralOperatorTemplateId(
    node: ast.LiteralOperatorTemplateIdAST,
    context: Context,
  ): void {
    this.accept(node.getLiteralOperatorId(), context);
    for (const element of node.getTemplateArgumentList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a OperatorFunctionTemplateId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitOperatorFunctionTemplateId(
    node: ast.OperatorFunctionTemplateIdAST,
    context: Context,
  ): void {
    this.accept(node.getOperatorFunctionId(), context);
    for (const element of node.getTemplateArgumentList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a GlobalNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitGlobalNestedNameSpecifier(
    node: ast.GlobalNestedNameSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a SimpleNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSimpleNestedNameSpecifier(
    node: ast.SimpleNestedNameSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
  }

  /**
   * Visit a DecltypeNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDecltypeNestedNameSpecifier(
    node: ast.DecltypeNestedNameSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getDecltypeSpecifier(), context);
  }

  /**
   * Visit a TemplateNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTemplateNestedNameSpecifier(
    node: ast.TemplateNestedNameSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getTemplateId(), context);
  }

  /**
   * Visit a DefaultFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDefaultFunctionBody(
    node: ast.DefaultFunctionBodyAST,
    context: Context,
  ): void {}

  /**
   * Visit a CompoundStatementFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCompoundStatementFunctionBody(
    node: ast.CompoundStatementFunctionBodyAST,
    context: Context,
  ): void {
    for (const element of node.getMemInitializerList()) {
      this.accept(element, context);
    }
    this.accept(node.getStatement(), context);
  }

  /**
   * Visit a TryStatementFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTryStatementFunctionBody(
    node: ast.TryStatementFunctionBodyAST,
    context: Context,
  ): void {
    for (const element of node.getMemInitializerList()) {
      this.accept(element, context);
    }
    this.accept(node.getStatement(), context);
    for (const element of node.getHandlerList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a DeleteFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDeleteFunctionBody(
    node: ast.DeleteFunctionBodyAST,
    context: Context,
  ): void {}

  /**
   * Visit a TypeTemplateArgument node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeTemplateArgument(
    node: ast.TypeTemplateArgumentAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a ExpressionTemplateArgument node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitExpressionTemplateArgument(
    node: ast.ExpressionTemplateArgumentAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a ThrowExceptionSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitThrowExceptionSpecifier(
    node: ast.ThrowExceptionSpecifierAST,
    context: Context,
  ): void {}

  /**
   * Visit a NoexceptSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNoexceptSpecifier(
    node: ast.NoexceptSpecifierAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a SimpleRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSimpleRequirement(
    node: ast.SimpleRequirementAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a CompoundRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCompoundRequirement(
    node: ast.CompoundRequirementAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
    this.accept(node.getTypeConstraint(), context);
  }

  /**
   * Visit a TypeRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeRequirement(node: ast.TypeRequirementAST, context: Context): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a NestedRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNestedRequirement(
    node: ast.NestedRequirementAST,
    context: Context,
  ): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a NewParenInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNewParenInitializer(
    node: ast.NewParenInitializerAST,
    context: Context,
  ): void {
    for (const element of node.getExpressionList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a NewBracedInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNewBracedInitializer(
    node: ast.NewBracedInitializerAST,
    context: Context,
  ): void {
    this.accept(node.getBracedInitList(), context);
  }

  /**
   * Visit a ParenMemInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitParenMemInitializer(
    node: ast.ParenMemInitializerAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
    for (const element of node.getExpressionList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a BracedMemInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBracedMemInitializer(
    node: ast.BracedMemInitializerAST,
    context: Context,
  ): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
    this.accept(node.getBracedInitList(), context);
  }

  /**
   * Visit a ThisLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitThisLambdaCapture(
    node: ast.ThisLambdaCaptureAST,
    context: Context,
  ): void {}

  /**
   * Visit a DerefThisLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDerefThisLambdaCapture(
    node: ast.DerefThisLambdaCaptureAST,
    context: Context,
  ): void {}

  /**
   * Visit a SimpleLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSimpleLambdaCapture(
    node: ast.SimpleLambdaCaptureAST,
    context: Context,
  ): void {}

  /**
   * Visit a RefLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitRefLambdaCapture(
    node: ast.RefLambdaCaptureAST,
    context: Context,
  ): void {}

  /**
   * Visit a RefInitLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitRefInitLambdaCapture(
    node: ast.RefInitLambdaCaptureAST,
    context: Context,
  ): void {
    this.accept(node.getInitializer(), context);
  }

  /**
   * Visit a InitLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitInitLambdaCapture(
    node: ast.InitLambdaCaptureAST,
    context: Context,
  ): void {
    this.accept(node.getInitializer(), context);
  }

  /**
   * Visit a EllipsisExceptionDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitEllipsisExceptionDeclaration(
    node: ast.EllipsisExceptionDeclarationAST,
    context: Context,
  ): void {}

  /**
   * Visit a TypeExceptionDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeExceptionDeclaration(
    node: ast.TypeExceptionDeclarationAST,
    context: Context,
  ): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    for (const element of node.getTypeSpecifierList()) {
      this.accept(element, context);
    }
    this.accept(node.getDeclarator(), context);
  }

  /**
   * Visit a CxxAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitCxxAttribute(node: ast.CxxAttributeAST, context: Context): void {
    this.accept(node.getAttributeUsingPrefix(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a GccAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitGccAttribute(node: ast.GccAttributeAST, context: Context): void {}

  /**
   * Visit a AlignasAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAlignasAttribute(node: ast.AlignasAttributeAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a AlignasTypeAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAlignasTypeAttribute(
    node: ast.AlignasTypeAttributeAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a AsmAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAsmAttribute(node: ast.AsmAttributeAST, context: Context): void {}

  /**
   * Visit a ScopedAttributeToken node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitScopedAttributeToken(
    node: ast.ScopedAttributeTokenAST,
    context: Context,
  ): void {}

  /**
   * Visit a SimpleAttributeToken node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitSimpleAttributeToken(
    node: ast.SimpleAttributeTokenAST,
    context: Context,
  ): void {}

  /**
   * Visit a GlobalModuleFragment node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitGlobalModuleFragment(
    node: ast.GlobalModuleFragmentAST,
    context: Context,
  ): void {
    for (const element of node.getDeclarationList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a PrivateModuleFragment node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitPrivateModuleFragment(
    node: ast.PrivateModuleFragmentAST,
    context: Context,
  ): void {
    for (const element of node.getDeclarationList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ModuleDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitModuleDeclaration(
    node: ast.ModuleDeclarationAST,
    context: Context,
  ): void {
    this.accept(node.getModuleName(), context);
    this.accept(node.getModulePartition(), context);
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a ModuleName node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitModuleName(node: ast.ModuleNameAST, context: Context): void {
    this.accept(node.getModuleQualifier(), context);
  }

  /**
   * Visit a ModuleQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitModuleQualifier(node: ast.ModuleQualifierAST, context: Context): void {
    this.accept(node.getModuleQualifier(), context);
  }

  /**
   * Visit a ModulePartition node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitModulePartition(node: ast.ModulePartitionAST, context: Context): void {
    this.accept(node.getModuleName(), context);
  }

  /**
   * Visit a ImportName node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitImportName(node: ast.ImportNameAST, context: Context): void {
    this.accept(node.getModulePartition(), context);
    this.accept(node.getModuleName(), context);
  }

  /**
   * Visit a InitDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitInitDeclarator(node: ast.InitDeclaratorAST, context: Context): void {
    this.accept(node.getDeclarator(), context);
    this.accept(node.getRequiresClause(), context);
    this.accept(node.getInitializer(), context);
  }

  /**
   * Visit a Declarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitDeclarator(node: ast.DeclaratorAST, context: Context): void {
    for (const element of node.getPtrOpList()) {
      this.accept(element, context);
    }
    this.accept(node.getCoreDeclarator(), context);
    for (const element of node.getDeclaratorChunkList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a UsingDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitUsingDeclarator(node: ast.UsingDeclaratorAST, context: Context): void {
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a Enumerator node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitEnumerator(node: ast.EnumeratorAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a TypeId node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeId(node: ast.TypeIdAST, context: Context): void {
    for (const element of node.getTypeSpecifierList()) {
      this.accept(element, context);
    }
    this.accept(node.getDeclarator(), context);
  }

  /**
   * Visit a Handler node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitHandler(node: ast.HandlerAST, context: Context): void {
    this.accept(node.getExceptionDeclaration(), context);
    this.accept(node.getStatement(), context);
  }

  /**
   * Visit a BaseSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitBaseSpecifier(node: ast.BaseSpecifierAST, context: Context): void {
    for (const element of node.getAttributeList()) {
      this.accept(element, context);
    }
    this.accept(node.getNestedNameSpecifier(), context);
    this.accept(node.getUnqualifiedId(), context);
  }

  /**
   * Visit a RequiresClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitRequiresClause(node: ast.RequiresClauseAST, context: Context): void {
    this.accept(node.getExpression(), context);
  }

  /**
   * Visit a ParameterDeclarationClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitParameterDeclarationClause(
    node: ast.ParameterDeclarationClauseAST,
    context: Context,
  ): void {
    for (const element of node.getParameterDeclarationList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a TrailingReturnType node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTrailingReturnType(
    node: ast.TrailingReturnTypeAST,
    context: Context,
  ): void {
    this.accept(node.getTypeId(), context);
  }

  /**
   * Visit a LambdaSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitLambdaSpecifier(node: ast.LambdaSpecifierAST, context: Context): void {}

  /**
   * Visit a TypeConstraint node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitTypeConstraint(node: ast.TypeConstraintAST, context: Context): void {
    this.accept(node.getNestedNameSpecifier(), context);
    for (const element of node.getTemplateArgumentList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a AttributeArgumentClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAttributeArgumentClause(
    node: ast.AttributeArgumentClauseAST,
    context: Context,
  ): void {}

  /**
   * Visit a Attribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAttribute(node: ast.AttributeAST, context: Context): void {
    this.accept(node.getAttributeToken(), context);
    this.accept(node.getAttributeArgumentClause(), context);
  }

  /**
   * Visit a AttributeUsingPrefix node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitAttributeUsingPrefix(
    node: ast.AttributeUsingPrefixAST,
    context: Context,
  ): void {}

  /**
   * Visit a NewPlacement node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNewPlacement(node: ast.NewPlacementAST, context: Context): void {
    for (const element of node.getExpressionList()) {
      this.accept(element, context);
    }
  }

  /**
   * Visit a NestedNamespaceSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   */
  visitNestedNamespaceSpecifier(
    node: ast.NestedNamespaceSpecifierAST,
    context: Context,
  ): void {}
}
