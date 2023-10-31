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

/**
 * AST Visitor.
 *
 * Base class for all AST visitors.
 */
export abstract class ASTVisitor<Context, Result> {
  constructor() {}

  /**
   * Visit TranslationUnit node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTranslationUnit(
    node: ast.TranslationUnitAST,
    context: Context,
  ): Result;

  /**
   * Visit ModuleUnit node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitModuleUnit(node: ast.ModuleUnitAST, context: Context): Result;

  /**
   * Visit SimpleDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSimpleDeclaration(
    node: ast.SimpleDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit AsmDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAsmDeclaration(
    node: ast.AsmDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit NamespaceAliasDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNamespaceAliasDefinition(
    node: ast.NamespaceAliasDefinitionAST,
    context: Context,
  ): Result;

  /**
   * Visit UsingDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitUsingDeclaration(
    node: ast.UsingDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit UsingEnumDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitUsingEnumDeclaration(
    node: ast.UsingEnumDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit UsingDirective node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitUsingDirective(
    node: ast.UsingDirectiveAST,
    context: Context,
  ): Result;

  /**
   * Visit StaticAssertDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitStaticAssertDeclaration(
    node: ast.StaticAssertDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit AliasDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAliasDeclaration(
    node: ast.AliasDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit OpaqueEnumDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitOpaqueEnumDeclaration(
    node: ast.OpaqueEnumDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit FunctionDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitFunctionDefinition(
    node: ast.FunctionDefinitionAST,
    context: Context,
  ): Result;

  /**
   * Visit TemplateDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTemplateDeclaration(
    node: ast.TemplateDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit ConceptDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConceptDefinition(
    node: ast.ConceptDefinitionAST,
    context: Context,
  ): Result;

  /**
   * Visit DeductionGuide node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDeductionGuide(
    node: ast.DeductionGuideAST,
    context: Context,
  ): Result;

  /**
   * Visit ExplicitInstantiation node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitExplicitInstantiation(
    node: ast.ExplicitInstantiationAST,
    context: Context,
  ): Result;

  /**
   * Visit ExportDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitExportDeclaration(
    node: ast.ExportDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit ExportCompoundDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitExportCompoundDeclaration(
    node: ast.ExportCompoundDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit LinkageSpecification node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitLinkageSpecification(
    node: ast.LinkageSpecificationAST,
    context: Context,
  ): Result;

  /**
   * Visit NamespaceDefinition node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNamespaceDefinition(
    node: ast.NamespaceDefinitionAST,
    context: Context,
  ): Result;

  /**
   * Visit EmptyDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitEmptyDeclaration(
    node: ast.EmptyDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit AttributeDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAttributeDeclaration(
    node: ast.AttributeDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit ModuleImportDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitModuleImportDeclaration(
    node: ast.ModuleImportDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit ParameterDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitParameterDeclaration(
    node: ast.ParameterDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit AccessDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAccessDeclaration(
    node: ast.AccessDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit ForRangeDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitForRangeDeclaration(
    node: ast.ForRangeDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit StructuredBindingDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitStructuredBindingDeclaration(
    node: ast.StructuredBindingDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit AsmOperand node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAsmOperand(node: ast.AsmOperandAST, context: Context): Result;

  /**
   * Visit AsmQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAsmQualifier(
    node: ast.AsmQualifierAST,
    context: Context,
  ): Result;

  /**
   * Visit AsmClobber node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAsmClobber(node: ast.AsmClobberAST, context: Context): Result;

  /**
   * Visit AsmGotoLabel node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAsmGotoLabel(
    node: ast.AsmGotoLabelAST,
    context: Context,
  ): Result;

  /**
   * Visit LabeledStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitLabeledStatement(
    node: ast.LabeledStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit CaseStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCaseStatement(
    node: ast.CaseStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit DefaultStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDefaultStatement(
    node: ast.DefaultStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit ExpressionStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitExpressionStatement(
    node: ast.ExpressionStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit CompoundStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCompoundStatement(
    node: ast.CompoundStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit IfStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitIfStatement(node: ast.IfStatementAST, context: Context): Result;

  /**
   * Visit ConstevalIfStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConstevalIfStatement(
    node: ast.ConstevalIfStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit SwitchStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSwitchStatement(
    node: ast.SwitchStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit WhileStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitWhileStatement(
    node: ast.WhileStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit DoStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDoStatement(node: ast.DoStatementAST, context: Context): Result;

  /**
   * Visit ForRangeStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitForRangeStatement(
    node: ast.ForRangeStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit ForStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitForStatement(
    node: ast.ForStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit BreakStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBreakStatement(
    node: ast.BreakStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit ContinueStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitContinueStatement(
    node: ast.ContinueStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit ReturnStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitReturnStatement(
    node: ast.ReturnStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit CoroutineReturnStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCoroutineReturnStatement(
    node: ast.CoroutineReturnStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit GotoStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitGotoStatement(
    node: ast.GotoStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit DeclarationStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDeclarationStatement(
    node: ast.DeclarationStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit TryBlockStatement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTryBlockStatement(
    node: ast.TryBlockStatementAST,
    context: Context,
  ): Result;

  /**
   * Visit CharLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCharLiteralExpression(
    node: ast.CharLiteralExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit BoolLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBoolLiteralExpression(
    node: ast.BoolLiteralExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit IntLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitIntLiteralExpression(
    node: ast.IntLiteralExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit FloatLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitFloatLiteralExpression(
    node: ast.FloatLiteralExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit NullptrLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNullptrLiteralExpression(
    node: ast.NullptrLiteralExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit StringLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitStringLiteralExpression(
    node: ast.StringLiteralExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit UserDefinedStringLiteralExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitUserDefinedStringLiteralExpression(
    node: ast.UserDefinedStringLiteralExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit ThisExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitThisExpression(
    node: ast.ThisExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit NestedExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNestedExpression(
    node: ast.NestedExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit IdExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitIdExpression(
    node: ast.IdExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit LambdaExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitLambdaExpression(
    node: ast.LambdaExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit FoldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitFoldExpression(
    node: ast.FoldExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit RightFoldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitRightFoldExpression(
    node: ast.RightFoldExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit LeftFoldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitLeftFoldExpression(
    node: ast.LeftFoldExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit RequiresExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitRequiresExpression(
    node: ast.RequiresExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit SubscriptExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSubscriptExpression(
    node: ast.SubscriptExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit CallExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCallExpression(
    node: ast.CallExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeConstruction node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeConstruction(
    node: ast.TypeConstructionAST,
    context: Context,
  ): Result;

  /**
   * Visit BracedTypeConstruction node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBracedTypeConstruction(
    node: ast.BracedTypeConstructionAST,
    context: Context,
  ): Result;

  /**
   * Visit MemberExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitMemberExpression(
    node: ast.MemberExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit PostIncrExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitPostIncrExpression(
    node: ast.PostIncrExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit CppCastExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCppCastExpression(
    node: ast.CppCastExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeidExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeidExpression(
    node: ast.TypeidExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeidOfTypeExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeidOfTypeExpression(
    node: ast.TypeidOfTypeExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit UnaryExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitUnaryExpression(
    node: ast.UnaryExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit AwaitExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAwaitExpression(
    node: ast.AwaitExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit SizeofExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSizeofExpression(
    node: ast.SizeofExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit SizeofTypeExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSizeofTypeExpression(
    node: ast.SizeofTypeExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit SizeofPackExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSizeofPackExpression(
    node: ast.SizeofPackExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit AlignofTypeExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAlignofTypeExpression(
    node: ast.AlignofTypeExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit AlignofExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAlignofExpression(
    node: ast.AlignofExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit NoexceptExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNoexceptExpression(
    node: ast.NoexceptExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit NewExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNewExpression(
    node: ast.NewExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit DeleteExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDeleteExpression(
    node: ast.DeleteExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit CastExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCastExpression(
    node: ast.CastExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit ImplicitCastExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitImplicitCastExpression(
    node: ast.ImplicitCastExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit BinaryExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBinaryExpression(
    node: ast.BinaryExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit ConditionalExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConditionalExpression(
    node: ast.ConditionalExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit YieldExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitYieldExpression(
    node: ast.YieldExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit ThrowExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitThrowExpression(
    node: ast.ThrowExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit AssignmentExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAssignmentExpression(
    node: ast.AssignmentExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit PackExpansionExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitPackExpansionExpression(
    node: ast.PackExpansionExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit DesignatedInitializerClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDesignatedInitializerClause(
    node: ast.DesignatedInitializerClauseAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeTraitsExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeTraitsExpression(
    node: ast.TypeTraitsExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit ConditionExpression node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConditionExpression(
    node: ast.ConditionExpressionAST,
    context: Context,
  ): Result;

  /**
   * Visit EqualInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitEqualInitializer(
    node: ast.EqualInitializerAST,
    context: Context,
  ): Result;

  /**
   * Visit BracedInitList node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBracedInitList(
    node: ast.BracedInitListAST,
    context: Context,
  ): Result;

  /**
   * Visit ParenInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitParenInitializer(
    node: ast.ParenInitializerAST,
    context: Context,
  ): Result;

  /**
   * Visit TemplateTypeParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTemplateTypeParameter(
    node: ast.TemplateTypeParameterAST,
    context: Context,
  ): Result;

  /**
   * Visit NonTypeTemplateParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNonTypeTemplateParameter(
    node: ast.NonTypeTemplateParameterAST,
    context: Context,
  ): Result;

  /**
   * Visit TypenameTypeParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypenameTypeParameter(
    node: ast.TypenameTypeParameterAST,
    context: Context,
  ): Result;

  /**
   * Visit ConstraintTypeParameter node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConstraintTypeParameter(
    node: ast.ConstraintTypeParameterAST,
    context: Context,
  ): Result;

  /**
   * Visit TypedefSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypedefSpecifier(
    node: ast.TypedefSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit FriendSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitFriendSpecifier(
    node: ast.FriendSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ConstevalSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConstevalSpecifier(
    node: ast.ConstevalSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ConstinitSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConstinitSpecifier(
    node: ast.ConstinitSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ConstexprSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConstexprSpecifier(
    node: ast.ConstexprSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit InlineSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitInlineSpecifier(
    node: ast.InlineSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit StaticSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitStaticSpecifier(
    node: ast.StaticSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ExternSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitExternSpecifier(
    node: ast.ExternSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ThreadLocalSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitThreadLocalSpecifier(
    node: ast.ThreadLocalSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ThreadSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitThreadSpecifier(
    node: ast.ThreadSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit MutableSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitMutableSpecifier(
    node: ast.MutableSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit VirtualSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitVirtualSpecifier(
    node: ast.VirtualSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ExplicitSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitExplicitSpecifier(
    node: ast.ExplicitSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit AutoTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAutoTypeSpecifier(
    node: ast.AutoTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit VoidTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitVoidTypeSpecifier(
    node: ast.VoidTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit SizeTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSizeTypeSpecifier(
    node: ast.SizeTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit SignTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSignTypeSpecifier(
    node: ast.SignTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit VaListTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitVaListTypeSpecifier(
    node: ast.VaListTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit IntegralTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitIntegralTypeSpecifier(
    node: ast.IntegralTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit FloatingPointTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitFloatingPointTypeSpecifier(
    node: ast.FloatingPointTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ComplexTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitComplexTypeSpecifier(
    node: ast.ComplexTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit NamedTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNamedTypeSpecifier(
    node: ast.NamedTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit AtomicTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAtomicTypeSpecifier(
    node: ast.AtomicTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit UnderlyingTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitUnderlyingTypeSpecifier(
    node: ast.UnderlyingTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ElaboratedTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitElaboratedTypeSpecifier(
    node: ast.ElaboratedTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit DecltypeAutoSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDecltypeAutoSpecifier(
    node: ast.DecltypeAutoSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit DecltypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDecltypeSpecifier(
    node: ast.DecltypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit PlaceholderTypeSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitPlaceholderTypeSpecifier(
    node: ast.PlaceholderTypeSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ConstQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConstQualifier(
    node: ast.ConstQualifierAST,
    context: Context,
  ): Result;

  /**
   * Visit VolatileQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitVolatileQualifier(
    node: ast.VolatileQualifierAST,
    context: Context,
  ): Result;

  /**
   * Visit RestrictQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitRestrictQualifier(
    node: ast.RestrictQualifierAST,
    context: Context,
  ): Result;

  /**
   * Visit EnumSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitEnumSpecifier(
    node: ast.EnumSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ClassSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitClassSpecifier(
    node: ast.ClassSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit TypenameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypenameSpecifier(
    node: ast.TypenameSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit PointerOperator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitPointerOperator(
    node: ast.PointerOperatorAST,
    context: Context,
  ): Result;

  /**
   * Visit ReferenceOperator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitReferenceOperator(
    node: ast.ReferenceOperatorAST,
    context: Context,
  ): Result;

  /**
   * Visit PtrToMemberOperator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitPtrToMemberOperator(
    node: ast.PtrToMemberOperatorAST,
    context: Context,
  ): Result;

  /**
   * Visit BitfieldDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBitfieldDeclarator(
    node: ast.BitfieldDeclaratorAST,
    context: Context,
  ): Result;

  /**
   * Visit ParameterPack node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitParameterPack(
    node: ast.ParameterPackAST,
    context: Context,
  ): Result;

  /**
   * Visit IdDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitIdDeclarator(
    node: ast.IdDeclaratorAST,
    context: Context,
  ): Result;

  /**
   * Visit NestedDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNestedDeclarator(
    node: ast.NestedDeclaratorAST,
    context: Context,
  ): Result;

  /**
   * Visit FunctionDeclaratorChunk node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitFunctionDeclaratorChunk(
    node: ast.FunctionDeclaratorChunkAST,
    context: Context,
  ): Result;

  /**
   * Visit ArrayDeclaratorChunk node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitArrayDeclaratorChunk(
    node: ast.ArrayDeclaratorChunkAST,
    context: Context,
  ): Result;

  /**
   * Visit NameId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNameId(node: ast.NameIdAST, context: Context): Result;

  /**
   * Visit DestructorId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDestructorId(
    node: ast.DestructorIdAST,
    context: Context,
  ): Result;

  /**
   * Visit DecltypeId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDecltypeId(node: ast.DecltypeIdAST, context: Context): Result;

  /**
   * Visit OperatorFunctionId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitOperatorFunctionId(
    node: ast.OperatorFunctionIdAST,
    context: Context,
  ): Result;

  /**
   * Visit LiteralOperatorId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitLiteralOperatorId(
    node: ast.LiteralOperatorIdAST,
    context: Context,
  ): Result;

  /**
   * Visit ConversionFunctionId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitConversionFunctionId(
    node: ast.ConversionFunctionIdAST,
    context: Context,
  ): Result;

  /**
   * Visit SimpleTemplateId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSimpleTemplateId(
    node: ast.SimpleTemplateIdAST,
    context: Context,
  ): Result;

  /**
   * Visit LiteralOperatorTemplateId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitLiteralOperatorTemplateId(
    node: ast.LiteralOperatorTemplateIdAST,
    context: Context,
  ): Result;

  /**
   * Visit OperatorFunctionTemplateId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitOperatorFunctionTemplateId(
    node: ast.OperatorFunctionTemplateIdAST,
    context: Context,
  ): Result;

  /**
   * Visit GlobalNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitGlobalNestedNameSpecifier(
    node: ast.GlobalNestedNameSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit SimpleNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSimpleNestedNameSpecifier(
    node: ast.SimpleNestedNameSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit DecltypeNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDecltypeNestedNameSpecifier(
    node: ast.DecltypeNestedNameSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit TemplateNestedNameSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTemplateNestedNameSpecifier(
    node: ast.TemplateNestedNameSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit DefaultFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDefaultFunctionBody(
    node: ast.DefaultFunctionBodyAST,
    context: Context,
  ): Result;

  /**
   * Visit CompoundStatementFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCompoundStatementFunctionBody(
    node: ast.CompoundStatementFunctionBodyAST,
    context: Context,
  ): Result;

  /**
   * Visit TryStatementFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTryStatementFunctionBody(
    node: ast.TryStatementFunctionBodyAST,
    context: Context,
  ): Result;

  /**
   * Visit DeleteFunctionBody node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDeleteFunctionBody(
    node: ast.DeleteFunctionBodyAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeTemplateArgument node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeTemplateArgument(
    node: ast.TypeTemplateArgumentAST,
    context: Context,
  ): Result;

  /**
   * Visit ExpressionTemplateArgument node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitExpressionTemplateArgument(
    node: ast.ExpressionTemplateArgumentAST,
    context: Context,
  ): Result;

  /**
   * Visit ThrowExceptionSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitThrowExceptionSpecifier(
    node: ast.ThrowExceptionSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit NoexceptSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNoexceptSpecifier(
    node: ast.NoexceptSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit SimpleRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSimpleRequirement(
    node: ast.SimpleRequirementAST,
    context: Context,
  ): Result;

  /**
   * Visit CompoundRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCompoundRequirement(
    node: ast.CompoundRequirementAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeRequirement(
    node: ast.TypeRequirementAST,
    context: Context,
  ): Result;

  /**
   * Visit NestedRequirement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNestedRequirement(
    node: ast.NestedRequirementAST,
    context: Context,
  ): Result;

  /**
   * Visit NewParenInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNewParenInitializer(
    node: ast.NewParenInitializerAST,
    context: Context,
  ): Result;

  /**
   * Visit NewBracedInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNewBracedInitializer(
    node: ast.NewBracedInitializerAST,
    context: Context,
  ): Result;

  /**
   * Visit ParenMemInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitParenMemInitializer(
    node: ast.ParenMemInitializerAST,
    context: Context,
  ): Result;

  /**
   * Visit BracedMemInitializer node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBracedMemInitializer(
    node: ast.BracedMemInitializerAST,
    context: Context,
  ): Result;

  /**
   * Visit ThisLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitThisLambdaCapture(
    node: ast.ThisLambdaCaptureAST,
    context: Context,
  ): Result;

  /**
   * Visit DerefThisLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDerefThisLambdaCapture(
    node: ast.DerefThisLambdaCaptureAST,
    context: Context,
  ): Result;

  /**
   * Visit SimpleLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSimpleLambdaCapture(
    node: ast.SimpleLambdaCaptureAST,
    context: Context,
  ): Result;

  /**
   * Visit RefLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitRefLambdaCapture(
    node: ast.RefLambdaCaptureAST,
    context: Context,
  ): Result;

  /**
   * Visit RefInitLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitRefInitLambdaCapture(
    node: ast.RefInitLambdaCaptureAST,
    context: Context,
  ): Result;

  /**
   * Visit InitLambdaCapture node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitInitLambdaCapture(
    node: ast.InitLambdaCaptureAST,
    context: Context,
  ): Result;

  /**
   * Visit EllipsisExceptionDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitEllipsisExceptionDeclaration(
    node: ast.EllipsisExceptionDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeExceptionDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeExceptionDeclaration(
    node: ast.TypeExceptionDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit CxxAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitCxxAttribute(
    node: ast.CxxAttributeAST,
    context: Context,
  ): Result;

  /**
   * Visit GccAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitGccAttribute(
    node: ast.GccAttributeAST,
    context: Context,
  ): Result;

  /**
   * Visit AlignasAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAlignasAttribute(
    node: ast.AlignasAttributeAST,
    context: Context,
  ): Result;

  /**
   * Visit AlignasTypeAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAlignasTypeAttribute(
    node: ast.AlignasTypeAttributeAST,
    context: Context,
  ): Result;

  /**
   * Visit AsmAttribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAsmAttribute(
    node: ast.AsmAttributeAST,
    context: Context,
  ): Result;

  /**
   * Visit ScopedAttributeToken node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitScopedAttributeToken(
    node: ast.ScopedAttributeTokenAST,
    context: Context,
  ): Result;

  /**
   * Visit SimpleAttributeToken node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitSimpleAttributeToken(
    node: ast.SimpleAttributeTokenAST,
    context: Context,
  ): Result;

  /**
   * Visit GlobalModuleFragment node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitGlobalModuleFragment(
    node: ast.GlobalModuleFragmentAST,
    context: Context,
  ): Result;

  /**
   * Visit PrivateModuleFragment node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitPrivateModuleFragment(
    node: ast.PrivateModuleFragmentAST,
    context: Context,
  ): Result;

  /**
   * Visit ModuleDeclaration node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitModuleDeclaration(
    node: ast.ModuleDeclarationAST,
    context: Context,
  ): Result;

  /**
   * Visit ModuleName node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitModuleName(node: ast.ModuleNameAST, context: Context): Result;

  /**
   * Visit ModuleQualifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitModuleQualifier(
    node: ast.ModuleQualifierAST,
    context: Context,
  ): Result;

  /**
   * Visit ModulePartition node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitModulePartition(
    node: ast.ModulePartitionAST,
    context: Context,
  ): Result;

  /**
   * Visit ImportName node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitImportName(node: ast.ImportNameAST, context: Context): Result;

  /**
   * Visit InitDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitInitDeclarator(
    node: ast.InitDeclaratorAST,
    context: Context,
  ): Result;

  /**
   * Visit Declarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitDeclarator(node: ast.DeclaratorAST, context: Context): Result;

  /**
   * Visit UsingDeclarator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitUsingDeclarator(
    node: ast.UsingDeclaratorAST,
    context: Context,
  ): Result;

  /**
   * Visit Enumerator node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitEnumerator(node: ast.EnumeratorAST, context: Context): Result;

  /**
   * Visit TypeId node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeId(node: ast.TypeIdAST, context: Context): Result;

  /**
   * Visit Handler node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitHandler(node: ast.HandlerAST, context: Context): Result;

  /**
   * Visit BaseSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitBaseSpecifier(
    node: ast.BaseSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit RequiresClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitRequiresClause(
    node: ast.RequiresClauseAST,
    context: Context,
  ): Result;

  /**
   * Visit ParameterDeclarationClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitParameterDeclarationClause(
    node: ast.ParameterDeclarationClauseAST,
    context: Context,
  ): Result;

  /**
   * Visit TrailingReturnType node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTrailingReturnType(
    node: ast.TrailingReturnTypeAST,
    context: Context,
  ): Result;

  /**
   * Visit LambdaSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitLambdaSpecifier(
    node: ast.LambdaSpecifierAST,
    context: Context,
  ): Result;

  /**
   * Visit TypeConstraint node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitTypeConstraint(
    node: ast.TypeConstraintAST,
    context: Context,
  ): Result;

  /**
   * Visit AttributeArgumentClause node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAttributeArgumentClause(
    node: ast.AttributeArgumentClauseAST,
    context: Context,
  ): Result;

  /**
   * Visit Attribute node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAttribute(node: ast.AttributeAST, context: Context): Result;

  /**
   * Visit AttributeUsingPrefix node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitAttributeUsingPrefix(
    node: ast.AttributeUsingPrefixAST,
    context: Context,
  ): Result;

  /**
   * Visit NewPlacement node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNewPlacement(
    node: ast.NewPlacementAST,
    context: Context,
  ): Result;

  /**
   * Visit NestedNamespaceSpecifier node.
   *
   * @param node The node to visit.
   * @param context The context.
   * @returns The result of the visit.
   */
  abstract visitNestedNamespaceSpecifier(
    node: ast.NestedNamespaceSpecifierAST,
    context: Context,
  ): Result;
}
