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

#pragma once

namespace cxx {

enum struct ASTKind {

  // AST
  TypeId,
  NestedNameSpecifier,
  UsingDeclarator,
  Handler,
  TemplateArgument,
  EnumBase,
  Enumerator,
  Declarator,
  BaseSpecifier,
  BaseClause,
  NewTypeId,
  ParameterDeclarationClause,
  ParametersAndQualifiers,
  LambdaIntroducer,
  LambdaDeclarator,
  TrailingReturnType,

  // InitializerAST
  EqualInitializer,
  BracedInitList,
  ParenInitializer,

  // NewInitializerAST
  NewParenInitializer,
  NewBracedInitializer,

  // ExceptionDeclarationAST
  EllipsisExceptionDeclaration,
  TypeExceptionDeclaration,

  // UnitAST
  TranslationUnit,
  ModuleUnit,

  // ExpressionAST
  ThisExpression,
  CharLiteralExpression,
  BoolLiteralExpression,
  IntLiteralExpression,
  FloatLiteralExpression,
  NullptrLiteralExpression,
  StringLiteralExpression,
  UserDefinedStringLiteralExpression,
  IdExpression,
  NestedExpression,
  LambdaExpression,
  UnaryExpression,
  BinaryExpression,
  AssignmentExpression,
  CallExpression,
  SubscriptExpression,
  MemberExpression,
  ConditionalExpression,
  CppCastExpression,
  NewExpression,

  // StatementAST
  LabeledStatement,
  CaseStatement,
  DefaultStatement,
  ExpressionStatement,
  CompoundStatement,
  IfStatement,
  SwitchStatement,
  WhileStatement,
  DoStatement,
  ForRangeStatement,
  ForStatement,
  BreakStatement,
  ContinueStatement,
  ReturnStatement,
  GotoStatement,
  CoroutineReturnStatement,
  DeclarationStatement,
  TryBlockStatement,

  // DeclarationAST
  FunctionDefinition,
  ConceptDefinition,
  ForRangeDeclaration,
  AliasDeclaration,
  SimpleDeclaration,
  StaticAssertDeclaration,
  EmptyDeclaration,
  AttributeDeclaration,
  OpaqueEnumDeclaration,
  UsingEnumDeclaration,
  NamespaceDefinition,
  NamespaceAliasDefinition,
  UsingDirective,
  UsingDeclaration,
  AsmDeclaration,
  ExportDeclaration,
  ModuleImportDeclaration,
  TemplateDeclaration,
  DeductionGuide,
  ExplicitInstantiation,
  ParameterDeclaration,
  LinkageSpecification,

  // NameAST
  SimpleName,
  DestructorName,
  DecltypeName,
  OperatorName,
  TemplateName,
  QualifiedName,

  // SpecifierAST
  TypedefSpecifier,
  FriendSpecifier,
  ConstevalSpecifier,
  ConstinitSpecifier,
  ConstexprSpecifier,
  InlineSpecifier,
  StaticSpecifier,
  ExternSpecifier,
  ThreadLocalSpecifier,
  ThreadSpecifier,
  MutableSpecifier,
  VirtualSpecifier,
  ExplicitSpecifier,
  AutoTypeSpecifier,
  VoidTypeSpecifier,
  IntegralTypeSpecifier,
  FloatingPointTypeSpecifier,
  ComplexTypeSpecifier,
  NamedTypeSpecifier,
  AtomicTypeSpecifier,
  UnderlyingTypeSpecifier,
  ElaboratedTypeSpecifier,
  DecltypeAutoSpecifier,
  DecltypeSpecifier,
  TypeofSpecifier,
  PlaceholderTypeSpecifier,
  CvQualifier,
  EnumSpecifier,
  ClassSpecifier,
  TypenameSpecifier,

  // CoreDeclaratorAST
  IdDeclarator,
  NestedDeclarator,

  // PtrOperatorAST
  PointerOperator,
  ReferenceOperator,
  PtrToMemberOperator,

  // DeclaratorModifierAST
  FunctionDeclarator,
  ArrayDeclarator,
};

}  // namespace cxx
