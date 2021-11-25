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
  EnumBase,
  Enumerator,
  Declarator,
  InitDeclarator,
  BaseSpecifier,
  BaseClause,
  NewTypeId,
  RequiresClause,
  ParameterDeclarationClause,
  ParametersAndQualifiers,
  LambdaIntroducer,
  LambdaDeclarator,
  TrailingReturnType,
  CtorInitializer,
  RequirementBody,
  TypeConstraint,
  GlobalModuleFragment,
  PrivateModuleFragment,
  ModuleDeclaration,
  ModuleName,
  ImportName,
  ModulePartition,

  // RequirementAST
  SimpleRequirement,
  CompoundRequirement,
  TypeRequirement,
  NestedRequirement,

  // TemplateArgumentAST
  TypeTemplateArgument,
  ExpressionTemplateArgument,

  // MemInitializerAST
  ParenMemInitializer,
  BracedMemInitializer,

  // LambdaCaptureAST
  ThisLambdaCapture,
  DerefThisLambdaCapture,
  SimpleLambdaCapture,
  RefLambdaCapture,
  RefInitLambdaCapture,
  InitLambdaCapture,

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

  // FunctionBodyAST
  DefaultFunctionBody,
  CompoundStatementFunctionBody,
  TryStatementFunctionBody,
  DeleteFunctionBody,

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
  RequiresExpression,
  NestedExpression,
  RightFoldExpression,
  LeftFoldExpression,
  FoldExpression,
  LambdaExpression,
  SizeofExpression,
  SizeofTypeExpression,
  SizeofPackExpression,
  TypeidExpression,
  TypeidOfTypeExpression,
  AlignofExpression,
  IsSameAsExpression,
  UnaryExpression,
  BinaryExpression,
  AssignmentExpression,
  BracedTypeConstruction,
  TypeConstruction,
  CallExpression,
  SubscriptExpression,
  MemberExpression,
  PostIncrExpression,
  ConditionalExpression,
  ImplicitCastExpression,
  CastExpression,
  CppCastExpression,
  NewExpression,
  DeleteExpression,
  ThrowExpression,
  NoexceptExpression,

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
  AccessDeclaration,
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
  ExportCompoundDeclaration,
  ModuleImportDeclaration,
  TemplateDeclaration,
  TypenameTypeParameter,
  TypenamePackTypeParameter,
  TemplateTypeParameter,
  TemplatePackTypeParameter,
  DeductionGuide,
  ExplicitInstantiation,
  ParameterDeclaration,
  LinkageSpecification,

  // NameAST
  SimpleName,
  DestructorName,
  DecltypeName,
  OperatorName,
  ConversionName,
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
  VaListTypeSpecifier,
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
  ConstQualifier,
  VolatileQualifier,
  RestrictQualifier,
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
