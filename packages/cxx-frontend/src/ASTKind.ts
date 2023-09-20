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

export enum ASTKind {
  // AST
  TypeId,
  UsingDeclarator,
  Handler,
  EnumBase,
  Enumerator,
  Declarator,
  InitDeclarator,
  BaseSpecifier,
  BaseClause,
  NewDeclarator,
  NewTypeId,
  RequiresClause,
  ParameterDeclarationClause,
  ParametersAndQualifiers,
  LambdaIntroducer,
  LambdaSpecifier,
  LambdaDeclarator,
  TrailingReturnType,
  CtorInitializer,
  RequirementBody,
  TypeConstraint,
  GlobalModuleFragment,
  PrivateModuleFragment,
  ModuleQualifier,
  ModuleName,
  ModuleDeclaration,
  ImportName,
  ModulePartition,
  AttributeArgumentClause,
  Attribute,
  AttributeUsingPrefix,
  Designator,
  NewPlacement,
  NestedNamespaceSpecifier,

  // NestedNameSpecifierAST
  GlobalNestedNameSpecifier,
  SimpleNestedNameSpecifier,
  DecltypeNestedNameSpecifier,
  TemplateNestedNameSpecifier,

  // ExceptionSpecifierAST
  ThrowExceptionSpecifier,
  NoexceptSpecifier,

  // ExpressionAST
  PackExpansionExpression,
  DesignatedInitializerClause,
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
  AlignofTypeExpression,
  AlignofExpression,
  TypeTraitsExpression,
  YieldExpression,
  AwaitExpression,
  UnaryExpression,
  BinaryExpression,
  AssignmentExpression,
  ConditionExpression,
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
  EqualInitializer,
  BracedInitList,
  ParenInitializer,

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

  // StatementAST
  LabeledStatement,
  CaseStatement,
  DefaultStatement,
  ExpressionStatement,
  CompoundStatement,
  IfStatement,
  ConstevalIfStatement,
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
  StructuredBindingDeclaration,
  StaticAssertDeclaration,
  EmptyDeclaration,
  AttributeDeclaration,
  OpaqueEnumDeclaration,
  NamespaceDefinition,
  NamespaceAliasDefinition,
  UsingDirective,
  UsingDeclaration,
  UsingEnumDeclaration,
  AsmDeclaration,
  ExportDeclaration,
  ExportCompoundDeclaration,
  ModuleImportDeclaration,
  TemplateDeclaration,
  TypenameTypeParameter,
  TemplateTypeParameter,
  TemplatePackTypeParameter,
  DeductionGuide,
  ExplicitInstantiation,
  ParameterDeclaration,
  LinkageSpecification,

  // UnqualifiedIdAST
  NameId,
  DestructorId,
  DecltypeId,
  OperatorFunctionId,
  LiteralOperatorId,
  ConversionFunctionId,
  SimpleTemplateId,
  LiteralOperatorTemplateId,
  OperatorFunctionTemplateId,

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
  SizeTypeSpecifier,
  SignTypeSpecifier,
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
  PlaceholderTypeSpecifier,
  ConstQualifier,
  VolatileQualifier,
  RestrictQualifier,
  EnumSpecifier,
  ClassSpecifier,
  TypenameSpecifier,

  // CoreDeclaratorAST
  BitfieldDeclarator,
  ParameterPack,
  IdDeclarator,
  NestedDeclarator,

  // PtrOperatorAST
  PointerOperator,
  ReferenceOperator,
  PtrToMemberOperator,

  // DeclaratorChunkAST
  FunctionDeclaratorChunk,
  ArrayDeclaratorChunk,

  // AttributeSpecifierAST
  CxxAttribute,
  GccAttribute,
  AlignasAttribute,
  AsmAttribute,

  // AttributeTokenAST
  ScopedAttributeToken,
  SimpleAttributeToken,
}
