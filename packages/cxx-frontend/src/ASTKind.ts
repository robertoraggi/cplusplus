// Generated file by: gen_ast_kind_ts.ts
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

export enum ASTKind {
  // UnitAST
  TranslationUnit,
  ModuleUnit,

  // DeclarationAST
  SimpleDeclaration,
  AsmDeclaration,
  NamespaceAliasDefinition,
  UsingDeclaration,
  UsingEnumDeclaration,
  UsingDirective,
  StaticAssertDeclaration,
  AliasDeclaration,
  OpaqueEnumDeclaration,
  FunctionDefinition,
  TemplateDeclaration,
  ConceptDefinition,
  DeductionGuide,
  ExplicitInstantiation,
  ExportDeclaration,
  ExportCompoundDeclaration,
  LinkageSpecification,
  NamespaceDefinition,
  EmptyDeclaration,
  AttributeDeclaration,
  ModuleImportDeclaration,
  ParameterDeclaration,
  AccessDeclaration,
  ForRangeDeclaration,
  StructuredBindingDeclaration,

  // AST
  AsmOperand,
  AsmQualifier,
  AsmClobber,
  AsmGotoLabel,
  Splicer,
  GlobalModuleFragment,
  PrivateModuleFragment,
  ModuleDeclaration,
  ModuleName,
  ModuleQualifier,
  ModulePartition,
  ImportName,
  InitDeclarator,
  Declarator,
  UsingDeclarator,
  Enumerator,
  TypeId,
  Handler,
  BaseSpecifier,
  RequiresClause,
  ParameterDeclarationClause,
  TrailingReturnType,
  LambdaSpecifier,
  TypeConstraint,
  AttributeArgumentClause,
  Attribute,
  AttributeUsingPrefix,
  NewPlacement,
  NestedNamespaceSpecifier,

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
  CoroutineReturnStatement,
  GotoStatement,
  DeclarationStatement,
  TryBlockStatement,

  // ExpressionAST
  CharLiteralExpression,
  BoolLiteralExpression,
  IntLiteralExpression,
  FloatLiteralExpression,
  NullptrLiteralExpression,
  StringLiteralExpression,
  UserDefinedStringLiteralExpression,
  ObjectLiteralExpression,
  ThisExpression,
  GenericSelectionExpression,
  NestedStatementExpression,
  NestedExpression,
  IdExpression,
  LambdaExpression,
  FoldExpression,
  RightFoldExpression,
  LeftFoldExpression,
  RequiresExpression,
  VaArgExpression,
  SubscriptExpression,
  CallExpression,
  TypeConstruction,
  BracedTypeConstruction,
  SpliceMemberExpression,
  MemberExpression,
  PostIncrExpression,
  CppCastExpression,
  BuiltinBitCastExpression,
  BuiltinOffsetofExpression,
  TypeidExpression,
  TypeidOfTypeExpression,
  SpliceExpression,
  GlobalScopeReflectExpression,
  NamespaceReflectExpression,
  TypeIdReflectExpression,
  ReflectExpression,
  LabelAddressExpression,
  UnaryExpression,
  AwaitExpression,
  SizeofExpression,
  SizeofTypeExpression,
  SizeofPackExpression,
  AlignofTypeExpression,
  AlignofExpression,
  NoexceptExpression,
  NewExpression,
  DeleteExpression,
  CastExpression,
  ImplicitCastExpression,
  BinaryExpression,
  ConditionalExpression,
  YieldExpression,
  ThrowExpression,
  AssignmentExpression,
  CompoundAssignmentExpression,
  PackExpansionExpression,
  DesignatedInitializerClause,
  TypeTraitExpression,
  ConditionExpression,
  EqualInitializer,
  BracedInitList,
  ParenInitializer,

  // GenericAssociationAST
  DefaultGenericAssociation,
  TypeGenericAssociation,

  // DesignatorAST
  DotDesignator,
  SubscriptDesignator,

  // TemplateParameterAST
  TemplateTypeParameter,
  NonTypeTemplateParameter,
  TypenameTypeParameter,
  ConstraintTypeParameter,

  // SpecifierAST
  TypedefSpecifier,
  FriendSpecifier,
  ConstevalSpecifier,
  ConstinitSpecifier,
  ConstexprSpecifier,
  InlineSpecifier,
  NoreturnSpecifier,
  StaticSpecifier,
  ExternSpecifier,
  RegisterSpecifier,
  ThreadLocalSpecifier,
  ThreadSpecifier,
  MutableSpecifier,
  VirtualSpecifier,
  ExplicitSpecifier,
  AutoTypeSpecifier,
  VoidTypeSpecifier,
  SizeTypeSpecifier,
  SignTypeSpecifier,
  BuiltinTypeSpecifier,
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
  AtomicQualifier,
  RestrictQualifier,
  EnumSpecifier,
  ClassSpecifier,
  TypenameSpecifier,
  SplicerTypeSpecifier,

  // PtrOperatorAST
  PointerOperator,
  ReferenceOperator,
  PtrToMemberOperator,

  // CoreDeclaratorAST
  BitfieldDeclarator,
  ParameterPack,
  IdDeclarator,
  NestedDeclarator,

  // DeclaratorChunkAST
  FunctionDeclaratorChunk,
  ArrayDeclaratorChunk,

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

  // NestedNameSpecifierAST
  GlobalNestedNameSpecifier,
  SimpleNestedNameSpecifier,
  DecltypeNestedNameSpecifier,
  TemplateNestedNameSpecifier,

  // FunctionBodyAST
  DefaultFunctionBody,
  CompoundStatementFunctionBody,
  TryStatementFunctionBody,
  DeleteFunctionBody,

  // TemplateArgumentAST
  TypeTemplateArgument,
  ExpressionTemplateArgument,

  // ExceptionSpecifierAST
  ThrowExceptionSpecifier,
  NoexceptSpecifier,

  // RequirementAST
  SimpleRequirement,
  CompoundRequirement,
  TypeRequirement,
  NestedRequirement,

  // NewInitializerAST
  NewParenInitializer,
  NewBracedInitializer,

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

  // ExceptionDeclarationAST
  EllipsisExceptionDeclaration,
  TypeExceptionDeclaration,

  // AttributeSpecifierAST
  CxxAttribute,
  GccAttribute,
  AlignasAttribute,
  AlignasTypeAttribute,
  AsmAttribute,

  // AttributeTokenAST
  ScopedAttributeToken,
  SimpleAttributeToken,
}
