// Generated file by: gen_traverse_ts.ts
// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
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

import { type ASTKind, children } from "./Semantic.js";
import type {
  AST,
  TranslationUnitAST,
  ModuleUnitAST,
  SimpleDeclarationAST,
  AsmDeclarationAST,
  NamespaceAliasDefinitionAST,
  UsingDeclarationAST,
  UsingEnumDeclarationAST,
  UsingDirectiveAST,
  StaticAssertDeclarationAST,
  AliasDeclarationAST,
  OpaqueEnumDeclarationAST,
  FunctionDefinitionAST,
  TemplateDeclarationAST,
  ConceptDefinitionAST,
  DeductionGuideAST,
  ExplicitInstantiationAST,
  ExportDeclarationAST,
  ExportCompoundDeclarationAST,
  LinkageSpecificationAST,
  NamespaceDefinitionAST,
  EmptyDeclarationAST,
  AttributeDeclarationAST,
  ModuleImportDeclarationAST,
  ParameterDeclarationAST,
  AccessDeclarationAST,
  ForRangeDeclarationAST,
  StructuredBindingDeclarationAST,
  AsmOperandAST,
  AsmQualifierAST,
  AsmClobberAST,
  AsmGotoLabelAST,
  SplicerAST,
  GlobalModuleFragmentAST,
  PrivateModuleFragmentAST,
  ModuleDeclarationAST,
  ModuleNameAST,
  ModuleQualifierAST,
  ModulePartitionAST,
  ImportNameAST,
  InitDeclaratorAST,
  DeclaratorAST,
  UsingDeclaratorAST,
  EnumeratorAST,
  TypeIdAST,
  HandlerAST,
  BaseSpecifierAST,
  RequiresClauseAST,
  ParameterDeclarationClauseAST,
  TrailingReturnTypeAST,
  LambdaSpecifierAST,
  TypeConstraintAST,
  AttributeArgumentClauseAST,
  AttributeAST,
  AttributeUsingPrefixAST,
  NewPlacementAST,
  NestedNamespaceSpecifierAST,
  LabeledStatementAST,
  CaseStatementAST,
  DefaultStatementAST,
  ExpressionStatementAST,
  CompoundStatementAST,
  IfStatementAST,
  ConstevalIfStatementAST,
  SwitchStatementAST,
  WhileStatementAST,
  DoStatementAST,
  ForRangeStatementAST,
  ForStatementAST,
  BreakStatementAST,
  ContinueStatementAST,
  ReturnStatementAST,
  CoroutineReturnStatementAST,
  GotoStatementAST,
  DeclarationStatementAST,
  TryBlockStatementAST,
  CharLiteralExpressionAST,
  BoolLiteralExpressionAST,
  IntLiteralExpressionAST,
  FloatLiteralExpressionAST,
  NullptrLiteralExpressionAST,
  StringLiteralExpressionAST,
  UserDefinedStringLiteralExpressionAST,
  ObjectLiteralExpressionAST,
  ThisExpressionAST,
  PackIndexExpressionAST,
  GenericSelectionExpressionAST,
  NestedStatementExpressionAST,
  DefaultInitializerExpressionAST,
  NestedExpressionAST,
  IdExpressionAST,
  LambdaExpressionAST,
  FoldExpressionAST,
  RightFoldExpressionAST,
  LeftFoldExpressionAST,
  RequiresExpressionAST,
  VaArgExpressionAST,
  SubscriptExpressionAST,
  CallExpressionAST,
  TypeConstructionAST,
  BracedTypeConstructionAST,
  SpliceMemberExpressionAST,
  MemberExpressionAST,
  PostIncrExpressionAST,
  CppCastExpressionAST,
  BuiltinBitCastExpressionAST,
  BuiltinOffsetofExpressionAST,
  TypeidExpressionAST,
  TypeidOfTypeExpressionAST,
  SpliceExpressionAST,
  GlobalScopeReflectExpressionAST,
  NamespaceReflectExpressionAST,
  TypeIdReflectExpressionAST,
  ReflectExpressionAST,
  LabelAddressExpressionAST,
  UnaryExpressionAST,
  AwaitExpressionAST,
  SizeofExpressionAST,
  SizeofTypeExpressionAST,
  SizeofPackExpressionAST,
  AlignofTypeExpressionAST,
  AlignofExpressionAST,
  NoexceptExpressionAST,
  NewExpressionAST,
  DeleteExpressionAST,
  CastExpressionAST,
  ImplicitCastExpressionAST,
  ConstExpressionAST,
  BinaryExpressionAST,
  ConditionalExpressionAST,
  YieldExpressionAST,
  ThrowExpressionAST,
  AssignmentExpressionAST,
  TargetExpressionAST,
  RightExpressionAST,
  CompoundAssignmentExpressionAST,
  PackExpansionExpressionAST,
  DesignatedInitializerClauseAST,
  TypeTraitExpressionAST,
  ConditionExpressionAST,
  EqualInitializerAST,
  BracedInitListAST,
  ParenInitializerAST,
  ThreeWayComparisonExpressionAST,
  DefaultGenericAssociationAST,
  TypeGenericAssociationAST,
  DotDesignatorAST,
  SubscriptDesignatorAST,
  TemplateTypeParameterAST,
  NonTypeTemplateParameterAST,
  TypenameTypeParameterAST,
  ConstraintTypeParameterAST,
  TypedefSpecifierAST,
  FriendSpecifierAST,
  ConstevalSpecifierAST,
  ConstinitSpecifierAST,
  ConstexprSpecifierAST,
  InlineSpecifierAST,
  NoreturnSpecifierAST,
  StaticSpecifierAST,
  ExternSpecifierAST,
  RegisterSpecifierAST,
  ThreadLocalSpecifierAST,
  ThreadSpecifierAST,
  MutableSpecifierAST,
  VirtualSpecifierAST,
  ExplicitSpecifierAST,
  AutoTypeSpecifierAST,
  VoidTypeSpecifierAST,
  SizeTypeSpecifierAST,
  SignTypeSpecifierAST,
  BuiltinTypeSpecifierAST,
  UnaryBuiltinTypeSpecifierAST,
  BinaryBuiltinTypeSpecifierAST,
  IntegralTypeSpecifierAST,
  FloatingPointTypeSpecifierAST,
  ComplexTypeSpecifierAST,
  NamedTypeSpecifierAST,
  AtomicTypeSpecifierAST,
  BitIntTypeSpecifierAST,
  UnderlyingTypeSpecifierAST,
  ElaboratedTypeSpecifierAST,
  DecltypeAutoSpecifierAST,
  DecltypeSpecifierAST,
  PlaceholderTypeSpecifierAST,
  ConstQualifierAST,
  VolatileQualifierAST,
  AtomicQualifierAST,
  RestrictQualifierAST,
  EnumSpecifierAST,
  ClassSpecifierAST,
  TypenameSpecifierAST,
  SplicerTypeSpecifierAST,
  PointerOperatorAST,
  ReferenceOperatorAST,
  PtrToMemberOperatorAST,
  BitfieldDeclaratorAST,
  ParameterPackAST,
  IdDeclaratorAST,
  NestedDeclaratorAST,
  FunctionDeclaratorChunkAST,
  ArrayDeclaratorChunkAST,
  NameIdAST,
  DestructorIdAST,
  DecltypeIdAST,
  OperatorFunctionIdAST,
  LiteralOperatorIdAST,
  ConversionFunctionIdAST,
  SimpleTemplateIdAST,
  LiteralOperatorTemplateIdAST,
  OperatorFunctionTemplateIdAST,
  GlobalNestedNameSpecifierAST,
  SimpleNestedNameSpecifierAST,
  DecltypeNestedNameSpecifierAST,
  TemplateNestedNameSpecifierAST,
  DefaultFunctionBodyAST,
  CompoundStatementFunctionBodyAST,
  TryStatementFunctionBodyAST,
  DeleteFunctionBodyAST,
  TypeTemplateArgumentAST,
  ExpressionTemplateArgumentAST,
  ThrowExceptionSpecifierAST,
  NoexceptSpecifierAST,
  SimpleRequirementAST,
  CompoundRequirementAST,
  TypeRequirementAST,
  NestedRequirementAST,
  NewParenInitializerAST,
  NewBracedInitializerAST,
  ParenMemInitializerAST,
  BracedMemInitializerAST,
  ThisLambdaCaptureAST,
  DerefThisLambdaCaptureAST,
  SimpleLambdaCaptureAST,
  RefLambdaCaptureAST,
  RefInitLambdaCaptureAST,
  InitLambdaCaptureAST,
  EllipsisExceptionDeclarationAST,
  TypeExceptionDeclarationAST,
  CxxAttributeAST,
  GccAttributeAST,
  AlignasAttributeAST,
  AlignasTypeAttributeAST,
  AsmAttributeAST,
  ScopedAttributeTokenAST,
  SimpleAttributeTokenAST,
  AttributeSpecifierAST,
  AttributeTokenAST,
  CoreDeclaratorAST,
  DeclarationAST,
  DeclaratorChunkAST,
  DesignatorAST,
  ExceptionDeclarationAST,
  ExceptionSpecifierAST,
  ExpressionAST,
  FunctionBodyAST,
  GenericAssociationAST,
  LambdaCaptureAST,
  MemInitializerAST,
  NestedNameSpecifierAST,
  NewInitializerAST,
  PtrOperatorAST,
  RequirementAST,
  SpecifierAST,
  StatementAST,
  TemplateArgumentAST,
  TemplateParameterAST,
  UnitAST,
  UnqualifiedIdAST,
} from "./Semantic.js";

export interface ASTNodes {
  TranslationUnit: TranslationUnitAST;
  ModuleUnit: ModuleUnitAST;
  SimpleDeclaration: SimpleDeclarationAST;
  AsmDeclaration: AsmDeclarationAST;
  NamespaceAliasDefinition: NamespaceAliasDefinitionAST;
  UsingDeclaration: UsingDeclarationAST;
  UsingEnumDeclaration: UsingEnumDeclarationAST;
  UsingDirective: UsingDirectiveAST;
  StaticAssertDeclaration: StaticAssertDeclarationAST;
  AliasDeclaration: AliasDeclarationAST;
  OpaqueEnumDeclaration: OpaqueEnumDeclarationAST;
  FunctionDefinition: FunctionDefinitionAST;
  TemplateDeclaration: TemplateDeclarationAST;
  ConceptDefinition: ConceptDefinitionAST;
  DeductionGuide: DeductionGuideAST;
  ExplicitInstantiation: ExplicitInstantiationAST;
  ExportDeclaration: ExportDeclarationAST;
  ExportCompoundDeclaration: ExportCompoundDeclarationAST;
  LinkageSpecification: LinkageSpecificationAST;
  NamespaceDefinition: NamespaceDefinitionAST;
  EmptyDeclaration: EmptyDeclarationAST;
  AttributeDeclaration: AttributeDeclarationAST;
  ModuleImportDeclaration: ModuleImportDeclarationAST;
  ParameterDeclaration: ParameterDeclarationAST;
  AccessDeclaration: AccessDeclarationAST;
  ForRangeDeclaration: ForRangeDeclarationAST;
  StructuredBindingDeclaration: StructuredBindingDeclarationAST;
  AsmOperand: AsmOperandAST;
  AsmQualifier: AsmQualifierAST;
  AsmClobber: AsmClobberAST;
  AsmGotoLabel: AsmGotoLabelAST;
  Splicer: SplicerAST;
  GlobalModuleFragment: GlobalModuleFragmentAST;
  PrivateModuleFragment: PrivateModuleFragmentAST;
  ModuleDeclaration: ModuleDeclarationAST;
  ModuleName: ModuleNameAST;
  ModuleQualifier: ModuleQualifierAST;
  ModulePartition: ModulePartitionAST;
  ImportName: ImportNameAST;
  InitDeclarator: InitDeclaratorAST;
  Declarator: DeclaratorAST;
  UsingDeclarator: UsingDeclaratorAST;
  Enumerator: EnumeratorAST;
  TypeId: TypeIdAST;
  Handler: HandlerAST;
  BaseSpecifier: BaseSpecifierAST;
  RequiresClause: RequiresClauseAST;
  ParameterDeclarationClause: ParameterDeclarationClauseAST;
  TrailingReturnType: TrailingReturnTypeAST;
  LambdaSpecifier: LambdaSpecifierAST;
  TypeConstraint: TypeConstraintAST;
  AttributeArgumentClause: AttributeArgumentClauseAST;
  Attribute: AttributeAST;
  AttributeUsingPrefix: AttributeUsingPrefixAST;
  NewPlacement: NewPlacementAST;
  NestedNamespaceSpecifier: NestedNamespaceSpecifierAST;
  LabeledStatement: LabeledStatementAST;
  CaseStatement: CaseStatementAST;
  DefaultStatement: DefaultStatementAST;
  ExpressionStatement: ExpressionStatementAST;
  CompoundStatement: CompoundStatementAST;
  IfStatement: IfStatementAST;
  ConstevalIfStatement: ConstevalIfStatementAST;
  SwitchStatement: SwitchStatementAST;
  WhileStatement: WhileStatementAST;
  DoStatement: DoStatementAST;
  ForRangeStatement: ForRangeStatementAST;
  ForStatement: ForStatementAST;
  BreakStatement: BreakStatementAST;
  ContinueStatement: ContinueStatementAST;
  ReturnStatement: ReturnStatementAST;
  CoroutineReturnStatement: CoroutineReturnStatementAST;
  GotoStatement: GotoStatementAST;
  DeclarationStatement: DeclarationStatementAST;
  TryBlockStatement: TryBlockStatementAST;
  CharLiteralExpression: CharLiteralExpressionAST;
  BoolLiteralExpression: BoolLiteralExpressionAST;
  IntLiteralExpression: IntLiteralExpressionAST;
  FloatLiteralExpression: FloatLiteralExpressionAST;
  NullptrLiteralExpression: NullptrLiteralExpressionAST;
  StringLiteralExpression: StringLiteralExpressionAST;
  UserDefinedStringLiteralExpression: UserDefinedStringLiteralExpressionAST;
  ObjectLiteralExpression: ObjectLiteralExpressionAST;
  ThisExpression: ThisExpressionAST;
  PackIndexExpression: PackIndexExpressionAST;
  GenericSelectionExpression: GenericSelectionExpressionAST;
  NestedStatementExpression: NestedStatementExpressionAST;
  DefaultInitializerExpression: DefaultInitializerExpressionAST;
  NestedExpression: NestedExpressionAST;
  IdExpression: IdExpressionAST;
  LambdaExpression: LambdaExpressionAST;
  FoldExpression: FoldExpressionAST;
  RightFoldExpression: RightFoldExpressionAST;
  LeftFoldExpression: LeftFoldExpressionAST;
  RequiresExpression: RequiresExpressionAST;
  VaArgExpression: VaArgExpressionAST;
  SubscriptExpression: SubscriptExpressionAST;
  CallExpression: CallExpressionAST;
  TypeConstruction: TypeConstructionAST;
  BracedTypeConstruction: BracedTypeConstructionAST;
  SpliceMemberExpression: SpliceMemberExpressionAST;
  MemberExpression: MemberExpressionAST;
  PostIncrExpression: PostIncrExpressionAST;
  CppCastExpression: CppCastExpressionAST;
  BuiltinBitCastExpression: BuiltinBitCastExpressionAST;
  BuiltinOffsetofExpression: BuiltinOffsetofExpressionAST;
  TypeidExpression: TypeidExpressionAST;
  TypeidOfTypeExpression: TypeidOfTypeExpressionAST;
  SpliceExpression: SpliceExpressionAST;
  GlobalScopeReflectExpression: GlobalScopeReflectExpressionAST;
  NamespaceReflectExpression: NamespaceReflectExpressionAST;
  TypeIdReflectExpression: TypeIdReflectExpressionAST;
  ReflectExpression: ReflectExpressionAST;
  LabelAddressExpression: LabelAddressExpressionAST;
  UnaryExpression: UnaryExpressionAST;
  AwaitExpression: AwaitExpressionAST;
  SizeofExpression: SizeofExpressionAST;
  SizeofTypeExpression: SizeofTypeExpressionAST;
  SizeofPackExpression: SizeofPackExpressionAST;
  AlignofTypeExpression: AlignofTypeExpressionAST;
  AlignofExpression: AlignofExpressionAST;
  NoexceptExpression: NoexceptExpressionAST;
  NewExpression: NewExpressionAST;
  DeleteExpression: DeleteExpressionAST;
  CastExpression: CastExpressionAST;
  ImplicitCastExpression: ImplicitCastExpressionAST;
  ConstExpression: ConstExpressionAST;
  BinaryExpression: BinaryExpressionAST;
  ConditionalExpression: ConditionalExpressionAST;
  YieldExpression: YieldExpressionAST;
  ThrowExpression: ThrowExpressionAST;
  AssignmentExpression: AssignmentExpressionAST;
  TargetExpression: TargetExpressionAST;
  RightExpression: RightExpressionAST;
  CompoundAssignmentExpression: CompoundAssignmentExpressionAST;
  PackExpansionExpression: PackExpansionExpressionAST;
  DesignatedInitializerClause: DesignatedInitializerClauseAST;
  TypeTraitExpression: TypeTraitExpressionAST;
  ConditionExpression: ConditionExpressionAST;
  EqualInitializer: EqualInitializerAST;
  BracedInitList: BracedInitListAST;
  ParenInitializer: ParenInitializerAST;
  ThreeWayComparisonExpression: ThreeWayComparisonExpressionAST;
  DefaultGenericAssociation: DefaultGenericAssociationAST;
  TypeGenericAssociation: TypeGenericAssociationAST;
  DotDesignator: DotDesignatorAST;
  SubscriptDesignator: SubscriptDesignatorAST;
  TemplateTypeParameter: TemplateTypeParameterAST;
  NonTypeTemplateParameter: NonTypeTemplateParameterAST;
  TypenameTypeParameter: TypenameTypeParameterAST;
  ConstraintTypeParameter: ConstraintTypeParameterAST;
  TypedefSpecifier: TypedefSpecifierAST;
  FriendSpecifier: FriendSpecifierAST;
  ConstevalSpecifier: ConstevalSpecifierAST;
  ConstinitSpecifier: ConstinitSpecifierAST;
  ConstexprSpecifier: ConstexprSpecifierAST;
  InlineSpecifier: InlineSpecifierAST;
  NoreturnSpecifier: NoreturnSpecifierAST;
  StaticSpecifier: StaticSpecifierAST;
  ExternSpecifier: ExternSpecifierAST;
  RegisterSpecifier: RegisterSpecifierAST;
  ThreadLocalSpecifier: ThreadLocalSpecifierAST;
  ThreadSpecifier: ThreadSpecifierAST;
  MutableSpecifier: MutableSpecifierAST;
  VirtualSpecifier: VirtualSpecifierAST;
  ExplicitSpecifier: ExplicitSpecifierAST;
  AutoTypeSpecifier: AutoTypeSpecifierAST;
  VoidTypeSpecifier: VoidTypeSpecifierAST;
  SizeTypeSpecifier: SizeTypeSpecifierAST;
  SignTypeSpecifier: SignTypeSpecifierAST;
  BuiltinTypeSpecifier: BuiltinTypeSpecifierAST;
  UnaryBuiltinTypeSpecifier: UnaryBuiltinTypeSpecifierAST;
  BinaryBuiltinTypeSpecifier: BinaryBuiltinTypeSpecifierAST;
  IntegralTypeSpecifier: IntegralTypeSpecifierAST;
  FloatingPointTypeSpecifier: FloatingPointTypeSpecifierAST;
  ComplexTypeSpecifier: ComplexTypeSpecifierAST;
  NamedTypeSpecifier: NamedTypeSpecifierAST;
  AtomicTypeSpecifier: AtomicTypeSpecifierAST;
  BitIntTypeSpecifier: BitIntTypeSpecifierAST;
  UnderlyingTypeSpecifier: UnderlyingTypeSpecifierAST;
  ElaboratedTypeSpecifier: ElaboratedTypeSpecifierAST;
  DecltypeAutoSpecifier: DecltypeAutoSpecifierAST;
  DecltypeSpecifier: DecltypeSpecifierAST;
  PlaceholderTypeSpecifier: PlaceholderTypeSpecifierAST;
  ConstQualifier: ConstQualifierAST;
  VolatileQualifier: VolatileQualifierAST;
  AtomicQualifier: AtomicQualifierAST;
  RestrictQualifier: RestrictQualifierAST;
  EnumSpecifier: EnumSpecifierAST;
  ClassSpecifier: ClassSpecifierAST;
  TypenameSpecifier: TypenameSpecifierAST;
  SplicerTypeSpecifier: SplicerTypeSpecifierAST;
  PointerOperator: PointerOperatorAST;
  ReferenceOperator: ReferenceOperatorAST;
  PtrToMemberOperator: PtrToMemberOperatorAST;
  BitfieldDeclarator: BitfieldDeclaratorAST;
  ParameterPack: ParameterPackAST;
  IdDeclarator: IdDeclaratorAST;
  NestedDeclarator: NestedDeclaratorAST;
  FunctionDeclaratorChunk: FunctionDeclaratorChunkAST;
  ArrayDeclaratorChunk: ArrayDeclaratorChunkAST;
  NameId: NameIdAST;
  DestructorId: DestructorIdAST;
  DecltypeId: DecltypeIdAST;
  OperatorFunctionId: OperatorFunctionIdAST;
  LiteralOperatorId: LiteralOperatorIdAST;
  ConversionFunctionId: ConversionFunctionIdAST;
  SimpleTemplateId: SimpleTemplateIdAST;
  LiteralOperatorTemplateId: LiteralOperatorTemplateIdAST;
  OperatorFunctionTemplateId: OperatorFunctionTemplateIdAST;
  GlobalNestedNameSpecifier: GlobalNestedNameSpecifierAST;
  SimpleNestedNameSpecifier: SimpleNestedNameSpecifierAST;
  DecltypeNestedNameSpecifier: DecltypeNestedNameSpecifierAST;
  TemplateNestedNameSpecifier: TemplateNestedNameSpecifierAST;
  DefaultFunctionBody: DefaultFunctionBodyAST;
  CompoundStatementFunctionBody: CompoundStatementFunctionBodyAST;
  TryStatementFunctionBody: TryStatementFunctionBodyAST;
  DeleteFunctionBody: DeleteFunctionBodyAST;
  TypeTemplateArgument: TypeTemplateArgumentAST;
  ExpressionTemplateArgument: ExpressionTemplateArgumentAST;
  ThrowExceptionSpecifier: ThrowExceptionSpecifierAST;
  NoexceptSpecifier: NoexceptSpecifierAST;
  SimpleRequirement: SimpleRequirementAST;
  CompoundRequirement: CompoundRequirementAST;
  TypeRequirement: TypeRequirementAST;
  NestedRequirement: NestedRequirementAST;
  NewParenInitializer: NewParenInitializerAST;
  NewBracedInitializer: NewBracedInitializerAST;
  ParenMemInitializer: ParenMemInitializerAST;
  BracedMemInitializer: BracedMemInitializerAST;
  ThisLambdaCapture: ThisLambdaCaptureAST;
  DerefThisLambdaCapture: DerefThisLambdaCaptureAST;
  SimpleLambdaCapture: SimpleLambdaCaptureAST;
  RefLambdaCapture: RefLambdaCaptureAST;
  RefInitLambdaCapture: RefInitLambdaCaptureAST;
  InitLambdaCapture: InitLambdaCaptureAST;
  EllipsisExceptionDeclaration: EllipsisExceptionDeclarationAST;
  TypeExceptionDeclaration: TypeExceptionDeclarationAST;
  CxxAttribute: CxxAttributeAST;
  GccAttribute: GccAttributeAST;
  AlignasAttribute: AlignasAttributeAST;
  AlignasTypeAttribute: AlignasTypeAttributeAST;
  AsmAttribute: AsmAttributeAST;
  ScopedAttributeToken: ScopedAttributeTokenAST;
  SimpleAttributeToken: SimpleAttributeTokenAST;
}

export interface ASTCategories {
  AttributeSpecifier: AttributeSpecifierAST;
  AttributeToken: AttributeTokenAST;
  CoreDeclarator: CoreDeclaratorAST;
  Declaration: DeclarationAST;
  DeclaratorChunk: DeclaratorChunkAST;
  Designator: DesignatorAST;
  ExceptionDeclaration: ExceptionDeclarationAST;
  ExceptionSpecifier: ExceptionSpecifierAST;
  Expression: ExpressionAST;
  FunctionBody: FunctionBodyAST;
  GenericAssociation: GenericAssociationAST;
  LambdaCapture: LambdaCaptureAST;
  MemInitializer: MemInitializerAST;
  NestedNameSpecifier: NestedNameSpecifierAST;
  NewInitializer: NewInitializerAST;
  PtrOperator: PtrOperatorAST;
  Requirement: RequirementAST;
  Specifier: SpecifierAST;
  Statement: StatementAST;
  TemplateArgument: TemplateArgumentAST;
  TemplateParameter: TemplateParameterAST;
  Unit: UnitAST;
  UnqualifiedId: UnqualifiedIdAST;
}

export type ASTCategory = keyof ASTCategories;

export type VisitorKey = ASTKind | ASTCategory;

const categoryOf: Partial<Record<ASTKind, ASTCategory>> = {};

for (const [category, kinds] of Object.entries({
  Unit: ["TranslationUnit", "ModuleUnit"],
  Declaration: [
    "SimpleDeclaration",
    "AsmDeclaration",
    "NamespaceAliasDefinition",
    "UsingDeclaration",
    "UsingEnumDeclaration",
    "UsingDirective",
    "StaticAssertDeclaration",
    "AliasDeclaration",
    "OpaqueEnumDeclaration",
    "FunctionDefinition",
    "TemplateDeclaration",
    "ConceptDefinition",
    "DeductionGuide",
    "ExplicitInstantiation",
    "ExportDeclaration",
    "ExportCompoundDeclaration",
    "LinkageSpecification",
    "NamespaceDefinition",
    "EmptyDeclaration",
    "AttributeDeclaration",
    "ModuleImportDeclaration",
    "ParameterDeclaration",
    "AccessDeclaration",
    "ForRangeDeclaration",
    "StructuredBindingDeclaration",
  ],
  Statement: [
    "LabeledStatement",
    "CaseStatement",
    "DefaultStatement",
    "ExpressionStatement",
    "CompoundStatement",
    "IfStatement",
    "ConstevalIfStatement",
    "SwitchStatement",
    "WhileStatement",
    "DoStatement",
    "ForRangeStatement",
    "ForStatement",
    "BreakStatement",
    "ContinueStatement",
    "ReturnStatement",
    "CoroutineReturnStatement",
    "GotoStatement",
    "DeclarationStatement",
    "TryBlockStatement",
  ],
  Expression: [
    "CharLiteralExpression",
    "BoolLiteralExpression",
    "IntLiteralExpression",
    "FloatLiteralExpression",
    "NullptrLiteralExpression",
    "StringLiteralExpression",
    "UserDefinedStringLiteralExpression",
    "ObjectLiteralExpression",
    "ThisExpression",
    "PackIndexExpression",
    "GenericSelectionExpression",
    "NestedStatementExpression",
    "DefaultInitializerExpression",
    "NestedExpression",
    "IdExpression",
    "LambdaExpression",
    "FoldExpression",
    "RightFoldExpression",
    "LeftFoldExpression",
    "RequiresExpression",
    "VaArgExpression",
    "SubscriptExpression",
    "CallExpression",
    "TypeConstruction",
    "BracedTypeConstruction",
    "SpliceMemberExpression",
    "MemberExpression",
    "PostIncrExpression",
    "CppCastExpression",
    "BuiltinBitCastExpression",
    "BuiltinOffsetofExpression",
    "TypeidExpression",
    "TypeidOfTypeExpression",
    "SpliceExpression",
    "GlobalScopeReflectExpression",
    "NamespaceReflectExpression",
    "TypeIdReflectExpression",
    "ReflectExpression",
    "LabelAddressExpression",
    "UnaryExpression",
    "AwaitExpression",
    "SizeofExpression",
    "SizeofTypeExpression",
    "SizeofPackExpression",
    "AlignofTypeExpression",
    "AlignofExpression",
    "NoexceptExpression",
    "NewExpression",
    "DeleteExpression",
    "CastExpression",
    "ImplicitCastExpression",
    "ConstExpression",
    "BinaryExpression",
    "ConditionalExpression",
    "YieldExpression",
    "ThrowExpression",
    "AssignmentExpression",
    "TargetExpression",
    "RightExpression",
    "CompoundAssignmentExpression",
    "PackExpansionExpression",
    "DesignatedInitializerClause",
    "TypeTraitExpression",
    "ConditionExpression",
    "EqualInitializer",
    "BracedInitList",
    "ParenInitializer",
    "ThreeWayComparisonExpression",
  ],
  GenericAssociation: ["DefaultGenericAssociation", "TypeGenericAssociation"],
  Designator: ["DotDesignator", "SubscriptDesignator"],
  TemplateParameter: [
    "TemplateTypeParameter",
    "NonTypeTemplateParameter",
    "TypenameTypeParameter",
    "ConstraintTypeParameter",
  ],
  Specifier: [
    "TypedefSpecifier",
    "FriendSpecifier",
    "ConstevalSpecifier",
    "ConstinitSpecifier",
    "ConstexprSpecifier",
    "InlineSpecifier",
    "NoreturnSpecifier",
    "StaticSpecifier",
    "ExternSpecifier",
    "RegisterSpecifier",
    "ThreadLocalSpecifier",
    "ThreadSpecifier",
    "MutableSpecifier",
    "VirtualSpecifier",
    "ExplicitSpecifier",
    "AutoTypeSpecifier",
    "VoidTypeSpecifier",
    "SizeTypeSpecifier",
    "SignTypeSpecifier",
    "BuiltinTypeSpecifier",
    "UnaryBuiltinTypeSpecifier",
    "BinaryBuiltinTypeSpecifier",
    "IntegralTypeSpecifier",
    "FloatingPointTypeSpecifier",
    "ComplexTypeSpecifier",
    "NamedTypeSpecifier",
    "AtomicTypeSpecifier",
    "BitIntTypeSpecifier",
    "UnderlyingTypeSpecifier",
    "ElaboratedTypeSpecifier",
    "DecltypeAutoSpecifier",
    "DecltypeSpecifier",
    "PlaceholderTypeSpecifier",
    "ConstQualifier",
    "VolatileQualifier",
    "AtomicQualifier",
    "RestrictQualifier",
    "EnumSpecifier",
    "ClassSpecifier",
    "TypenameSpecifier",
    "SplicerTypeSpecifier",
  ],
  PtrOperator: ["PointerOperator", "ReferenceOperator", "PtrToMemberOperator"],
  CoreDeclarator: [
    "BitfieldDeclarator",
    "ParameterPack",
    "IdDeclarator",
    "NestedDeclarator",
  ],
  DeclaratorChunk: ["FunctionDeclaratorChunk", "ArrayDeclaratorChunk"],
  UnqualifiedId: [
    "NameId",
    "DestructorId",
    "DecltypeId",
    "OperatorFunctionId",
    "LiteralOperatorId",
    "ConversionFunctionId",
    "SimpleTemplateId",
    "LiteralOperatorTemplateId",
    "OperatorFunctionTemplateId",
  ],
  NestedNameSpecifier: [
    "GlobalNestedNameSpecifier",
    "SimpleNestedNameSpecifier",
    "DecltypeNestedNameSpecifier",
    "TemplateNestedNameSpecifier",
  ],
  FunctionBody: [
    "DefaultFunctionBody",
    "CompoundStatementFunctionBody",
    "TryStatementFunctionBody",
    "DeleteFunctionBody",
  ],
  TemplateArgument: ["TypeTemplateArgument", "ExpressionTemplateArgument"],
  ExceptionSpecifier: ["ThrowExceptionSpecifier", "NoexceptSpecifier"],
  Requirement: [
    "SimpleRequirement",
    "CompoundRequirement",
    "TypeRequirement",
    "NestedRequirement",
  ],
  NewInitializer: ["NewParenInitializer", "NewBracedInitializer"],
  MemInitializer: ["ParenMemInitializer", "BracedMemInitializer"],
  LambdaCapture: [
    "ThisLambdaCapture",
    "DerefThisLambdaCapture",
    "SimpleLambdaCapture",
    "RefLambdaCapture",
    "RefInitLambdaCapture",
    "InitLambdaCapture",
  ],
  ExceptionDeclaration: [
    "EllipsisExceptionDeclaration",
    "TypeExceptionDeclaration",
  ],
  AttributeSpecifier: [
    "CxxAttribute",
    "GccAttribute",
    "AlignasAttribute",
    "AlignasTypeAttribute",
    "AsmAttribute",
  ],
  AttributeToken: ["ScopedAttributeToken", "SimpleAttributeToken"],
}) as [ASTCategory, ASTKind[]][])
  for (const kind of kinds) categoryOf[kind] = category;

export type VisitNodeFunction<S, T extends AST> = (
  path: NodePath<T>,
  state: S,
) => void;

export interface VisitNodeObject<S, T extends AST> {
  enter?: VisitNodeFunction<S, T>;
  exit?: VisitNodeFunction<S, T>;
}

export type VisitNode<S, T extends AST> =
  VisitNodeFunction<S, T> | VisitNodeObject<S, T>;

export type Visitor<S = undefined> = {
  [K in keyof (ASTNodes & ASTCategories)]?: VisitNode<
    S,
    (ASTNodes & ASTCategories)[K]
  >;
} & {
  enter?: VisitNodeFunction<S, AST>;
  exit?: VisitNodeFunction<S, AST>;
};

export class NodePath<T extends AST = AST> {
  readonly node: T;
  readonly parentPath: NodePath | undefined;
  readonly key: string | number;
  readonly listKey: string | undefined;

  #skipped = false;
  #stopped = false;

  constructor(
    node: T,
    parentPath?: NodePath,
    key: string | number = "",
    listKey?: string,
  ) {
    this.node = node;
    this.parentPath = parentPath;
    this.key = key;
    this.listKey = listKey;
  }

  get parent(): AST | undefined {
    return this.parentPath?.node;
  }

  get kind(): ASTKind {
    return this.node.kind;
  }

  get category(): ASTCategory | undefined {
    return categoryOf[this.node.kind];
  }

  get depth(): number {
    let depth = 0;
    for (let path = this.parentPath; path; path = path.parentPath) ++depth;
    return depth;
  }

  get shouldSkip(): boolean {
    return this.#skipped;
  }

  get shouldStop(): boolean {
    return this.#stopped;
  }

  skip(): void {
    this.#skipped = true;
  }

  stop(): void {
    this.#skipped = true;
    this.#stopped = true;
  }

  isTranslationUnit(this: NodePath): this is NodePath<TranslationUnitAST> {
    return this.node.kind === "TranslationUnit";
  }

  isModuleUnit(this: NodePath): this is NodePath<ModuleUnitAST> {
    return this.node.kind === "ModuleUnit";
  }

  isSimpleDeclaration(this: NodePath): this is NodePath<SimpleDeclarationAST> {
    return this.node.kind === "SimpleDeclaration";
  }

  isAsmDeclaration(this: NodePath): this is NodePath<AsmDeclarationAST> {
    return this.node.kind === "AsmDeclaration";
  }

  isNamespaceAliasDefinition(
    this: NodePath,
  ): this is NodePath<NamespaceAliasDefinitionAST> {
    return this.node.kind === "NamespaceAliasDefinition";
  }

  isUsingDeclaration(this: NodePath): this is NodePath<UsingDeclarationAST> {
    return this.node.kind === "UsingDeclaration";
  }

  isUsingEnumDeclaration(
    this: NodePath,
  ): this is NodePath<UsingEnumDeclarationAST> {
    return this.node.kind === "UsingEnumDeclaration";
  }

  isUsingDirective(this: NodePath): this is NodePath<UsingDirectiveAST> {
    return this.node.kind === "UsingDirective";
  }

  isStaticAssertDeclaration(
    this: NodePath,
  ): this is NodePath<StaticAssertDeclarationAST> {
    return this.node.kind === "StaticAssertDeclaration";
  }

  isAliasDeclaration(this: NodePath): this is NodePath<AliasDeclarationAST> {
    return this.node.kind === "AliasDeclaration";
  }

  isOpaqueEnumDeclaration(
    this: NodePath,
  ): this is NodePath<OpaqueEnumDeclarationAST> {
    return this.node.kind === "OpaqueEnumDeclaration";
  }

  isFunctionDefinition(
    this: NodePath,
  ): this is NodePath<FunctionDefinitionAST> {
    return this.node.kind === "FunctionDefinition";
  }

  isTemplateDeclaration(
    this: NodePath,
  ): this is NodePath<TemplateDeclarationAST> {
    return this.node.kind === "TemplateDeclaration";
  }

  isConceptDefinition(this: NodePath): this is NodePath<ConceptDefinitionAST> {
    return this.node.kind === "ConceptDefinition";
  }

  isDeductionGuide(this: NodePath): this is NodePath<DeductionGuideAST> {
    return this.node.kind === "DeductionGuide";
  }

  isExplicitInstantiation(
    this: NodePath,
  ): this is NodePath<ExplicitInstantiationAST> {
    return this.node.kind === "ExplicitInstantiation";
  }

  isExportDeclaration(this: NodePath): this is NodePath<ExportDeclarationAST> {
    return this.node.kind === "ExportDeclaration";
  }

  isExportCompoundDeclaration(
    this: NodePath,
  ): this is NodePath<ExportCompoundDeclarationAST> {
    return this.node.kind === "ExportCompoundDeclaration";
  }

  isLinkageSpecification(
    this: NodePath,
  ): this is NodePath<LinkageSpecificationAST> {
    return this.node.kind === "LinkageSpecification";
  }

  isNamespaceDefinition(
    this: NodePath,
  ): this is NodePath<NamespaceDefinitionAST> {
    return this.node.kind === "NamespaceDefinition";
  }

  isEmptyDeclaration(this: NodePath): this is NodePath<EmptyDeclarationAST> {
    return this.node.kind === "EmptyDeclaration";
  }

  isAttributeDeclaration(
    this: NodePath,
  ): this is NodePath<AttributeDeclarationAST> {
    return this.node.kind === "AttributeDeclaration";
  }

  isModuleImportDeclaration(
    this: NodePath,
  ): this is NodePath<ModuleImportDeclarationAST> {
    return this.node.kind === "ModuleImportDeclaration";
  }

  isParameterDeclaration(
    this: NodePath,
  ): this is NodePath<ParameterDeclarationAST> {
    return this.node.kind === "ParameterDeclaration";
  }

  isAccessDeclaration(this: NodePath): this is NodePath<AccessDeclarationAST> {
    return this.node.kind === "AccessDeclaration";
  }

  isForRangeDeclaration(
    this: NodePath,
  ): this is NodePath<ForRangeDeclarationAST> {
    return this.node.kind === "ForRangeDeclaration";
  }

  isStructuredBindingDeclaration(
    this: NodePath,
  ): this is NodePath<StructuredBindingDeclarationAST> {
    return this.node.kind === "StructuredBindingDeclaration";
  }

  isAsmOperand(this: NodePath): this is NodePath<AsmOperandAST> {
    return this.node.kind === "AsmOperand";
  }

  isAsmQualifier(this: NodePath): this is NodePath<AsmQualifierAST> {
    return this.node.kind === "AsmQualifier";
  }

  isAsmClobber(this: NodePath): this is NodePath<AsmClobberAST> {
    return this.node.kind === "AsmClobber";
  }

  isAsmGotoLabel(this: NodePath): this is NodePath<AsmGotoLabelAST> {
    return this.node.kind === "AsmGotoLabel";
  }

  isSplicer(this: NodePath): this is NodePath<SplicerAST> {
    return this.node.kind === "Splicer";
  }

  isGlobalModuleFragment(
    this: NodePath,
  ): this is NodePath<GlobalModuleFragmentAST> {
    return this.node.kind === "GlobalModuleFragment";
  }

  isPrivateModuleFragment(
    this: NodePath,
  ): this is NodePath<PrivateModuleFragmentAST> {
    return this.node.kind === "PrivateModuleFragment";
  }

  isModuleDeclaration(this: NodePath): this is NodePath<ModuleDeclarationAST> {
    return this.node.kind === "ModuleDeclaration";
  }

  isModuleName(this: NodePath): this is NodePath<ModuleNameAST> {
    return this.node.kind === "ModuleName";
  }

  isModuleQualifier(this: NodePath): this is NodePath<ModuleQualifierAST> {
    return this.node.kind === "ModuleQualifier";
  }

  isModulePartition(this: NodePath): this is NodePath<ModulePartitionAST> {
    return this.node.kind === "ModulePartition";
  }

  isImportName(this: NodePath): this is NodePath<ImportNameAST> {
    return this.node.kind === "ImportName";
  }

  isInitDeclarator(this: NodePath): this is NodePath<InitDeclaratorAST> {
    return this.node.kind === "InitDeclarator";
  }

  isDeclarator(this: NodePath): this is NodePath<DeclaratorAST> {
    return this.node.kind === "Declarator";
  }

  isUsingDeclarator(this: NodePath): this is NodePath<UsingDeclaratorAST> {
    return this.node.kind === "UsingDeclarator";
  }

  isEnumerator(this: NodePath): this is NodePath<EnumeratorAST> {
    return this.node.kind === "Enumerator";
  }

  isTypeId(this: NodePath): this is NodePath<TypeIdAST> {
    return this.node.kind === "TypeId";
  }

  isHandler(this: NodePath): this is NodePath<HandlerAST> {
    return this.node.kind === "Handler";
  }

  isBaseSpecifier(this: NodePath): this is NodePath<BaseSpecifierAST> {
    return this.node.kind === "BaseSpecifier";
  }

  isRequiresClause(this: NodePath): this is NodePath<RequiresClauseAST> {
    return this.node.kind === "RequiresClause";
  }

  isParameterDeclarationClause(
    this: NodePath,
  ): this is NodePath<ParameterDeclarationClauseAST> {
    return this.node.kind === "ParameterDeclarationClause";
  }

  isTrailingReturnType(
    this: NodePath,
  ): this is NodePath<TrailingReturnTypeAST> {
    return this.node.kind === "TrailingReturnType";
  }

  isLambdaSpecifier(this: NodePath): this is NodePath<LambdaSpecifierAST> {
    return this.node.kind === "LambdaSpecifier";
  }

  isTypeConstraint(this: NodePath): this is NodePath<TypeConstraintAST> {
    return this.node.kind === "TypeConstraint";
  }

  isAttributeArgumentClause(
    this: NodePath,
  ): this is NodePath<AttributeArgumentClauseAST> {
    return this.node.kind === "AttributeArgumentClause";
  }

  isAttribute(this: NodePath): this is NodePath<AttributeAST> {
    return this.node.kind === "Attribute";
  }

  isAttributeUsingPrefix(
    this: NodePath,
  ): this is NodePath<AttributeUsingPrefixAST> {
    return this.node.kind === "AttributeUsingPrefix";
  }

  isNewPlacement(this: NodePath): this is NodePath<NewPlacementAST> {
    return this.node.kind === "NewPlacement";
  }

  isNestedNamespaceSpecifier(
    this: NodePath,
  ): this is NodePath<NestedNamespaceSpecifierAST> {
    return this.node.kind === "NestedNamespaceSpecifier";
  }

  isLabeledStatement(this: NodePath): this is NodePath<LabeledStatementAST> {
    return this.node.kind === "LabeledStatement";
  }

  isCaseStatement(this: NodePath): this is NodePath<CaseStatementAST> {
    return this.node.kind === "CaseStatement";
  }

  isDefaultStatement(this: NodePath): this is NodePath<DefaultStatementAST> {
    return this.node.kind === "DefaultStatement";
  }

  isExpressionStatement(
    this: NodePath,
  ): this is NodePath<ExpressionStatementAST> {
    return this.node.kind === "ExpressionStatement";
  }

  isCompoundStatement(this: NodePath): this is NodePath<CompoundStatementAST> {
    return this.node.kind === "CompoundStatement";
  }

  isIfStatement(this: NodePath): this is NodePath<IfStatementAST> {
    return this.node.kind === "IfStatement";
  }

  isConstevalIfStatement(
    this: NodePath,
  ): this is NodePath<ConstevalIfStatementAST> {
    return this.node.kind === "ConstevalIfStatement";
  }

  isSwitchStatement(this: NodePath): this is NodePath<SwitchStatementAST> {
    return this.node.kind === "SwitchStatement";
  }

  isWhileStatement(this: NodePath): this is NodePath<WhileStatementAST> {
    return this.node.kind === "WhileStatement";
  }

  isDoStatement(this: NodePath): this is NodePath<DoStatementAST> {
    return this.node.kind === "DoStatement";
  }

  isForRangeStatement(this: NodePath): this is NodePath<ForRangeStatementAST> {
    return this.node.kind === "ForRangeStatement";
  }

  isForStatement(this: NodePath): this is NodePath<ForStatementAST> {
    return this.node.kind === "ForStatement";
  }

  isBreakStatement(this: NodePath): this is NodePath<BreakStatementAST> {
    return this.node.kind === "BreakStatement";
  }

  isContinueStatement(this: NodePath): this is NodePath<ContinueStatementAST> {
    return this.node.kind === "ContinueStatement";
  }

  isReturnStatement(this: NodePath): this is NodePath<ReturnStatementAST> {
    return this.node.kind === "ReturnStatement";
  }

  isCoroutineReturnStatement(
    this: NodePath,
  ): this is NodePath<CoroutineReturnStatementAST> {
    return this.node.kind === "CoroutineReturnStatement";
  }

  isGotoStatement(this: NodePath): this is NodePath<GotoStatementAST> {
    return this.node.kind === "GotoStatement";
  }

  isDeclarationStatement(
    this: NodePath,
  ): this is NodePath<DeclarationStatementAST> {
    return this.node.kind === "DeclarationStatement";
  }

  isTryBlockStatement(this: NodePath): this is NodePath<TryBlockStatementAST> {
    return this.node.kind === "TryBlockStatement";
  }

  isCharLiteralExpression(
    this: NodePath,
  ): this is NodePath<CharLiteralExpressionAST> {
    return this.node.kind === "CharLiteralExpression";
  }

  isBoolLiteralExpression(
    this: NodePath,
  ): this is NodePath<BoolLiteralExpressionAST> {
    return this.node.kind === "BoolLiteralExpression";
  }

  isIntLiteralExpression(
    this: NodePath,
  ): this is NodePath<IntLiteralExpressionAST> {
    return this.node.kind === "IntLiteralExpression";
  }

  isFloatLiteralExpression(
    this: NodePath,
  ): this is NodePath<FloatLiteralExpressionAST> {
    return this.node.kind === "FloatLiteralExpression";
  }

  isNullptrLiteralExpression(
    this: NodePath,
  ): this is NodePath<NullptrLiteralExpressionAST> {
    return this.node.kind === "NullptrLiteralExpression";
  }

  isStringLiteralExpression(
    this: NodePath,
  ): this is NodePath<StringLiteralExpressionAST> {
    return this.node.kind === "StringLiteralExpression";
  }

  isUserDefinedStringLiteralExpression(
    this: NodePath,
  ): this is NodePath<UserDefinedStringLiteralExpressionAST> {
    return this.node.kind === "UserDefinedStringLiteralExpression";
  }

  isObjectLiteralExpression(
    this: NodePath,
  ): this is NodePath<ObjectLiteralExpressionAST> {
    return this.node.kind === "ObjectLiteralExpression";
  }

  isThisExpression(this: NodePath): this is NodePath<ThisExpressionAST> {
    return this.node.kind === "ThisExpression";
  }

  isPackIndexExpression(
    this: NodePath,
  ): this is NodePath<PackIndexExpressionAST> {
    return this.node.kind === "PackIndexExpression";
  }

  isGenericSelectionExpression(
    this: NodePath,
  ): this is NodePath<GenericSelectionExpressionAST> {
    return this.node.kind === "GenericSelectionExpression";
  }

  isNestedStatementExpression(
    this: NodePath,
  ): this is NodePath<NestedStatementExpressionAST> {
    return this.node.kind === "NestedStatementExpression";
  }

  isDefaultInitializerExpression(
    this: NodePath,
  ): this is NodePath<DefaultInitializerExpressionAST> {
    return this.node.kind === "DefaultInitializerExpression";
  }

  isNestedExpression(this: NodePath): this is NodePath<NestedExpressionAST> {
    return this.node.kind === "NestedExpression";
  }

  isIdExpression(this: NodePath): this is NodePath<IdExpressionAST> {
    return this.node.kind === "IdExpression";
  }

  isLambdaExpression(this: NodePath): this is NodePath<LambdaExpressionAST> {
    return this.node.kind === "LambdaExpression";
  }

  isFoldExpression(this: NodePath): this is NodePath<FoldExpressionAST> {
    return this.node.kind === "FoldExpression";
  }

  isRightFoldExpression(
    this: NodePath,
  ): this is NodePath<RightFoldExpressionAST> {
    return this.node.kind === "RightFoldExpression";
  }

  isLeftFoldExpression(
    this: NodePath,
  ): this is NodePath<LeftFoldExpressionAST> {
    return this.node.kind === "LeftFoldExpression";
  }

  isRequiresExpression(
    this: NodePath,
  ): this is NodePath<RequiresExpressionAST> {
    return this.node.kind === "RequiresExpression";
  }

  isVaArgExpression(this: NodePath): this is NodePath<VaArgExpressionAST> {
    return this.node.kind === "VaArgExpression";
  }

  isSubscriptExpression(
    this: NodePath,
  ): this is NodePath<SubscriptExpressionAST> {
    return this.node.kind === "SubscriptExpression";
  }

  isCallExpression(this: NodePath): this is NodePath<CallExpressionAST> {
    return this.node.kind === "CallExpression";
  }

  isTypeConstruction(this: NodePath): this is NodePath<TypeConstructionAST> {
    return this.node.kind === "TypeConstruction";
  }

  isBracedTypeConstruction(
    this: NodePath,
  ): this is NodePath<BracedTypeConstructionAST> {
    return this.node.kind === "BracedTypeConstruction";
  }

  isSpliceMemberExpression(
    this: NodePath,
  ): this is NodePath<SpliceMemberExpressionAST> {
    return this.node.kind === "SpliceMemberExpression";
  }

  isMemberExpression(this: NodePath): this is NodePath<MemberExpressionAST> {
    return this.node.kind === "MemberExpression";
  }

  isPostIncrExpression(
    this: NodePath,
  ): this is NodePath<PostIncrExpressionAST> {
    return this.node.kind === "PostIncrExpression";
  }

  isCppCastExpression(this: NodePath): this is NodePath<CppCastExpressionAST> {
    return this.node.kind === "CppCastExpression";
  }

  isBuiltinBitCastExpression(
    this: NodePath,
  ): this is NodePath<BuiltinBitCastExpressionAST> {
    return this.node.kind === "BuiltinBitCastExpression";
  }

  isBuiltinOffsetofExpression(
    this: NodePath,
  ): this is NodePath<BuiltinOffsetofExpressionAST> {
    return this.node.kind === "BuiltinOffsetofExpression";
  }

  isTypeidExpression(this: NodePath): this is NodePath<TypeidExpressionAST> {
    return this.node.kind === "TypeidExpression";
  }

  isTypeidOfTypeExpression(
    this: NodePath,
  ): this is NodePath<TypeidOfTypeExpressionAST> {
    return this.node.kind === "TypeidOfTypeExpression";
  }

  isSpliceExpression(this: NodePath): this is NodePath<SpliceExpressionAST> {
    return this.node.kind === "SpliceExpression";
  }

  isGlobalScopeReflectExpression(
    this: NodePath,
  ): this is NodePath<GlobalScopeReflectExpressionAST> {
    return this.node.kind === "GlobalScopeReflectExpression";
  }

  isNamespaceReflectExpression(
    this: NodePath,
  ): this is NodePath<NamespaceReflectExpressionAST> {
    return this.node.kind === "NamespaceReflectExpression";
  }

  isTypeIdReflectExpression(
    this: NodePath,
  ): this is NodePath<TypeIdReflectExpressionAST> {
    return this.node.kind === "TypeIdReflectExpression";
  }

  isReflectExpression(this: NodePath): this is NodePath<ReflectExpressionAST> {
    return this.node.kind === "ReflectExpression";
  }

  isLabelAddressExpression(
    this: NodePath,
  ): this is NodePath<LabelAddressExpressionAST> {
    return this.node.kind === "LabelAddressExpression";
  }

  isUnaryExpression(this: NodePath): this is NodePath<UnaryExpressionAST> {
    return this.node.kind === "UnaryExpression";
  }

  isAwaitExpression(this: NodePath): this is NodePath<AwaitExpressionAST> {
    return this.node.kind === "AwaitExpression";
  }

  isSizeofExpression(this: NodePath): this is NodePath<SizeofExpressionAST> {
    return this.node.kind === "SizeofExpression";
  }

  isSizeofTypeExpression(
    this: NodePath,
  ): this is NodePath<SizeofTypeExpressionAST> {
    return this.node.kind === "SizeofTypeExpression";
  }

  isSizeofPackExpression(
    this: NodePath,
  ): this is NodePath<SizeofPackExpressionAST> {
    return this.node.kind === "SizeofPackExpression";
  }

  isAlignofTypeExpression(
    this: NodePath,
  ): this is NodePath<AlignofTypeExpressionAST> {
    return this.node.kind === "AlignofTypeExpression";
  }

  isAlignofExpression(this: NodePath): this is NodePath<AlignofExpressionAST> {
    return this.node.kind === "AlignofExpression";
  }

  isNoexceptExpression(
    this: NodePath,
  ): this is NodePath<NoexceptExpressionAST> {
    return this.node.kind === "NoexceptExpression";
  }

  isNewExpression(this: NodePath): this is NodePath<NewExpressionAST> {
    return this.node.kind === "NewExpression";
  }

  isDeleteExpression(this: NodePath): this is NodePath<DeleteExpressionAST> {
    return this.node.kind === "DeleteExpression";
  }

  isCastExpression(this: NodePath): this is NodePath<CastExpressionAST> {
    return this.node.kind === "CastExpression";
  }

  isImplicitCastExpression(
    this: NodePath,
  ): this is NodePath<ImplicitCastExpressionAST> {
    return this.node.kind === "ImplicitCastExpression";
  }

  isConstExpression(this: NodePath): this is NodePath<ConstExpressionAST> {
    return this.node.kind === "ConstExpression";
  }

  isBinaryExpression(this: NodePath): this is NodePath<BinaryExpressionAST> {
    return this.node.kind === "BinaryExpression";
  }

  isConditionalExpression(
    this: NodePath,
  ): this is NodePath<ConditionalExpressionAST> {
    return this.node.kind === "ConditionalExpression";
  }

  isYieldExpression(this: NodePath): this is NodePath<YieldExpressionAST> {
    return this.node.kind === "YieldExpression";
  }

  isThrowExpression(this: NodePath): this is NodePath<ThrowExpressionAST> {
    return this.node.kind === "ThrowExpression";
  }

  isAssignmentExpression(
    this: NodePath,
  ): this is NodePath<AssignmentExpressionAST> {
    return this.node.kind === "AssignmentExpression";
  }

  isTargetExpression(this: NodePath): this is NodePath<TargetExpressionAST> {
    return this.node.kind === "TargetExpression";
  }

  isRightExpression(this: NodePath): this is NodePath<RightExpressionAST> {
    return this.node.kind === "RightExpression";
  }

  isCompoundAssignmentExpression(
    this: NodePath,
  ): this is NodePath<CompoundAssignmentExpressionAST> {
    return this.node.kind === "CompoundAssignmentExpression";
  }

  isPackExpansionExpression(
    this: NodePath,
  ): this is NodePath<PackExpansionExpressionAST> {
    return this.node.kind === "PackExpansionExpression";
  }

  isDesignatedInitializerClause(
    this: NodePath,
  ): this is NodePath<DesignatedInitializerClauseAST> {
    return this.node.kind === "DesignatedInitializerClause";
  }

  isTypeTraitExpression(
    this: NodePath,
  ): this is NodePath<TypeTraitExpressionAST> {
    return this.node.kind === "TypeTraitExpression";
  }

  isConditionExpression(
    this: NodePath,
  ): this is NodePath<ConditionExpressionAST> {
    return this.node.kind === "ConditionExpression";
  }

  isEqualInitializer(this: NodePath): this is NodePath<EqualInitializerAST> {
    return this.node.kind === "EqualInitializer";
  }

  isBracedInitList(this: NodePath): this is NodePath<BracedInitListAST> {
    return this.node.kind === "BracedInitList";
  }

  isParenInitializer(this: NodePath): this is NodePath<ParenInitializerAST> {
    return this.node.kind === "ParenInitializer";
  }

  isThreeWayComparisonExpression(
    this: NodePath,
  ): this is NodePath<ThreeWayComparisonExpressionAST> {
    return this.node.kind === "ThreeWayComparisonExpression";
  }

  isDefaultGenericAssociation(
    this: NodePath,
  ): this is NodePath<DefaultGenericAssociationAST> {
    return this.node.kind === "DefaultGenericAssociation";
  }

  isTypeGenericAssociation(
    this: NodePath,
  ): this is NodePath<TypeGenericAssociationAST> {
    return this.node.kind === "TypeGenericAssociation";
  }

  isDotDesignator(this: NodePath): this is NodePath<DotDesignatorAST> {
    return this.node.kind === "DotDesignator";
  }

  isSubscriptDesignator(
    this: NodePath,
  ): this is NodePath<SubscriptDesignatorAST> {
    return this.node.kind === "SubscriptDesignator";
  }

  isTemplateTypeParameter(
    this: NodePath,
  ): this is NodePath<TemplateTypeParameterAST> {
    return this.node.kind === "TemplateTypeParameter";
  }

  isNonTypeTemplateParameter(
    this: NodePath,
  ): this is NodePath<NonTypeTemplateParameterAST> {
    return this.node.kind === "NonTypeTemplateParameter";
  }

  isTypenameTypeParameter(
    this: NodePath,
  ): this is NodePath<TypenameTypeParameterAST> {
    return this.node.kind === "TypenameTypeParameter";
  }

  isConstraintTypeParameter(
    this: NodePath,
  ): this is NodePath<ConstraintTypeParameterAST> {
    return this.node.kind === "ConstraintTypeParameter";
  }

  isTypedefSpecifier(this: NodePath): this is NodePath<TypedefSpecifierAST> {
    return this.node.kind === "TypedefSpecifier";
  }

  isFriendSpecifier(this: NodePath): this is NodePath<FriendSpecifierAST> {
    return this.node.kind === "FriendSpecifier";
  }

  isConstevalSpecifier(
    this: NodePath,
  ): this is NodePath<ConstevalSpecifierAST> {
    return this.node.kind === "ConstevalSpecifier";
  }

  isConstinitSpecifier(
    this: NodePath,
  ): this is NodePath<ConstinitSpecifierAST> {
    return this.node.kind === "ConstinitSpecifier";
  }

  isConstexprSpecifier(
    this: NodePath,
  ): this is NodePath<ConstexprSpecifierAST> {
    return this.node.kind === "ConstexprSpecifier";
  }

  isInlineSpecifier(this: NodePath): this is NodePath<InlineSpecifierAST> {
    return this.node.kind === "InlineSpecifier";
  }

  isNoreturnSpecifier(this: NodePath): this is NodePath<NoreturnSpecifierAST> {
    return this.node.kind === "NoreturnSpecifier";
  }

  isStaticSpecifier(this: NodePath): this is NodePath<StaticSpecifierAST> {
    return this.node.kind === "StaticSpecifier";
  }

  isExternSpecifier(this: NodePath): this is NodePath<ExternSpecifierAST> {
    return this.node.kind === "ExternSpecifier";
  }

  isRegisterSpecifier(this: NodePath): this is NodePath<RegisterSpecifierAST> {
    return this.node.kind === "RegisterSpecifier";
  }

  isThreadLocalSpecifier(
    this: NodePath,
  ): this is NodePath<ThreadLocalSpecifierAST> {
    return this.node.kind === "ThreadLocalSpecifier";
  }

  isThreadSpecifier(this: NodePath): this is NodePath<ThreadSpecifierAST> {
    return this.node.kind === "ThreadSpecifier";
  }

  isMutableSpecifier(this: NodePath): this is NodePath<MutableSpecifierAST> {
    return this.node.kind === "MutableSpecifier";
  }

  isVirtualSpecifier(this: NodePath): this is NodePath<VirtualSpecifierAST> {
    return this.node.kind === "VirtualSpecifier";
  }

  isExplicitSpecifier(this: NodePath): this is NodePath<ExplicitSpecifierAST> {
    return this.node.kind === "ExplicitSpecifier";
  }

  isAutoTypeSpecifier(this: NodePath): this is NodePath<AutoTypeSpecifierAST> {
    return this.node.kind === "AutoTypeSpecifier";
  }

  isVoidTypeSpecifier(this: NodePath): this is NodePath<VoidTypeSpecifierAST> {
    return this.node.kind === "VoidTypeSpecifier";
  }

  isSizeTypeSpecifier(this: NodePath): this is NodePath<SizeTypeSpecifierAST> {
    return this.node.kind === "SizeTypeSpecifier";
  }

  isSignTypeSpecifier(this: NodePath): this is NodePath<SignTypeSpecifierAST> {
    return this.node.kind === "SignTypeSpecifier";
  }

  isBuiltinTypeSpecifier(
    this: NodePath,
  ): this is NodePath<BuiltinTypeSpecifierAST> {
    return this.node.kind === "BuiltinTypeSpecifier";
  }

  isUnaryBuiltinTypeSpecifier(
    this: NodePath,
  ): this is NodePath<UnaryBuiltinTypeSpecifierAST> {
    return this.node.kind === "UnaryBuiltinTypeSpecifier";
  }

  isBinaryBuiltinTypeSpecifier(
    this: NodePath,
  ): this is NodePath<BinaryBuiltinTypeSpecifierAST> {
    return this.node.kind === "BinaryBuiltinTypeSpecifier";
  }

  isIntegralTypeSpecifier(
    this: NodePath,
  ): this is NodePath<IntegralTypeSpecifierAST> {
    return this.node.kind === "IntegralTypeSpecifier";
  }

  isFloatingPointTypeSpecifier(
    this: NodePath,
  ): this is NodePath<FloatingPointTypeSpecifierAST> {
    return this.node.kind === "FloatingPointTypeSpecifier";
  }

  isComplexTypeSpecifier(
    this: NodePath,
  ): this is NodePath<ComplexTypeSpecifierAST> {
    return this.node.kind === "ComplexTypeSpecifier";
  }

  isNamedTypeSpecifier(
    this: NodePath,
  ): this is NodePath<NamedTypeSpecifierAST> {
    return this.node.kind === "NamedTypeSpecifier";
  }

  isAtomicTypeSpecifier(
    this: NodePath,
  ): this is NodePath<AtomicTypeSpecifierAST> {
    return this.node.kind === "AtomicTypeSpecifier";
  }

  isBitIntTypeSpecifier(
    this: NodePath,
  ): this is NodePath<BitIntTypeSpecifierAST> {
    return this.node.kind === "BitIntTypeSpecifier";
  }

  isUnderlyingTypeSpecifier(
    this: NodePath,
  ): this is NodePath<UnderlyingTypeSpecifierAST> {
    return this.node.kind === "UnderlyingTypeSpecifier";
  }

  isElaboratedTypeSpecifier(
    this: NodePath,
  ): this is NodePath<ElaboratedTypeSpecifierAST> {
    return this.node.kind === "ElaboratedTypeSpecifier";
  }

  isDecltypeAutoSpecifier(
    this: NodePath,
  ): this is NodePath<DecltypeAutoSpecifierAST> {
    return this.node.kind === "DecltypeAutoSpecifier";
  }

  isDecltypeSpecifier(this: NodePath): this is NodePath<DecltypeSpecifierAST> {
    return this.node.kind === "DecltypeSpecifier";
  }

  isPlaceholderTypeSpecifier(
    this: NodePath,
  ): this is NodePath<PlaceholderTypeSpecifierAST> {
    return this.node.kind === "PlaceholderTypeSpecifier";
  }

  isConstQualifier(this: NodePath): this is NodePath<ConstQualifierAST> {
    return this.node.kind === "ConstQualifier";
  }

  isVolatileQualifier(this: NodePath): this is NodePath<VolatileQualifierAST> {
    return this.node.kind === "VolatileQualifier";
  }

  isAtomicQualifier(this: NodePath): this is NodePath<AtomicQualifierAST> {
    return this.node.kind === "AtomicQualifier";
  }

  isRestrictQualifier(this: NodePath): this is NodePath<RestrictQualifierAST> {
    return this.node.kind === "RestrictQualifier";
  }

  isEnumSpecifier(this: NodePath): this is NodePath<EnumSpecifierAST> {
    return this.node.kind === "EnumSpecifier";
  }

  isClassSpecifier(this: NodePath): this is NodePath<ClassSpecifierAST> {
    return this.node.kind === "ClassSpecifier";
  }

  isTypenameSpecifier(this: NodePath): this is NodePath<TypenameSpecifierAST> {
    return this.node.kind === "TypenameSpecifier";
  }

  isSplicerTypeSpecifier(
    this: NodePath,
  ): this is NodePath<SplicerTypeSpecifierAST> {
    return this.node.kind === "SplicerTypeSpecifier";
  }

  isPointerOperator(this: NodePath): this is NodePath<PointerOperatorAST> {
    return this.node.kind === "PointerOperator";
  }

  isReferenceOperator(this: NodePath): this is NodePath<ReferenceOperatorAST> {
    return this.node.kind === "ReferenceOperator";
  }

  isPtrToMemberOperator(
    this: NodePath,
  ): this is NodePath<PtrToMemberOperatorAST> {
    return this.node.kind === "PtrToMemberOperator";
  }

  isBitfieldDeclarator(
    this: NodePath,
  ): this is NodePath<BitfieldDeclaratorAST> {
    return this.node.kind === "BitfieldDeclarator";
  }

  isParameterPack(this: NodePath): this is NodePath<ParameterPackAST> {
    return this.node.kind === "ParameterPack";
  }

  isIdDeclarator(this: NodePath): this is NodePath<IdDeclaratorAST> {
    return this.node.kind === "IdDeclarator";
  }

  isNestedDeclarator(this: NodePath): this is NodePath<NestedDeclaratorAST> {
    return this.node.kind === "NestedDeclarator";
  }

  isFunctionDeclaratorChunk(
    this: NodePath,
  ): this is NodePath<FunctionDeclaratorChunkAST> {
    return this.node.kind === "FunctionDeclaratorChunk";
  }

  isArrayDeclaratorChunk(
    this: NodePath,
  ): this is NodePath<ArrayDeclaratorChunkAST> {
    return this.node.kind === "ArrayDeclaratorChunk";
  }

  isNameId(this: NodePath): this is NodePath<NameIdAST> {
    return this.node.kind === "NameId";
  }

  isDestructorId(this: NodePath): this is NodePath<DestructorIdAST> {
    return this.node.kind === "DestructorId";
  }

  isDecltypeId(this: NodePath): this is NodePath<DecltypeIdAST> {
    return this.node.kind === "DecltypeId";
  }

  isOperatorFunctionId(
    this: NodePath,
  ): this is NodePath<OperatorFunctionIdAST> {
    return this.node.kind === "OperatorFunctionId";
  }

  isLiteralOperatorId(this: NodePath): this is NodePath<LiteralOperatorIdAST> {
    return this.node.kind === "LiteralOperatorId";
  }

  isConversionFunctionId(
    this: NodePath,
  ): this is NodePath<ConversionFunctionIdAST> {
    return this.node.kind === "ConversionFunctionId";
  }

  isSimpleTemplateId(this: NodePath): this is NodePath<SimpleTemplateIdAST> {
    return this.node.kind === "SimpleTemplateId";
  }

  isLiteralOperatorTemplateId(
    this: NodePath,
  ): this is NodePath<LiteralOperatorTemplateIdAST> {
    return this.node.kind === "LiteralOperatorTemplateId";
  }

  isOperatorFunctionTemplateId(
    this: NodePath,
  ): this is NodePath<OperatorFunctionTemplateIdAST> {
    return this.node.kind === "OperatorFunctionTemplateId";
  }

  isGlobalNestedNameSpecifier(
    this: NodePath,
  ): this is NodePath<GlobalNestedNameSpecifierAST> {
    return this.node.kind === "GlobalNestedNameSpecifier";
  }

  isSimpleNestedNameSpecifier(
    this: NodePath,
  ): this is NodePath<SimpleNestedNameSpecifierAST> {
    return this.node.kind === "SimpleNestedNameSpecifier";
  }

  isDecltypeNestedNameSpecifier(
    this: NodePath,
  ): this is NodePath<DecltypeNestedNameSpecifierAST> {
    return this.node.kind === "DecltypeNestedNameSpecifier";
  }

  isTemplateNestedNameSpecifier(
    this: NodePath,
  ): this is NodePath<TemplateNestedNameSpecifierAST> {
    return this.node.kind === "TemplateNestedNameSpecifier";
  }

  isDefaultFunctionBody(
    this: NodePath,
  ): this is NodePath<DefaultFunctionBodyAST> {
    return this.node.kind === "DefaultFunctionBody";
  }

  isCompoundStatementFunctionBody(
    this: NodePath,
  ): this is NodePath<CompoundStatementFunctionBodyAST> {
    return this.node.kind === "CompoundStatementFunctionBody";
  }

  isTryStatementFunctionBody(
    this: NodePath,
  ): this is NodePath<TryStatementFunctionBodyAST> {
    return this.node.kind === "TryStatementFunctionBody";
  }

  isDeleteFunctionBody(
    this: NodePath,
  ): this is NodePath<DeleteFunctionBodyAST> {
    return this.node.kind === "DeleteFunctionBody";
  }

  isTypeTemplateArgument(
    this: NodePath,
  ): this is NodePath<TypeTemplateArgumentAST> {
    return this.node.kind === "TypeTemplateArgument";
  }

  isExpressionTemplateArgument(
    this: NodePath,
  ): this is NodePath<ExpressionTemplateArgumentAST> {
    return this.node.kind === "ExpressionTemplateArgument";
  }

  isThrowExceptionSpecifier(
    this: NodePath,
  ): this is NodePath<ThrowExceptionSpecifierAST> {
    return this.node.kind === "ThrowExceptionSpecifier";
  }

  isNoexceptSpecifier(this: NodePath): this is NodePath<NoexceptSpecifierAST> {
    return this.node.kind === "NoexceptSpecifier";
  }

  isSimpleRequirement(this: NodePath): this is NodePath<SimpleRequirementAST> {
    return this.node.kind === "SimpleRequirement";
  }

  isCompoundRequirement(
    this: NodePath,
  ): this is NodePath<CompoundRequirementAST> {
    return this.node.kind === "CompoundRequirement";
  }

  isTypeRequirement(this: NodePath): this is NodePath<TypeRequirementAST> {
    return this.node.kind === "TypeRequirement";
  }

  isNestedRequirement(this: NodePath): this is NodePath<NestedRequirementAST> {
    return this.node.kind === "NestedRequirement";
  }

  isNewParenInitializer(
    this: NodePath,
  ): this is NodePath<NewParenInitializerAST> {
    return this.node.kind === "NewParenInitializer";
  }

  isNewBracedInitializer(
    this: NodePath,
  ): this is NodePath<NewBracedInitializerAST> {
    return this.node.kind === "NewBracedInitializer";
  }

  isParenMemInitializer(
    this: NodePath,
  ): this is NodePath<ParenMemInitializerAST> {
    return this.node.kind === "ParenMemInitializer";
  }

  isBracedMemInitializer(
    this: NodePath,
  ): this is NodePath<BracedMemInitializerAST> {
    return this.node.kind === "BracedMemInitializer";
  }

  isThisLambdaCapture(this: NodePath): this is NodePath<ThisLambdaCaptureAST> {
    return this.node.kind === "ThisLambdaCapture";
  }

  isDerefThisLambdaCapture(
    this: NodePath,
  ): this is NodePath<DerefThisLambdaCaptureAST> {
    return this.node.kind === "DerefThisLambdaCapture";
  }

  isSimpleLambdaCapture(
    this: NodePath,
  ): this is NodePath<SimpleLambdaCaptureAST> {
    return this.node.kind === "SimpleLambdaCapture";
  }

  isRefLambdaCapture(this: NodePath): this is NodePath<RefLambdaCaptureAST> {
    return this.node.kind === "RefLambdaCapture";
  }

  isRefInitLambdaCapture(
    this: NodePath,
  ): this is NodePath<RefInitLambdaCaptureAST> {
    return this.node.kind === "RefInitLambdaCapture";
  }

  isInitLambdaCapture(this: NodePath): this is NodePath<InitLambdaCaptureAST> {
    return this.node.kind === "InitLambdaCapture";
  }

  isEllipsisExceptionDeclaration(
    this: NodePath,
  ): this is NodePath<EllipsisExceptionDeclarationAST> {
    return this.node.kind === "EllipsisExceptionDeclaration";
  }

  isTypeExceptionDeclaration(
    this: NodePath,
  ): this is NodePath<TypeExceptionDeclarationAST> {
    return this.node.kind === "TypeExceptionDeclaration";
  }

  isCxxAttribute(this: NodePath): this is NodePath<CxxAttributeAST> {
    return this.node.kind === "CxxAttribute";
  }

  isGccAttribute(this: NodePath): this is NodePath<GccAttributeAST> {
    return this.node.kind === "GccAttribute";
  }

  isAlignasAttribute(this: NodePath): this is NodePath<AlignasAttributeAST> {
    return this.node.kind === "AlignasAttribute";
  }

  isAlignasTypeAttribute(
    this: NodePath,
  ): this is NodePath<AlignasTypeAttributeAST> {
    return this.node.kind === "AlignasTypeAttribute";
  }

  isAsmAttribute(this: NodePath): this is NodePath<AsmAttributeAST> {
    return this.node.kind === "AsmAttribute";
  }

  isScopedAttributeToken(
    this: NodePath,
  ): this is NodePath<ScopedAttributeTokenAST> {
    return this.node.kind === "ScopedAttributeToken";
  }

  isSimpleAttributeToken(
    this: NodePath,
  ): this is NodePath<SimpleAttributeTokenAST> {
    return this.node.kind === "SimpleAttributeToken";
  }

  isAttributeSpecifier(
    this: NodePath,
  ): this is NodePath<AttributeSpecifierAST> {
    return categoryOf[this.node.kind] === "AttributeSpecifier";
  }

  isAttributeToken(this: NodePath): this is NodePath<AttributeTokenAST> {
    return categoryOf[this.node.kind] === "AttributeToken";
  }

  isCoreDeclarator(this: NodePath): this is NodePath<CoreDeclaratorAST> {
    return categoryOf[this.node.kind] === "CoreDeclarator";
  }

  isDeclaration(this: NodePath): this is NodePath<DeclarationAST> {
    return categoryOf[this.node.kind] === "Declaration";
  }

  isDeclaratorChunk(this: NodePath): this is NodePath<DeclaratorChunkAST> {
    return categoryOf[this.node.kind] === "DeclaratorChunk";
  }

  isDesignator(this: NodePath): this is NodePath<DesignatorAST> {
    return categoryOf[this.node.kind] === "Designator";
  }

  isExceptionDeclaration(
    this: NodePath,
  ): this is NodePath<ExceptionDeclarationAST> {
    return categoryOf[this.node.kind] === "ExceptionDeclaration";
  }

  isExceptionSpecifier(
    this: NodePath,
  ): this is NodePath<ExceptionSpecifierAST> {
    return categoryOf[this.node.kind] === "ExceptionSpecifier";
  }

  isExpression(this: NodePath): this is NodePath<ExpressionAST> {
    return categoryOf[this.node.kind] === "Expression";
  }

  isFunctionBody(this: NodePath): this is NodePath<FunctionBodyAST> {
    return categoryOf[this.node.kind] === "FunctionBody";
  }

  isGenericAssociation(
    this: NodePath,
  ): this is NodePath<GenericAssociationAST> {
    return categoryOf[this.node.kind] === "GenericAssociation";
  }

  isLambdaCapture(this: NodePath): this is NodePath<LambdaCaptureAST> {
    return categoryOf[this.node.kind] === "LambdaCapture";
  }

  isMemInitializer(this: NodePath): this is NodePath<MemInitializerAST> {
    return categoryOf[this.node.kind] === "MemInitializer";
  }

  isNestedNameSpecifier(
    this: NodePath,
  ): this is NodePath<NestedNameSpecifierAST> {
    return categoryOf[this.node.kind] === "NestedNameSpecifier";
  }

  isNewInitializer(this: NodePath): this is NodePath<NewInitializerAST> {
    return categoryOf[this.node.kind] === "NewInitializer";
  }

  isPtrOperator(this: NodePath): this is NodePath<PtrOperatorAST> {
    return categoryOf[this.node.kind] === "PtrOperator";
  }

  isRequirement(this: NodePath): this is NodePath<RequirementAST> {
    return categoryOf[this.node.kind] === "Requirement";
  }

  isSpecifier(this: NodePath): this is NodePath<SpecifierAST> {
    return categoryOf[this.node.kind] === "Specifier";
  }

  isStatement(this: NodePath): this is NodePath<StatementAST> {
    return categoryOf[this.node.kind] === "Statement";
  }

  isTemplateArgument(this: NodePath): this is NodePath<TemplateArgumentAST> {
    return categoryOf[this.node.kind] === "TemplateArgument";
  }

  isTemplateParameter(this: NodePath): this is NodePath<TemplateParameterAST> {
    return categoryOf[this.node.kind] === "TemplateParameter";
  }

  isUnit(this: NodePath): this is NodePath<UnitAST> {
    return categoryOf[this.node.kind] === "Unit";
  }

  isUnqualifiedId(this: NodePath): this is NodePath<UnqualifiedIdAST> {
    return categoryOf[this.node.kind] === "UnqualifiedId";
  }

  *[Symbol.iterator](): Generator<NodePath> {
    yield* this.children();
  }

  *children(): Generator<NodePath> {
    for (const { node, key, listKey } of children(this.node))
      yield new NodePath(node, this, key, listKey);
  }

  *ancestors(): Generator<NodePath> {
    for (let path = this.parentPath; path; path = path.parentPath) yield path;
  }

  *descendants(): Generator<NodePath> {
    for (const path of walk(this)) if (path !== this) yield path;
  }

  find(predicate: (path: NodePath) => boolean): NodePath | undefined {
    for (let path: NodePath | undefined = this; path; path = path.parentPath)
      if (predicate(path)) return path;
    return undefined;
  }

  findParent(predicate: (path: NodePath) => boolean): NodePath | undefined {
    return this.parentPath?.find(predicate);
  }

  traverse<S = undefined>(visitor: Visitor<S>, state?: S): S {
    for (const path of this.children())
      if (visit(path, visitor, state as S)) break;
    return state as S;
  }

  toString(): string {
    return this.node.kind;
  }
}

function pathOf(root: AST | NodePath): NodePath {
  return root instanceof NodePath ? root : new NodePath(root);
}

export function* walk(root: AST | NodePath): Generator<NodePath> {
  const stack = [pathOf(root)];

  while (stack.length) {
    const path = stack.pop()!;

    yield path;

    if (path.shouldStop) return;
    if (path.shouldSkip) continue;

    const children = [...path.children()];
    for (let i = children.length - 1; i >= 0; --i) stack.push(children[i]!);
  }
}

function dispatch<S>(
  path: NodePath,
  visitor: Visitor<S>,
  state: S,
  phase: "enter" | "exit",
): void {
  visitor[phase]?.(path, state);
  if (path.shouldStop) return;

  for (const key of [path.node.kind, categoryOf[path.node.kind]]) {
    if (!key) continue;
    const entry = visitor[key] as VisitNode<S, AST> | undefined;
    if (!entry) continue;
    if (typeof entry === "function") {
      if (phase === "enter") entry(path, state);
    } else {
      entry[phase]?.(path, state);
    }
    if (path.shouldStop) return;
  }
}

function visit<S>(root: NodePath, visitor: Visitor<S>, state: S): boolean {
  const stack: { path: NodePath; children?: NodePath[]; index: number }[] = [
    { path: root, index: 0 },
  ];

  while (stack.length) {
    const frame = stack.at(-1)!;

    if (!frame.children) {
      dispatch(frame.path, visitor, state, "enter");
      if (frame.path.shouldStop) return true;
      if (frame.path.shouldSkip) {
        stack.pop();
        continue;
      }
      frame.children = [...frame.path.children()];
    }

    const child = frame.children[frame.index++];
    if (child) {
      stack.push({ path: child, index: 0 });
      continue;
    }

    stack.pop();
    dispatch(frame.path, visitor, state, "exit");
    if (frame.path.shouldStop) return true;
  }

  return false;
}

export function traverse<S = undefined>(
  root: AST | NodePath,
  visitor: Visitor<S>,
  state?: S,
): S {
  visit(pathOf(root), visitor, state as S);
  return state as S;
}
