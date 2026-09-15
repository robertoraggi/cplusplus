// Generated file by: gen_reflection.ts
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

import { cxx } from "./cxx.js";
import { type SourceLocation } from "./SourceLocation.js";

export interface ModelOwner {
  getUnitHandle(): number;
  readonly disposed: boolean;
}

function disposedError(): Error {
  return new Error("Parser has been disposed");
}

export abstract class ModelObject {
  readonly #handle: number;

  constructor(
    handle: number,
    readonly modelOwner: ModelOwner,
  ) {
    this.#handle = handle;
  }

  get handle(): number {
    if (this.modelOwner.disposed) throw disposedError();
    return this.#handle;
  }
}

function objOf<T>(
  handle: number,
  owner: ModelOwner,
  ctor: new (handle: number, owner: ModelOwner) => T,
): T | undefined {
  if (!handle) return undefined;
  return new ctor(handle, owner);
}

function objAt<T>(
  handle: number,
  owner: ModelOwner,
  ctor: new (handle: number, owner: ModelOwner) => T,
): T {
  return new ctor(handle, owner);
}

function* listOf(
  owner: ModelOwner,
  head: number,
  of: (handle: number) => any,
): Iterable<any> {
  let it = head;
  while (it) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.getListValue(it));
    it = cxx.getListNext(it);
  }
}
const astConstructors: Array<
  new (handle: number, owner: ModelOwner, kind: ASTKind) => AST
> = [];

function astOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = cxx.getASTKind(handle);
  return new astConstructors[kind]!(handle, owner, kind);
}
const symbolConstructors: Array<
  new (handle: number, owner: ModelOwner, kind: SymbolKind) => Symbol
> = [];

function symbolOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = cxx.getSymbolKind(handle);
  return new symbolConstructors[kind]!(handle, owner, kind);
}
const typeConstructors: Array<
  new (handle: number, owner: ModelOwner, kind: TypeKind) => Type
> = [];

function typeOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = cxx.getTypeKind(handle);
  return new typeConstructors[kind]!(handle, owner, kind);
}
const nameConstructors: Array<
  new (handle: number, owner: ModelOwner, kind: NameKind) => Name
> = [];

function nameOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = cxx.getNameKind(handle);
  return new nameConstructors[kind]!(handle, owner, kind);
}
function* miscValItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const size = cxx.readMiscSize(handle, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readMiscItemVal(handle, slot, i));
  }
}
function* nameValItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const size = cxx.readNameSize(handle, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readNameItemVal(handle, slot, i));
  }
}
function* symbolItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const size = cxx.readSymbolSize(handle, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readSymbolItem(handle, slot, i));
  }
}
function* symbolValItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const size = cxx.readSymbolSize(handle, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readSymbolItemVal(handle, slot, i));
  }
}
function* symbolStringItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const size = cxx.readSymbolSize(handle, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readSymbolItemString(handle, slot, i));
  }
}
function* typeItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const size = cxx.readTypeSize(handle, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readTypeItem(handle, slot, i));
  }
}
const ConstComplexSlotBase = 0;
const ConstObjectSlotBase = ConstComplexSlotBase + 2;
const ConstAddressSlotBase = ConstObjectSlotBase + 3;
const ConstLabelAddressSlotBase = ConstAddressSlotBase + 5;
const ASTSlotBase = 0;
const AttributeSpecifierASTSlotBase = ASTSlotBase + 3;
const AttributeTokenASTSlotBase = AttributeSpecifierASTSlotBase + 4;
const CoreDeclaratorASTSlotBase = AttributeTokenASTSlotBase + 3;
const DeclarationASTSlotBase = CoreDeclaratorASTSlotBase + 3;
const DeclaratorChunkASTSlotBase = DeclarationASTSlotBase + 3;
const DesignatorASTSlotBase = DeclaratorChunkASTSlotBase + 3;
const ExceptionDeclarationASTSlotBase = DesignatorASTSlotBase + 3;
const ExceptionSpecifierASTSlotBase = ExceptionDeclarationASTSlotBase + 3;
const ExpressionASTSlotBase = ExceptionSpecifierASTSlotBase + 3;
const FunctionBodyASTSlotBase = ExpressionASTSlotBase + 5;
const GenericAssociationASTSlotBase = FunctionBodyASTSlotBase + 3;
const LambdaCaptureASTSlotBase = GenericAssociationASTSlotBase + 3;
const MemInitializerASTSlotBase = LambdaCaptureASTSlotBase + 3;
const NestedNameSpecifierASTSlotBase = MemInitializerASTSlotBase + 5;
const NewInitializerASTSlotBase = NestedNameSpecifierASTSlotBase + 4;
const PtrOperatorASTSlotBase = NewInitializerASTSlotBase + 3;
const RequirementASTSlotBase = PtrOperatorASTSlotBase + 3;
const SpecifierASTSlotBase = RequirementASTSlotBase + 3;
const StatementASTSlotBase = SpecifierASTSlotBase + 3;
const TemplateArgumentASTSlotBase = StatementASTSlotBase + 3;
const TemplateParameterASTSlotBase = TemplateArgumentASTSlotBase + 3;
const UnitASTSlotBase = TemplateParameterASTSlotBase + 6;
const UnqualifiedIdASTSlotBase = UnitASTSlotBase + 4;
const TranslationUnitASTSlotBase = UnqualifiedIdASTSlotBase + 3;
const ModuleUnitASTSlotBase = TranslationUnitASTSlotBase + 5;
const SimpleDeclarationASTSlotBase = ModuleUnitASTSlotBase + 8;
const AsmDeclarationASTSlotBase = SimpleDeclarationASTSlotBase + 8;
const NamespaceAliasDefinitionASTSlotBase = AsmDeclarationASTSlotBase + 15;
const UsingDeclarationASTSlotBase = NamespaceAliasDefinitionASTSlotBase + 11;
const UsingEnumDeclarationASTSlotBase = UsingDeclarationASTSlotBase + 6;
const UsingDirectiveASTSlotBase = UsingEnumDeclarationASTSlotBase + 6;
const StaticAssertDeclarationASTSlotBase = UsingDirectiveASTSlotBase + 9;
const AliasDeclarationASTSlotBase = StaticAssertDeclarationASTSlotBase + 12;
const OpaqueEnumDeclarationASTSlotBase = AliasDeclarationASTSlotBase + 12;
const FunctionDefinitionASTSlotBase = OpaqueEnumDeclarationASTSlotBase + 12;
const TemplateDeclarationASTSlotBase = FunctionDefinitionASTSlotBase + 9;
const ConceptDefinitionASTSlotBase = TemplateDeclarationASTSlotBase + 11;
const DeductionGuideASTSlotBase = ConceptDefinitionASTSlotBase + 10;
const ExplicitInstantiationASTSlotBase = DeductionGuideASTSlotBase + 14;
const ExportDeclarationASTSlotBase = ExplicitInstantiationASTSlotBase + 6;
const ExportCompoundDeclarationASTSlotBase = ExportDeclarationASTSlotBase + 5;
const LinkageSpecificationASTSlotBase =
  ExportCompoundDeclarationASTSlotBase + 7;
const NamespaceDefinitionASTSlotBase = LinkageSpecificationASTSlotBase + 9;
const EmptyDeclarationASTSlotBase = NamespaceDefinitionASTSlotBase + 15;
const AttributeDeclarationASTSlotBase = EmptyDeclarationASTSlotBase + 4;
const ModuleImportDeclarationASTSlotBase = AttributeDeclarationASTSlotBase + 5;
const ParameterDeclarationASTSlotBase = ModuleImportDeclarationASTSlotBase + 7;
const AccessDeclarationASTSlotBase = ParameterDeclarationASTSlotBase + 14;
const ForRangeDeclarationASTSlotBase = AccessDeclarationASTSlotBase + 6;
const StructuredBindingDeclarationASTSlotBase =
  ForRangeDeclarationASTSlotBase + 3;
const AsmOperandASTSlotBase = StructuredBindingDeclarationASTSlotBase + 13;
const AsmQualifierASTSlotBase = AsmOperandASTSlotBase + 12;
const AsmClobberASTSlotBase = AsmQualifierASTSlotBase + 5;
const AsmGotoLabelASTSlotBase = AsmClobberASTSlotBase + 5;
const SplicerASTSlotBase = AsmGotoLabelASTSlotBase + 5;
const GlobalModuleFragmentASTSlotBase = SplicerASTSlotBase + 9;
const PrivateModuleFragmentASTSlotBase = GlobalModuleFragmentASTSlotBase + 6;
const ModuleDeclarationASTSlotBase = PrivateModuleFragmentASTSlotBase + 8;
const ModuleNameASTSlotBase = ModuleDeclarationASTSlotBase + 9;
const ModuleQualifierASTSlotBase = ModuleNameASTSlotBase + 6;
const ModulePartitionASTSlotBase = ModuleQualifierASTSlotBase + 7;
const ImportNameASTSlotBase = ModulePartitionASTSlotBase + 5;
const InitDeclaratorASTSlotBase = ImportNameASTSlotBase + 6;
const DeclaratorASTSlotBase = InitDeclaratorASTSlotBase + 7;
const UsingDeclaratorASTSlotBase = DeclaratorASTSlotBase + 6;
const EnumeratorASTSlotBase = UsingDeclaratorASTSlotBase + 9;
const TypeIdASTSlotBase = EnumeratorASTSlotBase + 9;
const HandlerASTSlotBase = TypeIdASTSlotBase + 7;
const BaseSpecifierASTSlotBase = HandlerASTSlotBase + 9;
const RequiresClauseASTSlotBase = BaseSpecifierASTSlotBase + 15;
const ParameterDeclarationClauseASTSlotBase = RequiresClauseASTSlotBase + 5;
const TrailingReturnTypeASTSlotBase = ParameterDeclarationClauseASTSlotBase + 8;
const LambdaSpecifierASTSlotBase = TrailingReturnTypeASTSlotBase + 5;
const TypeConstraintASTSlotBase = LambdaSpecifierASTSlotBase + 5;
const AttributeArgumentClauseASTSlotBase = TypeConstraintASTSlotBase + 10;
const AttributeASTSlotBase = AttributeArgumentClauseASTSlotBase + 6;
const AttributeUsingPrefixASTSlotBase = AttributeASTSlotBase + 6;
const NewPlacementASTSlotBase = AttributeUsingPrefixASTSlotBase + 6;
const NestedNamespaceSpecifierASTSlotBase = NewPlacementASTSlotBase + 6;
const LabeledStatementASTSlotBase = NestedNamespaceSpecifierASTSlotBase + 9;
const CaseStatementASTSlotBase = LabeledStatementASTSlotBase + 7;
const DefaultStatementASTSlotBase = CaseStatementASTSlotBase + 7;
const ExpressionStatementASTSlotBase = DefaultStatementASTSlotBase + 5;
const CompoundStatementASTSlotBase = ExpressionStatementASTSlotBase + 6;
const IfStatementASTSlotBase = CompoundStatementASTSlotBase + 8;
const ConstevalIfStatementASTSlotBase = IfStatementASTSlotBase + 14;
const SwitchStatementASTSlotBase = ConstevalIfStatementASTSlotBase + 11;
const WhileStatementASTSlotBase = SwitchStatementASTSlotBase + 11;
const DoStatementASTSlotBase = WhileStatementASTSlotBase + 10;
const ForRangeStatementASTSlotBase = DoStatementASTSlotBase + 11;
const ForStatementASTSlotBase = ForRangeStatementASTSlotBase + 30;
const BreakStatementASTSlotBase = ForStatementASTSlotBase + 13;
const ContinueStatementASTSlotBase = BreakStatementASTSlotBase + 6;
const ReturnStatementASTSlotBase = ContinueStatementASTSlotBase + 6;
const CoroutineReturnStatementASTSlotBase = ReturnStatementASTSlotBase + 7;
const GotoStatementASTSlotBase = CoroutineReturnStatementASTSlotBase + 7;
const DeclarationStatementASTSlotBase = GotoStatementASTSlotBase + 11;
const TryBlockStatementASTSlotBase = DeclarationStatementASTSlotBase + 4;
const CharLiteralExpressionASTSlotBase = TryBlockStatementASTSlotBase + 7;
const BoolLiteralExpressionASTSlotBase = CharLiteralExpressionASTSlotBase + 8;
const IntLiteralExpressionASTSlotBase = BoolLiteralExpressionASTSlotBase + 7;
const FloatLiteralExpressionASTSlotBase = IntLiteralExpressionASTSlotBase + 8;
const NullptrLiteralExpressionASTSlotBase =
  FloatLiteralExpressionASTSlotBase + 8;
const StringLiteralExpressionASTSlotBase =
  NullptrLiteralExpressionASTSlotBase + 7;
const UserDefinedStringLiteralExpressionASTSlotBase =
  StringLiteralExpressionASTSlotBase + 8;
const ObjectLiteralExpressionASTSlotBase =
  UserDefinedStringLiteralExpressionASTSlotBase + 9;
const ThisExpressionASTSlotBase = ObjectLiteralExpressionASTSlotBase + 10;
const PackIndexExpressionASTSlotBase = ThisExpressionASTSlotBase + 6;
const GenericSelectionExpressionASTSlotBase =
  PackIndexExpressionASTSlotBase + 10;
const NestedStatementExpressionASTSlotBase =
  GenericSelectionExpressionASTSlotBase + 12;
const DefaultInitializerExpressionASTSlotBase =
  NestedStatementExpressionASTSlotBase + 8;
const NestedExpressionASTSlotBase = DefaultInitializerExpressionASTSlotBase + 7;
const IdExpressionASTSlotBase = NestedExpressionASTSlotBase + 8;
const LambdaExpressionASTSlotBase = IdExpressionASTSlotBase + 10;
const FoldExpressionASTSlotBase = LambdaExpressionASTSlotBase + 27;
const RightFoldExpressionASTSlotBase = FoldExpressionASTSlotBase + 14;
const LeftFoldExpressionASTSlotBase = RightFoldExpressionASTSlotBase + 11;
const RequiresExpressionASTSlotBase = LeftFoldExpressionASTSlotBase + 11;
const VaArgExpressionASTSlotBase = RequiresExpressionASTSlotBase + 12;
const SubscriptExpressionASTSlotBase = VaArgExpressionASTSlotBase + 11;
const CallExpressionASTSlotBase = SubscriptExpressionASTSlotBase + 11;
const TypeConstructionASTSlotBase = CallExpressionASTSlotBase + 11;
const BracedTypeConstructionASTSlotBase = TypeConstructionASTSlotBase + 10;
const SpliceMemberExpressionASTSlotBase = BracedTypeConstructionASTSlotBase + 8;
const MemberExpressionASTSlotBase = SpliceMemberExpressionASTSlotBase + 12;
const PostIncrExpressionASTSlotBase = MemberExpressionASTSlotBase + 13;
const CppCastExpressionASTSlotBase = PostIncrExpressionASTSlotBase + 10;
const BuiltinBitCastExpressionASTSlotBase = CppCastExpressionASTSlotBase + 13;
const BuiltinOffsetofExpressionASTSlotBase =
  BuiltinBitCastExpressionASTSlotBase + 11;
const TypeidExpressionASTSlotBase = BuiltinOffsetofExpressionASTSlotBase + 14;
const TypeidOfTypeExpressionASTSlotBase = TypeidExpressionASTSlotBase + 9;
const SpliceExpressionASTSlotBase = TypeidOfTypeExpressionASTSlotBase + 9;
const GlobalScopeReflectExpressionASTSlotBase = SpliceExpressionASTSlotBase + 6;
const NamespaceReflectExpressionASTSlotBase =
  GlobalScopeReflectExpressionASTSlotBase + 7;
const TypeIdReflectExpressionASTSlotBase =
  NamespaceReflectExpressionASTSlotBase + 9;
const ReflectExpressionASTSlotBase = TypeIdReflectExpressionASTSlotBase + 7;
const LabelAddressExpressionASTSlotBase = ReflectExpressionASTSlotBase + 7;
const UnaryExpressionASTSlotBase = LabelAddressExpressionASTSlotBase + 8;
const AwaitExpressionASTSlotBase = UnaryExpressionASTSlotBase + 10;
const SizeofExpressionASTSlotBase = AwaitExpressionASTSlotBase + 7;
const SizeofTypeExpressionASTSlotBase = SizeofExpressionASTSlotBase + 8;
const SizeofPackExpressionASTSlotBase = SizeofTypeExpressionASTSlotBase + 10;
const AlignofTypeExpressionASTSlotBase = SizeofPackExpressionASTSlotBase + 12;
const AlignofExpressionASTSlotBase = AlignofTypeExpressionASTSlotBase + 9;
const NoexceptExpressionASTSlotBase = AlignofExpressionASTSlotBase + 7;
const NewExpressionASTSlotBase = NoexceptExpressionASTSlotBase + 10;
const DeleteExpressionASTSlotBase = NewExpressionASTSlotBase + 16;
const CastExpressionASTSlotBase = DeleteExpressionASTSlotBase + 11;
const ImplicitCastExpressionASTSlotBase = CastExpressionASTSlotBase + 9;
const ConstExpressionASTSlotBase = ImplicitCastExpressionASTSlotBase + 9;
const BinaryExpressionASTSlotBase = ConstExpressionASTSlotBase + 7;
const ConditionalExpressionASTSlotBase = BinaryExpressionASTSlotBase + 11;
const YieldExpressionASTSlotBase = ConditionalExpressionASTSlotBase + 10;
const ThrowExpressionASTSlotBase = YieldExpressionASTSlotBase + 7;
const AssignmentExpressionASTSlotBase = ThrowExpressionASTSlotBase + 7;
const TargetExpressionASTSlotBase = AssignmentExpressionASTSlotBase + 11;
const RightExpressionASTSlotBase = TargetExpressionASTSlotBase + 5;
const CompoundAssignmentExpressionASTSlotBase = RightExpressionASTSlotBase + 5;
const PackExpansionExpressionASTSlotBase =
  CompoundAssignmentExpressionASTSlotBase + 13;
const DesignatedInitializerClauseASTSlotBase =
  PackExpansionExpressionASTSlotBase + 7;
const TypeTraitExpressionASTSlotBase =
  DesignatedInitializerClauseASTSlotBase + 8;
const ConditionExpressionASTSlotBase = TypeTraitExpressionASTSlotBase + 11;
const EqualInitializerASTSlotBase = ConditionExpressionASTSlotBase + 10;
const BracedInitListASTSlotBase = EqualInitializerASTSlotBase + 7;
const ParenInitializerASTSlotBase = BracedInitListASTSlotBase + 9;
const ThreeWayComparisonExpressionASTSlotBase = ParenInitializerASTSlotBase + 8;
const DefaultGenericAssociationASTSlotBase =
  ThreeWayComparisonExpressionASTSlotBase + 10;
const TypeGenericAssociationASTSlotBase =
  DefaultGenericAssociationASTSlotBase + 6;
const DotDesignatorASTSlotBase = TypeGenericAssociationASTSlotBase + 6;
const SubscriptDesignatorASTSlotBase = DotDesignatorASTSlotBase + 7;
const TemplateTypeParameterASTSlotBase = SubscriptDesignatorASTSlotBase + 6;
const NonTypeTemplateParameterASTSlotBase =
  TemplateTypeParameterASTSlotBase + 18;
const TypenameTypeParameterASTSlotBase =
  NonTypeTemplateParameterASTSlotBase + 7;
const ConstraintTypeParameterASTSlotBase =
  TypenameTypeParameterASTSlotBase + 13;
const TypedefSpecifierASTSlotBase = ConstraintTypeParameterASTSlotBase + 12;
const FriendSpecifierASTSlotBase = TypedefSpecifierASTSlotBase + 4;
const ConstevalSpecifierASTSlotBase = FriendSpecifierASTSlotBase + 4;
const ConstinitSpecifierASTSlotBase = ConstevalSpecifierASTSlotBase + 4;
const ConstexprSpecifierASTSlotBase = ConstinitSpecifierASTSlotBase + 4;
const InlineSpecifierASTSlotBase = ConstexprSpecifierASTSlotBase + 4;
const NoreturnSpecifierASTSlotBase = InlineSpecifierASTSlotBase + 4;
const StaticSpecifierASTSlotBase = NoreturnSpecifierASTSlotBase + 4;
const ExternSpecifierASTSlotBase = StaticSpecifierASTSlotBase + 4;
const RegisterSpecifierASTSlotBase = ExternSpecifierASTSlotBase + 4;
const ThreadLocalSpecifierASTSlotBase = RegisterSpecifierASTSlotBase + 4;
const ThreadSpecifierASTSlotBase = ThreadLocalSpecifierASTSlotBase + 4;
const MutableSpecifierASTSlotBase = ThreadSpecifierASTSlotBase + 4;
const VirtualSpecifierASTSlotBase = MutableSpecifierASTSlotBase + 4;
const ExplicitSpecifierASTSlotBase = VirtualSpecifierASTSlotBase + 4;
const AutoTypeSpecifierASTSlotBase = ExplicitSpecifierASTSlotBase + 7;
const VoidTypeSpecifierASTSlotBase = AutoTypeSpecifierASTSlotBase + 4;
const SizeTypeSpecifierASTSlotBase = VoidTypeSpecifierASTSlotBase + 4;
const SignTypeSpecifierASTSlotBase = SizeTypeSpecifierASTSlotBase + 5;
const BuiltinTypeSpecifierASTSlotBase = SignTypeSpecifierASTSlotBase + 5;
const UnaryBuiltinTypeSpecifierASTSlotBase =
  BuiltinTypeSpecifierASTSlotBase + 5;
const BinaryBuiltinTypeSpecifierASTSlotBase =
  UnaryBuiltinTypeSpecifierASTSlotBase + 8;
const IntegralTypeSpecifierASTSlotBase =
  BinaryBuiltinTypeSpecifierASTSlotBase + 10;
const FloatingPointTypeSpecifierASTSlotBase =
  IntegralTypeSpecifierASTSlotBase + 5;
const ComplexTypeSpecifierASTSlotBase =
  FloatingPointTypeSpecifierASTSlotBase + 5;
const NamedTypeSpecifierASTSlotBase = ComplexTypeSpecifierASTSlotBase + 4;
const AtomicTypeSpecifierASTSlotBase = NamedTypeSpecifierASTSlotBase + 8;
const BitIntTypeSpecifierASTSlotBase = AtomicTypeSpecifierASTSlotBase + 7;
const UnderlyingTypeSpecifierASTSlotBase = BitIntTypeSpecifierASTSlotBase + 8;
const ElaboratedTypeSpecifierASTSlotBase =
  UnderlyingTypeSpecifierASTSlotBase + 7;
const DecltypeAutoSpecifierASTSlotBase =
  ElaboratedTypeSpecifierASTSlotBase + 11;
const DecltypeSpecifierASTSlotBase = DecltypeAutoSpecifierASTSlotBase + 7;
const PlaceholderTypeSpecifierASTSlotBase = DecltypeSpecifierASTSlotBase + 8;
const ConstQualifierASTSlotBase = PlaceholderTypeSpecifierASTSlotBase + 5;
const VolatileQualifierASTSlotBase = ConstQualifierASTSlotBase + 4;
const AtomicQualifierASTSlotBase = VolatileQualifierASTSlotBase + 4;
const RestrictQualifierASTSlotBase = AtomicQualifierASTSlotBase + 4;
const EnumSpecifierASTSlotBase = RestrictQualifierASTSlotBase + 4;
const ClassSpecifierASTSlotBase = EnumSpecifierASTSlotBase + 15;
const TypenameSpecifierASTSlotBase = ClassSpecifierASTSlotBase + 16;
const SplicerTypeSpecifierASTSlotBase = TypenameSpecifierASTSlotBase + 9;
const PointerOperatorASTSlotBase = SplicerTypeSpecifierASTSlotBase + 5;
const ReferenceOperatorASTSlotBase = PointerOperatorASTSlotBase + 6;
const PtrToMemberOperatorASTSlotBase = ReferenceOperatorASTSlotBase + 6;
const BitfieldDeclaratorASTSlotBase = PtrToMemberOperatorASTSlotBase + 7;
const ParameterPackASTSlotBase = BitfieldDeclaratorASTSlotBase + 6;
const IdDeclaratorASTSlotBase = ParameterPackASTSlotBase + 5;
const NestedDeclaratorASTSlotBase = IdDeclaratorASTSlotBase + 8;
const FunctionDeclaratorChunkASTSlotBase = NestedDeclaratorASTSlotBase + 6;
const ArrayDeclaratorChunkASTSlotBase = FunctionDeclaratorChunkASTSlotBase + 15;
const NameIdASTSlotBase = ArrayDeclaratorChunkASTSlotBase + 8;
const DestructorIdASTSlotBase = NameIdASTSlotBase + 5;
const DecltypeIdASTSlotBase = DestructorIdASTSlotBase + 5;
const OperatorFunctionIdASTSlotBase = DecltypeIdASTSlotBase + 4;
const LiteralOperatorIdASTSlotBase = OperatorFunctionIdASTSlotBase + 8;
const ConversionFunctionIdASTSlotBase = LiteralOperatorIdASTSlotBase + 8;
const SimpleTemplateIdASTSlotBase = ConversionFunctionIdASTSlotBase + 5;
const LiteralOperatorTemplateIdASTSlotBase = SimpleTemplateIdASTSlotBase + 9;
const OperatorFunctionTemplateIdASTSlotBase =
  LiteralOperatorTemplateIdASTSlotBase + 7;
const GlobalNestedNameSpecifierASTSlotBase =
  OperatorFunctionTemplateIdASTSlotBase + 7;
const SimpleNestedNameSpecifierASTSlotBase =
  GlobalNestedNameSpecifierASTSlotBase + 5;
const DecltypeNestedNameSpecifierASTSlotBase =
  SimpleNestedNameSpecifierASTSlotBase + 8;
const TemplateNestedNameSpecifierASTSlotBase =
  DecltypeNestedNameSpecifierASTSlotBase + 6;
const DefaultFunctionBodyASTSlotBase =
  TemplateNestedNameSpecifierASTSlotBase + 9;
const CompoundStatementFunctionBodyASTSlotBase =
  DefaultFunctionBodyASTSlotBase + 6;
const TryStatementFunctionBodyASTSlotBase =
  CompoundStatementFunctionBodyASTSlotBase + 6;
const DeleteFunctionBodyASTSlotBase = TryStatementFunctionBodyASTSlotBase + 8;
const TypeTemplateArgumentASTSlotBase = DeleteFunctionBodyASTSlotBase + 6;
const ExpressionTemplateArgumentASTSlotBase =
  TypeTemplateArgumentASTSlotBase + 4;
const ThrowExceptionSpecifierASTSlotBase =
  ExpressionTemplateArgumentASTSlotBase + 4;
const NoexceptSpecifierASTSlotBase = ThrowExceptionSpecifierASTSlotBase + 6;
const SimpleRequirementASTSlotBase = NoexceptSpecifierASTSlotBase + 7;
const CompoundRequirementASTSlotBase = SimpleRequirementASTSlotBase + 5;
const TypeRequirementASTSlotBase = CompoundRequirementASTSlotBase + 10;
const NestedRequirementASTSlotBase = TypeRequirementASTSlotBase + 9;
const NewParenInitializerASTSlotBase = NestedRequirementASTSlotBase + 6;
const NewBracedInitializerASTSlotBase = NewParenInitializerASTSlotBase + 6;
const ParenMemInitializerASTSlotBase = NewBracedInitializerASTSlotBase + 4;
const BracedMemInitializerASTSlotBase = ParenMemInitializerASTSlotBase + 11;
const ThisLambdaCaptureASTSlotBase = BracedMemInitializerASTSlotBase + 9;
const DerefThisLambdaCaptureASTSlotBase = ThisLambdaCaptureASTSlotBase + 6;
const SimpleLambdaCaptureASTSlotBase = DerefThisLambdaCaptureASTSlotBase + 6;
const RefLambdaCaptureASTSlotBase = SimpleLambdaCaptureASTSlotBase + 8;
const RefInitLambdaCaptureASTSlotBase = RefLambdaCaptureASTSlotBase + 9;
const InitLambdaCaptureASTSlotBase = RefInitLambdaCaptureASTSlotBase + 9;
const EllipsisExceptionDeclarationASTSlotBase =
  InitLambdaCaptureASTSlotBase + 8;
const TypeExceptionDeclarationASTSlotBase =
  EllipsisExceptionDeclarationASTSlotBase + 4;
const CxxAttributeASTSlotBase = TypeExceptionDeclarationASTSlotBase + 7;
const GccAttributeASTSlotBase = CxxAttributeASTSlotBase + 10;
const AlignasAttributeASTSlotBase = GccAttributeASTSlotBase + 10;
const AlignasTypeAttributeASTSlotBase = AlignasAttributeASTSlotBase + 10;
const AsmAttributeASTSlotBase = AlignasTypeAttributeASTSlotBase + 10;
const ScopedAttributeTokenASTSlotBase = AsmAttributeASTSlotBase + 9;
const SimpleAttributeTokenASTSlotBase = ScopedAttributeTokenASTSlotBase + 8;
const LiteralSlotBase = 0;
const IntegerLiteralSlotBase = LiteralSlotBase + 2;
const FloatLiteralSlotBase = IntegerLiteralSlotBase + 4;
const StringLiteralSlotBase = FloatLiteralSlotBase + 4;
const CharLiteralSlotBase = StringLiteralSlotBase + 7;
const CommentLiteralSlotBase = CharLiteralSlotBase + 4;
const NameSlotBase = 0;
const IdentifierSlotBase = NameSlotBase + 2;
const OperatorIdSlotBase = IdentifierSlotBase + 10;
const DestructorIdSlotBase = OperatorIdSlotBase + 3;
const LiteralOperatorIdSlotBase = DestructorIdSlotBase + 3;
const ConversionFunctionIdSlotBase = LiteralOperatorIdSlotBase + 3;
const TemplateIdSlotBase = ConversionFunctionIdSlotBase + 3;
const SymbolSlotBase = 0;
const ScopeSymbolSlotBase = SymbolSlotBase + 54;
const NamespaceSymbolSlotBase = ScopeSymbolSlotBase + 58;
const ConceptSymbolSlotBase = NamespaceSymbolSlotBase + 62;
const DeductionGuideSymbolSlotBase = ConceptSymbolSlotBase + 63;
const BaseClassSymbolSlotBase = DeductionGuideSymbolSlotBase + 64;
const InjectedClassNameSymbolSlotBase = BaseClassSymbolSlotBase + 56;
const UnresolvedSymbolSlotBase = InjectedClassNameSymbolSlotBase + 55;
const ClassSymbolSlotBase = UnresolvedSymbolSlotBase + 54;
const EnumSymbolSlotBase = ClassSymbolSlotBase + 115;
const ScopedEnumSymbolSlotBase = EnumSymbolSlotBase + 61;
const FunctionSymbolSlotBase = ScopedEnumSymbolSlotBase + 60;
const OverloadSetSymbolSlotBase = FunctionSymbolSlotBase + 121;
const LambdaSymbolSlotBase = OverloadSetSymbolSlotBase + 57;
const FunctionParametersSymbolSlotBase = LambdaSymbolSlotBase + 65;
const TemplateParametersSymbolSlotBase = FunctionParametersSymbolSlotBase + 58;
const BlockSymbolSlotBase = TemplateParametersSymbolSlotBase + 59;
const TypeAliasSymbolSlotBase = BlockSymbolSlotBase + 59;
const VariableSymbolSlotBase = TypeAliasSymbolSlotBase + 67;
const FieldSymbolSlotBase = VariableSymbolSlotBase + 76;
const ParameterSymbolSlotBase = FieldSymbolSlotBase + 74;
const ParameterPackSymbolSlotBase = ParameterSymbolSlotBase + 56;
const TypeParameterSymbolSlotBase = ParameterPackSymbolSlotBase + 55;
const NonTypeParameterSymbolSlotBase = TypeParameterSymbolSlotBase + 54;
const TemplateTypeParameterSymbolSlotBase = NonTypeParameterSymbolSlotBase + 57;
const ConstraintTypeParameterSymbolSlotBase =
  TemplateTypeParameterSymbolSlotBase + 54;
const EnumeratorSymbolSlotBase = ConstraintTypeParameterSymbolSlotBase + 58;
const NamespaceAliasSymbolSlotBase = EnumeratorSymbolSlotBase + 55;
const UsingDeclarationSymbolSlotBase = NamespaceAliasSymbolSlotBase + 55;
const TypeSlotBase = 0;
const BuiltinVaListTypeSlotBase = TypeSlotBase + 1;
const BuiltinMetaInfoTypeSlotBase = BuiltinVaListTypeSlotBase + 1;
const VoidTypeSlotBase = BuiltinMetaInfoTypeSlotBase + 1;
const NullptrTypeSlotBase = VoidTypeSlotBase + 1;
const DecltypeAutoTypeSlotBase = NullptrTypeSlotBase + 1;
const AutoTypeSlotBase = DecltypeAutoTypeSlotBase + 1;
const BoolTypeSlotBase = AutoTypeSlotBase + 1;
const SignedCharTypeSlotBase = BoolTypeSlotBase + 1;
const ShortIntTypeSlotBase = SignedCharTypeSlotBase + 1;
const IntTypeSlotBase = ShortIntTypeSlotBase + 1;
const LongIntTypeSlotBase = IntTypeSlotBase + 1;
const LongLongIntTypeSlotBase = LongIntTypeSlotBase + 1;
const Int128TypeSlotBase = LongLongIntTypeSlotBase + 1;
const UnsignedCharTypeSlotBase = Int128TypeSlotBase + 1;
const UnsignedShortIntTypeSlotBase = UnsignedCharTypeSlotBase + 1;
const UnsignedIntTypeSlotBase = UnsignedShortIntTypeSlotBase + 1;
const UnsignedLongIntTypeSlotBase = UnsignedIntTypeSlotBase + 1;
const UnsignedLongLongIntTypeSlotBase = UnsignedLongIntTypeSlotBase + 1;
const UnsignedInt128TypeSlotBase = UnsignedLongLongIntTypeSlotBase + 1;
const CharTypeSlotBase = UnsignedInt128TypeSlotBase + 1;
const Char8TypeSlotBase = CharTypeSlotBase + 1;
const Char16TypeSlotBase = Char8TypeSlotBase + 1;
const Char32TypeSlotBase = Char16TypeSlotBase + 1;
const WideCharTypeSlotBase = Char32TypeSlotBase + 1;
const FloatTypeSlotBase = WideCharTypeSlotBase + 1;
const DoubleTypeSlotBase = FloatTypeSlotBase + 1;
const LongDoubleTypeSlotBase = DoubleTypeSlotBase + 1;
const Float16TypeSlotBase = LongDoubleTypeSlotBase + 1;
const QualTypeSlotBase = Float16TypeSlotBase + 1;
const BoundedArrayTypeSlotBase = QualTypeSlotBase + 5;
const UnboundedArrayTypeSlotBase = BoundedArrayTypeSlotBase + 3;
const PointerTypeSlotBase = UnboundedArrayTypeSlotBase + 2;
const LvalueReferenceTypeSlotBase = PointerTypeSlotBase + 2;
const RvalueReferenceTypeSlotBase = LvalueReferenceTypeSlotBase + 2;
const OverloadSetTypeSlotBase = RvalueReferenceTypeSlotBase + 2;
const FunctionTypeSlotBase = OverloadSetTypeSlotBase + 2;
const ClassTypeSlotBase = FunctionTypeSlotBase + 7;
const EnumTypeSlotBase = ClassTypeSlotBase + 5;
const ScopedEnumTypeSlotBase = EnumTypeSlotBase + 3;
const MemberObjectPointerTypeSlotBase = ScopedEnumTypeSlotBase + 3;
const MemberFunctionPointerTypeSlotBase = MemberObjectPointerTypeSlotBase + 3;
const NamespaceTypeSlotBase = MemberFunctionPointerTypeSlotBase + 3;
const TypeParameterTypeSlotBase = NamespaceTypeSlotBase + 2;
const TemplateTypeParameterTypeSlotBase = TypeParameterTypeSlotBase + 4;
const UnresolvedNameTypeSlotBase = TemplateTypeParameterTypeSlotBase + 5;
const UnresolvedBoundedArrayTypeSlotBase = UnresolvedNameTypeSlotBase + 4;
const UnresolvedUnderlyingTypeSlotBase = UnresolvedBoundedArrayTypeSlotBase + 3;
const UnresolvedBuiltinTypeSlotBase = UnresolvedUnderlyingTypeSlotBase + 2;
const BitIntTypeSlotBase = UnresolvedBuiltinTypeSlotBase + 3;
const UnsignedBitIntTypeSlotBase = BitIntTypeSlotBase + 2;
const UnresolvedBitIntTypeSlotBase = UnsignedBitIntTypeSlotBase + 2;
const VectorTypeSlotBase = UnresolvedBitIntTypeSlotBase + 3;
const UnresolvedVectorTypeSlotBase = VectorTypeSlotBase + 4;
const ComplexTypeSlotBase = UnresolvedVectorTypeSlotBase + 5;
const AtomicTypeSlotBase = ComplexTypeSlotBase + 2;
export class DefaultInitializerContext extends ModelObject {}
export class InitializerList extends ModelObject {}
export class ConstComplex extends ModelObject {
  get real():
    | { readonly index: 0; readonly value: bigint }
    | { readonly index: 1; readonly value: StringLiteral | undefined }
    | { readonly index: 2; readonly value: number }
    | { readonly index: 3; readonly value: number }
    | { readonly index: 4; readonly value: number }
    | { readonly index: 5; readonly value: Meta | undefined }
    | { readonly index: 6; readonly value: InitializerList | undefined }
    | { readonly index: 7; readonly value: ConstObject | undefined }
    | { readonly index: 8; readonly value: ConstAddress | undefined }
    | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
    | { readonly index: 10; readonly value: ConstComplex | undefined }
    | { readonly index: 11; readonly value: {} } {
    return ((item: any) =>
      item.index === 1
        ? { index: 1, value: objOf(item.value, this.modelOwner, StringLiteral) }
        : item.index === 5
          ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
          : item.index === 6
            ? {
                index: 6,
                value: objOf(item.value, this.modelOwner, InitializerList),
              }
            : item.index === 7
              ? {
                  index: 7,
                  value: objOf(item.value, this.modelOwner, ConstObject),
                }
              : item.index === 8
                ? {
                    index: 8,
                    value: objOf(item.value, this.modelOwner, ConstAddress),
                  }
                : item.index === 9
                  ? {
                      index: 9,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        ConstLabelAddress,
                      ),
                    }
                  : item.index === 10
                    ? {
                        index: 10,
                        value: objOf(item.value, this.modelOwner, ConstComplex),
                      }
                    : item)(
      cxx.readMiscVal(this.handle, ConstComplexSlotBase + 0),
    );
  }
  get imag():
    | { readonly index: 0; readonly value: bigint }
    | { readonly index: 1; readonly value: StringLiteral | undefined }
    | { readonly index: 2; readonly value: number }
    | { readonly index: 3; readonly value: number }
    | { readonly index: 4; readonly value: number }
    | { readonly index: 5; readonly value: Meta | undefined }
    | { readonly index: 6; readonly value: InitializerList | undefined }
    | { readonly index: 7; readonly value: ConstObject | undefined }
    | { readonly index: 8; readonly value: ConstAddress | undefined }
    | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
    | { readonly index: 10; readonly value: ConstComplex | undefined }
    | { readonly index: 11; readonly value: {} } {
    return ((item: any) =>
      item.index === 1
        ? { index: 1, value: objOf(item.value, this.modelOwner, StringLiteral) }
        : item.index === 5
          ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
          : item.index === 6
            ? {
                index: 6,
                value: objOf(item.value, this.modelOwner, InitializerList),
              }
            : item.index === 7
              ? {
                  index: 7,
                  value: objOf(item.value, this.modelOwner, ConstObject),
                }
              : item.index === 8
                ? {
                    index: 8,
                    value: objOf(item.value, this.modelOwner, ConstAddress),
                  }
                : item.index === 9
                  ? {
                      index: 9,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        ConstLabelAddress,
                      ),
                    }
                  : item.index === 10
                    ? {
                        index: 10,
                        value: objOf(item.value, this.modelOwner, ConstComplex),
                      }
                    : item)(
      cxx.readMiscVal(this.handle, ConstComplexSlotBase + 1),
    );
  }
}
export class ConstObject extends ModelObject {
  get type(): Type | undefined {
    return typeOf(
      cxx.readMisc(this.handle, ConstObjectSlotBase + 0),
      this.modelOwner,
    );
  }
  get members(): Iterable<{
    readonly symbol: Symbol | undefined;
    readonly value:
      | { readonly index: 0; readonly value: bigint }
      | { readonly index: 1; readonly value: StringLiteral | undefined }
      | { readonly index: 2; readonly value: number }
      | { readonly index: 3; readonly value: number }
      | { readonly index: 4; readonly value: number }
      | { readonly index: 5; readonly value: Meta | undefined }
      | { readonly index: 6; readonly value: InitializerList | undefined }
      | { readonly index: 7; readonly value: ConstObject | undefined }
      | { readonly index: 8; readonly value: ConstAddress | undefined }
      | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
      | { readonly index: 10; readonly value: ConstComplex | undefined }
      | { readonly index: 11; readonly value: {} };
  }> {
    return miscValItems(
      this.modelOwner,
      this.handle,
      ConstObjectSlotBase + 1,
      (item: any) =>
        ((item: any) => ({
          symbol: symbolOf(item.symbol, this.modelOwner),
          value: ((item: any) =>
            item.index === 1
              ? {
                  index: 1,
                  value: objOf(item.value, this.modelOwner, StringLiteral),
                }
              : item.index === 5
                ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
                : item.index === 6
                  ? {
                      index: 6,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        InitializerList,
                      ),
                    }
                  : item.index === 7
                    ? {
                        index: 7,
                        value: objOf(item.value, this.modelOwner, ConstObject),
                      }
                    : item.index === 8
                      ? {
                          index: 8,
                          value: objOf(
                            item.value,
                            this.modelOwner,
                            ConstAddress,
                          ),
                        }
                      : item.index === 9
                        ? {
                            index: 9,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              ConstLabelAddress,
                            ),
                          }
                        : item.index === 10
                          ? {
                              index: 10,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                ConstComplex,
                              ),
                            }
                          : item)(item.value),
        }))(item),
    );
  }
  get isUnion(): boolean {
    return cxx.readMisc(this.handle, ConstObjectSlotBase + 2) !== 0;
  }
}
export class Meta extends ModelObject {}
export class ConstAddress extends ModelObject {
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readMisc(this.handle, ConstAddressSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeInfoFor(): Type | undefined {
    return typeOf(
      cxx.readMisc(this.handle, ConstAddressSlotBase + 1),
      this.modelOwner,
    );
  }
  get owner(): ConstObject | undefined {
    return objOf(
      cxx.readMisc(this.handle, ConstAddressSlotBase + 2),
      this.modelOwner,
      ConstObject,
    );
  }
  get stringLiteral(): StringLiteral | undefined {
    return objOf(
      cxx.readMisc(this.handle, ConstAddressSlotBase + 3),
      this.modelOwner,
      StringLiteral,
    );
  }
  get offset(): bigint {
    return cxx.readMiscBigInt(this.handle, ConstAddressSlotBase + 4) as bigint;
  }
}
export class ConstLabelAddress extends ModelObject {
  get name(): string {
    return cxx.readMiscString(
      this.handle,
      ConstLabelAddressSlotBase + 0,
    ) as string;
  }
}
export abstract class AST extends ModelObject {
  readonly kind: ASTKind;
  constructor(handle: number, owner: ModelOwner, kind: ASTKind) {
    super(handle, owner);
    this.kind = kind;
  }
  abstract accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result;
  get startLocation(): SourceLocation | undefined {
    return cxx.getStartLocation(this.handle, this.modelOwner.getUnitHandle());
  }
  get endLocation(): SourceLocation | undefined {
    return cxx.getEndLocation(this.handle, this.modelOwner.getUnitHandle());
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ASTSlotBase + 2);
  }
}
export abstract class AttributeSpecifierAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, AttributeSpecifierASTSlotBase + 0);
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readASTVal(this.handle, AttributeSpecifierASTSlotBase + 1));
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeSpecifierASTSlotBase + 3);
  }
}
export abstract class AttributeTokenAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, AttributeTokenASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeTokenASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeTokenASTSlotBase + 2);
  }
}
export abstract class CoreDeclaratorAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, CoreDeclaratorASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CoreDeclaratorASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CoreDeclaratorASTSlotBase + 2);
  }
}
export abstract class DeclarationAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, DeclarationASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DeclarationASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DeclarationASTSlotBase + 2);
  }
}
export abstract class DeclaratorChunkAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, DeclaratorChunkASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DeclaratorChunkASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DeclaratorChunkASTSlotBase + 2);
  }
}
export abstract class DesignatorAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, DesignatorASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DesignatorASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DesignatorASTSlotBase + 2);
  }
}
export abstract class ExceptionDeclarationAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, ExceptionDeclarationASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExceptionDeclarationASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExceptionDeclarationASTSlotBase + 2);
  }
}
export abstract class ExceptionSpecifierAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, ExceptionSpecifierASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExceptionSpecifierASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExceptionSpecifierASTSlotBase + 2);
  }
}
export abstract class ExpressionAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, ExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(this.handle, ExpressionASTSlotBase + 1) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExpressionASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExpressionASTSlotBase + 4);
  }
}
export abstract class FunctionBodyAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, FunctionBodyASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, FunctionBodyASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, FunctionBodyASTSlotBase + 2);
  }
}
export abstract class GenericAssociationAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, GenericAssociationASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, GenericAssociationASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, GenericAssociationASTSlotBase + 2);
  }
}
export abstract class LambdaCaptureAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, LambdaCaptureASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LambdaCaptureASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LambdaCaptureASTSlotBase + 2);
  }
}
export abstract class MemInitializerAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, MemInitializerASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, MemInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, MemInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, MemInitializerASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, MemInitializerASTSlotBase + 4);
  }
}
export abstract class NestedNameSpecifierAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, NestedNameSpecifierASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NestedNameSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NestedNameSpecifierASTSlotBase + 3);
  }
}
export abstract class NewInitializerAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, NewInitializerASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NewInitializerASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NewInitializerASTSlotBase + 2);
  }
}
export abstract class PtrOperatorAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, PtrOperatorASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PtrOperatorASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PtrOperatorASTSlotBase + 2);
  }
}
export abstract class RequirementAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, RequirementASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RequirementASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RequirementASTSlotBase + 2);
  }
}
export abstract class SpecifierAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, SpecifierASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SpecifierASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SpecifierASTSlotBase + 2);
  }
}
export abstract class StatementAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, StatementASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, StatementASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, StatementASTSlotBase + 2);
  }
}
export abstract class TemplateArgumentAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, TemplateArgumentASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateArgumentASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateArgumentASTSlotBase + 2);
  }
}
export abstract class TemplateParameterAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, TemplateParameterASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TemplateParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, TemplateParameterASTSlotBase + 2);
  }
  get index(): number {
    return cxx.readAST(this.handle, TemplateParameterASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateParameterASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateParameterASTSlotBase + 5);
  }
}
export abstract class UnitAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, UnitASTSlotBase + 0);
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, UnitASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UnitASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UnitASTSlotBase + 3);
  }
}
export abstract class UnqualifiedIdAST extends AST {
  get internalId(): number {
    return cxx.readAST(this.handle, UnqualifiedIdASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UnqualifiedIdASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UnqualifiedIdASTSlotBase + 2);
  }
}
export class TranslationUnitAST extends UnitAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTranslationUnit(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TranslationUnitASTSlotBase + 0);
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TranslationUnitASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TranslationUnitASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TranslationUnitASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TranslationUnitASTSlotBase + 4);
  }
}
export class ModuleUnitAST extends UnitAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleUnit(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ModuleUnitASTSlotBase + 0);
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get globalModuleFragment(): GlobalModuleFragmentAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get moduleDeclaration(): ModuleDeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get privateModuleFragment(): PrivateModuleFragmentAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleUnitASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleUnitASTSlotBase + 7);
  }
}
export class SimpleDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get initDeclaratorList(): Iterable<InitDeclaratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 5);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 7);
  }
}
export class AsmDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get asmQualifierList(): Iterable<AsmQualifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get asmLoc(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 4);
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 5);
  }
  get outputOperandList(): Iterable<AsmOperandAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get inputOperandList(): Iterable<AsmOperandAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get clobberList(): Iterable<AsmClobberAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get gotoLabelList(): Iterable<AsmGotoLabelAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 9),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 10);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 11);
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 12),
      this.modelOwner,
      Literal,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 13);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 14);
  }
}
export class NamespaceAliasDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceAliasDefinition(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 0);
  }
  get namespaceLoc(): number {
    return cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 1);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 2);
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 3);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 6);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceAliasSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 10);
  }
}
export class UsingDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 0);
  }
  get usingLoc(): number {
    return cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 1);
  }
  get usingDeclaratorList(): Iterable<UsingDeclaratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 5);
  }
}
export class UsingEnumDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingEnumDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 0);
  }
  get usingLoc(): number {
    return cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 1);
  }
  get enumTypeSpecifier(): ElaboratedTypeSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 5);
  }
}
export class UsingDirectiveAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDirective(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get usingLoc(): number {
    return cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 2);
  }
  get namespaceLoc(): number {
    return cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 3);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 8);
  }
}
export class StaticAssertDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStaticAssertDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 0);
  }
  get staticAssertLoc(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 4);
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 5);
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 6),
      this.modelOwner,
      Literal,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 7);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 8);
  }
  get value(): boolean | undefined {
    return ((item: any) => (item === undefined ? undefined : item !== 0))(
      cxx.readASTVal(this.handle, StaticAssertDeclarationASTSlotBase + 9),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 11);
  }
}
export class AliasDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAliasDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 0);
  }
  get usingLoc(): number {
    return cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 1);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 2);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 4);
  }
  get gnuAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 7);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get symbol(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 11);
  }
}
export class OpaqueEnumDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOpaqueEnumDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 0);
  }
  get enumLoc(): number {
    return cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 1);
  }
  get classLoc(): number {
    return cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 2);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 6);
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get emicolonLoc(): number {
    return cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 8);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 11);
  }
}
export class FunctionDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFunctionDefinition(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get functionBody(): FunctionBodyAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 8);
  }
}
export class TemplateDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 0);
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 1);
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 2);
  }
  get templateParameterList(): Iterable<TemplateParameterAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 4);
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 8);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 10);
  }
}
export class ConceptDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConceptDefinition(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 0);
  }
  get conceptLoc(): number {
    return cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 1);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 2);
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 5);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): ConceptSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 9);
  }
}
export class DeductionGuideAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeductionGuide(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get explicitSpecifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 4);
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 6);
  }
  get arrowLoc(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 7);
  }
  get templateId(): SimpleTemplateIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 9);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get symbol(): DeductionGuideSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 12);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DeductionGuideASTSlotBase + 13);
  }
}
export class ExplicitInstantiationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExplicitInstantiation(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 0);
  }
  get externLoc(): number {
    return cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 1);
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 2);
  }
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 5);
  }
}
export class ExportDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExportDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ExportDeclarationASTSlotBase + 0);
  }
  get exportLoc(): number {
    return cxx.readAST(this.handle, ExportDeclarationASTSlotBase + 1);
  }
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExportDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExportDeclarationASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExportDeclarationASTSlotBase + 4);
  }
}
export class ExportCompoundDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExportCompoundDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 0);
  }
  get exportLoc(): number {
    return cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 1);
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 2);
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 6);
  }
}
export class LinkageSpecificationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLinkageSpecification(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 0);
  }
  get externLoc(): number {
    return cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 1);
  }
  get stringliteralLoc(): number {
    return cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 2);
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 3);
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 5);
  }
  get stringLiteral(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 6),
      this.modelOwner,
      StringLiteral,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 8);
  }
}
export class NamespaceDefinitionAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceDefinition(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 0);
  }
  get inlineLoc(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 1);
  }
  get namespaceLoc(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 2);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNamespaceSpecifierList(): Iterable<
    NestedNamespaceSpecifierAST | undefined
  > {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 5);
  }
  get extraAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 7);
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 9);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get isInline(): boolean {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 12) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 13);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 14);
  }
}
export class EmptyDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEmptyDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, EmptyDeclarationASTSlotBase + 0);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, EmptyDeclarationASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, EmptyDeclarationASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, EmptyDeclarationASTSlotBase + 3);
  }
}
export class AttributeDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AttributeDeclarationASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AttributeDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, AttributeDeclarationASTSlotBase + 2);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeDeclarationASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeDeclarationASTSlotBase + 4);
  }
}
export class ModuleImportDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleImportDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 0);
  }
  get importLoc(): number {
    return cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 1);
  }
  get importName(): ImportNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 6);
  }
}
export class ParameterDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get thisLoc(): number {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 2);
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 5);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get symbol(): ParameterSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get isThisIntroduced(): boolean {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 10) !== 0;
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 11) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 12);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 13);
  }
}
export class AccessDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAccessDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 0);
  }
  get accessLoc(): number {
    return cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 1);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 2);
  }
  get accessSpecifier(): TokenKind {
    return cxx.readAST(
      this.handle,
      AccessDeclarationASTSlotBase + 3,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 5);
  }
}
export class ForRangeDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForRangeDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ForRangeDeclarationASTSlotBase + 0);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ForRangeDeclarationASTSlotBase + 1);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ForRangeDeclarationASTSlotBase + 2);
  }
}
export class StructuredBindingDeclarationAST extends DeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStructuredBindingDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      StructuredBindingDeclarationASTSlotBase + 0,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get refQualifierLoc(): number {
    return cxx.readAST(
      this.handle,
      StructuredBindingDeclarationASTSlotBase + 3,
    );
  }
  get lbracketLoc(): number {
    return cxx.readAST(
      this.handle,
      StructuredBindingDeclarationASTSlotBase + 4,
    );
  }
  get bindingList(): Iterable<NameIdAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbracketLoc(): number {
    return cxx.readAST(
      this.handle,
      StructuredBindingDeclarationASTSlotBase + 6,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(
      this.handle,
      StructuredBindingDeclarationASTSlotBase + 8,
    );
  }
  get hiddenVariable(): InitDeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get bindingDeclaratorList(): Iterable<InitDeclaratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 10),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      StructuredBindingDeclarationASTSlotBase + 11,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      StructuredBindingDeclarationASTSlotBase + 12,
    );
  }
}
export class AsmOperandAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmOperand(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 0);
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 1);
  }
  get symbolicNameLoc(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 2);
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 3);
  }
  get constraintLiteralLoc(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 4);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 5);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 7);
  }
  get symbolicName(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get constraintLiteral(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 9),
      this.modelOwner,
      Literal,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AsmOperandASTSlotBase + 11);
  }
}
export class AsmQualifierAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmQualifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AsmQualifierASTSlotBase + 0);
  }
  get qualifierLoc(): number {
    return cxx.readAST(this.handle, AsmQualifierASTSlotBase + 1);
  }
  get qualifier(): TokenKind {
    return cxx.readAST(this.handle, AsmQualifierASTSlotBase + 2) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AsmQualifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AsmQualifierASTSlotBase + 4);
  }
}
export class AsmClobberAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmClobber(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AsmClobberASTSlotBase + 0);
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, AsmClobberASTSlotBase + 1);
  }
  get literal(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmClobberASTSlotBase + 2),
      this.modelOwner,
      StringLiteral,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AsmClobberASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AsmClobberASTSlotBase + 4);
  }
}
export class AsmGotoLabelAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmGotoLabel(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AsmGotoLabelASTSlotBase + 0);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, AsmGotoLabelASTSlotBase + 1);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, AsmGotoLabelASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AsmGotoLabelASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AsmGotoLabelASTSlotBase + 4);
  }
}
export class SplicerAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSplicer(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 0);
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 1);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 2);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SplicerASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get secondColonLoc(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 5);
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SplicerASTSlotBase + 8);
  }
}
export class GlobalModuleFragmentAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalModuleFragment(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 0);
  }
  get moduleLoc(): number {
    return cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 1);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 2);
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 5);
  }
}
export class PrivateModuleFragmentAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPrivateModuleFragment(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 0);
  }
  get moduleLoc(): number {
    return cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 1);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 2);
  }
  get privateLoc(): number {
    return cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 3);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 4);
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 7);
  }
}
export class ModuleDeclarationAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 0);
  }
  get exportLoc(): number {
    return cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 1);
  }
  get moduleLoc(): number {
    return cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 2);
  }
  get moduleName(): ModuleNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get modulePartition(): ModulePartitionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 8);
  }
}
export class ModuleNameAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleName(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ModuleNameASTSlotBase + 0);
  }
  get moduleQualifier(): ModuleQualifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleNameASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, ModuleNameASTSlotBase + 2);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ModuleNameASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleNameASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleNameASTSlotBase + 5);
  }
}
export class ModuleQualifierAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModuleQualifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 0);
  }
  get moduleQualifier(): ModuleQualifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 2);
  }
  get dotLoc(): number {
    return cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 3);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 6);
  }
}
export class ModulePartitionAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitModulePartition(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ModulePartitionASTSlotBase + 0);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, ModulePartitionASTSlotBase + 1);
  }
  get moduleName(): ModuleNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModulePartitionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ModulePartitionASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ModulePartitionASTSlotBase + 4);
  }
}
export class ImportNameAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitImportName(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ImportNameASTSlotBase + 0);
  }
  get headerLoc(): number {
    return cxx.readAST(this.handle, ImportNameASTSlotBase + 1);
  }
  get modulePartition(): ModulePartitionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ImportNameASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get moduleName(): ModuleNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ImportNameASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ImportNameASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ImportNameASTSlotBase + 5);
  }
}
export class InitDeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInitDeclarator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 0);
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 6);
  }
}
export class DeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeclarator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DeclaratorASTSlotBase + 0);
  }
  get ptrOpList(): Iterable<PtrOperatorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DeclaratorASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get coreDeclarator(): CoreDeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get declaratorChunkList(): Iterable<DeclaratorChunkAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DeclaratorASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DeclaratorASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DeclaratorASTSlotBase + 5);
  }
}
export class UsingDeclaratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUsingDeclarator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 0);
  }
  get typenameLoc(): number {
    return cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 1);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 4);
  }
  get symbol(): UsingDeclarationSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 6) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 8);
  }
}
export class EnumeratorAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEnumerator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, EnumeratorASTSlotBase + 0);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, EnumeratorASTSlotBase + 1);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, EnumeratorASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): EnumeratorSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, EnumeratorASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, EnumeratorASTSlotBase + 8);
  }
}
export class TypeIdAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeIdASTSlotBase + 0);
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeIdASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeIdASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeIdASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TypeIdASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeIdASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeIdASTSlotBase + 6);
  }
}
export class HandlerAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitHandler(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, HandlerASTSlotBase + 0);
  }
  get catchLoc(): number {
    return cxx.readAST(this.handle, HandlerASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, HandlerASTSlotBase + 2);
  }
  get exceptionDeclaration(): ExceptionDeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, HandlerASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, HandlerASTSlotBase + 4);
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, HandlerASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, HandlerASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, HandlerASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, HandlerASTSlotBase + 8);
  }
}
export class BaseSpecifierAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBaseSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get virtualOrAccessLoc(): number {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 2);
  }
  get otherVirtualOrAccessLoc(): number {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 3);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 5);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 7);
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 8) !== 0;
  }
  get isVirtual(): boolean {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 9) !== 0;
  }
  get isVariadic(): boolean {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 10) !== 0;
  }
  get accessSpecifier(): TokenKind {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 11) as TokenKind;
  }
  get symbol(): BaseClassSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 12),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 13);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 14);
  }
}
export class RequiresClauseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRequiresClause(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RequiresClauseASTSlotBase + 0);
  }
  get requiresLoc(): number {
    return cxx.readAST(this.handle, RequiresClauseASTSlotBase + 1);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RequiresClauseASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RequiresClauseASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RequiresClauseASTSlotBase + 4);
  }
}
export class ParameterDeclarationClauseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterDeclarationClause(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 0);
  }
  get parameterDeclarationList(): Iterable<
    ParameterDeclarationAST | undefined
  > {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 2);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 3);
  }
  get functionParametersSymbol(): FunctionParametersSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isVariadic(): boolean {
    return (
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 5) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 7);
  }
}
export class TrailingReturnTypeAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTrailingReturnType(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TrailingReturnTypeASTSlotBase + 0);
  }
  get minusGreaterLoc(): number {
    return cxx.readAST(this.handle, TrailingReturnTypeASTSlotBase + 1);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TrailingReturnTypeASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TrailingReturnTypeASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TrailingReturnTypeASTSlotBase + 4);
  }
}
export class LambdaSpecifierAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLambdaSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LambdaSpecifierASTSlotBase + 0);
  }
  get specifierLoc(): number {
    return cxx.readAST(this.handle, LambdaSpecifierASTSlotBase + 1);
  }
  get specifier(): TokenKind {
    return cxx.readAST(
      this.handle,
      LambdaSpecifierASTSlotBase + 2,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LambdaSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LambdaSpecifierASTSlotBase + 4);
  }
}
export class TypeConstraintAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeConstraint(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeConstraintASTSlotBase + 0);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, TypeConstraintASTSlotBase + 2);
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, TypeConstraintASTSlotBase + 3);
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, TypeConstraintASTSlotBase + 5);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): ConceptSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeConstraintASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeConstraintASTSlotBase + 9);
  }
}
export class AttributeArgumentClauseAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeArgumentClause(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 0);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 1);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 5);
  }
}
export class AttributeAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttribute(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AttributeASTSlotBase + 0);
  }
  get attributeToken(): AttributeTokenAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AttributeASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeArgumentClause(): AttributeArgumentClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AttributeASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, AttributeASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeASTSlotBase + 5);
  }
}
export class AttributeUsingPrefixAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAttributeUsingPrefix(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 0);
  }
  get usingLoc(): number {
    return cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 1);
  }
  get attributeNamespaceLoc(): number {
    return cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 2);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 5);
  }
}
export class NewPlacementAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewPlacement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NewPlacementASTSlotBase + 0);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NewPlacementASTSlotBase + 1);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NewPlacementASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NewPlacementASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NewPlacementASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NewPlacementASTSlotBase + 5);
  }
}
export class NestedNamespaceSpecifierAST extends AST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedNamespaceSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 0);
  }
  get inlineLoc(): number {
    return cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 1);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 2);
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 3);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get isInline(): boolean {
    return (
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 6) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 8);
  }
}
export class LabeledStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLabeledStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LabeledStatementASTSlotBase + 0);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, LabeledStatementASTSlotBase + 1);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, LabeledStatementASTSlotBase + 2);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LabeledStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, LabeledStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LabeledStatementASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LabeledStatementASTSlotBase + 6);
  }
}
export class CaseStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCaseStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CaseStatementASTSlotBase + 0);
  }
  get caseLoc(): number {
    return cxx.readAST(this.handle, CaseStatementASTSlotBase + 1);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CaseStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, CaseStatementASTSlotBase + 3);
  }
  get caseValue(): bigint {
    return cxx.readASTBigInt(
      this.handle,
      CaseStatementASTSlotBase + 4,
    ) as bigint;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CaseStatementASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CaseStatementASTSlotBase + 6);
  }
}
export class DefaultStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DefaultStatementASTSlotBase + 0);
  }
  get defaultLoc(): number {
    return cxx.readAST(this.handle, DefaultStatementASTSlotBase + 1);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, DefaultStatementASTSlotBase + 2);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DefaultStatementASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DefaultStatementASTSlotBase + 4);
  }
}
export class ExpressionStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExpressionStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 5);
  }
}
export class CompoundStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CompoundStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, CompoundStatementASTSlotBase + 2);
  }
  get statementList(): Iterable<StatementAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, CompoundStatementASTSlotBase + 4);
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CompoundStatementASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CompoundStatementASTSlotBase + 7);
  }
}
export class IfStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIfStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, IfStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get ifLoc(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 2);
  }
  get constexprLoc(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 4);
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 7);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get elseLoc(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 9);
  }
  get elseStatement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 12);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, IfStatementASTSlotBase + 13);
  }
}
export class ConstevalIfStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstevalIfStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get ifLoc(): number {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 2);
  }
  get exclaimLoc(): number {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 3);
  }
  get constvalLoc(): number {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 4);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get elseLoc(): number {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 6);
  }
  get elseStatement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get isNot(): boolean {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 8) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 10);
  }
}
export class SwitchStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSwitchStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SwitchStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get switchLoc(): number {
    return cxx.readAST(this.handle, SwitchStatementASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, SwitchStatementASTSlotBase + 3);
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, SwitchStatementASTSlotBase + 6);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SwitchStatementASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SwitchStatementASTSlotBase + 10);
  }
}
export class WhileStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitWhileStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, WhileStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get whileLoc(): number {
    return cxx.readAST(this.handle, WhileStatementASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, WhileStatementASTSlotBase + 3);
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, WhileStatementASTSlotBase + 5);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, WhileStatementASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, WhileStatementASTSlotBase + 9);
  }
}
export class DoStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDoStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DoStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get doLoc(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 2);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get whileLoc(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 4);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 5);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 7);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 8);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DoStatementASTSlotBase + 10);
  }
}
export class ForRangeStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForRangeStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get forLoc(): number {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 3);
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rangeDeclaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 6);
  }
  get rangeInitializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 8);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get beginInitializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get endInitializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 12),
      this.modelOwner,
    );
  }
  get increment(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 13),
      this.modelOwner,
    );
  }
  get element(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 14),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 15),
      this.modelOwner,
    );
  }
  get rangeVariable(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 16),
      this.modelOwner,
    );
  }
  get beginVariable(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 17),
      this.modelOwner,
    );
  }
  get endVariable(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 18),
      this.modelOwner,
    );
  }
  get beginFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 19),
      this.modelOwner,
    );
  }
  get endFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 20),
      this.modelOwner,
    );
  }
  get derefFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 21),
      this.modelOwner,
    );
  }
  get incrementFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 22),
      this.modelOwner,
    );
  }
  get notEqualFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 23),
      this.modelOwner,
    );
  }
  get usesMemberBeginEnd(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 24) !== 0;
  }
  get isPointerIterator(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 25) !== 0;
  }
  get notEqualRewritten(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 26) !== 0;
  }
  get notEqualReversed(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 27) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 28);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 29);
  }
}
export class ForStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitForStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ForStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ForStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get forLoc(): number {
    return cxx.readAST(this.handle, ForStatementASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, ForStatementASTSlotBase + 3);
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, ForStatementASTSlotBase + 6);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, ForStatementASTSlotBase + 8);
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ForStatementASTSlotBase + 11);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ForStatementASTSlotBase + 12);
  }
}
export class BreakStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBreakStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BreakStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BreakStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get breakLoc(): number {
    return cxx.readAST(this.handle, BreakStatementASTSlotBase + 2);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, BreakStatementASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BreakStatementASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BreakStatementASTSlotBase + 5);
  }
}
export class ContinueStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitContinueStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ContinueStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ContinueStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get continueLoc(): number {
    return cxx.readAST(this.handle, ContinueStatementASTSlotBase + 2);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, ContinueStatementASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ContinueStatementASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ContinueStatementASTSlotBase + 5);
  }
}
export class ReturnStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReturnStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ReturnStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ReturnStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get returnLoc(): number {
    return cxx.readAST(this.handle, ReturnStatementASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ReturnStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, ReturnStatementASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ReturnStatementASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ReturnStatementASTSlotBase + 6);
  }
}
export class CoroutineReturnStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCoroutineReturnStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get coreturnLoc(): number {
    return cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 6);
  }
}
export class GotoStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGotoStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get gotoLoc(): number {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 3);
  }
  get starLoc(): number {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 4);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 5);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 6);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get isIndirect(): boolean {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 8) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 10);
  }
}
export class DeclarationStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeclarationStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DeclarationStatementASTSlotBase + 0);
  }
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeclarationStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DeclarationStatementASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DeclarationStatementASTSlotBase + 3);
  }
}
export class TryBlockStatementAST extends StatementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTryBlockStatement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get tryLoc(): number {
    return cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 2);
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get handlerList(): Iterable<HandlerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 6);
  }
}
export class CharLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCharLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      CharLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 3);
  }
  get literal(): CharLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 4),
      this.modelOwner,
      CharLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 7);
  }
}
export class BoolLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBoolLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      BoolLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 3);
  }
  get isTrue(): boolean {
    return cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 4) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 6);
  }
}
export class IntLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIntLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      IntLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 3);
  }
  get literal(): IntegerLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 4),
      this.modelOwner,
      IntegerLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 7);
  }
}
export class FloatLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFloatLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      FloatLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 3);
  }
  get literal(): FloatLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 4),
      this.modelOwner,
      FloatLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 7);
  }
}
export class NullptrLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNullptrLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NullptrLiteralExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      NullptrLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NullptrLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, NullptrLiteralExpressionASTSlotBase + 3);
  }
  get literal(): TokenKind {
    return cxx.readAST(
      this.handle,
      NullptrLiteralExpressionASTSlotBase + 4,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NullptrLiteralExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NullptrLiteralExpressionASTSlotBase + 6);
  }
}
export class StringLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStringLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      StringLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 3);
  }
  get literal(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 4),
      this.modelOwner,
      StringLiteral,
    );
  }
  get encoding(): TokenKind {
    return cxx.readAST(
      this.handle,
      StringLiteralExpressionASTSlotBase + 5,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 7);
  }
}
export class UserDefinedStringLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUserDefinedStringLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      UserDefinedStringLiteralExpressionASTSlotBase + 0,
    );
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      UserDefinedStringLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(
        this.handle,
        UserDefinedStringLiteralExpressionASTSlotBase + 2,
      ),
      this.modelOwner,
    );
  }
  get literalLoc(): number {
    return cxx.readAST(
      this.handle,
      UserDefinedStringLiteralExpressionASTSlotBase + 3,
    );
  }
  get literal(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(
        this.handle,
        UserDefinedStringLiteralExpressionASTSlotBase + 4,
      ),
      this.modelOwner,
      StringLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(
        this.handle,
        UserDefinedStringLiteralExpressionASTSlotBase + 5,
      ),
      this.modelOwner,
    );
  }
  get encoding(): TokenKind {
    return cxx.readAST(
      this.handle,
      UserDefinedStringLiteralExpressionASTSlotBase + 6,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      UserDefinedStringLiteralExpressionASTSlotBase + 7,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      UserDefinedStringLiteralExpressionASTSlotBase + 8,
    );
  }
}
export class ObjectLiteralExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitObjectLiteralExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ObjectLiteralExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 3);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 5);
  }
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 9);
  }
}
export class ThisExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThisExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ThisExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ThisExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ThisExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get thisLoc(): number {
    return cxx.readAST(this.handle, ThisExpressionASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ThisExpressionASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ThisExpressionASTSlotBase + 5);
  }
}
export class PackIndexExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPackIndexExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      PackIndexExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get packExpression(): IdExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 4);
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 5);
  }
  get indexExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 7);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 9);
  }
}
export class GenericSelectionExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGenericSelectionExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      GenericSelectionExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get genericLoc(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 4);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 6);
  }
  get genericAssociationList(): Iterable<GenericAssociationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 8);
  }
  get matchedAssocIndex(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 9);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 11);
  }
}
export class NestedStatementExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedStatementExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      NestedStatementExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 3);
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 5);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 7);
  }
}
export class DefaultInitializerExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultInitializerExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      DefaultInitializerExpressionASTSlotBase + 0,
    );
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      DefaultInitializerExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, DefaultInitializerExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DefaultInitializerExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get context(): DefaultInitializerContext {
    return objAt(
      cxx.readAST(this.handle, DefaultInitializerExpressionASTSlotBase + 4),
      this.modelOwner,
      DefaultInitializerContext,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      DefaultInitializerExpressionASTSlotBase + 5,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      DefaultInitializerExpressionASTSlotBase + 6,
    );
  }
}
export class NestedExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NestedExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      NestedExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NestedExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NestedExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NestedExpressionASTSlotBase + 5);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NestedExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NestedExpressionASTSlotBase + 7);
  }
}
export class IdExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIdExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, IdExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      IdExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, IdExpressionASTSlotBase + 4);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, IdExpressionASTSlotBase + 7) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, IdExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, IdExpressionASTSlotBase + 9);
  }
}
export class LambdaExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLambdaExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      LambdaExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 3);
  }
  get captureDefaultLoc(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 4);
  }
  get captureList(): Iterable<LambdaCaptureAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 6);
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 7);
  }
  get templateParameterList(): Iterable<TemplateParameterAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 9);
  }
  get templateRequiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get expressionAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 11),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 12);
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 13),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 14);
  }
  get gnuAtributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 15),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lambdaSpecifierList(): Iterable<LambdaSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 16),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get exceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 17),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 18),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get trailingReturnType(): TrailingReturnTypeAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 19),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 20),
      this.modelOwner,
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 21),
      this.modelOwner,
    );
  }
  get captureDefault(): TokenKind {
    return cxx.readAST(
      this.handle,
      LambdaExpressionASTSlotBase + 22,
    ) as TokenKind;
  }
  get symbol(): LambdaSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 23),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 24),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 25);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 26);
  }
}
export class FoldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFoldExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      FoldExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 3);
  }
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 5);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 6);
  }
  get foldOpLoc(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 7);
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 9);
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      FoldExpressionASTSlotBase + 10,
    ) as TokenKind;
  }
  get foldOp(): TokenKind {
    return cxx.readAST(
      this.handle,
      FoldExpressionASTSlotBase + 11,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 12);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, FoldExpressionASTSlotBase + 13);
  }
}
export class RightFoldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRightFoldExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      RightFoldExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 5);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 6);
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 7);
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      RightFoldExpressionASTSlotBase + 8,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 10);
  }
}
export class LeftFoldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLeftFoldExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      LeftFoldExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 3);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 4);
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 5);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 7);
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      LeftFoldExpressionASTSlotBase + 8,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 10);
  }
}
export class RequiresExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRequiresExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      RequiresExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get requiresLoc(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 4);
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 6);
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 7);
  }
  get requirementList(): Iterable<RequirementAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 9);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 11);
  }
}
export class VaArgExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVaArgExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      VaArgExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get vaArgLoc(): number {
    return cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 4);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 6);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 8);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 10);
  }
}
export class SubscriptExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSubscriptExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      SubscriptExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 4);
  }
  get indexExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 6);
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 8) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 10);
  }
}
export class CallExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCallExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CallExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      CallExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, CallExpressionASTSlotBase + 4);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, CallExpressionASTSlotBase + 6);
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, CallExpressionASTSlotBase + 7) !== 0;
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CallExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CallExpressionASTSlotBase + 10);
  }
}
export class TypeConstructionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeConstruction(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeConstructionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      TypeConstructionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get typeSpecifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, TypeConstructionASTSlotBase + 4);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, TypeConstructionASTSlotBase + 6);
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeConstructionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeConstructionASTSlotBase + 9);
  }
}
export class BracedTypeConstructionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedTypeConstruction(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      BracedTypeConstructionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get typeSpecifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 7);
  }
}
export class SpliceMemberExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSpliceMemberExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      SpliceMemberExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get accessLoc(): number {
    return cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 4);
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 5);
  }
  get splicer(): SplicerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get accessOp(): TokenKind {
    return cxx.readAST(
      this.handle,
      SpliceMemberExpressionASTSlotBase + 8,
    ) as TokenKind;
  }
  get isTemplateIntroduced(): boolean {
    return (
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 9) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 11);
  }
}
export class MemberExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitMemberExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, MemberExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      MemberExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get accessLoc(): number {
    return cxx.readAST(this.handle, MemberExpressionASTSlotBase + 4);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, MemberExpressionASTSlotBase + 6);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get accessOp(): TokenKind {
    return cxx.readAST(
      this.handle,
      MemberExpressionASTSlotBase + 9,
    ) as TokenKind;
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, MemberExpressionASTSlotBase + 10) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, MemberExpressionASTSlotBase + 11);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, MemberExpressionASTSlotBase + 12);
  }
}
export class PostIncrExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPostIncrExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      PostIncrExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 4);
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      PostIncrExpressionASTSlotBase + 5,
    ) as TokenKind;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 7) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 9);
  }
}
export class CppCastExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCppCastExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      CppCastExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get castLoc(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 3);
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 4);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 6);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 7);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 9);
  }
  get castOp(): TokenKind {
    return cxx.readAST(
      this.handle,
      CppCastExpressionASTSlotBase + 10,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 11);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 12);
  }
}
export class BuiltinBitCastExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBuiltinBitCastExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      BuiltinBitCastExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get castLoc(): number {
    return cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 4);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 6);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 8);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 10);
  }
}
export class BuiltinOffsetofExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBuiltinOffsetofExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      BuiltinOffsetofExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get offsetofLoc(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 4);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 6);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 7);
  }
  get designatorList(): Iterable<DesignatorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 9);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 12);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 13);
  }
}
export class TypeidExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeidExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      TypeidExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get typeidLoc(): number {
    return cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 4);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 8);
  }
}
export class TypeidOfTypeExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeidOfTypeExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      TypeidOfTypeExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get typeidLoc(): number {
    return cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 4);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 8);
  }
}
export class SpliceExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSpliceExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SpliceExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      SpliceExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, SpliceExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get splicer(): SplicerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SpliceExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SpliceExpressionASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SpliceExpressionASTSlotBase + 5);
  }
}
export class GlobalScopeReflectExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalScopeReflectExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      GlobalScopeReflectExpressionASTSlotBase + 0,
    );
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      GlobalScopeReflectExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, GlobalScopeReflectExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get caretCaretLoc(): number {
    return cxx.readAST(
      this.handle,
      GlobalScopeReflectExpressionASTSlotBase + 3,
    );
  }
  get scopeLoc(): number {
    return cxx.readAST(
      this.handle,
      GlobalScopeReflectExpressionASTSlotBase + 4,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      GlobalScopeReflectExpressionASTSlotBase + 5,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      GlobalScopeReflectExpressionASTSlotBase + 6,
    );
  }
}
export class NamespaceReflectExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamespaceReflectExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      NamespaceReflectExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get caretCaretLoc(): number {
    return cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 3);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 4);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 8);
  }
}
export class TypeIdReflectExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeIdReflectExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      TypeIdReflectExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get caretCaretLoc(): number {
    return cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 3);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 6);
  }
}
export class ReflectExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReflectExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ReflectExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get caretCaretLoc(): number {
    return cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 6);
  }
}
export class LabelAddressExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLabelAddressExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      LabelAddressExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ampAmpLoc(): number {
    return cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 3);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 4);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 7);
  }
}
export class UnaryExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUnaryExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      UnaryExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      UnaryExpressionASTSlotBase + 5,
    ) as TokenKind;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 7) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 9);
  }
}
export class AwaitExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAwaitExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      AwaitExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get awaitLoc(): number {
    return cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 6);
  }
}
export class SizeofExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      SizeofExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get sizeofLoc(): number {
    return cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get value(): bigint | undefined {
    return cxx.readASTVal(this.handle, SizeofExpressionASTSlotBase + 5) as
      bigint | undefined;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 7);
  }
}
export class SizeofTypeExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofTypeExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      SizeofTypeExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get sizeofLoc(): number {
    return cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 4);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 6);
  }
  get value(): bigint | undefined {
    return cxx.readASTVal(this.handle, SizeofTypeExpressionASTSlotBase + 7) as
      bigint | undefined;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 9);
  }
}
export class SizeofPackExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeofPackExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      SizeofPackExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get sizeofLoc(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 3);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 4);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 5);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 6);
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 7);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 11);
  }
}
export class AlignofTypeExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignofTypeExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      AlignofTypeExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get alignofLoc(): number {
    return cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 4);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 8);
  }
}
export class AlignofExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignofExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      AlignofExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get alignofLoc(): number {
    return cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 6);
  }
}
export class NoexceptExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoexceptExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      NoexceptExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get noexceptLoc(): number {
    return cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 4);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 6);
  }
  get value(): boolean | undefined {
    return ((item: any) => (item === undefined ? undefined : item !== 0))(
      cxx.readASTVal(this.handle, NoexceptExpressionASTSlotBase + 7),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 9);
  }
}
export class NewExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      NewExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 3);
  }
  get newLoc(): number {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 4);
  }
  get newPlacement(): NewPlacementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 6);
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 9);
  }
  get newInitalizer(): NewInitializerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get objectType(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 12),
      this.modelOwner,
    );
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 13),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 14);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 15);
  }
}
export class DeleteExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeleteExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      DeleteExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 3);
  }
  get deleteLoc(): number {
    return cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 4);
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 5);
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 6);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 10);
  }
}
export class CastExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCastExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CastExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      CastExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, CastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, CastExpressionASTSlotBase + 3);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CastExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, CastExpressionASTSlotBase + 5);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CastExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CastExpressionASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CastExpressionASTSlotBase + 8);
  }
}
export class ImplicitCastExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitImplicitCastExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ImplicitCastExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get castKind(): ImplicitCastKind {
    return cxx.readAST(
      this.handle,
      ImplicitCastExpressionASTSlotBase + 4,
    ) as ImplicitCastKind;
  }
  get conversionFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return (
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 6) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 8);
  }
}
export class ConstExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConstExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ConstExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ConstExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get constValue():
    | { readonly index: 0; readonly value: bigint }
    | { readonly index: 1; readonly value: StringLiteral | undefined }
    | { readonly index: 2; readonly value: number }
    | { readonly index: 3; readonly value: number }
    | { readonly index: 4; readonly value: number }
    | { readonly index: 5; readonly value: Meta | undefined }
    | { readonly index: 6; readonly value: InitializerList | undefined }
    | { readonly index: 7; readonly value: ConstObject | undefined }
    | { readonly index: 8; readonly value: ConstAddress | undefined }
    | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
    | { readonly index: 10; readonly value: ConstComplex | undefined }
    | { readonly index: 11; readonly value: {} }
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : ((item: any) =>
            item.index === 1
              ? {
                  index: 1,
                  value: objOf(item.value, this.modelOwner, StringLiteral),
                }
              : item.index === 5
                ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
                : item.index === 6
                  ? {
                      index: 6,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        InitializerList,
                      ),
                    }
                  : item.index === 7
                    ? {
                        index: 7,
                        value: objOf(item.value, this.modelOwner, ConstObject),
                      }
                    : item.index === 8
                      ? {
                          index: 8,
                          value: objOf(
                            item.value,
                            this.modelOwner,
                            ConstAddress,
                          ),
                        }
                      : item.index === 9
                        ? {
                            index: 9,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              ConstLabelAddress,
                            ),
                          }
                        : item.index === 10
                          ? {
                              index: 10,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                ConstComplex,
                              ),
                            }
                          : item)(item))(
      cxx.readASTVal(this.handle, ConstExpressionASTSlotBase + 4),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConstExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConstExpressionASTSlotBase + 6);
  }
}
export class BinaryExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBinaryExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      BinaryExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 4);
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      BinaryExpressionASTSlotBase + 6,
    ) as TokenKind;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 8) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 10);
  }
}
export class ConditionalExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConditionalExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ConditionalExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get questionLoc(): number {
    return cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 4);
  }
  get iftrueExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 6);
  }
  get iffalseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 9);
  }
}
export class YieldExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitYieldExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, YieldExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      YieldExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, YieldExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get yieldLoc(): number {
    return cxx.readAST(this.handle, YieldExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, YieldExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, YieldExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, YieldExpressionASTSlotBase + 6);
  }
}
export class ThrowExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThrowExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ThrowExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get throwLoc(): number {
    return cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 6);
  }
}
export class AssignmentExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAssignmentExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      AssignmentExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 4);
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      AssignmentExpressionASTSlotBase + 6,
    ) as TokenKind;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 8) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 10);
  }
}
export class TargetExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTargetExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TargetExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      TargetExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TargetExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TargetExpressionASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TargetExpressionASTSlotBase + 4);
  }
}
export class RightExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRightExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RightExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      RightExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, RightExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RightExpressionASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RightExpressionASTSlotBase + 4);
  }
}
export class CompoundAssignmentExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundAssignmentExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      CompoundAssignmentExpressionASTSlotBase + 0,
    );
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      CompoundAssignmentExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get targetExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get opLoc(): number {
    return cxx.readAST(
      this.handle,
      CompoundAssignmentExpressionASTSlotBase + 4,
    );
  }
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get adjustExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      CompoundAssignmentExpressionASTSlotBase + 8,
    ) as TokenKind;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return (
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 10) !==
      0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      CompoundAssignmentExpressionASTSlotBase + 11,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      CompoundAssignmentExpressionASTSlotBase + 12,
    );
  }
}
export class PackExpansionExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPackExpansionExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      PackExpansionExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 6);
  }
}
export class DesignatedInitializerClauseAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDesignatedInitializerClause(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      DesignatedInitializerClauseASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get designatorList(): Iterable<DesignatorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 7);
  }
}
export class TypeTraitExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeTraitExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      TypeTraitExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get typeTraitLoc(): number {
    return cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 3);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 4);
  }
  get typeIdList(): Iterable<TypeIdAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 6);
  }
  get typeTrait(): BuiltinTypeTraitKind {
    return cxx.readAST(
      this.handle,
      TypeTraitExpressionASTSlotBase + 7,
    ) as BuiltinTypeTraitKind;
  }
  get value(): boolean | undefined {
    return ((item: any) => (item === undefined ? undefined : item !== 0))(
      cxx.readASTVal(this.handle, TypeTraitExpressionASTSlotBase + 8),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 10);
  }
}
export class ConditionExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConditionExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ConditionExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 9);
  }
}
export class EqualInitializerAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEqualInitializer(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, EqualInitializerASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      EqualInitializerASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, EqualInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, EqualInitializerASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EqualInitializerASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, EqualInitializerASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, EqualInitializerASTSlotBase + 6);
  }
}
export class BracedInitListAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedInitList(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BracedInitListASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      BracedInitListASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, BracedInitListASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, BracedInitListASTSlotBase + 3);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BracedInitListASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, BracedInitListASTSlotBase + 5);
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, BracedInitListASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BracedInitListASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BracedInitListASTSlotBase + 8);
  }
}
export class ParenInitializerAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParenInitializer(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ParenInitializerASTSlotBase + 0);
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ParenInitializerASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ParenInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, ParenInitializerASTSlotBase + 3);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParenInitializerASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, ParenInitializerASTSlotBase + 5);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ParenInitializerASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ParenInitializerASTSlotBase + 7);
  }
}
export class ThreeWayComparisonExpressionAST extends ExpressionAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThreeWayComparisonExpression(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      ThreeWayComparisonExpressionASTSlotBase + 0,
    );
  }
  get valueCategory(): ValueCategory {
    return cxx.readAST(
      this.handle,
      ThreeWayComparisonExpressionASTSlotBase + 1,
    ) as ValueCategory;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get comparison(): BinaryExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lessResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get equalResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get greaterResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get unorderedResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      ThreeWayComparisonExpressionASTSlotBase + 8,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      ThreeWayComparisonExpressionASTSlotBase + 9,
    );
  }
}
export class DefaultGenericAssociationAST extends GenericAssociationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultGenericAssociation(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 0);
  }
  get defaultLoc(): number {
    return cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 1);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 5);
  }
}
export class TypeGenericAssociationAST extends GenericAssociationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeGenericAssociation(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 0);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 5);
  }
}
export class DotDesignatorAST extends DesignatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDotDesignator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DotDesignatorASTSlotBase + 0);
  }
  get dotLoc(): number {
    return cxx.readAST(this.handle, DotDesignatorASTSlotBase + 1);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, DotDesignatorASTSlotBase + 2);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, DotDesignatorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DotDesignatorASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DotDesignatorASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DotDesignatorASTSlotBase + 6);
  }
}
export class SubscriptDesignatorAST extends DesignatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSubscriptDesignator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 0);
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 1);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 5);
  }
}
export class TemplateTypeParameterAST extends TemplateParameterAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateTypeParameter(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 2);
  }
  get index(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 3);
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 4);
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 5);
  }
  get templateParameterList(): Iterable<TemplateParameterAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 7);
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get classKeyLoc(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 9);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 10);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 11);
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 12);
  }
  get idExpression(): IdExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 13),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 14),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return (
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 15) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 16);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 17);
  }
}
export class NonTypeTemplateParameterAST extends TemplateParameterAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNonTypeTemplateParameter(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 2);
  }
  get index(): number {
    return cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 3);
  }
  get declaration(): ParameterDeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 6);
  }
}
export class TypenameTypeParameterAST extends TemplateParameterAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypenameTypeParameter(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 2);
  }
  get index(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 3);
  }
  get classKeyLoc(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 4);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 5);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 6);
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 7);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return (
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 10) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 11);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 12);
  }
}
export class ConstraintTypeParameterAST extends TemplateParameterAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstraintTypeParameter(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 2);
  }
  get index(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 3);
  }
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 5);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 6);
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 7);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 10);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 11);
  }
}
export class TypedefSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypedefSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypedefSpecifierASTSlotBase + 0);
  }
  get typedefLoc(): number {
    return cxx.readAST(this.handle, TypedefSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypedefSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypedefSpecifierASTSlotBase + 3);
  }
}
export class FriendSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFriendSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, FriendSpecifierASTSlotBase + 0);
  }
  get friendLoc(): number {
    return cxx.readAST(this.handle, FriendSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, FriendSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, FriendSpecifierASTSlotBase + 3);
  }
}
export class ConstevalSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstevalSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConstevalSpecifierASTSlotBase + 0);
  }
  get constevalLoc(): number {
    return cxx.readAST(this.handle, ConstevalSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConstevalSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConstevalSpecifierASTSlotBase + 3);
  }
}
export class ConstinitSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstinitSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConstinitSpecifierASTSlotBase + 0);
  }
  get constinitLoc(): number {
    return cxx.readAST(this.handle, ConstinitSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConstinitSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConstinitSpecifierASTSlotBase + 3);
  }
}
export class ConstexprSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstexprSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConstexprSpecifierASTSlotBase + 0);
  }
  get constexprLoc(): number {
    return cxx.readAST(this.handle, ConstexprSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConstexprSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConstexprSpecifierASTSlotBase + 3);
  }
}
export class InlineSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInlineSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, InlineSpecifierASTSlotBase + 0);
  }
  get inlineLoc(): number {
    return cxx.readAST(this.handle, InlineSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, InlineSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, InlineSpecifierASTSlotBase + 3);
  }
}
export class NoreturnSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoreturnSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NoreturnSpecifierASTSlotBase + 0);
  }
  get noreturnLoc(): number {
    return cxx.readAST(this.handle, NoreturnSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NoreturnSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NoreturnSpecifierASTSlotBase + 3);
  }
}
export class StaticSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitStaticSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, StaticSpecifierASTSlotBase + 0);
  }
  get staticLoc(): number {
    return cxx.readAST(this.handle, StaticSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, StaticSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, StaticSpecifierASTSlotBase + 3);
  }
}
export class ExternSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExternSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ExternSpecifierASTSlotBase + 0);
  }
  get externLoc(): number {
    return cxx.readAST(this.handle, ExternSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExternSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExternSpecifierASTSlotBase + 3);
  }
}
export class RegisterSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRegisterSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RegisterSpecifierASTSlotBase + 0);
  }
  get registerLoc(): number {
    return cxx.readAST(this.handle, RegisterSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RegisterSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RegisterSpecifierASTSlotBase + 3);
  }
}
export class ThreadLocalSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThreadLocalSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ThreadLocalSpecifierASTSlotBase + 0);
  }
  get threadLocalLoc(): number {
    return cxx.readAST(this.handle, ThreadLocalSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ThreadLocalSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ThreadLocalSpecifierASTSlotBase + 3);
  }
}
export class ThreadSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThreadSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ThreadSpecifierASTSlotBase + 0);
  }
  get threadLoc(): number {
    return cxx.readAST(this.handle, ThreadSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ThreadSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ThreadSpecifierASTSlotBase + 3);
  }
}
export class MutableSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitMutableSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, MutableSpecifierASTSlotBase + 0);
  }
  get mutableLoc(): number {
    return cxx.readAST(this.handle, MutableSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, MutableSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, MutableSpecifierASTSlotBase + 3);
  }
}
export class VirtualSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVirtualSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, VirtualSpecifierASTSlotBase + 0);
  }
  get virtualLoc(): number {
    return cxx.readAST(this.handle, VirtualSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, VirtualSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, VirtualSpecifierASTSlotBase + 3);
  }
}
export class ExplicitSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExplicitSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 0);
  }
  get explicitLoc(): number {
    return cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 6);
  }
}
export class AutoTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAutoTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AutoTypeSpecifierASTSlotBase + 0);
  }
  get autoLoc(): number {
    return cxx.readAST(this.handle, AutoTypeSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AutoTypeSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AutoTypeSpecifierASTSlotBase + 3);
  }
}
export class VoidTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVoidTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, VoidTypeSpecifierASTSlotBase + 0);
  }
  get voidLoc(): number {
    return cxx.readAST(this.handle, VoidTypeSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, VoidTypeSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, VoidTypeSpecifierASTSlotBase + 3);
  }
}
export class SizeTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSizeTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SizeTypeSpecifierASTSlotBase + 0);
  }
  get specifierLoc(): number {
    return cxx.readAST(this.handle, SizeTypeSpecifierASTSlotBase + 1);
  }
  get specifier(): TokenKind {
    return cxx.readAST(
      this.handle,
      SizeTypeSpecifierASTSlotBase + 2,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SizeTypeSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SizeTypeSpecifierASTSlotBase + 4);
  }
}
export class SignTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSignTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SignTypeSpecifierASTSlotBase + 0);
  }
  get specifierLoc(): number {
    return cxx.readAST(this.handle, SignTypeSpecifierASTSlotBase + 1);
  }
  get specifier(): TokenKind {
    return cxx.readAST(
      this.handle,
      SignTypeSpecifierASTSlotBase + 2,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SignTypeSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SignTypeSpecifierASTSlotBase + 4);
  }
}
export class BuiltinTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBuiltinTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BuiltinTypeSpecifierASTSlotBase + 0);
  }
  get specifierLoc(): number {
    return cxx.readAST(this.handle, BuiltinTypeSpecifierASTSlotBase + 1);
  }
  get specifier(): TokenKind {
    return cxx.readAST(
      this.handle,
      BuiltinTypeSpecifierASTSlotBase + 2,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BuiltinTypeSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BuiltinTypeSpecifierASTSlotBase + 4);
  }
}
export class UnaryBuiltinTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUnaryBuiltinTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 0);
  }
  get builtinLoc(): number {
    return cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 2);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 4);
  }
  get builtinKind(): UnaryBuiltinTypeKind {
    return cxx.readAST(
      this.handle,
      UnaryBuiltinTypeSpecifierASTSlotBase + 5,
    ) as UnaryBuiltinTypeKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 7);
  }
}
export class BinaryBuiltinTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBinaryBuiltinTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 0);
  }
  get builtinLoc(): number {
    return cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 2);
  }
  get leftTypeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 4);
  }
  get rightTypeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 6);
  }
  get builtinKind(): BinaryBuiltinTypeKind {
    return cxx.readAST(
      this.handle,
      BinaryBuiltinTypeSpecifierASTSlotBase + 7,
    ) as BinaryBuiltinTypeKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 9);
  }
}
export class IntegralTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIntegralTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, IntegralTypeSpecifierASTSlotBase + 0);
  }
  get specifierLoc(): number {
    return cxx.readAST(this.handle, IntegralTypeSpecifierASTSlotBase + 1);
  }
  get specifier(): TokenKind {
    return cxx.readAST(
      this.handle,
      IntegralTypeSpecifierASTSlotBase + 2,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, IntegralTypeSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, IntegralTypeSpecifierASTSlotBase + 4);
  }
}
export class FloatingPointTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFloatingPointTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, FloatingPointTypeSpecifierASTSlotBase + 0);
  }
  get specifierLoc(): number {
    return cxx.readAST(this.handle, FloatingPointTypeSpecifierASTSlotBase + 1);
  }
  get specifier(): TokenKind {
    return cxx.readAST(
      this.handle,
      FloatingPointTypeSpecifierASTSlotBase + 2,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, FloatingPointTypeSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, FloatingPointTypeSpecifierASTSlotBase + 4);
  }
}
export class ComplexTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitComplexTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ComplexTypeSpecifierASTSlotBase + 0);
  }
  get complexLoc(): number {
    return cxx.readAST(this.handle, ComplexTypeSpecifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ComplexTypeSpecifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ComplexTypeSpecifierASTSlotBase + 3);
  }
}
export class NamedTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNamedTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 0);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 2);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 4) !== 0;
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 7);
  }
}
export class AtomicTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAtomicTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 0);
  }
  get atomicLoc(): number {
    return cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 2);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 6);
  }
}
export class BitIntTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBitIntTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 0);
  }
  get bitintLoc(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 2);
  }
  get sizeExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 4);
  }
  get bitCount(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 5);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 7);
  }
}
export class UnderlyingTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitUnderlyingTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 0);
  }
  get underlyingTypeLoc(): number {
    return cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 2);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 6);
  }
}
export class ElaboratedTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitElaboratedTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 0);
  }
  get classLoc(): number {
    return cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 1);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 4);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get classKey(): TokenKind {
    return cxx.readAST(
      this.handle,
      ElaboratedTypeSpecifierASTSlotBase + 6,
    ) as TokenKind;
  }
  get isTemplateIntroduced(): boolean {
    return (
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 7) !== 0
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 10);
  }
}
export class DecltypeAutoSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeAutoSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 0);
  }
  get decltypeLoc(): number {
    return cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 2);
  }
  get autoLoc(): number {
    return cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 3);
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 6);
  }
}
export class DecltypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 0);
  }
  get decltypeLoc(): number {
    return cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 4);
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 7);
  }
}
export class PlaceholderTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPlaceholderTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, PlaceholderTypeSpecifierASTSlotBase + 0);
  }
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PlaceholderTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get specifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PlaceholderTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PlaceholderTypeSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PlaceholderTypeSpecifierASTSlotBase + 4);
  }
}
export class ConstQualifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConstQualifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConstQualifierASTSlotBase + 0);
  }
  get constLoc(): number {
    return cxx.readAST(this.handle, ConstQualifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConstQualifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConstQualifierASTSlotBase + 3);
  }
}
export class VolatileQualifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitVolatileQualifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, VolatileQualifierASTSlotBase + 0);
  }
  get volatileLoc(): number {
    return cxx.readAST(this.handle, VolatileQualifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, VolatileQualifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, VolatileQualifierASTSlotBase + 3);
  }
}
export class AtomicQualifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAtomicQualifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AtomicQualifierASTSlotBase + 0);
  }
  get atomicLoc(): number {
    return cxx.readAST(this.handle, AtomicQualifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AtomicQualifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AtomicQualifierASTSlotBase + 3);
  }
}
export class RestrictQualifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRestrictQualifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RestrictQualifierASTSlotBase + 0);
  }
  get restrictLoc(): number {
    return cxx.readAST(this.handle, RestrictQualifierASTSlotBase + 1);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RestrictQualifierASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RestrictQualifierASTSlotBase + 3);
  }
}
export class EnumSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEnumSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 0);
  }
  get enumLoc(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 1);
  }
  get classLoc(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 2);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 6);
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 8);
  }
  get enumeratorList(): Iterable<EnumeratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 9),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get commaLoc(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 10);
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 11);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 12),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 13);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 14);
  }
}
export class ClassSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitClassSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 0);
  }
  get classLoc(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 1);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get finalLoc(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 5);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 6);
  }
  get baseSpecifierList(): Iterable<BaseSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 8);
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 9),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 10);
  }
  get classKey(): TokenKind {
    return cxx.readAST(
      this.handle,
      ClassSpecifierASTSlotBase + 11,
    ) as TokenKind;
  }
  get symbol(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 12),
      this.modelOwner,
    );
  }
  get isFinal(): boolean {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 13) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 14);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 15);
  }
}
export class TypenameSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypenameSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 0);
  }
  get typenameLoc(): number {
    return cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 1);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 3);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 5) !== 0;
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 8);
  }
}
export class SplicerTypeSpecifierAST extends SpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSplicerTypeSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SplicerTypeSpecifierASTSlotBase + 0);
  }
  get typenameLoc(): number {
    return cxx.readAST(this.handle, SplicerTypeSpecifierASTSlotBase + 1);
  }
  get splicer(): SplicerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SplicerTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SplicerTypeSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SplicerTypeSpecifierASTSlotBase + 4);
  }
}
export class PointerOperatorAST extends PtrOperatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPointerOperator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, PointerOperatorASTSlotBase + 0);
  }
  get starLoc(): number {
    return cxx.readAST(this.handle, PointerOperatorASTSlotBase + 1);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PointerOperatorASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get cvQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PointerOperatorASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PointerOperatorASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PointerOperatorASTSlotBase + 5);
  }
}
export class ReferenceOperatorAST extends PtrOperatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitReferenceOperator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 0);
  }
  get refLoc(): number {
    return cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 1);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get refOp(): TokenKind {
    return cxx.readAST(
      this.handle,
      ReferenceOperatorASTSlotBase + 3,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 5);
  }
}
export class PtrToMemberOperatorAST extends PtrOperatorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitPtrToMemberOperator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 0);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get starLoc(): number {
    return cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 2);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get cvQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 6);
  }
}
export class BitfieldDeclaratorAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBitfieldDeclarator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 0);
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 2);
  }
  get sizeExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 5);
  }
}
export class ParameterPackAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParameterPack(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ParameterPackASTSlotBase + 0);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, ParameterPackASTSlotBase + 1);
  }
  get coreDeclarator(): CoreDeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParameterPackASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ParameterPackASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ParameterPackASTSlotBase + 4);
  }
}
export class IdDeclaratorAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitIdDeclarator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 0);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 2);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 5) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 7);
  }
}
export class NestedDeclaratorAST extends CoreDeclaratorAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedDeclarator(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 0);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 1);
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 5);
  }
}
export class FunctionDeclaratorChunkAST extends DeclaratorChunkAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitFunctionDeclaratorChunk(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 0);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 1);
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 3);
  }
  get cvQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get refLoc(): number {
    return cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 5);
  }
  get exceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get trailingReturnType(): TrailingReturnTypeAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get refOp(): TokenKind {
    return cxx.readAST(
      this.handle,
      FunctionDeclaratorChunkASTSlotBase + 9,
    ) as TokenKind;
  }
  get isFinal(): boolean {
    return (
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 10) !== 0
    );
  }
  get isOverride(): boolean {
    return (
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 11) !== 0
    );
  }
  get isPure(): boolean {
    return (
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 12) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 13);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 14);
  }
}
export class ArrayDeclaratorChunkAST extends DeclaratorChunkAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitArrayDeclaratorChunk(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 0);
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 1);
  }
  get typeQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 4);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 7);
  }
}
export class NameIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNameId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NameIdASTSlotBase + 0);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, NameIdASTSlotBase + 1);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NameIdASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NameIdASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NameIdASTSlotBase + 4);
  }
}
export class DestructorIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDestructorId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DestructorIdASTSlotBase + 0);
  }
  get tildeLoc(): number {
    return cxx.readAST(this.handle, DestructorIdASTSlotBase + 1);
  }
  get id(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DestructorIdASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DestructorIdASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DestructorIdASTSlotBase + 4);
  }
}
export class DecltypeIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DecltypeIdASTSlotBase + 0);
  }
  get decltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DecltypeIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeIdASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeIdASTSlotBase + 3);
  }
}
export class OperatorFunctionIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOperatorFunctionId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 0);
  }
  get operatorLoc(): number {
    return cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 1);
  }
  get opLoc(): number {
    return cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 2);
  }
  get openLoc(): number {
    return cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 3);
  }
  get closeLoc(): number {
    return cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 4);
  }
  get op(): TokenKind {
    return cxx.readAST(
      this.handle,
      OperatorFunctionIdASTSlotBase + 5,
    ) as TokenKind;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 7);
  }
}
export class LiteralOperatorIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLiteralOperatorId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 0);
  }
  get operatorLoc(): number {
    return cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 1);
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 2);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 3);
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 4),
      this.modelOwner,
      Literal,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 7);
  }
}
export class ConversionFunctionIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitConversionFunctionId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ConversionFunctionIdASTSlotBase + 0);
  }
  get operatorLoc(): number {
    return cxx.readAST(this.handle, ConversionFunctionIdASTSlotBase + 1);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConversionFunctionIdASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ConversionFunctionIdASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ConversionFunctionIdASTSlotBase + 4);
  }
}
export class SimpleTemplateIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleTemplateId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 0);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 1);
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 2);
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 4);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 8);
  }
}
export class LiteralOperatorTemplateIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitLiteralOperatorTemplateId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 0);
  }
  get literalOperatorId(): LiteralOperatorIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 2);
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 6);
  }
}
export class OperatorFunctionTemplateIdAST extends UnqualifiedIdAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitOperatorFunctionTemplateId(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 0);
  }
  get operatorFunctionId(): OperatorFunctionIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lessLoc(): number {
    return cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 2);
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterLoc(): number {
    return cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 6);
  }
}
export class GlobalNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGlobalNestedNameSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, GlobalNestedNameSpecifierASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, GlobalNestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, GlobalNestedNameSpecifierASTSlotBase + 2);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, GlobalNestedNameSpecifierASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, GlobalNestedNameSpecifierASTSlotBase + 4);
  }
}
export class SimpleNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleNestedNameSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 3);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 5);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 7);
  }
}
export class DecltypeNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDecltypeNestedNameSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get decltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 5);
  }
}
export class TemplateNestedNameSpecifierAST extends NestedNameSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTemplateNestedNameSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 3);
  }
  get templateId(): SimpleTemplateIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 5);
  }
  get isTemplateIntroduced(): boolean {
    return (
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 6) !== 0
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 8);
  }
}
export class DefaultFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDefaultFunctionBody(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 0);
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 1);
  }
  get defaultLoc(): number {
    return cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 2);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 5);
  }
}
export class CompoundStatementFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundStatementFunctionBody(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      CompoundStatementFunctionBodyASTSlotBase + 0,
    );
  }
  get colonLoc(): number {
    return cxx.readAST(
      this.handle,
      CompoundStatementFunctionBodyASTSlotBase + 1,
    );
  }
  get memInitializerList(): Iterable<MemInitializerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CompoundStatementFunctionBodyASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundStatementFunctionBodyASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      CompoundStatementFunctionBodyASTSlotBase + 4,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      CompoundStatementFunctionBodyASTSlotBase + 5,
    );
  }
}
export class TryStatementFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTryStatementFunctionBody(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 0);
  }
  get tryLoc(): number {
    return cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 1);
  }
  get colonLoc(): number {
    return cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 2);
  }
  get memInitializerList(): Iterable<MemInitializerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get handlerList(): Iterable<HandlerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 7);
  }
}
export class DeleteFunctionBodyAST extends FunctionBodyAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDeleteFunctionBody(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 0);
  }
  get equalLoc(): number {
    return cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 1);
  }
  get deleteLoc(): number {
    return cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 2);
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 5);
  }
}
export class TypeTemplateArgumentAST extends TemplateArgumentAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeTemplateArgument(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeTemplateArgumentASTSlotBase + 0);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeTemplateArgumentASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeTemplateArgumentASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeTemplateArgumentASTSlotBase + 3);
  }
}
export class ExpressionTemplateArgumentAST extends TemplateArgumentAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitExpressionTemplateArgument(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ExpressionTemplateArgumentASTSlotBase + 0);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExpressionTemplateArgumentASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ExpressionTemplateArgumentASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ExpressionTemplateArgumentASTSlotBase + 3);
  }
}
export class ThrowExceptionSpecifierAST extends ExceptionSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThrowExceptionSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 0);
  }
  get throwLoc(): number {
    return cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 2);
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 5);
  }
}
export class NoexceptSpecifierAST extends ExceptionSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNoexceptSpecifier(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 0);
  }
  get noexceptLoc(): number {
    return cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 1);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 2);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 4);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 6);
  }
}
export class SimpleRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleRequirement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SimpleRequirementASTSlotBase + 0);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleRequirementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, SimpleRequirementASTSlotBase + 2);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleRequirementASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleRequirementASTSlotBase + 4);
  }
}
export class CompoundRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCompoundRequirement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 0);
  }
  get lbraceLoc(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 1);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rbraceLoc(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 3);
  }
  get noexceptLoc(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 4);
  }
  get minusGreaterLoc(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 5);
  }
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 7);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 9);
  }
}
export class TypeRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeRequirement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeRequirementASTSlotBase + 0);
  }
  get typenameLoc(): number {
    return cxx.readAST(this.handle, TypeRequirementASTSlotBase + 1);
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeRequirementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateLoc(): number {
    return cxx.readAST(this.handle, TypeRequirementASTSlotBase + 3);
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeRequirementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, TypeRequirementASTSlotBase + 5);
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, TypeRequirementASTSlotBase + 6) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeRequirementASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeRequirementASTSlotBase + 8);
  }
}
export class NestedRequirementAST extends RequirementAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNestedRequirement(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NestedRequirementASTSlotBase + 0);
  }
  get requiresLoc(): number {
    return cxx.readAST(this.handle, NestedRequirementASTSlotBase + 1);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedRequirementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get semicolonLoc(): number {
    return cxx.readAST(this.handle, NestedRequirementASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NestedRequirementASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NestedRequirementASTSlotBase + 5);
  }
}
export class NewParenInitializerAST extends NewInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewParenInitializer(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 0);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 1);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 3);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 5);
  }
}
export class NewBracedInitializerAST extends NewInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitNewBracedInitializer(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, NewBracedInitializerASTSlotBase + 0);
  }
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewBracedInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, NewBracedInitializerASTSlotBase + 2);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, NewBracedInitializerASTSlotBase + 3);
  }
}
export class ParenMemInitializerAST extends MemInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitParenMemInitializer(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 5);
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 7);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 8);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 9);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 10);
  }
}
export class BracedMemInitializerAST extends MemInitializerAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitBracedMemInitializer(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 0);
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 6);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 8);
  }
}
export class ThisLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitThisLambdaCapture(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 0);
  }
  get thisLoc(): number {
    return cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 1);
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 5);
  }
}
export class DerefThisLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitDerefThisLambdaCapture(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 0);
  }
  get starLoc(): number {
    return cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 1);
  }
  get thisLoc(): number {
    return cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 2);
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 4);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 5);
  }
}
export class SimpleLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleLambdaCapture(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 0);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 1);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 2);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 7);
  }
}
export class RefLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRefLambdaCapture(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 0);
  }
  get ampLoc(): number {
    return cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 1);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 2);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 3);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 8);
  }
}
export class RefInitLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitRefInitLambdaCapture(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 0);
  }
  get ampLoc(): number {
    return cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 1);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 2);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 3);
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 8);
  }
}
export class InitLambdaCaptureAST extends LambdaCaptureAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitInitLambdaCapture(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 0);
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 1);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 2);
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 7);
  }
}
export class EllipsisExceptionDeclarationAST extends ExceptionDeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitEllipsisExceptionDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(
      this.handle,
      EllipsisExceptionDeclarationASTSlotBase + 0,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(
      this.handle,
      EllipsisExceptionDeclarationASTSlotBase + 1,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      EllipsisExceptionDeclarationASTSlotBase + 2,
    );
  }
  get lastSourceLocation(): number {
    return cxx.readAST(
      this.handle,
      EllipsisExceptionDeclarationASTSlotBase + 3,
    );
  }
}
export class TypeExceptionDeclarationAST extends ExceptionDeclarationAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitTypeExceptionDeclaration(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 0);
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 5);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 6);
  }
}
export class CxxAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitCxxAttribute(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, CxxAttributeASTSlotBase + 0);
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readASTVal(this.handle, CxxAttributeASTSlotBase + 1));
  }
  get lbracketLoc(): number {
    return cxx.readAST(this.handle, CxxAttributeASTSlotBase + 2);
  }
  get lbracket2Loc(): number {
    return cxx.readAST(this.handle, CxxAttributeASTSlotBase + 3);
  }
  get attributeUsingPrefix(): AttributeUsingPrefixAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbracketLoc(): number {
    return cxx.readAST(this.handle, CxxAttributeASTSlotBase + 6);
  }
  get rbracket2Loc(): number {
    return cxx.readAST(this.handle, CxxAttributeASTSlotBase + 7);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, CxxAttributeASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, CxxAttributeASTSlotBase + 9);
  }
}
export class GccAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitGccAttribute(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 0);
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readASTVal(this.handle, GccAttributeASTSlotBase + 1));
  }
  get attributeLoc(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 3);
  }
  get lparen2Loc(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 4);
  }
  get attributeList(): Iterable<AttributeAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GccAttributeASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 6);
  }
  get rparen2Loc(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 7);
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, GccAttributeASTSlotBase + 9);
  }
}
export class AlignasAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignasAttribute(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 0);
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readASTVal(this.handle, AlignasAttributeASTSlotBase + 1));
  }
  get alignasLoc(): number {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 3);
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 5);
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 6);
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 7) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 9);
  }
}
export class AlignasTypeAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAlignasTypeAttribute(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 0);
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readASTVal(this.handle, AlignasTypeAttributeASTSlotBase + 1));
  }
  get alignasLoc(): number {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 3);
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get ellipsisLoc(): number {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 5);
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 6);
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 7) !== 0;
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 8);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 9);
  }
}
export class AsmAttributeAST extends AttributeSpecifierAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitAsmAttribute(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, AsmAttributeASTSlotBase + 0);
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readASTVal(this.handle, AsmAttributeASTSlotBase + 1));
  }
  get asmLoc(): number {
    return cxx.readAST(this.handle, AsmAttributeASTSlotBase + 2);
  }
  get lparenLoc(): number {
    return cxx.readAST(this.handle, AsmAttributeASTSlotBase + 3);
  }
  get literalLoc(): number {
    return cxx.readAST(this.handle, AsmAttributeASTSlotBase + 4);
  }
  get rparenLoc(): number {
    return cxx.readAST(this.handle, AsmAttributeASTSlotBase + 5);
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmAttributeASTSlotBase + 6),
      this.modelOwner,
      Literal,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, AsmAttributeASTSlotBase + 7);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, AsmAttributeASTSlotBase + 8);
  }
}
export class ScopedAttributeTokenAST extends AttributeTokenAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitScopedAttributeToken(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 0);
  }
  get attributeNamespaceLoc(): number {
    return cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 1);
  }
  get scopeLoc(): number {
    return cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 2);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 3);
  }
  get attributeNamespace(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 6);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 7);
  }
}
export class SimpleAttributeTokenAST extends AttributeTokenAST {
  accept<Context, Result>(
    visitor: ASTVisitor<Context, Result>,
    context: Context,
  ): Result {
    return visitor.visitSimpleAttributeToken(this, context);
  }
  get internalId(): number {
    return cxx.readAST(this.handle, SimpleAttributeTokenASTSlotBase + 0);
  }
  get identifierLoc(): number {
    return cxx.readAST(this.handle, SimpleAttributeTokenASTSlotBase + 1);
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleAttributeTokenASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get firstSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleAttributeTokenASTSlotBase + 3);
  }
  get lastSourceLocation(): number {
    return cxx.readAST(this.handle, SimpleAttributeTokenASTSlotBase + 4);
  }
}
export class Literal extends ModelObject {
  get value(): string {
    return cxx.readLiteralString(this.handle, LiteralSlotBase + 0) as string;
  }
  get hashCode(): number {
    return cxx.readLiteral(this.handle, LiteralSlotBase + 1);
  }
}
export class IntegerLiteral extends Literal {
  get value(): string {
    return cxx.readLiteralString(
      this.handle,
      IntegerLiteralSlotBase + 0,
    ) as string;
  }
  get hashCode(): number {
    return cxx.readLiteral(this.handle, IntegerLiteralSlotBase + 1);
  }
  get integerValue(): bigint {
    return cxx.readLiteralBigInt(
      this.handle,
      IntegerLiteralSlotBase + 2,
    ) as bigint;
  }
  get components(): {
    readonly value: bigint;
    readonly integerPart: string;
    readonly userSuffix: string;
    readonly radix: IntegerLiteral_Radix;
    readonly isUnsigned: boolean;
    readonly isLongLong: boolean;
    readonly isLong: boolean;
    readonly hasSizeSuffix: boolean;
    readonly isWB: boolean;
    readonly bitIntWidth: number;
  } {
    return ((item: any) => ({
      value: item.value,
      integerPart: item.integerPart,
      userSuffix: item.userSuffix,
      radix: item.radix,
      isUnsigned: item.isUnsigned !== 0,
      isLongLong: item.isLongLong !== 0,
      isLong: item.isLong !== 0,
      hasSizeSuffix: item.hasSizeSuffix !== 0,
      isWB: item.isWB !== 0,
      bitIntWidth: item.bitIntWidth,
    }))(cxx.readLiteralVal(this.handle, IntegerLiteralSlotBase + 3));
  }
}
export class FloatLiteral extends Literal {
  get value(): string {
    return cxx.readLiteralString(
      this.handle,
      FloatLiteralSlotBase + 0,
    ) as string;
  }
  get hashCode(): number {
    return cxx.readLiteral(this.handle, FloatLiteralSlotBase + 1);
  }
  get floatValue(): number {
    return cxx.readLiteral(this.handle, FloatLiteralSlotBase + 2);
  }
  get components(): {
    readonly value: number;
    readonly literalPart: string;
    readonly userSuffix: string;
    readonly suffix: FloatLiteral_Components_FloatingPointSuffix;
    readonly isDouble: boolean;
    readonly isFloat: boolean;
    readonly isLongDouble: boolean;
  } {
    return ((item: any) => ({
      value: item.value,
      literalPart: item.literalPart,
      userSuffix: item.userSuffix,
      suffix: item.suffix,
      isDouble: item.isDouble !== 0,
      isFloat: item.isFloat !== 0,
      isLongDouble: item.isLongDouble !== 0,
    }))(cxx.readLiteralVal(this.handle, FloatLiteralSlotBase + 3));
  }
}
export class StringLiteral extends Literal {
  get value(): string {
    return cxx.readLiteralString(
      this.handle,
      StringLiteralSlotBase + 0,
    ) as string;
  }
  get hashCode(): number {
    return cxx.readLiteral(this.handle, StringLiteralSlotBase + 1);
  }
  get encoding(): StringLiteralEncoding {
    return cxx.readLiteral(
      this.handle,
      StringLiteralSlotBase + 2,
    ) as StringLiteralEncoding;
  }
  get isRaw(): boolean {
    return cxx.readLiteral(this.handle, StringLiteralSlotBase + 3) !== 0;
  }
  get stringValue(): string {
    return cxx.readLiteralString(
      this.handle,
      StringLiteralSlotBase + 4,
    ) as string;
  }
  get charCount(): number {
    return cxx.readLiteral(this.handle, StringLiteralSlotBase + 5);
  }
  get components(): {
    readonly value: string;
    readonly userSuffix: string;
    readonly encoding: StringLiteralEncoding;
    readonly isRaw: boolean;
  } {
    return ((item: any) => ({
      value: item.value,
      userSuffix: item.userSuffix,
      encoding: item.encoding,
      isRaw: item.isRaw !== 0,
    }))(cxx.readLiteralVal(this.handle, StringLiteralSlotBase + 6));
  }
}
export class CharLiteral extends Literal {
  get value(): string {
    return cxx.readLiteralString(
      this.handle,
      CharLiteralSlotBase + 0,
    ) as string;
  }
  get hashCode(): number {
    return cxx.readLiteral(this.handle, CharLiteralSlotBase + 1);
  }
  get charValue(): number {
    return cxx.readLiteral(this.handle, CharLiteralSlotBase + 2);
  }
  get components(): {
    readonly value: number;
    readonly prefix: string;
    readonly userSuffix: string;
  } {
    return cxx.readLiteralVal(this.handle, CharLiteralSlotBase + 3) as {
      readonly value: number;
      readonly prefix: string;
      readonly userSuffix: string;
    };
  }
}
export class CommentLiteral extends Literal {
  get value(): string {
    return cxx.readLiteralString(
      this.handle,
      CommentLiteralSlotBase + 0,
    ) as string;
  }
  get hashCode(): number {
    return cxx.readLiteral(this.handle, CommentLiteralSlotBase + 1);
  }
}
export abstract class Name extends ModelObject {
  readonly kind: NameKind;
  constructor(handle: number, owner: ModelOwner, kind: NameKind) {
    super(handle, owner);
    this.kind = kind;
  }
  get hashValue(): number {
    return cxx.readName(this.handle, NameSlotBase + 0);
  }
  get text(): string {
    return cxx.readNameString(this.handle, NameSlotBase + 1) as string;
  }
}
export class Identifier extends Name {
  get hashValue(): number {
    return cxx.readName(this.handle, IdentifierSlotBase + 0);
  }
  get isAnonymous(): boolean {
    return cxx.readName(this.handle, IdentifierSlotBase + 1) !== 0;
  }
  get name(): string {
    return cxx.readNameString(this.handle, IdentifierSlotBase + 2) as string;
  }
  get value(): string {
    return cxx.readNameString(this.handle, IdentifierSlotBase + 3) as string;
  }
  get isBuiltinTypeTrait(): boolean {
    return cxx.readName(this.handle, IdentifierSlotBase + 4) !== 0;
  }
  get builtinTypeTrait(): BuiltinTypeTraitKind {
    return cxx.readName(
      this.handle,
      IdentifierSlotBase + 5,
    ) as BuiltinTypeTraitKind;
  }
  get builtinFunction(): BuiltinFunctionKind {
    return cxx.readName(
      this.handle,
      IdentifierSlotBase + 6,
    ) as BuiltinFunctionKind;
  }
  get builtinTemplate(): BuiltinTemplateKind {
    return cxx.readName(
      this.handle,
      IdentifierSlotBase + 7,
    ) as BuiltinTemplateKind;
  }
  get wellKnownName(): WellKnownName {
    return cxx.readName(this.handle, IdentifierSlotBase + 8) as WellKnownName;
  }
  get text(): string {
    return cxx.readNameString(this.handle, IdentifierSlotBase + 9) as string;
  }
}
export class OperatorId extends Name {
  get hashValue(): number {
    return cxx.readName(this.handle, OperatorIdSlotBase + 0);
  }
  get op(): TokenKind {
    return cxx.readName(this.handle, OperatorIdSlotBase + 1) as TokenKind;
  }
  get text(): string {
    return cxx.readNameString(this.handle, OperatorIdSlotBase + 2) as string;
  }
}
export class DestructorId extends Name {
  get hashValue(): number {
    return cxx.readName(this.handle, DestructorIdSlotBase + 0);
  }
  get name(): Name | undefined {
    return nameOf(
      cxx.readName(this.handle, DestructorIdSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readNameString(this.handle, DestructorIdSlotBase + 2) as string;
  }
}
export class LiteralOperatorId extends Name {
  get hashValue(): number {
    return cxx.readName(this.handle, LiteralOperatorIdSlotBase + 0);
  }
  get name(): string {
    return cxx.readNameString(
      this.handle,
      LiteralOperatorIdSlotBase + 1,
    ) as string;
  }
  get text(): string {
    return cxx.readNameString(
      this.handle,
      LiteralOperatorIdSlotBase + 2,
    ) as string;
  }
}
export class ConversionFunctionId extends Name {
  get hashValue(): number {
    return cxx.readName(this.handle, ConversionFunctionIdSlotBase + 0);
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readName(this.handle, ConversionFunctionIdSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readNameString(
      this.handle,
      ConversionFunctionIdSlotBase + 2,
    ) as string;
  }
}
export class TemplateId extends Name {
  get hashValue(): number {
    return cxx.readName(this.handle, TemplateIdSlotBase + 0);
  }
  get name(): Name | undefined {
    return nameOf(
      cxx.readName(this.handle, TemplateIdSlotBase + 1),
      this.modelOwner,
    );
  }
  get arguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return nameValItems(
      this.modelOwner,
      this.handle,
      TemplateIdSlotBase + 2,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get text(): string {
    return cxx.readNameString(this.handle, TemplateIdSlotBase + 3) as string;
  }
}
export abstract class Symbol extends ModelObject {
  readonly kind: SymbolKind;
  constructor(handle: number, owner: ModelOwner, kind: SymbolKind) {
    super(handle, owner);
    this.kind = kind;
  }
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 10) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      SymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, SymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, SymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 51);
  }
  get text(): string {
    return cxx.readSymbolString(this.handle, SymbolSlotBase + 52) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 53) !== 0;
  }
}
export abstract class ScopeSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      ScopeSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ScopeSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, ScopeSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, ScopeSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ScopeSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, ScopeSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 55) !== 0;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      ScopeSymbolSlotBase + 56,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 57) !== 0;
  }
}
export class NamespaceSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      NamespaceSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      NamespaceSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, NamespaceSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, NamespaceSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      NamespaceSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, NamespaceSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 55) !== 0;
  }
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 56) !== 0;
  }
  get hasInlineNamespaces(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 57) !== 0;
  }
  get unnamedNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 58),
      this.modelOwner,
    );
  }
  get anonNamespaceIndex(): number | undefined {
    return cxx.readSymbolVal(this.handle, NamespaceSymbolSlotBase + 59) as
      number | undefined;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      NamespaceSymbolSlotBase + 60,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 61) !== 0;
  }
}
export class ConceptSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      ConceptSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ConceptSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, ConceptSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, ConceptSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 51);
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 53),
      this.modelOwner,
    );
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 54) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 55) !== 0;
  }
  get declaration(): ConceptDefinitionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get templateArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ConceptSymbolSlotBase + 57,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ConceptSymbolSlotBase + 58,
      (item: any) =>
        (item as any[]).map((item: any) =>
          ((item: any) =>
            item.index === 0
              ? { index: 0, value: typeOf(item.value, this.modelOwner) }
              : item.index === 1
                ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                : item.index === 2
                  ? {
                      index: 2,
                      value: ((item: any) =>
                        item.index === 1
                          ? {
                              index: 1,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                StringLiteral,
                              ),
                            }
                          : item.index === 5
                            ? {
                                index: 5,
                                value: objOf(item.value, this.modelOwner, Meta),
                              }
                            : item.index === 6
                              ? {
                                  index: 6,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    InitializerList,
                                  ),
                                }
                              : item.index === 7
                                ? {
                                    index: 7,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstObject,
                                    ),
                                  }
                                : item.index === 8
                                  ? {
                                      index: 8,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstAddress,
                                      ),
                                    }
                                  : item.index === 9
                                    ? {
                                        index: 9,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstLabelAddress,
                                        ),
                                      }
                                    : item.index === 10
                                      ? {
                                          index: 10,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstComplex,
                                          ),
                                        }
                                      : item)(item.value),
                    }
                  : item.index === 3
                    ? { index: 3, value: astOf(item.value, this.modelOwner) }
                    : item)(item),
        ),
    );
  }
  get primaryTemplateSymbol(): ConceptSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 59),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 60);
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      ConceptSymbolSlotBase + 61,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 62) !== 0;
  }
}
export class DeductionGuideSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      DeductionGuideSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      DeductionGuideSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, DeductionGuideSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, DeductionGuideSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 51);
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 53),
      this.modelOwner,
    );
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 54) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 55) !== 0;
  }
  get declaration(): DeductionGuideAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get templateArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      DeductionGuideSymbolSlotBase + 57,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      DeductionGuideSymbolSlotBase + 58,
      (item: any) =>
        (item as any[]).map((item: any) =>
          ((item: any) =>
            item.index === 0
              ? { index: 0, value: typeOf(item.value, this.modelOwner) }
              : item.index === 1
                ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                : item.index === 2
                  ? {
                      index: 2,
                      value: ((item: any) =>
                        item.index === 1
                          ? {
                              index: 1,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                StringLiteral,
                              ),
                            }
                          : item.index === 5
                            ? {
                                index: 5,
                                value: objOf(item.value, this.modelOwner, Meta),
                              }
                            : item.index === 6
                              ? {
                                  index: 6,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    InitializerList,
                                  ),
                                }
                              : item.index === 7
                                ? {
                                    index: 7,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstObject,
                                    ),
                                  }
                                : item.index === 8
                                  ? {
                                      index: 8,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstAddress,
                                      ),
                                    }
                                  : item.index === 9
                                    ? {
                                        index: 9,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstLabelAddress,
                                        ),
                                      }
                                    : item.index === 10
                                      ? {
                                          index: 10,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstComplex,
                                          ),
                                        }
                                      : item)(item.value),
                    }
                  : item.index === 3
                    ? { index: 3, value: astOf(item.value, this.modelOwner) }
                    : item)(item),
        ),
    );
  }
  get primaryTemplateSymbol(): DeductionGuideSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 59),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 60);
  }
  get isExplicit(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 61) !== 0;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      DeductionGuideSymbolSlotBase + 62,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 63) !== 0;
  }
}
export class BaseClassSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      BaseClassSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      BaseClassSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, BaseClassSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, BaseClassSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 51);
  }
  get isVirtual(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 52) !== 0;
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 53),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      BaseClassSymbolSlotBase + 54,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 55) !== 0;
  }
}
export class InjectedClassNameSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 9) !== 0
    );
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      InjectedClassNameSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      InjectedClassNameSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, InjectedClassNameSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(
      cxx.readSymbolVal(this.handle, InjectedClassNameSymbolSlotBase + 13),
    );
  }
  get isNodiscard(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 14) !== 0
    );
  }
  get isUsed(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 15) !== 0
    );
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 16) !== 0
    );
  }
  get isTrivialAbi(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 17) !== 0
    );
  }
  get hasDeducedReturnType(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 18) !== 0
    );
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 21) !== 0
    );
  }
  get isNamespaceAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 22) !== 0
    );
  }
  get isConcept(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 23) !== 0
    );
  }
  get isDeductionGuide(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 24) !== 0
    );
  }
  get isClass(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 25) !== 0
    );
  }
  get isEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 26) !== 0
    );
  }
  get isScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 27) !== 0
    );
  }
  get isFunction(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 28) !== 0
    );
  }
  get isTypeAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 29) !== 0
    );
  }
  get isVariable(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 30) !== 0
    );
  }
  get isField(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 31) !== 0
    );
  }
  get isParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 32) !== 0
    );
  }
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 33) !== 0
    );
  }
  get isEnumerator(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 34) !== 0
    );
  }
  get isFunctionParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 35) !== 0
    );
  }
  get isTemplateParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 36) !== 0
    );
  }
  get isBlock(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 37) !== 0
    );
  }
  get isLambda(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 38) !== 0
    );
  }
  get isTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 39) !== 0
    );
  }
  get isNonTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 40) !== 0
    );
  }
  get isTemplateTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 41) !== 0
    );
  }
  get isConstraintTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 42) !== 0
    );
  }
  get isOverloadSet(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 43) !== 0
    );
  }
  get isBaseClass(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 44) !== 0
    );
  }
  get isInjectedClassName(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 45) !== 0
    );
  }
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 46) !== 0
    );
  }
  get isUsingDeclaration(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 47) !== 0
    );
  }
  get isClassOrNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 48) !== 0
    );
  }
  get isNamespaceName(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 49) !== 0
    );
  }
  get isEnumOrScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 50) !== 0
    );
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 51);
  }
  get classSymbol(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      InjectedClassNameSymbolSlotBase + 53,
    ) as string;
  }
  get isType(): boolean {
    return (
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 54) !== 0
    );
  }
}
export class UnresolvedSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      UnresolvedSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      UnresolvedSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, UnresolvedSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, UnresolvedSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 51);
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      UnresolvedSymbolSlotBase + 52,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, UnresolvedSymbolSlotBase + 53) !== 0;
  }
}
export class ClassSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      ClassSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, ClassSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, ClassSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 18) !== 0;
  }
  get canonical(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, ClassSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 55) !== 0;
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 57),
      this.modelOwner,
    );
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 58) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 59) !== 0;
  }
  get declaration(): SpecifierAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 60),
      this.modelOwner,
    );
  }
  get templateArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 61,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 62,
      (item: any) =>
        (item as any[]).map((item: any) =>
          ((item: any) =>
            item.index === 0
              ? { index: 0, value: typeOf(item.value, this.modelOwner) }
              : item.index === 1
                ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                : item.index === 2
                  ? {
                      index: 2,
                      value: ((item: any) =>
                        item.index === 1
                          ? {
                              index: 1,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                StringLiteral,
                              ),
                            }
                          : item.index === 5
                            ? {
                                index: 5,
                                value: objOf(item.value, this.modelOwner, Meta),
                              }
                            : item.index === 6
                              ? {
                                  index: 6,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    InitializerList,
                                  ),
                                }
                              : item.index === 7
                                ? {
                                    index: 7,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstObject,
                                    ),
                                  }
                                : item.index === 8
                                  ? {
                                      index: 8,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstAddress,
                                      ),
                                    }
                                  : item.index === 9
                                    ? {
                                        index: 9,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstLabelAddress,
                                        ),
                                      }
                                    : item.index === 10
                                      ? {
                                          index: 10,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstComplex,
                                          ),
                                        }
                                      : item)(item.value),
                    }
                  : item.index === 3
                    ? { index: 3, value: astOf(item.value, this.modelOwner) }
                    : item)(item),
        ),
    );
  }
  get primaryTemplateSymbol(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 63),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 64);
  }
  get canonicalOrNull(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 65),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 66),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<ClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 67,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get instantiationSubstitutionDepth(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 68);
  }
  get instantiationSubstitutionArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 69,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get isUnion(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 70) !== 0;
  }
  get baseClasses(): Iterable<BaseClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 71,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get constructors(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 72,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declaredConstructors(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 73,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get constructorOverloadSet(): OverloadSetSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 74),
      this.modelOwner,
    );
  }
  get deductionGuides(): Iterable<DeductionGuideSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 75,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get conversionFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 76,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get implicitConversionFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 77,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get visibleConversionFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 78,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get destructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 79),
      this.modelOwner,
    );
  }
  get defaultConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 80),
      this.modelOwner,
    );
  }
  get copyConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 81),
      this.modelOwner,
    );
  }
  get moveConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 82),
      this.modelOwner,
    );
  }
  get copyAssignmentOperator(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 83),
      this.modelOwner,
    );
  }
  get moveAssignmentOperator(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 84),
      this.modelOwner,
    );
  }
  get hasUserDeclaredConstructors(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 85) !== 0;
  }
  get hasInheritedConstructors(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 86) !== 0;
  }
  get hasVirtualFunctions(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 87) !== 0;
  }
  get hasVirtualBaseClasses(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 88) !== 0;
  }
  get convertingConstructors(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 89,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get isFinal(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 90) !== 0;
  }
  get isComplete(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 91) !== 0;
  }
  get isFriend(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 92) !== 0;
  }
  get isPolymorphic(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 93) !== 0;
  }
  get isAbstract(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 94) !== 0;
  }
  get hasVirtualDestructor(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 95) !== 0;
  }
  get isAccessControlDisabled(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 96) !== 0;
  }
  get sizeInBytes(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 97);
  }
  get alignment(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 98);
  }
  get explicitAlignment(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 99);
  }
  get packAlignment(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 100);
  }
  get befriendingClasses(): Iterable<ClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 101,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get templateFriendships(): Iterable<{
    readonly arguments: ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >;
    readonly befriendingClass: ClassSymbol | undefined;
  }> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 102,
      (item: any) =>
        ((item: any) => ({
          arguments: (item.arguments as any[]).map((item: any) =>
            ((item: any) =>
              item.index === 0
                ? { index: 0, value: typeOf(item.value, this.modelOwner) }
                : item.index === 1
                  ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                  : item.index === 2
                    ? {
                        index: 2,
                        value: ((item: any) =>
                          item.index === 1
                            ? {
                                index: 1,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  StringLiteral,
                                ),
                              }
                            : item.index === 5
                              ? {
                                  index: 5,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    Meta,
                                  ),
                                }
                              : item.index === 6
                                ? {
                                    index: 6,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      InitializerList,
                                    ),
                                  }
                                : item.index === 7
                                  ? {
                                      index: 7,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstObject,
                                      ),
                                    }
                                  : item.index === 8
                                    ? {
                                        index: 8,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstAddress,
                                        ),
                                      }
                                    : item.index === 9
                                      ? {
                                          index: 9,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstLabelAddress,
                                          ),
                                        }
                                      : item.index === 10
                                        ? {
                                            index: 10,
                                            value: objOf(
                                              item.value,
                                              this.modelOwner,
                                              ConstComplex,
                                            ),
                                          }
                                        : item)(item.value),
                      }
                    : item.index === 3
                      ? { index: 3, value: astOf(item.value, this.modelOwner) }
                      : item)(item),
          ),
          befriendingClass: symbolOf(item.befriendingClass, this.modelOwner),
        }))(item),
    );
  }
  get baseClassRepetition(): {
    readonly nonDiamondRepeat: boolean;
    readonly diamondShaped: boolean;
  } {
    return ((item: any) => ({
      nonDiamondRepeat: item.nonDiamondRepeat !== 0,
      diamondShaped: item.diamondShaped !== 0,
    }))(cxx.readSymbolVal(this.handle, ClassSymbolSlotBase + 103));
  }
  get flags(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 104);
  }
  get isClosureType(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 105) !== 0;
  }
  get hasLambdaCapture(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 106) !== 0;
  }
  get capturedThisField(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 107),
      this.modelOwner,
    );
  }
  get closureDiscriminator(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 108);
  }
  get instantiationPattern(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 109),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      ClassSymbolSlotBase + 110,
    ) as string;
  }
  get templatePattern(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 111),
      this.modelOwner,
    );
  }
  get expandedTemplateArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 112,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get expandedTemplateArgumentTexts(): Iterable<string> {
    return symbolStringItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 113,
      (item: any) => item,
    );
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 114) !== 0;
  }
}
export class EnumSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      EnumSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      EnumSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, EnumSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, EnumSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      EnumSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, EnumSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 55) !== 0;
  }
  get hasFixedUnderlyingType(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 56) !== 0;
  }
  get isDefined(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 57) !== 0;
  }
  get underlyingType(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 58),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(this.handle, EnumSymbolSlotBase + 59) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 60) !== 0;
  }
}
export class ScopedEnumSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      ScopedEnumSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ScopedEnumSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, ScopedEnumSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, ScopedEnumSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ScopedEnumSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, ScopedEnumSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 55) !== 0;
  }
  get underlyingType(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get isDefined(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 57) !== 0;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      ScopedEnumSymbolSlotBase + 58,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 59) !== 0;
  }
}
export class FunctionSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      FunctionSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, FunctionSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, FunctionSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 18) !== 0;
  }
  get canonical(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, FunctionSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 55) !== 0;
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 57),
      this.modelOwner,
    );
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 58) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 59) !== 0;
  }
  get declaration(): FunctionDefinitionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 60),
      this.modelOwner,
    );
  }
  get templateArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 61,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 62,
      (item: any) =>
        (item as any[]).map((item: any) =>
          ((item: any) =>
            item.index === 0
              ? { index: 0, value: typeOf(item.value, this.modelOwner) }
              : item.index === 1
                ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                : item.index === 2
                  ? {
                      index: 2,
                      value: ((item: any) =>
                        item.index === 1
                          ? {
                              index: 1,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                StringLiteral,
                              ),
                            }
                          : item.index === 5
                            ? {
                                index: 5,
                                value: objOf(item.value, this.modelOwner, Meta),
                              }
                            : item.index === 6
                              ? {
                                  index: 6,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    InitializerList,
                                  ),
                                }
                              : item.index === 7
                                ? {
                                    index: 7,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstObject,
                                    ),
                                  }
                                : item.index === 8
                                  ? {
                                      index: 8,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstAddress,
                                      ),
                                    }
                                  : item.index === 9
                                    ? {
                                        index: 9,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstLabelAddress,
                                        ),
                                      }
                                    : item.index === 10
                                      ? {
                                          index: 10,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstComplex,
                                          ),
                                        }
                                      : item)(item.value),
                    }
                  : item.index === 3
                    ? { index: 3, value: astOf(item.value, this.modelOwner) }
                    : item)(item),
        ),
    );
  }
  get primaryTemplateSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 63),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 64);
  }
  get canonicalOrNull(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 65),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 66),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 67,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get functionParameters(): FunctionParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 68),
      this.modelOwner,
    );
  }
  get isDefined(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 69) !== 0;
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 70) !== 0;
  }
  get isExtern(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 71) !== 0;
  }
  get isFriend(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 72) !== 0;
  }
  get isImplicitObjectMemberFunction(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 73) !== 0;
  }
  get hasExplicitObjectParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 74) !== 0;
  }
  get explicitObjectParameter(): ParameterSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 75),
      this.modelOwner,
    );
  }
  get parameters(): Iterable<ParameterSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 76,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 77) !== 0;
  }
  get isConsteval(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 78) !== 0;
  }
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 79) !== 0;
  }
  get isVirtual(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 80) !== 0;
  }
  get isExplicit(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 81) !== 0;
  }
  get isDeleted(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 82) !== 0;
  }
  get isDefaulted(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 83) !== 0;
  }
  get isPure(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 84) !== 0;
  }
  get isOverride(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 85) !== 0;
  }
  get isFinal(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 86) !== 0;
  }
  get hasNoPrototype(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 87) !== 0;
  }
  get hasExceptionSpecifier(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 88) !== 0;
  }
  get isDefinitionRequired(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 89) !== 0;
  }
  get isNoReturn(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 90) !== 0;
  }
  get builtinKind(): BuiltinFunctionKind {
    return cxx.readSymbol(
      this.handle,
      FunctionSymbolSlotBase + 91,
    ) as BuiltinFunctionKind;
  }
  get trailingRequiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 92),
      this.modelOwner,
    );
  }
  get isConstructor(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 93) !== 0;
  }
  get isDestructor(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 94) !== 0;
  }
  get languageLinkage(): LanguageKind {
    return cxx.readSymbol(
      this.handle,
      FunctionSymbolSlotBase + 95,
    ) as LanguageKind;
  }
  get hasCLinkage(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 96) !== 0;
  }
  get externalName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 97),
      this.modelOwner,
    );
  }
  get aliasName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 98),
      this.modelOwner,
    );
  }
  get hasHiddenVisibility(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 99) !== 0;
  }
  get importModule(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 100),
      this.modelOwner,
    );
  }
  get importName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 101),
      this.modelOwner,
    );
  }
  get exportName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 102),
      this.modelOwner,
    );
  }
  get hasPendingBody(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 103) !== 0;
  }
  get hasUninstantiatedBody(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 104) !== 0;
  }
  get pendingBody():
    | {
        readonly originalDefinition: FunctionDefinitionAST | undefined;
        readonly templateArguments: ReadonlyArray<
          | { readonly index: 0; readonly value: Type | undefined }
          | { readonly index: 1; readonly value: Symbol | undefined }
          | {
              readonly index: 2;
              readonly value:
                | { readonly index: 0; readonly value: bigint }
                | {
                    readonly index: 1;
                    readonly value: StringLiteral | undefined;
                  }
                | { readonly index: 2; readonly value: number }
                | { readonly index: 3; readonly value: number }
                | { readonly index: 4; readonly value: number }
                | { readonly index: 5; readonly value: Meta | undefined }
                | {
                    readonly index: 6;
                    readonly value: InitializerList | undefined;
                  }
                | { readonly index: 7; readonly value: ConstObject | undefined }
                | {
                    readonly index: 8;
                    readonly value: ConstAddress | undefined;
                  }
                | {
                    readonly index: 9;
                    readonly value: ConstLabelAddress | undefined;
                  }
                | {
                    readonly index: 10;
                    readonly value: ConstComplex | undefined;
                  }
                | { readonly index: 11; readonly value: {} };
            }
          | { readonly index: 3; readonly value: ExpressionAST | undefined }
        >;
        readonly parentScope: ScopeSymbol | undefined;
        readonly depth: number;
      }
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : ((item: any) => ({
            originalDefinition: astOf(item.originalDefinition, this.modelOwner),
            templateArguments: (item.templateArguments as any[]).map(
              (item: any) =>
                ((item: any) =>
                  item.index === 0
                    ? { index: 0, value: typeOf(item.value, this.modelOwner) }
                    : item.index === 1
                      ? {
                          index: 1,
                          value: symbolOf(item.value, this.modelOwner),
                        }
                      : item.index === 2
                        ? {
                            index: 2,
                            value: ((item: any) =>
                              item.index === 1
                                ? {
                                    index: 1,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      StringLiteral,
                                    ),
                                  }
                                : item.index === 5
                                  ? {
                                      index: 5,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        Meta,
                                      ),
                                    }
                                  : item.index === 6
                                    ? {
                                        index: 6,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          InitializerList,
                                        ),
                                      }
                                    : item.index === 7
                                      ? {
                                          index: 7,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstObject,
                                          ),
                                        }
                                      : item.index === 8
                                        ? {
                                            index: 8,
                                            value: objOf(
                                              item.value,
                                              this.modelOwner,
                                              ConstAddress,
                                            ),
                                          }
                                        : item.index === 9
                                          ? {
                                              index: 9,
                                              value: objOf(
                                                item.value,
                                                this.modelOwner,
                                                ConstLabelAddress,
                                              ),
                                            }
                                          : item.index === 10
                                            ? {
                                                index: 10,
                                                value: objOf(
                                                  item.value,
                                                  this.modelOwner,
                                                  ConstComplex,
                                                ),
                                              }
                                            : item)(item.value),
                          }
                        : item.index === 3
                          ? {
                              index: 3,
                              value: astOf(item.value, this.modelOwner),
                            }
                          : item)(item),
            ),
            parentScope: symbolOf(item.parentScope, this.modelOwner),
            depth: item.depth,
          }))(item))(
      cxx.readSymbolVal(this.handle, FunctionSymbolSlotBase + 105),
    );
  }
  get pendingExceptionSpecification():
    | {
        readonly original: NoexceptSpecifierAST | undefined;
        readonly instance: NoexceptSpecifierAST | undefined;
        readonly originalFunction: FunctionSymbol | undefined;
        readonly templateArguments: ReadonlyArray<
          | { readonly index: 0; readonly value: Type | undefined }
          | { readonly index: 1; readonly value: Symbol | undefined }
          | {
              readonly index: 2;
              readonly value:
                | { readonly index: 0; readonly value: bigint }
                | {
                    readonly index: 1;
                    readonly value: StringLiteral | undefined;
                  }
                | { readonly index: 2; readonly value: number }
                | { readonly index: 3; readonly value: number }
                | { readonly index: 4; readonly value: number }
                | { readonly index: 5; readonly value: Meta | undefined }
                | {
                    readonly index: 6;
                    readonly value: InitializerList | undefined;
                  }
                | { readonly index: 7; readonly value: ConstObject | undefined }
                | {
                    readonly index: 8;
                    readonly value: ConstAddress | undefined;
                  }
                | {
                    readonly index: 9;
                    readonly value: ConstLabelAddress | undefined;
                  }
                | {
                    readonly index: 10;
                    readonly value: ConstComplex | undefined;
                  }
                | { readonly index: 11; readonly value: {} };
            }
          | { readonly index: 3; readonly value: ExpressionAST | undefined }
        >;
        readonly parentScope: ScopeSymbol | undefined;
        readonly depth: number;
        readonly state: PendingExceptionSpecificationState;
        readonly recursionDiagnosed: boolean;
      }
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : ((item: any) => ({
            original: astOf(item.original, this.modelOwner),
            instance: astOf(item.instance, this.modelOwner),
            originalFunction: symbolOf(item.originalFunction, this.modelOwner),
            templateArguments: (item.templateArguments as any[]).map(
              (item: any) =>
                ((item: any) =>
                  item.index === 0
                    ? { index: 0, value: typeOf(item.value, this.modelOwner) }
                    : item.index === 1
                      ? {
                          index: 1,
                          value: symbolOf(item.value, this.modelOwner),
                        }
                      : item.index === 2
                        ? {
                            index: 2,
                            value: ((item: any) =>
                              item.index === 1
                                ? {
                                    index: 1,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      StringLiteral,
                                    ),
                                  }
                                : item.index === 5
                                  ? {
                                      index: 5,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        Meta,
                                      ),
                                    }
                                  : item.index === 6
                                    ? {
                                        index: 6,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          InitializerList,
                                        ),
                                      }
                                    : item.index === 7
                                      ? {
                                          index: 7,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstObject,
                                          ),
                                        }
                                      : item.index === 8
                                        ? {
                                            index: 8,
                                            value: objOf(
                                              item.value,
                                              this.modelOwner,
                                              ConstAddress,
                                            ),
                                          }
                                        : item.index === 9
                                          ? {
                                              index: 9,
                                              value: objOf(
                                                item.value,
                                                this.modelOwner,
                                                ConstLabelAddress,
                                              ),
                                            }
                                          : item.index === 10
                                            ? {
                                                index: 10,
                                                value: objOf(
                                                  item.value,
                                                  this.modelOwner,
                                                  ConstComplex,
                                                ),
                                              }
                                            : item)(item.value),
                          }
                        : item.index === 3
                          ? {
                              index: 3,
                              value: astOf(item.value, this.modelOwner),
                            }
                          : item)(item),
            ),
            parentScope: symbolOf(item.parentScope, this.modelOwner),
            depth: item.depth,
            state: item.state,
            recursionDiagnosed: item.recursionDiagnosed !== 0,
          }))(item))(
      cxx.readSymbolVal(this.handle, FunctionSymbolSlotBase + 106),
    );
  }
  get vtableSlotIndex(): number {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 107);
  }
  get overriddenFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 108,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get befriendingClasses(): Iterable<ClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 109,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get templateFriendships(): Iterable<{
    readonly arguments: ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >;
    readonly befriendingClass: ClassSymbol | undefined;
  }> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 110,
      (item: any) =>
        ((item: any) => ({
          arguments: (item.arguments as any[]).map((item: any) =>
            ((item: any) =>
              item.index === 0
                ? { index: 0, value: typeOf(item.value, this.modelOwner) }
                : item.index === 1
                  ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                  : item.index === 2
                    ? {
                        index: 2,
                        value: ((item: any) =>
                          item.index === 1
                            ? {
                                index: 1,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  StringLiteral,
                                ),
                              }
                            : item.index === 5
                              ? {
                                  index: 5,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    Meta,
                                  ),
                                }
                              : item.index === 6
                                ? {
                                    index: 6,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      InitializerList,
                                    ),
                                  }
                                : item.index === 7
                                  ? {
                                      index: 7,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstObject,
                                      ),
                                    }
                                  : item.index === 8
                                    ? {
                                        index: 8,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstAddress,
                                        ),
                                      }
                                    : item.index === 9
                                      ? {
                                          index: 9,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstLabelAddress,
                                          ),
                                        }
                                      : item.index === 10
                                        ? {
                                            index: 10,
                                            value: objOf(
                                              item.value,
                                              this.modelOwner,
                                              ConstComplex,
                                            ),
                                          }
                                        : item)(item.value),
                      }
                    : item.index === 3
                      ? { index: 3, value: astOf(item.value, this.modelOwner) }
                      : item)(item),
          ),
          befriendingClass: symbolOf(item.befriendingClass, this.modelOwner),
        }))(item),
    );
  }
  get delegatingConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 111),
      this.modelOwner,
    );
  }
  get completeObjectVariant(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 112),
      this.modelOwner,
    );
  }
  get deletingDtorVariant(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 113),
      this.modelOwner,
    );
  }
  get structorPrincipal(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 114),
      this.modelOwner,
    );
  }
  get isStructorVariant(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 115) !== 0;
  }
  get inheritedConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 116),
      this.modelOwner,
    );
  }
  get inheritedConstructorOrigin(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 117),
      this.modelOwner,
    );
  }
  get isDeletingDtorVariant(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 118) !== 0;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      FunctionSymbolSlotBase + 119,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 120) !== 0;
  }
}
export class OverloadSetSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      OverloadSetSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      OverloadSetSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, OverloadSetSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, OverloadSetSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 51);
  }
  get functions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      OverloadSetSymbolSlotBase + 52,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declaredFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      OverloadSetSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDeclarations(): Iterable<UsingDeclarationSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      OverloadSetSymbolSlotBase + 54,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      OverloadSetSymbolSlotBase + 55,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 56) !== 0;
  }
}
export class LambdaSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      LambdaSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      LambdaSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, LambdaSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, LambdaSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      LambdaSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, LambdaSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 55) !== 0;
  }
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 56) !== 0;
  }
  get isConsteval(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 57) !== 0;
  }
  get isMutable(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 58) !== 0;
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 59) !== 0;
  }
  get isTemplate(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 60) !== 0;
  }
  get isInTemplate(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 61) !== 0;
  }
  get closureType(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 62),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      LambdaSymbolSlotBase + 63,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 64) !== 0;
  }
}
export class FunctionParametersSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 9) !== 0
    );
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      FunctionParametersSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionParametersSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, FunctionParametersSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(
      cxx.readSymbolVal(this.handle, FunctionParametersSymbolSlotBase + 13),
    );
  }
  get isNodiscard(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 14) !== 0
    );
  }
  get isUsed(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 15) !== 0
    );
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 16) !== 0
    );
  }
  get isTrivialAbi(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 17) !== 0
    );
  }
  get hasDeducedReturnType(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 18) !== 0
    );
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 21) !== 0
    );
  }
  get isNamespaceAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 22) !== 0
    );
  }
  get isConcept(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 23) !== 0
    );
  }
  get isDeductionGuide(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 24) !== 0
    );
  }
  get isClass(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 25) !== 0
    );
  }
  get isEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 26) !== 0
    );
  }
  get isScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 27) !== 0
    );
  }
  get isFunction(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 28) !== 0
    );
  }
  get isTypeAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 29) !== 0
    );
  }
  get isVariable(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 30) !== 0
    );
  }
  get isField(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 31) !== 0
    );
  }
  get isParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 32) !== 0
    );
  }
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 33) !== 0
    );
  }
  get isEnumerator(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 34) !== 0
    );
  }
  get isFunctionParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 35) !== 0
    );
  }
  get isTemplateParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 36) !== 0
    );
  }
  get isBlock(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 37) !== 0
    );
  }
  get isLambda(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 38) !== 0
    );
  }
  get isTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 39) !== 0
    );
  }
  get isNonTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 40) !== 0
    );
  }
  get isTemplateTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 41) !== 0
    );
  }
  get isConstraintTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 42) !== 0
    );
  }
  get isOverloadSet(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 43) !== 0
    );
  }
  get isBaseClass(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 44) !== 0
    );
  }
  get isInjectedClassName(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 45) !== 0
    );
  }
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 46) !== 0
    );
  }
  get isUsingDeclaration(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 47) !== 0
    );
  }
  get isClassOrNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 48) !== 0
    );
  }
  get isNamespaceName(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 49) !== 0
    );
  }
  get isEnumOrScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 50) !== 0
    );
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 52) !== 0
    );
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionParametersSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(
        this.handle,
        FunctionParametersSymbolSlotBase + 54,
      ) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 55) !== 0
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      FunctionParametersSymbolSlotBase + 56,
    ) as string;
  }
  get isType(): boolean {
    return (
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 57) !== 0
    );
  }
}
export class TemplateParametersSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 9) !== 0
    );
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      TemplateParametersSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TemplateParametersSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, TemplateParametersSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(
      cxx.readSymbolVal(this.handle, TemplateParametersSymbolSlotBase + 13),
    );
  }
  get isNodiscard(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 14) !== 0
    );
  }
  get isUsed(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 15) !== 0
    );
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 16) !== 0
    );
  }
  get isTrivialAbi(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 17) !== 0
    );
  }
  get hasDeducedReturnType(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 18) !== 0
    );
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 21) !== 0
    );
  }
  get isNamespaceAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 22) !== 0
    );
  }
  get isConcept(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 23) !== 0
    );
  }
  get isDeductionGuide(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 24) !== 0
    );
  }
  get isClass(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 25) !== 0
    );
  }
  get isEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 26) !== 0
    );
  }
  get isScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 27) !== 0
    );
  }
  get isFunction(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 28) !== 0
    );
  }
  get isTypeAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 29) !== 0
    );
  }
  get isVariable(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 30) !== 0
    );
  }
  get isField(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 31) !== 0
    );
  }
  get isParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 32) !== 0
    );
  }
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 33) !== 0
    );
  }
  get isEnumerator(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 34) !== 0
    );
  }
  get isFunctionParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 35) !== 0
    );
  }
  get isTemplateParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 36) !== 0
    );
  }
  get isBlock(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 37) !== 0
    );
  }
  get isLambda(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 38) !== 0
    );
  }
  get isTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 39) !== 0
    );
  }
  get isNonTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 40) !== 0
    );
  }
  get isTemplateTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 41) !== 0
    );
  }
  get isConstraintTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 42) !== 0
    );
  }
  get isOverloadSet(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 43) !== 0
    );
  }
  get isBaseClass(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 44) !== 0
    );
  }
  get isInjectedClassName(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 45) !== 0
    );
  }
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 46) !== 0
    );
  }
  get isUsingDeclaration(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 47) !== 0
    );
  }
  get isClassOrNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 48) !== 0
    );
  }
  get isNamespaceName(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 49) !== 0
    );
  }
  get isEnumOrScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 50) !== 0
    );
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 52) !== 0
    );
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TemplateParametersSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(
        this.handle,
        TemplateParametersSymbolSlotBase + 54,
      ) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 55) !== 0
    );
  }
  get isExplicitTemplateSpecialization(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 56) !== 0
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      TemplateParametersSymbolSlotBase + 57,
    ) as string;
  }
  get isType(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 58) !== 0
    );
  }
}
export class BlockSymbol extends ScopeSymbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      BlockSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      BlockSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, BlockSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, BlockSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BlockSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 51);
  }
  get empty(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 52) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      BlockSymbolSlotBase + 53,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, BlockSymbolSlotBase + 54) as any[]
    ).map((item: any) => symbolOf(item, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 55) !== 0;
  }
  get isOutermostBlockScope(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 56) !== 0;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      BlockSymbolSlotBase + 57,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 58) !== 0;
  }
}
export class TypeAliasSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      TypeAliasSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, TypeAliasSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, TypeAliasSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 18) !== 0;
  }
  get canonical(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 51);
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 53),
      this.modelOwner,
    );
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 54) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 55) !== 0;
  }
  get declaration(): AliasDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get templateArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 57,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 58,
      (item: any) =>
        (item as any[]).map((item: any) =>
          ((item: any) =>
            item.index === 0
              ? { index: 0, value: typeOf(item.value, this.modelOwner) }
              : item.index === 1
                ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                : item.index === 2
                  ? {
                      index: 2,
                      value: ((item: any) =>
                        item.index === 1
                          ? {
                              index: 1,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                StringLiteral,
                              ),
                            }
                          : item.index === 5
                            ? {
                                index: 5,
                                value: objOf(item.value, this.modelOwner, Meta),
                              }
                            : item.index === 6
                              ? {
                                  index: 6,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    InitializerList,
                                  ),
                                }
                              : item.index === 7
                                ? {
                                    index: 7,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstObject,
                                    ),
                                  }
                                : item.index === 8
                                  ? {
                                      index: 8,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstAddress,
                                      ),
                                    }
                                  : item.index === 9
                                    ? {
                                        index: 9,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstLabelAddress,
                                        ),
                                      }
                                    : item.index === 10
                                      ? {
                                          index: 10,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstComplex,
                                          ),
                                        }
                                      : item)(item.value),
                    }
                  : item.index === 3
                    ? { index: 3, value: astOf(item.value, this.modelOwner) }
                    : item)(item),
        ),
    );
  }
  get primaryTemplateSymbol(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 59),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 60);
  }
  get canonicalOrNull(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 61),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 62),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<TypeAliasSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 63,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get expansionTypeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 64),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      TypeAliasSymbolSlotBase + 65,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 66) !== 0;
  }
}
export class VariableSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      VariableSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, VariableSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, VariableSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 18) !== 0;
  }
  get canonical(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 51);
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 53),
      this.modelOwner,
    );
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 54) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 55) !== 0;
  }
  get declaration(): SimpleDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get templateArguments(): Iterable<
    | { readonly index: 0; readonly value: Type | undefined }
    | { readonly index: 1; readonly value: Symbol | undefined }
    | {
        readonly index: 2;
        readonly value:
          | { readonly index: 0; readonly value: bigint }
          | { readonly index: 1; readonly value: StringLiteral | undefined }
          | { readonly index: 2; readonly value: number }
          | { readonly index: 3; readonly value: number }
          | { readonly index: 4; readonly value: number }
          | { readonly index: 5; readonly value: Meta | undefined }
          | { readonly index: 6; readonly value: InitializerList | undefined }
          | { readonly index: 7; readonly value: ConstObject | undefined }
          | { readonly index: 8; readonly value: ConstAddress | undefined }
          | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
          | { readonly index: 10; readonly value: ConstComplex | undefined }
          | { readonly index: 11; readonly value: {} };
      }
    | { readonly index: 3; readonly value: ExpressionAST | undefined }
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 57,
      (item: any) =>
        ((item: any) =>
          item.index === 0
            ? { index: 0, value: typeOf(item.value, this.modelOwner) }
            : item.index === 1
              ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
              : item.index === 2
                ? {
                    index: 2,
                    value: ((item: any) =>
                      item.index === 1
                        ? {
                            index: 1,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              StringLiteral,
                            ),
                          }
                        : item.index === 5
                          ? {
                              index: 5,
                              value: objOf(item.value, this.modelOwner, Meta),
                            }
                          : item.index === 6
                            ? {
                                index: 6,
                                value: objOf(
                                  item.value,
                                  this.modelOwner,
                                  InitializerList,
                                ),
                              }
                            : item.index === 7
                              ? {
                                  index: 7,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    ConstObject,
                                  ),
                                }
                              : item.index === 8
                                ? {
                                    index: 8,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstAddress,
                                    ),
                                  }
                                : item.index === 9
                                  ? {
                                      index: 9,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstLabelAddress,
                                      ),
                                    }
                                  : item.index === 10
                                    ? {
                                        index: 10,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstComplex,
                                        ),
                                      }
                                    : item)(item.value),
                  }
                : item.index === 3
                  ? { index: 3, value: astOf(item.value, this.modelOwner) }
                  : item)(item),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<
      | { readonly index: 0; readonly value: Type | undefined }
      | { readonly index: 1; readonly value: Symbol | undefined }
      | {
          readonly index: 2;
          readonly value:
            | { readonly index: 0; readonly value: bigint }
            | { readonly index: 1; readonly value: StringLiteral | undefined }
            | { readonly index: 2; readonly value: number }
            | { readonly index: 3; readonly value: number }
            | { readonly index: 4; readonly value: number }
            | { readonly index: 5; readonly value: Meta | undefined }
            | { readonly index: 6; readonly value: InitializerList | undefined }
            | { readonly index: 7; readonly value: ConstObject | undefined }
            | { readonly index: 8; readonly value: ConstAddress | undefined }
            | {
                readonly index: 9;
                readonly value: ConstLabelAddress | undefined;
              }
            | { readonly index: 10; readonly value: ConstComplex | undefined }
            | { readonly index: 11; readonly value: {} };
        }
      | { readonly index: 3; readonly value: ExpressionAST | undefined }
    >
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 58,
      (item: any) =>
        (item as any[]).map((item: any) =>
          ((item: any) =>
            item.index === 0
              ? { index: 0, value: typeOf(item.value, this.modelOwner) }
              : item.index === 1
                ? { index: 1, value: symbolOf(item.value, this.modelOwner) }
                : item.index === 2
                  ? {
                      index: 2,
                      value: ((item: any) =>
                        item.index === 1
                          ? {
                              index: 1,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                StringLiteral,
                              ),
                            }
                          : item.index === 5
                            ? {
                                index: 5,
                                value: objOf(item.value, this.modelOwner, Meta),
                              }
                            : item.index === 6
                              ? {
                                  index: 6,
                                  value: objOf(
                                    item.value,
                                    this.modelOwner,
                                    InitializerList,
                                  ),
                                }
                              : item.index === 7
                                ? {
                                    index: 7,
                                    value: objOf(
                                      item.value,
                                      this.modelOwner,
                                      ConstObject,
                                    ),
                                  }
                                : item.index === 8
                                  ? {
                                      index: 8,
                                      value: objOf(
                                        item.value,
                                        this.modelOwner,
                                        ConstAddress,
                                      ),
                                    }
                                  : item.index === 9
                                    ? {
                                        index: 9,
                                        value: objOf(
                                          item.value,
                                          this.modelOwner,
                                          ConstLabelAddress,
                                        ),
                                      }
                                    : item.index === 10
                                      ? {
                                          index: 10,
                                          value: objOf(
                                            item.value,
                                            this.modelOwner,
                                            ConstComplex,
                                          ),
                                        }
                                      : item)(item.value),
                    }
                  : item.index === 3
                    ? { index: 3, value: astOf(item.value, this.modelOwner) }
                    : item)(item),
        ),
    );
  }
  get primaryTemplateSymbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 59),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 60);
  }
  get canonicalOrNull(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 61),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 62),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<VariableSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 63,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 64) !== 0;
  }
  get isThreadLocal(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 65) !== 0;
  }
  get isExtern(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 66) !== 0;
  }
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 67) !== 0;
  }
  get isConstinit(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 68) !== 0;
  }
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 69) !== 0;
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 70),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 71),
      this.modelOwner,
    );
  }
  get constValue():
    | { readonly index: 0; readonly value: bigint }
    | { readonly index: 1; readonly value: StringLiteral | undefined }
    | { readonly index: 2; readonly value: number }
    | { readonly index: 3; readonly value: number }
    | { readonly index: 4; readonly value: number }
    | { readonly index: 5; readonly value: Meta | undefined }
    | { readonly index: 6; readonly value: InitializerList | undefined }
    | { readonly index: 7; readonly value: ConstObject | undefined }
    | { readonly index: 8; readonly value: ConstAddress | undefined }
    | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
    | { readonly index: 10; readonly value: ConstComplex | undefined }
    | { readonly index: 11; readonly value: {} }
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : ((item: any) =>
            item.index === 1
              ? {
                  index: 1,
                  value: objOf(item.value, this.modelOwner, StringLiteral),
                }
              : item.index === 5
                ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
                : item.index === 6
                  ? {
                      index: 6,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        InitializerList,
                      ),
                    }
                  : item.index === 7
                    ? {
                        index: 7,
                        value: objOf(item.value, this.modelOwner, ConstObject),
                      }
                    : item.index === 8
                      ? {
                          index: 8,
                          value: objOf(
                            item.value,
                            this.modelOwner,
                            ConstAddress,
                          ),
                        }
                      : item.index === 9
                        ? {
                            index: 9,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              ConstLabelAddress,
                            ),
                          }
                        : item.index === 10
                          ? {
                              index: 10,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                ConstComplex,
                              ),
                            }
                          : item)(item))(
      cxx.readSymbolVal(this.handle, VariableSymbolSlotBase + 72),
    );
  }
  get explicitAlignment(): number {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 73);
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      VariableSymbolSlotBase + 74,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 75) !== 0;
  }
}
export class FieldSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      FieldSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FieldSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 51);
  }
  get isBitField(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 52) !== 0;
  }
  get bitFieldOffset(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 53);
  }
  get bitFieldWidth():
    | { readonly index: 0; readonly value: bigint }
    | { readonly index: 1; readonly value: StringLiteral | undefined }
    | { readonly index: 2; readonly value: number }
    | { readonly index: 3; readonly value: number }
    | { readonly index: 4; readonly value: number }
    | { readonly index: 5; readonly value: Meta | undefined }
    | { readonly index: 6; readonly value: InitializerList | undefined }
    | { readonly index: 7; readonly value: ConstObject | undefined }
    | { readonly index: 8; readonly value: ConstAddress | undefined }
    | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
    | { readonly index: 10; readonly value: ConstComplex | undefined }
    | { readonly index: 11; readonly value: {} }
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : ((item: any) =>
            item.index === 1
              ? {
                  index: 1,
                  value: objOf(item.value, this.modelOwner, StringLiteral),
                }
              : item.index === 5
                ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
                : item.index === 6
                  ? {
                      index: 6,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        InitializerList,
                      ),
                    }
                  : item.index === 7
                    ? {
                        index: 7,
                        value: objOf(item.value, this.modelOwner, ConstObject),
                      }
                    : item.index === 8
                      ? {
                          index: 8,
                          value: objOf(
                            item.value,
                            this.modelOwner,
                            ConstAddress,
                          ),
                        }
                      : item.index === 9
                        ? {
                            index: 9,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              ConstLabelAddress,
                            ),
                          }
                        : item.index === 10
                          ? {
                              index: 10,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                ConstComplex,
                              ),
                            }
                          : item)(item))(
      cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 54),
    );
  }
  get isExtern(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 55) !== 0;
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 56) !== 0;
  }
  get isThreadLocal(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 57) !== 0;
  }
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 58) !== 0;
  }
  get isConstinit(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 59) !== 0;
  }
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 60) !== 0;
  }
  get isMutable(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 61) !== 0;
  }
  get isNoUniqueAddress(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 62) !== 0;
  }
  get offsetInClass(): bigint | undefined {
    return cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 63) as
      bigint | undefined;
  }
  get localOffset(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 64);
  }
  get alignment(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 65);
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 66),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 67),
      this.modelOwner,
    );
  }
  get constValue():
    | { readonly index: 0; readonly value: bigint }
    | { readonly index: 1; readonly value: StringLiteral | undefined }
    | { readonly index: 2; readonly value: number }
    | { readonly index: 3; readonly value: number }
    | { readonly index: 4; readonly value: number }
    | { readonly index: 5; readonly value: Meta | undefined }
    | { readonly index: 6; readonly value: InitializerList | undefined }
    | { readonly index: 7; readonly value: ConstObject | undefined }
    | { readonly index: 8; readonly value: ConstAddress | undefined }
    | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
    | { readonly index: 10; readonly value: ConstComplex | undefined }
    | { readonly index: 11; readonly value: {} }
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : ((item: any) =>
            item.index === 1
              ? {
                  index: 1,
                  value: objOf(item.value, this.modelOwner, StringLiteral),
                }
              : item.index === 5
                ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
                : item.index === 6
                  ? {
                      index: 6,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        InitializerList,
                      ),
                    }
                  : item.index === 7
                    ? {
                        index: 7,
                        value: objOf(item.value, this.modelOwner, ConstObject),
                      }
                    : item.index === 8
                      ? {
                          index: 8,
                          value: objOf(
                            item.value,
                            this.modelOwner,
                            ConstAddress,
                          ),
                        }
                      : item.index === 9
                        ? {
                            index: 9,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              ConstLabelAddress,
                            ),
                          }
                        : item.index === 10
                          ? {
                              index: 10,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                ConstComplex,
                              ),
                            }
                          : item)(item))(
      cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 68),
    );
  }
  get isDefinitionRequired(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 69) !== 0;
  }
  get hasPendingInitializer(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 70) !== 0;
  }
  get hasInitializer(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 71) !== 0;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      FieldSymbolSlotBase + 72,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 73) !== 0;
  }
}
export class ParameterSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      ParameterSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ParameterSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, ParameterSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, ParameterSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 51);
  }
  get defaultArgument(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get isExplicitObject(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 53) !== 0;
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      ParameterSymbolSlotBase + 54,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 55) !== 0;
  }
}
export class ParameterPackSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      ParameterPackSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ParameterPackSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, ParameterPackSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, ParameterPackSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 51);
  }
  get elements(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ParameterPackSymbolSlotBase + 52,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      ParameterPackSymbolSlotBase + 53,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, ParameterPackSymbolSlotBase + 54) !== 0;
  }
}
export class TypeParameterSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      TypeParameterSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TypeParameterSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, TypeParameterSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, TypeParameterSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 51);
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      TypeParameterSymbolSlotBase + 52,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 53) !== 0;
  }
}
export class NonTypeParameterSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 9) !== 0
    );
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      NonTypeParameterSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      NonTypeParameterSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, NonTypeParameterSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(
      cxx.readSymbolVal(this.handle, NonTypeParameterSymbolSlotBase + 13),
    );
  }
  get isNodiscard(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 14) !== 0
    );
  }
  get isUsed(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 15) !== 0
    );
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 16) !== 0
    );
  }
  get isTrivialAbi(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 17) !== 0
    );
  }
  get hasDeducedReturnType(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 18) !== 0
    );
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 21) !== 0
    );
  }
  get isNamespaceAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 22) !== 0
    );
  }
  get isConcept(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 23) !== 0
    );
  }
  get isDeductionGuide(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 24) !== 0
    );
  }
  get isClass(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 25) !== 0
    );
  }
  get isEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 26) !== 0
    );
  }
  get isScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 27) !== 0
    );
  }
  get isFunction(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 28) !== 0
    );
  }
  get isTypeAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 29) !== 0
    );
  }
  get isVariable(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 30) !== 0
    );
  }
  get isField(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 31) !== 0
    );
  }
  get isParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 32) !== 0
    );
  }
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 33) !== 0
    );
  }
  get isEnumerator(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 34) !== 0
    );
  }
  get isFunctionParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 35) !== 0
    );
  }
  get isTemplateParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 36) !== 0
    );
  }
  get isBlock(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 37) !== 0
    );
  }
  get isLambda(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 38) !== 0
    );
  }
  get isTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 39) !== 0
    );
  }
  get isNonTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 40) !== 0
    );
  }
  get isTemplateTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 41) !== 0
    );
  }
  get isConstraintTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 42) !== 0
    );
  }
  get isOverloadSet(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 43) !== 0
    );
  }
  get isBaseClass(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 44) !== 0
    );
  }
  get isInjectedClassName(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 45) !== 0
    );
  }
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 46) !== 0
    );
  }
  get isUsingDeclaration(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 47) !== 0
    );
  }
  get isClassOrNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 48) !== 0
    );
  }
  get isNamespaceName(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 49) !== 0
    );
  }
  get isEnumOrScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 50) !== 0
    );
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 51);
  }
  get index(): number {
    return cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 52);
  }
  get depth(): number {
    return cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 53);
  }
  get objectType(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 54),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      NonTypeParameterSymbolSlotBase + 55,
    ) as string;
  }
  get isType(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 56) !== 0
    );
  }
}
export class TemplateTypeParameterSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 9) !== 0
    );
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      TemplateTypeParameterSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TemplateTypeParameterSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, TemplateTypeParameterSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(
      cxx.readSymbolVal(this.handle, TemplateTypeParameterSymbolSlotBase + 13),
    );
  }
  get isNodiscard(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 14) !==
      0
    );
  }
  get isUsed(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 15) !==
      0
    );
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 16) !==
      0
    );
  }
  get isTrivialAbi(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 17) !==
      0
    );
  }
  get hasDeducedReturnType(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 18) !==
      0
    );
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 21) !==
      0
    );
  }
  get isNamespaceAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 22) !==
      0
    );
  }
  get isConcept(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 23) !==
      0
    );
  }
  get isDeductionGuide(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 24) !==
      0
    );
  }
  get isClass(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 25) !==
      0
    );
  }
  get isEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 26) !==
      0
    );
  }
  get isScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 27) !==
      0
    );
  }
  get isFunction(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 28) !==
      0
    );
  }
  get isTypeAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 29) !==
      0
    );
  }
  get isVariable(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 30) !==
      0
    );
  }
  get isField(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 31) !==
      0
    );
  }
  get isParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 32) !==
      0
    );
  }
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 33) !==
      0
    );
  }
  get isEnumerator(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 34) !==
      0
    );
  }
  get isFunctionParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 35) !==
      0
    );
  }
  get isTemplateParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 36) !==
      0
    );
  }
  get isBlock(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 37) !==
      0
    );
  }
  get isLambda(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 38) !==
      0
    );
  }
  get isTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 39) !==
      0
    );
  }
  get isNonTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 40) !==
      0
    );
  }
  get isTemplateTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 41) !==
      0
    );
  }
  get isConstraintTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 42) !==
      0
    );
  }
  get isOverloadSet(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 43) !==
      0
    );
  }
  get isBaseClass(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 44) !==
      0
    );
  }
  get isInjectedClassName(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 45) !==
      0
    );
  }
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 46) !==
      0
    );
  }
  get isUsingDeclaration(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 47) !==
      0
    );
  }
  get isClassOrNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 48) !==
      0
    );
  }
  get isNamespaceName(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 49) !==
      0
    );
  }
  get isEnumOrScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 50) !==
      0
    );
  }
  get internalId(): number {
    return cxx.readSymbol(
      this.handle,
      TemplateTypeParameterSymbolSlotBase + 51,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      TemplateTypeParameterSymbolSlotBase + 52,
    ) as string;
  }
  get isType(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 53) !==
      0
    );
  }
}
export class ConstraintTypeParameterSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 2,
    );
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return (
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 9) !==
      0
    );
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 12,
      ),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(
      cxx.readSymbolVal(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 13,
      ),
    );
  }
  get isNodiscard(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 14,
      ) !== 0
    );
  }
  get isUsed(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 15,
      ) !== 0
    );
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 16,
      ) !== 0
    );
  }
  get isTrivialAbi(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 17,
      ) !== 0
    );
  }
  get hasDeducedReturnType(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 18,
      ) !== 0
    );
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 21,
      ) !== 0
    );
  }
  get isNamespaceAlias(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 22,
      ) !== 0
    );
  }
  get isConcept(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 23,
      ) !== 0
    );
  }
  get isDeductionGuide(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 24,
      ) !== 0
    );
  }
  get isClass(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 25,
      ) !== 0
    );
  }
  get isEnum(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 26,
      ) !== 0
    );
  }
  get isScopedEnum(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 27,
      ) !== 0
    );
  }
  get isFunction(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 28,
      ) !== 0
    );
  }
  get isTypeAlias(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 29,
      ) !== 0
    );
  }
  get isVariable(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 30,
      ) !== 0
    );
  }
  get isField(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 31,
      ) !== 0
    );
  }
  get isParameter(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 32,
      ) !== 0
    );
  }
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 33,
      ) !== 0
    );
  }
  get isEnumerator(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 34,
      ) !== 0
    );
  }
  get isFunctionParameters(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 35,
      ) !== 0
    );
  }
  get isTemplateParameters(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 36,
      ) !== 0
    );
  }
  get isBlock(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 37,
      ) !== 0
    );
  }
  get isLambda(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 38,
      ) !== 0
    );
  }
  get isTypeParameter(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 39,
      ) !== 0
    );
  }
  get isNonTypeParameter(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 40,
      ) !== 0
    );
  }
  get isTemplateTypeParameter(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 41,
      ) !== 0
    );
  }
  get isConstraintTypeParameter(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 42,
      ) !== 0
    );
  }
  get isOverloadSet(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 43,
      ) !== 0
    );
  }
  get isBaseClass(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 44,
      ) !== 0
    );
  }
  get isInjectedClassName(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 45,
      ) !== 0
    );
  }
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 46,
      ) !== 0
    );
  }
  get isUsingDeclaration(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 47,
      ) !== 0
    );
  }
  get isClassOrNamespace(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 48,
      ) !== 0
    );
  }
  get isNamespaceName(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 49,
      ) !== 0
    );
  }
  get isEnumOrScopedEnum(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 50,
      ) !== 0
    );
  }
  get internalId(): number {
    return cxx.readSymbol(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 51,
    );
  }
  get index(): number {
    return cxx.readSymbol(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 52,
    );
  }
  get depth(): number {
    return cxx.readSymbol(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 53,
    );
  }
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 54),
      this.modelOwner,
    );
  }
  get constraintExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 55),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 56,
    ) as string;
  }
  get isType(): boolean {
    return (
      cxx.readSymbol(
        this.handle,
        ConstraintTypeParameterSymbolSlotBase + 57,
      ) !== 0
    );
  }
}
export class EnumeratorSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      EnumeratorSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      EnumeratorSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, EnumeratorSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, EnumeratorSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 51);
  }
  get value():
    | { readonly index: 0; readonly value: bigint }
    | { readonly index: 1; readonly value: StringLiteral | undefined }
    | { readonly index: 2; readonly value: number }
    | { readonly index: 3; readonly value: number }
    | { readonly index: 4; readonly value: number }
    | { readonly index: 5; readonly value: Meta | undefined }
    | { readonly index: 6; readonly value: InitializerList | undefined }
    | { readonly index: 7; readonly value: ConstObject | undefined }
    | { readonly index: 8; readonly value: ConstAddress | undefined }
    | { readonly index: 9; readonly value: ConstLabelAddress | undefined }
    | { readonly index: 10; readonly value: ConstComplex | undefined }
    | { readonly index: 11; readonly value: {} }
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : ((item: any) =>
            item.index === 1
              ? {
                  index: 1,
                  value: objOf(item.value, this.modelOwner, StringLiteral),
                }
              : item.index === 5
                ? { index: 5, value: objOf(item.value, this.modelOwner, Meta) }
                : item.index === 6
                  ? {
                      index: 6,
                      value: objOf(
                        item.value,
                        this.modelOwner,
                        InitializerList,
                      ),
                    }
                  : item.index === 7
                    ? {
                        index: 7,
                        value: objOf(item.value, this.modelOwner, ConstObject),
                      }
                    : item.index === 8
                      ? {
                          index: 8,
                          value: objOf(
                            item.value,
                            this.modelOwner,
                            ConstAddress,
                          ),
                        }
                      : item.index === 9
                        ? {
                            index: 9,
                            value: objOf(
                              item.value,
                              this.modelOwner,
                              ConstLabelAddress,
                            ),
                          }
                        : item.index === 10
                          ? {
                              index: 10,
                              value: objOf(
                                item.value,
                                this.modelOwner,
                                ConstComplex,
                              ),
                            }
                          : item)(item))(
      cxx.readSymbolVal(this.handle, EnumeratorSymbolSlotBase + 52),
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      EnumeratorSymbolSlotBase + 53,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, EnumeratorSymbolSlotBase + 54) !== 0;
  }
}
export class NamespaceAliasSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 9) !== 0;
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      NamespaceAliasSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      NamespaceAliasSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, NamespaceAliasSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(cxx.readSymbolVal(this.handle, NamespaceAliasSymbolSlotBase + 13));
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 14) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 15) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 16) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 17) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 18) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 21) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 22) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 23) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 24) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 25) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 26) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 27) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 28) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 29) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 30) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 31) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 32) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 33) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 34) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 35) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 36) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 37) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 38) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 39) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 40) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 41) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 42) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 43) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 44) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 45) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 46) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 47) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 48) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 49) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 50) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 51);
  }
  get namespaceSymbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      NamespaceAliasSymbolSlotBase + 53,
    ) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 54) !== 0;
  }
}
export class UsingDeclarationSymbol extends Symbol {
  get name(): Name | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get location(): number {
    return cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 2);
  }
  get parent(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get enclosingNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get enclosingClass(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get enclosingFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
  get enclosingFunctionOrSelf(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 7),
      this.modelOwner,
    );
  }
  get next(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 8),
      this.modelOwner,
    );
  }
  get isHidden(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 9) !== 0
    );
  }
  get accessSpecifier(): AccessSpecifier {
    return cxx.readSymbol(
      this.handle,
      UsingDeclarationSymbolSlotBase + 10,
    ) as AccessSpecifier;
  }
  get abiTags(): Iterable<Identifier | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      UsingDeclarationSymbolSlotBase + 11,
      (item: any) => nameOf(item, this.modelOwner),
    );
  }
  get abiTagList(): ReadonlyArray<Identifier | undefined> | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) => nameOf(item, this.modelOwner)))(
      cxx.readSymbolVal(this.handle, UsingDeclarationSymbolSlotBase + 12),
    );
  }
  get attributes():
    | ReadonlyArray<{
        readonly attributeNamespace: Identifier | undefined;
        readonly name: Identifier | undefined;
        readonly arguments: ReadonlyArray<Identifier | undefined>;
      }>
    | undefined {
    return ((item: any) =>
      item === undefined
        ? undefined
        : (item as any[]).map((item: any) =>
            ((item: any) => ({
              attributeNamespace: nameOf(
                item.attributeNamespace,
                this.modelOwner,
              ),
              name: nameOf(item.name, this.modelOwner),
              arguments: (item.arguments as any[]).map((item: any) =>
                nameOf(item, this.modelOwner),
              ),
            }))(item),
          ))(
      cxx.readSymbolVal(this.handle, UsingDeclarationSymbolSlotBase + 13),
    );
  }
  get isNodiscard(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 14) !== 0
    );
  }
  get isUsed(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 15) !== 0
    );
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 16) !== 0
    );
  }
  get isTrivialAbi(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 17) !== 0
    );
  }
  get hasDeducedReturnType(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 18) !== 0
    );
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 20),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 21) !== 0
    );
  }
  get isNamespaceAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 22) !== 0
    );
  }
  get isConcept(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 23) !== 0
    );
  }
  get isDeductionGuide(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 24) !== 0
    );
  }
  get isClass(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 25) !== 0
    );
  }
  get isEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 26) !== 0
    );
  }
  get isScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 27) !== 0
    );
  }
  get isFunction(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 28) !== 0
    );
  }
  get isTypeAlias(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 29) !== 0
    );
  }
  get isVariable(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 30) !== 0
    );
  }
  get isField(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 31) !== 0
    );
  }
  get isParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 32) !== 0
    );
  }
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 33) !== 0
    );
  }
  get isEnumerator(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 34) !== 0
    );
  }
  get isFunctionParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 35) !== 0
    );
  }
  get isTemplateParameters(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 36) !== 0
    );
  }
  get isBlock(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 37) !== 0
    );
  }
  get isLambda(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 38) !== 0
    );
  }
  get isTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 39) !== 0
    );
  }
  get isNonTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 40) !== 0
    );
  }
  get isTemplateTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 41) !== 0
    );
  }
  get isConstraintTypeParameter(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 42) !== 0
    );
  }
  get isOverloadSet(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 43) !== 0
    );
  }
  get isBaseClass(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 44) !== 0
    );
  }
  get isInjectedClassName(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 45) !== 0
    );
  }
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 46) !== 0
    );
  }
  get isUsingDeclaration(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 47) !== 0
    );
  }
  get isClassOrNamespace(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 48) !== 0
    );
  }
  get isNamespaceName(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 49) !== 0
    );
  }
  get isEnumOrScopedEnum(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 50) !== 0
    );
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 51);
  }
  get declarator(): UsingDeclaratorAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 52),
      this.modelOwner,
    );
  }
  get target(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 53),
      this.modelOwner,
    );
  }
  get introducedFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      UsingDeclarationSymbolSlotBase + 54,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get text(): string {
    return cxx.readSymbolString(
      this.handle,
      UsingDeclarationSymbolSlotBase + 55,
    ) as string;
  }
  get isType(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 56) !== 0
    );
  }
}
export abstract class Type extends ModelObject {
  readonly kind: TypeKind;
  constructor(handle: number, owner: ModelOwner, kind: TypeKind) {
    super(handle, owner);
    this.kind = kind;
  }
  get text(): string {
    return cxx.readTypeString(this.handle, TypeSlotBase + 0) as string;
  }
}
export class BuiltinVaListType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      BuiltinVaListTypeSlotBase + 0,
    ) as string;
  }
}
export class BuiltinMetaInfoType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      BuiltinMetaInfoTypeSlotBase + 0,
    ) as string;
  }
}
export class VoidType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, VoidTypeSlotBase + 0) as string;
  }
}
export class NullptrType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, NullptrTypeSlotBase + 0) as string;
  }
}
export class DecltypeAutoType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      DecltypeAutoTypeSlotBase + 0,
    ) as string;
  }
}
export class AutoType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, AutoTypeSlotBase + 0) as string;
  }
}
export class BoolType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, BoolTypeSlotBase + 0) as string;
  }
}
export class SignedCharType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      SignedCharTypeSlotBase + 0,
    ) as string;
  }
}
export class ShortIntType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, ShortIntTypeSlotBase + 0) as string;
  }
}
export class IntType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, IntTypeSlotBase + 0) as string;
  }
}
export class LongIntType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, LongIntTypeSlotBase + 0) as string;
  }
}
export class LongLongIntType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      LongLongIntTypeSlotBase + 0,
    ) as string;
  }
}
export class Int128Type extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, Int128TypeSlotBase + 0) as string;
  }
}
export class UnsignedCharType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnsignedCharTypeSlotBase + 0,
    ) as string;
  }
}
export class UnsignedShortIntType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnsignedShortIntTypeSlotBase + 0,
    ) as string;
  }
}
export class UnsignedIntType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnsignedIntTypeSlotBase + 0,
    ) as string;
  }
}
export class UnsignedLongIntType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnsignedLongIntTypeSlotBase + 0,
    ) as string;
  }
}
export class UnsignedLongLongIntType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnsignedLongLongIntTypeSlotBase + 0,
    ) as string;
  }
}
export class UnsignedInt128Type extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnsignedInt128TypeSlotBase + 0,
    ) as string;
  }
}
export class CharType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, CharTypeSlotBase + 0) as string;
  }
}
export class Char8Type extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, Char8TypeSlotBase + 0) as string;
  }
}
export class Char16Type extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, Char16TypeSlotBase + 0) as string;
  }
}
export class Char32Type extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, Char32TypeSlotBase + 0) as string;
  }
}
export class WideCharType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, WideCharTypeSlotBase + 0) as string;
  }
}
export class FloatType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, FloatTypeSlotBase + 0) as string;
  }
}
export class DoubleType extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, DoubleTypeSlotBase + 0) as string;
  }
}
export class LongDoubleType extends Type {
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      LongDoubleTypeSlotBase + 0,
    ) as string;
  }
}
export class Float16Type extends Type {
  get text(): string {
    return cxx.readTypeString(this.handle, Float16TypeSlotBase + 0) as string;
  }
}
export class QualType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, QualTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get cvQualifiers(): CvQualifiers {
    return cxx.readType(this.handle, QualTypeSlotBase + 1) as CvQualifiers;
  }
  get isConst(): boolean {
    return cxx.readType(this.handle, QualTypeSlotBase + 2) !== 0;
  }
  get isVolatile(): boolean {
    return cxx.readType(this.handle, QualTypeSlotBase + 3) !== 0;
  }
  get text(): string {
    return cxx.readTypeString(this.handle, QualTypeSlotBase + 4) as string;
  }
}
export class BoundedArrayType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, BoundedArrayTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get size(): number {
    return cxx.readType(this.handle, BoundedArrayTypeSlotBase + 1);
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      BoundedArrayTypeSlotBase + 2,
    ) as string;
  }
}
export class UnboundedArrayType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, UnboundedArrayTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnboundedArrayTypeSlotBase + 1,
    ) as string;
  }
}
export class PointerType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, PointerTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(this.handle, PointerTypeSlotBase + 1) as string;
  }
}
export class LvalueReferenceType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, LvalueReferenceTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      LvalueReferenceTypeSlotBase + 1,
    ) as string;
  }
}
export class RvalueReferenceType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, RvalueReferenceTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      RvalueReferenceTypeSlotBase + 1,
    ) as string;
  }
}
export class OverloadSetType extends Type {
  get symbol(): OverloadSetSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, OverloadSetTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      OverloadSetTypeSlotBase + 1,
    ) as string;
  }
}
export class FunctionType extends Type {
  get returnType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, FunctionTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get parameterTypes(): Iterable<Type | undefined> {
    return typeItems(
      this.modelOwner,
      this.handle,
      FunctionTypeSlotBase + 1,
      (item: any) => typeOf(item, this.modelOwner),
    );
  }
  get isVariadic(): boolean {
    return cxx.readType(this.handle, FunctionTypeSlotBase + 2) !== 0;
  }
  get cvQualifiers(): CvQualifiers {
    return cxx.readType(this.handle, FunctionTypeSlotBase + 3) as CvQualifiers;
  }
  get refQualifier(): RefQualifier {
    return cxx.readType(this.handle, FunctionTypeSlotBase + 4) as RefQualifier;
  }
  get isNoexcept(): boolean {
    return cxx.readType(this.handle, FunctionTypeSlotBase + 5) !== 0;
  }
  get text(): string {
    return cxx.readTypeString(this.handle, FunctionTypeSlotBase + 6) as string;
  }
}
export class ClassType extends Type {
  get symbol(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, ClassTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get definition(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, ClassTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get isComplete(): boolean {
    return cxx.readType(this.handle, ClassTypeSlotBase + 2) !== 0;
  }
  get isUnion(): boolean {
    return cxx.readType(this.handle, ClassTypeSlotBase + 3) !== 0;
  }
  get text(): string {
    return cxx.readTypeString(this.handle, ClassTypeSlotBase + 4) as string;
  }
}
export class EnumType extends Type {
  get symbol(): EnumSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, EnumTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get underlyingType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, EnumTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(this.handle, EnumTypeSlotBase + 2) as string;
  }
}
export class ScopedEnumType extends Type {
  get symbol(): ScopedEnumSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, ScopedEnumTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get underlyingType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, ScopedEnumTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      ScopedEnumTypeSlotBase + 2,
    ) as string;
  }
}
export class MemberObjectPointerType extends Type {
  get classType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, MemberObjectPointerTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, MemberObjectPointerTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      MemberObjectPointerTypeSlotBase + 2,
    ) as string;
  }
}
export class MemberFunctionPointerType extends Type {
  get classType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, MemberFunctionPointerTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get functionType(): FunctionType | undefined {
    return typeOf(
      cxx.readType(this.handle, MemberFunctionPointerTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      MemberFunctionPointerTypeSlotBase + 2,
    ) as string;
  }
}
export class NamespaceType extends Type {
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, NamespaceTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(this.handle, NamespaceTypeSlotBase + 1) as string;
  }
}
export class TypeParameterType extends Type {
  get index(): number {
    return cxx.readType(this.handle, TypeParameterTypeSlotBase + 0);
  }
  get depth(): number {
    return cxx.readType(this.handle, TypeParameterTypeSlotBase + 1);
  }
  get isParameterPack(): boolean {
    return cxx.readType(this.handle, TypeParameterTypeSlotBase + 2) !== 0;
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      TypeParameterTypeSlotBase + 3,
    ) as string;
  }
}
export class TemplateTypeParameterType extends Type {
  get index(): number {
    return cxx.readType(this.handle, TemplateTypeParameterTypeSlotBase + 0);
  }
  get depth(): number {
    return cxx.readType(this.handle, TemplateTypeParameterTypeSlotBase + 1);
  }
  get isParameterPack(): boolean {
    return (
      cxx.readType(this.handle, TemplateTypeParameterTypeSlotBase + 2) !== 0
    );
  }
  get templateParameters(): Iterable<Type | undefined> {
    return typeItems(
      this.modelOwner,
      this.handle,
      TemplateTypeParameterTypeSlotBase + 3,
      (item: any) => typeOf(item, this.modelOwner),
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      TemplateTypeParameterTypeSlotBase + 4,
    ) as string;
  }
}
export class UnresolvedNameType extends Type {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedNameTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedNameTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get sourceLocationRange(): readonly [number, number] {
    return cxx.readTypeVal(
      this.handle,
      UnresolvedNameTypeSlotBase + 2,
    ) as readonly [number, number];
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnresolvedNameTypeSlotBase + 3,
    ) as string;
  }
}
export class UnresolvedBoundedArrayType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, UnresolvedBoundedArrayTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get size(): ExpressionAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedBoundedArrayTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnresolvedBoundedArrayTypeSlotBase + 2,
    ) as string;
  }
}
export class UnresolvedUnderlyingType extends Type {
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedUnderlyingTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnresolvedUnderlyingTypeSlotBase + 1,
    ) as string;
  }
}
export class UnresolvedBuiltinType extends Type {
  get builtinKind(): UnaryBuiltinTypeKind {
    return cxx.readType(
      this.handle,
      UnresolvedBuiltinTypeSlotBase + 0,
    ) as UnaryBuiltinTypeKind;
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedBuiltinTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnresolvedBuiltinTypeSlotBase + 2,
    ) as string;
  }
}
export class BitIntType extends Type {
  get numBits(): number {
    return cxx.readType(this.handle, BitIntTypeSlotBase + 0);
  }
  get text(): string {
    return cxx.readTypeString(this.handle, BitIntTypeSlotBase + 1) as string;
  }
}
export class UnsignedBitIntType extends Type {
  get numBits(): number {
    return cxx.readType(this.handle, UnsignedBitIntTypeSlotBase + 0);
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnsignedBitIntTypeSlotBase + 1,
    ) as string;
  }
}
export class UnresolvedBitIntType extends Type {
  get sizeExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedBitIntTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get isUnsigned(): boolean {
    return cxx.readType(this.handle, UnresolvedBitIntTypeSlotBase + 1) !== 0;
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnresolvedBitIntTypeSlotBase + 2,
    ) as string;
  }
}
export class VectorType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, VectorTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get elementCount(): number {
    return cxx.readType(this.handle, VectorTypeSlotBase + 1);
  }
  get vectorKind(): VectorKind {
    return cxx.readType(this.handle, VectorTypeSlotBase + 2) as VectorKind;
  }
  get text(): string {
    return cxx.readTypeString(this.handle, VectorTypeSlotBase + 3) as string;
  }
}
export class UnresolvedVectorType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, UnresolvedVectorTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get sizeExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedVectorTypeSlotBase + 1),
      this.modelOwner,
    );
  }
  get vectorKind(): VectorKind {
    return cxx.readType(
      this.handle,
      UnresolvedVectorTypeSlotBase + 2,
    ) as VectorKind;
  }
  get sizeKind(): VectorSizeKind {
    return cxx.readType(
      this.handle,
      UnresolvedVectorTypeSlotBase + 3,
    ) as VectorSizeKind;
  }
  get text(): string {
    return cxx.readTypeString(
      this.handle,
      UnresolvedVectorTypeSlotBase + 4,
    ) as string;
  }
}
export class ComplexType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, ComplexTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(this.handle, ComplexTypeSlotBase + 1) as string;
  }
}
export class AtomicType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, AtomicTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get text(): string {
    return cxx.readTypeString(this.handle, AtomicTypeSlotBase + 1) as string;
  }
}
export enum ASTKind {
  TranslationUnit = 0,
  ModuleUnit = 1,
  SimpleDeclaration = 2,
  AsmDeclaration = 3,
  NamespaceAliasDefinition = 4,
  UsingDeclaration = 5,
  UsingEnumDeclaration = 6,
  UsingDirective = 7,
  StaticAssertDeclaration = 8,
  AliasDeclaration = 9,
  OpaqueEnumDeclaration = 10,
  FunctionDefinition = 11,
  TemplateDeclaration = 12,
  ConceptDefinition = 13,
  DeductionGuide = 14,
  ExplicitInstantiation = 15,
  ExportDeclaration = 16,
  ExportCompoundDeclaration = 17,
  LinkageSpecification = 18,
  NamespaceDefinition = 19,
  EmptyDeclaration = 20,
  AttributeDeclaration = 21,
  ModuleImportDeclaration = 22,
  ParameterDeclaration = 23,
  AccessDeclaration = 24,
  ForRangeDeclaration = 25,
  StructuredBindingDeclaration = 26,
  AsmOperand = 27,
  AsmQualifier = 28,
  AsmClobber = 29,
  AsmGotoLabel = 30,
  Splicer = 31,
  GlobalModuleFragment = 32,
  PrivateModuleFragment = 33,
  ModuleDeclaration = 34,
  ModuleName = 35,
  ModuleQualifier = 36,
  ModulePartition = 37,
  ImportName = 38,
  InitDeclarator = 39,
  Declarator = 40,
  UsingDeclarator = 41,
  Enumerator = 42,
  TypeId = 43,
  Handler = 44,
  BaseSpecifier = 45,
  RequiresClause = 46,
  ParameterDeclarationClause = 47,
  TrailingReturnType = 48,
  LambdaSpecifier = 49,
  TypeConstraint = 50,
  AttributeArgumentClause = 51,
  Attribute = 52,
  AttributeUsingPrefix = 53,
  NewPlacement = 54,
  NestedNamespaceSpecifier = 55,
  LabeledStatement = 56,
  CaseStatement = 57,
  DefaultStatement = 58,
  ExpressionStatement = 59,
  CompoundStatement = 60,
  IfStatement = 61,
  ConstevalIfStatement = 62,
  SwitchStatement = 63,
  WhileStatement = 64,
  DoStatement = 65,
  ForRangeStatement = 66,
  ForStatement = 67,
  BreakStatement = 68,
  ContinueStatement = 69,
  ReturnStatement = 70,
  CoroutineReturnStatement = 71,
  GotoStatement = 72,
  DeclarationStatement = 73,
  TryBlockStatement = 74,
  CharLiteralExpression = 75,
  BoolLiteralExpression = 76,
  IntLiteralExpression = 77,
  FloatLiteralExpression = 78,
  NullptrLiteralExpression = 79,
  StringLiteralExpression = 80,
  UserDefinedStringLiteralExpression = 81,
  ObjectLiteralExpression = 82,
  ThisExpression = 83,
  PackIndexExpression = 84,
  GenericSelectionExpression = 85,
  NestedStatementExpression = 86,
  DefaultInitializerExpression = 87,
  NestedExpression = 88,
  IdExpression = 89,
  LambdaExpression = 90,
  FoldExpression = 91,
  RightFoldExpression = 92,
  LeftFoldExpression = 93,
  RequiresExpression = 94,
  VaArgExpression = 95,
  SubscriptExpression = 96,
  CallExpression = 97,
  TypeConstruction = 98,
  BracedTypeConstruction = 99,
  SpliceMemberExpression = 100,
  MemberExpression = 101,
  PostIncrExpression = 102,
  CppCastExpression = 103,
  BuiltinBitCastExpression = 104,
  BuiltinOffsetofExpression = 105,
  TypeidExpression = 106,
  TypeidOfTypeExpression = 107,
  SpliceExpression = 108,
  GlobalScopeReflectExpression = 109,
  NamespaceReflectExpression = 110,
  TypeIdReflectExpression = 111,
  ReflectExpression = 112,
  LabelAddressExpression = 113,
  UnaryExpression = 114,
  AwaitExpression = 115,
  SizeofExpression = 116,
  SizeofTypeExpression = 117,
  SizeofPackExpression = 118,
  AlignofTypeExpression = 119,
  AlignofExpression = 120,
  NoexceptExpression = 121,
  NewExpression = 122,
  DeleteExpression = 123,
  CastExpression = 124,
  ImplicitCastExpression = 125,
  ConstExpression = 126,
  BinaryExpression = 127,
  ConditionalExpression = 128,
  YieldExpression = 129,
  ThrowExpression = 130,
  AssignmentExpression = 131,
  TargetExpression = 132,
  RightExpression = 133,
  CompoundAssignmentExpression = 134,
  PackExpansionExpression = 135,
  DesignatedInitializerClause = 136,
  TypeTraitExpression = 137,
  ConditionExpression = 138,
  EqualInitializer = 139,
  BracedInitList = 140,
  ParenInitializer = 141,
  ThreeWayComparisonExpression = 142,
  DefaultGenericAssociation = 143,
  TypeGenericAssociation = 144,
  DotDesignator = 145,
  SubscriptDesignator = 146,
  TemplateTypeParameter = 147,
  NonTypeTemplateParameter = 148,
  TypenameTypeParameter = 149,
  ConstraintTypeParameter = 150,
  TypedefSpecifier = 151,
  FriendSpecifier = 152,
  ConstevalSpecifier = 153,
  ConstinitSpecifier = 154,
  ConstexprSpecifier = 155,
  InlineSpecifier = 156,
  NoreturnSpecifier = 157,
  StaticSpecifier = 158,
  ExternSpecifier = 159,
  RegisterSpecifier = 160,
  ThreadLocalSpecifier = 161,
  ThreadSpecifier = 162,
  MutableSpecifier = 163,
  VirtualSpecifier = 164,
  ExplicitSpecifier = 165,
  AutoTypeSpecifier = 166,
  VoidTypeSpecifier = 167,
  SizeTypeSpecifier = 168,
  SignTypeSpecifier = 169,
  BuiltinTypeSpecifier = 170,
  UnaryBuiltinTypeSpecifier = 171,
  BinaryBuiltinTypeSpecifier = 172,
  IntegralTypeSpecifier = 173,
  FloatingPointTypeSpecifier = 174,
  ComplexTypeSpecifier = 175,
  NamedTypeSpecifier = 176,
  AtomicTypeSpecifier = 177,
  BitIntTypeSpecifier = 178,
  UnderlyingTypeSpecifier = 179,
  ElaboratedTypeSpecifier = 180,
  DecltypeAutoSpecifier = 181,
  DecltypeSpecifier = 182,
  PlaceholderTypeSpecifier = 183,
  ConstQualifier = 184,
  VolatileQualifier = 185,
  AtomicQualifier = 186,
  RestrictQualifier = 187,
  EnumSpecifier = 188,
  ClassSpecifier = 189,
  TypenameSpecifier = 190,
  SplicerTypeSpecifier = 191,
  PointerOperator = 192,
  ReferenceOperator = 193,
  PtrToMemberOperator = 194,
  BitfieldDeclarator = 195,
  ParameterPack = 196,
  IdDeclarator = 197,
  NestedDeclarator = 198,
  FunctionDeclaratorChunk = 199,
  ArrayDeclaratorChunk = 200,
  NameId = 201,
  DestructorId = 202,
  DecltypeId = 203,
  OperatorFunctionId = 204,
  LiteralOperatorId = 205,
  ConversionFunctionId = 206,
  SimpleTemplateId = 207,
  LiteralOperatorTemplateId = 208,
  OperatorFunctionTemplateId = 209,
  GlobalNestedNameSpecifier = 210,
  SimpleNestedNameSpecifier = 211,
  DecltypeNestedNameSpecifier = 212,
  TemplateNestedNameSpecifier = 213,
  DefaultFunctionBody = 214,
  CompoundStatementFunctionBody = 215,
  TryStatementFunctionBody = 216,
  DeleteFunctionBody = 217,
  TypeTemplateArgument = 218,
  ExpressionTemplateArgument = 219,
  ThrowExceptionSpecifier = 220,
  NoexceptSpecifier = 221,
  SimpleRequirement = 222,
  CompoundRequirement = 223,
  TypeRequirement = 224,
  NestedRequirement = 225,
  NewParenInitializer = 226,
  NewBracedInitializer = 227,
  ParenMemInitializer = 228,
  BracedMemInitializer = 229,
  ThisLambdaCapture = 230,
  DerefThisLambdaCapture = 231,
  SimpleLambdaCapture = 232,
  RefLambdaCapture = 233,
  RefInitLambdaCapture = 234,
  InitLambdaCapture = 235,
  EllipsisExceptionDeclaration = 236,
  TypeExceptionDeclaration = 237,
  CxxAttribute = 238,
  GccAttribute = 239,
  AlignasAttribute = 240,
  AlignasTypeAttribute = 241,
  AsmAttribute = 242,
  ScopedAttributeToken = 243,
  SimpleAttributeToken = 244,
}
export enum SymbolKind {
  kNamespace = 0,
  kNamespaceAlias = 1,
  kConcept = 2,
  kDeductionGuide = 3,
  kClass = 4,
  kEnum = 5,
  kScopedEnum = 6,
  kFunction = 7,
  kTypeAlias = 8,
  kVariable = 9,
  kField = 10,
  kParameter = 11,
  kParameterPack = 12,
  kEnumerator = 13,
  kFunctionParameters = 14,
  kTemplateParameters = 15,
  kBlock = 16,
  kLambda = 17,
  kTypeParameter = 18,
  kNonTypeParameter = 19,
  kTemplateTypeParameter = 20,
  kConstraintTypeParameter = 21,
  kOverloadSet = 22,
  kBaseClass = 23,
  kInjectedClassName = 24,
  kUnresolved = 25,
  kUsingDeclaration = 26,
}
export enum TypeKind {
  kVoid = 0,
  kNullptr = 1,
  kDecltypeAuto = 2,
  kAuto = 3,
  kBool = 4,
  kSignedChar = 5,
  kShortInt = 6,
  kInt = 7,
  kLongInt = 8,
  kLongLongInt = 9,
  kInt128 = 10,
  kUnsignedChar = 11,
  kUnsignedShortInt = 12,
  kUnsignedInt = 13,
  kUnsignedLongInt = 14,
  kUnsignedLongLongInt = 15,
  kUnsignedInt128 = 16,
  kChar = 17,
  kChar8 = 18,
  kChar16 = 19,
  kChar32 = 20,
  kWideChar = 21,
  kFloat = 22,
  kDouble = 23,
  kLongDouble = 24,
  kFloat16 = 25,
  kQual = 26,
  kBoundedArray = 27,
  kUnboundedArray = 28,
  kPointer = 29,
  kLvalueReference = 30,
  kRvalueReference = 31,
  kFunction = 32,
  kClass = 33,
  kEnum = 34,
  kScopedEnum = 35,
  kMemberObjectPointer = 36,
  kMemberFunctionPointer = 37,
  kNamespace = 38,
  kTypeParameter = 39,
  kTemplateTypeParameter = 40,
  kUnresolvedName = 41,
  kUnresolvedBoundedArray = 42,
  kUnresolvedUnderlying = 43,
  kUnresolvedBuiltin = 44,
  kOverloadSet = 45,
  kBuiltinVaList = 46,
  kBuiltinMetaInfo = 47,
  kBitInt = 48,
  kUnsignedBitInt = 49,
  kUnresolvedBitInt = 50,
  kVector = 51,
  kUnresolvedVector = 52,
  kComplex = 53,
  kAtomic = 54,
}
export enum NameKind {
  kIdentifier = 0,
  kOperatorId = 1,
  kDestructorId = 2,
  kLiteralOperatorId = 3,
  kConversionFunctionId = 4,
  kTemplateId = 5,
}
export enum ValueCategory {
  kNone = 0,
  kLValue = 1,
  kXValue = 2,
  kPrValue = 3,
}
export enum TokenKind {
  T_EOF_SYMBOL = 0,
  T_ERROR = 1,
  T_COMMENT = 2,
  T_IDENTIFIER = 3,
  T_CHARACTER_LITERAL = 4,
  T_FLOATING_POINT_LITERAL = 5,
  T_INTEGER_LITERAL = 6,
  T_STRING_LITERAL = 7,
  T_USER_DEFINED_STRING_LITERAL = 8,
  T_UTF16_STRING_LITERAL = 9,
  T_UTF32_STRING_LITERAL = 10,
  T_UTF8_STRING_LITERAL = 11,
  T_WIDE_STRING_LITERAL = 12,
  T_PP_INTERNAL_VARIABLE = 13,
  T_CODE_COMPLETION = 14,
  T_PRAGMA_PACK = 15,
  T_AMP_AMP = 16,
  T_AMP_EQUAL = 17,
  T_AMP = 18,
  T_BAR_BAR = 19,
  T_BAR_EQUAL = 20,
  T_BAR = 21,
  T_CARET_CARET = 22,
  T_CARET_EQUAL = 23,
  T_CARET = 24,
  T_COLON_COLON = 25,
  T_COLON = 26,
  T_COMMA = 27,
  T_DELETE_ARRAY = 28,
  T_DOT_DOT_DOT = 29,
  T_DOT_STAR = 30,
  T_DOT = 31,
  T_EQUAL_EQUAL = 32,
  T_EQUAL = 33,
  T_EXCLAIM_EQUAL = 34,
  T_EXCLAIM = 35,
  T_GREATER_EQUAL = 36,
  T_GREATER_GREATER_EQUAL = 37,
  T_GREATER_GREATER = 38,
  T_GREATER = 39,
  T_HASH_HASH = 40,
  T_HASH = 41,
  T_LBRACE = 42,
  T_LBRACKET = 43,
  T_LESS_EQUAL_GREATER = 44,
  T_LESS_EQUAL = 45,
  T_LESS_LESS_EQUAL = 46,
  T_LESS_LESS = 47,
  T_LESS = 48,
  T_LPAREN = 49,
  T_MINUS_EQUAL = 50,
  T_MINUS_GREATER_STAR = 51,
  T_MINUS_GREATER = 52,
  T_MINUS_MINUS = 53,
  T_MINUS = 54,
  T_NEW_ARRAY = 55,
  T_PERCENT_EQUAL = 56,
  T_PERCENT = 57,
  T_PLUS_EQUAL = 58,
  T_PLUS_PLUS = 59,
  T_PLUS = 60,
  T_QUESTION = 61,
  T_RBRACE = 62,
  T_RBRACKET = 63,
  T_RPAREN = 64,
  T_SEMICOLON = 65,
  T_SLASH_EQUAL = 66,
  T_SLASH = 67,
  T_STAR_EQUAL = 68,
  T_STAR = 69,
  T_TILDE = 70,
  T__ATOMIC = 71,
  T__BITINT = 72,
  T__COMPLEX = 73,
  T__DECIMAL128 = 74,
  T__DECIMAL32 = 75,
  T__DECIMAL64 = 76,
  T__FLOAT16 = 77,
  T__GENERIC = 78,
  T__IMAGINARY = 79,
  T__NORETURN = 80,
  T___ATTRIBUTE__ = 81,
  T___BUILTIN_BIT_CAST = 82,
  T___BUILTIN_META_INFO = 83,
  T___BUILTIN_OFFSETOF = 84,
  T___BUILTIN_VA_ARG = 85,
  T___BUILTIN_VA_LIST = 86,
  T___COMPLEX__ = 87,
  T___EXTENSION__ = 88,
  T___FLOAT128 = 89,
  T___FLOAT80 = 90,
  T___IMAG__ = 91,
  T___INT128 = 92,
  T___INT128_T = 93,
  T___INT64 = 94,
  T___REAL__ = 95,
  T___RESTRICT__ = 96,
  T___THREAD = 97,
  T___UINT128_T = 98,
  T___UNDERLYING_TYPE = 99,
  T_ALIGNAS = 100,
  T_ALIGNOF = 101,
  T_ASM = 102,
  T_AUTO = 103,
  T_BOOL = 104,
  T_BREAK = 105,
  T_CASE = 106,
  T_CATCH = 107,
  T_CHAR = 108,
  T_CHAR16_T = 109,
  T_CHAR32_T = 110,
  T_CHAR8_T = 111,
  T_CLASS = 112,
  T_CO_AWAIT = 113,
  T_CO_RETURN = 114,
  T_CO_YIELD = 115,
  T_CONCEPT = 116,
  T_CONST = 117,
  T_CONST_CAST = 118,
  T_CONSTEVAL = 119,
  T_CONSTEXPR = 120,
  T_CONSTINIT = 121,
  T_CONTINUE = 122,
  T_DECLTYPE = 123,
  T_DEFAULT = 124,
  T_DELETE = 125,
  T_DO = 126,
  T_DOUBLE = 127,
  T_DYNAMIC_CAST = 128,
  T_ELSE = 129,
  T_ENUM = 130,
  T_EXPLICIT = 131,
  T_EXPORT = 132,
  T_EXTERN = 133,
  T_FALSE = 134,
  T_FLOAT = 135,
  T_FOR = 136,
  T_FRIEND = 137,
  T_GOTO = 138,
  T_IF = 139,
  T_IMPORT = 140,
  T_INLINE = 141,
  T_INT = 142,
  T_LONG = 143,
  T_MODULE = 144,
  T_MUTABLE = 145,
  T_NAMESPACE = 146,
  T_NEW = 147,
  T_NOEXCEPT = 148,
  T_NULLPTR = 149,
  T_OPERATOR = 150,
  T_PRIVATE = 151,
  T_PROTECTED = 152,
  T_PUBLIC = 153,
  T_REGISTER = 154,
  T_REINTERPRET_CAST = 155,
  T_REQUIRES = 156,
  T_RETURN = 157,
  T_SHORT = 158,
  T_SIGNED = 159,
  T_SIZEOF = 160,
  T_STATIC = 161,
  T_STATIC_ASSERT = 162,
  T_STATIC_CAST = 163,
  T_STRUCT = 164,
  T_SWITCH = 165,
  T_TEMPLATE = 166,
  T_THIS = 167,
  T_THREAD_LOCAL = 168,
  T_THROW = 169,
  T_TRUE = 170,
  T_TRY = 171,
  T_TYPEDEF = 172,
  T_TYPEID = 173,
  T_TYPENAME = 174,
  T_TYPEOF = 175,
  T_TYPEOF_UNQUAL = 176,
  T_UNION = 177,
  T_UNSIGNED = 178,
  T_USING = 179,
  T_VIRTUAL = 180,
  T_VOID = 181,
  T_VOLATILE = 182,
  T_WCHAR_T = 183,
  T_WHILE = 184,
  T_RESTRICT = 96,
  T___ALIGNOF__ = 101,
  T___ALIGNOF = 101,
  T___ASM__ = 102,
  T___ASM = 102,
  T___ATTRIBUTE = 81,
  T___DECLTYPE__ = 123,
  T___DECLTYPE = 123,
  T___INLINE__ = 141,
  T___INLINE = 141,
  T___RESTRICT = 96,
  T___TYPEOF__ = 175,
  T___TYPEOF = 175,
  T___VOLATILE__ = 182,
  T___VOLATILE = 182,
  T__ALIGNAS = 100,
  T__ALIGNOF = 101,
  T__ASM = 102,
  T__BOOL = 104,
  T__STATIC_ASSERT = 162,
  T__THREAD_LOCAL = 168,
  T_AND_EQ = 17,
  T_AND = 16,
  T_BITAND = 18,
  T_BITOR = 21,
  T_COMPL = 70,
  T_NOT_EQ = 34,
  T_NOT = 35,
  T_OR_EQ = 20,
  T_OR = 19,
  T_XOR_EQ = 23,
  T_XOR = 24,
}
export enum ImplicitCastKind {
  kIdentity = 0,
  kLValueToRValueConversion = 1,
  kArrayToPointerConversion = 2,
  kFunctionToPointerConversion = 3,
  kIntegralPromotion = 4,
  kFloatingPointPromotion = 5,
  kIntegralConversion = 6,
  kFloatingPointConversion = 7,
  kFloatingIntegralConversion = 8,
  kPointerConversion = 9,
  kPointerToMemberConversion = 10,
  kDerivedToBaseConversion = 11,
  kBaseToDerivedConversion = 12,
  kBooleanConversion = 13,
  kFunctionPointerConversion = 14,
  kQualificationConversion = 15,
  kVectorSplat = 16,
  kVectorConversion = 17,
  kAtomicToNonAtomic = 18,
  kNonAtomicToAtomic = 19,
  kRealToComplexConversion = 20,
  kComplexToRealConversion = 21,
  kComplexConversion = 22,
  kTemporaryMaterializationConversion = 23,
  kUserDefinedConversion = 24,
}
export enum BuiltinTypeTraitKind {
  T_NONE = 0,
  T___BUILTIN_TYPES_COMPATIBLE_P = 1,
  T___HAS_UNIQUE_OBJECT_REPRESENTATIONS = 2,
  T___HAS_VIRTUAL_DESTRUCTOR = 3,
  T___IS_ABSTRACT = 4,
  T___IS_AGGREGATE = 5,
  T___IS_ARITHMETIC = 6,
  T___IS_ARRAY = 7,
  T___IS_ASSIGNABLE = 8,
  T___IS_BASE_OF = 9,
  T___IS_BOUNDED_ARRAY = 10,
  T___IS_CLASS = 11,
  T___IS_COMPOUND = 12,
  T___IS_CONST = 13,
  T___IS_CONSTRUCTIBLE = 14,
  T___IS_CONVERTIBLE_TO = 15,
  T___IS_CONVERTIBLE = 16,
  T___IS_DESTRUCTIBLE = 17,
  T___IS_EMPTY = 18,
  T___IS_ENUM = 19,
  T___IS_FINAL = 20,
  T___IS_FLOATING_POINT = 21,
  T___IS_FUNCTION = 22,
  T___IS_FUNDAMENTAL = 23,
  T___IS_INTEGRAL = 24,
  T___IS_LAYOUT_COMPATIBLE = 25,
  T___IS_LITERAL_TYPE = 26,
  T___IS_LVALUE_REFERENCE = 27,
  T___IS_MEMBER_FUNCTION_POINTER = 28,
  T___IS_MEMBER_OBJECT_POINTER = 29,
  T___IS_MEMBER_POINTER = 30,
  T___IS_NOTHROW_ASSIGNABLE = 31,
  T___IS_NOTHROW_CONSTRUCTIBLE = 32,
  T___IS_NOTHROW_DESTRUCTIBLE = 33,
  T___IS_NULL_POINTER = 34,
  T___IS_OBJECT = 35,
  T___IS_POD = 36,
  T___IS_POINTER = 37,
  T___IS_POLYMORPHIC = 38,
  T___IS_REFERENCE = 39,
  T___IS_RVALUE_REFERENCE = 40,
  T___IS_SAME_AS = 41,
  T___IS_SAME = 42,
  T___IS_SCALAR = 43,
  T___IS_SCOPED_ENUM = 44,
  T___IS_SIGNED = 45,
  T___IS_STANDARD_LAYOUT = 46,
  T___IS_SWAPPABLE_WITH = 47,
  T___IS_TRIVIAL = 48,
  T___IS_TRIVIALLY_ASSIGNABLE = 49,
  T___IS_TRIVIALLY_CONSTRUCTIBLE = 50,
  T___IS_TRIVIALLY_COPYABLE = 51,
  T___IS_TRIVIALLY_DESTRUCTIBLE = 52,
  T___IS_UNBOUNDED_ARRAY = 53,
  T___IS_UNION = 54,
  T___IS_UNSIGNED = 55,
  T___IS_VOID = 56,
  T___IS_VOLATILE = 57,
  T___REFERENCE_CONSTRUCTS_FROM_TEMPORARY = 58,
  T___REFERENCE_CONVERTS_FROM_TEMPORARY = 59,
}
export enum UnaryBuiltinTypeKind {
  T_NONE = 0,
  T___ADD_LVALUE_REFERENCE = 1,
  T___ADD_POINTER = 2,
  T___ADD_RVALUE_REFERENCE = 3,
  T___DECAY = 4,
  T___MAKE_SIGNED = 5,
  T___MAKE_UNSIGNED = 6,
  T___REMOVE_ALL_EXTENTS = 7,
  T___REMOVE_CONST = 8,
  T___REMOVE_CV = 9,
  T___REMOVE_CVREF = 10,
  T___REMOVE_EXTENT = 11,
  T___REMOVE_POINTER = 12,
  T___REMOVE_REFERENCE_T = 13,
  T___REMOVE_RESTRICT = 14,
  T___REMOVE_VOLATILE = 15,
}
export enum BinaryBuiltinTypeKind {
  T_NONE = 0,
}
export enum IntegerLiteral_Radix {
  kDecimal = 0,
  kHexadecimal = 1,
  kOctal = 2,
  kBinary = 3,
}
export enum FloatLiteral_Components_FloatingPointSuffix {
  kNone = 0,
  kF = 1,
  kL = 2,
  kF16 = 3,
  kF32 = 4,
  kF64 = 5,
  kF128 = 6,
  kBF16 = 7,
}
export enum StringLiteralEncoding {
  kNone = 0,
  kWide = 1,
  kUtf8 = 2,
  kUtf16 = 3,
  kUtf32 = 4,
}
export enum BuiltinFunctionKind {
  T_NONE = 0,
  T___ATOMIC_ADD_FETCH = 1,
  T___ATOMIC_ALWAYS_LOCK_FREE = 2,
  T___ATOMIC_AND_FETCH = 3,
  T___ATOMIC_CLEAR = 4,
  T___ATOMIC_COMPARE_EXCHANGE = 5,
  T___ATOMIC_COMPARE_EXCHANGE_N = 6,
  T___ATOMIC_EXCHANGE = 7,
  T___ATOMIC_EXCHANGE_N = 8,
  T___ATOMIC_FETCH_ADD = 9,
  T___ATOMIC_FETCH_AND = 10,
  T___ATOMIC_FETCH_NAND = 11,
  T___ATOMIC_FETCH_OR = 12,
  T___ATOMIC_FETCH_SUB = 13,
  T___ATOMIC_FETCH_XOR = 14,
  T___ATOMIC_IS_LOCK_FREE = 15,
  T___ATOMIC_LOAD = 16,
  T___ATOMIC_LOAD_N = 17,
  T___ATOMIC_NAND_FETCH = 18,
  T___ATOMIC_OR_FETCH = 19,
  T___ATOMIC_SIGNAL_FENCE = 20,
  T___ATOMIC_STORE = 21,
  T___ATOMIC_STORE_N = 22,
  T___ATOMIC_SUB_FETCH = 23,
  T___ATOMIC_TEST_AND_SET = 24,
  T___ATOMIC_THREAD_FENCE = 25,
  T___ATOMIC_XOR_FETCH = 26,
  T___BUILTIN_COLUMN = 27,
  T___BUILTIN_FILE = 28,
  T___BUILTIN_FUNCTION = 29,
  T___BUILTIN_LINE = 30,
  T___BUILTIN__EXIT = 31,
  T___BUILTIN___COSPI = 32,
  T___BUILTIN___COSPIF = 33,
  T___BUILTIN___EXP10 = 34,
  T___BUILTIN___EXP10F = 35,
  T___BUILTIN___FINITE = 36,
  T___BUILTIN___FINITEF = 37,
  T___BUILTIN___FINITEL = 38,
  T___BUILTIN___SINPI = 39,
  T___BUILTIN___SINPIF = 40,
  T___BUILTIN___TANPI = 41,
  T___BUILTIN___TANPIF = 42,
  T___BUILTIN_ABORT = 43,
  T___BUILTIN_ABS = 44,
  T___BUILTIN_ACOS = 45,
  T___BUILTIN_ACOSF = 46,
  T___BUILTIN_ACOSH = 47,
  T___BUILTIN_ACOSHF = 48,
  T___BUILTIN_ACOSHL = 49,
  T___BUILTIN_ACOSL = 50,
  T___BUILTIN_ADD_OVERFLOW = 51,
  T___BUILTIN_ADDRESSOF = 52,
  T___BUILTIN_ALIGNED_ALLOC = 53,
  T___BUILTIN_ALLOCA = 54,
  T___BUILTIN_ASIN = 55,
  T___BUILTIN_ASINF = 56,
  T___BUILTIN_ASINH = 57,
  T___BUILTIN_ASINHF = 58,
  T___BUILTIN_ASINHL = 59,
  T___BUILTIN_ASINL = 60,
  T___BUILTIN_ASSUME_ALIGNED = 61,
  T___BUILTIN_ATAN = 62,
  T___BUILTIN_ATAN2 = 63,
  T___BUILTIN_ATAN2F = 64,
  T___BUILTIN_ATAN2L = 65,
  T___BUILTIN_ATANF = 66,
  T___BUILTIN_ATANH = 67,
  T___BUILTIN_ATANHF = 68,
  T___BUILTIN_ATANHL = 69,
  T___BUILTIN_ATANL = 70,
  T___BUILTIN_BCMP = 71,
  T___BUILTIN_BCOPY = 72,
  T___BUILTIN_BSWAP16 = 73,
  T___BUILTIN_BSWAP32 = 74,
  T___BUILTIN_BSWAP64 = 75,
  T___BUILTIN_BZERO = 76,
  T___BUILTIN_C23_VA_START = 77,
  T___BUILTIN_CABS = 78,
  T___BUILTIN_CABSF = 79,
  T___BUILTIN_CABSL = 80,
  T___BUILTIN_CACOS = 81,
  T___BUILTIN_CACOSF = 82,
  T___BUILTIN_CACOSH = 83,
  T___BUILTIN_CACOSHF = 84,
  T___BUILTIN_CACOSHL = 85,
  T___BUILTIN_CACOSL = 86,
  T___BUILTIN_CARG = 87,
  T___BUILTIN_CARGF = 88,
  T___BUILTIN_CARGL = 89,
  T___BUILTIN_CASIN = 90,
  T___BUILTIN_CASINF = 91,
  T___BUILTIN_CASINH = 92,
  T___BUILTIN_CASINHF = 93,
  T___BUILTIN_CASINHL = 94,
  T___BUILTIN_CASINL = 95,
  T___BUILTIN_CATAN = 96,
  T___BUILTIN_CATANF = 97,
  T___BUILTIN_CATANH = 98,
  T___BUILTIN_CATANHF = 99,
  T___BUILTIN_CATANHL = 100,
  T___BUILTIN_CATANL = 101,
  T___BUILTIN_CBRT = 102,
  T___BUILTIN_CBRTF = 103,
  T___BUILTIN_CBRTL = 104,
  T___BUILTIN_CCOS = 105,
  T___BUILTIN_CCOSF = 106,
  T___BUILTIN_CCOSH = 107,
  T___BUILTIN_CCOSHF = 108,
  T___BUILTIN_CCOSHL = 109,
  T___BUILTIN_CCOSL = 110,
  T___BUILTIN_CEIL = 111,
  T___BUILTIN_CEILF = 112,
  T___BUILTIN_CEILL = 113,
  T___BUILTIN_CEXP = 114,
  T___BUILTIN_CEXPF = 115,
  T___BUILTIN_CEXPL = 116,
  T___BUILTIN_CIMAG = 117,
  T___BUILTIN_CIMAGF = 118,
  T___BUILTIN_CIMAGL = 119,
  T___BUILTIN_CLOG = 120,
  T___BUILTIN_CLOGF = 121,
  T___BUILTIN_CLOGL = 122,
  T___BUILTIN_CLRSB = 123,
  T___BUILTIN_CLRSBL = 124,
  T___BUILTIN_CLRSBLL = 125,
  T___BUILTIN_CLZ = 126,
  T___BUILTIN_CLZL = 127,
  T___BUILTIN_CLZLL = 128,
  T___BUILTIN_CLZS = 129,
  T___BUILTIN_COMPLEX = 130,
  T___BUILTIN_CONJ = 131,
  T___BUILTIN_CONJF = 132,
  T___BUILTIN_CONJL = 133,
  T___BUILTIN_CONSTANT_P = 134,
  T___BUILTIN_COPYSIGN = 135,
  T___BUILTIN_COPYSIGNF = 136,
  T___BUILTIN_COPYSIGNL = 137,
  T___BUILTIN_CORO_DESTROY = 138,
  T___BUILTIN_CORO_DONE = 139,
  T___BUILTIN_CORO_PROMISE = 140,
  T___BUILTIN_CORO_RESUME = 141,
  T___BUILTIN_COS = 142,
  T___BUILTIN_COSF = 143,
  T___BUILTIN_COSH = 144,
  T___BUILTIN_COSHF = 145,
  T___BUILTIN_COSHL = 146,
  T___BUILTIN_COSL = 147,
  T___BUILTIN_CPOW = 148,
  T___BUILTIN_CPOWF = 149,
  T___BUILTIN_CPOWL = 150,
  T___BUILTIN_CPROJ = 151,
  T___BUILTIN_CPROJF = 152,
  T___BUILTIN_CPROJL = 153,
  T___BUILTIN_CREAL = 154,
  T___BUILTIN_CREALF = 155,
  T___BUILTIN_CREALL = 156,
  T___BUILTIN_CSIN = 157,
  T___BUILTIN_CSINF = 158,
  T___BUILTIN_CSINH = 159,
  T___BUILTIN_CSINHF = 160,
  T___BUILTIN_CSINHL = 161,
  T___BUILTIN_CSINL = 162,
  T___BUILTIN_CSQRT = 163,
  T___BUILTIN_CSQRTF = 164,
  T___BUILTIN_CSQRTL = 165,
  T___BUILTIN_CTAN = 166,
  T___BUILTIN_CTANF = 167,
  T___BUILTIN_CTANH = 168,
  T___BUILTIN_CTANHF = 169,
  T___BUILTIN_CTANHL = 170,
  T___BUILTIN_CTANL = 171,
  T___BUILTIN_CTZ = 172,
  T___BUILTIN_CTZL = 173,
  T___BUILTIN_CTZLL = 174,
  T___BUILTIN_CTZS = 175,
  T___BUILTIN_ERF = 176,
  T___BUILTIN_ERFC = 177,
  T___BUILTIN_ERFCF = 178,
  T___BUILTIN_ERFCL = 179,
  T___BUILTIN_ERFF = 180,
  T___BUILTIN_ERFL = 181,
  T___BUILTIN_EXIT = 182,
  T___BUILTIN_EXP = 183,
  T___BUILTIN_EXP2 = 184,
  T___BUILTIN_EXP2F = 185,
  T___BUILTIN_EXP2L = 186,
  T___BUILTIN_EXPECT = 187,
  T___BUILTIN_EXPF = 188,
  T___BUILTIN_EXPL = 189,
  T___BUILTIN_EXPM1 = 190,
  T___BUILTIN_EXPM1F = 191,
  T___BUILTIN_EXPM1L = 192,
  T___BUILTIN_FABS = 193,
  T___BUILTIN_FABSF = 194,
  T___BUILTIN_FABSL = 195,
  T___BUILTIN_FDIM = 196,
  T___BUILTIN_FDIMF = 197,
  T___BUILTIN_FDIML = 198,
  T___BUILTIN_FFS = 199,
  T___BUILTIN_FFSL = 200,
  T___BUILTIN_FFSLL = 201,
  T___BUILTIN_FINITE = 202,
  T___BUILTIN_FINITEF = 203,
  T___BUILTIN_FINITEL = 204,
  T___BUILTIN_FLOOR = 205,
  T___BUILTIN_FLOORF = 206,
  T___BUILTIN_FLOORL = 207,
  T___BUILTIN_FMA = 208,
  T___BUILTIN_FMAF = 209,
  T___BUILTIN_FMAL = 210,
  T___BUILTIN_FMAX = 211,
  T___BUILTIN_FMAXF = 212,
  T___BUILTIN_FMAXIMUM_NUM = 213,
  T___BUILTIN_FMAXIMUM_NUMF = 214,
  T___BUILTIN_FMAXIMUM_NUML = 215,
  T___BUILTIN_FMAXL = 216,
  T___BUILTIN_FMIN = 217,
  T___BUILTIN_FMINF = 218,
  T___BUILTIN_FMINIMUM_NUM = 219,
  T___BUILTIN_FMINIMUM_NUMF = 220,
  T___BUILTIN_FMINIMUM_NUML = 221,
  T___BUILTIN_FMINL = 222,
  T___BUILTIN_FMOD = 223,
  T___BUILTIN_FMODF = 224,
  T___BUILTIN_FMODL = 225,
  T___BUILTIN_FPCLASSIFY = 226,
  T___BUILTIN_FREXP = 227,
  T___BUILTIN_FREXPF = 228,
  T___BUILTIN_FREXPL = 229,
  T___BUILTIN_HUGE_VAL = 230,
  T___BUILTIN_HUGE_VALF = 231,
  T___BUILTIN_HUGE_VALL = 232,
  T___BUILTIN_HYPOT = 233,
  T___BUILTIN_HYPOTF = 234,
  T___BUILTIN_HYPOTL = 235,
  T___BUILTIN_ILOGB = 236,
  T___BUILTIN_ILOGBF = 237,
  T___BUILTIN_ILOGBL = 238,
  T___BUILTIN_INDEX = 239,
  T___BUILTIN_INF = 240,
  T___BUILTIN_INFF = 241,
  T___BUILTIN_INFL = 242,
  T___BUILTIN_INVOKE = 243,
  T___BUILTIN_IS_CONSTANT_EVALUATED = 244,
  T___BUILTIN_ISALNUM = 245,
  T___BUILTIN_ISALPHA = 246,
  T___BUILTIN_ISBLANK = 247,
  T___BUILTIN_ISCNTRL = 248,
  T___BUILTIN_ISDIGIT = 249,
  T___BUILTIN_ISFINITE = 250,
  T___BUILTIN_ISGRAPH = 251,
  T___BUILTIN_ISGREATER = 252,
  T___BUILTIN_ISGREATEREQUAL = 253,
  T___BUILTIN_ISINF = 254,
  T___BUILTIN_ISLESS = 255,
  T___BUILTIN_ISLESSEQUAL = 256,
  T___BUILTIN_ISLESSGREATER = 257,
  T___BUILTIN_ISLOWER = 258,
  T___BUILTIN_ISNAN = 259,
  T___BUILTIN_ISNORMAL = 260,
  T___BUILTIN_ISPRINT = 261,
  T___BUILTIN_ISPUNCT = 262,
  T___BUILTIN_ISSPACE = 263,
  T___BUILTIN_ISUNORDERED = 264,
  T___BUILTIN_ISUPPER = 265,
  T___BUILTIN_ISXDIGIT = 266,
  T___BUILTIN_LABS = 267,
  T___BUILTIN_LDEXP = 268,
  T___BUILTIN_LDEXPF = 269,
  T___BUILTIN_LDEXPL = 270,
  T___BUILTIN_LGAMMA = 271,
  T___BUILTIN_LGAMMAF = 272,
  T___BUILTIN_LGAMMAL = 273,
  T___BUILTIN_LLABS = 274,
  T___BUILTIN_LLRINT = 275,
  T___BUILTIN_LLRINTF = 276,
  T___BUILTIN_LLRINTL = 277,
  T___BUILTIN_LLROUND = 278,
  T___BUILTIN_LLROUNDF = 279,
  T___BUILTIN_LLROUNDL = 280,
  T___BUILTIN_LOG = 281,
  T___BUILTIN_LOG10 = 282,
  T___BUILTIN_LOG10F = 283,
  T___BUILTIN_LOG10L = 284,
  T___BUILTIN_LOG1P = 285,
  T___BUILTIN_LOG1PF = 286,
  T___BUILTIN_LOG1PL = 287,
  T___BUILTIN_LOG2 = 288,
  T___BUILTIN_LOG2F = 289,
  T___BUILTIN_LOG2L = 290,
  T___BUILTIN_LOGB = 291,
  T___BUILTIN_LOGBF = 292,
  T___BUILTIN_LOGBL = 293,
  T___BUILTIN_LOGF = 294,
  T___BUILTIN_LOGL = 295,
  T___BUILTIN_LRINT = 296,
  T___BUILTIN_LRINTF = 297,
  T___BUILTIN_LRINTL = 298,
  T___BUILTIN_LROUND = 299,
  T___BUILTIN_LROUNDF = 300,
  T___BUILTIN_LROUNDL = 301,
  T___BUILTIN_MEMCCPY = 302,
  T___BUILTIN_MEMCHR = 303,
  T___BUILTIN_MEMCMP = 304,
  T___BUILTIN_MEMCPY = 305,
  T___BUILTIN_MEMMOVE = 306,
  T___BUILTIN_MEMPCPY = 307,
  T___BUILTIN_MEMSET = 308,
  T___BUILTIN_MODF = 309,
  T___BUILTIN_MODFF = 310,
  T___BUILTIN_MODFL = 311,
  T___BUILTIN_MUL_OVERFLOW = 312,
  T___BUILTIN_NAN = 313,
  T___BUILTIN_NANF = 314,
  T___BUILTIN_NANL = 315,
  T___BUILTIN_NANS = 316,
  T___BUILTIN_NANSF = 317,
  T___BUILTIN_NANSL = 318,
  T___BUILTIN_NEARBYINT = 319,
  T___BUILTIN_NEARBYINTF = 320,
  T___BUILTIN_NEARBYINTL = 321,
  T___BUILTIN_NEXTAFTER = 322,
  T___BUILTIN_NEXTAFTERF = 323,
  T___BUILTIN_NEXTAFTERL = 324,
  T___BUILTIN_NEXTTOWARD = 325,
  T___BUILTIN_NEXTTOWARDF = 326,
  T___BUILTIN_NEXTTOWARDL = 327,
  T___BUILTIN_OPERATOR_DELETE = 328,
  T___BUILTIN_OPERATOR_NEW = 329,
  T___BUILTIN_PARITY = 330,
  T___BUILTIN_PARITYL = 331,
  T___BUILTIN_PARITYLL = 332,
  T___BUILTIN_POPCOUNT = 333,
  T___BUILTIN_POPCOUNTL = 334,
  T___BUILTIN_POPCOUNTLL = 335,
  T___BUILTIN_POW = 336,
  T___BUILTIN_POWF = 337,
  T___BUILTIN_POWL = 338,
  T___BUILTIN_REMAINDER = 339,
  T___BUILTIN_REMAINDERF = 340,
  T___BUILTIN_REMAINDERL = 341,
  T___BUILTIN_REMQUO = 342,
  T___BUILTIN_REMQUOF = 343,
  T___BUILTIN_REMQUOL = 344,
  T___BUILTIN_RINDEX = 345,
  T___BUILTIN_RINT = 346,
  T___BUILTIN_RINTF = 347,
  T___BUILTIN_RINTL = 348,
  T___BUILTIN_ROUND = 349,
  T___BUILTIN_ROUNDEVEN = 350,
  T___BUILTIN_ROUNDEVENF = 351,
  T___BUILTIN_ROUNDEVENL = 352,
  T___BUILTIN_ROUNDF = 353,
  T___BUILTIN_ROUNDL = 354,
  T___BUILTIN_SCALBLN = 355,
  T___BUILTIN_SCALBLNF = 356,
  T___BUILTIN_SCALBLNL = 357,
  T___BUILTIN_SCALBN = 358,
  T___BUILTIN_SCALBNF = 359,
  T___BUILTIN_SCALBNL = 360,
  T___BUILTIN_SIGNBIT = 361,
  T___BUILTIN_SIN = 362,
  T___BUILTIN_SINCOS = 363,
  T___BUILTIN_SINCOSF = 364,
  T___BUILTIN_SINCOSL = 365,
  T___BUILTIN_SINF = 366,
  T___BUILTIN_SINH = 367,
  T___BUILTIN_SINHF = 368,
  T___BUILTIN_SINHL = 369,
  T___BUILTIN_SINL = 370,
  T___BUILTIN_SOURCE_LOCATION = 371,
  T___BUILTIN_SQRT = 372,
  T___BUILTIN_SQRTF = 373,
  T___BUILTIN_SQRTL = 374,
  T___BUILTIN_STPCPY = 375,
  T___BUILTIN_STPNCPY = 376,
  T___BUILTIN_STRCASECMP = 377,
  T___BUILTIN_STRCAT = 378,
  T___BUILTIN_STRCHR = 379,
  T___BUILTIN_STRCMP = 380,
  T___BUILTIN_STRCPY = 381,
  T___BUILTIN_STRCSPN = 382,
  T___BUILTIN_STRDUP = 383,
  T___BUILTIN_STRERROR = 384,
  T___BUILTIN_STRLCAT = 385,
  T___BUILTIN_STRLCPY = 386,
  T___BUILTIN_STRLEN = 387,
  T___BUILTIN_STRNCASECMP = 388,
  T___BUILTIN_STRNCAT = 389,
  T___BUILTIN_STRNCMP = 390,
  T___BUILTIN_STRNCPY = 391,
  T___BUILTIN_STRNDUP = 392,
  T___BUILTIN_STRPBRK = 393,
  T___BUILTIN_STRRCHR = 394,
  T___BUILTIN_STRSPN = 395,
  T___BUILTIN_STRSTR = 396,
  T___BUILTIN_STRTOD = 397,
  T___BUILTIN_STRTOF = 398,
  T___BUILTIN_STRTOK = 399,
  T___BUILTIN_STRTOL = 400,
  T___BUILTIN_STRTOLD = 401,
  T___BUILTIN_STRTOLL = 402,
  T___BUILTIN_STRTOUL = 403,
  T___BUILTIN_STRTOULL = 404,
  T___BUILTIN_STRXFRM = 405,
  T___BUILTIN_SUB_OVERFLOW = 406,
  T___BUILTIN_TAN = 407,
  T___BUILTIN_TANF = 408,
  T___BUILTIN_TANH = 409,
  T___BUILTIN_TANHF = 410,
  T___BUILTIN_TANHL = 411,
  T___BUILTIN_TANL = 412,
  T___BUILTIN_TGAMMA = 413,
  T___BUILTIN_TGAMMAF = 414,
  T___BUILTIN_TGAMMAL = 415,
  T___BUILTIN_TOLOWER = 416,
  T___BUILTIN_TOUPPER = 417,
  T___BUILTIN_TRAP = 418,
  T___BUILTIN_TRUNC = 419,
  T___BUILTIN_TRUNCF = 420,
  T___BUILTIN_TRUNCL = 421,
  T___BUILTIN_UNREACHABLE = 422,
  T___BUILTIN_VA_COPY = 423,
  T___BUILTIN_VA_END = 424,
  T___BUILTIN_VA_START = 425,
  T___BUILTIN_VSNPRINTF = 426,
  T___BUILTIN_WCSCHR = 427,
  T___BUILTIN_WCSCMP = 428,
  T___BUILTIN_WCSLEN = 429,
  T___BUILTIN_WCSNCMP = 430,
  T___BUILTIN_WMEMCHR = 431,
  T___BUILTIN_WMEMCMP = 432,
  T___BUILTIN_WMEMCPY = 433,
  T___BUILTIN_WMEMMOVE = 434,
  T___C11_ATOMIC_COMPARE_EXCHANGE_STRONG = 435,
  T___C11_ATOMIC_COMPARE_EXCHANGE_WEAK = 436,
  T___C11_ATOMIC_EXCHANGE = 437,
  T___C11_ATOMIC_FETCH_ADD = 438,
  T___C11_ATOMIC_FETCH_AND = 439,
  T___C11_ATOMIC_FETCH_NAND = 440,
  T___C11_ATOMIC_FETCH_OR = 441,
  T___C11_ATOMIC_FETCH_SUB = 442,
  T___C11_ATOMIC_FETCH_XOR = 443,
  T___C11_ATOMIC_INIT = 444,
  T___C11_ATOMIC_IS_LOCK_FREE = 445,
  T___C11_ATOMIC_LOAD = 446,
  T___C11_ATOMIC_SIGNAL_FENCE = 447,
  T___C11_ATOMIC_STORE = 448,
  T___C11_ATOMIC_THREAD_FENCE = 449,
}
export enum BuiltinTemplateKind {
  T_NONE = 0,
  T___MAKE_INTEGER_SEQ = 1,
  T___TYPE_PACK_ELEMENT = 2,
  T___BUILTIN_COMMON_TYPE = 3,
}
export enum WellKnownName {
  T_NONE = 0,
  T_STD = 1,
  T_ALIGN_VAL_T = 2,
  T_DESTROYING_DELETE_T = 3,
  T_INITIALIZER_LIST = 4,
}
export enum AccessSpecifier {
  kPublic = 0,
  kProtected = 1,
  kPrivate = 2,
}
export enum Severity {
  Message = 0,
  Note = 1,
  Warning = 2,
  Error = 3,
  Fatal = 4,
}
export enum VTableLayout_SlotKind {
  kFunction = 0,
  kCompleteDtor = 1,
  kDeletingDtor = 2,
}
export enum LanguageKind {
  kC = 0,
  kCXX = 1,
}
export enum PendingExceptionSpecificationState {
  kUnresolved = 0,
  kResolving = 1,
  kResolved = 2,
}
export enum CvQualifiers {
  kNone = 0,
  kConst = 1,
  kVolatile = 2,
  kConstVolatile = 3,
}
export enum RefQualifier {
  kNone = 0,
  kLvalue = 1,
  kRvalue = 2,
}
export enum VectorKind {
  kGnu = 0,
  kExt = 1,
}
export enum VectorSizeKind {
  kBytes = 0,
  kElements = 1,
}
astConstructors[ASTKind.TranslationUnit] = TranslationUnitAST;
astConstructors[ASTKind.ModuleUnit] = ModuleUnitAST;
astConstructors[ASTKind.SimpleDeclaration] = SimpleDeclarationAST;
astConstructors[ASTKind.AsmDeclaration] = AsmDeclarationAST;
astConstructors[ASTKind.NamespaceAliasDefinition] = NamespaceAliasDefinitionAST;
astConstructors[ASTKind.UsingDeclaration] = UsingDeclarationAST;
astConstructors[ASTKind.UsingEnumDeclaration] = UsingEnumDeclarationAST;
astConstructors[ASTKind.UsingDirective] = UsingDirectiveAST;
astConstructors[ASTKind.StaticAssertDeclaration] = StaticAssertDeclarationAST;
astConstructors[ASTKind.AliasDeclaration] = AliasDeclarationAST;
astConstructors[ASTKind.OpaqueEnumDeclaration] = OpaqueEnumDeclarationAST;
astConstructors[ASTKind.FunctionDefinition] = FunctionDefinitionAST;
astConstructors[ASTKind.TemplateDeclaration] = TemplateDeclarationAST;
astConstructors[ASTKind.ConceptDefinition] = ConceptDefinitionAST;
astConstructors[ASTKind.DeductionGuide] = DeductionGuideAST;
astConstructors[ASTKind.ExplicitInstantiation] = ExplicitInstantiationAST;
astConstructors[ASTKind.ExportDeclaration] = ExportDeclarationAST;
astConstructors[ASTKind.ExportCompoundDeclaration] =
  ExportCompoundDeclarationAST;
astConstructors[ASTKind.LinkageSpecification] = LinkageSpecificationAST;
astConstructors[ASTKind.NamespaceDefinition] = NamespaceDefinitionAST;
astConstructors[ASTKind.EmptyDeclaration] = EmptyDeclarationAST;
astConstructors[ASTKind.AttributeDeclaration] = AttributeDeclarationAST;
astConstructors[ASTKind.ModuleImportDeclaration] = ModuleImportDeclarationAST;
astConstructors[ASTKind.ParameterDeclaration] = ParameterDeclarationAST;
astConstructors[ASTKind.AccessDeclaration] = AccessDeclarationAST;
astConstructors[ASTKind.ForRangeDeclaration] = ForRangeDeclarationAST;
astConstructors[ASTKind.StructuredBindingDeclaration] =
  StructuredBindingDeclarationAST;
astConstructors[ASTKind.AsmOperand] = AsmOperandAST;
astConstructors[ASTKind.AsmQualifier] = AsmQualifierAST;
astConstructors[ASTKind.AsmClobber] = AsmClobberAST;
astConstructors[ASTKind.AsmGotoLabel] = AsmGotoLabelAST;
astConstructors[ASTKind.Splicer] = SplicerAST;
astConstructors[ASTKind.GlobalModuleFragment] = GlobalModuleFragmentAST;
astConstructors[ASTKind.PrivateModuleFragment] = PrivateModuleFragmentAST;
astConstructors[ASTKind.ModuleDeclaration] = ModuleDeclarationAST;
astConstructors[ASTKind.ModuleName] = ModuleNameAST;
astConstructors[ASTKind.ModuleQualifier] = ModuleQualifierAST;
astConstructors[ASTKind.ModulePartition] = ModulePartitionAST;
astConstructors[ASTKind.ImportName] = ImportNameAST;
astConstructors[ASTKind.InitDeclarator] = InitDeclaratorAST;
astConstructors[ASTKind.Declarator] = DeclaratorAST;
astConstructors[ASTKind.UsingDeclarator] = UsingDeclaratorAST;
astConstructors[ASTKind.Enumerator] = EnumeratorAST;
astConstructors[ASTKind.TypeId] = TypeIdAST;
astConstructors[ASTKind.Handler] = HandlerAST;
astConstructors[ASTKind.BaseSpecifier] = BaseSpecifierAST;
astConstructors[ASTKind.RequiresClause] = RequiresClauseAST;
astConstructors[ASTKind.ParameterDeclarationClause] =
  ParameterDeclarationClauseAST;
astConstructors[ASTKind.TrailingReturnType] = TrailingReturnTypeAST;
astConstructors[ASTKind.LambdaSpecifier] = LambdaSpecifierAST;
astConstructors[ASTKind.TypeConstraint] = TypeConstraintAST;
astConstructors[ASTKind.AttributeArgumentClause] = AttributeArgumentClauseAST;
astConstructors[ASTKind.Attribute] = AttributeAST;
astConstructors[ASTKind.AttributeUsingPrefix] = AttributeUsingPrefixAST;
astConstructors[ASTKind.NewPlacement] = NewPlacementAST;
astConstructors[ASTKind.NestedNamespaceSpecifier] = NestedNamespaceSpecifierAST;
astConstructors[ASTKind.LabeledStatement] = LabeledStatementAST;
astConstructors[ASTKind.CaseStatement] = CaseStatementAST;
astConstructors[ASTKind.DefaultStatement] = DefaultStatementAST;
astConstructors[ASTKind.ExpressionStatement] = ExpressionStatementAST;
astConstructors[ASTKind.CompoundStatement] = CompoundStatementAST;
astConstructors[ASTKind.IfStatement] = IfStatementAST;
astConstructors[ASTKind.ConstevalIfStatement] = ConstevalIfStatementAST;
astConstructors[ASTKind.SwitchStatement] = SwitchStatementAST;
astConstructors[ASTKind.WhileStatement] = WhileStatementAST;
astConstructors[ASTKind.DoStatement] = DoStatementAST;
astConstructors[ASTKind.ForRangeStatement] = ForRangeStatementAST;
astConstructors[ASTKind.ForStatement] = ForStatementAST;
astConstructors[ASTKind.BreakStatement] = BreakStatementAST;
astConstructors[ASTKind.ContinueStatement] = ContinueStatementAST;
astConstructors[ASTKind.ReturnStatement] = ReturnStatementAST;
astConstructors[ASTKind.CoroutineReturnStatement] = CoroutineReturnStatementAST;
astConstructors[ASTKind.GotoStatement] = GotoStatementAST;
astConstructors[ASTKind.DeclarationStatement] = DeclarationStatementAST;
astConstructors[ASTKind.TryBlockStatement] = TryBlockStatementAST;
astConstructors[ASTKind.CharLiteralExpression] = CharLiteralExpressionAST;
astConstructors[ASTKind.BoolLiteralExpression] = BoolLiteralExpressionAST;
astConstructors[ASTKind.IntLiteralExpression] = IntLiteralExpressionAST;
astConstructors[ASTKind.FloatLiteralExpression] = FloatLiteralExpressionAST;
astConstructors[ASTKind.NullptrLiteralExpression] = NullptrLiteralExpressionAST;
astConstructors[ASTKind.StringLiteralExpression] = StringLiteralExpressionAST;
astConstructors[ASTKind.UserDefinedStringLiteralExpression] =
  UserDefinedStringLiteralExpressionAST;
astConstructors[ASTKind.ObjectLiteralExpression] = ObjectLiteralExpressionAST;
astConstructors[ASTKind.ThisExpression] = ThisExpressionAST;
astConstructors[ASTKind.PackIndexExpression] = PackIndexExpressionAST;
astConstructors[ASTKind.GenericSelectionExpression] =
  GenericSelectionExpressionAST;
astConstructors[ASTKind.NestedStatementExpression] =
  NestedStatementExpressionAST;
astConstructors[ASTKind.DefaultInitializerExpression] =
  DefaultInitializerExpressionAST;
astConstructors[ASTKind.NestedExpression] = NestedExpressionAST;
astConstructors[ASTKind.IdExpression] = IdExpressionAST;
astConstructors[ASTKind.LambdaExpression] = LambdaExpressionAST;
astConstructors[ASTKind.FoldExpression] = FoldExpressionAST;
astConstructors[ASTKind.RightFoldExpression] = RightFoldExpressionAST;
astConstructors[ASTKind.LeftFoldExpression] = LeftFoldExpressionAST;
astConstructors[ASTKind.RequiresExpression] = RequiresExpressionAST;
astConstructors[ASTKind.VaArgExpression] = VaArgExpressionAST;
astConstructors[ASTKind.SubscriptExpression] = SubscriptExpressionAST;
astConstructors[ASTKind.CallExpression] = CallExpressionAST;
astConstructors[ASTKind.TypeConstruction] = TypeConstructionAST;
astConstructors[ASTKind.BracedTypeConstruction] = BracedTypeConstructionAST;
astConstructors[ASTKind.SpliceMemberExpression] = SpliceMemberExpressionAST;
astConstructors[ASTKind.MemberExpression] = MemberExpressionAST;
astConstructors[ASTKind.PostIncrExpression] = PostIncrExpressionAST;
astConstructors[ASTKind.CppCastExpression] = CppCastExpressionAST;
astConstructors[ASTKind.BuiltinBitCastExpression] = BuiltinBitCastExpressionAST;
astConstructors[ASTKind.BuiltinOffsetofExpression] =
  BuiltinOffsetofExpressionAST;
astConstructors[ASTKind.TypeidExpression] = TypeidExpressionAST;
astConstructors[ASTKind.TypeidOfTypeExpression] = TypeidOfTypeExpressionAST;
astConstructors[ASTKind.SpliceExpression] = SpliceExpressionAST;
astConstructors[ASTKind.GlobalScopeReflectExpression] =
  GlobalScopeReflectExpressionAST;
astConstructors[ASTKind.NamespaceReflectExpression] =
  NamespaceReflectExpressionAST;
astConstructors[ASTKind.TypeIdReflectExpression] = TypeIdReflectExpressionAST;
astConstructors[ASTKind.ReflectExpression] = ReflectExpressionAST;
astConstructors[ASTKind.LabelAddressExpression] = LabelAddressExpressionAST;
astConstructors[ASTKind.UnaryExpression] = UnaryExpressionAST;
astConstructors[ASTKind.AwaitExpression] = AwaitExpressionAST;
astConstructors[ASTKind.SizeofExpression] = SizeofExpressionAST;
astConstructors[ASTKind.SizeofTypeExpression] = SizeofTypeExpressionAST;
astConstructors[ASTKind.SizeofPackExpression] = SizeofPackExpressionAST;
astConstructors[ASTKind.AlignofTypeExpression] = AlignofTypeExpressionAST;
astConstructors[ASTKind.AlignofExpression] = AlignofExpressionAST;
astConstructors[ASTKind.NoexceptExpression] = NoexceptExpressionAST;
astConstructors[ASTKind.NewExpression] = NewExpressionAST;
astConstructors[ASTKind.DeleteExpression] = DeleteExpressionAST;
astConstructors[ASTKind.CastExpression] = CastExpressionAST;
astConstructors[ASTKind.ImplicitCastExpression] = ImplicitCastExpressionAST;
astConstructors[ASTKind.ConstExpression] = ConstExpressionAST;
astConstructors[ASTKind.BinaryExpression] = BinaryExpressionAST;
astConstructors[ASTKind.ConditionalExpression] = ConditionalExpressionAST;
astConstructors[ASTKind.YieldExpression] = YieldExpressionAST;
astConstructors[ASTKind.ThrowExpression] = ThrowExpressionAST;
astConstructors[ASTKind.AssignmentExpression] = AssignmentExpressionAST;
astConstructors[ASTKind.TargetExpression] = TargetExpressionAST;
astConstructors[ASTKind.RightExpression] = RightExpressionAST;
astConstructors[ASTKind.CompoundAssignmentExpression] =
  CompoundAssignmentExpressionAST;
astConstructors[ASTKind.PackExpansionExpression] = PackExpansionExpressionAST;
astConstructors[ASTKind.DesignatedInitializerClause] =
  DesignatedInitializerClauseAST;
astConstructors[ASTKind.TypeTraitExpression] = TypeTraitExpressionAST;
astConstructors[ASTKind.ConditionExpression] = ConditionExpressionAST;
astConstructors[ASTKind.EqualInitializer] = EqualInitializerAST;
astConstructors[ASTKind.BracedInitList] = BracedInitListAST;
astConstructors[ASTKind.ParenInitializer] = ParenInitializerAST;
astConstructors[ASTKind.ThreeWayComparisonExpression] =
  ThreeWayComparisonExpressionAST;
astConstructors[ASTKind.DefaultGenericAssociation] =
  DefaultGenericAssociationAST;
astConstructors[ASTKind.TypeGenericAssociation] = TypeGenericAssociationAST;
astConstructors[ASTKind.DotDesignator] = DotDesignatorAST;
astConstructors[ASTKind.SubscriptDesignator] = SubscriptDesignatorAST;
astConstructors[ASTKind.TemplateTypeParameter] = TemplateTypeParameterAST;
astConstructors[ASTKind.NonTypeTemplateParameter] = NonTypeTemplateParameterAST;
astConstructors[ASTKind.TypenameTypeParameter] = TypenameTypeParameterAST;
astConstructors[ASTKind.ConstraintTypeParameter] = ConstraintTypeParameterAST;
astConstructors[ASTKind.TypedefSpecifier] = TypedefSpecifierAST;
astConstructors[ASTKind.FriendSpecifier] = FriendSpecifierAST;
astConstructors[ASTKind.ConstevalSpecifier] = ConstevalSpecifierAST;
astConstructors[ASTKind.ConstinitSpecifier] = ConstinitSpecifierAST;
astConstructors[ASTKind.ConstexprSpecifier] = ConstexprSpecifierAST;
astConstructors[ASTKind.InlineSpecifier] = InlineSpecifierAST;
astConstructors[ASTKind.NoreturnSpecifier] = NoreturnSpecifierAST;
astConstructors[ASTKind.StaticSpecifier] = StaticSpecifierAST;
astConstructors[ASTKind.ExternSpecifier] = ExternSpecifierAST;
astConstructors[ASTKind.RegisterSpecifier] = RegisterSpecifierAST;
astConstructors[ASTKind.ThreadLocalSpecifier] = ThreadLocalSpecifierAST;
astConstructors[ASTKind.ThreadSpecifier] = ThreadSpecifierAST;
astConstructors[ASTKind.MutableSpecifier] = MutableSpecifierAST;
astConstructors[ASTKind.VirtualSpecifier] = VirtualSpecifierAST;
astConstructors[ASTKind.ExplicitSpecifier] = ExplicitSpecifierAST;
astConstructors[ASTKind.AutoTypeSpecifier] = AutoTypeSpecifierAST;
astConstructors[ASTKind.VoidTypeSpecifier] = VoidTypeSpecifierAST;
astConstructors[ASTKind.SizeTypeSpecifier] = SizeTypeSpecifierAST;
astConstructors[ASTKind.SignTypeSpecifier] = SignTypeSpecifierAST;
astConstructors[ASTKind.BuiltinTypeSpecifier] = BuiltinTypeSpecifierAST;
astConstructors[ASTKind.UnaryBuiltinTypeSpecifier] =
  UnaryBuiltinTypeSpecifierAST;
astConstructors[ASTKind.BinaryBuiltinTypeSpecifier] =
  BinaryBuiltinTypeSpecifierAST;
astConstructors[ASTKind.IntegralTypeSpecifier] = IntegralTypeSpecifierAST;
astConstructors[ASTKind.FloatingPointTypeSpecifier] =
  FloatingPointTypeSpecifierAST;
astConstructors[ASTKind.ComplexTypeSpecifier] = ComplexTypeSpecifierAST;
astConstructors[ASTKind.NamedTypeSpecifier] = NamedTypeSpecifierAST;
astConstructors[ASTKind.AtomicTypeSpecifier] = AtomicTypeSpecifierAST;
astConstructors[ASTKind.BitIntTypeSpecifier] = BitIntTypeSpecifierAST;
astConstructors[ASTKind.UnderlyingTypeSpecifier] = UnderlyingTypeSpecifierAST;
astConstructors[ASTKind.ElaboratedTypeSpecifier] = ElaboratedTypeSpecifierAST;
astConstructors[ASTKind.DecltypeAutoSpecifier] = DecltypeAutoSpecifierAST;
astConstructors[ASTKind.DecltypeSpecifier] = DecltypeSpecifierAST;
astConstructors[ASTKind.PlaceholderTypeSpecifier] = PlaceholderTypeSpecifierAST;
astConstructors[ASTKind.ConstQualifier] = ConstQualifierAST;
astConstructors[ASTKind.VolatileQualifier] = VolatileQualifierAST;
astConstructors[ASTKind.AtomicQualifier] = AtomicQualifierAST;
astConstructors[ASTKind.RestrictQualifier] = RestrictQualifierAST;
astConstructors[ASTKind.EnumSpecifier] = EnumSpecifierAST;
astConstructors[ASTKind.ClassSpecifier] = ClassSpecifierAST;
astConstructors[ASTKind.TypenameSpecifier] = TypenameSpecifierAST;
astConstructors[ASTKind.SplicerTypeSpecifier] = SplicerTypeSpecifierAST;
astConstructors[ASTKind.PointerOperator] = PointerOperatorAST;
astConstructors[ASTKind.ReferenceOperator] = ReferenceOperatorAST;
astConstructors[ASTKind.PtrToMemberOperator] = PtrToMemberOperatorAST;
astConstructors[ASTKind.BitfieldDeclarator] = BitfieldDeclaratorAST;
astConstructors[ASTKind.ParameterPack] = ParameterPackAST;
astConstructors[ASTKind.IdDeclarator] = IdDeclaratorAST;
astConstructors[ASTKind.NestedDeclarator] = NestedDeclaratorAST;
astConstructors[ASTKind.FunctionDeclaratorChunk] = FunctionDeclaratorChunkAST;
astConstructors[ASTKind.ArrayDeclaratorChunk] = ArrayDeclaratorChunkAST;
astConstructors[ASTKind.NameId] = NameIdAST;
astConstructors[ASTKind.DestructorId] = DestructorIdAST;
astConstructors[ASTKind.DecltypeId] = DecltypeIdAST;
astConstructors[ASTKind.OperatorFunctionId] = OperatorFunctionIdAST;
astConstructors[ASTKind.LiteralOperatorId] = LiteralOperatorIdAST;
astConstructors[ASTKind.ConversionFunctionId] = ConversionFunctionIdAST;
astConstructors[ASTKind.SimpleTemplateId] = SimpleTemplateIdAST;
astConstructors[ASTKind.LiteralOperatorTemplateId] =
  LiteralOperatorTemplateIdAST;
astConstructors[ASTKind.OperatorFunctionTemplateId] =
  OperatorFunctionTemplateIdAST;
astConstructors[ASTKind.GlobalNestedNameSpecifier] =
  GlobalNestedNameSpecifierAST;
astConstructors[ASTKind.SimpleNestedNameSpecifier] =
  SimpleNestedNameSpecifierAST;
astConstructors[ASTKind.DecltypeNestedNameSpecifier] =
  DecltypeNestedNameSpecifierAST;
astConstructors[ASTKind.TemplateNestedNameSpecifier] =
  TemplateNestedNameSpecifierAST;
astConstructors[ASTKind.DefaultFunctionBody] = DefaultFunctionBodyAST;
astConstructors[ASTKind.CompoundStatementFunctionBody] =
  CompoundStatementFunctionBodyAST;
astConstructors[ASTKind.TryStatementFunctionBody] = TryStatementFunctionBodyAST;
astConstructors[ASTKind.DeleteFunctionBody] = DeleteFunctionBodyAST;
astConstructors[ASTKind.TypeTemplateArgument] = TypeTemplateArgumentAST;
astConstructors[ASTKind.ExpressionTemplateArgument] =
  ExpressionTemplateArgumentAST;
astConstructors[ASTKind.ThrowExceptionSpecifier] = ThrowExceptionSpecifierAST;
astConstructors[ASTKind.NoexceptSpecifier] = NoexceptSpecifierAST;
astConstructors[ASTKind.SimpleRequirement] = SimpleRequirementAST;
astConstructors[ASTKind.CompoundRequirement] = CompoundRequirementAST;
astConstructors[ASTKind.TypeRequirement] = TypeRequirementAST;
astConstructors[ASTKind.NestedRequirement] = NestedRequirementAST;
astConstructors[ASTKind.NewParenInitializer] = NewParenInitializerAST;
astConstructors[ASTKind.NewBracedInitializer] = NewBracedInitializerAST;
astConstructors[ASTKind.ParenMemInitializer] = ParenMemInitializerAST;
astConstructors[ASTKind.BracedMemInitializer] = BracedMemInitializerAST;
astConstructors[ASTKind.ThisLambdaCapture] = ThisLambdaCaptureAST;
astConstructors[ASTKind.DerefThisLambdaCapture] = DerefThisLambdaCaptureAST;
astConstructors[ASTKind.SimpleLambdaCapture] = SimpleLambdaCaptureAST;
astConstructors[ASTKind.RefLambdaCapture] = RefLambdaCaptureAST;
astConstructors[ASTKind.RefInitLambdaCapture] = RefInitLambdaCaptureAST;
astConstructors[ASTKind.InitLambdaCapture] = InitLambdaCaptureAST;
astConstructors[ASTKind.EllipsisExceptionDeclaration] =
  EllipsisExceptionDeclarationAST;
astConstructors[ASTKind.TypeExceptionDeclaration] = TypeExceptionDeclarationAST;
astConstructors[ASTKind.CxxAttribute] = CxxAttributeAST;
astConstructors[ASTKind.GccAttribute] = GccAttributeAST;
astConstructors[ASTKind.AlignasAttribute] = AlignasAttributeAST;
astConstructors[ASTKind.AlignasTypeAttribute] = AlignasTypeAttributeAST;
astConstructors[ASTKind.AsmAttribute] = AsmAttributeAST;
astConstructors[ASTKind.ScopedAttributeToken] = ScopedAttributeTokenAST;
astConstructors[ASTKind.SimpleAttributeToken] = SimpleAttributeTokenAST;
symbolConstructors[SymbolKind.kNamespace] = NamespaceSymbol;
symbolConstructors[SymbolKind.kConcept] = ConceptSymbol;
symbolConstructors[SymbolKind.kDeductionGuide] = DeductionGuideSymbol;
symbolConstructors[SymbolKind.kBaseClass] = BaseClassSymbol;
symbolConstructors[SymbolKind.kInjectedClassName] = InjectedClassNameSymbol;
symbolConstructors[SymbolKind.kUnresolved] = UnresolvedSymbol;
symbolConstructors[SymbolKind.kClass] = ClassSymbol;
symbolConstructors[SymbolKind.kEnum] = EnumSymbol;
symbolConstructors[SymbolKind.kScopedEnum] = ScopedEnumSymbol;
symbolConstructors[SymbolKind.kFunction] = FunctionSymbol;
symbolConstructors[SymbolKind.kOverloadSet] = OverloadSetSymbol;
symbolConstructors[SymbolKind.kLambda] = LambdaSymbol;
symbolConstructors[SymbolKind.kFunctionParameters] = FunctionParametersSymbol;
symbolConstructors[SymbolKind.kTemplateParameters] = TemplateParametersSymbol;
symbolConstructors[SymbolKind.kBlock] = BlockSymbol;
symbolConstructors[SymbolKind.kTypeAlias] = TypeAliasSymbol;
symbolConstructors[SymbolKind.kVariable] = VariableSymbol;
symbolConstructors[SymbolKind.kField] = FieldSymbol;
symbolConstructors[SymbolKind.kParameter] = ParameterSymbol;
symbolConstructors[SymbolKind.kParameterPack] = ParameterPackSymbol;
symbolConstructors[SymbolKind.kTypeParameter] = TypeParameterSymbol;
symbolConstructors[SymbolKind.kNonTypeParameter] = NonTypeParameterSymbol;
symbolConstructors[SymbolKind.kTemplateTypeParameter] =
  TemplateTypeParameterSymbol;
symbolConstructors[SymbolKind.kConstraintTypeParameter] =
  ConstraintTypeParameterSymbol;
symbolConstructors[SymbolKind.kEnumerator] = EnumeratorSymbol;
symbolConstructors[SymbolKind.kNamespaceAlias] = NamespaceAliasSymbol;
symbolConstructors[SymbolKind.kUsingDeclaration] = UsingDeclarationSymbol;
typeConstructors[TypeKind.kBuiltinVaList] = BuiltinVaListType;
typeConstructors[TypeKind.kBuiltinMetaInfo] = BuiltinMetaInfoType;
typeConstructors[TypeKind.kVoid] = VoidType;
typeConstructors[TypeKind.kNullptr] = NullptrType;
typeConstructors[TypeKind.kDecltypeAuto] = DecltypeAutoType;
typeConstructors[TypeKind.kAuto] = AutoType;
typeConstructors[TypeKind.kBool] = BoolType;
typeConstructors[TypeKind.kSignedChar] = SignedCharType;
typeConstructors[TypeKind.kShortInt] = ShortIntType;
typeConstructors[TypeKind.kInt] = IntType;
typeConstructors[TypeKind.kLongInt] = LongIntType;
typeConstructors[TypeKind.kLongLongInt] = LongLongIntType;
typeConstructors[TypeKind.kInt128] = Int128Type;
typeConstructors[TypeKind.kUnsignedChar] = UnsignedCharType;
typeConstructors[TypeKind.kUnsignedShortInt] = UnsignedShortIntType;
typeConstructors[TypeKind.kUnsignedInt] = UnsignedIntType;
typeConstructors[TypeKind.kUnsignedLongInt] = UnsignedLongIntType;
typeConstructors[TypeKind.kUnsignedLongLongInt] = UnsignedLongLongIntType;
typeConstructors[TypeKind.kUnsignedInt128] = UnsignedInt128Type;
typeConstructors[TypeKind.kChar] = CharType;
typeConstructors[TypeKind.kChar8] = Char8Type;
typeConstructors[TypeKind.kChar16] = Char16Type;
typeConstructors[TypeKind.kChar32] = Char32Type;
typeConstructors[TypeKind.kWideChar] = WideCharType;
typeConstructors[TypeKind.kFloat] = FloatType;
typeConstructors[TypeKind.kDouble] = DoubleType;
typeConstructors[TypeKind.kLongDouble] = LongDoubleType;
typeConstructors[TypeKind.kFloat16] = Float16Type;
typeConstructors[TypeKind.kQual] = QualType;
typeConstructors[TypeKind.kBoundedArray] = BoundedArrayType;
typeConstructors[TypeKind.kUnboundedArray] = UnboundedArrayType;
typeConstructors[TypeKind.kPointer] = PointerType;
typeConstructors[TypeKind.kLvalueReference] = LvalueReferenceType;
typeConstructors[TypeKind.kRvalueReference] = RvalueReferenceType;
typeConstructors[TypeKind.kOverloadSet] = OverloadSetType;
typeConstructors[TypeKind.kFunction] = FunctionType;
typeConstructors[TypeKind.kClass] = ClassType;
typeConstructors[TypeKind.kEnum] = EnumType;
typeConstructors[TypeKind.kScopedEnum] = ScopedEnumType;
typeConstructors[TypeKind.kMemberObjectPointer] = MemberObjectPointerType;
typeConstructors[TypeKind.kMemberFunctionPointer] = MemberFunctionPointerType;
typeConstructors[TypeKind.kNamespace] = NamespaceType;
typeConstructors[TypeKind.kTypeParameter] = TypeParameterType;
typeConstructors[TypeKind.kTemplateTypeParameter] = TemplateTypeParameterType;
typeConstructors[TypeKind.kUnresolvedName] = UnresolvedNameType;
typeConstructors[TypeKind.kUnresolvedBoundedArray] = UnresolvedBoundedArrayType;
typeConstructors[TypeKind.kUnresolvedUnderlying] = UnresolvedUnderlyingType;
typeConstructors[TypeKind.kUnresolvedBuiltin] = UnresolvedBuiltinType;
typeConstructors[TypeKind.kBitInt] = BitIntType;
typeConstructors[TypeKind.kUnsignedBitInt] = UnsignedBitIntType;
typeConstructors[TypeKind.kUnresolvedBitInt] = UnresolvedBitIntType;
typeConstructors[TypeKind.kVector] = VectorType;
typeConstructors[TypeKind.kUnresolvedVector] = UnresolvedVectorType;
typeConstructors[TypeKind.kComplex] = ComplexType;
typeConstructors[TypeKind.kAtomic] = AtomicType;
nameConstructors[NameKind.kIdentifier] = Identifier;
nameConstructors[NameKind.kOperatorId] = OperatorId;
nameConstructors[NameKind.kDestructorId] = DestructorId;
nameConstructors[NameKind.kLiteralOperatorId] = LiteralOperatorId;
nameConstructors[NameKind.kConversionFunctionId] = ConversionFunctionId;
nameConstructors[NameKind.kTemplateId] = TemplateId;
const childSlots: Array<ReadonlyArray<readonly [number, boolean]>> = [];
childSlots[ASTKind.TranslationUnit] = [[TranslationUnitASTSlotBase + 2, true]];
childSlots[ASTKind.ModuleUnit] = [
  [ModuleUnitASTSlotBase + 2, false],
  [ModuleUnitASTSlotBase + 3, false],
  [ModuleUnitASTSlotBase + 4, true],
  [ModuleUnitASTSlotBase + 5, false],
];
childSlots[ASTKind.SimpleDeclaration] = [
  [SimpleDeclarationASTSlotBase + 1, true],
  [SimpleDeclarationASTSlotBase + 2, true],
  [SimpleDeclarationASTSlotBase + 3, true],
  [SimpleDeclarationASTSlotBase + 4, false],
];
childSlots[ASTKind.AsmDeclaration] = [
  [AsmDeclarationASTSlotBase + 1, true],
  [AsmDeclarationASTSlotBase + 2, true],
  [AsmDeclarationASTSlotBase + 6, true],
  [AsmDeclarationASTSlotBase + 7, true],
  [AsmDeclarationASTSlotBase + 8, true],
  [AsmDeclarationASTSlotBase + 9, true],
];
childSlots[ASTKind.NamespaceAliasDefinition] = [
  [NamespaceAliasDefinitionASTSlotBase + 4, false],
  [NamespaceAliasDefinitionASTSlotBase + 5, false],
];
childSlots[ASTKind.UsingDeclaration] = [
  [UsingDeclarationASTSlotBase + 2, true],
];
childSlots[ASTKind.UsingEnumDeclaration] = [
  [UsingEnumDeclarationASTSlotBase + 2, false],
];
childSlots[ASTKind.UsingDirective] = [
  [UsingDirectiveASTSlotBase + 1, true],
  [UsingDirectiveASTSlotBase + 4, false],
  [UsingDirectiveASTSlotBase + 5, false],
];
childSlots[ASTKind.StaticAssertDeclaration] = [
  [StaticAssertDeclarationASTSlotBase + 3, false],
];
childSlots[ASTKind.AliasDeclaration] = [
  [AliasDeclarationASTSlotBase + 3, true],
  [AliasDeclarationASTSlotBase + 5, true],
  [AliasDeclarationASTSlotBase + 6, false],
];
childSlots[ASTKind.OpaqueEnumDeclaration] = [
  [OpaqueEnumDeclarationASTSlotBase + 3, true],
  [OpaqueEnumDeclarationASTSlotBase + 4, false],
  [OpaqueEnumDeclarationASTSlotBase + 5, false],
  [OpaqueEnumDeclarationASTSlotBase + 7, true],
];
childSlots[ASTKind.FunctionDefinition] = [
  [FunctionDefinitionASTSlotBase + 1, true],
  [FunctionDefinitionASTSlotBase + 2, true],
  [FunctionDefinitionASTSlotBase + 3, false],
  [FunctionDefinitionASTSlotBase + 4, false],
  [FunctionDefinitionASTSlotBase + 5, false],
];
childSlots[ASTKind.TemplateDeclaration] = [
  [TemplateDeclarationASTSlotBase + 3, true],
  [TemplateDeclarationASTSlotBase + 5, false],
  [TemplateDeclarationASTSlotBase + 6, false],
];
childSlots[ASTKind.ConceptDefinition] = [
  [ConceptDefinitionASTSlotBase + 4, false],
];
childSlots[ASTKind.DeductionGuide] = [
  [DeductionGuideASTSlotBase + 1, true],
  [DeductionGuideASTSlotBase + 2, false],
  [DeductionGuideASTSlotBase + 5, false],
  [DeductionGuideASTSlotBase + 8, false],
];
childSlots[ASTKind.ExplicitInstantiation] = [
  [ExplicitInstantiationASTSlotBase + 3, false],
];
childSlots[ASTKind.ExportDeclaration] = [
  [ExportDeclarationASTSlotBase + 2, false],
];
childSlots[ASTKind.ExportCompoundDeclaration] = [
  [ExportCompoundDeclarationASTSlotBase + 3, true],
];
childSlots[ASTKind.LinkageSpecification] = [
  [LinkageSpecificationASTSlotBase + 4, true],
];
childSlots[ASTKind.NamespaceDefinition] = [
  [NamespaceDefinitionASTSlotBase + 3, true],
  [NamespaceDefinitionASTSlotBase + 4, true],
  [NamespaceDefinitionASTSlotBase + 6, true],
  [NamespaceDefinitionASTSlotBase + 8, true],
];
childSlots[ASTKind.EmptyDeclaration] = [];
childSlots[ASTKind.AttributeDeclaration] = [
  [AttributeDeclarationASTSlotBase + 1, true],
];
childSlots[ASTKind.ModuleImportDeclaration] = [
  [ModuleImportDeclarationASTSlotBase + 2, false],
  [ModuleImportDeclarationASTSlotBase + 3, true],
];
childSlots[ASTKind.ParameterDeclaration] = [
  [ParameterDeclarationASTSlotBase + 1, true],
  [ParameterDeclarationASTSlotBase + 3, true],
  [ParameterDeclarationASTSlotBase + 4, false],
  [ParameterDeclarationASTSlotBase + 6, false],
];
childSlots[ASTKind.AccessDeclaration] = [];
childSlots[ASTKind.ForRangeDeclaration] = [];
childSlots[ASTKind.StructuredBindingDeclaration] = [
  [StructuredBindingDeclarationASTSlotBase + 1, true],
  [StructuredBindingDeclarationASTSlotBase + 2, true],
  [StructuredBindingDeclarationASTSlotBase + 5, true],
  [StructuredBindingDeclarationASTSlotBase + 7, false],
  [StructuredBindingDeclarationASTSlotBase + 9, false],
  [StructuredBindingDeclarationASTSlotBase + 10, true],
];
childSlots[ASTKind.AsmOperand] = [[AsmOperandASTSlotBase + 6, false]];
childSlots[ASTKind.AsmQualifier] = [];
childSlots[ASTKind.AsmClobber] = [];
childSlots[ASTKind.AsmGotoLabel] = [];
childSlots[ASTKind.Splicer] = [[SplicerASTSlotBase + 4, false]];
childSlots[ASTKind.GlobalModuleFragment] = [
  [GlobalModuleFragmentASTSlotBase + 3, true],
];
childSlots[ASTKind.PrivateModuleFragment] = [
  [PrivateModuleFragmentASTSlotBase + 5, true],
];
childSlots[ASTKind.ModuleDeclaration] = [
  [ModuleDeclarationASTSlotBase + 3, false],
  [ModuleDeclarationASTSlotBase + 4, false],
  [ModuleDeclarationASTSlotBase + 5, true],
];
childSlots[ASTKind.ModuleName] = [[ModuleNameASTSlotBase + 1, false]];
childSlots[ASTKind.ModuleQualifier] = [[ModuleQualifierASTSlotBase + 1, false]];
childSlots[ASTKind.ModulePartition] = [[ModulePartitionASTSlotBase + 2, false]];
childSlots[ASTKind.ImportName] = [
  [ImportNameASTSlotBase + 2, false],
  [ImportNameASTSlotBase + 3, false],
];
childSlots[ASTKind.InitDeclarator] = [
  [InitDeclaratorASTSlotBase + 1, false],
  [InitDeclaratorASTSlotBase + 2, false],
  [InitDeclaratorASTSlotBase + 3, false],
];
childSlots[ASTKind.Declarator] = [
  [DeclaratorASTSlotBase + 1, true],
  [DeclaratorASTSlotBase + 2, false],
  [DeclaratorASTSlotBase + 3, true],
];
childSlots[ASTKind.UsingDeclarator] = [
  [UsingDeclaratorASTSlotBase + 2, false],
  [UsingDeclaratorASTSlotBase + 3, false],
];
childSlots[ASTKind.Enumerator] = [
  [EnumeratorASTSlotBase + 2, true],
  [EnumeratorASTSlotBase + 4, false],
];
childSlots[ASTKind.TypeId] = [
  [TypeIdASTSlotBase + 1, true],
  [TypeIdASTSlotBase + 2, true],
  [TypeIdASTSlotBase + 3, false],
];
childSlots[ASTKind.Handler] = [
  [HandlerASTSlotBase + 3, false],
  [HandlerASTSlotBase + 5, false],
];
childSlots[ASTKind.BaseSpecifier] = [
  [BaseSpecifierASTSlotBase + 1, true],
  [BaseSpecifierASTSlotBase + 4, false],
  [BaseSpecifierASTSlotBase + 6, false],
];
childSlots[ASTKind.RequiresClause] = [[RequiresClauseASTSlotBase + 2, false]];
childSlots[ASTKind.ParameterDeclarationClause] = [
  [ParameterDeclarationClauseASTSlotBase + 1, true],
];
childSlots[ASTKind.TrailingReturnType] = [
  [TrailingReturnTypeASTSlotBase + 2, false],
];
childSlots[ASTKind.LambdaSpecifier] = [];
childSlots[ASTKind.TypeConstraint] = [
  [TypeConstraintASTSlotBase + 1, false],
  [TypeConstraintASTSlotBase + 4, true],
];
childSlots[ASTKind.AttributeArgumentClause] = [
  [AttributeArgumentClauseASTSlotBase + 2, true],
];
childSlots[ASTKind.Attribute] = [
  [AttributeASTSlotBase + 1, false],
  [AttributeASTSlotBase + 2, false],
];
childSlots[ASTKind.AttributeUsingPrefix] = [];
childSlots[ASTKind.NewPlacement] = [[NewPlacementASTSlotBase + 2, true]];
childSlots[ASTKind.NestedNamespaceSpecifier] = [];
childSlots[ASTKind.LabeledStatement] = [
  [LabeledStatementASTSlotBase + 3, false],
];
childSlots[ASTKind.CaseStatement] = [[CaseStatementASTSlotBase + 2, false]];
childSlots[ASTKind.DefaultStatement] = [];
childSlots[ASTKind.ExpressionStatement] = [
  [ExpressionStatementASTSlotBase + 1, true],
  [ExpressionStatementASTSlotBase + 2, false],
];
childSlots[ASTKind.CompoundStatement] = [
  [CompoundStatementASTSlotBase + 1, true],
  [CompoundStatementASTSlotBase + 3, true],
];
childSlots[ASTKind.IfStatement] = [
  [IfStatementASTSlotBase + 1, true],
  [IfStatementASTSlotBase + 5, false],
  [IfStatementASTSlotBase + 6, false],
  [IfStatementASTSlotBase + 8, false],
  [IfStatementASTSlotBase + 10, false],
];
childSlots[ASTKind.ConstevalIfStatement] = [
  [ConstevalIfStatementASTSlotBase + 1, true],
  [ConstevalIfStatementASTSlotBase + 5, false],
  [ConstevalIfStatementASTSlotBase + 7, false],
];
childSlots[ASTKind.SwitchStatement] = [
  [SwitchStatementASTSlotBase + 1, true],
  [SwitchStatementASTSlotBase + 4, false],
  [SwitchStatementASTSlotBase + 5, false],
  [SwitchStatementASTSlotBase + 7, false],
];
childSlots[ASTKind.WhileStatement] = [
  [WhileStatementASTSlotBase + 1, true],
  [WhileStatementASTSlotBase + 4, false],
  [WhileStatementASTSlotBase + 6, false],
];
childSlots[ASTKind.DoStatement] = [
  [DoStatementASTSlotBase + 1, true],
  [DoStatementASTSlotBase + 3, false],
  [DoStatementASTSlotBase + 6, false],
];
childSlots[ASTKind.ForRangeStatement] = [
  [ForRangeStatementASTSlotBase + 1, true],
  [ForRangeStatementASTSlotBase + 4, false],
  [ForRangeStatementASTSlotBase + 5, false],
  [ForRangeStatementASTSlotBase + 7, false],
  [ForRangeStatementASTSlotBase + 9, false],
  [ForRangeStatementASTSlotBase + 10, false],
  [ForRangeStatementASTSlotBase + 11, false],
  [ForRangeStatementASTSlotBase + 12, false],
  [ForRangeStatementASTSlotBase + 13, false],
  [ForRangeStatementASTSlotBase + 14, false],
];
childSlots[ASTKind.ForStatement] = [
  [ForStatementASTSlotBase + 1, true],
  [ForStatementASTSlotBase + 4, false],
  [ForStatementASTSlotBase + 5, false],
  [ForStatementASTSlotBase + 7, false],
  [ForStatementASTSlotBase + 9, false],
];
childSlots[ASTKind.BreakStatement] = [[BreakStatementASTSlotBase + 1, true]];
childSlots[ASTKind.ContinueStatement] = [
  [ContinueStatementASTSlotBase + 1, true],
];
childSlots[ASTKind.ReturnStatement] = [
  [ReturnStatementASTSlotBase + 1, true],
  [ReturnStatementASTSlotBase + 3, false],
];
childSlots[ASTKind.CoroutineReturnStatement] = [
  [CoroutineReturnStatementASTSlotBase + 1, true],
  [CoroutineReturnStatementASTSlotBase + 3, false],
];
childSlots[ASTKind.GotoStatement] = [
  [GotoStatementASTSlotBase + 1, true],
  [GotoStatementASTSlotBase + 2, false],
];
childSlots[ASTKind.DeclarationStatement] = [
  [DeclarationStatementASTSlotBase + 1, false],
];
childSlots[ASTKind.TryBlockStatement] = [
  [TryBlockStatementASTSlotBase + 1, true],
  [TryBlockStatementASTSlotBase + 3, false],
  [TryBlockStatementASTSlotBase + 4, true],
];
childSlots[ASTKind.CharLiteralExpression] = [
  [CharLiteralExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.BoolLiteralExpression] = [];
childSlots[ASTKind.IntLiteralExpression] = [
  [IntLiteralExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.FloatLiteralExpression] = [
  [FloatLiteralExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.NullptrLiteralExpression] = [];
childSlots[ASTKind.StringLiteralExpression] = [];
childSlots[ASTKind.UserDefinedStringLiteralExpression] = [
  [UserDefinedStringLiteralExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.ObjectLiteralExpression] = [
  [ObjectLiteralExpressionASTSlotBase + 4, false],
  [ObjectLiteralExpressionASTSlotBase + 6, false],
];
childSlots[ASTKind.ThisExpression] = [];
childSlots[ASTKind.PackIndexExpression] = [
  [PackIndexExpressionASTSlotBase + 3, false],
  [PackIndexExpressionASTSlotBase + 6, false],
];
childSlots[ASTKind.GenericSelectionExpression] = [
  [GenericSelectionExpressionASTSlotBase + 5, false],
  [GenericSelectionExpressionASTSlotBase + 7, true],
];
childSlots[ASTKind.NestedStatementExpression] = [
  [NestedStatementExpressionASTSlotBase + 4, false],
];
childSlots[ASTKind.DefaultInitializerExpression] = [
  [DefaultInitializerExpressionASTSlotBase + 3, false],
];
childSlots[ASTKind.NestedExpression] = [
  [NestedExpressionASTSlotBase + 4, false],
];
childSlots[ASTKind.IdExpression] = [
  [IdExpressionASTSlotBase + 3, false],
  [IdExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.LambdaExpression] = [
  [LambdaExpressionASTSlotBase + 5, true],
  [LambdaExpressionASTSlotBase + 8, true],
  [LambdaExpressionASTSlotBase + 10, false],
  [LambdaExpressionASTSlotBase + 11, true],
  [LambdaExpressionASTSlotBase + 13, false],
  [LambdaExpressionASTSlotBase + 15, true],
  [LambdaExpressionASTSlotBase + 16, true],
  [LambdaExpressionASTSlotBase + 17, false],
  [LambdaExpressionASTSlotBase + 18, true],
  [LambdaExpressionASTSlotBase + 19, false],
  [LambdaExpressionASTSlotBase + 20, false],
  [LambdaExpressionASTSlotBase + 21, false],
];
childSlots[ASTKind.FoldExpression] = [
  [FoldExpressionASTSlotBase + 4, false],
  [FoldExpressionASTSlotBase + 8, false],
];
childSlots[ASTKind.RightFoldExpression] = [
  [RightFoldExpressionASTSlotBase + 4, false],
];
childSlots[ASTKind.LeftFoldExpression] = [
  [LeftFoldExpressionASTSlotBase + 6, false],
];
childSlots[ASTKind.RequiresExpression] = [
  [RequiresExpressionASTSlotBase + 5, false],
  [RequiresExpressionASTSlotBase + 8, true],
];
childSlots[ASTKind.VaArgExpression] = [
  [VaArgExpressionASTSlotBase + 5, false],
  [VaArgExpressionASTSlotBase + 7, false],
];
childSlots[ASTKind.SubscriptExpression] = [
  [SubscriptExpressionASTSlotBase + 3, false],
  [SubscriptExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.CallExpression] = [
  [CallExpressionASTSlotBase + 3, false],
  [CallExpressionASTSlotBase + 5, true],
];
childSlots[ASTKind.TypeConstruction] = [
  [TypeConstructionASTSlotBase + 3, false],
  [TypeConstructionASTSlotBase + 5, true],
];
childSlots[ASTKind.BracedTypeConstruction] = [
  [BracedTypeConstructionASTSlotBase + 3, false],
  [BracedTypeConstructionASTSlotBase + 4, false],
];
childSlots[ASTKind.SpliceMemberExpression] = [
  [SpliceMemberExpressionASTSlotBase + 3, false],
  [SpliceMemberExpressionASTSlotBase + 6, false],
];
childSlots[ASTKind.MemberExpression] = [
  [MemberExpressionASTSlotBase + 3, false],
  [MemberExpressionASTSlotBase + 5, false],
  [MemberExpressionASTSlotBase + 7, false],
];
childSlots[ASTKind.PostIncrExpression] = [
  [PostIncrExpressionASTSlotBase + 3, false],
];
childSlots[ASTKind.CppCastExpression] = [
  [CppCastExpressionASTSlotBase + 5, false],
  [CppCastExpressionASTSlotBase + 8, false],
];
childSlots[ASTKind.BuiltinBitCastExpression] = [
  [BuiltinBitCastExpressionASTSlotBase + 5, false],
  [BuiltinBitCastExpressionASTSlotBase + 7, false],
];
childSlots[ASTKind.BuiltinOffsetofExpression] = [
  [BuiltinOffsetofExpressionASTSlotBase + 5, false],
  [BuiltinOffsetofExpressionASTSlotBase + 8, true],
];
childSlots[ASTKind.TypeidExpression] = [
  [TypeidExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.TypeidOfTypeExpression] = [
  [TypeidOfTypeExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.SpliceExpression] = [
  [SpliceExpressionASTSlotBase + 3, false],
];
childSlots[ASTKind.GlobalScopeReflectExpression] = [];
childSlots[ASTKind.NamespaceReflectExpression] = [];
childSlots[ASTKind.TypeIdReflectExpression] = [
  [TypeIdReflectExpressionASTSlotBase + 4, false],
];
childSlots[ASTKind.ReflectExpression] = [
  [ReflectExpressionASTSlotBase + 4, false],
];
childSlots[ASTKind.LabelAddressExpression] = [];
childSlots[ASTKind.UnaryExpression] = [[UnaryExpressionASTSlotBase + 4, false]];
childSlots[ASTKind.AwaitExpression] = [[AwaitExpressionASTSlotBase + 4, false]];
childSlots[ASTKind.SizeofExpression] = [
  [SizeofExpressionASTSlotBase + 4, false],
];
childSlots[ASTKind.SizeofTypeExpression] = [
  [SizeofTypeExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.SizeofPackExpression] = [];
childSlots[ASTKind.AlignofTypeExpression] = [
  [AlignofTypeExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.AlignofExpression] = [
  [AlignofExpressionASTSlotBase + 4, false],
];
childSlots[ASTKind.NoexceptExpression] = [
  [NoexceptExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.NewExpression] = [
  [NewExpressionASTSlotBase + 5, false],
  [NewExpressionASTSlotBase + 7, true],
  [NewExpressionASTSlotBase + 8, false],
  [NewExpressionASTSlotBase + 10, false],
];
childSlots[ASTKind.DeleteExpression] = [
  [DeleteExpressionASTSlotBase + 7, false],
];
childSlots[ASTKind.CastExpression] = [
  [CastExpressionASTSlotBase + 4, false],
  [CastExpressionASTSlotBase + 6, false],
];
childSlots[ASTKind.ImplicitCastExpression] = [
  [ImplicitCastExpressionASTSlotBase + 3, false],
];
childSlots[ASTKind.ConstExpression] = [[ConstExpressionASTSlotBase + 3, false]];
childSlots[ASTKind.BinaryExpression] = [
  [BinaryExpressionASTSlotBase + 3, false],
  [BinaryExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.ConditionalExpression] = [
  [ConditionalExpressionASTSlotBase + 3, false],
  [ConditionalExpressionASTSlotBase + 5, false],
  [ConditionalExpressionASTSlotBase + 7, false],
];
childSlots[ASTKind.YieldExpression] = [[YieldExpressionASTSlotBase + 4, false]];
childSlots[ASTKind.ThrowExpression] = [[ThrowExpressionASTSlotBase + 4, false]];
childSlots[ASTKind.AssignmentExpression] = [
  [AssignmentExpressionASTSlotBase + 3, false],
  [AssignmentExpressionASTSlotBase + 5, false],
];
childSlots[ASTKind.TargetExpression] = [];
childSlots[ASTKind.RightExpression] = [];
childSlots[ASTKind.CompoundAssignmentExpression] = [
  [CompoundAssignmentExpressionASTSlotBase + 3, false],
  [CompoundAssignmentExpressionASTSlotBase + 5, false],
  [CompoundAssignmentExpressionASTSlotBase + 6, false],
  [CompoundAssignmentExpressionASTSlotBase + 7, false],
];
childSlots[ASTKind.PackExpansionExpression] = [
  [PackExpansionExpressionASTSlotBase + 3, false],
];
childSlots[ASTKind.DesignatedInitializerClause] = [
  [DesignatedInitializerClauseASTSlotBase + 3, true],
  [DesignatedInitializerClauseASTSlotBase + 4, false],
];
childSlots[ASTKind.TypeTraitExpression] = [
  [TypeTraitExpressionASTSlotBase + 5, true],
];
childSlots[ASTKind.ConditionExpression] = [
  [ConditionExpressionASTSlotBase + 3, true],
  [ConditionExpressionASTSlotBase + 4, true],
  [ConditionExpressionASTSlotBase + 5, false],
  [ConditionExpressionASTSlotBase + 6, false],
];
childSlots[ASTKind.EqualInitializer] = [
  [EqualInitializerASTSlotBase + 4, false],
];
childSlots[ASTKind.BracedInitList] = [[BracedInitListASTSlotBase + 4, true]];
childSlots[ASTKind.ParenInitializer] = [
  [ParenInitializerASTSlotBase + 4, true],
];
childSlots[ASTKind.ThreeWayComparisonExpression] = [
  [ThreeWayComparisonExpressionASTSlotBase + 3, false],
];
childSlots[ASTKind.DefaultGenericAssociation] = [
  [DefaultGenericAssociationASTSlotBase + 3, false],
];
childSlots[ASTKind.TypeGenericAssociation] = [
  [TypeGenericAssociationASTSlotBase + 1, false],
  [TypeGenericAssociationASTSlotBase + 3, false],
];
childSlots[ASTKind.DotDesignator] = [];
childSlots[ASTKind.SubscriptDesignator] = [
  [SubscriptDesignatorASTSlotBase + 2, false],
];
childSlots[ASTKind.TemplateTypeParameter] = [
  [TemplateTypeParameterASTSlotBase + 6, true],
  [TemplateTypeParameterASTSlotBase + 8, false],
  [TemplateTypeParameterASTSlotBase + 13, false],
];
childSlots[ASTKind.NonTypeTemplateParameter] = [
  [NonTypeTemplateParameterASTSlotBase + 4, false],
];
childSlots[ASTKind.TypenameTypeParameter] = [
  [TypenameTypeParameterASTSlotBase + 8, false],
];
childSlots[ASTKind.ConstraintTypeParameter] = [
  [ConstraintTypeParameterASTSlotBase + 4, false],
  [ConstraintTypeParameterASTSlotBase + 8, false],
];
childSlots[ASTKind.TypedefSpecifier] = [];
childSlots[ASTKind.FriendSpecifier] = [];
childSlots[ASTKind.ConstevalSpecifier] = [];
childSlots[ASTKind.ConstinitSpecifier] = [];
childSlots[ASTKind.ConstexprSpecifier] = [];
childSlots[ASTKind.InlineSpecifier] = [];
childSlots[ASTKind.NoreturnSpecifier] = [];
childSlots[ASTKind.StaticSpecifier] = [];
childSlots[ASTKind.ExternSpecifier] = [];
childSlots[ASTKind.RegisterSpecifier] = [];
childSlots[ASTKind.ThreadLocalSpecifier] = [];
childSlots[ASTKind.ThreadSpecifier] = [];
childSlots[ASTKind.MutableSpecifier] = [];
childSlots[ASTKind.VirtualSpecifier] = [];
childSlots[ASTKind.ExplicitSpecifier] = [
  [ExplicitSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.AutoTypeSpecifier] = [];
childSlots[ASTKind.VoidTypeSpecifier] = [];
childSlots[ASTKind.SizeTypeSpecifier] = [];
childSlots[ASTKind.SignTypeSpecifier] = [];
childSlots[ASTKind.BuiltinTypeSpecifier] = [];
childSlots[ASTKind.UnaryBuiltinTypeSpecifier] = [
  [UnaryBuiltinTypeSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.BinaryBuiltinTypeSpecifier] = [
  [BinaryBuiltinTypeSpecifierASTSlotBase + 3, false],
  [BinaryBuiltinTypeSpecifierASTSlotBase + 5, false],
];
childSlots[ASTKind.IntegralTypeSpecifier] = [];
childSlots[ASTKind.FloatingPointTypeSpecifier] = [];
childSlots[ASTKind.ComplexTypeSpecifier] = [];
childSlots[ASTKind.NamedTypeSpecifier] = [
  [NamedTypeSpecifierASTSlotBase + 1, false],
  [NamedTypeSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.AtomicTypeSpecifier] = [
  [AtomicTypeSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.BitIntTypeSpecifier] = [
  [BitIntTypeSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.UnderlyingTypeSpecifier] = [
  [UnderlyingTypeSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.ElaboratedTypeSpecifier] = [
  [ElaboratedTypeSpecifierASTSlotBase + 2, true],
  [ElaboratedTypeSpecifierASTSlotBase + 3, false],
  [ElaboratedTypeSpecifierASTSlotBase + 5, false],
];
childSlots[ASTKind.DecltypeAutoSpecifier] = [];
childSlots[ASTKind.DecltypeSpecifier] = [
  [DecltypeSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.PlaceholderTypeSpecifier] = [
  [PlaceholderTypeSpecifierASTSlotBase + 1, false],
  [PlaceholderTypeSpecifierASTSlotBase + 2, false],
];
childSlots[ASTKind.ConstQualifier] = [];
childSlots[ASTKind.VolatileQualifier] = [];
childSlots[ASTKind.AtomicQualifier] = [];
childSlots[ASTKind.RestrictQualifier] = [];
childSlots[ASTKind.EnumSpecifier] = [
  [EnumSpecifierASTSlotBase + 3, true],
  [EnumSpecifierASTSlotBase + 4, false],
  [EnumSpecifierASTSlotBase + 5, false],
  [EnumSpecifierASTSlotBase + 7, true],
  [EnumSpecifierASTSlotBase + 9, true],
];
childSlots[ASTKind.ClassSpecifier] = [
  [ClassSpecifierASTSlotBase + 2, true],
  [ClassSpecifierASTSlotBase + 3, false],
  [ClassSpecifierASTSlotBase + 4, false],
  [ClassSpecifierASTSlotBase + 7, true],
  [ClassSpecifierASTSlotBase + 9, true],
];
childSlots[ASTKind.TypenameSpecifier] = [
  [TypenameSpecifierASTSlotBase + 2, false],
  [TypenameSpecifierASTSlotBase + 4, false],
];
childSlots[ASTKind.SplicerTypeSpecifier] = [
  [SplicerTypeSpecifierASTSlotBase + 2, false],
];
childSlots[ASTKind.PointerOperator] = [
  [PointerOperatorASTSlotBase + 2, true],
  [PointerOperatorASTSlotBase + 3, true],
];
childSlots[ASTKind.ReferenceOperator] = [
  [ReferenceOperatorASTSlotBase + 2, true],
];
childSlots[ASTKind.PtrToMemberOperator] = [
  [PtrToMemberOperatorASTSlotBase + 1, false],
  [PtrToMemberOperatorASTSlotBase + 3, true],
  [PtrToMemberOperatorASTSlotBase + 4, true],
];
childSlots[ASTKind.BitfieldDeclarator] = [
  [BitfieldDeclaratorASTSlotBase + 1, false],
  [BitfieldDeclaratorASTSlotBase + 3, false],
];
childSlots[ASTKind.ParameterPack] = [[ParameterPackASTSlotBase + 2, false]];
childSlots[ASTKind.IdDeclarator] = [
  [IdDeclaratorASTSlotBase + 1, false],
  [IdDeclaratorASTSlotBase + 3, false],
  [IdDeclaratorASTSlotBase + 4, true],
];
childSlots[ASTKind.NestedDeclarator] = [
  [NestedDeclaratorASTSlotBase + 2, false],
];
childSlots[ASTKind.FunctionDeclaratorChunk] = [
  [FunctionDeclaratorChunkASTSlotBase + 2, false],
  [FunctionDeclaratorChunkASTSlotBase + 4, true],
  [FunctionDeclaratorChunkASTSlotBase + 6, false],
  [FunctionDeclaratorChunkASTSlotBase + 7, true],
  [FunctionDeclaratorChunkASTSlotBase + 8, false],
];
childSlots[ASTKind.ArrayDeclaratorChunk] = [
  [ArrayDeclaratorChunkASTSlotBase + 2, true],
  [ArrayDeclaratorChunkASTSlotBase + 3, false],
  [ArrayDeclaratorChunkASTSlotBase + 5, true],
];
childSlots[ASTKind.NameId] = [];
childSlots[ASTKind.DestructorId] = [[DestructorIdASTSlotBase + 2, false]];
childSlots[ASTKind.DecltypeId] = [[DecltypeIdASTSlotBase + 1, false]];
childSlots[ASTKind.OperatorFunctionId] = [];
childSlots[ASTKind.LiteralOperatorId] = [];
childSlots[ASTKind.ConversionFunctionId] = [
  [ConversionFunctionIdASTSlotBase + 2, false],
];
childSlots[ASTKind.SimpleTemplateId] = [
  [SimpleTemplateIdASTSlotBase + 3, true],
];
childSlots[ASTKind.LiteralOperatorTemplateId] = [
  [LiteralOperatorTemplateIdASTSlotBase + 1, false],
  [LiteralOperatorTemplateIdASTSlotBase + 3, true],
];
childSlots[ASTKind.OperatorFunctionTemplateId] = [
  [OperatorFunctionTemplateIdASTSlotBase + 1, false],
  [OperatorFunctionTemplateIdASTSlotBase + 3, true],
];
childSlots[ASTKind.GlobalNestedNameSpecifier] = [];
childSlots[ASTKind.SimpleNestedNameSpecifier] = [
  [SimpleNestedNameSpecifierASTSlotBase + 2, false],
];
childSlots[ASTKind.DecltypeNestedNameSpecifier] = [
  [DecltypeNestedNameSpecifierASTSlotBase + 2, false],
];
childSlots[ASTKind.TemplateNestedNameSpecifier] = [
  [TemplateNestedNameSpecifierASTSlotBase + 2, false],
  [TemplateNestedNameSpecifierASTSlotBase + 4, false],
];
childSlots[ASTKind.DefaultFunctionBody] = [];
childSlots[ASTKind.CompoundStatementFunctionBody] = [
  [CompoundStatementFunctionBodyASTSlotBase + 2, true],
  [CompoundStatementFunctionBodyASTSlotBase + 3, false],
];
childSlots[ASTKind.TryStatementFunctionBody] = [
  [TryStatementFunctionBodyASTSlotBase + 3, true],
  [TryStatementFunctionBodyASTSlotBase + 4, false],
  [TryStatementFunctionBodyASTSlotBase + 5, true],
];
childSlots[ASTKind.DeleteFunctionBody] = [];
childSlots[ASTKind.TypeTemplateArgument] = [
  [TypeTemplateArgumentASTSlotBase + 1, false],
];
childSlots[ASTKind.ExpressionTemplateArgument] = [
  [ExpressionTemplateArgumentASTSlotBase + 1, false],
];
childSlots[ASTKind.ThrowExceptionSpecifier] = [];
childSlots[ASTKind.NoexceptSpecifier] = [
  [NoexceptSpecifierASTSlotBase + 3, false],
];
childSlots[ASTKind.SimpleRequirement] = [
  [SimpleRequirementASTSlotBase + 1, false],
];
childSlots[ASTKind.CompoundRequirement] = [
  [CompoundRequirementASTSlotBase + 2, false],
  [CompoundRequirementASTSlotBase + 6, false],
];
childSlots[ASTKind.TypeRequirement] = [
  [TypeRequirementASTSlotBase + 2, false],
  [TypeRequirementASTSlotBase + 4, false],
];
childSlots[ASTKind.NestedRequirement] = [
  [NestedRequirementASTSlotBase + 2, false],
];
childSlots[ASTKind.NewParenInitializer] = [
  [NewParenInitializerASTSlotBase + 2, true],
];
childSlots[ASTKind.NewBracedInitializer] = [
  [NewBracedInitializerASTSlotBase + 1, false],
];
childSlots[ASTKind.ParenMemInitializer] = [
  [ParenMemInitializerASTSlotBase + 3, false],
  [ParenMemInitializerASTSlotBase + 4, false],
  [ParenMemInitializerASTSlotBase + 6, true],
];
childSlots[ASTKind.BracedMemInitializer] = [
  [BracedMemInitializerASTSlotBase + 3, false],
  [BracedMemInitializerASTSlotBase + 4, false],
  [BracedMemInitializerASTSlotBase + 5, false],
];
childSlots[ASTKind.ThisLambdaCapture] = [
  [ThisLambdaCaptureASTSlotBase + 2, false],
];
childSlots[ASTKind.DerefThisLambdaCapture] = [];
childSlots[ASTKind.SimpleLambdaCapture] = [
  [SimpleLambdaCaptureASTSlotBase + 4, false],
];
childSlots[ASTKind.RefLambdaCapture] = [
  [RefLambdaCaptureASTSlotBase + 5, false],
];
childSlots[ASTKind.RefInitLambdaCapture] = [
  [RefInitLambdaCaptureASTSlotBase + 4, false],
];
childSlots[ASTKind.InitLambdaCapture] = [
  [InitLambdaCaptureASTSlotBase + 3, false],
];
childSlots[ASTKind.EllipsisExceptionDeclaration] = [];
childSlots[ASTKind.TypeExceptionDeclaration] = [
  [TypeExceptionDeclarationASTSlotBase + 1, true],
  [TypeExceptionDeclarationASTSlotBase + 2, true],
  [TypeExceptionDeclarationASTSlotBase + 3, false],
];
childSlots[ASTKind.CxxAttribute] = [
  [CxxAttributeASTSlotBase + 4, false],
  [CxxAttributeASTSlotBase + 5, true],
];
childSlots[ASTKind.GccAttribute] = [[GccAttributeASTSlotBase + 5, true]];
childSlots[ASTKind.AlignasAttribute] = [
  [AlignasAttributeASTSlotBase + 4, false],
];
childSlots[ASTKind.AlignasTypeAttribute] = [
  [AlignasTypeAttributeASTSlotBase + 4, false],
];
childSlots[ASTKind.AsmAttribute] = [];
childSlots[ASTKind.ScopedAttributeToken] = [];
childSlots[ASTKind.SimpleAttributeToken] = [];

export function* children(node: AST): Iterable<AST> {
  for (const [slot, isList] of childSlots[node.kind] ?? []) {
    const value = cxx.readAST(node.handle, slot);
    if (!isList) {
      const child = astOf(value, node.modelOwner);
      if (child) yield child;
      continue;
    }
    for (const child of listOf(node.modelOwner, value, (item: any) =>
      astOf(item, node.modelOwner),
    ))
      if (child) yield child;
  }
}

export function modelOf(owner: ModelOwner): {
  ast: UnitAST;
  globalScope: ScopeSymbol;
} {
  const unit = owner.getUnitHandle();
  return {
    ast: astOf(cxx.getUnitAST(unit), owner),
    globalScope: symbolOf(cxx.getGlobalScope(unit), owner),
  };
}
export abstract class ASTVisitor<Context, Result> {
  abstract visitTranslationUnit(
    node: TranslationUnitAST,
    context: Context,
  ): Result;
  abstract visitModuleUnit(node: ModuleUnitAST, context: Context): Result;
  abstract visitSimpleDeclaration(
    node: SimpleDeclarationAST,
    context: Context,
  ): Result;
  abstract visitAsmDeclaration(
    node: AsmDeclarationAST,
    context: Context,
  ): Result;
  abstract visitNamespaceAliasDefinition(
    node: NamespaceAliasDefinitionAST,
    context: Context,
  ): Result;
  abstract visitUsingDeclaration(
    node: UsingDeclarationAST,
    context: Context,
  ): Result;
  abstract visitUsingEnumDeclaration(
    node: UsingEnumDeclarationAST,
    context: Context,
  ): Result;
  abstract visitUsingDirective(
    node: UsingDirectiveAST,
    context: Context,
  ): Result;
  abstract visitStaticAssertDeclaration(
    node: StaticAssertDeclarationAST,
    context: Context,
  ): Result;
  abstract visitAliasDeclaration(
    node: AliasDeclarationAST,
    context: Context,
  ): Result;
  abstract visitOpaqueEnumDeclaration(
    node: OpaqueEnumDeclarationAST,
    context: Context,
  ): Result;
  abstract visitFunctionDefinition(
    node: FunctionDefinitionAST,
    context: Context,
  ): Result;
  abstract visitTemplateDeclaration(
    node: TemplateDeclarationAST,
    context: Context,
  ): Result;
  abstract visitConceptDefinition(
    node: ConceptDefinitionAST,
    context: Context,
  ): Result;
  abstract visitDeductionGuide(
    node: DeductionGuideAST,
    context: Context,
  ): Result;
  abstract visitExplicitInstantiation(
    node: ExplicitInstantiationAST,
    context: Context,
  ): Result;
  abstract visitExportDeclaration(
    node: ExportDeclarationAST,
    context: Context,
  ): Result;
  abstract visitExportCompoundDeclaration(
    node: ExportCompoundDeclarationAST,
    context: Context,
  ): Result;
  abstract visitLinkageSpecification(
    node: LinkageSpecificationAST,
    context: Context,
  ): Result;
  abstract visitNamespaceDefinition(
    node: NamespaceDefinitionAST,
    context: Context,
  ): Result;
  abstract visitEmptyDeclaration(
    node: EmptyDeclarationAST,
    context: Context,
  ): Result;
  abstract visitAttributeDeclaration(
    node: AttributeDeclarationAST,
    context: Context,
  ): Result;
  abstract visitModuleImportDeclaration(
    node: ModuleImportDeclarationAST,
    context: Context,
  ): Result;
  abstract visitParameterDeclaration(
    node: ParameterDeclarationAST,
    context: Context,
  ): Result;
  abstract visitAccessDeclaration(
    node: AccessDeclarationAST,
    context: Context,
  ): Result;
  abstract visitForRangeDeclaration(
    node: ForRangeDeclarationAST,
    context: Context,
  ): Result;
  abstract visitStructuredBindingDeclaration(
    node: StructuredBindingDeclarationAST,
    context: Context,
  ): Result;
  abstract visitAsmOperand(node: AsmOperandAST, context: Context): Result;
  abstract visitAsmQualifier(node: AsmQualifierAST, context: Context): Result;
  abstract visitAsmClobber(node: AsmClobberAST, context: Context): Result;
  abstract visitAsmGotoLabel(node: AsmGotoLabelAST, context: Context): Result;
  abstract visitSplicer(node: SplicerAST, context: Context): Result;
  abstract visitGlobalModuleFragment(
    node: GlobalModuleFragmentAST,
    context: Context,
  ): Result;
  abstract visitPrivateModuleFragment(
    node: PrivateModuleFragmentAST,
    context: Context,
  ): Result;
  abstract visitModuleDeclaration(
    node: ModuleDeclarationAST,
    context: Context,
  ): Result;
  abstract visitModuleName(node: ModuleNameAST, context: Context): Result;
  abstract visitModuleQualifier(
    node: ModuleQualifierAST,
    context: Context,
  ): Result;
  abstract visitModulePartition(
    node: ModulePartitionAST,
    context: Context,
  ): Result;
  abstract visitImportName(node: ImportNameAST, context: Context): Result;
  abstract visitInitDeclarator(
    node: InitDeclaratorAST,
    context: Context,
  ): Result;
  abstract visitDeclarator(node: DeclaratorAST, context: Context): Result;
  abstract visitUsingDeclarator(
    node: UsingDeclaratorAST,
    context: Context,
  ): Result;
  abstract visitEnumerator(node: EnumeratorAST, context: Context): Result;
  abstract visitTypeId(node: TypeIdAST, context: Context): Result;
  abstract visitHandler(node: HandlerAST, context: Context): Result;
  abstract visitBaseSpecifier(node: BaseSpecifierAST, context: Context): Result;
  abstract visitRequiresClause(
    node: RequiresClauseAST,
    context: Context,
  ): Result;
  abstract visitParameterDeclarationClause(
    node: ParameterDeclarationClauseAST,
    context: Context,
  ): Result;
  abstract visitTrailingReturnType(
    node: TrailingReturnTypeAST,
    context: Context,
  ): Result;
  abstract visitLambdaSpecifier(
    node: LambdaSpecifierAST,
    context: Context,
  ): Result;
  abstract visitTypeConstraint(
    node: TypeConstraintAST,
    context: Context,
  ): Result;
  abstract visitAttributeArgumentClause(
    node: AttributeArgumentClauseAST,
    context: Context,
  ): Result;
  abstract visitAttribute(node: AttributeAST, context: Context): Result;
  abstract visitAttributeUsingPrefix(
    node: AttributeUsingPrefixAST,
    context: Context,
  ): Result;
  abstract visitNewPlacement(node: NewPlacementAST, context: Context): Result;
  abstract visitNestedNamespaceSpecifier(
    node: NestedNamespaceSpecifierAST,
    context: Context,
  ): Result;
  abstract visitLabeledStatement(
    node: LabeledStatementAST,
    context: Context,
  ): Result;
  abstract visitCaseStatement(node: CaseStatementAST, context: Context): Result;
  abstract visitDefaultStatement(
    node: DefaultStatementAST,
    context: Context,
  ): Result;
  abstract visitExpressionStatement(
    node: ExpressionStatementAST,
    context: Context,
  ): Result;
  abstract visitCompoundStatement(
    node: CompoundStatementAST,
    context: Context,
  ): Result;
  abstract visitIfStatement(node: IfStatementAST, context: Context): Result;
  abstract visitConstevalIfStatement(
    node: ConstevalIfStatementAST,
    context: Context,
  ): Result;
  abstract visitSwitchStatement(
    node: SwitchStatementAST,
    context: Context,
  ): Result;
  abstract visitWhileStatement(
    node: WhileStatementAST,
    context: Context,
  ): Result;
  abstract visitDoStatement(node: DoStatementAST, context: Context): Result;
  abstract visitForRangeStatement(
    node: ForRangeStatementAST,
    context: Context,
  ): Result;
  abstract visitForStatement(node: ForStatementAST, context: Context): Result;
  abstract visitBreakStatement(
    node: BreakStatementAST,
    context: Context,
  ): Result;
  abstract visitContinueStatement(
    node: ContinueStatementAST,
    context: Context,
  ): Result;
  abstract visitReturnStatement(
    node: ReturnStatementAST,
    context: Context,
  ): Result;
  abstract visitCoroutineReturnStatement(
    node: CoroutineReturnStatementAST,
    context: Context,
  ): Result;
  abstract visitGotoStatement(node: GotoStatementAST, context: Context): Result;
  abstract visitDeclarationStatement(
    node: DeclarationStatementAST,
    context: Context,
  ): Result;
  abstract visitTryBlockStatement(
    node: TryBlockStatementAST,
    context: Context,
  ): Result;
  abstract visitCharLiteralExpression(
    node: CharLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitBoolLiteralExpression(
    node: BoolLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitIntLiteralExpression(
    node: IntLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitFloatLiteralExpression(
    node: FloatLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitNullptrLiteralExpression(
    node: NullptrLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitStringLiteralExpression(
    node: StringLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitUserDefinedStringLiteralExpression(
    node: UserDefinedStringLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitObjectLiteralExpression(
    node: ObjectLiteralExpressionAST,
    context: Context,
  ): Result;
  abstract visitThisExpression(
    node: ThisExpressionAST,
    context: Context,
  ): Result;
  abstract visitPackIndexExpression(
    node: PackIndexExpressionAST,
    context: Context,
  ): Result;
  abstract visitGenericSelectionExpression(
    node: GenericSelectionExpressionAST,
    context: Context,
  ): Result;
  abstract visitNestedStatementExpression(
    node: NestedStatementExpressionAST,
    context: Context,
  ): Result;
  abstract visitDefaultInitializerExpression(
    node: DefaultInitializerExpressionAST,
    context: Context,
  ): Result;
  abstract visitNestedExpression(
    node: NestedExpressionAST,
    context: Context,
  ): Result;
  abstract visitIdExpression(node: IdExpressionAST, context: Context): Result;
  abstract visitLambdaExpression(
    node: LambdaExpressionAST,
    context: Context,
  ): Result;
  abstract visitFoldExpression(
    node: FoldExpressionAST,
    context: Context,
  ): Result;
  abstract visitRightFoldExpression(
    node: RightFoldExpressionAST,
    context: Context,
  ): Result;
  abstract visitLeftFoldExpression(
    node: LeftFoldExpressionAST,
    context: Context,
  ): Result;
  abstract visitRequiresExpression(
    node: RequiresExpressionAST,
    context: Context,
  ): Result;
  abstract visitVaArgExpression(
    node: VaArgExpressionAST,
    context: Context,
  ): Result;
  abstract visitSubscriptExpression(
    node: SubscriptExpressionAST,
    context: Context,
  ): Result;
  abstract visitCallExpression(
    node: CallExpressionAST,
    context: Context,
  ): Result;
  abstract visitTypeConstruction(
    node: TypeConstructionAST,
    context: Context,
  ): Result;
  abstract visitBracedTypeConstruction(
    node: BracedTypeConstructionAST,
    context: Context,
  ): Result;
  abstract visitSpliceMemberExpression(
    node: SpliceMemberExpressionAST,
    context: Context,
  ): Result;
  abstract visitMemberExpression(
    node: MemberExpressionAST,
    context: Context,
  ): Result;
  abstract visitPostIncrExpression(
    node: PostIncrExpressionAST,
    context: Context,
  ): Result;
  abstract visitCppCastExpression(
    node: CppCastExpressionAST,
    context: Context,
  ): Result;
  abstract visitBuiltinBitCastExpression(
    node: BuiltinBitCastExpressionAST,
    context: Context,
  ): Result;
  abstract visitBuiltinOffsetofExpression(
    node: BuiltinOffsetofExpressionAST,
    context: Context,
  ): Result;
  abstract visitTypeidExpression(
    node: TypeidExpressionAST,
    context: Context,
  ): Result;
  abstract visitTypeidOfTypeExpression(
    node: TypeidOfTypeExpressionAST,
    context: Context,
  ): Result;
  abstract visitSpliceExpression(
    node: SpliceExpressionAST,
    context: Context,
  ): Result;
  abstract visitGlobalScopeReflectExpression(
    node: GlobalScopeReflectExpressionAST,
    context: Context,
  ): Result;
  abstract visitNamespaceReflectExpression(
    node: NamespaceReflectExpressionAST,
    context: Context,
  ): Result;
  abstract visitTypeIdReflectExpression(
    node: TypeIdReflectExpressionAST,
    context: Context,
  ): Result;
  abstract visitReflectExpression(
    node: ReflectExpressionAST,
    context: Context,
  ): Result;
  abstract visitLabelAddressExpression(
    node: LabelAddressExpressionAST,
    context: Context,
  ): Result;
  abstract visitUnaryExpression(
    node: UnaryExpressionAST,
    context: Context,
  ): Result;
  abstract visitAwaitExpression(
    node: AwaitExpressionAST,
    context: Context,
  ): Result;
  abstract visitSizeofExpression(
    node: SizeofExpressionAST,
    context: Context,
  ): Result;
  abstract visitSizeofTypeExpression(
    node: SizeofTypeExpressionAST,
    context: Context,
  ): Result;
  abstract visitSizeofPackExpression(
    node: SizeofPackExpressionAST,
    context: Context,
  ): Result;
  abstract visitAlignofTypeExpression(
    node: AlignofTypeExpressionAST,
    context: Context,
  ): Result;
  abstract visitAlignofExpression(
    node: AlignofExpressionAST,
    context: Context,
  ): Result;
  abstract visitNoexceptExpression(
    node: NoexceptExpressionAST,
    context: Context,
  ): Result;
  abstract visitNewExpression(node: NewExpressionAST, context: Context): Result;
  abstract visitDeleteExpression(
    node: DeleteExpressionAST,
    context: Context,
  ): Result;
  abstract visitCastExpression(
    node: CastExpressionAST,
    context: Context,
  ): Result;
  abstract visitImplicitCastExpression(
    node: ImplicitCastExpressionAST,
    context: Context,
  ): Result;
  abstract visitConstExpression(
    node: ConstExpressionAST,
    context: Context,
  ): Result;
  abstract visitBinaryExpression(
    node: BinaryExpressionAST,
    context: Context,
  ): Result;
  abstract visitConditionalExpression(
    node: ConditionalExpressionAST,
    context: Context,
  ): Result;
  abstract visitYieldExpression(
    node: YieldExpressionAST,
    context: Context,
  ): Result;
  abstract visitThrowExpression(
    node: ThrowExpressionAST,
    context: Context,
  ): Result;
  abstract visitAssignmentExpression(
    node: AssignmentExpressionAST,
    context: Context,
  ): Result;
  abstract visitTargetExpression(
    node: TargetExpressionAST,
    context: Context,
  ): Result;
  abstract visitRightExpression(
    node: RightExpressionAST,
    context: Context,
  ): Result;
  abstract visitCompoundAssignmentExpression(
    node: CompoundAssignmentExpressionAST,
    context: Context,
  ): Result;
  abstract visitPackExpansionExpression(
    node: PackExpansionExpressionAST,
    context: Context,
  ): Result;
  abstract visitDesignatedInitializerClause(
    node: DesignatedInitializerClauseAST,
    context: Context,
  ): Result;
  abstract visitTypeTraitExpression(
    node: TypeTraitExpressionAST,
    context: Context,
  ): Result;
  abstract visitConditionExpression(
    node: ConditionExpressionAST,
    context: Context,
  ): Result;
  abstract visitEqualInitializer(
    node: EqualInitializerAST,
    context: Context,
  ): Result;
  abstract visitBracedInitList(
    node: BracedInitListAST,
    context: Context,
  ): Result;
  abstract visitParenInitializer(
    node: ParenInitializerAST,
    context: Context,
  ): Result;
  abstract visitThreeWayComparisonExpression(
    node: ThreeWayComparisonExpressionAST,
    context: Context,
  ): Result;
  abstract visitDefaultGenericAssociation(
    node: DefaultGenericAssociationAST,
    context: Context,
  ): Result;
  abstract visitTypeGenericAssociation(
    node: TypeGenericAssociationAST,
    context: Context,
  ): Result;
  abstract visitDotDesignator(node: DotDesignatorAST, context: Context): Result;
  abstract visitSubscriptDesignator(
    node: SubscriptDesignatorAST,
    context: Context,
  ): Result;
  abstract visitTemplateTypeParameter(
    node: TemplateTypeParameterAST,
    context: Context,
  ): Result;
  abstract visitNonTypeTemplateParameter(
    node: NonTypeTemplateParameterAST,
    context: Context,
  ): Result;
  abstract visitTypenameTypeParameter(
    node: TypenameTypeParameterAST,
    context: Context,
  ): Result;
  abstract visitConstraintTypeParameter(
    node: ConstraintTypeParameterAST,
    context: Context,
  ): Result;
  abstract visitTypedefSpecifier(
    node: TypedefSpecifierAST,
    context: Context,
  ): Result;
  abstract visitFriendSpecifier(
    node: FriendSpecifierAST,
    context: Context,
  ): Result;
  abstract visitConstevalSpecifier(
    node: ConstevalSpecifierAST,
    context: Context,
  ): Result;
  abstract visitConstinitSpecifier(
    node: ConstinitSpecifierAST,
    context: Context,
  ): Result;
  abstract visitConstexprSpecifier(
    node: ConstexprSpecifierAST,
    context: Context,
  ): Result;
  abstract visitInlineSpecifier(
    node: InlineSpecifierAST,
    context: Context,
  ): Result;
  abstract visitNoreturnSpecifier(
    node: NoreturnSpecifierAST,
    context: Context,
  ): Result;
  abstract visitStaticSpecifier(
    node: StaticSpecifierAST,
    context: Context,
  ): Result;
  abstract visitExternSpecifier(
    node: ExternSpecifierAST,
    context: Context,
  ): Result;
  abstract visitRegisterSpecifier(
    node: RegisterSpecifierAST,
    context: Context,
  ): Result;
  abstract visitThreadLocalSpecifier(
    node: ThreadLocalSpecifierAST,
    context: Context,
  ): Result;
  abstract visitThreadSpecifier(
    node: ThreadSpecifierAST,
    context: Context,
  ): Result;
  abstract visitMutableSpecifier(
    node: MutableSpecifierAST,
    context: Context,
  ): Result;
  abstract visitVirtualSpecifier(
    node: VirtualSpecifierAST,
    context: Context,
  ): Result;
  abstract visitExplicitSpecifier(
    node: ExplicitSpecifierAST,
    context: Context,
  ): Result;
  abstract visitAutoTypeSpecifier(
    node: AutoTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitVoidTypeSpecifier(
    node: VoidTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitSizeTypeSpecifier(
    node: SizeTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitSignTypeSpecifier(
    node: SignTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitBuiltinTypeSpecifier(
    node: BuiltinTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitUnaryBuiltinTypeSpecifier(
    node: UnaryBuiltinTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitBinaryBuiltinTypeSpecifier(
    node: BinaryBuiltinTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitIntegralTypeSpecifier(
    node: IntegralTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitFloatingPointTypeSpecifier(
    node: FloatingPointTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitComplexTypeSpecifier(
    node: ComplexTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitNamedTypeSpecifier(
    node: NamedTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitAtomicTypeSpecifier(
    node: AtomicTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitBitIntTypeSpecifier(
    node: BitIntTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitUnderlyingTypeSpecifier(
    node: UnderlyingTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitElaboratedTypeSpecifier(
    node: ElaboratedTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitDecltypeAutoSpecifier(
    node: DecltypeAutoSpecifierAST,
    context: Context,
  ): Result;
  abstract visitDecltypeSpecifier(
    node: DecltypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitPlaceholderTypeSpecifier(
    node: PlaceholderTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitConstQualifier(
    node: ConstQualifierAST,
    context: Context,
  ): Result;
  abstract visitVolatileQualifier(
    node: VolatileQualifierAST,
    context: Context,
  ): Result;
  abstract visitAtomicQualifier(
    node: AtomicQualifierAST,
    context: Context,
  ): Result;
  abstract visitRestrictQualifier(
    node: RestrictQualifierAST,
    context: Context,
  ): Result;
  abstract visitEnumSpecifier(node: EnumSpecifierAST, context: Context): Result;
  abstract visitClassSpecifier(
    node: ClassSpecifierAST,
    context: Context,
  ): Result;
  abstract visitTypenameSpecifier(
    node: TypenameSpecifierAST,
    context: Context,
  ): Result;
  abstract visitSplicerTypeSpecifier(
    node: SplicerTypeSpecifierAST,
    context: Context,
  ): Result;
  abstract visitPointerOperator(
    node: PointerOperatorAST,
    context: Context,
  ): Result;
  abstract visitReferenceOperator(
    node: ReferenceOperatorAST,
    context: Context,
  ): Result;
  abstract visitPtrToMemberOperator(
    node: PtrToMemberOperatorAST,
    context: Context,
  ): Result;
  abstract visitBitfieldDeclarator(
    node: BitfieldDeclaratorAST,
    context: Context,
  ): Result;
  abstract visitParameterPack(node: ParameterPackAST, context: Context): Result;
  abstract visitIdDeclarator(node: IdDeclaratorAST, context: Context): Result;
  abstract visitNestedDeclarator(
    node: NestedDeclaratorAST,
    context: Context,
  ): Result;
  abstract visitFunctionDeclaratorChunk(
    node: FunctionDeclaratorChunkAST,
    context: Context,
  ): Result;
  abstract visitArrayDeclaratorChunk(
    node: ArrayDeclaratorChunkAST,
    context: Context,
  ): Result;
  abstract visitNameId(node: NameIdAST, context: Context): Result;
  abstract visitDestructorId(node: DestructorIdAST, context: Context): Result;
  abstract visitDecltypeId(node: DecltypeIdAST, context: Context): Result;
  abstract visitOperatorFunctionId(
    node: OperatorFunctionIdAST,
    context: Context,
  ): Result;
  abstract visitLiteralOperatorId(
    node: LiteralOperatorIdAST,
    context: Context,
  ): Result;
  abstract visitConversionFunctionId(
    node: ConversionFunctionIdAST,
    context: Context,
  ): Result;
  abstract visitSimpleTemplateId(
    node: SimpleTemplateIdAST,
    context: Context,
  ): Result;
  abstract visitLiteralOperatorTemplateId(
    node: LiteralOperatorTemplateIdAST,
    context: Context,
  ): Result;
  abstract visitOperatorFunctionTemplateId(
    node: OperatorFunctionTemplateIdAST,
    context: Context,
  ): Result;
  abstract visitGlobalNestedNameSpecifier(
    node: GlobalNestedNameSpecifierAST,
    context: Context,
  ): Result;
  abstract visitSimpleNestedNameSpecifier(
    node: SimpleNestedNameSpecifierAST,
    context: Context,
  ): Result;
  abstract visitDecltypeNestedNameSpecifier(
    node: DecltypeNestedNameSpecifierAST,
    context: Context,
  ): Result;
  abstract visitTemplateNestedNameSpecifier(
    node: TemplateNestedNameSpecifierAST,
    context: Context,
  ): Result;
  abstract visitDefaultFunctionBody(
    node: DefaultFunctionBodyAST,
    context: Context,
  ): Result;
  abstract visitCompoundStatementFunctionBody(
    node: CompoundStatementFunctionBodyAST,
    context: Context,
  ): Result;
  abstract visitTryStatementFunctionBody(
    node: TryStatementFunctionBodyAST,
    context: Context,
  ): Result;
  abstract visitDeleteFunctionBody(
    node: DeleteFunctionBodyAST,
    context: Context,
  ): Result;
  abstract visitTypeTemplateArgument(
    node: TypeTemplateArgumentAST,
    context: Context,
  ): Result;
  abstract visitExpressionTemplateArgument(
    node: ExpressionTemplateArgumentAST,
    context: Context,
  ): Result;
  abstract visitThrowExceptionSpecifier(
    node: ThrowExceptionSpecifierAST,
    context: Context,
  ): Result;
  abstract visitNoexceptSpecifier(
    node: NoexceptSpecifierAST,
    context: Context,
  ): Result;
  abstract visitSimpleRequirement(
    node: SimpleRequirementAST,
    context: Context,
  ): Result;
  abstract visitCompoundRequirement(
    node: CompoundRequirementAST,
    context: Context,
  ): Result;
  abstract visitTypeRequirement(
    node: TypeRequirementAST,
    context: Context,
  ): Result;
  abstract visitNestedRequirement(
    node: NestedRequirementAST,
    context: Context,
  ): Result;
  abstract visitNewParenInitializer(
    node: NewParenInitializerAST,
    context: Context,
  ): Result;
  abstract visitNewBracedInitializer(
    node: NewBracedInitializerAST,
    context: Context,
  ): Result;
  abstract visitParenMemInitializer(
    node: ParenMemInitializerAST,
    context: Context,
  ): Result;
  abstract visitBracedMemInitializer(
    node: BracedMemInitializerAST,
    context: Context,
  ): Result;
  abstract visitThisLambdaCapture(
    node: ThisLambdaCaptureAST,
    context: Context,
  ): Result;
  abstract visitDerefThisLambdaCapture(
    node: DerefThisLambdaCaptureAST,
    context: Context,
  ): Result;
  abstract visitSimpleLambdaCapture(
    node: SimpleLambdaCaptureAST,
    context: Context,
  ): Result;
  abstract visitRefLambdaCapture(
    node: RefLambdaCaptureAST,
    context: Context,
  ): Result;
  abstract visitRefInitLambdaCapture(
    node: RefInitLambdaCaptureAST,
    context: Context,
  ): Result;
  abstract visitInitLambdaCapture(
    node: InitLambdaCaptureAST,
    context: Context,
  ): Result;
  abstract visitEllipsisExceptionDeclaration(
    node: EllipsisExceptionDeclarationAST,
    context: Context,
  ): Result;
  abstract visitTypeExceptionDeclaration(
    node: TypeExceptionDeclarationAST,
    context: Context,
  ): Result;
  abstract visitCxxAttribute(node: CxxAttributeAST, context: Context): Result;
  abstract visitGccAttribute(node: GccAttributeAST, context: Context): Result;
  abstract visitAlignasAttribute(
    node: AlignasAttributeAST,
    context: Context,
  ): Result;
  abstract visitAlignasTypeAttribute(
    node: AlignasTypeAttributeAST,
    context: Context,
  ): Result;
  abstract visitAsmAttribute(node: AsmAttributeAST, context: Context): Result;
  abstract visitScopedAttributeToken(
    node: ScopedAttributeTokenAST,
    context: Context,
  ): Result;
  abstract visitSimpleAttributeToken(
    node: SimpleAttributeTokenAST,
    context: Context,
  ): Result;
}

export class RecursiveASTVisitor<Context> extends ASTVisitor<Context, void> {
  accept(node: AST | undefined, context: Context): void {
    node?.accept(this, context);
  }

  visitChildren(node: AST, context: Context): void {
    for (const child of children(node)) this.accept(child, context);
  }

  visitTranslationUnit(node: TranslationUnitAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitModuleUnit(node: ModuleUnitAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSimpleDeclaration(node: SimpleDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAsmDeclaration(node: AsmDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNamespaceAliasDefinition(
    node: NamespaceAliasDefinitionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitUsingDeclaration(node: UsingDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitUsingEnumDeclaration(
    node: UsingEnumDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitUsingDirective(node: UsingDirectiveAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitStaticAssertDeclaration(
    node: StaticAssertDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitAliasDeclaration(node: AliasDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitOpaqueEnumDeclaration(
    node: OpaqueEnumDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitFunctionDefinition(node: FunctionDefinitionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitTemplateDeclaration(
    node: TemplateDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitConceptDefinition(node: ConceptDefinitionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDeductionGuide(node: DeductionGuideAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitExplicitInstantiation(
    node: ExplicitInstantiationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitExportDeclaration(node: ExportDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitExportCompoundDeclaration(
    node: ExportCompoundDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitLinkageSpecification(
    node: LinkageSpecificationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNamespaceDefinition(
    node: NamespaceDefinitionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitEmptyDeclaration(node: EmptyDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAttributeDeclaration(
    node: AttributeDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitModuleImportDeclaration(
    node: ModuleImportDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitParameterDeclaration(
    node: ParameterDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitAccessDeclaration(node: AccessDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitForRangeDeclaration(
    node: ForRangeDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitStructuredBindingDeclaration(
    node: StructuredBindingDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitAsmOperand(node: AsmOperandAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAsmQualifier(node: AsmQualifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAsmClobber(node: AsmClobberAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAsmGotoLabel(node: AsmGotoLabelAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSplicer(node: SplicerAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitGlobalModuleFragment(
    node: GlobalModuleFragmentAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitPrivateModuleFragment(
    node: PrivateModuleFragmentAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitModuleDeclaration(node: ModuleDeclarationAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitModuleName(node: ModuleNameAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitModuleQualifier(node: ModuleQualifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitModulePartition(node: ModulePartitionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitImportName(node: ImportNameAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitInitDeclarator(node: InitDeclaratorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDeclarator(node: DeclaratorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitUsingDeclarator(node: UsingDeclaratorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitEnumerator(node: EnumeratorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitTypeId(node: TypeIdAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitHandler(node: HandlerAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitBaseSpecifier(node: BaseSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitRequiresClause(node: RequiresClauseAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitParameterDeclarationClause(
    node: ParameterDeclarationClauseAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTrailingReturnType(node: TrailingReturnTypeAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitLambdaSpecifier(node: LambdaSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitTypeConstraint(node: TypeConstraintAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAttributeArgumentClause(
    node: AttributeArgumentClauseAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitAttribute(node: AttributeAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAttributeUsingPrefix(
    node: AttributeUsingPrefixAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNewPlacement(node: NewPlacementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNestedNamespaceSpecifier(
    node: NestedNamespaceSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitLabeledStatement(node: LabeledStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitCaseStatement(node: CaseStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDefaultStatement(node: DefaultStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitExpressionStatement(
    node: ExpressionStatementAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitCompoundStatement(node: CompoundStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitIfStatement(node: IfStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitConstevalIfStatement(
    node: ConstevalIfStatementAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSwitchStatement(node: SwitchStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitWhileStatement(node: WhileStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDoStatement(node: DoStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitForRangeStatement(node: ForRangeStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitForStatement(node: ForStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitBreakStatement(node: BreakStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitContinueStatement(node: ContinueStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitReturnStatement(node: ReturnStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitCoroutineReturnStatement(
    node: CoroutineReturnStatementAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitGotoStatement(node: GotoStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDeclarationStatement(
    node: DeclarationStatementAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTryBlockStatement(node: TryBlockStatementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitCharLiteralExpression(
    node: CharLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitBoolLiteralExpression(
    node: BoolLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitIntLiteralExpression(
    node: IntLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitFloatLiteralExpression(
    node: FloatLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNullptrLiteralExpression(
    node: NullptrLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitStringLiteralExpression(
    node: StringLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitUserDefinedStringLiteralExpression(
    node: UserDefinedStringLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitObjectLiteralExpression(
    node: ObjectLiteralExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitThisExpression(node: ThisExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitPackIndexExpression(
    node: PackIndexExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitGenericSelectionExpression(
    node: GenericSelectionExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNestedStatementExpression(
    node: NestedStatementExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDefaultInitializerExpression(
    node: DefaultInitializerExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNestedExpression(node: NestedExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitIdExpression(node: IdExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitLambdaExpression(node: LambdaExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitFoldExpression(node: FoldExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitRightFoldExpression(
    node: RightFoldExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitLeftFoldExpression(node: LeftFoldExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitRequiresExpression(node: RequiresExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitVaArgExpression(node: VaArgExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSubscriptExpression(
    node: SubscriptExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitCallExpression(node: CallExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitTypeConstruction(node: TypeConstructionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitBracedTypeConstruction(
    node: BracedTypeConstructionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSpliceMemberExpression(
    node: SpliceMemberExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitMemberExpression(node: MemberExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitPostIncrExpression(node: PostIncrExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitCppCastExpression(node: CppCastExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitBuiltinBitCastExpression(
    node: BuiltinBitCastExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitBuiltinOffsetofExpression(
    node: BuiltinOffsetofExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypeidExpression(node: TypeidExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitTypeidOfTypeExpression(
    node: TypeidOfTypeExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSpliceExpression(node: SpliceExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitGlobalScopeReflectExpression(
    node: GlobalScopeReflectExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNamespaceReflectExpression(
    node: NamespaceReflectExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypeIdReflectExpression(
    node: TypeIdReflectExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitReflectExpression(node: ReflectExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitLabelAddressExpression(
    node: LabelAddressExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitUnaryExpression(node: UnaryExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAwaitExpression(node: AwaitExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSizeofExpression(node: SizeofExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSizeofTypeExpression(
    node: SizeofTypeExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSizeofPackExpression(
    node: SizeofPackExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitAlignofTypeExpression(
    node: AlignofTypeExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitAlignofExpression(node: AlignofExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNoexceptExpression(node: NoexceptExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNewExpression(node: NewExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDeleteExpression(node: DeleteExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitCastExpression(node: CastExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitImplicitCastExpression(
    node: ImplicitCastExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitConstExpression(node: ConstExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitBinaryExpression(node: BinaryExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitConditionalExpression(
    node: ConditionalExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitYieldExpression(node: YieldExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitThrowExpression(node: ThrowExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAssignmentExpression(
    node: AssignmentExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTargetExpression(node: TargetExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitRightExpression(node: RightExpressionAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitCompoundAssignmentExpression(
    node: CompoundAssignmentExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitPackExpansionExpression(
    node: PackExpansionExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDesignatedInitializerClause(
    node: DesignatedInitializerClauseAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypeTraitExpression(
    node: TypeTraitExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitConditionExpression(
    node: ConditionExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitEqualInitializer(node: EqualInitializerAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitBracedInitList(node: BracedInitListAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitParenInitializer(node: ParenInitializerAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitThreeWayComparisonExpression(
    node: ThreeWayComparisonExpressionAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDefaultGenericAssociation(
    node: DefaultGenericAssociationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypeGenericAssociation(
    node: TypeGenericAssociationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDotDesignator(node: DotDesignatorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSubscriptDesignator(
    node: SubscriptDesignatorAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTemplateTypeParameter(
    node: TemplateTypeParameterAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNonTypeTemplateParameter(
    node: NonTypeTemplateParameterAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypenameTypeParameter(
    node: TypenameTypeParameterAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitConstraintTypeParameter(
    node: ConstraintTypeParameterAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypedefSpecifier(node: TypedefSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitFriendSpecifier(node: FriendSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitConstevalSpecifier(node: ConstevalSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitConstinitSpecifier(node: ConstinitSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitConstexprSpecifier(node: ConstexprSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitInlineSpecifier(node: InlineSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNoreturnSpecifier(node: NoreturnSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitStaticSpecifier(node: StaticSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitExternSpecifier(node: ExternSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitRegisterSpecifier(node: RegisterSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitThreadLocalSpecifier(
    node: ThreadLocalSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitThreadSpecifier(node: ThreadSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitMutableSpecifier(node: MutableSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitVirtualSpecifier(node: VirtualSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitExplicitSpecifier(node: ExplicitSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAutoTypeSpecifier(node: AutoTypeSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitVoidTypeSpecifier(node: VoidTypeSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSizeTypeSpecifier(node: SizeTypeSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSignTypeSpecifier(node: SignTypeSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitBuiltinTypeSpecifier(
    node: BuiltinTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitUnaryBuiltinTypeSpecifier(
    node: UnaryBuiltinTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitBinaryBuiltinTypeSpecifier(
    node: BinaryBuiltinTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitIntegralTypeSpecifier(
    node: IntegralTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitFloatingPointTypeSpecifier(
    node: FloatingPointTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitComplexTypeSpecifier(
    node: ComplexTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNamedTypeSpecifier(node: NamedTypeSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAtomicTypeSpecifier(
    node: AtomicTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitBitIntTypeSpecifier(
    node: BitIntTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitUnderlyingTypeSpecifier(
    node: UnderlyingTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitElaboratedTypeSpecifier(
    node: ElaboratedTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDecltypeAutoSpecifier(
    node: DecltypeAutoSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDecltypeSpecifier(node: DecltypeSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitPlaceholderTypeSpecifier(
    node: PlaceholderTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitConstQualifier(node: ConstQualifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitVolatileQualifier(node: VolatileQualifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAtomicQualifier(node: AtomicQualifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitRestrictQualifier(node: RestrictQualifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitEnumSpecifier(node: EnumSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitClassSpecifier(node: ClassSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitTypenameSpecifier(node: TypenameSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSplicerTypeSpecifier(
    node: SplicerTypeSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitPointerOperator(node: PointerOperatorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitReferenceOperator(node: ReferenceOperatorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitPtrToMemberOperator(
    node: PtrToMemberOperatorAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitBitfieldDeclarator(node: BitfieldDeclaratorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitParameterPack(node: ParameterPackAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitIdDeclarator(node: IdDeclaratorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNestedDeclarator(node: NestedDeclaratorAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitFunctionDeclaratorChunk(
    node: FunctionDeclaratorChunkAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitArrayDeclaratorChunk(
    node: ArrayDeclaratorChunkAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNameId(node: NameIdAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDestructorId(node: DestructorIdAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDecltypeId(node: DecltypeIdAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitOperatorFunctionId(node: OperatorFunctionIdAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitLiteralOperatorId(node: LiteralOperatorIdAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitConversionFunctionId(
    node: ConversionFunctionIdAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSimpleTemplateId(node: SimpleTemplateIdAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitLiteralOperatorTemplateId(
    node: LiteralOperatorTemplateIdAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitOperatorFunctionTemplateId(
    node: OperatorFunctionTemplateIdAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitGlobalNestedNameSpecifier(
    node: GlobalNestedNameSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSimpleNestedNameSpecifier(
    node: SimpleNestedNameSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDecltypeNestedNameSpecifier(
    node: DecltypeNestedNameSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTemplateNestedNameSpecifier(
    node: TemplateNestedNameSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDefaultFunctionBody(
    node: DefaultFunctionBodyAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitCompoundStatementFunctionBody(
    node: CompoundStatementFunctionBodyAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTryStatementFunctionBody(
    node: TryStatementFunctionBodyAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitDeleteFunctionBody(node: DeleteFunctionBodyAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitTypeTemplateArgument(
    node: TypeTemplateArgumentAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitExpressionTemplateArgument(
    node: ExpressionTemplateArgumentAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitThrowExceptionSpecifier(
    node: ThrowExceptionSpecifierAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNoexceptSpecifier(node: NoexceptSpecifierAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitSimpleRequirement(node: SimpleRequirementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitCompoundRequirement(
    node: CompoundRequirementAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypeRequirement(node: TypeRequirementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNestedRequirement(node: NestedRequirementAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitNewParenInitializer(
    node: NewParenInitializerAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitNewBracedInitializer(
    node: NewBracedInitializerAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitParenMemInitializer(
    node: ParenMemInitializerAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitBracedMemInitializer(
    node: BracedMemInitializerAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitThisLambdaCapture(node: ThisLambdaCaptureAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitDerefThisLambdaCapture(
    node: DerefThisLambdaCaptureAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSimpleLambdaCapture(
    node: SimpleLambdaCaptureAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitRefLambdaCapture(node: RefLambdaCaptureAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitRefInitLambdaCapture(
    node: RefInitLambdaCaptureAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitInitLambdaCapture(node: InitLambdaCaptureAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitEllipsisExceptionDeclaration(
    node: EllipsisExceptionDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitTypeExceptionDeclaration(
    node: TypeExceptionDeclarationAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitCxxAttribute(node: CxxAttributeAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitGccAttribute(node: GccAttributeAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAlignasAttribute(node: AlignasAttributeAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitAlignasTypeAttribute(
    node: AlignasTypeAttributeAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitAsmAttribute(node: AsmAttributeAST, context: Context): void {
    this.visitChildren(node, context);
  }
  visitScopedAttributeToken(
    node: ScopedAttributeTokenAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
  visitSimpleAttributeToken(
    node: SimpleAttributeTokenAST,
    context: Context,
  ): void {
    this.visitChildren(node, context);
  }
}
