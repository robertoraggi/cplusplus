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

/**
 * The AST, the symbols, the types and the other classes of the semantic model.
 *
 * @module cxx-frontend/model
 */

import { cxx } from "./cxx.js";
import { type SourceLocation } from "./SourceLocation.js";
import { Token } from "./Token.js";

/** @category Core */
export interface ModelOwner {
  getUnitHandle(): number;
  readonly disposed: boolean;
}

function disposedError(): Error {
  return new Error("Parser has been disposed");
}

/** @category Core */
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

function optionalOf<T>(value: any, of: (item: any) => T): T | undefined {
  if (value === undefined) return undefined;
  return of(value);
}
function astOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = astKindNames[cxx.getASTKind(handle)]!;
  return new astConstructors[kind](handle, owner, kind);
}
function symbolOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = symbolKindNames[cxx.getSymbolKind(handle)]!;
  return new symbolConstructors[kind](handle, owner, kind);
}
function typeOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = typeKindNames[cxx.getTypeKind(handle)]!;
  return new typeConstructors[kind](handle, owner, kind);
}
function nameOf(handle: number, owner: ModelOwner): any {
  if (!handle) return undefined;
  const kind = nameKindNames[cxx.getNameKind(handle)]!;
  return new nameConstructors[kind](handle, owner, kind);
}
function* miscValItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const unit = owner.getUnitHandle();
  const size = cxx.readMiscSize(handle, unit, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readMiscItemVal(handle, unit, slot, i));
  }
}
function* nameValItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const unit = owner.getUnitHandle();
  const size = cxx.readNameSize(handle, unit, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readNameItemVal(handle, unit, slot, i));
  }
}
function* symbolItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const unit = owner.getUnitHandle();
  const size = cxx.readSymbolSize(handle, unit, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readSymbolItem(handle, unit, slot, i));
  }
}
function* symbolValItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const unit = owner.getUnitHandle();
  const size = cxx.readSymbolSize(handle, unit, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readSymbolItemVal(handle, unit, slot, i));
  }
}
function* symbolStringItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const unit = owner.getUnitHandle();
  const size = cxx.readSymbolSize(handle, unit, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readSymbolItemString(handle, unit, slot, i));
  }
}
function* typeItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const unit = owner.getUnitHandle();
  const size = cxx.readTypeSize(handle, unit, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readTypeItem(handle, unit, slot, i));
  }
}
function* typeValItems(
  owner: ModelOwner,
  handle: number,
  slot: number,
  of: (item: any) => any,
): Iterable<any> {
  const unit = owner.getUnitHandle();
  const size = cxx.readTypeSize(handle, unit, slot);
  for (let i = 0; i < size; ++i) {
    if (owner.disposed) throw disposedError();
    yield of(cxx.readTypeItemVal(handle, unit, slot, i));
  }
}
import { type TokenKind, tokenKindNames } from "./TokenKind.js";
/** @category Data Structures */
export type ConstValue =
  | bigint
  | StringLiteral
  | number
  | Meta
  | InitializerList
  | ConstObject
  | ConstAddress
  | ConstLabelAddress
  | ConstComplex
  | undefined;
/** @category Data Structures */
export interface ConstObject_Member {
  readonly symbol: Symbol | undefined;
  readonly value: ConstValue;
}
/** @category Data Structures */
export interface Attribute {
  readonly attributeNamespace: Identifier | undefined;
  readonly name: Identifier | undefined;
  readonly arguments: ReadonlyArray<Identifier | undefined>;
}
/** @category Data Structures */
export interface IntegerLiteral_Components {
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
}
/** @category Data Structures */
export interface FloatLiteral_Components {
  readonly value: number;
  readonly literalPart: string;
  readonly userSuffix: string;
  readonly suffix: FloatLiteral_Components_FloatingPointSuffix;
  readonly isDouble: boolean;
  readonly isFloat: boolean;
  readonly isLongDouble: boolean;
}
/** @category Data Structures */
export interface StringLiteral_Components {
  readonly value: string;
  readonly userSuffix: string;
  readonly encoding: StringLiteralEncoding;
  readonly isRaw: boolean;
}
/** @category Data Structures */
export interface CharLiteral_Components {
  readonly value: number;
  readonly prefix: string;
  readonly userSuffix: string;
}
/** @category Data Structures */
export type TemplateArgument =
  Type | Symbol | ConstValue | ExpressionAST | undefined;
/** @category Data Structures */
export interface InstantiationError {
  readonly location: Token | undefined;
  readonly message: string;
  readonly severity: Severity;
}
/** @category Data Structures */
export interface TemplateFriendship {
  readonly arguments: ReadonlyArray<TemplateArgument>;
  readonly befriendingClass: ClassSymbol | undefined;
}
/** @category Data Structures */
export interface ClassSymbol_BaseClassRepetition {
  readonly nonDiamondRepeat: boolean;
  readonly diamondShaped: boolean;
}
/** @category Data Structures */
export interface VTableLayout_Offset {
  readonly subject: Symbol | undefined;
  readonly value: bigint;
}
/** @category Data Structures */
export interface VTableLayout_CallOffset {
  readonly nonVirtual: bigint;
  readonly virtualOffset: bigint;
}
/** @category Data Structures */
export interface VTableLayout_Slot {
  readonly function: FunctionSymbol | undefined;
  readonly kind: VTableLayout_SlotKind;
  readonly introducingFunction: FunctionSymbol | undefined;
  readonly thisAdjustment: VTableLayout_CallOffset;
  readonly returnAdjustment: VTableLayout_CallOffset;
}
/** @category Data Structures */
export interface VTableLayout_Table {
  readonly base: ClassSymbol | undefined;
  readonly offset: bigint;
  readonly enclosingVirtualBase: ClassSymbol | undefined;
  readonly offsetToTop: bigint;
  readonly offsets: ReadonlyArray<VTableLayout_Offset>;
  readonly slots: ReadonlyArray<VTableLayout_Slot>;
}
/** @category Data Structures */
export interface VTableLayout_Group {
  readonly base: ClassSymbol | undefined;
  readonly offset: bigint;
  readonly tables: ReadonlyArray<VTableLayout_Table>;
}
/** @category Data Structures */
export interface VTableLayout_VTTEntry {
  readonly group: number;
  readonly table: number;
}
/** @category Data Structures */
export interface VTableLayout_SubVTT {
  readonly base: ClassSymbol | undefined;
  readonly index: number;
}
/** @category Data Structures */
export interface VTableLayout_EntryPoint {
  readonly function: FunctionSymbol | undefined;
  readonly kind: VTableLayout_SlotKind;
  readonly thisAdjustment: VTableLayout_CallOffset;
  readonly returnAdjustment: VTableLayout_CallOffset;
}
/** @category Data Structures */
export interface VTableLayout {
  readonly main: VTableLayout_Group;
  readonly constructionGroups: ReadonlyArray<VTableLayout_Group>;
  readonly vtt: ReadonlyArray<VTableLayout_VTTEntry>;
  readonly tableVTTIndices: ReadonlyArray<number>;
  readonly baseSubVTTs: ReadonlyArray<VTableLayout_SubVTT>;
  readonly virtualBaseSubVTTs: ReadonlyArray<VTableLayout_SubVTT>;
  readonly adjustingEntryPoints: ReadonlyArray<VTableLayout_EntryPoint>;
  readonly keyFunction: FunctionSymbol | undefined;
}
/** @category Data Structures */
export interface PendingInstantiation {
  readonly pattern: AST | undefined;
  readonly instance: AST | undefined;
  readonly templateArguments: ReadonlyArray<TemplateArgument>;
  readonly parentScope: ScopeSymbol | undefined;
  readonly depth: number;
  readonly state: PendingInstantiationState;
}
/** @category Data Structures */
export type ExceptionSpecification = boolean | ExpressionAST | undefined;
/** @category Data Structures */
export type SourceLocationRange = readonly [
  Token | undefined,
  Token | undefined,
];
function decodeConstValue(value: any, owner: ModelOwner): ConstValue {
  switch (value.index) {
    case 0:
      return BigInt(value.value);
    case 1:
      return objOf(value.value, owner, StringLiteral);
    case 5:
      return objOf(value.value, owner, Meta);
    case 6:
      return objOf(value.value, owner, InitializerList);
    case 7:
      return objOf(value.value, owner, ConstObject);
    case 8:
      return objOf(value.value, owner, ConstAddress);
    case 9:
      return objOf(value.value, owner, ConstLabelAddress);
    case 10:
      return objOf(value.value, owner, ConstComplex);
    case 11:
      return undefined;
    default:
      return value.value;
  }
}
function decodeConstObject_Member(
  value: any,
  owner: ModelOwner,
): ConstObject_Member {
  return {
    symbol: symbolOf(value.symbol, owner),
    value: decodeConstValue(value.value, owner),
  };
}
function decodeAttribute(value: any, owner: ModelOwner): Attribute {
  return {
    attributeNamespace: nameOf(value.attributeNamespace, owner),
    name: nameOf(value.name, owner),
    arguments: (value.arguments as any[]).map((element: any) =>
      nameOf(element, owner),
    ),
  };
}
function decodeIntegerLiteral_Components(
  value: any,
  owner: ModelOwner,
): IntegerLiteral_Components {
  return {
    value: value.value,
    integerPart: value.integerPart,
    userSuffix: value.userSuffix,
    radix: integerLiteral_RadixNames[value.radix]!,
    isUnsigned: value.isUnsigned !== 0,
    isLongLong: value.isLongLong !== 0,
    isLong: value.isLong !== 0,
    hasSizeSuffix: value.hasSizeSuffix !== 0,
    isWB: value.isWB !== 0,
    bitIntWidth: value.bitIntWidth,
  };
}
function decodeFloatLiteral_Components(
  value: any,
  owner: ModelOwner,
): FloatLiteral_Components {
  return {
    value: value.value,
    literalPart: value.literalPart,
    userSuffix: value.userSuffix,
    suffix: floatLiteral_Components_FloatingPointSuffixNames[value.suffix]!,
    isDouble: value.isDouble !== 0,
    isFloat: value.isFloat !== 0,
    isLongDouble: value.isLongDouble !== 0,
  };
}
function decodeStringLiteral_Components(
  value: any,
  owner: ModelOwner,
): StringLiteral_Components {
  return {
    value: value.value,
    userSuffix: value.userSuffix,
    encoding: stringLiteralEncodingNames[value.encoding]!,
    isRaw: value.isRaw !== 0,
  };
}
function decodeTemplateArgument(
  value: any,
  owner: ModelOwner,
): TemplateArgument {
  switch (value.index) {
    case 0:
      return typeOf(value.value, owner);
    case 1:
      return symbolOf(value.value, owner);
    case 2:
      return decodeConstValue(value.value, owner);
    case 3:
      return astOf(value.value, owner);
    default:
      return value.value;
  }
}
function decodeInstantiationError(
  value: any,
  owner: ModelOwner,
): InstantiationError {
  return {
    location: Token.from(value.location, owner),
    message: value.message,
    severity: severityNames[value.severity]!,
  };
}
function decodeTemplateFriendship(
  value: any,
  owner: ModelOwner,
): TemplateFriendship {
  return {
    arguments: (value.arguments as any[]).map((element: any) =>
      decodeTemplateArgument(element, owner),
    ),
    befriendingClass: symbolOf(value.befriendingClass, owner),
  };
}
function decodeClassSymbol_BaseClassRepetition(
  value: any,
  owner: ModelOwner,
): ClassSymbol_BaseClassRepetition {
  return {
    nonDiamondRepeat: value.nonDiamondRepeat !== 0,
    diamondShaped: value.diamondShaped !== 0,
  };
}
function decodeVTableLayout_Offset(
  value: any,
  owner: ModelOwner,
): VTableLayout_Offset {
  return {
    subject: symbolOf(value.subject, owner),
    value: value.value,
  };
}
function decodeVTableLayout_Slot(
  value: any,
  owner: ModelOwner,
): VTableLayout_Slot {
  return {
    function: symbolOf(value.function, owner),
    kind: vTableLayout_SlotKindNames[value.kind]!,
    introducingFunction: symbolOf(value.introducingFunction, owner),
    thisAdjustment: value.thisAdjustment,
    returnAdjustment: value.returnAdjustment,
  };
}
function decodeVTableLayout_Table(
  value: any,
  owner: ModelOwner,
): VTableLayout_Table {
  return {
    base: symbolOf(value.base, owner),
    offset: value.offset,
    enclosingVirtualBase: symbolOf(value.enclosingVirtualBase, owner),
    offsetToTop: value.offsetToTop,
    offsets: (value.offsets as any[]).map((element: any) =>
      decodeVTableLayout_Offset(element, owner),
    ),
    slots: (value.slots as any[]).map((element: any) =>
      decodeVTableLayout_Slot(element, owner),
    ),
  };
}
function decodeVTableLayout_Group(
  value: any,
  owner: ModelOwner,
): VTableLayout_Group {
  return {
    base: symbolOf(value.base, owner),
    offset: value.offset,
    tables: (value.tables as any[]).map((element: any) =>
      decodeVTableLayout_Table(element, owner),
    ),
  };
}
function decodeVTableLayout_SubVTT(
  value: any,
  owner: ModelOwner,
): VTableLayout_SubVTT {
  return {
    base: symbolOf(value.base, owner),
    index: value.index,
  };
}
function decodeVTableLayout_EntryPoint(
  value: any,
  owner: ModelOwner,
): VTableLayout_EntryPoint {
  return {
    function: symbolOf(value.function, owner),
    kind: vTableLayout_SlotKindNames[value.kind]!,
    thisAdjustment: value.thisAdjustment,
    returnAdjustment: value.returnAdjustment,
  };
}
function decodeVTableLayout(value: any, owner: ModelOwner): VTableLayout {
  return {
    main: decodeVTableLayout_Group(value.main, owner),
    constructionGroups: (value.constructionGroups as any[]).map(
      (element: any) => decodeVTableLayout_Group(element, owner),
    ),
    vtt: value.vtt,
    tableVTTIndices: value.tableVTTIndices,
    baseSubVTTs: (value.baseSubVTTs as any[]).map((element: any) =>
      decodeVTableLayout_SubVTT(element, owner),
    ),
    virtualBaseSubVTTs: (value.virtualBaseSubVTTs as any[]).map(
      (element: any) => decodeVTableLayout_SubVTT(element, owner),
    ),
    adjustingEntryPoints: (value.adjustingEntryPoints as any[]).map(
      (element: any) => decodeVTableLayout_EntryPoint(element, owner),
    ),
    keyFunction: symbolOf(value.keyFunction, owner),
  };
}
function decodePendingInstantiation(
  value: any,
  owner: ModelOwner,
): PendingInstantiation {
  return {
    pattern: astOf(value.pattern, owner),
    instance: astOf(value.instance, owner),
    templateArguments: (value.templateArguments as any[]).map((element: any) =>
      decodeTemplateArgument(element, owner),
    ),
    parentScope: symbolOf(value.parentScope, owner),
    depth: value.depth,
    state: pendingInstantiationStateNames[value.state]!,
  };
}
function decodeExceptionSpecification(
  value: any,
  owner: ModelOwner,
): ExceptionSpecification {
  switch (value.index) {
    case 0:
      return value.value !== 0;
    case 1:
      return astOf(value.value, owner);
    default:
      return value.value;
  }
}
function decodeSourceLocationRange(
  value: any,
  owner: ModelOwner,
): SourceLocationRange {
  return [Token.from(value[0], owner), Token.from(value[1], owner)] as const;
}
const ConstComplexSlotBase = 0;
const ConstObjectSlotBase = ConstComplexSlotBase + 2;
const ConstAddressSlotBase = ConstObjectSlotBase + 4;
const ConstLabelAddressSlotBase = ConstAddressSlotBase + 6;
const ASTSlotBase = 0;
const AttributeSpecifierASTSlotBase = ASTSlotBase + 3;
const ExpressionASTSlotBase = AttributeSpecifierASTSlotBase + 1;
const MemInitializerASTSlotBase = ExpressionASTSlotBase + 2;
const NestedNameSpecifierASTSlotBase = MemInitializerASTSlotBase + 2;
const TemplateParameterASTSlotBase = NestedNameSpecifierASTSlotBase + 1;
const UnitASTSlotBase = TemplateParameterASTSlotBase + 3;
const TranslationUnitASTSlotBase = UnitASTSlotBase + 1;
const ModuleUnitASTSlotBase = TranslationUnitASTSlotBase + 1;
const SimpleDeclarationASTSlotBase = ModuleUnitASTSlotBase + 4;
const AsmDeclarationASTSlotBase = SimpleDeclarationASTSlotBase + 5;
const NamespaceAliasDefinitionASTSlotBase = AsmDeclarationASTSlotBase + 12;
const UsingDeclarationASTSlotBase = NamespaceAliasDefinitionASTSlotBase + 8;
const UsingEnumDeclarationASTSlotBase = UsingDeclarationASTSlotBase + 3;
const UsingDirectiveASTSlotBase = UsingEnumDeclarationASTSlotBase + 3;
const StaticAssertDeclarationASTSlotBase = UsingDirectiveASTSlotBase + 6;
const AliasDeclarationASTSlotBase = StaticAssertDeclarationASTSlotBase + 9;
const OpaqueEnumDeclarationASTSlotBase = AliasDeclarationASTSlotBase + 9;
const FunctionDefinitionASTSlotBase = OpaqueEnumDeclarationASTSlotBase + 9;
const TemplateDeclarationASTSlotBase = FunctionDefinitionASTSlotBase + 6;
const ConceptDefinitionASTSlotBase = TemplateDeclarationASTSlotBase + 8;
const DeductionGuideASTSlotBase = ConceptDefinitionASTSlotBase + 7;
const ExplicitInstantiationASTSlotBase = DeductionGuideASTSlotBase + 11;
const ExportDeclarationASTSlotBase = ExplicitInstantiationASTSlotBase + 3;
const ExportCompoundDeclarationASTSlotBase = ExportDeclarationASTSlotBase + 2;
const LinkageSpecificationASTSlotBase =
  ExportCompoundDeclarationASTSlotBase + 4;
const NamespaceDefinitionASTSlotBase = LinkageSpecificationASTSlotBase + 6;
const EmptyDeclarationASTSlotBase = NamespaceDefinitionASTSlotBase + 12;
const AttributeDeclarationASTSlotBase = EmptyDeclarationASTSlotBase + 1;
const ModuleImportDeclarationASTSlotBase = AttributeDeclarationASTSlotBase + 2;
const ParameterDeclarationASTSlotBase = ModuleImportDeclarationASTSlotBase + 4;
const AccessDeclarationASTSlotBase = ParameterDeclarationASTSlotBase + 11;
const StructuredBindingDeclarationASTSlotBase =
  AccessDeclarationASTSlotBase + 3;
const AsmOperandASTSlotBase = StructuredBindingDeclarationASTSlotBase + 10;
const AsmQualifierASTSlotBase = AsmOperandASTSlotBase + 9;
const AsmClobberASTSlotBase = AsmQualifierASTSlotBase + 2;
const AsmGotoLabelASTSlotBase = AsmClobberASTSlotBase + 2;
const SplicerASTSlotBase = AsmGotoLabelASTSlotBase + 2;
const GlobalModuleFragmentASTSlotBase = SplicerASTSlotBase + 6;
const PrivateModuleFragmentASTSlotBase = GlobalModuleFragmentASTSlotBase + 3;
const ModuleDeclarationASTSlotBase = PrivateModuleFragmentASTSlotBase + 5;
const ModuleNameASTSlotBase = ModuleDeclarationASTSlotBase + 6;
const ModuleQualifierASTSlotBase = ModuleNameASTSlotBase + 3;
const ModulePartitionASTSlotBase = ModuleQualifierASTSlotBase + 4;
const ImportNameASTSlotBase = ModulePartitionASTSlotBase + 2;
const InitDeclaratorASTSlotBase = ImportNameASTSlotBase + 3;
const DeclaratorASTSlotBase = InitDeclaratorASTSlotBase + 4;
const UsingDeclaratorASTSlotBase = DeclaratorASTSlotBase + 3;
const EnumeratorASTSlotBase = UsingDeclaratorASTSlotBase + 6;
const TypeIdASTSlotBase = EnumeratorASTSlotBase + 6;
const HandlerASTSlotBase = TypeIdASTSlotBase + 4;
const BaseSpecifierASTSlotBase = HandlerASTSlotBase + 6;
const RequiresClauseASTSlotBase = BaseSpecifierASTSlotBase + 12;
const ParameterDeclarationClauseASTSlotBase = RequiresClauseASTSlotBase + 2;
const TrailingReturnTypeASTSlotBase = ParameterDeclarationClauseASTSlotBase + 5;
const LambdaSpecifierASTSlotBase = TrailingReturnTypeASTSlotBase + 2;
const TypeConstraintASTSlotBase = LambdaSpecifierASTSlotBase + 2;
const AttributeArgumentClauseASTSlotBase = TypeConstraintASTSlotBase + 7;
const AttributeASTSlotBase = AttributeArgumentClauseASTSlotBase + 3;
const AttributeUsingPrefixASTSlotBase = AttributeASTSlotBase + 3;
const NewPlacementASTSlotBase = AttributeUsingPrefixASTSlotBase + 3;
const NestedNamespaceSpecifierASTSlotBase = NewPlacementASTSlotBase + 3;
const LabeledStatementASTSlotBase = NestedNamespaceSpecifierASTSlotBase + 6;
const CaseStatementASTSlotBase = LabeledStatementASTSlotBase + 4;
const DefaultStatementASTSlotBase = CaseStatementASTSlotBase + 4;
const ExpressionStatementASTSlotBase = DefaultStatementASTSlotBase + 2;
const CompoundStatementASTSlotBase = ExpressionStatementASTSlotBase + 3;
const IfStatementASTSlotBase = CompoundStatementASTSlotBase + 5;
const ConstevalIfStatementASTSlotBase = IfStatementASTSlotBase + 11;
const SwitchStatementASTSlotBase = ConstevalIfStatementASTSlotBase + 8;
const WhileStatementASTSlotBase = SwitchStatementASTSlotBase + 8;
const DoStatementASTSlotBase = WhileStatementASTSlotBase + 7;
const ForRangeStatementASTSlotBase = DoStatementASTSlotBase + 8;
const ForStatementASTSlotBase = ForRangeStatementASTSlotBase + 27;
const BreakStatementASTSlotBase = ForStatementASTSlotBase + 10;
const ContinueStatementASTSlotBase = BreakStatementASTSlotBase + 3;
const ReturnStatementASTSlotBase = ContinueStatementASTSlotBase + 3;
const CoroutineReturnStatementASTSlotBase = ReturnStatementASTSlotBase + 4;
const GotoStatementASTSlotBase = CoroutineReturnStatementASTSlotBase + 4;
const DeclarationStatementASTSlotBase = GotoStatementASTSlotBase + 8;
const TryBlockStatementASTSlotBase = DeclarationStatementASTSlotBase + 1;
const CharLiteralExpressionASTSlotBase = TryBlockStatementASTSlotBase + 4;
const BoolLiteralExpressionASTSlotBase = CharLiteralExpressionASTSlotBase + 3;
const IntLiteralExpressionASTSlotBase = BoolLiteralExpressionASTSlotBase + 2;
const FloatLiteralExpressionASTSlotBase = IntLiteralExpressionASTSlotBase + 3;
const NullptrLiteralExpressionASTSlotBase =
  FloatLiteralExpressionASTSlotBase + 3;
const StringLiteralExpressionASTSlotBase =
  NullptrLiteralExpressionASTSlotBase + 2;
const UserDefinedStringLiteralExpressionASTSlotBase =
  StringLiteralExpressionASTSlotBase + 3;
const ObjectLiteralExpressionASTSlotBase =
  UserDefinedStringLiteralExpressionASTSlotBase + 4;
const ThisExpressionASTSlotBase = ObjectLiteralExpressionASTSlotBase + 5;
const PackIndexExpressionASTSlotBase = ThisExpressionASTSlotBase + 1;
const GenericSelectionExpressionASTSlotBase =
  PackIndexExpressionASTSlotBase + 5;
const NestedStatementExpressionASTSlotBase =
  GenericSelectionExpressionASTSlotBase + 7;
const DefaultInitializerExpressionASTSlotBase =
  NestedStatementExpressionASTSlotBase + 3;
const NestedExpressionASTSlotBase = DefaultInitializerExpressionASTSlotBase + 2;
const IdExpressionASTSlotBase = NestedExpressionASTSlotBase + 3;
const LambdaExpressionASTSlotBase = IdExpressionASTSlotBase + 5;
const FoldExpressionASTSlotBase = LambdaExpressionASTSlotBase + 22;
const RightFoldExpressionASTSlotBase = FoldExpressionASTSlotBase + 9;
const LeftFoldExpressionASTSlotBase = RightFoldExpressionASTSlotBase + 6;
const RequiresExpressionASTSlotBase = LeftFoldExpressionASTSlotBase + 6;
const VaArgExpressionASTSlotBase = RequiresExpressionASTSlotBase + 7;
const SubscriptExpressionASTSlotBase = VaArgExpressionASTSlotBase + 6;
const CallExpressionASTSlotBase = SubscriptExpressionASTSlotBase + 6;
const TypeConstructionASTSlotBase = CallExpressionASTSlotBase + 6;
const BracedTypeConstructionASTSlotBase = TypeConstructionASTSlotBase + 5;
const SpliceMemberExpressionASTSlotBase = BracedTypeConstructionASTSlotBase + 3;
const MemberExpressionASTSlotBase = SpliceMemberExpressionASTSlotBase + 7;
const PostIncrExpressionASTSlotBase = MemberExpressionASTSlotBase + 8;
const CppCastExpressionASTSlotBase = PostIncrExpressionASTSlotBase + 5;
const BuiltinBitCastExpressionASTSlotBase = CppCastExpressionASTSlotBase + 8;
const BuiltinConvertVectorExpressionASTSlotBase =
  BuiltinBitCastExpressionASTSlotBase + 6;
const BuiltinOffsetofExpressionASTSlotBase =
  BuiltinConvertVectorExpressionASTSlotBase + 6;
const TypeidExpressionASTSlotBase = BuiltinOffsetofExpressionASTSlotBase + 9;
const TypeidOfTypeExpressionASTSlotBase = TypeidExpressionASTSlotBase + 4;
const SpliceExpressionASTSlotBase = TypeidOfTypeExpressionASTSlotBase + 4;
const GlobalScopeReflectExpressionASTSlotBase = SpliceExpressionASTSlotBase + 1;
const NamespaceReflectExpressionASTSlotBase =
  GlobalScopeReflectExpressionASTSlotBase + 2;
const TypeIdReflectExpressionASTSlotBase =
  NamespaceReflectExpressionASTSlotBase + 4;
const ReflectExpressionASTSlotBase = TypeIdReflectExpressionASTSlotBase + 2;
const LabelAddressExpressionASTSlotBase = ReflectExpressionASTSlotBase + 2;
const UnaryExpressionASTSlotBase = LabelAddressExpressionASTSlotBase + 3;
const AwaitExpressionASTSlotBase = UnaryExpressionASTSlotBase + 5;
const SizeofExpressionASTSlotBase = AwaitExpressionASTSlotBase + 2;
const SizeofTypeExpressionASTSlotBase = SizeofExpressionASTSlotBase + 3;
const SizeofPackExpressionASTSlotBase = SizeofTypeExpressionASTSlotBase + 5;
const AlignofTypeExpressionASTSlotBase = SizeofPackExpressionASTSlotBase + 7;
const AlignofExpressionASTSlotBase = AlignofTypeExpressionASTSlotBase + 4;
const NoexceptExpressionASTSlotBase = AlignofExpressionASTSlotBase + 2;
const NewExpressionASTSlotBase = NoexceptExpressionASTSlotBase + 5;
const DeleteExpressionASTSlotBase = NewExpressionASTSlotBase + 12;
const CastExpressionASTSlotBase = DeleteExpressionASTSlotBase + 6;
const ImplicitCastExpressionASTSlotBase = CastExpressionASTSlotBase + 4;
const ConstExpressionASTSlotBase = ImplicitCastExpressionASTSlotBase + 4;
const BinaryExpressionASTSlotBase = ConstExpressionASTSlotBase + 2;
const ConditionalExpressionASTSlotBase = BinaryExpressionASTSlotBase + 6;
const YieldExpressionASTSlotBase = ConditionalExpressionASTSlotBase + 5;
const ThrowExpressionASTSlotBase = YieldExpressionASTSlotBase + 2;
const AssignmentExpressionASTSlotBase = ThrowExpressionASTSlotBase + 2;
const CompoundAssignmentExpressionASTSlotBase =
  AssignmentExpressionASTSlotBase + 6;
const PackExpansionExpressionASTSlotBase =
  CompoundAssignmentExpressionASTSlotBase + 8;
const DesignatedInitializerClauseASTSlotBase =
  PackExpansionExpressionASTSlotBase + 2;
const TypeTraitExpressionASTSlotBase =
  DesignatedInitializerClauseASTSlotBase + 3;
const ConditionExpressionASTSlotBase = TypeTraitExpressionASTSlotBase + 6;
const EqualInitializerASTSlotBase = ConditionExpressionASTSlotBase + 5;
const BracedInitListASTSlotBase = EqualInitializerASTSlotBase + 2;
const ParenInitializerASTSlotBase = BracedInitListASTSlotBase + 5;
const ThreeWayComparisonExpressionASTSlotBase = ParenInitializerASTSlotBase + 3;
const DefaultGenericAssociationASTSlotBase =
  ThreeWayComparisonExpressionASTSlotBase + 5;
const TypeGenericAssociationASTSlotBase =
  DefaultGenericAssociationASTSlotBase + 3;
const DotDesignatorASTSlotBase = TypeGenericAssociationASTSlotBase + 3;
const SubscriptDesignatorASTSlotBase = DotDesignatorASTSlotBase + 4;
const TemplateTypeParameterASTSlotBase = SubscriptDesignatorASTSlotBase + 3;
const NonTypeTemplateParameterASTSlotBase =
  TemplateTypeParameterASTSlotBase + 12;
const TypenameTypeParameterASTSlotBase =
  NonTypeTemplateParameterASTSlotBase + 1;
const ConstraintTypeParameterASTSlotBase = TypenameTypeParameterASTSlotBase + 8;
const TypedefSpecifierASTSlotBase = ConstraintTypeParameterASTSlotBase + 7;
const FriendSpecifierASTSlotBase = TypedefSpecifierASTSlotBase + 1;
const ConstevalSpecifierASTSlotBase = FriendSpecifierASTSlotBase + 1;
const ConstinitSpecifierASTSlotBase = ConstevalSpecifierASTSlotBase + 1;
const ConstexprSpecifierASTSlotBase = ConstinitSpecifierASTSlotBase + 1;
const InlineSpecifierASTSlotBase = ConstexprSpecifierASTSlotBase + 1;
const NoreturnSpecifierASTSlotBase = InlineSpecifierASTSlotBase + 1;
const StaticSpecifierASTSlotBase = NoreturnSpecifierASTSlotBase + 1;
const ExternSpecifierASTSlotBase = StaticSpecifierASTSlotBase + 1;
const RegisterSpecifierASTSlotBase = ExternSpecifierASTSlotBase + 1;
const ThreadLocalSpecifierASTSlotBase = RegisterSpecifierASTSlotBase + 1;
const ThreadSpecifierASTSlotBase = ThreadLocalSpecifierASTSlotBase + 1;
const MutableSpecifierASTSlotBase = ThreadSpecifierASTSlotBase + 1;
const VirtualSpecifierASTSlotBase = MutableSpecifierASTSlotBase + 1;
const ExplicitSpecifierASTSlotBase = VirtualSpecifierASTSlotBase + 1;
const AutoTypeSpecifierASTSlotBase = ExplicitSpecifierASTSlotBase + 4;
const VoidTypeSpecifierASTSlotBase = AutoTypeSpecifierASTSlotBase + 1;
const SizeTypeSpecifierASTSlotBase = VoidTypeSpecifierASTSlotBase + 1;
const SignTypeSpecifierASTSlotBase = SizeTypeSpecifierASTSlotBase + 2;
const BuiltinTypeSpecifierASTSlotBase = SignTypeSpecifierASTSlotBase + 2;
const UnaryBuiltinTypeSpecifierASTSlotBase =
  BuiltinTypeSpecifierASTSlotBase + 2;
const BinaryBuiltinTypeSpecifierASTSlotBase =
  UnaryBuiltinTypeSpecifierASTSlotBase + 5;
const IntegralTypeSpecifierASTSlotBase =
  BinaryBuiltinTypeSpecifierASTSlotBase + 7;
const FloatingPointTypeSpecifierASTSlotBase =
  IntegralTypeSpecifierASTSlotBase + 2;
const ComplexTypeSpecifierASTSlotBase =
  FloatingPointTypeSpecifierASTSlotBase + 2;
const NamedTypeSpecifierASTSlotBase = ComplexTypeSpecifierASTSlotBase + 1;
const AtomicTypeSpecifierASTSlotBase = NamedTypeSpecifierASTSlotBase + 5;
const BitIntTypeSpecifierASTSlotBase = AtomicTypeSpecifierASTSlotBase + 4;
const UnderlyingTypeSpecifierASTSlotBase = BitIntTypeSpecifierASTSlotBase + 5;
const ElaboratedTypeSpecifierASTSlotBase =
  UnderlyingTypeSpecifierASTSlotBase + 4;
const DecltypeAutoSpecifierASTSlotBase = ElaboratedTypeSpecifierASTSlotBase + 8;
const DecltypeSpecifierASTSlotBase = DecltypeAutoSpecifierASTSlotBase + 4;
const PlaceholderTypeSpecifierASTSlotBase = DecltypeSpecifierASTSlotBase + 5;
const ConstQualifierASTSlotBase = PlaceholderTypeSpecifierASTSlotBase + 2;
const VolatileQualifierASTSlotBase = ConstQualifierASTSlotBase + 1;
const AtomicQualifierASTSlotBase = VolatileQualifierASTSlotBase + 1;
const RestrictQualifierASTSlotBase = AtomicQualifierASTSlotBase + 1;
const EnumSpecifierASTSlotBase = RestrictQualifierASTSlotBase + 1;
const ClassSpecifierASTSlotBase = EnumSpecifierASTSlotBase + 12;
const TypenameSpecifierASTSlotBase = ClassSpecifierASTSlotBase + 14;
const SplicerTypeSpecifierASTSlotBase = TypenameSpecifierASTSlotBase + 6;
const PointerOperatorASTSlotBase = SplicerTypeSpecifierASTSlotBase + 2;
const ReferenceOperatorASTSlotBase = PointerOperatorASTSlotBase + 3;
const PtrToMemberOperatorASTSlotBase = ReferenceOperatorASTSlotBase + 3;
const BitfieldDeclaratorASTSlotBase = PtrToMemberOperatorASTSlotBase + 4;
const ParameterPackASTSlotBase = BitfieldDeclaratorASTSlotBase + 5;
const IdDeclaratorASTSlotBase = ParameterPackASTSlotBase + 2;
const NestedDeclaratorASTSlotBase = IdDeclaratorASTSlotBase + 5;
const FunctionDeclaratorChunkASTSlotBase = NestedDeclaratorASTSlotBase + 3;
const ArrayDeclaratorChunkASTSlotBase = FunctionDeclaratorChunkASTSlotBase + 12;
const NameIdASTSlotBase = ArrayDeclaratorChunkASTSlotBase + 5;
const DestructorIdASTSlotBase = NameIdASTSlotBase + 2;
const DecltypeIdASTSlotBase = DestructorIdASTSlotBase + 2;
const OperatorFunctionIdASTSlotBase = DecltypeIdASTSlotBase + 1;
const LiteralOperatorIdASTSlotBase = OperatorFunctionIdASTSlotBase + 5;
const ConversionFunctionIdASTSlotBase = LiteralOperatorIdASTSlotBase + 5;
const SimpleTemplateIdASTSlotBase = ConversionFunctionIdASTSlotBase + 2;
const LiteralOperatorTemplateIdASTSlotBase = SimpleTemplateIdASTSlotBase + 6;
const OperatorFunctionTemplateIdASTSlotBase =
  LiteralOperatorTemplateIdASTSlotBase + 4;
const GlobalNestedNameSpecifierASTSlotBase =
  OperatorFunctionTemplateIdASTSlotBase + 4;
const SimpleNestedNameSpecifierASTSlotBase =
  GlobalNestedNameSpecifierASTSlotBase + 1;
const DecltypeNestedNameSpecifierASTSlotBase =
  SimpleNestedNameSpecifierASTSlotBase + 4;
const TemplateNestedNameSpecifierASTSlotBase =
  DecltypeNestedNameSpecifierASTSlotBase + 2;
const DefaultFunctionBodyASTSlotBase =
  TemplateNestedNameSpecifierASTSlotBase + 5;
const CompoundStatementFunctionBodyASTSlotBase =
  DefaultFunctionBodyASTSlotBase + 3;
const TryStatementFunctionBodyASTSlotBase =
  CompoundStatementFunctionBodyASTSlotBase + 3;
const DeleteFunctionBodyASTSlotBase = TryStatementFunctionBodyASTSlotBase + 5;
const TypeTemplateArgumentASTSlotBase = DeleteFunctionBodyASTSlotBase + 3;
const ExpressionTemplateArgumentASTSlotBase =
  TypeTemplateArgumentASTSlotBase + 1;
const ThrowExceptionSpecifierASTSlotBase =
  ExpressionTemplateArgumentASTSlotBase + 1;
const NoexceptSpecifierASTSlotBase = ThrowExceptionSpecifierASTSlotBase + 3;
const SimpleRequirementASTSlotBase = NoexceptSpecifierASTSlotBase + 4;
const CompoundRequirementASTSlotBase = SimpleRequirementASTSlotBase + 2;
const TypeRequirementASTSlotBase = CompoundRequirementASTSlotBase + 7;
const NestedRequirementASTSlotBase = TypeRequirementASTSlotBase + 3;
const NewParenInitializerASTSlotBase = NestedRequirementASTSlotBase + 3;
const NewBracedInitializerASTSlotBase = NewParenInitializerASTSlotBase + 3;
const ParenMemInitializerASTSlotBase = NewBracedInitializerASTSlotBase + 1;
const BracedMemInitializerASTSlotBase = ParenMemInitializerASTSlotBase + 6;
const ThisLambdaCaptureASTSlotBase = BracedMemInitializerASTSlotBase + 4;
const DerefThisLambdaCaptureASTSlotBase = ThisLambdaCaptureASTSlotBase + 3;
const SimpleLambdaCaptureASTSlotBase = DerefThisLambdaCaptureASTSlotBase + 3;
const RefLambdaCaptureASTSlotBase = SimpleLambdaCaptureASTSlotBase + 5;
const RefInitLambdaCaptureASTSlotBase = RefLambdaCaptureASTSlotBase + 6;
const InitLambdaCaptureASTSlotBase = RefInitLambdaCaptureASTSlotBase + 6;
const EllipsisExceptionDeclarationASTSlotBase =
  InitLambdaCaptureASTSlotBase + 5;
const TypeExceptionDeclarationASTSlotBase =
  EllipsisExceptionDeclarationASTSlotBase + 1;
const CxxAttributeASTSlotBase = TypeExceptionDeclarationASTSlotBase + 4;
const GccAttributeASTSlotBase = CxxAttributeASTSlotBase + 6;
const AlignasAttributeASTSlotBase = GccAttributeASTSlotBase + 6;
const AlignasTypeAttributeASTSlotBase = AlignasAttributeASTSlotBase + 6;
const AsmAttributeASTSlotBase = AlignasTypeAttributeASTSlotBase + 6;
const ScopedAttributeTokenASTSlotBase = AsmAttributeASTSlotBase + 5;
const SimpleAttributeTokenASTSlotBase = ScopedAttributeTokenASTSlotBase + 5;
const LiteralSlotBase = 0;
const IntegerLiteralSlotBase = LiteralSlotBase + 2;
const FloatLiteralSlotBase = IntegerLiteralSlotBase + 2;
const StringLiteralSlotBase = FloatLiteralSlotBase + 2;
const CharLiteralSlotBase = StringLiteralSlotBase + 6;
const NameSlotBase = 0;
const IdentifierSlotBase = NameSlotBase + 2;
const OperatorIdSlotBase = IdentifierSlotBase + 8;
const DestructorIdSlotBase = OperatorIdSlotBase + 1;
const LiteralOperatorIdSlotBase = DestructorIdSlotBase + 1;
const ConversionFunctionIdSlotBase = LiteralOperatorIdSlotBase + 1;
const TemplateIdSlotBase = ConversionFunctionIdSlotBase + 1;
const SymbolSlotBase = 0;
const ScopeSymbolSlotBase = SymbolSlotBase + 58;
const NamespaceSymbolSlotBase = ScopeSymbolSlotBase + 4;
const ConceptSymbolSlotBase = NamespaceSymbolSlotBase + 4;
const DeductionGuideSymbolSlotBase = ConceptSymbolSlotBase + 9;
const BaseClassSymbolSlotBase = DeductionGuideSymbolSlotBase + 10;
const InjectedClassNameSymbolSlotBase = BaseClassSymbolSlotBase + 2;
const ClassSymbolSlotBase = InjectedClassNameSymbolSlotBase + 1;
const EnumSymbolSlotBase = ClassSymbolSlotBase + 63;
const ScopedEnumSymbolSlotBase = EnumSymbolSlotBase + 3;
const FunctionSymbolSlotBase = ScopedEnumSymbolSlotBase + 2;
const OverloadSetSymbolSlotBase = FunctionSymbolSlotBase + 74;
const LambdaSymbolSlotBase = OverloadSetSymbolSlotBase + 4;
const FunctionParametersSymbolSlotBase = LambdaSymbolSlotBase + 7;
const TemplateParametersSymbolSlotBase = FunctionParametersSymbolSlotBase + 1;
const BlockSymbolSlotBase = TemplateParametersSymbolSlotBase + 1;
const TypeAliasSymbolSlotBase = BlockSymbolSlotBase + 2;
const VariableSymbolSlotBase = TypeAliasSymbolSlotBase + 16;
const FieldSymbolSlotBase = VariableSymbolSlotBase + 26;
const ParameterSymbolSlotBase = FieldSymbolSlotBase + 25;
const ParameterPackSymbolSlotBase = ParameterSymbolSlotBase + 6;
const TypeParameterSymbolSlotBase = ParameterPackSymbolSlotBase + 1;
const NonTypeParameterSymbolSlotBase = TypeParameterSymbolSlotBase + 1;
const TemplateTypeParameterSymbolSlotBase = NonTypeParameterSymbolSlotBase + 5;
const ConstraintTypeParameterSymbolSlotBase =
  TemplateTypeParameterSymbolSlotBase + 1;
const EnumeratorSymbolSlotBase = ConstraintTypeParameterSymbolSlotBase + 6;
const NamespaceAliasSymbolSlotBase = EnumeratorSymbolSlotBase + 1;
const UsingDeclarationSymbolSlotBase = NamespaceAliasSymbolSlotBase + 1;
const TypeSlotBase = 0;
const QualTypeSlotBase = TypeSlotBase + 1;
const BoundedArrayTypeSlotBase = QualTypeSlotBase + 4;
const UnboundedArrayTypeSlotBase = BoundedArrayTypeSlotBase + 2;
const PointerTypeSlotBase = UnboundedArrayTypeSlotBase + 1;
const LvalueReferenceTypeSlotBase = PointerTypeSlotBase + 1;
const RvalueReferenceTypeSlotBase = LvalueReferenceTypeSlotBase + 1;
const OverloadSetTypeSlotBase = RvalueReferenceTypeSlotBase + 1;
const FunctionTypeSlotBase = OverloadSetTypeSlotBase + 1;
const ClassTypeSlotBase = FunctionTypeSlotBase + 9;
const EnumTypeSlotBase = ClassTypeSlotBase + 4;
const ScopedEnumTypeSlotBase = EnumTypeSlotBase + 2;
const MemberObjectPointerTypeSlotBase = ScopedEnumTypeSlotBase + 2;
const MemberFunctionPointerTypeSlotBase = MemberObjectPointerTypeSlotBase + 2;
const NamespaceTypeSlotBase = MemberFunctionPointerTypeSlotBase + 2;
const TypeParameterTypeSlotBase = NamespaceTypeSlotBase + 1;
const TemplateTypeParameterTypeSlotBase = TypeParameterTypeSlotBase + 3;
const TemplateTypeParameterSpecializationTypeSlotBase =
  TemplateTypeParameterTypeSlotBase + 4;
const PackExpansionTypeSlotBase =
  TemplateTypeParameterSpecializationTypeSlotBase + 2;
const DecltypeTypeSlotBase = PackExpansionTypeSlotBase + 1;
const UnresolvedNameTypeSlotBase = DecltypeTypeSlotBase + 1;
const UnresolvedBoundedArrayTypeSlotBase = UnresolvedNameTypeSlotBase + 3;
const UnresolvedUnderlyingTypeSlotBase = UnresolvedBoundedArrayTypeSlotBase + 2;
const UnresolvedBuiltinTypeSlotBase = UnresolvedUnderlyingTypeSlotBase + 1;
const BitIntTypeSlotBase = UnresolvedBuiltinTypeSlotBase + 2;
const UnsignedBitIntTypeSlotBase = BitIntTypeSlotBase + 1;
const UnresolvedBitIntTypeSlotBase = UnsignedBitIntTypeSlotBase + 1;
const VectorTypeSlotBase = UnresolvedBitIntTypeSlotBase + 2;
const UnresolvedVectorTypeSlotBase = VectorTypeSlotBase + 3;
const ComplexTypeSlotBase = UnresolvedVectorTypeSlotBase + 4;
const AtomicTypeSlotBase = ComplexTypeSlotBase + 1;
const SveTypeSlotBase = AtomicTypeSlotBase + 1;
/** @category Constant Values */
export class DefaultInitializerContext extends ModelObject {}
/** @category Constant Values */
export class InitializerList extends ModelObject {}
/** @category Constant Values */
export class ConstComplex extends ModelObject {
  get real(): ConstValue {
    return decodeConstValue(
      cxx.readMiscVal(this.handle, ConstComplexSlotBase + 0),
      this.modelOwner,
    );
  }
  get imag(): ConstValue {
    return decodeConstValue(
      cxx.readMiscVal(this.handle, ConstComplexSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category Constant Values */
export class ConstObject extends ModelObject {
  get type(): Type | undefined {
    return typeOf(
      cxx.readMisc(this.handle, ConstObjectSlotBase + 0),
      this.modelOwner,
    );
  }
  get isConstexprUnknown(): boolean {
    return cxx.readMisc(this.handle, ConstObjectSlotBase + 1) !== 0;
  }
  get members(): Iterable<ConstObject_Member> {
    return miscValItems(
      this.modelOwner,
      this.handle,
      ConstObjectSlotBase + 2,
      (item: any) => decodeConstObject_Member(item, this.modelOwner),
    );
  }
  get isUnion(): boolean {
    return cxx.readMisc(this.handle, ConstObjectSlotBase + 3) !== 0;
  }
}
/** @category Constant Values */
export class Meta extends ModelObject {}
/** @category Constant Values */
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
  get denotesWholeOwner(): boolean {
    return cxx.readMisc(this.handle, ConstAddressSlotBase + 5) !== 0;
  }
}
/** @category Constant Values */
export class ConstLabelAddress extends ModelObject {
  get name(): string {
    return cxx.readMiscString(
      this.handle,
      ConstLabelAddressSlotBase + 0,
    ) as string;
  }
}
/** @category AST Nodes */
export abstract class AST extends ModelObject {
  readonly kind: ASTKind;
  constructor(handle: number, owner: ModelOwner, kind: ASTKind) {
    super(handle, owner);
    this.kind = kind;
  }
  get startLocation(): SourceLocation | undefined {
    return cxx.getStartLocation(this.handle, this.modelOwner.getUnitHandle());
  }
  get endLocation(): SourceLocation | undefined {
    return cxx.getEndLocation(this.handle, this.modelOwner.getUnitHandle());
  }
  get internalId(): number {
    return cxx.readAST(this.handle, ASTSlotBase + 0);
  }
  get firstSourceLocation(): SourceLocation | undefined {
    return Token.from(
      cxx.readAST(this.handle, ASTSlotBase + 1),
      this.modelOwner,
    )?.location;
  }
  get lastSourceLocation(): SourceLocation | undefined {
    return Token.from(
      cxx.readAST(this.handle, ASTSlotBase + 2),
      this.modelOwner,
    )?.location;
  }
}
/** @category AST Nodes */
export abstract class AttributeSpecifierAST extends AST {
  get attributes(): ReadonlyArray<Attribute> | undefined {
    return optionalOf(
      cxx.readASTVal(this.handle, AttributeSpecifierASTSlotBase + 0),
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeAttribute(element, this.modelOwner),
        ),
    );
  }
}
/** @category AST Nodes */
export abstract class AttributeTokenAST extends AST {}
/** @category AST Nodes */
export abstract class CoreDeclaratorAST extends AST {}
/** @category AST Nodes */
export abstract class DeclarationAST extends AST {}
/** @category AST Nodes */
export abstract class DeclaratorChunkAST extends AST {}
/** @category AST Nodes */
export abstract class DesignatorAST extends AST {}
/** @category AST Nodes */
export abstract class ExceptionDeclarationAST extends AST {}
/** @category AST Nodes */
export abstract class ExceptionSpecifierAST extends AST {}
/** @category AST Nodes */
export abstract class ExpressionAST extends AST {
  get valueCategory(): ValueCategory {
    return valueCategoryNames[
      cxx.readAST(this.handle, ExpressionASTSlotBase + 0)
    ]!;
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export abstract class FunctionBodyAST extends AST {}
/** @category AST Nodes */
export abstract class GenericAssociationAST extends AST {}
/** @category AST Nodes */
export abstract class LambdaCaptureAST extends AST {}
/** @category AST Nodes */
export abstract class MemInitializerAST extends AST {
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, MemInitializerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, MemInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export abstract class NestedNameSpecifierAST extends AST {
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NestedNameSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export abstract class NewInitializerAST extends AST {}
/** @category AST Nodes */
export abstract class PtrOperatorAST extends AST {}
/** @category AST Nodes */
export abstract class RequirementAST extends AST {}
/** @category AST Nodes */
export abstract class SpecifierAST extends AST {}
/** @category AST Nodes */
export abstract class StatementAST extends AST {}
/** @category AST Nodes */
export abstract class TemplateArgumentAST extends AST {}
/** @category AST Nodes */
export abstract class TemplateParameterAST extends AST {
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TemplateParameterASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, TemplateParameterASTSlotBase + 1);
  }
  get index(): number {
    return cxx.readAST(this.handle, TemplateParameterASTSlotBase + 2);
  }
}
/** @category AST Nodes */
export abstract class UnitAST extends AST {
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, UnitASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export abstract class UnqualifiedIdAST extends AST {}
/** @category AST Nodes */
export class TranslationUnitAST extends UnitAST {
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TranslationUnitASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class ModuleUnitAST extends UnitAST {
  get globalModuleFragment(): GlobalModuleFragmentAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get moduleDeclaration(): ModuleDeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get privateModuleFragment(): PrivateModuleFragmentAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleUnitASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SimpleDeclarationAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get initDeclaratorList(): Iterable<InitDeclaratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AsmDeclarationAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get asmQualifierList(): Iterable<AsmQualifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get asmToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get outputOperandList(): Iterable<AsmOperandAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get inputOperandList(): Iterable<AsmOperandAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get clobberList(): Iterable<AsmClobberAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get gotoLabelList(): Iterable<AsmGotoLabelAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmDeclarationASTSlotBase + 11),
      this.modelOwner,
      Literal,
    );
  }
}
/** @category AST Nodes */
export class NamespaceAliasDefinitionAST extends DeclarationAST {
  get namespaceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceAliasSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamespaceAliasDefinitionASTSlotBase + 7),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class UsingDeclarationAST extends DeclarationAST {
  get usingToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get usingDeclaratorList(): Iterable<UsingDeclaratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class UsingEnumDeclarationAST extends DeclarationAST {
  get usingToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get enumTypeSpecifier(): ElaboratedTypeSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingEnumDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class UsingDirectiveAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get usingToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get namespaceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingDirectiveASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class StaticAssertDeclarationAST extends DeclarationAST {
  get staticAssertToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 5),
      this.modelOwner,
      Literal,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StaticAssertDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get value(): boolean | undefined {
    return optionalOf(
      cxx.readASTVal(this.handle, StaticAssertDeclarationASTSlotBase + 8),
      (item: any) => item !== 0,
    );
  }
}
/** @category AST Nodes */
export class AliasDeclarationAST extends DeclarationAST {
  get usingToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get gnuAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, AliasDeclarationASTSlotBase + 8),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class OpaqueEnumDeclarationAST extends DeclarationAST {
  get enumToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get classToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, OpaqueEnumDeclarationASTSlotBase + 8),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class FunctionDefinitionAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get functionBody(): FunctionBodyAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, FunctionDefinitionASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TemplateDeclarationAST extends DeclarationAST {
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateParameterList(): Iterable<TemplateParameterAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get depth(): number {
    return cxx.readAST(this.handle, TemplateDeclarationASTSlotBase + 7);
  }
}
/** @category AST Nodes */
export class ConceptDefinitionAST extends DeclarationAST {
  get conceptToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): ConceptSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ConceptDefinitionASTSlotBase + 6),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DeductionGuideAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get explicitSpecifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get arrowToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get templateId(): SimpleTemplateIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get symbol(): DeductionGuideSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DeductionGuideASTSlotBase + 10),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ExplicitInstantiationAST extends DeclarationAST {
  get externToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExplicitInstantiationASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ExportDeclarationAST extends DeclarationAST {
  get exportToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExportDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExportDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ExportCompoundDeclarationAST extends DeclarationAST {
  get exportToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExportCompoundDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class LinkageSpecificationAST extends DeclarationAST {
  get externToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get stringliteralToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get stringLiteral(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, LinkageSpecificationASTSlotBase + 5),
      this.modelOwner,
      StringLiteral,
    );
  }
}
/** @category AST Nodes */
export class NamespaceDefinitionAST extends DeclarationAST {
  get inlineToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get namespaceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNamespaceSpecifierList(): Iterable<
    NestedNamespaceSpecifierAST | undefined
  > {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get extraAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 7),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get isInline(): boolean {
    return cxx.readAST(this.handle, NamespaceDefinitionASTSlotBase + 11) !== 0;
  }
}
/** @category AST Nodes */
export class EmptyDeclarationAST extends DeclarationAST {
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EmptyDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AttributeDeclarationAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AttributeDeclarationASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AttributeDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ModuleImportDeclarationAST extends DeclarationAST {
  get importToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get importName(): ImportNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleImportDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ParameterDeclarationAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get thisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): ParameterSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get isThisIntroduced(): boolean {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 9) !== 0;
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, ParameterDeclarationASTSlotBase + 10) !== 0;
  }
}
/** @category AST Nodes */
export class AccessDeclarationAST extends DeclarationAST {
  get accessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get accessSpecifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, AccessDeclarationASTSlotBase + 2)
    ]!;
  }
}
/** @category AST Nodes */
export class ForRangeDeclarationAST extends DeclarationAST {}
/** @category AST Nodes */
export class StructuredBindingDeclarationAST extends DeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get refQualifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get bindingList(): Iterable<NameIdAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get hiddenVariable(): InitDeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get bindingDeclaratorList(): Iterable<InitDeclaratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, StructuredBindingDeclarationASTSlotBase + 9),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class AsmOperandAST extends AST {
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get symbolicNameToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get constraintLiteralToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbolicName(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get constraintLiteral(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmOperandASTSlotBase + 8),
      this.modelOwner,
      Literal,
    );
  }
}
/** @category AST Nodes */
export class AsmQualifierAST extends AST {
  get qualifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmQualifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get qualifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, AsmQualifierASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class AsmClobberAST extends AST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmClobberASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get literal(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmClobberASTSlotBase + 1),
      this.modelOwner,
      StringLiteral,
    );
  }
}
/** @category AST Nodes */
export class AsmGotoLabelAST extends AST {
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmGotoLabelASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, AsmGotoLabelASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SplicerAST extends AST {
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SplicerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SplicerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SplicerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SplicerASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get secondColonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SplicerASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SplicerASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class GlobalModuleFragmentAST extends AST {
  get moduleToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GlobalModuleFragmentASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class PrivateModuleFragmentAST extends AST {
  get moduleToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get privateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PrivateModuleFragmentASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class ModuleDeclarationAST extends AST {
  get exportToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get moduleToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get moduleName(): ModuleNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get modulePartition(): ModulePartitionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleDeclarationASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ModuleNameAST extends AST {
  get moduleQualifier(): ModuleQualifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleNameASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleNameASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ModuleNameASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ModuleQualifierAST extends AST {
  get moduleQualifier(): ModuleQualifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get dotToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ModuleQualifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ModulePartitionAST extends AST {
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ModulePartitionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get moduleName(): ModuleNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ModulePartitionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ImportNameAST extends AST {
  get headerToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ImportNameASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get modulePartition(): ModulePartitionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ImportNameASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get moduleName(): ModuleNameAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ImportNameASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class InitDeclaratorAST extends AST {
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, InitDeclaratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DeclaratorAST extends AST {
  get ptrOpList(): Iterable<PtrOperatorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DeclaratorASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get coreDeclarator(): CoreDeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get declaratorChunkList(): Iterable<DeclaratorChunkAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DeclaratorASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class UsingDeclaratorAST extends AST {
  get typenameToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): UsingDeclarationSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, UsingDeclaratorASTSlotBase + 5) !== 0;
  }
}
/** @category AST Nodes */
export class EnumeratorAST extends AST {
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): EnumeratorSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, EnumeratorASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeIdAST extends AST {
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeIdASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeIdASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeIdASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, TypeIdASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class HandlerAST extends AST {
  get catchToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, HandlerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, HandlerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get exceptionDeclaration(): ExceptionDeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, HandlerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, HandlerASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, HandlerASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, HandlerASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BaseSpecifierAST extends AST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get virtualOrAccessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get otherVirtualOrAccessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 7) !== 0;
  }
  get isVirtual(): boolean {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 8) !== 0;
  }
  get isVariadic(): boolean {
    return cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 9) !== 0;
  }
  get accessSpecifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 10)
    ]!;
  }
  get symbol(): BaseClassSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BaseSpecifierASTSlotBase + 11),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class RequiresClauseAST extends AST {
  get requiresToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RequiresClauseASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RequiresClauseASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ParameterDeclarationClauseAST extends AST {
  get parameterDeclarationList(): Iterable<
    ParameterDeclarationAST | undefined
  > {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get functionParametersSymbol(): FunctionParametersSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isVariadic(): boolean {
    return (
      cxx.readAST(this.handle, ParameterDeclarationClauseASTSlotBase + 4) !== 0
    );
  }
}
/** @category AST Nodes */
export class TrailingReturnTypeAST extends AST {
  get minusGreaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TrailingReturnTypeASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TrailingReturnTypeASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class LambdaSpecifierAST extends AST {
  get specifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get specifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, LambdaSpecifierASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class TypeConstraintAST extends AST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): ConceptSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypeConstraintASTSlotBase + 6),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AttributeArgumentClauseAST extends AST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AttributeArgumentClauseASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AttributeAST extends AST {
  get attributeToken(): AttributeTokenAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AttributeASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeArgumentClause(): AttributeArgumentClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AttributeASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AttributeASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AttributeUsingPrefixAST extends AST {
  get usingToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeNamespaceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AttributeUsingPrefixASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NewPlacementAST extends AST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewPlacementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NewPlacementASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewPlacementASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NestedNamespaceSpecifierAST extends AST {
  get inlineToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isInline(): boolean {
    return (
      cxx.readAST(this.handle, NestedNamespaceSpecifierASTSlotBase + 5) !== 0
    );
  }
}
/** @category AST Nodes */
export class LabeledStatementAST extends StatementAST {
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LabeledStatementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LabeledStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LabeledStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, LabeledStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class CaseStatementAST extends StatementAST {
  get caseToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CaseStatementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CaseStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CaseStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get caseValue(): bigint {
    return cxx.readASTBigInt(
      this.handle,
      CaseStatementASTSlotBase + 3,
    ) as bigint;
  }
}
/** @category AST Nodes */
export class DefaultStatementAST extends StatementAST {
  get defaultToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DefaultStatementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DefaultStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ExpressionStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExpressionStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class CompoundStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get statementList(): Iterable<StatementAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, CompoundStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class IfStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, IfStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get ifToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get constexprToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get elseToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get elseStatement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, IfStatementASTSlotBase + 10),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ConstevalIfStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get ifToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get exclaimToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get constvalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get elseToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get elseStatement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get isNot(): boolean {
    return cxx.readAST(this.handle, ConstevalIfStatementASTSlotBase + 7) !== 0;
  }
}
/** @category AST Nodes */
export class SwitchStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get switchToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SwitchStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class WhileStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get whileToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, WhileStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DoStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DoStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get doToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get whileToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DoStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ForRangeStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get forToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rangeDeclaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rangeInitializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get beginInitializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get endInitializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get increment(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 12),
      this.modelOwner,
    );
  }
  get element(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 13),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 14),
      this.modelOwner,
    );
  }
  get rangeVariable(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 15),
      this.modelOwner,
    );
  }
  get beginVariable(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 16),
      this.modelOwner,
    );
  }
  get endVariable(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 17),
      this.modelOwner,
    );
  }
  get beginFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 18),
      this.modelOwner,
    );
  }
  get endFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 19),
      this.modelOwner,
    );
  }
  get derefFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 20),
      this.modelOwner,
    );
  }
  get incrementFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 21),
      this.modelOwner,
    );
  }
  get notEqualFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 22),
      this.modelOwner,
    );
  }
  get usesMemberBeginEnd(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 23) !== 0;
  }
  get isPointerIterator(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 24) !== 0;
  }
  get notEqualRewritten(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 25) !== 0;
  }
  get notEqualReversed(): boolean {
    return cxx.readAST(this.handle, ForRangeStatementASTSlotBase + 26) !== 0;
  }
}
/** @category AST Nodes */
export class ForStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ForStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get forToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get initializer(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get statement(): StatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get symbol(): BlockSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ForStatementASTSlotBase + 9),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BreakStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BreakStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get breakToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BreakStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BreakStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ContinueStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ContinueStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get continueToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ContinueStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ContinueStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ReturnStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ReturnStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get returnToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ReturnStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ReturnStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ReturnStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class CoroutineReturnStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get coreturnToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CoroutineReturnStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class GotoStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get gotoToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get starToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, GotoStatementASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get isIndirect(): boolean {
    return cxx.readAST(this.handle, GotoStatementASTSlotBase + 7) !== 0;
  }
}
/** @category AST Nodes */
export class DeclarationStatementAST extends StatementAST {
  get declaration(): DeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeclarationStatementASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TryBlockStatementAST extends StatementAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get tryToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get handlerList(): Iterable<HandlerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryBlockStatementASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class CharLiteralExpressionAST extends ExpressionAST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get literal(): CharLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 1),
      this.modelOwner,
      CharLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CharLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BoolLiteralExpressionAST extends ExpressionAST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get isTrue(): boolean {
    return cxx.readAST(this.handle, BoolLiteralExpressionASTSlotBase + 1) !== 0;
  }
}
/** @category AST Nodes */
export class IntLiteralExpressionAST extends ExpressionAST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get literal(): IntegerLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 1),
      this.modelOwner,
      IntegerLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IntLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class FloatLiteralExpressionAST extends ExpressionAST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get literal(): FloatLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 1),
      this.modelOwner,
      FloatLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FloatLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NullptrLiteralExpressionAST extends ExpressionAST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NullptrLiteralExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get literal(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, NullptrLiteralExpressionASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class StringLiteralExpressionAST extends ExpressionAST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get literal(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 1),
      this.modelOwner,
      StringLiteral,
    );
  }
  get encoding(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, StringLiteralExpressionASTSlotBase + 2)
    ]!;
  }
}
/** @category AST Nodes */
export class UserDefinedStringLiteralExpressionAST extends ExpressionAST {
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(
        this.handle,
        UserDefinedStringLiteralExpressionASTSlotBase + 0,
      ),
      this.modelOwner,
    );
  }
  get literal(): StringLiteral | undefined {
    return objOf(
      cxx.readAST(
        this.handle,
        UserDefinedStringLiteralExpressionASTSlotBase + 1,
      ),
      this.modelOwner,
      StringLiteral,
    );
  }
  get literalOperatorCall(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(
        this.handle,
        UserDefinedStringLiteralExpressionASTSlotBase + 2,
      ),
      this.modelOwner,
    );
  }
  get encoding(): TokenKind {
    return tokenKindNames[
      cxx.readAST(
        this.handle,
        UserDefinedStringLiteralExpressionASTSlotBase + 3,
      )
    ]!;
  }
}
/** @category AST Nodes */
export class ObjectLiteralExpressionAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ObjectLiteralExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ThisExpressionAST extends ExpressionAST {
  get thisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThisExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class PackIndexExpressionAST extends ExpressionAST {
  get packExpression(): IdExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get indexExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PackIndexExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class GenericSelectionExpressionAST extends ExpressionAST {
  get genericToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get genericAssociationList(): Iterable<GenericAssociationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get matchedAssocIndex(): number {
    return cxx.readAST(this.handle, GenericSelectionExpressionASTSlotBase + 6);
  }
}
/** @category AST Nodes */
export class NestedStatementExpressionAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedStatementExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DefaultInitializerExpressionAST extends ExpressionAST {
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DefaultInitializerExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get context(): DefaultInitializerContext {
    return objAt(
      cxx.readAST(this.handle, DefaultInitializerExpressionASTSlotBase + 1),
      this.modelOwner,
      DefaultInitializerContext,
    );
  }
}
/** @category AST Nodes */
export class NestedExpressionAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class IdExpressionAST extends ExpressionAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, IdExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, IdExpressionASTSlotBase + 4) !== 0;
  }
}
/** @category AST Nodes */
export class LambdaExpressionAST extends ExpressionAST {
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get captureDefaultToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get captureList(): Iterable<LambdaCaptureAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get templateParameterList(): Iterable<TemplateParameterAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get templateRequiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get expressionAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 11),
      this.modelOwner,
    );
  }
  get gnuAtributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 12),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lambdaSpecifierList(): Iterable<LambdaSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 13),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get exceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 14),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 15),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get trailingReturnType(): TrailingReturnTypeAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 16),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 17),
      this.modelOwner,
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 18),
      this.modelOwner,
    );
  }
  get captureDefault(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 19)
    ]!;
  }
  get symbol(): LambdaSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 20),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, LambdaExpressionASTSlotBase + 21),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class FoldExpressionAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get foldOpToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 7)
    ]!;
  }
  get foldOp(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, FoldExpressionASTSlotBase + 8)
    ]!;
  }
}
/** @category AST Nodes */
export class RightFoldExpressionAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, RightFoldExpressionASTSlotBase + 5)
    ]!;
  }
}
/** @category AST Nodes */
export class LeftFoldExpressionAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, LeftFoldExpressionASTSlotBase + 5)
    ]!;
  }
}
/** @category AST Nodes */
export class RequiresExpressionAST extends ExpressionAST {
  get requiresToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get requirementList(): Iterable<RequirementAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RequiresExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class VaArgExpressionAST extends ExpressionAST {
  get vaArgToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, VaArgExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SubscriptExpressionAST extends ExpressionAST {
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get indexExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, SubscriptExpressionASTSlotBase + 5) !== 0;
  }
}
/** @category AST Nodes */
export class CallExpressionAST extends ExpressionAST {
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, CallExpressionASTSlotBase + 4) !== 0;
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, CallExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeConstructionAST extends ExpressionAST {
  get typeSpecifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypeConstructionASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BracedTypeConstructionAST extends ExpressionAST {
  get typeSpecifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BracedTypeConstructionASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SpliceMemberExpressionAST extends ExpressionAST {
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get accessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get splicer(): SplicerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get accessOp(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 5)
    ]!;
  }
  get isTemplateIntroduced(): boolean {
    return (
      cxx.readAST(this.handle, SpliceMemberExpressionASTSlotBase + 6) !== 0
    );
  }
}
/** @category AST Nodes */
export class MemberExpressionAST extends ExpressionAST {
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get accessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get accessOp(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, MemberExpressionASTSlotBase + 6)
    ]!;
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, MemberExpressionASTSlotBase + 7) !== 0;
  }
}
/** @category AST Nodes */
export class PostIncrExpressionAST extends ExpressionAST {
  get baseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 2)
    ]!;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, PostIncrExpressionASTSlotBase + 4) !== 0;
  }
}
/** @category AST Nodes */
export class CppCastExpressionAST extends ExpressionAST {
  get castToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get castOp(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, CppCastExpressionASTSlotBase + 7)
    ]!;
  }
}
/** @category AST Nodes */
export class BuiltinBitCastExpressionAST extends ExpressionAST {
  get castToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinBitCastExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BuiltinConvertVectorExpressionAST extends ExpressionAST {
  get convertVectorToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinConvertVectorExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinConvertVectorExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinConvertVectorExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinConvertVectorExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinConvertVectorExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinConvertVectorExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BuiltinOffsetofExpressionAST extends ExpressionAST {
  get offsetofToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get designatorList(): Iterable<DesignatorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 5),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BuiltinOffsetofExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeidExpressionAST extends ExpressionAST {
  get typeidToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeidExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeidOfTypeExpressionAST extends ExpressionAST {
  get typeidToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeidOfTypeExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SpliceExpressionAST extends ExpressionAST {
  get splicer(): SplicerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SpliceExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class GlobalScopeReflectExpressionAST extends ExpressionAST {
  get caretCaretToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GlobalScopeReflectExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GlobalScopeReflectExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NamespaceReflectExpressionAST extends ExpressionAST {
  get caretCaretToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamespaceReflectExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeIdReflectExpressionAST extends ExpressionAST {
  get caretCaretToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeIdReflectExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ReflectExpressionAST extends ExpressionAST {
  get caretCaretToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ReflectExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class LabelAddressExpressionAST extends ExpressionAST {
  get ampAmpToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, LabelAddressExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class UnaryExpressionAST extends ExpressionAST {
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 2)
    ]!;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, UnaryExpressionASTSlotBase + 4) !== 0;
  }
}
/** @category AST Nodes */
export class AwaitExpressionAST extends ExpressionAST {
  get awaitToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AwaitExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SizeofExpressionAST extends ExpressionAST {
  get sizeofToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SizeofExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get value(): bigint | undefined {
    return cxx.readASTVal(this.handle, SizeofExpressionASTSlotBase + 2) as
      bigint | undefined;
  }
}
/** @category AST Nodes */
export class SizeofTypeExpressionAST extends ExpressionAST {
  get sizeofToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofTypeExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get value(): bigint | undefined {
    return cxx.readASTVal(this.handle, SizeofTypeExpressionASTSlotBase + 4) as
      bigint | undefined;
  }
}
/** @category AST Nodes */
export class SizeofPackExpressionAST extends ExpressionAST {
  get sizeofToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SizeofPackExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AlignofTypeExpressionAST extends ExpressionAST {
  get alignofToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignofTypeExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AlignofExpressionAST extends ExpressionAST {
  get alignofToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignofExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NoexceptExpressionAST extends ExpressionAST {
  get noexceptToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NoexceptExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get value(): boolean | undefined {
    return optionalOf(
      cxx.readASTVal(this.handle, NoexceptExpressionASTSlotBase + 4),
      (item: any) => item !== 0,
    );
  }
}
/** @category AST Nodes */
export class NewExpressionAST extends ExpressionAST {
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get newToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get newPlacement(): NewPlacementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get newInitalizer(): NewInitializerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get objectType(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NewExpressionASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get hasAlignmentArgument(): boolean {
    return cxx.readAST(this.handle, NewExpressionASTSlotBase + 11) !== 0;
  }
}
/** @category AST Nodes */
export class DeleteExpressionAST extends ExpressionAST {
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get deleteToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DeleteExpressionASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class CastExpressionAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CastExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CastExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CastExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ImplicitCastExpressionAST extends ExpressionAST {
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get castKind(): ImplicitCastKind {
    return implicitCastKindNames[
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 1)
    ]!;
  }
  get conversionFunction(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return (
      cxx.readAST(this.handle, ImplicitCastExpressionASTSlotBase + 3) !== 0
    );
  }
}
/** @category AST Nodes */
export class ConstExpressionAST extends ExpressionAST {
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get constValue(): ConstValue | undefined {
    return optionalOf(
      cxx.readASTVal(this.handle, ConstExpressionASTSlotBase + 1),
      (item: any) => decodeConstValue(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class BinaryExpressionAST extends ExpressionAST {
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 3)
    ]!;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, BinaryExpressionASTSlotBase + 5) !== 0;
  }
}
/** @category AST Nodes */
export class ConditionalExpressionAST extends ExpressionAST {
  get condition(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get questionToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get iftrueExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get iffalseExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionalExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class YieldExpressionAST extends ExpressionAST {
  get yieldToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, YieldExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, YieldExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ThrowExpressionAST extends ExpressionAST {
  get throwToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ThrowExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AssignmentExpressionAST extends ExpressionAST {
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 3)
    ]!;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return cxx.readAST(this.handle, AssignmentExpressionASTSlotBase + 5) !== 0;
  }
}
/** @category AST Nodes */
export class TargetExpressionAST extends ExpressionAST {}
/** @category AST Nodes */
export class RightExpressionAST extends ExpressionAST {}
/** @category AST Nodes */
export class CompoundAssignmentExpressionAST extends ExpressionAST {
  get targetExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get leftExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rightExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get adjustExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 5)
    ]!;
  }
  get symbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get isVirtualDispatch(): boolean {
    return (
      cxx.readAST(this.handle, CompoundAssignmentExpressionASTSlotBase + 7) !==
      0
    );
  }
}
/** @category AST Nodes */
export class PackExpansionExpressionAST extends ExpressionAST {
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PackExpansionExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DesignatedInitializerClauseAST extends ExpressionAST {
  get designatorList(): Iterable<DesignatorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DesignatedInitializerClauseASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeTraitExpressionAST extends ExpressionAST {
  get typeTraitToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeIdList(): Iterable<TypeIdAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get typeTrait(): BuiltinTypeTraitKind {
    return builtinTypeTraitKindNames[
      cxx.readAST(this.handle, TypeTraitExpressionASTSlotBase + 4)
    ]!;
  }
  get value(): boolean | undefined {
    return optionalOf(
      cxx.readASTVal(this.handle, TypeTraitExpressionASTSlotBase + 5),
      (item: any) => item !== 0,
    );
  }
}
/** @category AST Nodes */
export class ConditionExpressionAST extends ExpressionAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ConditionExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class EqualInitializerAST extends ExpressionAST {
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EqualInitializerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EqualInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BracedInitListAST extends ExpressionAST {
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BracedInitListASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BracedInitListASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BracedInitListASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BracedInitListASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get implicitElement(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, BracedInitListASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ParenInitializerAST extends ExpressionAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParenInitializerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParenInitializerASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParenInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ThreeWayComparisonExpressionAST extends ExpressionAST {
  get comparison(): BinaryExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lessResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get equalResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get greaterResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unorderedResult(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThreeWayComparisonExpressionASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DefaultGenericAssociationAST extends GenericAssociationAST {
  get defaultToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DefaultGenericAssociationASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeGenericAssociationAST extends GenericAssociationAST {
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeGenericAssociationASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DotDesignatorAST extends DesignatorAST {
  get dotToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DotDesignatorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DotDesignatorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, DotDesignatorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DotDesignatorASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SubscriptDesignatorAST extends DesignatorAST {
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SubscriptDesignatorASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TemplateTypeParameterAST extends TemplateParameterAST {
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateParameterList(): Iterable<TemplateParameterAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get requiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get classKeyToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 6),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 8),
      this.modelOwner,
    );
  }
  get idExpression(): IdExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return (
      cxx.readAST(this.handle, TemplateTypeParameterASTSlotBase + 11) !== 0
    );
  }
}
/** @category AST Nodes */
export class NonTypeTemplateParameterAST extends TemplateParameterAST {
  get declaration(): ParameterDeclarationAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NonTypeTemplateParameterASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypenameTypeParameterAST extends TemplateParameterAST {
  get classKeyToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 6) !== 0;
  }
  get isSynthesized(): boolean {
    return cxx.readAST(this.handle, TypenameTypeParameterASTSlotBase + 7) !== 0;
  }
}
/** @category AST Nodes */
export class ConstraintTypeParameterAST extends TemplateParameterAST {
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get isSynthesized(): boolean {
    return (
      cxx.readAST(this.handle, ConstraintTypeParameterASTSlotBase + 6) !== 0
    );
  }
}
/** @category AST Nodes */
export class TypedefSpecifierAST extends SpecifierAST {
  get typedefToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypedefSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class FriendSpecifierAST extends SpecifierAST {
  get friendToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FriendSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ConstevalSpecifierAST extends SpecifierAST {
  get constevalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstevalSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ConstinitSpecifierAST extends SpecifierAST {
  get constinitToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstinitSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ConstexprSpecifierAST extends SpecifierAST {
  get constexprToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstexprSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class InlineSpecifierAST extends SpecifierAST {
  get inlineToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, InlineSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NoreturnSpecifierAST extends SpecifierAST {
  get noreturnToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NoreturnSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class StaticSpecifierAST extends SpecifierAST {
  get staticToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, StaticSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ExternSpecifierAST extends SpecifierAST {
  get externToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExternSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class RegisterSpecifierAST extends SpecifierAST {
  get registerToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RegisterSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ThreadLocalSpecifierAST extends SpecifierAST {
  get threadLocalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThreadLocalSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ThreadSpecifierAST extends SpecifierAST {
  get threadToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThreadSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class MutableSpecifierAST extends SpecifierAST {
  get mutableToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, MutableSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class VirtualSpecifierAST extends SpecifierAST {
  get virtualToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, VirtualSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ExplicitSpecifierAST extends SpecifierAST {
  get explicitToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ExplicitSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AutoTypeSpecifierAST extends SpecifierAST {
  get autoToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AutoTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class VoidTypeSpecifierAST extends SpecifierAST {
  get voidToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, VoidTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SizeTypeSpecifierAST extends SpecifierAST {
  get specifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SizeTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get specifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, SizeTypeSpecifierASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class SignTypeSpecifierAST extends SpecifierAST {
  get specifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SignTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get specifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, SignTypeSpecifierASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class BuiltinTypeSpecifierAST extends SpecifierAST {
  get specifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BuiltinTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get specifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, BuiltinTypeSpecifierASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class UnaryBuiltinTypeSpecifierAST extends SpecifierAST {
  get builtinToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get builtinKind(): UnaryBuiltinTypeKind {
    return unaryBuiltinTypeKindNames[
      cxx.readAST(this.handle, UnaryBuiltinTypeSpecifierASTSlotBase + 4)
    ]!;
  }
}
/** @category AST Nodes */
export class BinaryBuiltinTypeSpecifierAST extends SpecifierAST {
  get builtinToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get leftTypeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rightTypeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get builtinKind(): BinaryBuiltinTypeKind {
    return binaryBuiltinTypeKindNames[
      cxx.readAST(this.handle, BinaryBuiltinTypeSpecifierASTSlotBase + 6)
    ]!;
  }
}
/** @category AST Nodes */
export class IntegralTypeSpecifierAST extends SpecifierAST {
  get specifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IntegralTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get specifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, IntegralTypeSpecifierASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class FloatingPointTypeSpecifierAST extends SpecifierAST {
  get specifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FloatingPointTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get specifier(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, FloatingPointTypeSpecifierASTSlotBase + 1)
    ]!;
  }
}
/** @category AST Nodes */
export class ComplexTypeSpecifierAST extends SpecifierAST {
  get complexToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ComplexTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NamedTypeSpecifierAST extends SpecifierAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 3) !== 0;
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, NamedTypeSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AtomicTypeSpecifierAST extends SpecifierAST {
  get atomicToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AtomicTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BitIntTypeSpecifierAST extends SpecifierAST {
  get bitintToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get sizeExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get bitCount(): number {
    return cxx.readAST(this.handle, BitIntTypeSpecifierASTSlotBase + 4);
  }
}
/** @category AST Nodes */
export class UnderlyingTypeSpecifierAST extends SpecifierAST {
  get underlyingTypeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, UnderlyingTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ElaboratedTypeSpecifierAST extends SpecifierAST {
  get classToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get classKey(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 5)
    ]!;
  }
  get isTemplateIntroduced(): boolean {
    return (
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 6) !== 0
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ElaboratedTypeSpecifierASTSlotBase + 7),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DecltypeAutoSpecifierAST extends SpecifierAST {
  get decltypeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get autoToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeAutoSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DecltypeSpecifierAST extends SpecifierAST {
  get decltypeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get type(): Type | undefined {
    return typeOf(
      cxx.readAST(this.handle, DecltypeSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class PlaceholderTypeSpecifierAST extends SpecifierAST {
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PlaceholderTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get specifier(): SpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PlaceholderTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ConstQualifierAST extends SpecifierAST {
  get constToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConstQualifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class VolatileQualifierAST extends SpecifierAST {
  get volatileToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, VolatileQualifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AtomicQualifierAST extends SpecifierAST {
  get atomicToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AtomicQualifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class RestrictQualifierAST extends SpecifierAST {
  get restrictToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RestrictQualifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class EnumSpecifierAST extends SpecifierAST {
  get enumToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get classToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get enumeratorList(): Iterable<EnumeratorAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get commaToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 10),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, EnumSpecifierASTSlotBase + 11),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ClassSpecifierAST extends SpecifierAST {
  get classToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get finalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get baseSpecifierList(): Iterable<BaseSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get declarationList(): Iterable<DeclarationAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 8),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 9),
      this.modelOwner,
    );
  }
  get trailingAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 10),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get classKey(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, ClassSpecifierASTSlotBase + 11)
    ]!;
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
}
/** @category AST Nodes */
export class TypenameSpecifierAST extends SpecifierAST {
  get typenameToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 4) !== 0;
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypenameSpecifierASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SplicerTypeSpecifierAST extends SpecifierAST {
  get typenameToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SplicerTypeSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get splicer(): SplicerAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SplicerTypeSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class PointerOperatorAST extends PtrOperatorAST {
  get starToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PointerOperatorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PointerOperatorASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get cvQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PointerOperatorASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class ReferenceOperatorAST extends PtrOperatorAST {
  get refToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get refOp(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, ReferenceOperatorASTSlotBase + 2)
    ]!;
  }
}
/** @category AST Nodes */
export class PtrToMemberOperatorAST extends PtrOperatorAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get starToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get cvQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, PtrToMemberOperatorASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class BitfieldDeclaratorAST extends CoreDeclaratorAST {
  get unqualifiedId(): NameIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get sizeExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get trailingAttributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, BitfieldDeclaratorASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class ParameterPackAST extends CoreDeclaratorAST {
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParameterPackASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get coreDeclarator(): CoreDeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParameterPackASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class IdDeclaratorAST extends CoreDeclaratorAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get isTemplateIntroduced(): boolean {
    return cxx.readAST(this.handle, IdDeclaratorASTSlotBase + 4) !== 0;
  }
}
/** @category AST Nodes */
export class NestedDeclaratorAST extends CoreDeclaratorAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedDeclaratorASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class FunctionDeclaratorChunkAST extends DeclaratorChunkAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get parameterDeclarationClause(): ParameterDeclarationClauseAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get cvQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get refToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get exceptionSpecifier(): ExceptionSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 6),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get trailingReturnType(): TrailingReturnTypeAST | undefined {
    return astOf(
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 7),
      this.modelOwner,
    );
  }
  get refOp(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 8)
    ]!;
  }
  get isFinal(): boolean {
    return (
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 9) !== 0
    );
  }
  get isOverride(): boolean {
    return (
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 10) !== 0
    );
  }
  get isPure(): boolean {
    return (
      cxx.readAST(this.handle, FunctionDeclaratorChunkASTSlotBase + 11) !== 0
    );
  }
}
/** @category AST Nodes */
export class ArrayDeclaratorChunkAST extends DeclaratorChunkAST {
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeQualifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ArrayDeclaratorChunkASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class NameIdAST extends UnqualifiedIdAST {
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NameIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, NameIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DestructorIdAST extends UnqualifiedIdAST {
  get tildeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DestructorIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get id(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DestructorIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DecltypeIdAST extends UnqualifiedIdAST {
  get decltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DecltypeIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class OperatorFunctionIdAST extends UnqualifiedIdAST {
  get operatorToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get opToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get openToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get closeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get op(): TokenKind {
    return tokenKindNames[
      cxx.readAST(this.handle, OperatorFunctionIdASTSlotBase + 4)
    ]!;
  }
}
/** @category AST Nodes */
export class LiteralOperatorIdAST extends UnqualifiedIdAST {
  get operatorToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 3),
      this.modelOwner,
      Literal,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, LiteralOperatorIdASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ConversionFunctionIdAST extends UnqualifiedIdAST {
  get operatorToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ConversionFunctionIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ConversionFunctionIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SimpleTemplateIdAST extends UnqualifiedIdAST {
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SimpleTemplateIdASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class LiteralOperatorTemplateIdAST extends UnqualifiedIdAST {
  get literalOperatorId(): LiteralOperatorIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, LiteralOperatorTemplateIdASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class OperatorFunctionTemplateIdAST extends UnqualifiedIdAST {
  get operatorFunctionId(): OperatorFunctionIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lessToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateArgumentList(): Iterable<TemplateArgumentAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get greaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, OperatorFunctionTemplateIdASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class GlobalNestedNameSpecifierAST extends NestedNameSpecifierAST {
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GlobalNestedNameSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SimpleNestedNameSpecifierAST extends NestedNameSpecifierAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleNestedNameSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DecltypeNestedNameSpecifierAST extends NestedNameSpecifierAST {
  get decltypeSpecifier(): DecltypeSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DecltypeNestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TemplateNestedNameSpecifierAST extends NestedNameSpecifierAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get templateToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateId(): SimpleTemplateIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get isTemplateIntroduced(): boolean {
    return (
      cxx.readAST(this.handle, TemplateNestedNameSpecifierASTSlotBase + 4) !== 0
    );
  }
}
/** @category AST Nodes */
export class DefaultFunctionBodyAST extends FunctionBodyAST {
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get defaultToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DefaultFunctionBodyASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class CompoundStatementFunctionBodyAST extends FunctionBodyAST {
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundStatementFunctionBodyASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get memInitializerList(): Iterable<MemInitializerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CompoundStatementFunctionBodyASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundStatementFunctionBodyASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TryStatementFunctionBodyAST extends FunctionBodyAST {
  get tryToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get colonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get memInitializerList(): Iterable<MemInitializerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 2),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get statement(): CompoundStatementAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get handlerList(): Iterable<HandlerAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TryStatementFunctionBodyASTSlotBase + 4),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
}
/** @category AST Nodes */
export class DeleteFunctionBodyAST extends FunctionBodyAST {
  get equalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get deleteToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DeleteFunctionBodyASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeTemplateArgumentAST extends TemplateArgumentAST {
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeTemplateArgumentASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ExpressionTemplateArgumentAST extends TemplateArgumentAST {
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ExpressionTemplateArgumentASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ThrowExceptionSpecifierAST extends ExceptionSpecifierAST {
  get throwToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThrowExceptionSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NoexceptSpecifierAST extends ExceptionSpecifierAST {
  get noexceptToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NoexceptSpecifierASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SimpleRequirementAST extends RequirementAST {
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleRequirementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleRequirementASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class CompoundRequirementAST extends RequirementAST {
  get lbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get rbraceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get noexceptToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get minusGreaterToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 5),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CompoundRequirementASTSlotBase + 6),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeRequirementAST extends RequirementAST {
  get typenameToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeRequirementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeRequirementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, TypeRequirementASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NestedRequirementAST extends RequirementAST {
  get requiresToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedRequirementASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NestedRequirementASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get semicolonToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NestedRequirementASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NewParenInitializerAST extends NewInitializerAST {
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, NewParenInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class NewBracedInitializerAST extends NewInitializerAST {
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, NewBracedInitializerASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ParenMemInitializerAST extends MemInitializerAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get expressionList(): Iterable<ExpressionAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ParenMemInitializerASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class BracedMemInitializerAST extends MemInitializerAST {
  get nestedNameSpecifier(): NestedNameSpecifierAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get unqualifiedId(): UnqualifiedIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get bracedInitList(): BracedInitListAST | undefined {
    return astOf(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, BracedMemInitializerASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class ThisLambdaCaptureAST extends LambdaCaptureAST {
  get thisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, ThisLambdaCaptureASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class DerefThisLambdaCaptureAST extends LambdaCaptureAST {
  get starToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get thisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, DerefThisLambdaCaptureASTSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SimpleLambdaCaptureAST extends LambdaCaptureAST {
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, SimpleLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class RefLambdaCaptureAST extends LambdaCaptureAST {
  get ampToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, RefLambdaCaptureASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class RefInitLambdaCaptureAST extends LambdaCaptureAST {
  get ampToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, RefInitLambdaCaptureASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class InitLambdaCaptureAST extends LambdaCaptureAST {
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get symbol(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, InitLambdaCaptureASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class EllipsisExceptionDeclarationAST extends ExceptionDeclarationAST {
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, EllipsisExceptionDeclarationASTSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class TypeExceptionDeclarationAST extends ExceptionDeclarationAST {
  get attributeList(): Iterable<AttributeSpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 0),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get typeSpecifierList(): Iterable<SpecifierAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 1),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get declarator(): DeclaratorAST | undefined {
    return astOf(
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get symbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readAST(this.handle, TypeExceptionDeclarationASTSlotBase + 3),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class CxxAttributeAST extends AttributeSpecifierAST {
  get lbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lbracket2Token(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get attributeUsingPrefix(): AttributeUsingPrefixAST | undefined {
    return astOf(
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rbracketToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rbracket2Token(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, CxxAttributeASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class GccAttributeAST extends AttributeSpecifierAST {
  get attributeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GccAttributeASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GccAttributeASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get lparen2Token(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GccAttributeASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get attributeList(): Iterable<AttributeAST | undefined> {
    return listOf(
      this.modelOwner,
      cxx.readAST(this.handle, GccAttributeASTSlotBase + 3),
      (item: any) => astOf(item, this.modelOwner),
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GccAttributeASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get rparen2Token(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, GccAttributeASTSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class AlignasAttributeAST extends AttributeSpecifierAST {
  get alignasToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, AlignasAttributeASTSlotBase + 5) !== 0;
  }
}
/** @category AST Nodes */
export class AlignasTypeAttributeAST extends AttributeSpecifierAST {
  get alignasToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get ellipsisToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 4),
      this.modelOwner,
    );
  }
  get isPack(): boolean {
    return cxx.readAST(this.handle, AlignasTypeAttributeASTSlotBase + 5) !== 0;
  }
}
/** @category AST Nodes */
export class AsmAttributeAST extends AttributeSpecifierAST {
  get asmToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmAttributeASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get lparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmAttributeASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get literalToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmAttributeASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get rparenToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, AsmAttributeASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get literal(): Literal | undefined {
    return objOf(
      cxx.readAST(this.handle, AsmAttributeASTSlotBase + 4),
      this.modelOwner,
      Literal,
    );
  }
}
/** @category AST Nodes */
export class ScopedAttributeTokenAST extends AttributeTokenAST {
  get attributeNamespaceToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get scopeToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 1),
      this.modelOwner,
    );
  }
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 2),
      this.modelOwner,
    );
  }
  get attributeNamespace(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 3),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, ScopedAttributeTokenASTSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category AST Nodes */
export class SimpleAttributeTokenAST extends AttributeTokenAST {
  get identifierToken(): Token | undefined {
    return Token.from(
      cxx.readAST(this.handle, SimpleAttributeTokenASTSlotBase + 0),
      this.modelOwner,
    );
  }
  get identifier(): Identifier | undefined {
    return nameOf(
      cxx.readAST(this.handle, SimpleAttributeTokenASTSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category Literals */
export class Literal extends ModelObject {
  get value(): string {
    return cxx.readLiteralString(this.handle, LiteralSlotBase + 0) as string;
  }
  get hashCode(): number {
    return cxx.readLiteral(this.handle, LiteralSlotBase + 1);
  }
}
/** @category Literals */
export class IntegerLiteral extends Literal {
  get integerValue(): bigint {
    return cxx.readLiteralBigInt(
      this.handle,
      IntegerLiteralSlotBase + 0,
    ) as bigint;
  }
  get components(): IntegerLiteral_Components {
    return decodeIntegerLiteral_Components(
      cxx.readLiteralVal(this.handle, IntegerLiteralSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category Literals */
export class FloatLiteral extends Literal {
  get floatValue(): number {
    return cxx.readLiteral(this.handle, FloatLiteralSlotBase + 0);
  }
  get components(): FloatLiteral_Components {
    return decodeFloatLiteral_Components(
      cxx.readLiteralVal(this.handle, FloatLiteralSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category Literals */
export class StringLiteral extends Literal {
  get encoding(): StringLiteralEncoding {
    return stringLiteralEncodingNames[
      cxx.readLiteral(this.handle, StringLiteralSlotBase + 0)
    ]!;
  }
  get isRaw(): boolean {
    return cxx.readLiteral(this.handle, StringLiteralSlotBase + 1) !== 0;
  }
  get stringValue(): string {
    return cxx.readLiteralString(
      this.handle,
      StringLiteralSlotBase + 2,
    ) as string;
  }
  get codeUnitSize(): number {
    return cxx.readLiteral(this.handle, StringLiteralSlotBase + 3);
  }
  get charCount(): number {
    return cxx.readLiteral(this.handle, StringLiteralSlotBase + 4);
  }
  get components(): StringLiteral_Components {
    return decodeStringLiteral_Components(
      cxx.readLiteralVal(this.handle, StringLiteralSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category Literals */
export class CharLiteral extends Literal {
  get charValue(): number {
    return cxx.readLiteral(this.handle, CharLiteralSlotBase + 0);
  }
  get components(): CharLiteral_Components {
    return cxx.readLiteralVal(
      this.handle,
      CharLiteralSlotBase + 1,
    ) as CharLiteral_Components;
  }
}
/** @category Literals */
export class CommentLiteral extends Literal {}
/** @category Names */
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
/** @category Names */
export class Identifier extends Name {
  get isAnonymous(): boolean {
    return cxx.readName(this.handle, IdentifierSlotBase + 0) !== 0;
  }
  get name(): string {
    return cxx.readNameString(this.handle, IdentifierSlotBase + 1) as string;
  }
  get value(): string {
    return cxx.readNameString(this.handle, IdentifierSlotBase + 2) as string;
  }
  get isBuiltinTypeTrait(): boolean {
    return cxx.readName(this.handle, IdentifierSlotBase + 3) !== 0;
  }
  get builtinTypeTrait(): BuiltinTypeTraitKind {
    return builtinTypeTraitKindNames[
      cxx.readName(this.handle, IdentifierSlotBase + 4)
    ]!;
  }
  get builtinFunction(): BuiltinFunctionKind {
    return builtinFunctionKindNames[
      cxx.readName(this.handle, IdentifierSlotBase + 5)
    ]!;
  }
  get builtinTemplate(): BuiltinTemplateKind {
    return builtinTemplateKindNames[
      cxx.readName(this.handle, IdentifierSlotBase + 6)
    ]!;
  }
  get wellKnownName(): WellKnownName {
    return wellKnownNameNames[
      cxx.readName(this.handle, IdentifierSlotBase + 7)
    ]!;
  }
}
/** @category Names */
export class OperatorId extends Name {
  get op(): TokenKind {
    return tokenKindNames[cxx.readName(this.handle, OperatorIdSlotBase + 0)]!;
  }
}
/** @category Names */
export class DestructorId extends Name {
  get name(): Name | undefined {
    return nameOf(
      cxx.readName(this.handle, DestructorIdSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Names */
export class LiteralOperatorId extends Name {
  get name(): string {
    return cxx.readNameString(
      this.handle,
      LiteralOperatorIdSlotBase + 0,
    ) as string;
  }
}
/** @category Names */
export class ConversionFunctionId extends Name {
  get type(): Type | undefined {
    return typeOf(
      cxx.readName(this.handle, ConversionFunctionIdSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Names */
export class TemplateId extends Name {
  get name(): Name | undefined {
    return nameOf(
      cxx.readName(this.handle, TemplateIdSlotBase + 0),
      this.modelOwner,
    );
  }
  get arguments(): Iterable<TemplateArgument> {
    return nameValItems(
      this.modelOwner,
      this.handle,
      TemplateIdSlotBase + 1,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
}
/** @category Symbols */
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
  get location(): SourceLocation | undefined {
    return Token.from(
      cxx.readSymbol(this.handle, SymbolSlotBase + 2),
      this.modelOwner,
    )?.location;
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
    return accessSpecifierNames[
      cxx.readSymbol(this.handle, SymbolSlotBase + 10)
    ]!;
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
    return optionalOf(
      cxx.readSymbolVal(this.handle, SymbolSlotBase + 12),
      (item: any) =>
        (item as any[]).map((element: any) => nameOf(element, this.modelOwner)),
    );
  }
  get isWeak(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 13) !== 0;
  }
  get isNoInline(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 14) !== 0;
  }
  get isAlwaysInline(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 15) !== 0;
  }
  get attributes(): ReadonlyArray<Attribute> | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, SymbolSlotBase + 16),
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeAttribute(element, this.modelOwner),
        ),
    );
  }
  get isNodiscard(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 17) !== 0;
  }
  get isUsed(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 18) !== 0;
  }
  get isExcludedFromExplicitInstantiation(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 19) !== 0;
  }
  get isTrivialAbi(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 20) !== 0;
  }
  get hasDeducedReturnType(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 21) !== 0;
  }
  get canonical(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 22),
      this.modelOwner,
    );
  }
  get definition(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 23),
      this.modelOwner,
    );
  }
  get instantiationPattern(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, SymbolSlotBase + 24),
      this.modelOwner,
    );
  }
  get isNamespace(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 25) !== 0;
  }
  get isNamespaceAlias(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 26) !== 0;
  }
  get isConcept(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 27) !== 0;
  }
  get isDeductionGuide(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 28) !== 0;
  }
  get isClass(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 29) !== 0;
  }
  get isEnum(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 30) !== 0;
  }
  get isScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 31) !== 0;
  }
  get isFunction(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 32) !== 0;
  }
  get isTypeAlias(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 33) !== 0;
  }
  get isVariable(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 34) !== 0;
  }
  get isField(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 35) !== 0;
  }
  get isParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 36) !== 0;
  }
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 37) !== 0;
  }
  get isEnumerator(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 38) !== 0;
  }
  get isFunctionParameters(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 39) !== 0;
  }
  get isTemplateParameters(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 40) !== 0;
  }
  get isBlock(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 41) !== 0;
  }
  get isLambda(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 42) !== 0;
  }
  get isTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 43) !== 0;
  }
  get isNonTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 44) !== 0;
  }
  get isTemplateTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 45) !== 0;
  }
  get isConstraintTypeParameter(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 46) !== 0;
  }
  get isOverloadSet(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 47) !== 0;
  }
  get isBaseClass(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 48) !== 0;
  }
  get isInjectedClassName(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 49) !== 0;
  }
  get isUnresolved(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 50) !== 0;
  }
  get isUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 51) !== 0;
  }
  get isClassOrNamespace(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 52) !== 0;
  }
  get isNamespaceName(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 53) !== 0;
  }
  get isEnumOrScopedEnum(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 54) !== 0;
  }
  get internalId(): number {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 55);
  }
  get text(): string {
    return cxx.readSymbolString(this.handle, SymbolSlotBase + 56) as string;
  }
  get isType(): boolean {
    return cxx.readSymbol(this.handle, SymbolSlotBase + 57) !== 0;
  }
}
/** @category Symbols */
export abstract class ScopeSymbol extends Symbol {
  get empty(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 0) !== 0;
  }
  get members(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ScopeSymbolSlotBase + 1,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDirectives(): ReadonlyArray<ScopeSymbol | undefined> {
    return (
      cxx.readSymbolVal(this.handle, ScopeSymbolSlotBase + 2) as any[]
    ).map((element: any) => symbolOf(element, this.modelOwner));
  }
  get isTransparent(): boolean {
    return cxx.readSymbol(this.handle, ScopeSymbolSlotBase + 3) !== 0;
  }
}
/** @category Symbols */
export class NamespaceSymbol extends ScopeSymbol {
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 0) !== 0;
  }
  get hasInlineNamespaces(): boolean {
    return cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 1) !== 0;
  }
  get unnamedNamespace(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get anonNamespaceIndex(): number | undefined {
    return cxx.readSymbolVal(this.handle, NamespaceSymbolSlotBase + 3) as
      number | undefined;
  }
}
/** @category Symbols */
export class ConceptSymbol extends Symbol {
  get declaration(): ConceptDefinitionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get primaryTemplateSymbol(): ConceptSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 4);
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 5) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, ConceptSymbolSlotBase + 6) !== 0;
  }
  get templateArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ConceptSymbolSlotBase + 7,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<TemplateArgument>
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ConceptSymbolSlotBase + 8,
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeTemplateArgument(element, this.modelOwner),
        ),
    );
  }
}
/** @category Symbols */
export class DeductionGuideSymbol extends Symbol {
  get declaration(): DeductionGuideAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get primaryTemplateSymbol(): DeductionGuideSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 4);
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 5) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 6) !== 0;
  }
  get templateArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      DeductionGuideSymbolSlotBase + 7,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<TemplateArgument>
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      DeductionGuideSymbolSlotBase + 8,
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeTemplateArgument(element, this.modelOwner),
        ),
    );
  }
  get isExplicit(): boolean {
    return cxx.readSymbol(this.handle, DeductionGuideSymbolSlotBase + 9) !== 0;
  }
}
/** @category Symbols */
export class BaseClassSymbol extends Symbol {
  get isVirtual(): boolean {
    return cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 0) !== 0;
  }
  get symbol(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, BaseClassSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class InjectedClassNameSymbol extends Symbol {
  get classSymbol(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, InjectedClassNameSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class UnresolvedSymbol extends Symbol {}
/** @category Symbols */
export class ClassSymbol extends ScopeSymbol {
  get canonical(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get definition(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get declaration(): SpecifierAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get primaryTemplateSymbol(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 6);
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 7) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 8) !== 0;
  }
  get templateArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 9,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<TemplateArgument>
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 10,
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeTemplateArgument(element, this.modelOwner),
        ),
    );
  }
  get canonicalOrNull(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 11),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 12),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<ClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 13,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declarations(): Iterable<ClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 14,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get instantiationSubstitutionDepth(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 15);
  }
  get instantiationSubstitutionArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 16,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get isUnion(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 17) !== 0;
  }
  get baseClasses(): Iterable<BaseClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 18,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get constructors(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 19,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declaredConstructors(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 20,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get constructorOverloadSet(): OverloadSetSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 21),
      this.modelOwner,
    );
  }
  get deductionGuides(): Iterable<DeductionGuideSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 22,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get conversionFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 23,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get implicitConversionFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 24,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get visibleConversionFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 25,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get destructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 26),
      this.modelOwner,
    );
  }
  get defaultConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 27),
      this.modelOwner,
    );
  }
  get copyConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 28),
      this.modelOwner,
    );
  }
  get moveConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 29),
      this.modelOwner,
    );
  }
  get copyAssignmentOperator(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 30),
      this.modelOwner,
    );
  }
  get moveAssignmentOperator(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 31),
      this.modelOwner,
    );
  }
  get hasUserDeclaredConstructors(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 32) !== 0;
  }
  get hasInheritedConstructors(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 33) !== 0;
  }
  get hasVirtualFunctions(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 34) !== 0;
  }
  get hasVirtualBaseClasses(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 35) !== 0;
  }
  get convertingConstructors(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 36,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get isFinal(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 37) !== 0;
  }
  get isComplete(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 38) !== 0;
  }
  get isFriend(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 39) !== 0;
  }
  get isPolymorphic(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 40) !== 0;
  }
  get isAbstract(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 41) !== 0;
  }
  get hasVirtualDestructor(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 42) !== 0;
  }
  get isAccessControlDisabled(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 43) !== 0;
  }
  get sizeInBytes(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 44);
  }
  get alignment(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 45);
  }
  get explicitAlignment(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 46);
  }
  get packAlignment(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 47);
  }
  get befriendingClasses(): Iterable<ClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 48,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get templateFriendships(): Iterable<TemplateFriendship> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 49,
      (item: any) => decodeTemplateFriendship(item, this.modelOwner),
    );
  }
  get baseClassRepetition(): ClassSymbol_BaseClassRepetition {
    return decodeClassSymbol_BaseClassRepetition(
      cxx.readSymbolVal(this.handle, ClassSymbolSlotBase + 50),
      this.modelOwner,
    );
  }
  get flags(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 51);
  }
  get hasVirtualBaseSubobjects(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 52) !== 0;
  }
  get vtableLayout(): VTableLayout | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, ClassSymbolSlotBase + 53),
      (item: any) => decodeVTableLayout(item, this.modelOwner),
    );
  }
  get isClosureType(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 54) !== 0;
  }
  get hasLambdaCapture(): boolean {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 55) !== 0;
  }
  get capturedThisField(): FieldSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 56),
      this.modelOwner,
    );
  }
  get functionCallOperator(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 57),
      this.modelOwner,
    );
  }
  get closureDiscriminator(): number {
    return cxx.readSymbol(this.handle, ClassSymbolSlotBase + 58);
  }
  get instantiationTemplate(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 59),
      this.modelOwner,
    );
  }
  get templatePattern(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ClassSymbolSlotBase + 60),
      this.modelOwner,
    );
  }
  get expandedTemplateArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 61,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get expandedTemplateArgumentTexts(): Iterable<string> {
    return symbolStringItems(
      this.modelOwner,
      this.handle,
      ClassSymbolSlotBase + 62,
      (item: any) => item,
    );
  }
}
/** @category Symbols */
export class EnumSymbol extends ScopeSymbol {
  get hasFixedUnderlyingType(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 0) !== 0;
  }
  get isDefined(): boolean {
    return cxx.readSymbol(this.handle, EnumSymbolSlotBase + 1) !== 0;
  }
  get underlyingType(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, EnumSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class ScopedEnumSymbol extends ScopeSymbol {
  get underlyingType(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get isDefined(): boolean {
    return cxx.readSymbol(this.handle, ScopedEnumSymbolSlotBase + 1) !== 0;
  }
}
/** @category Symbols */
export class FunctionSymbol extends ScopeSymbol {
  get canonical(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get definition(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get declaration(): FunctionDefinitionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get primaryTemplateSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 6);
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 7) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 8) !== 0;
  }
  get templateArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 9,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<TemplateArgument>
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 10,
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeTemplateArgument(element, this.modelOwner),
        ),
    );
  }
  get canonicalOrNull(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 11),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 12),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 13,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declarations(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 14,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get functionParameters(): FunctionParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 15),
      this.modelOwner,
    );
  }
  get isDefined(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 16) !== 0;
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 17) !== 0;
  }
  get isExtern(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 18) !== 0;
  }
  get isFriend(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 19) !== 0;
  }
  get isImplicitObjectMemberFunction(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 20) !== 0;
  }
  get isNonStaticMemberFunction(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 21) !== 0;
  }
  get hasImplicitObjectParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 22) !== 0;
  }
  get hasExplicitObjectParameter(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 23) !== 0;
  }
  get explicitObjectParameter(): ParameterSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 24),
      this.modelOwner,
    );
  }
  get parameters(): Iterable<ParameterSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 25,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 26) !== 0;
  }
  get isConsteval(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 27) !== 0;
  }
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 28) !== 0;
  }
  get isVirtual(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 29) !== 0;
  }
  get isExplicit(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 30) !== 0;
  }
  get isDeleted(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 31) !== 0;
  }
  get isDefaulted(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 32) !== 0;
  }
  get isPure(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 33) !== 0;
  }
  get isOverride(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 34) !== 0;
  }
  get isFinal(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 35) !== 0;
  }
  get hasNoPrototype(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 36) !== 0;
  }
  get hasExceptionSpecifier(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 37) !== 0;
  }
  get isDefinitionRequired(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 38) !== 0;
  }
  get isNoReturn(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 39) !== 0;
  }
  get builtinKind(): BuiltinFunctionKind {
    return builtinFunctionKindNames[
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 40)
    ]!;
  }
  get trailingRequiresClause(): RequiresClauseAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 41),
      this.modelOwner,
    );
  }
  get isConstructor(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 42) !== 0;
  }
  get isDestructor(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 43) !== 0;
  }
  get languageLinkage(): LanguageKind {
    return languageKindNames[
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 44)
    ]!;
  }
  get hasCLinkage(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 45) !== 0;
  }
  get externalName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 46),
      this.modelOwner,
    );
  }
  get aliasName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 47),
      this.modelOwner,
    );
  }
  get hasHiddenVisibility(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 48) !== 0;
  }
  get importModule(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 49),
      this.modelOwner,
    );
  }
  get importName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 50),
      this.modelOwner,
    );
  }
  get exportName(): Identifier | undefined {
    return nameOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 51),
      this.modelOwner,
    );
  }
  get hasPendingBody(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 52) !== 0;
  }
  get hasUninstantiatedBody(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 53) !== 0;
  }
  get pendingBody(): PendingInstantiation | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, FunctionSymbolSlotBase + 54),
      (item: any) => decodePendingInstantiation(item, this.modelOwner),
    );
  }
  get hasDeferredImplicitExceptionSpecification(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 55) !== 0;
  }
  get pendingExceptionSpecification(): PendingInstantiation | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, FunctionSymbolSlotBase + 56),
      (item: any) => decodePendingInstantiation(item, this.modelOwner),
    );
  }
  get vtableSlotIndex(): number {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 57);
  }
  get overriddenFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 58,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get befriendingClasses(): Iterable<ClassSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 59,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get templateFriendships(): Iterable<TemplateFriendship> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      FunctionSymbolSlotBase + 60,
      (item: any) => decodeTemplateFriendship(item, this.modelOwner),
    );
  }
  get delegatingConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 61),
      this.modelOwner,
    );
  }
  get completeObjectVariant(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 62),
      this.modelOwner,
    );
  }
  get deletingDtorVariant(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 63),
      this.modelOwner,
    );
  }
  get structorPrincipal(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 64),
      this.modelOwner,
    );
  }
  get isStructorVariant(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 65) !== 0;
  }
  get isStructor(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 66) !== 0;
  }
  get hasBaseObjectVariant(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 67) !== 0;
  }
  get inheritedConstructor(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 68),
      this.modelOwner,
    );
  }
  get inheritedConstructorOrigin(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 69),
      this.modelOwner,
    );
  }
  get isDeletingDtorVariant(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 70) !== 0;
  }
  get hostScope(): ScopeSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 71),
      this.modelOwner,
    );
  }
  get hasFriendDefaultArgument(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 72) !== 0;
  }
  get hasFriendDefaultTemplateArgument(): boolean {
    return cxx.readSymbol(this.handle, FunctionSymbolSlotBase + 73) !== 0;
  }
}
/** @category Symbols */
export class OverloadSetSymbol extends Symbol {
  get functions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      OverloadSetSymbolSlotBase + 0,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declaredFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      OverloadSetSymbolSlotBase + 1,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get usingDeclarations(): Iterable<UsingDeclarationSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      OverloadSetSymbolSlotBase + 2,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get hasUnresolvedUsingDeclaration(): boolean {
    return cxx.readSymbol(this.handle, OverloadSetSymbolSlotBase + 3) !== 0;
  }
}
/** @category Symbols */
export class LambdaSymbol extends ScopeSymbol {
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 0) !== 0;
  }
  get isConsteval(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 1) !== 0;
  }
  get isMutable(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 2) !== 0;
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 3) !== 0;
  }
  get isTemplate(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 4) !== 0;
  }
  get isInTemplate(): boolean {
    return cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 5) !== 0;
  }
  get closureType(): ClassSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, LambdaSymbolSlotBase + 6),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class FunctionParametersSymbol extends ScopeSymbol {
  get cvQualifiers(): CvQualifiers {
    return cvQualifiersNames[
      cxx.readSymbol(this.handle, FunctionParametersSymbolSlotBase + 0)
    ]!;
  }
}
/** @category Symbols */
export class TemplateParametersSymbol extends ScopeSymbol {
  get isExplicitTemplateSpecialization(): boolean {
    return (
      cxx.readSymbol(this.handle, TemplateParametersSymbolSlotBase + 0) !== 0
    );
  }
}
/** @category Symbols */
export class BlockSymbol extends ScopeSymbol {
  get isOutermostBlockScope(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 0) !== 0;
  }
  get isFunctionTryHandler(): boolean {
    return cxx.readSymbol(this.handle, BlockSymbolSlotBase + 1) !== 0;
  }
}
/** @category Symbols */
export class TypeAliasSymbol extends Symbol {
  get canonical(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get definition(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get declaration(): AliasDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get primaryTemplateSymbol(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 6);
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 7) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 8) !== 0;
  }
  get templateArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 9,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<TemplateArgument>
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 10,
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeTemplateArgument(element, this.modelOwner),
        ),
    );
  }
  get canonicalOrNull(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 11),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): TypeAliasSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 12),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<TypeAliasSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 13,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declarations(): Iterable<TypeAliasSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      TypeAliasSymbolSlotBase + 14,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get expansionTypeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TypeAliasSymbolSlotBase + 15),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class VariableSymbol extends Symbol {
  get canonical(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get definition(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get declaration(): SimpleDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get templateDeclaration(): TemplateDeclarationAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 3),
      this.modelOwner,
    );
  }
  get templateParameters(): TemplateParametersSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get primaryTemplateSymbol(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
  get templateSpecializationIndex(): number {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 6);
  }
  get isSpecialization(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 7) !== 0;
  }
  get isTemplatePattern(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 8) !== 0;
  }
  get templateArguments(): Iterable<TemplateArgument> {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 9,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
  get externInstantiationDeclarations(): Iterable<
    ReadonlyArray<TemplateArgument>
  > {
    return symbolValItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 10,
      (item: any) =>
        (item as any[]).map((element: any) =>
          decodeTemplateArgument(element, this.modelOwner),
        ),
    );
  }
  get canonicalOrNull(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 11),
      this.modelOwner,
    );
  }
  get resolvedDefinition(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 12),
      this.modelOwner,
    );
  }
  get redeclarations(): Iterable<VariableSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 13,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get declarations(): Iterable<VariableSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      VariableSymbolSlotBase + 14,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 15) !== 0;
  }
  get isThreadLocal(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 16) !== 0;
  }
  get isExtern(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 17) !== 0;
  }
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 18) !== 0;
  }
  get isConstinit(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 19) !== 0;
  }
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 20) !== 0;
  }
  get isFunctionLocalPredefined(): boolean {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 21) !== 0;
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 22),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, VariableSymbolSlotBase + 23),
      this.modelOwner,
    );
  }
  get constValue(): ConstValue | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, VariableSymbolSlotBase + 24),
      (item: any) => decodeConstValue(item, this.modelOwner),
    );
  }
  get explicitAlignment(): number {
    return cxx.readSymbol(this.handle, VariableSymbolSlotBase + 25);
  }
}
/** @category Symbols */
export class FieldSymbol extends Symbol {
  get definition(): VariableSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
  get isBitField(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 1) !== 0;
  }
  get bitFieldOffset(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 2);
  }
  get bitFieldWidth(): ConstValue | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 3),
      (item: any) => decodeConstValue(item, this.modelOwner),
    );
  }
  get isExtern(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 4) !== 0;
  }
  get isStatic(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 5) !== 0;
  }
  get isThreadLocal(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 6) !== 0;
  }
  get isConstexpr(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 7) !== 0;
  }
  get isConstinit(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 8) !== 0;
  }
  get isInline(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 9) !== 0;
  }
  get isMutable(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 10) !== 0;
  }
  get isNoUniqueAddress(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 11) !== 0;
  }
  get offsetInClass(): bigint | undefined {
    return cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 12) as
      bigint | undefined;
  }
  get localOffset(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 13);
  }
  get alignment(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 14);
  }
  get explicitAlignment(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 15);
  }
  get isPacked(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 16) !== 0;
  }
  get effectiveAlignment(): number {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 17);
  }
  get initializer(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 18),
      this.modelOwner,
    );
  }
  get constructorSymbol(): FunctionSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, FieldSymbolSlotBase + 19),
      this.modelOwner,
    );
  }
  get constValue(): ConstValue | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 20),
      (item: any) => decodeConstValue(item, this.modelOwner),
    );
  }
  get isDefinitionRequired(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 21) !== 0;
  }
  get hasPendingInitializer(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 22) !== 0;
  }
  get hasInitializer(): boolean {
    return cxx.readSymbol(this.handle, FieldSymbolSlotBase + 23) !== 0;
  }
  get pendingInitializer(): PendingInstantiation | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, FieldSymbolSlotBase + 24),
      (item: any) => decodePendingInstantiation(item, this.modelOwner),
    );
  }
}
/** @category Symbols */
export class ParameterSymbol extends Symbol {
  get isParameterPack(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 0) !== 0;
  }
  get defaultArgument(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get hasDefaultArgument(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 2) !== 0;
  }
  get pendingDefaultArgument(): PendingInstantiation | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, ParameterSymbolSlotBase + 3),
      (item: any) => decodePendingInstantiation(item, this.modelOwner),
    );
  }
  get defaultArgumentSource(): ParameterSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get isExplicitObject(): boolean {
    return cxx.readSymbol(this.handle, ParameterSymbolSlotBase + 5) !== 0;
  }
}
/** @category Symbols */
export class ParameterPackSymbol extends Symbol {
  get elements(): Iterable<Symbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      ParameterPackSymbolSlotBase + 0,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
}
/** @category Symbols */
export class TypeParameterSymbol extends Symbol {
  get defaultArgument(): TemplateParameterAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TypeParameterSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class NonTypeParameterSymbol extends Symbol {
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 0) !== 0
    );
  }
  get defaultArgument(): TemplateParameterAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get index(): number {
    return cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 2);
  }
  get depth(): number {
    return cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 3);
  }
  get objectType(): Type | undefined {
    return typeOf(
      cxx.readSymbol(this.handle, NonTypeParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class TemplateTypeParameterSymbol extends Symbol {
  get defaultArgument(): TemplateParameterAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, TemplateTypeParameterSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class ConstraintTypeParameterSymbol extends Symbol {
  get isParameterPack(): boolean {
    return (
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 0) !==
      0
    );
  }
  get defaultArgument(): TemplateParameterAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get index(): number {
    return cxx.readSymbol(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 2,
    );
  }
  get depth(): number {
    return cxx.readSymbol(
      this.handle,
      ConstraintTypeParameterSymbolSlotBase + 3,
    );
  }
  get typeConstraint(): TypeConstraintAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 4),
      this.modelOwner,
    );
  }
  get constraintExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, ConstraintTypeParameterSymbolSlotBase + 5),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class EnumeratorSymbol extends Symbol {
  get value(): ConstValue | undefined {
    return optionalOf(
      cxx.readSymbolVal(this.handle, EnumeratorSymbolSlotBase + 0),
      (item: any) => decodeConstValue(item, this.modelOwner),
    );
  }
}
/** @category Symbols */
export class NamespaceAliasSymbol extends Symbol {
  get namespaceSymbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, NamespaceAliasSymbolSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Symbols */
export class UsingDeclarationSymbol extends Symbol {
  get isUnresolved(): boolean {
    return (
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 0) !== 0
    );
  }
  get declarator(): UsingDeclaratorAST | undefined {
    return astOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 1),
      this.modelOwner,
    );
  }
  get target(): Symbol | undefined {
    return symbolOf(
      cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 2),
      this.modelOwner,
    );
  }
  get targetFunctionCount(): number {
    return cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 3);
  }
  get targetUsingDeclarationCount(): number {
    return cxx.readSymbol(this.handle, UsingDeclarationSymbolSlotBase + 4);
  }
  get introducedFunctions(): Iterable<FunctionSymbol | undefined> {
    return symbolItems(
      this.modelOwner,
      this.handle,
      UsingDeclarationSymbolSlotBase + 5,
      (item: any) => symbolOf(item, this.modelOwner),
    );
  }
}
/** @category Types */
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
/** @category Types */
export class BuiltinVaListType extends Type {}
/** @category Types */
export class BuiltinMetaInfoType extends Type {}
/** @category Types */
export class VoidType extends Type {}
/** @category Types */
export class NullptrType extends Type {}
/** @category Types */
export class DecltypeAutoType extends Type {}
/** @category Types */
export class AutoType extends Type {}
/** @category Types */
export class BoolType extends Type {}
/** @category Types */
export class SignedCharType extends Type {}
/** @category Types */
export class ShortIntType extends Type {}
/** @category Types */
export class IntType extends Type {}
/** @category Types */
export class LongIntType extends Type {}
/** @category Types */
export class LongLongIntType extends Type {}
/** @category Types */
export class Int128Type extends Type {}
/** @category Types */
export class UnsignedCharType extends Type {}
/** @category Types */
export class UnsignedShortIntType extends Type {}
/** @category Types */
export class UnsignedIntType extends Type {}
/** @category Types */
export class UnsignedLongIntType extends Type {}
/** @category Types */
export class UnsignedLongLongIntType extends Type {}
/** @category Types */
export class UnsignedInt128Type extends Type {}
/** @category Types */
export class CharType extends Type {}
/** @category Types */
export class Char8Type extends Type {}
/** @category Types */
export class Char16Type extends Type {}
/** @category Types */
export class Char32Type extends Type {}
/** @category Types */
export class WideCharType extends Type {}
/** @category Types */
export class FloatType extends Type {}
/** @category Types */
export class DoubleType extends Type {}
/** @category Types */
export class LongDoubleType extends Type {}
/** @category Types */
export class Float16Type extends Type {}
/** @category Types */
export class Float128Type extends Type {}
/** @category Types */
export class QualType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, QualTypeSlotBase + 0),
      this.modelOwner,
    );
  }
  get cvQualifiers(): CvQualifiers {
    return cvQualifiersNames[cxx.readType(this.handle, QualTypeSlotBase + 1)]!;
  }
  get isConst(): boolean {
    return cxx.readType(this.handle, QualTypeSlotBase + 2) !== 0;
  }
  get isVolatile(): boolean {
    return cxx.readType(this.handle, QualTypeSlotBase + 3) !== 0;
  }
}
/** @category Types */
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
}
/** @category Types */
export class UnboundedArrayType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, UnboundedArrayTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class PointerType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, PointerTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class LvalueReferenceType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, LvalueReferenceTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class RvalueReferenceType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, RvalueReferenceTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class OverloadSetType extends Type {
  get symbol(): OverloadSetSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, OverloadSetTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
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
  get isEllipsisOnly(): boolean {
    return cxx.readType(this.handle, FunctionTypeSlotBase + 3) !== 0;
  }
  get cvQualifiers(): CvQualifiers {
    return cvQualifiersNames[
      cxx.readType(this.handle, FunctionTypeSlotBase + 4)
    ]!;
  }
  get refQualifier(): RefQualifier {
    return refQualifierNames[
      cxx.readType(this.handle, FunctionTypeSlotBase + 5)
    ]!;
  }
  get exceptionSpecification(): ExceptionSpecification {
    return decodeExceptionSpecification(
      cxx.readTypeVal(this.handle, FunctionTypeSlotBase + 6),
      this.modelOwner,
    );
  }
  get isNoexcept(): boolean {
    return cxx.readType(this.handle, FunctionTypeSlotBase + 7) !== 0;
  }
  get noexceptExpression(): ExpressionAST | undefined {
    return astOf(
      cxx.readType(this.handle, FunctionTypeSlotBase + 8),
      this.modelOwner,
    );
  }
}
/** @category Types */
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
}
/** @category Types */
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
}
/** @category Types */
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
}
/** @category Types */
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
}
/** @category Types */
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
}
/** @category Types */
export class NamespaceType extends Type {
  get symbol(): NamespaceSymbol | undefined {
    return symbolOf(
      cxx.readType(this.handle, NamespaceTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
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
}
/** @category Types */
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
}
/** @category Types */
export class TemplateTypeParameterSpecializationType extends Type {
  get templateParameter(): TemplateTypeParameterType | undefined {
    return typeOf(
      cxx.readType(
        this.handle,
        TemplateTypeParameterSpecializationTypeSlotBase + 0,
      ),
      this.modelOwner,
    );
  }
  get templateArguments(): Iterable<TemplateArgument> {
    return typeValItems(
      this.modelOwner,
      this.handle,
      TemplateTypeParameterSpecializationTypeSlotBase + 1,
      (item: any) => decodeTemplateArgument(item, this.modelOwner),
    );
  }
}
/** @category Types */
export class PackExpansionType extends Type {
  get pattern(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, PackExpansionTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class DecltypeType extends Type {
  get expression(): ExpressionAST | undefined {
    return astOf(
      cxx.readType(this.handle, DecltypeTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
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
  get sourceLocationRange(): SourceLocationRange {
    return decodeSourceLocationRange(
      cxx.readTypeVal(this.handle, UnresolvedNameTypeSlotBase + 2),
      this.modelOwner,
    );
  }
}
/** @category Types */
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
}
/** @category Types */
export class UnresolvedUnderlyingType extends Type {
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedUnderlyingTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class UnresolvedBuiltinType extends Type {
  get builtinKind(): UnaryBuiltinTypeKind {
    return unaryBuiltinTypeKindNames[
      cxx.readType(this.handle, UnresolvedBuiltinTypeSlotBase + 0)
    ]!;
  }
  get typeId(): TypeIdAST | undefined {
    return astOf(
      cxx.readType(this.handle, UnresolvedBuiltinTypeSlotBase + 1),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class BitIntType extends Type {
  get numBits(): number {
    return cxx.readType(this.handle, BitIntTypeSlotBase + 0);
  }
}
/** @category Types */
export class UnsignedBitIntType extends Type {
  get numBits(): number {
    return cxx.readType(this.handle, UnsignedBitIntTypeSlotBase + 0);
  }
}
/** @category Types */
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
}
/** @category Types */
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
    return vectorKindNames[cxx.readType(this.handle, VectorTypeSlotBase + 2)]!;
  }
}
/** @category Types */
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
    return vectorKindNames[
      cxx.readType(this.handle, UnresolvedVectorTypeSlotBase + 2)
    ]!;
  }
  get sizeKind(): VectorSizeKind {
    return vectorSizeKindNames[
      cxx.readType(this.handle, UnresolvedVectorTypeSlotBase + 3)
    ]!;
  }
}
/** @category Types */
export class ComplexType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, ComplexTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class AtomicType extends Type {
  get elementType(): Type | undefined {
    return typeOf(
      cxx.readType(this.handle, AtomicTypeSlotBase + 0),
      this.modelOwner,
    );
  }
}
/** @category Types */
export class SveType extends Type {
  get sveKind(): SveTypeKind {
    return sveTypeKindNames[cxx.readType(this.handle, SveTypeSlotBase + 0)]!;
  }
}
/** @category Enumerations */
export type ASTKind =
  | "TranslationUnit"
  | "ModuleUnit"
  | "SimpleDeclaration"
  | "AsmDeclaration"
  | "NamespaceAliasDefinition"
  | "UsingDeclaration"
  | "UsingEnumDeclaration"
  | "UsingDirective"
  | "StaticAssertDeclaration"
  | "AliasDeclaration"
  | "OpaqueEnumDeclaration"
  | "FunctionDefinition"
  | "TemplateDeclaration"
  | "ConceptDefinition"
  | "DeductionGuide"
  | "ExplicitInstantiation"
  | "ExportDeclaration"
  | "ExportCompoundDeclaration"
  | "LinkageSpecification"
  | "NamespaceDefinition"
  | "EmptyDeclaration"
  | "AttributeDeclaration"
  | "ModuleImportDeclaration"
  | "ParameterDeclaration"
  | "AccessDeclaration"
  | "ForRangeDeclaration"
  | "StructuredBindingDeclaration"
  | "AsmOperand"
  | "AsmQualifier"
  | "AsmClobber"
  | "AsmGotoLabel"
  | "Splicer"
  | "GlobalModuleFragment"
  | "PrivateModuleFragment"
  | "ModuleDeclaration"
  | "ModuleName"
  | "ModuleQualifier"
  | "ModulePartition"
  | "ImportName"
  | "InitDeclarator"
  | "Declarator"
  | "UsingDeclarator"
  | "Enumerator"
  | "TypeId"
  | "Handler"
  | "BaseSpecifier"
  | "RequiresClause"
  | "ParameterDeclarationClause"
  | "TrailingReturnType"
  | "LambdaSpecifier"
  | "TypeConstraint"
  | "AttributeArgumentClause"
  | "Attribute"
  | "AttributeUsingPrefix"
  | "NewPlacement"
  | "NestedNamespaceSpecifier"
  | "LabeledStatement"
  | "CaseStatement"
  | "DefaultStatement"
  | "ExpressionStatement"
  | "CompoundStatement"
  | "IfStatement"
  | "ConstevalIfStatement"
  | "SwitchStatement"
  | "WhileStatement"
  | "DoStatement"
  | "ForRangeStatement"
  | "ForStatement"
  | "BreakStatement"
  | "ContinueStatement"
  | "ReturnStatement"
  | "CoroutineReturnStatement"
  | "GotoStatement"
  | "DeclarationStatement"
  | "TryBlockStatement"
  | "CharLiteralExpression"
  | "BoolLiteralExpression"
  | "IntLiteralExpression"
  | "FloatLiteralExpression"
  | "NullptrLiteralExpression"
  | "StringLiteralExpression"
  | "UserDefinedStringLiteralExpression"
  | "ObjectLiteralExpression"
  | "ThisExpression"
  | "PackIndexExpression"
  | "GenericSelectionExpression"
  | "NestedStatementExpression"
  | "DefaultInitializerExpression"
  | "NestedExpression"
  | "IdExpression"
  | "LambdaExpression"
  | "FoldExpression"
  | "RightFoldExpression"
  | "LeftFoldExpression"
  | "RequiresExpression"
  | "VaArgExpression"
  | "SubscriptExpression"
  | "CallExpression"
  | "TypeConstruction"
  | "BracedTypeConstruction"
  | "SpliceMemberExpression"
  | "MemberExpression"
  | "PostIncrExpression"
  | "CppCastExpression"
  | "BuiltinBitCastExpression"
  | "BuiltinConvertVectorExpression"
  | "BuiltinOffsetofExpression"
  | "TypeidExpression"
  | "TypeidOfTypeExpression"
  | "SpliceExpression"
  | "GlobalScopeReflectExpression"
  | "NamespaceReflectExpression"
  | "TypeIdReflectExpression"
  | "ReflectExpression"
  | "LabelAddressExpression"
  | "UnaryExpression"
  | "AwaitExpression"
  | "SizeofExpression"
  | "SizeofTypeExpression"
  | "SizeofPackExpression"
  | "AlignofTypeExpression"
  | "AlignofExpression"
  | "NoexceptExpression"
  | "NewExpression"
  | "DeleteExpression"
  | "CastExpression"
  | "ImplicitCastExpression"
  | "ConstExpression"
  | "BinaryExpression"
  | "ConditionalExpression"
  | "YieldExpression"
  | "ThrowExpression"
  | "AssignmentExpression"
  | "TargetExpression"
  | "RightExpression"
  | "CompoundAssignmentExpression"
  | "PackExpansionExpression"
  | "DesignatedInitializerClause"
  | "TypeTraitExpression"
  | "ConditionExpression"
  | "EqualInitializer"
  | "BracedInitList"
  | "ParenInitializer"
  | "ThreeWayComparisonExpression"
  | "DefaultGenericAssociation"
  | "TypeGenericAssociation"
  | "DotDesignator"
  | "SubscriptDesignator"
  | "TemplateTypeParameter"
  | "NonTypeTemplateParameter"
  | "TypenameTypeParameter"
  | "ConstraintTypeParameter"
  | "TypedefSpecifier"
  | "FriendSpecifier"
  | "ConstevalSpecifier"
  | "ConstinitSpecifier"
  | "ConstexprSpecifier"
  | "InlineSpecifier"
  | "NoreturnSpecifier"
  | "StaticSpecifier"
  | "ExternSpecifier"
  | "RegisterSpecifier"
  | "ThreadLocalSpecifier"
  | "ThreadSpecifier"
  | "MutableSpecifier"
  | "VirtualSpecifier"
  | "ExplicitSpecifier"
  | "AutoTypeSpecifier"
  | "VoidTypeSpecifier"
  | "SizeTypeSpecifier"
  | "SignTypeSpecifier"
  | "BuiltinTypeSpecifier"
  | "UnaryBuiltinTypeSpecifier"
  | "BinaryBuiltinTypeSpecifier"
  | "IntegralTypeSpecifier"
  | "FloatingPointTypeSpecifier"
  | "ComplexTypeSpecifier"
  | "NamedTypeSpecifier"
  | "AtomicTypeSpecifier"
  | "BitIntTypeSpecifier"
  | "UnderlyingTypeSpecifier"
  | "ElaboratedTypeSpecifier"
  | "DecltypeAutoSpecifier"
  | "DecltypeSpecifier"
  | "PlaceholderTypeSpecifier"
  | "ConstQualifier"
  | "VolatileQualifier"
  | "AtomicQualifier"
  | "RestrictQualifier"
  | "EnumSpecifier"
  | "ClassSpecifier"
  | "TypenameSpecifier"
  | "SplicerTypeSpecifier"
  | "PointerOperator"
  | "ReferenceOperator"
  | "PtrToMemberOperator"
  | "BitfieldDeclarator"
  | "ParameterPack"
  | "IdDeclarator"
  | "NestedDeclarator"
  | "FunctionDeclaratorChunk"
  | "ArrayDeclaratorChunk"
  | "NameId"
  | "DestructorId"
  | "DecltypeId"
  | "OperatorFunctionId"
  | "LiteralOperatorId"
  | "ConversionFunctionId"
  | "SimpleTemplateId"
  | "LiteralOperatorTemplateId"
  | "OperatorFunctionTemplateId"
  | "GlobalNestedNameSpecifier"
  | "SimpleNestedNameSpecifier"
  | "DecltypeNestedNameSpecifier"
  | "TemplateNestedNameSpecifier"
  | "DefaultFunctionBody"
  | "CompoundStatementFunctionBody"
  | "TryStatementFunctionBody"
  | "DeleteFunctionBody"
  | "TypeTemplateArgument"
  | "ExpressionTemplateArgument"
  | "ThrowExceptionSpecifier"
  | "NoexceptSpecifier"
  | "SimpleRequirement"
  | "CompoundRequirement"
  | "TypeRequirement"
  | "NestedRequirement"
  | "NewParenInitializer"
  | "NewBracedInitializer"
  | "ParenMemInitializer"
  | "BracedMemInitializer"
  | "ThisLambdaCapture"
  | "DerefThisLambdaCapture"
  | "SimpleLambdaCapture"
  | "RefLambdaCapture"
  | "RefInitLambdaCapture"
  | "InitLambdaCapture"
  | "EllipsisExceptionDeclaration"
  | "TypeExceptionDeclaration"
  | "CxxAttribute"
  | "GccAttribute"
  | "AlignasAttribute"
  | "AlignasTypeAttribute"
  | "AsmAttribute"
  | "ScopedAttributeToken"
  | "SimpleAttributeToken";
const astKindNames: Record<number, ASTKind> = {
  0: "TranslationUnit",
  1: "ModuleUnit",
  2: "SimpleDeclaration",
  3: "AsmDeclaration",
  4: "NamespaceAliasDefinition",
  5: "UsingDeclaration",
  6: "UsingEnumDeclaration",
  7: "UsingDirective",
  8: "StaticAssertDeclaration",
  9: "AliasDeclaration",
  10: "OpaqueEnumDeclaration",
  11: "FunctionDefinition",
  12: "TemplateDeclaration",
  13: "ConceptDefinition",
  14: "DeductionGuide",
  15: "ExplicitInstantiation",
  16: "ExportDeclaration",
  17: "ExportCompoundDeclaration",
  18: "LinkageSpecification",
  19: "NamespaceDefinition",
  20: "EmptyDeclaration",
  21: "AttributeDeclaration",
  22: "ModuleImportDeclaration",
  23: "ParameterDeclaration",
  24: "AccessDeclaration",
  25: "ForRangeDeclaration",
  26: "StructuredBindingDeclaration",
  27: "AsmOperand",
  28: "AsmQualifier",
  29: "AsmClobber",
  30: "AsmGotoLabel",
  31: "Splicer",
  32: "GlobalModuleFragment",
  33: "PrivateModuleFragment",
  34: "ModuleDeclaration",
  35: "ModuleName",
  36: "ModuleQualifier",
  37: "ModulePartition",
  38: "ImportName",
  39: "InitDeclarator",
  40: "Declarator",
  41: "UsingDeclarator",
  42: "Enumerator",
  43: "TypeId",
  44: "Handler",
  45: "BaseSpecifier",
  46: "RequiresClause",
  47: "ParameterDeclarationClause",
  48: "TrailingReturnType",
  49: "LambdaSpecifier",
  50: "TypeConstraint",
  51: "AttributeArgumentClause",
  52: "Attribute",
  53: "AttributeUsingPrefix",
  54: "NewPlacement",
  55: "NestedNamespaceSpecifier",
  56: "LabeledStatement",
  57: "CaseStatement",
  58: "DefaultStatement",
  59: "ExpressionStatement",
  60: "CompoundStatement",
  61: "IfStatement",
  62: "ConstevalIfStatement",
  63: "SwitchStatement",
  64: "WhileStatement",
  65: "DoStatement",
  66: "ForRangeStatement",
  67: "ForStatement",
  68: "BreakStatement",
  69: "ContinueStatement",
  70: "ReturnStatement",
  71: "CoroutineReturnStatement",
  72: "GotoStatement",
  73: "DeclarationStatement",
  74: "TryBlockStatement",
  75: "CharLiteralExpression",
  76: "BoolLiteralExpression",
  77: "IntLiteralExpression",
  78: "FloatLiteralExpression",
  79: "NullptrLiteralExpression",
  80: "StringLiteralExpression",
  81: "UserDefinedStringLiteralExpression",
  82: "ObjectLiteralExpression",
  83: "ThisExpression",
  84: "PackIndexExpression",
  85: "GenericSelectionExpression",
  86: "NestedStatementExpression",
  87: "DefaultInitializerExpression",
  88: "NestedExpression",
  89: "IdExpression",
  90: "LambdaExpression",
  91: "FoldExpression",
  92: "RightFoldExpression",
  93: "LeftFoldExpression",
  94: "RequiresExpression",
  95: "VaArgExpression",
  96: "SubscriptExpression",
  97: "CallExpression",
  98: "TypeConstruction",
  99: "BracedTypeConstruction",
  100: "SpliceMemberExpression",
  101: "MemberExpression",
  102: "PostIncrExpression",
  103: "CppCastExpression",
  104: "BuiltinBitCastExpression",
  105: "BuiltinConvertVectorExpression",
  106: "BuiltinOffsetofExpression",
  107: "TypeidExpression",
  108: "TypeidOfTypeExpression",
  109: "SpliceExpression",
  110: "GlobalScopeReflectExpression",
  111: "NamespaceReflectExpression",
  112: "TypeIdReflectExpression",
  113: "ReflectExpression",
  114: "LabelAddressExpression",
  115: "UnaryExpression",
  116: "AwaitExpression",
  117: "SizeofExpression",
  118: "SizeofTypeExpression",
  119: "SizeofPackExpression",
  120: "AlignofTypeExpression",
  121: "AlignofExpression",
  122: "NoexceptExpression",
  123: "NewExpression",
  124: "DeleteExpression",
  125: "CastExpression",
  126: "ImplicitCastExpression",
  127: "ConstExpression",
  128: "BinaryExpression",
  129: "ConditionalExpression",
  130: "YieldExpression",
  131: "ThrowExpression",
  132: "AssignmentExpression",
  133: "TargetExpression",
  134: "RightExpression",
  135: "CompoundAssignmentExpression",
  136: "PackExpansionExpression",
  137: "DesignatedInitializerClause",
  138: "TypeTraitExpression",
  139: "ConditionExpression",
  140: "EqualInitializer",
  141: "BracedInitList",
  142: "ParenInitializer",
  143: "ThreeWayComparisonExpression",
  144: "DefaultGenericAssociation",
  145: "TypeGenericAssociation",
  146: "DotDesignator",
  147: "SubscriptDesignator",
  148: "TemplateTypeParameter",
  149: "NonTypeTemplateParameter",
  150: "TypenameTypeParameter",
  151: "ConstraintTypeParameter",
  152: "TypedefSpecifier",
  153: "FriendSpecifier",
  154: "ConstevalSpecifier",
  155: "ConstinitSpecifier",
  156: "ConstexprSpecifier",
  157: "InlineSpecifier",
  158: "NoreturnSpecifier",
  159: "StaticSpecifier",
  160: "ExternSpecifier",
  161: "RegisterSpecifier",
  162: "ThreadLocalSpecifier",
  163: "ThreadSpecifier",
  164: "MutableSpecifier",
  165: "VirtualSpecifier",
  166: "ExplicitSpecifier",
  167: "AutoTypeSpecifier",
  168: "VoidTypeSpecifier",
  169: "SizeTypeSpecifier",
  170: "SignTypeSpecifier",
  171: "BuiltinTypeSpecifier",
  172: "UnaryBuiltinTypeSpecifier",
  173: "BinaryBuiltinTypeSpecifier",
  174: "IntegralTypeSpecifier",
  175: "FloatingPointTypeSpecifier",
  176: "ComplexTypeSpecifier",
  177: "NamedTypeSpecifier",
  178: "AtomicTypeSpecifier",
  179: "BitIntTypeSpecifier",
  180: "UnderlyingTypeSpecifier",
  181: "ElaboratedTypeSpecifier",
  182: "DecltypeAutoSpecifier",
  183: "DecltypeSpecifier",
  184: "PlaceholderTypeSpecifier",
  185: "ConstQualifier",
  186: "VolatileQualifier",
  187: "AtomicQualifier",
  188: "RestrictQualifier",
  189: "EnumSpecifier",
  190: "ClassSpecifier",
  191: "TypenameSpecifier",
  192: "SplicerTypeSpecifier",
  193: "PointerOperator",
  194: "ReferenceOperator",
  195: "PtrToMemberOperator",
  196: "BitfieldDeclarator",
  197: "ParameterPack",
  198: "IdDeclarator",
  199: "NestedDeclarator",
  200: "FunctionDeclaratorChunk",
  201: "ArrayDeclaratorChunk",
  202: "NameId",
  203: "DestructorId",
  204: "DecltypeId",
  205: "OperatorFunctionId",
  206: "LiteralOperatorId",
  207: "ConversionFunctionId",
  208: "SimpleTemplateId",
  209: "LiteralOperatorTemplateId",
  210: "OperatorFunctionTemplateId",
  211: "GlobalNestedNameSpecifier",
  212: "SimpleNestedNameSpecifier",
  213: "DecltypeNestedNameSpecifier",
  214: "TemplateNestedNameSpecifier",
  215: "DefaultFunctionBody",
  216: "CompoundStatementFunctionBody",
  217: "TryStatementFunctionBody",
  218: "DeleteFunctionBody",
  219: "TypeTemplateArgument",
  220: "ExpressionTemplateArgument",
  221: "ThrowExceptionSpecifier",
  222: "NoexceptSpecifier",
  223: "SimpleRequirement",
  224: "CompoundRequirement",
  225: "TypeRequirement",
  226: "NestedRequirement",
  227: "NewParenInitializer",
  228: "NewBracedInitializer",
  229: "ParenMemInitializer",
  230: "BracedMemInitializer",
  231: "ThisLambdaCapture",
  232: "DerefThisLambdaCapture",
  233: "SimpleLambdaCapture",
  234: "RefLambdaCapture",
  235: "RefInitLambdaCapture",
  236: "InitLambdaCapture",
  237: "EllipsisExceptionDeclaration",
  238: "TypeExceptionDeclaration",
  239: "CxxAttribute",
  240: "GccAttribute",
  241: "AlignasAttribute",
  242: "AlignasTypeAttribute",
  243: "AsmAttribute",
  244: "ScopedAttributeToken",
  245: "SimpleAttributeToken",
};
/** @category Enumerations */
export type SymbolKind =
  | "Namespace"
  | "NamespaceAlias"
  | "Concept"
  | "DeductionGuide"
  | "Class"
  | "Enum"
  | "ScopedEnum"
  | "Function"
  | "TypeAlias"
  | "Variable"
  | "Field"
  | "Parameter"
  | "ParameterPack"
  | "Enumerator"
  | "FunctionParameters"
  | "TemplateParameters"
  | "Block"
  | "Lambda"
  | "TypeParameter"
  | "NonTypeParameter"
  | "TemplateTypeParameter"
  | "ConstraintTypeParameter"
  | "OverloadSet"
  | "BaseClass"
  | "InjectedClassName"
  | "Unresolved"
  | "UsingDeclaration";
const symbolKindNames: Record<number, SymbolKind> = {
  0: "Namespace",
  1: "NamespaceAlias",
  2: "Concept",
  3: "DeductionGuide",
  4: "Class",
  5: "Enum",
  6: "ScopedEnum",
  7: "Function",
  8: "TypeAlias",
  9: "Variable",
  10: "Field",
  11: "Parameter",
  12: "ParameterPack",
  13: "Enumerator",
  14: "FunctionParameters",
  15: "TemplateParameters",
  16: "Block",
  17: "Lambda",
  18: "TypeParameter",
  19: "NonTypeParameter",
  20: "TemplateTypeParameter",
  21: "ConstraintTypeParameter",
  22: "OverloadSet",
  23: "BaseClass",
  24: "InjectedClassName",
  25: "Unresolved",
  26: "UsingDeclaration",
};
/** @category Enumerations */
export type TypeKind =
  | "Void"
  | "Nullptr"
  | "DecltypeAuto"
  | "Auto"
  | "Bool"
  | "SignedChar"
  | "ShortInt"
  | "Int"
  | "LongInt"
  | "LongLongInt"
  | "Int128"
  | "UnsignedChar"
  | "UnsignedShortInt"
  | "UnsignedInt"
  | "UnsignedLongInt"
  | "UnsignedLongLongInt"
  | "UnsignedInt128"
  | "Char"
  | "Char8"
  | "Char16"
  | "Char32"
  | "WideChar"
  | "Float"
  | "Double"
  | "LongDouble"
  | "Float16"
  | "Float128"
  | "Qual"
  | "BoundedArray"
  | "UnboundedArray"
  | "Pointer"
  | "LvalueReference"
  | "RvalueReference"
  | "Function"
  | "Class"
  | "Enum"
  | "ScopedEnum"
  | "MemberObjectPointer"
  | "MemberFunctionPointer"
  | "Namespace"
  | "TypeParameter"
  | "TemplateTypeParameter"
  | "TemplateTypeParameterSpecialization"
  | "PackExpansion"
  | "Decltype"
  | "UnresolvedName"
  | "UnresolvedBoundedArray"
  | "UnresolvedUnderlying"
  | "UnresolvedBuiltin"
  | "OverloadSet"
  | "BuiltinVaList"
  | "BuiltinMetaInfo"
  | "BitInt"
  | "UnsignedBitInt"
  | "UnresolvedBitInt"
  | "Vector"
  | "UnresolvedVector"
  | "Complex"
  | "Atomic"
  | "Sve";
const typeKindNames: Record<number, TypeKind> = {
  0: "Void",
  1: "Nullptr",
  2: "DecltypeAuto",
  3: "Auto",
  4: "Bool",
  5: "SignedChar",
  6: "ShortInt",
  7: "Int",
  8: "LongInt",
  9: "LongLongInt",
  10: "Int128",
  11: "UnsignedChar",
  12: "UnsignedShortInt",
  13: "UnsignedInt",
  14: "UnsignedLongInt",
  15: "UnsignedLongLongInt",
  16: "UnsignedInt128",
  17: "Char",
  18: "Char8",
  19: "Char16",
  20: "Char32",
  21: "WideChar",
  22: "Float",
  23: "Double",
  24: "LongDouble",
  25: "Float16",
  26: "Float128",
  27: "Qual",
  28: "BoundedArray",
  29: "UnboundedArray",
  30: "Pointer",
  31: "LvalueReference",
  32: "RvalueReference",
  33: "Function",
  34: "Class",
  35: "Enum",
  36: "ScopedEnum",
  37: "MemberObjectPointer",
  38: "MemberFunctionPointer",
  39: "Namespace",
  40: "TypeParameter",
  41: "TemplateTypeParameter",
  42: "TemplateTypeParameterSpecialization",
  43: "PackExpansion",
  44: "Decltype",
  45: "UnresolvedName",
  46: "UnresolvedBoundedArray",
  47: "UnresolvedUnderlying",
  48: "UnresolvedBuiltin",
  49: "OverloadSet",
  50: "BuiltinVaList",
  51: "BuiltinMetaInfo",
  52: "BitInt",
  53: "UnsignedBitInt",
  54: "UnresolvedBitInt",
  55: "Vector",
  56: "UnresolvedVector",
  57: "Complex",
  58: "Atomic",
  59: "Sve",
};
/** @category Enumerations */
export type NameKind =
  | "Identifier"
  | "OperatorId"
  | "DestructorId"
  | "LiteralOperatorId"
  | "ConversionFunctionId"
  | "TemplateId";
const nameKindNames: Record<number, NameKind> = {
  0: "Identifier",
  1: "OperatorId",
  2: "DestructorId",
  3: "LiteralOperatorId",
  4: "ConversionFunctionId",
  5: "TemplateId",
};
/** @category Enumerations */
export type ValueCategory = "None" | "LValue" | "XValue" | "PrValue";
const valueCategoryNames: Record<number, ValueCategory> = {
  0: "None",
  1: "LValue",
  2: "XValue",
  3: "PrValue",
};
/** @category Enumerations */
export type ImplicitCastKind =
  | "Identity"
  | "LValueToRValueConversion"
  | "ArrayToPointerConversion"
  | "FunctionToPointerConversion"
  | "IntegralPromotion"
  | "FloatingPointPromotion"
  | "IntegralConversion"
  | "FloatingPointConversion"
  | "FloatingIntegralConversion"
  | "PointerConversion"
  | "PointerToMemberConversion"
  | "DerivedToBaseConversion"
  | "BaseToDerivedConversion"
  | "BooleanConversion"
  | "FunctionPointerConversion"
  | "QualificationConversion"
  | "VectorSplat"
  | "VectorConversion"
  | "AtomicToNonAtomic"
  | "NonAtomicToAtomic"
  | "RealToComplexConversion"
  | "ComplexToRealConversion"
  | "ComplexConversion"
  | "TemporaryMaterializationConversion"
  | "UserDefinedConversion";
const implicitCastKindNames: Record<number, ImplicitCastKind> = {
  0: "Identity",
  1: "LValueToRValueConversion",
  2: "ArrayToPointerConversion",
  3: "FunctionToPointerConversion",
  4: "IntegralPromotion",
  5: "FloatingPointPromotion",
  6: "IntegralConversion",
  7: "FloatingPointConversion",
  8: "FloatingIntegralConversion",
  9: "PointerConversion",
  10: "PointerToMemberConversion",
  11: "DerivedToBaseConversion",
  12: "BaseToDerivedConversion",
  13: "BooleanConversion",
  14: "FunctionPointerConversion",
  15: "QualificationConversion",
  16: "VectorSplat",
  17: "VectorConversion",
  18: "AtomicToNonAtomic",
  19: "NonAtomicToAtomic",
  20: "RealToComplexConversion",
  21: "ComplexToRealConversion",
  22: "ComplexConversion",
  23: "TemporaryMaterializationConversion",
  24: "UserDefinedConversion",
};
/** @category Enumerations */
export type BuiltinTypeTraitKind =
  | "none"
  | "__builtin_types_compatible_p"
  | "__has_trivial_constructor"
  | "__has_trivial_destructor"
  | "__has_unique_object_representations"
  | "__has_virtual_destructor"
  | "__is_abstract"
  | "__is_aggregate"
  | "__is_arithmetic"
  | "__is_array"
  | "__is_assignable"
  | "__is_base_of"
  | "__is_bounded_array"
  | "__is_class"
  | "__is_compound"
  | "__is_const"
  | "__is_constructible"
  | "__is_convertible_to"
  | "__is_convertible"
  | "__is_destructible"
  | "__is_empty"
  | "__is_enum"
  | "__is_final"
  | "__is_floating_point"
  | "__is_function"
  | "__is_fundamental"
  | "__is_integral"
  | "__is_layout_compatible"
  | "__is_literal_type"
  | "__is_lvalue_reference"
  | "__is_member_function_pointer"
  | "__is_member_object_pointer"
  | "__is_member_pointer"
  | "__is_nothrow_assignable"
  | "__is_nothrow_constructible"
  | "__is_nothrow_destructible"
  | "__is_null_pointer"
  | "__is_object"
  | "__is_pod"
  | "__is_pointer"
  | "__is_polymorphic"
  | "__is_reference"
  | "__is_rvalue_reference"
  | "__is_same_as"
  | "__is_same"
  | "__is_scalar"
  | "__is_scoped_enum"
  | "__is_signed"
  | "__is_standard_layout"
  | "__is_swappable_with"
  | "__is_trivial"
  | "__is_trivially_assignable"
  | "__is_trivially_constructible"
  | "__is_trivially_copyable"
  | "__is_trivially_destructible"
  | "__is_unbounded_array"
  | "__is_union"
  | "__is_unsigned"
  | "__is_void"
  | "__is_volatile"
  | "__reference_constructs_from_temporary"
  | "__reference_converts_from_temporary";
const builtinTypeTraitKindNames: Record<number, BuiltinTypeTraitKind> = {
  0: "none",
  1: "__builtin_types_compatible_p",
  2: "__has_trivial_constructor",
  3: "__has_trivial_destructor",
  4: "__has_unique_object_representations",
  5: "__has_virtual_destructor",
  6: "__is_abstract",
  7: "__is_aggregate",
  8: "__is_arithmetic",
  9: "__is_array",
  10: "__is_assignable",
  11: "__is_base_of",
  12: "__is_bounded_array",
  13: "__is_class",
  14: "__is_compound",
  15: "__is_const",
  16: "__is_constructible",
  17: "__is_convertible_to",
  18: "__is_convertible",
  19: "__is_destructible",
  20: "__is_empty",
  21: "__is_enum",
  22: "__is_final",
  23: "__is_floating_point",
  24: "__is_function",
  25: "__is_fundamental",
  26: "__is_integral",
  27: "__is_layout_compatible",
  28: "__is_literal_type",
  29: "__is_lvalue_reference",
  30: "__is_member_function_pointer",
  31: "__is_member_object_pointer",
  32: "__is_member_pointer",
  33: "__is_nothrow_assignable",
  34: "__is_nothrow_constructible",
  35: "__is_nothrow_destructible",
  36: "__is_null_pointer",
  37: "__is_object",
  38: "__is_pod",
  39: "__is_pointer",
  40: "__is_polymorphic",
  41: "__is_reference",
  42: "__is_rvalue_reference",
  43: "__is_same_as",
  44: "__is_same",
  45: "__is_scalar",
  46: "__is_scoped_enum",
  47: "__is_signed",
  48: "__is_standard_layout",
  49: "__is_swappable_with",
  50: "__is_trivial",
  51: "__is_trivially_assignable",
  52: "__is_trivially_constructible",
  53: "__is_trivially_copyable",
  54: "__is_trivially_destructible",
  55: "__is_unbounded_array",
  56: "__is_union",
  57: "__is_unsigned",
  58: "__is_void",
  59: "__is_volatile",
  60: "__reference_constructs_from_temporary",
  61: "__reference_converts_from_temporary",
};
/** @category Enumerations */
export type UnaryBuiltinTypeKind =
  | "none"
  | "__add_lvalue_reference"
  | "__add_pointer"
  | "__add_rvalue_reference"
  | "__decay"
  | "__make_signed"
  | "__make_unsigned"
  | "__remove_all_extents"
  | "__remove_const"
  | "__remove_cv"
  | "__remove_cvref"
  | "__remove_extent"
  | "__remove_pointer"
  | "__remove_reference_t"
  | "__remove_restrict"
  | "__remove_volatile";
const unaryBuiltinTypeKindNames: Record<number, UnaryBuiltinTypeKind> = {
  0: "none",
  1: "__add_lvalue_reference",
  2: "__add_pointer",
  3: "__add_rvalue_reference",
  4: "__decay",
  5: "__make_signed",
  6: "__make_unsigned",
  7: "__remove_all_extents",
  8: "__remove_const",
  9: "__remove_cv",
  10: "__remove_cvref",
  11: "__remove_extent",
  12: "__remove_pointer",
  13: "__remove_reference_t",
  14: "__remove_restrict",
  15: "__remove_volatile",
};
/** @category Enumerations */
export type BinaryBuiltinTypeKind = "none";
const binaryBuiltinTypeKindNames: Record<number, BinaryBuiltinTypeKind> = {
  0: "none",
};
/** @category Enumerations */
export type IntegerLiteral_Radix =
  "Decimal" | "Hexadecimal" | "Octal" | "Binary";
const integerLiteral_RadixNames: Record<number, IntegerLiteral_Radix> = {
  0: "Decimal",
  1: "Hexadecimal",
  2: "Octal",
  3: "Binary",
};
/** @category Enumerations */
export type FloatLiteral_Components_FloatingPointSuffix =
  "None" | "F" | "L" | "F16" | "F32" | "F64" | "F128" | "BF16" | "Q";
const floatLiteral_Components_FloatingPointSuffixNames: Record<
  number,
  FloatLiteral_Components_FloatingPointSuffix
> = {
  0: "None",
  1: "F",
  2: "L",
  3: "F16",
  4: "F32",
  5: "F64",
  6: "F128",
  7: "BF16",
  8: "Q",
};
/** @category Enumerations */
export type StringLiteralEncoding =
  "None" | "Wide" | "Utf8" | "Utf16" | "Utf32";
const stringLiteralEncodingNames: Record<number, StringLiteralEncoding> = {
  0: "None",
  1: "Wide",
  2: "Utf8",
  3: "Utf16",
  4: "Utf32",
};
/** @category Enumerations */
export type BuiltinFunctionKind =
  | "none"
  | "__atomic_add_fetch"
  | "__atomic_always_lock_free"
  | "__atomic_and_fetch"
  | "__atomic_clear"
  | "__atomic_compare_exchange"
  | "__atomic_compare_exchange_n"
  | "__atomic_exchange"
  | "__atomic_exchange_n"
  | "__atomic_fetch_add"
  | "__atomic_fetch_and"
  | "__atomic_fetch_nand"
  | "__atomic_fetch_or"
  | "__atomic_fetch_sub"
  | "__atomic_fetch_xor"
  | "__atomic_is_lock_free"
  | "__atomic_load"
  | "__atomic_load_n"
  | "__atomic_nand_fetch"
  | "__atomic_or_fetch"
  | "__atomic_signal_fence"
  | "__atomic_store"
  | "__atomic_store_n"
  | "__atomic_sub_fetch"
  | "__atomic_test_and_set"
  | "__atomic_thread_fence"
  | "__atomic_xor_fetch"
  | "__builtin_COLUMN"
  | "__builtin_FILE"
  | "__builtin_FUNCTION"
  | "__builtin_LINE"
  | "__builtin__Exit"
  | "__builtin___cospi"
  | "__builtin___cospif"
  | "__builtin___exp10"
  | "__builtin___exp10f"
  | "__builtin___finite"
  | "__builtin___finitef"
  | "__builtin___finitel"
  | "__builtin___sinpi"
  | "__builtin___sinpif"
  | "__builtin___tanpi"
  | "__builtin___tanpif"
  | "__builtin_abort"
  | "__builtin_abs"
  | "__builtin_acos"
  | "__builtin_acosf"
  | "__builtin_acosh"
  | "__builtin_acoshf"
  | "__builtin_acoshl"
  | "__builtin_acosl"
  | "__builtin_add_overflow"
  | "__builtin_addressof"
  | "__builtin_aligned_alloc"
  | "__builtin_alloca"
  | "__builtin_alloca_with_align"
  | "__builtin_asin"
  | "__builtin_asinf"
  | "__builtin_asinh"
  | "__builtin_asinhf"
  | "__builtin_asinhl"
  | "__builtin_asinl"
  | "__builtin_assume_aligned"
  | "__builtin_atan"
  | "__builtin_atan2"
  | "__builtin_atan2f"
  | "__builtin_atan2l"
  | "__builtin_atanf"
  | "__builtin_atanh"
  | "__builtin_atanhf"
  | "__builtin_atanhl"
  | "__builtin_atanl"
  | "__builtin_bcmp"
  | "__builtin_bcopy"
  | "__builtin_bswap16"
  | "__builtin_bswap32"
  | "__builtin_bswap64"
  | "__builtin_bzero"
  | "__builtin_c23_va_start"
  | "__builtin_cabs"
  | "__builtin_cabsf"
  | "__builtin_cabsl"
  | "__builtin_cacos"
  | "__builtin_cacosf"
  | "__builtin_cacosh"
  | "__builtin_cacoshf"
  | "__builtin_cacoshl"
  | "__builtin_cacosl"
  | "__builtin_carg"
  | "__builtin_cargf"
  | "__builtin_cargl"
  | "__builtin_casin"
  | "__builtin_casinf"
  | "__builtin_casinh"
  | "__builtin_casinhf"
  | "__builtin_casinhl"
  | "__builtin_casinl"
  | "__builtin_catan"
  | "__builtin_catanf"
  | "__builtin_catanh"
  | "__builtin_catanhf"
  | "__builtin_catanhl"
  | "__builtin_catanl"
  | "__builtin_cbrt"
  | "__builtin_cbrtf"
  | "__builtin_cbrtl"
  | "__builtin_ccos"
  | "__builtin_ccosf"
  | "__builtin_ccosh"
  | "__builtin_ccoshf"
  | "__builtin_ccoshl"
  | "__builtin_ccosl"
  | "__builtin_ceil"
  | "__builtin_ceilf"
  | "__builtin_ceill"
  | "__builtin_cexp"
  | "__builtin_cexpf"
  | "__builtin_cexpl"
  | "__builtin_cimag"
  | "__builtin_cimagf"
  | "__builtin_cimagl"
  | "__builtin_clog"
  | "__builtin_clogf"
  | "__builtin_clogl"
  | "__builtin_clrsb"
  | "__builtin_clrsbl"
  | "__builtin_clrsbll"
  | "__builtin_clz"
  | "__builtin_clzg"
  | "__builtin_clzl"
  | "__builtin_clzll"
  | "__builtin_clzs"
  | "__builtin_complex"
  | "__builtin_conj"
  | "__builtin_conjf"
  | "__builtin_conjl"
  | "__builtin_constant_p"
  | "__builtin_copysign"
  | "__builtin_copysignf"
  | "__builtin_copysignl"
  | "__builtin_coro_destroy"
  | "__builtin_coro_done"
  | "__builtin_coro_noop"
  | "__builtin_coro_promise"
  | "__builtin_coro_resume"
  | "__builtin_cos"
  | "__builtin_cosf"
  | "__builtin_cosh"
  | "__builtin_coshf"
  | "__builtin_coshl"
  | "__builtin_cosl"
  | "__builtin_cpow"
  | "__builtin_cpowf"
  | "__builtin_cpowl"
  | "__builtin_cproj"
  | "__builtin_cprojf"
  | "__builtin_cprojl"
  | "__builtin_creal"
  | "__builtin_crealf"
  | "__builtin_creall"
  | "__builtin_csin"
  | "__builtin_csinf"
  | "__builtin_csinh"
  | "__builtin_csinhf"
  | "__builtin_csinhl"
  | "__builtin_csinl"
  | "__builtin_csqrt"
  | "__builtin_csqrtf"
  | "__builtin_csqrtl"
  | "__builtin_ctan"
  | "__builtin_ctanf"
  | "__builtin_ctanh"
  | "__builtin_ctanhf"
  | "__builtin_ctanhl"
  | "__builtin_ctanl"
  | "__builtin_ctz"
  | "__builtin_ctzg"
  | "__builtin_ctzl"
  | "__builtin_ctzll"
  | "__builtin_ctzs"
  | "__builtin_erf"
  | "__builtin_erfc"
  | "__builtin_erfcf"
  | "__builtin_erfcl"
  | "__builtin_erff"
  | "__builtin_erfl"
  | "__builtin_exit"
  | "__builtin_exp"
  | "__builtin_exp2"
  | "__builtin_exp2f"
  | "__builtin_exp2l"
  | "__builtin_expect"
  | "__builtin_expf"
  | "__builtin_expl"
  | "__builtin_expm1"
  | "__builtin_expm1f"
  | "__builtin_expm1l"
  | "__builtin_fabs"
  | "__builtin_fabsf"
  | "__builtin_fabsf128"
  | "__builtin_fabsl"
  | "__builtin_fdim"
  | "__builtin_fdimf"
  | "__builtin_fdiml"
  | "__builtin_ffs"
  | "__builtin_ffsl"
  | "__builtin_ffsll"
  | "__builtin_finite"
  | "__builtin_finitef"
  | "__builtin_finitel"
  | "__builtin_floor"
  | "__builtin_floorf"
  | "__builtin_floorl"
  | "__builtin_fma"
  | "__builtin_fmaf"
  | "__builtin_fmal"
  | "__builtin_fmax"
  | "__builtin_fmaxf"
  | "__builtin_fmaximum_num"
  | "__builtin_fmaximum_numf"
  | "__builtin_fmaximum_numl"
  | "__builtin_fmaxl"
  | "__builtin_fmin"
  | "__builtin_fminf"
  | "__builtin_fminimum_num"
  | "__builtin_fminimum_numf"
  | "__builtin_fminimum_numl"
  | "__builtin_fminl"
  | "__builtin_fmod"
  | "__builtin_fmodf"
  | "__builtin_fmodl"
  | "__builtin_fpclassify"
  | "__builtin_frexp"
  | "__builtin_frexpf"
  | "__builtin_frexpl"
  | "__builtin_huge_val"
  | "__builtin_huge_valf"
  | "__builtin_huge_vall"
  | "__builtin_hypot"
  | "__builtin_hypotf"
  | "__builtin_hypotl"
  | "__builtin_ia32_pause"
  | "__builtin_ilogb"
  | "__builtin_ilogbf"
  | "__builtin_ilogbl"
  | "__builtin_index"
  | "__builtin_inf"
  | "__builtin_inff"
  | "__builtin_infl"
  | "__builtin_invoke"
  | "__builtin_is_constant_evaluated"
  | "__builtin_isalnum"
  | "__builtin_isalpha"
  | "__builtin_isblank"
  | "__builtin_iscntrl"
  | "__builtin_isdigit"
  | "__builtin_isfinite"
  | "__builtin_isgraph"
  | "__builtin_isgreater"
  | "__builtin_isgreaterequal"
  | "__builtin_isinf"
  | "__builtin_isless"
  | "__builtin_islessequal"
  | "__builtin_islessgreater"
  | "__builtin_islower"
  | "__builtin_isnan"
  | "__builtin_isnormal"
  | "__builtin_isprint"
  | "__builtin_ispunct"
  | "__builtin_isspace"
  | "__builtin_isunordered"
  | "__builtin_isupper"
  | "__builtin_isxdigit"
  | "__builtin_labs"
  | "__builtin_ldexp"
  | "__builtin_ldexpf"
  | "__builtin_ldexpl"
  | "__builtin_lgamma"
  | "__builtin_lgammaf"
  | "__builtin_lgammal"
  | "__builtin_llabs"
  | "__builtin_llrint"
  | "__builtin_llrintf"
  | "__builtin_llrintl"
  | "__builtin_llround"
  | "__builtin_llroundf"
  | "__builtin_llroundl"
  | "__builtin_log"
  | "__builtin_log10"
  | "__builtin_log10f"
  | "__builtin_log10l"
  | "__builtin_log1p"
  | "__builtin_log1pf"
  | "__builtin_log1pl"
  | "__builtin_log2"
  | "__builtin_log2f"
  | "__builtin_log2l"
  | "__builtin_logb"
  | "__builtin_logbf"
  | "__builtin_logbl"
  | "__builtin_logf"
  | "__builtin_logl"
  | "__builtin_lrint"
  | "__builtin_lrintf"
  | "__builtin_lrintl"
  | "__builtin_lround"
  | "__builtin_lroundf"
  | "__builtin_lroundl"
  | "__builtin_memccpy"
  | "__builtin_memchr"
  | "__builtin_memcmp"
  | "__builtin_memcpy"
  | "__builtin_memmove"
  | "__builtin_mempcpy"
  | "__builtin_memset"
  | "__builtin_modf"
  | "__builtin_modff"
  | "__builtin_modfl"
  | "__builtin_mul_overflow"
  | "__builtin_nan"
  | "__builtin_nanf"
  | "__builtin_nanl"
  | "__builtin_nans"
  | "__builtin_nansf"
  | "__builtin_nansl"
  | "__builtin_nearbyint"
  | "__builtin_nearbyintf"
  | "__builtin_nearbyintl"
  | "__builtin_nextafter"
  | "__builtin_nextafterf"
  | "__builtin_nextafterl"
  | "__builtin_nexttoward"
  | "__builtin_nexttowardf"
  | "__builtin_nexttowardl"
  | "__builtin_operator_delete"
  | "__builtin_operator_new"
  | "__builtin_parity"
  | "__builtin_parityl"
  | "__builtin_parityll"
  | "__builtin_popcount"
  | "__builtin_popcountg"
  | "__builtin_popcountl"
  | "__builtin_popcountll"
  | "__builtin_pow"
  | "__builtin_powf"
  | "__builtin_powl"
  | "__builtin_reduce_add"
  | "__builtin_reduce_and"
  | "__builtin_reduce_mul"
  | "__builtin_reduce_or"
  | "__builtin_reduce_xor"
  | "__builtin_remainder"
  | "__builtin_remainderf"
  | "__builtin_remainderl"
  | "__builtin_remquo"
  | "__builtin_remquof"
  | "__builtin_remquol"
  | "__builtin_rindex"
  | "__builtin_rint"
  | "__builtin_rintf"
  | "__builtin_rintl"
  | "__builtin_round"
  | "__builtin_roundeven"
  | "__builtin_roundevenf"
  | "__builtin_roundevenl"
  | "__builtin_roundf"
  | "__builtin_roundl"
  | "__builtin_scalbln"
  | "__builtin_scalblnf"
  | "__builtin_scalblnl"
  | "__builtin_scalbn"
  | "__builtin_scalbnf"
  | "__builtin_scalbnl"
  | "__builtin_signbit"
  | "__builtin_sin"
  | "__builtin_sincos"
  | "__builtin_sincosf"
  | "__builtin_sincosl"
  | "__builtin_sinf"
  | "__builtin_sinh"
  | "__builtin_sinhf"
  | "__builtin_sinhl"
  | "__builtin_sinl"
  | "__builtin_source_location"
  | "__builtin_sqrt"
  | "__builtin_sqrtf"
  | "__builtin_sqrtl"
  | "__builtin_stpcpy"
  | "__builtin_stpncpy"
  | "__builtin_strcasecmp"
  | "__builtin_strcat"
  | "__builtin_strchr"
  | "__builtin_strcmp"
  | "__builtin_strcpy"
  | "__builtin_strcspn"
  | "__builtin_strdup"
  | "__builtin_strerror"
  | "__builtin_strlcat"
  | "__builtin_strlcpy"
  | "__builtin_strlen"
  | "__builtin_strncasecmp"
  | "__builtin_strncat"
  | "__builtin_strncmp"
  | "__builtin_strncpy"
  | "__builtin_strndup"
  | "__builtin_strpbrk"
  | "__builtin_strrchr"
  | "__builtin_strspn"
  | "__builtin_strstr"
  | "__builtin_strtod"
  | "__builtin_strtof"
  | "__builtin_strtok"
  | "__builtin_strtol"
  | "__builtin_strtold"
  | "__builtin_strtoll"
  | "__builtin_strtoul"
  | "__builtin_strtoull"
  | "__builtin_strxfrm"
  | "__builtin_sub_overflow"
  | "__builtin_tan"
  | "__builtin_tanf"
  | "__builtin_tanh"
  | "__builtin_tanhf"
  | "__builtin_tanhl"
  | "__builtin_tanl"
  | "__builtin_tgamma"
  | "__builtin_tgammaf"
  | "__builtin_tgammal"
  | "__builtin_tolower"
  | "__builtin_toupper"
  | "__builtin_trap"
  | "__builtin_trunc"
  | "__builtin_truncf"
  | "__builtin_truncl"
  | "__builtin_unreachable"
  | "__builtin_va_copy"
  | "__builtin_va_end"
  | "__builtin_va_start"
  | "__builtin_vsnprintf"
  | "__builtin_wcschr"
  | "__builtin_wcscmp"
  | "__builtin_wcslen"
  | "__builtin_wcsncmp"
  | "__builtin_wmemchr"
  | "__builtin_wmemcmp"
  | "__builtin_wmemcpy"
  | "__builtin_wmemmove"
  | "__c11_atomic_compare_exchange_strong"
  | "__c11_atomic_compare_exchange_weak"
  | "__c11_atomic_exchange"
  | "__c11_atomic_fetch_add"
  | "__c11_atomic_fetch_and"
  | "__c11_atomic_fetch_nand"
  | "__c11_atomic_fetch_or"
  | "__c11_atomic_fetch_sub"
  | "__c11_atomic_fetch_xor"
  | "__c11_atomic_init"
  | "__c11_atomic_is_lock_free"
  | "__c11_atomic_load"
  | "__c11_atomic_signal_fence"
  | "__c11_atomic_store"
  | "__c11_atomic_thread_fence";
const builtinFunctionKindNames: Record<number, BuiltinFunctionKind> = {
  0: "none",
  1: "__atomic_add_fetch",
  2: "__atomic_always_lock_free",
  3: "__atomic_and_fetch",
  4: "__atomic_clear",
  5: "__atomic_compare_exchange",
  6: "__atomic_compare_exchange_n",
  7: "__atomic_exchange",
  8: "__atomic_exchange_n",
  9: "__atomic_fetch_add",
  10: "__atomic_fetch_and",
  11: "__atomic_fetch_nand",
  12: "__atomic_fetch_or",
  13: "__atomic_fetch_sub",
  14: "__atomic_fetch_xor",
  15: "__atomic_is_lock_free",
  16: "__atomic_load",
  17: "__atomic_load_n",
  18: "__atomic_nand_fetch",
  19: "__atomic_or_fetch",
  20: "__atomic_signal_fence",
  21: "__atomic_store",
  22: "__atomic_store_n",
  23: "__atomic_sub_fetch",
  24: "__atomic_test_and_set",
  25: "__atomic_thread_fence",
  26: "__atomic_xor_fetch",
  27: "__builtin_COLUMN",
  28: "__builtin_FILE",
  29: "__builtin_FUNCTION",
  30: "__builtin_LINE",
  31: "__builtin__Exit",
  32: "__builtin___cospi",
  33: "__builtin___cospif",
  34: "__builtin___exp10",
  35: "__builtin___exp10f",
  36: "__builtin___finite",
  37: "__builtin___finitef",
  38: "__builtin___finitel",
  39: "__builtin___sinpi",
  40: "__builtin___sinpif",
  41: "__builtin___tanpi",
  42: "__builtin___tanpif",
  43: "__builtin_abort",
  44: "__builtin_abs",
  45: "__builtin_acos",
  46: "__builtin_acosf",
  47: "__builtin_acosh",
  48: "__builtin_acoshf",
  49: "__builtin_acoshl",
  50: "__builtin_acosl",
  51: "__builtin_add_overflow",
  52: "__builtin_addressof",
  53: "__builtin_aligned_alloc",
  54: "__builtin_alloca",
  55: "__builtin_alloca_with_align",
  56: "__builtin_asin",
  57: "__builtin_asinf",
  58: "__builtin_asinh",
  59: "__builtin_asinhf",
  60: "__builtin_asinhl",
  61: "__builtin_asinl",
  62: "__builtin_assume_aligned",
  63: "__builtin_atan",
  64: "__builtin_atan2",
  65: "__builtin_atan2f",
  66: "__builtin_atan2l",
  67: "__builtin_atanf",
  68: "__builtin_atanh",
  69: "__builtin_atanhf",
  70: "__builtin_atanhl",
  71: "__builtin_atanl",
  72: "__builtin_bcmp",
  73: "__builtin_bcopy",
  74: "__builtin_bswap16",
  75: "__builtin_bswap32",
  76: "__builtin_bswap64",
  77: "__builtin_bzero",
  78: "__builtin_c23_va_start",
  79: "__builtin_cabs",
  80: "__builtin_cabsf",
  81: "__builtin_cabsl",
  82: "__builtin_cacos",
  83: "__builtin_cacosf",
  84: "__builtin_cacosh",
  85: "__builtin_cacoshf",
  86: "__builtin_cacoshl",
  87: "__builtin_cacosl",
  88: "__builtin_carg",
  89: "__builtin_cargf",
  90: "__builtin_cargl",
  91: "__builtin_casin",
  92: "__builtin_casinf",
  93: "__builtin_casinh",
  94: "__builtin_casinhf",
  95: "__builtin_casinhl",
  96: "__builtin_casinl",
  97: "__builtin_catan",
  98: "__builtin_catanf",
  99: "__builtin_catanh",
  100: "__builtin_catanhf",
  101: "__builtin_catanhl",
  102: "__builtin_catanl",
  103: "__builtin_cbrt",
  104: "__builtin_cbrtf",
  105: "__builtin_cbrtl",
  106: "__builtin_ccos",
  107: "__builtin_ccosf",
  108: "__builtin_ccosh",
  109: "__builtin_ccoshf",
  110: "__builtin_ccoshl",
  111: "__builtin_ccosl",
  112: "__builtin_ceil",
  113: "__builtin_ceilf",
  114: "__builtin_ceill",
  115: "__builtin_cexp",
  116: "__builtin_cexpf",
  117: "__builtin_cexpl",
  118: "__builtin_cimag",
  119: "__builtin_cimagf",
  120: "__builtin_cimagl",
  121: "__builtin_clog",
  122: "__builtin_clogf",
  123: "__builtin_clogl",
  124: "__builtin_clrsb",
  125: "__builtin_clrsbl",
  126: "__builtin_clrsbll",
  127: "__builtin_clz",
  128: "__builtin_clzg",
  129: "__builtin_clzl",
  130: "__builtin_clzll",
  131: "__builtin_clzs",
  132: "__builtin_complex",
  133: "__builtin_conj",
  134: "__builtin_conjf",
  135: "__builtin_conjl",
  136: "__builtin_constant_p",
  137: "__builtin_copysign",
  138: "__builtin_copysignf",
  139: "__builtin_copysignl",
  140: "__builtin_coro_destroy",
  141: "__builtin_coro_done",
  142: "__builtin_coro_noop",
  143: "__builtin_coro_promise",
  144: "__builtin_coro_resume",
  145: "__builtin_cos",
  146: "__builtin_cosf",
  147: "__builtin_cosh",
  148: "__builtin_coshf",
  149: "__builtin_coshl",
  150: "__builtin_cosl",
  151: "__builtin_cpow",
  152: "__builtin_cpowf",
  153: "__builtin_cpowl",
  154: "__builtin_cproj",
  155: "__builtin_cprojf",
  156: "__builtin_cprojl",
  157: "__builtin_creal",
  158: "__builtin_crealf",
  159: "__builtin_creall",
  160: "__builtin_csin",
  161: "__builtin_csinf",
  162: "__builtin_csinh",
  163: "__builtin_csinhf",
  164: "__builtin_csinhl",
  165: "__builtin_csinl",
  166: "__builtin_csqrt",
  167: "__builtin_csqrtf",
  168: "__builtin_csqrtl",
  169: "__builtin_ctan",
  170: "__builtin_ctanf",
  171: "__builtin_ctanh",
  172: "__builtin_ctanhf",
  173: "__builtin_ctanhl",
  174: "__builtin_ctanl",
  175: "__builtin_ctz",
  176: "__builtin_ctzg",
  177: "__builtin_ctzl",
  178: "__builtin_ctzll",
  179: "__builtin_ctzs",
  180: "__builtin_erf",
  181: "__builtin_erfc",
  182: "__builtin_erfcf",
  183: "__builtin_erfcl",
  184: "__builtin_erff",
  185: "__builtin_erfl",
  186: "__builtin_exit",
  187: "__builtin_exp",
  188: "__builtin_exp2",
  189: "__builtin_exp2f",
  190: "__builtin_exp2l",
  191: "__builtin_expect",
  192: "__builtin_expf",
  193: "__builtin_expl",
  194: "__builtin_expm1",
  195: "__builtin_expm1f",
  196: "__builtin_expm1l",
  197: "__builtin_fabs",
  198: "__builtin_fabsf",
  199: "__builtin_fabsf128",
  200: "__builtin_fabsl",
  201: "__builtin_fdim",
  202: "__builtin_fdimf",
  203: "__builtin_fdiml",
  204: "__builtin_ffs",
  205: "__builtin_ffsl",
  206: "__builtin_ffsll",
  207: "__builtin_finite",
  208: "__builtin_finitef",
  209: "__builtin_finitel",
  210: "__builtin_floor",
  211: "__builtin_floorf",
  212: "__builtin_floorl",
  213: "__builtin_fma",
  214: "__builtin_fmaf",
  215: "__builtin_fmal",
  216: "__builtin_fmax",
  217: "__builtin_fmaxf",
  218: "__builtin_fmaximum_num",
  219: "__builtin_fmaximum_numf",
  220: "__builtin_fmaximum_numl",
  221: "__builtin_fmaxl",
  222: "__builtin_fmin",
  223: "__builtin_fminf",
  224: "__builtin_fminimum_num",
  225: "__builtin_fminimum_numf",
  226: "__builtin_fminimum_numl",
  227: "__builtin_fminl",
  228: "__builtin_fmod",
  229: "__builtin_fmodf",
  230: "__builtin_fmodl",
  231: "__builtin_fpclassify",
  232: "__builtin_frexp",
  233: "__builtin_frexpf",
  234: "__builtin_frexpl",
  235: "__builtin_huge_val",
  236: "__builtin_huge_valf",
  237: "__builtin_huge_vall",
  238: "__builtin_hypot",
  239: "__builtin_hypotf",
  240: "__builtin_hypotl",
  241: "__builtin_ia32_pause",
  242: "__builtin_ilogb",
  243: "__builtin_ilogbf",
  244: "__builtin_ilogbl",
  245: "__builtin_index",
  246: "__builtin_inf",
  247: "__builtin_inff",
  248: "__builtin_infl",
  249: "__builtin_invoke",
  250: "__builtin_is_constant_evaluated",
  251: "__builtin_isalnum",
  252: "__builtin_isalpha",
  253: "__builtin_isblank",
  254: "__builtin_iscntrl",
  255: "__builtin_isdigit",
  256: "__builtin_isfinite",
  257: "__builtin_isgraph",
  258: "__builtin_isgreater",
  259: "__builtin_isgreaterequal",
  260: "__builtin_isinf",
  261: "__builtin_isless",
  262: "__builtin_islessequal",
  263: "__builtin_islessgreater",
  264: "__builtin_islower",
  265: "__builtin_isnan",
  266: "__builtin_isnormal",
  267: "__builtin_isprint",
  268: "__builtin_ispunct",
  269: "__builtin_isspace",
  270: "__builtin_isunordered",
  271: "__builtin_isupper",
  272: "__builtin_isxdigit",
  273: "__builtin_labs",
  274: "__builtin_ldexp",
  275: "__builtin_ldexpf",
  276: "__builtin_ldexpl",
  277: "__builtin_lgamma",
  278: "__builtin_lgammaf",
  279: "__builtin_lgammal",
  280: "__builtin_llabs",
  281: "__builtin_llrint",
  282: "__builtin_llrintf",
  283: "__builtin_llrintl",
  284: "__builtin_llround",
  285: "__builtin_llroundf",
  286: "__builtin_llroundl",
  287: "__builtin_log",
  288: "__builtin_log10",
  289: "__builtin_log10f",
  290: "__builtin_log10l",
  291: "__builtin_log1p",
  292: "__builtin_log1pf",
  293: "__builtin_log1pl",
  294: "__builtin_log2",
  295: "__builtin_log2f",
  296: "__builtin_log2l",
  297: "__builtin_logb",
  298: "__builtin_logbf",
  299: "__builtin_logbl",
  300: "__builtin_logf",
  301: "__builtin_logl",
  302: "__builtin_lrint",
  303: "__builtin_lrintf",
  304: "__builtin_lrintl",
  305: "__builtin_lround",
  306: "__builtin_lroundf",
  307: "__builtin_lroundl",
  308: "__builtin_memccpy",
  309: "__builtin_memchr",
  310: "__builtin_memcmp",
  311: "__builtin_memcpy",
  312: "__builtin_memmove",
  313: "__builtin_mempcpy",
  314: "__builtin_memset",
  315: "__builtin_modf",
  316: "__builtin_modff",
  317: "__builtin_modfl",
  318: "__builtin_mul_overflow",
  319: "__builtin_nan",
  320: "__builtin_nanf",
  321: "__builtin_nanl",
  322: "__builtin_nans",
  323: "__builtin_nansf",
  324: "__builtin_nansl",
  325: "__builtin_nearbyint",
  326: "__builtin_nearbyintf",
  327: "__builtin_nearbyintl",
  328: "__builtin_nextafter",
  329: "__builtin_nextafterf",
  330: "__builtin_nextafterl",
  331: "__builtin_nexttoward",
  332: "__builtin_nexttowardf",
  333: "__builtin_nexttowardl",
  334: "__builtin_operator_delete",
  335: "__builtin_operator_new",
  336: "__builtin_parity",
  337: "__builtin_parityl",
  338: "__builtin_parityll",
  339: "__builtin_popcount",
  340: "__builtin_popcountg",
  341: "__builtin_popcountl",
  342: "__builtin_popcountll",
  343: "__builtin_pow",
  344: "__builtin_powf",
  345: "__builtin_powl",
  346: "__builtin_reduce_add",
  347: "__builtin_reduce_and",
  348: "__builtin_reduce_mul",
  349: "__builtin_reduce_or",
  350: "__builtin_reduce_xor",
  351: "__builtin_remainder",
  352: "__builtin_remainderf",
  353: "__builtin_remainderl",
  354: "__builtin_remquo",
  355: "__builtin_remquof",
  356: "__builtin_remquol",
  357: "__builtin_rindex",
  358: "__builtin_rint",
  359: "__builtin_rintf",
  360: "__builtin_rintl",
  361: "__builtin_round",
  362: "__builtin_roundeven",
  363: "__builtin_roundevenf",
  364: "__builtin_roundevenl",
  365: "__builtin_roundf",
  366: "__builtin_roundl",
  367: "__builtin_scalbln",
  368: "__builtin_scalblnf",
  369: "__builtin_scalblnl",
  370: "__builtin_scalbn",
  371: "__builtin_scalbnf",
  372: "__builtin_scalbnl",
  373: "__builtin_signbit",
  374: "__builtin_sin",
  375: "__builtin_sincos",
  376: "__builtin_sincosf",
  377: "__builtin_sincosl",
  378: "__builtin_sinf",
  379: "__builtin_sinh",
  380: "__builtin_sinhf",
  381: "__builtin_sinhl",
  382: "__builtin_sinl",
  383: "__builtin_source_location",
  384: "__builtin_sqrt",
  385: "__builtin_sqrtf",
  386: "__builtin_sqrtl",
  387: "__builtin_stpcpy",
  388: "__builtin_stpncpy",
  389: "__builtin_strcasecmp",
  390: "__builtin_strcat",
  391: "__builtin_strchr",
  392: "__builtin_strcmp",
  393: "__builtin_strcpy",
  394: "__builtin_strcspn",
  395: "__builtin_strdup",
  396: "__builtin_strerror",
  397: "__builtin_strlcat",
  398: "__builtin_strlcpy",
  399: "__builtin_strlen",
  400: "__builtin_strncasecmp",
  401: "__builtin_strncat",
  402: "__builtin_strncmp",
  403: "__builtin_strncpy",
  404: "__builtin_strndup",
  405: "__builtin_strpbrk",
  406: "__builtin_strrchr",
  407: "__builtin_strspn",
  408: "__builtin_strstr",
  409: "__builtin_strtod",
  410: "__builtin_strtof",
  411: "__builtin_strtok",
  412: "__builtin_strtol",
  413: "__builtin_strtold",
  414: "__builtin_strtoll",
  415: "__builtin_strtoul",
  416: "__builtin_strtoull",
  417: "__builtin_strxfrm",
  418: "__builtin_sub_overflow",
  419: "__builtin_tan",
  420: "__builtin_tanf",
  421: "__builtin_tanh",
  422: "__builtin_tanhf",
  423: "__builtin_tanhl",
  424: "__builtin_tanl",
  425: "__builtin_tgamma",
  426: "__builtin_tgammaf",
  427: "__builtin_tgammal",
  428: "__builtin_tolower",
  429: "__builtin_toupper",
  430: "__builtin_trap",
  431: "__builtin_trunc",
  432: "__builtin_truncf",
  433: "__builtin_truncl",
  434: "__builtin_unreachable",
  435: "__builtin_va_copy",
  436: "__builtin_va_end",
  437: "__builtin_va_start",
  438: "__builtin_vsnprintf",
  439: "__builtin_wcschr",
  440: "__builtin_wcscmp",
  441: "__builtin_wcslen",
  442: "__builtin_wcsncmp",
  443: "__builtin_wmemchr",
  444: "__builtin_wmemcmp",
  445: "__builtin_wmemcpy",
  446: "__builtin_wmemmove",
  447: "__c11_atomic_compare_exchange_strong",
  448: "__c11_atomic_compare_exchange_weak",
  449: "__c11_atomic_exchange",
  450: "__c11_atomic_fetch_add",
  451: "__c11_atomic_fetch_and",
  452: "__c11_atomic_fetch_nand",
  453: "__c11_atomic_fetch_or",
  454: "__c11_atomic_fetch_sub",
  455: "__c11_atomic_fetch_xor",
  456: "__c11_atomic_init",
  457: "__c11_atomic_is_lock_free",
  458: "__c11_atomic_load",
  459: "__c11_atomic_signal_fence",
  460: "__c11_atomic_store",
  461: "__c11_atomic_thread_fence",
};
/** @category Enumerations */
export type BuiltinTemplateKind =
  | "none"
  | "__make_integer_seq"
  | "__type_pack_element"
  | "__builtin_common_type";
const builtinTemplateKindNames: Record<number, BuiltinTemplateKind> = {
  0: "none",
  1: "__make_integer_seq",
  2: "__type_pack_element",
  3: "__builtin_common_type",
};
/** @category Enumerations */
export type WellKnownName =
  | "none"
  | "std"
  | "align_val_t"
  | "byte"
  | "destroying_delete_t"
  | "initializer_list"
  | "nothrow_t"
  | "partial_ordering"
  | "source_location"
  | "strong_ordering"
  | "tuple_element"
  | "tuple_size"
  | "type_info"
  | "weak_ordering"
  | "equal"
  | "equivalent"
  | "greater"
  | "less"
  | "unordered"
  | "__func__"
  | "__FUNCTION__"
  | "__PRETTY_FUNCTION__";
const wellKnownNameNames: Record<number, WellKnownName> = {
  0: "none",
  1: "std",
  2: "align_val_t",
  3: "byte",
  4: "destroying_delete_t",
  5: "initializer_list",
  6: "nothrow_t",
  7: "partial_ordering",
  8: "source_location",
  9: "strong_ordering",
  10: "tuple_element",
  11: "tuple_size",
  12: "type_info",
  13: "weak_ordering",
  14: "equal",
  15: "equivalent",
  16: "greater",
  17: "less",
  18: "unordered",
  19: "__func__",
  20: "__FUNCTION__",
  21: "__PRETTY_FUNCTION__",
};
/** @category Enumerations */
export type AccessSpecifier = "Public" | "Protected" | "Private";
const accessSpecifierNames: Record<number, AccessSpecifier> = {
  0: "Public",
  1: "Protected",
  2: "Private",
};
/** @category Enumerations */
export type Severity = "Message" | "Note" | "Warning" | "Error" | "Fatal";
const severityNames: Record<number, Severity> = {
  0: "Message",
  1: "Note",
  2: "Warning",
  3: "Error",
  4: "Fatal",
};
/** @category Enumerations */
export type VTableLayout_SlotKind =
  "Function" | "CompleteDtor" | "DeletingDtor";
const vTableLayout_SlotKindNames: Record<number, VTableLayout_SlotKind> = {
  0: "Function",
  1: "CompleteDtor",
  2: "DeletingDtor",
};
/** @category Enumerations */
export type LanguageKind = "C" | "CXX";
const languageKindNames: Record<number, LanguageKind> = {
  0: "C",
  1: "CXX",
};
/** @category Enumerations */
export type PendingInstantiationState =
  "Unresolved" | "Resolving" | "RecursionDiagnosed" | "Resolved";
const pendingInstantiationStateNames: Record<
  number,
  PendingInstantiationState
> = {
  0: "Unresolved",
  1: "Resolving",
  2: "RecursionDiagnosed",
  3: "Resolved",
};
/** @category Enumerations */
export type CvQualifiers = "None" | "Const" | "Volatile" | "ConstVolatile";
const cvQualifiersNames: Record<number, CvQualifiers> = {
  0: "None",
  1: "Const",
  2: "Volatile",
  3: "ConstVolatile",
};
/** @category Enumerations */
export type RefQualifier = "None" | "Lvalue" | "Rvalue";
const refQualifierNames: Record<number, RefQualifier> = {
  0: "None",
  1: "Lvalue",
  2: "Rvalue",
};
/** @category Enumerations */
export type VectorKind = "Gnu" | "Ext";
const vectorKindNames: Record<number, VectorKind> = {
  0: "Gnu",
  1: "Ext",
};
/** @category Enumerations */
export type VectorSizeKind = "Bytes" | "Elements";
const vectorSizeKindNames: Record<number, VectorSizeKind> = {
  0: "Bytes",
  1: "Elements",
};
/** @category Enumerations */
export type SveTypeKind =
  | "T___SVBFLOAT16_T"
  | "T___SVBOOL_T"
  | "T___SVCOUNT_T"
  | "T___SVFLOAT16_T"
  | "T___SVFLOAT32_T"
  | "T___SVFLOAT64_T"
  | "T___SVINT16_T"
  | "T___SVINT32_T"
  | "T___SVINT64_T"
  | "T___SVINT8_T"
  | "T___SVMFLOAT8_T"
  | "T___SVUINT16_T"
  | "T___SVUINT32_T"
  | "T___SVUINT64_T"
  | "T___SVUINT8_T";
const sveTypeKindNames: Record<number, SveTypeKind> = {
  0: "T___SVBFLOAT16_T",
  1: "T___SVBOOL_T",
  2: "T___SVCOUNT_T",
  3: "T___SVFLOAT16_T",
  4: "T___SVFLOAT32_T",
  5: "T___SVFLOAT64_T",
  6: "T___SVINT16_T",
  7: "T___SVINT32_T",
  8: "T___SVINT64_T",
  9: "T___SVINT8_T",
  10: "T___SVMFLOAT8_T",
  11: "T___SVUINT16_T",
  12: "T___SVUINT32_T",
  13: "T___SVUINT64_T",
  14: "T___SVUINT8_T",
};
const astConstructors: Record<
  ASTKind,
  new (handle: number, owner: ModelOwner, kind: ASTKind) => AST
> = {
  TranslationUnit: TranslationUnitAST,
  ModuleUnit: ModuleUnitAST,
  SimpleDeclaration: SimpleDeclarationAST,
  AsmDeclaration: AsmDeclarationAST,
  NamespaceAliasDefinition: NamespaceAliasDefinitionAST,
  UsingDeclaration: UsingDeclarationAST,
  UsingEnumDeclaration: UsingEnumDeclarationAST,
  UsingDirective: UsingDirectiveAST,
  StaticAssertDeclaration: StaticAssertDeclarationAST,
  AliasDeclaration: AliasDeclarationAST,
  OpaqueEnumDeclaration: OpaqueEnumDeclarationAST,
  FunctionDefinition: FunctionDefinitionAST,
  TemplateDeclaration: TemplateDeclarationAST,
  ConceptDefinition: ConceptDefinitionAST,
  DeductionGuide: DeductionGuideAST,
  ExplicitInstantiation: ExplicitInstantiationAST,
  ExportDeclaration: ExportDeclarationAST,
  ExportCompoundDeclaration: ExportCompoundDeclarationAST,
  LinkageSpecification: LinkageSpecificationAST,
  NamespaceDefinition: NamespaceDefinitionAST,
  EmptyDeclaration: EmptyDeclarationAST,
  AttributeDeclaration: AttributeDeclarationAST,
  ModuleImportDeclaration: ModuleImportDeclarationAST,
  ParameterDeclaration: ParameterDeclarationAST,
  AccessDeclaration: AccessDeclarationAST,
  ForRangeDeclaration: ForRangeDeclarationAST,
  StructuredBindingDeclaration: StructuredBindingDeclarationAST,
  AsmOperand: AsmOperandAST,
  AsmQualifier: AsmQualifierAST,
  AsmClobber: AsmClobberAST,
  AsmGotoLabel: AsmGotoLabelAST,
  Splicer: SplicerAST,
  GlobalModuleFragment: GlobalModuleFragmentAST,
  PrivateModuleFragment: PrivateModuleFragmentAST,
  ModuleDeclaration: ModuleDeclarationAST,
  ModuleName: ModuleNameAST,
  ModuleQualifier: ModuleQualifierAST,
  ModulePartition: ModulePartitionAST,
  ImportName: ImportNameAST,
  InitDeclarator: InitDeclaratorAST,
  Declarator: DeclaratorAST,
  UsingDeclarator: UsingDeclaratorAST,
  Enumerator: EnumeratorAST,
  TypeId: TypeIdAST,
  Handler: HandlerAST,
  BaseSpecifier: BaseSpecifierAST,
  RequiresClause: RequiresClauseAST,
  ParameterDeclarationClause: ParameterDeclarationClauseAST,
  TrailingReturnType: TrailingReturnTypeAST,
  LambdaSpecifier: LambdaSpecifierAST,
  TypeConstraint: TypeConstraintAST,
  AttributeArgumentClause: AttributeArgumentClauseAST,
  Attribute: AttributeAST,
  AttributeUsingPrefix: AttributeUsingPrefixAST,
  NewPlacement: NewPlacementAST,
  NestedNamespaceSpecifier: NestedNamespaceSpecifierAST,
  LabeledStatement: LabeledStatementAST,
  CaseStatement: CaseStatementAST,
  DefaultStatement: DefaultStatementAST,
  ExpressionStatement: ExpressionStatementAST,
  CompoundStatement: CompoundStatementAST,
  IfStatement: IfStatementAST,
  ConstevalIfStatement: ConstevalIfStatementAST,
  SwitchStatement: SwitchStatementAST,
  WhileStatement: WhileStatementAST,
  DoStatement: DoStatementAST,
  ForRangeStatement: ForRangeStatementAST,
  ForStatement: ForStatementAST,
  BreakStatement: BreakStatementAST,
  ContinueStatement: ContinueStatementAST,
  ReturnStatement: ReturnStatementAST,
  CoroutineReturnStatement: CoroutineReturnStatementAST,
  GotoStatement: GotoStatementAST,
  DeclarationStatement: DeclarationStatementAST,
  TryBlockStatement: TryBlockStatementAST,
  CharLiteralExpression: CharLiteralExpressionAST,
  BoolLiteralExpression: BoolLiteralExpressionAST,
  IntLiteralExpression: IntLiteralExpressionAST,
  FloatLiteralExpression: FloatLiteralExpressionAST,
  NullptrLiteralExpression: NullptrLiteralExpressionAST,
  StringLiteralExpression: StringLiteralExpressionAST,
  UserDefinedStringLiteralExpression: UserDefinedStringLiteralExpressionAST,
  ObjectLiteralExpression: ObjectLiteralExpressionAST,
  ThisExpression: ThisExpressionAST,
  PackIndexExpression: PackIndexExpressionAST,
  GenericSelectionExpression: GenericSelectionExpressionAST,
  NestedStatementExpression: NestedStatementExpressionAST,
  DefaultInitializerExpression: DefaultInitializerExpressionAST,
  NestedExpression: NestedExpressionAST,
  IdExpression: IdExpressionAST,
  LambdaExpression: LambdaExpressionAST,
  FoldExpression: FoldExpressionAST,
  RightFoldExpression: RightFoldExpressionAST,
  LeftFoldExpression: LeftFoldExpressionAST,
  RequiresExpression: RequiresExpressionAST,
  VaArgExpression: VaArgExpressionAST,
  SubscriptExpression: SubscriptExpressionAST,
  CallExpression: CallExpressionAST,
  TypeConstruction: TypeConstructionAST,
  BracedTypeConstruction: BracedTypeConstructionAST,
  SpliceMemberExpression: SpliceMemberExpressionAST,
  MemberExpression: MemberExpressionAST,
  PostIncrExpression: PostIncrExpressionAST,
  CppCastExpression: CppCastExpressionAST,
  BuiltinBitCastExpression: BuiltinBitCastExpressionAST,
  BuiltinConvertVectorExpression: BuiltinConvertVectorExpressionAST,
  BuiltinOffsetofExpression: BuiltinOffsetofExpressionAST,
  TypeidExpression: TypeidExpressionAST,
  TypeidOfTypeExpression: TypeidOfTypeExpressionAST,
  SpliceExpression: SpliceExpressionAST,
  GlobalScopeReflectExpression: GlobalScopeReflectExpressionAST,
  NamespaceReflectExpression: NamespaceReflectExpressionAST,
  TypeIdReflectExpression: TypeIdReflectExpressionAST,
  ReflectExpression: ReflectExpressionAST,
  LabelAddressExpression: LabelAddressExpressionAST,
  UnaryExpression: UnaryExpressionAST,
  AwaitExpression: AwaitExpressionAST,
  SizeofExpression: SizeofExpressionAST,
  SizeofTypeExpression: SizeofTypeExpressionAST,
  SizeofPackExpression: SizeofPackExpressionAST,
  AlignofTypeExpression: AlignofTypeExpressionAST,
  AlignofExpression: AlignofExpressionAST,
  NoexceptExpression: NoexceptExpressionAST,
  NewExpression: NewExpressionAST,
  DeleteExpression: DeleteExpressionAST,
  CastExpression: CastExpressionAST,
  ImplicitCastExpression: ImplicitCastExpressionAST,
  ConstExpression: ConstExpressionAST,
  BinaryExpression: BinaryExpressionAST,
  ConditionalExpression: ConditionalExpressionAST,
  YieldExpression: YieldExpressionAST,
  ThrowExpression: ThrowExpressionAST,
  AssignmentExpression: AssignmentExpressionAST,
  TargetExpression: TargetExpressionAST,
  RightExpression: RightExpressionAST,
  CompoundAssignmentExpression: CompoundAssignmentExpressionAST,
  PackExpansionExpression: PackExpansionExpressionAST,
  DesignatedInitializerClause: DesignatedInitializerClauseAST,
  TypeTraitExpression: TypeTraitExpressionAST,
  ConditionExpression: ConditionExpressionAST,
  EqualInitializer: EqualInitializerAST,
  BracedInitList: BracedInitListAST,
  ParenInitializer: ParenInitializerAST,
  ThreeWayComparisonExpression: ThreeWayComparisonExpressionAST,
  DefaultGenericAssociation: DefaultGenericAssociationAST,
  TypeGenericAssociation: TypeGenericAssociationAST,
  DotDesignator: DotDesignatorAST,
  SubscriptDesignator: SubscriptDesignatorAST,
  TemplateTypeParameter: TemplateTypeParameterAST,
  NonTypeTemplateParameter: NonTypeTemplateParameterAST,
  TypenameTypeParameter: TypenameTypeParameterAST,
  ConstraintTypeParameter: ConstraintTypeParameterAST,
  TypedefSpecifier: TypedefSpecifierAST,
  FriendSpecifier: FriendSpecifierAST,
  ConstevalSpecifier: ConstevalSpecifierAST,
  ConstinitSpecifier: ConstinitSpecifierAST,
  ConstexprSpecifier: ConstexprSpecifierAST,
  InlineSpecifier: InlineSpecifierAST,
  NoreturnSpecifier: NoreturnSpecifierAST,
  StaticSpecifier: StaticSpecifierAST,
  ExternSpecifier: ExternSpecifierAST,
  RegisterSpecifier: RegisterSpecifierAST,
  ThreadLocalSpecifier: ThreadLocalSpecifierAST,
  ThreadSpecifier: ThreadSpecifierAST,
  MutableSpecifier: MutableSpecifierAST,
  VirtualSpecifier: VirtualSpecifierAST,
  ExplicitSpecifier: ExplicitSpecifierAST,
  AutoTypeSpecifier: AutoTypeSpecifierAST,
  VoidTypeSpecifier: VoidTypeSpecifierAST,
  SizeTypeSpecifier: SizeTypeSpecifierAST,
  SignTypeSpecifier: SignTypeSpecifierAST,
  BuiltinTypeSpecifier: BuiltinTypeSpecifierAST,
  UnaryBuiltinTypeSpecifier: UnaryBuiltinTypeSpecifierAST,
  BinaryBuiltinTypeSpecifier: BinaryBuiltinTypeSpecifierAST,
  IntegralTypeSpecifier: IntegralTypeSpecifierAST,
  FloatingPointTypeSpecifier: FloatingPointTypeSpecifierAST,
  ComplexTypeSpecifier: ComplexTypeSpecifierAST,
  NamedTypeSpecifier: NamedTypeSpecifierAST,
  AtomicTypeSpecifier: AtomicTypeSpecifierAST,
  BitIntTypeSpecifier: BitIntTypeSpecifierAST,
  UnderlyingTypeSpecifier: UnderlyingTypeSpecifierAST,
  ElaboratedTypeSpecifier: ElaboratedTypeSpecifierAST,
  DecltypeAutoSpecifier: DecltypeAutoSpecifierAST,
  DecltypeSpecifier: DecltypeSpecifierAST,
  PlaceholderTypeSpecifier: PlaceholderTypeSpecifierAST,
  ConstQualifier: ConstQualifierAST,
  VolatileQualifier: VolatileQualifierAST,
  AtomicQualifier: AtomicQualifierAST,
  RestrictQualifier: RestrictQualifierAST,
  EnumSpecifier: EnumSpecifierAST,
  ClassSpecifier: ClassSpecifierAST,
  TypenameSpecifier: TypenameSpecifierAST,
  SplicerTypeSpecifier: SplicerTypeSpecifierAST,
  PointerOperator: PointerOperatorAST,
  ReferenceOperator: ReferenceOperatorAST,
  PtrToMemberOperator: PtrToMemberOperatorAST,
  BitfieldDeclarator: BitfieldDeclaratorAST,
  ParameterPack: ParameterPackAST,
  IdDeclarator: IdDeclaratorAST,
  NestedDeclarator: NestedDeclaratorAST,
  FunctionDeclaratorChunk: FunctionDeclaratorChunkAST,
  ArrayDeclaratorChunk: ArrayDeclaratorChunkAST,
  NameId: NameIdAST,
  DestructorId: DestructorIdAST,
  DecltypeId: DecltypeIdAST,
  OperatorFunctionId: OperatorFunctionIdAST,
  LiteralOperatorId: LiteralOperatorIdAST,
  ConversionFunctionId: ConversionFunctionIdAST,
  SimpleTemplateId: SimpleTemplateIdAST,
  LiteralOperatorTemplateId: LiteralOperatorTemplateIdAST,
  OperatorFunctionTemplateId: OperatorFunctionTemplateIdAST,
  GlobalNestedNameSpecifier: GlobalNestedNameSpecifierAST,
  SimpleNestedNameSpecifier: SimpleNestedNameSpecifierAST,
  DecltypeNestedNameSpecifier: DecltypeNestedNameSpecifierAST,
  TemplateNestedNameSpecifier: TemplateNestedNameSpecifierAST,
  DefaultFunctionBody: DefaultFunctionBodyAST,
  CompoundStatementFunctionBody: CompoundStatementFunctionBodyAST,
  TryStatementFunctionBody: TryStatementFunctionBodyAST,
  DeleteFunctionBody: DeleteFunctionBodyAST,
  TypeTemplateArgument: TypeTemplateArgumentAST,
  ExpressionTemplateArgument: ExpressionTemplateArgumentAST,
  ThrowExceptionSpecifier: ThrowExceptionSpecifierAST,
  NoexceptSpecifier: NoexceptSpecifierAST,
  SimpleRequirement: SimpleRequirementAST,
  CompoundRequirement: CompoundRequirementAST,
  TypeRequirement: TypeRequirementAST,
  NestedRequirement: NestedRequirementAST,
  NewParenInitializer: NewParenInitializerAST,
  NewBracedInitializer: NewBracedInitializerAST,
  ParenMemInitializer: ParenMemInitializerAST,
  BracedMemInitializer: BracedMemInitializerAST,
  ThisLambdaCapture: ThisLambdaCaptureAST,
  DerefThisLambdaCapture: DerefThisLambdaCaptureAST,
  SimpleLambdaCapture: SimpleLambdaCaptureAST,
  RefLambdaCapture: RefLambdaCaptureAST,
  RefInitLambdaCapture: RefInitLambdaCaptureAST,
  InitLambdaCapture: InitLambdaCaptureAST,
  EllipsisExceptionDeclaration: EllipsisExceptionDeclarationAST,
  TypeExceptionDeclaration: TypeExceptionDeclarationAST,
  CxxAttribute: CxxAttributeAST,
  GccAttribute: GccAttributeAST,
  AlignasAttribute: AlignasAttributeAST,
  AlignasTypeAttribute: AlignasTypeAttributeAST,
  AsmAttribute: AsmAttributeAST,
  ScopedAttributeToken: ScopedAttributeTokenAST,
  SimpleAttributeToken: SimpleAttributeTokenAST,
};
const symbolConstructors: Record<
  SymbolKind,
  new (handle: number, owner: ModelOwner, kind: SymbolKind) => Symbol
> = {
  Namespace: NamespaceSymbol,
  Concept: ConceptSymbol,
  DeductionGuide: DeductionGuideSymbol,
  BaseClass: BaseClassSymbol,
  InjectedClassName: InjectedClassNameSymbol,
  Unresolved: UnresolvedSymbol,
  Class: ClassSymbol,
  Enum: EnumSymbol,
  ScopedEnum: ScopedEnumSymbol,
  Function: FunctionSymbol,
  OverloadSet: OverloadSetSymbol,
  Lambda: LambdaSymbol,
  FunctionParameters: FunctionParametersSymbol,
  TemplateParameters: TemplateParametersSymbol,
  Block: BlockSymbol,
  TypeAlias: TypeAliasSymbol,
  Variable: VariableSymbol,
  Field: FieldSymbol,
  Parameter: ParameterSymbol,
  ParameterPack: ParameterPackSymbol,
  TypeParameter: TypeParameterSymbol,
  NonTypeParameter: NonTypeParameterSymbol,
  TemplateTypeParameter: TemplateTypeParameterSymbol,
  ConstraintTypeParameter: ConstraintTypeParameterSymbol,
  Enumerator: EnumeratorSymbol,
  NamespaceAlias: NamespaceAliasSymbol,
  UsingDeclaration: UsingDeclarationSymbol,
};
const typeConstructors: Record<
  TypeKind,
  new (handle: number, owner: ModelOwner, kind: TypeKind) => Type
> = {
  BuiltinVaList: BuiltinVaListType,
  BuiltinMetaInfo: BuiltinMetaInfoType,
  Void: VoidType,
  Nullptr: NullptrType,
  DecltypeAuto: DecltypeAutoType,
  Auto: AutoType,
  Bool: BoolType,
  SignedChar: SignedCharType,
  ShortInt: ShortIntType,
  Int: IntType,
  LongInt: LongIntType,
  LongLongInt: LongLongIntType,
  Int128: Int128Type,
  UnsignedChar: UnsignedCharType,
  UnsignedShortInt: UnsignedShortIntType,
  UnsignedInt: UnsignedIntType,
  UnsignedLongInt: UnsignedLongIntType,
  UnsignedLongLongInt: UnsignedLongLongIntType,
  UnsignedInt128: UnsignedInt128Type,
  Char: CharType,
  Char8: Char8Type,
  Char16: Char16Type,
  Char32: Char32Type,
  WideChar: WideCharType,
  Float: FloatType,
  Double: DoubleType,
  LongDouble: LongDoubleType,
  Float16: Float16Type,
  Float128: Float128Type,
  Qual: QualType,
  BoundedArray: BoundedArrayType,
  UnboundedArray: UnboundedArrayType,
  Pointer: PointerType,
  LvalueReference: LvalueReferenceType,
  RvalueReference: RvalueReferenceType,
  OverloadSet: OverloadSetType,
  Function: FunctionType,
  Class: ClassType,
  Enum: EnumType,
  ScopedEnum: ScopedEnumType,
  MemberObjectPointer: MemberObjectPointerType,
  MemberFunctionPointer: MemberFunctionPointerType,
  Namespace: NamespaceType,
  TypeParameter: TypeParameterType,
  TemplateTypeParameter: TemplateTypeParameterType,
  TemplateTypeParameterSpecialization: TemplateTypeParameterSpecializationType,
  PackExpansion: PackExpansionType,
  Decltype: DecltypeType,
  UnresolvedName: UnresolvedNameType,
  UnresolvedBoundedArray: UnresolvedBoundedArrayType,
  UnresolvedUnderlying: UnresolvedUnderlyingType,
  UnresolvedBuiltin: UnresolvedBuiltinType,
  BitInt: BitIntType,
  UnsignedBitInt: UnsignedBitIntType,
  UnresolvedBitInt: UnresolvedBitIntType,
  Vector: VectorType,
  UnresolvedVector: UnresolvedVectorType,
  Complex: ComplexType,
  Atomic: AtomicType,
  Sve: SveType,
};
const nameConstructors: Record<
  NameKind,
  new (handle: number, owner: ModelOwner, kind: NameKind) => Name
> = {
  Identifier: Identifier,
  OperatorId: OperatorId,
  DestructorId: DestructorId,
  LiteralOperatorId: LiteralOperatorId,
  ConversionFunctionId: ConversionFunctionId,
  TemplateId: TemplateId,
};
const childSlots: Partial<
  Record<ASTKind, ReadonlyArray<readonly [number, boolean, string]>>
> = {
  TranslationUnit: [[TranslationUnitASTSlotBase + 0, true, "declarationList"]],
  ModuleUnit: [
    [ModuleUnitASTSlotBase + 0, false, "globalModuleFragment"],
    [ModuleUnitASTSlotBase + 1, false, "moduleDeclaration"],
    [ModuleUnitASTSlotBase + 2, true, "declarationList"],
    [ModuleUnitASTSlotBase + 3, false, "privateModuleFragment"],
  ],
  SimpleDeclaration: [
    [SimpleDeclarationASTSlotBase + 0, true, "attributeList"],
    [SimpleDeclarationASTSlotBase + 1, true, "declSpecifierList"],
    [SimpleDeclarationASTSlotBase + 2, true, "initDeclaratorList"],
    [SimpleDeclarationASTSlotBase + 3, false, "requiresClause"],
  ],
  AsmDeclaration: [
    [AsmDeclarationASTSlotBase + 0, true, "attributeList"],
    [AsmDeclarationASTSlotBase + 1, true, "asmQualifierList"],
    [AsmDeclarationASTSlotBase + 5, true, "outputOperandList"],
    [AsmDeclarationASTSlotBase + 6, true, "inputOperandList"],
    [AsmDeclarationASTSlotBase + 7, true, "clobberList"],
    [AsmDeclarationASTSlotBase + 8, true, "gotoLabelList"],
  ],
  NamespaceAliasDefinition: [
    [NamespaceAliasDefinitionASTSlotBase + 3, false, "nestedNameSpecifier"],
    [NamespaceAliasDefinitionASTSlotBase + 4, false, "unqualifiedId"],
  ],
  UsingDeclaration: [
    [UsingDeclarationASTSlotBase + 1, true, "usingDeclaratorList"],
  ],
  UsingEnumDeclaration: [
    [UsingEnumDeclarationASTSlotBase + 1, false, "enumTypeSpecifier"],
  ],
  UsingDirective: [
    [UsingDirectiveASTSlotBase + 0, true, "attributeList"],
    [UsingDirectiveASTSlotBase + 3, false, "nestedNameSpecifier"],
    [UsingDirectiveASTSlotBase + 4, false, "unqualifiedId"],
  ],
  StaticAssertDeclaration: [
    [StaticAssertDeclarationASTSlotBase + 2, false, "expression"],
  ],
  AliasDeclaration: [
    [AliasDeclarationASTSlotBase + 2, true, "attributeList"],
    [AliasDeclarationASTSlotBase + 4, true, "gnuAttributeList"],
    [AliasDeclarationASTSlotBase + 5, false, "typeId"],
  ],
  OpaqueEnumDeclaration: [
    [OpaqueEnumDeclarationASTSlotBase + 2, true, "attributeList"],
    [OpaqueEnumDeclarationASTSlotBase + 3, false, "nestedNameSpecifier"],
    [OpaqueEnumDeclarationASTSlotBase + 4, false, "unqualifiedId"],
    [OpaqueEnumDeclarationASTSlotBase + 6, true, "typeSpecifierList"],
  ],
  FunctionDefinition: [
    [FunctionDefinitionASTSlotBase + 0, true, "attributeList"],
    [FunctionDefinitionASTSlotBase + 1, true, "declSpecifierList"],
    [FunctionDefinitionASTSlotBase + 2, false, "declarator"],
    [FunctionDefinitionASTSlotBase + 3, false, "requiresClause"],
    [FunctionDefinitionASTSlotBase + 4, false, "functionBody"],
  ],
  TemplateDeclaration: [
    [TemplateDeclarationASTSlotBase + 2, true, "templateParameterList"],
    [TemplateDeclarationASTSlotBase + 4, false, "requiresClause"],
    [TemplateDeclarationASTSlotBase + 5, false, "declaration"],
  ],
  ConceptDefinition: [[ConceptDefinitionASTSlotBase + 3, false, "expression"]],
  DeductionGuide: [
    [DeductionGuideASTSlotBase + 0, true, "attributeList"],
    [DeductionGuideASTSlotBase + 1, false, "explicitSpecifier"],
    [DeductionGuideASTSlotBase + 4, false, "parameterDeclarationClause"],
    [DeductionGuideASTSlotBase + 7, false, "templateId"],
  ],
  ExplicitInstantiation: [
    [ExplicitInstantiationASTSlotBase + 2, false, "declaration"],
  ],
  ExportDeclaration: [[ExportDeclarationASTSlotBase + 1, false, "declaration"]],
  ExportCompoundDeclaration: [
    [ExportCompoundDeclarationASTSlotBase + 2, true, "declarationList"],
  ],
  LinkageSpecification: [
    [LinkageSpecificationASTSlotBase + 3, true, "declarationList"],
  ],
  NamespaceDefinition: [
    [NamespaceDefinitionASTSlotBase + 2, true, "attributeList"],
    [NamespaceDefinitionASTSlotBase + 3, true, "nestedNamespaceSpecifierList"],
    [NamespaceDefinitionASTSlotBase + 5, true, "extraAttributeList"],
    [NamespaceDefinitionASTSlotBase + 7, true, "declarationList"],
  ],
  EmptyDeclaration: [],
  AttributeDeclaration: [
    [AttributeDeclarationASTSlotBase + 0, true, "attributeList"],
  ],
  ModuleImportDeclaration: [
    [ModuleImportDeclarationASTSlotBase + 1, false, "importName"],
    [ModuleImportDeclarationASTSlotBase + 2, true, "attributeList"],
  ],
  ParameterDeclaration: [
    [ParameterDeclarationASTSlotBase + 0, true, "attributeList"],
    [ParameterDeclarationASTSlotBase + 2, true, "typeSpecifierList"],
    [ParameterDeclarationASTSlotBase + 3, false, "declarator"],
    [ParameterDeclarationASTSlotBase + 5, false, "expression"],
  ],
  AccessDeclaration: [],
  ForRangeDeclaration: [],
  StructuredBindingDeclaration: [
    [StructuredBindingDeclarationASTSlotBase + 0, true, "attributeList"],
    [StructuredBindingDeclarationASTSlotBase + 1, true, "declSpecifierList"],
    [StructuredBindingDeclarationASTSlotBase + 4, true, "bindingList"],
    [StructuredBindingDeclarationASTSlotBase + 6, false, "initializer"],
    [StructuredBindingDeclarationASTSlotBase + 8, false, "hiddenVariable"],
    [
      StructuredBindingDeclarationASTSlotBase + 9,
      true,
      "bindingDeclaratorList",
    ],
  ],
  AsmOperand: [[AsmOperandASTSlotBase + 5, false, "expression"]],
  AsmQualifier: [],
  AsmClobber: [],
  AsmGotoLabel: [],
  Splicer: [[SplicerASTSlotBase + 3, false, "expression"]],
  GlobalModuleFragment: [
    [GlobalModuleFragmentASTSlotBase + 2, true, "declarationList"],
  ],
  PrivateModuleFragment: [
    [PrivateModuleFragmentASTSlotBase + 4, true, "declarationList"],
  ],
  ModuleDeclaration: [
    [ModuleDeclarationASTSlotBase + 2, false, "moduleName"],
    [ModuleDeclarationASTSlotBase + 3, false, "modulePartition"],
    [ModuleDeclarationASTSlotBase + 4, true, "attributeList"],
  ],
  ModuleName: [[ModuleNameASTSlotBase + 0, false, "moduleQualifier"]],
  ModuleQualifier: [[ModuleQualifierASTSlotBase + 0, false, "moduleQualifier"]],
  ModulePartition: [[ModulePartitionASTSlotBase + 1, false, "moduleName"]],
  ImportName: [
    [ImportNameASTSlotBase + 1, false, "modulePartition"],
    [ImportNameASTSlotBase + 2, false, "moduleName"],
  ],
  InitDeclarator: [
    [InitDeclaratorASTSlotBase + 0, false, "declarator"],
    [InitDeclaratorASTSlotBase + 1, false, "requiresClause"],
    [InitDeclaratorASTSlotBase + 2, false, "initializer"],
  ],
  Declarator: [
    [DeclaratorASTSlotBase + 0, true, "ptrOpList"],
    [DeclaratorASTSlotBase + 1, false, "coreDeclarator"],
    [DeclaratorASTSlotBase + 2, true, "declaratorChunkList"],
  ],
  UsingDeclarator: [
    [UsingDeclaratorASTSlotBase + 1, false, "nestedNameSpecifier"],
    [UsingDeclaratorASTSlotBase + 2, false, "unqualifiedId"],
  ],
  Enumerator: [
    [EnumeratorASTSlotBase + 1, true, "attributeList"],
    [EnumeratorASTSlotBase + 3, false, "expression"],
  ],
  TypeId: [
    [TypeIdASTSlotBase + 0, true, "typeSpecifierList"],
    [TypeIdASTSlotBase + 1, true, "attributeList"],
    [TypeIdASTSlotBase + 2, false, "declarator"],
  ],
  Handler: [
    [HandlerASTSlotBase + 2, false, "exceptionDeclaration"],
    [HandlerASTSlotBase + 4, false, "statement"],
  ],
  BaseSpecifier: [
    [BaseSpecifierASTSlotBase + 0, true, "attributeList"],
    [BaseSpecifierASTSlotBase + 3, false, "nestedNameSpecifier"],
    [BaseSpecifierASTSlotBase + 5, false, "unqualifiedId"],
  ],
  RequiresClause: [[RequiresClauseASTSlotBase + 1, false, "expression"]],
  ParameterDeclarationClause: [
    [
      ParameterDeclarationClauseASTSlotBase + 0,
      true,
      "parameterDeclarationList",
    ],
  ],
  TrailingReturnType: [[TrailingReturnTypeASTSlotBase + 1, false, "typeId"]],
  LambdaSpecifier: [],
  TypeConstraint: [
    [TypeConstraintASTSlotBase + 0, false, "nestedNameSpecifier"],
    [TypeConstraintASTSlotBase + 3, true, "templateArgumentList"],
  ],
  AttributeArgumentClause: [
    [AttributeArgumentClauseASTSlotBase + 1, true, "expressionList"],
  ],
  Attribute: [
    [AttributeASTSlotBase + 0, false, "attributeToken"],
    [AttributeASTSlotBase + 1, false, "attributeArgumentClause"],
  ],
  AttributeUsingPrefix: [],
  NewPlacement: [[NewPlacementASTSlotBase + 1, true, "expressionList"]],
  NestedNamespaceSpecifier: [],
  LabeledStatement: [[LabeledStatementASTSlotBase + 2, false, "statement"]],
  CaseStatement: [[CaseStatementASTSlotBase + 1, false, "expression"]],
  DefaultStatement: [],
  ExpressionStatement: [
    [ExpressionStatementASTSlotBase + 0, true, "attributeList"],
    [ExpressionStatementASTSlotBase + 1, false, "expression"],
  ],
  CompoundStatement: [
    [CompoundStatementASTSlotBase + 0, true, "attributeList"],
    [CompoundStatementASTSlotBase + 2, true, "statementList"],
  ],
  IfStatement: [
    [IfStatementASTSlotBase + 0, true, "attributeList"],
    [IfStatementASTSlotBase + 4, false, "initializer"],
    [IfStatementASTSlotBase + 5, false, "condition"],
    [IfStatementASTSlotBase + 7, false, "statement"],
    [IfStatementASTSlotBase + 9, false, "elseStatement"],
  ],
  ConstevalIfStatement: [
    [ConstevalIfStatementASTSlotBase + 0, true, "attributeList"],
    [ConstevalIfStatementASTSlotBase + 4, false, "statement"],
    [ConstevalIfStatementASTSlotBase + 6, false, "elseStatement"],
  ],
  SwitchStatement: [
    [SwitchStatementASTSlotBase + 0, true, "attributeList"],
    [SwitchStatementASTSlotBase + 3, false, "initializer"],
    [SwitchStatementASTSlotBase + 4, false, "condition"],
    [SwitchStatementASTSlotBase + 6, false, "statement"],
  ],
  WhileStatement: [
    [WhileStatementASTSlotBase + 0, true, "attributeList"],
    [WhileStatementASTSlotBase + 3, false, "condition"],
    [WhileStatementASTSlotBase + 5, false, "statement"],
  ],
  DoStatement: [
    [DoStatementASTSlotBase + 0, true, "attributeList"],
    [DoStatementASTSlotBase + 2, false, "statement"],
    [DoStatementASTSlotBase + 5, false, "expression"],
  ],
  ForRangeStatement: [
    [ForRangeStatementASTSlotBase + 0, true, "attributeList"],
    [ForRangeStatementASTSlotBase + 3, false, "initializer"],
    [ForRangeStatementASTSlotBase + 4, false, "rangeDeclaration"],
    [ForRangeStatementASTSlotBase + 6, false, "rangeInitializer"],
    [ForRangeStatementASTSlotBase + 8, false, "statement"],
    [ForRangeStatementASTSlotBase + 9, false, "beginInitializer"],
    [ForRangeStatementASTSlotBase + 10, false, "endInitializer"],
    [ForRangeStatementASTSlotBase + 11, false, "condition"],
    [ForRangeStatementASTSlotBase + 12, false, "increment"],
    [ForRangeStatementASTSlotBase + 13, false, "element"],
  ],
  ForStatement: [
    [ForStatementASTSlotBase + 0, true, "attributeList"],
    [ForStatementASTSlotBase + 3, false, "initializer"],
    [ForStatementASTSlotBase + 4, false, "condition"],
    [ForStatementASTSlotBase + 6, false, "expression"],
    [ForStatementASTSlotBase + 8, false, "statement"],
  ],
  BreakStatement: [[BreakStatementASTSlotBase + 0, true, "attributeList"]],
  ContinueStatement: [
    [ContinueStatementASTSlotBase + 0, true, "attributeList"],
  ],
  ReturnStatement: [
    [ReturnStatementASTSlotBase + 0, true, "attributeList"],
    [ReturnStatementASTSlotBase + 2, false, "expression"],
  ],
  CoroutineReturnStatement: [
    [CoroutineReturnStatementASTSlotBase + 0, true, "attributeList"],
    [CoroutineReturnStatementASTSlotBase + 2, false, "expression"],
  ],
  GotoStatement: [
    [GotoStatementASTSlotBase + 0, true, "attributeList"],
    [GotoStatementASTSlotBase + 1, false, "expression"],
  ],
  DeclarationStatement: [
    [DeclarationStatementASTSlotBase + 0, false, "declaration"],
  ],
  TryBlockStatement: [
    [TryBlockStatementASTSlotBase + 0, true, "attributeList"],
    [TryBlockStatementASTSlotBase + 2, false, "statement"],
    [TryBlockStatementASTSlotBase + 3, true, "handlerList"],
  ],
  CharLiteralExpression: [
    [CharLiteralExpressionASTSlotBase + 2, false, "literalOperatorCall"],
  ],
  BoolLiteralExpression: [],
  IntLiteralExpression: [
    [IntLiteralExpressionASTSlotBase + 2, false, "literalOperatorCall"],
  ],
  FloatLiteralExpression: [
    [FloatLiteralExpressionASTSlotBase + 2, false, "literalOperatorCall"],
  ],
  NullptrLiteralExpression: [],
  StringLiteralExpression: [],
  UserDefinedStringLiteralExpression: [
    [
      UserDefinedStringLiteralExpressionASTSlotBase + 2,
      false,
      "literalOperatorCall",
    ],
  ],
  ObjectLiteralExpression: [
    [ObjectLiteralExpressionASTSlotBase + 1, false, "typeId"],
    [ObjectLiteralExpressionASTSlotBase + 3, false, "bracedInitList"],
  ],
  ThisExpression: [],
  PackIndexExpression: [
    [PackIndexExpressionASTSlotBase + 0, false, "packExpression"],
    [PackIndexExpressionASTSlotBase + 3, false, "indexExpression"],
  ],
  GenericSelectionExpression: [
    [GenericSelectionExpressionASTSlotBase + 2, false, "expression"],
    [GenericSelectionExpressionASTSlotBase + 4, true, "genericAssociationList"],
  ],
  NestedStatementExpression: [
    [NestedStatementExpressionASTSlotBase + 1, false, "statement"],
  ],
  DefaultInitializerExpression: [
    [DefaultInitializerExpressionASTSlotBase + 0, false, "expression"],
  ],
  NestedExpression: [[NestedExpressionASTSlotBase + 1, false, "expression"]],
  IdExpression: [
    [IdExpressionASTSlotBase + 0, false, "nestedNameSpecifier"],
    [IdExpressionASTSlotBase + 2, false, "unqualifiedId"],
  ],
  LambdaExpression: [
    [LambdaExpressionASTSlotBase + 2, true, "captureList"],
    [LambdaExpressionASTSlotBase + 5, true, "templateParameterList"],
    [LambdaExpressionASTSlotBase + 7, false, "templateRequiresClause"],
    [LambdaExpressionASTSlotBase + 8, true, "expressionAttributeList"],
    [LambdaExpressionASTSlotBase + 10, false, "parameterDeclarationClause"],
    [LambdaExpressionASTSlotBase + 12, true, "gnuAtributeList"],
    [LambdaExpressionASTSlotBase + 13, true, "lambdaSpecifierList"],
    [LambdaExpressionASTSlotBase + 14, false, "exceptionSpecifier"],
    [LambdaExpressionASTSlotBase + 15, true, "attributeList"],
    [LambdaExpressionASTSlotBase + 16, false, "trailingReturnType"],
    [LambdaExpressionASTSlotBase + 17, false, "requiresClause"],
    [LambdaExpressionASTSlotBase + 18, false, "statement"],
  ],
  FoldExpression: [
    [FoldExpressionASTSlotBase + 1, false, "leftExpression"],
    [FoldExpressionASTSlotBase + 5, false, "rightExpression"],
  ],
  RightFoldExpression: [
    [RightFoldExpressionASTSlotBase + 1, false, "expression"],
  ],
  LeftFoldExpression: [
    [LeftFoldExpressionASTSlotBase + 3, false, "expression"],
  ],
  RequiresExpression: [
    [RequiresExpressionASTSlotBase + 2, false, "parameterDeclarationClause"],
    [RequiresExpressionASTSlotBase + 5, true, "requirementList"],
  ],
  VaArgExpression: [
    [VaArgExpressionASTSlotBase + 2, false, "expression"],
    [VaArgExpressionASTSlotBase + 4, false, "typeId"],
  ],
  SubscriptExpression: [
    [SubscriptExpressionASTSlotBase + 0, false, "baseExpression"],
    [SubscriptExpressionASTSlotBase + 2, false, "indexExpression"],
  ],
  CallExpression: [
    [CallExpressionASTSlotBase + 0, false, "baseExpression"],
    [CallExpressionASTSlotBase + 2, true, "expressionList"],
  ],
  TypeConstruction: [
    [TypeConstructionASTSlotBase + 0, false, "typeSpecifier"],
    [TypeConstructionASTSlotBase + 2, true, "expressionList"],
  ],
  BracedTypeConstruction: [
    [BracedTypeConstructionASTSlotBase + 0, false, "typeSpecifier"],
    [BracedTypeConstructionASTSlotBase + 1, false, "bracedInitList"],
  ],
  SpliceMemberExpression: [
    [SpliceMemberExpressionASTSlotBase + 0, false, "baseExpression"],
    [SpliceMemberExpressionASTSlotBase + 3, false, "splicer"],
  ],
  MemberExpression: [
    [MemberExpressionASTSlotBase + 0, false, "baseExpression"],
    [MemberExpressionASTSlotBase + 2, false, "nestedNameSpecifier"],
    [MemberExpressionASTSlotBase + 4, false, "unqualifiedId"],
  ],
  PostIncrExpression: [
    [PostIncrExpressionASTSlotBase + 0, false, "baseExpression"],
  ],
  CppCastExpression: [
    [CppCastExpressionASTSlotBase + 2, false, "typeId"],
    [CppCastExpressionASTSlotBase + 5, false, "expression"],
  ],
  BuiltinBitCastExpression: [
    [BuiltinBitCastExpressionASTSlotBase + 2, false, "typeId"],
    [BuiltinBitCastExpressionASTSlotBase + 4, false, "expression"],
  ],
  BuiltinConvertVectorExpression: [
    [BuiltinConvertVectorExpressionASTSlotBase + 2, false, "expression"],
    [BuiltinConvertVectorExpressionASTSlotBase + 4, false, "typeId"],
  ],
  BuiltinOffsetofExpression: [
    [BuiltinOffsetofExpressionASTSlotBase + 2, false, "typeId"],
    [BuiltinOffsetofExpressionASTSlotBase + 5, true, "designatorList"],
  ],
  TypeidExpression: [[TypeidExpressionASTSlotBase + 2, false, "expression"]],
  TypeidOfTypeExpression: [
    [TypeidOfTypeExpressionASTSlotBase + 2, false, "typeId"],
  ],
  SpliceExpression: [[SpliceExpressionASTSlotBase + 0, false, "splicer"]],
  GlobalScopeReflectExpression: [],
  NamespaceReflectExpression: [],
  TypeIdReflectExpression: [
    [TypeIdReflectExpressionASTSlotBase + 1, false, "typeId"],
  ],
  ReflectExpression: [[ReflectExpressionASTSlotBase + 1, false, "expression"]],
  LabelAddressExpression: [],
  UnaryExpression: [[UnaryExpressionASTSlotBase + 1, false, "expression"]],
  AwaitExpression: [[AwaitExpressionASTSlotBase + 1, false, "expression"]],
  SizeofExpression: [[SizeofExpressionASTSlotBase + 1, false, "expression"]],
  SizeofTypeExpression: [
    [SizeofTypeExpressionASTSlotBase + 2, false, "typeId"],
  ],
  SizeofPackExpression: [],
  AlignofTypeExpression: [
    [AlignofTypeExpressionASTSlotBase + 2, false, "typeId"],
  ],
  AlignofExpression: [[AlignofExpressionASTSlotBase + 1, false, "expression"]],
  NoexceptExpression: [
    [NoexceptExpressionASTSlotBase + 2, false, "expression"],
  ],
  NewExpression: [
    [NewExpressionASTSlotBase + 2, false, "newPlacement"],
    [NewExpressionASTSlotBase + 4, true, "typeSpecifierList"],
    [NewExpressionASTSlotBase + 5, false, "declarator"],
    [NewExpressionASTSlotBase + 7, false, "newInitalizer"],
  ],
  DeleteExpression: [[DeleteExpressionASTSlotBase + 4, false, "expression"]],
  CastExpression: [
    [CastExpressionASTSlotBase + 1, false, "typeId"],
    [CastExpressionASTSlotBase + 3, false, "expression"],
  ],
  ImplicitCastExpression: [
    [ImplicitCastExpressionASTSlotBase + 0, false, "expression"],
  ],
  ConstExpression: [[ConstExpressionASTSlotBase + 0, false, "expression"]],
  BinaryExpression: [
    [BinaryExpressionASTSlotBase + 0, false, "leftExpression"],
    [BinaryExpressionASTSlotBase + 2, false, "rightExpression"],
  ],
  ConditionalExpression: [
    [ConditionalExpressionASTSlotBase + 0, false, "condition"],
    [ConditionalExpressionASTSlotBase + 2, false, "iftrueExpression"],
    [ConditionalExpressionASTSlotBase + 4, false, "iffalseExpression"],
  ],
  YieldExpression: [[YieldExpressionASTSlotBase + 1, false, "expression"]],
  ThrowExpression: [[ThrowExpressionASTSlotBase + 1, false, "expression"]],
  AssignmentExpression: [
    [AssignmentExpressionASTSlotBase + 0, false, "leftExpression"],
    [AssignmentExpressionASTSlotBase + 2, false, "rightExpression"],
  ],
  TargetExpression: [],
  RightExpression: [],
  CompoundAssignmentExpression: [
    [CompoundAssignmentExpressionASTSlotBase + 0, false, "targetExpression"],
    [CompoundAssignmentExpressionASTSlotBase + 2, false, "leftExpression"],
    [CompoundAssignmentExpressionASTSlotBase + 3, false, "rightExpression"],
    [CompoundAssignmentExpressionASTSlotBase + 4, false, "adjustExpression"],
  ],
  PackExpansionExpression: [
    [PackExpansionExpressionASTSlotBase + 0, false, "expression"],
  ],
  DesignatedInitializerClause: [
    [DesignatedInitializerClauseASTSlotBase + 0, true, "designatorList"],
    [DesignatedInitializerClauseASTSlotBase + 1, false, "initializer"],
  ],
  TypeTraitExpression: [
    [TypeTraitExpressionASTSlotBase + 2, true, "typeIdList"],
  ],
  ConditionExpression: [
    [ConditionExpressionASTSlotBase + 0, true, "attributeList"],
    [ConditionExpressionASTSlotBase + 1, true, "declSpecifierList"],
    [ConditionExpressionASTSlotBase + 2, false, "declarator"],
    [ConditionExpressionASTSlotBase + 3, false, "initializer"],
  ],
  EqualInitializer: [[EqualInitializerASTSlotBase + 1, false, "expression"]],
  BracedInitList: [[BracedInitListASTSlotBase + 1, true, "expressionList"]],
  ParenInitializer: [[ParenInitializerASTSlotBase + 1, true, "expressionList"]],
  ThreeWayComparisonExpression: [
    [ThreeWayComparisonExpressionASTSlotBase + 0, false, "comparison"],
  ],
  DefaultGenericAssociation: [
    [DefaultGenericAssociationASTSlotBase + 2, false, "expression"],
  ],
  TypeGenericAssociation: [
    [TypeGenericAssociationASTSlotBase + 0, false, "typeId"],
    [TypeGenericAssociationASTSlotBase + 2, false, "expression"],
  ],
  DotDesignator: [],
  SubscriptDesignator: [
    [SubscriptDesignatorASTSlotBase + 1, false, "expression"],
  ],
  TemplateTypeParameter: [
    [TemplateTypeParameterASTSlotBase + 2, true, "templateParameterList"],
    [TemplateTypeParameterASTSlotBase + 4, false, "requiresClause"],
    [TemplateTypeParameterASTSlotBase + 9, false, "idExpression"],
  ],
  NonTypeTemplateParameter: [
    [NonTypeTemplateParameterASTSlotBase + 0, false, "declaration"],
  ],
  TypenameTypeParameter: [
    [TypenameTypeParameterASTSlotBase + 4, false, "typeId"],
  ],
  ConstraintTypeParameter: [
    [ConstraintTypeParameterASTSlotBase + 0, false, "typeConstraint"],
    [ConstraintTypeParameterASTSlotBase + 4, false, "typeId"],
  ],
  TypedefSpecifier: [],
  FriendSpecifier: [],
  ConstevalSpecifier: [],
  ConstinitSpecifier: [],
  ConstexprSpecifier: [],
  InlineSpecifier: [],
  NoreturnSpecifier: [],
  StaticSpecifier: [],
  ExternSpecifier: [],
  RegisterSpecifier: [],
  ThreadLocalSpecifier: [],
  ThreadSpecifier: [],
  MutableSpecifier: [],
  VirtualSpecifier: [],
  ExplicitSpecifier: [[ExplicitSpecifierASTSlotBase + 2, false, "expression"]],
  AutoTypeSpecifier: [],
  VoidTypeSpecifier: [],
  SizeTypeSpecifier: [],
  SignTypeSpecifier: [],
  BuiltinTypeSpecifier: [],
  UnaryBuiltinTypeSpecifier: [
    [UnaryBuiltinTypeSpecifierASTSlotBase + 2, false, "typeId"],
  ],
  BinaryBuiltinTypeSpecifier: [
    [BinaryBuiltinTypeSpecifierASTSlotBase + 2, false, "leftTypeId"],
    [BinaryBuiltinTypeSpecifierASTSlotBase + 4, false, "rightTypeId"],
  ],
  IntegralTypeSpecifier: [],
  FloatingPointTypeSpecifier: [],
  ComplexTypeSpecifier: [],
  NamedTypeSpecifier: [
    [NamedTypeSpecifierASTSlotBase + 0, false, "nestedNameSpecifier"],
    [NamedTypeSpecifierASTSlotBase + 2, false, "unqualifiedId"],
  ],
  AtomicTypeSpecifier: [[AtomicTypeSpecifierASTSlotBase + 2, false, "typeId"]],
  BitIntTypeSpecifier: [
    [BitIntTypeSpecifierASTSlotBase + 2, false, "sizeExpression"],
  ],
  UnderlyingTypeSpecifier: [
    [UnderlyingTypeSpecifierASTSlotBase + 2, false, "typeId"],
  ],
  ElaboratedTypeSpecifier: [
    [ElaboratedTypeSpecifierASTSlotBase + 1, true, "attributeList"],
    [ElaboratedTypeSpecifierASTSlotBase + 2, false, "nestedNameSpecifier"],
    [ElaboratedTypeSpecifierASTSlotBase + 4, false, "unqualifiedId"],
  ],
  DecltypeAutoSpecifier: [],
  DecltypeSpecifier: [[DecltypeSpecifierASTSlotBase + 2, false, "expression"]],
  PlaceholderTypeSpecifier: [
    [PlaceholderTypeSpecifierASTSlotBase + 0, false, "typeConstraint"],
    [PlaceholderTypeSpecifierASTSlotBase + 1, false, "specifier"],
  ],
  ConstQualifier: [],
  VolatileQualifier: [],
  AtomicQualifier: [],
  RestrictQualifier: [],
  EnumSpecifier: [
    [EnumSpecifierASTSlotBase + 2, true, "attributeList"],
    [EnumSpecifierASTSlotBase + 3, false, "nestedNameSpecifier"],
    [EnumSpecifierASTSlotBase + 4, false, "unqualifiedId"],
    [EnumSpecifierASTSlotBase + 6, true, "typeSpecifierList"],
    [EnumSpecifierASTSlotBase + 8, true, "enumeratorList"],
  ],
  ClassSpecifier: [
    [ClassSpecifierASTSlotBase + 1, true, "attributeList"],
    [ClassSpecifierASTSlotBase + 2, false, "nestedNameSpecifier"],
    [ClassSpecifierASTSlotBase + 3, false, "unqualifiedId"],
    [ClassSpecifierASTSlotBase + 6, true, "baseSpecifierList"],
    [ClassSpecifierASTSlotBase + 8, true, "declarationList"],
    [ClassSpecifierASTSlotBase + 10, true, "trailingAttributeList"],
  ],
  TypenameSpecifier: [
    [TypenameSpecifierASTSlotBase + 1, false, "nestedNameSpecifier"],
    [TypenameSpecifierASTSlotBase + 3, false, "unqualifiedId"],
  ],
  SplicerTypeSpecifier: [
    [SplicerTypeSpecifierASTSlotBase + 1, false, "splicer"],
  ],
  PointerOperator: [
    [PointerOperatorASTSlotBase + 1, true, "attributeList"],
    [PointerOperatorASTSlotBase + 2, true, "cvQualifierList"],
  ],
  ReferenceOperator: [
    [ReferenceOperatorASTSlotBase + 1, true, "attributeList"],
  ],
  PtrToMemberOperator: [
    [PtrToMemberOperatorASTSlotBase + 0, false, "nestedNameSpecifier"],
    [PtrToMemberOperatorASTSlotBase + 2, true, "attributeList"],
    [PtrToMemberOperatorASTSlotBase + 3, true, "cvQualifierList"],
  ],
  BitfieldDeclarator: [
    [BitfieldDeclaratorASTSlotBase + 0, false, "unqualifiedId"],
    [BitfieldDeclaratorASTSlotBase + 1, true, "attributeList"],
    [BitfieldDeclaratorASTSlotBase + 3, false, "sizeExpression"],
    [BitfieldDeclaratorASTSlotBase + 4, true, "trailingAttributeList"],
  ],
  ParameterPack: [[ParameterPackASTSlotBase + 1, false, "coreDeclarator"]],
  IdDeclarator: [
    [IdDeclaratorASTSlotBase + 0, false, "nestedNameSpecifier"],
    [IdDeclaratorASTSlotBase + 2, false, "unqualifiedId"],
    [IdDeclaratorASTSlotBase + 3, true, "attributeList"],
  ],
  NestedDeclarator: [[NestedDeclaratorASTSlotBase + 1, false, "declarator"]],
  FunctionDeclaratorChunk: [
    [
      FunctionDeclaratorChunkASTSlotBase + 1,
      false,
      "parameterDeclarationClause",
    ],
    [FunctionDeclaratorChunkASTSlotBase + 3, true, "cvQualifierList"],
    [FunctionDeclaratorChunkASTSlotBase + 5, false, "exceptionSpecifier"],
    [FunctionDeclaratorChunkASTSlotBase + 6, true, "attributeList"],
    [FunctionDeclaratorChunkASTSlotBase + 7, false, "trailingReturnType"],
  ],
  ArrayDeclaratorChunk: [
    [ArrayDeclaratorChunkASTSlotBase + 1, true, "typeQualifierList"],
    [ArrayDeclaratorChunkASTSlotBase + 2, false, "expression"],
    [ArrayDeclaratorChunkASTSlotBase + 4, true, "attributeList"],
  ],
  NameId: [],
  DestructorId: [[DestructorIdASTSlotBase + 1, false, "id"]],
  DecltypeId: [[DecltypeIdASTSlotBase + 0, false, "decltypeSpecifier"]],
  OperatorFunctionId: [],
  LiteralOperatorId: [],
  ConversionFunctionId: [
    [ConversionFunctionIdASTSlotBase + 1, false, "typeId"],
  ],
  SimpleTemplateId: [
    [SimpleTemplateIdASTSlotBase + 2, true, "templateArgumentList"],
  ],
  LiteralOperatorTemplateId: [
    [LiteralOperatorTemplateIdASTSlotBase + 0, false, "literalOperatorId"],
    [LiteralOperatorTemplateIdASTSlotBase + 2, true, "templateArgumentList"],
  ],
  OperatorFunctionTemplateId: [
    [OperatorFunctionTemplateIdASTSlotBase + 0, false, "operatorFunctionId"],
    [OperatorFunctionTemplateIdASTSlotBase + 2, true, "templateArgumentList"],
  ],
  GlobalNestedNameSpecifier: [],
  SimpleNestedNameSpecifier: [
    [SimpleNestedNameSpecifierASTSlotBase + 0, false, "nestedNameSpecifier"],
  ],
  DecltypeNestedNameSpecifier: [
    [DecltypeNestedNameSpecifierASTSlotBase + 0, false, "decltypeSpecifier"],
  ],
  TemplateNestedNameSpecifier: [
    [TemplateNestedNameSpecifierASTSlotBase + 0, false, "nestedNameSpecifier"],
    [TemplateNestedNameSpecifierASTSlotBase + 2, false, "templateId"],
  ],
  DefaultFunctionBody: [],
  CompoundStatementFunctionBody: [
    [CompoundStatementFunctionBodyASTSlotBase + 1, true, "memInitializerList"],
    [CompoundStatementFunctionBodyASTSlotBase + 2, false, "statement"],
  ],
  TryStatementFunctionBody: [
    [TryStatementFunctionBodyASTSlotBase + 2, true, "memInitializerList"],
    [TryStatementFunctionBodyASTSlotBase + 3, false, "statement"],
    [TryStatementFunctionBodyASTSlotBase + 4, true, "handlerList"],
  ],
  DeleteFunctionBody: [],
  TypeTemplateArgument: [
    [TypeTemplateArgumentASTSlotBase + 0, false, "typeId"],
  ],
  ExpressionTemplateArgument: [
    [ExpressionTemplateArgumentASTSlotBase + 0, false, "expression"],
  ],
  ThrowExceptionSpecifier: [],
  NoexceptSpecifier: [[NoexceptSpecifierASTSlotBase + 2, false, "expression"]],
  SimpleRequirement: [[SimpleRequirementASTSlotBase + 0, false, "expression"]],
  CompoundRequirement: [
    [CompoundRequirementASTSlotBase + 1, false, "expression"],
    [CompoundRequirementASTSlotBase + 5, false, "typeConstraint"],
  ],
  TypeRequirement: [[TypeRequirementASTSlotBase + 1, false, "typeId"]],
  NestedRequirement: [[NestedRequirementASTSlotBase + 1, false, "expression"]],
  NewParenInitializer: [
    [NewParenInitializerASTSlotBase + 1, true, "expressionList"],
  ],
  NewBracedInitializer: [
    [NewBracedInitializerASTSlotBase + 0, false, "bracedInitList"],
  ],
  ParenMemInitializer: [
    [ParenMemInitializerASTSlotBase + 0, false, "nestedNameSpecifier"],
    [ParenMemInitializerASTSlotBase + 1, false, "unqualifiedId"],
    [ParenMemInitializerASTSlotBase + 3, true, "expressionList"],
  ],
  BracedMemInitializer: [
    [BracedMemInitializerASTSlotBase + 0, false, "nestedNameSpecifier"],
    [BracedMemInitializerASTSlotBase + 1, false, "unqualifiedId"],
    [BracedMemInitializerASTSlotBase + 2, false, "bracedInitList"],
  ],
  ThisLambdaCapture: [[ThisLambdaCaptureASTSlotBase + 1, false, "initializer"]],
  DerefThisLambdaCapture: [],
  SimpleLambdaCapture: [
    [SimpleLambdaCaptureASTSlotBase + 3, false, "initializer"],
  ],
  RefLambdaCapture: [[RefLambdaCaptureASTSlotBase + 4, false, "initializer"]],
  RefInitLambdaCapture: [
    [RefInitLambdaCaptureASTSlotBase + 3, false, "initializer"],
  ],
  InitLambdaCapture: [[InitLambdaCaptureASTSlotBase + 2, false, "initializer"]],
  EllipsisExceptionDeclaration: [],
  TypeExceptionDeclaration: [
    [TypeExceptionDeclarationASTSlotBase + 0, true, "attributeList"],
    [TypeExceptionDeclarationASTSlotBase + 1, true, "typeSpecifierList"],
    [TypeExceptionDeclarationASTSlotBase + 2, false, "declarator"],
  ],
  CxxAttribute: [
    [CxxAttributeASTSlotBase + 2, false, "attributeUsingPrefix"],
    [CxxAttributeASTSlotBase + 3, true, "attributeList"],
  ],
  GccAttribute: [[GccAttributeASTSlotBase + 3, true, "attributeList"]],
  AlignasAttribute: [[AlignasAttributeASTSlotBase + 2, false, "expression"]],
  AlignasTypeAttribute: [
    [AlignasTypeAttributeASTSlotBase + 2, false, "typeId"],
  ],
  AsmAttribute: [],
  ScopedAttributeToken: [],
  SimpleAttributeToken: [],
};

/** @category Traversal */
export interface ASTChild {
  readonly node: AST;
  readonly key: string | number;
  readonly listKey: string | undefined;
}

/** @category Traversal */
export function* children(node: AST): Generator<ASTChild> {
  for (const [slot, isList, key] of childSlots[node.kind] ?? []) {
    const value = cxx.readAST(node.handle, slot);
    if (!isList) {
      const child = astOf(value, node.modelOwner);
      if (child) yield { node: child, key, listKey: undefined };
      continue;
    }
    let index = 0;
    for (const child of listOf(node.modelOwner, value, (item: any) =>
      astOf(item, node.modelOwner),
    )) {
      if (child) yield { node: child, key: index, listKey: key };
      ++index;
    }
  }
}

/** @category Core */
export function unitOf(owner: ModelOwner): UnitAST {
  return astOf(cxx.getUnitAST(owner.getUnitHandle()), owner);
}
