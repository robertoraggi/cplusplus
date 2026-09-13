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

import {
  BinaryOp,
  CastKind,
  FloatKind,
  FloatPredicate,
  InitializerKind,
  InsertionPointKind,
  IntPredicate,
  Linkage,
  TypeKind,
  UnaryOp,
  type Access,
  type BlockRef,
  type CallInfo,
  type CleanupAction,
  type CleanupRegionRef,
  type CleanupTarget,
  type EmitterDelegate,
  type FunctionInfo,
  type FunctionRef,
  type GlobalInfo,
  type GlobalRef,
  type Initializer,
  type InsertionPoint,
  type InsertionPointRef,
  type ModuleInfo,
  type ModuleRef,
  type TokenIndex,
  type TypeRef,
  type ValueRef,
  type VTableInfo,
} from "./Emitter.js";

const FLOAT_WIDTH: Record<FloatKind, number> = {
  [FloatKind.Half]: 16,
  [FloatKind.Single]: 32,
  [FloatKind.Double]: 64,
  [FloatKind.X87DoubleExtended]: 80,
  [FloatKind.Quad]: 128,
};

interface TypeInfo {
  text: string;
  kind: TypeKind;
  width?: number;
  element?: TypeRef;
  parameters?: TypeRef[];
  results?: TypeRef[];
  members?: TypeRef[];
}

interface ValueInfo {
  type: TypeRef;
  isZero?: boolean;
}

interface BlockInfo {
  owner: FunctionRef;
  parameters: ValueRef[];
  terminated: boolean;
  opened: boolean;
}

interface FunctionBodyScope {
  function: FunctionRef;
  valueMark: ValueRef;
  blockMark: BlockRef;
}

interface FunctionRecord {
  name: string;
  parameters: TypeRef[];
  results: TypeRef[];
  hasBody: boolean;
  linkage: Linkage;
}

interface GlobalRecord {
  name: string;
  linkage: Linkage;
}

export class TraceEmitter implements EmitterDelegate {
  #lines: string[] = [];

  #nextType = 1;
  #nextValue = 1;
  #nextBlock = 1;
  #nextFunction = 1;
  #nextGlobal = 1;
  #nextModule = 1;
  #nextPoint = 1;
  #nextRegion = 1;

  #internedTypes = new Map<string, TypeRef>();
  #types = new Map<TypeRef, TypeInfo>();
  #values = new Map<ValueRef, ValueInfo>();
  #blocks = new Map<BlockRef, BlockInfo>();
  #functions = new Map<FunctionRef, FunctionRecord>();
  #functionsByName = new Map<string, FunctionRef>();
  #globals = new Map<GlobalRef, GlobalRecord>();
  #globalsByName = new Map<string, GlobalRef>();
  #points = new Map<InsertionPointRef, BlockRef>();

  #functionBodies: FunctionBodyScope[] = [];
  #insertionBlock: BlockRef = 0;
  #currentFunction: FunctionRef = 0;

  get trace(): string {
    return this.#lines.join("\n");
  }

  get liveHandles(): { values: number; blocks: number } {
    return { values: this.#values.size, blocks: this.#blocks.size };
  }

  beginFunctionBody(function_: FunctionRef): void {
    this.#functionBodies.push({
      function: function_,
      valueMark: this.#nextValue,
      blockMark: this.#nextBlock,
    });
  }

  endFunctionBody(function_: FunctionRef): void {
    const scope = this.#functionBodies.pop();
    if (!scope || scope.function !== function_)
      throw new Error("unbalanced function body scope");

    for (let value = scope.valueMark; value < this.#nextValue; ++value)
      this.#values.delete(value);

    for (let block = scope.blockMark; block < this.#nextBlock; ++block)
      this.#blocks.delete(block);
  }

  #emit(text: string): void {
    this.#lines.push((this.#insertionBlock ? "  " : "") + text);
  }

  #emitTop(text: string): void {
    this.#lines.push(text);
  }

  #typeText(ref: TypeRef): string {
    return this.#types.get(ref)?.text ?? "?";
  }

  #intern(text: string, info: Omit<TypeInfo, "text">): TypeRef {
    const existing = this.#internedTypes.get(text);
    if (existing !== undefined) return existing;
    const ref = this.#nextType++;
    this.#internedTypes.set(text, ref);
    this.#types.set(ref, { text, ...info });
    return ref;
  }

  #define(type: TypeRef, text: string): ValueRef {
    const ref = this.#nextValue++;
    this.#values.set(ref, { type });
    this.#emit(`%${ref} = ${text} : ${this.#typeText(type)}`);
    return ref;
  }

  #terminate(text: string): void {
    this.#emit(text);
    const block = this.#blocks.get(this.#insertionBlock);
    if (block) block.terminated = true;
  }

  #valueList(values: readonly ValueRef[]): string {
    return values.map((value) => `%${value}`).join(", ");
  }

  beginModule(info: ModuleInfo): ModuleRef {
    const ref = this.#nextModule++;
    this.#emitTop(
      `module "${info.name}"` +
        (info.sourceFile ? ` source "${info.sourceFile}"` : "") +
        (info.targetTriple ? ` triple "${info.targetTriple}"` : ""),
    );
    return ref;
  }

  endModule(): void {
    this.#emitTop("endmodule");
  }

  voidType(): TypeRef {
    return this.#intern("void", { kind: TypeKind.Void });
  }

  unresolvedType(): TypeRef {
    return this.#intern("unresolved", { kind: TypeKind.Unresolved });
  }

  integerType(bits: number): TypeRef {
    return this.#intern(`i${bits}`, { kind: TypeKind.Integer, width: bits });
  }

  floatingType(kind: FloatKind): TypeRef {
    const width = FLOAT_WIDTH[kind];
    return this.#intern(`f${width}`, { kind: TypeKind.Floating, width });
  }

  pointerType(elementType: TypeRef): TypeRef {
    return this.#intern(`ptr<${this.#typeText(elementType)}>`, {
      kind: TypeKind.Pointer,
      element: elementType,
    });
  }

  arrayType(elementType: TypeRef, size: number): TypeRef {
    return this.#intern(`[${size} x ${this.#typeText(elementType)}]`, {
      kind: TypeKind.Array,
      element: elementType,
    });
  }

  vectorType(elementType: TypeRef, elementCount: number): TypeRef {
    return this.#intern(`<${elementCount} x ${this.#typeText(elementType)}>`, {
      kind: TypeKind.Other,
      element: elementType,
    });
  }

  functionType(
    parameters: readonly TypeRef[],
    results: readonly TypeRef[],
    isVariadic: boolean,
  ): TypeRef {
    const inputs = parameters.map((type) => this.#typeText(type));
    if (isVariadic) inputs.push("...");
    const outputs = results.map((type) => this.#typeText(type));
    return this.#intern(`(${inputs.join(", ")}) -> (${outputs.join(", ")})`, {
      kind: TypeKind.Function,
      parameters: [...parameters],
      results: [...results],
    });
  }

  declareClassType(name: string): TypeRef {
    return this.#intern(`!${name}`, { kind: TypeKind.Class });
  }

  defineClassType(
    classType: TypeRef,
    members: readonly TypeRef[],
    isPacked: boolean,
  ): void {
    const info = this.#types.get(classType);
    if (info) info.members = [...members];
    this.#emit(
      `${this.#typeText(classType)} = ${isPacked ? "packed " : ""}{ ` +
        members.map((type) => this.#typeText(type)).join(", ") +
        " }",
    );
  }

  typeKind(type: TypeRef): TypeKind {
    return this.#types.get(type)?.kind ?? TypeKind.Other;
  }

  scalarWidth(type: TypeRef): number {
    return this.#types.get(type)?.width ?? 0;
  }

  elementType(type: TypeRef): TypeRef {
    return this.#types.get(type)?.element ?? 0;
  }

  typeOf(value: ValueRef): TypeRef {
    return this.#values.get(value)?.type ?? 0;
  }

  #initializerText(value: Initializer): string {
    switch (value.kind) {
      case InitializerKind.Integer:
        return String(value.integer);
      case InitializerKind.Floating:
        return String(value.floating);
      case InitializerKind.Bytes:
        return JSON.stringify(String.fromCharCode(...value.bytes));
      case InitializerKind.Aggregate:
        return `{ ${value.elements
          .map((element) => this.#initializerText(element))
          .join(", ")} }`;
      case InitializerKind.Null:
        return "null";
      case InitializerKind.Zero:
        return "zeroinitializer";
      case InitializerKind.ScalarZero:
        return "0";
      case InitializerKind.Undef:
        return "undef";
      case InitializerKind.SignalingNaN:
        return "snan";
      default:
        return "none";
    }
  }

  #isZeroInitializer(value: Initializer): boolean {
    switch (value.kind) {
      case InitializerKind.Integer:
        return value.integer === 0;
      case InitializerKind.Floating:
        return value.floating === 0;
      case InitializerKind.Null:
      case InitializerKind.Zero:
      case InitializerKind.ScalarZero:
        return true;
      default:
        return false;
    }
  }

  constant(loc: TokenIndex, type: TypeRef, value: Initializer): ValueRef {
    const ref = this.#define(type, `const ${this.#initializerText(value)}`);
    this.#values.get(ref)!.isZero = this.#isZeroInitializer(value);
    return ref;
  }

  isZeroConstant(value: ValueRef): boolean {
    return this.#values.get(value)?.isZero ?? false;
  }

  unaryOp(
    loc: TokenIndex,
    op: UnaryOp,
    type: TypeRef,
    value: ValueRef,
  ): ValueRef {
    return this.#define(type, `${UnaryOp[op]} %${value}`);
  }

  binaryOp(
    loc: TokenIndex,
    op: BinaryOp,
    lhs: ValueRef,
    rhs: ValueRef,
  ): ValueRef {
    return this.#define(this.typeOf(lhs), `${BinaryOp[op]} %${lhs}, %${rhs}`);
  }

  compareInt(
    loc: TokenIndex,
    predicate: IntPredicate,
    lhs: ValueRef,
    rhs: ValueRef,
  ): ValueRef {
    return this.#define(
      this.integerType(1),
      `icmp.${IntPredicate[predicate]} %${lhs}, %${rhs}`,
    );
  }

  compareFloat(
    loc: TokenIndex,
    predicate: FloatPredicate,
    lhs: ValueRef,
    rhs: ValueRef,
  ): ValueRef {
    return this.#define(
      this.integerType(1),
      `fcmp.${FloatPredicate[predicate]} %${lhs}, %${rhs}`,
    );
  }

  select(
    loc: TokenIndex,
    condition: ValueRef,
    ifTrue: ValueRef,
    ifFalse: ValueRef,
  ): ValueRef {
    return this.#define(
      this.typeOf(ifTrue),
      `select %${condition}, %${ifTrue}, %${ifFalse}`,
    );
  }

  convert(
    loc: TokenIndex,
    kind: CastKind,
    value: ValueRef,
    type: TypeRef,
  ): ValueRef {
    return this.#define(type, `${CastKind[kind]} %${value}`);
  }

  vectorSplat(
    loc: TokenIndex,
    vectorType: TypeRef,
    scalar: ValueRef,
  ): ValueRef {
    return this.#define(vectorType, `splat %${scalar}`);
  }

  todo(loc: TokenIndex, kind: number, message: string): ValueRef {
    return this.#define(
      this.unresolvedType(),
      `todo ${JSON.stringify(message)}`,
    );
  }

  allocate(
    loc: TokenIndex,
    pointerType: TypeRef,
    size: ValueRef,
    alignment: number,
  ): ValueRef {
    return this.#define(
      pointerType,
      `alloca${size ? ` %${size}` : ""} align ${alignment}`,
    );
  }

  #accessText(access: Access): string {
    const bitfield = access.bitfield
      ? ` bits [${access.bitfield.bitOffset}, ${access.bitfield.bitWidth}]`
      : "";
    return `align ${access.alignment}${bitfield}`;
  }

  load(
    loc: TokenIndex,
    valueType: TypeRef,
    address: ValueRef,
    access: Access,
  ): ValueRef {
    return this.#define(
      valueType,
      `load %${address} ${this.#accessText(access)}` +
        (access.isSigned ? " signed" : ""),
    );
  }

  store(
    loc: TokenIndex,
    value: ValueRef,
    address: ValueRef,
    access: Access,
  ): void {
    this.#emit(`store %${value} -> %${address} ${this.#accessText(access)}`);
  }

  memsetZero(loc: TokenIndex, address: ValueRef, size: number): void {
    this.#emit(`memset.zero %${address} size ${size}`);
  }

  memcpy(
    loc: TokenIndex,
    destination: ValueRef,
    source: ValueRef,
    size: number,
  ): void {
    this.#emit(`memcpy %${destination} <- %${source} size ${size}`);
  }

  pointerAdd(
    loc: TokenIndex,
    pointerType: TypeRef,
    base: ValueRef,
    offset: ValueRef,
  ): ValueRef {
    return this.#define(pointerType, `ptradd %${base}, %${offset}`);
  }

  pointerDiff(
    loc: TokenIndex,
    resultType: TypeRef,
    lhs: ValueRef,
    rhs: ValueRef,
  ): ValueRef {
    return this.#define(resultType, `ptrdiff %${lhs}, %${rhs}`);
  }

  subscript(
    loc: TokenIndex,
    pointerType: TypeRef,
    base: ValueRef,
    index: ValueRef,
  ): ValueRef {
    return this.#define(pointerType, `subscript %${base}[%${index}]`);
  }

  memberAddress(
    loc: TokenIndex,
    pointerType: TypeRef,
    base: ValueRef,
    index: number,
  ): ValueRef {
    return this.#define(pointerType, `member %${base}.${index}`);
  }

  extractValue(
    loc: TokenIndex,
    resultType: TypeRef,
    container: ValueRef,
    position: number,
  ): ValueRef {
    return this.#define(resultType, `extract %${container}[${position}]`);
  }

  insertValue(
    loc: TokenIndex,
    resultType: TypeRef,
    container: ValueRef,
    value: ValueRef,
    position: number,
  ): ValueRef {
    return this.#define(
      resultType,
      `insert %${container}[${position}] = %${value}`,
    );
  }

  addressOfSymbol(
    loc: TokenIndex,
    resultType: TypeRef,
    symbol: string,
  ): ValueRef {
    return this.#define(resultType, `addressof @${symbol}`);
  }

  call(loc: TokenIndex, info: CallInfo): ValueRef[] {
    const callee = info.callee ? `@${info.callee}` : `%${info.indirectCallee}`;
    const text = `call ${callee}(${this.#valueList(info.arguments)})`;
    if (!info.results.length) {
      this.#emit(text);
      return [];
    }
    return info.results.map((type) => this.#define(type, text));
  }

  ret(loc: TokenIndex, values: readonly ValueRef[]): void {
    this.#terminate(`return ${this.#valueList(values)}`.trimEnd());
  }

  unreachable(loc: TokenIndex): void {
    this.#terminate("unreachable");
  }

  createBlock(function_: FunctionRef): BlockRef {
    const owner = function_ || this.#currentFunction;
    const ref = this.#nextBlock++;
    this.#blocks.set(ref, {
      owner,
      parameters: [],
      terminated: false,
      opened: false,
    });
    const fn = this.#functions.get(owner);
    if (fn) fn.hasBody = true;
    return ref;
  }

  eraseBlock(block: BlockRef): void {
    this.#blocks.delete(block);
    if (this.#insertionBlock === block) this.#insertionBlock = 0;
  }

  insertionBlock(): BlockRef {
    return this.#insertionBlock;
  }

  #enterBlock(ref: BlockRef): void {
    this.#insertionBlock = ref;
    if (!ref) return;
    const block = this.#blocks.get(ref);
    if (!block) return;
    this.#currentFunction = block.owner;
    const parameters = block.parameters
      .map((value) => `%${value}: ${this.#typeText(this.typeOf(value))}`)
      .join(", ");
    this.#emitTop(
      `^bb${ref}${parameters ? `(${parameters})` : ""}:` +
        (block.opened ? " resumed" : ""),
    );
    block.opened = true;
  }

  setInsertionPoint(point: InsertionPoint): void {
    if (
      point.kind === InsertionPointKind.ModuleStart ||
      point.kind === InsertionPointKind.ModuleEnd
    ) {
      this.#enterBlock(0);
      return;
    }
    this.#enterBlock(point.block);
  }

  saveInsertionPoint(): InsertionPointRef {
    const ref = this.#nextPoint++;
    this.#points.set(ref, this.#insertionBlock);
    return ref;
  }

  restoreInsertionPoint(point: InsertionPointRef): void {
    const block = this.#points.get(point) ?? 0;
    this.#points.delete(point);
    this.#enterBlock(block);
  }

  hasTerminator(block: BlockRef): boolean {
    return this.#blocks.get(block)?.terminated ?? false;
  }

  addBlockParameter(block: BlockRef, type: TypeRef, loc: TokenIndex): ValueRef {
    const ref = this.#nextValue++;
    this.#values.set(ref, { type });
    this.#blocks.get(block)?.parameters.push(ref);
    return ref;
  }

  blockParameter(block: BlockRef, index: number): ValueRef {
    return this.#blocks.get(block)?.parameters[index] ?? 0;
  }

  blockParameterCount(block: BlockRef): number {
    return this.#blocks.get(block)?.parameters.length ?? 0;
  }

  branch(
    loc: TokenIndex,
    target: BlockRef,
    operands: readonly ValueRef[],
  ): void {
    this.#terminate(
      `br ^bb${target}` +
        (operands.length ? `(${this.#valueList(operands)})` : ""),
    );
  }

  condBranch(
    loc: TokenIndex,
    condition: ValueRef,
    trueDest: BlockRef,
    falseDest: BlockRef,
  ): void {
    this.#terminate(`cond_br %${condition}, ^bb${trueDest}, ^bb${falseDest}`);
  }

  switchBranch(
    loc: TokenIndex,
    flag: ValueRef,
    defaultDest: BlockRef,
    caseValues: readonly number[],
    caseDestinations: readonly BlockRef[],
  ): void {
    const cases = caseValues.map(
      (value, index) => `${value}: ^bb${caseDestinations[index]}`,
    );
    this.#terminate(
      `switch %${flag}, default ^bb${defaultDest} [${cases.join(", ")}]`,
    );
  }

  defineLabel(loc: TokenIndex, name: string, cleanupDepth: number): void {
    this.#emit(`label ${name} depth ${cleanupDepth}`);
  }

  #cleanupText(cleanups: readonly CleanupAction[]): string {
    if (!cleanups.length) return "";
    const actions = cleanups.map(
      (action) =>
        `%${action.address} ~@${action.destructor} depth ${action.depth}` +
        (action.activeFlag ? ` if %${action.activeFlag}` : ""),
    );
    return ` cleanups [${actions.join(", ")}]`;
  }

  branchWithCleanups(
    loc: TokenIndex,
    target: CleanupTarget,
    cleanups: readonly CleanupAction[],
  ): void {
    const destination = target.label ? target.label : `^bb${target.block}`;
    this.#terminate(`br ${destination}${this.#cleanupText(cleanups)}`);
  }

  indirectGoto(loc: TokenIndex, target: ValueRef): void {
    this.#terminate(`indirect_br %${target}`);
  }

  labelAddress(
    loc: TokenIndex,
    type: TypeRef,
    name: string,
    function_: FunctionRef,
  ): ValueRef {
    return this.#define(type, `labeladdr ${name} in @${function_}`);
  }

  resolveFunctionControlFlow(function_: FunctionRef): void {
    this.#emitTop(
      `resolve @${this.#functions.get(function_)?.name ?? function_}`,
    );
  }

  beginCleanupRegion(): CleanupRegionRef {
    return this.#nextRegion++;
  }

  endCleanupRegion(region: CleanupRegionRef): void {}

  activateConditionalCleanup(
    address: ValueRef,
    entry: BlockRef,
    region: CleanupRegionRef,
  ): ValueRef {
    return this.#define(
      this.pointerType(this.integerType(1)),
      `cleanup.flag %${address}`,
    );
  }

  declareFunction(loc: TokenIndex, info: FunctionInfo): FunctionRef {
    const existing = this.#functionsByName.get(info.name);
    if (existing !== undefined) return existing;
    const ref = this.#nextFunction++;
    const type = this.#types.get(info.type);
    this.#functions.set(ref, {
      name: info.name,
      parameters: type?.parameters ?? [],
      results: type?.results ?? [],
      hasBody: false,
      linkage: info.linkage,
    });
    this.#functionsByName.set(info.name, ref);
    this.#emitTop(
      `func @${info.name} : ${this.#typeText(info.type)} ` +
        `${Linkage[info.linkage]}` +
        (info.aliasName ? ` alias @${info.aliasName}` : "") +
        (info.importModule ? ` import_module "${info.importModule}"` : "") +
        (info.importName ? ` import_name "${info.importName}"` : "") +
        (info.exportName ? ` export_name "${info.exportName}"` : "") +
        (info.isUsed ? " used" : ""),
    );
    return ref;
  }

  findFunction(name: string): FunctionRef {
    return this.#functionsByName.get(name) ?? 0;
  }

  functionHasBody(function_: FunctionRef): boolean {
    return this.#functions.get(function_)?.hasBody ?? false;
  }

  functionParameterTypes(function_: FunctionRef): TypeRef[] {
    return [...(this.#functions.get(function_)?.parameters ?? [])];
  }

  functionResultTypes(function_: FunctionRef): TypeRef[] {
    return [...(this.#functions.get(function_)?.results ?? [])];
  }

  declareGlobal(loc: TokenIndex, info: GlobalInfo): GlobalRef {
    const existing = this.#globalsByName.get(info.name);
    if (existing !== undefined) return existing;
    const ref = this.#nextGlobal++;
    this.#globals.set(ref, { name: info.name, linkage: info.linkage });
    this.#globalsByName.set(info.name, ref);
    this.#emitTop(
      `global @${info.name} : ${this.#typeText(info.type)} ` +
        `${Linkage[info.linkage]}${info.isConstant ? " const" : ""}` +
        `${info.isUsed ? " used" : ""}` +
        ` = ${this.#initializerText(info.initializer)}`,
    );
    return ref;
  }

  findGlobal(name: string): GlobalRef {
    return this.#globalsByName.get(name) ?? 0;
  }

  globalLinkage(global: GlobalRef): Linkage {
    return this.#globals.get(global)?.linkage ?? Linkage.External;
  }

  symbolExists(name: string): boolean {
    return this.#functionsByName.has(name) || this.#globalsByName.has(name);
  }

  beginGlobalInitializer(global: GlobalRef): void {
    this.#emit(`init @${this.#globals.get(global)?.name ?? global}`);
  }

  globalConstructor(loc: TokenIndex, function_: FunctionRef): void {
    this.#emit(`global_ctor @${function_}`);
  }

  defineVTable(loc: TokenIndex, info: VTableInfo): void {
    this.#emitTop(
      `vtable @${info.name} typeinfo @${info.typeInfo} ` +
        `${Linkage[info.linkage]} tables ${info.tables.length}`,
    );
    for (const table of info.tables) {
      const slots = table.slots.map((slot) => (slot ? `@${slot}` : "null"));
      this.#emitTop(
        `  offset_to_top ${table.offsetToTop} slots [${slots.join(", ")}]`,
      );
    }
  }
}
