// TypeScript bindings for emscripten-generated code.  Automatically generated at compile time.
interface WasmModule {}

type EmbindString =
  ArrayBuffer | Uint8Array | Uint8ClampedArray | Int8Array | string;
export interface ClassHandle {
  isAliasOf(other: ClassHandle): boolean;
  delete(): void;
  deleteLater(): this;
  isDeleted(): boolean;
  // @ts-ignore - If targeting lower than ESNext, this symbol might not exist.
  [Symbol.dispose](): void;
  clone(): this;
}
export type UnitOptions = {
  appdir?: string | undefined;
  sysroot?: string | undefined;
  std?: "c++14" | "c++17" | "c++20" | "c++23" | "c++26" | undefined;
  defines?: string[] | undefined;
  undefines?: string[] | undefined;
  quoteIncludePaths?: string[] | undefined;
  includePaths?: string[] | undefined;
  systemIncludePaths?: string[] | undefined;
  debugInfo?: boolean | undefined;
  optimizationLevel?: number | undefined;
  exists?: ((path: string) => boolean) | undefined;
  readFile?: ((path: string) => Promise<string | undefined>) | undefined;
  shouldContinue?: (() => Promise<boolean>) | undefined;
};

export type EmitterDelegate = import("./Emitter.js").EmitterDelegate;

export type DiagnosticList = Array<{
  fileName: string;
  startLine: number;
  startColumn: number;
  endLine: number;
  endColumn: number;
  message: string;
  severity: "message" | "note" | "warning" | "error" | "fatal";
  notes: Array<{
    fileName: string;
    startLine: number;
    startColumn: number;
    endLine: number;
    endColumn: number;
    message: string;
  }>;
}>;

export interface Unit extends ClassHandle {
  getDiagnostics(): DiagnosticList;
  emitWith(_0: EmitterDelegate): void;
  getUnitHandle(): number;
  parse(): any;
  emitCode(_0: EmbindString): any;
}

export type LanguageServerOptions = {
  preamble?: boolean | undefined;
  appdir?: string | undefined;
  sysroot?: string | undefined;
  std?: "c++14" | "c++17" | "c++20" | "c++23" | "c++26" | undefined;
  defines?: string[] | undefined;
  undefines?: string[] | undefined;
  quoteIncludePaths?: string[] | undefined;
  includePaths?: string[] | undefined;
  systemIncludePaths?: string[] | undefined;
  exists?: ((path: string) => boolean) | undefined;
  readFile?: ((path: string) => Promise<string | undefined>) | undefined;
  shouldContinue?: (() => Promise<boolean>) | undefined;
  onTrace?:
    ((message: string, verbose: string | undefined) => void) | undefined;
  onMessage: (message: string) => void;
};

export interface LanguageServer extends ClassHandle {
  receive(_0: EmbindString): any;
}

interface EmbindModule {
  Unit: {};
  LanguageServer: {};
  createLanguageServer(_0: LanguageServerOptions): LanguageServer | null;
  getTokenKind(_0: number, _1: number): number;
  readSymbolSize(_0: number, _1: number): number;
  readTypeSize(_0: number, _1: number): number;
  readNameSize(_0: number, _1: number): number;
  readMiscSize(_0: number, _1: number): number;
  getASTKind(_0: number): number;
  getSymbolKind(_0: number): number;
  getTypeKind(_0: number): number;
  getNameKind(_0: number): number;
  getListValue(_0: number): number;
  getListNext(_0: number): number;
  getUnitAST(_0: number): number;
  getGlobalScope(_0: number): number;
  readASTBigInt(_0: number, _1: number): bigint;
  readLiteralBigInt(_0: number, _1: number): bigint;
  readMiscBigInt(_0: number, _1: number): bigint;
  readAST(_0: number, _1: number): number;
  readSymbol(_0: number, _1: number): number;
  readSymbolItem(_0: number, _1: number, _2: number): number;
  readType(_0: number, _1: number): number;
  readTypeItem(_0: number, _1: number, _2: number): number;
  readName(_0: number, _1: number): number;
  readLiteral(_0: number, _1: number): number;
  readMisc(_0: number, _1: number): number;
  createUnit(_0: EmbindString, _1: EmbindString, _2: UnitOptions): Unit | null;
  getTokenText(_0: number, _1: number): string;
  readSymbolString(_0: number, _1: number): string;
  readSymbolItemString(_0: number, _1: number, _2: number): string;
  readTypeString(_0: number, _1: number): string;
  readNameString(_0: number, _1: number): string;
  readLiteralString(_0: number, _1: number): string;
  readMiscString(_0: number, _1: number): string;
  getTokenLocation(_0: number, _1: number): any;
  getStartLocation(_0: number, _1: number): any;
  getEndLocation(_0: number, _1: number): any;
  readASTVal(_0: number, _1: number): any;
  readSymbolVal(_0: number, _1: number): any;
  readSymbolItemVal(_0: number, _1: number, _2: number): any;
  readTypeVal(_0: number, _1: number): any;
  readNameItemVal(_0: number, _1: number, _2: number): any;
  readLiteralVal(_0: number, _1: number): any;
  readMiscVal(_0: number, _1: number): any;
  readMiscItemVal(_0: number, _1: number, _2: number): any;
}

export type MainModule = WasmModule & EmbindModule;
export default function MainModuleFactory(
  options?: unknown,
): Promise<MainModule>;
