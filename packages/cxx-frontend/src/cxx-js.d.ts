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
  exists?: ((path: string) => boolean) | undefined;
  readFile?: ((path: string) => Promise<string | undefined>) | undefined;
};

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
  getHandle(): number;
  getUnitHandle(): number;
  parse(): any;
  emitCode(_0: EmbindString): any;
}

interface EmbindModule {
  Unit: {};
  getASTKind(_0: number): number;
  getListValue(_0: number): number;
  getListNext(_0: number): number;
  getASTSlot(_0: number, _1: number): number;
  getASTSlotKind(_0: number, _1: number): number;
  getASTSlotName(_0: number, _1: number): number;
  getASTSlotCount(_0: number, _1: number): number;
  getTokenKind(_0: number, _1: number): number;
  createUnit(_0: EmbindString, _1: EmbindString, _2: UnitOptions): Unit | null;
  getTokenText(_0: number, _1: number): string;
  getTokenLocation(_0: number, _1: number): any;
  getStartLocation(_0: number, _1: number): any;
  getEndLocation(_0: number, _1: number): any;
  getIdentifierValue(_0: number): any;
  getLiteralValue(_0: number): any;
}

export type MainModule = WasmModule & EmbindModule;
export default function MainModuleFactory(
  options?: unknown,
): Promise<MainModule>;
