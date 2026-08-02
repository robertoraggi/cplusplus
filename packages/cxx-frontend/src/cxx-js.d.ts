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
export interface Control extends ClassHandle {}

export interface DiagnosticsClient extends ClassHandle {
  setPreprocessor(_0: Preprocessor | null): void;
}

export interface Preprocessor extends ClassHandle {
  canResolveFiles(): boolean;
  setCanResolveFiles(_0: boolean): void;
  preprocess(_0: EmbindString, _1: EmbindString): string;
  addIncludePath(_0: EmbindString): void;
  defineMacro(_0: EmbindString, _1: EmbindString): void;
  undefineMacro(_0: EmbindString): void;
  currentPath(): string;
  setCurrentPath(_0: EmbindString): void;
}

export interface Lexer extends ClassHandle {
  preprocessing: boolean;
  keepComments: boolean;
  tokenAtStartOfLine(): boolean;
  tokenHasLeadingSpace(): boolean;
  tokenKind(): number;
  tokenOffset(): number;
  next(): number;
  tokenLength(): number;
  tokenText(): string;
}

export interface TranslationUnit extends ClassHandle {
  parse(_0: boolean): boolean;
  tokenCount(): number;
  getAST(): number;
  getUnitHandle(): number;
  setSource(_0: EmbindString, _1: EmbindString): void;
}

export interface Unit extends ClassHandle {
  getHandle(): number;
  getUnitHandle(): number;
  emitCode(_0: EmbindString): string;
  parse(): any;
  getDiagnostics(): any;
}

interface EmbindModule {
  Control: {
    new (): Control;
  };
  DiagnosticsClient: {
    new (): DiagnosticsClient;
  };
  Preprocessor: {
    new (_0: Control | null, _1: DiagnosticsClient | null): Preprocessor;
  };
  Lexer: {
    new (_0: EmbindString): Lexer;
  };
  TranslationUnit: {
    new (_0: DiagnosticsClient | null): TranslationUnit;
  };
  Unit: {};
  getASTKind(_0: number): number;
  getListValue(_0: number): number;
  getListNext(_0: number): number;
  getASTSlot(_0: number, _1: number): number;
  getASTSlotKind(_0: number, _1: number): number;
  getASTSlotName(_0: number, _1: number): number;
  getASTSlotCount(_0: number, _1: number): number;
  getTokenKind(_0: number, _1: number): number;
  getTokenText(_0: number, _1: number): string;
  createUnit(_0: EmbindString, _1: EmbindString, _2: any): Unit | null;
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
