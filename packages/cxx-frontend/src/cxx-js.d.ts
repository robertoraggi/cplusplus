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

import { SourceLocation } from "./SourceLocation";
import { TokenKind } from "./TokenKind";
import { ASTSlot } from "./ASTSlot";
import { Unit } from "./Unit";
import { ASTSlotKind } from "./ASTSlotKind";

interface Control {
  new (): Control;
  delete(): void;
}

interface DiagnosticsClient {
  new (): DiagnosticsClient;
  delete(): void;
}

interface TranslationUnit {
  new (control: Control, diagnosticsClient: DiagnosticsClient): TranslationUnit;
  delete(): void;

  getUnitHandle(): number;
  setSource(source: string, path: string): void;
  parse(checkTypes: boolean): boolean;
  tokenCount(): number;
  tokenAt(index: number): number;
  getAST(): number;
}

interface Preprocessor {
  new (control: Control, diagnosticsClient: DiagnosticsClient): Preprocessor;
  delete(): void;

  canResolveFiles(): boolean;
  setCanResolveFiles(value: boolean): void;
  currentPath(): string;
  setCurrentPath(path: string): void;
  defineMacro(name: string, value: string): void;
  undefineMacro(name: string): void;
  addIncludePath(path: string): void;

  preprocess(source: string, fileName: string): string;
}

interface Lexer {
  preprocessing: boolean;
  keepComments: boolean;

  new (source: string): Lexer;
  delete(): void;

  next(): number;
  tokenKind(): number;
  tokenAtStartOfLine(): boolean;
  tokenHasLeadingSpace(): boolean;
  tokenOffset(): number;
  tokenLength(): number;
  tokenText(): string;
}

export interface CXX {
  Control: Control;
  DiagnosticsClient: DiagnosticsClient;
  Preprocessor: Preprocessor;
  Lexer: Lexer;
  TranslationUnit: TranslationUnit;

  createUnit(source: string, path: string): Unit;
  getASTKind(handle: number): number;
  getASTSlot(handle: number, slot: number): number;
  getASTSlotKind(handle: number, slot: number): ASTSlotKind;
  getASTSlotName(handle: number, slot: number): ASTSlot;
  getASTSlotCount(handle: number, slot: number): number;
  getListValue(handle: number): number;
  getListNext(handle: number): number;
  getTokenText(handle: number, unitHandle: number): string;
  getTokenKind(handle: number, unitHandle: number): TokenKind;
  getTokenLocation(handle: number, unitHandle: number): SourceLocation;
  getStartLocation(handle: number, unitHandle: number): SourceLocation;
  getEndLocation(handle: number, unitHandle: number): SourceLocation;
  getLiteralValue(handle: number): string | undefined;
  getIdentifierValue(handle: number): string | undefined;
}

export default function ({
  wasm,
}: {
  wasm: Uint8Array | ArrayBuffer | WebAssembly.Module;
}): Promise<CXX>;
