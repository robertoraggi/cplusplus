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
import { type UnitOptions } from "./cxx-js.js";
import { isCxxLoaded } from "./loadCxx.js";
import { type Diagnostic } from "./Diagnostic.js";
import { type Unit } from "./Unit.js";
import { AST } from "./AST.js";
import { asyncDisposeSymbol, disposeSymbol } from "./disposeSymbols.js";

export const OutputCodeFormat = ["cxxir", "mlir", "llvm", "asm"] as const;
export type OutputCodeFormat = (typeof OutputCodeFormat)[number];

export interface ParseOptions extends UnitOptions {
  /**
   * Path to the file to parse.
   */
  path: string;

  /**
   * Source code to parse.
   */
  source: string;
}

/**
 * A parsed translation unit.
 *
 * Instances are created by {@link Parser.parse}, so a `Parser` is always fully
 * parsed and every accessor is synchronous.
 *
 * The AST and the tokens are owned by the parser, they must not be used after
 * the parser has been disposed.
 */
export class Parser implements Disposable, AsyncDisposable {
  #unit: Unit | undefined;
  readonly #ast: AST;

  private constructor(unit: Unit) {
    this.#unit = unit;

    const ast = AST.from(unit.getHandle(), this);

    if (!ast) {
      throw new Error("failed to create the AST");
    }

    this.#ast = ast;
  }

  /**
   * Parses the given source code.
   *
   * @param options the source code to parse and the include resolvers.
   * @returns the parsed translation unit.
   */
  static async parse(options: ParseOptions): Promise<Parser> {
    const { path, source, ...unitOptions } = options;

    if (typeof path !== "string") {
      throw new TypeError("expected parameter 'path' of type 'string'");
    }

    if (typeof source !== "string") {
      throw new TypeError("expected parameter 'source' of type 'string'");
    }

    if (!isCxxLoaded()) {
      throw new Error(
        "the cxx wasm module is not loaded, call loadCxx() first",
      );
    }

    const unit = cxx.createUnit(source, path, unitOptions);

    if (!unit) {
      throw new Error("failed to create the translation unit");
    }

    try {
      await unit.parse();
      return new Parser(unit);
    } catch (error) {
      unit.delete();
      throw error;
    }
  }

  /**
   * Returns the root of the AST.
   */
  get ast(): AST {
    if (!this.#unit) {
      throw disposedError();
    }

    return this.#ast;
  }

  /**
   * Returns the diagnostics collected while preprocessing and parsing.
   */
  get diagnostics(): Diagnostic[] {
    return this.#nativeUnit().getDiagnostics();
  }

  /**
   * Generates code in the given format.
   *
   * @param options the output format.
   * @returns the generated code.
   */
  emitCode({ format }: { format: OutputCodeFormat }): string {
    return this.#nativeUnit().emitCode(format);
  }

  /**
   * Returns the handle of the translation unit.
   */
  getUnitHandle(): number {
    return this.#nativeUnit().getUnitHandle();
  }

  /**
   * Releases the native resources owned by the parser.
   *
   * Calling `dispose` more than once is allowed, the AST and the tokens of a
   * disposed parser must not be used.
   */
  dispose(): void {
    this.#unit?.delete();
    this.#unit = undefined;
  }

  [disposeSymbol](): void {
    this.dispose();
  }

  async [asyncDisposeSymbol](): Promise<void> {
    this.dispose();
  }

  #nativeUnit(): Unit {
    if (!this.#unit) {
      throw disposedError();
    }

    return this.#unit;
  }
}

function disposedError(): Error {
  return new Error("Parser has been disposed");
}
