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

import initCxx, { cxx } from "./cxx";
import { Unit } from "./Unit";
import { AST } from "./AST";

interface ParserParams {
  /**
   * Path to the file to parse.
   */
  path: string;

  /**
   * Source code to parse.
   */
  source: string;

  /**
   * Function to resolve include directives.
   */
  resolve?: (
    name: string,
    kind: "quoted" | "angled",
    next: boolean,
  ) => Promise<string | undefined>;

  /**
   * Function to read files.
   */
  readFile?: (path: string) => Promise<string | undefined>;
}

export class Parser {
  #unit: Unit | undefined;
  #ast: AST | undefined;
  #pendingAST: Promise<AST> | undefined;

  static async init({
    wasm,
  }: {
    wasm: Uint8Array | ArrayBuffer | WebAssembly.Module;
  }) {
    return await initCxx({ wasm });
  }

  static async initFromURL(
    url: URL,
    { signal }: { signal?: AbortSignal } = {},
  ) {
    const response = await fetch(url, { signal });

    if (!response.ok) {
      throw new Error(`failed to fetch '${url}'`);
    }

    if (signal?.aborted) {
      throw new Error(`fetch '${url}' aborted`);
    }

    const wasm = await response.arrayBuffer();

    return await Parser.init({ wasm });
  }

  static isInitialized(): boolean {
    return cxx !== undefined && cxx !== null;
  }

  constructor(options: ParserParams) {
    const { path, source, resolve, readFile } = options;

    if (typeof path !== "string") {
      throw new TypeError("expected parameter 'path' of type 'string'");
    }

    if (typeof source !== "string") {
      throw new TypeError("expected parameter 'source' of type 'string'");
    }

    this.#unit = cxx.createUnit(source, path, { resolve, readFile });
  }

  async parse(): Promise<AST> {
    return await this.getASTAsync();
  }

  async emitIR(): Promise<string> {
    const _ = await this.getASTAsync();
    return this.#unit?.emitIR() ?? "";
  }

  async #parseHelper(): Promise<AST> {
    if (this.#pendingAST) {
      return await this.#pendingAST;
    }

    if (!this.#unit) {
      throw new Error("Parser has been disposed");
    }

    await this.#unit.parse();

    const ast = AST.from(this.#unit.getHandle(), this);

    if (!ast) {
      throw new Error("Failed to create AST");
    }

    this.#ast = ast;

    return ast;
  }

  dispose() {
    this.#unit?.delete();
    this.#unit = undefined;
    this.#ast = undefined;
    this.#pendingAST = undefined;
  }

  getUnitHandle(): number {
    return this.#unit?.getUnitHandle() ?? 0;
  }

  async getASTAsync(): Promise<AST> {
    this.#pendingAST ??= this.#parseHelper();
    return await this.#pendingAST;
  }

  // internal
  getAST(): AST | undefined {
    return this.#ast;
  }

  getDiagnostics() {
    return this.#unit?.getDiagnostics() ?? [];
  }
}
