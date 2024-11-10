// Copyright (c) 2024 Roberto Raggi <roberto.raggi@gmail.com>
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

  resolve?: (
    name: string,
    kind: "quoted" | "angled",
    next: boolean,
  ) => Promise<string | undefined>;
}

export class Parser {
  #unit: Unit | undefined;
  #ast: AST | undefined;

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
    const { path, source, resolve } = options;

    if (typeof path !== "string") {
      throw new TypeError("expected parameter 'path' of type 'string'");
    }

    if (typeof source !== "string") {
      throw new TypeError("expected parameter 'source' of type 'string'");
    }

    this.#unit = cxx.createUnit(source, path, { resolve });
  }

  async parse() {
    if (!this.#unit) {
      return;
    }
    await this.#unit.parse();
    this.#ast = AST.from(this.#unit.getHandle(), this);
  }

  dispose() {
    this.#unit?.delete();
    this.#unit = undefined;
    this.#ast = undefined;
  }

  getUnitHandle(): number {
    return this.#unit?.getUnitHandle() ?? 0;
  }

  getAST(): AST | undefined {
    return this.#ast;
  }

  getDiagnostics() {
    return this.#unit?.getDiagnostics() ?? [];
  }
}
