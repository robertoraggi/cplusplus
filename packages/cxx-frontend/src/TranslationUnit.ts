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

import { AST } from "./AST.js";
import {
  cxx,
  type DiagnosticsClientHandle,
  type TranslationUnitHandle,
} from "./cxx.js";
import { disposeSymbol } from "./disposeSymbols.js";
import { Token } from "./Token.js";
import { toReadableStream } from "./toReadableStream.js";

export class TranslationUnit implements Disposable {
  #diagnosticsClient: DiagnosticsClientHandle;
  #handle: TranslationUnitHandle;

  /**
   * Creates a new translation unit.
   */
  constructor() {
    this.#diagnosticsClient = new cxx.DiagnosticsClient();
    this.#handle = new cxx.TranslationUnit(this.#diagnosticsClient);
  }

  /**
   * Disposes the translation unit.
   */
  dispose() {
    this.#handle.delete();
    this.#diagnosticsClient.delete();
  }

  [disposeSymbol](): void {
    this.dispose();
  }

  /**
   * Preprocesses the given source code.
   *
   * @param source the source code
   * @param path the path of the source code
   */
  preprocess(source: string, path: string) {
    this.#handle.setSource(source, path);
  }

  /**
   * Parses the preprocessed code
   *
   * @returns the AST or undefined
   */
  parse(): AST | undefined {
    if (!this.#handle.parse(false)) {
      return undefined;
    }

    return this.ast;
  }

  /**
   * Returns the AST.
   *
   * @returns the AST or undefined
   */
  get ast(): AST | undefined {
    return AST.from(this.#handle.getAST(), this.#handle);
  }

  /**
   * Returns the preprocessed tokens.
   */
  tokens(): Iterable<Token> {
    return {
      [Symbol.iterator]: () => {
        const count = this.tokenCount();
        let index = 1;
        return {
          next: () => {
            if (index < count) {
              const token = this.tokenAt(index++);

              if (token !== undefined) {
                return { value: token, done: false };
              }
            }
            return { value: undefined, done: true };
          },
        };
      },
    };
  }

  /**
   * Returns the preprocessed tokens as a readable stream.
   *
   * The stream must be consumed, or cancelled, before this translation unit is
   * disposed.
   */
  tokenStream(): ReadableStream<Token> {
    return toReadableStream(this.tokens());
  }

  /**
   * Returns the number of tokens.
   *
   * @returns the number of tokens
   */
  tokenCount(): number {
    return this.#handle.tokenCount();
  }

  /**
   * Returns the token at the given index.
   *
   * @param index the index
   * @returns the token or undefined
   */
  tokenAt(index: number): Token | undefined {
    return Token.from(index, this.#handle);
  }
}
