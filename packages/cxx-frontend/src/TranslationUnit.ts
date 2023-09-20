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

import { AST } from "./AST";
import { cxx } from "./cxx";
import { Token } from "./Token";

export class TranslationUnit {
  #control: typeof cxx.Control;
  #diagnosticsClient: typeof cxx.DiagnosticsClient;
  #handle: typeof cxx.TranslationUnit;

  /**
   * Creates a new translation unit.
   */
  constructor() {
    this.#control = new cxx.Control();
    this.#diagnosticsClient = new cxx.DiagnosticsClient();
    this.#handle = new cxx.TranslationUnit(
      this.#control,
      this.#diagnosticsClient,
    );
  }

  /**
   * Disposes the translation unit.
   */
  dispose() {
    this.#handle.delete();
    this.#diagnosticsClient.delete();
    this.#control.delete();
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

    return this.getAST();
  }

  /**
   * Returns the AST.
   *
   * @returns the AST or undefined
   */
  getAST(): AST | undefined {
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
