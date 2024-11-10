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

import { cxx } from "./cxx";

interface PreprocessorOptions {
  systemIncludePaths?: string[];
}

export class Preprocessor {
  #control: typeof cxx.Control;
  #diagnosticClient: typeof cxx.DiagnosticsClient;
  #handle: typeof cxx.Preprocessor;

  /**
   * Creates an instance of Preprocessor.
   *
   * @param source
   */
  constructor({ systemIncludePaths }: PreprocessorOptions = {}) {
    this.#control = new cxx.Control();
    this.#diagnosticClient = new cxx.DiagnosticsClient();
    this.#handle = new cxx.Preprocessor(this.#control, this.#diagnosticClient);
    this.#diagnosticClient.setPreprocessor(this.#handle);

    systemIncludePaths?.forEach((path) => {
      this.#handle.addIncludePath(path);
    });
  }

  /**
   * Dispose the preprocessor.
   */
  dispose() {
    this.#handle.delete();
    this.#diagnosticClient.delete();
    this.#control.delete();
  }

  /**
   * Preprocess the given source code.
   *
   * @param source the source code to preprocess
   * @param fileName the file name of the source code
   * @returns the preprocessed source code
   */

  preprocess(source: string, fileName: string): string {
    return this.#handle.preprocess(source, fileName);
  }

  /**
   * Returns true if the preprocessor can resolve files.
   */
  get canResolveFiles(): boolean {
    return this.#handle.canResolveFiles();
  }

  /**
   * Sets whether the preprocessor can resolve files.
   */
  set canResolveFiles(value: boolean) {
    this.#handle.setCanResolveFiles(value);
  }

  /**
   * Returns the current path.
   */
  get currentPath(): string {
    return this.#handle.currentPath();
  }

  /**
   * Sets the current path.
   *
   * @param path the current path
   */
  set currentPath(path: string) {
    this.#handle.setCurrentPath(path);
  }

  /**
   * Defines a macro.
   *
   * @param name the name of the macro
   * @param value the value of the macro
   */
  defineMacro(name: string, value: string): void {
    this.#handle.defineMacro(name, value);
  }

  /**
   * Undefines a macro.
   *
   * @param name the name of the macro
   */
  undefineMacro(name: string): void {
    this.#handle.undefineMacro(name);
  }
}
