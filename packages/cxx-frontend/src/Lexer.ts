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

import { TokenKind } from "./TokenKind.js";
import { cxx } from "./cxx.js";

export class Lexer {

    #handle: typeof cxx.Lexer;

    /**
     * Creates a new lexer.
     *
     * @param source The source code to be tokenized.
     */
    constructor(source: string) {
        this.#handle = new cxx.Lexer(source);
    }

    /**
     * Disposes the lexer.
     */
    dispose() {
        this.#handle.delete();
    }

    /**
     * Returns the next token.
     * @returns The next token.
     */
    next(): TokenKind { return this.#handle.next(); }

    /**
     * Returns whether the lexer is configured to preprocess the input.
     */
    get preprocessing(): boolean { return this.#handle.preprocessing; }

    /**
     * Sets whether the lexer should preprocess the input.
     */
    set preprocessing(value: boolean) { this.#handle.preprocessing = value; }

    /**
     * Returns whether the lexer should keep comments.
     */
    get keepComments(): boolean { return this.#handle.keepComments; }

    /**
     * Sets whether the lexer should keep comments.
     */
    set keepComments(value: boolean) { this.#handle.keepComments = value; }

    /**
     * Returns the current token kind.
     */
    get tokenKind(): TokenKind { return this.#handle.tokenKind(); }

    /**
     * Returns whether the current token is at the start of a line.
     */
    get tokenAtStartOfLine(): boolean { return this.#handle.tokenAtStartOfLine(); }

    /**
     * Returns whether the current token has a leading space.
     */
    get tokenHasLeadingSpace(): boolean { return this.#handle.tokenHasLeadingSpace(); }

    /**
     * Returns the offset of the current token.
     */
    get tokenOffset(): number { return this.#handle.tokenOffset(); }

    /**
     * Returns the length of the current token.
     */
    get tokenLength(): number { return this.#handle.tokenLength(); }

    /**
     * Returns the text of the current token.
     */
    get tokenText(): string { return this.#handle.tokenText(); }
}
