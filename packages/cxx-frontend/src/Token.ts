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

import { cxx } from "./cxx";
import { SourceLocation } from "./SourceLocation";
import { TokenKind } from "./TokenKind";

interface TranslationUnitLike {
  getUnitHandle(): number;
}

export class Token {
  #handle: number;
  #unit: number;

  constructor(handle: number, parser: TranslationUnitLike) {
    this.#handle = handle;
    this.#unit = parser.getUnitHandle();
  }

  getHandle() {
    return this.#handle;
  }

  getKind(): TokenKind {
    return cxx.getTokenKind(this.#handle, this.#unit);
  }

  is(kind: TokenKind) {
    return this.getKind() === kind;
  }

  isNot(kind: TokenKind) {
    return this.getKind() !== kind;
  }

  getText(): string {
    return cxx.getTokenText(this.#handle, this.#unit);
  }

  getLocation(): SourceLocation {
    return cxx.getTokenLocation(this.#handle, this.#unit);
  }

  static from(handle: number, parser: TranslationUnitLike): Token | undefined {
    return handle ? new Token(handle, parser) : undefined;
  }
}
