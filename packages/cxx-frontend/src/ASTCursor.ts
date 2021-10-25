// Copyright (c) 2021 Roberto Raggi <roberto.raggi@gmail.com>
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
import { Token } from "./Token";
import { ASTSlotKind, cxx } from "./cxx.js";
import { Parser } from "./Parser.js";

class StackEntry {
  #parser: Parser;
  children?: Generator<AST | Token>;

  constructor(readonly owner: AST | Token, parser: Parser) {
    this.#parser = parser;
  }

  restart() {
    this.children = this.#initChildren();
  }

  *#initChildren() {
    if (this.owner instanceof Token) return;

    const slotCount = cxx.getASTSlotCount(this.owner.getHandle(), 0);

    const handle = this.owner.getHandle();

    for (let i = 0; i < slotCount; ++i) {
      const kind = cxx.getASTSlotKind(handle, i);

      if (kind === ASTSlotKind.Node) {

        const node = AST.from(cxx.getASTSlot(handle, i), this.#parser);

        if (node) yield node

      } else if (kind === ASTSlotKind.NodeList) {

        for (let it = cxx.getASTSlot(handle, i); it; it = cxx.getListNext(it)) {
          const node = AST.from(cxx.getListValue(it), this.#parser);

          if (node) yield node;
        }

      } else if (kind === ASTSlotKind.Token) {

        const token = Token.from(cxx.getASTSlot(handle, i), this.#parser);

        if (token) yield token;
      }
    }
  }
}

export class ASTCursor {
  readonly #parser: Parser;
  readonly #stack: StackEntry[] = [];

  constructor(readonly root: AST, parser: Parser) {
    this.#parser = parser;
    this.#stack.push(new StackEntry(root, this.#parser));
  }

  get node() {
    return this.#current.owner;
  }

  gotoFirstChild() {
    this.#current.restart();
    if (!this.#current.children) return false;
    const { value: childNode } = this.#current.children.next();
    if (!childNode) return false;
    this.#stack.push(new StackEntry(childNode, this.#parser));
    return true;
  }

  gotoNextSibling() {
    if (!this.#parent) return false;
    if (!this.#parent.children) return false;
    const { value: childNode } = this.#parent.children.next();
    if (!childNode) return false;
    this.#stack.pop();
    this.#stack.push(new StackEntry(childNode, this.#parser));
    return true;
  }

  gotoParent() {
    if (this.#current.owner === this.root) return false;
    this.#stack.pop();
    return true;
  }

  preVisit(accept: (node: AST | Token, depth: number) => void | boolean) {
    let done = false;
    let depth = 0;

    while (!done) {
      do {
        if (accept(this.node, depth) === true)
          return;
        ++depth;
      } while (this.gotoFirstChild());

      --depth;

      while (!this.gotoNextSibling()) {
        if (!this.gotoParent()) {
          done = true;
          break;
        }

        --depth;
      }
    }
  }

  get #current() {
    return this.#stack[this.#stack.length - 1];
  }

  get #parent() {
    return this.#stack[this.#stack.length - 2];
  }
}
