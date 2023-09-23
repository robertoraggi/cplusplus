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
import { Token } from "./Token";
import { cxx } from "./cxx";
import { ASTSlotKind } from "./ASTSlotKind";
import { ASTSlot } from "./ASTSlot";

interface TranslationUnitLike {
  getUnitHandle(): number;
}

interface StackNode {
  node: AST | Token;
  name?: ASTSlot;
}

class StackEntry {
  #parser: TranslationUnitLike;
  children?: Generator<StackNode, undefined>;

  constructor(
    readonly owner: StackNode,
    parser: TranslationUnitLike,
  ) {
    this.#parser = parser;
  }

  restart() {
    this.children = this.#initChildren();
  }

  *#initChildren() {
    if (!(this.owner.node instanceof AST)) {
      return undefined;
    }

    const handle = this.owner.node.getHandle();
    const slotCount = cxx.getASTSlotCount(handle, 0);

    for (let i = 0; i < slotCount; ++i) {
      const slotKind = cxx.getASTSlotKind(handle, i);
      const name = cxx.getASTSlotName(handle, i);

      if (slotKind === ASTSlotKind.Node) {
        const node = AST.from(cxx.getASTSlot(handle, i), this.#parser);

        if (node) yield { name, node };
      } else if (slotKind === ASTSlotKind.NodeList) {
        for (let it = cxx.getASTSlot(handle, i); it; it = cxx.getListNext(it)) {
          const node = AST.from(cxx.getListValue(it), this.#parser);

          if (node) yield { name, node };
        }
      } else if (slotKind === ASTSlotKind.Token) {
        const token = Token.from(cxx.getASTSlot(handle, i), this.#parser);
        if (token) yield { name, node: token };
      }
    }
  }
}

interface AcceptArgs {
  node: AST | Token;
  depth: number;
  slot?: ASTSlot;
}

/**
 * AST cursor.
 *
 * A cursor is used to traverse the AST.
 */
export class ASTCursor {
  readonly #parser: TranslationUnitLike;
  readonly #stack: StackEntry[] = [];

  /**
   * Constructs a new AST cursor.
   *
   * @param root The root node.
   * @param parser The parser that owns the AST.
   */
  constructor(
    readonly root: AST,
    parser: TranslationUnitLike,
  ) {
    this.#parser = parser;
    this.#stack.push(new StackEntry({ node: root }, this.#parser));
  }

  /**
   * Returns the current AST node or Token.
   */
  get node(): AST | Token {
    return this.#current.owner.node;
  }

  /**
   * Returns the current AST slot.
   */
  get slot(): ASTSlot | undefined {
    return this.#current.owner.name;
  }

  /**
   * Move the cursor to the first child of the current node.
   */
  gotoFirstChild() {
    this.#current.restart();
    if (!this.#current.children) return false;
    const { value: childNode } = this.#current.children.next();
    if (!childNode) return false;
    this.#stack.push(new StackEntry(childNode, this.#parser));
    return true;
  }

  /**
   * Move the cursor to the next sibling of the current node.
   */
  gotoNextSibling() {
    if (!this.#parent) return false;
    if (!this.#parent.children) return false;
    const { value: childNode } = this.#parent.children.next();
    if (!childNode) return false;
    this.#stack.pop();
    this.#stack.push(new StackEntry(childNode, this.#parser));
    return true;
  }

  /**
   * Move the cursor to the parent of the current node.
   */
  gotoParent() {
    if (this.#current.owner.node === this.root) return false;
    this.#stack.pop();
    return true;
  }

  /**
   * Pre-order traversal of the AST.
   *
   * @param accept The function to call for each node.
   */
  preVisit(accept: (args: AcceptArgs) => void | boolean) {
    let done = false;
    let depth = 0;

    while (!done) {
      do {
        if (accept({ node: this.node, slot: this.slot, depth }) === true)
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
