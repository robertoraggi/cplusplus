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

import { cpy_header } from "./cpy_header.js";
import { AST } from "./parseAST.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import { format } from "prettier";
import * as fs from "fs";

export async function gen_ast_ts({
  ast,
  output,
}: {
  ast: AST;
  output: string;
}) {
  const code: string[] = [];
  const emit = (line = "") => code.push(line);

  const by_bases = groupNodesByBaseType(ast);

  const baseClassNames = Array.from(by_bases.keys()).filter((b) => b !== "AST");
  baseClassNames.sort();

  for (const base of baseClassNames) {
    emit(`export abstract class ${base} extends AST { }`);
  }

  emit();

  const getterName = (name: string) =>
    `get${name[0].toUpperCase() + name.substr(1)}`;
  const nodeName = (name: string) => name.slice(0, -3);
  const tokenGetterName = (name: string) =>
    `get${name[0].toUpperCase() + name.slice(1, -3)}Token`;

  by_bases.forEach((nodes) => {
    nodes.forEach(({ name, base, members }) => {
      emit(`/**`);
      emit(` * ${name} node.`);
      emit(` */`);
      emit(`export class ${name} extends ${base} {`);

      emit(`    /**`);
      emit(`     * Traverse this node using the given visitor.`);
      emit(`     * @param visitor the visitor.`);
      emit(`     * @param context the context.`);
      emit(`     * @returns the result of the visit.`);
      emit(`     */`);
      emit(
        `    accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result {`
      );
      emit(`        return visitor.visit${nodeName(name)}(this, context);`);
      emit(`    }`);

      let slotCount = 0;
      members.forEach((m) => {
        switch (m.kind) {
          case "attribute":
            if (m.type === "bool") {
              emit();
              emit(`    /**`);
              emit(`     * Returns the ${m.name} attribute of this node`);
              emit(`     */`);
              emit(`    ${getterName(m.name)}(): boolean {`);
              emit(
                `        return cxx.getASTSlot(this.getHandle(), ${slotCount}) !== 0;`
              );
              emit(`    }`);
              ++slotCount;
            } else if (m.type === "TokenKind") {
              emit();
              emit(`    /**`);
              emit(`     * Returns the ${m.name} attribute of this node`);
              emit(`     */`);
              emit(`    ${getterName(m.name)}(): TokenKind {`);
              emit(
                `        return cxx.getASTSlot(this.getHandle(), ${slotCount});`
              );
              emit(`    }`);
              ++slotCount;
            } else if (m.type === "Identifier") {
              emit();
              emit(`    /**`);
              emit(`     * Returns the ${m.name} attribute of this node`);
              emit(`     */`);
              emit(`    ${getterName(m.name)}(): string | undefined {`);
              emit(
                `      const slot = cxx.getASTSlot(this.getHandle(), ${slotCount});`
              );
              emit(`      return cxx.getIdentifierValue(slot);`);
              emit(`    }`);
              ++slotCount;
            } else if (m.type.endsWith("Literal")) {
              emit();
              emit(`    /**`);
              emit(`     * Returns the ${m.name} attribute of this node`);
              emit(`     */`);
              emit(`    ${getterName(m.name)}(): string | undefined {`);
              emit(
                `      const slot = cxx.getASTSlot(this.getHandle(), ${slotCount});`
              );
              emit(`      return cxx.getLiteralValue(slot);`);
              emit(`    }`);
              ++slotCount;
            }
            break;
          case "node":
            emit();
            emit(`    /**`);
            emit(`     * Returns the ${m.name} of this node`);
            emit(`     */`);
            emit(`    ${getterName(m.name)}(): ${m.type} | undefined {`);
            emit(
              `        return AST.from<${m.type}>(cxx.getASTSlot(this.getHandle(), ${slotCount}), this.parser);`
            );
            emit(`    }`);
            ++slotCount;
            break;
          case "node-list":
            emit();
            emit(`    /**`);
            emit(`     * Returns the ${m.name} of this node`);
            emit(`     */`);
            emit(
              `    ${getterName(m.name)}(): Iterable<${m.type} | undefined> {`
            );
            emit(`let it = cxx.getASTSlot(this.getHandle(), 0);`);
            emit(`let value: ${m.type} | undefined;`);
            emit(`let done = false;`);
            emit(`const p = this.parser;`);
            emit(`function advance() {`);
            emit(`  done = it === 0;`);
            emit(`  if (done) return;`);
            emit(`  const ast = cxx.getListValue(it);`);
            emit(`  value = AST.from<${m.type}>(ast, p);`);
            emit(`  it = cxx.getListNext(it);`);
            emit(`};`);
            emit(`function next() {`);
            emit(`  advance();`);
            emit(`  return { done, value };`);
            emit(`};`);
            emit(`return {`);
            emit(`  [Symbol.iterator]() {`);
            emit(`    return { next };`);
            emit(`  },`);
            emit(`};`);
            emit(`    }`);
            ++slotCount;
            break;
          case "token": {
            const tokenName = m.name.slice(0, -3);
            emit();
            emit(`    /**`);
            emit(
              `     * Returns the location of the ${tokenName} token in this node`
            );
            emit(`     */`);
            emit(`    ${tokenGetterName(m.name)}(): Token | undefined {`);
            emit(
              `        return Token.from(cxx.getASTSlot(this.getHandle(), ${slotCount}), this.parser);`
            );
            emit(`    }`);
            ++slotCount;
            break;
          }
          case "token-list":
            // emit(`  List<SourceLocation>* ${m.name} = nullptr;`);
            ++slotCount;
            break;
        }
      });

      emit(`}`);
      emit();
    });
  });

  emit(
    `const AST_CONSTRUCTORS: Array<new (handle: number, kind: ASTKind, parser: TranslationUnitLike) => AST> = [`
  );
  by_bases.forEach((nodes) => {
    nodes.forEach(({ name }) => {
      emit(`    ${name},`);
    });
  });
  emit(`];`);

  const out = `${cpy_header}
import { cxx } from "./cxx";
import { SourceLocation } from "./SourceLocation";
import { ASTCursor } from "./ASTCursor";
import { ASTVisitor } from "./ASTVisitor";
import { ASTKind } from "./ASTKind";
import { Token } from "./Token";
import { TokenKind } from "./TokenKind";

/**
 * An interface that represents a translation unit.
 */
interface TranslationUnitLike {
    /**
     * Returns the handle of the translation unit.
     */
    getUnitHandle(): number;
}

/**
 * The base class of all the AST nodes.
 */
export abstract class AST {
    /**
     * Constructs an AST node.
     * @param handle the handle of the AST node.
     * @param kind the kind of the AST node.
     * @param parser the parser that owns the AST node.
     */
    constructor(private readonly handle: number,
        private readonly kind: ASTKind,
        protected readonly parser: TranslationUnitLike) {
    }

    /**
     * Returns the cursor of the AST node.
     *
     * The cursor is used to traverse the AST.
     *
     * @returns the cursor of the AST node.
     */
    walk(): ASTCursor {
        return new ASTCursor(this, this.parser);
    }

    /**
     * Returns the kind of the AST node.
     *
     * @returns the kind of the AST node.
     */
    getKind(): ASTKind {
        return this.kind;
    }

    /**
     * Returns true if the AST node is of the given kind.
     *
     * @param kind the kind to check.
     * @returns true if the AST node is of the given kind.
     */
    is(kind: ASTKind): boolean {
        return this.kind === kind;
    }

    /**
     * Returns true if the AST node is not of the given kind.
     *
     * @param kind the kind to check.
     * @returns true if the AST node is not of the given kind.
     */
    isNot(kind: ASTKind): boolean {
        return this.kind !== kind;
    }

    /**
     * Returns the handle of the AST node.
     *
     * @returns the handle of the AST node.
     */
    getHandle() {
        return this.handle;
    }

    /**
     * Returns the source location of the AST node.
     *
     * @returns the source location of the AST node.
     */
    getStartLocation(): SourceLocation | undefined {
        return cxx.getStartLocation(this.handle, this.parser.getUnitHandle());
    }

    /**
     * Returns the source location of the AST node.
     *
     * @returns the source location of the AST node.
     */
    getEndLocation(): SourceLocation | undefined {
        return cxx.getEndLocation(this.handle, this.parser.getUnitHandle());
    }

    /**
     * Accepts the given visitor.
     */
    abstract accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result;

    /**
     * Constructs an AST node from the given handle.
     *
     * @param handle the handle of the AST node.
     * @param parser the parser that owns the AST node.
     * @returns the AST node.
     */
    static from<T extends AST = AST>(handle: number, parser: TranslationUnitLike): T | undefined {
        if (handle) {
            const kind = cxx.getASTKind(handle) as ASTKind;
            const ast = new AST_CONSTRUCTORS[kind](handle, kind, parser) as T;
            return ast;
        }
        return;
    }
}

${code.join("\n")}
`;

  fs.writeFileSync(output, await format(out, { parser: "typescript" }));
}
