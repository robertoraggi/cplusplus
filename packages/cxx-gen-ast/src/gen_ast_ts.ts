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

import { cpy_header } from "./cpy_header.js";
import { AST } from "./parseAST.js";
import { groupNodesByBaseType } from "./groupNodesByBaseType.js";
import * as fs from "fs";

export function gen_ast_ts({ ast, output }: { ast: AST; output: string }) {
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
      emit(`export class ${name} extends ${base} {`);

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
              emit(`    ${getterName(m.name)}(): boolean {`);
              emit(
                `        return cxx.getASTSlot(this.getHandle(), ${slotCount}) !== 0;`
              );
              emit(`    }`);
              ++slotCount;
            } else if (m.type === "Identifier") {
              emit(`    ${getterName(m.name)}(): string | undefined {`);
              emit(
                `      const slot = cxx.getASTSlot(this.getHandle(), ${slotCount});`
              );
              emit(`      return cxx.getIdentifierValue(slot);`);
              emit(`    }`);
              ++slotCount;
            } else if (m.type.endsWith("Literal")) {
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
            emit(`    ${getterName(m.name)}(): ${m.type} | undefined {`);
            emit(
              `        return AST.from<${m.type}>(cxx.getASTSlot(this.getHandle(), ${slotCount}), this.parser);`
            );
            emit(`    }`);
            ++slotCount;
            break;
          case "node-list":
            emit(
              `    *${getterName(m.name)}(): Generator<${m.type} | undefined> {`
            );
            emit(
              `        for (let it = cxx.getASTSlot(this.getHandle(), ${slotCount}); it; it = cxx.getListNext(it)) {`
            );
            emit(
              `            yield AST.from<${m.type}>(cxx.getListValue(it), this.parser);`
            );
            emit(`        }`);
            emit(`    }`);
            ++slotCount;
            break;
          case "token":
            emit(`    ${tokenGetterName(m.name)}(): Token | undefined {`);
            emit(
              `        return Token.from(cxx.getASTSlot(this.getHandle(), ${slotCount}), this.parser);`
            );
            emit(`    }`);
            ++slotCount;
            break;
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
import { cxx } from "./cxx.js";
import { SourceLocation } from "./SourceLocation.js";
import { ASTCursor } from "./ASTCursor.js";
import { ASTVisitor } from "./ASTVisitor.js";
import { ASTKind } from "./ASTKind.js";
import { Token } from "./Token.js";

interface TranslationUnitLike {
    getUnitHandle(): number;
}

export abstract class AST {
    constructor(private readonly handle: number,
        private readonly kind: ASTKind,
        protected readonly parser: TranslationUnitLike) {
    }

    walk(): ASTCursor {
        return new ASTCursor(this, this.parser);
    }

    getKind(): ASTKind {
        return this.kind;
    }

    is(kind: ASTKind): boolean {
        return this.kind === kind;
    }

    isNot(kind: ASTKind): boolean {
        return this.kind !== kind;
    }

    getHandle() {
        return this.handle;
    }

    getStartLocation(): SourceLocation {
        return cxx.getStartLocation(this.handle, this.parser.getUnitHandle());
    }

    getEndLocation(): SourceLocation {
        return cxx.getEndLocation(this.handle, this.parser.getUnitHandle());
    }

    abstract accept<Context, Result>(visitor: ASTVisitor<Context, Result>, context: Context): Result;

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

  fs.writeFileSync(output, out);
}
