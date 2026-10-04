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

import * as fs from "node:fs";
import * as path from "node:path";
import { missingFrontend, modelSnapshotPath } from "./modelRefresh.ts";

const root = path.resolve(import.meta.dirname, "../../..");

interface AstClass {
  name: string;
  childFields: Map<string, string>;
}

const derivedAfterCopy = new Set([
  "ForRangeStatementAST.beginInitializer",
  "ForRangeStatementAST.endInitializer",
  "ForRangeStatementAST.condition",
  "ForRangeStatementAST.increment",
  "ForRangeStatementAST.element",
  "StructuredBindingDeclarationAST.hiddenVariable",
  "StructuredBindingDeclarationAST.bindingDeclaratorList",
]);

function isChildNodeType(typeName: string): boolean {
  return /AST\*/.test(typeName);
}

function astClasses(): Map<string, AstClass> {
  const model = JSON.parse(fs.readFileSync(modelSnapshotPath(root), "utf8"));
  const result = new Map<string, AstClass>();
  for (const cls of model.classes) {
    const name: string = cls.unqualifiedName;
    if (!name.endsWith("AST")) continue;
    const childFields = new Map<string, string>();
    for (const field of cls.fields ?? []) {
      if (isChildNodeType(field.typeName ?? ""))
        childFields.set(field.name, field.typeName);
    }
    result.set(name, { name, childFields });
  }
  return result;
}

async function rewriterSources(): Promise<string[]> {
  const directory = path.join(root, "src/parser/cxx");
  return fs
    .readdirSync(directory)
    .filter((file) => /^ast_rewriter.*\.cc$/.test(file))
    .map((file) => path.join(directory, file))
    .sort();
}

async function main(): Promise<number> {
  const missing = missingFrontend(root);
  if (missing) {
    console.error(`${missing} is missing; build it with npm run build:cxx-frontend`);
    return 2;
  }

  const { loadCxx, Parser } = await import("cxx-frontend");
  const S = await import("cxx-frontend/model");
  const { traverse } = await import("cxx-frontend/traverse");

  await loadCxx({
    wasm: fs.readFileSync(
      path.join(root, "packages/cxx-frontend/dist/wasm/cxx-js.wasm"),
    ),
  });

  const classes = astClasses();
  const assigned = new Map<string, Set<string>>();
  const visitedBy = new Map<string, Set<string>>();

  const sysroot =
    process.env.CXX_SYSROOT ?? path.join(root, "build.em/src/lib/wasi-sysroot");

  for (const file of await rewriterSources()) {
    const parser = await Parser.parse({
      appdir: path.join(root, "build.em/src/js"),
      sysroot,
      path: file,
      source: fs.readFileSync(file, "utf8"),
      includePaths: [
        path.join(root, "src/parser"),
        path.join(root, "src/codegen"),
      ],
      exists: fs.existsSync,
      readFile: async (name: string) => {
        try {
          return await fs.promises.readFile(name, "utf8");
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
          throw error;
        }
      },
    });

    try {
      traverse(parser.ast, {
        FunctionDefinition(definition: any) {
          const body = definition.node.functionBody;
          const fn = definition.node.symbol;
          if (!body || !fn) return;
          const visited = visitedClass(S, fn);
          if (!visited || !classes.has(visited)) return;
          let seen = assigned.get(visited);
          if (!seen) assigned.set(visited, (seen = new Set()));
          let files = visitedBy.get(visited);
          if (!files) visitedBy.set(visited, (files = new Set()));
          files.add(path.basename(file));
          traverse(body, {
            MemberExpression(member: any) {
              let base = member.node.baseExpression;
              while (base instanceof S.ImplicitCastExpressionAST)
                base = base.expression;
              if (!(base instanceof S.IdExpressionAST)) return;
              const owner = base.unqualifiedId;
              if (!(owner instanceof S.NameIdAST)) return;
              if (owner.identifier?.name !== "copy") return;
              const id = member.node.unqualifiedId;
              if (!(id instanceof S.NameIdAST)) return;
              const name = id.identifier?.name;
              if (name) seen.add(name);
            },
          });
        },
      });
    } finally {
      parser.dispose();
    }
  }

  let gaps = 0;
  const unaudited: string[] = [];
  for (const [name, cls] of [...classes].sort((a, b) => a[0].localeCompare(b[0]))) {
    const seen = assigned.get(name);
    if (!seen) {
      unaudited.push(name);
      continue;
    }
    for (const [field, typeName] of cls.childFields) {
      if (seen.has(field)) continue;
      if (derivedAfterCopy.has(`${name}.${field}`)) continue;
      console.log(`${name}.${field}: ${typeName} is never used on the copy in ${[...visitedBy.get(name)!].join(", ")}`);
      ++gaps;
    }
  }

  console.error(
    `${gaps} untouched child field(s); ${unaudited.length} AST classes have no rewriter function`,
  );
  return gaps ? 1 : 0;
}

function visitedClass(S: any, fn: any): string | undefined {
  const type = fn.type;
  if (!(type instanceof S.FunctionType)) return;
  const first = [...type.parameterTypes][0];
  const pointer = unqualified(S, first);
  if (!(pointer instanceof S.PointerType)) return;
  const element = unqualified(S, pointer.elementType);
  if (!(element instanceof S.ClassType)) return;
  return element.symbol?.text;
}

function unqualified(S: any, type: any): any {
  return type instanceof S.QualType ? type.elementType : type;
}

process.exitCode = await main();
