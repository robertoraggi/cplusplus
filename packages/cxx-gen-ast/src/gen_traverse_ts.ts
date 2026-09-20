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
import { spawnSync } from "node:child_process";
import { cpy_header } from "./cpy_header.ts";
import type { AST } from "./parseAST.ts";
import type { ModelIndex } from "./parseModel.ts";

const kindOf = (name: string) => name.replace(/AST$/, "");

export function gen_traverse_ts({
  ast,
  index,
  root,
}: {
  ast: AST;
  index: ModelIndex;
  root: string;
}) {
  const categoryOf = (name: string): string | undefined => {
    const base = index.classOf(`::cxx::${name}`)?.bases[0];
    const owner = base ? index.classOf(base.type) : undefined;
    if (!owner || owner.unqualifiedName === "AST") return undefined;
    if (index.classOf(owner.name)?.bases[0]?.type !== "::cxx::AST")
      throw new Error(`${owner.name} is not a direct base of AST`);
    return kindOf(owner.unqualifiedName);
  };

  const nodes = ast.nodes.map((node) => node.name);
  const categories = [...ast.bases];

  const byCategory = new Map<string, string[]>();
  for (const node of nodes) {
    const category = categoryOf(node);
    if (!category) continue;
    byCategory.set(category, [
      ...(byCategory.get(category) ?? []),
      kindOf(node),
    ]);
  }

  const out: string[] = [];

  out.push(`// Generated file by: gen_traverse_ts.ts\n${cpy_header}
import { type ASTKind, children } from "./Semantic.js";
import type {
  AST,
${[...nodes, ...categories].map((name) => `  ${name},`).join("\n")}
} from "./Semantic.js";`);

  out.push(`export interface ASTNodes {
${ast.nodes.map((node) => `  ${kindOf(node.name)}: ${node.name};`).join("\n")}
}

export interface ASTCategories {
${categories.map((name) => `  ${kindOf(name)}: ${name};`).join("\n")}
}

export type ASTCategory = keyof ASTCategories;

export type VisitorKey = ASTKind | ASTCategory;

const categoryOf: Partial<Record<ASTKind, ASTCategory>> = {};

for (const [category, kinds] of Object.entries({
${[...byCategory]
  .map(
    ([category, kinds]) =>
      `  ${category}: [${kinds.map((kind) => `"${kind}"`).join(", ")}],`,
  )
  .join("\n")}
}) as [ASTCategory, ASTKind[]][])
  for (const kind of kinds) categoryOf[kind] = category;`);

  out.push(`export type VisitNodeFunction<S, T extends AST> = (
  path: NodePath<T>,
  state: S,
) => void;

export interface VisitNodeObject<S, T extends AST> {
  enter?: VisitNodeFunction<S, T>;
  exit?: VisitNodeFunction<S, T>;
}

export type VisitNode<S, T extends AST> =
  | VisitNodeFunction<S, T>
  | VisitNodeObject<S, T>;

export type Visitor<S = undefined> = {
  [K in keyof (ASTNodes & ASTCategories)]?: VisitNode<
    S,
    (ASTNodes & ASTCategories)[K]
  >;
} & {
  enter?: VisitNodeFunction<S, AST>;
  exit?: VisitNodeFunction<S, AST>;
};

export class NodePath<T extends AST = AST> {
  readonly node: T;
  readonly parentPath: NodePath | undefined;
  readonly key: string | number;
  readonly listKey: string | undefined;

  #skipped = false;
  #stopped = false;

  constructor(
    node: T,
    parentPath?: NodePath,
    key: string | number = "",
    listKey?: string,
  ) {
    this.node = node;
    this.parentPath = parentPath;
    this.key = key;
    this.listKey = listKey;
  }

  get parent(): AST | undefined {
    return this.parentPath?.node;
  }

  get kind(): ASTKind {
    return this.node.kind;
  }

  get category(): ASTCategory | undefined {
    return categoryOf[this.node.kind];
  }

  get depth(): number {
    let depth = 0;
    for (let path = this.parentPath; path; path = path.parentPath) ++depth;
    return depth;
  }

  get shouldSkip(): boolean {
    return this.#skipped;
  }

  get shouldStop(): boolean {
    return this.#stopped;
  }

  skip(): void {
    this.#skipped = true;
  }

  stop(): void {
    this.#skipped = true;
    this.#stopped = true;
  }

${ast.nodes
  .map(
    (
      node,
    ) => `  is${kindOf(node.name)}(this: NodePath): this is NodePath<${node.name}> {
    return this.node.kind === "${kindOf(node.name)}";
  }`,
  )
  .join("\n\n")}

${categories
  .map(
    (name) => `  is${kindOf(name)}(this: NodePath): this is NodePath<${name}> {
    return categoryOf[this.node.kind] === "${kindOf(name)}";
  }`,
  )
  .join("\n\n")}

  *[Symbol.iterator](): Generator<NodePath> {
    yield* this.children();
  }

  *children(): Generator<NodePath> {
    for (const { node, key, listKey } of children(this.node))
      yield new NodePath(node, this, key, listKey);
  }

  *ancestors(): Generator<NodePath> {
    for (let path = this.parentPath; path; path = path.parentPath) yield path;
  }

  *descendants(): Generator<NodePath> {
    for (const path of walk(this)) if (path !== this) yield path;
  }

  find(predicate: (path: NodePath) => boolean): NodePath | undefined {
    for (let path: NodePath | undefined = this; path; path = path.parentPath)
      if (predicate(path)) return path;
    return undefined;
  }

  findParent(predicate: (path: NodePath) => boolean): NodePath | undefined {
    return this.parentPath?.find(predicate);
  }

  traverse<S = undefined>(visitor: Visitor<S>, state?: S): S {
    for (const path of this.children())
      if (visit(path, visitor, state as S)) break;
    return state as S;
  }

  toString(): string {
    return this.node.kind;
  }
}

function pathOf(root: AST | NodePath): NodePath {
  return root instanceof NodePath ? root : new NodePath(root);
}

export function* walk(root: AST | NodePath): Generator<NodePath> {
  const stack = [pathOf(root)];

  while (stack.length) {
    const path = stack.pop()!;

    yield path;

    if (path.shouldStop) return;
    if (path.shouldSkip) continue;

    const children = [...path.children()];
    for (let i = children.length - 1; i >= 0; --i) stack.push(children[i]!);
  }
}

function dispatch<S>(
  path: NodePath,
  visitor: Visitor<S>,
  state: S,
  phase: "enter" | "exit",
): void {
  visitor[phase]?.(path, state);
  if (path.shouldStop) return;

  for (const key of [path.node.kind, categoryOf[path.node.kind]]) {
    if (!key) continue;
    const entry = visitor[key] as VisitNode<S, AST> | undefined;
    if (!entry) continue;
    if (typeof entry === "function") {
      if (phase === "enter") entry(path, state);
    } else {
      entry[phase]?.(path, state);
    }
    if (path.shouldStop) return;
  }
}

function visit<S>(root: NodePath, visitor: Visitor<S>, state: S): boolean {
  const stack: { path: NodePath; children?: NodePath[]; index: number }[] = [
    { path: root, index: 0 },
  ];

  while (stack.length) {
    const frame = stack.at(-1)!;

    if (!frame.children) {
      dispatch(frame.path, visitor, state, "enter");
      if (frame.path.shouldStop) return true;
      if (frame.path.shouldSkip) {
        stack.pop();
        continue;
      }
      frame.children = [...frame.path.children()];
    }

    const child = frame.children[frame.index++];
    if (child) {
      stack.push({ path: child, index: 0 });
      continue;
    }

    stack.pop();
    dispatch(frame.path, visitor, state, "exit");
    if (frame.path.shouldStop) return true;
  }

  return false;
}

export function traverse<S = undefined>(
  root: AST | NodePath,
  visitor: Visitor<S>,
  state?: S,
): S {
  visit(pathOf(root), visitor, state as S);
  return state as S;
}`);

  const output = `${root}/packages/cxx-frontend/src/Traverse.ts`;
  fs.writeFileSync(output, out.join("\n\n"));

  const result = spawnSync(
    `${root}/node_modules/.bin/prettier`,
    ["--write", output],
    { encoding: "utf8" },
  );
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`prettier failed: ${result.stderr}`);
}
