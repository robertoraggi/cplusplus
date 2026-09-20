import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, Parser } from "cxx-frontend";
import * as S from "cxx-frontend/model";
import { NodePath, traverse, walk } from "cxx-frontend/traverse";

await loadCxx({
  wasm: await readFile(new URL("../dist/wasm/cxx-js.wasm", import.meta.url)),
});

const source = `
namespace outer {
namespace inner {
int answer() { return 42; }
}  // namespace inner
int other() { return answer(); }
}  // namespace outer
`;

async function parse() {
  return await Parser.parse({ path: "/traverse.cc", source });
}

test("a typed visitor is called for the nodes it names", async () => {
  const parser = await parse();
  try {
    const names = [];
    traverse(parser.model.ast, {
      NamespaceDefinition(path) {
        assert.ok(path.node instanceof S.NamespaceDefinitionAST);
        names.push(path.node.identifier?.name);
      },
    });
    assert.deepEqual(names, ["outer", "inner"]);
  } finally {
    parser.dispose();
  }
});

test("enter and exit bracket every node", async () => {
  const parser = await parse();
  try {
    let entered = 0;
    let exited = 0;
    let deepest = 0;
    traverse(parser.model.ast, {
      enter(path) {
        ++entered;
        deepest = Math.max(deepest, path.depth);
      },
      exit() {
        ++exited;
      },
    });
    assert.equal(entered, exited);
    assert.ok(entered > 10);
    assert.ok(deepest > 3);
  } finally {
    parser.dispose();
  }
});

test("a category key visits every node deriving from it", async () => {
  const parser = await parse();
  try {
    const declarations = [];
    traverse(parser.model.ast, {
      Declaration(path) {
        declarations.push(path.node.kind);
      },
    });
    assert.ok(declarations.includes("FunctionDefinition"));
    assert.ok(declarations.includes("NamespaceDefinition"));
  } finally {
    parser.dispose();
  }
});

test("an object visitor separates enter from exit", async () => {
  const parser = await parse();
  try {
    const order = [];
    traverse(parser.model.ast, {
      NamespaceDefinition: {
        enter(path) {
          order.push(`enter ${path.node.identifier?.name}`);
        },
        exit(path) {
          order.push(`exit ${path.node.identifier?.name}`);
        },
      },
    });
    assert.deepEqual(order, [
      "enter outer",
      "enter inner",
      "exit inner",
      "exit outer",
    ]);
  } finally {
    parser.dispose();
  }
});

test("skip prunes the subtree and stop ends the traversal", async () => {
  const parser = await parse();
  try {
    const skipped = [];
    traverse(parser.model.ast, {
      NamespaceDefinition(path) {
        skipped.push(path.node.identifier?.name);
        path.skip();
      },
    });
    assert.deepEqual(skipped, ["outer"]);

    let visited = 0;
    traverse(parser.model.ast, {
      enter(path) {
        ++visited;
        if (path.isFunctionDefinition()) path.stop();
      },
    });
    const total = traverse(
      parser.model.ast,
      { enter: (_path, state) => ++state.count },
      { count: 0 },
    ).count;
    assert.ok(visited < total);
  } finally {
    parser.dispose();
  }
});

test("state is threaded through and returned", async () => {
  const parser = await parse();
  try {
    const state = traverse(
      parser.model.ast,
      {
        FunctionDefinition(path, state) {
          state.functions.push(path.node.kind);
        },
      },
      { functions: [] },
    );
    assert.equal(state.functions.length, 2);
  } finally {
    parser.dispose();
  }
});

test("a path knows its parent, its key and its ancestors", async () => {
  const parser = await parse();
  try {
    let inner;
    traverse(parser.model.ast, {
      NamespaceDefinition(path) {
        if (path.node.identifier?.name === "inner") inner = path;
      },
    });
    assert.ok(inner instanceof NodePath);
    assert.equal(inner.listKey, "declarationList");
    assert.equal(inner.key, 0);
    assert.ok(inner.parent instanceof S.NamespaceDefinitionAST);
    assert.equal(inner.parentPath.node.identifier?.name, "outer");

    const ancestors = [...inner.ancestors()].map((path) => path.node.kind);
    assert.deepEqual(ancestors, ["NamespaceDefinition", "TranslationUnit"]);

    const unit = inner.find((path) => path.isTranslationUnit());
    assert.equal(unit.node.handle, parser.model.ast.handle);
    assert.equal(
      inner.findParent((path) => path.isTranslationUnit()),
      unit,
    );
  } finally {
    parser.dispose();
  }
});

test("walk yields every path and honours skip", async () => {
  const parser = await parse();
  try {
    const all = [...walk(parser.model.ast)];
    assert.equal(all[0].node.kind, "TranslationUnit");
    assert.ok(all.every((path) => path instanceof NodePath));

    const namespaces = [];
    for (const path of walk(parser.model.ast))
      if (path.isNamespaceDefinition())
        namespaces.push(path.node.identifier?.name);
    assert.deepEqual(namespaces, ["outer", "inner"]);

    const pruned = [];
    for (const path of walk(parser.model.ast)) {
      if (path.isFunctionDefinition()) {
        path.skip();
        continue;
      }
      if (path.isNameId()) pruned.push(path.node.identifier?.name);
    }
    assert.ok(!pruned.includes("answer"));

    const names = [];
    for (const path of walk(parser.model.ast))
      if (path.isNameId()) names.push(path.node.identifier?.name);
    assert.ok(names.includes("answer"));
  } finally {
    parser.dispose();
  }
});

test("a path traverses and iterates its own subtree", async () => {
  const parser = await parse();
  try {
    const root = new NodePath(parser.model.ast);
    const top = [...root];
    assert.ok(top.every((path) => path.isDeclaration()));
    assert.equal(top.filter((path) => path.isNamespaceDefinition()).length, 1);

    const inner = [...root.descendants()].find(
      (path) =>
        path.isNamespaceDefinition() && path.node.identifier?.name === "inner",
    );
    assert.ok(inner instanceof NodePath);

    const seen = [];
    inner.traverse({
      FunctionDefinition(path) {
        seen.push(path.node.kind);
      },
    });
    assert.deepEqual(seen, ["FunctionDefinition"]);

    const stopped = root.traverse(
      {
        Declaration(path, state) {
          state.push(path.node.kind);
          if (path.isNamespaceDefinition()) path.stop();
        },
      },
      [],
    );
    assert.equal(stopped.at(-1), "NamespaceDefinition");
    assert.ok(!stopped.includes("FunctionDefinition"));
  } finally {
    parser.dispose();
  }
});
