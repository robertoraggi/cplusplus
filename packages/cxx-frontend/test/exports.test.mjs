import assert from "node:assert/strict";
import test from "node:test";
import loadCxx, {
  isCxxLoaded,
  Linker,
  Parser,
  parse as namedParse,
} from "cxx-frontend";
import loadCxxNode from "cxx-frontend/node";
import parse from "cxx-frontend/parse";
import link from "cxx-frontend/link";
import * as model from "cxx-frontend/model";
import traverseDefault, {
  NodePath,
  traverse,
  walk,
} from "cxx-frontend/traverse";
import { LanguageServer } from "cxx-frontend/lsp";

await loadCxxNode();

test("each entry point exports what it says", async () => {
  assert.equal(typeof loadCxx, "function");
  assert.equal(typeof loadCxxNode, "function");
  assert.equal(isCxxLoaded(), true);
  assert.equal(parse, namedParse);
  assert.equal(typeof link, "function");
  assert.equal(typeof Linker.create, "function");
  assert.equal(typeof Parser.parse, "function");
  assert.equal(typeof LanguageServer, "function");
  assert.equal(typeof traverse, "function");
  assert.equal(traverseDefault, traverse);
  assert.equal(typeof walk, "function");
  assert.equal(typeof NodePath, "function");
  assert.equal(typeof model.children, "function");
  assert.ok(model.TranslationUnitAST);
});

test("the default parse entry parses", async () => {
  await using parser = await parse({ path: "/exports.cc", source: "int x;" });
  assert.deepEqual(parser.diagnostics, []);

  const { ast } = parser;
  assert.equal(ast.kind, "TranslationUnit");
  assert.ok(ast instanceof model.TranslationUnitAST);

  const declared = [...walk(ast)]
    .filter((path) => path.isInitDeclarator())
    .map((path) => path.node.symbol?.text);
  assert.ok(declared.includes("x"));

  const kinds = traverse(
    ast,
    { SimpleDeclaration: (path, seen) => seen.add(path.node.kind) },
    new Set(),
  );
  assert.ok(kinds.has("SimpleDeclaration"));
});
