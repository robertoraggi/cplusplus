import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, Parser, Semantic as S } from "../dist/index.js";

await loadCxx({
  wasm: await readFile(new URL("../dist/wasm/cxx-js.wasm", import.meta.url)),
});

test("read-only model exposes symbol identity, types and integer constants", async () => {
  const parser = await Parser.parse({
    path: "/model.cc",
    source:
      "struct Point { int x; }; enum class Answer : long long { value = 9007199254740993LL }; Point point;",
  });
  try {
    assert.deepEqual(parser.diagnostics, []);
    const { globalScope, ast } = parser.model;
    assert.ok(ast instanceof S.TranslationUnitAST);
    const point = [...globalScope.members].find((s) => s.text === "Point");
    assert.ok(point instanceof S.ClassSymbol);
    assert.equal(point.parent.handle, globalScope.handle);
    assert.equal(point.type.symbol.handle, point.handle);
    const field = [...point.members].find((s) => s.text === "x");
    assert.ok(field instanceof S.FieldSymbol);
    assert.ok(field.type instanceof S.IntType);
    const answer = [...globalScope.members].find((s) => s.text === "Answer");
    assert.ok(answer instanceof S.ScopedEnumSymbol);
    assert.equal([...answer.members][0].value.value, 9007199254740993n);
    assert.throws(() => {
      point.isFinal = true;
    }, TypeError);
    assert.equal(point.setName, undefined);
    parser.dispose();
    assert.throws(() => point.name, /disposed/);
    assert.throws(() => field.type, /disposed/);
  } finally {
    parser.dispose();
  }
});

test("the AST is traversed by kind, by children and by visitor", async () => {
  await using parser = await Parser.parse({
    path: "/traverse.cc",
    source: "int a; int f(int x) { return x; }",
  });

  assert.deepEqual(parser.diagnostics, []);

  const { ast } = parser.model;
  assert.equal(ast.kind, S.ASTKind.TranslationUnit);

  const declarations = [...ast.declarationList];
  const written = declarations.slice(-2);
  assert.ok(written[0] instanceof S.SimpleDeclarationAST);
  assert.ok(written[1] instanceof S.FunctionDefinitionAST);

  assert.deepEqual(
    [...S.children(ast)].map((node) => node.handle),
    declarations.map((node) => node.handle),
  );

  class Names extends S.RecursiveASTVisitor {
    found = [];
    visitIdExpression(node, context) {
      this.found.push(node.unqualifiedId?.identifier?.name);
      this.visitChildren(node, context);
    }
  }

  const names = new Names();
  names.accept(ast, undefined);
  assert.deepEqual(names.found, ["x"]);
});

test("declarations link back to the symbols the binder created", async () => {
  await using parser = await Parser.parse({
    path: "/links.cc",
    source:
      "namespace outer::inner { int add(int lhs, int rhs) { return lhs; } }",
  });

  assert.deepEqual(parser.diagnostics, []);

  const { ast, globalScope } = parser.model;

  assert.ok(ast.symbol instanceof S.NamespaceSymbol);
  assert.equal(ast.symbol.handle, globalScope.handle);

  const namespaceDefinition = [...ast.declarationList].find(
    (declaration) => declaration instanceof S.NamespaceDefinitionAST,
  );
  assert.equal(namespaceDefinition.symbol.text, "inner");

  const nested = [...namespaceDefinition.nestedNamespaceSpecifierList];
  assert.deepEqual(
    nested.map((specifier) => specifier.symbol.text),
    ["outer"],
  );

  const parameters = [];
  class Parameters extends S.RecursiveASTVisitor {
    visitParameterDeclaration(node, context) {
      parameters.push(node.symbol);
      this.visitChildren(node, context);
    }
  }
  new Parameters().accept(namespaceDefinition, undefined);

  assert.deepEqual(
    parameters.map((symbol) => symbol.text),
    ["lhs", "rhs"],
  );
  assert.ok(parameters[0] instanceof S.ParameterSymbol);
  assert.equal(parameters[0].type.text, "int");
});

test("a disposed parser fails every model read", async () => {
  const parser = await Parser.parse({ path: "/dispose.cc", source: "int a;" });
  const { ast } = parser.model;
  const declarations = ast.declarationList;
  parser.dispose();

  assert.equal(parser.disposed, true);
  assert.throws(() => ast.declarationList, /disposed/);
  assert.throws(() => [...declarations], /disposed/);
  assert.throws(() => [...S.children(ast)], /disposed/);
});
