import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, Parser } from "cxx-frontend";
import * as S from "cxx-frontend/model";
import { traverse } from "cxx-frontend/traverse";

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
    const { ast } = parser;
    const globalScope = ast.symbol;
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
    assert.equal([...answer.members][0].value, 9007199254740993n);
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

  const { ast } = parser;
  assert.equal(ast.kind, "TranslationUnit");

  const declarations = [...ast.declarationList];
  const written = declarations.slice(-2);
  assert.ok(written[0] instanceof S.SimpleDeclarationAST);
  assert.ok(written[1] instanceof S.FunctionDefinitionAST);

  assert.deepEqual(
    [...S.children(ast)].map((child) => child.node.handle),
    declarations.map((node) => node.handle),
  );

  const names = traverse(
    ast,
    {
      IdExpression(path, found) {
        found.push(path.node.unqualifiedId?.identifier?.name);
      },
    },
    [],
  );
  assert.deepEqual(names, ["x"]);
});

test("declarations link back to the symbols the binder created", async () => {
  await using parser = await Parser.parse({
    path: "/links.cc",
    source:
      "namespace outer::inner { int add(int lhs, int rhs) { return lhs; } }",
  });

  assert.deepEqual(parser.diagnostics, []);

  const { ast } = parser;

  assert.ok(ast.symbol instanceof S.NamespaceSymbol);

  const namespaceDefinition = [...ast.declarationList].find(
    (declaration) => declaration instanceof S.NamespaceDefinitionAST,
  );
  assert.equal(namespaceDefinition.symbol.text, "inner");

  const nested = [...namespaceDefinition.nestedNamespaceSpecifierList];
  assert.deepEqual(
    nested.map((specifier) => specifier.symbol.text),
    ["outer"],
  );

  const parameters = traverse(
    namespaceDefinition,
    {
      ParameterDeclaration(path, found) {
        found.push(path.node.symbol);
      },
    },
    [],
  );

  assert.deepEqual(
    parameters.map((symbol) => symbol.text),
    ["lhs", "rhs"],
  );
  assert.ok(parameters[0] instanceof S.ParameterSymbol);
  assert.equal(parameters[0].type.text, "int");
});

test("catch initialization exposes the selected constructor and arguments", async () => {
  await using parser = await Parser.parse({
    path: "/catch.cc",
    exceptions: true,
    source: `
struct Error {
  Error();
  Error(const Error&);
  Error(Error&, int value = 2);
};
void caught() { try { throw Error{}; } catch (Error value) {} }
void caughtConst() { try { throw Error{}; } catch (const Error value) {} }
`,
  });
  assert.deepEqual(parser.diagnostics, []);
  const handlers = traverse(
    parser.ast,
    {
      TypeExceptionDeclaration(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  assert.equal(handlers.length, 2);
  for (const handler of handlers) {
    assert.ok(handler.exceptionObject instanceof S.VariableSymbol);
    assert.equal(handler.initializer, undefined);
    assert.equal(handler.symbol.initializer, undefined);
    assert.deepEqual(
      [...S.children(handler)].map((child) => child.listKey ?? child.key),
      [...handler.typeSpecifierList]
        .map(() => "typeSpecifierList")
        .concat("declarator"),
    );
    const constructor = handler.symbol.constructorSymbol;
    assert.ok(constructor instanceof S.FunctionSymbol);
    assert.equal([...constructor.type.parameterTypes].length, 2);
    assert.ok(
      handler.exceptionObject.initializer instanceof S.ParenInitializerAST,
    );
    const arguments_ = [...handler.exceptionObject.initializer.expressionList];
    assert.equal(arguments_.length, 2);
    assert.ok(arguments_[0] instanceof S.IdExpressionAST);
    assert.equal(arguments_[0].symbol.handle, handler.exceptionObject.handle);
    assert.ok(arguments_[1] instanceof S.DefaultInitializerExpressionAST);
    assert.equal(arguments_[1].expression.literal.integerValue, 2n);
  }
  assert.equal(
    handlers[0].symbol.constructorSymbol.handle,
    handlers[1].symbol.constructorSymbol.handle,
  );
});

test("array cookie metadata retains only the invoked deallocation function", async () => {
  await using parser = await Parser.parse({
    path: "/cookie.cc",
    source: `
struct E {
  int value;
  static void* operator new[](__SIZE_TYPE__);
  static void operator delete[](void*, __SIZE_TYPE__);
};
void* operator new[](__SIZE_TYPE__, void*) noexcept;
void f(E* p, void* storage) {
  new E[2];
  ::new E[2];
  ::new (storage) E[2];
  new const E[2]{};
  delete[] p;
  ::delete[] p;
}
`,
  });
  assert.deepEqual(parser.diagnostics, []);
  const expressions = traverse(
    parser.ast,
    {
      NewExpression(path, found) {
        found.push(path.node);
      },
      DeleteExpression(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  assert.deepEqual(
    expressions.map((node) => node.arrayCookieRequired),
    [true, true, false, true, true, true],
  );
  for (const node of expressions) {
    assert.equal(node.arrayDeallocationFunction, undefined);
    assert.ok(node.symbol instanceof S.FunctionSymbol);
  }
  assert.ok(expressions[0].deallocationFunction instanceof S.FunctionSymbol);
  assert.equal(expressions[3].symbol.handle, expressions[0].symbol.handle);
});

test("a disposed parser fails every model read", async () => {
  const parser = await Parser.parse({ path: "/dispose.cc", source: "int a;" });
  const { ast } = parser;
  const declarations = ast.declarationList;
  parser.dispose();

  assert.equal(parser.disposed, true);
  assert.throws(() => ast.declarationList, /disposed/);
  assert.throws(() => [...declarations], /disposed/);
  assert.throws(() => [...S.children(ast)], /disposed/);
});

test("new expressions expose the selected failure deallocation function", async () => {
  await using parser = await Parser.parse({
    path: "/new-cleanup.cc",
    source: `
      struct Object {
        Object();
        static void* operator new(__SIZE_TYPE__);
        static void operator delete(void*);
        static void* operator new[](__SIZE_TYPE__, int&);
        static void operator delete[](void*, int&);
      };
      void make(int& placement) {
        new Object;
        new (placement) Object[3];
      }
    `,
  });
  assert.deepEqual(parser.diagnostics, []);
  const expressions = traverse(
    parser.ast,
    {
      NewExpression(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  assert.equal(expressions.length, 2);
  for (const expression of expressions) {
    assert.ok(expression.deallocationFunction instanceof S.FunctionSymbol);
    assert.equal(expression.deallocationFunction.isDeleted, false);
  }
  assert.equal(
    [...expressions[0].deallocationFunction.type.parameterTypes].length,
    1,
  );
  assert.equal(
    [...expressions[1].deallocationFunction.type.parameterTypes].length,
    2,
  );
});

test("thread local storage records the language spelling on its symbol", async () => {
  await using parser = await Parser.parse({
    path: "/thread.cc",
    source: `
thread_local int cpp;
__thread int gnu;
int ordinary;
struct Class {
  static inline thread_local int cpp;
  static __thread int gnu;
};
`,
  });
  assert.deepEqual(parser.diagnostics, []);
  const members = [...parser.ast.symbol.members];
  for (const scope of [
    members,
    [...members.find((s) => s.text === "Class").members],
  ]) {
    const cpp = scope.find((s) => s.text === "cpp");
    const gnu = scope.find((s) => s.text === "gnu");
    assert.equal(cpp.threadLocalKind, "Cxx");
    assert.equal(gnu.threadLocalKind, "Gnu");
    assert.equal(cpp.isThreadLocal, true);
    assert.equal(gnu.isThreadLocal, true);
  }
  const ordinary = members.find((s) => s.text === "ordinary");
  assert.equal(ordinary.threadLocalKind, "None");
  assert.equal(ordinary.isThreadLocal, false);
});

test("constructor checking preserves written initializer syntax and order", async () => {
  await using parser = await Parser.parse({
    path: "/constructor-syntax.cc",
    source: `
      namespace std {
        template<class T> struct initializer_list {
          const T* data;
          unsigned long size;
        };
      }
      struct Value { Value(std::initializer_list<int>, int = 7) {} };
      struct Owner {
        int first;
        Value value;
        int last = 9;
        Owner() : value{1, 2}, first(3) {}
      };
      Value make() { return Value{6}; }
      Value global{4, 5};
    `,
  });
  assert.deepEqual(
    parser.diagnostics.filter((d) => d.severity === "error"),
    [],
  );
  const constructors = traverse(
    parser.ast,
    {
      FunctionDefinition(path, found) {
        if (path.node.symbol.parent?.text === "Owner") found.push(path.node);
      },
    },
    [],
  );
  assert.equal(constructors.length, 1);
  const written = [...constructors[0].functionBody.memInitializerList];
  assert.deepEqual(
    written.map((node) => node.unqualifiedId.identifier.name),
    ["value", "first"],
  );
  const checked = [...constructors[0].symbol.constructorInitializers];
  assert.deepEqual(
    checked.map((node) => node.symbol.text),
    ["first", "value", "last"],
  );
  assert.equal(checked[1].handle, written[0].handle);
  assert.ok(written[0] instanceof S.BracedMemInitializerAST);
  assert.deepEqual(
    [...written[0].bracedInitList.expressionList].map((node) => node.kind),
    ["IntLiteralExpression", "IntLiteralExpression"],
  );
  const globalDeclaration = [...parser.ast.declarationList].at(-1);
  const declarator = [...globalDeclaration.initDeclaratorList][0];
  assert.ok(declarator.initializer instanceof S.BracedInitListAST);
  assert.equal([...declarator.initializer.expressionList].length, 2);
  assert.ok(declarator.symbol.initializer instanceof S.ParenInitializerAST);
  assert.notEqual(
    declarator.initializer.handle,
    declarator.symbol.initializer.handle,
  );
  const constructions = traverse(
    parser.ast,
    {
      BracedTypeConstruction(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  const construction = constructions[0];
  assert.ok(construction.bracedInitList.type);
  assert.equal([...construction.bracedInitList.expressionList].length, 1);
  const arguments_ = [...construction.initialization.expression.expressionList];
  assert.equal(arguments_.length, 2);
  assert.equal(arguments_[0].handle, construction.bracedInitList.handle);
});

test("range-for traversal contains only written loop syntax", async () => {
  await using parser = await Parser.parse({
    path: "/range-syntax.cc",
    source: `
      struct Range {
        int values[2];
        int* begin() { return values; }
        int* end() { return values + 2; }
      };
      int sum(Range& range) {
        int result = 0;
        for (int value : range) result += value;
        return result;
      }
    `,
  });
  assert.deepEqual(parser.diagnostics, []);
  const loops = traverse(
    parser.ast,
    {
      ForRangeStatement(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  assert.equal(loops.length, 1);
  const loop = loops[0];
  assert.deepEqual(
    [...S.children(loop)].map((child) => child.key),
    ["rangeDeclaration", "rangeInitializer", "statement"],
  );
  for (const key of [
    "beginInitializer",
    "endInitializer",
    "condition",
    "increment",
    "element",
    "rangeVariable",
    "beginVariable",
    "endVariable",
  ])
    assert.equal(key in loop, false);
  const declaration = [...loop.rangeDeclaration.initDeclaratorList][0];
  const expanded = [...loop.range.statement.statementList];
  assert.deepEqual(
    expanded.map((node) => node.kind),
    [
      "DeclarationStatement",
      "DeclarationStatement",
      "DeclarationStatement",
      "ForStatement",
    ],
  );
  const body = [...expanded[3].statement.statementList];
  assert.equal(body[1].handle, loop.statement.handle);
  assert.equal(declaration.initializer, undefined);
  assert.ok(declaration.symbol.initializer instanceof S.EqualInitializerAST);
});

test("structured binding traversal contains only written syntax", async () => {
  await using parser = await Parser.parse({
    path: "/binding-syntax.cc",
    source:
      "int sum() { int values[2] = {3, 4}; auto& [first, second] = values; return first + second; }",
  });
  assert.deepEqual(parser.diagnostics, []);
  const declarations = traverse(
    parser.ast,
    {
      StructuredBindingDeclaration(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  assert.equal(declarations.length, 1);
  const declaration = declarations[0];
  assert.equal("hiddenVariable" in declaration, false);
  assert.equal("bindingDeclaratorList" in declaration, false);
  assert.deepEqual(
    [...S.children(declaration)].map((child) => child.listKey ?? child.key),
    ["declSpecifierList", "bindingList", "bindingList", "initializer"],
  );
  const bindings = [...declaration.binding.bindingDeclaratorList];
  assert.deepEqual(
    bindings.map((node) => node.symbol.text),
    ["first", "second"],
  );
  assert.ok(bindings.every((node) => node.symbol.isStructuredBindingAlias));
  assert.ok(declaration.initializer instanceof S.EqualInitializerAST);
});

test("GNU case range endpoints are exposed and traversed in the AST", async () => {
  await using parser = await Parser.parse({
    path: "/ranges.cc",
    source:
      "int f(int n) { switch (n) { case 2 ... 4: return 1; default: return 0; } }",
  });
  assert.deepEqual(parser.diagnostics, []);
  const ranges = traverse(
    parser.ast,
    {
      CaseStatement(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  assert.equal(ranges.length, 1);
  assert.ok(ranges[0].expression instanceof S.IntLiteralExpressionAST);
  assert.ok(ranges[0].upperExpression instanceof S.IntLiteralExpressionAST);
  assert.ok(ranges[0].ellipsisToken);
  const literals = traverse(
    parser.ast,
    {
      IntLiteralExpression(path, found) {
        found.push(path.node.literal.integerValue);
      },
    },
    [],
  );
  assert.ok(literals.includes(2n));
  assert.ok(literals.includes(4n));
});

test("written initializers retain types and share expressions with normalization", async () => {
  await using parser = await Parser.parse({
    path: "/initializer-semantics.cc",
    source: `
      namespace std {
        template<class T> struct initializer_list {
          const T* data;
          unsigned long size;
          initializer_list() : data(nullptr), size(0) {}
          const T* begin();
          const T* end();
        };
      }
      struct Pair { long first; int second = 9; };
      long matrix[][2] = {{1, 2}};
      Pair pair{3};
      int sum() {
        int result = 0;
        for (int value : {1, 2}) result += value;
        return result;
      }
      struct Object {
        long values[2];
        Object() : values{4, 5} {}
      };
    `,
  });
  assert.deepEqual(parser.diagnostics, []);
  const declarations = traverse(
    parser.ast,
    {
      InitDeclarator(path, found) {
        if (
          path.node.symbol?.text === "matrix" ||
          path.node.symbol?.text === "pair"
        )
          found.push(path.node);
      },
    },
    [],
  );
  const [matrix, pair] = declarations;
  assert.equal(matrix.initializer.handle, matrix.symbol.initializer.handle);
  const outer = matrix.initializer.expression;
  assert.equal(outer.type.handle, matrix.symbol.type.handle);
  const inner = [...outer.expressionList][0];
  assert.equal(inner.type.handle, outer.type.elementType.handle);
  const literal = [...inner.expressionList][0];
  assert.ok(literal.type instanceof S.IntType);
  const converted = [...inner.initialization.arguments][0];
  assert.ok(converted instanceof S.ImplicitCastExpressionAST);
  assert.ok(converted.type instanceof S.LongIntType);
  assert.equal(converted.expression.handle, literal.handle);
  assert.equal([...pair.initializer.expressionList].length, 1);
  assert.equal([...pair.initializer.initialization.arguments].length, 2);
  assert.equal(pair.initializer.type.handle, pair.symbol.type.handle);
  const constructors = traverse(
    parser.ast,
    {
      FunctionDefinition(path, found) {
        if (path.node.symbol.parent?.text === "Object") found.push(path.node);
      },
    },
    [],
  );
  const constructor = constructors[0];
  const written = [...constructor.functionBody.memInitializerList][0];
  const checked = [...constructor.symbol.constructorInitializers][0];
  assert.equal(written.handle, checked.handle);
  assert.equal(written.bracedInitList.type.handle, written.symbol.type.handle);
  assert.equal(
    written.initialization.expression.handle,
    written.bracedInitList.handle,
  );
  const loops = traverse(
    parser.ast,
    {
      ForRangeStatement(path, found) {
        found.push(path.node);
      },
    },
    [],
  );
  const range = loops[0].rangeInitializer;
  assert.ok(range.type instanceof S.ClassType);
  assert.equal([...range.expressionList].length, 2);
  assert.equal(
    [...range.initialization.arguments][0].handle,
    [...range.expressionList][0].handle,
  );
});

test("constexpr folding and reference conversions preserve written initializers", async () => {
  await using parser = await Parser.parse({
    path: "/normalized-initializers.cc",
    source: `
      constexpr int answer() { return 123; }
      constexpr int value = answer();
      constexpr const int& reference() { return value; }
      constexpr const int& alias = reference();
      struct A { int a; };
      struct P { int p; };
      struct B : P, A { int b; };
      struct HoldsBase { A& r; };
      void bind(B& object) { HoldsBase holder{object}; }
    `,
  });
  assert.deepEqual(parser.diagnostics, []);
  const declarations = traverse(
    parser.ast,
    {
      InitDeclarator(path, found) {
        if (["value", "alias", "holder"].includes(path.node.symbol?.text))
          found.push(path.node);
      },
    },
    [],
  );
  const [value, alias, holder] = declarations;
  for (const declaration of [value, alias]) {
    assert.ok(declaration.initializer instanceof S.EqualInitializerAST);
    assert.ok(
      declaration.initializer.expression instanceof S.CallExpressionAST,
    );
    assert.ok(declaration.symbol.initializer instanceof S.ConstExpressionAST);
    assert.equal(
      declaration.symbol.initializer.expression.handle,
      declaration.initializer.expression.handle,
    );
  }
  assert.equal(value.symbol.constValue, 123n);
  assert.ok(alias.symbol.constValue instanceof S.ConstAddress);
  assert.equal(alias.symbol.constValue.symbol.handle, value.symbol.handle);
  const written = [...holder.initializer.expressionList][0];
  assert.ok(written instanceof S.IdExpressionAST);
  assert.equal(written.type.symbol.text, "B");
  const converted = [...holder.initializer.initialization.arguments][0];
  assert.ok(converted instanceof S.ImplicitCastExpressionAST);
  assert.equal(converted.castKind, "DerivedToBaseConversion");
  assert.equal(converted.type.symbol.text, "A");
  assert.equal(converted.expression.handle, written.handle);
});
