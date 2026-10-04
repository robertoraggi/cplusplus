import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, Parser, TraceEmitter } from "cxx-frontend";

const wasm = await readFile(
  new URL("../dist/wasm/cxx-js.wasm", import.meta.url),
);

await loadCxx({ wasm });

async function trace(
  source,
  { path = "trace.cc", debugInfo = true, emitter = new TraceEmitter() } = {},
) {
  const parser = await Parser.parse({ path, source, debugInfo });
  try {
    const errors = parser.diagnostics.filter(
      (d) => d.severity === "error" || d.severity === "fatal",
    );
    assert.deepEqual(errors, [], "the source must compile cleanly");
    parser.emitWith(emitter);
    return emitter.trace;
  } finally {
    parser.dispose();
  }
}

function checkWellFormed(text) {
  const defined = new Set();
  const blocks = new Set();
  const branched = new Set();

  for (const line of text.split("\n")) {
    const block = line.match(/^\^bb(\d+)(?:\(([^)]*)\))?:/);
    if (block) {
      blocks.add(block[1]);
      for (const [, parameter] of (block[2] ?? "").matchAll(/%(\d+)/g))
        defined.add(parameter);
      continue;
    }

    const [, result, rest] = line.match(/^\s*%(\d+) = (.*)$/) ?? [
      undefined,
      undefined,
      line,
    ];

    for (const [, used] of rest.matchAll(/%(\d+)/g))
      assert.ok(
        defined.has(used),
        `%${used} is used before it is defined:\n${line}`,
      );

    for (const [, target] of rest.matchAll(/\^bb(\d+)/g)) branched.add(target);

    if (result) defined.add(result);
  }

  for (const target of branched)
    assert.ok(
      blocks.has(target),
      `^bb${target} is branched to but never opened`,
    );

  return { defined, blocks };
}

test("a function body reaches the emitter as a value trace", async () => {
  const text = await trace(`
int add(int a, int b) { return a + b; }
`);

  checkWellFormed(text);

  assert.match(text, /^module "trace\.cc"/m);
  assert.match(text, /func @_Z3addii : \(i32, i32\) -> \(i32\) External/);
  assert.match(text, /AddSignedInt %\d+, %\d+ : i32/);
  assert.match(text, /^\s*return %\d+$/m);
  assert.match(text, /^endmodule$/m);
});

test("only signed arithmetic in a promoted type carries undefined overflow", async () => {
  const cases = [
    ["int f(int a, int b) { return a + b; }", /AddSignedInt/],
    ["long f(long a, long b) { return a - b; }", /SubSignedInt/],
    ["int f(int a, int b) { return a * b; }", /MulSignedInt/],
    ["int f(int a) { return -a; }", /SubSignedInt/],
    ["int f(int a) { return ++a; }", /AddSignedInt/],
    ["int f(int a) { return a--; }", /AddSignedInt/],
    ["int f(int& a, int b) { return a += b; }", /AddSignedInt/],
    [
      "unsigned f(unsigned a, unsigned b) { return a + b; }",
      /AddInt/,
      /(Add|Sub|Mul)SignedInt/,
    ],
    [
      "unsigned long f(unsigned long a) { a++; return a; }",
      /AddInt/,
      /(Add|Sub|Mul)SignedInt/,
    ],
    [
      "int f(int& a, unsigned b) { return a += b; }",
      /AddInt/,
      /(Add|Sub|Mul)SignedInt/,
    ],
    ["short f(short a, short b) { return a + b; }", /AddSignedInt/],
    ["void f(short& a) { a++; }", /AddInt/, /(Add|Sub|Mul)SignedInt/],
    ["void f(char& a) { ++a; }", /AddInt/, /(Add|Sub|Mul)SignedInt/],
  ];

  for (const [source, expected, forbidden] of cases) {
    const text = await trace(source);
    checkWellFormed(text);
    assert.match(text, expected, source);
    if (forbidden) assert.doesNotMatch(text, forbidden, source);
  }
});

test("inline attributes reach the emitter", async () => {
  const text = await trace(`
__attribute__((noinline)) int kept(int a) { return a + 1; }
__attribute__((always_inline)) inline int forced(int a) { return a + 2; }
inline int hinted(int a) { return a + 3; }
int plain(int a) { return a + 4; }
int entry(int a) { return kept(a) + forced(a) + hinted(a) + plain(a); }
`);

  checkWellFormed(text);

  assert.match(text, /func @_Z4kepti : \(i32\) -> \(i32\) External noinline$/m);
  assert.match(text, /func @_Z6forcedi : .* alwaysinline$/m);
  assert.match(text, /func @_Z6hintedi : .* inlinehint$/m);
  assert.match(text, /func @_Z5plaini : \(i32\) -> \(i32\) External$/m);
});

test("a function-try-block is reported as unsupported like a try statement", async () => {
  const functionTryBlock = await trace(`
struct A {
  int m;
  A(int v) try : m(v + 1) { m += 2; } catch (...) { }
};
A make(int v) { return A(v); }
`);
  const tryStatement = await trace(`
void f() { try { } catch (...) { } }
`);

  checkWellFormed(functionTryBlock);
  checkWellFormed(tryStatement);

  assert.match(functionTryBlock, /todo "try-statement-function-body"/);
  assert.match(tryStatement, /todo "try-block-statement"/);
});

test("control flow produces blocks that are opened before they are entered", async () => {
  const text = await trace(`
int clamp(int x) {
  if (x < 0) return 0;
  while (x > 100) x = x - 1;
  return x;
}
`);

  const { blocks } = checkWellFormed(text);

  assert.ok(blocks.size >= 4, `expected several blocks, got ${blocks.size}`);
  assert.match(text, /cond_br %\d+, \^bb\d+, \^bb\d+/);
  assert.match(text, /icmp\.Signed\w+ %\d+, %\d+ : i1/);
});

test("a switch reaches the emitter with its case values", async () => {
  const text = await trace(`
int pick(int x) {
  switch (x) {
    case 1: return 10;
    case 7: return 70;
    default: return 0;
  }
}
`);

  checkWellFormed(text);
  assert.match(text, /switch %\d+, default \^bb\d+ \[1: \^bb\d+, 7: \^bb\d+\]/);
});

test("calls, globals and member access reach the emitter", async () => {
  const text = await trace(`
struct Point { int x; int y; };

int counter;

int sum(Point p) { return p.x + p.y; }

int use() {
  Point p{1, 2};
  counter = sum(p);
  return counter;
}
`);

  checkWellFormed(text);

  assert.match(text, /global @counter : i32/);
  assert.match(text, /member %\d+\.\d+ : ptr<i32>/);
  assert.match(text, /call @_Z3sum5Point\(%\d+\) : i32/);
  assert.match(text, /addressof @counter : ptr<i32>/);
});

test("floating point and casts keep their result types", async () => {
  const text = await trace(`
double scale(int n) { return n * 1.5; }
`);

  checkWellFormed(text);
  assert.match(text, /SignedIntToFloat %\d+ : f64/);
  assert.match(text, /MulFloat %\d+, %\d+ : f64/);
});

test("the wasm import and export attributes reach the emitter", async () => {
  const text = await trace(`
extern "C" __attribute__((import_module("host")))
__attribute__((import_name("host_add"))) int imported_add(int, int);

extern "C" __attribute__((export_name("add"))) int exported_add(int a, int b) {
  return imported_add(a, b);
}

[[gnu::used]] static int kept = 7;

int anchor() { return kept; }
`);

  checkWellFormed(text);

  assert.match(
    text,
    /func @imported_add : \(i32, i32\) -> \(i32\) External import_module "host" import_name "host_add"$/m,
  );
  assert.match(
    text,
    /func @exported_add : \(i32, i32\) -> \(i32\) External export_name "add" used$/m,
  );
  assert.match(text, /global @_ZL4kept : i32 Internal used = /);
});

test("the trace is stable across runs", async () => {
  const source = `
int fib(int n) {
  if (n < 2) return n;
  return fib(n - 1) + fib(n - 2);
}
`;

  assert.equal(await trace(source), await trace(source));
});

test("value and block handles are released at the end of each function", async () => {
  const parser = await Parser.parse({
    path: "scopes.cc",
    source: `
int a(int x) { int t = x + 1; return t + x; }
int b(int x) { int t = x + 2; return t + x; }
int c(int x) { int t = x + 3; return t + x; }
`,
  });

  try {
    const emitter = new TraceEmitter();
    parser.emitWith(emitter);

    const { values, blocks } = emitter.liveHandles;
    assert.equal(values, 0, "every value handle should have been released");
    assert.equal(blocks, 0, "every block handle should have been released");

    const defined = (emitter.trace.match(/^\s*%\d+ = /gm) ?? []).length;
    assert.ok(defined > 20, `expected a real trace, got ${defined} values`);
  } finally {
    parser.dispose();
  }
});

test("nothing is emitted for a translation unit with errors", async () => {
  const parser = await Parser.parse({
    path: "broken.cc",
    source: "int f() { return undeclared_name; }\n",
  });

  try {
    const emitter = new TraceEmitter();
    parser.emitWith(emitter);
    assert.equal(emitter.trace, "");
  } finally {
    parser.dispose();
  }
});

test("debug metadata crosses the WASM boundary as typed objects and handles", async () => {
  const emitter = new TraceEmitter();
  const text = await trace(
    `
struct Node {
  Node* next;
  const volatile int* value;
  int data[3];
  int method(int arg) { return data[0] + arg; }
};
enum class Choice : unsigned { one, two };
int inspect(Node* node, Choice choice, int Node::* member) {
  int result = node->method(2);
  { int nested = node->data[1]; result += nested; }
  return result + (choice == Choice::one) + node->*member;
}
`,
    { path: "debug.cc", emitter },
  );
  checkWellFormed(text);
  const types = [...emitter.debug.types.values()];
  const scopes = [...emitter.debug.scopes.values()];
  const variables = [...emitter.debug.variables.values()];
  const unit = scopes.find((record) => record.kind === "CompileUnit");
  assert.deepEqual(unit.info, { file: "debug.cc", directory: "", isCxx: true });
  const node = types.find(
    (record) =>
      record.kind === "Composite" &&
      record.info.name === "Node" &&
      record.info.elements.length,
  );
  assert.equal(node.info.kind, "Structure");
  assert.equal(node.info.sizeInBits, 160);
  assert.equal(node.info.alignInBits, 32);
  assert.deepEqual(node.info.location, {
    file: "debug.cc",
    line: 2,
    column: 8,
  });
  const members = node.info.elements.map((ref) => emitter.debug.types.get(ref));
  assert.deepEqual(
    members.map((record) => record.info.name),
    ["next", "value", "data"],
  );
  assert.deepEqual(
    members.map((record) => record.info.offsetInBits),
    [0, 32, 64],
  );
  const array = types.find((record) => record.kind === "Array");
  assert.equal(array.info.count, 3);
  assert.equal(array.info.countBitWidth, 32);
  assert.equal(array.info.sizeInBits, 96);
  assert.ok(
    types.some(
      (record) => record.kind === "Derived" && record.info.kind === "Const",
    ),
  );
  assert.ok(
    types.some(
      (record) => record.kind === "Derived" && record.info.kind === "Volatile",
    ),
  );
  const choice = types.find(
    (record) => record.kind === "Composite" && record.info.name === "Choice",
  );
  assert.equal(choice.info.kind, "Enumeration");
  assert.equal(choice.info.isScopedEnum, true);
  const memberPointer = types.find(
    (record) =>
      record.kind === "Derived" && record.info.kind === "MemberPointer",
  );
  assert.equal(
    emitter.debug.types.get(memberPointer.info.classType).info.name,
    "Node",
  );
  const method = scopes.find(
    (record) => record.kind === "Function" && record.info.name === "method",
  );
  assert.equal(emitter.debug.scopes.get(method.info.scope).kind, "Type");
  assert.equal(emitter.debug.types.get(method.info.type).info.types.length, 3);
  assert.ok(method.info.loc > 0);
  const objectParameter = variables.find((info) => info.name === "this");
  assert.equal(objectParameter.isObjectParameter, true);
  assert.equal(objectParameter.argument, 1);
  assert.equal(
    emitter.debug.scopes.get(objectParameter.scope).info.name,
    "method",
  );
  const arg = variables.find((info) => info.name === "arg");
  assert.equal(arg.argument, 2);
  const nested = variables.find((info) => info.name === "nested");
  const block = emitter.debug.scopes.get(nested.scope);
  assert.equal(block.kind, "Block");
  assert.equal(block.info.location.line, 11);
  assert.equal(
    emitter.debug.scopes.get(block.info.parent).info.name,
    "inspect",
  );
});

test("debug emission respects the parser debugInfo option", async () => {
  const emitter = new TraceEmitter();
  const text = await trace("int f(int value) { return value; }", {
    debugInfo: false,
    emitter,
  });
  assert.equal(emitter.debug.types.size, 0);
  assert.equal(emitter.debug.scopes.size, 0);
  assert.equal(emitter.debug.variables.size, 0);
  assert.doesNotMatch(text, /debug[.]/);
});

test("delegates can omit the optional debug emitter", async () => {
  const emitter = new TraceEmitter();
  const delegate = new Proxy(emitter, {
    get(target, property) {
      if (property === "debug") return undefined;
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  const text = await trace("int f(int value) { return value; }", {
    emitter: delegate,
  });
  checkWellFormed(text);
  assert.match(text, /func @_Z1fi/);
  assert.equal(emitter.debug.scopes.size, 0);
});

test("switch constants cross the generated delegate boundary as exact bigint values", async () => {
  class SwitchTrace extends TraceEmitter {
    caseValues = [];
    switchBranch(loc, flag, defaultDest, values, destinations) {
      this.caseValues.push(...values);
      super.switchBranch(loc, flag, defaultDest, values, destinations);
    }
  }
  const emitter = new SwitchTrace();
  const text = await trace(
    `
int classify(unsigned long long value) {
  switch (value) {
    case 9007199254740993ULL: return 1;
    case 18446744073709551615ULL: return 2;
    default: return 0;
  }
}
`,
    { emitter },
  );
  assert.deepEqual(emitter.caseValues, [9007199254740993n, -1n]);
  assert.match(text, /9007199254740993: \^bb\d+/);
});
