import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, Parser, TraceEmitter } from "../dist/index.js";

const wasm = await readFile(
  new URL("../dist/wasm/cxx-js.wasm", import.meta.url),
);

await loadCxx({ wasm });

async function trace(source, { path = "trace.cc" } = {}) {
  const parser = await Parser.parse({ path, source });
  try {
    const errors = parser.diagnostics.filter(
      (d) => d.severity === "error" || d.severity === "fatal",
    );
    assert.deepEqual(errors, [], "the source must compile cleanly");
    const emitter = new TraceEmitter();
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
  assert.match(text, /AddInt %\d+, %\d+ : i32/);
  assert.match(text, /^\s*return %\d+$/m);
  assert.match(text, /^endmodule$/m);
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
