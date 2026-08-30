import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, Parser } from "../dist/index.js";

const wasm = await readFile(
  new URL("../dist/wasm/cxx-js.wasm", import.meta.url),
);

await loadCxx({ wasm });

const files = new Map([
  ["/quote/quote-only.h", "#define QUOTE_HEADER 1\n"],
  ["/user/user-only.h", "#define USER_HEADER 1\n"],
  ["/system/system-only.h", "#define SYSTEM_HEADER 1\n"],
]);

const source = `
#ifndef FLAG
#error FLAG is not defined
#endif

#if FLAG != 1
#error FLAG does not default to 1
#endif

#if VALUE != 42
#error VALUE has the wrong value
#endif

#if TWICE(3) != 6
#error function-like macro has the wrong value
#endif

#if __cplusplus != 201402L
#error __cplusplus was not overridden
#endif

#ifdef __wasm__
#error __wasm__ was not undefined
#endif

#include "quote-only.h"
#include <user-only.h>
#include <system-only.h>

static_assert(QUOTE_HEADER == 1);
static_assert(USER_HEADER == 1);
static_assert(SYSTEM_HEADER == 1);

int answer() { return VALUE; }
`;

const resolvers = {
  exists: (path) => files.has(path),
  readFile: async (path) => files.get(path),
};

test("Parser.parse configures preprocessing and debug information", async () => {
  const parser = await Parser.parse({
    source,
    path: "/source/main.cc",
    defines: ["FLAG", "VALUE=42", "TWICE(x)=((x) * 2)", "__cplusplus=201402L"],
    undefines: ["__wasm__", "__cplusplus"],
    quoteIncludePaths: ["/quote"],
    includePaths: ["/user"],
    systemIncludePaths: ["/system"],
    debugInfo: false,
    ...resolvers,
  });

  try {
    assert.deepEqual(parser.diagnostics, []);
    assert.doesNotMatch(parser.emitCode({ format: "mlir" }), /\bloc\(/);
  } finally {
    parser.dispose();
  }

  const parserWithDebugInfo = await Parser.parse({
    source: "int answer() { return 42; }",
    path: "/source/debug.cc",
  });

  try {
    assert.deepEqual(parserWithDebugInfo.diagnostics, []);
    assert.match(parserWithDebugInfo.emitCode({ format: "mlir" }), /\bloc\(/);
  } finally {
    parserWithDebugInfo.dispose();
  }
});

test("Parser.emitCode returns object code as bytes", async () => {
  const parser = await Parser.parse({
    source: "int answer() { return 42; }",
    path: "/source/obj.cc",
  });

  try {
    const text = parser.emitCode({ format: "asm" });
    assert.equal(typeof text, "string");
    assert.match(text, /answer/);

    const bytes = parser.emitCode({ format: "obj" });
    assert.ok(bytes instanceof Uint8Array);
    assert.deepEqual([...bytes.subarray(0, 4)], [0x00, 0x61, 0x73, 0x6d]);
  } finally {
    parser.dispose();
  }
});

test("diagnostics preserve severity, group notes, and gate codegen", async () => {
  const parserWithWarning = await Parser.parse({
    source: '#warning "keep compiling"\nint answer() { return 42; }',
    path: "/source/warning.cc",
  });

  try {
    assert.equal(parserWithWarning.diagnostics.length, 1);
    assert.equal(parserWithWarning.diagnostics[0].severity, "warning");
    assert.deepEqual(parserWithWarning.diagnostics[0].notes, []);
    assert.notEqual(parserWithWarning.emitCode({ format: "mlir" }), "");
  } finally {
    parserWithWarning.dispose();
  }

  const parserWithError = await Parser.parse({
    source: `
void select(int);
void select(long);
void test() { select("wrong"); }
`,
    path: "/source/error.cc",
  });

  try {
    assert.equal(parserWithError.diagnostics.length, 1);
    assert.equal(parserWithError.diagnostics[0].severity, "error");
    assert.equal(parserWithError.diagnostics[0].notes.length, 2);
    assert.equal(parserWithError.emitCode({ format: "mlir" }), "");
  } finally {
    parserWithError.dispose();
  }
});

test("the std option selects the value of __cplusplus", async () => {
  const expected = new Map([
    ["c++14", "201402L"],
    ["c++17", "201703L"],
    ["c++20", "202002L"],
    ["c++23", "202302L"],
    ["c++26", "202400L"],
  ]);

  for (const [std, value] of expected) {
    const parser = await Parser.parse({
      source: `
#if __cplusplus != ${value}
#error __cplusplus does not match the requested standard
#endif
int answer() { return 42; }
`,
      path: `/source/${std}.cc`,
      std,
    });

    try {
      assert.deepEqual(
        parser.diagnostics,
        [],
        `unexpected diagnostics for ${std}`,
      );
    } finally {
      parser.dispose();
    }
  }
});

test("the std option composes with user defines and undefines", async () => {
  const parser = await Parser.parse({
    source: `
#ifndef FLAG
#error FLAG is not defined
#endif
#ifdef __wasm__
#error __wasm__ was not undefined
#endif
#if __cplusplus != 202302L
#error __cplusplus does not match the requested standard
#endif
int answer() { return 42; }
`,
    path: "/source/compose.cc",
    std: "c++23",
    defines: ["FLAG"],
    undefines: ["__wasm__"],
  });

  try {
    assert.deepEqual(parser.diagnostics, []);
  } finally {
    parser.dispose();
  }
});

test("an aborted signal rejects Parser.parse", async () => {
  await assert.rejects(
    Parser.parse({
      source: "int answer() { return 42; }",
      path: "/source/aborted.cc",
      signal: AbortSignal.abort(),
    }),
    (error) => error.name === "AbortError",
  );
});

test("aborting while includes are resolved stops preprocessing early", async () => {
  const includeCount = 200;
  const controller = new AbortController();
  const requested = [];

  let source = "";
  for (let i = 0; i < includeCount; ++i) source += `#include "h${i}.h"\n`;
  source += "int answer() { return 42; }\n";

  await assert.rejects(
    Parser.parse({
      source,
      path: "/source/abort.cc",
      exists: () => true,
      readFile: async (path) => {
        requested.push(path);
        await new Promise((resolve) => setTimeout(resolve, 0));
        controller.abort();
        return "";
      },
      signal: controller.signal,
    }),
    (error) => error.name === "AbortError",
  );

  assert.ok(
    requested.length < includeCount / 2,
    `resolved ${requested.length} of ${includeCount} includes after aborting`,
  );
});

test("a signal that is never aborted does not affect parsing", async () => {
  const controller = new AbortController();

  const parser = await Parser.parse({
    source: "int answer() { return 42; }",
    path: "/source/not-aborted.cc",
    signal: controller.signal,
  });

  try {
    assert.deepEqual(parser.diagnostics, []);
    assert.ok(parser.ast);
  } finally {
    parser.dispose();
  }
});

test("a signal composes with resolved includes", async () => {
  const controller = new AbortController();

  const parser = await Parser.parse({
    source: `#include "a.h"
#include "b.h"
#include "c.h"
static_assert(A == 1);
static_assert(B == 2);
static_assert(C == 3);
int answer() { return A + B + C; }
`,
    path: "/source/includes.cc",
    exists: () => true,
    readFile: async (path) => {
      const name = path.slice(path.lastIndexOf("/") + 1, -2).toUpperCase();
      return `#define ${name} ${"ABC".indexOf(name) + 1}\n`;
    },
    signal: controller.signal,
  });

  try {
    assert.deepEqual(parser.diagnostics, []);
    assert.ok(parser.ast);
  } finally {
    parser.dispose();
  }
});

test("aborting during parsing stops a translation unit with no includes", async () => {
  let source = "";
  for (let i = 0; i < 20000; ++i) {
    source += `int f${i}(int x) { int a = x + ${i}; return a * 2; }\n`;
  }

  const started = performance.now();
  const parser = await Parser.parse({ source, path: "/source/big.cc" });
  const fullParseMs = performance.now() - started;
  parser.dispose();

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5);
  const abortStarted = performance.now();

  await assert.rejects(
    Parser.parse({
      source,
      path: "/source/big.cc",
      signal: controller.signal,
    }),
    (error) => error.name === "AbortError",
  );

  clearTimeout(timer);

  const abortedParseMs = performance.now() - abortStarted;

  assert.ok(
    abortedParseMs < fullParseMs / 2,
    `aborted parse took ${abortedParseMs.toFixed(0)}ms, full parse took ${fullParseMs.toFixed(0)}ms`,
  );
});
