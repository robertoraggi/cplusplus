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
