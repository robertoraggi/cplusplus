import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { openSync, closeSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { WASI } from "node:wasi";
import { loadCxx, Linker, LinkError, parse } from "cxx-frontend";
import link from "cxx-frontend/link";

const wasm = await readFile(
  new URL("../dist/wasm/cxx-js.wasm", import.meta.url),
);

await loadCxx({ wasm });

const sysroot = new URL(
  "../../../build.em/src/lib/wasi-sysroot",
  import.meta.url,
).pathname;

const skip = existsSync(sysroot) ? false : "the wasi sysroot is not built";

function countingReader() {
  const reader = async (file) => {
    reader.reads.push(file);
    try {
      return await readFile(file);
    } catch {
      return undefined;
    }
  };
  reader.reads = [];
  return reader;
}

async function compile(source) {
  await using parser = await parse({
    path: "/linker.cc",
    source,
    debugInfo: false,
  });
  assert.deepEqual(parser.diagnostics, []);
  return parser.emitCode({ format: "obj" });
}

function printing(text) {
  return `extern "C" int puts(const char*);
int main() { puts("${text}"); return 0; }`;
}

async function run(module) {
  const directory = await mkdtemp(path.join(tmpdir(), "cxx-linker-"));
  const output = path.join(directory, "stdout.txt");
  const fd = openSync(output, "w");

  try {
    const wasi = new WASI({
      version: "preview1",
      args: ["a.out"],
      stdout: fd,
      returnOnExit: true,
    });
    const compiled = await WebAssembly.compile(module);
    const instance = await WebAssembly.instantiate(
      compiled,
      wasi.getImportObject(),
    );
    const exitCode = wasi.start(instance);
    return { exitCode, stdout: readFileSync(output, "utf8") };
  } finally {
    closeSync(fd);
    await rm(directory, { recursive: true });
  }
}

test("links an object file into a runnable wasi module", { skip }, async () => {
  await using linker = await Linker.create({
    sysroot,
    readFile: countingReader(),
  });
  const module = await linker.link([
    { name: "main.o", data: await compile(printing("hello")) },
  ]);

  assert.deepEqual(await run(module), { exitCode: 0, stdout: "hello\n" });
});

test("reads the libraries once and reuses them", { skip }, async () => {
  const readFile = countingReader();
  await using linker = await Linker.create({ sysroot, readFile });
  const reads = readFile.reads.length;

  assert.deepEqual(readFile.reads.map((file) => path.basename(file)).sort(), [
    "crt1.o",
    "libc++.a",
    "libc++abi.a",
    "libc.a",
    "libclang_rt.builtins-wasm32.a",
  ]);

  for (const text of ["first", "second"]) {
    const module = await linker.link([{ data: await compile(printing(text)) }]);
    assert.equal((await run(module)).stdout, `${text}\n`);
  }

  assert.equal(readFile.reads.length, reads);
});

test("reports undefined symbols", { skip }, async () => {
  await using linker = await Linker.create({
    sysroot,
    readFile: countingReader(),
  });
  const object = await compile(
    'extern "C" int missing(); int main() { return missing(); }',
  );

  await assert.rejects(linker.link([{ data: object }]), (error) => {
    assert.ok(error instanceof LinkError);
    assert.match(error.message, /undefined symbol: missing/);
    return true;
  });

  const module = await linker.link([{ data: object }], {
    allowUndefined: true,
  });
  assert.ok(module.length > 0);
});

test("strips debug information and names", { skip }, async () => {
  await using linker = await Linker.create({
    sysroot,
    readFile: countingReader(),
  });
  const data = await compile(printing("strip"));
  const sizes = [];

  for (const strip of ["none", "debug", "all"]) {
    const module = await linker.link([{ data }], { strip });
    assert.equal((await run(module)).stdout, "strip\n");
    sizes.push(module.length);
  }

  assert.ok(sizes[0] > sizes[1], `${sizes}`);
  assert.ok(sizes[1] > sizes[2], `${sizes}`);
});

test("rejects after being disposed", { skip }, async () => {
  const linker = await Linker.create({ sysroot, readFile: countingReader() });
  const data = await compile(printing("disposed"));

  linker.dispose();

  assert.equal(linker.disposed, true);
  await assert.rejects(linker.link([{ data }]), /disposed/);
});

test("the one-shot link entry links", { skip }, async () => {
  const module = await link({
    sysroot,
    readFile: countingReader(),
    objects: [{ data: await compile(printing("one shot")) }],
  });

  assert.equal((await run(module)).stdout, "one shot\n");
});

test("fails when a library cannot be found", async () => {
  await assert.rejects(
    Linker.create({ sysroot: "/nonexistent", readFile: async () => undefined }),
    LinkError,
  );
});

test("an aborted signal stops the creation", { skip }, async () => {
  const controller = new AbortController();
  controller.abort(new Error("stop"));

  await assert.rejects(
    Linker.create({
      sysroot,
      readFile: countingReader(),
      signal: controller.signal,
    }),
    /stop/,
  );
});
