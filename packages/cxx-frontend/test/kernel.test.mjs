import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, Parser } from "cxx-frontend";

await loadCxx({
  wasm: await readFile(new URL("../dist/wasm/cxx-js.wasm", import.meta.url)),
});

const device = {
  target: "spirv32-unknown-unknown",
};

const saxpy = `
  #define device __attribute__((address_space(1)))
  #define constant __attribute__((address_space(2)))

  [[kernel, cxx::workgroup_size(64, 1, 1)]] void saxpy(
      [[cxx::buffer(0)]] device float* y,
      [[cxx::buffer(1)]] const device float* x,
      [[cxx::buffer(2)]] constant float& a,
      [[cxx::buffer(3)]] constant int& count,
      [[cxx::global_id]] unsigned i) {
    if (i >= unsigned(count)) return;
    y[i] = a * x[i] + y[i];
  }
`;

test("emits every checked kernel as WGSL with its manifest", async () => {
  await using parser = await Parser.parse({
    path: "/saxpy.cc",
    source: saxpy,
    ...device,
  });
  assert.deepEqual(parser.diagnostics, []);
  const { kernels, error } = parser.emitCode({ format: "wgsl" });
  assert.equal(error, "");
  assert.equal(kernels.length, 1);
  const [{ manifest, code }] = kernels;
  assert.equal(manifest.format, "wgsl");
  assert.equal(manifest.entryPoint, "saxpy");
  assert.deepEqual(manifest.localSize, [64, 1, 1]);
  assert.deepEqual(
    manifest.resources.map((r) => [
      r.name,
      r.binding,
      r.kind,
      r.elementType,
      r.access,
    ]),
    [
      ["y", 0, "storage", "f32", "read_write"],
      ["x", 1, "storage", "f32", "read"],
      ["a", 2, "uniform", "f32", "read"],
      ["count", 3, "uniform", "i32", "read"],
    ],
  );
  assert.deepEqual(manifest.invocationIds, [
    { parameter: 4, kind: "global", components: 1 },
  ]);
  assert.equal(
    manifest.sha256,
    createHash("sha256").update(code).digest("hex"),
  );
  assert.match(code, /@compute @workgroup_size\(64, 1, 1\)\nfn saxpy\(/);
});

test("default and explicit device profiles emit identical kernels", async () => {
  await using implicit = await Parser.parse({
    path: "/saxpy.cc",
    source: saxpy,
    ...device,
  });
  await using explicit = await Parser.parse({
    path: "/saxpy.cc",
    source: saxpy,
    ...device,
    deviceProfile: "vulkan1.1-f32",
  });
  for (const format of ["wgsl", "spirv"])
    assert.deepEqual(
      implicit.emitCode({ format }),
      explicit.emitCode({ format }),
    );
});

test("emits SPIR-V binaries for the same kernels", async () => {
  await using parser = await Parser.parse({
    path: "/saxpy.cc",
    source: saxpy,
    ...device,
  });
  const { kernels, error } = parser.emitCode({ format: "spirv" });
  assert.equal(error, "");
  const [{ manifest, code }] = kernels;
  assert.ok(code instanceof Uint8Array);
  assert.equal(
    new DataView(code.buffer, code.byteOffset).getUint32(0, true),
    0x07230203,
  );
  assert.equal(manifest.format, "spirv");
  assert.equal(
    manifest.sha256,
    createHash("sha256").update(code).digest("hex"),
  );
});

test("reports device diagnostics for kernels that cannot be compiled", async () => {
  await using parser = await Parser.parse({
    path: "/call.cc",
    source:
      `
      using DeviceFloat = float __attribute__((address_space(1)));
      extern "C" void helper();
      extern "C" [[kernel, cxx::workgroup_size(8, 1, 1)]] void broken(
          [[cxx::buffer(0)]] DeviceFloat* out) { helper(); }
    ` + saxpy,
    ...device,
  });
  const { kernels, error } = parser.emitCode({ format: "wgsl" });
  assert.deepEqual(
    kernels.map((k) => k.manifest.entryPoint),
    ["saxpy"],
  );
  assert.match(
    error,
    /\/call\.cc:\d+:\d+: error: device function 'helper' has no definition/,
  );
});

test("emits the device stages for the generic formats", async () => {
  await using parser = await Parser.parse({
    path: "/saxpy.cc",
    source: saxpy,
    ...device,
  });
  const gpu = parser.emitCode({ format: "mlir" });
  assert.match(gpu, /^gpu\.module @kernels \{/);
  assert.match(gpu, /gpu\.func @saxpy\(/);
  assert.doesNotMatch(gpu, /cxx\./);
  const spirv = parser.emitCode({ format: "asm" });
  assert.match(spirv, /^spirv\.module @__spv__kernels Logical GLSL450/);
  assert.match(spirv, /spirv\.EntryPoint "GLCompute" @saxpy/);
  const object = parser.emitCode({ format: "obj" });
  assert.equal(Buffer.from(object).readUInt32LE(0), 0x07230203);
  assert.equal(parser.emitCode({ format: "llvm" }), "");
});

test("rejects a device profile without its target", async () => {
  await assert.rejects(
    Parser.parse({
      path: "/saxpy.cc",
      source: saxpy,
      deviceProfile: "vulkan1.1-f32",
    }),
    /requires a spirv32 target/,
  );
});

test("LanguageServer switches to the device target and emits kernels", async () => {
  const { LanguageServer } = await import("cxx-frontend/lsp");
  const messages = [];
  const server = await LanguageServer.start({
    onMessage: (message) => messages.push(message),
  });
  const uri = "file:///saxpy.cc";
  const scheduled = () => new Promise((resolve) => setTimeout(resolve, 800));
  const emit = async (id, format) => {
    await server.receive({
      jsonrpc: "2.0",
      id,
      method: "cxx/emitCode",
      params: { textDocument: { uri }, format },
    });
    return messages.find((message) => message.id === id).result;
  };
  const latestDiagnostics = () =>
    messages
      .filter((m) => m.method === "textDocument/publishDiagnostics")
      .at(-1).params.diagnostics;
  const configure = (cxx) =>
    server.receive({
      jsonrpc: "2.0",
      method: "workspace/didChangeConfiguration",
      params: { settings: { cxx } },
    });

  try {
    await server.receive({ jsonrpc: "2.0", id: 1, method: "initialize" });
    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didOpen",
      params: {
        textDocument: { uri, languageId: "cpp", version: 1, text: saxpy },
      },
    });
    await scheduled();
    assert.match(latestDiagnostics()[0].message, /device profile/);
    assert.deepEqual(await emit(2, "wgsl"), {
      format: "wgsl",
      text: "",
      kernels: [],
    });

    await configure(device);
    await scheduled();
    assert.deepEqual(latestDiagnostics(), []);

    const wgsl = await emit(3, "wgsl");
    assert.equal(wgsl.kernels.length, 1);
    const [{ manifest, code }] = wgsl.kernels;
    assert.equal(manifest.entryPoint, "saxpy");
    assert.equal(manifest.format, "wgsl");
    assert.deepEqual(
      manifest.resources.map((r) => r.name),
      ["y", "x", "a", "count"],
    );
    assert.equal(
      createHash("sha256").update(code).digest("hex"),
      manifest.sha256,
    );
    assert.match(code, /@compute @workgroup_size\(64, 1, 1\)/);

    const gpu = await emit(5, "mlir");
    assert.equal(gpu.format, "mlir");
    assert.match(gpu.text, /gpu\.func @saxpy\(/);
    assert.equal(gpu.error, undefined);
    const spirvModule = await emit(6, "asm");
    assert.match(spirvModule.text, /spirv\.func @saxpy\(\)/);
    const llvm = await emit(7, "llvm");
    assert.match(llvm.error, /llvm needs a host target/);
    const linked = await emit(8, "wasm");
    assert.match(linked.error, /linking needs a host target/);

    const spirv = await emit(4, "spirv");
    const bytes = Buffer.from(spirv.kernels[0].code, "base64");
    assert.equal(bytes.readUInt32LE(0), 0x07230203);
    assert.equal(
      createHash("sha256").update(bytes).digest("hex"),
      spirv.kernels[0].manifest.sha256,
    );

    await configure({});
    await scheduled();
    assert.match(latestDiagnostics()[0].message, /device profile/);
  } finally {
    server.dispose();
  }
});
