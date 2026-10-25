import loadCxx from "cxx-frontend/node";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  Parser,
  kernelInputData,
  kernelLaneCount,
  parseKernelReflection,
} from "cxx-frontend";

await loadCxx();

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
  assert.equal(manifest.cxx.format, "wgsl");
  assert.equal(parseKernelReflection(manifest).entryPoint, "saxpy");
  assert.deepEqual(parseKernelReflection(manifest).localSize, [64, 1, 1]);
  assert.deepEqual(
    parseKernelReflection(manifest).resources.map((r) => [
      r.name,
      r.binding,
      r.kind,
      r.layout.elementType,
      r.access,
    ]),
    [
      ["y", 0, "storage", "f32", "read_write"],
      ["x", 1, "storage", "f32", "read"],
      ["a", 2, "uniform", "f32", "read"],
      ["count", 3, "uniform", "i32", "read"],
    ],
  );
  assert.deepEqual(manifest.cxx.entries.saxpy.invocationIds, [
    { parameter: 4, kind: "global", components: 1 },
  ]);
  assert.equal(
    manifest.cxx.sha256,
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
  assert.equal(manifest.cxx.format, "spirv");
  assert.equal(
    manifest.cxx.sha256,
    createHash("sha256").update(code).digest("hex"),
  );
});

test("the target selects the SPIR-V environment", async () => {
  for (const [target, environment, version] of [
    ["spirv32-unknown-unknown", "vulkan1.1", 0x00010300],
    ["spirv32-unknown-vulkan1.2", "vulkan1.2", 0x00010500],
    ["spirv32-unknown-vulkan1.3", "vulkan1.3", 0x00010600],
    ["spirv32-unknown-webgpu", "webgpu", 0x00010300],
  ]) {
    await using parser = await Parser.parse({
      path: "/saxpy.cc",
      source: saxpy,
      target,
    });
    assert.deepEqual(parser.diagnostics, []);
    const { kernels, error } = parser.emitCode({ format: "spirv" });
    assert.equal(error, "");
    const [{ manifest, code }] = kernels;
    assert.equal(manifest.cxx.targetEnvironment, environment);
    assert.equal(
      new DataView(code.buffer, code.byteOffset).getUint32(4, true),
      version,
    );
  }
});

test("describes class and array buffers with their C++ layout", async () => {
  await using parser = await Parser.parse({
    path: "/structs.cc",
    source: `
      #define device __attribute__((address_space(1)))
      #define constant __attribute__((address_space(2)))
      using float3 = float __attribute__((ext_vector_type(3)));
      struct Particle {
        float3 position;
        float mass;
        int id;
        float history[3];
      };
      struct Step {
        float dt;
        float damping[2];
        struct { float3 gravity; unsigned count; } world;
      };
      [[kernel, cxx::workgroup_size(64, 1, 1)]] void integrate(
          [[cxx::buffer(0)]] device Particle* particles,
          [[cxx::buffer(1)]] constant Step& step,
          [[cxx::global_id]] unsigned i) {
        if (i >= step.world.count) return;
        particles[i].position += step.world.gravity * step.dt;
        particles[i].history[i % 3] = particles[i].mass * step.damping[i & 1];
        particles[i].id += 1;
      }
    `,
    ...device,
  });
  assert.deepEqual(parser.diagnostics, []);
  for (const [format, feature] of [
    ["wgsl", "uniform_buffer_standard_layout"],
    ["spirv", "uniformBufferStandardLayout"],
  ]) {
    const { kernels, error } = parser.emitCode({ format });
    assert.equal(error, "");
    const [{ manifest, code }] = kernels;
    assert.deepEqual(manifest.cxx.requiredFeatures, [feature]);
    if (format === "wgsl")
      assert.match(code, /^requires uniform_buffer_standard_layout;/);
    const [particles, step] = parseKernelReflection(manifest).resources;
    assert.equal(particles.arrayStride, 48);
    assert.deepEqual(
      particles.layout.members.map((m) => [m.offset, m.layout.kind]),
      [
        [0, "value"],
        [16, "value"],
        [20, "value"],
        [24, "array"],
      ],
    );
    assert.equal(particles.layout.members[3].layout.stride, 4);
    assert.equal(step.size, 48);
    assert.equal(step.layout.kind, "struct");
    assert.deepEqual(
      step.layout.members.map((m) => m.offset),
      [0, 4, 16],
    );
    assert.equal(kernelLaneCount(particles.layout), 8);
    const words = kernelInputData(
      particles,
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    );
    assert.ok(words instanceof Uint32Array);
    const floats = new Float32Array(words.buffer);
    const ints = new Int32Array(words.buffer);
    assert.deepEqual(Array.from(floats.subarray(0, 4)), [1, 2, 3, 0]);
    assert.equal(floats[4], 4);
    assert.equal(ints[5], 5);
    assert.deepEqual(Array.from(floats.subarray(6, 9)), [6, 7, 8]);
    assert.equal(floats[12], 9);
    assert.equal(ints[17], 13);
    const uniform = new Float32Array(
      kernelInputData(step, [0.5, 0.25, 0.75, 0, -2.5, 0, 16]).buffer,
    );
    assert.deepEqual(
      Array.from(uniform.subarray(0, 8)),
      [0.5, 0.25, 0.75, 0, 0, -2.5, 0, 0],
    );
    assert.equal(new Uint32Array(uniform.buffer)[8], 16);
  }
});

test("emits vector storage buffers with their aligned array strides", async () => {
  for (const [scalar, elementType] of [
    ["float", "f32"],
    ["int", "i32"],
    ["unsigned", "u32"],
  ])
    for (const [lanes, stride] of [
      [2, 8],
      [3, 16],
      [4, 16],
    ]) {
      await using parser = await Parser.parse({
        path: "/vector-storage.cc",
        source: `
          using V = ${scalar} __attribute__((ext_vector_type(${lanes})));
          using D = V __attribute__((address_space(1)));
          [[kernel]] void vectors([[cxx::global_id]] unsigned i,
                                 [[cxx::buffer(0)]] const D* input,
                                 [[cxx::buffer(1)]] D* output) {
            output[i] = input[i] + ${scalar}(1);
          }
        `,
        ...device,
      });
      assert.deepEqual(parser.diagnostics, []);
      for (const format of ["wgsl", "spirv"]) {
        const { kernels, error } = parser.emitCode({ format });
        assert.equal(error, "");
        assert.equal(kernels.length, 1);
        for (const resource of parseKernelReflection(kernels[0].manifest)
          .resources) {
          assert.equal(resource.layout.components, lanes);
          assert.equal(resource.arrayStride, stride);
          assert.equal(resource.layout.elementType, elementType);
        }
      }
    }
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
  assert.deepEqual(kernels, []);
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
  assert.doesNotMatch(gpu, /\bcxx\.(?:load|store|ptradd)\b/);
  const spirv = parser.emitCode({ format: "asm" });
  assert.match(spirv, /^spirv\.module @__spv__kernels Logical GLSL450/);
  assert.match(spirv, /spirv\.EntryPoint "GLCompute" @saxpy/);
  const object = parser.emitCode({ format: "obj" });
  assert.equal(Buffer.from(object).readUInt32LE(0), 0x07230203);
  assert.equal(parser.emitCode({ format: "llvm" }), "");
});

test("keeps unnamed resources explicit without colliding with source names", async () => {
  await using parser = await Parser.parse({
    path: "/unnamed.cc",
    source: `
    using D = int __attribute__((address_space(1)));
    [[kernel]] void first([[cxx::buffer(0)]] D*, [[cxx::buffer(1)]] D* buffer) {}
    [[kernel]] void second([[cxx::buffer(0)]] D*, [[cxx::buffer(1)]] D* buffer_0) {}
    [[kernel]] void third([[cxx::buffer(0)]] D* buffer) {}
  `,
    ...device,
  });
  assert.deepEqual(parser.diagnostics, []);
  for (const format of ["wgsl", "spirv"]) {
    const { kernels, error } = parser.emitCode({ format });
    assert.equal(error, "");
    const { manifest } = kernels[0];
    for (const name of ["first", "second"]) {
      const entry = manifest.cxx.entries[name];
      assert.equal(entry.parameters[0].name, undefined);
      const layout = parseKernelReflection(manifest, name);
      assert.equal(layout.resources[0].name, "0:0");
      assert.equal(new Set(layout.resources.map((r) => r.name)).size, 2);
    }
  }
});

test("rejects ordinary device calls to kernel entries", async () => {
  await using parser = await Parser.parse({
    path: "/kernel-call.cc",
    source: `
    using D = int __attribute__((address_space(1)));
    [[kernel]] void dispatch_target([[cxx::buffer(0)]] D* output,
                                   [[cxx::global_id]] unsigned index) {
      output[index] = int(index);
    }
    void forward(D* output, unsigned index) { dispatch_target(output, index); }
    [[kernel]] void caller([[cxx::buffer(0)]] D* output,
                          [[cxx::global_id]] unsigned index) {
      forward(output, index);
    }
  `,
    ...device,
  });
  assert.deepEqual(parser.diagnostics, []);
  for (const format of ["wgsl", "spirv"]) {
    const { kernels, error } = parser.emitCode({ format });
    assert.deepEqual(kernels, []);
    assert.match(
      error,
      /device code cannot call a kernel entry point directly/,
    );
  }
});

test("reports missing module entries without producing a partial artifact", async () => {
  await using parser = await Parser.parse({
    path: "/empty.cc",
    source: "void helper() {}",
    ...device,
  });
  const { kernels, error } = parser.emitCode({ format: "spirv" });
  assert.deepEqual(kernels, []);
  assert.match(error, /requires a \[\[kernel\]\] entry point/);
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
    assert.equal(parseKernelReflection(manifest).entryPoint, "saxpy");
    assert.equal(manifest.cxx.format, "wgsl");
    assert.deepEqual(
      parseKernelReflection(manifest).resources.map((r) => r.name),
      ["y", "x", "a", "count"],
    );
    assert.equal(
      createHash("sha256").update(code).digest("hex"),
      manifest.cxx.sha256,
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
      spirv.kernels[0].manifest.cxx.sha256,
    );

    await configure({});
    await scheduled();
    assert.match(latestDiagnostics()[0].message, /device profile/);
  } finally {
    server.dispose();
  }
});

test("preserves workgroup allocations and atomic helper calls", async () => {
  await using parser = await Parser.parse({
    path: "/shared-atomic.cc",
    ...device,
    source: `
      using SharedInt = int __attribute__((address_space(3)));
      using DeviceInt = int __attribute__((address_space(1)));

      static int ticket(SharedInt& count) {
        return __atomic_fetch_add(&count, 1, 0);
      }

      [[cxx::kernel, cxx::workgroup_size(32, 1, 1)]]
      void shared_counter([[cxx::buffer(0)]] DeviceInt* output,
                          [[cxx::global_id]] unsigned index,
                          [[cxx::local_id]] unsigned lane) {
        SharedInt count;
        if (lane == 0) count = 0;
        __builtin_cxx_control_barrier(__MEMORY_SCOPE_WRKGRP, __MEMORY_SCOPE_WRKGRP, __ATOMIC_ACQ_REL, 1u << 3);
        output[index] = ticket(count);
      }
    `,
  });
  assert.deepEqual(parser.diagnostics, []);
  const wgsl = parser.emitCode({ format: "wgsl" });
  assert.equal(wgsl.error, "");
  assert.equal(wgsl.kernels.length, 1);
  const code = wgsl.kernels[0].code;
  assert.match(code, /requires unrestricted_pointer_parameters;/);
  assert.match(code, /var<workgroup> .*atomic<i32>/);
  assert.match(code, /ptr<workgroup, atomic<i32>>/);
  assert.match(code, /atomicAdd\(/);
  assert.match(code, /workgroupBarrier\(\);/);
  const spirv = parser.emitCode({ format: "spirv" });
  assert.equal(spirv.error, "");
  assert.equal(spirv.kernels.length, 1);
  assert.equal(
    new DataView(
      spirv.kernels[0].code.buffer,
      spirv.kernels[0].code.byteOffset,
    ).getUint32(0, true),
    0x07230203,
  );
});
