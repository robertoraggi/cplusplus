// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// <reference types="@webgpu/types" preserve="true" />

import type { KernelManifest, KernelResource } from "./Kernel.js";
import {
  kernelBufferSize,
  kernelWorkgroupCount,
  parseKernelReflection,
} from "./reflection.js";

export {
  kernelBufferSize,
  kernelInputData,
  kernelLaneCount,
  kernelLayoutAlignment,
  kernelLayoutSize,
  kernelWorkgroupCount,
  parseKernelReflection,
} from "./reflection.js";

export interface KernelPipelineOptions {
  code: string;
  reflection: unknown;
  entryPoint?: string;
  resourceNames?: readonly string[];
}

export type KernelBindings = Readonly<
  Record<string, GPUBuffer | GPUBufferBinding>
>;

function bindingType(resource: KernelResource): GPUBufferBindingType {
  if (resource.kind === "uniform") return "uniform";
  return resource.access === "read" ? "read-only-storage" : "storage";
}

async function checkArtifact(
  reflection: KernelManifest,
  code: string,
): Promise<void> {
  if (reflection.cxx?.format !== "wgsl") return;
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(code),
  );
  const hash = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  if (reflection.cxx.sha256 !== hash)
    throw new Error("WGSL does not match its reflection artifact hash");
}

export async function createKernelPipeline(
  device: GPUDevice,
  options: KernelPipelineOptions,
) {
  const reflection =
    typeof options.reflection === "string"
      ? JSON.parse(options.reflection)
      : options.reflection;
  const layout = parseKernelReflection(
    reflection,
    options.entryPoint,
    options.resourceNames,
  );
  await checkArtifact(reflection as KernelManifest, options.code);

  const groupCount = layout.resources.reduce(
    (count, resource) => Math.max(count, resource.set + 1),
    0,
  );
  if (groupCount > device.limits.maxBindGroups)
    throw new Error("descriptor sets exceed the device's bind group limit");
  const groups = Array.from({ length: groupCount }, (_, set) =>
    layout.resources.filter((resource) => resource.set === set),
  );

  device.pushErrorScope("validation");
  let compilation: Promise<GPUCompilationInfo>;
  let pending: Promise<GPUComputePipeline>;
  let validation: Promise<GPUError | null>;
  let bindGroupLayouts: GPUBindGroupLayout[];
  try {
    const module = device.createShaderModule({ code: options.code });
    compilation = module.getCompilationInfo();
    bindGroupLayouts = groups.map((resources) =>
      device.createBindGroupLayout({
        entries: resources.map((resource) => ({
          binding: resource.binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: bindingType(resource),
            minBindingSize: kernelBufferSize(resource),
          },
        })),
      }),
    );
    pending = device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts }),
      compute: { module, entryPoint: layout.entryPoint },
    });
  } finally {
    validation = device.popErrorScope();
  }

  const [info, result, failure] = await Promise.all([
    compilation,
    pending.then(
      (pipeline) => ({ pipeline }),
      (error: unknown) => ({ error }),
    ),
    validation,
  ]);
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length)
    throw new Error(
      errors
        .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`)
        .join("\n"),
    );
  if (failure) throw new Error(failure.message);
  if ("error" in result) throw result.error;

  function createBindGroups(bindings: KernelBindings): GPUBindGroup[] {
    return groups.map((resources, set) =>
      device.createBindGroup({
        layout: bindGroupLayouts[set]!,
        entries: resources.map((resource) => {
          if (!Object.hasOwn(bindings, resource.name))
            throw new Error(`missing buffer '${resource.name}'`);
          const value = bindings[resource.name]!;
          const binding = "buffer" in value ? value : { buffer: value };
          const size =
            binding.size ?? binding.buffer.size - (binding.offset ?? 0);
          if (size < kernelBufferSize(resource))
            throw new Error(`${resource.name}: buffer binding is too small`);
          return { binding: resource.binding, resource: binding };
        }),
      }),
    );
  }

  return {
    layout,
    pipeline: result.pipeline,
    createBindGroups,
    workgroupCount(invocations: readonly number[]) {
      return kernelWorkgroupCount(layout, invocations);
    },
  };
}
