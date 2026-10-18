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

/** @category Kernels */
export type KernelElementType = "f32" | "i32" | "u32";

/**
 * A runtime-sized storage buffer parameter of a checked kernel.
 *
 * @category Kernels
 */
export interface KernelStorageResource {
  parameter: number;
  name: string;
  set: number;
  binding: number;
  kind: "storage";
  elementType: KernelElementType;
  components: 1;
  storageClass: "StorageBuffer";
  arrayStride: number;
  memberOffset: number;
  access: "read" | "read_write";
}

/**
 * A uniform buffer parameter of a checked kernel: one scalar or vector of
 * `components` lanes.
 *
 * @category Kernels
 */
export interface KernelUniformResource {
  parameter: number;
  name: string;
  set: number;
  binding: number;
  kind: "uniform";
  elementType: KernelElementType;
  components: 1 | 2 | 3 | 4;
  storageClass: "Uniform";
  size: number;
  memberOffset: number;
  access: "read";
}

/**
 * One buffer parameter of a checked kernel.
 *
 * @category Kernels
 */
export type KernelResource = KernelStorageResource | KernelUniformResource;

export interface KernelInvocationId {
  parameter: number;
  kind: "global" | "local" | "workgroup";
  components: 1 | 2 | 3;
}

/**
 * The resource manifest Cxx emits next to a kernel artifact.
 *
 * @category Kernels
 */
export interface KernelManifest {
  schemaVersion: 3;
  producer: "cxx";
  format: "spirv" | "wgsl";
  entryPoint: string;
  profile: string;
  abiVersion: number;
  indexBits: number;
  localSize: [number, number, number];
  resources: KernelResource[];
  invocationIds: KernelInvocationId[];
  sha256: string;
  targetEnvironment: string;
}

/** @category Kernels */
export interface KernelArtifact<Code> {
  manifest: KernelManifest;
  code: Code;
}

/**
 * Every checked kernel of a translation unit, with the device diagnostics of
 * the kernels that could not be compiled.
 *
 * @category Kernels
 */
export interface KernelOutput<Code> {
  kernels: KernelArtifact<Code>[];
  error: string;
}
