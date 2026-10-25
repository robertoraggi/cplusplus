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
 * A scalar or a vector of `components` lanes.
 *
 * @category Kernels
 */
export interface KernelValueLayout {
  kind: "value";
  elementType: KernelElementType;
  components: 1 | 2 | 3 | 4;
}

/**
 * A fixed-size array whose elements are `stride` bytes apart.
 *
 * @category Kernels
 */
export interface KernelArrayLayout {
  kind: "array";
  element: KernelDataLayout;
  length: number;
  stride: number;
}

/** @category Kernels */
export interface KernelStructMember {
  name: string;
  offset: number;
  layout: KernelDataLayout;
}

/**
 * A class object laid out as in C++: members at their byte offsets, `size`
 * bytes including padding.
 *
 * @category Kernels
 */
export interface KernelStructLayout {
  kind: "struct";
  name: string;
  size: number;
  members: KernelStructMember[];
}

/**
 * The bytes of one buffer element, or of a uniform block.
 *
 * @category Kernels
 */
export type KernelDataLayout =
  KernelValueLayout | KernelArrayLayout | KernelStructLayout;

/**
 * A runtime-sized storage buffer parameter of a checked kernel: an array of
 * `layout` elements `arrayStride` bytes apart.
 *
 * @category Kernels
 */
export interface KernelStorageResource {
  name: string;
  set: number;
  binding: number;
  kind: "storage";
  layout: KernelDataLayout;
  arrayStride: number;
  memberOffset: number;
  access: "read" | "read_write";
}

/**
 * A uniform buffer parameter of a checked kernel: one `layout` object in a
 * block of `size` bytes.
 *
 * @category Kernels
 */
export interface KernelUniformResource {
  name: string;
  set: number;
  binding: number;
  kind: "uniform";
  layout: KernelDataLayout;
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

export interface KernelReflectionMember {
  name: string;
  type: string;
  offset?: number;
  array?: number[];
  array_size_is_literal?: boolean[];
  array_stride?: number;
}

export interface KernelReflectionType {
  name: string;
  members: KernelReflectionMember[];
}

export interface KernelReflectionBuffer {
  type: string;
  name: string;
  block_size: number;
  set: number;
  binding: number;
  readonly?: boolean;
}

export interface KernelReflectionEntry {
  name: string;
  mode: string;
  workgroup_size?: [number, number, number];
  workgroup_size_is_spec_constant_id?: [boolean, boolean, boolean];
}

export interface CxxKernelMetadata {
  format: "spirv" | "wgsl";
  profile: string;
  abiVersion: number;
  indexBits: number;
  entries: Record<
    string,
    {
      parameters: {
        parameter: number;
        name?: string;
        resource: string;
        elementType: KernelElementType | "struct" | "array";
      }[];
      invocationIds: KernelInvocationId[];
    }
  >;
  sha256: string;
  targetEnvironment: string;
  addressingModel?: string;
  memoryModel?: string;
  spirv?: { version: string; capabilities: string[]; extensions: string[] };
  requiredFeatures?: string[];
}

export interface KernelManifest {
  entryPoints: KernelReflectionEntry[];
  types?: Record<string, KernelReflectionType>;
  ssbos?: KernelReflectionBuffer[];
  ubos?: KernelReflectionBuffer[];
  cxx?: CxxKernelMetadata;
}

export interface KernelLayout {
  entryPoint: string;
  localSize: [number, number, number];
  resources: KernelResource[];
}

/** @category Kernels */
export interface KernelArtifact<Code> {
  manifest: KernelManifest;
  code: Code;
}

/**
 * The selected device module of a translation unit, with its diagnostics.
 *
 * @category Kernels
 */
export interface KernelOutput<Code> {
  kernels: KernelArtifact<Code>[];
  error: string;
}
