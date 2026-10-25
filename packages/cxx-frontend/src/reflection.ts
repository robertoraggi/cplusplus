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

import type {
  KernelDataLayout,
  KernelElementType,
  KernelLayout,
  KernelResource,
  KernelStructLayout,
} from "./Kernel.js";

export type * from "./Kernel.js";

type SourceElementType = KernelElementType | "struct" | "array";

const sourceElementTypes = new Set<string>([
  "f32",
  "i32",
  "u32",
  "struct",
  "array",
]);

function object(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    throw new Error(`${name}: expected an object`);
  return value as Record<string, unknown>;
}

function array(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${name}: expected an array`);
  return value;
}

function integer(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0)
    throw new Error(`${name}: expected a nonnegative safe integer`);
  return value;
}

function text(value: unknown, name: string): string {
  if (typeof value !== "string" || !value.length)
    throw new Error(`${name}: expected a nonempty string`);
  return value;
}

function sourceElementType(value: unknown, name: string): SourceElementType {
  if (typeof value !== "string" || !sourceElementTypes.has(value))
    throw new Error(`${name}: expected f32, i32, u32, struct or array`);
  return value as SourceElementType;
}

function valueLayout(type: unknown): KernelDataLayout | undefined {
  switch (type) {
    case "float":
      return { kind: "value", elementType: "f32", components: 1 };
    case "int":
      return { kind: "value", elementType: "i32", components: 1 };
    case "uint":
      return { kind: "value", elementType: "u32", components: 1 };
  }
  const match = typeof type === "string" && /^(i|u)?vec([234])$/.exec(type);
  if (!match) return undefined;
  const elementType =
    match[1] === "i" ? "i32" : match[1] === "u" ? "u32" : "f32";
  return {
    kind: "value",
    elementType,
    components: Number(match[2]) as 2 | 3 | 4,
  };
}

/**
 * The byte alignment of a buffer layout.
 *
 * @category Kernels
 */
export function kernelLayoutAlignment(layout: KernelDataLayout): number {
  switch (layout.kind) {
    case "value":
      return kernelLayoutSize(layout);
    case "array":
      return kernelLayoutAlignment(layout.element);
    case "struct":
      return Math.max(
        ...layout.members.map((member) => kernelLayoutAlignment(member.layout)),
      );
  }
}

/**
 * The size in bytes of one object of a buffer layout.
 *
 * @category Kernels
 */
export function kernelLayoutSize(layout: KernelDataLayout): number {
  switch (layout.kind) {
    case "value":
      return layout.components === 3 ? 16 : layout.components * 4;
    case "array":
      return layout.length * layout.stride;
    case "struct":
      return layout.size;
  }
}

/**
 * The number of scalar lanes in one object of a buffer layout, in the order
 * `kernelInputData` packs them.
 *
 * @category Kernels
 */
export function kernelLaneCount(layout: KernelDataLayout): number {
  switch (layout.kind) {
    case "value":
      return layout.components;
    case "array":
      return layout.length * kernelLaneCount(layout.element);
    case "struct":
      return layout.members.reduce(
        (lanes, member) => lanes + kernelLaneCount(member.layout),
        0,
      );
  }
}

function layoutExtent(layout: KernelDataLayout): number {
  switch (layout.kind) {
    case "value":
      return layout.components * 4;
    case "array":
      return (layout.length - 1) * layout.stride + layoutExtent(layout.element);
    case "struct": {
      const last = layout.members[layout.members.length - 1]!;
      return last.offset + layoutExtent(last.layout);
    }
  }
}

function checkAligned(layout: KernelDataLayout, offset: number, name: string) {
  if (offset % kernelLayoutAlignment(layout))
    throw new Error(`${name}: misaligned member offset`);
}

class LayoutReader {
  readonly #types: Record<string, unknown>;
  readonly #structs = new Map<string, KernelStructLayout>();
  readonly #active = new Set<string>();

  constructor(types: Record<string, unknown>) {
    this.#types = types;
  }

  members(typeId: string): {
    name: string;
    layout: KernelDataLayout;
    offset: number;
    dimensions: number[];
    stride?: number;
  }[] {
    if (!Object.hasOwn(this.#types, typeId))
      throw new Error(`missing type '${typeId}'`);
    const type = object(this.#types[typeId], typeId);
    return array(type.members, `${typeId}.members`).map((value, index) => {
      const member = object(value, `${typeId}.member`);
      const name =
        typeof member.name === "string" && member.name.length
          ? member.name
          : `_m${index}`;
      const offset = integer(member.offset, `${typeId}.${name}.offset`);
      const layout = valueLayout(member.type) ?? this.struct(member.type);
      checkAligned(layout, offset, `${typeId}.${name}`);
      if (member.array === undefined)
        return { name, layout, offset, dimensions: [] };
      const dimensions = array(member.array, `${name}.array`).map((size) =>
        integer(size, `${name}.array`),
      );
      const literal = array(
        member.array_size_is_literal,
        `${name}.array_size_is_literal`,
      );
      if (
        literal.length !== dimensions.length ||
        literal.some((flag) => flag !== true)
      )
        throw new Error(`${name}: specialized array sizes are unsupported`);
      const stride = integer(member.array_stride, `${name}.array_stride`);
      return { name, layout, offset, dimensions, stride };
    });
  }

  struct(type: unknown): KernelStructLayout {
    const typeId = typeof type === "string" ? type : "";
    const known = this.#structs.get(typeId);
    if (known) return known;
    if (!Object.hasOwn(this.#types, typeId))
      throw new Error(`unsupported buffer member type '${String(type)}'`);
    if (this.#active.has(typeId))
      throw new Error(`${typeId}: recursive buffer type`);
    this.#active.add(typeId);
    const members = this.members(typeId).map((member) => ({
      name: member.name,
      offset: member.offset,
      layout: arrayLayout(
        member.layout,
        member.dimensions,
        member.stride,
        member.name,
      ),
    }));
    this.#active.delete(typeId);
    if (!members.length) throw new Error(`${typeId}: empty buffer type`);
    const name = object(this.#types[typeId], typeId).name;
    const layout: KernelStructLayout = {
      kind: "struct",
      name: typeof name === "string" ? name : typeId,
      size: 0,
      members,
    };
    const last = members[members.length - 1]!;
    const alignment = kernelLayoutAlignment(layout);
    layout.size =
      Math.ceil((last.offset + kernelLayoutSize(last.layout)) / alignment) *
      alignment;
    this.#structs.set(typeId, layout);
    return layout;
  }
}

function arrayLayout(
  element: KernelDataLayout,
  dimensions: number[],
  stride: number | undefined,
  name: string,
): KernelDataLayout {
  if (!dimensions.length) return element;
  const strides = [stride!];
  for (const length of dimensions.slice(1).reverse()) {
    const outer = strides[0]!;
    if (!length || outer % length)
      throw new Error(`${name}: invalid array stride`);
    strides.unshift(outer / length);
  }
  let layout = element;
  dimensions.forEach((length, index) => {
    if (!length)
      throw new Error(
        `${name}: runtime-sized arrays are only valid as the outermost storage buffer dimension`,
      );
    const step = strides[index]!;
    if (step < kernelLayoutSize(layout) || step % kernelLayoutAlignment(layout))
      throw new Error(`${name}: invalid array stride`);
    layout = { kind: "array", element: layout, length, stride: step };
  });
  return layout;
}

function bufferResource(
  value: unknown,
  kind: "storage" | "uniform",
  reader: LayoutReader,
  source: SourceElementType | undefined,
): KernelResource {
  const buffer = object(value, "buffer");
  const name = text(buffer.name, "buffer.name");
  const typeId = text(buffer.type, `${name}.type`);
  if (buffer.array !== undefined)
    throw new Error(`${name}: descriptor arrays are unsupported`);
  if (buffer.readonly !== undefined && typeof buffer.readonly !== "boolean")
    throw new Error(`${name}.readonly: expected a boolean`);
  const members = reader.members(typeId);
  const blockSize = integer(buffer.block_size, `${name}.block_size`);
  const common = {
    name,
    set: integer(buffer.set, `${name}.set`),
    binding: integer(buffer.binding, `${name}.binding`),
  };
  if (kind === "uniform") {
    const struct = source === "struct" || members.length !== 1;
    const member = members[0]!;
    const layout = struct
      ? reader.struct(typeId)
      : arrayLayout(member.layout, member.dimensions, member.stride, name);
    const offset = struct ? 0 : member.offset;
    if (blockSize < offset + layoutExtent(layout))
      throw new Error(`${name}: truncated uniform block`);
    const alignment = kernelLayoutAlignment(layout);
    const size = Math.ceil(blockSize / alignment) * alignment;
    if (!Number.isSafeInteger(size))
      throw new Error(`${name}: uniform block is too large`);
    return {
      ...common,
      kind,
      layout,
      memberOffset: offset,
      access: "read",
      size,
    };
  }

  if (members.length !== 1)
    throw new Error(`${name}: expected a single runtime-sized array member`);
  const member = members[0]!;
  const dimensions = member.dimensions;
  if (!dimensions.length || dimensions[dimensions.length - 1] !== 0)
    throw new Error(`${name}: expected a runtime-sized array`);
  const arrayStride = member.stride!;
  const inner = dimensions.slice(0, -1);
  const count = inner.reduce((total, length) => total * length, 1);
  if (!count || arrayStride % count)
    throw new Error(`${name}: invalid array stride`);
  const outermost = inner[inner.length - 1];
  const layout = arrayLayout(
    member.layout,
    inner,
    outermost === undefined ? undefined : arrayStride / outermost,
    name,
  );
  if (
    arrayStride < kernelLayoutSize(layout) ||
    arrayStride % kernelLayoutAlignment(layout)
  )
    throw new Error(`${name}: invalid array stride`);
  if (blockSize !== member.offset)
    throw new Error(`${name}: invalid runtime array block size`);
  return {
    ...common,
    kind,
    layout,
    arrayStride,
    memberOffset: member.offset,
    access: buffer.readonly ? "read" : "read_write",
  };
}

interface SourceResource {
  name?: string | undefined;
  elementType?: SourceElementType | undefined;
}

function checkedSourceLayout(
  resource: KernelResource,
  elementType: SourceElementType,
): KernelDataLayout {
  const layout = resource.layout;
  const matches =
    layout.kind === "value"
      ? (elementType === "f32") === (layout.elementType === "f32")
      : elementType === layout.kind;
  if (!matches)
    throw new Error(
      `${resource.name}: source element type differs from reflection`,
    );
  if (layout.kind !== "value") return layout;
  return { ...layout, elementType: elementType as KernelElementType };
}

function sourceResources(
  metadata: Record<string, unknown>,
  entryPoint: string,
): Map<string, SourceResource> {
  const entries = object(metadata.entries, "cxx.entries");
  if (!Object.hasOwn(entries, entryPoint))
    throw new Error(`missing source contract for entry '${entryPoint}'`);
  const entry = object(entries[entryPoint], entryPoint);
  const sources = new Map<string, SourceResource>();
  const parameters = new Set<number>();
  for (const value of array(entry.parameters, `${entryPoint}.parameters`)) {
    const parameter = object(value, "source parameter");
    const index = integer(parameter.parameter, "parameter index");
    const resource = text(parameter.resource, "parameter resource");
    const name =
      parameter.name === undefined
        ? undefined
        : text(parameter.name, "parameter name");
    const elementType = sourceElementType(
      parameter.elementType,
      `${resource}.elementType`,
    );
    if (parameters.has(index) || sources.has(resource))
      throw new Error(`${entryPoint}: duplicate source parameter`);
    parameters.add(index);
    sources.set(resource, { name, elementType });
  }
  return sources;
}

function resourceAliases(
  reflection: Record<string, unknown>,
  entryPoint: string,
  resourceNames: readonly string[] | undefined,
): Map<string, SourceResource> | undefined {
  const aliases =
    reflection.cxx === undefined
      ? undefined
      : sourceResources(object(reflection.cxx, "cxx"), entryPoint);
  if (resourceNames === undefined) return aliases;
  const selected = new Map<string, SourceResource>();
  for (const value of array(resourceNames, "resource names")) {
    const name = text(value, "resource name");
    if (selected.has(name))
      throw new Error(`duplicate resource name '${name}'`);
    if (aliases && !aliases.has(name))
      throw new Error(
        `${entryPoint}: resource selection differs from source contract`,
      );
    selected.set(name, aliases ? aliases.get(name)! : { name });
  }
  if (aliases && selected.size !== aliases.size)
    throw new Error(
      `${entryPoint}: resource selection differs from source contract`,
    );
  return selected;
}

export function parseKernelReflection(
  input: unknown,
  entryPoint?: string,
  resourceNames?: readonly string[],
): KernelLayout {
  const reflection = object(
    typeof input === "string" ? JSON.parse(input) : input,
    "reflection",
  );
  const entries = array(reflection.entryPoints, "entryPoints").map((value) =>
    object(value, "entry point"),
  );
  const candidates = entries.filter((entry) => entry.mode === "comp");
  if (entryPoint === undefined && candidates.length !== 1)
    throw new Error(
      `select a compute entry point (${candidates.map((entry) => entry.name).join(", ")})`,
    );
  const matches =
    entryPoint === undefined
      ? candidates
      : candidates.filter((entry) => entry.name === entryPoint);
  if (matches.length !== 1)
    throw new Error(`expected one compute entry point '${entryPoint}'`);
  const entry = matches[0]!;
  const name = text(entry.name, "entry point name");
  const workgroup = array(entry.workgroup_size, `${name}.workgroup_size`);
  if (workgroup.length !== 3)
    throw new Error(`${name}: expected three workgroup dimensions`);
  const localSize = workgroup.map((size) =>
    integer(size, "workgroup dimension"),
  ) as [number, number, number];
  if (localSize.some((size) => size === 0))
    throw new Error(`${name}: workgroup dimensions must be positive`);
  if (entry.workgroup_size_is_spec_constant_id !== undefined) {
    const flags = array(
      entry.workgroup_size_is_spec_constant_id,
      "workgroup specialization flags",
    );
    if (flags.length !== 3 || flags.some((flag) => flag !== false))
      throw new Error(`${name}: specialized workgroup sizes are unsupported`);
  }

  for (const category of [
    "textures",
    "separate_images",
    "separate_samplers",
    "images",
    "push_constants",
    "subpass_inputs",
    "acceleration_structures",
    "atomic_counters",
  ])
    if (
      reflection[category] !== undefined &&
      array(reflection[category], category).length
    )
      throw new Error(`unsupported shader resources: ${category}`);

  const types =
    reflection.types === undefined ? {} : object(reflection.types, "types");
  const reader = new LayoutReader(types);
  const aliases = resourceAliases(reflection, name, resourceNames);
  if (candidates.length > 1 && aliases === undefined)
    throw new Error(
      `${name}: select resource names because raw reflection has no entry ownership`,
    );
  const found = new Set<string>();
  const resources: KernelResource[] = [];
  const bindings = new Set<string>();
  const names = new Set<string>();
  for (const [category, kind] of [
    ["ssbos", "storage"],
    ["ubos", "uniform"],
  ] as const) {
    if (reflection[category] === undefined) continue;
    for (const value of array(reflection[category], category)) {
      const physicalName = text(object(value, "buffer").name, "buffer name");
      if (aliases && !aliases.has(physicalName)) continue;
      if (found.has(physicalName))
        throw new Error(`duplicate resource name '${physicalName}'`);
      found.add(physicalName);
      const source = aliases?.get(physicalName);
      const resource = bufferResource(value, kind, reader, source?.elementType);
      if (aliases)
        resource.name = source?.name ?? `${resource.set}:${resource.binding}`;
      if (source?.elementType !== undefined)
        resource.layout = checkedSourceLayout(resource, source.elementType);
      const binding = `${resource.set}:${resource.binding}`;
      if (bindings.has(binding))
        throw new Error(`duplicate descriptor binding ${binding}`);
      if (names.has(resource.name))
        throw new Error(`duplicate resource name '${resource.name}'`);
      bindings.add(binding);
      names.add(resource.name);
      resources.push(resource);
    }
  }
  if (aliases)
    for (const resource of aliases.keys())
      if (!found.has(resource))
        throw new Error(`missing resource '${resource}'`);
  resources.sort((a, b) => a.set - b.set || a.binding - b.binding);
  return { entryPoint: name, localSize, resources };
}

export function kernelBufferSize(
  resource: KernelResource,
  elements = 1,
): number {
  integer(elements, "element count");
  const size =
    resource.kind === "uniform"
      ? resource.size
      : resource.memberOffset + Math.max(1, elements) * resource.arrayStride;
  if (!Number.isSafeInteger(size))
    throw new Error(`${resource.name}: buffer is too large`);
  return size;
}

function leafElementTypes(
  layout: KernelDataLayout,
  types: Set<KernelElementType>,
): Set<KernelElementType> {
  switch (layout.kind) {
    case "value":
      types.add(layout.elementType);
      break;
    case "array":
      leafElementTypes(layout.element, types);
      break;
    case "struct":
      for (const member of layout.members)
        leafElementTypes(member.layout, types);
  }
  return types;
}

/**
 * Packs `values` into the bytes of a buffer.
 *
 * @category Kernels
 */
export function kernelInputData(
  resource: KernelResource,
  values: ArrayLike<number>,
): Float32Array | Int32Array | Uint32Array {
  integer(values.length, "input length");
  const lanes = kernelLaneCount(resource.layout);
  if (resource.kind === "uniform" && values.length > lanes)
    throw new Error(`${resource.name}: too many uniform values`);
  const elements =
    resource.kind === "uniform" ? 1 : Math.ceil(values.length / lanes);
  const words = new Uint32Array(kernelBufferSize(resource, elements) / 4);
  const views = {
    f32: new Float32Array(words.buffer),
    i32: new Int32Array(words.buffer),
    u32: words,
  };
  let next = 0;
  const pack = (layout: KernelDataLayout, offset: number) => {
    switch (layout.kind) {
      case "value":
        for (let lane = 0; lane < layout.components; ++lane) {
          if (next === values.length) return;
          views[layout.elementType][offset / 4 + lane] = values[next++]!;
        }
        return;
      case "array":
        for (let index = 0; index < layout.length; ++index)
          pack(layout.element, offset + index * layout.stride);
        return;
      case "struct":
        for (const member of layout.members)
          pack(member.layout, offset + member.offset);
    }
  };
  const stride = resource.kind === "storage" ? resource.arrayStride : 0;
  for (let element = 0; element < elements; ++element)
    pack(resource.layout, resource.memberOffset + element * stride);
  const types = leafElementTypes(resource.layout, new Set());
  return types.size === 1 ? views[[...types][0]!] : words;
}

export function kernelWorkgroupCount(
  layout: KernelLayout,
  invocations: readonly number[],
): [number, number, number] {
  if (!Array.isArray(invocations))
    throw new Error("expected one to three invocation counts");
  if (invocations.length < 1 || invocations.length > 3)
    throw new Error("expected one to three invocation counts");
  return layout.localSize.map((size, i) =>
    Math.ceil(integer(invocations[i] ?? 1, "invocation count") / size),
  ) as [number, number, number];
}
