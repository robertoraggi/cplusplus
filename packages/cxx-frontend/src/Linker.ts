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

import { cxx } from "./cxx.js";
import { type LinkInput } from "./cxx-js.js";
import { isCxxLoaded } from "./loadCxx.js";
import { asyncDisposeSymbol, disposeSymbol } from "./disposeSymbols.js";

/** @category Linking */
export type StripMode = "none" | "debug" | "all";

/** @category Linking */
export interface LinkOptions {
  entry?: string;
  noEntry?: boolean;
  exports?: string[];
  allowUndefined?: boolean;
  gcSections?: boolean;
  stackSize?: number;
  stackFirst?: boolean;
  globalBase?: number;
  initialMemory?: number;
  maxMemory?: number;
  strip?: StripMode;
}

/** @category Linking */
export type ReadFile = (path: string) => Promise<Uint8Array | undefined>;

/** @category Linking */
export interface LinkerOptions extends LinkOptions {
  readFile: ReadFile;
  sysroot?: string;
  libraryPaths?: string[];
  startFiles?: string[];
  libraries?: string[];
  signal?: AbortSignal;
}

/** @category Linking */
export interface ObjectFile {
  name?: string;
  data: Uint8Array;
}

/** @category Linking */
export class LinkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LinkError";
  }
}

const defaultLibraryDirectory = "lib/wasm32-wasip1";
const defaultStartFiles = ["crt1.o"];
const defaultLibraries = ["c", "c++", "c++abi", "clang_rt.builtins-wasm32"];

interface Payloads {
  start: LinkInput[];
  libraries: LinkInput[];
}

/** @category Linking */
export class Linker implements Disposable, AsyncDisposable {
  #payloads: Payloads | undefined;
  readonly #defaults: LinkOptions;

  private constructor(payloads: Payloads, defaults: LinkOptions) {
    this.#payloads = payloads;
    this.#defaults = defaults;
  }

  static async create(options: LinkerOptions): Promise<Linker> {
    if (typeof options?.readFile !== "function") {
      throw new TypeError("expected parameter 'readFile' of type 'function'");
    }

    if (!isCxxLoaded()) {
      throw new Error(
        "the cxx wasm module is not loaded, call loadCxx() first",
      );
    }

    const { readFile, signal, ...rest } = options;
    const searchPaths = libraryPaths(rest);
    const startFiles = rest.startFiles ?? defaultStartFiles;
    const libraries = rest.libraries ?? defaultLibraries;

    signal?.throwIfAborted();

    const loaded: LinkInput[] = [];

    try {
      const pull = async (
        names: string[],
        resolve: (name: string) => string[],
      ): Promise<LinkInput[]> => {
        const inputs = await Promise.all(
          names.map(async (name) => {
            const input = await pullInput(readFile, name, resolve(name));
            loaded.push(input);
            return input;
          }),
        );
        signal?.throwIfAborted();
        return inputs;
      };

      const start = await pull(startFiles, (name) =>
        candidates(searchPaths, name),
      );
      const archives = await pull(libraries, (name) =>
        candidates(searchPaths, `lib${name}.a`),
      );

      return new Linker({ start, libraries: archives }, linkOptionsOf(rest));
    } catch (error) {
      for (const input of loaded) input.delete();
      throw error;
    }
  }

  get disposed(): boolean {
    return this.#payloads === undefined;
  }

  async link(
    objects: ObjectFile[],
    options: LinkOptions = {},
  ): Promise<Uint8Array> {
    return this.linkSync(objects, options);
  }

  linkSync(objects: ObjectFile[], options: LinkOptions = {}): Uint8Array {
    const payloads = this.#nativePayloads();
    const created: LinkInput[] = [];

    try {
      objects.forEach(({ name, data }, index) => {
        const input = new cxx.LinkInput(name ?? `input${index}.o`, data);
        created.push(input);
        const error = input.getError();
        if (error) throw new LinkError(error);
      });

      const { output, error } = cxx.link(
        [...payloads.start, ...created, ...payloads.libraries],
        { ...this.#defaults, ...options },
      );

      if (error) throw new LinkError(error);

      return output;
    } finally {
      for (const input of created) input.delete();
    }
  }

  dispose(): void {
    const payloads = this.#payloads;
    this.#payloads = undefined;

    if (!payloads) return;

    for (const input of [...payloads.start, ...payloads.libraries]) {
      input.delete();
    }
  }

  [disposeSymbol](): void {
    this.dispose();
  }

  async [asyncDisposeSymbol](): Promise<void> {
    this.dispose();
  }

  #nativePayloads(): Payloads {
    if (!this.#payloads) {
      throw new Error("Linker has been disposed");
    }

    return this.#payloads;
  }
}

function libraryPaths(options: Omit<LinkerOptions, "readFile">): string[] {
  if (options.libraryPaths) return options.libraryPaths;

  if (options.sysroot === undefined) {
    throw new TypeError(
      "expected one of the options 'sysroot' or 'libraryPaths'",
    );
  }

  return [`${options.sysroot}/${defaultLibraryDirectory}`];
}

function candidates(searchPaths: string[], name: string): string[] {
  if (name.includes("/")) return [name];
  return searchPaths.map((directory) => `${directory}/${name}`);
}

async function pullInput(
  readFile: ReadFile,
  name: string,
  paths: string[],
): Promise<LinkInput> {
  for (const path of paths) {
    const data = await readFile(path);

    if (data === undefined) continue;

    const input = new cxx.LinkInput(path, data);
    const error = input.getError();

    if (!error) return input;

    input.delete();
    throw new LinkError(error);
  }

  throw new LinkError(`unable to find ${name}`);
}

function linkOptionsOf(options: Omit<LinkerOptions, "readFile">): LinkOptions {
  const {
    sysroot: _sysroot,
    libraryPaths: _libraryPaths,
    startFiles: _startFiles,
    libraries: _libraries,
    ...linkOptions
  } = options;

  return linkOptions;
}
