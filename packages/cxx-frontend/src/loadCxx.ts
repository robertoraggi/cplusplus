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

import { cxx, instantiateCxx } from "./cxx.js";

/**
 * The wasm binary of the cxx frontend.
 */
export type WasmSource = Uint8Array | ArrayBuffer | WebAssembly.Module;

export interface LoadCxxOptions {
  /**
   * Raw wasm bytes or a precompiled module.
   */
  wasm?: WasmSource;

  /**
   * Fetch the wasm binary from this URL instead of passing bytes directly.
   */
  wasmURL?: string | URL;

  /**
   * Abort the underlying fetch when `wasmURL` is used.
   */
  signal?: AbortSignal;
}

let loading: Promise<void> | undefined;

/**
 * Loads and instantiates the cxx wasm module.
 *
 * Must be called, and its promise awaited, before `Parser` is used.
 *
 * Safe to call multiple times: subsequent calls return the same in-flight or
 * settled promise. The module is a process-wide singleton, a second, different
 * wasm binary cannot be loaded in the same JS realm.
 *
 * @param options the location of the wasm binary.
 */
export function loadCxx(options: LoadCxxOptions): Promise<void> {
  loading ??= load(options).catch((error) => {
    loading = undefined;
    throw error;
  });

  return loading;
}

/**
 * Returns true if `loadCxx` has already resolved.
 */
export function isCxxLoaded(): boolean {
  return cxx !== undefined && cxx !== null;
}

async function load(options: LoadCxxOptions): Promise<void> {
  await instantiateCxx(await getWasmSource(options));
}

async function getWasmSource({
  wasm,
  wasmURL,
  signal,
}: LoadCxxOptions): Promise<WasmSource> {
  if (wasm !== undefined) {
    return wasm;
  }

  if (wasmURL === undefined) {
    throw new TypeError("expected one of the options 'wasm' or 'wasmURL'");
  }

  return await fetchWasm(wasmURL, signal);
}

async function fetchWasm(
  url: string | URL,
  signal: AbortSignal | undefined,
): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal: signal ?? null });

  if (!response.ok) {
    throw new Error(`failed to fetch '${url}'`);
  }

  return await response.arrayBuffer();
}
