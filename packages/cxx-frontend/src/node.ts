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

/**
 * Node.js initialization of the bundled wasm module.
 *
 * @module cxx-frontend/node
 */

// @ts-ignore: node typings are intentionally not a dependency
import { readFile } from "node:fs/promises";
import { loadCxx as loadCxxFromBytes } from "./loadCxx.js";

/**
 * Loads the cxx wasm module bundled with this package.
 *
 * Node.js only. Must be awaited before `Parser` is used.
 *
 * Safe to call multiple times, see the `loadCxx` of `cxx-frontend`.
 */
export async function loadCxx(): Promise<void> {
  const wasm = await readFile(new URL("./wasm/cxx-js.wasm", import.meta.url));
  await loadCxxFromBytes({ wasm });
}

export default loadCxx;
