import wasmBinaryUrl from "cxx-frontend/dist/cxx-js.wasm?url";
import { Parser } from "cxx-frontend";

export class CxxFrontendClient {
  async load(signal?: AbortSignal) {
    const response = await fetch(wasmBinaryUrl, { signal });
    if (!response.ok) throw new Error("failed to load cxx-js.wasm");
    if (signal?.aborted) throw new Error("aborted");
    const data = await response.arrayBuffer();
    const wasmBinary = new Uint8Array(data);
    await Parser.init({ wasmBinary });
    return true;
  }
}
