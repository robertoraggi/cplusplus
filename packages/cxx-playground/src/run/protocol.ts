export type RunCommand =
  | { kind: "start"; wasm: Uint8Array<ArrayBuffer>; args: string[] }
  | { kind: "input"; data: Uint8Array<ArrayBuffer> }
  | { kind: "eof" }

export type RunMessage =
  | { kind: "output"; data: Uint8Array<ArrayBuffer> }
  | { kind: "reading" }
  | { kind: "stdin-unavailable" }
  | { kind: "exit"; code: number }
  | { kind: "trap"; message: string }

export interface WorkerScope {
  addEventListener<T>(
    type: "message",
    listener: (event: { data: T }) => void
  ): void
  postMessage(message: unknown, transfer?: Transferable[]): void
}
