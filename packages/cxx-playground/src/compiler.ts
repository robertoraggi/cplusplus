import type { Diagnostic, OutputCodeFormat } from "cxx-frontend"

export interface CompileOptions {
  source: string
  path: string
  format: OutputCodeFormat
}

export interface CompileResult {
  diagnostics: Diagnostic[]
  output: string
}

export type CompilerRequest =
  | { id: number; type: "initialize" }
  | { id: number; type: "compile"; options: CompileOptions }

type CompilerRequestPayload =
  | { type: "initialize" }
  | { type: "compile"; options: CompileOptions }

export type CompilerResponse =
  | { id: number; result: CompileResult | undefined }
  | { id: number; error: string }

interface PendingRequest {
  resolve: (result: CompileResult | undefined) => void
  reject: (error: Error) => void
}

let compilerWorker: Worker | undefined
let initialization: Promise<void> | undefined
let nextRequestId = 0
const pendingRequests = new Map<number, PendingRequest>()

function getCompilerWorker(): Worker {
  if (compilerWorker) return compilerWorker

  compilerWorker = new Worker(new URL("./compiler.worker.ts", import.meta.url), {
    type: "module",
  })
  compilerWorker.onmessage = ({ data }: MessageEvent<CompilerResponse>) => {
    const pending = pendingRequests.get(data.id)
    if (!pending) return

    pendingRequests.delete(data.id)
    if ("error" in data) {
      pending.reject(new Error(data.error))
    } else {
      pending.resolve(data.result)
    }
  }
  compilerWorker.onerror = ({ message }) => {
    const error = new Error(message || "C++ compiler worker failed")
    for (const pending of pendingRequests.values()) pending.reject(error)
    pendingRequests.clear()
    compilerWorker?.terminate()
    compilerWorker = undefined
    initialization = undefined
  }

  return compilerWorker
}

function sendRequest(
  request: CompilerRequestPayload
): Promise<CompileResult | undefined> {
  const id = nextRequestId++
  const message: CompilerRequest =
    request.type === "initialize"
      ? { id, type: request.type }
      : { id, type: request.type, options: request.options }

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { resolve, reject })
    getCompilerWorker().postMessage(message)
  })
}

export function loadCompiler(): Promise<void> {
  initialization ??= sendRequest({ type: "initialize" })
    .then(() => undefined)
    .catch((error) => {
      compilerWorker?.terminate()
      compilerWorker = undefined
      initialization = undefined
      throw error
    })
  return initialization
}

export async function compile(options: CompileOptions): Promise<CompileResult> {
  const result = await sendRequest({ type: "compile", options })
  if (!result) throw new Error("C++ compiler worker returned no result")
  return result
}
