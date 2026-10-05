import type * as monaco from "monaco-editor"
import { LspClient } from "./lib/lsp-client"
import { inputCodeModel } from "./input-code-model"
import LanguageServerWorker from "./lsp.worker?worker"

export const TextOutputCodeFormat = ["cxxir", "mlir", "llvm", "asm"] as const
export type TextOutputCodeFormat = (typeof TextOutputCodeFormat)[number]

interface EmitCodeResult {
  format: string
  text: string
  error?: string
}

let client: LspClient | undefined

export function startLanguageServer(): LspClient {
  client ??= new LspClient(new LanguageServerWorker())
  return client
}

export type ExecutableResult =
  | { kind: "built"; wasm: Uint8Array<ArrayBuffer> }
  | { kind: "failed"; message: string }

async function requestEmitCode(params: {
  format: string
  debugInfo: boolean
  optimizationLevel: number
}): Promise<EmitCodeResult | null> {
  const client = startLanguageServer()
  const uri = documentUri(inputCodeModel)
  await client.whenDocumentOpened(uri)

  return await client.sendRequest<EmitCodeResult | null>("cxx/emitCode", {
    textDocument: { uri },
    ...params,
  })
}

export async function emitCode({
  format,
  debugInfo,
  optimizationLevel,
}: {
  format: TextOutputCodeFormat
  debugInfo: boolean
  optimizationLevel: number
}): Promise<string> {
  const result = await requestEmitCode({ format, debugInfo, optimizationLevel })
  return result?.text ?? ""
}

export async function emitExecutable({
  debugInfo,
  optimizationLevel,
}: {
  debugInfo: boolean
  optimizationLevel: number
}): Promise<ExecutableResult> {
  const result = await requestEmitCode({
    format: "wasm",
    debugInfo,
    optimizationLevel,
  })

  if (!result) {
    return { kind: "failed", message: "this build cannot link programs" }
  }

  if (result.error) return { kind: "failed", message: result.error }

  if (!result.text) return { kind: "failed", message: "" }

  return { kind: "built", wasm: decodeBase64(result.text) }
}

function decodeBase64(text: string): Uint8Array<ArrayBuffer> {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; ++i) bytes[i] = binary.charCodeAt(i)
  return bytes
}

function documentUri(model: monaco.editor.ITextModel): string {
  return model.uri.toString(true).toLowerCase()
}
