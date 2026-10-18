import type * as monaco from "monaco-editor"
import type { KernelManifest } from "cxx-frontend"
import { LspClient } from "./lib/lsp-client"
import { inputCodeModel } from "./input-code-model"
import LanguageServerWorker from "./lsp.worker?worker"

export const Device = ["cpu", "gpu"] as const
export type Device = (typeof Device)[number]

export const TextOutputCodeFormat = [
  "cxxir",
  "mlir",
  "llvm",
  "asm",
  "wgsl",
] as const
export type TextOutputCodeFormat = (typeof TextOutputCodeFormat)[number]

export const outputFormatsByDevice: Record<Device, TextOutputCodeFormat[]> = {
  cpu: ["cxxir", "mlir", "llvm", "asm"],
  gpu: ["cxxir", "mlir", "asm", "wgsl"],
}

const deviceSettings: Record<Device, Record<string, string>> = {
  cpu: {},
  gpu: { target: "spirv32-unknown-unknown" },
}

export interface EmittedKernel {
  manifest: KernelManifest
  code: string
}

interface EmitCodeResult {
  format: string
  text: string
  error?: string
  kernels?: EmittedKernel[]
}

export interface KernelResult {
  kernels: EmittedKernel[]
  error: string
}

let client: LspClient | undefined

export function startLanguageServer(): LspClient {
  client ??= new LspClient(new LanguageServerWorker())
  return client
}

export async function configureDevice(device: Device): Promise<void> {
  await startLanguageServer().sendNotification(
    "workspace/didChangeConfiguration",
    { settings: { cxx: deviceSettings[device] } }
  )
}

export async function emitKernels({
  debugInfo,
  optimizationLevel,
}: {
  debugInfo: boolean
  optimizationLevel: number
}): Promise<KernelResult> {
  const result = await requestEmitCode({
    format: "wgsl",
    debugInfo,
    optimizationLevel,
  })
  return { kernels: result?.kernels ?? [], error: result?.error ?? "" }
}

export type ExecutableResult =
  | { kind: "built"; wasm: Uint8Array<ArrayBuffer> }
  | { kind: "failed"; message: string }

async function requestEmitCode(params: {
  format: string
  debugInfo: boolean
  mlirPipeline?: string
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
  mlirPipeline,
  optimizationLevel,
}: {
  format: TextOutputCodeFormat
  debugInfo: boolean
  mlirPipeline?: string
  optimizationLevel: number
}): Promise<string> {
  const result = await requestEmitCode({
    format,
    debugInfo,
    mlirPipeline,
    optimizationLevel,
  })
  if (!result?.error) return result?.text ?? ""
  const error = result.error
    .trimEnd()
    .split("\n")
    .map((line) => `// ${line}`)
    .join("\n")
  return result.text ? `${result.text}${error}\n` : `${error}\n`
}

export async function emitExecutable({
  debugInfo,
  mlirPipeline,
  optimizationLevel,
}: {
  debugInfo: boolean
  mlirPipeline?: string
  optimizationLevel: number
}): Promise<ExecutableResult> {
  const result = await requestEmitCode({
    format: "wasm",
    debugInfo,
    mlirPipeline,
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
