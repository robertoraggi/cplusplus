import wasmBinaryUrl from "cxx-frontend/wasm?url"
import { Parser, type Diagnostic, type OutputCodeFormat } from "cxx-frontend"

let ready: Promise<void> | null = null

export function loadCompiler(): Promise<void> {
  if (!ready) {
    ready = (async () => {
      const response = await fetch(wasmBinaryUrl)
      const wasm = await response.arrayBuffer()
      await Parser.init({ wasm })
    })()
  }
  return ready
}

export interface CompileResult {
  diagnostics: Diagnostic[]
  output: string
}

export async function compile({
  source,
  path,
  format,
}: {
  source: string
  path: string
  format: OutputCodeFormat
}): Promise<CompileResult> {
  const parser = new Parser({ source, path })
  try {
    await parser.parse()
    const diagnostics = parser.getDiagnostics()

    let output: string
    try {
      output = await parser.emitCode({ format })
    } catch (error) {
      output = `// ${format.toUpperCase()} codegen failed: ${(error as Error).message}`
    }

    return { diagnostics, output }
  } finally {
    parser.dispose()
  }
}
