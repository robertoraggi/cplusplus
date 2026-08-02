import wasmBinaryUrl from "cxx-frontend/wasm?url"
import {
  loadCxx,
  Parser,
  type Diagnostic,
  type OutputCodeFormat,
} from "cxx-frontend"

export function loadCompiler(): Promise<void> {
  return loadCxx({ wasmURL: wasmBinaryUrl })
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
  await using parser = await Parser.parse({ source, path })

  let output: string
  try {
    output = parser.emitCode({ format })
  } catch (error) {
    output = `// ${format.toUpperCase()} codegen failed: ${(error as Error).message}`
  }

  return { diagnostics: parser.diagnostics, output }
}
