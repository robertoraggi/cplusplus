import wasmBinaryUrl from "cxx-frontend/wasm?url"
import { loadCxx, Parser } from "cxx-frontend"
import { appdir, exists, loadSysroot, readFile, sysroot } from "./sysroot"
import type {
  CompileOptions,
  CompileResult,
  CompilerRequest,
  CompilerResponse,
} from "./compiler"

interface CompilerWorkerScope {
  onmessage: ((event: MessageEvent<CompilerRequest>) => void) | null
  postMessage(message: CompilerResponse): void
}

const scope = globalThis as unknown as CompilerWorkerScope
const initialization = Promise.all([
  loadCxx({ wasmURL: wasmBinaryUrl }),
  loadSysroot(),
])

async function compile({
  source,
  path,
  format,
  debugInfo,
}: CompileOptions): Promise<CompileResult> {
  await initialization

  const parser = await Parser.parse({
    source,
    path,
    appdir,
    sysroot,
    std: "c++14",
    debugInfo,
    exists,
    readFile,
  })

  try {
    let output: string
    try {
      output = parser.emitCode({ format })
    } catch (error) {
      output = `// ${format.toUpperCase()} codegen failed: ${errorMessage(error)}`
    }

    return { diagnostics: parser.diagnostics, output }
  } finally {
    parser.dispose()
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

scope.onmessage = async ({ data }) => {
  try {
    const result =
      data.type === "initialize"
        ? await initialization.then(() => undefined)
        : await compile(data.options)
    scope.postMessage({ id: data.id, result })
  } catch (error) {
    scope.postMessage({ id: data.id, error: errorMessage(error) })
  }
}
