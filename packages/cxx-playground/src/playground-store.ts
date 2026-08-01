import { useSyncExternalStore } from "react"
import * as monaco from "monaco-editor"
import wasmBinaryUrl from "cxx-frontend/wasm?url"
import { Parser, type Diagnostic } from "cxx-frontend"
import { registerMlirLanguage } from "./lib/mlir-language"

export interface SampleCode {
  id: string
  name: string
  code: string
}

const semaFiles = import.meta.glob<string>(
  "../../../tests/unit_tests/sema/*.cc",
  { query: "?raw", import: "default", eager: true }
)

export const samples: SampleCode[] = Object.entries(semaFiles)
  .map(([filePath, code]) => {
    const id = filePath.split("/").pop()!
    return { id, name: id.replace(/_/g, " "), code }
  })
  .sort((a, b) => a.name.localeCompare(b.name))

const defaultSample =
  samples.find((s) => s.id === "auto_template.cc") ?? samples[0]

registerMlirLanguage()

export const inputModel = monaco.editor.createModel(
  defaultSample?.code ?? "",
  "cpp",
  monaco.Uri.parse("file:///main.cc")
)

export const outputModel = monaco.editor.createModel(
  "",
  "mlir",
  monaco.Uri.parse("file:///main.mlir")
)

export type OutputFormat = "cxxir" | "mlir" | "llvm" | "asm"

interface Snapshot {
  isReady: boolean
  isCompiling: boolean
  diagnosticCount: number
  compileTimeMs: number | null
  currentSampleId: string
  outputFormat: OutputFormat
}

let snapshot: Snapshot = {
  isReady: false,
  isCompiling: false,
  diagnosticCount: 0,
  compileTimeMs: null,
  currentSampleId: defaultSample?.id ?? "",
  outputFormat: "cxxir",
}

const listeners = new Set<() => void>()

function publish(patch: Partial<Snapshot>) {
  snapshot = { ...snapshot, ...patch }
  listeners.forEach((listener) => listener())
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function getSnapshot() {
  return snapshot
}

export function usePlayground() {
  return useSyncExternalStore(subscribe, getSnapshot)
}

export function setOutputFormat(format: OutputFormat) {
  if (snapshot.outputFormat === format) return
  publish({ outputFormat: format })
  const languageMap: Record<OutputFormat, string> = {
    cxxir: "mlir",
    mlir: "mlir",
    llvm: "llvm",
    asm: "asm",
  }
  monaco.editor.setModelLanguage(outputModel, languageMap[format])
  if (snapshot.isReady) {
    compile()
  }
}

function applyDiagnostics(diagnostics: Diagnostic[]) {
  const markers: monaco.editor.IMarkerData[] = diagnostics.map(
    ({ startLine, startColumn, endLine, endColumn, message }) => ({
      severity: monaco.MarkerSeverity.Error,
      startLineNumber: startLine,
      startColumn,
      endLineNumber: endLine,
      endColumn,
      message,
      source: "C++ Compiler",
    })
  )
  monaco.editor.setModelMarkers(inputModel, "cxx", markers)
}

let isCompiling = false
let hasPendingCompile = false

async function compile() {
  if (isCompiling) {
    hasPendingCompile = true
    return
  }

  isCompiling = true
  publish({ isCompiling: true })

  try {
    do {
      hasPendingCompile = false
      const startTime = performance.now()
      const source = inputModel.getValue()
      const path = snapshot.currentSampleId || "main.cc"
      const parser = new Parser({ source, path })

      try {
        await parser.parse()
        const diagnostics = parser.getDiagnostics()
        applyDiagnostics(diagnostics)

        let output: string
        try {
          output = await parser.emitCode({ format: snapshot.outputFormat })
        } catch (error) {
          output = `// ${snapshot.outputFormat.toUpperCase()} codegen failed: ${(error as Error).message}`
        }

        outputModel.setValue(output)
        publish({
          diagnosticCount: diagnostics.length,
          compileTimeMs: Math.round((performance.now() - startTime) * 10) / 10,
        })
      } finally {
        parser.dispose()
      }
    } while (hasPendingCompile)
  } finally {
    isCompiling = false
    publish({ isCompiling: false })
  }
}

let debounceId: ReturnType<typeof setTimeout>
inputModel.onDidChangeContent(() => {
  clearTimeout(debounceId)
  debounceId = setTimeout(compile, 250)
})

export function loadSample(id: string) {
  const sample = samples.find((s) => s.id === id)
  if (!sample) return
  publish({ currentSampleId: id })
  inputModel.setValue(sample.code)
}

async function init() {
  const response = await fetch(wasmBinaryUrl)
  const wasm = await response.arrayBuffer()
  await Parser.init({ wasm })
  publish({ isReady: true })
  compile()
}

init()
