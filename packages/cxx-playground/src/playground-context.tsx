/* eslint-disable react-refresh/only-export-components */
import * as React from "react"
import * as monaco from "monaco-editor"
import type { OutputCodeFormat } from "cxx-frontend"
import { loadCompiler, compile } from "./compiler"
import { applyDiagnostics } from "./diagnostics"
import { inputCodeModel } from "./input-code-model"
import { outputCodeModel } from "./output-code-model"
import { defaultSample, samples } from "./samples"

const outputLanguageByFormat: Record<OutputCodeFormat, string> = {
  cxxir: "mlir",
  mlir: "mlir",
  llvm: "llvm",
  asm: "asm",
}

interface PlaygroundContextValue {
  isReady: boolean
  isCompiling: boolean
  diagnosticCount: number
  compileTimeMs: number | null
  currentSampleId: string
  outputFormat: OutputCodeFormat
  loadSample: (id: string) => void
  setOutputFormat: (format: OutputCodeFormat) => void
}

const PlaygroundContext = React.createContext<PlaygroundContextValue | null>(
  null
)

export function usePlayground() {
  const context = React.useContext(PlaygroundContext)
  if (!context) {
    throw new Error("usePlayground must be used within a PlaygroundProvider")
  }
  return context
}

export function PlaygroundProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [isReady, setIsReady] = React.useState(false)
  const [isCompiling, setIsCompiling] = React.useState(false)
  const [diagnosticCount, setDiagnosticCount] = React.useState(0)
  const [compileTimeMs, setCompileTimeMs] = React.useState<number | null>(null)
  const [currentSampleId, setCurrentSampleId] = React.useState(
    defaultSample?.id ?? ""
  )
  const [outputFormat, setOutputFormatState] =
    React.useState<OutputCodeFormat>("cxxir")

  const isCompilingRef = React.useRef(false)
  const pendingCompileRef = React.useRef(false)
  const compileRevisionRef = React.useRef(0)
  const currentSampleIdRef = React.useRef(currentSampleId)
  const outputFormatRef = React.useRef(outputFormat)
  const compileTimeoutRef = React.useRef<ReturnType<typeof setTimeout>>(null)

  const runCompile = React.useCallback(async () => {
    if (isCompilingRef.current) {
      pendingCompileRef.current = true
      return
    }

    isCompilingRef.current = true
    setIsCompiling(true)

    try {
      do {
        pendingCompileRef.current = false
        const compileRevision = compileRevisionRef.current
        const startTime = performance.now()

        const { diagnostics, output } = await compile({
          source: inputCodeModel.getValue(),
          path: currentSampleIdRef.current || "main.cc",
          format: outputFormatRef.current,
        })

        if (compileRevision === compileRevisionRef.current) {
          applyDiagnostics(inputCodeModel, diagnostics)
          outputCodeModel.setValue(output)
          setDiagnosticCount(diagnostics.length)
          setCompileTimeMs(
            Math.round((performance.now() - startTime) * 10) / 10
          )
        }
      } while (pendingCompileRef.current)
    } finally {
      isCompilingRef.current = false
      setIsCompiling(false)
    }
  }, [])

  const scheduleCompile = React.useCallback(
    (debounce: boolean) => {
      compileRevisionRef.current += 1
      if (compileTimeoutRef.current) clearTimeout(compileTimeoutRef.current)

      if (debounce) {
        pendingCompileRef.current = false
        compileTimeoutRef.current = setTimeout(() => {
          compileTimeoutRef.current = null
          runCompile()
        }, 250)
      } else {
        compileTimeoutRef.current = null
        runCompile()
      }
    },
    [runCompile]
  )

  React.useEffect(() => {
    let cancelled = false

    loadCompiler().then(() => {
      if (!cancelled) setIsReady(true)
    })

    return () => {
      cancelled = true
    }
  }, [])

  React.useEffect(() => {
    if (!isReady) return
    const disposable = inputCodeModel.onDidChangeContent(() => {
      scheduleCompile(true)
    })

    return () => {
      if (compileTimeoutRef.current) clearTimeout(compileTimeoutRef.current)
      disposable.dispose()
    }
  }, [isReady, scheduleCompile])

  React.useEffect(() => {
    if (isReady) scheduleCompile(false)
  }, [isReady, scheduleCompile])

  const loadSample = React.useCallback(
    (id: string) => {
      const sample = samples.find((s) => s.id === id)
      if (!sample) return
      currentSampleIdRef.current = id
      setCurrentSampleId(id)
      inputCodeModel.setValue(sample.code)
      scheduleCompile(false)
    },
    [scheduleCompile]
  )

  const setOutputFormat = React.useCallback(
    (format: OutputCodeFormat) => {
      if (outputFormatRef.current === format) return
      outputFormatRef.current = format
      setOutputFormatState(format)
      monaco.editor.setModelLanguage(
        outputCodeModel,
        outputLanguageByFormat[format]
      )
      scheduleCompile(false)
    },
    [scheduleCompile]
  )

  const value = React.useMemo<PlaygroundContextValue>(
    () => ({
      isReady,
      isCompiling,
      diagnosticCount,
      compileTimeMs,
      currentSampleId,
      outputFormat,
      loadSample,
      setOutputFormat,
    }),
    [
      isReady,
      isCompiling,
      diagnosticCount,
      compileTimeMs,
      currentSampleId,
      outputFormat,
      loadSample,
      setOutputFormat,
    ]
  )

  return (
    <PlaygroundContext.Provider value={value}>
      {children}
    </PlaygroundContext.Provider>
  )
}
