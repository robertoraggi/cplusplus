/* eslint-disable react-refresh/only-export-components */
import * as React from "react"
import * as monaco from "monaco-editor"
import {
  emitCode,
  startLanguageServer,
  type TextOutputCodeFormat,
} from "./language-server"
import { inputCodeModel } from "./input-code-model"
import { outputCodeModel } from "./output-code-model"
import { defaultSample, samples } from "./samples"
import { programRunner, type RunnerState } from "./program-runner"

const outputLanguageByFormat: Record<TextOutputCodeFormat, string> = {
  cxxir: "mlir",
  mlir: "mlir",
  llvm: "llvm",
  asm: "asm",
}

const emitCodeDebounceMs = 300

export type OutputTab = "output" | "console"

interface PlaygroundContextValue {
  isReady: boolean
  isCompiling: boolean
  diagnosticCount: number
  compileTimeMs: number | null
  currentSampleId: string
  outputFormat: TextOutputCodeFormat
  debugInfo: boolean
  optimize: boolean
  activeTab: OutputTab
  runnerState: RunnerState
  loadSample: (id: string) => void
  setOutputFormat: (format: TextOutputCodeFormat) => void
  setDebugInfo: (debugInfo: boolean) => void
  setOptimize: (optimize: boolean) => void
  setActiveTab: (tab: OutputTab) => void
  runProgram: () => void
  stopProgram: () => void
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
    React.useState<TextOutputCodeFormat>("cxxir")
  const [debugInfo, setDebugInfoState] = React.useState(false)
  const [optimize, setOptimizeState] = React.useState(false)
  const [activeTab, setActiveTab] = React.useState<OutputTab>("console")
  const runnerState = React.useSyncExternalStore(
    programRunner.subscribe,
    programRunner.getState
  )

  const isCompilingRef = React.useRef(false)
  const pendingCompileRef = React.useRef(false)
  const compileRevisionRef = React.useRef(0)
  const outputFormatRef = React.useRef(outputFormat)
  const debugInfoRef = React.useRef(debugInfo)
  const optimizeRef = React.useRef(optimize)
  const compileTimeoutRef = React.useRef<ReturnType<typeof setTimeout>>(null)

  const runEmitCode = React.useCallback(async () => {
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

        const output = await emitCode({
          format: outputFormatRef.current,
          debugInfo: debugInfoRef.current,
          optimizationLevel: optimizeRef.current ? 2 : 0,
        })

        setIsReady(true)

        if (compileRevision === compileRevisionRef.current) {
          outputCodeModel.setValue(output)
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

  const scheduleEmitCode = React.useCallback(
    (debounce: boolean) => {
      compileRevisionRef.current += 1
      if (compileTimeoutRef.current) clearTimeout(compileTimeoutRef.current)

      if (debounce) {
        pendingCompileRef.current = false
        compileTimeoutRef.current = setTimeout(() => {
          compileTimeoutRef.current = null
          runEmitCode()
        }, emitCodeDebounceMs)
      } else {
        compileTimeoutRef.current = null
        runEmitCode()
      }
    },
    [runEmitCode]
  )

  React.useEffect(() => {
    startLanguageServer()
    scheduleEmitCode(false)

    const disposable = inputCodeModel.onDidChangeContent(() => {
      scheduleEmitCode(true)
    })

    return () => {
      if (compileTimeoutRef.current) clearTimeout(compileTimeoutRef.current)
      disposable.dispose()
    }
  }, [scheduleEmitCode])

  React.useEffect(() => {
    const countDiagnostics = () => {
      setDiagnosticCount(
        monaco.editor.getModelMarkers({ resource: inputCodeModel.uri }).length
      )
    }

    countDiagnostics()

    const disposable = monaco.editor.onDidChangeMarkers((resources) => {
      const changed = resources.some(
        (resource) => resource.toString() === inputCodeModel.uri.toString()
      )
      if (changed) countDiagnostics()
    })

    return () => {
      disposable.dispose()
    }
  }, [])

  const loadSample = React.useCallback((id: string) => {
    const sample = samples.find((s) => s.id === id)
    if (!sample) return
    setCurrentSampleId(id)
    inputCodeModel.setValue(sample.code)
  }, [])

  const setOutputFormat = React.useCallback(
    (format: TextOutputCodeFormat) => {
      if (outputFormatRef.current === format) return
      outputFormatRef.current = format
      setOutputFormatState(format)
      monaco.editor.setModelLanguage(
        outputCodeModel,
        outputLanguageByFormat[format]
      )
      scheduleEmitCode(false)
    },
    [scheduleEmitCode]
  )

  const setDebugInfo = React.useCallback(
    (debugInfo: boolean) => {
      if (debugInfoRef.current === debugInfo) return
      debugInfoRef.current = debugInfo
      setDebugInfoState(debugInfo)
      scheduleEmitCode(false)
    },
    [scheduleEmitCode]
  )

  const setOptimize = React.useCallback(
    (optimize: boolean) => {
      if (optimizeRef.current === optimize) return
      optimizeRef.current = optimize
      setOptimizeState(optimize)
      scheduleEmitCode(false)
    },
    [scheduleEmitCode]
  )

  const runProgram = React.useCallback(() => {
    setActiveTab("console")
    void programRunner.run({
      debugInfo: debugInfoRef.current,
      optimizationLevel: optimizeRef.current ? 2 : 0,
    })
  }, [])

  const stopProgram = React.useCallback(() => {
    programRunner.stop()
  }, [])

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey)) return
      event.preventDefault()
      event.stopPropagation()
      if (programRunner.getState() === "idle") runProgram()
    }

    window.addEventListener("keydown", onKeyDown, { capture: true })

    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true })
    }
  }, [runProgram])

  const value = React.useMemo<PlaygroundContextValue>(
    () => ({
      isReady,
      isCompiling,
      diagnosticCount,
      compileTimeMs,
      currentSampleId,
      outputFormat,
      debugInfo,
      optimize,
      activeTab,
      runnerState,
      loadSample,
      setOutputFormat,
      setDebugInfo,
      setOptimize,
      setActiveTab,
      runProgram,
      stopProgram,
    }),
    [
      isReady,
      isCompiling,
      diagnosticCount,
      compileTimeMs,
      currentSampleId,
      outputFormat,
      debugInfo,
      optimize,
      activeTab,
      runnerState,
      loadSample,
      setOutputFormat,
      setDebugInfo,
      setOptimize,
      runProgram,
      stopProgram,
    ]
  )

  return (
    <PlaygroundContext.Provider value={value}>
      {children}
    </PlaygroundContext.Provider>
  )
}
