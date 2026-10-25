/* eslint-disable react-refresh/only-export-components */
import * as React from "react"
import * as monaco from "monaco-editor"
import {
  configureDevice,
  emitCode,
  emitKernels,
  outputFormatsByDevice,
  startLanguageServer,
  type Device,
  type KernelResult,
  type TextOutputCodeFormat,
} from "./language-server"
import { inputCodeModel } from "./input-code-model"
import { outputCodeModel } from "./output-code-model"
import { defaultSampleByDevice, samplesByDevice } from "./samples"
import { shaderCanvas } from "./gpu/shader-canvas"
import { programRunner, type RunnerState } from "./program-runner"
import {
  buildMLIRPipeline,
  defaultMLIRPasses,
  type MLIRPass,
} from "./mlir-pipeline"

const outputLanguageByFormat: Record<TextOutputCodeFormat, string> = {
  cxxir: "mlir",
  mlir: "mlir",
  llvm: "llvm",
  asm: "asm",
  wgsl: "wgsl",
}

const deviceOutputLanguage: Record<
  Device,
  Partial<Record<TextOutputCodeFormat, string>>
> = {
  cpu: {},
  gpu: { asm: "mlir" },
}

function outputLanguageOf(device: Device, format: TextOutputCodeFormat) {
  return deviceOutputLanguage[device][format] ?? outputLanguageByFormat[format]
}

const equivalentFormat: Partial<
  Record<TextOutputCodeFormat, TextOutputCodeFormat>
> = {
  llvm: "asm",
  wgsl: "asm",
}

const emitCodeDebounceMs = 300

export type OutputTab = "output" | "console" | "canvas"

function kernelText({ kernels, error }: KernelResult): string {
  const sections = kernels.map(({ manifest, code }) =>
    kernels.length === 1
      ? code
      : `// kernel ${manifest.entryPoints.map((entry) => entry.name).join(", ")}\n${code}`
  )
  if (error)
    sections.push(
      error
        .split("\n")
        .map((line) => `// ${line}`)
        .join("\n")
    )
  return sections.join("\n")
}

function presentKernels({ kernels, error }: KernelResult) {
  if (error) {
    shaderCanvas.fail(error)
    return
  }
  if (kernels.length !== 1) {
    shaderCanvas.fail("Expected one device module")
    return
  }
  const [module] = kernels
  shaderCanvas.load(module.manifest, module.code)
}

interface PlaygroundContextValue {
  isReady: boolean
  isCompiling: boolean
  device: Device
  diagnosticCount: number
  compileTimeMs: number | null
  currentSampleId: string
  outputFormat: TextOutputCodeFormat
  debugInfo: boolean
  mlirPasses: MLIRPass[]
  activeTab: OutputTab
  runnerState: RunnerState
  loadSample: (id: string) => void
  setDevice: (device: Device) => void
  setOutputFormat: (format: TextOutputCodeFormat) => void
  setDebugInfo: (debugInfo: boolean) => void
  setMLIRPasses: (passes: MLIRPass[]) => void
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
  const [device, setDeviceState] = React.useState<Device>("cpu")
  const [currentSampleId, setCurrentSampleId] = React.useState(
    defaultSampleByDevice.cpu?.id ?? ""
  )
  const [outputFormat, setOutputFormatState] =
    React.useState<TextOutputCodeFormat>("cxxir")
  const [debugInfo, setDebugInfoState] = React.useState(false)
  const [mlirPasses, setMLIRPassesState] = React.useState(defaultMLIRPasses)
  const [activeTab, setActiveTabState] = React.useState<OutputTab>("console")
  const runnerState = React.useSyncExternalStore(
    programRunner.subscribe,
    programRunner.getState
  )

  const isCompilingRef = React.useRef(false)
  const pendingCompileRef = React.useRef(false)
  const compileRevisionRef = React.useRef(0)
  const deviceRef = React.useRef<Device>("cpu")
  const activeTabRef = React.useRef<OutputTab>("console")
  const sourcesRef = React.useRef<Record<Device, string>>({
    cpu: defaultSampleByDevice.cpu?.code ?? "",
    gpu: defaultSampleByDevice.gpu?.code ?? "",
  })
  const sampleIdsRef = React.useRef<Record<Device, string>>({
    cpu: defaultSampleByDevice.cpu?.id ?? "",
    gpu: defaultSampleByDevice.gpu?.id ?? "",
  })
  const outputFormatRef = React.useRef(outputFormat)
  const debugInfoRef = React.useRef(debugInfo)
  const mlirPassesRef = React.useRef(mlirPasses)
  const compileTimeoutRef = React.useRef<ReturnType<typeof setTimeout>>(null)
  const loadingSourceRef = React.useRef(false)
  const kernelRunRef = React.useRef(0)

  const setSource = React.useCallback((code: string) => {
    loadingSourceRef.current = true
    try {
      inputCodeModel.setValue(code)
    } finally {
      loadingSourceRef.current = false
    }
  }, [])

  const runKernels = React.useCallback(async () => {
    const run = ++kernelRunRef.current
    const kernels = await emitKernels({
      debugInfo: debugInfoRef.current,
      optimizationLevel: 0,
    })
    if (run !== kernelRunRef.current || deviceRef.current !== "gpu") return
    presentKernels(kernels)
    shaderCanvas.markStale(false)
    shaderCanvas.restart()
  }, [])

  const emitsKernels = React.useCallback(
    () =>
      deviceRef.current === "gpu" &&
      activeTabRef.current === "output" &&
      outputFormatRef.current === "wgsl",
    []
  )

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

        const kernels = emitsKernels()
          ? await emitKernels({
              debugInfo: debugInfoRef.current,
              optimizationLevel: 0,
            })
          : undefined

        const output = kernels
          ? kernelText(kernels)
          : await emitCode({
              format: outputFormatRef.current,
              debugInfo: debugInfoRef.current,
              mlirPipeline: buildMLIRPipeline(mlirPassesRef.current),
              optimizationLevel: 0,
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
  }, [emitsKernels])

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
      if (deviceRef.current === "gpu" && !loadingSourceRef.current)
        shaderCanvas.markStale(true)
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

  const loadSample = React.useCallback(
    (id: string) => {
      const sample = samplesByDevice[deviceRef.current].find((s) => s.id === id)
      if (!sample) return
      sampleIdsRef.current[deviceRef.current] = id
      setCurrentSampleId(id)
      setSource(sample.code)
      if (deviceRef.current === "gpu") void runKernels()
    },
    [setSource, runKernels]
  )

  const showTab = React.useCallback(
    (tab: OutputTab) => {
      if (activeTabRef.current === tab) return
      const emittedKernels = emitsKernels()
      activeTabRef.current = tab
      setActiveTabState(tab)
      if (emitsKernels() !== emittedKernels) scheduleEmitCode(false)
    },
    [emitsKernels, scheduleEmitCode]
  )

  const setDevice = React.useCallback(
    (next: Device) => {
      const previous = deviceRef.current
      if (previous === next) return
      sourcesRef.current[previous] = inputCodeModel.getValue()
      deviceRef.current = next
      setDeviceState(next)
      if (next === "gpu") programRunner.stop()

      const formats = outputFormatsByDevice[next]
      let format = outputFormatRef.current
      if (!formats.includes(format)) format = equivalentFormat[format]!
      outputFormatRef.current = format
      setOutputFormatState(format)
      monaco.editor.setModelLanguage(
        outputCodeModel,
        outputLanguageOf(next, format)
      )

      const tab = activeTabRef.current
      if (next === "gpu" && tab === "console") {
        activeTabRef.current = "canvas"
        setActiveTabState("canvas")
      }
      if (next === "cpu" && tab === "canvas") {
        activeTabRef.current = "console"
        setActiveTabState("console")
      }

      setCurrentSampleId(sampleIdsRef.current[next])
      setSource(sourcesRef.current[next])
      scheduleEmitCode(false)
      void configureDevice(next).then(() => {
        if (deviceRef.current === "gpu") void runKernels()
      })
    },
    [scheduleEmitCode, setSource, runKernels]
  )

  const setOutputFormat = React.useCallback(
    (format: TextOutputCodeFormat) => {
      if (outputFormatRef.current === format) return
      outputFormatRef.current = format
      setOutputFormatState(format)
      monaco.editor.setModelLanguage(
        outputCodeModel,
        outputLanguageOf(deviceRef.current, format)
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

  const setMLIRPasses = React.useCallback(
    (passes: MLIRPass[]) => {
      mlirPassesRef.current = passes
      setMLIRPassesState(passes)
      scheduleEmitCode(false)
    },
    [scheduleEmitCode]
  )

  const runProgram = React.useCallback(() => {
    if (deviceRef.current === "gpu") {
      showTab("canvas")
      void runKernels()
      return
    }
    showTab("console")
    void programRunner.run({
      debugInfo: debugInfoRef.current,
      mlirPipeline: buildMLIRPipeline(mlirPassesRef.current),
      optimizationLevel: 0,
    })
  }, [showTab, runKernels])

  const stopProgram = React.useCallback(() => {
    if (deviceRef.current === "gpu") {
      shaderCanvas.setPaused(true)
      return
    }
    programRunner.stop()
  }, [])

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.isComposing) return
      if (event.altKey || event.shiftKey) return
      if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey)) return
      event.preventDefault()
      event.stopPropagation()
      if (deviceRef.current === "gpu" || programRunner.getState() === "idle")
        runProgram()
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
      device,
      diagnosticCount,
      compileTimeMs,
      currentSampleId,
      outputFormat,
      debugInfo,
      mlirPasses,
      activeTab,
      runnerState,
      loadSample,
      setDevice,
      setOutputFormat,
      setDebugInfo,
      setMLIRPasses,
      setActiveTab: showTab,
      runProgram,
      stopProgram,
    }),
    [
      isReady,
      isCompiling,
      device,
      diagnosticCount,
      compileTimeMs,
      currentSampleId,
      outputFormat,
      debugInfo,
      mlirPasses,
      activeTab,
      runnerState,
      loadSample,
      setDevice,
      setOutputFormat,
      setDebugInfo,
      setMLIRPasses,
      showTab,
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
