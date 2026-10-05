import * as monaco from "monaco-editor"
import { emitExecutable } from "./language-server"
import { inputCodeModel } from "./input-code-model"
import RunWorker from "./run.worker?worker"
import type { RunCommand, RunMessage } from "./run/protocol"
import {
  clearScreen,
  dim,
  red,
  toTerminalBytes,
  yellow,
} from "./run/terminal-text"

export type RunnerState = "idle" | "compiling" | "running"

export interface TerminalSink {
  write(data: Uint8Array<ArrayBuffer>): void
  focus(): void
}

export interface RunOptions {
  debugInfo: boolean
  optimizationLevel: number
}

const maxOutputBytes = 4 * 1024 * 1024

const encoder = new TextEncoder()

function markerLines(): string[] {
  return monaco.editor
    .getModelMarkers({ resource: inputCodeModel.uri })
    .filter((marker) => marker.severity >= monaco.MarkerSeverity.Error)
    .map(
      (marker) =>
        `main.cc:${marker.startLineNumber}:${marker.startColumn}: error: ${marker.message}`
    )
}

class ProgramRunner {
  #state: RunnerState = "idle"
  #generation = 0
  #worker: Worker | undefined
  #sink: TerminalSink | undefined
  #pending: Uint8Array<ArrayBuffer>[] = []
  #queue: Uint8Array[] = []
  #frame = 0
  #total = 0
  #followsCarriageReturn = false
  #line: string[] = []
  readonly #listeners = new Set<() => void>()

  subscribe = (listener: () => void) => {
    this.#listeners.add(listener)
    return () => {
      this.#listeners.delete(listener)
    }
  }

  getState = () => this.#state

  attach(sink: TerminalSink): () => void {
    this.#sink = sink
    for (const data of this.#pending.splice(0)) sink.write(data)
    return () => {
      if (this.#sink === sink) this.#sink = undefined
    }
  }

  async run(options: RunOptions): Promise<void> {
    this.stop()

    const generation = ++this.#generation
    this.#setState("compiling")
    this.#emit(clearScreen)
    this.#total = 0
    this.#followsCarriageReturn = false
    this.#line = []

    try {
      const result = await emitExecutable(options)
      if (generation !== this.#generation) return

      if (result.kind === "built") {
        this.#start(result.wasm)
        return
      }

      const lines = result.message ? [result.message] : markerLines()
      for (const line of lines.length ? lines : ["the program has errors"]) {
        this.#emit(red(line))
      }
    } catch (error) {
      if (generation !== this.#generation) return
      this.#emit(red(error instanceof Error ? error.message : String(error)))
    }

    this.#setState("idle")
  }

  stop(): void {
    const wasActive = this.#state !== "idle"
    this.#generation++
    this.#terminateWorker()
    this.#flush()
    if (!wasActive) return
    this.#emit(dim("[stopped]"))
    this.#setState("idle")
  }

  #start(wasm: Uint8Array<ArrayBuffer>) {
    const worker = new RunWorker()
    this.#worker = worker
    this.#setState("running")

    worker.onmessage = ({ data }: MessageEvent<RunMessage>) => {
      if (this.#worker !== worker) return
      this.#receive(data)
    }

    worker.onerror = (event) => {
      if (this.#worker !== worker) return
      this.#finish(red(event.message || "the program failed"))
    }

    const command: RunCommand = { kind: "start", wasm, args: ["main"] }
    worker.postMessage(command, [wasm.buffer])
  }

  input(text: string): void {
    if (this.#state !== "running" || text.startsWith("\x1b")) return

    for (const character of text) {
      if (character === "\x03") {
        this.#flush()
        this.#emit(encoder.encode("^C"))
        this.stop()
        return
      }

      this.#edit(character)
    }
  }

  #edit(character: string) {
    this.#flush()

    switch (character) {
      case "\r":
      case "\n":
        this.#emit(encoder.encode("\r\n"))
        this.#send(`${this.#line.join("")}\n`)
        this.#line = []
        break
      case "\x7f":
      case "\b":
        if (this.#line.pop() !== undefined) this.#emit(encoder.encode("\b \b"))
        break
      case "\x04":
        if (this.#line.length) this.#send(this.#line.join(""))
        else this.#worker?.postMessage({ kind: "eof" } satisfies RunCommand)
        this.#line = []
        break
      default:
        if (character < " ") break
        this.#line.push(character)
        this.#emit(encoder.encode(character))
    }
  }

  #send(text: string) {
    const data = encoder.encode(text)
    const command: RunCommand = { kind: "input", data }
    this.#worker?.postMessage(command, [data.buffer])
  }

  #receive(message: RunMessage) {
    switch (message.kind) {
      case "output":
        this.#enqueue(message.data)
        break
      case "reading":
        this.#sink?.focus()
        break
      case "stdin-unavailable":
        this.#flush()
        this.#emit(
          yellow("[this browser cannot suspend WebAssembly: stdin is empty]")
        )
        break
      case "exit":
        this.#flush()
        this.#finish(
          message.code === 0
            ? dim("[exited]")
            : yellow(`[exited with code ${message.code}]`)
        )
        break
      case "trap":
        this.#flush()
        this.#finish(red(`[trap: ${message.message}]`))
        break
    }
  }

  #finish(line: Uint8Array<ArrayBuffer>) {
    this.#terminateWorker()
    this.#generation++
    this.#emit(line)
    this.#setState("idle")
  }

  #enqueue(data: Uint8Array<ArrayBuffer>) {
    this.#total += data.length
    this.#queue.push(data)

    if (this.#total > maxOutputBytes) {
      this.#flush()
      this.#terminateWorker()
      this.#generation++
      this.#emit(red("[output limit exceeded]"))
      this.#setState("idle")
      return
    }

    if (this.#frame) return
    this.#frame = requestAnimationFrame(() => {
      this.#frame = 0
      this.#flush()
    })
  }

  #flush() {
    if (this.#frame) cancelAnimationFrame(this.#frame)
    this.#frame = 0

    if (!this.#queue.length) return

    const { bytes, followsCarriageReturn } = toTerminalBytes(
      this.#queue.splice(0),
      this.#followsCarriageReturn
    )
    this.#followsCarriageReturn = followsCarriageReturn
    this.#emit(bytes)
  }

  #emit(data: Uint8Array<ArrayBuffer>) {
    if (this.#sink) {
      this.#sink.write(data)
      return
    }
    this.#pending.push(data)
  }

  #terminateWorker() {
    this.#worker?.terminate()
    this.#worker = undefined
  }

  #setState(state: RunnerState) {
    if (this.#state === state) return
    this.#state = state
    for (const listener of this.#listeners) listener()
  }
}

export const programRunner = new ProgramRunner()
