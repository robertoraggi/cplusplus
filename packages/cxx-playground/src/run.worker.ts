import {
  ConsoleStdout,
  File,
  OpenFile,
  WASI,
  WASIProcExit,
  wasi as wasiDefinitions,
  type Fd,
} from "@bjorn3/browser_wasi_shim"
import type { RunCommand, RunMessage, WorkerScope } from "./run/protocol"

const scope = globalThis as unknown as WorkerScope

type WasiImport = WASI["wasiImport"]

const supportsSuspension =
  typeof WebAssembly.Suspending === "function" &&
  typeof WebAssembly.promising === "function"

function post(message: RunMessage) {
  const transfer = message.kind === "output" ? [message.data.buffer] : []
  scope.postMessage(message, transfer)
}

const stdout = () =>
  new ConsoleStdout((buffer) => post({ kind: "output", data: buffer.slice() }))

class StdinQueue {
  #chunks: Uint8Array[] = []
  #closed = false
  #wake: (() => void) | undefined

  push(data: Uint8Array) {
    this.#chunks.push(data)
    this.#wake?.()
  }

  close() {
    this.#closed = true
    this.#wake?.()
  }

  async read(size: number): Promise<Uint8Array> {
    for (;;) {
      const chunk = this.#chunks[0]

      if (chunk) {
        const taken = chunk.subarray(0, size)
        if (taken.length === chunk.length) this.#chunks.shift()
        else this.#chunks[0] = chunk.subarray(size)
        return taken
      }

      if (this.#closed) return new Uint8Array(0)

      post({ kind: "reading" })
      await new Promise<void>((resolve) => {
        this.#wake = resolve
      })
    }
  }
}

class UnavailableStdin extends OpenFile {
  #reported = false

  constructor() {
    super(new File([]))
  }

  fd_read(size: number) {
    if (!this.#reported) post({ kind: "stdin-unavailable" })
    this.#reported = true
    return super.fd_read(size)
  }
}

function suspendingFdRead(wasi: WASI, stdin: StdinQueue, original: WasiImport) {
  return new WebAssembly.Suspending(
    async (
      fd: number,
      iovsPointer: number,
      iovsLength: number,
      nread: number
    ) => {
      if (fd !== 0)
        return original["fd_read"]!(fd, iovsPointer, iovsLength, nread)

      const { memory } = wasi.inst.exports
      const iovecs = wasiDefinitions.Iovec.read_bytes_array(
        new DataView(memory.buffer),
        iovsPointer,
        iovsLength
      )
      const capacity = iovecs.reduce((sum, iovec) => sum + iovec.buf_len, 0)
      const data = await stdin.read(capacity)

      const bytes = new Uint8Array(memory.buffer)
      let offset = 0
      for (const iovec of iovecs) {
        const part = data.subarray(offset, offset + iovec.buf_len)
        bytes.set(part, iovec.buf)
        offset += part.length
        if (part.length < iovec.buf_len) break
      }

      new DataView(memory.buffer).setUint32(nread, data.length, true)
      return wasiDefinitions.ERRNO_SUCCESS
    }
  )
}

const stdin = new StdinQueue()

async function run(wasm: Uint8Array<ArrayBuffer>, args: string[]) {
  const fds: Fd[] = [
    supportsSuspension ? new OpenFile(new File([])) : new UnavailableStdin(),
    stdout(),
    stdout(),
  ]
  const wasi = new WASI(args, [], fds)
  const imports = {
    ...wasi.wasiImport,
    ...(supportsSuspension && {
      fd_read: suspendingFdRead(wasi, stdin, wasi.wasiImport),
    }),
  } as WebAssembly.ModuleImports

  try {
    const { instance } = await WebAssembly.instantiate(wasm, {
      wasi_snapshot_preview1: imports,
    })
    const exports = instance.exports as unknown as {
      memory: WebAssembly.Memory
      _start: () => unknown
    }
    wasi.inst = { exports }

    if (!supportsSuspension) {
      post({ kind: "exit", code: wasi.start({ exports }) })
      return
    }

    try {
      await WebAssembly.promising(exports._start)()
      post({ kind: "exit", code: 0 })
    } catch (error) {
      if (!(error instanceof WASIProcExit)) throw error
      post({ kind: "exit", code: error.code })
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    post({ kind: "trap", message })
  }
}

scope.addEventListener<RunCommand>("message", ({ data }) => {
  switch (data.kind) {
    case "start":
      void run(data.wasm, data.args)
      break
    case "input":
      stdin.push(data.data)
      break
    case "eof":
      stdin.close()
      break
  }
})
