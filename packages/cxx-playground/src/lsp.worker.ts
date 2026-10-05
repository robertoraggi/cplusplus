import wasmBinaryUrl from "cxx-frontend/wasm?url"
import { Linker, loadCxx } from "cxx-frontend"
import { LanguageServer, type MessagePortLike } from "cxx-frontend/lsp"
import {
  appdir,
  exists,
  loadSysroot,
  readBinaryFile,
  readDirectory,
  readFile,
  sysroot,
} from "./sysroot"

interface LanguageServerWorkerScope {
  addEventListener(
    type: "message",
    listener: (event: { data: unknown }) => void
  ): void
  postMessage(message: unknown): void
}

const scope = globalThis as unknown as LanguageServerWorkerScope

const inbox: unknown[] = []
let deliver: ((event: { data: unknown }) => void) | undefined

scope.addEventListener("message", (event) => {
  if (deliver) {
    deliver(event)
    return
  }
  inbox.push(event.data)
})

const port: MessagePortLike = {
  postMessage: (message) => scope.postMessage(message),
  addEventListener: (_type, listener) => {
    deliver = listener
    for (const data of inbox.splice(0)) listener({ data })
  },
}

let linker: Linker | undefined

async function loadLinker(): Promise<void> {
  try {
    linker = await Linker.create({ sysroot, readFile: readBinaryFile })
  } catch (error) {
    console.warn("Failed to load the libraries; programs cannot be run", error)
  }
}

Promise.all([loadCxx({ wasmURL: wasmBinaryUrl }), loadSysroot()])
  .then(() =>
    LanguageServer.serve({
      port,
      appdir,
      sysroot,
      std: "c++26",
      exists,
      readFile,
      readDirectory,
      linker: () => linker,
      onTrace: (message) => console.info(`[cxx-lsp] ${message}`),
    })
  )
  .then(loadLinker)
  .catch((error: unknown) => {
    console.error("Failed to start the language server", error)
  })
