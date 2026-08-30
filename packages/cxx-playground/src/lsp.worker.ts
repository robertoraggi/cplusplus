import wasmBinaryUrl from "cxx-frontend/wasm?url"
import { LanguageServer, loadCxx, type MessagePortLike } from "cxx-frontend"
import { appdir, exists, loadSysroot, readFile, sysroot } from "./sysroot"

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

Promise.all([loadCxx({ wasmURL: wasmBinaryUrl }), loadSysroot()])
  .then(() =>
    LanguageServer.serve({
      port,
      appdir,
      sysroot,
      std: "c++17",
      exists,
      readFile,
      onTrace: (message) => console.info(`[cxx-lsp] ${message}`),
    })
  )
  .catch((error: unknown) => {
    console.error("Failed to start the language server", error)
  })
