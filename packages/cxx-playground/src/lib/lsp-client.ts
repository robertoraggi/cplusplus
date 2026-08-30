import * as monaco from "monaco-editor"
import type { JsonObject, JsonRpcMessage, MessageTransport } from "cxx-frontend"

interface PendingRequest {
  resolve: (result: unknown) => void
  reject: (error: Error) => void
}

export class LspClient {
  readonly #transport: MessageTransport
  readonly #pending = new Map<string, PendingRequest>()
  readonly #openedDocuments = new Set<string>()
  readonly #documentOpenWaiters = new Map<string, Set<() => void>>()
  readonly #monacoClient: monaco.lsp.MonacoLspClient
  #listener: ((message: JsonRpcMessage) => void) | undefined
  #nextRequestId = 0

  constructor(worker: Worker) {
    this.#transport = monaco.lsp.createTransportToWorker(
      worker
    ) as unknown as MessageTransport

    this.#transport.setListener((message) => this.#receive(message))

    this.#monacoClient = new monaco.lsp.MonacoLspClient(this.#monacoTransport())
  }

  get monacoClient(): monaco.lsp.MonacoLspClient {
    return this.#monacoClient
  }

  whenDocumentOpened(uri: string): Promise<void> {
    if (this.#openedDocuments.has(uri)) return Promise.resolve()

    return new Promise((resolve) => {
      let waiters = this.#documentOpenWaiters.get(uri)
      if (!waiters) {
        waiters = new Set()
        this.#documentOpenWaiters.set(uri, waiters)
      }
      waiters.add(resolve)
    })
  }

  sendRequest<Result>(method: string, params: JsonObject): Promise<Result> {
    const id = `cxx:${this.#nextRequestId++}`

    return new Promise<Result>((resolve, reject) => {
      this.#pending.set(id, {
        resolve: resolve as (result: unknown) => void,
        reject,
      })

      this.#transport
        .send({ jsonrpc: "2.0", id, method, params })
        .catch((error: unknown) => {
          this.#pending.delete(id)
          reject(toError(error))
        })
    })
  }

  #monacoTransport(): MessageTransport {
    return {
      state: this.#transport.state,
      send: (message) => this.#sendMonacoMessage(message),
      setListener: (listener) => {
        this.#listener = listener
      },
      toString: () => this.#transport.toString(),
    }
  }

  async #sendMonacoMessage(message: JsonRpcMessage): Promise<void> {
    await this.#transport.send(message)

    const uri = openedDocumentUri(message)
    if (!uri) return

    this.#openedDocuments.add(uri)
    const waiters = this.#documentOpenWaiters.get(uri)
    if (!waiters) return

    this.#documentOpenWaiters.delete(uri)
    for (const resolve of waiters) resolve()
  }

  #receive(message: JsonRpcMessage): void {
    const id = typeof message.id === "string" ? message.id : undefined
    const pending = id === undefined ? undefined : this.#pending.get(id)

    if (id === undefined || !pending) {
      this.#listener?.(message)
      return
    }

    this.#pending.delete(id)

    if ("error" in message && message.error) {
      pending.reject(new Error(message.error.message))
      return
    }

    pending.resolve("result" in message ? message.result : undefined)
  }
}

function openedDocumentUri(message: JsonRpcMessage): string | undefined {
  if (!("method" in message)) return undefined
  if (message.method !== "textDocument/didOpen") return undefined
  if (!message.params || Array.isArray(message.params)) return undefined

  const textDocument = message.params.textDocument
  if (!textDocument || typeof textDocument !== "object") return undefined
  if (Array.isArray(textDocument)) return undefined

  const uri = textDocument.uri
  return typeof uri === "string" ? uri : undefined
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}
