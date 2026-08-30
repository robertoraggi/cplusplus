import * as monaco from "monaco-editor"
import type { JsonObject, JsonRpcMessage, MessageTransport } from "cxx-frontend"

interface PendingRequest {
  resolve: (result: unknown) => void
  reject: (error: Error) => void
}

export class LspClient {
  readonly #transport: MessageTransport
  readonly #pending = new Map<string, PendingRequest>()
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
      send: (message) => this.#transport.send(message),
      setListener: (listener) => {
        this.#listener = listener
      },
      toString: () => this.#transport.toString(),
    }
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

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}
