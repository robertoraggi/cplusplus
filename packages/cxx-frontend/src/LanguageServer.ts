// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import { cxx } from "./cxx.js";
import {
  type LanguageServer as NativeLanguageServer,
  type LanguageServerOptions as NativeLanguageServerOptions,
} from "./cxx-js.js";
import { isCxxLoaded } from "./loadCxx.js";
import { asyncDisposeSymbol, disposeSymbol } from "./disposeSymbols.js";
import { continueWithEventLoopYields } from "./eventLoop.js";

/**
 * The id of a JSON-RPC request.
 */
export type JsonRpcId = number | string;

/**
 * A JSON value.
 */
export type JsonValue =
  string | number | boolean | null | JsonObject | JsonArray;

/**
 * A JSON object.
 */
export interface JsonObject {
  [key: string]: JsonValue | undefined;
}

/**
 * A JSON array.
 */
export interface JsonArray extends Array<JsonValue> {}

/**
 * A JSON-RPC request or notification.
 *
 * Notifications are the messages without an `id`.
 */
export interface JsonRpcRequestMessage {
  jsonrpc: "2.0";
  method: string;
  params?: JsonArray | JsonObject;
  id?: JsonRpcId;
  result?: never;
}

/**
 * A JSON-RPC response.
 *
 * Either `result` or `error` is set.
 */
export interface JsonRpcResponseMessage {
  jsonrpc: "2.0";
  result?: JsonValue;
  error?: { code: number; message: string; data?: JsonValue };
  id: JsonRpcId | null;
  method?: never;
}

/**
 * A message exchanged with the language server.
 */
export type JsonRpcMessage = JsonRpcRequestMessage | JsonRpcResponseMessage;

/**
 * Subscribes a listener and returns its subscription.
 */
export interface Event<T> {
  (listener: (e: T) => void): { dispose(): void };
}

/**
 * A value that notifies its observers when it changes.
 */
export interface ValueWithChangeEvent<T> {
  readonly value: T;
  readonly onChange: Event<T>;
}

/**
 * The state of the connection to a language server.
 */
export type ConnectionState =
  | { state: "connecting" }
  | { state: "open" }
  | { state: "closed"; error: Error | undefined };

/**
 * A bidirectional JSON-RPC message channel.
 *
 * The shape is the one expected by the LSP client of the editors, it is
 * described here so that this package does not depend on them.
 */
export interface MessageTransport {
  readonly state: ValueWithChangeEvent<ConnectionState>;
  send(message: JsonRpcMessage): Promise<void>;
  setListener(listener: ((message: JsonRpcMessage) => void) | undefined): void;
  toString(): string;
}

/**
 * The options of {@link LanguageServer.start}.
 */
export interface LanguageServerOptions extends Omit<
  NativeLanguageServerOptions,
  "onMessage" | "onTrace" | "shouldContinue"
> {
  /**
   * Receives the requests, responses and notifications sent by the server.
   */
  onMessage(message: JsonRpcMessage): void;

  /**
   * Receives parser queue, phase, cancellation and code generation timings.
   */
  onTrace?(message: string, verbose?: string): void;
}

/**
 * The endpoint {@link LanguageServer.serve} exchanges messages with.
 *
 * A dedicated worker's global scope and a `MessagePort` both match it.
 */
export interface MessagePortLike {
  postMessage(message: unknown): void;
  addEventListener(
    type: "message",
    listener: (event: { data: unknown }) => void,
  ): void;
  start?(): void;
}

/**
 * The options of {@link LanguageServer.serve}.
 */
export interface ServeOptions extends Omit<LanguageServerOptions, "onMessage"> {
  /**
   * The endpoint the server reads the messages from and writes them to.
   */
  port: MessagePortLike;

  /**
   * Receives the messages sent by the server, after they are posted to
   * {@link ServeOptions.port}.
   */
  onMessage?(message: JsonRpcMessage): void;
}

/**
 * A language server backed by the cxx frontend.
 *
 * The server runs on the JavaScript thread, preprocessing and parsing
 * periodically yield to the event loop so every entry point is asynchronous.
 */
export class LanguageServer implements Disposable, AsyncDisposable {
  #server: NativeLanguageServer | undefined;
  #onMessage: (message: JsonRpcMessage) => void;
  #listener: ((message: JsonRpcMessage) => void) | undefined;
  readonly #state = new ConnectionStateValue();

  private constructor(
    server: NativeLanguageServer,
    onMessage: (message: JsonRpcMessage) => void,
  ) {
    this.#server = server;
    this.#onMessage = onMessage;
  }

  /**
   * Starts a language server.
   *
   * @param options the message sink and the include resolvers.
   * @returns the started language server.
   */
  static async start(options: LanguageServerOptions): Promise<LanguageServer> {
    const { onMessage, onTrace, ...serverOptions } = options;

    if (typeof onMessage !== "function") {
      throw new TypeError("expected parameter 'onMessage' of type 'function'");
    }

    if (onTrace !== undefined && typeof onTrace !== "function") {
      throw new TypeError("expected parameter 'onTrace' of type 'function'");
    }

    if (!isCxxLoaded()) {
      throw new Error(
        "the cxx wasm module is not loaded, call loadCxx() first",
      );
    }

    let languageServer: LanguageServer | undefined;
    const shouldContinue = continueWithEventLoopYields(() => true);

    const server = cxx.createLanguageServer({
      ...serverOptions,
      shouldContinue,
      onTrace,
      onMessage: (message: string) => {
        if (!languageServer) return;
        languageServer.#dispatch(JSON.parse(message) as JsonRpcMessage);
      },
    });

    if (!server) {
      throw new Error("failed to create the language server");
    }

    languageServer = new LanguageServer(server, onMessage);

    return languageServer;
  }

  /**
   * Starts a language server attached to a message endpoint.
   *
   * In a dedicated worker `port` is the worker global scope itself, the
   * messages are then plain JSON-RPC objects sent with `postMessage`.
   *
   * @param options the endpoint and the include resolvers.
   * @returns the started language server.
   */
  static async serve(options: ServeOptions): Promise<LanguageServer> {
    const { port, onMessage, ...serverOptions } = options;

    const languageServer = await LanguageServer.start({
      ...serverOptions,
      onMessage: (message) => {
        port.postMessage(message);
        onMessage?.(message);
      },
    });

    port.addEventListener("message", (event) => {
      void languageServer.receive(event.data as JsonRpcMessage);
    });

    port.start?.();

    return languageServer;
  }

  /**
   * Sends a message to the server.
   *
   * The returned promise settles when the server is done with the message, the
   * replies and the notifications it produced were already handed to
   * {@link LanguageServerOptions.onMessage}.
   *
   * @param message the request, response or notification to process.
   */
  async receive(message: JsonRpcMessage): Promise<void> {
    if (typeof message !== "object" || message === null) {
      throw new TypeError("expected a JSON-RPC message object");
    }

    if (Array.isArray(message)) {
      throw new TypeError("batched JSON-RPC messages are not supported");
    }

    await this.#nativeServer().receive(JSON.stringify(message));
  }

  /**
   * Returns a message transport for this server.
   *
   * The transport is the shape the LSP clients of the editors expect, it can be
   * given to a client running in the same realm as the server.
   */
  messageTransport(): MessageTransport {
    return {
      state: this.#state,
      send: (message) => this.receive(message),
      setListener: (listener) => {
        this.#listener = listener;
      },
      toString: () => "cxx-frontend language server",
    };
  }

  /**
   * Releases the native resources owned by the server.
   *
   * Calling `dispose` more than once is allowed, a disposed server must not
   * receive any further message.
   */
  dispose(): void {
    if (!this.#server) return;

    this.#server.delete();
    this.#server = undefined;
    this.#listener = undefined;
    this.#state.close();
  }

  [disposeSymbol](): void {
    this.dispose();
  }

  async [asyncDisposeSymbol](): Promise<void> {
    this.dispose();
  }

  #dispatch(message: JsonRpcMessage): void {
    this.#onMessage(message);
    this.#listener?.(message);
  }

  #nativeServer(): NativeLanguageServer {
    if (!this.#server) {
      throw new Error("LanguageServer has been disposed");
    }

    return this.#server;
  }
}

class ConnectionStateValue implements ValueWithChangeEvent<ConnectionState> {
  #value: ConnectionState = { state: "open" };
  readonly #listeners = new Set<(e: ConnectionState) => void>();

  get value(): ConnectionState {
    return this.#value;
  }

  get onChange(): Event<ConnectionState> {
    return (listener) => {
      this.#listeners.add(listener);
      return {
        dispose: () => {
          this.#listeners.delete(listener);
        },
      };
    };
  }

  close(): void {
    this.#value = { state: "closed", error: undefined };
    for (const listener of [...this.#listeners]) listener(this.#value);
  }
}
