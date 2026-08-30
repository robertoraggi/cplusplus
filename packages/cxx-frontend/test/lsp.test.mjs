import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadCxx, LanguageServer } from "../dist/index.js";

const wasm = await readFile(
  new URL("../dist/wasm/cxx-js.wasm", import.meta.url),
);

await loadCxx({ wasm });

const uri = "file:///main.cc";

const source = [
  "struct Widget { int value; void method(); };",
  "void ff() {",
  "  Widget w;",
  "  w.",
  "}",
  "",
].join("\n");

const files = new Map();

const resolvers = {
  exists: (path) => files.has(path),
  readFile: async (path) => files.get(path),
};

async function startServer() {
  const messages = [];

  const server = await LanguageServer.start({
    ...resolvers,
    onMessage: (message) => messages.push(message),
  });

  return { server, messages };
}

function responseOf(messages, id) {
  return messages.find((message) => message.id === id);
}

function notificationsOf(messages, method) {
  return messages.filter((message) => message.method === method);
}

function scheduled() {
  return new Promise((resolve) => setTimeout(resolve, 800));
}

test("LanguageServer answers initialize, diagnostics and completion", async () => {
  const { server, messages } = await startServer();

  try {
    await server.receive({ jsonrpc: "2.0", id: 1, method: "initialize" });

    const initialize = responseOf(messages, 1);
    assert.ok(initialize, "expected a response to initialize");
    assert.equal(initialize.result.serverInfo.name, "cxx-lsp");
    assert.ok(initialize.result.capabilities.completionProvider);
    assert.ok(initialize.result.capabilities.signatureHelpProvider);

    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didOpen",
      params: {
        textDocument: { uri, languageId: "cpp", version: 0, text: source },
      },
    });

    const [published] = notificationsOf(
      messages,
      "textDocument/publishDiagnostics",
    );

    assert.ok(published, "expected a publishDiagnostics notification");
    assert.equal(published.params.uri, uri);
    assert.ok(published.params.diagnostics.length > 0);
    assert.equal(published.params.diagnostics[0].severity, 1);

    await server.receive({
      jsonrpc: "2.0",
      id: 2,
      method: "textDocument/completion",
      params: {
        textDocument: { uri },
        position: { line: 3, character: 4 },
      },
    });

    const completion = responseOf(messages, 2);
    assert.ok(completion, "expected a response to textDocument/completion");

    const labels = completion.result.map((item) => item.label);
    assert.ok(labels.includes("value"), `expected 'value' in ${labels}`);
    assert.ok(labels.includes("method"), `expected 'method' in ${labels}`);

    const kindOf = (label) =>
      completion.result.find((item) => item.label === label)?.kind;

    assert.equal(kindOf("value"), 5);
    assert.equal(kindOf("method"), 2);
    assert.equal(kindOf("Widget"), 7);

    await server.receive({ jsonrpc: "2.0", id: 3, method: "shutdown" });
    assert.ok(responseOf(messages, 3), "expected a response to shutdown");

    await server.receive({ jsonrpc: "2.0", method: "exit" });
  } finally {
    server.dispose();
  }
});

test("LanguageServer completes an unqualified identifier prefix", async () => {
  const headers = new Map([
    ["/inc/stdio.h", "int printf(const char*, ...);\n"],
  ]);
  const messages = [];

  const server = await LanguageServer.start({
    includePaths: ["/inc"],
    exists: (path) => headers.has(path),
    readFile: async (path) => headers.get(path),
    onMessage: (message) => messages.push(message),
  });

  try {
    await server.receive({ jsonrpc: "2.0", id: 1, method: "initialize" });

    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didOpen",
      params: {
        textDocument: {
          uri,
          languageId: "cpp",
          version: 0,
          text: "#include <stdio.h>\nint main() { auto value = pri\n}\n",
        },
      },
    });

    await server.receive({
      jsonrpc: "2.0",
      id: 2,
      method: "textDocument/completion",
      params: {
        textDocument: { uri },
        position: { line: 1, character: 29 },
        context: { triggerKind: 1 },
      },
    });

    const completion = responseOf(messages, 2);
    assert.ok(completion, "expected a response to textDocument/completion");

    const labels = completion.result.map((item) => item.label);
    assert.ok(labels.includes("printf"), `expected 'printf' in ${labels}`);

    const printf = completion.result.find((item) => item.label === "printf");
    assert.deepEqual(printf.textEdit, {
      newText: "printf",
      range: {
        start: { line: 1, character: 26 },
        end: { line: 1, character: 29 },
      },
    });
  } finally {
    server.dispose();
  }
});

test("LanguageServer exposes a message transport", async () => {
  const { server } = await startServer();

  try {
    const transport = server.messageTransport();

    assert.deepEqual(transport.state.value, { state: "open" });

    const received = [];
    transport.setListener((message) => received.push(message));

    await transport.send({ jsonrpc: "2.0", id: 1, method: "initialize" });

    assert.equal(received.length, 1);
    assert.equal(received[0].id, 1);

    const states = [];
    transport.state.onChange((state) => states.push(state));

    server.dispose();

    assert.deepEqual(states, [{ state: "closed", error: undefined }]);
    assert.deepEqual(transport.state.value, {
      state: "closed",
      error: undefined,
    });
  } finally {
    server.dispose();
  }
});

test("LanguageServer rejects non-object messages", async () => {
  const { server } = await startServer();

  try {
    await assert.rejects(() => server.receive(null), TypeError);
    await assert.rejects(() => server.receive([]), TypeError);
  } finally {
    server.dispose();
  }
});

test("LanguageServer emits the generated code", async () => {
  const { server, messages } = await startServer();

  try {
    await server.receive({ jsonrpc: "2.0", id: 1, method: "initialize" });

    const initialize = responseOf(messages, 1);
    assert.deepEqual(initialize.result.capabilities.experimental, {
      cxxEmitCode: true,
    });

    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didOpen",
      params: {
        textDocument: {
          uri,
          languageId: "cpp",
          version: 0,
          text: "int main() { return 42; }\n",
        },
      },
    });

    const expected = {
      cxxir: /^module @/m,
      mlir: /^module @/m,
      llvm: /^target triple = /m,
      asm: /\.functype\s+main/,
    };

    for (const [format, pattern] of Object.entries(expected)) {
      const id = `emit-${format}`;

      await server.receive({
        jsonrpc: "2.0",
        id,
        method: "cxx/emitCode",
        params: { textDocument: { uri }, format, debugInfo: false },
      });

      const response = responseOf(messages, id);
      assert.ok(response, `expected a response for the ${format} format`);
      assert.equal(response.result.format, format);
      assert.match(response.result.text, pattern);
    }

    await server.receive({
      jsonrpc: "2.0",
      id: "emit-debug",
      method: "cxx/emitCode",
      params: { textDocument: { uri }, format: "cxxir", debugInfo: true },
    });

    assert.match(responseOf(messages, "emit-debug").result.text, /\bloc\(/);
  } finally {
    server.dispose();
  }
});

test("cxx/emitCode reuses the parse that produced the diagnostics", async () => {
  const headers = new Map([["/inc/helper.h", "int helper();\n"]]);
  let reads = 0;

  const messages = [];

  const server = await LanguageServer.start({
    includePaths: ["/inc"],
    exists: (path) => headers.has(path),
    readFile: async (path) => {
      reads++;
      return headers.get(path);
    },
    onMessage: (message) => messages.push(message),
  });

  const text = "#include <helper.h>\nint main() { return helper(); }\n";

  try {
    await server.receive({ jsonrpc: "2.0", id: 1, method: "initialize" });

    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didOpen",
      params: {
        textDocument: { uri, languageId: "cpp", version: 0, text },
      },
    });

    const afterDidOpen = reads;
    assert.equal(afterDidOpen, 1);

    await server.receive({
      jsonrpc: "2.0",
      id: 2,
      method: "cxx/emitCode",
      params: { textDocument: { uri }, format: "cxxir" },
    });

    assert.equal(reads, afterDidOpen);

    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didChange",
      params: {
        textDocument: { uri, version: 1 },
        contentChanges: [{ text: `${text}int other() { return 1; }\n` }],
      },
    });

    await scheduled();

    const afterDidChange = reads;
    assert.equal(afterDidChange, 2);

    await server.receive({
      jsonrpc: "2.0",
      id: 3,
      method: "cxx/emitCode",
      params: { textDocument: { uri }, format: "cxxir" },
    });

    assert.equal(reads, afterDidChange);
    assert.match(responseOf(messages, 3).result.text, /other/);

    const published = notificationsOf(
      messages,
      "textDocument/publishDiagnostics",
    );

    assert.deepEqual(
      published.map(({ params }) => params.version),
      [0, 1],
    );
  } finally {
    server.dispose();
  }
});

test("document edits cancel active and queued parser requests", async () => {
  const header = "struct Widget { int value; };\n";
  let readCount = 0;
  let releaseFirstRead;
  let markFirstReadStarted;
  const firstReadStarted = new Promise((resolve) => {
    markFirstReadStarted = resolve;
  });
  const messages = [];
  const traces = [];

  const server = await LanguageServer.start({
    includePaths: ["/inc"],
    exists: (path) => path === "/inc/widget.h",
    readFile: async () => {
      readCount++;

      if (readCount === 1) {
        markFirstReadStarted();
        await new Promise((resolve) => {
          releaseFirstRead = resolve;
        });
      }

      return header;
    },
    onMessage: (message) => messages.push(message),
    onTrace: (message) => traces.push(message),
  });

  const source =
    "#include <widget.h>\nint main() { Widget widget; return widget.value; }\n";

  try {
    await server.receive({ jsonrpc: "2.0", id: 1, method: "initialize" });

    const opening = server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didOpen",
      params: {
        textDocument: { uri, languageId: "cpp", version: 0, text: source },
      },
    });

    await firstReadStarted;

    await server.receive({
      jsonrpc: "2.0",
      id: 2,
      method: "textDocument/completion",
      params: {
        textDocument: { uri },
        position: { line: 1, character: 30 },
      },
    });

    await server.receive({
      jsonrpc: "2.0",
      id: 3,
      method: "cxx/emitCode",
      params: { textDocument: { uri }, format: "cxxir" },
    });

    await server.receive({
      jsonrpc: "2.0",
      id: 4,
      method: "textDocument/signatureHelp",
      params: {
        textDocument: { uri },
        position: { line: 1, character: 30 },
      },
    });

    await server.receive({
      jsonrpc: "2.0",
      method: "$/cancelRequest",
      params: { id: 2 },
    });

    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didChange",
      params: {
        textDocument: { uri, version: 1 },
        contentChanges: [
          { text: `${source}int freshMarker() { return 7; }\n` },
        ],
      },
    });

    releaseFirstRead();
    await opening;
    await scheduled();

    assert.equal(readCount, 2);
    assert.equal(responseOf(messages, 2).result, null);
    assert.equal(responseOf(messages, 3).result, null);
    assert.equal(responseOf(messages, 4).result, null);
    assert.deepEqual(
      notificationsOf(messages, "textDocument/publishDiagnostics").map(
        ({ params }) => params.version,
      ),
      [1],
    );
    assert.ok(
      traces.some((trace) =>
        trace.includes("event=invalidated file=/main.cc version=1 cancelled=3"),
      ),
    );
    assert.ok(
      traces.some((trace) =>
        trace.includes("event=skipped kind=completion file=/main.cc version=0"),
      ),
    );
    assert.ok(
      traces.some((trace) =>
        trace.includes("event=request-cancelled id=2 matched=true"),
      ),
    );
    assert.ok(
      traces.some((trace) =>
        trace.includes("phase=preprocessing file=/main.cc version=0"),
      ),
    );
  } finally {
    releaseFirstRead?.();
    server.dispose();
  }
});

test("LanguageServer ignores the documents it does not speak", async () => {
  const { server, messages } = await startServer();

  try {
    await server.receive({ jsonrpc: "2.0", id: 1, method: "initialize" });

    await server.receive({
      jsonrpc: "2.0",
      method: "textDocument/didOpen",
      params: {
        textDocument: {
          uri: "file:///main.mlir",
          languageId: "mlir",
          version: 0,
          text: "this is not C++ at all",
        },
      },
    });

    assert.deepEqual(
      notificationsOf(messages, "textDocument/publishDiagnostics"),
      [],
    );
  } finally {
    server.dispose();
  }
});

test("LanguageServer serves an editor session over a message transport", async () => {
  const { server, messages } = await startServer();
  const outputUri = "file:///main.mlir";

  const received = [];
  const transport = server.messageTransport();
  transport.setListener((message) => received.push(message));

  let nextId = 1;

  const request = async (method, params) => {
    const id = nextId++;
    await transport.send({ jsonrpc: "2.0", id, method, params });
    return received.find((message) => message.id === id);
  };

  const notify = (method, params) =>
    transport.send({ jsonrpc: "2.0", method, params });

  try {
    const initialize = await request("initialize", {
      processId: null,
      rootUri: null,
      capabilities: {},
    });

    assert.ok(initialize.result.capabilities.completionProvider);

    await notify("initialized", {});

    await notify("textDocument/didOpen", {
      textDocument: {
        uri,
        languageId: "cpp",
        version: 1,
        text: "struct Widget { int value; void method(); };\nvoid ff() {\n  Widget w;\n\n}\n",
      },
    });

    await notify("textDocument/didOpen", {
      textDocument: {
        uri: outputUri,
        languageId: "mlir",
        version: 1,
        text: "",
      },
    });

    assert.deepEqual(
      notificationsOf(messages, "textDocument/publishDiagnostics").map(
        ({ params }) => params.uri,
      ),
      [uri],
    );

    await notify("textDocument/didChange", {
      textDocument: { uri, version: 2 },
      contentChanges: [
        {
          range: {
            start: { line: 3, character: 0 },
            end: { line: 3, character: 0 },
          },
          rangeLength: 0,
          text: "  w.",
        },
      ],
    });

    await scheduled();

    const completion = await request("textDocument/completion", {
      textDocument: { uri },
      position: { line: 3, character: 4 },
      context: { triggerKind: 2, triggerCharacter: "." },
    });

    const labels = completion.result.map((item) => item.label);
    assert.ok(labels.includes("value"), `expected 'value' in ${labels}`);
    assert.ok(labels.includes("method"), `expected 'method' in ${labels}`);

    const untracked = await request("textDocument/completion", {
      textDocument: { uri: outputUri },
      position: { line: 0, character: 0 },
    });

    assert.equal(untracked.result, null);

    const untrackedCode = await request("cxx/emitCode", {
      textDocument: { uri: outputUri },
      format: "cxxir",
    });

    assert.equal(untrackedCode.result, null);

    await notify("textDocument/didChange", {
      textDocument: { uri, version: 3 },
      contentChanges: [
        {
          range: {
            start: { line: 3, character: 0 },
            end: { line: 3, character: 4 },
          },
          rangeLength: 4,
          text: "  w.method();",
        },
      ],
    });

    await notify("textDocument/didChange", {
      textDocument: { uri, version: 4 },
      contentChanges: [
        {
          range: {
            start: { line: 5, character: 0 },
            end: { line: 5, character: 0 },
          },
          rangeLength: 0,
          text: "int freshMarker() { return 7; }\n",
        },
      ],
    });

    await scheduled();

    const emitted = await request("cxx/emitCode", {
      textDocument: { uri },
      format: "cxxir",
    });

    assert.match(emitted.result.text, /freshMarker/);

    assert.deepEqual(
      notificationsOf(messages, "textDocument/publishDiagnostics").map(
        ({ params }) => params.version,
      ),
      [1, 2, 4],
    );
  } finally {
    server.dispose();
  }
});
