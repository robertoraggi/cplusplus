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

import {
  loadCxx,
  Parser,
  Semantic as S,
  Token,
  TraceEmitter,
} from "cxx-frontend";
import { statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const usage = `Usage: node index.js [options] <file...>

Options:
  --appdir <path>      locate the bundled headers next to <path>
  --sysroot <path>     resolve system headers from <path>
  --std <standard>     c++14, c++17, c++20, c++23 or c++26
  -I, --include <path> add <path> to the system include paths
  -D, --define <macro> define <macro>, as NAME or NAME=VALUE
  --ast                print the syntax tree of each file
  --trace              print the code generated for each file, as an emitter trace
  --symbols            print the symbols declared in the global scope
  -h, --help           print this message
`;

const options = {
  appdir: { type: "string" },
  sysroot: { type: "string" },
  std: { type: "string" },
  include: { type: "string", short: "I", multiple: true, default: [] },
  define: { type: "string", short: "D", multiple: true, default: [] },
  ast: { type: "boolean", default: false },
  trace: { type: "boolean", default: false },
  symbols: { type: "boolean", default: false },
  help: { type: "boolean", short: "h", default: false },
};

const styles = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  cyan: "\x1b[36m",
};

const colored = process.stdout.isTTY && !process.env.NO_COLOR;
const style = (name, text) =>
  colored ? `${styles[name]}${text}${styles.reset}` : text;

const severityStyle = {
  fatal: "red",
  error: "red",
  warning: "yellow",
  note: "cyan",
  message: "green",
};

class SourceCache {
  #texts = new Map();
  #lines = new Map();

  add(fileName, text) {
    this.#texts.set(fileName, text);
  }

  exists(fileName) {
    if (this.#texts.has(fileName)) return true;
    try {
      return statSync(fileName).isFile();
    } catch {
      return false;
    }
  }

  async read(fileName) {
    if (this.#texts.has(fileName)) return this.#texts.get(fileName);
    try {
      const text = await readFile(fileName, "utf8");
      this.add(fileName, text);
      return text;
    } catch (error) {
      if (error.code === "ENOENT") return undefined;
      throw error;
    }
  }

  lineAt(fileName, line) {
    if (!this.#texts.has(fileName)) return undefined;
    let lines = this.#lines.get(fileName);
    if (!lines) {
      lines = this.#texts.get(fileName).split("\n");
      this.#lines.set(fileName, lines);
    }
    return lines[line - 1];
  }
}

function location({ fileName, startLine, startColumn }) {
  return `${fileName}:${startLine}:${startColumn}`;
}

function printSnippet(sources, diagnostic) {
  const { fileName, startLine, startColumn, endLine, endColumn } = diagnostic;
  const text = sources.lineAt(fileName, startLine);
  if (text === undefined) return;

  const gutter = String(startLine);
  const blank = " ".repeat(gutter.length);
  const last = endLine === startLine ? endColumn : text.length + 1;
  const width = Math.max(1, last - startColumn);
  const marker = `${" ".repeat(startColumn - 1)}^${"~".repeat(width - 1)}`;

  console.error(`${style("dim", `${gutter} |`)} ${text}`);
  console.error(`${style("dim", `${blank} |`)} ${style("green", marker)}`);
}

function printDiagnostics(sources, diagnostics) {
  let errors = 0;
  let warnings = 0;

  for (const diagnostic of diagnostics) {
    const { severity, message } = diagnostic;
    if (severity === "error" || severity === "fatal") ++errors;
    if (severity === "warning") ++warnings;

    console.error(
      `${style("bold", location(diagnostic))}: ` +
        `${style(severityStyle[severity], severity)}: ${message}`,
    );
    printSnippet(sources, diagnostic);

    for (const note of diagnostic.notes) {
      console.error(
        `${style("bold", location(note))}: ${style("cyan", "note")}: ${note.message}`,
      );
      printSnippet(sources, note);
    }
  }

  const counted = [];
  if (errors) counted.push(`${errors} error${errors === 1 ? "" : "s"}`);
  if (warnings) counted.push(`${warnings} warning${warnings === 1 ? "" : "s"}`);
  if (counted.length) console.error(counted.join(", "), "generated.");

  return errors;
}

function semanticsOf(node) {
  const parts = [];
  const symbol = "symbol" in node ? node.symbol : undefined;
  if (symbol?.text) parts.push(`symbol '${symbol.text}'`);
  const type = "type" in node ? node.type : undefined;
  if (type) parts.push(`type '${type.text}'`);
  if (!parts.length) return "";
  return ` ${style("dim", `[${parts.join(", ")}]`)}`;
}

function printAst(node, depth = 0) {
  console.log(
    `${"  ".repeat(depth)}${style("cyan", S.ASTKind[node.kind])}${semanticsOf(node)}`,
  );
  for (const child of S.children(node)) printAst(child, depth + 1);
}

const kindName = (kinds, kind) => kinds[kind].replace(/^k/, "");

function fileOf(parser, location) {
  return Token.from(location, parser)?.getLocation()?.fileName;
}

function printSymbols(parser, scope, path) {
  for (const symbol of scope.members) {
    if (!symbol || fileOf(parser, symbol.location) !== path) continue;
    const type = symbol.type ? ` : ${symbol.type.text}` : "";
    console.log(
      `${style("cyan", kindName(S.SymbolKind, symbol.kind))} ${symbol.text}${style("dim", type)}`,
    );
  }
}

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options,
  });

  if (values.help || !positionals.length) {
    console.log(usage);
    return values.help ? 0 : 1;
  }

  const wasm = await readFile(
    fileURLToPath(import.meta.resolve("cxx-frontend/wasm")),
  );
  await loadCxx({ wasm });

  let failed = 0;

  for (const path of positionals) {
    const sources = new SourceCache();
    const source = await sources.read(path);

    await using parser = await Parser.parse({
      path,
      source,
      appdir: values.appdir,
      sysroot: values.sysroot,
      std: values.std,
      systemIncludePaths: values.include,
      defines: values.define,
      exists: (fileName) => sources.exists(fileName),
      readFile: async (fileName) => await sources.read(fileName),
    });

    if (printDiagnostics(sources, parser.diagnostics)) {
      ++failed;
      continue;
    }

    const { ast, globalScope } = parser.model;

    if (
      (values.ast || values.symbols || values.trace) &&
      positionals.length > 1
    )
      console.log(style("bold", `// ${path}`));

    if (values.ast) {
      console.log(style("cyan", S.ASTKind[ast.kind]));
      for (const declaration of ast.declarationList)
        if (declaration?.startLocation?.fileName === path)
          printAst(declaration, 1);
    }

    if (values.symbols) printSymbols(parser, globalScope, path);

    if (values.trace) {
      const emitter = new TraceEmitter();
      parser.emitWith(emitter);
      console.log(emitter.trace);
    }
  }

  return failed ? 1 : 0;
}

process.exitCode = await main();
