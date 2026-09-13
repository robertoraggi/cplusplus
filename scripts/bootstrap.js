#!/usr/bin/env zx

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

import { $, argv, chalk, echo, fs, glob, which, ProcessOutput } from "zx";
import { fileURLToPath } from "node:url";
import { availableParallelism, tmpdir } from "node:os";
import path from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const workspacePath = path.resolve(__dirname, "..");

$.cwd = workspacePath;
$.quiet = true;

const usage = `
Usage: node scripts/bootstrap.js [options] [-- extra cxx args]

Compiles the cxx frontend with cxx itself, links the resulting wasm module and
runs it on a preprocessed translation unit.

Options:
  --cxx=<path>          host cxx driver (default: cxx from PATH)
  --wasmtime[=<path>]   run the wasm with wasmtime instead of the node wasi runtime
  --build-dir=<path>    object and wasm output directory (default: build-bootstrap)
  --std=<std>           language standard (default: c++26)
  --opt=<flag>          optimization flag (default: -O3)
  --jobs=<n>            parallel compile jobs (default: number of cpus)
  --stack-size=<bytes>  linker stack size (default: 131072)
  --only=<substring>    compile only the sources whose path contains <substring>
  --check=<source>      source to syntax check (default: #include <iostream>)
  --clean               remove the build directory before compiling
  --syntax-only         compile to /dev/null, skip the link and run steps
  --skip-compile        reuse the objects already in the build directory
  --skip-link           reuse the wasm already in the build directory
  --skip-run            stop after linking
  --keep-going          report every compile failure instead of the first batch
  --verbose             echo every command line
  --help                show this message
`;

if (argv.help) {
  echo(usage.trim());
  process.exit(0);
}

const sourcePatterns = [
  "src/parser/cxx/*.cc",
  "src/codegen/cxx/codegen/*.cc",
  "src/lsp/cxx/lsp/*.cc",
  "src/frontend/cxx/*.cc",
  "build/_deps/simdjson-src/simdjson.cpp",
];

const includePaths = [
  "src/parser",
  "src/codegen",
  "src/lsp",
  "src/frontend",
  "build/_deps/simdjson-src",
];

const buildDir = String(argv["build-dir"] ?? "build-bootstrap");
const std = String(argv.std ?? "c++26");
const opt = String(argv.opt ?? "-O3");
const jobs = Number(argv.jobs ?? availableParallelism());
const stackSize = Number(argv["stack-size"] ?? 128 * 1024);
const syntaxOnly = Boolean(argv["syntax-only"]);
const extraArgs = argv._.map(String);

const version = JSON.parse(
  await fs.readFile(path.join(workspacePath, "package.json"), "utf8"),
).version;

const cxx = argv.cxx ? path.resolve(String(argv.cxx)) : await which("cxx");

const compileFlags = [
  `-std=${std}`,
  opt,
  ...includePaths.flatMap((dir) => ["-I", dir]),
  "-D_WASI_EMULATED_MMAN",
  `-DCXX_VERSION="${version}"`,
  ...extraArgs,
];

function formatDuration(startedAt) {
  return `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
}

function objectFileFor(source) {
  return path.join(buildDir, source.replace(/\.(cc|cpp)$/, ".o"));
}

async function collectSources() {
  const sources = await glob(sourcePatterns, { cwd: workspacePath });
  const only = argv.only ? String(argv.only) : undefined;
  const selected = only
    ? sources.filter((source) => source.includes(only))
    : sources;
  return selected.sort();
}

async function runPool(items, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.max(1, jobs) }, async () => {
    for (;;) {
      const index = next++;
      if (index >= items.length) break;
      results[index] = await worker(items[index], index);
    }
  });
  await Promise.all(runners);
  return results;
}

async function compile(sources) {
  const startedAt = Date.now();
  let done = 0;
  const failures = [];

  const directories = new Set(
    sources.map((source) => path.dirname(objectFileFor(source))),
  );

  if (!syntaxOnly) {
    for (const directory of directories) {
      await fs.mkdirp(path.join(workspacePath, directory));
    }
  }

  await runPool(sources, async (source) => {
    try {
      if (syntaxOnly) {
        await $`${cxx} -fsyntax-only ${compileFlags} ${source}`;
      } else {
        const output = objectFileFor(source);
        await $`${cxx} -c ${compileFlags} ${source} -o ${output}`;
      }
      echo(chalk.green(`[${++done}/${sources.length}] compiled ${source}`));
    } catch (error) {
      ++done;
      failures.push({ source, error });
      echo(chalk.red(`[${done}/${sources.length}] FAILED ${source}`));
      if (error instanceof ProcessOutput) {
        echo(error.stderr.trimEnd() || error.stdout.trimEnd());
      }
      if (!argv["keep-going"]) throw error;
    }
  }).catch((error) => {
    if (!(error instanceof ProcessOutput)) throw error;
  });

  echo(
    chalk.bold(
      `compiled ${sources.length - failures.length}/${sources.length} sources in ${formatDuration(startedAt)}`,
    ),
  );

  if (failures.length) {
    echo(chalk.red(`${failures.length} source(s) failed to compile`));
    for (const { source } of failures) echo(chalk.red(`  ${source}`));
    process.exit(1);
  }
}

async function link(sources) {
  const startedAt = Date.now();
  const objects = sources.map(objectFileFor);
  const missing = [];

  for (const object of objects) {
    if (!(await fs.pathExists(path.join(workspacePath, object)))) {
      missing.push(object);
    }
  }

  if (missing.length) {
    echo(chalk.red(`missing ${missing.length} object file(s):`));
    for (const object of missing) echo(chalk.red(`  ${object}`));
    process.exit(1);
  }

  const wasm = path.join(buildDir, "cxx.wasm");

  await $`${cxx} -Wl,-z,stack-size=${stackSize} -o ${wasm} ${objects}`;

  const { size } = await fs.stat(path.join(workspacePath, wasm));

  echo(
    chalk.bold(
      `linked ${wasm} (${(size / (1024 * 1024)).toFixed(1)} MiB) in ${formatDuration(startedAt)}`,
    ),
  );

  return wasm;
}

function silenceWasiExperimentalWarning() {
  const emitWarning = process.emitWarning;
  process.emitWarning = (warning, ...rest) => {
    if (String(warning).includes("WASI is an experimental feature")) return;
    return emitWarning.call(process, warning, ...rest);
  };
}

async function runWithNodeWasi(wasm, args, input) {
  silenceWasiExperimentalWarning();

  const { WASI } = await import("node:wasi");

  const stdinPath = path.join(tmpdir(), `cxx-bootstrap-${process.pid}.ii`);

  await fs.writeFile(stdinPath, input);

  const stdin = await fs.open(stdinPath, "r");

  try {
    const wasi = new WASI({
      version: "preview1",
      args: [path.basename(wasm), ...args],
      stdin,
      returnOnExit: true,
    });

    const module = await WebAssembly.compile(
      await fs.readFile(path.join(workspacePath, wasm)),
    );

    const instance = await WebAssembly.instantiate(
      module,
      wasi.getImportObject(),
    );

    return wasi.start(instance);
  } finally {
    await fs.close(stdin);
    await fs.remove(stdinPath);
  }
}

async function runWithWasmtime(wasm, args, input) {
  const wasmtime =
    typeof argv.wasmtime === "string"
      ? path.resolve(argv.wasmtime)
      : await which("wasmtime");

  const result = await $({ input, nothrow: true })`${wasmtime} ${wasm} ${args}`;

  if (result.stdout.trim()) echo(result.stdout.trimEnd());
  if (result.stderr.trim()) echo(result.stderr.trimEnd());

  return result.exitCode ?? 0;
}

async function run(wasm) {
  const startedAt = Date.now();

  const source = String(argv.check ?? "#include <iostream>\n");

  const preprocessed = await $({
    input: source,
  })`${cxx} -E -xc++ -std=${std} -`.text();

  const args = ["-fsyntax-only", "-"];

  const runtime = argv.wasmtime ? "wasmtime" : "node wasi";

  const exitCode = argv.wasmtime
    ? await runWithWasmtime(wasm, args, preprocessed)
    : await runWithNodeWasi(wasm, args, preprocessed);

  echo(
    chalk.bold(
      `ran ${wasm} on ${runtime} over ${preprocessed.split("\n").length} preprocessed lines in ${formatDuration(startedAt)}`,
    ),
  );

  if (exitCode !== 0) {
    echo(chalk.red(`${wasm} exited with status ${exitCode}`));
    process.exit(exitCode);
  }
}

async function main() {
  if (argv.verbose) {
    $.quiet = false;
    $.verbose = true;
  }

  if (argv.clean) {
    await fs.remove(path.join(workspacePath, buildDir));
  }

  const sources = await collectSources();

  if (!sources.length) {
    echo(chalk.red("no sources selected"));
    process.exit(1);
  }

  echo(chalk.bold(`cxx        ${cxx}`));
  echo(chalk.bold(`sources    ${sources.length}`));
  echo(chalk.bold(`jobs       ${jobs}`));
  echo(chalk.bold(`build dir  ${buildDir}`));

  if (!argv["skip-compile"]) await compile(sources);

  if (syntaxOnly) return;

  const wasm = argv["skip-link"]
    ? path.join(buildDir, "cxx.wasm")
    : await link(sources);

  if (argv["skip-run"]) return;

  await run(wasm);

  echo(chalk.green.bold("bootstrap ok"));
}

main().catch((error) => {
  if (error instanceof ProcessOutput) {
    echo(chalk.red(error.stderr.trimEnd() || error.stdout.trimEnd()));
    process.exit(error.exitCode ?? 1);
  }
  echo(chalk.red(String(error?.message ?? error)));
  process.exit(1);
});
