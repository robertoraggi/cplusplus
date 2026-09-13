import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { $ } from "zx";

$.verbose = true;

const scriptsDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = path.dirname(scriptsDirectory);
const buildDirectory = path.join(repositoryDirectory, "build.em");
const frontendDirectory = path.join(
  repositoryDirectory,
  "packages/cxx-frontend",
);
const sourceDirectory = path.join(frontendDirectory, "src");
const distributionDirectory = path.join(frontendDirectory, "dist");

$.cwd = repositoryDirectory;

function shouldSkipWasmOpt() {
  const value = process.env.CXX_NO_WASM_OPT;

  if (value === undefined || value === "" || value === "0") {
    return false;
  }

  if (value === "1") {
    return true;
  }

  throw new Error("CXX_NO_WASM_OPT must be 0 or 1");
}

async function configure() {
  let linkerFlags = "";

  if (shouldSkipWasmOpt()) {
    linkerFlags = "-O1 -sERROR_ON_WASM_CHANGES_AFTER_LINK";
  }

  await $`cmake --preset emscripten-mlir -DCMAKE_EXE_LINKER_FLAGS_RELEASE=${linkerFlags}`;
}

async function build() {
  await $`cmake --build --preset build-emscripten-mlir`;
}

async function installArtifacts() {
  const generatedDirectory = path.join(buildDirectory, "src/js");
  const generatedTypings = path.join(generatedDirectory, "cxx-js.d.ts");
  const sourceTypings = path.join(sourceDirectory, "cxx-js.d.ts");
  const wasmDirectory = path.join(distributionDirectory, "wasm");

  await mkdir(wasmDirectory, { recursive: true });
  await copyFile(generatedTypings, sourceTypings);
  await $`npm exec --workspace=cxx-frontend -- prettier --write ${sourceTypings}`;
  await Promise.all([
    copyFile(
      path.join(generatedDirectory, "cxx-js.js"),
      path.join(distributionDirectory, "cxx-js.js"),
    ),
    copyFile(
      path.join(generatedDirectory, "cxx-js.wasm"),
      path.join(wasmDirectory, "cxx-js.wasm"),
    ),
    copyFile(sourceTypings, path.join(distributionDirectory, "cxx-js.d.ts")),
  ]);
}

async function main() {
  await configure();
  await build();
  await installArtifacts();
}

await main();
