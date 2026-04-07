import { fileURLToPath } from "node:url";
import path from "node:path";
import { $, fs } from "zx";

$.verbose = true;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  // configure cmake using the emscripten presets
  await $`cmake --preset emscripten-mlir`;
  await $`cmake --build --preset build-emscripten`;

  // make sure packages/cxx-frontend/dist/wasm exists
  await fs.promises.mkdir(
    path.join(__dirname, "../packages/cxx-frontend/dist/wasm/"),
    { recursive: true },
  );

  // copy build.em/src/js/cxx-js.js to packages/cxx-frontend/src/
  await $`cp build.em/src/js/cxx-js.js ${path.join(__dirname, "../packages/cxx-frontend/dist/")}`;

  // copy build.em/src/js/cxx-js.wasm to packages/cxx-frontend/dist/wasm
  await $`cp build.em/src/js/cxx-js.wasm ${path.join(__dirname, "../packages/cxx-frontend/dist/wasm/")}`;
}

await main();
