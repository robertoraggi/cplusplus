import { copyFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = (path) => fileURLToPath(new URL(path, import.meta.url));

const wasm = root("../dist/wasm/cxx-js.wasm");

if (!existsSync(wasm)) {
  console.error(`missing ${wasm}, run "npm run build:cxx-frontend" first`);
  process.exit(1);
}

for (const name of ["LICENSE", "README.md"]) {
  copyFileSync(root(`../../../${name}`), root(`../${name}`));
}
