// Copyright (c) 2023 Roberto Raggi <roberto.raggi@gmail.com>
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

import * as esbuild from "esbuild";
import * as path from "path";
import * as process from "process";
import { fileURLToPath } from "url";
import { $, ProcessOutput, chalk } from "zx";

const buildScriptPath = path.dirname(fileURLToPath(import.meta.url));

const projectRootSourcePath = path.join(buildScriptPath, "../../..");

const cxxFrontendSourcePath = path.join(
  projectRootSourcePath,
  "packages/cxx-frontend",
);

$.cwd = cxxFrontendSourcePath;

async function main() {
  await $`${projectRootSourcePath}/scripts/build-emscripten.sh`;
  await $`cp ${projectRootSourcePath}/README.md .`;
  await $`cp ${projectRootSourcePath}/CHANGELOG.md .`;
  await $`cp ${projectRootSourcePath}/build.em/src/js/cxx-js.js src`;
  await $`mkdir -p dist/wasm/ dist/dts/`;
  await $`cp ${projectRootSourcePath}/build.em/src/js/cxx-js.wasm dist/wasm`;
  await $`cp ${cxxFrontendSourcePath}/src/*.d.ts dist/dts/`;
  await $`npm exec --package-typescript tsc`;

  /** @type{esbuild.BuildOptions} */
  const buildOptions = {
    bundle: true,
    entryPoints: ["./src/index.ts"],
    sourcemap: true,
    packages: "external",
  };

  esbuild.build({
    ...buildOptions,
    outfile: "dist/cjs/index.cjs",
    format: "cjs",
    platform: "node",
    define: {
      "import.meta.url": "import_meta_url",
    },
    inject: [
      path.join(cxxFrontendSourcePath, "scripts/import.meta.url-polyfill.js"),
    ],
  });

  // build the esm cxx-frontend package
  esbuild.build({
    ...buildOptions,
    outfile: "dist/esm/index.js",
    format: "esm",
    platform: "browser",
  });
}

main().catch((err) => {
  if (err instanceof ProcessOutput) {
    console.error(err.message);
  } else {
    console.error(err);
  }
  process.exit(1);
});
