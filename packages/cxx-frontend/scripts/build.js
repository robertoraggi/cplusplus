// Copyright (c) 2025 Roberto Raggi <roberto.raggi@gmail.com>
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

import "zx/globals";

import * as esbuild from "esbuild";
import * as path from "path";

$.verbose = true;

// get the root directory of the project
const projectRootSourcePath = path.join(__dirname, "../../..");

const hasMLIR = fs.existsSync(
  path.join(
    projectRootSourcePath,
    "build.em/llvm-project/install/bin/llvm-tblgen.js",
  ),
);

if (hasMLIR) {
  console.log("building with MLIR support");
}

// get the source path of the cxx-frontend package
const cxxFrontendSourcePath = path.join(
  projectRootSourcePath,
  "packages/cxx-frontend",
);

// set the current working directory to the cxx-frontend source path
$.cwd = cxxFrontendSourcePath;

build().catch(console.error);

async function build() {
  const sdk = await detectEmsdk();

  if ($.env.EMSCRIPTEN_ROOT) {
    await emsdkBuildPresets();
  } else if (argv.docker === true || !sdk) {
    await dockerBuild();
  } else if ($.env.EMSCRIPTEN_ROOT) {
    await emsdkBuildPresets();
  } else {
    await emsdkBuild(sdk);
  }

  await $`cp ${projectRootSourcePath}/README.md .`;
  await $`cp ${projectRootSourcePath}/CHANGELOG.md .`;
  await $`cp ${projectRootSourcePath}/build.em/src/js/cxx-js.js src`;
  await $`mkdir -p dist/wasm/ dist/dts/`;
  await $`cp ${projectRootSourcePath}/build.em/src/js/cxx-js.wasm dist/wasm`;
  await $`cp ${cxxFrontendSourcePath}/src/*.d.ts dist/dts/`;
  await $`npm exec --package typescript tsc`;

  /** @type{esbuild.BuildOptions} */
  const buildOptions = {
    bundle: true,
    entryPoints: ["./src/index.ts"],
    sourcemap: true,
    packages: "external",
  };

  // build the cjs cxx-frontend package
  esbuild.build({
    ...buildOptions,
    outfile: "dist/cjs/index.cjs",
    format: "cjs",
    platform: "node",
  });

  // build the esm cxx-frontend package
  esbuild.build({
    ...buildOptions,
    outfile: "dist/esm/index.js",
    format: "esm",
    platform: "browser",
  });
}

async function dockerBuild() {
  const docker = await which("docker", { nothrow: true });
  if (!docker) throw new Error("docker not found");

  const emscriptenCacheDir = path.join(os.homedir(), ".emscripten-cache");

  const cmakeOptions = [
    "-DCMAKE_INSTALL_PREFIX=build.em/install/usr",
    "-DKWGEN_EXECUTABLE=/usr/bin/kwgen",
    "-DFLATBUFFERS_FLATC_EXECUTABLE=/usr/bin/flatc",
    "-S .",
    "-B build.em",
  ];

  if (argv.debug) {
    cmakeOptions.push("-DCMAKE_BUILD_TYPE=Debug");
  } else {
    cmakeOptions.push(
      "-DCMAKE_BUILD_TYPE=MinSizeRel",
      "-DCXX_INTERPROCEDURAL_OPTIMIZATION=ON",
    );
  }

  if (hasMLIR) {
    cmakeOptions.push(
      `-DCXX_ENABLE_MLIR=YES`,
      `-DLLVM_DIR=/code/build.em/llvm-project/install/lib/cmake/llvm`,
      `-DMLIR_DIR=/code/build.em/llvm-project/install/lib/cmake/mlir`,
    );
  }

  const user = await $`id -u`;

  await $`mkdir -p ${emscriptenCacheDir}`;
  await $`${docker} build -t cxx-emsdk -f ${projectRootSourcePath}/Dockerfile.emsdk ${projectRootSourcePath}`;
  await $`${docker} run -t --rm -u ${user} -v ${emscriptenCacheDir}:/emsdk/upstream/emscripten/cache/ cxx-emsdk embuilder.py build MINIMAL --lto=thin`;
  await $`${docker} run -t --rm -u ${user} -v ${emscriptenCacheDir}:/emsdk/upstream/emscripten/cache/ -v ${projectRootSourcePath}:/code -w /code cxx-emsdk emcmake cmake ${cmakeOptions}`;
  await $`${docker} run -t --rm -u ${user} -v ${emscriptenCacheDir}:/emsdk/upstream/emscripten/cache/ -v ${projectRootSourcePath}:/code -w /code cxx-emsdk cmake --build build.em`;
}

async function emsdkBuild({ cmake, emcmake, flatc, kwgen }) {
  const CMAKE_INTERPROCEDURAL_OPTIMIZATION =
    $.env.CMAKE_INTERPROCEDURAL_OPTIMIZATION ?? "ON";

  const cmakeOptions = [
    `-DCMAKE_INSTALL_PREFIX=${projectRootSourcePath}/build.em/install/usr`,
    `-DKWGEN_EXECUTABLE=${kwgen}`,
    `-DFLATBUFFERS_FLATC_EXECUTABLE=${flatc}`,
    `-S ${projectRootSourcePath}`,
    `-B ${projectRootSourcePath}/build.em`,
  ];

  if (hasMLIR) {
    cmakeOptions.push(
      `-DCXX_ENABLE_MLIR=YES`,
      `-DLLVM_DIR=${projectRootSourcePath}/build.em/llvm-project/install/lib/cmake/llvm`,
      `-DMLIR_DIR=${projectRootSourcePath}/build.em/llvm-project/install/lib/cmake/mlir`,
    );
  }

  if (argv.debug) {
    cmakeOptions.push(`-DCMAKE_BUILD_TYPE=Debug`);
  } else {
    cmakeOptions.push(
      `-DCMAKE_BUILD_TYPE=MinSizeRel`,
      `-DCXX_INTERPROCEDURAL_OPTIMIZATION=${CMAKE_INTERPROCEDURAL_OPTIMIZATION}`,
    );
  }

  await $`${emcmake} ${cmake} ${cmakeOptions}`;

  await $`${cmake} --build ${projectRootSourcePath}/build.em --target install`;
}

async function emsdkBuildPresets() {
  const cmake = await which("cmake", { nothrow: true });

  const preset = hasMLIR ? "emscripten-mlir" : "emscripten";

  await $`${cmake} -S ${projectRootSourcePath} --preset ${preset}`;

  await $`${cmake} --build ${projectRootSourcePath}/build.em --target install`;
}

async function detectEmsdk() {
  const cmake = await which("cmake", { nothrow: true });
  if (!cmake) return null;

  const emcmake = await which("emcmake", { nothrow: true });
  if (!emcmake) return null;

  const flatc =
    argv.flatc ??
    $.env.FLATBUFFERS_FLATC_EXECUTABLE ??
    (await which("flatc", { nothrow: true }));
  if (!flatc) return null;

  const kwgen =
    argv.kwgen ??
    $.env.KWGEN_EXECUTABLE ??
    (await which("kwgen", { nothrow: true }));
  if (!kwgen) return null;

  return {
    cmake,
    emcmake,
    flatc,
    kwgen,
  };
}
