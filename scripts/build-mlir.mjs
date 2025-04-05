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

import * as zx from "zx";
import { $ } from "zx";

$.verbose = true;

async function download({ pkg, version, outdir = "." }) {
  const baseUrl = `https://github.com/llvm/llvm-project/releases/download/llvmorg-${version}/`;
  const fileName = `${pkg}-${version}.src.tar.xz`;
  const exists = await zx.fs.exists(zx.path.join(outdir, fileName));
  if (!exists) {
    const url = `${baseUrl}${fileName}`;
    const response = await zx.fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download ${url}: ${response.statusText}`);
    }
    const payload = await response.arrayBuffer();
    await zx.fs.writeFile(zx.path.join(outdir, fileName), Buffer.from(payload));
  }

  // unpack
  if (!(await zx.fs.exists(zx.path.join(outdir, pkg)))) {
    await zx.fs.mkdir(zx.path.join(outdir, pkg), { recursive: true });
    await $`tar xf ${outdir}/${fileName} -C ${outdir}/${pkg} --strip-components=1`.quiet();
  }
}

async function downloadLLVM({ packages, version, outdir }) {
  for (const pkg of packages) {
    await download({ pkg, version, outdir });
  }
}

async function main() {
  const version = "20.1.2";
  const packages = ["cmake", "third-party", "llvm", "mlir"];

  const llvm_source_dir = zx.path.resolve(
    zx.path.join("build.em", "llvm-project")
  );

  const llvm_build_dir = zx.path.resolve(
    zx.path.join("build.em", "llvm-project", "build")
  );

  const llvm_install_dir = zx.path.resolve(
    zx.path.join("build.em", "llvm-project", "install")
  );

  const llvm_cmake_options = [
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_FLAGS=-DLLVM_BUILD_STATIC",
    "-DCMAKE_EXE_LINKER_FLAGS=-sNODERAWFS -sEXIT_RUNTIME -sALLOW_MEMORY_GROWTH",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DLLVM_BUILD_EXTERNAL_COMPILER_RT=OFF",
    "-DLLVM_BUILD_TOOLS=OFF",
    "-DLLVM_ENABLE_EH=OFF",
    "-DLLVM_ENABLE_FFI=OFF",
    "-DLLVM_ENABLE_PROJECTS=mlir",
    "-DLLVM_ENABLE_RTTI=OFF",
    "-DLLVM_ENABLE_RUNTIMES=",
    "-DLLVM_ENABLE_Z3_SOLVER=OFF",
    "-DLLVM_INCLUDE_DOCS=OFF",
    "-DLLVM_INCLUDE_TESTS=OFF",
    "-DLLVM_INSTALL_UTILS=OFF",
    "-DLLVM_LINK_LLVM_DYLIB=OFF",
    "-DLLVM_OPTIMIZED_TABLEGEN=OFF",
    "-DLLVM_TARGETS_TO_BUILD=WebAssembly",
  ];

  await zx.fs.mkdir(llvm_source_dir, { recursive: true });
  await zx.fs.mkdir(llvm_build_dir, { recursive: true });
  await zx.fs.mkdir(llvm_install_dir, { recursive: true });

  await zx.fs.mkdir(llvm_source_dir, { recursive: true });
  await zx.fs.writeFile(zx.path.join(llvm_source_dir, ".gitignore"), "*");

  await downloadLLVM({ version, packages, outdir: llvm_source_dir });

  await $`emcmake cmake ${llvm_cmake_options} -S ${llvm_source_dir}/llvm -B ${llvm_build_dir}/llvm -DCMAKE_INSTALL_PREFIX=${llvm_install_dir}`;

  await $`cmake --build ${llvm_build_dir}/llvm --target install`;

  // fixup installation

  const executables = [
    "llvm-tblgen",
    "mlir-pdll",
    "mlir-tblgen",
    "tblgen-to-irdl",
  ];

  // copy wasm payloads and create stubs for the executables
  for (const app of executables) {
    await zx.fs.copyFile(
      zx.path.join(llvm_build_dir, "llvm", "bin", app + ".wasm"),
      zx.path.join(llvm_install_dir, "bin", app + ".wasm")
    );

    await zx.fs.writeFile(
      zx.path.join(llvm_install_dir, "bin", app),
      `#!/usr/bin/env node\nrequire("./${app}.js");`
    );

    await zx.fs.chmod(zx.path.join(llvm_install_dir, "bin", app), 0o755);
  }

  // fixup
  const cmake_files = [
    zx.path.join(
      llvm_install_dir,
      "lib",
      "cmake",
      "llvm",
      "LLVMExports-release.cmake"
    ),

    zx.path.join(
      llvm_install_dir,
      "lib",
      "cmake",
      "mlir",
      "MLIRTargets-release.cmake"
    ),
  ];

  // replace .js with "" in the cmake files
  for (const file of cmake_files) {
    // replace .js with ""
    const content = await zx.fs.readFile(file, "utf8");
    const newContent = content.replace(/\.js/g, "");
    await zx.fs.writeFile(file, newContent);
  }
}

try {
  await main();
} catch (e) {
  console.error(e.message);
}
