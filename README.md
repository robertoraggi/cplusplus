# A compiler front end for the C++ language

cxx-frontend is a work-in-progress compiler frontend for C++26 and C23

The compiler frontend is designed to be a powerful tool for developers, enabling them to parse, analyze, and modify C++ source code.
This project aims to provide a robust foundation for building a complete C++ frontend, staying up-to-date with the latest language features and standards.

Playground showing the cxx compiler frontend in action at https://robertoraggi.github.io/cplusplus/

The API Reference is available at https://robertoraggi.github.io/cplusplus/docs/

## Build the cxx compiler and the MLIR based backed

```sh
uv sync && source .venv/bin/activate # optonal, for the lit-based unit tests

cmake --preset default-mlir
cmake --build build
ctest --test-dir build --progress
```

The cxx tool will be available at **./build/src/frontend/cxx**, and the default target is **wasm32-wasip1**.

```bash
./build/src/frontend/cxx --help
Usage: cxx [options] file...
Options:
  --help                       Display this information
  -D <macro>[=<val>]           Define a <macro> with <val> as its value. If just <macro> is given, <val> is taken to be 1
  -I <dir>                     Add <dir> to the end of the main include path
  -L <dir>                     Add <dir> to the end of the library path
  -U <macro>                   Undefine <macro>
  -std=<standard>              Assume that the input sources are for <standard>, one of 'c++14', 'c++17', 'c++20', 'c++23', 'c++26', or 'c23'
  --sysroot=<directory>        Use <directory> as the root directory for headers and libraries
...
```

## Inspect and transform MLIR

After an MLIR-enabled build, use cxx and cxx-mlir-opt to inspect and transform MLIR, for example:

Convert the C++ code to MLIR and lift control flow to structured control flow.

```sh
echo 'int f(int n) { return n < 10 ? 123 : 321; }' |
    ./build/src/frontend/cxx -emit-ir -xc++ -std=c++26 - |
    ./build/src/mlir/opt/cxx-mlir-opt --cxx-lift-cf-to-scf
```

Compile to MLIR, lift control flow to structured control flow, convert back to control flow, and lower to LLVM IR:

```sh
echo 'int f(int n) { return n < 10 ? 123 : 321; }' |
    ./build/src/frontend/cxx -emit-ir -xc++ -std=c++26 - |
    ./build/src/mlir/opt/cxx-mlir-opt --cxx-lift-cf-to-scf |
    ./build/src/mlir/opt/cxx-mlir-opt --convert-scf-to-cf --cxx-to-llvm |
    mlir-translate --mlir-to-llvmir
```

## Stage 2 bootstrap (on macOS)

```bash
cmake --workflow default-mlir
cmake --workflow stage1-macos
cmake --workflow stage2-macos

ls -la build-stage*/src/frontend/cxx
```

## Build the NPM package and the Playground

The playground uses the Monaco Editor to demonstrate how to use the compiler frontend LSP implementation,
and the MLIR based codegen pipeline.

https://robertoraggi.github.io/cplusplus/

```bash
# set EMSCRIPTEN_ROOT to the emscripten root, e.g. on macOS with emscripten installed via homebrew
export EMSCRIPTEN_ROOT=/opt/homebrew/opt/emscripten/libexec/

uv sync
source .venv/bin/activate

npm ci
npm run download-mlir
npm run build:cxx-frontend
npm run build:playground-sysroot
npm run playground
```

## License

Copyright (c) 2026 Roberto Raggi roberto.raggi@gmail.com

Licensed under the [MIT](LICENSE) license.
