# A compiler front end for the C++ language

cxx-frontend is a work-in-progress compiler frontend for C++26 and C23

The compiler frontend is designed to be a powerful tool for developers, enabling them to parse, analyze, and modify C++ source code. This project aims to provide a robust foundation for building a complete C++ frontend, staying
up-to-date with the latest language features and standards.

The API Reference is available at https://robertoraggi.github.io/cplusplus/docs/

# Changelog and What's New

For updates, improvements, and recent features in cxx-frontend, please consult the [Changelog](CHANGELOG.md).

# Key Features

- **Syntax Analysis**: APIs to scan, preprocess, parse, and inspect the syntax of source code, making it a versatile tool for various code analysis tasks.

- **Multi-Language Support**: In addition to C++, the library provides APIs for TypeScript and JavaScript.

- **C++-26 and C23 Support**: Latest language enhancements, syntax, and features (WIP).

## Playground

The playground uses the Monaco Editor to demonstrate how to create a syntax checker and navigate the Abstract Syntax Tree (AST).

https://robertoraggi.github.io/cplusplus/

## Native Build and CLI tools

On Linux, macOS and Windows:

install the python packages required to run the unit tests (optional)

```sh
uv sync && source .venv/bin/activate
```

configure the source code

```sh
cmake --preset default
```

build

```sh
cmake --build build
```

run the unit tests

```sh
cd build
ctest --progress
```

Dump the AST to stdout

```sh
 ./build/src/frontend/cxx tests/manual/source.cc -ast-dump
```

## Build the npm package (requires docker)

prepare the package

```sh
npm ci
```

compile WASM and TypeScript code

```sh
npm run build:cxx-frontend
```

## Build for WASM/WASI (requires docker)

```sh
npm ci
npm run build:wasi
```

run the C++ front end CLI tool using wasmtime

```sh
wasmtime \
  --mapdir=/::build.wasi/install \
  --mapdir tests::tests \
  build.wasi/install/usr/bin/cxx.wasm -- \
  tests/manual/source.cc -ast-dump
```

## Installing from npm

To integrate the latest stable version of the C++ Compiler Frontend bindings into your project, you can install them from npm:

```sh
npm install cxx-frontend
```

Once installed, you can use the bindings in your Node.js or web projects as needed.

## Getting Started Using Example Projects

These projects are pre-configured and serve as starting points for various [use cases](https://github.com/robertoraggi/cplusplus/tree/main/templates).

For Node.js

```sh
npx degit robertoraggi/cplusplus/templates/cxx-parse cxx-parse
cd cxx-parse
npm install
node .
```

For web-based applications, use these commands to clone, set up, and start a development server:

```sh
npx degit robertoraggi/cplusplus/templates/cxx-browser-esm-vite cxx-browser-esm-vite
cd cxx-browser-esm-vite
npm install
npm run dev
```

## License

Copyright (c) 2025 Roberto Raggi roberto.raggi@gmail.com

Licensed under the [MIT](LICENSE) license.
