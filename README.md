# A compiler front end for the C++ language

cxx-frontend is a work-in-progress compiler frontend for C++23

The compiler frontend is designed to be a powerful tool for developers, enabling them to parse, analyze, and modify C++ source code. This project aims to provide a robust foundation for building a complete C++ frontend, staying
up-to-date with the latest language features and standards.

The API Reference is available at https://robertoraggi.github.io/cplusplus/docs/

# Changelog and What's New

For updates, improvements, and recent features in cxx-frontend, please consult the [Changelog](CHANGELOG.md).

# Key Features

- **Syntax Analysis**: APIs to scan, preprocess, parse, and inspect the syntax of source code, making it a versatile tool for various code analysis tasks.

- **Multi-Language Support**: In addition to C++, the library provides APIs for TypeScript and JavaScript.

- **C++-23 Support**: Latest language enhancements, syntax, and features (WIP).

## Syntax Checker and AST Browser Showcase

Storybook and CodeMirror are used to demonstrate how to create a syntax checker and navigate the Abstract Syntax Tree (AST)

https://robertoraggi.github.io/cplusplus/

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

## Build the npm package (requires docker)

```sh
# prepare the package
npm ci

# compile WASM and TypeScript code
npm run build:cxx-frontend
```

## Build for WASM/WASI (requires docker)

```sh
npm ci
npm run build:wasi

# run the C++ front end CLI tool using wasmtime
wasmtime \
  --mapdir=/::build.wasi/install \
  --mapdir tests::tests \
  build.wasi/install/usr/bin/cxx.wasm -- \
  tests/manual/source.cc -ast-dump
```

## Native Build and CLI tools

On Linux, macOS and Windows:

```sh
# install the python packages required to run the unit tests (optional)
pip install -r tests/unit_tests/requirements.txt

# configure the source code
cmake . \
 -G Ninja \
 -B build \
 -DCMAKE_BUILD_TYPE=Release \
 -DCXX_INTERPROCEDURAL_OPTIMIZATION=1

# build
cmake --build build

# run the unit tests
cd build
ctest --progress
```

## Serialize the AST

Use `-emit-ast` to serialize the AST of a C++ program to a flatbuffer binary file

```sh
# serialize the AST
$ ./build/src/frontend/cxx -emit-ast source.cc -o source.ast
```

You can use any flatbuffers supported decoder to read the AST, e.g.

```sh
# Use flatc to dump the AST to JSON
$ ./build/_deps/flatbuffers-build/flatc --raw-binary -t build/src/parser/cxx/ast.bfbs  -- source.ast

$ ll source.*
source.ast source.cc source.json
```

## License

Copyright (c) 2023 Roberto Raggi roberto.raggi@gmail.com

Licensed under the [MIT](LICENSE) license.
