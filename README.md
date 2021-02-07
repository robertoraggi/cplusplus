# cxx-frontend

A parser for C++20.

# Usage

```js
const fs = require("fs");
const process = require("process");

const { Parser } = require("cxx-frontend");

const path = process.argv[2];
const source = fs.readFileSync(path).toString();

const parser = new Parser({ path, source });

parser.parse();

console.log("diagnostics", parser.getDiagnostics());

console.log("ast", parser.getAST());

parser.dispose();
```

# Build the library

```sh
mkdir -p build.em

# configure the project
emcmake cmake -Bbuild.em \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CROSSCOMPILING_EMULATOR=$(which node) \
    .

# build
cmake --build build.em
```

# Build the npm package

```sh
cd packages/cxx-frontend

# install the npm dependencies
npm install

# copy the cxx-js WASM library to the dist folder.
npm run copy

# compile the source code.
npm run build

# build the package.
npm pack
```
