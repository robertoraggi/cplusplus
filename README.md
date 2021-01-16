# cxx-frontend

A parser for C++20.

# Usage

```js
const { parse } = require("cxx-frontend");

const path = "main.cc";

const source = `
int main() {
    if (x = 0 return 1;
}
`;

const result = parse({ path, source });

console.log(result);
```

# Build the library

```sh
mkdir -p build.em

# configure the project
emcmake cmake -Bbuild.em \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CROSSCOMPILING_EMULATOR=$(which node) .

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
