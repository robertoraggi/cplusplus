# A compiler front end for the C++ language

This repository contains a _work in progress_ compiler front end for C++ 20.

## Install

For the latest stable version:

```
npm install -g cxx-frontend
```

## Build

On Linux, macOS and Windows:

```sh
# configure cxx-frontend
cmake . \
 -G Ninja \
 -B build \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=1

# build cxx-frontend
cmake --build build
```

## Dump the AST

Use `-ast-dump` to dump the AST of a C++ program.

```sh
echo 'int main() { auto f = []{ return 1; }; return f(); }' |
  ./build/cxx-frontend  -ast-dump -
```

## Build the npm package

```sh
cd packages/cxx-frontend

# prepare the package
npm ci

# compile WASM and TypeScript code
npm run build

# build the package
npm pack
```

## Use the JavaScript API

```js
//
// example.mjs
//

import { Parser, AST, ASTKind } from "cxx-frontend";
import { readFile } from "fs/promises";
import { fileURLToPath } from "url";

const source = `
template <typename T>
concept CanAdd = requires(T n) {
  n + n;
};

auto twice(CanAdd auto n) {
  return n + n;
}

int main() {
  return twice(2);
}
`;

async function main() {
  const wasmBinaryFile = fileURLToPath(Parser.DEFAULT_WASM_BINARY_URL);

  const wasmBinary = await readFile(wasmBinaryFile);

  // initialize the parser
  await Parser.init({ wasmBinary });

  const parser = new Parser({ source, path: "source.cc" });

  parser.parse();

  const diagnostics = parser.getDiagnostics();

  if (diagnostics.length > 0) {
    console.log("diagnostics", diagnostics);
  }

  const ast = parser.getAST();

  if (ast) {
    ast.walk().preVisit((node, depth) => {
      if (node instanceof AST) {
        const ind = " ".repeat(depth * 2);
        const kind = ASTKind[node.getKind()];
        console.log(`${ind}${kind}`);
      }
    });
  }

  parser.dispose();
}

main().catch(console.error);
```
