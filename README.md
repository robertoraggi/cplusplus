# A Compiler Frontend for C++

This repository contains a _work in progress_ compiler frontend for C++ 20.

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
// fact.mjs
//

import { Parser, RecursiveASTVisitor, ASTKind } from "cxx-frontend";
import { readFile } from "fs/promises";
import { fileURLToPath } from "url";

class DumpAST extends RecursiveASTVisitor {
  #stack = [];

  accept(node) {
    if (!node) return;

    const nodeKind = ASTKind[node.getKind()];

    const entry = [nodeKind];

    const parentNode = this.#stack.slice(-1)[0];

    parentNode?.push(entry);

    this.#stack.push(entry);

    super.accept(node);

    this.#stack.pop();

    return entry;
  }
}

const source = `
int fact(int n) {
    if (n < 2) return 1;
    return n * fact(n - 1);
}

int main() {
    return fact(3);
}
`;

async function main() {
  const wasmBinaryFile = fileURLToPath(Parser.DEFAULT_WASM_BINARY_URL);

  const wasmBinary = await readFile(wasmBinaryFile);

  // initialize the parser
  await Parser.init({ wasmBinary });

  const parser = new Parser({ source, path: "fact.cc" });

  parser.parse();

  const diagnostics = parser.getDiagnostics();

  if (diagnostics.length > 0) {
    console.log("diagnostics", diagnostics);
  }

  const ast = parser.getAST();

  const tree = new DumpAST().accept(ast);

  parser.dispose();

  console.log(tree);
}

main().catch(console.error);
```
