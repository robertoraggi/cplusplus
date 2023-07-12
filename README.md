# A compiler front end for the C++ language

This repository contains a _work in progress_ compiler front end for C++ 20.

## Install

For the latest stable version of the JavaScript bindings:

```
npm install cxx-frontend
```

## Storybook

https://robertoraggi.github.io/cplusplus/

## Node.JS examples

- [Preprocess Source Files](./packages/cxx-frontend/examples/preprocess.mjs)
- [Tokenize Input](./packages/cxx-frontend/examples/tokenize.mjs)
- [Parse Translation Unit](./packages/cxx-frontend/examples/unit.mjs)

## JS Fiddle Playgrounds

- [Dump the Abstract Syntax Tree](https://jsfiddle.net/4x9yvw6s)
- [Check Syntax](https://jsfiddle.net/dfeLvy4a)

## Build

On Linux, macOS and Windows:

```sh
# install the python packages required to run the unit tests
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

## Build the npm package using docker

```sh
cd packages/cxx-frontend

# prepare the package
npm ci

# compile WASM and TypeScript code
npm run build

# build the package
npm pack
```

## Use the JavaScript API with Node.JS

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

  ast?.walk().preVisit((node, depth) => {
    if (node instanceof AST) {
      const ind = " ".repeat(depth * 2);
      const kind = ASTKind[node.getKind()];
      console.log(`${ind}${kind}`);
    }
  });

  parser.dispose();
}

main().catch(console.error);
```

## Use the JavaScript API in a web browser

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>C++ Playground</title>
  </head>
  <body>
    <script type="module">
      import {
        Parser,
        AST,
        ASTKind,
      } from "https://unpkg.com/cxx-frontend@latest/dist/index.js";

      const response = await fetch(Parser.DEFAULT_WASM_BINARY_URL);

      const wasmBinary = new Uint8Array(await response.arrayBuffer());

      await Parser.init({ wasmBinary });

      const source = `int main()\n{\n  return 0;\n}\n`;

      const parser = new Parser({
        path: "source.cc",
        source,
      });

      parser.parse();

      const rows = [];

      const ast = parser.getAST();

      ast?.walk().preVisit((node, depth) => {
        if (node instanceof AST)
          rows.push("  ".repeat(depth) + ASTKind[node.getKind()]);
      });

      parser.dispose();

      const sourceOutput = document.createElement("pre");
      sourceOutput.style.borderStyle = "solid";
      sourceOutput.innerText = source;
      document.body.appendChild(sourceOutput);

      const astOutput = document.createElement("pre");
      astOutput.style.borderStyle = "solid";
      astOutput.innerText = rows.join("\n");
      document.body.appendChild(astOutput);
    </script>
  </body>
</html>
```

## Release Notes

[Changelog](CHANGELOG.md)

## License

Copyright (c) 2023 Roberto Raggi roberto.raggi@gmail.com

Licensed under the [MIT](LICENSE) license.
