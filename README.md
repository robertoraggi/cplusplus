# cxx-frontend

A parser for C++20.

# Usage

```js
const { Parser, RecursiveASTVisitor, ASTKind } = require("cxx-frontend");

const source = `
int fact(int n) {
    if (n < 2) return 1;
    return n * fact(n - 1);
}

int main() {
    return fact(3);
}
`;

const parser = new Parser({ source, path: "fact.cc" });

parser.parse();

const diagnostics = parser.getDiagnostics();

if (diagnostics.length > 0) {
  console.log("diagnostics", diagnostics);
}

class DumpAST extends RecursiveASTVisitor {
  depth = 0;

  accept(ast) {
    if (ast) {
        const name = ASTKind[ast.getKind()];
        console.log(`${" ".repeat(this.depth * 2)}${name}`);
        ++this.depth;
        super.accept(ast);
        --this.depth;
    }
  }
}

parser.getAST().accept(new DumpAST());

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
