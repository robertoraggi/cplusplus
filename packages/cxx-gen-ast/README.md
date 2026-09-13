# cxx-gen-ast

Source-driven compiler code generation and read-only JavaScript reflection.

`npm run cxx-gen-ast` refreshes the semantic model, the AST supporting code, and the TypeScript API.

`npm run cxx-dump-model` prints the model, `--write` refreshes the snapshot, and `--check` verifies it is current.

The cxx-gen-ast requires cxx-frontend, it is a bit of a circular dependency, build it with:

```bash
npm ci
npm run build:cxx-frontend
```
