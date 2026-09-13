# cxx-gen-ast

Source-driven compiler code generation and read-only JavaScript reflection.

`npm run cxx-gen-ast` refreshes `semantic-model.json` from the built JavaScript
frontend and then generates AST support and semantic wrappers from it, so a
change to a declaration in `src/parser/cxx/` reaches the generated code in one
step. The snapshot is generated, not checked in: when the frontend has not been
built the generator reuses the snapshot it finds and says so, and fails with
what to build when there is none. `--no-refresh` keeps it on purpose.
`npm run cxx-gen-model` uses the same snapshot for persistence. Neither command
extracts declarations from `ast.h`.

`npm run cxx-dump-model` prints the model, `--write` refreshes the snapshot
alone, and `--check` verifies it is current.

See [the semantic API design](../../docs/design/semantic-api.md) for build
steps, ownership, and coverage. Run generator tests with
`npm test -w @robertoraggi/cxx-gen-ast`.
