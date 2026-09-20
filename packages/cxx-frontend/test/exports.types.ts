import loadCxx, {
  Parser,
  type Diagnostic,
  type WasmSource,
} from "cxx-frontend";
import parse from "cxx-frontend/parse";
import * as model from "cxx-frontend/model";
import { walk, type Visitor } from "cxx-frontend/traverse";
import { LanguageServer } from "cxx-frontend/lsp";

declare const wasm: WasmSource;

export async function main(): Promise<void> {
  await loadCxx({ wasm });

  await using parser: Parser = await parse({ path: "/x.cc", source: "int x;" });
  const diagnostics: ReadonlyArray<Diagnostic> = parser.diagnostics;
  void diagnostics;

  const unit: model.UnitAST = parser.model.ast;
  const scope: model.ScopeSymbol = parser.model.globalScope;
  void scope;

  for (const path of walk(unit))
    if (path.isFunctionDefinition()) void path.node.functionBody;

  const visitor: Visitor = {
    NamespaceDefinition(path) {
      void path.node.identifier?.name;
    },
  };
  void visitor;

  void LanguageServer;
}
