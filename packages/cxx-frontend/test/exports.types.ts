import loadCxx, {
  Linker,
  Parser,
  type Diagnostic,
  type WasmSource,
} from "cxx-frontend";
import parse from "cxx-frontend/parse";
import link from "cxx-frontend/link";
import * as model from "cxx-frontend/model";
import { walk, type Visitor } from "cxx-frontend/traverse";
import { LanguageServer } from "cxx-frontend/lsp";

declare const wasm: WasmSource;

export async function main(): Promise<void> {
  await loadCxx({ wasm });

  const readFile = async (_path: string): Promise<Uint8Array | undefined> =>
    undefined;
  const linked: Uint8Array = await link({
    sysroot: "/sysroot",
    readFile,
    objects: [{ name: "main.o", data: new Uint8Array() }],
  });
  await using linker: Linker = await Linker.create({
    readFile,
    libraryPaths: [],
  });
  const stripped: Uint8Array = await linker.link([], { strip: "all" });
  void linked;
  void stripped;

  await using parser: Parser = await parse({ path: "/x.cc", source: "int x;" });
  const diagnostics: ReadonlyArray<Diagnostic> = parser.diagnostics;
  void diagnostics;

  const unit: model.UnitAST = parser.ast;
  const scope: model.ScopeSymbol | undefined = unit.symbol;
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
