import { Parser, TokenKind, TranslationUnit, AST, ASTKind } from "../dist/index.js";
import { readFile } from "fs/promises";
import { fileURLToPath } from "url";

const source = `
int main() {
  return 0;
}
`;

async function main() {
  const wasmBinaryFile = fileURLToPath(Parser.DEFAULT_WASM_BINARY_URL);

  const wasmBinary = await readFile(wasmBinaryFile);

  // initialize the parser
  await Parser.init({ wasmBinary });

  const translationUnit = new TranslationUnit();

  translationUnit.preprocess(source, "main.cc");

  console.log("== Tokens:");

  for (const token of translationUnit.tokens()) {
    console.log(TokenKind[token.getKind()], token.getText());
  }

  const ast = translationUnit.parse();

  console.log();
  console.log("== AST:")

  ast?.walk().preVisit((node, depth) => {
    if (node instanceof AST) {
      const ind = " ".repeat(depth * 2);
      const kind = ASTKind[node.getKind()];
      console.log(`${ind}${kind}`);
    }
  });

  translationUnit.dispose();
}

main().catch(console.error);
