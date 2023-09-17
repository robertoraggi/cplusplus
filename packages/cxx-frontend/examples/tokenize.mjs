//
// tokenize.mjs
//

import {
  DEFAULT_WASM_BINARY_URL,
  Parser,
  Lexer,
  TokenKind,
} from "../dist/index.js";
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
  const wasmBinaryFile = fileURLToPath(DEFAULT_WASM_BINARY_URL);

  const wasmBinary = await readFile(wasmBinaryFile);

  // initialize the parser
  await Parser.init({ wasmBinary });

  const lexer = new Lexer(source);

  const tokens = [];

  while (true) {
    const token = lexer.next();

    const kind = TokenKind[token];
    const start = lexer.tokenOffset;
    const end = start + lexer.tokenLength;
    const text = source.slice(start, end);

    tokens.push({ text, kind, start, end });

    if (token === TokenKind.EOF_SYMBOL) {
      break;
    }
  }

  console.table(tokens);

  lexer.dispose();
}

main().catch(console.error);
