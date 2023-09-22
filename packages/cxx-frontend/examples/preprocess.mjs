import {
  DEFAULT_WASM_BINARY_URL,
  Parser,
  Preprocessor,
} from "../dist/esm/index.js";
import { readFile } from "fs/promises";
import { fileURLToPath } from "url";

const source = `
// this comment will be exluded

#define DECL_F(result, name, prefix, suffix) result prefix##name##suffix

#define EXIT_VALUE 0

#if 0

this line will be excluded

#else

DECL_F(int, main, __wasm, __)()
{
  return EXIT_VALUE;
}

#if __unix__
constexpr bool unix_target = true;
#else
constexpr bool unix_target = false;
#endif

#endif

#include "iostream"
`;

async function main() {
  const wasmBinaryFile = fileURLToPath(DEFAULT_WASM_BINARY_URL);

  const wasmBinary = await readFile(wasmBinaryFile);

  // initialize the parser
  await Parser.init({ wasm: wasmBinary });

  const preprocessor = new Preprocessor({
    systemIncludePaths: ["/usr/include"],

    fs: {
      existsSync: (path) => path === "/usr/include/iostream",

      readFileSync: (path) =>
        path === "/usr/include/iostream" ? "namespace std {}" : "",
    },
  });

  preprocessor.defineMacro("DEBUG", "1");
  preprocessor.defineMacro("__unix__", "1");

  const code = preprocessor.preprocess(source, "source.cc");

  console.log(code);

  preprocessor.dispose();
}

main().catch(console.error);
