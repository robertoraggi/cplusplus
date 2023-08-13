import { Parser, Preprocessor } from "../dist/index.js";
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
`;

async function main() {
  const wasmBinaryFile = fileURLToPath(Parser.DEFAULT_WASM_BINARY_URL);

  const wasmBinary = await readFile(wasmBinaryFile);

  // initialize the parser
  await Parser.init({ wasmBinary });

  const preprocessor = new Preprocessor();

  preprocessor.defineMacro("DEBUG", "1");
  preprocessor.defineMacro("__unix__", "1");

  const code = preprocessor.preprocess(source, "source.cc");

  console.log(code);

  preprocessor.dispose();
}

main().catch(console.error);
