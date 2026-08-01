import * as monaco from "monaco-editor"
import { registerMlirLanguage } from "./lib/mlir-language"
import { registerLlvmLanguage } from "./lib/llvm-language"
import { registerAsmLanguage } from "./lib/asm-language"

registerMlirLanguage()
registerLlvmLanguage()
registerAsmLanguage()

export const outputCodeModel = monaco.editor.createModel(
  "",
  "mlir",
  monaco.Uri.parse("file:///main.mlir")
)
