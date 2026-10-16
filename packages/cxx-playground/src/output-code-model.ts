import * as monaco from "monaco-editor"
import { registerMlirLanguage } from "./lib/mlir-language"
import { registerLlvmLanguage } from "./lib/llvm-language"
import { registerAsmLanguage } from "./lib/asm-language"
import { getIRFoldingRanges } from "./lib/ir-folding"

registerMlirLanguage()
registerLlvmLanguage()
registerAsmLanguage()

for (const language of ["mlir", "llvm"]) {
  monaco.languages.registerFoldingRangeProvider(language, {
    provideFoldingRanges(model) {
      return getIRFoldingRanges(model)
        .sort(
          (a, b) =>
            a.start - b.start ||
            Number(!!b.declarations) - Number(!!a.declarations)
        )
        .map(({ start, end, declarations }) => ({
          start,
          end,
          kind: declarations
            ? monaco.languages.FoldingRangeKind.Imports
            : undefined,
        }))
    },
  })
}

export const outputCodeModel = monaco.editor.createModel(
  "",
  "mlir",
  monaco.Uri.parse("file:///main.mlir")
)
