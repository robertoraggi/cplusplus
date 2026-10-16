import { outputCodeModel } from "./output-code-model"
import { useMonacoEditor } from "./lib/use-monaco-editor"

export default function OutputCode() {
  const containerRef = useMonacoEditor(
    outputCodeModel,
    {
      readOnly: true,
      cursorBlinking: "solid",
      minimap: { enabled: true },
      showFoldingControls: "always",
      foldingImportsByDefault: false,
    },
    (editor) => {
      const foldDeclarations = async () => {
        const version = outputCodeModel.getVersionId()
        await editor.getAction("editor.unfoldAll")?.run()
        if (
          editor.getModel() !== outputCodeModel ||
          outputCodeModel.getVersionId() !== version ||
          !["mlir", "llvm"].includes(outputCodeModel.getLanguageId())
        )
          return
        await editor.getAction("editor.toggleImportFold")?.run()
      }
      foldDeclarations()
      return editor.onDidChangeModelContent(foldDeclarations)
    }
  )

  return <div ref={containerRef} className="min-h-0 w-full flex-1" />
}
