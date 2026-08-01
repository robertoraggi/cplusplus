import { inputCodeModel } from "./input-code-model"
import { useMonacoEditor } from "./lib/use-monaco-editor"

export default function CodeEditor() {
  const containerRef = useMonacoEditor(inputCodeModel, {
    renderLineHighlight: "all",
    cursorBlinking: "smooth",
    cursorSmoothCaretAnimation: "on",
  })

  return <div ref={containerRef} className="min-h-0 w-full flex-1" />
}
