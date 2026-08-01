import { outputModel } from "./playground-store"
import { useMonacoEditor } from "./lib/use-monaco-editor"

export default function OutputCode() {
  const containerRef = useMonacoEditor(outputModel, {
    readOnly: true,
    cursorBlinking: "solid",
  })

  return <div ref={containerRef} className="min-h-0 w-full flex-1" />
}
