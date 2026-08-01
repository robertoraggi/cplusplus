import * as React from "react"
import * as monaco from "monaco-editor"

export function useMonacoEditor(
  model: monaco.editor.ITextModel,
  options?: monaco.editor.IStandaloneEditorConstructionOptions
) {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const optionsRef = React.useRef(options)

  React.useEffect(() => {
    const domElement = containerRef.current
    if (!domElement) return

    const editor = monaco.editor.create(domElement, {
      model,
      automaticLayout: true,
      theme: "vs-dark",
      lineNumbersMinChars: 3,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      padding: { top: 12, bottom: 12 },
      folding: true,
      smoothScrolling: true,
      ...optionsRef.current,
    })

    return () => editor.dispose()
  }, [model])

  return containerRef
}
