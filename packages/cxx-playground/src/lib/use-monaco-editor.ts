import * as React from "react"
import * as monaco from "monaco-editor"

export function useMonacoEditor(
  model: monaco.editor.ITextModel,
  options?: monaco.editor.IStandaloneEditorConstructionOptions,
  onCreate?: (editor: monaco.editor.IStandaloneCodeEditor) => monaco.IDisposable
) {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const optionsRef = React.useRef(options)
  const onCreateRef = React.useRef(onCreate)

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

    const subscription = onCreateRef.current?.(editor)
    return () => {
      subscription?.dispose()
      editor.dispose()
    }
  }, [model])

  return containerRef
}
