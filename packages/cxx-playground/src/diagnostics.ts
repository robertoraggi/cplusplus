import * as monaco from "monaco-editor"
import type { Diagnostic } from "cxx-frontend"

export function applyDiagnostics(
  model: monaco.editor.ITextModel,
  diagnostics: Diagnostic[]
) {
  const markers: monaco.editor.IMarkerData[] = diagnostics.map(
    ({ startLine, startColumn, endLine, endColumn, message }) => ({
      severity: monaco.MarkerSeverity.Error,
      startLineNumber: startLine,
      startColumn,
      endLineNumber: endLine,
      endColumn,
      message,
      source: "C++ Compiler",
    })
  )
  monaco.editor.setModelMarkers(model, "cxx", markers)
}
