import * as monaco from "monaco-editor"
import type {
  Diagnostic,
  DiagnosticNote,
  DiagnosticSeverity,
} from "cxx-frontend"

const markerSeverity: Record<DiagnosticSeverity, monaco.MarkerSeverity> = {
  message: monaco.MarkerSeverity.Info,
  note: monaco.MarkerSeverity.Hint,
  warning: monaco.MarkerSeverity.Warning,
  error: monaco.MarkerSeverity.Error,
  fatal: monaco.MarkerSeverity.Error,
}

function relatedInformation(
  model: monaco.editor.ITextModel,
  diagnosticFileName: string,
  note: DiagnosticNote
): monaco.editor.IRelatedInformation {
  return {
    resource:
      note.fileName === diagnosticFileName
        ? model.uri
        : monaco.Uri.file(note.fileName),
    message: note.message,
    startLineNumber: note.startLine,
    startColumn: note.startColumn,
    endLineNumber: note.endLine,
    endColumn: note.endColumn,
  }
}

export function applyDiagnostics(
  model: monaco.editor.ITextModel,
  diagnostics: Diagnostic[]
) {
  const markers: monaco.editor.IMarkerData[] = diagnostics.map(
    ({
      fileName,
      startLine,
      startColumn,
      endLine,
      endColumn,
      message,
      severity,
      notes,
    }) => ({
      severity: markerSeverity[severity],
      startLineNumber: startLine,
      startColumn,
      endLineNumber: endLine,
      endColumn,
      message,
      source: "C++ Compiler",
      relatedInformation: notes.map((note) =>
        relatedInformation(model, fileName, note)
      ),
    })
  )
  monaco.editor.setModelMarkers(model, "cxx", markers)
}
