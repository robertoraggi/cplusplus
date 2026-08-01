import * as monaco from "monaco-editor"
import { defaultSample } from "./samples"

export const inputCodeModel = monaco.editor.createModel(
  defaultSample?.code ?? "",
  "cpp",
  monaco.Uri.parse("file:///main.cc")
)
