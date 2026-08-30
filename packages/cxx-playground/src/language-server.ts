import type * as monaco from "monaco-editor"
import { LspClient } from "./lib/lsp-client"
import { inputCodeModel } from "./input-code-model"
import LanguageServerWorker from "./lsp.worker?worker"

export const TextOutputCodeFormat = ["cxxir", "mlir", "llvm", "asm"] as const
export type TextOutputCodeFormat = (typeof TextOutputCodeFormat)[number]

interface EmitCodeResult {
  format: TextOutputCodeFormat
  text: string
}

let client: LspClient | undefined

export function startLanguageServer(): LspClient {
  client ??= new LspClient(new LanguageServerWorker())
  return client
}

export async function emitCode({
  format,
  debugInfo,
}: {
  format: TextOutputCodeFormat
  debugInfo: boolean
}): Promise<string> {
  const result = await startLanguageServer().sendRequest<EmitCodeResult | null>(
    "cxx/emitCode",
    {
      textDocument: { uri: documentUri(inputCodeModel) },
      format,
      debugInfo,
    }
  )

  return result?.text ?? ""
}

function documentUri(model: monaco.editor.ITextModel): string {
  return model.uri.toString(true).toLowerCase()
}
