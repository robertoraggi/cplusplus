import type * as monaco from "monaco-editor"

export interface IRFoldingRange {
  start: number
  end: number
  declarations?: boolean
}

function isMLIRDeclaration(text: string) {
  if (/^[!#][\w.$-]+\s*=/.test(text)) return true
  if (/^(?:cxx\.global|memref\.global|llvm\.mlir\.global)\b/.test(text)) {
    return true
  }
  if (!/^(?:cxx|func|llvm)\.func\b/.test(text)) return false
  return !/\{\s*(?:loc\(.*\))?$/.test(text)
}

function isLLVMDeclaration(text: string) {
  if (/^%.*=\s*type\b/.test(text)) return true
  if (/^@(?:"(?:[^"\\]|\\.)*"|[\w.$-]+)\s*=/.test(text)) return true
  return /^declare\b/.test(text)
}

export function getIRFoldingRanges(model: monaco.editor.ITextModel) {
  let isDeclarationLine = isLLVMDeclaration
  if (model.getLanguageId() === "mlir") {
    isDeclarationLine = isMLIRDeclaration
  }

  const ranges: IRFoldingRange[] = []
  const blocks: { start: number; indent: number }[] = []
  let declarations: (IRFoldingRange & { indent: number }) | undefined
  let previousLine = 0
  let previousIndent = 0

  const finishDeclarations = () => {
    if (declarations && declarations.end > declarations.start) {
      ranges.push(declarations)
    }
    declarations = undefined
  }

  for (let line = 1; line <= model.getLineCount(); ++line) {
    const content = model.getLineContent(line)
    const text = content.trim()
    if (!text) continue

    const indent = model.getLineFirstNonWhitespaceColumn(line) - 1
    while (blocks.length && blocks[blocks.length - 1].indent >= indent) {
      const block = blocks.pop()!
      if (previousLine > block.start) {
        ranges.push({ start: block.start, end: previousLine })
      }
    }
    if (previousLine && indent > previousIndent) {
      blocks.push({ start: previousLine, indent: previousIndent })
    }
    previousLine = line
    previousIndent = indent

    if (
      declarations &&
      (indent > declarations.indent ||
        (indent === declarations.indent && /^[)>\]},]+$/.test(text)))
    ) {
      declarations.end = line
      continue
    }

    const isDeclaration = isDeclarationLine(text)

    if (!isDeclaration || declarations?.indent !== indent) {
      finishDeclarations()
    }
    if (isDeclaration) {
      declarations ??= { start: line, end: line, indent, declarations: true }
      declarations.end = line
    }
  }

  finishDeclarations()
  for (const block of blocks) {
    if (previousLine > block.start) {
      ranges.push({ start: block.start, end: previousLine })
    }
  }
  return ranges
}
