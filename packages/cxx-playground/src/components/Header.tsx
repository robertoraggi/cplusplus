import * as React from "react"
import {
  usePlayground,
  samples,
  loadSample,
  setOutputFormat,
} from "../playground-store"
import { Button } from "./ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"
import { ButtonGroup } from "./ui/button-group"

function statusLabel({
  isCompiling,
  diagnosticCount,
  compileTimeMs,
}: ReturnType<typeof usePlayground>) {
  if (isCompiling) return "compiling…"
  if (diagnosticCount > 0)
    return `${diagnosticCount} diagnostic${diagnosticCount === 1 ? "" : "s"}`
  if (compileTimeMs !== null) return `${compileTimeMs}ms`
  return ""
}

function GithubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
    </svg>
  )
}

export function Header() {
  const state = usePlayground()

  return (
    <header className="flex h-12 items-center justify-between border-b border-border/60 bg-background/85 px-4 backdrop-blur-md">
      <div className="flex items-center gap-3">
        <span className="font-mono text-xs text-muted-foreground">
          {statusLabel(state)}
        </span>
        <Select
          value={state.currentSampleId}
          onValueChange={(id) => id && loadSample(id)}
        >
          <SelectTrigger className="h-8 w-56 border-border/80 bg-muted/40 text-xs font-medium">
            <SelectValue placeholder="Select example..." />
          </SelectTrigger>
          <SelectContent
            align="center"
            className="max-h-80 min-w-56 overflow-y-auto"
          >
            {samples.map((sample) => (
              <SelectItem key={sample.id} value={sample.id} className="text-xs">
                {sample.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <ButtonGroup>
        <Button
          variant={state.outputFormat === "cxxir" ? "secondary" : "outline"}
          size="xs"
          onClick={() => setOutputFormat("cxxir")}
        >
          CXXIR
        </Button>
        <Button
          variant={state.outputFormat === "mlir" ? "secondary" : "outline"}
          size="xs"
          onClick={() => setOutputFormat("mlir")}
        >
          MLIR
        </Button>
        <Button
          variant={state.outputFormat === "llvm" ? "secondary" : "outline"}
          size="xs"
          onClick={() => setOutputFormat("llvm")}
        >
          LLVM IR
        </Button>
        <Button
          variant={state.outputFormat === "asm" ? "secondary" : "outline"}
          size="xs"
          onClick={() => setOutputFormat("asm")}
        >
          ASM
        </Button>
      </ButtonGroup>

      <Button
        variant="ghost"
        size="icon-xs"
        nativeButton={false}
        render={
          <a
            href="https://github.com/robertoraggi/cplusplus"
            target="_blank"
            rel="noreferrer"
            title="GitHub Repository"
          />
        }
      >
        <GithubIcon className="size-3.5" />
        <span className="sr-only">GitHub Repository</span>
      </Button>
    </header>
  )
}
