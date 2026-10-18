import * as React from "react"
import { Pause, Play, RotateCcw } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  baseResolution,
  resolutionScales,
  shaderCanvas,
  shaderInputs,
} from "../gpu/shader-canvas"
import { Button } from "./ui/button"
import { ButtonGroup } from "./ui/button-group"

const statusTone = {
  idle: "text-muted-foreground",
  running: "text-muted-foreground",
  paused: "text-muted-foreground",
  error: "text-destructive",
  unsupported: "text-destructive",
} as const

export function CanvasPanel({ active }: { active: boolean }) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const status = React.useSyncExternalStore(
    shaderCanvas.subscribe,
    shaderCanvas.getStatus
  )

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    shaderCanvas.attach(canvas).catch((error: unknown) => {
      shaderCanvas.fail(error instanceof Error ? error.message : String(error))
    })
    return () => shaderCanvas.detach()
  }, [])

  React.useEffect(() => {
    shaderCanvas.setVisible(active)
  }, [active])

  const trackMouse = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget
    const rect = canvas.getBoundingClientRect()
    const scale = Math.min(
      rect.width / canvas.width,
      rect.height / canvas.height
    )
    const left = rect.left + (rect.width - canvas.width * scale) / 2
    const top = rect.top + (rect.height - canvas.height * scale) / 2
    const x = (event.clientX - left) / scale
    const y = (event.clientY - top) / scale
    shaderCanvas.setMouse(x, y, (event.buttons & 1) !== 0)
  }

  const failed = status.kind === "error" || status.kind === "unsupported"
  const paused = status.kind === "paused"

  return (
    <div className="flex h-full min-h-0 flex-col bg-muted/30">
      <div className="relative flex min-h-0 flex-1 items-center justify-center p-4">
        <canvas
          ref={canvasRef}
          onPointerMove={trackMouse}
          onPointerDown={trackMouse}
          onPointerUp={trackMouse}
          className={cn(
            "size-full object-contain [image-rendering:pixelated]",
            failed && "opacity-40"
          )}
        />
        {failed && (
          <pre className="absolute inset-x-4 bottom-4 max-h-1/2 overflow-auto rounded-md border border-destructive/40 bg-background/95 p-3 font-mono text-xs whitespace-pre-wrap text-destructive">
            {status.message}
          </pre>
        )}
      </div>
      <div className="flex h-9 flex-none items-center gap-2 border-t border-border/60 px-3 font-mono text-xs">
        <Button
          variant="ghost"
          size="icon-xs"
          title={paused ? "Resume the animation" : "Pause the animation"}
          disabled={failed || status.kind === "idle"}
          onClick={() => shaderCanvas.togglePaused()}
        >
          {paused ? <Play /> : <Pause />}
          <span className="sr-only">
            {paused ? "Resume the animation" : "Pause the animation"}
          </span>
        </Button>
        <Button
          variant="ghost"
          size="icon-xs"
          title="Restart the time"
          disabled={failed || status.kind === "idle"}
          onClick={() => shaderCanvas.restart()}
        >
          <RotateCcw />
          <span className="sr-only">Restart the time</span>
        </Button>
        <ButtonGroup aria-label="Canvas resolution">
          {resolutionScales.map((scale) => (
            <Button
              key={scale}
              variant={status.scale === scale ? "default" : "outline"}
              size="xs"
              title={`Render at ${baseResolution * scale}x${baseResolution * scale}`}
              aria-pressed={status.scale === scale}
              onClick={() => shaderCanvas.setScale(scale)}
            >
              {scale}x
            </Button>
          ))}
        </ButtonGroup>
        <span className={cn("min-w-0 truncate", statusTone[status.kind])}>
          {failed ? status.kind : status.message}
        </span>
        {status.stale && !failed && (
          <span className="flex-none text-amber-500">(modified)</span>
        )}
        <span className="ml-auto truncate text-muted-foreground max-xl:hidden">
          image, {shaderInputs.join(", ")}
        </span>
      </div>
    </div>
  )
}
