import * as React from "react"
import { cn } from "@/lib/utils"
import OutputCode from "../output-code"
import { usePlayground } from "../playground-context"
import { TerminalPanel } from "./terminal-panel"
import { CanvasPanel } from "./canvas-panel"

function Pane({
  active,
  children,
}: {
  active: boolean
  children: React.ReactNode
}) {
  return (
    <div
      aria-hidden={!active}
      className={cn("absolute inset-0 flex flex-col", !active && "invisible")}
    >
      {children}
    </div>
  )
}

export function OutputPanel() {
  const { activeTab, device } = usePlayground()

  return (
    <section className="relative h-full min-h-0 min-w-0">
      <Pane active={activeTab === "output"}>
        <OutputCode />
      </Pane>
      <Pane active={activeTab === "console"}>
        <TerminalPanel active={activeTab === "console"} />
      </Pane>
      {device === "gpu" && (
        <Pane active={activeTab === "canvas"}>
          <CanvasPanel active={activeTab === "canvas"} />
        </Pane>
      )}
    </section>
  )
}
