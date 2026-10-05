import * as React from "react"
import { Terminal, type TerminalHandle } from "@wterm/react"
import { programRunner } from "../program-runner"

export function TerminalPanel({ active }: { active: boolean }) {
  const terminalRef = React.useRef<TerminalHandle>(null)
  const [ready, setReady] = React.useState(false)

  React.useEffect(() => {
    if (!ready) return
    return programRunner.attach({
      write: (data) => terminalRef.current?.write(data),
      focus: () => terminalRef.current?.focus(),
    })
  }, [ready])

  return (
    <Terminal
      ref={terminalRef}
      autoResize
      cursorBlink={false}
      renderingPaused={!active}
      onData={(data) => programRunner.input(data)}
      onReady={() => setReady(true)}
      className="size-full rounded-none! shadow-none!"
    />
  )
}
