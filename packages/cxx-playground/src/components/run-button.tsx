import { Play, Square } from "lucide-react"
import { usePlayground } from "../playground-context"
import { Button } from "./ui/button"
import { Kbd, KbdGroup } from "./ui/kbd"

const isApple =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform)

const shortcut = isApple ? ["⌘", "↵"] : ["Ctrl", "↵"]

export function RunButton({ floating = false }: { floating?: boolean }) {
  const { runnerState, runProgram, stopProgram } = usePlayground()
  const running = runnerState !== "idle"

  if (floating)
    return (
      <Button
        variant="outline"
        onClick={running ? stopProgram : runProgram}
        aria-label={running ? "Stop the program" : "Run the program"}
        title={running ? "Stop the program" : "Run the program"}
        className="size-12 rounded-full bg-background p-0 shadow-lg [&_svg]:size-5"
      >
        {running ? (
          <Square fill="currentColor" />
        ) : (
          <Play fill="currentColor" />
        )}
      </Button>
    )

  return (
    <Button
      variant="outline"
      onClick={running ? stopProgram : runProgram}
      aria-label={running ? "Stop the program" : "Run the program"}
      className="gap-1.5"
    >
      {running ? <Square fill="currentColor" /> : <Play fill="currentColor" />}
      <span className="w-8 text-left">{running ? "Stop" : "Run"}</span>
      <KbdGroup className={running ? "invisible" : undefined}>
        {shortcut.map((key) => (
          <Kbd key={key}>{key}</Kbd>
        ))}
      </KbdGroup>
    </Button>
  )
}
