import { Play, Square } from "lucide-react"
import { cn } from "@/lib/utils"
import { usePlayground } from "../playground-context"
import { Button } from "./ui/button"
import { Kbd, KbdGroup } from "./ui/kbd"

const isApple =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform)

const shortcut = isApple ? ["⌘", "↵"] : ["Ctrl", "↵"]

export function RunButton({ floating = false }: { floating?: boolean }) {
  const { runnerState, runProgram, stopProgram } = usePlayground()
  const running = runnerState !== "idle"

  return (
    <Button
      variant="outline"
      onClick={running ? stopProgram : runProgram}
      aria-label={running ? "Stop the program" : "Run the program"}
      className={cn(
        "w-36 justify-between gap-2",
        floating && "h-10 rounded-full bg-background px-4 shadow-lg"
      )}
    >
      {running ? <Square fill="currentColor" /> : <Play fill="currentColor" />}
      <span className="flex-1 text-left">{running ? "Stop" : "Run"}</span>
      {!running && (
        <KbdGroup>
          {shortcut.map((key) => (
            <Kbd key={key}>{key}</Kbd>
          ))}
        </KbdGroup>
      )}
    </Button>
  )
}
