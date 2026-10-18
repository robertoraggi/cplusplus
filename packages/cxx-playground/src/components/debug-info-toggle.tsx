import { Bug, BugOff } from "lucide-react"
import { Toggle } from "./ui/toggle"

export function DebugInfoToggle({
  debugInfo,
  setDebugInfo,
}: {
  debugInfo: boolean
  setDebugInfo: (debugInfo: boolean) => void
}) {
  const label = debugInfo ? "Hide debug info" : "Show debug info"

  return (
    <Toggle
      pressed={debugInfo}
      onPressedChange={setDebugInfo}
      title={label}
      aria-label={label}
      className="size-11 p-0 md:size-8"
    >
      {debugInfo ? <Bug /> : <BugOff />}
    </Toggle>
  )
}
