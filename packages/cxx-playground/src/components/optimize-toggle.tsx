import { Gauge, Snail } from "lucide-react"
import { Toggle } from "./ui/toggle"

export function OptimizeToggle({
  optimize,
  setOptimize,
}: {
  optimize: boolean
  setOptimize: (optimize: boolean) => void
}) {
  const label = optimize ? "Disable optimizations" : "Enable optimizations"

  return (
    <Toggle
      pressed={optimize}
      onPressedChange={setOptimize}
      title={label}
      aria-label={label}
    >
      {optimize ? <Gauge /> : <Snail />}
    </Toggle>
  )
}
