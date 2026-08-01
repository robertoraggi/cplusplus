import { samples } from "../samples"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"

export function SampleSelector({
  sampleId,
  onSelect,
}: {
  sampleId: string
  onSelect: (id: string) => void
}) {
  return (
    <Select value={sampleId} onValueChange={(id) => id && onSelect(id)}>
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
  )
}
