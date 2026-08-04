import { BracesIcon } from "lucide-react"
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
      <SelectContent align="center" className="max-h-64 w-80">
        {samples.map((sample) => (
          <SelectItem
            key={sample.id}
            value={sample.id}
            className="py-2 pr-9 pl-3 text-xs"
          >
            <BracesIcon className="size-4 text-muted-foreground" />
            {sample.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
