import type { SampleCode } from "../samples"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"

export function SampleSelector({
  samples,
  sampleId,
  onSelect,
}: {
  samples: SampleCode[]
  sampleId: string
  onSelect: (id: string) => void
}) {
  return (
    <Select value={sampleId} onValueChange={(id) => id && onSelect(id)}>
      <SelectTrigger
        aria-label="Select an example"
        title={sampleId}
        className="w-full min-w-0 border-border/80 bg-muted/40 text-xs font-medium data-[size=default]:h-11 md:w-32 md:data-[size=default]:h-8 lg:w-40 xl:w-56"
      >
        <SelectValue
          className="min-w-0 truncate"
          placeholder="Select example..."
        />
      </SelectTrigger>
      <SelectContent
        align="start"
        alignItemWithTrigger={false}
        className="max-h-80 w-80 max-w-[calc(100vw-2rem)]"
      >
        {samples.map((sample) => (
          <SelectItem
            key={sample.id}
            value={sample.id}
            className="py-2 pr-9 pl-3 text-xs"
          >
            {sample.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
