import { FileCodeIcon } from "lucide-react"
import { useIsWideLayout } from "../lib/use-media-query"
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
  const wide = useIsWideLayout()

  return (
    <Select value={sampleId} onValueChange={(id) => id && onSelect(id)}>
      <SelectTrigger
        aria-label="Select an example"
        title={sampleId}
        className={
          wide
            ? "h-8 w-56 border-border/80 bg-muted/40 text-xs font-medium"
            : "size-8 justify-center border-border/80 bg-muted/40 px-0 [&>svg:last-child]:hidden"
        }
      >
        {wide ? (
          <SelectValue placeholder="Select example..." />
        ) : (
          <FileCodeIcon />
        )}
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
