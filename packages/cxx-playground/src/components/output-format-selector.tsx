import { TextOutputCodeFormat } from "../compiler"
import { ButtonGroup } from "./ui/button-group"
import { Button } from "./ui/button"

export function OutputFormatSelector({
  outputFormat,
  setOutputFormat,
}: {
  outputFormat: TextOutputCodeFormat
  setOutputFormat: (format: TextOutputCodeFormat) => void
}) {
  return (
    <ButtonGroup>
      {TextOutputCodeFormat.map((format) => (
        <Button
          key={format}
          variant={outputFormat === format ? "default" : "outline"}
          size="xs"
          onClick={() => setOutputFormat(format)}
        >
          {format.toUpperCase()}
        </Button>
      ))}
    </ButtonGroup>
  )
}
