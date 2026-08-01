import { OutputCodeFormat } from "cxx-frontend"
import { ButtonGroup } from "./ui/button-group"
import { Button } from "./ui/button"

export function OutputFormatSelector({
  outputFormat,
  setOutputFormat,
}: {
  outputFormat: OutputCodeFormat
  setOutputFormat: (format: OutputCodeFormat) => void
}) {
  return (
    <ButtonGroup>
      {OutputCodeFormat.map((format) => (
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
