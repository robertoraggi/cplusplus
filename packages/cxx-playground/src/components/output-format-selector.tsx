import { TextOutputCodeFormat } from "../language-server"
import { ButtonGroup } from "./ui/button-group"
import { Button } from "./ui/button"

export function OutputFormatSelector({
  outputFormat,
  consoleActive,
  setOutputFormat,
  showConsole,
}: {
  outputFormat: TextOutputCodeFormat
  consoleActive: boolean
  setOutputFormat: (format: TextOutputCodeFormat) => void
  showConsole: () => void
}) {
  return (
    <ButtonGroup>
      {TextOutputCodeFormat.map((format) => (
        <Button
          key={format}
          variant={
            !consoleActive && outputFormat === format ? "default" : "outline"
          }
          size="xs"
          onClick={() => setOutputFormat(format)}
        >
          {format.toUpperCase()}
        </Button>
      ))}
      <Button
        variant={consoleActive ? "default" : "outline"}
        size="xs"
        onClick={showConsole}
      >
        CONSOLE
      </Button>
    </ButtonGroup>
  )
}
