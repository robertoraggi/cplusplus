import {
  outputFormatsByDevice,
  type Device,
  type TextOutputCodeFormat,
} from "../language-server"
import type { OutputTab } from "../playground-context"
import { useIsWideLayout } from "../lib/use-media-query"
import { ButtonGroup } from "./ui/button-group"
import { Button } from "./ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"

const formatLabels: Record<TextOutputCodeFormat, string> = {
  cxxir: "CXXIR",
  mlir: "MLIR",
  llvm: "LLVM",
  asm: "ASM",
  wgsl: "WEBGPU",
}

const deviceFormatLabels: Record<
  Device,
  Partial<Record<TextOutputCodeFormat, string>>
> = {
  cpu: {},
  gpu: { mlir: "GPU", asm: "SPIR-V" },
}

export function OutputFormatSelector({
  device,
  outputFormat,
  activeTab,
  setOutputFormat,
  setActiveTab,
}: {
  device: Device
  outputFormat: TextOutputCodeFormat
  activeTab: OutputTab
  setOutputFormat: (format: TextOutputCodeFormat) => void
  setActiveTab: (tab: OutputTab) => void
}) {
  const wide = useIsWideLayout()
  const runTab = device === "gpu" ? "canvas" : "console"
  const selected = activeTab === "output" ? outputFormat : runTab
  const items: {
    value: TextOutputCodeFormat | typeof runTab
    label: string
  }[] = [
    ...outputFormatsByDevice[device].map((format) => ({
      value: format,
      label: deviceFormatLabels[device][format] ?? formatLabels[format],
    })),
    { value: runTab, label: runTab.toUpperCase() },
  ]

  function selectOutput(value: TextOutputCodeFormat | typeof runTab | null) {
    if (!value) return
    if (value === "console" || value === "canvas") setActiveTab(value)
    else setOutputFormat(value)
  }

  if (!wide)
    return (
      <Select<TextOutputCodeFormat | typeof runTab>
        items={items}
        value={selected}
        onValueChange={selectOutput}
      >
        <SelectTrigger
          aria-label="Select output"
          className="w-36 border-border/80 bg-muted/40 text-xs font-medium data-[size=default]:h-11 md:w-32 md:data-[size=default]:h-8"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent align="end" alignItemWithTrigger={false}>
          {items.map(({ value, label }) => (
            <SelectItem key={value} value={value} className="px-3 py-3 md:py-2">
              {label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )

  return (
    <ButtonGroup aria-label="Output" className="flex-none">
      {items.map(({ value, label }) => (
        <Button
          key={value}
          variant={selected === value ? "default" : "outline"}
          size="xs"
          aria-pressed={selected === value}
          onClick={() => selectOutput(value)}
        >
          {label}
        </Button>
      ))}
    </ButtonGroup>
  )
}
