import { Cpu, Gpu } from "lucide-react"
import { Device } from "../language-server"
import { ButtonGroup } from "./ui/button-group"
import { Button } from "./ui/button"

const deviceIcons = { cpu: Cpu, gpu: Gpu } as const

const deviceTitles: Record<Device, string> = {
  cpu: "Compile for the CPU (wasm32)",
  gpu: "Compile kernels for the GPU (WebGPU)",
}

export function DeviceSelector({
  device,
  setDevice,
}: {
  device: Device
  setDevice: (device: Device) => void
}) {
  return (
    <ButtonGroup aria-label="Target device" className="flex-none">
      {Device.map((option) => {
        const Icon = deviceIcons[option]
        return (
          <Button
            key={option}
            variant={device === option ? "default" : "outline"}
            size="xs"
            className="h-11 px-3 md:h-6 md:px-2"
            title={deviceTitles[option]}
            aria-pressed={device === option}
            onClick={() => setDevice(option)}
          >
            <Icon />
            {option.toUpperCase()}
          </Button>
        )
      })}
    </ButtonGroup>
  )
}
