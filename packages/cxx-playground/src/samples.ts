import type { Device } from "./language-server"

export interface SampleCode {
  id: string
  name: string
  code: string
}

const semaFiles = import.meta.glob<string>(
  "../../../tests/unit_tests/sema/*.cc",
  { query: "?raw", import: "default", eager: true }
)

const gpuFiles = import.meta.glob<string>("./samples/gpu/*.cc", {
  query: "?raw",
  import: "default",
  eager: true,
})

function samplesOf(files: Record<string, string>): SampleCode[] {
  return Object.entries(files)
    .map(([filePath, code]) => {
      const id = filePath.split("/").pop()!
      return { id, name: id.replace(/_/g, " "), code }
    })
    .sort((a, b) => a.name.localeCompare(b.name))
}

export const samplesByDevice: Record<Device, SampleCode[]> = {
  cpu: samplesOf(semaFiles),
  gpu: samplesOf(gpuFiles),
}

function defaultOf(samples: SampleCode[], id: string): SampleCode | undefined {
  return samples.find((s) => s.id === id) ?? samples[0]
}

export const defaultSampleByDevice: Record<Device, SampleCode | undefined> = {
  cpu: defaultOf(samplesByDevice.cpu, "print_01.cc"),
  gpu: defaultOf(samplesByDevice.gpu, "circles.cc"),
}

export const defaultSample = defaultSampleByDevice.cpu
