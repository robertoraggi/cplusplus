export interface SampleCode {
  id: string
  name: string
  code: string
}

const semaFiles = import.meta.glob<string>(
  "../../../tests/unit_tests/sema/*.cc",
  { query: "?raw", import: "default", eager: true }
)

export const samples: SampleCode[] = Object.entries(semaFiles)
  .map(([filePath, code]) => {
    const id = filePath.split("/").pop()!
    return { id, name: id.replace(/_/g, " "), code }
  })
  .sort((a, b) => a.name.localeCompare(b.name))

export const defaultSample =
  samples.find((s) => s.id === "auto_template.cc") ?? samples[0]
