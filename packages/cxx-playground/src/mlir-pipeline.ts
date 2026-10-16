export const mlirPassOrder = [
  "convert-cxx-to-memref",
  "convert-cxx-to-ptr",
  "mem2reg",
  "cxx-lift-cf-to-scf",
  "canonicalize",
] as const

export type MLIRPass = (typeof mlirPassOrder)[number]

export const defaultMLIRPasses: MLIRPass[] = [...mlirPassOrder]

export function buildMLIRPipeline(selected: readonly MLIRPass[]): string {
  const passes: string[] = []
  for (const pass of mlirPassOrder) {
    if (!selected.includes(pass)) continue
    passes.push(pass === "cxx-lift-cf-to-scf" ? `cxx.func(${pass})` : pass)
  }
  return passes.join(",")
}
