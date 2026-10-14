export type MLIRPass =
  | "convert-cxx-to-memref"
  | "mem2reg"
  | "canonicalize"
  | "cxx-lift-cf-to-scf"
  | "convert-cxx-to-ptr"

export const defaultMLIRPasses: MLIRPass[] = [
  "mem2reg",
  "canonicalize",
  "cxx-lift-cf-to-scf",
  "convert-cxx-to-ptr",
]

export function buildMLIRPipeline(selected: readonly MLIRPass[]): string {
  const passes: string[] = []
  if (selected.includes("convert-cxx-to-memref"))
    passes.push("convert-cxx-to-memref")
  if (selected.includes("mem2reg")) passes.push("mem2reg")
  if (selected.includes("canonicalize")) passes.push("canonicalize")
  if (selected.includes("cxx-lift-cf-to-scf"))
    passes.push("cxx.func(cxx-lift-cf-to-scf)")
  if (selected.includes("convert-cxx-to-ptr")) passes.push("convert-cxx-to-ptr")
  if (
    selected.includes("convert-cxx-to-memref") &&
    !selected.includes("cxx-lift-cf-to-scf")
  )
    passes.push("convert-scf-to-cf")
  return passes.join(",")
}
