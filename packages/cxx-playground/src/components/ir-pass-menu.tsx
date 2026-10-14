// Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import {
  Boxes,
  Eraser,
  Layers,
  MemoryStick,
  TrendingUp,
  Workflow,
} from "lucide-react"
import type { MLIRPass } from "../mlir-pipeline"
import { usePlayground } from "../playground-context"
import { Select, SelectContent, SelectItem, SelectTrigger } from "./ui/select"

export function IRPassMenu() {
  const state = usePlayground()
  const choices = [
    {
      name: "MemRef",
      description: "Typed local storage",
      icon: Boxes,
      pass: "convert-cxx-to-memref" as const,
    },
    {
      name: "mem2reg",
      description: "Promote local slots",
      icon: TrendingUp,
      pass: "mem2reg" as const,
    },
    {
      name: "canonicalize",
      description: "Simplify IR",
      icon: Eraser,
      pass: "canonicalize" as const,
    },
    {
      name: "SCF",
      description: "Structured control flow",
      icon: Workflow,
      pass: "cxx-lift-cf-to-scf" as const,
    },
    {
      name: "Ptr",
      description: "Pointer memory operations",
      icon: MemoryStick,
      pass: "convert-cxx-to-ptr" as const,
    },
  ]
  const selected = state.mlirPasses

  return (
    <Select
      multiple
      value={selected}
      onValueChange={(names: MLIRPass[]) => state.setMLIRPasses(names)}
    >
      <SelectTrigger
        aria-label="Select MLIR passes"
        className="h-8 border-border/80 bg-muted/40 text-xs font-medium"
      >
        <Layers />
        Passes
        <span className="rounded bg-foreground/10 px-1.5 text-[10px] text-muted-foreground tabular-nums">
          {selected.length}
        </span>
      </SelectTrigger>
      <SelectContent
        align="end"
        alignItemWithTrigger={false}
        className="max-h-80 w-72 max-w-[calc(100vw-2rem)]"
      >
        {choices.map((choice) => (
          <SelectItem
            key={choice.pass}
            value={choice.pass}
            className="py-2 pr-9 pl-3 text-xs"
          >
            <choice.icon className="text-muted-foreground" />
            <span>{choice.name}</span>
            <span className="text-muted-foreground">{choice.description}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
