import type { KernelManifest, KernelResource } from "cxx-frontend"

export const baseResolution = 256
export const resolutionScales = [1, 2, 4] as const

export type ResolutionScale = (typeof resolutionScales)[number]

export const shaderInputs = [
  "iResolution",
  "iTime",
  "iTimeDelta",
  "iFrame",
  "iMouse",
] as const

type ShaderInput = (typeof shaderInputs)[number]

const imageName = "image"

const arrayTypes = {
  f32: Float32Array,
  i32: Int32Array,
  u32: Uint32Array,
} as const

function bindingType(resource: KernelResource): GPUBufferBindingType {
  if (resource.kind === "uniform") return "uniform"
  return resource.access === "read" ? "read-only-storage" : "storage"
}

function inputData(resource: KernelResource, values: number[]) {
  const ArrayType = arrayTypes[resource.elementType]
  if (resource.kind === "storage") return ArrayType.from(values)
  const data = new ArrayType(resource.size / 4)
  data.set(values.slice(0, resource.components))
  return data
}

const presentShader = /* wgsl */ `
struct Size {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<storage, read> image: array<u32>;
@group(0) @binding(1) var<uniform> size: Size;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let x = min(u32(position.x), size.width - 1u);
  let y = min(u32(position.y), size.height - 1u);
  return unpack4x8unorm(image[y * size.width + x]);
}
`

export interface ShaderStatus {
  kind: "idle" | "running" | "paused" | "error" | "unsupported"
  message: string
  scale: ResolutionScale
  stale: boolean
}

interface LoadedShader {
  manifest: KernelManifest
  pipeline: GPUComputePipeline
  bindGroup: GPUBindGroup
  presentGroup: GPUBindGroup
  inputs: Map<ShaderInput, { resource: KernelResource; buffer: GPUBuffer }>
  buffers: GPUBuffer[]
  workgroups: [number, number, number]
}

export class ShaderCanvas {
  #device: GPUDevice | undefined
  #context: GPUCanvasContext | undefined
  #presentPipeline: GPURenderPipeline | undefined
  #sizeBuffer: GPUBuffer | undefined
  #shader: LoadedShader | undefined
  #source: { manifest: KernelManifest; code: string } | undefined
  #generation = 0
  #loading = Promise.resolve()
  #frame = 0
  #time = 0
  #lastTimestamp: number | undefined
  #animation: number | undefined
  #visible = true
  #paused = false
  #scale: ResolutionScale = 2
  #canvas: HTMLCanvasElement | undefined
  #mouse = [0, 0, 0, 0]
  #status: ShaderStatus = {
    kind: "idle",
    message: "",
    scale: 1,
    stale: false,
  }
  readonly #listeners = new Set<() => void>()

  subscribe = (listener: () => void) => {
    this.#listeners.add(listener)
    return () => {
      this.#listeners.delete(listener)
    }
  }

  getStatus = () => this.#status

  async attach(canvas: HTMLCanvasElement): Promise<void> {
    const generation = ++this.#generation
    this.#canvas = canvas
    this.#resizeCanvas()
    if (!navigator.gpu) {
      this.#setStatus("unsupported", "WebGPU is not available in this browser")
      return
    }
    const adapter = await navigator.gpu.requestAdapter()
    if (generation !== this.#generation) return
    if (!adapter) {
      this.#setStatus("unsupported", "no WebGPU adapter is available")
      return
    }
    const device = await adapter.requestDevice()
    if (generation !== this.#generation) {
      device.destroy()
      return
    }
    const context = canvas.getContext("webgpu")
    if (!context) {
      this.#setStatus("unsupported", "the canvas has no WebGPU context")
      return
    }
    const format = navigator.gpu.getPreferredCanvasFormat()
    context.configure({ device, format, alphaMode: "opaque" })
    const module = device.createShaderModule({ code: presentShader })
    const presentPipeline = await device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    })
    if (generation !== this.#generation) {
      device.destroy()
      return
    }
    this.#presentPipeline = presentPipeline
    this.#sizeBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.#device = device
    this.#writeSize()
    this.#context = context
    device.lost.then((info) => {
      if (this.#device !== device) return
      this.fail(`the WebGPU device was lost: ${info.message}`)
    })
    if (this.#source) this.load(this.#source.manifest, this.#source.code)
  }

  detach(): void {
    ++this.#generation
    this.#cancelAnimation()
    this.#releaseShader()
    this.#context?.unconfigure()
    this.#sizeBuffer?.destroy()
    this.#device?.destroy()
    this.#device = undefined
    this.#context = undefined
    this.#presentPipeline = undefined
    this.#sizeBuffer = undefined
    this.#canvas = undefined
  }

  load(manifest: KernelManifest, code: string): void {
    this.#source = { manifest, code }
    if (!this.#device) return
    this.#loading = this.#loading.then(() => this.#load(manifest, code))
  }

  fail(message: string): void {
    this.#source = undefined
    this.#cancelAnimation()
    this.#releaseShader()
    this.#setStatus("error", message)
  }

  restart(): void {
    this.#time = 0
    this.#frame = 0
    this.#lastTimestamp = undefined
    this.#paused = false
    this.#updatePlayback()
    if (!this.#playing) this.#render(0)
  }

  setVisible(visible: boolean): void {
    this.#visible = visible
    this.#updatePlayback()
  }

  setPaused(paused: boolean): void {
    this.#paused = paused
    this.#updatePlayback()
  }

  togglePaused(): void {
    this.setPaused(!this.#paused)
  }

  markStale(stale: boolean): void {
    if (this.#status.stale === stale) return
    this.#status = { ...this.#status, stale }
    this.#notify()
  }

  setScale(scale: ResolutionScale): void {
    if (scale === this.#scale) return
    this.#scale = scale
    this.#resizeCanvas()
    this.#writeSize()
    if (this.#source) {
      this.load(this.#source.manifest, this.#source.code)
      return
    }
    this.#setStatus(this.#status.kind, this.#status.message)
  }

  setMouse(x: number, y: number, down: boolean): void {
    this.#mouse = [x, y, down ? 1 : 0, 0]
    if (!this.#playing) this.#render(0)
  }

  get width(): number {
    return baseResolution * this.#scale
  }

  get height(): number {
    return baseResolution * this.#scale
  }

  get #playing(): boolean {
    return this.#visible && !this.#paused
  }

  #updatePlayback(): void {
    if (!this.#shader) return
    if (this.#playing) {
      this.#setStatus("running", this.#describe(this.#shader.manifest))
      this.#scheduleFrame()
      return
    }
    this.#cancelAnimation()
    this.#lastTimestamp = undefined
    this.#setStatus("paused", this.#describe(this.#shader.manifest))
  }

  #resizeCanvas(): void {
    if (!this.#canvas) return
    this.#canvas.width = this.width
    this.#canvas.height = this.height
  }

  #writeSize(): void {
    if (!this.#device || !this.#sizeBuffer) return
    this.#device.queue.writeBuffer(
      this.#sizeBuffer,
      0,
      Uint32Array.of(this.width, this.height)
    )
  }

  async #load(manifest: KernelManifest, code: string): Promise<void> {
    const device = this.#device
    if (!device) return
    try {
      const shader = await this.#createShader(device, manifest, code)
      this.#releaseShader()
      this.#shader = shader
      this.#updatePlayback()
      if (!this.#playing) this.#render(0)
    } catch (error) {
      this.fail(error instanceof Error ? error.message : String(error))
    }
  }

  async #createShader(
    device: GPUDevice,
    manifest: KernelManifest,
    code: string
  ): Promise<LoadedShader> {
    const image = manifest.resources.find((r) => r.name === imageName)
    if (!image)
      throw new Error(
        `${manifest.entryPoint}: a canvas kernel writes the 'image' buffer`
      )
    if (
      image.kind !== "storage" ||
      image.elementType !== "u32" ||
      image.access !== "read_write"
    )
      throw new Error(
        "'image' must be a writable device buffer of unsigned int RGBA pixels"
      )
    for (const resource of manifest.resources) {
      if (resource.name === imageName) continue
      if (!(shaderInputs as readonly string[]).includes(resource.name))
        throw new Error(
          `unknown canvas buffer '${resource.name}'; the canvas provides image, ${shaderInputs.join(", ")}`
        )
    }

    device.pushErrorScope("validation")
    const module = device.createShaderModule({ code })
    const info = await module.getCompilationInfo()
    const errors = info.messages.filter((m) => m.type === "error")
    if (errors.length)
      throw new Error(
        errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("\n")
      )
    const layout = device.createBindGroupLayout({
      entries: manifest.resources.map((resource) => ({
        binding: resource.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: bindingType(resource) },
      })),
    })
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: manifest.entryPoint },
    })

    const buffers: GPUBuffer[] = []
    const inputs: LoadedShader["inputs"] = new Map()
    let imageBuffer: GPUBuffer | undefined
    const entries = manifest.resources.map((resource) => {
      const buffer = device.createBuffer({
        size: this.#bufferSize(resource),
        usage:
          (resource.kind === "uniform"
            ? GPUBufferUsage.UNIFORM
            : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST,
      })
      buffers.push(buffer)
      if (resource.name === imageName) imageBuffer = buffer
      else inputs.set(resource.name as ShaderInput, { resource, buffer })
      return { binding: resource.binding, resource: { buffer } }
    })
    const bindGroup = device.createBindGroup({ layout, entries })
    const presentGroup = device.createBindGroup({
      layout: this.#presentPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: imageBuffer! } },
        { binding: 1, resource: { buffer: this.#sizeBuffer! } },
      ],
    })
    const failure = await device.popErrorScope()
    if (failure) {
      for (const buffer of buffers) buffer.destroy()
      throw new Error(failure.message)
    }

    const [lx, ly, lz] = manifest.localSize
    return {
      manifest,
      pipeline,
      bindGroup,
      presentGroup,
      inputs,
      buffers,
      workgroups: [
        Math.ceil(this.width / lx),
        Math.ceil(this.height / ly),
        Math.ceil(1 / lz),
      ],
    }
  }

  #bufferSize(resource: KernelResource): number {
    if (resource.kind === "uniform") return resource.size
    if (resource.name === imageName)
      return this.width * this.height * resource.arrayStride
    const input = resource.name as ShaderInput
    return this.#inputValues(input, 0).length * resource.arrayStride
  }

  #inputValues(input: ShaderInput, delta: number): number[] {
    switch (input) {
      case "iResolution":
        return [this.width, this.height, 1]
      case "iTime":
        return [this.#time]
      case "iTimeDelta":
        return [delta]
      case "iFrame":
        return [this.#frame]
      case "iMouse":
        return this.#mouse
    }
  }

  #render(delta: number): void {
    const device = this.#device
    const context = this.#context
    const shader = this.#shader
    if (!device || !context || !shader || !this.#presentPipeline) return

    for (const [input, { resource, buffer }] of shader.inputs) {
      const values = this.#inputValues(input, delta)
      device.queue.writeBuffer(buffer, 0, inputData(resource, values))
    }

    const encoder = device.createCommandEncoder()
    const compute = encoder.beginComputePass()
    compute.setPipeline(shader.pipeline)
    compute.setBindGroup(0, shader.bindGroup)
    compute.dispatchWorkgroups(...shader.workgroups)
    compute.end()

    const present = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    })
    present.setPipeline(this.#presentPipeline)
    present.setBindGroup(0, shader.presentGroup)
    present.draw(3)
    present.end()
    device.queue.submit([encoder.finish()])
  }

  #scheduleFrame(): void {
    if (this.#animation !== undefined) return
    this.#animation = requestAnimationFrame((timestamp) => {
      this.#animation = undefined
      if (!this.#playing || !this.#shader) return
      const delta =
        this.#lastTimestamp === undefined
          ? 0
          : (timestamp - this.#lastTimestamp) / 1000
      this.#lastTimestamp = timestamp
      this.#time += delta
      this.#render(delta)
      this.#frame += 1
      this.#scheduleFrame()
    })
  }

  #cancelAnimation(): void {
    if (this.#animation === undefined) return
    cancelAnimationFrame(this.#animation)
    this.#animation = undefined
  }

  #releaseShader(): void {
    for (const buffer of this.#shader?.buffers ?? []) buffer.destroy()
    this.#shader = undefined
  }

  #describe(manifest: KernelManifest): string {
    const [x, y, z] = manifest.localSize
    return `${manifest.entryPoint} ${this.width}x${this.height} workgroup ${x}x${y}x${z}`
  }

  #setStatus(kind: ShaderStatus["kind"], message: string): void {
    this.#status = { ...this.#status, kind, message, scale: this.#scale }
    this.#notify()
  }

  #notify(): void {
    for (const listener of [...this.#listeners]) listener()
  }
}

export const shaderCanvas = new ShaderCanvas()
