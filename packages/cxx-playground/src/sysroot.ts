import { configure, fs } from "@zenfs/core"
import { Zip } from "@zenfs/archives"

export const sysroot = "/sysroot"
export const appdir = "/sysroot/bin"

let mounted: Promise<void> | undefined

export function loadSysroot(): Promise<void> {
  mounted ??= mountSysroot()
  return mounted
}

async function mountSysroot(): Promise<void> {
  const response = await fetch(`${import.meta.env.BASE_URL}sysroot.zip`)
  const data = await response.arrayBuffer()

  await configure({
    mounts: {
      [sysroot]: { backend: Zip, data, name: "sysroot.zip" },
    },
  })
}

const fileCache = new Map<string, string | undefined>()
const fileExistsCache = new Map<string, boolean>()

export function exists(path: string): boolean {
  if (fileExistsCache.has(path)) {
    return fileExistsCache.get(path)!
  }
  fileExistsCache.set(path, fs.existsSync(path))
  return fileExistsCache.get(path)!
}

export async function readFile(path: string): Promise<string | undefined> {
  if (fileCache.has(path)) {
    return fileCache.get(path)
  }
  try {
    const content = await fs.promises.readFile(path, "utf8")
    fileCache.set(path, content)
    return content
  } catch {
    return undefined
  }
}
