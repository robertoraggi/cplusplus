const encoder = new TextEncoder()

export const clearScreen = encoder.encode("\x1b[2J\x1b[3J\x1b[H")

export function styled(code: number, text: string): Uint8Array<ArrayBuffer> {
  return encoder.encode(`\x1b[${code}m${text}\x1b[0m\r\n`)
}

export const dim = (text: string) => styled(2, text)
export const red = (text: string) => styled(31, text)
export const yellow = (text: string) => styled(33, text)

export function toTerminalBytes(
  chunks: Uint8Array[],
  followsCarriageReturn: boolean
): { bytes: Uint8Array<ArrayBuffer>; followsCarriageReturn: boolean } {
  const size = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const bytes = new Uint8Array(size * 2)
  let length = 0
  let previous = followsCarriageReturn ? 13 : 0

  for (const chunk of chunks) {
    for (const byte of chunk) {
      if (byte === 10 && previous !== 13) bytes[length++] = 13
      bytes[length++] = byte
      previous = byte
    }
  }

  return {
    bytes: bytes.slice(0, length),
    followsCarriageReturn: previous === 13,
  }
}
