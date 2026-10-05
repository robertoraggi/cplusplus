declare namespace WebAssembly {
  class Suspending {
    constructor(callback: (...args: number[]) => Promise<unknown> | unknown)
  }

  function promising(
    callback: (...args: number[]) => unknown
  ): (...args: number[]) => Promise<unknown>
}
