import * as React from "react"
import { createPortal } from "react-dom"

export function usePaneHost() {
  const [host] = React.useState(() => {
    const element = document.createElement("div")
    element.className = "flex h-full min-h-0 min-w-0 flex-col"
    return element
  })

  const slot = React.useCallback(
    (container: HTMLElement | null) => {
      container?.appendChild(host)
    },
    [host]
  )

  const render = (children: React.ReactNode) => createPortal(children, host)

  return { slot, render }
}
