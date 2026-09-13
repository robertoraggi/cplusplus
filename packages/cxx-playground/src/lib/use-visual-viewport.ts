import * as React from "react"

export function useVisualViewport() {
  const containerRef = React.useRef<HTMLDivElement>(null)

  React.useLayoutEffect(() => {
    const container = containerRef.current
    const viewport = window.visualViewport
    if (!container || !viewport) return

    let animationFrame = 0

    const update = () => {
      animationFrame = 0
      container.style.left = `${viewport.offsetLeft}px`
      container.style.top = `${viewport.offsetTop}px`
      container.style.width = `${viewport.width}px`
      container.style.height = `${viewport.height}px`
    }

    const scheduleUpdate = () => {
      if (animationFrame !== 0) return
      animationFrame = window.requestAnimationFrame(update)
    }

    update()
    viewport.addEventListener("resize", scheduleUpdate)
    viewport.addEventListener("scroll", scheduleUpdate)

    return () => {
      viewport.removeEventListener("resize", scheduleUpdate)
      viewport.removeEventListener("scroll", scheduleUpdate)
      window.cancelAnimationFrame(animationFrame)
    }
  }, [])

  return containerRef
}
