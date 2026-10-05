import * as React from "react"

export function useMediaQuery(query: string) {
  return React.useSyncExternalStore(
    (notify) => {
      const list = window.matchMedia(query)
      list.addEventListener("change", notify)
      return () => list.removeEventListener("change", notify)
    },
    () => window.matchMedia(query).matches
  )
}

export function useIsWideLayout() {
  return useMediaQuery("(min-width: 1024px)")
}
