import CodeEditor from "./code-editor"
import OutputCode from "./output-code"
import { Header } from "./components/Header"
import { useVisualViewport } from "./lib/use-visual-viewport"
import { usePlayground } from "./playground-context"

export function App() {
  const { isReady } = usePlayground()
  const viewportRef = useVisualViewport()

  return (
    <div
      ref={viewportRef}
      className="fixed top-0 left-0 flex h-dvh w-dvw flex-col overflow-hidden bg-background font-sans text-foreground antialiased"
    >
      {!isReady ? (
        <div className="flex min-h-0 flex-1 items-center justify-center">
          <p className="animate-pulse font-mono text-xs text-muted-foreground">
            Loading C++ WASM Compiler…
          </p>
        </div>
      ) : (
        <>
          <Header />
          <main className="grid min-h-0 flex-1 grid-cols-1 divide-y divide-border/60 md:grid-cols-2 md:divide-x md:divide-y-0">
            <CodeEditor />
            <OutputCode />
          </main>
        </>
      )}
    </div>
  )
}

export default App
