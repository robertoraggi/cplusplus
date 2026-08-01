import CodeEditor from "./code-editor"
import OutputCode from "./output-code"
import { Header } from "./components/Header"
import { usePlayground } from "./playground-store"

export function App() {
  const { isReady } = usePlayground()

  if (!isReady) {
    return (
      <div className="flex h-dvh w-dvw items-center justify-center bg-background text-foreground">
        <p className="animate-pulse font-mono text-xs text-muted-foreground">
          Loading C++ WASM Compiler…
        </p>
      </div>
    )
  }

  return (
    <div className="flex h-dvh w-dvw flex-col overflow-hidden bg-background font-sans text-foreground antialiased">
      <Header />
      <main className="grid min-h-0 flex-1 grid-cols-1 divide-y divide-border/60 md:grid-cols-2 md:divide-x md:divide-y-0">
        <CodeEditor />
        <OutputCode />
      </main>
    </div>
  )
}

export default App
