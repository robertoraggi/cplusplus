import CodeEditor from "./code-editor"
import { OutputPanel } from "./components/output-panel"
import { Header } from "./components/Header"
import { useVisualViewport } from "./lib/use-visual-viewport"
import { RunButton } from "./components/run-button"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "./components/ui/resizable"
import { usePaneHost } from "./components/pane-host"
import { useIsWideLayout } from "./lib/use-media-query"
import { usePlayground } from "./playground-context"

export function App() {
  const { isReady } = usePlayground()
  const viewportRef = useVisualViewport()
  const wide = useIsWideLayout()
  const editorHost = usePaneHost()
  const outputHost = usePaneHost()

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
          {editorHost.render(<CodeEditor />)}
          {outputHost.render(<OutputPanel />)}
          <main className="relative min-h-0 flex-1">
            <ResizablePanelGroup
              key={wide ? "wide" : "narrow"}
              orientation={wide ? "horizontal" : "vertical"}
            >
              <ResizablePanel
                id={wide ? "editor" : "output"}
                defaultSize="50%"
                minSize="15%"
              >
                <div
                  ref={wide ? editorHost.slot : outputHost.slot}
                  className="h-full"
                />
              </ResizablePanel>
              <ResizableHandle
                withHandle={!wide}
                className="aria-[orientation=horizontal]:after:h-6 lg:after:w-2"
              />
              <ResizablePanel
                id={wide ? "output" : "editor"}
                defaultSize="50%"
                minSize="15%"
              >
                <div
                  ref={wide ? outputHost.slot : editorHost.slot}
                  className="h-full"
                />
              </ResizablePanel>
            </ResizablePanelGroup>
            {!wide && (
              <div className="absolute right-4 bottom-4">
                <RunButton floating />
              </div>
            )}
          </main>
        </>
      )}
    </div>
  )
}

export default App
