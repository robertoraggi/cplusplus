import { usePlayground } from "../playground-context"
import { Button } from "./ui/button"
import { SampleSelector } from "./sample-selector"
import { OutputFormatSelector } from "./output-format-selector"
import { DeviceSelector } from "./device-selector"
import { samplesByDevice } from "../samples"
import { IRPassMenu } from "./ir-pass-menu"
import { DebugInfoToggle } from "./debug-info-toggle"
import { RunButton } from "./run-button"

function statusLabel({
  diagnosticCount,
  compileTimeMs,
}: ReturnType<typeof usePlayground>) {
  if (diagnosticCount > 0)
    return `${diagnosticCount} diagnostic${diagnosticCount === 1 ? "" : "s"}`
  if (compileTimeMs !== null) return `${compileTimeMs}ms`
  return ""
}

// from the bootstrap icons
function GithubIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="size-4"
      fill="currentColor"
      viewBox="0 0 16 16"
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8" />
    </svg>
  )
}

export function Header() {
  const state = usePlayground()

  return (
    <header className="grid flex-none grid-cols-[minmax(0,1fr)_auto] items-center gap-2 border-b border-border/60 bg-background/85 p-2 backdrop-blur-md md:flex md:h-12 md:px-3 md:py-0 lg:gap-3 lg:px-4">
      <div className="flex min-w-0 items-center gap-2 md:flex-none lg:gap-3">
        <div className="max-lg:hidden">
          <RunButton />
        </div>
        <SampleSelector
          samples={samplesByDevice[state.device]}
          sampleId={state.currentSampleId}
          onSelect={state.loadSample}
        />
        <span className="hidden font-mono text-xs whitespace-nowrap text-muted-foreground tabular-nums 2xl:block">
          {statusLabel(state)}
        </span>
      </div>

      <div className="col-span-2 row-start-2 flex min-w-0 items-center justify-between gap-2 md:flex-1 md:justify-center">
        <DeviceSelector device={state.device} setDevice={state.setDevice} />
        <OutputFormatSelector
          device={state.device}
          outputFormat={state.outputFormat}
          activeTab={state.activeTab}
          setOutputFormat={(format) => {
            state.setActiveTab("output")
            state.setOutputFormat(format)
          }}
          setActiveTab={state.setActiveTab}
        />
      </div>

      <div className="col-start-2 row-start-1 flex flex-none items-center gap-1">
        <IRPassMenu />

        <DebugInfoToggle
          debugInfo={state.debugInfo}
          setDebugInfo={state.setDebugInfo}
        />

        <Button
          variant="ghost"
          size="icon-xs"
          className="size-11 md:size-6"
          nativeButton={false}
          render={
            <a
              href="https://github.com/robertoraggi/cplusplus"
              target="_blank"
              rel="noreferrer"
              title="GitHub Repository"
            />
          }
        >
          <GithubIcon />
          <span className="sr-only">GitHub Repository</span>
        </Button>
      </div>
    </header>
  )
}
