import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "./App.tsx"
import { PlaygroundProvider } from "./playground-context.tsx"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <PlaygroundProvider>
      <App />
    </PlaygroundProvider>
  </StrictMode>
)
