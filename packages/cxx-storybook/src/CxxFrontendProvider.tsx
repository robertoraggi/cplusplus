import React from "react";
import { CxxFrontendClient } from "./CxxFrontendClient";

interface CxxFrontendProviderProps {
  client: CxxFrontendClient;
  fallback?: React.ReactNode;
  children: React.ReactNode;
}

export function CxxFrontendProvider({
  client,
  fallback,
  children,
}: CxxFrontendProviderProps) {
  const [isLoaded, setIsLoaded] = React.useState(false);

  React.useEffect(() => {
    if (isLoaded) return;
    setIsLoaded(false);
    const controller = new AbortController();
    client.load(controller.signal).then(() => setIsLoaded(true));
    return () => controller.abort();
  }, [client, isLoaded]);

  return isLoaded ? <>{children}</> : fallback;
}
