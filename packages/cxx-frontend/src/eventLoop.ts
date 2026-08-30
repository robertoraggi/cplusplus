const yieldIntervalMs = 100;

function yieldToEventLoop(): Promise<void> {
  const { scheduler } = globalThis as unknown as {
    scheduler?: { yield?: () => Promise<void> };
  };

  if (scheduler?.yield) {
    return scheduler.yield();
  }

  return new Promise((resolve) => setTimeout(resolve, 0));
}

export function continueWithEventLoopYields(
  shouldContinue: () => boolean,
): () => Promise<boolean> {
  let lastYieldedAt = performance.now();

  return async () => {
    const now = performance.now();

    if (now - lastYieldedAt >= yieldIntervalMs) {
      lastYieldedAt = now;
      await yieldToEventLoop();
    }

    return shouldContinue();
  };
}
