/**
 * Generic retry helper with exponential backoff, full jitter, and stream-position awareness.
 *
 * This helper wraps pre-stream operations (HTTP connection establishment) and retries
 * transient failures. Once streaming starts, failures bubble — no auto-retry.
 */

export interface RetryContext {
  attempt: number;
  maxRetries: number;
  delayMs: number;
  err: unknown;
}

export interface WithRetryOptions {
  maxRetries: number;
  baseDelayMs?: number; // default 500
  maxDelayMs?: number; // default 32_000
  isRetryable: (err: unknown) => boolean;
  is529?: (err: unknown) => boolean;
  consecutive529Limit?: number; // default 3
  on529Fallback?: () => void;
  onFinalRetry?: () => void; // called before final attempt — may mutate closure state
  onRetry?: (ctx: RetryContext) => void;
}

function trackConsecutive529s(
  err: unknown,
  consecutive529s: number,
  options: WithRetryOptions,
): number {
  if (!options.is529?.(err)) {
    return 0;
  }

  const nextConsecutive529s = consecutive529s + 1;
  if (nextConsecutive529s < (options.consecutive529Limit ?? 3)) {
    return nextConsecutive529s;
  }

  options.on529Fallback?.();
  throw err;
}

function calculateRetryDelayMs(
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number,
): number {
  const exponent = Math.min(attempt, 10);
  const cap = Math.min(baseDelayMs * 2 ** exponent, maxDelayMs);
  return Math.floor(Math.random() * cap);
}

function sleep(delayMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delayMs));
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  opts: WithRetryOptions,
): Promise<T> {
  const {
    maxRetries,
    baseDelayMs = 500,
    maxDelayMs = 32_000,
    isRetryable,
    onFinalRetry,
    onRetry,
  } = opts;

  let consecutive529s = 0;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Call onFinalRetry before the last attempt (it may mutate closure state used by fn)
      if (attempt === maxRetries && onFinalRetry) {
        onFinalRetry();
      }

      return await fn();
    } catch (err) {
      consecutive529s = trackConsecutive529s(err, consecutive529s, opts);

      // Don't retry if error is not retryable or we've exhausted budget
      if (!isRetryable(err) || attempt >= maxRetries) {
        throw err;
      }

      const delayMs = calculateRetryDelayMs(attempt, baseDelayMs, maxDelayMs);

      onRetry?.({ attempt, maxRetries, delayMs, err });

      await sleep(delayMs);
    }
  }

  // Should never reach here; the loop exhausts all attempts and throws
  throw new Error(
    "withRetry: exhausted all attempts without success or final throw",
  );
}
