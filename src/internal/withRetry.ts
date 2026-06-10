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

export async function withRetry<T>(
  fn: () => Promise<T>,
  opts: WithRetryOptions,
): Promise<T> {
  const {
    maxRetries,
    baseDelayMs = 500,
    maxDelayMs = 32_000,
    isRetryable,
    is529,
    consecutive529Limit = 3,
    on529Fallback,
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
      // Track consecutive 529s (capacity exhaustion errors)
      if (is529?.(err)) {
        consecutive529s++;
        if (consecutive529s >= consecutive529Limit) {
          on529Fallback?.();
          throw err;
        }
      } else {
        consecutive529s = 0;
      }

      // Don't retry if error is not retryable or we've exhausted budget
      if (!isRetryable(err) || attempt >= maxRetries) {
        throw err;
      }

      // Compute delay: exponential backoff with full jitter
      const exponent = Math.min(attempt, 10); // Cap at 2^10 to prevent overflow
      const cap = Math.min(baseDelayMs * Math.pow(2, exponent), maxDelayMs);
      const delayMs = Math.floor(Math.random() * cap);

      onRetry?.({ attempt, maxRetries, delayMs, err });

      // Sleep before retrying
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  // Should never reach here; the loop exhausts all attempts and throws
  throw new Error(
    "withRetry: exhausted all attempts without success or final throw",
  );
}
