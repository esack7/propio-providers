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

export interface AttemptContext {
  attempt: number;
  maxRetries: number;
}

export interface AttemptResultContext extends AttemptContext {
  durationMs: number;
}

export interface AttemptFailureContext extends AttemptResultContext {
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
  onFinalRetry?: (ctx: AttemptContext) => void; // called before final attempt — may mutate closure state
  onRetry?: (ctx: RetryContext) => void;
  onAttemptStart?: (ctx: AttemptContext) => void;
  onAttemptSuccess?: (ctx: AttemptResultContext) => void;
  onAttemptFailure?: (ctx: AttemptFailureContext) => void;
  /**
   * Maps a physical attempt to its exponential-backoff position. Returning
   * null retries immediately. This supports ordered endpoint fallback within
   * one logical retry cycle without multiplying backoff waits.
   */
  getBackoffAttempt?: (attempt: number) => number | null;
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

function resolveRetryDelayMs(
  attempt: number,
  options: WithRetryOptions,
): number {
  const configuredBackoffAttempt = options.getBackoffAttempt?.(attempt);
  const backoffAttempt =
    configuredBackoffAttempt === undefined ? attempt : configuredBackoffAttempt;
  if (backoffAttempt === null) return 0;
  return calculateRetryDelayMs(
    backoffAttempt,
    options.baseDelayMs ?? 500,
    options.maxDelayMs ?? 32_000,
  );
}

function sleep(delayMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delayMs));
}

export async function withRetry<T>(
  fn: (context: AttemptContext) => Promise<T>,
  opts: WithRetryOptions,
): Promise<T> {
  const {
    maxRetries,
    isRetryable,
    onFinalRetry,
    onRetry,
    onAttemptStart,
    onAttemptSuccess,
    onAttemptFailure,
  } = opts;

  let consecutive529s = 0;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const startedAt = performance.now();
    try {
      // Call onFinalRetry before the last attempt (it may mutate closure state used by fn)
      if (attempt === maxRetries && onFinalRetry) {
        onFinalRetry({ attempt, maxRetries });
      }

      onAttemptStart?.({ attempt, maxRetries });
      const result = await fn({ attempt, maxRetries });
      onAttemptSuccess?.({
        attempt,
        maxRetries,
        durationMs: performance.now() - startedAt,
      });
      return result;
    } catch (err) {
      onAttemptFailure?.({
        attempt,
        maxRetries,
        durationMs: performance.now() - startedAt,
        err,
      });
      consecutive529s = trackConsecutive529s(err, consecutive529s, opts);

      // Don't retry if error is not retryable or we've exhausted budget
      if (!isRetryable(err) || attempt >= maxRetries) {
        throw err;
      }

      const delayMs = resolveRetryDelayMs(attempt, opts);

      onRetry?.({ attempt, maxRetries, delayMs, err });

      await sleep(delayMs);
    }
  }

  // Should never reach here; the loop exhausts all attempts and throws
  throw new Error(
    "withRetry: exhausted all attempts without success or final throw",
  );
}
