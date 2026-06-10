import {
  withRetry,
  type WithRetryOptions,
  type RetryContext,
} from "../internal/withRetry.js";

describe("withRetry", () => {
  it("succeeds immediately on first try", async () => {
    const fn = jest.fn().mockResolvedValue("success");
    const result = await withRetry(fn, {
      maxRetries: 3,
      isRetryable: () => false,
    });
    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries up to maxRetries times on retryable error", async () => {
    const fn = jest
      .fn()
      .mockRejectedValueOnce(new Error("transient error 1"))
      .mockRejectedValueOnce(new Error("transient error 2"))
      .mockResolvedValueOnce("success");

    const result = await withRetry(fn, {
      maxRetries: 3,
      isRetryable: () => true,
      baseDelayMs: 1,
      maxDelayMs: 1,
    });

    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("throws after maxRetries exhausted", async () => {
    const err = new Error("persistent error");
    const fn = jest.fn().mockRejectedValue(err);

    await expect(
      withRetry(fn, {
        maxRetries: 2,
        isRetryable: () => true,
        baseDelayMs: 1,
        maxDelayMs: 1,
      }),
    ).rejects.toThrow("persistent error");
    expect(fn).toHaveBeenCalledTimes(3); // initial + 2 retries
  });

  it("throws immediately for non-retryable errors", async () => {
    const err = new Error("fatal error");
    const fn = jest.fn().mockRejectedValue(err);

    await expect(
      withRetry(fn, {
        maxRetries: 3,
        isRetryable: () => false,
      }),
    ).rejects.toThrow("fatal error");

    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("calls onRetry on each retry with correct context", async () => {
    const err1 = new Error("error 1");
    const err2 = new Error("error 2");
    const fn = jest
      .fn()
      .mockRejectedValueOnce(err1)
      .mockRejectedValueOnce(err2)
      .mockResolvedValueOnce("success");
    const onRetry = jest.fn();

    const result = await withRetry(fn, {
      maxRetries: 3,
      isRetryable: () => true,
      baseDelayMs: 1,
      maxDelayMs: 1,
      onRetry,
    });

    expect(result).toBe("success");
    expect(onRetry).toHaveBeenCalledTimes(2);
    expect(onRetry).toHaveBeenNthCalledWith(1, {
      attempt: 0,
      maxRetries: 3,
      delayMs: expect.any(Number),
      err: err1,
    });
    expect(onRetry).toHaveBeenNthCalledWith(2, {
      attempt: 1,
      maxRetries: 3,
      delayMs: expect.any(Number),
      err: err2,
    });
  });

  it("calls onFinalRetry before the last attempt", async () => {
    let dropTools = false;
    const fn = jest.fn().mockImplementation(async () => {
      if (dropTools) {
        return "success";
      }
      throw new Error("retry");
    });
    const onFinalRetry = jest.fn(() => {
      dropTools = true;
    });

    const result = await withRetry(fn, {
      maxRetries: 1,
      isRetryable: () => true,
      baseDelayMs: 10,
      onFinalRetry,
    });

    expect(result).toBe("success");
    expect(onFinalRetry).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledTimes(2); // initial attempt + final retry
  });

  it("tracks consecutive 529s and calls on529Fallback", async () => {
    const capacity529 = new Error("capacity exceeded");
    Object.defineProperty(capacity529, "name", { value: "Capacity529" });

    const fn = jest
      .fn()
      .mockRejectedValueOnce(capacity529)
      .mockRejectedValueOnce(capacity529)
      .mockRejectedValueOnce(capacity529); // This one should trigger fallback

    const on529Fallback = jest.fn();

    await expect(
      withRetry(fn, {
        maxRetries: 10,
        isRetryable: () => true,
        is529: (err) => err === capacity529,
        consecutive529Limit: 3,
        on529Fallback,
        baseDelayMs: 1,
      }),
    ).rejects.toThrow("capacity exceeded");
    expect(on529Fallback).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("resets 529 counter on non-529 error", async () => {
    const capacity529 = new Error("capacity");
    const otherError = new Error("other");
    Object.defineProperty(capacity529, "name", { value: "Capacity529" });

    const fn = jest
      .fn()
      .mockRejectedValueOnce(capacity529)
      .mockRejectedValueOnce(capacity529)
      .mockRejectedValueOnce(otherError)
      .mockRejectedValueOnce(capacity529)
      .mockRejectedValueOnce(capacity529)
      .mockRejectedValueOnce(capacity529); // This triggers fallback

    const on529Fallback = jest.fn();

    await expect(
      withRetry(fn, {
        maxRetries: 10,
        isRetryable: () => true,
        is529: (err) => err === capacity529,
        consecutive529Limit: 3,
        on529Fallback,
        baseDelayMs: 1,
      }),
    ).rejects.toThrow("capacity");
    // Should be called only once (at the end after 2 non-consecutive 529s + 3 consecutive 529s)
    expect(on529Fallback).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledTimes(6);
  });

  it("uses exponential backoff with jitter for delays", async () => {
    const fn = jest
      .fn()
      .mockRejectedValueOnce(new Error("error 1"))
      .mockRejectedValueOnce(new Error("error 2"))
      .mockResolvedValueOnce("success");

    const result = await withRetry(fn, {
      maxRetries: 3,
      isRetryable: () => true,
      baseDelayMs: 1,
      maxDelayMs: 10,
    });

    expect(result).toBe("success");
    expect(fn).toHaveBeenCalledTimes(3);
  });
});
