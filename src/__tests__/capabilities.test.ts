import { ProviderContextLengthError, ProviderError } from "../types.js";
import { validateContextWindowTokens } from "../internal/capabilities.js";

describe("ProviderContextLengthError", () => {
  it("should be an instance of ProviderError", () => {
    const error = new ProviderContextLengthError("too long");
    expect(error).toBeInstanceOf(ProviderError);
    expect(error).toBeInstanceOf(ProviderContextLengthError);
  });

  it("should have the correct name", () => {
    const error = new ProviderContextLengthError("too long");
    expect(error.name).toBe("ProviderContextLengthError");
  });

  it("should preserve the original error", () => {
    const original = new Error("upstream error");
    const error = new ProviderContextLengthError("context exceeded", original);
    expect(error.originalError).toBe(original);
  });

  it("should be distinguishable from other ProviderError subclasses", () => {
    const contextError = new ProviderContextLengthError("too long");
    const genericError = new ProviderError("something else");

    expect(contextError instanceof ProviderContextLengthError).toBe(true);
    expect(genericError instanceof ProviderContextLengthError).toBe(false);
  });
});

describe("Provider capability validation", () => {
  it("should accept positive integer context windows", () => {
    expect(validateContextWindowTokens(32_000, "contextWindowTokens")).toBe(
      32_000,
    );
  });

  it.each([undefined, 0, -1, 1.5, "128000"])(
    "should reject invalid context window %p",
    (value) => {
      expect(() =>
        validateContextWindowTokens(value, "contextWindowTokens"),
      ).toThrow(/positive integer/);
    },
  );
});
