import { describe, expect, it } from "vitest";
import { Engine, NonReentrantBackwardError } from "../src/engine/engine";

describe("non-reentrant backward", () => {
  it("throws when backward is invoked recursively", () => {
    const engine = new Engine();

    expect(() =>
      engine._debug_backward(() => {
        engine._debug_backward(() => undefined);
      }),
    ).toThrow(NonReentrantBackwardError);
  });

  it("allows sequential backward calls", () => {
    const engine = new Engine();

    engine._debug_backward(() => undefined);
    expect(() => engine._debug_backward(() => undefined)).not.toThrow();
  });
});
