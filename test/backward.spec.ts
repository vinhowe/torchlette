import { describe, expect, it } from "vitest";
import {
  NonReentrantBackwardError,
  RuntimeEngine,
} from "../src/runtime/engine";

describe("non-reentrant backward", () => {
  it("throws when backward is invoked recursively", () => {
    const engine = new RuntimeEngine();

    expect(() =>
      engine._debug_backward(() => {
        engine._debug_backward(() => undefined);
      }),
    ).toThrow(NonReentrantBackwardError);
  });

  it("allows sequential backward calls", () => {
    const engine = new RuntimeEngine();

    engine._debug_backward(() => undefined);
    expect(() => engine._debug_backward(() => undefined)).not.toThrow();
  });
});
