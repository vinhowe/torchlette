import { describe, expect, it } from "vitest";
import { Engine, NonReentrantBackwardError } from "../src";

describe("backward rooting", () => {
  it("advances tokGlobal when backward is scheduled", () => {
    const engine = new Engine();
    const before = engine._debugSnapshot().tokGlobal.id;

    engine._debug_backward(() => undefined);

    const after = engine._debugSnapshot().tokGlobal.id;
    expect(after).not.toBe(before);

    const effects = engine.trace
      .snapshot()
      .filter((event) => event.type === "effect");
    expect(effects).toContainEqual({
      type: "effect",
      op: "backward_root",
      input: before,
      output: after,
    });
  });
});

describe("non-reentrant backward", () => {
  it("throws when backward is invoked recursively", () => {
    const engine = new Engine();

    expect(() =>
      engine._debug_backward(() => {
        engine._debug_backward(() => undefined);
      }),
    ).toThrow(NonReentrantBackwardError);
  });
});
