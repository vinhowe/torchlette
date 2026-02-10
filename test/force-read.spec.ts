import { describe, expect, it } from "vitest";

import { Engine } from "../src";

describe("host reads", () => {
  it("records a host_read effect", () => {
    const engine = new Engine();

    engine._debug_runEntryPoint(() => {
      engine.forceRead(5);
    });

    const effects = engine.trace
      .snapshot()
      .filter((event) => event.type === "effect");
    expect(effects.some((event) => event.op === "host_read:5")).toBe(true);
  });

  it("records a force plan with token roots", () => {
    const engine = new Engine();
    engine._debug_bindPendingLoc(1, 7);
    engine._debug_commitLocStore(7);

    engine.forceRead(1);

    const plans = engine.trace
      .snapshot()
      .filter((event) => event.type === "force_plan");
    expect(plans).toEqual([{ type: "force_plan", baseId: 1, tokenIds: [0] }]);
  });

  it("uses loc tokens when available", () => {
    const engine = new Engine();
    engine._debug_bindPendingLoc(2, 9);
    engine._debug_orderedAccessBase(2, "load");

    engine.forceRead(2);

    const plans = engine.trace
      .snapshot()
      .filter((event) => event.type === "force_plan");
    expect(plans).toContainEqual({
      type: "force_plan",
      baseId: 2,
      tokenIds: [1, 2],
    });
  });

  it("finalizes pending loc bindings before read", () => {
    const engine = new Engine();
    engine._debug_bindPendingLoc(1, 7);
    engine._debug_commitLocStore(7);

    engine.forceRead(1);

    const snapshot = engine._debugSnapshot();
    expect(snapshot.bindings["1"]).toMatchObject({
      kind: "loc",
      locId: 7,
    });
  });
});
