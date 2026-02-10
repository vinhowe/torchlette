import { describe, expect, it } from "vitest";
import { Engine } from "../src";

describe("token algebra", () => {
  it("afterAll is commutative, associative, and idempotent", () => {
    const engine = new Engine();
    const a = engine._debugCreateToken();
    const b = engine._debugCreateToken();
    const c = engine._debugCreateToken();

    const ab = engine.afterAll(a, b);
    const ba = engine.afterAll(b, a);
    expect(ab).toBe(ba);

    const abc1 = engine.afterAll(ab, c);
    const abc2 = engine.afterAll(a, engine.afterAll(b, c));
    expect(abc1).toBe(abc2);

    const aa = engine.afterAll(a, a);
    expect(aa).toBe(a);
  });

  it("afterAll rejects empty input", () => {
    const engine = new Engine();
    expect(() => engine.afterAll()).toThrow(
      "afterAll requires at least one token",
    );
  });
});

describe("join rule", () => {
  it("updates tokGlobal and tokLoc with a fresh token", () => {
    const engine = new Engine();
    const before = engine._debugSnapshot().tokGlobal.id;

    const tokOut = engine.orderedAccess(7, "load");
    const snapshot = engine._debugSnapshot();

    expect(tokOut.id).not.toBe(before);
    expect(snapshot.tokGlobal.id).toBe(tokOut.id);
    expect(snapshot.tokLoc["7"].id).toBe(tokOut.id);
  });
});

describe("effectful ops", () => {
  it("advance tokGlobal even if the return value is dropped", () => {
    const engine = new Engine();
    const before = engine._debugSnapshot().tokGlobal.id;

    engine.emitEffect("noop");

    const after = engine._debugSnapshot().tokGlobal.id;
    expect(after).not.toBe(before);
  });
});

describe("debug plan helpers", () => {
  it("exposes a stable plan and simulated state", () => {
    const engine = new Engine();
    const token = engine._debugCreateToken();

    engine.orderedAccess(3, "store");

    const plan = engine._debug_buildPlan([token]);
    const simulated = engine._debug_simulateCommit(plan);
    const snapshot = engine._debugSnapshot();

    expect(plan.rootTokenIds).toEqual([token.id]);
    expect(simulated.tokGlobalId).toBe(snapshot.tokGlobal.id);
    expect(simulated.tokLocIds["3"]).toBe(snapshot.tokLoc["3"].id);
  });
});

describe("trace recorder", () => {
  it("records deterministic token events", () => {
    const engine = new Engine();
    engine.orderedAccess(2, "load");

    expect(engine.trace.snapshot()).toEqual([
      {
        type: "after_all",
        inputs: [0],
        output: 0,
        outputKey: "0",
      },
      {
        type: "effect",
        op: "ordered_load",
        input: 0,
        output: 1,
        locId: 2,
      },
      { type: "set_token", target: "global", token: 1 },
      { type: "set_token", target: "loc", locId: 2, token: 1 },
    ]);
  });
});
