import { describe, expect, it } from "vitest";

import { Engine, type EngineTensor, PoisonedEngineError } from "../src";

describe("lifecycle: tidy/keep/dispose", () => {
  it("disposes tensors created in tidy unless kept or returned", () => {
    const engine = new Engine();
    let kept: EngineTensor | null = null;
    let dropped: EngineTensor | null = null;

    const returned = engine.tidy(() => {
      const a = engine.createTensor();
      const b = engine.createTensor();
      const c = engine.createTensor();

      engine.keep(b);
      kept = b;
      dropped = c;

      return a;
    });

    expect(returned.disposed).toBe(false);
    expect(returned.escapes).toBe(true);
    expect(kept?.disposed).toBe(false);
    expect(kept?.escapes).toBe(true);
    expect(dropped?.disposed).toBe(true);

    expect(engine._debug_getBasePinCount(returned.baseId)).toBe(1);
    expect(engine._debug_getBasePinCount(kept?.baseId ?? 0)).toBe(1);
    expect(engine._debug_getBasePinCount(dropped?.baseId ?? 0)).toBe(0);
  });

  it("allows cleanup-only disposal while busy or poisoned", () => {
    const engine = new Engine();
    const tensor = engine.createTensor();

    engine._debug_runEntryPoint(() => {
      expect(() => engine.dispose(tensor)).not.toThrow();
    });

    const tensor2 = engine.createTensor();
    engine._debug_poison();
    expect(() => engine.dispose(tensor2)).not.toThrow();
  });
});

describe("lifecycle: markStep", () => {
  it("resets tokens and clears loc tokens", async () => {
    const engine = new Engine();
    engine.orderedAccess(1, "access");

    const before = engine._debugSnapshot();
    expect(Object.keys(before.tokLoc)).toHaveLength(1);

    await engine.markStep();

    const after = engine._debugSnapshot();
    expect(after.tokGlobal.id).toBe(0);
    expect(Object.keys(after.tokLoc)).toHaveLength(0);
  });

  it("finalizes pending loc bindings when loc has value", async () => {
    const engine = new Engine();
    const tensor = engine.createTensor();

    engine._debug_bindPendingLoc(tensor.baseId, 7);
    engine._debug_commitLocStore(7);

    const before = engine._debugSnapshot();
    expect(before.bindings[tensor.baseId.toString()]?.kind).toBe("pending_loc");

    await engine.markStep();

    const after = engine._debugSnapshot();
    expect(after.bindings[tensor.baseId.toString()]?.kind).toBe("loc");
  });

  it("drains finalizeQueue during markStep", async () => {
    const engine = new Engine();
    engine._debug_enqueueFinalize({ id: 1 });

    await engine.markStep();

    const drains = engine.trace
      .snapshot()
      .filter((event) => event.type === "finalize_drain");
    expect(drains[0]).toEqual({ type: "finalize_drain", count: 1 });
  });

  it("throws when poisoned", async () => {
    const engine = new Engine();
    engine._debug_poison();

    await expect(engine.markStep()).rejects.toThrow(PoisonedEngineError);
  });
});
