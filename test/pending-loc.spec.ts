import { describe, expect, it } from "vitest";
import { CheckpointImpureRegionError, Engine } from "../src";

describe("pending loc initialization", () => {
  it("creates initTok via materialize store on first access", () => {
    const engine = new Engine();
    engine._debug_bindPendingLoc(1, 9);

    engine._debug_orderedAccessBase(1, "load");

    const snapshot = engine._debugSnapshot();
    const binding = snapshot.bindings["1"];
    expect(binding.kind).toBe("pending_loc");
    expect(binding.initTokId).toBeDefined();
    expect(binding.initTokKind).toBe("effect");

    const effects = engine.trace
      .snapshot()
      .filter((event) => event.type === "effect");

    expect(effects[0]).toMatchObject({
      op: "init_loc_store",
      output: binding.initTokId,
    });
    expect(effects[1]).toMatchObject({
      op: "ordered_load",
      input: binding.initTokId,
    });
  });

  it("uses a token-only init when subsumed by an initializing store", () => {
    const engine = new Engine();
    engine._debug_bindPendingLoc(2, 5);

    const initTok = engine._debug_ensureInitialized(2, {
      subsumedByStore: true,
    });

    const snapshot = engine._debugSnapshot();
    const binding = snapshot.bindings["2"];
    expect(binding.initTokId).toBe(initTok.id);
    expect(binding.initTokKind).toBe("token_only");
  });

  it("throws when ensuring initialization during recompute for persistent locs", () => {
    const engine = new Engine();
    engine._debug_bindPendingLoc(3, 7);
    engine._debug_setLocRole(7, "persistent");
    engine._debug_setRecomputeMode(true);

    expect(() => engine._debug_ensureInitialized(3)).toThrow(
      CheckpointImpureRegionError,
    );
  });
});
