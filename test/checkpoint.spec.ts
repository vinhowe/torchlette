import { describe, expect, it } from "vitest";
import { CheckpointImpureRegionError, Engine } from "../src";

describe("checkpoint recompute fences", () => {
  it("forbids pending-loc initialization during recompute", () => {
    const engine = new Engine();
    engine._debug_bindPendingLoc(1, 2);
    engine._debug_setRecomputeMode(true);

    expect(() => engine._debug_ensureInitialized(1)).toThrow(
      CheckpointImpureRegionError,
    );
  });

  it("forbids persistent loc stores during recompute", () => {
    const engine = new Engine();
    engine._debug_setLocRole(5, "persistent");
    engine._debug_setRecomputeMode(true);

    expect(() => engine._debug_recomputeLocStore(5)).toThrow(
      CheckpointImpureRegionError,
    );
  });

  it("forbids saved_state creation during recompute", () => {
    const engine = new Engine();
    engine._debug_setRecomputeMode(true);

    expect(() => engine._debug_writeSavedState()).toThrow(
      CheckpointImpureRegionError,
    );
  });

  it("forbids mutation of reachable bases during recompute", () => {
    const engine = new Engine();
    const pack = engine._debug_checkpointPack([7, 8]);
    engine._debug_startCheckpointRecompute(pack);

    expect(() => engine._debug_recomputeMutateBase(7, 1)).toThrow(
      CheckpointImpureRegionError,
    );
  });

  it("allows mutation of bases not in the reachable set during recompute", () => {
    const engine = new Engine();
    const pack = engine._debug_checkpointPack([2]);
    engine._debug_startCheckpointRecompute(pack);

    expect(() => engine._debug_recomputeMutateBase(9, 1)).not.toThrow();
  });

  it("records deterministic checkpoint trace events", () => {
    const engine = new Engine();
    const pack = engine._debug_checkpointPack([9, 9, 3]);
    engine._debug_startCheckpointRecompute(pack);
    engine._debug_finishCheckpointRecompute();

    const events = engine.trace
      .snapshot()
      .filter((event) => event.type.startsWith("checkpoint_"));

    expect(events).toEqual([
      {
        type: "checkpoint_pack",
        packId: pack.id,
        reachableBases: [3, 9],
      },
      {
        type: "checkpoint_recompute_start",
        packId: pack.id,
        reachableBases: [3, 9],
      },
      {
        type: "checkpoint_recompute_finish",
        packId: pack.id,
      },
    ]);
  });
});
