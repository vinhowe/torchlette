import { describe, expect, it } from "vitest";
import { Engine } from "../src";

describe("plan simulation", () => {
  it("predicts commit versions from trace events", () => {
    const engine = new Engine();
    engine._debug_scheduleLocAccess(1);
    engine._debug_commitLocStore(1);
    engine._debug_scheduleLocAccess(2);
    engine._debug_commitLocStore(2);
    engine._debug_baseCommit(5, 7);

    const planA = engine._debug_buildPlanFromTrace();
    const planB = engine._debug_buildPlanFromTrace(engine.trace.snapshot());

    expect(planA.eventKeys).toEqual(planB.eventKeys);
    expect(planA.orderedEvents.map((event) => event.name)).toEqual([
      "loc_schedule",
      "loc_commit",
      "loc_schedule",
      "loc_commit",
      "base_commit",
    ]);

    const predicted = engine._debug_simulateCommitPlan(planA);
    expect(predicted.locLogicalVersions).toEqual({ "1": 1, "2": 1 });
    expect(predicted.locVersions).toEqual({ "1": 1, "2": 1 });
    expect(predicted.baseCommitVersions).toEqual({ "5": 1 });
    expect(predicted.baseCommittedMutations).toEqual({ "5": [7] });
  });

  it("orders rng draws by drawNonce when opNonce ties", () => {
    const engine = new Engine();
    engine._debug_setRngBasis({ algorithmId: 0, seed: 1 });

    engine._debug_random(5);
    engine._debug_random(5);

    const plan = engine._debug_buildPlanFromTrace(engine.trace.snapshot());
    const basisEvents = plan.orderedEvents.filter(
      (event) => event.key.kind === "rng_basis",
    );
    const rngEvents = plan.orderedEvents.filter(
      (event) => event.key.kind === "rng_draw",
    );

    expect(basisEvents).toHaveLength(1);
    expect(rngEvents).toHaveLength(2);
    expect(rngEvents.map((event) => event.key.drawNonce)).toEqual([1, 2]);
  });

  it("records checkpoint rng lifecycle events", () => {
    const engine = new Engine();
    engine._debug_setRngBasis({ algorithmId: 1, seed: 2 });
    engine._debug_startCheckpointRecord();
    engine._debug_random(1);
    const draws = engine._debug_finishCheckpointRecord();
    engine._debug_startCheckpointReplay(draws);
    engine._debug_random(1);
    engine._debug_finishCheckpointReplay();

    const plan = engine._debug_buildPlanFromTrace();
    const lifecycleKinds = plan.orderedEvents
      .map((event) => event.key.kind)
      .filter((kind) => kind.startsWith("rng_checkpoint_"));

    expect(lifecycleKinds).toEqual([
      "rng_checkpoint_record_start",
      "rng_checkpoint_record_finish",
      "rng_checkpoint_replay_start",
      "rng_checkpoint_replay_finish",
    ]);
  });

  it("includes publish_save in the deterministic plan order", () => {
    const engine = new Engine();
    engine._debug_publishSave(1);
    engine._debug_publishSave(2);

    const plan = engine._debug_buildPlanFromTrace(engine.trace.snapshot());
    const predicted = engine._debug_simulateCommitPlan(plan);
    const publishKinds = plan.orderedEvents
      .filter((event) => event.key.kind === "publish_save")
      .map((event) => event.key.kind);

    expect(publishKinds).toEqual(["publish_save", "publish_save"]);
    expect(predicted.publishSaveCount).toBe(2);
  });
});
