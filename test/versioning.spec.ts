import { describe, expect, it } from "vitest";
import { Engine } from "../src";

describe("versioning", () => {
  it("increments locLogicalVersion at schedule time", () => {
    const engine = new Engine();

    engine._debug_scheduleLocAccess(1);
    let snapshot = engine._debugSnapshot();
    expect(snapshot.locs["1"].locLogicalVersion).toBe(1);
    expect(snapshot.locs["1"].locVersion).toBe(0);

    engine._debug_scheduleLocAccess(1);
    snapshot = engine._debugSnapshot();
    expect(snapshot.locs["1"].locLogicalVersion).toBe(2);
    expect(snapshot.locs["1"].locVersion).toBe(0);

    const locScheduleEvents = engine.trace
      .snapshot()
      .filter((event) => event.type === "loc_schedule");
    expect(locScheduleEvents).toEqual([
      { type: "loc_schedule", locId: 1, locLogicalVersion: 1 },
      { type: "loc_schedule", locId: 1, locLogicalVersion: 2 },
    ]);
  });

  it("increments locVersion only on committed stores", () => {
    const engine = new Engine();

    engine._debug_scheduleLocAccess(3);
    let snapshot = engine._debugSnapshot();
    expect(snapshot.locs["3"].locVersion).toBe(0);

    engine._debug_commitLocStore(3);
    snapshot = engine._debugSnapshot();
    expect(snapshot.locs["3"].locVersion).toBe(1);

    engine._debug_scheduleLocAccess(3);
    snapshot = engine._debugSnapshot();
    expect(snapshot.locs["3"].locVersion).toBe(1);

    engine._debug_commitLocStore(3);
    snapshot = engine._debugSnapshot();
    expect(snapshot.locs["3"].locVersion).toBe(2);

    const locCommitEvents = engine.trace
      .snapshot()
      .filter((event) => event.type === "loc_commit");
    expect(locCommitEvents).toEqual([
      { type: "loc_commit", locId: 3, locVersion: 1 },
      { type: "loc_commit", locId: 3, locVersion: 2 },
    ]);
  });

  it("increments baseCommitVersion only via base_commit and enforces uniqueness", () => {
    const engine = new Engine();

    expect(engine._debugSnapshot().bases["5"]).toBeUndefined();

    const first = engine._debug_baseCommit(5, 10);
    expect(first.baseCommitVersion).toBe(1);
    expect(first.committedMutations).toEqual([10]);

    expect(() => engine._debug_baseCommit(5, 10)).toThrow(
      "base_commit already recorded for mutId 10",
    );

    const second = engine._debug_baseCommit(5, 11);
    expect(second.baseCommitVersion).toBe(2);
    expect(second.committedMutations).toEqual([10, 11]);

    const baseCommitEvents = engine.trace
      .snapshot()
      .filter((event) => event.type === "base_commit");
    expect(baseCommitEvents).toEqual([
      { type: "base_commit", baseId: 5, mutId: 10, baseCommitVersion: 1 },
      { type: "base_commit", baseId: 5, mutId: 11, baseCommitVersion: 2 },
    ]);
  });
});
