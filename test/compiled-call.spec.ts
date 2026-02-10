import { describe, expect, it } from "vitest";
import { Engine, type SemanticSubeventSchedule } from "../src";

describe("compiled-call schedule expansion", () => {
  it("sets planInstanceId to callInstanceId for subevents", () => {
    const engine = new Engine();
    const schedule: SemanticSubeventSchedule = {
      graphInstanceId: 1,
      callInstanceId: 7,
      subevents: [
        {
          kind: "loc_commit",
          opNonce: 1,
          payload: { locId: 2 },
        },
        {
          kind: "base_commit",
          opNonce: 2,
          mutId: 5,
          payload: { baseId: 3, mutId: 5 },
        },
      ],
    };

    const plan = engine._debug_buildPlanFromSchedules([schedule]);
    const compiledEvents = plan.orderedEvents.filter(
      (event) => event.key.callInstanceId === schedule.callInstanceId,
    );

    expect(compiledEvents).toHaveLength(2);
    for (const event of compiledEvents) {
      expect(event.key.planInstanceId).toBe(schedule.callInstanceId);
      expect(event.key.graphInstanceId).toBe(schedule.graphInstanceId);
    }
  });

  it("expands compiled-call traces into schedules", () => {
    const engine = new Engine();
    const schedule: SemanticSubeventSchedule = {
      graphInstanceId: 2,
      callInstanceId: 9,
      subevents: [
        {
          kind: "loc_schedule",
          opNonce: 1,
          payload: { locId: 4 },
        },
      ],
    };

    engine._debug_emitCompiledCall(2, 9);

    const plan = engine._debug_buildPlanWithCompiledCalls({
      9: schedule,
    });

    const compiled = plan.orderedEvents.filter(
      (event) => event.key.callInstanceId === 9,
    );
    expect(compiled).toHaveLength(1);
    expect(compiled[0].key.planInstanceId).toBe(9);
    expect(compiled[0].key.graphInstanceId).toBe(2);
  });

  it("applies compiled-call subevents in commit simulation", () => {
    const engine = new Engine();
    const schedule: SemanticSubeventSchedule = {
      graphInstanceId: 3,
      callInstanceId: 11,
      subevents: [
        {
          kind: "loc_schedule",
          opNonce: 1,
          payload: { locId: 6 },
        },
        {
          kind: "loc_commit",
          opNonce: 2,
          payload: { locId: 6 },
        },
        {
          kind: "base_commit",
          opNonce: 3,
          mutId: 42,
          payload: { baseId: 7, mutId: 42 },
        },
        {
          kind: "publish_save",
          opNonce: 4,
        },
      ],
    };

    engine._debug_emitCompiledCall(3, 11);

    const plan = engine._debug_buildPlanWithCompiledCalls({
      11: schedule,
    });
    const predicted = engine._debug_simulateCommitPlan(plan);

    expect(predicted.locLogicalVersions).toEqual({ "6": 1 });
    expect(predicted.locVersions).toEqual({ "6": 1 });
    expect(predicted.baseCommitVersions).toEqual({ "7": 1 });
    expect(predicted.baseCommittedMutations).toEqual({ "7": [42] });
    expect(predicted.publishSaveCount).toBe(1);
  });
});
