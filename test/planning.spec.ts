import { describe, expect, it } from "vitest";
import { buildPlanLinearOrder, type EventKey, type PlanEvent } from "../src";

function key(
  graphInstanceId: number,
  callInstanceId: number,
  planInstanceId: number,
  opNonce: number,
  drawNonce: number,
  mutId: number,
  kind: string,
): EventKey {
  return {
    graphInstanceId,
    callInstanceId,
    planInstanceId,
    opNonce,
    drawNonce,
    mutId,
    kind,
  };
}

describe("plan linear order", () => {
  it("orders events by lexicographic EventKey", () => {
    const events: PlanEvent[] = [
      {
        name: "b",
        key: key(1, 1, 1, 2, 0, 0, "effect"),
      },
      {
        name: "a",
        key: key(1, 1, 1, 1, 0, 0, "effect"),
      },
      {
        name: "c",
        key: key(1, 2, 1, 0, 0, 0, "effect"),
      },
      {
        name: "d",
        key: key(2, 0, 0, 0, 0, 0, "effect"),
      },
    ];

    const plan = buildPlanLinearOrder(events);

    expect(plan.orderedEvents.map((event) => event.name)).toEqual([
      "a",
      "b",
      "c",
      "d",
    ]);
  });

  it("is deterministic across insertion order", () => {
    const events: PlanEvent[] = [
      { name: "x", key: key(1, 1, 1, 3, 0, 0, "effect") },
      { name: "y", key: key(1, 1, 1, 1, 0, 0, "effect") },
      { name: "z", key: key(1, 1, 1, 2, 0, 0, "effect") },
    ];

    const planA = buildPlanLinearOrder([events[0], events[1], events[2]]);
    const planB = buildPlanLinearOrder([events[2], events[0], events[1]]);

    expect(planA.eventKeys).toEqual(planB.eventKeys);
  });

  it("breaks ties by kind when numeric fields match", () => {
    const events: PlanEvent[] = [
      { name: "b", key: key(1, 1, 1, 1, 0, 0, "zeta") },
      { name: "a", key: key(1, 1, 1, 1, 0, 0, "alpha") },
    ];

    const plan = buildPlanLinearOrder(events);
    expect(plan.orderedEvents.map((event) => event.name)).toEqual(["a", "b"]);
  });

  it("orders by drawNonce when opNonce ties", () => {
    const events: PlanEvent[] = [
      { name: "late", key: key(1, 1, 1, 5, 2, 0, "random") },
      { name: "early", key: key(1, 1, 1, 5, 1, 0, "random") },
    ];

    const plan = buildPlanLinearOrder(events);
    expect(plan.orderedEvents.map((event) => event.name)).toEqual([
      "early",
      "late",
    ]);
  });
});
