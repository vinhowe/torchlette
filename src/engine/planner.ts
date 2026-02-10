export type EventKind = string;

export interface EventKey {
  graphInstanceId: number;
  callInstanceId: number;
  planInstanceId: number;
  opNonce: number;
  drawNonce: number;
  mutId: number;
  kind: EventKind;
}

export interface PlanEvent {
  key: EventKey;
  name: string;
  payload?: Record<string, number>;
}

export interface SemanticSubevent {
  kind: EventKind;
  opNonce: number;
  drawNonce?: number;
  mutId?: number;
  payload?: Record<string, number>;
}

export interface SemanticSubeventSchedule {
  graphInstanceId: number;
  callInstanceId: number;
  subevents: SemanticSubevent[];
}

export interface DebugPlanLinearOrder {
  orderedEvents: PlanEvent[];
  eventKeys: EventKey[];
}

export function compareEventKey(a: EventKey, b: EventKey): number {
  if (a.graphInstanceId !== b.graphInstanceId) {
    return a.graphInstanceId - b.graphInstanceId;
  }
  if (a.callInstanceId !== b.callInstanceId) {
    return a.callInstanceId - b.callInstanceId;
  }
  if (a.planInstanceId !== b.planInstanceId) {
    return a.planInstanceId - b.planInstanceId;
  }
  if (a.opNonce !== b.opNonce) {
    return a.opNonce - b.opNonce;
  }
  if (a.drawNonce !== b.drawNonce) {
    return a.drawNonce - b.drawNonce;
  }
  if (a.mutId !== b.mutId) {
    return a.mutId - b.mutId;
  }
  if (a.kind < b.kind) {
    return -1;
  }
  if (a.kind > b.kind) {
    return 1;
  }
  return 0;
}

export function buildPlanLinearOrder(
  events: PlanEvent[],
): DebugPlanLinearOrder {
  const orderedEvents = events
    .slice()
    .sort((left, right) => compareEventKey(left.key, right.key));
  return {
    orderedEvents,
    eventKeys: orderedEvents.map((event) => event.key),
  };
}

export function expandSemanticSubeventSchedule(
  schedule: SemanticSubeventSchedule,
): PlanEvent[] {
  return schedule.subevents.map((subevent) => ({
    name: subevent.kind,
    key: {
      graphInstanceId: schedule.graphInstanceId,
      callInstanceId: schedule.callInstanceId,
      planInstanceId: schedule.callInstanceId,
      opNonce: subevent.opNonce,
      drawNonce: subevent.drawNonce ?? 0,
      mutId: subevent.mutId ?? 0,
      kind: subevent.kind,
    },
    payload: subevent.payload,
  }));
}
