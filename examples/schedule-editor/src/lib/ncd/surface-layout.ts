import { axisById, wireById, wireElementsAtLevel } from "./model";
import type { NcdLevel, NcdTerm, NcdWire } from "./types";

export interface SurfaceColumnCost {
  column: number;
  memory: Record<NcdLevel, number>;
  transfer: Record<NcdLevel, number>;
  cumulative: Record<NcdLevel, number>;
}

export interface WireSurfaceLane {
  wire: NcdWire;
  y: number;
  height: number;
}

export interface SurfaceJam {
  target: "axis" | "box" | "residency";
  id: string;
  column?: number;
  reason: string;
}

export interface SurfaceEquivalence {
  before: NcdTerm;
  after: NcdTerm;
  label: string;
  nonce: number;
}

export const SURFACE_COLUMN_WIDTH = 176;
export const SURFACE_LEFT_GUTTER = 112;
export const SURFACE_TOP = 172;

export function surfaceColumnX(term: NcdTerm, column: number): number {
  const ordered = [...term.semantic.columns].sort((a, b) => a.index - b.index);
  const rank = ordered.findIndex((item) => item.index === column);
  return SURFACE_LEFT_GUTTER + Math.max(rank, 0) * SURFACE_COLUMN_WIDTH;
}

export function surfaceLanes(term: NcdTerm): WireSurfaceLane[] {
  let cursor = SURFACE_TOP;
  return term.semantic.wires.map((wire, index) => {
    const height = Math.max(62, wire.axisIds.length * 17 + 26);
    const lane = { wire, y: cursor, height };
    const sameTuple =
      wire.tupleGroup &&
      term.semantic.wires[index + 1]?.tupleGroup === wire.tupleGroup;
    cursor += height + (sameTuple ? 6 : 20);
    return lane;
  });
}

export function surfaceWorldSize(term: NcdTerm): {
  width: number;
  height: number;
} {
  const lanes = surfaceLanes(term);
  const last = lanes.at(-1);
  return {
    width:
      SURFACE_LEFT_GUTTER +
      Math.max(1, term.semantic.columns.length) * SURFACE_COLUMN_WIDTH,
    height: (last?.y ?? SURFACE_TOP) + (last?.height ?? 80) + 80,
  };
}

export function residencyAt(term: NcdTerm, wireId: string, column: number) {
  return term.decorations.residency.find(
    (item) => item.wireId === wireId && item.column === column,
  );
}

export function orderedResidencies(term: NcdTerm, wireId: string) {
  return term.decorations.residency
    .filter((item) => item.wireId === wireId)
    .sort((a, b) => a.column - b.column);
}

export function transitionKind(
  term: NcdTerm,
  wireId: string,
  column: number,
): "load" | "save" | "resident" | "global" {
  const states = orderedResidencies(term, wireId);
  const index = states.findIndex((item) => item.column === column);
  const state = states[index];
  const previous = states[index - 1];
  if (!state || state.level === "l0") {
    return previous?.level === "l1" ? "save" : "global";
  }
  return previous?.level === "l0" ? "load" : "resident";
}

export function wireExpression(term: NcdTerm, wire: NcdWire): string {
  return wire.axisIds.map((axisId) => axisById(term, axisId).label).join("·");
}

export function surfaceColumnCosts(term: NcdTerm): SurfaceColumnCost[] {
  const columns = [...term.semantic.columns].sort((a, b) => a.index - b.index);
  const cumulative: Record<NcdLevel, number> = { l0: 0, l1: 0 };

  return columns.map((column) => {
    const memory: Record<NcdLevel, number> = { l0: 0, l1: 0 };
    const transfer: Record<NcdLevel, number> = { l0: 0, l1: 0 };

    for (const state of term.decorations.residency.filter(
      (item) => item.column === column.index,
    )) {
      const wire = wireById(term, state.wireId);
      memory[state.level] += wireElementsAtLevel(term, wire, state.level);

      const states = orderedResidencies(term, state.wireId);
      const stateIndex = states.findIndex(
        (item) => item.column === column.index,
      );
      const previous = states[stateIndex - 1];
      if (previous && previous.level !== state.level) {
        const chargedLevel: NcdLevel =
          previous.level === "l1" || state.level === "l1" ? "l1" : "l0";
        transfer[chargedLevel] += wireElementsAtLevel(term, wire, chargedLevel);
      }
    }

    cumulative.l0 += transfer.l0;
    cumulative.l1 += transfer.l1;
    return {
      column: column.index,
      memory,
      transfer,
      cumulative: { ...cumulative },
    };
  });
}
