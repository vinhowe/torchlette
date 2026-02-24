import { EngineTensor, type RngBasis, type TraceTensor } from "./engine-types";

export function collectTensorHandles(value: unknown): EngineTensor[] {
  const out: EngineTensor[] = [];
  const seen = new Set<unknown>();

  const visit = (current: unknown) => {
    if (current === null || current === undefined) {
      return;
    }
    if (current instanceof EngineTensor) {
      out.push(current);
      return;
    }
    if (typeof current !== "object") {
      return;
    }
    if (seen.has(current)) {
      return;
    }
    seen.add(current);
    if (Array.isArray(current)) {
      for (const entry of current) {
        visit(entry);
      }
      return;
    }
    for (const entry of Object.values(current as Record<string, unknown>)) {
      visit(entry);
    }
  };

  visit(value);
  return out;
}

export function isThenable(value: unknown): value is Promise<unknown> {
  if (!value) {
    return false;
  }
  return typeof (value as Promise<unknown>).then === "function";
}

export function collectTraceTensorIds(value: unknown): number[] {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => collectTraceTensorIds(item));
  }
  if (isTraceTensor(value)) {
    return [value.id];
  }
  return [];
}

export function isTraceTensor(value: unknown): value is TraceTensor {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as TraceTensor;
  return typeof record.id === "number" && typeof record.epoch === "number";
}

export function computeRngValue(
  basis: RngBasis,
  opNonce: number,
  drawNonce: number,
): number {
  const seed = basis.seed >>> 0;
  const algo = basis.algorithmId >>> 0;
  const op = opNonce >>> 0;
  const draw = drawNonce >>> 0;
  let state =
    seed ^ Math.imul(algo, 0x9e3779b9) ^ op ^ Math.imul(draw, 0x85ebca6b);
  state = mix32(state);
  return (state >>> 0) / 2 ** 32;
}

export function mix32(value: number): number {
  let v = value >>> 0;
  v ^= v >>> 16;
  v = Math.imul(v, 0x7feb352d);
  v ^= v >>> 15;
  v = Math.imul(v, 0x846ca68b);
  v ^= v >>> 16;
  return v >>> 0;
}
