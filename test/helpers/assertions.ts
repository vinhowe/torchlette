import { expect } from "vitest";

export type Payload = { shape: number[]; values: (number | null)[] };

const DEFAULT_ATOL = 1e-5;
const DEFAULT_RTOL = 1e-4;

/**
 * Assert two tensor payloads are element-wise close within tolerance.
 * Null values (NaN/Inf from oracle) are skipped.
 */
export function assertClose(
  actual: Payload,
  expected: Payload,
  atol = DEFAULT_ATOL,
  rtol = DEFAULT_RTOL,
  label = "",
): void {
  expect(actual.shape).toEqual(expected.shape);
  for (let i = 0; i < actual.values.length; i++) {
    const a = actual.values[i];
    const e = expected.values[i];
    if (a === null || e === null) continue;
    const diff = Math.abs(a - e);
    expect(
      diff,
      `${label ? label + " " : ""}index ${i}: actual=${a}, expected=${e}, diff=${diff}`,
    ).toBeLessThanOrEqual(atol + rtol * Math.abs(e));
  }
}
