/**
 * Step-tape MEASUREMENT seam (docs/staged-execution-phase1.md, G0/G7).
 *
 * TORCHLETTE_TAPE_PROFILE=1 — per-step wall-time accumulators at the exact
 * seams the step-tape skips (plan collect / fingerprint / CSE+rewrite passes /
 * template lookup / compiled-replay JS vs Dawn calls). Default-off,
 * observation-only (every call site is guarded by a module-load-time const, so
 * the flag-off cost is one boolean test). Consumed by
 * examples/qwen3/timeline-decode.ts (the G0 table) and by profile drivers to
 * confirm WHICH seams a tape replay skipped (G7).
 *
 * SUNSET: this module, the flag, and every `TAPE_PROFILE` guard are DELETED at
 * the step-tape default-flip (phase 2+), together with TORCHLETTE_STEP_TAPE.
 * (The 1a-only slot-diff image recorder was removed in 1c — superseded by the
 * recorder's guard-3 byte-diff in src/core/step-tape.ts.)
 * Measurement instrumentation only; must not grow behavior.
 */

import { ENV } from "./env";

export const TAPE_PROFILE: boolean = ENV.TORCHLETTE_TAPE_PROFILE === "1";

// ---------------------------------------------------------------------------
// Wall-time accumulators (TORCHLETTE_TAPE_PROFILE)
// ---------------------------------------------------------------------------

const acc: Record<string, number> = Object.create(null);
const cnt: Record<string, number> = Object.create(null);

export function tpAdd(key: string, ms: number): void {
  acc[key] = (acc[key] ?? 0) + ms;
  cnt[key] = (cnt[key] ?? 0) + 1;
}

/** Reset all accumulators (harness calls this at each step start). */
export function tpReset(): void {
  for (const k of Object.keys(acc)) delete acc[k];
  for (const k of Object.keys(cnt)) delete cnt[k];
}

/** Snapshot accumulated ms + call counts since the last tpReset(). */
export function tpGet(): {
  ms: Record<string, number>;
  counts: Record<string, number>;
} {
  return { ms: { ...acc }, counts: { ...cnt } };
}
