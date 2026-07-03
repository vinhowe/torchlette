/**
 * Engine epochs — the monotonic id GPU-timeline lifetimes are keyed by.
 *
 * Stage 0 of the scoped-memory campaign (docs/scoped-memory-design.md §1):
 * markStep() is four fused roles, and the three GPU-timeline roles
 * (buffer-pool epoch, planner/replay scope + generation stamps, quiesce
 * ordering) are re-homed onto this counter. Steps CONSUME epochs (a step
 * boundary bumps the epoch; a quiesce bumps the epoch) rather than owning
 * them; the reclamation-boundary role (snapshot/releaseStepTemps diff)
 * still lives in the step machinery.
 *
 * The counter is a pure monotonic id. Nothing may depend on epoch values
 * being dense or step-aligned — consumers compare ids relatively
 * (`stamp > boundaryEpoch`) or for identity (planner registry lifetime).
 * GPU side effects (the pendingRelease → pool flush) live in the backend's
 * `advanceEpoch` wrapper (src/backend/webgpu/epoch.ts), NOT here: bumping
 * the counter is always safe; flushing the pool is only safe at
 * fence-completed / encoder-closed points.
 *
 * Tracing (the behavioral-identity gate for stages 0/2/3):
 *   TORCHLETTE_TRACE_EPOCHS=1  — ordered lifecycle event trace (pool
 *     flushes, demotion sweeps, releaseStepTemps, deferred-destroy batches).
 *     Level 1 deliberately EXCLUDES epoch ids so traces diff cleanly across
 *     refactors that change how often the counter bumps.
 *   TORCHLETTE_TRACE_EPOCHS=2  — level 1 plus epoch-advance lines.
 */
import { ENV } from "./env";

const TRACE_LEVEL = (() => {
  const v = ENV.TORCHLETTE_TRACE_EPOCHS;
  if (v === undefined || v === "" || v === "0") return 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 1;
})();

/** Emit an ordered lifecycle trace event (level >= 1). */
export function epochTrace(event: string): void {
  if (TRACE_LEVEL >= 1) console.log(`[etrace] ${event}`);
}

export function epochTraceEnabled(): boolean {
  return TRACE_LEVEL >= 1;
}

let epochId = 0;

/** The current engine epoch id. */
export function currentEpoch(): number {
  return epochId;
}

/**
 * Advance the engine epoch (counter only — no GPU side effects; see
 * src/backend/webgpu/epoch.ts for the pool-flushing quiesce advance).
 * Returns the NEW epoch id.
 */
export function bumpEpoch(reason: string): number {
  epochId++;
  if (TRACE_LEVEL >= 2) console.log(`[etrace2] epoch ${epochId} ${reason}`);
  return epochId;
}
