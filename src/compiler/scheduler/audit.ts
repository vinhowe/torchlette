/**
 * Scheduler audit: for each execution plan, run the live-range analyzer +
 * scheduler and record how much total buffer memory could be saved if
 * slots were reused according to first-fit / best-fit assignments.
 *
 * This is a derisking diagnostic — it never changes execution behavior.
 * Enabled by `TORCHLETTE_SCHEDULER_AUDIT=1`. Zero cost when disabled.
 *
 * Use `printSchedulerAuditSummary()` at end-of-run to see aggregate numbers
 * (e.g. in profile-training.ts or a training script).
 */
import { ENV } from "../../core/env";
import type { ExecutionPlan } from "../../graph/types";
import { computePeak, totalBytes, trivialAssignment } from "./cost-model";
import { analyzeLiveRanges } from "./live-range";
import { bestFitScheduler, firstFitScheduler } from "./schedulers";

// Read lazily so tools/tests can flip the flag at runtime.
function isEnabled(): boolean {
  return !!ENV.TORCHLETTE_SCHEDULER_AUDIT;
}

interface AuditSample {
  readonly nodes: number;
  readonly externalCount: number;
  readonly externalBytes: number;
  /** Trivial = one slot per node — sum of all tensor sizes. */
  readonly trivialTotalBytes: number;
  readonly firstFitTotalBytes: number;
  readonly bestFitTotalBytes: number;
  readonly firstFitSlots: number;
  readonly bestFitSlots: number;
  /** Theoretical lower bound: peak simultaneous bytes live. */
  readonly peakLiveBytes: number;
}

const samples: AuditSample[] = [];

/** Schedule the plan and record a sample. No-op when disabled. */
export function auditPlan(
  plan: ExecutionPlan,
  externalNodeIds?: ReadonlySet<number>,
): void {
  if (!isEnabled()) return;
  if (plan.nodes.length === 0) return;

  const ranges = analyzeLiveRanges(plan.nodes, externalNodeIds);
  const trivial = trivialAssignment(ranges);
  const ff = firstFitScheduler(ranges);
  const bf = bestFitScheduler(ranges);

  let externalCount = 0;
  let externalBytes = 0;
  for (const r of ranges.values()) {
    if (r.external) {
      externalCount++;
      externalBytes += r.size;
    }
  }

  samples.push({
    nodes: plan.nodes.length,
    externalCount,
    externalBytes,
    trivialTotalBytes: totalBytes(trivial),
    firstFitTotalBytes: totalBytes(ff),
    bestFitTotalBytes: totalBytes(bf),
    firstFitSlots: ff.slotSizes.size,
    bestFitSlots: bf.slotSizes.size,
    peakLiveBytes: computePeak(ranges, trivial),
  });
}

export function getSchedulerAuditSamples(): readonly AuditSample[] {
  return samples;
}

export function resetSchedulerAudit(): void {
  samples.length = 0;
}

export function isSchedulerAuditEnabled(): boolean {
  return isEnabled();
}

/** Dump per-plan table to stderr (for verbose investigation). */
export function printSchedulerAuditPerPlan(): void {
  if (!isEnabled()) return;
  if (samples.length === 0) return;
  const mb = (b: number) => (b / (1024 * 1024)).toFixed(1).padStart(7);
  console.error("");
  console.error("[scheduler-audit] per-plan breakdown:");
  console.error(
    "   idx nodes ext  trivMB  ffMB   bfMB  peakMB  schedulableMB  ff-save%",
  );
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const schedTrivial = s.trivialTotalBytes - s.externalBytes;
    const schedFf = s.firstFitTotalBytes - s.externalBytes;
    const savePct =
      schedTrivial > 0
        ? (((schedTrivial - schedFf) / schedTrivial) * 100).toFixed(0)
        : "--";
    console.error(
      `   ${String(i).padStart(3)} ${String(s.nodes).padStart(5)} ` +
        `${String(s.externalCount).padStart(4)} ` +
        `${mb(s.trivialTotalBytes)} ${mb(s.firstFitTotalBytes)} ` +
        `${mb(s.bestFitTotalBytes)} ${mb(s.peakLiveBytes)} ` +
        `${mb(schedTrivial)}→${mb(schedFf)} ${savePct.padStart(7)}%`,
    );
  }
  console.error("");
}

/** Pretty-print an aggregate summary to stderr. */
export function printSchedulerAuditSummary(): void {
  if (!isEnabled()) return;
  if (samples.length === 0) {
    console.error("[scheduler-audit] no plans recorded");
    return;
  }

  // Steady-state: skip the first sample (usually a materialization plan).
  const steady = samples.length > 1 ? samples.slice(1) : samples;

  let totalNodes = 0;
  let totalExternal = 0;
  let totalExternalBytes = 0;
  let totalTrivial = 0;
  let totalFirstFit = 0;
  let totalBestFit = 0;
  let totalPeak = 0;
  let maxPlanNodes = 0;
  let maxTrivial = 0;
  let maxFirstFit = 0;
  let maxPeakLive = 0;
  let maxExternalBytes = 0;
  for (const s of steady) {
    totalNodes += s.nodes;
    totalExternal += s.externalCount;
    totalExternalBytes += s.externalBytes;
    totalTrivial += s.trivialTotalBytes;
    totalFirstFit += s.firstFitTotalBytes;
    totalBestFit += s.bestFitTotalBytes;
    totalPeak += s.peakLiveBytes;
    if (s.nodes > maxPlanNodes) maxPlanNodes = s.nodes;
    if (s.trivialTotalBytes > maxTrivial) maxTrivial = s.trivialTotalBytes;
    if (s.firstFitTotalBytes > maxFirstFit) maxFirstFit = s.firstFitTotalBytes;
    if (s.peakLiveBytes > maxPeakLive) maxPeakLive = s.peakLiveBytes;
    if (s.externalBytes > maxExternalBytes) maxExternalBytes = s.externalBytes;
  }

  const n = steady.length;
  const mb = (b: number) => (b / (1024 * 1024)).toFixed(1);
  const pct = (saved: number, base: number) =>
    base > 0 ? ((saved / base) * 100).toFixed(1) : "0.0";

  console.error("");
  console.error("=".repeat(72));
  console.error(
    `[scheduler-audit] ${samples.length} plans recorded ` +
      `(${n} steady-state, skipped ${samples.length - n})`,
  );
  console.error("=".repeat(72));
  console.error(`  avg nodes/plan:        ${(totalNodes / n).toFixed(0)}`);
  console.error(`  max nodes/plan:        ${maxPlanNodes}`);
  console.error(
    `  avg external nodes:    ${(totalExternal / n).toFixed(0)} ` +
      `(${mb(totalExternalBytes / n)}MB — saved-for-backward, always live)`,
  );
  console.error("");
  // Externals (saved-for-backward) are unschedulable — each needs its own
  // slot regardless of algorithm. Break out internal-only savings.
  const avgExt = totalExternalBytes / n;
  const avgTrivInt = totalTrivial / n - avgExt;
  const avgFfInt = totalFirstFit / n - avgExt;
  const avgBfInt = totalBestFit / n - avgExt;
  console.error("  Per-plan averages:");
  console.error(
    `    trivial   totalBytes:  ${mb(totalTrivial / n).padStart(8)}MB ` +
      `(1 slot per tensor)`,
  );
  console.error(
    `    first-fit totalBytes:  ${mb(totalFirstFit / n).padStart(8)}MB ` +
      `(saved ${pct(totalTrivial - totalFirstFit, totalTrivial)}% overall, ` +
      `${pct(avgTrivInt - avgFfInt, avgTrivInt)}% of schedulable)`,
  );
  console.error(
    `    best-fit  totalBytes:  ${mb(totalBestFit / n).padStart(8)}MB ` +
      `(saved ${pct(totalTrivial - totalBestFit, totalTrivial)}% overall, ` +
      `${pct(avgTrivInt - avgBfInt, avgTrivInt)}% of schedulable)`,
  );
  console.error(
    `    peak live (lower bnd): ${mb(totalPeak / n).padStart(8)}MB ` +
      `(theoretical minimum)`,
  );
  console.error(
    `    external (unscheduled):${mb(avgExt).padStart(8)}MB ` +
      `(saved-for-backward, always live)`,
  );
  console.error(
    `    schedulable:           ${mb(avgTrivInt).padStart(8)}MB trivial → ` +
      `${mb(avgFfInt)}MB first-fit`,
  );
  console.error("");
  console.error("  Max plan:");
  console.error(`    trivial:     ${mb(maxTrivial).padStart(8)}MB`);
  console.error(
    `    first-fit:   ${mb(maxFirstFit).padStart(8)}MB ` +
      `(saved ${pct(maxTrivial - maxFirstFit, maxTrivial)}%)`,
  );
  console.error(`    peak live:   ${mb(maxPeakLive).padStart(8)}MB`);
  console.error(`    external:    ${mb(maxExternalBytes).padStart(8)}MB`);
  console.error("=".repeat(72));
  console.error("");
}

/** Max peak-live across all recorded plans (theoretical GPU floor). */
export function getMaxPeakLiveBytes(): number {
  let max = 0;
  for (const s of samples) {
    if (s.peakLiveBytes > max) max = s.peakLiveBytes;
  }
  return max;
}

/** Max first-fit totalBytes across all recorded plans. */
export function getMaxFirstFitBytes(): number {
  let max = 0;
  for (const s of samples) {
    if (s.firstFitTotalBytes > max) max = s.firstFitTotalBytes;
  }
  return max;
}
