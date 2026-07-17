/**
 * Stage-1 observed cross-plan liveness (docs/stage4-compile-from-ir.md §Stage-1;
 * task #43). Two gates:
 *
 *  GATE 2 — set-parity / seam-agreement (subprocess GPU). The build-from-IR
 *    path prunes each recurring template's harvest to the OBSERVED needed-set
 *    (consumed ∪ survived). The single-source-at-the-seam assertion is that the
 *    pruned path AGREES with the trusted reference (the default recorded
 *    cutover) on every observable value: bit-identical trajectory across the
 *    pruning-activation threshold (a dropped-needed result would crash loud or
 *    diverge), zero guard misses (nothing needed was ever pruned-then-demanded),
 *    and demonstrable pruning (results were actually removed). NOTE: the literal
 *    "observed set == cutover liveResultHarvestPairs" does NOT hold and is NOT
 *    the right invariant — the observed mechanism prunes MORE aggressively than
 *    the recorded cutover (it excludes harvested views of independently-live
 *    bases and measures survival AFTER markStep reclamation, not before). That
 *    the tighter set is EXACTLY sufficient is proven by the bit-identical
 *    trajectory + zero misses here and by the production ladder (fullstack,
 *    diloco, decode). See the probe + the campaign report.
 *
 *  GATE 3 — late-consumer guard (deterministic, CPU). A plan built AFTER the
 *    observation window that reads a PRUNED producer output misses at the
 *    consumer's bind-time external-slot resolution (before any side effect).
 *    Task #98 phase 5 DEMOTED the guardMiss recovery to a should-never-fire
 *    assertion: a matched pruned-producer miss (clean OR dirty) now throws a
 *    LOUD Error naming template/node/oi/stamp state — the old clean-recovery
 *    (RecoverableGuardMiss → evict + re-collect lowered) is DELETED because a
 *    zero-fire soak across the full config matrix proved it never fired (both
 *    prune-soundness classes are covered upstream: graphHeldAt + the
 *    recorded/witness harvest). An UNMATCHED miss (unregistered node / wrong
 *    output index) still returns false so the caller rethrows the original
 *    "Input not ready". This test drives the exact decision logic the bind-time
 *    seam (compiled-plan.ts phase 1) invokes.
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";
import {
  getCapturedRecordSkips,
  guardMiss,
  noteInPlaceCommit,
  observeConsumed,
  observeReadback,
  registerPrunedExecution,
  resetObservedLiveness,
  setObservedLivenessEnabled,
  setStepTapeReplayActive,
  stampResult,
} from "../src/executor/observed-liveness";
import type { StorageHandle } from "../src/graph/types";

const execFileP = promisify(execFile);
const PROBE = path.join(__dirname, "..", "tools", "t-observed-liveness-probe.ts");

async function runProbe(
  env: Record<string, string>,
): Promise<{
  losses: number[];
  resultSets: Record<string, string[]>;
  stats: {
    convergedTemplates: number;
    dirtyMisses: number;
    cleanMisses: number;
    prunedPairsRemoved: number;
  };
}> {
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout } = await execFileP("npx", ["tsx", PROBE], {
        env: { ...process.env, ...env, STEPS: "16" },
        timeout: 240_000,
        maxBuffer: 64 * 1024 * 1024,
      });
      const line = stdout
        .trim()
        .split("\n")
        .reverse()
        .find((l) => l.trim().startsWith('{"compiled"'));
      if (!line) throw new Error(`no probe JSON:\n${stdout.slice(-500)}`);
      return JSON.parse(line);
    } catch (e) {
      if (attempt >= 3) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

describe("observed cross-plan liveness — set-parity (gate 2)", () => {
  it(
    "build-from-IR-pruned agrees with the recorded cutover (trajectory + zero-miss + demonstrable pruning)",
    async () => {
      if (!(await canUseWebGPU())) return;
      // RE-BASED (task #43 recorded-build sunset): build-from-IR is the DEFAULT
      // ({}); the recorded-cutover reference is GONE, so the trusted reference
      // is now the LOWERED path (TORCHLETTE_COMPILED_PLAN=0) — the same reference
      // the surviving compiled==lowered gate uses. A dropped-needed result under
      // the pruned harvest still diverges/crashes against it.
      const bfir = await runProbe({});
      const reference = await runProbe({ TORCHLETTE_COMPILED_PLAN: "0" });

      // (1) Bit-identical trajectory across the pruning-activation threshold —
      //     a dropped-needed result would diverge or crash. This IS the seam
      //     agreement: the pruned harvest computes the lowered path's values.
      expect(bfir.losses.length).toBe(16);
      expect(reference.losses.length).toBe(16);
      let maxDiff = 0;
      for (let i = 0; i < bfir.losses.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(bfir.losses[i] - reference.losses[i]));
      }
      expect(maxDiff).toBeLessThan(1e-5);

      // (2) Zero guard misses — nothing needed was ever pruned-then-demanded.
      expect(bfir.stats.dirtyMisses).toBe(0);
      expect(bfir.stats.cleanMisses).toBe(0);

      // (3) Pruning demonstrably activated and removed results (the campaign's
      //     point — without this the "agreement" is trivial).
      expect(bfir.stats.convergedTemplates).toBeGreaterThanOrEqual(1);
      expect(bfir.stats.prunedPairsRemoved).toBeGreaterThan(0);
    },
    600_000,
  );
});

describe("observed cross-plan liveness — late-consumer guard (gate 3)", () => {
  const FP = 0xabcdef;
  const NI = 42;
  const OI = 0;
  const PRODUCER_NODE_ID = 100;

  beforeEach(() => {
    resetObservedLiveness();
    setObservedLivenessEnabled(true);
  });

  // Restore the module to its off default so enabling it here can't leak into
  // other test files sharing the worker.
  afterAll(() => {
    resetObservedLiveness();
    setObservedLivenessEnabled(false);
  });

  it("CLEAN miss → loud should-never-fire assertion naming template/node/oi (recovery DELETED)", () => {
    // Producer's pruned replay registered at in-place-commit count 0; nothing
    // mutated since. Pre-phase-5 this was the "recoverable" clean case
    // (RecoverableGuardMiss → re-collect lowered). Phase 5: it is a hard
    // assertion — a matched pruned-producer miss must never happen.
    registerPrunedExecution(FP, NI, OI, PRODUCER_NODE_ID);

    let caught: unknown;
    try {
      guardMiss(PRODUCER_NODE_ID, OI);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    const msg = (caught as Error).message;
    // Full-context diagnostic: template (hex), node, output index, clean state.
    expect(msg).toContain("guardMiss ASSERTION FIRED");
    expect(msg).toContain(`0x${FP.toString(16)}`);
    expect(msg).toContain(`node=${NI}`);
    expect(msg).toContain(`oi=${OI}`);
    expect(msg).toContain("clean");
  });

  it("DIRTY miss (in-place op committed since) → loud assertion naming template/node/oi + in-place count", () => {
    registerPrunedExecution(FP, NI, OI, PRODUCER_NODE_ID);
    // An in-place op (adam / copy_) committed between the producer's pruned
    // replay and the miss → recompute would read mutated storages → unsound.
    noteInPlaceCommit();

    let caught: unknown;
    try {
      guardMiss(PRODUCER_NODE_ID, OI);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    const msg = (caught as Error).message;
    // Diagnostic must name the template (hex), node, output index, and the
    // dirty (in-place-committed) classification.
    expect(msg).toContain("guardMiss ASSERTION FIRED");
    expect(msg).toContain(`0x${FP.toString(16)}`);
    expect(msg).toContain(`node=${NI}`);
    expect(msg).toContain(`oi=${OI}`);
    expect(msg.toLowerCase()).toContain("in-place");
  });

  it("miss on an UNREGISTERED node is not ours (returns false → caller rethrows original)", () => {
    // No registerPrunedExecution for this node id.
    expect(guardMiss(9999, 0)).toBe(false);
  });

  it("miss on the wrong output index of a registered node is not ours", () => {
    registerPrunedExecution(FP, NI, 0, PRODUCER_NODE_ID);
    expect(guardMiss(PRODUCER_NODE_ID, 1)).toBe(false);
  });
});

// [D5 — THE DECLARED-LIFETIME DIVIDEND; step-object phase 7 / ruling 2]
// CONTRACT CHANGE (named, like the lazy-execution precedent): the observation
// RECORDING is RETIRED on the captured path. Inside a declared step-tape replay
// (`setStepTapeReplayActive(true)`) liveness DERIVES from `crossPlanEdges` + the
// declared boundary survivors, so the recording hooks (`stampResult`,
// `observeConsumed`, `observeReadback`, `noteAliasedBase`) no-op — they change
// no decision there (the stage-1 measurement proved the convergence machinery
// they feed never activates under replay: `tools/t-d5-watcher-cost.ts`). The
// UNCAPTURED path is unchanged (the measurement found `everSurvived`
// load-bearing off-capture — the `[1,S,vocab]` CE-logits boundary survivor).
describe("observed cross-plan liveness — captured-path retirement (D5)", () => {
  const FP = 0x515151;
  const fakeSh = (id: number): StorageHandle => ({ id }) as StorageHandle;

  beforeEach(() => {
    resetObservedLiveness();
    setObservedLivenessEnabled(true);
    setStepTapeReplayActive(false);
  });
  afterAll(() => {
    resetObservedLiveness();
    setObservedLivenessEnabled(false);
    setStepTapeReplayActive(false);
  });

  it("UNCAPTURED: stampResult stamps + enrolls (observation records)", () => {
    const sh = fakeSh(1);
    stampResult(sh, FP, 7, 0);
    expect(sh.stamp).toEqual({ fp: FP, ni: 7, oi: 0 });
    expect(getCapturedRecordSkips()).toBe(0);
  });

  it("CAPTURED (replay active): stampResult no-ops — no stamp, skip counted", () => {
    setStepTapeReplayActive(true);
    const sh = fakeSh(2);
    stampResult(sh, FP, 7, 0);
    expect(sh.stamp).toBeUndefined();
    expect(getCapturedRecordSkips()).toBe(1);
  });

  it("CAPTURED: observeConsumed / observeReadback are no-ops (no throw, no record)", () => {
    // A value stamped OUTSIDE the replay (a build step) — then read DURING a
    // replay. The captured-path retirement must not record its consumption or
    // readback (derived from the edge set / declared ring outputs instead).
    const sh = fakeSh(3);
    stampResult(sh, FP, 9, 0); // build-step stamp (replay inactive)
    expect(sh.stamp).toBeDefined();
    setStepTapeReplayActive(true);
    expect(() => observeConsumed(sh, FP)).not.toThrow();
    expect(() => observeReadback(sh)).not.toThrow();
  });
});
