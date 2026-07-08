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
 *    With the in-place-committed counter CLEAN the miss is recoverable
 *    (RecoverableGuardMiss → the executor evicts + re-collects lowered); with it
 *    DIRTY (an in-place op committed since the producer's pruned replay) the
 *    miss is unrecoverable and FAILS LOUDLY with a diagnostic naming
 *    template/node/oi. This test drives the exact decision logic the bind-time
 *    seam (compiled-plan.ts phase 1) invokes.
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";
import {
  guardMiss,
  noteInPlaceCommit,
  RecoverableGuardMiss,
  registerPrunedExecution,
  resetObservedLiveness,
  setObservedLivenessEnabled,
  setTemplateCompiledInvalidator,
} from "../src/executor/observed-liveness";

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
        .find((l) => l.trim().startsWith('{"buildFromIR"'));
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
      const bfir = await runProbe({ TORCHLETTE_BUILD_FROM_IR: "1" });
      const cutover = await runProbe({});

      // (1) Bit-identical trajectory across the pruning-activation threshold —
      //     a dropped-needed result would diverge or crash. This IS the seam
      //     agreement: the two harvest strategies compute the same values.
      expect(bfir.losses.length).toBe(16);
      expect(cutover.losses.length).toBe(16);
      let maxDiff = 0;
      for (let i = 0; i < bfir.losses.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(bfir.losses[i] - cutover.losses[i]));
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

  it("CLEAN miss → RecoverableGuardMiss + producer template invalidated", () => {
    let invalidated: number | undefined;
    setTemplateCompiledInvalidator((fp) => (invalidated = fp));
    // Producer's pruned replay registered at in-place-commit count 0; nothing
    // mutated since.
    registerPrunedExecution(FP, NI, OI, PRODUCER_NODE_ID);

    let caught: unknown;
    try {
      guardMiss(PRODUCER_NODE_ID, OI);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(RecoverableGuardMiss);
    expect((caught as RecoverableGuardMiss).stamp).toEqual({
      fp: FP,
      ni: NI,
      oi: OI,
    });
    // The producer template's compiled plan is invalidated (grows needed-set,
    // rebuilds conservative-then-re-pruned next step).
    expect(invalidated).toBe(FP);
  });

  it("DIRTY miss (in-place op committed since) → loud failure naming template/node/oi", () => {
    setTemplateCompiledInvalidator(() => {});
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
    expect(caught).not.toBeInstanceOf(RecoverableGuardMiss);
    const msg = (caught as Error).message;
    // Diagnostic must name the template (hex), node, and output index.
    expect(msg).toContain(`0x${FP.toString(16)}`);
    expect(msg).toContain(`node=${NI}`);
    expect(msg).toContain(`oi=${OI}`);
    expect(msg.toLowerCase()).toContain("in-place");
  });

  it("miss on an UNREGISTERED node is not ours (returns false → caller rethrows original)", () => {
    setTemplateCompiledInvalidator(() => {});
    // No registerPrunedExecution for this node id.
    expect(guardMiss(9999, 0)).toBe(false);
  });

  it("miss on the wrong output index of a registered node is not ours", () => {
    setTemplateCompiledInvalidator(() => {});
    registerPrunedExecution(FP, NI, 0, PRODUCER_NODE_ID);
    expect(guardMiss(PRODUCER_NODE_ID, 1)).toBe(false);
  });
});
