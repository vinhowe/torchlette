/**
 * WITNESS-TIME HARVEST gate (task #98 phase 4, docs/step-object-design.md §4).
 *
 * The in-suite insurance for the #97 STOP (the recorded-build deletion has been
 * blocked TWICE by a config no prior gate exercised — this is the third
 * attempt's structural net). The witness recorder observes the LOWERED cross-plan
 * read of a checkpoint-recompute forward activation at end-of-step time and keeps
 * it in the generated harvest, so the prune that produced `Input not ready: node
 * contiguous[512,768]` cannot fire.
 *
 * This spec drives the CHECKPOINT cell (the exact #97 config: distil@512 dims +
 * selective checkpointing) and the EVENT-INCLUSIVE cells (GradScaler inf-skip, LR
 * milestone, §10 ruling 2) as subprocesses and asserts the phase-4 gate:
 *   - zero `Input not ready` (the #97 stage-3 negative assertion);
 *   - the witnessed harvest set is non-empty (the mechanism fired) — the
 *     shadow-coverage signal;
 *   - cleanMisses == dirtyMisses == 0 (no pruned-then-demanded read — the witness
 *     set kept every read pair, the operational §4.4 empty-diff);
 *   - [D3 refusal] the eager checkpoint cell is now the refusal's PROTECTED
 *     (lowered, arena-on) config → prunedPairsRemoved == 0 by design; compiled-path
 *     pruning + trajectory moved to the safe whole-step remat gate (t-d3-remat);
 *   - the interposed event (inf-skip / LR drop) produced NO corruption.
 *
 * The heavier matrix cells (medium@512, 124M chunked-sum) are run by the manual
 * driver `tools/t-witness-harvest-matrix.ts CELL=medium|chunked124m` — same tool,
 * excluded here only for suite runtime. GPU-less CI auto-skips (canUseWebGPU).
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";

const execFileP = promisify(execFile);
const MATRIX = path.join(
  __dirname,
  "..",
  "tools",
  "t-witness-harvest-matrix.ts",
);

interface Verdict {
  witnessHarvest: true;
  cell: string;
  pass: boolean;
  witnessedTemplates: number;
  witnessedPairs: number;
  witnessVariances: number;
  inputNotReady: number;
  threw: boolean;
  cleanMisses: number;
  dirtyMisses: number;
  prunedPairsRemoved: number;
  scalerInfObserved: boolean | null;
  lrDropped: boolean | null;
  steps: number;
}

async function runCell(cell: string): Promise<Verdict> {
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout } = await execFileP("npx", ["tsx", MATRIX], {
        env: { ...process.env, TORCHLETTE_STEP_TAPE: "record", CELL: cell },
        timeout: 300_000,
        maxBuffer: 64 * 1024 * 1024,
      });
      const line = stdout
        .trim()
        .split("\n")
        .reverse()
        .find((l) => l.trim().startsWith('{"witnessHarvest"'));
      if (!line) throw new Error(`no verdict JSON:\n${stdout.slice(-500)}`);
      return JSON.parse(line);
    } catch (e) {
      if (attempt >= 3) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

describe("witness-time harvest — the #97 checkpoint config (phase 4)", () => {
  it("distil@512 + selective checkpointing: witnessed harvest keeps the recompute activation (zero Input-not-ready, zero misses)", async () => {
    if (!(await canUseWebGPU())) return;
    const v = await runCell("checkpoint");
    // A. the #97 stage-3 negative assertion.
    expect(v.inputNotReady).toBe(0);
    expect(v.threw).toBe(false);
    // B. the mechanism fired (witnessed set non-empty — shadow coverage).
    expect(v.witnessedTemplates).toBeGreaterThanOrEqual(1);
    expect(v.witnessedPairs).toBeGreaterThan(0);
    // C. operational §4.4 empty-diff: no pruned-then-demanded read.
    expect(v.cleanMisses).toBe(0);
    expect(v.dirtyMisses).toBe(0);
    // D. [D3 refusal — SUNSET-BOUND] The eager checkpoint config is now the
    // refusal's PROTECTED config: the b66ead78 checkpoint+arena hazard keeps it
    // LOWERED (all-or-nothing per step), so there is NO compiled-plan harvest to
    // prune → prunedPairsRemoved == 0 BY DESIGN. This is the safe behavior the
    // setBufferArenaDisabled bypass also enforces (production checkpointing runs
    // arena-off = uncompiled). The #97 negative assertion (A) + the non-empty
    // LOWERED-witness set (B: 384 stamped pairs via stampLoweredActionOutputs) +
    // the zero-miss empty-diff (C) still guard the recompute-activation read on
    // the lowered path. COMPILED-path pruning + trajectory for checkpointing is
    // validated on the SAFE whole-step remat path by `tools/t-d3-remat.ts`
    // (hasCompiled=true, D3-READY). When the bypass/refusal sunset at P4, restore
    // `toBeGreaterThan(0)` here (eager checkpoint will compile safely then).
    expect(v.prunedPairsRemoved).toBe(0);
    // Steady reader: the witness set stabilized (no unbounded variance).
    expect(v.witnessVariances).toBe(0);
    expect(v.pass).toBe(true);
  }, 600_000);

  it("EVENT-INCLUSIVE — GradScaler inf-skip: the skip event produces no corruption, witness set survives", async () => {
    if (!(await canUseWebGPU())) return;
    const v = await runCell("scaler-inf");
    expect(v.inputNotReady).toBe(0);
    expect(v.threw).toBe(false);
    // A real inf-skip fired (the scaler backed off) — the event is exercised,
    // not merely present.
    expect(v.scalerInfObserved).toBe(true);
    // The witness set still covered the checkpoint reads across the event.
    expect(v.witnessedTemplates).toBeGreaterThanOrEqual(1);
    expect(v.pass).toBe(true);
  }, 600_000);

  it("EVENT-INCLUSIVE — LR-scheduler milestone: the LR drop flows through the declared slot, no corruption", async () => {
    if (!(await canUseWebGPU())) return;
    const v = await runCell("lr-milestone");
    expect(v.inputNotReady).toBe(0);
    expect(v.threw).toBe(false);
    // The milestone actually dropped the LR (event exercised) and the run
    // stayed clean through it (the LR is DATA in a declared scalar slot).
    expect(v.lrDropped).toBe(true);
    expect(v.witnessedTemplates).toBeGreaterThanOrEqual(1);
    expect(v.pass).toBe(true);
  }, 600_000);
});
