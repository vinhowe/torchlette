/**
 * Differential test: fused Adam kernel vs pure-graph elementwise Adam.
 *
 * The framework has TWO implementations of the same optimizer: the fused
 * WGSL kernel (`_stepFused`, WebGPU) and the pure tensor-op path
 * (`_stepElementwise`, the CPU fallback and the "describe AdamW like
 * PyTorch" reference — see docs/architecture-debt.md). Nothing compared
 * them until 2026-06-10, and they had silently forked: the elementwise
 * path applied L2 weight decay through the gradient even with adamW=true.
 *
 * Each trajectory runs in its OWN SUBPROCESS (tools/adam-trajectory-probe.ts):
 * the framework's module-global state (plan-template cache, params-sequence
 * cache) makes multiple Torchlette instances in one process interfere, so
 * in-process comparison is unreliable — the same one-engine-per-process
 * methodology as the parity harnesses.
 *
 * The three semantics tests compare against the SEQUENTIAL elementwise path
 * (compiled plan + fusion off) so they pin pure optimizer math. The last
 * three tests are the stage-1 (scalars-as-data) regression guards: per-step
 * scalars flow through the plan's scalar table (scalar-table.ts, refreshed
 * every execution), fused recipes adapt by demoting changed scalars to
 * runtime inputs, and the replay gate drops compiled plans whose inlined
 * scalar constants went stale. Before stage 1, the pure-graph optimizer was
 * silently wrong under EITHER optimization system (frozen 1-beta^t
 * coefficients — docs/architecture-debt.md).
 */

import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { beforeAll, describe, expect, it } from "vitest";
import { canUseWebGPU } from "../helpers/webgpu";

const execFileP = promisify(execFile);
const PROBE = path.join(__dirname, "..", "..", "tools", "adam-trajectory-probe.ts");
const STEPS = 6;

async function trajectory(env: Record<string, string>): Promise<number[][]> {
  // Retries with backoff: spawning a Dawn/WebGPU child from a GPU-holding
  // vitest worker fails transiently under suite load (adapter contention —
  // many GPU workers + subprocesses share the device).
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout } = await execFileP("npx", ["tsx", PROBE], {
        env: { ...process.env, STEPS: String(STEPS), ...env },
        timeout: 120_000,
        maxBuffer: 16 * 1024 * 1024,
      });
      // The probe prints exactly one JSON line to stdout.
      const line = stdout.trim().split("\n").pop()!;
      return JSON.parse(line);
    } catch (e) {
      if (attempt >= 3) throw e;
      await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
    }
  }
}

function maxAbsDiff(a: number[][], b: number[][]): number {
  let m = 0;
  for (let s = 0; s < a.length; s++) {
    for (let i = 0; i < a[s].length; i++) {
      m = Math.max(m, Math.abs(a[s][i] - b[s][i]));
    }
  }
  return m;
}

/** Pure-JS PyTorch Adam/AdamW ground truth (mirrors the probe's setup). */
function jsGroundTruth(opts: { adamW: boolean; wd: number }): number[][] {
  const N = 64;
  const initData = (seed: number): number[] => {
    const out: number[] = [];
    let x = seed;
    for (let i = 0; i < N; i++) {
      x = (x * 1103515245 + 12345) % 2147483648;
      out.push(((x / 2147483648) * 2 - 1) * 0.5);
    }
    return out;
  };
  const p = initData(7);
  const t = initData(99);
  const m = new Array(N).fill(0);
  const v = new Array(N).fill(0);
  const lr = 1e-2;
  const traj: number[][] = [];
  for (let s = 1; s <= STEPS; s++) {
    const bc1 = 1 - 0.9 ** s;
    const bc2 = 1 - 0.999 ** s;
    for (let i = 0; i < N; i++) {
      let g = 2 * (p[i] - t[i]);
      if (!opts.adamW && opts.wd !== 0) g += opts.wd * p[i]; // L2 through grad
      m[i] = 0.9 * m[i] + 0.1 * g;
      v[i] = 0.999 * v[i] + 0.001 * g * g;
      let update = (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + 1e-8);
      if (opts.adamW && opts.wd !== 0) update += lr * opts.wd * p[i]; // decoupled
      p[i] -= update;
    }
    traj.push([...p]);
  }
  return traj;
}

describe("fused vs elementwise Adam differential", { timeout: 600_000 }, () => {
  let webgpu = false;

  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it("AdamW (decoupled weight decay): fused == elementwise == ground truth", async () => {
    if (!webgpu) return;
    // Sequential spawns — concurrent Dawn/WebGPU init in sibling processes
    // is flaky on this driver.
    const fused = await trajectory({ ADAMW: "1", WD: "0.1" });
    const elementwise = await trajectory({ ADAMW: "1", WD: "0.1", ELEMENTWISE: "1", COMPILED: "0", FUSION: "0" });
    const truth = jsGroundTruth({ adamW: true, wd: 0.1 });
    expect(maxAbsDiff(fused, elementwise)).toBeLessThan(1e-5);
    expect(maxAbsDiff(fused, truth)).toBeLessThan(1e-5);
  });

  it("classic Adam (L2 through gradient): fused == elementwise == ground truth", async () => {
    if (!webgpu) return;
    const fused = await trajectory({ ADAMW: "0", WD: "0.1" });
    const elementwise = await trajectory({ ADAMW: "0", WD: "0.1", ELEMENTWISE: "1", COMPILED: "0", FUSION: "0" });
    const truth = jsGroundTruth({ adamW: false, wd: 0.1 });
    expect(maxAbsDiff(fused, elementwise)).toBeLessThan(1e-5);
    expect(maxAbsDiff(fused, truth)).toBeLessThan(1e-5);
  });

  it("no weight decay: fused == elementwise == ground truth", async () => {
    if (!webgpu) return;
    const fused = await trajectory({ ADAMW: "1", WD: "0" });
    const elementwise = await trajectory({ ADAMW: "1", WD: "0", ELEMENTWISE: "1", COMPILED: "0", FUSION: "0" });
    const truth = jsGroundTruth({ adamW: true, wd: 0 });
    expect(maxAbsDiff(fused, elementwise)).toBeLessThan(1e-5);
    expect(maxAbsDiff(fused, truth)).toBeLessThan(1e-5);
  });

  // Frozen-scalar regression guards (stage 1, scalars-as-data): per-step
  // scalars flow through the plan's scalar table (refreshed every execution),
  // so value-independent caches — compiled plans, fusion recipes — stay
  // faithful. These two tests FAILED (0.07-0.16 nat by step 6) before
  // scalar-table.ts + the fused-recipe adaptation landed.
  it("pure-graph optimizer is faithful under the compiled plan", async () => {
    if (!webgpu) return;
    const compiled = await trajectory({ ADAMW: "1", WD: "0", ELEMENTWISE: "1", FUSION: "0" });
    const lowered = await trajectory({ ADAMW: "1", WD: "0", ELEMENTWISE: "1", COMPILED: "0", FUSION: "0" });
    expect(maxAbsDiff(compiled, lowered)).toBeLessThan(1e-5);
  });

  it("pure-graph optimizer is faithful under fusion (cached-recipe adaptation)", async () => {
    if (!webgpu) return;
    const fusionOn = await trajectory({ ADAMW: "1", WD: "0", ELEMENTWISE: "1", COMPILED: "0" });
    const fusionOff = await trajectory({ ADAMW: "1", WD: "0", ELEMENTWISE: "1", COMPILED: "0", FUSION: "0" });
    expect(maxAbsDiff(fusionOn, fusionOff)).toBeLessThan(1e-5);
  });

  // Packing equivalence: Adam is elementwise, so splitting the same 64
  // elements across 3 differently-sized params must give the concatenated
  // trajectory of the 1-param run. The 3-param run exercises the foreach
  // path's REAL packing (cat in, narrow+copy_ back, packed m/v state) under
  // full default optimizations; the 1-param run is the sequential per-param
  // reference already pinned to pure-JS ground truth above. (The per-param
  // path itself is NOT used as the multi-param reference: it has a
  // pre-existing first-param state-corruption bug on WebGPU — see
  // docs/architecture-debt.md ledger.)
  it("foreach packing matches the single-param trajectory (3 params, default opts)", async () => {
    if (!webgpu) return;
    const packed = await trajectory({
      ADAMW: "1",
      WD: "0.1",
      NPARAMS: "3",
      ELEMENTWISE: "1",
    });
    const reference = await trajectory({
      ADAMW: "1",
      WD: "0.1",
      ELEMENTWISE: "1",
      FOREACH: "0",
      COMPILED: "0",
      FUSION: "0",
    });
    expect(maxAbsDiff(packed, reference)).toBeLessThan(1e-5);
  });

  // Per-param path across param counts: this was the silent-UAF bug — the
  // per-param path replaces this.expAvg[i] with a mid-step-created tensor
  // each step; markStep demoted it as a step temporary (not in the beginStep
  // snapshot), pooling its buffer while live, and with >1 param the loss
  // plan's outputs landed in the state buffer (sparse corruption, first
  // param only). Fixed by updating state IN PLACE (copy_) into the
  // persistent constructor tensors; the [lifetime] read guard in
  // getInputStorage now catches the class loudly.
  it("per-param path matches across param counts (UAF regression)", async () => {
    if (!webgpu) return;
    const multi = await trajectory({
      ADAMW: "1",
      WD: "0",
      NPARAMS: "2",
      ELEMENTWISE: "1",
      FOREACH: "0",
      COMPILED: "0",
      FUSION: "0",
    });
    const single = await trajectory({
      ADAMW: "1",
      WD: "0",
      ELEMENTWISE: "1",
      FOREACH: "0",
      COMPILED: "0",
      FUSION: "0",
    });
    expect(maxAbsDiff(multi, single)).toBeLessThan(1e-6);
  });

  // Late-varying scalar: the LR is CONSTANT through the recording executions
  // (so it gets inlined into fused-recipe WGSL and recorded into the compiled
  // plan), then changes at step 4. The replay gate's inlined-scalar staleness
  // check must drop the stale compiled plan; the lowered re-execution adapts
  // the recipe and re-records. This is the LR-schedule-with-warmup-plateau
  // shape that a record-time-only check would silently freeze.
  it("late LR change is honored under full default optimizations", async () => {
    if (!webgpu) return;
    const optimized = await trajectory({
      ADAMW: "1",
      WD: "0",
      ELEMENTWISE: "1",
      LR2: "0.001",
      LR2_AT: "4",
      STEPS: "8",
    });
    const sequential = await trajectory({
      ADAMW: "1",
      WD: "0",
      ELEMENTWISE: "1",
      LR2: "0.001",
      LR2_AT: "4",
      STEPS: "8",
      COMPILED: "0",
      FUSION: "0",
    });
    expect(maxAbsDiff(optimized, sequential)).toBeLessThan(1e-5);
    // Sanity: the schedule actually took effect (step deltas shrink ~10x).
    const d3 = Math.abs(sequential[3][0] - sequential[2][0]);
    const d5 = Math.abs(sequential[5][0] - sequential[4][0]);
    expect(d5).toBeLessThan(d3 * 0.5);
  });
});
