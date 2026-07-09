/**
 * GATE 4 (task #80 inc-2a, TEST-FIRST): the MULTI-GROUP packed-fused
 * differential.
 *
 * The default fused Adam path is the PACKED optimizer: it concatenates
 * grad/param/m/v across same-element-count params and dispatches ONE kernel
 * per size class, binding ONE lr tensor for the whole packed group. inc-2a
 * makes lr flow as a per-group persistent tensor DATA input, so the packed
 * grouping key MUST break the batch on lr-tensor identity — else two param
 * groups with different LR but the same param element count pack together and
 * ONE group's LR silently trains the OTHER group's params. That is the
 * frozen-scalar-family failure mode and is invisible to every single-group
 * test (fused-vs-elementwise.spec.ts is all single-group).
 *
 * This gate builds TWO AdamParamGroups with DIFFERENT lr AND wd, EQUAL param
 * element counts (so a numElements-only key would wrongly merge them), and
 * compares the packed fused trajectory to the sequential per-group reference
 * (elementwise, compiled+fusion off) over 20+ steps. If the grouping key is
 * wrong, the packed run applies group 0's lr to group 1's params and the
 * trajectories diverge — the assertion FAILS.
 *
 * Each trajectory runs in its OWN SUBPROCESS (tools/adam-trajectory-probe.ts,
 * TWO_GROUPS=1) — the one-engine-per-process methodology the sibling
 * differentials use.
 */

import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { beforeAll, describe, expect, it } from "vitest";
import { canUseWebGPU } from "../helpers/webgpu";

const execFileP = promisify(execFile);
const PROBE = path.join(
  __dirname,
  "..",
  "..",
  "tools",
  "adam-trajectory-probe.ts",
);
const STEPS = 24;

async function trajectory(env: Record<string, string>): Promise<number[][]> {
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout } = await execFileP("npx", ["tsx", PROBE], {
        env: { ...process.env, STEPS: String(STEPS), ...env },
        timeout: 180_000,
        maxBuffer: 16 * 1024 * 1024,
      });
      const line = stdout.trim().split("\n").pop()!;
      return JSON.parse(line);
    } catch (e) {
      if (attempt >= 5) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
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

/**
 * Pure-JS 2-group AdamW ground truth. Params 0..half-1 use (lr0, wd0), params
 * half.. use (lr1, wd1). Mirrors the probe's init + loss (per-param grad
 * 2*(p-target)). The probe splits N elements into NPARAMS equal params and
 * groups the first ceil(NPARAMS/2) into group 0.
 */
function twoGroupGroundTruth(opts: {
  nParams: number;
  lr0: number;
  wd0: number;
  lr1: number;
  wd1: number;
  adamW: boolean;
}): number[][] {
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
  const target = initData(99);
  const m = new Array(N).fill(0);
  const v = new Array(N).fill(0);

  // Reproduce the probe's equal split of N into nParams params.
  const per = Math.floor(N / opts.nParams);
  const sizes: number[] = [];
  let rest = N;
  for (let i = 0; i < opts.nParams; i++) {
    const take = i === opts.nParams - 1 ? rest : per;
    sizes.push(take);
    rest -= take;
  }
  const half = Math.ceil(opts.nParams / 2); // params [0,half) → group 0
  // element index → group
  const group = new Array(N).fill(1);
  {
    let off = 0;
    for (let pi = 0; pi < opts.nParams; pi++) {
      for (let k = 0; k < sizes[pi]; k++) group[off + k] = pi < half ? 0 : 1;
      off += sizes[pi];
    }
  }

  const traj: number[][] = [];
  for (let s = 1; s <= STEPS; s++) {
    const bc1 = 1 - 0.9 ** s;
    const bc2 = 1 - 0.999 ** s;
    for (let i = 0; i < N; i++) {
      const lr = group[i] === 0 ? opts.lr0 : opts.lr1;
      const wd = group[i] === 0 ? opts.wd0 : opts.wd1;
      let g = 2 * (p[i] - target[i]);
      if (!opts.adamW && wd !== 0) g += wd * p[i];
      m[i] = 0.9 * m[i] + 0.1 * g;
      v[i] = 0.999 * v[i] + 0.001 * g * g;
      let update = (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + 1e-8);
      if (opts.adamW && wd !== 0) update += lr * wd * p[i];
      p[i] -= update;
    }
    traj.push([...p]);
  }
  return traj;
}

describe(
  "Adam multi-group packed-fused differential (inc-2a Gate 4)",
  { timeout: 600_000 },
  () => {
    let webgpu = false;
    beforeAll(async () => {
      webgpu = await canUseWebGPU();
    });

    // NPARAMS=4, equal element counts, split into two groups: params 0,1 in
    // group 0 (lr=1e-2, wd=0.1); params 2,3 in group 1 (lr=1e-3, wd=0). The
    // two groups have the SAME param element count so packing WILL group them
    // if the key is numElements-only — the grouping key MUST additionally
    // break on lr-tensor identity.
    const cfg = {
      ADAMW: "1",
      NPARAMS: "4",
      TWO_GROUPS: "1",
      LR: "1e-2",
      WD: "0.1",
      LR_G1: "1e-3",
      WD_G1: "0",
    };

    it("packed fused == sequential per-group reference (24 steps)", async () => {
      if (!webgpu) return;
      // Default = fused + packed + compiled + fusion (the path with the hazard).
      const packed = await trajectory(cfg);
      // Reference: elementwise per-param, no packing, no compiled, no fusion.
      const reference = await trajectory({
        ...cfg,
        ELEMENTWISE: "1",
        FOREACH: "0",
        COMPILED: "0",
        FUSION: "0",
      });
      expect(maxAbsDiff(packed, reference)).toBeLessThan(1e-5);
    });

    it("packed fused == pure-JS two-group ground truth (24 steps)", async () => {
      if (!webgpu) return;
      const packed = await trajectory(cfg);
      const truth = twoGroupGroundTruth({
        nParams: 4,
        lr0: 1e-2,
        wd0: 0.1,
        lr1: 1e-3,
        wd1: 0,
        adamW: true,
      });
      expect(maxAbsDiff(packed, truth)).toBeLessThan(1e-5);
      // Sanity: the two groups actually move at different rates (so a
      // single-lr-for-all bug would be caught, not masked by tiny motion).
      // Group 1 (lr 10x smaller) moves ~10x less than group 0 per step.
      const g0d = Math.abs(truth[1][0] - truth[0][0]);
      const g1d = Math.abs(truth[1][63] - truth[0][63]);
      expect(g1d).toBeLessThan(g0d * 0.5);
    });
  },
);
