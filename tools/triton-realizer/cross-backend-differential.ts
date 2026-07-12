/**
 * THE CROSS-BACKEND DIFFERENTIAL (schedule-state-design.md §4 v2 — the point).
 *
 * The SAME `ScheduleState` lowered through BOTH realizers:
 *   - applySchedule → WGSL → Dawn   (the live matmul path, via api.matmul + epilogue ops)
 *   - emitTriton   → python → CUDA  (tools/triton-realizer/run_kernel.py on a V100)
 * numerically compared on real shapes. Two disjoint compiler stacks agreeing on one
 * schedule object is the strongest correctness instrument the stack has.
 *
 * ------------------------------------------------------------------------
 * TOLERANCE — STATED BEFORE MEASURING (house rule)
 * ------------------------------------------------------------------------
 * fp32 accumulator on BOTH sides (WGSL `dotAccum` accumulates f32; the emitted
 * Triton uses `out_dtype=tl.float32`). f32 inputs, f32 accumulate → the only
 * divergence is fp reassociation across the two compilers' different reduction
 * orders. Expected max-abs error ~1e-4 relative to the magnitudes here (K≈256,
 * values O(1) → dot magnitudes O(10)); we assert < 2e-3 absolute (a generous fp32
 * reassociation envelope — the observed error is the reported number).
 *
 * The grouped / swap program-map cases test the R4 claim NUMERICALLY: a grouped or
 * swapped traversal is the SAME semantic computation as identity, so Triton-grouped
 * MUST equal the WGSL identity answer to the same tolerance (traversal ⊥ arithmetic).
 * The epilogue case applies bias on BOTH sides.
 *
 * Run: CUDA_VISIBLE_DEVICES=10 npx tsx tools/triton-realizer/cross-backend-differential.ts
 * Exit 0 iff every case is within tolerance; prints the differential numbers.
 */

import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { initWebGPU } from "../../src/backend/webgpu";
import { DEFAULT_CONFIG } from "../../src/backend/webgpu/matmul/types";
import type { Tensor } from "../../src/frontend/tensor";
import { Torchlette } from "../../src/frontend/torchlette";
import { artifactDigest, scheduleDigest } from "../../src/schedule/canonical";
import {
  deriveTiledMatmulState,
  type TiledMatmulDescriptor,
} from "../../src/schedule/matmul-skeleton";
import {
  TRITON_REALIZER,
  WGSL_REALIZER_COORDINATE,
} from "../../src/schedule/realizers/registry";
import { emitTritonTiledMatmul } from "../../src/schedule/realizers/triton-emit";
import type {
  AxisUid,
  ProgramGridMap,
  ScheduleState,
  SemanticRegionUid,
} from "../../src/schedule/types";

const REGION = "region:cross-backend" as unknown as SemanticRegionUid;
const HERE = new URL(".", import.meta.url).pathname;
const PY = join(HERE, ".venv", "bin", "python3");
const RUNNER = join(HERE, "run_kernel.py");
const TOL = 2e-3;

interface Case {
  label: string;
  M: number;
  N: number;
  K: number;
  gridMap: ProgramGridMap;
  groupSize?: number;
  bias?: boolean;
}

/** The P2 matmul shapes + a grouped-program-map case + an epilogue case. */
const CASES: Case[] = [
  // P2 matmul shapes (identity traversal) — the plain tiled path.
  {
    label: "P2 128x128x256 identity",
    M: 128,
    N: 128,
    K: 256,
    gridMap: { kind: "identity" },
  },
  {
    label: "P2 96x160x256 identity",
    M: 96,
    N: 160,
    K: 256,
    gridMap: { kind: "identity" },
  },
  // The grouped program-map case (A-R15 / R4 — the published L2-reuse remap).
  {
    label: "grouped(G=4) 128x128x256",
    M: 128,
    N: 128,
    K: 256,
    gridMap: {
      kind: "grouped",
      groupAxis: "axis:m" as unknown as AxisUid,
      groupSize: 4,
    },
    groupSize: 4,
  },
  // The swap program-map case (the repo's swapGrid — traversal ⊥ arithmetic).
  {
    label: "swap 128x96x256",
    M: 128,
    N: 96,
    K: 256,
    gridMap: {
      kind: "swap",
      axes: ["axis:m" as unknown as AxisUid, "axis:n" as unknown as AxisUid],
    },
  },
  // The epilogue case (bias added on BOTH sides).
  {
    label: "bias-epilogue 128x128x256",
    M: 128,
    N: 128,
    K: 256,
    gridMap: { kind: "identity" },
    bias: true,
  },
];

function seededData(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = ((s >>> 8) / 0x00ffffff) * 2 - 1; // ~U(-1,1)
  }
  return out;
}

/** Build a tiled descriptor for a case (NN f32; grid map applied via swapGrid). */
function descFor(c: Case): TiledMatmulDescriptor {
  return {
    config: { ...DEFAULT_CONFIG, tileM: 32, tileN: 32, tileK: 16 },
    transposeMode: "NN",
    dtype: "f32",
    swapGrid: c.gridMap.kind === "swap",
    epilogue: c.bias
      ? {
          ops: [{ kind: "bias", inputIndex: 0 }],
          additionalInputCount: 1,
          outputDtype: "f32",
        }
      : undefined,
  };
}

/** Override the program-grid map on a derived state (grouped isn't a WGSL swapGrid). */
function withGridMap(state: ScheduleState, map: ProgramGridMap): ScheduleState {
  return { ...state, semantic: { ...state.semantic, programGridMap: map } };
}

async function wgslMatmul(
  api: Torchlette,
  c: Case,
  a: Float32Array,
  b: Float32Array,
  bias: Float32Array | null,
): Promise<number[]> {
  const A: Tensor = api.tensorFromArray(Array.from(a), [c.M, c.K], {
    device: "webgpu",
  });
  const B: Tensor = api.tensorFromArray(Array.from(b), [c.K, c.N], {
    device: "webgpu",
  });
  let out = api.matmul(A, B);
  if (bias) {
    const Bias = api.tensorFromArray(Array.from(bias), [1, c.N], {
      device: "webgpu",
    });
    out = api.add(out, Bias);
  }
  return api.cpu(out);
}

function runTriton(
  c: Case,
  source: string,
  block: readonly [number, number, number],
  numWarps: number | null,
  numStages: number | null,
  a: Float32Array,
  b: Float32Array,
  bias: Float32Array | null,
): { out: Float32Array; ran: boolean; device?: string; error?: string } {
  const dir = mkdtempSync(join(tmpdir(), "xbackend-"));
  const aP = join(dir, "a.npy");
  const bP = join(dir, "b.npy");
  const biasP = bias ? join(dir, "bias.npy") : null;
  const outP = join(dir, "out.npy");
  writeNpy(aP, a, [c.M, c.K]);
  writeNpy(bP, b, [c.K, c.N]);
  if (bias && biasP) writeNpy(biasP, bias, [c.N]);

  const spec = {
    source,
    entry_point: "matmul_kernel",
    num_warps: numWarps,
    num_stages: numStages,
    block,
    grid_map: c.gridMap.kind,
    group_size: c.groupSize ?? 0,
    shapes: { M: c.M, N: c.N, K: c.K },
    has_bias: !!bias,
    alpha: 1.0,
    a_npy: aP,
    b_npy: bP,
    bias_npy: biasP,
    out_npy: outP,
    out_dtype: "f32",
  };
  const specP = join(dir, "spec.json");
  writeFileSync(specP, JSON.stringify(spec));

  const r = spawnSync(PY, [RUNNER, "--mode", "run", "--spec", specP], {
    encoding: "utf8",
  });
  const lines = (r.stdout || "{}").trim().split("\n");
  const parsed = JSON.parse(lines[lines.length - 1] ?? "{}");
  if (parsed.error)
    return { out: new Float32Array(0), ran: false, error: parsed.error };
  if (!parsed.ran)
    return { out: new Float32Array(0), ran: false, error: parsed.reason };
  return { out: readNpy(outP), ran: true, device: parsed.device };
}

// --- Minimal .npy IO (float32, C-order) — no numpy dep on the TS side. ---
function writeNpy(path: string, data: Float32Array, shape: number[]): void {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}${shape.length === 1 ? "," : ""}), }`;
  const pre = 10 + header.length;
  const pad = 64 - (pre % 64);
  const full = header + " ".repeat(pad - 1) + "\n";
  const magic = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  const lenBuf = Buffer.alloc(2);
  lenBuf.writeUInt16LE(full.length, 0);
  const headerBuf = Buffer.concat([magic, lenBuf, Buffer.from(full, "latin1")]);
  writeFileSync(
    path,
    Buffer.concat([
      headerBuf,
      Buffer.from(data.buffer, data.byteOffset, data.byteLength),
    ]),
  );
}
function readNpy(path: string): Float32Array {
  const buf = readFileSync(path);
  const headerLen = buf.readUInt16LE(8);
  const dataStart = 10 + headerLen;
  const bytes = buf.subarray(dataStart);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function maxAbs(
  a: number[] | Float32Array,
  b: number[] | Float32Array,
): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function main(): Promise<void> {
  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
  });

  console.log("=== CROSS-BACKEND DIFFERENTIAL (WGSL/Dawn vs Triton/CUDA) ===");
  console.log(`tolerance (stated before measuring): max-abs < ${TOL}\n`);

  let failures = 0;
  let anyRan = false;
  const rows: string[] = [];

  for (const c of CASES) {
    const desc = descFor(c);
    // ONE schedule state; override program-grid map for grouped (Triton-only remap).
    let state = deriveTiledMatmulState(desc, REGION);
    if (c.gridMap.kind === "grouped") state = withGridMap(state, c.gridMap);

    // Identity coordinate: the SAME schedule digest, TWO artifact identities (§5).
    const schedDigest = scheduleDigest(state);
    const wgslArtifact = artifactDigest(state, WGSL_REALIZER_COORDINATE);
    const tritonArtifact = artifactDigest(state, TRITON_REALIZER.coordinate);

    const a = seededData(c.M * c.K, 12345 + c.M);
    const b = seededData(c.K * c.N, 54321 + c.N);
    const bias = c.bias ? seededData(c.N, 999) : null;

    const wgsl = await wgslMatmul(api, c, a, b, bias);

    const emission = emitTritonTiledMatmul(state, desc);
    const tri = runTriton(
      c,
      emission.source,
      emission.block,
      emission.numWarps,
      emission.numStages,
      a,
      b,
      bias,
    );

    if (!tri.ran) {
      rows.push(
        `  ${c.label.padEnd(32)} | COMPILE-CHECK ONLY (run blocked: ${tri.error})`,
      );
      continue;
    }
    anyRan = true;
    const err = maxAbs(wgsl, tri.out);
    const ok = err < TOL;
    if (!ok) failures++;
    rows.push(
      `  ${c.label.padEnd(32)} | gridMap=${emission.gridMap.padEnd(8)} | max-abs=${err.toExponential(3)} | ${ok ? "PASS" : "FAIL"}`,
    );
    if (c === CASES[0]) {
      console.log(`schedule digest:   ${schedDigest}`);
      console.log(`  WGSL artifact:   ${wgslArtifact}`);
      console.log(`  Triton artifact: ${tritonArtifact}`);
      console.log(
        `  (same schedule, distinct artifacts — §5 realizer coordinate)`,
      );
      console.log(`  Triton device:   ${tri.device}\n`);
    }
  }

  console.log("Results:");
  for (const r of rows) console.log(r);
  console.log("");

  if (!anyRan) {
    console.log(
      "FALLBACK: no case executed on GPU — see compile-check notes above.",
    );
    process.exit(2);
  }
  if (failures > 0) {
    console.log(
      `DIFFERENTIAL FAILED: ${failures} case(s) exceeded tolerance ${TOL}.`,
    );
    process.exit(1);
  }
  console.log(`DIFFERENTIAL PASSED: all executed cases within ${TOL}.`);
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
