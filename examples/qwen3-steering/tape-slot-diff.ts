/**
 * Step-tape phase-1a DELIVERABLE 2: empirical DynamicSlot enumeration
 * (docs/staged-execution-phase1.md §2.1/§2.4-guard-3, appended as §8).
 *
 * Byte-level diff of consecutive REAL decode steps: records (i) every
 * queue.writeBuffer upload (stable buffer identity, size, content hash —
 * the TAG_WRITE / TAG_UNIFORM / scalar-table / params writes all funnel
 * through writeBuffer), and (ii) every op payload + scalar-ref image per
 * plan position (via TORCHLETTE_TAPE_SLOTDIFF=1, src/core/tape-profile.ts).
 * Diffs step N vs N+1 to produce the complete varying-input list.
 *
 * Workloads:
 *   w1 — stock unsteered decode (static KV, greedy)
 *   w2 — steered decode, alpha FIXED (makeResidualHook from steering.ts)
 *   w3 — two generations, alpha=+3 then alpha=-3 (hazard #1: does the mul's
 *        scalar byte-image differ across recordings? what varies within one
 *        generation?)
 *
 * Run SOLO from repo root (env flag is read at module load, so it must be in
 * the environment, not set by this script):
 *   TORCHLETTE_TAPE_SLOTDIFF=1 npx tsx examples/qwen3-steering/tape-slot-diff.ts w1|w2|w3
 *
 * SUNSET: dies with the 1a instrumentation when phase 1c lands.
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
} from "../../src/backend/webgpu";
import {
  TAPE_SLOTDIFF,
  type TpPlanRecord,
  tpTakePlanRecords,
} from "../../src/core/tape-profile";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "../qwen3/loader";
import type { StaticKV } from "../qwen3/model";
import { makeResidualHook, type SteeringVector } from "./src/lib/steering";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"
const STEER_LAYER = 14; // mid layer of 28

// ---------------------------------------------------------------------------
// writeBuffer trace
// ---------------------------------------------------------------------------

type WriteRec = {
  seq: number;
  bufId: number;
  offset: number;
  bytes: number;
  hash: number;
  head: string; // first 4 f32 values (best-effort)
  src: string; // first /src/ stack frame (file:line)
};

let trace: WriteRec[] | null = null;
const bufIds = new WeakMap<object, number>();
let nextBufId = 1;

function fnvBytes(u8: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < u8.length; i++) {
    h ^= u8[i];
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function installWriteBufferTrace() {
  const dev = getWebGPUDevice();
  if (!dev) throw new Error("no device");
  const queue = dev.queue as unknown as {
    writeBuffer: (
      buf: object,
      offset: number,
      data: ArrayBuffer | ArrayBufferView,
      dataOffset?: number,
      size?: number,
    ) => void;
  };
  const orig = queue.writeBuffer.bind(queue);
  queue.writeBuffer = (buf, offset, data, dataOffset?, size?) => {
    if (trace) {
      let u8: Uint8Array;
      if (ArrayBuffer.isView(data)) {
        u8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      } else {
        u8 = new Uint8Array(data);
      }
      if (dataOffset) {
        // dataOffset is in ELEMENTS for views; none of our call sites use it,
        // so just flag it rather than mis-hash.
      }
      let id = bufIds.get(buf);
      if (id === undefined) {
        id = nextBufId++;
        bufIds.set(buf, id);
      }
      const f32 = new Float32Array(
        u8.buffer,
        u8.byteOffset,
        Math.min(4, Math.floor(u8.byteLength / 4)),
      );
      const srcLine =
        (new Error().stack ?? "")
          .split("\n")
          .find((l) => l.includes("/src/")) ?? "";
      trace.push({
        seq: trace.length,
        bufId: id,
        offset,
        bytes: u8.byteLength,
        hash: fnvBytes(u8),
        head: Array.from(f32)
          .map((v) => (Number.isInteger(v) ? String(v) : v.toPrecision(5)))
          .join(","),
        src: srcLine
          .trim()
          .replace(/^at /, "")
          .replace(/.*\/src\//, "src/")
          .replace(/\)$/, ""),
      });
    }
    return orig(buf, offset, data, dataOffset, size);
  };
}

// ---------------------------------------------------------------------------
// Decode driver
// ---------------------------------------------------------------------------

type StepCapture = { writes: WriteRec[]; plans: TpPlanRecord[] };

async function runGeneration(
  api: Torchlette,
  model: Awaited<ReturnType<typeof loadPretrainedQwen3>>,
  opts: {
    alpha: number;
    vec: SteeringVector | null;
    numSteps: number;
    captureSteps: Set<number>;
  },
): Promise<Map<number, StepCapture>> {
  const vocab = model.config.vocabSize;
  const hook = makeResidualHook(api, opts.vec, opts.alpha);
  const captures = new Map<number, StepCapture>();
  const tokens = [...PROMPT];
  const staticKV: StaticKV = model.allocStaticKV(512);
  const prevScope = api.setStepScopedCleanup(true);
  try {
    // Prefill (uncaptured)
    {
      const idx = api.tensorFromArray(tokens, [1, tokens.length]);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook: hook }),
      );
      const top = await api.readTopK(logits, 8, {
        offset: (tokens.length - 1) * vocab,
        length: vocab,
      });
      logits.dispose();
      tokens.push(top.indices[0]); // greedy
      await api.markStep();
      tpTakePlanRecords(); // discard prefill records
    }
    for (let i = 0; i < opts.numSteps; i++) {
      const capture = opts.captureSteps.has(i);
      if (capture) trace = [];
      tpTakePlanRecords(); // clear records from prior step
      const idx = api.tensorFromArray([tokens[tokens.length - 1]], [1, 1]);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook: hook }),
      );
      const top = await api.readTopK(logits, 8, { length: vocab });
      logits.dispose();
      tokens.push(top.indices[0]);
      await api.markStep();
      if (capture) {
        captures.set(i, { writes: trace!, plans: tpTakePlanRecords() });
        trace = null;
      }
    }
    staticKV.k.length = 0;
    staticKV.v.length = 0;
    await api.markStep();
  } finally {
    api.setStepScopedCleanup(prevScope);
  }
  console.log(`  tokens: ${tokens.join(",")}`);
  return captures;
}

// ---------------------------------------------------------------------------
// Diff + report
// ---------------------------------------------------------------------------

function diffWrites(a: WriteRec[], b: WriteRec[], labelA: string, labelB: string) {
  console.log(
    `\n--- writeBuffer trace diff: ${labelA} (${a.length} writes) vs ${labelB} (${b.length} writes) ---`,
  );
  if (a.length !== b.length) {
    console.log(
      `  !! STRUCTURAL: write COUNT differs (${a.length} vs ${b.length}) — step programs are not the same shape`,
    );
  }
  const n = Math.min(a.length, b.length);
  let same = 0;
  for (let i = 0; i < n; i++) {
    const wa = a[i];
    const wb = b[i];
    const bufSame = wa.bufId === wb.bufId;
    const contentSame = wa.hash === wb.hash && wa.bytes === wb.bytes;
    if (bufSame && contentSame) {
      same++;
      continue;
    }
    console.log(
      `  [${i}] ${contentSame ? "" : "CONTENT-DIFF"}${bufSame ? "" : " BUF-DIFF"} bytes=${wa.bytes}${wa.bytes !== wb.bytes ? `/${wb.bytes}` : ""} buf=${wa.bufId}${bufSame ? "" : `→${wb.bufId}`} src=${wa.src}`,
    );
    if (!contentSame) {
      console.log(`        ${labelA}: head=[${wa.head}]`);
      console.log(`        ${labelB}: head=[${wb.head}]`);
    }
  }
  console.log(`  (${same}/${n} writes byte-identical, stable buffers)`);
}

function diffPlans(a: TpPlanRecord[], b: TpPlanRecord[], labelA: string, labelB: string) {
  console.log(
    `\n--- plan payload/scalar diff: ${labelA} (${a.length} plans) vs ${labelB} (${b.length} plans) ---`,
  );
  if (a.length !== b.length) {
    console.log(`  !! plan COUNT differs (${a.length} vs ${b.length})`);
  }
  const n = Math.min(a.length, b.length);
  for (let p = 0; p < n; p++) {
    const pa = a[p];
    const pb = b[p];
    if (pa.fpPrimary !== pb.fpPrimary) {
      console.log(
        `  plan[${p}] TEMPLATE FINGERPRINT DIFFERS: 0x${pa.fpPrimary.toString(16)} (${pa.nodeCount} nodes) vs 0x${pb.fpPrimary.toString(16)} (${pb.nodeCount} nodes)`,
      );
    }
    const bn = new Map(pb.nodes.map((r) => [r.pos, r]));
    let diffs = 0;
    for (const ra of pa.nodes) {
      const rb = bn.get(ra.pos);
      if (!rb) continue;
      const payloadDiff = ra.payloadHash !== rb.payloadHash;
      const scalarDiff =
        ra.scalars.length !== rb.scalars.length ||
        ra.scalars.some((v, k) => !Object.is(v, rb.scalars[k]));
      if (!payloadDiff && !scalarDiff) continue;
      diffs++;
      console.log(
        `  plan[${p}] node[${ra.pos}] op=${ra.op}${payloadDiff ? " PAYLOAD-DIFF" : ""}${scalarDiff ? " SCALAR-DIFF" : ""}`,
      );
      if (payloadDiff) {
        console.log(`        ${labelA}: ${ra.payloadRepr}`);
        console.log(`        ${labelB}: ${rb.payloadRepr}`);
      }
      if (scalarDiff) {
        console.log(
          `        scalars ${labelA}=[${ra.scalars.join(",")}] ${labelB}=[${rb.scalars.join(",")}]`,
        );
      }
    }
    if (diffs === 0) {
      console.log(
        `  plan[${p}] fp=0x${pa.fpPrimary.toString(16)} nodes=${pa.nodeCount}: payloads+scalars IDENTICAL`,
      );
    }
  }
}

// ---------------------------------------------------------------------------

async function main() {
  const workload = process.argv[2] ?? "w1";
  if (!TAPE_SLOTDIFF) {
    throw new Error(
      "run with TORCHLETTE_TAPE_SLOTDIFF=1 (read at module load by src/core/tape-profile.ts)",
    );
  }
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, {
    maxSeqLen: 512,
    weightDtype: "f32",
  });
  installWriteBufferTrace();

  // Synthetic steering direction (structure-identical to a computed one; we
  // measure the SLOT MODEL, not steering quality). Deterministic values.
  const makeVec = (): SteeringVector => {
    const h = model.config.hiddenSize;
    const dir = new Float32Array(h);
    for (let i = 0; i < h; i++) dir[i] = Math.sin(i * 0.37) * 5;
    const direction = api.persist(api.tensorFromArray(dir, [h]));
    return {
      direction,
      layer: STEER_LAYER,
      hiddenSize: h,
      posPrompt: "synthetic+",
      negPrompt: "synthetic-",
    };
  };

  if (workload === "w1") {
    console.log("=== w1: stock unsteered decode — diff steps 8 vs 9 ===");
    const caps = await runGeneration(api, model, {
      alpha: 0,
      vec: null,
      numSteps: 11,
      captureSteps: new Set([8, 9]),
    });
    diffWrites(caps.get(8)!.writes, caps.get(9)!.writes, "step8", "step9");
    diffPlans(caps.get(8)!.plans, caps.get(9)!.plans, "step8", "step9");
  } else if (workload === "w2") {
    console.log("=== w2: steered decode, alpha FIXED at 4 — diff steps 8 vs 9 ===");
    const vec = makeVec();
    await api.markStep();
    const caps = await runGeneration(api, model, {
      alpha: 4,
      vec,
      numSteps: 11,
      captureSteps: new Set([8, 9]),
    });
    diffWrites(caps.get(8)!.writes, caps.get(9)!.writes, "step8", "step9");
    diffPlans(caps.get(8)!.plans, caps.get(9)!.plans, "step8", "step9");
  } else if (workload === "w3") {
    console.log(
      "=== w3: steered, alpha CHANGES across generations (+3 then -3) ===",
    );
    const vec = makeVec();
    await api.markStep();
    console.log("--- generation A (alpha=+3) ---");
    const capsA = await runGeneration(api, model, {
      alpha: 3,
      vec,
      numSteps: 11,
      captureSteps: new Set([8, 9]),
    });
    console.log("--- generation B (alpha=-3) ---");
    const capsB = await runGeneration(api, model, {
      alpha: -3,
      vec,
      numSteps: 11,
      captureSteps: new Set([0, 1, 8, 9]),
    });
    console.log("\n>>> WITHIN generation B (alpha fixed at -3): step 8 vs 9");
    diffWrites(capsB.get(8)!.writes, capsB.get(9)!.writes, "B.step8", "B.step9");
    diffPlans(capsB.get(8)!.plans, capsB.get(9)!.plans, "B.step8", "B.step9");
    console.log(
      "\n>>> ACROSS generations (alpha +3 vs -3), same step index (8):",
    );
    diffWrites(capsA.get(8)!.writes, capsB.get(8)!.writes, "A.step8", "B.step8");
    diffPlans(capsA.get(8)!.plans, capsB.get(8)!.plans, "A.step8", "B.step8");
    console.log(
      "\n>>> generation B FIRST steps (post-alpha-change re-adaptation): step0 write count=" +
        capsB.get(0)!.writes.length +
        ", step1 write count=" +
        capsB.get(1)!.writes.length +
        ", step8 write count=" +
        capsB.get(8)!.writes.length,
    );
    diffWrites(capsB.get(0)!.writes, capsB.get(1)!.writes, "B.step0", "B.step1");
  } else {
    throw new Error(`unknown workload ${workload} (use w1|w2|w3)`);
  }

  console.log("\nSLOT-DIFF DONE");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
