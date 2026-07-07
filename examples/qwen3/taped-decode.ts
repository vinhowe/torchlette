/**
 * Step-tape phase-1c driver: taped decode over the library seam, plus the
 * gate harnesses that need a taped generate loop (docs/staged-execution-phase1
 * §3 G3/G4/G5/G7/G8). Run SOLO with TORCHLETTE_STEP_TAPE=1.
 *
 * The decode loop is the library-controlled region §1 names. Per step:
 *   1. compute the app-level bucket key (model + KV bucket + steering
 *      structure + α — the guard declaration at the seam),
 *   2. if a skeleton is ready AND this is not a verify step, build the 4 fresh
 *      upload payloads and replay; on a guard miss fall through,
 *   3. otherwise run the normal forward (records + captures a candidate).
 * On a replay HIT the model's `cache.len` bookkeeping is skipped (model.forward
 * does it), so the driver advances it explicitly — the KV *buffers* were
 * updated inside the replayed plan (scatterAdd→copy_), only the JS counter is
 * the driver's to move.
 *
 *   npx tsx examples/qwen3/taped-decode.ts perf   [numSteps=24]
 *   npx tsx examples/qwen3/taped-decode.ts g3                      # frozen-α differential
 *   npx tsx examples/qwen3/taped-decode.ts g4                      # lifetime (with strict flags)
 *   npx tsx examples/qwen3/taped-decode.ts soak                    # long mixed session
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette, type Tensor } from "../../src/frontend/torchlette";
import { STEP_TAPE_REPLAY, STEP_TAPE_VERIFY_N } from "../../src/core/step-tape";
import type { StaticKV } from "./model";
import { kvBucketLen } from "./model";
import { loadPretrainedQwen3 } from "./loader";
import { buildDecodeUploads, staticKvId } from "./decode-uploads";
import { makeResidualHook, type SteeringVector } from "../qwen3-steering/src/lib/steering";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"
const STEER_LAYER = 14;

type Model = Awaited<ReturnType<typeof loadPretrainedQwen3>>;

interface DecodeOpts {
  vec: SteeringVector | null;
  alpha: number;
  verifyEvery: number; // 0 = never force the normal path for verify
}

/** One decode step through the tape seam. Returns logits + whether it replayed. */
async function tapedDecodeStep(
  api: Torchlette,
  model: Model,
  staticKV: StaticKV,
  tokenId: number,
  stepIdx: number,
  opts: DecodeOpts,
): Promise<{ logits: Tensor; taped: boolean }> {
  const posOffset = staticKV.len;
  const bucketLen = kvBucketLen(posOffset + 1, staticKV.maxSeqLen);
  const steered = opts.vec !== null;
  // α is a VALUE the driver varies → it belongs in the guard declaration
  // (§2.4 mitigation a). Distinct α ⇒ distinct bucket ⇒ its own tape; an α
  // change routes to the normal path (scalar-adapt re-lowers) then re-records.
  const kv = staticKvId(staticKV);
  const appKey = steered
    ? `steer:${STEER_LAYER}:a${opts.alpha}:kv${kv}:bkt${bucketLen}`
    : `stock:kv${kv}:bkt${bucketLen}`;
  api.setTapeContext(appKey, []);

  const hook = makeResidualHook(api, opts.vec, opts.alpha);
  // Verify steps run the NORMAL path so the executor cross-checks the skeleton
  // it WOULD have replayed (fp + command-stream diff) — driven by the env flag
  // TORCHLETTE_TAPE_VERIFY=N (STEP_TAPE_VERIFY_N).
  const doVerify = STEP_TAPE_VERIFY_N > 0 && stepIdx % STEP_TAPE_VERIFY_N === 0;

  if (api.tapeReadyFor(appKey) && !doVerify) {
    const uploads = buildDecodeUploads(
      model.config,
      posOffset,
      tokenId,
      bucketLen,
    );
    const logits = await api.tapeReplay(uploads);
    if (logits) {
      staticKV.len = posOffset + 1; // model.forward would have done this
      return { logits, taped: true };
    }
  }
  const idx = api.tensorFromArray([tokenId], [1, 1]);
  const { logits } = api.noGrad(() =>
    model.forward(idx, { staticKV, residualHook: hook }),
  );
  return { logits, taped: false };
}

/** Greedy generate `numTokens` via the tape seam. Returns tokens + timing +
 *  taped-step count. */
async function tapedGenerate(
  api: Torchlette,
  model: Model,
  opts: DecodeOpts & {
    numTokens: number;
    sample?: (i: number, taped: boolean) => void;
  },
): Promise<{ tokens: number[]; walls: number[]; tapedSteps: number }> {
  const vocab = model.config.vocabSize;
  const tokens = [...PROMPT];
  const staticKV = model.allocStaticKV(512);
  const prev = api.setStepScopedCleanup(true);
  const walls: number[] = [];
  let tapedSteps = 0;
  try {
    // Prefill (never taped — seqLen>1, distinct template).
    {
      const idx = api.tensorFromArray(tokens, [1, tokens.length]);
      const hook = makeResidualHook(api, opts.vec, opts.alpha);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook: hook }),
      );
      const top = await api.readTopK(logits, 8, {
        offset: (tokens.length - 1) * vocab,
        length: vocab,
      });
      logits.dispose();
      tokens.push(top.indices[0]);
      await api.markStep();
    }
    for (let i = 0; i < opts.numTokens; i++) {
      const t0 = performance.now();
      const { logits, taped } = await tapedDecodeStep(
        api,
        model,
        staticKV,
        tokens[tokens.length - 1],
        i,
        opts,
      );
      if (taped) tapedSteps++;
      const top = await api.readTopK(logits, 8, { length: vocab });
      logits.dispose();
      tokens.push(top.indices[0]);
      await api.markStep();
      walls.push(performance.now() - t0);
      opts.sample?.(i, taped);
    }
    staticKV.k.length = 0;
    staticKV.v.length = 0;
    await api.markStep();
  } finally {
    api.setStepScopedCleanup(prev);
  }
  return { tokens, walls, tapedSteps };
}

/** Untaped baseline generate (normal forward every step) for the differential
 *  / perf comparison. */
async function untapedGenerate(
  api: Torchlette,
  model: Model,
  opts: DecodeOpts & { numTokens: number },
): Promise<{ tokens: number[]; walls: number[] }> {
  const vocab = model.config.vocabSize;
  const tokens = [...PROMPT];
  const staticKV = model.allocStaticKV(512);
  const prev = api.setStepScopedCleanup(true);
  const walls: number[] = [];
  try {
    {
      const idx = api.tensorFromArray(tokens, [1, tokens.length]);
      const hook = makeResidualHook(api, opts.vec, opts.alpha);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook: hook }),
      );
      const top = await api.readTopK(logits, 8, {
        offset: (tokens.length - 1) * vocab,
        length: vocab,
      });
      logits.dispose();
      tokens.push(top.indices[0]);
      await api.markStep();
    }
    for (let i = 0; i < opts.numTokens; i++) {
      const t0 = performance.now();
      const idx = api.tensorFromArray([tokens[tokens.length - 1]], [1, 1]);
      const hook = makeResidualHook(api, opts.vec, opts.alpha);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook: hook }),
      );
      const top = await api.readTopK(logits, 8, { length: vocab });
      logits.dispose();
      tokens.push(top.indices[0]);
      await api.markStep();
      walls.push(performance.now() - t0);
    }
  } finally {
    api.setStepScopedCleanup(prev);
  }
  return { tokens, walls };
}

function steady(walls: number[]): number {
  const s = walls.slice(Math.min(6, walls.length - 1));
  return s.reduce((a, b) => a + b, 0) / s.length;
}

function makeVec(api: Torchlette, h: number): SteeringVector {
  const dir = new Float32Array(h);
  for (let i = 0; i < h; i++) dir[i] = Math.sin(i * 0.37) * 5;
  return {
    direction: api.persist(api.tensorFromArray(dir, [h])),
    layer: STEER_LAYER,
    hiddenSize: h,
    posPrompt: "synthetic+",
    negPrompt: "synthetic-",
  };
}

async function main() {
  const mode = process.argv[2] ?? "perf";
  if (!STEP_TAPE_REPLAY) {
    throw new Error("run with TORCHLETTE_STEP_TAPE=1");
  }
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, {
    maxSeqLen: 512,
    weightDtype: "f32",
  });
  let failed = false;

  if (mode === "perf") {
    const n = Number(process.argv[3] ?? 24);
    const un = await untapedGenerate(api, model, {
      vec: null,
      alpha: 0,
      verifyEvery: 0,
      numTokens: n,
    });
    const tp = await tapedGenerate(api, model, {
      vec: null,
      alpha: 0,
      verifyEvery: 0,
      numTokens: n,
    });
    const s = api.getStepTapeStats();
    console.log(`\n[perf] untaped steady = ${steady(un.walls).toFixed(2)} ms/token`);
    console.log(`[perf] taped   steady = ${steady(tp.walls).toFixed(2)} ms/token (${tp.tapedSteps}/${n} steps replayed)`);
    console.log(`[perf] replay stats:`, JSON.stringify(s.replay));
    console.log(`[perf] tokens match: ${JSON.stringify(un.tokens) === JSON.stringify(tp.tokens)}`);
    if (JSON.stringify(un.tokens) !== JSON.stringify(tp.tokens)) failed = true;
  } else if (mode === "g3") {
    // Frozen-α differential: taped α=3 then α=-3 must match a never-taped α=-3.
    const vec = makeVec(api, model.config.hiddenSize);
    await api.markStep();
    const a3 = await tapedGenerate(api, model, { vec, alpha: 3, verifyEvery: 0, numTokens: 20 });
    const bTaped = await tapedGenerate(api, model, { vec, alpha: -3, verifyEvery: 0, numTokens: 20 });
    // Fresh reference: never-taped α=-3 (untaped forward every step).
    const ref = await untapedGenerate(api, model, { vec, alpha: -3, verifyEvery: 0, numTokens: 20 });
    console.log(`\n[g3] taped α=3  :`, JSON.stringify(a3.tokens.slice(PROMPT.length)));
    console.log(`[g3] taped α=-3 :`, JSON.stringify(bTaped.tokens.slice(PROMPT.length)), `(replayed ${bTaped.tapedSteps}/20)`);
    console.log(`[g3] ref  α=-3  :`, JSON.stringify(ref.tokens.slice(PROMPT.length)));
    const match = JSON.stringify(bTaped.tokens) === JSON.stringify(ref.tokens);
    console.log(`[g3] taped-α=-3 == never-taped-α=-3: ${match}`);
    console.log(`[g3] stats:`, JSON.stringify(api.getStepTapeStats().replay));
    if (!match) failed = true;
    if (JSON.stringify(a3.tokens) === JSON.stringify(bTaped.tokens)) {
      console.log("[g3] !! FAIL: α=3 and α=-3 produced identical tokens (frozen-α)");
      failed = true;
    }
  } else if (mode === "g2") {
    // Shadow equivalence across a bucket crossing: a 200-token stock generation
    // crosses KV bucket 128→256 at posOffset 128. Run with TORCHLETTE_TAPE_VERIFY
    // =1 for full shadow. Expect: 0 verify diffs; ≥2 ready tapes (one per
    // bucket); the bucket transition shows as counted missNoTape (new appKey,
    // re-record) NOT a diff. Also compare taped vs a never-taped run.
    const tp = await tapedGenerate(api, model, {
      vec: null,
      alpha: 0,
      verifyEvery: 0,
      numTokens: 200,
    });
    const un = await untapedGenerate(api, model, {
      vec: null,
      alpha: 0,
      verifyEvery: 0,
      numTokens: 200,
    });
    const s = api.getStepTapeStats();
    console.log(`\n[g2] taped replayed ${tp.tapedSteps}/200; readyTapes=${s.replay.readyTapes}`);
    console.log(`[g2] tokens match never-taped: ${JSON.stringify(tp.tokens) === JSON.stringify(un.tokens)}`);
    console.log(`[g2] replay:`, JSON.stringify(s.replay));
    console.log(`[g2] recorder refusals=${s.recorder.refusals} tapeCount=${s.recorder.tapeCount}`);
    if (JSON.stringify(tp.tokens) !== JSON.stringify(un.tokens)) failed = true;
    if (s.replay.verifyDiffs > 0) failed = true;
    if (s.recorder.refusals > 0) failed = true;
    if (s.replay.readyTapes < 2) {
      console.log("[g2] !! FAIL: expected ≥2 per-bucket tapes on a bucket crossing");
      failed = true;
    }
  } else if (mode === "g4") {
    // Lifetime: reachable-storage flat across taped steps. Run with
    // TORCHLETTE_STRICT_LIFETIME=1 TORCHLETTE_STRICT_GPU=1 externally.
    const { storageTracker } = await import("../../src/graph/storage-tracker");
    const gen = await tapedGenerate(api, model, {
      vec: null,
      alpha: 0,
      verifyEvery: 0,
      numTokens: 100,
      sample: (i, taped) => {
        if (i % 20 === 19) {
          const s = storageTracker.stats();
          console.log(
            `[g4] step ${i + 1}${taped ? " (taped)" : ""}: reachable=${s.reachableStorages} total=${s.totalStorages}`,
          );
        }
      },
    });
    console.log(`\n[g4] replayed ${gen.tapedSteps}/100 steps`);
    console.log(`[g4] storage stats (final) =`, JSON.stringify(storageTracker.stats()));
    console.log(`[g4] stats:`, JSON.stringify(api.getStepTapeStats().replay));
  } else if (mode === "soak") {
    const vec = makeVec(api, model.config.hiddenSize);
    await api.markStep();
    const stockToks: string[] = [];
    for (const [i, spec] of [
      { vec: null, alpha: 0 },
      { vec, alpha: 4 },
      { vec, alpha: 4 },
      { vec, alpha: -2 },
      { vec: null, alpha: 0 },
      { vec, alpha: 6 },
    ].entries()) {
      const g = await tapedGenerate(api, model, {
        ...spec,
        verifyEvery: 16,
        numTokens: 40,
      });
      if (!spec.vec) stockToks.push(JSON.stringify(g.tokens));
      console.log(`[soak] gen ${i} (${spec.vec ? "steer α=" + spec.alpha : "stock"}): replayed ${g.tapedSteps}/40`);
    }
    // The two stock generations use FRESH KV each and the same prompt → they
    // must produce identical tokens. A mismatch = a skeleton bound to gen-0's
    // KV buffers was wrongly replayed in gen-4 (the cross-instance bug the
    // kv-nonce in the appKey defends against).
    if (stockToks.length === 2 && stockToks[0] !== stockToks[1]) {
      console.log("[soak] !! FAIL: the two stock generations diverged (cross-instance KV reuse)");
      failed = true;
    }
    const s = api.getStepTapeStats();
    console.log(`[soak] recorder:`, JSON.stringify(s.recorder));
    console.log(`[soak] replay:`, JSON.stringify(s.replay));
    if (s.replay.verifyDiffs > 0) {
      console.log("[soak] !! FAIL: verify diffs");
      failed = true;
    }
    if (s.recorder.refusals > 0) {
      console.log("[soak] !! FAIL: refusals");
      failed = true;
    }
  }

  console.log(failed ? "\nTAPED-DECODE FAIL" : "\nTAPED-DECODE PASS");
  process.exit(failed ? 1 : 0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
