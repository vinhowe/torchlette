/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — P1 MOTHER GATE.
 *
 * The unrolled-K greedy block (decodeBlock, static-KV path) must produce ids
 * BYTE-IDENTICAL to the per-token host-loop reference. This is the differential
 * the whole campaign is governed by (design §5): greedy ids identical over
 * prompts / K∈{1,4,8,16} / bucket-boundary crossings / EOS-mid-block, plus
 * submits + ms/tok measured block-vs-loop.
 *
 * Runs a SMALL random-init Qwen3 (one model → identical weights across arms, as
 * in t-uk-feedback). Both arms re-prefill on their own fresh static-KV cache:
 *   - HOST loop : per-token model.forward + host argmax (readback each step).
 *   - BLOCK     : decodeBlock — on-device argmax->gather feedback, ONE readback
 *                 per (bucket-clipped) block.
 *
 * Run: eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX \
 *        LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-block-diff.ts
 */
import {
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
} from "../src/backend/webgpu";
import {
  getSubmitCount,
  resetSubmitCount,
} from "../src/backend/webgpu/webgpu-state";
import { Torchlette } from "../src/frontend/torchlette";
import type { Qwen3Config, StaticKV } from "../packages/qwen3-browser/src/model";
import { KV_BUCKET, Qwen3 } from "../packages/qwen3-browser/src/model";
import {
  clipBlockToBucket,
  decodeBlock,
} from "../packages/qwen3-browser/src/generate";

const CONFIG: Qwen3Config = {
  vocabSize: 256,
  hiddenSize: 64,
  numLayers: 2,
  numHeads: 4,
  numKVHeads: 2,
  headDim: 16,
  intermediateSize: 128,
  ropeTheta: 1e6,
  rmsNormEps: 1e-6,
  maxSeqLen: 256,
};

let FAIL = 0;
const ok = (cond: boolean, msg: string) => {
  console.log(`${cond ? "PASS" : "FAIL"} — ${msg}`);
  if (!cond) FAIL++;
};

/** Prefill on a fresh cache; return the first greedy token (one host argmax). */
async function prefillFirst(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  kv: StaticKV,
): Promise<number> {
  const V = CONFIG.vocabSize;
  const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
  const logits = api.noGrad(() => model.forward(idx, { staticKV: kv }).logits);
  const S = logits.shape[1];
  const row = api.noGrad(() =>
    api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V)),
  );
  const data = new Float32Array(await api.cpu(row));
  let best = 0;
  for (let v = 1; v < V; v++) if (data[v] > data[best]) best = v;
  return best;
}

/** HOST reference: N greedy tokens, one readback per step. */
async function hostLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const V = CONFIG.vocabSize;
  let idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
  let logits = api.noGrad(() => model.forward(idx, { staticKV: kv }).logits);
  const ids: number[] = [];
  for (let i = 0; i < N; i++) {
    const S = logits.shape[1];
    const row = api.noGrad(() =>
      api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V)),
    );
    const data = new Float32Array(await api.cpu(row));
    let best = 0;
    for (let v = 1; v < V; v++) if (data[v] > data[best]) best = v;
    ids.push(best);
    idx = api.tensorFromArray([best], [1, 1]);
    logits = api.noGrad(() => model.forward(idx, { staticKV: kv }).logits);
    await api.markStep();
  }
  return ids;
}

/** BLOCK arm: N greedy tokens via repeated bucket-clipped decodeBlock calls. */
async function blockLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  K: number,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const firstTok = await prefillFirst(api, model, promptIds, kv);
  await api.markStep();
  const ids: number[] = [];
  let lastTok = firstTok;
  while (ids.length < N) {
    const { ids: blk } = await decodeBlock(api, model, kv, lastTok, K);
    await api.markStep();
    for (const id of blk) {
      if (ids.length >= N) break;
      ids.push(id);
    }
    lastTok = blk[blk.length - 1];
  }
  return ids;
}

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1234);
  const model = new Qwen3(api, { ...CONFIG });

  console.log(
    "=== P1 MOTHER GATE: decodeBlock == host-loop (random-init Qwen3) ===",
  );
  console.log(`config: ${JSON.stringify(CONFIG)}\n`);

  // Clip-logic unit checks (no GPU).
  ok(
    clipBlockToBucket(120, 16, 256) === 8,
    "clip: len120,K16 -> 8 (bucket edge 128)",
  );
  ok(
    clipBlockToBucket(128, 16, 256) === 16,
    "clip: len128,K16 -> 16 (fresh bucket)",
  );
  ok(
    clipBlockToBucket(250, 16, 256) === 6,
    "clip: len250,K16 -> 6 (maxSeq edge 256)",
  );
  ok(clipBlockToBucket(10, 1, 256) === 1, "clip: K1 -> 1");
  ok(KV_BUCKET === 128, "KV_BUCKET === 128");

  // --- differential across prompts / K / bucket crossings ---
  const prompts: { name: string; ids: number[]; N: number }[] = [
    { name: "short prompt", ids: [3, 14, 15, 92, 65], N: 24 },
    { name: "another prompt", ids: [200, 1, 77, 42], N: 24 },
    {
      name: "bucket-crossing (len~120)",
      ids: Array.from({ length: 120 }, (_, i) => (i * 7 + 3) % CONFIG.vocabSize),
      N: 24,
    },
  ];

  for (const p of prompts) {
    const ref = await hostLoop(api, model, p.ids, p.N);
    await api.markStep();
    for (const K of [1, 4, 8, 16]) {
      const blk = await blockLoop(api, model, p.ids, p.N, K);
      await api.markStep();
      const match =
        ref.length === blk.length && ref.every((t, i) => t === blk[i]);
      ok(
        match,
        `[${p.name}] K=${K}: block ids == host ids (${p.N} tokens)` +
          (match
            ? ""
            : `\n    host : [${ref.join(",")}]\n    block: [${blk.join(",")}]`),
      );
    }
  }

  // --- EOS-in-block: the "compute all K, truncate at readback" contract (§3.3).
  // The random-init greedy stream collapses to a constant token, so a *natural*
  // mid-block EOS is unreachable here (pretrained models vary — Probe 1's real
  // distilgpt2 emitted EOS at index 1). These checks prove the contract against
  // the actual stream regardless of degeneracy: block computes all K; stopIndex
  // is the FIRST occurrence of the stop in the produced ids; host truncation
  // keeps exactly the pre-stop tokens; and no-stop => no truncation.
  {
    const p = prompts[0];
    const K = 8;
    // Ground-truth block ids (no stop).
    const kv0 = model.allocStaticKV(CONFIG.maxSeqLen);
    const firstTok = await prefillFirst(api, model, p.ids, kv0);
    await api.markStep();
    const { ids: full } = await decodeBlock(api, model, kv0, firstTok, K);
    await api.markStep();

    // (a) a stop token present in the stream: pick an id that occurs, assert the
    //     contract against its TRUE first-occurrence index.
    const eos = full[3];
    const firstOcc = full.indexOf(eos);
    const kv1 = model.allocStaticKV(CONFIG.maxSeqLen);
    const ft1 = await prefillFirst(api, model, p.ids, kv1);
    await api.markStep();
    const { ids: blk, stopIndex } = await decodeBlock(api, model, kv1, ft1, K, {
      stopTokens: new Set([eos]),
    });
    await api.markStep();
    ok(
      blk.length === K && blk.every((t, i) => t === full[i]),
      "EOS-block: full K computed regardless of the stop (compute-all-K)",
    );
    ok(
      stopIndex === firstOcc,
      `EOS-block: stopIndex ${stopIndex} == first-occurrence ${firstOcc}`,
    );
    const truncated = blk.slice(0, stopIndex);
    ok(
      truncated.length === firstOcc && truncated.every((t, i) => t === full[i]),
      "EOS-block: host truncation keeps exactly the pre-stop tokens",
    );

    // (b) no stop present: stopIndex == K (no truncation).
    const kv2 = model.allocStaticKV(CONFIG.maxSeqLen);
    const ft2 = await prefillFirst(api, model, p.ids, kv2);
    await api.markStep();
    const { ids: blk2, stopIndex: si2 } = await decodeBlock(
      api,
      model,
      kv2,
      ft2,
      K,
      { stopTokens: new Set([CONFIG.vocabSize + 1]) }, // impossible token
    );
    await api.markStep();
    ok(
      si2 === blk2.length && blk2.length === K,
      "EOS-block: no stop present => stopIndex == K (no truncation)",
    );
  }

  // --- P2 COMPILED ARM: build-from-IR ENABLED vs DISABLED, both == host ---
  // The K-block runs through the whole-step build-from-IR compiler by default.
  // The P2 generators (arg-reduce + max/min + RMSNorm) now cover their nodes; the
  // block does not YET reach full compiled replay (fusedRoPE is the lone residual
  // uncovered op — see the P2 census; its generated-stream parity is gated
  // separately by tools/t-uk-generators-parity.ts on a fully-covering graph). The
  // load-bearing invariant here: the build-from-IR-ENABLED path (default) must be
  // byte-identical to the pure-lowered path (COMPILED_PLAN=0) AND to the host
  // reference — the partial coverage must never corrupt the block.
  {
    const p = prompts[0];
    const N = 24;
    const K = 8;
    const ref = await hostLoop(api, model, p.ids, N);
    await api.markStep();

    const prev = process.env.TORCHLETTE_COMPILED_PLAN;
    // arm A: build-from-IR ENABLED (default).
    delete process.env.TORCHLETTE_COMPILED_PLAN;
    const blkOn = await blockLoop(api, model, p.ids, N, K);
    await api.markStep();
    // arm B: build-from-IR DISABLED (pure lowered).
    process.env.TORCHLETTE_COMPILED_PLAN = "0";
    const blkOff = await blockLoop(api, model, p.ids, N, K);
    await api.markStep();
    if (prev === undefined) delete process.env.TORCHLETTE_COMPILED_PLAN;
    else process.env.TORCHLETTE_COMPILED_PLAN = prev;

    const eqOnHost =
      ref.length === blkOn.length && ref.every((t, i) => t === blkOn[i]);
    const eqOffHost =
      ref.length === blkOff.length && ref.every((t, i) => t === blkOff[i]);
    const eqOnOff =
      blkOn.length === blkOff.length && blkOn.every((t, i) => t === blkOff[i]);
    ok(eqOnHost, "compiled-arm: build-from-IR ENABLED block == host ids");
    ok(eqOffHost, "compiled-arm: build-from-IR DISABLED block == host ids");
    ok(eqOnOff, "compiled-arm: ENABLED == DISABLED (partial coverage is faithful)");
  }

  // --- submits + ms/tok: block vs loop over N tokens ---
  {
    const p = prompts[0];
    const N = 16;
    const K = 8;
    resetSubmitCount();
    const th0 = performance.now();
    await hostLoop(api, model, p.ids, N);
    const hostMs = performance.now() - th0;
    const hostSubmits = getSubmitCount();
    await api.markStep();
    resetSubmitCount();
    const tb0 = performance.now();
    await blockLoop(api, model, p.ids, N, K);
    const blockMs = performance.now() - tb0;
    const blockSubmits = getSubmitCount();
    await api.markStep();
    console.log(
      `\n--- economics (N=${N}, K=${K}) ---\n` +
        `host  : ${hostSubmits} submits, ${(hostMs / N).toFixed(2)} ms/tok (${hostMs.toFixed(1)} ms total)\n` +
        `block : ${blockSubmits} submits, ${(blockMs / N).toFixed(2)} ms/tok (${blockMs.toFixed(1)} ms total)\n` +
        `submit ratio host/block = ${(hostSubmits / Math.max(blockSubmits, 1)).toFixed(2)}x`,
    );
    ok(
      blockSubmits < hostSubmits,
      `block issues fewer submits than host (${blockSubmits} < ${hostSubmits})`,
    );
  }

  // P4 precondition: making the block compile must introduce ZERO uncaptured
  // GPU errors. The first-compile external-buffer-destroy transient: the
  // per-step `idx` upload is harvested into a co-owned planner-registry buffer
  // and bound as the slot-0 external of the NEXT forward plan; when the
  // producer template is invalidated at a step boundary, destroyCompiledPlanBuffers
  // freed that registry buffer while the live idx storage still backed it —
  // poisoning the consumer replay's submit ("used in submit while destroyed").
  // Fixed at the seam (compiled-plan.ts liveHarvestIdsForBuffer): the entry
  // buffer is PARKED while ANY live storage backs it, not only registerState'd
  // ones. This assertion is the gate.
  const uncaptured = getGpuUncapturedErrorCount();
  ok(
    uncaptured === 0,
    `zero uncaptured GPU errors across the whole run (external-destroy transient) — got ${uncaptured}`,
  );

  console.log(
    `\n=== VERDICT: ${
      FAIL === 0
        ? "PASS — decodeBlock byte-identical to host-loop across all cells"
        : `FAIL (${FAIL} checks)`
    } ===`,
  );
  process.exit(FAIL === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
