/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — P4 STEERING DIFFERENTIAL.
 *
 * The live consumers of the decode path are the SAE / activation-steering demos
 * (examples/qwen3-steering, examples/gemma2-sae-demo). Their core behavior is a
 * residual-stream steering hook threaded into model.forward on EVERY decode step
 * (steering.ts makeResidualHook: at layer L, x += alpha * direction). The P4
 * cutover routes decode through the unrolled-K block, so the hook must now apply
 * per-step INSIDE the block's graph. This gate proves the composition is exact:
 *
 *   1. GREEDY: the hooked block ids are BYTE-IDENTICAL to the hooked per-token
 *      host loop (the mother gate, with steering active) — over prompts / K /
 *      a bucket crossing. The hook composes with the on-device argmax->gather
 *      feedback.
 *   2. STEERING IS LIVE (alpha != 0): the hooked stream DIFFERS from the
 *      unsteered (alpha = 0 / no-hook) stream — the hook genuinely steers, so
 *      test 1 is a real composition proof, not a no-op agreeing with a no-op.
 *   3. ALPHA SCALES: a larger alpha produces a (generally) different stream than
 *      a smaller one — the steering magnitude flows through the block unchanged.
 *   4. COMPILED vs LOWERED: the hooked block composes with build-from-IR replay
 *      (the shipping default) AND the pure-lowered path, both == the host loop.
 *   5. Zero uncaptured GPU errors across the greedy composition (STRICT_GPU).
 *   6. SAMPLED (Gumbel) composition — CHARACTERIZATION ONLY (logged, not a hard
 *      gate): the sampled block path carries a PRE-EXISTING dropped-submit
 *      transient, so the Gumbel sampled cutover is opt-in until it is fixed.
 *
 * The hook here mirrors examples/qwen3-steering/src/lib/steering.ts
 * makeResidualHook (additive x += alpha*dir at one layer, persisted via
 * registerState) so it is the demo's actual steering mechanism, not a proxy.
 *
 * Run: eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX \
 *        LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-steering-diff.ts
 */
import {
  decodeBlock,
  gumbelUniform,
} from "../packages/qwen3-browser/src/generate";
import type {
  Qwen3Config,
  ResidualHook,
  StaticKV,
} from "../packages/qwen3-browser/src/model";
import { Qwen3 } from "../packages/qwen3-browser/src/model";
import { getGpuUncapturedErrorCount, getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

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
const eq = (a: number[], b: number[]) =>
  a.length === b.length && a.every((t, i) => t === b[i]);

/**
 * The demo's steering mechanism (mirrors examples/qwen3-steering steering.ts
 * makeResidualHook): at layer L, x += alpha * direction. Persisted via
 * registerState so it survives the generation's step-scoped cleanup. alpha=0 =>
 * no hook (unsteered identity).
 */
function makeHook(
  api: Torchlette,
  dir3d: ReturnType<Torchlette["registerState"]>,
  layer: number,
  alpha: number,
): ResidualHook | undefined {
  if (alpha === 0) return undefined;
  return (x, l) => (l === layer ? api.add(x, api.mul(dir3d, alpha)) : x);
}

async function prefillFirst(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  kv: StaticKV,
  hook?: ResidualHook,
): Promise<number> {
  const V = CONFIG.vocabSize;
  const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
  const logits = api.noGrad(
    () => model.forward(idx, { staticKV: kv, residualHook: hook }).logits,
  );
  const S = logits.shape[1];
  const row = api.noGrad(() =>
    api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V)),
  );
  const data = new Float32Array(await api.cpu(row));
  let best = 0;
  for (let v = 1; v < V; v++) if (data[v] > data[best]) best = v;
  return best;
}

/** HOST greedy reference with the steering hook threaded into every forward. */
async function hostLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  hook?: ResidualHook,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const V = CONFIG.vocabSize;
  let idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
  let logits = api.noGrad(
    () => model.forward(idx, { staticKV: kv, residualHook: hook }).logits,
  );
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
    logits = api.noGrad(
      () => model.forward(idx, { staticKV: kv, residualHook: hook }).logits,
    );
    await api.markStep();
  }
  return ids;
}

/** BLOCK greedy arm with the steering hook (composes inside the block graph). */
async function blockLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  K: number,
  hook?: ResidualHook,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const firstTok = await prefillFirst(api, model, promptIds, kv, hook);
  await api.markStep();
  const ids: number[] = [];
  let lastTok = firstTok;
  while (ids.length < N) {
    const { ids: blk } = await decodeBlock(api, model, kv, lastTok, K, {
      residualHook: hook,
    });
    await api.markStep();
    for (const id of blk) {
      if (ids.length >= N) break;
      ids.push(id);
    }
    lastTok = blk[blk.length - 1];
  }
  return ids;
}

/** HOST Gumbel reference with the hook — the block's per-position seeds. */
async function sampledHostLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  temperature: number,
  seed: number,
  hook?: ResidualHook,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const V = CONFIG.vocabSize;
  const firstTok = await prefillFirst(api, model, promptIds, kv, hook);
  await api.markStep();
  const startLen = kv.len;
  const ids: number[] = [];
  let idx = api.tensorFromArray([firstTok], [1, 1]);
  for (let j = 0; j < N; j++) {
    const logits = api.noGrad(
      () => model.forward(idx, { staticKV: kv, residualHook: hook }).logits,
    );
    const idT = api.noGrad(() => {
      const u = api.tensorFromArray(gumbelUniform(seed + startLen + j, V), [1, 1, V]);
      const g = api.neg(api.log(api.neg(api.log(u))));
      const scaled = api.add(api.div(logits, temperature), g);
      return api.argmax(scaled, { dim: -1, keepdim: false });
    });
    const tok = Math.round((await api.cpu(idT))[0]);
    ids.push(tok);
    idx = api.tensorFromArray([tok], [1, 1]);
    await api.markStep();
  }
  return ids;
}

/** BLOCK Gumbel arm with the hook. */
async function sampledBlockLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  K: number,
  temperature: number,
  seed: number,
  hook?: ResidualHook,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const firstTok = await prefillFirst(api, model, promptIds, kv, hook);
  await api.markStep();
  const ids: number[] = [];
  let lastTok = firstTok;
  while (ids.length < N) {
    const { ids: blk } = await decodeBlock(api, model, kv, lastTok, K, {
      residualHook: hook,
      sample: { temperature, seed },
    });
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

  console.log("=== P4 STEERING DIFFERENTIAL: hooked block == hooked loop ===");
  console.log(`config: ${JSON.stringify(CONFIG)}\n`);

  // A random steering direction [1,1,hidden], persisted (mirrors makeResidualHook).
  const H = CONFIG.hiddenSize;
  const dirArr = new Float32Array(H);
  let s = 987654321 >>> 0;
  for (let i = 0; i < H; i++) {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    dirArr[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1; // U(-1,1)
  }
  const dir3d = api.registerState(api.tensorFromArray([...dirArr], [1, 1, H]));
  const LAYER = 1;
  const ALPHA = 8.0; // large enough to visibly flip the greedy stream

  const prompts: { name: string; ids: number[]; N: number }[] = [
    { name: "short", ids: [3, 14, 15, 92, 65], N: 24 },
    {
      name: "bucket-crossing (len~120)",
      ids: Array.from({ length: 120 }, (_, i) => (i * 7 + 3) % CONFIG.vocabSize),
      N: 24,
    },
  ];

  // (1) GREEDY composition: hooked block == hooked host loop, over prompts/K.
  const hook = makeHook(api, dir3d, LAYER, ALPHA);
  for (const p of prompts) {
    const ref = await hostLoop(api, model, p.ids, p.N, hook);
    await api.markStep();
    for (const K of [1, 4, 8, 16]) {
      const blk = await blockLoop(api, model, p.ids, p.N, K, hook);
      await api.markStep();
      const match = eq(ref, blk);
      ok(
        match,
        `[${p.name}] K=${K}: HOOKED block == HOOKED host (a=${ALPHA}, L=${LAYER})` +
          (match ? "" : `\n    host : [${ref.join(",")}]\n    block: [${blk.join(",")}]`),
      );
    }
  }

  // (2) STEERING IS LIVE: hooked stream != unsteered stream (alpha=0).
  {
    const p = prompts[0];
    const steered = await blockLoop(api, model, p.ids, p.N, 8, hook);
    await api.markStep();
    const unsteered = await blockLoop(api, model, p.ids, p.N, 8, undefined);
    await api.markStep();
    ok(
      !eq(steered, unsteered),
      `steering is live: a=${ALPHA} stream != unsteered stream (hook is not a no-op)`,
    );
    // and the unsteered block still matches the unsteered host loop.
    const unsteeredHost = await hostLoop(api, model, p.ids, p.N, undefined);
    await api.markStep();
    ok(eq(unsteered, unsteeredHost), "unsteered block == unsteered host (baseline intact)");
  }

  // (3) ALPHA SCALES: a different alpha produces a different stream.
  {
    const p = prompts[0];
    const small = makeHook(api, dir3d, LAYER, 2.0);
    const big = makeHook(api, dir3d, LAYER, 16.0);
    const sBlk = await blockLoop(api, model, p.ids, p.N, 8, small);
    await api.markStep();
    const bBlk = await blockLoop(api, model, p.ids, p.N, 8, big);
    await api.markStep();
    // and each still matches its host loop (composition holds at every alpha).
    const sHost = await hostLoop(api, model, p.ids, p.N, small);
    await api.markStep();
    const bHost = await hostLoop(api, model, p.ids, p.N, big);
    await api.markStep();
    ok(eq(sBlk, sHost), "alpha=2: hooked block == hooked host");
    ok(eq(bBlk, bHost), "alpha=16: hooked block == hooked host");
    ok(!eq(sBlk, bBlk), "alpha scales: a=2 stream != a=16 stream");
  }

  // (4) COMPILED vs LOWERED with the hook active — the block's build-from-IR
  // replay must compose with the steering hook (the shipping default path).
  {
    const p = prompts[0];
    const ref = await hostLoop(api, model, p.ids, p.N, hook);
    await api.markStep();
    const prev = process.env.TORCHLETTE_COMPILED_PLAN;
    delete process.env.TORCHLETTE_COMPILED_PLAN; // build-from-IR ENABLED (default)
    const on = await blockLoop(api, model, p.ids, p.N, 8, hook);
    await api.markStep();
    process.env.TORCHLETTE_COMPILED_PLAN = "0"; // lowered
    const off = await blockLoop(api, model, p.ids, p.N, 8, hook);
    await api.markStep();
    if (prev === undefined) delete process.env.TORCHLETTE_COMPILED_PLAN;
    else process.env.TORCHLETTE_COMPILED_PLAN = prev;
    ok(eq(on, ref), "hooked block build-from-IR ENABLED == hooked host");
    ok(eq(off, ref), "hooked block LOWERED == hooked host");
    ok(eq(on, off), "hooked block ENABLED == LOWERED (steering faithful under compile)");
  }

  // (5) zero uncaptured GPU errors across the GREEDY run (STRICT_GPU class) —
  //     asserted BEFORE the sampled characterization below, which exercises the
  //     pre-existing sampled-path transient.
  const uncapturedGreedy = getGpuUncapturedErrorCount();
  ok(
    uncapturedGreedy === 0,
    `zero uncaptured GPU errors across the greedy composition — got ${uncapturedGreedy}`,
  );

  // (6) SAMPLED (Gumbel) composition — CHARACTERIZATION, not a hard gate. The
  // sampled block path carries a PRE-EXISTING dropped-submit transient (a
  // materialized first-token upload whose registry buffer is destroyed without
  // parking — `_lastHarvestIds` tracks harvest RESULTS, not external-input
  // uploads; visible as "used in submit while destroyed" here and in
  // t-uk-gumbel-parity). It is why the Gumbel sampled cutover is opt-IN, not
  // default-on (generate.ts unrolledKExplicit). We RUN it to record the current
  // state (hook logic is correct — gumbel-parity's unhooked block==host is the
  // logic proof), but the transient can corrupt these ids, so this section is
  // logged, NOT counted in the verdict. When the transient is fixed, promote
  // these to hard asserts and flip the Gumbel cutover default-on.
  {
    const temperature = 0.8;
    const SEED = 4242;
    const p = prompts[0];
    const before = getGpuUncapturedErrorCount();
    const ref = await sampledHostLoop(api, model, p.ids, p.N, temperature, SEED, hook);
    await api.markStep();
    for (const K of [1, 4, 8]) {
      const blk = await sampledBlockLoop(api, model, p.ids, p.N, K, temperature, SEED, hook);
      await api.markStep();
      const match = eq(ref, blk);
      console.log(
        `NOTE (characterization) — sampled K=${K}: HOOKED Gumbel block == HOOKED Gumbel host: ${match ? "match" : "MISMATCH (transient)"}`,
      );
    }
    const after = getGpuUncapturedErrorCount();
    console.log(
      `NOTE (characterization) — sampled-path uncaptured GPU errors this section: ${after - before} (pre-existing transient; sampled cutover opt-in until fixed)`,
    );
  }

  console.log(
    `\n=== VERDICT: ${
      FAIL === 0
        ? "PASS — the steering hook composes with the unrolled-K block, byte-identically"
        : `FAIL (${FAIL} checks)`
    } ===`,
  );
  process.exit(FAIL === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
