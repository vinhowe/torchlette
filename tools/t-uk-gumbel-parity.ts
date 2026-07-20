/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — P3 GUMBEL PARITY GATE.
 *
 * On-device stochastic sampling for the unrolled-K block is GUMBEL-MAX:
 *   id = argmax(logits/temperature + g),  g = -log(-log(u)),  u ~ U(0,1) on-device
 * with a per-POSITION seed (seed + absolutePosition) so the draw is a
 * deterministic function of position — RNG-as-DATA in the §3.5 sense: the seed is
 * host-set per position, the uniform is drawn on-device, and the selection closes
 * on-device (no per-token readback inside the block).
 *
 * The gate proves the sampled block is CORRECT and BYTE-REPRODUCIBLE:
 *  1. PARITY — the sampled block ids are byte-identical to a per-token host-loop
 *     reference that draws the SAME per-position seeds and computes the SAME
 *     on-device Gumbel-max (the sampling analogue of the greedy mother gate; both
 *     run identical device ops on identical inputs, so a match proves the on-device
 *     argmax->gather feedback carries the sampled id exactly as the readback loop
 *     would). Run over K∈{1,4,8} and a bucket crossing.
 *  2. DETERMINISM — the same seed decodes byte-identically twice (seed-as-data:
 *     nothing per-run-random leaks in).
 *  3. SEED SENSITIVITY — a different seed produces a different stream (the sampler
 *     is genuinely stochastic, not a disguised greedy).
 *  4. TEMPERATURE 0 — stays the greedy argmax path (== greedy decodeBlock).
 *  5. GUMBEL FORMULA (unit) — on a fixed logits + fixed uploaded uniform with an
 *     UNAMBIGUOUS margin, the on-device argmax(logits/temp + -log(-log(u)))
 *     equals the host computation (isolates the transform from fp near-ties).
 *
 * Run: eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX \
 *        LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-gumbel-parity.ts
 */

import {
  decodeBlock,
  gumbelUniform,
} from "../packages/qwen3-browser/src/generate";
import type {
  Qwen3Config,
  StaticKV,
} from "../packages/qwen3-browser/src/model";
import { Qwen3 } from "../packages/qwen3-browser/src/model";
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
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

/**
 * HOST reference: N Gumbel-max tokens, one readback per step, with the SAME
 * per-position seed and the SAME on-device Gumbel-max the block computes. This is
 * the per-token loop the unrolled block collapses; a byte match proves the
 * on-device feedback carries the sampled id exactly.
 */
async function sampledHostLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  temperature: number,
  seed: number,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const V = CONFIG.vocabSize;
  // Mirror the block EXACTLY: prefill greedily → firstTok (kv.len = promptLen),
  // then sample the subsequent tokens with the SAME per-position seed the block
  // uses (SEED + startLen + j, startLen = promptLen). One readback per step.
  const firstTok = await prefillFirst(api, model, promptIds, kv);
  await api.markStep();
  const startLen = kv.len; // = promptLen, the block's first-step absolute position
  const ids: number[] = [];
  let idx = api.tensorFromArray([firstTok], [1, 1]);
  for (let j = 0; j < N; j++) {
    const logits = api.noGrad(
      () => model.forward(idx, { staticKV: kv }).logits,
    ); // [1,1,V] — single-token forward, S=1
    const idT = api.noGrad(() => {
      // SAME single-source uniform + on-device transform the block uses.
      const u = api.tensorFromArray(gumbelUniform(seed + startLen + j, V), [
        1,
        1,
        V,
      ]);
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

/** BLOCK arm: N Gumbel-max tokens via repeated bucket-clipped decodeBlock. The
 *  first block's absolute start is kv.len after prefill (= promptLen); the host
 *  reference samples at pos = kv.len-1 after each forward — the SAME absolute
 *  positions, so the seeds line up. */
async function sampledBlockLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  K: number,
  temperature: number,
  seed: number,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const firstTok = await prefillFirst(api, model, promptIds, kv);
  await api.markStep();
  const ids: number[] = [];
  let lastTok = firstTok;
  while (ids.length < N) {
    const { ids: blk } = await decodeBlock(api, model, kv, lastTok, K, {
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

  console.log("=== P3 GUMBEL PARITY GATE (random-init Qwen3) ===");
  console.log(`config: ${JSON.stringify(CONFIG)}\n`);

  const temperature = 0.8;
  const SEED = 4242;
  const prompts: { name: string; ids: number[]; N: number }[] = [
    { name: "short", ids: [3, 14, 15, 92, 65], N: 20 },
    {
      name: "bucket-crossing (len~120)",
      ids: Array.from(
        { length: 120 },
        (_, i) => (i * 7 + 3) % CONFIG.vocabSize,
      ),
      N: 20,
    },
  ];

  // (1) PARITY: sampled block == sampled host loop, byte-identical, per K.
  for (const p of prompts) {
    const ref = await sampledHostLoop(
      api,
      model,
      p.ids,
      p.N,
      temperature,
      SEED,
    );
    await api.markStep();
    for (const K of [1, 4, 8]) {
      const blk = await sampledBlockLoop(
        api,
        model,
        p.ids,
        p.N,
        K,
        temperature,
        SEED,
      );
      await api.markStep();
      const match =
        ref.length === blk.length && ref.every((t, i) => t === blk[i]);
      ok(
        match,
        `[${p.name}] K=${K}: sampled block == sampled host loop (${p.N} toks, seed ${SEED})` +
          (match
            ? ""
            : `\n    host : [${ref.join(",")}]\n    block: [${blk.join(",")}]`),
      );
    }
  }

  // (2) DETERMINISM: same seed decodes byte-identically twice.
  {
    const p = prompts[0];
    const a = await sampledBlockLoop(
      api,
      model,
      p.ids,
      p.N,
      8,
      temperature,
      SEED,
    );
    await api.markStep();
    const b = await sampledBlockLoop(
      api,
      model,
      p.ids,
      p.N,
      8,
      temperature,
      SEED,
    );
    await api.markStep();
    ok(
      a.length === b.length && a.every((t, i) => t === b[i]),
      `determinism: same seed ${SEED} decodes byte-identically twice`,
    );

    // (3) SEED SENSITIVITY: a different seed produces a different stream. The
    // tiny random-init model is argmax-DOMINATED (one logit far ahead), so a
    // moderate temperature would never let gumbel flip the pick — use a HIGH
    // temperature (gumbel dominates logits ⇒ ~uniform sampling) so the seed
    // genuinely drives the stream. Parity (test 1) is byte-identical at ANY
    // temperature, so this does not weaken the correctness proof.
    const hotT = 30;
    const a2 = await sampledBlockLoop(api, model, p.ids, p.N, 8, hotT, SEED);
    await api.markStep();
    const c = await sampledBlockLoop(api, model, p.ids, p.N, 8, hotT, SEED + 1);
    await api.markStep();
    ok(
      !(a2.length === c.length && a2.every((t, i) => t === c[i])),
      `seed sensitivity (T=${hotT}): seed ${SEED} vs ${SEED + 1} differ (genuinely stochastic)`,
    );
  }

  // (4) TEMPERATURE 0: stays the greedy path (== greedy decodeBlock).
  {
    const p = prompts[0];
    const kvG = model.allocStaticKV(CONFIG.maxSeqLen);
    const ftG = await prefillFirst(api, model, p.ids, kvG);
    await api.markStep();
    const { ids: greedy } = await decodeBlock(api, model, kvG, ftG, 8);
    await api.markStep();
    const kvT = model.allocStaticKV(CONFIG.maxSeqLen);
    const ftT = await prefillFirst(api, model, p.ids, kvT);
    await api.markStep();
    const { ids: temp0 } = await decodeBlock(api, model, kvT, ftT, 8, {
      sample: { temperature: 0, seed: SEED },
    });
    await api.markStep();
    ok(
      greedy.length === temp0.length && greedy.every((t, i) => t === temp0[i]),
      "temperature=0 sample == greedy argmax (greedy path unchanged)",
    );
  }

  // (5) GUMBEL FORMULA (unit): on-device argmax(logits/temp + -log(-log(u)))
  //     == host computation, on a fixed logits + fixed uploaded uniform chosen so
  //     the argmax has a wide margin (isolates the transform from fp near-ties).
  {
    const V = 8;
    const t = 0.7;
    const logitsArr = [0.1, 0.2, 5.0, 0.05, 0.3, 0.15, 0.25, 0.1];
    // uniform close to 1 for the winner boosts its gumbel; keep others mid.
    const uArr = [0.5, 0.5, 0.999, 0.5, 0.5, 0.5, 0.5, 0.5];
    const logits = api.tensorFromArray(logitsArr, [1, V]);
    const u = api.tensorFromArray(uArr, [1, V]);
    const idT = api.noGrad(() => {
      const g = api.neg(api.log(api.neg(api.log(u))));
      const scaled = api.add(api.div(logits, t), g);
      return api.argmax(scaled, { dim: -1, keepdim: false });
    });
    const dev = Math.round((await api.cpu(idT))[0]);
    let host = 0;
    let hostBest = -Infinity;
    for (let i = 0; i < V; i++) {
      const g = -Math.log(-Math.log(uArr[i]));
      const s = logitsArr[i] / t + g;
      if (s > hostBest) {
        hostBest = s;
        host = i;
      }
    }
    ok(
      dev === host,
      `gumbel formula (unit): device argmax ${dev} == host argmax ${host}`,
    );
  }

  console.log(
    `\n=== VERDICT: ${
      FAIL === 0
        ? "PASS — on-device Gumbel-max sampling is correct, deterministic, and block==host"
        : `FAIL (${FAIL} checks)`
    } ===`,
  );
  process.exit(FAIL === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
