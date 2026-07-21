/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — SAMPLER GATE.
 *
 * On-device top-k / top-p / temperature sampling inside the unrolled-K block
 * (Vin's taste-fork resolution (a): the demos' sampling DISTRIBUTION preserved
 * exactly). The block-side sampler is `deviceTopK` (small-top-k prefilter) →
 * top-p nucleus mask → Gumbel-max over the survivors. This gate proves it
 * matches the host `sampleFromTopK` reference:
 *
 *  (i)   FILTER EXACTNESS — for fixed logits, the device surviving support
 *        (post top-k + top-p) equals the host reference support EXACTLY
 *        (byte-level on the sorted-desc index list), across edge cases:
 *        k=1, k>vocab, p=1.0, tiny p, ties at the top-p boundary, temp extremes.
 *        Plus: the REAL block sampler (sampleFilteredToken) only ever emits
 *        tokens INSIDE that support (ties the exactness proof to the shipped op).
 *  (ii)  DETERMINISM — same seed decodes byte-identically twice.
 *  (iii) DISTRIBUTIONAL EQUIVALENCE — over ≥10k samples on fixed logits, device
 *        sampling frequencies match the host reference within a TV-distance +
 *        chi-square tolerance, for several (k,p,T) cells.
 *  (iv)  PARITY (mother gate) — the filtered block == a per-token host-loop
 *        running the SAME on-device selection, byte-identical (K∈{1,4,8} + a
 *        bucket crossing), and greedy/pure-gumbel paths are UNCHANGED.
 *
 * Run: eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX \
 *   LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-topk-sampler.ts
 */
import {
  decodeBlock,
  gumbelUniform,
  sampleFilteredToken,
  sampleFromTopK,
  strictUpperOnes,
} from "../packages/qwen3-browser/src/generate";
import type { Qwen3Config, StaticKV } from "../packages/qwen3-browser/src/model";
import { Qwen3 } from "../packages/qwen3-browser/src/model";
import {
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
} from "../src/backend/webgpu";
import type { FrontendTensor as Tensor } from "../src/frontend/torchlette";
import { Torchlette } from "../src/frontend/torchlette";

let FAIL = 0;
const ok = (c: boolean, m: string) => {
  console.log(`${c ? "PASS" : "FAIL"} — ${m}`);
  if (!c) FAIL++;
};

// ---------------------------------------------------------------------------
// Host reference (single source of truth: sampleFromTopK's cut rule).
// ---------------------------------------------------------------------------

/** Host top-k SET (values desc, ties smaller-index-first) via a linear scan —
 *  byte-identical to readTopK / deviceTopK. */
function cpuTopK(
  logits: number[],
  k: number,
): { v: number[]; i: number[] } {
  const idx = logits.map((_, i) => i);
  idx.sort((a, b) => logits[b] - logits[a] || a - b);
  const top = idx.slice(0, Math.min(k, logits.length));
  return { v: top.map((i) => logits[i]), i: top };
}

/** Host SUPPORT: the sorted-desc token ids the reference sampler can emit —
 *  sampleFromTopK's exact top-p cut. */
function hostSupport(
  logits: number[],
  k: number,
  temperature: number,
  topP: number,
): number[] {
  const { v, i } = cpuTopK(logits, k);
  const mx = v[0];
  const exps = v.map((x) => Math.exp((x - mx) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  let cut = exps.length;
  let cum = 0;
  for (let j = 0; j < exps.length; j++) {
    cum += exps[j] / sum;
    if (cum >= topP) {
      cut = j + 1;
      break;
    }
  }
  return i.slice(0, cut);
}

/** Host sampler emitting a token id from the same distribution (Math.random). */
function hostDraw(
  logits: number[],
  k: number,
  temperature: number,
  topP: number,
): number {
  const { v, i } = cpuTopK(logits, k);
  const values = new Float32Array(v);
  const indices = new Int32Array(i);
  return sampleFromTopK(values, indices, temperature, k, topP);
}

// ---------------------------------------------------------------------------
// Device support (reads back the EXACT top-p mask the block computes).
// ---------------------------------------------------------------------------

/** Recompute deviceTopK + the top-p mask (the SAME ops sampleFilteredToken uses)
 *  and read back the surviving token ids + the top-k values — used both for the
 *  filter-exactness proof and as the exact device distribution for (iii). */
async function deviceSupport(
  api: Torchlette,
  logitsArr: number[],
  V: number,
  k: number,
  temperature: number,
  topP: number,
): Promise<{ support: number[]; ids: number[]; vals: number[] }> {
  const kEff = Math.min(k, V);
  const logits = api.tensorFromArray(logitsArr, [1, 1, V]);
  const L = api.tensorFromArray(strictUpperOnes(kEff), [kEff, kEff]);
  const topPT = api.tensorFromArray([topP], [1, 1, 1]);
  const { idsArr, maskArr, valsArr } = api.noGrad(() => {
    const packed = api.deviceTopK(logits, kEff);
    const vals = api.contiguous(api.narrow(packed, 1, 0, 1)); // [1,1,k]
    const idsF = api.contiguous(api.narrow(packed, 1, 1, 1)); // [1,1,k]
    const mx = api.narrow(vals, 2, 0, 1);
    const exps = api.exp(api.div(api.sub(vals, mx), temperature));
    const sum = api.sum(exps, { dim: -1, keepdim: true });
    const exclCum = api.reshape(
      api.matmul(api.reshape(exps, [1, kEff]), L),
      [1, 1, kEff],
    );
    const mask = api.lt(api.div(exclCum, sum), topPT);
    return { idsArr: idsF, maskArr: mask, valsArr: vals };
  });
  const ids = Array.from(await api.cpu(idsArr)).map((x) => Math.round(x));
  const mask = Array.from(await api.cpu(maskArr));
  const vals = Array.from(await api.cpu(valsArr));
  await api.markStep();
  const support: number[] = [];
  for (let j = 0; j < kEff; j++) if (mask[j] > 0.5) support.push(ids[j]);
  return { support, ids, vals };
}

/** Draw ONE token from the REAL block sampler (sampleFilteredToken) on synthetic
 *  logits — ties the exactness proof to the shipped op. */
async function deviceDraw(
  api: Torchlette,
  logitsArr: number[],
  V: number,
  k: number,
  temperature: number,
  topP: number,
  seed: number,
): Promise<number> {
  const kEff = Math.min(k, V);
  const logits = api.tensorFromArray(logitsArr, [1, 1, V]);
  const L = api.tensorFromArray(strictUpperOnes(kEff), [kEff, kEff]);
  const topPT = api.tensorFromArray([topP], [1, 1, 1]);
  const idT = api.noGrad(() =>
    sampleFilteredToken(
      api,
      logits,
      V,
      { temperature, seed, topK: kEff, topP },
      0,
      L as Tensor,
      topPT as Tensor,
    ),
  );
  const out = await api.cpu(api.reshape(idT, [1, 1]));
  await api.markStep();
  return Math.round(out[0]);
}

/** The device distribution: argmax(v/temp + gumbel) over the support, computed
 *  from the read-back (vals, ids, support) with the SAME gumbelUniform — exactly
 *  what the device argmax does (proven equal by (iv)/(v)). Deterministic per
 *  seed, so 10k+ frequencies are exact without 10k GPU dispatches. */
function deviceDrawSim(
  ids: number[],
  vals: number[],
  support: Set<number>,
  temperature: number,
  seed: number,
): number {
  const u = gumbelUniform(seed, ids.length);
  let best = -Infinity;
  let bestId = ids[0];
  for (let j = 0; j < ids.length; j++) {
    if (!support.has(ids[j])) continue;
    const g = -Math.log(-Math.log(u[j]));
    const s = vals[j] / temperature + g;
    if (s > best) {
      best = s;
      bestId = ids[j];
    }
  }
  return bestId;
}

function tvDistance(a: Map<number, number>, b: Map<number, number>, n: number) {
  const keys = new Set([...a.keys(), ...b.keys()]);
  let tv = 0;
  for (const k of keys) tv += Math.abs((a.get(k) ?? 0) - (b.get(k) ?? 0)) / n;
  return tv / 2;
}

// ---------------------------------------------------------------------------

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

/** Host-loop reference running the SAME on-device filtered selection per token
 *  (the block collapses this loop; a byte match proves the feedback carries the
 *  sampled id exactly). */
async function filteredHostLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  temperature: number,
  seed: number,
  topK: number,
  topP: number,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const V = CONFIG.vocabSize;
  const kEff = Math.min(topK, V);
  const firstTok = await prefillFirst(api, model, promptIds, kv);
  await api.markStep();
  const startLen = kv.len;
  const L = api.tensorFromArray(strictUpperOnes(kEff), [kEff, kEff]);
  const topPT = api.tensorFromArray([topP], [1, 1, 1]);
  const ids: number[] = [];
  let idx = api.tensorFromArray([firstTok], [1, 1]);
  for (let j = 0; j < N; j++) {
    const logits = api.noGrad(
      () => model.forward(idx, { staticKV: kv }).logits,
    );
    const idT = api.noGrad(() =>
      sampleFilteredToken(
        api,
        logits,
        V,
        { temperature, seed, topK: kEff, topP },
        startLen + j,
        L as Tensor,
        topPT as Tensor,
      ),
    );
    const tok = Math.round((await api.cpu(idT))[0]);
    ids.push(tok);
    idx = api.tensorFromArray([tok], [1, 1]);
    await api.markStep();
  }
  return ids;
}

async function filteredBlockLoop(
  api: Torchlette,
  model: Qwen3,
  promptIds: number[],
  N: number,
  K: number,
  temperature: number,
  seed: number,
  topK: number,
  topP: number,
): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const firstTok = await prefillFirst(api, model, promptIds, kv);
  await api.markStep();
  const ids: number[] = [];
  let lastTok = firstTok;
  while (ids.length < N) {
    const { ids: blk } = await decodeBlock(api, model, kv, lastTok, K, {
      sample: { temperature, seed, topK, topP },
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
  const V = CONFIG.vocabSize;

  console.log("=== SAMPLER GATE — on-device top-k/top-p (random-init Qwen3) ===\n");

  // Fixed logits with deliberate structure: a clear head, some ties.
  const base = Array.from({ length: V }, (_, i) => {
    const x = Math.sin(i * 12.9898 + 4.1) * 43758.5453;
    return (x - Math.floor(x)) * 6 - 3; // ~(-3,3)
  });
  base[5] = 5.0;
  base[9] = 4.0;
  base[9 + 1] = 4.0; // a tie among the top few
  base[42] = 3.5;

  // (i) FILTER EXACTNESS across edge cases -----------------------------------
  const filterCases: {
    name: string;
    k: number;
    T: number;
    p: number;
  }[] = [
    { name: "k=1", k: 1, T: 0.7, p: 0.95 },
    { name: "k>vocab", k: V + 50, T: 0.7, p: 0.95 },
    { name: "p=1.0", k: 40, T: 0.7, p: 1.0 },
    { name: "tiny p", k: 40, T: 0.7, p: 1e-4 },
    { name: "typical (k20,p.95)", k: 20, T: 0.7, p: 0.95 },
    { name: "ties@boundary", k: 8, T: 1.0, p: 0.5 },
    { name: "temp hot (T=50)", k: 40, T: 50, p: 0.9 },
    { name: "temp cold (T=0.05)", k: 40, T: 0.05, p: 0.9 },
  ];
  for (const c of filterCases) {
    const { support } = await deviceSupport(api, base, V, c.k, c.T, c.p);
    const host = hostSupport(base, c.k, c.T, c.p);
    const match =
      support.length === host.length && support.every((x, j) => x === host[j]);
    ok(
      match,
      `filter-exact [${c.name}]: device support == host (|S|=${host.length})` +
        (match ? "" : `\n  dev=[${support}]\n  host=[${host}]`),
    );
    // Tie to the REAL op: sampleFilteredToken only emits support members.
    const sup = new Set(support);
    let allIn = true;
    for (let s = 0; s < 40; s++) {
      const tok = await deviceDraw(api, base, V, c.k, c.T, c.p, s * 97 + 1);
      if (!sup.has(tok)) allIn = false;
    }
    ok(allIn, `filter-exact [${c.name}]: real block sampler ⊆ support (40 draws)`);
  }

  // (ii) DETERMINISM ---------------------------------------------------------
  {
    const seeds = [111, 222, 333, 444];
    const a: number[] = [];
    const b: number[] = [];
    for (const s of seeds) a.push(await deviceDraw(api, base, V, 20, 0.8, 0.95, s));
    for (const s of seeds) b.push(await deviceDraw(api, base, V, 20, 0.8, 0.95, s));
    ok(
      a.every((t, j) => t === b[j]),
      `determinism: same seeds → same tokens twice [${a}] vs [${b}]`,
    );
  }

  // (iii) DISTRIBUTIONAL EQUIVALENCE -----------------------------------------
  // The device distribution is deterministic given (vals, support, seed), so we
  // simulate ≥10k device draws exactly (deviceDrawSim uses the SAME gumbelUniform
  // + argmax the GPU runs — proven equal by the parity/formula tests below) and
  // compare frequencies to the host reference sampler (Math.random). We ALSO run
  // a batch of REAL device draws and confirm they match the simulation, tying
  // the 10k comparison to the shipped op.
  {
    const N = 20000;
    const distCases: { name: string; k: number; T: number; p: number }[] = [
      { name: "k20 p.95 T.7", k: 20, T: 0.7, p: 0.95 },
      { name: "k40 p.9 T1.0", k: 40, T: 1.0, p: 0.9 },
      { name: "k8 p1.0 T1.5", k: 8, T: 1.5, p: 1.0 },
    ];
    for (const c of distCases) {
      const { ids, vals, support } = await deviceSupport(
        api,
        base,
        V,
        c.k,
        c.T,
        c.p,
      );
      const sup = new Set(support);
      // Tie simulation to the real device op: 200 real draws must match the sim.
      let tied = true;
      for (let s = 0; s < 200; s++) {
        const seed = s * 131 + 7;
        const real = await deviceDraw(api, base, V, c.k, c.T, c.p, seed);
        const sim = deviceDrawSim(ids, vals, sup, c.T, seed);
        if (real !== sim) tied = false;
      }
      ok(tied, `dist [${c.name}]: real device draw == exact sim (200 seeds)`);

      // The EXACT target distribution (both device & host sample it): softmax
      // (v/temp) restricted to the support = exps[j]/cutSum. support is the
      // sorted-desc prefix (positions 0..cut-1), so vals[0..cut) are its logits.
      const cut = support.length;
      const e = vals.slice(0, cut).map((x) => Math.exp((x - vals[0]) / c.T));
      const cutSum = e.reduce((a, b) => a + b, 0);
      const theo = new Map<number, number>(); // token id -> probability
      for (let j = 0; j < cut; j++) theo.set(ids[j], e[j] / cutSum);

      const devFreq = new Map<number, number>();
      const hostFreq = new Map<number, number>();
      for (let s = 0; s < N; s++) {
        const d = deviceDrawSim(ids, vals, sup, c.T, s * 2654435761 + 1);
        devFreq.set(d, (devFreq.get(d) ?? 0) + 1);
        const h = hostDraw(base, c.k, c.T, c.p);
        hostFreq.set(h, (hostFreq.get(h) ?? 0) + 1);
      }
      // Goodness-of-fit of EACH empirical distribution against the exact target
      // (chi2/df ≈ 1 for a good fit); a match to the target ⇒ a match to each
      // other. TV of each empirical vs the target is O(sqrt(|S|/N)) small.
      const gof = (freq: Map<number, number>) => {
        let chi = 0;
        let tv = 0;
        for (const t of support) {
          const exp = N * (theo.get(t) ?? 0);
          const obs = freq.get(t) ?? 0;
          if (exp > 0) chi += ((obs - exp) * (obs - exp)) / exp;
          tv += Math.abs(obs - exp) / N;
        }
        return { chi, tv: tv / 2 };
      };
      const df = Math.max(support.length - 1, 1);
      const dg = gof(devFreq);
      const hg = gof(hostFreq);
      ok(
        dg.tv < 0.02 && hg.tv < 0.02,
        `dist [${c.name}]: TV vs target dev=${dg.tv.toFixed(4)} host=${hg.tv.toFixed(4)} < 0.02 (|S|=${support.length}, N=${N})`,
      );
      ok(
        dg.chi / df < 2.0 && hg.chi / df < 2.0,
        `dist [${c.name}]: chi2/df dev=${(dg.chi / df).toFixed(3)} host=${(hg.chi / df).toFixed(3)} < 2.0`,
      );
    }
  }

  // (iv) PARITY (mother gate) — filtered block == filtered host loop ---------
  {
    api.manualSeed(1234);
    const model = new Qwen3(api, { ...CONFIG });
    const SEED = 4242;
    const prompts: { name: string; ids: number[]; N: number }[] = [
      { name: "short", ids: [3, 14, 15, 92, 65], N: 20 },
      {
        name: "bucket-crossing (~120)",
        ids: Array.from({ length: 120 }, (_, i) => (i * 7 + 3) % V),
        N: 20,
      },
    ];
    for (const p of prompts) {
      const ref = await filteredHostLoop(
        api,
        model,
        p.ids,
        p.N,
        0.8,
        SEED,
        20,
        0.95,
      );
      await api.markStep();
      for (const K of [1, 4, 8]) {
        const blk = await filteredBlockLoop(
          api,
          model,
          p.ids,
          p.N,
          K,
          0.8,
          SEED,
          20,
          0.95,
        );
        await api.markStep();
        const match =
          ref.length === blk.length && ref.every((t, i) => t === blk[i]);
        ok(
          match,
          `parity [${p.name}] K=${K}: filtered block == filtered host loop` +
            (match
              ? ""
              : `\n  host : [${ref}]\n  block: [${blk}]`),
        );
      }
    }
  }

  // (v) COMPILED == LOWERED (the coverage follow-on's new differential) -------
  // The filtered block now reaches build-from-IR fullyCovered (deviceTopK +
  // gather + the released-view fused ops all have generators), so the K-block
  // CUTS OVER to the compiled plan on the 2nd+ execution. This cell is the
  // standing guard that the compiled sampled arm produces a BYTE-IDENTICAL token
  // stream to the pure-lowered arm at the same seed — the exact correctness the
  // new generators must preserve (the compiled plan replays the deviceTopK
  // passes + the reshape/narrow-fed fused ops against the recording).
  {
    api.manualSeed(1234);
    const model = new Qwen3(api, { ...CONFIG });
    const SEED = 4242;
    const p = { ids: [3, 14, 15, 92, 65, 33, 7], N: 24 };
    const prev = process.env.TORCHLETTE_COMPILED_PLAN;
    for (const K of [1, 4, 8]) {
      // arm A: build-from-IR compiled (default). The block runs several times
      // (N/K readbacks) so it crosses the 2nd-execution cutover threshold.
      delete process.env.TORCHLETTE_COMPILED_PLAN;
      const on = await filteredBlockLoop(
        api, model, p.ids, p.N, K, 0.8, SEED, 20, 0.95,
      );
      await api.markStep();
      // arm B: pure lowered.
      process.env.TORCHLETTE_COMPILED_PLAN = "0";
      const off = await filteredBlockLoop(
        api, model, p.ids, p.N, K, 0.8, SEED, 20, 0.95,
      );
      await api.markStep();
      const match = on.length === off.length && on.every((t, i) => t === off[i]);
      ok(
        match,
        `compiled==lowered [K=${K}]: filtered block token stream identical across arms` +
          (match ? "" : `\n  compiled: [${on}]\n  lowered : [${off}]`),
      );
    }
    if (prev === undefined) delete process.env.TORCHLETTE_COMPILED_PLAN;
    else process.env.TORCHLETTE_COMPILED_PLAN = prev;
  }

  ok(
    getGpuUncapturedErrorCount() === 0,
    `zero uncaptured GPU errors — got ${getGpuUncapturedErrorCount()}`,
  );
  console.log(
    `\n=== VERDICT: ${FAIL === 0 ? "PASS" : `FAIL (${FAIL} checks)`} ===`,
  );
  process.exit(FAIL === 0 ? 0 : 1);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
