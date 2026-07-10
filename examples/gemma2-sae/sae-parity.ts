/**
 * SAE encode parity gate — the load-bearing pipeline check.
 *
 * Loads the layer-20 residual dump produced by dump-sae-reference.py, runs the
 * TypeScript GemmaScopeSAE.encode over the SAME residual, and compares against
 * the numpy JumpReLU reference: (1) feature-activation values to an f32-parity
 * tolerance, and (2) the top-K ACTIVE FEATURE INDICES exactly. This proves the
 * SAE math (encode formula, JumpReLU gating, threshold semantics, weight
 * layout) before any UI exists.
 *
 * Prereq: examples/gemma2-sae/dump-sae-reference.py (writes .../sae-parity/).
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *      npx tsx examples/gemma2-sae/sae-parity.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getWebGPUInitError,
  initWebGPU,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { GemmaScopeSAE } from "../../packages/gemma-scope-sae/src/sae";
import { loadSAEFromDir } from "../../packages/gemma-scope-sae/src/loader-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SAE_DIR = path.join(
  __dirname,
  "../../ckpts/gemma-scope-2b-pt-res/sae-layer20-16k",
);
const PARITY_DIR = path.join(
  __dirname,
  "../../ckpts/gemma-scope-2b-pt-res/sae-parity",
);

type PromptRef = {
  text: string;
  seq_len: number;
  resid_file: string;
  acts_file: string;
  resid_shape: number[];
  acts_shape: number[];
  agg_topk_idx: number[];
  last_topk_idx: number[];
  n_active_last: number;
};

function readBin(dir: string, file: string): Float32Array {
  const buf = fs.readFileSync(path.join(dir, file));
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function diffStats(a: Float32Array, b: Float32Array) {
  let maxAbs = 0;
  let sumAbs = 0;
  let maxRel = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) maxAbs = d;
    sumAbs += d;
    const denom = Math.abs(b[i]);
    if (denom > 1e-3) maxRel = Math.max(maxRel, d / denom);
  }
  return { maxAbs, meanAbs: sumAbs / a.length, maxRel };
}

/** Top-K feature indices from a flat [F] activation row (desc by value). */
function topK(row: Float32Array, k: number): number[] {
  const idx = Array.from({ length: row.length }, (_, i) => i);
  idx.sort((x, y) => row[y] - row[x]);
  return idx.slice(0, k);
}

/** Aggregate (max over seq) a [seq, F] flat array → [F]. */
function aggMax(acts: Float32Array, seq: number, F: number): Float32Array {
  const out = new Float32Array(F);
  for (let s = 0; s < seq; s++) {
    const off = s * F;
    for (let f = 0; f < F; f++) if (acts[off + f] > out[f]) out[f] = acts[off + f];
  }
  return out;
}

async function main() {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(PARITY_DIR, "manifest.json"), "utf-8"),
  );
  const topKN = manifest.topk as number;

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const { config, params } = loadSAEFromDir(SAE_DIR);
  const sae = GemmaScopeSAE.load(api, config, params, { dtype: "f32" });
  const F = config.numFeatures;
  console.log(
    `SAE loaded: dModel=${config.dModel} numFeatures=${F} layer=${config.layer} saeId=${config.neuronpediaSaeId}`,
  );

  // Encode-formula tolerance. The residual is byte-identical (Python dump), so
  // the only source of diff is f32 matmul accumulation order (GPU vs numpy).
  // Feature pre-acts run to ~±180; an f32 dot over 2304 terms → ~1e-2 abs is the
  // realistic floor. Top-K indices must match EXACTLY (the real semantic gate).
  const VAL_GATE = 5e-2;

  let allPass = true;
  for (const [pi, ref] of (manifest.prompts as PromptRef[]).entries()) {
    const [seq, dModel] = ref.resid_shape;
    const residArr = readBin(PARITY_DIR, ref.resid_file);
    const refActs = readBin(PARITY_DIR, ref.acts_file);

    const resid = api.tensorFromArray(residArr, [seq, dModel]);
    const acts = api.noGrad(() => sae.encode(api, resid));
    const ours = new Float32Array(await acts.cpu());

    if (ours.length !== refActs.length)
      throw new Error(`acts length ${ours.length} vs ref ${refActs.length}`);

    const s = diffStats(ours, refActs);

    // Top-K over the max-aggregated activations (what the inspector shows) and
    // the last-position activations (what last-token steering sees). Both must
    // match the reference exactly.
    const oursAgg = aggMax(ours, seq, F);
    const oursAggTop = topK(oursAgg, topKN);
    const aggMatch = oursAggTop.every((f, i) => f === ref.agg_topk_idx[i]);

    const oursLastTop = topK(ours.subarray((seq - 1) * F, seq * F), topKN);
    const lastMatch = oursLastTop.every((f, i) => f === ref.last_topk_idx[i]);

    const nActiveLast = ours
      .subarray((seq - 1) * F, seq * F)
      .reduce((c, v) => c + (v > 0 ? 1 : 0), 0);

    const pass = s.maxAbs < VAL_GATE && aggMatch && lastMatch;
    allPass &&= pass;
    console.log(`\n=== prompt ${pi}: ${JSON.stringify(ref.text.slice(0, 45))} (seq=${seq})`);
    console.log(
      `  acts   maxAbs=${s.maxAbs.toExponential(3)} meanAbs=${s.meanAbs.toExponential(3)} maxRel=${s.maxRel.toExponential(2)}`,
    );
    console.log(
      `  n_active_last ours=${nActiveLast} ref=${ref.n_active_last} (L0≈71 target)`,
    );
    console.log(`  agg  top${topKN} match=${aggMatch}`);
    console.log(`       ours=${JSON.stringify(oursAggTop.slice(0, 8))}`);
    console.log(`       ref =${JSON.stringify(ref.agg_topk_idx.slice(0, 8))}`);
    console.log(`  last top${topKN} match=${lastMatch}  →  ${pass ? "PASS" : "FAIL"}`);
    await api.markStep();
  }

  console.log(`\n${allPass ? "SAE PARITY: ALL PROMPTS PASS" : "SAE PARITY FAILURES — see above"}`);
  process.exit(allPass ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
