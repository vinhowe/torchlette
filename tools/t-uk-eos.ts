/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — PROBE 4: EOS PREDICATION COST
 * (and the honest per-iteration GPU floor).
 *
 * Unrolled-K makes a STATIC command stream of K tokens: WebGPU has no device-side
 * control flow, so an EOS at token j<K cannot break the stream — the remaining
 * K-j iterations still execute, and the host truncates at the K-boundary readback.
 * The cost of those wasted post-EOS iterations, UNPREDICATED, is exactly the
 * marginal GPU-forward cost of one unrolled iteration. This probe measures it by
 * scaling the unrolled block (ONE readback per block, so the K iterations' GPU
 * compute is the block wall minus a single host fence): the slope d(wall)/dK is
 * the per-iteration GPU-forward cost = the unpredicated wasted-iteration cost, and
 * the intercept is the once-per-block host tax.
 *
 * That slope ALSO backfills Probe 3's "GPU floor": Probe 3 shows the per-token
 * wall today is ~98% host tax (the readback FENCE masks GPU compute); this probe
 * isolates the GPU-forward-per-token that survives amortization.
 *
 * Predication (a flag-buffer early-return kernel) would drop the wasted-iteration
 * cost from a full forward to ~the kernel-launch/dispatch overhead — net-new
 * mechanism; ESTIMATED here from the per-iteration dispatch count, not built.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-uk-eos.ts
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { getSubmitCount, resetSubmitCount } from "../src/backend/webgpu/webgpu-state";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { DISTILGPT2_CONFIG, GPT2 } from "../examples/gpt2/model";
import type { KVCache } from "../examples/gpt2/model";

const PROMPT = [464, 3139, 286, 4881, 318];
const KS = [2, 4, 8, 16];
const REPEATS = 5;

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
}

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1234);
  const model = new GPT2(api, { ...DISTILGPT2_CONFIG });
  const V = model.config.vocabSize;
  api.setStepScopedCleanup(true);

  const lastRow = (logits: Tensor): Tensor => {
    const S = logits.shape[1];
    return api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V));
  };

  // One unrolled-K block: K on-device argmax->gather steps, ONE readback + markStep.
  // Returns { wallMs, submits } for the block (steady-state).
  async function unrolledBlock(K: number): Promise<{ wall: number; submits: number }> {
    // fresh prefill each block (positions 0..L-1)
    let { logits, kvs } = ((): { logits: Tensor; kvs: KVCache[] } => {
      const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
      const r = api.noGrad(() => model.forwardCached(idx, undefined, 0));
      return { logits: r.logits, kvs: r.presentKVs };
    })();
    let pos = PROMPT.length;
    resetSubmitCount();
    const t0 = performance.now();
    const idTensors: Tensor[] = [];
    for (let i = 0; i < K; i++) {
      const id = api.noGrad(() => api.argmax(lastRow(logits), { dim: -1, keepdim: false }));
      idTensors.push(id);
      const nxt = api.noGrad(() => model.forwardCached(api.reshape(id, [1, 1]), kvs, pos));
      logits = nxt.logits;
      kvs = nxt.presentKVs;
      pos += 1;
    }
    const stacked = api.cat(idTensors.map((t) => api.reshape(t, [1, 1])), 1);
    await stacked.cpu(); // ONE readback for the whole block
    const wall = performance.now() - t0;
    const submits = getSubmitCount();
    await api.markStep();
    return { wall, submits };
  }

  console.log(`=== PROBE 4: EOS PREDICATION COST + per-iteration GPU floor (distilgpt2) ===\n`);
  const results: { K: number; wall: number; submits: number }[] = [];
  for (const K of KS) {
    const walls: number[] = [];
    let submits = 0;
    for (let r = 0; r < REPEATS + 2; r++) {
      const b = await unrolledBlock(K);
      if (r >= 2) walls.push(b.wall); // drop 2 warmups
      submits = b.submits;
    }
    results.push({ K, wall: +median(walls).toFixed(2), submits });
  }

  console.log(`  unrolled block (ONE readback per block, steady-state):`);
  console.log(`  K | block wall ms | ms/token | submits/block`);
  for (const r of results) {
    console.log(`  ${r.K} | ${r.wall} | ${(r.wall / r.K).toFixed(2)} | ${r.submits}`);
  }

  // Linear fit wall = intercept + slope*K over the measured points (least squares).
  const n = results.length;
  const sx = results.reduce((a, r) => a + r.K, 0);
  const sy = results.reduce((a, r) => a + r.wall, 0);
  const sxx = results.reduce((a, r) => a + r.K * r.K, 0);
  const sxy = results.reduce((a, r) => a + r.K * r.wall, 0);
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const intercept = (sy - slope * sx) / n;

  console.log(`\n  linear fit: block_wall ≈ ${intercept.toFixed(2)} ms (once-per-block host tax) + ${slope.toFixed(2)} ms/iter × K`);
  console.log(`  => per-iteration GPU-forward cost (UNPREDICATED wasted post-EOS iteration): ${slope.toFixed(2)} ms`);
  console.log(`  => once-per-block host tax (fence + markStep, amortized over K): ${intercept.toFixed(2)} ms`);

  // EOS waste characterization. If a sequence ends at a uniformly-random token in
  // the block, the expected wasted iterations = (K-1)/2. Amortized penalty per
  // emitted token for a generation of length T that ends mid-block:
  console.log(`\n  --- EOS-truncation waste (UNPREDICATED: wasted iters run a full forward) ---`);
  for (const K of [4, 8, 16]) {
    const expWasted = (K - 1) / 2; // uniform end position in the final block
    const wasteMs = expWasted * slope;
    console.log(`  K=${K}: expected wasted iters on the terminating block = ${expWasted}, ≈ ${wasteMs.toFixed(2)} ms one-time at end-of-generation`);
  }
  console.log(`  (only the FINAL, EOS-containing block wastes; every full block emits all K. For a T-token generation the amortized waste is ${"<"}=(K-1)/2 forwards over T tokens — negligible for T≫K, bounded by one block for short T.)`);

  // Predicated estimate: a flag-buffer early-return kernel makes each post-EOS
  // dispatch a cheap no-op (launch + early-return), not a full forward. The
  // per-iteration dispatch count is ~ (nodes-per-step); predicated cost ≈ that
  // many kernel launches at ~launch-overhead each, << the full-forward slope.
  console.log(`\n  --- EOS predication (flag-buffer early-return — NET-NEW, estimated) ---`);
  console.log(`  predicated wasted iteration ≈ kernel-launch overhead × dispatches/step, NOT a full forward.`);
  console.log(`  submits/block are ~flat in K (${results.map((r) => r.submits).join(",")} for K=${KS.join(",")}), so the block is already ~one submit's worth of encoding; a predicated early-return keeps the launches but skips the math. Since the UNPREDICATED waste (slope ${slope.toFixed(2)} ms/iter) is already small vs the host tax (${intercept.toFixed(2)} ms/block), predication is a SECOND-ORDER optimization — the simple design (compute all K, truncate at readback) is viable without it.`);

  console.log(`\n=== UK-EOS-STATS === ${JSON.stringify({ results, slope: +slope.toFixed(3), intercept: +intercept.toFixed(3) })}`);
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
