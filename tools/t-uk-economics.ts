/**
 * THE CRYSTAL PUSH — CAMPAIGN 1 (unrolled-K decode) — PROBE 3: THE ECONOMICS.
 *
 * Decode today is host-in-the-loop: each token forces a logits readback, the
 * host argmaxes it, builds the next single-token graph, and marks the step.
 * Unrolled-K collapses K tokens into ONE static command stream, so the
 * host-side costs (readback fence, JS graph build, markStep bookkeeping) are
 * paid ONCE PER K instead of once per token. This probe MEASURES the current
 * per-token host overhead, decomposed, so the K-amortized projection is honest.
 *
 * It instruments a real distilgpt2 growing-KV decode loop (the phase-1 decode
 * shape, `forwardCached`) and times, per token:
 *   - t_build    : constructing idx + forwardCached (lazy node construction, no force)
 *   - t_readback : `await logits.cpu()` — the GPU->CPU fence + mapAsync round-trip
 *   - t_hostarg  : the host argmax over the vocab row
 *   - t_markstep : `await api.markStep()` boundary bookkeeping
 *   - t_total    : wall per token
 * under two arms (tape OFF / tape ON via TORCHLETTE_STEP_TAPE), in isolated
 * child processes (the flag is a module-load const).
 *
 * The K-amortized projection: unrolled-K pays t_readback + t_hostarg + t_markstep
 * ONCE per K-block (a single K-id readback at the K-boundary; a single markStep),
 * and t_build becomes a once-per-trace compile amortized over the run. So the
 * per-token host tax collapses from (build+readback+hostarg+markstep) to
 * (readback+hostarg+markstep)/K + the amortized replay dispatch. Reported below.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-uk-economics.ts [steps=32]
 */
import { getWebGPUInitError, initWebGPU, destroyWebGPU } from "../src/backend/webgpu";
import { getSubmitCount, resetSubmitCount } from "../src/backend/webgpu/webgpu-state";
import { Torchlette } from "../src/frontend/torchlette";
import { DISTILGPT2_CONFIG, GPT2 } from "../examples/gpt2/model";
import type { KVCache } from "../examples/gpt2/model";
import { STEP_TAPE_RECORD } from "../src/core/step-tape";

const STEPS = Number(process.argv[2] ?? 32);
const PROMPT = [40, 716, 257];

interface Timing {
  build: number;
  readback: number;
  hostarg: number;
  markstep: number;
  total: number;
  submits: number;
}
interface ArmResult {
  tapeOn: boolean;
  perTok: Timing; // steady-state (late-half) medians
  tokens: number[];
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

async function run(tapeOn: boolean): Promise<ArmResult> {
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = new GPT2(api, { ...DISTILGPT2_CONFIG });

  let pastKVs: KVCache[] | undefined;
  let posOffset = 0;

  const firstLogits = api.noGrad(() => {
    const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
    const { logits, presentKVs } = model.forwardCached(idx, undefined, 0);
    pastKVs = presentKVs;
    return logits;
  });
  await firstLogits.cpu();
  firstLogits.dispose();
  posOffset = PROMPT.length;
  await api.markStep();

  api.setStepScopedCleanup(true);
  const b: number[] = [], r: number[] = [], h: number[] = [], m: number[] = [], tot: number[] = [];
  const sub: number[] = [];
  const tokens: number[] = [];
  let tok = 1;

  for (let i = 0; i < STEPS; i++) {
    resetSubmitCount();
    const tStart = performance.now();

    // t_build: lazy graph construction (no force)
    const t0 = performance.now();
    const idx = api.tensorFromArray([tok], [1, 1]);
    const { logits, presentKVs } = api.noGrad(() =>
      model.forwardCached(idx, pastKVs, posOffset),
    );
    pastKVs = presentKVs;
    const t1 = performance.now();

    // t_readback: the fence + GPU->CPU round-trip
    const data = new Float32Array(await logits.cpu());
    const t2 = performance.now();

    // t_hostarg: host argmax over the vocab row
    let best = 0;
    for (let v = 1; v < model.config.vocabSize; v++) if (data[v] > data[best]) best = v;
    tok = best;
    tokens.push(best);
    const t3 = performance.now();

    logits.dispose();
    posOffset += 1;

    // t_markstep: boundary bookkeeping
    await api.markStep();
    const t4 = performance.now();

    b.push(t1 - t0);
    r.push(t2 - t1);
    h.push(t3 - t2);
    m.push(t4 - t3);
    tot.push(t4 - tStart);
    sub.push(getSubmitCount());
  }

  const half = Math.floor(STEPS / 2);
  const late = <T>(xs: T[]) => xs.slice(half);
  return {
    tapeOn,
    tokens,
    perTok: {
      build: median(late(b)),
      readback: median(late(r)),
      hostarg: median(late(h)),
      markstep: median(late(m)),
      total: median(late(tot)),
      submits: median(late(sub)),
    },
  };
}

async function runArmInChild(tapeOn: boolean): Promise<ArmResult> {
  const { execFileSync } = await import("node:child_process");
  const env: Record<string, string> = { ...process.env, UKE_CHILD: "1" };
  if (tapeOn) env.TORCHLETTE_STEP_TAPE = "1";
  else delete env.TORCHLETTE_STEP_TAPE;
  const out = execFileSync(process.execPath, ["--import", "tsx", import.meta.filename, String(STEPS)], {
    env,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
    maxBuffer: 64 * 1024 * 1024,
  });
  const line = out.split("\n").find((l) => l.startsWith("=== UKE-ARM === "));
  if (!line) throw new Error(`no result line (tapeOn=${tapeOn})`);
  return JSON.parse(line.slice("=== UKE-ARM === ".length)) as ArmResult;
}

function fmt(t: Timing): string {
  return `build ${t.build.toFixed(2)} | readback ${t.readback.toFixed(2)} | hostarg ${t.hostarg.toFixed(2)} | markstep ${t.markstep.toFixed(2)} | TOTAL ${t.total.toFixed(2)} ms/tok | ${t.submits} submits`;
}

async function main() {
  if (process.env.UKE_CHILD === "1") {
    if (!(await initWebGPU())) {
      console.error(getWebGPUInitError() || "WebGPU init failed");
      process.exit(1);
    }
    const r = await run(STEP_TAPE_RECORD);
    console.log(`=== UKE-ARM === ${JSON.stringify(r)}`);
    destroyWebGPU();
    process.exit(0);
  }

  console.log(`=== PROBE 3: THE ECONOMICS (distilgpt2 growing-KV decode, ${STEPS} steps, late-half medians) ===`);
  const off = await runArmInChild(false);
  const on = await runArmInChild(true);
  const tokensMatch =
    off.tokens.length === on.tokens.length && off.tokens.every((t, i) => t === on.tokens[i]);

  console.log(`\ntape OFF : ${fmt(off.perTok)}`);
  console.log(`tape ON  : ${fmt(on.perTok)}`);
  console.log(`tokens byte-identical OFF vs ON: ${tokensMatch}`);

  // The K-amortized projection, per arm. The host tax that unrolled-K pays
  // once-per-K instead of once-per-token is readback + hostarg + markstep.
  // t_build becomes a once-per-trace compile (amortized ~0 per token at steady
  // state). Only the GPU compute (total - build - readback - hostarg - markstep)
  // and the amortized boundary survive per token.
  for (const arm of [off, on]) {
    const t = arm.perTok;
    const hostTax = t.readback + t.hostarg + t.markstep; // paid once-per-K under unrolled-K
    const gpuAndOther = Math.max(0, t.total - t.build - hostTax);
    console.log(`\n--- projection (tape ${arm.tapeOn ? "ON" : "OFF"}) ---`);
    console.log(`  per-token host tax paid EVERY token today (readback+hostarg+markstep): ${hostTax.toFixed(2)} ms`);
    console.log(`  per-token build (becomes once-per-trace compile): ${t.build.toFixed(2)} ms`);
    for (const K of [4, 8, 16]) {
      // unrolled-K per-token wall ~= gpuAndOther (still per iteration) + hostTax/K (amortized) + ~0 build
      const projected = gpuAndOther + hostTax / K;
      const speedup = t.total / projected;
      console.log(
        `  K=${K}: projected ~${projected.toFixed(2)} ms/tok  (vs ${t.total.toFixed(2)} today)  => ${speedup.toFixed(2)}x  [host tax amortized ${hostTax.toFixed(2)}->${(hostTax / K).toFixed(2)} ms/tok]`,
      );
    }
  }
  console.log(`\n=== UKE-ECON-STATS === ${JSON.stringify({ off: off.perTok, on: on.perTok, tokensMatch })}`);
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
