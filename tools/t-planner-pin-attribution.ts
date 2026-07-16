/**
 * PLANNER-PIN ATTRIBUTION PROBE — task #99 phase R0 failing-first gate
 * (docs/arena-recompute-design.md §1 + §5 Phase R0).
 *
 * Splits steady-state resident GPU bytes across three buckets on the distil@512 +
 * selective-checkpointing workload (the #97 config), in two modes:
 *   - arena  : the default arena-ON compiled/planner path (the Phase-3 PASS-B case).
 *   - free   : setBufferArenaDisabled(true) — the b66ead78 bypass (lowered path).
 *
 * Buckets (per §1):
 *   - arena-resident      : sum of GPUBuffer.size over arenaBufferSet
 *                           (the position-indexed per-lowered-plan arena).
 *   - planner-registry    : debugPlannerRegistryStats().materializedMB (the shared
 *                           plannerRegistry's materialized entries).
 *   - current (steady)    : gpuMemoryTracker current allocated bytes.
 *   - other = current - arena-resident - planner-registry (pool/params/optimizer).
 *
 * §1 measured attribution (A100 dw-2-1 device 10, 14 steps, late-step readings):
 *   | mode | current | arena-resident | planner-registry(materialized) |
 *   | arena-ON  | ~4585 MB | ~1.8 MB | ~2834 MB |
 *   | arena-free| ~1798 MB | 0.0 MB  | 0.0 MB   |
 *   The +155% delta is ENTIRELY the planner registry; the arena is empty (reclaimed).
 *
 * === MEASURED THIS COMMIT (A100 dw-2-1 device 10, distil@512 + selective ckpt, STEP_TAPE=record) ===
 *   arena-ON  : current=4584.7MB peak=4756.6MB | arena-resident=1.8MB
 *               planner-registry(materialized)=2833.8MB (total 4864.4, result 1919.4, 602 entries)
 *               other=1749.1MB
 *   arena-free: current=1798.3MB peak=3933.5MB | arena-resident=0MB planner-registry=0MB other=1798.3MB
 *   delta = 2786.4MB (+154.9%) — ENTIRELY the planner registry (matches §1).
 *   ASSERT verdict: FAIL (registry 2833.8MB > 2500MB AND arena 1.8MB < 5MB) — failing-first, as designed.
 *
 * === D3 PEAK MATRIX (2026-07-16, device VULKAN 0, commit 112c9fa4, 3 repeats, reproducible) ===
 * Added MODEL={distil,medium} + globalPeakMB (the true FIT watermark, warmup-inclusive,
 * never-reset). The D3 ruling gates the A/B on PEAK-parity; measured like-for-like:
 *   | config | mode       | current | steadyPeak | globalPeak(FIT) |
 *   | distil | arena-ON   | 4106.5  | 4278.5     | 5824.6–6680.3   |
 *   | distil | arena-free | 1798.3  | 3933.5     | 4414.5–4789.9   |
 *   | medium | arena-ON   | 11084.4 | 11306.7    | 18011.4         |
 *   | medium | arena-free | 6027.2  | 12789.6    | 15690.8         |
 * GATE (arena-ON peak ≤ arena-free peak +5%): distil steady +8.8% FAIL, global +21.6% FAIL;
 * medium steady −11.6% PASS, global +14.8% FAIL. The inversion is NOT robust — the ruling's
 * "arena-ON 4278.5 < arena-free 4790" compared arena-ON STEADY vs arena-free GLOBAL peak
 * (methodology mismatch). D3 STOPPED; the checkpoint bypass is RETAINED. steadyPeak is
 * rock-stable; globalPeak is warmup-arena noise.
 *
 * ASSERT MODE (--assert, run after both modes captured; reads /tmp result files):
 *   FAILS while the planner-registry materialized share exceeds the doc threshold
 *   (registry > 2500 MB AND arena-resident < 5 MB) — the deterministic negative the
 *   R2 fix must collapse. It PASSES only once R2 splits the multi-segment liveness
 *   and the registry footprint drops to within +5% of the arena-free current.
 *
 * Run (solo GPU, one mode per process — Dawn device-chain contention):
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=record \
 *     npx tsx tools/t-planner-pin-attribution.ts arena
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *     npx tsx tools/t-planner-pin-attribution.ts free
 *   npx tsx tools/t-planner-pin-attribution.ts --assert
 */

import * as fs from "node:fs";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { arenaBufferSet } from "../src/backend/webgpu/webgpu-state";
import { debugPlannerRegistryStats } from "../src/executor/compiled-plan";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim/index.ts";

const RESULT_DIR = "/tmp/t-planner-pin-attribution";
const REGISTRY_FAIL_MB = 2500; // §1 threshold: registry > 2.5 GB = FAIL
const ARENA_MAX_MB = 5; // §1 threshold: arena-resident < 5 MB (already reclaimed)

// Model dims. Default distil@512 (the #97 / witness-harvest CELL=checkpoint config);
// MODEL=medium selects gpt2-medium@512 for the D3 peak-parity robustness matrix.
// Both run with selective checkpointing. Result files are namespaced by model so the
// distil `--assert` gate (reads arena.json/free.json) is unaffected by medium runs.
const MODEL = process.env.MODEL === "medium" ? "medium" : "distil";
const DIMS =
  MODEL === "medium" ? { L: 24, H: 16, E: 1024 } : { L: 6, H: 12, E: 768 };
const L = DIMS.L;
const H = DIMS.H;
const E = DIMS.E;
const SEQ = 512;
const STEPS = 14;
const RESET_AT = 9; // reset the peak watermark once pool reuse settles (late-step steady state)
const VOCAB = 50257;
const BATCH = 1;

const mode = process.argv[2] ?? "arena";
const log = (m: string) => console.error(`[pin-attr:${mode}] ${m}`);

function arenaResidentBytes(): number {
  let bytes = 0;
  for (const buf of arenaBufferSet)
    bytes += (buf as { size?: number }).size ?? 0;
  return bytes;
}

function randInput(seq: number) {
  const inp = new Int32Array(BATCH * seq);
  const tgt = new Int32Array(BATCH * seq);
  for (let i = 0; i < BATCH * seq; i++) {
    inp[i] = Math.floor(Math.random() * VOCAB);
    tgt[i] = Math.floor(Math.random() * VOCAB);
  }
  return { inp, tgt };
}

async function runMode(arenaFree: boolean) {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "10000";

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  // Arena-free mode: the b66ead78 bypass — checkpointed plans run lowered, the
  // liveness early-release frees forward activations, the planner registry is
  // never populated (no compiled replay).
  if (arenaFree) api._runtime().setBufferArenaDisabled(true);

  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    {
      vocabSize: VOCAB,
      blockSize: 1024,
      numLayers: L,
      numHeads: H,
      embedDim: E,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );

  await api.beginStep();
  api.endStep();
  await api.markStep();

  model.train(true);
  const params = model.parameters();
  const opt = new Adam(
    params,
    { lr: 5e-4, weightDecay: 0.01, adamW: true },
    api,
  );

  log(
    `START L=${L} H=${H} E=${E} seq=${SEQ} steps=${STEPS} arenaFree=${arenaFree}`,
  );

  let steadyCurrent = 0;
  let steadyPeak = 0;
  let globalPeak = 0; // true FIT watermark: max resident over ALL steps incl. warmup
  let arenaResMB = 0;
  let registryMB = 0;
  const losses: number[] = [];

  for (let step = 0; step < STEPS; step++) {
    const { inp, tgt } = randInput(SEQ);
    await api.beginStep();
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    const loss = api.tidy(() => {
      const l = model.forwardWithLoss(input, target, {
        useCheckpoint: true,
        selectiveCheckpoint: true,
      }).loss;
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
    losses.push(lossVal);
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();

    // Track the true FIT watermark (never reset) — the max resident over the whole
    // run, warmup included. This is the "peak = FIT" quantity the D3 ruling gates on.
    globalPeak = Math.max(globalPeak, gpuMemoryTracker.getPeakUsageBytes());
    if (step === RESET_AT) {
      // @ts-expect-error test-only reset of the peak watermark
      gpuMemoryTracker.peakUsageBytes =
        gpuMemoryTracker.getCurrentAllocatedBytes();
    }
    if (step >= RESET_AT) {
      // Late-step steady-state readings (§1: arena-reclaim + pool reuse settled).
      const cur = gpuMemoryTracker.getCurrentAllocatedBytes();
      steadyCurrent = cur; // last reading = settled current
      steadyPeak = Math.max(steadyPeak, gpuMemoryTracker.getPeakUsageBytes());
      arenaResMB = +(arenaResidentBytes() / 1e6).toFixed(1);
      registryMB = debugPlannerRegistryStats().materializedMB;
    }
    if (step < 4 || step >= STEPS - 2)
      log(`step ${step}: loss=${lossVal.toFixed(4)}`);
  }

  const currentMB = +(steadyCurrent / 1e6).toFixed(1);
  const peakMB = +(steadyPeak / 1e6).toFixed(1);
  const globalPeakMB = +(globalPeak / 1e6).toFixed(1);
  const otherMB = +(currentMB - arenaResMB - registryMB).toFixed(1);
  const reg = debugPlannerRegistryStats();

  const result = {
    mode,
    model: MODEL,
    arenaFree,
    currentMB,
    peakMB,
    globalPeakMB,
    arenaResidentMB: arenaResMB,
    plannerRegistryMaterializedMB: registryMB,
    plannerRegistryTotalMB: reg.totalMB,
    plannerRegistryResultMB: reg.resultMB,
    plannerRegistryEntries: reg.entries,
    otherMB,
    firstLoss: +losses[0].toFixed(4),
    lastLoss: +losses[losses.length - 1].toFixed(4),
    allFinite: losses.every((l) => Number.isFinite(l)),
  };

  log("=== ATTRIBUTION ===");
  log(
    `current=${currentMB}MB steadyPeak=${peakMB}MB globalPeak=${globalPeakMB}MB | arena-resident=${arenaResMB}MB ` +
      `planner-registry(materialized)=${registryMB}MB (total ${reg.totalMB}, result ${reg.resultMB}, ${reg.entries} entries) | other=${otherMB}MB`,
  );

  fs.mkdirSync(RESULT_DIR, { recursive: true });
  const suffix = MODEL === "distil" ? "" : `-${MODEL}`;
  fs.writeFileSync(
    `${RESULT_DIR}/${mode}${suffix}.json`,
    JSON.stringify(result, null, 2),
  );
  process.stdout.write(`${JSON.stringify(result)}\n`);

  destroyWebGPU();
  process.exit(0);
}

function assertGate() {
  // Read both mode results; assert the §1 negative (registry-resident +155%).
  const arenaPath = `${RESULT_DIR}/arena.json`;
  const freePath = `${RESULT_DIR}/free.json`;
  if (!fs.existsSync(arenaPath) || !fs.existsSync(freePath)) {
    console.error(
      `[pin-attr:assert] FAIL: missing result files — run 'arena' and 'free' modes first (${arenaPath}, ${freePath})`,
    );
    process.exit(1);
  }
  const a = JSON.parse(fs.readFileSync(arenaPath, "utf8"));
  const f = JSON.parse(fs.readFileSync(freePath, "utf8"));
  const registryMB = a.plannerRegistryMaterializedMB as number;
  const arenaMB = a.arenaResidentMB as number;
  const arenaCur = a.currentMB as number;
  const freeCur = f.currentMB as number;
  const deltaMB = +(arenaCur - freeCur).toFixed(1);
  const deltaPct = +(((arenaCur - freeCur) / freeCur) * 100).toFixed(1);

  console.error(`[pin-attr:assert] === §1 ATTRIBUTION VERDICT ===`);
  console.error(
    `[pin-attr:assert] arena-ON:  current=${arenaCur}MB arena-resident=${arenaMB}MB planner-registry=${registryMB}MB`,
  );
  console.error(
    `[pin-attr:assert] arena-free: current=${freeCur}MB arena-resident=${f.arenaResidentMB}MB planner-registry=${f.plannerRegistryMaterializedMB}MB`,
  );
  console.error(
    `[pin-attr:assert] delta (arena-ON − arena-free) = ${deltaMB}MB (+${deltaPct}%) — the bytes the R2 fix must collapse`,
  );

  // The failing-first assertion: the +155% is registry-resident (registry big,
  // arena empty). PASSES only when R2 collapses the registry pin.
  const registryPinned =
    registryMB > REGISTRY_FAIL_MB && arenaMB < ARENA_MAX_MB;
  if (registryPinned) {
    console.error(
      `[pin-attr:assert] FAIL (EXPECTED PRE-R2): planner-registry ${registryMB}MB > ${REGISTRY_FAIL_MB}MB ` +
        `AND arena-resident ${arenaMB}MB < ${ARENA_MAX_MB}MB — the whole-step RESULT pin is intact. ` +
        `This is the deterministic negative the R2 multi-segment liveness fix erases.`,
    );
    process.exit(1);
  }
  console.error(
    `[pin-attr:assert] PASS: planner-registry ${registryMB}MB collapsed (≤ ${REGISTRY_FAIL_MB}MB or arena now holds it) — R2 landed.`,
  );
  process.exit(0);
}

async function main() {
  if (mode === "--assert") {
    assertGate();
    return;
  }
  if (mode !== "arena" && mode !== "free") {
    console.error(
      `[pin-attr] unknown mode '${mode}'; choices: arena | free | --assert`,
    );
    process.exit(1);
  }
  if (mode === "arena" && process.env.TORCHLETTE_STEP_TAPE !== "record") {
    log(
      "NOTE: arena mode wants TORCHLETTE_STEP_TAPE=record to exercise the compiled/planner path",
    );
  }
  await runMode(mode === "free");
}

main().catch((e) => {
  console.error(`[pin-attr:${mode}] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
