/**
 * LEDGER ATTACK PROBE (adversarial review of commit 9d9f757a's retain/release
 * ledger — `_retainedInputIds` in src/graph/node-factory.ts).
 *
 * Drives a REAL full-stack inner step (autocast f16 + gradient checkpointing +
 * GradScaler + grad clip + AdamW), random-init (no parity files needed), for
 * STEPS steps. This is the workload that maximally exercises the ledger's
 * attack surface:
 *   - CSE / mul-by-1 / identity-cast bypass (`redirectConsumers`) mutating
 *     node.inputs BETWEEN retainPlanInputRefs and releaseNodeInputRefs;
 *   - template HITS (2nd+ exec) that RE-APPLY those rewrites to fresh nodes
 *     after retain (executor.ts runPasses on the hit path);
 *   - multi-plan steps (backward plan + optimizer plan) that hold cross-plan
 *     grads across the whole step;
 *   - the GradScaler's rc=1 LIVE scale tensor read as node.inputs[1] by the
 *     fused unscaleGrad — the exact storage the phantom-release destroyed.
 *
 * INSTRUMENTS (all detection is passive; we do NOT modify src):
 *   - TORCHLETTE_RC_TRACE=1  -> refcount.ts logs "DOUBLE-RELEASE: <id> @ site"
 *     whenever any rc goes negative. A ledger over-release lands here as
 *     "@ plan.inputConsumed". We parse our own stderr for it.
 *   - A read of a reclaimed storage THROWS by default (task #73;
 *     TORCHLETTE_STRICT_LIFETIME=0 downgrades to warn). A phantom release that
 *     destroys a live input surfaces as a thrown RECLAIMED read (loud, not silent).
 *   - storageTracker.stats() per step -> reachable/total trajectory. A LEDGER
 *     LEAK (retain-without-release: _retainedInputIds pins forever) shows as a
 *     monotonically climbing reachable/total. Flat == no leak.
 *
 * Env: STEPS(=24) NUM_LAYERS(=4) NUM_HEADS(=4) EMBED_DIM(=128) SEQ_LEN(=128)
 *      BATCH_SIZE(=4) LR(=5e-4) GRAD_CLIP(=1.0) USE_AUTOCAST(=1) CHECKPOINT(=1)
 *      USE_SCALER(=1)  (STRICT/RC_TRACE set by the driver, not here)
 *
 * The probe SELF-INTERCEPTS console.warn to count DOUBLE-RELEASE lines so it
 * can PASS/FAIL without a human reading megabytes of trace. It leaves the raw
 * lines on stderr too (for forensics).
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { storageTracker } from "../src/graph/storage-tracker";

const STEPS = parseInt(process.env.STEPS ?? "24", 10);
const L = parseInt(process.env.NUM_LAYERS ?? "4", 10);
const H = parseInt(process.env.NUM_HEADS ?? "4", 10);
const E = parseInt(process.env.EMBED_DIM ?? "128", 10);
const SEQ = parseInt(process.env.SEQ_LEN ?? "128", 10);
const BATCH = parseInt(process.env.BATCH_SIZE ?? "4", 10);
const LR = parseFloat(process.env.LR ?? "5e-4");
const GRAD_CLIP = parseFloat(process.env.GRAD_CLIP ?? "1.0");
const USE_AUTOCAST = process.env.USE_AUTOCAST !== "0";
const CHECKPOINT = process.env.CHECKPOINT !== "0";
const USE_SCALER = process.env.USE_SCALER !== "0";
const VOCAB = 50257;

const log = (m: string) => console.error(`[ledger-attack] ${m}`);

// ---- DOUBLE-RELEASE interception -------------------------------------------
// refcount.ts emits console.warn("[rc] DOUBLE-RELEASE: <id> @ <site> ...").
// Count them (and specifically plan.inputConsumed ones) while still forwarding
// to stderr for forensics.
let doubleReleaseTotal = 0;
let doubleReleasePlan = 0;
const doubleReleaseSamples: string[] = [];
// Per-storage plan-site tally (populated only under RC_TRACE=verbose, which
// logs EVERY retain/release). Lets the probe self-verify the ledger balance
// invariant (retain@plan.input == release@plan.inputConsumed per storage)
// WITHOUT an external grep.
const planRetain = new Map<number, number>();
const planRelease = new Map<number, number>();
const rcLine = /\[rc\] (\d+): (retain|release) (\d+)→(-?\d+) @ (\S+)/;
const origWarn = console.warn.bind(console);
console.warn = (...args: unknown[]) => {
  const s = args.map((a) => String(a)).join(" ");
  if (s.includes("DOUBLE-RELEASE")) {
    doubleReleaseTotal++;
    if (s.includes("plan.inputConsumed")) {
      doubleReleasePlan++;
      if (doubleReleaseSamples.length < 20) doubleReleaseSamples.push(s);
    } else if (doubleReleaseSamples.length < 20) {
      doubleReleaseSamples.push(s);
    }
  }
  const m = rcLine.exec(s);
  if (m) {
    const sid = Number(m[1]);
    const kind = m[2];
    const site = m[5];
    if (site === "plan.input" && kind === "retain")
      planRetain.set(sid, (planRetain.get(sid) ?? 0) + 1);
    if (site === "plan.inputConsumed" && kind === "release")
      planRelease.set(sid, (planRelease.get(sid) ?? 0) + 1);
  }
  origWarn(...args);
};

function randInput() {
  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  for (let i = 0; i < BATCH * SEQ; i++) {
    inp[i] = Math.floor(Math.random() * VOCAB);
    tgt[i] = Math.floor(Math.random() * VOCAB);
  }
  return { inp, tgt };
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "6000";

  const api = new Torchlette("webgpu", { enableFusion: true });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    { vocabSize: VOCAB, blockSize: 1024, numLayers: L, numHeads: H, embedDim: E, dropoutRate: 0 },
    { device: "webgpu" },
  );

  await api.beginStep();
  api.endStep();
  await api.markStep();

  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: LR, weightDecay: 0.01, adamW: true }, api);
  const scaler = USE_SCALER ? new GradScaler(api, { initScale: 1024.0 }) : null;

  log(
    `START steps=${STEPS} L=${L} H=${H} E=${E} seq=${SEQ} batch=${BATCH} ` +
      `autocast=${USE_AUTOCAST} ckpt=${CHECKPOINT} scaler=${USE_SCALER} clip=${GRAD_CLIP} ` +
      `STRICT=${process.env.TORCHLETTE_STRICT_LIFETIME ?? "1 (default)"} ` +
      `RC_TRACE=${process.env.TORCHLETTE_RC_TRACE ?? "0"} ` +
      `TAPE=${process.env.TORCHLETTE_STEP_TAPE ?? "0"} ` +
      `COMPILED=${process.env.TORCHLETTE_COMPILED_PLAN ?? "default"}`,
  );

  const traj: Array<{ step: number; total: number; reachable: number; loss: number }> = [];

  for (let step = 0; step < STEPS; step++) {
    if (scaler) await scaler.resolveDeferred();
    const { inp, tgt } = randInput();

    await api.beginStep();
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], { device: "webgpu" });
    const loss = api.tidy(() => {
      const fwd = () =>
        model.forwardWithLoss(input, target, { useCheckpoint: CHECKPOINT }).loss;
      const l = USE_AUTOCAST ? api.autocast(fwd) : fwd();
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();

    const backwardTarget = scaler ? scaler.scale(loss) : loss;
    await backwardTarget.backward();
    if (scaler) scaler.unscale_(opt);
    if (GRAD_CLIP > 0) clipGradNorm_(api, params, GRAD_CLIP);
    if (scaler) {
      scaler.step(opt);
      scaler.update();
    } else {
      opt.step();
    }
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();

    const st = storageTracker.stats();
    traj.push({ step, total: st.totalStorages, reachable: st.reachableStorages, loss: lossVal });
    // TEMP livediff (task #96 residual attribution; env-gated):
    if (process.env.TORCHLETTE_LIVEDIFF === "1") {
      type LiveEntry = { id: number; shape: readonly number[]; dtype: string; view: boolean; base?: number };
      const g = globalThis as Record<string, unknown>;
      if (step === 8) g.__live8 = storageTracker.debugLiveSet();
      if (step === STEPS - 2 && g.__live8) {
        const a = g.__live8 as LiveEntry[];
        const b = storageTracker.debugLiveSet();
        const aIds = new Set(a.map((x) => x.id));
        const bIds = new Set(b.map((x) => x.id));
        const added = b.filter((x) => !aIds.has(x.id));
        const removed = a.filter((x) => !bIds.has(x.id));
        log(`LIVEDIFF s8->s${step}: +${added.length} -${removed.length} (net ${b.length - a.length})`);
        const hist = (xs: LiveEntry[]) => {
          const m = new Map<string, number>();
          for (const x of xs) {
            const k = `[${x.shape.join(",")}] ${x.dtype}${x.view ? " view" : ""}`;
            m.set(k, (m.get(k) ?? 0) + 1);
          }
          return [...m.entries()].sort((p, q) => q[1] - p[1]);
        };
        for (const [k, v] of hist(added).slice(0, 30)) log(`  LIVEDIFF + ${v}x ${k}`);
        for (const [k, v] of hist(removed).slice(0, 10)) log(`  LIVEDIFF - ${v}x ${k}`);
      }
    }
    if (step < 4 || step % 4 === 0 || step === STEPS - 1) {
      log(
        `step ${step}: loss=${lossVal.toFixed(6)} total=${st.totalStorages} ` +
          `reachable=${st.reachableStorages} dblRel=${doubleReleaseTotal}`,
      );
    }
    if (!Number.isFinite(lossVal)) {
      log(`FATAL: non-finite loss at step ${step} — possible corruption`);
      break;
    }
  }

  // ---- verdict ----
  // Steady-state leak check: compare late-window reachable/total to mid-window.
  // Warmup (template build, pool settle) inflates early steps; take steps>=half.
  const half = Math.floor(STEPS / 2);
  const late = traj.slice(half);
  const reMin = Math.min(...late.map((t) => t.reachable));
  const reMax = Math.max(...late.map((t) => t.reachable));
  const toMin = Math.min(...late.map((t) => t.total));
  const toMax = Math.max(...late.map((t) => t.total));

  log("=== TRAJECTORY (reachable/total per step) ===");
  for (const t of traj) log(`  s${t.step}: reachable=${t.reachable} total=${t.total}`);
  log(
    `=== late-window (steps ${half}..${STEPS - 1}) reachable [${reMin}..${reMax}] ` +
      `total [${toMin}..${toMax}] ===`,
  );
  log(`=== DOUBLE-RELEASE total=${doubleReleaseTotal} (plan.inputConsumed=${doubleReleasePlan}) ===`);
  for (const s of doubleReleaseSamples) log(`   ${s}`);

  // Ledger balance invariant (only meaningful under RC_TRACE=verbose): every
  // storage retained @plan.input is released @plan.inputConsumed exactly as
  // many times. An imbalance is a leak (retain>release) or over-release
  // (release>retain) BY THE LEDGER — the precise property under review.
  const balanceIds = new Set<number>([...planRetain.keys(), ...planRelease.keys()]);
  const imbalanced: Array<[number, number, number]> = [];
  for (const id of balanceIds) {
    const r = planRetain.get(id) ?? 0;
    const e = planRelease.get(id) ?? 0;
    if (r !== e) imbalanced.push([id, r, e]);
  }
  const balanceChecked = balanceIds.size > 0;

  const reachDrift = reMax - reMin;
  const totalDrift = toMax - toMin;
  const LEAK_TOL = 8; // small churn from template settle is fine; monotone climb is not
  let fail = false;
  if (doubleReleasePlan > 0) {
    log(`RESULT: FAIL — ${doubleReleasePlan} plan.inputConsumed DOUBLE-RELEASE (ledger over-release)`);
    fail = true;
  }
  if (balanceChecked) {
    log(`=== LEDGER BALANCE: ${balanceIds.size} plan-site storages, ${imbalanced.length} imbalanced ===`);
    for (const [id, r, e] of imbalanced.slice(0, 20))
      log(`   imbalanced id=${id} retain@plan.input=${r} release@plan.inputConsumed=${e}`);
    if (imbalanced.length > 0) {
      log(`RESULT: FAIL — ${imbalanced.length} plan-site imbalance (ledger leak/over-release)`);
      fail = true;
    }
  }
  if (reachDrift > LEAK_TOL || totalDrift > LEAK_TOL) {
    log(
      `RESULT: SUSPECT — late-window drift reachable=${reachDrift} total=${totalDrift} > tol ${LEAK_TOL} (possible LEDGER LEAK)`,
    );
    fail = true;
  }
  if (!fail) {
    log(
      `RESULT: PASS — no plan over-release; ${balanceChecked ? `ledger balanced (${balanceIds.size} storages)` : "balance-check skipped (need RC_TRACE=verbose)"}; late-window flat (reachDrift=${reachDrift} totalDrift=${totalDrift})`,
    );
  }

  // Machine-parseable verdict (consumed by test/rc-ledger.spec.ts). Emitted on
  // STDOUT (log() uses stderr) so the spec can parse it cleanly.
  console.log(
    JSON.stringify({
      rcLedger: {
        planSiteStorages: balanceIds.size,
        imbalanced: imbalanced.length,
        planDoubleRelease: doubleReleasePlan,
        balanceChecked,
        reachDrift,
        totalDrift,
        lateReachable: [reMin, reMax],
        lateTotal: [toMin, toMax],
        steps: STEPS,
      },
    }),
  );

  // Observed-liveness guardMiss telemetry (task #97 stage-1 baseline symptom):
  // cleanMisses = RecoverableGuardMiss recoveries at the compiled external-slot
  // seam (the count the recorded build's guardMiss net was silently absorbing);
  // claimMisses = misses on a stage-3 B RELEASED pair (a wrong last-reader
  // observation). Both must fall to ZERO once the lowered seam is governed by
  // derived liveness rather than the empirical last-reader guess.
  const { getObservedLivenessStats } = await import(
    "../src/executor/observed-liveness"
  );
  log(
    `=== OBSERVED-LIVENESS STATS: ${JSON.stringify(getObservedLivenessStats())} ===`,
  );

  await destroyWebGPU();
  process.exit(fail ? 2 : 0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  log(`STACK: ${(e as Error).stack}`);
  process.exit(1);
});
