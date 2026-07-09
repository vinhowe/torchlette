/**
 * Adam trajectory probe: trains a small param vector for a few steps and
 * prints the per-step param values as JSON. One engine per process — the
 * framework's module-global state (template cache, params-sequence cache,
 * shared-encoder state) makes multiple Torchlette instances in one process
 * interfere, so trajectory comparisons must isolate per process (same
 * methodology as tools/parity-fullstack-tl.ts).
 *
 * Used by test/optim/fused-vs-elementwise.spec.ts and as a CLI probe for
 * optimizer work (docs/architecture-debt.md stage 0/1).
 *
 * Env: ELEMENTWISE=1 (force pure-graph path), ADAMW=0/1, WD (weight decay),
 *      COMPILED=0 (disable compiled plan), FUSION=0 (disable fusion),
 *      STEPS (default 6), N (default 64), LR (default 1e-2),
 *      LR2 + LR2_AT (switch LR to LR2 from step LR2_AT — late-varying
 *      schedule probe: exercises inlined-scalar staleness detection after
 *      the compiled plan has already recorded).
 *
 *      TWO_GROUPS=1 (inc-2a Gate 4, MULTI-GROUP): build the optimizer with
 *      TWO AdamParamGroups. Params 0..k-1 in group 0 (lr=LR, wd=WD); params
 *      k.. in group 1 (lr=LR_G1, wd=WD_G1). Requires NPARAMS >= 2. The two
 *      groups have DIFFERENT lr (and wd) so the packed grouping key — which
 *      must key on lr-tensor identity — is exercised: if group 0's lr were
 *      silently applied to group 1's params (wrong key), this trajectory
 *      diverges from the sequential per-group reference. All params are the
 *      SAME element count on purpose (so a numElements-only packing key would
 *      wrongly merge the two groups).
 */

import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, CosineAnnealingLR, SGD } from "../src/optim";

const N = parseInt(process.env.N ?? "64", 10);
const STEPS = parseInt(process.env.STEPS ?? "6", 10);
const LR = parseFloat(process.env.LR ?? "1e-2");
const WD = parseFloat(process.env.WD ?? "0");
const ADAMW = process.env.ADAMW !== "0";

if (process.env.ELEMENTWISE === "1") process.env.TORCHLETTE_FUSED_ADAM = "0";
if (process.env.FOREACH === "0") process.env.TORCHLETTE_FOREACH_ADAM = "0";
if (process.env.COMPILED === "0") process.env.TORCHLETTE_COMPILED_PLAN = "0";

/** Deterministic pseudo-random init (reproducible across processes). */
function initData(seed: number): number[] {
  const out: number[] = [];
  let x = seed;
  for (let i = 0; i < N; i++) {
    x = (x * 1103515245 + 12345) % 2147483648;
    out.push(((x / 2147483648) * 2 - 1) * 0.5);
  }
  return out;
}

async function main() {
  if (!(await initWebGPU())) {
    console.error("WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", {
    enableFusion: process.env.FUSION !== "0",
  });
  // NPARAMS > 1 splits the same N elements across multiple differently-sized
  // params (exercises real packing: cat/narrow/copy_-back in foreach).
  // The CONCATENATED trajectory is identical to the single-param run, so
  // outputs are comparable across NPARAMS settings.
  const nParams = parseInt(process.env.NPARAMS ?? "1", 10);
  const twoGroups = process.env.TWO_GROUPS === "1";
  const pData = initData(7);
  const tData = initData(99);
  const splits: number[] = [];
  if (twoGroups) {
    // Equal-sized params (same element count) so a numElements-only packing
    // key would WRONGLY merge the two groups — the Gate-4 failure mode.
    const per = Math.floor(N / nParams);
    let rest = N;
    for (let i = 0; i < nParams; i++) {
      const take = i === nParams - 1 ? rest : per;
      splits.push(take);
      rest -= take;
    }
  } else {
    let rest = N;
    for (let i = 0; i < nParams; i++) {
      const take = i === nParams - 1 ? rest : Math.floor(rest / 2) || 1;
      splits.push(take);
      rest -= take;
    }
  }
  const params = [];
  const targets = [];
  let off = 0;
  for (const len of splits) {
    params.push(
      api.tensorFromArray(pData.slice(off, off + len), [len], {
        device: "webgpu",
        requiresGrad: true,
      }),
    );
    targets.push(
      api.tensorFromArray(tData.slice(off, off + len), [len], {
        device: "webgpu",
      }),
    );
    off += len;
  }
  // SGD=1: drive SGD instead of Adam (MOMENTUM env, default 0.9) — same
  // trajectory contract, same LR2/LR2_AT schedule knobs.
  let opt: Adam | SGD;
  if (process.env.SGD === "1") {
    opt = new SGD(
      params,
      {
        lr: LR,
        weightDecay: WD,
        momentum: parseFloat(process.env.MOMENTUM ?? "0.9"),
      },
      api,
    );
  } else if (twoGroups) {
    // inc-2a Gate 4: TWO param groups with different LR + wd. Split the params
    // roughly in half between the groups.
    const lrG1 = parseFloat(process.env.LR_G1 ?? "1e-3");
    const wdG1 = parseFloat(process.env.WD_G1 ?? "0");
    const half = Math.ceil(params.length / 2);
    opt = new Adam(
      [
        { params: params.slice(0, half), lr: LR, weightDecay: WD },
        { params: params.slice(half), lr: lrG1, weightDecay: wdG1 },
      ],
      { lr: LR, weightDecay: WD, adamW: ADAMW },
      api,
    );
  } else {
    opt = new Adam(params, { lr: LR, weightDecay: WD, adamW: ADAMW }, api);
  }

  const lr2 = process.env.LR2 ? parseFloat(process.env.LR2) : null;
  const lr2At = parseInt(process.env.LR2_AT ?? "4", 10);

  // COSINE=1 (inc-2a Gate 5): drive a CosineAnnealingLR schedule. Each step's
  // lr flows through opt.setLR → the persistent lr tensor; a compiled replay
  // must read the new value as DATA (not freeze it) — the LR-schedule
  // exactness seam.
  const cosine =
    process.env.COSINE === "1" && !(opt instanceof SGD)
      ? new CosineAnnealingLR(
          opt as Adam,
          parseInt(process.env.COSINE_TMAX ?? String(STEPS), 10),
          parseFloat(process.env.COSINE_ETAMIN ?? "1e-4"),
        )
      : null;

  const trajectory: number[][] = [];
  for (let step = 0; step < STEPS; step++) {
    if (lr2 !== null && step === lr2At) opt.setGroupLR(0, lr2);
    await api.beginStep();
    // L = sum((p - target)^2) → dL/dp = 2(p - target); grads evolve with p.
    let loss = null;
    for (let i = 0; i < params.length; i++) {
      const diff = api.sub(params[i], targets[i]);
      const partial = api.sum(api.mul(diff, diff));
      loss = loss ? api.add(loss, partial) : partial;
    }
    await loss!.backward();
    opt.step();
    // Cosine schedule: advance the lr for the NEXT step (writes the lr tensor).
    if (cosine) cosine.step();
    // ZEROGRAD_AFTER=1: defer zeroGrad until after markStep — probes whether
    // disposing grads BEFORE their lazy consumers (the optimizer chain)
    // execute is what corrupts state.
    if (process.env.ZEROGRAD_AFTER !== "1") opt.zeroGrad();
    api.endStep();
    await api.markStep();
    if (process.env.ZEROGRAD_AFTER === "1") opt.zeroGrad();
    if (process.env.ADAM_STATE_DUMP === "1") {
      const { _debugAdamState } = await import("../src/optim/adam");
      const rt = api._runtime();
      for (let i = 0; i < Math.min(2, params.length); i++) {
        const st = _debugAdamState(opt, i);
        const m = Array.from(await rt.cpu(st.m)).slice(0, 2);
        const v = Array.from(await rt.cpu(st.v)).slice(0, 2);
        console.error(
          `[state] step=${step} i=${i} m=${m.map((x) => Number(x).toFixed(6))} v=${v.map((x) => Number(x).toFixed(8))}`,
        );
      }
    }
    const stepVals: number[] = [];
    for (const p of params) stepVals.push(...Array.from(await p.cpu()));
    trajectory.push(stepVals);
  }

  // Write + exit in the flush callback: process.exit() truncates pending
  // pipe writes at 8KB, and Dawn's background threads prevent a natural exit.
  process.stdout.write(`${JSON.stringify(trajectory)}\n`, () => {
    process.exit(0);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
