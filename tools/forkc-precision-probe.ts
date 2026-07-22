/**
 * FORK-C PRECISION BLOCKER PROBE (derived-optimizer-realizer campaign, R3 re-open).
 *
 * Fork C delivers Adam bias-correction `bc=[bc1,bc2]` as a HOST-computed [2] live
 * scalar (host `expm1F32`) instead of computing it in a graph subgraph on GPU. The
 * blocker: the derived==authored parity then compares HOST expm1F32 against the
 * AUTHORED kernel's in-kernel GPU `exp` intrinsic — host-vs-GPU, not GPU-vs-GPU.
 *
 * This probe establishes, across a real trajectory range of t (1..50000) and both
 * betas, whether that comparison is (i) bit-exact in f32 or (ii) bounded by a NAMED
 * host-vs-GPU exp lemma, and PROPAGATES it to Δparam through the ACTUAL Adam kernel.
 *
 * Three measurements, one process:
 *   (A) RAW exp intrinsic: host Math.exp(y) vs GPU exp(y) for y = t·lnβ. Isolates
 *       the only host-vs-GPU divergence source (the Horner branch is pure f32 ops).
 *   (B) bc: host expm1F32 (fround discipline) vs GPU expm1 chain (the graph path
 *       _biasCorrection uses). Per-lane abs/rel + which branch.
 *   (C) PROPAGATED Δparam: the authored Adam kernel (GPU in-kernel bc from t) vs the
 *       derived Adam kernel fed HOST bc — exactly the fork-C parity — across the
 *       t-grid × wd modes. This is the exit-gate number generalized off STEP=7.
 *
 * GPU tool: reserve via tools/pick-gpu.sh. Env: N (param count).
 */
import { initWebGPU, getWebGPUInitError, webgpuBackend } from "../src/backend/webgpu";
import { createTileKernelDispatcher } from "../src/backend/webgpu/tile-dispatch";
import { realizeAdamStepSpec } from "../src/schedule/adam-skeleton";
import type { WebGPUTensor } from "../src/backend/webgpu/gpu-types";

const N = parseInt(process.env.N ?? "4096", 10);
const log = (m: string) => console.error(m);
const f = Math.fround;

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;
const LR = 1e-3;
const LNB1 = f(Math.log(BETA1));
const LNB2 = f(Math.log(BETA2));

// t grid spanning the trajectory: small t (Horner for bc2, exp for bc1),
// crossover for bc2 (t≈250), and the long tail.
const T_GRID = [1, 2, 3, 4, 5, 7, 10, 20, 50, 100, 200, 250, 300, 500, 1000, 5000, 10000, 50000];

/** Host expm1F32 — mirrors adam-skeleton emitExpm1 / the spec-test reference exactly. */
function expm1F32(y: number): number {
  const ay = Math.abs(y);
  if (ay < 0.25) {
    let r = f(1 / 120);
    r = f(f(1 / 24) + f(y * r));
    r = f(f(1 / 6) + f(y * r));
    r = f(f(1 / 2) + f(y * r));
    r = f(1 + f(y * r));
    return f(y * r);
  }
  return f(Math.exp(y) - 1);
}
const bcHost = (t: number, lnb: number): number => f(-expm1F32(f(t * lnb)));
const branch = (t: number, lnb: number): string => (Math.abs(f(t * lnb)) < 0.25 ? "horner" : "exp");

function randData(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

const t2 = (data: number[]): WebGPUTensor =>
  webgpuBackend.ops.tensorFromArray(data, [data.length]) as WebGPUTensor;

function baseUniforms(wd: number, dec: number): Record<string, number> {
  return {
    num_elements: N, beta1: BETA1, beta2: BETA2, ln_beta1: LNB1, ln_beta2: LNB2,
    eps: EPS, weight_decay: wd, decoupled_wd: dec,
    _pad0: 0, _pad1: 0, _pad2: 0, _pad3: 0,
  };
}

// ULP distance between two f32 values (via their bit patterns).
const fbuf = new ArrayBuffer(8);
const fv = new Float32Array(fbuf);
const iv = new Int32Array(fbuf);
function ulpDist(a: number, b: number): number {
  fv[0] = a; let ia = iv[0]!;
  fv[0] = b; let ib = iv[0]!;
  if (ia < 0) ia = 0x80000000 - ia | 0;
  if (ib < 0) ib = 0x80000000 - ib | 0;
  return Math.abs(ia - ib);
}

async function readback(t: WebGPUTensor): Promise<Float32Array> {
  return Float32Array.from(await webgpuBackend.ops.read(t));
}

async function main(): Promise<void> {
  const ready = await initWebGPU();
  if (!ready) throw new Error(`WebGPU init failed: ${getWebGPUInitError()}`);

  // ---- (A) RAW exp intrinsic: host Math.exp(y) vs GPU exp(y) ----
  // Build the full set of y = t·lnβ values across the grid, both betas, run GPU exp once.
  const ys: number[] = [];
  const yMeta: { t: number; beta: string; lnb: number; br: string }[] = [];
  for (const t of T_GRID) {
    ys.push(f(t * LNB1)); yMeta.push({ t, beta: "β1", lnb: LNB1, br: branch(t, LNB1) });
    ys.push(f(t * LNB2)); yMeta.push({ t, beta: "β2", lnb: LNB2, br: branch(t, LNB2) });
  }
  const yT = t2(ys);
  const gpuExp = await readback(webgpuBackend.ops.exp(yT) as WebGPUTensor);
  log(`\n=== (A) RAW exp intrinsic: host Math.exp vs GPU exp(y), y=t·lnβ ===`);
  let worstExpUlp = 0, worstExpRel = 0, worstExpUlpMag = 0;
  const worstExpWhere = { t: 0, beta: "", y: 0, h: 0, g: 0 };
  for (let i = 0; i < ys.length; i++) {
    const h = f(Math.exp(ys[i]!));
    const g = gpuExp[i]!;
    const ulp = ulpDist(h, g);
    // rel only meaningful where host exp is a normal, nonzero number.
    if (h > 1e-30) worstExpRel = Math.max(worstExpRel, Math.abs(h - g) / Math.abs(h));
    if (ulp > worstExpUlp) {
      worstExpUlp = ulp; worstExpUlpMag = Math.max(Math.abs(h), Math.abs(g));
      const m = yMeta[i]!;
      Object.assign(worstExpWhere, { t: m.t, beta: m.beta, y: ys[i]!, h, g });
    }
  }
  log(`  worst exp ULP=${worstExpUlp} at t=${worstExpWhere.t} ${worstExpWhere.beta} y=${worstExpWhere.y.toExponential(3)} (exp≈${worstExpUlpMag.toExponential(3)}, host=${worstExpWhere.h.toExponential(4)} gpu=${worstExpWhere.g.toExponential(4)})`);
  log(`  worst exp rel (where host exp>1e-30)=${worstExpRel.toExponential(3)}  (over ${ys.length} y values)`);

  // ---- (B) bc: host expm1F32 vs GPU expm1 chain (the _biasCorrection graph algorithm) ----
  // Replicate _biasCorrection on GPU element-wise via backend ops over the y vector.
  log(`\n=== (B) bc: host expm1F32 vs GPU expm1 chain (graph _biasCorrection algo) ===`);
  const ops = webgpuBackend.ops;
  const K = ys.length;
  const konst = (s: number) => t2(new Array(K).fill(s));
  const add = (a: WebGPUTensor, b: WebGPUTensor) => ops.add(a, b) as WebGPUTensor;
  const mul = (a: WebGPUTensor, b: WebGPUTensor) => ops.mul(a, b) as WebGPUTensor;
  // Horner on GPU: r=1/120; r=1/24+y*r; ...; series=y*r; large=exp(y)-1; where(|y|<.25, series, large)
  let r: WebGPUTensor = konst(1 / 120);
  r = add(konst(1 / 24), mul(yT, r));
  r = add(konst(1 / 6), mul(yT, r));
  r = add(konst(1 / 2), mul(yT, r));
  r = add(konst(1), mul(yT, r));
  const series = mul(yT, r);
  const large = ops.sub(ops.exp(yT) as WebGPUTensor, konst(1)) as WebGPUTensor;
  const absY = ops.abs(yT) as WebGPUTensor;
  const cond = ops.lt(absY, konst(0.25)) as WebGPUTensor;
  const expm1Gpu = await readback(ops.where(cond, series, large) as WebGPUTensor);
  let worstBcAbs = 0, worstBcRel = 0, worstBcUlp = 0;
  const rows: string[] = [];
  for (let i = 0; i < ys.length; i++) {
    const m = yMeta[i]!;
    const hostBc = f(-expm1F32(ys[i]!));
    const gpuBc = f(-expm1Gpu[i]!);
    const abs = Math.abs(hostBc - gpuBc);
    const rel = abs / Math.abs(hostBc);
    const ulp = ulpDist(hostBc, gpuBc);
    worstBcAbs = Math.max(worstBcAbs, abs);
    worstBcRel = Math.max(worstBcRel, rel);
    worstBcUlp = Math.max(worstBcUlp, ulp);
    if (ulp > 0 && (m.t <= 5 || m.t === 250 || m.t === 300 || m.t >= 10000))
      rows.push(`  t=${String(m.t).padStart(5)} ${m.beta} ${m.br.padEnd(6)} host=${hostBc.toExponential(6)} gpu=${gpuBc.toExponential(6)} ulp=${ulp} rel=${rel.toExponential(2)}`);
  }
  rows.forEach((s) => log(s));
  log(`  worst bc: abs=${worstBcAbs.toExponential(3)} rel=${worstBcRel.toExponential(3)} ulp=${worstBcUlp}`);

  // ---- (C) PROPAGATED Δparam through the ACTUAL Adam kernel across the t-grid ----
  log(`\n=== (C) PROPAGATED Δparam: authored (GPU in-kernel bc from t) vs derived (HOST bc) ===`);
  const grad = Array.from(randData(N, 1));
  const param = Array.from(randData(N, 2));
  const mArr = Array.from(randData(N, 3));
  const vArr = Array.from(randData(N, 4).map((x) => Math.abs(x)));

  const runOne = async (derived: boolean, t: number, wd: number, dec: number) => {
    const spec = realizeAdamStepSpec(false, false, false, derived);
    const disp = createTileKernelDispatcher(spec);
    const g = t2(grad), p = t2(param), mm = t2(mArr), vv = t2(vArr), lr = t2([LR]);
    const buffers: Record<string, WebGPUTensor["buffer"]> = {
      grad: g.buffer, param: p.buffer, m: mm.buffer, v: vv.buffer, lr: lr.buffer,
    };
    if (derived) buffers.bc = t2([bcHost(t, LNB1), bcHost(t, LNB2)]).buffer;
    else buffers.t = t2([t]).buffer;
    disp.dispatch(buffers, baseUniforms(wd, dec));
    return { param: await readback(p), m: await readback(mm), v: await readback(vv) };
  };
  const maxRel = (a: Float32Array, b: Float32Array) => {
    let mx = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i]! - b[i]!);
      mx = Math.max(mx, Math.min(d, d / (Math.abs(a[i]!) + 1e-8)));
    }
    return mx;
  };
  const cells = [
    { name: "no-wd", wd: 0, dec: 0 },
    { name: "L2", wd: 0.1, dec: 0 },
    { name: "AdamW", wd: 0.1, dec: 1 },
  ];
  let worstDp = 0, worstDm = 0, worstDv = 0;
  const worstAt = { dp: 0, cell: "" };
  for (const t of T_GRID) {
    for (const c of cells) {
      const a = await runOne(false, t, c.wd, c.dec);
      const d = await runOne(true, t, c.wd, c.dec);
      const dp = maxRel(a.param, d.param), dm = maxRel(a.m, d.m), dv = maxRel(a.v, d.v);
      if (dp > worstDp) { worstDp = dp; worstAt.dp = t; worstAt.cell = c.name; }
      worstDm = Math.max(worstDm, dm); worstDv = Math.max(worstDv, dv);
      if (t <= 5 || t === 250 || t >= 10000)
        log(`  t=${String(t).padStart(5)} ${c.name.padEnd(6)} Δparam=${dp.toExponential(2)} Δm=${dm.toExponential(2)} Δv=${dv.toExponential(2)}`);
    }
  }
  log(`\n  WORST over t-grid × wd: Δparam=${worstDp.toExponential(3)} (t=${worstAt.dp}, ${worstAt.cell})  Δm=${worstDm.toExponential(3)}  Δv=${worstDv.toExponential(3)}`);
  log(`\n=== VERDICT INPUTS: exp-ulp=${worstExpUlp} bc-ulp=${worstBcUlp} bc-rel=${worstBcRel.toExponential(3)} Δparam=${worstDp.toExponential(3)} ===`);
}

main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
