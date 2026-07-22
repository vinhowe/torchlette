/**
 * FORK-C PRECISION BLOCKER PROBE (derived-optimizer-realizer campaign, R3 re-open).
 *
 * Fork C delivers Adam bias-correction `bc=[bc1,bc2]` as a HOST-computed [2] live
 * scalar (host `expm1F32`) instead of computing it in a graph subgraph on GPU. The
 * blocker: the derived==authored parity then compares HOST expm1F32 against the
 * AUTHORED kernel's in-kernel GPU `exp` intrinsic — host-vs-GPU, not GPU-vs-GPU.
 *
 * This probe establishes the NAMED L3 lemma (host-vs-GPU exp, bc-absorbed) across
 * a real trajectory range of t (1..50000) and both betas:
 *   (A) RAW exp intrinsic: host Math.exp(y) vs GPU exp(y) for y = t·lnβ. Isolates
 *       the only host-vs-GPU divergence source (the Horner branch is pure f32 ops).
 *   (B) bc: host expm1F32 (fround discipline) vs GPU expm1 chain — per-lane
 *       abs/rel/ULP + which branch. Verdict: bc agrees to ≤1 ULP everywhere.
 * (The pre-R4 propagated-Δparam measurement — authored GPU-bc vs derived host-bc,
 * worst 5.96e-8, Δm=Δv bit-exact — is retired with the authored path; the surviving
 * numeric guard is tools/optterm-fold-parity.ts.)
 *
 * GPU tool: reserve via tools/pick-gpu.sh. Env: N (param count).
 */
import { initWebGPU, getWebGPUInitError, webgpuBackend } from "../src/backend/webgpu";
import type { WebGPUTensor } from "../src/backend/webgpu/gpu-types";

const N = parseInt(process.env.N ?? "4096", 10);
const log = (m: string) => console.error(m);
const f = Math.fround;

const BETA1 = 0.9;
const BETA2 = 0.999;
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
const branch = (t: number, lnb: number): string => (Math.abs(f(t * lnb)) < 0.25 ? "horner" : "exp");


const t2 = (data: number[]): WebGPUTensor =>
  webgpuBackend.ops.tensorFromArray(data, [data.length]) as WebGPUTensor;


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

  // NOTE (R4): the propagated Δparam through the Adam kernel (authored GPU-bc vs
  // derived host-bc) was measured pre-R4 at worst 5.96e-8 (Δm=Δv bit-exact) — the
  // authored path is now deleted, so this probe retains the load-bearing L3-lemma
  // evidence (A) raw exp + (B) bc host-vs-GPU. The surviving numeric guard for the
  // derived kernel is tools/optterm-fold-parity.ts (fold == program interpreter).
  log(`\n=== VERDICT INPUTS (L3 lemma): exp-ulp=${worstExpUlp} bc-ulp=${worstBcUlp} bc-abs=${worstBcAbs.toExponential(3)} bc-rel=${worstBcRel.toExponential(3)} ===`);
}

main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
