/**
 * CRYSTAL CAMPAIGN 3 — GPU leg of the derive-the-reference probes.
 *
 * Confirms that the DEFINITION interpreter (formula-as-data, evaluated on the
 * host in f64→f32) agrees with what the REAL WebGPU kernels actually compute,
 * for a representative slice of the elementwise family + one composite (RMSNorm).
 * This is the "byte-check on GPU" leg: it surfaces the CPU-vs-GPU numeric
 * divergence class (WGSL transcendentals / inverseSqrt / the pow precedent)
 * that a host-only probe cannot see.
 *
 *   forward:  api.<op>(x)         vs  definition-interpreted reference
 *   backward: input.grad          vs  adjoint-derived VJP  AND  table VJP
 *   composite: api.rmsnorm(x,w)   vs  primitive composition reference
 */
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const f32 = (v: number) => Math.fround(v);
const F = new Float32Array(1);
const U = new Uint32Array(F.buffer);
const bitsEq = (a: number, b: number) => {
  F[0] = a;
  const x = U[0];
  F[0] = b;
  return x === U[0];
};

interface Row {
  label: string;
  n: number;
  byteExact: number;
  maxAbs: number;
}
function compare(label: string, gpu: number[], ref: number[]): Row {
  let byteExact = 0,
    maxAbs = 0;
  for (let i = 0; i < gpu.length; i++) {
    const a = f32(gpu[i]);
    const b = f32(ref[i]);
    if (bitsEq(a, b)) byteExact++;
    if (Number.isFinite(a) && Number.isFinite(b)) maxAbs = Math.max(maxAbs, Math.abs(a - b));
  }
  return { label, n: gpu.length, byteExact, maxAbs };
}
const show = (r: Row) =>
  console.log(
    `  ${r.byteExact === r.n ? "EXACT  " : r.maxAbs < 1e-6 ? "~fp32  " : "DIVERGE"} ${r.label.padEnd(16)} ${r.byteExact}/${r.n} byte-exact  maxAbs=${r.maxAbs.toExponential(2)}`,
  );

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true, enableMemoryPlanning: true });

  const N = 256;
  const xs: number[] = [];
  let seed = 7;
  const rng = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < N; i++) xs.push((rng() - 0.5) * 10); // [-5,5]
  const xsPos = xs.map((v) => Math.abs(v) + 0.05); // positive domain for log/sqrt

  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

  // ---- forward: definition reference vs GPU kernel ----
  console.log("=".repeat(74));
  console.log("GPU FORWARD — definition reference vs real WebGPU kernel");
  console.log("=".repeat(74));
  const fwdCases: [string, number[], (x: number) => number, (t: any) => any][] = [
    ["sigmoid", xs, sigmoid, (t) => api.sigmoid(t)],
    ["silu", xs, (x) => x * sigmoid(x), (t) => api.silu(t)],
    ["tanh", xs, Math.tanh, (t) => api.tanh(t)],
    ["exp", xs, Math.exp, (t) => api.exp(t)],
    ["log", xsPos, Math.log, (t) => api.log(t)],
    ["sqrt", xsPos, Math.sqrt, (t) => api.sqrt(t)],
    ["rsqrt", xsPos, (x) => 1 / Math.sqrt(x), (t) => api.rsqrt(t)],
  ];
  for (const [name, input, ref, op] of fwdCases) {
    const t = api.tensorFromArray(input, [N], { device: "webgpu" });
    const out = op(t);
    const gpu = Array.from(await out.cpu());
    show(compare(name, gpu, input.map(ref)));
  }

  // ---- backward: input.grad vs adjoint-derived VJP and table VJP ----
  console.log("");
  console.log("=".repeat(74));
  console.log("GPU BACKWARD — engine grad vs adjoint-derived VJP vs table VJP");
  console.log("=".repeat(74));
  const G = xs.map((_, i) => (i % 3) - 1 + 0.5); // upstream grad seed
  async function gpuGrad(input: number[], op: (t: any) => any): Promise<number[]> {
    await api.beginStep();
    const t = api.tensorFromArray(input, [N], { device: "webgpu", requiresGrad: true });
    const gT = api.tensorFromArray(G, [N], { device: "webgpu" });
    const out = op(t);
    const loss = api.sum(api.mul(out, gT)) as any; // d loss/d t = G ⊙ dOut/dt
    await loss.backward();
    const grad = Array.from(await t.grad!.cpu());
    api.endStep();
    await api.markStep();
    return grad;
  }
  // adjoint-derived (canonical, no epsilon): sigmoid' = s(1-s); tanh' = 1-t²; exp'=exp; log'=1/x; sqrt'=0.5/√x
  const adjoint: Record<string, (x: number, i: number) => number> = {
    sigmoid: (x, i) => G[i] * sigmoid(x) * (1 - sigmoid(x)),
    tanh: (x, i) => G[i] * (1 - Math.tanh(x) ** 2),
    exp: (x, i) => G[i] * Math.exp(x),
    log: (x, i) => G[i] * (1 / x),
    sqrt: (x, i) => G[i] * (0.5 / Math.sqrt(x)),
  };
  // table VJP (as written in registry.ts — WITH the epsilon guards)
  const table: Record<string, (x: number, i: number) => number> = {
    sigmoid: adjoint.sigmoid,
    tanh: adjoint.tanh,
    exp: adjoint.exp,
    log: (x, i) => G[i] / (x + 1e-8),
    sqrt: (x, i) => G[i] * (0.5 / (Math.sqrt(x) + 1e-8)),
  };
  const bwdCases: [string, number[], (t: any) => any][] = [
    ["sigmoid", xs, (t) => api.sigmoid(t)],
    ["tanh", xs, (t) => api.tanh(t)],
    ["exp", xs, (t) => api.exp(t)],
    ["log", xsPos, (t) => api.log(t)],
    ["sqrt", xsPos, (t) => api.sqrt(t)],
  ];
  for (const [name, input, op] of bwdCases) {
    const grad = await gpuGrad(input, op);
    show(compare(`${name} vs adjoint`, grad, input.map((x, i) => adjoint[name](x, i))));
    show(compare(`${name} vs table`, grad, input.map((x, i) => table[name](x, i))));
  }

  // ---- composite: RMSNorm GPU kernel vs primitive composition ----
  console.log("");
  console.log("=".repeat(74));
  console.log("GPU COMPOSITE — api.rmsnorm vs primitive composition reference");
  console.log("=".repeat(74));
  {
    const B = 8,
      D = 64,
      eps = 1e-5;
    const xv: number[] = [];
    for (let i = 0; i < B * D; i++) xv.push((rng() - 0.5) * 8);
    const w: number[] = [];
    for (let i = 0; i < D; i++) w.push((rng() - 0.5) * 2);
    const t = api.tensorFromArray(xv, [B, D], { device: "webgpu" });
    const wt = api.tensorFromArray(w, [D], { device: "webgpu" });
    const out = Array.from(await api.rmsnorm(t, wt, eps).cpu());
    // composition reference (primitives: mul, mean, add, rsqrt, mul)
    const ref: number[] = [];
    for (let b = 0; b < B; b++) {
      let ss = 0;
      for (let d = 0; d < D; d++) ss += xv[b * D + d] * xv[b * D + d];
      const r = 1 / Math.sqrt(ss / D + eps);
      for (let d = 0; d < D; d++) ref.push(f32(xv[b * D + d] * r * w[d]));
    }
    show(compare("rmsnorm", out, ref));
  }

  console.log("\ngpu probe complete.");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
