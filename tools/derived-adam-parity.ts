/**
 * R2 differential — the DERIVED fused Adam body vs the AUTHORED one, single
 * dispatch, same inputs (derived-optimizer-realizer campaign, phase R2).
 *
 * The derived body (TORCHLETTE_DERIVED_ADAM path) folds the per-element update
 * from ADAMW_PROGRAM and takes bias correction as a [2] `bc`=[bc1,bc2] DATA input
 * (fork B). The authored body computes bias correction in-kernel from `t`. This
 * script dispatches BOTH over identical initial grad/param/m/v across the variant
 * corpus × wd modes, feeding the derived kernel `bc` computed by the SAME expm1
 * Horner the authored kernel uses in-kernel, and asserts the param/m/v outputs
 * agree.
 *
 * They are NOT byte-identical — two NAMED reassociation lemmas (ADAMW_V_NEW's
 * (g·g)·(1−β2) vs ((1−β2)·g)·g, and ADAMW_SCALED's divide-inside-sqrt vs the
 * authored √bc2-factored step_size). The gate is ≤ TOL (default 1e-6, well inside
 * the design's ≤1e-7 claim on the update magnitude at realistic scales).
 *
 * GPU tool: reserve a device via tools/pick-gpu.sh. Env: N, TOL, STEP (the t value).
 */

import { initWebGPU, getWebGPUInitError, webgpuBackend } from "../src/backend/webgpu";
import { createTileKernelDispatcher } from "../src/backend/webgpu/tile-dispatch";
import { realizeAdamStepSpec } from "../src/schedule/adam-skeleton";
import type { WebGPUTensor } from "../src/backend/webgpu/gpu-types";

const N = parseInt(process.env.N ?? "4096", 10);
const TOL = parseFloat(process.env.TOL ?? "1e-6");
const STEP = parseInt(process.env.STEP ?? "7", 10);

const log = (m: string) => console.error(`[derived-adam] ${m}`);

const BETA1 = 0.9;
const BETA2 = 0.999;
const EPS = 1e-8;
const LR = 1e-3;

/** expm1 via the SAME 5-term Horner the kernel / _biasCorrection use. */
function expm1(y: number): number {
  if (Math.abs(y) < 0.25) {
    let r = 1 / 120;
    r = 1 / 24 + y * r;
    r = 1 / 6 + y * r;
    r = 1 / 2 + y * r;
    r = 1 + y * r;
    return y * r;
  }
  return Math.exp(y) - 1;
}

function randData(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

function baseUniforms(weightDecay: number, decoupledWd: number): Record<string, number> {
  return {
    num_elements: N,
    beta1: BETA1,
    beta2: BETA2,
    ln_beta1: Math.fround(Math.log(BETA1)),
    ln_beta2: Math.fround(Math.log(BETA2)),
    eps: EPS,
    weight_decay: weightDecay,
    decoupled_wd: decoupledWd,
    _pad0: 0,
    _pad1: 0,
    _pad2: 0,
    _pad3: 0,
  };
}

const t2 = (data: Float32Array | number[]): WebGPUTensor =>
  webgpuBackend.ops.tensorFromArray(Array.from(data), [
    Array.isArray(data) ? data.length : data.length,
  ]) as WebGPUTensor;

async function runOne(
  derived: boolean,
  gradD: Float32Array,
  paramD: Float32Array,
  mD: Float32Array,
  vD: Float32Array,
  weightDecay: number,
  decoupledWd: number,
): Promise<{ param: Float32Array; m: Float32Array; v: Float32Array }> {
  const spec = realizeAdamStepSpec(false, false, false, derived);
  const dispatcher = createTileKernelDispatcher(spec);

  // Fresh in-place buffers (each kernel mutates param/m/v).
  const grad = t2(gradD);
  const param = t2(paramD);
  const m = t2(mD);
  const v = t2(vD);
  const lr = t2([LR]);

  const buffers: Record<string, WebGPUTensor["buffer"]> = {
    grad: grad.buffer,
    param: param.buffer,
    m: m.buffer,
    v: v.buffer,
    lr: lr.buffer,
  };
  if (derived) {
    const bc1 = -expm1(STEP * Math.fround(Math.log(BETA1)));
    const bc2 = -expm1(STEP * Math.fround(Math.log(BETA2)));
    buffers.bc = t2([bc1, bc2]).buffer;
  } else {
    buffers.t = t2([STEP]).buffer;
  }

  dispatcher.dispatch(buffers, baseUniforms(weightDecay, decoupledWd));

  return {
    param: Float32Array.from(await webgpuBackend.ops.read(param)),
    m: Float32Array.from(await webgpuBackend.ops.read(m)),
    v: Float32Array.from(await webgpuBackend.ops.read(v)),
  };
}

function maxRel(a: Float32Array, b: Float32Array): number {
  let mx = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]! - b[i]!);
    const r = d / (Math.abs(a[i]!) + 1e-8);
    mx = Math.max(mx, Math.min(d, r));
  }
  return mx;
}

async function main(): Promise<void> {
  const ready = await initWebGPU();
  if (!ready) throw new Error(`WebGPU init failed: ${getWebGPUInitError()}`);

  const grad = randData(N, 1);
  const param = randData(N, 2);
  const m = randData(N, 3);
  // v must stay ≥ 0 (it is a running mean of g²) so sqrt(v) is real.
  const v = randData(N, 4).map((x) => Math.abs(x));

  const cells: { name: string; wd: number; dec: number }[] = [
    { name: "no-wd", wd: 0, dec: 0 },
    { name: "L2 (wd=0.1)", wd: 0.1, dec: 0 },
    { name: "AdamW (wd=0.1)", wd: 0.1, dec: 1 },
  ];

  let allOk = true;
  for (const c of cells) {
    const auth = await runOne(false, grad, param, m, v, c.wd, c.dec);
    const der = await runOne(true, grad, param, m, v, c.wd, c.dec);
    const dp = maxRel(auth.param, der.param);
    const dm = maxRel(auth.m, der.m);
    const dv = maxRel(auth.v, der.v);
    const ok = dp <= TOL && dm <= TOL && dv <= TOL;
    log(
      `${c.name.padEnd(15)} Δparam=${dp.toExponential(2)} Δm=${dm.toExponential(2)} Δv=${dv.toExponential(2)} ${ok ? "✓" : "✗"} (tol=${TOL})`,
    );
    if (!ok) allOk = false;
  }
  log(allOk ? "DERIVED == AUTHORED (within reassociation tol)" : "PARITY FAILURE");
  if (!allOk) process.exit(1);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
