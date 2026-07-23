/**
 * COMPOSITE-CLOSURE F2 — the `Expr → tile-IR` FOLD parity guard (the SURVIVING
 * gate after A3 deleted the hand WGSL bodies). Both `lowerExprToTileIR` (the GPU
 * fold) and `evalScalar` (the CPU reference interpreter, interpret.ts) are
 * derivations of ONE `Expr` definition, so the GPU-realized activation must agree
 * with its CPU meaning — the "single source at seams; assert agreement" principle
 * applied to the forward activation body. For each activation this script
 * dispatches the fold kernel on the GPU and compares against the CPU `evalScalar`
 * of the SAME `Expr`, ULP-bounded across the CPU↔GPU transcendental seam.
 *
 * (History: at A1/A2 this probe compared the fold against the HAND `applyFusedOp`
 * body and MEASURED byte-identity for relu/sigmoid/silu/softplus/gelu_tanh/erf
 * and 1 f32 ULP for gelu_erf — the named L-EXPR reassociation lemma. After A3 the
 * hand bodies are gone; the CPU interpreter is the enduring reference.)
 *
 * GPU tool: reserve a device via tools/pick-gpu.sh, run serial-exclusive. Env:
 *   N (default 8192), TOL (default 1e-5, the CPU↔GPU seam), LO/HI (range -12..12).
 */

import {
  getWebGPUInitError,
  initWebGPU,
  webgpuBackend,
} from "../src/backend/webgpu";
import { lowerExprToTileIR } from "../src/backend/webgpu/expr-tile-fold";
import type { WebGPUTensor } from "../src/backend/webgpu/gpu-types";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import { createTileKernelDispatcher } from "../src/backend/webgpu/tile-dispatch";
import type {
  KernelContext,
  TileKernelSpec,
} from "../src/backend/webgpu/tile-ir";
import { SOFTPLUS_DEF, UNARY_DEFS } from "../src/ops/semantic/catalog";
import { GELU_ERF_DEF, GELU_TANH_DEF } from "../src/ops/semantic/composite";
import { type Expr, erf, x as X } from "../src/ops/semantic/expr";
import { evalScalar } from "../src/ops/semantic/interpret";

const N = parseInt(process.env.N ?? "8192", 10);
const TOL = parseFloat(process.env.TOL ?? "1e-5");
const LO = parseFloat(process.env.LO ?? "-12");
const HI = parseFloat(process.env.HI ?? "12");

const log = (m: string) => console.error(`[expr-fold-parity] ${m}`);

const byName = (n: string): Expr => {
  const d = UNARY_DEFS.find((u) => u.name === n);
  if (!d) throw new Error(`no UNARY_DEF '${n}'`);
  return d.expr;
};

const CELLS: { name: string; expr: Expr }[] = [
  { name: "relu", expr: byName("relu") },
  { name: "sigmoid", expr: byName("sigmoid") },
  { name: "silu", expr: byName("silu") },
  { name: "softplus", expr: SOFTPLUS_DEF.expr },
  { name: "gelu_tanh", expr: GELU_TANH_DEF.expr },
  { name: "gelu_erf", expr: GELU_ERF_DEF.expr },
  { name: "erf", expr: erf(X) },
];

function randData(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * (HI - LO) + LO;
  }
  return out;
}

async function checkCell(cell: {
  name: string;
  expr: Expr;
}): Promise<{ ok: boolean; max: number }> {
  const WG = 64;
  const spec: TileKernelSpec = {
    name: `foldparity_${cell.name}`,
    workgroupSize: WG,
    bindings: {
      x: { storage: "read", type: "f32" },
      out_fold: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: (u: Record<string, number>) => [Math.ceil((u.size ?? 0) / WG)],
    kernel(ctx: KernelContext) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      const xv = ctx.load("x", idx);
      ctx.emitStore(
        "out_fold",
        idx,
        lowerExprToTileIR(cell.expr, ctx, { x: xv }),
      );
    },
  };
  compileTileKernel(spec);
  const dispatcher = createTileKernelDispatcher(spec);

  const xData = randData(N, 0x1234 ^ cell.name.length);
  const xBuf = webgpuBackend.ops.tensorFromArray(Array.from(xData), [
    N,
  ]) as WebGPUTensor;
  const foldBuf = webgpuBackend.ops.tensorFromArray(new Array(N).fill(0), [
    N,
  ]) as WebGPUTensor;
  dispatcher.dispatch(
    { x: xBuf.buffer, out_fold: foldBuf.buffer },
    { size: N },
  );
  const fold = await webgpuBackend.ops.read(foldBuf);

  // CPU reference: evalScalar of the SAME Expr (the single meaning), f32-rounded.
  const env = { x: 0, y: 0, g: 0 };
  let max = 0;
  for (let i = 0; i < N; i++) {
    env.x = xData[i];
    const ref = Math.fround(evalScalar(cell.expr, env));
    const d = Math.abs(fold[i] - ref);
    const rel = d / (Math.abs(ref) + 1e-6);
    max = Math.max(max, Math.min(d, rel));
  }
  return { ok: max <= TOL, max };
}

async function main(): Promise<void> {
  const ready = await initWebGPU();
  if (!ready) throw new Error(`WebGPU init failed: ${getWebGPUInitError()}`);

  let allOk = true;
  for (const cell of CELLS) {
    const { ok, max } = await checkCell(cell);
    log(
      `${cell.name.padEnd(10)} fold-GPU vs interpret-CPU max=${max.toExponential(3)} ${ok ? "PASS" : "FAIL"} (tol=${TOL})`,
    );
    if (!ok) allOk = false;
  }
  log(allOk ? "ALL ACTIVATIONS PASS" : "PARITY FAILURE");
  if (!allOk) process.exit(1);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
