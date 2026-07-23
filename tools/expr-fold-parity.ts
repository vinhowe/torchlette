/**
 * COMPOSITE-CLOSURE F2 / A1 — the `Expr → tile-IR` FOLD vs the HAND activation
 * WGSL body (`applyFusedOp` / `BlockExpr`). The load-bearing A1 claim: the fold
 * (`lowerExprToTileIR`) and the hand DSL bodies are two derivations of ONE `Expr`
 * definition, so they must AGREE. For each activation this script builds ONE
 * kernel that computes BOTH the fold output and the hand output over the SAME
 * random `x`, dispatches on the GPU, and asserts:
 *
 *   byte-where-identical (delta 0)  — relu/sigmoid/silu/softplus/gelu_tanh/erf
 *                                     (identical algebra → identical WGSL)
 *   L-EXPR reassociation lemma      — gelu_erf: the fold computes erf(x·c) (erf
 *                                     takes its own abs → abs(x·c)) while the hand
 *                                     body folds abs into the scale (abs(x)·c);
 *                                     equal in exact arithmetic, MEASURED 1.19e-7
 *                                     = 1 f32 ULP. This is the one named
 *                                     fp-reassociation delta (L1/L2/L3 discipline).
 *   gelu clamp-drop safe            — gelu_tanh: fold (no clamp) vs hand (clamp
 *                                     ±10) is BYTE-IDENTICAL even over the wide
 *                                     range: tanh saturates identically where the
 *                                     clamp would engage (design §4.4, P2 precedent).
 *
 * GPU tool: reserve a device via tools/pick-gpu.sh, run serial-exclusive. Env:
 *   N (default 8192), TOL (default 2e-7 = 1 ULP + margin), LO/HI (range -12..12).
 */

import {
  getWebGPUInitError,
  initWebGPU,
  webgpuBackend,
} from "../src/backend/webgpu";
import { lowerExprToTileIR } from "../src/backend/webgpu/expr-tile-fold";
import { applyFusedOp } from "../src/backend/webgpu/fusion-tile-ir";
import type { WebGPUTensor } from "../src/backend/webgpu/gpu-types";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import { createTileKernelDispatcher } from "../src/backend/webgpu/tile-dispatch";
import type {
  BlockExpr,
  KernelContext,
  TileKernelSpec,
} from "../src/backend/webgpu/tile-ir";
import { SOFTPLUS_DEF, UNARY_DEFS } from "../src/ops/semantic/catalog";
import { GELU_ERF_DEF, GELU_TANH_DEF } from "../src/ops/semantic/composite";
import { type Expr, erf, x as X } from "../src/ops/semantic/expr";

const N = parseInt(process.env.N ?? "8192", 10);
const TOL = parseFloat(process.env.TOL ?? "2e-7");
const LO = parseFloat(process.env.LO ?? "-12");
const HI = parseFloat(process.env.HI ?? "12");

const log = (m: string) => console.error(`[expr-fold-parity] ${m}`);

const byName = (n: string): Expr => {
  const d = UNARY_DEFS.find((u) => u.name === n);
  if (!d) throw new Error(`no UNARY_DEF '${n}'`);
  return d.expr;
};

/** Each cell: the fold's Expr + the hand op-name for `applyFusedOp` (or a
 *  BlockExpr thunk for `erf`, which has no applyFusedOp case). */
type Cell = {
  name: string;
  expr: Expr;
  hand: string | ((x: BlockExpr) => BlockExpr);
};
const CELLS: Cell[] = [
  { name: "relu", expr: byName("relu"), hand: "relu" },
  { name: "sigmoid", expr: byName("sigmoid"), hand: "sigmoid" },
  { name: "silu", expr: byName("silu"), hand: "silu" },
  { name: "softplus", expr: SOFTPLUS_DEF.expr, hand: "softplus" },
  { name: "gelu_tanh", expr: GELU_TANH_DEF.expr, hand: "gelu" },
  { name: "gelu_erf", expr: GELU_ERF_DEF.expr, hand: "gelu_erf" },
  { name: "erf", expr: erf(X), hand: (x) => x.erf() },
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

async function checkCell(cell: Cell): Promise<{ ok: boolean; max: number }> {
  const WG = 64;
  const spec: TileKernelSpec = {
    name: `foldparity_${cell.name}`,
    workgroupSize: WG,
    bindings: {
      x: { storage: "read", type: "f32" },
      out_fold: { storage: "read_write", type: "f32" },
      out_hand: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: (u: Record<string, number>) => [Math.ceil((u.size ?? 0) / WG)],
    kernel(ctx: KernelContext) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      const xv = ctx.load("x", idx);
      const fold = lowerExprToTileIR(cell.expr, ctx, { x: xv });
      const hand =
        typeof cell.hand === "string"
          ? applyFusedOp(ctx, cell.hand, [xv])
          : cell.hand(xv);
      ctx.emitStore("out_fold", idx, fold);
      ctx.emitStore("out_hand", idx, hand);
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
  const handBuf = webgpuBackend.ops.tensorFromArray(new Array(N).fill(0), [
    N,
  ]) as WebGPUTensor;

  dispatcher.dispatch(
    { x: xBuf.buffer, out_fold: foldBuf.buffer, out_hand: handBuf.buffer },
    { size: N },
  );

  const fold = await webgpuBackend.ops.read(foldBuf);
  const hand = await webgpuBackend.ops.read(handBuf);
  let max = 0;
  for (let i = 0; i < N; i++) {
    const d = Math.abs(fold[i] - hand[i]);
    const rel = d / (Math.abs(hand[i]) + 1e-6);
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
    const kind = max === 0 ? "byte-identical" : "reassociated";
    log(
      `${cell.name.padEnd(10)} max=${max.toExponential(3)} ${kind.padEnd(14)} ${ok ? "PASS" : "FAIL"} (tol=${TOL})`,
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
