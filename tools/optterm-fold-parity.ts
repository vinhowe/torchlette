/**
 * R1 numeric differential — the `OptTerm → tile-IR` FOLD vs the `evalOptTerm`
 * GRAPH interpreter (derived-optimizer-realizer campaign;
 * docs/derived-optimizer-realizer-design.md ruling O1, phase R1).
 *
 * The load-bearing claim of R1: the fold and `evalOptTerm` are two derivations of
 * ONE algebra, so they must AGREE numerically. For each elementwise optimizer
 * program this script:
 *   1. draws a random flat buffer for every role the program reads;
 *   2. REFERENCE: interprets each output term (state updates + param update) with
 *      `evalOptTensor` over the runtime engine (the graph fold), reads it back;
 *   3. FOLD: compiles ONE elementwise kernel whose body is the SAME output terms
 *      folded by `lowerOptTermToTileIR` (each role = a `load` at idx), dispatches
 *      it on the GPU, reads it back;
 *   4. asserts the two agree — finiteness position-by-position AND ≤ TOL on every
 *      finite position (default 1e-6, the design's R1 gate).
 *
 * Muon asserts the STRUCTURAL refusal (the fold cannot lower its `mm`).
 *
 * GPU tool: reserve a device via tools/pick-gpu.sh, run serial-exclusive. Env:
 *   N (default 4096), TOL (default 1e-6), PROGS (csv; default adamw,sgd,sgd_momentum,lion).
 */

import { initWebGPU, getWebGPUInitError } from "../src/backend/webgpu";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import {
  createTileKernelDispatcher,
} from "../src/backend/webgpu/tile-dispatch";
import type {
  BlockExpr,
  KernelContext,
  TileKernelSpec,
} from "../src/backend/webgpu/tile-ir";
import { webgpuBackend } from "../src/backend/webgpu";
import type { WebGPUTensor } from "../src/backend/webgpu/gpu-types";
import { Torchlette } from "../src/frontend/torchlette";
import {
  ADAMW_PROGRAM,
  LION_PROGRAM,
  MUON_PROGRAM,
  type OptimizerProgram,
  type OptRoles,
  type OptTerm,
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
  evalOptTensor,
} from "../src/ops/semantic/optimizer";
import {
  lowerOptTermToTileIR,
  type FoldRoleBindings,
} from "../src/schedule/optterm-fold";
import { OptimizerPackRefusal } from "../src/optim/pack-optimizer";

const N = parseInt(process.env.N ?? "4096", 10);
const TOL = parseFloat(process.env.TOL ?? "1e-6");
const PROGS = (process.env.PROGS ?? "adamw,sgd,sgd_momentum,lion")
  .split(",")
  .map((s) => s.trim());

const CATALOG: Record<string, OptimizerProgram> = {
  adamw: ADAMW_PROGRAM,
  sgd: SGD_PROGRAM,
  sgd_momentum: SGD_MOMENTUM_PROGRAM,
  lion: LION_PROGRAM,
};

const log = (m: string) => console.error(`[fold-parity] ${m}`);

function collectRoles(t: OptTerm, into: Set<string> = new Set()): Set<string> {
  switch (t.k) {
    case "role":
      into.add(t.name);
      break;
    case "u":
      collectRoles(t.a, into);
      break;
    case "b":
      collectRoles(t.a, into);
      collectRoles(t.b, into);
      break;
    case "mm":
      collectRoles(t.a, into);
      collectRoles(t.b, into);
      break;
    default:
      break;
  }
  return into;
}

function outputTerms(p: OptimizerProgram): { name: string; expr: OptTerm }[] {
  return [
    ...p.stateUpdates.map((su) => ({ name: su.slot, expr: su.expr })),
    { name: "param", expr: p.paramUpdate },
  ];
}

/** Deterministic PRNG in [-1, 1]. */
function randData(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

async function checkProgram(
  api: Torchlette,
  p: OptimizerProgram,
): Promise<{ ok: boolean; maxAbs: number; finiteMismatch: number }> {
  const roleSet = new Set<string>();
  for (const t of outputTerms(p)) collectRoles(t.expr, roleSet);
  const roleNames = [...roleSet];

  // Random data per role — identical bytes fed to BOTH paths.
  const data: Record<string, Float32Array> = {};
  roleNames.forEach((r, i) => {
    data[r] = randData(N, 0x9e3779b9 ^ (i * 2654435761));
  });

  // -- REFERENCE: evalOptTensor over the runtime engine ----------------------
  const roleTensors: OptRoles = {};
  for (const r of roleNames) roleTensors[r] = api.tensorFromArray(data[r]!, [N])._unwrap();
  const refOut: Record<string, Float32Array> = {};
  for (const t of outputTerms(p)) {
    const rt = evalOptTensor(t.expr, api.runtime, roleTensors);
    refOut[t.name] = Float32Array.from(await api._wrap(rt).cpu());
  }

  // -- FOLD: compile+dispatch ONE elementwise kernel folding the same terms ---
  const WG = 64;
  const bindings: TileKernelSpec["bindings"] = {};
  for (const r of roleNames) bindings[r] = { storage: "read", type: "f32" };
  for (const t of outputTerms(p))
    bindings[`out_${t.name}`] = { storage: "read_write", type: "f32" };
  const spec: TileKernelSpec = {
    name: `fold_${p.name}`,
    workgroupSize: WG,
    bindings,
    uniforms: { size: "u32" },
    grid: (u: Record<string, number>) => [Math.ceil((u.size ?? 0) / WG)],
    kernel(ctx: KernelContext) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      const roleBind: Record<string, BlockExpr> = {};
      for (const r of roleNames) roleBind[r] = ctx.load(r, idx);
      const memo = new Map<OptTerm, BlockExpr>();
      for (const t of outputTerms(p)) {
        const val = lowerOptTermToTileIR(t.expr, ctx, roleBind as FoldRoleBindings, memo);
        ctx.emitStore(`out_${t.name}`, idx, val);
      }
    },
  };
  // Compile once so any codegen error surfaces cleanly before dispatch.
  compileTileKernel(spec);
  const dispatcher = createTileKernelDispatcher(spec);

  const roleBufs: Record<string, WebGPUTensor> = {};
  for (const r of roleNames)
    roleBufs[r] = webgpuBackend.ops.tensorFromArray(Array.from(data[r]!), [N]) as WebGPUTensor;
  const outTensors: Record<string, WebGPUTensor> = {};
  for (const t of outputTerms(p))
    outTensors[t.name] = webgpuBackend.ops.tensorFromArray(new Array(N).fill(0), [N]) as WebGPUTensor;

  const bufMap: Record<string, WebGPUTensor["buffer"]> = {};
  for (const r of roleNames) bufMap[r] = roleBufs[r]!.buffer;
  for (const t of outputTerms(p)) bufMap[`out_${t.name}`] = outTensors[t.name]!.buffer;

  dispatcher.dispatch(bufMap, { size: N });

  const foldOut: Record<string, Float32Array> = {};
  for (const t of outputTerms(p))
    foldOut[t.name] = Float32Array.from(await webgpuBackend.ops.read(outTensors[t.name]!));

  // -- COMPARE ---------------------------------------------------------------
  let maxAbs = 0;
  let finiteMismatch = 0;
  for (const t of outputTerms(p)) {
    const ref = refOut[t.name]!;
    const got = foldOut[t.name]!;
    for (let i = 0; i < N; i++) {
      const rf = Number.isFinite(ref[i]);
      const gf = Number.isFinite(got[i]);
      if (rf !== gf) {
        finiteMismatch++;
        continue;
      }
      if (rf && gf) {
        const rel = Math.abs(ref[i]! - got[i]!) / (Math.abs(ref[i]!) + 1e-8);
        maxAbs = Math.max(maxAbs, Math.min(Math.abs(ref[i]! - got[i]!), rel));
      }
    }
  }
  const ok = finiteMismatch === 0 && maxAbs <= TOL;
  return { ok, maxAbs, finiteMismatch };
}

async function main(): Promise<void> {
  const ready = await initWebGPU();
  if (!ready) throw new Error(`WebGPU init failed: ${getWebGPUInitError()}`);
  const api = new Torchlette("webgpu");

  let allOk = true;

  // Muon: the STRUCTURAL refusal (the fold cannot lower an mm node).
  {
    let refused = false;
    try {
      const spec: TileKernelSpec = {
        name: "fold_muon",
        workgroupSize: 64,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { size: "u32" },
        grid: () => [1],
        kernel(ctx) {
          const roleBind: Record<string, BlockExpr> = {};
          for (const r of collectRoles(MUON_PROGRAM.paramUpdate))
            roleBind[r] = ctx.f32(1);
          lowerOptTermToTileIR(MUON_PROGRAM.paramUpdate, ctx, roleBind as FoldRoleBindings);
        },
      };
      compileTileKernel(spec);
    } catch (e) {
      refused = e instanceof OptimizerPackRefusal;
      if (!refused) throw e;
    }
    log(`muon      refusal=${refused ? "OptimizerPackRefusal ✓" : "NO REFUSAL ✗"}`);
    if (!refused) allOk = false;
  }

  for (const name of PROGS) {
    const p = CATALOG[name];
    if (!p) {
      log(`unknown program '${name}' — skipped`);
      continue;
    }
    const { ok, maxAbs, finiteMismatch } = await checkProgram(api, p);
    log(
      `${name.padEnd(13)} maxAbs=${maxAbs.toExponential(3)} finiteMismatch=${finiteMismatch} ${ok ? "✓" : "✗"} (tol=${TOL})`,
    );
    if (!ok) allOk = false;
  }

  log(allOk ? "ALL PROGRAMS PASS" : "PARITY FAILURE");
  if (!allOk) process.exit(1);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
