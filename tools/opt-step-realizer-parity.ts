/**
 * R5a realizer differential — the PROGRAM-ROLES REALIZER (`lowerOptStepBody`,
 * opt-step-realizer.ts) vs the `evalOptTerm` GRAPH interpreter, for the FULL fused
 * in-place kernel of each optimizer (Adam / Lion / SGD / SGD+momentum).
 *
 * Where `optterm-fold-parity` proves the per-NODE fold, this proves the whole
 * ASSEMBLED kernel the realizer emits — the in-place param/state writes, the
 * scalar-DATA inputs (bc/lr), the runtime L2 vs decoupled weight-decay branches,
 * and the paramReadsPostState polarity. For each spec it:
 *   1. draws random grad/param/state buffers + bc/lr + a wd;
 *   2. REFERENCE: replays the realizer's own policy over `evalOptTerm` — the L2
 *      `g += wd·p` fold (decoupled_wd=0) or the decoupled `p' -= lr·wd·p` fold
 *      (decoupled_wd=1), state updates, and the wd-free param term — reads it back;
 *   3. KERNEL: dispatches the realizer kernel IN PLACE over copies of the same
 *      buffers, reads param + state back;
 *   4. asserts agreement (≤ TOL, default 1e-6) position-by-position.
 *
 * Both weight-decay policies are exercised per optimizer where meaningful (Adam
 * L2 + AdamW; SGD L2; Lion decoupled). GPU tool: reserve via tools/pick-gpu.sh.
 * Env: N (4096), TOL (1e-6).
 */

import {
  getWebGPUInitError,
  initWebGPU,
  webgpuBackend,
} from "../src/backend/webgpu";
import type { WebGPUTensor } from "../src/backend/webgpu/gpu-types";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import { createTileKernelDispatcher } from "../src/backend/webgpu/tile-dispatch";
import { Torchlette } from "../src/frontend/torchlette";
import {
  evalOptTensor,
  type OptRoles,
} from "../src/ops/semantic/optimizer";
import {
  lowerOptStepBody,
  type OptStepRealizerSpec,
} from "../src/schedule/opt-step-realizer";
import {
  ADAM_STEP_SPEC,
  LION_STEP_SPEC,
  SGD_MOMENTUM_STEP_SPEC,
  SGD_STEP_SPEC,
} from "../src/schedule/opt-step-specs";

const N = parseInt(process.env.N ?? "4096", 10);
const TOL = parseFloat(process.env.TOL ?? "1e-6");
const log = (m: string) => console.error(`[realizer-parity] ${m}`);

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

/** One test case: a spec + the hyper values + the wd policy. */
interface Case {
  label: string;
  spec: OptStepRealizerSpec;
  decoupledWd: boolean;
  wd: number;
  /** hyper uniform values (f32) — beta1/beta2/eps/mu/weight_decay as the spec needs. */
  hypers: Record<string, number>;
  /** scalar-DATA buffers by input name (bc has length 2, lr length 1). */
  scalar: Record<string, number[]>;
}

async function runCase(
  api: Torchlette,
  c: Case,
): Promise<{ ok: boolean; maxAbs: number }> {
  const { spec } = c;
  const lr = c.scalar.lr![0]!;
  const bc1 = c.scalar.bc?.[0];
  const bc2 = c.scalar.bc?.[1];

  // Random inputs (identical bytes to both paths).
  const grad = randData(N, 0x1234);
  const param = randData(N, 0x5678);
  const stateInit: Record<string, Float32Array> = {};
  spec.program.state.forEach((slot, i) => {
    // State is NON-NEGATIVE: Adam's v is a running sum of squares (≥0), so
    // √(v/bc2) is well-defined. Feeding negative v is UB (sqrt of a negative
    // diverges between the graph and tile-IR lowerings); m/velocity being ≥0 is
    // a harmless restriction for Lion/SGD (both paths still agree).
    stateInit[slot] = randData(N, 0xa000 + i).map(Math.abs);
  });

  // -- REFERENCE via evalOptTerm + the realizer's wd policy -------------------
  const rt = api.runtime;
  const t = (d: Float32Array) =>
    api.tensorFromArray(Array.from(d), [N])._unwrap();
  const gradT = t(grad);
  const paramT = t(param);
  // L2: g += wd*p (decoupled_wd==0 && wd>0).
  const gAdj =
    !c.decoupledWd && c.wd > 0 ? rt.add(gradT, rt.mul(paramT, c.wd)) : gradT;
  const roles: OptRoles = {
    g: gAdj,
    p: paramT,
    lr,
    wd: c.wd,
  };
  if (c.hypers.beta1 !== undefined) {
    roles.beta1 = c.hypers.beta1;
    roles.om_beta1 = 1 - c.hypers.beta1;
  }
  if (c.hypers.beta2 !== undefined) {
    roles.beta2 = c.hypers.beta2;
    roles.om_beta2 = 1 - c.hypers.beta2;
  }
  if (c.hypers.eps !== undefined) roles.eps = c.hypers.eps;
  if (c.hypers.mu !== undefined) roles.mu = c.hypers.mu;
  if (bc1 !== undefined) roles.bc1 = bc1;
  if (bc2 !== undefined) roles.bc2 = bc2;
  for (const slot of spec.program.state) roles[slot] = t(stateInit[slot]!);

  // State updates (read OLD state), THEN rebind for the param term.
  const stateNew: Record<string, ReturnType<typeof t>> = {};
  for (const su of spec.program.stateUpdates)
    stateNew[su.slot] = evalOptTensor(su.expr, rt, roles);
  const paramRoles: OptRoles = { ...roles };
  if (spec.paramReadsPostState)
    for (const slot of spec.program.state) paramRoles[slot] = stateNew[slot]!;
  let pNew = evalOptTensor(spec.paramUpdateNoWd, rt, paramRoles);
  // Decoupled: p' -= lr*wd*p.
  if (c.decoupledWd && c.wd > 0) pNew = rt.sub(pNew, rt.mul(paramT, lr * c.wd));

  const refParam = Float32Array.from(await api._wrap(pNew).cpu());
  const refState: Record<string, Float32Array> = {};
  for (const slot of spec.program.state)
    refState[slot] = Float32Array.from(await api._wrap(stateNew[slot]!).cpu());

  // -- KERNEL: dispatch the realizer kernel in place --------------------------
  const kSpec = lowerOptStepBody(spec, false, false, false);
  compileTileKernel(kSpec);
  const dispatcher = createTileKernelDispatcher(kSpec);

  const mk = (d: Float32Array | number[]) =>
    webgpuBackend.ops.tensorFromArray(Array.from(d), [
      d.length,
    ]) as WebGPUTensor;
  const gradBuf = mk(grad);
  const paramBuf = mk(param); // written in place
  const stateBufs: Record<string, WebGPUTensor> = {};
  for (const slot of spec.program.state) stateBufs[slot] = mk(stateInit[slot]!);
  const scalarBufs: Record<string, WebGPUTensor> = {};
  for (const inp of spec.scalarInputs)
    scalarBufs[inp.name] = mk(c.scalar[inp.name]!);

  const bufMap: Record<string, WebGPUTensor["buffer"]> = {
    grad: gradBuf.buffer,
    param: paramBuf.buffer,
  };
  for (const slot of spec.program.state) bufMap[slot] = stateBufs[slot]!.buffer;
  for (const inp of spec.scalarInputs)
    bufMap[inp.name] = scalarBufs[inp.name]!.buffer;

  const uniforms: Record<string, number> = {
    num_elements: N,
    decoupled_wd: c.decoupledWd ? 1 : 0,
    _pad0: 0,
    _pad1: 0,
    _pad2: 0,
    _pad3: 0,
  };
  for (const name of spec.f32Uniforms) {
    // ln_beta* are dead in the derived body but declared; fill with a real value.
    if (name === "ln_beta1") uniforms[name] = Math.log(c.hypers.beta1 ?? 0.9);
    else if (name === "ln_beta2")
      uniforms[name] = Math.log(c.hypers.beta2 ?? 0.999);
    else if (name === "weight_decay") uniforms[name] = c.wd;
    else uniforms[name] = c.hypers[name] ?? 0;
  }

  dispatcher.dispatch(bufMap, uniforms);

  const gotParam = Float32Array.from(await webgpuBackend.ops.read(paramBuf));
  const gotState: Record<string, Float32Array> = {};
  for (const slot of spec.program.state)
    gotState[slot] = Float32Array.from(
      await webgpuBackend.ops.read(stateBufs[slot]!),
    );

  // -- COMPARE ----------------------------------------------------------------
  let maxAbs = 0;
  const cmp = (ref: Float32Array, got: Float32Array) => {
    for (let i = 0; i < N; i++) {
      const a = ref[i]!;
      const b = got[i]!;
      if (!Number.isFinite(a) || !Number.isFinite(b)) {
        if (Number.isFinite(a) !== Number.isFinite(b)) maxAbs = Infinity;
        continue;
      }
      const rel = Math.abs(a - b) / (Math.abs(a) + 1e-8);
      maxAbs = Math.max(maxAbs, Math.min(Math.abs(a - b), rel));
    }
  };
  cmp(refParam, gotParam);
  for (const slot of spec.program.state) cmp(refState[slot]!, gotState[slot]!);

  const ok = maxAbs <= TOL;
  log(
    `${c.label.padEnd(26)} maxAbs=${maxAbs.toExponential(3)} ${ok ? "✓" : "✗"} (tol=${TOL})`,
  );
  return { ok, maxAbs };
}

async function main(): Promise<void> {
  const ready = await initWebGPU();
  if (!ready) throw new Error(`WebGPU init failed: ${getWebGPUInitError()}`);
  const api = new Torchlette("webgpu");

  const cases: Case[] = [
    {
      label: "adamw (decoupled wd)",
      spec: ADAM_STEP_SPEC,
      decoupledWd: true,
      wd: 0.01,
      hypers: { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
      scalar: { bc: [0.271, 0.049], lr: [0.003] },
    },
    {
      label: "adam (L2 wd)",
      spec: ADAM_STEP_SPEC,
      decoupledWd: false,
      wd: 0.05,
      hypers: { beta1: 0.9, beta2: 0.999, eps: 1e-8 },
      scalar: { bc: [0.271, 0.049], lr: [0.003] },
    },
    {
      label: "lion (decoupled wd)",
      spec: LION_STEP_SPEC,
      decoupledWd: true,
      wd: 0.02,
      hypers: { beta1: 0.9, beta2: 0.99 },
      scalar: { lr: [0.001] },
    },
    {
      label: "lion (wd=0)",
      spec: LION_STEP_SPEC,
      decoupledWd: true,
      wd: 0,
      hypers: { beta1: 0.9, beta2: 0.99 },
      scalar: { lr: [0.001] },
    },
    {
      label: "sgd_momentum (L2 wd)",
      spec: SGD_MOMENTUM_STEP_SPEC,
      decoupledWd: false,
      wd: 0.03,
      hypers: { mu: 0.9 },
      scalar: { lr: [0.01] },
    },
    {
      label: "sgd (L2 wd)",
      spec: SGD_STEP_SPEC,
      decoupledWd: false,
      wd: 0.04,
      hypers: {},
      scalar: { lr: [0.02] },
    },
    {
      label: "sgd (wd=0)",
      spec: SGD_STEP_SPEC,
      decoupledWd: false,
      wd: 0,
      hypers: {},
      scalar: { lr: [0.02] },
    },
  ];

  let allOk = true;
  for (const c of cases) {
    const { ok } = await runCase(api, c);
    if (!ok) allOk = false;
  }
  log(allOk ? "ALL REALIZER PARITY PASS" : "REALIZER PARITY FAILURE");
  if (!allOk) process.exit(1);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
