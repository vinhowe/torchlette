/**
 * The fused-optimizer SPECS (derived-optimizer-realizer campaign, R5a) — one
 * `OptStepRealizerSpec` per optimizer program. Each declares, as DATA, how the
 * generic program-roles realizer (`opt-step-realizer.ts`) assembles that
 * optimizer's fused kernel: which program to fold, the wd-free param term, the
 * state-read polarity (post-update vs old), the scalar-DATA inputs (bc/lr), and
 * the static-hyper uniforms. A new optimizer becomes a fused kernel by adding a
 * spec here — zero per-optimizer WGSL (design ruling O1, §3.4).
 *
 * The specs are the SINGLE SOURCE the schedule chokepoint, the backend dispatch,
 * and the realizer-parity differential all read — never re-derived per site.
 */

import {
  ADAMW_PROGRAM,
  ADAMW_SCALED,
  LION_PROGRAM,
  LION_STEP,
  oSub,
  role,
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
} from "../ops/semantic/optimizer";
import type { TileKernelSpec } from "../backend/webgpu/tile-ir";
import { lowerOptStepBody, type OptStepRealizerSpec } from "./opt-step-realizer";

/**
 * AdamW/Adam. bc=[bc1,bc2] + lr are scalar-DATA inputs (host-computed live
 * scalars); beta1/beta2/eps/weight_decay are uniforms (ln_beta1/ln_beta2 are
 * retained-but-dead so the setAdamConfigUniforms block is byte-unchanged). The
 * wd-free param term is `p − SCALED`; the decoupled `lr·wd·p` (AdamW) rides the
 * realizer's runtime `decoupled_wd` branch, L2 rides its `g += wd·p` branch.
 */
export const ADAM_STEP_SPEC: OptStepRealizerSpec = {
  program: ADAMW_PROGRAM,
  paramUpdateNoWd: oSub(role("p"), ADAMW_SCALED),
  paramReadsPostState: true,
  scalarInputs: [
    { name: "bc", length: 2, roles: ["bc1", "bc2"] },
    { name: "lr", length: 1, roles: ["lr"] },
  ],
  f32Uniforms: [
    "beta1",
    "beta2",
    "ln_beta1",
    "ln_beta2",
    "eps",
    "weight_decay",
  ],
  kernelName: "adamStep",
};

/**
 * Lion. State is a single β2-EMA `m`; the param term reads the OLD momentum
 * (paramReadsPostState=false — the sign step is a β1-interp of the pre-update m).
 * Lion has no L2 variant, so `decoupled_wd` is always 1 (the decoupled branch
 * fires). lr is the only scalar-DATA input.
 */
export const LION_STEP_SPEC: OptStepRealizerSpec = {
  program: LION_PROGRAM,
  paramUpdateNoWd: oSub(role("p"), LION_STEP),
  paramReadsPostState: false,
  scalarInputs: [{ name: "lr", length: 1, roles: ["lr"] }],
  f32Uniforms: ["beta1", "beta2", "weight_decay"],
  kernelName: "lionStep",
};

/**
 * SGD with momentum. State is the velocity `v` (read post-update: p' = p − lr·v').
 * L2 weight decay folds into `g` (the realizer's `decoupled_wd==0` branch), so the
 * param term is already wd-free.
 */
export const SGD_MOMENTUM_STEP_SPEC: OptStepRealizerSpec = {
  program: SGD_MOMENTUM_PROGRAM,
  paramUpdateNoWd: SGD_MOMENTUM_PROGRAM.paramUpdate,
  paramReadsPostState: true,
  scalarInputs: [{ name: "lr", length: 1, roles: ["lr"] }],
  f32Uniforms: ["mu", "weight_decay"],
  kernelName: "sgdMomentumStep",
};

/** Plain SGD (no state): p' = p − lr·g, with L2 folded into g. */
export const SGD_STEP_SPEC: OptStepRealizerSpec = {
  program: SGD_PROGRAM,
  paramUpdateNoWd: SGD_PROGRAM.paramUpdate,
  paramReadsPostState: true,
  scalarInputs: [{ name: "lr", length: 1, roles: ["lr"] }],
  f32Uniforms: ["weight_decay"],
  kernelName: "sgdStep",
};

/** The spec catalog, keyed by program name (the differential + wiring iterate it). */
export const OPT_STEP_SPECS: Record<string, OptStepRealizerSpec> = {
  adamw: ADAM_STEP_SPEC,
  lion: LION_STEP_SPEC,
  sgd_momentum: SGD_MOMENTUM_STEP_SPEC,
  sgd: SGD_STEP_SPEC,
};

/**
 * The GENERIC fused-optimizer realizer entry (derived-optimizer-realizer R5b) —
 * the de-named dispatch chokepoint the WebGPU backend routes through. Resolves
 * the spec by name (`OptStepConfig.spec`) and folds its body via the program-
 * roles realizer. For `adamw` this returns byte-for-byte the prior
 * `realizeAdamStepSpec` output (both are `lowerOptStepBody(ADAM_STEP_SPEC,…)`);
 * the schedule keeps `realizeAdamStepSpec`/the authored-seam differential as the
 * Adam-specific single-source guard. The backend never names an optimizer — it
 * passes `config.spec` through.
 */
export function realizeOptStepSpec(
  specName: string,
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  const spec = OPT_STEP_SPECS[specName];
  if (!spec)
    throw new Error(
      `realizeOptStepSpec: unknown optimizer spec '${specName}' (expected one of ${Object.keys(OPT_STEP_SPECS).join(", ")}).`,
    );
  return lowerOptStepBody(spec, useVec4, emitF16, emitUnscale);
}
