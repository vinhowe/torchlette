/**
 * The parallel-isomorphic-chain packer (chain-packing campaign P1).
 *
 * `packOptimizerProgram` is the optimizer-agnostic realization of ONE
 * isomorphism class (docs/chain-packing-design.md §2, §3): given N per-param
 * chains of the SAME `OptimizerProgram`, it emits ONE flat-packed chain of
 * ordinary runtime ops —
 *
 *     reshape → cat  →  evalOptTerm over the [Σ sizeᵢ] buffer  →  narrow → copy_
 *
 * so vertical fusion, the memory planner, and compiled replay pack and schedule
 * it with ZERO optimizer-specific machinery (design §2.4). The arithmetic is
 * interpreted ONCE over the concatenated buffer, so a class of N params costs one
 * fused chain, not N (design §1 corollary).
 *
 * This is `Adam._foreachGroupStep` generalized off Adam — the mechanism was
 * already proven there (architecture-debt.md stage-2: bit-exact vs ground truth,
 * optimizer cleanup 355→10 ms). Adam, Lion, and SGD are all thin callers: they
 * declare their program + the class's shared per-step DATA + per-param state, and
 * this primitive owns the pack/eval/unpack/dispose EFFECTS.
 *
 * DECLARED, never RECOGNIZED (design §2.1): the packing decision comes from the
 * caller's group-by on the §3 key; the packing execution is these graph nodes.
 *
 * Lifetime discipline (CLAUDE.md; design R4): the packed state slots are created
 * via `runtime.registerState` and updated IN PLACE via `copy_` (never
 * replace-and-hold); the param update reads state through the post-`copy_` refs
 * so a force of the params executes the state writes; the big packed
 * intermediates are disposed once the update graph is built.
 */

import type { RuntimeEngine } from "../runtime/engine";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import {
  evalOptTensor,
  type OptimizerProgram,
  type OptRoles,
  type OptTerm,
} from "../ops/semantic/optimizer";

/** A single per-parameter chain to fold into the class's packed dispatch. */
export interface PackItem {
  /**
   * Stable identity for the class SIGNATURE (design §3) — the optimizer's flat
   * param index. A membership/layout change across steps throws rather than
   * silently remapping the permanently-packed state (the UAF/corruption class).
   */
  id: number;
  /** The parameter storage (updated in place at unpack). */
  param: RuntimeTensor;
  /** The parameter's gradient (present — grad-absent params are filtered out). */
  grad: RuntimeTensor;
  /** Persistent per-param state, aligned to `program.state` order. Packed once. */
  state: RuntimeTensor[];
}

/** The class's permanently-packed state — one `[Σ size]` tensor per state slot. */
export interface PackedOptState {
  /** One packed tensor per `program.state` slot (same order). */
  slots: RuntimeTensor[];
  /** Membership+layout signature (`id:size,…`); a change across steps throws. */
  sig: string;
}

export interface PackSpec {
  program: OptimizerProgram;
  items: PackItem[];
  /**
   * Shared per-step DATA + constant-hyper roles the terms read (lr, eps, wd,
   * beta*, bc1, bc2, mu, …). `g`, `p`, and the `program.state` slots are bound by
   * the packer; anything else the terms reference must appear here.
   */
  sharedRoles: OptRoles;
  /**
   * The realizer-selected param-update term. Defaults to `program.paramUpdate`;
   * a realizer passes a leaner term (e.g. `sub(P, SCALED)`) to avoid a
   * full-model-size `+0` in the wd=0 case (design §2.5 / optimizer.ts).
   */
  paramUpdate?: OptTerm;
  /**
   * L2 weight-decay policy: given the packed grad `g` and param `p`, return the
   * adjusted packed grad (`g += wd·p`). Identity if omitted. This is the
   * realizer POLICY that selects L2-vs-decoupled (design §3 clause 1); the
   * packer stays agnostic.
   */
  adjustGrad?: (
    rt: RuntimeEngine,
    g: RuntimeTensor,
    p: RuntimeTensor,
  ) => RuntimeTensor;
  /**
   * true (default): the param term reads state through the POST-`copy_` refs
   * (Adam/SGD — p' uses the updated moments). false: it reads the OLD state
   * (Lion — its step direction is a β1-interp of the pre-update momentum).
   */
  paramReadsPostState?: boolean;
  /** The class's packed state from a prior step; undefined on the first pack. */
  prevState?: PackedOptState;
  /**
   * Extra caller per-step temps to dispose with the packed intermediates (e.g.
   * Adam's bc1/bc2 [1] tensors) — keeps the realizer's exact memory discipline.
   */
  disposeExtra?: RuntimeTensor[];
}

/** Typed refusal: a class whose term contains a contraction cannot be flat-packed. */
export class OptimizerPackRefusal extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OptimizerPackRefusal";
  }
}

/** Does an OptTerm contain a contraction (`mm`) node anywhere? (design §3 clause 4) */
export function optTermHasMatmul(t: OptTerm): boolean {
  switch (t.k) {
    case "mm":
      return true;
    case "u":
      return optTermHasMatmul(t.a);
    case "b":
      return optTermHasMatmul(t.a) || optTermHasMatmul(t.b);
    default:
      return false;
  }
}

/**
 * Assert every term in the class is elementwise-flattenable (design §3 clause 4,
 * §6.1). A `mm` (contraction) cannot be flat-packed — a `cat` of differently
 * shaped matrices is NOT one matmul, so silently flat-packing it would produce a
 * WRONG RESULT (the worst failure mode). This is a HARD gate, not a heuristic:
 * it throws `OptimizerPackRefusal` so the caller falls back to per-param dispatch.
 */
export function assertFlattenable(
  program: OptimizerProgram,
  paramUpdate: OptTerm,
): void {
  for (const su of program.stateUpdates) {
    if (optTermHasMatmul(su.expr))
      throw new OptimizerPackRefusal(
        `packOptimizerProgram: ${program.name} state '${su.slot}' contains a contraction (mm) node — not elementwise-flattenable; use per-param dispatch.`,
      );
  }
  if (optTermHasMatmul(paramUpdate))
    throw new OptimizerPackRefusal(
      `packOptimizerProgram: ${program.name} param update contains a contraction (mm) node — not elementwise-flattenable; use per-param dispatch.`,
    );
}

/**
 * Emit the packed chain for ONE isomorphism class. Returns the class's packed
 * state (new on the first call, `prevState` reused thereafter) for the caller to
 * cache and pass back next step.
 *
 * PRECONDITION: `items` are pre-grouped by the §3 isomorphism key (same program,
 * wd-mode, dtype, shared lr/t bindings) and every item's grad is present.
 */
export function packOptimizerProgram(
  rt: RuntimeEngine,
  spec: PackSpec,
): PackedOptState {
  const { program, items, sharedRoles, adjustGrad, prevState, disposeExtra } =
    spec;
  const paramUpdate = spec.paramUpdate ?? program.paramUpdate;
  const paramReadsPostState = spec.paramReadsPostState ?? true;

  if (items.length === 0)
    throw new Error("packOptimizerProgram: empty class (no items).");

  // HARD gate: no contraction node in a flat-packed class (design §3 clause 4).
  assertFlattenable(program, paramUpdate);

  // Sizes / offsets / signature (design §3: shapes MAY differ; boundaries live
  // only in host-side narrow args, never reach the GPU).
  const sizes = items.map((it) =>
    it.param.shape.reduce((a: number, b: number) => a * b, 1),
  );
  const offsets: number[] = [];
  let total = 0;
  for (const s of sizes) {
    offsets.push(total);
    total += s;
  }
  const sig = items.map((it, k) => `${it.id}:${sizes[k]}`).join(",");

  if (prevState && prevState.sig !== sig)
    throw new OptimizerPackRefusal(
      "packOptimizerProgram: the set of grad-bearing params in a class changed " +
        "across steps; packed state cannot be remapped. Use the per-param path.",
    );

  // Pack grads and params: [size_i] flats concatenated to one [total].
  const gFlat = items.map((it, k) => rt.reshape(it.grad, [sizes[k]!]));
  const G = rt.cat(gFlat);
  const pFlat = items.map((it, k) => rt.reshape(it.param, [sizes[k]!]));
  const P = rt.cat(pFlat);

  // Packed state: initialized ONCE by packing the current per-param state, then
  // registered (adopted into the step snapshot — else markStep demotes the
  // mid-step-created tensor and pools its live buffer: the silent-UAF class) and
  // updated IN PLACE thereafter so its buffers are stable across steps.
  let state = prevState;
  if (!state) {
    const slots = program.state.map((_, slotIdx) => {
      const flats = items.map((it, k) =>
        rt.reshape(it.state[slotIdx]!, [sizes[k]!]),
      );
      return rt.registerState(rt.cat(flats));
    });
    state = { slots, sig };
  }

  // L2 grad-fold POLICY (realizer-declared); identity if omitted.
  const gAdj = adjustGrad ? adjustGrad(rt, G, P) : G;

  // Bind roles: state slots → packed state, g → adjusted packed grad, p → packed
  // params, plus the caller's shared per-step DATA + constant hypers.
  const roles: OptRoles = { ...sharedRoles, g: gAdj, p: P };
  for (let j = 0; j < program.state.length; j++)
    roles[program.state[j]!] = state.slots[j]!;

  const sink: RuntimeTensor[] = [];

  const evalStateUpdates = (): void => {
    // Evaluate every state update reading the OLD packed state, THEN copy_ each
    // in place (so a slot read by a later term sees the pre-step value until its
    // own copy_ commits — matching the hand foreach/elementwise order).
    const news = program.stateUpdates.map((su) =>
      evalOptTensor(su.expr, rt, roles, sink),
    );
    program.stateUpdates.forEach((su, k) => {
      const slotIdx = program.state.indexOf(su.slot);
      rt.copy_(state!.slots[slotIdx]!, news[k]!);
    });
  };

  let pNew: RuntimeTensor;
  if (paramReadsPostState) {
    // Adam / SGD: state written first; p' reads the updated state through the
    // post-copy_ refs (the dangling-copy_ lifetime discipline).
    evalStateUpdates();
    pNew = evalOptTensor(paramUpdate, rt, roles, sink);
  } else {
    // Lion: p' reads the OLD momentum; the stored EMA is written afterwards.
    pNew = evalOptTensor(paramUpdate, rt, roles, sink);
    evalStateUpdates();
  }

  // Unpack: copy each segment back into its (persistent) param storage.
  for (let k = 0; k < items.length; k++) {
    const seg = rt.reshape(
      rt.narrow(pNew, 0, offsets[k]!, sizes[k]!),
      items[k]!.param.shape,
    );
    rt.copy_(items[k]!.param, seg);
  }

  // Dispose the big packed intermediates now that the update graph is built (the
  // IR nodes survive; disposal drops them from the live-pending registry so
  // liveness can release / kernels can donate their buffers). State slots and
  // params are NOT disposed (persistent). A Set dedups gAdj===G (no-L2 case).
  const toDispose = new Set<RuntimeTensor>([G, gAdj, P, ...sink]);
  if (disposeExtra) for (const t of disposeExtra) toDispose.add(t);
  for (const t of toDispose) (t as { dispose?: () => void }).dispose?.();

  return state;
}
