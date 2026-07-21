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
import { getMaxStorageBufferBindingSize } from "../backend/webgpu/gpu-context";
import { DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE } from "../backend/webgpu/shape-utils";
import { ENV } from "../core/env";

/** f32 element width — the pack is over `[total]` f32 flats (design §3 clause 3). */
const PACK_ELEM_BYTES = 4;

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

  // Unpack: copy each segment back into its (persistent) param storage. The
  // narrow-at-offset is a strided view, so the reshape MATERIALIZES a fresh
  // per-param contiguous buffer (reshape-of-non-contiguous is the one view op
  // that copies). These `seg`/`narrow` intermediates are step-scoped temporaries
  // whose refs the packer holds; they MUST be disposed after the `copy_`-back is
  // sequenced, or ~1 full-param-width buffer leaks per param per step (the
  // packed-path retention leak — buffer-donation design §2.2). Collected and
  // disposed alongside the big packed intermediates below.
  const unpackTemps: RuntimeTensor[] = [];
  for (let k = 0; k < items.length; k++) {
    const view = rt.narrow(pNew, 0, offsets[k]!, sizes[k]!);
    const seg = rt.reshape(view, items[k]!.param.shape);
    rt.copy_(items[k]!.param, seg);
    unpackTemps.push(view, seg);
  }

  // Dispose the big packed intermediates now that the update graph is built (the
  // IR nodes survive; disposal drops them from the live-pending registry so
  // liveness can release / kernels can donate their buffers). State slots and
  // params are NOT disposed (persistent). A Set dedups gAdj===G (no-L2 case).
  // The unpack `seg`/`narrow` materializations join the set — the `copy_`-back
  // holds the graph edge into `param`, so dropping the wrappers is safe and
  // reclaims the leaked per-param buffers (buffer-donation design P1).
  const toDispose = new Set<RuntimeTensor>([
    G,
    gAdj,
    P,
    ...sink,
    ...unpackTemps,
  ]);
  if (disposeExtra) for (const t of disposeExtra) toDispose.add(t);
  for (const t of toDispose) (t as { dispose?: () => void }).dispose?.();

  return state;
}

/**
 * The max cat SIZE (elements) a sub-class may reach before the backend chunks it.
 * SINGLE SOURCE at the seam: the same device limit the chunking decision reads
 * (`getMaxStorageBufferBindingSize`, delegating to
 * `device.limits.maxStorageBufferBindingSize`, with the shared
 * `DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE` fallback). A CPU/no-limit backend
 * returns Infinity → no split (the packed path is WebGPU-only in practice).
 */
function maxSubClassElems(device: string): number {
  // Test seam (born-with-sunset: retires when C3 is validated on the A100 exit
  // gate): a byte cap that forces the split on a box whose real binding limit is
  // huge (this V100/Dawn reports 2 GB, so a real model never chunks here — the
  // split can only be exercised end-to-end by overriding). Not a user feature.
  const override = ENV.TORCHLETTE_PACK_MAX_BYTES;
  if (override) return Math.floor(parseInt(override, 10) / PACK_ELEM_BYTES);
  if (device !== "webgpu") return Infinity;
  const maxBindingBytes =
    getMaxStorageBufferBindingSize() ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  return Math.floor(maxBindingBytes / PACK_ELEM_BYTES);
}

/**
 * [coverage C3] Split ONE isomorphism class's items into sequential sub-classes,
 * each with `Σ size ≤ maxElems` (so its packed `[Σ size]` cat stays under the
 * binding limit → a NORMAL fused elementwise segment, not chunked, so the
 * landed single-dispatch donation edge fires). DETERMINISTIC: a greedy sweep in
 * the given (stable, param-index) order — same params ⇒ same split ⇒ stable
 * compiled-plan templates. A single item larger than `maxElems` becomes its own
 * sub-class (unavoidably chunked — no split can shrink one param — but isolated,
 * never regressing today's one-class behavior). Returns `[items]` unchanged when
 * the whole class already fits (the ≤128 MB common case).
 */
export function splitPackItems(
  items: PackItem[],
  maxElems: number,
): PackItem[][] {
  if (!Number.isFinite(maxElems)) return [items];
  const sizeOf = (it: PackItem) =>
    it.param.shape.reduce((a: number, b: number) => a * b, 1);
  const subClasses: PackItem[][] = [];
  let cur: PackItem[] = [];
  let curSize = 0;
  for (const it of items) {
    const s = sizeOf(it);
    if (cur.length > 0 && curSize + s > maxElems) {
      subClasses.push(cur);
      cur = [];
      curSize = 0;
    }
    cur.push(it);
    curSize += s;
  }
  if (cur.length > 0) subClasses.push(cur);
  return subClasses;
}

/**
 * Realize ONE isomorphism class, SPLITTING it into sequential ≤128 MB sub-classes
 * (design §5.2 / coverage C3) when its `Σ size` would exceed the binding limit.
 * Each sub-class is realized by `packOptimizerProgram` as an ordinary
 * (non-chunked) fused segment, so the ratified donation edge fires per sub-class,
 * AND — because the planner frees each sub-class's `G`/`pNew` before the next —
 * the transient working set is bounded to one ≤128 MB sub-class instead of the
 * whole model width (the §5.2 dividend). Bit-exact with the single-cat path: the
 * elementwise update is position-independent, so splitting the cat changes only
 * buffer lifetimes, not arithmetic.
 *
 * Returns one `PackedOptState` per sub-class (the class's permanently-packed
 * state, split the same way); the caller caches the array and passes it back as
 * `prevStates` next step. The shared per-step `disposeExtra` temps (e.g. Adam's
 * bc1/bc2, read by EVERY sub-class) are disposed exactly ONCE, after the last
 * sub-class's update graph is built.
 */
export function packOptimizerClass(
  rt: RuntimeEngine,
  spec: PackSpec,
  prevStates?: PackedOptState[],
): PackedOptState[] {
  const device = spec.items[0]?.param.device ?? "webgpu";
  const subClasses = splitPackItems(spec.items, maxSubClassElems(device));
  const out: PackedOptState[] = [];
  for (let k = 0; k < subClasses.length; k++) {
    const isLast = k === subClasses.length - 1;
    out.push(
      packOptimizerProgram(rt, {
        ...spec,
        items: subClasses[k]!,
        prevState: prevStates?.[k],
        // Shared per-step temps are read by every sub-class — dispose once, after
        // the last (disposing earlier would UAF the next sub-class's role read).
        disposeExtra: isLast ? spec.disposeExtra : undefined,
      }),
    );
  }
  return out;
}
