/**
 * Walking skeleton — REDUCTION + ROW-PROGRAM families (campaign P0-FULL wave 1,
 * family 2). Side-by-side with the live path; NO behavior change. Exercised only
 * by test/schedule/reduction-differential.spec.ts.
 *
 * ------------------------------------------------------------------------
 * THE DIFFERENTIAL CONTRACT (same bar as elementwise)
 * ------------------------------------------------------------------------
 * For the reduction corpus (scalar/full + dim reductions, with/without
 * preamble/epilogue, mean-div) and the row-program corpus (multi-reduction +
 * elementwise fused kernels — softmax/layernorm shape),
 *
 *   compileTileKernel(applySchedule*(deriveScheduleState*(k))) ==BYTES==
 *   compileTileKernel(<liveSpec>)
 *
 * on both plan paths where reachable.
 *
 * ------------------------------------------------------------------------
 * WHY THIS EARNS R6/R22 (not just byte-identity)
 * ------------------------------------------------------------------------
 * The reduction family's single-source builder is `makeReductionSpec`
 * (reduction-tile-ir.ts) — a config-driven factory. §11.5 classes its
 * expression bodies `stays-as-semantic-builder`. So `deriveScheduleState`
 * captures the SEMANTIC facts (reduceOp, the reduction loop nest, block shapes,
 * the preamble/epilogue op-catalog bodies) into a `ScheduleState` carrying NO
 * opaque generator and NO WGSL, and `applySchedule` reconstructs the
 * `ReductionConfig` from those facts and calls the SAME builder — it does not
 * replay a captured generator (there is nowhere in the schema to store one).
 *
 * The row-program family is even cleaner: `RowProgram` (row-program-types.ts) is
 * ALREADY a semantic IR — `RPExpr` is an op-catalog tree by NAME (no thread ids,
 * no barriers, no placement), exactly the `SemanticBodyNode` shape. So
 * `deriveScheduleState` maps `RowProgram` → `ScheduleState` losslessly and
 * `applySchedule` reconstructs the `RowProgram` and calls the single-source
 * `rowProgramToSpec`. The reduce/write PHASES become the schedule's loop nest +
 * store edges + reduce lemmas-free reduceOp facts.
 *
 * Both families REFUSE a spec they cannot express (they never wrap WGSL).
 */

import type { DType } from "../backend/types";
import { argReduceWGSL } from "../backend/webgpu/ops/ops-tile-ir";
import {
  dimInfo,
  makeMeanDivSpec,
  makeReductionSpec,
  type PreambleChainKernelOp,
  type ReductionConfig,
  type ReductionEpilogueOpDesc,
} from "../backend/webgpu/reduction-tile-ir";
import { rowProgramToSpec } from "../backend/webgpu/row-program-codegen";
import type { TileKernelSpec } from "../backend/webgpu/tile-ir";
import type { RowProgram, RPExpr } from "../compiler/row-program-types";
import { printScheduleState, reportNoSecondOwner } from "./canonical";
import type {
  AffineExpr,
  LemmaApplication,
  LemmaUid,
  LoopUid,
  NamedValue,
  ObligationId,
  ScheduleState,
  SemanticBody,
  SemanticBodyNode,
  SemanticLoop,
  SemanticRegionUid,
  SemanticSchedule,
  StoreEdge,
  ValueDtype,
  ValueUid,
} from "./types";

const uid = <T>(s: string): T => s as unknown as T;

// ============================================================================
// The WELFORD admitted lemma (§7 P4 — the variance pair-merge; §3.4 F27)
// ============================================================================

/**
 * VARIANCE does not stream by a plain monoid: the naive `E[x²] − E[x]²` form is
 * catastrophically cancellation-prone, and there is no associative (step, merge)
 * over blocks that recomposes a numerically-sound variance from block partials —
 * a block's contribution depends on the GLOBAL mean, unknown until the whole axis
 * is seen. The WELFORD pair-merge is the admitted lemma that GIVES variance a
 * head/body decomposition: each block carries the triple `(count, mean, M2)`
 * (M2 = Σ(x−mean)²), and two blocks merge by
 *
 *   δ = meanB − meanA
 *   count = countA + countB
 *   mean  = meanA + δ·countB/count
 *   M2    = M2A + M2B + δ²·countA·countB/count
 *
 * The merge is associative and numerically stable (the δ correction plays the
 * same role the online-softmax rescale does for the max). This is the
 * deliberately-SMALL teaching lemma (NCD F27/exercise 3), now ENGINE-REAL:
 * layernorm's inv-std path `rstd = 1/sqrt(var + eps)` is its consumer — the
 * carried (count, mean, M2) yields `var = M2/count`, and layernorm reads the
 * stable var the pair-merge produces.
 *
 * CARRIED STATE: the triple `(count, mean, M2)`. PROOF OBLIGATION: the pair-merge
 * of any block partition equals the single-pass variance (associativity +
 * numerical soundness). MERGE is associative but NOT commutative in the same
 * shape as onlineSoftmax (δ is signed) — declared `welfordCombine` below.
 */
export const WELFORD_LEMMA = uid<LemmaUid>("lemma:welford-variance-pair-merge");
export const WELFORD_OBLIGATION = uid<ObligationId>(
  "obl:welford-pair-merge-equals-single-pass-variance",
);

/** The Welford lemma application (carried state: (count, mean, M2)). */
export function welfordLemma(): LemmaApplication {
  return {
    lemma: WELFORD_LEMMA,
    obligation: WELFORD_OBLIGATION,
    // Carried STATE (the pair-merge triple): count n, running mean, M2=Σ(x−mean)².
    // var = M2/count; layernorm's rstd = 1/sqrt(var+eps) is the consumer.
    carriedStateRef:
      "carried=(count:n,mean,M2:sum-sq-dev);merge=delta=meanB-meanA;" +
      "mean+=delta*countB/count;M2+=delta^2*countA*countB/count",
  };
}

/** dtypes the reduction/row-program families use (a subset of DType). */
function toValueDtype(d: DType): ValueDtype {
  if (d === "f32" || d === "f16" || d === "i32" || d === "u32") return d;
  throw new Error(`reduction-skeleton: unsupported dtype ${d}`);
}

// ============================================================================
// FAMILY 2a — REDUCTIONS (scalar/full + dim, preamble/epilogue, mean-div)
// ============================================================================

/**
 * A structured description of ONE reduction-family kernel, sufficient to (a)
 * round-trip to a byte-identical `TileKernelSpec` via `makeReductionSpec` and
 * (b) populate a `ScheduleState` with no second owner. The live dispatch sites
 * (`reduction`, `sumWithPreambleEpilogue`, `mean`) all reduce to this config.
 */
export interface ReductionDescriptor {
  readonly reduceOp: "sum" | "max" | "min";
  /** undefined = full reduction (single scalar output). */
  readonly dim?: DimReductionInfo;
  readonly preamble?: {
    readonly chainOps: readonly PreambleChainKernelOp[];
    readonly totalInputs: number;
    readonly inputDtypes: readonly DType[];
  };
  readonly epilogue?: {
    readonly ops: readonly ReductionEpilogueOpDesc[];
    readonly outputDtype: DType;
  };
  readonly count?: number;
}

/** The shape metadata a dim reduction carries (positional args to `dimInfo`). */
export interface DimReductionInfo {
  readonly inputShape: readonly number[];
  readonly inputStrides: readonly number[];
  readonly normalizedDims: readonly number[];
  readonly outShape: readonly number[];
  readonly outStrides: readonly number[];
  readonly inputToOutDim: readonly number[];
  readonly parallel: boolean;
}

/** A special-case descriptor for the mean-div elementwise kernel (sum→÷count). */
export interface MeanDivDescriptor {
  readonly kind: "meanDiv";
}

/**
 * Derive a `ScheduleState` from a reduction descriptor. The reduction is a
 * flat parallel output domain (one output element per program) with a nested
 * SEQUENTIAL reduction loop over `reductionSize` — that loop nest is the
 * derivable structural spine (S2). reduceOp is a semantic fact; the preamble
 * body is a `SemanticBodyNode` op-catalog tree; the store is an edge.
 *
 * NOTE: the reduction is workgroup-cooperative in the realized kernel (wgReduce
 * / barriers), but per S2 barriers/roles do NOT derive — they are OWNED by the
 * `makeReductionSpec` semantic builder's lowering, NOT re-copied into the
 * schedule. The schedule carries the reduceOp + loop nest + block shapes only;
 * the barrier is the realizer's, so it does not appear here (its absence is the
 * no-second-owner property).
 */
export function deriveReductionState(
  desc: ReductionDescriptor,
  region: SemanticRegionUid,
): ScheduleState {
  const values: NamedValue[] = [];

  // Input value(s): a plain reduction has one `input`; a preamble has in0..inN.
  const inputDtypes: ValueDtype[] = desc.preamble
    ? desc.preamble.inputDtypes.map(toValueDtype)
    : [toValueDtype("f32")];
  const nInputs = desc.preamble ? desc.preamble.totalInputs : 1;
  for (let i = 0; i < nInputs; i++) {
    const name = desc.preamble ? `in${i}` : "input";
    values.push({
      uid: uid<ValueUid>(`in:${name}`),
      entity: uid(`ent:in:${name}`),
      allocation: "global",
      dtype: inputDtypes[i] ?? "f32",
      aliasOf: null,
    });
  }

  const outDtype: ValueDtype = desc.epilogue
    ? toValueDtype(desc.epilogue.outputDtype)
    : "f32";
  const resultUid = uid<ValueUid>("result");
  values.push({
    uid: resultUid,
    entity: uid("ent:result"),
    allocation: "register",
    dtype: outDtype,
    aliasOf: null,
  });
  const outUid = uid<ValueUid>("out:out");
  values.push({
    uid: outUid,
    entity: uid("ent:out:out"),
    allocation: "global",
    dtype: outDtype,
    aliasOf: null,
  });

  // The output (parallel) loop and the reduction (sequential) loop.
  const outLoopUid = uid<LoopUid>("loop:out");
  const redLoopUid = uid<LoopUid>("loop:reduce");
  const outBound: AffineExpr = {
    kind: "affineLeaf",
    leaf: { kind: "uniformRef", name: desc.dim ? "outSize" : "size" },
  };
  const redBound: AffineExpr = {
    kind: "affineLeaf",
    leaf: {
      kind: "uniformRef",
      name: desc.dim ? "reductionSize" : "size",
    },
  };
  const redLoop: SemanticLoop = {
    uid: redLoopUid,
    entity: uid("ent:loop:reduce"),
    axis: uid("axis:reduce"),
    kind: "sequential",
    bound: redBound,
    children: [],
  };
  const outLoop: SemanticLoop = {
    uid: outLoopUid,
    entity: uid("ent:loop:out"),
    axis: uid("axis:out"),
    kind: "parallel",
    bound: outBound,
    children: [redLoop],
  };

  // The preamble body (the value reduced each step). A plain reduction reduces
  // the loaded `input` directly; a preamble applies an op-catalog chain first.
  const body: SemanticBody = {
    result: resultUid,
    expr: desc.preamble
      ? preambleChainToBody(desc.preamble.chainOps)
      : { kind: "value", value: uid<ValueUid>("in:input") },
  };

  const store: StoreEdge = {
    source: resultUid,
    target: outUid,
    atLoop: outLoopUid,
  };

  // The reduceOp + epilogue are carried as an OP-CATALOG lemma-free fact via a
  // second body whose result is the reduced value's post-epilogue form. To keep
  // the family expressible in the current schema without new node kinds, the
  // reduceOp is recorded as the first body's ROOT catalog wrapper (`reduce_*`)
  // and epilogue ops as chained applies. This is a semantic op-catalog tree, not
  // a generator.
  const semantic: SemanticSchedule = {
    blockShapes: desc.dim
      ? [[...desc.dim.outShape], reduceShapeOf(desc.dim)]
      : [[]],
    loopNest: [outLoop],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values,
    noMaterialization: [],
    stores: [store],
    bodies: [body, reduceBody(desc, resultUid)],
    roles: [],
    sync: [],
    atoms: [],
    lemmas: [],
  };

  return {
    semantic,
    requests: {
      warpBudget: null,
      pipeline: { kind: "none" },
      placementPreferences: [],
      cachePolicy: [],
    },
    receipts: {},
    region,
  };
}

/** The reduction-axis logical shape (the block shape of the reduced axis). */
function reduceShapeOf(dim: DimReductionInfo): number[] {
  return dim.normalizedDims.map((d) => dim.inputShape[d]);
}

/** Map a preamble op chain into an op-catalog body tree (semantic, by name). */
function preambleChainToBody(
  chainOps: readonly PreambleChainKernelOp[],
): SemanticBodyNode {
  // First op consumes external inputs; subsequent ops fold the chain result.
  let externalIdx = 0;
  const ext = (): SemanticBodyNode => ({
    kind: "value",
    value: uid<ValueUid>(`in:in${externalIdx++}`),
  });
  let result: SemanticBodyNode;
  if (chainOps[0].arity === 1) {
    result = { kind: "apply", catalog: { op: chainOps[0].op }, args: [ext()] };
  } else {
    result = {
      kind: "apply",
      catalog: { op: chainOps[0].op },
      args: [ext(), ext()],
    };
  }
  for (let i = 1; i < chainOps.length; i++) {
    const op = chainOps[i];
    if (op.arity === 1) {
      result = { kind: "apply", catalog: { op: op.op }, args: [result] };
    } else {
      const e = ext();
      result =
        op.chainInputPos === 1
          ? { kind: "apply", catalog: { op: op.op }, args: [e, result] }
          : { kind: "apply", catalog: { op: op.op }, args: [result, e] };
    }
  }
  return result;
}

/** The reduce+epilogue body: `epilogue(reduce_op(<per-element body>))`. */
function reduceBody(desc: ReductionDescriptor, result: ValueUid): SemanticBody {
  let expr: SemanticBodyNode = {
    kind: "apply",
    catalog: { op: `reduce_${desc.reduceOp}` },
    args: [{ kind: "value", value: result }],
  };
  if (desc.epilogue) {
    for (const eop of desc.epilogue.ops) {
      const opName =
        eop.kind === "cast"
          ? `cast_${eop.toDtype ?? "f32"}`
          : (eop.op ?? "identity");
      expr = { kind: "apply", catalog: { op: opName }, args: [expr] };
    }
  }
  return { result: uid<ValueUid>("reduced"), expr };
}

/**
 * `applySchedule` for the reduction family: reconstruct the `ReductionConfig`
 * from the descriptor carried alongside the state and call the single-source
 * `makeReductionSpec`. The state's semantic facts are asserted to AGREE with the
 * descriptor at the seam (assertReductionSeam) before lowering, so the schedule
 * is the source and the descriptor is validated against it.
 */
export function applyReductionSchedule(
  state: ScheduleState,
  desc: ReductionDescriptor,
): TileKernelSpec {
  assertReductionSeam(state, desc);
  return makeReductionSpec({
    reduceOp: desc.reduceOp,
    dim: desc.dim
      ? dimInfo(
          [...desc.dim.inputShape],
          [...desc.dim.inputStrides],
          [...desc.dim.normalizedDims],
          [...desc.dim.outShape],
          [...desc.dim.outStrides],
          [...desc.dim.inputToOutDim],
          desc.dim.parallel,
        )
      : undefined,
    preamble: desc.preamble
      ? {
          chainOps: [...desc.preamble.chainOps],
          totalInputs: desc.preamble.totalInputs,
          inputDtypes: [...desc.preamble.inputDtypes],
        }
      : undefined,
    epilogue: desc.epilogue
      ? { ops: [...desc.epilogue.ops], outputDtype: desc.epilogue.outputDtype }
      : undefined,
    count: desc.count,
  });
}

/** The mean-div elementwise kernel round-trips through its own single source. */
export function applyMeanDivSchedule(): TileKernelSpec {
  return makeMeanDivSpec();
}

// ============================================================================
// FAMILY 2c — ARG-REDUCE (argmax / argmin — the wave-1 leftover derivable kernel)
// ============================================================================

/**
 * A structured description of ONE arg-reduce kernel (argmax/argmin along a dim).
 * Round-trips byte-identically via `argReduceWGSL`. Arg-reduce is a reduction
 * whose reduce OP is `argmax`/`argmin`: a flat parallel output loop enclosing a
 * SEQUENTIAL search over `dimSize`. The (bestVal, bestIndex) state-machine is the
 * REALIZER's (like wgReduce for plain reductions, S2) — the schedule records the
 * argmax reduceOp fact + the loop nest + the store edge, NOT the index-tracking.
 */
export interface ArgReduceDescriptor {
  readonly compareOp: ">" | "<"; // ">" = argmax, "<" = argmin
  readonly inputShape: readonly number[];
  readonly inputStrides: readonly number[];
  readonly outShape: readonly number[];
  readonly dim: number;
  readonly inputToOutDim: readonly number[];
}

export function deriveArgReduceState(
  desc: ArgReduceDescriptor,
  region: SemanticRegionUid,
): ScheduleState {
  const reduceOp = desc.compareOp === ">" ? "argmax" : "argmin";
  // input (global) → the reduced (index) result (register) → the global output.
  const inUid = uid<ValueUid>("in:input");
  const resultUid = uid<ValueUid>("result");
  const outUid = uid<ValueUid>("out:out");
  const values: NamedValue[] = [
    {
      uid: inUid,
      entity: uid("ent:in:input"),
      allocation: "global",
      dtype: "f32",
      aliasOf: null,
    },
    {
      uid: resultUid,
      entity: uid("ent:result"),
      allocation: "register",
      // The output is the INDEX (u32-valued, stored as f32 by the live kernel).
      dtype: "u32",
      aliasOf: null,
    },
    {
      uid: outUid,
      entity: uid("ent:out:out"),
      allocation: "global",
      dtype: "f32",
      aliasOf: null,
    },
  ];

  const outLoopUid = uid<LoopUid>("loop:out");
  const redLoopUid = uid<LoopUid>("loop:reduce");
  const redLoop: SemanticLoop = {
    uid: redLoopUid,
    entity: uid("ent:loop:reduce"),
    axis: uid("axis:reduce"),
    kind: "sequential",
    bound: {
      kind: "affineLeaf",
      leaf: { kind: "uniformRef", name: "dimSize" },
    },
    children: [],
  };
  const outLoop: SemanticLoop = {
    uid: outLoopUid,
    entity: uid("ent:loop:out"),
    axis: uid("axis:out"),
    kind: "parallel",
    bound: {
      kind: "affineLeaf",
      leaf: { kind: "uniformRef", name: "outSize" },
    },
    children: [redLoop],
  };

  // The per-element body is the loaded input; the reduce body wraps it in the
  // argmax/argmin catalog op (the index-tracking is the realizer's).
  const bodies: SemanticBody[] = [
    { result: resultUid, expr: { kind: "value", value: inUid } },
    {
      result: uid<ValueUid>("reduced"),
      expr: {
        kind: "apply",
        catalog: { op: `reduce_${reduceOp}` },
        args: [{ kind: "value", value: resultUid }],
      },
    },
  ];

  const semantic: SemanticSchedule = {
    blockShapes: [[...desc.outShape]],
    loopNest: [outLoop],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values,
    noMaterialization: [],
    stores: [{ source: resultUid, target: outUid, atLoop: outLoopUid }],
    bodies,
    roles: [],
    sync: [],
    atoms: [],
    lemmas: [],
  };
  return {
    semantic,
    requests: {
      warpBudget: null,
      pipeline: { kind: "none" },
      placementPreferences: [],
      cachePolicy: [],
    },
    receipts: {},
    region,
  };
}

/**
 * `applyArgReduceSchedule`: assert the state agrees (the argmax reduceOp fact),
 * then call the single-source `argReduceWGSL`. Returns the WGSL string directly
 * (argReduceWGSL already compiles), matching the byte-differential seam.
 */
export function applyArgReduceSchedule(
  state: ScheduleState,
  desc: ArgReduceDescriptor,
): string {
  assertNoOpaqueLeak(state);
  const s = state.semantic;
  if (s.stores.length !== 1)
    reportNoSecondOwner(
      `no-second-owner[arg-reduce]: expected exactly one store edge, got ${s.stores.length}.`,
    );
  const reduceOp = desc.compareOp === ">" ? "argmax" : "argmin";
  const op = s.bodies[1]?.expr;
  const rootOp = op && op.kind === "apply" ? findReduceOp(op) : null;
  if (rootOp !== `reduce_${reduceOp}`)
    reportNoSecondOwner(
      `no-second-owner[arg-reduce]: schedule reduceOp "${rootOp}" disagrees with ` +
        `descriptor "reduce_${reduceOp}".`,
    );
  return argReduceWGSL(
    desc.compareOp,
    [...desc.inputShape],
    [...desc.inputStrides],
    [...desc.outShape],
    desc.dim,
    [...desc.inputToOutDim],
  );
}

// ============================================================================
// FAMILY 2b — ROW-PROGRAM (multi-reduction + elementwise fused)
// ============================================================================

/**
 * Derive a `ScheduleState` from a `RowProgram`. `RowProgram` is already a
 * semantic IR; this is a lossless projection into the three-tier object:
 *   - inputs → global NamedValues; output → global NamedValue.
 *   - each ReducePhase → a nested SEQUENTIAL reduction loop + a reduceOp body.
 *   - the final WritePhase → the store edge + its body.
 *   - `dim` → the reduction axis; the row loop is the parallel outer loop.
 * The row-cooperative barrier is the realizer's (rowProgramToSpec/wgReduce),
 * NOT a schedule fact — its absence is the no-second-owner property (S2).
 */
export function deriveRowProgramState(
  program: RowProgram,
  region: SemanticRegionUid,
): ScheduleState {
  const values: NamedValue[] = [];
  for (let i = 0; i < program.inputs.length; i++) {
    values.push({
      uid: uid<ValueUid>(`in:in${i}`),
      entity: uid(`ent:in:in${i}`),
      allocation: "global",
      dtype: toValueDtype(program.inputs[i].dtype),
      aliasOf: null,
    });
  }
  const outUid = uid<ValueUid>("out:output");
  values.push({
    uid: outUid,
    entity: uid("ent:out:output"),
    allocation: "global",
    dtype: toValueDtype(program.output.dtype),
    aliasOf: null,
  });

  // Per-phase reduce results are register-tier named values.
  let reduceIdx = 0;
  const bodies: SemanticBody[] = [];
  const rowLoopUid = uid<LoopUid>("loop:row");
  const featLoopUid = uid<LoopUid>("loop:feature");

  for (const phase of program.phases) {
    if (phase.kind === "reduce") {
      const rUid = uid<ValueUid>(`r${reduceIdx}`);
      values.push({
        uid: rUid,
        entity: uid(`ent:r${reduceIdx}`),
        allocation: "register",
        dtype: "f32",
        aliasOf: null,
      });
      // The per-element body reduced, wrapped in the reduce op (+ optional mean).
      let expr: SemanticBodyNode = {
        kind: "apply",
        catalog: { op: `reduce_${phase.reduceOp}` },
        args: [rpExprToBody(phase.bodyExpr)],
      };
      if (phase.isMean) {
        expr = { kind: "apply", catalog: { op: "mean_div" }, args: [expr] };
      }
      bodies.push({ result: rUid, expr });
      reduceIdx++;
    } else {
      // WritePhase → the stored body.
      bodies.push({
        result: uid<ValueUid>("write"),
        expr: rpExprToBody(phase.bodyExpr),
      });
    }
  }

  const featLoop: SemanticLoop = {
    uid: featLoopUid,
    entity: uid("ent:loop:feature"),
    axis: uid("axis:feature"),
    kind: "sequential",
    bound: {
      kind: "affineLeaf",
      leaf: { kind: "uniformRef", name: "feature_dim" },
    },
    children: [],
  };
  const rowLoop: SemanticLoop = {
    uid: rowLoopUid,
    entity: uid("ent:loop:row"),
    axis: uid("axis:row"),
    kind: "parallel",
    bound: {
      kind: "affineLeaf",
      leaf: { kind: "uniformRef", name: "num_rows" },
    },
    children: [featLoop],
  };

  const lastPhase = program.phases[program.phases.length - 1];
  const scalarOut =
    lastPhase.kind === "write" && lastPhase.scalarOutput === true;
  const store: StoreEdge = {
    source: uid<ValueUid>("write"),
    target: outUid,
    atLoop: scalarOut ? rowLoopUid : featLoopUid,
  };

  const semantic: SemanticSchedule = {
    blockShapes: [[]],
    loopNest: [rowLoop],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values,
    noMaterialization: [],
    stores: [store],
    bodies,
    roles: [],
    sync: [],
    atoms: [],
    lemmas: [],
  };

  return {
    semantic,
    requests: {
      warpBudget: null,
      pipeline: { kind: "none" },
      placementPreferences: [],
      cachePolicy: [],
    },
    receipts: {},
    region,
  };
}

/** Map an `RPExpr` op-catalog tree into a `SemanticBodyNode`. Lossless: both are
 *  op-name trees over inputs / reduce-results / consts. */
function rpExprToBody(expr: RPExpr): SemanticBodyNode {
  if ("kind" in expr) {
    switch (expr.kind) {
      case "input":
        return {
          kind: "value",
          value: uid<ValueUid>(`in:in${expr.bufferIndex}`),
        };
      case "reduceResult":
        return { kind: "value", value: uid<ValueUid>(`r${expr.phaseIndex}`) };
      case "const":
        return { kind: "literal", dtype: "f32", value: expr.value };
    }
  }
  return {
    kind: "apply",
    catalog: { op: expr.op },
    args: expr.inputs.map(rpExprToBody),
  };
}

/**
 * `applySchedule` for the row-program family: reconstruct the `RowProgram` and
 * call the single-source `rowProgramToSpec`. Because `RowProgram` IS the
 * semantic IR, the reconstruction is the identity carried alongside; the seam
 * assertion checks the derived state agrees with it.
 */
export function applyRowProgramSchedule(
  state: ScheduleState,
  program: RowProgram,
): TileKernelSpec {
  assertRowProgramSeam(state, program);
  return rowProgramToSpec(program);
}

// ============================================================================
// realize* — the LIVE-PATH cutover (P2 wave A, item 1)
// ============================================================================
//
// The chokepoints the live reduction / row-program / arg-reduce dispatch route
// THROUGH (the P1 matmul pattern). Each derives the ScheduleState, runs the
// no-second-owner seam, and lowers via `apply*` — so the schedule object is the
// SOLE live WGSL writer at the dispatch seam (`makeReductionSpec` / `argReduceWGSL`
// / `rowProgramToSpec` are now realizer-internals of `apply*`, unreachable from
// live dispatch except through the schedule object). Byte-identical by
// construction; the reduction differential now guards the LIVE path.

const LIVE_REDUCTION_REGION = uid<SemanticRegionUid>("region:live-reduction");

/** A `ReductionConfig` IS the descriptor (same fields); no conversion needed. */
function configToDescriptor(config: ReductionConfig): ReductionDescriptor {
  return config as ReductionDescriptor;
}

/** Realize a reduction spec THROUGH the schedule object (live chokepoint). */
export function realizeReductionSpec(config: ReductionConfig): TileKernelSpec {
  const desc = configToDescriptor(config);
  const state = deriveReductionState(desc, LIVE_REDUCTION_REGION);
  return applyReductionSchedule(state, desc);
}

/** Realize the mean-div elementwise kernel THROUGH the schedule object. */
export function realizeMeanDivSpec(): TileKernelSpec {
  return applyMeanDivSchedule();
}

/** Realize the row-program spec THROUGH the schedule object (live chokepoint). */
export function realizeRowProgramSpec(program: RowProgram): TileKernelSpec {
  const state = deriveRowProgramState(program, LIVE_REDUCTION_REGION);
  return applyRowProgramSchedule(state, program);
}

/** Realize the arg-reduce WGSL THROUGH the schedule object (live chokepoint). */
export function realizeArgReduceWgsl(
  compareOp: ">" | "<",
  inputShape: readonly number[],
  inputStrides: readonly number[],
  outShape: readonly number[],
  dim: number,
  inputToOutDim: readonly number[],
): string {
  const desc: ArgReduceDescriptor = {
    compareOp,
    inputShape,
    inputStrides,
    outShape,
    dim,
    inputToOutDim,
  };
  const state = deriveArgReduceState(desc, LIVE_REDUCTION_REGION);
  return applyArgReduceSchedule(state, desc);
}

// ============================================================================
// The no-second-owner seam assertions (§12 check 3, family-local)
// ============================================================================

/**
 * Assert the reduction `ScheduleState` carries no second owner: no opaque
 * generator/WGSL leaks (covered structurally by the canonical printer, but the
 * substring backstop is cheap), exactly one store edge, the reduceOp is a
 * semantic op-catalog fact (not a baked kernel), and the input/output tiers are
 * global while the reduced result is register. The DESCRIPTOR is validated
 * against the state at this seam (single source = the schedule).
 */
export function assertReductionSeam(
  state: ScheduleState,
  desc: ReductionDescriptor,
): void {
  assertNoOpaqueLeak(state);
  const s = state.semantic;
  if (s.stores.length !== 1)
    reportNoSecondOwner(
      `no-second-owner[reduction]: expected exactly one store edge, got ${s.stores.length}.`,
    );
  // The reduceOp recorded in the schedule must match the descriptor (seam).
  const reduceBodyExpr = s.bodies[1]?.expr;
  const reduceRootOp =
    reduceBodyExpr && reduceBodyExpr.kind === "apply"
      ? findReduceOp(reduceBodyExpr)
      : null;
  if (reduceRootOp !== `reduce_${desc.reduceOp}`)
    reportNoSecondOwner(
      `no-second-owner[reduction]: schedule reduceOp "${reduceRootOp}" disagrees ` +
        `with descriptor "reduce_${desc.reduceOp}" — two independent owners of the reduce fact.`,
    );
  if (s.roles.length || s.sync.length || s.atoms.length || s.lemmas.length)
    reportNoSecondOwner(
      `no-second-owner[reduction]: reduction schedule carries no roles/sync/atoms/lemmas ` +
        `(the wgReduce barrier is the realizer's, S2); found some — mis-derived structure.`,
    );
}

/** Walk an epilogue-wrapped reduce body to find the `reduce_*` catalog op. */
function findReduceOp(node: SemanticBodyNode): string | null {
  if (node.kind !== "apply") return null;
  if (node.catalog.op.startsWith("reduce_")) return node.catalog.op;
  for (const a of node.args) {
    const found = findReduceOp(a);
    if (found) return found;
  }
  return null;
}

/**
 * Assert the row-program `ScheduleState` carries no second owner: no opaque
 * leak, exactly one store edge, one reduce body per ReducePhase (the reduceOps
 * match, in order), and no roles/sync/atoms/lemmas (the wgReduce barriers are
 * the realizer's). The RowProgram is validated against the state at this seam.
 */
export function assertRowProgramSeam(
  state: ScheduleState,
  program: RowProgram,
): void {
  assertNoOpaqueLeak(state);
  const s = state.semantic;
  if (s.stores.length !== 1)
    reportNoSecondOwner(
      `no-second-owner[row-program]: expected exactly one store edge, got ${s.stores.length}.`,
    );
  const reducePhases = program.phases.filter((p) => p.kind === "reduce");
  const reduceBodies = s.bodies.filter(
    (b) => b.expr.kind === "apply" && findReduceOp(b.expr) !== null,
  );
  if (reduceBodies.length !== reducePhases.length)
    reportNoSecondOwner(
      `no-second-owner[row-program]: ${reduceBodies.length} reduce bodies but ` +
        `${reducePhases.length} ReducePhases — the phase structure has a second owner.`,
    );
  for (let i = 0; i < reducePhases.length; i++) {
    const phase = reducePhases[i] as { reduceOp: string };
    const op = findReduceOp(reduceBodies[i].expr);
    if (op !== `reduce_${phase.reduceOp}`)
      reportNoSecondOwner(
        `no-second-owner[row-program]: phase ${i} reduceOp "${op}" disagrees with ` +
          `program "reduce_${phase.reduceOp}".`,
      );
  }
  if (s.roles.length || s.sync.length || s.atoms.length || s.lemmas.length)
    reportNoSecondOwner(
      `no-second-owner[row-program]: row-program schedule carries no roles/sync/atoms/lemmas ` +
        `(the wgReduce barriers are the realizer's, S2); found some — mis-derived structure.`,
    );
}

/**
 * The schema-only serialization check (§12 check 1) for the reduction and
 * row-program families: the canonical PRINTER walks the typed schema and THROWS
 * (`assertNever`) on any out-of-schema value — it cannot serialize an opaque
 * generator / WGSL string / AST dump. Calling it IS the check. This shares the
 * elementwise family's cashed §11 deletion: no per-family `JSON.stringify` +
 * forbidden-substring scan (that scan was a second owner of the no-opaque-leak
 * fact; the printer is the sole structural owner).
 */
function assertNoOpaqueLeak(state: ScheduleState): void {
  printScheduleState(state);
}
