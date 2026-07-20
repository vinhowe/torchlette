/**
 * Execution Declaration: the command-stream stratum as DATA
 * (docs/execution-declaration-design.md — the seventh application of "the
 * latent decision becomes an object", the direct successor to schedule-state).
 *
 * An `ExecutionDeclaration` describes how one op-family node decomposes into an
 * ordered sequence of GPU commands, over ROLES (`input[i]` | `output` |
 * `params`) — NEVER a `GPUBuffer` — from which BOTH consumers derive:
 *   - the live dispatch is the INTERPRETER (binds live pooled buffers, encodes),
 *   - the compiled `GpuCommand[]` is the SERIALIZER (emits slot indices).
 * So the mirror between `dispatch.ts` and `stream-generate.ts` — two dialects of
 * the ONE fact "the command sequence" — stops existing.
 *
 * ## P0 scope — the schema + the ELEMENTWISE family
 * This file lands the schema and the elementwise family authored as data. The
 * elementwise command shape is uniform: ALLOC(output) → [UNIFORM(params) when
 * the config packer is volatile] → DISPATCH(kernel, bind=[…inputs, output,
 * params]).
 *
 * ## Decomposition is a WALKER transform, NOT a schema field (Vin's P0 refinement)
 * Chunking (>maxStorageBufferBindingSize) is NOT a per-declaration or per-family
 * concern. It is TRANSPORT WINDOWING: when the oversized operand's split axis is
 * PARALLEL — which for elementwise is EVERY axis — windowing is a universal
 * transform `windows = f(operand bytes, device binding limit, 256-byte
 * alignment)` applied identically by the interpreter and the serializer, with
 * window offsets/sizes as volatile uniforms. The declaration NEVER mentions it.
 * Parallelism is DERIVED (`ELEMENTWISE_SPLIT_AXIS_PARALLEL`), not declared;
 * oversized on a CARRIED axis (reductions) is a typed walker refusal —
 * schedule-state's territory (stream/kSplit + monoid obligations), not this
 * stratum's. Transform ordering: the layout prologue (`CONTIGUOUS_OPERANDS`,
 * generalized by `layout`) runs BEFORE windowing; co-bound operands get
 * co-partitioned windows from the shared index map. This supersedes the design
 * doc's "derived geometry carried in the declaration" phrasing.
 *
 * ## The schema gate (structural, not hopeful)
 * A `KernelRef` is a discriminated TAG, resolved to WGSL + workgroup geometry +
 * params by the realizer (`planBinaryDirect` / `planUnaryDirect` / …) — the
 * ScheduleState-realization analogue, one altitude down. No leaf of a
 * declaration can hold a `(node) => GpuCommand[]` generator, a callback, a
 * WGSL string, or a buffer. `assertNoGeneratorLeaf` walks a declaration and
 * proves it: an adapter that smuggles the old generator behind an opaque leaf
 * is UNCONSTRUCTIBLE, so "one source" is guaranteed by construction, not by the
 * differential passing.
 */

import {
  MAX_DEF,
  MEAN_DEF,
  MIN_DEF,
  type ReductionDef,
  reduceMonoidOf,
  SUM_DEF,
} from "../ops/semantic/reduction";

// ============================================================================
// The schema
// ============================================================================

/** Op-family identity. P0 declared `elementwise`; P1 adds `reduction`; P2 adds
 *  `matmul` (the matmul + matmul-epilogue family — the biggest duplication
 *  pair). */
export type OpFamilyUid = "elementwise" | "reduction" | "matmul";

/**
 * Per-operand role, resolved per-NODE from the input ref kind:
 *  - `bind`   : a tensor bound as a storage buffer (its strides/offset are
 *               folded into the WGSL — elementwise never requires contiguity).
 *  - `scalar` : a scalar-table PERSISTENT buffer (stable identity, value
 *               refreshed from the current step; NOT a stream write).
 */
export type OperandRole = "bind" | "scalar";

/**
 * A slot ROLE named by the skeleton. Never a `GPUBuffer`: the interpreter binds
 * a live pooled buffer to it, the serializer emits a slot index. `all-inputs`
 * expands to `input[0..arity-1]` in order (keeps the skeleton arity-agnostic —
 * unary/binary/where share one skeleton).
 */
export type SkeletonRole =
  | { role: "input"; index: number }
  | { role: "all-inputs" }
  | { role: "output" }
  | { role: "params" };

/**
 * The kernel a dispatch realizes — a TYPED REFERENCE, resolved to WGSL +
 * geometry + params by the realizer. NEVER an inline generator (the schema
 * gate). Discriminated by `kernel`; carries no callback and no WGSL.
 */
export type KernelRef =
  | { kernel: "binaryDirect" }
  | { kernel: "unaryDirect" }
  | { kernel: "castDirect" }
  | { kernel: "whereDirect" }
  | { kernel: "contiguousDirect" };

/**
 * One command in the ordered skeleton, over roles. Each maps 1:1 to a
 * `GpuCommand` tag at realize time (`alloc`→TAG_ALLOC, `uniform`→TAG_UNIFORM,
 * `dispatch`→TAG_DISPATCH). No leaf holds a buffer, a callback, or WGSL.
 */
export type SkeletonCommand =
  | {
      op: "alloc";
      /** The slot to allocate (always `output` for elementwise). */
      slot: SkeletonRole;
      /**
       * Which operand roles are pool-aliasing candidates for
       * `resolveOutputBuffer` (the `inputSlots` of TAG_ALLOC).
       * `poolable-binds` = every `bind` operand (scalar-table buffers are
       * persistent and excluded by `resolveOutputBuffer`).
       */
      aliasOperands: "poolable-binds";
    }
  | {
      /**
       * Re-pack the config buffer from the CURRENT step's node. The realizer
       * emits this ONLY when the config packer is VOLATILE (an input offset can
       * vary across replays — a narrow-fed view); volatility is a per-node fact
       * the realizer decides, not a declared constant. Present in the skeleton
       * to fix the ORDER (before the dispatch, matching `createParamsBuffer`).
       */
      op: "uniform";
      slot: SkeletonRole;
    }
  | {
      op: "dispatch";
      kernel: KernelRef;
      /** Bind-group order. Elementwise: `[all-inputs, output, params]`. */
      bindings: SkeletonRole[];
    };

/**
 * Whether the family's bindable operands must be raw-bindable (contiguous +
 * offset 0) before binding. Elementwise DIRECT is `strided-ok` (strides folded
 * into the WGSL); the `size-split` (chunked) decomposition binds sub-ranges
 * from element 0 and so requires `contiguous-offset0` — a DECOMPOSITION-local
 * requirement, checked by the realizer against live layout (generalizes
 * `CONTIGUOUS_OPERANDS`, which has no elementwise entries precisely because the
 * direct path folds strides).
 */
export type LayoutRequirement = "strided-ok" | "contiguous-offset0";

export interface ExecutionDeclaration {
  family: OpFamilyUid;
  /** Fixed arity; a node with a different input count is a typed refusal. */
  arity: number;
  /** How the kernel is realized (WGSL + geometry + params). */
  kernel: KernelRef;
  /** Layout requirement of the DIRECT path (windowing tightens it to
   *  `contiguous-offset0` at the walker — the prologue runs BEFORE windowing). */
  layout: LayoutRequirement;
  /** The ordered command shape, over roles. Shared family template. Carries NO
   *  decomposition field — chunking is a walker transform (see file header). */
  skeleton: SkeletonCommand[];
}

/**
 * DERIVED-not-authored (P0 interim). Elementwise is element-parallel on every
 * axis, so its oversized split axis is ALWAYS parallel — transport windowing
 * (the universal walker transform) applies wherever the kernel realizer supports
 * sub-range binding. When schedule-state receipts land, this is READ from the
 * composed `KernelRef`'s parallel-vs-carried axis classification rather than
 * asserted here; it is exposed as the smallest honest interim, sourced from the
 * schedule side, NOT a per-declaration authored choice. A carried split axis
 * (reductions) is a typed walker refusal, not this family's concern.
 */
export const ELEMENTWISE_SPLIT_AXIS_PARALLEL = true;

// ============================================================================
// The ELEMENTWISE family, authored as data
// ============================================================================

/**
 * WGSL binary op SYMBOL (a KERNEL-realization detail — the value fed to
 * `planBinaryDirect`). SINGLE SOURCE: the realizer reads this, replacing
 * stream-generate.ts's private `BINARY_OPS` map and the dispatch per-call
 * symbol.
 */
export const ELEMENTWISE_BINARY_WGSL: ReadonlyMap<string, string> = new Map([
  ["add", "+"],
  ["sub", "-"],
  ["mul", "*"],
  ["div", "/"],
  ["pow", "pow"],
  ["minimum", "min"],
  ["maximum", "max"],
]);

/**
 * Table-driven unary ops routed to `planUnaryDirect` with identity opKeys
 * (SINGLE SOURCE, replacing stream-generate.ts's private `UNARY_OPS` set).
 * `gelu`/`cast`/`contiguous`/`where` are elementwise too but take their own
 * kernel family (below); `gelu`'s opKey is payload-derived at the realizer.
 */
export const ELEMENTWISE_UNARY_OPS: ReadonlySet<string> = new Set([
  "sqrt",
  "relu",
  "exp",
  "log",
  "neg",
  "abs",
  "tanh",
  "sigmoid",
  "silu",
  "sin",
  "cos",
  "rsqrt",
  "floor",
  "ceil",
  "round",
  "sign",
  "isfinite",
]);

function elementwiseDecl(
  arity: number,
  kernel: KernelRef,
): ExecutionDeclaration {
  return {
    family: "elementwise",
    arity,
    kernel,
    layout: "strided-ok",
    skeleton: [
      {
        op: "alloc",
        slot: { role: "output" },
        aliasOperands: "poolable-binds",
      },
      { op: "uniform", slot: { role: "params" } },
      {
        op: "dispatch",
        kernel,
        bindings: [
          { role: "all-inputs" },
          { role: "output" },
          { role: "params" },
        ],
      },
    ],
  };
}

/**
 * The elementwise family declarations, keyed by `LazyIRNode.op`. Absorbs the
 * per-op classification that lived duplicated across `generateSequential`
 * (BINARY_OPS / UNARY_OPS / the cast|where|contiguous|gelu branches) and
 * `dispatchBinary`/`dispatchUnary`. A node whose op is here IS the elementwise
 * family; its arity and kernel family are DATA. Whether an oversized op WINDOWS
 * is NOT declared — it is derived at the walker from whether the kernel realizer
 * supports sub-range binding (binary/unary/contiguous: yes; cast/where: no
 * windowed realizer, so their oversized case falls to the direct plan's own
 * null → lowered).
 */
export const ELEMENTWISE_DECLARATIONS: ReadonlyMap<
  string,
  ExecutionDeclaration
> = new Map<string, ExecutionDeclaration>([
  ...[...ELEMENTWISE_BINARY_WGSL.keys()].map(
    (op) => [op, elementwiseDecl(2, { kernel: "binaryDirect" })] as const,
  ),
  ...[...ELEMENTWISE_UNARY_OPS].map(
    (op) => [op, elementwiseDecl(1, { kernel: "unaryDirect" })] as const,
  ),
  ["gelu", elementwiseDecl(1, { kernel: "unaryDirect" })],
  ["contiguous", elementwiseDecl(1, { kernel: "contiguousDirect" })],
  ["cast", elementwiseDecl(1, { kernel: "castDirect" })],
  ["where", elementwiseDecl(3, { kernel: "whereDirect" })],
]);

/** The elementwise-family declaration for `op`, or undefined (not this family). */
export function elementwiseDeclaration(
  op: string,
): ExecutionDeclaration | undefined {
  return ELEMENTWISE_DECLARATIONS.get(op);
}

/** Whether `op` is a declared elementwise-family op. */
export function isDeclaredElementwise(op: string): boolean {
  return ELEMENTWISE_DECLARATIONS.has(op);
}

// ============================================================================
// The REDUCTION family, authored as data (P1)
// ============================================================================

/**
 * The reduction MONOID — the associative combiner over the reduced axis (a
 * schedule-state monoid fact). `sum`/`max`/`min`. `mean` is NOT a monoid: it is
 * a `sum` reduction followed by a div-by-count epilogue (`meanEpilogue`).
 *
 * This label is now a PROJECTION of the semantic-derivation reduction monoid
 * (`reduceMonoidOf`, ops/semantic/reduction.ts) — the declaration below reads it
 * from the ONE definition rather than re-authoring the string (design §4 P1
 * unification). It matches the semantic `ReduceMonoidName` by construction.
 */
export type ReduceMonoid = "sum" | "max" | "min";

/**
 * A reduction op-family declaration. Reductions ARE the carried-axis case: the
 * declaration COMPOSES the answer, it never re-owns it. Only the composition
 * FACTS that are not derivable are authored here —
 *   - `monoid`: the op's combiner (the schedule-state monoid fact);
 *   - `meanEpilogue`: whether a `meanDiv` (÷count) stage composes after the
 *     reduce (`mean` = reduce(`sum`) ÷ count — a TWO-stage skeleton);
 *   - `layout`: input(0) is forced contiguous-offset0 before binding (a real
 *     generalization of this op's `CONTIGUOUS_OPERANDS` entry — the reduce
 *     kernels read their operand flat-from-element-0).
 * Everything STRUCTURAL is DERIVED at the walker, NOT stored (the P0 refinement,
 * extended to the carried axis):
 *   - full-vs-dim from `payload.dim` (the elementwise direct-vs-windowed
 *     analogue);
 *   - the OVERSIZED-on-the-carried-axis case (operand exceeds the binding limit)
 *     resolves to the MULTI-KERNEL partials+combine lowering that already
 *     exists (`planChunkedFullReduction`) — the existing chunked machinery is
 *     the REALIZED IMAGE of the declared two-stage skeleton; the walker derives
 *     it from `fullReductionNeedsChunking`, with the `sum` monoid supplying the
 *     partial and combine kernels. No `skeleton`/`decompose` field carries it
 *     (zero-schema-delta on the elementwise `ExecutionDeclaration`).
 * REJECTED extension: reusing the elementwise `ExecutionDeclaration.skeleton`
 * array for reductions — a static ALLOC→[UNIFORM]→DISPATCH list cannot express
 * the structurally-variable reduction command shape (full / dim / two-stage /
 * +meanDiv), which is DERIVED like elementwise windowing. The monoid + epilogue
 * are the only genuinely-authored facts, so the reduction declaration is a thin
 * sibling record, not a `skeleton` reuse.
 */
export interface ReductionDeclaration {
  family: "reduction";
  monoid: ReduceMonoid;
  meanEpilogue: boolean;
  layout: "contiguous-input";
}

// ============================================================================
// The MATMUL family, authored as data (P2)
// ============================================================================

/**
 * Matmul operand roles, in bind-group order. The variadic `epilogue-inputs`
 * role expands to the node's epilogue inputs (empty for a bare matmul). SINGLE
 * SOURCE for the `[a, b, out, params, ...epi]` binding array both matmul
 * generators hardcoded.
 */
export type MatmulBindRole =
  | "a"
  | "b"
  | "output"
  | "params"
  | "epilogue-inputs";

/**
 * A matmul op-family declaration. Matmul is the biggest duplication pair
 * (dispatch ~1510 + generation ~205). Following the P1 thin-sibling-record
 * ruling, only the genuinely-AUTHORED family facts live here; everything
 * structural (which route engaged, dispatch grid, pipeline, K-split geometry)
 * is DERIVED at the walker from the plan the realizer (`planTiledMatmul`)
 * returns — the KernelRef-resolution analogue P1's reduction walker CALLS. The
 * authored facts:
 *   - `bindingTemplate`: the operand-role bind order, replacing the two
 *     hand-mirrored `[a, b, out, params, ...epi]` arrays (bare + epilogue). The
 *     K-split path composes its OWN two-stage bindings (partials temp + reduce)
 *     from the plan — not this template.
 *   - `layout`: matmul binds operand storage buffers RAW — a simple-transpose
 *     input is a non-owning view on the SAME buffer, a contiguous input is
 *     itself; a contiguous-copy prologue is a typed CAPTURE bail
 *     (`planBareMatmul` → "contiguous-prologue"), never a walker prologue.
 *   - `route`: the tiled-vs-GEMV-vs-K-split decision (incl. the #95 inputCast
 *     admit and the #93 quantB exclusion) is decided ONCE inside
 *     `planTiledMatmul` (`selectMatmulChoice`) and carried on the plan as a
 *     `SelectionReceipt`. Every consumer — lowered dispatch, build-from-IR
 *     capture, and BOTH serializer walkers — READS that receipt's
 *     `gemvEngaged`; none re-parses the profiler label (retiring the #95
 *     replay-blind `plan.label?.startsWith("_gemv")` re-derivation the epilogue
 *     generator carried). K-split is the carried-axis two-stage schedule fact,
 *     COMPOSED from the plan (partials + combine), never re-owned — the
 *     reductions two-stage analogue.
 *   - `operandFormat`: the StorageFormat axes that participate in the route
 *     decision (#93/#95 rule — StorageFormat participates at every route
 *     decision): a packed-int8 B operand is route-EXCLUDED (a fused-dequant
 *     GEMV realizer the tiled-matmul stream cannot serialize → typed capture
 *     bail "quantized-operand"), inputCast (a read-wider stored dtype) is a
 *     route AXIS (admits GEMV NT). Declared so the route's format inputs are
 *     legible at ONE place, not re-discovered at each decision site.
 * REJECTED extension: reusing the elementwise `ExecutionDeclaration.skeleton`
 * for matmul — a static ALLOC→[UNIFORM]→DISPATCH list cannot express the
 * route-variable command shape (bare / epilogue / K-split two-stage), which is
 * DERIVED from the plan like elementwise windowing and reductions' two-stage.
 * The binding template + layout + route-source + operand-format axes are the
 * only genuinely-authored facts; zero-schema-delta on the sibling declarations.
 */
export interface MatmulDeclaration {
  family: "matmul";
  bindingTemplate: readonly MatmulBindRole[];
  layout: "raw-bindable";
  route: "receipt-consumed";
  operandFormat: {
    quantB: "route-excluded";
    inputCast: "route-axis";
  };
}

/**
 * The matmul family declaration. Keyed by `LazyIRNode.op === "matmul"` — the
 * ONE declaration covers both the bare matmul (a `sequential` action over a
 * `matmul` node) and the `matmul-epilogue` action (whose underlying node op is
 * also `matmul`); the walker branches on bare-vs-epilogue by entry point, not
 * by a second declaration.
 */
export const MATMUL_DECLARATION: MatmulDeclaration = {
  family: "matmul",
  bindingTemplate: ["a", "b", "output", "params", "epilogue-inputs"],
  layout: "raw-bindable",
  route: "receipt-consumed",
  operandFormat: { quantB: "route-excluded", inputCast: "route-axis" },
};

/** The matmul-family declaration for `op`, or undefined (not this family). */
export function matmulDeclaration(op: string): MatmulDeclaration | undefined {
  return op === "matmul" ? MATMUL_DECLARATION : undefined;
}

/** Whether `op` is a declared matmul-family op. */
export function isDeclaredMatmul(op: string): boolean {
  return op === "matmul";
}

/** Any command-stream declaration (the schema-gate accepts every family). */
export type AnyDeclaration =
  | ExecutionDeclaration
  | ReductionDeclaration
  | MatmulDeclaration;

/**
 * The reduction family declarations, keyed by `LazyIRNode.op`. `sum`/`mean` are
 * the node-level ops the serializer covers; `max`/`min` appear only as the
 * `batched-reduction` action's monoid, so they are declared here too and read by
 * the batched walker via the action's `reduceOp`. Absorbs the per-generator
 * `reduceOp = "sum"` hardcode (generateFullReduction/generateDimReduction/the
 * mean sum stage) and the batched action's `op` classification.
 */
/**
 * Build a reduction declaration from its semantic definition — the P1
 * unification (design §4). The two genuinely-authored facts DERIVE from the ONE
 * monoid definition rather than being re-spelled here:
 *   - `monoid`       = `reduceMonoidOf(def)` (projected from the combine's root);
 *   - `meanEpilogue` = `def.epilogue !== null` (mean carries the ÷count epilogue).
 * `layout` stays authored: it is a REALIZER fact (the reduce kernels read their
 * operand flat-from-element-0), not a semantic one. Zero-schema-delta on the
 * `ReductionDeclaration` interface — only the VALUES now single-source with the
 * definition, so the declaration and the definition cannot silently disagree on
 * the monoid label.
 */
function reductionDeclOf(def: ReductionDef): ReductionDeclaration {
  return {
    family: "reduction",
    monoid: reduceMonoidOf(def),
    meanEpilogue: def.epilogue !== null,
    layout: "contiguous-input",
  };
}

export const REDUCTION_DECLARATIONS: ReadonlyMap<string, ReductionDeclaration> =
  new Map<string, ReductionDeclaration>(
    [SUM_DEF, MEAN_DEF, MAX_DEF, MIN_DEF].map(
      (def) => [def.name, reductionDeclOf(def)] as const,
    ),
  );

/** The reduction-family declaration for `op`, or undefined (not this family). */
export function reductionDeclaration(
  op: string,
): ReductionDeclaration | undefined {
  return REDUCTION_DECLARATIONS.get(op);
}

/** Whether `op` is a declared reduction-family op. */
export function isDeclaredReduction(op: string): boolean {
  return REDUCTION_DECLARATIONS.has(op);
}

// ============================================================================
// The schema gate (structural — the "no second owner" enforcement)
// ============================================================================

/**
 * Prove a declaration is DATA: no leaf is a function (a generator/callback), a
 * `GPUBuffer`, or any object outside the closed schema shape. Throws with a
 * path on the first violation. This is what makes "the serializer and
 * interpreter read ONE object" structural rather than hopeful — an adapter that
 * replays the old per-op generator behind an opaque leaf cannot be authored,
 * because that leaf would be a function and this gate rejects it.
 */
export function assertNoGeneratorLeaf(
  decl: AnyDeclaration,
  path = "declaration",
): void {
  const walk = (value: unknown, p: string): void => {
    if (value === null || value === undefined) return;
    const t = typeof value;
    if (t === "function") {
      throw new Error(
        `ExecutionDeclaration schema gate: ${p} is a function — a declaration must be data (no serializable generator/callback).`,
      );
    }
    if (t === "string" || t === "number" || t === "boolean") return;
    if (t === "object") {
      // A raw GPUBuffer / typed array is a buffer leaf — also forbidden.
      if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
        throw new Error(
          `ExecutionDeclaration schema gate: ${p} is a buffer leaf — the declaration names roles, never buffers.`,
        );
      }
      if (Array.isArray(value)) {
        value.forEach((v, i) => {
          walk(v, `${p}[${i}]`);
        });
        return;
      }
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        walk(v, `${p}.${k}`);
      }
      return;
    }
    throw new Error(
      `ExecutionDeclaration schema gate: ${p} has non-data type ${t}.`,
    );
  };
  walk(decl, path);
}
