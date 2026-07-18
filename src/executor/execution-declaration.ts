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
 * params]). Chunked (>maxStorageBufferBindingSize) is the family's `size-split`
 * decomposition — same declaration, N dispatches derived by the shared splitter
 * (`computeChunkGeometry`), never enumerated here.
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

// ============================================================================
// The schema
// ============================================================================

/** Op-family identity. P0 declares only `elementwise`; P1–P4 add the rest. */
export type OpFamilyUid = "elementwise";

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

/**
 * The decomposition RULE (derived, never stored — the loop-nest-VIEW analogue):
 *  - `direct`    : one dispatch.
 *  - `size-split`: N dispatches when any operand/output exceeds
 *                  `maxStorageBufferBindingSize`, split by the shared
 *                  `computeChunkGeometry`. A device with a different limit
 *                  re-derives; nothing is baked (the stale-128 MB scar,
 *                  structurally prevented).
 */
export type DecompositionRule = "direct" | "size-split";

export interface ExecutionDeclaration {
  family: OpFamilyUid;
  /** Fixed arity; a node with a different input count is a typed refusal. */
  arity: number;
  /** How the kernel is realized (WGSL + geometry + params). */
  kernel: KernelRef;
  /** Layout requirement of the DIRECT path (`size-split` tightens to
   *  `contiguous-offset0` at the realizer). */
  layout: LayoutRequirement;
  /** Whether this op can `size-split` when oversized (else direct-only). */
  decompose: DecompositionRule;
  /** The ordered command shape, over roles. Shared family template. */
  skeleton: SkeletonCommand[];
}

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
  decompose: DecompositionRule,
): ExecutionDeclaration {
  return {
    family: "elementwise",
    arity,
    kernel,
    layout: "strided-ok",
    decompose,
    skeleton: [
      { op: "alloc", slot: { role: "output" }, aliasOperands: "poolable-binds" },
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
 * family; its arity, kernel family, and decomposition capability are DATA.
 *
 * `size-split`: binary and plain-unary decompose when oversized (their
 * dispatchers route to the chunked path). `cast`/`where`/`contiguous` are
 * direct-only (their dispatchers force contiguity / stay direct at the
 * oversized boundary) — a typed `direct` refusal is honest, not a bail.
 */
export const ELEMENTWISE_DECLARATIONS: ReadonlyMap<string, ExecutionDeclaration> =
  new Map<string, ExecutionDeclaration>([
    ...[...ELEMENTWISE_BINARY_WGSL.keys()].map(
      (op) =>
        [
          op,
          elementwiseDecl(2, { kernel: "binaryDirect" }, "size-split"),
        ] as const,
    ),
    ...[...ELEMENTWISE_UNARY_OPS].map(
      (op) =>
        [
          op,
          elementwiseDecl(1, { kernel: "unaryDirect" }, "size-split"),
        ] as const,
    ),
    ["gelu", elementwiseDecl(1, { kernel: "unaryDirect" }, "size-split")],
    ["cast", elementwiseDecl(1, { kernel: "castDirect" }, "direct")],
    ["contiguous", elementwiseDecl(1, { kernel: "contiguousDirect" }, "direct")],
    ["where", elementwiseDecl(3, { kernel: "whereDirect" }, "direct")],
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
  decl: ExecutionDeclaration,
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
        value.forEach((v, i) => walk(v, `${p}[${i}]`));
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
