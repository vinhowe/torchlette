/**
 * Operand layout requirements as DECLARED metadata (campaign #99 fork a).
 *
 * SINGLE SOURCE for "which INPUT operands of a kernel op must be laid out
 * raw-bindable (contiguous strides AND element offset 0) before the kernel
 * binds them flat-from-element-0". Two consumers derive from this:
 *
 *  - The stream generator (`stream-generate.ts` `resolveContiguousOperand`)
 *    SYNTHESIZES a contiguous-copy prologue for any declared-contiguous operand
 *    that resolves to LIVE strided storage — mirroring, byte-for-byte, the
 *    `asContiguous`/`ensureContiguous` copy the DISPATCH layer inserts. This is
 *    the load-bearing consumer: it closes the `[non-contiguous]` uncovered class
 *    for every declared op at once.
 *  - The DISPATCH layer's own `asContiguous`/`ensureContiguous` calls stay in
 *    place; they are ASSERTED to agree with this table by the t-stream-generate
 *    differential (a disagreeing entry produces a command-level DIVERGE, never a
 *    silent third copy).
 *
 * See docs/operand-layout-metadata-design.md for the indictment table and the
 * rejected shapes (per-operand alignment; a new op registry — both unjustified
 * by the inventory).
 */

/**
 * Per-op contiguity requirement over INPUT operand positions.
 *  - "all"      : every non-scalar input must be raw-bindable.
 *  - number[]   : only these input positions (reserved for future mixed ops
 *                 that raw-bind some operands and offset-fold others).
 */
export type ContiguousOperandSpec = "all" | readonly number[];

/**
 * The declaration. Keyed by `LazyIRNode.op` (the string the generator switches
 * on). Every entry here corresponds to a dispatch-site `asContiguous`/
 * `ensureContiguous` prologue in the inventory (docs table). Ops absent from
 * this map declare no contiguity requirement (their generators handle layout
 * themselves or their operands are contiguous by construction).
 *
 * All current entries are "all": the inventory shows each of these ops
 * raw-binds every bindable input (CE logits/grad, reduction input, unscale
 * grad, fused-kernel recipe inputs). A `number[]` entry would name specific
 * positions if an op ever mixed raw-bound and offset-folding operands.
 */
export const CONTIGUOUS_OPERANDS: ReadonlyMap<string, ContiguousOperandSpec> =
  new Map<string, ContiguousOperandSpec>([
    // Cross-entropy: logits (fwd+bwd) and grad_output (bwd) — targets is an
    // i32 coercion (ensureI32Targets), a distinct axis handled at its own site.
    ["fusedCrossEntropyForward", [0]],
    ["fusedCrossEntropyBackward", [0, 2]],
    // Reductions read their sole input as a contiguous buffer (full/dim/mean
    // and the batched-reduction fallback, which forces contiguity for every
    // reduce op — reductions.ts:893).
    ["sum", [0]],
    ["mean", [0]],
    ["max", [0]],
    ["min", [0]],
    // Arg-reduce (argmax/argmin) — the decode feedback selection. The kernel
    // derives addressing from contiguousStrides(inputShape) and binds flat from
    // element 0, so a strided/offset input must be materialized contiguous
    // first (comparison.ts argReduceOp forces it; the generator synthesizes the
    // same copy prologue).
    ["argmax", [0]],
    ["argmin", [0]],
    // unscaleGrad raw-binds grad_in (input 0); the scale operand is read live.
    ["unscaleGrad", [0]],
    // Fused RMSNorm forward raw-binds x (input 0) and weight (input 1) — both
    // asContiguous'd at dispatch (fused.ts). The decode residual + persistent
    // norm weight are contiguous by construction, so the prologue is empty.
    ["fusedRMSNormForward", [0, 1]],
  ]);

/** Whether input position `idx` of op `op` is declared contiguity-required. */
export function operandRequiresContiguous(op: string, idx: number): boolean {
  const spec = CONTIGUOUS_OPERANDS.get(op);
  if (spec === undefined) return false;
  if (spec === "all") return true;
  return spec.includes(idx);
}
