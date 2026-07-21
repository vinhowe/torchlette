/**
 * Semantic Derivation — the CONTRACTION adjoint (Crystal Campaign 3, closure §19).
 *
 * The framework's last two hand-written adjoints were `matmulBackward` and its
 * `linearBackward` specialization (design §4.2): the matmul VJP is an
 * index-TRANSPOSE fact about contraction that the elementwise chain-rule pass
 * (adjoint.ts) and the index-map transpose (index-map.ts) could not state. This
 * module states it ONCE, closing the calculus — every gradient in the framework
 * now derives from a fact.
 *
 * THE FACT (one declaration, `contractionAdjoint`): the adjoint of a contraction
 * is TWO contractions. For `C = op(A,ta) · op(B,tb)` (a batched matmul with
 * per-operand transpose flags — the SAME 2-operand form the optimizer frame's
 * `mm` node already carries, optimizer.ts), each operand's grad is a contraction
 * of the upstream grad `G = dC` with the OTHER operand, transpose flags fixed by
 * (ta,tb):
 *
 *     C = A · B          (ta=F, tb=F)  →  dA = G·Bᵀ ,   dB = Aᵀ·G     (matmul)
 *     C = A · Bᵀ         (ta=F, tb=T)  →  dA = G·B  ,   dB = Gᵀ·A     (linear)
 *     C = Aᵀ · B         (ta=T, tb=F)  →  dA = B·Gᵀ ,   dB = A·G
 *     C = Aᵀ · Bᵀ        (ta=T, tb=T)  →  dA = Bᵀ·Gᵀ,   dB = Gᵀ·Aᵀ
 *
 * The four rows are the ONE flag-flip rule below, not four cases. Both hand
 * adjoints are the (F,F) and (F,T) rows: `matmulBackward` built `[G·Bᵀ, Aᵀ·G]`,
 * `linearBackward` built `[G·W, Gᵀ·X]` (linear's forward is `X·Wᵀ`, the tb=T
 * row) plus `dBias = sumToShape(G, biasShape)` — the broadcast-adjoint reduction
 * the engine already owns (index-map.ts). The derived path emits the SAME graph
 * op-for-op (same transposes, same matmuls, same `sumToShape`), so trajectory
 * parity is byte-identical by construction; the autocast discipline (backward
 * reads the f16-cast saved operands) is preserved untouched — the fact only
 * chooses transpose flags, it does not change what is saved.
 *
 * The FACT is pure DATA→DATA (`contractionAdjoint`, unit-testable, refereed by
 * the torch oracle for all four flag combos). Its REALIZATION composes the
 * existing runtime kernels (`realizeContractionAdjoint`: `rt.transpose` +
 * `rt.matmul`), never re-owning them (design §4.4).
 */

import type { RuntimeEngine } from "../../runtime/engine";
import type { Tensor as RuntimeTensor } from "../../runtime/tensor";
import type { GradFn } from "../../frontend/types";

// ----------------------------------------------------------------------------
// The contraction schema — DATA (transpose flags only).
// ----------------------------------------------------------------------------

/** A forward contraction `C = op(A,ta) · op(B,tb)` — per-operand transpose flags. */
export interface Contraction {
  ta: boolean;
  tb: boolean;
}

/** A symbolic operand of an adjoint contraction: A, B, or the upstream grad G. */
export type ContractionOperand = "A" | "B" | "G";

/** One adjoint contraction `out = op(lhs,ta) · op(rhs,tb)` — also DATA. */
export interface AdjointContraction {
  lhs: ContractionOperand;
  rhs: ContractionOperand;
  ta: boolean;
  tb: boolean;
}

// ----------------------------------------------------------------------------
// The derived fact: the adjoint of a contraction is two contractions.
// ----------------------------------------------------------------------------

/**
 * DERIVE the adjoint of a contraction — the single source for the matmul/linear
 * VJP. `grad_A` contracts G with B, `grad_B` contracts G with A; the transpose
 * flags flip per (ta,tb) (see the four rows in the module header). Pure
 * DATA→DATA — the transpose DECISION derives; the movement is realized by `rt.*`.
 */
export function contractionAdjoint(fwd: Contraction): {
  dA: AdjointContraction;
  dB: AdjointContraction;
} {
  const { ta, tb } = fwd;
  const dA: AdjointContraction = ta
    ? { lhs: "B", rhs: "G", ta: tb, tb: true } // (B^tb) · Gᵀ
    : { lhs: "G", rhs: "B", ta: false, tb: !tb }; // G · (B^tb)ᵀ
  const dB: AdjointContraction = tb
    ? { lhs: "G", rhs: "A", ta: true, tb: ta } // Gᵀ · (A^ta)
    : { lhs: "A", rhs: "G", ta: !ta, tb: false }; // (A^ta)ᵀ · G
  return { dA, dB };
}

// ----------------------------------------------------------------------------
// The realizer — compose runtime kernels for the derived adjoint (design §4.4:
// the movement is REFERENCED, never re-owned). Pure movement, no formula.
// ----------------------------------------------------------------------------

/** The live operands an adjoint contraction reads. */
export interface ContractionAdjointCtx {
  rt: RuntimeEngine;
  A: RuntimeTensor;
  B: RuntimeTensor;
  G: RuntimeTensor;
}

/** Transpose the last two dims (a 2D/batched matmul operand view) when `flag`. */
function applyTranspose(
  rt: RuntimeEngine,
  t: RuntimeTensor,
  flag: boolean,
): RuntimeTensor {
  if (!flag) return t;
  const r = t.shape.length;
  return rt.transpose(t, { dim0: r - 2, dim1: r - 1 });
}

function operand(ref: ContractionOperand, ctx: ContractionAdjointCtx): RuntimeTensor {
  return ref === "A" ? ctx.A : ref === "B" ? ctx.B : ctx.G;
}

/** Realize a derived adjoint contraction over the runtime engine. */
export function realizeContractionAdjoint(
  ctx: ContractionAdjointCtx,
  adj: AdjointContraction,
): RuntimeTensor {
  const lhs = applyTranspose(ctx.rt, operand(adj.lhs, ctx), adj.ta);
  const rhs = applyTranspose(ctx.rt, operand(adj.rhs, ctx), adj.tb);
  return ctx.rt.matmul(lhs, rhs);
}

// ----------------------------------------------------------------------------
// The schema gate — an adjoint contraction is DATA (the `assertNoDefinitionBody`
// analogue for the contraction frame; covenant/R22 defense). No leaf is a
// closure/buffer; operands are known enum members, flags are booleans.
// ----------------------------------------------------------------------------

const OPERANDS = new Set<string>(["A", "B", "G"]);

/** Prove an `AdjointContraction` is DATA (operand enum + booleans only). */
export function assertNoContractionBody(adj: AdjointContraction, path = "adj"): void {
  const ok =
    adj != null &&
    typeof adj === "object" &&
    OPERANDS.has(adj.lhs) &&
    OPERANDS.has(adj.rhs) &&
    typeof adj.ta === "boolean" &&
    typeof adj.tb === "boolean";
  if (!ok)
    throw new Error(
      `Contraction schema gate: ${path} is not a DATA term (operand enum + booleans).`,
    );
}

// ----------------------------------------------------------------------------
// The frontend backward builders — the two admitted contraction adjoints,
// now DERIVED from the fact above (design §4.2). These are the dispatch stubs
// `torchlette.ts` routes matmul/linear through; the ADJOINT MATH lives in
// `contractionAdjoint` + `realizeContractionAdjoint`, not here — this is only
// the per-op bookkeeping (which grads are needed, saved-slot order, the bias
// reduction). The hand transpose+matmul bodies (custom-backward.ts) are deleted.
// ----------------------------------------------------------------------------

/** Context available to the contraction backward builders. */
export interface BackwardContext {
  rt: RuntimeEngine;
  sumToShape: (grad: RuntimeTensor, shape: number[]) => RuntimeTensor;
}

/** Matmul backward `C = A·B` — the (ta=F, tb=F) contraction adjoint. */
export function matmulBackward(
  ctx: BackwardContext,
  aShape: number[],
  bShape: number[],
): GradFn {
  const { dA, dB } = contractionAdjoint({ ta: false, tb: false });
  return (grad, getSaved) => {
    if (aShape.length < 2 || bShape.length < 2)
      throw new Error("matmul backward requires rank >= 2");
    const cctx: ContractionAdjointCtx = {
      rt: ctx.rt,
      A: getSaved(0)._unwrap(),
      B: getSaved(1)._unwrap(),
      G: grad,
    };
    return [
      ctx.sumToShape(realizeContractionAdjoint(cctx, dA), aShape),
      ctx.sumToShape(realizeContractionAdjoint(cctx, dB), bShape),
    ];
  };
}

/**
 * Linear backward `Y = X·Wᵀ (+ b)` — the (ta=F, tb=T) contraction adjoint fused
 * with the bias broadcast-adjoint. `dX = G·W`, `dW = Gᵀ·X` (the tb=T row, in W's
 * shape directly — the fusion that avoids materializing `Wᵀ`'s grad), `dBias`
 * the reduction of G to the bias shape. Only the requested grads are built.
 */
export function linearBackward(
  ctx: BackwardContext,
  inputShape: number[],
  weightShape: number[],
  needsInputGrad: boolean,
  needsWeightGrad: boolean,
  hasBias: boolean,
): GradFn {
  const { dA, dB } = contractionAdjoint({ ta: false, tb: true });
  return (grad, getSaved) => {
    if (inputShape.length < 2 || weightShape.length < 2)
      throw new Error("linear backward requires rank >= 2");
    // Operands: A = X (input, read by dB), B = W (weight, read by dA). Only the
    // needed operand is saved (dispatch policy); the filler is never realized.
    let i = 0;
    const weight = needsInputGrad ? getSaved(i++)._unwrap() : grad;
    const input = needsWeightGrad ? getSaved(i++)._unwrap() : grad;
    const cctx: ContractionAdjointCtx = { rt: ctx.rt, A: input, B: weight, G: grad };
    const gx = needsInputGrad
      ? ctx.sumToShape(realizeContractionAdjoint(cctx, dA), inputShape)
      : null;
    const gw = needsWeightGrad
      ? ctx.sumToShape(realizeContractionAdjoint(cctx, dB), weightShape)
      : null;
    const gb = hasBias ? ctx.sumToShape(grad, getSaved(i).shape) : null;
    return hasBias ? [gx, gw, gb as RuntimeTensor] : [gx, gw];
  };
}
