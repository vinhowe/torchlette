/**
 * fuse-matmul-sum: rewrite `sum(matmul(transpose(X, -2, -1), Y), dim=batch_dims)`
 * into a rank-2 matmul of flattened operands.
 *
 * This pattern appears in autograd weight gradients. For `dW = A^T @ grad`
 * over batched activations, the naive emission produces a [batch..., K, N]
 * intermediate that is immediately reduced to [K, N]. The intermediate is
 * B× larger than the final result. At distilgpt2 batch=4 seq=512 LM-head
 * this is 617MB that reduces to 154MB. Flattening the batch into the row
 * dim lets a single rank-2 matmul compute the summed gradient directly.
 *
 * Algebraic identity (verified in semantic tests):
 *   sum_b( transpose(X_b) @ Y_b )[k, n]
 *     = Σ_b Σ_m X_b[m, k] · Y_b[m, n]
 *     = Σ_(b, m) X[b, m, k] · Y[b, m, n]
 *     = (reshape(X, [B*M, K])^T @ reshape(Y, [B*M, N]))[k, n]
 */
import type { LazyIRNode } from "../../../graph/types";
import type { Rule } from "../engine";
import { capture, op, refDtype, refNode, refShape } from "../pattern";

export const fuseMatmulSumRule: Rule = {
  name: "fuse-matmul-sum",
  pattern: op("sum", {
    where: isLeadingBatchSum,
    inputs: [
      capture(
        "mm",
        op("matmul", {
          inputs: [
            capture(
              "tx",
              op("transpose", {
                where: isLastTwoDimTranspose,
                inputs: [capture("X")],
              }),
            ),
            capture("Y"),
          ],
        }),
      ),
    ],
  }),
  check: (bindings) => {
    const X = bindings.get("X")!;
    const Y = bindings.get("Y")!;
    const mm = refNode(bindings.get("mm")!);
    if (!mm) return false;
    const xShape = refShape(X);
    const yShape = refShape(Y);
    if (!xShape || !yShape) return false;

    const rank = mm.shape.length;
    if (rank < 3) return false; // need at least one batch dim
    if (xShape.length !== rank || yShape.length !== rank) return false;

    // M dims must match (contraction dim of the original matmul).
    if (xShape[rank - 2] !== yShape[rank - 2]) return false;
    // Batch dims must match elementwise between X and Y.
    for (let i = 0; i < rank - 2; i++) {
      if (xShape[i] !== yShape[i]) return false;
    }
    return true;
  },
  rewrite: (bindings, ctx, node) => {
    const X = bindings.get("X")!;
    const Y = bindings.get("Y")!;
    const xShape = refShape(X)!;
    const yShape = refShape(Y)!;
    const xDtype = refDtype(X)!;
    const yDtype = refDtype(Y)!;
    const device = node.device;

    const rank = xShape.length;
    const M = xShape[rank - 2];
    const K = xShape[rank - 1];
    const N = yShape[rank - 1];

    let batchProd = 1;
    for (let i = 0; i < rank - 2; i++) batchProd *= xShape[i];
    const totalRows = batchProd * M;

    // reshape(X, [B*M, K]) → transpose → [K, B*M]
    const xReshape = ctx.createNode(
      "reshape",
      [X],
      [totalRows, K],
      xDtype,
      device,
      { targetShape: [totalRows, K] },
    );
    const xTranspose = ctx.createNode(
      "transpose",
      [ctx.pendingRef(xReshape)],
      [K, totalRows],
      xDtype,
      device,
      { dim0: 0, dim1: 1 },
    );
    // reshape(Y, [B*M, N])
    const yReshape = ctx.createNode(
      "reshape",
      [Y],
      [totalRows, N],
      yDtype,
      device,
      { targetShape: [totalRows, N] },
    );

    // Remove the orphaned upstream chain. Pending tensors on the old
    // matmul/transpose were autograd saved-for-backward artifacts; by
    // the time backward is computing, they're no longer needed. Guard against
    // a SHARED / externally-held mm or tx (a user-shaped graph could reuse
    // either) — killing a node with other live readers or an external
    // RuntimeTensor produces "Input not ready" (the #67 class).
    const mmNode = refNode(bindings.get("mm")!)!;
    const txNode = refNode(bindings.get("tx")!)!;
    if (ctx.consumerCount(mmNode) <= 1 && !ctx.isExternal(mmNode)) {
      ctx.markDead(mmNode);
    }
    if (ctx.consumerCount(txNode) <= 1 && !ctx.isExternal(txNode)) {
      ctx.markDead(txNode);
    }

    // Mutate the matched sum node into a matmul in place. Preserves the
    // node id so RuntimeTensors pending on it (gradient outputs) keep
    // working, and downstream consumers don't need rewiring. See
    // engine.ts for the in-place-mutation contract.
    node.op = "matmul";
    node.inputs = [ctx.pendingRef(xTranspose), ctx.pendingRef(yReshape)];
    node.payload = undefined;
    return ctx.pendingRef(node);
  },
};

// ============================================================================
// Per-node predicates
// ============================================================================

function isLeadingBatchSum(node: LazyIRNode): boolean {
  const payload = node.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;
  if (!payload || payload.keepdim) return false;
  if (payload.dim == null) return false;
  const dims = Array.isArray(payload.dim)
    ? [...payload.dim].sort((a, b) => a - b)
    : [payload.dim];
  if (dims.length === 0) return false;
  // Must be contiguous leading dims [0, 1, ..., k-1]
  for (let i = 0; i < dims.length; i++) {
    if (dims[i] !== i) return false;
  }
  return true;
}

function isLastTwoDimTranspose(node: LazyIRNode): boolean {
  const payload = node.payload as { dim0?: number; dim1?: number } | undefined;
  if (!payload || payload.dim0 == null || payload.dim1 == null) return false;
  const rank = node.shape.length;
  const d0 = payload.dim0 < 0 ? rank + payload.dim0 : payload.dim0;
  const d1 = payload.dim1 < 0 ? rank + payload.dim1 : payload.dim1;
  return (
    (d0 === rank - 2 && d1 === rank - 1) || (d0 === rank - 1 && d1 === rank - 2)
  );
}
