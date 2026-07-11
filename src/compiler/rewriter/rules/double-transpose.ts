/**
 * double-transpose: rewrite `transpose(transpose(x, a, b), a, b)` → `x`.
 *
 * Transposing the same two dims twice is identity. Autograd and manual
 * gradient computations frequently emit these (e.g., transpose(grad, -2, -1)
 * followed by the backward of a transposed tensor), and the nested form
 * can survive into the plan if intermediate ops obscure the optimization.
 *
 * We only match the EXACT-same-dims case. Transposing different dims
 * doesn't simplify this way (becomes a permute).
 */
import type { LazyIRNode } from "../../../graph/types";
import type { Rule } from "../engine";
import { capture, op } from "../pattern";

export const doubleTransposeRule: Rule = {
  name: "double-transpose",
  pattern: op("transpose", {
    inputs: [
      capture(
        "inner",
        op("transpose", {
          inputs: [capture("X")],
        }),
      ),
    ],
  }),
  check: (bindings, node) => {
    const innerRef = bindings.get("inner")!;
    if (innerRef.kind !== "pending") return false;
    const innerNode = innerRef.node;
    const outer = node.payload as { dim0?: number; dim1?: number } | undefined;
    const inner = innerNode.payload as
      | { dim0?: number; dim1?: number }
      | undefined;
    if (!outer || !inner) return false;
    if (outer.dim0 == null || outer.dim1 == null) return false;
    if (inner.dim0 == null || inner.dim1 == null) return false;
    // Normalize negative dims against the OUTER's rank (same as inner's, since
    // transpose preserves rank).
    const rank = node.shape.length;
    const normalize = (d: number) => (d < 0 ? rank + d : d);
    const o0 = normalize(outer.dim0);
    const o1 = normalize(outer.dim1);
    const i0 = normalize(inner.dim0);
    const i1 = normalize(inner.dim1);
    // Same pair of dims (order doesn't matter).
    const outerPair = [Math.min(o0, o1), Math.max(o0, o1)];
    const innerPair = [Math.min(i0, i1), Math.max(i0, i1)];
    return outerPair[0] === innerPair[0] && outerPair[1] === innerPair[1];
  },
  rewrite: (bindings, _ctx, node) => {
    // The outer transpose is redundant. Replace this node with X (the grand-
    // input). Mark the inner transpose dead so it's removed too — but ONLY
    // when THIS outer transpose is its sole consumer. A SHARED inner view
    // (e.g. `kT` in `matmul(q, kT)` whose backward emits `transpose(kT)`)
    // still feeds other live nodes; killing it makes them read a removed node
    // ("Input not ready: transpose", task #67). markDead ignores external
    // refs, so this consumer-count guard is the only thing keeping a shared
    // inner alive.
    const X = bindings.get("X")!;
    const innerRef = bindings.get("inner")!;
    if (
      innerRef.kind === "pending" &&
      _ctx.consumerCount(innerRef.node) <= 1 &&
      !_ctx.isExternal(innerRef.node)
    ) {
      _ctx.markDead(innerRef.node);
    }
    // Return X as replacement. The engine will rewire consumers of `node`
    // to use X directly. We can't in-place-mutate here because `node`'s
    // shape and the target replacement's shape differ — X is the grandparent.
    void node;
    return X;
  },
};
