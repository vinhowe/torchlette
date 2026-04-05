/**
 * transitive-reshape: `reshape(reshape(x, s1), s2)` → `reshape(x, s2)`.
 *
 * Reshape is a view op — it rewrites strides without moving data (or forces
 * contiguous if the new shape isn't stride-compatible). Two reshapes in a
 * row produce the same result as a single reshape to the final shape:
 * they only depend on the final shape and the total element count.
 *
 * We require outputs of both reshapes to agree on element count (they
 * always should — reshape rejects shape mismatches at construction) and
 * that the intermediate reshape's other consumers wouldn't need its
 * specific shape. If the inner reshape has multiple consumers, this
 * rule doesn't fire — removing it would break those other consumers.
 */
import type { Rule } from "../engine";
import { capture, op } from "../pattern";

export const transitiveReshapeRule: Rule = {
  name: "transitive-reshape",
  pattern: op("reshape", {
    inputs: [
      capture(
        "inner",
        op("reshape", {
          inputs: [capture("X")],
        }),
      ),
    ],
  }),
  // No cross-capture check needed: if reshape is valid, element counts
  // are equal by construction. But we DO need to verify the inner reshape
  // has only ONE consumer (the outer reshape we're about to bypass). If
  // other consumers need the intermediate shape, we can't remove inner.
  // That's handled by the engine + DCE — if inner has other consumers,
  // it simply survives after we rewire outer's consumers.
  rewrite: (bindings, ctx, node) => {
    const X = bindings.get("X")!;
    const outerShape = node.shape;
    const outerPayload = node.payload as
      | { targetShape?: number[] }
      | undefined;
    // Mutate the outer reshape to point directly at X. Preserves outer's
    // node id so consumers don't need rewiring.
    node.inputs = [X];
    // Rewrite payload to match (some passes read it for shape validation).
    if (outerPayload) outerPayload.targetShape = outerShape.slice();
    // If inner becomes orphan (no other consumers), DCE will reclaim it.
    void ctx;
    return ctx.pendingRef(node);
  },
};
