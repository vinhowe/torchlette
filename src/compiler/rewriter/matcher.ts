/**
 * Pattern matcher: given a Pattern and a LazyRef, return Bindings or null.
 *
 * The matcher is a pure function. It does NOT mutate the pattern, the ref,
 * or any node it traverses. It collects captures into a fresh Bindings map
 * on each call.
 *
 * Matching rules:
 *   - `any`: succeeds against anything.
 *   - `capture(name)`: binds `name` to the ref (or verifies prior binding).
 *   - `capture(name, inner)`: like capture, but ref must also match `inner`.
 *   - `op(code, { inputs })`: ref must be pending, point to a node with
 *     matching op, and (if `inputs` given) arity must match exactly and
 *     each input ref must match the corresponding sub-pattern.
 *   - `scalar`: ref must be a scalar ref; optional predicate on value.
 *   - `materialized`: ref must be a materialized ref.
 */
import type { LazyRef } from "../../graph/types";
import type { Bindings, MutableBindings, Pattern } from "./pattern";

/** Match `pattern` against `ref`. Returns a fresh Bindings on success,
 *  or `null` on failure. */
export function match(pattern: Pattern, ref: LazyRef): Bindings | null {
  const bindings = new Map<string, LazyRef>();
  if (matchInto(pattern, ref, bindings)) return bindings;
  return null;
}

/** Internal: match into an existing bindings map (mutates on success). */
export function matchInto(
  pattern: Pattern,
  ref: LazyRef,
  bindings: MutableBindings,
): boolean {
  switch (pattern.kind) {
    case "any":
      return true;

    case "capture": {
      // If `inner` is provided, ref must also match it — evaluate first so
      // we don't leave a stray binding on failure.
      if (pattern.inner && !matchInto(pattern.inner, ref, bindings)) {
        return false;
      }
      const existing = bindings.get(pattern.name);
      if (existing !== undefined) {
        return refEquals(existing, ref);
      }
      bindings.set(pattern.name, ref);
      return true;
    }

    case "op": {
      if (ref.kind !== "pending") return false;
      const node = ref.node;
      if (node.op !== pattern.op) return false;
      if (pattern.where && !pattern.where(node)) return false;
      if (pattern.inputs !== undefined) {
        if (node.inputs.length !== pattern.inputs.length) return false;
        // Try all input matches; if any fails, roll back bindings added
        // during this op-pattern's traversal.
        const saved = snapshotBindings(bindings);
        for (let i = 0; i < pattern.inputs.length; i++) {
          if (!matchInto(pattern.inputs[i], node.inputs[i], bindings)) {
            restoreBindings(bindings, saved);
            return false;
          }
        }
      }
      return true;
    }

    case "scalar":
      if (ref.kind !== "scalar") return false;
      if (pattern.where && !pattern.where(ref.value)) return false;
      return true;

    case "materialized":
      return ref.kind === "materialized";
  }
}

// ============================================================================
// Ref equality
// ============================================================================

/** Structural equality for LazyRefs.
 *  - pending: same node id AND same outputIndex
 *  - materialized: same storage id
 *  - scalar: same value AND same dtype
 *  Cross-kind refs are never equal. */
export function refEquals(a: LazyRef, b: LazyRef): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "pending" && b.kind === "pending") {
    return (
      a.node.id === b.node.id &&
      (a.outputIndex ?? 0) === (b.outputIndex ?? 0)
    );
  }
  if (a.kind === "materialized" && b.kind === "materialized") {
    return a.storage.id === b.storage.id;
  }
  if (a.kind === "scalar" && b.kind === "scalar") {
    return a.value === b.value && a.dtype === b.dtype;
  }
  return false;
}

// ============================================================================
// Bindings rollback (for backtracking in nested matches)
// ============================================================================

function snapshotBindings(b: MutableBindings): string[] {
  return [...b.keys()];
}

function restoreBindings(b: MutableBindings, keysBefore: string[]): void {
  const allowed = new Set(keysBefore);
  for (const k of [...b.keys()]) {
    if (!allowed.has(k)) b.delete(k);
  }
}
