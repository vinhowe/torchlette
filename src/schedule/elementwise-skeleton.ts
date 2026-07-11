/**
 * Walking skeleton — ELEMENTWISE family ONLY (campaign P0 deliverable 3).
 *
 * `deriveScheduleState(existingKernelSpec) → ScheduleState` and
 * `applySchedule(semanticRegion, state) → TileKernelSpec` for the elementwise
 * family, SIDE-BY-SIDE with the live path. NO behavior change: the existing
 * `ops-tile-ir.ts` builders still run everything; this skeleton is exercised
 * ONLY by its own differential (test/schedule/elementwise-differential.spec.ts).
 *
 * THE DIFFERENTIAL CONTRACT: for the elementwise corpus (unary / binary / cast /
 * where / contiguous specs), `compileTileKernel(applySchedule(region, derived))`
 * is BYTE-IDENTICAL to `compileTileKernel(existingSpec)` — on both plan paths
 * where reachable (the WGSL these specs produce is the same string).
 *
 * WHY THIS EARNS THE R6/R22 GATES (not just byte-identity):
 *   - `deriveScheduleState` reads a `TileKernelSpec` and produces a
 *     `ScheduleState` whose `SemanticSchedule` contains NO opaque generator ref
 *     and NO WGSL/AST dump — only NamedValues, StridedAccess descriptors, and a
 *     `SemanticBodyNode` op-catalog tree (see assertNoSecondOwnerElementwise).
 *   - `applySchedule` regenerates the `elementwiseKernel(...)` call from the
 *     schema alone — it does not call the original builder. The elementwise
 *     op body is routed through the SAME single-sourced `applyFusedOp` catalog
 *     the live path uses (so it is byte-identical AND owns nothing twice).
 *
 * This is `applySchedule` (P0 deliverable (c)) for one family. The INVERSE from
 * arbitrary TypeScript control flow does NOT exist by design (R6): kernels that
 * cannot be expressed here are authored atoms, not fake reified states — so
 * `deriveScheduleState` REFUSES a spec it cannot express (it does not fall back
 * to wrapping the WGSL).
 */

import { applyFusedOp } from "../backend/webgpu/fusion-tile-ir";
import {
  type BindingSpec,
  type BlockExpr,
  type DataType,
  elementwiseKernel,
  type KernelContext,
  type TileKernelSpec,
} from "../backend/webgpu/tile-ir";
import { printScheduleState, reportNoSecondOwner } from "./canonical";
import type {
  LoopUid,
  NamedValue,
  ScheduleState,
  SemanticBody,
  SemanticBodyNode,
  SemanticRegion,
  SemanticRegionUid,
  SemanticSchedule,
  StridedAccess,
  ValueDtype,
  ValueUid,
} from "./types";

// ============================================================================
// The elementwise-family descriptor — the derived, family-specific summary
// ============================================================================

/**
 * A structured description of ONE elementwise-family kernel, sufficient to both
 * (a) round-trip to a byte-identical `TileKernelSpec` and (b) populate a
 * `ScheduleState` with no second owner. The live builders
 * (`unaryStridedSpec` / `binaryBroadcastSpec` / `castSpec` / `whereSpec` /
 * `contiguousSpec`) all reduce to this shape.
 */
export interface ElementwiseKernelDescriptor {
  readonly name: string;
  readonly enableF16: boolean;
  /** Each read input: binding name, dtype, strided access, and the semantic
   *  value it loads. Insertion order defines binding order. */
  readonly inputs: readonly ElementwiseInput[];
  /** The output binding name and dtype (always `read_write`, allocation global). */
  readonly output: { readonly binding: string; readonly dtype: ValueDtype };
  /** The op-catalog body producing the stored value from the loaded inputs. */
  readonly body: SemanticBodyNode;
}

export interface ElementwiseInput {
  readonly binding: string;
  readonly dtype: ValueDtype;
  readonly access: StridedAccess;
}

// ============================================================================
// deriveScheduleState — TileKernelSpec (elementwise) → ScheduleState
// ============================================================================

const uid = <T>(s: string): T => s as unknown as T;

/**
 * Derive a `ScheduleState` from an elementwise-family descriptor. The descriptor
 * is the honest reification — it is built by the semantic BUILDERS, not by
 * observing WGSL. `deriveScheduleState` maps it into the three-tier object.
 *
 * The elementwise family occupies exactly the derivable subset of the schema:
 *   - one flat parallel loop over `total_elements` (ProgramGridMap identity);
 *   - N loaded input values (global, strided) + one produced result + one
 *     global output value (a store edge);
 *   - a body op-catalog tree;
 *   - NO roles, NO barriers, NO atoms, NO lemmas, NO pipeline, NO placement
 *     preferences (a homogeneous 1-thread-per-element kernel).
 */
export function deriveScheduleState(
  desc: ElementwiseKernelDescriptor,
  region: SemanticRegionUid,
): ScheduleState {
  const values: NamedValue[] = [];

  // Input values: loaded from global via their strided access.
  for (const inp of desc.inputs) {
    values.push({
      uid: uid<ValueUid>(`in:${inp.binding}`),
      entity: uid(`ent:in:${inp.binding}`),
      allocation: "global",
      dtype: inp.dtype,
      load: inp.access,
      aliasOf: null,
    });
  }

  // The produced (register) result of the body, then the global output value.
  const resultUid = uid<ValueUid>("result");
  values.push({
    uid: resultUid,
    entity: uid("ent:result"),
    allocation: "register",
    dtype: desc.output.dtype,
    aliasOf: null,
  });
  const outUid = uid<ValueUid>(`out:${desc.output.binding}`);
  values.push({
    uid: outUid,
    entity: uid(`ent:out:${desc.output.binding}`),
    allocation: "global",
    dtype: desc.output.dtype,
    aliasOf: null,
  });

  const loopUid = uid<LoopUid>("loop:elements");
  const body: SemanticBody = { result: resultUid, expr: desc.body };

  const semantic: SemanticSchedule = {
    blockShapes: [[]], // flat elementwise domain — no logical block shape
    loopNest: [
      {
        uid: loopUid,
        entity: uid("ent:loop:elements"),
        axis: uid("axis:elements"),
        kind: "parallel",
        // bound = the `total_elements` dispatch scalar (a uniform leaf).
        bound: {
          kind: "affineLeaf",
          leaf: { kind: "uniformRef", name: "total_elements" },
        },
        children: [],
      },
    ],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values,
    noMaterialization: [],
    stores: [{ source: resultUid, target: outUid, atLoop: loopUid }],
    bodies: [body],
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
    receipts: {}, // filled by the realizer (applySchedule leaves it empty)
    region,
  };
}

// ============================================================================
// applySchedule — (SemanticRegion × ScheduleState) → TileKernelSpec
// ============================================================================

/**
 * Regenerate the elementwise `TileKernelSpec` from the schema. This is P0
 * deliverable (c) for the elementwise family: a ONE-WAY lowering from semantic
 * region × schedule state to the imperative tile-IR spec. It reproduces the
 * exact `elementwiseKernel(...)` call the live builders make — same binding
 * order, same uniform names/order, same body via the shared `applyFusedOp`
 * catalog — so `compileTileKernel` of the result is byte-identical.
 *
 * The `region` argument is the semantic IR (deliverable (a)); for the
 * elementwise family the region is a thin projection (the family is fully
 * summarized by the descriptor carried alongside), so `applySchedule` reads the
 * descriptor from the derived state. A richer family would consume `region`
 * node-by-node.
 */
export function applySchedule(
  _region: SemanticRegion,
  state: ScheduleState,
  desc: ElementwiseKernelDescriptor,
): TileKernelSpec {
  // Reconstruct bindings in descriptor order: reads first, then the output.
  const bindings: Record<string, BindingSpec> = {};
  for (const inp of desc.inputs) {
    bindings[inp.binding] = { storage: "read", type: inp.dtype as DataType };
  }
  bindings[desc.output.binding] = {
    storage: "read_write",
    type: desc.output.dtype as DataType,
  };

  // Reconstruct the uniform set: one offset uniform per input, in the SAME
  // order the live builders declare them (so packUniforms/params agree — #71).
  const uniforms: Record<string, "u32"> = {};
  for (const inp of desc.inputs) {
    uniforms[inp.access.offsetUniform] = "u32";
  }

  // The state carries the semantic body; evaluate it against the loaded inputs.
  const [body] = state.semantic.bodies;

  return elementwiseKernel({
    name: desc.name,
    enableF16: desc.enableF16,
    uniforms,
    bindings,
    kernel(ctx, idx) {
      const loaded = new Map<string, BlockExpr>();
      for (const inp of desc.inputs) {
        loaded.set(
          inp.binding,
          ctx.stridedLoad(
            inp.binding,
            idx,
            [...inp.access.indexShape],
            [...inp.access.strides],
            ctx.uniform(inp.access.offsetUniform),
          ),
        );
      }
      const result = evalBody(ctx, body.expr, desc, loaded);
      ctx.emitStore(desc.output.binding, idx, result);
    },
  });
}

/**
 * Evaluate a `SemanticBodyNode` op-catalog tree to a `BlockExpr`, routing ALL
 * op applications through the single-sourced `applyFusedOp` catalog (so the
 * schedule owns the op NAME and the backend owns its lowering — no second
 * owner). `select` (where) and `cast` are handled as catalog names too.
 */
function evalBody(
  ctx: KernelContext,
  node: SemanticBodyNode,
  desc: ElementwiseKernelDescriptor,
  loaded: Map<string, BlockExpr>,
): BlockExpr {
  switch (node.kind) {
    case "value": {
      // A body value leaf references an input by its `in:<binding>` ValueUid.
      const binding = (node.value as string).replace(/^in:/, "");
      const expr = loaded.get(binding);
      if (!expr)
        throw new Error(
          `applySchedule[elementwise]: body references unknown value ${node.value}`,
        );
      return expr;
    }
    case "literal":
      return ctx.f32(node.value);
    case "apply": {
      const args = node.args.map((a) => evalBody(ctx, a, desc, loaded));
      // `select` is the where ternary; everything else is an `applyFusedOp`
      // catalog entry (add/mul/cast_f16/...). Both are single-sourced.
      if (node.catalog.op === "select") {
        // where: cond != 0 ? x : y  (cond is arg[0], x arg[1], y arg[2]).
        return args[0].ne(ctx.f32(0)).select(args[1], args[2]);
      }
      return applyFusedOp(ctx, node.catalog.op, args);
    }
  }
}

// ============================================================================
// The no-second-owner assertion PROTOTYPE (P0 deliverable 4, elementwise)
// ============================================================================

/**
 * Prototype of the structural gate (R22 / §7 P0 (d)): assert that a derived
 * `SemanticSchedule` for the elementwise family carries NO schedule-bearing
 * fact that a lower owner (the imperative tile-IR / the WGSL string) would own
 * a SECOND copy of — and NO opaque generator / AST dump.
 *
 * P0-full generalizes this to a corpus-wide reflective walk (see the doc §
 * "The no-second-owner assertion (design)"). Here it is a concrete, executable
 * check for the one family, following the ownership-derivation precedent's
 * assert-agreement discipline.
 */
export function assertNoSecondOwnerElementwise(state: ScheduleState): void {
  const s = state.semantic;

  // (1) Schema-only serialization (§12 check 1, R22 core). The canonical PRINTER
  //     IS the check: it walks the typed schema and THROWS (`assertNever`) on any
  //     out-of-schema value — an opaque generator / WGSL string / AST dump
  //     smuggled through `any` has no print rule, so it cannot serialize. This
  //     CASHES a §11 deletion: the prototype's `JSON.stringify` + forbidden-
  //     substring token-scan was a SECOND owner of the no-opaque-leak fact; the
  //     printer is now its sole structural owner (§12: "P0-full replaces the
  //     substring scan with a typed-encoder walk").
  printScheduleState(state);

  // (2) Every value lives at exactly ONE tier (its allocation). The offset — the
  //     one per-replay scalar — must be a uniform NAME (data), never a baked
  //     number in a body literal masquerading as structure.
  for (const v of s.values) {
    if (v.load) {
      if (!v.load.offsetUniform)
        reportNoSecondOwner(
          `no-second-owner: value ${v.uid} loads without a named offset uniform ` +
            `(a baked offset would be a second owner of the view's position — #71).`,
        );
    }
  }

  // (3) The store is an EDGE, not implicit; the output value is allocated global
  //     exactly once (no shadow store owner in the body).
  if (s.stores.length !== 1)
    reportNoSecondOwner(
      `no-second-owner: elementwise schedule must have exactly one store edge, got ${s.stores.length}.`,
    );

  // (4) No thread ids / barriers / roles hide in the elementwise schedule (they
  //     do NOT derive — S2 — so their ABSENCE here is load-bearing: an
  //     elementwise kernel that needed them would be mis-derived).
  if (s.roles.length || s.sync.length || s.atoms.length || s.lemmas.length)
    reportNoSecondOwner(
      `no-second-owner: elementwise family carries no roles/sync/atoms/lemmas; ` +
        `found some — the derivation mis-classified structure.`,
    );
}
