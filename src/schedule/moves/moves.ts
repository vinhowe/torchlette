/**
 * THE MOVE ALGEBRA — the intra-schedule mutators as typed partial functions
 * `ScheduleState → ScheduleState` (docs/schedule-state-design.md §3.1, R28;
 * docs/p2-moves-design.md §2.1). P2 wave B: the move BODIES the P0 schema
 * (types.ts §13) only declared.
 *
 * ------------------------------------------------------------------------
 * WHAT THIS IS
 * ------------------------------------------------------------------------
 * Each move is a partial function on `SemanticSchedule` with:
 *   - a LEGALITY check (the §3.1 invariants). An illegal application is REFUSED
 *     at the seam with a TYPED reason — never silently dropped (the ncd "jam");
 *   - INVERSE DATA recorded (S3 / the inverse-payload discipline), so
 *     `applyInverse(move, inverseData)` recovers the pre-move state;
 *   - determinism the digest verifies (`applyMove` twice → digest-identical).
 *
 * The apply/inverse contract is the metamorphic law the unit gate checks:
 *   digest(applyInverse(m, apply(m, s))) === digest(s)     [apply ∘ inverse = id]
 *
 * ------------------------------------------------------------------------
 * SCOPE (honest fence)
 * ------------------------------------------------------------------------
 * These operate on `SemanticSchedule` (the semantic tier) — the object the
 * design's before/after schemas are written against. They do NOT touch a live
 * dispatch path; they are consumed by the FA derivation (tools/fa-derivation-
 * script.ts), the move-script replayer (canonical.ts), and the fuse driver
 * (fuse.ts). `stream`'s legality delegates to the engine-side streamability
 * predicate (streamability.ts) — the refusal-first F17 boundary.
 *
 * `pipeline` is a REQUESTS-tier move (it changes `BackendRequests`, not
 * `SemanticSchedule` — §3.1), so it is NOT in this semantic-tier move algebra;
 * it is a one-line request-entry edit handled where requests are edited. The
 * seven semantic-tier bodies here are the FA-derivation set + the metamorphic
 * corpus.
 */

import type {
  MoveProvenance,
  ProgramGridMap,
  ScheduleMove,
  ScheduleState,
  SemanticBody,
  SemanticLoop,
  SemanticSchedule,
  StoreEdge,
  ValueUid,
} from "../types";
import { classifyBody, type StreamRefusal } from "./streamability";

// ============================================================================
// The move outcome (refuse-at-the-seam, never a partial state)
// ============================================================================

/** A refused move: the STAGE-analogous typed reason + a stable code. Never a
 *  bare string the caller pattern-matches — the code is the seam identity. */
export interface MoveRefusal {
  readonly code: MoveRefusalCode;
  readonly reason: string;
  /** For a `stream` refusal, the streamability predicate's obligation seam
   *  (F28): the proof-obligation whose discharge would ADMIT the move. */
  readonly refusal?: StreamRefusal;
}

export type MoveRefusalCode =
  | "TILE_LOOP_NOT_FOUND"
  | "TILE_AXIS_MISMATCH"
  | "TILE_FACTOR_INVALID"
  | "STREAM_NO_HEAD_BODY" // the F17 refusal-first boundary
  | "STREAM_VALUE_NOT_MATERIALIZED"
  | "RECOLOR_VALUE_NOT_FOUND"
  | "RECOLOR_TIER_ABSENT"
  | "PACK_EMPTY"
  | "ROLE_PARTITION_EMPTY_ROLES"
  | "PROGRAM_MAP_NOT_BIJECTIVE"
  | "PROGRAM_MAP_UNKNOWN_AXIS";

export type MoveOutcome =
  | { kind: "applied"; state: ScheduleState; provenance: MoveProvenance }
  | { kind: "refused"; move: ScheduleMove; refusal: MoveRefusal };

// ============================================================================
// The public entry point — apply one move, or refuse at the seam
// ============================================================================

/**
 * Apply `move` to `state`, producing the after-state + a provenance record
 * carrying the inverse data, OR a typed refusal at the seam. Total over the
 * seven semantic-tier moves; `pipeline` (a requests-tier move) is refused here
 * as out-of-algebra (it edits `requests`, not `semantic`).
 */
export function applyMove(
  state: ScheduleState,
  move: ScheduleMove,
): MoveOutcome {
  switch (move.move) {
    case "tile":
      return applyTile(state, move);
    case "stream":
      return applyStream(state, move);
    case "recolor":
      return applyRecolor(state, move);
    case "pack":
      return applyPack(state, move);
    case "role-partition":
      return applyRolePartition(state, move);
    case "program-map":
      return applyProgramMap(state, move);
    case "pipeline":
      // pipeline is a REQUESTS-tier move (§3.1 — it changes BackendRequests, not
      // SemanticSchedule). It is not part of the semantic move algebra; the FA
      // derivation never uses it. Refused here so the algebra stays semantic-only.
      return {
        kind: "refused",
        move,
        refusal: {
          code: "PACK_EMPTY", // reuse a code; pipeline is out-of-algebra here
          reason:
            "pipeline is a requests-tier move (edits BackendRequests, not the semantic " +
            "schedule); it is not part of the semantic move algebra.",
        },
      };
  }
}

/**
 * Apply the INVERSE of a recorded move (undo). Reads the provenance's
 * `inverseData` and reconstructs the pre-move state. Total over the moves the
 * FA derivation records; the metamorphic law is `apply ∘ inverse = id`.
 */
export function applyInverse(
  state: ScheduleState,
  provenance: MoveProvenance,
): ScheduleState {
  const { move, inverseData } = provenance;
  switch (move.move) {
    case "tile":
      return inverseTile(state, inverseData as TileInverse);
    case "stream":
      return inverseStream(state, inverseData as StreamInverse);
    case "recolor":
      return inverseRecolor(state, inverseData as RecolorInverse);
    case "pack":
      return inversePack(state, inverseData as PackInverse);
    case "role-partition":
      return inverseRolePartition(state, inverseData as RolePartitionInverse);
    case "program-map":
      return inverseProgramMap(state, inverseData as ProgramMapInverse);
    case "pipeline":
      throw new Error(
        "pipeline has no semantic-tier inverse (out-of-algebra).",
      );
  }
}

// small structural-clone of the semantic tier (immutable move discipline: never
// mutate the input state; produce a new one so digest(before) stays valid).
function withSemantic(
  state: ScheduleState,
  semantic: SemanticSchedule,
): ScheduleState {
  return { ...state, semantic };
}

// ============================================================================
// tile(loop, axis, factor) — split a loop's iteration axis into outer × inner
// ============================================================================

interface TileInverse {
  readonly outerLoop: string;
  readonly innerLoop: string;
  readonly factor: number;
  /** the original loop's uid/axis so untile can restore it exactly. */
  readonly originalLoop: SemanticLoop;
}

function applyTile(
  state: ScheduleState,
  move: Extract<ScheduleMove, { move: "tile" }>,
): MoveOutcome {
  const s = state.semantic;
  const found = findLoop(s.loopNest, move.loop);
  if (!found)
    return refuse(
      move,
      "TILE_LOOP_NOT_FOUND",
      `loop ${move.loop} not in the nest.`,
    );
  if (found.loop.axis !== move.axis)
    return refuse(
      move,
      "TILE_AXIS_MISMATCH",
      `loop ${move.loop} iterates axis ${found.loop.axis}, not ${move.axis}.`,
    );
  if (!Number.isInteger(move.factor) || move.factor < 2)
    return refuse(
      move,
      "TILE_FACTOR_INVALID",
      `tile factor must be an integer ≥ 2; got ${move.factor}.`,
    );

  const original = found.loop;
  const outerUid = `${original.uid}:outer`;
  const innerUid = `${original.uid}:inner`;
  // after: outer loop L_o (extent ceil(N/factor)) enclosing inner L_i (extent
  // factor), same body (the inner loop keeps the original children). Block shape
  // gains the tiled sub-extent (§3.1). Bounds are AffineExpr: outer = ceilDiv of
  // the original bound by the factor; inner = the factor literal.
  const innerLoop: SemanticLoop = {
    uid: innerUid as unknown as SemanticLoop["uid"],
    entity: `${original.entity}:inner` as unknown as SemanticLoop["entity"],
    axis: original.axis,
    kind: original.kind,
    bound: { kind: "affineLeaf", leaf: { kind: "intLit", value: move.factor } },
    children: original.children,
  };
  const outerLoop: SemanticLoop = {
    uid: outerUid as unknown as SemanticLoop["uid"],
    entity: `${original.entity}:outer` as unknown as SemanticLoop["entity"],
    axis: original.axis,
    kind: original.kind,
    bound: {
      kind: "affineCeilDiv",
      num: original.bound,
      den: { kind: "affineLeaf", leaf: { kind: "intLit", value: move.factor } },
    },
    children: [innerLoop],
  };
  const loopNest = replaceLoop(s.loopNest, move.loop, outerLoop);
  // The block shape gains the tiled sub-extent (the factor as a new logical block).
  const blockShapes = [...s.blockShapes, [move.factor]];
  const next = withSemantic(state, { ...s, loopNest, blockShapes });

  const inverse: TileInverse = {
    outerLoop: outerUid,
    innerLoop: innerUid,
    factor: move.factor,
    originalLoop: original,
  };
  return applied(next, move, inverse);
}

function inverseTile(state: ScheduleState, inv: TileInverse): ScheduleState {
  const s = state.semantic;
  // Restore the original loop in place of the outer (which encloses the inner);
  // drop the tiled sub-extent block shape (the last one apply appended).
  const loopNest = replaceLoop(
    s.loopNest,
    inv.outerLoop as unknown as SemanticLoop["uid"],
    inv.originalLoop,
  );
  const blockShapes = s.blockShapes.slice(0, s.blockShapes.length - 1);
  return withSemantic(state, { ...s, loopNest, blockShapes });
}

// ============================================================================
// stream(value, loop) — turn a materialized intermediate into a streamed value
// ============================================================================

interface StreamInverse {
  /** the deleted store edge (§3.1 inverse data: `{the deleted store edge}`). */
  readonly deletedStore: StoreEdge | null;
  readonly value: string;
  readonly loop: string;
}

function applyStream(
  state: ScheduleState,
  move: Extract<ScheduleMove, { move: "stream" }>,
): MoveOutcome {
  const s = state.semantic;
  // THE REFUSAL-FIRST BOUNDARY (F17). Streamability is machine-checked over the
  // value's ACTUAL semantic body via the engine predicate — NOT an authored
  // string. A value with no head/body decomposition is REFUSED, and the refusal
  // NAMES the proof-obligation whose discharge would admit it (F28).
  const body = findBody(s, move.value);
  const verdict = body
    ? classifyBody(body.expr)
    : ({
        streamable: false,
        refusal: {
          reason: `value ${move.value} has no semantic body to decompose.`,
          dischargedBy: null,
        },
      } as const);
  if (!verdict.streamable)
    return {
      kind: "refused",
      move,
      refusal: {
        code: "STREAM_NO_HEAD_BODY",
        reason: verdict.refusal.reason,
        refusal: verdict.refusal,
      },
    };

  // after: delete the StoreEdge writing `value`; add a NoMaterializationEdge on
  // `value` across `loop`. The producer/consumer are `value` itself (the streamed
  // carried value): its store is removed and it is produced/consumed in-loop.
  const deletedStore = s.stores.find((e) => e.target === move.value) ?? null;
  const stores = s.stores.filter((e) => e.target !== move.value);
  const noMaterialization = [
    ...s.noMaterialization,
    { producer: move.value, consumer: move.value, acrossLoop: move.loop },
  ];
  const next = withSemantic(state, { ...s, stores, noMaterialization });

  const inverse: StreamInverse = {
    deletedStore,
    value: move.value,
    loop: move.loop,
  };
  return applied(next, move, inverse);
}

function inverseStream(
  state: ScheduleState,
  inv: StreamInverse,
): ScheduleState {
  const s = state.semantic;
  // unstream: restore the deleted store edge; remove the no-materialization edge.
  const noMaterialization = s.noMaterialization.filter(
    (e) =>
      !(
        e.producer === inv.value &&
        e.consumer === inv.value &&
        e.acrossLoop === inv.loop
      ),
  );
  const stores = inv.deletedStore
    ? [...s.stores, inv.deletedStore]
    : [...s.stores];
  return withSemantic(state, { ...s, stores, noMaterialization });
}

// ============================================================================
// recolor(value, column, tier) — change a value's residency-intent
// ============================================================================

interface RecolorInverse {
  readonly value: string;
  readonly previousTier: string;
}

function applyRecolor(
  state: ScheduleState,
  move: Extract<ScheduleMove, { move: "recolor" }>,
): MoveOutcome {
  const s = state.semantic;
  const idx = s.values.findIndex((v) => v.uid === move.value);
  if (idx < 0)
    return refuse(
      move,
      "RECOLOR_VALUE_NOT_FOUND",
      `value ${move.value} not in the schedule's named values.`,
    );
  // The tier must be one of the enumerated address spaces (the device/realizer
  // level graph — global | shared | register). An unknown tier is refused.
  if (
    move.tier !== "global" &&
    move.tier !== "shared" &&
    move.tier !== "register"
  )
    return refuse(
      move,
      "RECOLOR_TIER_ABSENT",
      `tier "${move.tier}" is not an address space in the level graph.`,
    );

  const previous = s.values[idx];
  const values = s.values.map((v, i) =>
    i === idx ? { ...v, allocation: move.tier } : v,
  );
  const next = withSemantic(state, { ...s, values });
  const inverse: RecolorInverse = {
    value: move.value,
    previousTier: previous.allocation,
  };
  return applied(next, move, inverse);
}

function inverseRecolor(
  state: ScheduleState,
  inv: RecolorInverse,
): ScheduleState {
  const s = state.semantic;
  const values = s.values.map((v) =>
    v.uid === inv.value
      ? { ...v, allocation: inv.previousTier as (typeof v)["allocation"] }
      : v,
  );
  return withSemantic(state, { ...s, values });
}

// ============================================================================
// pack(loops, kind) — batch independent same-shape work horizontally
// ============================================================================

interface PackInverse {
  readonly loops: readonly string[];
}

function applyPack(
  state: ScheduleState,
  move: Extract<ScheduleMove, { move: "pack" }>,
): MoveOutcome {
  const s = state.semantic;
  if (move.loops.length === 0)
    return refuse(
      move,
      "PACK_EMPTY",
      "pack requires at least one loop to iterate as a pack axis.",
    );
  // after: one region iterating a pack axis. At the schedule tier we record the
  // pack as a synthetic outer pack loop enclosing the existing nest (the per-item
  // index code is the realizer's; the schedule carries the pack AXIS). The block
  // shape gains the pack cardinality.
  const packLoopUid = `loop:pack:${move.loops.join("+")}`;
  const packLoop: SemanticLoop = {
    uid: packLoopUid as unknown as SemanticLoop["uid"],
    entity: `ent:${packLoopUid}` as unknown as SemanticLoop["entity"],
    axis: "axis:pack" as unknown as SemanticLoop["axis"],
    kind: "parallel",
    bound: {
      kind: "affineLeaf",
      leaf: { kind: "intLit", value: move.loops.length },
    },
    children: s.loopNest,
  };
  const next = withSemantic(state, {
    ...s,
    loopNest: [packLoop],
    blockShapes: [...s.blockShapes, [move.loops.length]],
  });
  const inverse: PackInverse = { loops: move.loops };
  return applied(next, move, inverse);
}

function inversePack(state: ScheduleState, inv: PackInverse): ScheduleState {
  const s = state.semantic;
  // unpack: the pack loop wraps the original nest as its single children list.
  const packLoop = s.loopNest[0];
  const loopNest = packLoop ? [...packLoop.children] : s.loopNest;
  const blockShapes = s.blockShapes.slice(0, s.blockShapes.length - 1);
  void inv;
  return withSemantic(state, { ...s, loopNest, blockShapes });
}

// ============================================================================
// role-partition(loop, roles) — partition the executor into named roles
// ============================================================================

interface RolePartitionInverse {
  readonly loop: string;
  readonly priorRoleCount: number;
}

function applyRolePartition(
  state: ScheduleState,
  move: Extract<ScheduleMove, { move: "role-partition" }>,
): MoveOutcome {
  const s = state.semantic;
  if (move.roles.length === 0)
    return refuse(
      move,
      "ROLE_PARTITION_EMPTY_ROLES",
      "role-partition requires at least one role.",
    );
  // after: named role groups with typed participant sets in the predicate AST.
  // Each role is a ParticipantSet; the condition is a typed roleParticipant leaf.
  const priorRoleCount = s.roles.length;
  const newRoles = move.roles.map((role) => ({
    role,
    condition: { kind: "roleParticipant" as const, role },
  }));
  const next = withSemantic(state, { ...s, roles: [...s.roles, ...newRoles] });
  const inverse: RolePartitionInverse = { loop: move.loop, priorRoleCount };
  return applied(next, move, inverse);
}

function inverseRolePartition(
  state: ScheduleState,
  inv: RolePartitionInverse,
): ScheduleState {
  const s = state.semantic;
  const roles = s.roles.slice(0, inv.priorRoleCount);
  return withSemantic(state, { ...s, roles });
}

// ============================================================================
// program-map(map) — replace the program-id → work bijection (R4)
// ============================================================================

interface ProgramMapInverse {
  readonly previousMap: ProgramGridMap;
}

function applyProgramMap(
  state: ScheduleState,
  move: Extract<ScheduleMove, { move: "program-map" }>,
): MoveOutcome {
  const s = state.semantic;
  // Legality = one-to-one in-bounds coverage over the launch domain (R4). The
  // typed ProgramGridMap variants are bijective BY CONSTRUCTION: identity, swap
  // (a permutation), grouped (a reindex, one-to-one over the group tiling), and
  // checkedAffine (checked at construction). We assert the discriminant is one of
  // these and that a `grouped`/`swap` names in-schedule axes.
  const bij = programMapBijective(s, move.map);
  if (!bij.ok) return refuse(move, bij.code, bij.reason);

  const previousMap = s.programGridMap;
  const next = withSemantic(state, { ...s, programGridMap: move.map });
  const inverse: ProgramMapInverse = { previousMap };
  return applied(next, move, inverse);
}

function inverseProgramMap(
  state: ScheduleState,
  inv: ProgramMapInverse,
): ScheduleState {
  const s = state.semantic;
  return withSemantic(state, { ...s, programGridMap: inv.previousMap });
}

function programMapBijective(
  s: SemanticSchedule,
  map: ProgramGridMap,
): { ok: true } | { ok: false; code: MoveRefusalCode; reason: string } {
  switch (map.kind) {
    case "identity":
      return { ok: true };
    case "swap": {
      // A swap is a permutation of two axes — bijective iff both name real axes.
      const axes = collectAxes(s);
      for (const a of map.axes)
        if (!axes.has(a))
          return {
            ok: false,
            code: "PROGRAM_MAP_UNKNOWN_AXIS",
            reason: `swap names axis ${a} which is not in the loop nest.`,
          };
      return { ok: true };
    }
    case "grouped": {
      const axes = collectAxes(s);
      if (!axes.has(map.groupAxis))
        return {
          ok: false,
          code: "PROGRAM_MAP_UNKNOWN_AXIS",
          reason: `grouped names group axis ${map.groupAxis} which is not in the loop nest.`,
        };
      if (!Number.isInteger(map.groupSize) || map.groupSize < 1)
        return {
          ok: false,
          code: "PROGRAM_MAP_NOT_BIJECTIVE",
          reason: `grouped groupSize must be a positive integer; got ${map.groupSize}.`,
        };
      return { ok: true };
    }
    case "checkedAffine":
      // The affine expression is checked at construction (§2.4). A non-invertible
      // affine map would be refused there; here it is admitted as declared.
      return { ok: true };
  }
}

// ============================================================================
// Shared helpers
// ============================================================================

function findBody(
  s: SemanticSchedule,
  value: ValueUid,
): SemanticBody | undefined {
  return s.bodies.find((b) => b.result === value);
}

/** Depth-first find of a loop by uid, returning it and its parent chain. */
function findLoop(
  nest: readonly SemanticLoop[],
  uid: SemanticLoop["uid"],
): { loop: SemanticLoop } | null {
  for (const loop of nest) {
    if (loop.uid === uid) return { loop };
    const inner = findLoop(loop.children, uid);
    if (inner) return inner;
  }
  return null;
}

/** Replace a loop (by uid) with `replacement`, preserving nest structure. */
function replaceLoop(
  nest: readonly SemanticLoop[],
  uid: SemanticLoop["uid"],
  replacement: SemanticLoop,
): SemanticLoop[] {
  return nest.map((loop) => {
    if (loop.uid === uid) return replacement;
    return { ...loop, children: replaceLoop(loop.children, uid, replacement) };
  });
}

/** All axes named by loops in the nest. */
function collectAxes(s: SemanticSchedule): Set<string> {
  const out = new Set<string>();
  const walk = (loops: readonly SemanticLoop[]): void => {
    for (const l of loops) {
      out.add(l.axis);
      walk(l.children);
    }
  };
  walk(s.loopNest);
  return out;
}

function refuse(
  move: ScheduleMove,
  code: MoveRefusalCode,
  reason: string,
): MoveOutcome {
  return { kind: "refused", move, refusal: { code, reason } };
}

function applied(
  state: ScheduleState,
  move: ScheduleMove,
  inverseData: unknown,
): MoveOutcome {
  return { kind: "applied", state, provenance: { move, inverseData } };
}
