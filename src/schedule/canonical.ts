/**
 * The canonical form — the PRINTER IS the canonical serialization (§12 check 1,
 * R27, Vin's wave-1 ruling).
 *
 * ------------------------------------------------------------------------
 * WHAT THIS IS (and what it deliberately is NOT)
 * ------------------------------------------------------------------------
 * The canonical serialization of a `ScheduleState` is a HUMAN-READABLE TEXT
 * form, not opaque bytes. `printScheduleState` walks the TYPED schema of
 * `types.ts` field by field and emits one line per fact; the strong digest is
 * `digest(printedText)`. Two properties make this the R22 core:
 *
 *   1. The printer is a TOTAL FUNCTION OVER THE SCHEMA and nothing else. Every
 *      branch is an exhaustive `switch` over a discriminated union or a walk of
 *      a declared field. An out-of-schema value (a `Function`, an `unknown`
 *      leaf, a WGSL string smuggled through `any`) has NO print rule — it hits
 *      the `assertNever` default and THROWS. You cannot serialize what the
 *      schema cannot express. This REPLACES the elementwise prototype's
 *      substring token-scan (`assertNoSecondOwnerElementwise` step 1).
 *
 *   2. There is NO PARSER. The canonical form is printer + replayer only
 *      (deferred-by-ruling: no grammar, no error-message machinery for
 *      hand-written input). A schedule program is a BASE STATE REF + an ordered
 *      list of typed move applications; `printMoveScript` emits it and
 *      `replayMoveScript` applies a script's moves to a base state, reproducing
 *      the state (digest-verified). Applying a printed script reproduces the
 *      state — that is the replay gate.
 *
 * The digest is a 128-bit FNV-1a over the printed UTF-8 text — strong (not the
 * 32-bit FNV R27 rejects), deterministic, and dependency-free (no node crypto,
 * so it runs in the browser). It reads NO Date, NO random, and NO Map/Set
 * iteration order: every collection the printer walks is either already ordered
 * (arrays) or emitted through `sortedEntries` (records), so the text — and thus
 * the digest — is stable across process restarts.
 *
 * See docs/schedule-state-design.md §5 (three separated identities) and §12
 * (the no-second-owner assertion).
 */

import { ENV } from "../core/env";
import type {
  AffineExpr,
  BackendRequests,
  CacheHint,
  LemmaApplication,
  NamedValue,
  NestOrderingRule,
  ParticipantSet,
  PipelineRequest,
  PredicateAstNode,
  ProgramGridMap,
  RealizationReceipts,
  ScheduleMove,
  ScheduleState,
  SemanticBodyNode,
  SemanticLoop,
  SemanticSchedule,
  Skeleton,
  StoreEdge,
  StridedAccess,
  SyncRelation,
  TypedParamSchema,
} from "./types";

// ============================================================================
// The versioned canonical schema tag (R27 — schema/compiler version fields)
// ============================================================================

/** Bump when the printed form changes (invalidates the semantic cache). */
export const CANONICAL_SCHEMA_VERSION = 1;

// ============================================================================
// The printer — a total walk of the typed schema
// ============================================================================

/**
 * A tiny append-only line sink. The printed form is line-oriented (one fact per
 * line, `key: value` or `kind(...)`), indentation-structured. Determinism is
 * structural — the sink never sorts; the walkers feed it in schema order.
 */
class Printer {
  private readonly lines: string[] = [];
  line(indent: number, text: string): void {
    this.lines.push("  ".repeat(indent) + text);
  }
  toString(): string {
    return this.lines.join("\n");
  }
}

/** Exhaustiveness sentinel: an out-of-schema discriminant reaches here and THROWS. */
function assertNever(x: never, ctx: string): never {
  throw new Error(
    `canonical: non-schema value in ${ctx}: ${JSON.stringify(x)} ` +
      `(the printer is a total function over the typed schema — an unknown/` +
      `Function/WGSL leaf has no print rule; R22).`,
  );
}

/** Record entries in a stable (sorted-by-key) order — no Map/Set iteration leak. */
function sortedEntries<V>(rec: Record<string, V>): Array<[string, V]> {
  return Object.keys(rec)
    .sort()
    .map((k) => [k, rec[k]] as [string, V]);
}

// ---- The typed predicate AST (§1) ----

function printPredicate(node: PredicateAstNode): string {
  switch (node.kind) {
    // Leaves
    case "loopRef":
      return `loop(${node.loop})`;
    case "valueRef":
      return `value(${node.value})`;
    case "axisRef":
      return `axis(${node.axis})`;
    case "roleParticipant":
      return `role(${node.role})`;
    case "intLit":
      return `int(${node.value})`;
    case "uniformRef":
      return `uniform(${node.name})`;
    // Combinators
    case "and":
      return `and[${node.terms.map(printPredicate).join(",")}]`;
    case "or":
      return `or[${node.terms.map(printPredicate).join(",")}]`;
    case "not":
      return `not(${printPredicate(node.term)})`;
    case "cmp":
      return `cmp(${node.op},${printPredicate(node.lhs)},${printPredicate(node.rhs)})`;
    case "range":
      return `range(${printPredicate(node.value)},${printPredicate(node.lo)},${printPredicate(node.hiExclusive)})`;
    case "member":
      return `member(${printPredicate(node.value)},[${node.set.map(printPredicate).join(",")}])`;
    // Affine
    case "affineAdd":
      return `+(${node.terms.map(printAffine).join(",")})`;
    case "affineMul":
      return `*(${node.factors.map(printAffine).join(",")})`;
    case "affineCeilDiv":
      return `ceilDiv(${printAffine(node.num)},${printAffine(node.den)})`;
    case "affineLeaf":
      return `leaf(${printPredicate(node.leaf)})`;
    default:
      return assertNever(node, "predicate");
  }
}

function printAffine(node: AffineExpr): string {
  return printPredicate(node);
}

// ---- Program grid map (§4) ----

function printProgramGridMap(m: ProgramGridMap): string {
  switch (m.kind) {
    case "identity":
      return "identity";
    case "swap":
      return `swap(${m.axes[0]},${m.axes[1]})`;
    case "grouped":
      return `grouped(${m.groupAxis},${m.groupSize})`;
    case "checkedAffine":
      return `affine(${printAffine(m.expr)})`;
    default:
      return assertNever(m, "programGridMap");
  }
}

// ---- Nest ordering (§7) ----

function printOrdering(o: NestOrderingRule): string {
  switch (o.kind) {
    case "flat":
      return "flat";
    case "rowMajor":
      return `rowMajor[${o.axes.join(",")}]`;
    case "colMajor":
      return `colMajor[${o.axes.join(",")}]`;
    default:
      return assertNever(o, "ordering");
  }
}

// ---- Strided access (§5) ----

function printAccess(a: StridedAccess): string {
  return `load(idx=[${a.indexShape.join(",")}],strides=[${a.strides.join(",")}],offset=${a.offsetUniform})`;
}

// ---- Named values (§5) ----

function printValue(p: Printer, indent: number, v: NamedValue): void {
  const parts = [
    `uid=${v.uid}`,
    `entity=${v.entity}`,
    `tier=${v.allocation}`,
    `dtype=${v.dtype}`,
    `alias=${v.aliasOf ?? "-"}`,
  ];
  // A `shared`-tier value is a STAGED tile (matmul A/B tiles, GEMV NN a-tile):
  // its residency is a workgroup-cooperative staging buffer, not an owned global.
  // The `[staged]` tag makes the staging edge legible without changing any state
  // that has no shared value (wave-0/1 states carry none, so their digest is
  // untouched — the tag is purely additive text on shared-tier lines).
  if (v.allocation === "shared") parts.push("[staged]");
  if (v.load) parts.push(printAccess(v.load));
  p.line(indent, `value ${parts.join(" ")}`);
}

// ---- Loop nest (§7) ----

function printLoop(p: Printer, indent: number, loop: SemanticLoop): void {
  p.line(
    indent,
    `loop uid=${loop.uid} entity=${loop.entity} axis=${loop.axis} ` +
      `kind=${loop.kind} bound=${printAffine(loop.bound)}`,
  );
  for (const c of loop.children) printLoop(p, indent + 1, c);
}

// ---- Semantic body (§6) ----

function printBody(node: SemanticBodyNode): string {
  switch (node.kind) {
    case "value":
      return `v(${node.value})`;
    case "literal":
      return `lit(${node.dtype}:${node.value})`;
    case "apply":
      return `${node.catalog.op}(${node.args.map(printBody).join(",")})`;
    default:
      return assertNever(node, "body");
  }
}

// ---- Store edges (§8) ----

function printStore(s: StoreEdge): string {
  return `store src=${s.source} -> tgt=${s.target} @${s.atLoop}`;
}

// ---- Sync relations (§2) ----

function printSync(r: SyncRelation): string {
  switch (r.kind) {
    case "memoryEffect":
      return `memoryEffect space=${r.space} value=${r.value} interval=${r.interval.fromLoop}..${r.interval.toLoop}`;
    case "barrier":
      return `barrier role=${r.participants.role} cond=${printPredicate(
        r.participants.condition,
      )} spaces=[${r.spaces.join(",")}] convergence=${r.convergence}`;
    case "atomic":
      return `atomic order=${r.order} visibility=${r.visibility}`;
    default:
      return assertNever(r, "sync");
  }
}

// ---- Roles (§3) ----

function printRole(ps: ParticipantSet): string {
  return `role name=${ps.role} cond=${printPredicate(ps.condition)}`;
}

// ---- The semantic tier ----

function printSemantic(p: Printer, s: SemanticSchedule): void {
  p.line(0, "semantic:");
  p.line(
    1,
    `blockShapes ${s.blockShapes.map((bs) => `[${bs.join(",")}]`).join(" ")}`,
  );
  p.line(1, `ordering ${printOrdering(s.ordering)}`);
  p.line(1, `programGridMap ${printProgramGridMap(s.programGridMap)}`);
  p.line(1, "loopNest:");
  for (const loop of s.loopNest) printLoop(p, 2, loop);
  p.line(1, "values:");
  for (const v of s.values) printValue(p, 2, v);
  p.line(1, "noMaterialization:");
  for (const e of s.noMaterialization)
    p.line(
      2,
      `noMat producer=${e.producer} consumer=${e.consumer} across=${e.acrossLoop}`,
    );
  p.line(1, "stores:");
  for (const st of s.stores) p.line(2, printStore(st));
  p.line(1, "bodies:");
  for (const b of s.bodies)
    p.line(2, `body ${b.result} = ${printBody(b.expr)}`);
  p.line(1, "roles:");
  for (const r of s.roles) p.line(2, printRole(r));
  p.line(1, "sync:");
  for (const y of s.sync) p.line(2, printSync(y));
  p.line(1, "atoms:");
  for (const a of s.atoms)
    p.line(
      2,
      `atom uid=${a.uid} op=${a.catalog.op} @${a.atLoop} role=${a.role ?? "-"} ` +
        `operands=[${a.operands.join(",")}] mult=${a.multiplicity}`,
    );
  p.line(1, "lemmas:");
  for (const l of s.lemmas)
    p.line(
      2,
      `lemma uid=${l.lemma} obligation=${l.obligation} carried=${l.carriedStateRef}`,
    );
}

// ---- The requests tier ----

function printPipeline(r: PipelineRequest): string {
  if (r.kind === "none") return "none";
  return (
    "staged[" +
    r.entries
      .map(
        (e) =>
          `entry(loop=${e.loop},loads=[${e.loadGroups.join(",")}],stages=${e.requestedStages})`,
      )
      .join(",") +
    "]"
  );
}

function printCacheHint(h: CacheHint): string {
  switch (h.kind) {
    case "loadCache":
      return `loadCache(${h.mode})`;
    case "evict":
      return `evict(${h.mode})`;
    case "volatile":
      return "volatile";
    default:
      return assertNever(h, "cacheHint");
  }
}

function printRequests(p: Printer, r: BackendRequests): void {
  p.line(0, "requests:");
  p.line(1, `warpBudget ${r.warpBudget ?? "-"}`);
  p.line(1, `pipeline ${printPipeline(r.pipeline)}`);
  p.line(1, "placementPreferences:");
  for (const pp of r.placementPreferences)
    p.line(
      2,
      `prefer value=${pp.value} tier=${pp.preferTier} interval=${pp.interval.fromLoop}..${pp.interval.toLoop}`,
    );
  p.line(1, "cachePolicy:");
  for (const h of r.cachePolicy) p.line(2, printCacheHint(h));
}

// ---- The receipts tier ----

function printReceipts(p: Printer, r: RealizationReceipts): void {
  p.line(0, "receipts:");
  if (r.workgroup) p.line(1, `workgroup [${r.workgroup.join(",")}]`);
  if (r.vecLoadForms)
    for (const v of r.vecLoadForms)
      p.line(1, `vecLoad value=${v.value} form=${v.form}`);
  if (r.realizedStages !== undefined)
    p.line(1, `realizedStages ${r.realizedStages}`);
  if (r.atomRealizations)
    for (const a of r.atomRealizations)
      p.line(1, `atomReal atom=${a.atom} realization=${a.realization}`);
  if (r.measurements)
    for (const m of r.measurements)
      p.line(
        1,
        `measure key=${m.key} ms=${m.milliseconds}` +
          (m.occupancyProxy !== undefined ? ` occ=${m.occupancyProxy}` : ""),
      );
}

/**
 * Print a full `ScheduleState` to its canonical human-readable text form. This
 * IS the canonical serialization: the digest hashes this string. The walk is a
 * total function over the schema — an out-of-schema value throws.
 */
export function printScheduleState(state: ScheduleState): string {
  const p = new Printer();
  p.line(0, `schedule-state v${CANONICAL_SCHEMA_VERSION}`);
  p.line(0, `region ${state.region}`);
  printSemantic(p, state.semantic);
  printRequests(p, state.requests);
  printReceipts(p, state.receipts);
  return p.toString();
}

/**
 * Print ONLY the semantic tier + region (the SEMANTIC identity input, §5). The
 * three separated identities each hash a different slice; this is the input to
 * `semanticDigest`.
 */
export function printSemanticIdentity(state: ScheduleState): string {
  const p = new Printer();
  p.line(0, `semantic-identity v${CANONICAL_SCHEMA_VERSION}`);
  p.line(0, `region ${state.region}`);
  printSemantic(p, state.semantic);
  return p.toString();
}

// ============================================================================
// The authored (opaque) skeleton print form (§6 / F3) — typed params VISIBLE,
// skeleton SEALED. The authored hatch is the ONLY permitted opacity: the print
// shows the kernel reference, the refusal reason, and the DECLARED typed
// parameter schema (params + dependent constraints + capability predicate), but
// the skeleton's internals stay opaque (F3 forbids loop/staging/role data — the
// opaque `Skeleton` variant has no field to hold them, so there is nothing to
// print). This makes an authored kernel legible WITHOUT letting it masquerade
// as a derived state: a `derived` skeleton prints its full ScheduleState; an
// `opaque` one prints only its declared surface.
// ============================================================================

/** Print a typed parameter schema (§6 R10/F7): params with domains + defaults,
 *  dependent constraints, and the capability predicate — a total schema walk. */
function printParamSchema(
  p: Printer,
  indent: number,
  s: TypedParamSchema,
): void {
  p.line(indent, "params:");
  for (const [name, spec] of sortedEntries(s.params))
    p.line(
      indent + 1,
      `param ${name} domain=[${spec.domain.join(",")}] default=${spec.default}`,
    );
  p.line(indent, "constraints:");
  for (const c of s.constraints) p.line(indent + 1, printPredicate(c));
  p.line(indent, `capability ${printPredicate(s.capabilityPredicate)}`);
}

/**
 * Print an admitted-lemma application (§3.4 F27/F28) — its LemmaUid, the
 * proof-obligation it discharges, and its first-class carried-state reference.
 */
function printLemma(l: LemmaApplication): string {
  return `lemma uid=${l.lemma} obligation=${l.obligation} carried=${l.carriedStateRef}`;
}

/**
 * Print an authored (opaque) or derived skeleton (§6 / F3). A `derived`
 * skeleton delegates to the full `printScheduleState`; an `opaque` one prints
 * the `authored` block: the sealed kernelRef + refusal reason + the typed
 * params. Optional `lemmas` (the admitted-lemma applications the authored kernel
 * carries, e.g. online-softmax) print under the block. Total over the union.
 */
export function printSkeleton(
  skeleton: Skeleton,
  lemmas: readonly LemmaApplication[] = [],
): string {
  if (skeleton.visibility === "derived")
    return printScheduleState(skeleton.schedule);
  if (skeleton.visibility === "opaque") {
    const p = new Printer();
    p.line(0, `authored-skeleton v${CANONICAL_SCHEMA_VERSION}`);
    p.line(0, "authored:");
    p.line(1, `kernelRef ${skeleton.kernelRef}`);
    p.line(1, `refusal ${skeleton.refusalReason}`);
    p.line(1, "skeleton sealed=opaque (F3: no loop/staging/role data)");
    printParamSchema(p, 1, skeleton.params);
    p.line(1, "admittedLemmas:");
    for (const l of lemmas) p.line(2, printLemma(l));
    return p.toString();
  }
  return assertNever(skeleton, "skeleton");
}

/** The digest of an authored skeleton's canonical print (the authored analogue
 *  of `scheduleDigest`). */
export function skeletonDigest(
  skeleton: Skeleton,
  lemmas: readonly LemmaApplication[] = [],
): string {
  return digestText(printSkeleton(skeleton, lemmas));
}

// ============================================================================
// The move-script — the SECOND textual artifact (Vin's wave-1 ruling)
// ============================================================================

/**
 * A schedule PROGRAM: a base-state reference + an ordered list of typed move
 * applications (the mutation-contract export format, formalized). Printable and
 * replayable — applying a printed script to its base reproduces the state
 * (digest-verified). This is NOT a parser input format; it is printer output +
 * a replayer that consumes the in-memory `MoveScript`, not the text.
 */
export interface MoveScript {
  /** The canonical digest of the base `ScheduleState` the script starts from. */
  readonly baseDigest: string;
  /** The ordered typed moves. */
  readonly moves: readonly ScheduleMove[];
}

/** Print one typed move to its canonical one-line form (total over the union). */
export function printMove(m: ScheduleMove): string {
  switch (m.move) {
    case "tile":
      return `tile loop=${m.loop} axis=${m.axis} factor=${m.factor}`;
    case "stream":
      return `stream value=${m.value} loop=${m.loop}`;
    case "recolor":
      return `recolor value=${m.value} column=${m.column} tier=${m.tier} role=${m.transitionRole}`;
    case "pack":
      return `pack loops=[${m.loops.join(",")}] kind=${m.kind}`;
    case "role-partition":
      return `role-partition loop=${m.loop} roles=[${m.roles.join(",")}]`;
    case "pipeline":
      return `pipeline loop=${m.loop} loads=[${m.loadGroups.join(",")}] stages=${m.requestedStages}`;
    case "program-map":
      return `program-map map=${printProgramGridMap(m.map)}`;
    default:
      return assertNever(m, "move");
  }
}

/** Print a move-script to canonical text (base ref + ordered move lines). */
export function printMoveScript(script: MoveScript): string {
  const lines = [
    `move-script v${CANONICAL_SCHEMA_VERSION}`,
    `base ${script.baseDigest}`,
    ...script.moves.map((m, i) => `${i}: ${printMove(m)}`),
  ];
  return lines.join("\n");
}

// ============================================================================
// The strong digest — 128-bit FNV-1a over the printed UTF-8 text (R27)
// ============================================================================

// FNV-1a 128-bit parameters. BigInt keeps the full 128 bits (not a 32-bit FNV).
const FNV_OFFSET_128 = 0x6c62272e07bb014262b821756295c58dn;
const FNV_PRIME_128 = 0x0000000001000000000000000000013bn;
const MASK_128 = (1n << 128n) - 1n;

/** UTF-8 bytes of a string, portable across Node and browser. */
function utf8Bytes(s: string): Uint8Array {
  if (typeof TextEncoder !== "undefined") return new TextEncoder().encode(s);
  // Minimal manual UTF-8 (defensive; TextEncoder is universal in our targets).
  const out: number[] = [];
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
    else
      out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
  }
  return Uint8Array.from(out);
}

/** Strong 128-bit digest of an arbitrary UTF-8 string, as 32 lowercase hex chars. */
export function digestText(text: string): string {
  let hash = FNV_OFFSET_128;
  const bytes = utf8Bytes(text);
  for (let i = 0; i < bytes.length; i++) {
    hash ^= BigInt(bytes[i]);
    hash = (hash * FNV_PRIME_128) & MASK_128;
  }
  return hash.toString(16).padStart(32, "0");
}

// ============================================================================
// The strict-mode gate for §12 check 3 (no new env flag — #92 seam-guard host)
// ============================================================================

let _warnedNoSecondOwner = false;

/**
 * Whether the no-second-owner runtime assertion at `applySchedule` THROWS
 * (strict) or warns. Reuses the EXISTING strict-lifetime soak flag — no new
 * `TORCHLETTE_*` flag is born (house complexity budget). Strict by default;
 * `TORCHLETTE_STRICT_LIFETIME=0` downgrades to warn during the soak window,
 * matching op-dispatch's reclaimed-read guard exactly.
 */
export function scheduleStrict(): boolean {
  return ENV.TORCHLETTE_STRICT_LIFETIME !== "0";
}

/**
 * Report a no-second-owner violation at the seam. Throws under strict (default);
 * warns once under the soak opt-out. This is the §12 check-3 enforcement point:
 * called by every family's `applySchedule` seam assertion on a REAL lowering.
 */
export function reportNoSecondOwner(message: string): void {
  if (scheduleStrict()) throw new Error(message);
  if (!_warnedNoSecondOwner) {
    _warnedNoSecondOwner = true;
    // eslint-disable-next-line no-console
    console.warn(
      `[schedule] ${message} (warn-only; TORCHLETTE_STRICT_LIFETIME=0)`,
    );
  }
}

/**
 * The FULL canonical digest of a `ScheduleState` — `digest(printedText)`.
 * A cache hit compares the full printed text (R27: FULL canonical equality on
 * every hit; sampling audits regeneration only). This is the artifact-cache
 * coordinate's content half.
 */
export function scheduleDigest(state: ScheduleState): string {
  return digestText(printScheduleState(state));
}

/**
 * The SEMANTIC identity digest — `digest(SemanticSchedule + region)` (§5). Two
 * states with the same semantic tier but different requests/receipts share this.
 */
export function semanticDigest(state: ScheduleState): string {
  return digestText(printSemanticIdentity(state));
}

/**
 * The REALIZER COORDINATE (§5 artifact-cache identity). The artifact-cache
 * identity = compilation identity + "the realization-receipt coordinate keyed to
 * the produced binary". A `ScheduleState` is realizer-agnostic (the same object
 * lowers through WGSL/Dawn OR Triton/CUDA — the cross-backend differential's whole
 * premise), so the produced ARTIFACT is keyed by WHICH realizer emitted it. This
 * is the coordinate that field: two byte-different artifacts (one WGSL, one Triton
 * source) from the ONE schedule digest must not collide in the artifact cache.
 *
 * A realizer coordinate names the realizer + its capability-profile version + the
 * pinned target (Appendix A: a v2 profile must pin a release/commit + target arch).
 */
export interface RealizerCoordinate {
  /** The realizer that produced the artifact (e.g. "wgsl-dawn", "triton"). */
  readonly realizer: string;
  /** The capability-profile version (§5 compilation-identity input). */
  readonly capabilityProfileVersion: number;
  /** The pinned target the artifact was emitted for (e.g. "sm_70", "webgpu"). */
  readonly targetArch: string;
}

/** Canonically serialize a realizer coordinate (sorted, explicit — R27). */
export function printRealizerCoordinate(coord: RealizerCoordinate): string {
  return (
    `realizer ${coord.realizer}` +
    ` profile-v${coord.capabilityProfileVersion}` +
    ` target ${coord.targetArch}`
  );
}

/**
 * The ARTIFACT identity digest (§5) — the schedule's FULL canonical print PLUS
 * the realizer coordinate. Two realizers lowering the SAME `ScheduleState` produce
 * DISTINCT artifact identities (their emitted text differs), so the artifact cache
 * keys correctly across the cross-backend split. `scheduleDigest` (realizer-blind)
 * is the artifact-cache coordinate's CONTENT half; this composes the realizer
 * coordinate onto it.
 */
export function artifactDigest(
  state: ScheduleState,
  coord: RealizerCoordinate,
): string {
  return digestText(
    `${printScheduleState(state)}\n${printRealizerCoordinate(coord)}`,
  );
}
