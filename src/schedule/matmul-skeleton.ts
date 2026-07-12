/**
 * Walking skeleton — MATMUL family (campaign P0-FULL wave 2). Side-by-side with
 * the live path; NO behavior change; NO dispatch cutover (that is P1). Exercised
 * only by test/schedule/matmul-differential.spec.ts.
 *
 * ------------------------------------------------------------------------
 * THE DIFFERENTIAL CONTRACT (same bar as elementwise / reduction)
 * ------------------------------------------------------------------------
 * For the matmul corpus (tiled + epilogue variants, GEMV NT/NN + epilogue +
 * quantB, split-K partials + reduction, batched, swap-grid),
 *
 *   compileTileKernel(applyMatmulSchedule*(deriveMatmulState*(k))) ==BYTES==
 *   <the live generateShader for k>
 *
 * where the live single-source generators are:
 *   - tiled/batched/swapGrid/kSplit-partials: generateTiledMatmulShaderTileIR(CodegenOptions)
 *   - split-K reduction pass:                 generateKSplitReductionShaderTileIR(count, dtype)
 *   - GEMV NT/NN + epilogue + quantB:         generateGemvShaderTileIR(GemvKernelOptions)
 *
 * ------------------------------------------------------------------------
 * THE TIER MAPPING (docs/schedule-state-design.md §11.4 — implemented exactly)
 * ------------------------------------------------------------------------
 *   tileM / tileN / tileK          → SemanticSchedule.blockShapes (rung-3 reuse).
 *   threadTileM/N, vectorWidth,    → RealizationReceipts (NEVER semantic identity
 *     exact vec-load form            — A-R1/A-R6).
 *   useSubgroups                   → BackendRequests.
 *   kSplit                         → the `tile` move on the reduction (K) axis —
 *                                     SEMANTIC (carries the fp-reorder lemma
 *                                     license as its admitted-lemma reference).
 *   swapGrid                       → ProgramGridMap {kind:"swap"} (the 8th move's
 *                                     first real derivation; R4 reification test).
 *   epilogue chain                 → no-materialization edges (register→register,
 *                                     no global store per op). epilogue ⊥ kSplit
 *                                     is a TYPED legality rule on the object.
 *   transposeMode / simple-transp. → no-materialization view facts (transpose flag,
 *                                     never a materialized contiguous copy).
 *   shared-memory staging (A/B     → the FIRST real staging edges + typed sync
 *     tiles, barriers, K-loop)       relations (Barrier participants/spaces, R3) +
 *                                     roles. STORED facts (S2 — they do NOT derive);
 *                                     wave 2 is their schema shakedown.
 *   validateConfig                 → the R10 typed parameter schema with dependent
 *                                     constraints (this family's declared constraint
 *                                     set); no-second-owner sees its facts owned ONCE.
 *
 * The matmul kernels REFUSE a spec they cannot express — they never wrap WGSL.
 * The staging/barriers of the TILED kernel live below `ctx.dotAccum` /
 * `ctx.load2D` in the live path (the block-op realizer owns their lowering); wave
 * 2 STORES them as semantic facts (the schema shakedown), and the no-second-owner
 * seam asserts they agree with the geometry the descriptor round-trips — they are
 * not a SECOND owner of the barrier lowering, they are the schedule's declaration
 * of the staging INTENT the realizer honors.
 */

import {
  type GemvKernelOptions,
  gemvSupportsEpilogue,
  generateGemvShaderTileIR,
  type QuantB,
} from "../backend/webgpu/matmul/gemv";
import {
  createTiledMatmulKernel,
  generateKSplitReductionShaderTileIR,
} from "../backend/webgpu/matmul/tile-matmul";
import type {
  CodegenOptions,
  EpilogueConfig,
  DType as MatmulDType,
  MatmulKernelConfig,
  TransposeMode,
} from "../backend/webgpu/matmul/types";
import { validateConfig } from "../backend/webgpu/matmul/types";
import type { TileKernelSpec } from "../backend/webgpu/tile-ir";
import { printScheduleState, reportNoSecondOwner } from "./canonical";
import type {
  AffineExpr,
  AxisUid,
  BackendRequests,
  LemmaUid,
  LoopUid,
  NamedValue,
  ObligationId,
  ParticipantSet,
  PredicateAstNode,
  ProgramGridMap,
  RealizationReceipts,
  RoleName,
  ScheduleState,
  SemanticBody,
  SemanticBodyNode,
  SemanticLoop,
  SemanticRegionUid,
  SemanticSchedule,
  StoreEdge,
  SyncRelation,
  TypedParamSchema,
  ValueDtype,
  ValueUid,
} from "./types";

const uid = <T>(s: string): T => s as unknown as T;

/** matmul dtypes (f16|f32) as ValueDtype. */
function toValueDtype(d: MatmulDType): ValueDtype {
  return d; // "f16" | "f32" are both ValueDtype members
}

// ============================================================================
// The K-split fp-reorder admitted lemma (§3.4) — kSplit's license
// ============================================================================

/**
 * K-split partitions the reduction axis across programs and sums the partials
 * in a second pass. Summing in a different order is NOT bit-identical fp
 * arithmetic — it is licensed by an admitted lemma (associativity-of-fp-sum,
 * accepted within the training tolerance envelope), not a free structural move.
 * The `tile`-on-K move carries this lemma as its admitted-lemma reference (the
 * tier mapping's "carries the fp-reorder lemma license" clause). Present as a
 * FIRST-CLASS lemma application so two states that differ only in kSplit are
 * DISTINGUISHABLE by their lemma set (F27/F28).
 */
const KSPLIT_LEMMA = uid<LemmaUid>("lemma:fp-sum-reassociation");
const KSPLIT_OBLIGATION = uid<ObligationId>("obl:ksplit-fp-reorder");

// ============================================================================
// The typed parameter schema (R10 + F7) — validateConfig's dependent constraints
// ============================================================================

/**
 * The matmul family's declared constraint set (§6, R10/F7). `validateConfig`
 * (types.ts) is the SINGLE OWNER of the dependent constraints (tile divisibility,
 * workgroup ≤ 256, shared-memory ≤ 16KB); this schema names them as typed
 * predicate-AST facts so an authored matmul kernel publishes them without a
 * second copy. The `applyMatmulSchedule` seam CALLS `validateConfig` — it does
 * not re-implement the arithmetic — so the facts are owned once (§12 check 3).
 *
 * The predicate leaves reference the config axes by name via `uniformRef` (the
 * schedule-context leaf domain); the numeric domains mirror `TUNING_SPACE`.
 */
export const MATMUL_PARAM_SCHEMA: TypedParamSchema = {
  params: {
    tileM: { domain: [32, 64, 128], default: 32 },
    tileN: { domain: [32, 64, 128], default: 32 },
    tileK: { domain: [8, 16, 32], default: 16 },
    threadTileM: { domain: [4, 8], default: 4 },
    threadTileN: { domain: [4, 8], default: 4 },
  },
  // Dependent constraints as typed predicate-AST nodes (display only in P0 —
  // the executable check is validateConfig at the seam; these are the DECLARED
  // shape of the same facts so an authored kernel can publish them).
  constraints: [
    // tileM % threadTileM == 0
    divisibleConstraint("tileM", "threadTileM"),
    // tileN % threadTileN == 0
    divisibleConstraint("tileN", "threadTileN"),
    // (tileM/threadTileM) * (tileN/threadTileN) <= 256   (workgroup size)
    workgroupConstraint(),
    // (tileM*tileK + tileK*tileN) * 4 <= 16384            (shared memory)
    sharedMemConstraint(),
  ],
  // The capability predicate: subgroup usage requires device support (a REQUEST,
  // F9). Bare tiled needs no capability; the predicate is trivially-true here and
  // sharpens when `useSubgroups` is requested.
  capabilityPredicate: { kind: "intLit", value: 1 },
};

function paramLeaf(name: string): PredicateAstNode {
  return { kind: "uniformRef", name };
}
function divisibleConstraint(a: string, b: string): PredicateAstNode {
  // Encoded as cmp(mod(a,b) == 0) — `mod` is a display projection; the executable
  // check is validateConfig. The AST names the two operands so the constraint has
  // exactly one owner (the config axes), no baked number.
  return {
    kind: "cmp",
    op: "==",
    lhs: {
      kind: "member",
      value: paramLeaf(a),
      set: [paramLeaf(b)],
    },
    rhs: { kind: "intLit", value: 0 },
  };
}
function workgroupConstraint(): PredicateAstNode {
  return {
    kind: "cmp",
    op: "<=",
    lhs: {
      kind: "and",
      terms: [
        paramLeaf("tileM"),
        paramLeaf("threadTileM"),
        paramLeaf("tileN"),
        paramLeaf("threadTileN"),
      ],
    },
    rhs: { kind: "intLit", value: 256 },
  };
}
function sharedMemConstraint(): PredicateAstNode {
  return {
    kind: "cmp",
    op: "<=",
    lhs: {
      kind: "and",
      terms: [paramLeaf("tileM"), paramLeaf("tileK"), paramLeaf("tileN")],
    },
    rhs: { kind: "intLit", value: 16384 },
  };
}

// ============================================================================
// FAMILY 3a — TILED matmul (incl. epilogue, batched, swap-grid, split-K partials)
// ============================================================================

/**
 * A structured description of ONE tiled-matmul kernel — the honest reification
 * of `CodegenOptions`, built by the dispatch geometry, not by observing WGSL.
 * It round-trips byte-identically via `generateTiledMatmulShaderTileIR` and
 * populates a `ScheduleState` with no second owner.
 */
export interface TiledMatmulDescriptor {
  readonly config: MatmulKernelConfig;
  readonly transposeMode: TransposeMode;
  readonly dtype: MatmulDType;
  readonly dtypeB?: MatmulDType;
  readonly epilogue?: EpilogueConfig;
  readonly batched?: boolean;
  readonly inputCastA?: MatmulDType;
  readonly inputCastB?: MatmulDType;
  /** ≥ 2 → this kernel writes RAW f32 partials for split index programId(2). */
  readonly kSplit?: number;
  readonly swapGrid?: boolean;
}

/**
 * Derive a `ScheduleState` from a tiled-matmul descriptor.
 *
 * The tiled matmul is a 2D PARALLEL output domain (one program per BLOCK_M×BLOCK_N
 * output tile, program ids on axes M and N; a batch/split axis on programId(2))
 * enclosing a SEQUENTIAL reduction loop over ceil(K/tileK) K-tiles — that nest is
 * the derivable structural spine (S2). On top of the spine, wave 2 STORES the
 * facts S2 says do NOT derive: the shared-memory staging edges for the A and B
 * tiles, the per-K-tile workgroup barrier, and the cooperative-load role.
 */
export function deriveTiledMatmulState(
  desc: TiledMatmulDescriptor,
  region: SemanticRegionUid,
): ScheduleState {
  const { config } = desc;
  const dtypeB = desc.dtypeB ?? desc.dtype;
  // Match createTiledMatmulKernel: kSplit forces f32 output (raw partials).
  const outputDtype: MatmulDType = desc.kSplit
    ? "f32"
    : (desc.epilogue?.outputDtype ??
      (desc.dtype === "f32" || dtypeB === "f32" ? "f32" : desc.dtype));

  // ---- Named values: A, B (global loads), the A/B shared tiles (staged),
  //      the register accumulator, epilogue inputs, and the global output. ----
  const values: NamedValue[] = [];
  const aUid = uid<ValueUid>("in:a");
  const bUid = uid<ValueUid>("in:b");
  const aTileUid = uid<ValueUid>("stage:a_tile");
  const bTileUid = uid<ValueUid>("stage:b_tile");
  const accUid = uid<ValueUid>("acc");
  const outUid = uid<ValueUid>("out:out");

  values.push({
    uid: aUid,
    entity: uid("ent:in:a"),
    allocation: "global",
    dtype: toValueDtype(desc.inputCastA ? "f32" : desc.dtype),
    aliasOf: null,
  });
  values.push({
    uid: bUid,
    entity: uid("ent:in:b"),
    allocation: "global",
    dtype: toValueDtype(desc.inputCastB ? "f32" : dtypeB),
    aliasOf: null,
  });
  // The staged shared-memory tiles (the schema shakedown): allocation "shared".
  values.push({
    uid: aTileUid,
    entity: uid("ent:stage:a_tile"),
    allocation: "shared",
    dtype: toValueDtype(desc.dtype),
    aliasOf: null,
  });
  values.push({
    uid: bTileUid,
    entity: uid("ent:stage:b_tile"),
    allocation: "shared",
    dtype: toValueDtype(dtypeB),
    aliasOf: null,
  });
  // The register accumulator (F1 register-residency intent — matmul's honest
  // instance of operand-residency).
  values.push({
    uid: accUid,
    entity: uid("ent:acc"),
    allocation: "register",
    dtype: "f32",
    aliasOf: null,
  });
  // Epilogue inputs (bias / residual) are global reads with no store of their own.
  const epiInputs: ValueUid[] = [];
  if (desc.epilogue) {
    for (let i = 0; i < desc.epilogue.additionalInputCount; i++) {
      const u = uid<ValueUid>(`in:epilogue_in${i}`);
      epiInputs.push(u);
      values.push({
        uid: u,
        entity: uid(`ent:in:epilogue_in${i}`),
        allocation: "global",
        dtype: "f32",
        aliasOf: null,
      });
    }
  }
  values.push({
    uid: outUid,
    entity: uid("ent:out:out"),
    allocation: "global",
    dtype: toValueDtype(desc.epilogue?.outputDtype ?? outputDtype),
    aliasOf: null,
  });

  // ---- The loop nest: parallel M, parallel N (block program), sequential K. ----
  const axM = uid<AxisUid>("axis:m");
  const axN = uid<AxisUid>("axis:n");
  const axK = uid<AxisUid>("axis:k");
  const loopM = uid<LoopUid>("loop:m");
  const loopN = uid<LoopUid>("loop:n");
  const loopK = uid<LoopUid>("loop:k");

  const kTilesBound: AffineExpr = {
    kind: "affineCeilDiv",
    num: { kind: "affineLeaf", leaf: { kind: "uniformRef", name: "k" } },
    den: { kind: "affineLeaf", leaf: { kind: "intLit", value: config.tileK } },
  };
  const kLoop: SemanticLoop = {
    uid: loopK,
    entity: uid("ent:loop:k"),
    axis: axK,
    kind: "sequential",
    bound: kTilesBound,
    children: [],
  };
  const nLoop: SemanticLoop = {
    uid: loopN,
    entity: uid("ent:loop:n"),
    axis: axN,
    kind: "parallel",
    bound: ceilDivUniform("n", config.tileN),
    children: [kLoop],
  };
  const mLoop: SemanticLoop = {
    uid: loopM,
    entity: uid("ent:loop:m"),
    axis: axM,
    kind: "parallel",
    bound: ceilDivUniform("m", config.tileM),
    children: [nLoop],
  };

  // ---- Roles + sync: the cooperative-load role + the per-K-tile barrier.
  //      These are STORED facts (S2 — they do NOT derive). The barrier
  //      participates in the shared address space across both tile stores. ----
  const coopRole = uid<RoleName>("cooperative-load");
  const roles: ParticipantSet[] = [
    {
      role: coopRole,
      // The whole workgroup cooperatively stages the A/B tiles (all invocations).
      condition: { kind: "intLit", value: 1 },
    },
  ];
  const sync: SyncRelation[] = [
    // Store into the A tile is visible before the compute reads it.
    {
      kind: "memoryEffect",
      space: "shared",
      value: aTileUid,
      interval: { fromLoop: loopK, toLoop: loopK },
    },
    {
      kind: "memoryEffect",
      space: "shared",
      value: bTileUid,
      interval: { fromLoop: loopK, toLoop: loopK },
    },
    // The per-K-tile workgroup barrier (uniform control flow — the tail groups
    // recompute so barriers stay convergent, matching the live kernels).
    {
      kind: "barrier",
      participants: { role: coopRole, condition: { kind: "intLit", value: 1 } },
      spaces: ["shared"],
      convergence: "uniform",
    },
  ];

  // ---- The bodies: the dot accumulation, then the epilogue chain (if any). ----
  const bodies: SemanticBody[] = [];
  // The reduction body: acc += dot(a_tile, b_tile), rooted at the matmul contract.
  bodies.push({
    result: accUid,
    expr: {
      kind: "apply",
      catalog: { op: "dot_accum" },
      args: [
        { kind: "value", value: aTileUid },
        { kind: "value", value: bTileUid },
      ],
    },
  });
  // The store body: alpha-scaled accumulator, then the epilogue chain.
  const storeResultUid = uid<ValueUid>("write");
  bodies.push({
    result: storeResultUid,
    expr: epilogueToBody(accUid, desc, epiInputs),
  });

  // ---- The no-materialization (fusion) edges: each epilogue op consumes the
  //      accumulator WITHOUT a global store between ops (register→register). ----
  const noMat = desc.kSplit
    ? []
    : (desc.epilogue?.ops ?? [])
        .filter((o) => o.kind !== "none")
        .map(() => ({
          producer: accUid,
          consumer: storeResultUid,
          acrossLoop: loopM,
        }));

  // ---- The store edge (an EDGE, F6). kSplit writes to the split-indexed base. ----
  const store: StoreEdge = {
    source: storeResultUid,
    target: outUid,
    atLoop: loopM,
  };

  // ---- The program-grid map: swapGrid → {kind:"swap", axes:[m,n]}. ----
  const programGridMap: ProgramGridMap = desc.swapGrid
    ? { kind: "swap", axes: [axM, axN] }
    : { kind: "identity" };

  // ---- kSplit is a `tile` of the K axis — recorded as a lemma application. ----
  const lemmas = desc.kSplit
    ? [
        {
          lemma: KSPLIT_LEMMA,
          obligation: KSPLIT_OBLIGATION,
          carriedStateRef: `ksplit=${desc.kSplit}`,
        },
      ]
    : [];

  const semantic: SemanticSchedule = {
    // Logical block shapes: the M×N output block, and the K reduction block.
    blockShapes: [[config.tileM, config.tileN], [config.tileK]],
    loopNest: [mLoop],
    ordering: { kind: "rowMajor", axes: [axM, axN] },
    programGridMap,
    values,
    noMaterialization: noMat,
    stores: [store],
    bodies,
    roles,
    sync,
    atoms: [],
    lemmas,
  };

  // useSubgroups → BackendRequests (a capability request, F9); threadTile /
  // vectorWidth are RECEIPTS (never semantic). warpBudget stays null (the
  // realizer default) — the workgroup geometry is a receipt.
  const requests: BackendRequests = {
    warpBudget: null,
    pipeline: { kind: "none" },
    placementPreferences: [
      // acc lives in registers (F1 residency intent → a request preference).
      {
        value: accUid,
        preferTier: "register",
        interval: { fromLoop: loopK, toLoop: loopM },
      },
    ],
    cachePolicy: [],
  };

  // Receipts start empty (the realizer fills them). We record the compilation-
  // derived facts the realizer WOULD report so the printer exercises them and
  // downstream P1 can compare: workgroup geometry, per-thread tile, vec width.
  const receipts: RealizationReceipts = {
    workgroup: [
      config.tileN / config.threadTileN,
      config.tileM / config.threadTileM,
      1,
    ],
    vecLoadForms: [
      { value: aUid, form: vecForm(config.vectorWidth) },
      { value: bUid, form: vecForm(config.vectorWidth) },
    ],
  };

  return { semantic, requests, receipts, region };
}

function vecForm(w: 1 | 2 | 4): "scalar" | "vec2" | "vec4" {
  return w === 4 ? "vec4" : w === 2 ? "vec2" : "scalar";
}

function ceilDivUniform(name: string, tile: number): AffineExpr {
  return {
    kind: "affineCeilDiv",
    num: { kind: "affineLeaf", leaf: { kind: "uniformRef", name } },
    den: { kind: "affineLeaf", leaf: { kind: "intLit", value: tile } },
  };
}

/** Map the epilogue chain into an op-catalog body over the accumulator. */
function epilogueToBody(
  accUid: ValueUid,
  desc: TiledMatmulDescriptor,
  epiInputs: readonly ValueUid[],
): SemanticBodyNode {
  let expr: SemanticBodyNode = { kind: "value", value: accUid };
  if (desc.kSplit || !desc.epilogue) return expr; // raw partials / bare matmul
  for (const op of desc.epilogue.ops) {
    switch (op.kind) {
      case "none":
        break;
      case "bias":
        expr = {
          kind: "apply",
          catalog: { op: "add" },
          args: [expr, { kind: "value", value: epiInputs[op.inputIndex] }],
        };
        break;
      case "unary":
        expr = {
          kind: "apply",
          catalog: { op: op.op ?? "identity" },
          args: [expr],
        };
        break;
      case "binary":
        expr = {
          kind: "apply",
          catalog: { op: op.op ?? "identity" },
          args: [expr, { kind: "value", value: epiInputs[op.inputIndex] }],
        };
        break;
      case "cast":
        expr = {
          kind: "apply",
          catalog: { op: `cast_${op.toDtype}` },
          args: [expr],
        };
        break;
    }
  }
  return expr;
}

/**
 * `applyMatmulSchedule` for the tiled family: reconstruct the `CodegenOptions`
 * from the descriptor carried alongside the state and call the single-source
 * spec builder `createTiledMatmulKernel`. The state's semantic facts are asserted
 * to AGREE with the descriptor at the seam (assertTiledSeam) before lowering; the
 * returned TileKernelSpec is compiled by the differential (its WGSL equals the
 * live `generateTiledMatmulShaderTileIR` byte-for-byte).
 */
export function applyTiledMatmulSchedule(
  state: ScheduleState,
  desc: TiledMatmulDescriptor,
): TileKernelSpec {
  assertTiledSeam(state, desc);
  const options: CodegenOptions = {
    config: desc.config,
    transposeMode: desc.transposeMode,
    dtype: desc.dtype,
    dtypeB: desc.dtypeB,
    epilogue: desc.epilogue,
    batched: desc.batched,
    inputCastA: desc.inputCastA,
    inputCastB: desc.inputCastB,
    kSplit: desc.kSplit,
    swapGrid: desc.swapGrid,
  };
  return createTiledMatmulKernel(options);
}

// ============================================================================
// FAMILY 3b — the SPLIT-K REDUCTION pass (partials → summed output)
// ============================================================================

/**
 * The split-K reduction is a flat parallel elementwise-shaped kernel: one program
 * per output element, summing `kSplitCount` partials and applying alpha. It is
 * NOT a matmul — it is the second half of the K-split `tile`+reduce. Round-trips
 * via `generateKSplitReductionShaderTileIR(count, dtype)`.
 */
export interface KSplitReductionDescriptor {
  readonly kSplitCount: number;
  readonly outputDtype: MatmulDType;
}

export function kSplitReductionWgsl(desc: KSplitReductionDescriptor): string {
  return generateKSplitReductionShaderTileIR(
    desc.kSplitCount,
    desc.outputDtype,
  );
}

// ============================================================================
// FAMILY 3c — GEMV (NT / NN, + epilogue seam, + quantB, + kSplit)
// ============================================================================

/**
 * A structured description of ONE GEMV kernel — the honest reification of
 * `GemvKernelOptions`. Round-trips byte-identically via
 * `generateGemvShaderTileIR`.
 *
 * NOTE (#93 / quant expressibility): the int8-grouped B-operand quant is carried
 * as OPERAND METADATA (`quantB`) on the B NamedValue, NOT as schedule structure —
 * the schedule stays the same computation-shape whether B is dense or quantized.
 * The seam's ROUTING (which realizer decodes the packed weight) is out of schedule
 * scope (a SELECTION fact, R9 registry territory). See the report's expressibility
 * paragraph.
 */
export interface GemvDescriptor {
  readonly mode: "nt" | "nn";
  readonly dtypeA: MatmulDType;
  readonly dtypeB: MatmulDType;
  /**
   * Read-wider-cast-on-load axis (#95 — twice flagged as cast-blind). The A/B
   * operand is physically stored in this (wider) dtype but participates in the
   * dot as the logical dtypeA/dtypeB; mirrors the tiled path's inputCastA/B. It
   * is a REALIZATION fact (the binding's stored dtype), NOT schedule structure —
   * so it rides on the descriptor and the derived A/B NamedValue records the
   * STORED dtype, exactly as the tiled skeleton does. Adding it here closes the
   * GEMV template's cast-blindness so the P1 applicability predicate can route
   * the f16-via-cast decode class (report §5) once GEMV grows a load-cast path.
   */
  readonly inputCastA?: MatmulDType;
  readonly inputCastB?: MatmulDType;
  readonly outputDtype: MatmulDType;
  readonly kSplit: boolean;
  readonly wgSize?: number;
  readonly rowsPerWg?: number;
  readonly vec4?: boolean;
  readonly epilogue?: EpilogueConfig;
  readonly quantB?: QuantB;
}

export function deriveGemvState(
  desc: GemvDescriptor,
  region: SemanticRegionUid,
): ScheduleState {
  const outDtype: MatmulDType =
    desc.mode === "nn" && desc.kSplit ? "f32" : desc.outputDtype;

  const values: NamedValue[] = [];
  const aUid = uid<ValueUid>("in:a");
  const bUid = uid<ValueUid>("in:b");
  const accUid = uid<ValueUid>("acc");
  const outUid = uid<ValueUid>("out:out");

  values.push({
    uid: aUid,
    entity: uid("ent:in:a"),
    allocation: "global",
    // #95 cast axis: when inputCastA is set the operand is STORED wider (f32) and
    // cast on load — the NamedValue records the stored dtype, mirroring the tiled
    // skeleton's `desc.inputCastA ? "f32" : desc.dtype`.
    dtype: toValueDtype(desc.inputCastA ?? desc.dtypeA),
    aliasOf: null,
  });
  // B carries the quant metadata as OPERAND metadata (via its dtype: packed
  // int8-grouped B is a u32 array in the realized kernel; the schedule records
  // the LOGICAL dtype and the quant descriptor is an operand fact, not structure).
  values.push({
    uid: bUid,
    entity: uid("ent:in:b"),
    allocation: "global",
    dtype: toValueDtype(desc.inputCastB ?? desc.dtypeB),
    aliasOf: null,
  });
  values.push({
    uid: accUid,
    entity: uid("ent:acc"),
    allocation: "register",
    dtype: "f32",
    aliasOf: null,
  });
  const epiInputs: ValueUid[] = [];
  if (desc.epilogue) {
    for (let i = 0; i < desc.epilogue.additionalInputCount; i++) {
      const u = uid<ValueUid>(`in:epilogue_in${i}`);
      epiInputs.push(u);
      values.push({
        uid: u,
        entity: uid(`ent:in:epilogue_in${i}`),
        allocation: "global",
        dtype: "f32",
        aliasOf: null,
      });
    }
  }
  values.push({
    uid: outUid,
    entity: uid("ent:out:out"),
    allocation: "global",
    dtype: toValueDtype(outDtype),
    aliasOf: null,
  });

  // M=1: the output loop is over N (row-vector × matrix). NN stages `a` into
  // shared memory (a_tile) + barriers; NT uses a segmented workgroup reduce (no
  // shared A tile, the reduce is the realizer's). Wave 2 stores the NN staging.
  const axN = uid<AxisUid>("axis:n");
  const axK = uid<AxisUid>("axis:k");
  const loopN = uid<LoopUid>("loop:n");
  const loopK = uid<LoopUid>("loop:k");
  const kLoop: SemanticLoop = {
    uid: loopK,
    entity: uid("ent:loop:k"),
    axis: axK,
    kind: "sequential",
    bound: { kind: "affineLeaf", leaf: { kind: "uniformRef", name: "k" } },
    children: [],
  };
  const nLoop: SemanticLoop = {
    uid: loopN,
    entity: uid("ent:loop:n"),
    axis: axN,
    kind: "parallel",
    bound: { kind: "affineLeaf", leaf: { kind: "uniformRef", name: "n" } },
    children: [kLoop],
  };

  const roles: ParticipantSet[] = [];
  const sync: SyncRelation[] = [];
  const values2 = values;
  if (desc.mode === "nn") {
    // NN stages `a` cooperatively into shared memory with per-chunk barriers.
    const aTileUid = uid<ValueUid>("stage:a_tile");
    values2.splice(2, 0, {
      uid: aTileUid,
      entity: uid("ent:stage:a_tile"),
      allocation: "shared",
      dtype: "f32",
      aliasOf: null,
    });
    const coopRole = uid<RoleName>("cooperative-load");
    roles.push({ role: coopRole, condition: { kind: "intLit", value: 1 } });
    sync.push(
      {
        kind: "memoryEffect",
        space: "shared",
        value: aTileUid,
        interval: { fromLoop: loopK, toLoop: loopK },
      },
      {
        kind: "barrier",
        participants: {
          role: coopRole,
          condition: { kind: "intLit", value: 1 },
        },
        spaces: ["shared"],
        convergence: "uniform",
      },
    );
  }

  // The reduction body: acc += a·b (dot). quantB dequant is a realizer detail
  // (unpackInt8Snorm · f16 scale) — the schedule records the dot contract; the
  // dequant is NOT a second owner (it is the operand-metadata realization).
  const bodies: SemanticBody[] = [
    {
      result: accUid,
      expr: {
        kind: "apply",
        catalog: { op: "dot_accum" },
        args: [
          { kind: "value", value: aUid },
          { kind: "value", value: bUid },
        ],
      },
    },
  ];
  const storeResultUid = uid<ValueUid>("write");
  bodies.push({
    result: storeResultUid,
    expr: gemvEpilogueToBody(accUid, desc, epiInputs),
  });

  const noMat = desc.kSplit
    ? []
    : (desc.epilogue?.ops ?? [])
        .filter((o) => o.kind !== "none")
        .map(() => ({
          producer: accUid,
          consumer: storeResultUid,
          acrossLoop: loopN,
        }));

  const store: StoreEdge = {
    source: storeResultUid,
    target: outUid,
    atLoop: loopN,
  };

  // NN kSplit is a `tile` of the K axis (partials → separate reduction pass) —
  // same fp-reorder lemma license as the tiled path.
  const lemmas =
    desc.mode === "nn" && desc.kSplit
      ? [
          {
            lemma: KSPLIT_LEMMA,
            obligation: KSPLIT_OBLIGATION,
            carriedStateRef: "gemv-nn-ksplit",
          },
        ]
      : [];

  const semantic: SemanticSchedule = {
    blockShapes: [[]], // GEMV: flat N output domain, one scalar per program
    loopNest: [nLoop],
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values: values2,
    noMaterialization: noMat,
    stores: [store],
    bodies,
    roles,
    sync,
    atoms: [],
    lemmas,
  };

  return {
    semantic,
    requests: {
      warpBudget: null,
      pipeline: { kind: "none" },
      placementPreferences: [],
      cachePolicy: [],
    },
    receipts:
      desc.mode === "nt" && desc.vec4
        ? {
            vecLoadForms: [
              { value: aUid, form: "vec4" },
              { value: bUid, form: "vec4" },
            ],
          }
        : {},
    region,
  };
}

function gemvEpilogueToBody(
  accUid: ValueUid,
  desc: GemvDescriptor,
  epiInputs: readonly ValueUid[],
): SemanticBodyNode {
  let expr: SemanticBodyNode = { kind: "value", value: accUid };
  if (desc.kSplit || !desc.epilogue) return expr;
  for (const op of desc.epilogue.ops) {
    switch (op.kind) {
      case "none":
        break;
      case "bias":
        expr = {
          kind: "apply",
          catalog: { op: "add" },
          args: [expr, { kind: "value", value: epiInputs[op.inputIndex] }],
        };
        break;
      case "unary":
        expr = {
          kind: "apply",
          catalog: { op: op.op ?? "identity" },
          args: [expr],
        };
        break;
      default:
        // GEMV only supports bias/unary (gemvSupportsEpilogue); binary/cast are
        // refused at the seam — never a second owner masquerading as a body op.
        break;
    }
  }
  return expr;
}

export function applyGemvSchedule(
  state: ScheduleState,
  desc: GemvDescriptor,
): string {
  assertGemvSeam(state, desc);
  const options: GemvKernelOptions = {
    mode: desc.mode,
    dtypeA: desc.dtypeA,
    dtypeB: desc.dtypeB,
    inputCastA: desc.inputCastA,
    inputCastB: desc.inputCastB,
    outputDtype: desc.outputDtype,
    kSplit: desc.kSplit,
    wgSize: desc.wgSize,
    rowsPerWg: desc.rowsPerWg,
    vec4: desc.vec4,
    epilogue: desc.epilogue,
    quantB: desc.quantB,
  };
  return generateGemvShaderTileIR(options);
}

// ============================================================================
// THE LIVE REALIZER ENTRY POINTS (P1 cutover — the schedule object is the SOLE
// WGSL writer for the matmul family). `matmul/dispatch.ts` calls THESE instead
// of the raw `generate*ShaderTileIR` generators; each derives a `ScheduleState`
// and applies it, so the WGSL is what the byte-differential guards on the LIVE
// path. `CodegenOptions` is structurally a `TiledMatmulDescriptor` (same fields).
// ============================================================================

/**
 * The region UID stamped on every LIVE-path matmul ScheduleState. The region is
 * a FOREIGN KEY (R8) — it identifies the semantic region behind the schedule,
 * not its contents — and does NOT enter the WGSL, so one constant is correct for
 * every live matmul kernel (the compilation identity that drives the pipeline
 * cache is `getShaderCacheKey`, unchanged).
 */
const LIVE_MATMUL_REGION = uid<SemanticRegionUid>("region:live-matmul");

/**
 * Realize the tiled-matmul `TileKernelSpec` THROUGH the schedule object: derive
 * a ScheduleState from the options, assert no-second-owner at the seam, and apply
 * it. Byte-identical to the retired `createTiledMatmulKernel(options)` because
 * `applyTiledMatmulSchedule` funnels back to the same builder — the difference is
 * that the schedule object is now the sole writer and the seam assertions run on
 * the live path.
 */
export function realizeTiledMatmulKernel(
  options: CodegenOptions,
): TileKernelSpec {
  const desc: TiledMatmulDescriptor = options;
  const state = deriveTiledMatmulState(desc, LIVE_MATMUL_REGION);
  return applyTiledMatmulSchedule(state, desc);
}

/**
 * Realize the GEMV WGSL THROUGH the schedule object. Same contract as the tiled
 * realizer; consumes `GemvKernelOptions` (the live GEMV plan's option bag) and
 * round-trips through `deriveGemvState`/`applyGemvSchedule`.
 */
export function realizeGemvWgsl(options: GemvKernelOptions): string {
  const desc: GemvDescriptor = {
    mode: options.mode,
    dtypeA: options.dtypeA,
    dtypeB: options.dtypeB,
    inputCastA: options.inputCastA,
    inputCastB: options.inputCastB,
    outputDtype: options.outputDtype,
    kSplit: options.kSplit,
    wgSize: options.wgSize,
    rowsPerWg: options.rowsPerWg,
    vec4: options.vec4,
    epilogue: options.epilogue,
    quantB: options.quantB,
  };
  const state = deriveGemvState(desc, LIVE_MATMUL_REGION);
  return applyGemvSchedule(state, desc);
}

/**
 * Realize the split-K reduction WGSL THROUGH the schedule object. The reduction
 * pass is the second half of the K-split `tile`+reduce (family 3b); it round-
 * trips via `kSplitReductionWgsl`.
 */
export function realizeKSplitReductionWgsl(
  kSplitCount: number,
  outputDtype: MatmulDType,
): string {
  return kSplitReductionWgsl({ kSplitCount, outputDtype });
}

// ============================================================================
// The no-second-owner seam assertions (§12 check 3, family-local)
// ============================================================================

/** The schema-only serialization check (§12 check 1): the printer is the check. */
function assertNoOpaqueLeak(state: ScheduleState): void {
  printScheduleState(state);
}

/**
 * Assert the tiled-matmul `ScheduleState` carries no second owner AND that the
 * epilogue ⊥ kSplit legality rule holds as a TYPED rule on the object (not a
 * scattered conditional). The descriptor is validated against the state at the
 * seam; `validateConfig` is CALLED (single owner of the dependent constraints).
 */
export function assertTiledSeam(
  state: ScheduleState,
  desc: TiledMatmulDescriptor,
): void {
  assertNoOpaqueLeak(state);
  // The R10 dependent constraints are owned by validateConfig — CALL it, do not
  // re-implement (§12 check 3: one owner of the tile-divisibility/smem facts).
  validateConfig(desc.config);

  const s = state.semantic;
  if (s.stores.length !== 1)
    reportNoSecondOwner(
      `no-second-owner[matmul-tiled]: expected exactly one store edge, got ${s.stores.length}.`,
    );

  // THE epilogue ⊥ kSplit TYPED LEGALITY RULE. K-split writes RAW f32 partials;
  // the epilogue must run once on the summed output, never per-split — so a
  // schedule with BOTH an epilogue chain AND the kSplit lemma is ILLEGAL. This
  // is the design's "the epilogue⊥kSplit incompatibility becomes a TYPED legality
  // rule on the object (not a scattered conditional)". The live code enforces it
  // in three scattered places (computeKSplitFactor returns 0 on hasEpilogue,
  // createTiledMatmulKernel gates postAcc on !kSplit, gemv throws); here it is ONE
  // rule read off the typed object.
  const hasEpilogueChain =
    !!desc.epilogue && desc.epilogue.ops.some((o) => o.kind !== "none");
  const hasKSplit = (desc.kSplit ?? 0) >= 2;
  if (hasEpilogueChain && hasKSplit)
    reportNoSecondOwner(
      `legality[matmul]: epilogue ⊥ kSplit — a K-split kernel writes raw f32 partials, ` +
        `so an epilogue chain cannot be applied per-split (it must run once on the summed ` +
        `output). This state carries both an epilogue chain and the kSplit lemma.`,
    );
  // The typed object must AGREE: kSplit present ⇒ lemma present + no no-mat edges.
  const lemmaHasKSplit = s.lemmas.some((l) => l.lemma === KSPLIT_LEMMA);
  if (hasKSplit !== lemmaHasKSplit)
    reportNoSecondOwner(
      `no-second-owner[matmul-tiled]: kSplit descriptor (${hasKSplit}) disagrees with the ` +
        `object's fp-reorder lemma (${lemmaHasKSplit}) — two owners of the K-split fact.`,
    );
  if (hasKSplit && s.noMaterialization.length !== 0)
    reportNoSecondOwner(
      `legality[matmul]: kSplit partials cannot carry no-materialization (fusion) edges.`,
    );

  // swapGrid ⇒ the program-grid map is a `swap`, not identity (R4 reification).
  const wantSwap = !!desc.swapGrid;
  const isSwap = s.programGridMap.kind === "swap";
  if (wantSwap !== isSwap)
    reportNoSecondOwner(
      `no-second-owner[matmul-tiled]: swapGrid descriptor (${wantSwap}) disagrees with the ` +
        `ProgramGridMap (${s.programGridMap.kind}) — swapGrid MUST reify as a program-map value (R4).`,
    );

  // The staging edges + barrier are STORED (S2 shakedown). A tiled matmul MUST
  // carry the cooperative-load role and at least one shared-space barrier — their
  // ABSENCE would mean the derivation mis-classified the staging.
  if (s.roles.length === 0)
    reportNoSecondOwner(
      `no-second-owner[matmul-tiled]: tiled matmul must carry the cooperative-load role (staging is stored, S2).`,
    );
  if (!s.sync.some((y) => y.kind === "barrier"))
    reportNoSecondOwner(
      `no-second-owner[matmul-tiled]: tiled matmul must carry a shared-space barrier (staging is stored, S2).`,
    );
}

/**
 * Assert the GEMV `ScheduleState` carries no second owner and the epilogue ⊥
 * kSplit rule holds; the quant descriptor is OPERAND metadata (not structure).
 */
export function assertGemvSeam(
  state: ScheduleState,
  desc: GemvDescriptor,
): void {
  assertNoOpaqueLeak(state);
  const s = state.semantic;
  if (s.stores.length !== 1)
    reportNoSecondOwner(
      `no-second-owner[gemv]: expected exactly one store edge, got ${s.stores.length}.`,
    );
  // GEMV epilogue must be a supported shape (bias/unary) — the seam refuses the
  // rest (single owner of the gemv-epilogue vocabulary).
  if (desc.epilogue && !gemvSupportsEpilogue(desc.epilogue))
    reportNoSecondOwner(
      `legality[gemv]: unsupported epilogue shape (see gemvSupportsEpilogue).`,
    );
  // epilogue ⊥ kSplit (same reason as tiled — raw partials).
  if (desc.epilogue && desc.epilogue.ops.length > 0 && desc.kSplit)
    reportNoSecondOwner(
      `legality[gemv]: epilogue ⊥ kSplit — partials must stay raw.`,
    );
  // quantB legality: NT mode only, no kSplit (operand-metadata constraint).
  if (desc.quantB) {
    if (desc.mode !== "nt")
      reportNoSecondOwner(`legality[gemv]: quantB requires NT mode.`);
    if (desc.kSplit)
      reportNoSecondOwner(`legality[gemv]: quantB incompatible with kSplit.`);
  }
}
