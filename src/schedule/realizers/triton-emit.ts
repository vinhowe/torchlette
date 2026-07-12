/**
 * emitTriton — the SECOND realizer (schedule-state-design.md §4 v2), tiled-matmul
 * family only. `SemanticSchedule` → deterministic, printable Triton source text.
 *
 * ------------------------------------------------------------------------
 * WHAT IS EMITTED, AND FROM WHERE (the authority horizon, triton-profile.ts)
 * ------------------------------------------------------------------------
 * The emitter writes ONLY what the emitted SOURCE determines (Appendix A authority
 * horizon `sourceDetermines`) plus the meta-parameters that are REQUESTS
 * (`num_warps`, `num_stages`). It emits NOTHING on the `ttgirOwns` side — no shared
 * staging, no lane/warp layout, no vector width, no register placement. That is the
 * RECEIPT BOUNDARY: Triton's compiler owns those, so the S1 receipts are not even
 * ours to write here; they are observed after compilation, never emitted.
 *
 * Mapping (this campaign's deliverable 2), READ FROM the ScheduleState:
 *   semantic.blockShapes[[M,N],[K]]  → BLOCK_M / BLOCK_N / BLOCK_K constexpr
 *   the sequential K loop (loopNest) → `for k in tl.range(0, tl.cdiv(K, BLOCK_K))`
 *   programGridMap identity          → pid_m = pid // grid_n; pid_n = pid % grid_n
 *              swap                   → pid_m/pid_n axes exchanged
 *              grouped(groupSize=G)   → the tutorial's L2-reuse grouped remap (A-R15/R4)
 *   requests.warpBudget              → num_warps=<n>  (omitted → Triton default)
 *   requests.pipeline none           → num_stages omitted
 *              staged(stages=s)       → num_stages=<s>
 *   bodies (epilogue chain)          → the fused epilogue inside the tl program
 * STAGING / roles / receipts are NOT emitted (recorded as the receipt boundary).
 *
 * View/receipt facts that ride on the descriptor in the WGSL path (transpose mode,
 * operand dtypes) ride here too (§11.4 classes them view/receipt, not semantic) —
 * they select pointer strides and `tl` dtypes, exactly as the WGSL lowering reads
 * them off `desc`. `assertTiledSeam` (matmul-skeleton) already proved desc≡state.
 *
 * DETERMINISM: the output is a pure function of (state, desc). No timestamps, no
 * device probing, no map iteration order. It is diffed and cached by
 * digest+realizer-coordinate (canonical.ts artifactIdentity).
 */

import type {
  EpilogueConfig,
  TransposeMode,
} from "../../backend/webgpu/matmul/types";
import type { TiledMatmulDescriptor } from "../matmul-skeleton";
import type { ProgramGridMap, ScheduleState, SemanticLoop } from "../types";

/** A typed refusal — the emitter never throws deep; it returns a reason (F8). */
export class TritonEmitRefusal extends Error {
  constructor(
    readonly element: string,
    reason: string,
  ) {
    super(`triton-emit refused [${element}]: ${reason}`);
    this.name = "TritonEmitRefusal";
  }
}

/** The emitted artifact: source text + the launch metadata the harness needs. */
export interface TritonEmission {
  /** The deterministic, printable Triton kernel source. */
  readonly source: string;
  /** The JIT entry-point name (stable, so the harness can `getattr`). */
  readonly entryPoint: string;
  /** num_warps request emitted (null → Triton default). */
  readonly numWarps: number | null;
  /** num_stages request emitted (null → omitted / `none`). */
  readonly numStages: number | null;
  /** The program-grid map kind emitted (for the report / receipt). */
  readonly gridMap: ProgramGridMap["kind"];
  /** The [BLOCK_M, BLOCK_N, BLOCK_K] constexpr values the launch binds. */
  readonly block: readonly [number, number, number];
  /** The receipt boundary, recorded honestly: what Triton's compiler owns. */
  readonly receiptBoundary: readonly string[];
}

// ============================================================================
// The tiled-matmul emitter
// ============================================================================

/** Find the single sequential (K) loop in the nest — the reduction spine. */
function findKLoop(nest: readonly SemanticLoop[]): SemanticLoop {
  for (const root of nest) {
    let node: SemanticLoop | undefined = root;
    while (node) {
      if (node.kind === "sequential") return node;
      node = node.children[0];
    }
  }
  throw new TritonEmitRefusal(
    "loop-nest",
    "tiled matmul must carry a sequential reduction (K) loop.",
  );
}

/** Map a Triton `tl` dtype name for an operand. */
function tlDtype(d: "f16" | "f32"): string {
  return d === "f16" ? "tl.float16" : "tl.float32";
}

/**
 * Emit the program-id remapping. `identity`/`swap` are a flat linear→(m,n)
 * decode; `grouped` is the tutorial's L2-reuse row-group remap (A-R15 / R4 — the
 * published counterexample the S1 ProgramGridMap coordinate exists FOR).
 */
function emitGridRemap(map: ProgramGridMap): string[] {
  switch (map.kind) {
    case "identity":
      return [
        "    pid = tl.program_id(axis=0)",
        "    grid_n = tl.cdiv(N, BLOCK_N)",
        "    pid_m = pid // grid_n",
        "    pid_n = pid % grid_n",
      ];
    case "swap":
      // Axes exchanged: the linear id decodes n-major (repo's swapGrid).
      return [
        "    pid = tl.program_id(axis=0)",
        "    grid_m = tl.cdiv(M, BLOCK_M)",
        "    pid_n = pid // grid_m",
        "    pid_m = pid % grid_m",
      ];
    case "grouped": {
      // The 03-matrix-multiplication tutorial grouped form (GROUP_M row groups).
      const g = map.groupSize;
      return [
        "    pid = tl.program_id(axis=0)",
        "    grid_m = tl.cdiv(M, BLOCK_M)",
        "    grid_n = tl.cdiv(N, BLOCK_N)",
        `    GROUP_M = ${g}`,
        "    width = GROUP_M * grid_n",
        "    group_id = pid // width",
        "    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)",
        "    pid_m = group_id * GROUP_M + ((pid % width) % group_size)",
        "    pid_n = (pid % width) // group_size",
      ];
    }
    case "checkedAffine":
      throw new TritonEmitRefusal(
        "program-map",
        "checkedAffine grid map not in the P2 tiled-matmul emit corpus.",
      );
  }
}

/** Emit the fused epilogue chain over the accumulator `acc` (register→register). */
function emitEpilogue(epilogue: EpilogueConfig | undefined): string[] {
  if (!epilogue) return [];
  const lines: string[] = [];
  for (const op of epilogue.ops) {
    switch (op.kind) {
      case "none":
        break;
      case "bias":
        // Row-broadcast bias over N (matches the WGSL load1d(offsN) epilogue).
        lines.push(
          `    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)`,
          `    acc = acc + bias[None, :]`,
        );
        break;
      case "unary":
        lines.push(...emitUnary(op.op));
        break;
      case "cast":
        // cast handled at store (out_dtype); a no-op in the register chain.
        break;
      default:
        throw new TritonEmitRefusal(
          "epilogue",
          `unsupported epilogue op kind in tiled emit corpus.`,
        );
    }
  }
  return lines;
}

/** Map a unary epilogue op to a Triton expression on `acc`. */
function emitUnary(op: string): string[] {
  switch (op) {
    case "relu":
      return ["    acc = tl.maximum(acc, 0.0)"];
    case "gelu":
    case "gelu_tanh":
      return [
        "    acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)))",
      ];
    case "sigmoid":
      return ["    acc = tl.sigmoid(acc)"];
    case "tanh":
      return ["    acc = tl.math.tanh(acc)"];
    case "exp":
      return ["    acc = tl.exp(acc)"];
    default:
      throw new TritonEmitRefusal(
        "epilogue-unary",
        `unary '${op}' not in the tiled emit corpus.`,
      );
  }
}

/**
 * Emit Triton source for a TILED matmul ScheduleState.
 *
 * Reads the SEMANTIC facts off `state` (block shapes, the K loop, the program-grid
 * map, the epilogue body, the num_warps/num_stages requests) and the VIEW/RECEIPT
 * facts off `desc` (transpose mode, operand/output dtypes) — the same split the
 * WGSL lowering uses. kSplit is REFUSED (the two-pass partials form is out of the
 * cross-backend differential's tiled emit corpus; a typed refusal, not a throw).
 */
export function emitTritonTiledMatmul(
  state: ScheduleState,
  desc: TiledMatmulDescriptor,
): TritonEmission {
  const s = state.semantic;

  // kSplit partials are refused (raw-partial two-pass is out of the emit corpus).
  if (desc.kSplit && desc.kSplit >= 2) {
    throw new TritonEmitRefusal(
      "kSplit",
      "split-K partials (two-pass) not in the tiled emit corpus; use a single-pass state.",
    );
  }

  // ---- Block shapes: [[BLOCK_M, BLOCK_N], [BLOCK_K]] (read off the semantic). ----
  const [mn, kBlock] = s.blockShapes;
  if (!mn || mn.length !== 2 || !kBlock || kBlock.length !== 1) {
    throw new TritonEmitRefusal(
      "blockShapes",
      `expected [[M,N],[K]] block shapes, got ${JSON.stringify(s.blockShapes)}.`,
    );
  }
  const [BLOCK_M, BLOCK_N] = mn;
  const BLOCK_K = kBlock[0];

  // ---- The K loop is the sequential reduction spine (structural, from state). ----
  findKLoop(s.loopNest); // asserts the reduction spine exists (S2)

  // ---- Program-grid map (identity | swap | grouped) — read off the semantic. ----
  const gridRemap = emitGridRemap(s.programGridMap);

  // ---- View/receipt facts from the descriptor (§11.4): transpose + dtypes. ----
  const transposeMode: TransposeMode = desc.transposeMode;
  const transA = transposeMode === "TN" || transposeMode === "TT";
  const transB = transposeMode === "NT" || transposeMode === "TT";
  const dtA = desc.inputCastA ? "f32" : desc.dtype;
  const dtB = desc.inputCastB ? "f32" : (desc.dtypeB ?? desc.dtype);
  const outDtype: "f16" | "f32" =
    desc.epilogue?.outputDtype ??
    (desc.dtype === "f32" || (desc.dtypeB ?? desc.dtype) === "f32"
      ? "f32"
      : desc.dtype);

  // A strides: [M,K] logical. transA ⇒ A is stored [K,M] (stride_am=1, stride_ak=lda).
  const aStrideM = transA ? "1" : "stride_am";
  const aStrideK = transA ? "stride_am" : "1";
  const bStrideK = transB ? "1" : "stride_bk";
  const bStrideN = transB ? "stride_bk" : "1";

  // ---- Requests: num_warps (warpBudget) + num_stages (pipeline). ----
  const numWarps = state.requests.warpBudget;
  const numStages =
    state.requests.pipeline.kind === "staged"
      ? Math.max(
          ...state.requests.pipeline.entries.map((e) => e.requestedStages),
        )
      : null;

  const hasBias =
    !!desc.epilogue && desc.epilogue.ops.some((o) => o.kind === "bias");
  const epilogueLines = emitEpilogue(desc.epilogue);

  // ---- Assemble deterministic source (fp32 accumulator both sides — the diff). ----
  const sig: string[] = [
    "a_ptr, b_ptr, c_ptr,",
    ...(hasBias ? ["bias_ptr,"] : []),
    "M, N, K,",
    "stride_am, stride_bk, stride_cm,",
    "alpha,",
    "BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,",
  ];

  const lines: string[] = [];
  lines.push("import triton");
  lines.push("import triton.language as tl");
  lines.push("");
  lines.push("");
  lines.push("@triton.jit");
  lines.push(`def matmul_kernel(`);
  for (const part of sig) lines.push(`    ${part}`);
  lines.push("):");
  lines.push(...gridRemap);
  lines.push("");
  lines.push("    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)");
  lines.push("    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)");
  lines.push("    offs_k = tl.arange(0, BLOCK_K)");
  lines.push("");
  // Pointers: A[M,K] (view: transA flips strides), B[K,N] (transB flips).
  lines.push(
    `    a_ptrs = a_ptr + (offs_m[:, None] * ${aStrideM} + offs_k[None, :] * ${aStrideK})`,
  );
  lines.push(
    `    b_ptrs = b_ptr + (offs_k[:, None] * ${bStrideK} + offs_n[None, :] * ${bStrideN})`,
  );
  lines.push("");
  lines.push("    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)");
  lines.push("    num_k = tl.cdiv(K, BLOCK_K)");
  // The K loop maps to tl.range (the semantic sequential loop).
  lines.push("    for kk in tl.range(0, num_k):");
  lines.push("        k_rem = K - kk * BLOCK_K");
  lines.push(
    `        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)`,
  );
  lines.push(
    `        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)`,
  );
  lines.push(
    `        acc += tl.dot(a.to(${tlDtype(dtA)}), b.to(${tlDtype(dtB)}), out_dtype=tl.float32)`,
  );
  lines.push(`        a_ptrs += BLOCK_K * ${aStrideK}`);
  lines.push(`        b_ptrs += BLOCK_K * ${bStrideK}`);
  lines.push("");
  lines.push("    acc = acc * alpha");
  lines.push(...epilogueLines);
  lines.push(`    c = acc.to(${tlDtype(outDtype)})`);
  lines.push(
    "    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :])",
  );
  lines.push("    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)");
  lines.push("    tl.store(c_ptrs, c, mask=c_mask)");
  lines.push("");

  return {
    source: lines.join("\n"),
    entryPoint: "matmul_kernel",
    numWarps,
    numStages,
    gridMap: s.programGridMap.kind,
    block: [BLOCK_M, BLOCK_N, BLOCK_K],
    receiptBoundary: [
      "shared-memory staging of A/B tiles (TTGIR owns)",
      "element-to-lane/warp layout (TTGIR owns)",
      "vector load width (TTGIR owns; A-R6)",
      "register allocation of acc (TTGIR owns)",
      "tensor-core lowering of tl.dot (TTGIR owns)",
      numStages === null
        ? "software-pipeline schedule (no num_stages requested)"
        : "software-pipeline schedule (num_stages requested, TTGIR realizes)",
    ],
  };
}
