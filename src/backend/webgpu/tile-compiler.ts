/**
 * Tile IR Compiler: Phase Decomposition + WGSL Codegen
 *
 * Lowers a tile kernel's IR DAG (built by tile-ir.ts) to WGSL shader code.
 *
 * Two compilation modes:
 *
 * **Auto-phase mode** (for 1D reductions like LayerNorm):
 * 1. Discover all nodes by walking from roots (stores).
 * 2. Assign phases: scalar preamble, reduce phases, inter-reduce scalars, store phase.
 * 3. Emit scalar `let` bindings between phases for reused expressions.
 * 4. Use wgslReduce() for reduce phases, strided loop for store phase.
 *
 * **Imperative mode** (for 2D tiled algorithms like matmul):
 * Detected when `ctx.statements.length > 0`. Walks the statement list and
 * emits WGSL directly — forRange→for loops, barrier→workgroupBarrier(), etc.
 * Expressions are inlined via the same `exprFor()` function.
 *
 * Sequential reductions share a single shared memory array. The generated
 * WGSL is structurally identical to hand-written kernels.
 */

import type {
  IRNode, TileKernelSpec,
  ReduceNode, StoreNode, BlockRangeNode,
  Statement, DataType,
  TileLoadStmt, DotStmt, AccOpStmt, TileLoad1DStmt, TileStoreStmt,
  TilePtr2D, TileMask2D,
} from "./tile-ir";
import { buildKernelIR, type KernelContext } from "./tile-ir";
// Inlined from wgsl-reduce.ts (deleted on main — tile-compiler is the only consumer)

interface ReduceChannel {
  smem: string;
  init: string;
  accumExpr: string;
  result: string;
  transform?: string;
}

interface WgslReduceOpts {
  wgSize: number;
  tid: string;
  dim: string;
  loopVar?: string;
  op: "sum" | "max";
  loopPreamble?: string;
  smem: string;
  init: string;
  accumExpr: string;
  result: string;
  transform?: string;
}

function wgslReduce(opts: WgslReduceOpts): string {
  const { wgSize, tid, dim, op } = opts;
  const loopVar = opts.loopVar ?? "i";
  const half = wgSize / 2;
  const ch: ReduceChannel = { smem: opts.smem, init: opts.init, accumExpr: opts.accumExpr, result: opts.result, transform: opts.transform };

  const lines: string[] = [];
  lines.push(`  {`);
  lines.push(`    var _acc = ${ch.init};`);
  lines.push(`    for (var ${loopVar} = ${tid}; ${loopVar} < ${dim}; ${loopVar} += ${wgSize}u) {`);
  if (opts.loopPreamble) {
    for (const pLine of opts.loopPreamble.split("\n")) {
      lines.push(`      ${pLine.trim()}`);
    }
  }
  if (op === "sum") {
    lines.push(`      _acc += ${ch.accumExpr};`);
  } else {
    lines.push(`      _acc = max(_acc, ${ch.accumExpr});`);
  }
  lines.push(`    }`);
  lines.push(`    ${ch.smem}[${tid}] = _acc;`);
  lines.push(`  }`);
  lines.push(`  workgroupBarrier();`);
  lines.push(``);

  // Tree reduction
  lines.push(`  for (var s = ${half}u; s > 0u; s >>= 1u) {`);
  lines.push(`    if (${tid} < s) {`);
  if (op === "sum") {
    lines.push(`      ${ch.smem}[${tid}] += ${ch.smem}[${tid} + s];`);
  } else {
    lines.push(`      ${ch.smem}[${tid}] = max(${ch.smem}[${tid}], ${ch.smem}[${tid} + s]);`);
  }
  lines.push(`    }`);
  lines.push(`    workgroupBarrier();`);
  lines.push(`  }`);

  const raw = `${ch.smem}[0]`;
  const value = ch.transform ? ch.transform.replace(/_/g, raw) : raw;
  lines.push(`  let ${ch.result} = ${value};`);

  return lines.join("\n");
}

// ============================================================================
// Node Graph Utilities
// ============================================================================

/** Get direct children (inputs/dependencies) of an IR node. */
function getChildren(node: IRNode): IRNode[] {
  switch (node.kind) {
    case "binary": return [node.lhs, node.rhs];
    case "unary": return [node.input];
    case "cast": return [node.input];
    case "bitcast": return [node.input];
    case "reduce": return [node.input];
    case "load": return [node.offsets];
    case "store": return [node.offsets, node.values];
    case "blockRange": return [node.extent];
    case "select": return [node.condition, node.trueVal, node.falseVal];
    case "cmp": return [node.lhs, node.rhs];
    case "sharedRead": return [node.idx];
    case "arrayRead": return [node.idx];
    case "subgroupShuffleXor": return [node.value, node.mask];
    case "programId":
    case "uniform":
    case "const":
    case "threadIdx":
    case "localIndex":
    case "namedRef":
    case "globalId":
      return [];
  }
}

/** Find the first blockRange node reachable from root (not crossing reduces). */
function findBlockRange(node: IRNode): BlockRangeNode | null {
  if (node.kind === "blockRange") return node;
  // Don't descend into reduce nodes — they turn block→scalar,
  // so blockRanges in their inputs don't define the store's loop.
  if (node.kind === "reduce") return null;
  for (const child of getChildren(node)) {
    const found = findBlockRange(child);
    if (found) return found;
  }
  return null;
}

/**
 * Collect all nodes reachable from root, stopping at reduce nodes
 * (treat them as leaves — their inputs belong to earlier phases).
 * Returns nodes in dependency order (leaves first).
 */
function collectSubtreeStopAtReduce(root: IRNode): IRNode[] {
  const result: IRNode[] = [];
  const visited = new Set<number>();
  function walk(n: IRNode) {
    if (visited.has(n.id)) return;
    visited.add(n.id);
    // Don't walk into reduce inputs — they belong to earlier phases
    if (n.kind === "reduce") {
      result.push(n);
      return;
    }
    for (const child of getChildren(n)) {
      walk(child);
    }
    result.push(n);
  }
  walk(root);
  return result;
}

// ============================================================================
// Phase Assignment
// ============================================================================

/**
 * Assign each node a phase number based on its reduce dependencies.
 *
 * - Nodes with no reduce dependency → phase 0 (scalar preamble)
 * - Reduce N's block input subtree → phase (2*N + 1) (reduce loop)
 * - Scalar nodes depending on reduce N but not N+1 → phase (2*N + 2) (post-reduce)
 * - Store phase → last phase
 *
 * Phase numbering: even = scalar, odd = reduce loop.
 * Phase 0 = preamble scalars
 * Phase 1 = reduce 0 loop
 * Phase 2 = post-reduce-0 scalars
 * Phase 3 = reduce 1 loop
 * Phase 4 = post-reduce-1 scalars
 * ...
 * Final even phase = store loop
 */
function assignPhases(
  reduces: ReduceNode[],
  stores: StoreNode[],
): Map<number, number> {
  const phaseOf = new Map<number, number>();

  // Phase assignment for reduce input subtrees: only BLOCK nodes
  // Scalar nodes get their natural phase based on reduce dependencies (below).
  for (let ri = 0; ri < reduces.length; ri++) {
    const reducePhase = 2 * ri + 1;
    const subtree = collectReduceInputTree(reduces[ri]);
    for (const n of subtree) {
      if (n.valueType === "block") {
        phaseOf.set(n.id, reducePhase);
      }
    }
    // The reduce node itself marks the boundary
    phaseOf.set(reduces[ri].id, reducePhase);
  }

  // Assign ALL remaining scalar nodes by finding their latest reduce dependency.
  // Walk from all roots (stores + reduces) to discover every node.
  const allReachable = discoverAllNodes(stores, reduces);
  for (const n of allReachable) {
    if (phaseOf.has(n.id)) continue;
    if (n.kind === "store") continue;
    if (n.valueType !== "scalar") continue;
    const latestReduce = findLatestReduceDep(n, reduces);
    if (latestReduce >= 0) {
      phaseOf.set(n.id, 2 * (latestReduce + 1));
    } else {
      phaseOf.set(n.id, 0);
    }
  }

  return phaseOf;
}

/**
 * Collect nodes in a reduce's input tree, stopping at earlier reduce results.
 */
function collectReduceInputTree(reduce: ReduceNode): IRNode[] {
  const result: IRNode[] = [];
  const visited = new Set<number>();
  function walk(n: IRNode) {
    if (visited.has(n.id)) return;
    visited.add(n.id);
    if (n.kind === "reduce" && n.id !== reduce.id) {
      // Stop at other reduce results
      return;
    }
    for (const child of getChildren(n)) {
      walk(child);
    }
    result.push(n);
  }
  walk(reduce.input);
  return result;
}

/**
 * Find the index of the latest reduce that `node` transitively depends on.
 * Returns -1 if no reduce dependency.
 */
function findLatestReduceDep(node: IRNode, reduces: ReduceNode[]): number {
  const reduceIds = new Map<number, number>(); // reduce node id → index
  for (let i = 0; i < reduces.length; i++) {
    reduceIds.set(reduces[i].id, i);
  }

  let latest = -1;
  const visited = new Set<number>();
  function walk(n: IRNode) {
    if (visited.has(n.id)) return;
    visited.add(n.id);
    const ri = reduceIds.get(n.id);
    if (ri !== undefined) {
      latest = Math.max(latest, ri);
      return; // Don't walk into reduce inputs
    }
    for (const child of getChildren(n)) {
      walk(child);
    }
  }
  walk(node);
  return latest;
}

// ============================================================================
// WGSL Expression Emission
// ============================================================================

type BindingMap = Map<number, string>;

/**
 * Emit a WGSL expression for an IR node.
 *
 * If the node is in `bindings`, returns its variable name.
 * Otherwise builds the expression recursively.
 *
 * @param loopVar - When non-null, blockRange nodes become this variable.
 */
function exprFor(node: IRNode, bindings: BindingMap, loopVar: string | null): string {
  const cached = bindings.get(node.id);
  if (cached !== undefined) return cached;

  switch (node.kind) {
    case "programId": {
      return `wid.${(["x", "y", "z"] as const)[node.dim]}`;
    }
    case "blockRange": {
      if (loopVar === null) throw new Error("blockRange outside loop context");
      return loopVar;
    }
    case "uniform":
      return `config.${node.name}`;
    case "const": {
      if (node.dataType === "f32") {
        const s = String(node.value);
        return s.includes(".") || s.includes("e") || s.includes("E") ? s : s + ".0";
      }
      if (node.dataType === "f16") {
        const s = String(node.value);
        const numStr = s.includes(".") || s.includes("e") || s.includes("E") ? s : s + ".0";
        return `f16(${numStr})`;
      }
      if (node.dataType === "u32") return `${node.value}u`;
      return `i32(${node.value})`;
    }
    case "load": {
      const offs = exprFor(node.offsets, bindings, loopVar);
      return `${node.binding}[${offs}]`;
    }
    case "binary": {
      const lhs = exprFor(node.lhs, bindings, loopVar);
      const rhs = exprFor(node.rhs, bindings, loopVar);
      switch (node.op) {
        case "add": return `(${lhs} + ${rhs})`;
        case "sub": return `(${lhs} - ${rhs})`;
        case "mul": return `(${lhs} * ${rhs})`;
        case "div": return `(${lhs} / ${rhs})`;
        case "mod": return `(${lhs} % ${rhs})`;
        case "and": return `(${lhs} & ${rhs})`;
        case "or": return `(${lhs} | ${rhs})`;
        case "shr": return `(${lhs} >> ${rhs})`;
        case "shl": return `(${lhs} << ${rhs})`;
        case "min": return `min(${lhs}, ${rhs})`;
        case "max": return `max(${lhs}, ${rhs})`;
      }
      break;
    }
    case "unary": {
      const input = exprFor(node.input, bindings, loopVar);
      switch (node.op) {
        case "neg": return `(-${input})`;
        case "rsqrt": return `inverseSqrt(${input})`;
        case "tanh": return `tanh(${input})`;
        case "floor": return `floor(${input})`;
        case "ceil": return `ceil(${input})`;
        case "not": return `!(${input})`;
        default: {
          const fn = ({ exp: "exp", log: "log", abs: "abs", sqrt: "sqrt" } as const)[node.op];
          return `${fn}(${input})`;
        }
      }
    }
    case "cast": {
      const input = exprFor(node.input, bindings, loopVar);
      return `${node.targetType}(${input})`;
    }
    case "bitcast": {
      const input = exprFor(node.input, bindings, loopVar);
      return `bitcast<${node.targetType}>(${input})`;
    }
    case "select": {
      const cond = exprFor(node.condition, bindings, loopVar);
      const t = exprFor(node.trueVal, bindings, loopVar);
      const f = exprFor(node.falseVal, bindings, loopVar);
      return `select(${f}, ${t}, ${cond})`;
    }
    case "cmp": {
      const lhs = exprFor(node.lhs, bindings, loopVar);
      const rhs = exprFor(node.rhs, bindings, loopVar);
      const op = ({ eq: "==", ne: "!=", lt: "<", le: "<=", gt: ">", ge: ">=" } as const)[node.op];
      return `(${lhs} ${op} ${rhs})`;
    }
    // -- Imperative mode nodes --
    case "threadIdx": {
      return `local_id.${(["x", "y", "z"] as const)[node.dim]}`;
    }
    case "localIndex": {
      return "local_idx";
    }
    case "sharedRead": {
      const idx = exprFor(node.idx, bindings, loopVar);
      return `${node.arrayName}[${idx}]`;
    }
    case "namedRef": {
      return node.name;
    }
    case "globalId": {
      return `gid.${(["x", "y", "z"] as const)[node.dim]}`;
    }
    case "subgroupShuffleXor": {
      const val = exprFor(node.value, bindings, loopVar);
      const mask = exprFor(node.mask, bindings, loopVar);
      return `subgroupShuffleXor(${val}, ${mask})`;
    }
    case "arrayRead": {
      const idx = exprFor(node.idx, bindings, loopVar);
      return `${node.arrayName}[${idx}]`;
    }
    case "reduce":
      throw new Error(`Reduce node ${node.id} not pre-bound — compiler bug`);
    case "store":
      throw new Error("Store nodes are not expressions");
    default:
      throw new Error(`Unknown node kind: ${(node as any).kind}`);
  }
}

// ============================================================================
// Variable Naming
// ============================================================================

let _varCounter = 0;

function freshVar(hint: string): string {
  return `_${hint}${_varCounter++}`;
}

// ============================================================================
// Discover All Nodes
// ============================================================================

/**
 * Discover ALL nodes reachable from stores (the roots of the DAG).
 * This includes nodes created by BlockExpr methods that aren't tracked in ctx.nodes.
 */
function discoverAllNodes(stores: StoreNode[], reduces: ReduceNode[]): IRNode[] {
  const allNodes: IRNode[] = [];
  const visited = new Set<number>();

  function walk(n: IRNode) {
    if (visited.has(n.id)) return;
    visited.add(n.id);
    for (const child of getChildren(n)) {
      walk(child);
    }
    allNodes.push(n);
  }

  // Walk from all stores and reduces
  for (const s of stores) walk(s);
  for (const r of reduces) walk(r);

  return allNodes;
}

// ============================================================================
// Main Compiler
// ============================================================================

/**
 * Compile a tile kernel specification into a complete WGSL shader string.
 *
 * Auto-detects mode:
 * - `ctx.statements.length > 0` → imperative mode
 * - Otherwise → auto-phase mode (original)
 */
export function compileTileKernel(spec: TileKernelSpec): string {
  _varCounter = 0;

  // 1. Build the IR DAG
  const ctx = buildKernelIR(spec);

  // 2. Route to appropriate compiler
  if (ctx.statements.length > 0) {
    // Check if statements contain tile-level ops that need lowering
    if (hasTileStatements(ctx.statements)) {
      const lowered = lowerTileStatements(ctx.statements, spec);
      return compileImperativeKernel(spec, ctx, lowered);
    }
    return compileImperativeKernel(spec, ctx);
  }

  return compileAutoPhaseKernel(spec, ctx);
}

// ============================================================================
// Auto-Phase Compiler (original)
// ============================================================================

function compileAutoPhaseKernel(spec: TileKernelSpec, ctx: KernelContext): string {
  // workgroupSize must be a number for auto-phase mode
  const wgSize = typeof spec.workgroupSize === "number" ? spec.workgroupSize : spec.workgroupSize[0] * spec.workgroupSize[1];

  // Find reduces and stores
  const reduces: ReduceNode[] = [];
  for (const n of ctx.nodes) {
    if (n.kind === "reduce") reduces.push(n);
  }

  // Discover all nodes (including those from BlockExpr methods)
  const allNodes = discoverAllNodes(ctx.stores, reduces);

  // Compute reference counts
  const refCounts = new Map<number, number>();
  for (const n of allNodes) {
    refCounts.set(n.id, refCounts.get(n.id) ?? 0);
    for (const child of getChildren(n)) {
      refCounts.set(child.id, (refCounts.get(child.id) ?? 0) + 1);
    }
  }

  // Assign phases
  const phaseOf = assignPhases(reduces, ctx.stores);

  // Binding map: node ID → WGSL variable name
  const bindings: BindingMap = new Map();

  const lines: string[] = [];

  // Feature enables
  if (spec.enableF16) {
    lines.push("enable f16;");
  }
  if (spec.enableSubgroups) {
    lines.push("enable subgroups;");
  }
  if (spec.enableF16 || spec.enableSubgroups) {
    lines.push("");
  }

  // -- Header: struct, bindings, shared memory --
  lines.push(emitUniformStruct(spec));
  lines.push("");
  lines.push(...emitBindings(spec));
  lines.push("");
  lines.push(`var<workgroup> sdata: array<f32, ${wgSize}>;`);
  lines.push("");

  // -- Function signature --
  lines.push(`@compute @workgroup_size(${wgSize})`);
  lines.push(`fn main(@builtin(local_invocation_id) lid: vec3<u32>,`);
  lines.push(`        @builtin(workgroup_id) wid: vec3<u32>) {`);
  lines.push(`  let tid = lid.x;`);

  // -- Phase 0: Scalar preamble --
  emitScalarBindings(allNodes, phaseOf, 0, bindings, refCounts, lines);

  // -- Reduce phases --
  for (let ri = 0; ri < reduces.length; ri++) {
    const postReducePhase = 2 * ri + 2;

    // Emit reduce loop
    lines.push("");
    emitReduceCode(reduces[ri], bindings, spec, wgSize, lines);

    // Post-reduce scalar bindings
    emitScalarBindings(allNodes, phaseOf, postReducePhase, bindings, refCounts, lines);
  }

  // -- Store phase --
  if (ctx.stores.length > 0) {
    emitStoreLoop(ctx.stores, bindings, spec, wgSize, lines);
  }

  lines.push(`}`);
  return lines.join("\n");
}

// ============================================================================
// Imperative Compiler (new)
// ============================================================================

function compileImperativeKernel(spec: TileKernelSpec, ctx: KernelContext, overrideStatements?: Statement[]): string {
  const lines: string[] = [];

  // Feature enables
  if (spec.enableF16) {
    lines.push("enable f16;");
  }
  if (spec.enableSubgroups) {
    lines.push("enable subgroups;");
  }
  if (spec.enableF16 || spec.enableSubgroups) {
    lines.push("");
  }

  // Uniform struct + bindings
  lines.push(emitUniformStruct(spec));
  lines.push("");
  lines.push(...emitBindings(spec));
  lines.push("");

  // Shared memory declarations
  for (const sa of ctx.sharedArrays) {
    const wgslType = sa.elemType;
    lines.push(`var<workgroup> ${sa.name}: array<${wgslType}, ${sa.size}>;`);
  }
  if (ctx.sharedArrays.length > 0) lines.push("");

  // Constants
  if (spec.constants) {
    for (const [name, value] of Object.entries(spec.constants)) {
      // Determine type from value: integers as u32, others as f32
      if (Number.isInteger(value) && value >= 0) {
        lines.push(`const ${name}: u32 = ${value}u;`);
      } else {
        const s = String(value);
        const numStr = s.includes(".") || s.includes("e") || s.includes("E") ? s : s + ".0";
        lines.push(`const ${name}: f32 = ${numStr};`);
      }
    }
    lines.push("");
  }

  // Function signature
  const [wgX, wgY] = typeof spec.workgroupSize === "number"
    ? [spec.workgroupSize, 1]
    : spec.workgroupSize;

  // Scan nodes to determine which builtins are actually used
  const needsGid = ctx.nodes.some(n => n.kind === "globalId");
  const needsWid = ctx.nodes.some(n => n.kind === "programId");
  const needsLocalId = ctx.nodes.some(n => n.kind === "threadIdx");
  const needsLocalIdx = ctx.nodes.some(n => n.kind === "localIndex");
  // Shared arrays and tile-level stmts require local_id/local_idx even if not explicitly referenced
  const hasTileOps = ctx.sharedArrays.length > 0 ||
    ctx.statements.some(s => s.kind === "tileLoad" || s.kind === "dot" || s.kind === "tileStore" || s.kind === "tileLoad1d" || s.kind === "accOp");
  const emitWid = needsWid || hasTileOps;
  const emitLocalId = needsLocalId || hasTileOps;
  const emitLocalIdx = needsLocalIdx || hasTileOps;

  lines.push(`@compute @workgroup_size(${wgX}, ${wgY})`);
  lines.push(`fn main(`);
  const params: string[] = [];
  if (needsGid)    params.push(`  @builtin(global_invocation_id) gid: vec3<u32>`);
  if (emitWid)     params.push(`  @builtin(workgroup_id) wid: vec3<u32>`);
  if (emitLocalId) params.push(`  @builtin(local_invocation_id) local_id: vec3<u32>`);
  if (emitLocalIdx) params.push(`  @builtin(local_invocation_index) local_idx: u32`);
  lines.push(params.join(",\n") + (params.length > 0 ? "," : ""));
  lines.push(`) {`);

  // Emit all statements
  const bindings: BindingMap = new Map();
  const stmts = overrideStatements ?? ctx.statements;

  if (spec.vectorize && spec.vectorize > 1) {
    // Auto-vectorization: unroll body VEC_WIDTH times
    const vecWidth = spec.vectorize;

    // Find all globalId(0) node IDs to override
    const gidXNodes = ctx.nodes.filter(n => n.kind === "globalId" && n.dim === 0);

    lines.push(`  let _base = gid.x * ${vecWidth}u;`);

    for (let v = 0; v < vecWidth; v++) {
      lines.push(`  // vec element ${v}`);
      lines.push(`  {`);
      // Override globalId(0) → (_base + v)
      const vecBindings = new Map(bindings);
      for (const n of gidXNodes) {
        vecBindings.set(n.id, `(_base + ${v}u)`);
      }
      for (const stmt of stmts) {
        emitStatement(stmt, vecBindings, lines, 2);
      }
      lines.push(`  }`);
    }
  } else {
    for (const stmt of stmts) {
      emitStatement(stmt, bindings, lines, 1);
    }
  }

  lines.push(`}`);
  return lines.join("\n");
}

function emitStatement(
  stmt: Statement,
  bindings: BindingMap,
  lines: string[],
  depth: number,
): void {
  const indent = "  ".repeat(depth);
  switch (stmt.kind) {
    case "let": {
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}let ${stmt.name} = ${val};`);
      break;
    }
    case "var": {
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}var ${stmt.name}: ${stmt.dtype} = ${val};`);
      break;
    }
    case "varArray": {
      lines.push(`${indent}var ${stmt.name}: array<${stmt.elemType}, ${stmt.size}>;`);
      if (!stmt.skipZeroInit) {
        // Zero-initialize
        lines.push(`${indent}for (var _zi = 0u; _zi < ${stmt.size}u; _zi = _zi + 1u) {`);
        const zero = stmt.elemType === "f32" ? "0.0" : stmt.elemType === "f16" ? "f16(0.0)" : "0u";
        lines.push(`${indent}  ${stmt.name}[_zi] = ${zero};`);
        lines.push(`${indent}}`);
      }
      break;
    }
    case "assign": {
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.name} = ${val};`);
      break;
    }
    case "addAssign": {
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.name} = ${stmt.name} + ${val};`);
      break;
    }
    case "indexAssign": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.arrayName}[${idx}] = ${val};`);
      break;
    }
    case "indexAddAssign": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.arrayName}[${idx}] = ${stmt.arrayName}[${idx}] + ${val};`);
      break;
    }
    case "forRange": {
      const start = exprFor(stmt.start, bindings, null);
      const bound = exprFor(stmt.bound, bindings, null);
      lines.push(`${indent}for (var ${stmt.varName} = ${start}; ${stmt.varName} < ${bound}; ${stmt.varName} = ${stmt.varName} + 1u) {`);
      for (const s of stmt.body) {
        emitStatement(s, bindings, lines, depth + 1);
      }
      lines.push(`${indent}}`);
      break;
    }
    case "if": {
      const cond = exprFor(stmt.condition, bindings, null);
      lines.push(`${indent}if (${cond}) {`);
      for (const s of stmt.body) {
        emitStatement(s, bindings, lines, depth + 1);
      }
      lines.push(`${indent}}`);
      break;
    }
    case "ifElse": {
      const cond = exprFor(stmt.condition, bindings, null);
      lines.push(`${indent}if (${cond}) {`);
      for (const s of stmt.body) {
        emitStatement(s, bindings, lines, depth + 1);
      }
      lines.push(`${indent}} else {`);
      for (const s of stmt.elseBody) {
        emitStatement(s, bindings, lines, depth + 1);
      }
      lines.push(`${indent}}`);
      break;
    }
    case "barrier": {
      lines.push(`${indent}workgroupBarrier();`);
      break;
    }
    case "sharedWrite": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.arrayName}[${idx}] = ${val};`);
      break;
    }
    case "guardedStore": {
      const cond = exprFor(stmt.condition, bindings, null);
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}if (${cond}) {`);
      lines.push(`${indent}  ${stmt.binding}[${idx}] = ${val};`);
      lines.push(`${indent}}`);
      break;
    }
    case "atomicOp": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      const fnName = {
        max: "atomicMax",
        min: "atomicMin",
        add: "atomicAdd",
        or: "atomicOr",
        and: "atomicAnd",
      }[stmt.op];
      lines.push(`${indent}${fnName}(&${stmt.binding}[${idx}], ${val});`);
      break;
    }
  }
}

// ============================================================================
// Scalar Binding Emission
// ============================================================================

/**
 * Emit `let` bindings for all scalar nodes assigned to `targetPhase`.
 *
 * Only binds "interesting" scalars (binary, unary, cast) — not raw programId,
 * uniform, or const nodes (those are inlined).
 *
 * Additionally binds programId/uniform nodes that have refcount > 1 (used in
 * multiple places, so binding avoids redundant access in loop bodies).
 */
function emitScalarBindings(
  allNodes: IRNode[],
  phaseOf: Map<number, number>,
  targetPhase: number,
  bindings: BindingMap,
  refCounts: Map<number, number>,
  lines: string[],
): void {
  // Collect candidate nodes for this phase
  const candidates: IRNode[] = [];
  for (const n of allNodes) {
    if (n.valueType !== "scalar") continue;
    if (bindings.has(n.id)) continue;
    if (n.kind === "reduce" || n.kind === "store") continue;
    if (phaseOf.get(n.id) !== targetPhase) continue;

    const refs = refCounts.get(n.id) ?? 0;

    // Always bind computed scalars (binary, unary, cast)
    if (n.kind === "binary" || n.kind === "unary" || n.kind === "cast") {
      candidates.push(n);
      continue;
    }

    // Bind programId/uniform if they have multiple references
    if ((n.kind === "programId" || n.kind === "uniform") && refs > 1) {
      candidates.push(n);
      continue;
    }
  }

  // Topological sort to ensure dependencies are bound first
  const sorted = topoSort(candidates);

  for (const node of sorted) {
    const expr = exprFor(node, bindings, null);
    const name = varNameFor(node);
    lines.push(`  let ${name} = ${expr};`);
    bindings.set(node.id, name);
  }
}

function varNameFor(node: IRNode): string {
  switch (node.kind) {
    case "programId": return freshVar("row");
    case "uniform": return freshVar(node.name);
    default: return freshVar("s");
  }
}

function topoSort(nodes: IRNode[]): IRNode[] {
  const ids = new Set(nodes.map(n => n.id));
  const sorted: IRNode[] = [];
  const visited = new Set<number>();

  function visit(node: IRNode) {
    if (visited.has(node.id)) return;
    visited.add(node.id);
    for (const child of getChildren(node)) {
      if (ids.has(child.id)) {
        const childNode = nodes.find(n => n.id === child.id);
        if (childNode) visit(childNode);
      }
    }
    sorted.push(node);
  }

  for (const n of nodes) visit(n);
  return sorted;
}

// ============================================================================
// Reduce Codegen
// ============================================================================

function emitReduceCode(
  reduce: ReduceNode,
  bindings: BindingMap,
  spec: TileKernelSpec,
  wgSize: number,
  lines: string[],
): void {
  const rangeNode = findBlockRange(reduce.input);
  if (!rangeNode) {
    throw new Error("Reduce input has no blockRange in dependency tree");
  }
  const dimExpr = exprFor(rangeNode.extent, bindings, null);

  // Build accumExpr + loopPreamble from the reduce's block input tree.
  // Stop at reduce boundaries — earlier reduces are already bound as scalars.
  const { accumExpr, loopPreamble } = buildAccumExpr(reduce.input, bindings);

  const resultName = freshVar("r");

  const reduceCode = wgslReduce({
    wgSize,
    tid: "tid",
    dim: dimExpr,
    op: reduce.op,
    smem: "sdata",
    init: reduce.op === "sum" ? "0.0" : "-3.402823e+38",
    accumExpr,
    result: resultName,
    loopPreamble: loopPreamble || undefined,
  });

  lines.push(reduceCode);
  bindings.set(reduce.id, resultName);
}

/**
 * Build accumExpr and loopPreamble for a reduce's input expression.
 *
 * Walks the reduce input's block subtree (stopping at other reduce nodes).
 * If the tree is simple (single load), inlines as accumExpr.
 * If complex, emits intermediate block computations as loopPreamble let bindings.
 */
function buildAccumExpr(
  input: IRNode,
  bindings: BindingMap,
): { accumExpr: string; loopPreamble: string | null } {
  // Collect block nodes in the input subtree, NOT crossing reduce boundaries
  const subtree = collectReduceInputSubtree(input);
  const blockNodes = subtree.filter(
    n => n.valueType === "block" && n.kind !== "blockRange",
  );

  if (blockNodes.length <= 1) {
    // Simple: just a load or single expression. Inline as accumExpr.
    return { accumExpr: exprFor(input, bindings, "i"), loopPreamble: null };
  }

  // Complex: emit all but the last as loopPreamble let bindings.
  const loopBindings = new Map(bindings);
  const preambleLines: string[] = [];

  for (const node of blockNodes) {
    if (node.id === input.id) continue; // Final expression → becomes accumExpr
    const expr = exprFor(node, loopBindings, "i");
    const name = freshVar("v");
    preambleLines.push(`let ${name} = ${expr};`);
    loopBindings.set(node.id, name);
  }

  const accumExpr = exprFor(input, loopBindings, "i");

  return {
    accumExpr,
    loopPreamble: preambleLines.length > 0 ? preambleLines.join("\n") : null,
  };
}

/**
 * Collect nodes in a reduce's input subtree, stopping at other reduce nodes.
 * Returns in dependency order (leaves first).
 */
function collectReduceInputSubtree(root: IRNode): IRNode[] {
  const result: IRNode[] = [];
  const visited = new Set<number>();
  function walk(n: IRNode) {
    if (visited.has(n.id)) return;
    visited.add(n.id);
    if (n.kind === "reduce") {
      // Stop: this reduce's result is already bound as a scalar
      result.push(n);
      return;
    }
    for (const child of getChildren(n)) {
      walk(child);
    }
    result.push(n);
  }
  walk(root);
  return result;
}

// ============================================================================
// Store Loop Codegen
// ============================================================================

function emitStoreLoop(
  stores: StoreNode[],
  bindings: BindingMap,
  spec: TileKernelSpec,
  wgSize: number,
  lines: string[],
): void {
  // Partition stores: scalar (no blockRange) vs block (has blockRange)
  const scalarStores: StoreNode[] = [];
  const blockGroups = new Map<number, StoreNode[]>(); // grouped by blockRange extent node ID

  for (const store of stores) {
    const rangeNode = findBlockRange(store);
    if (!rangeNode) {
      scalarStores.push(store);
    } else {
      const key = rangeNode.extent.id;
      let group = blockGroups.get(key);
      if (!group) {
        group = [];
        blockGroups.set(key, group);
      }
      group.push(store);
    }
  }

  // Emit scalar stores: thread 0 only
  if (scalarStores.length > 0) {
    lines.push("");
    lines.push(`  if (tid == 0u) {`);
    for (const store of scalarStores) {
      const offs = exprFor(store.offsets, bindings, null);
      const vals = exprFor(store.values, bindings, null);
      lines.push(`    ${store.binding}[${offs}] = ${vals};`);
    }
    lines.push(`  }`);
  }

  // Emit one strided loop per block group
  for (const [, groupStores] of blockGroups) {
    const rangeNode = findBlockRange(groupStores[0])!;
    const dimExpr = exprFor(rangeNode.extent, bindings, null);

    lines.push("");
    lines.push(`  for (var i = tid; i < ${dimExpr}; i += ${wgSize}u) {`);

    const loopBindings = new Map(bindings);

    for (const store of groupStores) {
      // Walk store value tree, stopping at reduces
      const subtree = collectSubtreeStopAtReduce(store.values);
      const blockNodes = subtree.filter(
        n => n.valueType === "block"
          && n.kind !== "blockRange"
          && n.kind !== "load"
          && !loopBindings.has(n.id),
      );

      for (const node of blockNodes) {
        if (loopBindings.has(node.id)) continue;
        const expr = exprFor(node, loopBindings, "i");
        const name = freshVar("v");
        lines.push(`    let ${name} = ${expr};`);
        loopBindings.set(node.id, name);
      }

      const offs = exprFor(store.offsets, loopBindings, "i");
      const vals = exprFor(store.values, loopBindings, "i");
      lines.push(`    ${store.binding}[${offs}] = ${vals};`);
    }

    lines.push(`  }`);
  }
}

// ============================================================================
// Binding / Uniform Codegen
// ============================================================================

function emitUniformStruct(spec: TileKernelSpec): string {
  const entries = Object.entries(spec.uniforms);
  const lines: string[] = [];
  lines.push(`struct TileConfig {`);

  for (const [name, type] of entries) {
    lines.push(`  ${name}: ${type},`);
  }

  lines.push(`};`);
  return lines.join("\n");
}

function emitBindings(spec: TileKernelSpec): string[] {
  const lines: string[] = [];
  let bindingIndex = 0;
  const uniformIdx = spec.uniformBindingIndex;
  const entries = Object.entries(spec.bindings);

  for (let i = 0; i < entries.length; i++) {
    // Insert uniform at the specified index if requested
    if (uniformIdx !== undefined && bindingIndex === uniformIdx) {
      lines.push(`@group(0) @binding(${bindingIndex}) var<uniform> config: TileConfig;`);
      bindingIndex++;
    }
    const [name, binding] = entries[i];
    if (binding.storage === "atomic") {
      // Atomic bindings use array<atomic<T>>
      lines.push(`@group(0) @binding(${bindingIndex}) var<storage, read_write> ${name}: array<atomic<${binding.type}>>;`);
    } else {
      const access = binding.storage === "read" ? "read" : "read_write";
      lines.push(`@group(0) @binding(${bindingIndex}) var<storage, ${access}> ${name}: array<${binding.type}>;`);
    }
    bindingIndex++;
  }

  // If uniform wasn't inserted yet (no uniformBindingIndex or it's after all bindings)
  if (uniformIdx === undefined || bindingIndex <= uniformIdx) {
    lines.push(`@group(0) @binding(${bindingIndex}) var<uniform> config: TileConfig;`);
  }

  return lines;
}

// ============================================================================
// Tile Statement Lowering Pass
// ============================================================================

/** Check if any statements (recursively) contain tile-level ops. */
function hasTileStatements(stmts: Statement[]): boolean {
  for (const s of stmts) {
    switch (s.kind) {
      case "tileLoad":
      case "dot":
      case "accOp":
      case "tileLoad1d":
      case "tileStore":
        return true;
      case "forRange":
        if (hasTileStatements(s.body)) return true;
        break;
      case "if":
        if (hasTileStatements(s.body)) return true;
        break;
      case "ifElse":
        if (hasTileStatements(s.body) || hasTileStatements(s.elseBody)) return true;
        break;
    }
  }
  return false;
}

/**
 * Lower tile-level statements to imperative statements.
 * Tile ops (tileLoad, dot, accOp, tileStore) are expanded into
 * cooperative loading loops, barrier + outer product, per-element loops, etc.
 *
 * Non-tile statements are passed through unchanged (with recursive lowering
 * of any nested bodies).
 */
function lowerTileStatements(stmts: Statement[], spec: TileKernelSpec): Statement[] {
  const result: Statement[] = [];
  for (const s of stmts) {
    switch (s.kind) {
      case "tileLoad":
        result.push(...lowerTileLoad(s, spec));
        break;
      case "dot":
        result.push(...lowerDot(s));
        break;
      case "accOp":
        result.push(...lowerAccOp(s));
        break;
      case "tileLoad1d":
        result.push(...lowerTileLoad1D(s));
        break;
      case "tileStore":
        result.push(...lowerTileStore(s, spec));
        break;
      case "forRange":
        result.push({
          ...s,
          body: lowerTileStatements(s.body, spec),
        });
        break;
      case "if":
        result.push({
          ...s,
          body: lowerTileStatements(s.body, spec),
        });
        break;
      case "ifElse":
        result.push({
          ...s,
          body: lowerTileStatements(s.body, spec),
          elseBody: lowerTileStatements(s.elseBody, spec),
        });
        break;
      default:
        result.push(s);
        break;
    }
  }
  return result;
}

/** Helper: create an IRNode inline for constants/ops used during lowering. */
function cU32(value: number): IRNode {
  return { id: -1, kind: "const", valueType: "scalar", dataType: "u32", value };
}
function cF32(value: number): IRNode {
  return { id: -1, kind: "const", valueType: "scalar", dataType: "f32", value };
}
function ref(name: string, dt: DataType = "u32"): IRNode {
  return { id: -1, kind: "namedRef", valueType: "scalar", dataType: dt, name };
}
function binOp(op: "add" | "sub" | "mul" | "div" | "mod", lhs: IRNode, rhs: IRNode, dt: DataType = "u32"): IRNode {
  return { id: -1, kind: "binary", op, lhs, rhs, valueType: "scalar", dataType: dt };
}
function cmpOp(op: "lt" | "le" | "gt" | "ge", lhs: IRNode, rhs: IRNode): IRNode {
  return { id: -1, kind: "cmp", op, lhs, rhs, valueType: "scalar", dataType: "u32" };
}
function andOp(lhs: IRNode, rhs: IRNode): IRNode {
  return { id: -1, kind: "binary", op: "and", lhs, rhs, valueType: "scalar", dataType: "u32" };
}
function castNode(input: IRNode, targetType: DataType): IRNode {
  return { id: -1, kind: "cast", input, targetType, valueType: input.valueType, dataType: targetType };
}
function sharedRead(arrayName: string, idx: IRNode, dt: DataType = "f32"): IRNode {
  return { id: -1, kind: "sharedRead", arrayName, idx, valueType: "scalar", dataType: dt };
}
function arrayRead(arrayName: string, idx: IRNode, dt: DataType = "f32"): IRNode {
  return { id: -1, kind: "arrayRead", arrayName, idx, valueType: "scalar", dataType: dt };
}
function loadBinding(binding: string, idx: IRNode, dt: DataType = "f32"): IRNode {
  return { id: -1, kind: "load", binding, offsets: idx, valueType: "block", dataType: dt };
}

function getTotalThreads(spec: TileKernelSpec): number {
  if (typeof spec.workgroupSize === "number") return spec.workgroupSize;
  return spec.workgroupSize[0] * spec.workgroupSize[1];
}

/**
 * Lower tileLoad → cooperative loading loop.
 *
 * Each thread loads `elemsPerThread` elements from global memory into shared.
 * Bounds-checked via the mask.
 */
function lowerTileLoad(stmt: TileLoadStmt, spec: TileKernelSpec): Statement[] {
  const { binding, ptr, mask, sharedName, tileRows, tileCols, elemType } = stmt;
  const totalElems = tileRows * tileCols;
  const totalThreads = getTotalThreads(spec);
  const elemsPerThread = Math.ceil(totalElems / totalThreads);

  const result: Statement[] = [];
  const iVar = `_ld_i`;
  const localIdx = ref("local_idx");

  // for (var _ld_i = 0u; _ld_i < elemsPerThread; _ld_i++) {
  const loopBody: Statement[] = [];

  // flat = local_idx * elemsPerThread + _ld_i
  const flatName = freshVar("flat");
  loopBody.push({
    kind: "let", name: flatName, dtype: "u32",
    value: binOp("add", binOp("mul", localIdx, cU32(elemsPerThread)), ref(iVar)),
  });

  // if (flat < totalElems) {
  const ifBody: Statement[] = [];
  const rowName = freshVar("row");
  const colName = freshVar("col");
  ifBody.push({
    kind: "let", name: rowName, dtype: "u32",
    value: binOp("div", ref(flatName), cU32(tileCols)),
  });
  ifBody.push({
    kind: "let", name: colName, dtype: "u32",
    value: binOp("mod", ref(flatName), cU32(tileCols)),
  });

  // globalIdx = base + (outerRange.base + row) * outerStride + (innerRange.base + col) * innerStride
  const globalRow = binOp("add", ptr.outerRange.base, ref(rowName));
  const globalCol = binOp("add", ptr.innerRange.base, ref(colName));
  const gIdxName = freshVar("gIdx");
  ifBody.push({
    kind: "let", name: gIdxName, dtype: "u32",
    value: binOp("add",
      binOp("add", ptr.baseOffset, binOp("mul", globalRow, ptr.outerStride)),
      binOp("mul", globalCol, ptr.innerStride),
    ),
  });

  // Mask check: (outerRange.base + row) < outerBound && (innerRange.base + col) < innerBound
  const maskCond = andOp(
    cmpOp("lt", globalRow, mask.outerBound),
    cmpOp("lt", globalCol, mask.innerBound),
  );

  // if (mask) { shared[flat] = f32(binding[gIdx]) } else { shared[flat] = 0.0 }
  // Use the actual binding dtype (e.g., f16) for the load, then cast to f32 for shared memory
  const bindingDtype = spec.bindings[binding]?.type ?? elemType;
  const loadExpr = loadBinding(binding, ref(gIdxName), bindingDtype);
  const loadAsF32 = bindingDtype === "f32" ? loadExpr : castNode(loadExpr, "f32");
  ifBody.push({
    kind: "ifElse",
    condition: maskCond,
    body: [{
      kind: "sharedWrite",
      arrayName: sharedName,
      idx: ref(flatName),
      value: loadAsF32,
    }],
    elseBody: [{
      kind: "sharedWrite",
      arrayName: sharedName,
      idx: ref(flatName),
      value: cF32(0),
    }],
  });

  loopBody.push({
    kind: "if",
    condition: cmpOp("lt", ref(flatName), cU32(totalElems)),
    body: ifBody,
  });

  result.push({
    kind: "forRange",
    varName: iVar,
    start: cU32(0),
    bound: cU32(elemsPerThread),
    body: loopBody,
  });

  return result;
}

/**
 * Lower dot → barrier + outer product loop.
 *
 * barrier()
 * for kk in 0..innerDim:
 *   load a_vals[threadTileM] from shared A
 *   load b_vals[threadTileN] from shared B
 *   outer product: acc[tm*N+tn] += a_vals[tm] * b_vals[tn]
 * barrier()
 */
function lowerDot(stmt: DotStmt): Statement[] {
  const { aTile, bTile, accName, threadTileM, threadTileN } = stmt;
  const innerDim = aTile.innerDim;
  const result: Statement[] = [];

  // barrier() before reading shared memory
  result.push({ kind: "barrier" });

  const kkVar = freshVar("kk");
  const kkLoop: Statement[] = [];

  // Load a_vals from shared A
  const aValsName = freshVar("a_vals");
  kkLoop.push({
    kind: "varArray", name: aValsName, elemType: "f32",
    size: threadTileM, skipZeroInit: true,
  });
  const tmVar1 = freshVar("tm");
  kkLoop.push({
    kind: "forRange", varName: tmVar1, start: cU32(0), bound: cU32(threadTileM),
    body: [{
      kind: "indexAssign",
      arrayName: aValsName,
      idx: ref(tmVar1),
      // tileA[(thread_row * threadTileM + tm) * innerDim + kk]
      value: sharedRead(aTile.sharedName,
        binOp("add",
          binOp("mul",
            binOp("add", binOp("mul", ref("thread_row"), cU32(threadTileM)), ref(tmVar1)),
            cU32(innerDim),
          ),
          ref(kkVar),
        ),
      ),
    }],
  });

  // Load b_vals from shared B
  const bValsName = freshVar("b_vals");
  kkLoop.push({
    kind: "varArray", name: bValsName, elemType: "f32",
    size: threadTileN, skipZeroInit: true,
  });
  const tnVar1 = freshVar("tn");
  kkLoop.push({
    kind: "forRange", varName: tnVar1, start: cU32(0), bound: cU32(threadTileN),
    body: [{
      kind: "indexAssign",
      arrayName: bValsName,
      idx: ref(tnVar1),
      // tileB[kk * bCols + thread_col * threadTileN + tn]
      value: sharedRead(bTile.sharedName,
        binOp("add",
          binOp("mul", ref(kkVar), cU32(bTile.cols)),
          binOp("add", binOp("mul", ref("thread_col"), cU32(threadTileN)), ref(tnVar1)),
        ),
      ),
    }],
  });

  // Outer product: acc[tm*N+tn] += a_vals[tm] * b_vals[tn]
  const tmVar2 = freshVar("tm");
  const tnVar2 = freshVar("tn");
  kkLoop.push({
    kind: "forRange", varName: tmVar2, start: cU32(0), bound: cU32(threadTileM),
    body: [{
      kind: "forRange", varName: tnVar2, start: cU32(0), bound: cU32(threadTileN),
      body: [{
        kind: "indexAddAssign",
        arrayName: accName,
        idx: binOp("add", binOp("mul", ref(tmVar2), cU32(threadTileN)), ref(tnVar2)),
        value: binOp("mul",
          arrayRead(aValsName, ref(tmVar2)),
          arrayRead(bValsName, ref(tnVar2)),
          "f32",
        ),
      }],
    }],
  });

  result.push({
    kind: "forRange",
    varName: kkVar,
    start: cU32(0),
    bound: cU32(innerDim),
    body: kkLoop,
  });

  // barrier() after shared memory use
  result.push({ kind: "barrier" });

  return result;
}

/**
 * Lower accOp → per-element loops on thread tile.
 */
function lowerAccOp(stmt: AccOpStmt): Statement[] {
  const { accName, threadTileM, threadTileN, op } = stmt;
  const result: Statement[] = [];

  switch (op.kind) {
    case "mulScalar": {
      // acc[tm*N+tn] *= value
      const tmVar = freshVar("tm");
      const tnVar = freshVar("tn");
      const accIdx = binOp("add", binOp("mul", ref(tmVar), cU32(threadTileN)), ref(tnVar));
      result.push({
        kind: "forRange", varName: tmVar, start: cU32(0), bound: cU32(threadTileM),
        body: [{
          kind: "forRange", varName: tnVar, start: cU32(0), bound: cU32(threadTileN),
          body: [{
            kind: "indexAssign",
            arrayName: accName,
            idx: accIdx,
            value: binOp("mul", arrayRead(accName, accIdx), op.value, "f32"),
          }],
        }],
      });
      break;
    }
    case "addRow": {
      // acc[tm*N+tn] += vals[tn]
      const tmVar = freshVar("tm");
      const tnVar = freshVar("tn");
      result.push({
        kind: "forRange", varName: tmVar, start: cU32(0), bound: cU32(threadTileM),
        body: [{
          kind: "forRange", varName: tnVar, start: cU32(0), bound: cU32(threadTileN),
          body: [{
            kind: "indexAddAssign",
            arrayName: accName,
            idx: binOp("add", binOp("mul", ref(tmVar), cU32(threadTileN)), ref(tnVar)),
            value: arrayRead(op.valuesArray, ref(tnVar)),
          }],
        }],
      });
      break;
    }
    case "apply": {
      // For each element: acc[i] = fn(acc[i])
      // The body captured by apply_ contains statements that compute the result.
      // We substitute the placeholder variable with acc[idx] reads.
      const tmVar = freshVar("tm");
      const tnVar = freshVar("tn");
      const accIdx = binOp("add", binOp("mul", ref(tmVar), cU32(threadTileN)), ref(tnVar));

      // Build inner body:
      // let _acc_val = acc[idx];
      // ... captured body statements ...
      // acc[idx] = resultNode;
      const innerBody: Statement[] = [];
      innerBody.push({
        kind: "let", name: op.valName, dtype: "f32",
        value: arrayRead(accName, accIdx),
      });
      innerBody.push(...op.body);
      innerBody.push({
        kind: "indexAssign",
        arrayName: accName,
        idx: accIdx,
        value: op.resultNode,
      });

      result.push({
        kind: "forRange", varName: tmVar, start: cU32(0), bound: cU32(threadTileM),
        body: [{
          kind: "forRange", varName: tnVar, start: cU32(0), bound: cU32(threadTileN),
          body: innerBody,
        }],
      });
      break;
    }
    case "castTo": {
      // No-op at the statement level — the cast is tracked on the Accumulator
      // and applied during tileStore lowering.
      break;
    }
  }
  return result;
}

/**
 * Lower tileLoad1d → per-thread register load.
 *
 * var array[threadTileN]
 * for tn in 0..threadTileN:
 *   array[tn] = binding[range.base + thread_col * threadTileN + tn]
 */
function lowerTileLoad1D(stmt: TileLoad1DStmt): Statement[] {
  const { binding, range, arrayName, size } = stmt;
  const result: Statement[] = [];

  result.push({
    kind: "varArray", name: arrayName, elemType: "f32",
    size, skipZeroInit: true,
  });

  const tnVar = freshVar("tn");
  result.push({
    kind: "forRange", varName: tnVar, start: cU32(0), bound: cU32(size),
    body: [{
      kind: "indexAssign",
      arrayName,
      idx: ref(tnVar),
      value: loadBinding(binding,
        binOp("add", range.base,
          binOp("add", binOp("mul", ref("thread_col"), cU32(size)), ref(tnVar)),
        ),
      ),
    }],
  });

  return result;
}

/**
 * Lower tileStore → bounds-checked output.
 *
 * for tm in 0..threadTileM:
 *   for tn in 0..threadTileN:
 *     let row = outer.base + thread_row * ttM + tm
 *     let col = inner.base + thread_col * ttN + tn
 *     if (row < outerBound && col < innerBound) {
 *       let idx = base + row * outerStride + col * innerStride
 *       out[idx] = maybecast(acc[tm*ttN + tn])
 *     }
 */
function lowerTileStore(stmt: TileStoreStmt, spec: TileKernelSpec): Statement[] {
  const { binding, ptr, mask, accName, threadTileM, threadTileN, accDtype } = stmt;
  const result: Statement[] = [];

  const tmVar = freshVar("tm");
  const tnVar = freshVar("tn");

  const innerBody: Statement[] = [];

  const rowName = freshVar("st_row");
  const colName = freshVar("st_col");
  innerBody.push({
    kind: "let", name: rowName, dtype: "u32",
    value: binOp("add", ptr.outerRange.base,
      binOp("add", binOp("mul", ref("thread_row"), cU32(threadTileM)), ref(tmVar)),
    ),
  });
  innerBody.push({
    kind: "let", name: colName, dtype: "u32",
    value: binOp("add", ptr.innerRange.base,
      binOp("add", binOp("mul", ref("thread_col"), cU32(threadTileN)), ref(tnVar)),
    ),
  });

  const maskCond = andOp(
    cmpOp("lt", ref(rowName), mask.outerBound),
    cmpOp("lt", ref(colName), mask.innerBound),
  );

  const idxName = freshVar("st_idx");
  const ifBody: Statement[] = [];
  ifBody.push({
    kind: "let", name: idxName, dtype: "u32",
    value: binOp("add",
      binOp("add", ptr.baseOffset, binOp("mul", ref(rowName), ptr.outerStride)),
      binOp("mul", ref(colName), ptr.innerStride),
    ),
  });

  // acc value, possibly cast
  const accIdx = binOp("add", binOp("mul", ref(tmVar), cU32(threadTileN)), ref(tnVar));
  let accVal: IRNode = arrayRead(accName, accIdx);
  if (accDtype && accDtype !== "f32") {
    accVal = castNode(accVal, accDtype);
  }

  ifBody.push({
    kind: "indexAssign",
    arrayName: binding,
    idx: ref(idxName),
    value: accVal,
  });

  innerBody.push({
    kind: "if",
    condition: maskCond,
    body: ifBody,
  });

  result.push({
    kind: "forRange", varName: tmVar, start: cU32(0), bound: cU32(threadTileM),
    body: [{
      kind: "forRange", varName: tnVar, start: cU32(0), bound: cU32(threadTileN),
      body: innerBody,
    }],
  });

  return result;
}

// ============================================================================
// Exports
// ============================================================================
export { getChildren };
