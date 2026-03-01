/**
 * Tile IR Compiler: Imperative WGSL Codegen
 *
 * Lowers a tile kernel's IR (built by tile-ir.ts) to WGSL shader code.
 *
 * Single compilation model: imperative statement stream. The developer writes
 * explicit statements (forRange, barrier, sharedArray, emitStore, etc.) for
 * full control over compute kernel algorithms.
 *
 * Walks the statement list and emits WGSL directly — forRange→for loops,
 * barrier→workgroupBarrier(), etc. Expressions are inlined via `exprFor()`.
 *
 * Also handles Block-ops tile statements (tileLoad, tileStore, dot, etc.)
 * which are lowered to scalar/vec4 WGSL before imperative emission.
 */

import type {
  IRNode, TileKernelSpec,
  Statement, DataType,
  TileLoadStmt, TileLoad1DStmt, TileStoreStmt,
  TilePtr2D, TileMask2D,
  BlockAllocStmt, BlockLoadStmt, BlockStoreStmt, BlockDotStmt,
  BlockReduceStmt, BlockUnaryStmt, BlockBinaryStmt,
  ThreadIdxNode,
} from "./tile-ir";
import { buildKernelIR, type KernelContext, elementwiseGrid } from "./tile-ir";
import { computeSafeVecWidth } from "./tile-access-analysis";
import { getSubgroupSupport } from "./matmul/types";
import { F32_NEG_MAX } from "./shape-utils";

/** Well-known uniform names that represent element counts for elementwise kernels. */
const ELEMENTWISE_UNIFORMS = new Set(["size", "total_elements", "num_elements", "outSize"]);

/** Find a u32 uniform that represents the element count, if any. */
function findElementwiseUniform(spec: TileKernelSpec): string | null {
  for (const [name, type] of Object.entries(spec.uniforms)) {
    if (type === "u32" && ELEMENTWISE_UNIFORMS.has(name)) return name;
  }
  return null;
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
 * @param loopVar - Unused (kept for signature stability). Always null.
 */
function exprFor(node: IRNode, bindings: BindingMap, loopVar: string | null): string {
  const cached = bindings.get(node.id);
  if (cached !== undefined) return cached;

  switch (node.kind) {
    case "programId": {
      return `wid.${(["x", "y", "z"] as const)[node.dim]}`;
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
        case "xor": return `(${lhs} ^ ${rhs})`;
        case "shr": return `(${lhs} >> ${rhs})`;
        case "shl": return `(${lhs} << ${rhs})`;
        case "min": return `min(${lhs}, ${rhs})`;
        case "max": return `max(${lhs}, ${rhs})`;
        case "pow": return `pow(${lhs}, ${rhs})`;
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
          const fn = ({ exp: "exp", log: "log", abs: "abs", sqrt: "sqrt", sin: "sin", cos: "cos", round: "round", sign: "sign", exp2: "exp2", log2: "log2" } as const)[node.op];
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
    case "numWorkgroups": {
      return `num_wg.${(["x", "y", "z"] as const)[node.dim]}`;
    }
    case "subgroupShuffleXor": {
      const val = exprFor(node.value, bindings, loopVar);
      const mask = exprFor(node.mask, bindings, loopVar);
      return `subgroupShuffleXor(${val}, ${mask})`;
    }
    case "subgroupAdd": {
      return `subgroupAdd(${exprFor(node.value, bindings, loopVar)})`;
    }
    case "subgroupMax": {
      return `subgroupMax(${exprFor(node.value, bindings, loopVar)})`;
    }
    case "subgroupMin": {
      return `subgroupMin(${exprFor(node.value, bindings, loopVar)})`;
    }
    case "subgroupBroadcastFirst": {
      return `subgroupBroadcastFirst(${exprFor(node.value, bindings, loopVar)})`;
    }
    case "subgroupInclusiveAdd": {
      return `subgroupInclusiveAdd(${exprFor(node.value, bindings, loopVar)})`;
    }
    case "vec4dot": {
      const a = node.a.map(n => exprFor(n, bindings, loopVar));
      const b = node.b.map(n => exprFor(n, bindings, loopVar));
      return `dot(vec4<f32>(${a.join(", ")}), vec4<f32>(${b.join(", ")}))`;
    }
    case "arrayRead": {
      const idx = exprFor(node.idx, bindings, loopVar);
      return `${node.arrayName}[${idx}]`;
    }
    // -- Vec4 native nodes --
    case "vec4Construct": {
      const x = exprFor(node.x, bindings, loopVar);
      const y = exprFor(node.y, bindings, loopVar);
      const z = exprFor(node.z, bindings, loopVar);
      const w = exprFor(node.w, bindings, loopVar);
      return `vec4<f32>(${x}, ${y}, ${z}, ${w})`;
    }
    case "vec4Splat": {
      const v = exprFor(node.value, bindings, loopVar);
      return `vec4<f32>(${v})`;
    }
    case "vec4NativeDot": {
      const a = exprFor(node.a, bindings, loopVar);
      const b = exprFor(node.b, bindings, loopVar);
      return `dot(${a}, ${b})`;
    }
    case "vec4Component": {
      const v = exprFor(node.value, bindings, loopVar);
      const comp = ["x", "y", "z", "w"][node.comp];
      return `${v}.${comp}`;
    }
    case "vec4Binary": {
      const a = exprFor(node.a, bindings, loopVar);
      const b = exprFor(node.b, bindings, loopVar);
      const op = node.op === "add" ? "+" : node.op === "sub" ? "-" : "*";
      return `(${a} ${op} ${b})`;
    }
    case "vec4ArrayRead": {
      const idx = exprFor(node.idx, bindings, loopVar);
      return `${node.arrayName}[${idx}]`;
    }
    case "vec4SharedRead": {
      const idx = exprFor(node.idx, bindings, loopVar);
      return `${node.arrayName}[${idx}]`;
    }
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
// Main Compiler
// ============================================================================

/**
 * Compile a tile kernel specification into a complete WGSL shader string.
 *
 * All kernels use imperative mode: the kernel function emits explicit statements
 * (forRange, barrier, sharedArray, emitStore, guardedStore, etc.).
 */
export function compileTileKernel(spec: TileKernelSpec): string {
  _varCounter = 0;

  // 0. Auto vec-width selection: if vectorize is "auto", use access analysis.
  //    Works with both globalId(0) and flatGlobalId() kernels. The access
  //    analysis recognizes tagged flatGlobalId nodes as stride-1 coalesced.
  if ((spec as any).vectorize === "auto" || (spec.vectorize === undefined && (spec as any).autoVectorize)) {
    const safeWidth = computeSafeVecWidth(spec);
    if (safeWidth > 1) {
      // Also adjust the grid to account for vecWidth (each thread processes
      // safeWidth elements, so we need fewer workgroups).
      const wgSize = typeof spec.workgroupSize === "number"
        ? spec.workgroupSize : spec.workgroupSize[0] * spec.workgroupSize[1];
      const elementUniform = findElementwiseUniform(spec);
      spec = {
        ...spec,
        vectorize: safeWidth,
        ...(elementUniform ? { grid: elementwiseGrid(wgSize, { vecWidth: safeWidth, elementUniform }) } : {}),
      };
    }
  }

  // 0b. Auto-detect subgroup support
  const sgSupport = getSubgroupSupport();
  const sgSize = spec.enableSubgroups
    ? (sgSupport?.subgroupSize ?? 32)
    : (sgSupport?.supported ? (sgSupport.subgroupSize ?? 32) : 0);

  // 1. Build the IR DAG (includes constant folding + CSE in makeNode)
  // Pass sgSize so reduction primitives can choose subgroup-optimized paths
  const ctx = buildKernelIR(spec, sgSize);

  // Auto-enable subgroups if the kernel's reduction primitives used subgroup ops
  if (sgSize > 0 && !spec.enableSubgroups && ctx._usesSubgroups) {
    spec = { ...spec, enableSubgroups: true };
  }

  // 2. Lower tile-level ops if present
  let stmts = ctx.statements;
  if (hasTileStatements(stmts)) {
    // Auto-emit thread_row/thread_col if not present (tile lowering needs them)
    if (!stmts.some(s => s.kind === "let" && s.name === "thread_row")) {
      const threadRowNode: ThreadIdxNode = { id: -1, kind: "threadIdx", dim: 1, valueType: "scalar", dataType: "u32" };
      const threadColNode: ThreadIdxNode = { id: -1, kind: "threadIdx", dim: 0, valueType: "scalar", dataType: "u32" };
      stmts = [
        { kind: "let", name: "thread_row", dtype: "u32", value: threadRowNode },
        { kind: "let", name: "thread_col", dtype: "u32", value: threadColNode },
        ...stmts,
      ];
    }
    stmts = lowerTileStatements(stmts, spec);
  }

  // 3. Automatic barrier insertion (opt-in)
  if (spec.autoBarriers) {
    stmts = insertBarriers(stmts);
  }

  // 4. Auto-CSE: inject let bindings for multi-use expressions
  stmts = autoCSE(stmts);

  // 5. Loop-invariant code motion (always-on)
  stmts = hoistLoopInvariants(stmts);

  // 6. Dead code elimination (remove unused let bindings)
  stmts = eliminateDeadCode(stmts);

  return compileImperativeKernel(spec, ctx, stmts);
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
  for (const sa of ctx.vec4SharedArrays) {
    lines.push(`var<workgroup> ${sa.name}: array<vec4<f32>, ${sa.size}>;`);
  }
  if (ctx.sharedArrays.length > 0 || ctx.vec4SharedArrays.length > 0) lines.push("");

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
  const hasFlatGidVec = (spec.vectorize ?? 0) > 1 && ctx.flatGlobalIdNodeIds.length > 0;
  const needsGid = ctx.nodes.some(n => n.kind === "globalId");
  const needsWid = ctx.nodes.some(n => n.kind === "programId") || hasFlatGidVec;
  const needsLocalId = ctx.nodes.some(n => n.kind === "threadIdx");
  const needsLocalIdx = ctx.nodes.some(n => n.kind === "localIndex") || hasFlatGidVec;
  const needsNumWg = ctx.nodes.some(n => n.kind === "numWorkgroups") || hasFlatGidVec;
  // Shared arrays and tile-level stmts require local_id/local_idx even if not explicitly referenced
  const hasTileOps = ctx.sharedArrays.length > 0 || ctx.vec4SharedArrays.length > 0 ||
    ctx.statements.some(s => s.kind === "tileLoad" || s.kind === "tileStore" || s.kind === "tileLoad1d");
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
  if (needsNumWg) params.push(`  @builtin(num_workgroups) num_wg: vec3<u32>`);
  lines.push(params.join(",\n") + (params.length > 0 ? "," : ""));
  lines.push(`) {`);

  // Emit all statements
  const bindings: BindingMap = new Map();
  const stmts = overrideStatements ?? ctx.statements;

  if (spec.vectorize && spec.vectorize > 1) {
    // Auto-vectorization: unroll body VEC_WIDTH times
    const vecWidth = spec.vectorize;
    const flatGidNodeIds = ctx.flatGlobalIdNodeIds;

    if (flatGidNodeIds.length > 0) {
      // flatGlobalId path: compute base from workgroup + local index
      // _flatBase = (wid.x + wid.y * num_wg.x) * (WG * VEC) + local_idx * VEC
      const wgTotal = typeof spec.workgroupSize === "number"
        ? spec.workgroupSize : spec.workgroupSize[0] * spec.workgroupSize[1];
      lines.push(`  let _flatBase = (wid.x + wid.y * num_wg.x) * ${wgTotal * vecWidth}u + local_idx * ${vecWidth}u;`);

      for (let v = 0; v < vecWidth; v++) {
        lines.push(`  // vec element ${v}`);
        lines.push(`  {`);
        const vecBindings = new Map(bindings);
        for (const nodeId of flatGidNodeIds) {
          vecBindings.set(nodeId, `(_flatBase + ${v}u)`);
        }
        for (const stmt of stmts) {
          emitStatement(stmt, vecBindings, lines, 2);
        }
        lines.push(`  }`);
      }
    } else {
      // globalId(0) path: compute base from gid.x
      const gidXNodes = ctx.nodes.filter(n => n.kind === "globalId" && n.dim === 0);
      lines.push(`  let _base = gid.x * ${vecWidth}u;`);

      for (let v = 0; v < vecWidth; v++) {
        lines.push(`  // vec element ${v}`);
        lines.push(`  {`);
        const vecBindings = new Map(bindings);
        for (const n of gidXNodes) {
          vecBindings.set(n.id, `(_base + ${v}u)`);
        }
        for (const stmt of stmts) {
          emitStatement(stmt, vecBindings, lines, 2);
        }
        lines.push(`  }`);
      }
    }
  } else {
    for (const stmt of stmts) {
      emitStatement(stmt, bindings, lines, 1);
    }
  }

  lines.push(`}`);
  return lines.join("\n");
}

/**
 * Detect statically-true guard conditions (e.g. `1u == 1u` or folded const(1)).
 * Returns true if the expression is a compile-time constant that evaluates to true.
 */
function isStaticTrue(node: IRNode): boolean {
  // Handle folded constant (from const-const comparison folding)
  if (node.kind === "const") return node.value !== 0;
  // Handle unfolded comparison (in case folding was bypassed)
  if (node.kind === "cmp") {
    const { op, lhs, rhs } = node;
    if (lhs.kind === "const" && rhs.kind === "const") {
      const l = lhs.value, r = rhs.value;
      switch (op) {
        case "eq": return l === r;
        case "ne": return l !== r;
        case "lt": return l < r;
        case "le": return l <= r;
        case "gt": return l > r;
        case "ge": return l >= r;
      }
    }
  }
  return false;
}

/** Detect statically-false guard conditions (folded const(0) or false comparisons). */
function isStaticFalse(node: IRNode): boolean {
  if (node.kind === "const") return node.value === 0;
  return false;
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
      // Register binding: subsequent exprFor calls for the same node ID
      // will return the variable name instead of re-expanding the expression.
      // Safe because: (1) all CSE-eligible nodes are pure, (2) scoped bindings
      // (new Map(bindings) for child scopes) prevent inner bindings leaking out.
      // Only bind nodes with valid IDs (>= 0) — tile lowering helpers create
      // raw nodes with id=-1 that must NOT be cached (they're all id=-1).
      if (stmt.value.id >= 0) {
        bindings.set(stmt.value.id, stmt.name);
      }
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
        const zero = stmt.elemType === "f32" ? "0.0" : stmt.elemType === "f16" ? "f16(0.0)" : "0u";
        if (stmt.size <= 16) {
          // Unroll zero-init for small arrays
          for (let i = 0; i < stmt.size; i++) {
            lines.push(`${indent}${stmt.name}[${i}u] = ${zero};`);
          }
        } else {
          lines.push(`${indent}for (var _zi = 0u; _zi < ${stmt.size}u; _zi = _zi + 1u) {`);
          lines.push(`${indent}  ${stmt.name}[_zi] = ${zero};`);
          lines.push(`${indent}}`);
        }
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
      // Check if we can unroll: both start and bound must be ConstNode
      const startConst = stmt.start.kind === "const" ? stmt.start.value : null;
      const boundConst = stmt.bound.kind === "const" ? stmt.bound.value : null;
      const tripCount = startConst !== null && boundConst !== null ? boundConst - startConst : null;
      const shouldUnroll = tripCount !== null && tripCount >= 0 &&
        (stmt.unroll === true || tripCount <= 16);

      if (shouldUnroll && tripCount !== null && startConst !== null) {
        // Emit unrolled iterations, each in its own block scope
        for (let i = 0; i < tripCount; i++) {
          const iterVal = startConst + i;
          lines.push(`${indent}{ // unrolled ${stmt.varName}=${iterVal}`);
          lines.push(`${indent}  const ${stmt.varName} = ${iterVal}u;`);
          const childBindings = new Map(bindings);
          for (const s of stmt.body) {
            emitStatement(s, childBindings, lines, depth + 1);
          }
          lines.push(`${indent}}`);
        }
      } else {
        const start = exprFor(stmt.start, bindings, null);
        const bound = exprFor(stmt.bound, bindings, null);
        lines.push(`${indent}for (var ${stmt.varName} = ${start}; ${stmt.varName} < ${bound}; ${stmt.varName} = ${stmt.varName} + 1u) {`);
        const childBindings = new Map(bindings);
        for (const s of stmt.body) {
          emitStatement(s, childBindings, lines, depth + 1);
        }
        lines.push(`${indent}}`);
      }
      break;
    }
    case "forStride": {
      const startConst = stmt.start.kind === "const" ? stmt.start.value : null;
      const boundConst = stmt.bound.kind === "const" ? stmt.bound.value : null;
      const strideVal = stmt.stride;

      // Case 1: Fully const — emit JS-time unrolled iterations (no guards needed)
      if (startConst !== null && boundConst !== null && strideVal > 0) {
        const tripCount = Math.ceil((boundConst - startConst) / strideVal);
        if (tripCount >= 0 && (stmt.unroll === true || tripCount <= 8)) {
          for (let i = 0; i < tripCount; i++) {
            const iterVal = startConst + i * strideVal;
            lines.push(`${indent}{ // unrolled ${stmt.varName}=${iterVal}`);
            lines.push(`${indent}  const ${stmt.varName} = ${iterVal}u;`);
            const childBindings = new Map(bindings);
            for (const s of stmt.body) {
              emitStatement(s, childBindings, lines, depth + 1);
            }
            lines.push(`${indent}}`);
          }
          break;
        }
      }

      // Case 2: Dynamic start, const bound — max trip count unrolling with guards.
      // Common pattern: stridedFor(tid, TILE_SIZE, WG) for cooperative loading.
      // Max trips = ceil(bound / stride) (assuming start ∈ [0, stride)).
      if (startConst === null && boundConst !== null && strideVal > 0) {
        const maxTrips = Math.ceil(boundConst / strideVal);
        if (maxTrips >= 1 && (stmt.unroll === true ? maxTrips <= 16 : maxTrips <= 8)) {
          const startExpr = exprFor(stmt.start, bindings, null);
          for (let i = 0; i < maxTrips; i++) {
            const ivExpr = i === 0 ? startExpr : `(${startExpr} + ${i * strideVal}u)`;
            lines.push(`${indent}{ // unrolled iter ${i}`);
            lines.push(`${indent}  let ${stmt.varName} = ${ivExpr};`);
            // Guard: skip check on first iteration when bound is a multiple of stride
            // (all threads are guaranteed valid on iter 0 in that case)
            const needsGuard = i > 0 || boundConst % strideVal !== 0;
            const childBindings = new Map(bindings);
            if (needsGuard) {
              lines.push(`${indent}  if (${stmt.varName} < ${boundConst}u) {`);
              for (const s of stmt.body) {
                emitStatement(s, childBindings, lines, depth + 2);
              }
              lines.push(`${indent}  }`);
            } else {
              for (const s of stmt.body) {
                emitStatement(s, childBindings, lines, depth + 1);
              }
            }
            lines.push(`${indent}}`);
          }
          break;
        }
      }

      // Case 3: Fallback — emit runtime loop
      {
        const start = exprFor(stmt.start, bindings, null);
        const bound = exprFor(stmt.bound, bindings, null);
        lines.push(`${indent}for (var ${stmt.varName} = ${start}; ${stmt.varName} < ${bound}; ${stmt.varName} = ${stmt.varName} + ${strideVal}u) {`);
        const childBindings = new Map(bindings);
        for (const s of stmt.body) {
          emitStatement(s, childBindings, lines, depth + 1);
        }
        lines.push(`${indent}}`);
      }
      break;
    }
    case "if": {
      if (isStaticTrue(stmt.condition)) {
        // Static-true: body emitted at parent scope — bindings propagate up
        for (const s of stmt.body) emitStatement(s, bindings, lines, depth);
      } else if (!isStaticFalse(stmt.condition)) {
        const cond = exprFor(stmt.condition, bindings, null);
        lines.push(`${indent}if (${cond}) {`);
        const childBindings = new Map(bindings);
        for (const s of stmt.body) {
          emitStatement(s, childBindings, lines, depth + 1);
        }
        lines.push(`${indent}}`);
      }
      break;
    }
    case "ifElse": {
      if (isStaticTrue(stmt.condition)) {
        for (const s of stmt.body) emitStatement(s, bindings, lines, depth);
      } else if (isStaticFalse(stmt.condition)) {
        for (const s of stmt.elseBody) emitStatement(s, bindings, lines, depth);
      } else {
        const cond = exprFor(stmt.condition, bindings, null);
        lines.push(`${indent}if (${cond}) {`);
        const childBindings1 = new Map(bindings);
        for (const s of stmt.body) {
          emitStatement(s, childBindings1, lines, depth + 1);
        }
        lines.push(`${indent}} else {`);
        const childBindings2 = new Map(bindings);
        for (const s of stmt.elseBody) {
          emitStatement(s, childBindings2, lines, depth + 1);
        }
        lines.push(`${indent}}`);
      }
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
    // Vec4 array statements
    case "vec4VarArray": {
      lines.push(`${indent}var ${stmt.name}: array<vec4<f32>, ${stmt.size}>;`);
      break;
    }
    case "vec4SharedArray": {
      // Handled at module scope, not inside fn
      break;
    }
    case "vec4ArrayWrite": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.arrayName}[${idx}] = ${val};`);
      break;
    }
    case "vec4ArrayAddAssign": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.arrayName}[${idx}] = ${stmt.arrayName}[${idx}] + ${val};`);
      break;
    }
    case "directStore": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      lines.push(`${indent}${stmt.binding}[${idx}] = ${val};`);
      break;
    }
    case "guardedStore": {
      if (isStaticFalse(stmt.condition)) {
        // Dead store — omit entirely
        break;
      }
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      if (isStaticTrue(stmt.condition)) {
        // Constant-fold: emit unconditional store
        lines.push(`${indent}${stmt.binding}[${idx}] = ${val};`);
      } else {
        const cond = exprFor(stmt.condition, bindings, null);
        lines.push(`${indent}if (${cond}) {`);
        lines.push(`${indent}  ${stmt.binding}[${idx}] = ${val};`);
        lines.push(`${indent}}`);
      }
      break;
    }
    case "atomicOp": {
      const idx = exprFor(stmt.idx, bindings, null);
      const val = exprFor(stmt.value, bindings, null);
      const fnName: Record<string, string> = {
        max: "atomicMax",
        min: "atomicMin",
        add: "atomicAdd",
        or: "atomicOr",
        and: "atomicAnd",
        xor: "atomicXor",
        exchange: "atomicExchange",
      };
      lines.push(`${indent}${fnName[stmt.op]}(&${stmt.binding}[${idx}], ${val});`);
      break;
    }
    case "atomicCAS": {
      const idx = exprFor(stmt.idx, bindings, null);
      const exp = exprFor(stmt.expected, bindings, null);
      const des = exprFor(stmt.desired, bindings, null);
      lines.push(`${indent}let _cas_result = atomicCompareExchangeWeak(&${stmt.binding}[${idx}], ${exp}, ${des});`);
      lines.push(`${indent}let ${stmt.oldValueVar} = _cas_result.old_value;`);
      lines.push(`${indent}let ${stmt.exchangedVar} = select(0u, 1u, _cas_result.exchanged);`);
      break;
    }
    case "return": {
      lines.push(`${indent}return;`);
      break;
    }
  }
}

// ============================================================================
// Dead Code Elimination
// ============================================================================

/** Collect all namedRef names referenced by an IR expression tree. */
function collectExprNames(node: IRNode, names: Set<string>): void {
  switch (node.kind) {
    case "namedRef": names.add(node.name); break;
    case "binary": collectExprNames(node.lhs, names); collectExprNames(node.rhs, names); break;
    case "unary": collectExprNames(node.input, names); break;
    case "cast": collectExprNames(node.input, names); break;
    case "bitcast": collectExprNames(node.input, names); break;
    case "cmp": collectExprNames(node.lhs, names); collectExprNames(node.rhs, names); break;
    case "select":
      collectExprNames(node.condition, names);
      collectExprNames(node.trueVal, names);
      collectExprNames(node.falseVal, names);
      break;
    case "load":
      collectExprNames(node.offsets, names);
      if (node.mask) collectExprNames(node.mask, names);
      break;
    case "sharedRead": collectExprNames(node.idx, names); break;
    case "arrayRead": collectExprNames(node.idx, names); break;
    case "subgroupShuffleXor":
      collectExprNames(node.value, names);
      collectExprNames(node.mask, names);
      break;
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd":
      collectExprNames(node.value, names);
      break;
    case "vec4dot":
      for (const n of node.a) collectExprNames(n, names);
      for (const n of node.b) collectExprNames(n, names);
      break;
    // Vec4 native nodes
    case "vec4Construct":
      collectExprNames(node.x, names); collectExprNames(node.y, names);
      collectExprNames(node.z, names); collectExprNames(node.w, names);
      break;
    case "vec4Splat": collectExprNames(node.value, names); break;
    case "vec4NativeDot": collectExprNames(node.a, names); collectExprNames(node.b, names); break;
    case "vec4Component": collectExprNames(node.value, names); break;
    case "vec4Binary": collectExprNames(node.a, names); collectExprNames(node.b, names); break;
    case "vec4ArrayRead": collectExprNames(node.idx, names); break;
    case "vec4SharedRead": collectExprNames(node.idx, names); break;
    // Leaf nodes (programId, uniform, const, threadIdx, localIndex, globalId): no names
  }
}

/** Collect all namedRef names from all expressions in a statement (recursively into bodies). */
function collectAllStmtNames(stmt: Statement, names: Set<string>): void {
  switch (stmt.kind) {
    case "let":
      collectExprNames(stmt.value, names);
      break;
    case "var":
      collectExprNames(stmt.value, names);
      break;
    case "assign":
    case "addAssign":
      collectExprNames(stmt.value, names);
      break;
    case "indexAssign":
    case "indexAddAssign":
      collectExprNames(stmt.idx, names);
      collectExprNames(stmt.value, names);
      break;
    case "forRange":
      collectExprNames(stmt.start, names);
      collectExprNames(stmt.bound, names);
      for (const s of stmt.body) collectAllStmtNames(s, names);
      break;
    case "forStride":
      collectExprNames(stmt.start, names);
      collectExprNames(stmt.bound, names);
      for (const s of stmt.body) collectAllStmtNames(s, names);
      break;
    case "if":
      collectExprNames(stmt.condition, names);
      for (const s of stmt.body) collectAllStmtNames(s, names);
      break;
    case "ifElse":
      collectExprNames(stmt.condition, names);
      for (const s of stmt.body) collectAllStmtNames(s, names);
      for (const s of stmt.elseBody) collectAllStmtNames(s, names);
      break;
    case "sharedWrite":
      collectExprNames(stmt.idx, names);
      collectExprNames(stmt.value, names);
      break;
    case "guardedStore":
      collectExprNames(stmt.condition, names);
      collectExprNames(stmt.idx, names);
      collectExprNames(stmt.value, names);
      break;
    case "directStore":
      collectExprNames(stmt.idx, names);
      collectExprNames(stmt.value, names);
      break;
    case "atomicOp":
      collectExprNames(stmt.idx, names);
      collectExprNames(stmt.value, names);
      break;
    case "atomicCAS":
      collectExprNames(stmt.idx, names);
      collectExprNames(stmt.expected, names);
      collectExprNames(stmt.desired, names);
      break;
    case "vec4ArrayWrite":
    case "vec4ArrayAddAssign":
      collectExprNames(stmt.idx, names);
      collectExprNames(stmt.value, names);
      break;
    // barrier, return, varArray, vec4VarArray, vec4SharedArray: no IRNode expressions to collect
  }
}

// ============================================================================
// Automatic Barrier Insertion
// ============================================================================

/**
 * Collect shared array names read by IR nodes within a statement (non-recursive into forRange).
 */
function getSharedReadsFromExpr(node: IRNode, names: Set<string>): void {
  switch (node.kind) {
    case "sharedRead":
      names.add(node.arrayName);
      getSharedReadsFromExpr(node.idx, names);
      return;
    case "binary":
      getSharedReadsFromExpr(node.lhs, names);
      getSharedReadsFromExpr(node.rhs, names);
      return;
    case "unary":
    case "cast":
      getSharedReadsFromExpr(node.input, names);
      return;
    case "cmp":
      getSharedReadsFromExpr(node.lhs, names);
      getSharedReadsFromExpr(node.rhs, names);
      return;
    case "select":
      getSharedReadsFromExpr(node.condition, names);
      getSharedReadsFromExpr(node.trueVal, names);
      getSharedReadsFromExpr(node.falseVal, names);
      return;
    case "load":
      getSharedReadsFromExpr(node.offsets, names);
      return;
    case "arrayRead":
      getSharedReadsFromExpr(node.idx, names);
      return;
    case "subgroupShuffleXor":
      getSharedReadsFromExpr(node.value, names);
      getSharedReadsFromExpr(node.mask, names);
      return;
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd":
      getSharedReadsFromExpr(node.value, names);
      return;
    case "vec4Construct":
      getSharedReadsFromExpr(node.x, names);
      getSharedReadsFromExpr(node.y, names);
      getSharedReadsFromExpr(node.z, names);
      getSharedReadsFromExpr(node.w, names);
      return;
    case "vec4Splat": getSharedReadsFromExpr(node.value, names); return;
    case "vec4NativeDot":
      getSharedReadsFromExpr(node.a, names);
      getSharedReadsFromExpr(node.b, names);
      return;
    case "vec4Component": getSharedReadsFromExpr(node.value, names); return;
    case "vec4Binary":
      getSharedReadsFromExpr(node.a, names);
      getSharedReadsFromExpr(node.b, names);
      return;
    case "vec4SharedRead":
      names.add(node.arrayName);
      getSharedReadsFromExpr(node.idx, names);
      return;
    case "vec4ArrayRead":
      getSharedReadsFromExpr(node.idx, names);
      return;
  }
}

/** Get shared array names read by a statement's expressions (not recursing into forRange). */
function getSharedReadsFromStmt(stmt: Statement): Set<string> {
  const reads = new Set<string>();
  switch (stmt.kind) {
    case "let":
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "var":
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "assign":
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "addAssign":
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "indexAssign":
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "indexAddAssign":
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "sharedWrite":
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "guardedStore":
      getSharedReadsFromExpr(stmt.condition, reads);
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "directStore":
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "atomicOp":
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.value, reads);
      break;
    case "atomicCAS":
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.expected, reads);
      getSharedReadsFromExpr(stmt.desired, reads);
      break;
    case "forRange":
      getSharedReadsFromExpr(stmt.start, reads);
      getSharedReadsFromExpr(stmt.bound, reads);
      // Don't recurse into body — forRange is a separate scope
      break;
    case "forStride":
      getSharedReadsFromExpr(stmt.start, reads);
      getSharedReadsFromExpr(stmt.bound, reads);
      break;
    case "if":
      getSharedReadsFromExpr(stmt.condition, reads);
      for (const s of stmt.body) {
        for (const r of getSharedReadsFromStmt(s)) reads.add(r);
      }
      break;
    case "ifElse":
      getSharedReadsFromExpr(stmt.condition, reads);
      for (const s of stmt.body) {
        for (const r of getSharedReadsFromStmt(s)) reads.add(r);
      }
      for (const s of stmt.elseBody) {
        for (const r of getSharedReadsFromStmt(s)) reads.add(r);
      }
      break;
    case "vec4ArrayWrite":
    case "vec4ArrayAddAssign":
      getSharedReadsFromExpr(stmt.idx, reads);
      getSharedReadsFromExpr(stmt.value, reads);
      if (stmt.isShared) {
        // Reading from shared in the value expression
      }
      break;
  }
  return reads;
}

/** Get shared array names written by a statement (not recursing into forRange). */
function getSharedWritesFromStmt(stmt: Statement): Set<string> {
  const writes = new Set<string>();
  switch (stmt.kind) {
    case "sharedWrite":
      writes.add(stmt.arrayName);
      break;
    case "vec4ArrayWrite":
    case "vec4ArrayAddAssign":
      if (stmt.isShared) writes.add(stmt.arrayName);
      break;
    case "if":
      for (const s of stmt.body) {
        for (const w of getSharedWritesFromStmt(s)) writes.add(w);
      }
      break;
    case "ifElse":
      for (const s of stmt.body) {
        for (const w of getSharedWritesFromStmt(s)) writes.add(w);
      }
      for (const s of stmt.elseBody) {
        for (const w of getSharedWritesFromStmt(s)) writes.add(w);
      }
      break;
  }
  return writes;
}

/**
 * Insert workgroupBarrier() between shared memory writes and subsequent reads.
 * Does not double-barrier: existing barriers clear the dirty set.
 */
export function insertBarriers(stmts: Statement[]): Statement[] {
  const result: Statement[] = [];
  const dirtyArrays = new Set<string>();

  for (const stmt of stmts) {
    // Recurse into forRange bodies first
    if (stmt.kind === "forRange") {
      const newBody = insertBarriersForLoop(stmt.body);
      result.push({ ...stmt, body: newBody } as Statement);
      // forRange may write shared memory — track it
      for (const s of stmt.body) {
        for (const w of getSharedWritesFromStmt(s)) dirtyArrays.add(w);
      }
      continue;
    }

    // Check if this statement reads from a dirty shared array
    const reads = getSharedReadsFromStmt(stmt);
    let needsBarrier = false;
    for (const r of reads) {
      if (dirtyArrays.has(r)) {
        needsBarrier = true;
        break;
      }
    }

    if (needsBarrier) {
      result.push({ kind: "barrier" });
      dirtyArrays.clear();
    }

    // Existing barrier clears dirty set
    if (stmt.kind === "barrier") {
      dirtyArrays.clear();
    }

    result.push(stmt);

    // Track writes
    for (const w of getSharedWritesFromStmt(stmt)) {
      dirtyArrays.add(w);
    }
  }

  return result;
}

/**
 * Insert barriers in a loop body, considering that the body repeats.
 * If the body writes shared memory and a later iteration reads it,
 * a barrier is needed at the loop boundary.
 */
function insertBarriersForLoop(body: Statement[]): Statement[] {
  // First pass: insert barriers within the body
  let result = insertBarriers(body);

  // Check if the body writes then reads the same shared array across iterations
  // i.e., writes at end feed into reads at start of next iteration
  const allWrites = new Set<string>();
  const allReads = new Set<string>();
  for (const s of body) {
    for (const w of getSharedWritesFromStmt(s)) allWrites.add(w);
    for (const r of getSharedReadsFromStmt(s)) allReads.add(r);
  }

  // If any written array is also read, we may need a barrier at the end
  let needsBoundaryBarrier = false;
  for (const w of allWrites) {
    if (allReads.has(w)) {
      needsBoundaryBarrier = true;
      break;
    }
  }

  if (needsBoundaryBarrier) {
    // Check if the last statement is already a barrier
    const last = result[result.length - 1];
    if (!last || last.kind !== "barrier") {
      result = [...result, { kind: "barrier" }];
    }
  }

  return result;
}

/**
 * Validate that shared memory accesses have proper barriers.
 * Returns warning strings for missing barriers (diagnostic, doesn't modify code).
 */
export function validateBarriers(stmts: Statement[]): string[] {
  const warnings: string[] = [];
  const dirtyArrays = new Set<string>();

  function walk(stmts: Statement[]): void {
    for (const stmt of stmts) {
      if (stmt.kind === "barrier") {
        dirtyArrays.clear();
        continue;
      }

      const reads = getSharedReadsFromStmt(stmt);
      for (const r of reads) {
        if (dirtyArrays.has(r)) {
          warnings.push(`Missing barrier: shared array '${r}' written then read without barrier`);
          dirtyArrays.clear();
          break;
        }
      }

      for (const w of getSharedWritesFromStmt(stmt)) {
        dirtyArrays.add(w);
      }

      if (stmt.kind === "forRange") {
        walk(stmt.body);
      } else if (stmt.kind === "if") {
        walk(stmt.body);
      } else if (stmt.kind === "ifElse") {
        walk(stmt.body);
        walk(stmt.elseBody);
      }
    }
  }

  walk(stmts);
  return warnings;
}

// ============================================================================
// Loop-Invariant Code Motion (LICM)
// ============================================================================

/**
 * Collect names modified inside a statement list (loop-variant names).
 * Includes: assigned vars, loop variables, shared arrays written.
 */
function collectModifiedNames(stmts: Statement[], names: Set<string>): void {
  for (const stmt of stmts) {
    switch (stmt.kind) {
      case "assign":
        names.add(stmt.name);
        break;
      case "addAssign":
        names.add(stmt.name);
        break;
      case "indexAssign":
        names.add(stmt.arrayName);
        break;
      case "indexAddAssign":
        names.add(stmt.arrayName);
        break;
      case "sharedWrite":
        names.add(stmt.arrayName);
        break;
      case "vec4ArrayWrite":
      case "vec4ArrayAddAssign":
        names.add(stmt.arrayName);
        break;
      case "var":
        names.add(stmt.name);
        break;
      case "atomicCAS":
        names.add(stmt.oldValueVar);
        names.add(stmt.exchangedVar);
        break;
      case "forRange":
        names.add(stmt.varName);
        collectModifiedNames(stmt.body, names);
        break;
      case "forStride":
        names.add(stmt.varName);
        collectModifiedNames(stmt.body, names);
        break;
      case "if":
        collectModifiedNames(stmt.body, names);
        break;
      case "ifElse":
        collectModifiedNames(stmt.body, names);
        collectModifiedNames(stmt.elseBody, names);
        break;
    }
  }
}

/**
 * Check if an IR expression depends on any of the given names (namedRef references).
 */
function exprDependsOn(node: IRNode, names: Set<string>): boolean {
  switch (node.kind) {
    case "namedRef":
      return names.has(node.name);
    case "binary":
      return exprDependsOn(node.lhs, names) || exprDependsOn(node.rhs, names);
    case "unary":
    case "cast":
      return exprDependsOn(node.input, names);
    case "cmp":
      return exprDependsOn(node.lhs, names) || exprDependsOn(node.rhs, names);
    case "select":
      return exprDependsOn(node.condition, names) ||
             exprDependsOn(node.trueVal, names) ||
             exprDependsOn(node.falseVal, names);
    case "load":
      // Global loads are never hoisted (memory may change between iterations)
      return true;
    case "sharedRead":
      // Only safe to hoist if no shared array is written in the loop
      return names.has(node.arrayName);
    case "arrayRead":
      return names.has(node.arrayName) || exprDependsOn(node.idx, names);
    case "subgroupShuffleXor":
      return exprDependsOn(node.value, names) || exprDependsOn(node.mask, names);
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd":
      return exprDependsOn(node.value, names);
    case "vec4Construct":
      return exprDependsOn(node.x, names) || exprDependsOn(node.y, names) ||
             exprDependsOn(node.z, names) || exprDependsOn(node.w, names);
    case "vec4Splat": return exprDependsOn(node.value, names);
    case "vec4NativeDot": return exprDependsOn(node.a, names) || exprDependsOn(node.b, names);
    case "vec4Component": return exprDependsOn(node.value, names);
    case "vec4Binary": return exprDependsOn(node.a, names) || exprDependsOn(node.b, names);
    case "vec4SharedRead": return names.has(node.arrayName);
    case "vec4ArrayRead": return names.has(node.arrayName) || exprDependsOn(node.idx, names);
    default:
      return false;
  }
}

/**
 * Check if an expression contains any global memory load.
 */
function exprContainsLoad(node: IRNode): boolean {
  switch (node.kind) {
    case "load":
      return true;
    case "binary":
      return exprContainsLoad(node.lhs) || exprContainsLoad(node.rhs);
    case "unary":
    case "cast":
      return exprContainsLoad(node.input);
    case "cmp":
      return exprContainsLoad(node.lhs) || exprContainsLoad(node.rhs);
    case "select":
      return exprContainsLoad(node.condition) ||
             exprContainsLoad(node.trueVal) ||
             exprContainsLoad(node.falseVal);
    case "sharedRead":
    case "vec4SharedRead":
      return false;
    case "arrayRead":
    case "vec4ArrayRead":
      return exprContainsLoad(node.idx);
    case "subgroupShuffleXor":
      return exprContainsLoad(node.value) || exprContainsLoad(node.mask);
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd":
      return exprContainsLoad(node.value);
    case "vec4Construct":
      return exprContainsLoad(node.x) || exprContainsLoad(node.y) ||
             exprContainsLoad(node.z) || exprContainsLoad(node.w);
    case "vec4Splat": return exprContainsLoad(node.value);
    case "vec4NativeDot": return exprContainsLoad(node.a) || exprContainsLoad(node.b);
    case "vec4Component": return exprContainsLoad(node.value);
    case "vec4Binary": return exprContainsLoad(node.a) || exprContainsLoad(node.b);
    default:
      return false;
  }
}

/**
 * Hoist loop-invariant `let` bindings out of `forRange` loops.
 *
 * Safety rules (conservative):
 * - Only hoists `let` bindings (immutable, no side effects)
 * - Never hoists expressions containing global `load` nodes
 * - Never hoists expressions referencing shared arrays written in the loop
 * - Never hoists across barriers (a let after a barrier stays after it)
 * - Recurse inner-to-outer (inner loops hoisted first)
 */
export function hoistLoopInvariants(stmts: Statement[]): Statement[] {
  const result: Statement[] = [];

  for (const stmt of stmts) {
    if (stmt.kind === "forRange" || stmt.kind === "forStride") {
      // First recurse into the body (inner-to-outer)
      const innerHoisted = hoistLoopInvariants(stmt.body);

      // Collect loop-variant names
      const variantNames = new Set<string>();
      variantNames.add(stmt.varName); // Loop variable is always variant
      collectModifiedNames(innerHoisted, variantNames);

      // Check if any shared array is written in the loop
      const sharedWritten = new Set<string>();
      collectSharedWriteNames(innerHoisted, sharedWritten);

      // Add shared written names to variant set so sharedRead of them won't hoist
      for (const sw of sharedWritten) variantNames.add(sw);

      // Partition body into hoistable and non-hoistable
      const hoisted: Statement[] = [];
      const remaining: Statement[] = [];
      let seenBarrier = false;

      for (const s of innerHoisted) {
        if (s.kind === "barrier") {
          seenBarrier = true;
          remaining.push(s);
          continue;
        }

        if (s.kind === "let" && !seenBarrier &&
            !exprDependsOn(s.value, variantNames) &&
            !exprContainsLoad(s.value)) {
          hoisted.push(s);
        } else {
          // If a let is NOT hoisted, mark its name as variant so subsequent
          // bindings that reference it won't be incorrectly hoisted either.
          if (s.kind === "let") variantNames.add(s.name);
          remaining.push(s);
        }
      }

      // Emit hoisted lets before the loop
      result.push(...hoisted);
      result.push({ ...stmt, body: remaining } as Statement);
    } else if (stmt.kind === "if") {
      result.push({ ...stmt, body: hoistLoopInvariants(stmt.body) } as Statement);
    } else if (stmt.kind === "ifElse") {
      result.push({
        ...stmt,
        body: hoistLoopInvariants(stmt.body),
        elseBody: hoistLoopInvariants(stmt.elseBody),
      } as Statement);
    } else {
      result.push(stmt);
    }
  }

  return result;
}

/** Collect shared array names written in a statement list. */
function collectSharedWriteNames(stmts: Statement[], names: Set<string>): void {
  for (const stmt of stmts) {
    if (stmt.kind === "sharedWrite") {
      names.add(stmt.arrayName);
    } else if (stmt.kind === "forRange" || stmt.kind === "forStride") {
      collectSharedWriteNames(stmt.body, names);
    } else if (stmt.kind === "if") {
      collectSharedWriteNames(stmt.body, names);
    } else if (stmt.kind === "ifElse") {
      collectSharedWriteNames(stmt.body, names);
      collectSharedWriteNames(stmt.elseBody, names);
    }
  }
}

// ============================================================================
// Dead Code Elimination
// ============================================================================

/**
 * Remove `let` statements whose bindings are never referenced.
 * Applies bottom-up: inner scopes are processed first, then outer scopes
 * propagate liveness backward through let chains.
 */
function eliminateDeadCode(stmts: Statement[]): Statement[] {
  // Step 1: Recurse into nested bodies (bottom-up)
  const processed = stmts.map(s => {
    switch (s.kind) {
      case "forRange": return { ...s, body: eliminateDeadCode(s.body) } as Statement;
      case "forStride": return { ...s, body: eliminateDeadCode(s.body) } as Statement;
      case "if":       return { ...s, body: eliminateDeadCode(s.body) } as Statement;
      case "ifElse":   return { ...s, body: eliminateDeadCode(s.body), elseBody: eliminateDeadCode(s.elseBody) } as Statement;
      default: return s;
    }
  });

  // Step 2: Collect names used by non-let statements (including nested bodies)
  const usedNames = new Set<string>();
  for (const s of processed) {
    if (s.kind !== "let") {
      collectAllStmtNames(s, usedNames);
    }
  }

  // Step 3: Reverse-propagate through let statements at this scope
  for (let i = processed.length - 1; i >= 0; i--) {
    const s = processed[i];
    if (s.kind === "let" && usedNames.has(s.name)) {
      collectExprNames(s.value, usedNames);
    }
  }

  // Step 4: Collect all node IDs from non-let statements (for auto-CSE awareness).
  // Auto-CSE injected `let` bindings work by node ID: codegen's bindings.set(node.id, name)
  // connects them. If a let's value.id appears in other statements, it's effectively live.
  const nonLetNodeIds = new Set<number>();
  for (const s of processed) {
    if (s.kind !== "let") {
      const nodeMap = new Map<number, IRNode>();
      collectStmtCSENodes(s, nodeMap);
      for (const id of nodeMap.keys()) nonLetNodeIds.add(id);
    }
  }

  // Step 5: Filter dead lets — keep if name is used OR if value's node ID is referenced
  return processed.filter(s => {
    if (s.kind !== "let") return true;
    if (usedNames.has(s.name)) return true;
    // Keep auto-CSE bindings: value node has valid ID and appears in other statements
    if (s.value.id >= 0 && nonLetNodeIds.has(s.value.id)) return true;
    return false;
  });
}

// ============================================================================
// Auto-CSE (Common Subexpression Elimination)
// ============================================================================

/**
 * Check if a node is trivial — cheap enough to duplicate in WGSL.
 * These nodes compile to a single token/word (variable name, literal, builtin).
 */
function isTrivialNode(node: IRNode): boolean {
  switch (node.kind) {
    case "const":
    case "namedRef":
    case "programId":
    case "threadIdx":
    case "localIndex":
    case "globalId":
    case "numWorkgroups":
    case "uniform":
      return true;
    default:
      return false;
  }
}

/**
 * Check if an expression tree contains any memory read (global load, shared read).
 * Such expressions should not be auto-CSE'd because the memory value
 * may change between uses (e.g., across barriers).
 */
function exprContainsMemoryRead(node: IRNode): boolean {
  switch (node.kind) {
    case "load":
    case "sharedRead":
    case "vec4SharedRead":
      return true;
    case "binary":
      return exprContainsMemoryRead(node.lhs) || exprContainsMemoryRead(node.rhs);
    case "unary":
    case "cast":
      return exprContainsMemoryRead(node.input);
    case "bitcast":
      return exprContainsMemoryRead(node.input);
    case "cmp":
      return exprContainsMemoryRead(node.lhs) || exprContainsMemoryRead(node.rhs);
    case "select":
      return exprContainsMemoryRead(node.condition) ||
             exprContainsMemoryRead(node.trueVal) ||
             exprContainsMemoryRead(node.falseVal);
    case "arrayRead":
    case "vec4ArrayRead":
      return exprContainsMemoryRead(node.idx);
    case "subgroupShuffleXor":
      return exprContainsMemoryRead(node.value) || exprContainsMemoryRead(node.mask);
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd":
      return exprContainsMemoryRead(node.value);
    case "vec4Construct":
      return exprContainsMemoryRead(node.x) || exprContainsMemoryRead(node.y) ||
             exprContainsMemoryRead(node.z) || exprContainsMemoryRead(node.w);
    case "vec4Splat": return exprContainsMemoryRead(node.value);
    case "vec4NativeDot": return exprContainsMemoryRead(node.a) || exprContainsMemoryRead(node.b);
    case "vec4Component": return exprContainsMemoryRead(node.value);
    case "vec4Binary": return exprContainsMemoryRead(node.a) || exprContainsMemoryRead(node.b);
    default:
      return false;
  }
}

/**
 * Walk an IR expression tree and collect non-trivial, memory-read-free candidate nodes.
 * Only considers nodes with valid IDs (>= 0, from makeNode).
 * Always recurses into children even for memory-read nodes, so pure sub-expressions
 * (like address calculations inside loads) can still be CSE'd.
 */
function collectExprCSECandidates(node: IRNode, nodes: Map<number, IRNode>): void {
  if (node.id < 0 || isTrivialNode(node)) return;
  // Register this node as a CSE candidate only if it's memory-read-free
  if (!exprContainsMemoryRead(node)) {
    nodes.set(node.id, node);
  }
  // Always recurse into children to find shared sub-expressions
  switch (node.kind) {
    case "binary":
      collectExprCSECandidates(node.lhs, nodes);
      collectExprCSECandidates(node.rhs, nodes);
      break;
    case "unary":
    case "cast":
      collectExprCSECandidates(node.input, nodes);
      break;
    case "bitcast":
      collectExprCSECandidates(node.input, nodes);
      break;
    case "cmp":
      collectExprCSECandidates(node.lhs, nodes);
      collectExprCSECandidates(node.rhs, nodes);
      break;
    case "select":
      collectExprCSECandidates(node.condition, nodes);
      collectExprCSECandidates(node.trueVal, nodes);
      collectExprCSECandidates(node.falseVal, nodes);
      break;
    case "subgroupShuffleXor":
      collectExprCSECandidates(node.value, nodes);
      collectExprCSECandidates(node.mask, nodes);
      break;
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd":
      collectExprCSECandidates(node.value, nodes);
      break;
    case "vec4Construct":
      collectExprCSECandidates(node.x, nodes);
      collectExprCSECandidates(node.y, nodes);
      collectExprCSECandidates(node.z, nodes);
      collectExprCSECandidates(node.w, nodes);
      break;
    case "vec4Splat": collectExprCSECandidates(node.value, nodes); break;
    case "vec4NativeDot":
      collectExprCSECandidates(node.a, nodes);
      collectExprCSECandidates(node.b, nodes);
      break;
    case "vec4Component": collectExprCSECandidates(node.value, nodes); break;
    case "vec4Binary":
      collectExprCSECandidates(node.a, nodes);
      collectExprCSECandidates(node.b, nodes);
      break;
    case "load":
      collectExprCSECandidates(node.offsets, nodes);
      if (node.mask) collectExprCSECandidates(node.mask, nodes);
      break;
    case "sharedRead":
      collectExprCSECandidates(node.idx, nodes);
      break;
    case "vec4SharedRead":
      collectExprCSECandidates(node.idx, nodes);
      break;
    case "arrayRead":
    case "vec4ArrayRead":
      collectExprCSECandidates(node.idx, nodes);
      break;
  }
}

/**
 * Collect all CSE candidate node IDs from all expressions in a statement
 * (including nested bodies like forRange, if, etc.).
 */
function collectStmtCSENodes(stmt: Statement, nodes: Map<number, IRNode>): void {
  switch (stmt.kind) {
    case "let":
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "var":
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "assign":
    case "addAssign":
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "indexAssign":
    case "indexAddAssign":
      collectExprCSECandidates(stmt.idx, nodes);
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "forRange":
      collectExprCSECandidates(stmt.start, nodes);
      collectExprCSECandidates(stmt.bound, nodes);
      for (const s of stmt.body) collectStmtCSENodes(s, nodes);
      break;
    case "forStride":
      collectExprCSECandidates(stmt.start, nodes);
      collectExprCSECandidates(stmt.bound, nodes);
      for (const s of stmt.body) collectStmtCSENodes(s, nodes);
      break;
    case "if":
      collectExprCSECandidates(stmt.condition, nodes);
      for (const s of stmt.body) collectStmtCSENodes(s, nodes);
      break;
    case "ifElse":
      collectExprCSECandidates(stmt.condition, nodes);
      for (const s of stmt.body) collectStmtCSENodes(s, nodes);
      for (const s of stmt.elseBody) collectStmtCSENodes(s, nodes);
      break;
    case "sharedWrite":
      collectExprCSECandidates(stmt.idx, nodes);
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "guardedStore":
      collectExprCSECandidates(stmt.condition, nodes);
      collectExprCSECandidates(stmt.idx, nodes);
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "directStore":
      collectExprCSECandidates(stmt.idx, nodes);
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "atomicOp":
      collectExprCSECandidates(stmt.idx, nodes);
      collectExprCSECandidates(stmt.value, nodes);
      break;
    case "atomicCAS":
      collectExprCSECandidates(stmt.idx, nodes);
      collectExprCSECandidates(stmt.expected, nodes);
      collectExprCSECandidates(stmt.desired, nodes);
      break;
    case "vec4ArrayWrite":
    case "vec4ArrayAddAssign":
      collectExprCSECandidates(stmt.idx, nodes);
      collectExprCSECandidates(stmt.value, nodes);
      break;
    // barrier, return, varArray, vec4VarArray, vec4SharedArray: no candidate expressions
  }
}

/**
 * Compute expression tree depth (0 for trivials, 1 for single op on trivials, etc.).
 */
function exprDepth(node: IRNode): number {
  if (isTrivialNode(node) || node.id < 0) return 0;
  switch (node.kind) {
    case "binary": return 1 + Math.max(exprDepth(node.lhs), exprDepth(node.rhs));
    case "unary": case "cast": case "bitcast": return 1 + exprDepth((node as any).input);
    case "cmp": return 1 + Math.max(exprDepth(node.lhs), exprDepth(node.rhs));
    case "select": return 1 + Math.max(exprDepth(node.condition), exprDepth(node.trueVal), exprDepth(node.falseVal));
    default: return 1;
  }
}

/**
 * Collect all node IDs that are proper sub-expressions of a given node.
 */
function collectSubExprIds(node: IRNode, ids: Set<number>): void {
  if (node.id < 0 || isTrivialNode(node)) return;
  switch (node.kind) {
    case "binary":
      ids.add(node.lhs.id); collectSubExprIds(node.lhs, ids);
      ids.add(node.rhs.id); collectSubExprIds(node.rhs, ids);
      break;
    case "unary": case "cast": case "bitcast":
      ids.add((node as any).input.id); collectSubExprIds((node as any).input, ids);
      break;
    case "cmp":
      ids.add(node.lhs.id); collectSubExprIds(node.lhs, ids);
      ids.add(node.rhs.id); collectSubExprIds(node.rhs, ids);
      break;
    case "select":
      ids.add(node.condition.id); collectSubExprIds(node.condition, ids);
      ids.add(node.trueVal.id); collectSubExprIds(node.trueVal, ids);
      ids.add(node.falseVal.id); collectSubExprIds(node.falseVal, ids);
      break;
  }
}

/**
 * Walk an IR expression tree and push every non-trivial, memory-read-free node
 * occurrence (NOT deduplicated). Same node appearing N times produces N entries.
 */
function countExprCSEOccurrences(node: IRNode, out: { id: number; node: IRNode }[]): void {
  if (node.id < 0 || isTrivialNode(node)) return;
  if (!exprContainsMemoryRead(node)) {
    out.push({ id: node.id, node });
  }
  switch (node.kind) {
    case "binary":
      countExprCSEOccurrences(node.lhs, out);
      countExprCSEOccurrences(node.rhs, out);
      break;
    case "unary": case "cast":
      countExprCSEOccurrences(node.input, out);
      break;
    case "bitcast":
      countExprCSEOccurrences(node.input, out);
      break;
    case "cmp":
      countExprCSEOccurrences(node.lhs, out);
      countExprCSEOccurrences(node.rhs, out);
      break;
    case "select":
      countExprCSEOccurrences(node.condition, out);
      countExprCSEOccurrences(node.trueVal, out);
      countExprCSEOccurrences(node.falseVal, out);
      break;
    case "subgroupShuffleXor":
      countExprCSEOccurrences(node.value, out);
      countExprCSEOccurrences(node.mask, out);
      break;
    case "subgroupAdd": case "subgroupMax": case "subgroupMin":
    case "subgroupBroadcastFirst": case "subgroupInclusiveAdd":
      countExprCSEOccurrences(node.value, out);
      break;
    case "vec4Construct":
      countExprCSEOccurrences(node.x, out);
      countExprCSEOccurrences(node.y, out);
      countExprCSEOccurrences(node.z, out);
      countExprCSEOccurrences(node.w, out);
      break;
    case "vec4Splat": countExprCSEOccurrences(node.value, out); break;
    case "vec4NativeDot":
      countExprCSEOccurrences(node.a, out);
      countExprCSEOccurrences(node.b, out);
      break;
    case "vec4Component": countExprCSEOccurrences(node.value, out); break;
    case "vec4Binary":
      countExprCSEOccurrences(node.a, out);
      countExprCSEOccurrences(node.b, out);
      break;
    case "load":
      countExprCSEOccurrences(node.offsets, out);
      if (node.mask) countExprCSEOccurrences(node.mask, out);
      break;
    case "sharedRead":
    case "vec4SharedRead":
    case "arrayRead":
    case "vec4ArrayRead":
      countExprCSEOccurrences(node.idx, out);
      break;
  }
}

/**
 * Count all CSE candidate occurrences from expressions in a single statement.
 * Unlike collectStmtCSENodes, does NOT deduplicate — same node appearing N
 * times produces N occurrences. Also does NOT recurse into child bodies
 * (forRange/forStride/if/ifElse) — those are handled by autoCSE's recursive
 * call into child scopes. Only counts expressions at this scope level (loop
 * bounds, conditions, and leaf statement expressions).
 */
function countStmtCSEOccurrences(stmt: Statement, out: { id: number; node: IRNode }[]): void {
  switch (stmt.kind) {
    case "let": case "var":
      countExprCSEOccurrences(stmt.value, out);
      break;
    case "assign": case "addAssign":
      countExprCSEOccurrences(stmt.value, out);
      break;
    case "indexAssign": case "indexAddAssign":
      countExprCSEOccurrences(stmt.idx, out);
      countExprCSEOccurrences(stmt.value, out);
      break;
    case "forRange":
      // Only count loop bounds at this scope — body handled by recursive autoCSE
      countExprCSEOccurrences(stmt.start, out);
      countExprCSEOccurrences(stmt.bound, out);
      break;
    case "forStride":
      countExprCSEOccurrences(stmt.start, out);
      countExprCSEOccurrences(stmt.bound, out);
      break;
    case "if":
      countExprCSEOccurrences(stmt.condition, out);
      break;
    case "ifElse":
      countExprCSEOccurrences(stmt.condition, out);
      break;
    case "sharedWrite":
      countExprCSEOccurrences(stmt.idx, out);
      countExprCSEOccurrences(stmt.value, out);
      break;
    case "guardedStore":
      countExprCSEOccurrences(stmt.condition, out);
      countExprCSEOccurrences(stmt.idx, out);
      countExprCSEOccurrences(stmt.value, out);
      break;
    case "directStore":
      countExprCSEOccurrences(stmt.idx, out);
      countExprCSEOccurrences(stmt.value, out);
      break;
    case "atomicOp":
      countExprCSEOccurrences(stmt.idx, out);
      countExprCSEOccurrences(stmt.value, out);
      break;
    case "atomicCAS":
      countExprCSEOccurrences(stmt.idx, out);
      countExprCSEOccurrences(stmt.expected, out);
      countExprCSEOccurrences(stmt.desired, out);
      break;
    case "vec4ArrayWrite": case "vec4ArrayAddAssign":
      countExprCSEOccurrences(stmt.idx, out);
      countExprCSEOccurrences(stmt.value, out);
      break;
  }
}

/**
 * Auto-CSE pass: inject `let` bindings for multi-use expression nodes.
 *
 * Processes bottom-up (inner scopes first). At each scope, counts per-statement
 * references and injects `let` bindings before the first statement that uses
 * each multi-referenced non-trivial expression. The codegen's `bindings.set`
 * mechanism then ensures subsequent references use the variable name.
 *
 * parentBoundIds tracks nodes already bound at ancestor scopes to avoid
 * redundant re-binding (which produces useless alias lets).
 *
 * LICM runs after this pass and hoists loop-invariant injected bindings.
 */
export function autoCSE(stmts: Statement[], parentBoundIds: Set<number> = new Set()): Statement[] {
  // Phase 1: Count references at this scope using two strategies:
  //
  // (a) Cross-statement sharing: per-statement dedup (walks into children).
  //     A node in statement A and statement B gets count += 2.
  //     This is the original behavior that correctly handles cross-scope sharing
  //     (e.g., a node used in a loop body AND after the loop).
  //
  // (b) Intra-expression sharing: per-occurrence counting (no child body recursion).
  //     A node used twice within a single leaf statement (e.g., x.mul(x)) gets
  //     count += 2 instead of += 1. This catches the case the dedup approach missed.
  //
  // We take the max count from both strategies for each node.
  const refCounts = new Map<number, { count: number; node: IRNode; firstIdx: number }>();
  const letValueIds = new Set<number>();

  for (let i = 0; i < stmts.length; i++) {
    const stmt = stmts[i];
    if (stmt.kind === "let" && stmt.value.id >= 0) {
      letValueIds.add(stmt.value.id);
    }

    // Strategy (a): deduped per-statement (walks into children for cross-scope detection)
    const nodeSet = new Map<number, IRNode>();
    collectStmtCSENodes(stmt, nodeSet);
    for (const [id, node] of nodeSet) {
      const entry = refCounts.get(id);
      if (entry) {
        entry.count++;
      } else {
        refCounts.set(id, { count: 1, node, firstIdx: i });
      }
    }

    // Strategy (b): occurrence counting at this scope level (no child body recursion)
    // Only add extra counts for nodes that appeared more than once per-occurrence
    // but only once in the dedup set (intra-expression sharing within this statement)
    const occurrences: { id: number; node: IRNode }[] = [];
    countStmtCSEOccurrences(stmt, occurrences);
    const occCounts = new Map<number, { count: number; node: IRNode }>();
    for (const { id, node } of occurrences) {
      const e = occCounts.get(id);
      if (e) e.count++;
      else occCounts.set(id, { count: 1, node });
    }
    for (const [id, { count: occCount, node }] of occCounts) {
      if (occCount > 1) {
        // This node appeared multiple times within this single statement's expressions
        // Ensure refCounts reflects this (may already be >= occCount from cross-statement)
        const entry = refCounts.get(id);
        if (entry && entry.count < occCount) {
          entry.count = occCount;
        } else if (!entry) {
          refCounts.set(id, { count: occCount, node, firstIdx: i });
        }
      }
    }
  }

  // Phase 2: Find multi-use candidates
  // Skip: already bound by existing let, bound at ancestor scope, or depth <= 1
  const candidates: { firstIdx: number; node: IRNode; id: number }[] = [];
  for (const [id, { count, node, firstIdx }] of refCounts) {
    if (count > 1 && !letValueIds.has(id) && !parentBoundIds.has(id) && exprDepth(node) > 1) {
      candidates.push({ firstIdx, node, id });
    }
  }

  // Phase 2b: Prune sub-expressions that are FULLY captured by parent candidates.
  // A sub-expression S of parent candidate P can be pruned only if ALL of S's
  // references are through parent candidates (no external references). Otherwise
  // S must get its own binding because codegen would inline it at external sites.
  const candidateSubExprs = new Map<number, Set<number>>(); // candidate id → sub-expr ids
  for (const c of candidates) {
    const subs = new Set<number>();
    collectSubExprIds(c.node, subs);
    candidateSubExprs.set(c.id, subs);
  }
  const toBind: { firstIdx: number; name: string; node: IRNode; id: number }[] = [];
  for (const c of candidates) {
    // Count how many parent candidate usages "cover" this candidate's references
    let parentCoverage = 0;
    for (const other of candidates) {
      if (other.id !== c.id && candidateSubExprs.get(other.id)!.has(c.id)) {
        parentCoverage += refCounts.get(other.id)!.count;
      }
    }
    // Keep if not a sub-expression of any candidate, or has external references
    if (parentCoverage === 0 || refCounts.get(c.id)!.count > parentCoverage) {
      toBind.push({ firstIdx: c.firstIdx, name: freshVar("cse"), node: c.node, id: c.id });
    }
  }

  // Build the set of IDs bound at this scope + ancestors for child recursion
  const childBoundIds = new Set(parentBoundIds);
  for (const b of toBind) {
    childBoundIds.add(b.node.id);
  }
  // Also include sub-expressions of bound nodes — once a parent expression is
  // bound to a variable, codegen won't expand its sub-expressions, so children
  // don't need separate bindings for them either.
  for (const b of toBind) {
    collectSubExprIds(b.node, childBoundIds);
  }

  // Phase 3: Recurse into child scopes (top-down: children see parent bindings)
  const processed = stmts.map(s => {
    switch (s.kind) {
      case "forRange": return { ...s, body: autoCSE(s.body, childBoundIds) } as Statement;
      case "forStride": return { ...s, body: autoCSE(s.body, childBoundIds) } as Statement;
      case "if": return { ...s, body: autoCSE(s.body, childBoundIds) } as Statement;
      case "ifElse": return { ...s, body: autoCSE(s.body, childBoundIds), elseBody: autoCSE(s.elseBody, childBoundIds) } as Statement;
      default: return s;
    }
  });

  if (toBind.length === 0) return processed;
  toBind.sort((a, b) => a.firstIdx - b.firstIdx);

  // Phase 4: Inject let bindings before first use
  const injByIdx = new Map<number, typeof toBind>();
  for (const b of toBind) {
    if (!injByIdx.has(b.firstIdx)) injByIdx.set(b.firstIdx, []);
    injByIdx.get(b.firstIdx)!.push(b);
  }

  // Topological sort bindings within each firstIdx group so that if binding B's
  // expression tree references binding A's node, A is emitted before B.
  for (const [, group] of injByIdx) {
    if (group.length > 1) {
      // Build sub-expression sets per binding
      const subSets = new Map<number, Set<number>>();
      for (const b of group) {
        subSets.set(b.id, candidateSubExprs.get(b.id) ?? new Set());
      }
      // Stable topological sort: binding A before B if B's sub-exprs contain A's id
      group.sort((a, b) => {
        const bDepsA = subSets.get(b.id)!.has(a.id);
        const aDepsB = subSets.get(a.id)!.has(b.id);
        if (bDepsA && !aDepsB) return -1; // A before B
        if (aDepsB && !bDepsA) return 1;  // B before A
        return 0; // no dependency — preserve order
      });
    }
  }

  const result: Statement[] = [];
  for (let i = 0; i < processed.length; i++) {
    const inj = injByIdx.get(i);
    if (inj) {
      for (const b of inj) {
        result.push({ kind: "let", name: b.name, value: b.node, dtype: b.node.dataType } as Statement);
      }
    }
    result.push(processed[i]);
  }

  return result;
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
 */



// ============================================================================
// Binding / Uniform Codegen
// ============================================================================

function emitUniformStruct(spec: TileKernelSpec): string {
  const entries = Object.entries(spec.uniforms);
  if (entries.length === 0) return ""; // No uniforms → no struct needed

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
  const hasUniforms = Object.keys(spec.uniforms).length > 0;

  for (let i = 0; i < entries.length; i++) {
    // Insert uniform at the specified index if requested
    if (hasUniforms && uniformIdx !== undefined && bindingIndex === uniformIdx) {
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
  if (hasUniforms && (uniformIdx === undefined || bindingIndex <= uniformIdx)) {
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
      case "tileLoad1d":
      case "tileStore":
      case "blockAlloc":
      case "blockLoad":
      case "blockStore":
      case "blockDot":
      case "blockReduce":
      case "blockUnary":
      case "blockBinary":
        return true;
      case "forRange":
        if (hasTileStatements(s.body)) return true;
        break;
      case "forStride":
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
 * Tile ops (tileLoad, tileStore) are expanded into cooperative loading loops,
 * per-element loops, etc.
 *
 * Non-tile statements are passed through unchanged (with recursive lowering
 * of any nested bodies).
 */
function lowerTileStatements(stmts: Statement[], spec: TileKernelSpec): Statement[] {
  const result: Statement[] = [];
  let i = 0;
  while (i < stmts.length) {
    const s = stmts[i];

    switch (s.kind) {
      case "tileLoad":
        result.push(...lowerTileLoad(s, spec));
        break;
      case "tileLoad1d":
        result.push(...lowerTileLoad1D(s));
        break;
      case "tileStore":
        result.push(...lowerTileStore(s, spec));
        break;
      // Block API statements
      case "blockAlloc":
        result.push(...lowerBlockAlloc(s));
        break;
      case "blockLoad":
        result.push(...lowerBlockLoad(s, spec));
        break;
      case "blockStore":
        result.push(...lowerBlockStore(s));
        break;
      case "blockDot":
        result.push(...lowerBlockDot(s));
        break;
      case "blockReduce":
        result.push(...lowerBlockReduce(s));
        break;
      case "blockUnary":
        result.push(...lowerBlockUnary(s));
        break;
      case "blockBinary":
        result.push(...lowerBlockBinary(s));
        break;
      case "forRange":
        result.push({
          ...s,
          body: lowerTileStatements(s.body, spec),
        });
        break;
      case "forStride":
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
    i++;
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
/** Check if an IR node is a constant with value 1 (for stride optimization). */
function isConstOne(node: IRNode): boolean {
  return node.kind === "const" && node.value === 1;
}
/** Multiply, but skip when one operand is const(1) — eliminates `* 1u` in generated code. */
function mulOrSkip(a: IRNode, b: IRNode, dt: DataType = "u32"): IRNode {
  if (isConstOne(b)) return a;
  if (isConstOne(a)) return b;
  return binOp("mul", a, b, dt);
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
function vec4DotExpr(
  a: [IRNode, IRNode, IRNode, IRNode],
  b: [IRNode, IRNode, IRNode, IRNode],
): IRNode {
  return { id: -1, kind: "vec4dot", a, b, valueType: "scalar", dataType: "f32" };
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
  const smemStride = stmt.smemStride ?? tileCols;
  const totalElems = tileRows * tileCols;
  const totalThreads = getTotalThreads(spec);
  const elemsPerThread = Math.ceil(totalElems / totalThreads);
  const isPadded = smemStride !== tileCols;

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

  // Bind globalRow/globalCol to let variables to avoid recomputation in index + mask
  const globalRowName = freshVar("gr");
  const globalColName = freshVar("gc");
  ifBody.push({
    kind: "let", name: globalRowName, dtype: "u32",
    value: binOp("add", ptr.outerRange.base, ref(rowName)),
  });
  ifBody.push({
    kind: "let", name: globalColName, dtype: "u32",
    value: binOp("add", ptr.innerRange.base, ref(colName)),
  });

  // globalIdx = base + globalRow * outerStride + globalCol * innerStride
  // Uses mulOrSkip to eliminate `* 1u` for non-transposed strides
  const gIdxName = freshVar("gIdx");
  ifBody.push({
    kind: "let", name: gIdxName, dtype: "u32",
    value: binOp("add",
      binOp("add", ptr.baseOffset, mulOrSkip(ref(globalRowName), ptr.outerStride)),
      mulOrSkip(ref(globalColName), ptr.innerStride),
    ),
  });

  // Mask check: globalRow < outerBound && globalCol < innerBound
  const maskCond = andOp(
    cmpOp("lt", ref(globalRowName), mask.outerBound),
    cmpOp("lt", ref(globalColName), mask.innerBound),
  );

  // Shared memory write index: row * smemStride + col (padded) or flat (unpadded)
  const smemIdx = isPadded
    ? binOp("add", binOp("mul", ref(rowName), cU32(smemStride)), ref(colName))
    : ref(flatName);

  // if (mask) { shared[idx] = f32(binding[gIdx]) } else { shared[idx] = 0.0 }
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
      idx: smemIdx,
      value: loadAsF32,
    }],
    elseBody: [{
      kind: "sharedWrite",
      arrayName: sharedName,
      idx: smemIdx,
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
      binOp("add", ptr.baseOffset, mulOrSkip(ref(rowName), ptr.outerStride)),
      mulOrSkip(ref(colName), ptr.innerStride),
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
// Block Statement Lowering (unified Block API)
// ============================================================================

/**
 * Lower blockAlloc → var array with zero-init or fill.
 */
function lowerBlockAlloc(stmt: BlockAllocStmt): Statement[] {
  const { name, rows, cols, elemType, initValue } = stmt;
  const size = rows * cols;
  const result: Statement[] = [];

  if (initValue !== undefined) {
    // Allocate without zero-init, then fill
    result.push({
      kind: "varArray", name, elemType, size, skipZeroInit: true,
    });
    const iVar = freshVar("fi");
    const fillVal: IRNode = elemType === "f32"
      ? cF32(initValue)
      : cU32(initValue);
    result.push({
      kind: "forRange", varName: iVar, start: cU32(0), bound: cU32(size),
      body: [{
        kind: "indexAssign",
        arrayName: name,
        idx: ref(iVar),
        value: fillVal,
      }],
    });
  } else {
    // Zero-initialized
    result.push({
      kind: "varArray", name, elemType, size,
    });
  }

  return result;
}

/**
 * Lower blockLoad → per-thread register load (thread ptr) or cooperative shared load (tile ptr).
 */
function lowerBlockLoad(stmt: BlockLoadStmt, spec: TileKernelSpec): Statement[] {
  if (stmt.ptrKind === "thread") {
    return lowerBlockLoadThread(stmt, spec);
  } else {
    return lowerBlockLoadTile(stmt, spec);
  }
}

/**
 * Lower blockLoad with thread ptr → per-thread register loading.
 *
 * Each thread loads rows*cols elements into a register array.
 * For rows=1, cols=D: load D elements starting at base.
 * For rows=R, cols=D: load R rows, each D elements, base + r*stride + d.
 * With guard: if false, zero-fill.
 */
function lowerBlockLoadThread(stmt: BlockLoadStmt, spec: TileKernelSpec): Statement[] {
  const { binding, name, rows, cols, elemType, threadBase, threadStride, guard } = stmt;
  const size = rows * cols;
  const result: Statement[] = [];
  const bindingDtype = spec.bindings[binding]?.type ?? elemType;
  const useVec4 = cols % 4 === 0 && bindingDtype === "f32";

  // Allocate register array
  result.push({
    kind: "varArray", name, elemType: "f32", size, skipZeroInit: true,
  });

  // Build loading loop
  const loadBody: Statement[] = [];

  if (rows === 1) {
    if (useVec4) {
      // Vec4 path: for (d4 = 0; d4 < cols/4; d4++) { 4× unrolled loads }
      const d4Var = freshVar("d4");
      const offVar = freshVar("off");
      const innerBody: Statement[] = [];

      innerBody.push({
        kind: "let", name: offVar, dtype: "u32",
        value: binOp("add", threadBase!, binOp("mul", ref(d4Var), cU32(4))),
      });
      for (let k = 0; k < 4; k++) {
        const loadIdx = k === 0 ? ref(offVar) : binOp("add", ref(offVar), cU32(k));
        let loadExpr: IRNode = loadBinding(binding, loadIdx, bindingDtype);
        innerBody.push({
          kind: "indexAssign",
          arrayName: name,
          idx: binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k)),
          value: loadExpr,
        });
      }

      loadBody.push({
        kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(cols / 4),
        body: innerBody,
      });
    } else {
      // Scalar path: for (d = 0; d < cols; d++) { reg[d] = buf[base + d]; }
      const dVar = freshVar("d");
      const innerBody: Statement[] = [];

      const loadIdx = binOp("add", threadBase!, ref(dVar));
      let loadExpr: IRNode = loadBinding(binding, loadIdx, bindingDtype);
      if (bindingDtype !== "f32") {
        loadExpr = castNode(loadExpr, "f32");
      }

      innerBody.push({
        kind: "indexAssign",
        arrayName: name,
        idx: ref(dVar),
        value: loadExpr,
      });

      loadBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(cols),
        body: innerBody,
      });
    }
  } else {
    // Multiple rows: for (r = 0; r < rows; r++) for (d = 0; d < cols; d++)
    const rVar = freshVar("r");
    const dVar = freshVar("d");
    const innerBody: Statement[] = [];

    const rowBase = binOp("add", threadBase!, binOp("mul", ref(rVar), threadStride!));
    const loadIdx = binOp("add", rowBase, ref(dVar));
    let loadExpr: IRNode = loadBinding(binding, loadIdx, bindingDtype);
    if (bindingDtype !== "f32") {
      loadExpr = castNode(loadExpr, "f32");
    }
    const regIdx = binOp("add", binOp("mul", ref(rVar), cU32(cols)), ref(dVar));

    innerBody.push({
      kind: "indexAssign",
      arrayName: name,
      idx: regIdx,
      value: loadExpr,
    });

    loadBody.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(rows),
      body: [{
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(cols),
        body: innerBody,
      }],
    });
  }

  if (guard) {
    // Guarded: if (guard) { load } else { zero-fill }
    const zeroBody: Statement[] = [];
    const zVar = freshVar("zi");
    zeroBody.push({
      kind: "forRange", varName: zVar, start: cU32(0), bound: cU32(size),
      body: [{
        kind: "indexAssign",
        arrayName: name,
        idx: ref(zVar),
        value: cF32(0),
      }],
    });
    result.push({
      kind: "ifElse",
      condition: guard,
      body: loadBody,
      elseBody: zeroBody,
    });
  } else {
    result.push(...loadBody);
  }

  return result;
}

/**
 * Lower blockLoad with tile ptr → cooperative shared memory loading.
 * Reuses the same logic as lowerTileLoad.
 */
function lowerBlockLoadTile(stmt: BlockLoadStmt, spec: TileKernelSpec): Statement[] {
  if (!stmt.tilePtr || !stmt.tileMask) {
    throw new Error("blockLoad with ptrKind=tile requires tilePtr and tileMask");
  }
  // Delegate to existing tile load lowering, with padded smemStride
  const smemStride = stmt.cols + 1;
  return lowerTileLoad({
    kind: "tileLoad",
    binding: stmt.binding,
    ptr: stmt.tilePtr,
    mask: stmt.tileMask,
    sharedName: stmt.name,
    tileRows: stmt.rows,
    tileCols: stmt.cols,
    elemType: stmt.elemType,
    smemStride,
  }, spec);
}

/**
 * Lower blockStore → per-thread store from register array to global memory.
 *
 * Each thread stores rows*cols elements from reg to buf.
 * For rows=1, cols=D: buf[base + d] = reg[d] for d in 0..D
 * With guard: only store when guard is true.
 */
function lowerBlockStore(stmt: BlockStoreStmt): Statement[] {
  const { binding, blockName, rows, cols, base, stride, guard } = stmt;
  const result: Statement[] = [];
  const useVec4 = rows === 1 && cols % 4 === 0;

  const storeBody: Statement[] = [];

  if (rows === 1 && useVec4) {
    // Vec4 path: for (d4 = 0; d4 < cols/4; d4++) { 4× unrolled stores }
    const d4Var = freshVar("sd4");
    const offVar = freshVar("soff");
    const innerBody: Statement[] = [];
    innerBody.push({
      kind: "let", name: offVar, dtype: "u32",
      value: binOp("add", base, binOp("mul", ref(d4Var), cU32(4))),
    });
    for (let k = 0; k < 4; k++) {
      const storeIdx = k === 0 ? ref(offVar) : binOp("add", ref(offVar), cU32(k));
      innerBody.push({
        kind: "indexAssign",
        arrayName: binding,
        idx: storeIdx,
        value: arrayRead(blockName, binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k))),
      });
    }
    storeBody.push({
      kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(cols / 4),
      body: innerBody,
    });
  } else if (rows === 1) {
    // Scalar single row: for (d = 0; d < cols; d++) { buf[base + d] = reg[d]; }
    const dVar = freshVar("sd");
    storeBody.push({
      kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(cols),
      body: [{
        kind: "indexAssign",
        arrayName: binding,
        idx: binOp("add", base, ref(dVar)),
        value: arrayRead(blockName, ref(dVar)),
      }],
    });
  } else {
    // Multiple rows
    const rVar = freshVar("sr");
    const dVar = freshVar("sd");
    storeBody.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(rows),
      body: [{
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(cols),
        body: [{
          kind: "indexAssign",
          arrayName: binding,
          idx: binOp("add", binOp("add", base, binOp("mul", ref(rVar), stride)), ref(dVar)),
          value: arrayRead(blockName, binOp("add", binOp("mul", ref(rVar), cU32(cols)), ref(dVar))),
        }],
      }],
    });
  }

  if (guard) {
    result.push({
      kind: "if",
      condition: guard,
      body: storeBody,
    });
  } else {
    result.push(...storeBody);
  }

  return result;
}

/**
 * Lower blockDot → inner product or outer product loops.
 *
 * Three patterns based on placement:
 * 1. register × shared^T → inner product (QK^T scores)
 * 2. register × shared   → inner product (PV output)
 * 3. shared × shared     → outer product (matmul)
 */
function lowerBlockDot(stmt: BlockDotStmt): Statement[] {
  const { aPlacement, bPlacement, bTransposed } = stmt;

  if (aPlacement === "shared" && bPlacement === "shared") {
    return lowerBlockDotSharedShared(stmt);
  } else if (aPlacement === "register" && bPlacement === "shared" && bTransposed) {
    return lowerBlockDotRegSharedT(stmt);
  } else if (aPlacement === "register" && bPlacement === "shared" && !bTransposed) {
    return lowerBlockDotRegSharedNN(stmt);
  } else {
    throw new Error(
      `blockDot: unsupported placement pattern: ${aPlacement} × ${bPlacement}` +
      (bTransposed ? "^T" : ""),
    );
  }
}

/**
 * shared × shared → register outer product (matmul pattern).
 *
 * A: shared [tileM × innerDim], B: shared [innerDim × tileN]
 * Each thread computes a [threadTileM × threadTileN] tile of the result.
 * Thread position: (thread_row, thread_col) from 2D workgroup layout.
 *
 * Lowering pattern (barrier + outer product):
 *   barrier()
 *   for kk in 0..innerDim:
 *     a_vals[tm] = A[(thread_row*ttM + tm) * innerDim + kk]
 *     b_vals[tn] = B[kk * bCols + thread_col*ttN + tn]
 *     acc[tm*ttN + tn] += a_vals[tm] * b_vals[tn]
 *   barrier()
 */
function lowerBlockDotSharedShared(stmt: BlockDotStmt): Statement[] {
  const { aName, bName, resultName, accName, aCols, bCols, threadTileM, threadTileN } = stmt;
  if (!threadTileM || !threadTileN) {
    throw new Error("shared×shared blockDot requires threadTileM and threadTileN");
  }

  const aSmemStride = stmt.aSmemStride ?? aCols;  // padded stride for A in shared memory
  const bSmemStride = stmt.bSmemStride ?? bCols;  // padded stride for B in shared memory
  const innerDim = aCols;         // K dimension (A cols = B rows for non-transposed)
  const isAccumulate = !!accName; // dotAccum vs dot

  const result: Statement[] = [];

  // barrier() before reading shared memory
  result.push({ kind: "barrier" });

  const kkVar = freshVar("kk");
  const kkLoop: Statement[] = [];

  // Load a_vals from shared A: a_vals[tm] = A[(thread_row*ttM + tm) * aSmemStride + kk]
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
      value: sharedRead(aName,
        binOp("add",
          binOp("mul",
            binOp("add", binOp("mul", ref("thread_row"), cU32(threadTileM)), ref(tmVar1)),
            cU32(aSmemStride),
          ),
          ref(kkVar),
        ),
      ),
    }],
  });

  // Load b_vals from shared B: b_vals[tn] = B[kk * bSmemStride + thread_col*ttN + tn]
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
      value: sharedRead(bName,
        binOp("add",
          binOp("mul", ref(kkVar), cU32(bSmemStride)),
          binOp("add", binOp("mul", ref("thread_col"), cU32(threadTileN)), ref(tnVar1)),
        ),
      ),
    }],
  });

  // Outer product: acc[tm*ttN + tn] += a_vals[tm] * b_vals[tn]
  const targetName = accName ?? resultName;
  const tmVar2 = freshVar("tm");
  const tnVar2 = freshVar("tn");
  kkLoop.push({
    kind: "forRange", varName: tmVar2, start: cU32(0), bound: cU32(threadTileM),
    body: [{
      kind: "forRange", varName: tnVar2, start: cU32(0), bound: cU32(threadTileN),
      body: [{
        kind: "indexAddAssign",
        arrayName: targetName,
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
 * register × shared^T → inner product (QK^T pattern).
 *
 * A: register [aRows × aCols], B: shared [bRows × bCols] (original, used transposed)
 * Result: [aRows × bRows]
 *
 * For scalar mode (aRows=1):
 *   for j in 0..bRows: s = 0; for d in 0..aCols: s += a[d] * b_smem[j*bCols + d]; result[j] = s;
 */
function lowerBlockDotRegSharedT(stmt: BlockDotStmt): Statement[] {
  const { aName, bName, resultName, accName, aRows, aCols, bRows, bCols } = stmt;
  const bStride = stmt.bSmemStride ?? bCols;  // padded stride for B in shared memory
  const result: Statement[] = [];
  const innerDim = aCols; // = bCols (dimension being contracted)
  const outCols = bRows;  // B^T has bRows columns
  const useVec4 = innerDim % 4 === 0;

  if (aRows === 1) {
    // Scalar mode: single row of A
    const jVar = freshVar("j");
    const sVar = freshVar("s");

    const jBody: Statement[] = [];
    // var s: f32 = 0.0;
    jBody.push({ kind: "var", name: sVar, dtype: "f32", value: cF32(0) });

    if (useVec4) {
      // Vec4 path: s += dot(vec4(a[d4*4..+3]), vec4(b[j*bStride+d4*4..+3]))
      const d4Var = freshVar("d4");
      jBody.push({
        kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(innerDim / 4),
        body: [{
          kind: "addAssign",
          name: sVar,
          value: vec4DotExpr(
            [0, 1, 2, 3].map(k =>
              arrayRead(aName, binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k))),
            ) as [IRNode, IRNode, IRNode, IRNode],
            [0, 1, 2, 3].map(k =>
              sharedRead(bName, binOp("add",
                binOp("add", binOp("mul", ref(jVar), cU32(bStride)), binOp("mul", ref(d4Var), cU32(4))),
                cU32(k),
              )),
            ) as [IRNode, IRNode, IRNode, IRNode],
          ),
        }],
      });
    } else {
      // Scalar path: for d in 0..innerDim: s += a[d] * b_smem[j*bStride + d]
      const dVar = freshVar("d");
      jBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(innerDim),
        body: [{
          kind: "addAssign",
          name: sVar,
          value: binOp("mul",
            arrayRead(aName, ref(dVar)),
            sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bStride)), ref(dVar))),
            "f32",
          ),
        }],
      });
    }

    // Store to result
    if (accName) {
      jBody.push({
        kind: "indexAddAssign",
        arrayName: resultName,
        idx: ref(jVar),
        value: ref(sVar, "f32"),
      });
    } else {
      jBody.push({
        kind: "indexAssign",
        arrayName: resultName,
        idx: ref(jVar),
        value: ref(sVar, "f32"),
      });
    }

    result.push({
      kind: "forRange", varName: jVar, start: cU32(0), bound: cU32(outCols),
      body: jBody,
    });
  } else {
    // Multi-row mode (general case)
    const rVar = freshVar("r");
    const jVar = freshVar("j");
    const sVar = freshVar("s");

    const rBody: Statement[] = [];
    const jBody: Statement[] = [];
    jBody.push({ kind: "var", name: sVar, dtype: "f32", value: cF32(0) });

    if (useVec4) {
      const d4Var = freshVar("d4");
      jBody.push({
        kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(innerDim / 4),
        body: [{
          kind: "addAssign",
          name: sVar,
          value: vec4DotExpr(
            [0, 1, 2, 3].map(k =>
              arrayRead(aName, binOp("add",
                binOp("add", binOp("mul", ref(rVar), cU32(aCols)), binOp("mul", ref(d4Var), cU32(4))),
                cU32(k),
              )),
            ) as [IRNode, IRNode, IRNode, IRNode],
            [0, 1, 2, 3].map(k =>
              sharedRead(bName, binOp("add",
                binOp("add", binOp("mul", ref(jVar), cU32(bStride)), binOp("mul", ref(d4Var), cU32(4))),
                cU32(k),
              )),
            ) as [IRNode, IRNode, IRNode, IRNode],
          ),
        }],
      });
    } else {
      const dVar = freshVar("d");
      jBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(innerDim),
        body: [{
          kind: "addAssign",
          name: sVar,
          value: binOp("mul",
            arrayRead(aName, binOp("add", binOp("mul", ref(rVar), cU32(aCols)), ref(dVar))),
            sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bStride)), ref(dVar))),
            "f32",
          ),
        }],
      });
    }

    const outIdx = binOp("add", binOp("mul", ref(rVar), cU32(outCols)), ref(jVar));
    if (accName) {
      jBody.push({ kind: "indexAddAssign", arrayName: resultName, idx: outIdx, value: ref(sVar, "f32") });
    } else {
      jBody.push({ kind: "indexAssign", arrayName: resultName, idx: outIdx, value: ref(sVar, "f32") });
    }
    rBody.push({
      kind: "forRange", varName: jVar, start: cU32(0), bound: cU32(outCols),
      body: jBody,
    });
    result.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(aRows),
      body: rBody,
    });
  }

  return result;
}

/**
 * register × shared (no transpose) → inner product (PV pattern).
 *
 * A: register [aRows × aCols], B: shared [bRows × bCols]
 * Result: [aRows × bCols]
 *
 * For scalar mode (aRows=1):
 *   result[d] = Σ_j a[j] * b_smem[j*bCols + d]
 *   Outer loop over j for sequential shared memory access:
 *   for j in 0..aCols: p = a[j]; for d in 0..bCols: result[d] += p * b_smem[j*bCols + d];
 */
function lowerBlockDotRegSharedNN(stmt: BlockDotStmt): Statement[] {
  const { aName, bName, resultName, accName, aRows, aCols, bRows, bCols } = stmt;
  const bStride = stmt.bSmemStride ?? bCols;  // padded stride for B in shared memory
  const result: Statement[] = [];
  const outCols = bCols;
  const useVec4 = outCols % 4 === 0;

  if (aRows === 1) {
    const jVar = freshVar("j");
    const pVar = freshVar("p");

    const jBody: Statement[] = [];
    // let p = a[j];
    jBody.push({
      kind: "let", name: pVar, dtype: "f32",
      value: arrayRead(aName, ref(jVar)),
    });

    if (useVec4) {
      // Vec4 path: for d4 in 0..outCols/4: result[d4*4+k] += p * b_smem[j*bStride+d4*4+k]
      const d4Var = freshVar("d4");
      const d4Body: Statement[] = [];
      for (let k = 0; k < 4; k++) {
        const regIdx = binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k));
        const smemIdx = binOp("add",
          binOp("add", binOp("mul", ref(jVar), cU32(bStride)), binOp("mul", ref(d4Var), cU32(4))),
          cU32(k),
        );
        d4Body.push({
          kind: "indexAddAssign",
          arrayName: resultName,
          idx: regIdx,
          value: binOp("mul", ref(pVar, "f32"), sharedRead(bName, smemIdx), "f32"),
        });
      }
      jBody.push({
        kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(outCols / 4),
        body: d4Body,
      });
    } else {
      // Scalar path: for d in 0..bCols: result[d] += p * b_smem[j*bStride + d]
      const dVar = freshVar("d");
      jBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(outCols),
        body: [{
          kind: "indexAddAssign",
          arrayName: resultName,
          idx: ref(dVar),
          value: binOp("mul",
            ref(pVar, "f32"),
            sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bStride)), ref(dVar))),
            "f32",
          ),
        }],
      });
    }

    result.push({
      kind: "forRange", varName: jVar, start: cU32(0), bound: cU32(aCols),
      body: jBody,
    });
  } else {
    // Multi-row general case
    const rVar = freshVar("r");
    const jVar = freshVar("j");
    const dVar = freshVar("d");
    const pVar = freshVar("p");

    const rBody: Statement[] = [];
    const jBody: Statement[] = [];
    jBody.push({
      kind: "let", name: pVar, dtype: "f32",
      value: arrayRead(aName, binOp("add", binOp("mul", ref(rVar), cU32(aCols)), ref(jVar))),
    });
    jBody.push({
      kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(outCols),
      body: [{
        kind: "indexAddAssign",
        arrayName: resultName,
        idx: binOp("add", binOp("mul", ref(rVar), cU32(outCols)), ref(dVar)),
        value: binOp("mul",
          ref(pVar, "f32"),
          sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bStride)), ref(dVar))),
          "f32",
        ),
      }],
    });
    rBody.push({
      kind: "forRange", varName: jVar, start: cU32(0), bound: cU32(aCols),
      body: jBody,
    });
    result.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(aRows),
      body: rBody,
    });
  }

  return result;
}

/**
 * Lower blockReduce → per-thread reduction loop.
 *
 * axis=1: reduce across columns → [R×1]
 *   result[r] = op(input[r*C + 0], ..., input[r*C + C-1])
 *
 * axis=0: reduce across rows → [1×C]
 *   result[c] = op(input[0*C + c], ..., input[(R-1)*C + c])
 */
function lowerBlockReduce(stmt: BlockReduceStmt): Statement[] {
  const { inputName, outputName, inputRows, inputCols, axis, op } = stmt;
  const result: Statement[] = [];

  if (axis === 1) {
    // Reduce across columns: [R×C] → [R×1]
    const outSize = inputRows;
    result.push({
      kind: "varArray", name: outputName, elemType: "f32", size: outSize, skipZeroInit: true,
    });

    const rVar = freshVar("rr");
    const cVar = freshVar("rc");
    const initVal = op === "max" ? cF32(F32_NEG_MAX) : cF32(0);

    const rBody: Statement[] = [];
    // Initialize result[r] = init
    rBody.push({
      kind: "indexAssign", arrayName: outputName, idx: ref(rVar),
      value: initVal,
    });
    // for c in 0..inputCols: result[r] = op(result[r], input[r*C + c])
    const inputIdx = binOp("add", binOp("mul", ref(rVar), cU32(inputCols)), ref(cVar));
    let accumExpr: IRNode;
    if (op === "sum") {
      accumExpr = binOp("add",
        arrayRead(outputName, ref(rVar)),
        arrayRead(inputName, inputIdx),
        "f32",
      );
    } else {
      // max
      accumExpr = {
        id: -1, kind: "binary", op: "max",
        lhs: arrayRead(outputName, ref(rVar)),
        rhs: arrayRead(inputName, inputIdx),
        valueType: "scalar", dataType: "f32",
      };
    }
    rBody.push({
      kind: "forRange", varName: cVar, start: cU32(0), bound: cU32(inputCols),
      body: [{
        kind: "indexAssign", arrayName: outputName, idx: ref(rVar),
        value: accumExpr,
      }],
    });

    result.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(inputRows),
      body: rBody,
    });
  } else {
    // axis === 0: Reduce across rows: [R×C] → [1×C]
    const outSize = inputCols;
    result.push({
      kind: "varArray", name: outputName, elemType: "f32", size: outSize, skipZeroInit: true,
    });

    const cVar = freshVar("rc");
    const rVar = freshVar("rr");
    const initVal = op === "max" ? cF32(F32_NEG_MAX) : cF32(0);

    const cBody: Statement[] = [];
    cBody.push({
      kind: "indexAssign", arrayName: outputName, idx: ref(cVar),
      value: initVal,
    });
    const inputIdx = binOp("add", binOp("mul", ref(rVar), cU32(inputCols)), ref(cVar));
    let accumExpr: IRNode;
    if (op === "sum") {
      accumExpr = binOp("add",
        arrayRead(outputName, ref(cVar)),
        arrayRead(inputName, inputIdx),
        "f32",
      );
    } else {
      accumExpr = {
        id: -1, kind: "binary", op: "max",
        lhs: arrayRead(outputName, ref(cVar)),
        rhs: arrayRead(inputName, inputIdx),
        valueType: "scalar", dataType: "f32",
      };
    }
    cBody.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(inputRows),
      body: [{
        kind: "indexAssign", arrayName: outputName, idx: ref(cVar),
        value: accumExpr,
      }],
    });

    result.push({
      kind: "forRange", varName: cVar, start: cU32(0), bound: cU32(inputCols),
      body: cBody,
    });
  }

  return result;
}

/**
 * Lower blockUnary → per-element unary operation.
 */
function lowerBlockUnary(stmt: BlockUnaryStmt): Statement[] {
  const { inputName, outputName, rows, cols, op, inPlace } = stmt;
  const size = rows * cols;
  const result: Statement[] = [];

  // Allocate output if not in-place
  if (!inPlace) {
    result.push({
      kind: "varArray", name: outputName, elemType: "f32", size, skipZeroInit: true,
    });
  }

  const iVar = freshVar("ui");
  const inputVal = arrayRead(inputName, ref(iVar));
  let outputVal: IRNode;
  switch (op) {
    case "exp":
      outputVal = { id: -1, kind: "unary", op: "exp", input: inputVal, valueType: "scalar", dataType: "f32" };
      break;
    case "log":
      outputVal = { id: -1, kind: "unary", op: "log", input: inputVal, valueType: "scalar", dataType: "f32" };
      break;
    case "neg":
      outputVal = { id: -1, kind: "unary", op: "neg", input: inputVal, valueType: "scalar", dataType: "f32" };
      break;
  }

  result.push({
    kind: "forRange", varName: iVar, start: cU32(0), bound: cU32(size),
    body: [{
      kind: "indexAssign",
      arrayName: outputName,
      idx: ref(iVar),
      value: outputVal,
    }],
  });

  return result;
}

/**
 * Lower blockBinary → per-element binary operation with broadcasting.
 *
 * Broadcasting rules:
 * - Same shape: element-wise
 * - [R×C] op [1×1]: scalar broadcast to all elements
 * - [R×C] op [R×1]: per-row scalar broadcast across columns
 * - [R×C] op [1×C]: per-column scalar broadcast across rows
 */
function lowerBlockBinary(stmt: BlockBinaryStmt): Statement[] {
  const { aName, bName, outputName, aRows, aCols, bRows, bCols, op, inPlace, bScalarExpr } = stmt;
  const outRows = Math.max(aRows, bRows);
  const outCols = Math.max(aCols, bCols);
  const outSize = outRows * outCols;
  const result: Statement[] = [];

  // Allocate output if not in-place
  if (!inPlace) {
    result.push({
      kind: "varArray", name: outputName, elemType: "f32", size: outSize, skipZeroInit: true,
    });
  }

  // Determine B value accessor
  const isSameDims = aRows === bRows && aCols === bCols;
  const isScalarB = bRows === 1 && bCols === 1;
  const isPerRowB = bRows === aRows && bCols === 1;
  const isPerColB = bRows === 1 && bCols === aCols;

  // For scalar B (including bScalarExpr), pre-read the value
  if (isScalarB) {
    const bValName = freshVar("bv");
    const bVal: IRNode = bScalarExpr ?? arrayRead(bName, cU32(0));
    result.push({
      kind: "let", name: bValName, dtype: "f32", value: bVal,
    });

    // Flat loop over all elements
    const iVar = freshVar("bi");
    result.push({
      kind: "forRange", varName: iVar, start: cU32(0), bound: cU32(outSize),
      body: [{
        kind: "indexAssign",
        arrayName: outputName,
        idx: ref(iVar),
        value: emitBinaryOp(op, arrayRead(aName, ref(iVar)), ref(bValName, "f32")),
      }],
    });
  } else if (isSameDims) {
    // Same shape: element-wise
    const iVar = freshVar("bi");
    result.push({
      kind: "forRange", varName: iVar, start: cU32(0), bound: cU32(outSize),
      body: [{
        kind: "indexAssign",
        arrayName: outputName,
        idx: ref(iVar),
        value: emitBinaryOp(op, arrayRead(aName, ref(iVar)), arrayRead(bName, ref(iVar))),
      }],
    });
  } else if (isPerRowB) {
    // [R×C] op [R×1]: broadcast B per-row
    const rVar = freshVar("br");
    const cVar = freshVar("bc");
    const bvName = freshVar("bv");
    result.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(outRows),
      body: [
        { kind: "let", name: bvName, dtype: "f32", value: arrayRead(bName, ref(rVar)) },
        {
          kind: "forRange", varName: cVar, start: cU32(0), bound: cU32(outCols),
          body: [{
            kind: "indexAssign",
            arrayName: outputName,
            idx: binOp("add", binOp("mul", ref(rVar), cU32(outCols)), ref(cVar)),
            value: emitBinaryOp(op,
              arrayRead(aName, binOp("add", binOp("mul", ref(rVar), cU32(aCols)), ref(cVar))),
              ref(bvName, "f32"),
            ),
          }],
        },
      ],
    });
  } else if (isPerColB) {
    // [R×C] op [1×C]: broadcast B per-column
    const rVar = freshVar("br");
    const cVar = freshVar("bc");
    result.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(outRows),
      body: [{
        kind: "forRange", varName: cVar, start: cU32(0), bound: cU32(outCols),
        body: [{
          kind: "indexAssign",
          arrayName: outputName,
          idx: binOp("add", binOp("mul", ref(rVar), cU32(outCols)), ref(cVar)),
          value: emitBinaryOp(op,
            arrayRead(aName, binOp("add", binOp("mul", ref(rVar), cU32(aCols)), ref(cVar))),
            arrayRead(bName, ref(cVar)),
          ),
        }],
      }],
    });
  } else {
    throw new Error(
      `blockBinary: unsupported broadcast [${aRows}×${aCols}] op [${bRows}×${bCols}]`,
    );
  }

  return result;
}

/** Emit a binary operation IR node. */
function emitBinaryOp(op: string, lhs: IRNode, rhs: IRNode): IRNode {
  if (op === "copy") {
    return rhs; // Just use the RHS value
  }
  if (op === "max" || op === "min") {
    // max/min are WGSL built-in functions handled by the binary node emitter
    return { id: -1, kind: "binary", op, lhs, rhs, valueType: "scalar", dataType: "f32" } as IRNode;
  }
  return binOp(op as "add" | "sub" | "mul" | "div", lhs, rhs, "f32");
}

