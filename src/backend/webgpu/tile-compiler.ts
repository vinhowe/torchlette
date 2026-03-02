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

import { getSubgroupSupport } from "./matmul/types";
import { computeSafeVecWidth } from "./tile-access-analysis";
import type {
  IRNode,
  Statement,
  ThreadIdxNode,
  TileKernelSpec,
} from "./tile-ir";
import { buildKernelIR, elementwiseGrid, type KernelContext } from "./tile-ir";
import {
  autoDetectTPR,
  computeBlockLayouts,
  freshVar,
  getActiveTPR,
  hasTileStatements,
  lowerTileStatements,
  resetLoweringState,
  setBlockLayouts,
  setTPR,
} from "./tile-lowering";

/** Well-known uniform names that represent element counts for elementwise kernels. */
const ELEMENTWISE_UNIFORMS = new Set([
  "size",
  "total_elements",
  "num_elements",
  "outSize",
]);

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
 */
function exprFor(node: IRNode, bindings: BindingMap): string {
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
        return s.includes(".") || s.includes("e") || s.includes("E")
          ? s
          : s + ".0";
      }
      if (node.dataType === "f16") {
        const s = String(node.value);
        const numStr =
          s.includes(".") || s.includes("e") || s.includes("E") ? s : s + ".0";
        return `f16(${numStr})`;
      }
      if (node.dataType === "u32") return `${node.value}u`;
      return `i32(${node.value})`;
    }
    case "load": {
      const offs = exprFor(node.offsets, bindings);
      return `${node.binding}[${offs}]`;
    }
    case "binary": {
      const lhs = exprFor(node.lhs, bindings);
      const rhs = exprFor(node.rhs, bindings);
      switch (node.op) {
        case "add":
          return `(${lhs} + ${rhs})`;
        case "sub":
          return `(${lhs} - ${rhs})`;
        case "mul":
          return `(${lhs} * ${rhs})`;
        case "div":
          return `(${lhs} / ${rhs})`;
        case "mod":
          return `(${lhs} % ${rhs})`;
        case "and":
          return `(${lhs} & ${rhs})`;
        case "or":
          return `(${lhs} | ${rhs})`;
        case "xor":
          return `(${lhs} ^ ${rhs})`;
        case "shr":
          return `(${lhs} >> ${rhs})`;
        case "shl":
          return `(${lhs} << ${rhs})`;
        case "min":
          return `min(${lhs}, ${rhs})`;
        case "max":
          return `max(${lhs}, ${rhs})`;
        case "pow":
          return `pow(${lhs}, ${rhs})`;
      }
      break;
    }
    case "unary": {
      const input = exprFor(node.input, bindings);
      switch (node.op) {
        case "neg":
          return `(-${input})`;
        case "rsqrt":
          return `inverseSqrt(${input})`;
        case "tanh":
          return `tanh(${input})`;
        case "floor":
          return `floor(${input})`;
        case "ceil":
          return `ceil(${input})`;
        case "not":
          return `!(${input})`;
        default: {
          const fn = (
            {
              exp: "exp",
              log: "log",
              abs: "abs",
              sqrt: "sqrt",
              sin: "sin",
              cos: "cos",
              round: "round",
              sign: "sign",
              exp2: "exp2",
              log2: "log2",
            } as const
          )[node.op];
          return `${fn}(${input})`;
        }
      }
    }
    case "cast": {
      const input = exprFor(node.input, bindings);
      return `${node.targetType}(${input})`;
    }
    case "bitcast": {
      const input = exprFor(node.input, bindings);
      return `bitcast<${node.targetType}>(${input})`;
    }
    case "select": {
      const cond = exprFor(node.condition, bindings);
      const t = exprFor(node.trueVal, bindings);
      const f = exprFor(node.falseVal, bindings);
      return `select(${f}, ${t}, ${cond})`;
    }
    case "cmp": {
      const lhs = exprFor(node.lhs, bindings);
      const rhs = exprFor(node.rhs, bindings);
      const op = (
        { eq: "==", ne: "!=", lt: "<", le: "<=", gt: ">", ge: ">=" } as const
      )[node.op];
      return `(${lhs} ${op} ${rhs})`;
    }
    // -- Imperative mode nodes --
    case "threadIdx": {
      return `local_id.${(["x", "y", "z"] as const)[node.dim]}`;
    }
    case "localIndex": {
      return getActiveTPR() > 1 ? "_logical_idx" : "local_idx";
    }
    case "sharedRead":
    case "arrayRead":
    case "vec4ArrayRead":
    case "vec4SharedRead": {
      return `${node.arrayName}[${exprFor(node.idx, bindings)}]`;
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
      const val = exprFor(node.value, bindings);
      const mask = exprFor(node.mask, bindings);
      return `subgroupShuffleXor(${val}, ${mask})`;
    }
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd": {
      return `${node.kind}(${exprFor(node.value, bindings)})`;
    }
    case "vec4dot": {
      const a = node.a.map((n) => exprFor(n, bindings));
      const b = node.b.map((n) => exprFor(n, bindings));
      return `dot(vec4<f32>(${a.join(", ")}), vec4<f32>(${b.join(", ")}))`;
    }
    // -- Vec4 native nodes --
    case "vec4Construct": {
      const x = exprFor(node.x, bindings);
      const y = exprFor(node.y, bindings);
      const z = exprFor(node.z, bindings);
      const w = exprFor(node.w, bindings);
      return `vec4<f32>(${x}, ${y}, ${z}, ${w})`;
    }
    case "vec4Splat": {
      const v = exprFor(node.value, bindings);
      return `vec4<f32>(${v})`;
    }
    case "vec4NativeDot": {
      const a = exprFor(node.a, bindings);
      const b = exprFor(node.b, bindings);
      return `dot(${a}, ${b})`;
    }
    case "vec4Component": {
      const v = exprFor(node.value, bindings);
      const comp = ["x", "y", "z", "w"][node.comp];
      return `${v}.${comp}`;
    }
    case "vec4Binary": {
      const a = exprFor(node.a, bindings);
      const b = exprFor(node.b, bindings);
      const op = node.op === "add" ? "+" : node.op === "sub" ? "-" : "*";
      return `(${a} ${op} ${b})`;
    }
    default:
      throw new Error(`Unknown node kind: ${(node as { kind: string }).kind}`);
  }
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
  resetLoweringState();

  // 0. Auto vec-width selection: if vectorize is "auto", use access analysis.
  //    Works with both globalId(0) and flatGlobalId() kernels. The access
  //    analysis recognizes tagged flatGlobalId nodes as stride-1 coalesced.
  if (spec.vectorize === undefined && spec.autoVectorize) {
    const safeWidth = computeSafeVecWidth(spec);
    if (safeWidth > 1) {
      const wgSize =
        typeof spec.workgroupSize === "number"
          ? spec.workgroupSize
          : spec.workgroupSize[0] * spec.workgroupSize[1];
      const elementUniform = findElementwiseUniform(spec);
      spec = {
        ...spec,
        vectorize: safeWidth,
        ...(elementUniform
          ? {
              grid: elementwiseGrid(wgSize, {
                vecWidth: safeWidth,
                elementUniform,
              }),
            }
          : {}),
      };
    }
  }

  // 0b. Auto-detect subgroup support
  const sgSupport = getSubgroupSupport();
  const sgSize = spec.enableSubgroups
    ? (sgSupport?.subgroupSize ?? 32)
    : sgSupport?.supported
      ? (sgSupport.subgroupSize ?? 32)
      : 0;

  // 1. Build the IR DAG (includes constant folding + CSE in makeNode)
  const ctx = buildKernelIR(spec, sgSize);

  // Auto-enable subgroups if the kernel's reduction primitives used subgroup ops
  if (sgSize > 0 && !spec.enableSubgroups && ctx._usesSubgroups) {
    spec = { ...spec, enableSubgroups: true };
  }

  // 2. Lower tile-level ops if present
  let stmts = ctx.statements;
  if (hasTileStatements(stmts)) {
    const tpr = autoDetectTPR(stmts, sgSize > 0);
    setTPR(tpr);
    setBlockLayouts(computeBlockLayouts(stmts, tpr));
    if (tpr > 1) {
      spec = { ...spec, enableSubgroups: true };
    }

    // Auto-emit thread_row/thread_col if not present (tile lowering needs them)
    if (!stmts.some((s) => s.kind === "let" && s.name === "thread_row")) {
      const threadRowNode: ThreadIdxNode = {
        id: -1,
        kind: "threadIdx",
        dim: 1,
        valueType: "scalar",
        dataType: "u32",
      };
      const threadColNode: ThreadIdxNode = {
        id: -1,
        kind: "threadIdx",
        dim: 0,
        valueType: "scalar",
        dataType: "u32",
      };
      stmts = [
        { kind: "let", name: "thread_row", dtype: "u32", value: threadRowNode },
        { kind: "let", name: "thread_col", dtype: "u32", value: threadColNode },
        ...stmts,
      ];
    }
    stmts = lowerTileStatements(stmts, spec);
  }

  // 3. Automatic barrier insertion (opt-in, or forced when TPR > 1)
  if (spec.autoBarriers || getActiveTPR() > 1) {
    stmts = insertBarriers(stmts);
  }

  // 4. Auto-CSE: inject let bindings for multi-use expressions
  stmts = autoCSE(stmts);

  // 5. Loop-invariant code motion (always-on)
  stmts = hoistLoopInvariants(stmts);

  // 6. Dead code elimination (remove unused let bindings)
  stmts = eliminateDeadCode(stmts);

  try {
    return compileImperativeKernel(spec, ctx, stmts);
  } finally {
    resetLoweringState();
  }
}

// ============================================================================
// Imperative Compiler (new)
// ============================================================================

function compileImperativeKernel(
  spec: TileKernelSpec,
  ctx: KernelContext,
  overrideStatements?: Statement[],
): string {
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
  if (ctx.sharedArrays.length > 0 || ctx.vec4SharedArrays.length > 0)
    lines.push("");

  // Constants
  if (spec.constants) {
    for (const [name, value] of Object.entries(spec.constants)) {
      // Determine type from value: integers as u32, others as f32
      if (Number.isInteger(value) && value >= 0) {
        lines.push(`const ${name}: u32 = ${value}u;`);
      } else {
        const s = String(value);
        const numStr =
          s.includes(".") || s.includes("e") || s.includes("E") ? s : s + ".0";
        lines.push(`const ${name}: f32 = ${numStr};`);
      }
    }
    lines.push("");
  }

  // Function signature — physical WG size = logical × TPR
  const [logicalWgX, wgY] =
    typeof spec.workgroupSize === "number"
      ? [spec.workgroupSize, 1]
      : spec.workgroupSize;
  const wgX = logicalWgX * getActiveTPR();

  // Scan nodes to determine which builtins are actually used
  const hasFlatGidVec =
    (spec.vectorize ?? 0) > 1 && ctx.flatGlobalIdNodeIds.length > 0;
  const needsGid = ctx.nodes.some((n) => n.kind === "globalId");
  const needsWid =
    ctx.nodes.some((n) => n.kind === "programId") || hasFlatGidVec;
  const needsLocalId = ctx.nodes.some((n) => n.kind === "threadIdx");
  const needsLocalIdx =
    ctx.nodes.some((n) => n.kind === "localIndex") || hasFlatGidVec;
  const needsNumWg =
    ctx.nodes.some((n) => n.kind === "numWorkgroups") || hasFlatGidVec;
  // Shared arrays and tile-level stmts require local_id/local_idx even if not explicitly referenced
  const hasTileOps =
    ctx.sharedArrays.length > 0 ||
    ctx.vec4SharedArrays.length > 0 ||
    ctx.statements.some(
      (s) =>
        s.kind === "tileLoad" ||
        s.kind === "tileStore" ||
        s.kind === "tileLoad1d",
    );
  const emitWid = needsWid || hasTileOps;
  const emitLocalId = needsLocalId || hasTileOps;
  const emitLocalIdx = needsLocalIdx || hasTileOps;

  lines.push(`@compute @workgroup_size(${wgX}, ${wgY})`);
  lines.push(`fn main(`);
  const params: string[] = [];
  if (needsGid) params.push(`  @builtin(global_invocation_id) gid: vec3<u32>`);
  if (emitWid) params.push(`  @builtin(workgroup_id) wid: vec3<u32>`);
  if (emitLocalId)
    params.push(`  @builtin(local_invocation_id) local_id: vec3<u32>`);
  if (emitLocalIdx)
    params.push(`  @builtin(local_invocation_index) local_idx: u32`);
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
      const wgTotal =
        typeof spec.workgroupSize === "number"
          ? spec.workgroupSize
          : spec.workgroupSize[0] * spec.workgroupSize[1];
      lines.push(
        `  let _flatBase = (wid.x + wid.y * num_wg.x) * ${wgTotal * vecWidth}u + local_idx * ${vecWidth}u;`,
      );

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
      const gidXNodes = ctx.nodes.filter(
        (n) => n.kind === "globalId" && n.dim === 0,
      );
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
    // Emit TPR (threads-per-row) helper variables for subgroup cooperative mode
    if (getActiveTPR() > 1) {
      lines.push(`  let _logical_idx = local_idx / ${getActiveTPR()}u;`);
      lines.push(`  let _sub_idx = local_idx % ${getActiveTPR()}u;`);
    }
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
      const l = lhs.value,
        r = rhs.value;
      switch (op) {
        case "eq":
          return l === r;
        case "ne":
          return l !== r;
        case "lt":
          return l < r;
        case "le":
          return l <= r;
        case "gt":
          return l > r;
        case "ge":
          return l >= r;
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
      const val = exprFor(stmt.value, bindings);
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
      const val = exprFor(stmt.value, bindings);
      lines.push(`${indent}var ${stmt.name}: ${stmt.dtype} = ${val};`);
      break;
    }
    case "varArray": {
      lines.push(
        `${indent}var ${stmt.name}: array<${stmt.elemType}, ${stmt.size}>;`,
      );
      if (!stmt.skipZeroInit) {
        const zero =
          stmt.elemType === "f32"
            ? "0.0"
            : stmt.elemType === "f16"
              ? "f16(0.0)"
              : "0u";
        if (stmt.size <= 16) {
          // Unroll zero-init for small arrays
          for (let i = 0; i < stmt.size; i++) {
            lines.push(`${indent}${stmt.name}[${i}u] = ${zero};`);
          }
        } else {
          lines.push(
            `${indent}for (var _zi = 0u; _zi < ${stmt.size}u; _zi = _zi + 1u) {`,
          );
          lines.push(`${indent}  ${stmt.name}[_zi] = ${zero};`);
          lines.push(`${indent}}`);
        }
      }
      break;
    }
    case "assign": {
      const val = exprFor(stmt.value, bindings);
      lines.push(`${indent}${stmt.name} = ${val};`);
      break;
    }
    case "addAssign": {
      const val = exprFor(stmt.value, bindings);
      lines.push(`${indent}${stmt.name} = ${stmt.name} + ${val};`);
      break;
    }
    case "indexAssign":
    case "sharedWrite":
    case "vec4ArrayWrite": {
      const idx = exprFor(stmt.idx, bindings);
      const val = exprFor(stmt.value, bindings);
      lines.push(`${indent}${stmt.arrayName}[${idx}] = ${val};`);
      break;
    }
    case "indexAddAssign":
    case "vec4ArrayAddAssign": {
      const idx = exprFor(stmt.idx, bindings);
      const val = exprFor(stmt.value, bindings);
      lines.push(
        `${indent}${stmt.arrayName}[${idx}] = ${stmt.arrayName}[${idx}] + ${val};`,
      );
      break;
    }
    case "forRange": {
      // Check if we can unroll: both start and bound must be ConstNode
      const startConst = stmt.start.kind === "const" ? stmt.start.value : null;
      const boundConst = stmt.bound.kind === "const" ? stmt.bound.value : null;
      const tripCount =
        startConst !== null && boundConst !== null
          ? boundConst - startConst
          : null;
      const shouldUnroll =
        tripCount !== null &&
        tripCount >= 0 &&
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
        const start = exprFor(stmt.start, bindings);
        const bound = exprFor(stmt.bound, bindings);
        lines.push(
          `${indent}for (var ${stmt.varName} = ${start}; ${stmt.varName} < ${bound}; ${stmt.varName} = ${stmt.varName} + 1u) {`,
        );
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
        if (
          maxTrips >= 1 &&
          (stmt.unroll === true ? maxTrips <= 16 : maxTrips <= 8)
        ) {
          const startExpr = exprFor(stmt.start, bindings);
          for (let i = 0; i < maxTrips; i++) {
            const ivExpr =
              i === 0 ? startExpr : `(${startExpr} + ${i * strideVal}u)`;
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
        const start = exprFor(stmt.start, bindings);
        const bound = exprFor(stmt.bound, bindings);
        lines.push(
          `${indent}for (var ${stmt.varName} = ${start}; ${stmt.varName} < ${bound}; ${stmt.varName} = ${stmt.varName} + ${strideVal}u) {`,
        );
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
        const cond = exprFor(stmt.condition, bindings);
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
        const cond = exprFor(stmt.condition, bindings);
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
    // Vec4 array statements
    case "vec4VarArray": {
      lines.push(`${indent}var ${stmt.name}: array<vec4<f32>, ${stmt.size}>;`);
      break;
    }
    case "vec4SharedArray": {
      // Handled at module scope, not inside fn
      break;
    }
    case "directStore": {
      const idx = exprFor(stmt.idx, bindings);
      const val = exprFor(stmt.value, bindings);
      lines.push(`${indent}${stmt.binding}[${idx}] = ${val};`);
      break;
    }
    case "guardedStore": {
      if (isStaticFalse(stmt.condition)) {
        // Dead store — omit entirely
        break;
      }
      const idx = exprFor(stmt.idx, bindings);
      const val = exprFor(stmt.value, bindings);
      if (isStaticTrue(stmt.condition)) {
        // Constant-fold: emit unconditional store
        lines.push(`${indent}${stmt.binding}[${idx}] = ${val};`);
      } else {
        const cond = exprFor(stmt.condition, bindings);
        lines.push(`${indent}if (${cond}) {`);
        lines.push(`${indent}  ${stmt.binding}[${idx}] = ${val};`);
        lines.push(`${indent}}`);
      }
      break;
    }
    case "atomicOp": {
      const idx = exprFor(stmt.idx, bindings);
      const val = exprFor(stmt.value, bindings);
      const fnName: Record<string, string> = {
        max: "atomicMax",
        min: "atomicMin",
        add: "atomicAdd",
        or: "atomicOr",
        and: "atomicAnd",
        xor: "atomicXor",
        exchange: "atomicExchange",
      };
      lines.push(
        `${indent}${fnName[stmt.op]}(&${stmt.binding}[${idx}], ${val});`,
      );
      break;
    }
    case "atomicCAS": {
      const idx = exprFor(stmt.idx, bindings);
      const exp = exprFor(stmt.expected, bindings);
      const des = exprFor(stmt.desired, bindings);
      lines.push(
        `${indent}let _cas_result = atomicCompareExchangeWeak(&${stmt.binding}[${idx}], ${exp}, ${des});`,
      );
      lines.push(`${indent}let ${stmt.oldValueVar} = _cas_result.old_value;`);
      lines.push(
        `${indent}let ${stmt.exchangedVar} = select(0u, 1u, _cas_result.exchanged);`,
      );
      break;
    }
    case "return": {
      lines.push(`${indent}return;`);
      break;
    }
  }
}

// ============================================================================
// Generic Expression Tree Walkers
// ============================================================================

/**
 * Short-circuit visitor over child expressions of an IR node.
 * Returns true if fn returns true for any child (enables early exit).
 */
function someExprChild(node: IRNode, fn: (child: IRNode) => boolean): boolean {
  switch (node.kind) {
    case "binary":
    case "cmp":
      return fn(node.lhs) || fn(node.rhs);
    case "unary":
    case "cast":
    case "bitcast":
      return fn(node.input);
    case "select":
      return fn(node.condition) || fn(node.trueVal) || fn(node.falseVal);
    case "load":
      return fn(node.offsets) || (node.mask ? fn(node.mask) : false);
    case "sharedRead":
    case "arrayRead":
    case "vec4ArrayRead":
    case "vec4SharedRead":
      return fn(node.idx);
    case "subgroupShuffleXor":
      return fn(node.value) || fn(node.mask);
    case "subgroupAdd":
    case "subgroupMax":
    case "subgroupMin":
    case "subgroupBroadcastFirst":
    case "subgroupInclusiveAdd":
    case "vec4Splat":
    case "vec4Component":
      return fn(node.value);
    case "vec4Construct":
      return fn(node.x) || fn(node.y) || fn(node.z) || fn(node.w);
    case "vec4NativeDot":
    case "vec4Binary":
      return fn(node.a) || fn(node.b);
    case "vec4dot":
      return node.a.some(fn) || node.b.some(fn);
    default:
      return false; // leaves: programId, uniform, const, threadIdx, localIndex, globalId, namedRef, numPrograms
  }
}

/** Call fn on each child expression of an IR node. */
function forEachExprChild(node: IRNode, fn: (child: IRNode) => void): void {
  someExprChild(node, (child) => {
    fn(child);
    return false;
  });
}

/** Call fn on each IR expression field of a statement (not recursing into bodies). */
function forEachStmtExpr(stmt: Statement, fn: (expr: IRNode) => void): void {
  switch (stmt.kind) {
    case "let":
    case "var":
      fn(stmt.value);
      break;
    case "assign":
    case "addAssign":
      fn(stmt.value);
      break;
    case "indexAssign":
    case "indexAddAssign":
      fn(stmt.idx);
      fn(stmt.value);
      break;
    case "sharedWrite":
      fn(stmt.idx);
      fn(stmt.value);
      break;
    case "guardedStore":
      fn(stmt.condition);
      fn(stmt.idx);
      fn(stmt.value);
      break;
    case "directStore":
      fn(stmt.idx);
      fn(stmt.value);
      break;
    case "atomicOp":
      fn(stmt.idx);
      fn(stmt.value);
      break;
    case "atomicCAS":
      fn(stmt.idx);
      fn(stmt.expected);
      fn(stmt.desired);
      break;
    case "vec4ArrayWrite":
    case "vec4ArrayAddAssign":
      fn(stmt.idx);
      fn(stmt.value);
      break;
    case "forRange":
    case "forStride":
      fn(stmt.start);
      fn(stmt.bound);
      break;
    case "if":
    case "ifElse":
      fn(stmt.condition);
      break;
    // barrier, return, varArray, vec4VarArray, vec4SharedArray: no expressions
  }
}

// ============================================================================
// Dead Code Elimination
// ============================================================================

/** Collect all namedRef names referenced by an IR expression tree. */
function collectExprNames(node: IRNode, names: Set<string>): void {
  if (node.kind === "namedRef") names.add(node.name);
  forEachExprChild(node, (child) => collectExprNames(child, names));
}

/** Collect all namedRef names from all expressions in a statement (recursively into bodies). */
function collectAllStmtNames(stmt: Statement, names: Set<string>): void {
  forEachStmtExpr(stmt, (e) => collectExprNames(e, names));
  if (
    stmt.kind === "forRange" ||
    stmt.kind === "forStride" ||
    stmt.kind === "if"
  ) {
    for (const s of stmt.body) collectAllStmtNames(s, names);
  } else if (stmt.kind === "ifElse") {
    for (const s of stmt.body) collectAllStmtNames(s, names);
    for (const s of stmt.elseBody) collectAllStmtNames(s, names);
  }
}

// ============================================================================
// Automatic Barrier Insertion
// ============================================================================

/** Collect shared array names read by IR nodes within a statement. */
function getSharedReadsFromExpr(node: IRNode, names: Set<string>): void {
  if (node.kind === "sharedRead" || node.kind === "vec4SharedRead")
    names.add(node.arrayName);
  forEachExprChild(node, (child) => getSharedReadsFromExpr(child, names));
}

/** Get shared array names read by a statement's expressions (not recursing into forRange). */
function getSharedReadsFromStmt(stmt: Statement): Set<string> {
  const reads = new Set<string>();
  forEachStmtExpr(stmt, (e) => getSharedReadsFromExpr(e, reads));
  // Recurse into if/ifElse bodies but not forRange/forStride (separate scope for barrier insertion)
  if (stmt.kind === "if") {
    for (const s of stmt.body)
      for (const r of getSharedReadsFromStmt(s)) reads.add(r);
  } else if (stmt.kind === "ifElse") {
    for (const s of stmt.body)
      for (const r of getSharedReadsFromStmt(s)) reads.add(r);
    for (const s of stmt.elseBody)
      for (const r of getSharedReadsFromStmt(s)) reads.add(r);
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

/** Deeply collect all shared reads from statements, recursing into ALL bodies (including forRange/forStride). */
function collectAllSharedReads(stmts: Statement[], names: Set<string>): void {
  for (const stmt of stmts) {
    forEachStmtExpr(stmt, (e) => getSharedReadsFromExpr(e, names));
    if (stmt.kind === "if") {
      collectAllSharedReads(stmt.body, names);
    } else if (stmt.kind === "ifElse") {
      collectAllSharedReads(stmt.body, names);
      collectAllSharedReads(stmt.elseBody, names);
    } else if (stmt.kind === "forRange" || stmt.kind === "forStride") {
      collectAllSharedReads(stmt.body, names);
    }
  }
}

/** Deeply collect all shared writes from statements, recursing into ALL bodies (including forRange/forStride). */
function collectAllSharedWrites(stmts: Statement[], names: Set<string>): void {
  for (const stmt of stmts) {
    for (const w of getSharedWritesFromStmt(stmt)) names.add(w);
    if (stmt.kind === "forRange" || stmt.kind === "forStride") {
      collectAllSharedWrites(stmt.body, names);
    }
    // if/ifElse already handled by getSharedWritesFromStmt
  }
}

/**
 * Insert workgroupBarrier() between shared memory writes and subsequent reads.
 * Does not double-barrier: existing barriers clear the dirty set.
 */
export function insertBarriers(stmts: Statement[]): Statement[] {
  const result: Statement[] = [];
  const dirtyArrays = new Set<string>();

  for (const stmt of stmts) {
    // Recurse into forRange/forStride bodies
    if (stmt.kind === "forRange" || stmt.kind === "forStride") {
      // Check if this loop reads from dirty shared arrays (deep scan)
      const loopReads = new Set<string>();
      collectAllSharedReads(stmt.body, loopReads);
      let loopNeedsBarrier = false;
      for (const r of loopReads) {
        if (dirtyArrays.has(r)) {
          loopNeedsBarrier = true;
          break;
        }
      }
      if (loopNeedsBarrier) {
        result.push({ kind: "barrier" });
        dirtyArrays.clear();
      }

      const newBody = insertBarriersForLoop(stmt.body);
      result.push({ ...stmt, body: newBody } as Statement);
      // forRange may write shared memory — track it (deep scan)
      const loopWrites = new Set<string>();
      collectAllSharedWrites(stmt.body, loopWrites);
      for (const w of loopWrites) dirtyArrays.add(w);
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
  collectAllSharedWrites(body, allWrites);
  collectAllSharedReads(body, allReads);

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
          warnings.push(
            `Missing barrier: shared array '${r}' written then read without barrier`,
          );
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
      case "addAssign":
      case "var":
        names.add(stmt.name);
        break;
      case "indexAssign":
      case "indexAddAssign":
      case "sharedWrite":
      case "vec4ArrayWrite":
      case "vec4ArrayAddAssign":
        names.add(stmt.arrayName);
        break;
      case "atomicCAS":
        names.add(stmt.oldValueVar);
        names.add(stmt.exchangedVar);
        break;
      case "forRange":
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

/** Check if an IR expression depends on any of the given names (namedRef references). */
function exprDependsOn(node: IRNode, names: Set<string>): boolean {
  switch (node.kind) {
    case "namedRef":
      return names.has(node.name);
    case "load":
      return true; // global loads always variant
    case "sharedRead":
    case "vec4SharedRead":
      return (
        names.has(node.arrayName) ||
        someExprChild(node, (c) => exprDependsOn(c, names))
      );
    case "arrayRead":
    case "vec4ArrayRead":
      return (
        names.has(node.arrayName) ||
        someExprChild(node, (c) => exprDependsOn(c, names))
      );
    default:
      return someExprChild(node, (c) => exprDependsOn(c, names));
  }
}

/** Check if an expression contains any global memory load. */
function exprContainsLoad(node: IRNode): boolean {
  if (node.kind === "load") return true;
  return someExprChild(node, exprContainsLoad);
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

        if (
          s.kind === "let" &&
          !seenBarrier &&
          !exprDependsOn(s.value, variantNames) &&
          !exprContainsLoad(s.value)
        ) {
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
      result.push({
        ...stmt,
        body: hoistLoopInvariants(stmt.body),
      } as Statement);
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

// ============================================================================
// Dead Code Elimination
// ============================================================================

/** Map a function over all nested bodies (forRange, forStride, if, ifElse). */
function mapStmtBodies(
  stmts: Statement[],
  fn: (body: Statement[]) => Statement[],
): Statement[] {
  return stmts.map((s) => {
    switch (s.kind) {
      case "forRange":
      case "forStride":
      case "if":
        return { ...s, body: fn(s.body) } as Statement;
      case "ifElse":
        return {
          ...s,
          body: fn(s.body),
          elseBody: fn(s.elseBody),
        } as Statement;
      default:
        return s;
    }
  });
}

/**
 * Remove `let` statements whose bindings are never referenced.
 * Applies bottom-up: inner scopes are processed first, then outer scopes
 * propagate liveness backward through let chains.
 */
function eliminateDeadCode(stmts: Statement[]): Statement[] {
  // Step 1: Recurse into nested bodies (bottom-up)
  const processed = mapStmtBodies(stmts, eliminateDeadCode);

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
  return processed.filter((s) => {
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
  if (
    node.kind === "load" ||
    node.kind === "sharedRead" ||
    node.kind === "vec4SharedRead"
  )
    return true;
  return someExprChild(node, exprContainsMemoryRead);
}

/**
 * Walk an IR expression tree and collect non-trivial, memory-read-free candidate nodes.
 * Only considers nodes with valid IDs (>= 0, from makeNode).
 * Always recurses into children even for memory-read nodes, so pure sub-expressions
 * (like address calculations inside loads) can still be CSE'd.
 */
function collectExprCSECandidates(
  node: IRNode,
  nodes: Map<number, IRNode>,
): void {
  if (node.id < 0 || isTrivialNode(node)) return;
  if (!exprContainsMemoryRead(node)) nodes.set(node.id, node);
  forEachExprChild(node, (child) => collectExprCSECandidates(child, nodes));
}

/** Collect all CSE candidate node IDs from all expressions in a statement (recursing into all bodies). */
function collectStmtCSENodes(
  stmt: Statement,
  nodes: Map<number, IRNode>,
): void {
  forEachStmtExpr(stmt, (e) => collectExprCSECandidates(e, nodes));
  if (
    stmt.kind === "forRange" ||
    stmt.kind === "forStride" ||
    stmt.kind === "if"
  ) {
    for (const s of stmt.body) collectStmtCSENodes(s, nodes);
  } else if (stmt.kind === "ifElse") {
    for (const s of stmt.body) collectStmtCSENodes(s, nodes);
    for (const s of stmt.elseBody) collectStmtCSENodes(s, nodes);
  }
}

/**
 * Compute expression tree depth (0 for trivials, 1 for single op on trivials, etc.).
 */
function exprDepth(node: IRNode): number {
  if (isTrivialNode(node) || node.id < 0) return 0;
  switch (node.kind) {
    case "binary":
      return 1 + Math.max(exprDepth(node.lhs), exprDepth(node.rhs));
    case "unary":
    case "cast":
    case "bitcast":
      return 1 + exprDepth(node.input);
    case "cmp":
      return 1 + Math.max(exprDepth(node.lhs), exprDepth(node.rhs));
    case "select":
      return (
        1 +
        Math.max(
          exprDepth(node.condition),
          exprDepth(node.trueVal),
          exprDepth(node.falseVal),
        )
      );
    default:
      return 1;
  }
}

/** Collect all node IDs that are proper sub-expressions of a given node. */
function collectSubExprIds(node: IRNode, ids: Set<number>): void {
  if (node.id < 0 || isTrivialNode(node)) return;
  forEachExprChild(node, (child) => {
    ids.add(child.id);
    collectSubExprIds(child, ids);
  });
}

/**
 * Walk an IR expression tree and push every non-trivial, memory-read-free node
 * occurrence (NOT deduplicated). Same node appearing N times produces N entries.
 */
function countExprCSEOccurrences(
  node: IRNode,
  out: { id: number; node: IRNode }[],
): void {
  if (node.id < 0 || isTrivialNode(node)) return;
  if (!exprContainsMemoryRead(node)) out.push({ id: node.id, node });
  forEachExprChild(node, (child) => countExprCSEOccurrences(child, out));
}

/**
 * Count all CSE candidate occurrences from expressions in a single statement.
 * Unlike collectStmtCSENodes, does NOT deduplicate — same node appearing N
 * times produces N occurrences. Also does NOT recurse into child bodies
 * (forRange/forStride/if/ifElse) — those are handled by autoCSE's recursive
 * call into child scopes. Only counts expressions at this scope level (loop
 * bounds, conditions, and leaf statement expressions).
 */
/** Count CSE occurrences at this scope level only (no body recursion — handled by autoCSE). */
function countStmtCSEOccurrences(
  stmt: Statement,
  out: { id: number; node: IRNode }[],
): void {
  forEachStmtExpr(stmt, (e) => countExprCSEOccurrences(e, out));
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
export function autoCSE(
  stmts: Statement[],
  parentBoundIds: Set<number> = new Set(),
): Statement[] {
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
  const refCounts = new Map<
    number,
    { count: number; node: IRNode; firstIdx: number }
  >();
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
    if (
      count > 1 &&
      !letValueIds.has(id) &&
      !parentBoundIds.has(id) &&
      exprDepth(node) > 1
    ) {
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
  const toBind: { firstIdx: number; name: string; node: IRNode; id: number }[] =
    [];
  for (const c of candidates) {
    // Count how many parent candidate usages "cover" this candidate's references
    let parentCoverage = 0;
    for (const other of candidates) {
      if (other.id !== c.id && candidateSubExprs.get(other.id)?.has(c.id)) {
        parentCoverage += refCounts.get(other.id)?.count;
      }
    }
    // Keep if not a sub-expression of any candidate, or has external references
    if (parentCoverage === 0 || refCounts.get(c.id)?.count > parentCoverage) {
      toBind.push({
        firstIdx: c.firstIdx,
        name: freshVar("cse"),
        node: c.node,
        id: c.id,
      });
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
  const processed = mapStmtBodies(stmts, (body) =>
    autoCSE(body, childBoundIds),
  );

  if (toBind.length === 0) return processed;
  toBind.sort((a, b) => a.firstIdx - b.firstIdx);

  // Phase 4: Inject let bindings before first use
  const injByIdx = new Map<number, typeof toBind>();
  for (const b of toBind) {
    if (!injByIdx.has(b.firstIdx)) injByIdx.set(b.firstIdx, []);
    injByIdx.get(b.firstIdx)?.push(b);
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
        const bDepsA = subSets.get(b.id)?.has(a.id);
        const aDepsB = subSets.get(a.id)?.has(b.id);
        if (bDepsA && !aDepsB) return -1; // A before B
        if (aDepsB && !bDepsA) return 1; // B before A
        return 0; // no dependency — preserve order
      });
    }
  }

  const result: Statement[] = [];
  for (let i = 0; i < processed.length; i++) {
    const inj = injByIdx.get(i);
    if (inj) {
      for (const b of inj) {
        result.push({
          kind: "let",
          name: b.name,
          value: b.node,
          dtype: b.node.dataType,
        } as Statement);
      }
    }
    result.push(processed[i]);
  }

  return result;
}

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
    if (
      hasUniforms &&
      uniformIdx !== undefined &&
      bindingIndex === uniformIdx
    ) {
      lines.push(
        `@group(0) @binding(${bindingIndex}) var<uniform> config: TileConfig;`,
      );
      bindingIndex++;
    }
    const [name, binding] = entries[i];
    if (binding.storage === "atomic") {
      // Atomic bindings use array<atomic<T>>
      lines.push(
        `@group(0) @binding(${bindingIndex}) var<storage, read_write> ${name}: array<atomic<${binding.type}>>;`,
      );
    } else {
      const access = binding.storage === "read" ? "read" : "read_write";
      lines.push(
        `@group(0) @binding(${bindingIndex}) var<storage, ${access}> ${name}: array<${binding.type}>;`,
      );
    }
    bindingIndex++;
  }

  // If uniform wasn't inserted yet (no uniformBindingIndex or it's after all bindings)
  if (hasUniforms && (uniformIdx === undefined || bindingIndex <= uniformIdx)) {
    lines.push(
      `@group(0) @binding(${bindingIndex}) var<uniform> config: TileConfig;`,
    );
  }

  return lines;
}

// Tile lowering functions imported from ./tile-lowering.ts
