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
  TileLoadStmt, DotStmt, AccOpStmt, TileLoad1DStmt, TileStoreStmt,
  TilePtr2D, TileMask2D,
  BlockAllocStmt, BlockLoadStmt, BlockStoreStmt, BlockDotStmt,
  BlockReduceStmt, BlockUnaryStmt, BlockBinaryStmt,
} from "./tile-ir";
import { buildKernelIR, type KernelContext } from "./tile-ir";


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
          const fn = ({ exp: "exp", log: "log", abs: "abs", sqrt: "sqrt", sin: "sin", cos: "cos", round: "round", sign: "sign" } as const)[node.op];
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
    case "vec4dot": {
      const a = node.a.map(n => exprFor(n, bindings, loopVar));
      const b = node.b.map(n => exprFor(n, bindings, loopVar));
      return `dot(vec4<f32>(${a.join(", ")}), vec4<f32>(${b.join(", ")}))`;
    }
    case "arrayRead": {
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

  // 1. Build the IR DAG (includes constant folding + CSE in makeNode)
  const ctx = buildKernelIR(spec);

  // 2. Lower tile-level ops if present
  let stmts = ctx.statements;
  if (hasTileStatements(stmts)) {
    stmts = lowerTileStatements(stmts, spec);
  }

  // 3. Dead code elimination (remove unused let bindings)
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
      if (isStaticTrue(stmt.condition)) {
        for (const s of stmt.body) emitStatement(s, bindings, lines, depth);
      } else if (!isStaticFalse(stmt.condition)) {
        const cond = exprFor(stmt.condition, bindings, null);
        lines.push(`${indent}if (${cond}) {`);
        for (const s of stmt.body) {
          emitStatement(s, bindings, lines, depth + 1);
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
        for (const s of stmt.body) {
          emitStatement(s, bindings, lines, depth + 1);
        }
        lines.push(`${indent}} else {`);
        for (const s of stmt.elseBody) {
          emitStatement(s, bindings, lines, depth + 1);
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
    case "vec4dot":
      for (const n of node.a) collectExprNames(n, names);
      for (const n of node.b) collectExprNames(n, names);
      break;
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
    // barrier, return, varArray: no IRNode expressions to collect
  }
}

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

  // Step 4: Filter dead lets
  return processed.filter(s => s.kind !== "let" || usedNames.has(s.name));
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
      case "dot":
      case "accOp":
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
  let i = 0;
  while (i < stmts.length) {
    const s = stmts[i];

    // Detect fusible chain: accOp* (with optional tileLoad1d) → tileStore
    // Fuse into a single tm/tn loop to avoid multiple passes over the accumulator.
    if (s.kind === "accOp" || s.kind === "tileLoad1d") {
      const chain = collectAccStoreChain(stmts, i);
      if (chain) {
        result.push(...lowerFusedAccStore(chain, spec));
        i = chain.endIdx;
        continue;
      }
    }

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

/**
 * Scan forward from `startIdx` to collect a chain of accOp/tileLoad1d ending with a tileStore
 * on the same accumulator. Returns null if no fusible chain is found.
 */
function collectAccStoreChain(
  stmts: Statement[],
  startIdx: number,
): { accOps: AccOpStmt[]; load1ds: TileLoad1DStmt[]; store: TileStoreStmt; endIdx: number } | null {
  let accName: string | null = null;
  const accOps: AccOpStmt[] = [];
  const load1ds: TileLoad1DStmt[] = [];

  let i = startIdx;
  while (i < stmts.length) {
    const s = stmts[i];
    if (s.kind === "accOp") {
      if (accName === null) accName = s.accName;
      if (s.accName !== accName) break;
      accOps.push(s);
      i++;
    } else if (s.kind === "tileLoad1d") {
      load1ds.push(s);
      i++;
    } else if (s.kind === "tileStore") {
      if (accName !== null && s.accName === accName && accOps.length > 0) {
        return { accOps, load1ds, store: s, endIdx: i + 1 };
      }
      break;
    } else {
      break;
    }
  }

  return null;
}

/**
 * Fuse accOp chain + tileStore into a single tm/tn loop.
 *
 * Instead of generating separate loops for each accOp and the store,
 * emit one loop that reads acc[idx], applies all transformations inline,
 * and writes to global memory. Matches production codegen output.
 */
function lowerFusedAccStore(
  chain: { accOps: AccOpStmt[]; load1ds: TileLoad1DStmt[]; store: TileStoreStmt },
  spec: TileKernelSpec,
): Statement[] {
  const result: Statement[] = [];

  // Emit tileLoad1d first (register loads for bias etc. — loop-invariant per row)
  for (const ld of chain.load1ds) {
    result.push(...lowerTileLoad1D(ld));
  }

  const { binding, ptr, mask, accName, threadTileM, threadTileN, accDtype } = chain.store;
  const tmVar = freshVar("tm");
  const tnVar = freshVar("tn");

  const innerBody: Statement[] = [];

  // Row/col computation
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

  const ifBody: Statement[] = [];
  const idxName = freshVar("st_idx");
  ifBody.push({
    kind: "let", name: idxName, dtype: "u32",
    value: binOp("add",
      binOp("add", ptr.baseOffset, mulOrSkip(ref(rowName), ptr.outerStride)),
      mulOrSkip(ref(colName), ptr.innerStride),
    ),
  });

  // Start with raw acc value
  const accIdx = binOp("add", binOp("mul", ref(tmVar), cU32(threadTileN)), ref(tnVar));
  let currentVal: IRNode = arrayRead(accName, accIdx);

  // Apply all accOps inline
  for (const op of chain.accOps) {
    switch (op.op.kind) {
      case "mulScalar":
        currentVal = binOp("mul", currentVal, op.op.value, "f32");
        break;
      case "addRow":
        currentVal = binOp("add", currentVal, arrayRead(op.op.valuesArray, ref(tnVar)), "f32");
        break;
      case "apply": {
        // Bind current value to the apply's input variable name, emit body, use result
        const valName = freshVar("fv");
        ifBody.push({ kind: "let", name: valName, dtype: "f32", value: currentVal });
        // The apply body references op.op.valName — we need to let-bind it
        ifBody.push({ kind: "let", name: op.op.valName, dtype: "f32", value: ref(valName, "f32") });
        ifBody.push(...op.op.body);
        currentVal = op.op.resultNode;
        break;
      }
      case "castTo":
        currentVal = castNode(currentVal, op.op.dtype);
        break;
    }
  }

  // Apply store's accDtype cast if not already cast by an accOp
  if (accDtype && accDtype !== "f32") {
    const lastOp = chain.accOps[chain.accOps.length - 1];
    if (!lastOp || lastOp.op.kind !== "castTo") {
      currentVal = castNode(currentVal, accDtype);
    }
  }

  ifBody.push({
    kind: "indexAssign",
    arrayName: binding,
    idx: ref(idxName),
    value: currentVal,
  });

  innerBody.push({ kind: "if", condition: maskCond, body: ifBody });

  result.push({
    kind: "forRange", varName: tmVar, start: cU32(0), bound: cU32(threadTileM),
    body: [{
      kind: "forRange", varName: tnVar, start: cU32(0), bound: cU32(threadTileN),
      body: innerBody,
    }],
  });

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
  // Delegate to existing tile load lowering
  return lowerTileLoad({
    kind: "tileLoad",
    binding: stmt.binding,
    ptr: stmt.tilePtr,
    mask: stmt.tileMask,
    sharedName: stmt.name,
    tileRows: stmt.rows,
    tileCols: stmt.cols,
    elemType: stmt.elemType,
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
 * Lowering mirrors the existing lowerDot() (tile-compiler.ts DotStmt handler):
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

  const innerDim = aCols;         // K dimension (A cols = B rows for non-transposed)
  const bColStride = bCols;       // B's column count for indexing
  const isAccumulate = !!accName; // dotAccum vs dot

  const result: Statement[] = [];

  // barrier() before reading shared memory
  result.push({ kind: "barrier" });

  const kkVar = freshVar("kk");
  const kkLoop: Statement[] = [];

  // Load a_vals from shared A: a_vals[tm] = A[(thread_row*ttM + tm) * innerDim + kk]
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
            cU32(innerDim),
          ),
          ref(kkVar),
        ),
      ),
    }],
  });

  // Load b_vals from shared B: b_vals[tn] = B[kk * bCols + thread_col*ttN + tn]
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
          binOp("mul", ref(kkVar), cU32(bColStride)),
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
      // Vec4 path: s += dot(vec4(a[d4*4..+3]), vec4(b[j*bCols+d4*4..+3]))
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
                binOp("add", binOp("mul", ref(jVar), cU32(bCols)), binOp("mul", ref(d4Var), cU32(4))),
                cU32(k),
              )),
            ) as [IRNode, IRNode, IRNode, IRNode],
          ),
        }],
      });
    } else {
      // Scalar path: for d in 0..innerDim: s += a[d] * b_smem[j*bCols + d]
      const dVar = freshVar("d");
      jBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(innerDim),
        body: [{
          kind: "addAssign",
          name: sVar,
          value: binOp("mul",
            arrayRead(aName, ref(dVar)),
            sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bCols)), ref(dVar))),
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
                binOp("add", binOp("mul", ref(jVar), cU32(bCols)), binOp("mul", ref(d4Var), cU32(4))),
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
            sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bCols)), ref(dVar))),
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
      // Vec4 path: for d4 in 0..outCols/4: result[d4*4+k] += p * b_smem[j*bCols+d4*4+k]
      const d4Var = freshVar("d4");
      const d4Body: Statement[] = [];
      for (let k = 0; k < 4; k++) {
        const regIdx = binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k));
        const smemIdx = binOp("add",
          binOp("add", binOp("mul", ref(jVar), cU32(bCols)), binOp("mul", ref(d4Var), cU32(4))),
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
      // Scalar path: for d in 0..bCols: result[d] += p * b_smem[j*bCols + d]
      const dVar = freshVar("d");
      jBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(outCols),
        body: [{
          kind: "indexAddAssign",
          arrayName: resultName,
          idx: ref(dVar),
          value: binOp("mul",
            ref(pVar, "f32"),
            sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bCols)), ref(dVar))),
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
          sharedRead(bName, binOp("add", binOp("mul", ref(jVar), cU32(bCols)), ref(dVar))),
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
    const initVal = op === "max" ? cF32(-3.402823e+38) : cF32(0);

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
    const initVal = op === "max" ? cF32(-3.402823e+38) : cF32(0);

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

