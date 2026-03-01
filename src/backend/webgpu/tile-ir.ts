/**
 * Tile IR: Triton-Style Block-Level Kernel DSL
 *
 * Provides a symbolic expression builder for writing WebGPU compute kernels
 * at the workgroup (block) level. Single imperative compilation model.
 *
 * The developer writes explicit statements (forRange, barrier, sharedArray,
 * emitStore, emitVar, treeReduceSum, etc.) for full control over compute
 * kernel algorithms. Expressions are symbolic and inlined during codegen.
 *
 * Also supports Block-ops tile statements (blockLoad, blockStore, dot, etc.)
 * for 2D tiled algorithms like matmul, which are lowered to scalar/vec4 WGSL.
 *
 * Usage:
 * ```typescript
 * const matmul: TileKernelSpec = {
 *   workgroupSize: [8, 8],
 *   kernel(ctx) {
 *     const tileA = ctx.sharedArray("tileA", 32 * 16);
 *     ctx.forRange(ctx.const(0, "u32"), numKTiles, (kTile) => {
 *       // cooperative load, barrier, compute
 *       ctx.barrier();
 *     });
 *   },
 * };
 * ```
 */

import {
  BlockOps, Block, TileRange, buildPtr, buildMask,
  type BlockPtr, type BlockThreadPtr, type BlockCoopPtr,
  type BlockLoadOpts, type BlockStorePtr,
} from "./tile-ops";

// Re-export tile-ops types so kernel authors only need one import
export {
  BlockOps, Block, TileRange, buildPtr, buildMask,
  type BlockPtr, type BlockThreadPtr, type BlockCoopPtr,
  type BlockLoadOpts, type BlockStorePtr,
} from "./tile-ops";
import { F32_NEG_MAX, F32_POS_MAX, MAX_WORKGROUPS_PER_DIM } from "./shape-utils";

// ============================================================================
// IR Node Types
// ============================================================================

type ValueType = "block" | "scalar";
export type DataType = "f32" | "f16" | "u32" | "i32";
type BinaryOp = "add" | "sub" | "mul" | "div" | "mod" | "and" | "or" | "xor" | "shr" | "shl" | "min" | "max" | "pow";
type UnaryOp = "rsqrt" | "exp" | "log" | "abs" | "neg" | "sqrt" | "tanh" | "floor" | "ceil" | "not" | "sin" | "cos" | "round" | "sign" | "exp2" | "log2";
type CmpOp = "eq" | "ne" | "lt" | "le" | "gt" | "ge";

export interface IRNodeBase {
  id: number;
  valueType: ValueType;
  dataType: DataType;
}

interface ProgramIdNode extends IRNodeBase {
  kind: "programId";
  dim: number; // 0=x, 1=y, 2=z
}

interface UniformNode extends IRNodeBase {
  kind: "uniform";
  name: string;
}

interface ConstNode extends IRNodeBase {
  kind: "const";
  value: number;
}

interface LoadNode extends IRNodeBase {
  kind: "load";
  binding: string;
  offsets: IRNode; // block expression for indices
  mask?: IRNode;   // optional mask (block expression, truthy = load, falsy = 0)
}


interface BinaryNode extends IRNodeBase {
  kind: "binary";
  op: BinaryOp;
  lhs: IRNode;
  rhs: IRNode;
}

interface UnaryNode extends IRNodeBase {
  kind: "unary";
  op: UnaryOp;
  input: IRNode;
}

interface CastNode extends IRNodeBase {
  kind: "cast";
  input: IRNode;
  targetType: DataType;
}

interface SelectNode extends IRNodeBase {
  kind: "select";
  condition: IRNode;
  trueVal: IRNode;
  falseVal: IRNode;
}

interface CmpNode extends IRNodeBase {
  kind: "cmp";
  op: CmpOp;
  lhs: IRNode;
  rhs: IRNode;
}

interface BitcastNode extends IRNodeBase {
  kind: "bitcast";
  input: IRNode;
  targetType: DataType;
}

// -- Imperative mode nodes --

export interface ThreadIdxNode extends IRNodeBase {
  kind: "threadIdx";
  dim: number; // 0=x, 1=y, 2=z
}

interface LocalIndexNode extends IRNodeBase {
  kind: "localIndex";
}

interface SharedReadNode extends IRNodeBase {
  kind: "sharedRead";
  arrayName: string;
  idx: IRNode;
}

interface NamedRefNode extends IRNodeBase {
  kind: "namedRef";
  name: string;
}

interface ArrayReadNode extends IRNodeBase {
  kind: "arrayRead";
  arrayName: string;
  idx: IRNode;
}

interface GlobalIdNode extends IRNodeBase {
  kind: "globalId";
  dim: number; // 0=x, 1=y, 2=z
}

/** Number of workgroups dispatched in a given dimension (like Triton's `tl.num_programs`). */
interface NumWorkgroupsNode extends IRNodeBase {
  kind: "numWorkgroups";
  dim: number; // 0=x, 1=y, 2=z
}

interface SubgroupShuffleXorNode extends IRNodeBase {
  kind: "subgroupShuffleXor";
  value: IRNode;
  mask: IRNode;
}

interface SubgroupAddNode extends IRNodeBase {
  kind: "subgroupAdd";
  value: IRNode;
}

interface SubgroupMaxNode extends IRNodeBase {
  kind: "subgroupMax";
  value: IRNode;
}

interface SubgroupMinNode extends IRNodeBase {
  kind: "subgroupMin";
  value: IRNode;
}

interface SubgroupBroadcastFirstNode extends IRNodeBase {
  kind: "subgroupBroadcastFirst";
  value: IRNode;
}

interface SubgroupInclusiveAddNode extends IRNodeBase {
  kind: "subgroupInclusiveAdd";
  value: IRNode;
}

/** dot(vec4<f32>(a0,a1,a2,a3), vec4<f32>(b0,b1,b2,b3)) → f32 scalar */
interface Vec4DotNode extends IRNodeBase {
  kind: "vec4dot";
  a: [IRNode, IRNode, IRNode, IRNode];
  b: [IRNode, IRNode, IRNode, IRNode];
}

// -- Native vec4 nodes (for attention-style subgroup cooperative kernels) --

/** vec4<f32>(x, y, z, w) — construct from 4 scalars */
interface Vec4ConstructNode extends IRNodeBase {
  kind: "vec4Construct";
  x: IRNode;
  y: IRNode;
  z: IRNode;
  w: IRNode;
}

/** vec4<f32>(v) — splat scalar to all 4 lanes */
interface Vec4SplatNode extends IRNodeBase {
  kind: "vec4Splat";
  value: IRNode;
}

/** dot(a, b) where a and b are vec4<f32> — returns f32 */
interface Vec4NativeDotNode extends IRNodeBase {
  kind: "vec4NativeDot";
  a: IRNode;
  b: IRNode;
}

/** v.x, v.y, v.z, v.w — extract component from vec4<f32> */
interface Vec4ComponentNode extends IRNodeBase {
  kind: "vec4Component";
  value: IRNode;
  comp: 0 | 1 | 2 | 3;
}

/** vec4 binary: add, sub, mul between two vec4s, or vec4 * scalar */
interface Vec4BinaryNode extends IRNodeBase {
  kind: "vec4Binary";
  op: "add" | "sub" | "mul";
  a: IRNode;
  b: IRNode;
}

/** Read from a vec4 array: name[index] → vec4<f32> */
interface Vec4ArrayReadNode extends IRNodeBase {
  kind: "vec4ArrayRead";
  arrayName: string;
  idx: IRNode;
}

/** Read from a vec4 shared array: name[index] → vec4<f32> */
interface Vec4SharedReadNode extends IRNodeBase {
  kind: "vec4SharedRead";
  arrayName: string;
  idx: IRNode;
}

export type IRNode =
  | ProgramIdNode
  | UniformNode
  | ConstNode
  | LoadNode
  | BinaryNode
  | UnaryNode
  | CastNode
  | SelectNode
  | CmpNode
  | BitcastNode
  | ThreadIdxNode
  | LocalIndexNode
  | SharedReadNode
  | NamedRefNode
  | ArrayReadNode
  | GlobalIdNode
  | SubgroupShuffleXorNode
  | SubgroupAddNode
  | SubgroupMaxNode
  | SubgroupMinNode
  | SubgroupBroadcastFirstNode
  | SubgroupInclusiveAddNode
  | Vec4DotNode
  | NumWorkgroupsNode
  | Vec4ConstructNode
  | Vec4SplatNode
  | Vec4NativeDotNode
  | Vec4ComponentNode
  | Vec4BinaryNode
  | Vec4ArrayReadNode
  | Vec4SharedReadNode;

// ============================================================================
// Tile-Level IR Types (block-level, compiler-lowered)
// ============================================================================

/** 1D range of tile offsets: [base, base+1, ..., base+size-1] */
export interface TileRangeInfo {
  base: IRNode;
  size: number;         // compile-time block dimension
}

/** 2D pointer block: base + outer * outerStride + inner * innerStride */
export interface TilePtr2D {
  baseOffset: IRNode;
  outerRange: TileRangeInfo;
  outerStride: IRNode;
  innerRange: TileRangeInfo;
  innerStride: IRNode;  // const(1) for contiguous
}

/** 2D mask: outer_cond & inner_cond */
export interface TileMask2D {
  outerRange: TileRangeInfo;
  outerBound: IRNode;
  innerRange: TileRangeInfo;
  innerBound: IRNode;
}

// --- Tile-level statement types ---

export interface TileLoadStmt {
  kind: "tileLoad";
  binding: string;
  ptr: TilePtr2D;
  mask: TileMask2D;
  sharedName: string;     // compiler-assigned
  tileRows: number;       // BLOCK dimension (not thread tile)
  tileCols: number;
  elemType: DataType;
}

export interface TileLoad1DStmt {
  kind: "tileLoad1d";
  binding: string;
  range: TileRangeInfo;
  arrayName: string;
  size: number;           // = threadTileN (derived by compiler)
}

export interface TileStoreStmt {
  kind: "tileStore";
  binding: string;
  ptr: TilePtr2D;
  mask: TileMask2D;
  accName: string;
  threadTileM: number;
  threadTileN: number;
  accDtype?: DataType;    // set by castTo_, affects store type conversion
}

// --- Block-level statement types (unified Block API) ---

export type BlockBinaryOp = "add" | "sub" | "mul" | "div" | "max" | "copy";
export type BlockUnaryOp = "exp" | "log" | "neg";
export type BlockReduceOp = "max" | "sum";

export interface BlockAllocStmt {
  kind: "blockAlloc";
  name: string;
  rows: number;
  cols: number;
  elemType: DataType;
  initValue?: number;  // undefined = zero-init, number = fill with this value
}

export interface BlockLoadStmt {
  kind: "blockLoad";
  binding: string;
  name: string;
  rows: number;
  cols: number;
  elemType: DataType;
  ptrKind: "thread" | "tile";
  // Thread ptr fields
  threadBase?: IRNode;
  threadStride?: IRNode;
  // Tile ptr fields (reuses existing types)
  tilePtr?: TilePtr2D;
  tileMask?: TileMask2D;
  // Thread-level guard (for thread ptr bounds checking)
  guard?: IRNode;
}

export interface BlockStoreStmt {
  kind: "blockStore";
  binding: string;
  blockName: string;
  rows: number;
  cols: number;
  base: IRNode;
  stride: IRNode;
  guard?: IRNode;
}

export interface BlockDotStmt {
  kind: "blockDot";
  aName: string;
  bName: string;
  resultName: string;
  accName?: string;       // if set, accumulate into existing block
  aPlacement: "register" | "shared";
  bPlacement: "register" | "shared";
  bTransposed: boolean;
  aRows: number;
  aCols: number;
  bRows: number;          // ORIGINAL (untransposed) rows
  bCols: number;          // ORIGINAL (untransposed) cols
  threadTileM?: number;   // per-thread M dim (for shared×shared outer product)
  threadTileN?: number;   // per-thread N dim (for shared×shared outer product)
  aSmemStride?: number;   // stride for A in shared memory (default: aCols)
  bSmemStride?: number;   // stride for B in shared memory (default: bCols)
}

export interface BlockReduceStmt {
  kind: "blockReduce";
  inputName: string;
  outputName: string;
  inputRows: number;
  inputCols: number;
  axis: number;
  op: BlockReduceOp;
}

export interface BlockUnaryStmt {
  kind: "blockUnary";
  inputName: string;
  outputName: string;
  rows: number;
  cols: number;
  op: BlockUnaryOp;
  inPlace: boolean;
}

export interface BlockBinaryStmt {
  kind: "blockBinary";
  aName: string;
  bName: string;
  outputName: string;
  aRows: number;
  aCols: number;
  bRows: number;
  bCols: number;
  op: BlockBinaryOp;
  inPlace: boolean;
  bScalarExpr?: IRNode;   // when set, B is a scalar expression (bName unused)
}

// ============================================================================
// Statement Types (imperative mode)
// ============================================================================

export type Statement =
  | LetStmt
  | VarStmt
  | VarArrayStmt
  | AssignStmt
  | AddAssignStmt
  | IndexAssignStmt
  | IndexAddAssignStmt
  | ForRangeStmt
  | ForStrideStmt
  | IfStmt
  | IfElseStmt
  | BarrierStmt
  | SharedWriteStmt
  | GuardedStoreStmt
  | DirectStoreStmt
  | AtomicOpStmt
  | AtomicCASStmt
  // Vec4 array statements
  | Vec4VarArrayStmt
  | Vec4SharedArrayStmt
  | Vec4ArrayWriteStmt
  | Vec4ArrayAddAssignStmt
  // Tile-level statements (lowered by tile compiler)
  | TileLoadStmt
  | TileLoad1DStmt
  | TileStoreStmt
  // Block-level statements (unified Block API, lowered by tile compiler)
  | BlockAllocStmt
  | BlockLoadStmt
  | BlockStoreStmt
  | BlockDotStmt
  | BlockReduceStmt
  | BlockUnaryStmt
  | BlockBinaryStmt
  | ReturnStmt;

interface ReturnStmt {
  kind: "return";
}

interface LetStmt {
  kind: "let";
  name: string;
  value: IRNode;
  dtype: DataType;
}

interface VarStmt {
  kind: "var";
  name: string;
  value: IRNode;
  dtype: DataType;
}

interface VarArrayStmt {
  kind: "varArray";
  name: string;
  elemType: DataType;
  size: number;
  skipZeroInit?: boolean;
}

interface AssignStmt {
  kind: "assign";
  name: string;
  value: IRNode;
}

interface AddAssignStmt {
  kind: "addAssign";
  name: string;
  value: IRNode;
}

interface IndexAssignStmt {
  kind: "indexAssign";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
}

interface IndexAddAssignStmt {
  kind: "indexAddAssign";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
}

interface ForRangeStmt {
  kind: "forRange";
  varName: string;
  start: IRNode;
  bound: IRNode;
  body: Statement[];
  /** When true, the compiler unrolls this loop (requires const bounds). */
  unroll?: boolean;
}

/** Strided for loop: `for (var i = start; i < bound; i += stride)`. */
interface ForStrideStmt {
  kind: "forStride";
  varName: string;
  start: IRNode;
  bound: IRNode;
  stride: number;
  body: Statement[];
  /** When true, the compiler unrolls this loop (requires const start/bound). */
  unroll?: boolean;
}

interface IfStmt {
  kind: "if";
  condition: IRNode;
  body: Statement[];
}

interface IfElseStmt {
  kind: "ifElse";
  condition: IRNode;
  body: Statement[];
  elseBody: Statement[];
}

interface BarrierStmt {
  kind: "barrier";
}

interface SharedWriteStmt {
  kind: "sharedWrite";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
}

interface GuardedStoreStmt {
  kind: "guardedStore";
  binding: string;
  condition: IRNode;
  idx: IRNode;
  value: IRNode;
}

interface DirectStoreStmt {
  kind: "directStore";
  binding: string;
  idx: IRNode;
  value: IRNode;
}

type AtomicOp = "max" | "min" | "add" | "or" | "and" | "xor" | "exchange";

interface AtomicOpStmt {
  kind: "atomicOp";
  binding: string;
  idx: IRNode;
  op: AtomicOp;
  value: IRNode;
}

interface AtomicCASStmt {
  kind: "atomicCAS";
  binding: string;
  idx: IRNode;
  expected: IRNode;
  desired: IRNode;
  /** Variable name to store the old value. */
  oldValueVar: string;
  /** Variable name to store the exchanged flag (bool). */
  exchangedVar: string;
}

// -- Vec4 array statement types --

/** var name: array<vec4<f32>, size>; (register-space vec4 array) */
interface Vec4VarArrayStmt {
  kind: "vec4VarArray";
  name: string;
  size: number;
}

/** var<workgroup> name: array<vec4<f32>, size>; */
interface Vec4SharedArrayStmt {
  kind: "vec4SharedArray";
  name: string;
  size: number;
}

/** name[index] = value; where value is vec4<f32> */
interface Vec4ArrayWriteStmt {
  kind: "vec4ArrayWrite";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
  isShared: boolean;
}

/** name[index] += value; where value is vec4<f32> */
interface Vec4ArrayAddAssignStmt {
  kind: "vec4ArrayAddAssign";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
  isShared: boolean;
}

// ============================================================================
// Symbolic Expression Wrapper
// ============================================================================

let nextNodeId = 0;

// ---- CSE cache (cleared per kernel compilation) ----
let cseCache = new Map<string, IRNode>();

// ---- Constant folding helpers ----

function isConstVal(node: IRNode, val: number): boolean {
  return node.kind === "const" && node.value === val;
}

function evalBinaryConst(op: BinaryOp, a: number, b: number): number | null {
  switch (op) {
    case "add": { const r = a + b; return isFinite(r) ? r : null; }
    case "sub": { const r = a - b; return isFinite(r) ? r : null; }
    case "mul": { const r = a * b; return isFinite(r) ? r : null; }
    case "div": return b !== 0 && isFinite(a / b) ? a / b : null;
    case "mod": return b !== 0 ? a % b : null;
    case "min": return Math.min(a, b);
    case "max": return Math.max(a, b);
    case "pow": { const r = Math.pow(a, b); return isFinite(r) ? r : null; }
    case "and": return ((a >>> 0) & (b >>> 0)) >>> 0;
    case "or":  return ((a >>> 0) | (b >>> 0)) >>> 0;
    case "xor": return ((a >>> 0) ^ (b >>> 0)) >>> 0;
    case "shr": return (a >>> 0) >>> (b >>> 0);
    case "shl": return ((a >>> 0) << (b >>> 0)) >>> 0;
    default: return null;
  }
}

function evalUnaryConst(op: UnaryOp, x: number): number | null {
  switch (op) {
    case "neg":   return -x;
    case "abs":   return Math.abs(x);
    case "exp":   { const r = Math.exp(x); return isFinite(r) ? r : null; }
    case "log":   return x > 0 ? Math.log(x) : null;
    case "sqrt":  return x >= 0 ? Math.sqrt(x) : null;
    case "rsqrt": return x > 0 ? 1 / Math.sqrt(x) : null;
    case "floor": return Math.floor(x);
    case "ceil":  return Math.ceil(x);
    case "sin":   return Math.sin(x);
    case "cos":   return Math.cos(x);
    case "tanh":  return Math.tanh(x);
    case "round": return Math.round(x);
    case "sign":  return Math.sign(x);
    case "not":   return x ? 0 : 1;
    case "exp2":  { const r = Math.pow(2, x); return isFinite(r) ? r : null; }
    case "log2":  return x > 0 ? Math.log2(x) : null;
    default: return null;
  }
}

/**
 * Try to fold a node at construction time. Returns:
 * - An existing IRNode (id >= 0) for algebraic simplification (reuses operand)
 * - A new ConstNode with id = -1 for constant folding (caller assigns real id)
 * - null if no optimization applies
 */
function tryFold(partial: Omit<IRNode, "id">): IRNode | null {
  if (partial.kind === "binary") {
    const { op, lhs, rhs, dataType } = partial as Omit<BinaryNode, "id">;
    // Constant folding: both operands are const
    if (lhs.kind === "const" && rhs.kind === "const") {
      const result = evalBinaryConst(op, lhs.value, rhs.value);
      if (result !== null) {
        return { kind: "const", value: result, valueType: "scalar", dataType, id: -1 } as ConstNode;
      }
    }
    // Algebraic simplifications
    switch (op) {
      case "add":
        if (isConstVal(rhs, 0)) return lhs;
        if (isConstVal(lhs, 0)) return rhs;
        break;
      case "sub":
        if (isConstVal(rhs, 0)) return lhs;
        break;
      case "mul":
        if (isConstVal(rhs, 1)) return lhs;
        if (isConstVal(lhs, 1)) return rhs;
        break;
      case "div":
        if (isConstVal(rhs, 1)) return lhs;
        break;
      case "and":
        if (isConstVal(rhs, 0) || isConstVal(lhs, 0))
          return { kind: "const", value: 0, valueType: "scalar", dataType: "u32", id: -1 } as ConstNode;
        break;
      case "or":
        if (isConstVal(rhs, 0)) return lhs;
        if (isConstVal(lhs, 0)) return rhs;
        break;
      case "xor":
        if (isConstVal(rhs, 0)) return lhs;
        if (isConstVal(lhs, 0)) return rhs;
        break;
    }
    return null;
  }

  if (partial.kind === "unary") {
    const { op, input, dataType } = partial as Omit<UnaryNode, "id">;
    // Constant folding
    if (input.kind === "const") {
      const result = evalUnaryConst(op, input.value);
      if (result !== null) {
        return { kind: "const", value: result, valueType: "scalar", dataType, id: -1 } as ConstNode;
      }
    }
    // neg(neg(x)) → x
    if (op === "neg" && input.kind === "unary" && input.op === "neg") {
      return input.input;
    }
    return null;
  }

  if (partial.kind === "cmp") {
    const { op, lhs, rhs } = partial as Omit<CmpNode, "id">;
    if (lhs.kind === "const" && rhs.kind === "const") {
      const l = lhs.value, r = rhs.value;
      let result: boolean;
      switch (op) {
        case "eq": result = l === r; break;
        case "ne": result = l !== r; break;
        case "lt": result = l < r; break;
        case "le": result = l <= r; break;
        case "gt": result = l > r; break;
        case "ge": result = l >= r; break;
      }
      return { kind: "const", value: result! ? 1 : 0, valueType: "scalar", dataType: "u32", id: -1 } as ConstNode;
    }
    return null;
  }

  if (partial.kind === "cast") {
    const { input, targetType } = partial as Omit<CastNode, "id">;
    if (input.kind === "const") {
      return { kind: "const", value: input.value, valueType: "scalar", dataType: targetType, id: -1 } as ConstNode;
    }
    return null;
  }

  return null;
}

// ---- CSE key computation ----

function cseKey(node: { kind: string;[key: string]: any }): string | null {
  switch (node.kind) {
    case "const":    return `K:${node.dataType}:${node.value}`;
    case "binary":   return `B:${node.op}:${node.lhs.id}:${node.rhs.id}`;
    case "unary":    return `U:${node.op}:${node.input.id}`;
    case "cmp":      return `C:${node.op}:${node.lhs.id}:${node.rhs.id}`;
    case "cast":     return `T:${node.targetType}:${node.input.id}`;
    case "bitcast":  return `BC:${node.targetType}:${node.input.id}`;
    case "select":   return `S:${node.condition.id}:${node.trueVal.id}:${node.falseVal.id}`;
    case "uniform":  return `UNI:${node.name}`;
    case "programId": return `PID:${node.dim}`;
    case "threadIdx": return `TID:${node.dim}`;
    case "localIndex": return "LI";
    case "globalId": return `GID:${node.dim}`;
    case "numWorkgroups": return `NWG:${node.dim}`;
    case "vec4Construct": return `V4C:${node.x.id}:${node.y.id}:${node.z.id}:${node.w.id}`;
    case "vec4Splat":     return `V4S:${node.value.id}`;
    case "vec4NativeDot": return `V4D:${node.a.id}:${node.b.id}`;
    case "vec4Component": return `V4X:${node.value.id}:${node.comp}`;
    case "vec4Binary":    return `V4B:${node.op}:${node.a.id}:${node.b.id}`;
    default: return null; // Not CSE-eligible (loads, sharedRead, namedRef, vec4ArrayRead, etc.)
  }
}

function makeNode<T extends IRNode>(partial: Omit<T, "id">): T {
  // Phase 1: Constant folding + algebraic simplification
  const folded = tryFold(partial as unknown as Omit<IRNode, "id">);
  if (folded !== null) {
    if (folded.id >= 0) return folded as unknown as T; // existing node (algebraic simp)
    // New const from folding — check CSE, then assign id
    const key = cseKey(folded);
    if (key !== null) {
      const existing = cseCache.get(key);
      if (existing) return existing as unknown as T;
    }
    (folded as any).id = nextNodeId++;
    if (key !== null) cseCache.set(key, folded);
    return folded as unknown as T;
  }

  // Phase 2: CSE — return cached node if structurally identical
  const key = cseKey(partial as any);
  if (key !== null) {
    const existing = cseCache.get(key);
    if (existing) return existing as T;
  }
  const node = { ...partial, id: nextNodeId++ } as T;
  if (key !== null) cseCache.set(key, node);
  return node;
}

/** Resolve a BlockExpr or number into an IRNode. Numbers become constants of the given type. */
function resolveArgAs(arg: BlockExpr | number, dtype: DataType = "f32"): IRNode {
  if (typeof arg === "number") {
    return makeNode<ConstNode>({ kind: "const", valueType: "scalar", dataType: dtype, value: arg });
  }
  return arg.node;
}
const resolveArg = (arg: BlockExpr | number) => resolveArgAs(arg, "f32");
const resolveArgU32 = (arg: BlockExpr | number) => resolveArgAs(arg, "u32");

/** Determine the resulting valueType: block if either operand is block. */
function promoteValueType(a: ValueType, b: ValueType): ValueType {
  return a === "block" || b === "block" ? "block" : "scalar";
}

/**
 * Symbolic expression wrapper. All operations return new BlockExpr instances,
 * building an IR DAG that the compiler lowers to WGSL.
 */
export class BlockExpr {
  constructor(readonly node: IRNode) {}

  // -- Private helpers for node construction --
  private _binOp(op: BinaryOp, other: BlockExpr | number, u32Rhs: boolean, dt: DataType): BlockExpr {
    const rhs = u32Rhs ? resolveArgU32(other) : resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op, lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType), dataType: dt,
    }));
  }

  private _unaryOp(op: UnaryOp, dt?: DataType): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op, input: this.node,
      valueType: this.node.valueType, dataType: dt ?? this.node.dataType,
    }));
  }

  // -- Arithmetic --
  add(other: BlockExpr | number) { return this._binOp("add", other, false, this.node.dataType); }
  sub(other: BlockExpr | number) { return this._binOp("sub", other, false, this.node.dataType); }
  mul(other: BlockExpr | number) { return this._binOp("mul", other, false, this.node.dataType); }
  div(other: BlockExpr | number) { return this._binOp("div", other, false, this.node.dataType); }
  mod(other: BlockExpr | number) { return this._binOp("mod", other, false, this.node.dataType); }
  min(other: BlockExpr | number) { return this._binOp("min", other, false, this.node.dataType); }
  max(other: BlockExpr | number) { return this._binOp("max", other, false, this.node.dataType); }
  and(other: BlockExpr | number) { return this._binOp("and", other, true, "u32"); }
  or(other: BlockExpr | number) { return this._binOp("or", other, true, "u32"); }
  xor(other: BlockExpr | number) { return this._binOp("xor", other, true, "u32"); }
  shr(other: BlockExpr | number) { return this._binOp("shr", other, true, this.node.dataType); }
  shl(other: BlockExpr | number) { return this._binOp("shl", other, true, this.node.dataType); }

  // -- Unary math --
  rsqrt() { return this._unaryOp("rsqrt", "f32"); }
  exp()   { return this._unaryOp("exp", "f32"); }
  log()   { return this._unaryOp("log", "f32"); }
  sqrt()  { return this._unaryOp("sqrt", "f32"); }
  tanh()  { return this._unaryOp("tanh", "f32"); }
  sin()   { return this._unaryOp("sin", "f32"); }
  cos()   { return this._unaryOp("cos", "f32"); }
  exp2()  { return this._unaryOp("exp2", "f32"); }
  log2()  { return this._unaryOp("log2", "f32"); }
  abs()   { return this._unaryOp("abs"); }
  neg()   { return this._unaryOp("neg"); }
  floor() { return this._unaryOp("floor"); }
  ceil()  { return this._unaryOp("ceil"); }
  round() { return this._unaryOp("round"); }
  sign()  { return this._unaryOp("sign"); }

  /** Sigmoid activation: 1 / (1 + exp(-x)). Compound — no new IR node. */
  sigmoid(): BlockExpr {
    const one = new BlockExpr(makeNode<ConstNode>({
      kind: "const", value: 1.0, valueType: "scalar", dataType: "f32",
    }));
    return one.div(one.add(this.neg().exp()));
  }

  /** Clamp x to [lo, hi]. Compound — uses max(lo, min(x, hi)). */
  clamp(lo: BlockExpr | number, hi: BlockExpr | number): BlockExpr {
    return this.max(lo).min(hi);
  }

  /** Fused multiply-add: this * b + c. Compound — GPU will fuse automatically. */
  fma(b: BlockExpr | number, c: BlockExpr | number): BlockExpr {
    return this.mul(b).add(c);
  }

  /** Ceiling division: cdiv(a, b) = (a + b - 1) / b. Compound. */
  cdiv(other: BlockExpr | number): BlockExpr {
    const b = typeof other === "number" ? new BlockExpr(resolveArg(other)) : other;
    return this.add(b.sub(1)).div(b);
  }

  /** Floor division (truncated toward negative infinity). For unsigned, same as div. Compound. */
  floorDiv(other: BlockExpr | number): BlockExpr {
    return this.div(other).floor();
  }

  /** Approximate erf(x) using Abramowitz & Stegun (max error ~1.5e-7). Compound. */
  erf(): BlockExpr {
    // erf(x) = sign(x) * (1 - poly(t) * exp(-x²))
    // where t = 1/(1 + p*|x|), poly = a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
    const mkOne = () => new BlockExpr(makeNode<ConstNode>({
      kind: "const", value: 1.0, valueType: "scalar", dataType: "f32",
    }));
    const signX = this.sign();
    const absX = this.abs();
    const t = mkOne().div(mkOne().add(absX.mul(0.3275911)));
    // Horner: ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
    const inner = t.mul(1.061405429).add(-1.453152027)
      .mul(t).add(1.421413741)
      .mul(t).add(-0.284496736)
      .mul(t).add(0.254829592);
    const poly = inner.mul(t);
    const expTerm = absX.neg().mul(absX).exp(); // exp(-x²)
    return signX.mul(mkOne().sub(poly.mul(expTerm)));
  }

  pow(other: BlockExpr | number) { return this._binOp("pow", other, false, "f32"); }

  // -- Type casts --
  private _castTo(dt: DataType): BlockExpr {
    if (this.node.dataType === dt) return this;
    return new BlockExpr(makeNode<CastNode>({
      kind: "cast", input: this.node, targetType: dt,
      valueType: this.node.valueType, dataType: dt,
    }));
  }

  toF32() { return this._castTo("f32"); }
  toU32() { return this._castTo("u32"); }
  toI32() { return this._castTo("i32"); }
  toF16() { return this._castTo("f16"); }

  // -- Bitcast (reinterpret bits, no conversion) --
  bitcastTo(dtype: DataType): BlockExpr {
    return new BlockExpr(makeNode<BitcastNode>({
      kind: "bitcast", input: this.node, targetType: dtype,
      valueType: this.node.valueType, dataType: dtype,
    }));
  }

  // -- Subgroup ops --
  private _subgroupOp(kind: string): BlockExpr {
    return new BlockExpr(makeNode({
      kind, value: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    } as any));
  }

  subgroupShuffleXor(mask: BlockExpr | number): BlockExpr {
    const m = resolveArgU32(mask);
    return new BlockExpr(makeNode<SubgroupShuffleXorNode>({
      kind: "subgroupShuffleXor", value: this.node, mask: m,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  subgroupAdd() { return this._subgroupOp("subgroupAdd"); }
  subgroupMax() { return this._subgroupOp("subgroupMax"); }
  subgroupMin() { return this._subgroupOp("subgroupMin"); }
  subgroupBroadcastFirst() { return this._subgroupOp("subgroupBroadcastFirst"); }
  subgroupInclusiveAdd() { return this._subgroupOp("subgroupInclusiveAdd"); }

  // -- Vec4 operations (for vec4-typed BlockExprs) --

  /** dot(this, other) — both must be vec4<f32>. Returns f32 scalar. */
  vec4Dot(other: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<Vec4NativeDotNode>({
      kind: "vec4NativeDot", a: this.node, b: other.node,
      valueType: "scalar", dataType: "f32",
    }));
  }

  /** Extract component from vec4: .x=0, .y=1, .z=2, .w=3. Returns f32 scalar. */
  vec4Component(comp: 0 | 1 | 2 | 3): BlockExpr {
    return new BlockExpr(makeNode<Vec4ComponentNode>({
      kind: "vec4Component", value: this.node, comp,
      valueType: "scalar", dataType: "f32",
    }));
  }

  private _vec4BinOp(op: "add" | "sub" | "mul", other: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<Vec4BinaryNode>({
      kind: "vec4Binary", op, a: this.node, b: other.node,
      valueType: "scalar", dataType: "f32",
    }));
  }
  vec4MulScalar(s: BlockExpr) { return this._vec4BinOp("mul", s); }
  vec4Add(other: BlockExpr) { return this._vec4BinOp("add", other); }
  vec4Sub(other: BlockExpr) { return this._vec4BinOp("sub", other); }
  vec4Mul(other: BlockExpr) { return this._vec4BinOp("mul", other); }

  // -- Comparisons --
  private _cmpOp(op: CmpOp, other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<CmpNode>({
      kind: "cmp", op, lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType), dataType: "u32",
    }));
  }

  eq(other: BlockExpr | number) { return this._cmpOp("eq", other); }
  ne(other: BlockExpr | number) { return this._cmpOp("ne", other); }
  lt(other: BlockExpr | number) { return this._cmpOp("lt", other); }
  le(other: BlockExpr | number) { return this._cmpOp("le", other); }
  gt(other: BlockExpr | number) { return this._cmpOp("gt", other); }
  ge(other: BlockExpr | number) { return this._cmpOp("ge", other); }

  // -- Logical NOT --
  not() { return this._unaryOp("not"); }

  // -- Select (ternary) --
  select(trueVal: BlockExpr | number, falseVal: BlockExpr | number): BlockExpr {
    const t = resolveArg(trueVal);
    const f = resolveArg(falseVal);
    return new BlockExpr(makeNode<SelectNode>({
      kind: "select", condition: this.node, trueVal: t, falseVal: f,
      valueType: promoteValueType(
        promoteValueType(this.node.valueType, t.valueType),
        f.valueType,
      ),
      dataType: t.dataType,
    }));
  }
}

// ============================================================================
// Imperative Mode Handle Classes
// ============================================================================

/** Handle for a shared memory array, returned by `ctx.sharedArray()`. */
export class SharedArrayHandle {
  constructor(
    readonly name: string,
    readonly size: number,
    readonly elemType: DataType,
    private readonly ctx: KernelContext,
  ) {}

  read(idx: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<SharedReadNode>({
      kind: "sharedRead", arrayName: this.name, idx: idx.node,
      valueType: "scalar", dataType: this.elemType,
    }));
  }

  write(idx: BlockExpr, value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "sharedWrite",
      arrayName: this.name,
      idx: idx.node,
      value: value.node,
    });
  }
}

/** Handle for a `var` binding, returned by `ctx.emitVar()`. */
export class VarHandle {
  constructor(
    readonly name: string,
    readonly dtype: DataType,
    private readonly ctx: KernelContext,
  ) {}

  get(): BlockExpr {
    return new BlockExpr(makeNode<NamedRefNode>({
      kind: "namedRef", name: this.name,
      valueType: "scalar", dataType: this.dtype,
    }));
  }

  set(value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "assign",
      name: this.name,
      value: value.node,
    });
  }

  addAssign(value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "addAssign",
      name: this.name,
      value: value.node,
    });
  }
}

/** Handle for a `var` array binding, returned by `ctx.emitVarArray()`. */
export class ArrayVarHandle {
  constructor(
    readonly name: string,
    readonly elemType: DataType,
    readonly size: number,
    private readonly ctx: KernelContext,
  ) {}

  get(idx: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<ArrayReadNode>({
      kind: "arrayRead", arrayName: this.name, idx: idx.node,
      valueType: "scalar", dataType: this.elemType,
    }));
  }

  set(idx: BlockExpr, value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "indexAssign",
      arrayName: this.name,
      idx: idx.node,
      value: value.node,
    });
  }

  addAssign(idx: BlockExpr, value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "indexAddAssign",
      arrayName: this.name,
      idx: idx.node,
      value: value.node,
    });
  }
}

/** Handle for a vec4 array (register or shared), returned by ctx.registerVec4Array() / ctx.sharedVec4Array(). */
export class Vec4ArrayHandle {
  constructor(
    readonly name: string,
    readonly size: number,
    readonly isShared: boolean,
    private readonly ctx: KernelContext,
  ) {}

  read(idx: BlockExpr): BlockExpr {
    const nodeKind = this.isShared ? "vec4SharedRead" : "vec4ArrayRead";
    return new BlockExpr(makeNode({
      kind: nodeKind,
      arrayName: this.name,
      idx: idx.node,
      valueType: "scalar" as ValueType,
      dataType: "f32" as DataType,
    } as any));
  }

  write(idx: BlockExpr, value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "vec4ArrayWrite",
      arrayName: this.name,
      idx: idx.node,
      value: value.node,
      isShared: this.isShared,
    });
  }

  addAssign(idx: BlockExpr, value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "vec4ArrayAddAssign",
      arrayName: this.name,
      idx: idx.node,
      value: value.node,
      isShared: this.isShared,
    });
  }
}

// ============================================================================
// Kernel Context
// ============================================================================

/**
 * The context object passed to kernel functions. Provides Triton-like primitives
 * for building the IR DAG (auto-phase mode) and imperative statements.
 */
export class KernelContext {
  /** All nodes created during kernel execution, in creation order. */
  readonly nodes: IRNode[] = [];
  /** Statement list (imperative mode). */
  readonly statements: Statement[] = [];
  /** Shared memory arrays declared by this kernel. */
  readonly sharedArrays: Array<{ name: string; size: number; elemType: DataType }> = [];
  /** Vec4 shared memory arrays declared by this kernel. */
  readonly vec4SharedArrays: Array<{ name: string; size: number }> = [];
  /** Binding specs from the kernel spec (for dataType resolution in load). */
  private readonly bindingSpecs: Record<string, BindingSpec>;
  /** Subgroup size (0 = no subgroup support). Set by compileTileKernel(). */
  readonly subgroupSize: number;

  /** Statement stack for nested scopes (forRange, ifThen). */
  private stmtStack: Statement[][] = [];
  /** Counter for unique loop variable names. */
  private loopVarCounter = 0;
  /** Counter for unique wgReduce shared memory / accumulator names. */
  private reduceCounter = 0;
  /** Whether this kernel's reduction primitives used subgroup operations. */
  _usesSubgroups = false;
  /** Node IDs of flatGlobalId() results (for auto-vectorization override). */
  readonly flatGlobalIdNodeIds: number[] = [];

  /** Workgroup size (product if 2D), set by buildKernelIR from spec.workgroupSize. */
  private _wgSize: number | [number, number] = 0;

  constructor(bindings?: Record<string, BindingSpec>, subgroupSize = 0) {
    this.bindingSpecs = bindings ?? {};
    this.subgroupSize = subgroupSize;
  }

  /** @internal Set workgroup size from spec. Called by buildKernelIR(). */
  _setWgSize(wgSize: number | [number, number]): void {
    this._wgSize = wgSize;
  }

  private trackNode<T extends IRNode>(node: T): T {
    this.nodes.push(node);
    return node;
  }

  /** Push a statement to the current scope (top of stack or root). */
  pushStatement(stmt: Statement): void {
    const target = this.stmtStack.length > 0
      ? this.stmtStack[this.stmtStack.length - 1]
      : this.statements;
    target.push(stmt);
  }

  // ---- Auto-phase mode API (unchanged) ----

  /** Workgroup ID for the given dimension (0=x, 1=y, 2=z). */
  programId(dim: number): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<ProgramIdNode>({
      kind: "programId", dim,
      valueType: "scalar", dataType: "u32",
    })));
  }

  /** Number of workgroups dispatched in given dimension (like Triton's `tl.num_programs`). */
  numPrograms(dim: number): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<NumWorkgroupsNode>({
      kind: "numWorkgroups", dim,
      valueType: "scalar", dataType: "u32",
    })));
  }

  /** Load from a storage buffer at given offsets. Optional mask for bounds. */
  load(binding: string, offsets: BlockExpr, mask?: BlockExpr): BlockExpr {
    const bindingType = this.bindingSpecs[binding]?.type ?? "f32";
    return new BlockExpr(this.trackNode(makeNode<LoadNode>({
      kind: "load", binding, offsets: offsets.node,
      ...(mask ? { mask: mask.node } : {}),
      valueType: "block", dataType: bindingType as DataType,
    } as any)));
  }

  /** Read a uniform config value. */
  uniform(name: string): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<UniformNode>({
      kind: "uniform", name,
      valueType: "scalar", dataType: "u32",
    })));
  }

  /** Create a scalar constant. */
  const(value: number, dataType: DataType = "f32"): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<ConstNode>({
      kind: "const", value,
      valueType: "scalar", dataType,
    })));
  }

  /** Shorthand: u32 constant. */
  u32(value: number): BlockExpr { return this.const(value, "u32"); }

  /** Shorthand: i32 constant. */
  i32(value: number): BlockExpr { return this.const(value, "i32"); }

  /** Shorthand: f32 constant. */
  f32(value: number): BlockExpr { return this.const(value, "f32"); }

  /** Shorthand: f16 constant. */
  f16(value: number): BlockExpr { return this.const(value, "f16"); }

  // ---- Imperative mode API ----

  /** Thread-local invocation ID for the given dimension (0=x, 1=y, 2=z). */
  threadIdx(dim: number): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<ThreadIdxNode>({
      kind: "threadIdx", dim,
      valueType: "scalar", dataType: "u32",
    })));
  }

  /** Flat local invocation index (local_invocation_index). */
  localIndex(): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<LocalIndexNode>({
      kind: "localIndex",
      valueType: "scalar", dataType: "u32",
    })));
  }

  /** Global invocation ID (global_invocation_id.x/y/z). */
  globalId(dim: number): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<GlobalIdNode>({
      kind: "globalId", dim,
      valueType: "scalar", dataType: "u32",
    })));
  }

  /**
   * Flat global thread index, correct for both 1D and 2D dispatch.
   * For 1D dispatch: gid.x (same as globalId(0)).
   * For 2D dispatch: (wg.x + wg.y * num_wg.x) * workgroupSize + local_index.
   */
  flatGlobalId(workgroupSize: number): BlockExpr {
    const flatWgId = this.programId(0).add(
      this.programId(1).mul(this.numPrograms(0)),
    );
    const result = flatWgId.mul(this.u32(workgroupSize)).add(this.localIndex());
    this.flatGlobalIdNodeIds.push(result.node.id);
    return result;
  }

  /**
   * Decompose a flat linear index into multi-dimensional coordinates.
   * Given shape [D0, D1, ..., Dn], returns [c0, c1, ..., cn] where
   * flatIdx = c0 * (D1*D2*...*Dn) + c1 * (D2*...*Dn) + ... + cn.
   *
   * Equivalent to numpy.unravel_index or Triton's manual `pid // cols, pid % cols`.
   */
  decomposeIndex(flatIdx: BlockExpr, shape: number[]): BlockExpr[] {
    const rank = shape.length;
    if (rank === 0) return [];
    if (rank === 1) return [flatIdx];

    const coords: BlockExpr[] = [];
    let rem = flatIdx;
    for (let d = 0; d < rank; d++) {
      // Stride = product of remaining dimensions
      let dimStride = 1;
      for (let j = d + 1; j < rank; j++) dimStride *= shape[j];

      if (d < rank - 1) {
        coords.push(rem.div(this.u32(dimStride)));
        rem = rem.mod(this.u32(dimStride));
      } else {
        coords.push(rem); // last coord is the remainder
      }
    }
    return coords;
  }

  /**
   * Linearize multi-dimensional coordinates using given strides.
   * Returns coords[0]*strides[0] + coords[1]*strides[1] + ... + offset.
   *
   * Handles stride=0 (broadcast) via constant folding (coord*0 → 0).
   */
  linearizeIndex(coords: BlockExpr[], strides: number[], offset = 0): BlockExpr {
    let result: BlockExpr = this.u32(offset);
    for (let d = 0; d < coords.length; d++) {
      result = result.add(coords[d].mul(this.u32(strides[d])));
    }
    return result;
  }

  /** Declare a workgroup shared memory array. Returns a handle for read/write. */
  sharedArray(name: string, size: number, elemType: DataType = "f32"): SharedArrayHandle {
    this.sharedArrays.push({ name, size, elemType });
    return new SharedArrayHandle(name, size, elemType, this);
  }

  /** Emit a `let` binding and return a reference to it. */
  emitLet(name: string, expr: BlockExpr): BlockExpr {
    this.pushStatement({
      kind: "let",
      name,
      value: expr.node,
      dtype: expr.node.dataType,
    });
    return new BlockExpr(makeNode<NamedRefNode>({
      kind: "namedRef", name,
      valueType: "scalar", dataType: expr.node.dataType,
    }));
  }

  /** Emit a `var` binding and return a handle for get/set/addAssign. */
  emitVar(name: string, dtype: DataType, init: BlockExpr): VarHandle {
    this.pushStatement({
      kind: "var",
      name,
      value: init.node,
      dtype,
    });
    return new VarHandle(name, dtype, this);
  }

  /** Emit a `var` array and return a handle. Zero-initialized by default. */
  emitVarArray(name: string, elemType: DataType, size: number, skipZeroInit?: boolean): ArrayVarHandle {
    this.pushStatement({
      kind: "varArray",
      name,
      elemType,
      size,
      skipZeroInit,
    });
    return new ArrayVarHandle(name, elemType, size, this);
  }

  // ---- Vec4 array support ----

  /** Declare a register-space vec4 array: `var name: array<vec4<f32>, size>;` */
  registerVec4Array(name: string, size: number): Vec4ArrayHandle {
    this.pushStatement({ kind: "vec4VarArray", name, size });
    return new Vec4ArrayHandle(name, size, false, this);
  }

  /** Declare a workgroup shared vec4 array: `var<workgroup> name: array<vec4<f32>, size>;` */
  sharedVec4Array(name: string, size: number): Vec4ArrayHandle {
    this.vec4SharedArrays.push({ name, size });
    return new Vec4ArrayHandle(name, size, true, this);
  }

  /** Construct vec4<f32>(x, y, z, w) from 4 scalar expressions. */
  vec4(x: BlockExpr, y: BlockExpr, z: BlockExpr, w: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<Vec4ConstructNode>({
      kind: "vec4Construct", x: x.node, y: y.node, z: z.node, w: w.node,
      valueType: "scalar", dataType: "f32",
    }));
  }

  /** Construct vec4<f32>(v) — splat scalar to all 4 lanes. */
  vec4Splat(v: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<Vec4SplatNode>({
      kind: "vec4Splat", value: v.node,
      valueType: "scalar", dataType: "f32",
    }));
  }

  /** Shared loop emission for forRange/forStride. */
  private _emitLoop(
    kind: "forRange" | "forStride", start: BlockExpr, bound: BlockExpr,
    body: (loopVar: BlockExpr) => void,
    extra?: { stride?: number; unroll?: boolean },
  ): void {
    const varName = `_lv${this.loopVarCounter++}`;
    const bodyStmts: Statement[] = [];
    this.stmtStack.push(bodyStmts);
    const loopVar = new BlockExpr(makeNode<NamedRefNode>({
      kind: "namedRef", name: varName, valueType: "scalar", dataType: "u32",
    }));
    body(loopVar);
    this.stmtStack.pop();
    const stmt: any = {
      kind, varName, start: start.node, bound: bound.node, body: bodyStmts,
    };
    if (extra?.stride != null) stmt.stride = extra.stride;
    if (extra?.unroll != null) stmt.unroll = extra.unroll;
    this.pushStatement(stmt);
  }

  /** Emit a for loop: `for (var v = start; v < bound; v++)`. */
  forRange(start: BlockExpr, bound: BlockExpr, body: (loopVar: BlockExpr) => void,
    opts?: { unroll?: boolean }): void {
    this._emitLoop("forRange", start, bound, body, opts?.unroll ? { unroll: true } : undefined);
  }

  /** Strided for loop: `for (var i = start; i < bound; i += stride)`.
   *  Auto-unrolled when start/bound are const and trip count ≤ 8.
   *  Set opts.unroll to force unrolling up to trip count 16. */
  forStride(start: BlockExpr, bound: BlockExpr, stride: number,
    body: (i: BlockExpr) => void, opts?: { unroll?: boolean }): void {
    this._emitLoop("forStride", start, bound, body, { stride, unroll: opts?.unroll });
  }

  /** Emit a compile-time unrolled loop. Body called N times with constant index. */
  forUnrolled(count: number, body: (i: BlockExpr) => void): void {
    for (let i = 0; i < count; i++) {
      body(this.u32(i));
    }
  }

  /** Emit an if-then block. */
  ifThen(cond: BlockExpr, body: () => void): void {
    const bodyStmts: Statement[] = [];
    this.stmtStack.push(bodyStmts);
    body();
    this.stmtStack.pop();
    this.pushStatement({
      kind: "if",
      condition: cond.node,
      body: bodyStmts,
    });
  }

  /** Emit an if-then-else block. */
  ifThenElse(cond: BlockExpr, body: () => void, elseBody: () => void): void {
    const bodyStmts: Statement[] = [];
    this.stmtStack.push(bodyStmts);
    body();
    this.stmtStack.pop();
    const elseStmts: Statement[] = [];
    this.stmtStack.push(elseStmts);
    elseBody();
    this.stmtStack.pop();
    this.pushStatement({
      kind: "ifElse",
      condition: cond.node,
      body: bodyStmts,
      elseBody: elseStmts,
    });
  }

  /** Emit a workgroupBarrier(). */
  barrier(): void {
    this.pushStatement({ kind: "barrier" });
  }

  /**
   * Capture statements emitted inside a callback into a separate array.
   * Used by tile-level ops to capture sub-bodies (e.g., apply() lambdas).
   */
  captureScope(body: () => void): Statement[] {
    const captured: Statement[] = [];
    this.stmtStack.push(captured);
    body();
    this.stmtStack.pop();
    return captured;
  }

  // ===========================================================================
  // Composable reduction / loop primitives
  // ===========================================================================

  /** Whether subgroup-optimized path should be used for the given workgroup size. */
  private canUseSubgroups(wgSize: number): boolean {
    const sg = this.subgroupSize;
    const ok = sg > 0 && wgSize >= sg * 2 && wgSize % sg === 0;
    if (ok) this._usesSubgroups = true;
    return ok;
  }

  /** Tree reduction: `smem[0] = op(smem[0..wgSize-1])`. Emits log2(wgSize) if+barrier pairs. */
  private _treeReduce(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number,
      op: "sum" | "max" | "min", combine: (a: BlockExpr, b: BlockExpr) => BlockExpr): void {
    if (this.canUseSubgroups(wgSize)) {
      this._treeReduceSubgroup(smem, tid, wgSize, op);
      return;
    }
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.ifThen(tid.lt(this.u32(stride)), () => {
        smem.write(tid, combine(smem.read(tid), smem.read(tid.add(this.u32(stride)))));
      });
      this.barrier();
    }
  }

  treeReduceSum(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number): void {
    this._treeReduce(smem, tid, wgSize, "sum", (a, b) => a.add(b));
  }

  treeReduceMax(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number): void {
    this._treeReduce(smem, tid, wgSize, "max", (a, b) => a.max(b));
  }

  treeReduceMin(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number): void {
    this._treeReduce(smem, tid, wgSize, "min", (a, b) => a.min(b));
  }

  /**
   * Subgroup-optimized tree reduction.
   * 1. Read from smem → subgroupAdd/Max → lane 0 of each subgroup writes back
   * 2. Small tree reduction across numSubgroups values
   */
  private _treeReduceSubgroup(
    smem: SharedArrayHandle, tid: BlockExpr, wgSize: number, op: "sum" | "max" | "min",
  ): void {
    const sg = this.subgroupSize;
    const numSubgroups = wgSize / sg;
    // Phase 1: subgroup reduction (0 barriers)
    const val = this.emitLet("_sgr_val", smem.read(tid));
    const sgReduced = this.emitLet("_sgr_red",
      op === "sum" ? val.subgroupAdd() : op === "max" ? val.subgroupMax() : val.subgroupMin());
    // Write subgroup leaders to smem[subgroupId]
    this.barrier(); // sync before smem rewrite
    this.ifThen(tid.mod(this.u32(sg)).eq(this.u32(0)), () => {
      smem.write(tid.div(this.u32(sg)), sgReduced);
    });
    this.barrier();
    // Phase 2: small tree reduction on numSubgroups values
    for (let stride = numSubgroups >> 1; stride >= 1; stride >>= 1) {
      this.ifThen(tid.lt(this.u32(stride)), () => {
        const a = smem.read(tid);
        const b = smem.read(tid.add(this.u32(stride)));
        smem.write(tid, op === "sum" ? a.add(b) : op === "max" ? a.max(b) : a.min(b));
      });
      this.barrier();
    }
  }

  /** Dual tree sum reduction: reduces two shared arrays in parallel. */
  dualTreeReduceSum(
    smem1: SharedArrayHandle, smem2: SharedArrayHandle,
    tid: BlockExpr, wgSize: number,
  ): void {
    if (this.canUseSubgroups(wgSize)) {
      const sg = this.subgroupSize;
      const numSubgroups = wgSize / sg;
      // Phase 1: subgroup reduction
      const v1 = this.emitLet("_sgrd1_v", smem1.read(tid));
      const v2 = this.emitLet("_sgrd2_v", smem2.read(tid));
      const r1 = this.emitLet("_sgrd1_r", v1.subgroupAdd());
      const r2 = this.emitLet("_sgrd2_r", v2.subgroupAdd());
      this.barrier();
      this.ifThen(tid.mod(this.u32(sg)).eq(this.u32(0)), () => {
        smem1.write(tid.div(this.u32(sg)), r1);
        smem2.write(tid.div(this.u32(sg)), r2);
      });
      this.barrier();
      // Phase 2: small tree reduction
      for (let stride = numSubgroups >> 1; stride >= 1; stride >>= 1) {
        this.ifThen(tid.lt(this.u32(stride)), () => {
          smem1.write(tid, smem1.read(tid).add(smem1.read(tid.add(this.u32(stride)))));
          smem2.write(tid, smem2.read(tid).add(smem2.read(tid.add(this.u32(stride)))));
        });
        this.barrier();
      }
      return;
    }
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.ifThen(tid.lt(this.u32(stride)), () => {
        smem1.write(tid, smem1.read(tid).add(smem1.read(tid.add(this.u32(stride)))));
        smem2.write(tid, smem2.read(tid).add(smem2.read(tid.add(this.u32(stride)))));
      });
      this.barrier();
    }
  }

  /** Generic tree reduction with user-defined combine function. */
  treeReduceGeneric(
    smem: SharedArrayHandle, tid: BlockExpr, wgSize: number,
    combine: (a: BlockExpr, b: BlockExpr) => BlockExpr,
  ): void {
    if (this.canUseSubgroups(wgSize)) {
      // Use subgroupShuffleXor butterfly for custom combine
      const sg = this.subgroupSize;
      const numSubgroups = wgSize / sg;
      let val = this.emitLet("_sgrg_val", smem.read(tid));
      for (let mask = sg >> 1; mask >= 1; mask >>= 1) {
        val = this.emitLet(`_sgrg_m${mask}`, combine(val, val.subgroupShuffleXor(mask)));
      }
      this.barrier();
      this.ifThen(tid.mod(this.u32(sg)).eq(this.u32(0)), () => {
        smem.write(tid.div(this.u32(sg)), val);
      });
      this.barrier();
      for (let stride = numSubgroups >> 1; stride >= 1; stride >>= 1) {
        this.ifThen(tid.lt(this.u32(stride)), () => {
          smem.write(tid, combine(smem.read(tid), smem.read(tid.add(this.u32(stride)))));
        });
        this.barrier();
      }
      return;
    }
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.ifThen(tid.lt(this.u32(stride)), () => {
        smem.write(tid, combine(smem.read(tid), smem.read(tid.add(this.u32(stride)))));
      });
      this.barrier();
    }
  }

  /**
   * Generic workgroup-cooperative reduction with user-defined combine.
   * Like `wgReduce` but takes a combine function and identity value.
   */
  wgReduceGeneric(
    tid: BlockExpr, count: BlockExpr, wgSize: number,
    identity: BlockExpr,
    bodyFn: (i: BlockExpr) => BlockExpr,
    combine: (a: BlockExpr, b: BlockExpr) => BlockExpr,
  ): BlockExpr {
    const id = this.reduceCounter++;
    const acc = this.emitVar(`_wgr${id}_a`, "f32", identity);
    this.stridedFor(tid, count, wgSize, (i) => {
      acc.set(combine(acc.get(), bodyFn(i)));
    });

    if (this.canUseSubgroups(wgSize)) {
      const sg = this.subgroupSize;
      const numSubgroups = wgSize / sg;
      // Butterfly subgroup reduction via shuffleXor
      let val = this.emitLet(`_wgr${id}_sv`, acc.get());
      for (let mask = sg >> 1; mask >= 1; mask >>= 1) {
        val = this.emitLet(`_wgr${id}_m${mask}`, combine(val, val.subgroupShuffleXor(mask)));
      }
      const smem = this.sharedArray(`_wgr${id}_s`, numSubgroups);
      this.ifThen(tid.mod(this.u32(sg)).eq(this.u32(0)), () => {
        smem.write(tid.div(this.u32(sg)), val);
      });
      this.barrier();
      for (let stride = numSubgroups >> 1; stride >= 1; stride >>= 1) {
        this.ifThen(tid.lt(this.u32(stride)), () => {
          smem.write(tid, combine(smem.read(tid), smem.read(tid.add(this.u32(stride)))));
        });
        this.barrier();
      }
      return smem.read(this.u32(0));
    }

    const smem = this.sharedArray(`_wgr${id}_s`, wgSize);
    smem.write(tid, acc.get());
    this.barrier();
    this.treeReduceGeneric(smem, tid, wgSize, combine);
    return smem.read(this.u32(0));
  }

  /** Strided loop: `for (var i = tid; i < bound; i += wgSize) { body(i); }` */
  stridedFor(
    tid: BlockExpr, bound: BlockExpr, wgSize: number,
    body: (i: BlockExpr) => void,
  ): void {
    this.forStride(tid, bound, wgSize, body);
  }

  /**
   * Workgroup-cooperative reduction. Allocates shared memory, accumulates via
   * stridedFor, writes to shared memory, barriers, and tree-reduces.
   * Returns the reduced scalar (`smem[0]` after reduction).
   *
   * With subgroups: accumulates in registers, reduces within subgroup via
   * subgroupAdd/Max, then small tree reduction across subgroup leaders.
   */
  wgReduce(
    op: "sum" | "max" | "min",
    tid: BlockExpr, count: BlockExpr, wgSize: number,
    bodyFn: (i: BlockExpr) => BlockExpr,
  ): BlockExpr {
    const id = this.reduceCounter++;
    const identity = op === "sum" ? 0.0 : op === "max" ? F32_NEG_MAX : F32_POS_MAX;
    const acc = this.emitVar(`_wgr${id}_a`, "f32", this.f32(identity));
    this.stridedFor(tid, count, wgSize, (i) => {
      if (op === "sum") {
        acc.addAssign(bodyFn(i));
      } else if (op === "max") {
        acc.set(acc.get().max(bodyFn(i)));
      } else {
        acc.set(acc.get().min(bodyFn(i)));
      }
    });

    if (this.canUseSubgroups(wgSize)) {
      const sg = this.subgroupSize;
      const numSubgroups = wgSize / sg;
      // Phase 1: subgroup reduction from registers (0 barriers, no smem write)
      const sgReduced = this.emitLet(`_wgr${id}_sgr`,
        op === "sum" ? acc.get().subgroupAdd() : op === "max" ? acc.get().subgroupMax() : acc.get().subgroupMin());
      // Phase 2: leaders write to small smem
      const smem = this.sharedArray(`_wgr${id}_s`, numSubgroups);
      this.ifThen(tid.mod(this.u32(sg)).eq(this.u32(0)), () => {
        smem.write(tid.div(this.u32(sg)), sgReduced);
      });
      this.barrier();
      // Phase 3: small tree reduction
      for (let stride = numSubgroups >> 1; stride >= 1; stride >>= 1) {
        this.ifThen(tid.lt(this.u32(stride)), () => {
          const a = smem.read(tid);
          const b = smem.read(tid.add(this.u32(stride)));
          smem.write(tid, op === "sum" ? a.add(b) : op === "max" ? a.max(b) : a.min(b));
        });
        this.barrier();
      }
      return smem.read(this.u32(0));
    }

    // Fallback: full shared memory tree reduction
    const smem = this.sharedArray(`_wgr${id}_s`, wgSize);
    smem.write(tid, acc.get());
    this.barrier();
    if (op === "sum") {
      this.treeReduceSum(smem, tid, wgSize);
    } else if (op === "max") {
      this.treeReduceMax(smem, tid, wgSize);
    } else {
      this.treeReduceMin(smem, tid, wgSize);
    }
    return smem.read(this.u32(0));
  }

  /**
   * Dual workgroup-cooperative sum reduction. Same as wgReduce but reduces
   * two values in parallel using a single stridedFor pass.
   * Returns `[smem1[0], smem2[0]]` after reduction.
   *
   * With subgroups: both accumulators are reduced via subgroupAdd, then
   * leaders write to small smem arrays for final tree reduction.
   */
  dualWgReduce(
    tid: BlockExpr, count: BlockExpr, wgSize: number,
    bodyFn: (i: BlockExpr) => [BlockExpr, BlockExpr],
  ): [BlockExpr, BlockExpr] {
    const id = this.reduceCounter++;
    const acc1 = this.emitVar(`_wgr${id}_a1`, "f32", this.f32(0.0));
    const acc2 = this.emitVar(`_wgr${id}_a2`, "f32", this.f32(0.0));
    this.stridedFor(tid, count, wgSize, (i) => {
      const [v1, v2] = bodyFn(i);
      acc1.addAssign(v1);
      acc2.addAssign(v2);
    });

    if (this.canUseSubgroups(wgSize)) {
      const sg = this.subgroupSize;
      const numSubgroups = wgSize / sg;
      const sgr1 = this.emitLet(`_wgr${id}_sgr1`, acc1.get().subgroupAdd());
      const sgr2 = this.emitLet(`_wgr${id}_sgr2`, acc2.get().subgroupAdd());
      const smem1 = this.sharedArray(`_wgr${id}_s1`, numSubgroups);
      const smem2 = this.sharedArray(`_wgr${id}_s2`, numSubgroups);
      this.ifThen(tid.mod(this.u32(sg)).eq(this.u32(0)), () => {
        smem1.write(tid.div(this.u32(sg)), sgr1);
        smem2.write(tid.div(this.u32(sg)), sgr2);
      });
      this.barrier();
      for (let stride = numSubgroups >> 1; stride >= 1; stride >>= 1) {
        this.ifThen(tid.lt(this.u32(stride)), () => {
          smem1.write(tid, smem1.read(tid).add(smem1.read(tid.add(this.u32(stride)))));
          smem2.write(tid, smem2.read(tid).add(smem2.read(tid.add(this.u32(stride)))));
        });
        this.barrier();
      }
      return [smem1.read(this.u32(0)), smem2.read(this.u32(0))];
    }

    // Fallback
    const smem1 = this.sharedArray(`_wgr${id}_s1`, wgSize);
    const smem2 = this.sharedArray(`_wgr${id}_s2`, wgSize);
    smem1.write(tid, acc1.get());
    smem2.write(tid, acc2.get());
    this.barrier();
    this.dualTreeReduceSum(smem1, smem2, tid, wgSize);
    return [smem1.read(this.u32(0)), smem2.read(this.u32(0))];
  }

  /** Emit an atomic operation: `atomicMax(&binding[idx], val)` etc. */
  atomicOp(binding: string, idx: BlockExpr, op: AtomicOp, val: BlockExpr): void {
    this.pushStatement({
      kind: "atomicOp",
      binding,
      idx: idx.node,
      op,
      value: val.node,
    });
  }

  /**
   * Emit an atomic compare-and-swap.
   * `atomicCompareExchangeWeak(&binding[idx], expected, desired)`
   * Returns `{ oldValue: BlockExpr, exchanged: BlockExpr }`.
   */
  atomicCAS(
    binding: string, idx: BlockExpr,
    expected: BlockExpr, desired: BlockExpr,
  ): { oldValue: BlockExpr; exchanged: BlockExpr } {
    const id = this.reduceCounter++;
    const oldVar = `_cas${id}_old`;
    const exchVar = `_cas${id}_ok`;
    this.pushStatement({
      kind: "atomicCAS",
      binding,
      idx: idx.node,
      expected: expected.node,
      desired: desired.node,
      oldValueVar: oldVar,
      exchangedVar: exchVar,
    });
    // Return named refs so subsequent code can read the results
    const oldNode = makeNode<NamedRefNode>({
      kind: "namedRef", name: oldVar,
      valueType: "scalar", dataType: "u32",
    });
    const exchNode = makeNode<NamedRefNode>({
      kind: "namedRef", name: exchVar,
      valueType: "scalar", dataType: "u32",
    });
    return {
      oldValue: new BlockExpr(oldNode),
      exchanged: new BlockExpr(exchNode),
    };
  }

  /** Emit an unconditional scalar store: `binding[idx] = val;`. */
  emitStore(binding: string, idx: BlockExpr, val: BlockExpr): void {
    this.pushStatement({
      kind: "directStore",
      binding,
      idx: idx.node,
      value: val.node,
    });
  }

  /** Store a vec4 value as 4 consecutive scalar elements.
   *  Expands to: store(binding, base+0, v.x); store(binding, base+1, v.y); ... */
  vec4Store(binding: string, baseIdx: BlockExpr, vec: BlockExpr): void {
    for (let c = 0; c < 4; c++) {
      const idx = c === 0 ? baseIdx : baseIdx.add(this.u32(c));
      this.emitStore(binding, idx, vec.vec4Component(c as 0 | 1 | 2 | 3));
    }
  }

  /** Emit an early `return` statement. */
  emitReturn(): void {
    this.pushStatement({ kind: "return" });
  }

  /** Emit a guarded store: `if (cond) { binding[idx] = val; }`. */
  guardedStore(binding: string, cond: BlockExpr, idx: BlockExpr, val: BlockExpr): void {
    this.pushStatement({
      kind: "guardedStore",
      binding,
      condition: cond.node,
      idx: idx.node,
      value: val.node,
    });
  }

  // -- Triton-like convenience APIs --

  /** Block-level where (like `tl.where`): returns trueVal where cond is true, falseVal otherwise. */
  blockWhere(cond: BlockExpr, trueVal: BlockExpr, falseVal: BlockExpr): BlockExpr {
    return cond.select(trueVal, falseVal);
  }

  /** Thread-distributed index range (like `tl.arange(0, BLOCK) + base`).
   *  Returns: programId(0) * blockSize + localIndex() + base. */
  blockRange(base: BlockExpr, blockSize: number): BlockExpr {
    return this.programId(0).mul(this.u32(blockSize)).add(this.localIndex()).add(base);
  }

  /**
   * Get the flat element index with automatic early-return bounds check.
   * Combines `flatGlobalId` + `ifThen(idx >= size, return)` into one call.
   *
   * Like Triton's pattern: `offsets = pid * BLOCK + tl.arange(0, BLOCK)`
   * followed by `mask = offsets < n` — but using early-return instead of masking
   * since WebGPU dispatches one thread per element.
   *
   * @param wgSize - Workgroup size (typically 256)
   * @param size   - Upper bound. String → uniform name; BlockExpr → direct. Default: "size"
   */
  elementIndex(wgSize: number, size?: BlockExpr | string): BlockExpr {
    const idx = this.flatGlobalId(wgSize);
    const bound = size === undefined ? this.uniform("size")
      : typeof size === "string" ? this.uniform(size)
      : size;
    this.ifThen(idx.ge(bound), () => this.emitReturn());
    return idx;
  }

  /**
   * Masked load from storage buffer (like `tl.load(ptr + offs, mask, other)`).
   * Returns the loaded value when idx < size, otherwise returns fallback.
   * Safe because WebGPU robust buffer access clamps out-of-bounds reads.
   *
   * @param binding  - Storage buffer name
   * @param idx      - Element index
   * @param size     - Upper bound for valid indices
   * @param fallback - Value for out-of-bounds (default: zero of binding's dtype)
   */
  blockLoad(binding: string, idx: BlockExpr, size: BlockExpr, fallback?: BlockExpr): BlockExpr {
    const val = this.load(binding, idx);
    const fb = fallback ?? this.const(0, this.bindingSpecs[binding]?.type as DataType ?? "f32");
    return idx.lt(size).select(val, fb);
  }

  /**
   * Masked store to buffer (like `tl.store(ptr + offs, val, mask)`).
   * Only writes when idx < size; no-op for out-of-bounds threads.
   *
   * @param binding - Storage buffer name
   * @param idx     - Element index
   * @param size    - Upper bound for valid indices
   * @param val     - Value to store
   */
  blockStore(binding: string, idx: BlockExpr, size: BlockExpr, val: BlockExpr): void {
    this.guardedStore(binding, idx.lt(size), idx, val);
  }

  /**
   * Map a flat index through a strided layout (like Triton's pointer arithmetic).
   * Combines `decomposeIndex` + `linearizeIndex` into one call.
   *
   * Common pattern it replaces:
   *   const coords = ctx.decomposeIndex(idx, shape);
   *   const offset = ctx.linearizeIndex(coords, strides, baseOffset);
   *
   * @param flatIdx     - Flat linear index to remap
   * @param indexShape  - Shape to decompose against (the "logical" shape)
   * @param strides     - Strides for re-linearization
   * @param baseOffset  - Base offset (default: 0)
   */
  stridedIndex(flatIdx: BlockExpr, indexShape: number[], strides: number[], baseOffset = 0): BlockExpr {
    const coords = this.decomposeIndex(flatIdx, indexShape);
    return this.linearizeIndex(coords, strides, baseOffset);
  }

  /**
   * Iterate over a range [start, end) with constant bounds (like `for i in range(n)`).
   * Sugar over `forRange` that accepts numbers instead of BlockExprs.
   *
   * @param start - Start value (inclusive)
   * @param end   - End value (exclusive), or the body if start=0
   * @param body  - Loop body receiving the iteration variable
   */
  range(start: number, end: number, body: (i: BlockExpr) => void): void {
    this.forRange(this.u32(start), this.u32(end), body);
  }

  /**
   * Cooperatively load two vec4 shared memory tiles in a single strided loop.
   * Uses one loop and one barrier instead of two, saving one barrier per tile pair.
   * Common pattern in attention kernels (load K and V simultaneously).
   *
   * @param tid        - Thread index within workgroup
   * @param wgSize     - Workgroup size
   * @param smemA      - First vec4 shared memory array
   * @param smemB      - Second vec4 shared memory array
   * @param count      - Number of vec4 elements per tile
   * @param loadFn     - Function mapping tile-local index → [valueA, valueB]
   */
  tileLoadPairVec4(
    tid: BlockExpr, wgSize: number,
    smemA: Vec4ArrayHandle, smemB: Vec4ArrayHandle,
    count: BlockExpr | number,
    loadFn: (i: BlockExpr) => [BlockExpr, BlockExpr],
  ): void {
    const bound = typeof count === "number" ? this.u32(count) : count;
    this.stridedFor(tid, bound, wgSize, (i) => {
      const [a, b] = loadFn(i);
      smemA.write(i, a);
      smemB.write(i, b);
    });
    this.barrier();
  }

  // -- Scan primitives (like `tl.cumsum`, `tl.associative_scan`) --

  /** Hillis-Steele inclusive parallel prefix scan in shared memory.
   *  After the call, smem[tid] = op(smem[0], ..., smem[tid]).
   *  Requires tid < wgSize.
   *
   *  With subgroups (sum only): subgroupInclusiveAdd within each subgroup,
   *  then cross-subgroup scan via sequential accumulation + broadcast. */
  inclusiveScan(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number, op: "sum" | "max"): void {
    if (op === "sum" && this.canUseSubgroups(wgSize)) {
      const sg = this.subgroupSize;
      const numSubgroups = wgSize / sg;
      // Phase 1: subgroup-local inclusive prefix sum
      const val = this.emitLet("_sgs_val", smem.read(tid));
      const sgPrefix = this.emitLet("_sgs_pfx", val.subgroupInclusiveAdd());
      this.barrier();
      smem.write(tid, sgPrefix);
      this.barrier();
      // Phase 2: cross-subgroup scan on subgroup totals (last element of each subgroup)
      // Use a small shared memory array for subgroup totals
      const sgTotals = this.sharedArray("_sgs_totals", numSubgroups);
      // Lane (sgSize-1) of each subgroup has the subgroup total
      this.ifThen(tid.mod(this.u32(sg)).eq(this.u32(sg - 1)), () => {
        sgTotals.write(tid.div(this.u32(sg)), sgPrefix);
      });
      this.barrier();
      // Sequential scan on the small array (numSubgroups is small, e.g. 8)
      // Only thread 0 does this
      this.ifThen(tid.eq(this.u32(0)), () => {
        for (let i = 1; i < numSubgroups; i++) {
          sgTotals.write(this.u32(i), sgTotals.read(this.u32(i)).add(sgTotals.read(this.u32(i - 1))));
        }
      });
      this.barrier();
      // Phase 3: add cross-subgroup prefix to all elements
      const sgId = this.emitLet("_sgs_sgid", tid.div(this.u32(sg)));
      this.ifThen(sgId.gt(this.u32(0)), () => {
        smem.write(tid, smem.read(tid).add(sgTotals.read(sgId.sub(this.u32(1)))));
      });
      this.barrier();
      return;
    }
    // Fallback: Hillis-Steele
    for (let stride = 1; stride < wgSize; stride *= 2) {
      this.barrier();
      const prev = smem.read(tid.sub(this.u32(stride)));
      const curr = smem.read(tid);
      const merged = op === "sum" ? curr.add(prev) : curr.max(prev);
      this.ifThen(tid.ge(this.u32(stride)), () => {
        smem.write(tid, merged);
      });
    }
    this.barrier();
  }

  // -- In-kernel RNG (Philox 2x32-10) --

  /**
   * Compute mulhi(a, b) — high 32 bits of a*b — via 16-bit decomposition.
   * Returns a BlockExpr for the result. Uses emitLet to keep WGSL readable.
   */
  private mulhi(prefix: string, a: BlockExpr, b: BlockExpr): BlockExpr {
    const aL = this.emitLet(`${prefix}_aL`, a.and(0xFFFF));
    const aH = this.emitLet(`${prefix}_aH`, a.shr(16));
    const bL = this.emitLet(`${prefix}_bL`, b.and(0xFFFF));
    const bH = this.emitLet(`${prefix}_bH`, b.shr(16));
    const ll = this.emitLet(`${prefix}_ll`, aL.mul(bL));
    const lh = this.emitLet(`${prefix}_lh`, aL.mul(bH));
    const hl = this.emitLet(`${prefix}_hl`, aH.mul(bL));
    const hh = this.emitLet(`${prefix}_hh`, aH.mul(bH));
    const mid = this.emitLet(`${prefix}_mid`, lh.add(hl));
    const carry = mid.lt(lh).select(this.u32(1), this.u32(0));
    const t = this.emitLet(`${prefix}_t`, ll.shr(16).add(mid.and(0xFFFF)));
    return hh.add(mid.shr(16)).add(carry).add(t.shr(16));
  }

  /**
   * Philox 2x32-10 counter-based PRNG. Returns two pseudorandom u32 values.
   *
   * Usage: `const [r0, r1] = ctx.philox2x32(seed, offset);`
   * - seed: a u32 key (e.g., from a uniform)
   * - offset: a u32 counter (e.g., globalId(0))
   *
   * The two outputs are independent random streams.
   */
  philox2x32(seed: BlockExpr, offset: BlockExpr): [BlockExpr, BlockExpr] {
    const PHILOX_M = 0xD256D193;  // Philox 2x32 multiplier
    const PHILOX_W = 0x9E3779B9;  // Key schedule constant (golden ratio)

    // Initial state
    let ctr0 = this.emitLet("_phi_c0", offset);
    let ctr1 = this.emitLet("_phi_c1", seed);
    let key = this.emitLet("_phi_k", seed);

    // 10 rounds of Philox mixing
    for (let r = 0; r < 10; r++) {
      const hi = this.mulhi(`_phi_r${r}`, this.u32(PHILOX_M), ctr0);
      const lo = this.emitLet(`_phi_lo${r}`, this.u32(PHILOX_M).mul(ctr0));
      ctr0 = this.emitLet(`_phi_c0_${r}`, hi.xor(key).xor(ctr1));
      ctr1 = this.emitLet(`_phi_c1_${r}`, lo);
      if (r < 9) {
        key = this.emitLet(`_phi_k${r}`, key.add(this.u32(PHILOX_W)));
      }
    }

    return [ctr0, ctr1];
  }

  /** Generate a uniform random f32 in [0, 1) from Philox 2x32-10. */
  randF32(seed: BlockExpr, offset: BlockExpr): BlockExpr {
    return this.randF32x2(seed, offset)[0];
  }

  /** Generate two uniform random f32s in [0, 1) from a single Philox call. */
  randF32x2(seed: BlockExpr, offset: BlockExpr): [BlockExpr, BlockExpr] {
    const [r0, r1] = this.philox2x32(seed, offset);
    const scale = this.f32(2.3283064365386963e-10); // 1.0 / 2^32
    return [r0.toF32().mul(scale), r1.toF32().mul(scale)];
  }

  // ---- Triton-like Tile API (delegates to internal BlockOps) ----

  /** Internal BlockOps instance, lazily created on first use. */
  private _blockOps?: BlockOps;
  /** Thread tile config, set by setThreadTile() or configureTiles(). */
  private _threadTile?: [number, number];

  private _requireOps(): BlockOps {
    if (!this._blockOps) {
      if (!this._wgSize) throw new Error("Kernel spec must have workgroupSize for tile operations");
      this._blockOps = new BlockOps(this, {
        wgSize: this._wgSize,
        ...(this._threadTile ? { threadTile: this._threadTile } : {}),
      });
    }
    return this._blockOps;
  }

  /**
   * Set thread tile dimensions for dot/store operations.
   * Call this before dotAccum() or tileStore() if not using configureTiles().
   */
  setThreadTile(threadTileM: number, threadTileN: number): void {
    this._threadTile = [threadTileM, threadTileN];
    if (this._blockOps) {
      this._blockOps.setThreadTile(threadTileM, threadTileN);
    }
  }

  /**
   * Initialize tile context: set thread tile, emit thread position bindings.
   * Returns threadRow/threadCol for use in epilogue addressing.
   *
   * Sugar for setThreadTile() + emitLet("thread_row"/thread_col").
   */
  configureTiles(config: {
    threadTileM: number; threadTileN: number;
  }): { threadRow: BlockExpr; threadCol: BlockExpr } {
    this.setThreadTile(config.threadTileM, config.threadTileN);
    const threadRow = this.emitLet("thread_row", this.threadIdx(1));
    const threadCol = this.emitLet("thread_col", this.threadIdx(0));
    return { threadRow, threadCol };
  }

  /** Access the underlying BlockOps (for epilogue callbacks). */
  get tileOps(): BlockOps {
    return this._requireOps();
  }

  /** Create a 1D offset range: [base, base+blockSize). ≈ tl.arange(base, base+size) */
  arange(base: BlockExpr, blockSize: number): TileRange {
    return this._requireOps().arange(base, blockSize);
  }

  /** Build a 2D pointer from base + outer + inner. ≈ Triton pointer arithmetic */
  tilePtr(base: BlockExpr, outer: TilePtrComponent, inner: TilePtrComponent): TilePtr {
    return buildPtr(base, outer, inner);
  }

  /** Build a 2D mask from outer + inner bounds. ≈ Triton mask construction */
  tileMask(outer: TileMaskComponent, inner: TileMaskComponent): TileMask {
    return buildMask(outer, inner);
  }

  /** Create a register block initialized to zero. ≈ tl.zeros([M, N]) */
  zeros(rows: number, cols: number): Block {
    return this._requireOps().zeros(rows, cols);
  }

  /** Create a register block filled with a constant. ≈ tl.full([M, N], val) */
  full(rows: number, cols: number, val: number): Block {
    return this._requireOps().full(rows, cols, val);
  }

  /** Cooperative tile load into shared memory. ≈ tl.load(ptr, mask) */
  load2D(binding: string, ptr: TilePtr, mask: TileMask): Block {
    return this._requireOps().loadTile(binding, ptr, mask);
  }

  /** 1D register load (e.g. bias vector). ≈ tl.load(bias_ptr) */
  load1D(binding: string, range: TileRange): Block {
    return this._requireOps().load1d(binding, range);
  }

  /** Block dot product: acc += a @ b. ≈ acc += tl.dot(a, b) */
  dotAccum(a: Block, b: Block, acc: Block): void {
    this._requireOps().dotAccum(a, b, acc);
  }

  /** Store block to global memory with bounds checking. ≈ tl.store(ptr, block, mask) */
  store2D(binding: string, block: Block, ptr: TilePtr, mask: TileMask): void {
    this._requireOps().storeTile(binding, block, ptr, mask);
  }

  // ---- Unified Block API (delegates to BlockOps) ----

  /** Block dot product: returns new result block. ≈ tl.dot(a, b) */
  dot(a: Block, b: Block): Block {
    return this._requireOps().dot(a, b);
  }

  /** Unified tile load: ptr type determines register vs shared placement. ≈ tl.load(ptr, mask) */
  tileLoad(binding: string, ptr: BlockPtr, opts: BlockLoadOpts): Block {
    return this._requireOps().load(binding, ptr, opts);
  }

  /** Store register tile to global memory. ≈ tl.store(ptr, block, mask) */
  tileStore(binding: string, block: Block, ptr: BlockStorePtr, opts?: { guard?: BlockExpr }): void {
    this._requireOps().store(binding, block, ptr, opts);
  }

  /** Load from strided layout: combined stridedIndex + load. */
  stridedLoad(binding: string, flatIdx: BlockExpr, indexShape: number[], strides: number[], baseOffset = 0): BlockExpr {
    return this.load(binding, this.stridedIndex(flatIdx, indexShape, strides, baseOffset));
  }

  // ---- Internal node creation helpers (for Block.get/set with proper node IDs) ----

  /** @internal Create a properly ID'd arrayRead node. */
  _makeArrayRead(arrayName: string, idx: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<ArrayReadNode>({
      kind: "arrayRead", arrayName, idx: idx.node,
      valueType: "scalar", dataType: "f32",
    }));
  }

  /** @internal Create a properly ID'd sharedRead node. */
  _makeSharedRead(arrayName: string, idx: BlockExpr): BlockExpr {
    return new BlockExpr(makeNode<SharedReadNode>({
      kind: "sharedRead", arrayName, idx: idx.node,
      valueType: "scalar", dataType: "f32",
    }));
  }

}

// ============================================================================
// Kernel Specification
// ============================================================================

export interface BindingSpec {
  storage: "read" | "read_write" | "atomic";
  type: DataType;
}

export type UniformType = "f32" | "u32" | "i32";

export interface TileKernelSpec {
  name: string;
  /** Workgroup size: scalar for 1D, [x, y] for 2D. */
  workgroupSize: number | [number, number];
  bindings: Record<string, BindingSpec>;
  uniforms: Record<string, UniformType>;
  /** WGSL `const` declarations emitted at module scope. */
  constants?: Record<string, number>;
  /** Whether to emit `enable f16;`. */
  enableF16?: boolean;
  /** Whether to emit `enable subgroups;`. */
  enableSubgroups?: boolean;
  /** Auto-vectorization width for elementwise kernels. When set (e.g., 4),
   *  each thread processes `vectorize` elements. GlobalId(0) is replaced with
   *  `gid.x * vectorize + offset` and the body is unrolled. Grid must be
   *  adjusted by the caller: `ceil(N / (workgroupSize * vectorize))`. */
  vectorize?: number;
  /** Binding index for the uniform struct. If set, uniform is inserted at this
   *  index among the storage bindings (e.g. 3 means after 3 storage bindings).
   *  If unset, uniform is appended after all storage bindings. */
  uniformBindingIndex?: number;
  /** When true, the compiler automatically inserts workgroupBarrier() calls
   *  between shared memory writes and subsequent reads. Opt-in to avoid
   *  double-barriering existing kernels with manual barriers. */
  autoBarriers?: boolean;
  /** Compute grid dimensions from uniform values.
   *  If omitted, auto-inferred from uniforms: a u32 uniform named "size",
   *  "total_elements", "num_elements", or "outSize" triggers elementwiseGrid. */
  grid?: GridFn;
  /** The kernel function that builds the IR DAG. */
  kernel: (ctx: KernelContext) => void;
}

// ============================================================================
// Grid Helpers
// ============================================================================

export type GridFn = (uniforms: Record<string, number>) => [number] | [number, number] | [number, number, number];

/**
 * Grid for 1-thread-per-element kernels with optional vectorization
 * and automatic 2D overflow when workgroups exceed 65535.
 */
export function elementwiseGrid(
  workgroupSize: number,
  opts?: { vecWidth?: number; elementUniform?: string },
): GridFn {
  const vw = opts?.vecWidth ?? 1;
  const uName = opts?.elementUniform ?? "total_elements";
  return (u) => {
    const totalWg = Math.ceil(u[uName] / (workgroupSize * vw));
    if (totalWg <= MAX_WORKGROUPS_PER_DIM) return [totalWg];
    return [Math.min(totalWg, MAX_WORKGROUPS_PER_DIM), Math.ceil(totalWg / MAX_WORKGROUPS_PER_DIM)];
  };
}

/** Grid for 1-workgroup-per-row kernels (e.g. LayerNorm, cross-entropy). */
export function perRowGrid(rowUniform?: string): GridFn {
  const name = rowUniform ?? "num_rows";
  return (u) => [u[name]];
}

/** Grid for simple 1D ceil-division dispatch. */
export function ceilDivGrid(divisor: number, elementUniform?: string): GridFn {
  const name = elementUniform ?? "total_elements";
  return (u) => [Math.ceil(u[name] / divisor)];
}

/** Grid that dispatches a single workgroup. */
export function singleWorkgroup(): GridFn {
  return () => [1];
}

/**
 * Tiled grid for multi-dimensional dispatch (e.g. attention, batched ops).
 * Each dimension is either:
 * - `{ uniform: string, tileSize: number }` — ceil(uniform / tileSize) workgroups
 * - `string` — raw uniform value (1 workgroup per unit, e.g. num_heads, batch_size)
 *
 * Example: `tiledGrid({ x: { uniform: "seq_len", tileSize: 16 }, y: "num_heads", z: "batch_size" })`
 * → `(u) => [ceil(seq_len/16), num_heads, batch_size]`
 */
export function tiledGrid(dims: {
  x: string | { uniform: string; tileSize: number };
  y?: string | { uniform: string; tileSize: number };
  z?: string | { uniform: string; tileSize: number };
}): GridFn {
  function resolve(dim: string | { uniform: string; tileSize: number }, u: Record<string, number>): number {
    if (typeof dim === "string") return u[dim];
    return Math.ceil(u[dim.uniform] / dim.tileSize);
  }
  return (u) => {
    const x = resolve(dims.x, u);
    if (dims.z !== undefined) return [x, resolve(dims.y!, u), resolve(dims.z, u)];
    if (dims.y !== undefined) return [x, resolve(dims.y, u)];
    return [x];
  };
}

/**
 * Flat 1D grid from the product of multiple uniforms.
 * Example: `productGrid("batch_size", "num_heads", "seq_len")`
 * → `(u) => [batch_size * num_heads * seq_len]`
 */
export function productGrid(...uniformNames: string[]): GridFn {
  return (u) => {
    let total = 1;
    for (const name of uniformNames) total *= u[name];
    return [total];
  };
}

// ============================================================================
// Kernel Factories
// ============================================================================

const DEFAULT_WG = 256;

/**
 * Factory for elementwise 1-thread-per-element kernels.
 * Absorbs the workgroupSize + elementwiseGrid + elementIndex boilerplate.
 *
 * Automatically provides: workgroupSize (256), `size` uniform (u32),
 * elementwiseGrid, and bounds-checked element index.
 *
 * ```typescript
 * const spec = elementwiseKernel({
 *   name: "fill",
 *   bindings: { out: { storage: "read_write", type: "f32" } },
 *   uniforms: { value: "f32" },
 *   kernel(ctx, idx) {
 *     ctx.emitStore("out", idx, ctx.uniform("value"));
 *   },
 * });
 * ```
 */
export function elementwiseKernel(config: {
  name: string;
  bindings: Record<string, BindingSpec>;
  uniforms?: Record<string, UniformType>;
  enableF16?: boolean;
  constants?: Record<string, number>;
  /** Override the element count uniform name (default: "size"). */
  sizeUniform?: string;
  kernel: (ctx: KernelContext, idx: BlockExpr) => void;
}): TileKernelSpec {
  const sizeU = config.sizeUniform ?? "size";
  return {
    name: config.name,
    workgroupSize: DEFAULT_WG,
    bindings: config.bindings,
    uniforms: { [sizeU]: "u32" as UniformType, ...config.uniforms },
    enableF16: config.enableF16,
    constants: config.constants,
    grid: elementwiseGrid(DEFAULT_WG, { elementUniform: sizeU }),
    kernel(ctx) {
      const idx = ctx.elementIndex(DEFAULT_WG, sizeU);
      config.kernel(ctx, idx);
    },
  };
}

/**
 * Factory for per-row reduction kernels (1 workgroup per row).
 * Absorbs the perRowGrid + row/tid/D/base setup boilerplate.
 *
 * Automatically provides: workgroupSize (256), `num_rows` + `feature_dim` uniforms,
 * perRowGrid, and computed row/tid/D/base values.
 *
 * ```typescript
 * const spec = perRowKernel({
 *   name: "softmax",
 *   bindings: { input: { storage: "read", type: "f32" }, output: { storage: "read_write", type: "f32" } },
 *   kernel(ctx, row, tid, D, base) {
 *     const maxVal = ctx.wgReduce("max", tid, D, 256, (i) => ctx.load("input", base.add(i)));
 *     // ...
 *   },
 * });
 * ```
 */
export function perRowKernel(config: {
  name: string;
  bindings: Record<string, BindingSpec>;
  uniforms?: Record<string, UniformType>;
  enableF16?: boolean;
  constants?: Record<string, number>;
  /** Override the row count uniform name (default: "num_rows"). */
  rowUniform?: string;
  /** Override the feature dim uniform name (default: "feature_dim"). */
  dimUniform?: string;
  kernel: (ctx: KernelContext, row: BlockExpr, tid: BlockExpr,
           D: BlockExpr, base: BlockExpr) => void;
}): TileKernelSpec {
  const rowU = config.rowUniform ?? "num_rows";
  const dimU = config.dimUniform ?? "feature_dim";
  return {
    name: config.name,
    workgroupSize: DEFAULT_WG,
    bindings: config.bindings,
    uniforms: { [rowU]: "u32" as UniformType, [dimU]: "u32" as UniformType, ...config.uniforms },
    enableF16: config.enableF16,
    constants: config.constants,
    grid: perRowGrid(rowU),
    kernel(ctx) {
      const row  = ctx.programId(0);
      const tid  = ctx.localIndex();
      const D    = ctx.uniform(dimU);
      const base = row.mul(D);
      config.kernel(ctx, row, tid, D, base);
    },
  };
}

// Well-known elementwise uniform names for auto-inference
const ELEMENTWISE_UNIFORM_NAMES = new Set([
  "size", "total_elements", "num_elements", "outSize",
]);

/**
 * Infer a grid function from the spec's uniform declarations.
 * Returns null if no auto-inference pattern matches.
 */
export function inferGrid(spec: TileKernelSpec): GridFn | null {
  const wgSize = typeof spec.workgroupSize === "number"
    ? spec.workgroupSize : spec.workgroupSize[0] * spec.workgroupSize[1];
  for (const [name, type] of Object.entries(spec.uniforms)) {
    if (type === "u32" && ELEMENTWISE_UNIFORM_NAMES.has(name)) {
      return elementwiseGrid(wgSize, { elementUniform: name });
    }
  }
  return null;
}

/**
 * Resolve the grid function for a spec: use explicit grid, fall back to inference.
 * Throws if neither explicit nor inferred grid is available.
 */
export function resolveGrid(spec: TileKernelSpec): GridFn {
  if (spec.grid) return spec.grid;
  const inferred = inferGrid(spec);
  if (inferred) return inferred;
  throw new Error(`tile-ir: no grid function for kernel "${spec.name}" and no auto-inference matched. ` +
    `Add an explicit \`grid\` or use a well-known uniform name (${[...ELEMENTWISE_UNIFORM_NAMES].join(", ")}).`);
}

// ============================================================================
// Autotuning Types
// ============================================================================

/** Tunable parameter definition. */
export type TuneParam = {
  values: number[];    // Candidate values (e.g. [32, 64, 128])
  default: number;     // Default when autotuning disabled
};

/** A factory that generates a kernel spec from tunable config values. */
export type TileKernelSpecFactory = (config: Record<string, number>) => TileKernelSpec;

/** Autotuning metadata attached to a factory. */
export interface AutotuneConfig {
  factory: TileKernelSpecFactory;
  params: Record<string, TuneParam>;
  /** Constraints: functions that return true if a config combo is valid. */
  constraints?: Array<(config: Record<string, number>) => boolean>;
  /** Optional: narrow param space based on runtime uniform values. */
  pruneForShape?: (uniforms: Record<string, number>) => Record<string, TuneParam>;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Execute a kernel spec's kernel function and return the captured context.
 * This builds the IR DAG without generating WGSL.
 */
export function buildKernelIR(spec: TileKernelSpec, subgroupSize = 0): KernelContext {
  nextNodeId = 0;
  cseCache = new Map();
  try {
    const ctx = new KernelContext(spec.bindings, subgroupSize);
    ctx._setWgSize(spec.workgroupSize);
    spec.kernel(ctx);
    return ctx;
  } finally {
    // Don't restore — each compilation gets fresh IDs
  }
}
