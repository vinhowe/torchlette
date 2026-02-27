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
 * Also supports Block-ops tile statements (tileLoad, tileStore, dot, etc.)
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

// ============================================================================
// IR Node Types
// ============================================================================

export type ValueType = "block" | "scalar";
export type DataType = "f32" | "f16" | "u32" | "i32";
export type BinaryOp = "add" | "sub" | "mul" | "div" | "mod" | "and" | "or" | "shr" | "shl" | "min" | "max" | "pow";
export type UnaryOp = "rsqrt" | "exp" | "log" | "abs" | "neg" | "sqrt" | "tanh" | "floor" | "ceil" | "not" | "sin" | "cos" | "round" | "sign" | "exp2" | "log2";
export type CmpOp = "eq" | "ne" | "lt" | "le" | "gt" | "ge";

export interface IRNodeBase {
  id: number;
  valueType: ValueType;
  dataType: DataType;
}

export interface ProgramIdNode extends IRNodeBase {
  kind: "programId";
  dim: number; // 0=x, 1=y, 2=z
}

export interface UniformNode extends IRNodeBase {
  kind: "uniform";
  name: string;
}

export interface ConstNode extends IRNodeBase {
  kind: "const";
  value: number;
}

export interface LoadNode extends IRNodeBase {
  kind: "load";
  binding: string;
  offsets: IRNode; // block expression for indices
  mask?: IRNode;   // optional mask (block expression, truthy = load, falsy = 0)
}


export interface BinaryNode extends IRNodeBase {
  kind: "binary";
  op: BinaryOp;
  lhs: IRNode;
  rhs: IRNode;
}

export interface UnaryNode extends IRNodeBase {
  kind: "unary";
  op: UnaryOp;
  input: IRNode;
}

export interface CastNode extends IRNodeBase {
  kind: "cast";
  input: IRNode;
  targetType: DataType;
}

export interface SelectNode extends IRNodeBase {
  kind: "select";
  condition: IRNode;
  trueVal: IRNode;
  falseVal: IRNode;
}

export interface CmpNode extends IRNodeBase {
  kind: "cmp";
  op: CmpOp;
  lhs: IRNode;
  rhs: IRNode;
}

export interface BitcastNode extends IRNodeBase {
  kind: "bitcast";
  input: IRNode;
  targetType: DataType;
}

// -- Imperative mode nodes --

export interface ThreadIdxNode extends IRNodeBase {
  kind: "threadIdx";
  dim: number; // 0=x, 1=y, 2=z
}

export interface LocalIndexNode extends IRNodeBase {
  kind: "localIndex";
}

export interface SharedReadNode extends IRNodeBase {
  kind: "sharedRead";
  arrayName: string;
  idx: IRNode;
}

export interface NamedRefNode extends IRNodeBase {
  kind: "namedRef";
  name: string;
}

export interface ArrayReadNode extends IRNodeBase {
  kind: "arrayRead";
  arrayName: string;
  idx: IRNode;
}

export interface GlobalIdNode extends IRNodeBase {
  kind: "globalId";
  dim: number; // 0=x, 1=y, 2=z
}

/** Number of workgroups dispatched in a given dimension (like Triton's `tl.num_programs`). */
export interface NumWorkgroupsNode extends IRNodeBase {
  kind: "numWorkgroups";
  dim: number; // 0=x, 1=y, 2=z
}

export interface SubgroupShuffleXorNode extends IRNodeBase {
  kind: "subgroupShuffleXor";
  value: IRNode;
  mask: IRNode;
}

/** dot(vec4<f32>(a0,a1,a2,a3), vec4<f32>(b0,b1,b2,b3)) → f32 scalar */
export interface Vec4DotNode extends IRNodeBase {
  kind: "vec4dot";
  a: [IRNode, IRNode, IRNode, IRNode];
  b: [IRNode, IRNode, IRNode, IRNode];
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
  | Vec4DotNode
  | NumWorkgroupsNode;

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

/** Accumulator operation kind. */
export type AccOpKind =
  | { kind: "mulScalar"; value: IRNode }
  | { kind: "addRow"; valuesArray: string; size: number }
  | { kind: "apply"; body: Statement[]; resultNode: IRNode; valName: string }
  | { kind: "castTo"; dtype: DataType };

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

export interface DotStmt {
  kind: "dot";
  aTile: { sharedName: string; rows: number; innerDim: number };
  bTile: { sharedName: string; innerDim: number; cols: number };
  accName: string;
  threadTileM: number;
  threadTileN: number;
}

export interface AccOpStmt {
  kind: "accOp";
  accName: string;
  threadTileM: number;
  threadTileN: number;
  op: AccOpKind;
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
  | IfStmt
  | IfElseStmt
  | BarrierStmt
  | SharedWriteStmt
  | GuardedStoreStmt
  | DirectStoreStmt
  | AtomicOpStmt
  // Tile-level statements (lowered by tile compiler)
  | TileLoadStmt
  | DotStmt
  | AccOpStmt
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

export interface ReturnStmt {
  kind: "return";
}

export interface LetStmt {
  kind: "let";
  name: string;
  value: IRNode;
  dtype: DataType;
}

export interface VarStmt {
  kind: "var";
  name: string;
  value: IRNode;
  dtype: DataType;
}

export interface VarArrayStmt {
  kind: "varArray";
  name: string;
  elemType: DataType;
  size: number;
  skipZeroInit?: boolean;
}

export interface AssignStmt {
  kind: "assign";
  name: string;
  value: IRNode;
}

export interface AddAssignStmt {
  kind: "addAssign";
  name: string;
  value: IRNode;
}

export interface IndexAssignStmt {
  kind: "indexAssign";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
}

export interface IndexAddAssignStmt {
  kind: "indexAddAssign";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
}

export interface ForRangeStmt {
  kind: "forRange";
  varName: string;
  start: IRNode;
  bound: IRNode;
  body: Statement[];
}

export interface IfStmt {
  kind: "if";
  condition: IRNode;
  body: Statement[];
}

export interface IfElseStmt {
  kind: "ifElse";
  condition: IRNode;
  body: Statement[];
  elseBody: Statement[];
}

export interface BarrierStmt {
  kind: "barrier";
}

export interface SharedWriteStmt {
  kind: "sharedWrite";
  arrayName: string;
  idx: IRNode;
  value: IRNode;
}

export interface GuardedStoreStmt {
  kind: "guardedStore";
  binding: string;
  condition: IRNode;
  idx: IRNode;
  value: IRNode;
}

export interface DirectStoreStmt {
  kind: "directStore";
  binding: string;
  idx: IRNode;
  value: IRNode;
}

export type AtomicOp = "max" | "min" | "add" | "or" | "and";

export interface AtomicOpStmt {
  kind: "atomicOp";
  binding: string;
  idx: IRNode;
  op: AtomicOp;
  value: IRNode;
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
    default: return null; // Not CSE-eligible (loads, sharedRead, namedRef, etc.)
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

/** Resolve a BlockExpr or number into an IRNode. Numbers become f32 constants. */
function resolveArg(arg: BlockExpr | number): IRNode {
  if (typeof arg === "number") {
    return makeNode<ConstNode>({
      kind: "const",
      valueType: "scalar",
      dataType: "f32",
      value: arg,
    });
  }
  return arg.node;
}

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

  // -- Arithmetic --
  add(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "add",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  sub(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "sub",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  mul(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "mul",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  div(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "div",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  mod(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "mod",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  and(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "and",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  min(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "min",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  max(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "max",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  or(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "or",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  shr(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "shr",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  shl(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "shl",
      lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: this.node.dataType,
    }));
  }

  // -- Unary math --
  rsqrt(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "rsqrt", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  exp(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "exp", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  log(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "log", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  sqrt(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "sqrt", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  abs(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "abs", input: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  neg(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "neg", input: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  tanh(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "tanh", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  floor(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "floor", input: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  ceil(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "ceil", input: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  sin(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "sin", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  cos(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "cos", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  round(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "round", input: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  sign(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "sign", input: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  exp2(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "exp2", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  log2(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "log2", input: this.node,
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

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

  pow(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<BinaryNode>({
      kind: "binary", op: "pow", lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "f32",
    }));
  }

  // -- Type casts --
  toF32(): BlockExpr {
    if (this.node.dataType === "f32") return this;
    return new BlockExpr(makeNode<CastNode>({
      kind: "cast", input: this.node, targetType: "f32",
      valueType: this.node.valueType, dataType: "f32",
    }));
  }

  toU32(): BlockExpr {
    if (this.node.dataType === "u32") return this;
    return new BlockExpr(makeNode<CastNode>({
      kind: "cast", input: this.node, targetType: "u32",
      valueType: this.node.valueType, dataType: "u32",
    }));
  }

  toI32(): BlockExpr {
    if (this.node.dataType === "i32") return this;
    return new BlockExpr(makeNode<CastNode>({
      kind: "cast", input: this.node, targetType: "i32",
      valueType: this.node.valueType, dataType: "i32",
    }));
  }

  toF16(): BlockExpr {
    if (this.node.dataType === "f16") return this;
    return new BlockExpr(makeNode<CastNode>({
      kind: "cast", input: this.node, targetType: "f16",
      valueType: this.node.valueType, dataType: "f16",
    }));
  }

  // -- Bitcast (reinterpret bits, no conversion) --
  bitcastTo(dtype: DataType): BlockExpr {
    return new BlockExpr(makeNode<BitcastNode>({
      kind: "bitcast", input: this.node, targetType: dtype,
      valueType: this.node.valueType, dataType: dtype,
    }));
  }

  // -- Subgroup ops --
  subgroupShuffleXor(mask: BlockExpr | number): BlockExpr {
    const m = resolveArg(mask);
    return new BlockExpr(makeNode<SubgroupShuffleXorNode>({
      kind: "subgroupShuffleXor", value: this.node, mask: m,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

  // -- Comparisons --
  eq(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<CmpNode>({
      kind: "cmp", op: "eq", lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  ne(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<CmpNode>({
      kind: "cmp", op: "ne", lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  lt(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<CmpNode>({
      kind: "cmp", op: "lt", lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  le(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<CmpNode>({
      kind: "cmp", op: "le", lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  gt(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<CmpNode>({
      kind: "cmp", op: "gt", lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  ge(other: BlockExpr | number): BlockExpr {
    const rhs = resolveArg(other);
    return new BlockExpr(makeNode<CmpNode>({
      kind: "cmp", op: "ge", lhs: this.node, rhs,
      valueType: promoteValueType(this.node.valueType, rhs.valueType),
      dataType: "u32",
    }));
  }

  // -- Logical NOT --
  not(): BlockExpr {
    return new BlockExpr(makeNode<UnaryNode>({
      kind: "unary", op: "not", input: this.node,
      valueType: this.node.valueType, dataType: this.node.dataType,
    }));
  }

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
  /** Binding specs from the kernel spec (for dataType resolution in load). */
  private readonly bindingSpecs: Record<string, BindingSpec>;

  /** Statement stack for nested scopes (forRange, ifThen). */
  private stmtStack: Statement[][] = [];
  /** Counter for unique loop variable names. */
  private loopVarCounter = 0;
  /** Counter for unique wgReduce shared memory / accumulator names. */
  private reduceCounter = 0;

  constructor(bindings?: Record<string, BindingSpec>) {
    this.bindingSpecs = bindings ?? {};
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

  /** Emit a for loop: `for (var v = start; v < bound; v++)`. */
  forRange(start: BlockExpr, bound: BlockExpr, body: (loopVar: BlockExpr) => void): void {
    const varName = `_lv${this.loopVarCounter++}`;
    const bodyStmts: Statement[] = [];
    this.stmtStack.push(bodyStmts);
    const loopVar = new BlockExpr(makeNode<NamedRefNode>({
      kind: "namedRef", name: varName,
      valueType: "scalar", dataType: "u32",
    }));
    body(loopVar);
    this.stmtStack.pop();
    this.pushStatement({
      kind: "forRange",
      varName,
      start: start.node,
      bound: bound.node,
      body: bodyStmts,
    });
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

  /** Tree sum reduction: `smem[0] = sum(smem[0..wgSize-1])`. Emits log2(wgSize) if+barrier pairs. */
  treeReduceSum(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number): void {
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.ifThen(tid.lt(this.u32(stride)), () => {
        smem.write(tid, smem.read(tid).add(smem.read(tid.add(this.u32(stride)))));
      });
      this.barrier();
    }
  }

  /** Tree max reduction: `smem[0] = max(smem[0..wgSize-1])`. */
  treeReduceMax(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number): void {
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.ifThen(tid.lt(this.u32(stride)), () => {
        smem.write(tid, smem.read(tid).max(smem.read(tid.add(this.u32(stride)))));
      });
      this.barrier();
    }
  }

  /** Dual tree sum reduction: reduces two shared arrays in parallel. */
  dualTreeReduceSum(
    smem1: SharedArrayHandle, smem2: SharedArrayHandle,
    tid: BlockExpr, wgSize: number,
  ): void {
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.ifThen(tid.lt(this.u32(stride)), () => {
        smem1.write(tid, smem1.read(tid).add(smem1.read(tid.add(this.u32(stride)))));
        smem2.write(tid, smem2.read(tid).add(smem2.read(tid.add(this.u32(stride)))));
      });
      this.barrier();
    }
  }

  /** Strided loop: `for (var i = tid; i < bound; i += wgSize) { body(i); }` */
  stridedFor(
    tid: BlockExpr, bound: BlockExpr, wgSize: number,
    body: (i: BlockExpr) => void,
  ): void {
    const maxIters = bound.add(this.u32(wgSize - 1)).div(this.u32(wgSize));
    this.forRange(this.u32(0), maxIters, (iter) => {
      const i = tid.add(iter.mul(this.u32(wgSize)));
      this.ifThen(i.lt(bound), () => {
        body(i);
      });
    });
  }

  /**
   * Workgroup-cooperative reduction. Allocates shared memory, accumulates via
   * stridedFor, writes to shared memory, barriers, and tree-reduces.
   * Returns the reduced scalar (`smem[0]` after reduction).
   */
  wgReduce(
    op: "sum" | "max",
    tid: BlockExpr, count: BlockExpr, wgSize: number,
    bodyFn: (i: BlockExpr) => BlockExpr,
  ): BlockExpr {
    const id = this.reduceCounter++;
    const smem = this.sharedArray(`_wgr${id}_s`, wgSize);
    const identity = op === "sum" ? 0.0 : -3.402823e+38;
    const acc = this.emitVar(`_wgr${id}_a`, "f32", this.f32(identity));
    this.stridedFor(tid, count, wgSize, (i) => {
      if (op === "sum") {
        acc.addAssign(bodyFn(i));
      } else {
        acc.set(acc.get().max(bodyFn(i)));
      }
    });
    smem.write(tid, acc.get());
    this.barrier();
    if (op === "sum") {
      this.treeReduceSum(smem, tid, wgSize);
    } else {
      this.treeReduceMax(smem, tid, wgSize);
    }
    return smem.read(this.u32(0));
  }

  /**
   * Dual workgroup-cooperative sum reduction. Same as wgReduce but reduces
   * two values in parallel using a single stridedFor pass.
   * Returns `[smem1[0], smem2[0]]` after reduction.
   */
  dualWgReduce(
    tid: BlockExpr, count: BlockExpr, wgSize: number,
    bodyFn: (i: BlockExpr) => [BlockExpr, BlockExpr],
  ): [BlockExpr, BlockExpr] {
    const id = this.reduceCounter++;
    const smem1 = this.sharedArray(`_wgr${id}_s1`, wgSize);
    const smem2 = this.sharedArray(`_wgr${id}_s2`, wgSize);
    const acc1 = this.emitVar(`_wgr${id}_a1`, "f32", this.f32(0.0));
    const acc2 = this.emitVar(`_wgr${id}_a2`, "f32", this.f32(0.0));
    this.stridedFor(tid, count, wgSize, (i) => {
      const [v1, v2] = bodyFn(i);
      acc1.addAssign(v1);
      acc2.addAssign(v2);
    });
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

  /** Emit an unconditional scalar store: `binding[idx] = val;`. */
  emitStore(binding: string, idx: BlockExpr, val: BlockExpr): void {
    this.pushStatement({
      kind: "directStore",
      binding,
      idx: idx.node,
      value: val.node,
    });
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

  // -- Scan primitives (like `tl.cumsum`, `tl.associative_scan`) --

  /** Hillis-Steele inclusive parallel prefix scan in shared memory.
   *  After the call, smem[tid] = op(smem[0], ..., smem[tid]).
   *  Requires tid < wgSize. */
  inclusiveScan(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number, op: "sum" | "max"): void {
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

  /** Hillis-Steele exclusive parallel prefix scan in shared memory.
   *  After the call, smem[tid] = op(smem[0], ..., smem[tid-1]), smem[0] = identity.
   *  Identity: 0.0 for sum, -Infinity for max. */
  exclusiveScan(smem: SharedArrayHandle, tid: BlockExpr, wgSize: number, op: "sum" | "max"): void {
    // Run inclusive scan first
    this.inclusiveScan(smem, tid, wgSize, op);
    // Shift right: exclusive[i] = inclusive[i-1], exclusive[0] = identity
    const inclusive = smem.read(tid);
    this.barrier();
    const identity = op === "sum" ? this.f32(0.0) : this.f32(-1e38);
    const shifted = this.blockWhere(tid.gt(this.u32(0)), smem.read(tid.sub(this.u32(1))), identity);
    smem.write(tid, shifted);
    // Need barrier before anyone reads the shifted values
    this.barrier();
    // Suppress unused variable warning — inclusive was read before barrier
    void inclusive;
  }

  // -- Generic associative scan (like `tl.associative_scan`) --

  /** Hillis-Steele inclusive scan with a user-defined associative combine function.
   *  After the call, smem[tid] = combine(smem[0], combine(smem[1], ...smem[tid])).
   *  `combine(a, b)` must be associative. */
  associativeScan(
    smem: SharedArrayHandle,
    tid: BlockExpr,
    wgSize: number,
    combine: (a: BlockExpr, b: BlockExpr) => BlockExpr,
  ): void {
    for (let stride = 1; stride < wgSize; stride *= 2) {
      this.barrier();
      // All threads read their values into registers before any writes
      const curr = this.emitLet(`scan_c_${stride}`, smem.read(tid));
      const prev = this.emitLet(`scan_p_${stride}`, smem.read(tid.sub(this.u32(stride))));
      const merged = combine(prev, curr);
      this.barrier(); // ensure all reads complete before writes
      this.ifThen(tid.ge(this.u32(stride)), () => {
        smem.write(tid, merged);
      });
    }
    this.barrier();
  }

  // -- Argmax / Argmin reductions --

  /** Tree reduction finding the index of the maximum value.
   *  `valSmem[tid]` holds values, `idxSmem[tid]` holds corresponding indices.
   *  After the call, `valSmem[0]` = max value, `idxSmem[0]` = its index. */
  treeReduceArgmax(
    valSmem: SharedArrayHandle,
    idxSmem: SharedArrayHandle,
    tid: BlockExpr,
    wgSize: number,
  ): void {
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.barrier();
      this.ifThen(tid.lt(this.u32(stride)), () => {
        const myVal = this.emitLet(`argmax_my_${stride}`, valSmem.read(tid));
        const otherVal = this.emitLet(`argmax_other_${stride}`, valSmem.read(tid.add(this.u32(stride))));
        const myIdx = this.emitLet(`argmax_myidx_${stride}`, idxSmem.read(tid));
        const otherIdx = this.emitLet(`argmax_otheridx_${stride}`, idxSmem.read(tid.add(this.u32(stride))));
        const cond = otherVal.gt(myVal);
        valSmem.write(tid, cond.select(otherVal, myVal));
        idxSmem.write(tid, cond.select(otherIdx, myIdx));
      });
    }
  }

  /** Tree reduction finding the index of the minimum value.
   *  `valSmem[tid]` holds values, `idxSmem[tid]` holds corresponding indices.
   *  After the call, `valSmem[0]` = min value, `idxSmem[0]` = its index. */
  treeReduceArgmin(
    valSmem: SharedArrayHandle,
    idxSmem: SharedArrayHandle,
    tid: BlockExpr,
    wgSize: number,
  ): void {
    for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
      this.barrier();
      this.ifThen(tid.lt(this.u32(stride)), () => {
        const myVal = this.emitLet(`argmin_my_${stride}`, valSmem.read(tid));
        const otherVal = this.emitLet(`argmin_other_${stride}`, valSmem.read(tid.add(this.u32(stride))));
        const myIdx = this.emitLet(`argmin_myidx_${stride}`, idxSmem.read(tid));
        const otherIdx = this.emitLet(`argmin_otheridx_${stride}`, idxSmem.read(tid.add(this.u32(stride))));
        const cond = otherVal.lt(myVal);
        valSmem.write(tid, cond.select(otherVal, myVal));
        idxSmem.write(tid, cond.select(otherIdx, myIdx));
      });
    }
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
  /** Compute grid dimensions from uniform values. */
  grid: (uniforms: Record<string, number>) => [number] | [number, number] | [number, number, number];
  /** The kernel function that builds the IR DAG. */
  kernel: (ctx: KernelContext) => void;
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

/** Reset node ID counter (for testing). */
export function resetNodeIdCounter(): void {
  nextNodeId = 0;
}

/**
 * Execute a kernel spec's kernel function and return the captured context.
 * This builds the IR DAG without generating WGSL.
 */
export function buildKernelIR(spec: TileKernelSpec): KernelContext {
  nextNodeId = 0;
  cseCache = new Map();
  try {
    const ctx = new KernelContext(spec.bindings);
    spec.kernel(ctx);
    return ctx;
  } finally {
    // Don't restore — each compilation gets fresh IDs
  }
}
