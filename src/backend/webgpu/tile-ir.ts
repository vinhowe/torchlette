/**
 * Tile IR: Triton-Style Block-Level Kernel DSL
 *
 * Provides a symbolic expression builder for writing WebGPU compute kernels
 * at the workgroup (block) level. Two compilation modes:
 *
 * **Auto-phase mode** (original): The developer writes block/scalar expressions
 * with loads, reduces, and stores. The compiler discovers phases automatically
 * (scalar preamble → reduce loops → store loop).
 *
 * **Imperative mode** (new): The developer writes explicit statements (forRange,
 * barrier, sharedArray, etc.) for full control over 2D tiled algorithms like
 * matmul. Detected when `ctx.statements.length > 0`.
 *
 * Two expression types:
 * - Block: varies per thread (loaded data, blockRange). Compiles to loop bodies.
 * - Scalar: uniform across workgroup (programId, reduce result, uniform). Compiles to single statements.
 *
 * Usage (auto-phase):
 * ```typescript
 * const layerNormFwd = tileKernel({
 *   name: "layerNormFwd",
 *   workgroupSize: 256,
 *   bindings: { x: { storage: "read", type: "f32" }, ... },
 *   uniforms: { numRows: "u32", featureDim: "u32", eps: "f32" },
 *   kernel(ctx) {
 *     const row = ctx.programId(0);
 *     const D = ctx.uniform("featureDim");
 *     const offs = row.mul(D).add(ctx.blockRange(D));
 *     const xVals = ctx.load("x", offs);
 *     const mean = ctx.reduce(xVals, "sum").div(D.toF32());
 *     // ...
 *   },
 * });
 * ```
 *
 * Usage (imperative):
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
export type BinaryOp = "add" | "sub" | "mul" | "div" | "mod" | "and" | "min" | "max";
export type UnaryOp = "rsqrt" | "exp" | "log" | "abs" | "neg" | "sqrt" | "tanh" | "floor" | "ceil" | "not";
export type ReduceOp = "sum" | "max";
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

export interface BlockRangeNode extends IRNodeBase {
  kind: "blockRange";
  extent: IRNode; // dimension expression (scalar)
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
}

export interface StoreNode extends IRNodeBase {
  kind: "store";
  binding: string;
  offsets: IRNode; // block expression for indices
  values: IRNode;  // block expression for data
}

export interface ReduceNode extends IRNodeBase {
  kind: "reduce";
  input: IRNode; // block expression to reduce
  op: ReduceOp;
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

export type IRNode =
  | ProgramIdNode
  | BlockRangeNode
  | UniformNode
  | ConstNode
  | LoadNode
  | StoreNode
  | ReduceNode
  | BinaryNode
  | UnaryNode
  | CastNode
  | SelectNode
  | CmpNode
  | ThreadIdxNode
  | LocalIndexNode
  | SharedReadNode
  | NamedRefNode
  | ArrayReadNode;

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
  // Tile-level statements (lowered by tile compiler)
  | TileLoadStmt
  | DotStmt
  | AccOpStmt
  | TileLoad1DStmt
  | TileStoreStmt;

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

// ============================================================================
// Symbolic Expression Wrapper
// ============================================================================

let nextNodeId = 0;

function makeNode<T extends IRNode>(partial: Omit<T, "id">): T {
  return { ...partial, id: nextNodeId++ } as T;
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
  /** Store operations, in order (auto-phase mode). */
  readonly stores: StoreNode[] = [];
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

  /**
   * Strided block range: indices [tid, tid+WG, tid+2*WG, ...] up to extent.
   * This is the standard parallel-for pattern used by all workgroup kernels.
   */
  blockRange(extent: BlockExpr): BlockExpr {
    return new BlockExpr(this.trackNode(makeNode<BlockRangeNode>({
      kind: "blockRange", extent: extent.node,
      valueType: "block", dataType: "u32",
    })));
  }

  /** Load from a storage buffer at given offsets. */
  load(binding: string, offsets: BlockExpr): BlockExpr {
    const bindingType = this.bindingSpecs[binding]?.type ?? "f32";
    return new BlockExpr(this.trackNode(makeNode<LoadNode>({
      kind: "load", binding, offsets: offsets.node,
      valueType: "block", dataType: bindingType as DataType,
    })));
  }

  /** Store values to a storage buffer at block offsets (auto-phase mode). */
  store(binding: string, offsets: BlockExpr, values: BlockExpr): void {
    const node = this.trackNode(makeNode<StoreNode>({
      kind: "store", binding, offsets: offsets.node, values: values.node,
      valueType: "block", dataType: "f32",
    }));
    this.stores.push(node);
  }

  /**
   * Workgroup-level parallel reduction. Takes a block expression and returns
   * a scalar result available to all threads (via shared memory + barriers).
   */
  reduce(input: BlockExpr, op: ReduceOp): BlockExpr {
    if (input.node.valueType !== "block") {
      throw new Error("reduce() requires a block expression");
    }
    return new BlockExpr(this.trackNode(makeNode<ReduceNode>({
      kind: "reduce", input: input.node, op,
      valueType: "scalar", dataType: "f32",
    })));
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
}

// ============================================================================
// Kernel Specification
// ============================================================================

export interface BindingSpec {
  storage: "read" | "read_write";
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
  /** Binding index for the uniform struct. If set, uniform is inserted at this
   *  index among the storage bindings (e.g. 3 means after 3 storage bindings).
   *  If unset, uniform is appended after all storage bindings. */
  uniformBindingIndex?: number;
  /** Compute grid dimensions from uniform values. */
  grid: (uniforms: Record<string, number>) => [number] | [number, number] | [number, number, number];
  /** The kernel function that builds the IR DAG. */
  kernel: (ctx: KernelContext) => void;
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
  try {
    const ctx = new KernelContext(spec.bindings);
    spec.kernel(ctx);
    return ctx;
  } finally {
    // Don't restore — each compilation gets fresh IDs
  }
}
