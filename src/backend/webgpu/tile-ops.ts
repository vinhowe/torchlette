/**
 * Tile Ops: Triton-like block-level kernel API.
 *
 * The user writes at the block level — thinking in BLOCK_M × BLOCK_N tiles,
 * never about threads. The compiler (tile-compiler.ts) handles thread mapping,
 * shared memory, barriers, and cooperative loading.
 *
 * All methods push tile-level Statement nodes. No imperative code emission.
 */

import type {
  DataType,
  IRNode,
  Statement,
  TileRangeInfo,
  TilePtr2D,
  TileMask2D,
  BlockBinaryOp,
  BlockUnaryOp,
  BlockReduceOp,
} from "./tile-ir";
import { BlockExpr, KernelContext } from "./tile-ir";

// ============================================================================
// TileRange: 1D range of block offsets [base, base+size)
// ============================================================================

/** Component of a 2D pointer: a range broadcast to a specific dimension. */
export interface TilePtrComponent {
  range: TileRangeInfo;
  stride: IRNode;
  dim: "outer" | "inner";
}

/** Component of a 2D mask: a bound on one dimension. */
export interface TileMaskComponent {
  range: TileRangeInfo;
  bound: IRNode;
}

export class TileRange {
  constructor(
    readonly info: TileRangeInfo,
  ) {}

  /** Broadcast to outer (row) dimension: [:, None] * stride */
  outer(stride?: BlockExpr): TilePtrComponent {
    const defaultStride: IRNode = {
      id: -1, kind: "const", valueType: "scalar", dataType: "u32", value: 1,
    };
    return {
      range: this.info,
      stride: stride ? stride.node : defaultStride,
      dim: "outer",
    };
  }

  /** Broadcast to inner (col) dimension: [None, :] * stride (default stride=1) */
  inner(stride?: BlockExpr): TilePtrComponent {
    const defaultStride: IRNode = {
      id: -1, kind: "const", valueType: "scalar", dataType: "u32", value: 1,
    };
    return {
      range: this.info,
      stride: stride ? stride.node : defaultStride,
      dim: "inner",
    };
  }

  /** Range < bound: produces a mask component for this dimension. */
  lt(bound: BlockExpr): TileMaskComponent {
    return {
      range: this.info,
      bound: bound.node,
    };
  }
}

// ============================================================================
// TilePtr: 2D pointer block
// ============================================================================

export class TilePtr {
  constructor(readonly data: TilePtr2D) {}

  /** Advance pointer by blockDim * stride (e.g. a_ptrs += BLOCK_K * stride_ak) */
  advance(blockDim: number, stride: BlockExpr): TilePtr {
    // new base = old base + blockDim * stride
    const advanceNode: IRNode = {
      id: -1, kind: "binary", op: "add",
      lhs: this.data.baseOffset,
      rhs: {
        id: -1, kind: "binary", op: "mul",
        lhs: { id: -1, kind: "const", valueType: "scalar", dataType: "u32", value: blockDim },
        rhs: stride.node,
        valueType: "scalar", dataType: "u32",
      },
      valueType: "scalar", dataType: "u32",
    };
    return new TilePtr({
      ...this.data,
      baseOffset: advanceNode,
    });
  }
}

// ============================================================================
// TileMask: 2D mask
// ============================================================================

export class TileMask {
  constructor(readonly data: TileMask2D) {}
}

// ============================================================================
// Pointer construction helpers
// ============================================================================

/**
 * Build a 2D pointer from base + outer component + inner component.
 * Usage: `buildPtr(base, offsM.outer(strideAm), offsK.inner(strideAk))`
 */
export function buildPtr(
  base: BlockExpr,
  outer: TilePtrComponent,
  inner: TilePtrComponent,
): TilePtr {
  // Ensure outer is actually outer, inner is actually inner
  const outerComp = outer.dim === "outer" ? outer : inner;
  const innerComp = inner.dim === "inner" ? inner : outer;
  return new TilePtr({
    baseOffset: base.node,
    outerRange: outerComp.range,
    outerStride: outerComp.stride,
    innerRange: innerComp.range,
    innerStride: innerComp.stride,
  });
}

/**
 * Build a 2D mask from outer bound & inner bound.
 * Usage: `buildMask(offsM.lt(m), offsN.lt(n))`
 */
export function buildMask(
  outer: TileMaskComponent,
  inner: TileMaskComponent,
): TileMask {
  return new TileMask({
    outerRange: outer.range,
    outerBound: outer.bound,
    innerRange: inner.range,
    innerBound: inner.bound,
  });
}

// ============================================================================
// Block API: Unified block type with automatic placement
// ============================================================================

/** Per-thread pointer: each thread loads its own rows → register placement */
export interface BlockThreadPtr {
  kind: "thread";
  base: BlockExpr;       // starting global offset for this thread
  stride: BlockExpr;     // row stride in global buffer
}

/** Cooperative pointer: all threads share the load → shared memory placement */
export interface BlockCoopPtr {
  kind: "tile";
  baseOffset: BlockExpr;
  outerRange: TileRangeInfo;
  innerRange: TileRangeInfo;
  outerStride: BlockExpr;
  innerStride: BlockExpr;
  outerBound: BlockExpr;
  innerBound: BlockExpr;
}

export type BlockPtr = BlockThreadPtr | BlockCoopPtr;

export interface BlockLoadOpts {
  rows: number;
  cols: number;
  guard?: BlockExpr;
}

/** Pointer for storing register data back to global memory. */
export interface BlockStorePtr {
  base: BlockExpr;
  stride: BlockExpr;
}

/**
 * Unified Block type: 2D register or shared memory tile.
 *
 * Placement is decided by how the block was created:
 * - `ops.load(binding, threadPtr, ...)` → register
 * - `ops.load(binding, tilePtr, ...)` → shared
 * - `ops.zeros(...)` / `ops.full(...)` → register
 * - Arithmetic/unary/reduce results → register
 *
 * The compiler uses placement + shapes to decide lowering strategy.
 */
export class Block {
  /** @internal */ _transposed = false;
  /** @internal Original rows before any T() call (for storage indexing) */
  readonly _origRows: number;
  /** @internal Original cols before any T() call (for storage indexing) */
  readonly _origCols: number;
  /** @internal Padded column stride for shared memory (cols + pad). Register blocks use cols. */
  readonly smemStride: number;

  constructor(
    readonly placement: "register" | "shared",
    readonly rows: number,
    readonly cols: number,
    readonly name: string,
    private readonly ctx: KernelContext,
    private readonly ops: BlockOps,
    origRows?: number,
    origCols?: number,
    smemStride?: number,
  ) {
    this._origRows = origRows ?? rows;
    this._origCols = origCols ?? cols;
    this.smemStride = smemStride ?? cols;
  }

  get transposed(): boolean { return this._transposed; }

  // ---- Transpose (metadata only, no data movement) ----
  T(): Block {
    const b = new Block(
      this.placement, this.cols, this.rows, this.name,
      this.ctx, this.ops, this._origRows, this._origCols, this.smemStride,
    );
    b._transposed = !this._transposed;
    return b;
  }

  // ---- Arithmetic (returns new register Block) ----
  add(other: Block | BlockExpr): Block { return this.ops._binary(this, other, "add"); }
  sub(other: Block | BlockExpr): Block { return this.ops._binary(this, other, "sub"); }
  mul(other: Block | BlockExpr): Block { return this.ops._binary(this, other, "mul"); }
  div(other: Block | BlockExpr): Block { return this.ops._binary(this, other, "div"); }

  // ---- Elementwise max (Block) or Reduction (axis number) ----
  max(arg: number | Block): Block {
    if (typeof arg === "number") {
      return this.ops._reduce(this, arg, "max");
    }
    return this.ops._binary(this, arg, "max");
  }

  sum(axis: number): Block {
    return this.ops._reduce(this, axis, "sum");
  }

  // ---- In-place variants ----
  mul_(other: Block | BlockExpr): void { this.ops._binary(this, other, "mul", true); }
  add_(other: Block | BlockExpr): void { this.ops._binary(this, other, "add", true); }
  sub_(other: Block | BlockExpr): void { this.ops._binary(this, other, "sub", true); }
  exp_(): void { this.ops._unary(this, "exp", true); }

  /** Copy values from other into this block (with broadcasting). */
  assign(other: Block): void { this.ops._binary(this, other, "copy", true); }

  /** Accumulate: this += other (with broadcasting). */
  addAssign(other: Block): void { this.ops._binary(this, other, "add", true); }

  // ---- Unary (returns new register Block) ----
  exp(): Block { return this.ops._unary(this, "exp"); }
  log(): Block { return this.ops._unary(this, "log"); }
  neg(): Block { return this.ops._unary(this, "neg"); }

  /**
   * Apply an elementwise function in-place: block[i] = fn(block[i]).
   * Used for epilogue ops (relu, gelu, etc.) on register blocks.
   */
  apply_(fn: (val: BlockExpr) => BlockExpr): void {
    const size = this.rows * this.cols;
    const valName = `_apply_val_${this.name}`;
    const valRef = new BlockExpr({
      id: -1, kind: "namedRef", name: valName,
      valueType: "scalar", dataType: "f32",
    } as any);
    // Capture the function body as IR statements
    const body = this.ctx.captureScope(() => {
      const result = fn(valRef);
      this.ctx.pushStatement({
        kind: "assign", name: `_apply_result_${this.name}`, value: result.node,
      } as any);
    });
    // Extract result node from the synthetic assign
    const lastStmt = body[body.length - 1];
    const resultNode = lastStmt && lastStmt.kind === "assign" ? (lastStmt as any).value : valRef.node;
    if (lastStmt && lastStmt.kind === "assign" && (lastStmt as any).name === `_apply_result_${this.name}`) {
      body.pop();
    }
    // Emit forRange over all elements
    const iVar = `_apply_i_${this.name}`;
    const innerBody: Statement[] = [];
    // let valName = block[i]
    innerBody.push({
      kind: "let", name: valName, dtype: "f32" as any,
      value: { id: -1, kind: "arrayRead", arrayName: this.name, idx: { id: -1, kind: "namedRef", name: iVar, valueType: "scalar", dataType: "u32" }, valueType: "scalar", dataType: "f32" } as any,
    });
    innerBody.push(...body);
    // block[i] = result
    innerBody.push({
      kind: "indexAssign",
      arrayName: this.name,
      idx: { id: -1, kind: "namedRef", name: iVar, valueType: "scalar", dataType: "u32" } as any,
      value: resultNode,
    });
    this.ctx.pushStatement({
      kind: "forRange",
      varName: iVar,
      start: { id: -1, kind: "const", valueType: "scalar", dataType: "u32", value: 0 } as any,
      bound: { id: -1, kind: "const", valueType: "scalar", dataType: "u32", value: size } as any,
      body: innerBody,
    });
  }

  /**
   * Mark this block for dtype conversion at store time.
   * The actual cast is applied by the store lowering.
   */
  castTo_(dtype: DataType): void {
    this._castDtype = dtype;
  }
  /** @internal */ _castDtype?: DataType;

}

/**
 * BlockOps: Unified block-level kernel API.
 *
 * Triton-like semantics:
 * - ONE `load` whose ptr type determines placement (register vs shared)
 * - ONE `dot` whose operand placements determine lowering (inner vs outer product)
 * - Arithmetic just works with automatic broadcasting
 */
export class BlockOps {
  private blockCounter = 0;
  /** @internal */ readonly wgSize: number;
  /** @internal */ threadTileM?: number;
  /** @internal */ threadTileN?: number;

  constructor(
    private readonly ctx: KernelContext,
    config: {
      wgSize: number | [number, number];
      threadTile?: [number, number];
    },
  ) {
    this.wgSize = typeof config.wgSize === "number"
      ? config.wgSize
      : config.wgSize[0] * config.wgSize[1];
    if (config.threadTile) {
      this.threadTileM = config.threadTile[0];
      this.threadTileN = config.threadTile[1];
    }
  }

  /** Set thread tile dimensions (can be called after construction). */
  setThreadTile(m: number, n: number): void {
    this.threadTileM = m;
    this.threadTileN = n;
  }

  private freshName(): string {
    return `blk_${this.blockCounter++}`;
  }

  // ---- Allocation ----

  /** Create a register block initialized to zero. */
  zeros(rows: number, cols: number): Block {
    const name = this.freshName();
    this.ctx.pushStatement({
      kind: "blockAlloc",
      name,
      rows,
      cols,
      elemType: "f32",
    });
    return new Block("register", rows, cols, name, this.ctx, this);
  }

  /** Create a register block filled with a constant value. */
  full(rows: number, cols: number, val: number): Block {
    const name = this.freshName();
    this.ctx.pushStatement({
      kind: "blockAlloc",
      name,
      rows,
      cols,
      elemType: "f32",
      initValue: val,
    });
    return new Block("register", rows, cols, name, this.ctx, this);
  }

  // ---- Load ----

  /** Unified load: ptr type determines placement (register vs shared). */
  load(binding: string, ptr: BlockPtr, opts: BlockLoadOpts): Block {
    const name = this.freshName();
    const { rows, cols, guard } = opts;
    const bindingType = (this.ctx as any).bindingSpecs?.[binding]?.type ?? "f32";

    if (ptr.kind === "thread") {
      // Per-thread load → register placement
      this.ctx.pushStatement({
        kind: "blockLoad",
        binding,
        name,
        rows,
        cols,
        elemType: bindingType,
        ptrKind: "thread",
        threadBase: ptr.base.node,
        threadStride: ptr.stride.node,
        guard: guard?.node,
      });
      return new Block("register", rows, cols, name, this.ctx, this);
    } else {
      // Cooperative load → shared memory placement
      // Pad column stride by 1 to avoid shared memory bank conflicts.
      // For stride S, rows differing by 32/S map to the same bank. The +1
      // breaks this alignment (e.g. tileK=8 → stride 9, no 4-way conflict).
      const smemStride = cols + 1;
      this.ctx.sharedArrays.push({
        name,
        size: rows * smemStride,
        elemType: "f32",
      });
      this.ctx.pushStatement({
        kind: "blockLoad",
        binding,
        name,
        rows,
        cols,
        elemType: bindingType,
        ptrKind: "tile",
        tilePtr: {
          baseOffset: ptr.baseOffset.node,
          outerRange: ptr.outerRange,
          outerStride: ptr.outerStride.node,
          innerRange: ptr.innerRange,
          innerStride: ptr.innerStride.node,
        },
        tileMask: {
          outerRange: ptr.outerRange,
          outerBound: ptr.outerBound.node,
          innerRange: ptr.innerRange,
          innerBound: ptr.innerBound.node,
        },
      });
      return new Block("shared", rows, cols, name, this.ctx, this, undefined, undefined, smemStride);
    }
  }

  // ---- Dot ----

  /** Unified dot product: operand placements determine lowering. */
  dot(a: Block, b: Block): Block {
    const name = this.freshName();
    // Result shape depends on the dot pattern
    let outRows: number;
    let outCols: number;
    if (a.placement === "shared" && b.placement === "shared") {
      // Outer product: result is per-thread [threadTileM × threadTileN]
      if (!this.threadTileM || !this.threadTileN) {
        throw new Error("shared×shared dot requires threadTile in BlockOps config");
      }
      outRows = this.threadTileM;
      outCols = this.threadTileN;
    } else {
      outRows = a.rows;
      outCols = b._transposed ? b._origRows : b.cols;
    }
    // Allocate result block
    this.ctx.pushStatement({
      kind: "blockAlloc",
      name,
      rows: outRows,
      cols: outCols,
      elemType: "f32",
    });
    this.ctx.pushStatement({
      kind: "blockDot",
      aName: a.name,
      bName: b.name,
      resultName: name,
      aPlacement: a.placement,
      bPlacement: b.placement,
      bTransposed: b._transposed,
      aRows: a.rows,
      aCols: a.cols,
      bRows: b._origRows,
      bCols: b._origCols,
      threadTileM: this.threadTileM,
      threadTileN: this.threadTileN,
      aSmemStride: a.placement === "shared" ? a.smemStride : undefined,
      bSmemStride: b.placement === "shared" ? b.smemStride : undefined,
    });
    return new Block("register", outRows, outCols, name, this.ctx, this);
  }

  /** Dot with accumulation: acc += a @ b. */
  dotAccum(a: Block, b: Block, acc: Block): void {
    this.ctx.pushStatement({
      kind: "blockDot",
      aName: a.name,
      bName: b.name,
      resultName: acc.name,
      accName: acc.name,
      aPlacement: a.placement,
      bPlacement: b.placement,
      bTransposed: b._transposed,
      aRows: a.rows,
      aCols: a.cols,
      bRows: b._origRows,
      bCols: b._origCols,
      threadTileM: this.threadTileM,
      threadTileN: this.threadTileN,
      aSmemStride: a.placement === "shared" ? a.smemStride : undefined,
      bSmemStride: b.placement === "shared" ? b.smemStride : undefined,
    });
  }

  // ---- Store ----

  /** Store register block to global memory. */
  store(binding: string, block: Block, ptr: BlockStorePtr, opts?: { guard?: BlockExpr }): void {
    this.ctx.pushStatement({
      kind: "blockStore",
      binding,
      blockName: block.name,
      rows: block.rows,
      cols: block.cols,
      base: ptr.base.node,
      stride: ptr.stride.node,
      guard: opts?.guard?.node,
    });
  }

  /**
   * Store a register block using tile-level 2D addressing with bounds checking.
   * Thread-tile-aware: each thread stores its threadTileM × threadTileN slice.
   */
  storeTile(binding: string, block: Block, ptr: TilePtr, mask: TileMask): void {
    if (!this.threadTileM || !this.threadTileN) {
      throw new Error("storeTile requires threadTile in BlockOps config");
    }
    this.ctx.pushStatement({
      kind: "tileStore",
      binding,
      ptr: ptr.data,
      mask: mask.data,
      accName: block.name,
      threadTileM: this.threadTileM,
      threadTileN: this.threadTileN,
      accDtype: block._castDtype,
    });
  }

  // ---- Tile-level helpers (for matmul patterns) ----

  /** Create a 1D range: [base, base+1, ..., base+blockSize-1] */
  arange(base: BlockExpr, blockSize: number): TileRange {
    return new TileRange({ base: base.node, size: blockSize });
  }

  /**
   * Load a 1D tile into per-thread registers (e.g., bias vector).
   * Returns a Block of shape [1, threadTileN].
   */
  load1d(binding: string, range: TileRange): Block {
    if (!this.threadTileN) {
      throw new Error("load1d requires threadTile in BlockOps config");
    }
    const name = this.freshName();
    this.ctx.pushStatement({
      kind: "tileLoad1d",
      binding,
      range: range.info,
      arrayName: name,
      size: this.threadTileN,
    });
    return new Block("register", 1, this.threadTileN, name, this.ctx, this);
  }

  /**
   * Cooperative tile load from global memory into shared memory.
   * Uses TilePtr/TileMask for 2D bounds-checked loading.
   */
  loadTile(binding: string, ptr: TilePtr, mask: TileMask): Block {
    const name = this.freshName();
    const tileRows = ptr.data.outerRange.size;
    const tileCols = ptr.data.innerRange.size;
    const smemStride = tileCols + 1;

    // Declare shared memory (padded by 1 column to avoid bank conflicts)
    this.ctx.sharedArrays.push({
      name,
      size: tileRows * smemStride,
      elemType: "f32",
    });

    this.ctx.pushStatement({
      kind: "tileLoad",
      binding,
      ptr: ptr.data,
      mask: mask.data,
      sharedName: name,
      tileRows,
      tileCols,
      elemType: "f32",
      smemStride,
    });

    return new Block("shared", tileRows, tileCols, name, this.ctx, this, undefined, undefined, smemStride);
  }

  // ---- Internal helpers (called by Block methods) ----

  /** @internal */
  _binary(a: Block, b: Block | BlockExpr, op: BlockBinaryOp, inPlace = false): Block {
    const outputName = inPlace ? a.name : this.freshName();
    const isBlock = b instanceof Block;
    this.ctx.pushStatement({
      kind: "blockBinary",
      aName: a.name,
      bName: isBlock ? b.name : "",
      outputName,
      aRows: a.rows, aCols: a.cols,
      bRows: isBlock ? b.rows : 1,
      bCols: isBlock ? b.cols : 1,
      op, inPlace,
      ...(isBlock ? {} : { bScalarExpr: b.node }),
    });
    if (inPlace) return a;
    const outRows = isBlock ? Math.max(a.rows, b.rows) : a.rows;
    const outCols = isBlock ? Math.max(a.cols, b.cols) : a.cols;
    return new Block("register", outRows, outCols, outputName, this.ctx, this);
  }

  /** @internal */
  _unary(a: Block, op: BlockUnaryOp, inPlace = false): Block {
    const outputName = inPlace ? a.name : this.freshName();
    this.ctx.pushStatement({
      kind: "blockUnary",
      inputName: a.name, outputName,
      rows: a.rows, cols: a.cols, op, inPlace,
    });
    return inPlace ? a : new Block("register", a.rows, a.cols, outputName, this.ctx, this);
  }

  /** @internal */
  _reduce(a: Block, axis: number, op: BlockReduceOp): Block {
    const name = this.freshName();
    const outRows = axis === 0 ? 1 : a.rows;
    const outCols = axis === 1 ? 1 : a.cols;
    this.ctx.pushStatement({
      kind: "blockReduce",
      inputName: a.name,
      outputName: name,
      inputRows: a.rows,
      inputCols: a.cols,
      axis,
      op,
    });
    return new Block("register", outRows, outCols, name, this.ctx, this);
  }
}
