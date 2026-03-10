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
  BlockBinaryOp,
  BlockReduceOp,
  BlockUnaryOp,
  DataType,
  IRNode,
  Statement,
  TileMask2D,
  TilePtr2D,
  TileRangeInfo,
} from "./tile-ir";
import { BlockExpr, type KernelContext } from "./tile-ir";

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
  constructor(readonly info: TileRangeInfo) {}

  /** Broadcast to outer (row) dimension: [:, None] * stride */
  outer(stride?: BlockExpr): TilePtrComponent {
    const defaultStride: IRNode = {
      id: -1,
      kind: "const",
      valueType: "scalar",
      dataType: "u32",
      value: 1,
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
      id: -1,
      kind: "const",
      valueType: "scalar",
      dataType: "u32",
      value: 1,
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
      id: -1,
      kind: "binary",
      op: "add",
      lhs: this.data.baseOffset,
      rhs: {
        id: -1,
        kind: "binary",
        op: "mul",
        lhs: {
          id: -1,
          kind: "const",
          valueType: "scalar",
          dataType: "u32",
          value: blockDim,
        },
        rhs: stride.node,
        valueType: "scalar",
        dataType: "u32",
      },
      valueType: "scalar",
      dataType: "u32",
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
  base: BlockExpr; // starting global offset for this thread
  stride: BlockExpr; // row stride in global buffer
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
  /** Override shared memory element type for cooperative loads.
   * When set to "f16" with an f32 binding, data is converted f32→f16 on
   * store to shared memory (halving smem footprint) and f16→f32 on read. */
  smemElemType?: DataType;
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
  /** @internal Column stride for shared memory indexing. Always equals cols (no padding). */
  readonly smemStride: number;
  /** @internal Shared memory element type ("f16" for native f16 storage, "f32" default). */
  readonly smemElemType: DataType;
  /** @internal True if shared memory uses array<vec4<f32>> layout. */
  readonly smemVec4: boolean;

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
    smemElemType?: DataType,
    smemVec4?: boolean,
  ) {
    this._origRows = origRows ?? rows;
    this._origCols = origCols ?? cols;
    this.smemStride = smemStride ?? cols;
    this.smemElemType = smemElemType ?? "f32";
    this.smemVec4 = smemVec4 ?? false;
  }

  get transposed(): boolean {
    return this._transposed;
  }

  // ---- Element access (for hybrid scalar/block patterns like attention masking) ----

  /** Read a scalar from the block at flat index, or 2D (row, col). */
  get(row: BlockExpr, col?: BlockExpr): BlockExpr {
    const idx =
      col !== undefined ? row.mul(this.ctx.u32(this.cols)).add(col) : row;
    if (this.placement === "shared") {
      return this.ctx._makeSharedRead(this.name, idx);
    }
    return this.ctx._makeArrayRead(this.name, idx);
  }

  /** Write a scalar to the block at flat index. */
  set(idx: BlockExpr, value: BlockExpr): void {
    const kind = this.placement === "shared" ? "sharedWrite" : "indexAssign";
    this.ctx.pushStatement({
      kind,
      arrayName: this.name,
      idx: idx.node,
      value: value.node,
    });
  }

  // ---- Transpose (metadata only, no data movement) ----
  T(): Block {
    const b = new Block(
      this.placement,
      this.cols,
      this.rows,
      this.name,
      this.ctx,
      this.ops,
      this._origRows,
      this._origCols,
      this.smemStride,
      this.smemElemType,
      this.smemVec4,
    );
    b._transposed = !this._transposed;
    return b;
  }

  // ---- Arithmetic (returns new register Block) ----
  add(other: Block | BlockExpr): Block {
    return this.ops._binary(this, other, "add");
  }
  sub(other: Block | BlockExpr): Block {
    return this.ops._binary(this, other, "sub");
  }
  mul(other: Block | BlockExpr): Block {
    return this.ops._binary(this, other, "mul");
  }
  div(other: Block | BlockExpr): Block {
    return this.ops._binary(this, other, "div");
  }

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
  mul_(other: Block | BlockExpr): void {
    this.ops._binary(this, other, "mul", true);
  }
  add_(other: Block | BlockExpr): void {
    this.ops._binary(this, other, "add", true);
  }
  sub_(other: Block | BlockExpr): void {
    this.ops._binary(this, other, "sub", true);
  }
  exp_(): void {
    this.ops._unary(this, "exp", true);
  }

  /** Copy values from other into this block (with broadcasting). */
  assign(other: Block): void {
    this.ops._binary(this, other, "copy", true);
  }

  /** Accumulate: this += other (with broadcasting). */
  addAssign(other: Block): void {
    this.ops._binary(this, other, "add", true);
  }

  // ---- Unary (returns new register Block) ----
  exp(): Block {
    return this.ops._unary(this, "exp");
  }
  log(): Block {
    return this.ops._unary(this, "log");
  }
  neg(): Block {
    return this.ops._unary(this, "neg");
  }

  /**
   * Apply an elementwise function in-place: block[i] = fn(block[i]).
   * Used for epilogue ops (relu, gelu, etc.) on register blocks.
   */
  apply_(fn: (val: BlockExpr) => BlockExpr): void {
    const size = this.rows * this.cols;
    const valName = `_apply_val_${this.name}`;
    const valRef = new BlockExpr({
      id: -1,
      kind: "namedRef",
      name: valName,
      valueType: "scalar",
      dataType: "f32",
    } as IRNode);
    // Capture the function body as IR statements
    const body = this.ctx.captureScope(() => {
      const result = fn(valRef);
      this.ctx.pushStatement({
        kind: "assign",
        name: `_apply_result_${this.name}`,
        value: result.node,
      });
    });
    // Extract result node from the synthetic assign
    const lastStmt = body[body.length - 1];
    const resultNode =
      lastStmt && lastStmt.kind === "assign" ? lastStmt.value : valRef.node;
    if (
      lastStmt &&
      lastStmt.kind === "assign" &&
      lastStmt.name === `_apply_result_${this.name}`
    ) {
      body.pop();
    }
    // Emit forRange over all elements
    const iVar = `_apply_i_${this.name}`;
    const innerBody: Statement[] = [];
    // let valName = block[i]
    innerBody.push({
      kind: "let",
      name: valName,
      dtype: "f32" satisfies DataType,
      value: {
        id: -1,
        kind: "arrayRead",
        arrayName: this.name,
        idx: {
          id: -1,
          kind: "namedRef",
          name: iVar,
          valueType: "scalar",
          dataType: "u32",
        } as IRNode,
        valueType: "scalar",
        dataType: "f32",
      } as IRNode,
    });
    innerBody.push(...body);
    // block[i] = result
    innerBody.push({
      kind: "indexAssign",
      arrayName: this.name,
      idx: {
        id: -1,
        kind: "namedRef",
        name: iVar,
        valueType: "scalar",
        dataType: "u32",
      } as IRNode,
      value: resultNode,
    });
    this.ctx.pushStatement({
      kind: "forRange",
      varName: iVar,
      start: {
        id: -1,
        kind: "const",
        valueType: "scalar",
        dataType: "u32",
        value: 0,
      } as IRNode,
      bound: {
        id: -1,
        kind: "const",
        valueType: "scalar",
        dataType: "u32",
        value: size,
      } as IRNode,
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
    this.wgSize =
      typeof config.wgSize === "number"
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
    const bindingType = this.ctx.getBindingType(binding);

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
      // Use native f16 shared memory when binding is f16, or when explicitly requested
      // via opts.smemElemType. f16 smem halves footprint (allowing larger tiles),
      // reduces bank conflicts, and converts on read (f16→f32 widening).
      const smemElemType: DataType =
        opts.smemElemType ?? (bindingType === "f16" ? "f16" : "f32");
      const smemStride = cols;
      // Auto-detect vec4 shared memory: same criteria as loadTile.
      // Skip for matmul (threadTileM set): the shared×shared dot K-loop
      // reads individual scalars, and vec4 component extraction overhead
      // outweighs cooperative load savings (benchmarked: +17-36% regression).
      const useVec4Smem =
        cols % 4 === 0 &&
        smemElemType === "f32" &&
        smemStride % 4 === 0 &&
        !this.threadTileM;
      if (useVec4Smem) {
        this.ctx.vec4SharedArrays.push({
          name,
          size: (rows * smemStride) / 4,
        });
      } else {
        this.ctx.sharedArrays.push({
          name,
          size: rows * smemStride,
          elemType: smemElemType,
        });
      }
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
        smemElemType: smemElemType !== bindingType ? smemElemType : undefined,
        smemVec4: useVec4Smem || undefined,
      });
      return new Block(
        "shared",
        rows,
        cols,
        name,
        this.ctx,
        this,
        undefined,
        undefined,
        smemStride,
        smemElemType,
        useVec4Smem,
      );
    }
  }

  // ---- Dot ----

  /** Push a blockDot statement with common fields derived from a and b. */
  private _pushBlockDot(
    a: Block,
    b: Block,
    resultName: string,
    accName?: string,
  ): void {
    this.ctx.pushStatement({
      kind: "blockDot",
      aName: a.name,
      bName: b.name,
      resultName,
      ...(accName !== undefined && { accName }),
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
      aSmemElemType: a.placement === "shared" ? a.smemElemType : undefined,
      bSmemElemType: b.placement === "shared" ? b.smemElemType : undefined,
    });
  }

  /** Unified dot product: operand placements determine lowering. */
  dot(a: Block, b: Block): Block {
    const name = this.freshName();
    let outRows: number, outCols: number;
    if (a.placement === "shared" && b.placement === "shared") {
      if (!this.threadTileM || !this.threadTileN) {
        throw new Error(
          "shared×shared dot requires threadTile in BlockOps config",
        );
      }
      outRows = this.threadTileM;
      outCols = this.threadTileN;
    } else {
      outRows = a.rows;
      outCols = b._transposed ? b._origRows : b.cols;
    }
    this.ctx.pushStatement({
      kind: "blockAlloc",
      name,
      rows: outRows,
      cols: outCols,
      elemType: "f32",
    });
    this._pushBlockDot(a, b, name);
    return new Block("register", outRows, outCols, name, this.ctx, this);
  }

  /** Dot with accumulation: acc += a @ b. */
  dotAccum(a: Block, b: Block, acc: Block): void {
    this._pushBlockDot(a, b, acc.name, acc.name);
  }

  /**
   * Single-row dot product: returns scalar = dot(regBlock[0,:], sharedBlock[rowIdx,:]).
   * Used inside range() loops for fused attention inner loops.
   * regBlock must be register (1×D), sharedBlock must be shared (K×D).
   * With TPR, handles subgroup reduction automatically.
   */
  dotRow(regBlock: Block, sharedBlock: Block, rowIdx: BlockExpr): BlockExpr {
    if (regBlock.placement !== "register" || regBlock.rows !== 1) {
      throw new Error("dotRow: regBlock must be register with rows=1");
    }
    if (sharedBlock.placement !== "shared") {
      throw new Error("dotRow: sharedBlock must be shared");
    }
    const resultName = this.freshName() + "_s";
    this.ctx.pushStatement({
      kind: "blockDotRow",
      aName: regBlock.name,
      bName: sharedBlock.name,
      resultName,
      rowIdx: rowIdx.node,
      aCols: regBlock.cols,
      bCols: sharedBlock._origCols,
      bSmemStride: sharedBlock.smemStride,
      bSmemElemType:
        sharedBlock.smemElemType !== "f32"
          ? sharedBlock.smemElemType
          : undefined,
    });
    return this.ctx._makeRef(resultName, "f32");
  }

  /**
   * Single-row scaled accumulation: accBlock[:] += scalar * sharedBlock[rowIdx,:].
   * Used inside range() loops for fused attention inner loops.
   * accBlock must be register (1×D), sharedBlock must be shared (K×D).
   */
  accumRow(
    accBlock: Block,
    scalar: BlockExpr,
    sharedBlock: Block,
    rowIdx: BlockExpr,
  ): void {
    if (accBlock.placement !== "register" || accBlock.rows !== 1) {
      throw new Error("accumRow: accBlock must be register with rows=1");
    }
    if (sharedBlock.placement !== "shared") {
      throw new Error("accumRow: sharedBlock must be shared");
    }
    // Store scalar to a named variable so lowering can reference it
    const scalarName = this.freshName() + "_p";
    this.ctx.pushStatement({
      kind: "let",
      name: scalarName,
      dtype: "f32" as const,
      value: scalar.node,
    });
    this.ctx.pushStatement({
      kind: "blockAccumRow",
      accName: accBlock.name,
      bName: sharedBlock.name,
      scalarName,
      rowIdx: rowIdx.node,
      accCols: accBlock.cols,
      bCols: sharedBlock._origCols,
      bSmemStride: sharedBlock.smemStride,
      bSmemElemType:
        sharedBlock.smemElemType !== "f32"
          ? sharedBlock.smemElemType
          : undefined,
    });
  }

  // ---- Store ----

  /** Store register block to global memory. */
  store(
    binding: string,
    block: Block,
    ptr: BlockStorePtr,
    opts?: { guard?: BlockExpr },
  ): void {
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
  loadTile(
    binding: string,
    ptr: TilePtr,
    mask: TileMask,
    opts?: {
      smemElemType?: DataType;
      smemPadding?: number;
      reuseShared?: Block;
    },
  ): Block {
    const tileRows = ptr.data.outerRange.size;
    const tileCols = ptr.data.innerRange.size;
    const padding = opts?.smemPadding ?? 0;
    const bindingType = this.ctx.getBindingType(binding);

    // Reuse existing shared memory allocation if specified
    const reuse = opts?.reuseShared;
    if (reuse) {
      // Validate dimensions match
      if (reuse.rows !== tileRows || reuse.cols !== tileCols) {
        throw new Error(
          `reuseShared dimensions mismatch: expected ${tileRows}×${tileCols}, got ${reuse.rows}×${reuse.cols}`,
        );
      }
      const name = reuse.name;
      const smemStride = reuse.smemStride;
      const smemElemType = reuse.smemElemType;

      // Emit load into existing shared array (no new allocation)
      this.ctx.pushStatement({
        kind: "tileLoad",
        binding,
        ptr: ptr.data,
        mask: mask.data,
        sharedName: name,
        tileRows,
        tileCols,
        elemType: bindingType,
        smemElemType: smemElemType !== bindingType ? smemElemType : undefined,
        smemStride: smemStride !== tileCols ? smemStride : undefined,
        smemVec4: reuse.smemVec4 || undefined,
      });

      return new Block(
        "shared",
        tileRows,
        tileCols,
        name,
        this.ctx,
        this,
        undefined,
        undefined,
        smemStride,
        smemElemType,
        reuse.smemVec4,
      );
    }

    const name = this.freshName();
    const smemStride = tileCols + padding;

    // Use native f16 shared memory when binding is f16, or when explicitly overridden
    const smemElemType: DataType =
      opts?.smemElemType ?? (bindingType === "f16" ? "f16" : "f32");

    // Use vec4 shared memory when: cols divisible by 4, f32 type, no padding.
    // Skip for matmul (threadTileM set) — see load() comment for rationale.
    const useVec4Smem =
      tileCols % 4 === 0 &&
      smemElemType === "f32" &&
      smemStride % 4 === 0 &&
      !this.threadTileM;

    // Declare shared memory
    if (useVec4Smem) {
      this.ctx.vec4SharedArrays.push({
        name,
        size: (tileRows * smemStride) / 4,
      });
    } else {
      this.ctx.sharedArrays.push({
        name,
        size: tileRows * smemStride,
        elemType: smemElemType,
      });
    }

    this.ctx.pushStatement({
      kind: "tileLoad",
      binding,
      ptr: ptr.data,
      mask: mask.data,
      sharedName: name,
      tileRows,
      tileCols,
      elemType: bindingType,
      smemElemType: smemElemType !== bindingType ? smemElemType : undefined,
      smemStride: padding > 0 ? smemStride : undefined,
      smemVec4: useVec4Smem || undefined,
    });

    return new Block(
      "shared",
      tileRows,
      tileCols,
      name,
      this.ctx,
      this,
      undefined,
      undefined,
      smemStride,
      smemElemType,
      useVec4Smem,
    );
  }

  // ---- Internal helpers (called by Block methods) ----

  /** @internal */
  _binary(
    a: Block,
    b: Block | BlockExpr,
    op: BlockBinaryOp,
    inPlace = false,
  ): Block {
    const outputName = inPlace ? a.name : this.freshName();
    const isBlock = b instanceof Block;
    this.ctx.pushStatement({
      kind: "blockBinary",
      aName: a.name,
      bName: isBlock ? b.name : "",
      outputName,
      aRows: a.rows,
      aCols: a.cols,
      bRows: isBlock ? b.rows : 1,
      bCols: isBlock ? b.cols : 1,
      op,
      inPlace,
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
      inputName: a.name,
      outputName,
      rows: a.rows,
      cols: a.cols,
      op,
      inPlace,
    });
    return inPlace
      ? a
      : new Block("register", a.rows, a.cols, outputName, this.ctx, this);
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
