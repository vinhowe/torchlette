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
  AccOpKind,
} from "./tile-ir";
import { BlockExpr, KernelContext } from "./tile-ir";

// ============================================================================
// Configuration
// ============================================================================

export interface TileConfig {
  BLOCK_M: number;
  BLOCK_N: number;
  BLOCK_K: number;
  threadTileM: number;
  threadTileN: number;
  wgSizeX: number;
  wgSizeY: number;
}

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
// Tile / Tile1D: handles for loaded tiles
// ============================================================================

export class Tile {
  constructor(
    readonly sharedName: string,
    readonly rows: number,
    readonly cols: number,
  ) {}
}

export class Tile1D {
  constructor(
    readonly arrayName: string,
    readonly size: number,
  ) {}
}

// ============================================================================
// Accumulator: block-sized [BLOCK_M, BLOCK_N], thread-mapped internally
// ============================================================================

export class Accumulator {
  /** Set by castTo_; the compiler reads this when emitting the store. */
  _castDtype: DataType | undefined;

  constructor(
    readonly name: string,
    readonly blockM: number,
    readonly blockN: number,
    private readonly threadTileM: number,
    private readonly threadTileN: number,
    private readonly ctx: KernelContext,
  ) {}

  /** Read a single element at thread-tile coordinates (tm, tn). */
  get(tm: BlockExpr, tn: BlockExpr): BlockExpr {
    const ttN = this.ctx.const(this.threadTileN, "u32");
    return new BlockExpr({
      id: -1, kind: "arrayRead", arrayName: this.name,
      idx: tm.mul(ttN).add(tn).node,
      valueType: "scalar", dataType: "f32",
    } as any);
  }

  /** acc *= scalar */
  mul_(value: BlockExpr): void {
    this.ctx.pushStatement({
      kind: "accOp",
      accName: this.name,
      threadTileM: this.threadTileM,
      threadTileN: this.threadTileN,
      op: { kind: "mulScalar", value: value.node },
    });
  }

  /** acc[tm, :] += tile1d[:] (broadcast 1D across rows) */
  add_(tile1d: Tile1D): void {
    this.ctx.pushStatement({
      kind: "accOp",
      accName: this.name,
      threadTileM: this.threadTileM,
      threadTileN: this.threadTileN,
      op: { kind: "addRow", valuesArray: tile1d.arrayName, size: tile1d.size },
    });
  }

  /** Apply an elementwise function to each element of the accumulator. */
  apply_(fn: (val: BlockExpr) => BlockExpr): void {
    const valName = `_acc_val`;
    const valRef = new BlockExpr({
      id: -1, kind: "namedRef", name: valName,
      valueType: "scalar", dataType: "f32",
    });
    const body = this.ctx.captureScope(() => {
      const result = fn(valRef);
      // Store the result back into the accumulator element
      this.ctx.pushStatement({
        kind: "assign",
        name: "_acc_result",
        value: result.node,
      });
    });
    // Find the result node from the last statement in body
    const lastStmt = body[body.length - 1];
    const resultNode = lastStmt && lastStmt.kind === "assign" ? lastStmt.value : valRef.node;
    // Remove the synthetic assign — the lowering pass uses resultNode directly
    if (lastStmt && lastStmt.kind === "assign" && (lastStmt as any).name === "_acc_result") {
      body.pop();
    }
    this.ctx.pushStatement({
      kind: "accOp",
      accName: this.name,
      threadTileM: this.threadTileM,
      threadTileN: this.threadTileN,
      op: { kind: "apply", body, resultNode, valName },
    });
  }

  /** Cast accumulator to a different dtype (applied at store time). */
  castTo_(dtype: DataType): void {
    this._castDtype = dtype;
    this.ctx.pushStatement({
      kind: "accOp",
      accName: this.name,
      threadTileM: this.threadTileM,
      threadTileN: this.threadTileN,
      op: { kind: "castTo", dtype },
    });
  }
}

// ============================================================================
// TileOps: main API class
// ============================================================================

export class TileOps {
  private readonly totalThreads: number;
  private tileCounter = 0;
  private arrayCounter = 0;

  constructor(
    private readonly ctx: KernelContext,
    private readonly config: TileConfig,
  ) {
    this.totalThreads = config.wgSizeX * config.wgSizeY;
  }

  /** Create a 1D range: [base, base+1, ..., base+blockSize-1] */
  arange(base: BlockExpr, blockSize: number): TileRange {
    return new TileRange({ base: base.node, size: blockSize });
  }

  /** Create a block-sized accumulator initialized to zero. */
  zeros(blockM: number, blockN: number): Accumulator {
    const name = `acc`;
    // Push a varArray for the thread-tile portion
    this.ctx.pushStatement({
      kind: "varArray",
      name,
      elemType: "f32",
      size: this.config.threadTileM * this.config.threadTileN,
    });
    return new Accumulator(
      name, blockM, blockN,
      this.config.threadTileM, this.config.threadTileN,
      this.ctx,
    );
  }

  /** Cooperative tile load from global memory into shared memory. */
  load(binding: string, ptr: TilePtr, mask: TileMask): Tile {
    const sharedName = `tile_${this.tileCounter++}`;
    const tileRows = ptr.data.outerRange.size;
    const tileCols = ptr.data.innerRange.size;

    // Declare shared memory
    this.ctx.sharedArrays.push({
      name: sharedName,
      size: tileRows * tileCols,
      elemType: "f32",
    });

    this.ctx.pushStatement({
      kind: "tileLoad",
      binding,
      ptr: ptr.data,
      mask: mask.data,
      sharedName,
      tileRows,
      tileCols,
      elemType: "f32",
    });

    return new Tile(sharedName, tileRows, tileCols);
  }

  /** Load a 1D tile into per-thread registers (e.g., bias vector). */
  load1d(binding: string, range: TileRange): Tile1D {
    const arrayName = `arr_${this.arrayCounter++}`;
    const size = this.config.threadTileN;

    this.ctx.pushStatement({
      kind: "tileLoad1d",
      binding,
      range: range.info,
      arrayName,
      size,
    });

    return new Tile1D(arrayName, size);
  }

  /** Block-level dot product: acc += a @ b. Handles barriers + outer product. */
  dot(a: Tile, b: Tile, acc: Accumulator): void {
    this.ctx.pushStatement({
      kind: "dot",
      aTile: { sharedName: a.sharedName, rows: a.rows, innerDim: a.cols },
      bTile: { sharedName: b.sharedName, innerDim: b.rows, cols: b.cols },
      accName: acc.name,
      threadTileM: this.config.threadTileM,
      threadTileN: this.config.threadTileN,
    });
  }

  /** Store accumulator to global memory with bounds checking. */
  store(binding: string, ptr: TilePtr, acc: Accumulator, mask: TileMask): void {
    this.ctx.pushStatement({
      kind: "tileStore",
      binding,
      ptr: ptr.data,
      mask: mask.data,
      accName: acc.name,
      threadTileM: this.config.threadTileM,
      threadTileN: this.config.threadTileN,
      accDtype: acc._castDtype,
    });
  }
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
