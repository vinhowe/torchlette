/**
 * tile-lowering.ts — Tile statement lowering pass.
 *
 * Lowers high-level tile/block statements (tileLoad, tileStore, blockDot, etc.)
 * from the tile-IR into imperative statements (forRange, sharedWrite, indexAssign, etc.)
 * that the codegen phase can emit as WGSL.
 *
 * Extracted from tile-compiler.ts to separate lowering logic from WGSL codegen.
 */

import type {
  IRNode, TileKernelSpec,
  Statement, DataType,
  TileLoadStmt, TileLoad1DStmt, TileStoreStmt,
  BlockAllocStmt, BlockLoadStmt, BlockStoreStmt, BlockDotStmt,
  BlockReduceStmt, BlockUnaryStmt, BlockBinaryStmt,
} from "./tile-ir";
import { F32_NEG_MAX } from "./shape-utils";

// ============================================================================
// Module-level State
// ============================================================================

let _varCounter = 0;

/** Module-level TPR state — set during compilation, reset after. */
let _activeTPR = 1;
let _blockLayouts: Map<string, BlockLayout> = new Map();

// ============================================================================
// State Management Functions
// ============================================================================

export function freshVar(hint: string): string {
  return `_${hint}${_varCounter++}`;
}

export function resetLoweringState(): void {
  _varCounter = 0;
  _activeTPR = 1;
  _blockLayouts = new Map();
}

export function setTPR(tpr: number): void {
  _activeTPR = tpr;
}

export function setBlockLayouts(layouts: Map<string, BlockLayout>): void {
  _blockLayouts = layouts;
}

export function getActiveTPR(): number {
  return _activeTPR;
}

// ============================================================================
// BlockLayout Type
// ============================================================================

/** Block layout for TPR-aware lowering. */
export type BlockLayout = "full" | "distributed" | "replicated";

// ============================================================================
// Analysis Helpers
// ============================================================================

/** Get the physical column count for a named block, accounting for TPR distribution. */
function getPhysCols(name: string, logicalCols: number): number {
  if (_activeTPR <= 1) return logicalCols;
  const layout = _blockLayouts.get(name);
  return layout === "distributed" ? logicalCols / _activeTPR : logicalCols;
}

/** Recursively collect all BlockDotStmt from a statement tree. */
function collectBlockDots(stmts: Statement[]): BlockDotStmt[] {
  const dots: BlockDotStmt[] = [];
  function scan(ss: Statement[]) {
    for (const s of ss) {
      if (s.kind === "blockDot") dots.push(s);
      if ("body" in s && Array.isArray((s as any).body)) scan((s as any).body);
      if ("elseBody" in s && Array.isArray((s as any).elseBody)) scan((s as any).elseBody);
    }
  }
  scan(stmts);
  return dots;
}

/**
 * Auto-detect whether this kernel benefits from TPR > 1 (threads-per-row).
 * Returns 4 if register×shared block dots with wide register operands are found,
 * otherwise returns 1 (no optimization).
 */
export function autoDetectTPR(stmts: Statement[], sgSupported: boolean): number {
  if (!sgSupported) return 1;

  const dots = collectBlockDots(stmts);
  if (dots.length === 0) return 1;

  // Don't enable TPR if kernel has shared×shared dots (uses 2D thread tiles)
  if (dots.some(d => d.aPlacement === "shared" && d.bPlacement === "shared")) return 1;

  // Need at least one register×shared dot with single-row register operand
  const hasRegSharedDot = dots.some(d =>
    d.aPlacement === "register" && d.bPlacement === "shared" && d.aRows === 1);
  if (!hasRegSharedDot) return 1;

  // Verify all distributed dimensions are compatible with TPR=4
  const layouts = new Map<string, BlockLayout>();
  for (const dot of dots) {
    if (dot.aPlacement === "register" && dot.bPlacement === "shared") {
      if (dot.bTransposed) {
        // RegSharedT: A is distributed (inner dim), result is replicated
        if (dot.aCols < 16 || dot.aCols % 4 !== 0) return 1;
        if (layouts.get(dot.aName) === "replicated") return 1;
        layouts.set(dot.aName, "distributed");
        layouts.set(dot.resultName, "replicated");
        if (dot.accName) {
          if (layouts.get(dot.accName) === "distributed") return 1;
          layouts.set(dot.accName, "replicated");
        }
      } else {
        // RegSharedNN: A is replicated, result is distributed (output cols)
        if (dot.bCols < 16 || dot.bCols % 4 !== 0) return 1;
        if (layouts.get(dot.aName) === "distributed") return 1;
        layouts.set(dot.aName, "replicated");
        if (layouts.get(dot.resultName) === "replicated") return 1;
        layouts.set(dot.resultName, "distributed");
        if (dot.accName) {
          if (layouts.get(dot.accName) === "replicated") return 1;
          layouts.set(dot.accName, "distributed");
        }
      }
    }
  }

  return 4;
}

/**
 * Compute block layouts for TPR-aware lowering.
 * Scans all block dot statements to determine which blocks are distributed/replicated.
 */
export function computeBlockLayouts(stmts: Statement[], tpr: number): Map<string, BlockLayout> {
  const layouts = new Map<string, BlockLayout>();
  if (tpr <= 1) return layouts;

  const dots = collectBlockDots(stmts);
  for (const dot of dots) {
    if (dot.aPlacement === "register" && dot.bPlacement === "shared") {
      if (dot.bTransposed) {
        layouts.set(dot.aName, "distributed");
        layouts.set(dot.resultName, "replicated");
        if (dot.accName) layouts.set(dot.accName, "replicated");
      } else {
        layouts.set(dot.aName, "replicated");
        layouts.set(dot.resultName, "distributed");
        if (dot.accName) layouts.set(dot.accName, "distributed");
      }
    }
  }

  return layouts;
}

// ============================================================================
// IR Constructor Utilities
// ============================================================================

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
function binOp(op: "add" | "sub" | "mul" | "div" | "mod" | "max" | "min", lhs: IRNode, rhs: IRNode, dt: DataType = "u32"): IRNode {
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
/** Create a subgroupShuffleXor IR node for butterfly reduction. */
function shuffleXorNode(value: IRNode, mask: number): IRNode {
  return {
    id: -1, kind: "subgroupShuffleXor",
    value, mask: cU32(mask),
    valueType: "scalar", dataType: "f32",
  } as IRNode;
}

// ============================================================================
// Reduction / Memory Helpers
// ============================================================================

/** Emit butterfly reduction statements for TPR threads (sum or max). */
function emitButterflyReduce(sVar: string, tpr: number, op: "sum" | "max"): Statement[] {
  const stmts: Statement[] = [];
  for (let mask = 1; mask < tpr; mask *= 2) {
    const shuffled = shuffleXorNode(ref(sVar, "f32"), mask);
    if (op === "sum") {
      stmts.push({ kind: "addAssign", name: sVar, value: shuffled });
    } else {
      stmts.push({
        kind: "assign", name: sVar,
        value: binOp("max", ref(sVar, "f32"), shuffled, "f32"),
      });
    }
  }
  return stmts;
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
  const logical = typeof spec.workgroupSize === "number"
    ? spec.workgroupSize : spec.workgroupSize[0] * spec.workgroupSize[1];
  return logical * _activeTPR;
}

// ============================================================================
// Tile Statement Detection
// ============================================================================

/** Check if any statements (recursively) contain tile-level ops. */
export function hasTileStatements(stmts: Statement[]): boolean {
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

// ============================================================================
// Main Lowering Entry Point
// ============================================================================

/**
 * Lower tile-level statements to imperative statements.
 * Tile ops (tileLoad, tileStore) are expanded into cooperative loading loops,
 * per-element loops, etc.
 *
 * Non-tile statements are passed through unchanged (with recursive lowering
 * of any nested bodies).
 */
export function lowerTileStatements(stmts: Statement[], spec: TileKernelSpec): Statement[] {
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

// ============================================================================
// Lowering Functions
// ============================================================================

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

  // Shared memory write index: flat (row-major, no padding)
  const smemIdx = ref(flatName);

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
  const pCols = getPhysCols(name, cols);
  const size = rows * pCols;
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
  const pCols = getPhysCols(name, cols);
  const size = rows * pCols;
  const isDistributed = _activeTPR > 1 && pCols !== cols;
  const result: Statement[] = [];
  const bindingDtype = spec.bindings[binding]?.type ?? elemType;
  const useVec4 = pCols % 4 === 0 && bindingDtype === "f32";

  // Allocate register array (physical size)
  result.push({
    kind: "varArray", name, elemType: "f32", size, skipZeroInit: true,
  });

  // Build loading loop
  const loadBody: Statement[] = [];

  // For distributed blocks: shift base by _sub_idx * physCols
  const effectiveBase = isDistributed
    ? binOp("add", threadBase!, binOp("mul", ref("_sub_idx"), cU32(pCols)))
    : threadBase!;

  if (rows === 1) {
    if (useVec4) {
      // Vec4 path: for (d4 = 0; d4 < pCols/4; d4++) { 4× unrolled loads }
      const d4Var = freshVar("d4");
      const offVar = freshVar("off");
      const innerBody: Statement[] = [];

      innerBody.push({
        kind: "let", name: offVar, dtype: "u32",
        value: binOp("add", effectiveBase, binOp("mul", ref(d4Var), cU32(4))),
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
        kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(pCols / 4),
        body: innerBody,
      });
    } else {
      // Scalar path: for (d = 0; d < pCols; d++) { reg[d] = buf[base + d]; }
      const dVar = freshVar("d");
      const innerBody: Statement[] = [];

      const loadIdx = binOp("add", effectiveBase, ref(dVar));
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
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(pCols),
        body: innerBody,
      });
    }
  } else {
    // Multiple rows: for (r = 0; r < rows; r++) for (d = 0; d < pCols; d++)
    const rVar = freshVar("r");
    const dVar = freshVar("d");
    const innerBody: Statement[] = [];

    const rowBase = binOp("add", threadBase!, binOp("mul", ref(rVar), threadStride!));
    // For distributed: each row also offset by _sub_idx * physCols
    const effRowBase = isDistributed
      ? binOp("add", rowBase, binOp("mul", ref("_sub_idx"), cU32(pCols)))
      : rowBase;
    const loadIdx = binOp("add", effRowBase, ref(dVar));
    let loadExpr: IRNode = loadBinding(binding, loadIdx, bindingDtype);
    if (bindingDtype !== "f32") {
      loadExpr = castNode(loadExpr, "f32");
    }
    const regIdx = binOp("add", binOp("mul", ref(rVar), cU32(pCols)), ref(dVar));

    innerBody.push({
      kind: "indexAssign",
      arrayName: name,
      idx: regIdx,
      value: loadExpr,
    });

    loadBody.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(rows),
      body: [{
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(pCols),
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
  const pCols = getPhysCols(blockName, cols);
  const isDistributed = _activeTPR > 1 && pCols !== cols;
  const result: Statement[] = [];
  const useVec4 = rows === 1 && pCols % 4 === 0;

  const storeBody: Statement[] = [];

  // For distributed blocks: shift base by _sub_idx * physCols
  const effectiveBase = isDistributed
    ? binOp("add", base, binOp("mul", ref("_sub_idx"), cU32(pCols)))
    : base;

  if (rows === 1 && useVec4) {
    // Vec4 path: for (d4 = 0; d4 < pCols/4; d4++) { 4× unrolled stores }
    const d4Var = freshVar("sd4");
    const offVar = freshVar("soff");
    const innerBody: Statement[] = [];
    innerBody.push({
      kind: "let", name: offVar, dtype: "u32",
      value: binOp("add", effectiveBase, binOp("mul", ref(d4Var), cU32(4))),
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
      kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(pCols / 4),
      body: innerBody,
    });
  } else if (rows === 1) {
    // Scalar single row: for (d = 0; d < pCols; d++) { buf[base + d] = reg[d]; }
    const dVar = freshVar("sd");
    storeBody.push({
      kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(pCols),
      body: [{
        kind: "indexAssign",
        arrayName: binding,
        idx: binOp("add", effectiveBase, ref(dVar)),
        value: arrayRead(blockName, ref(dVar)),
      }],
    });
  } else {
    // Multiple rows
    const rVar = freshVar("sr");
    const dVar = freshVar("sd");
    const rowBase = binOp("add", base, binOp("mul", ref(rVar), stride));
    const effRowBase = isDistributed
      ? binOp("add", rowBase, binOp("mul", ref("_sub_idx"), cU32(pCols)))
      : rowBase;
    storeBody.push({
      kind: "forRange", varName: rVar, start: cU32(0), bound: cU32(rows),
      body: [{
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(pCols),
        body: [{
          kind: "indexAssign",
          arrayName: binding,
          idx: binOp("add", effRowBase, ref(dVar)),
          value: arrayRead(blockName, binOp("add", binOp("mul", ref(rVar), cU32(pCols)), ref(dVar))),
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

  const aSmemStride = stmt.aSmemStride ?? aCols;  // stride for A in shared memory
  const bSmemStride = stmt.bSmemStride ?? bCols;  // stride for B in shared memory
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
  const bStride = stmt.bSmemStride ?? bCols;  // stride for B in shared memory
  const result: Statement[] = [];
  const innerDim = aCols; // = bCols (dimension being contracted)
  const outCols = bRows;  // B^T has bRows columns

  // With TPR: A is distributed, each thread has physInnerDim elements
  const physInnerDim = _activeTPR > 1 ? innerDim / _activeTPR : innerDim;
  const useVec4 = physInnerDim % 4 === 0;

  if (aRows === 1) {
    // Scalar mode: single row of A
    const jVar = freshVar("j");
    const sVar = freshVar("s");

    const jBody: Statement[] = [];
    // var s: f32 = 0.0;
    jBody.push({ kind: "var", name: sVar, dtype: "f32", value: cF32(0) });

    // B access base: j*bStride (+ _sub_idx*physInnerDim for TPR)
    const bRowBase = (d: IRNode) => {
      const base = binOp("add", binOp("mul", ref(jVar), cU32(bStride)), d);
      return _activeTPR > 1
        ? binOp("add", binOp("mul", ref("_sub_idx"), cU32(physInnerDim)), base)
        : base;
    };

    if (useVec4) {
      // Vec4 path: partial inner loop over physInnerDim/4 vec4 dots
      const d4Var = freshVar("d4");
      jBody.push({
        kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(physInnerDim / 4),
        body: [{
          kind: "addAssign",
          name: sVar,
          value: vec4DotExpr(
            [0, 1, 2, 3].map(k =>
              arrayRead(aName, binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k))),
            ) as [IRNode, IRNode, IRNode, IRNode],
            [0, 1, 2, 3].map(k =>
              sharedRead(bName, bRowBase(binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k)))),
            ) as [IRNode, IRNode, IRNode, IRNode],
          ),
        }],
      });
    } else {
      // Scalar path: for d in 0..physInnerDim
      const dVar = freshVar("d");
      jBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(physInnerDim),
        body: [{
          kind: "addAssign",
          name: sVar,
          value: binOp("mul",
            arrayRead(aName, ref(dVar)),
            sharedRead(bName, bRowBase(ref(dVar))),
            "f32",
          ),
        }],
      });
    }

    // Butterfly reduction across TPR threads (result becomes replicated)
    if (_activeTPR > 1) {
      jBody.push(...emitButterflyReduce(sVar, _activeTPR, "sum"));
    }

    // Store to result (replicated — all TPR threads write same value)
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
    // Multi-row mode (general case — no TPR for multi-row)
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
  const bStride = stmt.bSmemStride ?? bCols;  // stride for B in shared memory
  const result: Statement[] = [];
  const outCols = bCols;

  // With TPR: result is distributed, each thread handles physOutCols columns
  const physOutCols = _activeTPR > 1 ? outCols / _activeTPR : outCols;
  const useVec4 = physOutCols % 4 === 0;

  if (aRows === 1) {
    const jVar = freshVar("j");
    const pVar = freshVar("p");

    const jBody: Statement[] = [];
    // let p = a[j];  (A is replicated — all threads read same value)
    jBody.push({
      kind: "let", name: pVar, dtype: "f32",
      value: arrayRead(aName, ref(jVar)),
    });

    // B access: j*bStride + (sub_idx*physOutCols + d) for distributed output
    const bColBase = (d: IRNode) => {
      const colIdx = _activeTPR > 1
        ? binOp("add", binOp("mul", ref("_sub_idx"), cU32(physOutCols)), d)
        : d;
      return binOp("add", binOp("mul", ref(jVar), cU32(bStride)), colIdx);
    };

    if (useVec4) {
      // Vec4 path: for d4 in 0..physOutCols/4
      const d4Var = freshVar("d4");
      const d4Body: Statement[] = [];
      for (let k = 0; k < 4; k++) {
        const regIdx = binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k));
        const smemIdx = bColBase(binOp("add", binOp("mul", ref(d4Var), cU32(4)), cU32(k)));
        d4Body.push({
          kind: "indexAddAssign",
          arrayName: resultName,
          idx: regIdx,
          value: binOp("mul", ref(pVar, "f32"), sharedRead(bName, smemIdx), "f32"),
        });
      }
      jBody.push({
        kind: "forRange", varName: d4Var, start: cU32(0), bound: cU32(physOutCols / 4),
        body: d4Body,
      });
    } else {
      // Scalar path: for d in 0..physOutCols
      const dVar = freshVar("d");
      jBody.push({
        kind: "forRange", varName: dVar, start: cU32(0), bound: cU32(physOutCols),
        body: [{
          kind: "indexAddAssign",
          arrayName: resultName,
          idx: ref(dVar),
          value: binOp("mul",
            ref(pVar, "f32"),
            sharedRead(bName, bColBase(ref(dVar))),
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
    // Multi-row general case (no TPR for multi-row)
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
  const inputLayout = _blockLayouts.get(inputName);
  const isDistributed = _activeTPR > 1 && inputLayout === "distributed";
  const physInputCols = isDistributed ? inputCols / _activeTPR : inputCols;

  // Unified axis reduction: axis=1 reduces columns [R×C]→[R×1], axis=0 reduces rows [R×C]→[1×C]
  const outerDim = axis === 1 ? inputRows : (isDistributed ? physInputCols : inputCols);
  const innerDim = axis === 1 ? physInputCols : inputRows;
  result.push({
    kind: "varArray", name: outputName, elemType: "f32", size: outerDim, skipZeroInit: true,
  });

  const outerVar = freshVar("ro");
  const innerVar = freshVar("ri");
  const initVal = op === "max" ? cF32(F32_NEG_MAX) : cF32(0);

  // Input index: row * physCols + col
  const rowRef = axis === 1 ? ref(outerVar) : ref(innerVar);
  const colRef = axis === 1 ? ref(innerVar) : ref(outerVar);
  const inputIdx = binOp("add", binOp("mul", rowRef, cU32(physInputCols)), colRef);
  const accumExpr = binOp(op === "sum" ? "add" : "max",
    arrayRead(outputName, ref(outerVar)),
    arrayRead(inputName, inputIdx),
    "f32",
  );

  const outerBody: Statement[] = [];
  outerBody.push({ kind: "indexAssign", arrayName: outputName, idx: ref(outerVar), value: initVal });
  outerBody.push({
    kind: "forRange", varName: innerVar, start: cU32(0), bound: cU32(innerDim),
    body: [{ kind: "indexAssign", arrayName: outputName, idx: ref(outerVar), value: accumExpr }],
  });

  // For distributed blocks with axis=1: cross-thread butterfly reduction
  if (isDistributed && axis === 1) {
    // After local reduce, output[r] has the partial sum/max for this thread's portion.
    // Need to combine across TPR threads.
    const tmpVar = freshVar("rv");
    outerBody.push({
      kind: "var", name: tmpVar, dtype: "f32",
      value: arrayRead(outputName, ref(outerVar)),
    });
    outerBody.push(...emitButterflyReduce(tmpVar, _activeTPR, op === "sum" ? "sum" : "max"));
    outerBody.push({
      kind: "indexAssign", arrayName: outputName, idx: ref(outerVar),
      value: ref(tmpVar, "f32"),
    });
  }

  result.push({
    kind: "forRange", varName: outerVar, start: cU32(0), bound: cU32(outerDim),
    body: outerBody,
  });

  return result;
}

/**
 * Lower blockUnary → per-element unary operation.
 */
function lowerBlockUnary(stmt: BlockUnaryStmt): Statement[] {
  const { inputName, outputName, rows, cols, op, inPlace } = stmt;
  const pCols = getPhysCols(inPlace ? outputName : inputName, cols);
  const size = rows * pCols;
  const result: Statement[] = [];

  // Allocate output if not in-place
  if (!inPlace) {
    result.push({
      kind: "varArray", name: outputName, elemType: "f32", size, skipZeroInit: true,
    });
  }

  const iVar = freshVar("ui");
  const inputVal = arrayRead(inputName, ref(iVar));
  const outputVal: IRNode = { id: -1, kind: "unary", op, input: inputVal, valueType: "scalar", dataType: "f32" };

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
  // Use physical size for distributed output blocks
  const physOutCols = getPhysCols(outputName, outCols);
  const outSize = outRows * physOutCols;
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
  if (op === "copy") return rhs;
  return binOp(op as any, lhs, rhs, "f32");
}
