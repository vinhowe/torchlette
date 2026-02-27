/**
 * Tile-IR Memory Access Pattern Analysis
 *
 * Static analysis pass that examines load/store index expressions in tile-IR
 * to determine memory access patterns: coalescing, stride, and safe vector width.
 *
 * Inspired by Triton's AxisInfoAnalysis — tracks how thread indices map to
 * memory addresses to automatically detect coalesced access and select
 * optimal vector widths. Tracks divisibility and constant terms for alignment.
 */

import type {
  IRNode, TileKernelSpec, Statement,
} from "./tile-ir";
import { buildKernelIR } from "./tile-ir";

// ============================================================================
// Types
// ============================================================================

/** Describes how thread indices map to memory addresses for a load/store. */
export interface AccessPattern {
  /** The buffer binding name. */
  bindingName: string;
  /** "load" or "store". */
  accessType: "load" | "store";
  /** Stride of consecutive threadIdx(0) / localIndex / globalId(0) values.
   *  1 = coalesced, 0 = broadcast, N>1 = strided, "unknown" = data-dependent. */
  innerStride: number | "unknown";
  /** Whether consecutive threads in the x-dimension access consecutive addresses. */
  isCoalesced: boolean;
  /** Maximum safe vector width (1, 2, or 4) based on contiguity + alignment analysis. */
  maxVecWidth: 1 | 2 | 4;
  /** GCD of the base address expression (for alignment reasoning). */
  baseDivisibility: number | null;
  /** Constant term in the address expression. */
  baseConstantTerm: number | null;
  /** IR node ID of the load/store for diagnostics. */
  nodeId: number;
}

// ============================================================================
// Symbolic Stride Evaluation
// ============================================================================

/**
 * Symbolic representation of a linear expression in thread/workgroup IDs:
 *   value = constantTerm + innerCoeff * threadIdx.x + (other terms)
 *
 * We track the coefficient of the "inner thread dimension" (threadIdx.x,
 * localIndex % wgSizeX, or globalId.x) since that determines coalescing,
 * plus divisibility and constant term for alignment reasoning.
 */
interface SymbolicExpr {
  /** Coefficient of the innermost thread index (threadIdx.x / globalId.x). */
  innerCoeff: number | "unknown";
  /** Whether the expression has any data-dependent terms (loop vars, loads). */
  hasDataDep: boolean;
  /** Additive constant in the expression (null if unknown). */
  constantTerm: number | null;
  /** GCD divisibility of the entire expression (null if unknown).
   *  For alignment: if divisibility >= 4, vec4 loads are safe. */
  divisibility: number | null;
}

function gcd(a: number, b: number): number {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

function gcdN(...vals: number[]): number {
  let result = vals[0];
  for (let i = 1; i < vals.length; i++) {
    result = gcd(result, vals[i]);
  }
  return result;
}

const UNKNOWN: SymbolicExpr = { innerCoeff: "unknown", hasDataDep: true, constantTerm: null, divisibility: null };
const ZERO: SymbolicExpr = { innerCoeff: 0, hasDataDep: false, constantTerm: 0, divisibility: 0 };

/**
 * Evaluate the symbolic stride of an IR expression with respect to the
 * innermost thread dimension.
 *
 * Returns the coefficient of threadIdx(0) / globalId(0) in the expression,
 * plus divisibility and constant term for alignment reasoning.
 */
function evalSymbolic(node: IRNode, wgSizeX: number): SymbolicExpr {
  switch (node.kind) {
    case "globalId":
      // globalId(0) = programId(0) * wgSizeX + threadIdx(0)
      // innerCoeff = 1, constantTerm = 0, divisibility = 1
      return node.dim === 0
        ? { innerCoeff: 1, hasDataDep: false, constantTerm: 0, divisibility: 1 }
        : ZERO;

    case "threadIdx":
      return node.dim === 0
        ? { innerCoeff: 1, hasDataDep: false, constantTerm: 0, divisibility: 1 }
        : ZERO;

    case "localIndex":
      // localIndex = threadIdx.x + threadIdx.y * wgSizeX
      // Coefficient of threadIdx.x is 1
      return { innerCoeff: 1, hasDataDep: false, constantTerm: 0, divisibility: 1 };

    case "programId":
      // Workgroup ID — constant within a workgroup, stride 0 w.r.t. threads
      // Divisibility depends on context (wgSizeX for dim 0)
      return node.dim === 0
        ? { innerCoeff: 0, hasDataDep: false, constantTerm: null, divisibility: null }
        : { innerCoeff: 0, hasDataDep: false, constantTerm: null, divisibility: null };

    case "uniform":
      // Uniform: constant within workgroup, unknown value
      return { innerCoeff: 0, hasDataDep: false, constantTerm: null, divisibility: null };

    case "const": {
      const v = node.value;
      return { innerCoeff: 0, hasDataDep: false, constantTerm: v, divisibility: v === 0 ? 0 : Math.abs(v) };
    }

    case "binary": {
      const lhs = evalSymbolic(node.lhs, wgSizeX);
      const rhs = evalSymbolic(node.rhs, wgSizeX);

      switch (node.op) {
        case "add":
        case "sub": {
          if (lhs.innerCoeff === "unknown" || rhs.innerCoeff === "unknown") {
            return { innerCoeff: "unknown", hasDataDep: lhs.hasDataDep || rhs.hasDataDep, constantTerm: null, divisibility: null };
          }
          const coeff = node.op === "add"
            ? lhs.innerCoeff + rhs.innerCoeff
            : lhs.innerCoeff - rhs.innerCoeff;
          // constantTerm: add/sub of known constants
          const ct = (lhs.constantTerm !== null && rhs.constantTerm !== null)
            ? (node.op === "add" ? lhs.constantTerm + rhs.constantTerm : lhs.constantTerm - rhs.constantTerm)
            : null;
          // divisibility: gcd of both sides' divisibilities
          const div = computeDivisibilityBinary(lhs.divisibility, rhs.divisibility, coeff);
          return { innerCoeff: coeff, hasDataDep: lhs.hasDataDep || rhs.hasDataDep, constantTerm: ct, divisibility: div };
        }

        case "mul": {
          // (a + bx) * (c + dx)
          // If one side is constant (coeff=0, no data dep), multiply
          if (lhs.innerCoeff === 0 && !lhs.hasDataDep) {
            const constVal = extractConstant(node.lhs);
            if (constVal !== null && rhs.innerCoeff !== "unknown") {
              const coeff = constVal * rhs.innerCoeff;
              const ct = (rhs.constantTerm !== null) ? constVal * rhs.constantTerm : null;
              const div = (rhs.divisibility !== null) ? Math.abs(constVal) * rhs.divisibility : null;
              return { innerCoeff: coeff, hasDataDep: rhs.hasDataDep, constantTerm: ct, divisibility: div };
            }
            if (rhs.innerCoeff === 0) {
              // Both constant w.r.t. threads
              const ct = (lhs.constantTerm !== null && rhs.constantTerm !== null) ? lhs.constantTerm * rhs.constantTerm : null;
              const div = (lhs.divisibility !== null && rhs.divisibility !== null)
                ? lhs.divisibility * rhs.divisibility : null;
              return { innerCoeff: 0, hasDataDep: lhs.hasDataDep || rhs.hasDataDep, constantTerm: ct, divisibility: div };
            }
          }
          if (rhs.innerCoeff === 0 && !rhs.hasDataDep) {
            const constVal = extractConstant(node.rhs);
            if (constVal !== null && lhs.innerCoeff !== "unknown") {
              const coeff = constVal * lhs.innerCoeff;
              const ct = (lhs.constantTerm !== null) ? constVal * lhs.constantTerm : null;
              const div = (lhs.divisibility !== null) ? Math.abs(constVal) * lhs.divisibility : null;
              return { innerCoeff: coeff, hasDataDep: lhs.hasDataDep, constantTerm: ct, divisibility: div };
            }
            if (lhs.innerCoeff === 0) {
              const ct = (lhs.constantTerm !== null && rhs.constantTerm !== null) ? lhs.constantTerm * rhs.constantTerm : null;
              const div = (lhs.divisibility !== null && rhs.divisibility !== null)
                ? lhs.divisibility * rhs.divisibility : null;
              return { innerCoeff: 0, hasDataDep: lhs.hasDataDep || rhs.hasDataDep, constantTerm: ct, divisibility: div };
            }
          }
          return UNKNOWN;
        }

        case "div":
        case "mod":
        case "shr":
        case "shl":
        case "and":
        case "or":
        case "xor":
          // These break linearity — stride becomes unknown unless both sides are constant
          if (lhs.innerCoeff === 0 && rhs.innerCoeff === 0 && !lhs.hasDataDep && !rhs.hasDataDep) {
            return ZERO;
          }
          return UNKNOWN;

        default:
          return UNKNOWN;
      }
    }

    case "cast":
      // Casts preserve stride (e.g. u32→i32)
      return evalSymbolic(node.input, wgSizeX);

    case "unary":
      // Unary ops (neg, abs, etc.) don't preserve linear stride
      if (node.op === "neg") {
        const inner = evalSymbolic(node.input, wgSizeX);
        if (inner.innerCoeff === "unknown") return UNKNOWN;
        return {
          innerCoeff: -inner.innerCoeff,
          hasDataDep: inner.hasDataDep,
          constantTerm: inner.constantTerm !== null ? -inner.constantTerm : null,
          divisibility: inner.divisibility,
        };
      }
      return UNKNOWN;

    case "select":
      return UNKNOWN;

    case "cmp":
      return UNKNOWN;

    case "load":
    case "sharedRead":
    case "arrayRead":
      // Data-dependent — stride is unknown
      return UNKNOWN;

    case "numWorkgroups":
      // Grid dimensions — constant within a workgroup, unknown value
      return { innerCoeff: 0, hasDataDep: false, constantTerm: null, divisibility: null };

    case "namedRef":
      // Named refs are typically loop variables — data-dependent
      return { innerCoeff: 0, hasDataDep: true, constantTerm: null, divisibility: null };

    default:
      return UNKNOWN;
  }
}

/**
 * Compute divisibility for add/sub of two symbolic expressions.
 * The divisibility of a sum is the GCD of the divisibilities.
 */
function computeDivisibilityBinary(lhsDiv: number | null, rhsDiv: number | null, _coeff: number): number | null {
  if (lhsDiv === null || rhsDiv === null) return null;
  if (lhsDiv === 0) return rhsDiv;
  if (rhsDiv === 0) return lhsDiv;
  return gcd(lhsDiv, rhsDiv);
}

/**
 * Extract a compile-time constant value from an IR node.
 */
function extractConstant(node: IRNode): number | null {
  if (node.kind === "const") return node.value;
  return null;
}

// ============================================================================
// Access Pattern Extraction
// ============================================================================

/**
 * Analyze all memory access patterns in a tile kernel spec.
 *
 * Walks the statement tree, finds load nodes and store statements,
 * and evaluates their index expressions symbolically to determine
 * coalescing and vector width.
 */
export function analyzeAccessPatterns(spec: TileKernelSpec): AccessPattern[] {
  const ctx = buildKernelIR(spec);
  const wgSizeX = typeof spec.workgroupSize === "number"
    ? spec.workgroupSize
    : spec.workgroupSize[0];

  const patterns: AccessPattern[] = [];

  // Walk statements for stores
  walkStatements(ctx.statements, wgSizeX, patterns);

  // Walk all nodes for inline loads (LoadNode in expressions)
  for (const node of ctx.nodes) {
    if (node.kind === "load") {
      const sym = evalSymbolic(node.offsets, wgSizeX);
      patterns.push(makePattern(node.binding, "load", sym, node.id));
    }
  }

  return patterns;
}

function walkStatements(stmts: Statement[], wgSizeX: number, patterns: AccessPattern[]): void {
  for (const stmt of stmts) {
    switch (stmt.kind) {
      case "directStore": {
        const sym = evalSymbolic(stmt.idx, wgSizeX);
        patterns.push(makePattern(stmt.binding, "store", sym, stmt.idx.id));
        break;
      }
      case "guardedStore": {
        const sym = evalSymbolic(stmt.idx, wgSizeX);
        patterns.push(makePattern(stmt.binding, "store", sym, stmt.idx.id));
        break;
      }
      case "forRange":
        walkStatements(stmt.body, wgSizeX, patterns);
        break;
      case "if":
        walkStatements(stmt.body, wgSizeX, patterns);
        break;
      case "ifElse":
        walkStatements(stmt.body, wgSizeX, patterns);
        walkStatements(stmt.elseBody, wgSizeX, patterns);
        break;
    }
  }
}

function makePattern(
  bindingName: string,
  accessType: "load" | "store",
  sym: SymbolicExpr,
  nodeId: number,
): AccessPattern {
  const innerStride = sym.innerCoeff;
  const isCoalesced = innerStride === 1;
  let maxVecWidth: 1 | 2 | 4 = 1;
  if (isCoalesced && !sym.hasDataDep) {
    // Coalesced + no data deps → vec4 by default (threads access contiguous addresses).
    // Use divisibility to *further* constrain when we know the base alignment:
    // - If base divisibility >= 4 (or unknown/null) → vec4
    // - If base divisibility is known but < 4 → use it to limit
    // Note: divisibility of the entire expression (including stride*threadId) is
    // dominated by the stride. For stride=1, the base alignment determines vec width.
    const ct = sym.constantTerm;
    if (ct !== null && ct % 4 !== 0) {
      // Known constant offset that's not 4-aligned
      if (ct % 2 === 0) {
        maxVecWidth = 2;
      } else {
        maxVecWidth = 1;
      }
    } else {
      // constantTerm is null (unknown but uniform) or 4-aligned → vec4 safe
      maxVecWidth = 4;
    }
  } else if (isCoalesced) {
    // Coalesced but with data-dependent base — vec4 may still be safe
    // if the base is aligned, but we conservatively use 1
    maxVecWidth = 1;
  }
  return {
    bindingName, accessType, innerStride, isCoalesced, maxVecWidth,
    baseDivisibility: sym.divisibility, baseConstantTerm: sym.constantTerm,
    nodeId,
  };
}

// ============================================================================
// Diagnostic Report
// ============================================================================

/**
 * Produce a human-readable report of access patterns.
 */
export function reportAccessPatterns(spec: TileKernelSpec): string {
  const patterns = analyzeAccessPatterns(spec);
  const lines: string[] = [`Access Analysis for kernel "${spec.name}":`];

  for (const p of patterns) {
    const strideStr = p.innerStride === "unknown" ? "unknown"
      : p.innerStride === 0 ? "broadcast (stride=0)"
      : p.innerStride === 1 ? "coalesced (stride=1)"
      : `strided (stride=${p.innerStride})`;

    const vecStr = p.maxVecWidth > 1 ? `vec${p.maxVecWidth} OK` : "scalar only";
    const divStr = p.baseDivisibility !== null ? ` div=${p.baseDivisibility}` : "";
    const prefix = (!p.isCoalesced && p.innerStride !== 0 && p.innerStride !== "unknown")
      ? "\u26A0 " : "  ";

    lines.push(`${prefix}${p.accessType} ${p.bindingName} \u2014 ${strideStr}, ${vecStr}${divStr}`);
  }

  return lines.join("\n");
}

// ============================================================================
// Vec Width Selection
// ============================================================================

/**
 * Compute the maximum safe vectorization width for a kernel spec
 * based on its access patterns. Returns 1 if any access is non-coalesced
 * or has data-dependent indices.
 */
export function computeSafeVecWidth(spec: TileKernelSpec): 1 | 2 | 4 {
  const patterns = analyzeAccessPatterns(spec);
  if (patterns.length === 0) return 1;

  let minWidth: 1 | 2 | 4 = 4;
  for (const p of patterns) {
    if (p.maxVecWidth < minWidth) {
      minWidth = p.maxVecWidth as 1 | 2 | 4;
    }
  }
  return minWidth;
}
