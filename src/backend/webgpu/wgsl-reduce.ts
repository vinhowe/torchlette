/**
 * WGSL Parallel Reduction Codegen
 *
 * Generates the workgroup-level tree reduction pattern used by cross-entropy,
 * layernorm, and attention kernels: strided accumulation loop → shared memory
 * store → barrier → power-of-2 tree reduction → result binding.
 *
 * Supports single-channel (most cases) and multi-channel (layernorm backward
 * c1+c2) reductions. All 12 reduction instances across the kernel files use
 * this single codegen function.
 */

/** A single reduction channel (accumulator + shared memory array). */
export interface ReduceChannel {
  /** Shared memory array name (must be declared separately in the shader). */
  smem: string;
  /** Initial accumulator value (e.g. "0.0" for sum, "-3.402823e+38" for max). */
  init: string;
  /** Expression to accumulate per element. May reference the loop variable. */
  accumExpr: string;
  /** Result variable name (becomes a `let` binding after reduction). */
  result: string;
  /** Optional transform on the reduced value. Use `_` as placeholder for smem[0]. */
  transform?: string;
}

interface WgslReduceBase {
  /** Workgroup size (must be power of 2). */
  wgSize: number;
  /** Thread ID variable name in the shader (e.g. "tid"). */
  tid: string;
  /** Dimension size expression in WGSL (e.g. "V", "D", "D_dim"). */
  dim: string;
  /** Loop variable name (default: "i"). */
  loopVar?: string;
  /** Reduction operator. */
  op: "sum" | "max";
  /** Extra WGSL statements at start of each loop iteration, before accumulation. */
  loopPreamble?: string;
}

/** Single-channel reduction (flat fields). */
type WgslReduceSingle = WgslReduceBase & ReduceChannel;

/** Multi-channel reduction (channels array). */
type WgslReduceMulti = WgslReduceBase & { channels: ReduceChannel[] };

export type WgslReduceOpts = WgslReduceSingle | WgslReduceMulti;

/**
 * Generate WGSL for a parallel workgroup reduction.
 *
 * Returns a block of WGSL code with 2-space base indent (matching shader
 * function body). Insert at column 0 in template strings:
 *
 * ```typescript
 * return `
 * ...
 * fn main(...) {
 *   let tid = lid.x;
 *
 * ${wgslReduce({ wgSize: 256, tid: "tid", dim: "V", op: "max", ... })}
 *
 *   // rest of shader
 * }
 * `;
 * ```
 */
export function wgslReduce(opts: WgslReduceOpts): string {
  const { wgSize, tid, dim, op } = opts;
  const loopVar = opts.loopVar ?? "i";
  const half = wgSize / 2;

  // Normalize to channels array
  const channels: ReduceChannel[] = "channels" in opts
    ? opts.channels
    : [{ smem: opts.smem, init: opts.init, accumExpr: opts.accumExpr, result: opts.result, transform: opts.transform }];

  const lines: string[] = [];

  // Block-scope the accumulators so multiple reductions in one shader don't
  // conflict on `var _acc` declarations (WGSL forbids redeclaration in same scope).
  lines.push(`  {`);
  for (let c = 0; c < channels.length; c++) {
    const name = channels.length === 1 ? "_acc" : `_acc${c}`;
    lines.push(`    var ${name} = ${channels[c].init};`);
  }
  lines.push(`    for (var ${loopVar} = ${tid}; ${loopVar} < ${dim}; ${loopVar} += ${wgSize}u) {`);
  if (opts.loopPreamble) {
    for (const pLine of opts.loopPreamble.split("\n")) {
      lines.push(`      ${pLine.trim()}`);
    }
  }
  for (let c = 0; c < channels.length; c++) {
    const name = channels.length === 1 ? "_acc" : `_acc${c}`;
    if (op === "sum") {
      lines.push(`      ${name} += ${channels[c].accumExpr};`);
    } else {
      lines.push(`      ${name} = max(${name}, ${channels[c].accumExpr});`);
    }
  }
  lines.push(`    }`);
  for (let c = 0; c < channels.length; c++) {
    const name = channels.length === 1 ? "_acc" : `_acc${c}`;
    lines.push(`    ${channels[c].smem}[${tid}] = ${name};`);
  }
  lines.push(`  }`);
  lines.push(`  workgroupBarrier();`);
  lines.push(``);

  // Tree reduction
  lines.push(`  for (var s = ${half}u; s > 0u; s >>= 1u) {`);
  lines.push(`    if (${tid} < s) {`);
  for (let c = 0; c < channels.length; c++) {
    const sm = channels[c].smem;
    if (op === "sum") {
      lines.push(`      ${sm}[${tid}] += ${sm}[${tid} + s];`);
    } else {
      lines.push(`      ${sm}[${tid}] = max(${sm}[${tid}], ${sm}[${tid} + s]);`);
    }
  }
  lines.push(`    }`);
  lines.push(`    workgroupBarrier();`);
  lines.push(`  }`);

  // Result bindings
  for (let c = 0; c < channels.length; c++) {
    const ch = channels[c];
    const raw = `${ch.smem}[0]`;
    const value = ch.transform ? ch.transform.replace(/_/g, raw) : raw;
    lines.push(`  let ${ch.result} = ${value};`);
  }

  return lines.join("\n");
}
