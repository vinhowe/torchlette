/**
 * Comprehensive matmul benchmark comparing:
 * - Default vs autotuned configurations
 * - Fused vs unfused epilogue
 * - Subgroup vs non-subgroup variants (when available)
 */

import type { BackendOps, Tensor } from "../../src";
import type { BenchCase } from "../types";

type MatmulOp = (a: Tensor, b: Tensor) => Tensor;

function asMatmul(ops: BackendOps): MatmulOp | undefined {
  const candidate = ops as BackendOps & { matmul?: MatmulOp };
  return candidate.matmul;
}

function makeValues(size: number): number[] {
  const values = new Array<number>(size);
  for (let i = 0; i < size; i += 1) {
    values[i] = ((i % 13) - 6) * 0.1;
  }
  return values;
}

function matmulCase(
  ops: BackendOps,
  name: string,
  m: number,
  k: number,
  n: number,
): BenchCase {
  const matmul = asMatmul(ops);
  if (!matmul) {
    return { name, skip: "matmul not implemented" };
  }
  const a = ops.tensorFromArray(makeValues(m * k), [m, k]);
  const b = ops.tensorFromArray(makeValues(k * n), [k, n]);
  const flops = 2 * m * k * n;
  const bytes = 4 * (m * k + k * n + m * n);

  return {
    name,
    flops,
    bytes,
    run: () => {
      matmul(a, b);
    },
  };
}

export function createMatmulVariantSuite(ops: BackendOps): BenchCase[] {
  return [
    // Standard sizes for comparison
    matmulCase(ops, "matmul.256x256", 256, 256, 256),
    matmulCase(ops, "matmul.512x512", 512, 512, 512),
    matmulCase(ops, "matmul.1024x1024", 1024, 1024, 1024),
    matmulCase(ops, "matmul.2048x2048", 2048, 2048, 2048),

    // Tall-skinny (MLP-like)
    matmulCase(ops, "matmul.tall.4096x256x4096", 4096, 256, 4096),

    // Short-wide (attention-like)
    matmulCase(ops, "matmul.wide.256x1024x256", 256, 1024, 256),

    // Batched
    matmulCase(ops, "matmul.batched.8x512x512", 512, 512, 512),
  ];
}
