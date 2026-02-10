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
    values[i] = (i % 13) - 6;
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

export function createMatmulSuite(ops: BackendOps): BenchCase[] {
  return [
    // Square GEMM (common benchmark sizes)
    matmulCase(ops, "matmul.128x128", 128, 128, 128),
    matmulCase(ops, "matmul.256x256", 256, 256, 256),
    matmulCase(ops, "matmul.512x512", 512, 512, 512),
    matmulCase(ops, "matmul.1024x1024", 1024, 1024, 1024),
    matmulCase(ops, "matmul.2048x2048", 2048, 2048, 2048),

    // Tall-skinny (common in MLPs: large batch, small hidden)
    matmulCase(ops, "matmul.tall.4096x256x4096", 4096, 256, 4096),
    matmulCase(ops, "matmul.tall.2048x128x2048", 2048, 128, 2048),

    // Short-wide (common in attention: small seq, large hidden)
    matmulCase(ops, "matmul.wide.256x1024x256", 256, 1024, 256),

    // Non-tile-aligned (stress test edge cases)
    matmulCase(ops, "matmul.odd.100x150x80", 100, 150, 80),
    matmulCase(ops, "matmul.odd.333x444x555", 333, 444, 555),

    // GEMV-like (batch=1 or output=1)
    matmulCase(ops, "matmul.gemv.1x1024x1024", 1, 1024, 1024),
    matmulCase(ops, "matmul.gemv.1024x1024x1", 1024, 1024, 1),
  ];
}
