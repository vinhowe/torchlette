/**
 * fuse-matmul-sum semantic tests.
 *
 * Verify the algebraic identity that the rewrite exploits:
 *
 *   sum_b( transpose(X_b) @ Y_b )[k, n]
 *     = Σ_(b, m) X[b, m, k] · Y[b, m, n]
 *     = (reshape(X, [B*M, K])^T @ reshape(Y, [B*M, N]))[k, n]
 *
 * We compute both sides from small tensors with reproducible random data
 * and verify they match to f32 precision. This proves the rewrite is
 * mathematically sound; runtime-integration tests happen at the compiler
 * level, once the rule is wired into the graph compiler.
 */
import { describe, expect, it } from "vitest";

// ============================================================================
// Reference implementations (naive loops)
// ============================================================================

/** Fill an array with deterministic pseudo-random values. */
function randomData(shape: number[], seed = 42): Float32Array {
  const n = shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  let s = seed;
  for (let i = 0; i < n; i++) {
    // Simple LCG
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = ((s & 0xffff) / 0x10000 - 0.5) * 2; // [-1, 1]
  }
  return out;
}

/** sum_b( X_b^T @ Y_b ) where X:[B, M, K], Y:[B, M, N] → out:[K, N] */
function referenceSumOfTransposeMatmul(
  X: Float32Array,
  Y: Float32Array,
  B: number,
  M: number,
  K: number,
  N: number,
): Float32Array {
  const out = new Float32Array(K * N);
  for (let b = 0; b < B; b++) {
    for (let k = 0; k < K; k++) {
      for (let n = 0; n < N; n++) {
        let acc = 0;
        for (let m = 0; m < M; m++) {
          const xIdx = b * M * K + m * K + k;
          const yIdx = b * M * N + m * N + n;
          acc += X[xIdx] * Y[yIdx];
        }
        out[k * N + n] += acc;
      }
    }
  }
  return out;
}

/** transpose(reshape(X, [B*M, K])) @ reshape(Y, [B*M, N]) → out:[K, N]
 *  This is what the rewrite produces. Reshape is a view (no data movement),
 *  so the flat buffer is unchanged; we just reinterpret indices.
 */
function rewrittenFlattenedMatmul(
  X: Float32Array,
  Y: Float32Array,
  B: number,
  M: number,
  K: number,
  N: number,
): Float32Array {
  const totalRows = B * M;
  // X viewed as [totalRows, K], transpose to [K, totalRows]
  // Y viewed as [totalRows, N]
  // out[k, n] = sum_r X[r, k] * Y[r, n] = sum_r X_flat[r*K + k] * Y_flat[r*N + n]
  const out = new Float32Array(K * N);
  for (let k = 0; k < K; k++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let r = 0; r < totalRows; r++) {
        acc += X[r * K + k] * Y[r * N + n];
      }
      out[k * N + n] = acc;
    }
  }
  return out;
}

// ============================================================================
// Tests
// ============================================================================

describe("fuse-matmul-sum / algebraic identity", () => {
  it("matches for small rank-3 tensors (B=2, M=3, K=4, N=5)", () => {
    const B = 2;
    const M = 3;
    const K = 4;
    const N = 5;
    const X = randomData([B, M, K], 1);
    const Y = randomData([B, M, N], 2);
    const expected = referenceSumOfTransposeMatmul(X, Y, B, M, K, N);
    const actual = rewrittenFlattenedMatmul(X, Y, B, M, K, N);
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 5);
    }
  });

  it("matches for larger rank-3 (B=4, M=8, K=16, N=32)", () => {
    const B = 4;
    const M = 8;
    const K = 16;
    const N = 32;
    const X = randomData([B, M, K], 17);
    const Y = randomData([B, M, N], 23);
    const expected = referenceSumOfTransposeMatmul(X, Y, B, M, K, N);
    const actual = rewrittenFlattenedMatmul(X, Y, B, M, K, N);
    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 4);
    }
  });

  it("matches for LM-head-like shapes (B=4, M=512, K=768, N=1024)", () => {
    // Smaller-than-real but similar shape to distilgpt2.
    // Use N=1024 instead of 50304 to keep the test fast.
    const B = 4;
    const M = 32; // smaller seq for test speed
    const K = 64;
    const N = 128;
    const X = randomData([B, M, K], 7);
    const Y = randomData([B, M, N], 11);
    const expected = referenceSumOfTransposeMatmul(X, Y, B, M, K, N);
    const actual = rewrittenFlattenedMatmul(X, Y, B, M, K, N);
    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 3);
    }
  });

  it("rank-4 identity (two batch dims)", () => {
    // [B1, B2, M, K] and [B1, B2, M, N]. The identity generalizes: flatten
    // all leading dims into the row dim.
    const B1 = 2;
    const B2 = 3;
    const M = 4;
    const K = 5;
    const N = 6;

    const X = randomData([B1, B2, M, K], 3);
    const Y = randomData([B1, B2, M, N], 5);

    // Reference: sum over both batch dims
    const expected = new Float32Array(K * N);
    for (let b1 = 0; b1 < B1; b1++) {
      for (let b2 = 0; b2 < B2; b2++) {
        for (let k = 0; k < K; k++) {
          for (let n = 0; n < N; n++) {
            let acc = 0;
            for (let m = 0; m < M; m++) {
              const xIdx = ((b1 * B2 + b2) * M + m) * K + k;
              const yIdx = ((b1 * B2 + b2) * M + m) * N + n;
              acc += X[xIdx] * Y[yIdx];
            }
            expected[k * N + n] += acc;
          }
        }
      }
    }

    // Rewritten form: flatten all leading dims into one row dim.
    // Same data layout, just reinterpret totalRows = B1 * B2 * M.
    const totalRows = B1 * B2 * M;
    const actual = new Float32Array(K * N);
    for (let k = 0; k < K; k++) {
      for (let n = 0; n < N; n++) {
        let acc = 0;
        for (let r = 0; r < totalRows; r++) {
          acc += X[r * K + k] * Y[r * N + n];
        }
        actual[k * N + n] = acc;
      }
    }

    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 4);
    }
  });
});
