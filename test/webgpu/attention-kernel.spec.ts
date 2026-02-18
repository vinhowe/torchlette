import { describe, expect, it } from "vitest";
import { Torchlette } from "../../src";
import { initWebGPU } from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("fused flash attention", { timeout: 60000 }, () => {
  /**
   * Helper: compute decomposed attention on CPU for reference.
   * q, k, v: [B, H, N, D] as flat arrays
   */
  function decomposedAttention(
    qData: number[], kData: number[], vData: number[],
    B: number, H: number, N: number, D: number,
    scale: number, isCausal: boolean,
  ): number[] {
    const output = new Float32Array(B * H * N * D);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        for (let i = 0; i < N; i++) {
          // Compute scores for row i
          const scores: number[] = [];
          for (let j = 0; j < N; j++) {
            if (isCausal && j > i) {
              scores.push(-Infinity);
            } else {
              let dot = 0;
              for (let d = 0; d < D; d++) {
                const qIdx = ((b * H + h) * N + i) * D + d;
                const kIdx = ((b * H + h) * N + j) * D + d;
                dot += qData[qIdx] * kData[kIdx];
              }
              scores.push(dot * scale);
            }
          }

          // Softmax
          const maxScore = Math.max(...scores);
          const expScores = scores.map(s => Math.exp(s - maxScore));
          const sumExp = expScores.reduce((a, b) => a + b, 0);
          const attnWeights = expScores.map(e => e / sumExp);

          // Weighted sum of V
          for (let d = 0; d < D; d++) {
            let val = 0;
            for (let j = 0; j < N; j++) {
              const vIdx = ((b * H + h) * N + j) * D + d;
              val += attnWeights[j] * vData[vIdx];
            }
            output[((b * H + h) * N + i) * D + d] = val;
          }
        }
      }
    }
    return Array.from(output);
  }

  it("forward matches CPU decomposed path (small, non-causal)", async () => {
    await initWebGPU();

    const B = 1, H = 1, N = 4, D = 8;
    const scale = 1.0 / Math.sqrt(D);

    // Generate deterministic test data
    const qData: number[] = [];
    const kData: number[] = [];
    const vData: number[] = [];
    for (let i = 0; i < B * H * N * D; i++) {
      qData.push(Math.sin(i * 0.1) * 0.5);
      kData.push(Math.cos(i * 0.13) * 0.5);
      vData.push(Math.sin(i * 0.17 + 1) * 0.5);
    }

    // GPU fused path
    const gpu = new Torchlette("webgpu");
    const q = gpu.tensorFromArray(qData, [B, H, N, D]);
    const k = gpu.tensorFromArray(kData, [B, H, N, D]);
    const v = gpu.tensorFromArray(vData, [B, H, N, D]);

    const result = gpu.scaledDotProductAttention(q, k, v, scale, false);
    const gpuArr = await result.cpu();

    // CPU reference
    const cpuArr = decomposedAttention(qData, kData, vData, B, H, N, D, scale, false);

    console.log("GPU (non-causal):", gpuArr.slice(0, 8));
    console.log("CPU (non-causal):", cpuArr.slice(0, 8));

    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-3);
    }

    gpu.markStep();
  });

  it("forward matches CPU decomposed path (causal)", async () => {
    await initWebGPU();

    const B = 1, H = 2, N = 8, D = 16;
    const scale = 1.0 / Math.sqrt(D);

    const qData: number[] = [];
    const kData: number[] = [];
    const vData: number[] = [];
    for (let i = 0; i < B * H * N * D; i++) {
      qData.push(Math.sin(i * 0.07) * 0.3);
      kData.push(Math.cos(i * 0.11) * 0.3);
      vData.push(Math.sin(i * 0.13 + 2) * 0.3);
    }

    const gpu = new Torchlette("webgpu");
    const q = gpu.tensorFromArray(qData, [B, H, N, D]);
    const k = gpu.tensorFromArray(kData, [B, H, N, D]);
    const v = gpu.tensorFromArray(vData, [B, H, N, D]);

    const result = gpu.scaledDotProductAttention(q, k, v, scale, true);
    const gpuArr = await result.cpu();

    const cpuArr = decomposedAttention(qData, kData, vData, B, H, N, D, scale, true);

    console.log("GPU (causal):", gpuArr.slice(0, 8));
    console.log("CPU (causal):", cpuArr.slice(0, 8));

    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-3);
    }

    gpu.markStep();
  });

  it("forward works with multi-batch, multi-head", async () => {
    await initWebGPU();

    const B = 2, H = 4, N = 16, D = 16;
    const scale = 1.0 / Math.sqrt(D);

    const qData: number[] = [];
    const kData: number[] = [];
    const vData: number[] = [];
    for (let i = 0; i < B * H * N * D; i++) {
      qData.push(Math.sin(i * 0.03) * 0.2);
      kData.push(Math.cos(i * 0.05) * 0.2);
      vData.push(Math.sin(i * 0.07 + 3) * 0.2);
    }

    const gpu = new Torchlette("webgpu");
    const q = gpu.tensorFromArray(qData, [B, H, N, D]);
    const k = gpu.tensorFromArray(kData, [B, H, N, D]);
    const v = gpu.tensorFromArray(vData, [B, H, N, D]);

    const result = gpu.scaledDotProductAttention(q, k, v, scale, true);
    const gpuArr = await result.cpu();

    const cpuArr = decomposedAttention(qData, kData, vData, B, H, N, D, scale, true);

    let maxError = 0;
    for (let i = 0; i < gpuArr.length; i++) {
      maxError = Math.max(maxError, Math.abs(gpuArr[i] - cpuArr[i]));
    }
    console.log(`Multi-batch/head max error: ${maxError}`);
    expect(maxError).toBeLessThan(1e-3);

    gpu.markStep();
  });

  it("forward works with seq_len=64 (full tile)", async () => {
    await initWebGPU();

    const B = 1, H = 1, N = 64, D = 16;
    const scale = 1.0 / Math.sqrt(D);

    const qData: number[] = [];
    const kData: number[] = [];
    const vData: number[] = [];
    for (let i = 0; i < B * H * N * D; i++) {
      qData.push(Math.sin(i * 0.02) * 0.1);
      kData.push(Math.cos(i * 0.03) * 0.1);
      vData.push(Math.sin(i * 0.04 + 1) * 0.1);
    }

    const gpu = new Torchlette("webgpu");
    const q = gpu.tensorFromArray(qData, [B, H, N, D]);
    const k = gpu.tensorFromArray(kData, [B, H, N, D]);
    const v = gpu.tensorFromArray(vData, [B, H, N, D]);

    const result = gpu.scaledDotProductAttention(q, k, v, scale, true);
    const gpuArr = await result.cpu();

    const cpuArr = decomposedAttention(qData, kData, vData, B, H, N, D, scale, true);

    let maxError = 0;
    for (let i = 0; i < gpuArr.length; i++) {
      maxError = Math.max(maxError, Math.abs(gpuArr[i] - cpuArr[i]));
    }
    console.log(`seq_len=64 max error: ${maxError}`);
    expect(maxError).toBeLessThan(1e-3);

    gpu.markStep();
  });

  it("forward works with seq_len=128 (multiple tiles)", async () => {
    await initWebGPU();

    const B = 1, H = 2, N = 128, D = 32;
    const scale = 1.0 / Math.sqrt(D);

    const qData: number[] = [];
    const kData: number[] = [];
    const vData: number[] = [];
    for (let i = 0; i < B * H * N * D; i++) {
      qData.push(Math.sin(i * 0.01) * 0.1);
      kData.push(Math.cos(i * 0.015) * 0.1);
      vData.push(Math.sin(i * 0.02 + 1) * 0.1);
    }

    const gpu = new Torchlette("webgpu");
    const q = gpu.tensorFromArray(qData, [B, H, N, D]);
    const k = gpu.tensorFromArray(kData, [B, H, N, D]);
    const v = gpu.tensorFromArray(vData, [B, H, N, D]);

    const result = gpu.scaledDotProductAttention(q, k, v, scale, true);
    const gpuArr = await result.cpu();

    const cpuArr = decomposedAttention(qData, kData, vData, B, H, N, D, scale, true);

    let maxError = 0;
    for (let i = 0; i < gpuArr.length; i++) {
      maxError = Math.max(maxError, Math.abs(gpuArr[i] - cpuArr[i]));
    }
    console.log(`seq_len=128 max error: ${maxError}`);
    expect(maxError).toBeLessThan(1e-3);

    gpu.markStep();
  });

  it("backward produces correct gradients", async () => {
    await initWebGPU();

    const B = 1, H = 1, N = 4, D = 8;
    const scale = 1.0 / Math.sqrt(D);

    const qData: number[] = [];
    const kData: number[] = [];
    const vData: number[] = [];
    for (let i = 0; i < B * H * N * D; i++) {
      qData.push(Math.sin(i * 0.1) * 0.3);
      kData.push(Math.cos(i * 0.13) * 0.3);
      vData.push(Math.sin(i * 0.17 + 1) * 0.3);
    }

    // GPU backward
    const gpu = new Torchlette("webgpu");
    const qGpu = gpu.tensorFromArray(qData, [B, H, N, D], { requiresGrad: true });
    const kGpu = gpu.tensorFromArray(kData, [B, H, N, D], { requiresGrad: true });
    const vGpu = gpu.tensorFromArray(vData, [B, H, N, D], { requiresGrad: true });

    const gpuResult = gpu.scaledDotProductAttention(qGpu, kGpu, vGpu, scale, true);
    const gpuLoss = gpuResult.sum();
    await gpuLoss.backward();

    const gpuDQ = await qGpu.grad!.cpu();
    const gpuDK = await kGpu.grad!.cpu();
    const gpuDV = await vGpu.grad!.cpu();

    // CPU backward (decomposed) for reference
    const cpu = new Torchlette("cpu");
    const qCpu = cpu.tensorFromArray(qData, [B, H, N, D], { requiresGrad: true });
    const kCpu = cpu.tensorFromArray(kData, [B, H, N, D], { requiresGrad: true });
    const vCpu = cpu.tensorFromArray(vData, [B, H, N, D], { requiresGrad: true });

    const cpuResult = cpu.scaledDotProductAttention(qCpu, kCpu, vCpu, scale, true);
    const cpuLoss = cpuResult.sum();
    await cpuLoss.backward();

    const cpuDQ = await qCpu.grad!.cpu();
    const cpuDK = await kCpu.grad!.cpu();
    const cpuDV = await vCpu.grad!.cpu();

    console.log("GPU dQ:", gpuDQ.slice(0, 8));
    console.log("CPU dQ:", cpuDQ.slice(0, 8));
    console.log("GPU dK:", gpuDK.slice(0, 8));
    console.log("CPU dK:", cpuDK.slice(0, 8));
    console.log("GPU dV:", gpuDV.slice(0, 8));
    console.log("CPU dV:", cpuDV.slice(0, 8));

    // Compare dQ
    let maxDQ = 0;
    for (let i = 0; i < gpuDQ.length; i++) {
      maxDQ = Math.max(maxDQ, Math.abs(gpuDQ[i] - cpuDQ[i]));
    }
    console.log(`max dQ error: ${maxDQ}`);
    expect(maxDQ).toBeLessThan(1e-2);

    // Compare dK
    let maxDK = 0;
    for (let i = 0; i < gpuDK.length; i++) {
      maxDK = Math.max(maxDK, Math.abs(gpuDK[i] - cpuDK[i]));
    }
    console.log(`max dK error: ${maxDK}`);
    expect(maxDK).toBeLessThan(1e-2);

    // Compare dV
    let maxDV = 0;
    for (let i = 0; i < gpuDV.length; i++) {
      maxDV = Math.max(maxDV, Math.abs(gpuDV[i] - cpuDV[i]));
    }
    console.log(`max dV error: ${maxDV}`);
    expect(maxDV).toBeLessThan(1e-2);

    gpu.markStep();
    cpu.markStep();
  });

  it("backward with larger causal attention", async () => {
    await initWebGPU();

    const B = 1, H = 2, N = 16, D = 16;
    const scale = 1.0 / Math.sqrt(D);

    const qData: number[] = [];
    const kData: number[] = [];
    const vData: number[] = [];
    for (let i = 0; i < B * H * N * D; i++) {
      qData.push(Math.sin(i * 0.05) * 0.2);
      kData.push(Math.cos(i * 0.07) * 0.2);
      vData.push(Math.sin(i * 0.09 + 2) * 0.2);
    }

    // GPU backward
    const gpu = new Torchlette("webgpu");
    const qGpu = gpu.tensorFromArray(qData, [B, H, N, D], { requiresGrad: true });
    const kGpu = gpu.tensorFromArray(kData, [B, H, N, D], { requiresGrad: true });
    const vGpu = gpu.tensorFromArray(vData, [B, H, N, D], { requiresGrad: true });

    const gpuResult = gpu.scaledDotProductAttention(qGpu, kGpu, vGpu, scale, true);
    const gpuLoss = gpuResult.sum();
    await gpuLoss.backward();

    const gpuDQ = await qGpu.grad!.cpu();
    const gpuDK = await kGpu.grad!.cpu();
    const gpuDV = await vGpu.grad!.cpu();

    // CPU backward
    const cpu = new Torchlette("cpu");
    const qCpu = cpu.tensorFromArray(qData, [B, H, N, D], { requiresGrad: true });
    const kCpu = cpu.tensorFromArray(kData, [B, H, N, D], { requiresGrad: true });
    const vCpu = cpu.tensorFromArray(vData, [B, H, N, D], { requiresGrad: true });

    const cpuResult = cpu.scaledDotProductAttention(qCpu, kCpu, vCpu, scale, true);
    const cpuLoss = cpuResult.sum();
    await cpuLoss.backward();

    const cpuDQ = await qCpu.grad!.cpu();
    const cpuDK = await kCpu.grad!.cpu();
    const cpuDV = await vCpu.grad!.cpu();

    let maxDQ = 0, maxDK = 0, maxDV = 0;
    for (let i = 0; i < gpuDQ.length; i++) {
      maxDQ = Math.max(maxDQ, Math.abs(gpuDQ[i] - cpuDQ[i]));
    }
    for (let i = 0; i < gpuDK.length; i++) {
      maxDK = Math.max(maxDK, Math.abs(gpuDK[i] - cpuDK[i]));
    }
    for (let i = 0; i < gpuDV.length; i++) {
      maxDV = Math.max(maxDV, Math.abs(gpuDV[i] - cpuDV[i]));
    }

    console.log(`Larger causal backward - max errors: dQ=${maxDQ}, dK=${maxDK}, dV=${maxDV}`);
    expect(maxDQ).toBeLessThan(5e-2);
    expect(maxDK).toBeLessThan(5e-2);
    expect(maxDV).toBeLessThan(5e-2);

    gpu.markStep();
    cpu.markStep();
  });
});
