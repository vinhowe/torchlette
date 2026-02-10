#!/usr/bin/env npx tsx
/**
 * Minimal LoRA test - single layer, fixed weights.
 * Compare with PyTorch output from test-minimal-lora.py
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

// PyTorch reference values from test-minimal-lora.py
const PYTORCH_REFERENCE = {
  base_out_sum: 3.440000295639038,
  lora_out_sum: 0.0,
  out_sum: 3.440000295639038,
  loss: 3.440000295639038,
  lora_A_grad_sum: 0.0,
  lora_B_grad_sum: 0.0,
  // Individual B gradients: each [0.768, -0.768]
  lora_B_grad_00: 0.7680000066757202,
};

async function runTest(device: 'cpu' | 'webgpu', enableMemoryPlanning: boolean): Promise<{
  base_out_sum: number;
  lora_out_sum: number;
  out_sum: number;
  loss: number;
  lora_A_grad_sum: number;
  lora_B_grad_sum: number;
  lora_B_grad_00: number;
}> {
  const api = new Torchlette(device, {
    enableFusion: false,
    enableMemoryPlanning,
  });

  const batch = 1;
  const seq = 4;
  const inFeatures = 8;
  const outFeatures = 8;
  const rank = 2;
  const alpha = 4.0;
  const scale = alpha / rank;

  // Fixed input (same as PyTorch)
  const xData = new Float32Array([
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ]);
  const x = api.tensorFromArray(xData, [batch, seq, inFeatures], { device });

  // Fixed base weight [out, in] (same as PyTorch)
  const baseWeightData = new Float32Array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,
    0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1,
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,
    0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0,
    0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3,
  ]);
  const baseWeight = api.tensorFromArray(baseWeightData, [outFeatures, inFeatures], { device });

  // LoRA A: [rank, in] (same as PyTorch)
  const loraAData = new Float32Array([
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
    -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08,
  ]);
  const loraA = api.tensorFromArray(loraAData, [rank, inFeatures], { device, requiresGrad: true });

  // LoRA B: [out, rank] - zeros (same as PyTorch)
  const loraB = api.tensorFromArray(new Float32Array(outFeatures * rank), [outFeatures, rank], { device, requiresGrad: true });

  // Forward pass
  // base: x @ W^T (W is [out, in], so W^T is [in, out])
  const baseOut = api.matmul(x, baseWeight.transpose({ dim0: 0, dim1: 1 }));
  const baseOutSum = await baseOut.sum().item();

  // LoRA: x @ A^T @ B^T * scale (A is [rank, in], B is [out, rank])
  const xA = api.matmul(x, loraA.transpose({ dim0: 0, dim1: 1 })); // [batch, seq, rank]
  const loraOut = api.matmul(xA, loraB.transpose({ dim0: 0, dim1: 1 })); // [batch, seq, out]
  const scaledLoraOut = api.mul(loraOut, api.tensorFromArray([scale], [], { device }));
  const loraOutSum = await scaledLoraOut.sum().item();

  // Combined output
  const out = api.add(baseOut, scaledLoraOut);
  const outSum = await out.sum().item();

  // Loss = sum of output
  const loss = out.sum();
  const lossVal = await loss.item();

  // Backward
  await loss.backward();

  // Get gradients
  const loraAGradSum = loraA.grad ? await loraA.grad.sum().item() : 0;
  const loraBGradSum = loraB.grad ? await loraB.grad.sum().item() : 0;

  // Get first element of B grad for detailed comparison
  let loraBGrad00 = 0;
  if (loraB.grad) {
    const bGradData = await api.cpu(loraB.grad);
    loraBGrad00 = bGradData[0];
  }

  return {
    base_out_sum: baseOutSum,
    lora_out_sum: loraOutSum,
    out_sum: outSum,
    loss: lossVal,
    lora_A_grad_sum: loraAGradSum,
    lora_B_grad_sum: loraBGradSum,
    lora_B_grad_00: loraBGrad00,
  };
}

function compare(name: string, pytorch: number, torchlette: number, tol: number = 1e-4): boolean {
  const diff = Math.abs(pytorch - torchlette);
  const relDiff = diff / (Math.abs(pytorch) + 1e-8);
  const pass = relDiff < tol || diff < 1e-6;
  const status = pass ? '✓' : '✗';
  console.log(`  ${name.padEnd(20)}: PyTorch=${pytorch.toFixed(6).padStart(12)}, Torchlette=${torchlette.toFixed(6).padStart(12)}, diff=${diff.toExponential(2)} ${status}`);
  return pass;
}

async function main(): Promise<void> {
  console.log('Minimal LoRA Test - PyTorch vs Torchlette');
  console.log('=========================================');
  console.log('');

  await initWebGPU();

  const configs = [
    { device: 'cpu' as const, memoryPlanning: false, name: 'CPU' },
    { device: 'webgpu' as const, memoryPlanning: false, name: 'WebGPU (no MP)' },
    { device: 'webgpu' as const, memoryPlanning: true, name: 'WebGPU (with MP)' },
  ];

  const results: { name: string; passed: boolean }[] = [];

  for (const config of configs) {
    console.log(`\n--- ${config.name} ---`);

    try {
      const result = await runTest(config.device, config.memoryPlanning);

      let allPass = true;
      allPass = compare('base_out_sum', PYTORCH_REFERENCE.base_out_sum, result.base_out_sum) && allPass;
      allPass = compare('lora_out_sum', PYTORCH_REFERENCE.lora_out_sum, result.lora_out_sum) && allPass;
      allPass = compare('out_sum', PYTORCH_REFERENCE.out_sum, result.out_sum) && allPass;
      allPass = compare('loss', PYTORCH_REFERENCE.loss, result.loss) && allPass;
      allPass = compare('lora_A_grad_sum', PYTORCH_REFERENCE.lora_A_grad_sum, result.lora_A_grad_sum) && allPass;
      allPass = compare('lora_B_grad_sum', PYTORCH_REFERENCE.lora_B_grad_sum, result.lora_B_grad_sum) && allPass;
      allPass = compare('lora_B_grad[0,0]', PYTORCH_REFERENCE.lora_B_grad_00, result.lora_B_grad_00) && allPass;

      results.push({ name: config.name, passed: allPass });
      console.log(`\n  Result: ${allPass ? 'PASSED ✓' : 'FAILED ✗'}`);
    } catch (e) {
      console.log(`  ERROR: ${e}`);
      results.push({ name: config.name, passed: false });
    }
  }

  console.log('\n=========================================');
  console.log('SUMMARY');
  console.log('=========================================');
  for (const r of results) {
    console.log(`  ${r.name}: ${r.passed ? 'PASSED ✓' : 'FAILED ✗'}`);
  }

  const allPassed = results.every(r => r.passed);
  console.log(`\nOverall: ${allPassed ? 'ALL TESTS PASSED ✓' : 'SOME TESTS FAILED ✗'}`);
  process.exit(allPassed ? 0 : 1);
}

main().catch(console.error);
