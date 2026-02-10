#!/usr/bin/env npx tsx
/**
 * Debug batched matmul discrepancy between CPU and WebGPU
 */

import { cpuBackend } from '../../src/backend/cpu';
import { webgpuBackend, initWebGPU, tensorFromArrayWithDtype } from '../../src/backend/webgpu';
import type { WebGPUTensor } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  await initWebGPU();

  // Test case: x [1, 4, 8] @ W^T [8, 8] = [1, 4, 8]
  // Using the exact values from minimal LoRA test

  const xData = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ];

  const wData = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,
    0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1,
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,
    0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0,
    0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3,
  ];

  // Create CPU tensors using backend's own method
  const xCpu = cpuBackend.ops.tensorFromArray(xData, [1, 4, 8]);
  const wCpu = cpuBackend.ops.tensorFromArray(wData, [8, 8]);
  const wTCpu = cpuBackend.ops.transpose(wCpu, { dim0: 0, dim1: 1 });

  // Create WebGPU tensors
  const xGpu = tensorFromArrayWithDtype(Float32Array.from(xData), [1, 4, 8], 'f32');
  const wGpu = tensorFromArrayWithDtype(Float32Array.from(wData), [8, 8], 'f32');
  const wTGpu = webgpuBackend.ops.transpose(wGpu, { dim0: 0, dim1: 1 });

  console.log('=== Test: x [1, 4, 8] @ W^T [8, 8] ===\n');

  // First check transposed weights
  console.log('W shape:', wCpu.shape, '(CPU)', wGpu.shape, '(WebGPU)');
  console.log('W^T shape:', wTCpu.shape, '(CPU)', (wTGpu as WebGPUTensor).shape, '(WebGPU)');

  const wtCpuData = await cpuBackend.ops.read(wTCpu);
  const wtGpuData = await webgpuBackend.ops.read(wTGpu as WebGPUTensor);

  console.log('\nW^T values comparison (row by row):');
  let allMatch = true;
  for (let i = 0; i < 8; i++) {
    const cpuRow = Array.from(wtCpuData.slice(i * 8, (i + 1) * 8));
    const gpuRow = Array.from(wtGpuData.slice(i * 8, (i + 1) * 8));
    const match = cpuRow.every((v, j) => Math.abs(v - gpuRow[j]) < 1e-6);
    if (!match) allMatch = false;
    console.log(`  Row ${i}: ${match ? '✓' : '✗'}`);
    if (!match) {
      console.log(`    CPU: [${cpuRow.map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`    GPU: [${gpuRow.map(v => v.toFixed(4)).join(', ')}]`);
    }
  }
  console.log(`  All rows match: ${allMatch ? '✓' : '✗'}`);

  // Test 1: Simple 2D matmul (no batching)
  console.log('\n=== Test 1: 2D matmul [4, 8] @ [8, 8] (no batch dim) ===');
  const x2dCpu = cpuBackend.ops.tensorFromArray(xData, [4, 8]);
  const x2dGpu = tensorFromArrayWithDtype(Float32Array.from(xData), [4, 8], 'f32');

  const result2dCpu = cpuBackend.ops.matmul(x2dCpu, wTCpu);
  const result2dGpu = await webgpuBackend.ops.matmul(x2dGpu, wTGpu as WebGPUTensor);

  const data2dCpu = await cpuBackend.ops.read(result2dCpu);
  const data2dGpu = await webgpuBackend.ops.read(result2dGpu);

  const sum2dCpu = data2dCpu.reduce((a, b) => a + b, 0);
  const sum2dGpu = data2dGpu.reduce((a, b) => a + b, 0);
  console.log(`  CPU sum: ${sum2dCpu.toFixed(6)}`);
  console.log(`  GPU sum: ${sum2dGpu.toFixed(6)}`);
  console.log(`  Match: ${Math.abs(sum2dCpu - sum2dGpu) < 1e-4 ? '✓' : '✗'}`);

  console.log(`  CPU first 8 values: [${Array.from(data2dCpu.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`  GPU first 8 values: [${Array.from(data2dGpu.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

  // Test 2: 3D matmul (with batching)
  console.log('\n=== Test 2: 3D matmul [1, 4, 8] @ [8, 8] (with batch dim) ===');
  const result3dCpu = cpuBackend.ops.matmul(xCpu, wTCpu);
  const result3dGpu = await webgpuBackend.ops.matmul(xGpu, wTGpu as WebGPUTensor);

  const data3dCpu = await cpuBackend.ops.read(result3dCpu);
  const data3dGpu = await webgpuBackend.ops.read(result3dGpu);

  const sum3dCpu = data3dCpu.reduce((a, b) => a + b, 0);
  const sum3dGpu = data3dGpu.reduce((a, b) => a + b, 0);
  console.log(`  CPU sum: ${sum3dCpu.toFixed(6)}`);
  console.log(`  GPU sum: ${sum3dGpu.toFixed(6)}`);
  console.log(`  Match: ${Math.abs(sum3dCpu - sum3dGpu) < 1e-4 ? '✓' : '✗'}`);

  console.log(`  CPU first 8 values: [${Array.from(data3dCpu.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`  GPU first 8 values: [${Array.from(data3dGpu.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

  // Test 3: Element-wise comparison
  console.log('\n=== Element-wise comparison ===');
  let maxDiff = 0;
  let maxDiffIdx = 0;
  for (let i = 0; i < data3dCpu.length; i++) {
    const diff = Math.abs(data3dCpu[i] - data3dGpu[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
      maxDiffIdx = i;
    }
  }
  console.log(`  Max difference: ${maxDiff.toExponential(4)} at index ${maxDiffIdx}`);
  console.log(`  CPU[${maxDiffIdx}] = ${data3dCpu[maxDiffIdx].toFixed(6)}`);
  console.log(`  GPU[${maxDiffIdx}] = ${data3dGpu[maxDiffIdx].toFixed(6)}`);

  // Test 4: Check if W^T needs to be made contiguous
  console.log('\n=== Test 4: Using contiguous W^T on GPU ===');
  const wTGpuContiguous = webgpuBackend.ops.contiguous(wTGpu as WebGPUTensor);
  const result3dGpuContiguous = await webgpuBackend.ops.matmul(xGpu, wTGpuContiguous);
  const data3dGpuContiguous = await webgpuBackend.ops.read(result3dGpuContiguous);
  const sum3dGpuContiguous = data3dGpuContiguous.reduce((a, b) => a + b, 0);
  console.log(`  GPU sum (contiguous W^T): ${sum3dGpuContiguous.toFixed(6)}`);
  console.log(`  Match CPU: ${Math.abs(sum3dCpu - sum3dGpuContiguous) < 1e-4 ? '✓' : '✗'}`);

  // Test 5: Check wTGpu internal properties
  console.log('\n=== W^T tensor properties (WebGPU) ===');
  const wTGpuTensor = wTGpu as WebGPUTensor;
  console.log(`  Shape: [${wTGpuTensor.shape}]`);
  console.log(`  Strides: [${wTGpuTensor.strides}]`);
  console.log(`  Offset: ${wTGpuTensor.offset}`);
  console.log(`  isContiguous: ${wTGpuTensor.isContiguous}`);

  const wTGpuContiguousTensor = wTGpuContiguous as WebGPUTensor;
  console.log('\nContiguous W^T:');
  console.log(`  Shape: [${wTGpuContiguousTensor.shape}]`);
  console.log(`  Strides: [${wTGpuContiguousTensor.strides}]`);
  console.log(`  Offset: ${wTGpuContiguousTensor.offset}`);
  console.log(`  isContiguous: ${wTGpuContiguousTensor.isContiguous}`);

  // Test 6: CPU without transpose - just to verify raw matmul
  console.log('\n=== Test 6: Verify matmul without transpose ===');
  // Create W already in transposed form
  const wTData = Array.from(wtCpuData);
  const wTDirect_Cpu = cpuBackend.ops.tensorFromArray(wTData, [8, 8]);
  const wTDirect_Gpu = tensorFromArrayWithDtype(Float32Array.from(wTData), [8, 8], 'f32');

  const resultDirect_Cpu = cpuBackend.ops.matmul(xCpu, wTDirect_Cpu);
  const resultDirect_Gpu = await webgpuBackend.ops.matmul(xGpu, wTDirect_Gpu);

  const dataDirect_Cpu = await cpuBackend.ops.read(resultDirect_Cpu);
  const dataDirect_Gpu = await webgpuBackend.ops.read(resultDirect_Gpu);

  const sumDirect_Cpu = dataDirect_Cpu.reduce((a, b) => a + b, 0);
  const sumDirect_Gpu = dataDirect_Gpu.reduce((a, b) => a + b, 0);
  console.log(`  CPU sum (direct, no transpose): ${sumDirect_Cpu.toFixed(6)}`);
  console.log(`  GPU sum (direct, no transpose): ${sumDirect_Gpu.toFixed(6)}`);
  console.log(`  Match: ${Math.abs(sumDirect_Cpu - sumDirect_Gpu) < 1e-4 ? '✓' : '✗'}`);
}

main().catch(console.error);
