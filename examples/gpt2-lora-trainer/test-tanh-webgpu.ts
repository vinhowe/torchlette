#!/usr/bin/env npx tsx
/**
 * Test WebGPU tanh with large inputs to find NaN source.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  console.log('Testing WebGPU tanh...');
  await initWebGPU();

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  console.log(`\n${'='.repeat(60)}`);
  console.log('WebGPU tanh test');
  console.log('='.repeat(60));

  // Test tanh with various values
  const testValues = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    -10, -20, -50, -100
  ];

  console.log('\nWebGPU tanh forward:');
  for (const val of testValues) {
    const x = api.tensorFromArray([val], [1], { device: 'webgpu' });
    const y = x.tanh();
    const yVal = await y.item();
    console.log(`tanh(${val.toString().padStart(4)}) = ${yVal.toFixed(10)}`);
  }

  // Now test tanh in the GELU backward context
  console.log('\nManual GELU gradient components for x=11:');
  {
    const x = 11;
    const sqrt2OverPi = 0.7978845608;
    const coeff1 = 0.044715;
    const coeff2 = 0.134145;

    const x2 = x * x;
    const x3 = x * x * x;
    const inner = sqrt2OverPi * (x + coeff1 * x3);

    console.log(`  x = ${x}`);
    console.log(`  x^2 = ${x2}`);
    console.log(`  x^3 = ${x3}`);
    console.log(`  inner = ${inner.toFixed(4)}`);

    // Use WebGPU to compute tanh(inner)
    const innerTensor = api.tensorFromArray([inner], [1], { device: 'webgpu' });
    const tanhInner = await innerTensor.tanh().item();
    console.log(`  tanh(inner) = ${tanhInner.toFixed(10)}`);

    // Compute sech^2
    const sech2 = 1 - tanhInner * tanhInner;
    console.log(`  sech^2 = ${sech2.toFixed(10)}`);

    // Check if sech^2 is exactly 0 or some small denormalized number
    const tanhTensor = api.tensorFromArray([tanhInner], [1], { device: 'webgpu' });
    const tanh2 = await tanhTensor.mul(tanhTensor).item();
    console.log(`  tanh^2 = ${tanh2.toFixed(20)}`);

    const oneTensor = api.tensorFromArray([1], [], { device: 'webgpu' });
    const sech2WebGPU = await api.sub(oneTensor, tanhTensor.mul(tanhTensor)).item();
    console.log(`  sech^2 (WebGPU) = ${sech2WebGPU.toFixed(20)}`);

    // Compute pdf term
    const pdfTerm = 1 + coeff2 * x2;
    console.log(`  pdfTerm = ${pdfTerm.toFixed(10)}`);

    // pdf = sqrt2OverPi * pdfTerm * sech2
    const pdf = sqrt2OverPi * pdfTerm * sech2;
    console.log(`  pdf = ${pdf.toFixed(10)}`);

    // cdf = 0.5 * (1 + tanh(inner))
    const cdf = 0.5 * (1 + tanhInner);
    console.log(`  cdf = ${cdf.toFixed(10)}`);

    // xPdfHalf = x * pdf * 0.5
    const xPdfHalf = x * pdf * 0.5;
    console.log(`  xPdfHalf = ${xPdfHalf.toFixed(10)}`);

    // geluGrad = cdf + xPdfHalf
    const geluGrad = cdf + xPdfHalf;
    console.log(`  geluGrad = ${geluGrad.toFixed(10)}`);
  }

  // Now do the same computation entirely in WebGPU tensors
  console.log('\nFull WebGPU tensor computation for GELU grad at x=11:');
  {
    const x = api.tensorFromArray([11], [1], { device: 'webgpu' });
    const sqrt2OverPi = api.tensorFromArray([0.7978845608], [], { device: 'webgpu' });
    const half = api.tensorFromArray([0.5], [], { device: 'webgpu' });
    const one = api.tensorFromArray([1], [], { device: 'webgpu' });
    const coeff1 = api.tensorFromArray([0.044715], [], { device: 'webgpu' });
    const coeff2 = api.tensorFromArray([0.134145], [], { device: 'webgpu' });

    const x2 = x.mul(x);
    const x3 = api.mul(x2, x);

    console.log(`  x = ${await x.item()}`);
    console.log(`  x^3 = ${await x3.item()}`);

    // inner = sqrt2OverPi * (x + coeff1 * x3)
    const innerVal = api.mul(sqrt2OverPi, api.add(x, api.mul(coeff1, x3)));
    console.log(`  inner = ${await innerVal.item()}`);

    const tanhInner = innerVal.tanh();
    console.log(`  tanh(inner) = ${await tanhInner.item()}`);

    // sech^2 = 1 - tanh^2
    const tanh2 = tanhInner.mul(tanhInner);
    console.log(`  tanh^2 = ${await tanh2.item()}`);

    const sech2 = api.sub(one, tanh2);
    console.log(`  sech^2 = ${await sech2.item()}`);

    // Check if it's negative (would cause NaN in sqrt, but we don't sqrt here)
    const sech2Val = await sech2.item();
    if (sech2Val < 0) console.log('  WARNING: sech^2 is NEGATIVE!');
    if (Number.isNaN(sech2Val)) console.log('  WARNING: sech^2 is NaN!');

    // pdfTerm = 1 + coeff2 * x^2
    const pdfTerm = api.add(one, api.mul(coeff2, x2));
    console.log(`  pdfTerm = ${await pdfTerm.item()}`);

    // pdf = sqrt2OverPi * pdfTerm * sech2
    const pdf = api.mul(api.mul(sqrt2OverPi, pdfTerm), sech2);
    console.log(`  pdf = ${await pdf.item()}`);

    // cdf = 0.5 * (1 + tanh(inner))
    const cdf = api.mul(half, api.add(one, tanhInner));
    console.log(`  cdf = ${await cdf.item()}`);

    // xPdfHalf = x * pdf * 0.5
    const xPdfHalf = api.mul(api.mul(x, pdf), half);
    console.log(`  xPdfHalf = ${await xPdfHalf.item()}`);

    // geluGrad = cdf + xPdfHalf
    const geluGrad = api.add(cdf, xPdfHalf);
    console.log(`  geluGrad = ${await geluGrad.item()}`);
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
