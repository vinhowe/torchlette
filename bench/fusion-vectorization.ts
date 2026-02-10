/**
 * Fusion Vectorization Benchmark
 *
 * Compares performance of vectorized (vec4/vec2) vs scalar fused kernels.
 */

import { performance } from "node:perf_hooks";
import {
  initWebGPU,
  getWebGPUInitError,
  getWebGPUDevice,
  syncWebGPU,
} from "../src/backend/webgpu";
import {
  generateFusedKernel,
  type FusedKernelRecipe,
} from "../src/backend/webgpu/fusion-codegen";

const warmupIters = Number.parseInt(process.env.BENCH_WARMUP ?? "3", 10);
const runIters = Number.parseInt(process.env.BENCH_ITERS ?? "7", 10);

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

interface BenchCase {
  name: string;
  recipe: FusedKernelRecipe;
}

// Create test cases with various sizes
function createCases(): BenchCase[] {
  const sizes = [
    [1024],           // 1K elements
    [1024, 1024],     // 1M elements
    [256, 256, 16],   // 1M elements batched
    [4096, 256],      // 1M tall-skinny
  ];

  const cases: BenchCase[] = [];

  for (const shape of sizes) {
    const totalElements = shape.reduce((a, b) => a * b, 1);
    const sizeName = `${(totalElements / 1024).toFixed(0)}K`;

    // Simple unary (relu)
    cases.push({
      name: `relu ${sizeName} [${shape.join("x")}]`,
      recipe: {
        id: `relu_${sizeName}`,
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape, dtype: "f32", isOutput: true },
        ],
        inputs: [{ id: 100, index: 0, shape, dtype: "f32" }],
        outputShape: shape,
        outputDtype: "f32",
      },
    });

    // Binary + unary chain (add -> relu)
    cases.push({
      name: `add_relu ${sizeName} [${shape.join("x")}]`,
      recipe: {
        id: `add_relu_${sizeName}`,
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape, dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape, dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape, dtype: "f32" },
          { id: 101, index: 1, shape, dtype: "f32" },
        ],
        outputShape: shape,
        outputDtype: "f32",
      },
    });

    // Longer chain (mul -> add -> gelu)
    cases.push({
      name: `mul_add_gelu ${sizeName} [${shape.join("x")}]`,
      recipe: {
        id: `mul_add_gelu_${sizeName}`,
        nodes: [
          { id: 1, op: "mul", inputs: [-1, -2], shape, dtype: "f32" },
          { id: 2, op: "add", inputs: [1, -3], shape, dtype: "f32" },
          { id: 3, op: "gelu", inputs: [2], shape, dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape, dtype: "f32" },
          { id: 101, index: 1, shape, dtype: "f32" },
          { id: 102, index: 2, shape, dtype: "f32" },
        ],
        outputShape: shape,
        outputDtype: "f32",
      },
    });
  }

  return cases;
}

async function runBenchmark(
  device: GPUDevice,
  recipe: FusedKernelRecipe,
  vectorize: boolean,
): Promise<{ msMedian: number; workItems: number; vectorWidth: number }> {
  const kernel = generateFusedKernel(recipe, { vectorize });

  // Create shader module and pipeline
  const module = device.createShaderModule({ code: kernel.source });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  // Create buffers
  const totalElements = recipe.outputShape.reduce((a, b) => a * b, 1);
  const buffers: GPUBuffer[] = [];

  // Input buffers
  for (const input of recipe.inputs) {
    const inputElements = input.shape.reduce((a, b) => a * b, 1);
    const buffer = device.createBuffer({
      size: inputElements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    buffers.push(buffer);
  }

  // Output buffer
  const outputBuffer = device.createBuffer({
    size: totalElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  buffers.push(outputBuffer);

  // Params buffer
  const paramsBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([totalElements]));
  buffers.push(paramsBuffer);

  // Create bind group
  const entries = buffers.map((buffer, i) => ({
    binding: i,
    resource: { buffer },
  }));

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(kernel.workItems / kernel.workgroupSize);

  // Warmup
  for (let i = 0; i < warmupIters; i++) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  await syncWebGPU();

  // Timed runs
  const durations: number[] = [];
  for (let i = 0; i < runIters; i++) {
    const start = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await syncWebGPU();
    const end = performance.now();
    durations.push(end - start);
  }

  // Cleanup
  for (const buffer of buffers) {
    buffer.destroy();
  }

  return {
    msMedian: median(durations),
    workItems: kernel.workItems,
    vectorWidth: kernel.vectorWidth,
  };
}

async function main() {
  console.log("Fusion Vectorization Benchmark");
  console.log("==============================\n");
  console.log(`Warmup: ${warmupIters}, Iterations: ${runIters}\n`);

  const ready = await initWebGPU();
  if (!ready) {
    const error = getWebGPUInitError();
    console.error(`WebGPU init failed: ${error}`);
    process.exit(1);
  }

  // Get device from the initialized context
  const ctx = getWebGPUDevice();
  if (!ctx) {
    console.error("No GPU device available");
    process.exit(1);
  }
  const { device } = ctx;

  const cases = createCases();

  console.log("Case                                    | Scalar (ms) | Vectorized (ms) | Speedup | Vec Width");
  console.log("-".repeat(100));

  for (const benchCase of cases) {
    const scalarResult = await runBenchmark(device as any, benchCase.recipe, false);
    const vectorResult = await runBenchmark(device as any, benchCase.recipe, true);

    const speedup = scalarResult.msMedian / vectorResult.msMedian;
    const speedupStr = speedup >= 1 ? `${speedup.toFixed(2)}x` : `${(1/speedup).toFixed(2)}x slower`;

    console.log(
      `${benchCase.name.padEnd(39)} | ${scalarResult.msMedian.toFixed(3).padStart(11)} | ${vectorResult.msMedian.toFixed(3).padStart(15)} | ${speedupStr.padStart(7)} | vec${vectorResult.vectorWidth}`,
    );
  }

  console.log("\nDone.");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
