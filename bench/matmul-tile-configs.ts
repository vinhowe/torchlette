/**
 * Benchmark comparing different tile configurations for matmul.
 * Tests the impact of tile sizes on performance.
 */

import {
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
} from "../src/backend/webgpu";
import {
  DEFAULT_CONFIG,
  dispatchTiledMatmul,
  getSubgroupSupport,
  type MatmulKernelConfig,
  validateConfig,
} from "../src/backend/webgpu/matmul";

function getDevice() {
  const ctx = getWebGPUDevice();
  if (!ctx) {
    throw new Error("WebGPU context not available");
  }
  return ctx;
}

const GPUBufferUsage = {
  STORAGE: 0x0080,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
};

function makeValues(size: number): Float32Array {
  const values = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    values[i] = ((i % 13) - 6) * 0.1;
  }
  return values;
}

async function benchmark(
  _name: string,
  runFn: () => void,
  warmup: number,
  iters: number,
): Promise<number> {
  // Warmup
  for (let i = 0; i < warmup; i++) {
    runFn();
  }
  await syncWebGPU();

  // Timed runs
  const times: number[] = [];
  for (let i = 0; i < iters; i++) {
    const start = performance.now();
    runFn();
    await syncWebGPU();
    const end = performance.now();
    times.push(end - start);
  }

  // Return median
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

function configToString(config: MatmulKernelConfig): string {
  return `${config.tileM}x${config.tileN}x${config.tileK}_t${config.threadTileM}x${config.threadTileN}`;
}

type BenchResult = {
  name: string;
  config: string;
  medianMs: number;
  gflops: number;
  valid: boolean;
};

async function benchmarkConfig(
  m: number,
  n: number,
  k: number,
  config: MatmulKernelConfig,
  warmup: number,
  iters: number,
): Promise<BenchResult> {
  const { device, queue } = getDevice();
  const flops = 2 * m * n * k;

  // Create buffers
  const aData = makeValues(m * k);
  const bData = makeValues(k * n);

  const aBuffer = device.createBuffer({
    size: m * k * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bBuffer = device.createBuffer({
    size: k * n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outBuffer = device.createBuffer({
    size: m * n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  queue.writeBuffer(aBuffer, 0, aData);
  queue.writeBuffer(bBuffer, 0, bData);

  try {
    const ms = await benchmark(
      configToString(config),
      () => {
        dispatchTiledMatmul({
          device,
          queue,
          a: aBuffer,
          b: bBuffer,
          out: outBuffer,
          m,
          n,
          k,
          config,
          dtype: "f32",
        });
      },
      warmup,
      iters,
    );

    return {
      name: configToString(config),
      config: configToString(config),
      medianMs: ms,
      gflops: flops / (ms * 1e6),
      valid: true,
    };
  } catch (_e) {
    return {
      name: configToString(config),
      config: configToString(config),
      medianMs: Infinity,
      gflops: 0,
      valid: false,
    };
  }
}

function printResults(title: string, results: BenchResult[]): void {
  console.log(`\n${"=".repeat(80)}`);
  console.log(title);
  console.log("=".repeat(80));
  console.log(
    "Config".padEnd(25) +
      "Time(ms)".padStart(12) +
      "GFLOPs/s".padStart(12) +
      "vs Default".padStart(12),
  );
  console.log("-".repeat(80));

  // Find default result for comparison
  const defaultResult = results.find(
    (r) => r.name === configToString(DEFAULT_CONFIG),
  );
  const defaultGflops = defaultResult?.gflops || 1;

  // Sort by GFLOPs (best first)
  const sorted = [...results]
    .filter((r) => r.valid)
    .sort((a, b) => b.gflops - a.gflops);

  for (const r of sorted) {
    const speedup = r.gflops / defaultGflops;
    const speedupStr =
      speedup >= 1
        ? `+${((speedup - 1) * 100).toFixed(1)}%`
        : `-${((1 - speedup) * 100).toFixed(1)}%`;

    console.log(
      r.config.padEnd(25) +
        r.medianMs.toFixed(3).padStart(12) +
        r.gflops.toFixed(2).padStart(12) +
        speedupStr.padStart(12),
    );
  }

  const failed = results.filter((r) => !r.valid);
  if (failed.length > 0) {
    console.log(`\nSkipped ${failed.length} invalid configurations`);
  }
}

async function main(): Promise<void> {
  console.log("Initializing WebGPU...");
  const ok = await initWebGPU();
  if (!ok) {
    console.error("Failed to initialize WebGPU:", getWebGPUInitError());
    process.exit(1);
  }

  const subgroupSupport = getSubgroupSupport();
  console.log(
    `\nSubgroup support: ${subgroupSupport?.supported ? "YES" : "NO"}`,
  );

  const warmup = parseInt(process.env.BENCH_WARMUP || "3", 10);
  const iters = parseInt(process.env.BENCH_ITERS || "5", 10);
  console.log(`Benchmark settings: warmup=${warmup}, iters=${iters}`);

  // Configurations to test
  const configs: MatmulKernelConfig[] = [
    // Default
    { ...DEFAULT_CONFIG },

    // Larger tiles
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 64 },
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 64, tileK: 32 },
    {
      ...DEFAULT_CONFIG,
      tileM: 128,
      tileN: 128,
      tileK: 16,
      threadTileM: 8,
      threadTileN: 8,
    },

    // Asymmetric tiles
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 32 },
    { ...DEFAULT_CONFIG, tileM: 32, tileN: 64 },
    {
      ...DEFAULT_CONFIG,
      tileM: 128,
      tileN: 32,
      threadTileM: 8,
      threadTileN: 4,
    },
    {
      ...DEFAULT_CONFIG,
      tileM: 32,
      tileN: 128,
      threadTileM: 4,
      threadTileN: 8,
    },

    // Different K tiles
    { ...DEFAULT_CONFIG, tileK: 8 },
    { ...DEFAULT_CONFIG, tileK: 32 },
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 64, tileK: 8 },

    // Larger thread tiles
    { ...DEFAULT_CONFIG, threadTileM: 8, threadTileN: 8, tileM: 64, tileN: 64 },
    { ...DEFAULT_CONFIG, threadTileM: 8, threadTileN: 4, tileM: 64, tileN: 32 },
  ];

  // Filter to valid configs
  const validConfigs = configs.filter((c) => {
    try {
      validateConfig(c);
      return true;
    } catch {
      return false;
    }
  });

  console.log(`\nTesting ${validConfigs.length} valid configurations...\n`);

  // Test on 1024x1024
  const size1024Results: BenchResult[] = [];
  console.log("Testing 1024x1024...");
  for (const config of validConfigs) {
    const result = await benchmarkConfig(
      1024,
      1024,
      1024,
      config,
      warmup,
      iters,
    );
    size1024Results.push(result);
    process.stdout.write(".");
  }
  printResults("1024x1024 - Tile Configuration Comparison", size1024Results);

  // Test on 2048x2048
  const size2048Results: BenchResult[] = [];
  console.log("\n\nTesting 2048x2048...");
  for (const config of validConfigs) {
    const result = await benchmarkConfig(
      2048,
      2048,
      2048,
      config,
      warmup,
      iters,
    );
    size2048Results.push(result);
    process.stdout.write(".");
  }
  printResults("2048x2048 - Tile Configuration Comparison", size2048Results);

  // Test on 512x512
  const size512Results: BenchResult[] = [];
  console.log("\n\nTesting 512x512...");
  for (const config of validConfigs) {
    const result = await benchmarkConfig(512, 512, 512, config, warmup, iters);
    size512Results.push(result);
    process.stdout.write(".");
  }
  printResults("512x512 - Tile Configuration Comparison", size512Results);

  // Summary
  console.log("\n" + "=".repeat(80));
  console.log("BEST CONFIGURATIONS BY SIZE");
  console.log("=".repeat(80));

  const best512 = size512Results
    .filter((r) => r.valid)
    .sort((a, b) => b.gflops - a.gflops)[0];
  const best1024 = size1024Results
    .filter((r) => r.valid)
    .sort((a, b) => b.gflops - a.gflops)[0];
  const best2048 = size2048Results
    .filter((r) => r.valid)
    .sort((a, b) => b.gflops - a.gflops)[0];

  console.log(
    `\n512x512:   ${best512?.config.padEnd(25)} ${best512?.gflops.toFixed(2)} GFLOPs/s`,
  );
  console.log(
    `1024x1024: ${best1024?.config.padEnd(25)} ${best1024?.gflops.toFixed(2)} GFLOPs/s`,
  );
  console.log(
    `2048x2048: ${best2048?.config.padEnd(25)} ${best2048?.gflops.toFixed(2)} GFLOPs/s`,
  );

  console.log("\n\nBenchmark complete!");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
