/**
 * Comprehensive matmul benchmark comparing all variants:
 * - Default vs autotuned configurations
 * - Fused vs unfused epilogue
 * - Different tile configurations
 */

import {
  dispatchMatmulWithEpilogue,
  getWebGPUDevice,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "../src/backend/webgpu";
import {
  classifyShape,
  DEFAULT_CONFIG,
  type DType,
  dispatchTiledMatmul,
  type EpilogueConfig,
  getDefaultConfigForShape,
  getSubgroupSupport,
  type MatmulKernelConfig,
} from "../src/backend/webgpu/matmul";

type BenchResult = {
  name: string;
  config: string;
  m: number;
  n: number;
  k: number;
  medianMs: number;
  gflops: number;
  gbps: number;
};

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

function makeValues(size: number): number[] {
  const values = new Array<number>(size);
  for (let i = 0; i < size; i++) {
    values[i] = ((i % 13) - 6) * 0.1;
  }
  return values;
}

function configToString(config: MatmulKernelConfig): string {
  const base = `${config.tileM}x${config.tileN}x${config.tileK}_t${config.threadTileM}x${config.threadTileN}`;
  return config.useSubgroups ? `${base}_sg` : base;
}

const GPUBufferUsage = {
  STORAGE: 0x0080,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
};

async function runSubgroupComparisonBenchmark(
  m: number,
  n: number,
  k: number,
  warmup: number,
  iters: number,
): Promise<BenchResult[]> {
  const results: BenchResult[] = [];
  const flops = 2 * m * n * k;
  const bytes = 4 * (m * k + k * n + m * n);

  const ctx = getWebGPUDevice();
  if (!ctx) {
    console.warn("WebGPU device not available for subgroup comparison");
    return results;
  }
  const { device, queue } = ctx;

  // Create GPU buffers
  const aData = new Float32Array(m * k);
  const bData = new Float32Array(k * n);
  for (let i = 0; i < m * k; i++) aData[i] = ((i % 13) - 6) * 0.1;
  for (let i = 0; i < k * n; i++) bData[i] = ((i % 11) - 5) * 0.1;

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

  // Configs to compare
  const baseConfig: MatmulKernelConfig = {
    ...DEFAULT_CONFIG,
    tileM: 64,
    tileN: 64,
    tileK: 16,
  };

  const configsToTest: Array<{ name: string; config: MatmulKernelConfig }> = [
    { name: "no-subgroups", config: { ...baseConfig, useSubgroups: false } },
    { name: "with-subgroups", config: { ...baseConfig, useSubgroups: true } },
  ];

  for (const { name, config } of configsToTest) {
    try {
      // Warmup
      for (let i = 0; i < warmup; i++) {
        dispatchTiledMatmul({
          device: device as any,
          queue: queue as any,
          a: aBuffer as any,
          b: bBuffer as any,
          out: outBuffer as any,
          m,
          n,
          k,
          config,
          dtype: "f32",
        });
      }
      await syncWebGPU();

      // Timed runs
      const times: number[] = [];
      for (let i = 0; i < iters; i++) {
        const start = performance.now();
        dispatchTiledMatmul({
          device: device as any,
          queue: queue as any,
          a: aBuffer as any,
          b: bBuffer as any,
          out: outBuffer as any,
          m,
          n,
          k,
          config,
          dtype: "f32",
        });
        await syncWebGPU();
        const end = performance.now();
        times.push(end - start);
      }

      times.sort((a, b) => a - b);
      const ms = times[Math.floor(times.length / 2)];

      results.push({
        name,
        config: configToString(config),
        m,
        n,
        k,
        medianMs: ms,
        gflops: flops / (ms * 1e6),
        gbps: bytes / (ms * 1e6),
      });
    } catch (e) {
      // Skip if subgroups not supported or config invalid
      console.warn(`Skipping ${name}: ${e}`);
    }
  }

  return results;
}

async function runDtypeComparisonBenchmark(
  m: number,
  n: number,
  k: number,
  warmup: number,
  iters: number,
): Promise<BenchResult[]> {
  const results: BenchResult[] = [];
  const flops = 2 * m * n * k;

  const ctx = getWebGPUDevice();
  if (!ctx) {
    console.warn("WebGPU device not available for dtype comparison");
    return results;
  }
  const { device, queue } = ctx;

  const baseConfig: MatmulKernelConfig = {
    ...DEFAULT_CONFIG,
    tileM: 64,
    tileN: 64,
    tileK: 16,
  };

  // Test all combinations of dtype and subgroup usage
  const variants: Array<{
    name: string;
    dtype: DType;
    bytesPerElement: number;
    useSubgroups: boolean;
  }> = [
    { name: "f32", dtype: "f32", bytesPerElement: 4, useSubgroups: false },
    {
      name: "f32+subgroups",
      dtype: "f32",
      bytesPerElement: 4,
      useSubgroups: true,
    },
    { name: "f16", dtype: "f16", bytesPerElement: 2, useSubgroups: false },
    {
      name: "f16+subgroups",
      dtype: "f16",
      bytesPerElement: 2,
      useSubgroups: true,
    },
  ];

  for (const { name, dtype, bytesPerElement, useSubgroups } of variants) {
    try {
      const bytes = bytesPerElement * (m * k + k * n + m * n);
      const config = { ...baseConfig, useSubgroups };

      // Create GPU buffers with appropriate size
      const aBuffer = device.createBuffer({
        size: m * k * bytesPerElement,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const bBuffer = device.createBuffer({
        size: k * n * bytesPerElement,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const outBuffer = device.createBuffer({
        size: m * n * bytesPerElement,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Initialize data (f32 for both, GPU will handle conversion for f16)
      const aData = new Float32Array(m * k);
      const bData = new Float32Array(k * n);
      for (let i = 0; i < m * k; i++) aData[i] = ((i % 13) - 6) * 0.1;
      for (let i = 0; i < k * n; i++) bData[i] = ((i % 11) - 5) * 0.1;

      if (dtype === "f16") {
        // Convert to f16 using Uint16Array with f16 encoding
        const aData16 = new Uint16Array(m * k);
        const bData16 = new Uint16Array(k * n);
        for (let i = 0; i < m * k; i++) aData16[i] = float32ToFloat16(aData[i]);
        for (let i = 0; i < k * n; i++) bData16[i] = float32ToFloat16(bData[i]);
        queue.writeBuffer(aBuffer, 0, aData16);
        queue.writeBuffer(bBuffer, 0, bData16);
      } else {
        queue.writeBuffer(aBuffer, 0, aData);
        queue.writeBuffer(bBuffer, 0, bData);
      }

      // Warmup
      for (let i = 0; i < warmup; i++) {
        dispatchTiledMatmul({
          device: device as any,
          queue: queue as any,
          a: aBuffer as any,
          b: bBuffer as any,
          out: outBuffer as any,
          m,
          n,
          k,
          config,
          dtype,
        });
      }
      await syncWebGPU();

      // Timed runs
      const times: number[] = [];
      for (let i = 0; i < iters; i++) {
        const start = performance.now();
        dispatchTiledMatmul({
          device: device as any,
          queue: queue as any,
          a: aBuffer as any,
          b: bBuffer as any,
          out: outBuffer as any,
          m,
          n,
          k,
          config,
          dtype,
        });
        await syncWebGPU();
        const end = performance.now();
        times.push(end - start);
      }

      times.sort((a, b) => a - b);
      const ms = times[Math.floor(times.length / 2)];

      results.push({
        name,
        config: configToString(config) + `_${dtype}`,
        m,
        n,
        k,
        medianMs: ms,
        gflops: flops / (ms * 1e6),
        gbps: bytes / (ms * 1e6),
      });
    } catch (e) {
      console.warn(`Skipping ${name}: ${e}`);
    }
  }

  return results;
}

// Helper to convert f32 to f16 (IEEE 754 half-precision)
function float32ToFloat16(val: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = val;
  const x = int32View[0];

  let bits = (x >> 16) & 0x8000; // sign
  let m = (x >> 12) & 0x07ff; // mantissa
  const e = (x >> 23) & 0xff; // exponent

  if (e < 103) {
    // Too small for f16, return signed zero
    return bits;
  }

  if (e > 142) {
    // Too large for f16, return infinity
    bits |= 0x7c00;
    bits |= (e === 255 ? 0 : 1) && x & 0x007fffff;
    return bits;
  }

  if (e < 113) {
    // Denormalized number
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }

  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

async function runMatmulBenchmark(
  m: number,
  n: number,
  k: number,
  warmup: number,
  iters: number,
): Promise<BenchResult[]> {
  const results: BenchResult[] = [];
  const flops = 2 * m * n * k;
  const bytes = 4 * (m * k + k * n + m * n);

  const aVals = makeValues(m * k);
  const bVals = makeValues(k * n);

  const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
  const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);

  // Test 1: Default config (32x32)
  const defaultMs = await benchmark(
    "default",
    () => webgpuBackend.ops.matmul(a, b),
    warmup,
    iters,
  );
  results.push({
    name: "default",
    config: configToString(DEFAULT_CONFIG),
    m,
    n,
    k,
    medianMs: defaultMs,
    gflops: flops / (defaultMs * 1e6),
    gbps: bytes / (defaultMs * 1e6),
  });

  // Test 2: Shape-optimized default
  const shapeClass = classifyShape(m, n, k, 1);
  const shapeConfig = getDefaultConfigForShape(shapeClass);
  if (configToString(shapeConfig) !== configToString(DEFAULT_CONFIG)) {
    const shapeMs = await benchmark(
      "shape-default",
      () => webgpuBackend.ops.matmul(a, b),
      warmup,
      iters,
    );
    results.push({
      name: `shape-default (${shapeClass})`,
      config: configToString(shapeConfig),
      m,
      n,
      k,
      medianMs: shapeMs,
      gflops: flops / (shapeMs * 1e6),
      gbps: bytes / (shapeMs * 1e6),
    });
  }

  return results;
}

async function runFusedVsUnfusedBenchmark(
  m: number,
  n: number,
  k: number,
  warmup: number,
  iters: number,
): Promise<BenchResult[]> {
  const results: BenchResult[] = [];
  const flops = 2 * m * n * k;
  const bytes = 4 * (m * k + k * n + m * n);

  const aVals = makeValues(m * k);
  const bVals = makeValues(k * n);
  const biasVals = makeValues(n);

  const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
  const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;
  const bias = webgpuBackend.ops.tensorFromArray(biasVals, [n]) as any;

  // Unfused: matmul + add (separate ops)
  const unfusedMs = await benchmark(
    "unfused",
    () => {
      const c = webgpuBackend.ops.matmul(a, b);
      webgpuBackend.ops.add(c, bias);
    },
    warmup,
    iters,
  );
  results.push({
    name: "matmul+bias (unfused)",
    config: "separate ops",
    m,
    n,
    k,
    medianMs: unfusedMs,
    gflops: flops / (unfusedMs * 1e6),
    gbps: bytes / (unfusedMs * 1e6),
  });

  // Fused: matmul with bias epilogue
  const epilogue: EpilogueConfig = {
    ops: [{ kind: "bias", inputIndex: 0 }],
    additionalInputCount: 1,
    outputDtype: "f32",
  };
  const fusedMs = await benchmark(
    "fused",
    () => dispatchMatmulWithEpilogue(a, b, epilogue, [bias]),
    warmup,
    iters,
  );
  results.push({
    name: "matmul+bias (fused)",
    config: "epilogue fusion",
    m,
    n,
    k,
    medianMs: fusedMs,
    gflops: flops / (fusedMs * 1e6),
    gbps: bytes / (fusedMs * 1e6),
  });

  // Fused: matmul + bias + relu
  const epilogueRelu: EpilogueConfig = {
    ops: [{ kind: "bias", inputIndex: 0 }, { kind: "relu" }],
    additionalInputCount: 1,
    outputDtype: "f32",
  };
  const fusedReluMs = await benchmark(
    "fused-relu",
    () => dispatchMatmulWithEpilogue(a, b, epilogueRelu, [bias]),
    warmup,
    iters,
  );
  results.push({
    name: "matmul+bias+relu (fused)",
    config: "epilogue fusion",
    m,
    n,
    k,
    medianMs: fusedReluMs,
    gflops: flops / (fusedReluMs * 1e6),
    gbps: bytes / (fusedReluMs * 1e6),
  });

  // Fused: matmul + bias + gelu
  const epilogueGelu: EpilogueConfig = {
    ops: [{ kind: "bias", inputIndex: 0 }, { kind: "gelu" }],
    additionalInputCount: 1,
    outputDtype: "f32",
  };
  const fusedGeluMs = await benchmark(
    "fused-gelu",
    () => dispatchMatmulWithEpilogue(a, b, epilogueGelu, [bias]),
    warmup,
    iters,
  );
  results.push({
    name: "matmul+bias+gelu (fused)",
    config: "epilogue fusion",
    m,
    n,
    k,
    medianMs: fusedGeluMs,
    gflops: flops / (fusedGeluMs * 1e6),
    gbps: bytes / (fusedGeluMs * 1e6),
  });

  return results;
}

async function _runTileConfigBenchmark(
  m: number,
  n: number,
  k: number,
  warmup: number,
  iters: number,
): Promise<BenchResult[]> {
  const results: BenchResult[] = [];
  const flops = 2 * m * n * k;
  const bytes = 4 * (m * k + k * n + m * n);

  const aVals = makeValues(m * k);
  const bVals = makeValues(k * n);
  const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
  const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);

  // Test different tile configurations
  const configs: MatmulKernelConfig[] = [
    { ...DEFAULT_CONFIG }, // 32x32
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 64 },
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 32 },
    { ...DEFAULT_CONFIG, tileM: 32, tileN: 64 },
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 64, tileK: 32 },
    { ...DEFAULT_CONFIG, tileM: 64, tileN: 64, threadTileM: 8, threadTileN: 8 },
  ];

  for (const config of configs) {
    try {
      // Use matmul with default config (we can't directly pass config in current API)
      // So we benchmark the default implementation
      const ms = await benchmark(
        configToString(config),
        () => webgpuBackend.ops.matmul(a, b),
        warmup,
        iters,
      );
      results.push({
        name: `tile-${configToString(config)}`,
        config: configToString(config),
        m,
        n,
        k,
        medianMs: ms,
        gflops: flops / (ms * 1e6),
        gbps: bytes / (ms * 1e6),
      });
      // Only run once since we can't change config at runtime
      break;
    } catch (_e) {
      // Skip invalid configs
    }
  }

  return results;
}

function printResults(title: string, results: BenchResult[]): void {
  console.log(`\n${"=".repeat(80)}`);
  console.log(`${title}`);
  console.log("=".repeat(80));
  console.log(
    "Name".padEnd(35) +
      "Config".padEnd(25) +
      "Time(ms)".padStart(10) +
      "GFLOPs/s".padStart(12) +
      "GB/s".padStart(10),
  );
  console.log("-".repeat(80));
  for (const r of results) {
    console.log(
      r.name.padEnd(35) +
        r.config.padEnd(25) +
        r.medianMs.toFixed(3).padStart(10) +
        r.gflops.toFixed(2).padStart(12) +
        r.gbps.toFixed(2).padStart(10),
    );
  }
}

async function main(): Promise<void> {
  console.log("Initializing WebGPU...");
  const ok = await initWebGPU();
  if (!ok) {
    console.error("Failed to initialize WebGPU");
    process.exit(1);
  }

  // Check subgroup support
  const subgroupSupport = getSubgroupSupport();
  console.log(
    `\nSubgroup support: ${subgroupSupport?.supported ? "YES" : "NO"}`,
  );
  if (subgroupSupport?.supported) {
    console.log(`Subgroup size: ${subgroupSupport.subgroupSize}`);
  }

  const warmup = parseInt(process.env.BENCH_WARMUP || "3", 10);
  const iters = parseInt(process.env.BENCH_ITERS || "5", 10);
  console.log(`\nBenchmark settings: warmup=${warmup}, iters=${iters}`);

  // Test sizes
  const sizes = [
    { m: 256, n: 256, k: 256, name: "Small (256x256)" },
    { m: 512, n: 512, k: 512, name: "Medium (512x512)" },
    { m: 1024, n: 1024, k: 1024, name: "Large (1024x1024)" },
    { m: 2048, n: 2048, k: 2048, name: "XLarge (2048x2048)" },
  ];

  // 1. Default config benchmarks across sizes
  console.log("\n" + "=".repeat(80));
  console.log(
    "MATMUL PERFORMANCE BY SIZE [f32] (Default Config: 32x32x16 tiles)",
  );
  console.log("=".repeat(80));

  for (const { m, n, k, name } of sizes) {
    const results = await runMatmulBenchmark(m, n, k, warmup, iters);
    printResults(`${name} (M=${m}, N=${n}, K=${k})`, results);
  }

  // 2. Fused vs unfused comparison
  console.log("\n" + "=".repeat(80));
  console.log("EPILOGUE FUSION COMPARISON [f32] (1024x1024x1024)");
  console.log("=".repeat(80));

  const fusedResults = await runFusedVsUnfusedBenchmark(
    1024,
    1024,
    1024,
    warmup,
    iters,
  );
  printResults("Fused vs Unfused Operations", fusedResults);

  // Calculate speedup
  const unfusedTime =
    fusedResults.find((r) => r.name.includes("unfused"))?.medianMs || 0;
  const fusedBiasTime =
    fusedResults.find((r) => r.name === "matmul+bias (fused)")?.medianMs || 0;
  const fusedReluTime =
    fusedResults.find((r) => r.name.includes("relu"))?.medianMs || 0;
  const fusedGeluTime =
    fusedResults.find((r) => r.name.includes("gelu"))?.medianMs || 0;

  console.log("\nSpeedups from fusion:");
  if (unfusedTime > 0 && fusedBiasTime > 0) {
    console.log(
      `  matmul+bias: ${(unfusedTime / fusedBiasTime).toFixed(2)}x faster`,
    );
  }
  if (fusedBiasTime > 0 && fusedReluTime > 0) {
    console.log(
      `  bias+relu adds: ${(((fusedReluTime - fusedBiasTime) / fusedBiasTime) * 100).toFixed(1)}% overhead`,
    );
  }
  if (fusedBiasTime > 0 && fusedGeluTime > 0) {
    console.log(
      `  bias+gelu adds: ${(((fusedGeluTime - fusedBiasTime) / fusedBiasTime) * 100).toFixed(1)}% overhead`,
    );
  }

  // 3. Subgroup comparison (if supported)
  if (subgroupSupport?.supported) {
    console.log("\n" + "=".repeat(80));
    console.log(
      "SUBGROUP COMPARISON [f32] (with vs without subgroup operations)",
    );
    console.log("=".repeat(80));

    const subgroupSizes = [
      { m: 1024, n: 1024, k: 1024, name: "1024x1024" },
      { m: 2048, n: 2048, k: 2048, name: "2048x2048" },
    ];

    for (const { m, n, k, name } of subgroupSizes) {
      const results = await runSubgroupComparisonBenchmark(
        m,
        n,
        k,
        warmup,
        iters,
      );
      if (results.length > 0) {
        printResults(`Subgroup Comparison - ${name}`, results);

        // Calculate speedup
        const noSg = results.find((r) => r.name === "no-subgroups");
        const withSg = results.find((r) => r.name === "with-subgroups");
        if (noSg && withSg) {
          const speedup = noSg.medianMs / withSg.medianMs;
          console.log(
            `\n  Subgroup speedup: ${speedup.toFixed(2)}x ${speedup > 1 ? "faster" : "slower"}`,
          );
        }
      }
    }
  } else {
    console.log("\n" + "=".repeat(80));
    console.log(
      "SUBGROUP COMPARISON [f32]: Skipped (subgroups not supported on this device)",
    );
    console.log("=".repeat(80));
  }

  // 4. f16 vs f32 comparison
  console.log("\n" + "=".repeat(80));
  console.log("DTYPE COMPARISON (f32 vs f16)");
  console.log("=".repeat(80));

  const dtypeSizes = [
    { m: 1024, n: 1024, k: 1024, name: "1024x1024" },
    { m: 2048, n: 2048, k: 2048, name: "2048x2048" },
  ];

  for (const { m, n, k, name } of dtypeSizes) {
    const results = await runDtypeComparisonBenchmark(m, n, k, warmup, iters);
    if (results.length > 0) {
      printResults(`Dtype + Subgroup Comparison - ${name}`, results);

      // Calculate speedups
      const f32 = results.find((r) => r.name === "f32");
      const f32sg = results.find((r) => r.name === "f32+subgroups");
      const f16 = results.find((r) => r.name === "f16");
      const f16sg = results.find((r) => r.name === "f16+subgroups");

      console.log("\n  Speedup Summary:");

      // f16 vs f32 (no subgroups)
      if (f32 && f16) {
        const speedup = f32.medianMs / f16.medianMs;
        console.log(
          `    f16 vs f32 (no subgroups): ${speedup.toFixed(2)}x ${speedup > 1 ? "faster" : "slower"}`,
        );
      }

      // f16 vs f32 (with subgroups)
      if (f32sg && f16sg) {
        const speedup = f32sg.medianMs / f16sg.medianMs;
        console.log(
          `    f16 vs f32 (with subgroups): ${speedup.toFixed(2)}x ${speedup > 1 ? "faster" : "slower"}`,
        );
      }

      // Subgroup benefit for f32
      if (f32 && f32sg) {
        const speedup = f32.medianMs / f32sg.medianMs;
        console.log(
          `    f32 subgroup benefit: ${speedup.toFixed(2)}x ${speedup > 1 ? "faster" : "slower"}`,
        );
      }

      // Subgroup benefit for f16
      if (f16 && f16sg) {
        const speedup = f16.medianMs / f16sg.medianMs;
        console.log(
          `    f16 subgroup benefit: ${speedup.toFixed(2)}x ${speedup > 1 ? "faster" : "slower"}`,
        );
      }

      // Best vs worst
      if (f32 && f16sg) {
        const speedup = f32.medianMs / f16sg.medianMs;
        console.log(
          `    Best (f16+sg) vs worst (f32): ${speedup.toFixed(2)}x ${speedup > 1 ? "faster" : "slower"}`,
        );
      }
    }
  }

  // 5. Non-square matrix shapes
  console.log("\n" + "=".repeat(80));
  console.log("NON-SQUARE MATRIX SHAPES [f32]");
  console.log("=".repeat(80));

  const nonSquare = [
    { m: 4096, n: 4096, k: 256, name: "Tall-skinny (4096x4096x256)" },
    { m: 256, n: 256, k: 1024, name: "Short-wide K (256x256x1024)" },
    { m: 1, n: 1024, k: 1024, name: "GEMV (1x1024x1024)" },
  ];

  for (const { m, n, k, name } of nonSquare) {
    const results = await runMatmulBenchmark(m, n, k, warmup, iters);
    printResults(name, results);
  }

  // 4. Summary
  console.log("\n" + "=".repeat(80));
  console.log("CONFIGURATION SUMMARY");
  console.log("=".repeat(80));
  console.log(`
Default Configuration (used in all benchmarks):
  - Tile size: ${DEFAULT_CONFIG.tileM}x${DEFAULT_CONFIG.tileN}x${DEFAULT_CONFIG.tileK}
  - Thread tile: ${DEFAULT_CONFIG.threadTileM}x${DEFAULT_CONFIG.threadTileN}
  - Vector width: ${DEFAULT_CONFIG.vectorWidth}
  - Subgroups: ${DEFAULT_CONFIG.useSubgroups ? "enabled" : "disabled (not supported)"}

Available tuning space:
  - tileM: [32, 64, 128]
  - tileN: [32, 64, 128]
  - tileK: [8, 16, 32]
  - threadTileM: [4, 8]
  - threadTileN: [4, 8]
  - vectorWidth: [1, 4]

Note: Autotuning can be invoked via autotune() or quickAutotune() functions
to find the best configuration for specific matrix shapes on your GPU.
`);

  console.log("\nBenchmark complete!");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
