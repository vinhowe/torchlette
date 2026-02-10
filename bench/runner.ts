import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";

import { ops, setBackend } from "../src";
import {
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
} from "../src/backend/webgpu";
import { createMatmulSuite } from "./suites/matmul";
import type { BenchCase, BenchResult } from "./types";

const warmupIters = Number.parseInt(process.env.BENCH_WARMUP ?? "3", 10);
const runIters = Number.parseInt(process.env.BENCH_ITERS ?? "10", 10);

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

async function runCase(
  benchCase: BenchCase,
  sync?: () => Promise<void>,
): Promise<BenchResult> {
  if (benchCase.skip || !benchCase.run) {
    return {
      name: benchCase.name,
      status: "skipped",
      reason: benchCase.skip ?? "missing run",
    };
  }

  for (let i = 0; i < warmupIters; i += 1) {
    await benchCase.run();
    if (sync) {
      await sync();
    }
  }

  const durations: number[] = [];
  for (let i = 0; i < runIters; i += 1) {
    const start = performance.now();
    await benchCase.run();
    if (sync) {
      await sync();
    }
    const end = performance.now();
    durations.push(end - start);
  }

  const msMedian = median(durations);
  const seconds = msMedian / 1000;
  const flopsPerSec =
    benchCase.flops && seconds > 0 ? benchCase.flops / seconds : undefined;
  const bytesPerSec =
    benchCase.bytes && seconds > 0 ? benchCase.bytes / seconds : undefined;

  return {
    name: benchCase.name,
    status: "ok",
    iterations: runIters,
    msMedian,
    flopsPerSec,
    bytesPerSec,
  };
}

async function run(): Promise<void> {
  const useWebGPU = Boolean(process.env.TORCHLETTE_WEBGPU);
  let sync: (() => Promise<void>) | undefined;
  if (useWebGPU) {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
    setBackend("webgpu");
    sync = syncWebGPU;
  }
  const cases = createMatmulSuite(ops());
  const results: BenchResult[] = [];
  for (const benchCase of cases) {
    results.push(await runCase(benchCase, sync));
  }
  const output = {
    timestamp: new Date().toISOString(),
    warmupIters,
    runIters,
    results,
  };

  const outDir = path.resolve("bench", "results");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "latest.json");
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));

  for (const result of results) {
    if (result.status === "skipped") {
      console.log(`${result.name}: skipped (${result.reason})`);
      continue;
    }
    const gflops =
      result.flopsPerSec !== undefined
        ? (result.flopsPerSec / 1e9).toFixed(2)
        : "n/a";
    const gbytes =
      result.bytesPerSec !== undefined
        ? (result.bytesPerSec / 1e9).toFixed(2)
        : "n/a";
    console.log(
      `${result.name}: ${result.msMedian.toFixed(3)} ms (GFLOPs/s=${gflops}, GB/s=${gbytes})`,
    );
  }
  if (useWebGPU) {
    process.exit(0);
  }
}

void run();
