import { defineConfig } from "vitest/config";

// GPU test files — call initWebGPU() or canUseWebGPU() and need single-fork
// execution to avoid Dawn device contention.
// Regenerate: grep -rl "initWebGPU\|canUseWebGPU" test/ --include="*.spec.ts" | grep -v browser | sort
const GPU_TEST_FILES = [
  "test/webgpu/*.spec.ts",
  "test/webgpu.spec.ts",
  "test/amp-speed-verification.spec.ts",
  "test/batch-execution-test.spec.ts",
  "test/checkpoint-early-release.spec.ts",
  "test/checkpoint-memory-profile.spec.ts",
  "test/checkpoint-scale-analysis.spec.ts",
  "test/checkpoint-segmentation.spec.ts",
  "test/distilgpt2-checkpoint-memory.spec.ts",
  "test/distilgpt2-finetune.spec.ts",
  "test/distilgpt2-full-finetuning.spec.ts",
  "test/frontend-dtype.spec.ts",
  "test/memory-aware-scheduler.spec.ts",
  "test/optim/grad-scaler.spec.ts",
  "test/oracle/gpt2-checkpoint-parity.spec.ts",
  "test/true-segmentation-benchmark.spec.ts",
];

// Heavyweight training tests excluded from default `npm run test`.
// Run via `npm run test:full` or `vitest run --project slow`.
const SLOW_TEST_FILES = ["test/gpt2-memorization.spec.ts"];

export default defineConfig({
  test: {
    coverage: {
      provider: "istanbul",
      reporter: ["text", "text-summary", "lcov"],
      include: ["src/**/*.ts"],
    },
    projects: [
      {
        test: {
          name: "cpu",
          pool: "forks",
          poolOptions: { forks: { execArgv: ["--expose-gc"] } },
          include: ["test/**/*.spec.ts"],
          exclude: [...GPU_TEST_FILES, "test/browser/**/*.spec.ts"],
        },
      },
      {
        test: {
          name: "webgpu",
          pool: "forks",
          poolOptions: {
            forks: { singleFork: true, execArgv: ["--expose-gc"] },
          },
          include: GPU_TEST_FILES,
          exclude: [...SLOW_TEST_FILES, "test/browser/**/*.spec.ts"],
          hookTimeout: 30_000,
          testTimeout: 30_000,
        },
      },
      {
        test: {
          name: "slow",
          pool: "forks",
          poolOptions: {
            forks: { singleFork: true, execArgv: ["--expose-gc"] },
          },
          include: SLOW_TEST_FILES,
          hookTimeout: 30_000,
          testTimeout: 300_000,
        },
      },
    ],
  },
});
