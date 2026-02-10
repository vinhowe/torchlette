# torchlette

A WebGPU-accelerated tensor library for TypeScript with PyTorch-like semantics.

## Features

- **WebGPU Backend**: High-performance GPU compute via WebGPU
- **Tiled Matrix Multiplication**: Optimized matmul with shared memory tiling
- **Subgroup Support**: Hardware subgroup operations when available (with fallback)
- **Epilogue Fusion**: Fuse bias, activations (relu, gelu, silu) into matmul
- **AMP (f16)**: Automatic mixed precision with f16 compute, f32 accumulation
- **Autotuning**: Runtime tuning to find optimal tile configurations
- **ND Batched Operations**: Full broadcasting support for batched matmul

## Installation

```bash
npm install torchlette
```

## Usage

```typescript
import { tensor, matmul } from "torchlette";

const a = tensor([[1, 2], [3, 4]]);
const b = tensor([[5, 6], [7, 8]]);
const c = matmul(a, b);
```

## Development

```bash
npm install
npm run lint
npm run test
npm run build
```

## Testing

```bash
# Run all unit tests
npm run test

# Run WebGPU tests (Node.js with Dawn)
TORCHLETTE_WEBGPU=1 npm run test:webgpu

# Run browser WebGPU tests (Chromium via Playwright)
npm run test:browser
```

The browser tests use Vitest's browser mode with Playwright to run WebGPU tests in a real Chromium browser. This tests the native browser WebGPU implementation rather than the Node.js Dawn bindings.

## Benchmarking

### Node.js Benchmarks (Dawn)

Run matmul benchmarks using the Node.js WebGPU backend (Dawn):

```bash
# Build first
npm run build

# 1. Basic matmul benchmarks (standard sizes)
TORCHLETTE_WEBGPU=1 BENCH_WARMUP=2 BENCH_ITERS=5 npm run bench

# 2. Comprehensive comparison (fused vs unfused, different shapes)
TORCHLETTE_WEBGPU=1 BENCH_WARMUP=3 BENCH_ITERS=7 npx tsx bench/matmul-comparison.ts

# 3. Tile configuration comparison (finds best config per size)
TORCHLETTE_WEBGPU=1 BENCH_WARMUP=2 BENCH_ITERS=5 npx tsx bench/matmul-tile-configs.ts
```

### Browser Benchmarks (Native WebGPU)

Run benchmarks in a real browser using native WebGPU:

```bash
# Start the benchmark server
npm run bench:browser

# Then open http://localhost:8080/bench/browser/ in Chrome
```

The browser benchmark page provides:
- **Quick Benchmark**: 256x256, 512x512, 1024x1024 matrices
- **Full Benchmark**: All sizes including non-square and GEMV
- **Matmul Only**: Large matrices (1024, 2048, 4096) with more iterations

### What the benchmarks measure

- **Basic benchmarks**: Standard matmul performance at 256x256, 512x512, 1024x1024, 2048x2048
- **Comparison benchmarks**: Fused vs unfused epilogue operations, non-square shapes, GEMV
- **Tile config benchmarks**: Tests 13 different tile configurations to show autotuning impact

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCHLETTE_WEBGPU` | `0` | Set to `1` to enable WebGPU backend |
| `BENCH_WARMUP` | `3` | Number of warmup iterations before timing |
| `BENCH_ITERS` | `5` | Number of timed iterations (median reported) |

### Example output

```
================================================================================
MATMUL PERFORMANCE BY SIZE (Default Config: 32x32x16 tiles)
================================================================================

Small (256x256) (M=256, N=256, K=256)
Name                               Config                     Time(ms)    GFLOPs/s
--------------------------------------------------------------------------------
default                            32x32x16_t4x4                 0.730       45.97

Large (1024x1024) (M=1024, N=1024, K=1024)
Name                               Config                     Time(ms)    GFLOPs/s
--------------------------------------------------------------------------------
default                            32x32x16_t4x4                 3.921      547.64
```

## Project Structure

- `src/backend/webgpu/` - WebGPU backend implementation
- `src/backend/webgpu/matmul/` - Tiled matmul with autotuning
  - `codegen.ts` - WGSL shader generation
  - `dispatch.ts` - Kernel dispatch logic
  - `epilogue.ts` - Epilogue fusion detection
  - `subgroup.ts` - Subgroup-accelerated variant
  - `autotune.ts` - Autotuning infrastructure
- `bench/` - Benchmark scripts
- `test/` - Test suites

## License

MIT
