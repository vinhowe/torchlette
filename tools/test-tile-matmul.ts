/**
 * Test: compare tile-IR matmul WGSL with existing codegen for various configs.
 */

import { generateTiledMatmulShader, type CodegenOptions } from "../src/backend/webgpu/matmul/codegen";
import { generateTiledMatmulShaderTileIR } from "../src/backend/webgpu/matmul/tile-matmul";
import type { MatmulKernelConfig } from "../src/backend/webgpu/matmul/types";

const DEFAULT_CONFIG: MatmulKernelConfig = {
  tileM: 32, tileN: 32, tileK: 16,
  threadTileM: 4, threadTileN: 4,
  useSubgroups: false, vectorWidth: 1,
};

const configs: Array<{ name: string; options: CodegenOptions }> = [
  {
    name: "Basic NN 32x32x16",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NN",
      dtype: "f32",
    },
  },
  {
    name: "TN transpose (backward dW pattern)",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "TN",
      dtype: "f32",
    },
  },
  {
    name: "NT transpose (backward dX pattern)",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NT",
      dtype: "f32",
    },
  },
  {
    name: "Batched NN",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NN",
      dtype: "f32",
      batched: true,
    },
  },
  {
    name: "With bias epilogue",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NN",
      dtype: "f32",
      epilogue: {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      },
    },
  },
  {
    name: "With cast+bias+gelu epilogue",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NN",
      dtype: "f16",
      epilogue: {
        ops: [
          { kind: "cast", toDtype: "f32" },
          { kind: "bias", inputIndex: 0 },
          { kind: "unary", op: "gelu" },
          { kind: "cast", toDtype: "f16" },
        ],
        additionalInputCount: 1,
        outputDtype: "f16",
      },
    },
  },
  {
    name: "K-split factor 4",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NN",
      dtype: "f32",
      kSplit: 4,
    },
  },
  {
    name: "64x64x8 t8x4 (large tile)",
    options: {
      config: {
        tileM: 64, tileN: 64, tileK: 8,
        threadTileM: 8, threadTileN: 4,
        useSubgroups: false, vectorWidth: 1,
      },
      transposeMode: "NN",
      dtype: "f32",
    },
  },
  {
    name: "f16 input, f32 output",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NN",
      dtype: "f16",
      dtypeB: "f16",
    },
  },
  {
    name: "Mixed dtype (f16 A, f32 B)",
    options: {
      config: DEFAULT_CONFIG,
      transposeMode: "NN",
      dtype: "f16",
      dtypeB: "f32",
    },
  },
];

for (const { name, options } of configs) {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`CONFIG: ${name}`);
  console.log("=".repeat(70));

  try {
    const tileIR = generateTiledMatmulShaderTileIR(options);
    console.log("\n--- Tile IR WGSL ---\n");
    console.log(tileIR);
  } catch (e: any) {
    console.error(`ERROR: ${e.message}`);
  }
}

process.exit(0);
