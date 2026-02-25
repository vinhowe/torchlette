/**
 * Generate WGSL from both codegen paths for side-by-side diff.
 */
import { generateTiledMatmulShader, type CodegenOptions } from "../src/backend/webgpu/matmul/codegen";
import { generateTiledMatmulShaderTileIR } from "../src/backend/webgpu/matmul/tile-matmul";
import type { MatmulKernelConfig } from "../src/backend/webgpu/matmul/types";
import * as fs from "fs";

const DEFAULT_CONFIG: MatmulKernelConfig = {
  tileM: 32, tileN: 32, tileK: 16,
  threadTileM: 4, threadTileN: 4,
  useSubgroups: false, vectorWidth: 1,
};

const configs: Array<{ name: string; options: CodegenOptions }> = [
  {
    name: "basic_nn",
    options: { config: DEFAULT_CONFIG, transposeMode: "NN", dtype: "f32" },
  },
  {
    name: "tn_transpose",
    options: { config: DEFAULT_CONFIG, transposeMode: "TN", dtype: "f32" },
  },
  {
    name: "bias_epilogue",
    options: {
      config: DEFAULT_CONFIG, transposeMode: "NN", dtype: "f32",
      epilogue: { ops: [{ kind: "bias", inputIndex: 0 }], additionalInputCount: 1, outputDtype: "f32" },
    },
  },
  {
    name: "ksplit4",
    options: { config: DEFAULT_CONFIG, transposeMode: "NN", dtype: "f32", kSplit: 4 },
  },
  {
    name: "f16_input",
    options: { config: DEFAULT_CONFIG, transposeMode: "NN", dtype: "f16", dtypeB: "f16" },
  },
];

for (const { name, options } of configs) {
  const existing = generateTiledMatmulShader(options);
  const tileIR = generateTiledMatmulShaderTileIR(options);

  const norm = (s: string) => s.split("\n").map(l => l.trimEnd()).filter(l => l.length > 0).join("\n");
  fs.writeFileSync(`/tmp/existing-${name}.wgsl`, norm(existing) + "\n");
  fs.writeFileSync(`/tmp/tileir-${name}.wgsl`, norm(tileIR) + "\n");
}

process.exit(0);
