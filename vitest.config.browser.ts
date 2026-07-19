import * as path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      // Browser-side example code (examples/*) imports torchlette by package
      // name; resolve it to the BROWSER source entry so those imports load the
      // same source the tests do (the package `browser` field points at built
      // dist, which we don't want to test here). Keeps example + tests on one
      // build of the source.
      //
      // `torchlette/nn` is a published SUBPATH export (→ dist/nn.js). Shared
      // browser code (packages/gpt2-browser/src/gpt2-lora.ts) imports it; alias
      // it to source too so the lora-training-trajectory spec loads gpt2-lora
      // from source without pulling in built dist. More-specific key first so
      // it wins over the bare `torchlette` alias.
      "torchlette/nn": path.resolve(__dirname, "src/nn/index.ts"),
      torchlette: path.resolve(__dirname, "src/browser.ts"),
    },
  },
  test: {
    include: ["test/browser/**/*.spec.ts"],
    browser: {
      enabled: true,
      provider: "playwright",
      instances: [
        {
          browser: "chromium",
          launch: {
            // WebGPU requires these flags in headless mode
            args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
          },
        },
      ],
    },
    // Longer timeout for GPU operations
    testTimeout: 30000,
  },
});
